import CryptoKit
import Foundation

/// Identity constraints for a local snapshot. Test fixtures supply their own
/// digest table; the catalog supplies the immutable production Qwen table.
struct LocalPinnedSnapshotIdentity: Sendable, Equatable {
    let repository: String
    let revision: String
    let sidecarSHA256: [String: String]
    let enforcesOfficialTensorMap: Bool

    init(repository: String,
         revision: String,
         sidecarSHA256: [String: String],
         enforcesOfficialTensorMap: Bool) {
        self.repository = repository
        self.revision = revision
        self.sidecarSHA256 = sidecarSHA256
        self.enforcesOfficialTensorMap = enforcesOfficialTensorMap
    }

    init(source: ModelSourceCatalog.LocalSnapshotSource) {
        self.init(repository: source.repository,
                  revision: source.revision,
                  sidecarSHA256: source.sidecarSHA256,
                  enforcesOfficialTensorMap: source.enforcesOfficialTensorMap)
    }
}

/// Accepted metadata only. Header/index agreement does not authenticate shard
/// payload bytes after their bounded safetensors headers.
struct LocalPinnedSnapshot: Sendable {
    let indexSHA256: String
    let shardFilenames: [String]
    let tensors: [SourceTensor]
}

enum LocalPinnedSnapshotLoader {
    private static let indexFilename = "model.safetensors.index.json"
    private static let indexMaximumBytes: UInt64 = 4 * 1024 * 1024
    private static let sidecarMaximumBytes: UInt64 = 32 * 1024 * 1024

    static func load(snapshotDirectory: String,
                     expectedIdentity: LocalPinnedSnapshotIdentity) throws -> LocalPinnedSnapshot {
        let sidecars = try loadSidecars(snapshotDirectory: snapshotDirectory,
                                        expectedIdentity: expectedIdentity)
        guard let indexData = sidecars[indexFilename] else {
            throw RepackError.installStateCorrupt(path: snapshotDirectory,
                                                   detail: "missing index sidecar")
        }
        let indexPath = try leafPath(snapshotDirectory, indexFilename)
        let indexSHA256 = sha256(indexData)
        guard indexSHA256 == expectedIdentity.sidecarSHA256[indexFilename] else {
            throw RepackError.sourceFingerprintRejected(path: indexPath, sha256: indexSHA256)
        }
        let weightMap = try parseWeightMap(indexData, path: indexPath)
        let shardFilenames = try shardNames(weightMap, path: indexPath)
        try rejectUnreferencedShards(snapshotDirectory: snapshotDirectory,
                                     referenced: Set(shardFilenames))

        var tensors: [SourceTensor] = []
        tensors.reserveCapacity(weightMap.count)
        for shard in shardFilenames {
            let shardPath = try leafPath(snapshotDirectory, shard)
            let header = try Safetensors.parseLocalHeader(path: shardPath)
            try validateNonOverlapping(header: header)
            let expectedNames = Set(weightMap.compactMap { $0.value == shard ? $0.key : nil })
            let actualNames = Set(header.tensors.map(\.name))
            guard actualNames == expectedNames else {
                throw RepackError.indexJsonInvalid(path: indexPath,
                                                   detail: "tensor names disagree for \(shard)")
            }
            for tensor in header.tensors {
                guard tensor.dtype == .bf16 else {
                    throw RepackError.dtypeMismatch(name: tensor.name,
                                                    detail: "local Qwen snapshots require BF16")
                }
            }
            tensors.append(contentsOf: header.tensors)
        }
        guard tensors.count == weightMap.count else {
            throw RepackError.indexJsonInvalid(path: indexPath,
                                               detail: "index/header tensor count differs")
        }
        if expectedIdentity.enforcesOfficialTensorMap {
            guard expectedIdentity.repository == ModelSourceCatalog.qwen.repository,
                  expectedIdentity.revision == ModelSourceCatalog.qwen.revision,
                  shardFilenames.count == 26 else {
                throw RepackError.configurationInvalid(detail: "official Qwen identity is invalid")
            }
            let categories = try QwenOfficialTensorMap.classify(
                tensors.map { QwenOfficialTensorDescriptor(name: $0.name, dataType: $0.dtype) })
            guard categories.filter({ $0 == .mtpOmitted }).count == 19 else {
                throw RepackError.indexJsonInvalid(path: indexPath, detail: "expected 19 MTP tensors")
            }
        }
        return LocalPinnedSnapshot(indexSHA256: indexSHA256,
                                   shardFilenames: shardFilenames,
                                   tensors: tensors)
    }

    private static func loadSidecars(
        snapshotDirectory: String,
        expectedIdentity: LocalPinnedSnapshotIdentity
    ) throws -> [String: Data] {
        var result: [String: Data] = [:]
        for filename in expectedIdentity.sidecarSHA256.keys.sorted() {
            let path = try leafPath(snapshotDirectory, filename)
            let cap = filename == indexFilename ? indexMaximumBytes : sidecarMaximumBytes
            let data = try Posix.readBoundedData(path, maximumBytes: cap)
            let digest = sha256(data)
            guard digest == expectedIdentity.sidecarSHA256[filename] else {
                throw RepackError.sourceFingerprintRejected(path: path, sha256: digest)
            }
            result[filename] = data
        }
        return result
    }

    private static func parseWeightMap(_ data: Data, path: String) throws -> [String: String] {
        do {
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let map = root["weight_map"] as? [String: String], !map.isEmpty else {
                throw RepackError.indexJsonInvalid(path: path, detail: "no weight_map")
            }
            return map
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.indexJsonInvalid(path: path, detail: "\(error)")
        }
    }

    private static func shardNames(_ weightMap: [String: String], path: String) throws -> [String] {
        var names = Set<String>()
        for shard in weightMap.values {
            guard isLeafName(shard), shard.hasSuffix(".safetensors") else {
                throw RepackError.indexJsonInvalid(path: path, detail: "unsafe shard name \(shard)")
            }
            names.insert(shard)
        }
        return names.sorted()
    }

    private static func rejectUnreferencedShards(snapshotDirectory: String,
                                                  referenced: Set<String>) throws {
        let names = try FileManager.default.contentsOfDirectory(atPath: snapshotDirectory)
        for name in names where name.hasSuffix(".safetensors") && !referenced.contains(name) {
            throw RepackError.installStateCorrupt(path: snapshotDirectory,
                                                   detail: "unreferenced shard \(name)")
        }
    }

    private static func validateNonOverlapping(header: Safetensors.Header) throws {
        let sorted = header.tensors.sorted { $0.absoluteOffset < $1.absoluteOffset }
        for (previous, next) in zip(sorted, sorted.dropFirst()) {
            let (end, overflow) = previous.absoluteOffset.addingReportingOverflow(previous.sizeBytes)
            guard !overflow, end <= next.absoluteOffset else {
                throw RepackError.safetensorsHeaderInvalid(
                    path: header.path,
                    detail: "tensor ranges overlap: \(previous.name), \(next.name)")
            }
        }
    }

    private static func leafPath(_ directory: String, _ name: String) throws -> String {
        guard isLeafName(name) else {
            throw RepackError.installStateCorrupt(path: directory, detail: "unsafe leaf name \(name)")
        }
        return (directory as NSString).appendingPathComponent(name)
    }

    private static func isLeafName(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".." && !name.contains("/") && !name.contains("\\")
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
