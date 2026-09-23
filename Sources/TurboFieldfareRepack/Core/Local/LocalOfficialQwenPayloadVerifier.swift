import Darwin
import Foundation

struct LocalOfficialQwenPayloadIdentity: Sendable, Equatable {
    let checksumManifestSHA256: String
    let shardSetSHA256: String
    let shardBytes: UInt64
    let shardCount: Int
}

struct LocalOfficialQwenPayloadVerifierOperations {
    var readChecksumManifest: (_ path: String, _ maximumBytes: UInt64) throws -> Data
    var inspectAndHashFile: (
        _ path: String,
        _ progress: @escaping (UInt64) throws -> Void
    ) throws -> (size: UInt64, sha256: String)
    var cancellationCheck: () throws -> Void

    static var production: Self {
        Self(
            readChecksumManifest: { path, maximumBytes in
                try Posix.readBoundedData(path, maximumBytes: maximumBytes)
            },
            inspectAndHashFile: { path, progress in
                let descriptor = try Posix.openReadNoFollow(path)
                defer { close(descriptor) }
                let size = try Posix.fileSize(fd: descriptor, path: path)
                let digest = try Sha256Stream.hashFileDescriptor(
                    descriptor,
                    displayPath: path,
                    tileBytes: WriterCore.tileBytes,
                    noCache: true,
                    onProgress: progress)
                guard try Posix.descriptorMatchesPath(descriptor, path: path) else {
                    throw RepackError.installStateIncompatible(
                        detail: "official Qwen shard changed while it was authenticated")
                }
                return (size, digest)
            },
            cancellationCheck: { try Task.checkCancellation() })
    }
}

enum LocalOfficialQwenPayloadVerifier {
    static let checksumManifestFile = "SHA256SUMS"
    static let checksumManifestBytes: UInt64 = 3_571
    static let checksumManifestSHA256 =
        "1378d1bb153694b13c20641fa5d2485dede401d1ce4c267953ba4a855fc59e7e"
    static let expectedShardBytes: UInt64 = 71_903_776_776
    static let expectedShardCount = 26
    static let expectedLineCount = 38

    private static let expectedSidecars: Set<String> = [
        "LICENSE", "README.md", "chat_template.jinja", "config.json",
        "configuration.json", "generation_config.json", "merges.txt",
        "model.safetensors.index.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    ]

    static func verify(
        snapshotDirectory: String,
        shardFilenames: [String],
        operations: LocalOfficialQwenPayloadVerifierOperations = .production,
        progress: @escaping (_ completedBytes: UInt64, _ totalBytes: UInt64) -> Void = { _, _ in }
    ) throws -> LocalOfficialQwenPayloadIdentity {
        try operations.cancellationCheck()
        let checksumPath = (snapshotDirectory as NSString)
            .appendingPathComponent(checksumManifestFile)
        let data = try operations.readChecksumManifest(
            checksumPath, checksumManifestBytes)
        guard UInt64(data.count) == checksumManifestBytes else {
            throw RepackError.sourceFingerprintRejected(
                path: checksumPath, sha256: digest(data))
        }
        let observedManifestDigest = digest(data)
        guard observedManifestDigest == checksumManifestSHA256 else {
            throw RepackError.sourceFingerprintRejected(
                path: checksumPath, sha256: observedManifestDigest)
        }
        let entries = try parse(data, path: checksumPath)
        let shardSet = Set(shardFilenames)
        guard shardFilenames.count == expectedShardCount,
              shardSet.count == expectedShardCount,
              entries.count == expectedLineCount,
              Set(entries.keys).subtracting(shardSet) == expectedSidecars,
              shardSet.allSatisfy(isExpectedShardName) else {
            throw RepackError.installStateCorrupt(
                path: checksumPath,
                detail: "official Qwen checksum inventory is not the pinned 26-shard/12-sidecar set")
        }

        var totalBytes: UInt64 = 0
        var aggregate = Sha256Stream()
        Data((checksumManifestSHA256 + "\n").utf8).withUnsafeBytes {
            aggregate.update($0)
        }
        for name in shardFilenames.sorted() {
            try operations.cancellationCheck()
            guard let expectedDigest = entries[name] else {
                throw RepackError.installStateCorrupt(
                    path: checksumPath, detail: "missing checksum for \(name)")
            }
            let path = (snapshotDirectory as NSString).appendingPathComponent(name)
            let base = totalBytes
            let inspected = try operations.inspectAndHashFile(path) { completed in
                let sum = base.addingReportingOverflow(completed)
                guard !sum.overflow else {
                    throw RepackError.configurationInvalid(
                        detail: "official Qwen authentication progress overflows UInt64")
                }
                progress(sum.partialValue, expectedShardBytes)
                try operations.cancellationCheck()
            }
            guard inspected.sha256.lowercased() == expectedDigest else {
                throw RepackError.sourceFingerprintRejected(
                    path: path, sha256: inspected.sha256.lowercased())
            }
            let sum = totalBytes.addingReportingOverflow(inspected.size)
            guard !sum.overflow else {
                throw RepackError.configurationInvalid(
                    detail: "official Qwen shard bytes overflow UInt64")
            }
            totalBytes = sum.partialValue
            Data(("\(name)\t\(inspected.size)\t\(expectedDigest)\n").utf8)
                .withUnsafeBytes { aggregate.update($0) }
            progress(totalBytes, expectedShardBytes)
        }
        guard totalBytes == expectedShardBytes else {
            throw RepackError.installStateCorrupt(
                path: snapshotDirectory,
                detail: "official Qwen shards total \(totalBytes), expected \(expectedShardBytes)")
        }
        return LocalOfficialQwenPayloadIdentity(
            checksumManifestSHA256: observedManifestDigest,
            shardSetSHA256: aggregate.finalizeHexString(),
            shardBytes: totalBytes,
            shardCount: shardFilenames.count)
    }

    private static func parse(_ data: Data, path: String) throws -> [String: String] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw RepackError.installStateCorrupt(
                path: path, detail: "checksum manifest is not UTF-8")
        }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        guard lines.count == expectedLineCount else {
            throw RepackError.installStateCorrupt(
                path: path, detail: "checksum manifest must contain 38 lines")
        }
        var result: [String: String] = [:]
        for line in lines {
            guard line.count > 66 else {
                throw RepackError.installStateCorrupt(
                    path: path, detail: "malformed checksum line")
            }
            let digestEnd = line.index(line.startIndex, offsetBy: 64)
            let separatorEnd = line.index(digestEnd, offsetBy: 2)
            let hash = String(line[..<digestEnd])
            let separator = line[digestEnd..<separatorEnd]
            let name = String(line[separatorEnd...])
            guard separator == "  ", isSHA256(hash), isSafeLeaf(name),
                  result.updateValue(hash, forKey: name) == nil else {
                throw RepackError.installStateCorrupt(
                    path: path, detail: "invalid or duplicate checksum entry")
            }
        }
        return result
    }

    private static func isExpectedShardName(_ name: String) -> Bool {
        guard name.hasPrefix("model-"), name.hasSuffix("-of-00026.safetensors"),
              name.count == "model-00001-of-00026.safetensors".count else {
            return false
        }
        let start = name.index(name.startIndex, offsetBy: "model-".count)
        let end = name.index(start, offsetBy: 5)
        guard let ordinal = Int(name[start..<end]) else { return false }
        return (1...expectedShardCount).contains(ordinal)
    }

    private static func isSafeLeaf(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".."
            && !name.contains("/") && !name.contains("\\") && !name.contains("\0")
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy {
            $0.isNumber || ("a"..."f").contains($0)
        }
    }

    private static func digest(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }
}
