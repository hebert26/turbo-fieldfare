import CryptoKit
import Darwin
import Foundation
import TurboFieldfareFormat

/// A local record of one successful full verification. It is not a signature,
/// a filesystem seal, a runtime-admission ticket, or proof of publisher origin.
/// Anyone able to replace both local files and this receipt can forge the local
/// reopen evidence. Trusted reopen compares stat fingerprints; only fullSha256
/// rehashes all payload bytes against the immutable official pins.
public struct OfficialSourceTrustReceipt: Codable, Equatable, Sendable {
    public static let schemaVersion = 1

    public struct Fingerprint: Codable, Equatable, Sendable {
        public let device: UInt64
        public let inode: UInt64
        public let mode: UInt32
        public let size: UInt64
        public let modifiedSeconds: UInt64
        public let modifiedNanoseconds: UInt32
        public let changedSeconds: UInt64
        public let changedNanoseconds: UInt32

        fileprivate init(_ info: stat) throws {
            guard info.st_size >= 0, info.st_mtimespec.tv_sec >= 0,
                  info.st_ctimespec.tv_sec >= 0,
                  (0..<1_000_000_000).contains(info.st_mtimespec.tv_nsec),
                  (0..<1_000_000_000).contains(info.st_ctimespec.tv_nsec) else {
                throw OfficialSourceTrust.TrustError.invalid("unrepresentable file fingerprint")
            }
            device = UInt64(info.st_dev)
            inode = UInt64(info.st_ino)
            mode = UInt32(info.st_mode)
            size = UInt64(info.st_size)
            modifiedSeconds = UInt64(info.st_mtimespec.tv_sec)
            modifiedNanoseconds = UInt32(info.st_mtimespec.tv_nsec)
            changedSeconds = UInt64(info.st_ctimespec.tv_sec)
            changedNanoseconds = UInt32(info.st_ctimespec.tv_nsec)
        }

        private enum CodingKeys: String, CodingKey {
            case device, inode, mode, size, modifiedSeconds, modifiedNanoseconds
            case changedSeconds, changedNanoseconds
        }

        public init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: ReceiptKey.self)
            try requireKeys(values.allKeys.map(\.stringValue), expected: [
                "device", "inode", "mode", "size", "modifiedSeconds",
                "modifiedNanoseconds", "changedSeconds", "changedNanoseconds",
            ])
            let keyed = try decoder.container(keyedBy: CodingKeys.self)
            device = try keyed.decode(UInt64.self, forKey: .device)
            inode = try keyed.decode(UInt64.self, forKey: .inode)
            mode = try keyed.decode(UInt32.self, forKey: .mode)
            size = try keyed.decode(UInt64.self, forKey: .size)
            modifiedSeconds = try keyed.decode(UInt64.self, forKey: .modifiedSeconds)
            modifiedNanoseconds = try keyed.decode(UInt32.self, forKey: .modifiedNanoseconds)
            changedSeconds = try keyed.decode(UInt64.self, forKey: .changedSeconds)
            changedNanoseconds = try keyed.decode(UInt32.self, forKey: .changedNanoseconds)
            guard modifiedNanoseconds < 1_000_000_000,
                  changedNanoseconds < 1_000_000_000 else {
                throw OfficialSourceTrust.TrustError.invalid("invalid nanosecond component")
            }
        }
    }

    public struct FileEntry: Codable, Equatable, Sendable {
        public let filename: String
        public let sha256: String
        public let fingerprint: Fingerprint

        private enum CodingKeys: String, CodingKey { case filename, sha256, fingerprint }

        fileprivate init(filename: String, sha256: String, fingerprint: Fingerprint) {
            self.filename = filename
            self.sha256 = sha256
            self.fingerprint = fingerprint
        }

        public init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: ReceiptKey.self)
            try requireKeys(values.allKeys.map(\.stringValue),
                            expected: ["filename", "sha256", "fingerprint"])
            let keyed = try decoder.container(keyedBy: CodingKeys.self)
            filename = try keyed.decode(String.self, forKey: .filename)
            sha256 = try keyed.decode(String.self, forKey: .sha256)
            fingerprint = try keyed.decode(Fingerprint.self, forKey: .fingerprint)
        }
    }

    public let version: Int
    public let descriptorContentSHA256: String
    public let markerSHA256: String
    public let logicalModelPath: String
    public let sourceRoot: String
    public let checksumManifestSHA256: String
    public let shardSetSHA256: String
    public let shardBytes: UInt64
    public let rootFingerprint: Fingerprint
    public let files: [FileEntry]

    private enum CodingKeys: String, CodingKey {
        case version, descriptorContentSHA256, markerSHA256, logicalModelPath
        case sourceRoot, checksumManifestSHA256, shardSetSHA256, shardBytes
        case rootFingerprint, files
    }

    fileprivate init(descriptor: OfficialSourceDescriptor, markerSHA256: String,
                     logicalModelPath: String, identity: OfficialQwenPayloadIdentity,
                     snapshot: SourceSnapshot, digests: [String: String]) throws {
        version = Self.schemaVersion
        descriptorContentSHA256 = descriptor.contentSHA256
        self.markerSHA256 = markerSHA256
        self.logicalModelPath = logicalModelPath
        sourceRoot = descriptor.sourceRoot
        checksumManifestSHA256 = identity.checksumManifestSHA256
        shardSetSHA256 = identity.shardSetSHA256
        shardBytes = identity.shardBytes
        rootFingerprint = snapshot.root
        files = try snapshot.files.keys.sorted().map { name in
            guard let fingerprint = snapshot.files[name], let digest = digests[name] else {
                throw OfficialSourceTrust.TrustError.invalid("missing verified file evidence")
            }
            return FileEntry(filename: name, sha256: digest, fingerprint: fingerprint)
        }
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: ReceiptKey.self)
        try requireKeys(values.allKeys.map(\.stringValue), expected: [
            "version", "descriptorContentSHA256", "markerSHA256",
            "logicalModelPath", "sourceRoot", "checksumManifestSHA256",
            "shardSetSHA256", "shardBytes", "rootFingerprint", "files",
        ])
        let keyed = try decoder.container(keyedBy: CodingKeys.self)
        version = try keyed.decode(Int.self, forKey: .version)
        descriptorContentSHA256 = try keyed.decode(String.self, forKey: .descriptorContentSHA256)
        markerSHA256 = try keyed.decode(String.self, forKey: .markerSHA256)
        logicalModelPath = try keyed.decode(String.self, forKey: .logicalModelPath)
        sourceRoot = try keyed.decode(String.self, forKey: .sourceRoot)
        checksumManifestSHA256 = try keyed.decode(String.self, forKey: .checksumManifestSHA256)
        shardSetSHA256 = try keyed.decode(String.self, forKey: .shardSetSHA256)
        shardBytes = try keyed.decode(UInt64.self, forKey: .shardBytes)
        rootFingerprint = try keyed.decode(Fingerprint.self, forKey: .rootFingerprint)
        files = try keyed.decode([FileEntry].self, forKey: .files)
    }
}

private struct ReceiptKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

private func requireKeys(_ actual: [String], expected: Set<String>) throws {
    guard Set(actual) == expected else {
        throw OfficialSourceTrust.TrustError.invalid("missing or unknown receipt field")
    }
}

public enum OfficialSourceTrustPolicy: Sendable, Equatable {
    case fullSha256
    case sizeCheckTrustedReceipt
}

/// Package-only observations made next to actual registration/source syscalls.
/// Stat events are not payload reads. This seam covers the cheap/trusted paths;
/// public fullSha256 streams shards through the separate pinned verifier.
package enum OfficialSourceTrustIOEvent: Sendable, Equatable {
    case statFile(path: String)
    case openFile(path: String)
    case readFile(path: String, offset: UInt64, requestedBytes: Int)
}

public struct OfficialSourceMetadataProbe: Sendable, Equatable {
    public enum ReceiptPresence: Sendable, Equatable {
        case missing
        case presentUntrusted
        case invalid
    }
    public let descriptor: OfficialSourceDescriptor
    public let receipt: ReceiptPresence
}

/// Source-only integrity policy. This API never admits a source to the runtime.
/// Full verification streams the immutable 26 pinned whole-shard hashes through
/// the production verifier; trusted reopen and probes do not open payload files.
public enum OfficialSourceTrust {
    public static let receiptFilename = "official-source-receipt.json"
    public static let maximumReceiptBytes: UInt64 = 128 * 1024

    package enum PublicationCheckpoint: Sendable {
        case stageSynced
        case beforePublish
    }

    /// Test-only simulation. It never calls the pinned full verifier and its
    /// small-byte receipt cannot pass the public trusted-reopen byte-count pin.
    /// Test real bounded hashing separately via the accepted hashRegularFile
    /// production seam; callback success is not authentication.
    package static func verifySynthetic(
        at logicalModelURL: URL, policy: OfficialSourceTrustPolicy,
        expectedShardBytes: UInt64,
        fullVerification: @escaping () throws -> Void,
        checkpoint: (PublicationCheckpoint) throws -> Void = { _ in },
        syncDirectory: (Int32) -> Int32 = { fsync($0) },
        observeIO: ((OfficialSourceTrustIOEvent) -> Void)? = nil,
        trustedCheckpoint: () throws -> Void = {},
        stagedReceiptCheckpoint: ((URL) throws -> Void)? = nil
    ) throws -> OfficialSourceTrustReceipt {
        guard expectedShardBytes < OfficialQwenPayloadVerifier.expectedShardBytes else {
            throw TrustError.invalid("synthetic byte count must not equal official pin")
        }
        return try verifyInternal(at: logicalModelURL, policy: policy,
                                  progress: { _, _ in },
                                  simulation: (bytes: expectedShardBytes,
                                               fullVerification: fullVerification),
                                  checkpoint: checkpoint, syncDirectory: syncDirectory,
                                  observeIO: observeIO,
                                  trustedCheckpoint: trustedCheckpoint,
                                  stagedReceiptCheckpoint: stagedReceiptCheckpoint)
    }

    public enum TrustError: Error, Sendable, Equatable {
        case missing
        case invalid(String)
        case stale(String)
        case io(path: String, errno: Int32)
        /// A receipt was committed, but the source or registration failed the
        /// post-commit recheck. Do not treat this as an unpublished failure.
        case publishedStale(String)
        /// A complete receipt was committed, but the parent fsync failed.
        /// Inspect/reverify instead of assuming an unpublished failure.
        case publishedDurabilityUnknown(path: String, errno: Int32)
    }

    /// A probe reports only whether bounded receipt metadata is present and
    /// syntactically valid. It never returns a trusted identity or opens shards.
    public static func probe(at logicalModelURL: URL) throws -> OfficialSourceMetadataProbe {
        try probeInternal(at: logicalModelURL, observeIO: nil)
    }

    /// Exercises the same cheap path with events at the actual open/pread/stat
    /// sites. Observation cannot supply bytes, file descriptors, or trust.
    package static func probeObserved(
        at logicalModelURL: URL,
        observeIO: @escaping (OfficialSourceTrustIOEvent) -> Void
    ) throws -> OfficialSourceMetadataProbe {
        try probeInternal(at: logicalModelURL, observeIO: observeIO)
    }

    private static func probeInternal(
        at logicalModelURL: URL,
        observeIO: ((OfficialSourceTrustIOEvent) -> Void)?
    ) throws -> OfficialSourceMetadataProbe {
        let descriptor = try OfficialSourceRegistration.inspect(
            at: logicalModelURL, observeIO: observeIO)
        let directory = try SourceDirectory(path: logicalModelURL.path,
                                            observeIO: observeIO)
        defer { close(directory.fd) }
        let presence: OfficialSourceMetadataProbe.ReceiptPresence
        do {
            let bytes = try directory.read(leaf: receiptFilename, cap: maximumReceiptBytes)
            let receipt = try decodeStrict(bytes)
            let marker = try directory.read(
                leaf: OfficialSourceDescriptor.markerFilename,
                cap: OfficialSourceDescriptor.maximumMarkerBytes)
            try validateBinding(receipt, descriptor: descriptor,
                                logicalPath: logicalModelURL.path,
                                markerSHA: hash(marker))
            presence = .presentUntrusted
        } catch TrustError.missing {
            presence = .missing
        } catch TrustError.invalid(_), TrustError.stale(_) {
            presence = .invalid
        }
        // Cancellation and filesystem I/O failures are not evidence of an
        // invalid receipt; propagate them instead of downgrading the probe.
        return OfficialSourceMetadataProbe(descriptor: descriptor, receipt: presence)
    }

    /// Public production entry point: no caller-supplied pins, hashes or file
    /// fingerprints. In particular, a metadata probe cannot mint a receipt.
    public static func verify(
        at logicalModelURL: URL, policy: OfficialSourceTrustPolicy,
        progress: @escaping (UInt64, UInt64) -> Void = { _, _ in }
    ) throws -> OfficialSourceTrustReceipt {
        try verifyInternal(at: logicalModelURL, policy: policy, progress: progress,
                           simulation: nil, checkpoint: { _ in },
                           syncDirectory: { fsync($0) }, observeIO: nil,
                           trustedCheckpoint: {}, stagedReceiptCheckpoint: nil)
    }

    private static func verifyInternal(
        at logicalModelURL: URL, policy: OfficialSourceTrustPolicy,
        progress: @escaping (UInt64, UInt64) -> Void,
        simulation: (bytes: UInt64, fullVerification: () throws -> Void)?,
        checkpoint: (PublicationCheckpoint) throws -> Void,
        syncDirectory: (Int32) -> Int32,
        observeIO: ((OfficialSourceTrustIOEvent) -> Void)?,
        trustedCheckpoint: () throws -> Void,
        stagedReceiptCheckpoint: ((URL) throws -> Void)?
    ) throws -> OfficialSourceTrustReceipt {
        try Task.checkCancellation()
        let descriptor = try OfficialSourceRegistration.inspect(
            at: logicalModelURL, observeIO: observeIO)
        let logicalPath = logicalModelURL.path
        guard logicalPath != descriptor.sourceRoot,
              !logicalPath.hasPrefix(descriptor.sourceRoot + "/"),
              !descriptor.sourceRoot.hasPrefix(logicalPath + "/") else {
            throw TrustError.invalid("source and registration directories overlap")
        }
        let registered = try SourceDirectory(path: logicalPath,
                                             observeIO: observeIO)
        defer { close(registered.fd) }
        let marker = try registered.read(
            leaf: OfficialSourceDescriptor.markerFilename,
            cap: OfficialSourceDescriptor.maximumMarkerBytes)
        let markerSHA = hash(marker)
        guard try OfficialSourceDescriptor.decodeStrict(data: marker) == descriptor else {
            throw TrustError.stale("marker changed during inspection")
        }
        let source = try SourceDirectory(path: descriptor.sourceRoot,
                                         observeIO: observeIO)
        defer { close(source.fd) }
        switch policy {
        case .sizeCheckTrustedReceipt:
            let observed = try registered.readWithFingerprint(
                leaf: receiptFilename, cap: maximumReceiptBytes)
            let receipt = try decodeStrict(observed.bytes)
            try validateBinding(receipt, descriptor: descriptor,
                                logicalPath: logicalModelURL.path,
                                markerSHA: markerSHA,
                                expectedBytes: simulation?.bytes ?? OfficialQwenPayloadVerifier.expectedShardBytes)
            let now = try source.snapshot(names: expectedFiles)
            try requireMatching(receipt, snapshot: now)
            // A package-only race hook. The final bounded marker/receipt reads
            // and named-directory check below still run after any mutation.
            try trustedCheckpoint()
            try Task.checkCancellation()
            try checkMarker(registered, expectedSHA: markerSHA)
            // A retained directory FD alone does not prove that its logical
            // path still names it. Nor may a removed/replaced receipt retain
            // trusted status after the initial bounded read.
            let current = try registered.readWithFingerprint(
                leaf: receiptFilename, cap: maximumReceiptBytes)
            guard current.fingerprint == observed.fingerprint,
                  current.bytes == observed.bytes else {
                throw TrustError.stale("trusted receipt changed during reopen")
            }
            try requireMatching(receipt, snapshot: source.snapshot(names: expectedFiles))
            try SourceDirectory.checkNamedRegistration(registered)
            try Task.checkCancellation()
            return receipt
        case .fullSha256:
            let before = try source.snapshot(names: expectedFiles)
            try Task.checkCancellation()
            let identity: OfficialQwenPayloadIdentity
            let digests: [String: String]
            if let simulation {
                try simulation.fullVerification()
                identity = try simulatedIdentity(snapshot: before, bytes: simulation.bytes,
                                                 descriptor: descriptor)
                digests = simulatedDigests(descriptor: descriptor)
            } else {
                identity = try OfficialQwenPayloadVerifier.verify(
                    snapshotDirectory: descriptor.sourceRoot,
                    shardFilenames: descriptor.shards.map(\.filename), progress: progress)
                guard identity.shardCount == OfficialQwenPayloadVerifier.expectedShardCount,
                      identity.shardBytes == OfficialQwenPayloadVerifier.expectedShardBytes,
                      identity.checksumManifestSHA256 == OfficialQwenPayloadVerifier.checksumManifestSHA256 else {
                    throw TrustError.invalid("pinned full verifier returned inconsistent inventory")
                }
                digests = try verifySidecarsAndHeaders(
                    source: source, descriptor: descriptor)
            }
            let after = try source.snapshot(names: expectedFiles)
            guard before == after else { throw TrustError.stale("source changed during full verification") }
            try checkMarker(registered, expectedSHA: markerSHA)
            let receipt = try OfficialSourceTrustReceipt(
                descriptor: descriptor, markerSHA256: markerSHA,
                logicalModelPath: logicalModelURL.path,
                identity: identity, snapshot: after, digests: digests)
            // Reject a verifier/snapshot disagreement before any staging:
            // the receipt aggregate must encode these exact observed sizes.
            try validateBinding(receipt, descriptor: descriptor,
                                logicalPath: logicalPath, markerSHA: markerSHA,
                                expectedBytes: simulation?.bytes ?? OfficialQwenPayloadVerifier.expectedShardBytes)
            try publish(receipt, registered: registered, source: source,
                        initial: after, markerSHA: markerSHA,
                        checkpoint: checkpoint, syncDirectory: syncDirectory,
                        stagedReceiptCheckpoint: stagedReceiptCheckpoint)
            return receipt
        }
    }

    private static let sidecars: Set<String> = [
        "LICENSE", "README.md", "chat_template.jinja", "config.json",
        "configuration.json", "generation_config.json", "merges.txt",
        "model.safetensors.index.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    ]
    private static var expectedFiles: Set<String> {
        sidecars.union([OfficialQwenPayloadVerifier.checksumManifestFile])
            .union(OfficialQwenSourceIdentity.pinned.shards.map(\.filename))
    }

    private static func simulatedDigests(
        descriptor: OfficialSourceDescriptor
    ) -> [String: String] {
        var result = Dictionary(uniqueKeysWithValues:
            descriptor.shards.map { ($0.filename, $0.sha256) })
        for name in sidecars {
            result[name] = descriptor.sidecarSHA256[name] ?? String(repeating: "0", count: 64)
        }
        result[OfficialQwenPayloadVerifier.checksumManifestFile] =
            OfficialQwenPayloadVerifier.checksumManifestSHA256
        return result
    }

    private static func simulatedIdentity(
        snapshot: SourceSnapshot, bytes: UInt64,
        descriptor: OfficialSourceDescriptor
    ) throws -> OfficialQwenPayloadIdentity {
        var total: UInt64 = 0
        var aggregate = SHA256()
        let manifestSHA = OfficialQwenPayloadVerifier.checksumManifestSHA256
        aggregate.update(data: Data((manifestSHA + "\n").utf8))
        for shard in descriptor.shards.sorted(by: { $0.filename < $1.filename }) {
            guard let stamp = snapshot.files[shard.filename] else {
                throw TrustError.invalid("synthetic shard is missing")
            }
            let next = total.addingReportingOverflow(stamp.size)
            guard !next.overflow else { throw TrustError.invalid("synthetic bytes overflow") }
            total = next.partialValue
            aggregate.update(data: Data(("\(shard.filename)\t\(stamp.size)\t\(shard.sha256)\n").utf8))
        }
        guard total == bytes else {
            throw TrustError.invalid("synthetic shard byte count disagrees")
        }
        return OfficialQwenPayloadIdentity(
            checksumManifestSHA256: manifestSHA,
            shardSetSHA256: aggregate.finalize().map { String(format: "%02x", $0) }.joined(),
            shardBytes: bytes, shardCount: descriptor.shards.count)
    }

    private static func verifySidecarsAndHeaders(
        source: SourceDirectory, descriptor: OfficialSourceDescriptor
    ) throws -> [String: String] {
        let manifest = try source.read(
            leaf: OfficialQwenPayloadVerifier.checksumManifestFile,
            cap: OfficialQwenPayloadVerifier.checksumManifestBytes)
        guard UInt64(manifest.count) == OfficialQwenPayloadVerifier.checksumManifestBytes,
              hash(manifest) == OfficialQwenPayloadVerifier.checksumManifestSHA256 else {
            throw TrustError.invalid("pinned checksum manifest changed")
        }
        let entries = try parsePinnedChecksums(manifest)
        var digests = entries
        digests[OfficialQwenPayloadVerifier.checksumManifestFile] = hash(manifest)
        for name in sidecars.sorted() {
            try Task.checkCancellation()
            guard try source.fingerprint(leaf: name).size <= 1024 * 1024 * 1024 else {
                throw TrustError.invalid("sidecar exceeds 1 GiB bounded source policy: \(name)")
            }
            let path = descriptor.sourceRoot + "/" + name
            let actual = try OfficialQwenPayloadVerifier.hashRegularFile(path: path)
            guard actual.sha256 == entries[name] else {
                throw TrustError.stale("sidecar digest disagrees with pinned manifest: \(name)")
            }
        }
        // The pinned index is read through a retained no-follow descriptor and
        // its actual bytes are covered by the preceding whole-file sidecar hash.
        let index = try source.read(
            leaf: "model.safetensors.index.json",
            cap: OfficialSafetensorsSource.maximumIndexBytes)
        guard hash(index) == descriptor.sidecarSHA256["model.safetensors.index.json"] else {
            throw TrustError.stale("source index changed")
        }
        let weightMap = try OfficialSafetensorsSource.parseIndex(index)
        guard Set(weightMap.values) == Set(descriptor.shards.map(\.filename)) else {
            throw TrustError.invalid("index shard set differs from descriptor")
        }
        var allNames = Set<String>()
        for shard in descriptor.shards {
            try Task.checkCancellation()
            let header = try source.readHeader(leaf: shard.filename)
            try OfficialSafetensorsSource.validateShard(
                header, weightMap: weightMap, shardName: shard.filename)
            for tensor in header.tensors {
                guard allNames.insert(tensor.name).inserted else {
                    throw TrustError.invalid("duplicate tensor across shards")
                }
            }
        }
        guard allNames == Set(weightMap.keys) else {
            throw TrustError.invalid("index/header tensor membership differs")
        }
        return digests
    }

    private static func parsePinnedChecksums(_ data: Data) throws -> [String: String] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw TrustError.invalid("checksum manifest is not UTF-8")
        }
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        if lines.last?.isEmpty == true { lines.removeLast() }
        guard lines.count == OfficialQwenPayloadVerifier.expectedLineCount else {
            throw TrustError.invalid("checksum manifest line count")
        }
        var map: [String: String] = [:]
        for line in lines {
            guard line.utf8.count > 66 else { throw TrustError.invalid("checksum line too short") }
            let digest = String(line.prefix(64))
            let separator = line.dropFirst(64).prefix(2)
            let name = String(line.dropFirst(66))
            guard separator == "  ", isHash(digest), isSafeLeaf(name),
                  map.updateValue(digest, forKey: name) == nil else {
                throw TrustError.invalid("malformed or duplicate checksum entry")
            }
        }
        guard Set(map.keys) == expectedFiles.subtracting([OfficialQwenPayloadVerifier.checksumManifestFile]),
              descriptorSidecarHashesAgree(map) else {
            throw TrustError.invalid("checksum set differs from immutable official inventory")
        }
        return map
    }

    private static func descriptorSidecarHashesAgree(_ map: [String: String]) -> Bool {
        let pin = OfficialQwenSourceIdentity.pinned
        return pin.sidecarSHA256.allSatisfy { map[$0.key] == $0.value }
            && pin.shards.allSatisfy { map[$0.filename] == $0.sha256 }
    }

    private static func validateBinding(
        _ receipt: OfficialSourceTrustReceipt,
        descriptor: OfficialSourceDescriptor, logicalPath: String,
        markerSHA: String, expectedBytes: UInt64 = OfficialQwenPayloadVerifier.expectedShardBytes
    ) throws {
        guard receipt.version == OfficialSourceTrustReceipt.schemaVersion,
              receipt.descriptorContentSHA256 == descriptor.contentSHA256,
              receipt.sourceRoot == descriptor.sourceRoot,
              receipt.logicalModelPath == logicalPath,
              receipt.markerSHA256 == markerSHA,
              receipt.checksumManifestSHA256 == OfficialQwenPayloadVerifier.checksumManifestSHA256,
              receipt.shardBytes == expectedBytes else {
            throw TrustError.invalid("receipt identity/location/version mismatch")
        }
        guard isHash(receipt.shardSetSHA256), isHash(receipt.markerSHA256),
              isHash(receipt.descriptorContentSHA256),
              receipt.files.count == expectedFiles.count else {
            throw TrustError.invalid("receipt shape or digest invalid")
        }
        let names = receipt.files.map(\.filename)
        guard names == names.sorted(), Set(names) == expectedFiles else {
            throw TrustError.invalid("receipt file set or order differs from pinned inventory")
        }
        for file in receipt.files {
            guard isSafeLeaf(file.filename), isHash(file.sha256),
                  (file.fingerprint.mode & UInt32(S_IFMT)) == UInt32(S_IFREG),
                  file.fingerprint.size <= OfficialQwenPayloadVerifier.expectedShardBytes else {
                throw TrustError.invalid("receipt file identity or fingerprint invalid")
            }
            if file.filename == OfficialQwenPayloadVerifier.checksumManifestFile {
                guard file.sha256 == OfficialQwenPayloadVerifier.checksumManifestSHA256 else {
                    throw TrustError.invalid("receipt checksum manifest digest invalid")
                }
            } else if let pin = descriptor.shards.first(where: { $0.filename == file.filename }) {
                guard file.sha256 == pin.sha256 else {
                    throw TrustError.invalid("receipt shard digest differs from pin")
                }
            } else if let pin = descriptor.sidecarSHA256[file.filename] {
                guard file.sha256 == pin else {
                    throw TrustError.invalid("receipt sidecar digest differs from pin")
                }
            }
        }
        guard (receipt.rootFingerprint.mode & UInt32(S_IFMT)) == UInt32(S_IFDIR) else {
            throw TrustError.invalid("receipt root fingerprint is not a directory")
        }
        // The aggregate uses the same sorted encoding as the production full
        // verifier. Neither a caller-supplied digest nor a decoded receipt may
        // substitute a different shard set or invented total-byte count.
        let byName = Dictionary(uniqueKeysWithValues:
            receipt.files.map { ($0.filename, $0) })
        var total: UInt64 = 0
        var aggregate = SHA256()
        aggregate.update(data: Data((receipt.checksumManifestSHA256 + "\n").utf8))
        for shard in descriptor.shards.sorted(by: { $0.filename < $1.filename }) {
            guard let file = byName[shard.filename] else {
                throw TrustError.invalid("receipt shard is missing")
            }
            let next = total.addingReportingOverflow(file.fingerprint.size)
            guard !next.overflow else { throw TrustError.invalid("receipt shard bytes overflow") }
            total = next.partialValue
            aggregate.update(data: Data(("\(shard.filename)\t\(file.fingerprint.size)\t\(shard.sha256)\n").utf8))
        }
        guard total == expectedBytes,
              receipt.shardSetSHA256 == aggregate.finalize().map({
                  String(format: "%02x", $0)
              }).joined() else {
            throw TrustError.invalid("receipt shard-set identity or size differs from pinned verifier")
        }
    }

    private static func requireMatching(_ receipt: OfficialSourceTrustReceipt,
                                        snapshot: SourceSnapshot) throws {
        guard receipt.rootFingerprint == snapshot.root,
              receipt.files.count == snapshot.files.count,
              receipt.files.allSatisfy({ snapshot.files[$0.filename] == $0.fingerprint }) else {
            throw TrustError.stale("source files changed since full verification")
        }
    }

    private static func checkMarker(_ registered: SourceDirectory,
                                    expectedSHA: String) throws {
        let bytes = try registered.read(
            leaf: OfficialSourceDescriptor.markerFilename,
            cap: OfficialSourceDescriptor.maximumMarkerBytes)
        guard hash(bytes) == expectedSHA else {
            throw TrustError.stale("registration marker changed during verification")
        }
    }

    /// Receipt publication is the commit boundary. The same nonblocking
    /// sibling lock as the packed installer and Task 6.7 registrar is held
    /// through rechecks and the atomic rename. A failed post-rename parent
    /// sync cannot be reported as an unpublished failure.
    private static func publish(
        _ receipt: OfficialSourceTrustReceipt,
        registered: SourceDirectory, source: SourceDirectory,
        initial: SourceSnapshot, markerSHA: String,
        checkpoint: (PublicationCheckpoint) throws -> Void,
        syncDirectory: (Int32) -> Int32,
        stagedReceiptCheckpoint: ((URL) throws -> Void)?
    ) throws {
        try Task.checkCancellation()
        let bytes = try JSONEncoder().encode(receipt)
        guard UInt64(bytes.count) <= maximumReceiptBytes else {
            throw TrustError.invalid("receipt exceeds bounded size")
        }
        _ = try decodeStrict(bytes)
        let (parent, leaf) = try SourceDirectory.parentAndLeaf(registered.path)
        defer { close(parent.fd) }
        let lockName = leaf.precomposedStringWithCanonicalMapping + ".install.lock"
        let lock = try parent.lock(leaf: lockName)
        defer { _ = flock(lock, LOCK_UN); close(lock) }
        try SourceDirectory.checkEntry(parent: parent.fd, leaf: leaf,
                                       held: registered.fd)
        let previous = try registered.optionalFingerprint(leaf: receiptFilename)
        // Stage in the sibling parent. The registration itself must remain
        // marker-only or marker-plus-receipt at every observable instant.
        let stage = ".official-source-receipt-stage-\(UUID().uuidString)"
        let fd = openat(parent.fd, stage,
                        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw TrustError.io(path: stage, errno: errno) }
        var created = stat()
        guard fstat(fd, &created) == 0 else {
            let problem = errno
            close(fd)
            // Without a held identity, do not delete a possibly substituted
            // pathname. Report the staging failure instead.
            throw TrustError.io(path: stage, errno: problem)
        }
        var ownsStage = true
        defer {
            close(fd)
            if ownsStage {
                var named = stat()
                if fstatat(parent.fd, stage, &named, AT_SYMLINK_NOFOLLOW) == 0,
                   (named.st_mode & S_IFMT) == S_IFREG,
                   named.st_dev == created.st_dev, named.st_ino == created.st_ino {
                    _ = unlinkat(parent.fd, stage, 0)
                }
            }
        }
        try bytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var offset = 0
            while offset < raw.count {
                try Task.checkCancellation()
                let wrote = write(fd, base.advanced(by: offset), raw.count - offset)
                if wrote < 0, errno == EINTR { continue }
                guard wrote > 0 else {
                    throw TrustError.io(path: stage, errno: wrote < 0 ? errno : EIO)
                }
                offset += wrote
            }
        }
        guard fsync(fd) == 0 else { throw TrustError.io(path: stage, errno: errno) }
        var staged = stat()
        guard fstat(fd, &staged) == 0,
              (staged.st_mode & S_IFMT) == S_IFREG,
              staged.st_size == off_t(bytes.count) else {
            throw TrustError.stale("staged receipt changed during write")
        }
        try checkpoint(.stageSynced)
        // The package-only hook exposes only our owned sibling stage path so
        // tests can mutate it deterministically; it cannot provide bytes or
        // mint official evidence. Both hooks precede all final race checks.
        try stagedReceiptCheckpoint?(
            URL(fileURLWithPath: parent.path, isDirectory: true)
                .appendingPathComponent(stage))
        try checkpoint(.beforePublish)
        try Task.checkCancellation()
        guard try source.snapshot(names: expectedFiles) == initial else {
            throw TrustError.stale("source changed before receipt publication")
        }
        try checkMarker(registered, expectedSHA: markerSHA)
        try SourceDirectory.checkEntry(parent: parent.fd, leaf: leaf,
                                       held: registered.fd)
        var namedStage = stat()
        guard fstatat(parent.fd, stage, &namedStage, AT_SYMLINK_NOFOLLOW) == 0,
              namedStage.st_dev == created.st_dev,
              namedStage.st_ino == created.st_ino,
              try OfficialSourceTrustReceipt.Fingerprint(namedStage)
                  == OfficialSourceTrustReceipt.Fingerprint(staged) else {
            throw TrustError.stale("staged receipt changed")
        }
        try parent.checkLock(name: lockName, held: lock)
        guard try registered.optionalFingerprint(leaf: receiptFilename) == previous else {
            throw TrustError.stale("prior receipt changed before publication")
        }
        try SourceDirectory.checkNamedRegistration(registered)
        // The final source, marker, registration, stage, lock and prior-receipt
        // checks above immediately precede the atomic cross-directory rename.
        let commit: Int32
        if previous == nil {
            // No-replace protects a concurrently created registration receipt.
            commit = renameatx_np(parent.fd, stage, registered.fd,
                                  receiptFilename, UInt32(RENAME_EXCL))
        } else {
            // The existing bounded regular receipt stays intact until this
            // atomic replacement; compliant writers share the sibling lock.
            commit = renameat(parent.fd, stage, registered.fd, receiptFilename)
        }
        guard commit == 0 else {
            throw TrustError.io(path: receiptFilename, errno: errno)
        }
        ownsStage = false
        // No cancellation after publication: caller must not assume no receipt.
        errno = 0
        guard syncDirectory(registered.fd) == 0 else {
            throw TrustError.publishedDurabilityUnknown(
                path: registered.path, errno: errno == 0 ? EIO : errno)
        }
        errno = 0
        guard syncDirectory(parent.fd) == 0 else {
            throw TrustError.publishedDurabilityUnknown(
                path: parent.path, errno: errno == 0 ? EIO : errno)
        }
        // The rename has committed. Even a failed recheck now describes a
        // published receipt, not a failed attempt that preserved prior state.
        do {
            guard try source.snapshot(names: expectedFiles) == initial else {
                throw TrustError.stale("published receipt is already stale")
            }
            try checkMarker(registered, expectedSHA: markerSHA)
            try SourceDirectory.checkNamedRegistration(registered)
        } catch {
            throw TrustError.publishedStale("post-publication recheck failed: \(error)")
        }
    }

    /// Package-only strict parser for synthetic/hostile receipt tests. This
    /// validates the exact wire schema, pinned inventory and aggregate; it is
    /// not a trusted reopen without the marker, location and stat checks.
    package static func decodeReceiptStrict(data: Data) throws -> OfficialSourceTrustReceipt {
        try decodeStrict(data)
    }

    private static func decodeStrict(_ data: Data) throws -> OfficialSourceTrustReceipt {
        guard !data.isEmpty, UInt64(data.count) <= maximumReceiptBytes else {
            throw TrustError.invalid("receipt exceeds bounded size or is empty")
        }
        var scanner = ReceiptJSONScanner(data: data)
        do {
            try scanner.validate()
            let receipt = try JSONDecoder().decode(OfficialSourceTrustReceipt.self, from: data)
            try validateReceiptShape(receipt)
            return receipt
        } catch {
            throw TrustError.invalid("receipt JSON/schema: \(error)")
        }
    }

    private static func validateReceiptShape(_ receipt: OfficialSourceTrustReceipt) throws {
        guard receipt.version == OfficialSourceTrustReceipt.schemaVersion,
              isHash(receipt.descriptorContentSHA256), isHash(receipt.markerSHA256),
              isHash(receipt.checksumManifestSHA256), isHash(receipt.shardSetSHA256),
              receipt.checksumManifestSHA256 == OfficialQwenPayloadVerifier.checksumManifestSHA256,
              (receipt.rootFingerprint.mode & UInt32(S_IFMT)) == UInt32(S_IFDIR),
              receipt.files.count == expectedFiles.count else {
            throw TrustError.invalid("receipt version, identity, root or file count invalid")
        }
        let names = receipt.files.map(\.filename)
        guard names == names.sorted(), Set(names) == expectedFiles else {
            throw TrustError.invalid("receipt inventory is not the exact sorted official set")
        }
        let pinned = OfficialQwenSourceIdentity.pinned
        let shardPins = Dictionary(uniqueKeysWithValues:
            pinned.shards.map { ($0.filename, $0.sha256) })
        var byName: [String: OfficialSourceTrustReceipt.FileEntry] = [:]
        for file in receipt.files {
            guard isSafeLeaf(file.filename), isHash(file.sha256),
                  (file.fingerprint.mode & UInt32(S_IFMT)) == UInt32(S_IFREG),
                  file.fingerprint.size <= OfficialQwenPayloadVerifier.expectedShardBytes else {
                throw TrustError.invalid("invalid receipt file evidence")
            }
            if let digest = shardPins[file.filename] ?? pinned.sidecarSHA256[file.filename] {
                guard file.sha256 == digest else {
                    throw TrustError.invalid("file digest differs from immutable pin")
                }
            } else if file.filename == OfficialQwenPayloadVerifier.checksumManifestFile {
                guard file.sha256 == receipt.checksumManifestSHA256,
                      file.fingerprint.size == OfficialQwenPayloadVerifier.checksumManifestBytes else {
                    throw TrustError.invalid("receipt manifest evidence differs from pin")
                }
            }
            byName[file.filename] = file
        }
        var total: UInt64 = 0
        var aggregate = SHA256()
        aggregate.update(data: Data((receipt.checksumManifestSHA256 + "\n").utf8))
        for shard in pinned.shards.sorted(by: { $0.filename < $1.filename }) {
            guard let file = byName[shard.filename] else {
                throw TrustError.invalid("missing shard aggregate evidence")
            }
            let sum = total.addingReportingOverflow(file.fingerprint.size)
            guard !sum.overflow else { throw TrustError.invalid("shard aggregate overflow") }
            total = sum.partialValue
            aggregate.update(data: Data(("\(shard.filename)\t\(file.fingerprint.size)\t\(shard.sha256)\n").utf8))
        }
        guard total == receipt.shardBytes,
              receipt.shardSetSHA256 == aggregate.finalize().map({
                  String(format: "%02x", $0)
              }).joined() else {
            throw TrustError.invalid("receipt shard aggregate binding invalid")
        }
    }

    private static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private static func isHash(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }
    private static func isSafeLeaf(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".."
            && !value.contains("/") && !value.contains("\\") && !value.contains("\0")
    }
}

private struct SourceSnapshot: Equatable {
    let root: OfficialSourceTrustReceipt.Fingerprint
    let files: [String: OfficialSourceTrustReceipt.Fingerprint]
}

/// Retained no-follow directory traversal and metadata access. Trusted/probe
/// use fstatat only on shard leaves; no shard FD or payload range is opened.
private final class SourceDirectory {
    let path: String
    let fd: Int32
    private let observeIO: ((OfficialSourceTrustIOEvent) -> Void)?

    init(path: String,
         observeIO: ((OfficialSourceTrustIOEvent) -> Void)? = nil) throws {
        self.path = path
        self.observeIO = observeIO
        guard path.hasPrefix("/"), path != "/", !path.hasSuffix("/"),
              !path.contains("//"), !path.contains("\0") else {
            throw OfficialSourceTrust.TrustError.invalid("unsafe absolute directory path")
        }
        let pieces = path.dropFirst().split(separator: "/", omittingEmptySubsequences: false)
        guard pieces.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw OfficialSourceTrust.TrustError.invalid("dot/empty path component")
        }
        var current = open("/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard current >= 0 else { throw OfficialSourceTrust.TrustError.io(path: "/", errno: errno) }
        var soFar = ""
        for piece in pieces {
            soFar += "/" + piece
            let next = openat(current, String(piece),
                              O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
            if next < 0 {
                let problem = errno
                close(current)
                throw OfficialSourceTrust.TrustError.io(path: soFar, errno: problem)
            }
            close(current)
            current = next
        }
        fd = current
    }

    static func parentAndLeaf(_ path: String) throws -> (SourceDirectory, String) {
        guard let slash = path.lastIndex(of: "/"), slash != path.startIndex else {
            throw OfficialSourceTrust.TrustError.invalid("unsafe registration directory")
        }
        return (try SourceDirectory(path: String(path[..<slash])), String(path[path.index(after: slash)...]))
    }

    /// Reopen the parent by its logical no-follow path, then compare the
    /// named registration with the retained directory identity. Checking
    /// only through a retained parent FD can accept a detached parent tree.
    static func checkNamedRegistration(_ registered: SourceDirectory) throws {
        let location: (parent: SourceDirectory, leaf: String)
        do {
            location = try parentAndLeaf(registered.path)
        } catch {
            throw OfficialSourceTrust.TrustError.stale(
                "registration parent path no longer names the retained directory")
        }
        defer { close(location.parent.fd) }
        try checkEntry(parent: location.parent.fd, leaf: location.leaf,
                       held: registered.fd)
    }

    static func checkEntry(parent: Int32, leaf: String, held: Int32) throws {
        var named = stat(), actual = stat()
        guard fstatat(parent, leaf, &named, AT_SYMLINK_NOFOLLOW) == 0,
              fstat(held, &actual) == 0,
              named.st_dev == actual.st_dev, named.st_ino == actual.st_ino,
              (named.st_mode & S_IFMT) == S_IFDIR else {
            throw OfficialSourceTrust.TrustError.stale("registration directory changed")
        }
    }

    func lock(leaf: String) throws -> Int32 {
        let lock = openat(fd, leaf,
                          O_RDWR | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0o600)
        guard lock >= 0 else { throw OfficialSourceTrust.TrustError.io(path: leaf, errno: errno) }
        do {
            var info = stat(), named = stat()
            guard fstat(lock, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
                  info.st_nlink == 1 else {
                throw OfficialSourceTrust.TrustError.invalid("unsafe install lock")
            }
            guard flock(lock, LOCK_EX | LOCK_NB) == 0 else {
                throw OfficialSourceTrust.TrustError.io(path: leaf, errno: errno)
            }
            guard fstatat(fd, leaf, &named, AT_SYMLINK_NOFOLLOW) == 0,
                  named.st_dev == info.st_dev, named.st_ino == info.st_ino else {
                throw OfficialSourceTrust.TrustError.stale("install lock changed")
            }
            return lock
        } catch {
            _ = flock(lock, LOCK_UN)
            close(lock)
            throw error
        }
    }

    func checkLock(name: String, held: Int32) throws {
        var named = stat(), actual = stat()
        guard fstatat(fd, name, &named, AT_SYMLINK_NOFOLLOW) == 0,
              fstat(held, &actual) == 0,
              (named.st_mode & S_IFMT) == S_IFREG,
              named.st_nlink == 1,
              named.st_dev == actual.st_dev, named.st_ino == actual.st_ino else {
            throw OfficialSourceTrust.TrustError.stale("install lock changed before publication")
        }
    }

    func optionalFingerprint(leaf: String) throws -> OfficialSourceTrustReceipt.Fingerprint? {
        do { return try fingerprint(leaf: leaf) }
        catch OfficialSourceTrust.TrustError.missing { return nil }
    }

    func fingerprint(leaf: String) throws -> OfficialSourceTrustReceipt.Fingerprint {
        var info = stat()
        observeIO?(.statFile(path: path + "/" + leaf))
        guard fstatat(fd, leaf, &info, AT_SYMLINK_NOFOLLOW) == 0 else {
            if errno == ENOENT {
                if leaf == OfficialSourceTrust.receiptFilename {
                    throw OfficialSourceTrust.TrustError.missing
                }
                throw OfficialSourceTrust.TrustError.stale("required file missing: \(leaf)")
            }
            throw OfficialSourceTrust.TrustError.io(path: leaf, errno: errno)
        }
        guard (info.st_mode & S_IFMT) == S_IFREG, info.st_nlink == 1 else {
            throw OfficialSourceTrust.TrustError.invalid("nonregular or linked file: \(leaf)")
        }
        return try OfficialSourceTrustReceipt.Fingerprint(info)
    }

    func snapshot(names: Set<String>) throws -> SourceSnapshot {
        try Task.checkCancellation()
        var rootInfo = stat()
        guard fstat(fd, &rootInfo) == 0 else {
            throw OfficialSourceTrust.TrustError.io(path: path, errno: errno)
        }
        let root = try OfficialSourceTrustReceipt.Fingerprint(rootInfo)
        guard (root.mode & UInt32(S_IFMT)) == UInt32(S_IFDIR) else {
            throw OfficialSourceTrust.TrustError.invalid("source is not a directory")
        }
        var files: [String: OfficialSourceTrustReceipt.Fingerprint] = [:]
        for name in names.sorted() {
            try Task.checkCancellation()
            files[name] = try fingerprint(leaf: name)
        }
        let reopened = try SourceDirectory(path: path, observeIO: observeIO)
        defer { close(reopened.fd) }
        var after = stat()
        guard fstat(reopened.fd, &after) == 0,
              try OfficialSourceTrustReceipt.Fingerprint(after) == root else {
            throw OfficialSourceTrust.TrustError.stale("source root changed during inspection")
        }
        return SourceSnapshot(root: root, files: files)
    }

    func read(leaf: String, cap: UInt64) throws -> Data {
        try readWithFingerprint(leaf: leaf, cap: cap).bytes
    }

    func readWithFingerprint(
        leaf: String, cap: UInt64
    ) throws -> (bytes: Data, fingerprint: OfficialSourceTrustReceipt.Fingerprint) {
        let leafPath = path + "/" + leaf
        observeIO?(.openFile(path: leafPath))
        let file = openat(fd, leaf, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard file >= 0 else {
            if errno == ENOENT {
                if leaf == OfficialSourceTrust.receiptFilename {
                    throw OfficialSourceTrust.TrustError.missing
                }
                throw OfficialSourceTrust.TrustError.stale("required file missing: \(leaf)")
            }
            throw OfficialSourceTrust.TrustError.io(path: leaf, errno: errno)
        }
        defer { close(file) }
        let original = try fingerprint(leaf: leaf)
        var held = stat()
        guard fstat(file, &held) == 0,
              try OfficialSourceTrustReceipt.Fingerprint(held) == original,
              original.size <= cap, original.size <= UInt64(Int.max) else {
            throw OfficialSourceTrust.TrustError.invalid("unsafe or oversized metadata: \(leaf)")
        }
        var bytes = Data(count: Int(original.size))
        try bytes.withUnsafeMutableBytes { raw in
            var offset = 0
            while offset < raw.count {
                try Task.checkCancellation()
                observeIO?(.readFile(path: leafPath, offset: UInt64(offset),
                                     requestedBytes: raw.count - offset))
                let got = pread(file, raw.baseAddress?.advanced(by: offset),
                                raw.count - offset, off_t(offset))
                if got < 0, errno == EINTR { continue }
                guard got > 0 else {
                    throw OfficialSourceTrust.TrustError.io(
                        path: leaf, errno: got < 0 ? errno : EIO)
                }
                offset += got
            }
        }
        var final = stat()
        guard fstat(file, &final) == 0,
              try OfficialSourceTrustReceipt.Fingerprint(final) == original,
              try fingerprint(leaf: leaf) == original else {
            throw OfficialSourceTrust.TrustError.stale("metadata changed while read: \(leaf)")
        }
        return (bytes, original)
    }

    func readHeader(leaf: String) throws -> OfficialSafetensorsSource.Header {
        let leafPath = path + "/" + leaf
        observeIO?(.openFile(path: leafPath))
        let file = openat(fd, leaf, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard file >= 0 else { throw OfficialSourceTrust.TrustError.io(path: leaf, errno: errno) }
        defer { close(file) }
        let original = try fingerprint(leaf: leaf)
        var held = stat()
        guard fstat(file, &held) == 0,
              try OfficialSourceTrustReceipt.Fingerprint(held) == original else {
            throw OfficialSourceTrust.TrustError.stale("shard changed before metadata read")
        }
        let header = try OfficialSafetensorsSource.readHeader(
            path: path + "/" + leaf, fileSize: original.size) { offset, count in
                try Task.checkCancellation()
                guard offset <= UInt64(Int64.max), count >= 0,
                      UInt64(count) <= UInt64(Int64.max) - offset else {
                    throw OfficialSourceTrust.TrustError.invalid("header offset overflow")
                }
                var bytes = Data(count: count)
                try bytes.withUnsafeMutableBytes { raw in
                    var got = 0
                    while got < count {
                        try Task.checkCancellation()
                        observeIO?(.readFile(path: leafPath,
                                             offset: offset + UInt64(got),
                                             requestedBytes: count - got))
                        let n = pread(file, raw.baseAddress?.advanced(by: got),
                                      count - got, off_t(offset + UInt64(got)))
                        if n < 0, errno == EINTR { continue }
                        guard n > 0 else {
                            throw OfficialSourceTrust.TrustError.io(
                                path: leaf, errno: n < 0 ? errno : EIO)
                        }
                        got += n
                    }
                }
                return bytes
            }
        guard try fingerprint(leaf: leaf) == original,
              fstat(file, &held) == 0,
              try OfficialSourceTrustReceipt.Fingerprint(held) == original else {
            throw OfficialSourceTrust.TrustError.stale("shard changed during header read")
        }
        return header
    }
}

/// Duplicate-aware raw pass before JSONDecoder. All numbers in this schema
/// are exact nonnegative decimal integers, never bool/fraction/exponent.
private struct ReceiptJSONScanner {
    private let bytes: [UInt8]
    private var offset = 0
    private let decoder = JSONDecoder()
    private static let maxDepth = 32
    init(data: Data) { bytes = Array(data) }

    mutating func validate() throws {
        skipSpace()
        try value(depth: 0)
        skipSpace()
        guard offset == bytes.count else { throw bad() }
    }

    private mutating func value(depth: Int) throws {
        guard depth <= Self.maxDepth, offset < bytes.count else { throw bad() }
        switch bytes[offset] {
        case 0x7b: try object(depth: depth)
        case 0x5b: try array(depth: depth)
        case 0x22: _ = try string()
        case 0x30...0x39: try integer()
        default: throw bad()
        }
    }

    private mutating func object(depth: Int) throws {
        offset += 1
        skipSpace()
        if take(0x7d) { return }
        var keys = Set<String>()
        while true {
            let range = try string()
            let key = try decoder.decode(String.self, from: Data(bytes[range]))
            guard keys.insert(key).inserted else { throw bad() }
            skipSpace()
            guard take(0x3a) else { throw bad() }
            skipSpace()
            try value(depth: depth + 1)
            skipSpace()
            if take(0x7d) { return }
            guard take(0x2c) else { throw bad() }
            skipSpace()
        }
    }
    private mutating func array(depth: Int) throws {
        offset += 1
        skipSpace()
        if take(0x5d) { return }
        while true {
            try value(depth: depth + 1)
            skipSpace()
            if take(0x5d) { return }
            guard take(0x2c) else { throw bad() }
            skipSpace()
        }
    }
    private mutating func string() throws -> Range<Int> {
        guard take(0x22) else { throw bad() }
        let start = offset - 1
        while offset < bytes.count {
            let b = bytes[offset]; offset += 1
            if b == 0x22 { return start..<offset }
            guard b >= 0x20 else { throw bad() }
            if b == 0x5c {
                guard offset < bytes.count else { throw bad() }
                let escaped = bytes[offset]; offset += 1
                if escaped == 0x75 {
                    for _ in 0..<4 {
                        guard offset < bytes.count,
                              (0x30...0x39).contains(bytes[offset]) ||
                              (0x41...0x46).contains(bytes[offset]) ||
                              (0x61...0x66).contains(bytes[offset]) else { throw bad() }
                        offset += 1
                    }
                } else if ![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].contains(escaped) {
                    throw bad()
                }
            }
        }
        throw bad()
    }
    private mutating func integer() throws {
        if take(0x30) {
            guard offset == bytes.count || !(0x30...0x39).contains(bytes[offset]) else {
                throw bad()
            }
            return
        }
        guard offset < bytes.count, (0x31...0x39).contains(bytes[offset]) else { throw bad() }
        while offset < bytes.count, (0x30...0x39).contains(bytes[offset]) { offset += 1 }
    }
    private mutating func skipSpace() {
        while offset < bytes.count, [0x20, 0x09, 0x0a, 0x0d].contains(bytes[offset]) {
            offset += 1
        }
    }
    private mutating func take(_ b: UInt8) -> Bool {
        guard offset < bytes.count, bytes[offset] == b else { return false }
        offset += 1
        return true
    }
    private func bad() -> OfficialSourceTrust.TrustError {
        .invalid("invalid, duplicate, or non-integer receipt JSON")
    }
}
