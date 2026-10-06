import CryptoKit
import Darwin
import Foundation

/// A successful full-file check against the immutable official checksum pin.
/// It describes bytes observed during this call, not a permanent filesystem seal.
public struct OfficialQwenPayloadIdentity: Sendable, Equatable {
    public let checksumManifestSHA256: String
    public let shardSetSHA256: String
    public let shardBytes: UInt64
    public let shardCount: Int
}

public enum OfficialQwenPayloadVerificationError: Error, Sendable, Equatable {
    case fileOpenFailed(path: String, errno: Int32)
    case fileStatFailed(path: String, errno: Int32)
    case preadShort(path: String, expected: Int, got: Int, errno: Int32)
    case sourceFingerprintRejected(path: String, sha256: String)
    case installStateCorrupt(path: String, detail: String)
    case installStateIncompatible(detail: String)
    case configurationInvalid(detail: String)
}

/// Only package-level tests and the RepackCore compatibility adapter may inject
/// operations. The public verifier always opens and hashes actual files with
/// immutable official pins; a callback result is not physical authentication.
package struct OfficialQwenPayloadVerifierOperations {
    package var readChecksumManifest: (_ path: String, _ maximumBytes: UInt64) throws -> Data
    package var inspectAndHashFile: (
        _ path: String, _ progress: @escaping (UInt64) throws -> Void
    ) throws -> (size: UInt64, sha256: String)
    package var cancellationCheck: () throws -> Void
    package var finalRecheck: () throws -> Void

    package init(
        readChecksumManifest: @escaping (String, UInt64) throws -> Data,
        inspectAndHashFile: @escaping (
            String, @escaping (UInt64) throws -> Void
        ) throws -> (size: UInt64, sha256: String),
        cancellationCheck: @escaping () throws -> Void,
        finalRecheck: @escaping () throws -> Void = {}
    ) {
        self.readChecksumManifest = readChecksumManifest
        self.inspectAndHashFile = inspectAndHashFile
        self.cancellationCheck = cancellationCheck
        self.finalRecheck = finalRecheck
    }

    package static func production(snapshotDirectory: String) throws -> Self {
        let files = try PinnedFileReader(directory: snapshotDirectory)
        return Self(
            readChecksumManifest: { path, maximumBytes in
                try files.readManifest(path: path, maximumBytes: maximumBytes)
            },
            inspectAndHashFile: { path, progress in
                try files.hashShard(path: path, progress: progress)
            },
            cancellationCheck: { try Task.checkCancellation() },
            finalRecheck: { try files.recheckAll() })
    }
}

/// Full source authentication, separate from Safetensors metadata validation.
/// The manifest occupies at most 3,571 bytes; each shard is read in 512 KiB
/// tiles with one retained descriptor. Space is O(tile bytes + inventory).
/// A later filesystem mutation still requires a new verification before use.
public enum OfficialQwenPayloadVerifier {
    public static let checksumManifestFile = "SHA256SUMS"
    public static let checksumManifestBytes: UInt64 = 3_571
    public static let checksumManifestSHA256 =
        "1378d1bb153694b13c20641fa5d2485dede401d1ce4c267953ba4a855fc59e7e"
    public static let expectedShardBytes: UInt64 = 71_903_776_776
    public static let expectedShardCount = 26
    public static let expectedLineCount = 38

    private static let expectedSidecars: Set<String> = [
        "LICENSE", "README.md", "chat_template.jinja", "config.json",
        "configuration.json", "generation_config.json", "merges.txt",
        "model.safetensors.index.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    ]

    /// No expected identity, checksum manifest, or digest may be supplied by a
    /// caller. This path performs real full-file reads, not injected results.
    public static func verify(
        snapshotDirectory: String,
        shardFilenames: [String],
        progress: @escaping (_ completedBytes: UInt64, _ totalBytes: UInt64) -> Void = { _, _ in }
    ) throws -> OfficialQwenPayloadIdentity {
        try Task.checkCancellation()
        let operations = try OfficialQwenPayloadVerifierOperations.production(
            snapshotDirectory: snapshotDirectory)
        return try verify(snapshotDirectory: snapshotDirectory, shardFilenames: shardFilenames,
                          operations: operations, progress: progress)
    }

    /// Package-only bounded metadata read for the legacy operations adapter.
    package static func readChecksumManifest(
        path: String, maximumBytes: UInt64
    ) throws -> Data {
        try Task.checkCancellation()
        let files = try PinnedFileReader(directory: (path as NSString).deletingLastPathComponent)
        let data = try files.readManifest(path: path, maximumBytes: maximumBytes)
        try files.recheckAll()
        try Task.checkCancellation()
        return data
    }

    /// Exercise the same descriptor-retaining, 512 KiB production hash path on
    /// tiny independent files. This does not accept alternate official pins or
    /// return a pinned payload identity.
    package static func hashRegularFile(
        path: String,
        progress: @escaping (UInt64) throws -> Void = { _ in }
    ) throws -> (size: UInt64, sha256: String) {
        try Task.checkCancellation()
        let files = try PinnedFileReader(directory: (path as NSString).deletingLastPathComponent)
        let result = try files.hashShard(path: path, progress: progress)
        try files.recheckAll()
        try Task.checkCancellation()
        return result
    }

    /// A package-only compatibility/test seam. Injected digests do not attest
    /// physical payload bytes; the public production overload never uses them.
    package static func verify(
        snapshotDirectory: String,
        shardFilenames: [String],
        operations: OfficialQwenPayloadVerifierOperations,
        progress: @escaping (_ completedBytes: UInt64, _ totalBytes: UInt64) -> Void = { _, _ in }
    ) throws -> OfficialQwenPayloadIdentity {
        try operations.cancellationCheck()
        let checksumPath = (snapshotDirectory as NSString)
            .appendingPathComponent(checksumManifestFile)
        let data = try operations.readChecksumManifest(checksumPath, checksumManifestBytes)
        let observedManifestDigest = digest(data)
        guard UInt64(data.count) == checksumManifestBytes,
              observedManifestDigest == checksumManifestSHA256 else {
            throw OfficialQwenPayloadVerificationError.sourceFingerprintRejected(
                path: checksumPath, sha256: observedManifestDigest)
        }
        let entries = try parse(data, path: checksumPath)
        let pinned = OfficialQwenSourceIdentity.pinned
        let pinnedShards = Dictionary(uniqueKeysWithValues:
            pinned.shards.map { ($0.filename, $0.sha256) })
        let shardSet = Set(shardFilenames)
        guard shardFilenames.count == expectedShardCount,
              shardSet.count == expectedShardCount,
              shardSet == Set(pinnedShards.keys),
              entries.count == expectedLineCount,
              Set(entries.keys).subtracting(shardSet) == expectedSidecars,
              pinnedShards.allSatisfy({ entries[$0.key] == $0.value }),
              pinned.sidecarSHA256.allSatisfy({ entries[$0.key] == $0.value }) else {
            throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                path: checksumPath,
                detail: "official Qwen checksum inventory is not the pinned 26-shard/12-sidecar set")
        }

        var totalBytes: UInt64 = 0
        var aggregate = SHA256()
        aggregate.update(data: Data((checksumManifestSHA256 + "\n").utf8))
        for name in shardFilenames.sorted() {
            try operations.cancellationCheck()
            guard let expectedDigest = entries[name] else {
                throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                    path: checksumPath, detail: "missing checksum for \(name)")
            }
            let path = (snapshotDirectory as NSString).appendingPathComponent(name)
            let base = totalBytes
            var lastProgress: UInt64 = 0
            let inspected = try operations.inspectAndHashFile(path) { completed in
                let sum = base.addingReportingOverflow(completed)
                guard completed >= lastProgress, !sum.overflow,
                      sum.partialValue <= expectedShardBytes else {
                    throw OfficialQwenPayloadVerificationError.configurationInvalid(
                        detail: "official Qwen authentication progress is invalid or overflows UInt64")
                }
                lastProgress = completed
                try operations.cancellationCheck()
                progress(sum.partialValue, expectedShardBytes)
                try operations.cancellationCheck()
            }
            try operations.cancellationCheck()
            guard inspected.sha256.lowercased() == expectedDigest else {
                throw OfficialQwenPayloadVerificationError.sourceFingerprintRejected(
                    path: path, sha256: inspected.sha256.lowercased())
            }
            let sum = totalBytes.addingReportingOverflow(inspected.size)
            guard !sum.overflow else {
                throw OfficialQwenPayloadVerificationError.configurationInvalid(
                    detail: "official Qwen shard bytes overflow UInt64")
            }
            totalBytes = sum.partialValue
            guard totalBytes <= expectedShardBytes else {
                throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                    path: snapshotDirectory,
                    detail: "official Qwen shards exceed \(expectedShardBytes) bytes")
            }
            aggregate.update(data: Data(("\(name)\t\(inspected.size)\t\(expectedDigest)\n").utf8))
            progress(totalBytes, expectedShardBytes)
            try operations.cancellationCheck()
        }
        guard totalBytes == expectedShardBytes else {
            throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                path: snapshotDirectory,
                detail: "official Qwen shards total \(totalBytes), expected \(expectedShardBytes)")
        }
        try operations.cancellationCheck()
        try operations.finalRecheck()
        try operations.cancellationCheck()
        return OfficialQwenPayloadIdentity(
            checksumManifestSHA256: observedManifestDigest,
            shardSetSHA256: hex(aggregate.finalize()),
            shardBytes: totalBytes,
            shardCount: shardFilenames.count)
    }

    private static func parse(_ data: Data, path: String) throws -> [String: String] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                path: path, detail: "checksum manifest is not UTF-8")
        }
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        guard lines.count == expectedLineCount else {
            throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                path: path, detail: "checksum manifest must contain 38 lines")
        }
        var entries: [String: String] = [:]
        for line in lines {
            guard line.count > 66 else {
                throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                    path: path, detail: "malformed checksum line")
            }
            let digestEnd = line.index(line.startIndex, offsetBy: 64)
            let separatorEnd = line.index(digestEnd, offsetBy: 2)
            let hash = String(line[..<digestEnd])
            let separator = line[digestEnd..<separatorEnd]
            let name = String(line[separatorEnd...])
            guard separator == "  ", isSHA256(hash), isSafeLeaf(name),
                  entries.updateValue(hash, forKey: name) == nil else {
                throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                    path: path, detail: "invalid or duplicate checksum entry")
            }
        }
        return entries
    }

    private static func isSafeLeaf(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".."
            && !name.contains("/") && !name.contains("\\") && !name.contains("\0")
    }

    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (97...102).contains($0)
        }
    }

    private static func digest(_ data: Data) -> String {
        hex(SHA256.hash(data: data))
    }

    fileprivate static func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
}

/// A stat snapshot is checked against both the still-open descriptor and the
/// directory entry, including nanosecond modification and change times.
private struct FileStamp: Equatable {
    let device: dev_t
    let inode: ino_t
    let mode: mode_t
    let size: off_t
    let modifiedSeconds: Int
    let modifiedNanoseconds: Int
    let changedSeconds: Int
    let changedNanoseconds: Int

    init(_ info: stat) {
        device = info.st_dev
        inode = info.st_ino
        mode = info.st_mode
        size = info.st_size
        modifiedSeconds = Int(info.st_mtimespec.tv_sec)
        modifiedNanoseconds = Int(info.st_mtimespec.tv_nsec)
        changedSeconds = Int(info.st_ctimespec.tv_sec)
        changedNanoseconds = Int(info.st_ctimespec.tv_nsec)
    }
}

/// Owns one root directory descriptor for the whole verification call. Every
/// leaf uses openat/fstatat relative to that descriptor, so a parent pathname
/// swap cannot redirect later reads. No shard descriptor outlives its hash.
private final class PinnedFileReader {
    private static let tileBytes = 512 * 1024
    private let directory: String
    private let directoryFD: Int32
    private let directoryStamp: FileStamp
    private var inspected: [(leaf: String, stamp: FileStamp)] = []

    init(directory: String) throws {
        self.directory = directory
        let fd = open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else {
            throw OfficialQwenPayloadVerificationError.fileOpenFailed(path: directory, errno: errno)
        }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFDIR else {
            let problem = errno == 0 ? EINVAL : errno
            close(fd)
            throw OfficialQwenPayloadVerificationError.fileStatFailed(path: directory, errno: problem)
        }
        directoryFD = fd
        directoryStamp = FileStamp(info)
        do {
            try checkRoot()
        } catch {
            close(fd)
            throw error
        }
    }

    deinit { close(directoryFD) }

    func readManifest(path: String, maximumBytes: UInt64) throws -> Data {
        let leaf = OfficialQwenPayloadVerifier.checksumManifestFile
        let fd = try openLeaf(leaf, path: path)
        defer { close(fd) }
        let original = try stamp(fd: fd, path: path)
        guard original.size >= 0, UInt64(original.size) <= maximumBytes,
              UInt64(original.size) <= UInt64(Int.max) else {
            throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                path: path, detail: "checksum manifest exceeds \(maximumBytes)-byte cap")
        }
        var data = Data(count: Int(original.size))
        try data.withUnsafeMutableBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                try Task.checkCancellation()
                let count = buffer.count - offset
                let got = pread(fd, buffer.baseAddress?.advanced(by: offset), count, off_t(offset))
                if got < 0, errno == EINTR { continue }
                guard got > 0 else {
                    throw OfficialQwenPayloadVerificationError.preadShort(
                        path: path, expected: count, got: 0, errno: got < 0 ? errno : 0)
                }
                offset += got
            }
        }
        try ensureEOF(fd: fd, size: UInt64(original.size), path: path)
        try checkUnchanged(fd: fd, leaf: leaf, original: original, path: path)
        inspected.append((leaf, original))
        return data
    }

    func hashShard(
        path: String, progress: @escaping (UInt64) throws -> Void
    ) throws -> (size: UInt64, sha256: String) {
        let leaf = (path as NSString).lastPathComponent
        let fd = try openLeaf(leaf, path: path)
        defer { close(fd) }
        let original = try stamp(fd: fd, path: path)
        guard original.size >= 0, UInt64(original.size) <= OfficialQwenPayloadVerifier.expectedShardBytes,
              original.size <= off_t(Int64.max - Int64(Self.tileBytes)) else {
            throw OfficialQwenPayloadVerificationError.installStateCorrupt(
                path: path, detail: "official Qwen shard size is invalid")
        }
        // Match the former Sha256Stream.hashFileDescriptor(..., noCache: true)
        // behavior. Failure to set this advisory flag does not fail hashing.
        _ = fcntl(fd, F_NOCACHE, 1)
        let fileBytes = UInt64(original.size)
        var completed: UInt64 = 0
        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: Self.tileBytes)
        while completed < fileBytes {
            try Task.checkCancellation()
            let requested = Int(min(UInt64(Self.tileBytes), fileBytes - completed))
            let got = buffer.withUnsafeMutableBytes {
                pread(fd, $0.baseAddress, requested, off_t(completed))
            }
            if got < 0, errno == EINTR { continue }
            guard got > 0 else {
                throw OfficialQwenPayloadVerificationError.preadShort(
                    path: path, expected: requested, got: 0, errno: got < 0 ? errno : 0)
            }
            buffer.withUnsafeBytes { raw in
                hasher.update(bufferPointer: UnsafeRawBufferPointer(
                    start: raw.baseAddress, count: got))
            }
            let next = completed.addingReportingOverflow(UInt64(got))
            guard !next.overflow, next.partialValue <= fileBytes else {
                throw OfficialQwenPayloadVerificationError.configurationInvalid(
                    detail: "official Qwen hash byte count overflows or exceeds file size")
            }
            completed = next.partialValue
            try progress(completed)
            try Task.checkCancellation()
        }
        try ensureEOF(fd: fd, size: fileBytes, path: path)
        try checkUnchanged(fd: fd, leaf: leaf, original: original, path: path)
        try Task.checkCancellation()
        inspected.append((leaf, original))
        return (completed, OfficialQwenPayloadVerifier.hex(hasher.finalize()))
    }

    func recheckAll() throws {
        try checkRoot()
        for (leaf, original) in inspected {
            try Task.checkCancellation()
            let path = (directory as NSString).appendingPathComponent(leaf)
            try recheckLeaf(leaf, path: path, original: original)
        }
        try checkRoot()
        try Task.checkCancellation()
    }

    private func recheckLeaf(_ leaf: String, path: String, original: FileStamp) throws {
        let fd = try openLeaf(leaf, path: path)
        defer { close(fd) }
        try checkUnchanged(fd: fd, leaf: leaf, original: original, path: path)
    }

    private func openLeaf(_ leaf: String, path: String) throws -> Int32 {
        let fd = openat(directoryFD, leaf, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            throw OfficialQwenPayloadVerificationError.fileOpenFailed(path: path, errno: errno)
        }
        do {
            let info = try statDescriptor(fd, path: path)
            guard info.st_mode & S_IFMT == S_IFREG else {
                throw OfficialQwenPayloadVerificationError.fileStatFailed(path: path, errno: EINVAL)
            }
            return fd
        } catch {
            close(fd)
            throw error
        }
    }

    private func stamp(fd: Int32, path: String) throws -> FileStamp {
        let info = try statDescriptor(fd, path: path)
        guard info.st_mode & S_IFMT == S_IFREG else {
            throw OfficialQwenPayloadVerificationError.fileStatFailed(path: path, errno: EINVAL)
        }
        return FileStamp(info)
    }

    private func statDescriptor(_ fd: Int32, path: String) throws -> stat {
        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw OfficialQwenPayloadVerificationError.fileStatFailed(path: path, errno: errno)
        }
        return info
    }

    private func checkUnchanged(
        fd: Int32, leaf: String, original: FileStamp, path: String
    ) throws {
        let current = try stamp(fd: fd, path: path)
        var leafInfo = stat()
        guard fstatat(directoryFD, leaf, &leafInfo, AT_SYMLINK_NOFOLLOW) == 0 else {
            throw OfficialQwenPayloadVerificationError.fileStatFailed(path: path, errno: errno)
        }
        guard current == original, FileStamp(leafInfo) == original else {
            throw OfficialQwenPayloadVerificationError.installStateIncompatible(
                detail: "official Qwen source file changed while it was authenticated: \(path)")
        }
    }

    private func checkRoot() throws {
        var info = stat()
        guard lstat(directory, &info) == 0 else {
            throw OfficialQwenPayloadVerificationError.fileStatFailed(path: directory, errno: errno)
        }
        guard FileStamp(info) == directoryStamp else {
            throw OfficialQwenPayloadVerificationError.installStateIncompatible(
                detail: "official Qwen source directory changed while it was authenticated")
        }
    }

    private func ensureEOF(fd: Int32, size: UInt64, path: String) throws {
        guard size <= UInt64(Int64.max) else {
            throw OfficialQwenPayloadVerificationError.configurationInvalid(
                detail: "official Qwen file offset is not representable")
        }
        var byte: UInt8 = 0
        while true {
            try Task.checkCancellation()
            let got: Int = withUnsafeMutablePointer(to: &byte) { pointer in
                pread(fd, pointer, 1, off_t(size))
            }
            if got < 0, errno == EINTR { continue }
            guard got >= 0 else {
                throw OfficialQwenPayloadVerificationError.preadShort(
                    path: path, expected: 1, got: 0, errno: errno)
            }
            guard got == 0 else {
                throw OfficialQwenPayloadVerificationError.installStateIncompatible(
                    detail: "official Qwen source file grew while it was authenticated: \(path)")
            }
            return
        }
    }
}
