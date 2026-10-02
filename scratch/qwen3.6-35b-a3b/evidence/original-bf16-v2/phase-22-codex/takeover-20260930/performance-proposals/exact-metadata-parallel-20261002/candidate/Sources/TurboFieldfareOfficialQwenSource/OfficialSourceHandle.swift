import CryptoKit
import Darwin
import Foundation
import TurboFieldfareFormat

/// Protected access to the names pinned by a registered source descriptor.
/// This is neither payload authentication nor permission to load source weights.
package enum OfficialSourceHandleError: Error, Sendable, Equatable {
    case invalidPath(String)
    case notAllowed(String)
    case replaced(String)
    case notRegular(String)
    case io(path: String, errno: Int32)
    case invalidTensor(String)
    case range(String)
    case shortRead(String)
}

package final class OfficialSourceHandle {
    package enum Checkpoint: Sendable {
        case registrationOpened
        case sourceOpened
        case fileOpened
        case beforeEnumerationReturn
    }

    /// An admitted header record, bound to this handle and the exact shard FD
    /// on which its BF16 metadata was parsed. No caller-provided range can
    /// construct this token. The token closes its own FD on deinitialization.
    package final class TensorRange {
        fileprivate let owner: OfficialSourceHandle
        fileprivate let shardName: String
        fileprivate let tensorName: String
        fileprivate let fd: Int32
        fileprivate let identity: FileIdentity
        fileprivate let fileSize: UInt64
        fileprivate let absoluteOffset: UInt64
        fileprivate let sizeBytes: UInt64
        package let shape: [UInt64]
        // Read-only metadata copied from this handle's admitted BF16 header.
        // Neither a name nor an offset is independently an admission token.
        package var admittedTensorName: String { tensorName }
        package var admittedShardName: String { shardName }
        package var admittedAbsoluteOffset: UInt64 { absoluteOffset }
        package var admittedByteCount: UInt64 { sizeBytes }

        fileprivate init(owner: OfficialSourceHandle, shardName: String, tensorName: String,
                         fd: Int32, identity: FileIdentity, absoluteOffset: UInt64,
                         sizeBytes: UInt64, shape: [UInt64]) {
            self.owner = owner
            self.shardName = shardName
            self.tensorName = tensorName
            self.fd = fd
            self.identity = identity
            self.fileSize = UInt64(identity.size)
            self.absoluteOffset = absoluteOffset
            self.sizeBytes = sizeBytes
            self.shape = shape
        }

        deinit { close(fd) }
    }

    package static let maximumTensorReadBytes: UInt64 = 8 * 1024 * 1024
    // Internal fault-injection seam: @testable shared-target tests may observe
    // syscalls, but package clients (including runtime) cannot obtain the raw
    // FD/buffer callback or bypass the normal five-argument read API.
    enum TensorPreadResult { case bytes(Int), interrupted, failure(Int32) }
    typealias TensorPread = (Int32, UnsafeMutableRawPointer, Int, off_t) -> TensorPreadResult
    enum TensorReadCheckpoint { case beforePayloadRead, afterPayloadRead, beforeReturn }
    /// Internal test limits only shrink the real off_t, allocation and syscall bounds.
    struct TensorReadLimits {
        let maximumOffset: UInt64
        let maximumAllocation: UInt64
        let maximumSyscall: UInt64
        init(maximumOffset: UInt64, maximumAllocation: UInt64,
             maximumSyscall: UInt64) {
            self.maximumOffset = maximumOffset
            self.maximumAllocation = maximumAllocation
            self.maximumSyscall = maximumSyscall
        }
    }

    package let sourceRootURL: URL
    private let registrationPath: String
    private let sourcePath: String
    private let registrationFD: Int32
    private let sourceFD: Int32
    private let registrationIdentity: DirectoryIdentity
    private let sourceIdentity: DirectoryIdentity
    private let markerIdentity: FileIdentity
    private let markerBytes: Data
    private let allowedNames: Set<String>
    private let shardNames: Set<String>
    private let checkpoint: (Checkpoint) throws -> Void
    // Constructed only from this initializer's proven paths and pinned marker.
    // No filesystem observation or caller-supplied receipt value is cached.
    private struct ValidatedDirectoryPath {
        let path: String
    }
    private struct FastValidation {
        let registration: ValidatedDirectoryPath
        let source: ValidatedDirectoryPath
        let markerSHA256: String
        let descriptorContentSHA256: String
        let expectedInventory: Set<String>
    }
    private let fastValidation: FastValidation?
    private let useMembershipScan: Bool
    // Calls can race on a retained handle. Only short value comparisons and
    // first-acceptance inserts occur under this lock; no filesystem I/O does.
    private let identityLock = NSLock()
    private var acceptedFiles: [String: FileIdentity] = [:]
    private let ioMeasurementLock = NSLock()
    private var ioMeasurement: OfficialSourceIOMeasurement?

    /// A second request cannot replace another request's numeric collector.
    package func installIOMeasurement(_ measurement: OfficialSourceIOMeasurement) -> Bool {
        ioMeasurementLock.lock(); defer { ioMeasurementLock.unlock() }
        guard ioMeasurement == nil else { return false }
        ioMeasurement = measurement
        return true
    }

    package func removeIOMeasurement(_ measurement: OfficialSourceIOMeasurement) {
        ioMeasurementLock.lock(); defer { ioMeasurementLock.unlock() }
        if ioMeasurement === measurement { ioMeasurement = nil }
    }

    private func currentIOMeasurement() -> OfficialSourceIOMeasurement? {
        ioMeasurementLock.lock(); defer { ioMeasurementLock.unlock() }
        return ioMeasurement
    }

    package convenience init(registrationURL: URL) throws {
        try self.init(registrationURL: registrationURL, checkpoint: { _ in })
    }

    /// A package-only interruption seam; it supplies no file data, identities,
    /// pins, or descriptors and cannot relax the production validation path.
    package init(registrationURL: URL, checkpoint: @escaping (Checkpoint) throws -> Void) throws {
        let registrationPath = try Self.absolutePath(registrationURL)
        let registered = try Self.openDirectory(registrationPath)
        do {
            try checkpoint(.registrationOpened)
            try Self.checkLayout(registered, path: registrationPath)
            let marker = try Self.readMarker(registered, path: registrationPath)
            let descriptor: OfficialSourceDescriptor
            do {
                descriptor = try OfficialSourceDescriptor.decodeStrict(data: marker.bytes)
                try OfficialSourceDescriptorValidation.validate(descriptor)
            } catch {
                throw OfficialSourceHandleError.invalidPath("invalid pinned registered descriptor: \(error)")
            }
            let sourcePath = try Self.absolutePath(URL(fileURLWithPath: descriptor.sourceRoot, isDirectory: true))
            guard sourcePath == descriptor.sourceRoot else {
                throw OfficialSourceHandleError.invalidPath("source root path changed by URL normalization")
            }
            let registrationParts = registrationPath.dropFirst().split(separator: "/").map(String.init)
            let sourceParts = sourcePath.dropFirst().split(separator: "/").map(String.init)
            guard !Self.isAncestry(registrationParts, sourceParts),
                  !Self.isAncestry(sourceParts, registrationParts) else {
                throw OfficialSourceHandleError.invalidPath("registration overlaps source root")
            }
            let source = try Self.openDirectory(sourcePath)
            do {
                try checkpoint(.sourceOpened)
                let allowed = Set(descriptor.shards.map(\.filename)).union(descriptor.sidecarSHA256.keys)
                guard allowed.count == descriptor.shards.count + descriptor.sidecarSHA256.count else {
                    throw OfficialSourceHandleError.invalidPath("duplicate source names")
                }
                let registeredIdentity = try Self.directoryIdentity(registered, path: registrationPath)
                let sourceIdentity = try Self.directoryIdentity(source, path: sourcePath)
                // Reopen the names before ownership passes to self. A thrown
                // initializer must close each local FD exactly once.
                try Self.requireNamedDirectory(registrationPath, retained: registeredIdentity)
                try Self.requireNamedDirectory(sourcePath, retained: sourceIdentity)
                try Self.checkLayout(registered, path: registrationPath)
                let currentMarker = try Self.readMarker(registered, path: registrationPath)
                guard currentMarker.identity == marker.identity,
                      currentMarker.bytes == marker.bytes else {
                    throw OfficialSourceHandleError.replaced("registered descriptor changed")
                }
                self.registrationPath = registrationPath
                self.sourcePath = sourcePath
                self.sourceRootURL = URL(fileURLWithPath: sourcePath, isDirectory: true)
                self.registrationFD = registered
                self.sourceFD = source
                self.registrationIdentity = registeredIdentity
                self.sourceIdentity = sourceIdentity
                self.markerIdentity = marker.identity
                self.markerBytes = marker.bytes
                self.allowedNames = allowed
                self.shardNames = Set(descriptor.shards.map(\.filename))
                self.checkpoint = checkpoint
                self.useMembershipScan = ProcessInfo.processInfo.environment[
                    "TURBO_QWEN_SOURCE_MEMBERSHIP_SCAN"] == "1"
                if ProcessInfo.processInfo.environment["TURBO_QWEN_SOURCE_VALIDATION_FAST"] == "1" {
                    let extra: Set<String> = ["LICENSE", "README.md", "chat_template.jinja",
                                              "merges.txt", "vocab.json",
                                              OfficialQwenPayloadVerifier.checksumManifestFile]
                    self.fastValidation = FastValidation(
                        registration: ValidatedDirectoryPath(path: registrationPath),
                        source: ValidatedDirectoryPath(path: sourcePath),
                        markerSHA256: SHA256.hash(data: marker.bytes).map {
                            String(format: "%02x", $0)
                        }.joined(),
                        descriptorContentSHA256: descriptor.contentSHA256,
                        expectedInventory: allowed.union(extra))
                } else {
                    self.fastValidation = nil
                }
            } catch {
                close(source)
                throw error
            }
        } catch {
            close(registered)
            throw error
        }
    }

    deinit {
        close(sourceFD)
        close(registrationFD)
    }

    /// Read the immutable registered marker already held by this handle;
    /// unlike source-root metadata reads this addresses the logical text
    /// registration, and grants neither payload nor public trust admission.
    package func registeredDescriptor() throws -> OfficialSourceDescriptor {
        try validateBinding()
        let descriptor = try OfficialSourceDescriptor.decodeStrict(data: markerBytes)
        try validateBinding()
        return descriptor
    }

    package func validateBinding() throws {
        try Task.checkCancellation()
        if let fastValidation {
            try Self.requireNamedDirectory(fastValidation.registration, retained: registrationIdentity)
            try Self.requireNamedDirectory(fastValidation.source, retained: sourceIdentity)
        } else {
            try Self.requireNamedDirectory(registrationPath, retained: registrationIdentity)
            try Self.requireNamedDirectory(sourcePath, retained: sourceIdentity)
        }
        try Self.checkLayout(registrationFD, path: registrationPath)
        let marker = try Self.readMarker(registrationFD, path: registrationPath)
        guard marker.identity == markerIdentity, marker.bytes == markerBytes else {
            throw OfficialSourceHandleError.replaced("registered descriptor changed")
        }
        try Task.checkCancellation()
    }

    /// Reopen every name already admitted through this retained handle. Its
    /// recorded file identities must still match; only bounded stat/header-free
    /// checks run here, and no filesystem I/O holds the identity lock.
    package func validateAcceptedFiles() throws {
        identityLock.lock()
        let names = acceptedFiles.keys.sorted()
        identityLock.unlock()
        for name in names {
            try Task.checkCancellation()
            let fd = try openFile(name)
            close(fd)
        }
        try validateBinding()
    }

    /// Bind an already verified receipt to THIS retained directory and its
    /// actual named source files. A verifier returning before this handle was
    /// acquired is not sufficient: every receipt fingerprint must still match
    /// the held FD AND its current literal directory entry. No payload is read.
    /// The receipt v1 records marker bytes by SHA, not marker inode; the
    /// handle's own marker inode is rechecked by validateBinding on both sides.
    package func validateTrustedReceipt(_ receipt: OfficialSourceTrustReceipt,
                                        metadataCost: OfficialSourceMetadataCost? = nil) throws {
        let totalStart = metadataCost?.start()
        defer { metadataCost?.end(.total, start: totalStart) }
        func requireBinding() throws {
            let start = metadataCost?.start()
            defer { metadataCost?.end(.binding, start: start) }
            try validateBinding()
        }
        try requireBinding()
        func requireReceiptNames() throws -> Set<String> {
            let start = metadataCost?.start()
            defer { metadataCost?.end(.receiptNames, start: start) }
            let markerDigest: String
            let descriptorContentSHA256: String
            if let fastValidation {
                markerDigest = fastValidation.markerSHA256
                descriptorContentSHA256 = fastValidation.descriptorContentSHA256
            } else {
                markerDigest = SHA256.hash(data: markerBytes).map {
                    String(format: "%02x", $0)
                }.joined()
                let descriptor = try OfficialSourceDescriptor.decodeStrict(data: markerBytes)
                descriptorContentSHA256 = descriptor.contentSHA256
            }
            guard receipt.logicalModelPath == registrationPath,
                  receipt.sourceRoot == sourcePath,
                  receipt.markerSHA256 == markerDigest,
                  receipt.descriptorContentSHA256 == descriptorContentSHA256 else {
                throw OfficialSourceHandleError.replaced("trusted source marker or path changed")
            }
            // The handle's 26 shards and 7 pinned sidecars are already restricted
            // by the descriptor. The verifier also fingerprints six inert sidecars
            // which this helper may inspect but never makes openFile-allowlisted.
            let expected: Set<String>
            if let fastValidation {
                expected = fastValidation.expectedInventory
            } else {
                let extra: Set<String> = ["LICENSE", "README.md", "chat_template.jinja",
                                          "merges.txt", "vocab.json",
                                          OfficialQwenPayloadVerifier.checksumManifestFile]
                expected = allowedNames.union(extra)
            }
            let names = receipt.files.map(\.filename)
            guard names.count == expected.count, Set(names) == expected,
                  names == names.sorted() else {
                throw OfficialSourceHandleError.replaced("trusted receipt file inventory changed")
            }
            return expected
        }
        let expected = try requireReceiptNames()
        func requireRoot() throws {
            let start = metadataCost?.start()
            defer { metadataCost?.end(.root, start: start) }
            var held = stat()
            guard fstat(sourceFD, &held) == 0,
                  Self.matchesFingerprint(held, receipt.rootFingerprint) else {
                throw OfficialSourceHandleError.replaced("trusted source root changed")
            }
            let named: Int32
            if let fastValidation {
                named = try Self.openDirectory(fastValidation.source)
            } else {
                named = try Self.openDirectory(sourcePath)
            }
            defer { close(named) }
            var current = stat()
            guard fstat(named, &current) == 0,
                  Self.matchesFingerprint(current, receipt.rootFingerprint) else {
                throw OfficialSourceHandleError.replaced("trusted source root name changed")
            }
        }
        func heldStat(_ fd: Int32, _ info: inout stat) -> Int32 {
            let start = metadataCost?.start()
            defer { metadataCost?.end(.heldStat, start: start) }
            return fstat(fd, &info)
        }
        func namedStat(_ name: String, _ info: inout stat) -> Int32 {
            let start = metadataCost?.start()
            defer { metadataCost?.end(.namedStat, start: start) }
            return fstatat(sourceFD, name, &info, AT_SYMLINK_NOFOLLOW)
        }
        func requireFile(_ fd: Int32, entry: OfficialSourceTrustReceipt.FileEntry) throws {
            let start = metadataCost?.start()
            defer { metadataCost?.end(.fileCheckAndRemember, start: start) }
            var held = stat()
            var named = stat()
            guard heldStat(fd, &held) == 0,
                  namedStat(entry.filename, &named) == 0,
                  (held.st_mode & S_IFMT) == S_IFREG, held.st_nlink == 1,
                  Self.matchesFingerprint(held, entry.fingerprint),
                  Self.matchesFingerprint(named, entry.fingerprint),
                  FileIdentity(held) == FileIdentity(named) else {
                throw OfficialSourceHandleError.replaced(
                    "trusted source file changed: \(entry.filename)")
            }
            if allowedNames.contains(entry.filename) {
                try remember(name: entry.filename, identity: FileIdentity(held))
            }
        }
        func scan() throws {
            try requireRoot()
            let listingStart = metadataCost?.start()
            let currentNames: Set<String>
            do { currentNames = try Self.entryNames(sourceFD, path: sourcePath) }
            catch {
                metadataCost?.end(.listing, start: listingStart)
                throw error
            }
            metadataCost?.end(.listing, start: listingStart)
            guard expected.isSubset(of: currentNames) else {
                throw OfficialSourceHandleError.replaced("trusted source file name disappeared")
            }
            for entry in receipt.files {
                try Task.checkCancellation()
                let openStart = metadataCost?.start()
                let fd = openat(sourceFD, entry.filename,
                                O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
                metadataCost?.end(.fileOpen, start: openStart)
                guard fd >= 0 else {
                    throw OfficialSourceHandleError.replaced(
                        "trusted source file unavailable: \(entry.filename)")
                }
                defer {
                    let start = metadataCost?.start()
                    close(fd)
                    metadataCost?.end(.fileClose, start: start)
                }
                try requireFile(fd, entry: entry)
            }
            try requireRoot()
        }
        try scan()
        try requireBinding()
        try scan()
        try requireBinding()
    }

    // This factory proves one full serial success for this handle and receipt.
    package func metadataParallelDiagnostic(_ receipt: OfficialSourceTrustReceipt) throws -> MetadataParallelDiagnostic {
        try validateTrustedReceipt(receipt)
        return MetadataParallelDiagnostic(handle: self, receipt: receipt)
    }

    package struct MetadataParallelResources: Codable, Sendable {
        package let maxActiveLanes: Int
        package let maxActiveFileDescriptors: Int
        package let finalActiveLanes: Int
        package let finalActiveFileDescriptors: Int
    }

    // The synchronous diagnostic owns this value. It is not Sendable.
    package final class MetadataParallelDiagnostic {
        private let handle: OfficialSourceHandle
        private let receipt: OfficialSourceTrustReceipt
        fileprivate init(handle: OfficialSourceHandle, receipt: OfficialSourceTrustReceipt) {
            self.handle = handle
            self.receipt = receipt
        }
        package func validate(workers: Int, metadataCost: OfficialSourceMetadataCost) throws -> MetadataParallelResources {
            precondition(workers == 1 || workers == 2 || workers == 4)
            if workers == 1 {
                try handle.validateTrustedReceipt(receipt, metadataCost: metadataCost)
                return MetadataParallelResources(maxActiveLanes: 1, maxActiveFileDescriptors: 1,
                    finalActiveLanes: 0, finalActiveFileDescriptors: 0)
            }
            return try handle.validateMetadataParallel(receipt, workers: workers, metadataCost: metadataCost)
        }
    }

    // Only counters cross lanes. No filesystem work happens under this lock.
    private final class MetadataResourceCounts: @unchecked Sendable {
        private let lock = NSLock()
        private var lanes = 0
        private var files = 0
        private var peakLanes = 0
        private var peakFiles = 0
        func lane(_ delta: Int) {
            lock.lock()
            defer { lock.unlock() }
            lanes += delta
            peakLanes = max(peakLanes, lanes)
        }
        func file(_ delta: Int) {
            lock.lock()
            defer { lock.unlock() }
            files += delta
            peakFiles = max(peakFiles, files)
        }
        func snapshot() -> MetadataParallelResources {
            lock.lock()
            defer { lock.unlock() }
            return MetadataParallelResources(maxActiveLanes: peakLanes, maxActiveFileDescriptors: peakFiles,
                finalActiveLanes: lanes, finalActiveFileDescriptors: files)
        }
    }

    private struct MetadataLaneResult {
        let cost: OfficialSourceMetadataCostSnapshot
        let error: (Int, any Error)?
    }

    // One dispatch job owns each lane. Its result is protected by resultLock.
    // Shared handle access is limited to immutable fields and locked remember.
    // The strong handle keeps both directory descriptors alive through the join.
    private final class MetadataLane: @unchecked Sendable {
        private let handle: OfficialSourceHandle
        private let entries: [OfficialSourceTrustReceipt.FileEntry]
        private let range: Range<Int>
        private let clockEnabled: Bool
        private let resources: MetadataResourceCounts
        private let resultLock = NSLock()
        private var result: MetadataLaneResult?
        init(handle: OfficialSourceHandle, entries: [OfficialSourceTrustReceipt.FileEntry],
             range: Range<Int>, clockEnabled: Bool, resources: MetadataResourceCounts) {
            self.handle = handle
            self.entries = entries
            self.range = range
            self.clockEnabled = clockEnabled
            self.resources = resources
        }
        func run() {
            resources.lane(1)
            defer { resources.lane(-1) }
            let cost = OfficialSourceMetadataCost(clockEnabled: clockEnabled)
            var firstError: (Int, any Error)?
            for index in range {
                do { try handle.metadataEntry(entries[index], cost: cost, resources: resources) }
                catch { if firstError == nil { firstError = (index, error) } }
            }
            resultLock.lock()
            result = MetadataLaneResult(cost: cost.snapshot(), error: firstError)
            resultLock.unlock()
        }
        func completed() -> MetadataLaneResult {
            resultLock.lock()
            defer { resultLock.unlock() }
            return result!
        }
    }

    private func metadataEntry(_ entry: OfficialSourceTrustReceipt.FileEntry,
                               cost: OfficialSourceMetadataCost,
                               resources: MetadataResourceCounts) throws {
        // Dispatch does not inherit caller task cancellation. This is standalone only.
        try Task.checkCancellation()
        let openStart = cost.start()
        let fd = openat(sourceFD, entry.filename, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        cost.end(.fileOpen, start: openStart)
        guard fd >= 0 else {
            throw OfficialSourceHandleError.replaced("trusted source file unavailable: \(entry.filename)")
        }
        resources.file(1)
        defer {
            let start = cost.start()
            close(fd)
            cost.end(.fileClose, start: start)
            resources.file(-1)
        }
        func heldStat(_ info: inout stat) -> Int32 {
            let start = cost.start()
            defer { cost.end(.heldStat, start: start) }
            return fstat(fd, &info)
        }
        func namedStat(_ info: inout stat) -> Int32 {
            let start = cost.start()
            defer { cost.end(.namedStat, start: start) }
            return fstatat(sourceFD, entry.filename, &info, AT_SYMLINK_NOFOLLOW)
        }
        let checkStart = cost.start()
        defer { cost.end(.fileCheckAndRemember, start: checkStart) }
        var held = stat()
        var named = stat()
        guard heldStat(&held) == 0,
              namedStat(&named) == 0,
              (held.st_mode & S_IFMT) == S_IFREG, held.st_nlink == 1,
              Self.matchesFingerprint(held, entry.fingerprint),
              Self.matchesFingerprint(named, entry.fingerprint),
              FileIdentity(held) == FileIdentity(named) else {
            throw OfficialSourceHandleError.replaced("trusted source file changed: \(entry.filename)")
        }
        if allowedNames.contains(entry.filename) {
            try remember(name: entry.filename, identity: FileIdentity(held))
        }
    }

    private func validateMetadataParallel(_ receipt: OfficialSourceTrustReceipt,
                                          workers: Int,
                                          metadataCost: OfficialSourceMetadataCost) throws -> MetadataParallelResources {
        let resources = MetadataResourceCounts()
        let totalStart = metadataCost.start()
        defer { metadataCost.end(.total, start: totalStart) }
        func requireBinding() throws {
            let start = metadataCost.start()
            defer { metadataCost.end(.binding, start: start) }
            try validateBinding()
        }
        try requireBinding()
        func requireReceiptNames() throws -> Set<String> {
            let start = metadataCost.start()
            defer { metadataCost.end(.receiptNames, start: start) }
            let markerDigest: String
            let descriptorContentSHA256: String
            if let fastValidation {
                markerDigest = fastValidation.markerSHA256
                descriptorContentSHA256 = fastValidation.descriptorContentSHA256
            } else {
                markerDigest = SHA256.hash(data: markerBytes).map {
                    String(format: "%02x", $0)
                }.joined()
                let descriptor = try OfficialSourceDescriptor.decodeStrict(data: markerBytes)
                descriptorContentSHA256 = descriptor.contentSHA256
            }
            guard receipt.logicalModelPath == registrationPath,
                  receipt.sourceRoot == sourcePath,
                  receipt.markerSHA256 == markerDigest,
                  receipt.descriptorContentSHA256 == descriptorContentSHA256 else {
                throw OfficialSourceHandleError.replaced("trusted source marker or path changed")
            }
            // The handle's 26 shards and 7 pinned sidecars are already restricted
            // by the descriptor. The verifier also fingerprints six inert sidecars
            // which this helper may inspect but never makes openFile-allowlisted.
            let expected: Set<String>
            if let fastValidation {
                expected = fastValidation.expectedInventory
            } else {
                let extra: Set<String> = ["LICENSE", "README.md", "chat_template.jinja",
                                          "merges.txt", "vocab.json",
                                          OfficialQwenPayloadVerifier.checksumManifestFile]
                expected = allowedNames.union(extra)
            }
            let names = receipt.files.map(\.filename)
            guard names.count == expected.count, Set(names) == expected,
                  names == names.sorted() else {
                throw OfficialSourceHandleError.replaced("trusted receipt file inventory changed")
            }
            return expected
        }
        let expected = try requireReceiptNames()
        func requireRoot() throws {
            let start = metadataCost.start()
            defer { metadataCost.end(.root, start: start) }
            var held = stat()
            guard fstat(sourceFD, &held) == 0,
                  Self.matchesFingerprint(held, receipt.rootFingerprint) else {
                throw OfficialSourceHandleError.replaced("trusted source root changed")
            }
            let named: Int32
            if let fastValidation {
                named = try Self.openDirectory(fastValidation.source)
            } else {
                named = try Self.openDirectory(sourcePath)
            }
            defer { close(named) }
            var current = stat()
            guard fstat(named, &current) == 0,
                  Self.matchesFingerprint(current, receipt.rootFingerprint) else {
                throw OfficialSourceHandleError.replaced("trusted source root name changed")
            }
        }
        func scan() throws {
            try requireRoot()
            let listingStart = metadataCost.start()
            let currentNames: Set<String>
            do { currentNames = try Self.entryNames(sourceFD, path: sourcePath) }
            catch {
                metadataCost.end(.listing, start: listingStart)
                throw error
            }
            metadataCost.end(.listing, start: listingStart)
            guard expected.isSubset(of: currentNames) else {
                throw OfficialSourceHandleError.replaced("trusted source file name disappeared")
            }
            let jobs = (0..<workers).map { lane in
                MetadataLane(handle: self, entries: receipt.files,
                    range: (lane * receipt.files.count / workers)..<((lane + 1) * receipt.files.count / workers),
                    clockEnabled: metadataCost.clockEnabled, resources: resources)
            }
            let group = DispatchGroup()
            for job in jobs {
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    defer { group.leave() }
                    job.run()
                }
            }
            group.wait()
            var firstError: (Int, any Error)?
            for job in jobs {
                let result = job.completed()
                metadataCost.merge(result.cost)
                if let error = result.error,
                   firstError == nil || error.0 < firstError!.0 { firstError = error }
            }
            if let firstError { throw firstError.1 }
            try requireRoot()
        }
        try scan()
        try requireBinding()
        try scan()
        try requireBinding()
        return resources.snapshot()
    }

    private static func matchesFingerprint(
        _ info: stat, _ expected: OfficialSourceTrustReceipt.Fingerprint
    ) -> Bool {
        guard info.st_size >= 0,
              let device = UInt64(exactly: info.st_dev),
              let inode = UInt64(exactly: info.st_ino),
              let modified = UInt64(exactly: info.st_mtimespec.tv_sec),
              let changed = UInt64(exactly: info.st_ctimespec.tv_sec),
              let modifiedNanos = UInt32(exactly: info.st_mtimespec.tv_nsec),
              let changedNanos = UInt32(exactly: info.st_ctimespec.tv_nsec) else {
            return false
        }
        return device == expected.device && inode == expected.inode
            && UInt32(info.st_mode) == expected.mode
            && UInt64(info.st_size) == expected.size
            && modified == expected.modifiedSeconds
            && modifiedNanos == expected.modifiedNanoseconds
            && changed == expected.changedSeconds
            && changedNanos == expected.changedNanoseconds
    }

    /// Read only one of the two pinned tokenizer sidecars through this retained
    /// source directory. The caller's cap may shrink, never increase, the fixed
    /// per-file cap. A changed entry or short read never returns partial data.
    package func readBoundedSidecar(_ name: String, maximumBytes: UInt64) throws -> Data {
        let hardLimit: UInt64
        switch name {
        case "tokenizer.json": hardLimit = 16 * 1024 * 1024
        case "tokenizer_config.json": hardLimit = 256 * 1024
        default: throw OfficialSourceHandleError.notAllowed(name)
        }
        guard allowedNames.contains(name), maximumBytes > 0 else {
            throw OfficialSourceHandleError.notAllowed(name)
        }
        let cap = min(maximumBytes, hardLimit)
        let fd = try openFile(name)
        defer { close(fd) }
        let original = try Self.fileIdentity(fd, path: name)
        guard original.size >= 0, UInt64(original.size) <= cap,
              let count = Int(exactly: original.size) else {
            throw OfficialSourceHandleError.range("sidecar exceeds bounded size: \(name)")
        }
        var bytes = Data(count: count)
        try bytes.withUnsafeMutableBytes { raw in
            var offset = 0
            while offset < count {
                try Task.checkCancellation()
                let got = pread(fd, raw.baseAddress?.advanced(by: offset),
                                count - offset, off_t(offset))
                if got < 0 && errno == EINTR { continue }
                if got < 0 { throw OfficialSourceHandleError.io(path: name, errno: errno) }
                guard got > 0, got <= count - offset else {
                    throw OfficialSourceHandleError.shortRead("sidecar changed or ended: \(name)")
                }
                offset += got
            }
        }
        try validateBinding()
        var named = stat()
        guard fstatat(sourceFD, name, &named, AT_SYMLINK_NOFOLLOW) == 0,
              try Self.fileIdentity(fd, path: name) == original,
              FileIdentity(named) == original else {
            throw OfficialSourceHandleError.replaced("tokenizer sidecar changed: \(name)")
        }
        // openFile also enforces the handle's first-accepted identity and the
        // literal directory entry after the final read.
        let reopened = try openFile(name)
        close(reopened)
        try Task.checkCancellation()
        return bytes
    }

    /// Caller owns the returned FD. No shard bytes are read by this method.
    package func openFile(_ name: String) throws -> Int32 {
        try requireAllowed(name)
        try validateBinding()
        // On case-insensitive volumes, openat can resolve a different spelling.
        // Require the literal directory entry before opening it.
        guard try Self.entryNames(sourceFD, path: sourcePath).contains(name) else {
            throw OfficialSourceHandleError.replaced("allowlisted name is absent: \(name)")
        }
        let fd = openat(sourceFD, name, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw Self.openError(sourcePath + "/" + name) }
        do {
            try checkpoint(.fileOpened)
            let held = try Self.fileIdentity(fd, path: name)
            guard (held.mode & S_IFMT) == S_IFREG, held.links == 1 else {
                throw OfficialSourceHandleError.notRegular(name)
            }
            var named = stat()
            guard fstatat(sourceFD, name, &named, AT_SYMLINK_NOFOLLOW) == 0 else {
                throw OfficialSourceHandleError.replaced("source file disappeared: \(name)")
            }
            guard held == FileIdentity(named) else {
                throw OfficialSourceHandleError.replaced("source file changed: \(name)")
            }
            try validateBinding()
            guard try Self.entryNames(sourceFD, path: sourcePath).contains(name),
                  try Self.fileIdentity(fd, path: name) == held else {
                throw OfficialSourceHandleError.replaced("source file changed: \(name)")
            }
            var final = stat()
            guard fstatat(sourceFD, name, &final, AT_SYMLINK_NOFOLLOW) == 0,
                  FileIdentity(final) == held else {
                throw OfficialSourceHandleError.replaced("source file replaced: \(name)")
            }
            try remember(name: name, identity: held)
            return fd
        } catch {
            close(fd)
            throw error
        }
    }

    package func basenames() throws -> Set<String> {
        try validateBinding()
        let first = try inventory()
        try checkpoint(.beforeEnumerationReturn)
        try validateBinding()
        guard try inventory() == first else {
            throw OfficialSourceHandleError.replaced("source inventory changed")
        }
        try remember(first)
        return Set(first.keys)
    }

    /// Admit only an actual BF16 entry from the strict, bounded Safetensors
    /// header. The header is read from the SAME retained shard FD placed in
    /// the token; the payload is never read during admission.
    package func admitTensor(shardName: String, tensorName: String) throws -> TensorRange {
        guard shardNames.contains(shardName), !tensorName.isEmpty else {
            throw OfficialSourceHandleError.notAllowed(shardName + ":" + tensorName)
        }
        let fd = try openFile(shardName)
        do {
            let identity = try Self.fileIdentity(fd, path: shardName)
            guard identity.size >= 0 else { throw OfficialSourceHandleError.notRegular(shardName) }
            try validateRetainedFile(fd, shardName: shardName, identity: identity)
            let header: OfficialSafetensorsSource.Header
            do {
                header = try OfficialSafetensorsSource.readHeader(
                    path: sourcePath + "/" + shardName,
                    fileSize: UInt64(identity.size),
                    readAt: { offset, count in
                        guard offset <= UInt64(Int64.max), count >= 0,
                              UInt64(count) <= UInt64(Int64.max) - offset else {
                            throw OfficialSourceHandleError.range("header offset exceeds off_t")
                        }
                        if count == 0 { return Data() }
                        var bytes = Data(count: count)
                        try bytes.withUnsafeMutableBytes { raw in
                            guard let base = raw.baseAddress else {
                                throw OfficialSourceHandleError.shortRead("empty header buffer")
                            }
                            var total = 0
                            while total < count {
                                try Task.checkCancellation()
                                let got = Darwin.pread(fd, base.advanced(by: total),
                                                        count - total, off_t(offset + UInt64(total)))
                                if got < 0 {
                                    if errno == EINTR { continue }
                                    throw OfficialSourceHandleError.io(path: shardName, errno: errno)
                                }
                                guard got > 0, got <= count - total else {
                                    throw OfficialSourceHandleError.shortRead(
                                        "EOF or impossible shard header byte count")
                                }
                                total += got
                            }
                        }
                        return bytes
                    })
            } catch let error as OfficialSourceHandleError { throw error }
            catch let error as CancellationError { throw error }
            catch { throw OfficialSourceHandleError.invalidTensor("invalid shard header: \(error)") }
            guard let tensor = header.tensors.first(where: { $0.name == tensorName }),
                  tensor.dtype == "BF16" else {
                throw OfficialSourceHandleError.invalidTensor("missing or non-BF16 tensor: \(tensorName)")
            }
            try validateRetainedFile(fd, shardName: shardName, identity: identity)
            try Task.checkCancellation()
            // No throwing work after ownership of fd passes into the token.
            return TensorRange(owner: self, shardName: shardName, tensorName: tensorName,
                               fd: fd, identity: identity,
                               absoluteOffset: tensor.absoluteOffset, sizeBytes: tensor.sizeBytes,
                               shape: tensor.shape)
        } catch {
            close(fd)
            throw error
        }
    }

    /// Exact byte access relative to an admitted tensor, including odd and
    /// non-page-aligned slices. A zero-byte read at the end is valid. The
    /// caller's budget cannot raise the fixed 8 MiB per-call ceiling.
    package func preadTensorRange(_ token: TensorRange, byteOffset: UInt64,
                                  byteCount: UInt64, expectedByteCount: UInt64,
                                  allocationBudget: UInt64) throws -> Data {
        try preadTensorRange(token, byteOffset: byteOffset, byteCount: byteCount,
                             expectedByteCount: expectedByteCount,
                             allocationBudget: allocationBudget,
                             readAt: { fd, buffer, count, offset in
                                let n = Darwin.pread(fd, buffer, count, offset)
                                if n >= 0 { return .bytes(n) }
                                if errno == EINTR { return .interrupted }
                                return .failure(errno)
                             }, checkpoint: { _ in }, limits: nil)
    }

    /// Direct protected read into caller-owned memory. On failure the buffer
    /// may be partially written and MUST be discarded. No Data is allocated.
    package func preadTensorRange(_ token: TensorRange, byteOffset: UInt64,
                                  byteCount: UInt64, expectedByteCount: UInt64,
                                  into destination: UnsafeMutableRawBufferPointer) throws {
        try preadTensorRange(token, byteOffset: byteOffset, byteCount: byteCount,
                             expectedByteCount: expectedByteCount, into: destination,
                             readAt: { fd, buffer, count, offset in
                                let n = Darwin.pread(fd, buffer, count, offset)
                                if n >= 0 { return .bytes(n) }
                                if errno == EINTR { return .interrupted }
                                return .failure(errno)
                             }, checkpoint: { _ in }, limits: nil)
    }

    /// Internal direct-buffer fault seam. It cannot be invoked by package
    /// runtime clients; failure leaves the supplied destination invalid.
    func preadTensorRange(_ token: TensorRange, byteOffset: UInt64,
                          byteCount: UInt64, expectedByteCount: UInt64,
                          into destination: UnsafeMutableRawBufferPointer,
                          readAt: TensorPread,
                          checkpoint: (TensorReadCheckpoint) throws -> Void,
                          limits: TensorReadLimits? = nil) throws {
        try readTensorRangeCore(token, byteOffset: byteOffset, byteCount: byteCount,
                                expectedByteCount: expectedByteCount,
                                allocationBudget: UInt64(destination.count), into: destination,
                                readAt: readAt, checkpoint: checkpoint, limits: limits)
    }

    /// Internal shared-target test injection uses the same core, token and
    /// checks. Runtime package clients cannot access this raw-buffer seam.
    func preadTensorRange(_ token: TensorRange, byteOffset: UInt64,
                          byteCount: UInt64, expectedByteCount: UInt64,
                          allocationBudget: UInt64, readAt: TensorPread,
                          checkpoint: (TensorReadCheckpoint) throws -> Void,
                          limits: TensorReadLimits? = nil) throws -> Data {
        // Validate geometry and identity before allocating the return value.
        _ = try readPlan(token, byteOffset: byteOffset, byteCount: byteCount,
                         expectedByteCount: expectedByteCount,
                         allocationBudget: allocationBudget, limits: limits)
        try validateRetainedFile(token.fd, shardName: token.shardName,
                                 identity: token.identity)
        var result = Data(count: Int(byteCount))
        try result.withUnsafeMutableBytes { raw in
            try readTensorRangeCore(token, byteOffset: byteOffset, byteCount: byteCount,
                                    expectedByteCount: expectedByteCount,
                                    allocationBudget: allocationBudget, into: raw,
                                    readAt: readAt, checkpoint: checkpoint, limits: limits)
        }
        return result
    }

    private func readPlan(_ token: TensorRange, byteOffset: UInt64,
                          byteCount: UInt64, expectedByteCount: UInt64,
                          allocationBudget: UInt64, limits: TensorReadLimits?) throws
        -> (absolute: UInt64, tile: Int) {
        guard token.owner === self else {
            throw OfficialSourceHandleError.invalidTensor("foreign admission token")
        }
        let limit = min(min(allocationBudget, Self.maximumTensorReadBytes),
                        limits?.maximumAllocation ?? UInt64.max)
        guard byteCount == expectedByteCount, byteCount <= limit,
              byteCount <= UInt64(Int.max), byteCount <= UInt64(Int64.max) else {
            throw OfficialSourceHandleError.range("count, expected count or allocation budget invalid")
        }
        let (relativeEnd, relativeOverflow) = byteOffset.addingReportingOverflow(byteCount)
        let (absolute, absoluteOverflow) = token.absoluteOffset.addingReportingOverflow(byteOffset)
        let (absoluteEnd, endOverflow) = absolute.addingReportingOverflow(byteCount)
        let maximumOffset = min(UInt64(Int64.max), limits?.maximumOffset ?? UInt64.max)
        guard !relativeOverflow, !absoluteOverflow, !endOverflow,
              relativeEnd <= token.sizeBytes, absoluteEnd <= token.fileSize,
              absolute <= maximumOffset, absoluteEnd <= maximumOffset else {
            throw OfficialSourceHandleError.range("tensor/file/off_t boundary or arithmetic overflow")
        }
        // Read directly into the admitted caller destination, within the
        // unchanged 8 MiB per-call ceiling. Each actual syscall retains its
        // before/after checks, with at most 4 MiB between those checks.
        let tile = min(UInt64(4 * 1024 * 1024), limits?.maximumSyscall ?? UInt64.max)
        guard byteCount == 0 || tile > 0 else {
            throw OfficialSourceHandleError.range("syscall capacity is zero")
        }
        return (absolute, Int(tile))
    }

    private func readTensorRangeCore(_ token: TensorRange, byteOffset: UInt64,
                                     byteCount: UInt64, expectedByteCount: UInt64,
                                     allocationBudget: UInt64,
                                     into destination: UnsafeMutableRawBufferPointer,
                                     readAt: TensorPread,
                                     checkpoint: (TensorReadCheckpoint) throws -> Void,
                                     limits: TensorReadLimits?) throws {
        let measurement = currentIOMeasurement()
        var measuredRead = measurement?.beginRead(byteCount: byteCount)
        defer { if let measuredRead { measurement?.merge(measuredRead) } }
        guard UInt64(destination.count) == byteCount else {
            throw OfficialSourceHandleError.range("destination capacity must equal requested bytes")
        }
        let plan = try readPlan(token, byteOffset: byteOffset, byteCount: byteCount,
                                expectedByteCount: expectedByteCount,
                                allocationBudget: allocationBudget, limits: limits)
        // Payload reads validate before/after each syscall; empty reads still
        // validate before return. Reject entry cancellation without a duplicate check.
        try Task.checkCancellation()
        var total = 0
        while total < destination.count {
            try Task.checkCancellation()
            try checkpoint(.beforePayloadRead)
            try OfficialSourceIOMeasurement.validate(&measuredRead, site: .beforeRead) {
                try validateRetainedFile(token.fd, shardName: token.shardName, identity: token.identity)
            }
            let requested = min(destination.count - total, plan.tile)
            let fileOffset = plan.absolute + UInt64(total) // bounded by readPlan
            guard let base = destination.baseAddress else {
                throw OfficialSourceHandleError.range("missing destination storage")
            }
            let outcome = OfficialSourceIOMeasurement.pread(&measuredRead, requested: requested) {
                readAt(token.fd, base.advanced(by: total), requested, off_t(fileOffset))
            }
            switch outcome {
            case .interrupted: continue
            case .failure(let code) where code == EINTR: continue
            case .failure(let code):
                throw OfficialSourceHandleError.io(path: token.shardName,
                                                   errno: code == 0 ? EIO : code)
            case .bytes(let got):
                guard got > 0 else {
                    throw OfficialSourceHandleError.shortRead("EOF before requested tensor bytes")
                }
                guard got <= requested else {
                    throw OfficialSourceHandleError.shortRead("impossible pread byte count")
                }
                total += got
            }
            try checkpoint(.afterPayloadRead)
            try OfficialSourceIOMeasurement.validate(&measuredRead, site: .afterRead) {
                try validateRetainedFile(token.fd, shardName: token.shardName, identity: token.identity)
            }
        }
        try checkpoint(.beforeReturn)
        try OfficialSourceIOMeasurement.validate(&measuredRead, site: .beforeReturn) {
            try validateRetainedFile(token.fd, shardName: token.shardName, identity: token.identity)
        }
        try Task.checkCancellation()
        measuredRead?.failed = false
    }

    private func validateRetainedFile(_ fd: Int32, shardName: String,
                                      identity: FileIdentity) throws {
        try validateBinding()
        let containsShard: Bool
        if useMembershipScan {
            containsShard = try Self.entryContains(sourceFD, path: sourcePath, name: shardName)
        } else {
            containsShard = try Self.entryNames(sourceFD, path: sourcePath).contains(shardName)
        }
        guard containsShard else {
            throw OfficialSourceHandleError.replaced("source shard name changed: \(shardName)")
        }
        let held = try Self.fileIdentity(fd, path: shardName)
        var named = stat()
        guard fstatat(sourceFD, shardName, &named, AT_SYMLINK_NOFOLLOW) == 0,
              held == identity, FileIdentity(named) == identity,
              (held.mode & S_IFMT) == S_IFREG, held.links == 1 else {
            throw OfficialSourceHandleError.replaced("admitted shard changed: \(shardName)")
        }
        try remember(name: shardName, identity: identity)
    }

    private func inventory() throws -> [String: FileIdentity] {
        let names = try Self.entryNames(sourceFD, path: sourcePath).intersection(allowedNames)
        var result: [String: FileIdentity] = [:]
        for name in names.sorted() {
            try Task.checkCancellation()
            var info = stat()
            guard fstatat(sourceFD, name, &info, AT_SYMLINK_NOFOLLOW) == 0 else {
                throw OfficialSourceHandleError.replaced("source entry changed: \(name)")
            }
            let identity = FileIdentity(info)
            guard (identity.mode & S_IFMT) == S_IFREG, identity.links == 1 else {
                throw OfficialSourceHandleError.notRegular(name)
            }
            result[name] = identity
        }
        return result
    }

    private func remember(name: String, identity: FileIdentity) throws {
        guard fastValidation != nil else {
            try remember([name: identity])
            return
        }
        identityLock.lock()
        defer { identityLock.unlock() }
        if let accepted = acceptedFiles[name], accepted != identity {
            throw OfficialSourceHandleError.replaced("previously accepted source file changed: \(name)")
        }
        acceptedFiles[name] = identity
    }

    private func remember(_ identities: [String: FileIdentity]) throws {
        identityLock.lock()
        defer { identityLock.unlock() }
        for (name, identity) in identities {
            if let accepted = acceptedFiles[name], accepted != identity {
                throw OfficialSourceHandleError.replaced("previously accepted source file changed: \(name)")
            }
        }
        for (name, identity) in identities { acceptedFiles[name] = identity }
    }

    private func requireAllowed(_ name: String) throws {
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"),
              !name.contains("\\"), !name.contains("\0"), allowedNames.contains(name) else {
            throw OfficialSourceHandleError.notAllowed(name)
        }
    }

    private static func absolutePath(_ url: URL) throws -> String {
        guard url.isFileURL else { throw OfficialSourceHandleError.invalidPath("not a file URL") }
        let path = url.path
        guard path.hasPrefix("/"), path != "/", !path.hasSuffix("/"),
              !path.contains("//"), !path.contains("\0") else {
            throw OfficialSourceHandleError.invalidPath("unsafe absolute path")
        }
        let parts = path.dropFirst().split(separator: "/", omittingEmptySubsequences: false)
        guard parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw OfficialSourceHandleError.invalidPath("unsafe path component")
        }
        return path
    }

    private static func isAncestry(_ prefix: [String], _ path: [String]) -> Bool {
        prefix.count <= path.count && zip(prefix, path).allSatisfy {
            GTurboPathValidator.appleFilesystemKey($0.0)
                == GTurboPathValidator.appleFilesystemKey($0.1)
        }
    }

    private static func openDirectory(_ path: String) throws -> Int32 {
        _ = try absolutePath(URL(fileURLWithPath: path, isDirectory: true))
        // Fresh named-path resolution rejects symlinks in every component,
        // enforces ancestor search access and opens the final directory readably.
        // O_NOFOLLOW_ANY must not be combined with O_NOFOLLOW.
        let fd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW_ANY | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw openError(path) }
        return fd
    }

    // Only FastValidation's init-proven immutable paths use this overload.
    // Still resolves the name freshly with the identical flags and errors.
    private static func openDirectory(_ validated: ValidatedDirectoryPath) throws -> Int32 {
        let path = validated.path
        let fd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW_ANY | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw openError(path) }
        return fd
    }

    private static func requireNamedDirectory(_ validated: ValidatedDirectoryPath,
                                              retained: DirectoryIdentity) throws {
        let path = validated.path
        let fd: Int32
        do { fd = try openDirectory(validated) }
        catch { throw OfficialSourceHandleError.replaced("directory path changed: \(path)") }
        defer { close(fd) }
        guard try directoryIdentity(fd, path: path) == retained else {
            throw OfficialSourceHandleError.replaced("directory replaced: \(path)")
        }
    }

    private static func requireNamedDirectory(_ path: String, retained: DirectoryIdentity) throws {
        let fd: Int32
        do { fd = try openDirectory(path) }
        catch { throw OfficialSourceHandleError.replaced("directory path changed: \(path)") }
        defer { close(fd) }
        guard try directoryIdentity(fd, path: path) == retained else {
            throw OfficialSourceHandleError.replaced("directory replaced: \(path)")
        }
    }

    private static func directoryIdentity(_ fd: Int32, path: String) throws -> DirectoryIdentity {
        var info = stat()
        guard fstat(fd, &info) == 0 else { throw OfficialSourceHandleError.io(path: path, errno: errno) }
        guard (info.st_mode & S_IFMT) == S_IFDIR else {
            throw OfficialSourceHandleError.invalidPath("not a directory: \(path)")
        }
        return DirectoryIdentity(device: info.st_dev, inode: info.st_ino)
    }

    private static func fileIdentity(_ fd: Int32, path: String) throws -> FileIdentity {
        var info = stat()
        guard fstat(fd, &info) == 0 else { throw OfficialSourceHandleError.io(path: path, errno: errno) }
        return FileIdentity(info)
    }

    private static func readMarker(_ directory: Int32, path: String) throws -> (bytes: Data, identity: FileIdentity) {
        let name = OfficialSourceDescriptor.markerFilename
        let fd = openat(directory, name, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw openError(path + "/" + name) }
        defer { close(fd) }
        let original = try fileIdentity(fd, path: name)
        guard (original.mode & S_IFMT) == S_IFREG, original.links == 1,
              original.size >= 0, UInt64(original.size) <= OfficialSourceDescriptor.maximumMarkerBytes else {
            throw OfficialSourceHandleError.notRegular("unbounded or nonregular marker")
        }
        var data = Data(count: Int(original.size))
        try data.withUnsafeMutableBytes { raw in
            var offset = 0
            while offset < raw.count {
                try Task.checkCancellation()
                let got = pread(fd, raw.baseAddress?.advanced(by: offset), raw.count - offset, off_t(offset))
                if got < 0 && errno == EINTR { continue }
                guard got > 0 else {
                    throw OfficialSourceHandleError.io(path: name, errno: got < 0 ? errno : EIO)
                }
                offset += got
            }
        }
        var named = stat()
        guard fstatat(directory, name, &named, AT_SYMLINK_NOFOLLOW) == 0,
              try fileIdentity(fd, path: name) == original,
              FileIdentity(named) == original else {
            throw OfficialSourceHandleError.replaced("registered marker changed")
        }
        return (data, original)
    }

    private static func checkLayout(_ fd: Int32, path: String) throws {
        let names = try entryNames(fd, path: path)
        guard names.contains(OfficialSourceDescriptor.markerFilename),
              names.isSubset(of: [OfficialSourceDescriptor.markerFilename,
                                  OfficialSourceTrust.receiptFilename]) else {
            throw OfficialSourceHandleError.replaced("registration layout changed")
        }
        if names.contains(OfficialSourceTrust.receiptFilename) {
            let name = OfficialSourceTrust.receiptFilename
            var info = stat()
            guard fstatat(fd, name, &info, AT_SYMLINK_NOFOLLOW) == 0 else {
                throw OfficialSourceHandleError.replaced("receipt entry changed")
            }
            guard (info.st_mode & S_IFMT) == S_IFREG, info.st_nlink == 1,
                  info.st_size >= 0, UInt64(info.st_size) <= OfficialSourceTrust.maximumReceiptBytes else {
                throw OfficialSourceHandleError.notRegular("unsafe receipt entry")
            }
        }
    }

    private static func entryNames(_ fd: Int32, path: String) throws -> Set<String> {
        // A duplicate shares the retained directory's cursor with concurrent readers.
        // Opening "." relative to the retained descriptor gives this scan its own cursor
        // without resolving a caller-controlled path or following a replacement symlink.
        let scanFD = openat(fd, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard scanFD >= 0 else { throw OfficialSourceHandleError.io(path: path, errno: errno) }
        do {
            guard try directoryIdentity(fd, path: path) == directoryIdentity(scanFD, path: path) else {
                throw OfficialSourceHandleError.replaced("directory stream changed: \(path)")
            }
        } catch {
            close(scanFD)
            throw error
        }
        guard let stream = fdopendir(scanFD) else {
            let error = OfficialSourceHandleError.io(path: path, errno: errno)
            close(scanFD)
            throw error
        }
        defer { closedir(stream) }
        var names = Set<String>()
        errno = 0
        while let entry = readdir(stream) {
            let name = withUnsafeBytes(of: entry.pointee.d_name) { raw -> String in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            if name != "." && name != ".." { names.insert(name) }
            errno = 0
        }
        guard errno == 0 else { throw OfficialSourceHandleError.io(path: path, errno: errno) }
        guard try directoryIdentity(fd, path: path) == directoryIdentity(scanFD, path: path) else {
            throw OfficialSourceHandleError.replaced("directory stream changed: \(path)")
        }
        return names
    }

    // Only retained shard-name membership uses this opt-in full scan.
    // Decode every entry identically and continue through EOF after a match.
    private static func entryContains(_ fd: Int32, path: String, name wanted: String) throws -> Bool {
        // A duplicate shares the retained directory's cursor with concurrent readers.
        // Opening "." relative to the retained descriptor gives this scan its own cursor
        // without resolving a caller-controlled path or following a replacement symlink.
        let scanFD = openat(fd, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard scanFD >= 0 else { throw OfficialSourceHandleError.io(path: path, errno: errno) }
        do {
            guard try directoryIdentity(fd, path: path) == directoryIdentity(scanFD, path: path) else {
                throw OfficialSourceHandleError.replaced("directory stream changed: \(path)")
            }
        } catch {
            close(scanFD)
            throw error
        }
        guard let stream = fdopendir(scanFD) else {
            let error = OfficialSourceHandleError.io(path: path, errno: errno)
            close(scanFD)
            throw error
        }
        defer { closedir(stream) }
        var found = false
        errno = 0
        while let entry = readdir(stream) {
            let name = withUnsafeBytes(of: entry.pointee.d_name) { raw -> String in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            if name != "." && name != "..", name == wanted { found = true }
            errno = 0
        }
        guard errno == 0 else { throw OfficialSourceHandleError.io(path: path, errno: errno) }
        guard try directoryIdentity(fd, path: path) == directoryIdentity(scanFD, path: path) else {
            throw OfficialSourceHandleError.replaced("directory stream changed: \(path)")
        }
        return found
    }

    private static func openError(_ path: String) -> OfficialSourceHandleError {
        let code = errno
        if code == ELOOP || code == ENOTDIR {
            return .invalidPath("unsafe or symlink path: \(path)")
        }
        return .io(path: path, errno: code)
    }
}

private struct DirectoryIdentity: Equatable {
    let device: dev_t
    let inode: ino_t
}

private struct FileIdentity: Equatable {
    let device: dev_t
    let inode: ino_t
    let mode: mode_t
    let links: nlink_t
    let size: off_t
    let modifiedSeconds: time_t
    let modifiedNanoseconds: Int
    let changedSeconds: time_t
    let changedNanoseconds: Int

    init(_ info: stat) {
        device = info.st_dev
        inode = info.st_ino
        mode = info.st_mode
        links = info.st_nlink
        size = info.st_size
        modifiedSeconds = info.st_mtimespec.tv_sec
        modifiedNanoseconds = info.st_mtimespec.tv_nsec
        changedSeconds = info.st_ctimespec.tv_sec
        changedNanoseconds = info.st_ctimespec.tv_nsec
    }
}
