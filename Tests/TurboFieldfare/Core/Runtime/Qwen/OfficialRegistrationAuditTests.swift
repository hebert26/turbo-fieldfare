import Darwin
import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Opt-in Phase 21 operation. Ordinary package test runs skip this test. This
/// driver uses production registration and the public pinned full verifier;
/// it never constructs a runtime model, Metal context, or inference session.
@Suite struct OfficialRegistrationAuditTests {
    private static let source = URL(fileURLWithPath:
        "/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0",
        isDirectory: true)
    private static let text = URL(fileURLWithPath:
        "/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b.gturbo",
        isDirectory: true)
    private static let vision = URL(fileURLWithPath:
        "/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b.vision.gturbo",
        isDirectory: true)

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_P21_REGISTER_EXISTING"] == "1"))
    func registerExactExistingBF16SourceWithoutPayloadCopy() throws {
        let pinned = OfficialQwenSourceIdentity.pinned
        let descriptor = try OfficialSourceDescriptor(
            repository: pinned.repository, revision: pinned.revision,
            storageProfile: pinned.storageProfile, sidecarSHA256: pinned.sidecarSHA256,
            shards: pinned.shards.map {
                OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
            }, sourceRoot: Self.source.path)
        let manager = FileManager.default
        var existingVision = stat()
        guard lstat(Self.vision.path, &existingVision) != 0, errno == ENOENT else {
            throw Phase21OperationError.destinationConflict
        }
        if manager.fileExists(atPath: Self.text.path) {
            guard try OfficialSourceRegistration.inspect(at: Self.text) == descriptor else {
                throw Phase21OperationError.destinationConflict
            }
        } else {
            _ = try OfficialSourceRegistration.register(
                markerData: JSONEncoder().encode(descriptor), at: Self.text)
        }
        let markerOnly = try OfficialSourceTrust.probe(at: Self.text)
        #expect(markerOnly.descriptor == descriptor)
        #expect(markerOnly.receipt == .missing || markerOnly.receipt == .presentUntrusted)
        print("P21 marker published, descriptor content SHA256 \(descriptor.contentSHA256)")
        fflush(stdout)

        var lastReported: UInt64 = 0
        let receipt = try OfficialSourceTrust.verify(at: Self.text, policy: .fullSha256) {
            completed, total in
            if completed == total || completed - lastReported >= 4 * 1_073_741_824 {
                print("P21 verified bytes \(completed)/\(total)")
                fflush(stdout)
                lastReported = completed
            }
        }
        #expect(receipt.shardBytes == OfficialQwenPayloadVerifier.expectedShardBytes)
        #expect(receipt.files.filter { $0.filename.hasSuffix(".safetensors") }.count == 26)
        let reopened = try OfficialSourceTrust.verify(
            at: Self.text, policy: .sizeCheckTrustedReceipt)
        #expect(reopened == receipt)
        #expect(try OfficialSourceRegistration.inspect(at: Self.text) == descriptor)
        #expect(try ModelFamilyAdmission.classify(directoryURL: Self.text)
            == .qwenOfficialSource(descriptor))
        print("P21 trusted reopen passed, shard set SHA256 \(receipt.shardSetSHA256)")
        fflush(stdout)

        let companion = try Self.makeVisionDescriptor(
            descriptor: descriptor, receipt: reopened)
        try Self.publishVisionMetadata(companion)
        let saved = try Data(contentsOf: Self.vision.appendingPathComponent(
            OfficialSourceVisionDescriptor.filename))
        #expect(try OfficialSourceVisionDescriptor.decodeStrict(saved) == companion)
        #expect(try manager.contentsOfDirectory(atPath: Self.vision.path)
            == [OfficialSourceVisionDescriptor.filename])
        #expect(try OfficialSourceTrust.verify(
            at: Self.text, policy: .sizeCheckTrustedReceipt) == reopened)
        print("P21 vision metadata published, tensors \(companion.tensors.count), bytes \(saved.count)")
        fflush(stdout)
    }

    private static func makeVisionDescriptor(
        descriptor: OfficialSourceDescriptor,
        receipt: OfficialSourceTrustReceipt
    ) throws -> OfficialSourceVisionDescriptor {
        guard receipt.descriptorContentSHA256 == descriptor.contentSHA256,
              receipt.sourceRoot == source.path,
              receipt.logicalModelPath == text.path,
              let processorDigest = OfficialQwenSourceIdentity.pinned.sidecarSHA256[
                "preprocessor_config.json"] else {
            throw Phase21OperationError.invalidBinding
        }
        let handle = try OfficialSourceHandle(registrationURL: text)
        try handle.validateTrustedReceipt(receipt)
        let sourceDirectory = try GTurboModelDirectory(protectedOfficialSource: handle)
        let indexBytes = try sourceDirectory.readMetadata(
            "model.safetensors.index.json", maxBytes: 256 * 1024)
        struct SourceIndex: Decodable { let weight_map: [String: String] }
        let index = try JSONDecoder().decode(SourceIndex.self, from: indexBytes)
        let config = QwenVisionConfig.official
        try config.validate()
        let contract = config.tensorContract.sorted { $0.name < $1.name }
        guard contract.count == 333 else { throw Phase21OperationError.invalidVisionContract }
        var tensors: [OfficialSourceVisionDescriptor.Tensor] = []
        tensors.reserveCapacity(contract.count)
        for expected in contract {
            try Task.checkCancellation()
            guard expected.storage == .bf16,
                  let shard = index.weight_map[expected.name] else {
                throw Phase21OperationError.invalidVisionContract
            }
            let admitted = try handle.admitTensor(
                shardName: shard, tensorName: expected.name)
            guard admitted.shape == expected.shape else {
                throw Phase21OperationError.invalidVisionContract
            }
            tensors.append(.init(name: expected.name, shape: expected.shape))
        }
        try handle.validateTrustedReceipt(receipt)
        return OfficialSourceVisionDescriptor(
            textContentSHA256: descriptor.contentSHA256,
            processorConfigSHA256: processorDigest,
            processorProfile: GTurboQwenVisionProcessorProfileV2(
                processorClass: "Qwen3VLProcessor",
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                spatialMergeSize: config.spatialMergeSize),
            tensors: tensors)
    }

    private static func publishVisionMetadata(
        _ descriptor: OfficialSourceVisionDescriptor
    ) throws {
        let bytes = try JSONEncoder().encode(descriptor)
        guard try OfficialSourceVisionDescriptor.decodeStrict(bytes) == descriptor else {
            throw Phase21OperationError.invalidVisionContract
        }
        var existingVision = stat()
        guard lstat(vision.path, &existingVision) != 0, errno == ENOENT else {
            throw Phase21OperationError.destinationConflict
        }
        let parent = vision.deletingLastPathComponent()
        let stageName = ".phase21-source-vision-\(UUID().uuidString)"
        let stage = parent.appendingPathComponent(stageName, isDirectory: true)
        guard mkdir(stage.path, 0o700) == 0 else {
            throw Phase21OperationError.posix("mkdir stage", errno)
        }
        // A precommit failure leaves only this small, uniquely named metadata
        // stage for inspection. Do not recursively remove an untrusted path.
        let stagedManifest = stage.appendingPathComponent(OfficialSourceVisionDescriptor.filename)
        let fd = open(stagedManifest.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw Phase21OperationError.posix("open manifest", errno) }
        defer { close(fd) }
        try bytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var offset = 0
            while offset < raw.count {
                let count = write(fd, base.advanced(by: offset), raw.count - offset)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else {
                    throw Phase21OperationError.posix("write manifest", count < 0 ? errno : EIO)
                }
                offset += count
            }
        }
        guard fsync(fd) == 0 else { throw Phase21OperationError.posix("fsync manifest", errno) }
        var writtenManifest = stat()
        guard fstat(fd, &writtenManifest) == 0 else {
            throw Phase21OperationError.posix("fstat manifest", errno)
        }
        let stageFD = open(stage.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard stageFD >= 0 else { throw Phase21OperationError.posix("open stage", errno) }
        defer { close(stageFD) }
        guard fsync(stageFD) == 0 else { throw Phase21OperationError.posix("fsync stage", errno) }
        let parentFD = open(parent.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard parentFD >= 0 else { throw Phase21OperationError.posix("open parent", errno) }
        defer { close(parentFD) }
        var heldStage = stat(), namedStage = stat(), namedManifest = stat(), currentManifest = stat()
        guard fstat(stageFD, &heldStage) == 0,
              fstatat(parentFD, stageName, &namedStage, AT_SYMLINK_NOFOLLOW) == 0,
              fstat(fd, &currentManifest) == 0,
              fstatat(stageFD, OfficialSourceVisionDescriptor.filename,
                      &namedManifest, AT_SYMLINK_NOFOLLOW) == 0 else {
            throw Phase21OperationError.posix("recheck staged metadata", errno)
        }
        guard (heldStage.st_mode & S_IFMT) == S_IFDIR,
              heldStage.st_dev == namedStage.st_dev,
              heldStage.st_ino == namedStage.st_ino,
              heldStage.st_mode == namedStage.st_mode,
              (writtenManifest.st_mode & S_IFMT) == S_IFREG,
              writtenManifest.st_nlink == 1,
              writtenManifest.st_size == off_t(bytes.count),
              Self.sameFile(writtenManifest, currentManifest),
              Self.sameFile(writtenManifest, namedManifest) else {
            throw Phase21OperationError.invalidVisionContract
        }
        try requireSingleVisionManifest(stageFD)
        try Task.checkCancellation()
        guard renameatx_np(parentFD, stageName, parentFD, vision.lastPathComponent,
                           UInt32(RENAME_EXCL)) == 0 else {
            throw Phase21OperationError.posix("publish vision", errno)
        }
        guard fsync(parentFD) == 0 else {
            throw Phase21OperationError.posix("fsync published parent", errno)
        }
    }

    private static func sameFile(_ a: stat, _ b: stat) -> Bool {
        a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_mode == b.st_mode
            && a.st_nlink == b.st_nlink && a.st_size == b.st_size
            && a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec
            && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec
            && a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec
            && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec
    }

    private static func requireSingleVisionManifest(_ fd: Int32) throws {
        let copy = dup(fd)
        guard copy >= 0 else { throw Phase21OperationError.posix("dup stage", errno) }
        guard let stream = fdopendir(copy) else {
            let problem = errno
            close(copy)
            throw Phase21OperationError.posix("fdopendir stage", problem)
        }
        defer { closedir(stream) }
        rewinddir(stream)
        var seen = false
        errno = 0
        while let item = readdir(stream) {
            let name = withUnsafeBytes(of: item.pointee.d_name) { raw -> String in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            if name == "." || name == ".." { continue }
            guard name == OfficialSourceVisionDescriptor.filename, !seen else {
                throw Phase21OperationError.invalidVisionContract
            }
            seen = true
            errno = 0
        }
        guard errno == 0, seen else {
            throw Phase21OperationError.invalidVisionContract
        }
    }
}

private enum Phase21OperationError: Error {
    case invalidBinding
    case invalidVisionContract
    case destinationConflict
    case posix(String, Int32)
}
