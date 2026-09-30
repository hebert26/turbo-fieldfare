import Darwin
import Foundation
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

/// Checks an installed original-BF16 image companion without creating a model,
/// Metal device, mapped weight buffer, or inference session. Trusted file
/// fingerprints bind the pinned index and source shards without payload reads.
public enum OfficialQwenSourceVisionMetadataProbe {
    public static func verify(textModelURL: URL, visionURL: URL) throws {
        let descriptor = try OfficialSourceRegistration.inspect(at: textModelURL)
        let receipt = try OfficialSourceTrust.verify(
            at: textModelURL, policy: .sizeCheckTrustedReceipt)
        guard receipt.descriptorContentSHA256 == descriptor.contentSHA256,
              receipt.logicalModelPath == textModelURL.path,
              receipt.sourceRoot == descriptor.sourceRoot else {
            throw VisionPackError.incompatibleTextArtifact
        }
        let source = try OfficialSourceHandle(registrationURL: textModelURL)
        try source.validateTrustedReceipt(receipt)
        let vision = try readCompanion(at: visionURL)
        let config = QwenVisionConfig.official
        try config.validate()
        guard vision.textContentSHA256 == descriptor.contentSHA256,
              vision.processorConfigSHA256 == OfficialQwenSourceIdentity.pinned
                .sidecarSHA256["preprocessor_config.json"],
              vision.processorProfile == GTurboQwenVisionProcessorProfileV2(
                processorClass: "Qwen3VLProcessor",
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                spatialMergeSize: config.spatialMergeSize) else {
            throw VisionPackError.incompatibleTextArtifact
        }
        let contract = config.tensorContract.sorted { $0.name < $1.name }
        guard vision.tensors.count == contract.count,
              zip(vision.tensors, contract).allSatisfy({ actual, expected in
                  actual.name == expected.name && actual.shape == expected.shape
                    && expected.storage == .bf16
              }) else {
            throw VisionPackError.invalidMetadata("source vision tensor inventory mismatch")
        }
        let directory = try GTurboModelDirectory(protectedOfficialSource: source)
        let indexBytes = try directory.readMetadata(
            "model.safetensors.index.json", maxBytes: 256 * 1024)
        struct Index: Decodable { let weight_map: [String: String] }
        let index = try JSONDecoder().decode(Index.self, from: indexBytes)
        let pinnedShards = Set(OfficialQwenSourceIdentity.pinned.shards.map(\.filename))
        for tensor in vision.tensors {
            try Task.checkCancellation()
            guard let shard = index.weight_map[tensor.name],
                  pinnedShards.contains(shard) else {
                throw VisionPackError.invalidMetadata("source vision tensor is not indexed")
            }
        }
        try source.validateTrustedReceipt(receipt)
        guard try readCompanion(at: visionURL) == vision else {
            throw VisionPackError.invalidMetadata("source vision metadata changed")
        }
        _ = try OfficialSourceTrust.verify(
            at: textModelURL, policy: .sizeCheckTrustedReceipt)
    }

    private static func readCompanion(at url: URL) throws -> OfficialSourceVisionDescriptor {
        let directoryFD = open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directoryFD >= 0 else {
            throw VisionPackError.packNotFound(url.path)
        }
        defer { close(directoryFD) }
        var heldDirectory = stat()
        guard fstat(directoryFD, &heldDirectory) == 0,
              (heldDirectory.st_mode & S_IFMT) == S_IFDIR else {
            throw VisionPackError.invalidMetadata("source vision directory changed")
        }
        try requireSingleManifest(directoryFD)
        let name = OfficialSourceVisionDescriptor.filename
        let fd = openat(directoryFD, name, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            throw VisionPackError.invalidMetadata("source vision manifest unavailable")
        }
        defer { close(fd) }
        var before = stat()
        guard fstat(fd, &before) == 0,
              (before.st_mode & S_IFMT) == S_IFREG,
              before.st_nlink == 1,
              before.st_size >= 0,
              before.st_size <= OfficialSourceVisionDescriptor.maximumBytes else {
            throw VisionPackError.invalidMetadata("source vision manifest is not bounded")
        }
        var bytes = Data(count: Int(before.st_size))
        try bytes.withUnsafeMutableBytes { raw in
            var offset = 0
            while offset < raw.count {
                try Task.checkCancellation()
                let count = pread(fd, raw.baseAddress?.advanced(by: offset),
                                  raw.count - offset, off_t(offset))
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else {
                    throw VisionPackError.invalidMetadata("source vision manifest short read")
                }
                offset += count
            }
        }
        var after = stat(), named = stat(), currentDirectory = stat()
        guard fstat(fd, &after) == 0,
              fstatat(directoryFD, name, &named, AT_SYMLINK_NOFOLLOW) == 0,
              sameFile(before, after), sameFile(before, named),
              fstat(directoryFD, &currentDirectory) == 0,
              sameFile(heldDirectory, currentDirectory) else {
            throw VisionPackError.invalidMetadata("source vision manifest changed")
        }
        let namedDirectory = open(url.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard namedDirectory >= 0 else {
            throw VisionPackError.invalidMetadata("source vision directory disappeared")
        }
        defer { close(namedDirectory) }
        var namedDirectoryInfo = stat()
        guard fstat(namedDirectory, &namedDirectoryInfo) == 0,
              sameFile(heldDirectory, namedDirectoryInfo) else {
            throw VisionPackError.invalidMetadata("source vision directory replaced")
        }
        try requireSingleManifest(directoryFD)
        return try OfficialSourceVisionDescriptor.decodeStrict(bytes)
    }

    private static func requireSingleManifest(_ fd: Int32) throws {
        let copy = dup(fd)
        guard copy >= 0 else {
            throw VisionPackError.invalidMetadata("source vision directory unavailable")
        }
        guard let stream = fdopendir(copy) else {
            close(copy)
            throw VisionPackError.invalidMetadata("source vision directory unavailable")
        }
        defer { closedir(stream) }
        rewinddir(stream)
        var seen = false
        errno = 0
        while let entry = readdir(stream) {
            let name = withUnsafeBytes(of: entry.pointee.d_name) { raw -> String in
                String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
            }
            if name == "." || name == ".." { continue }
            guard name == OfficialSourceVisionDescriptor.filename, !seen else {
                throw VisionPackError.invalidMetadata("source companion must contain metadata only")
            }
            seen = true
            errno = 0
        }
        guard errno == 0, seen else {
            throw VisionPackError.invalidMetadata("source vision manifest missing")
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
}
