import Foundation
import Darwin
import Darwin.Mach
import TurboFieldfareFormat

/// Counters tracked during a repack run. Emitted to JSON via the
/// `--copy-audit` flag.
public final class RepackAudit {
    public var sourceBytesRead: UInt64 = 0
    public var outputBytesWritten: UInt64 = 0
    public var intentionalCopyBytes: UInt64 = 0
    public var byteCopyTiles: UInt64 = 0
    public var largestScratchBytes: Int = 0
    public var wholeFileHeapBuffers: Bool = false
    public var platformMemmoveObserved: Bool = false
    public var mallocCountObserved: Int = 0
    public var bitWidthOverridesHonored: Int = 0
    public var sourceSnapshotSha256: String = ""
    public var tensorsDroppedMultimodal: [String] = []
    public var wallTimeSeconds: Double = 0
    public var peakRssBytes: UInt64 = 0
    public var outputFiles: [OutputFile] = []
    public var rssSamples: [UInt64] = []
    public var packedExpertLayoutMode: String = "identity"
    public var packedExpertLayoutOrderPath: String?
    public var packedExpertLayoutOrderSha256: String?
    public var packedExpertLayoutStrategy: String?
    public var packedExpertReorderedLayerCount: Int = 0
    public var packedExpertLayoutAuditLogicalIDCount: Int = 0
    public var packedExpertLayoutOffsetValidationPassed: Bool = false
    public var remoteBytesDownloaded: UInt64 = 0
    public var remoteGapBytesDownloaded: UInt64 = 0
    public var remoteRangeRequests: UInt64 = 0
    public var remoteRangeRetries: UInt64 = 0
    public var remoteRangeStreamingSupported: Bool = false
    public var largestRemoteTransferBytes: Int = 0
    public var largestRemotePayloadHeapBytes: Int = 0
    public var remoteRepoID: String?
    public var remoteRequestedRevision: String?
    public var remoteResolvedCommit: String?
    public var remoteRetries: [RemoteRetryRecord] = []
    public var stalePartialsRemoved: [String] = []

    public init() {}

    public struct OutputFile {
        public let relativePath: String
        public let size: UInt64
        public let sha256: String
    }

    public struct RemoteRetryRecord {
        public let label: String
        public let attempt: Int
        public let detail: String
    }

    public func recordTile(bytes: Int) {
        byteCopyTiles &+= 1
        intentionalCopyBytes &+= UInt64(bytes)
    }

    public func recordWrite(bytes: Int) {
        outputBytesWritten &+= UInt64(bytes)
    }

    public func recordRead(bytes: Int) {
        sourceBytesRead &+= UInt64(bytes)
    }

    public func recordRemoteRetry(label: String, attempt: Int, detail: String) {
        remoteRangeRetries &+= 1
        remoteRetries.append(RemoteRetryRecord(label: label, attempt: attempt, detail: detail))
    }

    public func sampleRSS() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            let phys = UInt64(info.phys_footprint)
            if phys > peakRssBytes { peakRssBytes = phys }
            rssSamples.append(phys)
        }
    }

    /// Audits a published text v2 directory from the format-validated
    /// manifest, not from writer-private descriptors.
    func verifyTransformedPack(
        rootDirectory: String,
        manifest verified: GTurboVerifiedManifestV2,
        receiptData: Data? = nil
    ) throws {
        try Self.verifyDirectory(
            rootDirectory: rootDirectory,
            manifestDigest: verified.manifestSHA256,
            files: verified.manifest.files,
            regions: verified.manifest.tensorRegions.map {
                ($0.name, $0.file, $0.offset, $0.size)
            },
            ignoredNames: Set(verified.manifest.ignoredTensors.map(\.name)),
            receiptData: receiptData)
    }

    /// Audits an optional v2 vision companion with the same exact-file and
    /// zero-gap rules as the text pack.
    func verifyTransformedVisionPack(
        rootDirectory: String,
        manifest verified: GTurboVerifiedVisionManifestV2,
        receiptData: Data? = nil
    ) throws {
        let path = (rootDirectory as NSString).appendingPathComponent("manifest.json")
        let data = try Posix.readBoundedData(
            path, maximumBytes: GTurboVisionFormatV2.metadataMaxBytes)
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        let digest = hasher.finalizeHexString()
        guard try GTurboVisionManifestV2Codec.decode(data) == verified else {
            throw RepackError.configurationInvalid(
                detail: "transformed vision manifest changed after validation")
        }
        try Self.verifyDirectory(
            rootDirectory: rootDirectory,
            manifestDigest: digest,
            files: verified.manifest.files,
            regions: verified.manifest.tensorRegions.map {
                ($0.name, $0.file, $0.offset, $0.size)
            },
            ignoredNames: [],
            receiptData: receiptData)
    }

    public func toJSONData(outputDir: String) throws -> Data {
        var filesArr: [[String: Any]] = []
        for f in outputFiles {
            filesArr.append([
                "path": f.relativePath,
                "size": f.size,
                "sha256": f.sha256
            ])
        }
        var retryArr: [[String: Any]] = []
        for retry in remoteRetries {
            retryArr.append([
                "label": retry.label,
                "attempt": retry.attempt,
                "detail": retry.detail
            ])
        }
        var dict: [String: Any] = [
            "output_dir": outputDir,
            "peak_rss_bytes": peakRssBytes,
            "rss_sample_count": rssSamples.count,
            "source_bytes_read": sourceBytesRead,
            "output_bytes_written": outputBytesWritten,
            "intentional_copy_bytes": intentionalCopyBytes,
            "byte_copy_tiles": byteCopyTiles,
            "largest_scratch_bytes": largestScratchBytes,
            "whole_file_heap_buffers": wholeFileHeapBuffers,
            "platform_memmove_observed": platformMemmoveObserved,
            "malloc_count_observed": mallocCountObserved,
            "bit_width_overrides_honored": bitWidthOverridesHonored,
            "source_snapshot_sha256": sourceSnapshotSha256,
            "tensors_dropped_multimodal": tensorsDroppedMultimodal,
            "wall_time_s": wallTimeSeconds,
            "packed_expert_layout_mode": packedExpertLayoutMode,
            "packed_expert_reordered_layer_count": packedExpertReorderedLayerCount,
            "packed_expert_layout_audit_logical_id_count": packedExpertLayoutAuditLogicalIDCount,
            "packed_expert_layout_offset_validation_passed": packedExpertLayoutOffsetValidationPassed,
            "remote_bytes_downloaded": remoteBytesDownloaded,
            "remote_gap_bytes_downloaded": remoteGapBytesDownloaded,
            "remote_range_requests": remoteRangeRequests,
            "remote_range_retries": remoteRangeRetries,
            "remote_range_streaming_supported": remoteRangeStreamingSupported,
            "largest_remote_transfer_bytes": largestRemoteTransferBytes,
            "largest_remote_payload_heap_bytes": largestRemotePayloadHeapBytes,
            "remote_retries": retryArr,
            "stale_partials_removed": stalePartialsRemoved,
            "output_files": filesArr
        ]
        if let packedExpertLayoutOrderPath {
            dict["packed_expert_layout_order_path"] = packedExpertLayoutOrderPath
        }
        if let packedExpertLayoutOrderSha256 {
            dict["packed_expert_layout_order_sha256"] = packedExpertLayoutOrderSha256
        }
        if let packedExpertLayoutStrategy {
            dict["packed_expert_layout_strategy"] = packedExpertLayoutStrategy
        }
        if let remoteRepoID {
            dict["remote_repo_id"] = remoteRepoID
        }
        if let remoteRequestedRevision {
            dict["remote_requested_revision"] = remoteRequestedRevision
        }
        if let remoteResolvedCommit {
            dict["remote_resolved_commit"] = remoteResolvedCommit
        }
        return try JSONSerialization.data(withJSONObject: dict,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    private static func verifyDirectory(
        rootDirectory: String,
        manifestDigest: String,
        files: [String: GTurboManifestFileV1],
        regions: [(name: String, file: String, offset: UInt64, size: UInt64)],
        ignoredNames: Set<String>,
        receiptData: Data?
    ) throws {
        guard try Posix.entryKind(rootDirectory) == .directory else {
            throw RepackError.configurationInvalid(
                detail: "transformed audit root is not a directory")
        }
        let inventory = try recursiveInventory(rootDirectory: rootDirectory)
        var expectedFiles = Set(files.keys).union(["manifest.json"])
        if receiptData != nil {
            expectedFiles.insert(VerifiedInstallReceiptWriter.fileName)
        }
        guard inventory.files == expectedFiles else {
            throw RepackError.configurationInvalid(
                detail: "transformed audit found missing or extra files")
        }
        let expectedDirectories = Set(expectedFiles.flatMap(Self.parentDirectories))
        guard inventory.directories == expectedDirectories else {
            throw RepackError.configurationInvalid(
                detail: "transformed audit found missing or extra directories")
        }
        let manifestPath = (rootDirectory as NSString)
            .appendingPathComponent("manifest.json")
        guard try Sha256Stream.hashFile(
            path: manifestPath, tileBytes: WriterCore.tileBytes,
            noCache: true, noFollow: true) == manifestDigest else {
            throw RepackError.configurationInvalid(
                detail: "transformed audit manifest digest mismatch")
        }
        if let receiptData {
            let receiptPath = (rootDirectory as NSString)
                .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
            let observed = try Posix.readBoundedData(
                receiptPath, maximumBytes: 4 * 1024 * 1024)
            guard observed == receiptData else {
                throw RepackError.configurationInvalid(
                    detail: "transformed audit receipt bytes changed")
            }
        }

        for (relativePath, expected) in files {
            let path = (rootDirectory as NSString).appendingPathComponent(relativePath)
            let fd = try Posix.openReadNoFollow(path)
            defer { close(fd) }
            let size = try Posix.fileSize(fd: fd, path: path)
            guard size == expected.size else {
                throw RepackError.configurationInvalid(
                    detail: "transformed audit size mismatch for \(relativePath)")
            }
            let digest = try Sha256Stream.hashFileDescriptor(
                fd, displayPath: path, tileBytes: WriterCore.tileBytes, noCache: true)
            guard digest == expected.sha256.lowercased() else {
                throw RepackError.configurationInvalid(
                    detail: "transformed audit digest mismatch for \(relativePath)")
            }
        }

        let names = regions.map(\.name)
        guard Set(names).count == names.count,
              ignoredNames.isDisjoint(with: names) else {
            throw RepackError.configurationInvalid(
                detail: "transformed audit tensor omission or uniqueness mismatch")
        }
        let byFile = Dictionary(grouping: regions, by: \.file)
        for (relativePath, expected) in files where !relativePath.hasSuffix(".json") {
            let sorted = (byFile[relativePath] ?? []).sorted { $0.offset < $1.offset }
            guard !sorted.isEmpty else {
                throw RepackError.configurationInvalid(
                    detail: "transformed audit has unaudited binary file \(relativePath)")
            }
            var cursor: UInt64 = 0
            for region in sorted {
                guard !region.name.isEmpty, region.size > 0,
                      region.offset >= cursor, region.offset % 16_384 == 0 else {
                    throw RepackError.configurationInvalid(
                        detail: "transformed audit region is invalid or overlapping")
                }
                if region.offset > cursor {
                    try verifyZeroes(
                        path: (rootDirectory as NSString).appendingPathComponent(relativePath),
                        offset: cursor, size: region.offset - cursor)
                }
                let end = region.offset.addingReportingOverflow(region.size)
                guard !end.overflow, end.partialValue <= expected.size else {
                    throw RepackError.configurationInvalid(
                        detail: "transformed audit region exceeds file")
                }
                cursor = end.partialValue
            }
            if cursor < expected.size {
                try verifyZeroes(
                    path: (rootDirectory as NSString).appendingPathComponent(relativePath),
                    offset: cursor, size: expected.size - cursor)
            }
        }
        guard Set(byFile.keys).isSubset(of: Set(files.keys)) else {
            throw RepackError.configurationInvalid(
                detail: "transformed audit region refers to undeclared file")
        }
    }

    private static func recursiveInventory(
        rootDirectory: String
    ) throws -> (files: Set<String>, directories: Set<String>) {
        var files = Set<String>()
        var directories = Set<String>()
        func visit(_ relativeDirectory: String) throws {
            let absolute = relativeDirectory.isEmpty ? rootDirectory
                : (rootDirectory as NSString).appendingPathComponent(relativeDirectory)
            for name in try FileManager.default.contentsOfDirectory(atPath: absolute).sorted() {
                guard name != ".", name != "..", !name.contains("/"), !name.contains("\0") else {
                    throw RepackError.configurationInvalid(
                        detail: "unsafe transformed audit entry")
                }
                let relative = relativeDirectory.isEmpty
                    ? name : relativeDirectory + "/" + name
                let path = (rootDirectory as NSString).appendingPathComponent(relative)
                switch try Posix.entryKind(path) {
                case .regular: files.insert(relative)
                case .directory:
                    directories.insert(relative)
                    try visit(relative)
                case .absent, .symlink, .other:
                    throw RepackError.configurationInvalid(
                        detail: "transformed audit encountered non-regular entry")
                }
            }
        }
        try visit("")
        return (files, directories)
    }

    private static func parentDirectories(_ path: String) -> [String] {
        var parts = path.split(separator: "/").map(String.init)
        var result: [String] = []
        while parts.count > 1 {
            _ = parts.removeLast()
            result.append(parts.joined(separator: "/"))
        }
        return result
    }

    private static func verifyZeroes(path: String, offset: UInt64, size: UInt64) throws {
        guard size > 0 else { return }
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        let count = Int(min(UInt64(WriterCore.tileBytes), size))
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: count, alignment: 16_384)
        defer { buffer.deallocate() }
        var position = offset
        var remaining = size
        while remaining > 0 {
            let amount = Int(min(UInt64(buffer.count), remaining))
            try Posix.preadAll(
                fd: fd, path: path, buf: buffer.baseAddress!,
                count: amount, offset: position)
            guard buffer.bindMemory(to: UInt8.self).prefix(amount)
                .allSatisfy({ $0 == 0 }) else {
                throw RepackError.configurationInvalid(
                    detail: "transformed audit padding is not zero-filled")
            }
            position += UInt64(amount)
            remaining -= UInt64(amount)
        }
    }
}
