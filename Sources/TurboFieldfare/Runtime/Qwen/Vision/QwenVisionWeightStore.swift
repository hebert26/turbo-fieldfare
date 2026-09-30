import Darwin
import Foundation
import Metal
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

enum QwenVisionWeightGroup: Hashable, Sendable, CustomStringConvertible {
    case patchAndPosition
    case block(Int)
    case merger

    var description: String {
        switch self {
        case .patchAndPosition: "patch+position"
        case .block(let index): "block\(index)"
        case .merger: "merger"
        }
    }
}

struct QwenVisionMappedGroupDiagnostic: Equatable, Sendable {
    let group: QwenVisionWeightGroup
    let residentBytes: UInt64
    let pageAlignedMappedBytes: UInt64
}

struct QwenVisionWeightStoreDiagnostics: Equatable, Sendable {
    let tensorCount: Int
    let rawTensorBytes: UInt64
    let payloadBytes: UInt64
    let groups: [QwenVisionMappedGroupDiagnostic]

    var maximumResidentBytes: UInt64 { groups.map(\.residentBytes).max() ?? 0 }
    var maximumPageAlignedMappedBytes: UInt64 {
        groups.map(\.pageAlignedMappedBytes).max() ?? 0
    }
}

final class QwenVisionMappedGroupLease: @unchecked Sendable {
    let group: QwenVisionWeightGroup
    private let resident: ResidentBuffer?
    let buffer: MTLBuffer
    let offsets: [String: Int]
    let diagnostic: QwenVisionMappedGroupDiagnostic

    init(
        group: QwenVisionWeightGroup,
        resident: ResidentBuffer? = nil,
        buffer: MTLBuffer,
        offsets: [String: Int],
        diagnostic: QwenVisionMappedGroupDiagnostic
    ) {
        self.group = group
        self.resident = resident
        self.buffer = buffer
        self.offsets = offsets
        self.diagnostic = diagnostic
    }

    func offset(of name: String) throws -> Int {
        guard let value = offsets[name] else {
            throw VisionPackError.invalidMetadata("group \(group) has no tensor \(name)")
        }
        return value
    }
}

final class QwenVisionWeightStore {
    let directoryURL: URL
    let manifest: GTurboVisionManifestV2
    let diagnostics: QwenVisionWeightStoreDiagnostics

    private let directory: GTurboModelDirectory
    private let weightsFD: Int32
    private let regions: [String: GTurboTensorRegionV2]
    private let groupRegions: [QwenVisionWeightGroup: [GTurboTensorRegionV2]]

    static func open(
        directoryURL: URL,
        compatibleTextManifestSHA256: String,
        config: QwenVisionConfig = .official
    ) throws -> QwenVisionWeightStore {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: directoryURL.standardizedFileURL.path,
            isDirectory: &isDirectory), isDirectory.boolValue else {
            throw VisionPackError.packNotFound(directoryURL.standardizedFileURL.path)
        }
        let directory: GTurboModelDirectory
        do { directory = try GTurboModelDirectory(rootURL: directoryURL) }
        catch { throw VisionPackError.invalidMetadata("\(error)") }

        let manifestData: Data
        do {
            manifestData = try directory.readMetadata(
                GTurboVisionFormatV1.manifestFile,
                maxBytes: GTurboVisionFormatV2.metadataMaxBytes)
        } catch {
            throw VisionPackError.invalidMetadata("\(error)")
        }
        let verified: GTurboVerifiedVisionManifestV2
        do { verified = try GTurboVisionManifestV2Codec.decode(manifestData) }
        catch { throw VisionPackError.invalidMetadata("\(error)") }
        let manifest = verified.manifest
        guard manifest.compatibleTextManifestSHA256.lowercased()
                == compatibleTextManifestSHA256.lowercased() else {
            throw VisionPackError.incompatibleTextArtifact
        }

        let manifestSHA = Sha256Verifier.hashData(manifestData)
        let receipt: VerifiedInstallReceipt
        do {
            receipt = try VerifiedInstallReceiptReader.load(directoryURL: directoryURL)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt, directoryURL: directoryURL, manifestSha256: manifestSHA)
        } catch {
            throw VisionPackError.invalidReceipt("\(error)")
        }
        guard receipt.sourceRepoID == manifest.modelID,
              receipt.sourceRevision == manifest.sourceRevision else {
            throw VisionPackError.invalidReceipt("source identity mismatch")
        }
        let expectedReceiptFiles = Set([
            GTurboVisionFormatV1.manifestFile,
            GTurboVisionFormatV2.weightsFile,
            GTurboVisionFormatV2.processorFile,
        ])
        guard Set(receipt.files.keys) == expectedReceiptFiles else {
            throw VisionPackError.invalidReceipt("vision receipt file set mismatch")
        }
        guard let manifestReceipt = receipt.files[GTurboVisionFormatV1.manifestFile],
              manifestReceipt.size == UInt64(manifestData.count),
              manifestReceipt.sha256.lowercased() == manifestSHA.lowercased() else {
            throw VisionPackError.invalidReceipt("manifest entry mismatch")
        }
        for path in [GTurboVisionFormatV2.weightsFile, GTurboVisionFormatV2.processorFile] {
            guard let file = manifest.files[path], let received = receipt.files[path],
                  file.size == received.size,
                  file.sha256.lowercased() == received.sha256.lowercased() else {
                throw VisionPackError.invalidReceipt("payload entry mismatch for \(path)")
            }
        }

        let expectedEntries = Set([
            GTurboVisionFormatV1.manifestFile,
            GTurboVisionFormatV2.weightsFile,
            GTurboVisionFormatV2.processorFile,
            VerifiedInstallReceiptReader.fileName,
        ])
        let entries: Set<String>
        do {
            entries = try directory.basenames().filter {
                !GTurboVisionFormatV1.isSidecarEntry($0)
            }
        } catch {
            throw VisionPackError.invalidMetadata("\(error)")
        }
        guard entries == expectedEntries else {
            throw VisionPackError.invalidDirectoryEntries(
                GTurboVisionFormatV1.entryDifference(
                    expected: expectedEntries, actual: entries))
        }
        for (path, file) in manifest.files {
            let actual: UInt64
            do { actual = try directory.fileSize(path) }
            catch { throw VisionPackError.invalidMetadata("\(error)") }
            guard actual == file.size else {
                throw VisionPackError.fileSizeMismatch(
                    file: path, expected: file.size, actual: actual)
            }
        }
        guard let processorEntry = manifest.files[GTurboVisionFormatV2.processorFile] else {
            throw VisionPackError.invalidMetadata("processor file is missing")
        }
        let processorFD: Int32
        do { processorFD = try directory.openFile(GTurboVisionFormatV2.processorFile) }
        catch { throw VisionPackError.invalidMetadata("\(error)") }
        defer { close(processorFD) }
        do {
            try Sha256Verifier.verifyFile(
                fileDescriptor: processorFD,
                named: GTurboVisionFormatV2.processorFile,
                expectedHex: processorEntry.sha256)
        } catch {
            throw VisionPackError.invalidMetadata("processor digest mismatch")
        }

        try config.validate()
        let contract = config.tensorContract
        guard manifest.tensorRegions.count == contract.count,
              contract.count == 333 else {
            throw VisionPackError.invalidMetadata(
                "vision tensor count \(manifest.tensorRegions.count) != 333")
        }
        let byName = Dictionary(
            uniqueKeysWithValues: manifest.tensorRegions.map { ($0.name, $0) })
        guard byName.count == contract.count else {
            throw VisionPackError.invalidMetadata("duplicate vision tensor")
        }
        for expected in contract {
            guard let region = byName[expected.name],
                  region.shape == expected.shape,
                  region.storage == expected.storage,
                  region.quantizationCategory == nil else {
                throw VisionPackError.invalidMetadata(
                    "vision tensor contract mismatch for \(expected.name)")
            }
            let elements = try product(region.shape, name: region.name)
            let (expectedBytes, overflow) = elements.multipliedReportingOverflow(by: 2)
            guard !overflow, region.size == expectedBytes else {
                throw VisionPackError.invalidMetadata(
                    "BF16 byte size mismatch for \(region.name)")
            }
        }

        var grouped: [QwenVisionWeightGroup: [GTurboTensorRegionV2]] = [:]
        for region in manifest.tensorRegions {
            let group = try group(for: region.name, depth: config.depth)
            grouped[group, default: []].append(region)
        }
        let expectedGroups = [QwenVisionWeightGroup.patchAndPosition]
            + (0..<config.depth).map(QwenVisionWeightGroup.block)
            + [.merger]
        guard Set(grouped.keys) == Set(expectedGroups) else {
            throw VisionPackError.invalidMetadata("vision execution groups are incomplete")
        }
        let pageSize = UInt64(getpagesize())
        let groupDiagnostics = try expectedGroups.map { group -> QwenVisionMappedGroupDiagnostic in
            guard let values = grouped[group] else {
                throw VisionPackError.invalidMetadata("missing group \(group)")
            }
            return try diagnostic(group: group, regions: values, pageSize: pageSize)
        }
        let rawBytes = try contract.reduce(UInt64(0)) { partial, tensor in
            let elements = try product(tensor.shape, name: tensor.name)
            let (bytes, overflow) = elements.multipliedReportingOverflow(by: 2)
            let (sum, sumOverflow) = partial.addingReportingOverflow(bytes)
            guard !overflow, !sumOverflow else {
                throw VisionPackError.invalidMetadata("raw tensor byte overflow")
            }
            return sum
        }
        guard rawBytes == 893_142_496,
              let payloadBytes = manifest.files[GTurboVisionFormatV2.weightsFile]?.size else {
            throw VisionPackError.invalidMetadata("official BF16 payload accounting mismatch")
        }

        let weightsFD: Int32
        do { weightsFD = try directory.openFile(GTurboVisionFormatV2.weightsFile) }
        catch { throw VisionPackError.invalidMetadata("\(error)") }
        do {
            guard let weightsEntry = manifest.files[GTurboVisionFormatV2.weightsFile] else {
                close(weightsFD)
                throw VisionPackError.invalidMetadata("weights file is missing")
            }
            var status = stat()
            guard fstat(weightsFD, &status) == 0,
                  (status.st_mode & S_IFMT) == S_IFREG,
                  status.st_size >= 0,
                  UInt64(status.st_size) == weightsEntry.size else {
                close(weightsFD)
                throw VisionPackError.fileSizeMismatch(
                    file: GTurboVisionFormatV2.weightsFile,
                    expected: weightsEntry.size,
                    actual: status.st_size >= 0 ? UInt64(status.st_size) : 0)
            }
            try Sha256Verifier.verifyFile(
                fileDescriptor: weightsFD,
                named: GTurboVisionFormatV2.weightsFile,
                expectedHex: weightsEntry.sha256)
        } catch let error as VisionPackError {
            throw error
        } catch {
            close(weightsFD)
            throw VisionPackError.invalidMetadata("weights digest mismatch: \(error)")
        }
        return QwenVisionWeightStore(
            directoryURL: directoryURL.standardizedFileURL,
            directory: directory, manifest: manifest, weightsFD: weightsFD,
            regions: byName, groupRegions: grouped,
            diagnostics: QwenVisionWeightStoreDiagnostics(
                tensorCount: contract.count, rawTensorBytes: rawBytes,
                payloadBytes: payloadBytes, groups: groupDiagnostics))
    }

    private init(
        directoryURL: URL,
        directory: GTurboModelDirectory,
        manifest: GTurboVisionManifestV2,
        weightsFD: Int32,
        regions: [String: GTurboTensorRegionV2],
        groupRegions: [QwenVisionWeightGroup: [GTurboTensorRegionV2]],
        diagnostics: QwenVisionWeightStoreDiagnostics
    ) {
        self.directoryURL = directoryURL
        self.directory = directory
        self.manifest = manifest
        self.weightsFD = weightsFD
        self.regions = regions
        self.groupRegions = groupRegions
        self.diagnostics = diagnostics
    }

    deinit { close(weightsFD) }

    func mapGroup(
        _ group: QwenVisionWeightGroup,
        device: MTLDevice
    ) throws -> QwenVisionMappedGroupLease {
        guard let values = groupRegions[group],
              let diagnostic = diagnostics.groups.first(where: { $0.group == group }),
              let start = values.map(\.offset).min() else {
            throw VisionPackError.invalidMetadata("missing group \(group)")
        }
        let maximum = UInt64(device.maxBufferLength)
        guard diagnostic.residentBytes <= maximum,
              diagnostic.pageAlignedMappedBytes <= maximum else {
            throw VisionPackError.regionExceedsDevice(
                name: group.description,
                bytes: max(diagnostic.residentBytes, diagnostic.pageAlignedMappedBytes),
                maximum: maximum)
        }
        let resident = try ResidentBuffer(
            fileURL: directoryURL.appendingPathComponent(GTurboVisionFormatV2.weightsFile),
            fileOffset: start,
            residentSize: diagnostic.residentBytes,
            device: device,
            fileDescriptor: weightsFD)
        let offsets = Dictionary(uniqueKeysWithValues: values.map {
            ($0.name, Int($0.offset - start))
        })
        return QwenVisionMappedGroupLease(
            group: group, resident: resident, buffer: resident.buffer,
            offsets: offsets, diagnostic: diagnostic)
    }

    func region(named name: String) -> GTurboTensorRegionV2? { regions[name] }

    private static func group(
        for name: String, depth: Int
    ) throws -> QwenVisionWeightGroup {
        if name.hasPrefix("model.visual.patch_embed.")
            || name == "model.visual.pos_embed.weight" {
            return .patchAndPosition
        }
        if name.hasPrefix("model.visual.merger.") { return .merger }
        let prefix = "model.visual.blocks."
        guard name.hasPrefix(prefix) else {
            throw VisionPackError.invalidMetadata("unclassified tensor \(name)")
        }
        let suffix = name.dropFirst(prefix.count)
        guard let dot = suffix.firstIndex(of: "."),
              let index = Int(suffix[..<dot]), (0..<depth).contains(index) else {
            throw VisionPackError.invalidMetadata("invalid block tensor \(name)")
        }
        return .block(index)
    }

    private static func diagnostic(
        group: QwenVisionWeightGroup,
        regions: [GTurboTensorRegionV2],
        pageSize: UInt64
    ) throws -> QwenVisionMappedGroupDiagnostic {
        guard let start = regions.map(\.offset).min() else {
            throw VisionPackError.invalidMetadata("empty group \(group)")
        }
        var end: UInt64 = 0
        for region in regions {
            let (candidate, overflow) = region.offset.addingReportingOverflow(region.size)
            guard !overflow else {
                throw VisionPackError.invalidMetadata("group extent overflow")
            }
            end = max(end, candidate)
        }
        guard end > start else {
            throw VisionPackError.invalidMetadata("empty group extent")
        }
        let resident = end - start
        let shift = start % pageSize
        let (mapped, overflow) = resident.addingReportingOverflow(shift)
        guard !overflow else {
            throw VisionPackError.invalidMetadata("mapped group extent overflow")
        }
        return QwenVisionMappedGroupDiagnostic(
            group: group, residentBytes: resident,
            pageAlignedMappedBytes: mapped)
    }

    private static func product(_ shape: [UInt64], name: String) throws -> UInt64 {
        try shape.reduce(UInt64(1)) { partial, dimension in
            let (result, overflow) = partial.multipliedReportingOverflow(by: dimension)
            guard !overflow else {
                throw VisionPackError.invalidMetadata("shape overflow for \(name)")
            }
            return result
        }
    }
}

/// A source companion has exactly one metadata file; its BF16 bytes are read
/// only from the already retained protected source handle, one requested group
/// at a time. No vision payload is written alongside the text registration.
/// Immutable admission and retained read-only descriptors. Calls allocate
/// independent group buffers; revalidation uses fresh directory cursors and
/// positional reads, so an actor-to-actor commit check can share this owner.
final class QwenOfficialSourceVisionWeightStore: @unchecked Sendable {
    let descriptor: OfficialSourceVisionDescriptor
    let config: QwenVisionConfig
    private let model: QwenOfficialSourceModel
    private let groups: [QwenVisionWeightGroup: [OfficialSourceVisionDescriptor.Tensor]]
    private let directory: GTurboModelDirectory
    private let directoryFD: Int32
    private let manifestFD: Int32
    private let directoryIdentity: VisionFileIdentity
    private let manifestIdentity: VisionFileIdentity
    private let manifestBytes: Data

    private struct VisionFileIdentity: Equatable {
        let device: dev_t
        let inode: ino_t
        let mode: mode_t
        let links: nlink_t
        let size: off_t
        let mtime: timespec
        let ctime: timespec

        init(_ info: stat) {
            device = info.st_dev
            inode = info.st_ino
            mode = info.st_mode
            links = info.st_nlink
            size = info.st_size
            mtime = info.st_mtimespec
            ctime = info.st_ctimespec
        }

        static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.device == rhs.device && lhs.inode == rhs.inode
                && lhs.mode == rhs.mode && lhs.links == rhs.links
                && lhs.size == rhs.size
                && lhs.mtime.tv_sec == rhs.mtime.tv_sec
                && lhs.mtime.tv_nsec == rhs.mtime.tv_nsec
                && lhs.ctime.tv_sec == rhs.ctime.tv_sec
                && lhs.ctime.tv_nsec == rhs.ctime.tv_nsec
        }
    }

    static func open(directoryURL: URL, model: QwenOfficialSourceModel,
                     config: QwenVisionConfig = .official) throws -> Self {
        try config.validate()
        guard config.matchesOfficialWeightLayout ||
                (model.sourceIdentity == nil && config.allowsFixtureGeometry) else {
            throw VisionPackError.invalidMetadata("unsupported source vision geometry")
        }
        let directory: GTurboModelDirectory
        do { directory = try GTurboModelDirectory(rootURL: directoryURL) }
        catch { throw VisionPackError.packNotFound(directoryURL.path) }
        guard try directory.basenames() == Set([OfficialSourceVisionDescriptor.filename]) else {
            throw VisionPackError.invalidMetadata("source companion must contain metadata only")
        }
        let bytes = try directory.readMetadata(
            OfficialSourceVisionDescriptor.filename,
            maxBytes: UInt64(OfficialSourceVisionDescriptor.maximumBytes))
        let descriptor = try OfficialSourceVisionDescriptor.decodeStrict(bytes)
        try model.revalidateSource()
        guard descriptor.textContentSHA256 == model.sourceContentSHA256,
              descriptor.processorConfigSHA256 == (try model.visionProcessorSHA256()),
              descriptor.processorProfile == GTurboQwenVisionProcessorProfileV2(
                  processorClass: "Qwen3VLProcessor",
                  imageProcessorType: "Qwen2VLImageProcessorFast",
                  patchSize: config.patchSize,
                  temporalPatchSize: config.temporalPatchSize,
                  spatialMergeSize: config.spatialMergeSize) else {
            throw VisionPackError.incompatibleTextArtifact
        }
        let contract = config.tensorContract.sorted { $0.name < $1.name }
        guard descriptor.tensors.count == contract.count,
              zip(descriptor.tensors, contract).allSatisfy({ tensor, expected in
                  tensor.name == expected.name && tensor.shape == expected.shape
                    && expected.storage == .bf16 && model.hasSourceTensor(tensor.name)
              }) else {
            throw VisionPackError.invalidMetadata("source vision tensor inventory mismatch")
        }
        var groups: [QwenVisionWeightGroup: [OfficialSourceVisionDescriptor.Tensor]] = [:]
        for tensor in descriptor.tensors {
            let name = tensor.name
            let group: QwenVisionWeightGroup
            if name.hasPrefix("model.visual.patch_embed.") || name == "model.visual.pos_embed.weight" {
                group = .patchAndPosition
            } else if name.hasPrefix("model.visual.merger.") {
                group = .merger
            } else {
                let prefix = "model.visual.blocks."
                guard name.hasPrefix(prefix),
                      let number = name.dropFirst(prefix.count).split(separator: ".").first,
                      let index = Int(number), (0..<config.depth).contains(index) else {
                    throw VisionPackError.invalidMetadata("unclassified source vision tensor")
                }
                group = .block(index)
            }
            groups[group, default: []].append(tensor)
        }
        guard groups.count == config.depth + 2 else {
            throw VisionPackError.invalidMetadata("incomplete source vision groups")
        }
        let directoryFD = Darwin.open(directoryURL.standardizedFileURL.path,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard directoryFD >= 0 else {
            throw VisionPackError.invalidMetadata("source companion directory unavailable")
        }
        let manifestFD = openat(directoryFD, OfficialSourceVisionDescriptor.filename,
                                O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard manifestFD >= 0 else {
            close(directoryFD)
            throw VisionPackError.invalidMetadata("source companion manifest unavailable")
        }
        var directoryInfo = stat(), manifestInfo = stat()
        guard fstat(directoryFD, &directoryInfo) == 0,
              fstat(manifestFD, &manifestInfo) == 0,
              (directoryInfo.st_mode & S_IFMT) == S_IFDIR,
              (manifestInfo.st_mode & S_IFMT) == S_IFREG,
              manifestInfo.st_nlink == 1,
              manifestInfo.st_size >= 0,
              manifestInfo.st_size <= OfficialSourceVisionDescriptor.maximumBytes else {
            close(manifestFD)
            close(directoryFD)
            throw VisionPackError.invalidMetadata("source companion identity invalid")
        }
        let store = Self(descriptor: descriptor, config: config, model: model,
                         groups: groups, directory: directory,
                         directoryFD: directoryFD, manifestFD: manifestFD,
                         directoryIdentity: VisionFileIdentity(directoryInfo),
                         manifestIdentity: VisionFileIdentity(manifestInfo), manifestBytes: bytes)
        try store.revalidate()
        return store
    }

    private init(descriptor: OfficialSourceVisionDescriptor, config: QwenVisionConfig,
                 model: QwenOfficialSourceModel,
                 groups: [QwenVisionWeightGroup: [OfficialSourceVisionDescriptor.Tensor]],
                 directory: GTurboModelDirectory, directoryFD: Int32, manifestFD: Int32,
                 directoryIdentity: VisionFileIdentity,
                 manifestIdentity: VisionFileIdentity, manifestBytes: Data) {
        self.descriptor = descriptor
        self.config = config
        self.model = model
        self.groups = groups
        self.directory = directory
        self.directoryFD = directoryFD
        self.manifestFD = manifestFD
        self.directoryIdentity = directoryIdentity
        self.manifestIdentity = manifestIdentity
        self.manifestBytes = manifestBytes
    }

    deinit { close(manifestFD); close(directoryFD) }

    /// Check both the retained FD and current literal named entries. The
    /// metadata cap makes the byte comparison bounded and prevents a same-size
    /// in-place rewrite from keeping an old companion admitted.
    func revalidate() throws {
        try model.revalidateSource()
        var heldDirectory = stat(), heldManifest = stat()
        guard fstat(directoryFD, &heldDirectory) == 0,
              fstat(manifestFD, &heldManifest) == 0,
              VisionFileIdentity(heldDirectory) == directoryIdentity,
              VisionFileIdentity(heldManifest) == manifestIdentity else {
            throw VisionPackError.invalidMetadata("source companion changed")
        }
        let currentDirectory = Darwin.open(directory.rootURL.path,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard currentDirectory >= 0 else {
            throw VisionPackError.invalidMetadata("source companion directory replaced")
        }
        defer { close(currentDirectory) }
        var namedDirectory = stat()
        guard fstat(currentDirectory, &namedDirectory) == 0,
              VisionFileIdentity(namedDirectory) == directoryIdentity,
              try GTurboModelDirectory(rootURL: directory.rootURL).basenames()
                  == Set([OfficialSourceVisionDescriptor.filename]) else {
            throw VisionPackError.invalidMetadata("source companion directory changed")
        }
        let currentManifest = openat(currentDirectory, OfficialSourceVisionDescriptor.filename,
                                     O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard currentManifest >= 0 else {
            throw VisionPackError.invalidMetadata("source companion manifest replaced")
        }
        defer { close(currentManifest) }
        var namedManifest = stat()
        guard fstat(currentManifest, &namedManifest) == 0,
              VisionFileIdentity(namedManifest) == manifestIdentity,
              manifestBytes.count == Int(manifestIdentity.size) else {
            throw VisionPackError.invalidMetadata("source companion manifest changed")
        }
        let current = try directory.readMetadata(
            fileDescriptor: currentManifest, relativePath: OfficialSourceVisionDescriptor.filename,
            maxBytes: UInt64(OfficialSourceVisionDescriptor.maximumBytes))
        guard current == manifestBytes,
              fstat(manifestFD, &heldManifest) == 0,
              VisionFileIdentity(heldManifest) == manifestIdentity,
              fstat(currentDirectory, &namedDirectory) == 0,
              VisionFileIdentity(namedDirectory) == directoryIdentity else {
            throw VisionPackError.invalidMetadata("source companion metadata changed")
        }
        try model.revalidateSource()
    }

    /// O(group BF16 bytes), with a single shared Metal allocation and at most
    /// one 8 MiB protected read tile. The returned lease owns the buffer until
    /// the caller has observed command completion, including cancellation.
    func mapGroup(_ group: QwenVisionWeightGroup, device: MTLDevice)
        throws -> QwenVisionMappedGroupLease {
        guard device === model.context.device, let tensors = groups[group], !tensors.isEmpty else {
            throw VisionPackError.invalidMetadata("source group/device unavailable: \(group)")
        }
        try revalidate()
        let alignment = 256
        var offsets: [String: Int] = [:]
        var count = 0
        for tensor in tensors {
            guard count <= Int.max - (alignment - 1) else {
                throw VisionPackError.invalidMetadata("vision group offset overflow")
            }
            count = (count + alignment - 1) & ~(alignment - 1)
            let elements = tensor.shape.reduce(UInt64(1)) { partial, dimension in
                let (next, overflow) = partial.multipliedReportingOverflow(by: dimension)
                return overflow ? UInt64.max : next
            }
            guard elements <= UInt64((Int.max - count) / 2) else {
                throw VisionPackError.invalidMetadata("vision group size overflow")
            }
            offsets[tensor.name] = count
            count += Int(elements) * 2
        }
        let maximum = min(UInt64(device.maxBufferLength), 200 * 1024 * 1024)
        guard count > 0, UInt64(count) <= maximum else {
            throw VisionPackError.regionExceedsDevice(
                name: group.description, bytes: UInt64(count), maximum: maximum)
        }
        guard let buffer = device.makeBuffer(length: count, options: .storageModeShared) else {
            throw VisionPackError.invalidMetadata("source vision group allocation failed")
        }
        buffer.label = "qwen.source.vision.\(group)"
        // On a failed read the partially filled buffer is discarded, never
        // submitted to Metal. Each token is admitted from the retained header.
        for tensor in tensors {
            try Task.checkCancellation()
            let token = try model.admitVisionTensor(tensor.name)
            let expected = tensor.shape.reduce(UInt64(1), *) * 2
            guard token.shape == tensor.shape, token.admittedByteCount == expected,
                  let offset = offsets[tensor.name] else {
                throw VisionPackError.invalidMetadata("source vision header mismatch: \(tensor.name)")
            }
            var position: UInt64 = 0
            while position < expected {
                let length = min(expected - position, OfficialSourceHandle.maximumTensorReadBytes)
                let start = offset + Int(position)
                try model.source.preadTensorRange(
                    token, byteOffset: position, byteCount: length,
                    expectedByteCount: length,
                    into: UnsafeMutableRawBufferPointer(
                        start: buffer.contents().advanced(by: start), count: Int(length)))
                position += length
            }
        }
        try revalidate()
        return QwenVisionMappedGroupLease(
            group: group, buffer: buffer, offsets: offsets,
            diagnostic: QwenVisionMappedGroupDiagnostic(
                group: group, residentBytes: UInt64(count),
                pageAlignedMappedBytes: UInt64(count)))
    }
}
