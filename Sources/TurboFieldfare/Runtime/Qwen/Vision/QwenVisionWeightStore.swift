import Darwin
import Foundation
import Metal
import TurboFieldfareFormat

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
    let resident: ResidentBuffer
    let offsets: [String: Int]
    let diagnostic: QwenVisionMappedGroupDiagnostic

    init(
        group: QwenVisionWeightGroup,
        resident: ResidentBuffer,
        offsets: [String: Int],
        diagnostic: QwenVisionMappedGroupDiagnostic
    ) {
        self.group = group
        self.resident = resident
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
            group: group, resident: resident,
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
