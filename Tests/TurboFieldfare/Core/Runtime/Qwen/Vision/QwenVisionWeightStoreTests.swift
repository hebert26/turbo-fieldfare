import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct QwenVisionWeightStoreTests {
    @Test func officialContractHas333BF16TensorsAndExactRawBytes() throws {
        let config = QwenVisionConfig.official
        #expect(config.tensorContract.count == 333)
        #expect(config.tensorContract.allSatisfy { $0.storage == .bf16 })
        #expect(config.rawTensorElementCount == 446_571_248)
        #expect(config.rawTensorElementCount * 2 == 893_142_496)
        #expect(config.tensorContract.first?.name
            == "model.visual.patch_embed.proj.weight")
        #expect(config.tensorContract.last?.name
            == "model.visual.merger.linear_fc2.bias")
    }

    @Test func missingCompanionFailsBeforeAnyMapping() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-qwen-vision-\(UUID().uuidString)")
        #expect(throws: VisionPackError.packNotFound(url.standardizedFileURL.path)) {
            try QwenVisionWeightStore.open(
                directoryURL: url,
                compatibleTextManifestSHA256: String(repeating: "a", count: 64))
        }
    }

    @Test func syntheticVerifiedCompanionMapsExactlyOneOrderedGroupAtATime() throws {
        let companion = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: companion) }
        let store = try QwenVisionWeightStore.open(
            directoryURL: companion,
            compatibleTextManifestSHA256: String(repeating: "a", count: 64))
        let diagnostics = store.diagnostics
        #expect(diagnostics.tensorCount == 333)
        #expect(diagnostics.rawTensorBytes == 893_142_496)
        #expect(diagnostics.groups.count == 29)
        #expect(diagnostics.groups.first?.group == .patchAndPosition)
        #expect(diagnostics.groups.dropFirst().dropLast().map(\.group)
            == (0..<27).map(QwenVisionWeightGroup.block))
        #expect(diagnostics.groups.last?.group == .merger)
        #expect(diagnostics.groups.allSatisfy {
            $0.pageAlignedMappedBytes >= $0.residentBytes
        })

        let device = try #require(MTLCreateSystemDefaultDevice())
        let orderedGroups = [QwenVisionWeightGroup.patchAndPosition]
            + (0..<27).map(QwenVisionWeightGroup.block)
            + [.merger]
        for group in orderedGroups {
            let lease = try store.mapGroup(group, device: device)
            #expect(lease.group == group)
            #expect(lease.diagnostic.pageAlignedMappedBytes
                <= UInt64(device.maxBufferLength))
            #expect(lease.diagnostic.residentBytes
                <= UInt64(device.maxBufferLength))
            let expected = try #require(
                diagnostics.groups.first(where: { $0.group == group }))
            #expect(lease.diagnostic.pageAlignedMappedBytes
                == expected.pageAlignedMappedBytes)
        }
    }

    @Test func wrongTextDigestIsRejectedBeforeMapping() throws {
        let companion = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: companion) }
        #expect(throws: VisionPackError.incompatibleTextArtifact) {
            try QwenVisionWeightStore.open(
                directoryURL: companion,
                compatibleTextManifestSHA256: String(repeating: "b", count: 64))
        }
    }

    @Test func wrongFamilyRevisionAndProcessorProfileAreRejected() throws {
        let mutations: [(String, (inout [String: Any]) -> Void)] = [
            ("family", { $0["family"] = "gemma4" }),
            ("revision", { $0["sourceRevision"] = "not-the-pinned-revision" }),
            ("profile", {
                guard var profile = $0["processorProfile"] as? [String: Any] else { return }
                profile["patchSize"] = 2
                $0["processorProfile"] = profile
            }),
        ]
        for (label, mutation) in mutations {
            let companion = try makeSyntheticCompanion()
            defer { try? FileManager.default.removeItem(at: companion) }
            try rewriteManifest(companion, mutation: mutation)
            expectMetadataFailure(companion, label: label)
        }
    }

    @Test func missingBadSchemaAndFileBoundReceiptsAreRejected() throws {
        let missing = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: missing) }
        try FileManager.default.removeItem(
            at: missing.appendingPathComponent(VerifiedInstallReceiptReader.fileName))
        expectReceiptFailure(missing, label: "missing receipt")

        let badSchema = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: badSchema) }
        try rewriteReceipt(badSchema) { $0["schemaVersion"] = 2 }
        expectReceiptFailure(badSchema, label: "schema")

        let missingBinding = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: missingBinding) }
        try rewriteReceipt(missingBinding) { receipt in
            guard var files = receipt["files"] as? [String: Any] else { return }
            files.removeValue(forKey: GTurboVisionFormatV2.weightsFile)
            receipt["files"] = files
        }
        expectReceiptFailure(missingBinding, label: "file binding")
    }

    @Test func manifestNamesShapesCorruptionAndUnsafeEntriesAreRejected() throws {
        let unsupported = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: unsupported) }
        try rewriteManifest(unsupported) { manifest in
            guard var regions = manifest["tensorRegions"] as? [[String: Any]],
                  !regions.isEmpty else { return }
            regions[0]["name"] = "model.visual.unsupported.weight"
            manifest["tensorRegions"] = regions
        }
        expectMetadataFailure(unsupported, label: "unsupported tensor name")

        let wrongShape = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: wrongShape) }
        try rewriteManifest(wrongShape) { manifest in
            guard var regions = manifest["tensorRegions"] as? [[String: Any]],
                  !regions.isEmpty else { return }
            regions[0]["shape"] = [1]
            manifest["tensorRegions"] = regions
        }
        expectMetadataFailure(wrongShape, label: "unsupported tensor shape")

        let corrupt = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: corrupt) }
        let weights = corrupt.appendingPathComponent(GTurboVisionFormatV2.weightsFile)
        let handle = try FileHandle(forWritingTo: weights)
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data([1]))
        try handle.close()
        expectMetadataFailure(corrupt, label: "corrupt weight payload")

        let unsafe = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: unsafe) }
        let extra = unsafe.appendingPathComponent("unexpected.bin")
        guard FileManager.default.createFile(atPath: extra.path, contents: Data([0])) else {
            throw VisionPackError.invalidMetadata("test fixture file creation failed")
        }
        let failure = try #require(openFailure(unsafe))
        guard case .invalidDirectoryEntries = failure else {
            Issue.record("unsafe entry returned unexpected error: \(failure)")
            return
        }
    }

    @Test func missingTensorPathIsRejectedAfterGroupAdmission() throws {
        let companion = try makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: companion) }
        let context = try MetalContext()
        let store = try QwenVisionWeightStore.open(
            directoryURL: companion,
            compatibleTextManifestSHA256: String(repeating: "a", count: 64))
        let lease = try store.mapGroup(.patchAndPosition, device: context.device)
        #expect(throws: VisionPackError.invalidMetadata(
            "group patch+position has no tensor model.visual.missing")) {
            try lease.offset(of: "model.visual.missing")
        }
    }

    @Test func configContractRejectsUnsupportedGeometry() throws {
        let invalid = QwenVisionConfig(
            temporalPatchSize: 1,
            spatialMergeSize: 2)
        #expect(throws: QwenVisionError.invalidConfiguration) {
            try invalid.validate()
        }
        let invalidMerge = QwenVisionConfig(
            temporalPatchSize: 2,
            spatialMergeSize: 3,
            maximumPatchRows: 2_520,
            maximumMergedRows: 280)
        #expect(throws: QwenVisionError.invalidConfiguration) {
            try invalidMerge.validate()
        }
    }

    private func expectMetadataFailure(_ root: URL, label: String) {
        guard let failure = openFailure(root) else {
            Issue.record("\(label) unexpectedly opened")
            return
        }
        guard case .invalidMetadata = failure else {
            Issue.record("\(label) returned unexpected error: \(failure)")
            return
        }
    }

    private func expectReceiptFailure(_ root: URL, label: String) {
        guard let failure = openFailure(root) else {
            Issue.record("\(label) unexpectedly opened")
            return
        }
        guard case .invalidReceipt = failure else {
            Issue.record("\(label) returned unexpected error: \(failure)")
            return
        }
    }

    private func openFailure(_ root: URL) -> VisionPackError? {
        do {
            _ = try QwenVisionWeightStore.open(
                directoryURL: root,
                compatibleTextManifestSHA256: String(repeating: "a", count: 64))
            return nil
        } catch let error as VisionPackError {
            return error
        } catch {
            Issue.record("unexpected non-vision error: \(error)")
            return nil
        }
    }

    private func rewriteManifest(
        _ root: URL,
        mutation: (inout [String: Any]) throws -> Void = { _ in }
    ) throws {
        let manifestURL = root.appendingPathComponent(GTurboVisionFormatV1.manifestFile)
        var manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any] ?? [:]
        try mutation(&manifest)
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try data.write(to: manifestURL)
        let receiptURL = root.appendingPathComponent(VerifiedInstallReceiptReader.fileName)
        var receipt = try JSONSerialization.jsonObject(
            with: Data(contentsOf: receiptURL)) as? [String: Any] ?? [:]
        let digest = Sha256Verifier.hashData(data)
        receipt["manifestSha256"] = digest
        receipt["sourceRepoID"] = manifest["modelID"]
        receipt["sourceRevision"] = manifest["sourceRevision"]
        if var files = receipt["files"] as? [String: Any],
           var manifestEntry = files[GTurboVisionFormatV1.manifestFile] as? [String: Any] {
            manifestEntry["size"] = data.count
            manifestEntry["sha256"] = digest
            files[GTurboVisionFormatV1.manifestFile] = manifestEntry
            receipt["files"] = files
        }
        let receiptData = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        try receiptData.write(to: receiptURL)
    }

    private func rewriteReceipt(
        _ root: URL,
        mutation: (inout [String: Any]) throws -> Void
    ) throws {
        let receiptURL = root.appendingPathComponent(VerifiedInstallReceiptReader.fileName)
        var receipt = try JSONSerialization.jsonObject(
            with: Data(contentsOf: receiptURL)) as? [String: Any] ?? [:]
        try mutation(&receipt)
        let data = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        try data.write(to: receiptURL)
    }

    func makeSyntheticCompanion() throws -> URL {
        let phase16 = URL(fileURLWithPath: FileManager.default.currentDirectoryPath,
                          isDirectory: true)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/evidence/phase-16", isDirectory: true)
        let root = phase16.appendingPathComponent(UUID().uuidString, isDirectory: true)
        var keepRoot = false
        defer {
            if !keepRoot { try? FileManager.default.removeItem(at: root) }
        }
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        let processor = Data(
            """
            {
                "size": {
                    "longest_edge": 16777216,
                    "shortest_edge": 65536
                },
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [
                    0.5,
                    0.5,
                    0.5
                ],
                "image_std": [
                    0.5,
                    0.5,
                    0.5
                ],
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast"
            }
            """.utf8)
        let processorDigest = Sha256Verifier.hashData(processor)
        #expect(processorDigest
            == GTurboFormatV2.qwenSidecarSHA256["preprocessor_config.json"])
        let processorURL = root.appendingPathComponent(
            GTurboVisionFormatV2.processorFile)
        try processor.write(to: processorURL)

        let config = QwenVisionConfig.official
        var offset: UInt64 = 0
        var regions: [GTurboTensorRegionV2] = []
        for tensor in config.tensorContract {
            let alignment = UInt64(GTurboVisionFormatV2.alignmentBytes)
            let remainder = offset % alignment
            if remainder != 0 { offset += alignment - remainder }
            let elements = tensor.shape.reduce(UInt64(1), *)
            let size = elements * 2
            regions.append(.init(
                name: tensor.name,
                file: GTurboVisionFormatV2.weightsFile,
                offset: offset,
                size: size,
                shape: tensor.shape,
                storage: .bf16))
            offset += size
        }
        let weightsURL = root.appendingPathComponent(GTurboVisionFormatV2.weightsFile)
        FileManager.default.createFile(atPath: weightsURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: weightsURL)
        try handle.seek(toOffset: offset - 1)
        try handle.write(contentsOf: Data([0]))
        try handle.close()
        let weightsDigest = try Sha256Verifier.hashFile(at: weightsURL)
        let files: [String: GTurboManifestFileV1] = [
            GTurboVisionFormatV2.weightsFile: .init(
                size: offset, sha256: weightsDigest),
            GTurboVisionFormatV2.processorFile: .init(
                size: UInt64(processor.count), sha256: processorDigest),
        ]
        let manifest = GTurboVisionManifestV2(
            family: .qwen3_6,
            modelID: GTurboFormatV2.qwenRepository,
            sourceRevision: GTurboFormatV2.qwenRevision,
            processorProfile: .init(
                processorClass: "Qwen3VLProcessor",
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2),
            processorConfigSHA256: processorDigest,
            compatibleTextManifestSHA256: String(repeating: "a", count: 64),
            visionPayloadSHA256: weightsDigest,
            supportsStillImages: true,
            supportsVideo: false,
            files: files,
            tensorRegions: regions)
        let manifestData = try GTurboVisionManifestV2Codec.encode(manifest)
        let manifestURL = root.appendingPathComponent(GTurboVisionFormatV1.manifestFile)
        try manifestData.write(to: manifestURL)
        let manifestDigest = Sha256Verifier.hashData(manifestData)
        let receipt = VerifiedInstallReceipt(
            manifestSha256: manifestDigest,
            modelDirectoryPath: root.standardizedFileURL.path,
            sourceRepoID: GTurboFormatV2.qwenRepository,
            sourceRevision: GTurboFormatV2.qwenRevision,
            verificationTimestamp: "fixture",
            toolVersion: "fixture",
            files: [
                GTurboVisionFormatV1.manifestFile: .init(
                    size: UInt64(manifestData.count), sha256: manifestDigest),
                GTurboVisionFormatV2.weightsFile: .init(
                    size: offset, sha256: weightsDigest),
                GTurboVisionFormatV2.processorFile: .init(
                    size: UInt64(processor.count), sha256: processorDigest),
            ])
        let receiptURL = root.appendingPathComponent(
            VerifiedInstallReceiptReader.fileName)
        try JSONEncoder().encode(receipt).write(to: receiptURL)
        keepRoot = true
        return root
    }
}
