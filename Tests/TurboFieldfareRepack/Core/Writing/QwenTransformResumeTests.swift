import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite struct QwenTransformResumeTests {
    @Test func everySyntheticInterruptionResumesToCleanDirectoryBytes() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try makeSource(in: root)
        let clean = root.appendingPathComponent("clean.gturbo").path
        let cleanResult = try SyntheticTransformedPackWriter.write(
            input: .init(source: source, outputDirectory: clean))
        let expected = try directoryBytes(cleanResult.outputDirectory)

        for stopAfterGroup in 1...4 {
            let output = root.appendingPathComponent("resume-\(stopAfterGroup).gturbo").path
            #expect(throws: CancellationError.self) {
                _ = try SyntheticTransformedPackWriter.write(
                    input: .init(source: source, outputDirectory: output),
                    afterGroup: { count in
                        if count == stopAfterGroup { throw CancellationError() }
                    })
            }
            let resumed = try SyntheticTransformedPackWriter.write(
                input: .init(source: source, outputDirectory: output, resume: true))
            #expect(resumed.structuralFixtureOnly)
            #expect(resumed.completedGroupCount == 4)
            #expect(try directoryBytes(resumed.outputDirectory) == expected)
        }
    }

    @Test func changedPayloadRefusesResumeBeforeMutatingOwnedPartial() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try makeSource(in: root)
        let output = root.appendingPathComponent("changed.gturbo").path
        #expect(throws: CancellationError.self) {
            _ = try SyntheticTransformedPackWriter.write(
                input: .init(source: source, outputDirectory: output),
                afterGroup: { _ in throw CancellationError() })
        }
        let paths = try RemoteInstallPaths(outputDirectory: output)
        let weights = (paths.partialDirectory as NSString).appendingPathComponent("model_weights.bin")
        let before = try Data(contentsOf: URL(fileURLWithPath: weights))
        var changed = try Data(contentsOf: URL(fileURLWithPath: source.shardPath))
        changed[0] = 0x80
        try changed.write(to: URL(fileURLWithPath: source.shardPath))

        #expect(throws: RepackError.self) {
            _ = try SyntheticTransformedPackWriter.write(
                input: .init(source: source, outputDirectory: output, resume: true))
        }
        #expect(try Data(contentsOf: URL(fileURLWithPath: weights)) == before)
    }

    @Test func publishesCodecValidNonQwenFixtureAndAuditRejectsCorruption() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try makeSource(in: root)
        let output = root.appendingPathComponent("published.gturbo").path
        let result = try SyntheticTransformedPackWriter.write(
            input: .init(source: source, outputDirectory: output))
        let manifestPath = URL(fileURLWithPath: result.outputDirectory)
            .appendingPathComponent("manifest.json")
        guard case let .v2(verified) = try GTurboManifestDocumentCodec.decode(
            Data(contentsOf: manifestPath)) else {
            Issue.record("expected v2 fixture manifest")
            return
        }
        #expect(verified.manifest.family == .gemma4)
        #expect(verified.manifest.modelID == "fixture/transformed-v2")
        #expect(!FileManager.default.fileExists(
            atPath: URL(fileURLWithPath: result.outputDirectory)
                .appendingPathComponent("verified-install.json").path))
        try RepackAudit().verifyTransformedPack(
            rootDirectory: result.outputDirectory, manifest: verified)

        let weights = URL(fileURLWithPath: result.outputDirectory)
            .appendingPathComponent("model_weights.bin")
        var corrupt = try Data(contentsOf: weights)
        corrupt[0] ^= 0x01
        try corrupt.write(to: weights)
        #expect(throws: RepackError.self) {
            try RepackAudit().verifyTransformedPack(
                rootDirectory: result.outputDirectory, manifest: verified)
        }
    }

    @Test func routedSourceMutationThenRestorationRefusesResumeBeforePayloadWrite() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("routed-source.bf16")
        let original = try routedSourceData()
        try original.write(to: sourceURL)
        let plan = try routedProductionPlan(sourcePath: sourceURL.path)
        let output = root.appendingPathComponent("routed.gturbo").path

        var mutated = false
        var interruptAfterCommit = false
        var firstPass = TransformedTensorWriterOperations.production
        let productionRead = firstPass.read
        firstPass.read = { fd, path, offset, count in
            let data = try productionRead(fd, path, offset, count)
            if !mutated,
               path == sourceURL.path,
               offset == 260,
               count == 66_560 {
                var changed = original
                changed[260] ^= 0x01
                try changed.write(to: sourceURL)
                mutated = true
            }
            return data
        }
        let productionSync = firstPass.sync
        firstPass.sync = { fd, path in
            try productionSync(fd, path)
            if path.hasSuffix("packed_experts/layer-000.bin") {
                interruptAfterCommit = true
            }
        }
        firstPass.cancellationCheck = {
            if interruptAfterCommit { throw CancellationError() }
        }
        #expect(throws: CancellationError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(outputDirectory: output, reserveBytes: 0),
                operations: firstPass)
        }
        #expect(mutated)
        try original.write(to: sourceURL)

        let paths = try RemoteInstallPaths(outputDirectory: output)
        let expertPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts/layer-000.bin")
        let beforeResume = try Data(contentsOf: URL(fileURLWithPath: expertPath))
        var attemptedPayloadWrite = false
        var resume = TransformedTensorWriterOperations.production
        let productionWrite = resume.write
        resume.write = { fd, path, bytes, offset in
            let written = try productionWrite(fd, path, bytes, offset)
            if path == expertPath {
                attemptedPayloadWrite = true
                throw CancellationError()
            }
            return written
        }

        // Unsafe baseline negative control: it restores the original full-source
        // hash, accepts tainted committed expert output, and reaches a new write.
        #expect(throws: RepackError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(outputDirectory: output, resume: true, reserveBytes: 0),
                operations: resume)
        }
        #expect(!attemptedPayloadWrite)
        #expect(try Data(contentsOf: URL(fileURLWithPath: expertPath)) == beforeResume)
        #expect(!FileManager.default.fileExists(atPath: paths.finalDirectory))
    }

    @Test func routedDestinationCorruptionRefusesResumeBeforePayloadWrite() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("clean-routed-source.bf16")
        try routedSourceData().write(to: sourceURL)
        let output = root.appendingPathComponent("corrupt-routed.gturbo").path
        let plan = try routedProductionPlan(sourcePath: sourceURL.path)
        var cancelAfterCommit = false
        var interrupted = TransformedTensorWriterOperations.production
        let productionSync = interrupted.sync
        interrupted.sync = { fd, path in
            try productionSync(fd, path)
            if path.hasSuffix("packed_experts/layer-000.bin") { cancelAfterCommit = true }
        }
        interrupted.cancellationCheck = {
            if cancelAfterCommit { throw CancellationError() }
        }
        #expect(throws: CancellationError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan, options: .init(outputDirectory: output, reserveBytes: 0),
                operations: interrupted)
        }

        let paths = try RemoteInstallPaths(outputDirectory: output)
        let expertPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts/layer-000.bin")
        var corrupt = try Data(contentsOf: URL(fileURLWithPath: expertPath))
        corrupt[0] ^= 0x01
        try corrupt.write(to: URL(fileURLWithPath: expertPath))
        let beforeResume = corrupt
        var attemptedPayloadWrite = false
        var resume = TransformedTensorWriterOperations.production
        resume.write = { _, path, _, _ in
            if path == expertPath { attemptedPayloadWrite = true }
            throw CancellationError()
        }
        #expect(throws: RepackError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(outputDirectory: output, resume: true, reserveBytes: 0),
                operations: resume)
        }
        #expect(!attemptedPayloadWrite)
        #expect(try Data(contentsOf: URL(fileURLWithPath: expertPath)) == beforeResume)
    }

    @Test func routedLayoutCorruptionRefusesResumeBeforePayloadWrite() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("layout-source.bf16")
        try routedSourceData().write(to: sourceURL)
        let output = root.appendingPathComponent("corrupt-layout.gturbo").path
        let plan = try routedProductionPlan(sourcePath: sourceURL.path)
        var cancelAfterCommit = false
        var interrupted = TransformedTensorWriterOperations.production
        let productionSync = interrupted.sync
        interrupted.sync = { fd, path in
            try productionSync(fd, path)
            if path.hasSuffix("packed_experts/layer-000.bin") { cancelAfterCommit = true }
        }
        interrupted.cancellationCheck = {
            if cancelAfterCommit { throw CancellationError() }
        }
        #expect(throws: CancellationError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan, options: .init(outputDirectory: output, reserveBytes: 0),
                operations: interrupted)
        }

        let paths = try RemoteInstallPaths(outputDirectory: output)
        let layoutPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts/layout.json")
        var corruptedLayout = try Data(contentsOf: URL(fileURLWithPath: layoutPath))
        #expect(!corruptedLayout.isEmpty)
        corruptedLayout[0] ^= 0x01
        try corruptedLayout.write(to: URL(fileURLWithPath: layoutPath))
        let beforeResume = corruptedLayout
        var attemptedPayloadWrite = false
        var resume = TransformedTensorWriterOperations.production
        resume.write = { _, _, _, _ in
            attemptedPayloadWrite = true
            throw CancellationError()
        }

        #expect(throws: RepackError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(outputDirectory: output, resume: true, reserveBytes: 0),
                operations: resume)
        }
        #expect(!attemptedPayloadWrite)
        #expect(try Data(contentsOf: URL(fileURLWithPath: layoutPath)) == beforeResume)
        #expect(!FileManager.default.fileExists(atPath: paths.finalDirectory))
    }

    @Test func qwenDurableProgressStopsAfterSyncAndCheckpointFailure() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("reporting-source.bf16")
        try routedSourceData().write(to: sourceURL)
        let plan = try routedProductionPlan(sourcePath: sourceURL.path)

        let failedOutput = root.appendingPathComponent("reporting-sync-failure.gturbo").path
        var failingOperations = TransformedTensorWriterOperations.production
        let productionSync = failingOperations.sync
        var syncCount = 0
        failingOperations.sync = { fd, path in
            syncCount += 1
            if syncCount == 2 { throw QwenReportingTestError.syncFailed }
            try productionSync(fd, path)
        }
        let failedProgress = QwenDurableProgressBox()
        #expect(throws: QwenReportingTestError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(
                    outputDirectory: failedOutput,
                    reserveBytes: 0,
                    durableProgress: { failedProgress.append($0) }),
                operations: failingOperations)
        }
        #expect(failedProgress.snapshot == [0, 4])
        let failedPaths = try RemoteInstallPaths(outputDirectory: failedOutput)
        let failedCheckpoint = try RemoteInstallCheckpoint.load(
            from: failedPaths.checkpointFile)
        #expect(failedCheckpoint.transformProgress?.totalCompletedUnitCount == 4)

        let checkpointFailureOutput = root
            .appendingPathComponent("reporting-checkpoint-failure.gturbo").path
        let checkpointFailurePaths = try RemoteInstallPaths(
            outputDirectory: checkpointFailureOutput)
        var checkpointOperations = TransformedTensorWriterOperations.production
        let checkpointProductionSync = checkpointOperations.sync
        var checkpointSyncCount = 0
        checkpointOperations.sync = { fd, path in
            checkpointSyncCount += 1
            try checkpointProductionSync(fd, path)
            if checkpointSyncCount == 1 {
                try FileManager.default.createDirectory(
                    atPath: checkpointFailurePaths.checkpointFile + ".tmp",
                    withIntermediateDirectories: true)
            }
        }
        let checkpointProgress = QwenDurableProgressBox()
        #expect(throws: RepackError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(
                    outputDirectory: checkpointFailureOutput,
                    reserveBytes: 0,
                    durableProgress: { checkpointProgress.append($0) }),
                operations: checkpointOperations)
        }
        #expect(checkpointSyncCount == 1)
        #expect(checkpointProgress.snapshot == [0])
        let checkpoint = try RemoteInstallCheckpoint.load(
            from: checkpointFailurePaths.checkpointFile)
        #expect(checkpoint.transformProgress?.totalCompletedUnitCount == 0)
    }

    @Test func qwenFactoryMetadataPlanCodecRoundTripsAllArtifactKinds() throws {
        let plan = try metadataOnlyQwenPlan()
        let layout = try QwenTransformedPackWriter.layoutData(plan: plan)
        #expect(layout.count == Int(try #require(plan.artifacts.first {
            $0.relativePath == "packed_experts/layout.json"
        }).size))
        let digest = String(repeating: "a", count: 64)
        let outputs = Dictionary(uniqueKeysWithValues: plan.artifacts.map {
            ($0.relativePath, RepackAudit.OutputFile(
                relativePath: $0.relativePath, size: $0.size, sha256: digest))
        })
        let manifest = try QwenTransformedPackWriter.makeTextManifest(
            plan: plan, outputFiles: outputs)
        #expect(manifest.manifest.family == .qwen3_6)
        #expect(Set(manifest.manifest.files.keys) == Set(plan.artifacts.map(\.relativePath)))
        #expect(manifest.manifest.tensorRegions.contains {
            $0.file == "model_weights.bin" && $0.name == "fixture.text"
        })
        let expertRegions = manifest.manifest.tensorRegions.filter {
            $0.file == "packed_experts/layer-000.bin"
        }
        #expect(expertRegions.count == 256)
        #expect(expertRegions.allSatisfy {
            $0.offset.isMultiple(of: GTurboFormatV2.alignmentBytes)
                && $0.size == 82
        })
        #expect(manifest.manifest.expertStride == GTurboFormatV2.alignmentBytes)
        // This factory accepts QwenRepackPlan only. The synthetic writer has
        // no overload here, so a fixture cannot be passed into this API.
        #expect(plan.textTensors.count == 1)
    }

    @Test func qwenVisionFactoryCodecRoundTripsSeparateCompanionMetadata() throws {
        let plan = try metadataOnlyQwenPlan()
        let digest = String(repeating: "a", count: 64)
        let source = SourceTensor(name: "fixture.vision", shardPath: "metadata-only",
                                  dtype: .bf16, shape: [2, 65], absoluteOffset: 0,
                                  sizeBytes: 260)
        let vision = QwenVisionCompanionPlan(
            artifact: .init(relativePath: "vision_weights.bin",
                            size: GTurboFormatV2.alignmentBytes),
            tensors: [.init(source: source, relativeFile: "vision_weights.bin",
                            executionPosition: 0, fileOffset: 0, regionSize: 260,
                            storage: .retainedBF16, quantizationCategory: nil,
                            affineComponents: nil)])
        let processorSHA = try #require(
            plan.provenance.validatedSidecarSHA256[GTurboVisionFormatV2.processorFile])
        let manifest = QwenTransformedPackWriter.makeVisionManifest(
            plan: plan, vision: vision, processorBytes: 7, processorSHA256: processorSHA,
            textManifestSHA256: digest, weightsSHA256: digest)
        let verified = try GTurboVisionManifestV2Codec.decode(
            GTurboVisionManifestV2Codec.encode(manifest))
        #expect(verified.manifest.family == .qwen3_6)
        #expect(verified.manifest.modelID == GTurboFormatV2.qwenRepository)
        #expect(verified.manifest.files["vision_weights.bin"]?.size == GTurboFormatV2.alignmentBytes)
        #expect(verified.manifest.tensorRegions == manifest.tensorRegions)
        #expect(!verified.manifest.files.keys.contains("model_weights.bin"))
        #expect(verified.manifest.compatibleTextManifestSHA256 == digest)
    }

    @Test func insufficientPreflightCreatesNoPayloadAndCompletedDestinationIsProtected() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try makeSource(in: root)
        let constrained = root.appendingPathComponent("constrained.gturbo").path
        #expect(throws: RepackError.self) {
            _ = try SyntheticTransformedPackWriter.write(
                input: .init(source: source, outputDirectory: constrained,
                             reserveBytes: UInt64.max))
        }
        let constrainedPaths = try RemoteInstallPaths(outputDirectory: constrained)
        #expect(!FileManager.default.fileExists(atPath: constrainedPaths.finalDirectory))
        #expect(!FileManager.default.fileExists(atPath: constrainedPaths.partialDirectory))

        let complete = root.appendingPathComponent("complete.gturbo").path
        _ = try SyntheticTransformedPackWriter.write(
            input: .init(source: source, outputDirectory: complete))
        #expect(throws: RepackError.self) {
            _ = try SyntheticTransformedPackWriter.write(
                input: .init(source: source, outputDirectory: complete))
        }
    }

    @Test func publishedQwenTextReceiptBindsManifestAndRuntimeSchema() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("receipt-source.bf16")
        try routedSourceData().write(to: sourceURL)
        let plan = try routedProductionPlan(sourcePath: sourceURL.path)
        let output = root.appendingPathComponent("receipt.gturbo").path

        let result = try QwenTransformedPackWriter.write(
            plan: plan,
            options: .init(outputDirectory: output, reserveBytes: 0))
        let finalDirectory = try RemoteInstallPaths(outputDirectory: output).finalDirectory
        #expect(result.textReceiptPath ==
                URL(fileURLWithPath: finalDirectory)
                    .appendingPathComponent("verified-install.json").path)
        #expect(FileManager.default.fileExists(atPath: result.textReceiptPath))

        let manifestPath = URL(fileURLWithPath: finalDirectory)
            .appendingPathComponent("manifest.json")
        let manifestBytes = try Data(contentsOf: manifestPath)
        let receiptBytes = try Data(contentsOf: URL(fileURLWithPath: result.textReceiptPath))
        let receipt = try #require(
            try JSONSerialization.jsonObject(with: receiptBytes) as? [String: Any])
        #expect(receipt["schemaVersion"] as? Int == 1)
        #expect(receipt["manifestSha256"] as? String == result.textManifestSHA256)
        #expect(receipt["modelDirectoryPath"] as? String == finalDirectory)
        #expect(receipt["verificationTimestamp"] as? String ==
                "1970-01-01T00:00:00Z")
        #expect(receipt["sourceRepoID"] as? String == GTurboFormatV2.qwenRepository)
        #expect(receipt["sourceRevision"] as? String == GTurboFormatV2.qwenRevision)
        #expect(receipt["sourceIndexSHA256"] as? String ==
                GTurboFormatV2.qwenSourceIndexSHA256)
        #expect(receipt["planFingerprint"] as? String == plan.canonicalFingerprint)
        #expect(receipt["quantizationPolicySHA256"] as? String ==
                BF16AffineQuantizationPolicy.policySHA256)
        #expect(receipt["converterVersion"] as? String ==
                TransformedTensorWriter.converterVersion)
        let files = try #require(receipt["files"] as? [String: Any])
        let manifestEntry = try #require(files["manifest.json"] as? [String: Any])
        let manifestEntrySize = try #require(manifestEntry["size"] as? NSNumber)
        #expect(manifestEntrySize.uint64Value == UInt64(manifestBytes.count))
        #expect(manifestEntry["sha256"] as? String == result.textManifestSHA256)
        #expect(result.textReceiptSHA256 == digest(data: receiptBytes))
        _ = try GTurboManifestDocumentCodec.decode(manifestBytes)

        #expect(throws: RepackError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan, options: .init(outputDirectory: output, reserveBytes: 0))
        }
    }

    @Test func qwenFactoryBatchesAtDurableBoundaryAndResumesCommittedPrefix() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("factory-boundary-source.bf16")
        try factoryBoundarySourceData().write(to: sourceURL)
        let plan = try factoryBoundaryPlan(sourcePath: sourceURL.path)
        #expect(plan.expertLayers.first?.sources.first?.source.shape == [256, 2, 65])
        #expect(plan.expertLayers.first?.sources.first?.perExpertShape == [2, 65])
        #expect(plan.expertLayers.first?.sources.first?.source.sizeBytes == 66_560)
        let output = root.appendingPathComponent("factory-boundary.gturbo").path
        let paths = try RemoteInstallPaths(outputDirectory: output)
        let modelPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("model_weights.bin")

        var interrupted = TransformedTensorWriterOperations.production
        var syncCount = 0
        var cancelAfterBoundary = false
        let productionSync = interrupted.sync
        interrupted.sync = { fd, path in
            try productionSync(fd, path)
            if path == modelPath {
                syncCount += 1
                if syncCount == 1 { cancelAfterBoundary = true }
            }
        }
        interrupted.cancellationCheck = {
            if cancelAfterBoundary { throw CancellationError() }
        }

        #expect(throws: CancellationError.self) {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(outputDirectory: output, reserveBytes: 0),
                operations: interrupted)
        }
        // 65,538 affine groups are enough to exercise one 65,536-group
        // durability boundary and the final two-group boundary. The
        // checkpoint must contain the committed boundary, not a per-group
        // prefix guessed by the cancellation callback.
        #expect(syncCount == 1)
        let checkpoint = try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
        #expect(checkpoint.transformProgress?.completedUnitCount == 65_536)
        #expect(checkpoint.transformProgress?.totalCompletedUnitCount == 65_536)
        #expect(try Data(contentsOf: URL(fileURLWithPath: modelPath)).count
                == Int(plan.artifacts.first { $0.relativePath == "model_weights.bin" }!.size))

        var resumed = TransformedTensorWriterOperations.production
        let expertPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts/layer-000.bin")
        var resumedSyncPaths = [String]()
        let resumedProgress = QwenDurableProgressBox()
        let productionResumeSync = resumed.sync
        resumed.sync = { fd, path in
            try productionResumeSync(fd, path)
            resumedSyncPaths.append(path)
        }
        let resumedResult = try QwenTransformedPackWriter.write(
            plan: plan,
            options: .init(
                outputDirectory: output,
                resume: true,
                reserveBytes: 0,
                durableProgress: { resumedProgress.append($0) }),
            operations: resumed)
        #expect(resumedResult.completedGroupCount == 65_538 + 1_024)
        #expect(resumedSyncPaths.filter { $0 == modelPath }.count == 1)
        #expect(resumedSyncPaths.filter { $0 == expertPath }.count == 256)
        #expect(resumedSyncPaths.count == 257)
        let durable = resumedProgress.snapshot
        #expect(durable.first == 65_536)
        #expect(durable == durable.sorted())
        #expect(durable.last == UInt64(resumedResult.completedGroupCount))
        #expect(durable.count == resumedSyncPaths.count + 1)
        #expect(resumedResult.maximumObservedTransformScratchBytes ==
                BF16AffineTransformReader.maximumGroupBytes
                + StreamingBF16AffineQuantizer.maximumScratchPayloadBytes)

        let resumedFinalDirectory = try RemoteInstallPaths(outputDirectory: output).finalDirectory
        let resumedBytes = try directoryBytes(resumedFinalDirectory)
        try FileManager.default.removeItem(atPath: resumedFinalDirectory)
        _ = try QwenTransformedPackWriter.write(
            plan: plan,
            options: .init(outputDirectory: output, reserveBytes: 0))
        let cleanFinalDirectory = try RemoteInstallPaths(outputDirectory: output).finalDirectory
        #expect(cleanFinalDirectory == resumedFinalDirectory)
        let cleanBytes = try directoryBytes(cleanFinalDirectory)
        #expect(resumedBytes == cleanBytes)
    }

    @Test func pairedQwenPublicationFailureRollsBackBothOwnedDirectories() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("paired-source.bf16")
        var sourceData = try routedSourceData()
        sourceData.append(Data(repeating: 0x7f, count: 260))
        try sourceData.write(to: sourceURL)
        let plan = try routedVisionProductionPlan(sourcePath: sourceURL.path)
        let output = root.appendingPathComponent("paired.gturbo").path
        let visionOutput = root.appendingPathComponent("paired.vision.gturbo").path
        var publication = QwenPublicationOperations.production
        publication.beforeFinalAudit = { target in
            if case .vision = target { throw PublicationFailure.audit }
        }

        do {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(outputDirectory: output,
                               visionOutputDirectory: visionOutput,
                               visionProcessorData: canonicalProcessorData(),
                               reserveBytes: 0),
                publicationOperations: publication)
            Issue.record("paired publication unexpectedly succeeded")
        } catch is PublicationFailure {
            // The injected final audit failure is the expected publication error.
        }

        let textPaths = try RemoteInstallPaths(outputDirectory: output)
        let visionPaths = try RemoteInstallPaths(outputDirectory: visionOutput)
        #expect(!FileManager.default.fileExists(atPath: textPaths.finalDirectory))
        #expect(!FileManager.default.fileExists(atPath: visionPaths.finalDirectory))
        #expect(FileManager.default.fileExists(atPath: textPaths.partialDirectory))
        #expect(FileManager.default.fileExists(atPath: visionPaths.partialDirectory))
        #expect(FileManager.default.fileExists(atPath: textPaths.checkpointFile))
    }

    @Test func pairedQwenPublicationRollbackFailureReturnsOwnedStateError() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("rollback-source.bf16")
        var sourceData = try routedSourceData()
        sourceData.append(Data(repeating: 0x3f, count: 260))
        try sourceData.write(to: sourceURL)
        let plan = try routedVisionProductionPlan(sourcePath: sourceURL.path)
        let output = root.appendingPathComponent("rollback.gturbo").path
        let visionOutput = root.appendingPathComponent("rollback.vision.gturbo").path
        var publication = QwenPublicationOperations.production
        var renameCount = 0
        let productionRename = publication.rename
        publication.rename = { source, destination in
            renameCount += 1
            // text partial -> final, vision partial -> final, vision final -> partial,
            // then fail while restoring text final -> partial.
            if renameCount == 4 { throw PublicationFailure.rollback }
            try productionRename(source, destination)
        }
        publication.beforeFinalAudit = { target in
            if case .vision = target { throw PublicationFailure.audit }
        }

        do {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(outputDirectory: output,
                               visionOutputDirectory: visionOutput,
                               visionProcessorData: canonicalProcessorData(),
                               reserveBytes: 0),
                publicationOperations: publication)
            Issue.record("paired publication unexpectedly succeeded")
        } catch let error as RepackError {
            #expect(String(describing: error).contains("rollback failed"))
        }

        let textPaths = try RemoteInstallPaths(outputDirectory: output)
        #expect(FileManager.default.fileExists(atPath: textPaths.finalDirectory))
        #expect(FileManager.default.fileExists(atPath: textPaths.checkpointFile))
    }

    @Test func checkpointCleanupFailureKeepsPublishedOutputWithoutRollback() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("cleanup-source.bf16")
        var sourceData = try routedSourceData()
        sourceData.append(Data(repeating: 0x7f, count: 260))
        try sourceData.write(to: sourceURL)
        let plan = try routedVisionProductionPlan(sourcePath: sourceURL.path)
        let output = root.appendingPathComponent("cleanup.gturbo").path
        let visionOutput = root.appendingPathComponent("cleanup.vision.gturbo").path
        var publication = QwenPublicationOperations.production
        var fsyncCount = 0
        var renameCount = 0
        let productionFsync = publication.fsyncDirectory
        publication.fsyncDirectory = { path in
            fsyncCount += 1
            // Text and vision publication each fsync once. The third parent
            // fsync is checkpoint cleanup durability after both final audits.
            if fsyncCount == 3 { throw PublicationFailure.cleanup }
            try productionFsync(path)
        }
        let productionRename = publication.rename
        publication.rename = { source, destination in
            renameCount += 1
            try productionRename(source, destination)
        }

        do {
            _ = try QwenTransformedPackWriter.write(
                plan: plan,
                options: .init(
                    outputDirectory: output,
                    visionOutputDirectory: visionOutput,
                    visionProcessorData: canonicalProcessorData(),
                    reserveBytes: 0),
                publicationOperations: publication)
            Issue.record("checkpoint cleanup unexpectedly succeeded")
        } catch let error as RepackError {
            #expect(String(describing: error).contains("published outputs remain authoritative"))
        }

        let paths = try RemoteInstallPaths(outputDirectory: output)
        let visionPaths = try RemoteInstallPaths(outputDirectory: visionOutput)
        #expect(fsyncCount == 3)
        #expect(renameCount == 2)
        #expect(FileManager.default.fileExists(atPath: paths.finalDirectory))
        #expect(FileManager.default.fileExists(atPath: visionPaths.finalDirectory))
        #expect(!FileManager.default.fileExists(atPath: paths.partialDirectory))
        #expect(!FileManager.default.fileExists(atPath: visionPaths.partialDirectory))
    }
}

private enum PublicationFailure: Error {
    case audit
    case rollback
    case cleanup
}

private func temporaryRoot() throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("turbofieldfare-qwen-resume-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func makeSource(in root: URL) throws -> SourceTensor {
    let path = root.appendingPathComponent("tiny.bf16")
    var values = [Float](arrayLiteral: 0, 1, 2, 3)
    values += Array(repeating: 0, count: 61)
    values += values
    var data = Data()
    for value in values {
        var bits = StreamingBF16AffineQuantizer.encodeBF16(value).littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    try data.write(to: path)
    return .init(name: "fixture.tensor", shardPath: path.path, dtype: .bf16,
                 shape: [2, 65], absoluteOffset: 0, sizeBytes: UInt64(data.count))
}

private func routedSourceData() throws -> Data {
    var values = [Float](arrayLiteral: 0, 1, 2, 3)
    values += Array(repeating: 0, count: 61)
    values += values
    var tensor = Data()
    for value in values {
        var bits = StreamingBF16AffineQuantizer.encodeBF16(value).littleEndian
        withUnsafeBytes(of: &bits) { tensor.append(contentsOf: $0) }
    }
    var source = tensor
    for _ in 0..<256 { source.append(tensor) }
    return source
}

private func routedProductionPlan(sourcePath: String) throws -> QwenRepackPlan {
    let plan = try metadataOnlyQwenPlan()
    let text = SourceTensor(name: "fixture.text", shardPath: sourcePath,
                            dtype: .bf16, shape: [2, 65], absoluteOffset: 0,
                            sizeBytes: 260)
    let expert = SourceTensor(name: "fixture.expert", shardPath: sourcePath,
                              dtype: .bf16, shape: [256, 2, 65], absoluteOffset: 260,
                              sizeBytes: 66_560)
    let textTensors = plan.textTensors.map {
        QwenPlannedTensor(source: text, relativeFile: $0.relativeFile,
                          executionPosition: $0.executionPosition,
                          fileOffset: $0.fileOffset, regionSize: $0.regionSize,
                          storage: $0.storage,
                          quantizationCategory: $0.quantizationCategory,
                          affineComponents: $0.affineComponents)
    }
    let expertLayers = plan.expertLayers.map { layer in
        QwenExpertLayerPlan(
            layerIndex: layer.layerIndex, relativePath: layer.relativePath,
            expertsPerLayer: layer.expertsPerLayer, expertStride: layer.expertStride,
            sources: layer.sources.map {
                QwenRoutedSourcePlan(role: $0.role, source: expert,
                                     perExpertShape: $0.perExpertShape,
                                     components: $0.components)
            }, fileSize: layer.fileSize)
    }
    return QwenRepackPlan(
        architecture: plan.architecture, provenance: plan.provenance,
        quantizationGroups: plan.quantizationGroups,
        runtimeRecurrentStateStorage: plan.runtimeRecurrentStateStorage,
        textTensors: textTensors, expertLayers: expertLayers,
        visionTensorNames: plan.visionTensorNames, visionCompanion: nil,
        omittedMTPNames: plan.omittedMTPNames, artifacts: plan.artifacts,
        requiredArtifactBytes: plan.requiredArtifactBytes, scratch: plan.scratch,
        canonicalFingerprint: plan.canonicalFingerprint)
}

private func routedVisionProductionPlan(sourcePath: String) throws -> QwenRepackPlan {
    let plan = try routedProductionPlan(sourcePath: sourcePath)
    let visionSource = SourceTensor(name: "fixture.vision", shardPath: sourcePath,
                                    dtype: .bf16, shape: [2, 65], absoluteOffset: 66_820,
                                    sizeBytes: 260)
    let vision = QwenVisionCompanionPlan(
        artifact: .init(relativePath: "vision_weights.bin",
                        size: GTurboFormatV2.alignmentBytes),
        tensors: [.init(source: visionSource, relativeFile: "vision_weights.bin",
                        executionPosition: 0, fileOffset: 0, regionSize: 260,
                        storage: .retainedBF16, quantizationCategory: nil,
                        affineComponents: nil)])
    let artifacts = plan.artifacts + [vision.artifact]
    return QwenRepackPlan(
        architecture: plan.architecture, provenance: plan.provenance,
        quantizationGroups: plan.quantizationGroups,
        runtimeRecurrentStateStorage: plan.runtimeRecurrentStateStorage,
        textTensors: plan.textTensors, expertLayers: plan.expertLayers,
        visionTensorNames: ["fixture.vision"], visionCompanion: vision,
        omittedMTPNames: plan.omittedMTPNames, artifacts: artifacts,
        requiredArtifactBytes: artifacts.reduce(0) { $0 + $1.size },
        scratch: plan.scratch, canonicalFingerprint: plan.canonicalFingerprint)
}

private func factoryBoundarySourceData() throws -> Data {
    let row = try routedSourceData().prefix(130)
    var data = Data()
    data.reserveCapacity(8_519_940)
    for _ in 0..<32_769 { data.append(contentsOf: row) }
    data.append(Data(try routedSourceData().dropFirst(260)))
    return data
}

private func factoryBoundaryPlan(sourcePath: String) throws -> QwenRepackPlan {
    let base = try metadataOnlyQwenPlan()
    let boundaryRows: UInt64 = 32_769
    let boundaryShape: [UInt64] = [boundaryRows, 65]
    let boundaryBytes = boundaryRows * 65 * 2
    let components = try QwenRepackPlanner.affineComponentSizes(
        shape: boundaryShape, bitWidth: .int4)
    let textSource = SourceTensor(
        name: "fixture.boundary-text", shardPath: sourcePath, dtype: .bf16,
        shape: boundaryShape, absoluteOffset: 0, sizeBytes: boundaryBytes)
    let expertOffset = boundaryBytes
    let expertSource = SourceTensor(
        name: "fixture.expert", shardPath: sourcePath, dtype: .bf16,
        shape: [256, 2, 65], absoluteOffset: expertOffset, sizeBytes: 66_560)
    let textTensor = QwenPlannedTensor(
        source: textSource, relativeFile: "model_weights.bin", executionPosition: 0,
        fileOffset: 0, regionSize: components.totalSize, storage: .affineInt4,
        quantizationCategory: .attention, affineComponents: components)
    let expertLayers = base.expertLayers.map { layer in
        QwenExpertLayerPlan(
            layerIndex: layer.layerIndex, relativePath: layer.relativePath,
            expertsPerLayer: layer.expertsPerLayer, expertStride: layer.expertStride,
            sources: layer.sources.map {
                QwenRoutedSourcePlan(
                    role: $0.role, source: expertSource,
                    perExpertShape: $0.perExpertShape, components: $0.components)
            }, fileSize: layer.fileSize)
    }
    let alignment = UInt64(GTurboFormatV2.alignmentBytes)
    let residentSize = ((components.totalSize + alignment - 1) / alignment) * alignment
    let artifacts = base.artifacts.map { artifact in
        artifact.relativePath == "model_weights.bin"
            ? QwenPlannedArtifact(relativePath: artifact.relativePath, size: residentSize)
            : artifact
    }
    return QwenRepackPlan(
        architecture: base.architecture, provenance: base.provenance,
        quantizationGroups: base.quantizationGroups,
        runtimeRecurrentStateStorage: base.runtimeRecurrentStateStorage,
        textTensors: [textTensor], expertLayers: expertLayers,
        visionTensorNames: base.visionTensorNames, visionCompanion: base.visionCompanion,
        omittedMTPNames: base.omittedMTPNames, artifacts: artifacts,
        requiredArtifactBytes: artifacts.reduce(0) { $0 + $1.size },
        scratch: base.scratch,
        canonicalFingerprint: String(repeating: "b", count: 64))
}

private func canonicalProcessorData() -> Data {
    let text = """
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
"""
    return Data(text.utf8)
}

private func metadataOnlyQwenPlan() throws -> QwenRepackPlan {
    let layers = (0..<40).map { ($0 + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention" }
    let config: [String: Any] = [
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe", "tie_word_embeddings": false,
        "image_token_id": 248_056, "video_token_id": 248_057,
        "vision_start_token_id": 248_053, "vision_end_token_id": 248_054,
        "text_config": [
            "model_type": "qwen3_5_moe_text", "dtype": "bfloat16",
            "hidden_size": 2_048, "num_hidden_layers": 40, "layer_types": layers,
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
            "attn_output_gate": true, "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16, "linear_key_head_dim": 128,
            "linear_num_value_heads": 32, "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32", "partial_rotary_factor": 0.25,
            "rope_parameters": ["rope_theta": 10_000_000, "mrope_interleaved": true, "mrope_section": [11, 11, 10]],
            "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512, "vocab_size": 248_320,
            "tie_word_embeddings": false, "hidden_act": "silu", "bos_token_id": 248_044,
            "eos_token_id": 248_044, "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
            "max_position_embeddings": 262_144, "rms_norm_eps": 0.000001,
        ],
        "vision_config": [
            "depth": 27, "hidden_size": 1_152, "intermediate_size": 4_304,
            "num_heads": 16, "num_position_embeddings": 2_304, "in_channels": 3,
            "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
            "out_hidden_size": 2_048, "hidden_act": "gelu_pytorch_tanh",
        ],
    ]
    let architecture = try ArchInfo.loadQwenMetadataFixture(
        configData: JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]))
    let components = try QwenRepackPlanner.affineComponentSizes(
        shape: [2, 65], bitWidth: .int4)
    let text = SourceTensor(name: "fixture.text", shardPath: "metadata-only",
                            dtype: .bf16, shape: [2, 65], absoluteOffset: 0, sizeBytes: 260)
    let expert = SourceTensor(name: "fixture.expert", shardPath: "metadata-only",
                              dtype: .bf16, shape: [256, 2, 65], absoluteOffset: 260,
                              sizeBytes: 66_560)
    let layer = QwenExpertLayerPlan(
        layerIndex: 0, relativePath: "packed_experts/layer-000.bin",
        expertsPerLayer: 256, expertStride: GTurboFormatV2.alignmentBytes,
        sources: [.init(role: "down", source: expert, perExpertShape: [2, 65], components: components)],
        fileSize: 256 * GTurboFormatV2.alignmentBytes)
    let layoutObject: [String: Any] = ["version": 2, "layers": [[
        "layer": 0, "path": "packed_experts/layer-000.bin", "experts": 256,
        "stride": GTurboFormatV2.alignmentBytes, "sources": [[
            "name": "fixture.expert", "role": "down", "shape": [2, 65],
            "valuesOffset": components.valuesOffset, "valuesSize": components.valuesSize,
            "scalesOffset": components.scalesOffset, "scalesSize": components.scalesSize,
            "biasesOffset": components.biasesOffset, "biasesSize": components.biasesSize,
        ]],
    ]]]
    let layoutSize = UInt64(try JSONSerialization.data(
        withJSONObject: layoutObject, options: [.sortedKeys]).count)
    let digest = String(repeating: "a", count: 64)
    let provenance = QwenPlanningProvenance(
        repository: GTurboFormatV2.qwenRepository, revision: GTurboFormatV2.qwenRevision,
        validatedSidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
        observedIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
        observedConfigSHA256: digest,
        quantizationPolicySHA256: BF16AffineQuantizationPolicy.policySHA256)
    let omitted = ["fc.weight", "layers.0.input_layernorm.weight", "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert.down_proj.weight", "layers.0.mlp.shared_expert.gate_proj.weight", "layers.0.mlp.shared_expert.up_proj.weight", "layers.0.mlp.shared_expert_gate.weight", "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight", "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"].map { "mtp." + $0 }
    let artifacts = [
        QwenPlannedArtifact(relativePath: "model_weights.bin", size: GTurboFormatV2.alignmentBytes),
        QwenPlannedArtifact(relativePath: "packed_experts/layer-000.bin", size: layer.fileSize),
        QwenPlannedArtifact(relativePath: "packed_experts/layout.json", size: layoutSize),
    ]
    return QwenRepackPlan(
        architecture: architecture, provenance: provenance,
        quantizationGroups: BF16AffineQuantizationPolicy.manifestQuantizationGroups,
        runtimeRecurrentStateStorage: .fp32,
        textTensors: [.init(source: text, relativeFile: "model_weights.bin", executionPosition: 0,
                            fileOffset: 0, regionSize: components.totalSize, storage: .affineInt4,
                            quantizationCategory: .attention, affineComponents: components)],
        expertLayers: [layer], visionTensorNames: [], visionCompanion: nil,
        omittedMTPNames: omitted, artifacts: artifacts,
        requiredArtifactBytes: artifacts.reduce(0) { $0 + $1.size },
        scratch: .init(transformTileBytes: WriterCore.tileBytes, maximumGroupBytes: 128,
                       quantizerPayloadBytes: 1, maximumPayloadBufferBytes: WriterCore.tileBytes + 129),
        canonicalFingerprint: digest)
}

private enum QwenReportingTestError: Error {
    case syncFailed
}

private final class QwenDurableProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [UInt64] = []

    func append(_ value: UInt64) {
        lock.lock(); defer { lock.unlock() }
        values.append(value)
    }

    var snapshot: [UInt64] {
        lock.lock(); defer { lock.unlock() }
        return values
    }
}

private func directoryBytes(_ root: String) throws -> [String: Data] {
    let rootURL = URL(fileURLWithPath: root, isDirectory: true)
    let enumerator = FileManager.default.enumerator(
        at: rootURL, includingPropertiesForKeys: [.isRegularFileKey])!
    var result: [String: Data] = [:]
    for case let url as URL in enumerator {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey])
        guard values.isRegularFile == true else { continue }
        let relative = url.path.dropFirst(rootURL.path.count + 1)
        result[String(relative)] = try Data(contentsOf: url)
    }
    return result
}

private func digest(data: Data) -> String {
    var stream = Sha256Stream()
    data.withUnsafeBytes { stream.update($0) }
    return stream.finalizeHexString()
}
