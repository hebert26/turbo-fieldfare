import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// Workflow tests use the internal operation seam with a tiny official-shaped
/// plan. The source file is synthetic BF16 data. No prepared shard payload is
/// opened or copied.
@Suite(.serialized)
struct LocalQwenStreamingRepackerTests {
    @Test func preflightReportsDeterministicCombinedBudgetAndProvenance() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let options = fixture.options()

        let preflight = try LocalQwenStreamingRepacker.preflight(
            options: options, operations: fixture.operations())

        #expect(preflight.sourceRepository == GTurboFormatV2.qwenRepository)
        #expect(preflight.sourceRevision == GTurboFormatV2.qwenRevision)
        #expect(preflight.sourceIndexSHA256 == GTurboFormatV2.qwenSourceIndexSHA256)
        #expect(preflight.sourcePayloadSHA256 == fixture.payload.shardSetSHA256)
        #expect(preflight.planFingerprint == fixture.plan.canonicalFingerprint)
        #expect(preflight.quantizationPolicySHA256 ==
                BF16AffineQuantizationPolicy.policySHA256)
        #expect(preflight.converterVersion == TransformedTensorWriter.converterVersion)
        #expect(preflight.artifactBytes >= fixture.plan.requiredArtifactBytes)
        #expect(preflight.requiredBytes > preflight.artifactBytes)
        #expect(preflight.availableBytes >= preflight.requiredBytes)
        #expect(preflight.diskRequirements.count == 1)
        #expect(!FileManager.default.fileExists(atPath: options.outputDirectory))
    }

    @Test func runReportsPreparedPreflightOnceBeforeWriterAndPropagatesScratch() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let events = WorkflowCallbackEventBox()
        var operations = fixture.operations()
        operations.writePack = { _, _ in
            events.append("write")
            return syntheticWorkflowWriteResult(maximumScratchBytes: 12_345)
        }

        let result = try LocalQwenStreamingRepacker.run(
            options: fixture.options(),
            operations: operations,
            preflightReport: { _ in events.append("preflight") },
            progress: { _ in })

        #expect(events.snapshot == ["preflight", "write"])
        #expect(result.maximumObservedTransformScratchBytes == 12_345)
        #expect(result.completedUnitCount == 17)
    }

    @Test func failedPrepareDoesNotReportPreflightOrInvokeWriter() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let events = WorkflowCallbackEventBox()
        var operations = fixture.operations()
        operations.makePlan = { _, _, _ in
            events.append("makePlan")
            throw RepackError.configurationInvalid(detail: "reporting fixture plan failure")
        }
        operations.writePack = { _, _ in
            events.append("write")
            return syntheticWorkflowWriteResult(maximumScratchBytes: 1)
        }

        #expect(throws: RepackError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: fixture.options(),
                operations: operations,
                preflightReport: { _ in events.append("preflight") },
                progress: { _ in })
        }
        #expect(events.snapshot == ["makePlan"])
    }

    @Test func legacyTrailingProgressClosureReceivesStagesAndDurableCounts() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let progressBox = WorkflowProgressBox()

        let result = try LocalQwenStreamingRepacker.run(
            options: fixture.options(),
            operations: fixture.operations()
        ) { update in
            progressBox.append(update)
        }

        let updates = progressBox.snapshot
        #expect(updates.contains { $0.stage == .planning })
        #expect(updates.contains { $0.stage == .validatingSource })
        #expect(updates.contains { $0.stage == .preflighting })
        #expect(updates.contains { $0.stage == .converting })
        #expect(updates.contains { $0.stage == .publishing })
        let durable = updates.compactMap { $0.durableCompletedUnitCount }
        #expect(durable.first == 0)
        #expect(durable == durable.sorted())
        #expect(durable.last == UInt64(result.completedUnitCount))
    }

    @Test func pairedPreflightUsesOneCombinedBudgetForTextAndVisionDestinations() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let options = fixture.options(includeVision: true)
        let preflight = try LocalQwenStreamingRepacker.preflight(
            options: options, operations: fixture.operations())

        #expect(preflight.visionOutputDirectory != nil)
        #expect(preflight.diskRequirements.count == 1)
        #expect(preflight.diskRequirements.first?.paths.count == 2)
        #expect(preflight.artifactBytes >= fixture.visionPlan.requiredArtifactBytes)
        #expect(preflight.requiredBytes > preflight.artifactBytes)
        #expect(preflight.availableBytes >= preflight.requiredBytes)
        #expect(!FileManager.default.fileExists(atPath: options.outputDirectory))
        #expect(options.visionOutputDirectory.map {
            !FileManager.default.fileExists(atPath: $0)
        } == true)
    }

    @Test func sourcePlanAndDiskFailuresCreateNoPartialOrTouchGemma() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let output = fixture.root.appendingPathComponent("failure.gturbo").path
        let options = fixture.options(output: output)
        let gemma = fixture.root.appendingPathComponent("gemma.gturbo")
        try Data("preserve".utf8).write(to: gemma)

        var sourceFailure = fixture.operations()
        sourceFailure.loadSnapshot = { _, _ in
            throw RepackError.sourceFingerprintRejected(path: "fixture", sha256: "bad")
        }
        #expect(throws: RepackError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: options, operations: sourceFailure)
        }
        #expect(!FileManager.default.fileExists(atPath: output))
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
        #expect(try Data(contentsOf: gemma) == Data("preserve".utf8))

        var planFailure = fixture.operations()
        planFailure.makePlan = { _, _, _ in
            throw RepackError.configurationInvalid(detail: "tiny planner failure")
        }
        #expect(throws: RepackError.self) {
            _ = try LocalQwenStreamingRepacker.preflight(
                options: options, operations: planFailure)
        }
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))

        let diskFailure = fixture.operations()
        let constrained = fixture.options(output: fixture.root.appendingPathComponent("disk.gturbo").path,
                                          reserveBytes: UInt64.max)
        #expect(throws: RepackError.self) {
            _ = try LocalQwenStreamingRepacker.preflight(
                options: constrained, operations: diskFailure)
        }
        #expect(!FileManager.default.fileExists(atPath: constrained.outputDirectory))
        #expect(!FileManager.default.fileExists(atPath: constrained.outputDirectory + ".partial"))
    }

    @Test func cancellationResumeProducesSameDirectoryBytesAsCleanRun() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        // Receipts bind the exact final directory path. Establish a clean
        // install, capture every published byte, then reuse that same final
        // destination for the interrupted and resumed install.
        let output = fixture.root.appendingPathComponent("same-destination.gturbo").path

        let cleanResult = try LocalQwenStreamingRepacker.run(
            options: fixture.options(output: output, includeVision: true),
            operations: fixture.operations())
        let cleanBytes = try directoryBytes(cleanResult.textOutputDirectory)
        let cleanReceiptBytes = cleanBytes["verified-install.json"]
        let cleanVisionOutput = try #require(cleanResult.visionOutputDirectory)
        let cleanVisionBytes = try directoryBytes(cleanVisionOutput)
        let cleanVisionReceiptBytes = cleanVisionBytes["verified-install.json"]
        try removeOwnedOutput(output)
        try removeOwnedOutput(cleanVisionOutput)
        let cancelBox = CancellationBox(cancelAfter: 4)
        var interrupted = fixture.operations()
        interrupted.writePack = { plan, options in
            var writer = TransformedTensorWriterOperations.production
            let productionCheck = writer.cancellationCheck
            writer.cancellationCheck = {
                let count = cancelBox.next()
                if count == cancelBox.cancelAfter { throw CancellationError() }
                try productionCheck()
            }
            return try QwenTransformedPackWriter.write(
                plan: plan, options: options, operations: writer)
        }
        #expect(throws: CancellationError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: fixture.options(output: output, includeVision: true),
                operations: interrupted)
        }
        let paths = try RemoteInstallPaths(outputDirectory: output)
        #expect(FileManager.default.fileExists(atPath: paths.partialDirectory))
        #expect(FileManager.default.fileExists(atPath: paths.checkpointFile))
        let visionPaths = try RemoteInstallPaths(outputDirectory: cleanVisionOutput)
        #expect(FileManager.default.fileExists(atPath: visionPaths.partialDirectory))
        // Paired Qwen output has one text-bound checkpoint for both partials.
        #expect(!FileManager.default.fileExists(atPath: visionPaths.checkpointFile))

        let resumedResult = try LocalQwenStreamingRepacker.run(
            options: fixture.options(output: output, resume: true, includeVision: true),
            operations: fixture.operations())
        #expect(resumedResult.textManifestSHA256 == cleanResult.textManifestSHA256)
        #expect(resumedResult.textReceiptSHA256 == cleanResult.textReceiptSHA256)
        #expect(resumedResult.visionOutputDirectory == cleanResult.visionOutputDirectory)
        #expect(resumedResult.visionManifestSHA256 == cleanResult.visionManifestSHA256)
        #expect(resumedResult.visionReceiptSHA256 == cleanResult.visionReceiptSHA256)
        let resumedBytes = try directoryBytes(resumedResult.textOutputDirectory)
        #expect(resumedBytes == cleanBytes)
        #expect(resumedBytes["verified-install.json"] == cleanReceiptBytes)
        let resumedVisionBytes = try directoryBytes(
            try #require(resumedResult.visionOutputDirectory))
        #expect(resumedVisionBytes == cleanVisionBytes)
        #expect(resumedVisionBytes["verified-install.json"] == cleanVisionReceiptBytes)
    }

    @Test func resumedDiskUsageCountsOnlyRemainingTextAndVisionWrites() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let output = fixture.root.appendingPathComponent("remaining.gturbo").path
        let cancelBox = CancellationBox(cancelAfter: 4)
        var interrupted = fixture.operations()
        interrupted.writePack = { plan, options in
            var writer = TransformedTensorWriterOperations.production
            let productionCheck = writer.cancellationCheck
            writer.cancellationCheck = {
                if cancelBox.next() == cancelBox.cancelAfter { throw CancellationError() }
                try productionCheck()
            }
            return try QwenTransformedPackWriter.write(
                plan: plan, options: options, operations: writer)
        }

        #expect(throws: CancellationError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: fixture.options(output: output, includeVision: true),
                operations: interrupted)
        }

        let visionOutput = try #require(
            fixture.options(output: output, includeVision: true).visionOutputDirectory)
        let usage = try QwenTransformedPackWriter.resumeDiskUsage(
            plan: fixture.visionPlan,
            outputDirectory: output,
            visionOutputDirectory: visionOutput)
        #expect(usage.textRemainingArtifactBytes > 0)
        #expect(usage.textRemainingArtifactBytes < fixture.plan.requiredArtifactBytes)
        #expect(usage.visionRemainingArtifactBytes > 0)
        let visionArtifactBytes = try #require(fixture.visionPlan.visionCompanion?.artifact.size)
        #expect(usage.visionRemainingArtifactBytes < visionArtifactBytes)

        let freshPreflight = try LocalQwenStreamingRepacker.preflight(
            options: fixture.options(output: output, includeVision: true),
            operations: fixture.operations())
        let resumedPreflight = try LocalQwenStreamingRepacker.preflight(
            options: fixture.options(output: output, resume: true, includeVision: true),
            operations: fixture.operations())
        let freshRequirement = try #require(freshPreflight.diskRequirements.first)
        let resumedRequirement = try #require(resumedPreflight.diskRequirements.first)
        let textReduction = fixture.plan.requiredArtifactBytes
            - usage.textRemainingArtifactBytes
        let visionReduction = visionArtifactBytes - usage.visionRemainingArtifactBytes
        let metadataAndCheckpointReduction = usage.existingTextMetadataBytes
            + usage.existingVisionMetadataBytes
            + usage.existingVisionProcessorBytes
            + RemoteInstallCheckpoint.maximumBytes
        #expect(freshRequirement.requiredBytes > resumedRequirement.requiredBytes)
        #expect(freshRequirement.requiredBytes - resumedRequirement.requiredBytes ==
                textReduction + visionReduction + metadataAndCheckpointReduction)
    }

    @Test func sourcePayloadMutationBetweenTransformAndPublicationRefusesFinalRename() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let output = fixture.root.appendingPathComponent("mutated.gturbo").path
        let box = PayloadMutationBox(first: fixture.payload)
        var operations = fixture.operations()
        operations.verifyPayload = { directory, shards, progress in
            let value = box.next()
            progress(value.shardBytes, value.shardBytes)
            return value
        }

        #expect(throws: RepackError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: fixture.options(output: output), operations: operations)
        }
        #expect(!FileManager.default.fileExists(atPath: output))
        #expect(FileManager.default.fileExists(atPath: output + ".partial"))
        #expect(FileManager.default.fileExists(
            atPath: try RemoteInstallPaths(outputDirectory: output).checkpointFile))
    }

    @Test func discardRemovesOnlyOwnedQwenPartialAndCheckpoint() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let output = fixture.root.appendingPathComponent("discard.gturbo").path
        let unrelated = fixture.root.appendingPathComponent("unrelated.partial")
        try FileManager.default.createDirectory(at: unrelated, withIntermediateDirectories: true)
        try Data("keep".utf8).write(to: unrelated.appendingPathComponent("keep.txt"))
        let box = CancellationBox(cancelAfter: 4)
        var interrupted = fixture.operations()
        interrupted.writePack = { plan, options in
            var writer = TransformedTensorWriterOperations.production
            let productionCheck = writer.cancellationCheck
            writer.cancellationCheck = {
                if box.next() == box.cancelAfter { throw CancellationError() }
                try productionCheck()
            }
            return try QwenTransformedPackWriter.write(
                plan: plan, options: options, operations: writer)
        }
        #expect(throws: CancellationError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: fixture.options(output: output), operations: interrupted)
        }

        try LocalQwenStreamingRepacker.discardPartial(
            options: fixture.options(output: output), operations: fixture.operations())
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
        #expect(!FileManager.default.fileExists(
            atPath: try RemoteInstallPaths(outputDirectory: output).checkpointFile))
        #expect(FileManager.default.fileExists(atPath: unrelated.path))
        #expect(try Data(contentsOf: unrelated.appendingPathComponent("keep.txt")) ==
                Data("keep".utf8))
    }

    @Test func publishedReceiptsVerifyAgainstRuntimeSchemaAndCompletedOutputIsProtected() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let options = fixture.options()
        let result = try LocalQwenStreamingRepacker.run(
            options: options, operations: fixture.operations())
        let verified = try LocalQwenStreamingRepacker.verifyPublished(
            options: options, operations: fixture.operations())

        #expect(verified.textOutputDirectory == result.textOutputDirectory)
        #expect(verified.textManifestSHA256 == result.textManifestSHA256)
        #expect(verified.textReceiptSHA256 == result.textReceiptSHA256)
        #expect(result.textReceiptPath.hasSuffix("verified-install.json"))
        #expect(FileManager.default.fileExists(atPath: result.textReceiptPath))
        #expect(throws: RepackError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: options, operations: fixture.operations())
        }
    }

    @Test func invalidOwnedCheckpointCannotDiscardUnrelatedPartial() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let output = fixture.root.appendingPathComponent("foreign.gturbo").path
        let paths = try RemoteInstallPaths(outputDirectory: output)
        try FileManager.default.createDirectory(atPath: paths.partialDirectory,
                                                withIntermediateDirectories: true)
        let foreign = RemoteInstallCheckpoint(
            repoID: "mlx-community/gemma", requestedRevision: "wrong",
            resolvedCommit: String(repeating: "f", count: 40),
            sourceIndexSHA256: String(repeating: "a", count: 64),
            planFingerprint: String(repeating: "b", count: 64),
            totalSourceBytes: 1)
        try foreign.write(to: paths.checkpointFile, parentDirectory: paths.parentDirectory)

        #expect(throws: RepackError.self) {
            try LocalQwenStreamingRepacker.discardPartial(
                options: fixture.options(output: output), operations: fixture.operations())
        }
        #expect(FileManager.default.fileExists(atPath: paths.partialDirectory))
        #expect(FileManager.default.fileExists(atPath: paths.checkpointFile))
    }

    @Test func discardRefusesSubstitutedPartialBeforeDeletingItsContents() throws {
        let fixture = try TinyLocalQwenFixture.make()
        defer { fixture.remove() }
        let output = fixture.root.appendingPathComponent("substituted.gturbo").path
        let cancelBox = CancellationBox(cancelAfter: 4)
        var interrupted = fixture.operations()
        interrupted.writePack = { plan, options in
            var writer = TransformedTensorWriterOperations.production
            let productionCheck = writer.cancellationCheck
            writer.cancellationCheck = {
                if cancelBox.next() == cancelBox.cancelAfter { throw CancellationError() }
                try productionCheck()
            }
            return try QwenTransformedPackWriter.write(
                plan: plan, options: options, operations: writer)
        }
        #expect(throws: CancellationError.self) {
            _ = try LocalQwenStreamingRepacker.run(
                options: fixture.options(output: output), operations: interrupted)
        }

        let paths = try RemoteInstallPaths(outputDirectory: output)
        let displaced = paths.partialDirectory + ".displaced"
        try FileManager.default.moveItem(atPath: paths.partialDirectory, toPath: displaced)
        try FileManager.default.createDirectory(
            atPath: paths.partialDirectory, withIntermediateDirectories: true)
        let substituted = (paths.partialDirectory as NSString).appendingPathComponent("unrelated.bin")
        try Data("substituted partial".utf8).write(to: URL(fileURLWithPath: substituted))

        #expect(throws: RepackError.self) {
            try LocalQwenStreamingRepacker.discardPartial(
                options: fixture.options(output: output), operations: fixture.operations())
        }
        #expect(FileManager.default.fileExists(atPath: substituted))
        #expect(FileManager.default.fileExists(atPath: paths.checkpointFile))
        #expect(FileManager.default.fileExists(atPath: displaced))
    }
}

private func syntheticWorkflowWriteResult(
    maximumScratchBytes: Int
) -> QwenTransformedPackWriteResult {
    .init(
        textManifestSHA256: String(repeating: "a", count: 64),
        textReceiptSHA256: String(repeating: "b", count: 64),
        textReceiptPath: "fixture/verified-install.json",
        visionManifestSHA256: nil,
        visionReceiptSHA256: nil,
        visionReceiptPath: nil,
        visionManifestWritten: false,
        completedGroupCount: 17,
        maximumObservedTransformScratchBytes: maximumScratchBytes)
}

private final class WorkflowCallbackEventBox: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []

    func append(_ value: String) {
        lock.lock(); defer { lock.unlock() }
        values.append(value)
    }

    var snapshot: [String] {
        lock.lock(); defer { lock.unlock() }
        return values
    }
}

private final class WorkflowProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [LocalQwenStreamingRepackProgress] = []

    func append(_ value: LocalQwenStreamingRepackProgress) {
        lock.lock(); defer { lock.unlock() }
        values.append(value)
    }

    var snapshot: [LocalQwenStreamingRepackProgress] {
        lock.lock(); defer { lock.unlock() }
        return values
    }
}

private final class CancellationBox: @unchecked Sendable {
    let cancelAfter: Int
    private var count = 0
    private let lock = NSLock()

    init(cancelAfter: Int) { self.cancelAfter = cancelAfter }

    func next() -> Int {
        lock.lock(); defer { lock.unlock() }
        count += 1
        return count
    }
}

private final class PayloadMutationBox: @unchecked Sendable {
    let first: LocalOfficialQwenPayloadIdentity
    private var count = 0
    private let lock = NSLock()

    init(first: LocalOfficialQwenPayloadIdentity) { self.first = first }

    func next() -> LocalOfficialQwenPayloadIdentity {
        lock.lock(); defer { lock.unlock() }
        count += 1
        if count == 1 { return first }
        return .init(
            checksumManifestSHA256: first.checksumManifestSHA256,
            shardSetSHA256: String(repeating: "e", count: 64),
            shardBytes: first.shardBytes,
            shardCount: first.shardCount)
    }
}

private struct TinyLocalQwenFixture {
    let root: URL
    let sourceDirectory: URL
    let sourcePath: URL
    let plan: QwenRepackPlan
    let visionPlan: QwenRepackPlan
    let snapshot: LocalPinnedSnapshot
    let payload: LocalOfficialQwenPayloadIdentity

    static func make() throws -> Self {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-qwen-workflow-\(UUID().uuidString)",
                                   isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let sourceDirectory = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceDirectory, withIntermediateDirectories: true)
        let sourcePath = sourceDirectory.appendingPathComponent("tiny.bf16")
        var sourceData = Self.routedSourceData()
        sourceData.append(Data(repeating: 0x5a, count: 260))
        try sourceData.write(to: sourcePath)
        try canonicalProcessorData().write(
            to: sourceDirectory.appendingPathComponent(GTurboVisionFormatV2.processorFile))
        let plan = try Self.tinyPlan(sourcePath: sourcePath.path)
        let visionPlan = try Self.tinyVisionPlan(sourcePath: sourcePath.path, plan: plan)
        let snapshot = LocalPinnedSnapshot(
            indexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
            shardFilenames: ["tiny.safetensors"], tensors: [])
        let payload = LocalOfficialQwenPayloadIdentity(
            checksumManifestSHA256: String(repeating: "c", count: 64),
            shardSetSHA256: String(repeating: "d", count: 64),
            shardBytes: UInt64(66_820), shardCount: 1)
        return .init(root: root, sourceDirectory: sourceDirectory, sourcePath: sourcePath,
                     plan: plan, visionPlan: visionPlan, snapshot: snapshot, payload: payload)
    }

    func options(
        output: String? = nil, resume: Bool = false, reserveBytes: UInt64 = 1,
        includeVision: Bool = false
    ) -> LocalQwenStreamingRepackOptions {
        .init(sourceDirectory: sourceDirectory.path,
              outputDirectory: output ?? root.appendingPathComponent("output.gturbo").path,
              visionOutputDirectory: includeVision
                ? root.appendingPathComponent("output.vision.gturbo").path : nil,
              resume: resume, reserveBytes: reserveBytes)
    }

    func operations() -> LocalQwenStreamingRepackerOperations {
        let snapshot = self.snapshot
        let plan = self.plan
        let visionPlan = self.visionPlan
        let payload = self.payload
        return .init(
            loadSnapshot: { _, _ in snapshot },
            makePlan: { _, _, includeVision in includeVision ? visionPlan : plan },
            verifyPayload: { _, _, progress in
                progress(payload.shardBytes, payload.shardBytes)
                return payload
            },
            writePack: { plan, options in
                try QwenTransformedPackWriter.write(plan: plan, options: options)
            })
    }

    func remove() { try? FileManager.default.removeItem(at: root) }

    private static func routedSourceData() -> Data {
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

    private static func tinyPlan(sourcePath: String) throws -> QwenRepackPlan {
        let layers = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention"
        }
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
                "rope_parameters": ["rope_theta": 10_000_000, "mrope_interleaved": true,
                                     "mrope_section": [11, 11, 10]],
                "num_experts": 256, "num_experts_per_tok": 8,
                "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
                "vocab_size": 248_320, "tie_word_embeddings": false,
                "hidden_act": "silu", "bos_token_id": 248_044, "eos_token_id": 248_044,
                "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
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
        let text = SourceTensor(name: "fixture.text", shardPath: sourcePath,
                                dtype: .bf16, shape: [2, 65], absoluteOffset: 0,
                                sizeBytes: 260)
        let expert = SourceTensor(name: "fixture.expert", shardPath: sourcePath,
                                  dtype: .bf16, shape: [256, 2, 65], absoluteOffset: 260,
                                  sizeBytes: 66_560)
        let layer = QwenExpertLayerPlan(
            layerIndex: 0, relativePath: "packed_experts/layer-000.bin",
            expertsPerLayer: 256, expertStride: GTurboFormatV2.alignmentBytes,
            sources: [.init(role: "down", source: expert, perExpertShape: [2, 65],
                             components: components)],
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
        let omitted = [
            "fc.weight", "layers.0.input_layernorm.weight",
            "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight",
            "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.v_proj.weight", "norm.weight",
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ].map { "mtp." + $0 }
        let artifacts = [
            QwenPlannedArtifact(relativePath: "model_weights.bin",
                                 size: GTurboFormatV2.alignmentBytes),
            QwenPlannedArtifact(relativePath: "packed_experts/layer-000.bin",
                                 size: layer.fileSize),
            QwenPlannedArtifact(relativePath: "packed_experts/layout.json", size: layoutSize),
        ]
        return QwenRepackPlan(
            architecture: architecture, provenance: provenance,
            quantizationGroups: BF16AffineQuantizationPolicy.manifestQuantizationGroups,
            runtimeRecurrentStateStorage: .fp32,
            textTensors: [.init(source: text, relativeFile: "model_weights.bin",
                                executionPosition: 0, fileOffset: 0,
                                regionSize: components.totalSize, storage: .affineInt4,
                                quantizationCategory: .attention,
                                affineComponents: components)],
            expertLayers: [layer], visionTensorNames: [], visionCompanion: nil,
            omittedMTPNames: omitted, artifacts: artifacts,
            requiredArtifactBytes: artifacts.reduce(0) { $0 + $1.size },
            scratch: .init(transformTileBytes: WriterCore.tileBytes,
                           maximumGroupBytes: 128, quantizerPayloadBytes: 1,
                           maximumPayloadBufferBytes: WriterCore.tileBytes + 129),
            canonicalFingerprint: digest)
    }

    private static func tinyVisionPlan(
        sourcePath: String, plan: QwenRepackPlan
    ) throws -> QwenRepackPlan {
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

private func removeOwnedOutput(_ output: String) throws {
    let paths = try RemoteInstallPaths(outputDirectory: output)
    for path in [paths.finalDirectory, paths.partialDirectory,
                 paths.checkpointFile, paths.lockFile] {
        try? FileManager.default.removeItem(atPath: path)
    }
}
