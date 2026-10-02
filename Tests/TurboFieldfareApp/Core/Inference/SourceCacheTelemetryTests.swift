import Darwin
import Foundation
import Metal
import Synchronization
import Testing
import TurboFieldfareDecodeProtocol
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfare
@testable import TurboFieldfareAppCore

/// P24's source-cache telemetry test uses a tiny, locally generated source
/// registration. It exercises the protected BF16 source reader and the real
/// source runner, while keeping original weights out of the test.
@Suite(.serialized) struct SourceCacheTelemetryTests {
    @Test func sourceClientCheckpointDryRunPreservesToolTransactionAndReceiptIdentity() async throws {
        let harness = try await SourceCheckpointHarness.make()
        defer { harness.fixture.remove() }
        let call = try await harness.generateToolCall(query: "snow")
        try await harness.synchronize(call: call)
        let before = await harness.generation.status()
        let beforeSnapshot = try await harness.generation.diagnosticSnapshot()
        let request = try sourceCheckpointRequest(call: call, record: "", commit: false)

        let receipt = try await harness.client.contextCheckpoint(request)

        #expect(!receipt.committed)
        #expect(receipt.checkpointID == request.checkpointID)
        #expect(receipt.replacementEpoch == request.replacementEpoch)
        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)
        #expect(await harness.client.conversationTokenCount
            == before.committed.retainedTokenIDs.count)
    }

    @Test func sourceClientCheckpointRejectsMismatchedToolResultBeforeMutation() async throws {
        let harness = try await SourceCheckpointHarness.make()
        defer { harness.fixture.remove() }
        let call = try await harness.generateToolCall(query: "snow")
        try await harness.synchronize(call: call)
        let before = await harness.generation.status()
        let beforeSnapshot = try await harness.generation.diagnosticSnapshot()
        var request = try sourceCheckpointRequest(call: call, record: "", commit: false)
        request.result.callID = "different-call-id"

        await #expect(throws: (any Error).self) {
            try await harness.client.contextCheckpoint(request)
        }

        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)
    }

    @Test func sourceClientAssessesNonemptyCheckpointWithoutRebuildingState() async throws {
        let stages = SourceCheckpointStageRecorder()
        let harness = try await SourceCheckpointHarness.make(
            hooks: QwenOfficialSourceTransactionHooks(
                afterActualGPUSubmission: { stage in stages.append("submit:\(stage)") },
                afterActualGPUCompletion: { stage in stages.append("complete:\(stage)") }))
        defer { harness.fixture.remove() }
        let call = try await harness.generateToolCall(query: "snow")
        try await harness.synchronize(call: call)

        let before = await harness.generation.status()
        let beforeSnapshot = try await harness.generation.diagnosticSnapshot()
        let maxContext = 1_024
        let resultMessage = ModelChatMessage(
            role: .tool, content: "synthetic result",
            toolCallID: call.id, name: call.name)
        let continuationTokenCount = try harness.codec.encodeContinuation(
            messages: [resultMessage],
            options: .init(enableThinking: false))
            .count
        let expectedExisting = await harness.client.conversationTokenCount
            + continuationTokenCount
        let expectedReplacement = try harness.codec.encodePrompt(
            messages: [ModelChatMessage(role: .user, content: "short checkpoint")],
            tools: [sourceLookupTool()],
            options: .init(enableThinking: false, preserveThinking: true))
            .count
        let expectedGenerationReserve = min(128, max(1, maxContext / 8))
        let expectedResultReserve = max(
            min(1_024, max(1, maxContext / 64)), continuationTokenCount * 2)
        let expectedFinalReserve = min(128, max(1, maxContext / 32))
        let expectedReserve = expectedGenerationReserve
            + expectedResultReserve + expectedFinalReserve

        stages.reset()
        let rejectedRequest = try sourceCheckpointRequest(
            call: call, record: "short checkpoint", commit: false,
            force: false, trigger: .sustainedSlowDecode)
        let rejected = try await harness.client.contextCheckpoint(rejectedRequest)
        #expect(!rejected.committed)
        #expect(!rejected.needed)
        #expect(rejected.existingPromptTokens == expectedExisting)
        #expect(rejected.replacementPromptTokens == expectedReplacement)
        #expect(rejected.reserveTokens == expectedReserve)
        #expect(rejected.resultAllowanceTokens == expectedResultReserve)
        #expect(rejected.performanceMinimumSavingsTokens == 4_096)
        #expect(rejected.retainedImageCount == 0)
        #expect(rejected.retainedImageRows == 0)
        #expect(rejected.retainedFeatureBytes == 0)
        #expect(rejected.reserveTokens > 0)
        #expect(stages.snapshot.isEmpty,
                "a rejected assessment must not rebuild or prefill the source state")
        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)

        stages.reset()
        let stop = AppGenerationStop()
        stop.requestStop()
        let cancelledRequest = try sourceCheckpointRequest(
            call: call, record: "short checkpoint", commit: false,
            force: true, trigger: .sustainedSlowDecode)
        await #expect(throws: CancellationError.self) {
            try await harness.client.contextCheckpoint(cancelledRequest, stop: stop)
        }
        #expect(stages.snapshot.isEmpty)
        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)

        stages.reset()
        let invalidRequest = try sourceCheckpointRequest(
            call: call, record: MultimodalPromptRenderer.placeholder, commit: false,
            force: false, trigger: .sustainedSlowDecode)
        await #expect(throws: AppInferenceError.self) {
            try await harness.client.contextCheckpoint(invalidRequest)
        }
        #expect(stages.snapshot.isEmpty)
        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)

        stages.reset()
        let invalidCommitRequest = try sourceCheckpointRequest(
            call: call, record: MultimodalPromptRenderer.placeholder, commit: true,
            force: true, trigger: .explicitComparison)
        await #expect(throws: AppInferenceError.self) {
            try await harness.client.contextCheckpoint(invalidCommitRequest)
        }
        #expect(stages.snapshot.isEmpty)
        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)

        stages.reset()
        let forcedRequest = try sourceCheckpointRequest(
            call: call, record: "short checkpoint", commit: false,
            force: true, trigger: .sustainedSlowDecode)
        let forcedAssessment = try await harness.client.contextCheckpoint(forcedRequest)
        #expect(!forcedAssessment.committed)
        #expect(forcedAssessment.needed)
        #expect(forcedAssessment.existingPromptTokens == expectedExisting)
        #expect(forcedAssessment.replacementPromptTokens == expectedReplacement)
        #expect(forcedAssessment.reserveTokens == expectedReserve)
        #expect(forcedAssessment.resultAllowanceTokens == expectedResultReserve)
        #expect(forcedAssessment.performanceMinimumSavingsTokens == 4_096)
        #expect(forcedAssessment.retainedImageCount == 0)
        #expect(forcedAssessment.retainedImageRows == 0)
        #expect(forcedAssessment.retainedFeatureBytes == 0)
        #expect(stages.snapshot.isEmpty,
                "a forced assessment must not rebuild or prefill the source state")
        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)
    }

    @Test func sourceClientCheckpointCancellationLeavesSourceTurnUnchanged() async throws {
        let harness = try await SourceCheckpointHarness.make()
        defer { harness.fixture.remove() }
        let call = try await harness.generateToolCall(query: "snow")
        try await harness.synchronize(call: call)
        let before = await harness.generation.status()
        let beforeSnapshot = try await harness.generation.diagnosticSnapshot()
        let request = try sourceCheckpointRequest(call: call, record: "", commit: false)
        let stop = AppGenerationStop()
        stop.requestStop()

        await #expect(throws: CancellationError.self) {
            try await harness.client.contextCheckpoint(request, stop: stop)
        }

        #expect(await harness.generation.status() == before)
        #expect(try await harness.generation.diagnosticSnapshot() == beforeSnapshot)
    }

    @Test func sourceAssessmentUsesExpandedRowsForOrderedImageFrames() throws {
        let context = try MetalContext()
        let tokenizer = try QwenTokenizer.loadOfficialSidecar(
            from: sourceCheckpointTokenizerDirectory)
        let codec = QwenChatCodec(tokenizer: tokenizer)
        let architecture = sourceAssessmentQwenArchitecture()
        let visionConfig = QwenVisionConfig(
            outputHiddenSize: 32, allowsFixtureGeometry: true)
        let first = try sourceAssessmentVisionFeatures(
            context: context, rows: 1, marker: 1, digest: "a")
        let second = try sourceAssessmentVisionFeatures(
            context: context, rows: 4, marker: 2, digest: "b")
        let messages = [ModelChatMessage(
            role: .user,
            content: .parts([
                .text("before"), .image(.init(id: "first")),
                .text("between"), .image(.init(id: "second")), .text("after"),
            ]))]
        let encoded = try codec.encodePrompt(
            messages: messages, tools: [], options: .init(enableThinking: false))
        let normalized = try normalizeQwenCodecImageFrames(
            encoded, architecture: architecture)
        let expanded = try MultimodalPromptRenderer.expandingQwenImageTokens(
            normalized, features: [first, second], architecture: architecture,
            config: visionConfig)
        let imageRows = first.tokenCount + second.tokenCount
        let estimate = RealInferenceSession.qwenExpandedTokenCount(
            encodedCount: encoded.count, imageRows: imageRows, imageCount: 2)

        #expect(normalized.count == encoded.count - 4)
        #expect(estimate == expanded.embeddingTokenIDs.count)
        #expect(expanded.effectiveTokenIDs.count == normalized.count + imageRows + 2)
        #expect(expanded.imageSpans.map { $0.features.tokenCount } == [1, 4])
        #expect(expanded.imageSpans.map { $0.features.owner.imageDigest } == [
            String(repeating: "a", count: 64), String(repeating: "b", count: 64),
        ])

        let textOnly = try codec.encodePrompt(
            messages: [ModelChatMessage(role: .user, content: "text only")],
            tools: [], options: .init(enableThinking: false))
        #expect(RealInferenceSession.qwenExpandedTokenCount(
            encodedCount: textOnly.count, imageRows: 0, imageCount: 0) == textOnly.count)
        #expect(RealInferenceSession.qwenExpandedTokenCount(
            encodedCount: Int.max, imageRows: Int.max, imageCount: 1) == Int.max)
    }

    @Test func sourceAllocationIsVisibleThroughClientAndSurvivesReset() async throws {
        let fixture = try SourceCacheTelemetryFixture.make()
        defer { fixture.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: fixture.registrationURL,
            context: context,
            residencyBudgetBytes: fixture.residencyBudgetBytes)
        let generation = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 8,
            expertSlotCount: fixture.expertSlotCount)
        let session = RealInferenceSession()
        try await session.installQwenOfficialSourceFixture(
            model: model, generation: generation,
            key: fixture.sessionKey, context: context)
        let client = RealInferenceClient(session: session)

        #expect(client.currentExpertCacheBytes == 0)
        let config = GenerationConfig(
            maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
            repetitionPenalty: 1, seed: 0, stopStrings: [], extraStopTokens: [])
        let result = try await generation.generatePreparedTurn(
            promptTokenIDs: [0, 1], config: config)
        #expect(result.metrics.retainedTokenIDs.count >= 2)

        let summary = try #require(generation.currentRoutedExpertCacheSummary)
        #expect(summary.allocatedBytes > 0)
        #expect(summary.allocatedBytes == fixture.expectedCacheBytes)
        #expect(summary.peakAllocatedBytes == summary.allocatedBytes)
        #expect(summary.misses > 0)
        #expect(client.currentExpertCacheBytes == summary.allocatedBytes)

        try await generation.reset()
        #expect(client.currentExpertCacheBytes == summary.allocatedBytes)
        #expect(generation.currentRoutedExpertCacheSummary == summary)

        await session.unload()
        #expect(client.currentExpertCacheBytes == nil)
    }

    @Test func failedProtectedMapStillPublishesAllocationWithoutFalseHitOrMiss() async throws {
        let fixture = try SourceCacheTelemetryFixture.make()
        defer { fixture.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: fixture.registrationURL,
            context: context,
            residencyBudgetBytes: fixture.residencyBudgetBytes)
        let generation = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 8,
            expertSlotCount: fixture.expertSlotCount,
            hooks: QwenOfficialSourceTransactionHooks(
                beforeProtectedExpertRead: { _, _, _ in
                    throw SourceCacheTelemetryFailure.protectedRead
                }))
        let session = RealInferenceSession()
        try await session.installQwenOfficialSourceFixture(
            model: model, generation: generation,
            key: fixture.sessionKey, context: context)
        let client = RealInferenceClient(session: session)

        var observedFailure: SourceCacheTelemetryFailure?
        do {
            _ = try await generation.generatePreparedTurn(
                promptTokenIDs: [0], config: GenerationConfig(maxNewTokens: 1))
        } catch let error as SourceCacheTelemetryFailure {
            observedFailure = error
        } catch {
            Issue.record("protected source map threw an unexpected error: \(error)")
        }
        #expect(observedFailure == .protectedRead)

        let summary = try #require(generation.currentRoutedExpertCacheSummary)
        // The injected read fails in the first layer, before the second
        // coordinator can be constructed. The published allocation therefore
        // covers one layer while hit and miss counters remain zero.
        #expect(summary.allocatedBytes == fixture.expectedLayerCacheBytes)
        #expect(summary.peakAllocatedBytes == fixture.expectedLayerCacheBytes)
        #expect(summary.hits == 0)
        #expect(summary.misses == 0)
        #expect(client.currentExpertCacheBytes == fixture.expectedLayerCacheBytes)

        await session.unload()
        #expect(client.currentExpertCacheBytes == nil)
    }

    @Test func replacingSourceInstallationDropsOldTelemetryOwnerBeforeNewAllocation() async throws {
        let fixture = try SourceCacheTelemetryFixture.make()
        defer { fixture.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: fixture.registrationURL,
            context: context,
            // Keep the first generation resident while installing and warming
            // its replacement. This test owns the replacement transition, so
            // both cache reservations must fit concurrently.
            residencyBudgetBytes: fixture.residencyBudgetBytes
                + fixture.expectedCacheBytes)
        let first = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 8,
            expertSlotCount: fixture.expertSlotCount)
        let second = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 8,
            expertSlotCount: fixture.expertSlotCount)
        let session = RealInferenceSession()
        try await session.installQwenOfficialSourceFixture(
            model: model, generation: first,
            key: fixture.sessionKey, context: context)
        let client = RealInferenceClient(session: session)
        let config = GenerationConfig(maxNewTokens: 1)
        _ = try await first.generatePreparedTurn(promptTokenIDs: [0], config: config)
        #expect(client.currentExpertCacheBytes == fixture.expectedCacheBytes)

        try await session.installQwenOfficialSourceFixture(
            model: model, generation: second,
            key: fixture.sessionKey, context: context)
        #expect(client.currentExpertCacheBytes == 0)
        #expect(first.currentRoutedExpertCacheSummary?.allocatedBytes
            == fixture.expectedCacheBytes)

        _ = try await second.generatePreparedTurn(promptTokenIDs: [1], config: config)
        #expect(client.currentExpertCacheBytes == fixture.expectedCacheBytes)
        await session.unload()
        #expect(client.currentExpertCacheBytes == nil)
    }
}

private enum SourceCheckpointFixtureFailure: Error {
    case noToolCall
}

private final class SourceToolCallRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var calls: [ParsedToolCall] = []

    func record(_ call: ParsedToolCall) {
        lock.lock()
        calls.append(call)
        lock.unlock()
    }

    var last: ParsedToolCall? {
        lock.lock()
        defer { lock.unlock() }
        return calls.last
    }
}

private final class SourceCheckpointStageRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stages: [String] = []

    func append(_ stage: String) {
        lock.lock()
        stages.append(stage)
        lock.unlock()
    }

    func reset() {
        lock.lock()
        stages.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    var snapshot: [String] {
        lock.lock()
        defer { lock.unlock() }
        return stages
    }
}

private struct SourceCheckpointHarness {
    let fixture: SourceCacheTelemetryFixture
    let generation: QwenOfficialSourceConversationGenerationSession
    let session: RealInferenceSession
    let client: RealInferenceClient
    let codec: QwenChatCodec

    static func make(
        hooks: QwenOfficialSourceTransactionHooks = .none
    ) async throws -> Self {
        let fixture = try SourceCacheTelemetryFixture.make()
        do {
            let context = try MetalContext()
            let model = try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: fixture.registrationURL,
                context: context,
                residencyBudgetBytes: fixture.residencyBudgetBytes)
            let generation = try await QwenOfficialSourceConversationGenerationSession(
                fixtureModel: model, context: context, maxContext: 32,
                expertSlotCount: fixture.expertSlotCount, hooks: hooks)
            let session = RealInferenceSession()
            // Keep the host checkpoint policy above the tiny runner's context so
            // the assessment exercises metadata sizing rather than capacity.
            let checkpointKey = SessionLoadKey(
                directory: fixture.sessionKey.directory, maxContext: 1_024,
                options: fixture.sessionKey.options,
                forceLogitsHead: fixture.sessionKey.forceLogitsHead)
            try await session.installQwenOfficialSourceFixture(
                model: model, generation: generation,
                key: checkpointKey, context: context)
            let tokenizer = try QwenTokenizer.loadOfficialSidecar(
                from: sourceCheckpointTokenizerDirectory)
            let codec = QwenChatCodec(tokenizer: tokenizer)
            return Self(fixture: fixture, generation: generation,
                        session: session, client: RealInferenceClient(session: session),
                        codec: codec)
        } catch {
            fixture.remove()
            throw error
        }
    }

    func generateToolCall(query: String) async throws -> ParsedToolCall {
        let recorder = SourceToolCallRecorder()
        let steps = sourcePreparedToolSteps(query: query)
        let result = try await generation.generatePreparedToolTurn(
            promptTokenIDs: [0, 1], steps: steps, tools: [sourceLookupTool()],
            config: sourceToolConfig(maxNewTokens: steps.count),
            onEvent: { event in
                if case .toolCall(let call) = event { recorder.record(call) }
            })
        guard result.reason == .toolCalls, let observed = recorder.last else {
            throw SourceCheckpointFixtureFailure.noToolCall
        }
        return observed
    }

    func synchronize(call: ParsedToolCall) async throws {
        try await session.synchronizeQwenOfficialSourceFixtureTurn(
            codec: codec, tools: [sourceLookupTool()], toolCalls: [call])
    }
}

private func sourceLookupTool() -> ModelChatToolDefinition {
    ModelChatToolDefinition(function: .init(
        name: "lookup", description: "synthetic lookup",
        parameters: .object([
            .init("type", .string("object")),
            .init("properties", .object([
                .init("query", .object([
                    .init("type", .string("string")),
                    .init("enum", .array([.string("snow")])),
                ])),
            ])),
            .init("required", .array([.string("query")])),
            .init("additionalProperties", .bool(false)),
        ])))
}

private func sourcePreparedToolSteps(query: String) -> [QwenOfficialSourcePreparedToolStep] {
    let text = "<tool_call>\n<function=lookup>\n<parameter=query>\n\(query)\n"
        + "</parameter>\n</function>\n</tool_call>"
    let bytes = Array(text.utf8)
    var steps: [QwenOfficialSourcePreparedToolStep] = []
    for start in stride(from: 0, to: bytes.count, by: 32) {
        let end = min(start + 32, bytes.count)
        steps.append(.token(
            id: Int32(steps.count % 4),
            decoded: String(decoding: bytes[start..<end], as: UTF8.self)))
    }
    steps.append(.modelEOS(id: 4, tokenizerTail: ""))
    return steps
}

private func sourceToolConfig(maxNewTokens: Int) -> GenerationConfig {
    var config = GenerationConfig(
        maxNewTokens: maxNewTokens, temperature: 0, topK: nil, topP: nil,
        repetitionPenalty: 1, seed: 0, stopStrings: [], extraStopTokens: [])
    config.logitTransform = .raw
    return config
}

private func sourceCheckpointRequest(
    call: ParsedToolCall, record: String, commit: Bool,
    force: Bool = true, trigger: DecodeContextCheckpointTrigger? = nil
) throws -> DecodeContextCheckpointRequest {
    DecodeContextCheckpointRequest(
        checkpointID: UUID(), sourceEpoch: UUID(), sourceTurnIndex: 0,
        replacementEpoch: UUID(),
        pendingCall: DecodeToolCall(
            id: call.id, name: call.name,
            argumentsJSON: try call.arguments.encoded()),
        result: DecodeToolResult(
            callID: call.id, name: call.name, content: "synthetic result"),
        record: record, commit: commit, force: force, trigger: trigger,
        generationAllowance: 128, finalAnswerAllowance: 128,
        permitsScreenshot: false)
}

private func sourceAssessmentQwenArchitecture() -> QwenArchConfig {
    let layers: [GTurboQwenLayerTypeV2] = (0..<40).map {
        ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
    }
    let wire = GTurboQwenArchitectureV2(
        hiddenSize: 2_048, numLayers: 40, layerTypes: layers,
        numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256,
        attentionOutputGate: true, linearConvolutionKernel: 4,
        linearKeyHeads: 16, linearKeyHeadDimension: 128,
        linearValueHeads: 32, linearValueHeadDimension: 128,
        recurrentStateType: .fp32, partialRotaryFactor: 0.25,
        ropeTheta: 10_000_000, mropeInterleaved: true,
        mropeSections: [11, 11, 10], numberOfExperts: 256,
        expertsPerToken: 8, routedExpertIntermediateSize: 512,
        sharedExpertIntermediateSize: 512, vocabularySize: 248_320,
        tiedWordEmbeddings: false, hiddenActivation: "silu",
        bosTokenID: 248_044, eosTokenID: 248_044, imageTokenID: 248_056,
        videoTokenID: 248_057, visionStartTokenID: 248_053,
        visionEndTokenID: 248_054)
    return QwenArchConfig(wire: wire)
}

private func sourceAssessmentVisionFeatures(
    context: MetalContext, rows: Int, marker: Float, digest: String
) throws -> QwenVisionFeatures {
    let grid = try QwenVisionGrid(
        temporal: 1, height: rows == 1 ? 2 : 4, width: rows == 1 ? 2 : 4)
    let position = try QwenMRoPEPosition(temporal: 0, height: 0, width: 0)
    let profile = GTurboQwenVisionProcessorProfileV2(
        processorClass: "Qwen3VLProcessor",
        imageProcessorType: "Qwen2VLImageProcessorFast",
        patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
    return try QwenVisionFeatures(
        device: context.device,
        features: [Float](repeating: marker, count: rows * 32),
        positions: Array(repeating: position, count: rows),
        imageDigest: String(repeating: digest, count: 64),
        processorDigest: String(repeating: "c", count: 64),
        profile: profile, grid: grid, hiddenSize: 32)
}

private let sourceCheckpointTokenizerDirectory = URL(
    fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent(
        "scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0",
        isDirectory: true)

private enum SourceCacheTelemetryFailure: Error {
    case protectedRead
}

/// The app test target cannot import the core test target's fixture module, so
/// this helper emits only the two real safetensors shards needed by the source
/// admission seam. Every tensor is deterministic BF16 data and no model bytes
/// are copied from the original checkpoint.
private struct SourceCacheTelemetryFixture {
    private struct Tensor {
        let name: String
        let shape: [Int]
        let words: [UInt16]

        var byteCount: UInt64 { UInt64(words.count * MemoryLayout<UInt16>.stride) }
    }

    let root: URL
    let registrationURL: URL
    let sessionKey: SessionLoadKey
    let residencyBudgetBytes: UInt64
    let expectedLayerCacheBytes: UInt64
    let expectedCacheBytes: UInt64
    let expertSlotCount = Self.topK

    private static let identity = OfficialQwenSourceIdentity.pinned
    private static let hidden = 8
    private static let vocabulary = 5
    private static let experts = 9
    private static let topK = 8
    private static let routedIntermediate = 2

    func remove() { try? FileManager.default.removeItem(at: root) }

    static func make() throws -> Self {
        let manager = FileManager.default
        // OfficialSourceHandle rejects source roots that retain macOS's
        // /var -> /private/var alias. Use the same realpath pattern as the
        // accepted source fixture rather than Foundation URL normalization.
        var canonicalTemporaryParent = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(manager.temporaryDirectory.path, &canonicalTemporaryParent) != nil else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let temporaryParentBytes = canonicalTemporaryParent.prefix { $0 != 0 }
            .map { UInt8(bitPattern: $0) }
        let temporaryParent = URL(
            fileURLWithPath: String(decoding: temporaryParentBytes, as: UTF8.self),
            isDirectory: true)
        let root = temporaryParent.appendingPathComponent(
            "source-cache-telemetry-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: false,
                                    attributes: [.posixPermissions: 0o700])
        do {
            let sourceRoot = root.appendingPathComponent("source", isDirectory: true)
            try manager.createDirectory(at: sourceRoot, withIntermediateDirectories: false,
                                        attributes: [.posixPermissions: 0o700])
            var canonicalSourceRoot = [CChar](repeating: 0, count: Int(PATH_MAX))
            guard realpath(sourceRoot.path, &canonicalSourceRoot) != nil else {
                throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
            }
            let sourcePathBytes = canonicalSourceRoot.prefix { $0 != 0 }
                .map { UInt8(bitPattern: $0) }
            let canonicalSourceRootURL = URL(
                fileURLWithPath: String(decoding: sourcePathBytes, as: UTF8.self),
                isDirectory: true)
            let registrationParent = root.appendingPathComponent("models", isDirectory: true)
            try manager.createDirectory(at: registrationParent, withIntermediateDirectories: false,
                                        attributes: [.posixPermissions: 0o700])
            let registrationURL = registrationParent.appendingPathComponent(
                "fixture.gturbo", isDirectory: true)
            let tensors = makeTensors()
            let routedNames = Set(tensors.filter { $0.name.contains("mlp.experts.") }
                .map(\.name))
            var mapping: [String: String] = [:]
            let shardNames = [identity.shards[0].filename, identity.shards[1].filename]
            for tensor in tensors {
                mapping[tensor.name] = routedNames.contains(tensor.name)
                    ? shardNames[1] : shardNames[0]
            }
            try configJSON().write(
                to: canonicalSourceRootURL.appendingPathComponent("config.json"),
                options: .withoutOverwriting)
            let index: [String: Any] = ["weight_map": mapping]
            try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys]).write(
                to: canonicalSourceRootURL.appendingPathComponent("model.safetensors.index.json"),
                options: .withoutOverwriting)
            let resident = tensors.filter { !routedNames.contains($0.name) }
            let routed = tensors.filter { routedNames.contains($0.name) }
            try writeShard(resident, to: canonicalSourceRootURL.appendingPathComponent(shardNames[0]))
            try writeShard(routed, to: canonicalSourceRootURL.appendingPathComponent(shardNames[1]))

            let descriptor = try OfficialSourceDescriptor(
                repository: identity.repository, revision: identity.revision,
                storageProfile: identity.storageProfile, sidecarSHA256: identity.sidecarSHA256,
                shards: identity.shards.map {
                    OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
                }, sourceRoot: canonicalSourceRootURL.path)
            _ = try OfficialSourceRegistration.register(
                markerData: JSONEncoder().encode(descriptor), at: registrationURL)

            let residentBF16Bytes = resident.reduce(UInt64(0)) { $0 + $1.byteCount }
            // QwenBF16Weights keeps matrices as BF16 and decodes its vectors
            // into FP32 storage. The raw vector words are therefore not an
            // additional resident allocation.
            let decodedVectorElements: UInt64 = 76
            let rawVectorBytes = decodedVectorElements * UInt64(MemoryLayout<UInt16>.stride)
            let decodedVectorBytes = decodedVectorElements * UInt64(MemoryLayout<Float>.stride)
            let residentBytes = residentBF16Bytes - rawVectorBytes + decodedVectorBytes
            let pairBytes = UInt64(hidden * routedIntermediate * 6)
            let expectedLayerCacheBytes = pairBytes * UInt64(topK)
            let expectedCacheBytes = expectedLayerCacheBytes * 2
            let budget = residentBytes + expectedCacheBytes
            let key = SessionLoadKey(
                directory: registrationURL, maxContext: 8,
                options: AppRuntimeOptions(), forceLogitsHead: false)
            return Self(root: root, registrationURL: registrationURL, sessionKey: key,
                        residencyBudgetBytes: budget,
                        expectedLayerCacheBytes: expectedLayerCacheBytes,
                        expectedCacheBytes: expectedCacheBytes)
        } catch {
            try? manager.removeItem(at: root)
            throw error
        }
    }

    private static func configJSON() throws -> Data {
        let root: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": [
                "model_type": "qwen3_5_moe_text", "dtype": "bfloat16",
                "mamba_ssm_dtype": "float32", "hidden_size": hidden,
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "linear_attention"],
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "attn_output_gate": true, "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 1, "linear_key_head_dim": 2,
                "linear_num_value_heads": 1, "linear_value_head_dim": 2,
                "partial_rotary_factor": 0.5,
                "rope_parameters": ["rope_theta": 10_000, "mrope_section": [1]],
                "num_experts": experts, "num_experts_per_tok": topK,
                "moe_intermediate_size": routedIntermediate,
                "shared_expert_intermediate_size": 2, "vocab_size": vocabulary,
                "tie_word_embeddings": false, "hidden_act": "silu",
                "rms_norm_eps": 1e-6, "bos_token_id": 0, "eos_token_id": 4,
            ],
            "image_token_id": 3, "video_token_id": 2,
            "vision_start_token_id": 0, "vision_end_token_id": 4,
        ]
        return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }

    private static func makeTensors() -> [Tensor] {
        var result: [Tensor] = []
        func add(_ name: String, _ shape: [Int], fill: UInt16 = 0) {
            result.append(Tensor(name: name, shape: shape,
                                 words: [UInt16](repeating: fill, count: shape.reduce(1, *))))
        }
        add("model.language_model.embed_tokens.weight", [vocabulary, hidden])
        add("lm_head.weight", [vocabulary, hidden])
        add("model.language_model.norm.weight", [hidden], fill: 0x3f80)
        for layer in 0..<2 {
            let prefix = "model.language_model.layers.\(layer)."
            add(prefix + "input_layernorm.weight", [hidden], fill: 0x3f80)
            add(prefix + "post_attention_layernorm.weight", [hidden], fill: 0x3f80)
            var router = [UInt16](repeating: 0, count: experts * hidden)
            for expert in 0..<topK { router[expert * hidden] = 0x3f80 }
            router[(experts - 1) * hidden] = 0xbf80
            result.append(Tensor(name: prefix + "mlp.gate.weight", shape: [experts, hidden], words: router))
            add(prefix + "mlp.shared_expert.gate_proj.weight", [2, hidden])
            add(prefix + "mlp.shared_expert.up_proj.weight", [2, hidden])
            add(prefix + "mlp.shared_expert.down_proj.weight", [hidden, 2])
            add(prefix + "mlp.shared_expert_gate.weight", [1, hidden])
            add(prefix + "mlp.experts.gate_up_proj", [experts, 2 * routedIntermediate, hidden])
            add(prefix + "mlp.experts.down_proj", [experts, hidden, routedIntermediate])
            if layer == 0 {
                add(prefix + "self_attn.q_proj.weight", [16, hidden])
                add(prefix + "self_attn.k_proj.weight", [4, hidden])
                add(prefix + "self_attn.v_proj.weight", [4, hidden])
                add(prefix + "self_attn.o_proj.weight", [hidden, 8])
                add(prefix + "self_attn.q_norm.weight", [4], fill: 0x3f80)
                add(prefix + "self_attn.k_norm.weight", [4], fill: 0x3f80)
            } else {
                let linear = prefix + "linear_attn."
                add(linear + "in_proj_qkv.weight", [6, hidden])
                add(linear + "in_proj_z.weight", [2, hidden])
                add(linear + "in_proj_b.weight", [1, hidden])
                add(linear + "in_proj_a.weight", [1, hidden])
                add(linear + "out_proj.weight", [hidden, 2])
                add(linear + "conv1d.weight", [6, 1, 4])
                add(linear + "norm.weight", [2], fill: 0x3f80)
                add(linear + "A_log", [1])
                add(linear + "dt_bias", [1])
            }
        }
        return result
    }

    private static func writeShard(_ tensors: [Tensor], to url: URL) throws {
        var payload = Data()
        var header: [String: [String: Any]] = [:]
        for tensor in tensors.sorted(by: { $0.name < $1.name }) {
            let start = payload.count
            payload.append(Data(repeating: 0, count: tensor.words.count * 2))
            header[tensor.name] = [
                "dtype": "BF16", "shape": tensor.shape,
                "data_offsets": [start, payload.count],
            ]
        }
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        headerData.append(Data(repeating: 0x20, count: (8 - headerData.count % 8) % 8))
        var headerLength = UInt64(headerData.count).littleEndian
        var file = Data(bytes: &headerLength, count: MemoryLayout<UInt64>.size)
        file.append(headerData)
        file.append(payload)
        try file.write(to: url, options: .withoutOverwriting)
    }
}
