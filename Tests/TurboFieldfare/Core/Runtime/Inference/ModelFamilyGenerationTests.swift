import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite(.serialized) struct ModelFamilyGenerationTests {
    @Test func tinyQwenFacadeGeneratesWithVerifiedIdentityAndClampedLimit() async throws {
        let session = try QwenP18FixtureSupport.session(maxContext: 4)
        #expect(session.family == .qwen3_6)
        #expect(session.verifiedIdentity?.family == .qwen3_6)

        let events = GenerationEventRecorder()
        let result = try await session.generate(
            .init(
                prompt: .raw("fixture prompt"),
                config: .init(
                    maxNewTokens: 100, temperature: 0, topK: nil, topP: nil,
                    repetitionPenalty: 1, seed: 0, stopStrings: [], extraStopTokens: [])),
            onEvent: events.append)

        #expect(result.promptTokens == 3)
        #expect(result.newTokens == 1)
        #expect(result.reason == .maxTokens || result.reason == .eos)
        let values = events.values
        guard case .prefill(_, let total) = values.first else {
            Issue.record("the facade did not report prefill progress")
            return
        }
        let progress = values.compactMap { event -> Int? in
            guard case .prefill(let progress, let maximum) = event else { return nil }
            #expect(maximum == 3)
            return progress
        }
        #expect(progress == [1, 2, 3])
        #expect(total == 3)
    }

    @Test func realCodecPreservesToolSchemaAndArgumentObjectOrder() throws {
        let codec = try QwenP18FixtureSupport.codec()
        let schema = ModelChatJSONValue.object([
            .init("type", .string("object")),
            .init("properties", .object([
                .init("z", .object([
                    .init("type", .string("integer")),
                ])),
                .init("a", .object([
                    .init("type", .string("string")),
                ])),
            ])),
            .init("required", .array([.string("z")])),
        ])
        let tool = ModelChatToolDefinition(
            function: .init(name: "lookup", description: "look it up", parameters: schema))
        let messages = [
            ModelChatMessage(role: .user, content: "find it"),
            ModelChatMessage(
                role: .assistant,
                content: "",
                toolCalls: [.init(
                    name: "lookup",
                    arguments: .object([
                        .init("z", .integer(1)), .init("a", .string("x")),
                    ]))]),
        ]
        let rendered = try codec.renderPrompt(
            messages: messages,
            tools: [tool],
            options: .init(enableThinking: false))
        #expect(rendered.range(of: "\"name\": \"lookup\"") != nil)
        #expect(rendered.range(of: "\"description\": \"look it up\"") != nil)
        #expect(rendered.range(of: "\"parameters\": {\"type\": \"object\"") != nil)
        let zProperty = try #require(rendered.range(of: "\"z\": {\"type\": \"integer\"}"))
        let aProperty = try #require(rendered.range(of: "\"a\": {\"type\": \"string\"}"))
        #expect(zProperty.lowerBound < aProperty.lowerBound)
        let zArgument = try #require(rendered.range(of: "<parameter=z>"))
        let aArgument = try #require(rendered.range(of: "<parameter=a>"))
        #expect(zArgument.lowerBound < aArgument.lowerBound)
    }

    @Test func realCodecNormalizesAndExpandsOrderedImageFramesOnce() throws {
        let context = try MetalContext()
        let codec = try QwenP18FixtureSupport.codec()
        let architecture = QwenTestArchitecture.qwen36
        let firstID = "image-first"
        let secondID = "image-second"
        let messages = [ModelChatMessage(
            role: .user,
            content: .parts([
                .text("before"), .image(.init(id: firstID)),
                .text("between"), .image(.init(id: secondID)), .text("after"),
            ]))]
        let encoded = try codec.encodePrompt(
            messages: messages, tools: [], options: .init(enableThinking: false))
        let normalized = try normalizeQwenCodecImageFrames(
            encoded, architecture: architecture)
        let imageToken = Int32(architecture.imageTokenID)
        #expect(normalized.filter { $0 == imageToken }.count == 2)
        #expect(!normalized.contains(Int32(architecture.visionStartTokenID)))
        #expect(!normalized.contains(Int32(architecture.visionEndTokenID)))

        let first = try QwenP18FixtureSupport.qwenFeatures(context: context, marker: 1)
        let second = try QwenP18FixtureSupport.qwenFeatures(context: context, marker: 2)
        let expanded = try MultimodalPromptRenderer.expandingQwenImageTokens(
            normalized, features: [first, second], architecture: architecture)
        let expectedCount = try qwenExpandedPromptTokenCount(
            normalizedTokenIDs: normalized,
            imageMergedRows: [first.tokenCount, second.tokenCount],
            imageTokenID: imageToken)
        #expect(expanded.effectiveTokenIDs.count == expectedCount)
        #expect(expanded.imageSpans.map { $0.features.owner.imageDigest }
            == [first.owner.imageDigest, second.owner.imageDigest])
        let start = Int32(architecture.visionStartTokenID)
        let end = Int32(architecture.visionEndTokenID)
        #expect(expanded.effectiveTokenIDs.filter { $0 == start }.count == 2)
        #expect(expanded.effectiveTokenIDs.filter { $0 == end }.count == 2)
        for span in expanded.imageSpans {
            #expect(expanded.effectiveTokenIDs[span.tokenRange.lowerBound - 1] == start)
            #expect(expanded.effectiveTokenIDs[span.tokenRange.upperBound] == end)
        }
    }

    @Test func tinyTextFixtureRejectsImageOverridesBeforeVisionAdmission() async throws {
        let session = try QwenP18FixtureSupport.session()
        let message = ModelChatMessage(
            role: .user,
            content: .parts([
                .text("describe"), .image(.init(id: "image-1")),
            ]))
        do {
            _ = try await session.generate(
                .init(
                    prompt: .chat(messages: [message], tools: [], thinking: .automatic),
                    imagesByID: ["image-1": URL(fileURLWithPath: "/does/not/exist.png")],
                    config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                                  repetitionPenalty: 1, seed: 0)),
                onEvent: { _ in })
            Issue.record("the tiny text fixture accepted an image override")
        } catch let error as ModelFamilyGenerationError {
            #expect(error == .unsupportedInput(
                "tiny text fixture prompt overrides cannot carry images"))
        }
    }

    @Test func unsupportedVideoAndAudioAreRejectedByTheFamilyCodec() async throws {
        let session = try QwenP18FixtureSupport.session()
        for part in [ModelChatContentPart.video(.init(id: "v")), .audio(.init(id: "a"))] {
            do {
                _ = try await session.generate(
                    .init(
                        prompt: .chat(
                            messages: [.init(role: .user, content: .parts([part]))],
                            tools: [], thinking: .automatic),
                        config: .init(maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
                                      repetitionPenalty: 1, seed: 0)),
                    onEvent: { _ in })
                Issue.record("unsupported media was accepted: \(part)")
            } catch {
                #expect(String(describing: error).localizedCaseInsensitiveContains(
                    part == .video(.init(id: "v")) ? "video" : "audio"))
            }
        }
    }

    @Test func overlappingGenerationReportsBusyAtDeterministicGate() async throws {
        let gate = BusyHookGate()
        let session = try QwenP18FixtureSupport.session(
            maxContext: 32, fixtureAfterBusyAcquired: { await gate.enter() })
        let request = QwenP18FixtureSupport.request(maxNewTokens: 8)
        let first = generationTask(session: session, request: request)
        defer {
            first.cancel()
            Task { await gate.releaseFirst() }
        }
        try await bounded { await gate.waitForFirst() }
        let second = generationTask(session: session, request: request)
        defer { second.cancel() }
        let secondResult = try await bounded(
            second,
            cleanup: {
                first.cancel()
                await gate.releaseFirst()
            })
        await gate.releaseFirst()
        let firstResult = try await bounded(first)
        guard case .failure(let error) = secondResult else {
            Issue.record("overlapping generation unexpectedly succeeded")
            return
        }
        #expect((error as? ModelFamilyGenerationError) == .busy)
        guard case .success = firstResult else {
            Issue.record("the gated first generation did not finish")
            return
        }
    }

    @Test func busyStateClearsAfterValidationFailure() async throws {
        let gate = BusyHookGate()
        let session = try QwenP18FixtureSupport.session(
            maxContext: 32, fixtureAfterBusyAcquired: { await gate.enter() })
        let invalid = generationTask(
            session: session,
            request: QwenP18FixtureSupport.request(maxNewTokens: 0))
        defer { invalid.cancel() }
        try await bounded { await gate.waitForFirst() }
        await gate.releaseFirst()
        let invalidResult = try await bounded(invalid)
        guard case .failure = invalidResult else {
            Issue.record("invalid generation unexpectedly succeeded")
            return
        }
        let valid = generationTask(
            session: session,
            request: QwenP18FixtureSupport.request(maxNewTokens: 1))
        defer { valid.cancel() }
        let validResult = try await bounded(valid)
        guard case .success = validResult else {
            Issue.record("busy state was not cleared after validation failure")
            return
        }
    }

    @Test func busyStateClearsAfterCancellation() async throws {
        let gate = BusyHookGate()
        let session = try QwenP18FixtureSupport.session(
            maxContext: 32, fixtureAfterBusyAcquired: { await gate.enter() })
        let first = generationTask(
            session: session,
            request: QwenP18FixtureSupport.request(maxNewTokens: 1))
        defer {
            first.cancel()
            Task { await gate.releaseFirst() }
        }
        try await bounded { await gate.waitForFirst() }
        first.cancel()
        await gate.releaseFirst()
        let cancelled = try await bounded(first)
        guard case .failure(let error) = cancelled else {
            Issue.record("cancelled generation unexpectedly succeeded")
            return
        }
        #expect(error is CancellationError)
        let valid = generationTask(
            session: session,
            request: QwenP18FixtureSupport.request(maxNewTokens: 1))
        defer { valid.cancel() }
        let validResult = try await bounded(valid)
        guard case .success = validResult else {
            Issue.record("busy state was not cleared after cancellation")
            return
        }
    }

    @Test func toolCallsPublishOnlyAfterModelEOSAndCompleteParserFrame() throws {
        let terminations: [QwenGenerationTermination] = [
            .modelEOS, .tokenStop, .maxTokens, .stopString, .cancelled,
        ]
        for termination in terminations {
            var decoder = QwenStructuredAssistantDecoder(
                tools: [QwenP18FixtureSupport.tool()],
                startsInThoughtChannel: false,
                idGenerator: { "host-id" })
            _ = try decoder.consume(QwenP18FixtureSupport.toolFrame())
            var events: [StructuredAssistantEvent] = []
            try finalizeQwenStructuredTurn(
                decoder: &decoder,
                tokenizerTail: "",
                termination: termination,
                publish: { events.append($0) })
            let toolCalls = events.compactMap { event -> ParsedToolCall? in
                guard case .toolCall(let call) = event else { return nil }
                return call
            }
            #expect(toolCalls.count == (termination == .modelEOS ? 1 : 0))
            if termination == .modelEOS {
                #expect(toolCalls[0].id == "host-id")
                #expect(toolCalls[0].name == "lookup")
            }
        }
    }

    @Test func cancelledFinalizerChecksCancellationBeforeToolPublication() async throws {
        let gate = AsyncGate()
        let published = StructuredAssistantEventRecorder()
        let task = Task { () throws -> [StructuredAssistantEvent] in
            await gate.wait()
            var decoder = QwenStructuredAssistantDecoder(
                tools: [QwenP18FixtureSupport.tool()],
                startsInThoughtChannel: false,
                idGenerator: { "host-id" })
            _ = try decoder.consume(QwenP18FixtureSupport.toolFrame())
            var events: [StructuredAssistantEvent] = []
            try finalizeQwenStructuredTurn(
                decoder: &decoder,
                tokenizerTail: "",
                termination: .modelEOS,
                publish: {
                    published.append($0)
                    events.append($0)
                })
            return events
        }
        task.cancel()
        await gate.signal()
        let result = await task.result
        #expect(published.values.isEmpty)
        guard case .failure(let error) = result else {
            Issue.record("cancelled finalizer unexpectedly published a tool call")
            return
        }
        #expect(error is CancellationError)
    }
}

/// Shared P18 fixture assembly. It reads the already-present tokenizer
/// sidecars directly and loads only the tiny embedded model records.
enum QwenP18FixtureSupport {
    static let tinyPromptTokenIDs: [Int32] = [1, 4, 7]

    static func request(maxNewTokens: Int) -> ModelFamilyGenerationRequest {
        .init(
            prompt: .raw("fixture prompt"),
            config: .init(
                maxNewTokens: maxNewTokens, temperature: 0, topK: nil, topP: nil,
                repetitionPenalty: 1, seed: 0))
    }

    static func tool() -> ModelChatToolDefinition {
        let schema = ModelChatJSONValue.object([
            .init("type", .string("object")),
            .init("properties", .object([
                .init("query", .object([.init("type", .string("string"))])),
            ])),
            .init("required", .array([.string("query")])),
            .init("additionalProperties", .bool(false)),
        ])
        return ModelChatToolDefinition(function: .init(
            name: "lookup", description: "decoder test tool", parameters: schema))
    }

    static func toolFrame() -> String {
        "<tool_call>\n<function=lookup>\n<parameter=query>\n"
            + "snow\n</parameter>\n</function>\n</tool_call>"
    }

    static func codec() throws -> QwenChatCodec {
        QwenChatCodec(tokenizer: try QwenTokenizer.loadOfficialSidecar(from: sourceDirectory))
    }

    static func session(
        maxContext: Int = 32,
        fixtureAfterBusyAcquired: (@Sendable () async -> Void)? = nil
    ) throws -> ModelFamilyGenerationSession {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let bundle = LoadedModelFamilyBundle(
            runtime: .qwen(model),
            family: .qwen3_6,
            verifiedIdentity: try identity(),
            qwenCodec: try codec())
        return ModelFamilyGenerationSession(
            bundle: bundle,
            context: context,
            modelDirectoryURL: sourceDirectory,
            maxContext: maxContext,
            runtimeConfiguration: .production,
            fixturePromptTokenIDs: tinyPromptTokenIDs,
            fixtureAfterBusyAcquired: fixtureAfterBusyAcquired)
    }

    static func qwenFeatures(context: MetalContext, marker: Float) throws -> QwenVisionFeatures {
        let grid = try QwenVisionGrid(temporal: 1, height: 2, width: 2)
        let position = try QwenMRoPEPosition(temporal: 0, height: 0, width: 0)
        let profile = GTurboQwenVisionProcessorProfileV2(
            processorClass: "Qwen3VLProcessor",
            imageProcessorType: "Qwen2VLImageProcessorFast",
            patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
        return try QwenVisionFeatures(
            device: context.device,
            features: [Float](
                repeating: marker,
                count: QwenVisionConfig.official.outputHiddenSize),
            positions: [position],
            imageDigest: String(repeating: marker == 1 ? "a" : "b", count: 64),
            processorDigest: String(repeating: "c", count: 64),
            profile: profile,
            grid: grid)
    }

    static func identity() throws -> LoadedRuntimeIdentity {
        let digest = String(repeating: "a", count: 64)
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2_048,
            numLayers: 40,
            layerTypes: (0..<40).map {
                ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
            },
            numAttentionHeads: 16,
            numKeyValueHeads: 2,
            headDimension: 256,
            attentionOutputGate: true,
            linearConvolutionKernel: 4,
            linearKeyHeads: 16,
            linearKeyHeadDimension: 128,
            linearValueHeads: 32,
            linearValueHeadDimension: 128,
            recurrentStateType: .fp32,
            partialRotaryFactor: 0.25,
            ropeTheta: 10_000_000,
            mropeInterleaved: true,
            mropeSections: [11, 11, 10],
            numberOfExperts: 256,
            expertsPerToken: 8,
            routedExpertIntermediateSize: 512,
            sharedExpertIntermediateSize: 512,
            vocabularySize: 248_320,
            tiedWordEmbeddings: false,
            hiddenActivation: "silu",
            bosTokenID: 248_044,
            eosTokenID: 248_044,
            imageTokenID: 248_056,
            videoTokenID: 248_057,
            visionStartTokenID: 248_053,
            visionEndTokenID: 248_054)
        let groups = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState
                ? GTurboQuantizationGroupV2(category: category, storage: .fp32)
                : GTurboQuantizationGroupV2(
                    category: category, storage: .affineInt4, groupSize: 64,
                    scaleType: "bf16", biasType: "bf16")
        }
        let manifest = GTurboManifestV2(
            family: .qwen3_6,
            requiredFeatures: [.familyDispatch, .verifiedIdentity,
                               .qwenHybridAttention, .qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository,
            architecture: .qwen3_6(architecture),
            provenance: .init(
                sourceRepository: GTurboFormatV2.qwenRepository,
                sourceRevision: GTurboFormatV2.qwenRevision,
                sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
                sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
                quantizationPolicySHA256: digest),
            quantization: groups,
            ignoredTensors: [],
            files: [:],
            tensorRegions: [],
            expertsPerLayer: 256,
            numLayers: 40,
            expertStride: 16_384)
        let descriptor = try InstalledModelDescriptor.validated(
            textManifest: manifest, textManifestSHA256: digest)
        return LoadedRuntimeIdentity(descriptor: descriptor)
    }

    private static var sourceDirectory: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0",
                isDirectory: true)
    }
}

private final class GenerationEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [ModelFamilyGenerationEvent] = []

    func append(_ event: ModelFamilyGenerationEvent) {
        lock.withLock { recorded.append(event) }
    }

    var values: [ModelFamilyGenerationEvent] {
        lock.withLock { recorded }
    }
}

private final class StructuredAssistantEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [StructuredAssistantEvent] = []

    func append(_ event: StructuredAssistantEvent) {
        lock.withLock { recorded.append(event) }
    }

    var values: [StructuredAssistantEvent] {
        lock.withLock { recorded }
    }
}

private actor AsyncGate {
    private var signaled = false
    private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]

    func signal() {
        guard !signaled else { return }
        signaled = true
        let pending = waiters.values
        waiters.removeAll(keepingCapacity: false)
        for waiter in pending { waiter.resume() }
    }

    func wait() async {
        if signaled { return }
        let id = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if signaled {
                    continuation.resume()
                } else {
                    waiters[id] = continuation
                }
            }
        } onCancel: {
            Task { await self.cancel(id: id) }
        }
    }

    private func cancel(id: UUID) {
        waiters.removeValue(forKey: id)?.resume()
    }
}

private actor BusyHookGate {
    private let acquired = AsyncGate()
    private let release = AsyncGate()
    private var invocationCount = 0

    func enter() async {
        invocationCount += 1
        if invocationCount == 1 {
            await acquired.signal()
            await release.wait()
        }
    }

    func waitForFirst() async { await acquired.wait() }
    func releaseFirst() async { await release.signal() }
}

private enum TestWaitError: Error, Sendable {
    case timedOut
}

private enum BoundedOutcome<T: Sendable>: Sendable {
    case value(T)
    case timedOut
}

private actor OneShotRace<T: Sendable> {
    private var completed = false
    private var result: BoundedOutcome<T>?
    private var continuation: CheckedContinuation<BoundedOutcome<T>, Never>?

    func wait() async -> BoundedOutcome<T> {
        if let result { return result }
        return await withCheckedContinuation { continuation in
            if let result {
                continuation.resume(returning: result)
            } else {
                self.continuation = continuation
            }
        }
    }

    func finish(_ result: BoundedOutcome<T>) {
        guard !completed else { return }
        completed = true
        self.result = result
        let continuation = self.continuation
        self.continuation = nil
        continuation?.resume(returning: result)
    }
}

private func bounded<T: Sendable>(
    _ operation: @escaping @Sendable () async -> T,
    cleanup: @escaping @Sendable () async -> Void = {}
) async throws -> T {
    try await bounded(Task { await operation() }, cleanup: cleanup)
}

private func bounded<T: Sendable>(
    _ task: Task<T, Never>,
    cleanup: @escaping @Sendable () async -> Void = {}
) async throws -> T {
    let race = OneShotRace<T>()
    let completion = Task {
        let value = await task.value
        await race.finish(.value(value))
    }
    let timeout = Task {
        do {
            try await Task.sleep(nanoseconds: 5_000_000_000)
            await race.finish(.timedOut)
        } catch is CancellationError {
            // The operation completed before the timeout.
        }
    }
    let outcome = await race.wait()
    timeout.cancel()
    switch outcome {
    case .value(let value):
        _ = await completion.value
        return value
    case .timedOut:
        task.cancel()
        completion.cancel()
        await cleanup()
        throw TestWaitError.timedOut
    }
}

private func generationTask(
    session: ModelFamilyGenerationSession,
    request: ModelFamilyGenerationRequest
) -> Task<Result<ModelFamilyGenerationResult, Error>, Never> {
    Task {
        do {
            return .success(try await session.generate(request) { _ in })
        } catch {
            return .failure(error)
        }
    }
}
