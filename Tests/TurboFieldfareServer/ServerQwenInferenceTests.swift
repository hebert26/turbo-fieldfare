import Foundation
import NIOCore
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

private enum P19QwenDriverPlan: Sendable {
    case success(ModelFamilyGenerationResult, [ModelFamilyGenerationEvent])
    case cancelled
    case contextOverflow
    case failed
}

private actor P19QwenScriptedDriver: ServerQwenGenerationDriver {
    let preflightValue: ModelFamilyGenerationPreflight
    let plan: P19QwenDriverPlan
    private(set) var preflightRequests: [ModelFamilyGenerationRequest] = []
    private(set) var generationRequests: [ModelFamilyGenerationRequest] = []

    init(
        preflight: ModelFamilyGenerationPreflight = .init(promptTokens: 17, imageCount: 0),
        plan: P19QwenDriverPlan
    ) {
        self.preflightValue = preflight
        self.plan = plan
    }

    func preflight(
        _ request: ModelFamilyGenerationRequest
    ) async throws -> ModelFamilyGenerationPreflight {
        preflightRequests.append(request)
        return preflightValue
    }

    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @escaping @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        generationRequests.append(request)
        switch plan {
        case .success(let result, let events):
            for event in events { onEvent(event) }
            return result
        case .cancelled:
            throw CancellationError()
        case .contextOverflow:
            throw ModelFamilyGenerationError.contextOverflow(
                prompt: 17, maxNew: 8, maximum: 20)
        case .failed:
            throw P19QwenDriverError.failed
        }
    }
}

private enum P19QwenDriverError: Error, Sendable {
    case failed
}

private final class P19QwenEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [ServerInferenceEvent] = []

    func append(_ event: ServerInferenceEvent) {
        lock.lock()
        values.append(event)
        lock.unlock()
    }

    var events: [ServerInferenceEvent] {
        lock.lock()
        let copy = values
        lock.unlock()
        return copy
    }
}

@Suite("Server Qwen inference")
struct ServerQwenInferenceTests {
    @Test func qwenAdapterForwardsPromptImagesThinkingAndUsage() async throws {
        let imageURL = try Self.makeFixtureFile(named: "first.bin", bytes: [1, 2, 3])
        defer { try? FileManager.default.removeItem(at: imageURL.deletingLastPathComponent()) }
        let call = ParsedToolCall(
            id: "call_000000000000000000000001",
            name: "lookup",
            arguments: .object([
                "z": .integer(1),
                "a": .string("kept")
            ]),
            argumentsJSON: #"{"z":1,"a":"kept"}"#)
        let result = ModelFamilyGenerationResult(
            reason: .eos, promptTokens: 17, newTokens: 4,
            prefillSeconds: 0.01, decodeSeconds: 0.02)
        let driver = P19QwenScriptedDriver(
            preflight: .init(promptTokens: 17, imageCount: 1),
            plan: .success(result, [
                .prefill(done: 1, total: 17),
                .text("ans"),
                .text("wer"),
                .toolCall(call)
            ]))
        let session = ServerQwenModelSession(driver: driver)
        let request = Self.validatedQwenRequest(
            prompt: .chat(
                messages: [ModelChatMessage(
                    role: .user,
                    content: .parts([
                        .text("look at this"),
                        .image(.init(id: "server-image-1"))
                    ]),
                    reasoningContent: "private thought")],
                tools: [Self.lookupTool()],
                thinking: .enabled),
            images: ["server-image-1": imageURL])
        let recorder = P19QwenEventRecorder()

        let completion = try await session.generate(request) {
            recorder.append($0)
        }

        #expect(session.requestFamily == .qwen3_6)
        #expect(completion.content == "answer")
        #expect(completion.toolCalls == [call])
        #expect(completion.finishReason == "tool_calls")
        #expect(completion.usage == OpenAIUsage(
            promptTokens: 17, completionTokens: 4, totalTokens: 21, cachedTokens: 0))
        #expect(recorder.events == [
            .content("ans"), .content("wer"), .toolCall(call)
        ])

        let expectedPrompt = try #require(request.qwenPrompt)
        let preflightRequest = try #require(await driver.preflightRequests.first)
        let generationRequest = try #require(await driver.generationRequests.first)
        #expect(preflightRequest.prompt == expectedPrompt)
        #expect(generationRequest.prompt == expectedPrompt)
        #expect(generationRequest.imagesByID == request.qwenImagesByID)
        #expect(generationRequest.visionResidency == .onDemand)
        #expect(generationRequest.config.maxNewTokens == request.generationConfig.maxNewTokens)
        #expect(generationRequest.config.temperature == request.generationConfig.temperature)
        #expect(generationRequest.config.topK == request.generationConfig.topK)
        #expect(generationRequest.config.topP == request.generationConfig.topP)
        #expect(generationRequest.config.repetitionPenalty == request.generationConfig.repetitionPenalty)
        #expect(generationRequest.config.seed == request.generationConfig.seed)
    }

    @Test func qwenAdapterUsesPreflightTokensAndMapsMaxBudgetToLength() async throws {
        let result = ModelFamilyGenerationResult(
            reason: .maxTokens, promptTokens: 23, newTokens: 8,
            prefillSeconds: 0.01, decodeSeconds: 0.02)
        let driver = P19QwenScriptedDriver(
            preflight: .init(promptTokens: 23, imageCount: 0),
            plan: .success(result, [.text("partial")]))
        let session = ServerQwenModelSession(driver: driver)
        let request = Self.validatedQwenRequest(
            prompt: .chat(
                messages: [ModelChatMessage(role: .user, content: "continue")],
                tools: [], thinking: .automatic))

        let prepared = try await session.prepare(request)
        #expect(prepared.promptTokenCount == 23)
        let completion = try await session.generate(prepared) { _ in }

        #expect(completion.content == "partial")
        #expect(completion.toolCalls.isEmpty)
        #expect(completion.finishReason == "length")
        #expect(completion.usage.promptTokens == 23)
        #expect(completion.usage.promptTokensDetails.cachedTokens == 0)
        #expect(await driver.preflightRequests.count == 1)
        #expect(await driver.generationRequests.count == 1)
    }

    @Test func qwenAdapterPreservesCancellationAndDriverFailures() async throws {
        let request = Self.validatedQwenRequest(
            prompt: .chat(
                messages: [ModelChatMessage(role: .user, content: "cancel")],
                tools: [], thinking: .automatic))

        do {
            let driver = P19QwenScriptedDriver(plan: .cancelled)
            let session = ServerQwenModelSession(driver: driver)
            _ = try await session.generate(request) { _ in }
            Issue.record("cancelled generation unexpectedly completed")
        } catch is CancellationError {
            // The adapter must preserve cancellation so the HTTP layer can close
            // an interrupted stream without turning it into a server failure.
        }

        do {
            let driver = P19QwenScriptedDriver(plan: .failed)
            let session = ServerQwenModelSession(driver: driver)
            _ = try await session.generate(request) { _ in }
            Issue.record("driver failure unexpectedly completed")
        } catch is P19QwenDriverError {
            // Unknown driver failures remain observable to the caller.
        }
    }

    @Test func qwenAdapterMapsGenerationAdmissionFailureToRequestError() async throws {
        let driver = P19QwenScriptedDriver(plan: .contextOverflow)
        let session = ServerQwenModelSession(driver: driver)
        let request = Self.validatedQwenRequest(
            prompt: .chat(
                messages: [ModelChatMessage(role: .user, content: "too long")],
                tools: [], thinking: .automatic))

        do {
            _ = try await session.generate(request) { _ in }
            Issue.record("context overflow unexpectedly completed")
        } catch let error as ServerRequestError {
            #expect(error.envelope.error.code == "invalid_request")
            #expect(error.envelope.error.message.contains("context overflow"))
        }
    }

    @Test func qwenValidationPreservesHistoryToolsReasoningImageOrderAndLease() throws {
        let source = #"""
        {"model":"qwen-test","thinking":"on","messages":[{"role":"system","content":"policy"},{"role":"assistant","content":"working","reasoning_content":"hidden","tool_calls":[{"id":"call_000000000000000000000001","type":"function","function":{"name":"lookup","arguments":"{\"z\":1,\"a\":\"kept\"}"}}]},{"role":"tool","tool_call_id":"call_000000000000000000000001","content":"result"},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,YWJj"}},{"type":"text","text":"middle"},{"type":"image_url","image_url":{"url":"data:image/png;base64,ZGVm"}},{"type":"text","text":"question"}]}],"tools":[{"type":"function","function":{"name":"lookup","description":"ordered","parameters":{"type":"object","properties":{"z":{"type":"integer"},"a":{"type":"string"}},"required":["z","a"]}}}],"stream":false,"max_completion_tokens":32}
        """#
        var parsed: (ValidatedChatRequest, ParsedChatRequestBody)? =
            try Self.parseQwen(source)
        var validated = parsed!.0
        let imageURLs = Array(validated.qwenImagesByID.values)
        let leaseDirectory = parsed!.1.lease?.directoryURL
        parsed = nil
        guard case .chat(let messages, let tools, let thinking) = validated.qwenPrompt else {
            Issue.record("Qwen validation did not produce a chat prompt")
            return
        }
        #expect(thinking == .enabled)
        #expect(messages.count == 4)
        #expect(messages[1].reasoningContent == "hidden")
        #expect(messages[1].toolCalls.count == 1)
        #expect(messages[1].toolCalls[0].arguments == .object([
            .init("z", .integer(1)), .init("a", .string("kept"))
        ]))
        let userContent = try #require(messages[3].content)
        #expect(userContent == .parts([
            .image(.init(id: "server-image-1")),
            .text("middle"),
            .image(.init(id: "server-image-2")),
            .text("question")
        ]))
        #expect(tools.count == 1)
        #expect(tools[0].function.parameters == .object([
            .init("type", .string("object")),
            .init("properties", .object([
                .init("z", .object([.init("type", .string("integer"))])),
                .init("a", .object([.init("type", .string("string"))]))
            ])),
            .init("required", .array([.string("z"), .string("a")]))
        ]))

        let first = try #require(validated.qwenImagesByID["server-image-1"])
        let second = try #require(validated.qwenImagesByID["server-image-2"])
        #expect(try Data(contentsOf: first) == Data("abc".utf8))
        #expect(try Data(contentsOf: second) == Data("def".utf8))
        #expect(imageURLs.count == 2)
        #expect(leaseDirectory != nil)
        let retainedFirst = first
        let retainedSecond = second
        validated = ValidatedChatRequest(
            messages: [], tools: [], stream: false, includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 1),
            maximumCompletionTokens: 1)
        #expect(!FileManager.default.fileExists(atPath: retainedFirst.path))
        #expect(!FileManager.default.fileExists(atPath: retainedSecond.path))
        if let leaseDirectory {
            #expect(!FileManager.default.fileExists(atPath: leaseDirectory.path))
        }
    }

    @Test func qwenValidationRejectsVideoAndAudioBeforeGeneration() throws {
        for (type, expectedKind) in [
            ("video_url", "video"), ("input_video", "video"),
            ("audio_url", "audio"), ("input_audio", "audio")
        ] {
            let source = #"{"model":"qwen-test","messages":[{"role":"user","content":[{"type":"\#(type)","\#(type)":{"url":"data:application/octet-stream;base64,YQ=="}}]}]}"#
            let parser = StreamingChatRequestBody()
            var buffer = ByteBuffer(bytes: Data(source.utf8))
            try parser.feed(&buffer)
            let parsed = try parser.finish()
            let request = try JSONDecoder().decode(
                OpenAIChatRequest.self, from: parsed.json)
            do {
                _ = try OpenAIRequestValidator.validate(
                    request, modelID: "qwen-test",
                    preStagedImages: parsed.stagedImages,
                    attachmentLease: parsed.lease,
                    family: .qwen,
                    sanitizedJSON: parsed.json)
                Issue.record("unsupported \(type) input was accepted")
            } catch let error as ServerRequestError {
                #expect(error.envelope.error.code == "unsupported_content")
                #expect(error.envelope.error.message.contains(expectedKind))
            }
        }
    }

    @Test func gemmaThinkingAutoRemainsAcceptedByLegacyValidator() throws {
        let source = #"{"model":"gemma-test","thinking":"auto","messages":[{"role":"user","content":"hello"}]}"#
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: Data(source.utf8))
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "gemma-test")
        #expect(validated.messages.count == 1)
        #expect(validated.qwenPrompt == nil)
    }

    private static func validatedQwenRequest(
        prompt: ModelFamilyGenerationPrompt,
        images: [String: URL] = [:]
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [], tools: [], stream: false, includeUsage: true,
            generationConfig: GenerationConfig(
                maxNewTokens: 32, temperature: 0.2, topK: 32,
                topP: 0.9, repetitionPenalty: 1.05, seed: 7),
            maximumCompletionTokens: 32,
            qwenPrompt: prompt, qwenImagesByID: images)
    }

    private static func lookupTool() -> ModelChatToolDefinition {
        ModelChatToolDefinition(function: .init(
            name: "lookup", description: "ordered",
            parameters: .object([
                .init("type", .string("object")),
                .init("properties", .object([
                    .init("z", .object([.init("type", .string("integer"))])),
                    .init("a", .object([.init("type", .string("string"))]))
                ]))
            ])))
    }

    private static func parseQwen(
        _ source: String
    ) throws -> (ValidatedChatRequest, ParsedChatRequestBody) {
        let parser = StreamingChatRequestBody()
        var buffer = ByteBuffer(bytes: Data(source.utf8))
        try parser.feed(&buffer)
        let parsed = try parser.finish()
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: parsed.json)
        let validated = try OpenAIRequestValidator.validate(
            request, modelID: "qwen-test",
            preStagedImages: parsed.stagedImages,
            attachmentLease: parsed.lease,
            family: .qwen,
            sanitizedJSON: parsed.json)
        return (validated, parsed)
    }

    private static func makeFixtureFile(
        named name: String, bytes: [UInt8]
    ) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbo-fieldfare-p19-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent(name)
        try Data(bytes).write(to: file, options: .atomic)
        return file
    }
}
