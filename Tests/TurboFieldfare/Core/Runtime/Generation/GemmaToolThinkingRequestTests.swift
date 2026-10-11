import Foundation
import Testing
@testable import TurboFieldfare

@Suite
struct GemmaToolThinkingRequestTests {
    @Test
    func emptyToolAwareCompletionIsRetryableOnlyWhenNoCallWasProduced() {
        for reason in [StopReason.endOfTurn, .maxTokens, .eos, .stopString] {
            #expect(GemmaMultimodalConversation.isRetryableEmptyToolCompletion(
                text: "", hasToolCalls: false, reason: reason))
        }
        #expect(!GemmaMultimodalConversation.isRetryableEmptyToolCompletion(
            text: "", hasToolCalls: false, reason: .cancelled))
        #expect(!GemmaMultimodalConversation.isRetryableEmptyToolCompletion(
            text: "answer", hasToolCalls: false, reason: .endOfTurn))
        #expect(!GemmaMultimodalConversation.isRetryableEmptyToolCompletion(
            text: "", hasToolCalls: true, reason: .toolCalls))
    }

    @Test
    func bundledFixtureLoadsLocallyAndMissingFilesFailBeforeTokenizerLoad() async throws {
        let folder = try Self.fixtureFolder()
        let tokenizer = try await Self.loadLocalTokenizer(from: folder)
        #expect(tokenizer.vocabSize > 0)

        let incomplete = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: incomplete, withIntermediateDirectories: true)
        defer {
            do {
                try FileManager.default.removeItem(at: incomplete)
            } catch {
                Issue.record("Could not remove the temporary incomplete fixture: \(error)")
            }
        }
        try Data("{}".utf8).write(to: incomplete.appendingPathComponent("tokenizer.json"))
        do {
            _ = try await Self.loadLocalTokenizer(from: incomplete)
            Issue.record("An incomplete local fixture must fail before tokenizer loading")
        } catch FixtureError.missingFile(let name) {
            #expect(name == "chat_template.jinja")
        }
    }

    @Test
    func sendToolResultsConsumesAssessedBridgeOnlyWhenThinkingModeMatches() async throws {
        let tokenizer = try await Self.fixtureTokenizer()
        for requestMode in [true, false] {
            let setup = Self.resultConversation(tokenizer: tokenizer, output: .answer)
            await setup.conversation.seedAssessedResultBridgeForTesting(
                result: setup.result, calls: [setup.call],
                tokens: [Self.assessedSentinel], thinkingEnabled: true)
            let completion = try await setup.conversation.sendToolResults(
                [setup.result], config: Self.config,
                toolThinkingEnabled: requestMode)
            #expect(completion.turn.text == "done")
            #expect(setup.recorder.prompts.count == 1)
            let prompt = setup.recorder.prompts[0]
            #expect(prompt.contains(Self.assessedSentinel) == requestMode)
            let state = await setup.conversation.templateStateForTesting()
            #expect(!state.hasAssessedResultBridge)
        }
    }

    @Test
    func retryableEmptyResultRestoresAssessedBridgeForNextPhysicalRequest() async throws {
        let tokenizer = try await Self.fixtureTokenizer()
        let setup = Self.resultConversation(tokenizer: tokenizer, output: .emptyThenAnswer)
        await setup.conversation.seedAssessedResultBridgeForTesting(
            result: setup.result, calls: [setup.call],
            tokens: [Self.assessedSentinel], thinkingEnabled: true)
        let first = try await setup.conversation.sendToolResults(
            [setup.result], config: Self.config, toolThinkingEnabled: true)
        #expect(first.turn.text.isEmpty)
        let restoredState = await setup.conversation.templateStateForTesting()
        #expect(restoredState.hasAssessedResultBridge)
        let second = try await setup.conversation.sendToolResults(
            [setup.result], config: Self.config, toolThinkingEnabled: true)
        #expect(second.turn.text == "done")
        #expect(setup.recorder.prompts.count == 2)
        #expect(setup.recorder.prompts.allSatisfy { $0.contains(Self.assessedSentinel) })
        let consumedState = await setup.conversation.templateStateForTesting()
        #expect(!consumedState.hasAssessedResultBridge)
    }

    @Test
    func sendToolResultsRejectsWrongResultIdentityBeforeConsumingAssessedBridge() async throws {
        let tokenizer = try await Self.fixtureTokenizer()
        let setup = Self.resultConversation(tokenizer: tokenizer, output: .answer)
        await setup.conversation.seedAssessedResultBridgeForTesting(
            result: setup.result, calls: [setup.call],
            tokens: [Self.assessedSentinel], thinkingEnabled: true)
        let wrong = ConversationToolResult(
            callID: "wrong-call", name: setup.call.name, content: "facts")
        do {
            _ = try await setup.conversation.sendToolResults(
                [wrong], config: Self.config, toolThinkingEnabled: true)
            Issue.record("A result for a different call must be rejected")
        } catch MultimodalConversationError.invalidToolContinuation {
            #expect(setup.recorder.prompts.isEmpty)
            let state = await setup.conversation.templateStateForTesting()
            #expect(state.hasAssessedResultBridge)
        }
    }

    @Test
    func resumeCheckpointConsumesOpeningOnlyWhenThinkingModeMatches() async throws {
        let tokenizer = try await Self.fixtureTokenizer()
        for requestMode in [true, false] {
            let output: OutputPlan = requestMode ? .answer : .emptyThenAnswer
            let setup = Self.checkpointConversation(tokenizer: tokenizer, output: output)
            let id = UUID()
            await setup.conversation.seedCheckpointOpeningForTesting(
                id: id, messages: setup.messages, tools: [setup.tool],
                tokens: [Self.assessedSentinel], thinkingEnabled: true)
            let first = try await setup.conversation.resumeCheckpoint(
                id: id, config: Self.config, toolThinkingEnabled: requestMode,
                prefillConfig: .defaultChunked, checkCancellation: {}, shouldStop: nil,
                captureToolFailureEvidence: false, maximumConsecutiveInvisibleTokens: nil,
                captureThoughtPreview: false, onProgress: nil, onStructuredProgress: nil)
            let completion: StructuredConversationTurnResult
            if requestMode {
                completion = first
            } else {
                #expect(first.turn.text.isEmpty)
                let pendingState = await setup.conversation.templateStateForTesting()
                #expect(pendingState.checkpointID == id)
                #expect(pendingState.checkpointThinkingEnabled == false)
                completion = try await setup.conversation.resumeCheckpoint(
                    id: id, config: Self.config, toolThinkingEnabled: false,
                    prefillConfig: .defaultChunked, checkCancellation: {}, shouldStop: nil,
                    captureToolFailureEvidence: false, maximumConsecutiveInvisibleTokens: nil,
                    captureThoughtPreview: false, onProgress: nil, onStructuredProgress: nil)
            }
            #expect(completion.turn.text == "done")
            let prompts = setup.recorder.prompts
            #expect(prompts.count == (requestMode ? 1 : 2))
            #expect(prompts.allSatisfy {
                $0.contains(Self.assessedSentinel) == requestMode
            })
            let state = await setup.conversation.templateStateForTesting()
            #expect(state.checkpointID == nil)
            #expect(state.checkpointThinkingEnabled == nil)
        }
    }

    @Test
    func resumeCheckpointRejectsWrongIdentityBeforeCompletion() async throws {
        let tokenizer = try await Self.fixtureTokenizer()
        let setup = Self.checkpointConversation(tokenizer: tokenizer)
        let id = UUID()
        await setup.conversation.seedCheckpointOpeningForTesting(
            id: id, messages: setup.messages, tools: [setup.tool],
            tokens: [Self.assessedSentinel], thinkingEnabled: true)
        do {
            _ = try await setup.conversation.resumeCheckpoint(
                id: UUID(), config: Self.config, toolThinkingEnabled: true,
                prefillConfig: .defaultChunked, checkCancellation: {}, shouldStop: nil,
                captureToolFailureEvidence: false, maximumConsecutiveInvisibleTokens: nil,
                captureThoughtPreview: false, onProgress: nil, onStructuredProgress: nil)
            Issue.record("A different checkpoint ID must be rejected")
        } catch MultimodalConversationError.invalidToolContinuation {
            #expect(setup.recorder.prompts.isEmpty)
            let state = await setup.conversation.templateStateForTesting()
            #expect(state.checkpointID == id)
        }
    }

    @Test
    func requestTemplateKeepsTheFixtureSpecificOpeningAndContinuationBehavior() async throws {
        let loaded = try await Self.fixtureTokenizer()
        let user = GFTokenizer.Message(role: .user, content: "Inspect the current screen.")
        let call = GFTokenizer.HistoricalToolCall(
            id: "fixture-call", name: "inspect", arguments: .object([:]))
        let assistant = GFTokenizer.Message(role: .assistant, content: nil, toolCalls: [call])
        let toolResult = GFTokenizer.Message(role: .tool, content: "Current screen facts.",
            toolCallID: call.id, name: call.name)
        let tool = Self.tool
        let on = GemmaToolRequestTemplate(loadedTokenizer: loaded, thinkingEnabled: true)
        let off = GemmaToolRequestTemplate(loadedTokenizer: loaded, thinkingEnabled: false)
        #expect(on.tokenizer.enableToolThinking)
        #expect(!off.tokenizer.enableToolThinking)
        #expect(on.canReuse(thinkingEnabled: true))
        #expect(!on.canReuse(thinkingEnabled: false))
        #expect(off.canReuse(thinkingEnabled: false))
        #expect(!off.canReuse(thinkingEnabled: true))
        let opening = try on.encodeChat(messages: [user], tools: [tool])
        #expect(!opening.isEmpty)
        let continuation = try on.encodeResultContinuation(
            cachedMessages: [user], assistant: assistant,
            incomingMessages: [user, assistant, toolResult], tools: [tool])
        #expect(!continuation.isEmpty)
    }

    private enum FixtureError: Error {
        case missingFile(String)
    }

    private enum OutputPlan {
        case answer
        case emptyThenAnswer
    }

    private struct ResultSetup {
        let conversation: GemmaMultimodalConversation
        let recorder: CompletionRecorder
        let call: GFTokenizer.HistoricalToolCall
        let result: ConversationToolResult
    }

    private struct CheckpointSetup {
        let conversation: GemmaMultimodalConversation
        let recorder: CompletionRecorder
        let messages: [GFTokenizer.Message]
        let tool: GFTokenizer.FunctionDefinition
    }

    private static let assessedSentinel: Int32 = 90_001
    private static let config = GenerationConfig(maxNewTokens: 32, temperature: 0)
    private static let tool = GFTokenizer.FunctionDefinition(
        name: "inspect", description: "Inspect current state.",
        parameters: .object(["type": .string("object")]))

    private static func fixtureFolder() throws -> URL {
        guard let resourceURL = Bundle.module.resourceURL else {
            throw FixtureError.missingFile("resource root")
        }
        return resourceURL.appendingPathComponent("lesson-tokenizer", isDirectory: true)
    }

    private static func loadLocalTokenizer(from folder: URL) async throws -> GFTokenizer {
        let required = ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]
        for name in required {
            let file = folder.appendingPathComponent(name)
            guard FileManager.default.isReadableFile(atPath: file.path) else {
                throw FixtureError.missingFile(name)
            }
        }
        return try await GFTokenizer.load(from: folder)
    }

    private static func fixtureTokenizer() async throws -> GFTokenizer {
        try await loadLocalTokenizer(from: fixtureFolder())
    }

    private static func resultConversation(
        tokenizer: GFTokenizer, output: OutputPlan
    ) -> ResultSetup {
        let call = GFTokenizer.HistoricalToolCall(
            id: "pending-call", name: tool.name, arguments: .object([:]))
        let user = GFTokenizer.Message(role: .user, content: "Inspect current state.")
        let assistant = GFTokenizer.Message(role: .assistant, content: nil, toolCalls: [call])
        let baseline: [Int32] = [11, 13]
        let recorder = CompletionRecorder(
            tokenizer: tokenizer, output: output, initialPosition: baseline.count)
        let conversation = GemmaMultimodalConversation(
            tokenizer: tokenizer, testCompletion: recorder.testCompletion,
            messages: [user, assistant], tools: [tool], tokenIDs: baseline, maxContext: 32_768)
        let result = ConversationToolResult(
            callID: call.id, name: call.name, content: "Current state is ready.")
        return ResultSetup(conversation: conversation, recorder: recorder, call: call, result: result)
    }

    private static func checkpointConversation(
        tokenizer: GFTokenizer, output: OutputPlan = .answer
    ) -> CheckpointSetup {
        let messages = [GFTokenizer.Message(role: .user, content: "Continue the task.")]
        let recorder = CompletionRecorder(
            tokenizer: tokenizer, output: output, initialPosition: 0)
        let conversation = GemmaMultimodalConversation(
            tokenizer: tokenizer, testCompletion: recorder.testCompletion,
            messages: [], tools: [tool], tokenIDs: [], maxContext: 32_768)
        return CheckpointSetup(
            conversation: conversation, recorder: recorder, messages: messages, tool: tool)
    }

    private final class CompletionRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private let tokenizer: GFTokenizer
        private let output: OutputPlan
        private var positionValue: Int
        private var invocation = 0
        private var recordedPrompts: [[Int32]] = []

        init(tokenizer: GFTokenizer, output: OutputPlan, initialPosition: Int) {
            self.tokenizer = tokenizer
            self.output = output
            positionValue = initialPosition
        }

        var prompts: [[Int32]] { lock.withLock { recordedPrompts } }

        var testCompletion: GemmaMultimodalConversation.TestCompletion {
            GemmaMultimodalConversation.TestCompletion(
                position: { self.lock.withLock { self.positionValue } },
                rewind: { target in self.lock.withLock { self.positionValue = target } },
                reset: { self.lock.withLock { self.positionValue = 0 } },
                run: { prompt, _, cached, progress in
                    let empty: Bool = self.lock.withLock {
                        self.invocation += 1
                        self.recordedPrompts.append(prompt)
                        if case .emptyThenAnswer = self.output { return self.invocation == 1 }
                        return false
                    }
                    let emitted: [Int32] = empty
                        ? [] : [self.tokenizer.channelStartID, self.tokenizer.eosID]
                    if !empty {
                        progress(.token(index: 0, id: self.tokenizer.channelStartID, delta: ""))
                        progress(.token(index: 1, id: self.tokenizer.eosID, delta: "final\n"))
                        progress(.tail("done"))
                    }
                    let finalPosition = prompt.count + emitted.count
                    self.lock.withLock { self.positionValue = finalPosition }
                    return RawDecodeResult(
                        prefillTokens: prompt.count, cachedPromptTokens: cached,
                        computedPrefillTokens: prompt.count - cached,
                        prefillSeconds: 0, newTokens: emitted.count, decodeSeconds: 0,
                        reason: .endOfTurn, kvPosition: finalPosition,
                        kvBackedTokenIDs: prompt + emitted, uncommittedBoundaryTokenIDs: [])
                })
        }
    }
}
