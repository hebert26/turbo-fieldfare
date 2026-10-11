import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct AppModelFormatRecoveryTests {
    @MainActor
    @Test func agentRunWithoutRecoveryTriggerSendsThinkingOff() async throws {
        let client = FormatClient(failures: 0)
        let model = readyModel(client)
        model.run()
        try await finish(model)
        #expect(client.requests.count == 1)
        #expect(client.requests[0].runtimeOptions.toolThinkingEnabled == false)
        #expect(model.error == nil)
        #expect(model.outputText == "Completed")
    }

    @MainActor
    @Test func rejectedProposalsReceiveCorrectionsAndCanFinish() async throws {
        let client = FormatClient(failures: 0, invalidProposals: 3)
        let model = readyModel(client)
        model.run()
        try await finish(model)
        #expect(client.requests.count == 4)
        #expect(model.error == nil)
        #expect(model.outputText == "Completed")
        #expect(model.conversation.canSend)
        for request in client.requests.dropFirst() {
            guard case .results(let results) = request.toolTurn else {
                Issue.record("Missing correction result")
                continue
            }
            #expect(results.first?.content.contains("Return a corrected response") == true)
            #expect(results.first?.content.contains("Allowed actions now:") == true)
        }
        #expect(!model.outputAgentActivities.contains { $0.kind == .mcpRequest })
    }

    @MainActor
    @Test func rejectedProposalLimitKeepsChatUsableAndResetsModelSession() async throws {
        let client = FormatClient(failures: 0, invalidProposals: 6)
        let model = readyModel(client)
        let originalEpoch = model.conversation.epoch
        model.run()
        try await finish(model)
        #expect(client.requests.count == 6)
        #expect(model.error?.userMessage.contains("Agent paused") == true)
        #expect(model.conversation.canSend)
        #expect(!model.conversation.isLineageLost)
        #expect(model.conversation.epoch != originalEpoch)
        model.promptText = "Continue the remaining work"
        model.run()
        try await finish(model)
        #expect(client.requests.count == 7)
        #expect(model.error == nil)
        #expect(model.outputText == "Completed")
        #expect(client.requests.last?.conversationEpoch != originalEpoch)
        #expect(!model.outputAgentActivities.contains { $0.kind == .mcpRequest })
    }

    @MainActor
    @Test func malformedFirstResponseIsCorrectedInTheSameTurn() async throws {
        let client = FormatClient(failures: 1)
        let model = readyModel(client)
        model.run()
        try await finish(model)
        let requests = client.requests
        #expect(requests.count == 2)
        #expect(model.error == nil)
        #expect(model.outputText == "Completed")
        #expect(model.conversation.committedTurns == 1)
        #expect(requests[0].conversationEpoch == requests[1].conversationEpoch)
        #expect(requests[0].turnIndex == requests[1].turnIndex)
        #expect(requests[0].runtimeOptions.toolThinkingEnabled == false)
        #expect(requests[1].runtimeOptions.toolThinkingEnabled == true)
        #expect(requests[1].prompt.hasPrefix(requests[0].prompt))
        #expect(requests[1].prompt.contains("Host format correction"))
        #expect(requests[0].toolTurn == requests[1].toolTurn)
        #expect(client.finishedFailedStreamBeforeRetry)
        #expect(model.outputAgentActivities.contains {
            $0.kind == .generationRecovery && $0.status == .succeeded
        })
        #expect(!model.outputAgentActivities.contains { $0.kind == .mcpRequest })
    }

    @MainActor
    @Test func repeatedMalformedOutputStopsAfterTwoCorrections() async throws {
        let client = FormatClient(failures: 3)
        let model = readyModel(client)
        model.run()
        try await finish(model)
        #expect(client.requests.count == 3)
        #expect(client.requests.map { $0.runtimeOptions.toolThinkingEnabled } == [false, true, false])
        #expect(model.error?.userMessage.contains("after two format corrections") == true)
        #expect(model.conversation.committedTurns == 0)
        #expect(!model.outputAgentActivities.contains { $0.kind == .mcpRequest })
    }

    @MainActor
    @Test func missingRecoveryReceiptNeverRetries() async throws {
        let client = FormatClient(failures: 1, canRegenerate: false)
        let model = readyModel(client)
        model.run()
        try await finish(model)
        #expect(client.requests.count == 1)
        #expect(model.error?.userMessage.contains("invalid tool request") == true)
        #expect(!model.outputAgentActivities.contains { $0.kind == .mcpRequest })
    }

    @MainActor
    @Test func aPublishedCallPreventsFormatRetry() async throws {
        let client = FormatClient(failures: 1, publishesCall: true)
        let model = readyModel(client)
        model.run()
        try await finish(model)
        #expect(client.requests.count == 1)
        #expect(model.error != nil)
        #expect(!model.outputAgentActivities.contains { $0.kind == .mcpRequest })
    }

    @MainActor
    @Test func thoughtRecoveryContinuesThroughAUsableLocalHistoryCall() async throws {
        let client = try RecoveryScript(specialOffIsMalformed: false)
        let model = readyAgentModel(client)
        model.run()
        try await finish(model)
        let requests = client.requests
        #expect(requests.count == 4)
        #expect(requests.map { $0.runtimeOptions.toolThinkingEnabled } == [
            false, true, false, false,
        ])
        #expect(requests[0].toolTurn == requests[1].toolTurn)
        #expect(requests[1].toolTurn == requests[2].toolTurn)
        #expect(requests[0].conversationEpoch == requests[1].conversationEpoch)
        #expect(requests[1].conversationEpoch == requests[2].conversationEpoch)
        #expect(requests[0].turnIndex == requests[1].turnIndex)
        #expect(requests[1].turnIndex == requests[2].turnIndex)
        guard case .user? = requests[0].toolTurn,
              case .user? = requests[1].toolTurn,
              case .user? = requests[2].toolTurn,
              case .results(let historyResults)? = requests[3].toolTurn else {
            Issue.record("Recovery must preserve the user turn and consume the history result")
            return
        }
        #expect(historyResults.count == 1)
        #expect(historyResults[0].name == "task_history_read")
        #expect(historyResults[0].callID == "recovery-history")
        #expect(model.error == nil)
        #expect(model.outputText == "Completed after history")
        #expect(client.historyCalls == 1)
    }

    @MainActor
    @Test func malformedSpecialOffRecoveryStopsAfterOnePhysicalRetry() async throws {
        let client = try RecoveryScript(specialOffIsMalformed: true)
        let model = readyAgentModel(client)
        model.run()
        try await finish(model)
        let requests = client.requests
        #expect(requests.count == 3)
        #expect(requests.map { $0.runtimeOptions.toolThinkingEnabled } == [false, true, false])
        #expect(requests[0].toolTurn == requests[1].toolTurn)
        #expect(requests[1].toolTurn == requests[2].toolTurn)
        #expect(requests[0].conversationEpoch == requests[1].conversationEpoch)
        #expect(requests[1].conversationEpoch == requests[2].conversationEpoch)
        #expect(requests[0].turnIndex == requests[1].turnIndex)
        #expect(requests[1].turnIndex == requests[2].turnIndex)
        #expect(client.historyCalls == 0)
        #expect(model.error?.userMessage.contains("original malformed special-OFF") == true)
        #expect(model.error?.userMessage.contains("after two format corrections") == false)
    }

    @Test func correctionKeepsCheckpointIdentityAndToolResultImages() {
        let checkpoint = AppToolTurn.checkpoint(UUID())
        #expect(checkpoint.correctingMalformedResponse() == checkpoint)
        let image = AppImageAttachment(fileURL: URL(fileURLWithPath: "/tmp/fixture.png"),
            displayName: "fixture.png", encodedBytes: 1, sha256: String(repeating: "a", count: 64))
        let result = AppToolResult(callID: "completed-action", name: "visioncapture_navigate",
            content: "verified action; current choices", imageAttachments: [image])
        guard case .results(let corrected) = AppToolTurn.results([result]).correctingMalformedResponse() else {
            Issue.record("Missing corrected tool result")
            return
        }
        #expect(corrected.count == 1)
        #expect(corrected[0].callID == result.callID)
        #expect(corrected[0].name == result.name)
        #expect(corrected[0].imageAttachments == result.imageAttachments)
        #expect(corrected[0].content.hasPrefix(result.content))
    }

    @MainActor
    private func readyModel(_ client: FormatClient) -> AppModel {
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(modelDirectory: directory, client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.setAgentModeEnabled(true)
        model.promptText = "Check the current screen"
        return model
    }

    @MainActor
    private func readyAgentModel(_ client: RecoveryScript) -> AppModel {
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(modelDirectory: directory, client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.setAgentModeEnabled(true)
        model.setAgentBundleIdentifier("com.example.fixture")
        model.setAgentSimulatorUDID("00000000-0000-0000-0000-000000000001")
        model.promptText = "Check the current screen"
        return model
    }

    @MainActor
    private func finish(_ model: AppModel) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while model.isRunning && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(!model.isRunning)
    }

    private final class FormatClient: AppModelLifecycleClient, @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [AppGenerationRequest] = []
        private var failedStreamFinished = false
        private var drainWasRespected = false
        let failures: Int
        let canRegenerate: Bool
        let publishesCall: Bool
        let invalidProposals: Int

        init(failures: Int, canRegenerate: Bool = true, publishesCall: Bool = false,
             invalidProposals: Int = 0) {
            self.failures = failures
            self.canRegenerate = canRegenerate
            self.publishesCall = publishesCall
            self.invalidProposals = invalidProposals
        }

        var requests: [AppGenerationRequest] { lock.withLock { recorded } }
        var finishedFailedStreamBeforeRetry: Bool { lock.withLock { drainWasRespected } }
        func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                          options: AppRuntimeOptions, forceLogitsHead: Bool,
                          onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {}
        func unload() async {}
        func resetConversation(epoch: UUID) async throws {}
        func shutdownForTermination() {}
        func cancel() {}

        func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
            let attempt = lock.withLock {
                if !recorded.isEmpty { drainWasRespected = failedStreamFinished }
                recorded.append(request)
                return recorded.count
            }
            return AsyncThrowingStream { continuation in
                Task {
                    if attempt <= invalidProposals {
                        continuation.yield(.toolCall(AppToolCall(id: "rejected-\(attempt)",
                            name: "visioncapture_navigate", arguments: .object([
                                "action": .string("tap"), "target": .string("old-target")
                            ]))))
                        continuation.yield(.finished(AppDiagnostics(generatedTokens: 1,
                            stopReason: .toolCalls, promptTokenCount: 1,
                            timeToFirstTokenSeconds: nil, decodeSeconds: 0.1,
                            tokensPerSecond: 10, peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions())))
                        continuation.finish()
                    } else if attempt <= failures {
                        if publishesCall {
                            continuation.yield(.toolCall(AppToolCall(id: "must-not-send",
                                name: "visioncapture_navigate", arguments: .object(["action": .string("launch")]))))
                        }
                        let error = AppInferenceError.structuredToolFailure(
                            message: "malformed", canRegenerateToolResult: canRegenerate, evidence: nil)
                        continuation.yield(.failed(error, partial: nil))
                        try? await Task.sleep(for: .milliseconds(10))
                        lock.withLock { failedStreamFinished = true }
                        continuation.finish(throwing: error)
                    } else {
                        continuation.yield(.token(AppTokenEvent(index: 0,
                            textDelta: "Completed", elapsedDecodeSeconds: 0.1)))
                        continuation.yield(.finished(AppDiagnostics(generatedTokens: 1,
                            stopReason: .endOfTurn, promptTokenCount: 1,
                            timeToFirstTokenSeconds: nil, decodeSeconds: 0.1,
                            tokensPerSecond: 10, peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions())))
                        continuation.finish()
                    }
                }
            }
        }
    }

    private final class RecoveryScript: AppModelLifecycleClient, @unchecked Sendable {
        private let lock = NSLock()
        private var recorded: [AppGenerationRequest] = []
        private var historyCount = 0
        private let specialOffIsMalformed: Bool
        private let thoughtError: AppInferenceError

        init(specialOffIsMalformed: Bool) throws {
            self.specialOffIsMalformed = specialOffIsMalformed
            thoughtError = try Self.repeatedThoughtError()
        }

        var requests: [AppGenerationRequest] { lock.withLock { recorded } }
        var historyCalls: Int { lock.withLock { historyCount } }

        func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                          options: AppRuntimeOptions, forceLogitsHead: Bool,
                          onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {}

        func unload() async {}
        func resetConversation(epoch: UUID) async throws {}
        func shutdownForTermination() {}
        func cancel() {}

        func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
            let attempt = lock.withLock {
                recorded.append(request)
                return recorded.count
            }
            return AsyncThrowingStream { continuation in
                Task {
                    switch attempt {
                    case 1:
                        continuation.yield(.finished(Self.diagnostics(request, reason: .endOfTurn)))
                        continuation.finish()
                    case 2:
                        continuation.yield(.failed(self.thoughtError, partial: nil))
                        continuation.finish(throwing: self.thoughtError)
                    case 3 where self.specialOffIsMalformed:
                        let error = AppInferenceError.structuredToolFailure(
                            message: "original malformed special-OFF response",
                            canRegenerateToolResult: true, evidence: nil)
                        continuation.yield(.failed(error, partial: nil))
                        continuation.finish(throwing: error)
                    case 3:
                        self.lock.withLock { self.historyCount += 1 }
                        continuation.yield(.toolCall(AppToolCall(
                            id: "recovery-history", name: "task_history_read",
                            arguments: .object([
                                "observation_id": .string("missing-observation"),
                            ]))))
                        continuation.yield(.finished(Self.diagnostics(request, reason: .toolCalls)))
                        continuation.finish()
                    default:
                        continuation.yield(.token(AppTokenEvent(
                            index: 0, textDelta: "Completed after history",
                            elapsedDecodeSeconds: 0.1)))
                        continuation.yield(.finished(Self.diagnostics(request, reason: .endOfTurn)))
                        continuation.finish()
                    }
                }
            }
        }

        private static func diagnostics(
            _ request: AppGenerationRequest, reason: AppStopReason
        ) -> AppDiagnostics {
            AppDiagnostics(generatedTokens: 1, stopReason: reason, promptTokenCount: 1,
                timeToFirstTokenSeconds: nil, decodeSeconds: 0.1, tokensPerSecond: 10,
                peakMemoryBytes: nil,
                runtimeOptions: AppRuntimeOptions(
                    toolThinkingEnabled: request.runtimeOptions.toolThinkingEnabled))
        }

        private static func repeatedThoughtError() throws -> AppInferenceError {
            let json = """
            {
              "canRetryToolResult": false,
              "pendingCallID": null,
              "pendingToolName": null,
              "restoredTokenCount": 9000,
              "requiresRebuild": false,
              "generatedTokens": 200,
              "thinkingTokens": 160,
              "blockTokens": 20,
              "repetitions": 8
            }
            """
            let receipt = try JSONDecoder().decode(
                ThoughtRepetitionRecovery.self, from: Data(json.utf8))
            return .repeatedThought(receipt)
        }
    }
}
