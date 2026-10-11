import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

/// The chat_ended trace record: one reason for every way an agent run ends.
@Suite struct AgentRunEndTests {
    private func result(_ stopReason: AppStopReason, followUp: String? = nil) -> VisionCaptureAgentRunResult {
        VisionCaptureAgentRunResult(
            answer: "done",
            diagnostics: AppDiagnostics(
                generatedTokens: 1, stopReason: stopReason, promptTokenCount: 1, timeToFirstTokenSeconds: nil,
                decodeSeconds: 0.1, tokensPerSecond: 10, peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions()),
            followUpPrompt: followUp)
    }

    @Test
    func everyEndPathHasItsReason() {
        // Runs that return.
        #expect(AppModel.agentRunEndReason(result(.endOfTurn)) == "final_answer")
        #expect(AppModel.agentRunEndReason(result(.endOfTurn, followUp: "Continue.")) == "final_answer_with_follow_up")
        #expect(AppModel.agentRunEndReason(result(.cancelled)) == "user_cancel")
        // Runs that throw.
        #expect(AppModel.agentRunEndReason(CancellationError()) == "user_cancel")
        #expect(AppModel.agentRunEndReason(
            VisionCaptureAgentError.proposalCorrectionExhausted("paused")) == "proposal_pause")
        let refused = VisionCaptureServerOutcome(verdict: "failed", reason: "No visible Simulator window was found.",
                                                 reasonCode: "SIMULATOR_WINDOW_NOT_FOUND", dispatchAttempted: false)
        let hostRefusals: [VisionCaptureAgentError] = [
            .mcpOutcome(refused), .mcpRefused("refused"), .unsupportedSystemInteraction("SYSTEM_ALERT_PRESENT"),
            .returnedIdentityMismatch(fieldPath: "udid", refusalCode: nil), .sessionIdentityMismatch,
            .launchOutcomeUnproven("unproven"),
        ]
        for error in hostRefusals { #expect(AppModel.agentRunEndReason(error) == "host_refusal", "\(error)") }
        for error: VisionCaptureAgentError in [.mcpUnavailable("socket closed"), .skillReadFailed("missing")] {
            #expect(AppModel.agentRunEndReason(error) == "host_unavailable", "\(error)")
        }
        let loopErrors: [VisionCaptureAgentError] = [
            .invalidConfiguration("bad"), .malformedCall("bad"), .navigationUnavailable("no"), .identityMismatch,
            .unsupportedVisualRequest, .noProgress("stuck"), .incompleteAnswer,
        ]
        for error in loopErrors { #expect(AppModel.agentRunEndReason(error) == "loop_error", "\(error)") }
        #expect(AppModel.agentRunEndReason(AppInferenceError.invalidRequest("bad")) == "model_error")
        #expect(AppModel.agentRunEndReason(AppInferenceError.cancelled) == "user_cancel")
        #expect(AppModel.agentRunEndReason(AppInferenceError.structuredToolFailure(
            message: "malformed", canRegenerateToolResult: true, evidence: nil)) == "malformed_request_pause")
        #expect(AppModel.agentRunEndReason(AppInferenceError.structuredToolFailure(
            message: "malformed", canRegenerateToolResult: false, evidence: nil)) == "model_error")
        struct Other: Error {}
        #expect(AppModel.agentRunEndReason(Other()) == "other_error")
    }

    @Test
    func theTraceWritesOneChatEndedRecord() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("chat-ended-\(UUID().uuidString).jsonl").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        let trace = try #require(AgentInferenceTrace.forTesting(path: path))
        await trace.chatEnded(reason: "host_refusal", detail: "VisionCapture: error (SIMULATOR_WINDOW_NOT_FOUND)")
        let lines = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n")
        #expect(lines.count == 1)
        let record = try #require(try JSONDecoder().decode(JSONValue.self, from: Data(lines[0].utf8)).objectValue)
        #expect(record["event"] == .string("chat_ended"))
        #expect(record["reason"] == .string("host_refusal"))
        #expect(record["detail"] == .string("VisionCapture: error (SIMULATOR_WINDOW_NOT_FOUND)"))
        // No step ran yet: the step fields are null, the record is still written.
        #expect(record["step_index"] == .null)
    }

    @Test
    func traceRecordsActualThinkingSwitchReason() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("thinking-switch-" + UUID().uuidString + ".jsonl").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        let trace = try #require(AgentInferenceTrace.forTesting(path: path))
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/fixture-model"), prompt: "fixture",
            runtimeOptions: AppRuntimeOptions(toolThinkingEnabled: true))
        _ = await trace.begin(request, thinkingReason: "no_tool_call")
        let lines = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n")
        #expect(lines.contains { line in
            guard let object = try? JSONDecoder().decode(
                JSONValue.self, from: Data(line.utf8)).objectValue else { return false }
            return object["event"] == .string("thinking_switch")
                && object["enabled"] == .bool(true)
                && object["reason"] == .string("no_tool_call")
        })
    }

    @Test
    func thoughtRecoveryKeepsInternalReasonAndMapsTraceReason() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("thinking-recovery-" + UUID().uuidString + ".jsonl").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        let trace = try #require(AgentInferenceTrace.forTesting(path: path))
        let epoch = UUID()
        let request = AgentThinkingRequest(enabled: true, reason: "tool_result_error")
        let enabled = await request.selection()
        let onRequest = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/fixture-model"), prompt: "fixture",
            runtimeOptions: AppRuntimeOptions(toolThinkingEnabled: enabled.enabled),
            conversationEpoch: epoch, turnIndex: 0,
            toolTurn: .user(developerPrompt: nil, tools: []))
        _ = await trace.begin(onRequest, thinkingReason: enabled.traceReason)

        await request.recoverThought()
        let recovery = await request.selection()
        #expect(recovery == AgentThinkingSelection(enabled: false, reason: "thought_repetition_recovery"))
        #expect(recovery.traceReason == "repetition_retry")
        let offRequest = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/fixture-model"), prompt: "fixture",
            runtimeOptions: AppRuntimeOptions(toolThinkingEnabled: recovery.enabled),
            continuesConversation: true, conversationTokens: 1,
            conversationEpoch: epoch, turnIndex: 1,
            toolTurn: .results([AppToolResult(
                callID: "call", name: "visioncapture_navigate", content: "retained result")]))
        _ = await trace.begin(offRequest, thinkingReason: recovery.traceReason)

        let lines = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n")
        let records = try lines.map { try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8)) }
        let switches = records.filter { $0.objectValue?["event"] == .string("thinking_switch") }
        #expect(switches.count == 2)
        #expect(switches[0].objectValue?["reason"] == .string("tool_result_error"))
        #expect(switches[1].objectValue?["enabled"] == .bool(false))
        #expect(switches[1].objectValue?["reason"] == .string("repetition_retry"))
        #expect(switches[1].objectValue?["conversation_id"] == .string(epoch.uuidString))
        #expect(switches[1].objectValue?["turn_index"] == .integer(1))
    }

    @Test
    func traceEmitsOneRecordPerModeChangeWithStepIdentity() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("thinking-sequence-" + UUID().uuidString + ".jsonl").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        let trace = try #require(AgentInferenceTrace.forTesting(path: path))
        let epoch = UUID()
        func request(_ enabled: Bool, turn: Int) -> AppGenerationRequest {
            AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp/fixture-model"),
                prompt: "fixture", runtimeOptions: AppRuntimeOptions(toolThinkingEnabled: enabled),
                conversationEpoch: epoch, turnIndex: turn,
                toolTurn: turn == 0 ? .user(developerPrompt: nil, tools: [])
                    : .results([AppToolResult(callID: "call", name: "visioncapture_navigate", content: "result")]))
        }
        let requests = [request(false, turn: 0), request(false, turn: 1),
            request(true, turn: 1), request(false, turn: 1)]
        let reasons = ["agent_default", "agent_default", "tool_result_error", "one_request_complete"]
        var steps: [AgentInferenceTrace.Step] = []
        for index in requests.indices {
            steps.append(await trace.begin(requests[index], thinkingReason: reasons[index]))
        }
        let lines = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n")
        let records = try lines.map { try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8)) }
        let switches = records.filter { $0.objectValue?["event"] == .string("thinking_switch") }
        #expect(switches.count == 3)
        for index in requests.indices {
            let step = steps[index]
            let inputRecord = try #require(records.first { record in
                record.objectValue?["event"] == .string("input")
                    && record.objectValue?["step_index"] == .integer(Int64(step.index))
            }?.objectValue)
            let input = try #require(inputRecord["input"]?.objectValue)
            #expect(inputRecord["step_id"] == .string(step.id.uuidString))
            #expect(inputRecord["conversation_id"] == .string(epoch.uuidString))
            #expect(inputRecord["turn_index"] == .integer(Int64(requests[index].turnIndex ?? -1)))
            #expect(input["settings"]?.objectValue?["enable_thinking"]
                == .bool(requests[index].runtimeOptions.toolThinkingEnabled))
            let matchingSwitch = switches.first {
                $0.objectValue?["step_index"] == .integer(Int64(step.index))
            }?.objectValue
            if index == 1 {
                #expect(matchingSwitch == nil, "A consecutive duplicate mode records input without a switch")
                continue
            }
            let switchRecord = try #require(matchingSwitch)
            #expect(switchRecord["step_id"] == .string(step.id.uuidString))
            #expect(switchRecord["enabled"] == .bool(requests[index].runtimeOptions.toolThinkingEnabled))
            #expect(switchRecord["reason"] == .string(reasons[index]))
            #expect(records.first { $0.objectValue?["event"] == .string("thinking_switch")
                && $0.objectValue?["step_index"] == .integer(Int64(step.index)) }?.objectValue?["conversation_id"]
                == .string(epoch.uuidString))
            #expect(records.first { $0.objectValue?["event"] == .string("thinking_switch")
                && $0.objectValue?["step_index"] == .integer(Int64(step.index)) }?.objectValue?["turn_index"]
                == .integer(Int64(requests[index].turnIndex ?? -1)))
        }

        let receiptJSON = """
        {
          "canRetryToolResult": true,
          "pendingCallID": "call",
          "pendingToolName": "visioncapture_navigate",
          "restoredTokenCount": 9000,
          "requiresRebuild": false,
          "generatedTokens": 200,
          "thinkingTokens": 160,
          "blockTokens": 20,
          "repetitions": 8
        }
        """
        let receipt = try JSONDecoder().decode(ThoughtRepetitionRecovery.self, from: Data(receiptJSON.utf8))
        await trace.finish(steps[2], content: "", calls: [], diagnostics: nil,
            error: "repeated", thoughtRepetitionRecovery: receipt)
        let updatedLines = try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n")
        let updatedRecords = try updatedLines.map {
            try JSONDecoder().decode(JSONValue.self, from: Data($0.utf8))
        }
        let outputRecord = try #require(updatedRecords.first {
            $0.objectValue?["event"] == .string("error")
                && $0.objectValue?["step_index"] == .integer(Int64(steps[2].index))
        }?.objectValue)
        let output = try #require(outputRecord["output"]?.objectValue)
        #expect(output["thought_repetition_recovery"]?.objectValue?["pendingCallID"] == .string("call"))
    }

    @Test
    func closedTraceWriterDoesNotThrowDuringInferenceRecords() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("thinking-closed-" + UUID().uuidString + ".jsonl").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        let trace = try #require(AgentInferenceTrace.forTesting(path: path))
        await trace.closeWriterForTesting()
        let request = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp/fixture-model"),
            prompt: "fixture", runtimeOptions: AppRuntimeOptions(toolThinkingEnabled: false))
        let step = await trace.begin(request, thinkingReason: "agent_default")
        await trace.finish(step, content: "done", calls: [], diagnostics: nil)
    }

    @Test(arguments: RetryMatrixCase.allCases)
    func everyRetainedBoundaryGetsOneSameTurnOffRetry(_ testCase: RetryMatrixCase) async throws {
        let source = try await Self.matrixTurn(for: testCase.entry)
        let loop = VisionCaptureToolLoop()
        let thinking = AgentThinkingRequest(enabled: true, reason: source.selection.reason)
        let calls = AgentRunMutex<[(turn: AppToolTurn, selection: AgentThinkingSelection)]>([])
        let recoveryStatuses = AgentRunMutex<[AppAgentActivity.Status]>([])
        let unexpectedHostRequests = AgentRunMutex<[String]>([])
        let trapCategory = Self.hostTrapCategory(for: testCase.entry)
        await loop.setHostRequestForTesting { arguments in
            let request: String
            if case .string(let value)? = arguments.objectValue?["request"] { request = value }
            else { request = "unknown request" }
            unexpectedHostRequests.withLock { $0.append(request) }
            throw VisionCaptureAgentError.noProgress(
                "Unexpected \(trapCategory) host access during the common retry: \(request)")
        }
        var returned: VisionCaptureModelCompletion?
        var terminal: Error?
        do {
            returned = try await loop.generateModelDecision(
                turn: source.turn, thinking: thinking,
                activity: { event in
                    if case .generationRecovery(_, _, let status) = event {
                        recoveryStatuses.withLock { $0.append(status) }
                    }
                },
                inference: { turn, request in
                    let selection = await request.selection()
                    calls.withLock { $0.append((turn, selection)) }
                    if calls.withLock({ $0.count }) == 1 {
                        throw try Self.matrixReceipt(for: turn)
                    }
                    if testCase.offOutcome == .emptyCompletion {
                        return VisionCaptureModelCompletion(content: "", toolCalls: [],
                            diagnostics: VisionCaptureThinkingPolicyTests.diagnostics(stopReason: .endOfTurn,
                                thinking: selection.enabled))
                    }
                    return VisionCaptureThinkingPolicyTests.completion(
                        call: VisionCaptureThinkingPolicyTests.observeCall("matrix-off-usable"),
                        thinking: selection.enabled)
                })
        } catch {
            terminal = error
        }

        let attempts = calls.withLock { $0 }
        #expect(attempts.count == 2, "\(testCase): the common site must make one physical OFF retry")
        guard attempts.count == 2 else {
            let escaped = String(describing: terminal)
            let message = "\(testCase): expected an OFF callback after the ON receipt; received \(attempts.count)."
            Issue.record("\(message) Escaped error: \(escaped)")
            return
        }
        #expect(terminal.map { String(describing: $0) } == nil,
            "\(testCase): the common retry should return its OFF completion")
        #expect(attempts[0].turn == source.turn)
        #expect(attempts[1].turn == source.turn)
        #expect(attempts[0].selection == AgentThinkingSelection(
            enabled: true, reason: source.selection.reason))
        #expect(attempts[1].selection == AgentThinkingSelection(
            enabled: false, reason: "thought_repetition_recovery"))
        let finalSelection = await thinking.selection()
        #expect(finalSelection == AgentThinkingSelection(
            enabled: false, reason: "thought_repetition_recovery"))
        #expect(recoveryStatuses.withLock { $0 } == [.dispatching, .succeeded])
        #expect(unexpectedHostRequests.withLock { $0 }.isEmpty,
            "\(testCase): no host read, capture, or dispatch may occur in the common retry")
        #expect(source.selection.enabled)
        guard let returned else {
            Issue.record("\(testCase): the common site did not return the OFF completion")
            return
        }
        if testCase.offOutcome == .emptyCompletion {
            #expect(returned.content.isEmpty)
            #expect(returned.toolCalls.isEmpty)
            #expect(returned.diagnostics.stopReason == .endOfTurn)
        } else {
            #expect(returned.toolCalls.count == 1)
            #expect(returned.toolCalls.first?.id == "matrix-off-usable")
            #expect(returned.toolCalls.first?.name == "visioncapture_navigate")
        }
        #expect(source.counterfactual == testCase.entry.isCounterfactual,
            "\(testCase): source-derived packet cases must be identified honestly")
        if !source.counterfactual {
            #expect(source.selection.reason == Self.expectedTrigger(for: testCase.entry))
        }
        if testCase.entry == .navigationResults || testCase.entry == .noVisionPackResult {
            guard case .results(let results) = source.turn, results.count == 1 else {
                Issue.record("The successful navigation prelude must retain one result packet")
                return
            }
            let packet: JSONValue
            do {
                packet = try JSONDecoder().decode(JSONValue.self, from: Data(results[0].content.utf8))
            } catch {
                Issue.record("The successful navigation packet must remain valid JSON: \(error)")
                return
            }
            #expect(packet.objectValue?["last_action"]?.objectValue?["verdict"] == .string("verified"))
            #expect(packet.objectValue?["last_action"]?.objectValue?["dispatch_attempted"] == .bool(true))
            if testCase.entry == .noVisionPackResult {
                #expect(AppVisionPackInstallationProbe.status(
                    at: VisionCaptureThinkingPolicyTests.configuration.modelDirectory) != .complete)
            }
        }
        if source.counterfactual {
            Self.expectCounterfactualEvidence(source, entry: testCase.entry)
        }
    }

    enum RetryMatrixEntry: String, CaseIterable, Sendable, Equatable {
        case navigationResults
        case checkpoint
        case localHistoryResults
        case guardedRefusalResult
        case openingUser
        case repeatedObserveResults
        case unknownToolCorrection
        case observationRefusal
        case staleRecovery
        case deliveryUnknown
        case noVisionPackResult
        case textContextCapacity
        case retainedImageCapacity
        case supplementaryReadFailure
        case supplementaryCaptureFailure

        var isCounterfactual: Bool {
            switch self {
            case .observationRefusal, .staleRecovery, .textContextCapacity,
                 .retainedImageCapacity, .supplementaryReadFailure, .supplementaryCaptureFailure:
                return true
            default:
                return false
            }
        }
    }

    enum RetryMatrixOffOutcome: Sendable, Equatable {
        case usableToolCall
        case emptyCompletion
    }

    struct RetryMatrixCase: Sendable, CustomStringConvertible {
        let entry: RetryMatrixEntry
        let offOutcome: RetryMatrixOffOutcome

        static let allCases = RetryMatrixEntry.allCases.flatMap { entry in
            [RetryMatrixOffOutcome.usableToolCall, .emptyCompletion].map {
                RetryMatrixCase(entry: entry, offOutcome: $0)
            }
        }

        var description: String { "\(entry.rawValue)/\(offOutcome)" }
    }

    private struct MatrixTurnSource {
        let turn: AppToolTurn
        let selection: AgentThinkingSelection
        let counterfactual: Bool
        let maxContextTokens: Int?
    }

    private struct StopAtMatrixBoundary: Error {}

    private static let matrixRestoredTokenCount = 9_000

    @Test(arguments: [RetryMatrixEntry.openingUser, .checkpoint])
    func openingAndCheckpointRejectPendingResultReceipts(_ entry: RetryMatrixEntry) async throws {
        let source = try await Self.matrixTurn(for: entry)
        let loop = VisionCaptureToolLoop()
        let unexpectedHostRequests = AgentRunMutex<[String]>([])
        await loop.setHostRequestForTesting { arguments in
            let request: String
            if case .string(let value)? = arguments.objectValue?["request"] { request = value }
            else { request = "unknown request" }
            unexpectedHostRequests.withLock { $0.append(request) }
            throw VisionCaptureAgentError.noProgress(
                "Unexpected host request in boundary negative: \(request)")
        }
        let thinking = AgentThinkingRequest(enabled: true, reason: source.selection.reason)
        let attempts = AgentRunMutex<[(AppToolTurn, AgentThinkingSelection)]>([])
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.generateModelDecision(
                turn: source.turn, thinking: thinking, activity: { _ in },
                inference: { turn, request in
                    let selection = await request.selection()
                    attempts.withLock { $0.append((turn, selection)) }
                    throw try VisionCaptureThinkingPolicyTests.receipt(
                        callID: "pending-call-at-\(entry.rawValue)")
                })
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .noProgress(let message)? = terminal else {
            Issue.record("A pending-result receipt at \(entry) must fail closed as noProgress")
            return
        }
        #expect(message.contains("mismatched retained model boundary"))
        let observed = attempts.withLock { $0 }
        #expect(observed.count == 1, "The invalid receipt must not cause a physical OFF call")
        if let attempt = observed.first {
            #expect(attempt.0 == source.turn)
            #expect(attempt.1.enabled)
        }
        #expect(unexpectedHostRequests.withLock { $0 }.isEmpty)
        let finalSelection = await thinking.selection()
        #expect(finalSelection == AgentThinkingSelection(enabled: true, reason: source.selection.reason))
    }

    private static func matrixTurn(for entry: RetryMatrixEntry) async throws -> MatrixTurnSource {
        if entry.isCounterfactual {
            return try await counterfactualMatrixTurn(for: entry)
        }

        let loop = VisionCaptureToolLoop()
        let failedOrdinal = entry == .guardedRefusalResult ? 2 : nil
        let unknownOrdinal = entry == .deliveryUnknown ? 2 : nil
        let swipeVerdicts: [String]
        switch entry {
        case .navigationResults, .noVisionPackResult:
            swipeVerdicts = ["verified"]
        case .guardedRefusalResult, .deliveryUnknown:
            swipeVerdicts = ["inconclusive"]
        default:
            swipeVerdicts = ["failed"]
        }
        let host = VisionCaptureThinkingPolicyTests.HostFixture(failedSwipeOrdinal: failedOrdinal,
            deliveryUnknownSwipeOrdinal: unknownOrdinal,
            swipeVerdicts: swipeVerdicts)
        await loop.setFlowSessionForTesting("flow-35")
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let captured = AgentRunMutex<MatrixTurnSource?>(nil)
        let requests = AgentRunMutex(0)
        let forceCheckpoint = AgentRunMutex(false)
        do {
            _ = try await loop.run(configuration: VisionCaptureThinkingPolicyTests.configuration, activity: { _ in },
                maxContextTokens: 32_768,
                checkpoint: { proposal in
                    let isSwipe = proposal.call.arguments.objectValue?["action"] == .string("swipe")
                    return DecodeContextCheckpointReceipt(
                        checkpointID: proposal.id, replacementEpoch: proposal.replacementEpoch,
                        committed: proposal.commit,
                        needed: entry == .checkpoint && !proposal.commit && isSwipe,
                        existingPromptTokens: 9_000, replacementPromptTokens: 1_000,
                        reserveTokens: 100, resultAllowanceTokens: 100, retainedImageCount: 0,
                        retainedImageRows: 0, retainedFeatureBytes: 0, preparationSeconds: 0)
                },
                forceCheckpoint: { forceCheckpoint.withLock { $0 } }) { turn, request in
                    let selection = await request.selection()
                    let index = requests.withLock { value in value += 1; return value }
                    if selection.enabled, Self.isMatrixTarget(turn, entry: entry) {
                        captured.withLock {
                            $0 = MatrixTurnSource(turn: turn, selection: selection,
                                counterfactual: false, maxContextTokens: nil)
                        }
                        throw StopAtMatrixBoundary()
                    }
                    switch entry {
                    case .navigationResults, .noVisionPackResult:
                        switch index {
                        case 1:
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.observeCall("matrix-setup"),
                                thinking: selection.enabled)
                        case 2:
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.swipeCall("matrix-swipe"),
                                thinking: selection.enabled)
                        case 3:
                            return VisionCaptureModelCompletion(content: "", toolCalls: [],
                                diagnostics: VisionCaptureThinkingPolicyTests.diagnostics(
                                    stopReason: .endOfTurn, thinking: selection.enabled))
                        default:
                            return VisionCaptureThinkingPolicyTests.finalCompletion(thinking: selection.enabled)
                        }
                    case .guardedRefusalResult, .deliveryUnknown:
                        switch index {
                        case 1:
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.observeCall("matrix-setup"),
                                thinking: selection.enabled)
                        case 2, 3:
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.swipeCall("matrix-swipe-\(index)"),
                                thinking: selection.enabled)
                        default:
                            return VisionCaptureThinkingPolicyTests.finalCompletion(thinking: selection.enabled)
                        }
                    case .checkpoint:
                        switch index {
                        case 1:
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.observeCall("checkpoint-setup"),
                                thinking: selection.enabled)
                        case 2:
                            forceCheckpoint.withLock { $0 = true }
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.swipeCall("checkpoint-trigger"),
                                thinking: selection.enabled)
                        default:
                            return VisionCaptureThinkingPolicyTests.finalCompletion(thinking: selection.enabled)
                        }
                    case .localHistoryResults:
                        if index <= 2 {
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.historyCall("matrix-history-\(index)"),
                                thinking: selection.enabled)
                        }
                        return VisionCaptureThinkingPolicyTests.finalCompletion(thinking: selection.enabled)
                    case .openingUser:
                        return VisionCaptureModelCompletion(content: "", toolCalls: [],
                            diagnostics: VisionCaptureThinkingPolicyTests.diagnostics(stopReason: .endOfTurn,
                                thinking: selection.enabled))
                    case .repeatedObserveResults:
                        if index <= 2 {
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.observeCall("matrix-observe-\(index)"),
                                thinking: selection.enabled)
                        }
                        return VisionCaptureThinkingPolicyTests.finalCompletion(thinking: selection.enabled)
                    case .unknownToolCorrection:
                        if index <= 2 {
                            return VisionCaptureThinkingPolicyTests.completion(
                                call: VisionCaptureThinkingPolicyTests.unknownCall("matrix-unknown-\(index)"),
                                thinking: selection.enabled)
                        }
                        return VisionCaptureThinkingPolicyTests.finalCompletion(thinking: selection.enabled)
                    case .observationRefusal, .staleRecovery, .textContextCapacity,
                         .retainedImageCapacity, .supplementaryReadFailure, .supplementaryCaptureFailure:
                        Issue.record("Counterfactual rows must use production packet builders")
                        return VisionCaptureThinkingPolicyTests.finalCompletion(thinking: selection.enabled)
                    }
                }
            Issue.record("The full-loop prelude did not reach an ON model boundary for \(entry)")
        } catch is StopAtMatrixBoundary {
            // Capture stops at the exact retained ON input, before its inference can retry.
        }
        guard let result = captured.withLock({ $0 }) else {
            throw VisionCaptureAgentError.noProgress("Could not capture the \(entry) model boundary")
        }
        return result
    }

    private static func isMatrixTarget(_ turn: AppToolTurn, entry: RetryMatrixEntry) -> Bool {
        switch (entry, turn) {
        case (.checkpoint, .checkpoint): return true
        case (.openingUser, .user): return true
        case (.localHistoryResults, .results(let results)):
            return results.count == 1 && results[0].name == "task_history_read"
        case (.unknownToolCorrection, .results(let results)):
            return results.count == 1 && results[0].name == "task_future_unknown"
        case (.navigationResults, .results(let results)),
             (.guardedRefusalResult, .results(let results)),
             (.repeatedObserveResults, .results(let results)),
             (.deliveryUnknown, .results(let results)),
             (.noVisionPackResult, .results(let results)):
            return results.count == 1 && results[0].name == "visioncapture_navigate"
        default:
            return false
        }
    }

    private static func expectedTrigger(for entry: RetryMatrixEntry) -> String {
        switch entry {
        case .openingUser: return "no_tool_call"
        case .checkpoint: return "tool_proof_failed"
        case .navigationResults, .noVisionPackResult: return "no_tool_call"
        case .localHistoryResults, .repeatedObserveResults, .unknownToolCorrection:
            return "repeated_tool_call"
        case .guardedRefusalResult, .deliveryUnknown: return "tool_result_error"
        case .observationRefusal, .staleRecovery, .textContextCapacity, .retainedImageCapacity,
             .supplementaryReadFailure, .supplementaryCaptureFailure:
            return "tool_proof_failed"
        }
    }

    private static func hostTrapCategory(for entry: RetryMatrixEntry) -> String {
        switch entry {
        case .supplementaryReadFailure: "supplementary-read trap"
        case .supplementaryCaptureFailure: "supplementary-capture trap"
        default: "unexpected host-access trap"
        }
    }

    private static func expectCounterfactualEvidence(
        _ source: MatrixTurnSource, entry: RetryMatrixEntry
    ) {
        guard case .results(let results) = source.turn, results.count == 1 else {
            Issue.record("Counterfactual builder rows must retain one result packet")
            return
        }
        let packet: JSONValue
        do {
            packet = try JSONDecoder().decode(JSONValue.self, from: Data(results[0].content.utf8))
        } catch {
            Issue.record("Counterfactual production packet must remain valid JSON: \(error)")
            return
        }
        let body = packet.objectValue ?? [:]
        let expectedVerdict = entry == .retainedImageCapacity ? "observed" : "failed"
        #expect(body["last_action"]?.objectValue?["verdict"] == .string(expectedVerdict))
        switch entry {
        case .observationRefusal:
            #expect(body["observation"]?.objectValue?["read_result"] == .string("refreshed"))
            #expect(body["guidance"] == .string(
                "Screen text positions unavailable: the screenshot reply had no content. "
                    + "The screen changed during observation. These choices come from a fresh read. "
                    + "No pending input was sent. Choose again and do not assume a form was saved."))
            #expect(body["last_action"]?.objectValue?["dispatch_attempted"] == .bool(false))
            #expect(body["observation_refusal"] == nil,
                "The model-facing packet keeps the production projection, not private executor metadata")
        case .staleRecovery:
            #expect(body["guidance"] == .string(
                "Screen text positions unavailable: the screenshot reply had no content. "
                    + "Nothing was pressed. The follow-up read found unchanged screen content. "
                    + "You may choose the target marked confirmation_attempts: 1 once, "
                    + "or observe and choose a different action. "
                    + "The refused press was not retried."))
            #expect(body["last_action"]?.objectValue?["dispatch_attempted"] == nil)
            #expect(body["last_action"]?.objectValue?["screen_changed"] == .bool(false))
        case .textContextCapacity:
            #expect(results[0].content.utf8.count > 128 * 1_024)
            #expect(source.maxContextTokens == 64)
            #expect((source.maxContextTokens ?? .max) < Self.matrixRestoredTokenCount,
                "The configured context limit is below the receipt's restored-token count")
        case .retainedImageCapacity:
            #expect(results[0].imageAttachments.count == 1)
            let capacity = VisionImageTokenBudget.capacity(maxContext: 9_000, reservedTextTokens: 9_000)
            #expect(capacity == 0)
            #expect(results[0].imageAttachments.count > capacity)
            #expect(body["image"]?.objectValue?["mime_type"] == .string("image/png"))
        case .supplementaryReadFailure, .supplementaryCaptureFailure:
            #expect(body["last_action"]?.objectValue?["action"] == .string("swipe"))
            #expect(body["last_action"]?.objectValue?["verdict"] == .string("failed"))
            #expect(body["supplementary_read"] == nil)
            #expect(body["supplementary_capture"] == nil)
        default:
            Issue.record("A reachable full-loop row was mislabeled as counterfactual")
        }
    }

    private static func counterfactualMatrixTurn(
        for entry: RetryMatrixEntry
    ) async throws -> MatrixTurnSource {
        let loop = VisionCaptureToolLoop()
        let image = AppImageAttachment(
            id: UUID(), fileURL: URL(fileURLWithPath: "/tmp/task-35d-fixture.png"),
            displayName: "fixture.png", encodedBytes: 1, sha256: String(repeating: "a", count: 64))
        let packet: String
        var images: [AppImageAttachment] = []
        switch entry {
        case .retainedImageCapacity:
            packet = try await loop.screenshotPacketForTesting(
                read: VisionCaptureThinkingPolicyTests.HostFixture.result(), screenTextReply: .object([:]),
                configuration: VisionCaptureThinkingPolicyTests.configuration, images: [image])
            images = [image]
        default:
            var body: [String: JSONValue] = [
                "proof": .object([
                    "verdict": .string("failed"), "reason_code": .string("fixture"),
                    "action": .string("swipe"),
                ]),
            ]
            switch entry {
            case .observationRefusal:
                body["observation_refusal"] = .object([
                    "error": .string("CACHE_ACTION_CAPABILITY_STALE"),
                    "cache_authorization_phase": .string("core_observation_completion"),
                    "cache_authorization_reason": .string("live_screen_binding_mismatch"),
                    "recovery_action": .string("observe_and_inspect"),
                    "recovery_reason": .string("observation_topology_changed_before_completion"),
                    "observation_grant_retired": .bool(true),
                    "cache_used": .bool(false), "cache_revalidation_used": .bool(false),
                    "revalidation_verified": .bool(false), "fresh_authority_recorded": .bool(false),
                    "dispatch_attempted": .bool(false),
                ])
                body["observation_outcome"] = .string("refreshed")
                body["outcome"] = .string("not_dispatched")
                body["dispatch_attempted"] = .bool(false)
                body["instruction"] = .string(
                    "The screen changed during observation. These choices come from a fresh read. "
                        + "No pending input was sent. Choose again and do not assume a form was saved.")
            case .staleRecovery:
                body["outcome"] = .string("not_dispatched_reobserved")
                body["stale_recovery"] = .object([
                    "code": .string("CACHE_ACTION_CAPABILITY_STALE"),
                    "dispatch_attempted": .bool(false), "screen_changed": .bool(false),
                    "remaining_confirmation_attempts": .integer(1),
                ])
                body["screen_changed"] = .bool(false)
                body["instruction"] = .string(
                    "Nothing was pressed. The follow-up read found unchanged screen content. "
                        + "You may choose the target marked confirmation_attempts: 1 once, "
                        + "or observe and choose a different action. "
                        + "The refused press was not retried.")
            case .textContextCapacity:
                body["instruction"] = .string(String(repeating: "x", count: 128 * 1_024))
            default:
                break
            }
            let built = try await loop.actionPacketForTesting(
                operation: entry == .observationRefusal ? "observe" : "swipe",
                read: VisionCaptureThinkingPolicyTests.HostFixture.result(), body: body,
                configuration: VisionCaptureThinkingPolicyTests.configuration)
            packet = built.packet
        }
        #expect(!packet.contains("test_shape"))
        let content = packet
        let result = AppToolResult(callID: "matrix-\(entry.rawValue)",
            name: "visioncapture_navigate", content: content, imageAttachments: images)
        let reason = entry == .retainedImageCapacity ? "no_tool_call" : "tool_proof_failed"
        return MatrixTurnSource(turn: .results([result]),
            selection: AgentThinkingSelection(enabled: true, reason: reason),
            counterfactual: true,
            maxContextTokens: entry == .textContextCapacity ? 64 : nil)
    }

    private static func matrixReceipt(for turn: AppToolTurn) throws -> AppInferenceError {
        switch turn {
        case .user, .checkpoint:
            return try VisionCaptureThinkingPolicyTests.openingReceipt()
        case .results(let results):
            guard results.count == 1, let result = results.first else {
                throw VisionCaptureAgentError.noProgress("The matrix result turn must contain one result")
            }
            return try VisionCaptureThinkingPolicyTests.receipt(
                callID: result.callID, toolName: result.name)
        }
    }
}

private final class AgentRunMutex<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<Result>(_ body: (inout Value) -> Result) -> Result {
        lock.lock(); defer { lock.unlock() }
        return body(&value)
    }
}
