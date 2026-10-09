import Foundation
import Testing
import TurboFieldfare
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
        #expect(AppModel.agentRunEndReason(VisionCaptureAgentError.proposalCorrectionExhausted("paused")) == "proposal_pause")
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
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("chat-ended-\(UUID().uuidString).jsonl").path
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
}
