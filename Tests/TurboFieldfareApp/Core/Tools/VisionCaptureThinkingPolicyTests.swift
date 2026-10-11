import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct VisionCaptureThinkingPolicyTests {
    @Test
    func emptyCompletionUsesOneThinkingRequestThenReturnsToOff() async throws {
        let loop = VisionCaptureToolLoop()
        let modes = Mutex<[AgentThinkingSelection]>([])
        let attempts = Mutex(0)
        let configuration = VisionCaptureAgentConfiguration(
            bundleIdentifier: "com.example.fixture",
            simulatorUDID: "00000000-0000-0000-0000-000000000001",
            modelDirectory: URL(fileURLWithPath: "/tmp/fixture-model"))
        let result = try await loop.run(configuration: configuration, activity: { _ in }) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            let attempt = attempts.withLock { value in
                value += 1
                return value
            }
            let content = attempt == 1 ? "" : "completed"
            return VisionCaptureModelCompletion(
                content: content, toolCalls: [],
                diagnostics: AppDiagnostics(
                    generatedTokens: 1, stopReason: .endOfTurn, promptTokenCount: 1,
                    timeToFirstTokenSeconds: nil, decodeSeconds: 0.1,
                    tokensPerSecond: 10, peakMemoryBytes: nil,
                    runtimeOptions: AppRuntimeOptions(toolThinkingEnabled: selection.enabled)))
        }
        #expect(result.answer == "completed")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "no_tool_call"),
        ])
    }

    @Test
    func secondEmptyCompletionEndsWithIncompleteAnswer() async throws {
        let loop = VisionCaptureToolLoop()
        let modes = Mutex<[AgentThinkingSelection]>([])
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                return VisionCaptureModelCompletion(content: "", toolCalls: [],
                    diagnostics: Self.diagnostics(stopReason: .endOfTurn, thinking: selection.enabled))
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .incompleteAnswer? = terminal else {
            Issue.record("A second empty completion must use the existing incompleteAnswer error")
            return
        }
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "no_tool_call"),
        ])
    }

    @Test
    func selectionTriggersAreOneShotAndRecoveryIsOff() async {
        let request = AgentThinkingRequest()
        #expect(await request.selection() == AgentThinkingSelection(enabled: false, reason: "agent_default"))
        #expect(await request.retryNoCall(unparseable: false))
        #expect(await request.selection() == AgentThinkingSelection(enabled: true, reason: "no_tool_call"))
        #expect(!(await request.retryNoCall(unparseable: false)))
        await request.recoverThought()
        #expect(await request.selection() == AgentThinkingSelection(
            enabled: false, reason: "thought_repetition_recovery"))
    }

    @Test
    func malformedCorrectionUsesThinkingOnceThenStops() async {
        let request = AgentThinkingRequest()
        await request.correctMalformed()
        #expect(await request.selection() == AgentThinkingSelection(
            enabled: true, reason: "unparseable_tool_call"))
        await request.correctMalformed()
        #expect(await request.selection() == AgentThinkingSelection(
            enabled: false, reason: "one_request_complete"))

        let correctionAfterOn = AgentThinkingRequest(enabled: true, reason: "tool_result_error")
        await correctionAfterOn.correctMalformed()
        #expect(await correctionAfterOn.selection() == AgentThinkingSelection(
            enabled: false, reason: "one_request_complete"))
    }

    @Test
    func settledErrorTakesPrecedenceOverProofFailureAndRepeat() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let describesAtOnBoundary = Mutex(0)
        let host = HostFixture(failedSwipeOrdinal: 2, swipeVerdicts: ["inconclusive"])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { turn, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            turns.withLock { $0.append(turn) }
            switch modes.withLock({ $0.count }) {
            case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
            case 2: return Self.completion(call: Self.swipeCall("first-swipe"), thinking: selection.enabled)
            case 3: return Self.completion(call: Self.swipeCall("second-swipe"), thinking: selection.enabled)
            case 4:
                describesAtOnBoundary.withLock { $0 = host.describeCount }
                throw try Self.receipt(callID: "second-swipe")
            case 5:
                #expect(host.describeCount == describesAtOnBoundary.withLock { $0 })
                return Self.completion(call: Self.observeCall("safe-off-observe"), thinking: selection.enabled)
            case 6:
                #expect(host.describeCount == describesAtOnBoundary.withLock { $0 } + 1)
                return Self.finalCompletion(thinking: selection.enabled)
            default: return Self.finalCompletion(thinking: selection.enabled)
            }
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "tool_result_error"),
            AgentThinkingSelection(enabled: false, reason: "thought_repetition_recovery"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
        ])
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 2)
        Self.expectSameResultTurn(turns.withLock { $0 }, onIndex: 3, offIndex: 4,
            callID: "second-swipe")
    }

    @Test
    func repeatedThinkingRetainsTheCurrentActionOfferForTheOffDecision() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let host = HostFixture(publishesReadyAction: true)
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let hostRequestsAtOnBoundary = Mutex(0)
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { turn, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            turns.withLock { $0.append(turn) }
            switch modes.withLock({ $0.count }) {
            case 1:
                return Self.completion(call: Self.observeCall("offer-observe-1"), thinking: selection.enabled)
            case 2:
                return Self.completion(call: Self.observeCall("offer-observe-2"), thinking: selection.enabled)
            case 3:
                hostRequestsAtOnBoundary.withLock { $0 = host.requestCount }
                throw try Self.receipt(callID: "offer-observe-2")
            case 4:
                #expect(host.requestCount == hostRequestsAtOnBoundary.withLock { $0 })
                let choiceID = try Self.firstChoiceID(in: turn)
                return Self.completion(call: AppToolCall(id: "offer-tap", name: "visioncapture_navigate",
                    arguments: .object(["action": .string("tap"), "target": .string(choiceID)])),
                    thinking: selection.enabled)
            default:
                return Self.finalCompletion(thinking: selection.enabled)
            }
        }

        #expect(result.answer == "done")
        #expect(modes.withLock { $0 }.map(\.enabled) == [false, false, true, false, false])
        #expect(modes.withLock { $0 }[2].reason == "repeated_tool_call")
        #expect(modes.withLock { $0 }[3].reason == "thought_repetition_recovery")
        #expect(modes.withLock { $0 }[4].reason == "agent_default")
        let capturedTurns = turns.withLock { $0 }
        #expect(capturedTurns.count == 5)
        if capturedTurns.count >= 4 {
            #expect(capturedTurns[2] == capturedTurns[3],
                "The OFF decision must receive the exact result packet and choices from the ON request")
            let onChoiceID = try Self.firstChoiceID(in: capturedTurns[2])
            let offChoiceID = try Self.firstChoiceID(in: capturedTurns[3])
            #expect(onChoiceID == offChoiceID)
        }
        let afterOn = Array(host.requests.withLock { $0 }.dropFirst(hostRequestsAtOnBoundary.withLock { $0 }))
        #expect(afterOn.first == "activate computer use",
            "The retained offer must proceed into the normal pointer action path without a refresh")
        let clickIndex = afterOn.firstIndex(of: "click pointer")
        #expect(clickIndex != nil)
        #expect(afterOn.filter { $0 == "click pointer" }.count == 1)
        #expect(afterOn.filter { $0 == "tap cached action" }.isEmpty)
        if let clickIndex {
            #expect(afterOn[..<clickIndex].filter {
                $0 == "describe screen" || $0 == "screenshot"
            }.isEmpty, "No extra read may retire the retained choice before pointer dispatch")
        }
    }

    @Test
    func guardedErrorOffRetryWithNoCallEndsIncompleteAnswer() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let describesAtOnBoundary = Mutex(0)
        let host = HostFixture(failedSwipeOrdinal: 2, swipeVerdicts: ["inconclusive"])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in }) { turn, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                turns.withLock { $0.append(turn) }
                switch modes.withLock({ $0.count }) {
                case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
                case 2: return Self.completion(call: Self.swipeCall("first-swipe"), thinking: selection.enabled)
                case 3: return Self.completion(call: Self.swipeCall("second-swipe"), thinking: selection.enabled)
                case 4:
                    describesAtOnBoundary.withLock { $0 = host.describeCount }
                    throw try Self.receipt(callID: "second-swipe")
                default:
                    #expect(host.describeCount == describesAtOnBoundary.withLock { $0 })
                    return VisionCaptureModelCompletion(content: "", toolCalls: [],
                        diagnostics: Self.diagnostics(stopReason: .endOfTurn, thinking: selection.enabled))
                }
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .incompleteAnswer? = terminal else {
            Issue.record("An empty special OFF retry must end with incompleteAnswer")
            return
        }
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "tool_result_error"),
            AgentThinkingSelection(enabled: false, reason: "thought_repetition_recovery"),
        ])
        #expect(host.describeCount == describesAtOnBoundary.withLock { $0 })
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 2)
        Self.expectSameResultTurn(turns.withLock { $0 }, onIndex: 3, offIndex: 4,
            callID: "second-swipe")
    }

    @Test(arguments: ["wrong-id", "wrong-name"])
    func guardedErrorRecoveryRejectsMismatchedReceiptIdentity(_ mismatch: String) async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let modes = Mutex<[AgentThinkingSelection]>([])
        let describesAtOnBoundary = Mutex(0)
        let host = HostFixture(failedSwipeOrdinal: 2, swipeVerdicts: ["inconclusive"])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                switch modes.withLock({ $0.count }) {
                case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
                case 2: return Self.completion(call: Self.swipeCall("first-swipe"), thinking: selection.enabled)
                case 3: return Self.completion(call: Self.swipeCall("second-swipe"), thinking: selection.enabled)
                default:
                    describesAtOnBoundary.withLock { $0 = host.describeCount }
                    if mismatch == "wrong-id" {
                        throw try Self.receipt(callID: "different-call")
                    }
                    throw try Self.receipt(callID: "second-swipe", toolName: "task_history_read")
                }
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .noProgress? = terminal else {
            Issue.record("A guarded-error receipt with mismatched identity must fail closed")
            return
        }
        #expect(modes.withLock { $0 }.map(\.enabled) == [false, false, false, true])
        #expect(host.describeCount == describesAtOnBoundary.withLock { $0 })
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 2)
    }

    @Test(arguments: ["inconclusive", "unverified", "not_dispatched"])
    func nearbyProofVerdictsDoNotEnableThinking(_ verdict: String) async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let modes = Mutex<[AgentThinkingSelection]>([])
        let host = HostFixture(swipeVerdicts: [verdict])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            switch modes.withLock({ $0.count }) {
            case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
            case 2: return Self.completion(call: Self.swipeCall("up"), thinking: selection.enabled)
            default: return Self.finalCompletion(thinking: selection.enabled)
            }
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
        ])
    }

    @Test
    func repeatedSemanticObserveProposalIgnoresCallIDsAndObjectKeyOrder() async throws {
        let loop = VisionCaptureToolLoop()
        let modes = Mutex<[AgentThinkingSelection]>([])
        let host = HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            let index = modes.withLock { $0.count }
            if index <= 2 {
                return Self.completion(call: Self.observeCall("repeat-\(index)", intent: "inspect current state",
                    reverseKeys: index == 2), thinking: selection.enabled)
            }
            return Self.finalCompletion(thinking: selection.enabled)
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "repeated_tool_call"),
        ])
    }

    @Test
    func changedSemanticArgumentsDoNotEnableThinking() async throws {
        let loop = VisionCaptureToolLoop()
        let modes = Mutex<[AgentThinkingSelection]>([])
        let host = HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            let index = modes.withLock { $0.count }
            if index <= 2 {
                return Self.completion(call: Self.observeCall("different-\(index)",
                    intent: index == 1 ? "inspect current state" : "confirm current state"),
                    thinking: selection.enabled)
            }
            return Self.finalCompletion(thinking: selection.enabled)
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
        ])
    }

    @Test
    func changingSwipeDirectionDoesNotCountAsARepeatedProposal() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let modes = Mutex<[AgentThinkingSelection]>([])
        let host = HostFixture(swipeVerdicts: ["verified", "verified"])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            switch modes.withLock({ $0.count }) {
            case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
            case 2: return Self.completion(call: Self.swipeCall("up"), thinking: selection.enabled)
            case 3: return Self.completion(call: Self.swipeCall("down", direction: "down"), thinking: selection.enabled)
            default: return Self.finalCompletion(thinking: selection.enabled)
            }
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 }.allSatisfy { !$0.enabled })
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 1)
        #expect(host.requests.withLock { $0.filter { $0 == "swipe down" }.count } == 1)
    }

    @Test
    func differentAllowedToolNameDoesNotCountAsRepeatedProposal() async throws {
        let loop = VisionCaptureToolLoop()
        let modes = Mutex<[AgentThinkingSelection]>([])
        let host = HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            switch modes.withLock({ $0.count }) {
            case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
            case 2: return Self.completion(call: Self.historyCall("history"), thinking: selection.enabled)
            default: return Self.finalCompletion(thinking: selection.enabled)
            }
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
        ])
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" || $0 == "swipe down" }.isEmpty })
    }

    @Test
    func multiStepWithoutTriggerRemainsOff() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let modes = Mutex<[AgentThinkingSelection]>([])
        let host = HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { _ in }) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            let index = modes.withLock { $0.count }
            if index == 1 { return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled) }
            if index == 2 {
                return Self.completion(
                    call: Self.swipeCall("ordinary-action"), thinking: selection.enabled)
            }
            return Self.finalCompletion(thinking: selection.enabled)
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
        ])
    }

    @Test
    func repeatedThoughtOnRetryIsTerminalAndDoesNotReplayAction() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let host = HostFixture(swipeVerdicts: ["failed"])
        let modes = Mutex<[AgentThinkingSelection]>([])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in },
                maxContextTokens: 32_768) { _, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                switch modes.withLock({ $0.count }) {
                case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
                case 2: return Self.completion(call: Self.swipeCall("second-repetition"), thinking: selection.enabled)
                case 3: throw try Self.receipt(callID: "second-repetition")
                case 4: throw try Self.receipt(callID: "second-repetition")
                default: return Self.finalCompletion(thinking: selection.enabled)
                }
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .noProgress? = terminal else {
            Issue.record("A second repeated-thought receipt on the OFF recovery must stop the run")
            return
        }
        #expect(modes.withLock { $0 }.map(\.enabled) == [false, false, true, false])
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 1)
    }

    @Test
    func emptyOffThoughtRecoveryUsesTheExistingIncompleteAnswerError() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let host = HostFixture(swipeVerdicts: ["failed"])
        let modes = Mutex<[AgentThinkingSelection]>([])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in },
                maxContextTokens: 32_768) { _, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                switch modes.withLock({ $0.count }) {
                case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
                case 2: return Self.completion(call: Self.swipeCall("empty-recovery"), thinking: selection.enabled)
                case 3: throw try Self.receipt(callID: "empty-recovery")
                default: return VisionCaptureModelCompletion(content: "", toolCalls: [],
                    diagnostics: Self.diagnostics(stopReason: .endOfTurn, thinking: selection.enabled))
                }
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .incompleteAnswer? = terminal else {
            Issue.record("An empty OFF recovery should end with incompleteAnswer")
            return
        }
        #expect(modes.withLock { $0 }.map(\.enabled) == [false, false, true, false])
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 1)
    }

    @Test
    func openingThoughtRecoveryRejectsPendingToolReceipt() async throws {
        let loop = VisionCaptureToolLoop()
        let modes = Mutex<[AgentThinkingSelection]>([])
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in }) {
                _, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                if selection.enabled { throw try Self.receipt(callID: "unexpected-pending") }
                return VisionCaptureModelCompletion(content: "", toolCalls: [],
                    diagnostics: Self.diagnostics(stopReason: .endOfTurn, thinking: false))
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .noProgress? = terminal else {
            Issue.record("Opening recovery must reject a pending-tool receipt")
            return
        }
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "no_tool_call"),
        ])
    }

    @Test(arguments: [false, true])
    func navigationReceiptIdentityMismatchFailsClosed(_ wrongToolName: Bool) async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let host = HostFixture(swipeVerdicts: ["failed"])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let modes = Mutex<[AgentThinkingSelection]>([])
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in }) {
                _, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                switch modes.withLock({ $0.count }) {
                case 1:
                    return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
                case 2:
                    return Self.completion(call: Self.swipeCall("bound-swipe"), thinking: selection.enabled)
                default:
                    throw try Self.receipt(
                        callID: wrongToolName ? "bound-swipe" : "wrong-swipe",
                        toolName: wrongToolName ? "task_history_read" : "visioncapture_navigate")
                }
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .noProgress? = terminal else {
            Issue.record("A navigation receipt with a mismatched identity must fail closed")
            return
        }
        #expect(modes.withLock { $0 }.map(\.enabled) == [false, false, true])
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 1)
    }

    @Test(arguments: [false, true])
    func historyReceiptIdentityMismatchFailsClosed(_ wrongToolName: Bool) async throws {
        let loop = VisionCaptureToolLoop()
        let host = HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let modes = Mutex<[AgentThinkingSelection]>([])
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in }) {
                _, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                switch modes.withLock({ $0.count }) {
                case 1:
                    return Self.completion(
                        call: Self.historyCall("history-first"), thinking: selection.enabled)
                case 2:
                    return Self.completion(
                        call: Self.historyCall("history-second"), thinking: selection.enabled)
                default:
                    throw try Self.receipt(
                        callID: wrongToolName ? "history-second" : "wrong-history",
                        toolName: wrongToolName ? "visioncapture_navigate" : "task_history_read")
                }
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .noProgress? = terminal else {
            Issue.record("A history receipt with a mismatched identity must fail closed")
            return
        }
        #expect(modes.withLock { $0 }.map(\.enabled) == [false, false, true])
        #expect(host.requests.withLock { $0.isEmpty })
    }

    @Test
    func legacyOffRecoveryDoesNotConsumeTheLaterOnRequestEntitlement() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let host = HostFixture(swipeVerdicts: ["inconclusive", "failed"])
        let modes = Mutex<[AgentThinkingSelection]>([])
        let recoveries = Mutex(0)
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let result = try await loop.run(configuration: Self.configuration, activity: { event in
            if case .generationRecovery(_, _, .dispatching) = event { recoveries.withLock { $0 += 1 } }
        }, maxContextTokens: 32_768) { _, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            switch modes.withLock({ $0.count }) {
            case 1: return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
            case 2: return Self.completion(call: Self.swipeCall("legacy-boundary"), thinking: selection.enabled)
            case 3: throw try Self.receipt(callID: "legacy-boundary")
            case 4:
                return Self.completion(
                    call: Self.swipeCall("later-on-boundary", direction: "down"),
                    thinking: selection.enabled)
            case 5: throw try Self.receipt(callID: "later-on-boundary")
            default: return Self.finalCompletion(thinking: selection.enabled)
            }
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "thought_repetition_recovery"),
            AgentThinkingSelection(enabled: true, reason: "tool_proof_failed"),
            AgentThinkingSelection(enabled: false, reason: "thought_repetition_recovery"),
        ])
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 1)
        #expect(host.requests.withLock { $0.filter { $0 == "swipe down" }.count } == 1)
        // The ordinary OFF legacy path emits two dispatching events; the later
        // ON common retry emits one. There is no fourth recovery dispatch.
        #expect(recoveries.withLock { $0 } == 3)
    }

    @Test
    func directOnRecoveryDoesNotRestoreTheLegacyOffRecoveryLatch() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-35")
        let host = HostFixture(swipeVerdicts: ["inconclusive"])
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let describesAtHistoryOn = Mutex(0)
        let describesAtNavigationOff = Mutex(0)
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: Self.configuration, activity: { _ in },
                maxContextTokens: 32_768) { turn, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                turns.withLock { $0.append(turn) }
                switch modes.withLock({ $0.count }) {
                case 1:
                    return Self.completion(call: Self.observeCall("setup"), thinking: selection.enabled)
                case 2:
                    return Self.completion(call: Self.historyCall("history-first"), thinking: selection.enabled)
                case 3:
                    return Self.completion(call: Self.historyCall("history-second"), thinking: selection.enabled)
                case 4:
                    describesAtHistoryOn.withLock { $0 = host.describeCount }
                    throw try Self.receipt(callID: "history-second", toolName: "task_history_read")
                case 5:
                    #expect(host.describeCount == describesAtHistoryOn.withLock { $0 })
                    return Self.completion(call: Self.swipeCall("after-direct-on"), thinking: selection.enabled)
                case 6:
                    describesAtNavigationOff.withLock { $0 = host.describeCount }
                    throw try Self.receipt(callID: "after-direct-on")
                default:
                    return Self.finalCompletion(thinking: selection.enabled)
                }
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .noProgress? = terminal else {
            Issue.record("Direct history recovery must retain the latch through unchanged navigation facts")
            return
        }
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "repeated_tool_call"),
            AgentThinkingSelection(enabled: false, reason: "thought_repetition_recovery"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
        ])
        #expect(host.requests.withLock { $0.filter { $0 == "swipe up" }.count } == 1)
        #expect(host.describeCount == describesAtNavigationOff.withLock { $0 })
        #expect(describesAtNavigationOff.withLock { $0 } >= describesAtHistoryOn.withLock { $0 })
        let capturedTurns = turns.withLock { $0 }
        #expect(capturedTurns.count == 6)
        if capturedTurns.count == 6 {
            if case .results(let onResults) = capturedTurns[3],
               case .results(let offResults) = capturedTurns[4] {
                #expect(onResults == offResults)
                #expect(onResults.first?.callID == "history-second")
                #expect(onResults.first?.name == "task_history_read")
            } else {
                Issue.record("The direct history retry must retain its original results turn")
            }
        }
    }

    static func diagnostics(stopReason: AppStopReason, thinking: Bool) -> AppDiagnostics {
        AppDiagnostics(generatedTokens: 1, stopReason: stopReason, promptTokenCount: 1,
            timeToFirstTokenSeconds: nil, decodeSeconds: 0.1, tokensPerSecond: 10,
            peakMemoryBytes: nil, runtimeOptions: AppRuntimeOptions(toolThinkingEnabled: thinking))
    }

    static let configuration = VisionCaptureAgentConfiguration(
        bundleIdentifier: "com.example.fixture",
        simulatorUDID: "00000000-0000-0000-0000-000000000001",
        modelDirectory: URL(fileURLWithPath: "/tmp/fixture-model"))

    static func observeCall(_ id: String, intent: String? = nil, reverseKeys: Bool = false) -> AppToolCall {
        var arguments: [String: JSONValue] = ["action": .string("observe")]
        if let intent { arguments["intent"] = .string(intent) }
        if reverseKeys, let intent {
            arguments = ["intent": .string(intent), "action": .string("observe")]
        }
        return AppToolCall(id: id, name: "visioncapture_navigate", arguments: .object(arguments))
    }

    static func swipeCall(_ id: String, direction: String = "up") -> AppToolCall {
        AppToolCall(id: id, name: "visioncapture_navigate", arguments: .object([
            "action": .string("swipe"), "direction": .string(direction),
        ]))
    }

    static func historyCall(_ id: String) -> AppToolCall {
        AppToolCall(id: id, name: "task_history_read", arguments: .object([
            "observation_id": .string("missing-observation"),
        ]))
    }

    static func unknownCall(_ id: String) -> AppToolCall {
        AppToolCall(id: id, name: "task_future_unknown", arguments: .object([
            "observation_id": .string("missing-observation"),
        ]))
    }

    static func completion(call: AppToolCall, thinking: Bool) -> VisionCaptureModelCompletion {
        VisionCaptureModelCompletion(content: "", toolCalls: [call],
            diagnostics: diagnostics(stopReason: .toolCalls, thinking: thinking))
    }

    static func finalCompletion(thinking: Bool) -> VisionCaptureModelCompletion {
        VisionCaptureModelCompletion(content: "done", toolCalls: [],
            diagnostics: diagnostics(stopReason: .endOfTurn, thinking: thinking))
    }

    private static func firstChoiceID(in turn: AppToolTurn) throws -> String {
        guard case .results(let results) = turn, results.count == 1 else {
            throw VisionCaptureAgentError.noProgress("The retained result turn was not a single result")
        }
        let packet: JSONValue
        do {
            packet = try JSONDecoder().decode(JSONValue.self, from: Data(results[0].content.utf8))
        } catch {
            throw VisionCaptureAgentError.noProgress("The retained result packet was invalid: \(error)")
        }
        guard case .array(let choices)? = packet.objectValue?["choices"],
              let first = choices.first?.objectValue,
              case .string(let id)? = first["id"] else {
            throw VisionCaptureAgentError.noProgress("The retained result did not publish a tap choice")
        }
        return id
    }

    static func receipt(
        callID: String, toolName: String = "visioncapture_navigate"
    ) throws -> AppInferenceError {
        let json = """
        {
          "canRetryToolResult": true,
          "pendingCallID": "\(callID)",
          "pendingToolName": "\(toolName)",
          "restoredTokenCount": 9000,
          "requiresRebuild": false,
          "generatedTokens": 200,
          "thinkingTokens": 160,
          "blockTokens": 20,
          "repetitions": 8
        }
        """
        let value = try JSONDecoder().decode(ThoughtRepetitionRecovery.self, from: Data(json.utf8))
        return .repeatedThought(value)
    }

    static func openingReceipt() throws -> AppInferenceError {
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
        let value = try JSONDecoder().decode(ThoughtRepetitionRecovery.self, from: Data(json.utf8))
        return .repeatedThought(value)
    }

    final class HostFixture: @unchecked Sendable {
        fileprivate let requests = Mutex<[String]>([])
        fileprivate var describeCount: Int {
            requests.withLock { $0.filter { $0 == "describe screen" }.count }
        }
        private let failedSwipeOrdinal: Int?
        private let deliveryUnknownSwipeOrdinal: Int?
        private let publishesReadyAction: Bool
        private let swipeVerdicts: Mutex<[String]>
        private let swipeCount = Mutex(0)

        var requestCount: Int { requests.withLock { $0.count } }

        init(
            failedSwipeOrdinal: Int? = nil, deliveryUnknownSwipeOrdinal: Int? = nil,
            swipeVerdicts: [String] = [], publishesReadyAction: Bool = false
        ) {
            self.failedSwipeOrdinal = failedSwipeOrdinal
            self.deliveryUnknownSwipeOrdinal = deliveryUnknownSwipeOrdinal
            self.swipeVerdicts = Mutex(swipeVerdicts)
            self.publishesReadyAction = publishesReadyAction
        }

        func reply(_ arguments: JSONValue) async throws -> VisionCaptureMCPResult {
            let request: String
            if case .string(let value)? = arguments.objectValue?["request"] { request = value }
            else { request = "" }
            requests.withLock { $0.append(request) }
            if request == "describe screen" {
                return Self.result(publishesReadyAction: publishesReadyAction)
            }
            if request == "activate computer use" {
                return Self.result(includesComputerUseTaskIdentity: true)
            }
            if request == "click pointer" {
                return Self.result(dispatchAttempted: true, verdict: "verified", action: "click",
                    includesComputerUseTaskIdentity: true)
            }
            if request == "hide pointer" { return Self.result() }
            if request == "swipe up" || request == "swipe down" {
                let ordinal = swipeCount.withLock { value in value += 1; return value }
                let verdict = swipeVerdicts.withLock { values in values.isEmpty ? "verified" : values.removeFirst() }
                if deliveryUnknownSwipeOrdinal == ordinal {
                    return Self.result(isError: true, refusalCode: "CACHE_DISPATCH_UNPROVEN",
                        dispatchAttempted: nil, verdict: "failed", action: "swipe")
                }
                if failedSwipeOrdinal == ordinal {
                    return Self.result(isError: true, refusalCode: "GUARDED_TARGET_REJECTED",
                        dispatchAttempted: false, guarded: true, verdict: "failed", action: "swipe")
                }
                return Self.result(dispatchAttempted: true, verdict: verdict, action: "swipe")
            }
            if request == "tap cached action" {
                return Self.result(dispatchAttempted: true, verdict: "verified", action: "tap")
            }
            return Self.result()
        }

        static func result(
            isError: Bool = false, refusalCode: String? = nil,
            dispatchAttempted: Bool? = nil, guarded: Bool = false,
            verdict: String? = nil, action: String? = nil,
            includesComputerUseTaskIdentity: Bool = false,
            publishesReadyAction: Bool = false
        ) -> VisionCaptureMCPResult {
            var elements: [JSONValue] = []
            if publishesReadyAction {
                elements.append(.object([
                    "element_id": .string("matrix-button"),
                    "identifier": .string("matrix-button"),
                    "label": .string("Continue"),
                    "role": .string("button"),
                    "type": .string("XCUIElementTypeButton"),
                    "visible": .bool(true),
                    "enabled": .bool(true),
                    "selected": .bool(false),
                    "x_norm": .integer(500),
                    "y_norm": .integer(400),
                ]))
            }
            var value: [String: JSONValue] = [
                "session_id": .string("flow-35"), "session_kind": .string("flow"),
                "view": .object(["signature_fine": .string("screen-1"),
                                 "signature_coarse": .string("screen-1")]),
                "elements": .array(elements),
            ]
            if includesComputerUseTaskIdentity {
                value["computer_use_task_id"] = .string("00000000-0000-0000-0000-000000000035")
                value["computer_use_generation"] = .integer(1)
            }
            if publishesReadyAction {
                value["cache"] = .object([
                    "state": .string("ready"),
                    "actions": .array([.object([
                        "action": .string("tap"), "selector": .string("matrix-button"),
                        "role": .string("button"), "action_capability": .string("matrix-capability"),
                    ])]),
                ])
            }
            if let verdict { value["proof"] = .object(["verdict": .string(verdict),
                "reason_code": .string("fixture"), "action": .string(action ?? "observe")]) }
            if dispatchAttempted == true {
                value["dispatch"] = .object(["submission_started": .bool(true),
                    "delivery_acknowledged": .bool(true)])
            }
            return VisionCaptureMCPResult(value: .object(value), isError: isError,
                refusalCode: refusalCode, dispatchAttempted: dispatchAttempted,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: guarded,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }
    }

    private static func expectSameResultTurn(
        _ turns: [AppToolTurn], onIndex: Int, offIndex: Int, callID: String
    ) {
        guard turns.indices.contains(onIndex), turns.indices.contains(offIndex),
              case .results(let onResults) = turns[onIndex],
              case .results(let offResults) = turns[offIndex],
              onResults.count == 1, offResults.count == 1 else {
            Issue.record("The ON and special OFF requests must retain the same single-result turn")
            return
        }
        let onResult = onResults[0]
        let offResult = offResults[0]
        #expect(onResults == offResults)
        #expect(onResult.callID == callID)
        #expect(onResult.name == "visioncapture_navigate")
        #expect(onResult.imageAttachments == offResult.imageAttachments)
        #expect(onResult.content == offResult.content)
        do {
            let packet = try JSONDecoder().decode(JSONValue.self, from: Data(onResult.content.utf8))
            let body = packet.objectValue
            #expect(body?["last_action"]?.objectValue?["dispatch_attempted"] == .bool(false))
            #expect(body?["allowed_next"] != nil)
            if case .string(let guidance)? = body?["guidance"] {
                #expect(guidance.contains("rejected this exact target before submission"))
            } else {
                Issue.record("The retained refusal guidance must remain in the model-facing packet")
            }
        } catch {
            Issue.record("The retained guarded refusal result must remain valid JSON: \(error)")
        }
    }

}

private final class Mutex<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<Result>(_ body: (inout Value) -> Result) -> Result {
        lock.lock(); defer { lock.unlock() }
        return body(&value)
    }
}
