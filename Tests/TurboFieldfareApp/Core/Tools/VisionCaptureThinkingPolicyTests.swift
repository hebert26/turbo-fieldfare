import Foundation
import CoreGraphics
import ImageIO
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

    @Test(arguments: Task43Refusal.allCases)
    func rejectedProposalTurnsThinkingOnWithOneFreshCurrentImage(_ refusal: Task43Refusal) async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43")
        let host = Task43HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        // The normal screenshot path remains in use; this only admits the
        // repository-local vision fixture for the fake-host run.
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let result = try await loop.run(
            configuration: Self.configuration, activity: { _ in }, maxContextTokens: 32_768) { turn, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            turns.withLock { $0.append(turn) }
            switch modes.withLock({ $0.count }) {
            case 1:
                return Self.completion(call: Self.observeCall("43-setup"), thinking: selection.enabled)
            case 2:
                let refusalTarget: String?
                switch refusal {
                case .unsupportedTap, .emptyType:
                    refusalTarget = try Self.choiceID(in: turn, operation: "type")
                    if refusal == .unsupportedTap {
                        try Self.expectChoice(refusalTarget!, in: turn, doesNotSupport: "tap")
                    }
                default:
                    refusalTarget = nil
                }
                return Self.completion(
                    call: refusal.call(id: "43-refused", target: refusalTarget), thinking: selection.enabled)
            case 3:
                #expect(selection == AgentThinkingSelection(
                    enabled: true, reason: "previous_proposal_refused"))
                #expect(host.screenshotCount == 1)
                #expect(host.ocrScreenshotCount == 1)
                let target = try Self.choiceID(in: turn, operation: "tap")
                return Self.completion(call: Self.tapCall("43-valid", target: target), thinking: selection.enabled)
            default:
                return Self.finalCompletion(thinking: selection.enabled)
            }
        }

        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "previous_proposal_refused"),
            AgentThinkingSelection(enabled: false, reason: "one_request_complete"),
        ])
        let capturedTurns = turns.withLock { $0 }
        guard capturedTurns.count >= 4,
              case .results(let refusalResults) = capturedTurns[2], refusalResults.count == 1,
              case .results(let successResults) = capturedTurns[3], successResults.count == 1 else {
            Issue.record("Task 43 must retain one result packet for the refusal and action")
            return
        }
        let refusalResult = refusalResults[0]
        #expect(refusalResult.imageAttachments.count == 1)
        let refusalBody = try Self.object(from: refusalResult.content)
        #expect(refusalBody["last_action"]?.objectValue?["verdict"] == .string("not_sent"))
        #expect(refusalBody["last_action"]?.objectValue?["dispatch_attempted"] != .bool(true))
        #expect(refusalBody["observation"]?.objectValue?["current_image_evidence"] == .bool(true))
        try Self.expectCurrentChoiceOperations(in: refusalBody)
        #expect(successResults[0].imageAttachments.isEmpty,
            "The ON request's valid action must not inherit a second Task 43 image")
        #expect(host.mutationCount == 1)
        #expect(host.screenshotCount == 1)
        #expect(host.ocrScreenshotCount == 2)
    }

    @Test
    func consecutiveRejectedProposalsEachGetASeparateOnImageBeforeOneSentAction() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-chain")
        let host = Task43HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let result = try await loop.run(
            configuration: Self.configuration, activity: { _ in }, maxContextTokens: 32_768) { turn, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            turns.withLock { $0.append(turn) }
            switch modes.withLock({ $0.count }) {
            case 1:
                return Self.completion(call: Self.observeCall("43-chain-setup"), thinking: selection.enabled)
            case 2:
                return Self.completion(
                    call: Task43Refusal.expiredChoice.call(id: "43-chain-first"), thinking: selection.enabled)
            case 3:
                #expect(host.screenshotCount == 1)
                #expect(host.ocrScreenshotCount == 1)
                return Self.completion(
                    call: Task43Refusal.unknownTarget.call(id: "43-chain-second"), thinking: selection.enabled)
            case 4:
                #expect(host.screenshotCount == 2)
                #expect(host.ocrScreenshotCount == 2)
                let target = try Self.choiceID(in: turn, operation: "tap")
                return Self.completion(
                    call: Self.tapCall("43-chain-valid", target: target), thinking: selection.enabled)
            default:
                return Self.finalCompletion(thinking: selection.enabled)
            }
        }

        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "previous_proposal_refused"),
            AgentThinkingSelection(enabled: true, reason: "previous_proposal_refused"),
            AgentThinkingSelection(enabled: false, reason: "one_request_complete"),
        ])
        let capturedTurns = turns.withLock { $0 }
        guard capturedTurns.count >= 5,
              case .results(let firstResults) = capturedTurns[2], firstResults.count == 1,
              case .results(let secondResults) = capturedTurns[3], secondResults.count == 1,
              case .results(let successResults) = capturedTurns[4], successResults.count == 1 else {
            Issue.record("The consecutive-refusal chain must retain both correction result packets")
            return
        }
        #expect(firstResults[0].imageAttachments.count == 1)
        #expect(secondResults[0].imageAttachments.count == 1)
        #expect(firstResults[0].imageAttachments[0].sha256 != secondResults[0].imageAttachments[0].sha256)
        #expect(try Self.object(from: firstResults[0].content)["last_action"]?.objectValue?["verdict"]
            == .string("not_sent"))
        #expect(try Self.object(from: secondResults[0].content)["last_action"]?.objectValue?["verdict"]
            == .string("not_sent"))
        #expect(successResults[0].imageAttachments.isEmpty)
        #expect(host.mutationCount == 1)
        #expect(host.screenshotCount == 2)
        #expect(host.ocrScreenshotCount == 3)
    }

    @Test
    func successfulOffActionDoesNotAddTask43ImageOrThinkingTrigger() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-success")
        let host = Task43HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let result = try await loop.run(
            configuration: Self.configuration, activity: { _ in }, maxContextTokens: 32_768) { turn, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            turns.withLock { $0.append(turn) }
            switch modes.withLock({ $0.count }) {
            case 1:
                return Self.completion(call: Self.observeCall("43-success-setup"), thinking: selection.enabled)
            case 2:
                let target = try Self.choiceID(in: turn, operation: "tap")
                return Self.completion(call: Self.tapCall("43-success", target: target), thinking: selection.enabled)
            default:
                return Self.finalCompletion(thinking: selection.enabled)
            }
        }

        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
        ])
        let capturedTurns = turns.withLock { $0 }
        if capturedTurns.count >= 3, case .results(let finalResults) = capturedTurns[2], finalResults.count == 1 {
            #expect(finalResults[0].imageAttachments.isEmpty)
        } else {
            Issue.record("A successful OFF action must provide a final result without a Task 43 image")
        }
        #expect(host.mutationCount == 1)
    }

    @Test(arguments: [false, true])
    func unavailableOrUnusableVisualSupportContinuesWithTextOnlyRecovery(_ unusableRead: Bool) async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-fail-closed")
        let host = Task43HostFixture(failPairedRead: unusableRead)
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: unusableRead,
            configuration: Self.configuration, commitTarget: true)
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let result = try await loop.run(
                configuration: Self.configuration, activity: { _ in }, maxContextTokens: 32_768) { turn, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                turns.withLock { $0.append(turn) }
                switch modes.withLock({ $0.count }) {
                case 1:
                    return Self.completion(call: Self.observeCall("43-fail-closed-setup"), thinking: selection.enabled)
                case 2:
                    return Self.completion(
                        call: Task43Refusal.expiredChoice.call(id: "43-fail-closed-refusal"),
                        thinking: selection.enabled)
                case 3 where unusableRead:
                    return Self.completion(
                        call: Self.observeCall("43-fail-closed-reobserve"), thinking: selection.enabled)
                case 3:
                    #expect(host.screenshotCount == 0)
                    #expect(host.mutationCount == 0)
                    let target = try Self.choiceID(in: turn, operation: "tap")
                    return Self.completion(
                        call: Self.tapCall("43-fail-closed-valid", target: target), thinking: selection.enabled)
                default:
                    return Self.finalCompletion(thinking: selection.enabled)
                }
            }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "previous_proposal_refused"),
            AgentThinkingSelection(enabled: false, reason: "one_request_complete"),
        ])
        #expect(host.requests.withLock { $0.filter { $0 == "click pointer" }.count } == (unusableRead ? 0 : 1))
        #expect(host.screenshotCount == (unusableRead ? 1 : 0))
        #expect(host.ocrScreenshotCount == 0)
        #expect(host.mutationCount == (unusableRead ? 0 : 1))
        guard turns.withLock({ $0.count }) >= 3,
              case .results(let correctionResults) = turns.withLock({ $0 })[2], correctionResults.count == 1 else {
            Issue.record("Text-only recovery must retain its correction result")
            return
        }
        #expect(correctionResults[0].imageAttachments.isEmpty)
        #expect(try Self.object(from: correctionResults[0].content)["last_action"]?.objectValue?["verdict"]
            == .string("not_sent"))
    }

    @Test
    func insufficientRecoveryCapacityContinuesWithTextOnlyRecovery() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-capacity")
        let host = Task43HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let inferenceCount = Mutex(0)
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let result = try await loop.run(
            configuration: Self.configuration, activity: { _ in }, maxContextTokens: 1) { turn, request in
                let selection = await request.selection()
                modes.withLock { $0.append(selection) }
                turns.withLock { $0.append(turn) }
                let attempt = inferenceCount.withLock { value in value += 1; return value }
                if attempt == 1 {
                    return Self.completion(call: Self.observeCall("43-capacity-setup"), thinking: selection.enabled)
                }
                if attempt == 2 {
                    return Self.completion(
                        call: Task43Refusal.expiredChoice.call(id: "43-capacity-refusal"), thinking: selection.enabled)
                }
                if attempt == 3 {
                    let target = try Self.choiceID(in: turn, operation: "tap")
                    return Self.completion(
                        call: Self.tapCall("43-capacity-valid", target: target), thinking: selection.enabled)
                }
                return Self.finalCompletion(thinking: selection.enabled)
            }
        #expect(result.answer == "done")
        #expect(inferenceCount.withLock { $0 } == 4)
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "previous_proposal_refused"),
            AgentThinkingSelection(enabled: false, reason: "one_request_complete"),
        ])
        #expect(host.screenshotCount == 0)
        #expect(host.mutationCount == 1)
        guard turns.withLock({ $0.count }) >= 3,
              case .results(let correctionResults) = turns.withLock({ $0 })[2], correctionResults.count == 1 else {
            Issue.record("Capacity recovery must retain its correction result")
            return
        }
        #expect(correctionResults[0].imageAttachments.isEmpty)
        #expect(try Self.object(from: correctionResults[0].content)["last_action"]?.objectValue?["verdict"]
            == .string("not_sent"))
    }

    @Test
    func primaryCaptureFailureContinuesWithTextOnlyCorrection() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-capture-failure")
        let host = Task43HostFixture(failPrimaryCapture: true)
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let modes = Mutex<[AgentThinkingSelection]>([])
        let turns = Mutex<[AppToolTurn]>([])
        let result = try await loop.run(
            configuration: Self.configuration, activity: { _ in }, maxContextTokens: 32_768) { turn, request in
            let selection = await request.selection()
            modes.withLock { $0.append(selection) }
            turns.withLock { $0.append(turn) }
            switch modes.withLock({ $0.count }) {
            case 1:
                return Self.completion(call: Self.observeCall("43-capture-failure-setup"), thinking: selection.enabled)
            case 2:
                return Self.completion(
                    call: Task43Refusal.expiredChoice.call(id: "43-capture-failure-refusal"),
                    thinking: selection.enabled)
            case 3:
                return Self.completion(
                    call: Self.observeCall("43-capture-failure-reobserve"), thinking: selection.enabled)
            default:
                return Self.finalCompletion(thinking: selection.enabled)
            }
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "previous_proposal_refused"),
            AgentThinkingSelection(enabled: false, reason: "one_request_complete"),
        ])
        #expect(host.screenshotCount == 1)
        #expect(host.ocrScreenshotCount == 0)
        #expect(host.mutationCount == 0)
        guard turns.withLock({ $0.count }) >= 3,
              case .results(let correctionResults) = turns.withLock({ $0 })[2], correctionResults.count == 1 else {
            Issue.record("Capture failure recovery must retain its correction result")
            return
        }
        #expect(correctionResults[0].imageAttachments.isEmpty)
        #expect(try Self.object(from: correctionResults[0].content)["last_action"]?.objectValue?["verdict"]
            == .string("not_sent"))
    }

    @Test(arguments: Task43HostFixture.Failure.allCases)
    func injectedCaptureSafetyFailuresPropagateWithoutCorrection(_ failure: Task43HostFixture.Failure) async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-injected-\(failure.rawValue)")
        let host = Task43HostFixture(failure: failure)
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let inferenceCount = Mutex(0)
        var terminal: Error?
        do {
            _ = try await loop.run(
                configuration: Self.configuration, activity: { _ in }, maxContextTokens: 32_768) { _, request in
                inferenceCount.withLock { $0 += 1 }
                let selection = await request.selection()
                return inferenceCount.withLock { $0 } == 1
                    ? Self.completion(call: Self.observeCall("43-safety-setup"), thinking: selection.enabled)
                    : Self.completion(
                        call: Task43Refusal.expiredChoice.call(id: "43-safety-refusal"), thinking: selection.enabled)
            }
        } catch {
            terminal = error
        }
        #expect(inferenceCount.withLock { $0 } == 2)
        #expect(host.mutationCount == 0)
        #expect(host.screenshotCount == (failure == .cancellation ? 0 : 1))
        #expect(host.ocrScreenshotCount == 0)
        switch failure {
        case .cancellation:
            #expect(terminal is CancellationError)
        case .returnedIdentity:
            guard case .returnedIdentityMismatch? = terminal as? VisionCaptureAgentError else {
                Issue.record("Returned identity failure must propagate unchanged")
                return
            }
        case .sessionIdentity:
            guard case .sessionIdentityMismatch? = terminal as? VisionCaptureAgentError else {
                Issue.record("Session identity failure must propagate unchanged")
                return
            }
        }
    }

    @Test
    func completedPacketCapacityFallsBackToZeroImageCorrection() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-completed-capacity")
        let host = Task43HostFixture(largePairedRead: true)
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let turns = Mutex<[AppToolTurn]>([])
        let modes = Mutex<[AgentThinkingSelection]>([])
        let result = try await loop.run(
            configuration: Self.configuration, activity: { _ in }, maxContextTokens: 16_384) { turn, request in
            let selection = await request.selection()
            turns.withLock { $0.append(turn) }
            modes.withLock { $0.append(selection) }
            switch modes.withLock({ $0.count }) {
            case 1:
                return Self.completion(
                    call: Self.observeCall("43-completed-capacity-setup"), thinking: selection.enabled)
            case 2:
                return Self.completion(
                    call: Task43Refusal.expiredChoice.call(id: "43-completed-capacity-refusal"),
                    thinking: selection.enabled)
            case 3:
                return Self.completion(
                    call: Self.observeCall("43-completed-capacity-reobserve"), thinking: selection.enabled)
            default:
                return Self.finalCompletion(thinking: selection.enabled)
            }
        }
        #expect(result.answer == "done")
        #expect(modes.withLock { $0 } == [
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: false, reason: "agent_default"),
            AgentThinkingSelection(enabled: true, reason: "previous_proposal_refused"),
            AgentThinkingSelection(enabled: false, reason: "one_request_complete"),
        ])
        #expect(host.screenshotCount == 1)
        #expect(host.ocrScreenshotCount == 1)
        guard turns.withLock({ $0.count }) >= 3,
              case .results(let correctionResults) = turns.withLock({ $0 })[2], correctionResults.count == 1 else {
            Issue.record("Completed-packet capacity fallback must retain its correction result")
            return
        }
        #expect(correctionResults[0].imageAttachments.isEmpty)
        #expect(try Self.object(from: correctionResults[0].content)["last_action"]?.objectValue?["verdict"]
            == .string("not_sent"))
    }

    @Test
    func invalidConfigurationStopsAtFirstToolBeforeHostOrCorrectionInference() async throws {
        let loop = VisionCaptureToolLoop()
        let host = Task43HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        let invalid = VisionCaptureAgentConfiguration(
            bundleIdentifier: "",
            simulatorUDID: "not-a-udid",
            modelDirectory: Self.configuration.modelDirectory)
        let inferenceCount = Mutex(0)
        var terminal: VisionCaptureAgentError?
        do {
            _ = try await loop.run(configuration: invalid, activity: { _ in }) { _, request in
                let selection = await request.selection()
                inferenceCount.withLock { $0 += 1 }
                return Self.completion(call: Self.observeCall("43-invalid-config"), thinking: selection.enabled)
            }
        } catch let error as VisionCaptureAgentError {
            terminal = error
        }
        guard case .invalidConfiguration? = terminal else {
            Issue.record("Invalid configuration must stop at the first ordinary tool proposal")
            return
        }
        #expect(inferenceCount.withLock { $0 } == 1)
        #expect(host.requests.withLock { $0.isEmpty })
    }

    @Test
    func returnedIdentityMismatchStopsBeforeVisualCorrectionInferenceOrAction() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-identity-mismatch")
        let host = Task43HostFixture(returnedUDID: "00000000-0000-0000-0000-000000000099")
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let inferenceCount = Mutex(0)
        var terminal: VisionImageError?
        do {
            _ = try await loop.run(
                configuration: Self.configuration, activity: { _ in }, maxContextTokens: 32_768) { _, request in
                let selection = await request.selection()
                let attempt = inferenceCount.withLock { value in value += 1; return value }
                if attempt == 1 {
                    return Self.completion(call: Self.observeCall("43-identity-setup"), thinking: selection.enabled)
                }
                return Self.completion(
                    call: Task43Refusal.expiredChoice.call(id: "43-identity-refusal"), thinking: selection.enabled)
            }
        } catch let error as VisionImageError {
            terminal = error
        }
        guard case .invalidSource(let detail)? = terminal else {
            Issue.record("Returned device identity must fail with VisionImageError.invalidSource")
            return
        }
        #expect(detail == "Screenshot belongs to a different simulator device.")
        #expect(inferenceCount.withLock { $0 } == 2)
        #expect(host.screenshotCount == 1)
        #expect(host.ocrScreenshotCount == 0)
        #expect(host.mutationCount == 0)
    }

    @Test
    func sixthRefusalStopsBeforeAnotherVisualCorrection() async throws {
        let loop = VisionCaptureToolLoop()
        await loop.setFlowSessionForTesting("flow-43-sixth-refusal")
        let host = Task43HostFixture()
        await loop.setHostRequestForTesting { arguments in try await host.reply(arguments) }
        _ = try await loop.actionPacketForTesting(
            operation: "observe", read: nil, body: [:], visionPackComplete: true,
            configuration: Self.configuration, commitTarget: true)
        let inferenceCount = Mutex(0)
        var terminal: Error?
        do {
            _ = try await loop.run(
                configuration: Self.configuration, activity: { _ in }, maxContextTokens: 65_536) { _, request in
                let selection = await request.selection()
                let attempt = inferenceCount.withLock { value in value += 1; return value }
                if attempt == 1 {
                    return Self.completion(call: Self.observeCall("43-sixth-setup"), thinking: selection.enabled)
                }
                return Self.completion(
                    call: Self.unknownCall("43-sixth-refusal-\(attempt)"), thinking: selection.enabled)
            }
        } catch {
            terminal = error
        }
        #expect(terminal != nil)
        #expect(inferenceCount.withLock { $0 } == 7)
        #expect(host.mutationCount == 0)
        #expect(host.screenshotCount == 5)
        #expect(host.ocrScreenshotCount == 5)
        #expect(host.totalScreenshotCount == 10)
        guard case .proposalCorrectionExhausted? = terminal as? VisionCaptureAgentError else {
            Issue.record("The sixth refusal must pause with proposalCorrectionExhausted before another capture")
            return
        }
    }

    static func diagnostics(stopReason: AppStopReason, thinking: Bool) -> AppDiagnostics {
        AppDiagnostics(generatedTokens: 1, stopReason: stopReason, promptTokenCount: 1,
            conversationTokens: 1,
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

    static func tapCall(_ id: String, target: String) -> AppToolCall {
        AppToolCall(id: id, name: "visioncapture_navigate", arguments: .object([
            "action": .string("tap"), "target": .string(target),
        ]))
    }

    private static func object(from content: String) throws -> [String: JSONValue] {
        guard let object = try JSONDecoder().decode(JSONValue.self, from: Data(content.utf8)).objectValue else {
            throw VisionCaptureAgentError.noProgress("Task 43 fixture returned a non-object packet")
        }
        return object
    }

    private static func expectCurrentChoiceOperations(in body: [String: JSONValue]) throws {
        guard case .array(let choices)? = body["choices"] else {
            Issue.record("The fresh refusal correction must publish current choices")
            return
        }
        let operations = choices.compactMap { choice -> [String: JSONValue]? in
            guard case .object(let object) = choice else { return nil }
            return object
        }
        guard let tap = operations.first(where: { object in
            guard case .array(let values)? = object["operations"] else { return false }
            return values.contains(.string("tap"))
        }), let type = operations.first(where: { object in
            guard case .array(let values)? = object["operations"] else { return false }
            return values.contains(.string("type"))
        }) else {
            Issue.record("The fresh refusal correction must retain tap and type choices")
            return
        }
        guard case .array(let tapOperations)? = tap["operations"],
              case .array(let typeOperations)? = type["operations"] else {
            Issue.record("The fresh refusal correction must publish operation lists")
            return
        }
        #expect(tapOperations.contains(.string("tap")))
        #expect(typeOperations.contains(.string("type")))
    }

    private static func choiceID(in turn: AppToolTurn, operation: String) throws -> String {
        guard case .results(let results) = turn, results.count == 1,
              let packet = try? JSONDecoder().decode(
                  JSONValue.self, from: Data(results[0].content.utf8)),
              case .array(let choices)? = packet.objectValue?["choices"] else {
            throw VisionCaptureAgentError.noProgress("The current result did not publish choices")
        }
        guard let choice = choices.compactMap(\.objectValue).first(where: { object in
            guard case .array(let values)? = object["operations"] else { return false }
            return values.contains(.string(operation))
        }), case .string(let id)? = choice["id"] else {
            throw VisionCaptureAgentError.noProgress("The current result did not publish a \(operation) choice")
        }
        return id
    }

    private static func expectChoice(
        _ id: String, in turn: AppToolTurn, doesNotSupport operation: String
    ) throws {
        guard case .results(let results) = turn, results.count == 1,
              let packet = try? JSONDecoder().decode(
                  JSONValue.self, from: Data(results[0].content.utf8)),
              case .array(let choices)? = packet.objectValue?["choices"],
              let choice = choices.compactMap(\.objectValue).first(where: { $0["id"] == .string(id) }),
              case .array(let operations)? = choice["operations"] else {
            throw VisionCaptureAgentError.noProgress("The current result did not publish choice (id)")
        }
        #expect(!operations.contains(.string(operation)))
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

    enum Task43Refusal: String, CaseIterable, Sendable {
        case expiredChoice
        case unknownTarget
        case unsupportedTap
        case emptyType
        case unknownToolName

        func call(id: String, target: String? = nil) -> AppToolCall {
            switch self {
            case .expiredChoice:
                return VisionCaptureThinkingPolicyTests.tapCall(id, target: "c999")
            case .unknownTarget:
                return VisionCaptureThinkingPolicyTests.tapCall(id, target: "unknown-target")
            case .unsupportedTap:
                return VisionCaptureThinkingPolicyTests.tapCall(id, target: target ?? "unsupported-target")
            case .emptyType:
                return AppToolCall(id: id, name: "visioncapture_navigate", arguments: .object([
                    "action": .string("type"), "target": .string(target ?? "unsupported-target"), "text": .string(""),
                ]))
            case .unknownToolName:
                return AppToolCall(id: id, name: "task_future_unknown", arguments: .object([
                    "action": .string("tap"), "target": .string("unknown-target"),
                ]))
            }
        }

    }

    final class Task43HostFixture: @unchecked Sendable {
        enum Failure: String, CaseIterable, Sendable {
            case cancellation
            case returnedIdentity
            case sessionIdentity
        }

        fileprivate let requests = Mutex<[String]>([])
        private let screenshotOrdinal = Mutex(0)
        private let primaryScreenshotOrdinal = Mutex(0)
        private let ocrScreenshotOrdinal = Mutex(0)
        private let mutationOrdinal = Mutex(0)
        private let failPairedRead: Bool
        private let failPrimaryCapture: Bool
        private let largePairedRead: Bool
        private let returnedUDID: String
        private let failure: Failure?

        var mutationCount: Int { mutationOrdinal.withLock { $0 } }
        var screenshotCount: Int { primaryScreenshotOrdinal.withLock { $0 } }
        var ocrScreenshotCount: Int { ocrScreenshotOrdinal.withLock { $0 } }
        var totalScreenshotCount: Int { screenshotOrdinal.withLock { $0 } }

        init(
            failPairedRead: Bool = false,
            failPrimaryCapture: Bool = false,
            largePairedRead: Bool = false,
            returnedUDID: String = VisionCaptureThinkingPolicyTests.configuration.simulatorUDID,
            failure: Failure? = nil
        ) {
            self.failPairedRead = failPairedRead
            self.failPrimaryCapture = failPrimaryCapture
            self.largePairedRead = largePairedRead
            self.returnedUDID = returnedUDID
            self.failure = failure
        }

        func reply(_ arguments: JSONValue) async throws -> VisionCaptureMCPResult {
            let request: String
            if case .string(let value)? = arguments.objectValue?["request"] { request = value }
            else { request = "" }
            requests.withLock { $0.append(request) }
            switch request {
            case "take a screenshot":
                if let failure {
                    switch failure {
                    case .cancellation: throw CancellationError()
                    case .returnedIdentity, .sessionIdentity: break
                    }
                }
                let includeOCR = arguments.objectValue?["parameters"]?.objectValue?["include_ocr"] == .bool(true)
                let ordinal = screenshotOrdinal.withLock { value in value += 1; return value }
                if !includeOCR { primaryScreenshotOrdinal.withLock { $0 += 1 } }
                if failPrimaryCapture && !includeOCR {
                    return Self.failedReadResult()
                }
                if includeOCR {
                    ocrScreenshotOrdinal.withLock { $0 += 1 }
                }
                return try Self.screenshotResult(ordinal: ordinal, returnedUDID: returnedUDID)
            case "click pointer", "tap cached action":
                mutationOrdinal.withLock { $0 += 1 }
                return Self.actionResult()
            case "activate computer use":
                return Self.activationResult()
            case "hide pointer":
                return Self.alertResult()
            case "describe system alert":
                return Self.alertResult()
            default:
                if let failure, screenshotOrdinal.withLock({ $0 }) > 0, request == "describe screen" {
                    switch failure {
                    case .cancellation: throw CancellationError()
                    case .returnedIdentity: throw VisionCaptureAgentError.returnedIdentityMismatch(
                        fieldPath: "session_id", refusalCode: nil)
                    case .sessionIdentity: throw VisionCaptureAgentError.sessionIdentityMismatch
                    }
                }
                if failPairedRead, screenshotOrdinal.withLock({ $0 }) > 0, request == "describe screen" {
                    return Self.failedReadResult()
                }
                return Self.screenResult(
                    signature: "task-43-screen-\(screenshotOrdinal.withLock { $0 })",
                    largeLabel: largePairedRead && screenshotOrdinal.withLock({ $0 }) > 0
                        && request == "describe screen")
            }
        }

        private static func screenResult(signature: String, largeLabel: Bool = false) -> VisionCaptureMCPResult {
            let label = largeLabel
                ? String(repeating: "Current generic action ", count: 1_000)
                : "Primary action"
            return VisionCaptureMCPResult(
                value: .object([
                    "view": .object([
                        "signature_coarse": .string("task-43-screen-coarse"),
                        "signature_fine": .string(signature),
                    ]),
                    "elements": .array([
                        .object([
                            "element_id": .string("c1"), "role": .string("button"),
                            "type": .string("XCUIElementTypeButton"), "label": .string(label),
                            "visible": .bool(true), "enabled": .bool(true),
                            "x_norm": .integer(240), "y_norm": .integer(420),
                        ]),
                        .object([
                            "element_id": .string("c2"), "role": .string("text_field"),
                            "type": .string("XCUIElementTypeTextField"), "label": .string("Input field"),
                            "visible": .bool(true), "enabled": .bool(true),
                            "x_norm": .integer(720), "y_norm": .integer(420),
                        ]),
                    ]),
                ]),
                isError: false, refusalCode: nil, dispatchAttempted: nil,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: false,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }

        private static func actionResult() -> VisionCaptureMCPResult {
            VisionCaptureMCPResult(
                value: .object([
                    "proof": .object([
                        "verdict": .string("verified"), "reason_code": .string("fixture"),
                        "action": .string("click"),
                    ]),
                    "dispatch": .object([
                        "submission_started": .bool(true), "delivery_acknowledged": .bool(true),
                    ]),
                ]),
                isError: false, refusalCode: nil, dispatchAttempted: true,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: false,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }

        private static func activationResult() -> VisionCaptureMCPResult {
            VisionCaptureMCPResult(
                value: .object([
                    "computer_use_task_id": .string("00000000-0000-0000-0000-000000000043"),
                    "computer_use_generation": .integer(1),
                ]),
                isError: false, refusalCode: nil, dispatchAttempted: nil,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: false,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }

        private static func alertResult() -> VisionCaptureMCPResult {
            VisionCaptureMCPResult(
                value: .object(["system_alert": .object(["present": .bool(false)])]),
                isError: false, refusalCode: nil, dispatchAttempted: nil,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: false,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }

        private static func failedReadResult() -> VisionCaptureMCPResult {
            VisionCaptureMCPResult(
                value: .object(["error": .string("paired accessibility read unavailable")]),
                isError: true, refusalCode: "SCREEN_READ_UNAVAILABLE", dispatchAttempted: false,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: false,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }

        private static func screenshotResult(ordinal: Int, returnedUDID: String) throws -> VisionCaptureMCPResult {
            let data = try png(variant: ordinal)
            let metadata = try JSONValue.object([
                "udid": .string(returnedUDID),
                "width": .integer(1), "height": .integer(1),
                "ocr": .object(["blocks": .array([])]),
            ]).encoded()
            return VisionCaptureMCPResult(
                value: .object(["content": .array([
                    .object([
                        "type": .string("image"), "mimeType": .string("image/png"),
                        "data": .string(data.base64EncodedString()),
                    ]),
                    .object(["type": .string("text"), "text": .string(metadata)]),
                ])]),
                isError: false, refusalCode: nil, dispatchAttempted: nil,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: false,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }

        private static func png(variant: Int) throws -> Data {
            let context = try #require(CGContext(data: nil, width: 2, height: 2,
                bitsPerComponent: 8, bytesPerRow: 8, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            let red = CGFloat((variant * 53).isMultiple(of: 255) ? 0.2 : Double((variant * 53) % 255) / 255)
            context.setFillColor(CGColor(red: red, green: 0.35, blue: 0.6, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
            let image = try #require(context.makeImage())
            let data = NSMutableData()
            let destination = try #require(CGImageDestinationCreateWithData(
                data, "public.png" as CFString, 1, nil))
            CGImageDestinationAddImage(destination, image, nil)
            #expect(CGImageDestinationFinalize(destination))
            return data as Data
        }
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
