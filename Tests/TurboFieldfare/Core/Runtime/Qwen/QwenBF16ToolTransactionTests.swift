import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Prepared tool text is synthetic: these tests exercise production parsing and
/// transaction handling, not whether the tiny fixture model generated it.
@Suite(.serialized)
struct QwenBF16ToolTransactionTests {
    @Test func completeCallIsNotPublishedBeforeCommitAndRetainsFixtureIdentity() async throws {
        let events = QwenBF16ToolEventRecorder()
        let precommit = QwenBF16TransactionTestRecorder()
        let inspectBeforeCommit = QwenBF16TransactionTestSwitch()
        let harness = try await makeToolHarness(
            hooks: QwenOfficialSourceTransactionHooks(beforeTurnCommit: {
                if inspectBeforeCommit.consume() {
                    precommit.append("calls:\(events.snapshot().count)")
                }
            }))
        defer { harness.source.remove() }
        let before = try await commitToolBaseline(harness)
        inspectBeforeCommit.arm()
        let steps = preparedToolSteps(toolFrame(query: "snow"))

        let result = try await harness.session.generatePreparedToolTurn(
            promptTokenIDs: [2, 1], steps: steps, tools: [lookupTool()],
            config: toolConfig(maxNewTokens: steps.count),
            onEvent: observeToolEvents(events, session: harness.session))
        let after = try await harness.session.diagnosticSnapshot()
        let committed = await harness.session.committedJournalSnapshot()
        let observed = events.snapshot()

        #expect(precommit.snapshot == ["calls:0"],
                "the synchronous precommit hook must see no published tool calls")
        #expect(result.reason == .toolCalls)
        #expect(result.sourceIdentity == nil,
                "the codec-free fixture must not claim an official identity")
        #expect(result.sourceIdentity == after.sourceIdentity)
        #expect(after.sourceIdentity == before.sourceIdentity)
        #expect(observed.count == 1)
        if let call = observed.first {
            #expect(!call.id.isEmpty)
            #expect(call.name == "lookup")
            #expect(call.arguments == .object(["query": .string("snow")]))
        }
        #expect(committed.activeTransaction == nil)
        #expect(committed.retainedTokenIDs == after.retainedTokenIDs)
        #expect(committed.consumedTokenIDs == after.consumedTokenIDs)
        #expect(committed.currentLogits == after.currentLogits)
        #expect(committed.sourceIdentity == after.sourceIdentity)
    }

    @Test func malformedAndIncompleteCallsEmitNothingAndRestorePriorSnapshot() async throws {
        let harness = try await makeToolHarness()
        defer { harness.source.remove() }
        let before = try await commitToolBaseline(harness)
        let cases = [
            toolFrame(query: "rain"), // violates the literal enum schema
            "<tool_call>\n<function=lookup>\n<parameter=query>\nsnow\n"
                + "</parameter>\n</function>\n</tool_",
        ]

        for text in cases {
            let events = QwenBF16ToolEventRecorder()
            let steps = preparedToolSteps(text)
            let failed = await captureToolOperation {
                try await harness.session.generatePreparedToolTurn(
                    promptTokenIDs: [1, 3], steps: steps, tools: [lookupTool()],
                    config: toolConfig(maxNewTokens: steps.count),
                    onEvent: observeToolEvents(events, session: harness.session))
            }
            #expect(failed.value == nil)
            #expect(failed.error != nil)
            #expect(events.snapshot().isEmpty,
                    "malformed or incomplete output must not publish a tool event")
            let after = try await harness.session.diagnosticSnapshot()
            #expect(after == before,
                    "failure must restore prior tokens, KV, and recurrent state")
        }
    }

    @Test func multipleCompleteCallsArePublishedOnlyAfterTheirSharedCommit() async throws {
        let harness = try await makeToolHarness()
        defer { harness.source.remove() }
        _ = try await commitToolBaseline(harness)
        let events = QwenBF16ToolEventRecorder()
        let steps = preparedToolSteps(toolFrame(query: "snow") + "\n" + toolFrame(query: "snow"))

        let result = try await harness.session.generatePreparedToolTurn(
            promptTokenIDs: [2, 1], steps: steps, tools: [lookupTool()],
            config: toolConfig(maxNewTokens: steps.count),
            onEvent: observeToolEvents(events, session: harness.session))
        let after = try await harness.session.diagnosticSnapshot()
        let committed = await harness.session.committedJournalSnapshot()
        let observed = events.snapshot()

        #expect(result.reason == .toolCalls)
        #expect(observed.count == 2)
        if observed.count == 2 {
            #expect(observed[0].name == "lookup")
            #expect(observed[0].arguments == .object(["query": .string("snow")]))
            #expect(observed[1].name == "lookup")
            #expect(observed[1].arguments == .object(["query": .string("snow")]))
            #expect(observed[0].id != observed[1].id)
        }
        #expect(committed.activeTransaction == nil)
        #expect(committed.retainedTokenIDs == after.retainedTokenIDs)
        #expect(committed.sourceIdentity == nil)
        #expect(result.sourceIdentity == nil)
    }

    @Test func validFirstThenMalformedSecondPublishesNothingAndRestoresExactSnapshot() async throws {
        let harness = try await makeToolHarness()
        defer { harness.source.remove() }
        let before = try await commitToolBaseline(harness)
        let events = QwenBF16ToolEventRecorder()
        let text = toolFrame(query: "snow") + "\n" + toolFrame(query: "rain")
        let steps = preparedToolSteps(text)

        let failed = await captureToolOperation {
            try await harness.session.generatePreparedToolTurn(
                promptTokenIDs: [1, 2], steps: steps, tools: [lookupTool()],
                config: toolConfig(maxNewTokens: steps.count),
                onEvent: observeToolEvents(events, session: harness.session))
        }

        #expect(failed.value == nil)
        #expect(failed.error != nil)
        #expect(events.snapshot().isEmpty,
                "a valid prefix remains provisional when the later call fails validation")
        let after = try await harness.session.diagnosticSnapshot()
        #expect(after == before)
    }

    @Test func maxTokenTerminationWithoutModelEOSEmitsNothingAndRestoresSnapshot() async throws {
        let harness = try await makeToolHarness()
        defer { harness.source.remove() }
        let before = try await commitToolBaseline(harness)
        let events = QwenBF16ToolEventRecorder()
        let steps = preparedToolSteps(toolFrame(query: "snow"), includeModelEOS: false)

        let failed = await captureToolOperation {
            try await harness.session.generatePreparedToolTurn(
                promptTokenIDs: [2, 3], steps: steps, tools: [lookupTool()],
                config: toolConfig(maxNewTokens: steps.count),
                onEvent: observeToolEvents(events, session: harness.session))
        }

        #expect(failed.value == nil)
        #expect(failed.error != nil)
        #expect(events.snapshot().isEmpty)
        let after = try await harness.session.diagnosticSnapshot()
        #expect(after == before)
    }

    @Test func hostStopAfterPrefillDeterministicallyRollsBackWithoutToolEvent() async throws {
        let harness = try await makeToolHarness()
        defer { harness.source.remove() }
        let before = try await commitToolBaseline(harness)
        let events = QwenBF16ToolEventRecorder()
        let stop = QwenBF16TransactionTestSwitch()
        let steps = preparedToolSteps(toolFrame(query: "snow"))
        let observer = observeToolEvents(events, session: harness.session) { event in
            if case let .prefill(done, total) = event, done == total { stop.arm() }
        }

        let failed = await captureToolOperation {
            try await harness.session.generatePreparedToolTurn(
                promptTokenIDs: [1, 3], steps: steps, tools: [lookupTool()],
                config: toolConfig(maxNewTokens: steps.count),
                shouldStop: { stop.isArmed }, onEvent: observer)
        }

        #expect(failed.value == nil)
        #expect(failed.error is CancellationError)
        #expect(events.snapshot().isEmpty)
        let after = try await harness.session.diagnosticSnapshot()
        #expect(after == before)
    }

    @Test func sourceMutationAtCompletedPrefillRollsBackBeforeToolPublication() async throws {
        let harness = try await makeToolHarness()
        defer { harness.source.remove() }
        let before = try await commitToolBaseline(harness)
        let events = QwenBF16ToolEventRecorder()
        let mutation = QwenBF16TransactionTestRecorder()
        let armed = QwenBF16TransactionTestSwitch()
        armed.arm()
        let steps = preparedToolSteps(toolFrame(query: "snow"))
        let observer = observeToolEvents(events, session: harness.session) { event in
            guard case let .prefill(done, total) = event, done == total,
                  armed.consume() else { return }
            do {
                try harness.source.mutateRoutedShardPayloadInPlace()
                mutation.append("mutated")
            } catch {
                mutation.append("mutation-failed")
            }
        }

        let failed = await captureToolOperation {
            try await harness.session.generatePreparedToolTurn(
                promptTokenIDs: [2, 2], steps: steps, tools: [lookupTool()],
                config: toolConfig(maxNewTokens: steps.count),
                onEvent: observer)
        }

        #expect(mutation.snapshot == ["mutated"],
                "source bytes change deterministically after prompt prefill and before commit")
        #expect(failed.value == nil)
        #expect(failed.error != nil)
        #expect(events.snapshot().isEmpty)
        let after = try await harness.session.diagnosticSnapshot()
        #expect(after == before)
    }
}

private struct QwenBF16ToolHarness {
    let source: QwenBF16TextRunnerFixture.Source
    let session: QwenOfficialSourceConversationGenerationSession
}

private func makeToolHarness(
    hooks: QwenOfficialSourceTransactionHooks = .none
) async throws -> QwenBF16ToolHarness {
    let source = try QwenBF16TextRunnerFixture.make()
    do {
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let session = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 512,
            expertSlotCount: QwenBF16TextRunnerFixture.topK, hooks: hooks)
        return QwenBF16ToolHarness(source: source, session: session)
    } catch {
        source.remove()
        throw error
    }
}

private func commitToolBaseline(
    _ harness: QwenBF16ToolHarness
) async throws -> QwenOfficialSourceConversationDiagnosticSnapshot {
    let result = try await harness.session.generatePreparedTurn(
        promptTokenIDs: [1, 2], config: toolConfig(maxNewTokens: 1))
    #expect(result.sourceIdentity == nil)
    let snapshot = try await harness.session.diagnosticSnapshot()
    #expect(snapshot.sourceIdentity == nil)
    return snapshot
}

private func lookupTool() -> ModelChatToolDefinition {
    ModelChatToolDefinition(function: ModelChatFunctionDefinition(
        name: "lookup", description: "inert fixture tool", parameters: .object([
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

/// Literal structured protocol frame, independent of parser-produced output.
private func toolFrame(query: String) -> String {
    "<tool_call>\n<function=lookup>\n<parameter=query>\n\(query)\n"
        + "</parameter>\n</function>\n</tool_call>"
}

private func preparedToolSteps(
    _ text: String,
    includeModelEOS: Bool = true
) -> [QwenOfficialSourcePreparedToolStep] {
    precondition(text.utf8.allSatisfy { $0 < 0x80 })
    let bytes = Array(text.utf8)
    var steps: [QwenOfficialSourcePreparedToolStep] = []
    for start in stride(from: 0, to: bytes.count, by: 32) {
        let end = min(start + 32, bytes.count)
        let chunk = Array(bytes[start..<end])
        steps.append(.token(
            id: Int32(steps.count % 4), decoded: String(decoding: chunk, as: UTF8.self)))
    }
    if includeModelEOS {
        // A prepared terminal signal, not sampled model output.
        steps.append(.modelEOS(id: 4, tokenizerTail: ""))
    }
    return steps
}

private func toolConfig(maxNewTokens: Int) -> GenerationConfig {
    var config = GenerationConfig(
        maxNewTokens: maxNewTokens, temperature: 0, topK: nil, topP: nil,
        repetitionPenalty: 1, seed: 0, stopStrings: [], extraStopTokens: [])
    config.logitTransform = .raw
    return config
}

private struct QwenToolCallObservation {
    let id: String
    let name: String
    let arguments: JSONValue
}

private final class QwenBF16ToolEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var calls: [QwenToolCallObservation] = []

    func record(call: ParsedToolCall) {
        lock.lock()
        calls.append(.init(id: call.id, name: call.name, arguments: call.arguments))
        lock.unlock()
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return calls.count
    }

    func snapshot() -> [QwenToolCallObservation] {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }
}

private func observeToolEvents(
    _ recorder: QwenBF16ToolEventRecorder,
    session: QwenOfficialSourceConversationGenerationSession,
    forwarding: @escaping @Sendable (QwenConversationGenerationEvent) -> Void = { _ in }
) -> @Sendable (QwenConversationGenerationEvent) -> Void {
    { event in
        if case .toolCall(let call) = event {
            recorder.record(call: call)
        } else {
            forwarding(event)
        }
    }
}

private func captureToolOperation<Value>(
    _ operation: () async throws -> Value
) async -> (value: Value?, error: Error?) {
    do { return (try await operation(), nil) }
    catch { return (nil, error) }
}
