import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenConversationStateTests {
    private let prompt: [Int32] = [1, 4, 7]
    private let firstGenerated: [Int32] = [2, 14, 6]

    @Test func fixtureStateStartsWithAnEmptyCommittedAggregate() async throws {
        let state = try await makeState(maxContext: 32)
        let status = await state.status()

        #expect(status.activeTransaction == nil)
        #expect(status.working == nil)
        #expect(status.committed.retainedTokenIDs.isEmpty)
        #expect(status.committed.consumedTokenCount == 0)
        #expect(status.committed.pendingTokenCount == 0)
        #expect(status.committed.logicalStateBytes > 0)
    }

    @Test func commitTracksConsumedAndPendingTokensSeparately() async throws {
        let state = try await makeState(maxContext: 32)
        let transaction = try await state.begin()
        try await state.prefill(prompt, transaction: transaction) { _, _ in }
        try await state.advance(firstGenerated[0], transaction: transaction)
        try await state.advance(firstGenerated[1], transaction: transaction)

        let working = await state.status()
        let aggregate = try #require(working.working)
        #expect(aggregate.retainedTokenIDs == prompt + Array(firstGenerated.prefix(2)))
        #expect(aggregate.consumedTokenCount == prompt.count + 1)
        #expect(aggregate.pendingTokenCount == 1)
        #expect(working.activeTransaction == transaction)

        let committed = try await state.commit(transaction: transaction)
        #expect(committed == aggregate)
        #expect((await state.status()).working == nil)
    }

    @Test func nextTurnConsumesCommittedPendingExactlyOnceBeforeNewPrompt() async throws {
        let state = try await makeState(maxContext: 32)
        let first = try await state.begin()
        try await state.prefill(prompt, transaction: first) { _, _ in }
        try await state.advance(firstGenerated[0], transaction: first)
        try await state.advance(firstGenerated[1], transaction: first)
        _ = try await state.commit(transaction: first)

        let second = try await state.begin()
        let afterPending = try #require((await state.status()).working)
        #expect(afterPending.retainedTokenIDs == prompt + Array(firstGenerated.prefix(2)))
        #expect(afterPending.consumedTokenCount == prompt.count + 2)
        #expect(afterPending.pendingTokenCount == 0)

        try await state.prefill([3], transaction: second) { _, _ in }
        try await state.advance(firstGenerated[2], transaction: second)
        let result = try await state.commit(transaction: second)
        #expect(result.retainedTokenIDs == prompt + [2, 14, 3, 6])
        #expect(result.consumedTokenCount == 6)
        #expect(result.pendingTokenCount == 1)
    }

    @Test func facadeSoftStopCommitsAcceptedPendingTokenAndNextTurnConsumesItOnce() async throws {
        let state = try await makeState(maxContext: 32)
        let conversation = MultimodalConversation(qwenState: state, maxContext: 32)
        let stopRequested = Mutex(false)
        let firstTurn = TokenizedConversationTurn(
            promptTokenIDs: prompt,
            generatedTokenIDs: Array(firstGenerated.prefix(2)))
        let softStop = try await conversation.applyTokenizedTurn(
            firstTurn,
            shouldStop: { stopRequested.withLock { $0 } },
            onProgress: { progress in
                if case .accepted(index: 0, tokenID: 2) = progress {
                    stopRequested.withLock { $0 = true }
                }
            })
        #expect(softStop.reason == .softStop)
        #expect(softStop.acceptedGeneratedTokenIDs == [2])
        #expect(softStop.metrics.retainedTokenIDs == prompt + [2])
        #expect(softStop.metrics.consumedTokenCount == prompt.count)
        #expect(softStop.metrics.pendingTokenCount == 1)

        let secondTurn = try await conversation.applyTokenizedTurn(
            TokenizedConversationTurn(promptTokenIDs: [3], generatedTokenIDs: [14]))
        #expect(secondTurn.reason == .complete)
        #expect(secondTurn.metrics.retainedTokenIDs == prompt + [2, 3, 14])
        #expect(secondTurn.metrics.consumedTokenCount == prompt.count + 2)
        #expect(secondTurn.metrics.pendingTokenCount == 1)
    }

    @Test func hiddenSuffixRemovesPendingThenConsumedTokenByReplay() async throws {
        let state = try await makeState(maxContext: 32)
        let transaction = try await state.begin()
        try await state.prefill(prompt, transaction: transaction) { _, _ in }
        for token in firstGenerated { try await state.advance(token, transaction: transaction) }

        try await state.removeSuffix(tokenCount: 2, transaction: transaction)
        let working = try #require((await state.status()).working)
        #expect(working.retainedTokenIDs == prompt + [firstGenerated[0]])
        #expect(working.consumedTokenCount == prompt.count)
        #expect(working.pendingTokenCount == 1)

        let result = try await state.commit(transaction: transaction)
        #expect(result.retainedTokenIDs == prompt + [firstGenerated[0]])
        #expect(result.consumedTokenCount == prompt.count)
        #expect(result.pendingTokenCount == 1)
    }

    @Test func invalidHiddenSuffixLeavesWorkingTransactionUntouched() async throws {
        let state = try await makeState(maxContext: 32)
        let transaction = try await state.begin()
        try await state.prefill(prompt, transaction: transaction) { _, _ in }
        try await state.advance(firstGenerated[0], transaction: transaction)
        let before = try #require((await state.status()).working)

        await #expect(throws: ConversationStateTransactionError.self) {
            try await state.removeSuffix(tokenCount: 2, transaction: transaction)
        }
        let after = try #require((await state.status()).working)
        #expect(after == before)
        try await state.rollback(transaction: transaction)
    }

    @Test func rollbackRestoresEveryAggregateAndAllowsCleanReuse() async throws {
        let state = try await makeState(maxContext: 32)
        let initialSnapshot = try await state.diagnosticSnapshot()
        let transaction = try await state.begin()
        try await state.prefill(prompt, transaction: transaction) { _, _ in }
        try await state.advance(firstGenerated[0], transaction: transaction)
        try await state.advance(firstGenerated[1], transaction: transaction)
        try await state.rollback(transaction: transaction)

        let afterRollback = await state.status()
        #expect(afterRollback.committed.retainedTokenIDs.isEmpty)
        #expect(afterRollback.working == nil)
        let restoredSnapshot = try await state.diagnosticSnapshot()
        assertDiagnosticStateClose(
            restoredSnapshot, initialSnapshot, label: "empty rollback")

        let clean = try await state.begin()
        try await state.prefill(prompt, transaction: clean) { _, _ in }
        try await state.advance(firstGenerated[0], transaction: clean)
        let expected = try await state.commit(transaction: clean)
        #expect(expected.retainedTokenIDs == prompt + [firstGenerated[0]])
        #expect(expected.pendingTokenCount == 1)
        let reusedSnapshot = try await state.diagnosticSnapshot()
        #expect(reusedSnapshot.currentLogits != nil)
    }

    @Test func capacityCheckpointMaterializesPendingTokenAndNextTurnDoesNotConsumeItTwice() async throws {
        let state = try await makeState(maxContext: 32)
        let initial = try await state.begin()
        try await state.prefill(prompt, transaction: initial) { _, _ in }
        try await state.advance(firstGenerated[0], transaction: initial)
        let beforeCheckpoint = try await state.commit(transaction: initial)
        #expect(beforeCheckpoint.pendingTokenCount == 1)

        // Beginning the checkpoint operation consumes the old pending token once.
        let transaction = try await state.begin()
        let retained = prompt + [firstGenerated[0]]
        try await state.rebuildCheckpoint(
            retaining: retained, transaction: transaction) { _, _ in }
        let rebuilt = try await state.commit(transaction: transaction)

        let cleanState = try await makeState(maxContext: 32)
        let clean = try await cleanState.begin()
        try await cleanState.prefill(retained, transaction: clean) { _, _ in }
        let expected = try await cleanState.commit(transaction: clean)
        #expect(rebuilt == expected)
        #expect(rebuilt.retainedTokenIDs == retained)
        #expect(rebuilt.consumedTokenCount == retained.count)
        #expect(rebuilt.pendingTokenCount == 0)
        let rebuiltSnapshot = try await state.diagnosticSnapshot()
        let expectedSnapshot = try await cleanState.diagnosticSnapshot()
        assertDiagnosticStateClose(
            rebuiltSnapshot, expectedSnapshot,
            label: "capacity checkpoint", compareReplayMetadata: false)
        #expect(rebuiltSnapshot.replayGeneration == expectedSnapshot.replayGeneration + 1)

        // A later turn starts from the fully consumed checkpoint; it must not
        // consume the old pending token a second time.
        let next = try await state.begin()
        let atNextStart = try #require((await state.status()).working)
        #expect(atNextStart.retainedTokenIDs == retained)
        #expect(atNextStart.consumedTokenCount == retained.count)
        #expect(atNextStart.pendingTokenCount == 0)
        try await state.prefill([3], transaction: next) { _, _ in }
        try await state.advance(firstGenerated[1], transaction: next)
        let actualNext = try await state.commit(transaction: next)

        let cleanNextState = try await makeState(maxContext: 32)
        let cleanNext = try await cleanNextState.begin()
        try await cleanNextState.prefill(retained, transaction: cleanNext) { _, _ in }
        try await cleanNextState.prefill([3], transaction: cleanNext) { _, _ in }
        try await cleanNextState.advance(firstGenerated[1], transaction: cleanNext)
        let expectedNext = try await cleanNextState.commit(transaction: cleanNext)
        #expect(actualNext == expectedNext)
        let actualNextSnapshot = try await state.diagnosticSnapshot()
        let expectedNextSnapshot = try await cleanNextState.diagnosticSnapshot()
        assertDiagnosticStateClose(
            actualNextSnapshot, expectedNextSnapshot,
            label: "post-checkpoint continuation", compareReplayMetadata: false)
        #expect(actualNextSnapshot.replayGeneration == rebuiltSnapshot.replayGeneration)
    }

    @Test func checkpointInputFailurePreservesTheCommittedLineageForNonzeroContinuation() async throws {
        let state = try await makeState(maxContext: 32)
        let initial = try await state.begin()
        try await state.prefill(prompt, transaction: initial) { _, _ in }
        let prefix = try await state.commit(transaction: initial)

        let transaction = try await state.begin()
        await #expect(throws: QwenTextRunnerError.invalidToken(id: 19)) {
            try await state.rebuildCheckpoint(
                retaining: prefix.retainedTokenIDs + [19], transaction: transaction) { _, _ in }
        }
        let stillUsable = await state.status()
        #expect(stillUsable.activeTransaction == transaction)
        try await state.rollback(transaction: transaction)

        let continuation = try await state.begin()
        try await state.advance(firstGenerated[0], transaction: continuation)
        let result = try await state.commit(transaction: continuation)
        #expect(result.retainedTokenIDs == prompt + [firstGenerated[0]])
        #expect(result.consumedTokenCount == prompt.count)
        #expect(result.pendingTokenCount == 1)
    }

    @Test func failedCheckpointReplayRestoresProducerEpochAndContinuesFromNonzeroPositionMatchingCleanReference() async throws {
        let gate = ReplayFailureGate()
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model,
            context: context,
            maxContext: 32,
            executionHooks: QwenTextExecutionHooks(
                beforeLayer: { layer in try await gate.check(layer) }),
            publicationHooks: .none)

        let initial = try await state.begin()
        try await state.prefill(prompt, transaction: initial) { _, _ in }
        try await state.advance(firstGenerated[0], transaction: initial)
        let prefix = try await state.commit(transaction: initial)
        let prefixSnapshot = try await state.diagnosticSnapshot()
        #expect(prefix.pendingTokenCount == 1)

        // The old pending token is consumed at begin, then replay fails only
        // after its first replacement token has actually mutated the runner.
        let failed = try await state.begin()
        await gate.arm()
        await #expect(throws: InjectedReplayFailure.self) {
            try await state.rebuildCheckpoint(
                retaining: prefix.retainedTokenIDs + [firstGenerated[1]],
                transaction: failed) { _, _ in }
        }
        #expect(await gate.completedPriorToken)
        #expect((await state.status()).activeTransaction == failed)
        try await state.rollback(transaction: failed)
        let restored = await state.status()
        #expect(restored.committed == prefix)
        #expect(restored.committed.pendingTokenCount == 1)
        let restoredSnapshot = try await state.diagnosticSnapshot()
        assertDiagnosticStateClose(
            restoredSnapshot, prefixSnapshot, label: "failed replay rollback")
        #expect(restoredSnapshot.producerEpoch == prefixSnapshot.producerEpoch)
        #expect(restoredSnapshot.currentLogits != nil)

        let continuation = try await state.begin()
        let afterPending = try #require((await state.status()).working)
        #expect(afterPending.pendingTokenCount == 0)
        try await state.advance(firstGenerated[1], transaction: continuation)
        let actual = try await state.commit(transaction: continuation)

        let cleanState = try await makeState(maxContext: 32)
        let clean = try await cleanState.begin()
        try await cleanState.prefill(prompt, transaction: clean) { _, _ in }
        try await cleanState.advance(firstGenerated[0], transaction: clean)
        try await cleanState.advance(firstGenerated[1], transaction: clean)
        let expected = try await cleanState.commit(transaction: clean)

        #expect(actual == expected)
        #expect(actual.retainedTokenIDs == prompt + [firstGenerated[0], firstGenerated[1]])
        #expect(actual.consumedTokenCount == prompt.count + 1)
        #expect(actual.pendingTokenCount == 1)
        let actualSnapshot = try await state.diagnosticSnapshot()
        let expectedSnapshot = try await cleanState.diagnosticSnapshot()
        assertDiagnosticStateClose(
            actualSnapshot, expectedSnapshot, label: "failed replay continuation")
        #expect(actualSnapshot.producerEpoch == prefixSnapshot.producerEpoch)
    }

    @Test func cancellationWaitsForSubmittedGPUOwnerBeforeRollbackAndReuse() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let event = try #require(context.device.makeSharedEvent())
        let eventBox = SharedEventBox(event)
        let blocker = try #require(context.queue.makeCommandBuffer())
        blocker.encodeWaitForEvent(event, value: 1)
        blocker.commit()
        let submitted = AsyncLatch()
        let release = AsyncLatch()
        let outcome = CancellationOutcome()
        let state = try await QwenConversationState(
            model: model,
            context: context,
            maxContext: 32,
            executionHooks: QwenTextExecutionHooks(
                afterCommandSubmission: { stage in
                    guard stage == "linearConvolution" else { return }
                    await submitted.signal()
                    await release.wait()
                    eventBox.event.signaledValue = 1
                }),
            publicationHooks: .none)
        let transaction = try await state.begin()
        let operation = Task {
            do {
                try await state.prefill([1], transaction: transaction) { _, _ in }
                await outcome.record(finished: true)
            } catch {
                await outcome.record(finished: true)
            }
        }

        await submitted.wait()
        operation.cancel()
        #expect(await outcome.finished == false)
        #expect((await state.status()).activeTransaction == transaction)

        await release.signal()
        _ = await operation.value
        #expect(await outcome.finished)
        try await state.rollback(transaction: transaction)

        let retry = try await state.begin()
        try await state.prefill([1], transaction: retry) { _, _ in }
        let result = try await state.commit(transaction: retry)
        #expect(result.retainedTokenIDs == [1])
        #expect(result.consumedTokenCount == 1)
        #expect(result.pendingTokenCount == 0)
    }

    @Test func staleHandlesCannotMutateAfterCommitOrReset() async throws {
        let state = try await makeState(maxContext: 32)
        let transaction = try await state.begin()
        try await state.prefill([1], transaction: transaction) { _, _ in }
        _ = try await state.commit(transaction: transaction)

        await #expect(throws: ConversationStateTransactionError.staleTransaction) {
            try await state.advance(2, transaction: transaction)
        }
        try await state.reset()
        let resetSnapshot = try await state.diagnosticSnapshot()
        #expect(resetSnapshot.runnerState.sequenceLength == 0)
        #expect(resetSnapshot.currentLogits == nil)
        await #expect(throws: ConversationStateTransactionError.staleTransaction) {
            try await state.rollback(transaction: transaction)
        }
        #expect((await state.status()).committed.retainedTokenIDs.isEmpty)
    }

    private func makeState(maxContext: Int) async throws -> QwenConversationState {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        return try await QwenConversationState(
            model: model, context: context, maxContext: maxContext)
    }
}

func assertDiagnosticStateClose(
    _ actual: QwenConversationStateDiagnosticSnapshot,
    _ expected: QwenConversationStateDiagnosticSnapshot,
    label: String,
    compareReplayMetadata: Bool = true
) {
    #expect(actual.retainedTokenIDs == expected.retainedTokenIDs, "\(label): retained journal")
    #expect(actual.consumedTokenIDs == expected.consumedTokenIDs, "\(label): consumed journal")
    #expect(actual.pendingAcceptedToken == expected.pendingAcceptedToken, "\(label): pending token")
    #expect(actual.textRoPEDelta == expected.textRoPEDelta, "\(label): RoPE metadata")
    if compareReplayMetadata {
        #expect(actual.replayGeneration == expected.replayGeneration, "\(label): replay generation")
    }
    #expect(actual.logicalStateBytes == expected.logicalStateBytes, "\(label): logical bytes")
    #expect(actual.producerEpoch == expected.producerEpoch, "\(label): producer epoch")
    #expect(actual.runnerState.architectureIdentity == expected.runnerState.architectureIdentity,
            "\(label): architecture identity")
    #expect(actual.runnerState.sequenceLength == expected.runnerState.sequenceLength,
            "\(label): sequence length")
    #expect(actual.runnerState.layers.count == expected.runnerState.layers.count,
            "\(label): layer count")
    guard actual.runnerState.layers.count == expected.runnerState.layers.count else { return }

    for (index, pair) in zip(actual.runnerState.layers, expected.runnerState.layers).enumerated() {
        switch pair {
        case let (.linear(actualHistory, actualRecurrent), .linear(expectedHistory, expectedRecurrent)):
            assertFloatArraysClose(actualHistory, expectedHistory,
                                   label: "\(label) layer \(index) convolution")
            assertFloatArraysClose(actualRecurrent, expectedRecurrent,
                                   label: "\(label) layer \(index) recurrent")
        case let (.full(actualKey, actualValue), .full(expectedKey, expectedValue)):
            assertFloatArraysClose(actualKey, expectedKey,
                                   label: "\(label) layer \(index) key")
            assertFloatArraysClose(actualValue, expectedValue,
                                   label: "\(label) layer \(index) value")
        default:
            Issue.record("\(label) layer \(index) changed state kind")
        }
    }

    switch (actual.currentLogits, expected.currentLogits) {
    case let (.some(actualLogits), .some(expectedLogits)):
        // These are the producer's existing FP16 scratch values, not FP32
        // model outputs. Require exact quantized-value recovery separately.
        #expect(actualLogits == expectedLogits, "\(label): FP16 logits scratch")
    case (.none, .none):
        break
    default:
        Issue.record("\(label): logits validity changed")
    }
}

func assertFloatArraysClose(_ actual: [Float], _ expected: [Float], label: String) {
    #expect(actual.count == expected.count, "\(label): element count")
    guard actual.count == expected.count else { return }
    let scale = max(actual.map { abs($0) }.max() ?? 0,
                    expected.map { abs($0) }.max() ?? 0)
    let limit = QwenTextFixtureSupport.stateAbsoluteTolerance
        + QwenTextFixtureSupport.stateRelativeTolerance * scale
    let maximum = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    #expect(maximum <= limit, "\(label): max FP32 delta \(maximum) > \(limit)")
}

private struct InjectedReplayFailure: Error, Equatable {}

private final class SharedEventBox: @unchecked Sendable {
    let event: MTLSharedEvent

    init(_ event: MTLSharedEvent) {
        self.event = event
    }
}

private actor AsyncLatch {
    private var signaled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        guard !signaled else { return }
        signaled = true
        let pending = waiters
        waiters.removeAll(keepingCapacity: false)
        for waiter in pending { waiter.resume() }
    }

    func wait() async {
        if signaled { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }
}

private actor CancellationOutcome {
    private(set) var finished = false

    func record(finished: Bool) {
        self.finished = finished
    }
}

private actor ReplayFailureGate {
    private var armed = false
    private var layerZeroCalls = 0
    private(set) var completedPriorToken = false

    func arm() {
        armed = true
    }

    func check(_ layer: Int) throws {
        guard armed, layer == 0 else { return }
        layerZeroCalls += 1
        if layerZeroCalls == 2 {
            completedPriorToken = true
            throw InjectedReplayFailure()
        }
    }
}
