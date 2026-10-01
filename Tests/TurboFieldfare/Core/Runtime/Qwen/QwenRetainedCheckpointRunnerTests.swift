import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Runner-level retained-checkpoint coverage stays on the two-layer synthetic
/// fixture. The fixture has a full-attention layer followed by a linear layer,
/// so a protected expert-read failure can occur after the linear state commits.
@Suite(.serialized)
struct QwenRetainedCheckpointRunnerTests {
    @Test
    func directProduceFailureAfterLinearCommitRestoresAndRetriesAgainstCPUOracle() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let failure = RetainedCheckpointOneShot()
        let stages = RetainedCheckpointStageLog()
        let runner = try QwenOfficialSourceRunner(
            model: harness.model,
            maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(
                beforeProtectedExpertRead: { layer, _, _ in
                    if layer == 1, failure.consume() {
                        throw RetainedCheckpointRunnerFailure.protectedRead
                    }
                },
                afterActualGPUCompletion: { stage in
                    stages.append("complete:\(stage)")
                }))
        let before = try await runner.diagnosticSnapshot()
        failure.arm()

        await #expect(throws: RetainedCheckpointRunnerFailure.self) {
            _ = try await runner.produce(token: 1, position: 0)
        }
        #expect(stages.values.contains("complete:linear.commit"),
                "the injected failure must follow the earlier linear state commit")
        #expect(await runner.position == 0)
        #expect(try await runner.diagnosticSnapshot() == before,
                "direct token failure must restore the retained token baseline")

        var oracle = harness.source.oracle()
        let expected = oracle.append(token: 1, position: 0)
        let retried = try await runner.produce(token: 1, position: 0)
        assertRunnerLogits(retried, expected: expected.logits, label: "retained-checkpoint retry")
        #expect(await runner.position == 1)
    }

    @Test
    func directProduceFailureWithTurnCheckpointRestoresCPUBaselineAndKeepsRollbackUsable() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let failure = RetainedCheckpointOneShot()
        let runner = try QwenOfficialSourceRunner(
            model: harness.model,
            maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(
                beforeProtectedExpertRead: { layer, _, _ in
                    if layer == 1, failure.consume() {
                        throw RetainedCheckpointRunnerFailure.protectedRead
                    }
                }))
        let before = try await runner.diagnosticSnapshot()
        try await runner.beginTurn()
        failure.arm()

        await #expect(throws: RetainedCheckpointRunnerFailure.self) {
            _ = try await runner.produce(token: 1, position: 0)
        }
        #expect(await runner.position == 0)
        #expect(try await runner.diagnosticSnapshot() == before,
                "a turn baseline must restore the exact CPU snapshot after token failure")

        // The retained token checkpoint has been detached by CPU restore, while
        // the turn checkpoint remains usable for its explicit rollback contract.
        try await runner.rollbackTurn()
        #expect(await runner.position == 0)
        #expect(try await runner.diagnosticSnapshot() == before)

        var oracle = harness.source.oracle()
        let expected = oracle.append(token: 1, position: 0)
        let retried = try await runner.produce(token: 1, position: 0)
        assertRunnerLogits(retried, expected: expected.logits, label: "CPU-baseline retry")
        #expect(await runner.position == 1)
    }

    @Test
    func taskCancellationDuringLinearSubmissionRestoresBothCheckpointKindsAndRetries() async throws {
        for hasTurnCheckpoint in [false, true] {
            let harness = try makeRunnerHarness()
            defer { harness.source.remove() }
            let gate = RetainedCheckpointAsyncGate()
            let pause = RetainedCheckpointOneShot()
            let runner = try QwenOfficialSourceRunner(
                model: harness.model,
                maxContext: 16,
                expertSlotCount: QwenBF16TextRunnerFixture.topK,
                hooks: QwenOfficialSourceTransactionHooks(
                    afterActualGPUSubmission: { stage in
                        if stage == "linear.convolution", pause.consume() {
                            await gate.holdUntilReleased()
                        }
                    }))
            let before = try await runner.diagnosticSnapshot()
            if hasTurnCheckpoint {
                try await runner.beginTurn()
            }
            pause.arm()
            let operation = Task { () throws -> [Float] in
                try await runner.produce(token: 1, position: 0)
            }
            await gate.waitUntilEntered()
            operation.cancel()
            await gate.release()
            await #expect(throws: CancellationError.self) {
                _ = try await operation.value
            }

            #expect(await runner.position == 0)
            #expect(try await runner.diagnosticSnapshot() == before,
                    "linear cancellation must restore the token or turn baseline")
            if hasTurnCheckpoint {
                try await runner.rollbackTurn()
                #expect(await runner.position == 0)
                #expect(try await runner.diagnosticSnapshot() == before,
                        "turn rollback must remain usable after token cancellation")
            }
            var oracle = harness.source.oracle()
            let expected = oracle.append(token: 1, position: 0)
            let retried = try await runner.produce(token: 1, position: 0)
            assertRunnerLogits(
                retried, expected: expected.logits,
                label: hasTurnCheckpoint
                    ? "turn-checkpoint cancellation retry"
                    : "retained-checkpoint cancellation retry")
            #expect(await runner.position == 1)
        }
    }

    @Test
    func repeatedTokenSequenceMatchesIndependentCPUOracleAcrossLinearStateUpdates() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let runner = try QwenOfficialSourceRunner(
            model: harness.model,
            maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let tokens: [Int32] = [1, 1, 1]
        var oracle = harness.source.oracle()
        for (position, token) in tokens.enumerated() {
            let expected = oracle.append(token: token, position: position)
            let actual = try await runner.produce(token: token, position: position)
            assertRunnerLogits(
                actual, expected: expected.logits,
                label: "repeated-token logits at position \(position)")
            #expect(await runner.position == position + 1)
        }
    }
}

private struct RetainedRunnerHarness {
    let source: QwenBF16TextRunnerFixture.Source
    let model: QwenOfficialSourceModel
}

private func makeRunnerHarness() throws -> RetainedRunnerHarness {
    let source = try QwenBF16TextRunnerFixture.make()
    let context = try MetalContext()
    do {
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        return RetainedRunnerHarness(source: source, model: model)
    } catch {
        source.remove()
        throw error
    }
}

private enum RetainedCheckpointRunnerFailure: Error {
    case protectedRead
}

private final class RetainedCheckpointOneShot: @unchecked Sendable {
    private let lock = NSLock()
    private var armed = false

    func arm() {
        lock.lock()
        armed = true
        lock.unlock()
    }

    func consume() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard armed else { return false }
        armed = false
        return true
    }
}

private final class RetainedCheckpointStageLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [String] = []

    func append(_ entry: String) {
        lock.lock()
        entries.append(entry)
        lock.unlock()
    }

    var values: [String] {
        lock.lock()
        defer { lock.unlock() }
        return entries
    }
}

private actor RetainedCheckpointAsyncGate {
    private var entered = false
    private var released = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func holdUntilReleased() async {
        entered = true
        let enteredWaiters = self.enteredWaiters
        self.enteredWaiters.removeAll(keepingCapacity: false)
        for waiter in enteredWaiters { waiter.resume() }
        guard !released else { return }
        await withCheckedContinuation { continuation in
            releaseWaiters.append(continuation)
        }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        let releaseWaiters = self.releaseWaiters
        self.releaseWaiters.removeAll(keepingCapacity: false)
        for waiter in releaseWaiters { waiter.resume() }
    }
}

private func assertRunnerLogits(
    _ actual: [Float],
    expected: [Float],
    label: String
) {
    let matches = actual.count == expected.count
        && zip(actual, expected).allSatisfy { actual, expected in
            actual.isFinite && expected.isFinite
                && abs(actual - expected) <= max(
                    QwenBF16TextRunnerFixture.absoluteTolerance,
                    QwenBF16TextRunnerFixture.relativeTolerance * abs(expected))
        }
    #expect(matches, "\(label) must match the independent CPU oracle")
}
