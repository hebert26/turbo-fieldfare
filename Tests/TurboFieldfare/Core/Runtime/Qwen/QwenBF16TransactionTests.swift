import Darwin
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized) struct QwenBF16TransactionTests {
    @Test func selectedNineSlotCacheRunsEightExpertTinyTurn() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.expectedResidentBytes,
            expertCacheSlots: 9,
            expertCachePolicy: .lfu)
        let session = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 16,
            expertSlotCount: 9)
        let before = await session.cacheDiagnostics()
        #expect(before.slotCount == 9)
        #expect(before.policy == .lfu)
        #expect(before.integrityPolicy == nil)
        #expect(before.allocatedBytes == 0)

        var oracle = QwenBF16TransactionOracle(source: source)
        let prompt: [Int32] = [1, 2]
        let expected = oracle.turn(prompt: prompt, newTokenCount: 2)
        let result = try await session.generatePreparedTurn(
            promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(result, expected: expected, promptCount: prompt.count)
        let after = await session.cacheDiagnostics()
        let pairBytes = UInt64(model.architecture.hiddenSize)
            * UInt64(model.architecture.routedIntermediateSize) * 6
        #expect(after.allocatedBytes == pairBytes * 9)
        #expect(after.routedExpertCount == 8)
    }

    @Test func sourceConversationCommitsMultipleTurnsAtOraclePositions() async throws {
        let harness = try await makeSourceConversationHarness()
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let firstPrompt: [Int32] = [1, 2]
        let firstExpected = oracle.turn(prompt: firstPrompt, newTokenCount: 2)
        let first = try await harness.session.generatePreparedTurn(
            promptTokenIDs: firstPrompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(first, expected: firstExpected, promptCount: firstPrompt.count)
        var snapshot = try await harness.session.diagnosticSnapshot()
        assertCommitted(snapshot, matches: firstExpected)

        let secondPrompt: [Int32] = [2, 1, 3]
        let secondExpected = oracle.turn(prompt: secondPrompt, newTokenCount: 2)
        let second = try await harness.session.generatePreparedTurn(
            promptTokenIDs: secondPrompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(second, expected: secondExpected, promptCount: secondPrompt.count)
        snapshot = try await harness.session.diagnosticSnapshot()
        assertCommitted(snapshot, matches: secondExpected)
        #expect(secondExpected.lastConsumedPosition > firstExpected.lastConsumedPosition,
                "the second turn must use the retained absolute position lineage")
    }

    @Test func sourceCompositionPreservesLoadedModelContextForPreparedTurn() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }

        let contextB = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: contextB,
            residencyBudgetBytes: source.expectedResidentBytes)
        let contextA = try MetalContext()
        #expect(contextA.queue !== contextB.queue,
                "the proposed factory context must not share the loaded model queue")

        let runtime = ModelFamilyRuntime.qwenOfficialSource(model)
        let selectedContext = ModelFamilyGenerationSession.contextForLoadedRuntime(
            runtime, proposedContext: contextA)
        #expect(selectedContext.device === contextB.device)
        #expect(selectedContext.queue === contextB.queue,
                "source composition must retain the context that owns its resident buffers")
        #expect(selectedContext.queue !== contextA.queue)

        let session = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model,
            context: selectedContext,
            maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: .none)
        var oracle = QwenBF16TransactionOracle(source: source)
        let prompt: [Int32] = [1, 2]
        let expected = oracle.turn(prompt: prompt, newTokenCount: 2)
        let result = try await session.generatePreparedTurn(
            promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(result, expected: expected, promptCount: prompt.count)
        let snapshot = try await session.diagnosticSnapshot()
        assertCommitted(snapshot, matches: expected)
    }

    @Test func protectedReadFailureRestoresExactPrefixAndRetryMatchesCPUOracle() async throws {
        let readFailure = QwenBF16TransactionTestSwitch()
        let hooks = QwenOfficialSourceTransactionHooks(
            beforeProtectedExpertRead: { layer, _, _ in
                if layer == 1 && readFailure.consume() {
                    throw QwenBF16TransactionInjectedFailure.protectedRead
                }
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [1, 2]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        let retryPrompt: [Int32] = [3, 1]
        readFailure.arm()
        let failed = await captureAsync {
            try await harness.session.generatePreparedTurn(
                promptTokenIDs: retryPrompt, config: greedyConfig(maxNewTokens: 2))
        }
        #expect(failed.value == nil, "a protected expert read failure must not return a turn result")
        #expect(failed.error != nil)
        let afterFailure = try await harness.session.diagnosticSnapshot()
        #expect(afterFailure == before,
                "read failure must restore accepted tokens, logits, positions and both KV families")

        let retryExpected = oracle.turn(prompt: retryPrompt, newTokenCount: 2)
        let retried = try await harness.session.generatePreparedTurn(
            promptTokenIDs: retryPrompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(retried, expected: retryExpected, promptCount: retryPrompt.count)
        let afterRetry = try await harness.session.diagnosticSnapshot()
        assertCommitted(afterRetry, matches: retryExpected)
    }

    @Test func failureAfterSecondSourceMoeSubmissionSettlesBeforeRollbackAndAllowsReuse() async throws {
        let submitFailure = QwenBF16TransactionTestSwitch()
        let stages = QwenBF16TransactionTestRecorder()
        let hooks = QwenOfficialSourceTransactionHooks(
            afterActualGPUSubmission: { stage in
                let submissionCount = stages.appendAndCount(
                    "submit:\(stage)", matchingPrefix: "submit:source.moe")
                if stage == "source.moe" {
                    stages.append("source.moe.submission.\(submissionCount)")
                    if submissionCount == 2, submitFailure.consume() {
                        stages.append("source.moe.failure.2")
                        throw QwenBF16TransactionInjectedFailure.gpuSubmission
                    }
                }
            },
            afterActualGPUCompletion: { stage in
                stages.appendAndCount(
                    "complete:\(stage)", matchingPrefix: "complete:source.moe")
            },
            beforeRollbackRestore: {
                stages.append("restore")
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [2, 1]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        stages.removeAll()
        submitFailure.arm()
        let retryPrompt: [Int32] = [1, 3]
        let failed = await captureAsync {
            try await harness.session.generatePreparedTurn(
                promptTokenIDs: retryPrompt, config: greedyConfig(maxNewTokens: 2))
        }
        #expect(failed.value == nil, "post-submit failure must not publish an accepted result")
        #expect(failed.error != nil)
        let trace = stages.snapshot
        let sourceMoeSubmissions = trace.indices.filter { trace[$0] == "submit:source.moe" }
        let sourceMoeCompletions = trace.indices.filter { trace[$0] == "complete:source.moe" }
        guard sourceMoeSubmissions.count == 2,
              sourceMoeCompletions.count == 2,
              let secondSubmissionMarker = trace.firstIndex(of: "source.moe.submission.2"),
              let failureIndex = trace.firstIndex(of: "source.moe.failure.2"),
              let restoreIndex = trace.firstIndex(of: "restore") else {
            Issue.record("expected exactly two source.moe submissions/completions, targeted failure, and rollback; got \(trace)")
            return
        }
        #expect(sourceMoeSubmissions[0] < sourceMoeCompletions[0])
        #expect(sourceMoeCompletions[0] < sourceMoeSubmissions[1])
        #expect(sourceMoeSubmissions[1] < secondSubmissionMarker)
        #expect(secondSubmissionMarker < failureIndex,
                "the injected error must originate at the second source.moe submission hook")
        #expect(failureIndex < sourceMoeCompletions[1],
                "the completion observer must run after the failing submitted command settles")
        #expect(sourceMoeCompletions[1] < restoreIndex,
                "rollback must not restore or release state before the failing source.moe work settles")
        let afterFailure = try await harness.session.diagnosticSnapshot()
        #expect(afterFailure == before,
                "GPU failure must restore the exact accepted prefix and runner snapshot")

        let retryExpected = oracle.turn(prompt: retryPrompt, newTokenCount: 2)
        let retried = try await harness.session.generatePreparedTurn(
            promptTokenIDs: retryPrompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(retried, expected: retryExpected, promptCount: retryPrompt.count)
        let afterRetry = try await harness.session.diagnosticSnapshot()
        assertCommitted(afterRetry, matches: retryExpected)
    }

    @Test func shouldStopCancellationAfterPrefillRestoresTheAcceptedTurnAndRetries() async throws {
        let harness = try await makeSourceConversationHarness()
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [1, 3]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        let stop = QwenBF16TransactionTestSwitch()
        let progress = QwenBF16TransactionTestRecorder()
        let prompt: [Int32] = [2, 1]
        let cancelled = await captureAsync {
            try await harness.session.generatePreparedTurn(
                promptTokenIDs: prompt,
                config: greedyConfig(maxNewTokens: 2),
                shouldStop: { stop.isArmed },
                onEvent: { event in
                    if case let .prefill(done, total) = event {
                        progress.append("prefill:\(done):\(total)")
                        if done == total { stop.arm() }
                    }
                })
        }
        #expect(cancelled.value == nil,
                "host stop after prompt progress must cancel, not commit, the turn")
        #expect(cancelled.error is CancellationError)
        #expect(progress.snapshot.contains("prefill:\(prompt.count):\(prompt.count)"))
        let afterCancellation = try await harness.session.diagnosticSnapshot()
        #expect(afterCancellation == before,
                "shouldStop cancellation must restore the complete previously accepted snapshot")

        let retryExpected = oracle.turn(prompt: prompt, newTokenCount: 2)
        let retried = try await harness.session.generatePreparedTurn(
            promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(retried, expected: retryExpected, promptCount: prompt.count)
        let afterRetry = try await harness.session.diagnosticSnapshot()
        assertCommitted(afterRetry, matches: retryExpected)
    }

    @Test func shouldStopAtPrecommitAfterLastSampleRollsBackTheWholeTurn() async throws {
        let gate = QwenBF16TransactionAsyncGate()
        let armGate = QwenBF16TransactionTestSwitch()
        let stop = QwenBF16TransactionTestSwitch()
        let hooks = QwenOfficialSourceTransactionHooks(
            beforeTurnCommit: {
                if armGate.consume() {
                    stop.arm()
                    await gate.suspend()
                }
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [2, 1]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        let prompt: [Int32] = [3, 2]
        armGate.arm()
        let session = harness.session
        let operation = Task {
            await captureAsync {
                try await session.generatePreparedTurn(
                    promptTokenIDs: prompt,
                    config: greedyConfig(maxNewTokens: 2),
                    shouldStop: { stop.isArmed })
            }
        }
        await gate.waitUntilEntered()
        let suspendedJournal = await harness.session.committedJournalSnapshot()
        assertCommittedJournal(suspendedJournal, matches: before)
        #expect(suspendedJournal.activeTransaction != nil,
                "the turn remains active while suspended before commit")
        #expect(stop.isArmed,
                "the host stop becomes true after the final sample at the precommit barrier")
        await gate.release()
        let cancelled = await operation.value
        #expect(cancelled.value == nil,
                "shouldStop after the last sample must throw instead of committing")
        #expect(cancelled.error is CancellationError)
        let after = try await harness.session.diagnosticSnapshot()
        #expect(after == before,
                "precommit host cancellation must restore the full prior KV and token lineage")

        let retryExpected = oracle.turn(prompt: prompt, newTokenCount: 2)
        let retried = try await harness.session.generatePreparedTurn(
            promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(retried, expected: retryExpected, promptCount: prompt.count)
        let afterRetry = try await harness.session.diagnosticSnapshot()
        assertCommitted(afterRetry, matches: retryExpected)
    }

    @Test func taskCancellationAfterGPUSubmissionWaitsForCompletionAndRestores() async throws {
        let gate = QwenBF16TransactionAsyncGate()
        let armGate = QwenBF16TransactionTestSwitch()
        let stages = QwenBF16TransactionTestRecorder()
        let hooks = QwenOfficialSourceTransactionHooks(
            afterActualGPUSubmission: { stage in
                stages.append("submit:\(stage)")
                if armGate.consume() { await gate.suspend() }
            },
            afterActualGPUCompletion: { stage in
                stages.append("complete:\(stage)")
            },
            beforeRollbackRestore: {
                stages.append("restore")
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [2, 3]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        stages.removeAll()
        armGate.arm()
        let session = harness.session
        let prompt: [Int32] = [3, 1]
        let operation = Task {
            await captureAsync {
                try await session.generatePreparedTurn(
                    promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 2))
            }
        }
        await gate.waitUntilEntered()
        operation.cancel()
        await gate.release()
        let cancelled = await operation.value
        #expect(cancelled.value == nil,
                "Task cancellation after real GPU submission must not return a partial result")
        #expect(cancelled.error is CancellationError)
        let trace = stages.snapshot
        guard let submission = trace.first(where: { $0.hasPrefix("submit:") }),
              let stage = submission.split(separator: ":", maxSplits: 1).last,
              let completionIndex = trace.firstIndex(of: "complete:\(stage)"),
              let restoreIndex = trace.firstIndex(of: "restore") else {
            Issue.record("expected submitted GPU work to complete before rollback; got \(trace)")
            return
        }
        let submissionIndex = trace.firstIndex(of: submission)!
        #expect(submissionIndex < completionIndex)
        #expect(completionIndex < restoreIndex,
                "rollback must wait for the submitted command before restoring or reusing state")
        let afterCancellation = try await harness.session.diagnosticSnapshot()
        #expect(afterCancellation == before)

        let retryExpected = oracle.turn(prompt: prompt, newTokenCount: 2)
        let retried = try await harness.session.generatePreparedTurn(
            promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 2))
        assertResult(retried, expected: retryExpected, promptCount: prompt.count)
        let afterRetry = try await harness.session.diagnosticSnapshot()
        assertCommitted(afterRetry, matches: retryExpected)
    }

    @Test func precommitFailureDiscardsProvisionalTokensAndPrefillEvents() async throws {
        let gate = QwenBF16TransactionAsyncGate()
        let armGate = QwenBF16TransactionTestSwitch()
        let hooks = QwenOfficialSourceTransactionHooks(
            beforeTurnCommit: {
                if armGate.consume() {
                    await gate.suspend()
                    throw QwenBF16TransactionInjectedFailure.precommit
                }
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [1, 2]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        let prompt: [Int32] = [3, 2]
        let events = QwenBF16TransactionTestRecorder()
        armGate.arm()
        let session = harness.session
        let operation = Task {
            await captureAsync {
                try await session.generatePreparedTurn(
                    promptTokenIDs: prompt,
                    config: greedyConfig(maxNewTokens: 2),
                    onEvent: { event in
                        if case let .prefill(done, total) = event {
                            events.append("prefill:\(done):\(total)")
                        }
                    })
            }
        }
        await gate.waitUntilEntered()
        let suspendedJournal = await harness.session.committedJournalSnapshot()
        assertCommittedJournal(suspendedJournal, matches: before)
        #expect(suspendedJournal.activeTransaction != nil,
                "the turn remains provisional while suspended before commit")
        #expect(events.snapshot.contains("prefill:\(prompt.count):\(prompt.count)"),
                "prepared fixture exposes prefill progress but no decoded text stream")
        await gate.release()
        let failed = await operation.value
        #expect(failed.value == nil, "precommit failure must not return accepted tokens")
        #expect(failed.error != nil)
        let afterFailure = try await harness.session.diagnosticSnapshot()
        #expect(afterFailure == before,
                "provisional state and events must not alter the prior committed turn")
    }

    @Test func inPlaceSourceMutationBeforeTurnInvalidatesWithoutChangingAcceptedPrefix() async throws {
        let harness = try await makeSourceConversationHarness()
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)
        let prompt: [Int32] = [1, 3]
        let expected = oracle.turn(prompt: prompt, newTokenCount: 1)
        let result = try await harness.session.generatePreparedTurn(
            promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(result, expected: expected, promptCount: prompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: expected)

        try harness.source.mutateRoutedShardPayloadInPlace()
        let invalidated = await captureAsync {
            try await harness.session.generatePreparedTurn(
                promptTokenIDs: [2, 1], config: greedyConfig(maxNewTokens: 2))
        }
        #expect(invalidated.value == nil,
                "an in-place protected-source mutation must reject the next turn")
        #expect(invalidated.error != nil)
        let after = try await harness.session.diagnosticSnapshot()
        #expect(after == before,
                "source invalidation must not publish or retain any partial turn state")
    }

    @Test func sourceReplacementDuringTurnInvalidatesAtNextConsumedToken() async throws {
        let gate = QwenBF16TransactionAsyncGate()
        let armGate = QwenBF16TransactionTestSwitch()
        let hooks = QwenOfficialSourceTransactionHooks(
            betweenConsumedTokens: { _ in
                if armGate.consume() { await gate.suspend() }
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [2, 1]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        armGate.arm()
        let session = harness.session
        let prompt: [Int32] = [1, 2]
        let operation = Task {
            await captureAsync {
                try await session.generatePreparedTurn(
                    promptTokenIDs: prompt, config: greedyConfig(maxNewTokens: 2))
            }
        }
        await gate.waitUntilEntered()
        try harness.source.replaceRoutedShardWithIdenticalBytes()
        await gate.release()
        let failed = await operation.value
        #expect(failed.value == nil,
                "a same-content file replacement during a turn must invalidate its source lineage")
        #expect(failed.error != nil)
        let after = try await harness.session.diagnosticSnapshot()
        #expect(after == before,
                "replacement must roll back the exact previously accepted runner and token state")
    }

    @Test func reentrantTurnIsRejectedWithoutDisturbingSuspendedTurn() async throws {
        let gate = QwenBF16TransactionAsyncGate()
        let armGate = QwenBF16TransactionTestSwitch()
        let hooks = QwenOfficialSourceTransactionHooks(
            betweenConsumedTokens: { _ in
                if armGate.consume() { await gate.suspend() }
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [3, 1]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        armGate.arm()
        let session = harness.session
        let firstPrompt: [Int32] = [2, 3]
        let firstOperation = Task {
            await captureAsync {
                try await session.generatePreparedTurn(
                    promptTokenIDs: firstPrompt, config: greedyConfig(maxNewTokens: 2))
            }
        }
        await gate.waitUntilEntered()
        let suspended = await harness.session.committedJournalSnapshot()
        assertCommittedJournal(suspended, matches: before)
        #expect(suspended.activeTransaction != nil)
        let secondAttempt = await captureAsync {
            try await harness.session.generatePreparedTurn(
                promptTokenIDs: [4], config: greedyConfig(maxNewTokens: 1))
        }
        #expect(secondAttempt.value == nil,
                "a concurrent turn must be rejected while the source transaction is active")
        #expect(secondAttempt.error != nil)
        let afterRejectedTurn = await harness.session.committedJournalSnapshot()
        assertCommittedJournal(afterRejectedTurn, matches: suspended)
        #expect(afterRejectedTurn.activeTransaction == suspended.activeTransaction,
                "the rejected reentrant turn must leave the admitted transaction active")
        await gate.release()

        let expected = oracle.turn(prompt: firstPrompt, newTokenCount: 2)
        let first = await firstOperation.value
        guard let committed = first.value else {
            Issue.record("the admitted turn should complete after the reentrant request is rejected: \(String(describing: first.error))")
            return
        }
        assertResult(committed, expected: expected, promptCount: firstPrompt.count)
        let after = try await harness.session.diagnosticSnapshot()
        assertCommitted(after, matches: expected)
    }

    @Test func rollbackFailurePoisonsSessionWithoutAcceptingTheFailedTurn() async throws {
        let readFailure = QwenBF16TransactionTestSwitch()
        let rollbackFailure = QwenBF16TransactionTestSwitch()
        let hooks = QwenOfficialSourceTransactionHooks(
            beforeProtectedExpertRead: { layer, _, _ in
                if layer == 1 && readFailure.consume() {
                    throw QwenBF16TransactionInjectedFailure.protectedRead
                }
            },
            beforeRollbackRestore: {
                if rollbackFailure.consume() {
                    throw QwenBF16TransactionInjectedFailure.rollbackRestore
                }
            })
        let harness = try await makeSourceConversationHarness(hooks: hooks)
        defer { harness.source.remove() }
        var oracle = QwenBF16TransactionOracle(source: harness.source)

        let initialPrompt: [Int32] = [1, 2]
        let initialExpected = oracle.turn(prompt: initialPrompt, newTokenCount: 1)
        let initial = try await harness.session.generatePreparedTurn(
            promptTokenIDs: initialPrompt, config: greedyConfig(maxNewTokens: 1))
        assertResult(initial, expected: initialExpected, promptCount: initialPrompt.count)
        let before = try await harness.session.diagnosticSnapshot()
        assertCommitted(before, matches: initialExpected)

        readFailure.arm()
        rollbackFailure.arm()
        let failed = await captureAsync {
            try await harness.session.generatePreparedTurn(
                promptTokenIDs: [3, 1], config: greedyConfig(maxNewTokens: 2))
        }
        #expect(failed.value == nil, "rollback failure cannot produce an accepted turn result")
        #expect(failed.error != nil)
        let poisoned = await harness.session.committedJournalSnapshot()
        assertCommittedJournal(poisoned, matches: before, expectedUnusable: true)
        #expect(poisoned.unusable,
                "a failed restore must close the source conversation against reuse")
        #expect(poisoned.activeTransaction == nil,
                "the failed turn must not remain active after rollback failure")

        let reuse = await captureAsync {
            try await harness.session.generatePreparedTurn(
                promptTokenIDs: [2], config: greedyConfig(maxNewTokens: 1))
        }
        #expect(reuse.value == nil, "a poisoned source conversation must not be reused")
        #expect(reuse.error != nil)
        let stillPoisoned = await harness.session.committedJournalSnapshot()
        assertCommittedJournal(stillPoisoned, matches: before, expectedUnusable: true)
        #expect(stillPoisoned.unusable)
        #expect(stillPoisoned.activeTransaction == nil)
    }
}

private struct QwenBF16SourceConversationHarness {
    let source: QwenBF16TextRunnerFixture.Source
    let context: MetalContext
    let model: QwenOfficialSourceModel
    let session: QwenOfficialSourceConversationGenerationSession
}

private func makeSourceConversationHarness(
    hooks: QwenOfficialSourceTransactionHooks = .none
) async throws -> QwenBF16SourceConversationHarness {
    let source = try QwenBF16TextRunnerFixture.make()
    let context = try MetalContext()
    let model = try QwenOfficialSourceModel.loadSyntheticFixture(
        registrationURL: source.registrationURL,
        context: context,
        residencyBudgetBytes: source.expectedResidentBytes)
    let session = try await QwenOfficialSourceConversationGenerationSession(
        fixtureModel: model,
        context: context,
        maxContext: 16,
        expertSlotCount: QwenBF16TextRunnerFixture.topK,
        hooks: hooks)
    return QwenBF16SourceConversationHarness(
        source: source, context: context, model: model, session: session)
}

private struct QwenBF16ExpectedTurn {
    let generatedTokenIDs: [Int32]
    let retainedTokenIDs: [Int32]
    let consumedTokenIDs: [Int32]
    let pendingAcceptedToken: Int32?
    let currentLogits: [Float]
    let lastConsumedPosition: Int
}

/// Stateful CPU conversation oracle. Positions advance only when an accepted
/// token is consumed as the next model input; the last generated token remains
/// pending exactly as it does at a committed conversation boundary.
private struct QwenBF16TransactionOracle {
    private var reference: QwenBF16TextRunnerFixture.Reference
    private var nextPosition = 0
    private var retained: [Int32] = []
    private var consumed: [Int32] = []
    private var pending: Int32?
    private var logits: [Float]?

    init(source: QwenBF16TextRunnerFixture.Source) {
        reference = source.oracle()
    }

    mutating func turn(prompt: [Int32], newTokenCount: Int) -> QwenBF16ExpectedTurn {
        precondition(!prompt.isEmpty && newTokenCount > 0)
        if let pending {
            _ = consume(pending)
            self.pending = nil
        }
        for token in prompt {
            _ = consume(token)
            retained.append(token)
        }

        var generated: [Int32] = []
        generated.reserveCapacity(newTokenCount)
        for index in 0..<newTokenCount {
            let token = expectedFP16RawGreedyToken(logits!)
            generated.append(token)
            retained.append(token)
            pending = token
            if index + 1 < newTokenCount {
                _ = consume(token)
                pending = nil
            }
        }
        return QwenBF16ExpectedTurn(
            generatedTokenIDs: generated,
            retainedTokenIDs: retained,
            consumedTokenIDs: consumed,
            pendingAcceptedToken: pending,
            currentLogits: logits!,
            lastConsumedPosition: nextPosition - 1)
    }

    @discardableResult
    private mutating func consume(_ token: Int32) -> QwenBF16TextRunnerFixture.TokenResult {
        let result = reference.append(token: token, position: nextPosition)
        nextPosition += 1
        consumed.append(token)
        logits = result.logits
        return result
    }
}

private enum QwenBF16TransactionInjectedFailure: Error {
    case protectedRead
    case gpuSubmission
    case precommit
    case rollbackRestore
}

private func greedyConfig(maxNewTokens: Int) -> GenerationConfig {
    var config = GenerationConfig(
        maxNewTokens: maxNewTokens,
        temperature: 0,
        topK: nil,
        topP: nil,
        repetitionPenalty: 1,
        seed: 0,
        stopStrings: [],
        extraStopTokens: [])
    config.logitTransform = .raw
    return config
}

/// CPU-only reference for the existing FP16 raw-logit sampler boundary. The
/// expected input logits always come from the independent BF16 CPU oracle.
private func expectedFP16RawGreedyToken(_ logits: [Float]) -> Int32 {
    let half = logits.map(Float16.init)
    let maximum = half.map(Float.init).max() ?? 0
    let exponentials = half.map { expf(Float($0) - maximum) }
    let denominator = exponentials.reduce(Float(0), +)
    let probabilities = exponentials.map { Float16($0 / denominator) }
    let highest = probabilities.max() ?? 0
    return Int32(probabilities.firstIndex(of: highest) ?? 0)
}

private func assertResult(
    _ result: QwenConversationGenerationResult,
    expected: QwenBF16ExpectedTurn,
    promptCount: Int
) {
    #expect(result.reason == .maxTokens)
    #expect(result.promptTokens == promptCount)
    #expect(result.newTokens == expected.generatedTokenIDs.count)
    #expect(result.acceptedGeneratedTokenIDs == expected.generatedTokenIDs)
}

private func assertCommitted(
    _ snapshot: QwenOfficialSourceConversationDiagnosticSnapshot,
    matches expected: QwenBF16ExpectedTurn
) {
    #expect(snapshot.retainedTokenIDs == expected.retainedTokenIDs)
    #expect(snapshot.consumedTokenIDs == expected.consumedTokenIDs)
    #expect(snapshot.pendingAcceptedToken == expected.pendingAcceptedToken)
    #expect(snapshot.activeTransaction == nil)
    #expect(!snapshot.unusable)
    guard let actualLogits = snapshot.currentLogits else {
        Issue.record("committed source conversation must retain its current FP32 logits")
        return
    }
    #expect(valuesWithinTolerance(
        actualLogits, expected.currentLogits,
        absolute: QwenBF16TextRunnerFixture.absoluteTolerance,
        relative: QwenBF16TextRunnerFixture.relativeTolerance),
        "current source logits must match the independent CPU reference at position \(expected.lastConsumedPosition)")
}

private func assertCommittedJournal(
    _ actual: QwenOfficialSourceCommittedJournalSnapshot,
    matches expected: QwenOfficialSourceConversationDiagnosticSnapshot,
    expectedUnusable: Bool? = nil
) {
    #expect(actual.retainedTokenIDs == expected.retainedTokenIDs)
    #expect(actual.consumedTokenIDs == expected.consumedTokenIDs)
    #expect(actual.pendingAcceptedToken == expected.pendingAcceptedToken)
    #expect(actual.currentLogits == expected.currentLogits)
    #expect(actual.sourceIdentity == expected.sourceIdentity)
    #expect(actual.unusable == (expectedUnusable ?? expected.unusable))
}

private func assertCommittedJournal(
    _ actual: QwenOfficialSourceCommittedJournalSnapshot,
    matches expected: QwenOfficialSourceCommittedJournalSnapshot,
    expectedUnusable: Bool? = nil
) {
    #expect(actual.retainedTokenIDs == expected.retainedTokenIDs)
    #expect(actual.consumedTokenIDs == expected.consumedTokenIDs)
    #expect(actual.pendingAcceptedToken == expected.pendingAcceptedToken)
    #expect(actual.currentLogits == expected.currentLogits)
    #expect(actual.sourceIdentity == expected.sourceIdentity)
    #expect(actual.unusable == (expectedUnusable ?? expected.unusable))
}

// In-flight checks use the committed-only journal; full runner snapshots are
// reserved for idle, recoverable conversation states.

private func valuesWithinTolerance(
    _ actual: [Float],
    _ expected: [Float],
    absolute: Float,
    relative: Float
) -> Bool {
    guard actual.count == expected.count else { return false }
    return zip(actual, expected).allSatisfy { actual, expected in
        guard actual.isFinite, expected.isFinite else { return false }
        return abs(actual - expected) <= max(absolute, relative * abs(expected))
    }
}

private func captureAsync<Value>(
    _ operation: () async throws -> Value
) async -> (value: Value?, error: Error?) {
    do {
        return (try await operation(), nil)
    } catch {
        return (nil, error)
    }
}
