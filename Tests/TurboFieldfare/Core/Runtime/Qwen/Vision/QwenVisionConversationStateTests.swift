import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite(.serialized) struct QwenVisionConversationStateTests {
    @Test func lineageReservesReplacementAndRejects1261stRowBeforeAllocation() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let firstTransaction = try await state.begin()
        let oldOwner = try makeLineageOwner(rows: 630, context: context)
        let oldReservation = try await state.reserveLineage(
            plannedRows: 630,
            requestedAllocationBytes: oldOwner.requestedAllocationBytes,
            transaction: firstTransaction)
        try await state.consumeLineage(
            oldReservation,
            replacement: QwenImageLineage(owners: [oldOwner], textRoPEDelta: 0),
            transaction: firstTransaction)
        let replacementTransaction = firstTransaction
        let replacementOwner = try makeLineageOwner(rows: 630, context: context)
        let replacementReservation = try await state.reserveLineage(
            plannedRows: 630,
            requestedAllocationBytes: replacementOwner.requestedAllocationBytes,
            transaction: replacementTransaction)
        let beforeUnboundCommit = await state.lineageDiagnostics()
        let beforeUnboundStatus = await state.status()
        #expect(beforeUnboundCommit.visibleRows == 0)
        #expect(beforeUnboundCommit.liveOwnerRows == 1_260)
        do {
            _ = try await state.commit(transaction: firstTransaction)
            Issue.record("owner-only lineage unexpectedly committed")
        } catch let error as ConversationStateTransactionError {
            guard case .invalidBoundary = error else {
                Issue.record("unexpected unbound commit error: \(error)")
                return
            }
        }
        let afterUnboundCommit = await state.lineageDiagnostics()
        let afterUnboundStatus = await state.status()
        #expect(afterUnboundCommit == beforeUnboundCommit)
        #expect(afterUnboundStatus == beforeUnboundStatus)
        #expect(afterUnboundCommit.highWaterOwnerRows == 1_260)
        #expect(afterUnboundCommit.highWaterRequestedBytes
            == beforeUnboundCommit.highWaterRequestedBytes)

        let reserved = await state.lineageDiagnostics()
        #expect(reserved.liveOwnerRows == 1_260)

        do {
            _ = try await state.reserveLineage(
                plannedRows: 1, requestedAllocationBytes: 1,
                transaction: replacementTransaction)
            Issue.record("1,261st lineage row unexpectedly reserved")
        } catch let error as QwenConversationLineageError {
            #expect(error == .liveRowsExceeded(requested: 1_261, maximum: 1_260))
        } catch {
            Issue.record("unexpected lineage quota error: \(error)")
        }
        let afterReject = await state.lineageDiagnostics()
        #expect(afterReject == beforeUnboundCommit)
        try await state.cancelLineage(
            replacementReservation, transaction: replacementTransaction)
        try await state.rollback(transaction: replacementTransaction)
    }

    @Test func removeSuffixReplaysPreparedMultimodalPrompt() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let transaction = try await state.begin()
        let owner = try makeLineageOwner(rows: 6, context: context)
        let fixture = try makePreparedImage(owner: owner)
        let reservation = try await state.reserveLineage(
            plannedRows: owner.rowCount,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: transaction)
        try await state.prefillMultimodal(
            fixture.prepared, lineage: fixture.lineage,
            reservation: reservation, transaction: transaction)
        for tokenID: Int32 in [9, 10, 11] {
            try await state.advance(tokenID, transaction: transaction)
        }
        try await state.removeSuffix(tokenCount: 2, transaction: transaction)
        let actualMetrics = try await state.commit(transaction: transaction)
        #expect(actualMetrics.retainedTokenIDs == fixture.prepared.tokenIDs + [9])
        let actual = try await state.diagnosticSnapshot()

        let clean = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let cleanTransaction = try await clean.begin()
        let cleanReservation = try await clean.reserveLineage(
            plannedRows: owner.rowCount,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: cleanTransaction)
        try await clean.prefillMultimodal(
            fixture.prepared, lineage: fixture.lineage,
            reservation: cleanReservation, transaction: cleanTransaction)
        try await clean.advance(9, transaction: cleanTransaction)
        _ = try await clean.commit(transaction: cleanTransaction)
        let expected = try await clean.diagnosticSnapshot()

        #expect(actual.lineage.visibleRows == 6)
        #expect(actual.lineage.allocationIDs == Set([owner.allocationID]))
        #expect(actual.lineage.ownedRequestedBytes == owner.requestedAllocationBytes)
        #expect(actual.textRoPEDelta == expected.textRoPEDelta)
        #expect(actual.runnerState == expected.runnerState)
        #expect(actual.currentLogits == expected.currentLogits)
        expectSemanticLineageEqual(actual.lineage, expected.lineage)
    }

    @Test func multimodalAppendRetainsCumulativeLineageAndOccurrenceProvenance() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)

        let firstTransaction = try await state.begin()
        let firstOwner = try makeLineageOwner(rows: 1, context: context)
        let firstFixture = try makePreparedSingleImage(owner: firstOwner)
        let firstReservation = try await state.reserveLineage(
            plannedRows: firstOwner.rowCount,
            requestedAllocationBytes: firstOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(firstFixture.prepared),
            transaction: firstTransaction)
        try await state.prefillMultimodal(
            firstFixture.prepared, lineage: firstFixture.lineage,
            reservation: firstReservation, transaction: firstTransaction)
        _ = try await state.commit(transaction: firstTransaction)

        let secondTransaction = try await state.begin()
        let secondOwner = try makeLineageOwner(rows: 1, context: context)
        let secondFixture = try makePreparedSingleImage(
            owner: secondOwner, positionOffset: firstFixture.prepared.tokenIDs.count)
        let secondReservation = try await state.reserveLineage(
            plannedRows: secondOwner.rowCount,
            requestedAllocationBytes: secondOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(secondFixture.prepared),
            transaction: secondTransaction)
        try await state.prefillMultimodal(
            secondFixture.prepared, lineage: secondFixture.lineage,
            reservation: secondReservation, transaction: secondTransaction)
        _ = try await state.commit(transaction: secondTransaction)

        let snapshot = try await state.diagnosticSnapshot()
        #expect(snapshot.lineage.visibleRows == 2)
        #expect(snapshot.lineage.allocationIDs == Set([
            firstOwner.allocationID, secondOwner.allocationID]))
        #expect(snapshot.lineage.ownedRequestedBytes
            == firstOwner.requestedAllocationBytes + secondOwner.requestedAllocationBytes)
        #expect(snapshot.lineage.provenanceRequestedBytes == 240)
        #expect(snapshot.lineage.liveRequestedBytes == 520)
        #expect(snapshot.lineage.highWaterRequestedBytes == 520)
        #expect(snapshot.lineage.occurrences.count == 2)
        let firstOccurrence = snapshot.lineage.occurrences[0]
        let secondOccurrence = snapshot.lineage.occurrences[1]
        #expect(firstOccurrence.ownerAllocationID == firstOwner.allocationID)
        #expect(firstOccurrence.absoluteTokenRange == 1..<2)
        #expect(firstOccurrence.exactPositions == Array(firstFixture.prepared.positions[1..<2]))
        #expect(secondOccurrence.ownerAllocationID == secondOwner.allocationID)
        #expect(secondOccurrence.absoluteTokenRange == 5..<6)
        #expect(secondOccurrence.exactPositions == Array(secondFixture.prepared.positions[1..<2]))
        #expect(firstOccurrence.segmentID != secondOccurrence.segmentID)
    }

    @Test func occurrenceDiagnosticsPreserveOwnerDigestsAcrossCommitAndRollback() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let originalImageDigest = String(repeating: "a", count: 64)
        let processorDigest = String(repeating: "b", count: 64)
        let originalOwner = try makeLineageOwner(
            rows: 6, context: context,
            imageDigest: originalImageDigest,
            processorDigest: processorDigest)
        let original = try makePreparedImage(owner: originalOwner)
        let initial = try await state.begin()
        let initialReservation = try await state.reserveLineage(
            plannedRows: originalOwner.rowCount,
            requestedAllocationBytes: originalOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(original.prepared),
            transaction: initial)
        try await state.prefillMultimodal(
            original.prepared, lineage: original.lineage,
            reservation: initialReservation, transaction: initial)
        _ = try await state.commit(transaction: initial)
        let committed = try await state.diagnosticSnapshot()
        #expect(committed.lineage.occurrences.count == 1)
        #expect(committed.lineage.occurrences[0].imageDigest == originalImageDigest)
        #expect(committed.lineage.occurrences[0].processorDigest == processorDigest)

        let changedImageDigest = String(repeating: "c", count: 64)
        let changedOwner = try makeLineageOwner(
            rows: 6, context: context,
            imageDigest: changedImageDigest,
            processorDigest: processorDigest)
        let changed = try makePreparedSecondImage(owner: changedOwner)
        let rollback = try await state.begin()
        try await state.prefill([9, 10], transaction: rollback) { _, _ in }
        let changedReservation = try await state.reserveLineage(
            plannedRows: changedOwner.rowCount,
            requestedAllocationBytes: changedOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(changed.prepared),
            transaction: rollback)
        try await state.prefillMultimodal(
            changed.prepared, lineage: changed.lineage,
            reservation: changedReservation, transaction: rollback)
        let working = try await state.diagnosticSnapshot()
        #expect(working.lineage.occurrences.map(\.imageDigest)
            == [originalImageDigest, changedImageDigest])
        #expect(working.lineage.occurrences.map(\.processorDigest)
            == [processorDigest, processorDigest])

        try await state.rollback(transaction: rollback)
        let restored = try await state.diagnosticSnapshot()
        #expect(restored.runnerState == committed.runnerState)
        #expect(restored.retainedTokenIDs == committed.retainedTokenIDs)
        #expect(restored.consumedTokenIDs == committed.consumedTokenIDs)
        #expect(restored.pendingAcceptedToken == committed.pendingAcceptedToken)
        #expect(restored.currentLogits == committed.currentLogits)
        #expect(restored.textRoPEDelta == committed.textRoPEDelta)
        #expect(restored.replayGeneration == committed.replayGeneration)
        #expect(restored.logicalStateBytes == committed.logicalStateBytes)
        #expect(restored.producerEpoch == committed.producerEpoch)
        expectSemanticLineageEqual(restored.lineage, committed.lineage)
        #expect(restored.lineage.occurrences.map(\.imageDigest)
            == [originalImageDigest])
        #expect(restored.lineage.occurrences.map(\.processorDigest)
            == [processorDigest])
    }

    @Test func multimodalAppendRejects631stVisibleOccurrenceBeforeDecoderSubmission() async throws {
        let submissions = SubmissionCounter()
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 2_048,
            executionHooks: QwenTextExecutionHooks(
                afterCommandSubmission: { _ in await submissions.record() }),
            publicationHooks: .none)
        let initialTransaction = try await state.begin()
        let initialOwner = try makeLineageOwner(rows: 6, context: context)
        let initial = try makePreparedImage(owner: initialOwner)
        let initialReservation = try await state.reserveLineage(
            plannedRows: 6,
            requestedAllocationBytes: initialOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(initial.prepared),
            transaction: initialTransaction)
        try await state.prefillMultimodal(
            initial.prepared, lineage: initial.lineage,
            reservation: initialReservation, transaction: initialTransaction)
        _ = try await state.commit(transaction: initialTransaction)
        await submissions.reset()

        let transaction = try await state.begin()
        let incomingOwner = try makeLineageOwner(
            rows: 625, context: context,
            grid: try QwenVisionGrid(temporal: 1, height: 50, width: 50))
        let incoming = try makePreparedVisibleQuotaImage(owner: incomingOwner)
        let reservation = try await state.reserveLineage(
            plannedRows: 625,
            requestedAllocationBytes: incomingOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(incoming.prepared),
            transaction: transaction)
        let before = try await state.diagnosticSnapshot()
        let statusBefore = await state.status()
        #expect(before.lineage.visibleRows == 6)
        #expect(before.lineage.liveOwnerRows == 631)
        do {
            try await state.prefillMultimodal(
                incoming.prepared, lineage: incoming.lineage,
                reservation: reservation, transaction: transaction)
            Issue.record("631st visible occurrence unexpectedly appended")
        } catch let error as QwenConversationLineageError {
            #expect(error == .visibleRowsExceeded(requested: 631, maximum: 630))
        } catch {
            Issue.record("unexpected visible-row error: \(error)")
        }
        let after = try await state.diagnosticSnapshot()
        let statusAfter = await state.status()
        #expect(after == before)
        #expect(statusAfter == statusBefore)
        let submissionCount = await submissions.count()
        #expect(submissionCount == 0)
        try await state.cancelLineage(reservation, transaction: transaction)
        try await state.rollback(transaction: transaction)
    }

    @Test func reservationOverflowIsTypedAndDoesNotMutateLedger() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let transaction = try await state.begin()
        let first = try await state.reserveLineage(
            plannedRows: 0, requestedAllocationBytes: 1,
            transaction: transaction)
        let before = await state.lineageDiagnostics()

        do {
            _ = try await state.reserveLineage(
                plannedRows: 0, requestedAllocationBytes: Int.max,
                transaction: transaction)
            Issue.record("overflowing reservation unexpectedly succeeded")
        } catch let error as QwenConversationLineageError {
            #expect(error == .requestedBytesExceeded(
                requested: Int.max,
                maximum: QwenVisionResourceLimits.provisional.maximumMutablePreparedBytes))
        } catch {
            Issue.record("unexpected reservation error: \(error)")
        }
        let after = await state.lineageDiagnostics()
        #expect(after == before)
        try await state.cancelLineage(first, transaction: transaction)
        try await state.rollback(transaction: transaction)
    }

    @Test func cancelLineageDuringSuspendedPrefillIsBusy() async throws {
        let gate = VisionSubmissionGate()
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32,
            executionHooks: QwenTextExecutionHooks(
                afterCommandSubmission: { _ in await gate.holdFirstSubmission() }),
            publicationHooks: .none)
        let transaction = try await state.begin()
        let owner = try makeLineageOwner(rows: 6, context: context)
        let fixture = try makePreparedImage(owner: owner)
        let reservation = try await state.reserveLineage(
            plannedRows: owner.rowCount,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: transaction)
        let operationOutcome = BoundedBoolOutcome()
        let operation = Task {
            do {
                try await state.prefillMultimodal(
                    fixture.prepared, lineage: fixture.lineage,
                    reservation: reservation, transaction: transaction)
                await operationOutcome.record(false)
            } catch {
                await operationOutcome.record(true)
            }
        }

        guard await gate.waitUntilSubmitted(timeoutNanoseconds: 5_000_000_000) else {
            Issue.record("multimodal prefill submission was not observed within 5 seconds")
            operation.cancel()
            await gate.release()
            if await operationOutcome.wait(timeoutNanoseconds: 5_000_000_000) {
                #expect(await operationOutcome.value == false)
            } else {
                Issue.record(
                    "multimodal prefill task did not settle within 5 seconds after the submission timeout")
            }
            return
        }
        await #expect(throws: ConversationStateTransactionError.busy) {
            try await state.cancelLineage(reservation, transaction: transaction)
        }
        #expect((await state.lineageDiagnostics()).liveOwnerRows == owner.rowCount)
        await gate.release()
        guard await operationOutcome.wait(timeoutNanoseconds: 5_000_000_000) else {
            Issue.record("multimodal prefill did not settle within 5 seconds after release")
            return
        }
        operation.cancel()
        #expect(await operationOutcome.value == false)
        let metrics = try await state.commit(transaction: transaction)
        #expect(metrics.retainedTokenIDs == fixture.prepared.tokenIDs)
        let snapshot = try await state.diagnosticSnapshot()
        #expect(snapshot.lineage.visibleRows == owner.rowCount)
    }

    @Test func reservationLedgerCommitsBeforeOwnerAllocation() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let transaction = try await state.begin()
        // Reserve the complete requested owner footprint before constructing
        // any Metal-backed owner. This is the pre-allocation admission seam.
        let plannedBytes = 88_200
        let reservation = try await state.reserveLineage(
            plannedRows: 630,
            requestedAllocationBytes: plannedBytes,
            transaction: transaction)
        let reserved = await state.lineageDiagnostics()
        #expect(reserved.visibleRows == 0)
        #expect(reserved.liveOwnerRows == 630)
        #expect(reserved.liveRequestedBytes == plannedBytes)
        let owner = try makeLineageOwner(rows: 630, context: context)
        #expect(owner.requestedAllocationBytes == plannedBytes)
        try await state.consumeLineage(
            reservation,
            replacement: QwenImageLineage(owners: [owner]),
            transaction: transaction)
        try await state.rollback(transaction: transaction)
        let after = await state.lineageDiagnostics()
        #expect(after.visibleRows == 0)
        #expect(after.liveOwnerRows == 0)
    }

    @Test func multimodalPrefillCommitDecodeRollbackAndRebuildMatchCleanCheckpoint() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let transaction = try await state.begin()
        // Reserve before constructing any Metal-backed owner.
        let ownerBytes = 6 * 32 * MemoryLayout<Float>.stride
            + 6 * 3 * MemoryLayout<Int32>.stride
        let owner = try makeLineageOwner(rows: 6, context: context)
        let fixture = try makePreparedImage(owner: owner)
        let reservation = try await state.reserveLineage(
            plannedRows: 6, requestedAllocationBytes: ownerBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: transaction)
        try await state.prefillMultimodal(
            fixture.prepared, lineage: fixture.lineage,
            reservation: reservation, transaction: transaction)
        let committed = try await state.commit(transaction: transaction)
        #expect(committed.consumedTokenCount == 8)
        #expect(committed.pendingTokenCount == 0)
        let imageSnapshot = try await state.diagnosticSnapshot()
        #expect(imageSnapshot.runnerState.sequenceLength == 8)
        #expect(imageSnapshot.textRoPEDelta == -3)
        #expect(imageSnapshot.lineage.visibleRows == 6)
        #expect(imageSnapshot.lineage.ownedRequestedBytes == ownerBytes)
        #expect(imageSnapshot.lineage.provenanceRequestedBytes == 184)
        #expect(imageSnapshot.lineage.liveRequestedBytes == 1_024)

        let decode = try await state.begin()
        try await state.advance(9, transaction: decode)
        _ = try await state.commit(transaction: decode)
        let beforeRollback = try await state.diagnosticSnapshot()
        #expect(beforeRollback.pendingAcceptedToken == 9)
        #expect(beforeRollback.runnerState.sequenceLength == 8)
        let consuming = try await state.begin()
        let afterDecode = try await state.diagnosticSnapshot()
        #expect(afterDecode.pendingAcceptedToken == nil)
        #expect(afterDecode.runnerState.sequenceLength == 9)
        #expect(afterDecode.textRoPEDelta == -3)
        try await state.rollback(transaction: consuming)
        let rolledBack = try await state.diagnosticSnapshot()
        #expect(rolledBack.producerEpoch == beforeRollback.producerEpoch)
        #expect(rolledBack.runnerState == beforeRollback.runnerState)
        #expect(rolledBack.retainedTokenIDs == beforeRollback.retainedTokenIDs)
        #expect(rolledBack.currentLogits == beforeRollback.currentLogits)
        #expect(rolledBack.lineage == beforeRollback.lineage)

        // Rebuild from the complete multimodal checkpoint, then compare with
        // a clean state that executes the same prepared image turn.
        let rebuild = try await state.begin()
        let rebuildReservation = try await state.reserveLineage(
            plannedRows: 6, requestedAllocationBytes: ownerBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: rebuild)
        try await state.rebuildMultimodalCheckpoint(
            prepared: fixture.prepared, lineage: fixture.lineage,
            reservation: rebuildReservation, transaction: rebuild)
        _ = try await state.commit(transaction: rebuild)
        let rebuilt = try await state.diagnosticSnapshot()
        #expect(rebuilt.producerEpoch == rolledBack.producerEpoch)

        let clean = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let cleanTransaction = try await clean.begin()
        let cleanReservation = try await clean.reserveLineage(
            plannedRows: 6, requestedAllocationBytes: ownerBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: cleanTransaction)
        try await clean.prefillMultimodal(
            fixture.prepared, lineage: fixture.lineage,
            reservation: cleanReservation, transaction: cleanTransaction)
        _ = try await clean.commit(transaction: cleanTransaction)
        let expected = try await clean.diagnosticSnapshot()
        #expect(rebuilt.runnerState == expected.runnerState)
        #expect(rebuilt.retainedTokenIDs == expected.retainedTokenIDs)
        #expect(rebuilt.currentLogits == expected.currentLogits)
        #expect(rebuilt.textRoPEDelta == expected.textRoPEDelta)
        #expect(rebuilt.lineage.visibleRows == expected.lineage.visibleRows)
        #expect(rebuilt.lineage.allocationIDs == expected.lineage.allocationIDs)
        #expect(rebuilt.lineage.ownedRequestedBytes == expected.lineage.ownedRequestedBytes)
        #expect(rebuilt.lineage.highWaterOwnerRows == 12)
        #expect(rebuilt.lineage.highWaterRequestedBytes == 2_048)
        #expect(expected.lineage.highWaterOwnerRows == 6)
        #expect(expected.lineage.highWaterRequestedBytes == 1_024)
    }

    @Test func changedOwnerDigestAndProfileAreRejectedBeforeDecoderMutation() async throws {
        try await assertMetadataRejection(changedDigest: true)
        try await assertMetadataRejection(changedDigest: false)
    }

    @Test func staleSubmittedMultimodalCompletionRetainsReservationAndRestoresState() async throws {
        let gate = VisionSubmissionGate()
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32,
            executionHooks: QwenTextExecutionHooks(
                afterCommandSubmission: { _ in await gate.holdFirstSubmission() }),
            publicationHooks: .none)
        let transaction = try await state.begin()
        let ownerBytes = 6 * 32 * MemoryLayout<Float>.stride
            + 6 * 3 * MemoryLayout<Int32>.stride
        let owner = try makeLineageOwner(rows: 6, context: context)
        let fixture = try makePreparedImage(owner: owner)
        let reservation = try await state.reserveLineage(
            plannedRows: 6, requestedAllocationBytes: ownerBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: transaction)
        let before = try await state.diagnosticSnapshot()
        let operationOutcome = BoundedBoolOutcome()
        let operation = Task {
            do {
                try await state.prefillMultimodal(
                    fixture.prepared, lineage: fixture.lineage,
                    reservation: reservation, transaction: transaction)
                await operationOutcome.record(false)
            } catch {
                await operationOutcome.record(true)
            }
        }
        guard await gate.waitUntilSubmitted(timeoutNanoseconds: 5_000_000_000) else {
            Issue.record("stale multimodal submission was not observed within 5 seconds")
            operation.cancel()
            await gate.release()
            if await operationOutcome.wait(timeoutNanoseconds: 5_000_000_000) {
                #expect(await operationOutcome.value == true)
            } else {
                Issue.record(
                    "stale multimodal task did not settle within 5 seconds after the submission timeout")
            }
            return
        }
        operation.cancel()
        #expect((await state.status()).activeTransaction == transaction)
        await gate.release()
        guard await operationOutcome.wait(timeoutNanoseconds: 5_000_000_000) else {
            Issue.record("stale multimodal completion did not settle within 5 seconds")
            return
        }
        #expect(await operationOutcome.value == true)
        let after = try await state.diagnosticSnapshot()
        #expect(after.producerEpoch == before.producerEpoch)
        #expect(after.runnerState == before.runnerState)
        #expect(after.retainedTokenIDs == before.retainedTokenIDs)
        #expect(after.currentLogits == before.currentLogits)
        #expect(after.lineage.visibleRows == 0)
        #expect(after.lineage.liveOwnerRows == 6)
        try await state.cancelLineage(reservation, transaction: transaction)
        try await state.rollback(transaction: transaction)
    }

    @Test func lineageOwnerRejectsGridRowMismatch() throws {
        let context = try MetalContext()
        let invalidGrid = try QwenVisionGrid(temporal: 1, height: 2, width: 2)
        #expect(throws: QwenVisionError.invalidPositions) {
            try makeLineageOwner(rows: 630, context: context, grid: invalidGrid)
        }
    }

    @Test func lineageAliasesCountOnceAndCancellationIsTerminal() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let transaction = try await state.begin()
        let owner = try makeLineageOwner(rows: 2, context: context)
        let aliased = QwenImageLineage(owners: [owner, owner])
        #expect(aliased.rowCount == 2)
        #expect(aliased.allocationIDs.count == 1)
        let copiedOwner = try makeLineageOwner(rows: 2, context: context)
        let copied = QwenImageLineage(owners: [owner, copiedOwner])
        #expect(copied.rowCount == 4)
        #expect(copied.allocationIDs.count == 2)
        #expect(copied.ownedRequestedBytes
            == owner.requestedAllocationBytes + copiedOwner.requestedAllocationBytes)
        let reservation = try await state.reserveLineage(
            plannedRows: 2,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            transaction: transaction)
        try await state.cancelLineage(reservation, transaction: transaction)
        do {
            try await state.cancelLineage(reservation, transaction: transaction)
            Issue.record("lineage reservation cancelled twice")
        } catch let error as QwenConversationLineageError {
            #expect(error == .invalidReservation)
        } catch {
            Issue.record("unexpected cancellation error: \(error)")
        }
        _ = try await state.rollback(transaction: transaction)
    }

    @Test func ownerOnlyLineageCannotCommitOrConsumeTokens() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let transaction = try await state.begin()
        let owner = try makeLineageOwner(rows: 1, context: context)
        let reservation = try await state.reserveLineage(
            plannedRows: owner.rowCount,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            transaction: transaction)
        try await state.consumeLineage(
            reservation,
            replacement: QwenImageLineage(owners: [owner]),
            transaction: transaction)
        let before = await state.lineageDiagnostics()
        #expect(before.visibleRows == 0)
        #expect(before.liveOwnerRows == 1)
        do {
            _ = try await state.commit(transaction: transaction)
            Issue.record("owner-only lineage unexpectedly committed")
        } catch let error as ConversationStateTransactionError {
            guard case .invalidBoundary = error else {
                Issue.record("unexpected unbound commit error: \(error)")
                return
            }
        }
        let unchangedAfterCommit = await state.lineageDiagnostics()
        #expect(unchangedAfterCommit == before)
        await #expect(throws: ConversationStateTransactionError.invalidBoundary(
            "owner-only image accounting is not publishable")) {
            try await state.prefill([1], transaction: transaction) { _, _ in }
        }
        let unchangedAfterTokenGuard = await state.lineageDiagnostics()
        #expect(unchangedAfterTokenGuard == before)
        try await state.rollback(transaction: transaction)
        let after = await state.lineageDiagnostics()
        #expect(after.visibleRows == 0)
        #expect(after.liveOwnerRows == 0)

        // A consumed prompt step cannot be replaced by owner-only accounting.
        let consumedState = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let consumedTransaction = try await consumedState.begin()
        try await consumedState.prefill([1], transaction: consumedTransaction) { _, _ in }
        let consumedOwner = try makeLineageOwner(rows: 1, context: context)
        let consumedReservation = try await consumedState.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: consumedOwner.requestedAllocationBytes,
            transaction: consumedTransaction)
        try await assertOwnerOnlyConsumeRejected(
            state: consumedState, reservation: consumedReservation,
            owner: consumedOwner, transaction: consumedTransaction)

        // An accepted-but-pending token is also a non-token-free boundary.
        let pendingState = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let pendingTransaction = try await pendingState.begin()
        try await pendingState.advance(1, transaction: pendingTransaction)
        let pendingOwner = try makeLineageOwner(rows: 1, context: context)
        let pendingReservation = try await pendingState.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: pendingOwner.requestedAllocationBytes,
            transaction: pendingTransaction)
        try await assertOwnerOnlyConsumeRejected(
            state: pendingState, reservation: pendingReservation,
            owner: pendingOwner, transaction: pendingTransaction)

        // Bound image occurrences cannot be overwritten by owner-only state.
        let boundState = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let boundTransaction = try await boundState.begin()
        let boundOwner = try makeLineageOwner(rows: 1, context: context)
        let boundFixture = try makePreparedSingleImage(owner: boundOwner)
        let boundReservation = try await boundState.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: boundOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(boundFixture.prepared),
            transaction: boundTransaction)
        try await boundState.prefillMultimodal(
            boundFixture.prepared, lineage: boundFixture.lineage,
            reservation: boundReservation, transaction: boundTransaction)
        let replacementOwner = try makeLineageOwner(rows: 1, context: context)
        let replacementReservation = try await boundState.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: replacementOwner.requestedAllocationBytes,
            transaction: boundTransaction)
        try await assertOwnerOnlyConsumeRejected(
            state: boundState, reservation: replacementReservation,
            owner: replacementOwner, transaction: boundTransaction)
    }

    @Test func sharedOwnerOccurrencesCountVisibleRowsAndPhysicalStorageSeparately() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let transaction = try await state.begin()
        let owner = try makeLineageOwner(rows: 1, context: context)
        let first = try makePreparedSingleImage(owner: owner)
        let firstReservation = try await state.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(first.prepared),
            transaction: transaction)
        try await state.prefillMultimodal(
            first.prepared, lineage: first.lineage,
            reservation: firstReservation, transaction: transaction)
        let second = try makePreparedSingleImage(owner: owner, positionOffset: 4)
        let secondReservation = try await state.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(second.prepared),
            transaction: transaction)
        try await state.prefillMultimodal(
            second.prepared, lineage: second.lineage,
            reservation: secondReservation, transaction: transaction)
        _ = try await state.commit(transaction: transaction)

        let snapshot = try await state.diagnosticSnapshot()
        #expect(snapshot.lineage.visibleRows == 2)
        #expect(snapshot.lineage.allocationIDs == Set([owner.allocationID]))
        #expect(snapshot.lineage.ownedRequestedBytes == owner.requestedAllocationBytes)
        #expect(snapshot.lineage.provenanceRequestedBytes == 240)
        #expect(snapshot.lineage.liveOwnerRows == 1)
        #expect(snapshot.lineage.liveRequestedBytes == 380)
        #expect(snapshot.lineage.occurrences.count == 2)
        #expect(snapshot.lineage.occurrences[0].ownerAllocationID == owner.allocationID)
        #expect(snapshot.lineage.occurrences[1].ownerAllocationID == owner.allocationID)
        #expect(snapshot.lineage.occurrences[0].absoluteTokenRange == 1..<2)
        #expect(snapshot.lineage.occurrences[1].absoluteTokenRange == 5..<6)
        #expect(snapshot.lineage.occurrences[0].segmentID
            != snapshot.lineage.occurrences[1].segmentID)
    }

    @Test func twoCommittedImageTurnsWithTextMatchFullHistoryRebuild() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let firstOwner = try makeLineageOwner(rows: 6, context: context)
        let firstFixture = try makePreparedImage(owner: firstOwner)
        let firstTransaction = try await state.begin()
        let firstReservation = try await state.reserveLineage(
            plannedRows: 6,
            requestedAllocationBytes: firstOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(firstFixture.prepared),
            transaction: firstTransaction)
        try await state.prefillMultimodal(
            firstFixture.prepared, lineage: firstFixture.lineage,
            reservation: firstReservation, transaction: firstTransaction)
        _ = try await state.commit(transaction: firstTransaction)
        let firstSnapshot = try await state.diagnosticSnapshot()
        #expect(firstSnapshot.textRoPEDelta == -3)
        #expect(firstSnapshot.lineage.visibleRows == 6)
        #expect(firstSnapshot.lineage.provenanceRequestedBytes == 184)
        #expect(firstSnapshot.lineage.liveRequestedBytes == 1_024)

        let splitTransaction = try await state.begin()
        try await state.prefill([9, 10], transaction: splitTransaction) { _, _ in }
        let secondOwner = try makeLineageOwner(
            rows: 6, context: context,
            imageDigest: String(repeating: "c", count: 64))
        let secondFixture = try makePreparedSecondImage(owner: secondOwner)
        let secondReservation = try await state.reserveLineage(
            plannedRows: 6,
            requestedAllocationBytes: secondOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(secondFixture.prepared),
            transaction: splitTransaction)
        try await state.prefillMultimodal(
            secondFixture.prepared, lineage: secondFixture.lineage,
            reservation: secondReservation, transaction: splitTransaction)
        _ = try await state.commit(transaction: splitTransaction)
        let splitSnapshot = try await state.diagnosticSnapshot()
        #expect(splitSnapshot.retainedTokenIDs
            == firstFixture.prepared.tokenIDs + [9, 10] + secondFixture.prepared.tokenIDs)
        #expect(splitSnapshot.textRoPEDelta == -6)
        #expect(splitSnapshot.lineage.visibleRows == 12)
        #expect(splitSnapshot.lineage.provenanceRequestedBytes == 368)
        #expect(splitSnapshot.lineage.liveRequestedBytes == 2_048)
        #expect(splitSnapshot.lineage.occurrences.map(\.absoluteTokenRange)
            == [1..<7, 11..<17])

        let fixture = try makePreparedTwoImage(
            firstOwner: firstOwner, secondOwner: secondOwner)
        let ownerBytes = firstOwner.requestedAllocationBytes
            + secondOwner.requestedAllocationBytes
        let rebuild = try await state.begin()
        let rebuildReservation = try await state.reserveLineage(
            plannedRows: 12, requestedAllocationBytes: ownerBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: rebuild)
        try await state.rebuildMultimodalCheckpoint(
            prepared: fixture.prepared, lineage: fixture.lineage,
            reservation: rebuildReservation, transaction: rebuild)
        let rebuilt = try await state.diagnosticSnapshot()
        #expect(rebuilt.retainedTokenIDs == fixture.prepared.tokenIDs)
        #expect(rebuilt.runnerState.sequenceLength == 18)
        #expect(rebuilt.textRoPEDelta == -6)
        #expect(rebuilt.lineage.visibleRows == 12)
        #expect(rebuilt.lineage.provenanceRequestedBytes == 368)
        #expect(rebuilt.lineage.highWaterOwnerRows == 24)
        #expect(rebuilt.lineage.highWaterRequestedBytes == 4_096)

        let clean = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let cleanTransaction = try await clean.begin()
        let cleanReservation = try await clean.reserveLineage(
            plannedRows: 12, requestedAllocationBytes: ownerBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: cleanTransaction)
        try await clean.prefillMultimodal(
            fixture.prepared, lineage: fixture.lineage,
            reservation: cleanReservation, transaction: cleanTransaction)
        let cleanImage = try await clean.diagnosticSnapshot()
        #expect(splitSnapshot.runnerState == rebuilt.runnerState)
        #expect(splitSnapshot.currentLogits == rebuilt.currentLogits)
        #expect(splitSnapshot.retainedTokenIDs == rebuilt.retainedTokenIDs)
        #expect(splitSnapshot.textRoPEDelta == rebuilt.textRoPEDelta)
        expectCurrentImageLineageEqual(splitSnapshot.lineage, rebuilt.lineage)
        #expect(splitSnapshot.runnerState == cleanImage.runnerState)
        #expect(splitSnapshot.currentLogits == cleanImage.currentLogits)
        #expect(splitSnapshot.retainedTokenIDs == cleanImage.retainedTokenIDs)
        #expect(splitSnapshot.textRoPEDelta == cleanImage.textRoPEDelta)
        expectSemanticLineageEqual(splitSnapshot.lineage, cleanImage.lineage)

        for tokenID: Int32 in [9, 10, 11] {
            try await state.advance(tokenID, transaction: rebuild)
        }
        try await state.removeSuffix(tokenCount: 2, transaction: rebuild)
        let actualMetrics = try await state.commit(transaction: rebuild)
        #expect(actualMetrics.retainedTokenIDs == fixture.prepared.tokenIDs + [9])
        let actual = try await state.diagnosticSnapshot()

        try await clean.advance(9, transaction: cleanTransaction)
        _ = try await clean.commit(transaction: cleanTransaction)
        let expected = try await clean.diagnosticSnapshot()
        #expect(actual.runnerState == expected.runnerState)
        #expect(actual.currentLogits == expected.currentLogits)
        #expect(actual.textRoPEDelta == expected.textRoPEDelta)
        expectSemanticLineageEqual(actual.lineage, expected.lineage)
        #expect(actual.lineage.highWaterOwnerRows == 24)
        #expect(actual.lineage.highWaterRequestedBytes == 4_096)
        #expect(expected.lineage.highWaterOwnerRows == 12)
        #expect(expected.lineage.highWaterRequestedBytes == 2_048)
    }

    @Test func removeSuffixAfterMultimodalRebuildUsesReplacementOrigin() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let originalOwner = try makeLineageOwner(rows: 6, context: context)
        let original = try makePreparedImage(owner: originalOwner)
        let originalTransaction = try await state.begin()
        let originalReservation = try await state.reserveLineage(
            plannedRows: 6,
            requestedAllocationBytes: originalOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(original.prepared),
            transaction: originalTransaction)
        try await state.prefillMultimodal(
            original.prepared, lineage: original.lineage,
            reservation: originalReservation, transaction: originalTransaction)
        _ = try await state.commit(transaction: originalTransaction)

        let replacementOwner = try makeLineageOwner(rows: 6, context: context)
        let replacement = try makePreparedImage(owner: replacementOwner)
        let transaction = try await state.begin()
        let replacementReservation = try await state.reserveLineage(
            plannedRows: 6,
            requestedAllocationBytes: replacementOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(replacement.prepared),
            transaction: transaction)
        try await state.rebuildMultimodalCheckpoint(
            prepared: replacement.prepared, lineage: replacement.lineage,
            reservation: replacementReservation, transaction: transaction)
        try await state.advance(9, transaction: transaction)
        try await state.advance(10, transaction: transaction)
        try await state.advance(11, transaction: transaction)
        try await state.removeSuffix(tokenCount: 2, transaction: transaction)
        let metrics = try await state.commit(transaction: transaction)
        #expect(metrics.retainedTokenIDs == replacement.prepared.tokenIDs + [9])

        let actual = try await state.diagnosticSnapshot()
        let clean = try await QwenConversationState(
            model: model, context: context, maxContext: 32)
        let cleanTransaction = try await clean.begin()
        let cleanReservation = try await clean.reserveLineage(
            plannedRows: 6,
            requestedAllocationBytes: replacementOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(replacement.prepared),
            transaction: cleanTransaction)
        try await clean.prefillMultimodal(
            replacement.prepared, lineage: replacement.lineage,
            reservation: cleanReservation, transaction: cleanTransaction)
        try await clean.advance(9, transaction: cleanTransaction)
        _ = try await clean.commit(transaction: cleanTransaction)
        let expected = try await clean.diagnosticSnapshot()
        #expect(actual.runnerState == expected.runnerState)
        #expect(actual.currentLogits == expected.currentLogits)
        #expect(actual.retainedTokenIDs == expected.retainedTokenIDs)
        expectSemanticLineageEqual(actual.lineage, expected.lineage)
    }

    @Test func removeSuffixReplayFailureRestoresCompleteTransaction() async throws {
        let failures = ReplayFailureSwitch()
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32,
            executionHooks: QwenTextExecutionHooks(
                beforeLayer: { layer in try await failures.beforeLayer(layer) }),
            publicationHooks: .none)
        let transaction = try await state.begin()
        let owner = try makeLineageOwner(rows: 6, context: context)
        let fixture = try makePreparedImage(owner: owner)
        let reservation = try await state.reserveLineage(
            plannedRows: owner.rowCount,
            requestedAllocationBytes: owner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(fixture.prepared),
            transaction: transaction)
        try await state.prefillMultimodal(
            fixture.prepared, lineage: fixture.lineage,
            reservation: reservation, transaction: transaction)
        try await state.prefill([9], transaction: transaction) { _, _ in }
        try await state.advance(10, transaction: transaction)
        try await state.advance(11, transaction: transaction)
        let before = try await state.diagnosticSnapshot()
        await failures.arm(failingOperation: 2)
        do {
            try await state.removeSuffix(tokenCount: 1, transaction: transaction)
            Issue.record("partial replay unexpectedly succeeded")
        } catch let error as ReplayFailure {
            #expect(error == .injected)
        } catch {
            Issue.record("unexpected replay failure: \(error)")
        }
        let startedOperations = await failures.startedOperationCount()
        #expect(startedOperations == 2)
        let after = try await state.diagnosticSnapshot()
        #expect(after == before)
        #expect((await state.status()).activeTransaction == transaction)
        await failures.disarm()
        try await state.removeSuffix(tokenCount: 1, transaction: transaction)
        let retryMetrics = try await state.commit(transaction: transaction)
        #expect(retryMetrics.retainedTokenIDs == fixture.prepared.tokenIDs + [9, 10])
    }

    @Test func multimodalPositionOracleUsesPerBoundaryDeltas() throws {
        let grid = try QwenVisionGrid(temporal: 1, height: 4, width: 6)
        let plan = try QwenMultimodalPositions.make(
            tokenCount: 18, imageRanges: [1..<7, 11..<17],
            grids: [grid, grid], merge: 2, maximumRows: 630)
        let expected: [[Int32]] = [
            [0, 0, 0],
            [1, 1, 1], [1, 1, 2], [1, 1, 3],
            [1, 2, 1], [1, 2, 2], [1, 2, 3],
            [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7],
            [8, 8, 8], [8, 8, 9], [8, 8, 10],
            [8, 9, 8], [8, 9, 9], [8, 9, 10],
            [11, 11, 11],
        ]
        #expect(plan.textRoPEDelta == -6)
        #expect(plan.positions.map(\.values) == expected)
    }

    private func expectCurrentImageLineageEqual(
        _ actual: QwenConversationLineageDiagnostics,
        _ expected: QwenConversationLineageDiagnostics
    ) {
        #expect(actual.visibleRows == expected.visibleRows)
        #expect(actual.logicalBytes == expected.logicalBytes)
        #expect(actual.allocationIDs == expected.allocationIDs)
        #expect(actual.ownedRequestedBytes == expected.ownedRequestedBytes)
        #expect(actual.provenanceRequestedBytes == expected.provenanceRequestedBytes)
        #expect(actual.occurrences.count == expected.occurrences.count)
        for (left, right) in zip(actual.occurrences, expected.occurrences) {
            #expect(left.ownerAllocationID == right.ownerAllocationID)
            #expect(left.absoluteTokenRange == right.absoluteTokenRange)
            #expect(left.exactPositions == right.exactPositions)
            #expect(left.imageDigest == right.imageDigest)
            #expect(left.processorDigest == right.processorDigest)
        }
    }

    private func expectSemanticLineageEqual(
        _ actual: QwenConversationLineageDiagnostics,
        _ expected: QwenConversationLineageDiagnostics
    ) {
        #expect(actual.visibleRows == expected.visibleRows)
        #expect(actual.logicalBytes == expected.logicalBytes)
        #expect(actual.allocationIDs == expected.allocationIDs)
        #expect(actual.ownedRequestedBytes == expected.ownedRequestedBytes)
        #expect(actual.provenanceRequestedBytes == expected.provenanceRequestedBytes)
        #expect(actual.liveOwnerRows == expected.liveOwnerRows)
        #expect(actual.liveRequestedBytes == expected.liveRequestedBytes)
        #expect(actual.occurrences.count == expected.occurrences.count)
        for (left, right) in zip(actual.occurrences, expected.occurrences) {
            #expect(left.ownerAllocationID == right.ownerAllocationID)
            #expect(left.absoluteTokenRange == right.absoluteTokenRange)
            #expect(left.exactPositions == right.exactPositions)
            #expect(left.imageDigest == right.imageDigest)
            #expect(left.processorDigest == right.processorDigest)
        }
    }

    private func assertOwnerOnlyConsumeRejected(
        state: QwenConversationState,
        reservation: QwenLineageReservation,
        owner: QwenRetainedFeatureOwner,
        transaction: ConversationTransactionID
    ) async throws {
        let before = await state.lineageDiagnostics()
        do {
            try await state.consumeLineage(
                reservation,
                replacement: QwenImageLineage(owners: [owner]),
                transaction: transaction)
            Issue.record("owner-only consume unexpectedly succeeded")
        } catch let error as QwenConversationLineageError {
            #expect(error == .invalidReservation)
        } catch {
            Issue.record("unexpected owner-only consume error: \(error)")
        }
        let after = await state.lineageDiagnostics()
        #expect(after == before)
        try await state.cancelLineage(reservation, transaction: transaction)
        try await state.rollback(transaction: transaction)
    }

    private func makeProvenancePlan(
        _ prepared: QwenPreparedPrefill
    ) throws -> QwenLineageProvenancePlan {
        try QwenLineageProvenancePlan(
            tokenCount: prepared.tokenIDs.count,
            positionCount: prepared.positions.count,
            featureOverrideCount: prepared.featureOverrides.count)
    }

    private func makeLineageOwner(
        rows: Int, context: MetalContext, grid: QwenVisionGrid? = nil,
        imageDigest: String = String(repeating: "a", count: 64),
        processorDigest: String = String(repeating: "b", count: 64),
        profile: GTurboQwenVisionProcessorProfileV2 = GTurboQwenVisionProcessorProfileV2(
            processorClass: "Qwen3VLProcessor",
            imageProcessorType: "Qwen2VLImageProcessorFast",
            patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
    ) throws -> QwenRetainedFeatureOwner {
        let ownerGrid: QwenVisionGrid
        if let grid {
            ownerGrid = grid
        } else {
            switch rows {
            case 630: ownerGrid = try QwenVisionGrid(temporal: 1, height: 42, width: 60)
            case 6: ownerGrid = try QwenVisionGrid(temporal: 1, height: 4, width: 6)
            case 2: ownerGrid = try QwenVisionGrid(temporal: 1, height: 2, width: 4)
            case 1: ownerGrid = try QwenVisionGrid(temporal: 1, height: 2, width: 2)
            default: throw QwenVisionError.invalidPositions
            }
        }
        guard try ownerGrid.mergedRows(merge: 2) == rows else {
            throw QwenVisionError.invalidPositions
        }
        var positions: [QwenMRoPEPosition] = []
        positions.reserveCapacity(rows)
        for temporal in 0..<ownerGrid.temporal {
            for height in 0..<(ownerGrid.height / 2) {
                for width in 0..<(ownerGrid.width / 2) {
                    positions.append(try QwenMRoPEPosition(
                        temporal: temporal, height: height, width: width))
                }
            }
        }
        return try QwenRetainedFeatureOwner(
            device: context.device,
            features: [Float](repeating: 0.25, count: rows * 32),
            positions: positions,
            imageDigest: imageDigest,
            processorDigest: processorDigest,
            profile: profile,
            grid: ownerGrid,
            hiddenSize: 32)
    }

    private struct PreparedImage {
        let prepared: QwenPreparedPrefill
        let lineage: QwenImageLineage
    }

    private func makePreparedImage(
        owner: QwenRetainedFeatureOwner
    ) throws -> PreparedImage {
        let tokenIDs: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        let plan = try QwenMultimodalPositions.make(
            tokenCount: tokenIDs.count,
            imageRanges: [1..<7],
            grids: [owner.grid], merge: 2, maximumRows: 630)
        let prepared = try QwenPreparedPrefill(
            tokenIDs: tokenIDs,
            featureOverrides: [QwenPreparedFeatureOverride(
                tokenRange: 1..<7, owner: owner)],
            positions: plan.positions,
            textRoPEDelta: plan.textRoPEDelta)
        return PreparedImage(
            prepared: prepared,
            lineage: QwenImageLineage(
                owners: [owner], textRoPEDelta: plan.textRoPEDelta))
    }

    private func makePreparedVisibleQuotaImage(
        owner: QwenRetainedFeatureOwner
    ) throws -> PreparedImage {
        let tokenIDs: [Int32] = (0..<626).map { Int32($0 % 19) }
        let plan = try QwenMultimodalPositions.make(
            tokenCount: tokenIDs.count,
            imageRanges: [1..<626],
            grids: [owner.grid], merge: 2, maximumRows: 630)
        let prepared = try QwenPreparedPrefill(
            tokenIDs: tokenIDs,
            featureOverrides: [QwenPreparedFeatureOverride(
                tokenRange: 1..<626, owner: owner)],
            positions: plan.positions,
            textRoPEDelta: plan.textRoPEDelta)
        return PreparedImage(
            prepared: prepared,
            lineage: QwenImageLineage(
                owners: [owner], textRoPEDelta: plan.textRoPEDelta))
    }

    private func makePreparedSecondImage(
        owner: QwenRetainedFeatureOwner
    ) throws -> PreparedImage {
        let tokenIDs: [Int32] = [11, 12, 13, 14, 15, 16, 17, 18]
        let plan = try QwenMultimodalPositions.make(
            tokenCount: tokenIDs.count,
            imageRanges: [1..<7],
            grids: [owner.grid], merge: 2, maximumRows: 630)
        let positions = try plan.positions.map {
            try QwenMRoPEPosition(
                temporal: Int($0.temporal) + 7,
                height: Int($0.height) + 7,
                width: Int($0.width) + 7)
        }
        let prepared = try QwenPreparedPrefill(
            tokenIDs: tokenIDs,
            featureOverrides: [QwenPreparedFeatureOverride(
                tokenRange: 1..<7, owner: owner)],
            positions: positions, textRoPEDelta: -6)
        return PreparedImage(
            prepared: prepared,
            lineage: QwenImageLineage(owners: [owner], textRoPEDelta: -6))
    }

    private func assertMetadataRejection(changedDigest: Bool) async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 32)

        // Establish a committed image turn. The next image must retain the
        // processor identity and profile of this committed lineage.
        let initialTransaction = try await state.begin()
        let initialOwner = try makeLineageOwner(rows: 1, context: context)
        let initialPrepared = try makePreparedSingleImage(owner: initialOwner)
        let initialReservation = try await state.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: initialOwner.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(initialPrepared.prepared),
            transaction: initialTransaction)
        try await state.prefillMultimodal(
            initialPrepared.prepared, lineage: initialPrepared.lineage,
            reservation: initialReservation, transaction: initialTransaction)
        _ = try await state.commit(transaction: initialTransaction)

        let transaction = try await state.begin()
        let changed: QwenRetainedFeatureOwner
        if changedDigest {
            changed = try makeLineageOwner(
                rows: 1, context: context,
                imageDigest: String(repeating: "d", count: 64),
                processorDigest: String(repeating: "c", count: 64))
        } else {
            changed = try makeLineageOwner(
                rows: 1, context: context,
                imageDigest: String(repeating: "d", count: 64),
                profile: GTurboQwenVisionProcessorProfileV2(
                    processorClass: "Qwen3VLProcessor",
                    imageProcessorType: "Qwen2VLImageProcessorFast",
                    patchSize: 8, temporalPatchSize: 2, spatialMergeSize: 2))
        }
        let candidate = try makePreparedSingleImage(owner: changed)
        let lineage = QwenImageLineage(owners: [changed])
        let reservation = try await state.reserveLineage(
            plannedRows: 1,
            requestedAllocationBytes: changed.requestedAllocationBytes,
            provenancePlan: try makeProvenancePlan(candidate.prepared),
            transaction: transaction)
        let before = try await state.diagnosticSnapshot()
        do {
            try await state.prefillMultimodal(
                candidate.prepared, lineage: lineage,
                reservation: reservation, transaction: transaction)
            Issue.record("changed committed owner metadata unexpectedly passed")
        } catch let error as QwenConversationLineageError {
            #expect(error == .invalidReservation)
        } catch {
            Issue.record("unexpected metadata error: \(error)")
        }
        let after = try await state.diagnosticSnapshot()
        #expect(after == before)
        #expect(after.lineage.visibleRows == 1)
        #expect(after.lineage.liveOwnerRows == 2)
        try await state.cancelLineage(reservation, transaction: transaction)
        _ = try await state.rollback(transaction: transaction)
    }

    private func makePreparedTwoImage(
        firstOwner: QwenRetainedFeatureOwner,
        secondOwner: QwenRetainedFeatureOwner
    ) throws -> PreparedImage {
        let tokenIDs: [Int32] = (1...18).map { Int32($0) }
        let plan = try QwenMultimodalPositions.make(
            tokenCount: tokenIDs.count,
            imageRanges: [1..<7, 11..<17],
            grids: [firstOwner.grid, secondOwner.grid],
            merge: 2, maximumRows: 630)
        let prepared = try QwenPreparedPrefill(
            tokenIDs: tokenIDs,
            featureOverrides: [
                QwenPreparedFeatureOverride(tokenRange: 1..<7, owner: firstOwner),
                QwenPreparedFeatureOverride(tokenRange: 11..<17, owner: secondOwner),
            ],
            positions: plan.positions,
            textRoPEDelta: plan.textRoPEDelta)
        return PreparedImage(
            prepared: prepared,
            lineage: QwenImageLineage(
                owners: [firstOwner, secondOwner],
                textRoPEDelta: plan.textRoPEDelta))
    }

    private func makePreparedSingleImage(
        owner: QwenRetainedFeatureOwner,
        positionOffset: Int = 0
    ) throws -> PreparedImage {
        let tokenIDs: [Int32] = [1, 2, 3, 4]
        let plan = try QwenMultimodalPositions.make(
            tokenCount: tokenIDs.count, imageRanges: [1..<2],
            grids: [owner.grid], merge: 2, maximumRows: 630)
        let positions = try plan.positions.map {
            try QwenMRoPEPosition(
                temporal: Int($0.temporal) + positionOffset,
                height: Int($0.height) + positionOffset,
                width: Int($0.width) + positionOffset)
        }
        let prepared = try QwenPreparedPrefill(
            tokenIDs: tokenIDs,
            featureOverrides: [QwenPreparedFeatureOverride(
                tokenRange: 1..<2, owner: owner)],
            positions: positions, textRoPEDelta: plan.textRoPEDelta)
        return PreparedImage(
            prepared: prepared,
            lineage: QwenImageLineage(
                owners: [owner], textRoPEDelta: plan.textRoPEDelta))
    }
}

private actor SubmissionCounter {
    private var submissions = 0

    func record() {
        submissions += 1
    }

    func reset() {
        submissions = 0
    }

    func count() -> Int {
        submissions
    }
}

private actor ReplayFailureSwitch {
    private var failingOperation: Int?
    private var startedOperations = 0

    func arm(failingOperation: Int) {
        self.failingOperation = failingOperation
        startedOperations = 0
    }

    func disarm() {
        failingOperation = nil
    }

    func startedOperationCount() -> Int {
        startedOperations
    }

    func beforeLayer(_ layer: Int) throws {
        guard layer == 0 else { return }
        startedOperations += 1
        if startedOperations == failingOperation {
            throw ReplayFailure.injected
        }
    }
}

private enum ReplayFailure: Error, Equatable, Sendable {
    case injected
}

private actor BoundedBoolOutcome {
    private var result: Bool?

    func record(_ result: Bool) {
        guard self.result == nil else { return }
        self.result = result
    }

    var value: Bool? { result }

    func wait(timeoutNanoseconds: UInt64) async -> Bool {
        var elapsed: UInt64 = 0
        while elapsed < timeoutNanoseconds {
            if result != nil { return true }
            do {
                try await Task.sleep(nanoseconds: 10_000_000)
            } catch {
                return false
            }
            elapsed += 10_000_000
        }
        return result != nil
    }
}

private actor VisionSubmissionGate {
    private var submitted = false
    private var released = false

    func holdFirstSubmission() async {
        guard !submitted else { return }
        submitted = true
        while !released {
            do {
                try await Task.sleep(nanoseconds: 10_000_000)
            } catch {
                return
            }
        }
    }

    func waitUntilSubmitted(timeoutNanoseconds: UInt64) async -> Bool {
        var elapsed: UInt64 = 0
        while elapsed < timeoutNanoseconds {
            if submitted { return true }
            do {
                try await Task.sleep(nanoseconds: 10_000_000)
            } catch {
                return false
            }
            elapsed += 10_000_000
        }
        return submitted
    }

    func release() {
        released = true
    }
}
