import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct ConversationStateTransactionTests {
    private let prompt: [Int32] = [1, 4, 7]

    @Test func transactionPublishesOneCommittedAggregateAndRetainsProgressBoundary() async throws {
        let transaction = try await makeTransaction()
        let id = try await transaction.begin()
        try await transaction.prefill(prompt, transaction: id) { _, _ in }
        try await transaction.advance(2, transaction: id)

        let active = await transaction.status()
        let working = try #require(active.working)
        #expect(active.activeTransaction == id)
        #expect(working.retainedTokenIDs == prompt + [2])
        #expect(working.consumedTokenCount == prompt.count)
        #expect(working.pendingTokenCount == 1)
        #expect(working.logicalStateBytes > 0)

        let committed = try await transaction.commit(transaction: id)
        #expect(committed == working)
        let settled = await transaction.status()
        #expect(settled.committed == committed)
        #expect(settled.working == nil)
        #expect(settled.activeTransaction == nil)
    }

    @Test func transactionRollbackRestoresThePriorAggregateAfterInvalidInput() async throws {
        let transaction = try await makeTransaction()
        let id = try await transaction.begin()
        try await transaction.prefill(prompt, transaction: id) { _, _ in }
        let before = await transaction.status()

        await #expect(throws: QwenTextRunnerError.invalidToken(id: 19)) {
            try await transaction.advance(19, transaction: id)
        }
        #expect(await transaction.status() == before)
        try await transaction.rollback(transaction: id)
        let after = await transaction.status()
        #expect(after.committed.retainedTokenIDs.isEmpty)
        #expect(after.working == nil)
        #expect(after.activeTransaction == nil)
    }

    @Test func staleAndOverlappingHandlesCannotPublishOrClobberTheCommittedState() async throws {
        let transaction = try await makeTransaction()
        let first = try await transaction.begin()
        await #expect(throws: ConversationStateTransactionError.busy) {
            _ = try await transaction.begin()
        }
        try await transaction.prefill([1], transaction: first) { _, _ in }
        let committed = try await transaction.commit(transaction: first)

        await #expect(throws: ConversationStateTransactionError.staleTransaction) {
            try await transaction.advance(2, transaction: first)
        }
        #expect(await transaction.status().committed == committed)

        try await transaction.reset()
        #expect((await transaction.status()).committed.retainedTokenIDs.isEmpty)
        await #expect(throws: ConversationStateTransactionError.staleTransaction) {
            try await transaction.rollback(transaction: first)
        }
    }

    @Test func resetRejectsAnActiveTransactionAndThenClearsTheCommittedLineage() async throws {
        let transaction = try await makeTransaction()
        let id = try await transaction.begin()
        try await transaction.prefill(prompt, transaction: id) { _, _ in }
        await #expect(throws: ConversationStateTransactionError.busy) {
            try await transaction.reset()
        }
        try await transaction.rollback(transaction: id)
        try await transaction.reset()

        let status = await transaction.status()
        #expect(status.committed.retainedTokenIDs.isEmpty)
        #expect(status.committed.consumedTokenCount == 0)
        #expect(status.committed.pendingTokenCount == 0)
        #expect(status.working == nil)
    }

    @Test func checkpointRebuildReportsTheSameLogicalMetricsAsAReplayFromZero() async throws {
        let transaction = try await makeTransaction()
        let initial = try await transaction.begin()
        try await transaction.prefill(prompt, transaction: initial) { _, _ in }
        let original = try await transaction.commit(transaction: initial)

        let checkpoint = try await transaction.begin()
        try await transaction.rebuildCheckpoint(
            retaining: original.retainedTokenIDs,
            transaction: checkpoint) { _, _ in }
        let rebuilt = try await transaction.commit(transaction: checkpoint)
        let rebuiltSnapshot = try await transaction.diagnosticSnapshot()

        let cleanState = try await makeTransaction()
        let clean = try await cleanState.begin()
        try await cleanState.prefill(original.retainedTokenIDs, transaction: clean) { _, _ in }
        _ = try await cleanState.commit(transaction: clean)
        let expectedSnapshot = try await cleanState.diagnosticSnapshot()

        #expect(rebuilt.retainedTokenIDs == original.retainedTokenIDs)
        #expect(rebuilt.consumedTokenCount == original.consumedTokenCount)
        #expect(rebuilt.pendingTokenCount == original.pendingTokenCount)
        #expect(rebuilt.logicalStateBytes == original.logicalStateBytes)
        #expect(rebuiltSnapshot.currentLogits != nil)
        assertDiagnosticStateClose(
            rebuiltSnapshot, expectedSnapshot,
            label: "successful no-pending checkpoint", compareReplayMetadata: false)
        #expect(rebuiltSnapshot.replayGeneration == expectedSnapshot.replayGeneration + 1)
        #expect(rebuiltSnapshot.producerEpoch == expectedSnapshot.producerEpoch)
    }

    private func makeTransaction() async throws -> QwenConversationState {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        return try await QwenConversationState(
            model: model, context: context, maxContext: 32)
    }
}
