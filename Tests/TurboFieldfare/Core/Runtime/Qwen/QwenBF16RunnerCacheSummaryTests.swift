import Metal
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

private struct CacheReadFailureForSummaryTest: Error {}
private struct CacheInitializationFailureForSummaryTest: Error {}

private func failAfterCacheReservation(
    _ model: QwenOfficialSourceModel
) throws {
    let reservation = try model.reserveExpertCache()
    try withExtendedLifetime(reservation) {
        throw CacheInitializationFailureForSummaryTest()
    }
}

@Suite(.serialized) struct QwenBF16RunnerCacheSummaryTests {
    @Test func reportsSuccessfulMappingsAndRetainsLifetimeCountsAcrossReset() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let runner = try model.makeRunner(
            context: context,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            maxContext: 3)
        let expectedCacheBytes = QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
            expertSlotCount: QwenBF16TextRunnerFixture.topK)

        let initial = await runner.routedExpertCacheSummary()
        #expect(initial.configuredSlots == QwenBF16TextRunnerFixture.topK)
        #expect(initial.effectiveSlots == QwenBF16TextRunnerFixture.topK)
        #expect(initial.policy == model.expertCachePolicy.rawValue)
        #expect(initial.allocatedBytes == 0)
        #expect(initial.peakAllocatedBytes == 0)
        #expect(initial.hits == 0)
        #expect(initial.misses == 0)

        await #expect(throws: QwenTextRunnerError.invalidPosition(expected: 0, actual: 1)) {
            try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[0], position: 1)
        }
        await #expect(throws: QwenTextRunnerError.invalidToken(id: -1)) {
            try await runner.produce(token: -1, position: 0)
        }
        let rejected = await runner.routedExpertCacheSummary()
        #expect(rejected.hits == 0)
        #expect(rejected.misses == 0)

        _ = try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[0], position: 0)
        let firstToken = await runner.routedExpertCacheSummary()
        #expect(firstToken.hits == 0)
        #expect(firstToken.misses == 2 * UInt64(QwenBF16TextRunnerFixture.topK))
        #expect(firstToken.allocatedBytes == expectedCacheBytes)
        #expect(firstToken.peakAllocatedBytes == expectedCacheBytes)
        let firstDiagnostics = await runner.cacheDiagnostics()
        #expect(firstDiagnostics.summary == firstToken)

        try await runner.resetConversation()
        _ = try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[0], position: 0)
        let repeatedToken = await runner.routedExpertCacheSummary()
        #expect(repeatedToken.hits == 2 * UInt64(QwenBF16TextRunnerFixture.topK))
        #expect(repeatedToken.misses == firstToken.misses)
        #expect(repeatedToken.allocatedBytes == expectedCacheBytes)
        #expect(repeatedToken.peakAllocatedBytes == expectedCacheBytes)

        try await runner.resetConversation()
        let reset = await runner.routedExpertCacheSummary()
        #expect(reset == repeatedToken)
    }

    @Test func failedProtectedMapDoesNotAddHitsOrMisses() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let runner = try QwenOfficialSourceRunner(
            model: model,
            maxContext: 3,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(
                beforeProtectedExpertRead: { _, _, _ in
                    throw CacheReadFailureForSummaryTest()
                }))
        let expectedLayerCacheBytes = QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
            expertSlotCount: QwenBF16TextRunnerFixture.topK, layerCount: 1)

        await #expect(throws: CacheReadFailureForSummaryTest.self) {
            try await runner.produce(
                token: QwenBF16TextRunnerFixture.inputTokens[0], position: 0)
        }
        let failed = await runner.routedExpertCacheSummary()
        #expect(failed.configuredSlots == QwenBF16TextRunnerFixture.topK)
        #expect(failed.allocatedBytes == expectedLayerCacheBytes)
        #expect(failed.peakAllocatedBytes == expectedLayerCacheBytes)
        #expect(failed.hits == 0)
        #expect(failed.misses == 0)
    }

    @Test func modelQuotaRejectsSecondRunnerAndReleasesAfterReservationScopeFailureAndDeinit() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())

        // This models a runner initialization failure after reservation. The
        // helper's scope ends during the throw, so ARC must return the full
        // reservation before the following runner is created.
        #expect(throws: CacheInitializationFailureForSummaryTest.self) {
            try failAfterCacheReservation(model)
        }

        var retained: QwenOfficialSourceRunner? = try model.makeRunner(
            context: context,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            maxContext: 3)
        _ = try await retained!.produce(
            token: QwenBF16TextRunnerFixture.inputTokens[0], position: 0)
        let retainedSummary = await retained!.routedExpertCacheSummary()
        #expect(retainedSummary.allocatedBytes
            == QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
                expertSlotCount: QwenBF16TextRunnerFixture.topK))
        #expect(throws: QwenBF16ExpertCacheError.budgetExceeded) {
            _ = try model.makeRunner(
                context: context,
                expertSlotCount: QwenBF16TextRunnerFixture.topK,
                maxContext: 3)
        }

        retained = nil
        _ = try model.makeRunner(
            context: context,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            maxContext: 3)
    }
}
