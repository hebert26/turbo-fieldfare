import Metal
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

private struct CacheReadFailureForSummaryTest: Error {}

@Suite(.serialized) struct QwenBF16RunnerCacheSummaryTests {
    @Test func reportsSuccessfulMappingsAndRetainsLifetimeCountsAcrossReset() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.expectedResidentBytes)
        let runner = try model.makeRunner(
            context: context,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            maxContext: 3)

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
        #expect(firstToken.allocatedBytes == 0)
        #expect(firstToken.peakAllocatedBytes == 768)
        let firstDiagnostics = await runner.cacheDiagnostics()
        #expect(firstDiagnostics.summary == firstToken)

        _ = try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[1], position: 1)
        let secondToken = await runner.routedExpertCacheSummary()
        // The source runtime creates a temporary coordinator per layer and
        // token, so each routed expert fetch is cold by design.
        #expect(secondToken.hits == 0)
        #expect(secondToken.misses == 4 * UInt64(QwenBF16TextRunnerFixture.topK))
        #expect(secondToken.allocatedBytes == 0)
        #expect(secondToken.peakAllocatedBytes == 768)

        try await runner.resetConversation()
        let reset = await runner.routedExpertCacheSummary()
        #expect(reset.hits == secondToken.hits)
        #expect(reset.misses == secondToken.misses)
        #expect(reset.peakAllocatedBytes == secondToken.peakAllocatedBytes)
    }

    @Test func failedProtectedMapDoesNotAddHitsOrMisses() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.expectedResidentBytes)
        let runner = try QwenOfficialSourceRunner(
            model: model,
            maxContext: 3,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(
                beforeProtectedExpertRead: { _, _, _ in
                    throw CacheReadFailureForSummaryTest()
                }))

        await #expect(throws: CacheReadFailureForSummaryTest.self) {
            try await runner.produce(
                token: QwenBF16TextRunnerFixture.inputTokens[0], position: 0)
        }
        let failed = await runner.routedExpertCacheSummary()
        #expect(failed.configuredSlots == QwenBF16TextRunnerFixture.topK)
        #expect(failed.allocatedBytes == 0)
        #expect(failed.peakAllocatedBytes == 768)
        #expect(failed.hits == 0)
        #expect(failed.misses == 0)
    }
}
