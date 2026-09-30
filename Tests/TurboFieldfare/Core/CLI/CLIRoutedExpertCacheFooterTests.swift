import Testing
import TurboFieldfare

@testable import TurboFieldfareCLICore

@Suite struct CLIRoutedExpertCacheFooterTests {
    @Test func footerReportsLifetimeScopeConfigurationAllocationAndCounts() {
        let summary = RoutedExpertCacheSummary(
            configuredSlots: 16,
            effectiveSlots: 12,
            policy: "lfu",
            allocatedBytes: 0,
            peakAllocatedBytes: 4_194_304,
            hits: 9,
            misses: 27)

        #expect(routedExpertCacheFooter(summary) == "\n[expert-cache scope=lifetime configured-slots=16 effective-slots=12 policy=lfu allocated-bytes=0 peak-allocated-bytes=4194304 hits=9 misses=27]\n")
    }
}
