import Darwin
import Foundation
import Metal
import Testing

@testable import TurboFieldfare

extension PreadExpertStreamerTests {
    @Test func successfulPlanCountsIncludeAllHitsButExcludePreviewsAndFailedBatches() throws {
        let url = try Self.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let streamer = try PreadExpertStreamer(
            layout: Self.makeLayout(path: url.path),
            device: try MetalContext().device,
            slotCount: 2)

        let coldPlan = streamer.planExpertsCached(experts: [0])
        _ = streamer.expertCachePlanBuffers(coldPlan)
        #expect(streamer.successfulCachePlanCounts.hits == 0)
        #expect(streamer.successfulCachePlanCounts.misses == 0)

        _ = try streamer.executeExpertCachePlan(coldPlan)
        #expect(streamer.successfulCachePlanCounts.hits == 0)
        #expect(streamer.successfulCachePlanCounts.misses == 1)

        let allHitPlan = streamer.planExpertsCached(experts: [0])
        #expect(allHitPlan.hits == 1)
        #expect(allHitPlan.misses.isEmpty)
        _ = streamer.expertCachePlanBuffers(allHitPlan)
        #expect(streamer.successfulCachePlanCounts.hits == 0)
        _ = try streamer.executeExpertCachePlan(allHitPlan)
        #expect(streamer.successfulCachePlanCounts.hits == 1)
        #expect(streamer.successfulCachePlanCounts.misses == 1)

        // Expert 0 is resident, while expert 1 becomes an unreadable miss.
        // The mixed plan must not report its hit or partial miss as completed.
        try Data().write(to: url)
        let failedPlan = streamer.planExpertsCached(experts: [0, 1])
        #expect(failedPlan.hits == 1)
        #expect(failedPlan.misses.count == 1)
        #expect(throws: Error.self) {
            _ = try streamer.executeExpertCachePlan(failedPlan)
        }
        #expect(streamer.successfulCachePlanCounts.hits == 1)
        #expect(streamer.successfulCachePlanCounts.misses == 1)
    }
}
