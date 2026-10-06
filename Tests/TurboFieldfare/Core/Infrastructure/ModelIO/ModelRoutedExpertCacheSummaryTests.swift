import Foundation
import Metal
import Testing

@testable import TurboFieldfare

@Suite struct ModelRoutedExpertCacheSummaryTests {
    @Test func reportsSlotsBytesSuccessfulFetchesAndLifetimeAcrossRelease() async throws {
        let directory = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = try Model.load(
            directoryURL: directory,
            device: try #require(MTLCreateSystemDefaultDevice()),
            expecting: .gemma4Toy(),
            streamingMode: .pread(slotCount: 2))

        let initial = model.routedExpertCacheSummary
        #expect(initial.configuredSlots == 2)
        #expect(initial.effectiveSlots == 2)
        #expect(initial.policy == model.expertCachePolicy.rawValue)
        #expect(initial.allocatedBytes == 0)
        #expect(initial.peakAllocatedBytes == 0)
        #expect(initial.hits == 0)
        #expect(initial.misses == 0)

        let coldPlan = try model.planRoutedExperts(layer: 1, experts: [4, 5])
        let cold = try #require(coldPlan)
        _ = try model.routedExpertBuffers(for: cold)
        let planned = model.routedExpertCacheSummary
        #expect(planned.hits == 0)
        #expect(planned.misses == 0)
        #expect(planned.allocatedBytes > 0)
        #expect(planned.peakAllocatedBytes == planned.allocatedBytes)
        _ = try await model.fetchRoutedExperts(plan: cold)

        let allHitPlan = try model.planRoutedExperts(layer: 1, experts: [4, 5])
        let allHit = try #require(allHitPlan)
        #expect(allHit.hits == 2)
        #expect(allHit.misses.isEmpty)
        _ = try model.routedExpertBuffers(for: allHit)
        let previewed = model.routedExpertCacheSummary
        #expect(previewed.hits == 0)
        #expect(previewed.misses == 2)
        _ = try await model.fetchRoutedExperts(plan: allHit)

        let beforeRelease = model.routedExpertCacheSummary
        #expect(beforeRelease.hits == 2)
        #expect(beforeRelease.misses == 2)
        #expect(beforeRelease.allocatedBytes > 0)
        let released = model.prepareExpertResidencyForVision(.onDemand)
        #expect(released.releasedLayerCount == 1)

        let afterRelease = model.routedExpertCacheSummary
        #expect(afterRelease.configuredSlots == beforeRelease.configuredSlots)
        #expect(afterRelease.effectiveSlots == beforeRelease.effectiveSlots)
        #expect(afterRelease.allocatedBytes == 0)
        #expect(afterRelease.peakAllocatedBytes == beforeRelease.peakAllocatedBytes)
        #expect(afterRelease.hits == 2)
        #expect(afterRelease.misses == 2)

        _ = try await model.fetchRoutedExperts(layer: 1, experts: [4, 5])
        let reopened = model.routedExpertCacheSummary
        #expect(reopened.allocatedBytes > 0)
        #expect(reopened.peakAllocatedBytes == beforeRelease.peakAllocatedBytes)
        #expect(reopened.hits == 2)
        #expect(reopened.misses == 4)
    }
}
