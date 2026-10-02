import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized)
struct QwenBF16CacheResidencyTests {
    @Test func registersOneSetOnlyWhenAllLayersExistAndRejectsInvalidRegistrations() throws {
        let context = try MetalContext()
        let owner = try QwenBF16CacheResidency(queue: context.queue, layerCount: 2, slotCount: 2)
        let first = try allocations(context.device, count: 4)
        let second = try allocations(context.device, count: 4)
        let empty = owner.snapshot()
        #expect(empty.allocationCount == 0 && empty.expectedAllocationCount == 8)
        #expect(!empty.requested && empty.registeredLayerCount == 0)
        try owner.register(layer: 0, allocations: first)
        let partial = owner.snapshot()
        #expect(partial.allocationCount == 4 && partial.registeredLayerCount == 1)
        #expect(!partial.requested)
        #expect(throws: QwenBF16CacheResidencyError.duplicateLayer(0)) {
            try owner.register(layer: 0, allocations: first)
        }
        #expect(throws: QwenBF16CacheResidencyError.invalidLayer(-1)) {
            try owner.register(layer: -1, allocations: second)
        }
        #expect(throws: QwenBF16CacheResidencyError.invalidLayer(2)) {
            try owner.register(layer: 2, allocations: second)
        }
        #expect(throws: QwenBF16CacheResidencyError.invalidAllocations) {
            try owner.register(layer: 1, allocations: Array(second.prefix(3)))
        }
        #expect(throws: QwenBF16CacheResidencyError.invalidAllocations) {
            try owner.register(layer: 1, allocations: [second[0], second[0], second[2], second[3]])
        }
        #expect(throws: QwenBF16CacheResidencyError.invalidAllocations) {
            try owner.register(layer: 1, allocations: first)
        }
        #expect(owner.snapshot() == partial, "invalid registration must not mutate the pending set")
        try owner.register(layer: 1, allocations: second)
        let complete = owner.snapshot()
        #expect(complete.requested && complete.registeredLayerCount == 2)
        #expect(complete.allocationCount == complete.expectedAllocationCount)
        #expect(complete.allocatedSize >= UInt64(8 * 256))
        #expect(throws: QwenBF16CacheResidencyError.duplicateLayer(1)) {
            try owner.register(layer: 1, allocations: second)
        }
        #expect(owner.snapshot() == complete)
    }

    @Test func rejectsCheckedCountOverflowAndPrivateAllocations() throws {
        let context = try MetalContext()
        for (layers, slots) in [(0, 1), (1, 0), (-1, 2), (1, Int.max), (Int.max, 2)] {
            #expect(throws: QwenBF16CacheResidencyError.invalidGeometry) {
                _ = try QwenBF16CacheResidency(queue: context.queue, layerCount: layers, slotCount: slots)
            }
        }
        let owner = try QwenBF16CacheResidency(queue: context.queue, layerCount: 1, slotCount: 1)
        let shared = try #require(context.device.makeBuffer(length: 256, options: .storageModeShared))
        let privateBuffer = try #require(context.device.makeBuffer(length: 256, options: .storageModePrivate))
        #expect(throws: QwenBF16CacheResidencyError.invalidAllocations) {
            try owner.register(layer: 0, allocations: [shared, privateBuffer])
        }
        #expect(owner.snapshot().allocationCount == 0)
        #expect(!owner.snapshot().requested)
    }

    @Test func actualCoordinatorSlotsAreRegisteredAndLeaseRetainsOwner() async throws {
        let source = try cacheSource()
        defer { source.remove() }
        let context = try MetalContext()
        let config = try QwenMoEConfiguration(hiddenSize: 2, expertCount: 8,
            routedIntermediateSize: 3, sharedIntermediateSize: 1)
        var owner: QwenBF16CacheResidency? = try QwenBF16CacheResidency(
            queue: context.queue, layerCount: 1, slotCount: 2)
        weak var weakOwner = owner
        var coordinator: QwenBF16ExpertMappingCoordinator? = try QwenBF16ExpertMappingCoordinator(
            source: source.handle, names: cacheNames(source), layer: 0, configuration: config,
            device: context.device, slotCount: 2, residencyBudget: 72, cacheResidency: owner)
        #expect(owner?.snapshot().allocationCount == 4)
        #expect(owner?.snapshot().requested == true)
        var lease: QwenBF16ExpertLease? = try await coordinator!.map(expertIDs: [7])
        #expect(lease!.diagnostics.hits == 0 && lease!.diagnostics.misses == 1)
        let firstIDs = lease!.experts.map { [ObjectIdentifier($0.gateUp), ObjectIdentifier($0.down)] }
        try lease!.cancel()
        lease = nil
        let hit = try await coordinator!.map(expertIDs: [7])
        #expect(hit.diagnostics.hits == 1 && hit.diagnostics.misses == 0)
        #expect(hit.experts.map { [ObjectIdentifier($0.gateUp), ObjectIdentifier($0.down)] } == firstIDs)
        #expect(owner?.snapshot().allocationCount == 4)
        // A real lease owns the coordinator even when its original references go.
        coordinator = nil
        owner = nil
        #expect(weakOwner != nil)
        let destination = try #require(context.device.makeBuffer(length: 24, options: .storageModeShared))
        destination.contents().initializeMemory(as: UInt8.self, repeating: 0xff, count: 24)
        let command = try hit.submit(on: context.queue) { command in
            let blit = try #require(command.makeBlitCommandEncoder())
            // Read cached bytes into an independent output without changing cache.
            blit.copy(from: hit.experts[0].gateUp, sourceOffset: 0, to: destination, destinationOffset: 0, size: 24)
            blit.endEncoding()
        }
        _ = await command.completed()
        #expect(command.status == .completed && command.error == nil)
        #expect(hit.snapshot().succeeded == true)
        let copied = Array(UnsafeBufferPointer(
            start: destination.contents().assumingMemoryBound(to: UInt16.self), count: 12))
        #expect(copied == [UInt16](repeating: 0x3f80, count: 12))
        // The canceled first lease is nil. Only the live hit/command can retain
        // its coordinator here; there is no auxiliary coordinator closure.
        withExtendedLifetime(hit) { #expect(weakOwner != nil) }
        // Completed callback captures may live until command release; no extra
        // wait or immediate post-await weak-deallocation claim is introduced.
    }

    @Test func enabledRunnerMatchesDisabledOutputAndCacheAccounting() async throws {
        let off = try await runSyntheticRunner(false)
        let on = try await runSyntheticRunner(true)
        #expect(off.bits == on.bits)
        #expect(off.summary == on.summary)
        #expect(on.summary.allocatedBytes == QwenBF16TextRunnerFixture.expectedExpertCacheBytes(expertSlotCount: 8))
        #expect(on.summary.hits > 0 && on.summary.misses > 0)
    }

    @Test func ownerHasNoQueueCycleWithCompleteOrIncompleteSet() throws {
        let context = try MetalContext()
        for complete in [false, true] {
            var owner: QwenBF16CacheResidency? = try QwenBF16CacheResidency(
                queue: context.queue, layerCount: complete ? 1 : 2, slotCount: 1)
            weak var weakOwner = owner
            try owner!.register(layer: 0, allocations: allocations(context.device, count: 2))
            #expect(owner!.snapshot().requested == complete)
            owner = nil
            #expect(weakOwner == nil, "queue/set must not retain the owner")
        }
    }
}

private func allocations(_ device: MTLDevice, count: Int) throws -> [MTLBuffer] {
    try (0..<count).map { _ in try #require(device.makeBuffer(length: 256, options: .storageModeShared)) }
}

private func cacheSource() throws -> QwenBF16ExpertCacheSourceFixture {
    try QwenBF16ExpertCacheSourceFixture.make(
        firstShard: [QwenBF16ExpertCacheLiteralTensor(name: "residency.gate", shape: [8, 6, 2], words: [UInt16](repeating: 0x3f80, count: 96))],
        secondShard: [QwenBF16ExpertCacheLiteralTensor(name: "residency.down", shape: [8, 2, 3], words: [UInt16](repeating: 0x3f00, count: 48))])
}

private func cacheNames(_ source: QwenBF16ExpertCacheSourceFixture) -> QwenBF16RoutedSourceNames {
    QwenBF16RoutedSourceNames(gateUpShardName: source.gateUpShardName, gateUpTensorName: "residency.gate",
                            downShardName: source.downShardName, downTensorName: "residency.down")
}

private func runSyntheticRunner(_ enabled: Bool) async throws -> (bits: [[UInt32]], summary: RoutedExpertCacheSummary) {
    let source = try QwenBF16TextRunnerFixture.make()
    defer { source.remove() }
    let context = try MetalContext()
    let model = try QwenOfficialSourceModel.loadSyntheticFixture(registrationURL: source.registrationURL,
        context: context, residencyBudgetBytes: source.totalResidencyBudget())
    let runner = try QwenOfficialSourceRunner(model: model, maxContext: 3, expertSlotCount: 8,
                                            useExpertCacheResidency: enabled)
    var output: [[UInt32]] = []
    _ = try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[0], position: 0)
    // Reset retains the existing cache; second token attempt exercises hits.
    try await runner.resetConversation()
    let logits = try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[0], position: 0)
    output.append(logits.map(\.bitPattern))
    return (output, await runner.routedExpertCacheSummary())
}
