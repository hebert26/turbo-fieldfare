import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenFullAttentionKVTests {
    @Test func allocatesOnlyTheTenManifestSelectedLayersWithDistinctKV() throws {
        let cache = try makeCache()
        #expect(cache.fullLayerIndices == stride(from: 3, through: 39, by: 4).map { $0 })
        #expect(cache.fullLayerCount == 10)
        let first = try cache.view(layer: 3)
        let second = try cache.view(layer: 7)
        #expect(first.key !== first.value)
        #expect(first.key !== second.key)
        #expect(first.value !== second.value)
        #expect(throws: QwenFullAttentionKVError.notFullAttentionLayer(0)) { _ = try cache.view(layer: 0) }
    }

    @Test func refusesOverflowOutOfOrderAndEarlyReuseBeforePublishing() throws {
        let cache = try makeCache(maxContext: 3)
        #expect(throws: QwenFullAttentionKVError.outOfOrderWrite(layer: 3, expected: 0, actual: 1)) {
            _ = try cache.reserveWrite(layer: 3, position: 1, tokenCount: 1)
        }
        #expect(throws: QwenFullAttentionKVError.capacityExceeded(start: 0, count: 4, capacity: 3)) {
            _ = try cache.reserveWrite(layer: 3, position: 0, tokenCount: 4)
        }
        let write = try cache.reserveWrite(layer: 3, position: 0, tokenCount: 1)
        #expect(throws: QwenFullAttentionKVError.writeAlreadyPending(layer: 3)) {
            _ = try cache.reserveWrite(layer: 3, position: 0, tokenCount: 1)
        }
        try cache.abandon(write)
        #expect(try cache.committedPosition(layer: 3) == 0)
    }

    @Test func writeAndReadLeasesAreOneShotAndHoldStateUntilCompletion() throws {
        let cache = try makeCache()
        let queue = try #require(device().makeCommandQueue())
        let write = try cache.reserveWrite(layer: 3, position: 0, tokenCount: 1)
        let writeBuffer = try #require(queue.makeCommandBuffer())
        try cache.trackLastUse(of: write, on: writeBuffer)
        #expect(throws: QwenFullAttentionKVError.writeAlreadySubmitted) { try cache.trackLastUse(of: write, on: writeBuffer) }
        #expect(throws: QwenFullAttentionKVError.writeAlreadySubmitted) { try cache.abandon(write) }
        writeBuffer.commit(); writeBuffer.waitUntilCompleted()
        #expect(writeBuffer.status == .completed)
        #expect(try cache.committedPosition(layer: 3) == 1)
        for layer in cache.fullLayerIndices.dropFirst() {
            try commit(try cache.reserveWrite(layer: layer, position: 0, tokenCount: 1), cache: cache, queue: queue)
        }

        let read = try cache.reserveRead(layers: [3])
        let readBuffer = try #require(queue.makeCommandBuffer())
        try cache.trackLastUse(of: read, on: readBuffer)
        #expect(throws: QwenFullAttentionKVError.readAlreadySubmitted) { try cache.trackLastUse(of: read, on: readBuffer) }
        #expect(throws: QwenFullAttentionKVError.readAlreadySubmitted) { try cache.abandon(read) }
        #expect(throws: QwenFullAttentionKVError.activeGPUUse) { try cache.snapshot() }
        readBuffer.commit(); readBuffer.waitUntilCompleted()
        _ = try cache.snapshot()
    }

    @Test func splitWritesSnapshotRestoreAndSuffixOverwriteInvalidateFutureSnapshot() throws {
        let cache = try makeCache(maxContext: 5)
        let queue = try #require(device().makeCommandQueue())
        try commitAllLayers(cache: cache, position: 0, tokenCount: 1, queue: queue)
        let early = try cache.snapshot()
        try commitAllLayers(cache: cache, position: 1, tokenCount: 2, queue: queue)
        let future = try cache.snapshot()
        #expect(future.position == 3)
        try cache.restore(early)
        #expect(try cache.committedPosition(layer: 3) == 1)
        try commitAllLayers(cache: cache, position: 1, tokenCount: 1, queue: queue)
        #expect(throws: QwenFullAttentionKVError.invalidSnapshotLineage) { try cache.restore(future) }
        try cache.reset()
        #expect(throws: QwenFullAttentionKVError.invalidSnapshotLineage) { try cache.restore(early) }
    }

    @Test func pendingWriteBlocksSnapshotResetAndRestoreAndFailedReservationDoesNotPublish() throws {
        let cache = try makeCache()
        let snapshot = try cache.snapshot()
        let pending = try cache.reserveWrite(layer: 3, position: 0, tokenCount: 1)
        #expect(throws: QwenFullAttentionKVError.activeGPUUse) { _ = try cache.snapshot() }
        #expect(throws: QwenFullAttentionKVError.activeGPUUse) { try cache.restore(snapshot) }
        #expect(throws: QwenFullAttentionKVError.activeGPUUse) { try cache.reset() }
        try cache.abandon(pending)
        #expect(try cache.committedPosition(layer: 3) == 0)
    }

    @Test func abandoningWriteCancelsPublicationAndReleasesTheSameRange() throws {
        let cache = try makeCache()
        let before = try cache.snapshot()
        let pending = try cache.reserveWrite(layer: 3, position: 0, tokenCount: 1)
        pending.key.contents().assumingMemoryBound(to: Float.self)[pending.keyOffset / MemoryLayout<Float>.stride] = 7
        pending.value.contents().assumingMemoryBound(to: Float.self)[pending.valueOffset / MemoryLayout<Float>.stride] = -3
        try cache.abandon(pending)

        let after = try cache.snapshot()
        #expect(after == before)
        #expect(try cache.committedPosition(layer: 3) == before.position)

        // A cancelled reservation releases admission and permits the exact same
        // append range to be reserved again without publishing stale bytes.
        let reacquired = try cache.reserveWrite(layer: 3, position: before.position, tokenCount: 1)
        try cache.abandon(reacquired)
        #expect(try cache.view(layer: 3).validTokenCount == before.position)
    }

    @Test func replacementCheckpointRestoresOverwrittenPrefixBytesAndCursors() throws {
        let cache = try makeCache(maxContext: 4)
        let queue = try #require(device().makeCommandQueue())

        for (layerIndex, layer) in cache.fullLayerIndices.enumerated() {
            let write = try cache.reserveWrite(layer: layer, position: 0, tokenCount: 2)
            fill(write, keyBase: Float(layerIndex + 1), valueBase: -Float(layerIndex + 1))
            try commit(write, cache: cache, queue: queue)
        }
        let original = try cache.fullLayerIndices.map { layer in
            try kvBytes(cache: cache, layer: layer)
        }
        let replacement = try cache.retainReplacementCheckpoint()

        try cache.reset()
        for (layerIndex, layer) in cache.fullLayerIndices.enumerated() {
            let write = try cache.reserveWrite(layer: layer, position: 0, tokenCount: 2)
            fill(write, keyBase: Float(100 + layerIndex), valueBase: Float(-100 - layerIndex))
            try commit(write, cache: cache, queue: queue)
        }
        #expect(try cache.fullLayerIndices.map { layer in
            try kvBytes(cache: cache, layer: layer)
        } != original)

        try cache.restoreReplacementCheckpoint(replacement)
        #expect(cache.fullLayerIndices.allSatisfy { layer in
            (try? cache.committedPosition(layer: layer)) == 2
        })
        #expect(try cache.fullLayerIndices.map { layer in
            try kvBytes(cache: cache, layer: layer)
        } == original)
        cache.discardReplacementCheckpoint()
    }
}

private func device() -> MTLDevice { MTLCreateSystemDefaultDevice()! }
private func makeCache(maxContext: Int = 8) throws -> QwenFullAttentionKV {
    try QwenFullAttentionKV(device: device(), fullAttentionLayerMask: (0..<40).map { $0.isMultiple(of: 4) ? 0 : (($0 + 1).isMultiple(of: 4) ? 1 : 0) }, maxContext: maxContext, keyValueHeadCount: 2, headDimension: 4)
}
private func commit(_ write: QwenFullAttentionKVWrite, cache: QwenFullAttentionKV, queue: MTLCommandQueue) throws {
    let commandBuffer = try #require(queue.makeCommandBuffer())
    try cache.trackLastUse(of: write, on: commandBuffer)
    commandBuffer.commit(); commandBuffer.waitUntilCompleted()
    #expect(commandBuffer.status == .completed)
}
private func commitAllLayers(cache: QwenFullAttentionKV, position: Int, tokenCount: Int, queue: MTLCommandQueue) throws {
    for layer in cache.fullLayerIndices {
        try commit(try cache.reserveWrite(layer: layer, position: position, tokenCount: tokenCount), cache: cache, queue: queue)
    }
}

private func fill(_ write: QwenFullAttentionKVWrite, keyBase: Float, valueBase: Float) {
    let elementCount = write.tokenCount * write.strideBytes / MemoryLayout<Float>.stride
    let key = write.key.contents().advanced(by: write.keyOffset)
        .assumingMemoryBound(to: Float.self)
    let value = write.value.contents().advanced(by: write.valueOffset)
        .assumingMemoryBound(to: Float.self)
    for index in 0..<elementCount {
        key[index] = keyBase + Float(index) * 0.25
        value[index] = valueBase - Float(index) * 0.5
    }
}

private struct KVBytes: Equatable {
    let key: Data
    let value: Data
}

private func kvBytes(cache: QwenFullAttentionKV, layer: Int) throws -> KVBytes {
    let view = try cache.view(layer: layer)
    let byteCount = view.validTokenCount * view.strideBytes
    return KVBytes(
        key: Data(bytes: view.key.contents().advanced(by: view.keyOffset), count: byteCount),
        value: Data(bytes: view.value.contents().advanced(by: view.valueOffset), count: byteCount))
}
