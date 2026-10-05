import Foundation
import Darwin
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct GemmaVerifyCacheTests {
    @Test func heapCacheKeepsLoadedBytesAfterStreamerRelease() throws {
        let context = try MetalContext()
        let stride = Int(getpagesize())
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("heap-cache-\(UUID().uuidString)")
        var data = Data(repeating: 0, count: stride * 128)
        data.withUnsafeMutableBytes { bytes in
            for expert in 0..<128 { bytes.storeBytes(of: UInt32(expert + 1000), toByteOffset: expert * stride, as: UInt32.self) }
        }
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let layout = StreamLayout(path: url.path, streamOffset: 0, streamSize: UInt64(data.count),
                                  expertsPerLayer: 128, expertStride: UInt64(stride))
        var streamer: PreadExpertStreamer? = try PreadExpertStreamer(layout: layout,
            device: context.device, slotCount: 64)
        let loaded = try streamer!.loadExpertsCached(experts: Array(0..<8))
        var lease = try #require(streamer?.beginGPURead()) as PreadExpertStreamer.GPUReadLease?
        if context.device.hasUnifiedMemory && ProcessInfo.processInfo.physicalMemory >= 32 * 1_024 * 1_024 * 1_024 {
            #expect(lease?.heap != nil)
        }
        #expect(lease?.populatedBuffers.count == 8)
        lease?.release()
        lease = nil
        streamer = nil
        let output = try #require(context.device.makeBuffer(length: 32, options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        let copy = try #require(command.makeBlitCommandEncoder())
        for (index, value) in loaded.enumerated() {
            copy.copy(from: value.buffer, sourceOffset: 0, to: output, destinationOffset: index * 4, size: 4)
        }
        copy.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try checkCommandBufferError(command)
        let values = output.contents().assumingMemoryBound(to: UInt32.self)
        for index in 0..<8 { #expect(values[index] == UInt32(index + 1000)) }
    }

    @Test(arguments: [-1, 0, 3, GPUExpertCache.maxLayers - 1])
    func secondTokenMissStopsLaterWork(missingLayer: Int) throws {
        let context = try MetalContext()
        let routed = try GemmaVerifyMoE(context: context)
        let cache = try GemmaVerifyCache(context: context, selectedLength: 64, groupedLength: routed.argumentLength)
        let rms = try RMSNorm(context: context)
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try #require(context.device.makeBuffer(bytes: values,
                length: values.count * MemoryLayout<T>.stride, options: .storageModeShared))
        }
        let input = try buffer([Float16](repeating: 1, count: 64))
        let weight = try buffer([UInt16](repeating: 0x3f80, count: 64))
        let blobs = try (0..<16).map { try buffer([UInt32($0)]) }
        let routes: [[UInt32]] = [Array(0..<8), [0, 1, 2, 3, 8, 9, 10, 11]]
        let indices = try routes.map { try buffer($0) }
        for missing in [missingLayer, -1] {
            cache.prepare([])
            let command = try #require(context.queue.makeCommandBuffer())
            var prefixes: [MTLBuffer] = []
            var tails: [MTLBuffer] = []
            for index in cache.layers.indices {
                let layer = cache.layers[index]
                layer.dispatch.reset()
                let addresses = layer.base.slots.contents().assumingMemoryBound(to: UInt64.self)
                let mapping = layer.base.slotByExpert.contents().assumingMemoryBound(to: UInt32.self)
                for expert in 0..<128 { mapping[expert] = .max }
                for slot in 0..<16 {
                    addresses[slot] = blobs[slot].gpuAddress
                    mapping[slot] = UInt32(slot)
                }
                if index == missing { mapping[11] = .max }
                layer.routes.contents().initializeMemory(as: UInt32.self, repeating: .max, count: 16)
                let prefix = try buffer([Float16](repeating: -7, count: 64))
                let tail = try buffer([Float16](repeating: -7, count: 64))
                prefixes.append(prefix)
                tails.append(tail)
                for _ in 0..<GemmaVerifyCache.prefixDispatches {
                    rms.encodeBF16W(commandBuffer: command, x: input, weight: weight,
                                    out: prefix, d: 64, eps: 1e-6, conditional: layer.dispatch)
                }
                try cache.encodeLookup(command: command, index: index, slotCount: 16, indices: indices)
                while layer.dispatch.count < GemmaVerifyCache.layerDispatches {
                    rms.encodeBF16W(commandBuffer: command, x: input, weight: weight,
                                    out: tail, d: 64, eps: 1e-6, conditional: layer.dispatch)
                }
            }
            command.commit()
            command.waitUntilCompleted()
            try checkCommandBufferError(command)
            #expect(cache.firstMissingLayer == (missing < 0 ? nil : missing))
            for index in cache.layers.indices {
                let prefixRan = missing < 0 || index <= missing
                let tailRan = missing < 0 || index < missing
                #expect(prefixes[index].contents().load(as: Float16.self) == (prefixRan ? 1 : -7))
                #expect(tails[index].contents().load(as: Float16.self) == (tailRan ? 1 : -7))
                let layer = cache.layers[index]
                for token in 0..<2 {
                    let selected = (token == 0 ? layer.base.selected : layer.secondSelected)
                        .contents().assumingMemoryBound(to: UInt64.self)
                    for slot in 0..<8 {
                        let expert = Int(routes[token][slot])
                        #expect(layer.completedRoutes[token][slot] == (prefixRan ? expert : Int(UInt32.max)))
                        if tailRan { #expect(selected[slot] == blobs[expert].gpuAddress) }
                    }
                }
            }
        }
    }

    @Test func hitGroupsRecordOnceAndRejectPartialUpdates() throws {
        let context = try MetalContext()
        let stride = Int(getpagesize())
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("verify-cache-\(UUID().uuidString)")
        try Data(repeating: 0x5a, count: stride * 16).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let layout = StreamLayout(path: url.path, streamOffset: 0, streamSize: UInt64(stride * 16),
                                  expertsPerLayer: 16, expertStride: UInt64(stride))
        let streamer = try PreadExpertStreamer(layout: layout, device: context.device, slotCount: 16)
        _ = try streamer.loadExpertsCached(experts: Array(0..<8))
        let lease = try #require(streamer.beginGPURead())
        defer { lease.release() }
        #expect(lease.buffers.count == 16)
        #expect(lease.populatedBuffers.count == 8)
        #expect(Set(lease.populatedBuffers.map(\.gpuAddress))
            == Set(zip(lease.buffers, lease.experts).compactMap { buffer, expert in
                expert >= 0 ? buffer.gpuAddress : nil
            }))
        #expect(throws: (any Error).self) {
            try lease.recordCompletedHitGroups([Array(0..<8), Array(8..<16)])
        }
        #expect(streamer.successfulCachePlanCounts.hits == 0)
        try lease.recordCompletedHitGroups([Array(0..<8), Array((0..<8).reversed())])
        #expect(streamer.successfulCachePlanCounts.hits == 16)
        #expect(throws: (any Error).self) { try lease.recordCompletedHitGroups([Array(0..<8)]) }
        #expect(streamer.successfulCachePlanCounts.hits == 16)
    }
}
