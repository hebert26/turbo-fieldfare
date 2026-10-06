import Foundation
import Darwin
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct GPUExpertCacheTests {
    @Test(arguments: [-1, 0, 1, GPUExpertCache.maxLayers - 1])
    func missStopsOnlyUnfinishedWork(missingLayer: Int) throws {
        let context = try MetalContext()
        let cache = try GPUExpertCache(context: context, selectedLength: 64)
        let rms = try RMSNorm(context: context)
        let select = try context.pipeline("router_topk_select_k8",
            constants: [MetalFunctionConstant(index: 40, value: .uint32(128)),
                        MetalFunctionConstant(index: 43, value: .bool(true))])
        let device = context.device
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try #require(device.makeBuffer(bytes: values,
                length: values.count * MemoryLayout<T>.stride, options: .storageModeShared))
        }
        let input = try buffer([Float16](repeating: 1, count: 64))
        let normWeight = try buffer([UInt16](repeating: 0x3f80, count: 64))
        let scale = try buffer((0..<128).map { Quantization.bf16Bits(0.5 + Float($0) / 128) })
        let blobs = try (0..<64).map { try buffer([UInt32($0)]) }

        // Reuse the same control buffers after a stopped batch.
        for missing in [missingLayer, -1] {
            cache.reset()
            let command = try #require(context.queue.makeCommandBuffer())
            var prefixes: [MTLBuffer] = []
            var tails: [MTLBuffer] = []
            var indices: [MTLBuffer] = []
            var weights: [MTLBuffer] = []
            var referenceIndices: [MTLBuffer] = []
            var referenceWeights: [MTLBuffer] = []
            for index in 0..<GPUExpertCache.maxLayers {
                let layer = cache.layers[index]
                layer.dispatch.reset()
                let addresses = layer.slots.contents().assumingMemoryBound(to: UInt64.self)
                let mapping = layer.slotByExpert.contents().assumingMemoryBound(to: UInt32.self)
                for expert in 0..<128 { mapping[expert] = .max }
                for slot in 0..<64 {
                    addresses[slot] = blobs[slot].gpuAddress
                    mapping[127 - slot] = UInt32(slot)
                }
                if index == missing { mapping[127] = .max }
                layer.routes.contents().initializeMemory(as: UInt32.self, repeating: .max, count: 8)
                let logits = try buffer((0..<128).map { Float($0 / 2) * 0.125 })
                let outIndices = try buffer([UInt32](repeating: .max, count: 8))
                let outWeights = try buffer([UInt16](repeating: 0xffff, count: 8))
                let refIndices = try buffer([UInt32](repeating: .max, count: 8))
                let refWeights = try buffer([UInt16](repeating: 0xffff, count: 8))
                let prefix = try buffer([Float16](repeating: -7, count: 64))
                let tail = try buffer([Float16](repeating: -7, count: 64))
                prefixes.append(prefix)
                tails.append(tail)
                indices.append(outIndices)
                weights.append(outWeights)
                referenceIndices.append(refIndices)
                referenceWeights.append(refWeights)

                let encoder = try #require(command.makeComputeCommandEncoder())
                encoder.setComputePipelineState(select)
                encoder.setBuffer(logits, offset: 0, index: 0)
                encoder.setBuffer(scale, offset: 0, index: 1)
                encoder.setBuffer(refIndices, offset: 0, index: 2)
                encoder.setBuffer(refWeights, offset: 0, index: 3)
                var expertCount: UInt32 = 128
                encoder.setBytes(&expertCount, length: 4, index: 4)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                encoder.endEncoding()

                rms.encodeBF16W(commandBuffer: command, x: input, weight: normWeight,
                                out: prefix, d: 64, eps: 1e-6, conditional: layer.dispatch)
                cache.selection(index: index, slotCount: 64).encode(
                    commandBuffer: command, logits: logits, perExpertScale: scale,
                    perExpertScaleOffset: 0, outIndices: outIndices, outWeights: outWeights)
                rms.encodeBF16W(commandBuffer: command, x: input, weight: normWeight,
                                out: tail, d: 64, eps: 1e-6, conditional: layer.dispatch)
            }
            command.commit()
            command.waitUntilCompleted()
            try checkCommandBufferError(command)
            #expect(cache.firstMissingLayer == (missing < 0 ? nil : missing))
            for index in 0..<GPUExpertCache.maxLayers {
                let prefixRan = missing < 0 || index <= missing
                let tailRan = missing < 0 || index < missing
                #expect(prefixes[index].contents().load(as: Float16.self) == (prefixRan ? 1 : -7))
                #expect(tails[index].contents().load(as: Float16.self) == (tailRan ? 1 : -7))
                let actual = indices[index].contents().assumingMemoryBound(to: UInt32.self)
                let expected = referenceIndices[index].contents().assumingMemoryBound(to: UInt32.self)
                let actualWeights = weights[index].contents().assumingMemoryBound(to: UInt16.self)
                let expectedWeights = referenceWeights[index].contents().assumingMemoryBound(to: UInt16.self)
                let selected = cache.layers[index].selected.contents().assumingMemoryBound(to: UInt64.self)
                for route in 0..<8 {
                    if prefixRan {
                        #expect(actual[route] == expected[route])
                        #expect(actualWeights[route] == expectedWeights[route])
                        #expect(cache.layers[index].completedRoutes[route] == Int(expected[route]))
                    } else {
                        #expect(actual[route] == .max)
                        #expect(actualWeights[route] == 0xffff)
                        #expect(cache.layers[index].completedRoutes[route] == Int(UInt32.max))
                    }
                    if tailRan {
                        #expect(selected[route] == blobs[127 - Int(expected[route])].gpuAddress)
                    }
                }
            }
        }
    }

    @Test func releasedLeaseCannotUnlockItsReplacement() throws {
        let context = try MetalContext()
        let stride = Int(getpagesize())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gpu-cache-\(UUID().uuidString).bin")
        try Data(repeating: 0x5a, count: stride * 16).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let layout = StreamLayout(path: url.path, streamOffset: 0,
                                  streamSize: UInt64(stride * 16), expertsPerLayer: 16,
                                  expertStride: UInt64(stride))
        let streamer = try PreadExpertStreamer(layout: layout, device: context.device, slotCount: 8)
        #expect(streamer.beginGPURead() == nil)
        _ = try streamer.loadExpertsCached(experts: Array(0..<8))
        var old: PreadExpertStreamer.GPUReadLease? = try #require(streamer.beginGPURead())
        #expect(streamer.planExpertsCachedIfPossible(experts: Array(8..<16)) == nil)
        #expect(throws: (any Error).self) { try streamer.loadExpert(layer: 0, expert: 8, slot: 0) }
        try old?.recordCompletedHits(Array(0..<8))
        #expect(streamer.successfulCachePlanCounts.hits == 8)
        #expect(throws: (any Error).self) { try old?.recordCompletedHits(Array(0..<8)) }
        old?.release()
        let replacement = try #require(streamer.beginGPURead())
        old = nil
        #expect(streamer.beginGPURead() == nil)
        #expect(streamer.planExpertsCachedIfPossible(experts: Array(8..<16)) == nil)
        replacement.release()
        #expect(streamer.planExpertsCachedIfPossible(experts: Array(8..<16)) != nil)
    }
}
