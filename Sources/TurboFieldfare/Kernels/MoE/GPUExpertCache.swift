import Metal

/// Reuses a small set of control buffers. It does not retain expert weights.
final class GPUExpertCache {
    static let maxLayers = 8
    static let maxExperts = 128
    static let maxBatchBytes = 2 * 1_024 * 1_024 * 1_024
    static let routerDispatchCount = 9
    static let layerDispatchCount = 17

    final class Layer {
        let dispatch: DecodeDispatch
        let slots: MTLBuffer
        let slotByExpert: MTLBuffer
        let selected: MTLBuffer
        let routes: MTLBuffer

        init(device: MTLDevice, selectedLength: Int,
             dispatches: MTLBuffer, index: Int) throws {
            dispatch = DecodeDispatch(buffer: dispatches,
                                      byteOffset: index * DecodeDispatch.capacity
                                          * DecodeDispatch.argumentStride)
            func buffer(_ size: Int) throws -> MTLBuffer {
                guard let result = device.makeBuffer(length: size, options: .storageModeShared) else {
                    throw MetalError.noDevice
                }
                return result
            }
            slots = try buffer(maxExperts * MemoryLayout<UInt64>.stride)
            slotByExpert = try buffer(maxExperts * MemoryLayout<UInt32>.stride)
            selected = try buffer(selectedLength)
            routes = try buffer(MoE.maxStreamedExperts * MemoryLayout<UInt32>.stride)
        }

        func prepare(_ lease: PreadExpertStreamer.GPUReadLease) {
            precondition(lease.buffers.count <= maxExperts)
            precondition(lease.buffers.count == lease.experts.count)
            dispatch.reset()
            let addresses = slots.contents().assumingMemoryBound(to: UInt64.self)
            let mapping = slotByExpert.contents().assumingMemoryBound(to: UInt32.self)
            for expert in 0..<maxExperts { mapping[expert] = UInt32.max }
            for (slot, buffer) in lease.buffers.enumerated() {
                addresses[slot] = buffer.gpuAddress
                let expert = lease.experts[slot]
                if expert >= 0 && expert < maxExperts {
                    mapping[expert] = UInt32(slot)
                }
            }
        }

        var completedRoutes: [Int] {
            let pointer = routes.contents().assumingMemoryBound(to: UInt32.self)
            return (0..<MoE.maxStreamedExperts).map { Int(pointer[$0]) }
        }
    }

    let layers: [Layer]
    private let stoppedLayer: MTLBuffer
    private let selectPSO: MTLComputePipelineState

    struct Selection {
        fileprivate let cache: GPUExpertCache
        fileprivate let index: Int
        fileprivate let slotCount: Int

        func encode(commandBuffer: MTLCommandBuffer, logits: MTLBuffer,
                    perExpertScale: MTLBuffer, perExpertScaleOffset: Int,
                    outIndices: MTLBuffer, outWeights: MTLBuffer) {
            cache.encodeSelection(commandBuffer: commandBuffer, index: index,
                                  slotCount: slotCount, logits: logits,
                                  perExpertScale: perExpertScale,
                                  perExpertScaleOffset: perExpertScaleOffset,
                                  outIndices: outIndices, outWeights: outWeights)
        }
    }

    init(context: MetalContext, selectedLength: Int) throws {
        precondition(selectedLength >= MoE.maxStreamedExperts * MemoryLayout<UInt64>.stride)
        selectPSO = try context.pipeline("router_topk_select_k8_cached",
            constants: [MetalFunctionConstant(index: 40, value: .uint32(128)),
                        MetalFunctionConstant(index: 43, value: .bool(true))])
        let dispatchBytes = Self.maxLayers * DecodeDispatch.capacity * DecodeDispatch.argumentStride
        guard let stopped = context.device.makeBuffer(length: 4, options: .storageModeShared),
              let dispatches = context.device.makeBuffer(length: dispatchBytes,
                                                        options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        stoppedLayer = stopped
        layers = try (0..<Self.maxLayers).map { index in
            try Layer(device: context.device, selectedLength: selectedLength,
                      dispatches: dispatches, index: index)
        }
    }

    func reset() {
        stoppedLayer.contents().storeBytes(of: UInt32.max, as: UInt32.self)
    }

    var firstMissingLayer: Int? {
        let value = stoppedLayer.contents().load(as: UInt32.self)
        return value == UInt32.max ? nil : Int(value)
    }

    func selection(index: Int, slotCount: Int) -> Selection {
        precondition(slotCount > 0 && slotCount <= Self.maxExperts)
        return Selection(cache: self, index: index, slotCount: slotCount)
    }

    private func encodeSelection(commandBuffer: MTLCommandBuffer, index: Int,
                                 slotCount: Int, logits: MTLBuffer,
                                 perExpertScale: MTLBuffer, perExpertScaleOffset: Int,
                                 outIndices: MTLBuffer, outWeights: MTLBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let layer = layers[index]
        encoder.setComputePipelineState(selectPSO)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        encoder.setBuffer(outIndices, offset: 0, index: 2)
        encoder.setBuffer(outWeights, offset: 0, index: 3)
        var expertCount = UInt32(Self.maxExperts)
        encoder.setBytes(&expertCount, length: 4, index: 4)
        encoder.setBuffer(layer.slots, offset: 0, index: 5)
        encoder.setBuffer(layer.slotByExpert, offset: 0, index: 6)
        encoder.setBuffer(layer.selected, offset: 0, index: 7)
        encoder.setBuffer(layer.routes, offset: 0, index: 8)
        encoder.setBuffer(stoppedLayer, offset: 0, index: 9)
        encoder.setBuffer(layer.dispatch.resolved, offset: 0, index: 10)
        var firstRoutedDispatch = UInt32(layer.dispatch.byteOffset / DecodeDispatch.argumentStride
            + layer.dispatch.count + 1)
        var dispatchLimit = UInt32(Self.maxLayers * DecodeDispatch.capacity)
        var layerIndex = UInt32(index)
        var slots = UInt32(slotCount)
        encoder.setBytes(&firstRoutedDispatch, length: 4, index: 11)
        encoder.setBytes(&dispatchLimit, length: 4, index: 12)
        encoder.setBytes(&layerIndex, length: 4, index: 13)
        encoder.setBytes(&slots, length: 4, index: 14)
        layer.dispatch.encode(encoder, groups: MTLSize(width: 1, height: 1, depth: 1),
                              threads: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
