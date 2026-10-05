import Metal

/// Holds control data for two-token verification. Expert weights stay in the bounded cache.
final class GemmaVerifyCache {
    static let prefixDispatches = 13
    static let layerDispatches = 22

    final class Layer {
        let base: GPUExpertCache.Layer
        let secondSelected: MTLBuffer
        let routes: MTLBuffer
        let groupedArguments: MTLBuffer
        let groupedSlots: MTLBuffer
        var dispatch: DecodeDispatch { base.dispatch }

        init(device: MTLDevice, selectedLength: Int, groupedLength: Int, dispatches: MTLBuffer, index: Int) throws {
            base = try GPUExpertCache.Layer(device: device, selectedLength: selectedLength,
                                            dispatches: dispatches, index: index)
            guard let selected = device.makeBuffer(length: selectedLength, options: .storageModeShared),
                  let routes = device.makeBuffer(length: 16 * 4, options: .storageModeShared),
                  let groupedArguments = device.makeBuffer(length: groupedLength, options: .storageModeShared),
                  let groupedSlots = device.makeBuffer(length: 56 * 7 * 4, options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            secondSelected = selected
            self.routes = routes
            self.groupedArguments = groupedArguments
            self.groupedSlots = groupedSlots
        }

        var completedRoutes: [[Int]] {
            let values = routes.contents().assumingMemoryBound(to: UInt32.self)
            return (0..<2).map { token in (0..<8).map { Int(values[token * 8 + $0]) } }
        }
    }

    let layers: [Layer]
    private let stopped: MTLBuffer
    private let resolve: MTLComputePipelineState

    init(context: MetalContext, selectedLength: Int, groupedLength: Int) throws {
        resolve = try context.pipeline("gemma_verify_cache_resolve")
        guard let stopped = context.device.makeBuffer(length: 4, options: .storageModeShared),
              let dispatches = context.device.makeBuffer(
                length: GPUExpertCache.maxLayers * DecodeDispatch.capacity * DecodeDispatch.argumentStride,
                options: .storageModeShared) else { throw MetalError.noDevice }
        self.stopped = stopped
        layers = try (0..<GPUExpertCache.maxLayers).map {
            try Layer(device: context.device, selectedLength: selectedLength, groupedLength: groupedLength, dispatches: dispatches, index: $0)
        }
    }

    func prepare(_ leases: [PreadExpertStreamer.GPUReadLease]) {
        precondition(leases.count <= layers.count)
        stopped.contents().storeBytes(of: UInt32.max, as: UInt32.self)
        for (index, lease) in leases.enumerated() { layers[index].base.prepare(lease) }
    }

    var firstMissingLayer: Int? {
        let value = stopped.contents().load(as: UInt32.self)
        return value == UInt32.max ? nil : Int(value)
    }

    func encodeLookup(command: MTLCommandBuffer, index: Int, slotCount: Int,
                      indices: [MTLBuffer]) throws {
        precondition(indices.count == 2)
        let state = layers[index]
        guard state.dispatch.count == Self.prefixDispatches,
              let encoder = command.makeComputeCommandEncoder() else {
            throw MetalError.commandBufferFailed("Invalid verification cache prefix")
        }
        encoder.setComputePipelineState(resolve)
        encoder.setBuffer(state.base.slots, offset: 0, index: 0)
        encoder.setBuffer(state.base.slotByExpert, offset: 0, index: 1)
        encoder.setBuffer(indices[0], offset: 0, index: 2)
        encoder.setBuffer(indices[1], offset: 0, index: 3)
        encoder.setBuffer(state.base.selected, offset: 0, index: 4)
        encoder.setBuffer(state.secondSelected, offset: 0, index: 5)
        encoder.setBuffer(state.routes, offset: 0, index: 6)
        encoder.setBuffer(stopped, offset: 0, index: 7)
        encoder.setBuffer(state.dispatch.resolved, offset: 0, index: 8)
        var firstTail = UInt32(state.dispatch.byteOffset / DecodeDispatch.argumentStride + state.dispatch.count + 1)
        var limit = UInt32(GPUExpertCache.maxLayers * DecodeDispatch.capacity)
        var layer = UInt32(index)
        var count = UInt32(slotCount)
        encoder.setBytes(&firstTail, length: 4, index: 9)
        encoder.setBytes(&limit, length: 4, index: 10)
        encoder.setBytes(&layer, length: 4, index: 11)
        encoder.setBytes(&count, length: 4, index: 12)
        encoder.setBuffer(state.groupedArguments, offset: 0, index: 13)
        encoder.setBuffer(state.groupedSlots, offset: 0, index: 14)
        state.dispatch.encode(encoder, groups: MTLSize(width: 1, height: 1, depth: 1),
                              threads: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
