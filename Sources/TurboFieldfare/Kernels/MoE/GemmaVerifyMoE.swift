import Metal

/// Groups the verification work by expert while keeping each token's route order.
final class GemmaVerifyMoE {
    private let gateUp, down, reduce: MTLComputePipelineState
    private let pairGateUp, pairDown: MTLComputePipelineState
    var useStaticPair = true
    private let encoder: MTLArgumentEncoder
    private let arguments, slots: MTLBuffer
    private let partials: [MTLBuffer]
    var argumentLength: Int { encoder.encodedLength }

    init(context: MetalContext) throws {
        gateUp = try context.pipeline("gemma_verify_expert_gate_up", constants: [], maxTotalThreadsPerThreadgroup: 128)
        down = try context.pipeline("gemma_verify_expert_down", constants: [], maxTotalThreadsPerThreadgroup: 128)
        let pair = [MetalFunctionConstant(index: 91, value: .uint32(2))]
        pairGateUp = try context.pipeline("gemma_verify_expert_gate_up", constants: pair, maxTotalThreadsPerThreadgroup: 128)
        pairDown = try context.pipeline("gemma_verify_expert_down", constants: pair, maxTotalThreadsPerThreadgroup: 128)
        reduce = try context.pipeline("gemma_verify_expert_reduce")
        guard let function = context.library.makeFunction(name: "gemma_verify_expert_gate_up") else {
            throw MetalError.missingFunction("gemma_verify_expert_gate_up")
        }
        encoder = function.makeArgumentEncoder(bufferIndex: 0)
        func buffer(_ bytes: Int) throws -> MTLBuffer {
            guard let result = context.device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw MetalError.noDevice
            }
            return result
        }
        arguments = try buffer(encoder.encodedLength)
        slots = try buffer(40 * 5 * 4)
        partials = try (0..<5).map { _ in try buffer(8 * 2816 * 4) }
    }

    /// The caller must finish the prior layer before reusing this argument buffer.
    func encode(command: MTLCommandBuffer, rows: [GemmaVerifyRow], experts: [Int],
                blobs: [MTLBuffer], routes: [[Int]], offsets: MoEExpertOffsets) throws {
        guard (1...5).contains(rows.count), rows.count == routes.count,
              experts.count <= 40, experts.count == blobs.count,
              routes.allSatisfy({ $0.count == 8 && Set($0).count == 8 }) else {
            throw ModelError.indexCorrupt(detail: "Invalid verification routes")
        }
        let indices = Dictionary(uniqueKeysWithValues: experts.enumerated().map { ($0.element, $0.offset) })
        let mapping = slots.contents().assumingMemoryBound(to: UInt32.self)
        for index in 0..<(40 * 5) { mapping[index] = UInt32.max }
        for (token, route) in routes.enumerated() {
            for (slot, expert) in route.enumerated() {
                guard let index = indices[expert] else { throw ModelError.indexCorrupt(detail: "Missing verification route") }
                mapping[index * 5 + token] = UInt32(slot)
            }
        }
        encoder.setArgumentBuffer(arguments, offset: 0)
        for index in 0..<40 { encoder.setBuffer(index < blobs.count ? blobs[index] : nil, offset: 0, index: index) }
        bindRows(rows, arguments: arguments)
        try dispatch(command: command, rows: rows, arguments: arguments, slots: slots,
            blobs: blobs, heap: nil, expertCount: experts.count, offsets: offsets, conditional: nil)
    }

    func encodeCached(command: MTLCommandBuffer, rows: [GemmaVerifyRow], arguments: MTLBuffer,
                      slots: MTLBuffer, blobs: [MTLBuffer], heap: MTLHeap?, offsets: MoEExpertOffsets,
                      conditional: DecodeDispatch) throws {
        encoder.setArgumentBuffer(arguments, offset: 0)
        bindRows(rows, arguments: arguments)
        try dispatch(command: command, rows: rows, arguments: arguments, slots: slots,
            blobs: blobs, heap: heap, expertCount: rows.count * 8, offsets: offsets, conditional: conditional)
    }

    private func bindRows(_ rows: [GemmaVerifyRow], arguments: MTLBuffer) {
        for index in 0..<5 {
            let row = rows[min(index, rows.count - 1)]
            encoder.setBuffer(row.routedX, offset: 0, index: 40 + index)
            encoder.setBuffer(row.moeActs, offset: 0, index: 45 + index)
            encoder.setBuffer(row.outWeights, offset: 0, index: 50 + index)
            encoder.setBuffer(partials[index], offset: 0, index: 55 + index)
            encoder.setBuffer(row.h2Buf, offset: 0, index: 60 + index)
        }
    }

    private func dispatch(command: MTLCommandBuffer, rows: [GemmaVerifyRow], arguments: MTLBuffer,
                          slots: MTLBuffer, blobs: [MTLBuffer], heap: MTLHeap?, expertCount: Int,
                          offsets: MoEExpertOffsets, conditional: DecodeDispatch?) throws {
        var offsets = offsets
        var tokenCount = UInt32(rows.count)
        let paired = useStaticPair && rows.count == 2
        for (pipeline, width) in [(paired ? pairGateUp : gateUp, 704), (paired ? pairDown : down, 2816)] {
            guard let compute = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
            compute.setComputePipelineState(pipeline)
            compute.setBuffer(arguments, offset: 0, index: 0)
            compute.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
            compute.setBuffer(slots, offset: 0, index: 2)
            compute.setBytes(&tokenCount, length: 4, index: 3)
            if let heap {
                compute.useHeap(heap)
            } else {
                let resources: [MTLResource] = blobs
                compute.useResources(resources, usage: .read)
            }
            for (index, row) in rows.enumerated() {
                compute.useResource(row.routedX, usage: .read)
                compute.useResource(row.moeActs, usage: [.read, .write])
                compute.useResource(row.outWeights, usage: .read)
                compute.useResource(partials[index], usage: .write)
            }
            compute.dispatchDecode(MTLSize(width: width / 4, height: expertCount, depth: 1),
                threads: MTLSize(width: 128, height: 1, depth: 1), conditional: conditional)
            compute.endEncoding()
        }
        guard let compute = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        compute.setComputePipelineState(reduce)
        compute.setBuffer(arguments, offset: 0, index: 0)
        for (index, row) in rows.enumerated() {
            compute.useResource(partials[index], usage: .read)
            compute.useResource(row.h2Buf, usage: .write)
        }
        compute.dispatchDecode(MTLSize(width: 11, height: rows.count, depth: 1),
            threads: MTLSize(width: 256, height: 1, depth: 1), conditional: conditional)
        compute.endEncoding()
    }
}
