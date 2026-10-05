import Metal

/// Runs each token's small operations in a shared dispatch.
final class GemmaVerifyFusions {
    private let norm, postAttention, tail, gelu, epilogue: MTLComputePipelineState

    init(context: MetalContext) throws {
        norm = try context.pipeline("gemma_verify_norm")
        postAttention = try context.pipeline("gemma_verify_fused_post_attn_setup")
        tail = try context.pipeline("gemma_verify_fused_layer_tail")
        gelu = try context.pipeline("gemma_verify_gelu")
        epilogue = try context.pipeline("gemma_verify_epilogue")
    }

    private func encoder(_ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
                         _ buffers: [[(MTLBuffer, Int)]]) throws -> MTLComputeCommandEncoder {
        precondition(!buffers.isEmpty && (1...5).contains(buffers[0].count))
        precondition(buffers.allSatisfy { $0.count == buffers[0].count })
        guard let encoder = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        encoder.setComputePipelineState(pipeline)
        for (group, rows) in buffers.enumerated() {
            for index in 0..<5 {
                let value = rows[min(index, rows.count - 1)]
                encoder.setBuffer(value.0, offset: value.1, index: group * 5 + index)
            }
        }
        return encoder
    }

    private func finish(_ encoder: MTLComputeCommandEncoder, rows: Int,
                        width: Int = 1, conditional: DecodeDispatch?) {
        encoder.dispatchDecode(MTLSize(width: width, height: rows, depth: 1),
            threads: MTLSize(width: 256, height: 1, depth: 1), conditional: conditional)
        encoder.endEncoding()
    }

    private func ranges(for count: Int) -> [Range<Int>] {
        precondition((1...7).contains(count))
        return count <= 5 ? [0..<count] : [0..<5, 5..<count]
    }

    func normalize(command: MTLCommandBuffer, inputs: [MTLBuffer], outputs: [MTLBuffer],
                   weight: TensorView, conditional: DecodeDispatch? = nil) throws {
        precondition(inputs.count == outputs.count)
        for range in ranges(for: inputs.count) {
            let input = inputs[range].map { ($0, 0) }
            let output = outputs[range].map { ($0, 0) }
            let encoder = try encoder(command, norm, [input, output])
            encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 10)
            finish(encoder, rows: range.count, conditional: conditional)
        }
    }

    func postAttention(command: MTLCommandBuffer, rows: [GemmaVerifyRow],
                       post: TensorView, preFFN: TensorView, preFFN2: TensorView,
                       conditional: DecodeDispatch? = nil) throws {
        for range in ranges(for: rows.count) {
            let chunk = rows[range]
            let encoder = try encoder(command, postAttention, [chunk.map { ($0.hidden, 0) },
                chunk.map { ($0.oOut, 0) }, chunk.map { ($0.denseX, 0) },
                chunk.map { ($0.routedX, 0) }, chunk.map { ($0.routerInput, 0) }])
            for (index, weight) in [post, preFFN, preFFN2].enumerated() {
                encoder.setBuffer(weight.buffer, offset: Int(weight.offset), index: 25 + index)
            }
            finish(encoder, rows: range.count, conditional: conditional)
        }
    }

    func tail(command: MTLCommandBuffer, rows: [GemmaVerifyRow],
              postFFN2: TensorView, postFFN: TensorView, scalar: Float,
              conditional: DecodeDispatch? = nil) throws {
        for range in ranges(for: rows.count) {
            let chunk = rows[range]
            let encoder = try encoder(command, tail, [chunk.map { ($0.h2Buf, 0) },
                chunk.map { ($0.h1Buf, 0) }, chunk.map { ($0.hidden, 0) }])
            encoder.setBuffer(postFFN2.buffer, offset: Int(postFFN2.offset), index: 15)
            encoder.setBuffer(postFFN.buffer, offset: Int(postFFN.offset), index: 16)
            var scalar = scalar
            encoder.setBytes(&scalar, length: 4, index: 17)
            finish(encoder, rows: range.count, conditional: conditional)
        }
    }

    func gelu(command: MTLCommandBuffer, rows: [GemmaVerifyRow], count: UInt32,
              conditional: DecodeDispatch? = nil) throws {
        for range in ranges(for: rows.count) {
            let chunk = rows[range]
            let encoder = try encoder(command, gelu, [chunk.map { ($0.denseScratchGate, 0) },
                chunk.map { ($0.denseScratchUp, 0) }, chunk.map { ($0.denseScratchAct, 0) }])
            var count = count
            encoder.setBytes(&count, length: 4, index: 15)
            finish(encoder, rows: range.count, width: (Int(count) + 255) / 256,
                conditional: conditional)
        }
    }

    func epilogue(command: MTLCommandBuffer, rows: [GemmaVerifyRow],
                  keys: [(buffer: MTLBuffer, offset: Int)], values: [(buffer: MTLBuffer, offset: Int)],
                  qNorm: TensorView, kNorm: TensorView, headDim: UInt32,
                  heads: UInt32, kvHeads: UInt32, position: Int, theta: Float, rotatedPairs: UInt32,
                  conditional: DecodeDispatch? = nil) throws {
        precondition(rows.count == keys.count && rows.count == values.count)
        for range in ranges(for: rows.count) {
            let chunk = rows[range]
            let encoder = try encoder(command, epilogue, [chunk.map { ($0.qScratch, 0) },
                Array(keys[range]), Array(values[range])])
            encoder.setBuffer(qNorm.buffer, offset: Int(qNorm.offset), index: 15)
            encoder.setBuffer(kNorm.buffer, offset: Int(kNorm.offset), index: 16)
            var dimensions = [headDim, heads, kvHeads, UInt32(position + range.lowerBound)]
            for index in dimensions.indices {
                encoder.setBytes(&dimensions[index], length: 4, index: 17 + index)
            }
            var theta = theta, eps: Float = 1e-6
            var rotatedPairs = rotatedPairs
            encoder.setBytes(&theta, length: 4, index: 21)
            encoder.setBytes(&rotatedPairs, length: 4, index: 22)
            encoder.setBytes(&eps, length: 4, index: 23)
            finish(encoder, rows: range.count, width: Int(heads + 2 * kvHeads),
                conditional: conditional)
        }
    }
}
