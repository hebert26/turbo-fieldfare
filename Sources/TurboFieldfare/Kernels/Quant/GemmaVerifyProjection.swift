import Metal

/// A small batch with the same sums as the decode projection.
final class GemmaVerifyProjection {
    private let pipelines: [MTLComputePipelineState]
    private let manyPipelines: [MTLComputePipelineState]
    private let mma3Pipelines: [MTLComputePipelineState]
    private let mma4Pipelines: [MTLComputePipelineState]
    private let mma8Pipelines: [MTLComputePipelineState]
    private let wideHead: MTLComputePipelineState
    var useWideHead = false
    private let gelu: MTLComputePipelineState

    init(context: MetalContext) throws {
        wideHead = try context.pipeline("gemma_verify_int4_rows4", constants: [
            MetalFunctionConstant(index: 90, value: .uint32(2))
        ], maxTotalThreadsPerThreadgroup: 128)
        gelu = try context.pipeline("gelu_mul_fp16")
        pipelines = try (1...5).map { count in
            try context.pipeline("gemma_verify_int4", constants: [
                MetalFunctionConstant(index: 90, value: .uint32(UInt32(count)))
            ], maxTotalThreadsPerThreadgroup: 128)
        }
        manyPipelines = try (1...5).map { count in
            try context.pipeline("gemma_verify_int4_many", constants: [
                MetalFunctionConstant(index: 90, value: .uint32(UInt32(count)))
            ], maxTotalThreadsPerThreadgroup: 128)
        }
        mma3Pipelines = try (6...7).map { count in
            try context.pipeline("gemma_verify_int4_mma", constants: [
                MetalFunctionConstant(index: 90, value: .uint32(UInt32(count))),
                MetalFunctionConstant(index: 92, value: .uint32(3)),
            ], maxTotalThreadsPerThreadgroup: 256)
        }
        mma4Pipelines = try (6...7).map { count in
            try context.pipeline("gemma_verify_int4_mma", constants: [
                MetalFunctionConstant(index: 90, value: .uint32(UInt32(count))),
                MetalFunctionConstant(index: 92, value: .uint32(4)),
            ], maxTotalThreadsPerThreadgroup: 256)
        }
        mma8Pipelines = try (6...7).map { count in
            try context.pipeline("gemma_verify_int4_mma", constants: [
                MetalFunctionConstant(index: 90, value: .uint32(UInt32(count))),
                MetalFunctionConstant(index: 92, value: .uint32(8)),
            ], maxTotalThreadsPerThreadgroup: 256)
        }
    }

    func encodeMany(command: MTLCommandBuffer, projections: [SharedExpertProjection],
                    inputs: [(buffer: MTLBuffer, offset: Int)],
                    outputs: [[(buffer: MTLBuffer, offset: Int)]],
                    conditional: DecodeDispatch? = nil) throws {
        guard (1...3).contains(projections.count), (1...7).contains(inputs.count),
              outputs.count == projections.count,
              projections.allSatisfy({ $0.cols == projections[0].cols && $0.cols % 64 == 0 && $0.weightsOffset % 2 == 0 }),
              inputs.allSatisfy({ $0.offset >= 0 && $0.offset % 8 == 0 &&
                  $0.offset <= $0.buffer.length - Int(projections[0].cols) * 2 }),
              zip(projections, outputs).allSatisfy({ projection, output in
                  output.count == inputs.count && output.allSatisfy {
                      $0.offset >= 0 && $0.offset % 2 == 0 &&
                      $0.offset <= $0.buffer.length - Int(projection.rows) * 2
                  }
              }), inputs.count <= 5 || conditional == nil else {
            throw SharedExpertError.dimensionMismatch("Invalid joined verification projection")
        }
        if inputs.count > 5 {
            for index in projections.indices {
                try encode(command: command, projection: projections[index],
                    inputs: inputs, outputs: outputs[index])
            }
            return
        }
        guard let encoder = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        encoder.setComputePipelineState(manyPipelines[inputs.count - 1])
        for index in 0..<3 {
            let matrix = projections[min(index, projections.count - 1)]
            encoder.setBuffer(matrix.weights, offset: matrix.weightsOffset, index: index * 3)
            encoder.setBuffer(matrix.scales, offset: matrix.scalesOffset, index: index * 3 + 1)
            encoder.setBuffer(matrix.biases, offset: matrix.biasesOffset, index: index * 3 + 2)
            let output = outputs[min(index, outputs.count - 1)]
            for token in 0..<5 {
                let value = output[min(token, output.count - 1)]
                encoder.setBuffer(value.buffer, offset: value.offset, index: 14 + index * 5 + token)
            }
        }
        for token in 0..<5 {
            let value = inputs[min(token, inputs.count - 1)]
            encoder.setBuffer(value.buffer, offset: value.offset, index: 9 + token)
        }
        var rowCounts = (0..<3).map { projections[min($0, projections.count - 1)].rows }
        var columns = projections[0].cols
        encoder.setBytes(&rowCounts, length: 12, index: 29)
        encoder.setBytes(&columns, length: 4, index: 30)
        encoder.dispatchDecode(MTLSize(width: (Int(rowCounts.max()!) + 3) / 4,
            height: 1, depth: projections.count), threads: MTLSize(width: 128, height: 1, depth: 1),
            conditional: conditional)
        encoder.endEncoding()
    }

    func encodeGelu(command: MTLCommandBuffer, gate: MTLBuffer, up: MTLBuffer,
                    output: MTLBuffer, count: UInt32, conditional: DecodeDispatch? = nil) throws {
        guard let encoder = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        encoder.setComputePipelineState(gelu)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var count = count
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.dispatchDecode(MTLSize(width: (Int(count) + 255) / 256, height: 1, depth: 1),
                               threads: MTLSize(width: 256, height: 1, depth: 1), conditional: conditional)
        encoder.endEncoding()
    }

    func encode(command: MTLCommandBuffer, projection: SharedExpertProjection,
                inputs: [(buffer: MTLBuffer, offset: Int)],
                outputs: [(buffer: MTLBuffer, offset: Int)], conditional: DecodeDispatch? = nil) throws {
        guard (1...7).contains(inputs.count), outputs.count == inputs.count,
              projection.cols % 64 == 0, projection.weightsOffset % 2 == 0,
              inputs.allSatisfy({ $0.offset >= 0 && $0.offset % 8 == 0 &&
                  $0.offset <= $0.buffer.length - Int(projection.cols) * 2 }),
              outputs.allSatisfy({ $0.offset >= 0 && $0.offset % 2 == 0 &&
                  $0.offset <= $0.buffer.length - Int(projection.rows) * 2 }) else {
            throw SharedExpertError.dimensionMismatch("Invalid verification projection")
        }
        let wide = useWideHead && inputs.count == 2 && projection.rows == 262144 && projection.cols == 2816
        let matrix = inputs.count > 5
        let quantGroups = Int(projection.cols / 64)
        let splits = quantGroups.isMultiple(of: 8) ? 8
            : (quantGroups.isMultiple(of: 4) ? 4 : 3)
        guard !matrix || Int(projection.cols / 64) % splits == 0 else {
            throw SharedExpertError.dimensionMismatch("Invalid matrix verification projection")
        }
        guard let encoder = command.makeComputeCommandEncoder() else { throw MetalError.noQueue }
        let pipeline = matrix
            ? (splits == 8 ? mma8Pipelines : (splits == 4 ? mma4Pipelines : mma3Pipelines))[inputs.count - 6]
            : (wide ? wideHead : pipelines[inputs.count - 1])
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(projection.weights, offset: projection.weightsOffset, index: 0)
        encoder.setBuffer(projection.scales, offset: projection.scalesOffset, index: 1)
        encoder.setBuffer(projection.biases, offset: projection.biasesOffset, index: 2)
        if matrix {
            for index in 0..<7 {
                let bound = min(index, inputs.count - 1)
                encoder.setBuffer(inputs[bound].buffer, offset: inputs[bound].offset, index: 3 + index)
                encoder.setBuffer(outputs[bound].buffer, offset: outputs[bound].offset, index: 10 + index)
            }
            var rows = projection.rows
            var columns = projection.cols
            encoder.setBytes(&rows, length: 4, index: 17)
            encoder.setBytes(&columns, length: 4, index: 18)
            encoder.dispatchDecode(MTLSize(width: (Int(rows) + 7) / 8, height: 1, depth: 1),
                threads: MTLSize(width: splits * 32, height: 1, depth: 1), conditional: nil)
            encoder.endEncoding()
            return
        }
        for index in 0..<5 {
            let bound = min(index, inputs.count - 1)
            encoder.setBuffer(inputs[bound].buffer, offset: inputs[bound].offset, index: 3 + index)
            encoder.setBuffer(outputs[bound].buffer, offset: outputs[bound].offset, index: 8 + index)
        }
        var rows = projection.rows
        var columns = projection.cols
        encoder.setBytes(&rows, length: 4, index: 13)
        encoder.setBytes(&columns, length: 4, index: 14)
        let rowsPerGroup = wide ? 16 : 4
        encoder.dispatchDecode(MTLSize(width: (Int(rows) + rowsPerGroup - 1) / rowsPerGroup, height: 1, depth: 1),
                               threads: MTLSize(width: 128, height: 1, depth: 1), conditional: conditional)
        encoder.endEncoding()
    }
}
