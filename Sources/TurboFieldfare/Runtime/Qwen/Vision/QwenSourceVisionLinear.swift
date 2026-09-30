import Accelerate
import Metal

/// The source CPU reference uses Accelerate addmm, whose reduction changes
/// with matrix geometry. Preserve its complete shape and bias-seeded output.
enum QwenSourceVisionLinear {
    static func stagingByteCount(inputWidth: Int, outputWidth: Int) throws -> Int {
        let weights = try product(inputWidth, outputWidth)
        let (elements, overflow) = weights.addingReportingOverflow(outputWidth)
        guard !overflow else { throw QwenVisionError.arithmeticOverflow("source FC1 staging") }
        return try product(elements, MemoryLayout<Float>.stride)
    }

    static func project(
        rows: Int, inputWidth: Int, outputWidth: Int,
        input: MTLBuffer, weight: MTLBuffer, weightOffset: Int,
        bias: MTLBuffer, biasOffset: Int, output: MTLBuffer, staging: MTLBuffer
    ) throws {
        try Task.checkCancellation()
        guard rows > 0, inputWidth > 0, outputWidth > 0,
              rows <= Int(Int32.max), inputWidth <= Int(Int32.max),
              outputWidth <= Int(Int32.max) else {
            throw QwenVisionError.invalidFeatureShape
        }
        let weights = try product(inputWidth, outputWidth)
        let inputBytes = try product(try product(rows, inputWidth), 4)
        let outputBytes = try product(try product(rows, outputWidth), 4)
        let weightBytes = try product(weights, 2)
        let biasBytes = try product(outputWidth, 2)
        guard input.length >= inputBytes, output.length >= outputBytes,
              staging.length >= (try stagingByteCount(inputWidth: inputWidth, outputWidth: outputWidth)),
              weightOffset >= 0, weightOffset <= weight.length,
              weightBytes <= weight.length - weightOffset,
              biasOffset >= 0, biasOffset <= bias.length,
              biasBytes <= bias.length - biasOffset,
              weightOffset.isMultiple(of: 2), biasOffset.isMultiple(of: 2) else {
            throw QwenVisionError.invalidFeatureShape
        }
        let sourceWeight = weight.contents().advanced(by: weightOffset)
            .assumingMemoryBound(to: UInt16.self)
        let sourceBias = bias.contents().advanced(by: biasOffset)
            .assumingMemoryBound(to: UInt16.self)
        let converted = staging.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<weights {
            if index.isMultiple(of: 65_536) { try Task.checkCancellation() }
            converted[index] = Float(bitPattern: UInt32(sourceWeight[index]) << 16)
        }
        let convertedBias = converted.advanced(by: weights)
        for column in 0..<outputWidth {
            convertedBias[column] = Float(bitPattern: UInt32(sourceBias[column]) << 16)
        }
        let result = output.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<rows {
            try Task.checkCancellation()
            result.advanced(by: row * outputWidth).update(from: convertedBias, count: outputWidth)
        }
        try Task.checkCancellation()
        // Row-major Y[M,N] is column-major Y^T[N,M]. W is stored [N,K],
        // so transpose its column-major [K,N] view, preserving Torch's call.
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    Int32(outputWidth), Int32(rows), Int32(inputWidth), 1,
                    converted, Int32(inputWidth),
                    input.contents().assumingMemoryBound(to: Float.self), Int32(inputWidth),
                    1, result, Int32(outputWidth))
        try Task.checkCancellation()
    }

    private static func product(_ first: Int, _ second: Int) throws -> Int {
        let (value, overflow) = first.multipliedReportingOverflow(by: second)
        guard first >= 0, second >= 0, !overflow else {
            throw QwenVisionError.arithmeticOverflow("source FC1 staging")
        }
        return value
    }
}
