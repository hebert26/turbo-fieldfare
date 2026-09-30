import Darwin
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenVisionGELUPrecisionTests {
    @Test func sourceInPlaceProductionTanhGELUMatchesPinnedTorchBits() async throws {
        // Normal values are selected from independently saved Torch 2.10 CPU
        // GELU oracle arrays. Signed-zero and subnormal cases are tested by the
        // direct in-place GELU kernel below, without a preceding dot product.
        // Oracle receipt:
        // phase23-vision-diagnosis/gelu-cpu-002/receipt.json, SHA-256
        // c0f3e1ffb87deecdcb3eb74d90858201754644006f1c92c1f7170439f9b4a1cd.
        let preactivationBits: [UInt32] = [
            0xc0a00000, 0xc0400000, 0xbf800000, 0xbf000000,
            0x00000000, 0x3f000000, 0x3f800000, 0x40a00000,
        ]
        let expectedBits: [UInt32] = [
            0xb4a00000, 0xbb6e6200, 0xbe229e90, 0xbe1dfd26,
            0x00000000, 0x3eb1016d, 0x3f57585c, 0x409fffff,
        ]
        let context = try MetalContext()
        let actual = try await runSourceInPlaceGELU(
            context: context, values: preactivationBits.map { Float(bitPattern: $0) },
            functionName: "qwen_source_vision_gelu_tanh_in_place",
            dispatchCount: preactivationBits.count)
        #expect(actual.map(\.bitPattern) == expectedBits)
    }

    @Test func sourcePrivateProductionMergerGELUMatchesPinnedTorchBits() async throws {
        // Expected bits come from the same pinned CPU receipt's official.gelu-erf array.
        let inputBits: [UInt32] = [
            0xc0a00000, 0xbf800000, 0x807fffff, 0x80000001,
            0x80000000, 0x00000000, 0x00000001, 0x007fffff,
            0x3f800000, 0x40a00000, 0xc0a00000, 0xbf800000,
            0x007fffff, 0x00000001, 0x80000000, 0x00000000,
        ]
        let expectedBits: [UInt32] = [
            0xb5c80000, 0xbe227686, 0x80400000, 0x80000000,
            0x80000000, 0x00000000, 0x00000000, 0x00400000,
            0x3f57625e, 0x409ffffd, 0xb5c80000, 0xbe227686,
            0x00400000, 0x00000000, 0x80000000, 0x00000000,
        ]
        let context = try MetalContext()
        let actual = try await runSourceInPlaceGELU(
            context: context, values: inputBits.map { Float(bitPattern: $0) })
        #expect(actual.map(\.bitPattern) == expectedBits)
    }

    @Test func productionLinearTanhGELUKernelStaysFiniteAcrossSaturationAndTransition() async throws {
        let preactivations: [Float] = [
            -20, -13, -10, -5, -3, -1, -0.5, -0.1,
            0, 0.1, 0.5, 1, 3, 5, 10, 13, 20,
        ]
        let context = try MetalContext()
        let input: [Float] = [0.5, -0.25]
        let weights: [Float] = preactivations.flatMap { [$0 * 2, 0] }
        let bias = [Float](repeating: 0, count: preactivations.count)
        let inputBuffer = makeBuffer(input, device: context.device)
        let weightBuffer = makeBuffer(weights, device: context.device)
        let biasBuffer = makeBuffer(bias, device: context.device)
        let fastOutput = try await runLinearGELU(
            context: context, input: inputBuffer, weight: weightBuffer, bias: biasBuffer,
            outputCount: preactivations.count, mathFloatingPointFunctions: nil)
        // QwenVisionRuntime now builds this production pipeline with .precise.
        let output = try await runLinearGELU(
            context: context, input: inputBuffer, weight: weightBuffer, bias: biasBuffer,
            outputCount: preactivations.count, mathFloatingPointFunctions: .precise)

        for (option, values) in [("nil", fastOutput), ("precise", output)] {
            #expect(values.allSatisfy { $0.isFinite }, "\(option) output must be finite")
            for index in preactivations.indices {
                let expected = exactGELU(Double(preactivations[index]))
                let tolerance = 6e-4 + 1e-5 * abs(expected)
                #expect(abs(Double(values[index]) - expected) <= tolerance,
                        "\(option) GELU(\(preactivations[index])) actual=\(values[index]) expected=\(expected)")
            }

            for value in [Float(10), 13, 20] {
                let index = try #require(preactivations.firstIndex(of: value))
                #expect(values[index] == value)
            }
            for value in [Float(-10), -13, -20] {
                let index = try #require(preactivations.firstIndex(of: value))
                #expect(abs(values[index]) <= 1e-7)
            }
        }
    }
}

private struct VisionGELUTestParameters {
    var rows: UInt32
    var paddedRows: UInt32
    var inputWidth: UInt32
    var outputWidth: UInt32
    var intermediateWidth: UInt32
    var heads: UInt32
    var gridHeight: UInt32
    var gridWidth: UInt32
    var mergeSize: UInt32
    var positionCount: UInt32
    var epsilon: Float
    var weightScalarBytes: UInt32
    var patchScalarBytes: UInt32
    var reserved: UInt32
}

private func runLinearGELU(
    context: MetalContext,
    input: MTLBuffer,
    weight: MTLBuffer,
    bias: MTLBuffer,
    outputCount: Int,
    mathFloatingPointFunctions: MTLMathFloatingPointFunctions?,
    includeQwenSourceMath: Bool = false,
    functionName: String = "qwen_vision_linear_gelu_tanh"
) async throws -> [Float] {
    // Same production module, kernel, and safe math mode as runtime pipeline creation.
    let library = try MetalContext.privateLibrary(
        device: context.device, module: "qwen_vision", mathMode: .safe,
        mathFloatingPointFunctions: mathFloatingPointFunctions,
        includeQwenSourceMath: includeQwenSourceMath)
    let function = try #require(library.makeFunction(name: functionName))
    let pipeline = try await context.device.makeComputePipelineState(function: function)
    let output = try #require(context.device.makeBuffer(
        length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared))
    var parameters = VisionGELUTestParameters(
        rows: 1, paddedRows: 1, inputWidth: 2,
        outputWidth: UInt32(outputCount), intermediateWidth: UInt32(outputCount), heads: 1,
        gridHeight: 1, gridWidth: 1, mergeSize: 1, positionCount: 1,
        epsilon: 1e-6, weightScalarBytes: 4, patchScalarBytes: 4, reserved: 0)
    let command = try #require(context.queue.makeCommandBuffer())
    let encoder = try #require(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBytes(&parameters, length: MemoryLayout<VisionGELUTestParameters>.stride, index: 0)
    encoder.setBuffer(input, offset: 0, index: 1)
    encoder.setBuffer(weight, offset: 0, index: 2)
    encoder.setBuffer(bias, offset: 0, index: 3)
    encoder.setBuffer(output, offset: 0, index: 4)
    let groupWidth = min(outputCount, pipeline.maxTotalThreadsPerThreadgroup)
    encoder.dispatchThreads(
        MTLSize(width: outputCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    return Array(UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: Float.self), count: outputCount))
}

private func runSourceInPlaceGELU(
    context: MetalContext, values: [Float], functionName: String = "qwen_source_vision_gelu_in_place",
    dispatchCount: Int? = nil
) async throws -> [Float] {
    #expect(values.count.isMultiple(of: 8))
    let library = try MetalContext.privateLibrary(
        device: context.device, module: "qwen_vision", mathMode: .safe,
        mathFloatingPointFunctions: .precise, includeQwenSourceMath: true)
    let function = try #require(library.makeFunction(name: functionName))
    let pipeline = try await context.device.makeComputePipelineState(function: function)
    let buffer = makeBuffer(values, device: context.device)
    var parameters = VisionGELUTestParameters(
        rows: 1, paddedRows: 1, inputWidth: UInt32(values.count),
        outputWidth: UInt32(values.count), intermediateWidth: UInt32(values.count), heads: 1,
        gridHeight: 1, gridWidth: 1, mergeSize: 1, positionCount: 1,
        epsilon: 1e-6, weightScalarBytes: 4, patchScalarBytes: 4, reserved: 0)
    let command = try #require(context.queue.makeCommandBuffer())
    let encoder = try #require(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBytes(&parameters, length: MemoryLayout<VisionGELUTestParameters>.stride, index: 0)
    encoder.setBuffer(buffer, offset: 0, index: 1)
    let workCount = dispatchCount ?? (values.count / 4)
    let groupWidth = min(workCount, pipeline.maxTotalThreadsPerThreadgroup)
    encoder.dispatchThreads(
        MTLSize(width: workCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    return Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: values.count))
}

private func makeBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(
        bytes: values, length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared)!
}

/// Independent exact-GELU reference. The production kernel uses the model's
/// tanh approximation, whose known approximation error fits the bound above.
private func exactGELU(_ value: Double) -> Double {
    0.5 * value * (1 + Darwin.erf(value / Foundation.sqrt(2)))
}
