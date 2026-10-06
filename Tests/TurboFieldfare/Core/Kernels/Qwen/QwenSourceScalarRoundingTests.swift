import Darwin
import Metal
import Testing
@testable import TurboFieldfare

/// Exercises the production source-mode causal-convolution kernel with the
/// scalar SiLU path. The expected values come from Swift's system libm, not
/// from the Metal helper under test.
@Suite(.serialized) struct QwenSourceScalarRoundingTests {
    @Test func sourceScalarConvolutionMatchesSystemExpAtSavedOperandAndBoundedNeighbors() throws {
        let savedOperand = Float(bitPattern: 0xbb9faac2)
        let operands: [Float] = [
            savedOperand.nextDown,
            savedOperand,
            savedOperand.nextUp,
            Float(bitPattern: 0x80000000),
            Float.zero,
            Float(-128).nextUp,
            -128,
            Float(128).nextDown,
            128,
        ]

        // The saved probe is the current convolution input and the three
        // preceding history terms are zero. With the final weight equal to
        // one, each nonzero FP32 convolution sum is exactly its input operand.
        let configuration = try QwenGatedDeltaNetConfiguration(
            hiddenSize: 1, keyHeadCount: 1, valueHeadCount: 1,
            keyHeadDimension: 1, valueHeadDimension: 1)
        let context = try MetalContext()
        let runtime = try QwenGatedDeltaNet(
            context: context, configuration: configuration, useOfficialSourceMath: true)
        let channelCount = configuration.convolutionChannelCount
        let tokenCount = operands.count
        let geometry = try QwenLinearAttentionGeometry(
            convolutionWidth: 4, convolutionChannelCount: channelCount,
            valueHeadCount: configuration.valueHeadCount,
            keyHeadDimension: configuration.keyHeadDimension,
            valueHeadDimension: configuration.valueHeadDimension)
        let state = try QwenLinearAttentionState(
            device: context.device, linearAttentionLayerMask: [1], geometry: geometry,
            expectedLinearLayerCount: 1)
        let update = try state.reserveUpdate(layer: 0)

        // Qwen's convolution kernel consumes channel-major buffers. Only the
        // first channel carries the scalar probes. Other channels stay zero so
        // their exact zero outputs provide a small untouched-channel control.
        let input = operands + [Float](repeating: 0, count: (channelCount - 1) * tokenCount)
        var weights = [Float](repeating: 0, count: channelCount * configuration.convolutionWidth)
        weights[configuration.convolutionWidth - 1] = 1
        let inputBuffer = try makeBuffer(input, device: context.device)
        let weightsBuffer = try makeBuffer(weights, device: context.device)
        let outputBuffer = try makeBuffer(
            [Float](repeating: -91, count: input.count), device: context.device)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        do {
            try runtime.encodeCausalConvolution(
                commandBuffer: commandBuffer, channelMajorInput: inputBuffer,
                weights: weightsBuffer, update: update, channelMajorOutput: outputBuffer,
                tokenCount: tokenCount)
            try state.submit(update, on: commandBuffer)
        } catch {
            try? state.abort(update)
            throw error
        }
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        if let error = commandBuffer.error { throw error }

        // Match the kernel's initial +0 accumulator for the signed-zero
        // boundary while keeping every nonzero sum equal to its input.
        let sums = operands.map { Float.zero + $0 }
        let expectedExp = sums.map { Darwin.expf(-$0) }
        let expectedSiLU = zip(sums, expectedExp).map { value, exponential in
            value / (1 + exponential)
        }
        let actual = readFloats(outputBuffer, count: input.count)
        let actualProbe = Array(actual.prefix(tokenCount))
        #expect(actualProbe.map(\.bitPattern) == expectedSiLU.map(\.bitPattern))
        #expect(actual.dropFirst(tokenCount).allSatisfy { $0.bitPattern == 0 })

        // The saved operand is the captured one-bit rounding discriminator:
        // system expf produces 0x3f80a00f and the resulting SiLU is 0xbb1f472c.
        let savedIndex = try #require(operands.firstIndex(of: savedOperand))
        #expect(expectedExp[savedIndex].bitPattern == 0x3f80a00f)
        #expect(expectedSiLU[savedIndex].bitPattern == 0xbb1f472c)
        #expect(actualProbe[savedIndex].bitPattern == 0xbb1f472c)

        let committed = try state.layerState(0)
        #expect(Array(committed.convolutionHistory[0..<4]) == Array(operands.suffix(4)))
        #expect(committed.recurrentMatrix == [0])
    }

    @Test func sourceScalarConvolutionMatchesSystemExpForNormalTinyInputsWithSubnormalSiLUOutputs() throws {
        let operands: [Float] = [
            Float(bitPattern: 0x00800000),
            Float(bitPattern: 0x00800001),
            Float(bitPattern: 0x00ffffff),
            Float(bitPattern: 0x01000000),
            Float(bitPattern: 0x01000001),
            Float(bitPattern: 0x80800000),
            Float(bitPattern: 0x80800001),
            Float(bitPattern: 0x80ffffff),
            Float(bitPattern: 0x81000000),
            Float(bitPattern: 0x81000001),
            Float(bitPattern: 0x80000000),
            Float.zero,
        ]
        let actual = try runSourceScalarConvolution(operands: operands)

        // Main157 recorded that GPU convolution can flush subnormal operands
        // or intermediates before the scalar helper. This test stays within
        // the normal-input production scope and checks its subnormal SiLU
        // outputs exactly. Direct raw-tiny activation proof 156 is separate.
        // The initial +0 accumulator changes only the sign of a negative zero.
        let sums = operands.map { Float.zero + $0 }
        let expectedExp = sums.map { Darwin.expf(-$0) }
        let expectedSiLU = zip(sums, expectedExp).map { value, exponential in
            value / (1 + exponential)
        }
        #expect(actual.map(\.bitPattern) == expectedSiLU.map(\.bitPattern))

        // The activation result for the minimum normal input is subnormal,
        // while the 2^-125 boundary rounds back to minimum normal.
        let minimumNormal = try #require(operands.firstIndex {
            $0.bitPattern == 0x00800000
        })
        let twoToTheMinus125 = try #require(operands.firstIndex {
            $0.bitPattern == 0x01000000
        })
        let negativeMinimumNormal = try #require(operands.firstIndex {
            $0.bitPattern == 0x80800000
        })
        let negativeTwoToTheMinus125 = try #require(operands.firstIndex {
            $0.bitPattern == 0x81000000
        })
        #expect(expectedExp[minimumNormal].bitPattern == 0x3f800000)
        #expect(expectedSiLU[minimumNormal].bitPattern == 0x00400000)
        #expect(expectedSiLU[twoToTheMinus125].bitPattern == 0x00800000)
        #expect(expectedSiLU[negativeMinimumNormal].bitPattern == 0x80400000)
        #expect(expectedSiLU[negativeTwoToTheMinus125].bitPattern == 0x80800000)
    }
}

private enum QwenSourceScalarTestError: Error {
    case bufferAllocation
}

private func makeBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
    let bytes = values.count * MemoryLayout<Float>.stride
    guard let buffer = values.withUnsafeBytes({ raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: bytes, options: .storageModeShared)
    }) else {
        throw QwenSourceScalarTestError.bufferAllocation
    }
    return buffer
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
}

private func runSourceScalarConvolution(operands: [Float]) throws -> [Float] {
    let configuration = try QwenGatedDeltaNetConfiguration(
        hiddenSize: 1, keyHeadCount: 1, valueHeadCount: 1,
        keyHeadDimension: 1, valueHeadDimension: 1)
    let context = try MetalContext()
    let runtime = try QwenGatedDeltaNet(
        context: context, configuration: configuration, useOfficialSourceMath: true)
    let channelCount = configuration.convolutionChannelCount
    let tokenCount = operands.count
    let geometry = try QwenLinearAttentionGeometry(
        convolutionWidth: 4, convolutionChannelCount: channelCount,
        valueHeadCount: configuration.valueHeadCount,
        keyHeadDimension: configuration.keyHeadDimension,
        valueHeadDimension: configuration.valueHeadDimension)
    let state = try QwenLinearAttentionState(
        device: context.device, linearAttentionLayerMask: [1], geometry: geometry,
        expectedLinearLayerCount: 1)
    let update = try state.reserveUpdate(layer: 0)
    let input = operands + [Float](repeating: 0, count: (channelCount - 1) * tokenCount)
    var weights = [Float](repeating: 0, count: channelCount * configuration.convolutionWidth)
    weights[configuration.convolutionWidth - 1] = 1
    let inputBuffer = try makeBuffer(input, device: context.device)
    let weightsBuffer = try makeBuffer(weights, device: context.device)
    let outputBuffer = try makeBuffer(
        [Float](repeating: -91, count: input.count), device: context.device)
    let commandBuffer = try #require(context.queue.makeCommandBuffer())
    do {
        try runtime.encodeCausalConvolution(
            commandBuffer: commandBuffer, channelMajorInput: inputBuffer,
            weights: weightsBuffer, update: update, channelMajorOutput: outputBuffer,
            tokenCount: tokenCount)
        try state.submit(update, on: commandBuffer)
    } catch {
        try? state.abort(update)
        throw error
    }
    commandBuffer.waitUntilCompleted()
    #expect(commandBuffer.status == .completed)
    if let error = commandBuffer.error { throw error }
    return Array(readFloats(outputBuffer, count: input.count).prefix(tokenCount))
}
