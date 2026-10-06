import Metal
import Testing
@testable import TurboFieldfare

/// The old source kernel remains in the candidate library. This test invokes
/// it directly, then compares it with the cached 128-column production route
/// using the same deterministic one-token input and nonzero state.
@Suite(.serialized)
struct QwenCachedRecurrenceGPUParityTests {
    @Test func cachedOneTokenMultiHeadKernelMatchesOldSourceKernelBitwise() async throws {
        let context = try MetalContext()
        let headCount = 3
        let vectorCount = headCount * 128
        let configuration = try QwenGatedDeltaNetConfiguration(
            hiddenSize: vectorCount, keyHeadCount: headCount, valueHeadCount: headCount,
            keyHeadDimension: 128, valueHeadDimension: 128)
        let runtime = try QwenGatedDeltaNet(
            context: context, configuration: configuration, useOfficialSourceMath: true)
        let geometry = try QwenLinearAttentionGeometry(
            convolutionWidth: 4,
            convolutionChannelCount: configuration.convolutionChannelCount,
            valueHeadCount: headCount, keyHeadDimension: 128, valueHeadDimension: 128)
        let newOwner = try QwenLinearAttentionState(
            device: context.device, linearAttentionLayerMask: [1], geometry: geometry)
        let oldOwner = try QwenLinearAttentionState(
            device: context.device, linearAttentionLayerMask: [1], geometry: geometry)
        let state = deterministicState(headCount: headCount)
        let query = deterministicVector(
            seed: 17, count: vectorCount, includesNegativeAndZero: true)
        let key = deterministicVector(
            seed: 31, count: vectorCount, includesNegativeAndZero: true)
        let value = deterministicVector(
            seed: 47, count: vectorCount, includesNegativeAndZero: true)
        let logDecay: [Float] = [-0.125, -0.0625, -0.25]
        let beta: [Float] = [0.375, 0.625, 0.1875]

        let newUpdate = try newOwner.reserveUpdate(layer: 0)
        let oldUpdate = try oldOwner.reserveUpdate(layer: 0)
        writeFloats(state, to: newUpdate.recurrentMatrix)
        writeFloats(state, to: oldUpdate.recurrentMatrix)
        let queryBuffer = try floatBuffer(query, device: context.device)
        let keyBuffer = try floatBuffer(key, device: context.device)
        let valueBuffer = try floatBuffer(value, device: context.device)
        let decayBuffer = try floatBuffer(logDecay, device: context.device)
        let betaBuffer = try floatBuffer(beta, device: context.device)
        let newOutput = try emptyFloatBuffer(count: vectorCount, device: context.device)
        let oldOutput = try emptyFloatBuffer(count: vectorCount, device: context.device)

        let newCommand = try #require(context.queue.makeCommandBuffer())
        try runtime.encodeRecurrence(
            commandBuffer: newCommand, query: queryBuffer, key: keyBuffer,
            value: valueBuffer, logDecay: decayBuffer, beta: betaBuffer,
            update: newUpdate, output: newOutput, tokenCount: 1, initialToken: false)
        try newOwner.submit(newUpdate, on: newCommand)

        let oldCommand = try #require(context.queue.makeCommandBuffer())
        try encodeOldSourceKernel(
            commandBuffer: oldCommand, context: context, query: queryBuffer,
            key: keyBuffer, value: valueBuffer, logDecay: decayBuffer,
            beta: betaBuffer, state: oldUpdate.recurrentMatrix, output: oldOutput,
            headCount: headCount)
        try oldOwner.submit(oldUpdate, on: oldCommand)

        await newCommand.completed()
        await oldCommand.completed()
        #expect(newCommand.status == .completed)
        #expect(oldCommand.status == .completed)
        if let error = newCommand.error { throw error }
        if let error = oldCommand.error { throw error }
        await newOwner.waitUntilIdle()
        await oldOwner.waitUntilIdle()

        let newOutputBits = readFloats(newOutput, count: vectorCount).map(\.bitPattern)
        let oldOutputBits = readFloats(oldOutput, count: vectorCount).map(\.bitPattern)
        let newStateBits = readFloats(newUpdate.recurrentMatrix, count: state.count).map(\.bitPattern)
        let oldStateBits = readFloats(oldUpdate.recurrentMatrix, count: state.count).map(\.bitPattern)
        #expect(newOutputBits == oldOutputBits)
        #expect(newStateBits == oldStateBits)
        #expect(newOutputBits.contains { $0 != 0 })
        #expect(newStateBits.contains { $0 != Float.zero.bitPattern })
    }
}

private struct SourceRecurrenceParameters {
    var tokenCount: UInt32
    var headCount: UInt32
    var keyDimension: UInt32
    var valueDimension: UInt32
    var epsilon: Float
    var initialToken: UInt32
    var queryScale: Float
    var queryDivisor: Float
}

private func encodeOldSourceKernel(
    commandBuffer: MTLCommandBuffer, context: MetalContext,
    query: MTLBuffer, key: MTLBuffer, value: MTLBuffer,
    logDecay: MTLBuffer, beta: MTLBuffer, state: MTLBuffer, output: MTLBuffer,
    headCount: Int
) throws {
    let library = try MetalContext.privateLibrary(
        device: context.device, module: "qwen_linear_attention", mathMode: .safe,
        mathFloatingPointFunctions: .precise, includeQwenSourceMath: true)
    let function = try #require(library.makeFunction(name: "qwen_source_linear_recurrence_fp32"))
    let pipeline = try context.device.makeComputePipelineState(function: function)
    var parameters = SourceRecurrenceParameters(
        tokenCount: 1, headCount: UInt32(headCount), keyDimension: 128, valueDimension: 128,
        epsilon: 1e-6, initialToken: 0,
        queryScale: Float(pow(Double(128), -0.5)),
        queryDivisor: Float(sqrt(Double(128))))
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBytes(
        &parameters, length: MemoryLayout<SourceRecurrenceParameters>.stride,
        index: QwenMetalBufferIndex.parameters.rawValue)
    encoder.setBuffer(query, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
    encoder.setBuffer(key, offset: 0, index: QwenMetalBufferIndex.weights.rawValue)
    encoder.setBuffer(value, offset: 0, index: QwenMetalBufferIndex.scales.rawValue)
    encoder.setBuffer(logDecay, offset: 0, index: QwenMetalBufferIndex.biases.rawValue)
    encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
    encoder.setBuffer(beta, offset: 0, index: QwenMetalBufferIndex.scratch.rawValue)
    encoder.setBuffer(state, offset: 0, index: QwenMetalBufferIndex.state.rawValue)
    encoder.dispatchThreads(
        MTLSize(width: headCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: headCount, height: 1, depth: 1))
    encoder.endEncoding()
}

private func deterministicVector(seed: Int, count: Int, includesNegativeAndZero: Bool) -> [Float] {
    (0..<count).map { index in
        if includesNegativeAndZero, index % 17 == 0 { return 0 }
        let magnitude = Float((index * 13 + seed) % 29 + 1) / 64
        return index % 3 == 0 ? -magnitude : magnitude
    }
}

private func deterministicState(headCount: Int) -> [Float] {
    (0..<(headCount * 128 * 128)).map { index in
        if index % 23 == 0 { return 0 }
        let magnitude = Float((index * 7 + 5) % 31 + 1) / 128
        return index % 5 == 0 ? -magnitude : magnitude
    }
}

private func floatBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
    let bytes = values.count * MemoryLayout<Float>.stride
    return try values.withUnsafeBytes { raw in
        guard let base = raw.baseAddress,
              let buffer = device.makeBuffer(bytes: base, length: bytes, options: .storageModeShared)
        else { throw MetalError.noDevice }
        return buffer
    }
}

private func emptyFloatBuffer(count: Int, device: MTLDevice) throws -> MTLBuffer {
    try floatBuffer([Float](repeating: 0, count: count), device: device)
}

private func writeFloats(_ values: [Float], to buffer: MTLBuffer) {
    values.withUnsafeBytes { raw in
        buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
    }
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
}
