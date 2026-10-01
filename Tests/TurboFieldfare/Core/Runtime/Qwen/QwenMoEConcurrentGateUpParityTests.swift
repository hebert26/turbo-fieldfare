import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Compares the original per-expert gate/up encoding schedule with the staged
/// concurrent encoder. Every rank owns a different activation row, while the
/// down phase still visits experts by ascending expert ID.
@Suite(.serialized) struct QwenMoEConcurrentGateUpParityTests {
    @Test func concurrentGateUpPreservesRankRowsAndOrderedDownAdd() async throws {
        for expertCount in [1, 8] {
            let result = try await runCase(expertCount: expertCount)
            #expect(result.serialActivationBits == result.concurrentActivationBits,
                    "gate/up scratch rows changed for \(expertCount) routed expert(s)")
            #expect(result.serialOutputBits == result.concurrentOutputBits,
                    "down/add output changed for \(expertCount) routed expert(s)")

            // The fixture gives each original rank a different matrix. Equal
            // rows here would allow a rank/scratch collision to pass unnoticed.
            let rowSize = 512
            let rowBits = strideRows(result.concurrentActivationBits, rowSize: rowSize)
            #expect(rowBits.allSatisfy { !$0.allSatisfy { $0 == Float.zero.bitPattern } })
            if expertCount == 8 {
                #expect(Set(rowBits).count == expertCount,
                        "all eight routed ranks must retain distinct scratch rows")
            }
        }
    }
}

private struct ScheduleResult {
    let serialActivationBits: [UInt32]
    let concurrentActivationBits: [UInt32]
    let serialOutputBits: [UInt32]
    let concurrentOutputBits: [UInt32]
}

private struct RoutedParameters {
    var hiddenSize: UInt32
    var intermediateSize: UInt32
    var scratchOffset: UInt32
    var reserved: UInt32 = 0
}

private func runCase(expertCount: Int) async throws -> ScheduleResult {
    let hiddenSize = 512
    let intermediateSize = 512
    let context = try MetalContext()
    let library = try MetalContext.privateLibrary(
        device: context.device, module: "qwen_moe", mathMode: .safe,
        mathFloatingPointFunctions: .precise, includeQwenSourceMath: true)
    let gateUpFunction = try #require(
        library.makeFunction(name: "qwen_moe_routed_gate_up_bf16_cooperative"))
    let downFunction = try #require(
        library.makeFunction(name: "qwen_moe_routed_down_add_bf16_cooperative"))
    let gateUpPipeline = try await context.device.makeComputePipelineState(function: gateUpFunction)
    let downPipeline = try await context.device.makeComputePipelineState(function: downFunction)
    #expect(gateUpPipeline.threadExecutionWidth == 32)
    #expect(downPipeline.threadExecutionWidth == 32)

    let expertIDs = expertCount == 1
        ? [5]
        : [7, 2, 6, 0, 5, 1, 4, 3]
    let orderedRanks = expertIDs.indices.sorted { expertIDs[$0] < expertIDs[$1] }
    if expertCount == 8 {
        #expect(expertIDs != orderedRanks.map { expertIDs[$0] })
    }

    let hidden = try floatBuffer(
        (0..<hiddenSize).map { index in
            let magnitude = Float((index * 17 + 11) % 97 + 1) / 97
            return index.isMultiple(of: 3) ? -magnitude : magnitude
        }, device: context.device)
    let gateUpBuffers = try (0..<expertCount).map { rank in
        try bf16Buffer(
            makeGateUpWords(rank: rank, hiddenSize: hiddenSize,
                            intermediateSize: intermediateSize), device: context.device)
    }
    let downBuffers = try (0..<expertCount).map { rank in
        try bf16Buffer(
            makeDownWords(rank: rank, hiddenSize: hiddenSize,
                          intermediateSize: intermediateSize), device: context.device)
    }
    let routeWeights = try floatBuffer(
        normalizedWeights(count: expertCount), device: context.device)

    let serialActivation = try zeroFloatBuffer(
        count: expertCount * intermediateSize, device: context.device)
    let concurrentActivation = try zeroFloatBuffer(
        count: expertCount * intermediateSize, device: context.device)
    let serialOutput = try zeroFloatBuffer(count: hiddenSize, device: context.device)
    let concurrentOutput = try zeroFloatBuffer(count: hiddenSize, device: context.device)

    let serialCommand = try #require(context.queue.makeCommandBuffer())
    encodeGateUps(
        command: serialCommand, pipeline: gateUpPipeline, hidden: hidden,
        gateUpBuffers: gateUpBuffers, activation: serialActivation,
        orderedRanks: orderedRanks, intermediateSize: intermediateSize,
        hiddenSize: hiddenSize, concurrent: false)
    encodeDownAdds(
        command: serialCommand, pipeline: downPipeline, activation: serialActivation,
        downBuffers: downBuffers, output: serialOutput, routeWeights: routeWeights,
        orderedRanks: orderedRanks, intermediateSize: intermediateSize,
        hiddenSize: hiddenSize)
    serialCommand.commit()

    let concurrentCommand = try #require(context.queue.makeCommandBuffer())
    encodeGateUps(
        command: concurrentCommand, pipeline: gateUpPipeline, hidden: hidden,
        gateUpBuffers: gateUpBuffers, activation: concurrentActivation,
        orderedRanks: orderedRanks, intermediateSize: intermediateSize,
        hiddenSize: hiddenSize, concurrent: true)
    encodeDownAdds(
        command: concurrentCommand, pipeline: downPipeline,
        activation: concurrentActivation, downBuffers: downBuffers,
        output: concurrentOutput, routeWeights: routeWeights,
        orderedRanks: orderedRanks, intermediateSize: intermediateSize,
        hiddenSize: hiddenSize)
    concurrentCommand.commit()

    _ = await serialCommand.completed()
    _ = await concurrentCommand.completed()
    try checkCommandBufferError(serialCommand)
    try checkCommandBufferError(concurrentCommand)

    return ScheduleResult(
        serialActivationBits: readFloats(serialActivation,
                                         count: expertCount * intermediateSize),
        concurrentActivationBits: readFloats(concurrentActivation,
                                             count: expertCount * intermediateSize),
        serialOutputBits: readFloats(serialOutput, count: hiddenSize),
        concurrentOutputBits: readFloats(concurrentOutput, count: hiddenSize))
}

private func encodeGateUps(
    command: MTLCommandBuffer, pipeline: MTLComputePipelineState,
    hidden: MTLBuffer, gateUpBuffers: [MTLBuffer], activation: MTLBuffer,
    orderedRanks: [Int], intermediateSize: Int, hiddenSize: Int,
    concurrent: Bool
) {
    let encoder = command.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial)!
    defer { encoder.endEncoding() }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(hidden, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
    encoder.setBuffer(activation, offset: 0, index: QwenMetalBufferIndex.scratch.rawValue)
    for rank in orderedRanks {
        var parameters = RoutedParameters(
            hiddenSize: UInt32(hiddenSize), intermediateSize: UInt32(intermediateSize),
            scratchOffset: UInt32(rank * intermediateSize))
        encoder.setBytes(&parameters, length: MemoryLayout<RoutedParameters>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(gateUpBuffers[rank], offset: 0,
                          index: QwenMetalBufferIndex.weights.rawValue)
        encoder.useResource(gateUpBuffers[rank], usage: .read)
        encoder.dispatchThreadgroups(
            MTLSize(width: intermediateSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}

private func encodeDownAdds(
    command: MTLCommandBuffer, pipeline: MTLComputePipelineState,
    activation: MTLBuffer, downBuffers: [MTLBuffer], output: MTLBuffer,
    routeWeights: MTLBuffer, orderedRanks: [Int], intermediateSize: Int,
    hiddenSize: Int
) {
    for rank in orderedRanks {
        let encoder = command.makeComputeCommandEncoder()!
        var parameters = RoutedParameters(
            hiddenSize: UInt32(hiddenSize), intermediateSize: UInt32(intermediateSize),
            scratchOffset: UInt32(rank * intermediateSize))
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<RoutedParameters>.stride,
                         index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(activation, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
        encoder.setBuffer(downBuffers[rank], offset: 0,
                          index: QwenMetalBufferIndex.weights.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        encoder.setBuffer(routeWeights,
                          offset: rank * MemoryLayout<Float>.stride,
                          index: QwenMetalBufferIndex.state.rawValue)
        encoder.useResource(downBuffers[rank], usage: .read)
        encoder.dispatchThreadgroups(
            MTLSize(width: hiddenSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

private func makeGateUpWords(rank: Int, hiddenSize: Int, intermediateSize: Int) -> [UInt16] {
    (0..<(2 * intermediateSize * hiddenSize)).map { index in
        let value = Float((index * 13 + rank * 29) % 61 + 3) / 64
        let signed = ((index + rank) & 1) == 0 ? value : -value
        return UInt16(signed.bitPattern >> 16)
    }
}

private func makeDownWords(rank: Int, hiddenSize: Int, intermediateSize: Int) -> [UInt16] {
    (0..<(hiddenSize * intermediateSize)).map { index in
        let value = Float((index * 7 + rank * 37) % 53 + 2) / 64
        let signed = ((index + rank) % 3) == 0 ? -value : value
        return UInt16(signed.bitPattern >> 16)
    }
}

private func normalizedWeights(count: Int) -> [Float] {
    if count == 1 { return [1] }
    let raw = (0..<count).map { Float($0 + 1) }
    let total = raw.reduce(0, +)
    return raw.map { $0 / total }
}

private func floatBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
    try values.withUnsafeBytes { raw in
        guard let base = raw.baseAddress,
              let buffer = device.makeBuffer(bytes: base, length: raw.count,
                                             options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        return buffer
    }
}

private func zeroFloatBuffer(count: Int, device: MTLDevice) throws -> MTLBuffer {
    try floatBuffer([Float](repeating: 0, count: count), device: device)
}

private func bf16Buffer(_ values: [UInt16], device: MTLDevice) throws -> MTLBuffer {
    try values.withUnsafeBytes { raw in
        guard let base = raw.baseAddress,
              let buffer = device.makeBuffer(bytes: base, length: raw.count,
                                             options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        return buffer
    }
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return (0..<count).map { pointer[$0].bitPattern }
}

private func strideRows(_ values: [UInt32], rowSize: Int) -> [[UInt32]] {
    stride(from: 0, to: values.count, by: rowSize).map {
        Array(values[$0..<min($0 + rowSize, values.count)])
    }
}
