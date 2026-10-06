import Foundation
import Metal
import Testing
import CryptoKit
@testable import TurboFieldfare

/// Direct production-kernel coverage for the two official source recurrence
/// cases. Inputs are synthetic FP32 tensors, independent of BF16 projection
/// outputs, and the expected checkpoints below come from pinned Transformers.
@Suite(.serialized) struct QwenBF16AttentionParityTests {
    private static let keyDimension = 128
    private static let valueDimension = 4
    private static let epsilon: Float = 1e-6
    private static let queryScale: Float = Float(pow(Double(keyDimension), -0.5))
    private static let queryDivisor: Float = Float(sqrt(Double(keyDimension)))

    private func fixtureVector(
        seed: Int, count: Int, multiplier: Int, modulus: Int,
        offset: Int, scale: Float
    ) -> [Float] {
        (0..<count).map { index in
            Float((index * multiplier + seed) % modulus - offset) * scale
        }
    }

    private var query0: [Float] {
        fixtureVector(seed: 7, count: Self.keyDimension, multiplier: 37,
                      modulus: 101, offset: 50, scale: 0.0013)
    }
    private var key0: [Float] {
        fixtureVector(seed: 11, count: Self.keyDimension, multiplier: 53,
                      modulus: 89, offset: 44, scale: 0.0017)
    }
    private var query1: [Float] {
        fixtureVector(seed: 19, count: Self.keyDimension, multiplier: 43,
                      modulus: 97, offset: 48, scale: 0.0011)
    }
    private var key1: [Float] {
        fixtureVector(seed: 23, count: Self.keyDimension, multiplier: 31,
                      modulus: 103, offset: 51, scale: 0.0015)
    }
    private var initialState: [Float] {
        fixtureVector(seed: 3, count: Self.keyDimension * Self.valueDimension,
                      multiplier: 29, modulus: 113, offset: 56, scale: 0.0007)
    }

    private func makeBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
        let bytes = values.count * MemoryLayout<Float>.stride
        guard let buffer = values.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: bytes, options: .storageModeShared)
        }) else { throw MetalError.noDevice }
        return buffer
    }

    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    private func sourceReduction(_ values: [Float]) -> Float {
        // Double accumulation is only the independent test oracle's sum. The
        // pinned official literal checks below prevent this oracle from being
        // the only evidence for the source-specific operation order.
        Float(values.reduce(Double.zero) { $0 + Double($1) })
    }

    private func normalized(_ vector: [Float]) -> [Float] {
        let squares = vector.map { $0 * $0 }
        let inverse = 1 / sqrt(sourceReduction(squares) + Self.epsilon)
        return vector.map { $0 * inverse }
    }

    private func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        sourceReduction(zip(lhs, rhs).map { $0.0 * $0.1 })
    }

    private struct AttentionReference {
        var output: [Float]
        var state: [Float]
    }

    /// One-token specialization of the official chunk source path, with a
    /// caller-supplied nonzero incoming recurrent matrix.
    private func sourceChunkToken(
        query: [Float], key: [Float], value: [Float], logDecay: Float,
        beta: Float, state incoming: [Float]
    ) -> AttentionReference {
        let q = normalized(query)
        let k = normalized(key)
        let decay = Float(Foundation.exp(Double(logDecay)))
        let qScaled = q.map { $0 * Self.queryScale }
        let attention = dot(qScaled, k)
        var result = [Float](repeating: 0, count: Self.valueDimension)
        var state = incoming
        for valueIndex in 0..<Self.valueDimension {
            let column = (0..<Self.keyDimension).map { incoming[$0 * Self.valueDimension + valueIndex] }
            let betaKey = k.map { $0 * beta }
            let decayedBetaKey = betaKey.map { $0 * decay }
            let prediction = dot(decayedBetaKey, column)
            let interAttention = dot(qScaled.map { $0 * decay }, column)
            let delta = value[valueIndex] * beta - prediction
            for keyIndex in 0..<Self.keyDimension {
                let index = keyIndex * Self.valueDimension + valueIndex
                state[index] = (incoming[index] * decay) + k[keyIndex] * delta
            }
            result[valueIndex] = interAttention + attention * delta
        }
        return AttentionReference(output: result, state: state)
    }

    /// One-token specialization of the official cached recurrent source path.
    private func sourceCachedToken(
        query: [Float], key: [Float], value: [Float], logDecay: Float,
        beta: Float, state incoming: [Float]
    ) -> AttentionReference {
        let q = normalized(query)
        let k = normalized(key)
        let qScaled = q.map { $0 / Self.queryDivisor }
        let decay = Float(Foundation.exp(Double(logDecay)))
        var state = incoming.map { $0 * decay }
        var result = [Float](repeating: 0, count: Self.valueDimension)
        for valueIndex in 0..<Self.valueDimension {
            let base = valueIndex
            let before = (0..<Self.keyDimension).map {
                state[$0 * Self.valueDimension + base]
            }
            let prediction = dot(before, k)
            let delta = (value[valueIndex] - prediction) * beta
            for keyIndex in 0..<Self.keyDimension {
                let index = keyIndex * Self.valueDimension + base
                state[index] = state[index] + k[keyIndex] * delta
            }
            let after = (0..<Self.keyDimension).map {
                state[$0 * Self.valueDimension + base]
            }
            result[valueIndex] = dot(after, qScaled)
        }
        return AttentionReference(output: result, state: state)
    }

    private func sourceGatedNorm(
        _ input: [Float], gate: [Float], weights: [Float]
    ) -> [Float] {
        let mean = sourceReduction(input.map { $0 * $0 }) / Float(input.count)
        let inverse = 1 / sqrt(mean + Self.epsilon)
        return input.indices.map { index in
            let normalized = input[index] * inverse
            let weighted = weights[index] * normalized
            let silu = gate[index] / (1 + Float(Foundation.exp(Double(-gate[index]))))
            return weighted * silu
        }
    }

    @Test func initialChunkAndCachedRecurrenceMatchPinnedCPUAndGatedNorm() async throws {
        let context = try MetalContext()
        let configuration = try QwenGatedDeltaNetConfiguration(
            hiddenSize: 1, keyHeadCount: 1, valueHeadCount: 1,
            keyHeadDimension: Self.keyDimension,
            valueHeadDimension: Self.valueDimension, epsilon: Self.epsilon)
        let runtime = try QwenGatedDeltaNet(
            context: context, configuration: configuration, useOfficialSourceMath: true)
        let geometry = try QwenLinearAttentionGeometry(
            convolutionWidth: 4, convolutionChannelCount: 1,
            valueHeadCount: 1, keyHeadDimension: Self.keyDimension,
            valueHeadDimension: Self.valueDimension)
        let stateOwner = try QwenLinearAttentionState(
            device: context.device, linearAttentionLayerMask: [1], geometry: geometry)

        let firstExpected = sourceChunkToken(
            query: query0, key: key0, value: [0.25, -0.30, 0.14, 0.41],
            logDecay: -0.11, beta: 0.37, state: initialState)
        let normWeights0: [Float] = [0.75, 1.125, -0.25, 0.5]
        let gate0: [Float] = [0.15, -0.7, 0.8, -1.2]
        let expectedGated0 = sourceGatedNorm(firstExpected.output, gate: gate0, weights: normWeights0)
        #expect(initialState.contains { $0 != 0 })

        let firstUpdate = try stateOwner.reserveUpdate(layer: 0)
        writeAttentionFloats(initialState, to: firstUpdate.recurrentMatrix)
        let firstBuffers = try recurrenceBuffers(
            context: context, query: query0, key: key0,
            value: [0.25, -0.30, 0.14, 0.41], logDecay: -0.11, beta: 0.37,
            gate: gate0, normWeights: normWeights0)
        let firstCommand = try #require(context.queue.makeCommandBuffer())
        try runtime.encodeRecurrence(
            commandBuffer: firstCommand, query: firstBuffers.query,
            key: firstBuffers.key, value: firstBuffers.value,
            logDecay: firstBuffers.logDecay, beta: firstBuffers.beta,
            update: firstUpdate, output: firstBuffers.core,
            tokenCount: 1, initialToken: true)
        try runtime.encodeGatedRMSNorm(
            commandBuffer: firstCommand, input: firstBuffers.core,
            gate: firstBuffers.gate, weights: firstBuffers.normWeights,
            output: firstBuffers.gated, tokenCount: 1)
        try stateOwner.submit(firstUpdate, on: firstCommand)
        await firstCommand.completed()
        try checkCommandBufferError(firstCommand)
        await stateOwner.waitUntilIdle()
        let actualChunk = read(firstBuffers.core, count: Self.valueDimension)
        let actualState1 = read(firstUpdate.recurrentMatrix, count: Self.keyDimension * Self.valueDimension)
        let actualGated0 = read(firstBuffers.gated, count: Self.valueDimension)
        expectAttentionClose(actualChunk, firstExpected.output, label: "initial chunk core")
        expectAttentionClose(actualState1, firstExpected.state, label: "initial chunk state")
        expectAttentionClose(actualGated0, expectedGated0, label: "source gated RMSNorm")
        expectOfficialAttentionProbes(
            actualChunk, firstExpected.output, actualState1, firstExpected.state,
            actualGated0, expectedGated0)

        let secondExpected = sourceCachedToken(
            query: query1, key: key1, value: [-0.17, 0.29, 0.36, -0.22],
            logDecay: -0.075, beta: 0.61, state: firstExpected.state)
        let normWeights1: [Float] = [0.125, -0.5, 1.25, 0.75]
        let gate1: [Float] = [-0.4, 0.55, -1.1, 0.9]
        let expectedGated1 = sourceGatedNorm(secondExpected.output, gate: gate1, weights: normWeights1)
        let secondUpdate = try stateOwner.reserveUpdate(layer: 0)
        writeAttentionFloats(firstExpected.state, to: secondUpdate.recurrentMatrix)
        let secondBuffers = try recurrenceBuffers(
            context: context, query: query1, key: key1,
            value: [-0.17, 0.29, 0.36, -0.22], logDecay: -0.075, beta: 0.61,
            gate: gate1, normWeights: normWeights1)
        let secondCommand = try #require(context.queue.makeCommandBuffer())
        try runtime.encodeRecurrence(
            commandBuffer: secondCommand, query: secondBuffers.query,
            key: secondBuffers.key, value: secondBuffers.value,
            logDecay: secondBuffers.logDecay, beta: secondBuffers.beta,
            update: secondUpdate, output: secondBuffers.core,
            tokenCount: 1, initialToken: false)
        try runtime.encodeGatedRMSNorm(
            commandBuffer: secondCommand, input: secondBuffers.core,
            gate: secondBuffers.gate, weights: secondBuffers.normWeights,
            output: secondBuffers.gated, tokenCount: 1)
        try stateOwner.submit(secondUpdate, on: secondCommand)
        await secondCommand.completed()
        try checkCommandBufferError(secondCommand)
        await stateOwner.waitUntilIdle()
        let actualCached = read(secondBuffers.core, count: Self.valueDimension)
        let actualState2 = read(secondUpdate.recurrentMatrix, count: Self.keyDimension * Self.valueDimension)
        let actualGated1 = read(secondBuffers.gated, count: Self.valueDimension)
        expectAttentionClose(actualCached, secondExpected.output, label: "cached recurrent core")
        expectAttentionClose(actualState2, secondExpected.state, label: "cached recurrent state")
        expectAttentionClose(actualGated1, expectedGated1, label: "cached source gated RMSNorm")
        expectOfficialCachedProbes(
            actualCached, secondExpected.output, actualState2, secondExpected.state,
            actualGated1, expectedGated1)

        let wrongInitialRoute = sourceCachedToken(
            query: query0, key: key0, value: [0.25, -0.30, 0.14, 0.41],
            logDecay: -0.11, beta: 0.37, state: initialState)
        #expect(maxAttentionDifference(firstExpected.output, wrongInitialRoute.output) > 0,
                "fixture must expose the source's one-ULP initial/cached order difference")
        let resetStateRoute = sourceCachedToken(
            query: query1, key: key1, value: [-0.17, 0.29, 0.36, -0.22],
            logDecay: -0.075, beta: 0.61,
            state: [Float](repeating: 0, count: initialState.count))
        #expect(maxAttentionDifference(secondExpected.output, resetStateRoute.output) > 1e-5,
                "fixture must detect loss of the nonzero committed state")
    }

    @Test func cached128RecurrenceMatchesPinnedTorchReduction() async throws {
        let headCount = 32
        let dimension = 128
        let stateCount = headCount * dimension * dimension
        let vectorCount = headCount * dimension
        let state = cached128Values(count: stateCount, seed: 0x513cef, scale: 4)
        let query = cached128Values(count: vectorCount, seed: 0x71635d, scale: 1)
        let key = cached128Values(count: vectorCount, seed: 0x3411ac, scale: 1)
        let value = cached128Values(count: vectorCount, seed: 0x2d45a3, scale: 0.25)
        let logDecay = (0..<headCount).map { -Float(($0 % 5) + 1) / 256 }
        let beta = (0..<headCount).map { Float(9 + $0 % 7) / 16 }
        #expect(state.contains { $0 != 0 }, "cached fixture must start from nonzero state")

        let expectedDataURL = try #require(Bundle.module.url(
            forResource: "cached-recurrence-expected", withExtension: "fp32.bin",
            subdirectory: "attention-cached-128"))
        let sequentialControlURL = try #require(Bundle.module.url(
            forResource: "cached-recurrence-sequential-control", withExtension: "fp32.bin",
            subdirectory: "attention-cached-128"))
        let fixtureURL = try #require(Bundle.module.url(
            forResource: "cached-recurrence", withExtension: "json",
            subdirectory: "attention-cached-128"))
        let fixture = try JSONDecoder().decode(
            Cached128AttentionFixture.self, from: Data(contentsOf: fixtureURL))
        let expectedData = try Data(contentsOf: expectedDataURL)
        let sequentialControlData = try Data(contentsOf: sequentialControlURL)
        #expect(expectedData.count == vectorCount * MemoryLayout<Float>.stride)
        #expect(sequentialControlData.count == expectedData.count)
        #expect(fixture.outputCount == vectorCount)
        #expect(SHA256.hash(data: expectedData).map { String(format: "%02x", $0) }.joined()
                    == fixture.expectedOutputSHA256)
        #expect(SHA256.hash(data: sequentialControlData)
                    .map { String(format: "%02x", $0) }.joined()
                    == fixture.sequentialControlSHA256)
        let expectedOutput = expectedData.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        let sequentialControl = sequentialControlData.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        #expect(expectedOutput.count == vectorCount)

        let configuration = try QwenGatedDeltaNetConfiguration(
            hiddenSize: vectorCount, keyHeadCount: headCount, valueHeadCount: headCount,
            keyHeadDimension: dimension, valueHeadDimension: dimension, epsilon: Self.epsilon)
        let context = try MetalContext()
        let runtime = try QwenGatedDeltaNet(
            context: context, configuration: configuration, useOfficialSourceMath: true)
        let geometry = try QwenLinearAttentionGeometry(
            convolutionWidth: 4, convolutionChannelCount: configuration.convolutionChannelCount,
            valueHeadCount: headCount, keyHeadDimension: dimension,
            valueHeadDimension: dimension)
        let stateOwner = try QwenLinearAttentionState(
            device: context.device, linearAttentionLayerMask: [1], geometry: geometry)
        let update = try stateOwner.reserveUpdate(layer: 0)
        writeAttentionFloats(state, to: update.recurrentMatrix)
        let buffers = try cached128RecurrenceBuffers(
            context: context, query: query, key: key, value: value,
            logDecay: logDecay, beta: beta)
        let command = try #require(context.queue.makeCommandBuffer())
        try runtime.encodeRecurrence(
            commandBuffer: command, query: buffers.query, key: buffers.key,
            value: buffers.value, logDecay: buffers.logDecay, beta: buffers.beta,
            update: update, output: buffers.core, tokenCount: 1, initialToken: false)
        try stateOwner.submit(update, on: command)
        await command.completed()
        try checkCommandBufferError(command)
        await stateOwner.waitUntilIdle()

        let actualOutput = readAttentionFloats(buffers.core, count: vectorCount)
        let actualState = readAttentionFloats(update.recurrentMatrix, count: stateCount)
        expectCached128Close(actualOutput, expectedOutput, label: "cached 128 Torch output")
        #expect(fixture.stateProbeIndices.count == fixture.stateProbeBits.count)
        let expectedProbeValues = fixture.stateProbeBits.map(Float.init(bitPattern:))
        expectCached128Close(
            fixture.stateProbeIndices.map { actualState[$0] }, expectedProbeValues,
            label: "cached Torch state probes")

        #expect(fixture.oldSequentialFrozenMismatches > 0)
        #expect(frozenAttentionMismatchCount(sequentialControl, expectedOutput)
                    == fixture.oldSequentialFrozenMismatches,
                "fixture must fail the prior left-to-right cached reduction")
    }
}

private struct Cached128AttentionFixture: Decodable {
    let outputCount: Int
    let expectedOutputSHA256: String
    let sequentialControlSHA256: String
    let oldSequentialFrozenMismatches: Int
    let stateProbeIndices: [Int]
    let stateProbeBits: [UInt32]
}

private func cached128Values(count: Int, seed: UInt32, scale: Float) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
        state ^= state &<< 13
        state ^= state >> 17
        state ^= state &<< 5
        return (Float(state >> 8) / 8_388_608 - 1) * scale
    }
}

private func cached128RecurrenceBuffers(
    context: MetalContext, query: [Float], key: [Float], value: [Float],
    logDecay: [Float], beta: [Float]
) throws -> AttentionBuffers {
    try AttentionBuffers(
        query: attentionBuffer(query, context: context),
        key: attentionBuffer(key, context: context),
        value: attentionBuffer(value, context: context),
        logDecay: attentionBuffer(logDecay, context: context),
        beta: attentionBuffer(beta, context: context),
        gate: attentionBuffer([Float](repeating: 0, count: value.count), context: context),
        normWeights: attentionBuffer([Float](repeating: 1, count: 128), context: context),
        core: attentionBuffer([Float](repeating: 0, count: value.count), context: context),
        gated: attentionBuffer([Float](repeating: 0, count: value.count), context: context))
}

private func expectCached128Close(_ actual: [Float], _ expected: [Float], label: String) {
    #expect(actual.count == expected.count, "\(label) count")
    #expect(frozenAttentionMismatchCount(actual, expected) == 0,
            "\(label) outside frozen Torch tolerance")
}

private func frozenAttentionMismatchCount(_ actual: [Float], _ expected: [Float]) -> Int {
    guard actual.count == expected.count else { return max(actual.count, expected.count) }
    return zip(actual, expected).filter { pair in
        !pair.0.isFinite || !pair.1.isFinite
            || abs(pair.0 - pair.1) > 1e-7 + 1e-6 * abs(pair.1)
    }.count
}

private struct AttentionBuffers {
    let query: MTLBuffer
    let key: MTLBuffer
    let value: MTLBuffer
    let logDecay: MTLBuffer
    let beta: MTLBuffer
    let gate: MTLBuffer
    let normWeights: MTLBuffer
    let core: MTLBuffer
    let gated: MTLBuffer
}

private func attentionBuffer(_ values: [Float], context: MetalContext) throws -> MTLBuffer {
    try values.withUnsafeBytes { raw in
        guard let base = raw.baseAddress,
              let buffer = context.device.makeBuffer(
                bytes: base, length: raw.count, options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        return buffer
    }
}

private func recurrenceBuffers(
    context: MetalContext, query: [Float], key: [Float], value: [Float],
    logDecay: Float, beta: Float, gate: [Float], normWeights: [Float]
) throws -> AttentionBuffers {
    try AttentionBuffers(
        query: attentionBuffer(query, context: context),
        key: attentionBuffer(key, context: context),
        value: attentionBuffer(value, context: context),
        logDecay: attentionBuffer([logDecay], context: context),
        beta: attentionBuffer([beta], context: context),
        gate: attentionBuffer(gate, context: context),
        normWeights: attentionBuffer(normWeights, context: context),
        core: attentionBuffer([Float](repeating: 0, count: value.count), context: context),
        gated: attentionBuffer([Float](repeating: 0, count: value.count), context: context))
}

private func writeAttentionFloats(_ values: [Float], to buffer: MTLBuffer) {
    values.withUnsafeBytes { raw in
        buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
    }
}

private func readAttentionFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
}

private func expectAttentionClose(
    _ actual: [Float], _ expected: [Float], label: String
) {
    #expect(actual.count == expected.count, "\(label) count")
    let failures = zip(actual, expected).filter { pair in
        !pair.0.isFinite || !pair.1.isFinite || abs(pair.0 - pair.1) > 1e-6 + 1e-5 * abs(pair.1)
    }
    #expect(failures.isEmpty, "\(label) outside frozen tolerance: \(failures.prefix(3))")
}

private func maxAttentionDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
    guard lhs.count == rhs.count else { return .infinity }
    return zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
}

private func expectOfficialAttentionProbes(
    _ actualOutput: [Float], _ referenceOutput: [Float],
    _ actualState: [Float], _ referenceState: [Float],
    _ actualGated: [Float], _ referenceGated: [Float]
) {
    let outputBits: [UInt32] = [973013230, 3129784484, 3090853120, 3094553920]
    let gatedBits: [UInt32] = [1020227748, 1048010330, 1001212984, 1004744000]
    let stateIndices = [0, 1, 2, 3, 63, 127, 128, 255, 511]
    let stateBits: [UInt32] = [
        3174487128, 3140469628, 3138308390, 999749696, 3173530749,
        1012021492, 1015916168, 999955149, 3158133171,
    ]
    let officialOutput = outputBits.map { Float(bitPattern: $0) }
    let officialGated = gatedBits.map { Float(bitPattern: $0) }
    let officialState = stateBits.map { Float(bitPattern: $0) }
    expectAttentionClose(referenceOutput, officialOutput, label: "pinned Torch initial output")
    expectAttentionClose(actualOutput, officialOutput, label: "GPU versus pinned Torch initial output")
    expectAttentionClose(referenceGated, officialGated, label: "pinned Torch initial gated output")
    expectAttentionClose(actualGated, officialGated, label: "GPU versus pinned Torch initial gated output")
    expectAttentionClose(
        stateIndices.map { referenceState[$0] }, officialState,
        label: "pinned Torch initial state probes")
    expectAttentionClose(
        stateIndices.map { actualState[$0] }, officialState,
        label: "GPU versus pinned Torch initial state probes")
}

private func expectOfficialCachedProbes(
    _ actualOutput: [Float], _ referenceOutput: [Float],
    _ actualState: [Float], _ referenceState: [Float],
    _ actualGated: [Float], _ referenceGated: [Float]
) {
    let outputBits: [UInt32] = [3111090304, 960470640, 982037876, 960179852]
    let gatedBits: [UInt32] = [998482409, 3169114533, 3198206053, 1033489333]
    let stateIndices = [0, 1, 2, 3, 63, 127, 128, 255, 511]
    let stateBits: [UInt32] = [
        3171302091, 3163397629, 3165305990, 1014363507, 3175307662,
        1008555369, 999787532, 1015563735, 3155177954,
    ]
    let officialOutput = outputBits.map { Float(bitPattern: $0) }
    let officialGated = gatedBits.map { Float(bitPattern: $0) }
    let officialState = stateBits.map { Float(bitPattern: $0) }
    expectAttentionClose(referenceOutput, officialOutput, label: "pinned Torch cached output")
    expectAttentionClose(actualOutput, officialOutput, label: "GPU versus pinned Torch cached output")
    expectAttentionClose(referenceGated, officialGated, label: "pinned Torch cached gated output")
    expectAttentionClose(actualGated, officialGated, label: "GPU versus pinned Torch cached gated output")
    expectAttentionClose(
        stateIndices.map { referenceState[$0] }, officialState,
        label: "pinned Torch cached state probes")
    expectAttentionClose(
        stateIndices.map { actualState[$0] }, officialState,
        label: "GPU versus pinned Torch cached state probes")
}
