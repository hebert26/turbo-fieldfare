import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenGatedDeltaNetTests {
    @Test func validatesTinyAndOfficialGeometryAndRejectsMalformedConfiguration() throws {
        let tiny = try tinyConfiguration()
        #expect(tiny.hiddenSize == 5)
        #expect(tiny.keyDimension == 2)
        #expect(tiny.valueDimension == 6)
        #expect(tiny.convolutionChannelCount == 10)
        #expect(tiny.headsPerKeyHead == 2)

        let official = try QwenGatedDeltaNetConfiguration.official()
        #expect(official.hiddenSize == 2_048)
        #expect(official.keyHeadCount == 16)
        #expect(official.valueHeadCount == 32)
        #expect(official.convolutionWidth == 4)
        #expect(throws: QwenGatedDeltaNetError.invalidConfiguration(
            field: "convolutionWidth", value: 3)) {
            try QwenGatedDeltaNetConfiguration(
                hiddenSize: 5, keyHeadCount: 1, valueHeadCount: 2,
                keyHeadDimension: 2, valueHeadDimension: 3, convolutionWidth: 3)
        }
        #expect(throws: QwenGatedDeltaNetError.invalidConfiguration(
            field: "valueHeadCount/keyHeadCount", value: 3)) {
            try QwenGatedDeltaNetConfiguration(
                hiddenSize: 5, keyHeadCount: 2, valueHeadCount: 3,
                keyHeadDimension: 2, valueHeadDimension: 3)
        }
        #expect(throws: QwenGatedDeltaNetError.invalidFloatingPointConfiguration(field: "epsilon")) {
            try QwenGatedDeltaNetConfiguration(
                hiddenSize: 5, keyHeadCount: 1, valueHeadCount: 1,
                keyHeadDimension: 2, valueHeadDimension: 3, epsilon: 0)
        }
    }

    @Test func frozenP3CausalConvolutionMatchesPinnedOracleForFiveTokensAndPartitions() throws {
        let section = try fixtureSection("causalConvolution")
        let config = try QwenGatedDeltaNetConfiguration(
            hiddenSize: 1, keyHeadCount: 1, valueHeadCount: 1,
            keyHeadDimension: 1, valueHeadDimension: 4)
        let channelCount = config.convolutionChannelCount
        let tokenCount = 5
        let channelMajorInput = try tensor(section, "input")
        let weights = try tensor(section, "weight")
        let expectedOutput = channelMajorToTokenMajor(
            try tensor(section, "fullOutput"), channels: channelCount, tokens: tokenCount)
        let expectedHistory = try tensor(section, "finalHistory")
        let input = channelMajorToTokenMajor(channelMajorInput, channels: channelCount, tokens: tokenCount)
        let initialHistory = [Float](repeating: 0, count: channelCount * 4)

        let full = try QwenGatedDeltaNet.causalConvolution(
            configuration: config, input: input, weights: weights, initialHistory: initialHistory)
        expectClose(full.output, expectedOutput, tolerance: 1e-5)
        expectClose(full.finalHistory, expectedHistory, tolerance: 1e-5)

        for partition in [[1, 1, 1, 1, 1], [3, 2], [4, 1], [2, 2, 1]] {
            var history = initialHistory
            var output: [Float] = []
            var start = 0
            for count in partition {
                let end = start + count
                let piece = try QwenGatedDeltaNet.causalConvolution(
                    configuration: config,
                    input: Array(input[(start * channelCount)..<(end * channelCount)]),
                    weights: weights,
                    initialHistory: history)
                output += piece.output
                history = piece.finalHistory
                start = end
            }
            #expect(start == tokenCount)
            expectClose(output, expectedOutput, tolerance: 1e-5)
            expectClose(history, expectedHistory, tolerance: 2e-5)
        }

        let empty = try QwenGatedDeltaNet.causalConvolution(
            configuration: config, input: [], weights: weights, initialHistory: initialHistory)
        #expect(empty.output.isEmpty)
        #expect(empty.finalHistory == initialHistory)
    }

    @Test func frozenP3RecurrentOneTokenChunkedAndSplitStatesMatchPinnedFP32Oracle() throws {
        let section = try fixtureSection("linearAttention")
        let config = try QwenGatedDeltaNetConfiguration(
            hiddenSize: 32, keyHeadCount: 2, valueHeadCount: 2,
            keyHeadDimension: 4, valueHeadDimension: 4)
        let query = try tensor(section, "query")
        let key = try tensor(section, "key")
        let value = try tensor(section, "value")
        let logDecay = try tensor(section, "logDecayFP32")
        let beta = try tensor(section, "beta")
        let expectedOutput = try tensor(section, "tokenAtATimeOutput")
        let expectedState = try tensor(section, "tokenAtATimeFinalState")
        let queryWidth = config.valueHeadCount * config.keyHeadDimension
        let valueWidth = config.valueHeadCount * config.valueHeadDimension
        let scalarWidth = config.valueHeadCount
        let initialState = [Float](repeating: 0, count: expectedState.count)

        let one = try QwenGatedDeltaNet.recurrent(
            configuration: config,
            query: Array(query.prefix(queryWidth)), key: Array(key.prefix(queryWidth)),
            value: Array(value.prefix(valueWidth)), logDecay: Array(logDecay.prefix(scalarWidth)),
            beta: Array(beta.prefix(scalarWidth)), initialState: initialState, chunkSize: 1)
        expectClose(one.output, Array(try tensor(section, "oneTokenCase", nested: "expectedOutput")), tolerance: 1e-5)

        for chunkSize in [1, 2, 3, 5, 9] {
            let result = try QwenGatedDeltaNet.recurrent(
                configuration: config, query: query, key: key, value: value,
                logDecay: logDecay, beta: beta, initialState: initialState, chunkSize: chunkSize)
            expectClose(result.output, expectedOutput, tolerance: 1e-5)
            expectClose(result.finalState, expectedState, tolerance: 2e-5)
        }

        for partition in [[1, 1, 1, 1, 1], [3, 2], [2, 2, 1], [4, 1]] {
            var state = initialState
            var output: [Float] = []
            var start = 0
            for count in partition {
                let end = start + count
                let piece = try QwenGatedDeltaNet.recurrent(
                    configuration: config,
                    query: Array(query[(start * queryWidth)..<(end * queryWidth)]),
                    key: Array(key[(start * queryWidth)..<(end * queryWidth)]),
                    value: Array(value[(start * valueWidth)..<(end * valueWidth)]),
                    logDecay: Array(logDecay[(start * scalarWidth)..<(end * scalarWidth)]),
                    beta: Array(beta[(start * scalarWidth)..<(end * scalarWidth)]),
                    initialState: state, chunkSize: 2)
                output += piece.output
                state = piece.finalState
                start = end
            }
            #expect(start == 5)
            expectClose(output, expectedOutput, tolerance: 1e-5)
            expectClose(state, expectedState, tolerance: 2e-5)
        }

        let empty = try QwenGatedDeltaNet.recurrent(
            configuration: config, query: [], key: [], value: [], logDecay: [], beta: [],
            initialState: initialState, chunkSize: 2)
        #expect(empty.output.isEmpty)
        #expect(empty.finalState == initialState)
    }

    @Test func fp16DecayMutationIsRejectedByAnIndependentStrongNegativeControl() throws {
        let configuration = try QwenGatedDeltaNetConfiguration(
            hiddenSize: 4, keyHeadCount: 1, valueHeadCount: 1,
            keyHeadDimension: 2, valueHeadDimension: 2)
        let tokenCount = 64
        let query: [Float] = Array(repeating: Float(1), count: tokenCount * 2).enumerated().map { $0.offset.isMultiple(of: 2) ? Float(1) : Float(0) }
        let key = query
        let value = [Float](repeating: 0, count: tokenCount * 2)
        let logDecay = [Float](repeating: -0.013579, count: tokenCount)
        let beta = [Float](repeating: 0, count: tokenCount)
        let initialState: [Float] = [4, -3, 1.5, -2]

        let expected = independentRecurrence(
            configuration: configuration, query: query, key: key, value: value,
            logDecay: logDecay, beta: beta, initialState: initialState)
        // This is the plausible precision mutant: quantize the logarithmic decay
        // to Float16 before exponentiation. The chosen non-representable decay and
        // 64-token history make its separation materially exceed both P3 budgets.
        let fp16Decay = logDecay.map { Float(Float16($0)) }
        let mutated = independentRecurrence(
            configuration: configuration, query: query, key: key, value: value,
            logDecay: fp16Decay, beta: beta, initialState: initialState)
        let outputSeparation = maxDifference(expected.output, mutated.output)
        let stateSeparation = maxDifference(expected.state, mutated.state)
        #expect(outputSeparation > 1e-4)
        #expect(stateSeparation > 1e-4)

        let actual = try QwenGatedDeltaNet.recurrent(
            configuration: configuration, query: query, key: key, value: value,
            logDecay: logDecay, beta: beta, initialState: initialState, chunkSize: 7)
        expectClose(actual.output, expected.output, tolerance: 1e-5)
        expectClose(actual.finalState, expected.state, tolerance: 2e-5)
    }

    @Test func frozenP3GatedRMSNormUsesDirectWeightAndSiLUGate() throws {
        let section = try fixtureSection("linearOutputGate")
        let config = try QwenGatedDeltaNetConfiguration(
            hiddenSize: 4, keyHeadCount: 1, valueHeadCount: 1,
            keyHeadDimension: 1, valueHeadDimension: 4)
        let input = try tensor(section, "input")
        let gate = try tensor(section, "gate")
        let weights = try tensor(section, "weight")
        let expected = try tensor(section, "output")
        let actual = try QwenGatedDeltaNet.gatedRMSNorm(
            configuration: config, input: input, gate: gate, weights: weights)
        expectClose(actual, expected, tolerance: 1e-5)
        let sigmoidGate = gate.map(sigmoid)
        #expect(maxDifference(actual, zip(input, sigmoidGate).map(*)) > 1e-3)
    }

    @Test func independentEndToEndOracleCoversPinnedProjectionOrderAndIntermediateValues() throws {
        let configuration = try tinyConfiguration()
        let tokenCount = 5
        let input = synthetic(count: tokenCount * configuration.hiddenSize, phase: 0.2, scale: 0.11)
        let weights = makeWeights(configuration)
        let initial = QwenLinearAttentionLayerState(
            convolutionHistory: synthetic(
                count: configuration.convolutionChannelCount * 4, phase: -0.4, scale: 0.03),
            recurrentMatrix: synthetic(
                count: configuration.valueHeadCount * configuration.keyHeadDimension
                    * configuration.valueHeadDimension, phase: 0.7, scale: 0.02))
        let expected = independentEvaluate(
            configuration: configuration, input: input, weights: weights,
            initialState: initial)
        let actual = try QwenGatedDeltaNet.evaluate(
            configuration: configuration, input: input, weights: weights,
            initialState: initial, chunkSize: 2)
        expectClose(actual.qkvProjection, expected.qkvProjection, tolerance: 1e-5)
        expectClose(actual.convolvedQKV, expected.convolvedQKV, tolerance: 1e-5)
        expectClose(actual.query, expected.query, tolerance: 1e-5)
        expectClose(actual.key, expected.key, tolerance: 1e-5)
        expectClose(actual.value, expected.value, tolerance: 1e-5)
        expectClose(actual.gate, expected.gate, tolerance: 1e-5)
        expectClose(actual.beta, expected.beta, tolerance: 1e-5)
        expectClose(actual.logDecay, expected.logDecay, tolerance: 1e-5)
        expectClose(actual.recurrentOutput, expected.recurrentOutput, tolerance: 1e-5)
        expectClose(actual.normalizedGatedOutput, expected.normalizedGatedOutput, tolerance: 1e-5)
        expectClose(actual.output, expected.output, tolerance: 1e-5)
        expectClose(actual.finalState.convolutionHistory, expected.finalState.convolutionHistory, tolerance: 2e-5)
        expectClose(actual.finalState.recurrentMatrix, expected.finalState.recurrentMatrix, tolerance: 2e-5)

        var tokenOutputs: [Float] = []
        var state = initial
        for token in 0..<tokenCount {
            let stepInput = Array(input[(token * configuration.hiddenSize)..<((token + 1) * configuration.hiddenSize)])
            let step = try QwenGatedDeltaNet.evaluate(
                configuration: configuration, input: stepInput, weights: weights,
                initialState: state, chunkSize: 1)
            tokenOutputs += step.output
            state = step.finalState
        }
        expectClose(tokenOutputs, actual.output, tolerance: 2e-5)
        expectClose(state.convolutionHistory, actual.finalState.convolutionHistory, tolerance: 2e-5)
        expectClose(state.recurrentMatrix, actual.finalState.recurrentMatrix, tolerance: 2e-5)

        let empty = try QwenGatedDeltaNet.evaluate(
            configuration: configuration, input: [], weights: weights,
            initialState: initial, chunkSize: 2)
        #expect(empty.output.isEmpty)
        #expect(empty.finalState == initial)
    }

    @Test func realGPUPipelinesExecuteForAwkwardLengthAndMatchIndependentCPUResults() throws {
        let context = try MetalContext()
        let configuration = try tinyConfiguration()
        let runtime = try QwenGatedDeltaNet(context: context, configuration: configuration)
        let tokenCount = 5
        let channelCount = configuration.convolutionChannelCount
        let input = synthetic(count: tokenCount * channelCount, phase: 0.9, scale: 0.08)
        let convWeights = synthetic(count: channelCount * 4, phase: 1.4, scale: 0.04)
        let query = synthetic(
            count: tokenCount * configuration.valueHeadCount * configuration.keyHeadDimension,
            phase: 0.2, scale: 0.12)
        let key = synthetic(
            count: tokenCount * configuration.valueHeadCount * configuration.keyHeadDimension,
            phase: 0.5, scale: 0.12)
        let value = synthetic(
            count: tokenCount * configuration.valueHeadCount * configuration.valueHeadDimension,
            phase: 0.8, scale: 0.12)
        let logDecay = synthetic(count: tokenCount * configuration.valueHeadCount, phase: 1.1, scale: 0.1)
        let beta = synthetic(count: tokenCount * configuration.valueHeadCount, phase: 1.7, scale: 0.1).map(sigmoid)
        let gate = synthetic(
            count: tokenCount * configuration.valueHeadCount * configuration.valueHeadDimension,
            phase: 2.1, scale: 0.12)
        let normWeights = synthetic(count: configuration.valueHeadDimension, phase: 2.5, scale: 0.03).map { 1 + $0 }
        let geometry = try QwenLinearAttentionGeometry(
            convolutionWidth: 4, convolutionChannelCount: channelCount,
            valueHeadCount: configuration.valueHeadCount,
            keyHeadDimension: configuration.keyHeadDimension,
            valueHeadDimension: configuration.valueHeadDimension)
        let state = try QwenLinearAttentionState(
            device: context.device, linearAttentionLayerMask: [1], geometry: geometry,
            expectedLinearLayerCount: 1)
        let update = try state.reserveUpdate(layer: 0)
        let qkvInput = try floatBuffer(input, device: context.device)
        let layoutOutput = try emptyFloatBuffer(count: input.count, device: context.device)
        let convWeightBuffer = try floatBuffer(convWeights, device: context.device)
        let convOutput = try emptyFloatBuffer(count: input.count, device: context.device)
        let queryBuffer = try floatBuffer(query, device: context.device)
        let keyBuffer = try floatBuffer(key, device: context.device)
        let valueBuffer = try floatBuffer(value, device: context.device)
        let decayBuffer = try floatBuffer(logDecay, device: context.device)
        let betaBuffer = try floatBuffer(beta, device: context.device)
        let recurrenceOutput = try emptyFloatBuffer(
            count: value.count, device: context.device)
        let gateBuffer = try floatBuffer(gate, device: context.device)
        let normBuffer = try floatBuffer(normWeights, device: context.device)
        let gatedOutput = try emptyFloatBuffer(count: value.count, device: context.device)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        do {
            try runtime.encodeLayout(
                commandBuffer: commandBuffer, input: qkvInput, output: layoutOutput,
                tokenCount: tokenCount, channelCount: channelCount, tokenToChannel: true)
            try runtime.encodeCausalConvolution(
                commandBuffer: commandBuffer, channelMajorInput: layoutOutput,
                weights: convWeightBuffer, update: update, channelMajorOutput: convOutput,
                tokenCount: tokenCount)
            try runtime.encodeRecurrence(
                commandBuffer: commandBuffer, query: queryBuffer, key: keyBuffer,
                value: valueBuffer, logDecay: decayBuffer, beta: betaBuffer,
                update: update, output: recurrenceOutput, tokenCount: tokenCount)
            try runtime.encodeGatedRMSNorm(
                commandBuffer: commandBuffer, input: recurrenceOutput, gate: gateBuffer,
                weights: normBuffer, output: gatedOutput, tokenCount: tokenCount)
            try state.submit(update, on: commandBuffer)
        } catch {
            try? state.abort(update)
            throw error
        }
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        if let error = commandBuffer.error { throw error }

        let cpuConv = try QwenGatedDeltaNet.causalConvolution(
            configuration: configuration, input: input, weights: convWeights,
            initialHistory: [Float](repeating: 0, count: channelCount * 4))
        let cpuRecurrence = try QwenGatedDeltaNet.recurrent(
            configuration: configuration, query: query, key: key, value: value,
            logDecay: logDecay, beta: beta,
            initialState: [Float](repeating: 0, count: configuration.valueHeadCount
                * configuration.keyHeadDimension * configuration.valueHeadDimension), chunkSize: 2)
        let cpuGated = try QwenGatedDeltaNet.gatedRMSNorm(
            configuration: configuration, input: cpuRecurrence.output,
            gate: gate, weights: normWeights)
        expectClose(
            readFloats(layoutOutput, count: input.count),
            tokenMajorToChannelMajor(input, channels: channelCount, tokens: tokenCount), tolerance: 1e-5)
        expectClose(
            readFloats(convOutput, count: input.count),
            tokenMajorToChannelMajor(cpuConv.output, channels: channelCount, tokens: tokenCount), tolerance: 1e-5)
        expectClose(readFloats(recurrenceOutput, count: value.count), cpuRecurrence.output, tolerance: 2e-5)
        expectClose(readFloats(gatedOutput, count: value.count), cpuGated, tolerance: 2e-5)
        let committed = try state.layerState(0)
        expectClose(committed.convolutionHistory, cpuConv.finalHistory, tolerance: 2e-5)
        expectClose(committed.recurrentMatrix, cpuRecurrence.finalState, tolerance: 2e-5)
    }
}

private enum TestResourceError: Error { case bufferAllocation }

private func tinyConfiguration() throws -> QwenGatedDeltaNetConfiguration {
    try QwenGatedDeltaNetConfiguration(
        hiddenSize: 5, keyHeadCount: 1, valueHeadCount: 2,
        keyHeadDimension: 2, valueHeadDimension: 3)
}

private func synthetic(count: Int, phase: Float, scale: Float) -> [Float] {
    (0..<count).map { sin(Float($0) * 0.37 + phase) * scale }
}

private func makeWeights(_ configuration: QwenGatedDeltaNetConfiguration) -> QwenGatedDeltaNetWeights {
    let c = configuration.convolutionChannelCount
    let h = configuration.hiddenSize
    let v = configuration.valueDimension
    let heads = configuration.valueHeadCount
    return QwenGatedDeltaNetWeights(
        qkvProjection: synthetic(count: c * h, phase: 0.11, scale: 0.15),
        zProjection: synthetic(count: v * h, phase: 0.31, scale: 0.15),
        bProjection: synthetic(count: heads * h, phase: 0.51, scale: 0.15),
        aProjection: synthetic(count: heads * h, phase: 0.71, scale: 0.15),
        convolution: synthetic(count: c * 4, phase: 0.91, scale: 0.08),
        timeStepBias: synthetic(count: heads, phase: 1.11, scale: 0.05),
        aLog: synthetic(count: heads, phase: 1.31, scale: 0.05),
        normalization: synthetic(count: configuration.valueHeadDimension, phase: 1.51, scale: 0.03).map { 1 + $0 },
        outputProjection: synthetic(count: h * v, phase: 1.71, scale: 0.15))
}

private struct IndependentResult {
    let qkvProjection, convolvedQKV, query, key, value, gate, beta, logDecay: [Float]
    let recurrentOutput, normalizedGatedOutput, output: [Float]
    let finalState: QwenLinearAttentionLayerState
}

private func independentEvaluate(
    configuration: QwenGatedDeltaNetConfiguration,
    input: [Float],
    weights: QwenGatedDeltaNetWeights,
    initialState: QwenLinearAttentionLayerState
) -> IndependentResult {
    let tokenCount = input.count / configuration.hiddenSize
    let channels = configuration.convolutionChannelCount
    let keyWidth = configuration.keyDimension
    let valueWidth = configuration.valueDimension
    let qkv = independentProject(input, weights: weights.qkvProjection,
                                 rows: channels, columns: configuration.hiddenSize)
    let gate = independentProject(input, weights: weights.zProjection,
                                  rows: valueWidth, columns: configuration.hiddenSize)
    let rawBeta = independentProject(input, weights: weights.bProjection,
                                     rows: configuration.valueHeadCount, columns: configuration.hiddenSize)
    let rawA = independentProject(input, weights: weights.aProjection,
                                  rows: configuration.valueHeadCount, columns: configuration.hiddenSize)
    let conv = independentConvolution(input: qkv, channels: channels, weights: weights.convolution,
                                      history: initialState.convolutionHistory)
    var compactQ: [Float] = []; var compactK: [Float] = []; var value: [Float] = []
    for token in 0..<tokenCount {
        let base = token * channels
        compactQ += conv.output[base..<(base + keyWidth)]
        compactK += conv.output[(base + keyWidth)..<(base + keyWidth * 2)]
        value += conv.output[(base + keyWidth * 2)..<(base + channels)]
    }
    let query = expand(compactQ, tokenCount: tokenCount, configuration: configuration)
    let key = expand(compactK, tokenCount: tokenCount, configuration: configuration)
    let beta = rawBeta.map(sigmoid)
    var logDecay = [Float](repeating: 0, count: rawA.count)
    for token in 0..<tokenCount {
        for head in 0..<configuration.valueHeadCount {
            let index = token * configuration.valueHeadCount + head
            logDecay[index] = -exp(weights.aLog[head])
                * softplus(rawA[index] + weights.timeStepBias[head])
        }
    }
    let recurrence = independentRecurrence(
        configuration: configuration, query: query, key: key, value: value,
        logDecay: logDecay, beta: beta, initialState: initialState.recurrentMatrix)
    let normalized = independentGatedNorm(
        configuration: configuration, input: recurrence.output, gate: gate,
        weights: weights.normalization)
    let output = independentProject(
        normalized, weights: weights.outputProjection,
        rows: configuration.hiddenSize, columns: valueWidth)
    return IndependentResult(
        qkvProjection: qkv, convolvedQKV: conv.output, query: query, key: key,
        value: value, gate: gate, beta: beta, logDecay: logDecay,
        recurrentOutput: recurrence.output, normalizedGatedOutput: normalized,
        output: output,
        finalState: QwenLinearAttentionLayerState(
            convolutionHistory: conv.history, recurrentMatrix: recurrence.state))
}

private struct IndependentConv { let output, history: [Float] }
private func independentConvolution(
    input: [Float], channels: Int, weights: [Float], history initial: [Float]
) -> IndependentConv {
    var history = initial
    let tokens = input.count / channels
    var output = [Float](repeating: 0, count: input.count)
    for token in 0..<tokens {
        for channel in 0..<channels {
            let base = channel * 4
            history[base] = history[base + 1]
            history[base + 1] = history[base + 2]
            history[base + 2] = history[base + 3]
            history[base + 3] = input[token * channels + channel]
            var sum: Float = 0
            for index in 0..<4 { sum += history[base + index] * weights[base + index] }
            output[token * channels + channel] = silu(sum)
        }
    }
    return IndependentConv(output: output, history: history)
}

private struct IndependentRecurrence { let output, state: [Float] }
private func independentRecurrence(
    configuration: QwenGatedDeltaNetConfiguration,
    query: [Float], key: [Float], value: [Float], logDecay: [Float], beta: [Float],
    initialState: [Float]
) -> IndependentRecurrence {
    let tokens = query.count / (configuration.valueHeadCount * configuration.keyHeadDimension)
    let heads = configuration.valueHeadCount
    let kd = configuration.keyHeadDimension
    let vd = configuration.valueHeadDimension
    var state = initialState
    var output = [Float](repeating: 0, count: tokens * heads * vd)
    for token in 0..<tokens {
        for head in 0..<heads {
            let qBase = (token * heads + head) * kd
            let vBase = (token * heads + head) * vd
            let stateBase = head * kd * vd
            var qSquare: Float = 0; var kSquare: Float = 0
            for k in 0..<kd {
                qSquare += query[qBase + k] * query[qBase + k]
                kSquare += key[qBase + k] * key[qBase + k]
            }
            let qInverse = 1 / sqrt(qSquare + configuration.epsilon) / sqrt(Float(kd))
            let kInverse = 1 / sqrt(kSquare + configuration.epsilon)
            let decay = exp(logDecay[token * heads + head])
            for k in 0..<kd { for v in 0..<vd { state[stateBase + k * vd + v] *= decay } }
            for v in 0..<vd {
                var prediction: Float = 0
                for k in 0..<kd { prediction += state[stateBase + k * vd + v] * key[qBase + k] * kInverse }
                let delta = (value[vBase + v] - prediction) * beta[token * heads + head]
                for k in 0..<kd { state[stateBase + k * vd + v] += key[qBase + k] * kInverse * delta }
            }
            for v in 0..<vd {
                var sum: Float = 0
                for k in 0..<kd { sum += state[stateBase + k * vd + v] * query[qBase + k] * qInverse }
                output[vBase + v] = sum
            }
        }
    }
    return IndependentRecurrence(output: output, state: state)
}

private func independentGatedNorm(
    configuration: QwenGatedDeltaNetConfiguration,
    input: [Float], gate: [Float], weights: [Float]
) -> [Float] {
    let vd = configuration.valueHeadDimension
    var output = [Float](repeating: 0, count: input.count)
    for base in stride(from: 0, to: input.count, by: vd) {
        let squareSum = (0..<vd).reduce(Float.zero) { $0 + input[base + $1] * input[base + $1] }
        let inverse = 1 / sqrt(squareSum / Float(vd) + configuration.epsilon)
        for d in 0..<vd { output[base + d] = input[base + d] * inverse * weights[d] * silu(gate[base + d]) }
    }
    return output
}

private func independentProject(_ input: [Float], weights: [Float], rows: Int, columns: Int) -> [Float] {
    let tokens = input.count / columns
    var output = [Float](repeating: 0, count: tokens * rows)
    for token in 0..<tokens {
        for row in 0..<rows {
            var sum: Float = 0
            for column in 0..<columns { sum += input[token * columns + column] * weights[row * columns + column] }
            output[token * rows + row] = sum
        }
    }
    return output
}

private func expand(
    _ compact: [Float], tokenCount: Int, configuration: QwenGatedDeltaNetConfiguration
) -> [Float] {
    let heads = configuration.valueHeadCount
    let kd = configuration.keyHeadDimension
    var result = [Float](repeating: 0, count: tokenCount * heads * kd)
    for token in 0..<tokenCount {
        for head in 0..<heads {
            let sourceHead = head / configuration.headsPerKeyHead
            for dimension in 0..<kd {
                result[(token * heads + head) * kd + dimension] =
                    compact[(token * configuration.keyHeadCount + sourceHead) * kd + dimension]
            }
        }
    }
    return result
}

private func channelMajorToTokenMajor(_ values: [Float], channels: Int, tokens: Int) -> [Float] {
    var result = [Float](repeating: 0, count: values.count)
    for channel in 0..<channels {
        for token in 0..<tokens { result[token * channels + channel] = values[channel * tokens + token] }
    }
    return result
}

private func tokenMajorToChannelMajor(_ values: [Float], channels: Int, tokens: Int) -> [Float] {
    var result = [Float](repeating: 0, count: values.count)
    for token in 0..<tokens {
        for channel in 0..<channels { result[channel * tokens + token] = values[token * channels + channel] }
    }
    return result
}

private func fixtureSection(_ name: String) throws -> [String: Any] {
    guard let url = Bundle.module.url(forResource: "qwen36-tiny-fixtures", withExtension: "json") else { throw FixtureError.missing }
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
    guard let root = object as? [String: Any], let section = root[name] as? [String: Any] else { throw FixtureError.invalid(name) }
    return section
}

private func tensor(_ object: [String: Any], _ key: String, nested: String? = nil) throws -> [Float] {
    var value: Any? = object[key]
    if let nested { value = (value as? [String: Any])?[nested] }
    guard let tensor = value as? [String: Any], let values = tensor["values"] as? [NSNumber] else { throw FixtureError.invalid(key) }
    return values.map(\.floatValue)
}

private enum FixtureError: Error { case missing, invalid(String) }
private func sigmoid(_ value: Float) -> Float { 1 / (1 + exp(-value)) }
private func silu(_ value: Float) -> Float { value * sigmoid(value) }
private func softplus(_ value: Float) -> Float { value > 20 ? value : log1p(exp(value)) }
private func maxDifference(_ lhs: [Float], _ rhs: [Float]) -> Float { zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0 }
private func expectClose(_ actual: [Float], _ expected: [Float], tolerance: Float, sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(actual.count == expected.count, sourceLocation: sourceLocation)
    guard actual.count == expected.count else { return }
    for (index, pair) in zip(actual, expected).enumerated() {
        #expect(abs(pair.0 - pair.1) <= tolerance + tolerance * abs(pair.1), "index \(index): \(pair.0) != \(pair.1)", sourceLocation: sourceLocation)
    }
}
private func floatBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Float>.stride, options: .storageModeShared) else { throw TestResourceError.bufferAllocation }
    return buffer
}
private func emptyFloatBuffer(count: Int, device: MTLDevice) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared) else { throw TestResourceError.bufferAllocation }
    return buffer
}
private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] { Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count)) }
