import Foundation
import Metal

enum QwenGatedDeltaNetError: Error, Equatable, Sendable {
    case invalidConfiguration(field: String, value: Int)
    case invalidFloatingPointConfiguration(field: String)
    case invalidCount(field: String, expected: Int, actual: Int)
    case arithmeticOverflow(operation: String)
    case bufferTooSmall(name: String, required: Int, actual: Int)
    case commandBufferAlreadySubmitted
    case commandEncoderUnavailable
    case invalidPipelineLimit
    case aliasedBuffers
}

struct QwenGatedDeltaNetConfiguration: Equatable, Sendable {
    let hiddenSize: Int
    let keyHeadCount: Int
    let valueHeadCount: Int
    let keyHeadDimension: Int
    let valueHeadDimension: Int
    let convolutionWidth: Int
    let epsilon: Float

    var keyDimension: Int { keyHeadCount * keyHeadDimension }
    var valueDimension: Int { valueHeadCount * valueHeadDimension }
    var convolutionChannelCount: Int { keyDimension * 2 + valueDimension }
    var headsPerKeyHead: Int { valueHeadCount / keyHeadCount }

    init(
        hiddenSize: Int,
        keyHeadCount: Int,
        valueHeadCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        convolutionWidth: Int = 4,
        epsilon: Float = 1e-6
    ) throws {
        for (field, value) in [
            ("hiddenSize", hiddenSize), ("keyHeadCount", keyHeadCount),
            ("valueHeadCount", valueHeadCount), ("keyHeadDimension", keyHeadDimension),
            ("valueHeadDimension", valueHeadDimension),
        ] where value <= 0 || UInt32(exactly: value) == nil {
            throw QwenGatedDeltaNetError.invalidConfiguration(field: field, value: value)
        }
        guard convolutionWidth == 4 else {
            throw QwenGatedDeltaNetError.invalidConfiguration(
                field: "convolutionWidth", value: convolutionWidth)
        }
        guard valueHeadCount.isMultiple(of: keyHeadCount) else {
            throw QwenGatedDeltaNetError.invalidConfiguration(
                field: "valueHeadCount/keyHeadCount", value: valueHeadCount)
        }
        guard epsilon.isFinite, epsilon > 0 else {
            throw QwenGatedDeltaNetError.invalidFloatingPointConfiguration(field: "epsilon")
        }
        _ = try Self.checkedMultiply(keyHeadCount, keyHeadDimension, operation: "key dimension")
        _ = try Self.checkedMultiply(valueHeadCount, valueHeadDimension, operation: "value dimension")
        self.hiddenSize = hiddenSize
        self.keyHeadCount = keyHeadCount
        self.valueHeadCount = valueHeadCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
        self.convolutionWidth = convolutionWidth
        self.epsilon = epsilon
    }

    static func official() throws -> Self {
        try Self(
            hiddenSize: 2_048,
            keyHeadCount: 16,
            valueHeadCount: 32,
            keyHeadDimension: 128,
            valueHeadDimension: 128)
    }

    fileprivate static func checkedMultiply(
        _ lhs: Int, _ rhs: Int, operation: String
    ) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw QwenGatedDeltaNetError.arithmeticOverflow(operation: operation) }
        return result
    }
}

struct QwenGatedDeltaNetWeights: Sendable {
    /// Row-major `[2 * keyDimension + valueDimension, hiddenSize]`.
    let qkvProjection: [Float]
    /// Row-major `[valueDimension, hiddenSize]`.
    let zProjection: [Float]
    /// Row-major `[valueHeadCount, hiddenSize]`.
    let bProjection: [Float]
    let aProjection: [Float]
    /// Row-major `[convolutionChannelCount, 4]`, oldest to current.
    let convolution: [Float]
    let timeStepBias: [Float]
    let aLog: [Float]
    /// Direct RMSNorm weight, not a residual `(1 + weight)` parameter.
    let normalization: [Float]
    /// Row-major `[hiddenSize, valueDimension]`.
    let outputProjection: [Float]
}

struct QwenCausalConvolutionResult: Equatable, Sendable {
    /// Token-major `[token, channel]`, after SiLU.
    let output: [Float]
    /// Channel-major `[channel, width]`, raw projected input history.
    let finalHistory: [Float]
}

struct QwenGatedDeltaRecurrenceResult: Equatable, Sendable {
    /// Token-major `[token, valueHead, valueDimension]`.
    let output: [Float]
    /// `[valueHead, keyDimension, valueDimension]` in FP32.
    let finalState: [Float]
}

struct QwenGatedDeltaNetResult: Sendable {
    let qkvProjection: [Float]
    let convolvedQKV: [Float]
    let query: [Float]
    let key: [Float]
    let value: [Float]
    let gate: [Float]
    let beta: [Float]
    let logDecay: [Float]
    let recurrentOutput: [Float]
    let normalizedGatedOutput: [Float]
    let output: [Float]
    let finalState: QwenLinearAttentionLayerState
}

/// Qwen Gated DeltaNet reference order plus concrete correctness Metal kernels.
final class QwenGatedDeltaNet {
    private struct LayoutParameters {
        var tokenCount: UInt32
        var channelCount: UInt32
        var tokenToChannel: UInt32
        var reserved: UInt32 = 0
    }

    private struct ConvolutionParameters {
        var tokenCount: UInt32
        var channelCount: UInt32
        var width: UInt32
        var reserved: UInt32 = 0
    }

    private struct RecurrenceParameters {
        var tokenCount: UInt32
        var headCount: UInt32
        var keyDimension: UInt32
        var valueDimension: UInt32
        var epsilon: Float
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }

    private struct GatedNormParameters {
        var tokenCount: UInt32
        var headCount: UInt32
        var valueDimension: UInt32
        var reserved: UInt32 = 0
        var epsilon: Float
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }

    let configuration: QwenGatedDeltaNetConfiguration
    private let layoutPipeline: MTLComputePipelineState
    private let convolutionPipeline: MTLComputePipelineState
    private let recurrencePipeline: MTLComputePipelineState
    private let gatedNormPipeline: MTLComputePipelineState

    init(context: MetalContext, configuration: QwenGatedDeltaNetConfiguration) throws {
        self.configuration = configuration
        layoutPipeline = try context.pipeline("qwen_linear_layout")
        convolutionPipeline = try context.pipeline("qwen_linear_causal_conv")
        recurrencePipeline = try context.pipeline("qwen_linear_recurrence_fp32")
        gatedNormPipeline = try context.pipeline("qwen_linear_gated_rmsnorm")
    }

    static func evaluate(
        configuration: QwenGatedDeltaNetConfiguration,
        input: [Float],
        weights: QwenGatedDeltaNetWeights,
        initialState: QwenLinearAttentionLayerState,
        chunkSize: Int
    ) throws -> QwenGatedDeltaNetResult {
        guard input.count.isMultiple(of: configuration.hiddenSize) else {
            throw QwenGatedDeltaNetError.invalidCount(
                field: "input", expected: configuration.hiddenSize, actual: input.count)
        }
        guard chunkSize > 0 else {
            throw QwenGatedDeltaNetError.invalidConfiguration(field: "chunkSize", value: chunkSize)
        }
        try validate(weights: weights, configuration: configuration)
        try validate(state: initialState, configuration: configuration)
        let tokenCount = input.count / configuration.hiddenSize
        if tokenCount == 0 {
            return QwenGatedDeltaNetResult(
                qkvProjection: [], convolvedQKV: [], query: [], key: [], value: [], gate: [],
                beta: [], logDecay: [], recurrentOutput: [], normalizedGatedOutput: [], output: [],
                finalState: initialState)
        }

        let qkv = project(
            input, weights: weights.qkvProjection,
            rows: configuration.convolutionChannelCount,
            columns: configuration.hiddenSize)
        let gate = project(
            input, weights: weights.zProjection,
            rows: configuration.valueDimension,
            columns: configuration.hiddenSize)
        let rawBeta = project(
            input, weights: weights.bProjection,
            rows: configuration.valueHeadCount,
            columns: configuration.hiddenSize)
        let rawA = project(
            input, weights: weights.aProjection,
            rows: configuration.valueHeadCount,
            columns: configuration.hiddenSize)
        let convolution = try causalConvolution(
            configuration: configuration,
            input: qkv,
            weights: weights.convolution,
            initialHistory: initialState.convolutionHistory)

        let keyWidth = configuration.keyDimension
        let valueWidth = configuration.valueDimension
        var compactQuery = [Float](); var compactKey = [Float](); var value = [Float]()
        compactQuery.reserveCapacity(tokenCount * keyWidth)
        compactKey.reserveCapacity(tokenCount * keyWidth)
        value.reserveCapacity(tokenCount * valueWidth)
        for token in 0..<tokenCount {
            let base = token * configuration.convolutionChannelCount
            compactQuery.append(contentsOf: convolution.output[base..<(base + keyWidth)])
            compactKey.append(contentsOf: convolution.output[(base + keyWidth)..<(base + 2 * keyWidth)])
            value.append(contentsOf: convolution.output[(base + 2 * keyWidth)..<(base + 2 * keyWidth + valueWidth)])
        }
        let query = expandKeyHeads(compactQuery, tokenCount: tokenCount, configuration: configuration)
        let key = expandKeyHeads(compactKey, tokenCount: tokenCount, configuration: configuration)
        let beta = rawBeta.map(sigmoid)
        var logDecay = [Float](repeating: 0, count: rawA.count)
        for token in 0..<tokenCount {
            for head in 0..<configuration.valueHeadCount {
                let index = token * configuration.valueHeadCount + head
                logDecay[index] = -exp(weights.aLog[head])
                    * softplus(rawA[index] + weights.timeStepBias[head])
            }
        }
        let recurrence = try recurrent(
            configuration: configuration,
            query: query,
            key: key,
            value: value,
            logDecay: logDecay,
            beta: beta,
            initialState: initialState.recurrentMatrix,
            chunkSize: chunkSize)
        let normalizedGated = try gatedRMSNorm(
            configuration: configuration,
            input: recurrence.output,
            gate: gate,
            weights: weights.normalization)
        let output = project(
            normalizedGated,
            weights: weights.outputProjection,
            rows: configuration.hiddenSize,
            columns: configuration.valueDimension)
        return QwenGatedDeltaNetResult(
            qkvProjection: qkv,
            convolvedQKV: convolution.output,
            query: query,
            key: key,
            value: value,
            gate: gate,
            beta: beta,
            logDecay: logDecay,
            recurrentOutput: recurrence.output,
            normalizedGatedOutput: normalizedGated,
            output: output,
            finalState: QwenLinearAttentionLayerState(
                convolutionHistory: convolution.finalHistory,
                recurrentMatrix: recurrence.finalState))
    }

    static func causalConvolution(
        configuration: QwenGatedDeltaNetConfiguration,
        input: [Float],
        weights: [Float],
        initialHistory: [Float]
    ) throws -> QwenCausalConvolutionResult {
        let channels = configuration.convolutionChannelCount
        guard input.count.isMultiple(of: channels) else {
            throw QwenGatedDeltaNetError.invalidCount(
                field: "convolution input", expected: channels, actual: input.count)
        }
        try requireCount(
            weights, field: "convolution weights",
            expected: channels * configuration.convolutionWidth)
        try requireCount(
            initialHistory, field: "convolution history",
            expected: channels * configuration.convolutionWidth)
        let tokenCount = input.count / channels
        var history = initialHistory
        var output = [Float](repeating: 0, count: input.count)
        for token in 0..<tokenCount {
            for channel in 0..<channels {
                let base = channel * configuration.convolutionWidth
                for index in 0..<(configuration.convolutionWidth - 1) {
                    history[base + index] = history[base + index + 1]
                }
                history[base + configuration.convolutionWidth - 1] = input[token * channels + channel]
                var sum: Float = 0
                for index in 0..<configuration.convolutionWidth {
                    sum += history[base + index] * weights[base + index]
                }
                output[token * channels + channel] = silu(sum)
            }
        }
        return QwenCausalConvolutionResult(output: output, finalHistory: history)
    }

    static func recurrent(
        configuration: QwenGatedDeltaNetConfiguration,
        query: [Float],
        key: [Float],
        value: [Float],
        logDecay: [Float],
        beta: [Float],
        initialState: [Float],
        chunkSize: Int
    ) throws -> QwenGatedDeltaRecurrenceResult {
        guard chunkSize > 0 else {
            throw QwenGatedDeltaNetError.invalidConfiguration(field: "chunkSize", value: chunkSize)
        }
        let queryTokenWidth = configuration.valueHeadCount * configuration.keyHeadDimension
        guard query.count.isMultiple(of: queryTokenWidth) else {
            throw QwenGatedDeltaNetError.invalidCount(
                field: "query", expected: queryTokenWidth, actual: query.count)
        }
        let tokenCount = query.count / queryTokenWidth
        try requireCount(key, field: "key", expected: query.count)
        try requireCount(
            value, field: "value", expected: tokenCount * configuration.valueDimension)
        try requireCount(
            logDecay, field: "logDecay", expected: tokenCount * configuration.valueHeadCount)
        try requireCount(beta, field: "beta", expected: logDecay.count)
        let stateCount = configuration.valueHeadCount
            * configuration.keyHeadDimension * configuration.valueHeadDimension
        try requireCount(initialState, field: "initialState", expected: stateCount)

        var state = initialState
        var output = [Float](repeating: 0, count: tokenCount * configuration.valueDimension)
        var chunkStart = 0
        while chunkStart < tokenCount {
            let chunkEnd = min(tokenCount, chunkStart + chunkSize)
            for token in chunkStart..<chunkEnd {
                for head in 0..<configuration.valueHeadCount {
                    let qBase = (token * configuration.valueHeadCount + head)
                        * configuration.keyHeadDimension
                    let valueBase = (token * configuration.valueHeadCount + head)
                        * configuration.valueHeadDimension
                    let stateBase = head * configuration.keyHeadDimension
                        * configuration.valueHeadDimension
                    var qNormSquared: Float = 0
                    var kNormSquared: Float = 0
                    for k in 0..<configuration.keyHeadDimension {
                        qNormSquared += query[qBase + k] * query[qBase + k]
                        kNormSquared += key[qBase + k] * key[qBase + k]
                    }
                    let inverseQNorm = 1 / sqrt(qNormSquared + configuration.epsilon)
                        / sqrt(Float(configuration.keyHeadDimension))
                    let inverseKNorm = 1 / sqrt(kNormSquared + configuration.epsilon)
                    let decay = exp(logDecay[token * configuration.valueHeadCount + head])
                    let step = beta[token * configuration.valueHeadCount + head]
                    for k in 0..<configuration.keyHeadDimension {
                        for v in 0..<configuration.valueHeadDimension {
                            state[stateBase + k * configuration.valueHeadDimension + v] *= decay
                        }
                    }
                    for v in 0..<configuration.valueHeadDimension {
                        var prediction: Float = 0
                        for k in 0..<configuration.keyHeadDimension {
                            prediction += state[stateBase + k * configuration.valueHeadDimension + v]
                                * key[qBase + k] * inverseKNorm
                        }
                        let delta = (value[valueBase + v] - prediction) * step
                        for k in 0..<configuration.keyHeadDimension {
                            state[stateBase + k * configuration.valueHeadDimension + v] +=
                                key[qBase + k] * inverseKNorm * delta
                        }
                    }
                    for v in 0..<configuration.valueHeadDimension {
                        var sum: Float = 0
                        for k in 0..<configuration.keyHeadDimension {
                            sum += state[stateBase + k * configuration.valueHeadDimension + v]
                                * query[qBase + k] * inverseQNorm
                        }
                        output[valueBase + v] = sum
                    }
                }
            }
            chunkStart = chunkEnd
        }
        return QwenGatedDeltaRecurrenceResult(output: output, finalState: state)
    }

    static func gatedRMSNorm(
        configuration: QwenGatedDeltaNetConfiguration,
        input: [Float],
        gate: [Float],
        weights: [Float]
    ) throws -> [Float] {
        try requireCount(gate, field: "gate", expected: input.count)
        try requireCount(
            weights, field: "normalization weights", expected: configuration.valueHeadDimension)
        guard input.count.isMultiple(of: configuration.valueHeadDimension) else {
            throw QwenGatedDeltaNetError.invalidCount(
                field: "gated norm input", expected: configuration.valueHeadDimension,
                actual: input.count)
        }
        var output = [Float](repeating: 0, count: input.count)
        for vector in 0..<(input.count / configuration.valueHeadDimension) {
            let base = vector * configuration.valueHeadDimension
            var squareSum: Float = 0
            for index in 0..<configuration.valueHeadDimension {
                squareSum += input[base + index] * input[base + index]
            }
            let inverseRMS = 1 / sqrt(
                squareSum / Float(configuration.valueHeadDimension) + configuration.epsilon)
            for index in 0..<configuration.valueHeadDimension {
                output[base + index] = input[base + index] * inverseRMS * weights[index]
                    * silu(gate[base + index])
            }
        }
        return output
    }

    func encodeLayout(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        tokenCount: Int,
        channelCount: Int,
        tokenToChannel: Bool
    ) throws {
        guard input !== output else { throw QwenGatedDeltaNetError.aliasedBuffers }
        let count = try checkedMultiply(tokenCount, channelCount, operation: "layout elements")
        guard count > 0 else { return }
        let bytes = try checkedMultiply(count, MemoryLayout<Float>.stride, operation: "layout bytes")
        try requireBuffer(input, name: "layout input", bytes: bytes)
        try requireBuffer(output, name: "layout output", bytes: bytes)
        var parameters = LayoutParameters(
            tokenCount: try uint32(tokenCount, field: "tokenCount"),
            channelCount: try uint32(channelCount, field: "channelCount"),
            tokenToChannel: tokenToChannel ? 1 : 0)
        try encode(
            commandBuffer: commandBuffer, pipeline: layoutPipeline, count: count,
            parameters: &parameters,
            buffers: [(input, QwenMetalBufferIndex.input.rawValue),
                      (output, QwenMetalBufferIndex.output.rawValue)])
    }

    func encodeCausalConvolution(
        commandBuffer: MTLCommandBuffer,
        channelMajorInput: MTLBuffer,
        weights: MTLBuffer,
        update: QwenLinearAttentionUpdate,
        channelMajorOutput: MTLBuffer,
        tokenCount: Int
    ) throws {
        let channels = configuration.convolutionChannelCount
        let elements = try checkedMultiply(tokenCount, channels, operation: "convolution elements")
        guard elements > 0 else { return }
        try requireFloatBuffer(channelMajorInput, name: "convolution input", count: elements)
        try requireFloatBuffer(channelMajorOutput, name: "convolution output", count: elements)
        try requireFloatBuffer(
            weights, name: "convolution weights", count: channels * configuration.convolutionWidth)
        try requireFloatBuffer(
            update.convolutionHistory, name: "convolution history",
            count: channels * configuration.convolutionWidth)
        var parameters = ConvolutionParameters(
            tokenCount: try uint32(tokenCount, field: "tokenCount"),
            channelCount: try uint32(channels, field: "channelCount"),
            width: try uint32(configuration.convolutionWidth, field: "convolutionWidth"))
        try encode(
            commandBuffer: commandBuffer, pipeline: convolutionPipeline, count: channels,
            parameters: &parameters,
            buffers: [(channelMajorInput, QwenMetalBufferIndex.input.rawValue),
                      (weights, QwenMetalBufferIndex.weights.rawValue),
                      (channelMajorOutput, QwenMetalBufferIndex.output.rawValue),
                      (update.convolutionHistory, QwenMetalBufferIndex.state.rawValue)])
    }

    func encodeRecurrence(
        commandBuffer: MTLCommandBuffer,
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        logDecay: MTLBuffer,
        beta: MTLBuffer,
        update: QwenLinearAttentionUpdate,
        output: MTLBuffer,
        tokenCount: Int
    ) throws {
        let heads = configuration.valueHeadCount
        let qCount = tokenCount * heads * configuration.keyHeadDimension
        let vCount = tokenCount * heads * configuration.valueHeadDimension
        let scalarCount = tokenCount * heads
        guard tokenCount > 0 else { return }
        try requireFloatBuffer(query, name: "query", count: qCount)
        try requireFloatBuffer(key, name: "key", count: qCount)
        try requireFloatBuffer(value, name: "value", count: vCount)
        try requireFloatBuffer(logDecay, name: "logDecay", count: scalarCount)
        try requireFloatBuffer(beta, name: "beta", count: scalarCount)
        try requireFloatBuffer(output, name: "recurrence output", count: vCount)
        try requireFloatBuffer(
            update.recurrentMatrix, name: "recurrent state",
            count: heads * configuration.keyHeadDimension * configuration.valueHeadDimension)
        var parameters = RecurrenceParameters(
            tokenCount: try uint32(tokenCount, field: "tokenCount"),
            headCount: try uint32(heads, field: "headCount"),
            keyDimension: try uint32(configuration.keyHeadDimension, field: "keyHeadDimension"),
            valueDimension: try uint32(configuration.valueHeadDimension, field: "valueHeadDimension"),
            epsilon: configuration.epsilon)
        try encode(
            commandBuffer: commandBuffer, pipeline: recurrencePipeline, count: heads,
            parameters: &parameters,
            buffers: [(query, QwenMetalBufferIndex.input.rawValue),
                      (key, QwenMetalBufferIndex.weights.rawValue),
                      (value, QwenMetalBufferIndex.scales.rawValue),
                      (logDecay, QwenMetalBufferIndex.biases.rawValue),
                      (output, QwenMetalBufferIndex.output.rawValue),
                      (beta, QwenMetalBufferIndex.scratch.rawValue),
                      (update.recurrentMatrix, QwenMetalBufferIndex.state.rawValue)])
    }

    func encodeGatedRMSNorm(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        gate: MTLBuffer,
        weights: MTLBuffer,
        output: MTLBuffer,
        tokenCount: Int
    ) throws {
        let itemCount = tokenCount * configuration.valueHeadCount
        let elements = itemCount * configuration.valueHeadDimension
        guard itemCount > 0 else { return }
        try requireFloatBuffer(input, name: "gated norm input", count: elements)
        try requireFloatBuffer(gate, name: "gate", count: elements)
        try requireFloatBuffer(weights, name: "norm weights", count: configuration.valueHeadDimension)
        try requireFloatBuffer(output, name: "gated norm output", count: elements)
        var parameters = GatedNormParameters(
            tokenCount: try uint32(tokenCount, field: "tokenCount"),
            headCount: try uint32(configuration.valueHeadCount, field: "headCount"),
            valueDimension: try uint32(configuration.valueHeadDimension, field: "valueHeadDimension"),
            epsilon: configuration.epsilon)
        try encode(
            commandBuffer: commandBuffer, pipeline: gatedNormPipeline, count: itemCount,
            parameters: &parameters,
            buffers: [(input, QwenMetalBufferIndex.input.rawValue),
                      (weights, QwenMetalBufferIndex.weights.rawValue),
                      (output, QwenMetalBufferIndex.output.rawValue),
                      (gate, QwenMetalBufferIndex.scratch.rawValue)])
    }

    private func encode<T>(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        count: Int,
        parameters: inout T,
        buffers: [(MTLBuffer, Int)]
    ) throws {
        guard commandBuffer.status == .notEnqueued else {
            throw QwenGatedDeltaNetError.commandBufferAlreadySubmitted
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenGatedDeltaNetError.commandEncoderUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        withUnsafeBytes(of: &parameters) { bytes in
            encoder.setBytes(
                bytes.baseAddress!, length: bytes.count,
                index: QwenMetalBufferIndex.parameters.rawValue)
        }
        for (buffer, index) in buffers { encoder.setBuffer(buffer, offset: 0, index: index) }
        let maximum = pipeline.maxTotalThreadsPerThreadgroup
        guard maximum > 0 else { encoder.endEncoding(); throw QwenGatedDeltaNetError.invalidPipelineLimit }
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(count, maximum), height: 1, depth: 1))
        encoder.endEncoding()
    }

    private static func validate(
        weights: QwenGatedDeltaNetWeights,
        configuration: QwenGatedDeltaNetConfiguration
    ) throws {
        try requireCount(weights.qkvProjection, field: "qkvProjection",
                         expected: configuration.convolutionChannelCount * configuration.hiddenSize)
        try requireCount(weights.zProjection, field: "zProjection",
                         expected: configuration.valueDimension * configuration.hiddenSize)
        let scalarProjection = configuration.valueHeadCount * configuration.hiddenSize
        try requireCount(weights.bProjection, field: "bProjection", expected: scalarProjection)
        try requireCount(weights.aProjection, field: "aProjection", expected: scalarProjection)
        try requireCount(weights.convolution, field: "convolution",
                         expected: configuration.convolutionChannelCount * configuration.convolutionWidth)
        try requireCount(weights.timeStepBias, field: "timeStepBias", expected: configuration.valueHeadCount)
        try requireCount(weights.aLog, field: "aLog", expected: configuration.valueHeadCount)
        try requireCount(weights.normalization, field: "normalization", expected: configuration.valueHeadDimension)
        try requireCount(weights.outputProjection, field: "outputProjection",
                         expected: configuration.hiddenSize * configuration.valueDimension)
    }

    private static func validate(
        state: QwenLinearAttentionLayerState,
        configuration: QwenGatedDeltaNetConfiguration
    ) throws {
        try requireCount(state.convolutionHistory, field: "convolutionHistory",
                         expected: configuration.convolutionChannelCount * configuration.convolutionWidth)
        try requireCount(state.recurrentMatrix, field: "recurrentMatrix",
                         expected: configuration.valueHeadCount * configuration.keyHeadDimension
                            * configuration.valueHeadDimension)
    }

    private static func project(
        _ input: [Float], weights: [Float], rows: Int, columns: Int
    ) -> [Float] {
        let tokenCount = input.count / columns
        var output = [Float](repeating: 0, count: tokenCount * rows)
        for token in 0..<tokenCount {
            for row in 0..<rows {
                var sum: Float = 0
                for column in 0..<columns {
                    sum += input[token * columns + column] * weights[row * columns + column]
                }
                output[token * rows + row] = sum
            }
        }
        return output
    }

    private static func expandKeyHeads(
        _ compact: [Float], tokenCount: Int, configuration: QwenGatedDeltaNetConfiguration
    ) -> [Float] {
        var expanded = [Float](
            repeating: 0,
            count: tokenCount * configuration.valueHeadCount * configuration.keyHeadDimension)
        for token in 0..<tokenCount {
            for valueHead in 0..<configuration.valueHeadCount {
                let keyHead = valueHead / configuration.headsPerKeyHead
                for dimension in 0..<configuration.keyHeadDimension {
                    expanded[(token * configuration.valueHeadCount + valueHead)
                        * configuration.keyHeadDimension + dimension] =
                        compact[(token * configuration.keyHeadCount + keyHead)
                            * configuration.keyHeadDimension + dimension]
                }
            }
        }
        return expanded
    }

    private static func requireCount(
        _ values: [Float], field: String, expected: Int
    ) throws {
        guard values.count == expected else {
            throw QwenGatedDeltaNetError.invalidCount(
                field: field, expected: expected, actual: values.count)
        }
    }

    private func requireFloatBuffer(_ buffer: MTLBuffer, name: String, count: Int) throws {
        try requireBuffer(
            buffer, name: name,
            bytes: try checkedMultiply(count, MemoryLayout<Float>.stride, operation: "\(name) bytes"))
    }

    private func requireBuffer(_ buffer: MTLBuffer, name: String, bytes: Int) throws {
        guard buffer.length >= bytes else {
            throw QwenGatedDeltaNetError.bufferTooSmall(
                name: name, required: bytes, actual: buffer.length)
        }
    }

    private func checkedMultiply(_ lhs: Int, _ rhs: Int, operation: String) throws -> Int {
        try QwenGatedDeltaNetConfiguration.checkedMultiply(lhs, rhs, operation: operation)
    }

    private func uint32(_ value: Int, field: String) throws -> UInt32 {
        guard value >= 0, let result = UInt32(exactly: value) else {
            throw QwenGatedDeltaNetError.invalidConfiguration(field: field, value: value)
        }
        return result
    }
}

private func sigmoid(_ value: Float) -> Float { 1 / (1 + exp(-value)) }
private func silu(_ value: Float) -> Float { value * sigmoid(value) }
private func softplus(_ value: Float) -> Float {
    value > 20 ? value : log1p(exp(value))
}
