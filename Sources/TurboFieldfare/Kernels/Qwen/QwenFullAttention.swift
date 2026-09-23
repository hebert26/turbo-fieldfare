import Foundation
import Metal

enum QwenFullAttentionError: Error, Equatable, Sendable {
    case invalidConfiguration(field: String, value: Int)
    case invalidFloatingPointConfiguration(field: String)
    case invalidCount(field: String, expected: Int, actual: Int)
    case invalidPosition(expected: Int, actual: Int)
    case positionOutOfRange(Int)
    case arithmeticOverflow(operation: String)
    case bufferTooSmall(name: String, required: Int, actual: Int)
    case aliasedNormRoPEBuffers
    case commandBufferAlreadySubmitted
    case commandEncoderUnavailable
    case invalidPipelineLimit
}

/// Geometry and numerical constants for Qwen full attention.
///
/// The tiny P3 oracle deliberately uses a different rotary fraction from the
/// official model. Callers must therefore provide the rotary span explicitly;
/// it is never inferred from the head dimension.
struct QwenFullAttentionConfiguration: Equatable, Sendable {
    let queryHeadCount: Int
    let keyValueHeadCount: Int
    let headDimension: Int
    let rotaryDimension: Int
    let theta: Float
    let epsilon: Float

    init(
        queryHeadCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        rotaryDimension: Int,
        theta: Float = 10_000_000,
        epsilon: Float = 1e-6
    ) throws {
        guard queryHeadCount > 0 else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "queryHeadCount", value: queryHeadCount)
        }
        guard keyValueHeadCount > 0 else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "keyValueHeadCount", value: keyValueHeadCount)
        }
        guard queryHeadCount.isMultiple(of: keyValueHeadCount) else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "queryHeadCount/keyValueHeadCount", value: queryHeadCount)
        }
        guard headDimension > 0 else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "headDimension", value: headDimension)
        }
        guard rotaryDimension > 0,
              rotaryDimension <= headDimension,
              rotaryDimension.isMultiple(of: 2) else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "rotaryDimension", value: rotaryDimension)
        }
        guard theta.isFinite, theta > 0 else {
            throw QwenFullAttentionError.invalidFloatingPointConfiguration(field: "theta")
        }
        guard epsilon.isFinite, epsilon > 0 else {
            throw QwenFullAttentionError.invalidFloatingPointConfiguration(field: "epsilon")
        }

        self.queryHeadCount = queryHeadCount
        self.keyValueHeadCount = keyValueHeadCount
        self.headDimension = headDimension
        self.rotaryDimension = rotaryDimension
        self.theta = theta
        self.epsilon = epsilon
    }

    static func official() throws -> Self {
        try Self(
            queryHeadCount: 16,
            keyValueHeadCount: 2,
            headDimension: 256,
            rotaryDimension: 64,
            theta: 10_000_000,
            epsilon: 1e-6)
    }
}

struct QwenFullAttentionResult: Sendable {
    /// Token-major `[token, queryHead, headDimension]`, before RMSNorm.
    let queryProjection: [Float]
    /// Token-major normalized tensors, before RoPE.
    let normalizedQuery: [Float]
    let normalizedKey: [Float]
    /// Token-major tensors after partial RoPE.
    let rotatedQuery: [Float]
    let rotatedKey: [Float]
    /// `[queryToken, queryHead, totalKeyToken]`.
    let attentionWeights: [Float]
    /// Token-major flattened heads.
    let preOutputGate: [Float]
    let outputGate: [Float]
    let gatedAttention: [Float]
    let output: [Float]
}

/// Qwen full-attention reference ordering plus concrete Metal preprocessing
/// pipelines. Metal covers Q/K normalization with partial RoPE and the output
/// gate; QK scoring, causal FP32 softmax, value accumulation, and token-major
/// layout remain in the CPU reference operation. That operation is intentionally
/// straightforward and is used for small deterministic correctness inputs, not
/// production model execution or full-GPU qualification.
final class QwenFullAttention {
    private struct NormRoPEParameters {
        var tokenCount: UInt32
        var headCount: UInt32
        var headDimension: UInt32
        var rotaryDimension: UInt32
        var startPosition: UInt32
        var theta: Float
        var epsilon: Float
        var reserved: UInt32 = 0
    }

    private struct MRoPEParameters {
        var tokenCount: UInt32
        var headCount: UInt32
        var headDimension: UInt32
        var rotaryDimension: UInt32
        var temporalSection: UInt32
        var heightSection: UInt32
        var widthSection: UInt32
        var reserved: UInt32 = 0
        var theta: Float
        var epsilon: Float
    }

    private struct GateParameters {
        var elementCount: UInt32
        var reserved0: UInt32 = 0
        var reserved1: UInt32 = 0
        var reserved2: UInt32 = 0
    }

    let configuration: QwenFullAttentionConfiguration
    private let device: MTLDevice
    private let normRoPEPipeline: MTLComputePipelineState
    private let normMRoPEPipeline: MTLComputePipelineState
    private let gatePipeline: MTLComputePipelineState

    init(context: MetalContext, configuration: QwenFullAttentionConfiguration) throws {
        self.configuration = configuration
        device = context.device
        normRoPEPipeline = try context.pipeline("qwen_qk_norm_partial_rope")
        normMRoPEPipeline = try context.pipeline("qwen_qk_norm_partial_mrope")
        gatePipeline = try context.pipeline("qwen_full_attention_sigmoid_gate")
    }

    /// Encodes one Q or K tensor. Input/output are token-major FP32 tensors;
    /// weights are the stored Qwen residual RMSNorm weights.
    func encodeNormAndPartialRoPE(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        weight: MTLBuffer,
        output: MTLBuffer,
        tokenCount: Int,
        headCount: Int,
        startPosition: Int
    ) throws {
        guard input !== output else {
            throw QwenFullAttentionError.aliasedNormRoPEBuffers
        }
        guard tokenCount > 0 else {
            throw QwenFullAttentionError.invalidConfiguration(field: "tokenCount", value: tokenCount)
        }
        guard headCount == configuration.queryHeadCount
                || headCount == configuration.keyValueHeadCount else {
            throw QwenFullAttentionError.invalidConfiguration(field: "headCount", value: headCount)
        }
        let itemCount = try Self.checkedMultiply(tokenCount, headCount, operation: "norm/RoPE items")
        let elementCount = try Self.checkedMultiply(
            itemCount, configuration.headDimension, operation: "norm/RoPE elements")
        let requiredBytes = try Self.checkedMultiply(
            elementCount, MemoryLayout<Float>.stride, operation: "norm/RoPE bytes")
        try Self.requireBuffer(input, named: "input", bytes: requiredBytes)
        try Self.requireBuffer(output, named: "output", bytes: requiredBytes)
        let weightBytes = try Self.checkedMultiply(
            configuration.headDimension,
            MemoryLayout<Float>.stride,
            operation: "norm weight bytes")
        try Self.requireBuffer(weight, named: "weight", bytes: weightBytes)
        let checkedPosition = try Self.checkedPositionRange(
            start: startPosition, tokenCount: tokenCount)

        guard commandBuffer.status == .notEnqueued else {
            throw QwenFullAttentionError.commandBufferAlreadySubmitted
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenFullAttentionError.commandEncoderUnavailable
        }
        var parameters = NormRoPEParameters(
            tokenCount: try Self.uint32(tokenCount, field: "tokenCount"),
            headCount: try Self.uint32(headCount, field: "headCount"),
            headDimension: try Self.uint32(configuration.headDimension, field: "headDimension"),
            rotaryDimension: try Self.uint32(configuration.rotaryDimension, field: "rotaryDimension"),
            startPosition: checkedPosition,
            theta: configuration.theta,
            epsilon: configuration.epsilon)
        encoder.setComputePipelineState(normRoPEPipeline)
        encoder.setBytes(
            &parameters,
            length: MemoryLayout<NormRoPEParameters>.stride,
            index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(input, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
        encoder.setBuffer(weight, offset: 0, index: QwenMetalBufferIndex.weights.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        try dispatch(encoder: encoder, pipeline: normRoPEPipeline, count: itemCount)
        encoder.endEncoding()
    }

    /// Encodes explicit token-major `[t,h,w]` positions. Cache indices remain
    /// contiguous state and are deliberately not accepted by this API.
    func encodeNormAndPartialMRoPE(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        weight: MTLBuffer,
        positions: MTLBuffer,
        output: MTLBuffer,
        tokenCount: Int,
        headCount: Int,
        sections: [Int]
    ) throws {
        guard input !== output else { throw QwenFullAttentionError.aliasedNormRoPEBuffers }
        guard tokenCount > 0, sections.count == 3,
              sections.allSatisfy({ $0 > 0 }),
              sections.reduce(0, +) == configuration.rotaryDimension / 2 else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "mropeSections", value: sections.reduce(0, +))
        }
        guard headCount == configuration.queryHeadCount
                || headCount == configuration.keyValueHeadCount else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "headCount", value: headCount)
        }
        let itemCount = try Self.checkedMultiply(
            tokenCount, headCount, operation: "M-RoPE items")
        let elementCount = try Self.checkedMultiply(
            itemCount, configuration.headDimension, operation: "M-RoPE elements")
        let requiredBytes = try Self.checkedMultiply(
            elementCount, MemoryLayout<Float>.stride, operation: "M-RoPE bytes")
        try Self.requireBuffer(input, named: "input", bytes: requiredBytes)
        try Self.requireBuffer(output, named: "output", bytes: requiredBytes)
        try Self.requireBuffer(
            weight, named: "weight",
            bytes: configuration.headDimension * MemoryLayout<Float>.stride)
        try Self.requireBuffer(
            positions, named: "positions",
            bytes: tokenCount * 3 * MemoryLayout<Int32>.stride)
        guard commandBuffer.status == .notEnqueued else {
            throw QwenFullAttentionError.commandBufferAlreadySubmitted
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenFullAttentionError.commandEncoderUnavailable
        }
        var parameters = MRoPEParameters(
            tokenCount: try Self.uint32(tokenCount, field: "tokenCount"),
            headCount: try Self.uint32(headCount, field: "headCount"),
            headDimension: try Self.uint32(configuration.headDimension, field: "headDimension"),
            rotaryDimension: try Self.uint32(configuration.rotaryDimension, field: "rotaryDimension"),
            temporalSection: try Self.uint32(sections[0], field: "temporalSection"),
            heightSection: try Self.uint32(sections[1], field: "heightSection"),
            widthSection: try Self.uint32(sections[2], field: "widthSection"),
            theta: configuration.theta,
            epsilon: configuration.epsilon)
        encoder.setComputePipelineState(normMRoPEPipeline)
        encoder.setBytes(
            &parameters, length: MemoryLayout<MRoPEParameters>.stride,
            index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(input, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
        encoder.setBuffer(weight, offset: 0, index: QwenMetalBufferIndex.weights.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        encoder.setBuffer(positions, offset: 0, index: QwenMetalBufferIndex.scratch.rawValue)
        try dispatch(encoder: encoder, pipeline: normMRoPEPipeline, count: itemCount)
        encoder.endEncoding()
    }

    /// Applies `attention * sigmoid(rawGate)` in token-major order.
    func encodeOutputGate(
        commandBuffer: MTLCommandBuffer,
        attention: MTLBuffer,
        rawGate: MTLBuffer,
        output: MTLBuffer,
        elementCount: Int
    ) throws {
        guard elementCount > 0 else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "elementCount", value: elementCount)
        }
        let requiredBytes = try Self.checkedMultiply(
            elementCount, MemoryLayout<Float>.stride, operation: "gate bytes")
        try Self.requireBuffer(attention, named: "attention", bytes: requiredBytes)
        try Self.requireBuffer(rawGate, named: "rawGate", bytes: requiredBytes)
        try Self.requireBuffer(output, named: "output", bytes: requiredBytes)
        guard commandBuffer.status == .notEnqueued else {
            throw QwenFullAttentionError.commandBufferAlreadySubmitted
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw QwenFullAttentionError.commandEncoderUnavailable
        }
        var parameters = GateParameters(
            elementCount: try Self.uint32(elementCount, field: "elementCount"))
        encoder.setComputePipelineState(gatePipeline)
        encoder.setBytes(
            &parameters,
            length: MemoryLayout<GateParameters>.stride,
            index: QwenMetalBufferIndex.parameters.rawValue)
        encoder.setBuffer(attention, offset: 0, index: QwenMetalBufferIndex.input.rawValue)
        encoder.setBuffer(output, offset: 0, index: QwenMetalBufferIndex.output.rawValue)
        encoder.setBuffer(rawGate, offset: 0, index: QwenMetalBufferIndex.scratch.rawValue)
        try dispatch(encoder: encoder, pipeline: gatePipeline, count: elementCount)
        encoder.endEncoding()
    }

    /// Evaluates prefill or decode from already projected Q+gate, K, and V.
    /// Cached K/V are the prior token-major, normalized/rotated K and raw V.
    static func evaluate(
        configuration: QwenFullAttentionConfiguration,
        queryAndGateProjection: [Float],
        keyProjection: [Float],
        valueProjection: [Float],
        queryNormWeight: [Float],
        keyNormWeight: [Float],
        positions: [Int],
        mropePositions: [QwenMRoPEPosition]? = nil,
        mropeSections: [Int]? = nil,
        cachedKeys: [Float] = [],
        cachedValues: [Float] = [],
        outputProjection: [Float],
        outputDimension: Int,
        outputBias: [Float]? = nil
    ) throws -> QwenFullAttentionResult {
        let tokenCount = positions.count
        guard tokenCount > 0 else {
            throw QwenFullAttentionError.invalidConfiguration(field: "tokenCount", value: tokenCount)
        }
        let queryWidth = try checkedMultiply(
            configuration.queryHeadCount,
            configuration.headDimension,
            operation: "query width")
        let keyValueWidth = try checkedMultiply(
            configuration.keyValueHeadCount,
            configuration.headDimension,
            operation: "key/value width")
        let doubledQueryWidth = try checkedMultiply(queryWidth, 2, operation: "doubled query width")
        try requireCount(
            queryAndGateProjection,
            field: "queryAndGateProjection",
            expected: try checkedMultiply(tokenCount, doubledQueryWidth, operation: "query projection count"))
        try requireCount(
            keyProjection,
            field: "keyProjection",
            expected: try checkedMultiply(tokenCount, keyValueWidth, operation: "key projection count"))
        try requireCount(
            valueProjection,
            field: "valueProjection",
            expected: try checkedMultiply(tokenCount, keyValueWidth, operation: "value projection count"))
        try requireCount(
            queryNormWeight, field: "queryNormWeight", expected: configuration.headDimension)
        try requireCount(
            keyNormWeight, field: "keyNormWeight", expected: configuration.headDimension)
        guard cachedKeys.count == cachedValues.count,
              cachedKeys.count.isMultiple(of: keyValueWidth) else {
            throw QwenFullAttentionError.invalidCount(
                field: "cachedKeys/cachedValues", expected: cachedKeys.count, actual: cachedValues.count)
        }
        let cachedTokenCount = cachedKeys.count / keyValueWidth
        for (index, position) in positions.enumerated() {
            let expected = try checkedAdd(cachedTokenCount, index, operation: "expected position")
            guard position == expected else {
                throw QwenFullAttentionError.invalidPosition(expected: expected, actual: position)
            }
            guard UInt32(exactly: position) != nil else {
                throw QwenFullAttentionError.positionOutOfRange(position)
            }
        }
        guard outputDimension > 0 else {
            throw QwenFullAttentionError.invalidConfiguration(
                field: "outputDimension", value: outputDimension)
        }
        try requireCount(
            outputProjection,
            field: "outputProjection",
            expected: try checkedMultiply(outputDimension, queryWidth, operation: "output projection count"))
        if let outputBias {
            try requireCount(outputBias, field: "outputBias", expected: outputDimension)
        }

        var query = [Float]()
        var rawGate = [Float]()
        query.reserveCapacity(tokenCount * queryWidth)
        rawGate.reserveCapacity(tokenCount * queryWidth)
        for token in 0..<tokenCount {
            let tokenBase = token * doubledQueryWidth
            for head in 0..<configuration.queryHeadCount {
                let headBase = tokenBase + head * configuration.headDimension * 2
                query.append(contentsOf: queryAndGateProjection[
                    headBase..<(headBase + configuration.headDimension)])
                let gateBase = headBase + configuration.headDimension
                rawGate.append(contentsOf: queryAndGateProjection[
                    gateBase..<(gateBase + configuration.headDimension)])
            }
        }

        let normalizedQuery = normalize(
            query,
            weights: queryNormWeight,
            tokenCount: tokenCount,
            headCount: configuration.queryHeadCount,
            configuration: configuration)
        let normalizedKey = normalize(
            keyProjection,
            weights: keyNormWeight,
            tokenCount: tokenCount,
            headCount: configuration.keyValueHeadCount,
            configuration: configuration)
        if let mropePositions {
            guard mropePositions.count == tokenCount,
                  let mropeSections,
                  mropeSections.count == 3,
                  mropeSections.reduce(0, +) == configuration.rotaryDimension / 2 else {
                throw QwenFullAttentionError.invalidCount(
                    field: "mropePositions/sections", expected: tokenCount,
                    actual: mropePositions.count)
            }
        } else if mropeSections != nil {
            throw QwenFullAttentionError.invalidCount(
                field: "mropePositions", expected: tokenCount, actual: 0)
        }
        let rotatedQuery = try rotate(
            normalizedQuery,
            positions: positions,
            mropePositions: mropePositions,
            mropeSections: mropeSections,
            headCount: configuration.queryHeadCount,
            configuration: configuration)
        let rotatedKey = try rotate(
            normalizedKey,
            positions: positions,
            mropePositions: mropePositions,
            mropeSections: mropeSections,
            headCount: configuration.keyValueHeadCount,
            configuration: configuration)
        let allKeys = cachedKeys + rotatedKey
        let allValues = cachedValues + valueProjection
        let totalKeyTokens = cachedTokenCount + tokenCount
        let groupSize = configuration.queryHeadCount / configuration.keyValueHeadCount
        let scale = 1 / Float(configuration.headDimension).squareRoot()

        var attentionWeights = [Float](
            repeating: 0,
            count: tokenCount * configuration.queryHeadCount * totalKeyTokens)
        var preOutputGate = [Float](repeating: 0, count: tokenCount * queryWidth)
        for token in 0..<tokenCount {
            let maximumKey = cachedTokenCount + token
            for queryHead in 0..<configuration.queryHeadCount {
                let keyValueHead = queryHead / groupSize
                let queryBase = (token * configuration.queryHeadCount + queryHead)
                    * configuration.headDimension
                var scores = [Float]()
                scores.reserveCapacity(maximumKey + 1)
                for keyToken in 0...maximumKey {
                    let keyBase = (keyToken * configuration.keyValueHeadCount + keyValueHead)
                        * configuration.headDimension
                    var dot: Float = 0
                    for dimension in 0..<configuration.headDimension {
                        dot += rotatedQuery[queryBase + dimension]
                            * allKeys[keyBase + dimension]
                    }
                    scores.append(dot * scale)
                }
                let maximumScore = scores.max() ?? 0
                var denominator: Float = 0
                for index in scores.indices {
                    scores[index] = Float(Foundation.exp(Double(scores[index] - maximumScore)))
                    denominator += scores[index]
                }
                let weightBase = (token * configuration.queryHeadCount + queryHead)
                    * totalKeyTokens
                let outputBase = queryBase
                for keyToken in scores.indices {
                    let probability = scores[keyToken] / denominator
                    attentionWeights[weightBase + keyToken] = probability
                    let valueBase = (keyToken * configuration.keyValueHeadCount + keyValueHead)
                        * configuration.headDimension
                    for dimension in 0..<configuration.headDimension {
                        preOutputGate[outputBase + dimension] += probability
                            * allValues[valueBase + dimension]
                    }
                }
            }
        }

        let outputGate = rawGate.map { 1 / (1 + Float(Foundation.exp(Double(-$0)))) }
        let gatedAttention = zip(preOutputGate, outputGate).map(*)
        var output = [Float](repeating: 0, count: tokenCount * outputDimension)
        for token in 0..<tokenCount {
            let inputBase = token * queryWidth
            let outputBase = token * outputDimension
            for row in 0..<outputDimension {
                var value = outputBias?[row] ?? 0
                let weightBase = row * queryWidth
                for column in 0..<queryWidth {
                    value += outputProjection[weightBase + column]
                        * gatedAttention[inputBase + column]
                }
                output[outputBase + row] = value
            }
        }

        return QwenFullAttentionResult(
            queryProjection: query,
            normalizedQuery: normalizedQuery,
            normalizedKey: normalizedKey,
            rotatedQuery: rotatedQuery,
            rotatedKey: rotatedKey,
            attentionWeights: attentionWeights,
            preOutputGate: preOutputGate,
            outputGate: outputGate,
            gatedAttention: gatedAttention,
            output: output)
    }

    private static func normalize(
        _ input: [Float],
        weights: [Float],
        tokenCount: Int,
        headCount: Int,
        configuration: QwenFullAttentionConfiguration
    ) -> [Float] {
        var result = [Float](repeating: 0, count: input.count)
        let dimension = configuration.headDimension
        for item in 0..<(tokenCount * headCount) {
            let base = item * dimension
            var squareSum: Float = 0
            for index in 0..<dimension {
                squareSum += input[base + index] * input[base + index]
            }
            let inverseRMS = 1 / (squareSum / Float(dimension) + configuration.epsilon).squareRoot()
            for index in 0..<dimension {
                result[base + index] = input[base + index]
                    * inverseRMS * (1 + weights[index])
            }
        }
        return result
    }

    private static func rotate(
        _ input: [Float],
        positions: [Int],
        mropePositions: [QwenMRoPEPosition]?,
        mropeSections: [Int]?,
        headCount: Int,
        configuration: QwenFullAttentionConfiguration
    ) throws -> [Float] {
        var result = input
        let dimension = configuration.headDimension
        let rotaryDimension = configuration.rotaryDimension
        let half = rotaryDimension / 2
        for token in positions.indices {
            for head in 0..<headCount {
                let base = (token * headCount + head) * dimension
                for frequencyIndex in 0..<half {
                    let exponent = Float(2 * frequencyIndex) / Float(rotaryDimension)
                    let inverseFrequency = Float(
                        Foundation.pow(Double(configuration.theta), Double(-exponent)))
                    let position: Int
                    if let mropePositions, let mropeSections {
                        let axis = try QwenMultimodalPositions.axis(
                            frequencyIndex: frequencyIndex, sections: mropeSections)
                        position = Int(mropePositions[token].value(axis: axis))
                    } else {
                        position = positions[token]
                    }
                    let angle = Float(position) * inverseFrequency
                    let cosine = Float(Foundation.cos(Double(angle)))
                    let sine = Float(Foundation.sin(Double(angle)))
                    let first = input[base + frequencyIndex]
                    let second = input[base + frequencyIndex + half]
                    result[base + frequencyIndex] = first * cosine - second * sine
                    result[base + frequencyIndex + half] = second * cosine + first * sine
                }
            }
        }
        return result
    }

    private func dispatch(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        count: Int
    ) throws {
        let maximum = min(
            pipeline.maxTotalThreadsPerThreadgroup,
            device.maxThreadsPerThreadgroup.width)
        guard maximum > 0 else { throw QwenFullAttentionError.invalidPipelineLimit }
        let preferred = max(1, pipeline.threadExecutionWidth)
        let width = min(count, min(maximum, preferred))
        guard width > 0 else { throw QwenFullAttentionError.invalidPipelineLimit }
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private static func checkedPositionRange(start: Int, tokenCount: Int) throws -> UInt32 {
        guard start >= 0 else { throw QwenFullAttentionError.positionOutOfRange(start) }
        let last = try checkedAdd(start, tokenCount - 1, operation: "last position")
        guard UInt32(exactly: last) != nil, let result = UInt32(exactly: start) else {
            throw QwenFullAttentionError.positionOutOfRange(last)
        }
        return result
    }

    private static func requireBuffer(_ buffer: MTLBuffer, named name: String, bytes: Int) throws {
        guard buffer.length >= bytes else {
            throw QwenFullAttentionError.bufferTooSmall(
                name: name, required: bytes, actual: buffer.length)
        }
    }

    private static func requireCount(
        _ values: [Float], field: String, expected: Int
    ) throws {
        guard values.count == expected else {
            throw QwenFullAttentionError.invalidCount(
                field: field, expected: expected, actual: values.count)
        }
    }

    private static func checkedMultiply(
        _ lhs: Int, _ rhs: Int, operation: String
    ) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw QwenFullAttentionError.arithmeticOverflow(operation: operation) }
        return result
    }

    private static func checkedAdd(
        _ lhs: Int, _ rhs: Int, operation: String
    ) throws -> Int {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw QwenFullAttentionError.arithmeticOverflow(operation: operation) }
        return result
    }

    private static func uint32(_ value: Int, field: String) throws -> UInt32 {
        guard let result = UInt32(exactly: value) else {
            throw QwenFullAttentionError.invalidConfiguration(field: field, value: value)
        }
        return result
    }
}
