import Darwin
import Foundation
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Test-owned two-layer source fixture and CPU oracle for Phase 12. Payloads
/// are deterministic literal BF16 words; expected values never come from a
/// runtime model, Metal result, checkpoint, or shared Phase 11 fixture.
struct QwenBF16TextRunnerFixture {
    static let hiddenSize = 8
    static let vocabularySize = 5
    static let expertCount = 9
    static let topK = 8
    static let layerCount = 2
    static let routedIntermediateSize = 2
    static let sharedIntermediateSize = 2
    static let queryHeads = 2
    static let keyValueHeads = 1
    static let headDimension = 4
    static let rotaryDimension = 2
    static let linearKeyHeads = 1
    static let linearKeyDimension = 2
    static let linearValueHeads = 1
    static let linearValueDimension = 2
    static let convolutionWidth = 4
    static let epsilon: Float = 1e-6
    static let ropeTheta: Float = 10_000
    static let inputTokens: [Int32] = [1, 2]
    static let expectedDecodedVectorElements = 76

    /// Frozen before any runtime submission. Eight-wide projection dots,
    /// two-token causal attention/recurrence, and two composed MoE layers use
    /// FP32 accumulators. This envelope covers FP32 reduction/FMA and exp,
    /// sqrt, sine, and cosine variation; it must not be widened after GPU data.
    static let absoluteTolerance: Float = 2e-5
    static let relativeTolerance: Float = 2e-5
    static let negativeControlMinimum: Float = 1e-4
    static let routerCutoffMinimum: Float = 1e-2

    static let embeddingName = "model.language_model.embed_tokens.weight"
    static let outputHeadName = "lm_head.weight"
    static let finalNormName = "model.language_model.norm.weight"

    static let layer0Prefix = "model.language_model.layers.0."
    static let layer1Prefix = "model.language_model.layers.1."

    static let residentMatrixNames: Set<String> = [
        embeddingName, outputHeadName,
        layer0Prefix + "mlp.gate.weight",
        layer0Prefix + "mlp.shared_expert.gate_proj.weight",
        layer0Prefix + "mlp.shared_expert.up_proj.weight",
        layer0Prefix + "mlp.shared_expert.down_proj.weight",
        layer0Prefix + "mlp.shared_expert_gate.weight",
        layer0Prefix + "self_attn.q_proj.weight",
        layer0Prefix + "self_attn.k_proj.weight",
        layer0Prefix + "self_attn.v_proj.weight",
        layer0Prefix + "self_attn.o_proj.weight",
        layer1Prefix + "mlp.gate.weight",
        layer1Prefix + "mlp.shared_expert.gate_proj.weight",
        layer1Prefix + "mlp.shared_expert.up_proj.weight",
        layer1Prefix + "mlp.shared_expert.down_proj.weight",
        layer1Prefix + "mlp.shared_expert_gate.weight",
        layer1Prefix + "linear_attn.in_proj_qkv.weight",
        layer1Prefix + "linear_attn.in_proj_z.weight",
        layer1Prefix + "linear_attn.in_proj_b.weight",
        layer1Prefix + "linear_attn.in_proj_a.weight",
        layer1Prefix + "linear_attn.out_proj.weight",
    ]

    static let routedTensorNames: Set<String> = [
        layer0Prefix + "mlp.experts.gate_up_proj",
        layer0Prefix + "mlp.experts.down_proj",
        layer1Prefix + "mlp.experts.gate_up_proj",
        layer1Prefix + "mlp.experts.down_proj",
    ]

    struct LiteralTensor: Equatable, Sendable {
        let name: String
        let shape: [Int]
        let words: [UInt16]

        init(name: String, shape: [Int], words: [UInt16]) {
            precondition(!name.isEmpty && !shape.isEmpty && shape.allSatisfy { $0 > 0 })
            precondition(shape.reduce(1, *) == words.count)
            self.name = name
            self.shape = shape
            self.words = words
        }

        var byteCount: UInt64 { UInt64(words.count * MemoryLayout<UInt16>.stride) }
        var floats: [Float] { words.map(Self.floatFromBF16) }

        static func floatFromBF16(_ word: UInt16) -> Float {
            Float(bitPattern: UInt32(word) << 16)
        }
    }

    struct Source: Sendable {
        let root: URL
        let sourceRoot: URL
        let registrationURL: URL
        let shardNames: [String]
        let tensors: [String: LiteralTensor]
        let tensorToShard: [String: String]
        let configJSON: Data
        let indexJSON: Data
        let expectedResidentBF16Bytes: UInt64
        let expectedFP32VectorBytes: UInt64
        /// Dense BF16 matrices plus decoded FP32 vectors. Expert cache bytes
        /// are added only by totalResidencyBudget.
        let expectedResidentBytes: UInt64

        /// Dense BF16 matrices and decoded FP32 vectors only. The routed
        /// expert cache is accounted separately by totalResidencyBudget.
        func totalResidencyBudget(
            expertSlotCount: Int = QwenBF16TextRunnerFixture.topK
        ) -> UInt64 {
            let cacheBytes = QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
                expertSlotCount: expertSlotCount,
                layerCount: QwenBF16TextRunnerFixture.layerCount)
            let (total, overflow) = expectedResidentBytes.addingReportingOverflow(cacheBytes)
            precondition(!overflow, "synthetic fixture residency budget overflow")
            return total
        }

        func remove() {
            try? FileManager.default.removeItem(at: root)
        }

        func tensor(_ name: String) -> LiteralTensor {
            guard let tensor = tensors[name] else {
                preconditionFailure("missing test literal tensor: \(name)")
            }
            return tensor
        }

        func oracle() -> Reference {
            Reference(tensors: tensors)
        }
    }

    struct ReferenceMutation: Sendable {
        var suppressFullAttention = false
        var suppressLinearAttention = false
        var suppressSharedExpert = false
        var suppressRoutedExperts = false
        var omitEarlierFullAttentionKeys = false
        var excludeHighestRouterExpert = false
        var resetLinearStateEachToken = false
        var useIncorrectPlainRMSGain = false

        static let none = Self()
    }

    struct LayerTrace: Sendable {
        let mixerOutput: [Float]
        let sharedOutput: [Float]
        let routedOutput: [Float]
        let selectedExperts: [Int]
        let routerCutoffGap: Float
    }

    struct TokenResult: Sendable {
        let token: Int32
        let position: Int
        let logits: [Float]
        let predictedToken: Int32
        let layers: [LayerTrace]
    }

    struct Reference {
        private static let hiddenSize = QwenBF16TextRunnerFixture.hiddenSize
        private static let vocabularySize = QwenBF16TextRunnerFixture.vocabularySize
        private static let expertCount = QwenBF16TextRunnerFixture.expertCount
        private static let topK = QwenBF16TextRunnerFixture.topK
        private static let routedIntermediateSize =
            QwenBF16TextRunnerFixture.routedIntermediateSize
        private static let sharedIntermediateSize =
            QwenBF16TextRunnerFixture.sharedIntermediateSize
        private static let queryHeads = QwenBF16TextRunnerFixture.queryHeads
        private static let keyValueHeads = QwenBF16TextRunnerFixture.keyValueHeads
        private static let headDimension = QwenBF16TextRunnerFixture.headDimension
        private static let rotaryDimension = QwenBF16TextRunnerFixture.rotaryDimension
        private static let linearKeyHeads = QwenBF16TextRunnerFixture.linearKeyHeads
        private static let linearKeyDimension = QwenBF16TextRunnerFixture.linearKeyDimension
        private static let linearValueHeads = QwenBF16TextRunnerFixture.linearValueHeads
        private static let linearValueDimension = QwenBF16TextRunnerFixture.linearValueDimension
        private static let convolutionWidth = QwenBF16TextRunnerFixture.convolutionWidth
        private static let epsilon = QwenBF16TextRunnerFixture.epsilon
        private static let ropeTheta = QwenBF16TextRunnerFixture.ropeTheta
        private static let embeddingName = QwenBF16TextRunnerFixture.embeddingName
        private static let outputHeadName = QwenBF16TextRunnerFixture.outputHeadName
        private static let finalNormName = QwenBF16TextRunnerFixture.finalNormName
        private static let layer0Prefix = QwenBF16TextRunnerFixture.layer0Prefix
        private static let layer1Prefix = QwenBF16TextRunnerFixture.layer1Prefix

        private struct State {
            var fullKeys: [Float] = []
            var fullValues: [Float] = []
            var convolutionHistory = [Float](repeating: 0,
                count: 6 * QwenBF16TextRunnerFixture.convolutionWidth)
            var recurrent = [Float](repeating: 0,
                count: QwenBF16TextRunnerFixture.linearValueHeads
                    * QwenBF16TextRunnerFixture.linearKeyDimension
                    * QwenBF16TextRunnerFixture.linearValueDimension)
        }

        private let tensors: [String: LiteralTensor]
        private var state = State()

        fileprivate init(tensors: [String: LiteralTensor]) {
            self.tensors = tensors
        }

        mutating func append(
            token: Int32,
            position: Int,
            mutation: ReferenceMutation = .none
        ) -> TokenResult {
            precondition(Int(token) >= 0 && Int(token) < Self.vocabularySize)
            precondition(position >= 0)
            let embedding = floats(Self.embeddingName)
            let hiddenStart = Int(token) * Self.hiddenSize
            var hidden = Array(embedding[hiddenStart..<(hiddenStart + Self.hiddenSize)])
            var traces: [LayerTrace] = []
            traces.reserveCapacity(2)

            let full = fullAttention(
                input: Self.rmsNorm(
                    hidden, weights: floats(Self.layer0Prefix + "input_layernorm.weight"),
                    useIncorrectPlainGain: mutation.useIncorrectPlainRMSGain),
                position: position,
                omitEarlierKeys: mutation.omitEarlierFullAttentionKeys)
            let fullOutput = mutation.suppressFullAttention
                ? [Float](repeating: 0, count: Self.hiddenSize) : full.output
            hidden = zip(hidden, fullOutput).map(+)
            let fullPostNorm = Self.rmsNorm(
                hidden, weights: floats(Self.layer0Prefix + "post_attention_layernorm.weight"),
                useIncorrectPlainGain: mutation.useIncorrectPlainRMSGain)
            let fullMoE = moe(layerPrefix: Self.layer0Prefix, input: fullPostNorm,
                              mutation: mutation)
            hidden = zip(zip(hidden, fullMoE.routed), fullMoE.shared).map {
                $0.0.0 + (mutation.suppressRoutedExperts ? 0 : $0.0.1)
                    + (mutation.suppressSharedExpert ? 0 : $0.1)
            }
            traces.append(LayerTrace(
                mixerOutput: full.output,
                sharedOutput: fullMoE.shared,
                routedOutput: fullMoE.routed,
                selectedExperts: fullMoE.experts,
                routerCutoffGap: fullMoE.cutoffGap))

            let linearLayerInput = Self.rmsNorm(
                hidden, weights: floats(Self.layer1Prefix + "input_layernorm.weight"),
                useIncorrectPlainGain: mutation.useIncorrectPlainRMSGain)
            let linear = linearAttention(
                input: linearLayerInput,
                resetState: mutation.resetLinearStateEachToken)
            let linearOutput = mutation.suppressLinearAttention
                ? [Float](repeating: 0, count: Self.hiddenSize) : linear.output
            hidden = zip(hidden, linearOutput).map(+)
            let linearPostNorm = Self.rmsNorm(
                hidden, weights: floats(Self.layer1Prefix + "post_attention_layernorm.weight"),
                useIncorrectPlainGain: mutation.useIncorrectPlainRMSGain)
            let linearMoE = moe(layerPrefix: Self.layer1Prefix, input: linearPostNorm,
                                mutation: mutation)
            hidden = zip(zip(hidden, linearMoE.routed), linearMoE.shared).map {
                $0.0.0 + (mutation.suppressRoutedExperts ? 0 : $0.0.1)
                    + (mutation.suppressSharedExpert ? 0 : $0.1)
            }
            traces.append(LayerTrace(
                mixerOutput: linear.output,
                sharedOutput: linearMoE.shared,
                routedOutput: linearMoE.routed,
                selectedExperts: linearMoE.experts,
                routerCutoffGap: linearMoE.cutoffGap))

            let finalHidden = Self.rmsNorm(
                hidden, weights: floats(Self.finalNormName),
                useIncorrectPlainGain: mutation.useIncorrectPlainRMSGain)
            let logits = Self.project(finalHidden, bits: tensors[Self.outputHeadName]!.words,
                                      rows: Self.vocabularySize, columns: Self.hiddenSize)
            return TokenResult(
                token: token,
                position: position,
                logits: logits,
                predictedToken: Self.greedyToken(logits),
                layers: traces)
        }

        mutating func run(
            tokens: [Int32],
            mutation: ReferenceMutation = .none
        ) -> [TokenResult] {
            state = State()
            var result: [TokenResult] = []
            result.reserveCapacity(tokens.count)
            for (position, token) in tokens.enumerated() {
                result.append(append(token: token, position: position, mutation: mutation))
            }
            return result
        }

        func maxMagnitude(_ values: [Float]) -> Float {
            values.map { abs($0) }.max() ?? 0
        }

        private func floats(_ name: String) -> [Float] {
            tensors[name]!.floats
        }

        private mutating func fullAttention(
            input: [Float], position: Int, omitEarlierKeys: Bool
        ) -> (output: [Float], rotatedKeys: [Float]) {
            let prefix = Self.layer0Prefix + "self_attn."
            let queryAndGate = Self.project(input, bits: tensors[prefix + "q_proj.weight"]!.words,
                                            rows: 2 * Self.queryHeads * Self.headDimension,
                                            columns: Self.hiddenSize)
            let key = Self.project(input, bits: tensors[prefix + "k_proj.weight"]!.words,
                                   rows: Self.keyValueHeads * Self.headDimension,
                                   columns: Self.hiddenSize)
            let value = Self.project(input, bits: tensors[prefix + "v_proj.weight"]!.words,
                                     rows: Self.keyValueHeads * Self.headDimension,
                                     columns: Self.hiddenSize)
            var query = [Float](repeating: 0, count: Self.queryHeads * Self.headDimension)
            var outputGate = [Float](repeating: 0, count: Self.queryHeads * Self.headDimension)
            for head in 0..<Self.queryHeads {
                for dimension in 0..<Self.headDimension {
                    let source = head * (2 * Self.headDimension)
                    query[head * Self.headDimension + dimension]
                        = queryAndGate[source + dimension]
                    outputGate[head * Self.headDimension + dimension]
                        = queryAndGate[source + Self.headDimension + dimension]
                }
            }
            let queryNorm = floats(prefix + "q_norm.weight")
            let keyNorm = floats(prefix + "k_norm.weight")
            query = Self.headRMSNorm(query, heads: Self.queryHeads, weights: queryNorm)
            var normalizedKey = Self.headRMSNorm(key, heads: Self.keyValueHeads, weights: keyNorm)
            query = Self.partialRoPE(query, heads: Self.queryHeads, position: position)
            normalizedKey = Self.partialRoPE(normalizedKey, heads: Self.keyValueHeads, position: position)
            let priorKeys = state.fullKeys
            let priorValues = state.fullValues
            state.fullKeys += normalizedKey
            state.fullValues += value
            let keyStream: [Float]
            let valueStream: [Float]
            if omitEarlierKeys {
                keyStream = normalizedKey
                valueStream = value
            } else {
                keyStream = state.fullKeys
                valueStream = state.fullValues
            }
            _ = priorKeys
            _ = priorValues
            let keyTokenCount = keyStream.count / (Self.keyValueHeads * Self.headDimension)
            var attended = [Float](repeating: 0, count: Self.queryHeads * Self.headDimension)
            let groupSize = Self.queryHeads / Self.keyValueHeads
            let scale = 1 / sqrt(Float(Self.headDimension))
            for head in 0..<Self.queryHeads {
                let kvHead = head / groupSize
                let queryStart = head * Self.headDimension
                var scores: [Float] = []
                for keyToken in 0..<keyTokenCount {
                    let keyStart = (keyToken * Self.keyValueHeads + kvHead) * Self.headDimension
                    var score: Float = 0
                    for dimension in 0..<Self.headDimension {
                        score += query[queryStart + dimension]
                            * keyStream[keyStart + dimension]
                    }
                    scores.append(score * scale)
                }
                let maximum = scores.max() ?? 0
                let exponentials = scores.map { Self.exp($0 - maximum) }
                let denominator = exponentials.reduce(Float(0), +)
                for keyToken in 0..<keyTokenCount {
                    let probability = exponentials[keyToken] / denominator
                    let valueStart = (keyToken * Self.keyValueHeads + kvHead) * Self.headDimension
                    for dimension in 0..<Self.headDimension {
                        attended[queryStart + dimension] += probability
                            * valueStream[valueStart + dimension]
                    }
                }
            }
            for index in attended.indices {
                attended[index] *= Self.sigmoid(outputGate[index])
            }
            let output = Self.project(attended,
                                      bits: tensors[prefix + "o_proj.weight"]!.words,
                                      rows: Self.hiddenSize,
                                      columns: Self.queryHeads * Self.headDimension)
            return (output, normalizedKey)
        }

        private mutating func linearAttention(
            input: [Float], resetState: Bool
        ) -> (output: [Float], recurrentOutput: [Float]) {
            let prefix = Self.layer1Prefix + "linear_attn."
            let localHistory = resetState
                ? [Float](repeating: 0, count: 6 * Self.convolutionWidth)
                : state.convolutionHistory
            let localRecurrent = resetState
                ? [Float](repeating: 0, count: Self.linearValueHeads
                    * Self.linearKeyDimension * Self.linearValueDimension)
                : state.recurrent
            let projectedQKV = Self.project(
                input, bits: tensors[prefix + "in_proj_qkv.weight"]!.words,
                rows: 6, columns: Self.hiddenSize)
            let z = Self.project(input, bits: tensors[prefix + "in_proj_z.weight"]!.words,
                                 rows: Self.linearValueDimension, columns: Self.hiddenSize)
            let rawBeta = Self.project(input, bits: tensors[prefix + "in_proj_b.weight"]!.words,
                                       rows: Self.linearValueHeads, columns: Self.hiddenSize)
            let rawA = Self.project(input, bits: tensors[prefix + "in_proj_a.weight"]!.words,
                                    rows: Self.linearValueHeads, columns: Self.hiddenSize)
            let convolution = floats(prefix + "conv1d.weight")
            var history = localHistory
            var convolved = [Float](repeating: 0, count: 6)
            for channel in 0..<6 {
                let base = channel * Self.convolutionWidth
                for index in 0..<(Self.convolutionWidth - 1) {
                    history[base + index] = history[base + index + 1]
                }
                history[base + Self.convolutionWidth - 1] = projectedQKV[channel]
                var sum: Float = 0
                for index in 0..<Self.convolutionWidth {
                    sum += history[base + index] * convolution[base + index]
                }
                convolved[channel] = Self.silu(sum)
            }
            let keyWidth = Self.linearKeyHeads * Self.linearKeyDimension
            let valueWidth = Self.linearValueHeads * Self.linearValueDimension
            let query = Array(convolved[0..<keyWidth])
            let key = Array(convolved[keyWidth..<(2 * keyWidth)])
            let value = Array(convolved[(2 * keyWidth)..<(2 * keyWidth + valueWidth)])
            let beta = Self.sigmoid(rawBeta[0])
            let aLog = floats(prefix + "A_log")[0]
            let timeBias = floats(prefix + "dt_bias")[0]
            let logDecay = -Self.exp(aLog) * Self.softplus(rawA[0] + timeBias)
            let decay = Self.exp(logDecay)
            var recurrent = localRecurrent
            for index in recurrent.indices { recurrent[index] *= decay }
            var querySquare: Float = 0
            var keySquare: Float = 0
            for index in 0..<Self.linearKeyDimension {
                querySquare += query[index] * query[index]
                keySquare += key[index] * key[index]
            }
            let inverseQueryNorm = 1 / sqrt(querySquare + Self.epsilon)
                / sqrt(Float(Self.linearKeyDimension))
            let inverseKeyNorm = 1 / sqrt(keySquare + Self.epsilon)
            var recurrentOutput = [Float](repeating: 0, count: valueWidth)
            for valueIndex in 0..<Self.linearValueDimension {
                var prediction: Float = 0
                for keyIndex in 0..<Self.linearKeyDimension {
                    prediction += recurrent[keyIndex * Self.linearValueDimension + valueIndex]
                        * key[keyIndex] * inverseKeyNorm
                }
                let delta = (value[valueIndex] - prediction) * beta
                for keyIndex in 0..<Self.linearKeyDimension {
                    recurrent[keyIndex * Self.linearValueDimension + valueIndex] +=
                        key[keyIndex] * inverseKeyNorm * delta
                }
            }
            for valueIndex in 0..<Self.linearValueDimension {
                var sum: Float = 0
                for keyIndex in 0..<Self.linearKeyDimension {
                    sum += recurrent[keyIndex * Self.linearValueDimension + valueIndex]
                        * query[keyIndex] * inverseQueryNorm
                }
                recurrentOutput[valueIndex] = sum
            }
            let normalization = floats(prefix + "norm.weight")
            var gated = [Float](repeating: 0, count: valueWidth)
            var squareSum: Float = 0
            for valueIndex in 0..<Self.linearValueDimension {
                squareSum += recurrentOutput[valueIndex] * recurrentOutput[valueIndex]
            }
            let inverseRMS = 1 / sqrt(squareSum / Float(Self.linearValueDimension) + Self.epsilon)
            for valueIndex in 0..<Self.linearValueDimension {
                gated[valueIndex] = recurrentOutput[valueIndex] * inverseRMS
                    * normalization[valueIndex] * Self.silu(z[valueIndex])
            }
            let output = Self.project(gated, bits: tensors[prefix + "out_proj.weight"]!.words,
                                      rows: Self.hiddenSize, columns: valueWidth)
            if !resetState {
                state.convolutionHistory = history
                state.recurrent = recurrent
            }
            return (output, recurrentOutput)
        }

        private func moe(
            layerPrefix: String,
            input: [Float],
            mutation: ReferenceMutation
        ) -> (routed: [Float], shared: [Float], experts: [Int], cutoffGap: Float) {
            let routerBits = tensors[layerPrefix + "mlp.gate.weight"]!.words
            let routerLogits = Self.project(input, bits: routerBits,
                                            rows: Self.expertCount, columns: Self.hiddenSize)
            let sorted = routerLogits.indices.sorted {
                if routerLogits[$0] == routerLogits[$1] { return $0 < $1 }
                return routerLogits[$0] > routerLogits[$1]
            }
            let ordered: [Int]
            if mutation.excludeHighestRouterExpert {
                ordered = Array(sorted.dropFirst().prefix(Self.topK))
            } else {
                ordered = Array(sorted.prefix(Self.topK))
            }
            let cutoffGap = routerLogits[sorted[Self.topK - 1]] - routerLogits[sorted[Self.topK]]
            let maxLogit = routerLogits.max() ?? 0
            let exponentials = routerLogits.map { Self.exp($0 - maxLogit) }
            let probabilities = exponentials.map { $0 / exponentials.reduce(Float(0), +) }
            let selectedTotal = ordered.reduce(Float(0)) { $0 + probabilities[$1] }
            let routingWeights = ordered.map { probabilities[$0] / selectedTotal }

            var routed = [Float](repeating: 0, count: Self.hiddenSize)
            for (rank, expert) in ordered.enumerated() {
                let gateUp = tensors[layerPrefix + "mlp.experts.gate_up_proj"]!.words
                let expertGateUpOffset = expert * (2 * Self.routedIntermediateSize * Self.hiddenSize)
                let expertGateUpRange = expertGateUpOffset..<(expertGateUpOffset + 2 * Self.routedIntermediateSize * Self.hiddenSize)
                let projected = Self.project(
                    input,
                    bits: Array(gateUp[expertGateUpRange]),
                    rows: 2 * Self.routedIntermediateSize, columns: Self.hiddenSize)
                let activation = (0..<Self.routedIntermediateSize).map {
                    Self.silu(projected[$0]) * projected[Self.routedIntermediateSize + $0]
                }
                let downBits = tensors[layerPrefix + "mlp.experts.down_proj"]!.words
                let downOffset = expert * (Self.hiddenSize * Self.routedIntermediateSize)
                let down = Self.project(
                    activation,
                    bits: Array(downBits[downOffset..<(downOffset + Self.hiddenSize * Self.routedIntermediateSize)]),
                    rows: Self.hiddenSize, columns: Self.routedIntermediateSize)
                for index in 0..<Self.hiddenSize {
                    routed[index] += down[index] * routingWeights[rank]
                }
            }

            let sharedPrefix = layerPrefix + "mlp.shared_expert."
            let sharedGate = Self.project(
                input, bits: tensors[sharedPrefix + "gate_proj.weight"]!.words,
                rows: Self.sharedIntermediateSize, columns: Self.hiddenSize)
            let sharedUp = Self.project(
                input, bits: tensors[sharedPrefix + "up_proj.weight"]!.words,
                rows: Self.sharedIntermediateSize, columns: Self.hiddenSize)
            let sharedActivation = (0..<Self.sharedIntermediateSize).map {
                Self.silu(sharedGate[$0]) * sharedUp[$0]
            }
            let sharedDown = Self.project(
                sharedActivation,
                bits: tensors[sharedPrefix + "down_proj.weight"]!.words,
                rows: Self.hiddenSize, columns: Self.sharedIntermediateSize)
            let outputGate = Self.project(
                input, bits: tensors[layerPrefix + "mlp.shared_expert_gate.weight"]!.words,
                rows: 1, columns: Self.hiddenSize)[0]
            let sharedGateValue = Self.sigmoid(outputGate)
            let shared = sharedDown.map { $0 * sharedGateValue }
            return (routed, shared, ordered, cutoffGap)
        }

        private static func project(
            _ input: [Float], bits: [UInt16], rows: Int, columns: Int
        ) -> [Float] {
            precondition(input.count == columns && bits.count == rows * columns)
            var output = [Float](repeating: 0, count: rows)
            for row in 0..<rows {
                var sum: Float = 0
                for column in 0..<columns {
                    sum += input[column] * LiteralTensor.floatFromBF16(bits[row * columns + column])
                }
                output[row] = sum
            }
            return output
        }

        private static func rmsNorm(
            _ input: [Float], weights: [Float], useIncorrectPlainGain: Bool
        ) -> [Float] {
            precondition(input.count == Self.hiddenSize && weights.count == Self.hiddenSize)
            var squareSum: Float = 0
            for value in input { squareSum += value * value }
            let inverseRMS = 1 / sqrt(squareSum / Float(Self.hiddenSize) + Self.epsilon)
            return input.indices.map { index in
                let gain = useIncorrectPlainGain ? weights[index] : 1 + weights[index]
                return input[index] * inverseRMS * gain
            }
        }

        private static func headRMSNorm(
            _ input: [Float], heads: Int, weights: [Float]
        ) -> [Float] {
            precondition(input.count == heads * Self.headDimension
                         && weights.count == Self.headDimension)
            var output = [Float](repeating: 0, count: input.count)
            for head in 0..<heads {
                let base = head * Self.headDimension
                var squareSum: Float = 0
                for dimension in 0..<Self.headDimension {
                    squareSum += input[base + dimension] * input[base + dimension]
                }
                let inverseRMS = 1 / sqrt(squareSum / Float(Self.headDimension) + Self.epsilon)
                for dimension in 0..<Self.headDimension {
                    output[base + dimension] = input[base + dimension] * inverseRMS
                        * (1 + weights[dimension])
                }
            }
            return output
        }

        private static func partialRoPE(
            _ input: [Float], heads: Int, position: Int
        ) -> [Float] {
            var output = input
            let half = Self.rotaryDimension / 2
            for head in 0..<heads {
                let base = head * Self.headDimension
                for frequencyIndex in 0..<half {
                    let exponent = Float(2 * frequencyIndex) / Float(Self.rotaryDimension)
                    let inverseFrequency = Float(pow(Double(Self.ropeTheta), Double(-exponent)))
                    let angle = Float(position) * inverseFrequency
                    let cosine = Float(cos(Double(angle)))
                    let sine = Float(sin(Double(angle)))
                    let first = input[base + frequencyIndex]
                    let second = input[base + frequencyIndex + half]
                    output[base + frequencyIndex] = first * cosine - second * sine
                    output[base + frequencyIndex + half] = second * cosine + first * sine
                }
            }
            return output
        }

        private static func exp(_ value: Float) -> Float {
            Float(Foundation.exp(Double(value)))
        }

        private static func sigmoid(_ value: Float) -> Float {
            1 / (1 + exp(-value))
        }

        private static func silu(_ value: Float) -> Float {
            value * sigmoid(value)
        }

        private static func softplus(_ value: Float) -> Float {
            max(value, 0) + Float(Foundation.log1p(Double(exp(-abs(value)))))
        }

        private static func greedyToken(_ logits: [Float]) -> Int32 {
            Int32(logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0)
        }
    }

    let root: URL
    let sourceRoot: URL
    let registrationURL: URL
    let shardNames: [String]
    let tensors: [String: LiteralTensor]
    let tensorToShard: [String: String]
    let configJSON: Data
    let indexJSON: Data
    let expectedResidentBytes: UInt64

    static func expectedExpertCacheBytes(
        expertSlotCount: Int,
        layerCount: Int = QwenBF16TextRunnerFixture.layerCount
    ) -> UInt64 {
        precondition(expertSlotCount > 0 && layerCount > 0)
        let pairBytes = UInt64(hiddenSize) * UInt64(routedIntermediateSize) * 6
        let (slotBytes, slotOverflow) = pairBytes.multipliedReportingOverflow(
            by: UInt64(expertSlotCount))
        let (cacheBytes, layerOverflow) = slotBytes.multipliedReportingOverflow(
            by: UInt64(layerCount))
        precondition(!slotOverflow && !layerOverflow,
                     "synthetic fixture expert cache budget overflow")
        return cacheBytes
    }

    static func make(
        configJSON overrideConfig: Data? = nil,
        shapeOverrides: [String: [Int]] = [:],
        outOfRangeTensor: String? = nil,
        missingIndexMappings: Set<String> = [],
        indexShardOverrides: [String: String] = [:]
    ) throws -> Source {
        let tensors = Dictionary(uniqueKeysWithValues: makeLiteralTensors().map {
            ($0.name, $0)
        })
        let identity = OfficialQwenSourceIdentity.pinned
        guard identity.shards.count >= 2 else {
            throw QwenBF16TextRunnerFixtureError.missingPinnedShardPair
        }
        let shardNames = [identity.shards[0].filename, identity.shards[1].filename]
        var mapping = Dictionary(uniqueKeysWithValues: tensors.keys.map { name in
            (name, routedTensorNames.contains(name) ? shardNames[1] : shardNames[0])
        })
        for (name, shardName) in indexShardOverrides { mapping[name] = shardName }
        for name in missingIndexMappings { mapping.removeValue(forKey: name) }
        let configJSON = overrideConfig ?? officialConfigJSON()
        let indexJSON = indexJSON(mapping: mapping)
        let manager = FileManager.default
        var canonicalTemporaryParent = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(manager.temporaryDirectory.path, &canonicalTemporaryParent) != nil else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let temporaryParentBytes = canonicalTemporaryParent.prefix { $0 != 0 }
            .map { UInt8(bitPattern: $0) }
        let temporaryParent = URL(
            fileURLWithPath: String(decoding: temporaryParentBytes, as: UTF8.self),
            isDirectory: true)
        let root = temporaryParent.appendingPathComponent(
            "qwen-bf16-text-runner-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: false,
                                    attributes: [.posixPermissions: 0o700])
        do {
            let sourceRoot = root.appendingPathComponent("source", isDirectory: true)
            let registrationParent = root.appendingPathComponent("models", isDirectory: true)
            try manager.createDirectory(at: sourceRoot, withIntermediateDirectories: false,
                                        attributes: [.posixPermissions: 0o700])
            try manager.createDirectory(at: registrationParent, withIntermediateDirectories: false,
                                        attributes: [.posixPermissions: 0o700])
            var residentCanonical = [CChar](repeating: 0, count: Int(PATH_MAX))
            guard realpath(sourceRoot.path, &residentCanonical) != nil else {
                throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
            }
            let sourcePathBytes = residentCanonical.prefix { $0 != 0 }
                .map { UInt8(bitPattern: $0) }
            let canonicalSourceRoot = URL(
                fileURLWithPath: String(decoding: sourcePathBytes, as: UTF8.self),
                isDirectory: true)
            let registrationURL = registrationParent.appendingPathComponent(
                "synthetic.gturbo", isDirectory: true)

            try configJSON.write(
                to: canonicalSourceRoot.appendingPathComponent("config.json"),
                options: .withoutOverwriting)
            try indexJSON.write(
                to: canonicalSourceRoot.appendingPathComponent("model.safetensors.index.json"),
                options: .withoutOverwriting)
            let residentTensors = tensors.values.filter { !routedTensorNames.contains($0.name) }
                .sorted { $0.name < $1.name }
            let routedTensors = tensors.values.filter { routedTensorNames.contains($0.name) }
                .sorted { $0.name < $1.name }
            let firstShard = try safetensorsFile(
                residentTensors, shapeOverrides: shapeOverrides,
                outOfRangeTensor: outOfRangeTensor)
            let secondShard = try safetensorsFile(
                routedTensors, shapeOverrides: shapeOverrides,
                outOfRangeTensor: outOfRangeTensor)
            try firstShard.write(
                to: canonicalSourceRoot.appendingPathComponent(shardNames[0]),
                options: .withoutOverwriting)
            try secondShard.write(
                to: canonicalSourceRoot.appendingPathComponent(shardNames[1]),
                options: .withoutOverwriting)

            let descriptor = try OfficialSourceDescriptor(
                repository: identity.repository,
                revision: identity.revision,
                storageProfile: identity.storageProfile,
                sidecarSHA256: identity.sidecarSHA256,
                shards: identity.shards.map {
                    OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
                },
                sourceRoot: canonicalSourceRoot.path)
            _ = try OfficialSourceRegistration.register(
                markerData: JSONEncoder().encode(descriptor), at: registrationURL)
            let expectedResidentBF16Bytes = tensors.values
                .filter { residentMatrixNames.contains($0.name) }
                .reduce(UInt64(0)) { $0 + $1.byteCount }
            let expectedFP32VectorBytes = UInt64(expectedDecodedVectorElements
                * MemoryLayout<Float>.stride)
            let expectedResidentBytes = expectedResidentBF16Bytes + expectedFP32VectorBytes
            return Source(
                root: root,
                sourceRoot: canonicalSourceRoot,
                registrationURL: registrationURL,
                shardNames: shardNames,
                tensors: Dictionary(uniqueKeysWithValues: tensors.values.map { ($0.name, $0) }),
                tensorToShard: mapping,
                configJSON: configJSON,
                indexJSON: indexJSON,
                expectedResidentBF16Bytes: expectedResidentBF16Bytes,
                expectedFP32VectorBytes: expectedFP32VectorBytes,
                expectedResidentBytes: expectedResidentBytes)
        } catch {
            try? manager.removeItem(at: root)
            throw error
        }
    }

    static func officialConfigJSON(attentionHeads: Int = queryHeads) -> Data {
        let root: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "text_config": [
                "model_type": "qwen3_5_moe_text",
                "dtype": "bfloat16",
                "mamba_ssm_dtype": "float32",
                "hidden_size": hiddenSize,
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "linear_attention"],
                "num_attention_heads": attentionHeads,
                "num_key_value_heads": keyValueHeads,
                "head_dim": headDimension,
                "attn_output_gate": true,
                "linear_conv_kernel_dim": convolutionWidth,
                "linear_num_key_heads": linearKeyHeads,
                "linear_key_head_dim": linearKeyDimension,
                "linear_num_value_heads": linearValueHeads,
                "linear_value_head_dim": linearValueDimension,
                "partial_rotary_factor": 0.5,
                "rope_parameters": ["rope_theta": ropeTheta, "mrope_section": [1]],
                "num_experts": expertCount,
                "num_experts_per_tok": topK,
                "moe_intermediate_size": routedIntermediateSize,
                "shared_expert_intermediate_size": sharedIntermediateSize,
                "vocab_size": vocabularySize,
                "tie_word_embeddings": false,
                "hidden_act": "silu",
                "rms_norm_eps": Double(1e-6),
                "bos_token_id": 0,
                "eos_token_id": 4,
            ],
            "image_token_id": 3,
            "video_token_id": 2,
            "vision_start_token_id": 0,
            "vision_end_token_id": 4,
        ]
        return try! JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }

    private static func indexJSON(mapping: [String: String]) -> Data {
        let root: [String: Any] = ["weight_map": mapping]
        return try! JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    }

    private static let smallMatrixPalette: [UInt16] = [
        0x3d00, // 0.03125
        0x3d80, // 0.0625
        0x3e00, // 0.125
        0x3e80, // 0.25
    ]
    private static let embeddingPalette: [UInt16] = [
        0x3e00, // 0.125
        0x3e80, // 0.25
        0x3f00, // 0.5
        0x3f40, // 0.75
    ]
    private static let routerCoefficients: [UInt16] = [
        0x3f00, // 0.5
        0x3ee0, // 0.4375
        0x3ec0, // 0.375
        0x3ea0, // 0.3125
        0x3e80, // 0.25
        0x3e40, // 0.1875
        0x3e00, // 0.125
        0x3d80, // 0.0625
        0xbf00, // -0.5: safely excluded by the Top-8 cutoff
    ]

    private static func makeLiteralTensors() -> [LiteralTensor] {
        var tensors: [LiteralTensor] = []
        func matrix(_ name: String, _ shape: [Int], salt: Int,
                    palette: [UInt16] = smallMatrixPalette) {
            tensors.append(LiteralTensor(
                name: name, shape: shape,
                words: patternWords(shape: shape, salt: salt, palette: palette)))
        }
        func vector(_ name: String, _ words: [UInt16]) {
            tensors.append(LiteralTensor(name: name, shape: [words.count], words: words))
        }

        matrix(embeddingName, [vocabularySize, hiddenSize], salt: 3,
               palette: embeddingPalette)
        matrix(outputHeadName, [vocabularySize, hiddenSize], salt: 19,
               palette: embeddingPalette)
        vector(finalNormName, [0x3f00, 0x3f80, 0x3fc0, 0x3f40,
                               0x3f80, 0x3f00, 0x3fc0, 0x3f40])

        addLayerCommon(&tensors, prefix: layer0Prefix, salt: 11)
        matrix(layer0Prefix + "self_attn.q_proj.weight",
               [2 * queryHeads * headDimension, hiddenSize], salt: 21)
        matrix(layer0Prefix + "self_attn.k_proj.weight",
               [keyValueHeads * headDimension, hiddenSize], salt: 22)
        matrix(layer0Prefix + "self_attn.v_proj.weight",
               [keyValueHeads * headDimension, hiddenSize], salt: 23)
        matrix(layer0Prefix + "self_attn.o_proj.weight",
               [hiddenSize, queryHeads * headDimension], salt: 24)
        vector(layer0Prefix + "self_attn.q_norm.weight",
               [0x3d00, 0x3d80, 0x3e00, 0x3d80])
        vector(layer0Prefix + "self_attn.k_norm.weight",
               [0x3e00, 0x3d00, 0x3d80, 0x3e00])
        matrix(layer0Prefix + "mlp.experts.gate_up_proj",
               [expertCount, 2 * routedIntermediateSize, hiddenSize], salt: 31)
        matrix(layer0Prefix + "mlp.experts.down_proj",
               [expertCount, hiddenSize, routedIntermediateSize], salt: 32)

        addLayerCommon(&tensors, prefix: layer1Prefix, salt: 41)
        let linear = layer1Prefix + "linear_attn."
        matrix(linear + "in_proj_qkv.weight", [6, hiddenSize], salt: 51)
        matrix(linear + "in_proj_z.weight", [linearValueDimension, hiddenSize], salt: 52)
        matrix(linear + "in_proj_b.weight", [linearValueHeads, hiddenSize], salt: 53)
        matrix(linear + "in_proj_a.weight", [linearValueHeads, hiddenSize], salt: 54)
        matrix(linear + "out_proj.weight", [hiddenSize, linearValueDimension], salt: 55)
        tensors.append(LiteralTensor(
            name: linear + "conv1d.weight",
            shape: [6, 1, convolutionWidth],
            words: patternWords(shape: [6, 1, convolutionWidth],
                                salt: 56, palette: smallMatrixPalette)))
        vector(linear + "norm.weight", [0x3f00, 0x3f80])
        vector(linear + "A_log", [0x0000])
        vector(linear + "dt_bias", [0x3d00])
        matrix(layer1Prefix + "mlp.experts.gate_up_proj",
               [expertCount, 2 * routedIntermediateSize, hiddenSize], salt: 61)
        matrix(layer1Prefix + "mlp.experts.down_proj",
               [expertCount, hiddenSize, routedIntermediateSize], salt: 62)
        return tensors
    }

    private static func addLayerCommon(
        _ tensors: inout [LiteralTensor],
        prefix: String,
        salt: Int
    ) {
        let firstGamma: [UInt16] = [
            0x3f00, 0x3f80, 0x3fc0, 0x3f40, 0x3f80, 0x3f00, 0x3fc0, 0x3f40,
        ]
        let secondGamma: [UInt16] = [
            0x3f80, 0x3f40, 0x3f00, 0x3fc0, 0x3f40, 0x3fc0, 0x3f80, 0x3f00,
        ]
        let inputGamma = salt.isMultiple(of: 2) ? firstGamma : secondGamma
        let postGamma = salt.isMultiple(of: 2) ? secondGamma : firstGamma
        tensors.append(LiteralTensor(
            name: prefix + "input_layernorm.weight", shape: [hiddenSize],
            words: inputGamma))
        tensors.append(LiteralTensor(
            name: prefix + "post_attention_layernorm.weight", shape: [hiddenSize],
            words: postGamma))
        var routerWords = [UInt16](repeating: 0, count: expertCount * hiddenSize)
        let selected = Array(routerCoefficients.prefix(topK))
        for expert in 0..<topK {
            routerWords[expert * hiddenSize]
                = selected[(expert + salt % topK) % topK]
        }
        routerWords[(expertCount - 1) * hiddenSize] = routerCoefficients[expertCount - 1]
        tensors.append(LiteralTensor(
            name: prefix + "mlp.gate.weight",
            shape: [expertCount, hiddenSize], words: routerWords))
        tensors.append(LiteralTensor(
            name: prefix + "mlp.shared_expert.gate_proj.weight",
            shape: [sharedIntermediateSize, hiddenSize],
            words: patternWords(shape: [sharedIntermediateSize, hiddenSize],
                                salt: salt + 2, palette: smallMatrixPalette)))
        tensors.append(LiteralTensor(
            name: prefix + "mlp.shared_expert.up_proj.weight",
            shape: [sharedIntermediateSize, hiddenSize],
            words: patternWords(shape: [sharedIntermediateSize, hiddenSize],
                                salt: salt + 3, palette: smallMatrixPalette)))
        tensors.append(LiteralTensor(
            name: prefix + "mlp.shared_expert.down_proj.weight",
            shape: [hiddenSize, sharedIntermediateSize],
            words: patternWords(shape: [hiddenSize, sharedIntermediateSize],
                                salt: salt + 4, palette: smallMatrixPalette)))
        tensors.append(LiteralTensor(
            name: prefix + "mlp.shared_expert_gate.weight",
            shape: [1, hiddenSize],
            words: patternWords(shape: [1, hiddenSize], salt: salt + 5,
                                palette: smallMatrixPalette)))
    }

    private static func patternWords(
        shape: [Int], salt: Int, palette: [UInt16]
    ) -> [UInt16] {
        let columns = shape.last!
        let count = shape.reduce(1, *)
        return (0..<count).map { index in
            let row = index / columns
            let column = index % columns
            return palette[(row * 3 + column * 5 + salt * 7 + index / 3) % palette.count]
        }
    }

    private static func safetensorsFile(
        _ tensors: [LiteralTensor],
        shapeOverrides: [String: [Int]],
        outOfRangeTensor: String?
    ) throws -> Data {
        var payload = Data()
        var header: [String: [String: Any]] = [:]
        var outOfRangeOffsets: (name: String, start: Int, end: Int)?
        for tensor in tensors {
            let start = payload.count
            for word in tensor.words {
                var value = word.littleEndian
                withUnsafeBytes(of: &value) { payload.append(contentsOf: $0) }
            }
            let shape = shapeOverrides[tensor.name] ?? tensor.shape
            let actualEnd = payload.count
            if tensor.name == outOfRangeTensor {
                outOfRangeOffsets = (tensor.name, start, actualEnd)
            }
            header[tensor.name] = [
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, actualEnd],
            ]
        }
        if let outOfRangeOffsets, var entry = header[outOfRangeOffsets.name] {
            // Shift both bounds equally: shape byte count remains valid while
            // the protected interval starts at the physical payload end.
            let shift = payload.count - outOfRangeOffsets.start
            entry["data_offsets"] = [
                outOfRangeOffsets.start + shift,
                outOfRangeOffsets.end + shift,
            ]
            header[outOfRangeOffsets.name] = entry
        }
        var headerBytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        headerBytes.append(contentsOf: repeatElement(
            UInt8(0x20), count: (8 - headerBytes.count % 8) % 8))
        var headerLength = UInt64(headerBytes.count).littleEndian
        var file = withUnsafeBytes(of: &headerLength) { Data($0) }
        file.append(headerBytes)
        file.append(payload)
        return file
    }
}

enum QwenBF16TextRunnerFixtureError: Error {
    case missingPinnedShardPair
}

extension QwenBF16TextRunnerFixture {
    static func maxDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        return zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
    }
}
