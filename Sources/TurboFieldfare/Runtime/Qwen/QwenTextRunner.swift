import Foundation
import Metal
import os

public enum QwenTextRunnerError: Error, Equatable, Sendable {
    case emptyInput
    case alreadyPrefilled(position: Int)
    case invalidToken(id: Int32)
    case stateArchitectureMismatch
    case invalidState(detail: String)
    case operationInProgress
    case cancelled
    case invalidPosition(expected: Int, actual: Int)
    case logitsBufferTooSmall(expected: Int, actual: Int)
    case gpuExecution(stage: String, detail: String)
    case invalidTransaction
    case execution(detail: String)
}

struct QwenTextLayerTrace: Equatable, Sendable {
    let layer: Int
    let inputNormalized: [Float]
    let postAttentionNormalized: [Float]
    let postResidualHidden: [Float]
    let normalizedQuery: [Float]?
    let normalizedKey: [Float]?
    let rotatedQuery: [Float]?
    let rotatedKey: [Float]?

    init(
        layer: Int,
        inputNormalized: [Float],
        postAttentionNormalized: [Float],
        postResidualHidden: [Float],
        normalizedQuery: [Float]? = nil,
        normalizedKey: [Float]? = nil,
        rotatedQuery: [Float]? = nil,
        rotatedKey: [Float]? = nil
    ) {
        self.layer = layer
        self.inputNormalized = inputNormalized
        self.postAttentionNormalized = postAttentionNormalized
        self.postResidualHidden = postResidualHidden
        self.normalizedQuery = normalizedQuery
        self.normalizedKey = normalizedKey
        self.rotatedQuery = rotatedQuery
        self.rotatedKey = rotatedKey
    }
}

struct QwenPreparedFeatureOverride: Sendable {
    let tokenRange: Range<Int>
    let owner: QwenRetainedFeatureOwner
}

struct QwenPreparedPrefill: Sendable {
    let tokenIDs: [Int32]
    let featureOverrides: [QwenPreparedFeatureOverride]
    let positions: [QwenMRoPEPosition]
    let textRoPEDelta: Int

    init(
        tokenIDs: [Int32],
        featureOverrides: [QwenPreparedFeatureOverride],
        positions: [QwenMRoPEPosition],
        textRoPEDelta: Int
    ) throws {
        guard !tokenIDs.isEmpty, positions.count == tokenIDs.count else {
            throw QwenTextRunnerError.invalidState(detail: "prepared positions do not match tokens")
        }
        var previousUpper = 0
        for override in featureOverrides {
            guard !override.tokenRange.isEmpty,
                  override.tokenRange.lowerBound >= previousUpper,
                  override.tokenRange.upperBound <= tokenIDs.count,
                  override.tokenRange.count == override.owner.rowCount else {
                throw QwenTextRunnerError.invalidState(detail: "prepared feature override is invalid")
            }
            previousUpper = override.tokenRange.upperBound
        }
        self.tokenIDs = tokenIDs
        self.featureOverrides = featureOverrides
        self.positions = positions
        self.textRoPEDelta = textRoPEDelta
    }
}

enum QwenTextLayerState: Equatable, Sendable {
    case linear(convolutionHistory: [Float], recurrentMatrix: [Float])
    case full(key: [Float], value: [Float])
}

struct QwenTextRunnerState: Equatable, Sendable {
    let architectureIdentity: String
    let sequenceLength: Int
    let layers: [QwenTextLayerState]
}

struct QwenTextRunnerOutput: Equatable, Sendable {
    let tokenCount: Int
    /// Actual embedding rows after image overrides and before layer zero.
    let preparedHidden: [Float]
    let finalHidden: [Float]
    /// Raw CausalLM logits. No Gemma softcap is applied by this runner.
    let logits: [Float]
    let layers: [QwenTextLayerTrace]
    let state: QwenTextRunnerState
}

/// Concrete normalized/stateful Qwen decoder. It composes the Phase 9–11
/// reference operations over values decoded from the production pack format.
/// Those component operations keep their documented CPU/GPU boundary: the
/// complete tiny-model oracle path is deterministic CPU execution, while the
/// same Qwen component types retain their separately validated Metal entry
/// points for Q/K norm+RoPE, gating, DeltaNet state, and MoE operations.
final class QwenTextReferenceRunner: @unchecked Sendable {
    private struct LinearWeights {
        let delta: QwenGatedDeltaNetWeights
    }

    private struct AttentionWeights {
        let query: [Float]
        let key: [Float]
        let value: [Float]
        let output: [Float]
        let queryNorm: [Float]
        let keyNorm: [Float]
    }

    private struct MoEWeights {
        let router: [Float]
        let shared: QwenMoEDenseSharedExpert
    }

    private enum MixerWeights {
        case linear(LinearWeights)
        case full(AttentionWeights)
    }

    private struct LayerWeights {
        let inputNorm: [Float]
        let postAttentionNorm: [Float]
        let mixer: MixerWeights
        let moe: MoEWeights
    }

    let model: QwenTextModel
    private let architecture: QwenTextArchitecture
    private let embeddingWeights: [Float]
    private let outputHeadWeights: [Float]
    private let finalNormWeights: [Float]
    private let layerWeights: [LayerWeights]
    private let linearConfiguration: QwenGatedDeltaNetConfiguration
    private let attentionConfiguration: QwenFullAttentionConfiguration
    private let moeConfiguration: QwenMoEConfiguration
    private let lock = NSLock()
    private var committedState: QwenTextRunnerState
    private var routedExpertCache: [String: QwenMoEDenseExpert] = [:]

    var position: Int {
        lock.withLock { committedState.sequenceLength }
    }

    init(model: QwenTextModel) throws {
        self.model = model
        architecture = model.architecture
        do {
            embeddingWeights = try model.embedding.decodedFloat32()
            outputHeadWeights = try model.outputHead.decodedFloat32()
            finalNormWeights = try model.finalNorm.decodedFloat32()
            linearConfiguration = try QwenGatedDeltaNetConfiguration(
                hiddenSize: architecture.hiddenSize,
                keyHeadCount: architecture.linearKeyHeads,
                valueHeadCount: architecture.linearValueHeads,
                keyHeadDimension: architecture.linearKeyDimension,
                valueHeadDimension: architecture.linearValueDimension,
                convolutionWidth: architecture.convolutionWidth)
            attentionConfiguration = try QwenFullAttentionConfiguration(
                queryHeadCount: architecture.queryHeads,
                keyValueHeadCount: architecture.keyValueHeads,
                headDimension: architecture.headDimension,
                rotaryDimension: Int(
                    Double(architecture.headDimension) * architecture.partialRotaryFactor),
                theta: Float(architecture.ropeTheta))
            moeConfiguration = try QwenMoEConfiguration(
                hiddenSize: architecture.hiddenSize,
                expertCount: architecture.experts,
                topK: architecture.expertsPerToken,
                routedIntermediateSize: architecture.routedIntermediateSize,
                sharedIntermediateSize: architecture.sharedIntermediateSize)
            var decodedLayers: [LayerWeights] = []
            decodedLayers.reserveCapacity(architecture.layers)
            for layer in 0..<architecture.layers {
                let inputNorm = try model.inputNorm(layer: layer).decodedFloat32()
                let postNorm = try model.postAttentionNorm(layer: layer).decodedFloat32()
                let mixer: MixerWeights
                switch architecture.layerKinds[layer] {
                case .linearAttention:
                    mixer = .linear(LinearWeights(delta: QwenGatedDeltaNetWeights(
                        qkvProjection: try model.linearTensor(
                            layer: layer, suffix: "in_proj_qkv.weight").decodedFloat32(),
                        zProjection: try model.linearTensor(
                            layer: layer, suffix: "in_proj_z.weight").decodedFloat32(),
                        bProjection: try model.linearTensor(
                            layer: layer, suffix: "in_proj_b.weight").decodedFloat32(),
                        aProjection: try model.linearTensor(
                            layer: layer, suffix: "in_proj_a.weight").decodedFloat32(),
                        convolution: try model.linearTensor(
                            layer: layer, suffix: "conv1d.weight").decodedFloat32(),
                        timeStepBias: try model.linearTensor(
                            layer: layer, suffix: "dt_bias").decodedFloat32(),
                        aLog: try model.linearTensor(
                            layer: layer, suffix: "A_log").decodedFloat32(),
                        normalization: try model.linearTensor(
                            layer: layer, suffix: "norm.weight").decodedFloat32(),
                        outputProjection: try model.linearTensor(
                            layer: layer, suffix: "out_proj.weight").decodedFloat32())))
                case .fullAttention:
                    mixer = .full(AttentionWeights(
                        query: try model.attentionTensor(
                            layer: layer, suffix: "q_proj.weight").decodedFloat32(),
                        key: try model.attentionTensor(
                            layer: layer, suffix: "k_proj.weight").decodedFloat32(),
                        value: try model.attentionTensor(
                            layer: layer, suffix: "v_proj.weight").decodedFloat32(),
                        output: try model.attentionTensor(
                            layer: layer, suffix: "o_proj.weight").decodedFloat32(),
                        queryNorm: try model.attentionTensor(
                            layer: layer, suffix: "q_norm.weight").decodedFloat32(),
                        keyNorm: try model.attentionTensor(
                            layer: layer, suffix: "k_norm.weight").decodedFloat32()))
                }
                let shared = QwenMoEDenseSharedExpert(
                    gate: try model.sharedExpert(layer: layer, role: "gate").decodedFloat32(),
                    up: try model.sharedExpert(layer: layer, role: "up").decodedFloat32(),
                    down: try model.sharedExpert(layer: layer, role: "down").decodedFloat32(),
                    outputGate: try model.sharedExpert(
                        layer: layer, role: "output_gate").decodedFloat32())
                decodedLayers.append(LayerWeights(
                    inputNorm: inputNorm,
                    postAttentionNorm: postNorm,
                    mixer: mixer,
                    moe: MoEWeights(
                        router: try model.router(layer: layer).decodedFloat32(),
                        shared: shared)))
            }
            layerWeights = decodedLayers
        } catch let error as QwenTextRunnerError {
            throw error
        } catch {
            throw QwenTextRunnerError.execution(detail: "model construction: \(error)")
        }
        committedState = Self.zeroState(
            architecture: architecture, identity: model.mappingIdentity)
    }

    func reset() {
        lock.withLock {
            committedState = Self.zeroState(
                architecture: architecture, identity: model.mappingIdentity)
        }
    }

    func prefill(tokens: [Int32]) throws -> QwenTextRunnerOutput {
        try lock.withLock {
            guard committedState.sequenceLength == 0 else {
                throw QwenTextRunnerError.alreadyPrefilled(
                    position: committedState.sequenceLength)
            }
            return try appendLocked(tokens: tokens)
        }
    }

    func decode(token: Int32) throws -> QwenTextRunnerOutput {
        try lock.withLock { try appendLocked(tokens: [token]) }
    }

    func append(tokens: [Int32]) throws -> QwenTextRunnerOutput {
        try lock.withLock { try appendLocked(tokens: tokens) }
    }

    func snapshot() -> QwenTextRunnerState {
        lock.withLock { committedState }
    }

    /// Validate-then-commit restore. A rejected snapshot cannot partially
    /// replace any layer state.
    func restore(_ state: QwenTextRunnerState) throws {
        try lock.withLock {
            try validate(state)
            committedState = state
        }
    }

    private func appendLocked(tokens: [Int32]) throws -> QwenTextRunnerOutput {
        guard !tokens.isEmpty else { throw QwenTextRunnerError.emptyInput }
        for token in tokens where token < 0 || Int(token) >= architecture.vocabularySize {
            throw QwenTextRunnerError.invalidToken(id: token)
        }
        let initial = committedState
        do {
            var hidden: [Float] = []
            hidden.reserveCapacity(tokens.count * architecture.hiddenSize)
            for token in tokens {
                let start = Int(token) * architecture.hiddenSize
                hidden.append(contentsOf: embeddingWeights[start..<(start + architecture.hiddenSize)])
            }
            let preparedHidden = hidden
            var nextLayerStates: [QwenTextLayerState] = []
            var traces: [QwenTextLayerTrace] = []
            nextLayerStates.reserveCapacity(architecture.layers)
            traces.reserveCapacity(architecture.layers)

            for layer in 0..<architecture.layers {
                let weights = layerWeights[layer]
                let residual = hidden
                let inputNormalized = try rmsNorm(
                    hidden, storedWeight: weights.inputNorm,
                    width: architecture.hiddenSize)
                let mixerOutput: [Float]
                let nextMixerState: QwenTextLayerState
                var normalizedQuery: [Float]?
                var normalizedKey: [Float]?
                var rotatedQuery: [Float]?
                var rotatedKey: [Float]?
                switch (weights.mixer, initial.layers[layer]) {
                case let (.linear(linear), .linear(history, recurrent)):
                    let result = try QwenGatedDeltaNet.evaluate(
                        configuration: linearConfiguration,
                        input: inputNormalized,
                        weights: linear.delta,
                        initialState: QwenLinearAttentionLayerState(
                            convolutionHistory: history,
                            recurrentMatrix: recurrent),
                        chunkSize: max(1, tokens.count))
                    mixerOutput = result.output
                    nextMixerState = .linear(
                        convolutionHistory: result.finalState.convolutionHistory,
                        recurrentMatrix: result.finalState.recurrentMatrix)
                case let (.full(attention), .full(cachedKey, cachedValue)):
                    let queryWidth = 2 * architecture.queryHeads * architecture.headDimension
                    let keyValueWidth = architecture.keyValueHeads * architecture.headDimension
                    let q = project(
                        inputNormalized, matrix: attention.query,
                        rows: queryWidth, columns: architecture.hiddenSize)
                    let k = project(
                        inputNormalized, matrix: attention.key,
                        rows: keyValueWidth, columns: architecture.hiddenSize)
                    let v = project(
                        inputNormalized, matrix: attention.value,
                        rows: keyValueWidth, columns: architecture.hiddenSize)
                    let positions = Array(
                        initial.sequenceLength..<(initial.sequenceLength + tokens.count))
                    let result = try QwenFullAttention.evaluate(
                        configuration: attentionConfiguration,
                        queryAndGateProjection: q,
                        keyProjection: k,
                        valueProjection: v,
                        queryNormWeight: attention.queryNorm,
                        keyNormWeight: attention.keyNorm,
                        positions: positions,
                        cachedKeys: cachedKey,
                        cachedValues: cachedValue,
                        outputProjection: attention.output,
                        outputDimension: architecture.hiddenSize)
                    mixerOutput = result.output
                    normalizedQuery = result.normalizedQuery
                    normalizedKey = result.normalizedKey
                    rotatedQuery = result.rotatedQuery
                    rotatedKey = result.rotatedKey
                    nextMixerState = .full(
                        key: cachedKey + result.rotatedKey,
                        value: cachedValue + v)
                default:
                    throw QwenTextRunnerError.stateArchitectureMismatch
                }
                hidden = zip(residual, mixerOutput).map(+)
                let postAttentionNormalized = try rmsNorm(
                    hidden, storedWeight: weights.postAttentionNorm,
                    width: architecture.hiddenSize)
                let routerLogits = project(
                    postAttentionNormalized,
                    matrix: weights.moe.router,
                    rows: architecture.experts,
                    columns: architecture.hiddenSize)
                let route = try QwenMoE.route(
                    logits: routerLogits, configuration: moeConfiguration)
                var experts: [Int: QwenMoEDenseExpert] = [:]
                for expert in Set(route.selectedExpertIDs.flatMap { $0 }) {
                    experts[expert] = try routedExpert(layer: layer, expert: expert)
                }
                let moe = try QwenMoE.evaluate(
                    input: postAttentionNormalized,
                    routerLogits: routerLogits,
                    configuration: moeConfiguration,
                    routedExperts: experts,
                    sharedExpert: weights.moe.shared)
                hidden = zip(hidden, moe.output).map(+)
                nextLayerStates.append(nextMixerState)
                traces.append(QwenTextLayerTrace(
                    layer: layer,
                    inputNormalized: inputNormalized,
                    postAttentionNormalized: postAttentionNormalized,
                    postResidualHidden: hidden,
                    normalizedQuery: normalizedQuery,
                    normalizedKey: normalizedKey,
                    rotatedQuery: rotatedQuery,
                    rotatedKey: rotatedKey))
            }
            let finalHidden = try rmsNorm(
                hidden, storedWeight: finalNormWeights,
                width: architecture.hiddenSize)
            let logits = project(
                finalHidden,
                matrix: outputHeadWeights,
                rows: architecture.vocabularySize,
                columns: architecture.hiddenSize)
            let next = QwenTextRunnerState(
                architectureIdentity: model.mappingIdentity,
                sequenceLength: initial.sequenceLength + tokens.count,
                layers: nextLayerStates)
            try validate(next)
            committedState = next
            return QwenTextRunnerOutput(
                tokenCount: tokens.count,
                preparedHidden: preparedHidden,
                finalHidden: finalHidden,
                logits: logits,
                layers: traces,
                state: next)
        } catch let error as QwenTextRunnerError {
            throw error
        } catch {
            throw QwenTextRunnerError.execution(detail: "\(error)")
        }
    }

    private func routedExpert(layer: Int, expert: Int) throws -> QwenMoEDenseExpert {
        let key = "\(layer):\(expert)"
        if let cached = routedExpertCache[key] { return cached }
        let value = QwenMoEDenseExpert(
            gateUp: try model.routedExpert(
                layer: layer, expert: expert, role: "gate_up").decodedFloat32(),
            down: try model.routedExpert(
                layer: layer, expert: expert, role: "down").decodedFloat32())
        routedExpertCache[key] = value
        return value
    }

    private func validate(_ state: QwenTextRunnerState) throws {
        guard state.architectureIdentity == model.mappingIdentity else {
            throw QwenTextRunnerError.stateArchitectureMismatch
        }
        guard state.sequenceLength >= 0, state.layers.count == architecture.layers else {
            throw QwenTextRunnerError.invalidState(detail: "sequence or layer count")
        }
        let convolutionCount = linearConfiguration.convolutionChannelCount
            * linearConfiguration.convolutionWidth
        let recurrentCount = linearConfiguration.valueHeadCount
            * linearConfiguration.keyHeadDimension
            * linearConfiguration.valueHeadDimension
        let kvWidth = architecture.keyValueHeads * architecture.headDimension
        for layer in 0..<architecture.layers {
            switch (architecture.layerKinds[layer], state.layers[layer]) {
            case let (.linearAttention, .linear(history, recurrent)):
                guard history.count == convolutionCount,
                      recurrent.count == recurrentCount,
                      history.allSatisfy(\.isFinite),
                      recurrent.allSatisfy(\.isFinite) else {
                    throw QwenTextRunnerError.invalidState(detail: "linear layer \(layer)")
                }
            case let (.fullAttention, .full(key, value)):
                let expected = state.sequenceLength * kvWidth
                guard key.count == expected, value.count == expected,
                      key.allSatisfy(\.isFinite), value.allSatisfy(\.isFinite) else {
                    throw QwenTextRunnerError.invalidState(detail: "full layer \(layer)")
                }
            default:
                throw QwenTextRunnerError.stateArchitectureMismatch
            }
        }
    }

    fileprivate static func zeroState(
        architecture: QwenTextArchitecture,
        identity: String
    ) -> QwenTextRunnerState {
        let channels = 2 * architecture.linearKeyHeads * architecture.linearKeyDimension
            + architecture.linearValueHeads * architecture.linearValueDimension
        let convolutionCount = channels * architecture.convolutionWidth
        let recurrentCount = architecture.linearValueHeads
            * architecture.linearKeyDimension * architecture.linearValueDimension
        return QwenTextRunnerState(
            architectureIdentity: identity,
            sequenceLength: 0,
            layers: architecture.layerKinds.map {
                switch $0 {
                case .linearAttention:
                    .linear(
                        convolutionHistory: [Float](repeating: 0, count: convolutionCount),
                        recurrentMatrix: [Float](repeating: 0, count: recurrentCount))
                case .fullAttention:
                    .full(key: [], value: [])
                }
            })
    }
}

struct QwenTextExecutionDiagnostics: Equatable, Sendable {
    var fullAttentionNormRoPESubmissions = 0
    var fullAttentionGateSubmissions = 0
    var linearLayoutSubmissions = 0
    var linearConvolutionSubmissions = 0
    var linearRecurrenceSubmissions = 0
    var linearGatedNormSubmissions = 0
    var mappedExpertSubmissions = 0
    var completedCommandBuffers = 0
    var discardedCommandBuffers = 0
    var convolutionOutputsConsumed = 0
}

struct QwenTextHybridResult: Equatable, Sendable {
    let output: QwenTextRunnerOutput
    let diagnostics: QwenTextExecutionDiagnostics
}

struct QwenTextExecutionHooks: Sendable {
    let beforeLayer: @Sendable (Int) async throws -> Void
    let afterCommandSubmission: @Sendable (String) async -> Void

    init(
        beforeLayer: @escaping @Sendable (Int) async throws -> Void = { _ in },
        afterCommandSubmission: @escaping @Sendable (String) async -> Void = { _ in }
    ) {
        self.beforeLayer = beforeLayer
        self.afterCommandSubmission = afterCommandSubmission
    }

    static let none = Self()
}

struct QwenTextPublicationHooks: Sendable {
    let beforeLockedWrite: @Sendable () async -> Void
    let afterLockedWriteBeforeFinish: @Sendable () async -> Void

    init(
        beforeLockedWrite: @escaping @Sendable () async -> Void = {},
        afterLockedWriteBeforeFinish: @escaping @Sendable () async -> Void = {}
    ) {
        self.beforeLockedWrite = beforeLockedWrite
        self.afterLockedWriteBeforeFinish = afterLockedWriteBeforeFinish
    }

    static let none = Self()
}

actor QwenTextRunner {
    private struct LinearWeights {
        let delta: QwenGatedDeltaNetWeights
        let convolutionBuffer: MTLBuffer
        let normBuffer: MTLBuffer
    }

    private struct AttentionWeights {
        let query: [Float]
        let key: [Float]
        let value: [Float]
        let output: [Float]
        let queryNorm: [Float]
        let keyNorm: [Float]
        let queryNormBuffer: MTLBuffer
        let keyNormBuffer: MTLBuffer
    }

    private enum MixerWeights {
        case linear(LinearWeights)
        case full(AttentionWeights)
    }

    private struct LayerWeights {
        let inputNorm: [Float]
        let postAttentionNorm: [Float]
        let mixer: MixerWeights
        let router: [Float]
        let sharedBindings: QwenMoESharedAffineBindings
    }

    private struct PreparedTransaction {
        let identifier: UInt64
        let priorLinearState: QwenLinearAttentionSnapshot
        let nextState: QwenTextRunnerState
        let result: QwenTextHybridResult
    }

    struct PreparedProduction: Sendable {
        let identifier: UInt64
        let result: QwenTextHybridResult
    }

    let model: QwenTextModel
    private let architecture: QwenTextArchitecture
    private let context: MetalContext
    private let embeddingWeights: [Float]
    private let outputHeadWeights: [Float]
    private let finalNormWeights: [Float]
    private let layerWeights: [LayerWeights]
    private let linearConfiguration: QwenGatedDeltaNetConfiguration
    private let attentionConfiguration: QwenFullAttentionConfiguration
    private let moeConfiguration: QwenMoEConfiguration
    private let linearRuntime: QwenGatedDeltaNet
    private let attentionRuntime: QwenFullAttention
    private let moeRuntime: QwenMoE
    private let linearState: QwenLinearAttentionState
    private let expertCoordinators: [QwenExpertMappingCoordinator]
    private let moeScratch: [QwenMoEScratch]
    private let executionHooks: QwenTextExecutionHooks
    private var committedState: QwenTextRunnerState
    private var inFlight = false
    private var preparedTransaction: PreparedTransaction?
    private var nextTransactionIdentifier: UInt64 = 1
    private var adapterGeneration: UInt64?

    var position: Int { committedState.sequenceLength }

    /// Bytes of the routed-expert cache buffers owned by every layer. Each
    /// coordinator owns one distinct streamer, so every buffer is counted once.
    var expertCacheAllocatedBytes: UInt64 {
        expertCoordinators.reduce(UInt64(0)) {
            $0 + $1.allocatedCacheBytes
        }
    }

    init(
        model: QwenTextModel,
        context: MetalContext,
        expertSlotCount: Int = 8,
        executionHooks: QwenTextExecutionHooks = .none
    ) throws {
        guard expertSlotCount >= model.architecture.expertsPerToken else {
            throw QwenTextRunnerError.execution(detail: "expert slot count is below top-k")
        }
        self.model = model
        self.context = context
        self.executionHooks = executionHooks
        architecture = model.architecture
        do {
            embeddingWeights = try model.embedding.decodedFloat32()
            outputHeadWeights = try model.outputHead.decodedFloat32()
            finalNormWeights = try model.finalNorm.decodedFloat32()
            linearConfiguration = try QwenGatedDeltaNetConfiguration(
                hiddenSize: architecture.hiddenSize,
                keyHeadCount: architecture.linearKeyHeads,
                valueHeadCount: architecture.linearValueHeads,
                keyHeadDimension: architecture.linearKeyDimension,
                valueHeadDimension: architecture.linearValueDimension,
                convolutionWidth: architecture.convolutionWidth)
            attentionConfiguration = try QwenFullAttentionConfiguration(
                queryHeadCount: architecture.queryHeads,
                keyValueHeadCount: architecture.keyValueHeads,
                headDimension: architecture.headDimension,
                rotaryDimension: Int(Double(architecture.headDimension) * architecture.partialRotaryFactor),
                theta: Float(architecture.ropeTheta))
            moeConfiguration = try QwenMoEConfiguration(
                hiddenSize: architecture.hiddenSize,
                expertCount: architecture.experts,
                topK: architecture.expertsPerToken,
                routedIntermediateSize: architecture.routedIntermediateSize,
                sharedIntermediateSize: architecture.sharedIntermediateSize)
            linearRuntime = try QwenGatedDeltaNet(context: context, configuration: linearConfiguration)
            attentionRuntime = try QwenFullAttention(context: context, configuration: attentionConfiguration)
            moeRuntime = try QwenMoE(context: context, configuration: moeConfiguration)
            let geometry = try QwenLinearAttentionGeometry(
                convolutionWidth: architecture.convolutionWidth,
                convolutionChannelCount: linearConfiguration.convolutionChannelCount,
                valueHeadCount: architecture.linearValueHeads,
                keyHeadDimension: architecture.linearKeyDimension,
                valueHeadDimension: architecture.linearValueDimension)
            let linearMask = architecture.layerKinds.map { $0 == .linearAttention ? UInt8(1) : UInt8(0) }
            linearState = try QwenLinearAttentionState(
                device: context.device,
                linearAttentionLayerMask: linearMask,
                geometry: geometry,
                expectedLinearLayerCount: linearMask.filter { $0 == 1 }.count)
            var decoded: [LayerWeights] = []
            var coordinators: [QwenExpertMappingCoordinator] = []
            var scratches: [QwenMoEScratch] = []
            for layer in 0..<architecture.layers {
                let mixer: MixerWeights
                switch architecture.layerKinds[layer] {
                case .linearAttention:
                    let delta = QwenGatedDeltaNetWeights(
                        qkvProjection: try model.linearTensor(layer: layer, suffix: "in_proj_qkv.weight").decodedFloat32(),
                        zProjection: try model.linearTensor(layer: layer, suffix: "in_proj_z.weight").decodedFloat32(),
                        bProjection: try model.linearTensor(layer: layer, suffix: "in_proj_b.weight").decodedFloat32(),
                        aProjection: try model.linearTensor(layer: layer, suffix: "in_proj_a.weight").decodedFloat32(),
                        convolution: try model.linearTensor(layer: layer, suffix: "conv1d.weight").decodedFloat32(),
                        timeStepBias: try model.linearTensor(layer: layer, suffix: "dt_bias").decodedFloat32(),
                        aLog: try model.linearTensor(layer: layer, suffix: "A_log").decodedFloat32(),
                        normalization: try model.linearTensor(layer: layer, suffix: "norm.weight").decodedFloat32(),
                        outputProjection: try model.linearTensor(layer: layer, suffix: "out_proj.weight").decodedFloat32())
                    mixer = .linear(LinearWeights(
                        delta: delta,
                        convolutionBuffer: try Self.floatBuffer(delta.convolution, device: context.device, label: "qwen.linear.conv.\(layer)"),
                        normBuffer: try Self.floatBuffer(delta.normalization, device: context.device, label: "qwen.linear.norm.\(layer)")))
                case .fullAttention:
                    let queryNorm = try model.attentionTensor(
                        layer: layer, suffix: "q_norm.weight").decodedFloat32()
                    let keyNorm = try model.attentionTensor(
                        layer: layer, suffix: "k_norm.weight").decodedFloat32()
                    mixer = .full(AttentionWeights(
                        query: try model.attentionTensor(layer: layer, suffix: "q_proj.weight").decodedFloat32(),
                        key: try model.attentionTensor(layer: layer, suffix: "k_proj.weight").decodedFloat32(),
                        value: try model.attentionTensor(layer: layer, suffix: "v_proj.weight").decodedFloat32(),
                        output: try model.attentionTensor(layer: layer, suffix: "o_proj.weight").decodedFloat32(),
                        queryNorm: queryNorm,
                        keyNorm: keyNorm,
                        queryNormBuffer: try Self.floatBuffer(
                            queryNorm, device: context.device,
                            label: "qwen.attention.qnorm.\(layer)"),
                        keyNormBuffer: try Self.floatBuffer(
                            keyNorm, device: context.device,
                            label: "qwen.attention.knorm.\(layer)")))
                }
                let shared = try QwenMoESharedAffineBindings(
                    gate: model.sharedExpert(layer: layer, role: "gate").makeAffineBinding(device: context.device),
                    up: model.sharedExpert(layer: layer, role: "up").makeAffineBinding(device: context.device),
                    down: model.sharedExpert(layer: layer, role: "down").makeAffineBinding(device: context.device),
                    outputGate: model.sharedExpert(layer: layer, role: "output_gate").makeAffineBinding(device: context.device))
                decoded.append(LayerWeights(
                    inputNorm: try model.inputNorm(layer: layer).decodedFloat32(),
                    postAttentionNorm: try model.postAttentionNorm(layer: layer).decodedFloat32(),
                    mixer: mixer,
                    router: try model.router(layer: layer).decodedFloat32(),
                    sharedBindings: shared))
                coordinators.append(try model.makeExpertCoordinator(
                    layer: layer, device: context.device, slotCount: expertSlotCount))
                scratches.append(try moeRuntime.makeScratch())
            }
            layerWeights = decoded
            expertCoordinators = coordinators
            moeScratch = scratches
        } catch let error as QwenTextRunnerError {
            throw error
        } catch {
            throw QwenTextRunnerError.execution(detail: "hybrid construction: \(error)")
        }
        committedState = QwenTextReferenceRunner.zeroState(
            architecture: architecture, identity: model.mappingIdentity)
    }

    /// Waits for the linear-state owner, not merely Metal command-buffer
    /// completion. Callers use this before restoring a conversation boundary.
    func waitUntilIdle() async {
        await linearState.waitUntilIdle()
    }

    func reset() async throws {
        await waitUntilIdle()
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        try linearState.reset()
        committedState = QwenTextReferenceRunner.zeroState(
            architecture: architecture, identity: model.mappingIdentity)
        adapterGeneration = nil
    }

    func snapshot() throws -> QwenTextRunnerState {
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        return committedState
    }

    /// Restores a nonempty conversation boundary without changing the
    /// producer generation. Clearing `adapterGeneration` here would make the
    /// next producer call require position zero and reject valid continuation.
    func restore(_ state: QwenTextRunnerState) async throws {
        await waitUntilIdle()
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        try validate(state)
        try restoreLinearState(from: state)
        committedState = state
    }

    func prefill(tokens: [Int32]) async throws -> QwenTextHybridResult {
        guard committedState.sequenceLength == 0 else {
            throw QwenTextRunnerError.alreadyPrefilled(position: committedState.sequenceLength)
        }
        return try await runAndPublish(tokens: tokens)
    }

    func prefill(prepared input: QwenPreparedPrefill) async throws -> QwenTextHybridResult {
        guard committedState.sequenceLength == 0 else {
            throw QwenTextRunnerError.alreadyPrefilled(position: committedState.sequenceLength)
        }
        return try await runAndPublish(prepared: input)
    }

    func decode(token: Int32) async throws -> QwenTextHybridResult {
        try await runAndPublish(tokens: [token])
    }

    func append(tokens: [Int32]) async throws -> QwenTextHybridResult {
        try await runAndPublish(tokens: tokens)
    }

    func append(prepared input: QwenPreparedPrefill) async throws -> QwenTextHybridResult {
        try await runAndPublish(prepared: input)
    }

    private func runAndPublish(tokens: [Int32]) async throws -> QwenTextHybridResult {
        let prepared = try await prepare(tokens: tokens)
        commitPrepared(
            identifier: prepared.identifier,
            publishedState: prepared.result.output.state)
        return prepared.result
    }

    private func runAndPublish(prepared input: QwenPreparedPrefill) async throws -> QwenTextHybridResult {
        let transaction = try await prepare(
            tokens: input.tokenIDs,
            featureOverrides: input.featureOverrides,
            mropePositions: input.positions,
            declaredTextRoPEDelta: input.textRoPEDelta)
        commitPrepared(
            identifier: transaction.identifier,
            publishedState: transaction.result.output.state)
        return transaction.result
    }

    func prepareProduction(
        token: Int32,
        cachePosition: Int,
        ropePosition: QwenMRoPEPosition?,
        generation: UInt64,
        isCurrent: @escaping @Sendable () -> Bool
    ) async throws -> PreparedProduction {
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        if adapterGeneration != generation {
            guard cachePosition == 0 else {
                throw QwenTextRunnerError.invalidPosition(expected: 0, actual: cachePosition)
            }
            try linearState.reset()
            committedState = QwenTextReferenceRunner.zeroState(
                architecture: architecture, identity: model.mappingIdentity)
            adapterGeneration = generation
        }
        guard cachePosition == committedState.sequenceLength else {
            throw QwenTextRunnerError.invalidPosition(
                expected: committedState.sequenceLength, actual: cachePosition)
        }
        let prepared = try await prepare(
            tokens: [token],
            mropePositions: ropePosition.map { [$0] },
            declaredTextRoPEDelta: ropePosition.map {
                Int($0.temporal) - cachePosition
            },
            isCurrent: isCurrent)
        return PreparedProduction(identifier: prepared.identifier, result: prepared.result)
    }

    func prepareMultimodalProduction(
        _ input: QwenPreparedPrefill,
        generation: UInt64,
        isCurrent: @escaping @Sendable () -> Bool
    ) async throws -> PreparedProduction {
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        if adapterGeneration != generation {
            guard committedState.sequenceLength == 0 else {
                throw QwenTextRunnerError.invalidPosition(
                    expected: 0, actual: committedState.sequenceLength)
            }
            try linearState.reset()
            committedState = QwenTextReferenceRunner.zeroState(
                architecture: architecture, identity: model.mappingIdentity)
            adapterGeneration = generation
        }
        let prepared = try await prepare(
            tokens: input.tokenIDs,
            featureOverrides: input.featureOverrides,
            mropePositions: input.positions,
            declaredTextRoPEDelta: input.textRoPEDelta,
            isCurrent: isCurrent)
        return PreparedProduction(
            identifier: prepared.identifier, result: prepared.result)
    }

    /// The locked caller-buffer write has already made publication
    /// irrevocable. There is exactly one admitted transaction, so even an
    /// internal receipt mismatch commits that transaction rather than leaving
    /// visible logits paired with rolled-back state.
    func commitPrepared(identifier _: UInt64, publishedState: QwenTextRunnerState) {
        // No ordinary caller can manufacture a receipt. The fallback preserves
        // state/logit pairing even if an internal bookkeeping defect removed
        // the sole admitted transaction after the caller-buffer write.
        committedState = preparedTransaction?.nextState ?? publishedState
        preparedTransaction = nil
        inFlight = false
    }

    /// Discard remains recoverable. Metal completion and state-owner completion
    /// handlers do not have a documented ordering, so restore begins only after
    /// the state owner has cleared every pending update.
    func discardPrepared(identifier: UInt64) async throws {
        guard let transaction = preparedTransaction,
              transaction.identifier == identifier else {
            throw QwenTextRunnerError.invalidTransaction
        }
        await linearState.waitUntilIdle()
        do {
            try linearState.restore(transaction.priorLinearState)
        } catch {
            // Keep admission closed: the caller must not reuse state whose
            // coherent rollback could not be established.
            throw QwenTextRunnerError.execution(
                detail: "discard rollback failed: \(error)")
        }
        preparedTransaction = nil
        inFlight = false
    }

    private func prepare(
        tokens: [Int32],
        featureOverrides: [QwenPreparedFeatureOverride] = [],
        mropePositions: [QwenMRoPEPosition]? = nil,
        declaredTextRoPEDelta: Int? = nil,
        isCurrent: @escaping @Sendable () -> Bool = { true }
    ) async throws -> PreparedTransaction {
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        guard !tokens.isEmpty else { throw QwenTextRunnerError.emptyInput }
        for token in tokens where token < 0 || Int(token) >= architecture.vocabularySize {
            throw QwenTextRunnerError.invalidToken(id: token)
        }
        if let mropePositions {
            guard mropePositions.count == tokens.count,
                  let declaredTextRoPEDelta else {
                throw QwenTextRunnerError.invalidState(
                    detail: "M-RoPE positions require one declared text delta")
            }
            if featureOverrides.isEmpty {
                for (offset, position) in mropePositions.enumerated() {
                    let expected = committedState.sequenceLength
                        + offset + declaredTextRoPEDelta
                    guard position.temporal == expected,
                          position.height == expected,
                          position.width == expected else {
                        throw QwenTextRunnerError.invalidState(
                            detail: "continuation M-RoPE coordinate/delta mismatch")
                    }
                }
            } else {
                guard let maximum = mropePositions.flatMap(\.values).max(),
                      Int(maximum) + 1
                        - (committedState.sequenceLength + tokens.count)
                        == declaredTextRoPEDelta else {
                    throw QwenTextRunnerError.invalidState(
                        detail: "multimodal M-RoPE plan/delta mismatch")
                }
            }
        } else if declaredTextRoPEDelta != nil {
            throw QwenTextRunnerError.invalidState(
                detail: "text delta requires M-RoPE positions")
        }
        var previousUpper = 0
        for override in featureOverrides {
            guard override.tokenRange.lowerBound >= previousUpper,
                  override.tokenRange.upperBound <= tokens.count,
                  override.tokenRange.count == override.owner.rowCount,
                  override.owner.hiddenSize == architecture.hiddenSize else {
                throw QwenTextRunnerError.invalidState(detail: "feature override shape mismatch")
            }
            previousUpper = override.tokenRange.upperBound
        }
        inFlight = true
        let initial = committedState
        let initialLinear: QwenLinearAttentionSnapshot
        do {
            initialLinear = try linearState.clone()
        } catch {
            inFlight = false
            throw QwenTextRunnerError.execution(detail: "linear snapshot: \(error)")
        }
        do {
            var diagnostics = QwenTextExecutionDiagnostics()
            var hidden: [Float] = []
            for token in tokens {
                let start = Int(token) * architecture.hiddenSize
                hidden.append(contentsOf: embeddingWeights[start..<(start + architecture.hiddenSize)])
            }
            for override in featureOverrides {
                let values = override.owner.features()
                let lower = override.tokenRange.lowerBound * architecture.hiddenSize
                let upper = override.tokenRange.upperBound * architecture.hiddenSize
                hidden.replaceSubrange(lower..<upper, with: values)
            }
            let preparedHidden = hidden
            var nextLayers: [QwenTextLayerState] = []
            var traces: [QwenTextLayerTrace] = []
            for layer in 0..<architecture.layers {
                try await executionHooks.beforeLayer(layer)
                try Task.checkCancellation()
                guard isCurrent() else { throw QwenTextRunnerError.cancelled }
                let weights = layerWeights[layer]
                let residual = hidden
                let normalized = try rmsNorm(hidden, storedWeight: weights.inputNorm, width: architecture.hiddenSize)
                let mixer: [Float]
                let mixerState: QwenTextLayerState
                var normalizedQuery: [Float]?
                var normalizedKey: [Float]?
                var rotatedQuery: [Float]?
                var rotatedKey: [Float]?
                switch (weights.mixer, initial.layers[layer]) {
                case let (.linear(linear), .linear):
                    let result = try await runLinear(
                        layer: layer, input: normalized, weights: linear,
                        tokenCount: tokens.count, diagnostics: &diagnostics)
                    mixer = result.output
                    let state = try linearState.layerState(layer)
                    mixerState = .linear(
                        convolutionHistory: state.convolutionHistory,
                        recurrentMatrix: state.recurrentMatrix)
                case let (.full(attention), .full(cachedKey, cachedValue)):
                    let result = try await runAttention(
                        input: normalized, weights: attention,
                        tokenCount: tokens.count, startPosition: initial.sequenceLength,
                        mropePositions: mropePositions,
                        cachedKey: cachedKey, cachedValue: cachedValue,
                        diagnostics: &diagnostics)
                    mixer = result.output
                    normalizedQuery = result.normalizedQuery
                    normalizedKey = result.normalizedKey
                    rotatedQuery = result.rotatedQuery
                    rotatedKey = result.rotatedKey
                    mixerState = .full(key: cachedKey + result.key, value: cachedValue + result.value)
                default:
                    throw QwenTextRunnerError.stateArchitectureMismatch
                }
                hidden = zip(residual, mixer).map(+)
                let postNorm = try rmsNorm(
                    hidden, storedWeight: weights.postAttentionNorm,
                    width: architecture.hiddenSize)
                let moeOutput = try await runMoE(
                    layer: layer, input: postNorm, weights: weights,
                    tokenCount: tokens.count, diagnostics: &diagnostics)
                hidden = zip(hidden, moeOutput).map(+)
                nextLayers.append(mixerState)
                traces.append(QwenTextLayerTrace(
                    layer: layer, inputNormalized: normalized,
                    postAttentionNormalized: postNorm, postResidualHidden: hidden,
                    normalizedQuery: normalizedQuery,
                    normalizedKey: normalizedKey,
                    rotatedQuery: rotatedQuery,
                    rotatedKey: rotatedKey))
            }
            try Task.checkCancellation()
            guard isCurrent() else { throw QwenTextRunnerError.cancelled }
            let finalHidden = try rmsNorm(
                hidden, storedWeight: finalNormWeights, width: architecture.hiddenSize)
            let logits = project(
                finalHidden, matrix: outputHeadWeights,
                rows: architecture.vocabularySize, columns: architecture.hiddenSize)
            let next = QwenTextRunnerState(
                architectureIdentity: model.mappingIdentity,
                sequenceLength: initial.sequenceLength + tokens.count,
                layers: nextLayers)
            try validate(next)
            let output = QwenTextRunnerOutput(
                tokenCount: tokens.count, preparedHidden: preparedHidden,
                finalHidden: finalHidden,
                logits: logits, layers: traces, state: next)
            let identifier = nextTransactionIdentifier
            nextTransactionIdentifier &+= 1
            let transaction = PreparedTransaction(
                identifier: identifier, priorLinearState: initialLinear,
                nextState: next,
                result: QwenTextHybridResult(output: output, diagnostics: diagnostics))
            preparedTransaction = transaction
            return transaction
        } catch {
            let operationError: Error = error is CancellationError
                ? QwenTextRunnerError.cancelled : error
            await linearState.waitUntilIdle()
            do {
                try linearState.restore(initialLinear)
            } catch let rollbackError {
                // Keep admission closed when coherent rollback cannot be
                // established; surface the failure instead of trapping or
                // allowing unsafe reuse.
                throw QwenTextRunnerError.execution(
                    detail: "operation \(operationError); rollback \(rollbackError)")
            }
            preparedTransaction = nil
            inFlight = false
            throw operationError
        }
    }

    private func runLinear(
        layer: Int,
        input: [Float],
        weights: LinearWeights,
        tokenCount: Int,
        diagnostics: inout QwenTextExecutionDiagnostics
    ) async throws -> (output: [Float], convolution: [Float]) {
        let delta = weights.delta
        let channels = linearConfiguration.convolutionChannelCount
        let qkv = project(input, matrix: delta.qkvProjection, rows: channels, columns: architecture.hiddenSize)
        let rawGate = project(input, matrix: delta.zProjection, rows: linearConfiguration.valueDimension, columns: architecture.hiddenSize)
        let rawBeta = project(input, matrix: delta.bProjection, rows: architecture.linearValueHeads, columns: architecture.hiddenSize)
        let rawA = project(input, matrix: delta.aProjection, rows: architecture.linearValueHeads, columns: architecture.hiddenSize)
        let qkvBuffer = try Self.floatBuffer(qkv, device: context.device, label: "qwen.linear.qkv")
        let layoutBuffer = try Self.emptyFloatBuffer(count: qkv.count, device: context.device, label: "qwen.linear.layout")
        let convolutionBuffer = try Self.emptyFloatBuffer(count: qkv.count, device: context.device, label: "qwen.linear.convolution")
        let firstUpdate = try linearState.reserveUpdate(layer: layer)
        guard let firstCommand = context.queue.makeCommandBuffer() else {
            try? linearState.abort(firstUpdate)
            throw QwenTextRunnerError.gpuExecution(stage: "linear convolution", detail: "command buffer unavailable")
        }
        do {
            try linearRuntime.encodeLayout(
                commandBuffer: firstCommand, input: qkvBuffer, output: layoutBuffer,
                tokenCount: tokenCount, channelCount: channels, tokenToChannel: true)
            try linearRuntime.encodeCausalConvolution(
                commandBuffer: firstCommand, channelMajorInput: layoutBuffer,
                weights: weights.convolutionBuffer, update: firstUpdate,
                channelMajorOutput: convolutionBuffer, tokenCount: tokenCount)
            diagnostics.linearLayoutSubmissions += 1
            diagnostics.linearConvolutionSubmissions += 1
            try linearState.submit(firstUpdate, on: firstCommand)
        } catch {
            try? linearState.abort(firstUpdate)
            throw error
        }
        await executionHooks.afterCommandSubmission("linearConvolution")
        try await awaitCompletion(
            firstCommand, stage: "linear convolution", diagnostics: &diagnostics,
            onCancel: { try? self.linearState.cancel(firstUpdate) })
        await linearState.waitUntilIdle()
        let channelMajor = Self.readFloats(convolutionBuffer, count: qkv.count)
        var convolved = [Float](repeating: 0, count: qkv.count)
        for channel in 0..<channels {
            for token in 0..<tokenCount {
                convolved[token * channels + channel] = channelMajor[channel * tokenCount + token]
            }
        }
        diagnostics.convolutionOutputsConsumed += 1
        let keyWidth = linearConfiguration.keyDimension
        let valueWidth = linearConfiguration.valueDimension
        var compactQ: [Float] = []; var compactK: [Float] = []; var value: [Float] = []
        for token in 0..<tokenCount {
            let base = token * channels
            compactQ.append(contentsOf: convolved[base..<(base + keyWidth)])
            compactK.append(contentsOf: convolved[(base + keyWidth)..<(base + 2 * keyWidth)])
            value.append(contentsOf: convolved[(base + 2 * keyWidth)..<(base + 2 * keyWidth + valueWidth)])
        }
        let query = expandKeyHeads(compactQ, tokenCount: tokenCount)
        let key = expandKeyHeads(compactK, tokenCount: tokenCount)
        let beta = rawBeta.map { 1 / (1 + Float(Foundation.exp(Double(-$0)))) }
        var logDecay = [Float](repeating: 0, count: rawA.count)
        for token in 0..<tokenCount {
            for head in 0..<architecture.linearValueHeads {
                let index = token * architecture.linearValueHeads + head
                let x = rawA[index] + delta.timeStepBias[head]
                let softplus = x > 20 ? x : Float(Foundation.log1p(Foundation.exp(Double(x))))
                logDecay[index] = -Float(Foundation.exp(Double(delta.aLog[head]))) * softplus
            }
        }
        let queryBuffer = try Self.floatBuffer(query, device: context.device, label: "qwen.linear.query")
        let keyBuffer = try Self.floatBuffer(key, device: context.device, label: "qwen.linear.key")
        let valueBuffer = try Self.floatBuffer(value, device: context.device, label: "qwen.linear.value")
        let decayBuffer = try Self.floatBuffer(logDecay, device: context.device, label: "qwen.linear.decay")
        let betaBuffer = try Self.floatBuffer(beta, device: context.device, label: "qwen.linear.beta")
        let gateBuffer = try Self.floatBuffer(rawGate, device: context.device, label: "qwen.linear.gate")
        let recurrenceBuffer = try Self.emptyFloatBuffer(count: value.count, device: context.device, label: "qwen.linear.recurrence")
        let gatedBuffer = try Self.emptyFloatBuffer(count: value.count, device: context.device, label: "qwen.linear.gated")
        let secondUpdate = try linearState.reserveUpdate(layer: layer)
        guard let secondCommand = context.queue.makeCommandBuffer() else {
            try? linearState.abort(secondUpdate)
            throw QwenTextRunnerError.gpuExecution(stage: "linear recurrence", detail: "command buffer unavailable")
        }
        do {
            try linearRuntime.encodeRecurrence(
                commandBuffer: secondCommand, query: queryBuffer, key: keyBuffer,
                value: valueBuffer, logDecay: decayBuffer, beta: betaBuffer,
                update: secondUpdate, output: recurrenceBuffer, tokenCount: tokenCount)
            try linearRuntime.encodeGatedRMSNorm(
                commandBuffer: secondCommand, input: recurrenceBuffer, gate: gateBuffer,
                weights: weights.normBuffer, output: gatedBuffer, tokenCount: tokenCount)
            diagnostics.linearRecurrenceSubmissions += 1
            diagnostics.linearGatedNormSubmissions += 1
            try linearState.submit(secondUpdate, on: secondCommand)
        } catch {
            try? linearState.abort(secondUpdate)
            throw error
        }
        await executionHooks.afterCommandSubmission("linearRecurrence")
        try await awaitCompletion(
            secondCommand, stage: "linear recurrence", diagnostics: &diagnostics,
            onCancel: { try? self.linearState.cancel(secondUpdate) })
        await linearState.waitUntilIdle()
        let gated = Self.readFloats(gatedBuffer, count: value.count)
        return (project(
            gated, matrix: delta.outputProjection,
            rows: architecture.hiddenSize, columns: valueWidth), convolved)
    }

    private func runAttention(
        input: [Float], weights: AttentionWeights, tokenCount: Int,
        startPosition: Int, mropePositions: [QwenMRoPEPosition]?,
        cachedKey: [Float], cachedValue: [Float],
        diagnostics: inout QwenTextExecutionDiagnostics
    ) async throws -> (
        output: [Float], key: [Float], value: [Float],
        normalizedQuery: [Float], normalizedKey: [Float],
        rotatedQuery: [Float], rotatedKey: [Float]
    ) {
        let queryWidth = architecture.queryHeads * architecture.headDimension
        let keyValueWidth = architecture.keyValueHeads * architecture.headDimension
        let projected = project(input, matrix: weights.query, rows: 2 * queryWidth, columns: architecture.hiddenSize)
        var query: [Float] = []; var rawGate: [Float] = []
        for token in 0..<tokenCount {
            let base = token * 2 * queryWidth
            for head in 0..<architecture.queryHeads {
                let headBase = base + head * architecture.headDimension * 2
                query.append(contentsOf: projected[headBase..<(headBase + architecture.headDimension)])
                rawGate.append(contentsOf: projected[(headBase + architecture.headDimension)..<(headBase + 2 * architecture.headDimension)])
            }
        }
        let key = project(input, matrix: weights.key, rows: keyValueWidth, columns: architecture.hiddenSize)
        let value = project(input, matrix: weights.value, rows: keyValueWidth, columns: architecture.hiddenSize)
        let queryInput = try Self.floatBuffer(query, device: context.device, label: "qwen.attention.query")
        let keyInput = try Self.floatBuffer(key, device: context.device, label: "qwen.attention.key")
        let queryOutput = try Self.emptyFloatBuffer(count: query.count, device: context.device, label: "qwen.attention.rotatedQuery")
        let keyOutput = try Self.emptyFloatBuffer(count: key.count, device: context.device, label: "qwen.attention.rotatedKey")
        guard let normCommand = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.gpuExecution(stage: "attention norm/RoPE", detail: "command buffer unavailable")
        }
        if let mropePositions {
            let flatPositions = mropePositions.flatMap { [$0.temporal, $0.height, $0.width] }
            let positionBuffer = try Self.int32Buffer(
                flatPositions, device: context.device, label: "qwen.attention.mropePositions")
            try attentionRuntime.encodeNormAndPartialMRoPE(
                commandBuffer: normCommand, input: queryInput,
                weight: weights.queryNormBuffer, positions: positionBuffer,
                output: queryOutput, tokenCount: tokenCount,
                headCount: architecture.queryHeads,
                sections: architecture.mropeSections)
            try attentionRuntime.encodeNormAndPartialMRoPE(
                commandBuffer: normCommand, input: keyInput,
                weight: weights.keyNormBuffer, positions: positionBuffer,
                output: keyOutput, tokenCount: tokenCount,
                headCount: architecture.keyValueHeads,
                sections: architecture.mropeSections)
        } else {
            try attentionRuntime.encodeNormAndPartialRoPE(
                commandBuffer: normCommand, input: queryInput, weight: weights.queryNormBuffer,
                output: queryOutput, tokenCount: tokenCount,
                headCount: architecture.queryHeads, startPosition: startPosition)
            try attentionRuntime.encodeNormAndPartialRoPE(
                commandBuffer: normCommand, input: keyInput, weight: weights.keyNormBuffer,
                output: keyOutput, tokenCount: tokenCount,
                headCount: architecture.keyValueHeads, startPosition: startPosition)
        }
        diagnostics.fullAttentionNormRoPESubmissions += 2
        normCommand.commit()
        await executionHooks.afterCommandSubmission("attentionNormRoPE")
        try await awaitCompletion(normCommand, stage: "attention norm/RoPE", diagnostics: &diagnostics)
        let rotatedQuery = Self.readFloats(queryOutput, count: query.count)
        let rotatedKey = Self.readFloats(keyOutput, count: key.count)
        let preGate = attentionValues(
            rotatedQuery: rotatedQuery, allKeys: cachedKey + rotatedKey,
            allValues: cachedValue + value, tokenCount: tokenCount,
            cachedTokenCount: cachedKey.count / keyValueWidth)
        let attentionBuffer = try Self.floatBuffer(preGate, device: context.device, label: "qwen.attention.values")
        let gateBuffer = try Self.floatBuffer(rawGate, device: context.device, label: "qwen.attention.gate")
        let gatedBuffer = try Self.emptyFloatBuffer(count: preGate.count, device: context.device, label: "qwen.attention.gated")
        guard let gateCommand = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.gpuExecution(stage: "attention gate", detail: "command buffer unavailable")
        }
        try attentionRuntime.encodeOutputGate(
            commandBuffer: gateCommand, attention: attentionBuffer,
            rawGate: gateBuffer, output: gatedBuffer, elementCount: preGate.count)
        diagnostics.fullAttentionGateSubmissions += 1
        gateCommand.commit()
        await executionHooks.afterCommandSubmission("attentionGate")
        try await awaitCompletion(gateCommand, stage: "attention gate", diagnostics: &diagnostics)
        let gated = Self.readFloats(gatedBuffer, count: preGate.count)
        return (
            project(gated, matrix: weights.output,
                    rows: architecture.hiddenSize, columns: queryWidth),
            rotatedKey, value,
            Self.normalizeAttentionHeads(
                query, storedWeight: weights.queryNorm,
                dimension: architecture.headDimension,
                epsilon: attentionConfiguration.epsilon),
            Self.normalizeAttentionHeads(
                key, storedWeight: weights.keyNorm,
                dimension: architecture.headDimension,
                epsilon: attentionConfiguration.epsilon),
            rotatedQuery, rotatedKey)
    }

    private func runMoE(
        layer: Int, input: [Float], weights: LayerWeights, tokenCount: Int,
        diagnostics: inout QwenTextExecutionDiagnostics
    ) async throws -> [Float] {
        let logits = project(input, matrix: weights.router, rows: architecture.experts, columns: architecture.hiddenSize)
        let route = try QwenMoE.route(logits: logits, configuration: moeConfiguration)
        var output = [Float](repeating: 0, count: input.count)
        for token in 0..<tokenCount {
            try Task.checkCancellation()
            let ids = route.selectedExpertIDs[token]
            let lease = try await expertCoordinators[layer].map(expertIDs: ids)
            let hidden = Array(input[(token * architecture.hiddenSize)..<((token + 1) * architecture.hiddenSize)])
            let hiddenBuffer = try Self.floatBuffer(hidden, device: context.device, label: "qwen.moe.hidden")
            let routingBuffer = try Self.floatBuffer(route.normalizedWeights[token], device: context.device, label: "qwen.moe.weights")
            let outputBuffer = try Self.emptyFloatBuffer(count: architecture.hiddenSize, device: context.device, label: "qwen.moe.output")
            let command: MTLCommandBuffer
            do {
                command = try moeRuntime.submitExperts(
                    hidden: hiddenBuffer, lease: lease, routingWeights: routingBuffer,
                    sharedBindings: weights.sharedBindings, scratch: moeScratch[layer],
                    output: outputBuffer)
                diagnostics.mappedExpertSubmissions += 1
            } catch {
                try? lease.cancel()
                throw error
            }
            await executionHooks.afterCommandSubmission("mappedExperts")
            try await awaitCompletion(
                command, stage: "mapped experts", diagnostics: &diagnostics,
                onCancel: { try? lease.cancel() })
            let values = Self.readFloats(outputBuffer, count: architecture.hiddenSize)
            output.replaceSubrange(
                (token * architecture.hiddenSize)..<((token + 1) * architecture.hiddenSize),
                with: values)
        }
        return output
    }

    private func attentionValues(
        rotatedQuery: [Float], allKeys: [Float], allValues: [Float],
        tokenCount: Int, cachedTokenCount: Int
    ) -> [Float] {
        let dimension = architecture.headDimension
        let queryWidth = architecture.queryHeads * dimension
        let totalKeyTokens = cachedTokenCount + tokenCount
        let groupSize = architecture.queryHeads / architecture.keyValueHeads
        let scale = 1 / Float(dimension).squareRoot()
        var output = [Float](repeating: 0, count: tokenCount * queryWidth)
        for token in 0..<tokenCount {
            let maximumKey = cachedTokenCount + token
            for queryHead in 0..<architecture.queryHeads {
                let kvHead = queryHead / groupSize
                let queryBase = (token * architecture.queryHeads + queryHead) * dimension
                var scores = [Float](repeating: 0, count: maximumKey + 1)
                for keyToken in 0...maximumKey {
                    let keyBase = (keyToken * architecture.keyValueHeads + kvHead) * dimension
                    for index in 0..<dimension {
                        scores[keyToken] += rotatedQuery[queryBase + index] * allKeys[keyBase + index]
                    }
                    scores[keyToken] *= scale
                }
                let maximum = scores.max() ?? 0
                var denominator: Float = 0
                for index in scores.indices {
                    scores[index] = Float(Foundation.exp(Double(scores[index] - maximum)))
                    denominator += scores[index]
                }
                for keyToken in scores.indices {
                    let probability = scores[keyToken] / denominator
                    let valueBase = (keyToken * architecture.keyValueHeads + kvHead) * dimension
                    for index in 0..<dimension {
                        output[queryBase + index] += probability * allValues[valueBase + index]
                    }
                }
            }
        }
        precondition(allKeys.count == totalKeyTokens * architecture.keyValueHeads * dimension)
        return output
    }

    private func expandKeyHeads(_ compact: [Float], tokenCount: Int) -> [Float] {
        var expanded = [Float](
            repeating: 0,
            count: tokenCount * architecture.linearValueHeads * architecture.linearKeyDimension)
        let headsPerKey = architecture.linearValueHeads / architecture.linearKeyHeads
        for token in 0..<tokenCount {
            for valueHead in 0..<architecture.linearValueHeads {
                let keyHead = valueHead / headsPerKey
                for dimension in 0..<architecture.linearKeyDimension {
                    expanded[(token * architecture.linearValueHeads + valueHead)
                        * architecture.linearKeyDimension + dimension] =
                        compact[(token * architecture.linearKeyHeads + keyHead)
                            * architecture.linearKeyDimension + dimension]
                }
            }
        }
        return expanded
    }

    private func awaitCompletion(
        _ commandBuffer: MTLCommandBuffer,
        stage: String,
        diagnostics: inout QwenTextExecutionDiagnostics,
        onCancel: @escaping @Sendable () -> Void = {}
    ) async throws {
        await withTaskCancellationHandler {
            await commandBuffer.completed()
        } onCancel: {
            onCancel()
        }
        guard commandBuffer.status == .completed, commandBuffer.error == nil else {
            diagnostics.discardedCommandBuffers += 1
            throw QwenTextRunnerError.gpuExecution(
                stage: stage,
                detail: commandBuffer.error?.localizedDescription
                    ?? "status \(commandBuffer.status.rawValue)")
        }
        diagnostics.completedCommandBuffers += 1
        try Task.checkCancellation()
    }

    private func restoreLinearState(from state: QwenTextRunnerState) throws {
        var layers: [Int: QwenLinearAttentionLayerState] = [:]
        for layer in 0..<architecture.layers {
            if case let .linear(history, recurrent) = state.layers[layer] {
                layers[layer] = QwenLinearAttentionLayerState(
                    convolutionHistory: history, recurrentMatrix: recurrent)
            }
        }
        let geometry = linearState.geometry
        try linearState.restore(QwenLinearAttentionSnapshot(geometry: geometry, layers: layers))
    }

    private func validate(_ state: QwenTextRunnerState) throws {
        guard state.architectureIdentity == model.mappingIdentity else {
            throw QwenTextRunnerError.stateArchitectureMismatch
        }
        guard state.sequenceLength >= 0, state.layers.count == architecture.layers else {
            throw QwenTextRunnerError.invalidState(detail: "sequence or layer count")
        }
        let convolutionCount = linearConfiguration.convolutionChannelCount * linearConfiguration.convolutionWidth
        let recurrentCount = linearConfiguration.valueHeadCount * linearConfiguration.keyHeadDimension * linearConfiguration.valueHeadDimension
        let kvWidth = architecture.keyValueHeads * architecture.headDimension
        for layer in 0..<architecture.layers {
            switch (architecture.layerKinds[layer], state.layers[layer]) {
            case let (.linearAttention, .linear(history, recurrent)):
                guard history.count == convolutionCount, recurrent.count == recurrentCount,
                      history.allSatisfy(\.isFinite), recurrent.allSatisfy(\.isFinite) else {
                    throw QwenTextRunnerError.invalidState(detail: "linear layer \(layer)")
                }
            case let (.fullAttention, .full(key, value)):
                let expected = state.sequenceLength * kvWidth
                guard key.count == expected, value.count == expected,
                      key.allSatisfy(\.isFinite), value.allSatisfy(\.isFinite) else {
                    throw QwenTextRunnerError.invalidState(detail: "full layer \(layer)")
                }
            default: throw QwenTextRunnerError.stateArchitectureMismatch
            }
        }
    }

    private static func floatBuffer(
        _ values: [Float], device: MTLDevice, label: String
    ) throws -> MTLBuffer {
        let bytes = values.count * MemoryLayout<Float>.stride
        guard bytes > 0,
              let buffer = device.makeBuffer(bytes: values, length: bytes, options: .storageModeShared) else {
            throw QwenTextRunnerError.execution(detail: "buffer allocation failed: \(label)")
        }
        buffer.label = label
        return buffer
    }

    private static func int32Buffer(
        _ values: [Int32], device: MTLDevice, label: String
    ) throws -> MTLBuffer {
        let bytes = values.count * MemoryLayout<Int32>.stride
        guard bytes > 0,
              let buffer = device.makeBuffer(
                bytes: values, length: bytes, options: .storageModeShared) else {
            throw QwenTextRunnerError.execution(
                detail: "buffer allocation failed: \(label)")
        }
        buffer.label = label
        return buffer
    }

    private static func normalizeAttentionHeads(
        _ values: [Float], storedWeight: [Float],
        dimension: Int, epsilon: Float
    ) -> [Float] {
        var output = [Float](repeating: 0, count: values.count)
        for item in 0..<(values.count / dimension) {
            let base = item * dimension
            var sum: Float = 0
            for index in 0..<dimension {
                sum += values[base + index] * values[base + index]
            }
            let inverse = 1 / sqrt(sum / Float(dimension) + epsilon)
            for index in 0..<dimension {
                output[base + index] = values[base + index]
                    * inverse * (1 + storedWeight[index])
            }
        }
        return output
    }

    private static func emptyFloatBuffer(
        count: Int, device: MTLDevice, label: String
    ) throws -> MTLBuffer {
        guard count > 0,
              let buffer = device.makeBuffer(
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            throw QwenTextRunnerError.execution(detail: "buffer allocation failed: \(label)")
        }
        buffer.label = label
        return buffer
    }

    private static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }
}

/// `MTLBuffer` predates Swift concurrency annotations. This wrapper is sent
/// only into the synchronous epoch-lock closure; the adapter owns the sole CPU
/// write, and no caller can observe it until that closure returns.
private struct QwenTextLogitsPublicationBuffer: @unchecked Sendable {
    let value: MTLBuffer
}

final class QwenTextLogitProducer: LogitProducer {
    private let runner: QwenTextRunner
    private let vocabularySize: Int
    private let epoch = OSAllocatedUnfairLock(initialState: UInt64(0))
    private let hooks: QwenTextPublicationHooks

    init(runner: QwenTextRunner, vocabularySize: Int, hooks: QwenTextPublicationHooks) {
        self.runner = runner
        self.vocabularySize = vocabularySize
        self.hooks = hooks
    }

    func reset() {
        epoch.withLock { $0 &+= 1 }
    }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await produce(
            token: token, cachePosition: position,
            ropePosition: nil, into: logits)
    }

    func produce(
        token: Int32,
        cachePosition: Int,
        ropePosition: QwenMRoPEPosition,
        into logits: MTLBuffer
    ) async throws {
        try await produce(
            token: token, cachePosition: cachePosition,
            ropePosition: Optional(ropePosition), into: logits)
    }

    func prefill(
        prepared input: QwenPreparedPrefill,
        into logits: MTLBuffer
    ) async throws {
        let requiredBytes = vocabularySize * MemoryLayout<Float16>.stride
        guard logits.length >= requiredBytes else {
            throw QwenTextRunnerError.logitsBufferTooSmall(
                expected: requiredBytes, actual: logits.length)
        }
        let generation = epoch.withLock { $0 }
        let prepared = try await runner.prepareMultimodalProduction(
            input, generation: generation,
            isCurrent: { [epoch] in epoch.withLock { $0 == generation } })
        let allLogits = prepared.result.output.logits
        guard allLogits.count >= vocabularySize else {
            try await runner.discardPrepared(identifier: prepared.identifier)
            throw QwenTextRunnerError.execution(
                detail: "multimodal prefill produced no final logits row")
        }
        let fp16 = allLogits.suffix(vocabularySize).map(Float16.init)
        let publicationBuffer = QwenTextLogitsPublicationBuffer(value: logits)
        await hooks.beforeLockedWrite()
        let published = epoch.withLock { current -> Bool in
            guard current == generation, !Task.isCancelled else { return false }
            _ = fp16.withUnsafeBytes { source in
                memcpy(publicationBuffer.value.contents(), source.baseAddress!, requiredBytes)
            }
            return true
        }
        guard published else {
            try await runner.discardPrepared(identifier: prepared.identifier)
            throw QwenTextRunnerError.cancelled
        }
        await hooks.afterLockedWriteBeforeFinish()
        await runner.commitPrepared(
            identifier: prepared.identifier,
            publishedState: prepared.result.output.state)
    }

    private func produce(
        token: Int32,
        cachePosition: Int,
        ropePosition: QwenMRoPEPosition?,
        into logits: MTLBuffer
    ) async throws {
        let requiredBytes = vocabularySize * MemoryLayout<Float16>.stride
        guard logits.length >= requiredBytes else {
            throw QwenTextRunnerError.logitsBufferTooSmall(
                expected: requiredBytes, actual: logits.length)
        }
        let generation = epoch.withLock { $0 }
        let prepared = try await runner.prepareProduction(
            token: token, cachePosition: cachePosition, ropePosition: ropePosition,
            generation: generation,
            isCurrent: { [epoch] in epoch.withLock { $0 == generation } })
        let fp16 = prepared.result.output.logits.map(Float16.init)
        let publicationBuffer = QwenTextLogitsPublicationBuffer(value: logits)
        await hooks.beforeLockedWrite()
        let published = epoch.withLock { current -> Bool in
            guard current == generation, !Task.isCancelled else { return false }
            _ = fp16.withUnsafeBytes { source in
                memcpy(publicationBuffer.value.contents(), source.baseAddress!, requiredBytes)
            }
            return true
        }
        guard published else {
            try await runner.discardPrepared(identifier: prepared.identifier)
            throw QwenTextRunnerError.cancelled
        }
        await hooks.afterLockedWriteBeforeFinish()
        await runner.commitPrepared(
            identifier: prepared.identifier,
            publishedState: prepared.result.output.state)
    }

    func snapshot() async throws -> QwenTextRunnerState {
        try await runner.snapshot()
    }

    var generationEpoch: UInt64 {
        epoch.withLock { $0 }
    }
}

extension QwenTextModel {
    func makeReferenceRunner() throws -> QwenTextReferenceRunner {
        try QwenTextReferenceRunner(model: self)
    }

    func makeRunner(
        context: MetalContext,
        expertSlotCount: Int = 8,
        executionHooks: QwenTextExecutionHooks = .none
    ) throws -> QwenTextRunner {
        try QwenTextRunner(
            model: self,
            context: context,
            expertSlotCount: expertSlotCount,
            executionHooks: executionHooks)
    }

    func makeLogitProducer(
        context: MetalContext,
        expertSlotCount: Int = 8,
        publicationHooks: QwenTextPublicationHooks = .none
    ) throws -> QwenTextLogitProducer {
        try QwenTextLogitProducer(
            runner: makeRunner(context: context, expertSlotCount: expertSlotCount),
            vocabularySize: architecture.vocabularySize,
            hooks: publicationHooks)
    }
}

private func rmsNorm(
    _ input: [Float],
    storedWeight: [Float],
    width: Int,
    epsilon: Float = 1e-6
) throws -> [Float] {
    guard input.count.isMultiple(of: width), storedWeight.count == width else {
        throw QwenTextRunnerError.execution(detail: "RMSNorm shape mismatch")
    }
    var output = [Float](repeating: 0, count: input.count)
    for vector in 0..<(input.count / width) {
        let base = vector * width
        var squareSum: Float = 0
        for index in 0..<width {
            squareSum += input[base + index] * input[base + index]
        }
        let inverseRMS = 1 / sqrt(squareSum / Float(width) + epsilon)
        for index in 0..<width {
            output[base + index] = input[base + index]
                * inverseRMS * (1 + storedWeight[index])
        }
    }
    return output
}

private func project(
    _ input: [Float],
    matrix: [Float],
    rows: Int,
    columns: Int
) -> [Float] {
    precondition(input.count.isMultiple(of: columns))
    precondition(matrix.count == rows * columns)
    let tokenCount = input.count / columns
    var output = [Float](repeating: 0, count: tokenCount * rows)
    for token in 0..<tokenCount {
        for row in 0..<rows {
            var sum: Float = 0
            for column in 0..<columns {
                sum += input[token * columns + column]
                    * matrix[row * columns + column]
            }
            output[token * rows + row] = sum
        }
    }
    return output
}
