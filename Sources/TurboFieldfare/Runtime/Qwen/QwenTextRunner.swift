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
        let positions = Dictionary(uniqueKeysWithValues: layers.keys.map { ($0, 0) })
        try linearState.restore(QwenLinearAttentionSnapshot(
            geometry: geometry, layers: layers, positions: positions))
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

/// Names of four independently admitted, resident BF16 matrices. The weight
/// owner validates both their role and the actual Safetensors header geometry.
struct QwenBF16FullAttentionNames: Sendable {
    let query: String
    let key: String
    let value: String
    let output: String
}

/// Internal deterministic failure seam. Production uses `.none`; the hooks
/// cannot provide weights, offsets, command buffers or committed KV storage.
struct QwenBF16FullAttentionHooks: Sendable {
    enum Checkpoint: Sendable {
        case beforeSubmission(stage: String)
        case afterSubmission(stage: String)
        case beforeCommit
    }

    let checkpoint: @Sendable (Checkpoint) async throws -> Void
    /// A supplied callback retains its original submission/completion boundary.
    let requiresSeparateGPUStages: Bool

    init(checkpoint: (@Sendable (Checkpoint) async throws -> Void)? = nil) {
        self.checkpoint = checkpoint ?? { _ in }
        requiresSeparateGPUStages = checkpoint != nil
    }

    static let none = Self()
}

/// Additive one-token full-attention component, not a source runtime factory.
/// The input row has already passed the layer's input RMSNorm. Only short FP32
/// vectors/scores are materialized; all four full matrices stay resident BF16.
/// No command buffer touches the committed KV until both rows are published.
actor QwenBF16FullAttentionStep {
    private let context: MetalContext
    private let weights: QwenBF16Weights
    private let names: QwenBF16FullAttentionNames
    private let attention: QwenFullAttention
    private let kv: QwenFullAttentionKV
    private let layer: Int
    private let hiddenSize: Int
    private let queryWidth: Int
    private let keyValueWidth: Int
    private let queryNormBuffer: MTLBuffer
    private let keyNormBuffer: MTLBuffer
    private let hooks: QwenBF16FullAttentionHooks
    private var inFlight = false

    init(context: MetalContext, weights: QwenBF16Weights,
         names: QwenBF16FullAttentionNames,
         configuration: QwenFullAttentionConfiguration,
         hiddenSize: Int, layer: Int, kv: QwenFullAttentionKV,
         queryNorm: [Float], keyNorm: [Float],
         hooks: QwenBF16FullAttentionHooks = .none) throws {
        let queryWidth = try Self.product(
            configuration.queryHeadCount, configuration.headDimension, label: "query width")
        let keyValueWidth = try Self.product(
            configuration.keyValueHeadCount, configuration.headDimension,
            label: "key/value width")
        let doubledQuery = try Self.product(2, queryWidth, label: "query/gate width")
        guard hiddenSize > 0, hiddenSize <= Int(UInt32.max),
              doubledQuery <= Int(UInt32.max), keyValueWidth <= Int(UInt32.max),
              kv.fullLayerIndices.contains(layer),
              kv.keyValueHeadCount == configuration.keyValueHeadCount,
              kv.headDimension == configuration.headDimension,
              queryNorm.count == configuration.headDimension,
              keyNorm.count == configuration.headDimension,
              queryNorm.allSatisfy(\.isFinite), keyNorm.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 full-attention head/norm/KV geometry")
        }
        let view = try kv.view(layer: layer)
        let expectedKVStride = try Self.product(
            keyValueWidth, MemoryLayout<Float>.stride, label: "KV stride")
        guard view.key.device === context.device, view.value.device === context.device,
              view.key.storageMode == .shared, view.value.storageMode == .shared,
              view.strideBytes == expectedKVStride else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 full-attention KV device/stride")
        }
        try weights.requireShape(names.query, role: .dense,
                                 rows: doubledQuery, columns: hiddenSize)
        try weights.requireShape(names.key, role: .dense,
                                 rows: keyValueWidth, columns: hiddenSize)
        try weights.requireShape(names.value, role: .dense,
                                 rows: keyValueWidth, columns: hiddenSize)
        try weights.requireShape(names.output, role: .dense,
                                 rows: hiddenSize, columns: queryWidth)
        let queryNormBuffer = try Self.floatBuffer(
            queryNorm, device: context.device, label: "qwen.bf16.attention.qnorm")
        let keyNormBuffer = try Self.floatBuffer(
            keyNorm, device: context.device, label: "qwen.bf16.attention.knorm")
        self.context = context
        self.weights = weights
        self.names = names
        attention = try QwenFullAttention(
            context: context, configuration: configuration, useOfficialSourceMath: true)
        self.kv = kv
        self.layer = layer
        self.hiddenSize = hiddenSize
        self.queryWidth = queryWidth
        self.keyValueWidth = keyValueWidth
        self.queryNormBuffer = queryNormBuffer
        self.keyNormBuffer = keyNormBuffer
        self.hooks = hooks
    }

    func committedPosition() throws -> Int {
        try kv.committedPosition(layer: layer)
    }

    /// Prepare one complete candidate, settle every submitted GPU operation,
    /// and publish K, V and position together only after the final checkpoint.
    /// Reentrant actor calls fail closed while this method is suspended.
    func append(hidden: [Float], mropePosition: QwenMRoPEPosition? = nil,
                mropeSections: [Int] = []) async throws -> [Float] {
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        guard hidden.count == hiddenSize, hidden.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 attention input row")
        }
        inFlight = true
        defer { inFlight = false }
        try Task.checkCancellation()
        let position = try kv.committedPosition(layer: layer)
        let write = try kv.reserveWrite(layer: layer, position: position, tokenCount: 1)
        var published = false
        defer { if !published { try? kv.abandon(write) } }
        let cache = try kv.view(layer: layer)
        let qAndGateCount = try Self.product(2, queryWidth, label: "query/gate row")
        let input = try Self.floatBuffer(hidden, device: context.device,
                                         label: "qwen.bf16.attention.input")
        let projected = try emptyBuffer(qAndGateCount, label: "qwen.bf16.attention.qgate")
        let key = try emptyBuffer(keyValueWidth, label: "qwen.bf16.attention.key")
        let value = try emptyBuffer(keyValueWidth, label: "qwen.bf16.attention.value")
        let projectionCommand = try commandBuffer(stage: "qkv")
        try weights.encodeProjection(commandBuffer: projectionCommand,
                                     tensorName: names.query, input: input,
                                     tokenCount: 1, output: projected)
        try weights.encodeProjection(commandBuffer: projectionCommand,
                                     tensorName: names.key, input: input,
                                     tokenCount: 1, output: key)
        try weights.encodeProjection(commandBuffer: projectionCommand,
                                     tensorName: names.value, input: input,
                                     tokenCount: 1, output: value)
        try await submitAndSettle(projectionCommand, stage: "qkv")

        // The Q source row is [head, query then gate, dimension]. The two
        // kernel inputs are [head, dimension], so copy one short row only.
        let qAndGate = projected.contents().assumingMemoryBound(to: Float.self)
        var queryValues = [Float](repeating: 0, count: queryWidth)
        var gateValues = [Float](repeating: 0, count: queryWidth)
        let dimension = attention.configuration.headDimension
        for head in 0..<attention.configuration.queryHeadCount {
            for column in 0..<dimension {
                let target = head * dimension + column
                let source = head * 2 * dimension + column
                queryValues[target] = qAndGate[source]
                gateValues[target] = qAndGate[source + dimension]
            }
        }
        guard queryValues.allSatisfy(\.isFinite), gateValues.allSatisfy(\.isFinite) else {
            throw QwenFullAttentionError.nonfiniteAttention
        }
        let queryInput = try Self.floatBuffer(queryValues, device: context.device,
                                               label: "qwen.bf16.attention.query")
        let rotatedQuery = try emptyBuffer(queryWidth, label: "qwen.bf16.attention.rotatedQuery")
        let rotatedKey = try emptyBuffer(keyValueWidth, label: "qwen.bf16.attention.rotatedKey")
        let normCommand = try commandBuffer(stage: "normRoPE")
        if let mropePosition {
            let sections = mropeSections == [1]
                && attention.configuration.rotaryDimension == 2
                ? [1, 0, 0] : mropeSections
            let values = mropePosition.values
            guard let positionBuffer = context.device.makeBuffer(
                bytes: values, length: values.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared) else {
                throw QwenTextRunnerError.execution(detail: "source M-RoPE position allocation")
            }
            try attention.encodeNormAndPartialMRoPE(
                commandBuffer: normCommand, input: queryInput, weight: queryNormBuffer,
                positions: positionBuffer, output: rotatedQuery, tokenCount: 1,
                headCount: attention.configuration.queryHeadCount, sections: sections)
            try attention.encodeNormAndPartialMRoPE(
                commandBuffer: normCommand, input: key, weight: keyNormBuffer,
                positions: positionBuffer, output: rotatedKey, tokenCount: 1,
                headCount: attention.configuration.keyValueHeadCount, sections: sections)
            try await submitAndSettle(normCommand, stage: "normRoPE")
        } else {
            try attention.encodeNormAndPartialRoPE(
                commandBuffer: normCommand, input: queryInput, weight: queryNormBuffer,
                output: rotatedQuery, tokenCount: 1,
                headCount: attention.configuration.queryHeadCount, startPosition: position)
            try attention.encodeNormAndPartialRoPE(
                commandBuffer: normCommand, input: key, weight: keyNormBuffer,
                output: rotatedKey, tokenCount: 1,
                headCount: attention.configuration.keyValueHeadCount, startPosition: position)
            try await submitAndSettle(normCommand, stage: "normRoPE")
        }
        let preGate = try attention.attentionStep(
            rotatedQuery: Self.readFloats(rotatedQuery, count: queryWidth),
            rotatedKey: Self.readFloats(rotatedKey, count: keyValueWidth),
            value: Self.readFloats(value, count: keyValueWidth), cache: cache)
        let attentionInput = try Self.floatBuffer(preGate, device: context.device,
                                                  label: "qwen.bf16.attention.values")
        let gateInput = try Self.floatBuffer(gateValues, device: context.device,
                                             label: "qwen.bf16.attention.gate")
        let gated = try emptyBuffer(queryWidth, label: "qwen.bf16.attention.gated")
        let gateCommand = try commandBuffer(stage: "gate")
        try attention.encodeOutputGate(
            commandBuffer: gateCommand, attention: attentionInput,
            rawGate: gateInput, output: gated, elementCount: queryWidth)
        if hooks.requiresSeparateGPUStages {
            try await submitAndSettle(gateCommand, stage: "gate")
        }
        // Unobserved stages share a command, keeping the same encoder order.
        let output = try emptyBuffer(hiddenSize, label: "qwen.bf16.attention.output")
        let outputCommand = hooks.requiresSeparateGPUStages
            ? try commandBuffer(stage: "output") : gateCommand
        try weights.encodeProjection(commandBuffer: outputCommand,
                                     tensorName: names.output, input: gated,
                                     tokenCount: 1, output: output)
        try await submitAndSettle(outputCommand,
            stage: hooks.requiresSeparateGPUStages ? "output" : "gate-output")
        let result = Self.readFloats(output, count: hiddenSize)
        guard result.allSatisfy(\.isFinite) else {
            throw QwenFullAttentionError.nonfiniteAttention
        }
        try await hooks.checkpoint(.beforeCommit)
        try Task.checkCancellation()
        try kv.commitStaged(write, key: rotatedKey, value: value)
        published = true
        return result
    }

    private func submitAndSettle(_ command: MTLCommandBuffer,
                                 stage: String) async throws {
        try await hooks.checkpoint(.beforeSubmission(stage: stage))
        try Task.checkCancellation()
        command.commit()
        // A hook may suspend, throw or observe cancellation after submission.
        // Regardless, settle this command before any caller releases scratch.
        var hookError: Error?
        do { try await hooks.checkpoint(.afterSubmission(stage: stage)) }
        catch { hookError = error }
        // Match the existing hybrid runner's cancellation-safe completion
        // boundary: cancellation never releases submitted GPU scratch early.
        await withTaskCancellationHandler {
            await command.completed()
        } onCancel: {
            // GPU work cannot be cancelled; always await its actual completion.
        }
        if let hookError { throw hookError }
        guard command.status == .completed, command.error == nil else {
            throw QwenTextRunnerError.gpuExecution(
                stage: stage, detail: command.error?.localizedDescription
                    ?? "status \(command.status.rawValue)")
        }
        try Task.checkCancellation()
    }

    private func commandBuffer(stage: String) throws -> MTLCommandBuffer {
        guard let command = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.gpuExecution(
                stage: stage, detail: "command buffer unavailable")
        }
        return command
    }

    private func emptyBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
        let bytes = try Self.product(elements, MemoryLayout<Float>.stride, label: label)
        guard bytes <= context.device.maxBufferLength,
              let buffer = context.device.makeBuffer(length: bytes, options: .storageModeShared)
        else { throw QwenTextRunnerError.execution(detail: "BF16 attention allocation: \(label)") }
        buffer.label = label
        return buffer
    }

    private static func floatBuffer(_ values: [Float], device: MTLDevice,
                                    label: String) throws -> MTLBuffer {
        let bytes = try product(values.count, MemoryLayout<Float>.stride, label: label)
        guard bytes <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
        else { throw QwenTextRunnerError.execution(detail: "BF16 attention allocation: \(label)") }
        try values.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else {
                throw QwenTextRunnerError.execution(detail: "BF16 attention empty input")
            }
            buffer.contents().copyMemory(from: base, byteCount: bytes)
        }
        buffer.label = label
        return buffer
    }

    private static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let base = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    private static func product(_ lhs: Int, _ rhs: Int, label: String) throws -> Int {
        guard lhs > 0, rhs > 0 else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 attention \(label) is empty")
        }
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 attention \(label) overflow")
        }
        return result
    }
}


/// Five named dense BF16 matrices, selected from the already admitted resident set.
struct QwenBF16LinearNames: Sendable {
    let qkv: String
    let z: String
    let b: String
    let a: String
    let output: String
}

/// Only small FP32 vectors are kept beside resident BF16 matrices.
struct QwenBF16LinearVectors: Sendable {
    let convolution: [Float]
    let normalization: [Float]
    let aLog: [Float]
    let timeStepBias: [Float]
}

struct QwenBF16LinearHooks: Sendable {
    enum Checkpoint: Sendable {
        case beforeSubmission(stage: String)
        case afterSubmission(stage: String)
        case beforeCommit
    }

    let checkpoint: @Sendable (Checkpoint) async throws -> Void
    /// Observations never read a stage before that stage has settled.
    let requiresSeparateGPUStages: Bool
    /// First input position, stage, actual FP32 activations. Default nil.
    let observeActivation: (@Sendable (Int, String, [Float]) -> Void)?

    init(observeActivation: (@Sendable (Int, String, [Float]) -> Void)? = nil,
         checkpoint: (@Sendable (Checkpoint) async throws -> Void)? = nil) {
        self.checkpoint = checkpoint ?? { _ in }
        self.observeActivation = observeActivation
        requiresSeparateGPUStages = checkpoint != nil || observeActivation != nil
    }

    static let none = Self()
}

/// A separate BF16 linear step. The packed runner's two early state submissions
/// are deliberately untouched. One reservation spans convolution, recurrence
/// and output; none of its GPU writes alias committed history or matrix storage.
actor QwenBF16LinearStep {
    private let context: MetalContext
    private let weights: QwenBF16Weights
    private let names: QwenBF16LinearNames
    private let runtime: QwenGatedDeltaNet
    private let configuration: QwenGatedDeltaNetConfiguration
    private let state: QwenLinearAttentionState
    private let layer: Int
    private let hooks: QwenBF16LinearHooks
    private let convolutionWeights: MTLBuffer
    private let normWeights: MTLBuffer
    private let aLog: [Float]
    private let timeStepBias: [Float]
    private var inFlight = false

    init(context: MetalContext, weights: QwenBF16Weights,
         names: QwenBF16LinearNames,
         configuration: QwenGatedDeltaNetConfiguration,
         vectors: QwenBF16LinearVectors, layer: Int,
         state: QwenLinearAttentionState,
         hooks: QwenBF16LinearHooks = .none) throws {
        let channels = configuration.convolutionChannelCount
        let valueWidth = configuration.valueDimension
        let recurrent = try Self.product(
            configuration.valueHeadCount,
            try Self.product(configuration.keyHeadDimension,
                             configuration.valueHeadDimension, label: "recurrent dimensions"),
            label: "recurrent elements")
        let convolutionElements = try Self.product(
            channels, configuration.convolutionWidth, label: "convolution weights")
        guard state.isBound(to: context.device),
              weights.inspectedChunks.allSatisfy({ $0.buffer.device === context.device }),
              state.linearLayerIndices.contains(layer),
              state.geometry.convolutionWidth == configuration.convolutionWidth,
              state.geometry.convolutionChannelCount == channels,
              state.geometry.valueHeadCount == configuration.valueHeadCount,
              state.geometry.keyHeadDimension == configuration.keyHeadDimension,
              state.geometry.valueHeadDimension == configuration.valueHeadDimension,
              state.geometry.recurrentElementCount == recurrent,
              vectors.convolution.count == convolutionElements,
              vectors.normalization.count == configuration.valueHeadDimension,
              vectors.aLog.count == configuration.valueHeadCount,
              vectors.timeStepBias.count == configuration.valueHeadCount,
              vectors.convolution.allSatisfy(\.isFinite),
              vectors.normalization.allSatisfy(\.isFinite),
              vectors.aLog.allSatisfy({
                  $0.isFinite && Float(Foundation.exp(Double($0))).isFinite
              }),
              vectors.timeStepBias.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear configuration/vectors/state")
        }
        try weights.requireShape(names.qkv, role: .dense,
                                 rows: channels, columns: configuration.hiddenSize)
        try weights.requireShape(names.z, role: .dense,
                                 rows: valueWidth, columns: configuration.hiddenSize)
        try weights.requireShape(names.b, role: .dense,
                                 rows: configuration.valueHeadCount, columns: configuration.hiddenSize)
        try weights.requireShape(names.a, role: .dense,
                                 rows: configuration.valueHeadCount, columns: configuration.hiddenSize)
        try weights.requireShape(names.output, role: .dense,
                                 rows: configuration.hiddenSize, columns: valueWidth)
        let conv = try Self.floatBuffer(vectors.convolution, device: context.device,
                                        label: "qwen.bf16.linear.convWeights")
        let norm = try Self.floatBuffer(vectors.normalization, device: context.device,
                                        label: "qwen.bf16.linear.normWeights")
        self.context = context
        self.weights = weights
        self.names = names
        self.configuration = configuration
        runtime = try QwenGatedDeltaNet(context: context, configuration: configuration,
                                      useOfficialSourceMath: true)
        self.state = state
        self.layer = layer
        self.hooks = hooks
        convolutionWeights = conv
        normWeights = norm
        aLog = vectors.aLog
        timeStepBias = vectors.timeStepBias
    }

    func snapshot() throws -> QwenBF16LinearStepSnapshot {
        try state.layerSnapshot(layer)
    }

    func append(normalizedHidden: [Float], tokenCount: Int) async throws -> [Float] {
        guard !inFlight else { throw QwenTextRunnerError.operationInProgress }
        // Bound per-call scratch and all UInt32 arithmetic used inside Metal.
        guard (1...256).contains(tokenCount) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear token count")
        }
        let hidden = configuration.hiddenSize
        let channels = configuration.convolutionChannelCount
        let keyWidth = configuration.keyDimension
        let valueWidth = configuration.valueDimension
        let heads = configuration.valueHeadCount
        let queryWidth = try Self.product(heads, configuration.keyHeadDimension,
                                          label: "expanded query width")
        let inputCount = try Self.product(tokenCount, hidden, label: "input")
        let qkvCount = try Self.product(tokenCount, channels, label: "QKV")
        let valueCount = try Self.product(tokenCount, valueWidth, label: "value")
        let scalarCount = try Self.product(tokenCount, heads, label: "scalars")
        let queryCount = try Self.product(tokenCount, queryWidth, label: "query")
        guard normalizedHidden.count == inputCount,
              normalizedHidden.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear normalized input")
        }
        // No GPU buffer is allocated before every per-stage allocation size is
        // proven representable and below this device's physical per-buffer cap.
        for count in [inputCount, qkvCount, valueCount, scalarCount, queryCount,
                      configuration.valueHeadDimension] {
            _ = try Self.checkedBytes(count, device: context.device)
        }
        let position = try state.committedPosition(layer: layer)
        let (_, positionOverflow) = position.addingReportingOverflow(tokenCount)
        guard !positionOverflow else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear position overflow")
        }
        inFlight = true
        defer { inFlight = false }
        try Task.checkCancellation()
        let update = try state.reserveUpdate(layer: layer)
        var published = false
        defer { if !published { try? state.cancel(update) } }

        let input = try Self.floatBuffer(normalizedHidden, device: context.device,
                                          label: "qwen.bf16.linear.input")
        let qkv = try emptyBuffer(qkvCount, label: "qwen.bf16.linear.qkv")
        let gate = try emptyBuffer(valueCount, label: "qwen.bf16.linear.z")
        let rawBeta = try emptyBuffer(scalarCount, label: "qwen.bf16.linear.b")
        let rawA = try emptyBuffer(scalarCount, label: "qwen.bf16.linear.a")
        let projection = try commandBuffer(stage: "projections")
        try weights.encodeProjection(commandBuffer: projection, tensorName: names.qkv,
                                     input: input, tokenCount: tokenCount, output: qkv)
        try weights.encodeProjection(commandBuffer: projection, tensorName: names.z,
                                     input: input, tokenCount: tokenCount, output: gate)
        try weights.encodeProjection(commandBuffer: projection, tensorName: names.b,
                                     input: input, tokenCount: tokenCount, output: rawBeta)
        try weights.encodeProjection(commandBuffer: projection, tensorName: names.a,
                                     input: input, tokenCount: tokenCount, output: rawA)
        if hooks.requiresSeparateGPUStages {
            try await submitAndSettle(projection, stage: "projections")
        }
        if let observe = hooks.observeActivation {
            observe(position, "input", normalizedHidden)
            observe(position, "qkv", Self.readFloats(qkv, count: qkvCount))
            observe(position, "z", Self.readFloats(gate, count: valueCount))
            observe(position, "b", Self.readFloats(rawBeta, count: scalarCount))
            observe(position, "a", Self.readFloats(rawA, count: scalarCount))
        }

        let channelMajor = try emptyBuffer(qkvCount, label: "qwen.bf16.linear.channelMajor")
        let convolvedMajor = try emptyBuffer(qkvCount, label: "qwen.bf16.linear.convolvedMajor")
        let convolved = try emptyBuffer(qkvCount, label: "qwen.bf16.linear.convolved")
        // One deferred state reservation still owns both combined commands.
        let convolution = hooks.requiresSeparateGPUStages
            ? try commandBuffer(stage: "convolution") : projection
        try runtime.encodeLayout(commandBuffer: convolution, input: qkv,
                                 output: channelMajor, tokenCount: tokenCount,
                                 channelCount: channels, tokenToChannel: true)
        try runtime.encodeCausalConvolution(
            commandBuffer: convolution, channelMajorInput: channelMajor,
            weights: convolutionWeights, update: update,
            channelMajorOutput: convolvedMajor, tokenCount: tokenCount)
        try runtime.encodeLayout(commandBuffer: convolution, input: convolvedMajor,
                                 output: convolved, tokenCount: tokenCount,
                                 channelCount: channels, tokenToChannel: false)
        try await submitAndSettle(convolution,
            stage: hooks.requiresSeparateGPUStages ? "convolution" : "projections-convolution",
            update: update)

        let projected = Self.readFloats(convolved, count: qkvCount)
        let betaRaw = Self.readFloats(rawBeta, count: scalarCount)
        let aRaw = Self.readFloats(rawA, count: scalarCount)
        guard projected.allSatisfy(\.isFinite), betaRaw.allSatisfy(\.isFinite),
              aRaw.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear nonfinite projections")
        }
        hooks.observeActivation?(position, "convolved", projected)
        var query = [Float](repeating: 0, count: queryCount)
        var key = [Float](repeating: 0, count: queryCount)
        var value = [Float](repeating: 0, count: valueCount)
        var beta = [Float](repeating: 0, count: scalarCount)
        var decay = [Float](repeating: 0, count: scalarCount)
        for token in 0..<tokenCount {
            for head in 0..<heads {
                let sourceHead = head / configuration.headsPerKeyHead
                let sourceBase = token * channels + sourceHead * configuration.keyHeadDimension
                let targetBase = (token * heads + head) * configuration.keyHeadDimension
                for dimension in 0..<configuration.keyHeadDimension {
                    query[targetBase + dimension] = projected[sourceBase + dimension]
                    key[targetBase + dimension] = projected[sourceBase + keyWidth + dimension]
                }
            }
            let sourceBase = token * channels + 2 * keyWidth
            let valueBase = token * valueWidth
            for index in 0..<valueWidth {
                value[valueBase + index] = projected[sourceBase + index]
            }
            for head in 0..<heads {
                let index = token * heads + head
                beta[index] = 1 / (1 + QwenOfficialSourceRouterArithmetic.exponential(-betaRaw[index]))
                let x = aRaw[index] + timeStepBias[head]
                let softplus = x > 20 ? x
                    : QwenOfficialSourcePositiveLog1p.evaluate(QwenOfficialSourceRouterArithmetic.exponential(x))
                decay[index] = -QwenOfficialSourceRouterArithmetic.exponential(aLog[head]) * softplus
            }
        }
        guard beta.allSatisfy(\.isFinite), decay.allSatisfy(\.isFinite),
              query.allSatisfy(\.isFinite), key.allSatisfy(\.isFinite),
              value.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear nonfinite recurrence inputs")
        }
        if let observe = hooks.observeActivation {
            observe(position, "query-raw", query)
            observe(position, "key-raw", key)
            observe(position, "value", value)
            observe(position, "beta", beta)
            observe(position, "log-decay", decay)
        }
        let queryBuffer = try Self.floatBuffer(query, device: context.device,
                                                label: "qwen.bf16.linear.query")
        let keyBuffer = try Self.floatBuffer(key, device: context.device,
                                              label: "qwen.bf16.linear.key")
        let valueBuffer = try Self.floatBuffer(value, device: context.device,
                                                label: "qwen.bf16.linear.value")
        let betaBuffer = try Self.floatBuffer(beta, device: context.device,
                                               label: "qwen.bf16.linear.beta")
        let decayBuffer = try Self.floatBuffer(decay, device: context.device,
                                                label: "qwen.bf16.linear.decay")
        let recurrenceOutput = try emptyBuffer(valueCount, label: "qwen.bf16.linear.recurrence")
        let gated = try emptyBuffer(valueCount, label: "qwen.bf16.linear.gated")
        let recurrence = try commandBuffer(stage: "recurrence")
        try runtime.encodeRecurrence(
            commandBuffer: recurrence, query: queryBuffer, key: keyBuffer,
            value: valueBuffer, logDecay: decayBuffer, beta: betaBuffer,
            update: update, output: recurrenceOutput, tokenCount: tokenCount,
            initialToken: position == 0)
        try runtime.encodeGatedRMSNorm(
            commandBuffer: recurrence, input: recurrenceOutput, gate: gate,
            weights: normWeights, output: gated, tokenCount: tokenCount)
        if hooks.requiresSeparateGPUStages {
            try await submitAndSettle(recurrence, stage: "recurrence", update: update)
        }
        if let observe = hooks.observeActivation {
            observe(position, "core", Self.readFloats(recurrenceOutput, count: valueCount))
            observe(position, "gated", Self.readFloats(gated, count: valueCount))
        }

        let output = try emptyBuffer(inputCount, label: "qwen.bf16.linear.output")
        let outputCommand = hooks.requiresSeparateGPUStages
            ? try commandBuffer(stage: "output") : recurrence
        try weights.encodeProjection(commandBuffer: outputCommand,
                                     tensorName: names.output, input: gated,
                                     tokenCount: tokenCount, output: output)
        try await submitAndSettle(outputCommand,
            stage: hooks.requiresSeparateGPUStages ? "output" : "recurrence-output",
            update: hooks.requiresSeparateGPUStages ? nil : update)
        let result = Self.readFloats(output, count: inputCount)
        guard result.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear nonfinite output")
        }
        hooks.observeActivation?(position, "output", result)
        try await hooks.checkpoint(.beforeCommit)
        try Task.checkCancellation()
        try state.commitDeferred(update, tokenCount: tokenCount)
        published = true
        return result
    }

    private func submitAndSettle(_ command: MTLCommandBuffer, stage: String,
                                 update: QwenLinearAttentionUpdate? = nil) async throws {
        try await hooks.checkpoint(.beforeSubmission(stage: stage))
        try Task.checkCancellation()
        if let update { try state.submitDeferred(update, on: command) }
        else { command.commit() }
        var hookError: Error?
        do { try await hooks.checkpoint(.afterSubmission(stage: stage)) }
        catch { hookError = error }
        // Cancellation or a throwing hook must not free any submitted buffer.
        await withTaskCancellationHandler {
            await command.completed()
        } onCancel: {
            // Metal work must settle; never release the reserved pair early.
        }
        if let update { await state.waitForDeferredStage(update) }
        if let hookError { throw hookError }
        guard command.status == .completed, command.error == nil else {
            throw QwenTextRunnerError.gpuExecution(
                stage: stage, detail: command.error?.localizedDescription
                    ?? "status \(command.status.rawValue)")
        }
        try Task.checkCancellation()
    }

    private func commandBuffer(stage: String) throws -> MTLCommandBuffer {
        guard let command = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.gpuExecution(
                stage: stage, detail: "command buffer unavailable")
        }
        return command
    }

    private func emptyBuffer(_ count: Int, label: String) throws -> MTLBuffer {
        try Self.allocate(count, device: context.device, label: label)
    }

    private static func floatBuffer(_ values: [Float], device: MTLDevice,
                                    label: String) throws -> MTLBuffer {
        let buffer = try allocate(values.count, device: device, label: label)
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                buffer.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return buffer
    }

    private static func allocate(_ count: Int, device: MTLDevice,
                                 label: String) throws -> MTLBuffer {
        let bytes = try checkedBytes(count, device: device)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw QwenTextRunnerError.execution(detail: "BF16 linear allocation: \(label)")
        }
        buffer.label = label
        return buffer
    }

    private static func checkedBytes(_ count: Int, device: MTLDevice) throws -> Int {
        let bytes = try product(count, MemoryLayout<Float>.stride, label: "buffer bytes")
        guard bytes <= device.maxBufferLength else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear Metal buffer limit")
        }
        return bytes
    }

    private static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let base = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    private static func product(_ lhs: Int, _ rhs: Int, label: String) throws -> Int {
        guard lhs > 0, rhs > 0 else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear empty \(label)")
        }
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow, value <= Int(UInt32.max) else {
            throw QwenTextRunnerError.invalidState(detail: "BF16 linear overflow \(label)")
        }
        return value
    }
}
