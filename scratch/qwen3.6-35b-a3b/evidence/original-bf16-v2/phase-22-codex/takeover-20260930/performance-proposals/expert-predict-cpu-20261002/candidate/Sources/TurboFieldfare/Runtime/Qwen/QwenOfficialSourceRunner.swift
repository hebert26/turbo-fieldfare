import Foundation
import Metal
import Synchronization

struct QwenOfficialSourceRunnerDiagnosticSnapshot: Sendable, Equatable {
    let position: Int
    let linear: QwenLinearAttentionSnapshot
    let fullKVPositions: [Int: Int]
    let committedKeys: [Int: [Float]]
    let committedValues: [Int: [Float]]
}

struct QwenOfficialSourceCacheDiagnostics: Sendable, Equatable {
    let slotCount: Int
    let policy: ExpertCachePolicy
    let integrityPolicy: ModelIntegrityPolicy?
    let allocatedBytes: UInt64
    let routedExpertCount: Int
    let summary: RoutedExpertCacheSummary
}

/// Explicit grouped observations have different timing from token publication:
/// input rows are formed first, routes are layer-major, and only final logits emit.
struct QwenSourceGroupedPrefillCapture: Sendable {
    let observeRoute: (@Sendable (Int, Int, [Float], [Int], [Float], Float) -> Void)?
    let observeConsumedInput: (@Sendable (Int, Int32, [Float]?, QwenMRoPEPosition?) -> Void)?
    let observeFinalRawLogits: (@Sendable (Int, Int32, [Float]) -> Void)?

    init(observeRoute: (@Sendable (Int, Int, [Float], [Int], [Float], Float) -> Void)? = nil,
         observeConsumedInput: (@Sendable (Int, Int32, [Float]?, QwenMRoPEPosition?) -> Void)? = nil,
         observeFinalRawLogits: (@Sendable (Int, Int32, [Float]) -> Void)? = nil) {
        self.observeRoute = observeRoute
        self.observeConsumedInput = observeConsumedInput
        self.observeFinalRawLogits = observeFinalRawLogits
    }
}

struct QwenSourceGroupedPrefillDiagnostics: Sendable, Equatable {
    enum Mode: Sendable, Equatable { case grouped, tokenMajor }
    let mode: Mode
    let tokenCount: Int
    var completedLayers: Int = 0
    var groupedLinearBatchSizes: [Int: [Int]] = [:]
    var mappedUniqueExperts: [Int: Int] = [:]
    var mappingHits: UInt64 = 0
    var mappingMisses: UInt64 = 0
}

/// The original-BF16 source decoder. One actor owns position and the persistent
/// convolution/recurrent and full-attention caches. Only one token can be in
/// flight, including while GPU submissions are suspended. Full-vocabulary
/// logits remain raw FP32 until the service's explicit sampler conversion.
actor QwenOfficialSourceRunner {
    private let model: QwenOfficialSourceModel
    private let context: MetalContext
    private let linearState: QwenLinearAttentionState
    private let fullKV: QwenFullAttentionKV
    private let linearSteps: [Int: QwenBF16LinearStep]
    private let fullSteps: [Int: QwenBF16FullAttentionStep]
    private let moe: QwenMoE
    private let moeConfiguration: QwenMoEConfiguration
    private let expertSlotCount: Int
    private let predictionCapture: QwenExpertPredictionCapture?
    private let hooks: QwenOfficialSourceTransactionHooks
    private let useGPULinearPreparation: Bool
    private let useGroupedLinearPrefill: Bool
    /// Only this actor mutates the dictionary. A GPU lease can independently
    /// retain a coordinator, whose hooks retain the same quota reservation.
    private final class ExpertCacheStorage: @unchecked Sendable {
        let reservation: QwenOfficialSourceModel.ExpertCacheReservation
        let residency: QwenBF16CacheResidency?
        var coordinators: [Int: QwenBF16ExpertMappingCoordinator] = [:]

        init(reservation: QwenOfficialSourceModel.ExpertCacheReservation,
             residency: QwenBF16CacheResidency?) {
            self.reservation = reservation
            self.residency = residency
        }

        deinit { coordinators.removeAll() }
    }
    private let expertCacheStorage: ExpertCacheStorage
    private struct TurnCheckpoint {
        let linear: QwenLinearAttentionSnapshot
        var full: QwenFullAttentionKVSnapshot
        let position: Int
        var replacementFull: QwenFullAttentionKV.ReplacementCheckpoint? = nil
    }
    private var turnCheckpoint: TurnCheckpoint?
    private var lastPrefillDiagnostics: QwenSourceGroupedPrefillDiagnostics?
    private var committedPosition = 0
    private var inFlight = false
    private var prefillInFlight = false
    private var unusable = false
    private var lastRoutedExpertCount = 0
    private var allocatedCacheBytes: UInt64 = 0
    private var peakAllocatedCacheBytes: UInt64 = 0
    private var successfulCacheHits: UInt64 = 0
    private var successfulCacheMisses: UInt64 = 0
    // Read-only accounting is available without awaiting decode or retaining buffers.
    private nonisolated let cacheSummarySnapshot = Mutex<RoutedExpertCacheSummary?>(nil)

    nonisolated var currentRoutedExpertCacheSummary: RoutedExpertCacheSummary? {
        cacheSummarySnapshot.withLock { $0 }
    }

    var position: Int { committedPosition }
    var isUnusable: Bool { unusable }

    init(model: QwenOfficialSourceModel, maxContext: Int,
         expertSlotCount: Int,
         hooks: QwenOfficialSourceTransactionHooks = .none,
         useExpertCacheResidency: Bool? = nil,
         predictionCapture: QwenExpertPredictionCapture? = nil) throws {
        let architecture = model.architecture
        let context = model.context
        let useGroupedLinearPrefill = ProcessInfo.processInfo.environment[
            "TURBO_QWEN_GROUPED_LINEAR_PREFILL"] == "1"
        guard maxContext > 0, maxContext <= Int(UInt32.max),
              expertSlotCount >= architecture.expertsPerToken,
              expertSlotCount <= architecture.experts,
              expertSlotCount == model.expertCacheSlots else {
            throw QwenTextRunnerError.invalidState(detail: "source context/cache geometry")
        }
        let linearConfiguration = try QwenGatedDeltaNetConfiguration(
            hiddenSize: architecture.hiddenSize,
            keyHeadCount: architecture.linearKeyHeads,
            valueHeadCount: architecture.linearValueHeads,
            keyHeadDimension: architecture.linearKeyDimension,
            valueHeadDimension: architecture.linearValueDimension,
            convolutionWidth: architecture.convolutionWidth)
        let linearGeometry = try QwenLinearAttentionGeometry(
            convolutionWidth: architecture.convolutionWidth,
            convolutionChannelCount: linearConfiguration.convolutionChannelCount,
            valueHeadCount: architecture.linearValueHeads,
            keyHeadDimension: architecture.linearKeyDimension,
            valueHeadDimension: architecture.linearValueDimension)
        // Reserve every possible layer cache before the runner's first GPU
        // allocation. Failed initialization returns the quota through ARC.
        let reservation = try model.reserveExpertCache()
        let residencyEnabled = useExpertCacheResidency ?? (ProcessInfo.processInfo.environment[
            "TURBO_QWEN_EXPERT_CACHE_RESIDENCY"] == "1")
        let residency = residencyEnabled ? try QwenBF16CacheResidency(
            queue: context.queue, layerCount: architecture.layers, slotCount: expertSlotCount) : nil
        let expertCacheStorage = ExpertCacheStorage(reservation: reservation, residency: residency)
        let linearState = try QwenLinearAttentionState(
            device: context.device,
            linearAttentionLayerMask: architecture.fullAttentionLayerMask.map { 1 - $0 },
            geometry: linearGeometry,
            expectedLinearLayerCount: model.sourceIdentity == nil ? 1 : 30)
        let fullKV = try QwenFullAttentionKV(
            device: context.device,
            fullAttentionLayerMask: architecture.fullAttentionLayerMask,
            maxContext: maxContext,
            keyValueHeadCount: architecture.keyValueHeads,
            headDimension: architecture.headDimension,
            expectedLayerCount: architecture.layers,
            expectedFullLayerCount: model.sourceIdentity == nil ? 1 : 10)
        let attentionConfiguration = try QwenFullAttentionConfiguration(
            queryHeadCount: architecture.queryHeads,
            keyValueHeadCount: architecture.keyValueHeads,
            headDimension: architecture.headDimension,
            rotaryDimension: Int(Double(architecture.headDimension)
                * architecture.partialRotaryFactor),
            theta: Float(architecture.ropeTheta))
        var linearSteps: [Int: QwenBF16LinearStep] = [:]
        var fullSteps: [Int: QwenBF16FullAttentionStep] = [:]
        for layer in 0..<architecture.layers {
            let weights = model.layers[layer]
            if let names = weights.fullNames,
               let queryNorm = weights.queryNorm,
               let keyNorm = weights.keyNorm {
                let stepHooks: QwenBF16FullAttentionHooks
                if hooks.requiresSeparateGPUStages {
                    stepHooks = QwenBF16FullAttentionHooks { checkpoint in
                        switch checkpoint {
                        case let .afterSubmission(stage):
                            try await hooks.afterActualGPUSubmission("full.\(stage)")
                        case .beforeCommit:
                            try await hooks.afterActualGPUCompletion("full.commit")
                        case .beforeSubmission: break
                        }
                    }
                } else {
                    stepHooks = .none
                }
                fullSteps[layer] = try QwenBF16FullAttentionStep(
                    context: context, weights: weights.weights, names: names,
                    configuration: attentionConfiguration,
                    hiddenSize: architecture.hiddenSize, layer: layer, kv: fullKV,
                    queryNorm: queryNorm, keyNorm: keyNorm,
                    hooks: stepHooks)
            } else if let names = weights.linearNames,
                      let vectors = weights.linearVectors {
                let linearObserver: (@Sendable (Int, String, [Float]) -> Void)?
                if let observe = hooks.observeActivation {
                    linearObserver = { position, stage, values in
                        observe(position, layer, "linear.\(stage)", values)
                    }
                } else {
                    linearObserver = nil
                }
                let stepHooks: QwenBF16LinearHooks
                if hooks.requiresSeparateGPUStages {
                    stepHooks = QwenBF16LinearHooks(observeActivation: linearObserver) { checkpoint in
                        switch checkpoint {
                        case let .afterSubmission(stage):
                            try await hooks.afterActualGPUSubmission("linear.\(stage)")
                        case .beforeCommit:
                            try await hooks.afterActualGPUCompletion("linear.commit")
                        case .beforeSubmission: break
                        }
                    }
                } else {
                    stepHooks = .none
                }
                linearSteps[layer] = try QwenBF16LinearStep(
                    context: context, weights: weights.weights, names: names,
                    configuration: linearConfiguration, vectors: vectors,
                    layer: layer, state: linearState,
                    hooks: stepHooks, useGroupedCachedRecurrence: useGroupedLinearPrefill)
            } else {
                throw QwenTextRunnerError.stateArchitectureMismatch
            }
        }
        let moeConfiguration = try QwenMoEConfiguration(
            hiddenSize: architecture.hiddenSize,
            expertCount: architecture.experts,
            topK: architecture.expertsPerToken,
            routedIntermediateSize: architecture.routedIntermediateSize,
            sharedIntermediateSize: architecture.sharedIntermediateSize)
        self.model = model
        self.context = context
        self.predictionCapture = predictionCapture
        self.hooks = hooks
        // Internal trial only, fixed for this runner's lifetime. Legacy
        // diagnostic hooks still select the linear step's separated fallback.
        useGPULinearPreparation = ProcessInfo.processInfo.environment[
            "TURBO_QWEN_GPU_LINEAR_PREPARATION"] == "1"
        let useExpertProjectionBatch = ProcessInfo.processInfo.environment[
            "TURBO_QWEN_EXPERT_PROJECTION_BATCH"] == "1"
        self.useGroupedLinearPrefill = useGroupedLinearPrefill
        self.linearState = linearState
        self.fullKV = fullKV
        self.linearSteps = linearSteps
        self.fullSteps = fullSteps
        self.moeConfiguration = moeConfiguration
        moe = try QwenMoE(context: context, configuration: moeConfiguration,
                          bf16ProjectionBatch: useExpertProjectionBatch)
        self.expertSlotCount = expertSlotCount
        self.expertCacheStorage = expertCacheStorage
        cacheSummarySnapshot.withLock {
            $0 = RoutedExpertCacheSummary(
                configuredSlots: expertSlotCount, effectiveSlots: expertSlotCount,
                policy: model.expertCachePolicy.rawValue, allocatedBytes: 0,
                peakAllocatedBytes: 0, hits: 0, misses: 0)
        }
    }

    func cacheDiagnostics() -> QwenOfficialSourceCacheDiagnostics {
        QwenOfficialSourceCacheDiagnostics(
            slotCount: expertSlotCount, policy: model.expertCachePolicy,
            integrityPolicy: model.sourceIntegrityPolicy,
            allocatedBytes: allocatedCacheBytes,
            routedExpertCount: lastRoutedExpertCount,
            summary: routedExpertCacheSummary())
    }

    private func publishCacheSummary() {
        let summary = routedExpertCacheSummary()
        cacheSummarySnapshot.withLock { $0 = summary }
    }

    func routedExpertCacheSummary() -> RoutedExpertCacheSummary {
        RoutedExpertCacheSummary(
            configuredSlots: expertSlotCount, effectiveSlots: expertSlotCount,
            policy: model.expertCachePolicy.rawValue,
            allocatedBytes: allocatedCacheBytes,
            peakAllocatedBytes: peakAllocatedCacheBytes,
            hits: successfulCacheHits, misses: successfulCacheMisses)
    }

    func produce(token: Int32, position: Int,
                 featureRow: [Float]? = nil,
                 mropePosition: QwenMRoPEPosition? = nil) async throws -> [Float] {
        if let location = QwenCacheMapMeasurement.capture?.qwenProductionLocation(
            position: position, tokenCount: 1, forward: true) {
            return try await QwenProductionTimingMeasurement.$location.withValue(location) {
                let span = QwenProductionTimingMeasurement.span(.forward)
                defer { span?.finish() }
                return try await produceToken(token: token, position: position, featureRow: featureRow,
                    mropePosition: mropePosition, withinPrefill: false)
            }
        }
        return try await produceToken(token: token, position: position, featureRow: featureRow,
                                      mropePosition: mropePosition, withinPrefill: false)
    }

    private func produceToken(token: Int32, position: Int,
                              featureRow: [Float]?, mropePosition: QwenMRoPEPosition?,
                              withinPrefill: Bool) async throws -> [Float] {
        guard !inFlight, !unusable, !prefillInFlight || withinPrefill else {
            throw QwenTextRunnerError.operationInProgress
        }
        guard position == committedPosition else {
            throw QwenTextRunnerError.invalidPosition(expected: committedPosition, actual: position)
        }
        guard token >= 0, Int(token) < model.architecture.vocabularySize else {
            throw QwenTextRunnerError.invalidToken(id: token)
        }
        guard position < fullKV.maxContext else {
            throw QwenTextRunnerError.invalidState(detail: "source context full")
        }
        if let featureRow {
            guard token == Int32(model.visionArchitecture.imageTokenID),
                  featureRow.count == model.architecture.hiddenSize,
                  featureRow.allSatisfy(\.isFinite), mropePosition != nil else {
                throw QwenTextRunnerError.invalidState(detail: "source image row geometry")
            }
        }
        inFlight = true
        defer { inFlight = false }
        try Task.checkCancellation()
        try model.revalidateSource()
        let linearCheckpoint = try linearState.retainCheckpoint()
        let fullSnapshot = try fullKV.snapshot()
        do {
            let initial: [Float]
            if let featureRow { initial = featureRow }
            else { initial = try await embedding(token) }
            var hidden = initial
            hooks.observeActivation?(position, -1, "embedding", initial)
            let width = model.architecture.hiddenSize
            for layer in model.layers.indices {
                try Task.checkCancellation()
                let weights = model.layers[layer]
                let normalized = try qwenOfficialSourceRMSNorm(hidden, weights.inputNorm)
                hooks.observeActivation?(position, layer, "input-norm", normalized)
                let mixer: [Float]
                if let full = fullSteps[layer] {
                    mixer = try await full.append(
                        hidden: normalized, mropePosition: mropePosition,
                        mropeSections: model.architecture.mropeSections)
                } else if let linear = linearSteps[layer] {
                    mixer = try await linear.append(normalizedHidden: normalized, tokenCount: 1,
                        useGPUPreparation: useGPULinearPreparation)
                } else { throw QwenTextRunnerError.stateArchitectureMismatch }
                guard mixer.count == width else { throw QwenTextRunnerError.stateArchitectureMismatch }
                hooks.observeActivation?(position, layer, "mixer", mixer)
                hidden = zip(hidden, mixer).map(+)
                predictionCapture?.earlyResidual(position: position, layer: layer, hidden: hidden)
                hooks.observeActivation?(position, layer, "residual-after-mixer", hidden)
                let post = try qwenOfficialSourceRMSNorm(hidden, weights.postNorm)
                hooks.observeActivation?(position, layer, "post-norm", post)
                let routed = try await moeStep(input: post, layer: layer)
                guard routed.count == width else { throw QwenTextRunnerError.stateArchitectureMismatch }
                hidden = zip(hidden, routed).map(+)
                guard hidden.allSatisfy(\.isFinite) else {
                    throw QwenTextRunnerError.execution(detail: "nonfinite source hidden")
                }
                hooks.observeActivation?(position, layer, "residual-after-moe", hidden)
            }
            hooks.observeActivation?(position, -1, "final-hidden", hidden)
            let final = try qwenOfficialSourceRMSNorm(hidden, model.finalNorm)
            hooks.observeActivation?(position, -1, "final-norm", final)
            let logits = try await projection(
                model.entryWeights, model.headName, input: final,
                outputCount: model.architecture.vocabularySize, timingStage: .head)
            try Task.checkCancellation()
            try model.revalidateSource()
            guard logits.count == model.architecture.vocabularySize,
                  logits.allSatisfy(\.isFinite) else {
                throw QwenTextRunnerError.execution(detail: "nonfinite source logits")
            }
            hooks.observeRawLogits(position, token, logits)
            committedPosition += 1
            hooks.observeConsumedInput?(position, token, featureRow, mropePosition)
            return logits
        } catch {
            // All lower-level methods await submitted GPU work before throwing.
            // If restoration itself fails, permanently close this runner.
            await linearState.waitUntilIdle()
            do {
                // A transactional token failure must rewind directly to the
                // turn baseline. Rewinding only the token branches KV lineage
                // and invalidates the earlier turn snapshot.
                if let baseline = turnCheckpoint {
                    try linearState.restore(baseline.linear)
                } else {
                    try linearState.restore(linearCheckpoint)
                }
                if let checkpoint = turnCheckpoint { try restoreFullTurnCheckpoint(checkpoint) }
                else { try fullKV.restore(fullSnapshot) }
                if let baseline = turnCheckpoint {
                    committedPosition = baseline.position
                    turnCheckpoint?.full = try fullKV.snapshot()
                }
            } catch let rollbackError {
                unusable = true
                throw QwenTextRunnerError.execution(
                    detail: "source token failed: \(error); rollback failed: \(rollbackError)")
            }
            throw error
        }
    }

    func groupedPrefillDiagnostics() -> QwenSourceGroupedPrefillDiagnostics? {
        lastPrefillDiagnostics
    }

    /// Prompt-only execution. Decode retains produce's one-token contract.
    /// The optional capture explicitly observes provisional grouped work.
    func prefill(tokenIDs: [Int32], position: Int,
                 featureRowAt: (@Sendable (Int) throws -> [Float]?)? = nil,
                 mropePositions: [QwenMRoPEPosition]? = nil,
                 groupedCapture: QwenSourceGroupedPrefillCapture? = nil,
                 onProgress: @Sendable (Int, Int) async -> Void = { _, _ in }) async throws -> [Float] {
        guard !inFlight, !unusable, !prefillInFlight else {
            throw QwenTextRunnerError.operationInProgress
        }
        guard position == committedPosition else {
            throw QwenTextRunnerError.invalidPosition(expected: committedPosition, actual: position)
        }
        guard !tokenIDs.isEmpty, position >= 0,
              tokenIDs.count <= fullKV.maxContext - position,
              mropePositions == nil || mropePositions?.count == tokenIDs.count else {
            throw QwenTextRunnerError.invalidState(detail: "source prefill context/positions")
        }
        let width = model.architecture.hiddenSize
        let topK = model.architecture.expertsPerToken
        for (index, token) in tokenIDs.enumerated() {
            try Task.checkCancellation()
            guard token >= 0, Int(token) < model.architecture.vocabularySize else {
                throw QwenTextRunnerError.invalidToken(id: token)
            }
            if let feature = try featureRowAt?(index) {
                guard token == Int32(model.visionArchitecture.imageTokenID),
                      feature.count == width, feature.allSatisfy(\.isFinite),
                      mropePositions != nil else {
                    throw QwenTextRunnerError.invalidState(detail: "source prefill image row geometry")
                }
            }
        }
        let rowElements = try prefillProduct(tokenIDs.count, width)
        let routeElements = try prefillProduct(tokenIDs.count, topK)
        // Count both work-list representations, IDs, weights and final logits.
        // Prepared owners and the existing rollback baseline retain their own budgets.
        let matrixBytes = try prefillProduct(try prefillProduct(rowElements, 4), 3)
        let routeStride = MemoryLayout<Int>.stride + MemoryLayout<Float>.stride
            + 2 * MemoryLayout<QwenBF16GroupedExpertWork>.stride + MemoryLayout<Bool>.stride
        let routeBytes = try prefillProduct(routeElements, routeStride)
        let (workspaceBytes, workspaceOverflow) = matrixBytes.addingReportingOverflow(routeBytes)
        let (boundedBytes, finalOverflow) = workspaceBytes.addingReportingOverflow(
            try prefillProduct(model.architecture.vocabularySize, MemoryLayout<Float>.stride))
        // Explicit capture replaces only the legacy per-token raw/input
        // observers during this prompt. Decode keeps those hooks unchanged.
        let tokenMajor = groupedCapture == nil
            ? hooks.requiresTokenMajorPrefill : hooks.requiresOrderedPrefillStages
        let grouped = !tokenMajor && !workspaceOverflow && !finalOverflow
            && boundedBytes <= 128 * 1024 * 1024
            && rowElements <= Int(UInt32.max)
            && rowElements * MemoryLayout<Float>.stride <= context.device.maxBufferLength
        lastPrefillDiagnostics = QwenSourceGroupedPrefillDiagnostics(
            mode: grouped ? .grouped : .tokenMajor, tokenCount: tokenIDs.count)

        let ownsCheckpoint = turnCheckpoint == nil
        if ownsCheckpoint { try beginTurn() }
        prefillInFlight = true
        defer { prefillInFlight = false }
        if !grouped {
            do {
                var logits: [Float] = []
                for (index, token) in tokenIDs.enumerated() {
                    logits = try await produceToken(token: token, position: position + index,
                        featureRow: try featureRowAt?(index),
                        mropePosition: mropePositions?[index], withinPrefill: true)
                    await onProgress(index + 1, tokenIDs.count)
                }
                try Task.checkCancellation()
                if ownsCheckpoint {
                    // Release the batch gate only for this synchronous final
                    // validation. No suspension follows successful release.
                    prefillInFlight = false
                    do { try finishTurn() }
                    catch { prefillInFlight = true; throw error }
                }
                return logits
            } catch {
                // produce restores the active turn baseline and refreshes lineage.
                // Token recovery may already have failed closed. Its error
                // includes that failure, and another restore would obscure it
                // with invalidTransaction. An open runner still must restore.
                if ownsCheckpoint, !unusable {
                    do {
                        try await restoreTurnBaselineCore(withinPrefill: true)
                        turnCheckpoint = nil
                    } catch let rollbackError {
                        throw QwenTextRunnerError.execution(
                            detail: "source prefill failed: \(error); rollback failed: \(rollbackError)")
                    }
                }
                throw error
            }
        }

        inFlight = true
        defer { inFlight = false }
        do {
            try Task.checkCancellation()
            try model.revalidateSource()
            let residualRows = try buffer(elements: rowElements,
                stride: MemoryLayout<Float>.stride, label: "source prefill residual rows")
            let inputRows = try buffer(elements: rowElements,
                stride: MemoryLayout<Float>.stride, label: "source prefill expert inputs")
            let outputRows = try buffer(elements: rowElements,
                stride: MemoryLayout<Float>.stride, label: "source prefill routed outputs")
            let routingWeights = try buffer(elements: routeElements,
                stride: MemoryLayout<Float>.stride, label: "source prefill route weights")
            let residual = residualRows.contents().assumingMemoryBound(to: Float.self)
            let inputs = inputRows.contents().assumingMemoryBound(to: Float.self)
            let routeValues = routingWeights.contents().assumingMemoryBound(to: Float.self)
            let scratch = try moe.makeScratch()
            for (index, token) in tokenIDs.enumerated() {
                try Task.checkCancellation()
                let feature = try featureRowAt?(index)
                let row: [Float]
                if let feature {
                    guard token == Int32(model.visionArchitecture.imageTokenID),
                          mropePositions != nil else {
                        throw QwenTextRunnerError.invalidState(detail: "source prefill image row geometry")
                    }
                    row = feature
                } else { row = try await embedding(token) }
                guard row.count == width, row.allSatisfy(\.isFinite) else {
                    throw QwenTextRunnerError.execution(detail: "source prefill embedding row")
                }
                for column in 0..<width { residual[index * width + column] = row[column] }
                groupedCapture?.observeConsumedInput?(
                    position + index, token, feature, mropePositions?[index])
            }
            for layer in model.layers.indices {
                try Task.checkCancellation()
                let row = model.layers[layer]
                var routingExpertIDs = [Int](repeating: 0, count: routeElements)
                var workByExpert: [Int: [QwenBF16GroupedExpertWork]] = [:]
                // These rows all belong to the same previous-layer output.
                // Bound extra host/GPU scratch independently of prompt length.
                var linearBatchOutput: [Float] = []
                var linearBatchStart = 0
                var linearBatchEnd = 0
                for index in tokenIDs.indices {
                    try Task.checkCancellation()
                    let hidden = Array(UnsafeBufferPointer(
                        start: residual.advanced(by: index * width), count: width))
                    let mixer: [Float]
                    if useGroupedLinearPrefill, !hooks.requiresOrderedPrefillStages,
                       let linear = linearSteps[layer] {
                        if index >= linearBatchEnd {
                            linearBatchStart = index
                            linearBatchEnd = min(index + 16, tokenIDs.count)
                            var normalizedRows: [Float] = []
                            normalizedRows.reserveCapacity((linearBatchEnd - index) * width)
                            for tokenIndex in index..<linearBatchEnd {
                                try Task.checkCancellation()
                                let inputRow = Array(UnsafeBufferPointer(
                                    start: residual.advanced(by: tokenIndex * width), count: width))
                                normalizedRows += try qwenOfficialSourceRMSNorm(inputRow, row.inputNorm)
                            }
                            linearBatchOutput = []
                            var consumed = 0
                            // Preserve the exact initial-token source formulation.
                            // The existing outer prefill checkpoint owns rollback
                            // if this prefix succeeds but any later operation fails.
                            if position + index == 0 {
                                linearBatchOutput += try await linear.append(
                                    normalizedHidden: Array(normalizedRows.prefix(width)), tokenCount: 1,
                                    useGPUPreparation: useGPULinearPreparation)
                                consumed = 1
                                lastPrefillDiagnostics?.groupedLinearBatchSizes[layer, default: []].append(1)
                            }
                            let remaining = linearBatchEnd - index - consumed
                            if remaining > 0 {
                                linearBatchOutput += try await linear.append(
                                    normalizedHidden: Array(normalizedRows.dropFirst(consumed * width)),
                                    tokenCount: remaining, useGPUPreparation: useGPULinearPreparation)
                                lastPrefillDiagnostics?.groupedLinearBatchSizes[layer, default: []].append(remaining)
                            }
                            guard linearBatchOutput.count == (linearBatchEnd - index) * width else {
                                throw QwenTextRunnerError.stateArchitectureMismatch
                            }
                        }
                        let offset = (index - linearBatchStart) * width
                        mixer = Array(linearBatchOutput[offset..<(offset + width)])
                    } else {
                        let normalized = try qwenOfficialSourceRMSNorm(hidden, row.inputNorm)
                        if let full = fullSteps[layer] {
                            mixer = try await full.append(hidden: normalized,
                                mropePosition: mropePositions?[index],
                                mropeSections: model.architecture.mropeSections)
                        } else if let linear = linearSteps[layer] {
                            mixer = try await linear.append(normalizedHidden: normalized, tokenCount: 1,
                                useGPUPreparation: useGPULinearPreparation)
                        } else { throw QwenTextRunnerError.stateArchitectureMismatch }
                    }
                    guard mixer.count == width else {
                        throw QwenTextRunnerError.stateArchitectureMismatch
                    }
                    let afterMixer = zip(hidden, mixer).map(+)
                    let post = try qwenOfficialSourceRMSNorm(afterMixer, row.postNorm)
                    for column in 0..<width {
                        residual[index * width + column] = afterMixer[column]
                        inputs[index * width + column] = post[column]
                    }
                    let logits = try await projection(row.weights, row.routerName,
                        input: post, outputCount: model.architecture.experts)
                    let routing = try QwenMoE.route(logits: logits,
                        configuration: moeConfiguration, arithmetic: .officialSourceCPU)
                    guard let ids = routing.selectedExpertIDs.first,
                          let weights = routing.normalizedWeights.first,
                          ids.count == topK, weights.count == topK,
                          Set(ids).count == topK,
                          ids.allSatisfy({ $0 >= 0 && $0 < model.architecture.experts }),
                          weights.allSatisfy({ $0.isFinite && $0 >= 0 }),
                          abs(weights.reduce(Float(0), +) - 1) <= 0.001 else {
                        throw QwenTextRunnerError.execution(detail: "source prefill Top-8 route")
                    }
                    let ranked = logits.indices.sorted {
                        logits[$0] == logits[$1] ? $0 < $1 : logits[$0] > logits[$1]
                    }
                    let cutoff = ranked.count > topK
                        ? logits[ranked[topK - 1]] - logits[ranked[topK]] : 0
                    hooks.observeRoute(position + index, layer, logits, ids, weights, cutoff)
                    groupedCapture?.observeRoute?(position + index, layer, logits, ids, weights, cutoff)
                    for rank in 0..<topK {
                        routingExpertIDs[index * topK + rank] = ids[rank]
                        routeValues[index * topK + rank] = weights[rank]
                        workByExpert[ids[rank], default: []].append(
                            QwenBF16GroupedExpertWork(expertID: ids[rank],
                                tokenIndex: index, routeRank: rank))
                    }
                }
                let uniqueExperts = workByExpert.keys.sorted()
                // Check the complete immutable layer plan once. Each original
                // token/rank must contribute exactly once across all chunks.
                var seen = [Bool](repeating: false, count: routeElements)
                for expert in uniqueExperts {
                    for item in workByExpert[expert] ?? [] {
                        let routeIndex = item.tokenIndex * topK + item.routeRank
                        guard !seen[routeIndex], routingExpertIDs[routeIndex] == expert else {
                            throw QwenTextRunnerError.invalidState(detail: "source prefill duplicate route work")
                        }
                        seen[routeIndex] = true
                    }
                }
                guard seen.allSatisfy({ $0 }) else {
                    throw QwenTextRunnerError.invalidState(detail: "source prefill incomplete route work")
                }
                lastPrefillDiagnostics?.mappedUniqueExperts[layer] = uniqueExperts.count
                let coordinator = try expertCoordinator(layer: layer)
                var firstCommand = true
                var groupStart = 0
                while groupStart < uniqueExperts.count {
                    let groupEnd = min(groupStart + expertSlotCount, uniqueExperts.count)
                    let groupIDs = Array(uniqueExperts[groupStart..<groupEnd])
                    let groupWork = groupIDs.flatMap { workByExpert[$0] ?? [] }
                    var workStart = 0
                    while workStart < groupWork.count {
                        try Task.checkCancellation()
                        let workEnd = min(workStart + QwenMoE.maximumGroupedContributionsPerCommand,
                                          groupWork.count)
                        let work = Array(groupWork[workStart..<workEnd])
                        // Load the complete group once. Subsequent chunks can map
                        // their current subset as hits without evicting this group.
                        let requested = workStart == 0 ? groupIDs : Array(Set(work.map(\.expertID))).sorted()
                        try model.revalidateSource()
                        let measurement = QwenCacheMapMeasurement.capture.map {
                            QwenCacheMapContext(capture: $0, layer: layer,
                                                position: position, tokenCount: tokenIDs.count)
                        }
                        let lease = try await coordinator.map(expertIDs: requested, measurement: measurement)
                        successfulCacheHits += UInt64(lease.diagnostics.hits)
                        successfulCacheMisses += UInt64(lease.diagnostics.misses)
                        publishCacheSummary()
                        lastPrefillDiagnostics?.mappingHits += UInt64(lease.diagnostics.hits)
                        lastPrefillDiagnostics?.mappingMisses += UInt64(lease.diagnostics.misses)
                        lastRoutedExpertCount = topK
                        do {
                            try model.revalidateSource()
                            try await moe.submitGroupedExpertsBF16(
                                hiddenRows: inputRows, routingExpertIDs: routingExpertIDs,
                                routingWeights: routingWeights, outputRows: outputRows,
                                tokenCount: tokenIDs.count, lease: lease, work: work,
                                sharedWeights: row.weights, scratch: scratch,
                                initializeOutput: firstCommand)
                        } catch {
                            try? lease.cancel()
                            throw error
                        }
                        firstCommand = false
                        workStart = workEnd
                    }
                    groupStart = groupEnd
                }
                for index in tokenIDs.indices {
                    try Task.checkCancellation()
                    let post = Array(UnsafeBufferPointer(
                        start: inputs.advanced(by: index * width), count: width))
                    let output = outputRows.contents().assumingMemoryBound(to: Float.self)
                    let routed = Array(UnsafeBufferPointer(
                        start: output.advanced(by: index * width), count: width))
                    let sharedInput = try floats(post, label: "source prefill shared input")
                    let sharedOutput = try floats(routed, label: "source prefill shared output")
                    let command = try commandBuffer()
                    try moe.encodeSharedBF16(commandBuffer: command, hidden: sharedInput,
                        weights: row.weights, names: row.sharedNames,
                        scratch: scratch, output: sharedOutput)
                    try await settle(command, stage: "source.prefill.shared")
                    let combined = read(sharedOutput, count: width)
                    for column in 0..<width {
                        residual[index * width + column] = residual[index * width + column] + combined[column]
                    }
                    guard (0..<width).allSatisfy({ residual[index * width + $0].isFinite }) else {
                        throw QwenTextRunnerError.execution(detail: "nonfinite source prefill hidden")
                    }
                    if layer == model.layers.count - 1, index + 1 < tokenIDs.count {
                        await onProgress(index + 1, tokenIDs.count)
                    }
                }
                lastPrefillDiagnostics?.completedLayers += 1
            }
            let finalHidden = Array(UnsafeBufferPointer(
                start: residual.advanced(by: (tokenIDs.count - 1) * width), count: width))
            let final = try qwenOfficialSourceRMSNorm(finalHidden, model.finalNorm)
            let logits = try await projection(model.entryWeights, model.headName,
                input: final, outputCount: model.architecture.vocabularySize)
            guard logits.count == model.architecture.vocabularySize,
                  logits.allSatisfy(\.isFinite) else {
                throw QwenTextRunnerError.execution(detail: "nonfinite source prefill logits")
            }
            groupedCapture?.observeFinalRawLogits?(position + tokenIDs.count - 1,
                                                  tokenIDs[tokenIDs.count - 1], logits)
            await onProgress(tokenIDs.count, tokenIDs.count)
            try Task.checkCancellation()
            try model.revalidateSource()
            let expected = position + tokenIDs.count
            let full = try fullKV.snapshot()
            let linearPositionsMatch = try linearSteps.keys.allSatisfy {
                try linearState.committedPosition(layer: $0) == expected
            }
            guard full.position == expected, linearPositionsMatch else {
                throw QwenTextRunnerError.invalidState(detail: "source prefill final layer positions")
            }
            committedPosition = expected
            if ownsCheckpoint { turnCheckpoint = nil }
            return logits
        } catch {
            await linearState.waitUntilIdle()
            do {
                guard let checkpoint = turnCheckpoint else {
                    throw QwenTextRunnerError.invalidTransaction
                }
                try linearState.restore(checkpoint.linear)
                try restoreFullTurnCheckpoint(checkpoint)
                committedPosition = checkpoint.position
                turnCheckpoint?.full = try fullKV.snapshot()
                if ownsCheckpoint { turnCheckpoint = nil }
            } catch let rollbackError {
                unusable = true
                throw QwenTextRunnerError.execution(
                    detail: "source prefill failed: \(error); rollback failed: \(rollbackError)")
            }
            throw error
        }
    }

    private func prefillProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard lhs > 0, rhs > 0, !overflow else {
            throw QwenTextRunnerError.invalidState(detail: "source prefill workspace overflow")
        }
        return value
    }

    /// Exactly one turn baseline; full KV's owner/lineage stays opaque here.
    /// Individual tokens still have their own existing atomic failure recovery.
    func beginTurn() throws {
        guard !inFlight, !unusable, !prefillInFlight, turnCheckpoint == nil else {
            throw QwenTextRunnerError.operationInProgress
        }
        try model.revalidateSource()
        let linear = try linearState.clone()
        let full = try fullKV.snapshot()
        guard full.position == committedPosition,
              linear.positions.values.allSatisfy({ $0 == committedPosition }) else {
            throw QwenTextRunnerError.invalidState(detail: "source turn baseline position")
        }
        turnCheckpoint = TurnCheckpoint(
            linear: linear, full: full, position: committedPosition)
    }

    /// Restores the turn's pre-write state, then refreshes KV lineage so that
    /// a second failure during suffix replay can still roll back the same turn.
    func restoreTurnBaseline() async throws {
        try await restoreTurnBaselineCore(withinPrefill: false)
    }

    private func restoreTurnBaselineCore(withinPrefill: Bool) async throws {
        guard !inFlight, !unusable, !prefillInFlight || withinPrefill,
              let checkpoint = turnCheckpoint else {
            throw QwenTextRunnerError.invalidTransaction
        }
        await linearState.waitUntilIdle()
        do {
            try hooks.beforeRollbackRestore()
            try linearState.restore(checkpoint.linear)
            try restoreFullTurnCheckpoint(checkpoint)
            committedPosition = checkpoint.position
            let refreshed = try fullKV.snapshot()
            turnCheckpoint?.full = refreshed
        } catch {
            unusable = true
            throw error
        }
    }

    func rollbackTurn() async throws {
        try await restoreTurnBaseline()
        fullKV.discardReplacementCheckpoint()
        turnCheckpoint = nil
    }

    /// The final synchronous validation runs after every suspension and before
    /// the only irreversible checkpoint release. If it throws, the checkpoint
    /// remains available to rollback; no await or throw follows its success.
    func finishTurn(validateCompanion: @Sendable () throws -> Void = {}) throws {
        guard !inFlight, !unusable, !prefillInFlight, turnCheckpoint != nil else {
            throw QwenTextRunnerError.invalidTransaction
        }
        try model.revalidateSource()
        try validateCompanion()
        fullKV.discardReplacementCheckpoint()
        turnCheckpoint = nil
    }

    private func restoreFullTurnCheckpoint(_ checkpoint: TurnCheckpoint) throws {
        if let replacement = checkpoint.replacementFull {
            try fullKV.restoreReplacementCheckpoint(replacement)
        } else { try fullKV.restore(checkpoint.full) }
    }

    func beginCheckpointReplacement() throws {
        guard !inFlight, !unusable, !prefillInFlight,
              var checkpoint = turnCheckpoint, checkpoint.replacementFull == nil else {
            throw QwenTextRunnerError.invalidTransaction
        }
        try model.revalidateSource()
        checkpoint.replacementFull = try fullKV.retainReplacementCheckpoint()
        turnCheckpoint = checkpoint
        try linearState.reset()
        try fullKV.reset()
        committedPosition = 0
    }

    func resetConversation() throws {
        guard !inFlight, !unusable, !prefillInFlight, turnCheckpoint == nil else {
            throw QwenTextRunnerError.operationInProgress
        }
        try model.revalidateSource()
        do {
            try linearState.reset()
            try fullKV.reset()
            committedPosition = 0
        } catch {
            unusable = true
            throw error
        }
    }

    /// Diagnostic copies of committed FP32 state, never the BF16 weights or
    /// speculative KV tail. Shared Metal buffers are idle at this actor gate.
    func diagnosticSnapshot() throws -> QwenOfficialSourceRunnerDiagnosticSnapshot {
        guard !inFlight, !unusable, !prefillInFlight else {
            throw QwenTextRunnerError.operationInProgress
        }
        let linear = try linearState.clone()
        let full = try fullKV.snapshot()
        guard full.position == committedPosition else {
            throw QwenTextRunnerError.invalidState(detail: "source diagnostic cursor")
        }
        var keys: [Int: [Float]] = [:]
        var values: [Int: [Float]] = [:]
        var positions: [Int: Int] = [:]
        for layer in fullKV.fullLayerIndices {
            let view = try fullKV.view(layer: layer)
            let (bytes, overflow) = view.validTokenCount.multipliedReportingOverflow(
                by: view.strideBytes)
            guard !overflow, bytes <= view.key.length,
                  bytes <= view.value.length,
                  bytes.isMultiple(of: MemoryLayout<Float>.stride) else {
                throw QwenTextRunnerError.invalidState(detail: "source KV diagnostic bounds")
            }
            let count = bytes / MemoryLayout<Float>.stride
            keys[layer] = read(view.key, count: count)
            values[layer] = read(view.value, count: count)
            positions[layer] = view.validTokenCount
        }
        return QwenOfficialSourceRunnerDiagnosticSnapshot(
            position: committedPosition, linear: linear, fullKVPositions: positions,
            committedKeys: keys, committedValues: values)
    }

    private func embedding(_ token: Int32) async throws -> [Float] {
        let ids = try buffer(elements: 1, stride: MemoryLayout<UInt32>.stride,
                             label: "source token")
        ids.contents().storeBytes(of: UInt32(token), as: UInt32.self)
        let output = try buffer(elements: model.architecture.hiddenSize,
                                stride: MemoryLayout<Float>.stride,
                                label: "source embedding")
        let command = try commandBuffer()
        // These IDs are owned only by this call and remain immutable through
        // settlement, allowing the encoder to bind only their weight chunks.
        try model.entryWeights.encodeEmbeddingFromImmutableIDs(
            commandBuffer: command, tensorName: model.embeddingName,
            tokenIDs: ids, tokenCount: 1, output: output)
        try await settle(command, stage: "source.embedding")
        return read(output, count: model.architecture.hiddenSize)
    }

    private func projection(_ weights: QwenBF16Weights, _ name: String,
                            input: [Float], outputCount: Int,
                            timingStage: QwenProductionStage = .router) async throws -> [Float] {
        let inputBuffer = try floats(input, label: "source projection input")
        let outputBuffer = try buffer(elements: outputCount,
                                      stride: MemoryLayout<Float>.stride,
                                      label: "source projection output")
        let command = try commandBuffer()
        try weights.encodeProjection(commandBuffer: command, tensorName: name,
                                     input: inputBuffer, tokenCount: 1,
                                     output: outputBuffer)
        try await settle(command, stage: "source.projection", timingStage: timingStage)
        return read(outputBuffer, count: outputCount)
    }

    // Hint-only post-decode operation. Original model operations keep all checks.
    func diagnosticCPUExpertCandidates(hidden: [Float], layer: Int) throws -> [Int] {
        guard predictionCapture?.isDrained() == true, !inFlight, !prefillInFlight, !unusable,
              model.layers.indices.contains(layer), layer > 0 else {
            throw QwenTextRunnerError.invalidState(detail: "CPU hint diagnostic state")
        }
        try Task.checkCancellation()
        let row = model.layers[layer]
        return try QwenCPUExpertHint.candidates(hidden: hidden, postNorm: row.postNorm,
            weights: row.weights, routerName: row.routerName,
            experts: model.architecture.experts, columns: model.architecture.hiddenSize)
    }

    private func moeStep(input: [Float], layer: Int) async throws -> [Float] {
        if let location = QwenProductionTimingMeasurement.location, location.layer != layer {
            return try await QwenProductionTimingMeasurement.$location.withValue(location.atLayer(layer)) {
                try await moeStep(input: input, layer: layer)
            }
        }
        let row = model.layers[layer]
        let routeLogits = try await projection(row.weights, row.routerName,
            input: input, outputCount: model.architecture.experts)
        let routing = try QwenMoE.route(logits: routeLogits,
                                       configuration: moeConfiguration,
                                       arithmetic: .officialSourceCPU)
        guard let experts = routing.selectedExpertIDs.first,
              let routeWeights = routing.normalizedWeights.first,
              experts.count == model.architecture.expertsPerToken,
              routeWeights.count == model.architecture.expertsPerToken else {
            throw QwenTextRunnerError.execution(detail: "source Top-8 routing invalid")
        }
        let ranked = routeLogits.indices.sorted {
            routeLogits[$0] == routeLogits[$1]
                ? $0 < $1 : routeLogits[$0] > routeLogits[$1]
        }
        let cutoff = ranked.count > model.architecture.expertsPerToken
            ? routeLogits[ranked[model.architecture.expertsPerToken - 1]]
                - routeLogits[ranked[model.architecture.expertsPerToken]]
            : 0
        hooks.observeRoute(committedPosition, layer, routeLogits,
                           experts, routeWeights, cutoff)
        try model.revalidateSource()
        let coordinator = try expertCoordinator(layer: layer)
        lastRoutedExpertCount = experts.count
        let measurement = QwenCacheMapMeasurement.capture.map {
            QwenCacheMapContext(capture: $0, layer: layer, position: committedPosition, tokenCount: 1)
        }
        let mapSpan = QwenProductionTimingMeasurement.span(.expertMap)
        let lease: QwenBF16ExpertLease
        let predictionMap = predictionCapture.flatMap { capture in
            capture.includes(committedPosition)
                ? QwenExpertPredictionMapContext(capture: capture, position: committedPosition, layer: layer) : nil
        }
        predictionCapture?.mapStart(position: committedPosition, layer: layer)
        do { lease = try await coordinator.map(expertIDs: experts, measurement: measurement,
                                               prediction: predictionMap) }
        catch { mapSpan?.finish(); throw error }
        mapSpan?.finish()
        predictionCapture?.mapEnd(position: committedPosition, layer: layer)
        // Mapping has succeeded. Later GPU failure/cancellation cannot undo
        // these completed cache reads, even when the token is rolled back.
        successfulCacheHits += UInt64(lease.diagnostics.hits)
        successfulCacheMisses += UInt64(lease.diagnostics.misses)
        publishCacheSummary()
        do {
            // Protected reads validate their retained shard before and after
            // every slice. Recheck the complete source once after mapping so
            // changes to other receipt files reject this layer before GPU use.
            // Keep this inside the lease cleanup scope if validation throws.
            try model.revalidateSource()
            let hiddenBuffer = try floats(input, label: "source MoE input")
            let routingBuffer = try floats(routeWeights, label: "source MoE routing")
            let output = try buffer(elements: model.architecture.hiddenSize,
                                    stride: MemoryLayout<Float>.stride,
                                    label: "source MoE output")
            let scratch = try moe.makeScratch()
            let timing = QwenProductionTimingMeasurement.command(.moe)
            let command = try moe.submitExpertsBF16(
                hidden: hiddenBuffer, lease: lease, routingWeights: routingBuffer,
                sharedWeights: row.weights, sharedNames: row.sharedNames,
                scratch: scratch, output: output, timing: timing)
            try await settleSubmitted(command, stage: "source.moe", timing: timing)
            return read(output, count: model.architecture.hiddenSize)
        } catch {
            try? lease.cancel()
            throw error
        }
    }

    private func expertCoordinator(layer: Int) throws -> QwenBF16ExpertMappingCoordinator {
        if let retained = expertCacheStorage.coordinators[layer] { return retained }
        guard model.layers.indices.contains(layer) else {
            throw QwenExpertMappingError.invalidLayer(layer)
        }
        let bytes = model.expertCacheBytesPerLayer
        let (nextAllocated, overflow) = allocatedCacheBytes.addingReportingOverflow(bytes)
        guard !overflow, nextAllocated <= expertCacheStorage.reservation.bytes else {
            throw QwenBF16ExpertCacheError.budgetExceeded
        }
        let hooks = self.hooks
        let reservation = expertCacheStorage.reservation
        let coordinator = try QwenBF16ExpertMappingCoordinator(
            source: model.source, names: model.layers[layer].routedNames, layer: layer,
            configuration: moeConfiguration, device: context.device,
            slotCount: expertSlotCount, residencyBudget: bytes,
            cachePolicy: model.expertCachePolicy,
            readHooks: QwenBF16ExpertReadHooks { checkpoint in
                // A submitted lease can outlive the runner. Retain its full
                // quota until this coordinator's last owner releases it.
                try withExtendedLifetime(reservation) {
                    guard case let .beforeProtectedRead(expert, stream) = checkpoint else { return }
                    try hooks.beforeProtectedExpertRead(layer, expert, stream)
                }
            }, cacheResidency: expertCacheStorage.residency)
        guard coordinator.allocatedCacheBytes == bytes else {
            throw QwenBF16ExpertCacheError.invalidGeometry
        }
        expertCacheStorage.coordinators[layer] = coordinator
        allocatedCacheBytes = nextAllocated
        peakAllocatedCacheBytes = max(peakAllocatedCacheBytes, allocatedCacheBytes)
        // Allocation is real even if the subsequent map fails.
        publishCacheSummary()
        return coordinator
    }

    private func floats(_ values: [Float], label: String) throws -> MTLBuffer {
        let result = try buffer(elements: values.count,
                                stride: MemoryLayout<Float>.stride, label: label)
        values.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                result.contents().copyMemory(from: base, byteCount: raw.count)
            }
        }
        return result
    }

    private func buffer(elements: Int, stride: Int, label: String) throws -> MTLBuffer {
        let (bytes, overflow) = elements.multipliedReportingOverflow(by: stride)
        guard elements > 0, !overflow, bytes <= context.device.maxBufferLength,
              let result = context.device.makeBuffer(length: bytes,
                                                      options: .storageModeShared) else {
            throw QwenTextRunnerError.execution(detail: "source allocation: \(label)")
        }
        result.label = label
        return result
    }

    private func commandBuffer() throws -> MTLCommandBuffer {
        guard let command = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.execution(detail: "source command buffer unavailable")
        }
        return command
    }

    private func settle(_ command: MTLCommandBuffer, stage: String,
                        timingStage: QwenProductionStage = .embedding) async throws {
        try Task.checkCancellation()
        let timing = QwenProductionTimingMeasurement.command(timingStage)
        timing?.willCommit(command)
        command.commit()
        timing?.didCommit()
        try await settleSubmitted(command, stage: stage, timing: timing)
    }

    private func settleSubmitted(_ command: MTLCommandBuffer,
                                 stage: String, timing: QwenProductionCommand? = nil) async throws {
        var hookError: Error?
        do { try await hooks.afterActualGPUSubmission(stage) }
        catch { hookError = error }
        timing?.willWait()
        await withTaskCancellationHandler {
            await command.completed()
        } onCancel: {
            // GPU work cannot be cancelled. Keep all buffers alive until done.
        }
        timing?.resumed(command)
        // This observer reports terminal GPU settlement, not successful GPU
        // execution. Run it even if the earlier submission gate failed; its
        // own failure must not hide that original error or the Metal status.
        var completionHookError: Error?
        do { try await hooks.afterActualGPUCompletion(stage) }
        catch { completionHookError = error }
        if let hookError { throw hookError }
        guard command.status == .completed, command.error == nil else {
            throw QwenTextRunnerError.gpuExecution(
                stage: stage, detail: command.error?.localizedDescription
                    ?? "status \(command.status.rawValue)")
        }
        if let completionHookError { throw completionHookError }
        try Task.checkCancellation()
    }

    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
