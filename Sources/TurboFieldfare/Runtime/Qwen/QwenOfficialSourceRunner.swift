import Foundation
import Metal

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
    private let hooks: QwenOfficialSourceTransactionHooks
    /// Only this actor mutates the dictionary. A GPU lease can independently
    /// retain a coordinator, whose hooks retain the same quota reservation.
    private final class ExpertCacheStorage: @unchecked Sendable {
        let reservation: QwenOfficialSourceModel.ExpertCacheReservation
        var coordinators: [Int: QwenBF16ExpertMappingCoordinator] = [:]

        init(reservation: QwenOfficialSourceModel.ExpertCacheReservation) {
            self.reservation = reservation
        }

        deinit { coordinators.removeAll() }
    }
    private let expertCacheStorage: ExpertCacheStorage
    private struct TurnCheckpoint {
        let linear: QwenLinearAttentionSnapshot
        var full: QwenFullAttentionKVSnapshot
        let position: Int
    }
    private var turnCheckpoint: TurnCheckpoint?
    private var committedPosition = 0
    private var inFlight = false
    private var unusable = false
    private var lastRoutedExpertCount = 0
    private var allocatedCacheBytes: UInt64 = 0
    private var peakAllocatedCacheBytes: UInt64 = 0
    private var successfulCacheHits: UInt64 = 0
    private var successfulCacheMisses: UInt64 = 0

    var position: Int { committedPosition }
    var isUnusable: Bool { unusable }

    init(model: QwenOfficialSourceModel, maxContext: Int,
         expertSlotCount: Int,
         hooks: QwenOfficialSourceTransactionHooks = .none) throws {
        let architecture = model.architecture
        let context = model.context
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
        let expertCacheStorage = ExpertCacheStorage(
            reservation: try model.reserveExpertCache())
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
                    hooks: stepHooks)
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
        self.hooks = hooks
        self.linearState = linearState
        self.fullKV = fullKV
        self.linearSteps = linearSteps
        self.fullSteps = fullSteps
        self.moeConfiguration = moeConfiguration
        moe = try QwenMoE(context: context, configuration: moeConfiguration)
        self.expertSlotCount = expertSlotCount
        self.expertCacheStorage = expertCacheStorage
    }

    func cacheDiagnostics() -> QwenOfficialSourceCacheDiagnostics {
        QwenOfficialSourceCacheDiagnostics(
            slotCount: expertSlotCount, policy: model.expertCachePolicy,
            integrityPolicy: model.sourceIntegrityPolicy,
            allocatedBytes: allocatedCacheBytes,
            routedExpertCount: lastRoutedExpertCount,
            summary: routedExpertCacheSummary())
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
        guard !inFlight, !unusable else { throw QwenTextRunnerError.operationInProgress }
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
        let linearSnapshot = try linearState.clone()
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
                    mixer = try await linear.append(normalizedHidden: normalized, tokenCount: 1)
                } else { throw QwenTextRunnerError.stateArchitectureMismatch }
                guard mixer.count == width else { throw QwenTextRunnerError.stateArchitectureMismatch }
                hooks.observeActivation?(position, layer, "mixer", mixer)
                hidden = zip(hidden, mixer).map(+)
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
                outputCount: model.architecture.vocabularySize)
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
                try linearState.restore(turnCheckpoint?.linear ?? linearSnapshot)
                try fullKV.restore(turnCheckpoint?.full ?? fullSnapshot)
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

    /// Exactly one turn baseline; full KV's owner/lineage stays opaque here.
    /// Individual tokens still have their own existing atomic failure recovery.
    func beginTurn() throws {
        guard !inFlight, !unusable, turnCheckpoint == nil else {
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
        guard !inFlight, !unusable, let checkpoint = turnCheckpoint else {
            throw QwenTextRunnerError.invalidTransaction
        }
        await linearState.waitUntilIdle()
        do {
            try hooks.beforeRollbackRestore()
            try linearState.restore(checkpoint.linear)
            try fullKV.restore(checkpoint.full)
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
        turnCheckpoint = nil
    }

    /// The final synchronous validation runs after every suspension and before
    /// the only irreversible checkpoint release. If it throws, the checkpoint
    /// remains available to rollback; no await or throw follows its success.
    func finishTurn(validateCompanion: @Sendable () throws -> Void = {}) throws {
        guard !inFlight, !unusable, turnCheckpoint != nil else {
            throw QwenTextRunnerError.invalidTransaction
        }
        try model.revalidateSource()
        try validateCompanion()
        turnCheckpoint = nil
    }

    func resetConversation() throws {
        guard !inFlight, !unusable, turnCheckpoint == nil else {
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
        guard !inFlight, !unusable else {
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
        try model.entryWeights.encodeEmbedding(
            commandBuffer: command, tensorName: model.embeddingName,
            tokenIDs: ids, tokenCount: 1, output: output)
        try await settle(command, stage: "source.embedding")
        return read(output, count: model.architecture.hiddenSize)
    }

    private func projection(_ weights: QwenBF16Weights, _ name: String,
                            input: [Float], outputCount: Int) async throws -> [Float] {
        let inputBuffer = try floats(input, label: "source projection input")
        let outputBuffer = try buffer(elements: outputCount,
                                      stride: MemoryLayout<Float>.stride,
                                      label: "source projection output")
        let command = try commandBuffer()
        try weights.encodeProjection(commandBuffer: command, tensorName: name,
                                     input: inputBuffer, tokenCount: 1,
                                     output: outputBuffer)
        try await settle(command, stage: "source.projection")
        return read(outputBuffer, count: outputCount)
    }

    private func moeStep(input: [Float], layer: Int) async throws -> [Float] {
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
        let lease = try await coordinator.map(expertIDs: experts)
        // Mapping has succeeded. Later GPU failure/cancellation cannot undo
        // these completed cache reads, even when the token is rolled back.
        successfulCacheHits += UInt64(lease.diagnostics.hits)
        successfulCacheMisses += UInt64(lease.diagnostics.misses)
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
            let command = try moe.submitExpertsBF16(
                hidden: hiddenBuffer, lease: lease, routingWeights: routingBuffer,
                sharedWeights: row.weights, sharedNames: row.sharedNames,
                scratch: scratch, output: output)
            try await settleSubmitted(command, stage: "source.moe")
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
            })
        guard coordinator.allocatedCacheBytes == bytes else {
            throw QwenBF16ExpertCacheError.invalidGeometry
        }
        expertCacheStorage.coordinators[layer] = coordinator
        allocatedCacheBytes = nextAllocated
        peakAllocatedCacheBytes = max(peakAllocatedCacheBytes, allocatedCacheBytes)
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

    private func settle(_ command: MTLCommandBuffer, stage: String) async throws {
        try Task.checkCancellation()
        command.commit()
        try await settleSubmitted(command, stage: stage)
    }

    private func settleSubmitted(_ command: MTLCommandBuffer,
                                 stage: String) async throws {
        var hookError: Error?
        do { try await hooks.afterActualGPUSubmission(stage) }
        catch { hookError = error }
        await withTaskCancellationHandler {
            await command.completed()
        } onCancel: {
            // GPU work cannot be cancelled. Keep all buffers alive until done.
        }
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
