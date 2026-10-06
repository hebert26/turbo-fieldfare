import Darwin
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

@Suite struct QwenBF16TextRunnerOracleTests {
    @Test func independentOracleUsesZeroCenteredRMSGainAndDistinguishesPlainGainBug() throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }

        var expectedOracle = source.oracle()
        let expected = expectedOracle.run(tokens: QwenBF16TextRunnerFixture.inputTokens)
        var plainGainMutation = QwenBF16TextRunnerFixture.ReferenceMutation()
        plainGainMutation.useIncorrectPlainRMSGain = true
        var plainGainOracle = source.oracle()
        let wrongPlainGain = plainGainOracle.run(
            tokens: QwenBF16TextRunnerFixture.inputTokens,
            mutation: plainGainMutation)

        #expect(expected.count == QwenBF16TextRunnerFixture.inputTokens.count)
        #expect(wrongPlainGain.count == expected.count)
        for index in expected.indices {
            #expect(expected[index].logits.count == QwenBF16TextRunnerFixture.vocabularySize)
            #expect(expected[index].logits.allSatisfy { (value: Float) in value.isFinite })
            #expect(QwenBF16TextRunnerFixture.maxDifference(
                expected[index].logits, wrongPlainGain[index].logits)
                > QwenBF16TextRunnerFixture.negativeControlMinimum,
                "plain stored-weight RMSNorm must be observably wrong")
            #expect(expected[index].layers.count == 2)
            for layer in expected[index].layers {
                #expect(layer.selectedExperts.count == QwenBF16TextRunnerFixture.topK)
                #expect(Set(layer.selectedExperts).count == QwenBF16TextRunnerFixture.topK)
                #expect(layer.routerCutoffGap > QwenBF16TextRunnerFixture.routerCutoffMinimum)
            }
        }

        let controls: [(String, QwenBF16TextRunnerFixture.ReferenceMutation)] = [
            ("full attention", .init(suppressFullAttention: true)),
            ("linear attention", .init(suppressLinearAttention: true)),
            ("shared experts", .init(suppressSharedExpert: true)),
            ("routed experts", .init(suppressRoutedExperts: true)),
            ("earlier full-attention keys", .init(omitEarlierFullAttentionKeys: true)),
            ("linear recurrent state", .init(resetLinearStateEachToken: true)),
            ("router cutoff", .init(excludeHighestRouterExpert: true)),
        ]
        for (label, mutation) in controls {
            var changedOracle = source.oracle()
            let changed = changedOracle.run(
                tokens: QwenBF16TextRunnerFixture.inputTokens,
                mutation: mutation)
            let delta = QwenBF16TextRunnerFixture.maxDifference(
                expected.last!.logits, changed.last!.logits)
            #expect(delta > QwenBF16TextRunnerFixture.negativeControlMinimum,
                    "fixture does not distinguish \(label)")
        }
    }

    @Test func perValueToleranceRejectsOutOfLimitAndNonfiniteControls() {
        let absolute = QwenBF16TextRunnerFixture.absoluteTolerance
        let relative = QwenBF16TextRunnerFixture.relativeTolerance
        #expect(qwenValuesWithinTolerance([0.25], [0.25], absolute: absolute, relative: relative))
        #expect(!qwenValuesWithinTolerance(
            [1, 0.25003], [1, 0.25], absolute: absolute, relative: relative),
            "low-magnitude values use their own expected-value tolerance")
        #expect(!qwenValuesWithinTolerance(
            [.infinity], [0], absolute: absolute, relative: relative))
        #expect(!qwenValuesWithinTolerance(
            [0], [.nan], absolute: absolute, relative: relative))
        #expect(!qwenValuesWithinTolerance(
            [1], [1, 2], absolute: absolute, relative: relative))
    }
}

@Suite(.serialized) struct QwenBF16TextRunnerTests {
    @Test func sourceRunnerMatchesSequentialCPUOracleAndPreservesPositionOnRejects() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        #expect(model.sourceIdentity == nil)
        #expect(model.residentWeightBytes == source.expectedResidentBytes)
        #expect(model.expertCacheBytesPerLayer
            == QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
                expertSlotCount: QwenBF16TextRunnerFixture.topK, layerCount: 1))
        #expect(model.expertCacheReservationBytes
            == QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
                expertSlotCount: QwenBF16TextRunnerFixture.topK))
        #expect(source.totalResidencyBudget()
            == model.residentWeightBytes + model.expertCacheReservationBytes)
        #expect(model.architecture.layers == 2)
        #expect(model.architecture.vocabularySize == QwenBF16TextRunnerFixture.vocabularySize)

        let runner = try model.makeRunner(
            context: context, expertSlotCount: QwenBF16TextRunnerFixture.topK, maxContext: 3)
        let residentBufferIdentities = residentQwenBufferIdentities(model)
        var expectedOracle = source.oracle()
        let expected = expectedOracle.run(tokens: QwenBF16TextRunnerFixture.inputTokens)
        let initialPosition = await runner.position
        #expect(initialPosition == 0)
        let invalidPositionError = await capturedAsyncError {
            try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[0], position: 1)
        }
        let rejectedWrongPosition = matchesQwenTextRunnerError(invalidPositionError) { error in
            guard case let .invalidPosition(expected, actual) = error else { return false }
            return expected == 0 && actual == 1
        }
        #expect(rejectedWrongPosition, "wrong-position submission must be rejected")
        let invalidTokenError = await capturedAsyncError {
            try await runner.produce(token: -1, position: 0)
        }
        let rejectedInvalidToken = matchesQwenTextRunnerError(invalidTokenError) { error in
            guard case let .invalidToken(id) = error else { return false }
            return id == -1
        }
        #expect(rejectedInvalidToken, "out-of-range token must be rejected")
        let positionAfterRejections = await runner.position
        #expect(positionAfterRejections == 0)

        for (position, token) in QwenBF16TextRunnerFixture.inputTokens.enumerated() {
            let logits = try await runner.produce(token: token, position: position)
            #expect(logits.count == QwenBF16TextRunnerFixture.vocabularySize)
            #expect(logits.allSatisfy { (value: Float) in value.isFinite })
            #expect(greedyToken(logits) == expected[position].predictedToken)
            assertQwenClose(
                logits, expected[position].logits,
                absolute: QwenBF16TextRunnerFixture.absoluteTolerance,
                relative: QwenBF16TextRunnerFixture.relativeTolerance,
                label: "source token \(position) full-vocabulary FP32 logits")
            let positionAfterSubmission = await runner.position
            #expect(positionAfterSubmission == position + 1)
        }
        let stalePositionError = await capturedAsyncError {
            try await runner.produce(token: QwenBF16TextRunnerFixture.inputTokens[0], position: 1)
        }
        let rejectedStalePosition = matchesQwenTextRunnerError(stalePositionError) { error in
            guard case let .invalidPosition(expected, actual) = error else { return false }
            return expected == 2 && actual == 1
        }
        #expect(rejectedStalePosition, "stale position must be rejected")
        let finalPosition = await runner.position
        #expect(finalPosition == QwenBF16TextRunnerFixture.inputTokens.count)
        #expect(residentQwenBufferIdentities(model) == residentBufferIdentities,
                "sequential submissions must retain the same immutable resident BF16 buffers")
    }

    @Test func residentBF16VectorsAndPairedExpertCacheBytesAreAccountedSeparately() throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        #expect(model.sourceIdentity == nil)

        let chunks = model.entryWeights.inspectedChunks
            + model.layers.flatMap { $0.weights.inspectedChunks }
        #expect(chunks.count == QwenBF16TextRunnerFixture.residentMatrixNames.count)
        #expect(Set(chunks.map(\.name)) == QwenBF16TextRunnerFixture.residentMatrixNames)
        #expect(chunks.allSatisfy { $0.buffer.device === context.device })
        let actualBF16Bytes = chunks.reduce(UInt64(0)) { $0 + UInt64($1.buffer.length) }
        #expect(actualBF16Bytes == source.expectedResidentBF16Bytes)
        #expect(source.expectedResidentBF16Bytes == 1_376)

        let vectorElements = decodedVectorElementCount(model)
        #expect(vectorElements == QwenBF16TextRunnerFixture.expectedDecodedVectorElements)
        #expect(source.expectedFP32VectorBytes == UInt64(vectorElements * MemoryLayout<Float>.stride))
        #expect(source.expectedFP32VectorBytes == 304)
        #expect(source.expectedResidentBytes == actualBF16Bytes + source.expectedFP32VectorBytes)
        #expect(source.expectedResidentBytes == 1_680)

        let gateUpName = QwenBF16TextRunnerFixture.layer0Prefix + "mlp.experts.gate_up_proj"
        let downName = QwenBF16TextRunnerFixture.layer0Prefix + "mlp.experts.down_proj"
        let handle = try OfficialSourceHandle(registrationURL: source.registrationURL)
        let names = QwenBF16RoutedSourceNames(
            gateUpShardName: try #require(source.tensorToShard[gateUpName]),
            gateUpTensorName: gateUpName,
            downShardName: try #require(source.tensorToShard[downName]),
            downTensorName: downName)
        let pairBytes = UInt64(
            (2 * QwenBF16TextRunnerFixture.routedIntermediateSize
                * QwenBF16TextRunnerFixture.hiddenSize
                + QwenBF16TextRunnerFixture.hiddenSize
                    * QwenBF16TextRunnerFixture.routedIntermediateSize)
                * MemoryLayout<UInt16>.stride)
        let expectedPairBytes = QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
            expertSlotCount: 1, layerCount: 1)
        let expectedLayerCacheBytes = QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
            expertSlotCount: QwenBF16TextRunnerFixture.topK, layerCount: 1)
        let expectedAggregateCacheBytes = QwenBF16TextRunnerFixture.expectedExpertCacheBytes(
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        #expect(pairBytes == expectedPairBytes)
        #expect(expectedLayerCacheBytes
            == pairBytes * UInt64(QwenBF16TextRunnerFixture.topK))
        #expect(expectedAggregateCacheBytes
            == expectedLayerCacheBytes * UInt64(QwenBF16TextRunnerFixture.layerCount))
        let pairedCache = try QwenBF16PairedExpertCache(
            source: handle, names: names,
            expertCount: QwenBF16TextRunnerFixture.expertCount,
            hiddenSize: QwenBF16TextRunnerFixture.hiddenSize,
            intermediateSize: QwenBF16TextRunnerFixture.routedIntermediateSize,
            device: context.device,
            slotCount: QwenBF16TextRunnerFixture.topK,
            residencyBudget: pairBytes * UInt64(QwenBF16TextRunnerFixture.topK))
        #expect(pairedCache.gateUpBytes == 64)
        #expect(pairedCache.downBytes == 32)
        #expect(pairedCache.allocatedCacheBytes == 768)
        #expect(pairedCache.allocatedCacheBytes == expectedLayerCacheBytes)
        #expect(source.expectedResidentBytes + pairedCache.allocatedCacheBytes == 2_448)
    }

    @Test func productionSamplerConsumesExactFP16PublicationOfSourceRunnerLogits() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let runner = try model.makeRunner(
            context: context, expertSlotCount: QwenBF16TextRunnerFixture.topK, maxContext: 3)
        var expectedOracle = source.oracle()
        let expected = expectedOracle.run(tokens: QwenBF16TextRunnerFixture.inputTokens)
        var rawLogits: [Float] = []
        for (position, token) in QwenBF16TextRunnerFixture.inputTokens.enumerated() {
            rawLogits = try await runner.produce(token: token, position: position)
            assertQwenClose(
                rawLogits, expected[position].logits,
                absolute: QwenBF16TextRunnerFixture.absoluteTolerance,
                relative: QwenBF16TextRunnerFixture.relativeTolerance,
                label: "sampler-boundary source logits at token \(position)")
        }

        let vocabularySize = QwenBF16TextRunnerFixture.vocabularySize
        let scratch = try RawCompletionScratch(context: context, vocab: vocabularySize)
        try QwenSourceSamplerBoundary.publishFP32Logits(
            rawLogits, into: scratch.logits, vocabularySize: vocabularySize)
        let published = scratch.logits.contents().assumingMemoryBound(to: Float16.self)
        let expectedHalf = rawLogits.map(Float16.init)
        for index in 0..<vocabularySize {
            #expect(published[index] == expectedHalf[index], "FP32-to-FP16 value \(index)")
            #expect(published[index].isFinite)
        }

        var config = GenerationConfig(
            maxNewTokens: 1, temperature: 0, topK: nil, topP: nil,
            repetitionPenalty: 1, seed: 0, stopStrings: [], extraStopTokens: [])
        config.logitTransform = .raw
        guard let command = context.queue.makeCommandBuffer() else {
            Issue.record("unable to allocate the production sampler command buffer")
            return
        }
        let path = scratch.sampler.sample(
            commandBuffer: command, logits: scratch.logits, probs: scratch.probs,
            history: QwenBF16TextRunnerFixture.inputTokens, config: config,
            position: 0, outToken: scratch.outToken)
        #expect(path == .greedyGPU)
        command.commit()
        await command.completed()
        #expect(command.status == .completed)
        #expect(command.error == nil)
        let selected = Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
        #expect(selected == fp16RawSamplerGreedyToken(expected.last!.logits))
    }

    @Test func FP16BoundaryRejectsOverflowWithoutPartiallyPublishing() throws {
        let context = try MetalContext()
        let vocabularySize = QwenBF16TextRunnerFixture.vocabularySize
        let scratch = try RawCompletionScratch(context: context, vocab: vocabularySize)
        let sentinel = Float16(bitPattern: 0x3555)
        let destination = scratch.logits.contents().assumingMemoryBound(to: Float16.self)
        for index in 0..<vocabularySize { destination[index] = sentinel }

        let nonfinitePublicationError = capturedError {
            try QwenSourceSamplerBoundary.publishFP32Logits(
                [0, 1, .infinity, 3, 4], into: scratch.logits,
                vocabularySize: vocabularySize)
        }
        let rejectsNonfinitePublication = matchesQwenTextRunnerError(nonfinitePublicationError) { error in
            guard case let .execution(detail) = error else { return false }
            return detail == "nonfinite raw source logit"
        }
        #expect(rejectsNonfinitePublication)
        for index in 0..<vocabularySize {
            #expect(destination[index].bitPattern == sentinel.bitPattern,
                    "nonfinite rejection must not publish a partial row")
        }

        let overflowPublicationError = capturedError {
            try QwenSourceSamplerBoundary.publishFP32Logits(
                [0, 1, 2, Float.greatestFiniteMagnitude, 4],
                into: scratch.logits, vocabularySize: vocabularySize)
        }
        let rejectsOverflowPublication = matchesQwenTextRunnerError(overflowPublicationError) { error in
            guard case let .execution(detail) = error else { return false }
            return detail == "source logit overflows FP16 sampler"
        }
        #expect(rejectsOverflowPublication)
        for index in 0..<vocabularySize {
            #expect(destination[index].bitPattern == sentinel.bitPattern,
                    "overflow rejection must not publish a partial row")
        }
    }

    @Test func syntheticFactoryRejectsZeroAttentionHeadsBeforeRunner() throws {
        let context = try MetalContext()
        let invalidConfiguration = try QwenBF16TextRunnerFixture.make(
            configJSON: QwenBF16TextRunnerFixture.officialConfigJSON(attentionHeads: 0))
        defer { invalidConfiguration.remove() }
        let error = capturedError {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: invalidConfiguration.registrationURL,
                context: context,
                residencyBudgetBytes: invalidConfiguration.totalResidencyBudget())
        }
        let rejectsInvalidArchitecture = (error as? ModelError)
            == .indexCorrupt(detail: "invalid Qwen text architecture")
        #expect(rejectsInvalidArchitecture,
                "zero query heads must fail architecture validation before runner creation")
    }

    @Test func syntheticFactoryRejectsThreeHeadsAtContradictoryQProjectionHeader() throws {
        let context = try MetalContext()
        let qProjection = QwenBF16TextRunnerFixture.layer0Prefix + "self_attn.q_proj.weight"
        let source = try QwenBF16TextRunnerFixture.make(
            configJSON: QwenBF16TextRunnerFixture.officialConfigJSON(attentionHeads: 3))
        defer { source.remove() }
        let error = capturedError {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: source.registrationURL,
                context: context,
                residencyBudgetBytes: source.totalResidencyBudget())
        }
        let rejectsHeaderConflict = matchesQwenBF16WeightError(error) { actual in
            actual == .invalidGeometry("header shape disagrees: \(qProjection)")
        }
        #expect(rejectsHeaderConflict,
                "three query heads with one KV head are valid; the existing q_proj header conflicts")
    }

    @Test func syntheticFactoryRejectsInsufficientResidencyBudget() throws {
        let context = try MetalContext()
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let denseOnlyError = capturedError {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: source.registrationURL,
                context: context,
                residencyBudgetBytes: source.expectedResidentBytes)
        }
        #expect(matchesQwenBF16WeightError(denseOnlyError) {
            $0 == .budgetExceeded
        }, "dense-only admission must not omit the checked expert-cache reservation")

        let cacheShortfallError = capturedError {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: source.registrationURL,
                context: context,
                residencyBudgetBytes: source.totalResidencyBudget() - 1)
        }
        let rejectsCacheShortfall = (cacheShortfallError as? ModelError)
            == .indexCorrupt(detail:
                "invalid source vector or budget: "
                    + "model.language_model.layers.1.linear_attn.dt_bias")
        #expect(rejectsCacheShortfall,
                "a one-byte cache shortfall must fail the bounded resident-vector budget")

        let error = capturedError {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: source.registrationURL,
                context: context,
                residencyBudgetBytes: 1)
        }
        let rejectsInsufficientBudget = matchesQwenBF16WeightError(error) { actual in
            actual == .budgetExceeded
        }
        #expect(rejectsInsufficientBudget)
    }

    @Test func syntheticFactoryRejectsQueryProjectionHeaderShapeConflictPrecisely() throws {
        let context = try MetalContext()
        let qProjection = QwenBF16TextRunnerFixture.layer0Prefix + "self_attn.q_proj.weight"
        let source = try QwenBF16TextRunnerFixture.make(
            shapeOverrides: [qProjection: [8, 16]])
        defer { source.remove() }
        let error = capturedError {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: source.registrationURL,
                context: context,
                residencyBudgetBytes: source.totalResidencyBudget())
        }
        let rejectsShapeConflict = matchesQwenBF16WeightError(error) { actual in
            actual == .invalidGeometry("header shape disagrees: \(qProjection)")
        }
        #expect(rejectsShapeConflict,
                "the explicit q_proj header-shape conflict must name the affected tensor")
    }

    @Test func syntheticFactoryRejectsOutOfRangeProtectedHeaderPrecisely() throws {
        let context = try MetalContext()
        let tensorName = QwenBF16TextRunnerFixture.finalNormName
        let source = try QwenBF16TextRunnerFixture.make(outOfRangeTensor: tensorName)
        defer { source.remove() }
        let error = capturedError {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: source.registrationURL,
                context: context,
                residencyBudgetBytes: source.totalResidencyBudget())
        }
        let rejectsProtectedRange = matchesProtectedHeaderRangeError(error, tensorName: tensorName)
        #expect(rejectsProtectedRange,
                "declared byte count is shape-correct but its protected shard interval is out of range")
    }

    @Test func syntheticFactoryRejectsMisShapedRoutedGateUpHeadersDuringLoad() throws {
        let context = try MetalContext()
        for layer in 0..<2 {
            let tensorName = "model.language_model.layers.\(layer).mlp.experts.gate_up_proj"
            let wrongShape = [QwenBF16TextRunnerFixture.expertCount,
                              QwenBF16TextRunnerFixture.hiddenSize,
                              2 * QwenBF16TextRunnerFixture.routedIntermediateSize]
            let source = try QwenBF16TextRunnerFixture.make(
                shapeOverrides: [tensorName: wrongShape])
            defer { source.remove() }
            #expect(wrongShape.reduce(1, *) == source.tensor(tensorName).words.count,
                    "gate_up shape mutation must preserve the declared BF16 byte count")
            let error = sourceModelLoadError(source, context: context)
            #expect(matchesRoutedHeaderError(error, tensorName: tensorName),
                    "load must reject routed gate_up geometry with invalidHeader before runner construction")
        }
    }

    @Test func syntheticFactoryRejectsMisShapedRoutedDownHeadersDuringLoad() throws {
        let context = try MetalContext()
        for layer in 0..<2 {
            let tensorName = "model.language_model.layers.\(layer).mlp.experts.down_proj"
            let wrongShape = [QwenBF16TextRunnerFixture.expertCount,
                              QwenBF16TextRunnerFixture.routedIntermediateSize,
                              QwenBF16TextRunnerFixture.hiddenSize]
            let source = try QwenBF16TextRunnerFixture.make(
                shapeOverrides: [tensorName: wrongShape])
            defer { source.remove() }
            #expect(wrongShape.reduce(1, *) == source.tensor(tensorName).words.count,
                    "down shape mutation must preserve the declared BF16 byte count")
            let error = sourceModelLoadError(source, context: context)
            #expect(matchesRoutedHeaderError(error, tensorName: tensorName),
                    "load must reject routed down geometry with invalidHeader before runner construction")
        }
    }

    @Test func syntheticFactoryRejectsOutOfFileRoutedGateUpHeadersDuringLoad() throws {
        let context = try MetalContext()
        for layer in 0..<2 {
            let tensorName = "model.language_model.layers.\(layer).mlp.experts.gate_up_proj"
            let source = try QwenBF16TextRunnerFixture.make(outOfRangeTensor: tensorName)
            defer { source.remove() }
            let error = sourceModelLoadError(source, context: context)
            #expect(matchesProtectedHeaderRangeError(error, tensorName: tensorName),
                    "load must reject the protected routed gate_up range before runner construction")
        }
    }

    @Test func syntheticFactoryRejectsOutOfFileRoutedDownHeadersDuringLoad() throws {
        let context = try MetalContext()
        for layer in 0..<2 {
            let tensorName = "model.language_model.layers.\(layer).mlp.experts.down_proj"
            let source = try QwenBF16TextRunnerFixture.make(outOfRangeTensor: tensorName)
            defer { source.remove() }
            let error = sourceModelLoadError(source, context: context)
            #expect(matchesProtectedHeaderRangeError(error, tensorName: tensorName),
                    "load must reject the protected routed down range before runner construction")
        }
    }

    @Test func syntheticFactoryRejectsMissingRoutedIndexMappingDuringLoad() throws {
        let context = try MetalContext()
        let tensorName = "model.language_model.layers.1.mlp.experts.gate_up_proj"
        let source = try QwenBF16TextRunnerFixture.make(
            missingIndexMappings: [tensorName])
        defer { source.remove() }
        let error = sourceModelLoadError(source, context: context)
        #expect((error as? ModelError) == .tensorNotFound(name: tensorName),
                "a missing routed index entry must fail source-model loading by tensor name")
    }

    @Test func syntheticFactoryRejectsWrongRoutedShardMappingDuringLoad() throws {
        let context = try MetalContext()
        let tensorName = "model.language_model.layers.1.mlp.experts.down_proj"
        let wrongShard = OfficialQwenSourceIdentity.pinned.shards[0].filename
        let source = try QwenBF16TextRunnerFixture.make(
            indexShardOverrides: [tensorName: wrongShard])
        defer { source.remove() }
        #expect(source.tensorToShard[tensorName] == wrongShard)
        #expect(wrongShard != source.shardNames[1])
        let error = sourceModelLoadError(source, context: context)
        #expect(matchesProtectedTensorAdmissionError(error, tensorName: tensorName),
                "the protected handle must reject a routed tensor mapped to the dense shard")
    }
}

private func residentQwenBufferIdentities(_ model: QwenOfficialSourceModel) -> [ObjectIdentifier] {
    let chunks = model.entryWeights.inspectedChunks
        + model.layers.flatMap { $0.weights.inspectedChunks }
    return chunks.map { ObjectIdentifier($0.buffer) }
}

private func decodedVectorElementCount(_ model: QwenOfficialSourceModel) -> Int {
    model.finalNorm.count + model.layers.reduce(0) { total, layer in
        total + layer.inputNorm.count + layer.postNorm.count
            + (layer.queryNorm?.count ?? 0) + (layer.keyNorm?.count ?? 0)
            + (layer.linearVectors.map {
                $0.convolution.count + $0.normalization.count + $0.aLog.count + $0.timeStepBias.count
            } ?? 0)
    }
}

private func greedyToken(_ logits: [Float]) -> Int32 {
    let maximum = logits.max() ?? 0
    return Int32(logits.firstIndex(of: maximum) ?? 0)
}

private func fp16RawSamplerGreedyToken(_ logits: [Float]) -> Int32 {
    let half = logits.map(Float16.init)
    let maximum = half.map(Float.init).max() ?? 0
    let exponentials = half.map { expf(Float($0) - maximum) }
    let denominator = exponentials.reduce(Float(0), +)
    let probabilities = exponentials.map { Float16($0 / denominator) }
    let highest = probabilities.max() ?? 0
    return Int32(probabilities.firstIndex(of: highest) ?? 0)
}

private func capturedError<Value>(_ operation: () throws -> Value) -> Error? {
    do {
        _ = try operation()
        return nil
    } catch {
        return error
    }
}

private func capturedAsyncError<Value>(
    _ operation: () async throws -> Value
) async -> Error? {
    do {
        _ = try await operation()
        return nil
    } catch {
        return error
    }
}

private func matchesQwenTextRunnerError(
    _ error: Error?, matching predicate: (QwenTextRunnerError) -> Bool
) -> Bool {
    guard let error = error as? QwenTextRunnerError else { return false }
    return predicate(error)
}

private func matchesQwenBF16WeightError(
    _ error: Error?, matching predicate: (QwenBF16WeightError) -> Bool
) -> Bool {
    guard let error = error as? QwenBF16WeightError else { return false }
    return predicate(error)
}

private func sourceModelLoadError(
    _ source: QwenBF16TextRunnerFixture.Source,
    context: MetalContext
) -> Error? {
    capturedError {
        try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
    }
}

private func matchesRoutedHeaderError(_ error: Error?, tensorName: String) -> Bool {
    guard let cacheError = error as? QwenBF16ExpertCacheError,
          case let .invalidHeader(actualName) = cacheError else { return false }
    return actualName == tensorName
}

private func matchesProtectedTensorAdmissionError(_ error: Error?, tensorName: String) -> Bool {
    guard let sourceError = error as? OfficialSourceHandleError,
          case let .invalidTensor(detail) = sourceError else { return false }
    return detail.contains(tensorName)
}

private func matchesProtectedHeaderRangeError(_ error: Error?, tensorName: String) -> Bool {
    guard let sourceError = error as? OfficialSourceHandleError,
          case let .invalidTensor(detail) = sourceError else { return false }
    let expectedPrefix = "invalid shard header: tensorOutOfRange(name: \"\(tensorName)\""
    return detail.hasPrefix(expectedPrefix)
        && detail.contains(", end:")
        && detail.contains(", fileSize:")
}

private func qwenValuesWithinTolerance(
    _ actual: [Float], _ expected: [Float], absolute: Float, relative: Float
) -> Bool {
    guard actual.count == expected.count else { return false }
    for index in actual.indices {
        let actualValue = actual[index]
        let expectedValue = expected[index]
        guard actualValue.isFinite && expectedValue.isFinite else { return false }
        let limit = absolute + relative * abs(expectedValue)
        guard abs(actualValue - expectedValue) <= limit else { return false }
    }
    return true
}

@discardableResult
private func assertQwenClose(
    _ actual: [Float], _ expected: [Float], absolute: Float, relative: Float, label: String
) -> Float {
    guard actual.count == expected.count else {
        Issue.record("\(label): count \(actual.count) != expected \(expected.count)")
        return .infinity
    }
    let maximum = zip(actual, expected).map { pair -> Float in
        guard pair.0.isFinite && pair.1.isFinite else { return .infinity }
        return abs(pair.0 - pair.1)
    }.max() ?? 0
    #expect(qwenValuesWithinTolerance(
        actual, expected, absolute: absolute, relative: relative),
        "\(label): per-value tolerance failed; max absolute error \(maximum)")
    return maximum
}
