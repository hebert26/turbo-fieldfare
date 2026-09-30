import Darwin
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenBF16FullAttentionTests {
    @Test func actualGPUFullAttentionUsesSelectedBF16MatricesAndMatchesIndependentCPUAcrossSequentialCache() async throws {
        let context = try MetalContext()
        let source = try QwenBF16SyntheticSource.make(
            tensors: FullAttentionBF16Fixture.tensors())
        defer { source.remove() }
        let weights = try FullAttentionBF16Fixture.weights(
            context: context, source: source)
        assertAwkwardRowChunks(weights)

        let kv = try makeKV(context: context, capacity: 4)
        try initializeKV(kv, byte: 0xa5)
        let configuration = try FullAttentionBF16Fixture.configuration
        let step = try makeStep(
            context: context, weights: weights, kv: kv, configuration: configuration)

        var cachedKeys: [Float] = []
        var cachedValues: [Float] = []
        var lastReference: FullAttentionReference?
        var lastInput: [Float] = []
        for (position, hidden) in FullAttentionBF16Fixture.tokens.enumerated() {
            let expected = independentFullAttentionStep(
                hidden: hidden, position: position,
                cachedKeys: cachedKeys, cachedValues: cachedValues,
                configuration: configuration,
                queryNorm: FullAttentionBF16Fixture.queryNorm,
                keyNorm: FullAttentionBF16Fixture.keyNorm)
            let actual = try await step.append(hidden: hidden)
            expectFrozenAttentionClose(
                actual, expected.output, label: "token \(position) output")
            let actualIsFinite = actual.allSatisfy { $0.isFinite }
            #expect(actualIsFinite)
            let stepCommittedPosition = try await step.committedPosition()
            #expect(stepCommittedPosition == position + 1)

            let view = try kv.view(layer: FullAttentionBF16Fixture.layer)
            #expect(view.validTokenCount == position + 1)
            let layerCommittedPosition = try kv.committedPosition(
                layer: FullAttentionBF16Fixture.layer)
            #expect(layerCommittedPosition == position + 1)
            let expectedKeys = cachedKeys + expected.rotatedKey
            let expectedValues = cachedValues + expected.value
            expectFrozenAttentionClose(
                readFloats(view.key, count: expectedKeys.count), expectedKeys,
                label: "token \(position) committed rotated-key prefix")
            expectFrozenAttentionClose(
                readFloats(view.value, count: expectedValues.count), expectedValues,
                label: "token \(position) committed value prefix")

            cachedKeys = expectedKeys
            cachedValues = expectedValues
            lastReference = expected
            lastInput = hidden
        }

        // The names are deliberately non-canonical: the real GPU result and
        // these independent wrong-path controls exercise the caller's selectors.
        let final = try #require(lastReference)
        let finalPosition = FullAttentionBF16Fixture.tokens.count - 1
        let priorKeys = Array(cachedKeys.dropLast(FullAttentionBF16Fixture.keyValueWidth))
        let priorValues = Array(cachedValues.dropLast(FullAttentionBF16Fixture.keyValueWidth))
        let variants: [(String, FullAttentionReference)] = [
            ("ungated", independentFullAttentionStep(
                hidden: lastInput, position: finalPosition,
                cachedKeys: priorKeys, cachedValues: priorValues,
                configuration: configuration,
                queryNorm: FullAttentionBF16Fixture.queryNorm,
                keyNorm: FullAttentionBF16Fixture.keyNorm,
                applyOutputGate: false)),
            ("wrong GQA grouping", independentFullAttentionStep(
                hidden: lastInput, position: finalPosition,
                cachedKeys: priorKeys, cachedValues: priorValues,
                configuration: configuration,
                queryNorm: FullAttentionBF16Fixture.queryNorm,
                keyNorm: FullAttentionBF16Fixture.keyNorm,
                useContiguousGroupedHeads: false)),
            ("current-token-only attention", independentFullAttentionStep(
                hidden: lastInput, position: finalPosition,
                cachedKeys: priorKeys, cachedValues: priorValues,
                configuration: configuration,
                queryNorm: FullAttentionBF16Fixture.queryNorm,
                keyNorm: FullAttentionBF16Fixture.keyNorm,
                attendToCachedTokens: false)),
            ("wrong RoPE span", independentFullAttentionStep(
                hidden: lastInput, position: finalPosition,
                cachedKeys: priorKeys, cachedValues: priorValues,
                configuration: configuration,
                queryNorm: FullAttentionBF16Fixture.queryNorm,
                keyNorm: FullAttentionBF16Fixture.keyNorm,
                rotaryDimensionOverride: 2)),
            ("wrong output matrix", independentFullAttentionStep(
                hidden: lastInput, position: finalPosition,
                cachedKeys: priorKeys, cachedValues: priorValues,
                configuration: configuration,
                queryNorm: FullAttentionBF16Fixture.queryNorm,
                keyNorm: FullAttentionBF16Fixture.keyNorm,
                outputBits: Array(FullAttentionBF16Fixture.outputBits.reversed()))),
        ]
        for (label, variant) in variants {
            let delta = maxDifference(final.output, variant.output)
            #expect(delta > FullAttentionBF16Fixture.negativeControlMinimum,
                    "negative control \(label) is not separated: \(delta)")
        }
    }

    @Test func invalidInputAndProjectionGeometryPreserveEveryKVByteAndPosition() async throws {
        let context = try MetalContext()
        let source = try QwenBF16SyntheticSource.make(
            tensors: FullAttentionBF16Fixture.tensors())
        defer { source.remove() }
        let weights = try FullAttentionBF16Fixture.weights(
            context: context, source: source)
        let kv = try makeKV(context: context, capacity: 4)
        try initializeKV(kv, byte: 0x6d)
        let configuration = try FullAttentionBF16Fixture.configuration
        let step = try makeStep(
            context: context, weights: weights, kv: kv, configuration: configuration)

        let emptyBefore = try captureKV(kv)
        var rejectedHiddenShape = false
        do {
            _ = try await step.append(
                hidden: Array(FullAttentionBF16Fixture.tokens[0].dropLast()))
        } catch let error as QwenTextRunnerError {
            if case .invalidState(detail: _) = error { rejectedHiddenShape = true }
        }
        #expect(rejectedHiddenShape)
        let stateAfterInvalidShape = try captureKV(kv)
        #expect(stateAfterInvalidShape == emptyBefore)
        let positionAfterInvalidShape = try await step.committedPosition()
        #expect(positionAfterInvalidShape == 0)

        _ = try await step.append(hidden: FullAttentionBF16Fixture.tokens[0])
        let committedBefore = try captureKV(kv)
        let wrongHeadGeometry = try QwenFullAttentionConfiguration(
            queryHeadCount: 2, keyValueHeadCount: 2,
            headDimension: FullAttentionBF16Fixture.headDimension,
            rotaryDimension: FullAttentionBF16Fixture.rotaryDimension,
            theta: FullAttentionBF16Fixture.theta,
            epsilon: FullAttentionBF16Fixture.epsilon)
        var rejectedProjectionGeometry = false
        do {
            _ = try makeStep(
                context: context, weights: weights, kv: kv,
                configuration: wrongHeadGeometry)
        } catch {
            rejectedProjectionGeometry = true
        }
        #expect(rejectedProjectionGeometry)
        let stateAfterGeometryRejection = try captureKV(kv)
        #expect(stateAfterGeometryRejection == committedBefore)
        let positionAfterGeometryRejection = try await step.committedPosition()
        #expect(positionAfterGeometryRejection == 1)
    }

    @Test func capacityOverflowPreservesPreviouslyCommittedKVBytesAndPosition() async throws {
        let context = try MetalContext()
        let source = try QwenBF16SyntheticSource.make(
            tensors: FullAttentionBF16Fixture.tensors())
        defer { source.remove() }
        let weights = try FullAttentionBF16Fixture.weights(
            context: context, source: source)
        let kv = try makeKV(context: context, capacity: 2)
        try initializeKV(kv, byte: 0x39)
        let configuration = try FullAttentionBF16Fixture.configuration
        let step = try makeStep(
            context: context, weights: weights, kv: kv, configuration: configuration)
        for hidden in FullAttentionBF16Fixture.tokens.prefix(2) {
            _ = try await step.append(hidden: hidden)
        }

        let before = try captureKV(kv)
        var gotCapacityError = false
        do {
            _ = try await step.append(hidden: FullAttentionBF16Fixture.tokens[2])
        } catch let error as QwenFullAttentionKVError {
            if case .capacityExceeded(start: 2, count: 1, capacity: 2) = error {
                gotCapacityError = true
            }
        }
        #expect(gotCapacityError)
        let stateAfterCapacityRejection = try captureKV(kv)
        #expect(stateAfterCapacityRejection == before)
        let positionAfterCapacityRejection = try await step.committedPosition()
        #expect(positionAfterCapacityRejection == 2)
    }

    @Test func rejectsNonfiniteBF16AttentionActivationWithoutPublishingKV() async throws {
        let context = try MetalContext()
        var queryBits = FullAttentionBF16Fixture.queryBits
        queryBits[0] = QwenBF16TestFixture.nanBits
        let source = try QwenBF16SyntheticSource.make(
            tensors: FullAttentionBF16Fixture.tensors(queryBits: queryBits))
        defer { source.remove() }
        let weights = try FullAttentionBF16Fixture.weights(
            context: context, source: source)
        let kv = try makeKV(context: context, capacity: 2)
        try initializeKV(kv, byte: 0x51)
        let configuration = try FullAttentionBF16Fixture.configuration
        let step = try makeStep(
            context: context, weights: weights, kv: kv, configuration: configuration)
        let before = try captureKV(kv)

        var rejectedNonfinite = false
        do {
            _ = try await step.append(hidden: FullAttentionBF16Fixture.tokens[0])
        } catch let error as QwenFullAttentionError {
            rejectedNonfinite = error == .nonfiniteAttention
        }
        #expect(rejectedNonfinite)
        let stateAfterNonfiniteRejection = try captureKV(kv)
        #expect(stateAfterNonfiniteRejection == before)
        let positionAfterNonfiniteRejection = try await step.committedPosition()
        #expect(positionAfterNonfiniteRejection == 0)
    }

    @Test func injectedFailuresAtEverySubmissionBoundaryAndBeforeCommitPreserveAllKVBytes() async throws {
        let context = try MetalContext()
        let source = try QwenBF16SyntheticSource.make(
            tensors: FullAttentionBF16Fixture.tensors())
        defer { source.remove() }
        let weights = try FullAttentionBF16Fixture.weights(
            context: context, source: source)
        let kv = try makeKV(context: context, capacity: 4)
        try initializeKV(kv, byte: 0x27)
        let configuration = try FullAttentionBF16Fixture.configuration
        let baseline = try makeStep(
            context: context, weights: weights, kv: kv, configuration: configuration)
        _ = try await baseline.append(hidden: FullAttentionBF16Fixture.tokens[0])
        let before = try captureKV(kv)
        let targets: [FullAttentionCheckpointTarget] = [
            .beforeSubmission("qkv"), .afterSubmission("qkv"),
            .beforeSubmission("normRoPE"), .afterSubmission("normRoPE"),
            .beforeSubmission("gate"), .afterSubmission("gate"),
            .beforeSubmission("output"), .afterSubmission("output"),
            .beforeCommit,
        ]

        for target in targets {
            let hooks = QwenBF16FullAttentionHooks(checkpoint: { checkpoint in
                if target.matches(checkpoint) { throw InjectedAttentionFailure() }
            })
            let candidate = try makeStep(
                context: context, weights: weights, kv: kv,
                configuration: configuration, hooks: hooks)
            var injected = false
            do {
                _ = try await candidate.append(hidden: FullAttentionBF16Fixture.tokens[1])
            } catch is InjectedAttentionFailure {
                injected = true
            }
            #expect(injected, "checkpoint did not inject at \(target)")
            let stateAfterInjectedFailure = try captureKV(kv)
            #expect(stateAfterInjectedFailure == before,
                    "failure at \(target) changed committed KV bytes or positions")
            let positionAfterInjectedFailure = try await candidate.committedPosition()
            #expect(positionAfterInjectedFailure == 1)
        }
    }

    @Test func actualTaskCancellationAtEverySubmissionBoundaryAndBeforeCommitPreservesAllKVBytes() async throws {
        let context = try MetalContext()
        let source = try QwenBF16SyntheticSource.make(
            tensors: FullAttentionBF16Fixture.tensors())
        defer { source.remove() }
        let weights = try FullAttentionBF16Fixture.weights(
            context: context, source: source)
        let kv = try makeKV(context: context, capacity: 4)
        try initializeKV(kv, byte: 0x18)
        let configuration = try FullAttentionBF16Fixture.configuration
        let baseline = try makeStep(
            context: context, weights: weights, kv: kv, configuration: configuration)
        _ = try await baseline.append(hidden: FullAttentionBF16Fixture.tokens[0])
        let before = try captureKV(kv)
        let targets: [FullAttentionCheckpointTarget] = [
            .beforeSubmission("qkv"), .afterSubmission("qkv"),
            .beforeSubmission("normRoPE"), .afterSubmission("normRoPE"),
            .beforeSubmission("gate"), .afterSubmission("gate"),
            .beforeSubmission("output"), .afterSubmission("output"),
            .beforeCommit,
        ]

        for target in targets {
            let gate = AsyncCheckpointGate()
            let hooks = QwenBF16FullAttentionHooks(checkpoint: { checkpoint in
                if target.matches(checkpoint) { await gate.suspendUntilReleased() }
            })
            let candidate = try makeStep(
                context: context, weights: weights, kv: kv,
                configuration: configuration, hooks: hooks)
            let task = Task {
                try await candidate.append(hidden: FullAttentionBF16Fixture.tokens[1])
            }
            await gate.waitUntilEntered()
            task.cancel()
            await gate.release()
            var cancelled = false
            do {
                _ = try await task.value
            } catch is CancellationError {
                cancelled = true
            }
            #expect(cancelled, "task cancellation was not observed at \(target)")
            let stateAfterCancellation = try captureKV(kv)
            #expect(stateAfterCancellation == before,
                    "cancellation at \(target) changed committed KV bytes or positions")
            let positionAfterCancellation = try await candidate.committedPosition()
            #expect(positionAfterCancellation == 1)
        }
    }

    @Test func reentrantAppendDuringSubmittedProjectionIsRejectedWithoutPublishingTwice() async throws {
        let context = try MetalContext()
        let source = try QwenBF16SyntheticSource.make(
            tensors: FullAttentionBF16Fixture.tensors())
        defer { source.remove() }
        let weights = try FullAttentionBF16Fixture.weights(
            context: context, source: source)
        let kv = try makeKV(context: context, capacity: 4)
        try initializeKV(kv, byte: 0x43)
        let configuration = try FullAttentionBF16Fixture.configuration
        let gate = AsyncCheckpointGate()
        let hooks = QwenBF16FullAttentionHooks(checkpoint: { checkpoint in
            if FullAttentionCheckpointTarget.afterSubmission("qkv").matches(checkpoint) {
                await gate.suspendUntilReleased()
            }
        })
        let step = try makeStep(
            context: context, weights: weights, kv: kv,
            configuration: configuration, hooks: hooks)
        let before = try captureKV(kv)
        let first = Task {
            try await step.append(hidden: FullAttentionBF16Fixture.tokens[0])
        }
        await gate.waitUntilEntered()
        var reentrantAppendRejected = false
        do {
            _ = try await step.append(hidden: FullAttentionBF16Fixture.tokens[1])
        } catch let error as QwenTextRunnerError {
            if case .operationInProgress = error { reentrantAppendRejected = true }
        }
        #expect(reentrantAppendRejected)
        let stateDuringFirstSubmission = try captureKV(kv)
        #expect(stateDuringFirstSubmission == before)
        await gate.release()
        _ = try await first.value
        let positionAfterFirstAppend = try await step.committedPosition()
        #expect(positionAfterFirstAppend == 1)
    }
}

/// Frozen before GPU execution: elementwise abs(error) <= 2e-5 +
/// 2e-5*abs(expected), with no additional ULP allowance. Literal BF16
/// projections are exact inputs; the fixed margin covers FP32 reduction/FMA,
/// Metal RMSNorm/RoPE, and stable-softmax exp/div variation through O projection.
private enum FullAttentionBF16Fixture {
    static let hiddenSize = 5
    static let queryHeadCount = 4
    static let keyValueHeadCount = 2
    static let headDimension = 5
    static let rotaryDimension = 4
    static let queryRows = 2 * queryHeadCount * headDimension
    static let queryWidth = queryHeadCount * headDimension
    static let keyValueWidth = keyValueHeadCount * headDimension
    static let layer = 3
    static let theta: Float = 10_000
    static let epsilon: Float = 1e-6
    static let maximumChunkBytes: UInt64 = 43
    static let absoluteTolerance: Float = 2e-5
    static let relativeTolerance: Float = 2e-5
    static let negativeControlMinimum: Float = 5e-4

    static let names = QwenBF16FullAttentionNames(
        query: "synthetic.attention.q",
        key: "synthetic.attention.k",
        value: "synthetic.attention.v",
        output: "synthetic.attention.o")

    // Literal BF16 words vary by query head and place query and gate rows in
    // Qwen's [head, query-dimension, gate-dimension] projection order.
    private static let queryPatterns: [[UInt16]] = [
        [0x3f80, 0x3f00, 0xbf00, 0x3e80, 0x3e00],
        [0xbe80, 0x3f80, 0x3f00, 0xbe00, 0x3e80],
        [0x3f00, 0xbe80, 0x3e00, 0x3f80, 0xbf00],
        [0x3e00, 0x3e80, 0x3f00, 0xbf80, 0x3d80],
        [0xbf00, 0x3e00, 0x3f80, 0x3e80, 0xbe00],
        [0x3e80, 0xbf00, 0x3e00, 0x3f00, 0x3f80],
        [0xbf80, 0x3f00, 0x3e80, 0xbe00, 0x3f00],
        [0x3d80, 0xbe00, 0x3e80, 0x3f00, 0xbf00],
    ]
    private static let keyPatterns: [[UInt16]] = [
        [0x3f00, 0xbe80, 0x3e00, 0x3d80, 0x3e80],
        [0xbf00, 0x3f00, 0x3e80, 0xbe00, 0x3e00],
        [0x3e00, 0xbf00, 0x3f80, 0x3e80, 0xbe80],
        [0x3f80, 0x3e00, 0xbe80, 0x3f00, 0x3d80],
        [0xbe80, 0x3e80, 0xbf00, 0x3f80, 0x3e00],
    ]
    private static let valuePatterns: [[UInt16]] = [
        [0x3e80, 0x3f00, 0xbe00, 0x3f40, 0x3d80],
        [0xbf00, 0x3e00, 0x3f00, 0xbe80, 0x3f80],
        [0x3f40, 0xbf00, 0x3e80, 0x3d80, 0xbe00],
        [0x3e00, 0x3f80, 0xbe80, 0x3f00, 0xbf00],
        [0xbe00, 0x3e80, 0x3f40, 0xbf80, 0x3e00],
    ]
    private static let outputPatterns: [[UInt16]] = [
        [0x3f00, 0xbe80, 0x3e00, 0x3e80, 0xbe00],
        [0xbf00, 0x3e80, 0x3f00, 0xbe00, 0x3e80],
        [0x3e00, 0xbf00, 0x3f40, 0x3d80, 0x3f00],
        [0x3e80, 0x3f80, 0xbe80, 0x3f00, 0xbd80],
        [0xbf80, 0x3e00, 0x3e80, 0xbf00, 0x3f40],
    ]

    static let queryBits: [UInt16] = (0..<queryRows).flatMap { row in
        queryPatterns[(row * 3 + 1) % queryPatterns.count]
    }
    // Shift each KV head's literal row pattern so the independent wrong-GQA
    // grouping control cannot collapse to identical key/value heads.
    static let keyBits: [UInt16] = (0..<(keyValueHeadCount * headDimension)).flatMap { row in
        let head = row / headDimension
        return keyPatterns[(row * 3 + 1 + head * 2) % keyPatterns.count]
    }
    static let valueBits: [UInt16] = (0..<(keyValueHeadCount * headDimension)).flatMap { row in
        let head = row / headDimension
        return valuePatterns[(row * 2 + 1 + head * 2) % valuePatterns.count]
    }
    static let outputBits: [UInt16] = (0..<hiddenSize).flatMap { row in
        Array(repeating: outputPatterns[row], count: queryHeadCount).flatMap { $0 }
    }

    static let queryNorm: [Float] = [0.125, -0.0625, 0.09375, 0.03125, -0.125]
    static let keyNorm: [Float] = [-0.09375, 0.0625, 0.125, -0.03125, 0.078125]
    static let tokens: [[Float]] = [
        [0.25, -0.5, 0.75, 0.125, -0.25],
        [-0.125, 0.5, 0.25, -0.75, 0.5],
        [0.75, 0.25, -0.375, 0.5, 0.125],
    ]

    static var configuration: QwenFullAttentionConfiguration {
        get throws {
            try QwenFullAttentionConfiguration(
                queryHeadCount: queryHeadCount,
                keyValueHeadCount: keyValueHeadCount,
                headDimension: headDimension,
                rotaryDimension: rotaryDimension,
                theta: theta,
                epsilon: epsilon)
        }
    }

    static func tensors(queryBits override: [UInt16]? = nil) -> [QwenBF16LiteralTensor] {
        let selectedQueryBits = override ?? queryBits
        return [
            QwenBF16LiteralTensor(name: names.query, rows: queryRows,
                                  columns: hiddenSize, bits: selectedQueryBits),
            QwenBF16LiteralTensor(name: names.key,
                                  rows: keyValueHeadCount * headDimension,
                                  columns: hiddenSize, bits: keyBits),
            QwenBF16LiteralTensor(name: names.value,
                                  rows: keyValueHeadCount * headDimension,
                                  columns: hiddenSize, bits: valueBits),
            QwenBF16LiteralTensor(name: names.output, rows: hiddenSize,
                                  columns: queryWidth, bits: outputBits),
        ]
    }

    static func weights(
        context: MetalContext,
        source: QwenBF16SyntheticSource
    ) throws -> QwenBF16Weights {
        try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [
                spec(source, name: names.query, rows: queryRows, columns: hiddenSize),
                spec(source, name: names.key,
                     rows: keyValueHeadCount * headDimension, columns: hiddenSize),
                spec(source, name: names.value,
                     rows: keyValueHeadCount * headDimension, columns: hiddenSize),
                spec(source, name: names.output, rows: hiddenSize, columns: queryWidth),
            ], residencyBudget: 800,
            maximumChunkBytes: maximumChunkBytes,
            checkpoint: { _ in })
    }
}

private struct FullAttentionReference {
    let output: [Float]
    let rotatedKey: [Float]
    let value: [Float]
}

private func makeStep(
    context: MetalContext,
    weights: QwenBF16Weights,
    kv: QwenFullAttentionKV,
    configuration: QwenFullAttentionConfiguration,
    hooks: QwenBF16FullAttentionHooks = .none
) throws -> QwenBF16FullAttentionStep {
    try QwenBF16FullAttentionStep(
        context: context, weights: weights, names: FullAttentionBF16Fixture.names,
        configuration: configuration,
        hiddenSize: FullAttentionBF16Fixture.hiddenSize,
        layer: FullAttentionBF16Fixture.layer,
        kv: kv,
        queryNorm: FullAttentionBF16Fixture.queryNorm,
        keyNorm: FullAttentionBF16Fixture.keyNorm,
        hooks: hooks)
}

private func spec(
    _ source: QwenBF16SyntheticSource,
    name: String,
    rows: Int,
    columns: Int
) -> QwenBF16TensorSpec {
    QwenBF16TensorSpec(name: name, shardName: source.shardName,
                       role: .dense, rows: rows, columns: columns)
}

private func makeKV(context: MetalContext, capacity: Int) throws -> QwenFullAttentionKV {
    let mask: [UInt8] = (0..<40).map { index in
        (index + 1).isMultiple(of: 4) ? 1 : 0
    }
    return try QwenFullAttentionKV(
        device: context.device, fullAttentionLayerMask: mask,
        maxContext: capacity,
        keyValueHeadCount: FullAttentionBF16Fixture.keyValueHeadCount,
        headDimension: FullAttentionBF16Fixture.headDimension)
}

private func initializeKV(_ kv: QwenFullAttentionKV, byte: Int32) throws {
    for layer in kv.fullLayerIndices {
        let view = try kv.view(layer: layer)
        memset(view.key.contents(), byte, view.key.length)
        memset(view.value.contents(), byte, view.value.length)
    }
}

private struct KVLayerImage: Equatable {
    let position: Int
    let key: Data
    let value: Data
}

private struct KVImage: Equatable {
    let layers: [Int: KVLayerImage]
}

private func captureKV(_ kv: QwenFullAttentionKV) throws -> KVImage {
    var layers: [Int: KVLayerImage] = [:]
    for layer in kv.fullLayerIndices {
        let view = try kv.view(layer: layer)
        layers[layer] = KVLayerImage(
            position: try kv.committedPosition(layer: layer),
            key: Data(bytes: view.key.contents(), count: view.key.length),
            value: Data(bytes: view.value.contents(), count: view.value.length))
    }
    return KVImage(layers: layers)
}

private func assertAwkwardRowChunks(_ weights: QwenBF16Weights) {
    let chunks = weights.inspectedChunks
    let queryRows = chunks.filter { $0.name == FullAttentionBF16Fixture.names.query }
    let keyRows = chunks.filter { $0.name == FullAttentionBF16Fixture.names.key }
    let valueRows = chunks.filter { $0.name == FullAttentionBF16Fixture.names.value }
    let outputRows = chunks.filter { $0.name == FullAttentionBF16Fixture.names.output }
    #expect(queryRows.map(\.firstRow) == Array(stride(from: 0, to: 40, by: 4)))
    #expect(queryRows.map(\.rowCount) == Array(repeating: 4, count: 10))
    #expect(keyRows.map(\.firstRow) == [0, 4, 8])
    #expect(keyRows.map(\.rowCount) == [4, 4, 2])
    #expect(valueRows.map(\.firstRow) == [0, 4, 8])
    #expect(valueRows.map(\.rowCount) == [4, 4, 2])
    #expect(outputRows.map(\.firstRow) == [0, 1, 2, 3, 4])
    #expect(outputRows.map(\.rowCount) == Array(repeating: 1, count: 5))
    #expect(chunks.count == queryRows.count + keyRows.count + valueRows.count + outputRows.count)
    #expect(chunks.allSatisfy {
        $0.buffer.storageMode == .shared && $0.buffer.length <= 43
    })
}

/// Independent CPU full-layer oracle. It decodes only the literal test BF16
/// matrices and never calls QwenFullAttention.evaluate or reads GPU outputs.
private func independentFullAttentionStep(
    hidden: [Float],
    position: Int,
    cachedKeys: [Float],
    cachedValues: [Float],
    configuration: QwenFullAttentionConfiguration,
    queryNorm: [Float],
    keyNorm: [Float],
    applyOutputGate: Bool = true,
    useContiguousGroupedHeads: Bool = true,
    attendToCachedTokens: Bool = true,
    rotaryDimensionOverride: Int? = nil,
    outputBits: [UInt16] = FullAttentionBF16Fixture.outputBits
) -> FullAttentionReference {
    let qHeads = configuration.queryHeadCount
    let kvHeads = configuration.keyValueHeadCount
    let dimension = configuration.headDimension
    let queryWidth = qHeads * dimension
    let kvWidth = kvHeads * dimension
    let rotaryDimension = rotaryDimensionOverride ?? configuration.rotaryDimension

    let queryAndGate = QwenBF16TestFixture.cpuProjection(
        input: hidden, matrixBits: FullAttentionBF16Fixture.queryBits,
        rows: FullAttentionBF16Fixture.queryRows, columns: FullAttentionBF16Fixture.hiddenSize)
    let keyProjection = QwenBF16TestFixture.cpuProjection(
        input: hidden, matrixBits: FullAttentionBF16Fixture.keyBits,
        rows: kvWidth, columns: FullAttentionBF16Fixture.hiddenSize)
    let value = QwenBF16TestFixture.cpuProjection(
        input: hidden, matrixBits: FullAttentionBF16Fixture.valueBits,
        rows: kvWidth, columns: FullAttentionBF16Fixture.hiddenSize)

    var query: [Float] = []
    var rawGate: [Float] = []
    query.reserveCapacity(queryWidth)
    rawGate.reserveCapacity(queryWidth)
    for head in 0..<qHeads {
        let base = head * dimension * 2
        query.append(contentsOf: queryAndGate[base..<(base + dimension)])
        rawGate.append(contentsOf: queryAndGate[(base + dimension)..<(base + 2 * dimension)])
    }

    let normalizedQuery = independentHeadRMSNorm(
        query, headCount: qHeads, dimension: dimension,
        storedWeight: queryNorm, epsilon: configuration.epsilon)
    let normalizedKey = independentHeadRMSNorm(
        keyProjection, headCount: kvHeads, dimension: dimension,
        storedWeight: keyNorm, epsilon: configuration.epsilon)
    let rotatedQuery = independentPartialRoPE(
        normalizedQuery, headCount: qHeads, dimension: dimension,
        rotaryDimension: rotaryDimension, theta: configuration.theta,
        position: position)
    let rotatedKey = independentPartialRoPE(
        normalizedKey, headCount: kvHeads, dimension: dimension,
        rotaryDimension: rotaryDimension, theta: configuration.theta,
        position: position)

    let pastKeys = attendToCachedTokens ? cachedKeys : []
    let pastValues = attendToCachedTokens ? cachedValues : []
    let allKeys = pastKeys + rotatedKey
    let allValues = pastValues + value
    let keyTokenCount = allKeys.count / kvWidth
    let groupSize = qHeads / kvHeads
    let scale = 1 / Float(dimension).squareRoot()
    var context = [Float](repeating: 0, count: queryWidth)
    for queryHead in 0..<qHeads {
        let kvHead = useContiguousGroupedHeads
            ? queryHead / groupSize
            : queryHead % kvHeads
        let queryBase = queryHead * dimension
        var scores: [Float] = []
        for keyToken in 0..<keyTokenCount {
            let keyBase = (keyToken * kvHeads + kvHead) * dimension
            var score: Float = 0
            for component in 0..<dimension {
                score += rotatedQuery[queryBase + component]
                    * allKeys[keyBase + component]
            }
            scores.append(score * scale)
        }
        let maximum = scores.max() ?? 0
        var denominator: Float = 0
        for index in scores.indices {
            scores[index] = Float(Foundation.exp(Double(scores[index] - maximum)))
            denominator += scores[index]
        }
        for keyToken in scores.indices {
            let probability = scores[keyToken] / denominator
            let valueBase = (keyToken * kvHeads + kvHead) * dimension
            for component in 0..<dimension {
                context[queryBase + component] += probability
                    * allValues[valueBase + component]
            }
        }
    }

    let gated = zip(context, rawGate).map { attention, gate in
        attention * (applyOutputGate ? independentSigmoid(gate) : 1)
    }
    let output = QwenBF16TestFixture.cpuProjection(
        input: gated, matrixBits: outputBits,
        rows: FullAttentionBF16Fixture.hiddenSize, columns: queryWidth)
    return FullAttentionReference(output: output, rotatedKey: rotatedKey, value: value)
}

private func independentHeadRMSNorm(
    _ values: [Float],
    headCount: Int,
    dimension: Int,
    storedWeight: [Float],
    epsilon: Float
) -> [Float] {
    var result = [Float](repeating: 0, count: values.count)
    for head in 0..<headCount {
        let base = head * dimension
        var squareSum: Float = 0
        for component in 0..<dimension {
            squareSum += values[base + component] * values[base + component]
        }
        let inverseRMS = 1 / (squareSum / Float(dimension) + epsilon).squareRoot()
        for component in 0..<dimension {
            result[base + component] = values[base + component]
                * inverseRMS * (1 + storedWeight[component])
        }
    }
    return result
}

private func independentPartialRoPE(
    _ values: [Float],
    headCount: Int,
    dimension: Int,
    rotaryDimension: Int,
    theta: Float,
    position: Int
) -> [Float] {
    var result = values
    let half = rotaryDimension / 2
    for head in 0..<headCount {
        let base = head * dimension
        for frequency in 0..<half {
            let exponent = Float(2 * frequency) / Float(rotaryDimension)
            let inverseFrequency = Float(
                Foundation.pow(Double(theta), Double(-exponent)))
            let angle = Float(position) * inverseFrequency
            let cosine = Float(Foundation.cos(Double(angle)))
            let sine = Float(Foundation.sin(Double(angle)))
            let first = values[base + frequency]
            let second = values[base + frequency + half]
            result[base + frequency] = first * cosine - second * sine
            result[base + frequency + half] = second * cosine + first * sine
        }
    }
    return result
}

private func independentSigmoid(_ value: Float) -> Float {
    1 / (1 + Float(Foundation.exp(Double(-value))))
}

private enum FullAttentionCheckpointTarget: Sendable, CustomStringConvertible {
    case beforeSubmission(String)
    case afterSubmission(String)
    case beforeCommit

    func matches(_ checkpoint: QwenBF16FullAttentionHooks.Checkpoint) -> Bool {
        switch (self, checkpoint) {
        case let (.beforeSubmission(expected), .beforeSubmission(actual)),
             let (.afterSubmission(expected), .afterSubmission(actual)):
            return expected == actual
        case (.beforeCommit, .beforeCommit):
            return true
        default:
            return false
        }
    }

    var description: String {
        switch self {
        case let .beforeSubmission(stage): return "beforeSubmission(\(stage))"
        case let .afterSubmission(stage): return "afterSubmission(\(stage))"
        case .beforeCommit: return "beforeCommit"
        }
    }
}

private struct InjectedAttentionFailure: Error, Sendable {}

private actor AsyncCheckpointGate {
    private var entered = false
    private var entryContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func suspendUntilReleased() async {
        entered = true
        entryContinuation?.resume()
        entryContinuation = nil
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { continuation in
            entryContinuation = continuation
        }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let values = buffer.contents().assumingMemoryBound(to: Float.self)
    return Array(UnsafeBufferPointer(start: values, count: count))
}

private func expectFrozenAttentionClose(
    _ actual: [Float], _ expected: [Float], label: String
) {
    #expect(actual.count == expected.count, "\(label): count mismatch")
    for (index, pair) in zip(actual, expected).enumerated() {
        #expect(pair.0.isFinite && pair.1.isFinite,
                "\(label) index \(index): expected finite values, got \(pair.0), \(pair.1)")
        let bound = FullAttentionBF16Fixture.absoluteTolerance
            + FullAttentionBF16Fixture.relativeTolerance * abs(pair.1)
        #expect(abs(pair.0 - pair.1) <= bound,
                "\(label) index \(index): \(pair.0) != \(pair.1), bound \(bound)")
    }
}

private func maxDifference(_ left: [Float], _ right: [Float]) -> Float {
    guard left.count == right.count else { return .infinity }
    return zip(left, right).map { abs($0 - $1) }.max() ?? 0
}
