import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Test-owned BF16 matrices and FP32 vectors for the Phase 10 independent oracle.
/// Matrix words are selected from Phase 8's literal BF16 table; no runtime or
/// candidate output participates in fixture generation or expected arithmetic.
private enum QwenBF16LinearFixture {
    static let hiddenSize = 5
    static let keyHeadCount = 2
    static let valueHeadCount = 4
    static let keyHeadDimension = 2
    static let valueHeadDimension = 3
    static let keyDimension = keyHeadCount * keyHeadDimension
    static let valueDimension = valueHeadCount * valueHeadDimension
    static let convolutionChannelCount = keyDimension * 2 + valueDimension
    static let convolutionWidth = 4
    static let tokenCount = 3
    static let epsilon: Float = 1e-6

    static let qkvName = "synthetic.linear.qkv"
    static let zName = "synthetic.linear.z"
    static let bName = "synthetic.linear.b"
    static let aName = "synthetic.linear.a"
    static let outputName = "synthetic.linear.output"

    /// Frozen before the first GPU run: elementwise abs(error) <= 2e-5 +
    /// 2e-5*abs(expected). Projection dots have at most 12 terms, the recurrence
    /// has key/value widths 2/3, and this fixture uses three tokens with small,
    /// bounded state and decay. The rule covers FP32 reduction/FMA and exp/sqrt
    /// variation; it must not be widened in response to a candidate result.
    static let absoluteTolerance: Float = 2e-5
    static let relativeTolerance: Float = 2e-5

    /// Negative controls must differ from the correct CPU formulation by more
    /// than this fixture-level observability floor, independently of GPU output.
    static let negativeControlMinimum: Float = 1e-4

    static let qkvBits = qkvMatrixBits()
    static let zBits = matrixBits(rows: valueDimension, columns: hiddenSize, salt: 2)
    static let bBits = matrixBits(rows: valueHeadCount, columns: hiddenSize, salt: 3)
    static let aBits = matrixBits(rows: valueHeadCount, columns: hiddenSize, salt: 4)
    static let outputBits = matrixBits(rows: hiddenSize, columns: valueDimension, salt: 5)

    /// FP32 small vectors remain separate from the literal BF16 projection set.
    static let convolution = (0..<(convolutionChannelCount * convolutionWidth)).map {
        Float(($0 * 3) % 11 - 5) * 0.0125
    }
    static let normalization: [Float] = [0.75, 1.0, 1.25]
    static let aLog: [Float] = [-0.25, 0.0, 0.2, 0.35]
    static let timeStepBias: [Float] = [-0.1, 0.05, 0.15, -0.2]

    static let tokens: [Float] = [
        0.25, -0.5, 0.125, 0.75, -0.25,
        -0.125, 0.5, 0.25, -0.75, 0.5,
        0.75, 0.25, -0.375, 0.5, 0.125,
    ]

    static let initialState = IndependentLinearState(
        convolutionHistory: (0..<(convolutionChannelCount * convolutionWidth)).map {
            Float(($0 * 5) % 17 - 8) * 0.0025
        },
        recurrentMatrix: (0..<(valueHeadCount * keyHeadDimension * valueHeadDimension)).map {
            Float(($0 * 7) % 13 - 6) * 0.004
        })

    static func literalTensors() -> [QwenBF16LiteralTensor] {
        [
            QwenBF16LiteralTensor(name: qkvName, rows: convolutionChannelCount,
                                  columns: hiddenSize, bits: qkvBits),
            QwenBF16LiteralTensor(name: zName, rows: valueDimension,
                                  columns: hiddenSize, bits: zBits),
            QwenBF16LiteralTensor(name: bName, rows: valueHeadCount,
                                  columns: hiddenSize, bits: bBits),
            QwenBF16LiteralTensor(name: aName, rows: valueHeadCount,
                                  columns: hiddenSize, bits: aBits),
            QwenBF16LiteralTensor(name: outputName, rows: hiddenSize,
                                  columns: valueDimension, bits: outputBits),
        ]
    }

    private static func qkvMatrixBits() -> [UInt16] {
        let literal = QwenBF16TestFixture.matrixBits
        return (0..<convolutionChannelCount).flatMap { row in
            let roleSalt: Int
            if row < keyDimension {
                let head = row / keyHeadDimension
                let dimension = row % keyHeadDimension
                roleSalt = 1 + head * 5 + dimension
            } else if row < 2 * keyDimension {
                let keyRow = row - keyDimension
                let head = keyRow / keyHeadDimension
                let dimension = keyRow % keyHeadDimension
                roleSalt = 11 + head * 5 + dimension
            } else {
                let valueRow = row - 2 * keyDimension
                let head = valueRow / valueHeadDimension
                let dimension = valueRow % valueHeadDimension
                roleSalt = 17 + head * 3 + dimension
            }
            return (0..<hiddenSize).map { column in
                literal[(roleSalt * 7 + column * 11 + row * 3) % literal.count]
            }
        }
    }

    private static func matrixBits(rows: Int, columns: Int, salt: Int) -> [UInt16] {
        let literal = QwenBF16TestFixture.matrixBits
        return (0..<(rows * columns)).map { index in
            // Interleave row/column and tensor salt through the Phase 8 literal
            // table so adjacent rows and the five projections are distinguishable.
            literal[(index * 7 + (index / columns) * 3 + salt * 11) % literal.count]
        }
    }
}

private struct IndependentLinearState: Equatable {
    var convolutionHistory: [Float]
    var recurrentMatrix: [Float]
}

private struct IndependentLinearResult {
    let output: [Float]
    let finalState: IndependentLinearState
}

private struct IndependentLinearMutations {
    var moduloGroupedKeyHeads = false
    var resetConvolutionHistoryPerToken = false
    var reverseOutputRows = false
}

/// Pure CPU formulation from the linear-attention equations and literal BF16
/// words. It deliberately does not call QwenGatedDeltaNet or any candidate path.
private func independentLinearAttention(
    normalizedHidden: [Float],
    initialState: IndependentLinearState,
    mutations: IndependentLinearMutations = IndependentLinearMutations()
) -> IndependentLinearResult {
    let f = QwenBF16LinearFixture.self
    precondition(normalizedHidden.count.isMultiple(of: f.hiddenSize))
    precondition(initialState.convolutionHistory.count == f.convolutionChannelCount * f.convolutionWidth)
    precondition(initialState.recurrentMatrix.count == f.valueHeadCount
        * f.keyHeadDimension * f.valueHeadDimension)
    let tokenCount = normalizedHidden.count / f.hiddenSize

    let qkv = independentProjection(normalizedHidden, bits: f.qkvBits,
                                    rows: f.convolutionChannelCount, columns: f.hiddenSize)
    let z = independentProjection(normalizedHidden, bits: f.zBits,
                                  rows: f.valueDimension, columns: f.hiddenSize)
    let rawB = independentProjection(normalizedHidden, bits: f.bBits,
                                     rows: f.valueHeadCount, columns: f.hiddenSize)
    let rawA = independentProjection(normalizedHidden, bits: f.aBits,
                                     rows: f.valueHeadCount, columns: f.hiddenSize)

    var history = initialState.convolutionHistory
    var convolved = [Float](repeating: 0, count: qkv.count)
    for token in 0..<tokenCount {
        var tokenHistory = mutations.resetConvolutionHistoryPerToken
            ? initialState.convolutionHistory : history
        for channel in 0..<f.convolutionChannelCount {
            let stateBase = channel * f.convolutionWidth
            tokenHistory[stateBase] = tokenHistory[stateBase + 1]
            tokenHistory[stateBase + 1] = tokenHistory[stateBase + 2]
            tokenHistory[stateBase + 2] = tokenHistory[stateBase + 3]
            tokenHistory[stateBase + 3] = qkv[token * f.convolutionChannelCount + channel]
            var sum: Float = 0
            for tap in 0..<f.convolutionWidth {
                sum += tokenHistory[stateBase + tap] * f.convolution[stateBase + tap]
            }
            convolved[token * f.convolutionChannelCount + channel] = independentSiLU(sum)
        }
        history = tokenHistory
    }

    var recurrent = initialState.recurrentMatrix
    var recurrentOutput = [Float](repeating: 0, count: tokenCount * f.valueDimension)
    let headsPerKeyHead = f.valueHeadCount / f.keyHeadCount
    for token in 0..<tokenCount {
        let qkvBase = token * f.convolutionChannelCount
        for valueHead in 0..<f.valueHeadCount {
            let keyHead = mutations.moduloGroupedKeyHeads
                ? valueHead % f.keyHeadCount
                : valueHead / headsPerKeyHead
            let queryBase = qkvBase + keyHead * f.keyHeadDimension
            let keyBase = qkvBase + f.keyDimension + keyHead * f.keyHeadDimension
            let valueBase = qkvBase + f.keyDimension * 2 + valueHead * f.valueHeadDimension
            let qStateBase = valueHead * f.keyHeadDimension * f.valueHeadDimension
            let scalarBase = token * f.valueHeadCount + valueHead
            let b = independentProjectionValue(rawB[scalarBase])
            let logDecay = -Foundation.exp(Double(f.aLog[valueHead]))
                * independentSoftplus(rawA[scalarBase] + f.timeStepBias[valueHead])
            let decay = Float(Foundation.exp(logDecay))

            var qSquare: Float = 0
            var kSquare: Float = 0
            for dimension in 0..<f.keyHeadDimension {
                let qValue = convolved[queryBase + dimension]
                let kValue = convolved[keyBase + dimension]
                qSquare += qValue * qValue
                kSquare += kValue * kValue
            }
            let qScale = 1 / sqrt(qSquare + f.epsilon) / sqrt(Float(f.keyHeadDimension))
            let kScale = 1 / sqrt(kSquare + f.epsilon)

            for keyDimension in 0..<f.keyHeadDimension {
                for valueDimension in 0..<f.valueHeadDimension {
                    recurrent[qStateBase + keyDimension * f.valueHeadDimension + valueDimension]
                        *= decay
                }
            }
            for valueDimension in 0..<f.valueHeadDimension {
                var prediction: Float = 0
                for keyDimension in 0..<f.keyHeadDimension {
                    let stateIndex = qStateBase + keyDimension * f.valueHeadDimension + valueDimension
                    prediction += recurrent[stateIndex]
                        * convolved[keyBase + keyDimension] * kScale
                }
                let delta = (convolved[valueBase + valueDimension] - prediction) * b
                for keyDimension in 0..<f.keyHeadDimension {
                    let stateIndex = qStateBase + keyDimension * f.valueHeadDimension + valueDimension
                    recurrent[stateIndex] += convolved[keyBase + keyDimension] * kScale * delta
                }
            }
            for valueDimension in 0..<f.valueHeadDimension {
                var sum: Float = 0
                for keyDimension in 0..<f.keyHeadDimension {
                    let stateIndex = qStateBase + keyDimension * f.valueHeadDimension + valueDimension
                    sum += recurrent[stateIndex] * convolved[queryBase + keyDimension] * qScale
                }
                recurrentOutput[token * f.valueDimension
                    + valueHead * f.valueHeadDimension + valueDimension] = sum
            }
        }
    }

    var gated = [Float](repeating: 0, count: recurrentOutput.count)
    for token in 0..<tokenCount {
        for head in 0..<f.valueHeadCount {
            let base = token * f.valueDimension + head * f.valueHeadDimension
            var squareSum: Float = 0
            for dimension in 0..<f.valueHeadDimension {
                squareSum += recurrentOutput[base + dimension] * recurrentOutput[base + dimension]
            }
            let inverseRMS = 1 / sqrt(squareSum / Float(f.valueHeadDimension) + f.epsilon)
            for dimension in 0..<f.valueHeadDimension {
                gated[base + dimension] = recurrentOutput[base + dimension] * inverseRMS
                    * f.normalization[dimension] * independentSiLU(z[base + dimension])
            }
        }
    }

    var output = independentProjection(gated, bits: f.outputBits,
                                       rows: f.hiddenSize, columns: f.valueDimension)
    if mutations.reverseOutputRows {
        let normal = output
        for token in 0..<tokenCount {
            for row in 0..<f.hiddenSize {
                output[token * f.hiddenSize + row] = normal[token * f.hiddenSize
                    + (f.hiddenSize - 1 - row)]
            }
        }
    }
    return IndependentLinearResult(
        output: output,
        finalState: IndependentLinearState(
            convolutionHistory: history, recurrentMatrix: recurrent))
}

private func independentProjection(
    _ input: [Float], bits: [UInt16], rows: Int, columns: Int
) -> [Float] {
    let tokenCount = input.count / columns
    var output = [Float](repeating: 0, count: tokenCount * rows)
    for token in 0..<tokenCount {
        for row in 0..<rows {
            var sum: Float = 0
            for column in 0..<columns {
                let weight = QwenBF16TestFixture.float(bits[row * columns + column])
                sum += input[token * columns + column] * weight
            }
            output[token * rows + row] = sum
        }
    }
    return output
}

private func independentProjectionValue(_ value: Float) -> Float {
    1 / (1 + Float(Foundation.exp(Double(-value))))
}

private func independentSiLU(_ value: Float) -> Float {
    value * independentProjectionValue(value)
}

private func independentSoftplus(_ value: Float) -> Double {
    if value > 20 { return Double(value) }
    return log1p(Foundation.exp(Double(value)))
}

private func maxLinearDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
    guard lhs.count == rhs.count else { return .infinity }
    return zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
}

@Suite(.serialized) struct QwenBF16LinearAttentionTests {
    @Test func literalFixtureCoversFiveBF16ProjectionShapesDistinctHeadsAndFP32Vectors() {
        let f = QwenBF16LinearFixture.self
        let tensors = f.literalTensors()
        #expect(tensors.map(\.name) == [f.qkvName, f.zName, f.bName, f.aName, f.outputName])
        #expect(tensors.map(\.rows) == [20, 12, 4, 4, 5])
        #expect(tensors.map(\.columns) == [5, 5, 5, 5, 12])
        #expect(tensors.allSatisfy { $0.bits.count == $0.rows * $0.columns })
        let firstKeyHeadStart = f.keyDimension * f.hiddenSize
        let keyHeadRowCount = f.keyHeadDimension * f.hiddenSize
        let firstKeyHeadRange = firstKeyHeadStart..<(firstKeyHeadStart + keyHeadRowCount)
        let secondKeyHeadStart = firstKeyHeadStart + keyHeadRowCount
        let secondKeyHeadRange = secondKeyHeadStart..<(secondKeyHeadStart + keyHeadRowCount)
        let firstKeyHead = Array(f.qkvBits[firstKeyHeadRange])
        let secondKeyHead = Array(f.qkvBits[secondKeyHeadRange])
        #expect(firstKeyHead != secondKeyHead,
                "fixture key heads must exercise grouped-head mapping")
        #expect(f.convolution.count == f.convolutionChannelCount * f.convolutionWidth)
        #expect(f.normalization.count == f.valueHeadDimension)
        #expect(f.aLog.count == f.valueHeadCount)
        #expect(f.timeStepBias.count == f.valueHeadCount)
        #expect(f.tokens.count == f.tokenCount * f.hiddenSize)
        let fp32VectorsAreFinite = (f.convolution + f.normalization + f.aLog + f.timeStepBias)
            .allSatisfy { value in value.isFinite }
        #expect(fp32VectorsAreFinite)
    }

    @Test func independentCPUFormulationHasObservableMultiTokenControls() {
        let f = QwenBF16LinearFixture.self
        let correct = independentLinearAttention(
            normalizedHidden: f.tokens, initialState: f.initialState)
        let wrongHeadGrouping = independentLinearAttention(
            normalizedHidden: f.tokens, initialState: f.initialState,
            mutations: IndependentLinearMutations(moduloGroupedKeyHeads: true))
        let resetHistory = independentLinearAttention(
            normalizedHidden: f.tokens, initialState: f.initialState,
            mutations: IndependentLinearMutations(resetConvolutionHistoryPerToken: true))
        let reversedOutputRows = independentLinearAttention(
            normalizedHidden: f.tokens, initialState: f.initialState,
            mutations: IndependentLinearMutations(reverseOutputRows: true))

        #expect(correct.output.count == f.tokenCount * f.hiddenSize)
        #expect(correct.finalState.convolutionHistory.count
            == f.convolutionChannelCount * f.convolutionWidth)
        #expect(correct.finalState.recurrentMatrix.count
            == f.valueHeadCount * f.keyHeadDimension * f.valueHeadDimension)
        let outputIsFinite = correct.output.allSatisfy { value in value.isFinite }
        let convolutionHistoryIsFinite = correct.finalState.convolutionHistory
            .allSatisfy { value in value.isFinite }
        let recurrentMatrixIsFinite = correct.finalState.recurrentMatrix
            .allSatisfy { value in value.isFinite }
        #expect(outputIsFinite)
        #expect(convolutionHistoryIsFinite)
        #expect(recurrentMatrixIsFinite)
        #expect(maxLinearDifference(correct.output, wrongHeadGrouping.output)
            > f.negativeControlMinimum)
        #expect(maxLinearDifference(correct.output, resetHistory.output)
            > f.negativeControlMinimum)
        #expect(maxLinearDifference(correct.output, reversedOutputRows.output)
            > f.negativeControlMinimum)
    }

    @Test func nonzeroPositionCloneRestoreAndResetToZeroAreAtomic() throws {
        let fixture = QwenBF16LinearFixture.self
        let context = try MetalContext()
        let state = try makeLinearState(context: context, position: 23)
        let before = try state.layerSnapshot(QwenBF16LinearTestSupport.layer)
        let clone = try state.clone()
        #expect(clone.positions[QwenBF16LinearTestSupport.layer] == 23)

        try state.reset()
        let reset = try state.layerSnapshot(QwenBF16LinearTestSupport.layer)
        #expect(reset.position == 0)
        #expect(reset.state.convolutionHistory.allSatisfy { $0 == 0 })
        #expect(reset.state.recurrentMatrix.allSatisfy { $0 == 0 })

        try state.restore(clone)
        let restored = try state.layerSnapshot(QwenBF16LinearTestSupport.layer)
        #expect(restored.position == 23)
        #expect(restored.state == before.state)

        // A legacy snapshot without positions deliberately restores position 0.
        let legacy = QwenLinearAttentionSnapshot(
            geometry: clone.geometry, layers: clone.layers)
        try state.restore(legacy)
        let legacyRestored = try state.layerSnapshot(QwenBF16LinearTestSupport.layer)
        #expect(legacyRestored.position == 0)
        #expect(legacyRestored.state == before.state)

        try state.restore(clone)
        try state.reset()
        let finalReset = try state.layerSnapshot(QwenBF16LinearTestSupport.layer)
        #expect(finalReset.position == 0)
        #expect(finalReset.state.convolutionHistory.allSatisfy { $0 == 0 })
        #expect(finalReset.state.recurrentMatrix.allSatisfy { $0 == 0 })
        #expect(fixture.initialState.convolutionHistory.contains { $0 != 0 })
    }

    @Test func actualGPUProjectionConvolutionRecurrenceGatedNormAndOutputMatchCPU() async throws {
        let fixture = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: fixture.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        assertResidentLinearBF16(weights)

        let batchState = try makeLinearState(context: context)
        let batchStep = try makeLinearStep(
            context: context, weights: weights, state: batchState)
        let expectedBatch = independentLinearAttention(
            normalizedHidden: fixture.tokens, initialState: fixture.initialState)
        let batchOutput = try await batchStep.append(
            normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)
        expectLinearClose(batchOutput, expectedBatch.output, label: "three-token GPU output")
        let batchSnapshot = try await batchStep.snapshot()
        #expect(batchSnapshot.position == 7 + fixture.tokenCount)
        expectLinearStateClose(batchSnapshot.state, expectedBatch.finalState,
                               label: "three-token committed state")

        let wrongHeadGrouping = independentLinearAttention(
            normalizedHidden: fixture.tokens, initialState: fixture.initialState,
            mutations: IndependentLinearMutations(moduloGroupedKeyHeads: true))
        let resetHistory = independentLinearAttention(
            normalizedHidden: fixture.tokens, initialState: fixture.initialState,
            mutations: IndependentLinearMutations(resetConvolutionHistoryPerToken: true))
        let reversedOutputRows = independentLinearAttention(
            normalizedHidden: fixture.tokens, initialState: fixture.initialState,
            mutations: IndependentLinearMutations(reverseOutputRows: true))
        #expect(maxLinearDifference(batchOutput, wrongHeadGrouping.output)
            > fixture.negativeControlMinimum)
        #expect(maxLinearDifference(batchOutput, resetHistory.output)
            > fixture.negativeControlMinimum)
        #expect(maxLinearDifference(batchOutput, reversedOutputRows.output)
            > fixture.negativeControlMinimum)

        // A second actor with identical committed input state submits the same
        // tokens one at a time. Its per-token outputs and snapshots must agree
        // with the independent sequential CPU recurrence and the batched result.
        let sequentialState = try makeLinearState(context: context)
        let sequentialStep = try makeLinearStep(
            context: context, weights: weights, state: sequentialState)
        var cpuState = fixture.initialState
        var sequentialOutput: [Float] = []
        for token in 0..<fixture.tokenCount {
            let start = token * fixture.hiddenSize
            let hidden = Array(fixture.tokens[start..<(start + fixture.hiddenSize)])
            let expected = independentLinearAttention(
                normalizedHidden: hidden, initialState: cpuState)
            let actual = try await sequentialStep.append(
                normalizedHidden: hidden, tokenCount: 1)
            expectLinearClose(actual, expected.output, label: "sequential token \(token)")
            sequentialOutput += actual
            cpuState = expected.finalState
            let snapshot = try await sequentialStep.snapshot()
            #expect(snapshot.position == 7 + token + 1)
            expectLinearStateClose(snapshot.state, cpuState,
                                   label: "sequential state \(token)")
        }
        expectLinearClose(sequentialOutput, expectedBatch.output,
                          label: "batched versus sequential output")
        let sequentialFinal = try await sequentialStep.snapshot()
        expectLinearStateClose(sequentialFinal.state, expectedBatch.finalState,
                               label: "batched versus sequential final state")
    }

    @Test func unobservedBatchedPathMatchesActivationObservedPathAndPreservesTraceBoundaries() async throws {
        let fixture = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: fixture.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        let expected = independentLinearAttention(
            normalizedHidden: fixture.tokens, initialState: fixture.initialState)

        let unobservedState = try makeLinearState(context: context)
        let unobserved = try makeLinearStep(
            context: context, weights: weights, state: unobservedState)
        let unobservedOutput = try await unobserved.append(
            normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)

        let trace = LinearActivationTraceRecorder()
        let observedState = try makeLinearState(context: context)
        let observed = try makeLinearStep(
            context: context, weights: weights, state: observedState,
            hooks: QwenBF16LinearHooks(observeActivation: { _, stage, values in
                trace.record(stage: stage, count: values.count)
            }))
        let observedOutput = try await observed.append(
            normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)

        expectLinearClose(unobservedOutput, expected.output,
                          label: "unobserved batched output")
        expectLinearClose(observedOutput, expected.output,
                          label: "activation-observed output")
        expectLinearClose(unobservedOutput, observedOutput,
                          label: "batched versus activation-observed output")
        let unobservedSnapshot = try await unobserved.snapshot()
        let observedSnapshot = try await observed.snapshot()
        expectLinearStateClose(unobservedSnapshot.state, expected.finalState,
                               label: "unobserved batched state")
        expectLinearStateClose(observedSnapshot.state, expected.finalState,
                               label: "activation-observed state")
        #expect(unobservedSnapshot == observedSnapshot)

        let records = trace.snapshot()
        #expect(records.map(\.stage) == [
            "input", "qkv", "z", "b", "a", "convolved",
            "query-raw", "key-raw", "value", "beta", "log-decay",
            "core", "gated", "output",
        ])
        #expect(records.allSatisfy { $0.count > 0 })
    }

    @Test func injectedFailureAtEverySubmissionBoundaryRollsBackAndAllowsReuse() async throws {
        let fixture = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: fixture.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)

        for target in LinearCheckpointTarget.all {
            let state = try makeLinearState(context: context)
            let failure = OneShotLinearFailure(target: target)
            let hooks = QwenBF16LinearHooks(checkpoint: { checkpoint in
                try await failure.checkpoint(checkpoint)
            })
            let step = try makeLinearStep(
                context: context, weights: weights, state: state, hooks: hooks)
            let before = try await step.snapshot()
            var injected = false
            do {
                _ = try await step.append(
                    normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)
            } catch is InjectedLinearFailure {
                injected = true
            } catch {
                Issue.record("Unexpected failure at \(target): \(error)")
            }
            #expect(injected, "checkpoint did not inject at \(target)")
            let afterFailure = try await step.snapshot()
            #expect(afterFailure == before,
                    "failure at \(target) partially committed state or position")

            // The same actor and state owner must accept a fresh append after
            // both pre-submit aborts and fully settled post-submit failures.
            let expectedRetry = independentLinearAttention(
                normalizedHidden: fixture.tokens, initialState: fixture.initialState)
            let retry = try await step.append(
                normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)
            expectLinearClose(retry, expectedRetry.output,
                              label: "retry after \(target)")
            let afterRetry = try await step.snapshot()
            #expect(afterRetry.position == before.position + fixture.tokenCount)
            expectLinearStateClose(afterRetry.state, expectedRetry.finalState,
                                   label: "reusable state after \(target)")
        }
    }

    @Test func cancellationAtEverySubmissionBoundaryRollsBackAndAllowsReuse() async throws {
        let fixture = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: fixture.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)

        for target in LinearCheckpointTarget.all {
            let state = try makeLinearState(context: context)
            let gate = AsyncLinearCheckpointGate()
            let hooks = QwenBF16LinearHooks(checkpoint: { checkpoint in
                if target.matches(checkpoint) { await gate.suspendUntilReleased() }
            })
            let step = try makeLinearStep(
                context: context, weights: weights, state: state, hooks: hooks)
            let before = try await step.snapshot()
            let append = Task {
                try await step.append(
                    normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)
            }
            await gate.waitUntilEntered()
            append.cancel()
            await gate.release()

            var cancelled = false
            do {
                _ = try await append.value
            } catch is CancellationError {
                cancelled = true
            } catch {
                Issue.record("Unexpected cancellation result at \(target): \(error)")
            }
            #expect(cancelled, "cancellation was not observed at \(target)")
            let afterCancellation = try await step.snapshot()
            #expect(afterCancellation == before,
                    "cancellation at \(target) partially committed state or position")

            let expectedRetry = independentLinearAttention(
                normalizedHidden: fixture.tokens, initialState: fixture.initialState)
            let retry = try await step.append(
                normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)
            expectLinearClose(retry, expectedRetry.output,
                              label: "retry after cancellation at \(target)")
            let afterRetry = try await step.snapshot()
            #expect(afterRetry.position == before.position + fixture.tokenCount)
            expectLinearStateClose(afterRetry.state, expectedRetry.finalState,
                                   label: "reusable state after cancellation at \(target)")
        }
    }

    @Test func rejectsMalformedInputStateGeometryAndFiniteViolationsWithoutMutation() async throws {
        let fixture = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: fixture.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        let state = try makeLinearState(context: context)
        let step = try makeLinearStep(context: context, weights: weights, state: state)
        let before = try await step.snapshot()

        let malformedInputs: [([Float], Int)] = [
            ([], 0),
            ([Float](repeating: 0, count: 257 * fixture.hiddenSize), 257),
            (Array(fixture.tokens.prefix(fixture.hiddenSize - 1)), 1),
            ([Float.nan] + Array(fixture.tokens.prefix(fixture.hiddenSize - 1)), 1),
        ]
        for (input, tokenCount) in malformedInputs {
            let rejected = await hasInvalidLinearInput(step, input: input, tokenCount: tokenCount)
            #expect(rejected, "accepted malformed input with tokenCount \(tokenCount)")
            let after = try await step.snapshot()
            #expect(after == before, "invalid input changed committed state")
        }

        let wrongGeometry = try QwenLinearAttentionGeometry(
            convolutionWidth: 4,
            convolutionChannelCount: fixture.convolutionChannelCount + 1,
            valueHeadCount: fixture.valueHeadCount,
            keyHeadDimension: fixture.keyHeadDimension,
            valueHeadDimension: fixture.valueHeadDimension)
        let wrongStateSeed = IndependentLinearState(
            convolutionHistory: [Float](repeating: 0, count: wrongGeometry.convolutionElementCount),
            recurrentMatrix: [Float](repeating: 0, count: wrongGeometry.recurrentElementCount))
        let wrongState = try makeLinearState(
            context: context, initial: wrongStateSeed, geometry: wrongGeometry)
        let wrongStateRejected = throwsAnyLinearError {
            _ = try makeLinearStep(context: context, weights: weights, state: wrongState)
        }
        #expect(wrongStateRejected)

        let names = QwenBF16LinearNames(
            qkv: "missing.qkv", z: fixture.zName, b: fixture.bName,
            a: fixture.aName, output: fixture.outputName)
        let missingNameRejected = throwsAnyLinearError {
            _ = try makeLinearStep(context: context, weights: weights, state: state, names: names)
        }
        #expect(missingNameRejected)
        let shortVectors = QwenBF16LinearVectors(
            convolution: Array(fixture.convolution.dropLast()),
            normalization: fixture.normalization, aLog: fixture.aLog,
            timeStepBias: fixture.timeStepBias)
        let shortVectorsRejected = throwsAnyLinearError {
            _ = try makeLinearStep(context: context, weights: weights, state: state,
                                   vectors: shortVectors)
        }
        #expect(shortVectorsRejected)
        var nonfiniteALog = fixture.aLog
        nonfiniteALog[0] = .greatestFiniteMagnitude
        let nonfiniteVectors = QwenBF16LinearVectors(
            convolution: fixture.convolution, normalization: fixture.normalization,
            aLog: nonfiniteALog, timeStepBias: fixture.timeStepBias)
        let nonfiniteVectorsRejected = throwsAnyLinearError {
            _ = try makeLinearStep(context: context, weights: weights, state: state,
                                   vectors: nonfiniteVectors)
        }
        #expect(nonfiniteVectorsRejected)

        let overflowState = try makeLinearState(context: context, position: Int.max)
        let overflowStep = try makeLinearStep(
            context: context, weights: weights, state: overflowState)
        let beforeOverflow = try await overflowStep.snapshot()
        let overflowRejected = await hasInvalidLinearInput(
            overflowStep, input: Array(fixture.tokens.prefix(fixture.hiddenSize)), tokenCount: 1)
        #expect(overflowRejected, "position addition overflow was accepted")
        let afterOverflow = try await overflowStep.snapshot()
        #expect(afterOverflow == beforeOverflow)

        let invalidDivisibility = throwsAnyLinearError {
            _ = try QwenGatedDeltaNetConfiguration(
                hiddenSize: fixture.hiddenSize, keyHeadCount: 2, valueHeadCount: 3,
                keyHeadDimension: fixture.keyHeadDimension,
                valueHeadDimension: fixture.valueHeadDimension)
        }
        #expect(invalidDivisibility)
        let invalidProduct = throwsAnyLinearError {
            _ = try QwenGatedDeltaNetConfiguration(
                hiddenSize: fixture.hiddenSize,
                keyHeadCount: Int(UInt32.max), valueHeadCount: Int(UInt32.max),
                keyHeadDimension: Int(UInt32.max), valueHeadDimension: 1)
        }
        #expect(invalidProduct)
    }

    @Test func reentrantAppendDuringSubmittedProjectionIsRejectedWithoutDoubleCommit() async throws {
        let fixture = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: fixture.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        let state = try makeLinearState(context: context)
        let gate = AsyncLinearCheckpointGate()
        let hooks = QwenBF16LinearHooks(checkpoint: { checkpoint in
            if case .afterSubmission(stage: "projections") = checkpoint {
                await gate.suspendUntilReleased()
            }
        })
        let step = try makeLinearStep(
            context: context, weights: weights, state: state, hooks: hooks)
        let first = Task {
            try await step.append(
                normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)
        }
        await gate.waitUntilEntered()

        var reentrantRejected = false
        do {
            _ = try await step.append(
                normalizedHidden: fixture.tokens, tokenCount: fixture.tokenCount)
        } catch let error as QwenTextRunnerError {
            if case .operationInProgress = error { reentrantRejected = true }
        }
        #expect(reentrantRejected)
        let committedPositionDuringProjection = try state.committedPosition(
            layer: QwenBF16LinearTestSupport.layer)
        let committedStateDuringProjection = try state.layerState(
            QwenBF16LinearTestSupport.layer)
        #expect(committedPositionDuringProjection == 7)
        expectLinearStateClose(committedStateDuringProjection, fixture.initialState,
                               label: "committed state while projection is submitted",
                               absoluteTolerance: 0, relativeTolerance: 0)

        await gate.release()
        let firstOutput = try await first.value
        let expected = independentLinearAttention(
            normalizedHidden: fixture.tokens, initialState: fixture.initialState)
        expectLinearClose(firstOutput, expected.output, label: "first reentrant append")
        let finalSnapshot = try await step.snapshot()
        #expect(finalSnapshot.position == 7 + fixture.tokenCount)
        expectLinearStateClose(finalSnapshot.state, expected.finalState,
                               label: "single committed reentrant append")
    }
}

extension QwenBF16LinearAttentionTests {
    @Test func gpuPreparationReplacesTwoSettledCommandsWithOne() async throws {
        let f = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: f.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        for prepared in [false, true] {
            let step = try makeLinearStep(context: context, weights: weights,
                                          state: makeLinearState(context: context))
            let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
            #expect(capture.beginQwenCacheMaps(layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
            #expect(capture.beginQwenProductionTiming())
            capture.setQwenCacheMapPhase(.decode)
            let location = try #require(capture.qwenProductionLocation(position: 7, tokenCount: 1, forward: true))
            _ = try await QwenProductionTimingMeasurement.$location.withValue(location) {
                try await step.append(normalizedHidden: Array(f.tokens.prefix(f.hiddenSize)),
                                      tokenCount: 1, useGPUPreparation: prepared)
            }
            capture.endQwenProductionTiming()
            capture.finish(status: 0)
            struct Batch: Decodable { let records: [[UInt64]] }
            var rows: [[UInt64]] = []
            while let batch = capture.drainJSONBatch(maximumBytes: RuntimeMeasurementCapture.maximumJSONBatchBytes) {
                rows += try JSONDecoder().decode(Batch.self, from: batch.data).records
            }
            let commands = rows.filter { $0.count == 6 && $0[0] == 161 }
            let waits = rows.filter { $0.count == 6 && $0[0] == 162 }
            #expect(commands.count == (prepared ? 1 : 2))
            #expect(waits.count == commands.count)
            #expect(commands.map { $0[2] } == (prepared
                ? [QwenProductionStage.linearPreparedStep.rawValue]
                : [QwenProductionStage.linearProjectionsConvolution.rawValue,
                   QwenProductionStage.linearRecurrenceOutput.rawValue]))
        }
    }

    @Test func gpuPreparationCombinedPathMatchesCPUAcrossCommittedStepsExactly() async throws {
        let f = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: f.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        for initialPosition in [0, 7] {
            let cpu = try makeLinearStep(context: context, weights: weights,
                state: makeLinearState(context: context, position: initialPosition))
            let gpu = try makeLinearStep(context: context, weights: weights,
                state: makeLinearState(context: context, position: initialPosition))
            for count in [1, 3, 1, 3] {
                let input = Array(f.tokens.prefix(count * f.hiddenSize))
                let expected = try await cpu.append(normalizedHidden: input, tokenCount: count)
                let actual = try await gpu.append(normalizedHidden: input, tokenCount: count,
                                                  useGPUPreparation: true)
                #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
                let a = try await gpu.snapshot(), e = try await cpu.snapshot()
                #expect(a.position == e.position)
                #expect(a.state.convolutionHistory.map(\.bitPattern) == e.state.convolutionHistory.map(\.bitPattern))
                #expect(a.state.recurrentMatrix.map(\.bitPattern) == e.state.recurrentMatrix.map(\.bitPattern))
            }
        }
    }

    @Test func gpuPreparationCombinedFailuresAndCancellationPreserveStateAndPermitRetry() async throws {
        let f = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: f.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        let stage = "projections-convolution-preparation-recurrence-output"
        let targets: [LinearCheckpointTarget] = [.beforeSubmission(stage), .afterSubmission(stage), .beforeCommit]
        for target in targets {
            for cancellation in [false, true] {
                let step = try makeLinearStep(context: context, weights: weights,
                                              state: makeLinearState(context: context))
                let before = try await step.snapshot()
                let failure = OneShotLinearFailure(target: target)
                let gate = AsyncLinearCheckpointGate()
                let hooks = QwenBF16LinearPreparationHooks(checkpoint: { checkpoint in
                    if cancellation {
                        if target.matches(checkpoint) { await gate.suspendUntilReleased() }
                    } else { try await failure.checkpoint(checkpoint) }
                })
                let operation = Task {
                    try await step.append(normalizedHidden: f.tokens, tokenCount: f.tokenCount,
                                          useGPUPreparation: true, preparationHooks: hooks)
                }
                if cancellation {
                    await gate.waitUntilEntered()
                    operation.cancel()
                    await gate.release()
                }
                var rejected = false
                do { _ = try await operation.value }
                catch is CancellationError { rejected = cancellation }
                catch is InjectedLinearFailure { rejected = !cancellation }
                catch { Issue.record("Unexpected fused failure: \(error)") }
                #expect(rejected)
                #expect(try await step.snapshot() == before)
                let control = try makeLinearStep(context: context, weights: weights,
                                                 state: makeLinearState(context: context))
                let expected = try await control.append(normalizedHidden: f.tokens, tokenCount: f.tokenCount)
                let retry = try await step.append(normalizedHidden: f.tokens, tokenCount: f.tokenCount,
                                                 useGPUPreparation: true)
                #expect(retry.map(\.bitPattern) == expected.map(\.bitPattern))
                let actualState = try await step.snapshot()
                let expectedState = try await control.snapshot()
                #expect(actualState == expectedState)
            }
        }
    }

    @Test func gpuPreparationRejectsActualNonfiniteProjectionsAndDecayBeforePublication() async throws {
        let f = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: f.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        for decayOverflow in [false, true] {
            let original = QwenBF16LinearTestSupport.vectors
            let vectors = decayOverflow ? QwenBF16LinearVectors(
                convolution: original.convolution, normalization: original.normalization,
                aLog: Array(repeating: 88, count: f.valueHeadCount),
                timeStepBias: Array(repeating: 0, count: f.valueHeadCount)) : original
            let step = try makeLinearStep(context: context, weights: weights,
                state: makeLinearState(context: context), vectors: vectors)
            let before = try await step.snapshot()
            let invalid = decayOverflow ? Array(f.tokens.prefix(f.hiddenSize)).map { $0 * 1000 }
                : Array(repeating: Float.greatestFiniteMagnitude, count: f.hiddenSize)
            var rejected = false
            do { _ = try await step.append(normalizedHidden: invalid, tokenCount: 1, useGPUPreparation: true) }
            catch QwenTextRunnerError.invalidState(let detail) {
                rejected = true
                #expect(detail.contains(decayOverflow ? "recurrence inputs" : "projections"))
            }
            #expect(rejected)
            #expect(try await step.snapshot() == before)
            let control = try makeLinearStep(context: context, weights: weights,
                state: makeLinearState(context: context), vectors: vectors)
            let safe = Array(repeating: Float.zero, count: f.hiddenSize)
            let expected = try await control.append(normalizedHidden: safe, tokenCount: 1)
            let actual = try await step.append(normalizedHidden: safe, tokenCount: 1, useGPUPreparation: true)
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
            let actualState = try await step.snapshot()
                let expectedState = try await control.snapshot()
                #expect(actualState == expectedState)
        }
    }

    @Test func gpuPreparationOptInPreservesObservedFallbackStages() async throws {
        let f = QwenBF16LinearFixture.self
        let source = try QwenBF16SyntheticSource.make(tensors: f.literalTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeLinearWeights(context: context, source: source)
        let trace = LinearActivationTraceRecorder()
        let step = try makeLinearStep(context: context, weights: weights,
            state: makeLinearState(context: context), hooks: QwenBF16LinearHooks(
                observeActivation: { _, stage, values in trace.record(stage: stage, count: values.count) }))
        _ = try await step.append(normalizedHidden: f.tokens, tokenCount: f.tokenCount,
            useGPUPreparation: true, preparationHooks: QwenBF16LinearPreparationHooks(checkpoint: { _ in
                Issue.record("Fused checkpoint must not execute in observed fallback")
            }))
        #expect(trace.snapshot().map(\.stage) == ["input", "qkv", "z", "b", "a", "convolved",
            "query-raw", "key-raw", "value", "beta", "log-decay", "core", "gated", "output"])
    }
}

private enum QwenBF16LinearTestSupport {
    static let layer = 2
    static let initialPosition = 7

    static var configuration: QwenGatedDeltaNetConfiguration {
        get throws {
            try QwenGatedDeltaNetConfiguration(
                hiddenSize: QwenBF16LinearFixture.hiddenSize,
                keyHeadCount: QwenBF16LinearFixture.keyHeadCount,
                valueHeadCount: QwenBF16LinearFixture.valueHeadCount,
                keyHeadDimension: QwenBF16LinearFixture.keyHeadDimension,
                valueHeadDimension: QwenBF16LinearFixture.valueHeadDimension,
                epsilon: QwenBF16LinearFixture.epsilon)
        }
    }

    static var geometry: QwenLinearAttentionGeometry {
        get throws {
            try QwenLinearAttentionGeometry(
                convolutionWidth: QwenBF16LinearFixture.convolutionWidth,
                convolutionChannelCount: QwenBF16LinearFixture.convolutionChannelCount,
                valueHeadCount: QwenBF16LinearFixture.valueHeadCount,
                keyHeadDimension: QwenBF16LinearFixture.keyHeadDimension,
                valueHeadDimension: QwenBF16LinearFixture.valueHeadDimension)
        }
    }

    static var names: QwenBF16LinearNames {
        QwenBF16LinearNames(qkv: QwenBF16LinearFixture.qkvName,
                            z: QwenBF16LinearFixture.zName,
                            b: QwenBF16LinearFixture.bName,
                            a: QwenBF16LinearFixture.aName,
                            output: QwenBF16LinearFixture.outputName)
    }

    static var vectors: QwenBF16LinearVectors {
        QwenBF16LinearVectors(convolution: QwenBF16LinearFixture.convolution,
                              normalization: QwenBF16LinearFixture.normalization,
                              aLog: QwenBF16LinearFixture.aLog,
                              timeStepBias: QwenBF16LinearFixture.timeStepBias)
    }
}

private func makeLinearWeights(
    context: MetalContext, source: QwenBF16SyntheticSource
) throws -> QwenBF16Weights {
    let fixture = QwenBF16LinearFixture.self
    let specifications = [
        QwenBF16TensorSpec(name: fixture.qkvName, shardName: source.shardName,
                           role: .dense, rows: fixture.convolutionChannelCount,
                           columns: fixture.hiddenSize),
        QwenBF16TensorSpec(name: fixture.zName, shardName: source.shardName,
                           role: .dense, rows: fixture.valueDimension,
                           columns: fixture.hiddenSize),
        QwenBF16TensorSpec(name: fixture.bName, shardName: source.shardName,
                           role: .dense, rows: fixture.valueHeadCount,
                           columns: fixture.hiddenSize),
        QwenBF16TensorSpec(name: fixture.aName, shardName: source.shardName,
                           role: .dense, rows: fixture.valueHeadCount,
                           columns: fixture.hiddenSize),
        QwenBF16TensorSpec(name: fixture.outputName, shardName: source.shardName,
                           role: .dense, rows: fixture.hiddenSize,
                           columns: fixture.valueDimension),
    ]
    let residentBytes = specifications.reduce(UInt64(0)) { total, spec in
        total + UInt64(spec.rows * spec.columns * MemoryLayout<UInt16>.stride)
    }
    return try QwenBF16Weights(
        context: context, source: source.handle, specifications: specifications,
        residencyBudget: residentBytes, maximumChunkBytes: 43,
        checkpoint: { _ in })
}

private func makeLinearState(
    context: MetalContext,
    position: Int = QwenBF16LinearTestSupport.initialPosition,
    initial: IndependentLinearState = QwenBF16LinearFixture.initialState,
    geometry: QwenLinearAttentionGeometry? = nil
) throws -> QwenLinearAttentionState {
    let layer = QwenBF16LinearTestSupport.layer
    let selectedGeometry: QwenLinearAttentionGeometry
    if let geometry {
        selectedGeometry = geometry
    } else {
        selectedGeometry = try QwenBF16LinearTestSupport.geometry
    }
    var mask = [UInt8](repeating: 0, count: layer + 1)
    mask[layer] = 1
    let state = try QwenLinearAttentionState(
        device: context.device, linearAttentionLayerMask: mask,
        geometry: selectedGeometry, expectedLinearLayerCount: 1)
    let layerState = QwenLinearAttentionLayerState(
        convolutionHistory: initial.convolutionHistory,
        recurrentMatrix: initial.recurrentMatrix)
    try state.restore(QwenLinearAttentionSnapshot(
        geometry: selectedGeometry, layers: [layer: layerState],
        positions: [layer: position]))
    return state
}

private func makeLinearStep(
    context: MetalContext,
    weights: QwenBF16Weights,
    state: QwenLinearAttentionState,
    hooks: QwenBF16LinearHooks = .none,
    names: QwenBF16LinearNames = QwenBF16LinearTestSupport.names,
    vectors: QwenBF16LinearVectors = QwenBF16LinearTestSupport.vectors
) throws -> QwenBF16LinearStep {
    let configuration = try QwenBF16LinearTestSupport.configuration
    return try QwenBF16LinearStep(
        context: context, weights: weights, names: names,
        configuration: configuration, vectors: vectors,
        layer: QwenBF16LinearTestSupport.layer, state: state, hooks: hooks)
}

private func assertResidentLinearBF16(_ weights: QwenBF16Weights) {
    let fixture = QwenBF16LinearFixture.self
    let expected: [(String, [UInt16])] = [
        (fixture.qkvName, fixture.qkvBits), (fixture.zName, fixture.zBits),
        (fixture.bName, fixture.bBits), (fixture.aName, fixture.aBits),
        (fixture.outputName, fixture.outputBits),
    ]
    for (name, bits) in expected {
        let chunks = weights.inspectedChunks.filter { $0.name == name }
        #expect(!chunks.isEmpty, "missing resident BF16 chunks for \(name)")
        #expect(chunks.allSatisfy { $0.buffer.storageMode == .shared })
        let residentWords = chunks.flatMap { readResidentBF16Words($0.buffer) }
        #expect(residentWords == bits, "resident BF16 words changed for \(name)")
    }
}

private func readResidentBF16Words(_ buffer: MTLBuffer) -> [UInt16] {
    let words = buffer.contents().assumingMemoryBound(to: UInt16.self)
    return Array(UnsafeBufferPointer(
        start: words, count: buffer.length / MemoryLayout<UInt16>.stride))
}

private enum LinearCheckpointTarget: Sendable, CustomStringConvertible {
    case beforeSubmission(String)
    case afterSubmission(String)
    case beforeCommit

    static let all: [LinearCheckpointTarget] = {
        let stages = ["projections", "convolution", "recurrence", "output"]
        return stages.flatMap { [.beforeSubmission($0), .afterSubmission($0)] }
            + [.beforeCommit]
    }()

    func matches(_ checkpoint: QwenBF16LinearHooks.Checkpoint) -> Bool {
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

private struct InjectedLinearFailure: Error, Sendable {}

private final class LinearActivationTraceRecorder: @unchecked Sendable {
    struct Record: Equatable, Sendable {
        let stage: String
        let count: Int
    }

    private let lock = NSLock()
    private var records: [Record] = []

    func record(stage: String, count: Int) {
        lock.lock()
        records.append(Record(stage: stage, count: count))
        lock.unlock()
    }

    func snapshot() -> [Record] {
        lock.lock()
        let result = records
        lock.unlock()
        return result
    }
}

private actor OneShotLinearFailure {
    private let target: LinearCheckpointTarget
    private var fired = false

    init(target: LinearCheckpointTarget) { self.target = target }

    func checkpoint(_ checkpoint: QwenBF16LinearHooks.Checkpoint) throws {
        guard !fired, target.matches(checkpoint) else { return }
        fired = true
        throw InjectedLinearFailure()
    }
}

private actor AsyncLinearCheckpointGate {
    private var entered = false
    private var released = false
    private var entryContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func suspendUntilReleased() async {
        entered = true
        entryContinuation?.resume()
        entryContinuation = nil
        guard !released else { return }
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
        released = true
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private func hasInvalidLinearInput(
    _ step: QwenBF16LinearStep, input: [Float], tokenCount: Int
) async -> Bool {
    do {
        _ = try await step.append(normalizedHidden: input, tokenCount: tokenCount)
        return false
    } catch let error as QwenTextRunnerError {
        if case .invalidState = error { return true }
        return false
    } catch {
        return false
    }
}

private func throwsAnyLinearError(_ operation: () throws -> Void) -> Bool {
    do { try operation(); return false }
    catch { return true }
}

private func expectLinearClose(
    _ actual: [Float], _ expected: [Float], label: String,
    absoluteTolerance: Float = QwenBF16LinearFixture.absoluteTolerance,
    relativeTolerance: Float = QwenBF16LinearFixture.relativeTolerance
) {
    #expect(actual.count == expected.count, "\(label): count mismatch")
    guard actual.count == expected.count else { return }
    for (index, pair) in zip(actual, expected).enumerated() {
        #expect(pair.0.isFinite && pair.1.isFinite,
                "\(label) index \(index): non-finite \(pair.0), \(pair.1)")
        let bound = absoluteTolerance + relativeTolerance * abs(pair.1)
        #expect(abs(pair.0 - pair.1) <= bound,
                "\(label) index \(index): \(pair.0) != \(pair.1), bound \(bound)")
    }
}

private func expectLinearStateClose(
    _ actual: QwenLinearAttentionLayerState,
    _ expected: IndependentLinearState,
    label: String,
    absoluteTolerance: Float = QwenBF16LinearFixture.absoluteTolerance,
    relativeTolerance: Float = QwenBF16LinearFixture.relativeTolerance
) {
    expectLinearClose(actual.convolutionHistory, expected.convolutionHistory,
                      label: "\(label) convolution history",
                      absoluteTolerance: absoluteTolerance,
                      relativeTolerance: relativeTolerance)
    expectLinearClose(actual.recurrentMatrix, expected.recurrentMatrix,
                      label: "\(label) recurrent matrix",
                      absoluteTolerance: absoluteTolerance,
                      relativeTolerance: relativeTolerance)
}

private func expectLinearStateClose(
    _ actual: QwenLinearAttentionLayerState,
    _ expected: QwenLinearAttentionLayerState,
    label: String,
    absoluteTolerance: Float = QwenBF16LinearFixture.absoluteTolerance,
    relativeTolerance: Float = QwenBF16LinearFixture.relativeTolerance
) {
    expectLinearClose(actual.convolutionHistory, expected.convolutionHistory,
                      label: "\(label) convolution history",
                      absoluteTolerance: absoluteTolerance,
                      relativeTolerance: relativeTolerance)
    expectLinearClose(actual.recurrentMatrix, expected.recurrentMatrix,
                      label: "\(label) recurrent matrix",
                      absoluteTolerance: absoluteTolerance,
                      relativeTolerance: relativeTolerance)
}
