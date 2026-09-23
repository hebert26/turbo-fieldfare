import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite(.serialized) struct QwenMultimodalDecoderOracleTests {
    private let tolerance: Float = 1e-5

    @Test func preparedRowsAndFullAttentionTraceMatchIndependentReference() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let fixture = try makeFixture(model: model, context: context)
        let runner = try model.makeRunner(context: context)
        let actual = try await runner.prefill(prepared: fixture.prepared)
        let expected = try QwenMultimodalDecoderReference.evaluate(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            input: fixture.referenceInput)

        assertClose(actual.output.preparedHidden, expected.preparedRows,
                    label: "prepared hidden")
        // This oracle deliberately proves two seams, not full-model logits:
        // prepared-row splicing, then layer-3 attention from its actual
        // pre-projection input. It does not duplicate preceding layers.
        let fullTrace = try #require(actual.output.layers.first {
            $0.layer == 3 && $0.normalizedQuery != nil && $0.rotatedQuery != nil
        })
        let attentionInput = try QwenMultimodalDecoderReference.evaluateAttentionInput(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            rows: fullTrace.inputNormalized,
            positions: fixture.referenceInput.positions)
        assertClose(try #require(fullTrace.normalizedQuery), attentionInput.normalizedQuery,
                    label: "layer-3 normalized query")
        assertClose(try #require(fullTrace.normalizedKey), attentionInput.normalizedKey,
                    label: "layer-3 normalized key")
        assertClose(try #require(fullTrace.rotatedQuery), attentionInput.rotatedQuery,
                    label: "layer-3 rotated query")
        assertClose(try #require(fullTrace.rotatedKey), attentionInput.rotatedKey,
                    label: "layer-3 rotated key")
        #expect(actual.output.state.sequenceLength == fixture.referenceInput.positions.count)
    }

    @Test func wrongRowsAndSwappedAxesCannotMatchPreparedDecoderTrace() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let fixture = try makeFixture(model: model, context: context)
        let baseline = try QwenMultimodalDecoderReference.evaluate(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            input: fixture.referenceInput)

        var swappedRows = fixture.referenceInput
        swappedRows.featureRows.swapAt(0, 5)
        let wrongRows = try QwenMultimodalDecoderReference.evaluate(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            input: swappedRows)
        #expect(maxDifference(baseline.preparedRows, wrongRows.preparedRows) > tolerance)

        var swappedAxes = fixture.referenceInput
        swappedAxes.positions = swappedAxes.positions.map {
            guard $0.count == 3 else { return $0 }
            return [$0[0], $0[2], $0[1]]
        }
        let wrongAxes = try QwenMultimodalDecoderReference.evaluate(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            input: swappedAxes)
        #expect(maxDifference(baseline.rotatedQuery, wrongAxes.rotatedQuery) > tolerance)
        #expect(maxDifference(baseline.rotatedKey, wrongAxes.rotatedKey) > tolerance)
    }

    @Test func nonzeroDeltaAppendMatchesLocalAttentionOracleAndRejectsMismatch() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let fixture = try makeFixture(model: model, context: context)
        let runner = try model.makeRunner(context: context)
        _ = try await runner.prefill(prepared: fixture.prepared)

        // Cache index 11 plus retained image delta -3 maps to absolute 8.
        let coordinate = try QwenMRoPEPosition(temporal: 8, height: 8, width: 8)
        let appendInput = try QwenPreparedPrefill(
            tokenIDs: [1], featureOverrides: [], positions: [coordinate],
            textRoPEDelta: -3)
        let appended = try await runner.append(prepared: appendInput)
        let trace = try #require(appended.output.layers.first {
            $0.layer == 3 && $0.normalizedQuery != nil && $0.rotatedQuery != nil
        })
        let expected = try QwenMultimodalDecoderReference.evaluateAttentionInput(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            rows: trace.inputNormalized,
            positions: [[8, 8, 8]])
        assertClose(try #require(trace.normalizedQuery), expected.normalizedQuery,
                    label: "append normalized query")
        assertClose(try #require(trace.normalizedKey), expected.normalizedKey,
                    label: "append normalized key")
        assertClose(try #require(trace.rotatedQuery), expected.rotatedQuery,
                    label: "append rotated query")
        assertClose(try #require(trace.rotatedKey), expected.rotatedKey,
                    label: "append rotated key")
        #expect(appended.output.state.sequenceLength == 12)

        let beforeReject = try await runner.snapshot()
        let inconsistent = try QwenPreparedPrefill(
            tokenIDs: [1], featureOverrides: [],
            positions: [try QwenMRoPEPosition(temporal: 10, height: 10, width: 10)],
            textRoPEDelta: -3)
        do {
            _ = try await runner.append(prepared: inconsistent)
            Issue.record("inconsistent append metadata unexpectedly passed")
        } catch let error as QwenTextRunnerError {
            #expect(error == .invalidState(
                detail: "continuation M-RoPE coordinate/delta mismatch"))
        } catch {
            Issue.record("unexpected append error: \(error)")
        }
        let afterReject = try await runner.snapshot()
        #expect(beforeReject == afterReject)
    }

    @Test func sectionOrderAndDeltaControlsChangeIndependentRoPEExpectation() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let fixture = try makeFixture(model: model, context: context)
        let baseline = try QwenMultimodalDecoderReference.evaluate(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            input: fixture.referenceInput)

        let wrongSections = try QwenMultimodalDecoderReference.Configuration(
            hiddenSize: 32,
            queryHeadCount: 2,
            keyValueHeadCount: 1,
            headDimension: 16,
            rotaryDimension: 12,
            ropeTheta: 10_000_000,
            epsilon: 1e-6,
            mropeSections: [2, 3, 1])
        let sectionControl = try QwenMultimodalDecoderReference.evaluate(
            configuration: wrongSections,
            weights: fixture.referenceWeights,
            input: fixture.referenceInput)
        #expect(maxDifference(baseline.rotatedQuery, sectionControl.rotatedQuery) > tolerance)
        #expect(maxDifference(baseline.rotatedKey, sectionControl.rotatedKey) > tolerance)

        var zeroDeltaInput = fixture.referenceInput
        zeroDeltaInput.decodeDelta = 0
        let zeroDelta = try QwenMultimodalDecoderReference.evaluate(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            input: zeroDeltaInput)
        var oppositeDeltaInput = fixture.referenceInput
        oppositeDeltaInput.decodeDelta = 3
        let oppositeDelta = try QwenMultimodalDecoderReference.evaluate(
            configuration: fixture.referenceConfiguration,
            weights: fixture.referenceWeights,
            input: oppositeDeltaInput)
        #expect(maxDifference(baseline.decodeQuery, zeroDelta.decodeQuery) > tolerance)
        #expect(maxDifference(baseline.decodeKey, zeroDelta.decodeKey) > tolerance)
        #expect(maxDifference(baseline.decodeQuery, oppositeDelta.decodeQuery) > tolerance)
        #expect(baseline.decodeCoordinate == 8)
        #expect(zeroDelta.decodeCoordinate == 11)
        #expect(oppositeDelta.decodeCoordinate == 14)
    }

    private struct Fixture {
        let prepared: QwenPreparedPrefill
        let referenceConfiguration: QwenMultimodalDecoderReference.Configuration
        let referenceWeights: QwenMultimodalDecoderReference.ProjectionWeights
        var referenceInput: QwenMultimodalDecoderReference.Input
    }

    private func makeFixture(model: QwenTextModel, context: MetalContext) throws -> Fixture {
        let configuration = try QwenMultimodalDecoderReference.Configuration.p12Tiny()
        let qAll = try model.attentionTensor(layer: 3, suffix: "q_proj.weight")
            .decodedFloat32()
        let k = try model.attentionTensor(layer: 3, suffix: "k_proj.weight")
            .decodedFloat32()
        let qNorm = try model.attentionTensor(layer: 3, suffix: "q_norm.weight")
            .decodedFloat32()
        let kNorm = try model.attentionTensor(layer: 3, suffix: "k_norm.weight")
            .decodedFloat32()
        let qWidth = configuration.queryHeadCount * configuration.headDimension
        // q_proj stores each query head's rows interleaved with its gate
        // rows. Extract only Q, preserving the production head order.
        let q = (0..<configuration.queryHeadCount).flatMap { head in
            let headBase = head * 2 * configuration.headDimension
                * configuration.hiddenSize
            let headEnd = headBase + configuration.headDimension
                * configuration.hiddenSize
            return qAll[headBase..<headEnd]
        }
        #expect(q.count == qWidth * configuration.hiddenSize)
        let weights = QwenMultimodalDecoderReference.ProjectionWeights(
            query: q, key: k, queryNorm: qNorm, keyNorm: kNorm)
        let embedding = try model.embedding.decodedFloat32()
        let tokenIDs: [Int32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        let tokenRows = tokenIDs.flatMap { token in
            let base = Int(token) * configuration.hiddenSize
            return embedding[base..<(base + configuration.hiddenSize)]
        }
        let first = [Float](repeating: 0.125, count: configuration.hiddenSize)
        let second = (0..<configuration.hiddenSize).map {
            Float($0 + 1) * -0.0625
        }
        let zero = [Float](repeating: 0, count: configuration.hiddenSize)
        let featureRows = [first, zero, zero, zero, zero, second]
        let grid = try QwenVisionGrid(temporal: 1, height: 4, width: 6)
        // Frozen P3 tiny-seam positions, token-major [t, h, w]. Keep these
        // literals independent of the production position planner.
        let positions = [
            [0, 0, 0], [1, 1, 1],
            [2, 2, 2], [2, 2, 3], [2, 2, 4],
            [2, 3, 2], [2, 3, 3], [2, 3, 4],
            [5, 5, 5], [6, 6, 6], [7, 7, 7],
        ]
        let ropePositions = try positions.map {
            try QwenMRoPEPosition(temporal: $0[0], height: $0[1], width: $0[2])
        }
        let profile = GTurboQwenVisionProcessorProfileV2(
            processorClass: "Qwen3VLProcessor",
            imageProcessorType: "Qwen2VLImageProcessorFast",
            patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
        let owner = try QwenRetainedFeatureOwner(
            device: context.device,
            features: featureRows.flatMap { $0 },
            positions: Array(ropePositions[2..<8]),
            imageDigest: String(repeating: "a", count: 64),
            processorDigest: String(repeating: "b", count: 64),
            profile: profile,
            grid: grid,
            hiddenSize: configuration.hiddenSize)
        let override = QwenPreparedFeatureOverride(
            tokenRange: 2..<8, owner: owner)
        let prepared = try QwenPreparedPrefill(
            tokenIDs: tokenIDs,
            featureOverrides: [override],
            positions: ropePositions,
            textRoPEDelta: -3)
        let input = QwenMultimodalDecoderReference.Input(
            tokenRows: tokenRows,
            featureRows: featureRows,
            padRows: Array(2..<8),
            positions: positions,
            decodeBasePosition: tokenIDs.count,
            decodeDelta: -3,
            decodeRow: Array(embedding[configuration.hiddenSize..<(2 * configuration.hiddenSize)]))
        return Fixture(
            prepared: prepared,
            referenceConfiguration: configuration,
            referenceWeights: weights,
            referenceInput: input)
    }

    private func assertClose(_ actual: [Float], _ expected: [Float], label: String) {
        #expect(actual.count == expected.count, "\(label) count")
        #expect(maxDifference(actual, expected) <= tolerance, "\(label) mismatch")
    }

    private func maxDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        return zip(lhs, rhs).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
    }
}
