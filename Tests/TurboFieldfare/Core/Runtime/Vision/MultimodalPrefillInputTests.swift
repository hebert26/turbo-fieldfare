import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct MultimodalPrefillInputTests {
    @Test func validatesProjectedFeatureShapeAndDualTokenStreams() throws {
        let context = try MetalContext()
        let rows = 4
        let bytes = rows * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride
        let buffer = try #require(context.device.makeBuffer(
            length: bytes,
            options: .storageModePrivate))
        let features = VisionFeatures(
            buffer: buffer,
            tokenCount: rows,
            hiddenSize: VisionConfig().textHiddenSize,
            gpuNanoseconds: 0,
            scratchBytes: bytes,
            attentionVariant: .native72Q16,
            projectorPath: .affineThreadgroupF16,
            expertResidencyTransition: nil,
            preprocessing: nil)

        let input = try MultimodalPrefillInput(
            effectiveTokenIDs: [10, 20, 20, 20, 20, 30],
            embeddingTokenIDs: [10, 0, 0, 0, 0, 30],
            imageTokenRange: 1..<5,
            imageFeatures: features)

        #expect(input.imageTokenRange == 1..<5)
        #expect(input.effectiveTokenIDs[1] == 20)
        #expect(input.embeddingTokenIDs[1] == 0)
    }

    @Test func rejectsFeatureTokenCountMismatch() throws {
        let context = try MetalContext()
        let buffer = try #require(context.device.makeBuffer(
            length: 3 * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride,
            options: .storageModePrivate))
        let features = VisionFeatures(
            buffer: buffer,
            tokenCount: 3,
            hiddenSize: VisionConfig().textHiddenSize,
            gpuNanoseconds: 0,
            scratchBytes: buffer.length,
            attentionVariant: .native72Q16,
            projectorPath: .affineThreadgroupF16,
            expertResidencyTransition: nil,
            preprocessing: nil)

        #expect(throws: MultimodalPrefillInputError.featureShapeMismatch) {
            try MultimodalPrefillInput(
                effectiveTokenIDs: [10, 20, 20, 30],
                embeddingTokenIDs: [10, 0, 0, 30],
                imageTokenRange: 1..<3,
                imageFeatures: features)
        }
    }

    @Test func qwenInputUsesQwenWidthAndAcceptsVisibleHistoryAt630() throws {
        let context = try MetalContext()
        let architecture = QwenTestArchitecture.qwen36
        #expect(architecture.imageTokenID == 248_056)
        let imageTokenID = Int32(architecture.imageTokenID)
        let grid = try QwenVisionGrid(temporal: 1, height: 42, width: 60)
        let features = try qwenFeatures(rows: 630, grid: grid, context: context)
        let positions = try QwenMultimodalPositions.make(
            tokenCount: 632,
            imageRanges: [1..<631],
            grids: [grid],
            maximumRows: 630)
        let input = try QwenMultimodalPrefillInput(
            effectiveTokenIDs: [Int32](repeating: imageTokenID, count: 632),
            embeddingTokenIDs: [Int32](repeating: imageTokenID, count: 632),
            imageSpans: [QwenMultimodalImageSpan(
                tokenRange: 1..<631, features: features)],
            positionPlan: positions)
        #expect(input.imageSpans.first?.features.hiddenSize == 2_048)
        #expect(input.imageSpans.first?.features.tokenCount == 630)
        #expect(input.positionPlan.positions.count == 632)
    }

    @Test func qwenInputRejects631VisibleRowsWithTypedError() throws {
        let context = try MetalContext()
        let architecture = QwenTestArchitecture.qwen36
        #expect(architecture.imageTokenID == 248_056)
        let imageTokenID = Int32(architecture.imageTokenID)
        let firstGrid = try QwenVisionGrid(temporal: 1, height: 42, width: 54)
        let secondGrid = try QwenVisionGrid(temporal: 1, height: 16, width: 16)
        let first = try qwenFeatures(rows: 567, grid: firstGrid, context: context)
        let second = try qwenFeatures(rows: 64, grid: secondGrid, context: context)
        let spans = [
            QwenMultimodalImageSpan(tokenRange: 1..<568, features: first),
            QwenMultimodalImageSpan(tokenRange: 569..<633, features: second),
        ]
        let positions = try QwenMultimodalPositions.make(
            tokenCount: 633,
            imageRanges: spans.map(\.tokenRange),
            grids: [firstGrid, secondGrid],
            maximumRows: 631)
        #expect(throws: MultimodalPrefillInputError.invalidImageTokenRange) {
            try QwenMultimodalPrefillInput(
                effectiveTokenIDs: [Int32](repeating: imageTokenID, count: 633),
                embeddingTokenIDs: [Int32](repeating: imageTokenID, count: 633),
                imageSpans: spans,
                positionPlan: positions)
        }
    }

    @Test func acceptsOrderedNonOverlappingImageSpans() throws {
        let context = try MetalContext()
        func features(_ rows: Int) throws -> VisionFeatures {
            let bytes = rows * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride
            let buffer = try #require(context.device.makeBuffer(
                length: bytes, options: .storageModePrivate))
            return VisionFeatures(
                buffer: buffer, tokenCount: rows,
                hiddenSize: VisionConfig().textHiddenSize,
                gpuNanoseconds: 0, scratchBytes: bytes,
                attentionVariant: .native72Q16,
                projectorPath: .affineThreadgroupF16,
                expertResidencyTransition: nil, preprocessing: nil)
        }
        let input = try MultimodalPrefillInput(
            effectiveTokenIDs: [1, 2, 2, 3, 4, 4, 4, 5],
            embeddingTokenIDs: [1, 0, 0, 3, 0, 0, 0, 5],
            imageSpans: [
                MultimodalImageSpan(tokenRange: 1..<3, features: try features(2)),
                MultimodalImageSpan(tokenRange: 4..<7, features: try features(3)),
            ])
        #expect(input.imageSpans.map(\.tokenRange) == [1..<3, 4..<7])
    }

    private func qwenFeatures(
        rows: Int, grid: QwenVisionGrid, context: MetalContext
    ) throws -> QwenVisionFeatures {
        let positions = try (0..<rows).map { index in
            try QwenMRoPEPosition(temporal: 0, height: index, width: 0)
        }
        let profile = GTurboQwenVisionProcessorProfileV2(
            processorClass: "Qwen3VLProcessor",
            imageProcessorType: "Qwen2VLImageProcessorFast",
            patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
        return try QwenVisionFeatures(
            device: context.device,
            features: [Float](repeating: 0.25, count: rows * 2_048),
            positions: positions,
            imageDigest: String(repeating: "a", count: 64),
            processorDigest: String(repeating: "b", count: 64),
            profile: profile,
            grid: grid)
    }
}
