import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct MultimodalPromptRendererTests {
    @Test func expandsImagesInContentOrderAndZerosEmbeddingRows() async throws {
        let tokenizer = try await GFTokenizer.load()
        let context = try MetalContext()
        let firstID = UUID()
        let secondID = UUID()
        let first = try features(rows: 2, context: context)
        let second = try features(rows: 3, context: context)

        let input = try MultimodalPromptRenderer.render(
            messages: [MultimodalMessage(
                role: .user,
                content: [
                    .text("before"), .image(id: firstID),
                    .text("between"), .image(id: secondID), .text("after"),
                ])],
            featuresByID: [secondID: second, firstID: first],
            tokenizer: tokenizer)

        #expect(input.imageSpans.map(\.features.tokenCount) == [2, 3])
        for span in input.imageSpans {
            #expect(input.embeddingTokenIDs[span.tokenRange].allSatisfy { $0 == 0 })
            #expect(input.effectiveTokenIDs[span.tokenRange].allSatisfy {
                $0 == MultimodalPromptRenderer.imageTokenID
            })
            #expect(input.effectiveTokenIDs[span.tokenRange.lowerBound - 1]
                == MultimodalPromptRenderer.beginImageTokenID)
            #expect(input.effectiveTokenIDs[span.tokenRange.upperBound]
                == MultimodalPromptRenderer.endImageTokenID)
        }
    }

    @Test func expandsQwenImagesUsingValidatedArchitectureIDsAndOrder() throws {
        let context = try MetalContext()
        let architecture = QwenTestArchitecture.qwen36
        #expect(architecture.imageTokenID == 248_056)
        #expect(architecture.videoTokenID == 248_057)
        #expect(architecture.visionStartTokenID == 248_053)
        #expect(architecture.visionEndTokenID == 248_054)
        let first = try qwenFeatures(context: context, marker: 1)
        let second = try qwenFeatures(context: context, marker: 2)
        let input = try MultimodalPromptRenderer.expandingQwenImageTokens(
            [10, Int32(architecture.imageTokenID), 11,
             Int32(architecture.imageTokenID), 12],
            features: [first, second], architecture: architecture)

        #expect(input.imageSpans.map(\.tokenRange) == [2..<3, 6..<7])
        #expect(input.effectiveTokenIDs[0] == 10)
        #expect(input.effectiveTokenIDs[1] == Int32(architecture.visionStartTokenID))
        #expect(input.effectiveTokenIDs[2] == Int32(architecture.imageTokenID))
        #expect(input.effectiveTokenIDs[3] == Int32(architecture.visionEndTokenID))
        #expect(input.effectiveTokenIDs[4] == 11)
        #expect(input.effectiveTokenIDs[5] == Int32(architecture.visionStartTokenID))
        #expect(input.effectiveTokenIDs[6] == Int32(architecture.imageTokenID))
        #expect(input.effectiveTokenIDs[7] == Int32(architecture.visionEndTokenID))
        #expect(input.effectiveTokenIDs[8] == 12)
        #expect(input.imageSpans.map(\.features.owner.grid) == [first.owner.grid, second.owner.grid])
    }

    @Test func rejectsReservedMarkersAndMissingFeatures() async throws {
        let tokenizer = try await GFTokenizer.load()
        #expect(throws: MultimodalPromptRendererError.reservedImageMarker) {
            try MultimodalPromptRenderer.render(
                messages: [MultimodalMessage(
                    role: .user,
                    content: [.text("typed <|image|> marker")])],
                featuresByID: [:],
                tokenizer: tokenizer)
        }
        let missing = UUID()
        #expect(throws: MultimodalPromptRendererError.missingImage(missing)) {
            try MultimodalPromptRenderer.render(
                messages: [MultimodalMessage(
                    role: .user,
                    content: [.image(id: missing)])],
                featuresByID: [:],
                tokenizer: tokenizer)
        }
    }

    private func qwenFeatures(context: MetalContext, marker: Float) throws -> QwenVisionFeatures {
        let grid = try QwenVisionGrid(temporal: 1, height: 2, width: 2)
        let position = try QwenMRoPEPosition(temporal: 0, height: 0, width: 0)
        let profile = GTurboQwenVisionProcessorProfileV2(
            processorClass: "Qwen3VLProcessor",
            imageProcessorType: "Qwen2VLImageProcessorFast",
            patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
        return try QwenVisionFeatures(
            device: context.device,
            features: [Float](
                repeating: marker,
                count: QwenVisionConfig.official.outputHiddenSize),
            positions: [position],
            imageDigest: String(repeating: "a", count: 64),
            processorDigest: String(repeating: "b", count: 64),
            profile: profile,
            grid: grid)
    }

    private func features(rows: Int, context: MetalContext) throws -> VisionFeatures {
        let bytes = rows * VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride
        let buffer = try #require(context.device.makeBuffer(
            length: bytes, options: .storageModePrivate))
        return VisionFeatures(
            buffer: buffer,
            tokenCount: rows,
            hiddenSize: VisionConfig().textHiddenSize,
            gpuNanoseconds: 0,
            scratchBytes: bytes,
            attentionVariant: .native72Q16,
            projectorPath: .affineThreadgroupF16,
            expertResidencyTransition: nil,
            preprocessing: nil)
    }
}
