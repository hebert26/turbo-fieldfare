@testable import TurboFieldfare
import TurboFieldfareFormat

/// Pinned wire metadata for the tiny Qwen test seam. This intentionally lives
/// in Tests so production has no test-only official-architecture accessor.
enum QwenTestArchitecture {
    static let qwen36: QwenArchConfig = {
        let layers: [GTurboQwenLayerTypeV2] = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
        }
        let wire = GTurboQwenArchitectureV2(
            hiddenSize: 2_048,
            numLayers: 40,
            layerTypes: layers,
            numAttentionHeads: 16,
            numKeyValueHeads: 2,
            headDimension: 256,
            attentionOutputGate: true,
            linearConvolutionKernel: 4,
            linearKeyHeads: 16,
            linearKeyHeadDimension: 128,
            linearValueHeads: 32,
            linearValueHeadDimension: 128,
            recurrentStateType: .fp32,
            partialRotaryFactor: 0.25,
            ropeTheta: 10_000_000,
            mropeInterleaved: true,
            mropeSections: [11, 11, 10],
            numberOfExperts: 256,
            expertsPerToken: 8,
            routedExpertIntermediateSize: 512,
            sharedExpertIntermediateSize: 512,
            vocabularySize: 248_320,
            tiedWordEmbeddings: false,
            hiddenActivation: "silu",
            bosTokenID: 248_044,
            eosTokenID: 248_044,
            imageTokenID: 248_056,
            videoTokenID: 248_057,
            visionStartTokenID: 248_053,
            visionEndTokenID: 248_054)
        return QwenArchConfig(wire: wire)
    }()
}
