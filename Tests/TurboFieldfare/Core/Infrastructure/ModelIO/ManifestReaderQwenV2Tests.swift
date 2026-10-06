import Foundation
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite struct ManifestReaderQwenV2Tests {
    private let digest = String(repeating: "a", count: 64)

    private func data() throws -> Data {
        let layers: [GTurboQwenLayerTypeV2] = (0..<40).map { ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention }
        let arch = GTurboQwenArchitectureV2(hiddenSize: 2048, numLayers: 40, layerTypes: layers,
            numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256, attentionOutputGate: true,
            linearConvolutionKernel: 4, linearKeyHeads: 16, linearKeyHeadDimension: 128,
            linearValueHeads: 32, linearValueHeadDimension: 128, recurrentStateType: .fp32,
            partialRotaryFactor: 0.25, ropeTheta: 10_000_000, mropeInterleaved: true, mropeSections: [11,11,10],
            numberOfExperts: 256, expertsPerToken: 8, routedExpertIntermediateSize: 512,
            sharedExpertIntermediateSize: 512, vocabularySize: 248_320, tiedWordEmbeddings: false,
            hiddenActivation: "silu", bosTokenID: 248_044, eosTokenID: 248_044, imageTokenID: 248_056,
            videoTokenID: 248_057, visionStartTokenID: 248_053, visionEndTokenID: 248_054)
        let groups = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState ? GTurboQuantizationGroupV2(category: category, storage: .fp32) :
                GTurboQuantizationGroupV2(category: category, storage: .affineInt4, groupSize: 64, scaleType: "bf16", biasType: "bf16")
        }
        let omissions = ["fc.weight","layers.0.input_layernorm.weight","layers.0.mlp.experts.down_proj","layers.0.mlp.experts.gate_up_proj","layers.0.mlp.gate.weight","layers.0.mlp.shared_expert.down_proj.weight","layers.0.mlp.shared_expert.gate_proj.weight","layers.0.mlp.shared_expert.up_proj.weight","layers.0.mlp.shared_expert_gate.weight","layers.0.post_attention_layernorm.weight","layers.0.self_attn.k_norm.weight","layers.0.self_attn.k_proj.weight","layers.0.self_attn.o_proj.weight","layers.0.self_attn.q_norm.weight","layers.0.self_attn.q_proj.weight","layers.0.self_attn.v_proj.weight","norm.weight","pre_fc_norm_embedding.weight","pre_fc_norm_hidden.weight"].map { GTurboIgnoredTensorV2(name: "mtp." + $0, reason: .unsupportedMTP) }
        return try GTurboManifestV2Codec.encode(.init(family: .qwen3_6,
            requiredFeatures: [.familyDispatch,.verifiedIdentity,.qwenHybridAttention,.qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository, architecture: .qwen3_6(arch),
            provenance: .init(sourceRepository: GTurboFormatV2.qwenRepository, sourceRevision: GTurboFormatV2.qwenRevision, sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256, sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256, quantizationPolicySHA256: digest),
            quantization: groups, ignoredTensors: omissions,
            files: ["model_weights.bin": .init(size: 32_768, sha256: digest), "packed_experts/layout.json": .init(size: 1, sha256: digest)],
            tensorRegions: [.init(name: "embed", file: "model_weights.bin", offset: 0, size: 16, shape: [1], storage: .affineInt4, quantizationCategory: .embedding)], expertsPerLayer: 256, numLayers: 40, expertStride: 16_384))
    }

    @Test func verifiedReaderDerivesQwenIdentityAndConfigurationFromBytes() throws {
        let loaded = try ManifestReader.decodeVerified(data: data())
        #expect(loaded.descriptor.modelID == GTurboFormatV2.qwenRepository)
        guard case let .qwen3_6(architecture) = loaded.architecture else { Issue.record("expected Qwen runtime configuration"); return }
        #expect(architecture.vocabularySize == 248_320)
        #expect(!architecture.tiedWordEmbeddings)
    }

    @Test func legacyAdapterStillRejectsV1ShapedVersionTwoAsUnsupported() throws {
        let root: [String: Any] = ["magic": "GTURBO", "versionMajor": 2, "versionMinor": 0]
        let invalid = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: ModelError.self) {
            try ManifestReader.decode(data: invalid, expecting: .gemma4Toy())
        }
    }
}
