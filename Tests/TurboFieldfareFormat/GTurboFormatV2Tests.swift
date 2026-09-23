import Foundation
import Testing
@testable import TurboFieldfareFormat

enum V2TestFixture {
    static let digest = String(repeating: "a", count: 64)

    static func manifest() -> GTurboManifestV2 {
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2_048, numLayers: 40,
            layerTypes: (0..<40).map { ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention },
            numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256,
            attentionOutputGate: true, linearConvolutionKernel: 4,
            linearKeyHeads: 16, linearKeyHeadDimension: 128,
            linearValueHeads: 32, linearValueHeadDimension: 128,
            recurrentStateType: .fp32, partialRotaryFactor: 0.25,
            ropeTheta: 10_000_000, mropeInterleaved: true, mropeSections: [11, 11, 10],
            numberOfExperts: 256, expertsPerToken: 8,
            routedExpertIntermediateSize: 512, sharedExpertIntermediateSize: 512,
            vocabularySize: 248_320, tiedWordEmbeddings: false, hiddenActivation: "silu",
            bosTokenID: 248_044, eosTokenID: 248_044, imageTokenID: 248_056,
            videoTokenID: 248_057, visionStartTokenID: 248_053, visionEndTokenID: 248_054)
        let affine: (GTurboQuantizationCategoryV2) -> GTurboQuantizationGroupV2 = {
            .init(category: $0, storage: .affineInt4, groupSize: 64, scaleType: "bf16", biasType: "bf16")
        }
        let quantization = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState ? .init(category: category, storage: .fp32) : affine(category)
        }
        let ignored = [
            "fc.weight", "layers.0.input_layernorm.weight", "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert.down_proj.weight", "layers.0.mlp.shared_expert.gate_proj.weight", "layers.0.mlp.shared_expert.up_proj.weight", "layers.0.mlp.shared_expert_gate.weight", "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight", "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ].map { GTurboIgnoredTensorV2(name: "mtp." + $0, reason: .unsupportedMTP) }
        return GTurboManifestV2(
            family: .qwen3_6,
            requiredFeatures: [.familyDispatch, .verifiedIdentity, .qwenHybridAttention, .qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository, architecture: .qwen3_6(architecture),
            provenance: .init(sourceRepository: GTurboFormatV2.qwenRepository,
                              sourceRevision: GTurboFormatV2.qwenRevision,
                              sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
                              sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
                              quantizationPolicySHA256: digest),
            quantization: quantization, ignoredTensors: ignored,
            files: ["model_weights.bin": .init(size: 32_768, sha256: digest),
                    "packed_experts/layout.json": .init(size: 1, sha256: digest)],
            tensorRegions: [.init(name: "embed", file: "model_weights.bin", offset: 0,
                                  size: 16, shape: [1], storage: .affineInt4,
                                  quantizationCategory: .embedding)],
            expertsPerLayer: 256, numLayers: 40, expertStride: 16_384)
    }

    static func gemmaManifest() -> GTurboManifestV2 {
        let arch = GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1, headDim: 16,
            fullHeadDim: 32, vocabSize: 1_024, slidingWindow: 128,
            finalLogitSoftcap: 30, ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2, topKExperts: 1,
            tieWordEmbeddings: true, attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0])
        let categories: [GTurboQuantizationCategoryV2] = [.embedding, .attention, .router, .sharedExpert, .routedExpert]
        let groups = categories.map { GTurboQuantizationGroupV2(category: $0, storage: .affineInt4, groupSize: 64, scaleType: "bf16", biasType: "bf16") }
        return GTurboManifestV2(
            family: .gemma4, requiredFeatures: [.familyDispatch, .verifiedIdentity],
            modelID: "fixture/gemma", architecture: .gemma4(arch),
            provenance: .init(sourceRepository: "fixture/gemma", sourceRevision: "fixture-revision",
                              sourceIndexSHA256: digest, sidecarSHA256: ["config.json": digest],
                              quantizationPolicySHA256: digest),
            quantization: groups, ignoredTensors: [],
            files: ["model_weights.bin": .init(size: 16_384, sha256: digest),
                    "packed_experts/layout.json": .init(size: 1, sha256: digest)],
            tensorRegions: [.init(name: "embed", file: "model_weights.bin", offset: 0,
                                  size: 16, shape: [1], storage: .affineInt4,
                                  quantizationCategory: .embedding)],
            expertsPerLayer: 2, numLayers: 1, expertStride: 16_384)
    }
}

@Suite struct GTurboFormatV2Tests {
    @Test func validQwenDocumentDispatchesAndDerivesPinnedDescriptor() throws {
        let data = try GTurboManifestV2Codec.encode(V2TestFixture.manifest())
        let document = try GTurboManifestDocumentCodec.decode(data)
        guard case let .v2(verified) = document else { Issue.record("expected v2 dispatch"); return }
        #expect(verified.descriptor.family == .qwen3_6)
        #expect(verified.descriptor.modelID == GTurboFormatV2.qwenRepository)
    }

    @Test func validGemmaV2UsesGemmaDiscriminatorWithoutQwenFields() throws {
        let data = try GTurboManifestV2Codec.encode(V2TestFixture.gemmaManifest())
        guard case let .v2(verified) = try GTurboManifestDocumentCodec.decode(data) else {
            Issue.record("expected v2 dispatch"); return
        }
        #expect(verified.descriptor.family == .gemma4)
        #expect(verified.descriptor.quantization.count == 5)
    }

    @Test func rejectsUnknownMajorAndCrossFamilyDiscriminator() throws {
        let data = try GTurboManifestV2Codec.encode(V2TestFixture.manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["versionMajor"] = 99
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestDocumentCodec.decode(JSONSerialization.data(withJSONObject: root))
        }
        root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["family"] = "gemma4"
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
        root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["family"] = "unrecognized-family"
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
        root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var features = try #require(root["requiredFeatures"] as? [String])
        features.append("unrecognized-feature")
        root["requiredFeatures"] = features
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
    }

    @Test(arguments: ["requiredFeatures", "quantization", "ignoredTensors", "tensorRegions"])
    func rejectsMissingRequiredContractSections(_ key: String) throws {
        let data = try GTurboManifestV2Codec.encode(V2TestFixture.manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root.removeValue(forKey: key)
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
    }

    @Test func rejectsHostileArchitectureProvenanceQuantizationAndStorageMutations() throws {
        let data = try GTurboManifestV2Codec.encode(V2TestFixture.manifest())
        func rejects(_ mutate: (inout [String: Any]) throws -> Void) throws {
            var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            try mutate(&root)
            #expect(throws: GTurboFormatError.self) {
                try GTurboManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
            }
        }
        try rejects { root in
            var architecture = try #require(root["architecture"] as? [String: Any])
            var configuration = try #require(architecture["configuration"] as? [String: Any])
            var layers = try #require(configuration["layerTypes"] as? [String])
            layers[0] = "fullAttention"
            configuration["layerTypes"] = layers; architecture["configuration"] = configuration; root["architecture"] = architecture
        }
        try rejects { root in
            var architecture = try #require(root["architecture"] as? [String: Any])
            var configuration = try #require(architecture["configuration"] as? [String: Any])
            configuration["recurrentStateType"] = "bf16"
            architecture["configuration"] = configuration; root["architecture"] = architecture
        }
        try rejects { root in
            var architecture = try #require(root["architecture"] as? [String: Any])
            var configuration = try #require(architecture["configuration"] as? [String: Any])
            configuration["vocabularySize"] = 1; configuration["tiedWordEmbeddings"] = true
            architecture["configuration"] = configuration; root["architecture"] = architecture
        }
        try rejects { root in
            var provenance = try #require(root["provenance"] as? [String: Any])
            provenance["sourceRevision"] = "forged"; root["provenance"] = provenance
        }
        try rejects { root in
            var groups = try #require(root["quantization"] as? [[String: Any]])
            groups.removeLast(); root["quantization"] = groups
        }
        try rejects { root in
            var ignored = try #require(root["ignoredTensors"] as? [[String: Any]])
            ignored.removeLast(); root["ignoredTensors"] = ignored
        }
        try rejects { root in
            var files = try #require(root["files"] as? [String: Any])
            files["../escape.bin"] = files.removeValue(forKey: "model_weights.bin")
            root["files"] = files
        }
        try rejects { root in
            var regions = try #require(root["tensorRegions"] as? [[String: Any]])
            regions[0]["offset"] = 32_768
            root["tensorRegions"] = regions
        }
        try rejects { root in
            var regions = try #require(root["tensorRegions"] as? [[String: Any]])
            var overlapping = regions[0]
            overlapping["name"] = "overlapping"
            regions.append(overlapping)
            root["tensorRegions"] = regions
        }
    }
}
