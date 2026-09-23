import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// The family admission tests deliberately use metadata-only directories. A
/// verified descriptor is an identity/configuration result, not evidence that
/// payload files exist or that a runtime has been constructed.
@Suite(.serialized) struct ModelFamilyRuntimeTests {
    @Test func classifiesGemmaV1WithoutOpeningPayloadFiles() throws {
        let (directory, _) = try ManifestReaderTests.writeToyManifest(
            ["quant": ManifestReaderTests.quant()], config: .gemma4_26B_A4B)
        defer { try? FileManager.default.removeItem(at: directory) }

        let admission = try ModelFamilyAdmission.classify(directoryURL: directory)
        guard case .gemmaV1 = admission else {
            Issue.record("expected legacy Gemma admission")
            return
        }
    }

    @Test func classifiesOfficialQwenMetadataWithoutConstructingPayloadRuntime() throws {
        let directory = try Self.writeOfficialQwenMetadataDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let admission = try ModelFamilyAdmission.classify(directoryURL: directory)
        guard case let .qwenV2(loaded) = admission else {
            Issue.record("expected verified Qwen v2 admission")
            return
        }
        #expect(loaded.descriptor.family == .qwen3_6)
        #expect(loaded.descriptor.modelID == GTurboFormatV2.qwenRepository)
        guard case let .qwen3_6(configuration) = loaded.architecture else {
            Issue.record("expected Qwen architecture")
            return
        }
        #expect(configuration.vocabularySize == 248_320)
        #expect(!configuration.tiedWordEmbeddings)
    }

    @Test func rejectsWrongFamilyIdentityBeforePayloadMapping() throws {
        let data = try Self.officialQwenMetadata()
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["modelID"] = "fixture/not-qwen"
        let directory = try Self.writeMetadataDirectory(
            data: try JSONSerialization.data(withJSONObject: root))
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect {
            _ = try ModelFamilyAdmission.classify(directoryURL: directory)
        } throws: { error in
            guard case ModelError.indexCorrupt = error else { return false }
            return true
        }
    }

    @Test func rejectsQwenMetadataFamilyMutationDuringAdmission() throws {
        let data = try Self.officialQwenMetadata()
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["family"] = "gemma4"
        let directory = try Self.writeMetadataDirectory(
            data: try JSONSerialization.data(withJSONObject: root))
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect {
            _ = try ModelFamilyAdmission.classify(directoryURL: directory)
        } throws: { error in
            guard case ModelError.indexCorrupt = error else { return false }
            return true
        }
    }

    @Test func classifiedQwenWithNoPayloadFailsBeforeRuntimeCreation() throws {
        let directory = try Self.writeOfficialQwenMetadataDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let admission = try ModelFamilyAdmission.classify(directoryURL: directory)
        guard case .qwenV2 = admission else {
            Issue.record("the payload test requires Qwen admission")
            return
        }

        let context = try MetalContext()
        #expect {
            _ = try ModelFamilyRuntime.load(
                directoryURL: directory, device: context.device)
        } throws: { error in
            guard case let ModelError.missingFile(name) = error else { return false }
            return name == "model_weights.bin" || name == "packed_experts/layout.json"
        }
    }

    @Test func classifiedQwenWithWrongPayloadSizeFailsAfterAdmission() throws {
        let directory = try Self.writeOfficialQwenMetadataDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(repeating: 0, count: 1).write(
            to: directory.appendingPathComponent("model_weights.bin"))
        let admission = try ModelFamilyAdmission.classify(directoryURL: directory)
        guard case .qwenV2 = admission else {
            Issue.record("the payload test requires Qwen admission")
            return
        }

        let context = try MetalContext()
        #expect {
            _ = try ModelFamilyRuntime.load(
                directoryURL: directory, device: context.device)
        } throws: { error in
            guard case let ModelError.tensorSizeMismatch(name, expected, actual) = error else {
                return false
            }
            return name == "model_weights.bin" && expected == 32_768 && actual == 1
        }
    }

    @Test func classifiedQwenWithWrongPayloadHashFailsAfterAdmission() throws {
        let directory = try Self.writeOfficialQwenMetadataDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(repeating: 0, count: 32_768).write(
            to: directory.appendingPathComponent("model_weights.bin"))
        try Data(repeating: 0, count: 1).write(
            to: directory.appendingPathComponent("packed_experts/layout.json"))
        let admission = try ModelFamilyAdmission.classify(directoryURL: directory)
        guard case .qwenV2 = admission else {
            Issue.record("the payload test requires Qwen admission")
            return
        }

        let context = try MetalContext()
        #expect {
            _ = try ModelFamilyRuntime.load(
                directoryURL: directory, device: context.device)
        } throws: { error in
            guard case let ModelError.checksumMismatch(file) = error else { return false }
            return file == "model_weights.bin" || file == "packed_experts/layout.json"
        }
    }

    private static func writeOfficialQwenMetadataDirectory() throws -> URL {
        try writeMetadataDirectory(data: officialQwenMetadata())
    }

    private static func writeMetadataDirectory(data: Data) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-admission-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("packed_experts"),
            withIntermediateDirectories: true)
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    private static func officialQwenMetadata() throws -> Data {
        let digest = String(repeating: "a", count: 64)
        let layers: [GTurboQwenLayerTypeV2] = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
        }
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2_048, numLayers: 40, layerTypes: layers,
            numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256,
            attentionOutputGate: true, linearConvolutionKernel: 4,
            linearKeyHeads: 16, linearKeyHeadDimension: 128,
            linearValueHeads: 32, linearValueHeadDimension: 128,
            recurrentStateType: .fp32, partialRotaryFactor: 0.25,
            ropeTheta: 10_000_000, mropeInterleaved: true,
            mropeSections: [11, 11, 10], numberOfExperts: 256,
            expertsPerToken: 8, routedExpertIntermediateSize: 512,
            sharedExpertIntermediateSize: 512, vocabularySize: 248_320,
            tiedWordEmbeddings: false, hiddenActivation: "silu",
            bosTokenID: 248_044, eosTokenID: 248_044,
            imageTokenID: 248_056, videoTokenID: 248_057,
            visionStartTokenID: 248_053, visionEndTokenID: 248_054)
        let quantization = GTurboQuantizationCategoryV2.allCases.map { category in
            if category == .recurrentState {
                return GTurboQuantizationGroupV2(category: category, storage: .fp32)
            }
            return GTurboQuantizationGroupV2(
                category: category, storage: .affineInt4, groupSize: 64,
                scaleType: "bf16", biasType: "bf16")
        }
        let ignoredNames = [
            "fc.weight", "layers.0.input_layernorm.weight",
            "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight",
            "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ].map { GTurboIgnoredTensorV2(name: "mtp." + $0, reason: .unsupportedMTP) }
        return try GTurboManifestV2Codec.encode(.init(
            family: .qwen3_6,
            requiredFeatures: [.familyDispatch, .verifiedIdentity,
                               .qwenHybridAttention, .qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository,
            architecture: .qwen3_6(architecture),
            provenance: .init(
                sourceRepository: GTurboFormatV2.qwenRepository,
                sourceRevision: GTurboFormatV2.qwenRevision,
                sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
                sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
                quantizationPolicySHA256: digest),
            quantization: quantization, ignoredTensors: ignoredNames,
            files: [
                "model_weights.bin": .init(size: 32_768, sha256: digest),
                "packed_experts/layout.json": .init(size: 1, sha256: digest),
            ],
            tensorRegions: [.init(
                name: "embed", file: "model_weights.bin", offset: 0,
                size: 16, shape: [1], storage: .affineInt4,
                quantizationCategory: .embedding)],
            expertsPerLayer: 256, numLayers: 40, expertStride: 16_384))
    }
}
