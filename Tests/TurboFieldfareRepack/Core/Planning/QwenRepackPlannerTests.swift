import Foundation
import Testing
@testable import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

@Suite
struct QwenRepackPlannerTests {
    @Test func affineSizingUsesActualRowLocalPackedBytes() throws {
        let layout = try QwenRepackPlanner.affineComponentSizes(
            shape: [2, 65], bitWidth: .int4)

        // Each 65-element row is [64, 1]: 32 + 1 bytes per row. A naive
        // whole-tensor ceil(130 * 4 / 8) would incorrectly produce 65.
        #expect(layout.valuesOffset == 0)
        #expect(layout.valuesSize == 66)
        #expect(layout.scalesOffset == 66)
        #expect(layout.scalesSize == 8)
        #expect(layout.biasesOffset == 74)
        #expect(layout.biasesSize == 8)
        #expect(layout.totalSize == 82)
    }

    @Test func provenanceRejectsEveryPinnedBindingMutation() throws {
        let valid = validProvenance()
        try QwenRepackPlanner.validateProvenance(valid)

        let mutations: [QwenPlanningProvenance] = [
            .init(repository: "wrong", revision: valid.revision,
                  validatedSidecarSHA256: valid.validatedSidecarSHA256,
                  observedIndexSHA256: valid.observedIndexSHA256,
                  observedConfigSHA256: valid.observedConfigSHA256,
                  quantizationPolicySHA256: valid.quantizationPolicySHA256),
            .init(repository: valid.repository, revision: "wrong",
                  validatedSidecarSHA256: valid.validatedSidecarSHA256,
                  observedIndexSHA256: valid.observedIndexSHA256,
                  observedConfigSHA256: valid.observedConfigSHA256,
                  quantizationPolicySHA256: valid.quantizationPolicySHA256),
            .init(repository: valid.repository, revision: valid.revision,
                  validatedSidecarSHA256: ["config.json": "wrong"],
                  observedIndexSHA256: valid.observedIndexSHA256,
                  observedConfigSHA256: valid.observedConfigSHA256,
                  quantizationPolicySHA256: valid.quantizationPolicySHA256),
            .init(repository: valid.repository, revision: valid.revision,
                  validatedSidecarSHA256: valid.validatedSidecarSHA256,
                  observedIndexSHA256: "wrong",
                  observedConfigSHA256: valid.observedConfigSHA256,
                  quantizationPolicySHA256: valid.quantizationPolicySHA256),
            .init(repository: valid.repository, revision: valid.revision,
                  validatedSidecarSHA256: valid.validatedSidecarSHA256,
                  observedIndexSHA256: valid.observedIndexSHA256,
                  observedConfigSHA256: "wrong",
                  quantizationPolicySHA256: valid.quantizationPolicySHA256),
            .init(repository: valid.repository, revision: valid.revision,
                  validatedSidecarSHA256: valid.validatedSidecarSHA256,
                  observedIndexSHA256: valid.observedIndexSHA256,
                  observedConfigSHA256: valid.observedConfigSHA256,
                  quantizationPolicySHA256: "wrong"),
        ]
        for mutation in mutations {
            #expect(throws: RepackError.self) {
                try QwenRepackPlanner.validateProvenance(mutation)
            }
        }
    }

    @Test func strictFixtureArchitectureRejectsMissingAndMutatedVisionFields() throws {
        let valid = try fixtureArchitecture()
        #expect(valid.text.numLayers == 40)
        #expect(valid.text.layerTypes.filter { $0 == .fullAttention }.count == 10)
        #expect(valid.vision.depth == 27)

        var missing = configObject()
        var vision = try #require(missing["vision_config"] as? [String: Any])
        vision.removeValue(forKey: "depth")
        missing["vision_config"] = vision
        #expect(throws: RepackError.self) {
            try ArchInfo.loadQwenMetadataFixture(configData: try jsonData(missing))
        }

        var wrong = configObject()
        vision = try #require(wrong["vision_config"] as? [String: Any])
        vision["patch_size"] = 14
        wrong["vision_config"] = vision
        #expect(throws: RepackError.self) {
            try ArchInfo.loadQwenMetadataFixture(configData: try jsonData(wrong))
        }
    }

    @Test func officialMetadataFixturePlansAllCategoriesWithStableLayout() throws {
        let architecture = try fixtureArchitecture()
        let tensors = try sourceTensors(architecture: architecture)
        #expect(tensors.count == 1_045)

        let first = try QwenRepackPlanner.planMetadataFixture(
            architecture: architecture, tensors: tensors,
            outputDirectory: "/tmp/qwen-one", includeVision: true)
        let second = try QwenRepackPlanner.planMetadataFixture(
            architecture: architecture, tensors: tensors,
            outputDirectory: "/a/different/absolute/root", includeVision: true)
        let textOnly = try QwenRepackPlanner.planMetadataFixture(
            architecture: architecture, tensors: tensors,
            outputDirectory: "/tmp/qwen-text-only", includeVision: false)

        #expect(first.textTensors.count == 613)
        #expect(first.expertLayers.count == 40)
        #expect(first.visionTensorNames.count == 333)
        #expect(first.visionCompanion?.tensors.count == 333)
        #expect(textOnly.visionTensorNames.count == 333)
        #expect(textOnly.visionCompanion == nil)
        #expect(first.omittedMTPNames.count == 19)
        #expect(first.canonicalFingerprint == second.canonicalFingerprint)
        #expect(first.textTensors.allSatisfy { $0.fileOffset.isMultiple(of: 16_384) })
        #expect(first.expertLayers.allSatisfy {
            $0.expertsPerLayer == 256 && $0.expertStride.isMultiple(of: 16_384)
                && $0.fileSize == 256 * $0.expertStride && $0.sources.count == 2
        })
        #expect(first.expertLayers.allSatisfy { $0.sources.map(\.role) == ["down", "gate_up"] })
        #expect(first.artifacts.map(\.relativePath).contains("model_weights.bin"))
        #expect(first.artifacts.map(\.relativePath).contains("vision_weights.bin"))
        #expect(first.scratch.maximumPayloadBufferBytes == first.scratch.transformTileBytes
            + first.scratch.maximumGroupBytes + first.scratch.quantizerPayloadBytes)
    }

    @Test func malformedMetadataIsRejectedBeforeLayout() throws {
        let architecture = try fixtureArchitecture()
        let valid = try sourceTensors(architecture: architecture)
        var cases: [[SourceTensor]] = []

        var wrongName = valid
        wrongName[0] = replacing(wrongName[0], name: "unrecognized.weight")
        cases.append(wrongName)
        var wrongDtype = valid
        wrongDtype[1] = replacing(wrongDtype[1], dtype: .fp32)
        cases.append(wrongDtype)
        var wrongShape = valid
        wrongShape[2] = replacing(wrongShape[2], shape: [1])
        cases.append(wrongShape)
        var wrongMTP = valid
        let mtpIndex = try #require(wrongMTP.firstIndex { $0.name == "mtp.fc.weight" })
        wrongMTP[mtpIndex] = replacing(wrongMTP[mtpIndex], name: "mtp.unexpected.weight")
        cases.append(wrongMTP)
        var unsafeRange = valid
        unsafeRange[3] = replacing(unsafeRange[3], absoluteOffset: UInt64.max)
        cases.append(unsafeRange)

        for tensors in cases {
            #expect(throws: Error.self) {
                try QwenRepackPlanner.planMetadataFixture(
                    architecture: architecture, tensors: tensors,
                    outputDirectory: "/tmp/qwen-invalid", includeVision: false)
            }
        }

        #expect(throws: RepackError.self) {
            try QwenRepackPlanner.affineComponentSizes(
                shape: [UInt64.max, 65], bitWidth: .int8)
        }
    }

    @Test func strictFixtureArchitectureRejectsWrongLayerScheduleAndExpertCount() throws {
        var wrongSchedule = configObject()
        var text = try #require(wrongSchedule["text_config"] as? [String: Any])
        var layers = try #require(text["layer_types"] as? [String])
        layers[0] = "full_attention"
        text["layer_types"] = layers
        wrongSchedule["text_config"] = text
        #expect(throws: RepackError.self) {
            try ArchInfo.loadQwenMetadataFixture(configData: try jsonData(wrongSchedule))
        }

        var wrongExperts = configObject()
        text = try #require(wrongExperts["text_config"] as? [String: Any])
        text["num_experts"] = 255
        wrongExperts["text_config"] = text
        #expect(throws: RepackError.self) {
            try ArchInfo.loadQwenMetadataFixture(configData: try jsonData(wrongExperts))
        }
    }

    private func validProvenance() -> QwenPlanningProvenance {
        .init(repository: ModelSourceCatalog.qwen.repository,
              revision: ModelSourceCatalog.qwen.revision,
              validatedSidecarSHA256: ModelSourceCatalog.qwen.sidecarSHA256,
              observedIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
              observedConfigSHA256: GTurboFormatV2.qwenSidecarSHA256["config.json"]!,
              quantizationPolicySHA256: BF16AffineQuantizationPolicy.policySHA256)
    }

    private func fixtureArchitecture() throws -> ArchInfo.QwenPlanningArchitecture {
        try ArchInfo.loadQwenMetadataFixture(configData: try jsonData(configObject()))
    }

    private func configObject() -> [String: Any] {
        let layers = (0..<40).map { ($0 + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention" }
        return [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe", "tie_word_embeddings": false,
            "image_token_id": 248_056, "video_token_id": 248_057,
            "vision_start_token_id": 248_053, "vision_end_token_id": 248_054,
            "text_config": [
                "model_type": "qwen3_5_moe_text", "dtype": "bfloat16",
                "hidden_size": 2_048, "num_hidden_layers": 40, "layer_types": layers,
                "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
                "attn_output_gate": true, "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 16, "linear_key_head_dim": 128,
                "linear_num_value_heads": 32, "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32", "partial_rotary_factor": 0.25,
                "rope_parameters": ["rope_theta": 10_000_000, "mrope_interleaved": true, "mrope_section": [11, 11, 10]],
                "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512, "vocab_size": 248_320,
                "tie_word_embeddings": false, "hidden_act": "silu", "bos_token_id": 248_044,
                "eos_token_id": 248_044, "full_attention_interval": 4,
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "max_position_embeddings": 262_144, "rms_norm_eps": 0.000001,
            ],
            "vision_config": [
                "depth": 27, "hidden_size": 1_152, "intermediate_size": 4_304,
                "num_heads": 16, "num_position_embeddings": 2_304, "in_channels": 3,
                "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "out_hidden_size": 2_048, "hidden_act": "gelu_pytorch_tanh",
            ],
        ]
    }

    private func jsonData(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func sourceTensors(architecture: ArchInfo.QwenPlanningArchitecture) throws -> [SourceTensor] {
        var offset: UInt64 = 0
        return try Qwen36OfficialMetadata.indexNames.map { name in
            let shape = try expectedShape(name, architecture: architecture)
            let bytes = try shape.reduce(UInt64(1)) { partial, value in
                let (product, overflow) = partial.multipliedReportingOverflow(by: value)
                guard !overflow else { throw RepackError.configurationInvalid(detail: "test shape overflow") }
                return product
            }.multipliedReportingOverflow(by: 2).partialValue
            let tensor = SourceTensor(name: name, shardPath: "model-00001-of-00026.safetensors",
                                      dtype: .bf16, shape: shape, absoluteOffset: offset, sizeBytes: bytes)
            offset += bytes
            return tensor
        }
    }

    private func expectedShape(_ name: String, architecture: ArchInfo.QwenPlanningArchitecture) throws -> [UInt64] {
        let h: UInt64 = 2_048, e: UInt64 = 512, n: UInt64 = 256
        if name == "lm_head.weight" || name == "model.language_model.embed_tokens.weight" { return [248_320, h] }
        if name == "mtp.fc.weight" { return [h, 2 * h] }
        if name == "model.language_model.norm.weight" || name == "mtp.norm.weight" || name.hasPrefix("mtp.pre_fc_norm") { return [h] }
        if name.hasPrefix("model.visual.") { return try visionShape(name) }
        let layerName = name.hasPrefix("mtp.layers.0.") ? String(name.dropFirst("mtp.layers.0.".count)) : String(name.split(separator: ".layers.", maxSplits: 1).last!.split(separator: ".", maxSplits: 1).dropFirst().joined(separator: "."))
        return try layerShape(layerName, h: h, e: e, n: n)
    }

    private func layerShape(_ suffix: String, h: UInt64, e: UInt64, n: UInt64) throws -> [UInt64] {
        switch suffix {
        case "input_layernorm.weight", "post_attention_layernorm.weight": return [h]
        case "mlp.gate.weight": return [n, h]
        case "mlp.shared_expert_gate.weight": return [1, h]
        case "mlp.shared_expert.down_proj.weight": return [h, e]
        case "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight": return [e, h]
        case "mlp.experts.down_proj": return [n, h, e]
        case "mlp.experts.gate_up_proj": return [n, 2 * e, h]
        case "self_attn.q_norm.weight", "self_attn.k_norm.weight": return [256]
        case "self_attn.q_proj.weight": return [8_192, h]
        case "self_attn.k_proj.weight", "self_attn.v_proj.weight": return [512, h]
        case "self_attn.o_proj.weight": return [h, 4_096]
        case "linear_attn.A_log", "linear_attn.dt_bias": return [32]
        case "linear_attn.conv1d.weight": return [8_192, 1, 4]
        case "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight": return [32, h]
        case "linear_attn.in_proj_qkv.weight": return [8_192, h]
        case "linear_attn.in_proj_z.weight": return [4_096, h]
        case "linear_attn.norm.weight": return [128]
        case "linear_attn.out_proj.weight": return [h, 4_096]
        default: throw RepackError.configurationInvalid(detail: "unknown test tensor \(suffix)")
        }
    }

    private func visionShape(_ name: String) throws -> [UInt64] {
        if name == "model.visual.patch_embed.proj.weight" { return [1_152, 3, 2, 16, 16] }
        if name == "model.visual.patch_embed.proj.bias" { return [1_152] }
        if name == "model.visual.pos_embed.weight" { return [2_304, 1_152] }
        if name == "model.visual.merger.norm.weight" || name == "model.visual.merger.norm.bias" { return [1_152] }
        if name == "model.visual.merger.linear_fc1.weight" { return [4_608, 4_608] }
        if name == "model.visual.merger.linear_fc1.bias" { return [4_608] }
        if name == "model.visual.merger.linear_fc2.weight" { return [2_048, 4_608] }
        if name == "model.visual.merger.linear_fc2.bias" { return [2_048] }
        let suffix = name.split(separator: ".blocks.", maxSplits: 1)[1].split(separator: ".", maxSplits: 1)[1]
        switch suffix {
        case "attn.proj.weight": return [1_152, 1_152]
        case "attn.proj.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "mlp.linear_fc2.bias": return [1_152]
        case "attn.qkv.weight": return [3_456, 1_152]
        case "attn.qkv.bias": return [3_456]
        case "mlp.linear_fc1.weight": return [4_304, 1_152]
        case "mlp.linear_fc1.bias": return [4_304]
        case "mlp.linear_fc2.weight": return [1_152, 4_304]
        default: throw RepackError.configurationInvalid(detail: "unknown vision test tensor")
        }
    }

    private func replacing(_ tensor: SourceTensor, name: String? = nil, dtype: SourceTensor.Dtype? = nil,
                           shape: [UInt64]? = nil, absoluteOffset: UInt64? = nil) -> SourceTensor {
        .init(name: name ?? tensor.name, shardPath: tensor.shardPath, dtype: dtype ?? tensor.dtype,
              shape: shape ?? tensor.shape, absoluteOffset: absoluteOffset ?? tensor.absoluteOffset,
              sizeBytes: tensor.sizeBytes)
    }
}
