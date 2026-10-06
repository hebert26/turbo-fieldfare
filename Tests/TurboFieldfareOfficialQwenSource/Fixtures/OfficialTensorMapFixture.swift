import TurboFieldfareOfficialQwenSource

/// Independent pinned metadata fixture for the Qwen 3.6 tensor index.
///
/// The name sets are built from the publisher's `config.json` architecture
/// (40 layers, the explicit full-attention indices, 27 vision blocks) and the
/// tensor-member templates in the pinned index. MTP names are literal entries
/// copied from `model.safetensors.index.json`. Nothing here calls the production
/// classifier or derives expected categories from its output.
enum OfficialTensorMapFixture {
    static let expectedTotal = 1_045
    static let expectedTextResidentCount = 613
    static let expectedRoutedExpertCount = 80
    static let expectedVisionCount = 333
    static let expectedMTPCount = 19

    static let hiddenLayerCount = 40
    static let fullAttentionLayers: Set<Int> = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    static let visionBlockCount = 27

    static let globalTextNames = [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    ]

    static let sharedTextMembers = [
        "mlp.gate.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert_gate.weight",
    ]

    static let linearAttentionMembers = [
        "linear_attn.A_log",
        "linear_attn.conv1d.weight",
        "linear_attn.dt_bias",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
    ]

    static let fullAttentionMembers = [
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    ]

    static let visionBlockMembers = [
        "attn.proj.bias",
        "attn.proj.weight",
        "attn.qkv.bias",
        "attn.qkv.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.bias",
        "mlp.linear_fc2.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]

    static let mtpNames = [
        "mtp.fc.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ]

    static let visionSingletonNames = [
        "merger.linear_fc1.bias",
        "merger.linear_fc1.weight",
        "merger.linear_fc2.bias",
        "merger.linear_fc2.weight",
        "merger.norm.bias",
        "merger.norm.weight",
        "patch_embed.proj.bias",
        "patch_embed.proj.weight",
        "pos_embed.weight",
    ]

    /// Expected category per name from the independent architecture templates.
    static var expectedCategoryByName: [String: OfficialQwenTensorCategory] {
        var expected: [String: OfficialQwenTensorCategory] = [:]
        for name in globalTextNames { expected[name] = .textResident }

        for layer in 0..<hiddenLayerCount {
            let prefix = "model.language_model.layers.\(layer)."
            expected[prefix + "input_layernorm.weight"] = .textResident
            expected[prefix + "post_attention_layernorm.weight"] = .textResident
            for member in sharedTextMembers { expected[prefix + member] = .textResident }

            expected[prefix + "mlp.experts.down_proj"] = .routedExpert
            expected[prefix + "mlp.experts.gate_up_proj"] = .routedExpert

            let attentionMembers = fullAttentionLayers.contains(layer)
                ? fullAttentionMembers : linearAttentionMembers
            for member in attentionMembers { expected[prefix + member] = .textResident }
        }

        for block in 0..<visionBlockCount {
            let prefix = "model.visual.blocks.\(block)."
            for member in visionBlockMembers { expected[prefix + member] = .vision }
        }
        for name in visionSingletonNames { expected["model.visual." + name] = .vision }
        for name in mtpNames { expected[name] = .mtpPresentUnsupported }
        return expected
    }

    /// Sorted input is deterministic; each descriptor uses the pinned exact dtype string.
    static var descriptors: [OfficialQwenTensorDescriptor] {
        expectedCategoryByName.keys.sorted().map {
            OfficialQwenTensorDescriptor(name: $0, dataType: "BF16")
        }
    }
}
