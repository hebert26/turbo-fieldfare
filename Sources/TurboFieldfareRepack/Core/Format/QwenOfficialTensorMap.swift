import Foundation

struct QwenOfficialTensorDescriptor: Sendable, Equatable {
    let name: String
    let dataType: SourceTensor.Dtype
}

enum QwenOfficialTensorCategory: Sendable, Equatable {
    case textResident
    case routedExpert
    case vision
    case mtpOmitted
}

/// Recognizes only the tensor names in the pinned official Qwen 3.6 index.
/// The expected names are generated from the pinned architecture rather than
/// accepted by broad prefixes, so a name cannot be admitted by resemblance.
enum QwenOfficialTensorMap {
    static func classify(
        _ tensors: [QwenOfficialTensorDescriptor]
    ) throws -> [QwenOfficialTensorCategory] {
        guard tensors.count == expectedCategories.count,
              Set(tensors.map(\.name)).count == tensors.count,
              tensors.allSatisfy({ $0.dataType == .bf16 }) else {
            throw QwenOfficialValidationError.invalidTensorSet
        }

        var categories: [QwenOfficialTensorCategory] = []
        categories.reserveCapacity(tensors.count)
        for tensor in tensors {
            guard let category = expectedCategories[tensor.name] else {
                throw QwenOfficialValidationError.invalidTensor
            }
            categories.append(category)
        }
        guard Set(tensors.map(\.name)) == Set(expectedCategories.keys) else {
            throw QwenOfficialValidationError.invalidTensorSet
        }
        return categories
    }

    private static let expectedCategories: [String: QwenOfficialTensorCategory] = {
        var categories: [String: QwenOfficialTensorCategory] = [
            "lm_head.weight": .textResident,
            "model.language_model.embed_tokens.weight": .textResident,
            "model.language_model.norm.weight": .textResident
        ]

        let commonMLP = [
            "mlp.gate.weight", "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert_gate.weight"
        ]
        let linearAttention = [
            "linear_attn.A_log", "linear_attn.conv1d.weight", "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight", "linear_attn.out_proj.weight"
        ]
        let fullAttention = [
            "self_attn.k_norm.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight"
        ]
        for layer in 0..<40 {
            let prefix = "model.language_model.layers.\(layer)."
            categories[prefix + "input_layernorm.weight"] = .textResident
            categories[prefix + "post_attention_layernorm.weight"] = .textResident
            for suffix in commonMLP { categories[prefix + suffix] = .textResident }
            categories[prefix + "mlp.experts.down_proj"] = .routedExpert
            categories[prefix + "mlp.experts.gate_up_proj"] = .routedExpert
            for suffix in layer % 4 == 3 ? fullAttention : linearAttention {
                categories[prefix + suffix] = .textResident
            }
        }

        let visionMembers = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight", "mlp.linear_fc2.bias",
            "mlp.linear_fc2.weight", "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight"
        ]
        for block in 0..<27 {
            let prefix = "model.visual.blocks.\(block)."
            for suffix in visionMembers { categories[prefix + suffix] = .vision }
        }
        for suffix in [
            "model.visual.merger.linear_fc1.bias", "model.visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc2.bias", "model.visual.merger.linear_fc2.weight",
            "model.visual.merger.norm.bias", "model.visual.merger.norm.weight",
            "model.visual.patch_embed.proj.bias", "model.visual.patch_embed.proj.weight",
            "model.visual.pos_embed.weight"
        ] { categories[suffix] = .vision }

        for suffix in [
            "fc.weight", "layers.0.input_layernorm.weight", "layers.0.mlp.experts.down_proj",
            "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.gate.weight",
            "layers.0.mlp.shared_expert.down_proj.weight", "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight", "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight",
            "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.v_proj.weight", "norm.weight", "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight"
        ] { categories["mtp." + suffix] = .mtpOmitted }
        return categories
    }()
}
