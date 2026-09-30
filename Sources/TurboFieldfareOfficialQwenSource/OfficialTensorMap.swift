/// One tensor name and the exact Safetensors dtype token from its bounded header.
/// This descriptor is independent of RepackCore's SourceTensor and of any file path.
public struct OfficialQwenTensorDescriptor: Sendable, Equatable {
    public let name: String
    public let dataType: String

    public init(name: String, dataType: String) {
        self.name = name
        self.dataType = dataType
    }
}

public enum OfficialQwenTensorCategory: Sendable, Equatable, Hashable {
    case textResident
    case routedExpert
    case vision
    /// Present in the official index, but never a runnable or routed expert.
    case mtpPresentUnsupported
}

public enum OfficialQwenTensorMapError: Error, Sendable, Equatable {
    case invalidTensorSet
    case invalidTensor
}

/// Recognizes only the 1,045 names in the pinned official Qwen 3.6 index.
/// The sole classification definition is generated from the pinned architecture,
/// not broad prefixes. Validation uses only supplied metadata and never opens shards.
public enum OfficialQwenTensorMap {
    /// Returns one category per input descriptor, in input order, in expected
    /// O(n) time and O(n) auxiliary space. Any ordering of the set is accepted.
    public static func classify(
        _ tensors: [OfficialQwenTensorDescriptor]
    ) throws -> [OfficialQwenTensorCategory] {
        guard tensors.count == expectedCategories.count,
              Set(tensors.map(\.name)).count == tensors.count,
              tensors.allSatisfy({ $0.dataType == "BF16" }) else {
            throw OfficialQwenTensorMapError.invalidTensorSet
        }

        var categories: [OfficialQwenTensorCategory] = []
        categories.reserveCapacity(tensors.count)
        for tensor in tensors {
            guard let category = expectedCategories[tensor.name] else {
                throw OfficialQwenTensorMapError.invalidTensor
            }
            categories.append(category)
        }
        guard Set(tensors.map(\.name)) == Set(expectedCategories.keys) else {
            throw OfficialQwenTensorMapError.invalidTensorSet
        }
        return categories
    }

    private static let expectedCategories: [String: OfficialQwenTensorCategory] = {
        var categories: [String: OfficialQwenTensorCategory] = [
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
        ] { categories["mtp." + suffix] = .mtpPresentUnsupported }
        return categories
    }()
}
