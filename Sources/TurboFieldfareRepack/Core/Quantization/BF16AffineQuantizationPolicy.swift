import Foundation
import TurboFieldfareFormat

/// A closed storage decision for every tensor in the pinned official Qwen index.
///
/// The bit allocation is provisional until later model-quality qualification.
/// This policy establishes deterministic conversion and provenance only.
enum BF16AffineTensorStorage: String, Sendable, Equatable {
    case retainedBF16
    case affineInt4
    case affineInt8
    case omittedMTP

    var affineBitWidth: AffineBitWidth? {
        switch self {
        case .affineInt4: .int4
        case .affineInt8: .int8
        case .retainedBF16, .omittedMTP: nil
        }
    }
}

enum BF16AffineGroupingAxis: String, Sendable, Equatable {
    case lastDimension
}

struct BF16AffinePolicyDecision: Sendable, Equatable {
    let tensorName: String
    let storage: BF16AffineTensorStorage
    let manifestCategory: GTurboQuantizationCategoryV2?
    let groupingAxis: BF16AffineGroupingAxis?
    let groupSize: Int?
    let rule: String
    let isOverride: Bool
}

enum BF16AffineQuantizationPolicyError: Error, Sendable, Equatable {
    case unsupportedTensor(String)
    case sourceMustBeBF16(String)
    case invalidOfficialTensorSet
}

/// Production policy for the exact tensor names admitted by
/// `QwenOfficialTensorMap`. It never accepts a tensor through a broad prefix.
enum BF16AffineQuantizationPolicy {
    static let affineGroupSize = 64
    static let policyFormat = "TurboFieldfare.Qwen36.BF16AffinePolicy.v1"

    static let allDecisions: [BF16AffinePolicyDecision] = {
        makeDecisions().values.sorted { $0.tensorName < $1.tensorName }
    }()

    /// Canonical provenance identity. Every exact tensor decision, rule,
    /// override flag, grouping choice, and manifest category participates.
    static let policySHA256 = digest(canonicalLines: canonicalPolicyLines)

    /// Hashes the canonical record framing used by v2 provenance. Keeping the
    /// framing explicit lets callers prove that any policy-record mutation
    /// produces a different identity before a manifest is written.
    static func digest(canonicalLines: [String]) -> String {
        var stream = Sha256Stream()
        for line in canonicalLines {
            Data((line + "\n").utf8).withUnsafeBytes { stream.update($0) }
        }
        return stream.finalizeHexString()
    }

    static let canonicalPolicyLines: [String] = {
        let profiles = [
            "profile\tembedding\taffineInt8\t64\tbf16\tbf16",
            "profile\tattention\taffineInt4\t64\tbf16\tbf16",
            "profile\tlinearAttention\taffineInt8\t64\tbf16\tbf16",
            "profile\trouter\taffineInt8\t64\tbf16\tbf16",
            "profile\tsharedExpert\taffineInt4\t64\tbf16\tbf16",
            "profile\troutedExpert\taffineInt4\t64\tbf16\tbf16",
            "profile\toutputHead\taffineInt8\t64\tbf16\tbf16",
            "profile\tnormalization\tbf16\tnone\tnone\tnone",
            "profile\trecurrentState\tfp32\tnone\tnone\tnone",
        ]
        return [policyFormat] + profiles + allDecisions.map { decision in
            [
                decision.tensorName,
                decision.storage.rawValue,
                decision.manifestCategory?.rawValue ?? "none",
                decision.groupingAxis?.rawValue ?? "none",
                decision.groupSize.map(String.init) ?? "none",
                decision.rule,
                decision.isOverride ? "override" : "base",
            ].joined(separator: "\t")
        }
    }()

    /// The exhaustive v2 text profile consumed by later manifest planning.
    /// Vision regions use these same storage profiles where quantized; retained
    /// vision tensors have no quantization category in the companion manifest.
    static let manifestQuantizationGroups: [GTurboQuantizationGroupV2] = [
        affine(.embedding, .affineInt8),
        affine(.attention, .affineInt4),
        affine(.linearAttention, .affineInt8),
        affine(.router, .affineInt8),
        affine(.sharedExpert, .affineInt4),
        affine(.routedExpert, .affineInt4),
        affine(.outputHead, .affineInt8),
        GTurboQuantizationGroupV2(category: .normalization, storage: .bf16),
        GTurboQuantizationGroupV2(category: .recurrentState, storage: .fp32),
    ]

    static var overrides: [BF16AffinePolicyDecision] {
        allDecisions.filter(\.isOverride)
    }

    static func decision(
        for tensor: QwenOfficialTensorDescriptor
    ) throws -> BF16AffinePolicyDecision {
        guard tensor.dataType == .bf16 else {
            throw BF16AffineQuantizationPolicyError.sourceMustBeBF16(tensor.name)
        }
        guard let decision = decisionsByName[tensor.name] else {
            throw BF16AffineQuantizationPolicyError.unsupportedTensor(tensor.name)
        }
        return decision
    }

    /// Validates that the input is exactly the full P1 tensor set, then returns
    /// decisions in caller order. This binds policy coverage to the frozen P1
    /// classifier rather than maintaining a second permissive allowlist.
    static func decisions(
        for tensors: [QwenOfficialTensorDescriptor]
    ) throws -> [BF16AffinePolicyDecision] {
        do {
            _ = try QwenOfficialTensorMap.classify(tensors)
        } catch {
            throw BF16AffineQuantizationPolicyError.invalidOfficialTensorSet
        }
        return try tensors.map(decision(for:))
    }

    private static let decisionsByName = makeDecisions()

    private static func affine(
        _ category: GTurboQuantizationCategoryV2,
        _ storage: GTurboStorageTypeV2
    ) -> GTurboQuantizationGroupV2 {
        GTurboQuantizationGroupV2(
            category: category,
            storage: storage,
            groupSize: affineGroupSize,
            scaleType: "bf16",
            biasType: "bf16")
    }

    private static func makeDecisions() -> [String: BF16AffinePolicyDecision] {
        var result: [String: BF16AffinePolicyDecision] = [:]

        func add(
            _ name: String,
            storage: BF16AffineTensorStorage,
            category: GTurboQuantizationCategoryV2?,
            rule: String,
            override: Bool = false
        ) {
            precondition(result[name] == nil, "duplicate policy tensor: \(name)")
            let affine = storage.affineBitWidth != nil
            result[name] = BF16AffinePolicyDecision(
                tensorName: name,
                storage: storage,
                manifestCategory: category,
                groupingAxis: affine ? .lastDimension : nil,
                groupSize: affine ? affineGroupSize : nil,
                rule: rule,
                isOverride: override)
        }

        add("lm_head.weight", storage: .affineInt8, category: .outputHead,
            rule: "untied-output-head-int8")
        add("model.language_model.embed_tokens.weight", storage: .affineInt8,
            category: .embedding, rule: "token-embedding-int8")
        add("model.language_model.norm.weight", storage: .retainedBF16,
            category: .normalization, rule: "normalization-bf16")

        let shared = [
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
        ]
        let linearMembers = [
            "linear_attn.A_log", "linear_attn.conv1d.weight", "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
            "linear_attn.out_proj.weight",
        ]
        for layer in 0..<40 {
            let prefix = "model.language_model.layers.\(layer)."
            add(prefix + "input_layernorm.weight", storage: .retainedBF16,
                category: .normalization, rule: "normalization-bf16")
            add(prefix + "post_attention_layernorm.weight", storage: .retainedBF16,
                category: .normalization, rule: "normalization-bf16")
            add(prefix + "mlp.gate.weight", storage: .affineInt8,
                category: .router, rule: "router-int8")
            add(prefix + "mlp.shared_expert_gate.weight", storage: .affineInt8,
                category: .router, rule: "router-int8")
            for suffix in shared {
                add(prefix + suffix, storage: .affineInt4,
                    category: .sharedExpert, rule: "shared-expert-int4")
            }
            add(prefix + "mlp.experts.down_proj", storage: .affineInt4,
                category: .routedExpert, rule: "routed-expert-int4")
            add(prefix + "mlp.experts.gate_up_proj", storage: .affineInt4,
                category: .routedExpert, rule: "routed-expert-int4")

            if layer % 4 == 3 {
                for suffix in ["self_attn.k_norm.weight", "self_attn.q_norm.weight"] {
                    add(prefix + suffix, storage: .retainedBF16,
                        category: .normalization, rule: "normalization-bf16")
                }
                for suffix in [
                    "self_attn.k_proj.weight", "self_attn.o_proj.weight",
                    "self_attn.q_proj.weight", "self_attn.v_proj.weight",
                ] {
                    add(prefix + suffix, storage: .affineInt4,
                        category: .attention, rule: "full-attention-int4")
                }
            } else {
                add(prefix + "linear_attn.norm.weight", storage: .retainedBF16,
                    category: .normalization, rule: "normalization-bf16")
                for suffix in linearMembers {
                    add(prefix + suffix, storage: .affineInt8,
                        category: .linearAttention, rule: "linear-attention-int8")
                }
            }
        }

        let visionMembers = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight", "mlp.linear_fc2.bias",
            "mlp.linear_fc2.weight", "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
        ]
        for block in 0..<27 {
            let prefix = "model.visual.blocks.\(block)."
            for suffix in visionMembers {
                add(prefix + suffix, storage: .retainedBF16, category: nil,
                    rule: "vision-retained-bf16", override: true)
            }
        }
        for suffix in [
            "model.visual.merger.linear_fc1.bias", "model.visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc2.bias", "model.visual.merger.linear_fc2.weight",
            "model.visual.merger.norm.bias", "model.visual.merger.norm.weight",
            "model.visual.patch_embed.proj.bias", "model.visual.patch_embed.proj.weight",
            "model.visual.pos_embed.weight",
        ] {
            add(suffix, storage: .retainedBF16, category: nil,
                rule: "vision-retained-bf16", override: true)
        }

        for suffix in [
            "fc.weight", "layers.0.input_layernorm.weight", "layers.0.mlp.experts.down_proj",
            "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.gate.weight",
            "layers.0.mlp.shared_expert.down_proj.weight", "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight", "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight",
            "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.v_proj.weight", "norm.weight", "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        ] {
            add("mtp." + suffix, storage: .omittedMTP, category: nil,
                rule: "unsupported-mtp-omission")
        }
        return result
    }
}
