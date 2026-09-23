import Foundation
import Testing

@testable import TurboFieldfareRepackCore

/// Independent classification tests for the pinned official Qwen3.6-35B-A3B tensor index.
///
/// The expected values come from two sources that are not the classifier under test:
/// 1. `Qwen36OfficialMetadata`, a frozen literal fixture derived from the official index.
/// 2. A structural derivation built here from the pinned `config.json` values (40 layers,
///    `layer_types`, `full_attention_interval`, `vision_config.depth`, `mtp_num_hidden_layers`)
///    plus the documented architecture member templates.
///
/// Phase 1 has no planner, so every negative case asserts that `classify` throws before
/// returning an accepted value. The consuming pre-planning gate is proven by Phase 6.
@Suite struct QwenOfficialTensorMapTests {
    private typealias Fixture = Qwen36OfficialMetadata

    private static let fixtureNames = Fixture.indexNames

    private static func descriptors(
        _ names: [String],
        dataType: SourceTensor.Dtype = .bf16
    ) -> [QwenOfficialTensorDescriptor] {
        names.map { QwenOfficialTensorDescriptor(name: $0, dataType: dataType) }
    }

    /// Returns the thrown `QwenOfficialValidationError`, `nil` when the call was accepted, and
    /// records an issue only for an unexpected error type. Callers assert the exact case, so an
    /// accepted input fails the negative-control assertion rather than passing silently.
    private static func rejection(
        _ names: [String],
        dataType: SourceTensor.Dtype = .bf16
    ) -> QwenOfficialValidationError? {
        do {
            _ = try QwenOfficialTensorMap.classify(descriptors(names, dataType: dataType))
            return nil
        } catch let error as QwenOfficialValidationError {
            return error
        } catch {
            Issue.record("classify threw a non-QwenOfficialValidationError: \(error)")
            return nil
        }
    }

    /// The fixture set with one name replaced by another. The replacement is chosen so the
    /// set keeps 1045 unique names and only the classified name is wrong.
    private static func replacing(_ victim: String, with replacement: String) -> [String] {
        var names = fixtureNames
        guard let index = names.firstIndex(of: victim) else {
            Issue.record("fixture does not contain the name this case replaces: \(victim)")
            return names
        }
        names[index] = replacement
        return names
    }

    private static func categoryByName(
        _ names: [String],
        _ categories: [QwenOfficialTensorCategory]
    ) -> [String: QwenOfficialTensorCategory] {
        var map = [String: QwenOfficialTensorCategory](minimumCapacity: names.count)
        for (index, name) in names.enumerated() { map[name] = categories[index] }
        return map
    }

    // MARK: - Acceptance

    @Test func everyIndexedNameClassifiesExactlyOnceAgainstTheFrozenFixture() throws {
        let names = Self.fixtureNames
        #expect(names.count == 1045)

        let categories = try QwenOfficialTensorMap.classify(Self.descriptors(names))
        try #require(categories.count == names.count)

        var missingFromFixture: [String] = []
        var mismatched: [String] = []
        for (index, name) in names.enumerated() {
            guard let expected = Fixture.expectedCategoryByName[name] else {
                missingFromFixture.append(name)
                continue
            }
            if categories[index] != expected { mismatched.append(name) }
        }

        #expect(missingFromFixture.isEmpty)
        #expect(mismatched.isEmpty, "misclassified: \(mismatched.prefix(5))")
        // Control on the rejection helper: the unmutated pinned set reports no error, so the
        // negative controls below are not vacuously satisfied.
        #expect(Self.rejection(names) == nil)
    }

    @Test func categoryCountsMatchTheRecordedIndexSplit() throws {
        let categories = try QwenOfficialTensorMap.classify(Self.descriptors(Self.fixtureNames))
        let counts = Dictionary(grouping: categories, by: { $0 }).mapValues(\.count)

        #expect(counts[.textResident] == 613)
        #expect(counts[.routedExpert] == 80)
        #expect(counts[.vision] == 333)
        #expect(counts[.mtpOmitted] == 19)
        let classifiedTotal = counts.values.reduce(0, +)
        #expect(classifiedTotal == 1045)
        let recordedTotal = Fixture.textResidentNames.count + Fixture.routedExpertNames.count
            + Fixture.visionNames.count + Fixture.mtpOmittedNames.count
        #expect(recordedTotal == 1045)
        #expect(counts == Fixture.expectedCounts)
    }

    @Test func categoriesAreExclusiveAndCoverEveryDescriptorOnce() throws {
        let names = Self.fixtureNames
        let categories = try QwenOfficialTensorMap.classify(Self.descriptors(names))
        let map = Self.categoryByName(names, categories)

        #expect(map.count == 1045)
        #expect(Set(map.keys) == Set(names))

        var sets: [QwenOfficialTensorCategory: Set<String>] = [:]
        for (name, category) in map { sets[category, default: []].insert(name) }
        let union = sets.values.reduce(into: Set<String>()) { $0.formUnion($1) }
        #expect(union.count == 1045)
        for (category, members) in sets {
            for (otherCategory, otherMembers) in sets where otherCategory != category {
                #expect(members.isDisjoint(with: otherMembers),
                        "\(category) overlaps \(otherCategory)")
            }
        }
        let classifiedTotal = sets.values.map(\.count).reduce(0, +)
        #expect(classifiedTotal == 1045)
    }

    // MARK: - Structural derivation from the pinned config

    @Test func classifierAgreesWithTheConfigDerivedNameSet() throws {
        let names = Self.fixtureNames
        let categories = try QwenOfficialTensorMap.classify(Self.descriptors(names))
        let actual = Self.categoryByName(names, categories)

        // Layer split derived from `config.json` `text_config.layer_types`, not from the
        // fixture's per-category name arrays.
        #expect(Fixture.layerTypes.count == Fixture.numHiddenLayers)
        let fullLayerIndices = Fixture.layerTypes.enumerated()
            .filter { $0.element == "full_attention" }.map(\.offset)
        let linearLayerIndices = Fixture.layerTypes.enumerated()
            .filter { $0.element == "linear_attention" }.map(\.offset)
        #expect(fullLayerIndices == Array(stride(from: 3, to: Fixture.numHiddenLayers, by: 4)))
        let expectedFullCount = Fixture.numHiddenLayers / Fixture.fullAttentionInterval
        #expect(fullLayerIndices.count == expectedFullCount)
        #expect(linearLayerIndices.count == 30)
        #expect(fullLayerIndices.count + linearLayerIndices.count == Fixture.numHiddenLayers)
        #expect(Fixture.fullAttentionInterval == 4)
        #expect(Set(fullLayerIndices).isDisjoint(with: Set(linearLayerIndices)))

        let commonMLP = [
            "mlp.gate.weight", "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert_gate.weight",
        ]
        let expertProjections = [
            "mlp.experts.down_proj": QwenOfficialTensorCategory.routedExpert,
            "mlp.experts.gate_up_proj": .routedExpert,
        ]
        let linearAttention = [
            "linear_attn.A_log", "linear_attn.conv1d.weight", "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight", "linear_attn.out_proj.weight",
        ]
        let fullAttention = [
            "self_attn.k_norm.weight", "self_attn.k_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight",
        ]

        func layerMembers(forLayer index: Int, prefix: String) -> [String: QwenOfficialTensorCategory] {
            var members: [String: QwenOfficialTensorCategory] = [
                "\(prefix)input_layernorm.weight": .textResident,
                "\(prefix)post_attention_layernorm.weight": .textResident,
            ]
            for suffix in commonMLP { members["\(prefix)\(suffix)"] = .textResident }
            for (suffix, category) in expertProjections { members["\(prefix)\(suffix)"] = category }
            let attention = Fixture.layerTypes[index] == "full_attention" ? fullAttention : linearAttention
            for suffix in attention { members["\(prefix)\(suffix)"] = .textResident }
            return members
        }

        var expected: [String: QwenOfficialTensorCategory] = [
            "lm_head.weight": .textResident,
            "model.language_model.embed_tokens.weight": .textResident,
            "model.language_model.norm.weight": .textResident,
        ]

        for layer in 0..<Fixture.numHiddenLayers {
            expected.merge(
                layerMembers(forLayer: layer, prefix: "model.language_model.layers.\(layer).")
            ) { _, new in new }
        }

        // Member counts derived from the two attention shapes.
        for layer in linearLayerIndices {
            let members = expected.keys.filter { $0.hasPrefix("model.language_model.layers.\(layer).") }
            #expect(members.count == 18, "layer \(layer) is linear_attention")
        }
        for layer in fullLayerIndices {
            let members = expected.keys.filter { $0.hasPrefix("model.language_model.layers.\(layer).") }
            #expect(members.count == 15, "layer \(layer) is full_attention")
        }

        let visionBlockMembers = [
            "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias", "attn.qkv.weight",
            "mlp.linear_fc1.bias", "mlp.linear_fc1.weight", "mlp.linear_fc2.bias",
            "mlp.linear_fc2.weight", "norm1.bias", "norm1.weight", "norm2.bias", "norm2.weight",
        ]
        for block in 0..<Fixture.visionDepth {
            for suffix in visionBlockMembers {
                expected["model.visual.blocks.\(block).\(suffix)"] = .vision
            }
        }
        for suffix in [
            "merger.linear_fc1.bias", "merger.linear_fc1.weight", "merger.linear_fc2.bias",
            "merger.linear_fc2.weight", "merger.norm.bias", "merger.norm.weight",
            "patch_embed.proj.bias", "patch_embed.proj.weight", "pos_embed.weight",
        ] {
            expected["model.visual.\(suffix)"] = .vision
        }

        // MTP is one layer (`mtp_num_hidden_layers`) that shares the full-attention member
        // template, plus its four standalone tensors.
        let mtpPrefix = "mtp.layers.0."
        for suffix in [
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "mlp.experts.down_proj", "mlp.experts.gate_up_proj",
        ] {
            expected["\(mtpPrefix)\(suffix)"] = .mtpOmitted
        }
        for suffix in commonMLP { expected["\(mtpPrefix)\(suffix)"] = .mtpOmitted }
        for suffix in fullAttention { expected["\(mtpPrefix)\(suffix)"] = .mtpOmitted }
        for suffix in [
            "fc.weight", "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ] {
            expected["mtp.\(suffix)"] = .mtpOmitted
        }

        #expect(Fixture.visionDepth == 27)
        #expect(Fixture.mtpNumHiddenLayers == 1)
        #expect(expected.count == 1045)
        #expect(expected.values.filter { $0 == .textResident }.count == 613)
        #expect(expected.values.filter { $0 == .routedExpert }.count == 80)
        #expect(expected.values.filter { $0 == .vision }.count == 333)
        #expect(expected.values.filter { $0 == .mtpOmitted }.count == 19)
        #expect(actual == expected)
    }

    @Test func routedExpertsAreOnlyTheTwoPerLayerExpertProjections() throws {
        let names = Self.fixtureNames
        let categories = try QwenOfficialTensorMap.classify(Self.descriptors(names))
        var routed: Set<String> = []
        for (index, name) in names.enumerated() where categories[index] == .routedExpert {
            routed.insert(name)
        }

        #expect(routed.count == 80)
        let expectedRoutedCount = 2 * Fixture.numHiddenLayers
        #expect(routed.count == expectedRoutedCount)
        for name in routed {
            #expect(name.hasPrefix("model.language_model.layers."))
            #expect(name.hasSuffix(".mlp.experts.down_proj") || name.hasSuffix(".mlp.experts.gate_up_proj"))
            #expect(!name.hasSuffix(".weight"))
            #expect(!name.hasPrefix("mtp."))
        }
        #expect(routed.contains("model.language_model.layers.0.mlp.experts.down_proj"))
        #expect(routed.contains("model.language_model.layers.39.mlp.experts.gate_up_proj"))
        // The MTP expert tensors are intentional omissions, never routed experts.
        #expect(!routed.contains("mtp.layers.0.mlp.experts.down_proj"))
        #expect(!routed.contains("mtp.layers.0.mlp.experts.gate_up_proj"))
    }

    @Test func visionNamesStayInsideThePinnedVisualTower() throws {
        let names = Self.fixtureNames
        let categories = try QwenOfficialTensorMap.classify(Self.descriptors(names))
        var vision: Set<String> = []
        for (index, name) in names.enumerated() where categories[index] == .vision {
            vision.insert(name)
        }

        #expect(vision.count == 333)
        let expectedVisionCount = Fixture.visionDepth * 12 + 9
        #expect(vision.count == expectedVisionCount)

        var blockIndices: Set<Int> = []
        for name in vision {
            #expect(name.hasPrefix("model.visual."))
            let isBlock = name.hasPrefix("model.visual.blocks.")
            let isSingleton = [
                "model.visual.merger.", "model.visual.patch_embed.",
            ].contains { name.hasPrefix($0) } || name == "model.visual.pos_embed.weight"
            #expect(isBlock || isSingleton, "unexpected vision name shape: \(name)")
            if isBlock, let block = Self.blockIndex(of: name) { blockIndices.insert(block) }
        }

        // Block bounds come from `config.json` `vision_config.depth`.
        #expect(blockIndices == Set(0..<Fixture.visionDepth))
        #expect(blockIndices.count == 27)
        #expect(vision.contains("model.visual.blocks.0.norm1.weight"))
        #expect(vision.contains("model.visual.blocks.26.norm2.bias"))
        #expect(vision.contains("model.visual.merger.norm.weight"))
        #expect(vision.contains("model.visual.patch_embed.proj.weight"))
        #expect(vision.contains("model.visual.pos_embed.weight"))
        // Language-model names never classify as vision and vice versa.
        #expect(!vision.contains("model.language_model.norm.weight"))
    }

    @Test func exactlyNineteenMtpNamesAreIntentionalOmissions() throws {
        let names = Self.fixtureNames
        let categories = try QwenOfficialTensorMap.classify(Self.descriptors(names))
        var mtp: Set<String> = []
        for (index, name) in names.enumerated() where categories[index] == .mtpOmitted {
            mtp.insert(name)
        }

        #expect(mtp.count == 19)
        #expect(mtp.count == Fixture.mtpNameCount)
        #expect(mtp == Set(Fixture.mtpOmittedNames))
        #expect(mtp.allSatisfy { $0.hasPrefix("mtp.") })
        #expect(mtp.contains("mtp.layers.0.mlp.experts.gate_up_proj"))
        #expect(mtp.contains("mtp.layers.0.self_attn.k_proj.weight"))
        #expect(mtp.contains("mtp.pre_fc_norm_hidden.weight"))
        // No language-model or vision name is an omission.
        #expect(!mtp.contains("model.language_model.layers.3.self_attn.k_proj.weight"))
    }

    private static func blockIndex(of name: String) -> Int? {
        let marker = "model.visual.blocks."
        guard name.hasPrefix(marker) else { return nil }
        let rest = name.dropFirst(marker.count)
        let digits = rest.prefix(while: { $0.isNumber })
        return digits.isEmpty ? nil : Int(digits)
    }

    // MARK: - Negative controls: names

    @Test(arguments: [
        // A full-attention member on a linear layer.
        ("model.language_model.layers.4.linear_attn.A_log",
         "model.language_model.layers.4.self_attn.k_proj.weight"),
        // A linear-attention member on a full layer (layer 3 is full_attention).
        ("model.language_model.layers.3.self_attn.k_proj.weight",
         "model.language_model.layers.3.linear_attn.A_log"),
        // Layer index one past the pinned bound.
        ("model.language_model.layers.0.input_layernorm.weight",
         "model.language_model.layers.40.input_layernorm.weight"),
        // Vision block one past `vision_config.depth`.
        ("model.visual.blocks.0.norm1.weight", "model.visual.blocks.27.norm1.weight"),
        // MTP layer one past `mtp_num_hidden_layers`.
        ("mtp.fc.weight", "mtp.layers.1.input_layernorm.weight"),
        // Routed expert renamed to a member that does not exist.
        ("model.language_model.layers.0.mlp.experts.down_proj",
         "model.language_model.layers.0.mlp.experts.up_proj"),
        // Routed expert with a `.weight` suffix it does not have.
        ("model.language_model.layers.0.mlp.experts.down_proj",
         "model.language_model.layers.0.mlp.experts.down_proj.weight"),
        // Shared expert renamed onto the routed-expert namespace.
        ("model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
         "model.language_model.layers.0.mlp.experts.down_proj.weight"),
        // Global head with a bias instead of a weight.
        ("lm_head.weight", "lm_head.bias"),
        // Truncated prefix that a prefix-only matcher could accept.
        ("model.language_model.layers.1.linear_attn.A_log",
         "model.language_model.layers.1.linear_attn.A"),
        // Vision merger member that does not exist.
        ("model.visual.merger.norm.weight", "model.visual.merger.linear_fc3.weight"),
        // Case and separator changes.
        ("model.language_model.norm.weight", "Model.Language_Model.norm.weight"),
        // Leading whitespace.
        ("lm_head.weight", " lm_head.weight"),
        // Trailing whitespace.
        ("model.language_model.norm.weight", "model.language_model.norm.weight "),
        // Empty name.
        ("lm_head.weight", ""),
        // Shard-style name that never appears in the index.
        ("lm_head.weight", "model-00001-of-00026.safetensors"),
    ])
    func unknownOrMisplacedNamesAreRejected(victim: String, replacement: String) {
        let names = Self.replacing(victim, with: replacement)
        #expect(Self.rejection(names) == .invalidTensor)
    }

    // MARK: - Negative controls: set size, uniqueness, dtype

    @Test func missingNameIsRejected() {
        let names = Array(Self.fixtureNames.dropLast())
        #expect(names.count == 1044)
        #expect(Self.rejection(names) == .invalidTensorSet)
    }

    @Test func extraNameIsRejected() {
        var names = Self.fixtureNames
        names.append("model.language_model.layers.0.mlp.experts.down_proj.weight")
        #expect(names.count == 1046)
        #expect(Self.rejection(names) == .invalidTensorSet)
    }

    @Test func duplicateNameIsRejected() {
        let names = Self.replacing(
            "model.language_model.norm.weight", with: "lm_head.weight"
        )
        #expect(Set(names).count == 1044)
        #expect(names.count == 1045)
        #expect(Self.rejection(names) == .invalidTensorSet)
    }

    @Test func emptyDescriptorListIsRejected() {
        #expect(Self.rejection([]) == .invalidTensorSet)
    }

    @Test func nonBfloat16DescriptorIsRejected() {
        let names = Self.fixtureNames
        // All-fp32 set.
        #expect(Self.rejection(names, dataType: .fp32) == .invalidTensorSet)
        // One fp32 descriptor among 1044 bf16 descriptors.
        var descriptors = Self.descriptors(Array(names.dropLast()))
        descriptors.append(
            QwenOfficialTensorDescriptor(name: names[names.count - 1], dataType: .fp32)
        )
        do {
            _ = try QwenOfficialTensorMap.classify(descriptors)
            Issue.record("classify accepted one non-BF16 descriptor")
        } catch let error as QwenOfficialValidationError {
            #expect(error == .invalidTensorSet)
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}
