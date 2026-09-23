import Testing
@testable import TurboFieldfareRepackCore

@Suite struct BF16AffineQuantizationPolicyTests {
    @Test func closedPolicyHasTheNineP2LegalProfilesAndStableCanonicalIdentity() throws {
        let groups = BF16AffineQuantizationPolicy.manifestQuantizationGroups
        #expect(groups.count == 9)
        #expect(Set(groups.map(\.category)).count == 9)

        let storage = Dictionary(uniqueKeysWithValues: groups.map { ($0.category, $0.storage) })
        #expect(storage[.embedding] == .affineInt8)
        #expect(storage[.attention] == .affineInt4)
        #expect(storage[.linearAttention] == .affineInt8)
        #expect(storage[.router] == .affineInt8)
        #expect(storage[.sharedExpert] == .affineInt4)
        #expect(storage[.routedExpert] == .affineInt4)
        #expect(storage[.outputHead] == .affineInt8)
        #expect(storage[.normalization] == .bf16)
        #expect(storage[.recurrentState] == .fp32)
        #expect(BF16AffineQuantizationPolicy.policySHA256.count == 64)
        let decisionLines = Array(BF16AffineQuantizationPolicy.canonicalPolicyLines.dropFirst(10))
        #expect(decisionLines == decisionLines.sorted())

        var changed = BF16AffineQuantizationPolicy.canonicalPolicyLines
        let ruleIndex = try #require(changed.firstIndex { $0.contains("full-attention-int4") })
        changed[ruleIndex] = changed[ruleIndex].replacingOccurrences(
            of: "full-attention-int4", with: "changed-test-rule")
        #expect(BF16AffineQuantizationPolicy.digest(canonicalLines: changed) !=
            BF16AffineQuantizationPolicy.policySHA256)
    }

    @Test func exactRepresentativesUseP2LegalMetadataAndVisionAndMTPStayUncategorized() throws {
        func decision(_ name: String) throws -> BF16AffinePolicyDecision {
            try BF16AffineQuantizationPolicy.decision(for: .init(name: name, dataType: .bf16))
        }
        let attention = try decision("model.language_model.layers.3.self_attn.q_proj.weight")
        #expect(attention.storage == .affineInt4)
        #expect(attention.manifestCategory == .attention)
        #expect(attention.groupingAxis == .lastDimension)
        #expect(attention.groupSize == 64)

        let linear = try decision("model.language_model.layers.0.linear_attn.A_log")
        #expect(linear.storage == .affineInt8)
        #expect(linear.manifestCategory == .linearAttention)
        #expect(linear.groupSize == 64)

        let norm = try decision("model.language_model.layers.0.linear_attn.norm.weight")
        #expect(norm.storage == .retainedBF16)
        #expect(norm.manifestCategory == .normalization)
        #expect(norm.groupSize == nil)

        let vision = try decision("model.visual.blocks.0.attn.qkv.weight")
        #expect(vision.storage == .retainedBF16)
        #expect(vision.manifestCategory == nil)
        #expect(vision.groupSize == nil)
        #expect(vision.isOverride)

        let mtp = try decision("mtp.fc.weight")
        #expect(mtp.storage == .omittedMTP)
        #expect(mtp.manifestCategory == nil)
        #expect(mtp.groupingAxis == nil)
    }

    @Test func allFrozenDescriptorsHaveExactlyOneLegalPolicyDecision() throws {
        let descriptors = Qwen36OfficialMetadata.indexNames.map {
            QwenOfficialTensorDescriptor(name: $0, dataType: .bf16)
        }
        let decisions = try BF16AffineQuantizationPolicy.decisions(for: descriptors)
        #expect(descriptors.count == 1045)
        #expect(decisions.count == 1045)
        #expect(Set(decisions.map(\.tensorName)).count == 1045)
        #expect(decisions.filter { $0.storage == .omittedMTP }.count == 19)
        #expect(decisions.filter { $0.manifestCategory == .embedding }.count == 1)
        #expect(decisions.filter { $0.manifestCategory == .outputHead }.count == 1)
        #expect(decisions.allSatisfy { decision in
            if decision.storage.affineBitWidth == nil {
                return decision.groupingAxis == nil && decision.groupSize == nil
            }
            return decision.groupingAxis == .lastDimension && decision.groupSize == 64
        })
    }

    @Test func policyRejectsUnknownAndNonBF16Descriptors() {
        #expect(throws: BF16AffineQuantizationPolicyError.self) {
            try BF16AffineQuantizationPolicy.decision(for: .init(name: "not.a.pinned.tensor", dataType: .bf16))
        }
        #expect(throws: BF16AffineQuantizationPolicyError.self) {
            try BF16AffineQuantizationPolicy.decision(for: .init(
                name: "lm_head.weight", dataType: .fp32))
        }
    }
}
