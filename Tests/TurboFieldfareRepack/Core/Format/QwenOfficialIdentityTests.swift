import Foundation
import Testing

@testable import TurboFieldfareRepackCore

/// Independent identity tests for the pinned official Qwen3.6-35B-A3B source contract.
///
/// Every value comes from `Qwen36OfficialMetadata`, which was transcribed from the official
/// metadata sidecars (`SHA256SUMS`, `config.json`, `tokenizer_config.json`,
/// `preprocessor_config.json`, `model.safetensors.index.json`) before the production
/// validator was written. Each case mutates exactly one field of otherwise valid metadata.
///
/// Phase 1 has no planner, so every negative case asserts that `validate` throws before
/// returning an accepted value. The consuming pre-planning gate is proven by Phase 6.
@Suite struct QwenOfficialIdentityTests {
    private typealias Fixture = Qwen36OfficialMetadata

    // MARK: - Builders

    private static func validConfiguration(
        modelType: String = Fixture.modelType,
        architectures: [String] = [Fixture.architecture],
        textModelType: String = Fixture.textModelType,
        numHiddenLayers: Int = Fixture.numHiddenLayers,
        hiddenSize: Int = Fixture.hiddenSize,
        numAttentionHeads: Int = Fixture.numAttentionHeads,
        numKeyValueHeads: Int = Fixture.numKeyValueHeads,
        headDim: Int = Fixture.headDim,
        numExperts: Int = Fixture.numExperts,
        numExpertsPerToken: Int = Fixture.numExpertsPerToken,
        vocabSize: Int = Fixture.vocabSize,
        layerTypes: [String] = Fixture.layerTypes,
        fullAttentionInterval: Int = Fixture.fullAttentionInterval,
        mtpNumHiddenLayers: Int = Fixture.mtpNumHiddenLayers,
        tieWordEmbeddings: Bool = Fixture.tieWordEmbeddings
    ) -> QwenOfficialConfiguration {
        QwenOfficialConfiguration(
            modelType: modelType, architectures: architectures, textModelType: textModelType,
            numHiddenLayers: numHiddenLayers, hiddenSize: hiddenSize,
            numAttentionHeads: numAttentionHeads, numKeyValueHeads: numKeyValueHeads,
            headDim: headDim, numExperts: numExperts, numExpertsPerToken: numExpertsPerToken,
            vocabSize: vocabSize, layerTypes: layerTypes,
            fullAttentionInterval: fullAttentionInterval,
            mtpNumHiddenLayers: mtpNumHiddenLayers, tieWordEmbeddings: tieWordEmbeddings)
    }

    private static func validTokenizerBinding(
        vocabSize: Int = Fixture.vocabSize,
        imageTokenID: Int = Fixture.imageTokenID,
        tokenizerClass: String = Fixture.tokenizerClass
    ) -> QwenOfficialTokenizerBinding {
        QwenOfficialTokenizerBinding(
            vocabSize: vocabSize, imageTokenID: imageTokenID, tokenizerClass: tokenizerClass)
    }

    private static func validProcessorBinding(
        processorClass: String = Fixture.processorClass,
        imageProcessorType: String = Fixture.imageProcessorType,
        patchSize: Int = Fixture.preprocessorPatchSize,
        mergeSize: Int = Fixture.preprocessorMergeSize
    ) -> QwenOfficialProcessorBinding {
        QwenOfficialProcessorBinding(
            processorClass: processorClass, imageProcessorType: imageProcessorType,
            patchSize: patchSize, mergeSize: mergeSize)
    }

    private static func metadata(
        repository: String = Fixture.repository,
        revision: String = Fixture.revision,
        sidecarSHA256: [String: String] = Fixture.sidecarSHA256,
        configuration: QwenOfficialConfiguration = validConfiguration(),
        tokenizerBinding: QwenOfficialTokenizerBinding = validTokenizerBinding(),
        processorBinding: QwenOfficialProcessorBinding = validProcessorBinding()
    ) -> QwenOfficialSourceMetadata {
        QwenOfficialSourceMetadata(
            repository: repository, revision: revision, sidecarSHA256: sidecarSHA256,
            configuration: configuration, tokenizerBinding: tokenizerBinding,
            processorBinding: processorBinding)
    }

    /// Returns the thrown error, `nil` when validation was accepted, and records an issue only
    /// for an unexpected error type. Callers assert the exact case, so an accepted input fails the
    /// negative-control assertion rather than passing silently.
    private static func rejection(
        _ metadata: QwenOfficialSourceMetadata
    ) -> QwenOfficialValidationError? {
        do {
            try QwenOfficialIdentity.validate(metadata)
            return nil
        } catch let error as QwenOfficialValidationError {
            return error
        } catch {
            Issue.record("validate threw a non-QwenOfficialValidationError: \(error)")
            return nil
        }
    }

    // MARK: - Fixture integrity

    @Test func fixturePinsTheSevenBehaviourDefiningSidecars() {
        let pinned: Set<String> = [
            "config.json", "configuration.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "preprocessor_config.json", "model.safetensors.index.json",
        ]
        #expect(Set(Fixture.sidecarSHA256.keys) == pinned)
        #expect(Set(Fixture.pinnedSidecarNames) == pinned)
        #expect(Fixture.pinnedSidecarNames.count == 7)
        #expect(!Fixture.pinnedSidecarNames.contains("chat_template.jinja"))

        for (name, digest) in Fixture.sidecarSHA256 {
            #expect(digest.count == 64, "\(name) digest length")
            #expect(digest == digest.lowercased(), "\(name) is not lowercase")
            #expect(digest.allSatisfy { $0.isNumber || ("a"..."f").contains($0) },
                    "\(name) is not hexadecimal")
        }

        // Spot-check two values against `SHA256SUMS`.
        #expect(Fixture.sidecarSHA256["config.json"]
            == "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99")
        #expect(Fixture.sidecarSHA256["model.safetensors.index.json"]
            == "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83")
    }

    @Test func fixtureIdentityAndIndexSplitAreSelfConsistent() {
        #expect(Fixture.repository == "Qwen/Qwen3.6-35B-A3B")
        #expect(Fixture.revision == "995ad96eacd98c81ed38be0c5b274b04031597b0")
        #expect(Fixture.revision.count == 40)

        let groups = [
            Fixture.textResidentNames, Fixture.routedExpertNames,
            Fixture.visionNames, Fixture.mtpOmittedNames,
        ]
        #expect(groups.map(\.count) == [613, 80, 333, 19])
        let all = groups.flatMap { $0 }
        #expect(all.count == 1045)
        #expect(Set(all).count == 1045, "the fixture repeats a name")
        #expect(Fixture.indexNames.count == 1045)
        #expect(Fixture.tensorCount == 1045)
        #expect(Fixture.mtpNameCount == 19)
        #expect(Fixture.expectedCounts[.textResident] == 613)
        #expect(Fixture.expectedCounts[.routedExpert] == 80)
        #expect(Fixture.expectedCounts[.vision] == 333)
        #expect(Fixture.expectedCounts[.mtpOmitted] == 19)
    }

    @Test func fixturePinsTheQwen36ArchitectureNames() {
        // `qwen3_5_moe` is the field value this exact Qwen3.6 checkpoint uses. It is not a
        // substitute for Qwen3.5 and does not by itself authorise a different source.
        #expect(Fixture.modelType == "qwen3_5_moe")
        #expect(Fixture.textModelType == "qwen3_5_moe_text")
        #expect(Fixture.architecture == "Qwen3_5MoeForConditionalGeneration")
        #expect(Fixture.numHiddenLayers == 40)
        #expect(Fixture.layerTypes.count == 40)
        #expect(Fixture.fullAttentionInterval == 4)
        #expect(Fixture.mtpNumHiddenLayers == 1)
        #expect(Fixture.tieWordEmbeddings == false)
        #expect(Fixture.visionDepth == 27)
    }

    // MARK: - Acceptance

    @Test func pinnedIdentityValidatesFromMetadataAlone() throws {
        // No file is opened: the pinned values alone must satisfy the contract.
        try QwenOfficialIdentity.validate(Self.metadata())
        // Control on the rejection helper: unmutated pinned metadata reports no error, so
        // the negative controls below are not vacuously satisfied.
        #expect(Self.rejection(Self.metadata()) == nil)
    }

    // MARK: - Repository and revision

    @Test(arguments: [
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B ",
        " Qwen/Qwen3.6-35B-A3B",
        "qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35b-a3b",
        "mlx-community/Qwen3.6-35B-A3B",
        "Qwen/Qwen3-30B-A3B",
        "",
    ])
    func repositoryMutationsAreRejected(repository: String) {
        #expect(Self.rejection(Self.metadata(repository: repository)) == .invalidRepository,
                "accepted repository \(repository)")
    }

    @Test(arguments: [
        "995ad96eacd98c81ed38be0c5b274b04031597b",     // 39 characters
        "995ad96eacd98c81ed38be0c5b274b04031597b00",  // 41 characters
        "995ad96eacd98c81ed38be0c5b274b04031597b1",   // one character changed
        "095ad96eacd98c81ed38be0c5b274b04031597b0",
        "995ad96eacd98c81ed38be0c5b274b04031597b0".uppercased(),
        "995ad96eacd98c81ed38be0c5b274b04031597bg",   // not hexadecimal
        "995ad96eacd98c81ed38be0c5b274b04031597b0\n",
        "main",
        "",
    ])
    func revisionMutationsAreRejected(revision: String) {
        #expect(Self.rejection(Self.metadata(revision: revision)) == .invalidRevision,
                "accepted revision \(revision)")
    }

    // MARK: - Sidecar digests

    @Test(arguments: [
        "config.json", "configuration.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "preprocessor_config.json", "model.safetensors.index.json",
    ])
    func oneByteFlipInAnyPinnedDigestIsRejected(name: String) throws {
        var digests = Fixture.sidecarSHA256
        let original = try #require(digests[name])
        digests[name] = String(original.dropLast()) + (original.hasSuffix("0") ? "1" : "0")
        #expect(digests[name] != original)
        #expect(Self.rejection(Self.metadata(sidecarSHA256: digests)) == .invalidSidecarDigests,
                "accepted a changed \(name) digest")
    }

    @Test func missingExtraAndMalformedDigestsAreRejected() throws {
        var missing = Fixture.sidecarSHA256
        missing.removeValue(forKey: "tokenizer.json")
        #expect(Self.rejection(Self.metadata(sidecarSHA256: missing)) == .invalidSidecarDigests)

        var extra = Fixture.sidecarSHA256
        extra["chat_template.jinja"] = String(repeating: "a", count: 64)
        #expect(Self.rejection(Self.metadata(sidecarSHA256: extra)) == .invalidSidecarDigests)

        var empty = Fixture.sidecarSHA256
        empty.removeAll()
        #expect(Self.rejection(Self.metadata(sidecarSHA256: empty)) == .invalidSidecarDigests)

        let short = String(repeating: "a", count: 63)
        let long = String(repeating: "a", count: 65)
        let nonHex = String(repeating: "z", count: 64)
        for replacement in [short, long, nonHex, ""] {
            var digests = Fixture.sidecarSHA256
            digests["config.json"] = replacement
            #expect(Self.rejection(Self.metadata(sidecarSHA256: digests))
                == .invalidSidecarDigests, "accepted digest \(replacement.count) characters")
        }
    }

    // MARK: - Configuration

    @Test func configurationSizeMutationsAreRejected() {
        let cases: [(String, QwenOfficialConfiguration)] = [
            ("numHiddenLayers", Self.validConfiguration(numHiddenLayers: 39)),
            ("numHiddenLayers too large", Self.validConfiguration(numHiddenLayers: 41)),
            ("hiddenSize", Self.validConfiguration(hiddenSize: 2_047)),
            ("numAttentionHeads", Self.validConfiguration(numAttentionHeads: 32)),
            ("numKeyValueHeads", Self.validConfiguration(numKeyValueHeads: 4)),
            ("headDim", Self.validConfiguration(headDim: 128)),
            ("numExperts", Self.validConfiguration(numExperts: 128)),
            ("numExpertsPerToken", Self.validConfiguration(numExpertsPerToken: 16)),
            ("vocabSize", Self.validConfiguration(vocabSize: 248_319)),
            ("fullAttentionInterval", Self.validConfiguration(fullAttentionInterval: 2)),
            ("tieWordEmbeddings", Self.validConfiguration(tieWordEmbeddings: true)),
        ]
        for (label, configuration) in cases {
            #expect(Self.rejection(Self.metadata(configuration: configuration))
                == .invalidConfiguration, "accepted \(label) mutation")
        }
    }

    @Test func architectureAndMtpMutationsAreRejected() {
        let architectureCases: [(String, [String])] = [
            ("Qwen3_5MoeForCausalLM", ["Qwen3_5MoeForCausalLM"]),
            ("empty architectures", []),
            ("two architectures", ["Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"]),
        ]
        for (label, architectures) in architectureCases {
            #expect(Self.rejection(Self.metadata(
                configuration: Self.validConfiguration(architectures: architectures)))
                == .invalidConfiguration, "accepted \(label)")
        }

        for modelType in ["qwen3_moe", "qwen3_5", "qwen3.5", "Qwen3_5Moe", ""] {
            #expect(Self.rejection(Self.metadata(
                configuration: Self.validConfiguration(modelType: modelType)))
                == .invalidConfiguration, "accepted model_type \(modelType)")
        }
        for textModelType in ["qwen3_5_moe", "qwen3_moe_text", ""] {
            #expect(Self.rejection(Self.metadata(
                configuration: Self.validConfiguration(textModelType: textModelType)))
                == .invalidConfiguration, "accepted text model_type \(textModelType)")
        }

        var swapped = Fixture.layerTypes
        swapped.swapAt(3, 4)
        #expect(Self.rejection(Self.metadata(
            configuration: Self.validConfiguration(layerTypes: swapped)))
            == .invalidConfiguration, "accepted a moved full_attention layer")

        var renamed = Fixture.layerTypes
        renamed[3] = "Full_Attention"
        #expect(Self.rejection(Self.metadata(
            configuration: Self.validConfiguration(layerTypes: renamed)))
            == .invalidConfiguration, "accepted an unknown layer type string")

        #expect(Self.rejection(Self.metadata(
            configuration: Self.validConfiguration(layerTypes: Array(Fixture.layerTypes.dropLast()))))
            == .invalidConfiguration, "accepted a short layer_types array")
        #expect(Self.rejection(Self.metadata(
            configuration: Self.validConfiguration(
                layerTypes: Fixture.layerTypes + ["linear_attention"])))
            == .invalidConfiguration, "accepted a long layer_types array")

        #expect(Self.rejection(Self.metadata(
            configuration: Self.validConfiguration(mtpNumHiddenLayers: 2)))
            == .invalidConfiguration, "accepted mtp_num_hidden_layers 2")
        #expect(Self.rejection(Self.metadata(
            configuration: Self.validConfiguration(mtpNumHiddenLayers: 0)))
            == .invalidConfiguration, "accepted mtp_num_hidden_layers 0")
    }

    // MARK: - Bindings

    @Test func tokenizerBindingMutationsAreRejected() {
        let cases: [(String, QwenOfficialTokenizerBinding)] = [
            ("vocabSize", Self.validTokenizerBinding(vocabSize: 248_319)),
            ("imageTokenID", Self.validTokenizerBinding(imageTokenID: 248_057)),
            ("imageTokenID off by one", Self.validTokenizerBinding(imageTokenID: 248_055)),
            ("tokenizerClass", Self.validTokenizerBinding(tokenizerClass: "Qwen2TokenizerFast")),
            ("empty tokenizerClass", Self.validTokenizerBinding(tokenizerClass: "")),
        ]
        for (label, binding) in cases {
            #expect(Self.rejection(Self.metadata(tokenizerBinding: binding))
                == .invalidTokenizerBinding, "accepted \(label) mutation")
        }
    }

    @Test func processorBindingMutationsAreRejected() {
        let cases: [(String, QwenOfficialProcessorBinding)] = [
            ("processorClass", Self.validProcessorBinding(processorClass: "Qwen2VLProcessor")),
            ("processorClass empty", Self.validProcessorBinding(processorClass: "")),
            ("imageProcessorType", Self.validProcessorBinding(
                imageProcessorType: "Qwen2VLImageProcessor")),
            ("patchSize", Self.validProcessorBinding(patchSize: 14)),
            ("mergeSize", Self.validProcessorBinding(mergeSize: 4)),
        ]
        for (label, binding) in cases {
            #expect(Self.rejection(Self.metadata(processorBinding: binding))
                == .invalidProcessorBinding, "accepted \(label) mutation")
        }
    }

    @Test func twoFieldsChangedAtOnceIsStillRejected() {
        var digests = Fixture.sidecarSHA256
        digests["tokenizer.json"] = String(repeating: "0", count: 64)
        #expect(Self.rejection(Self.metadata(
            repository: "Qwen/Qwen3.5-35B-A3B", sidecarSHA256: digests))
            == .invalidRepository)
    }
}
