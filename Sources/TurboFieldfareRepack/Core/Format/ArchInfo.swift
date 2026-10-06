import CoreFoundation
import Foundation
import TurboFieldfareFormat

/// Architecture facts mirrored into the installed manifest. Gemma keeps its
/// existing fields; Qwen additionally carries a strictly verified v2 profile.
struct ArchInfo: Sendable, Equatable {
    struct QwenVisionArchitecture: Sendable, Equatable {
        let depth: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numHeads: Int
        let numPositionEmbeddings: Int
        let inputChannels: Int
        let patchSize: Int
        let temporalPatchSize: Int
        let spatialMergeSize: Int
        let outputHiddenSize: Int
        let hiddenActivation: String
    }

    struct QwenPlanningArchitecture: Sendable, Equatable {
        let text: GTurboQwenArchitectureV2
        let vision: QwenVisionArchitecture
        let architectureName: String
        let modelType: String
        let textModelType: String
        let sourceDType: String
        let fullAttentionInterval: Int
        let mtpNumHiddenLayers: Int
        let mtpUsesDedicatedEmbeddings: Bool
        let maxPositionEmbeddings: Int
        let rmsNormEpsilon: Double
        let observedConfigSHA256: String

        fileprivate init(
            text: GTurboQwenArchitectureV2,
            vision: QwenVisionArchitecture,
            architectureName: String,
            modelType: String,
            textModelType: String,
            sourceDType: String,
            fullAttentionInterval: Int,
            mtpNumHiddenLayers: Int,
            mtpUsesDedicatedEmbeddings: Bool,
            maxPositionEmbeddings: Int,
            rmsNormEpsilon: Double,
            observedConfigSHA256: String
        ) {
            self.text = text
            self.vision = vision
            self.architectureName = architectureName
            self.modelType = modelType
            self.textModelType = textModelType
            self.sourceDType = sourceDType
            self.fullAttentionInterval = fullAttentionInterval
            self.mtpNumHiddenLayers = mtpNumHiddenLayers
            self.mtpUsesDedicatedEmbeddings = mtpUsesDedicatedEmbeddings
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.rmsNormEpsilon = rmsNormEpsilon
            self.observedConfigSHA256 = observedConfigSHA256
        }
    }

    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 otherwise. Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String
    let qwenPlanningArchitecture: QwenPlanningArchitecture?

    /// Preserves the existing Gemma construction surface used by fixtures.
    init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        qwenPlanningArchitecture = nil
    }

    private init(qwen: QwenPlanningArchitecture) {
        hiddenSize = qwen.text.hiddenSize
        intermediateSize = qwen.text.sharedExpertIntermediateSize
        moeIntermediateSize = qwen.text.routedExpertIntermediateSize
        numHeads = qwen.text.numAttentionHeads
        numKVHeads = qwen.text.numKeyValueHeads
        numFullKVHeads = qwen.text.numKeyValueHeads
        headDim = qwen.text.headDimension
        fullHeadDim = qwen.text.headDimension
        vocabSize = qwen.text.vocabularySize
        slidingWindow = qwen.maxPositionEmbeddings
        finalLogitSoftcap = 0
        ropeTheta = qwen.text.ropeTheta
        fullRopeTheta = qwen.text.ropeTheta
        partialRotaryFactor = qwen.text.partialRotaryFactor
        numLayers = qwen.text.numLayers
        numExperts = qwen.text.numberOfExperts
        topKExperts = qwen.text.expertsPerToken
        tieWordEmbeddings = qwen.text.tiedWordEmbeddings
        attentionKEqV = false
        fullAttentionLayerMask = qwen.text.layerTypes.map {
            $0 == .fullAttention ? 1 : 0
        }
        hiddenActivation = qwen.text.hiddenActivation
        qwenPlanningArchitecture = qwen
    }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        let root = try jsonObject(data, path: configPath)
        let architectures = root["architectures"] as? [String]
        if root["model_type"] as? String == "qwen3_5_moe"
            || architectures?.contains("Qwen3_5MoeForConditionalGeneration") == true {
            let digest = sha256(data)
            guard digest == GTurboFormatV2.qwenSidecarSHA256["config.json"] else {
                throw RepackError.sourceFingerprintRejected(path: configPath, sha256: digest)
            }
            return ArchInfo(qwen: try parseQwen(
                root: root,
                observedConfigSHA256: digest,
                path: configPath))
        }
        return try loadGemma(root: root, configPath: configPath)
    }

    /// Internal metadata-only seam. It parses the same exact Qwen contract but
    /// does not mint production provenance; only the planner's fixture result
    /// accepts its value.
    static func loadQwenMetadataFixture(configData: Data) throws -> QwenPlanningArchitecture {
        let path = "<qwen-metadata-fixture>"
        return try parseQwen(
            root: jsonObject(configData, path: path),
            observedConfigSHA256: sha256(configData),
            path: path)
    }

    private static func loadGemma(
        root: [String: Any],
        configPath: String
    ) throws -> ArchInfo {
        guard let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        func i(_ key: String) throws -> Int { try requiredInt(tc, key, configPath) }
        func d(_ key: String) throws -> Double { try requiredDouble(tc, key, configPath) }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = number(ropeFull["partial_rotary_factor"]) ?? 0.25
        let fullTheta = number(ropeFull["rope_theta"]) ?? 1_000_000.0
        let swaTheta = number(ropeSWA["rope_theta"]) ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let activation = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        return ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_global_key_value_heads"),
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: activation)
    }

    private static func parseQwen(
        root: [String: Any],
        observedConfigSHA256: String,
        path: String
    ) throws -> QwenPlanningArchitecture {
        guard let architectures = root["architectures"] as? [String],
              architectures == ["Qwen3_5MoeForConditionalGeneration"],
              root["model_type"] as? String == "qwen3_5_moe",
              let text = root["text_config"] as? [String: Any],
              let vision = root["vision_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: path, detail: "incomplete Qwen configuration")
        }
        let textModelType = try requiredString(text, "model_type", path)
        let sourceDType = try requiredString(text, "dtype", path)
        let layerNames = try requiredStrings(text, "layer_types", path)
        let layerTypes: [GTurboQwenLayerTypeV2] = try layerNames.map {
            switch $0 {
            case "linear_attention": return .linearAttention
            case "full_attention": return .fullAttention
            default:
                throw RepackError.configJsonInvalid(path: path, detail: "invalid Qwen layer type \($0)")
            }
        }
        guard let rope = text["rope_parameters"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: path, detail: "missing rope_parameters")
        }
        let stateDType = try requiredString(text, "mamba_ssm_dtype", path)
        guard stateDType == "float32" else {
            throw RepackError.configJsonInvalid(path: path, detail: "Qwen recurrent state must be float32")
        }
        let qwenText = GTurboQwenArchitectureV2(
            hiddenSize: try requiredInt(text, "hidden_size", path),
            numLayers: try requiredInt(text, "num_hidden_layers", path),
            layerTypes: layerTypes,
            numAttentionHeads: try requiredInt(text, "num_attention_heads", path),
            numKeyValueHeads: try requiredInt(text, "num_key_value_heads", path),
            headDimension: try requiredInt(text, "head_dim", path),
            attentionOutputGate: try requiredBool(text, "attn_output_gate", path),
            linearConvolutionKernel: try requiredInt(text, "linear_conv_kernel_dim", path),
            linearKeyHeads: try requiredInt(text, "linear_num_key_heads", path),
            linearKeyHeadDimension: try requiredInt(text, "linear_key_head_dim", path),
            linearValueHeads: try requiredInt(text, "linear_num_value_heads", path),
            linearValueHeadDimension: try requiredInt(text, "linear_value_head_dim", path),
            recurrentStateType: .fp32,
            partialRotaryFactor: try requiredDouble(text, "partial_rotary_factor", path),
            ropeTheta: try requiredDouble(rope, "rope_theta", path),
            mropeInterleaved: try requiredBool(rope, "mrope_interleaved", path),
            mropeSections: try requiredInts(rope, "mrope_section", path),
            numberOfExperts: try requiredInt(text, "num_experts", path),
            expertsPerToken: try requiredInt(text, "num_experts_per_tok", path),
            routedExpertIntermediateSize: try requiredInt(text, "moe_intermediate_size", path),
            sharedExpertIntermediateSize: try requiredInt(text, "shared_expert_intermediate_size", path),
            vocabularySize: try requiredInt(text, "vocab_size", path),
            tiedWordEmbeddings: try requiredBool(text, "tie_word_embeddings", path),
            hiddenActivation: try requiredString(text, "hidden_act", path),
            bosTokenID: try requiredInt(text, "bos_token_id", path),
            eosTokenID: try requiredInt(text, "eos_token_id", path),
            imageTokenID: try requiredInt(root, "image_token_id", path),
            videoTokenID: try requiredInt(root, "video_token_id", path),
            visionStartTokenID: try requiredInt(root, "vision_start_token_id", path),
            visionEndTokenID: try requiredInt(root, "vision_end_token_id", path))
        let qwenVision = QwenVisionArchitecture(
            depth: try requiredInt(vision, "depth", path),
            hiddenSize: try requiredInt(vision, "hidden_size", path),
            intermediateSize: try requiredInt(vision, "intermediate_size", path),
            numHeads: try requiredInt(vision, "num_heads", path),
            numPositionEmbeddings: try requiredInt(vision, "num_position_embeddings", path),
            inputChannels: try requiredInt(vision, "in_channels", path),
            patchSize: try requiredInt(vision, "patch_size", path),
            temporalPatchSize: try requiredInt(vision, "temporal_patch_size", path),
            spatialMergeSize: try requiredInt(vision, "spatial_merge_size", path),
            outputHiddenSize: try requiredInt(vision, "out_hidden_size", path),
            hiddenActivation: try requiredString(vision, "hidden_act", path))
        let result = QwenPlanningArchitecture(
            text: qwenText,
            vision: qwenVision,
            architectureName: architectures[0],
            modelType: "qwen3_5_moe",
            textModelType: textModelType,
            sourceDType: sourceDType,
            fullAttentionInterval: try requiredInt(text, "full_attention_interval", path),
            mtpNumHiddenLayers: try requiredInt(text, "mtp_num_hidden_layers", path),
            mtpUsesDedicatedEmbeddings: try requiredBool(text, "mtp_use_dedicated_embeddings", path),
            maxPositionEmbeddings: try requiredInt(text, "max_position_embeddings", path),
            rmsNormEpsilon: try requiredDouble(text, "rms_norm_eps", path),
            observedConfigSHA256: observedConfigSHA256)
        try validateQwen(result, topLevelTie: try requiredBool(root, "tie_word_embeddings", path), path: path)
        return result
    }

    private static func validateQwen(
        _ qwen: QwenPlanningArchitecture,
        topLevelTie: Bool,
        path: String
    ) throws {
        let expectedLayers = (0..<40).map {
            ($0 + 1).isMultiple(of: 4)
                ? GTurboQwenLayerTypeV2.fullAttention : .linearAttention
        }
        let t = qwen.text
        let v = qwen.vision
        guard qwen.architectureName == "Qwen3_5MoeForConditionalGeneration",
              qwen.modelType == "qwen3_5_moe",
              qwen.textModelType == "qwen3_5_moe_text",
              qwen.sourceDType == "bfloat16",
              t.hiddenSize == 2_048, t.numLayers == 40,
              t.layerTypes == expectedLayers,
              t.numAttentionHeads == 16, t.numKeyValueHeads == 2,
              t.headDimension == 256, t.attentionOutputGate,
              t.linearConvolutionKernel == 4,
              t.linearKeyHeads == 16, t.linearKeyHeadDimension == 128,
              t.linearValueHeads == 32, t.linearValueHeadDimension == 128,
              t.recurrentStateType == .fp32,
              t.partialRotaryFactor == 0.25, t.ropeTheta == 10_000_000,
              t.mropeInterleaved, t.mropeSections == [11, 11, 10],
              t.numberOfExperts == 256, t.expertsPerToken == 8,
              t.routedExpertIntermediateSize == 512,
              t.sharedExpertIntermediateSize == 512,
              t.vocabularySize == 248_320,
              !t.tiedWordEmbeddings, !topLevelTie,
              t.hiddenActivation == "silu",
              t.bosTokenID == 248_044, t.eosTokenID == 248_044,
              t.imageTokenID == 248_056, t.videoTokenID == 248_057,
              t.visionStartTokenID == 248_053, t.visionEndTokenID == 248_054,
              qwen.fullAttentionInterval == 4,
              qwen.mtpNumHiddenLayers == 1,
              !qwen.mtpUsesDedicatedEmbeddings,
              qwen.maxPositionEmbeddings == 262_144,
              qwen.rmsNormEpsilon == 0.000001,
              v == QwenVisionArchitecture(
                depth: 27, hiddenSize: 1_152, intermediateSize: 4_304,
                numHeads: 16, numPositionEmbeddings: 2_304,
                inputChannels: 3, patchSize: 16, temporalPatchSize: 2,
                spatialMergeSize: 2, outputHiddenSize: 2_048,
                hiddenActivation: "gelu_pytorch_tanh") else {
            throw RepackError.configJsonInvalid(path: path, detail: "unsupported Qwen architecture")
        }
    }

    private static func jsonObject(_ data: Data, path: String) throws -> [String: Any] {
        do {
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: path, detail: "not a JSON object")
            }
            return root
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.configJsonInvalid(path: path, detail: "\(error)")
        }
    }

    private static func requiredInt(
        _ object: [String: Any], _ key: String, _ path: String
    ) throws -> Int {
        guard let number = object[key] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite,
              number.doubleValue == number.doubleValue.rounded(),
              number.doubleValue >= Double(Int.min),
              number.doubleValue <= Double(Int.max) else {
            throw RepackError.configJsonInvalid(path: path, detail: "missing or invalid \(key)")
        }
        return number.intValue
    }

    private static func requiredDouble(
        _ object: [String: Any], _ key: String, _ path: String
    ) throws -> Double {
        guard let raw = object[key] as? NSNumber,
              CFGetTypeID(raw) != CFBooleanGetTypeID(),
              raw.doubleValue.isFinite else {
            throw RepackError.configJsonInvalid(path: path, detail: "missing or invalid \(key)")
        }
        return raw.doubleValue
    }

    private static func requiredBool(
        _ object: [String: Any], _ key: String, _ path: String
    ) throws -> Bool {
        guard let value = object[key] as? Bool else {
            throw RepackError.configJsonInvalid(path: path, detail: "missing or invalid \(key)")
        }
        return value
    }

    private static func requiredString(
        _ object: [String: Any], _ key: String, _ path: String
    ) throws -> String {
        guard let value = object[key] as? String, !value.isEmpty else {
            throw RepackError.configJsonInvalid(path: path, detail: "missing or invalid \(key)")
        }
        return value
    }

    private static func requiredStrings(
        _ object: [String: Any], _ key: String, _ path: String
    ) throws -> [String] {
        guard let value = object[key] as? [String], !value.isEmpty else {
            throw RepackError.configJsonInvalid(path: path, detail: "missing or invalid \(key)")
        }
        return value
    }

    private static func requiredInts(
        _ object: [String: Any], _ key: String, _ path: String
    ) throws -> [Int] {
        guard let values = object[key] as? [Any], !values.isEmpty else {
            throw RepackError.configJsonInvalid(path: path, detail: "missing or invalid \(key)")
        }
        return try values.enumerated().map { index, value in
            try requiredInt(["value": value], "value", "\(path):\(key)[\(index)]")
        }
    }

    private static func number(_ value: Any?) -> Double? {
        (value as? NSNumber)?.doubleValue
    }

    private static func sha256(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }
}
