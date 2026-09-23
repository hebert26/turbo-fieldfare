import Foundation

/// Metadata that must match the pinned official Qwen 3.6 source before it can
/// be accepted by a later repack planning phase. This type deliberately holds
/// metadata only; Phase 1 never opens a safetensors shard or tensor payload.
struct QwenOfficialSourceMetadata: Sendable, Equatable {
    let repository: String
    let revision: String
    let sidecarSHA256: [String: String]
    let configuration: QwenOfficialConfiguration
    let tokenizerBinding: QwenOfficialTokenizerBinding
    let processorBinding: QwenOfficialProcessorBinding
}

struct QwenOfficialConfiguration: Sendable, Equatable {
    let modelType: String
    let architectures: [String]
    let textModelType: String
    let numHiddenLayers: Int
    let hiddenSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let numExperts: Int
    let numExpertsPerToken: Int
    let vocabSize: Int
    let layerTypes: [String]
    let fullAttentionInterval: Int
    let mtpNumHiddenLayers: Int
    let tieWordEmbeddings: Bool
}

struct QwenOfficialTokenizerBinding: Sendable, Equatable {
    let vocabSize: Int
    let imageTokenID: Int
    let tokenizerClass: String
}

struct QwenOfficialProcessorBinding: Sendable, Equatable {
    let processorClass: String
    let imageProcessorType: String
    let patchSize: Int
    let mergeSize: Int
}

enum QwenOfficialValidationError: Error, Sendable, Equatable {
    case invalidRepository
    case invalidRevision
    case invalidSidecarDigests
    case invalidConfiguration
    case invalidTokenizerBinding
    case invalidProcessorBinding
    case invalidTensorSet
    case invalidTensor
}

/// Validates the immutable source identity for Qwen/Qwen3.6-35B-A3B.
enum QwenOfficialIdentity {
    static func validate(_ metadata: QwenOfficialSourceMetadata) throws {
        guard metadata.repository == "Qwen/Qwen3.6-35B-A3B" else {
            throw QwenOfficialValidationError.invalidRepository
        }
        guard metadata.revision == "995ad96eacd98c81ed38be0c5b274b04031597b0",
              metadata.revision.count == 40,
              metadata.revision.allSatisfy({ $0.isASCII && ($0.isNumber || ("a"..."f").contains($0)) }) else {
            throw QwenOfficialValidationError.invalidRevision
        }
        guard metadata.sidecarSHA256 == expectedSidecarSHA256 else {
            throw QwenOfficialValidationError.invalidSidecarDigests
        }

        let configuration = metadata.configuration
        guard configuration.modelType == "qwen3_5_moe",
              configuration.architectures == ["Qwen3_5MoeForConditionalGeneration"],
              configuration.textModelType == "qwen3_5_moe_text",
              configuration.numHiddenLayers == 40,
              configuration.hiddenSize == 2_048,
              configuration.numAttentionHeads == 16,
              configuration.numKeyValueHeads == 2,
              configuration.headDim == 256,
              configuration.numExperts == 256,
              configuration.numExpertsPerToken == 8,
              configuration.vocabSize == 248_320,
              configuration.layerTypes == expectedLayerTypes,
              configuration.fullAttentionInterval == 4,
              configuration.mtpNumHiddenLayers == 1,
              configuration.tieWordEmbeddings == false else {
            throw QwenOfficialValidationError.invalidConfiguration
        }

        guard metadata.tokenizerBinding == QwenOfficialTokenizerBinding(
            vocabSize: 248_320, imageTokenID: 248_056, tokenizerClass: "Qwen2Tokenizer"
        ) else {
            throw QwenOfficialValidationError.invalidTokenizerBinding
        }
        guard metadata.processorBinding == QwenOfficialProcessorBinding(
            processorClass: "Qwen3VLProcessor", imageProcessorType: "Qwen2VLImageProcessorFast",
            patchSize: 16, mergeSize: 2
        ) else {
            throw QwenOfficialValidationError.invalidProcessorBinding
        }
    }

    private static let expectedLayerTypes = (0..<40).map {
        $0.isMultiple(of: 4) == false && ($0 + 1).isMultiple(of: 4)
            ? "full_attention" : "linear_attention"
    }

    private static let expectedSidecarSHA256 = [
        "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "model.safetensors.index.json": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ]
}
