import Foundation
import TurboFieldfareOfficialQwenSource

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
        // Preserve the RepackCore metadata shape and error cases while sharing
        // its repository/revision/sidecar pins with the BF16 source contract.
        let pinned = OfficialQwenSourceIdentity.pinned
        let source = OfficialQwenSourceIdentity(
            repository: metadata.repository,
            revision: metadata.revision,
            storageProfile: pinned.storageProfile,
            sidecarSHA256: metadata.sidecarSHA256,
            shards: pinned.shards)
        do {
            try OfficialQwenIdentity.validate(source)
        } catch let error as OfficialQwenIdentityError {
            switch error {
            case .invalidRepository: throw QwenOfficialValidationError.invalidRepository
            case .invalidRevision: throw QwenOfficialValidationError.invalidRevision
            case .invalidSidecarDigests: throw QwenOfficialValidationError.invalidSidecarDigests
            // These fields are supplied by the pinned value above, never by a RepackCore caller.
            case .invalidStorageProfile, .invalidShards:
                throw QwenOfficialValidationError.invalidSidecarDigests
            }
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

}
