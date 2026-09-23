import CryptoKit
import Foundation

package enum GTurboFormatV2 {
    package static let magic = GTurboFormatV1.magic
    package static let versionMajor = 2
    package static let versionMinor = 0
    package static let alignmentBytes = GTurboFormatV1.alignmentBytes

    package static let qwenRepository = "Qwen/Qwen3.6-35B-A3B"
    package static let qwenRevision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    package static let qwenSourceIndexSHA256 =
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"

    package static let qwenSidecarSHA256: [String: String] = [
        "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "model.safetensors.index.json": qwenSourceIndexSHA256,
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    ]
}

package enum GTurboModelFamilyV2: String, Codable, CaseIterable, Sendable {
    case gemma4
    case qwen3_6
}

package enum GTurboRequiredFeatureV2: String, Codable, CaseIterable, Sendable {
    case familyDispatch
    case verifiedIdentity
    case qwenHybridAttention
    case qwenMTPExcluded
}

package enum GTurboQwenLayerTypeV2: String, Codable, Sendable {
    case linearAttention
    case fullAttention
}

package enum GTurboRecurrentStateTypeV2: String, Codable, Sendable {
    case fp32
}

package struct GTurboQwenArchitectureV2: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let numLayers: Int
    package let layerTypes: [GTurboQwenLayerTypeV2]
    package let numAttentionHeads: Int
    package let numKeyValueHeads: Int
    package let headDimension: Int
    package let attentionOutputGate: Bool
    package let linearConvolutionKernel: Int
    package let linearKeyHeads: Int
    package let linearKeyHeadDimension: Int
    package let linearValueHeads: Int
    package let linearValueHeadDimension: Int
    package let recurrentStateType: GTurboRecurrentStateTypeV2
    package let partialRotaryFactor: Double
    package let ropeTheta: Double
    package let mropeInterleaved: Bool
    package let mropeSections: [Int]
    package let numberOfExperts: Int
    package let expertsPerToken: Int
    package let routedExpertIntermediateSize: Int
    package let sharedExpertIntermediateSize: Int
    package let vocabularySize: Int
    package let tiedWordEmbeddings: Bool
    package let hiddenActivation: String
    package let bosTokenID: Int
    package let eosTokenID: Int
    package let imageTokenID: Int
    package let videoTokenID: Int
    package let visionStartTokenID: Int
    package let visionEndTokenID: Int

    package init(
        hiddenSize: Int, numLayers: Int, layerTypes: [GTurboQwenLayerTypeV2],
        numAttentionHeads: Int, numKeyValueHeads: Int, headDimension: Int,
        attentionOutputGate: Bool, linearConvolutionKernel: Int,
        linearKeyHeads: Int, linearKeyHeadDimension: Int,
        linearValueHeads: Int, linearValueHeadDimension: Int,
        recurrentStateType: GTurboRecurrentStateTypeV2,
        partialRotaryFactor: Double, ropeTheta: Double,
        mropeInterleaved: Bool, mropeSections: [Int],
        numberOfExperts: Int, expertsPerToken: Int,
        routedExpertIntermediateSize: Int, sharedExpertIntermediateSize: Int,
        vocabularySize: Int, tiedWordEmbeddings: Bool, hiddenActivation: String,
        bosTokenID: Int, eosTokenID: Int, imageTokenID: Int, videoTokenID: Int,
        visionStartTokenID: Int, visionEndTokenID: Int
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.layerTypes = layerTypes
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDimension = headDimension
        self.attentionOutputGate = attentionOutputGate
        self.linearConvolutionKernel = linearConvolutionKernel
        self.linearKeyHeads = linearKeyHeads
        self.linearKeyHeadDimension = linearKeyHeadDimension
        self.linearValueHeads = linearValueHeads
        self.linearValueHeadDimension = linearValueHeadDimension
        self.recurrentStateType = recurrentStateType
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.mropeInterleaved = mropeInterleaved
        self.mropeSections = mropeSections
        self.numberOfExperts = numberOfExperts
        self.expertsPerToken = expertsPerToken
        self.routedExpertIntermediateSize = routedExpertIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.vocabularySize = vocabularySize
        self.tiedWordEmbeddings = tiedWordEmbeddings
        self.hiddenActivation = hiddenActivation
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
        self.imageTokenID = imageTokenID
        self.videoTokenID = videoTokenID
        self.visionStartTokenID = visionStartTokenID
        self.visionEndTokenID = visionEndTokenID
    }
}

package enum GTurboArchitectureV2: Codable, Equatable, Sendable {
    case gemma4(GTurboManifestArchV1)
    case qwen3_6(GTurboQwenArchitectureV2)

    private enum CodingKeys: String, CodingKey { case family, configuration }

    package init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let family = try values.decode(GTurboModelFamilyV2.self, forKey: .family)
        switch family {
        case .gemma4:
            self = .gemma4(try values.decode(GTurboManifestArchV1.self, forKey: .configuration))
        case .qwen3_6:
            self = .qwen3_6(try values.decode(GTurboQwenArchitectureV2.self, forKey: .configuration))
        }
    }

    package func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .gemma4(configuration):
            try values.encode(GTurboModelFamilyV2.gemma4, forKey: .family)
            try values.encode(configuration, forKey: .configuration)
        case let .qwen3_6(configuration):
            try values.encode(GTurboModelFamilyV2.qwen3_6, forKey: .family)
            try values.encode(configuration, forKey: .configuration)
        }
    }

    package var family: GTurboModelFamilyV2 {
        switch self {
        case .gemma4: .gemma4
        case .qwen3_6: .qwen3_6
        }
    }
}

package struct GTurboConversionProvenanceV2: Codable, Equatable, Sendable {
    package let sourceRepository: String
    package let sourceRevision: String
    package let sourceIndexSHA256: String
    package let sidecarSHA256: [String: String]
    package let quantizationPolicySHA256: String

    package init(sourceRepository: String, sourceRevision: String,
                 sourceIndexSHA256: String, sidecarSHA256: [String: String],
                 quantizationPolicySHA256: String) {
        self.sourceRepository = sourceRepository
        self.sourceRevision = sourceRevision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.sidecarSHA256 = sidecarSHA256
        self.quantizationPolicySHA256 = quantizationPolicySHA256
    }
}

package enum GTurboQuantizationCategoryV2: String, Codable, CaseIterable, Sendable {
    case embedding
    case attention
    case linearAttention
    case router
    case sharedExpert
    case routedExpert
    case outputHead
    case normalization
    case recurrentState
}

package enum GTurboStorageTypeV2: String, Codable, Sendable {
    case bf16
    case fp32
    case affineInt4
    case affineInt8
}

package struct GTurboQuantizationGroupV2: Codable, Equatable, Sendable {
    package let category: GTurboQuantizationCategoryV2
    package let storage: GTurboStorageTypeV2
    package let groupSize: Int?
    package let scaleType: String?
    package let biasType: String?

    package init(category: GTurboQuantizationCategoryV2,
                 storage: GTurboStorageTypeV2, groupSize: Int? = nil,
                 scaleType: String? = nil, biasType: String? = nil) {
        self.category = category
        self.storage = storage
        self.groupSize = groupSize
        self.scaleType = scaleType
        self.biasType = biasType
    }
}

package enum GTurboIgnoredTensorReasonV2: String, Codable, Sendable {
    case unsupportedMTP
}

package struct GTurboIgnoredTensorV2: Codable, Equatable, Sendable {
    package let name: String
    package let reason: GTurboIgnoredTensorReasonV2

    package init(name: String, reason: GTurboIgnoredTensorReasonV2) {
        self.name = name
        self.reason = reason
    }
}

package struct GTurboTensorRegionV2: Codable, Equatable, Sendable {
    package let name: String
    package let file: String
    package let offset: UInt64
    package let size: UInt64
    package let shape: [UInt64]
    package let storage: GTurboStorageTypeV2
    package let quantizationCategory: GTurboQuantizationCategoryV2?

    package init(name: String, file: String, offset: UInt64, size: UInt64,
                 shape: [UInt64], storage: GTurboStorageTypeV2,
                 quantizationCategory: GTurboQuantizationCategoryV2? = nil) {
        self.name = name
        self.file = file
        self.offset = offset
        self.size = size
        self.shape = shape
        self.storage = storage
        self.quantizationCategory = quantizationCategory
    }
}

package struct GTurboManifestV2: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let family: GTurboModelFamilyV2
    package let requiredFeatures: [GTurboRequiredFeatureV2]
    package let modelID: String
    package let architecture: GTurboArchitectureV2
    package let provenance: GTurboConversionProvenanceV2
    package let quantization: [GTurboQuantizationGroupV2]
    package let ignoredTensors: [GTurboIgnoredTensorV2]
    package let files: [String: GTurboManifestFileV1]
    package let tensorRegions: [GTurboTensorRegionV2]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64

    package init(
        magic: String = GTurboFormatV2.magic,
        versionMajor: Int = GTurboFormatV2.versionMajor,
        versionMinor: Int = GTurboFormatV2.versionMinor,
        family: GTurboModelFamilyV2,
        requiredFeatures: [GTurboRequiredFeatureV2], modelID: String,
        architecture: GTurboArchitectureV2,
        provenance: GTurboConversionProvenanceV2,
        quantization: [GTurboQuantizationGroupV2],
        ignoredTensors: [GTurboIgnoredTensorV2],
        files: [String: GTurboManifestFileV1],
        tensorRegions: [GTurboTensorRegionV2], expertsPerLayer: Int,
        numLayers: Int, expertStride: UInt64
    ) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.family = family
        self.requiredFeatures = requiredFeatures
        self.modelID = modelID
        self.architecture = architecture
        self.provenance = provenance
        self.quantization = quantization
        self.ignoredTensors = ignoredTensors
        self.files = files
        self.tensorRegions = tensorRegions
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
    }
}

/// A v2 manifest that has passed the format-owned structural and identity checks.
/// There is deliberately no package-visible initializer.
package struct GTurboVerifiedManifestV2: Sendable, Equatable {
    package let manifest: GTurboManifestV2
    package let manifestSHA256: String
    package let descriptor: InstalledModelDescriptor

    fileprivate init(manifest: GTurboManifestV2, manifestSHA256: String,
                     descriptor: InstalledModelDescriptor) {
        self.manifest = manifest
        self.manifestSHA256 = manifestSHA256
        self.descriptor = descriptor
    }
}

package enum GTurboManifestDocument: Sendable, Equatable {
    case v1(GTurboManifestV1)
    case v2(GTurboVerifiedManifestV2)
}

package enum GTurboManifestHeaderCodec {
    private struct Header: Decodable {
        let magic: String
        let versionMajor: Int
        let versionMinor: Int
    }

    package static func version(in data: Data) throws -> (major: Int, minor: Int) {
        let header: Header
        do { header = try JSONDecoder().decode(Header.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        guard header.magic == GTurboFormatV2.magic else {
            throw GTurboFormatError.invalid(field: "manifest.magic", reason: "expected GTURBO")
        }
        guard header.versionMajor >= 0, header.versionMinor >= 0 else {
            throw GTurboFormatError.invalid(field: "manifest.version", reason: "negative version")
        }
        return (header.versionMajor, header.versionMinor)
    }
}

package enum GTurboManifestDocumentCodec {
    package static func decode(_ data: Data) throws -> GTurboManifestDocument {
        switch try GTurboManifestHeaderCodec.version(in: data).major {
        case GTurboFormatV1.versionMajor:
            return .v1(try GTurboManifestCodec.decode(data))
        case GTurboFormatV2.versionMajor:
            return .v2(try GTurboManifestV2Codec.decode(data))
        default:
            throw GTurboFormatError.invalid(field: "manifest.version", reason: "unsupported major version")
        }
    }
}

package enum GTurboManifestV2Codec {
    package static func decode(_ data: Data) throws -> GTurboVerifiedManifestV2 {
        let manifest: GTurboManifestV2
        do { manifest = try JSONDecoder().decode(GTurboManifestV2.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        try GTurboManifestV2Validator.validate(manifest)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let descriptor = try InstalledModelDescriptor.validated(
            textManifest: manifest, textManifestSHA256: digest)
        return GTurboVerifiedManifestV2(
            manifest: manifest, manifestSHA256: digest, descriptor: descriptor)
    }

    package static func encode(_ manifest: GTurboManifestV2) throws -> Data {
        try GTurboManifestV2Validator.validate(manifest)
        do {
            let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch let error as GTurboFormatError {
            throw error
        } catch {
            throw GTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }
}

package enum GTurboManifestV2Validator {
    package static func validate(_ manifest: GTurboManifestV2) throws {
        guard manifest.magic == GTurboFormatV2.magic else {
            throw GTurboFormatError.invalid(field: "manifest.magic", reason: "expected GTURBO")
        }
        guard manifest.versionMajor == GTurboFormatV2.versionMajor,
              manifest.versionMinor == GTurboFormatV2.versionMinor else {
            throw GTurboFormatError.invalid(field: "manifest.version", reason: "unsupported v2 version")
        }
        guard manifest.family == manifest.architecture.family else {
            throw GTurboFormatError.invalid(field: "manifest.family", reason: "architecture discriminator disagrees")
        }
        let featureSet = Set(manifest.requiredFeatures)
        guard featureSet.count == manifest.requiredFeatures.count else {
            throw GTurboFormatError.invalid(field: "manifest.requiredFeatures", reason: "duplicate feature")
        }
        let commonFeatures: Set<GTurboRequiredFeatureV2> = [.familyDispatch, .verifiedIdentity]
        guard commonFeatures.isSubset(of: featureSet) else {
            throw GTurboFormatError.invalid(field: "manifest.requiredFeatures", reason: "missing required feature")
        }
        try validateProvenance(manifest.provenance, family: manifest.family)
        try validateQuantization(manifest.quantization, family: manifest.family)
        try validateFiles(
            manifest.files, tensorRegions: manifest.tensorRegions,
            quantization: manifest.quantization)
        guard manifest.expertsPerLayer > 0, manifest.numLayers > 0,
              manifest.expertStride > 0,
              manifest.expertStride % GTurboFormatV2.alignmentBytes == 0 else {
            throw GTurboFormatError.invalid(field: "manifest.streaming", reason: "invalid dimensions or stride")
        }

        switch manifest.architecture {
        case let .gemma4(architecture):
            try validateGemma(architecture)
            guard manifest.ignoredTensors.isEmpty,
                  manifest.numLayers == architecture.numLayers,
                  manifest.expertsPerLayer == architecture.numExperts,
                  !featureSet.contains(.qwenHybridAttention),
                  !featureSet.contains(.qwenMTPExcluded) else {
                throw GTurboFormatError.invalid(field: "manifest.gemma4", reason: "Qwen-only metadata")
            }
        case let .qwen3_6(architecture):
            guard featureSet.contains(.qwenHybridAttention),
                  featureSet.contains(.qwenMTPExcluded) else {
                throw GTurboFormatError.invalid(field: "manifest.requiredFeatures", reason: "missing Qwen feature")
            }
            try validateQwen(architecture)
            guard manifest.modelID == GTurboFormatV2.qwenRepository,
                  manifest.numLayers == architecture.numLayers,
                  manifest.expertsPerLayer == architecture.numberOfExperts else {
                throw GTurboFormatError.invalid(field: "manifest.qwen", reason: "identity or dimensions disagree")
            }
            try validateIgnoredTensors(manifest.ignoredTensors)
        }
    }

    private static func validateGemma(_ architecture: GTurboManifestArchV1) throws {
        guard architecture.hiddenSize > 0,
              architecture.ffnIntermediate > 0,
              architecture.moeIntermediateSize > 0,
              architecture.numHeads > 0,
              architecture.numKVHeads > 0,
              architecture.numFullKVHeads > 0,
              architecture.headDim > 0,
              architecture.fullHeadDim > 0,
              architecture.vocabSize > 0,
              architecture.slidingWindow > 0,
              architecture.finalLogitSoftcap.isFinite,
              architecture.ropeTheta.isFinite, architecture.ropeTheta > 0,
              architecture.fullRopeTheta.isFinite, architecture.fullRopeTheta > 0,
              architecture.partialRotaryFactor.isFinite,
              (0...1).contains(architecture.partialRotaryFactor),
              architecture.numLayers > 0,
              architecture.numExperts > 0,
              architecture.topKExperts > 0,
              architecture.topKExperts <= architecture.numExperts,
              !architecture.hiddenActivation.isEmpty,
              architecture.fullAttentionLayerMask.count == architecture.numLayers,
              architecture.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw GTurboFormatError.invalid(
                field: "manifest.architecture.gemma4", reason: "invalid architecture")
        }
    }

    private static func validateQwen(_ architecture: GTurboQwenArchitectureV2) throws {
        let expectedLayers = (0..<40).map { index in
            (index + 1).isMultiple(of: 4)
                ? GTurboQwenLayerTypeV2.fullAttention : .linearAttention
        }
        guard architecture.hiddenSize == 2_048,
              architecture.numLayers == 40,
              architecture.layerTypes == expectedLayers,
              architecture.numAttentionHeads == 16,
              architecture.numKeyValueHeads == 2,
              architecture.headDimension == 256,
              architecture.attentionOutputGate,
              architecture.linearConvolutionKernel == 4,
              architecture.linearKeyHeads == 16,
              architecture.linearKeyHeadDimension == 128,
              architecture.linearValueHeads == 32,
              architecture.linearValueHeadDimension == 128,
              architecture.recurrentStateType == .fp32,
              architecture.partialRotaryFactor == 0.25,
              architecture.ropeTheta == 10_000_000,
              architecture.mropeInterleaved,
              architecture.mropeSections == [11, 11, 10],
              architecture.numberOfExperts == 256,
              architecture.expertsPerToken == 8,
              architecture.routedExpertIntermediateSize == 512,
              architecture.sharedExpertIntermediateSize == 512,
              architecture.vocabularySize == 248_320,
              !architecture.tiedWordEmbeddings,
              architecture.hiddenActivation == "silu",
              architecture.bosTokenID == 248_044,
              architecture.eosTokenID == 248_044,
              architecture.imageTokenID == 248_056,
              architecture.videoTokenID == 248_057,
              architecture.visionStartTokenID == 248_053,
              architecture.visionEndTokenID == 248_054 else {
            throw GTurboFormatError.invalid(field: "manifest.architecture.qwen3_6", reason: "unsupported architecture")
        }
    }

    private static func validateProvenance(
        _ provenance: GTurboConversionProvenanceV2, family: GTurboModelFamilyV2
    ) throws {
        try gturboValidateSHA256(provenance.sourceIndexSHA256, field: "manifest.provenance.sourceIndexSHA256")
        try gturboValidateSHA256(provenance.quantizationPolicySHA256, field: "manifest.provenance.quantizationPolicySHA256")
        for (name, digest) in provenance.sidecarSHA256 {
            try GTurboPathValidator.validateBasename(name, field: "manifest.provenance.sidecarSHA256.\(name)")
            try gturboValidateSHA256(digest, field: "manifest.provenance.sidecarSHA256.\(name)")
        }
        if family == .qwen3_6 {
            guard provenance.sourceRepository == GTurboFormatV2.qwenRepository,
                  provenance.sourceRevision == GTurboFormatV2.qwenRevision,
                  provenance.sourceIndexSHA256 == GTurboFormatV2.qwenSourceIndexSHA256,
                  provenance.sidecarSHA256 == GTurboFormatV2.qwenSidecarSHA256 else {
                throw GTurboFormatError.invalid(field: "manifest.provenance", reason: "not the pinned official Qwen source")
            }
        } else {
            guard !provenance.sourceRepository.isEmpty,
                  !provenance.sourceRevision.isEmpty,
                  !provenance.sidecarSHA256.isEmpty else {
                throw GTurboFormatError.invalid(field: "manifest.provenance", reason: "missing source identity")
            }
        }
    }

    private static func validateQuantization(
        _ groups: [GTurboQuantizationGroupV2], family: GTurboModelFamilyV2
    ) throws {
        let gemmaCategories: Set<GTurboQuantizationCategoryV2> = [
            .embedding, .attention, .router, .sharedExpert, .routedExpert,
        ]
        let expectedCategories = family == .qwen3_6
            ? Set(GTurboQuantizationCategoryV2.allCases) : gemmaCategories
        guard groups.count == expectedCategories.count,
              Set(groups.map(\.category)) == expectedCategories else {
            throw GTurboFormatError.invalid(field: "manifest.quantization", reason: "categories must be exhaustive and unique")
        }
        for group in groups {
            switch group.storage {
            case .bf16, .fp32:
                guard group.groupSize == nil, group.scaleType == nil, group.biasType == nil else {
                    throw GTurboFormatError.invalid(field: "manifest.quantization.\(group.category.rawValue)", reason: "unquantized storage has affine metadata")
                }
            case .affineInt4, .affineInt8:
                guard group.groupSize == 64,
                      group.scaleType?.lowercased() == "bf16",
                      group.biasType?.lowercased() == "bf16" else {
                    throw GTurboFormatError.invalid(field: "manifest.quantization.\(group.category.rawValue)", reason: "unsupported affine group")
                }
            }
        }
        if family == .qwen3_6 {
            guard groups.first(where: { $0.category == .recurrentState })?.storage == .fp32 else {
                throw GTurboFormatError.invalid(field: "manifest.quantization.recurrentState", reason: "Qwen state must remain FP32")
            }
        }
    }

    private static func validateFiles(
        _ files: [String: GTurboManifestFileV1],
        tensorRegions: [GTurboTensorRegionV2],
        quantization: [GTurboQuantizationGroupV2]
    ) throws {
        guard files["model_weights.bin"] != nil,
              files["packed_experts/layout.json"] != nil else {
            throw GTurboFormatError.invalid(field: "manifest.files", reason: "missing required file")
        }
        var canonicalPaths = Set<String>()
        for (path, file) in files {
            try GTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = GTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.insert(key).inserted,
                  key != "manifest.json", key != "verified-install.json",
                  key != "tokenizer" else {
                throw GTurboFormatError.invalid(field: "manifest.files.\(path)", reason: "duplicate or reserved path")
            }
            try gturboValidateSHA256(file.sha256, field: "manifest.files.\(path).sha256")
        }
        for key in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                guard !canonicalPaths.contains(components.joined(separator: "/")) else {
                    throw GTurboFormatError.invalid(field: "manifest.files", reason: "file/directory prefix collision")
                }
            }
        }
        guard !tensorRegions.isEmpty,
              Set(tensorRegions.map(\.name)).count == tensorRegions.count else {
            throw GTurboFormatError.invalid(field: "manifest.tensorRegions", reason: "empty or duplicate tensor")
        }
        let quantizationByCategory = Dictionary(
            uniqueKeysWithValues: quantization.map { ($0.category, $0) })
        var rangesByFile: [String: [(UInt64, UInt64)]] = [:]
        for region in tensorRegions {
            guard !region.name.isEmpty, !region.name.contains("\0"),
                  let file = files[region.file], region.size > 0,
                  !region.shape.isEmpty, region.shape.allSatisfy({ $0 > 0 }),
                  region.offset % GTurboFormatV2.alignmentBytes == 0 else {
                throw GTurboFormatError.invalid(field: "manifest.tensorRegions", reason: "invalid tensor region")
            }
            guard let category = region.quantizationCategory,
                  quantizationByCategory[category]?.storage == region.storage else {
                throw GTurboFormatError.invalid(
                    field: "manifest.tensorRegions.quantizationCategory",
                    reason: "region storage disagrees with quantization profile")
            }
            let end = try gturboCheckedAdd(region.offset, region.size, field: "manifest.tensorRegions.range")
            guard end <= file.size else {
                throw GTurboFormatError.invalid(field: "manifest.tensorRegions.range", reason: "region exceeds file")
            }
            rangesByFile[region.file, default: []].append((region.offset, end))
        }
        for ranges in rangesByFile.values {
            let sorted = ranges.sorted { $0.0 < $1.0 }
            var previousEnd: UInt64 = 0
            for range in sorted {
                guard range.0 >= previousEnd else {
                    throw GTurboFormatError.invalid(field: "manifest.tensorRegions.range", reason: "overlapping regions")
                }
                previousEnd = range.1
            }
        }
    }

    private static func validateIgnoredTensors(_ tensors: [GTurboIgnoredTensorV2]) throws {
        guard tensors.allSatisfy({ $0.reason == .unsupportedMTP }),
              Set(tensors.map(\.name)).count == tensors.count,
              Set(tensors.map(\.name)) == qwenIgnoredTensorNames else {
            throw GTurboFormatError.invalid(field: "manifest.ignoredTensors", reason: "expected exact MTP omission set")
        }
    }

    private static let qwenIgnoredTensorNames: Set<String> = Set([
        "fc.weight", "layers.0.input_layernorm.weight", "layers.0.mlp.experts.down_proj",
        "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.gate.weight",
        "layers.0.mlp.shared_expert.down_proj.weight", "layers.0.mlp.shared_expert.gate_proj.weight",
        "layers.0.mlp.shared_expert.up_proj.weight", "layers.0.mlp.shared_expert_gate.weight",
        "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.v_proj.weight", "norm.weight", "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    ].map { "mtp." + $0 })
}

package func gturboValidateSHA256(_ value: String, field: String) throws {
    guard value.count == 64,
          value.unicodeScalars.allSatisfy({
              (48...57).contains($0.value) || (65...70).contains($0.value)
                  || (97...102).contains($0.value)
          }) else {
        throw GTurboFormatError.invalid(field: field, reason: "expected 64 hexadecimal characters")
    }
}
