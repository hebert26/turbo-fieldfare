import Foundation

public enum InstalledModelFamily: String, Codable, Equatable, Sendable {
    case gemma4
    case qwen3_6
}

public struct InstalledVisionDescriptor: Codable, Equatable, Sendable {
    public let sourceRevision: String
    public let processorConfigSHA256: String
    public let compatibleTextManifestSHA256: String
    public let visionPayloadSHA256: String
    public let supportsStillImages: Bool
    public let supportsVideo: Bool

    fileprivate init(
        sourceRevision: String, processorConfigSHA256: String,
        compatibleTextManifestSHA256: String, visionPayloadSHA256: String,
        supportsStillImages: Bool, supportsVideo: Bool
    ) {
        self.sourceRevision = sourceRevision
        self.processorConfigSHA256 = processorConfigSHA256
        self.compatibleTextManifestSHA256 = compatibleTextManifestSHA256
        self.visionPayloadSHA256 = visionPayloadSHA256
        self.supportsStillImages = supportsStillImages
        self.supportsVideo = supportsVideo
    }
}

public enum InstalledVisionStatus: Codable, Equatable, Sendable {
    case unavailable
    case verified(InstalledVisionDescriptor)
}

public struct InstalledQuantizationDescriptor: Codable, Equatable, Sendable {
    public let category: String
    public let storage: String
    public let groupSize: Int?
    public let scaleType: String?
    public let biasType: String?

    fileprivate init(group: GTurboQuantizationGroupV2) {
        category = group.category.rawValue
        storage = group.storage.rawValue
        groupSize = group.groupSize
        scaleType = group.scaleType
        biasType = group.biasType
    }
}

/// Identity emitted by the format validator after a manifest has passed all
/// structural and pinned-source checks. Product call sites can inspect and
/// serialize this value, but cannot construct it from a display label.
public struct InstalledModelDescriptor: Codable, Equatable, Sendable {
    public let family: InstalledModelFamily
    public let modelID: String
    public let sourceRevision: String
    public let formatMajor: Int
    public let formatMinor: Int
    public let sourceIndexSHA256: String
    public let quantizationPolicySHA256: String
    public let textManifestSHA256: String
    public let quantization: [InstalledQuantizationDescriptor]
    public let vision: InstalledVisionStatus

    private enum CodingKeys: String, CodingKey {
        case family
        case modelID
        case sourceRevision
        case formatMajor
        case formatMinor
        case sourceIndexSHA256
        case quantizationPolicySHA256
        case textManifestSHA256
        case quantization
        case vision
    }

    private init(
        family: InstalledModelFamily, modelID: String, sourceRevision: String,
        formatMajor: Int, formatMinor: Int, sourceIndexSHA256: String,
        quantizationPolicySHA256: String, textManifestSHA256: String,
        quantization: [InstalledQuantizationDescriptor],
        vision: InstalledVisionStatus
    ) throws {
        self.family = family
        self.modelID = modelID
        self.sourceRevision = sourceRevision
        self.formatMajor = formatMajor
        self.formatMinor = formatMinor
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.quantizationPolicySHA256 = quantizationPolicySHA256
        self.textManifestSHA256 = textManifestSHA256
        self.quantization = quantization
        self.vision = vision
        try validate()
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            family: values.decode(InstalledModelFamily.self, forKey: .family),
            modelID: values.decode(String.self, forKey: .modelID),
            sourceRevision: values.decode(String.self, forKey: .sourceRevision),
            formatMajor: values.decode(Int.self, forKey: .formatMajor),
            formatMinor: values.decode(Int.self, forKey: .formatMinor),
            sourceIndexSHA256: values.decode(String.self, forKey: .sourceIndexSHA256),
            quantizationPolicySHA256: values.decode(String.self, forKey: .quantizationPolicySHA256),
            textManifestSHA256: values.decode(String.self, forKey: .textManifestSHA256),
            quantization: values.decode(
                [InstalledQuantizationDescriptor].self, forKey: .quantization),
            vision: values.decode(InstalledVisionStatus.self, forKey: .vision))
    }

    public func encode(to encoder: Encoder) throws {
        try validate()
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(family, forKey: .family)
        try values.encode(modelID, forKey: .modelID)
        try values.encode(sourceRevision, forKey: .sourceRevision)
        try values.encode(formatMajor, forKey: .formatMajor)
        try values.encode(formatMinor, forKey: .formatMinor)
        try values.encode(sourceIndexSHA256, forKey: .sourceIndexSHA256)
        try values.encode(quantizationPolicySHA256, forKey: .quantizationPolicySHA256)
        try values.encode(textManifestSHA256, forKey: .textManifestSHA256)
        try values.encode(quantization, forKey: .quantization)
        try values.encode(vision, forKey: .vision)
    }

    private func validate() throws {
        guard formatMajor == GTurboFormatV2.versionMajor,
              formatMinor == GTurboFormatV2.versionMinor else {
            throw GTurboFormatError.invalid(field: "descriptor.format", reason: "unsupported version")
        }
        try gturboValidateSHA256(sourceIndexSHA256, field: "descriptor.sourceIndexSHA256")
        try gturboValidateSHA256(
            quantizationPolicySHA256, field: "descriptor.quantizationPolicySHA256")
        try gturboValidateSHA256(
            textManifestSHA256, field: "descriptor.textManifestSHA256")
        let gemmaCategories: Set<String> = [
            GTurboQuantizationCategoryV2.embedding.rawValue,
            GTurboQuantizationCategoryV2.attention.rawValue,
            GTurboQuantizationCategoryV2.router.rawValue,
            GTurboQuantizationCategoryV2.sharedExpert.rawValue,
            GTurboQuantizationCategoryV2.routedExpert.rawValue,
        ]
        let expectedCategories = family == .qwen3_6
            ? Set(GTurboQuantizationCategoryV2.allCases.map(\.rawValue))
            : gemmaCategories
        let categories = Set(quantization.map(\.category))
        guard quantization.count == expectedCategories.count,
              categories == expectedCategories else {
            throw GTurboFormatError.invalid(
                field: "descriptor.quantization", reason: "incomplete profile")
        }
        for group in quantization {
            guard let storage = GTurboStorageTypeV2(rawValue: group.storage) else {
                throw GTurboFormatError.invalid(
                    field: "descriptor.quantization", reason: "unknown storage")
            }
            switch storage {
            case .bf16, .fp32:
                guard group.groupSize == nil,
                      group.scaleType == nil, group.biasType == nil else {
                    throw GTurboFormatError.invalid(
                        field: "descriptor.quantization", reason: "invalid unquantized group")
                }
            case .affineInt4, .affineInt8:
                guard group.groupSize == 64,
                      group.scaleType?.lowercased() == "bf16",
                      group.biasType?.lowercased() == "bf16" else {
                    throw GTurboFormatError.invalid(
                        field: "descriptor.quantization", reason: "invalid affine group")
                }
            }
        }
        if family == .qwen3_6 {
            guard quantization.first(where: {
                $0.category == GTurboQuantizationCategoryV2.recurrentState.rawValue
            })?.storage == GTurboStorageTypeV2.fp32.rawValue else {
                throw GTurboFormatError.invalid(
                    field: "descriptor.quantization", reason: "recurrent state is not FP32")
            }
            guard modelID == GTurboFormatV2.qwenRepository,
                  sourceRevision == GTurboFormatV2.qwenRevision,
                  sourceIndexSHA256 == GTurboFormatV2.qwenSourceIndexSHA256 else {
                throw GTurboFormatError.invalid(field: "descriptor.identity", reason: "not pinned Qwen")
            }
        } else {
            guard !modelID.isEmpty, !sourceRevision.isEmpty else {
                throw GTurboFormatError.invalid(field: "descriptor.identity", reason: "missing identity")
            }
        }
        if case let .verified(companion) = vision {
            guard family == .qwen3_6,
                  companion.sourceRevision == sourceRevision,
                  companion.processorConfigSHA256
                    == GTurboFormatV2.qwenSidecarSHA256["preprocessor_config.json"],
                  companion.compatibleTextManifestSHA256 == textManifestSHA256,
                  companion.supportsStillImages,
                  !companion.supportsVideo else {
                throw GTurboFormatError.invalid(field: "descriptor.vision", reason: "unbound companion")
            }
            try gturboValidateSHA256(
                companion.processorConfigSHA256, field: "descriptor.vision.processorConfigSHA256")
            try gturboValidateSHA256(
                companion.compatibleTextManifestSHA256,
                field: "descriptor.vision.compatibleTextManifestSHA256")
            try gturboValidateSHA256(
                companion.visionPayloadSHA256, field: "descriptor.vision.visionPayloadSHA256")
        }
    }

    static func validated(
        textManifest: GTurboManifestV2,
        textManifestSHA256: String,
        visionManifest: GTurboVisionManifestV2? = nil
    ) throws -> InstalledModelDescriptor {
        let family: InstalledModelFamily = switch textManifest.family {
        case .gemma4: .gemma4
        case .qwen3_6: .qwen3_6
        }
        let vision: InstalledVisionStatus
        if let visionManifest {
            vision = .verified(InstalledVisionDescriptor(
                sourceRevision: visionManifest.sourceRevision,
                processorConfigSHA256: visionManifest.processorConfigSHA256,
                compatibleTextManifestSHA256: visionManifest.compatibleTextManifestSHA256,
                visionPayloadSHA256: visionManifest.visionPayloadSHA256,
                supportsStillImages: visionManifest.supportsStillImages,
                supportsVideo: visionManifest.supportsVideo))
        } else {
            vision = .unavailable
        }
        return try InstalledModelDescriptor(
            family: family,
            modelID: textManifest.modelID,
            sourceRevision: textManifest.provenance.sourceRevision,
            formatMajor: textManifest.versionMajor,
            formatMinor: textManifest.versionMinor,
            sourceIndexSHA256: textManifest.provenance.sourceIndexSHA256,
            quantizationPolicySHA256: textManifest.provenance.quantizationPolicySHA256,
            textManifestSHA256: textManifestSHA256,
            quantization: textManifest.quantization.map(
                InstalledQuantizationDescriptor.init(group:)),
            vision: vision)
    }
}

package extension GTurboVerifiedManifestV2 {
    func descriptor(binding companion: GTurboVerifiedVisionManifestV2) throws
        -> InstalledModelDescriptor {
        guard manifest.family == companion.manifest.family,
              manifest.provenance.sourceRevision == companion.manifest.sourceRevision,
              manifestSHA256 == companion.manifest.compatibleTextManifestSHA256 else {
            throw GTurboFormatError.invalid(field: "descriptor.vision", reason: "text/vision identity mismatch")
        }
        return try InstalledModelDescriptor.validated(
            textManifest: manifest, textManifestSHA256: manifestSHA256,
            visionManifest: companion.manifest)
    }
}
