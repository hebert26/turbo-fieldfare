import Foundation

package enum GTurboVisionFormatV2 {
    package static let magic = GTurboVisionFormatV1.magic
    package static let artifactKind = "qwen3_6_vision_companion"
    package static let versionMajor = 2
    package static let versionMinor = 0
    package static let metadataMaxBytes = GTurboVisionFormatV1.metadataMaxBytes
    package static let alignmentBytes = GTurboFormatV2.alignmentBytes
    package static let weightsFile = "vision_weights.bin"
    package static let processorFile = "preprocessor_config.json"
}

package struct GTurboQwenVisionProcessorProfileV2: Codable, Equatable, Sendable {
    package let processorClass: String
    package let imageProcessorType: String
    package let patchSize: Int
    package let temporalPatchSize: Int
    package let spatialMergeSize: Int

    package init(processorClass: String, imageProcessorType: String,
                 patchSize: Int, temporalPatchSize: Int,
                 spatialMergeSize: Int) {
        self.processorClass = processorClass
        self.imageProcessorType = imageProcessorType
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.spatialMergeSize = spatialMergeSize
    }
}

package struct GTurboVisionManifestV2: Codable, Equatable, Sendable {
    package let magic: String
    package let artifactKind: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let family: GTurboModelFamilyV2
    package let modelID: String
    package let sourceRevision: String
    package let processorProfile: GTurboQwenVisionProcessorProfileV2
    package let processorConfigSHA256: String
    package let compatibleTextManifestSHA256: String
    package let visionPayloadSHA256: String
    package let supportsStillImages: Bool
    package let supportsVideo: Bool
    package let files: [String: GTurboManifestFileV1]
    package let tensorRegions: [GTurboTensorRegionV2]

    package init(
        magic: String = GTurboVisionFormatV2.magic,
        artifactKind: String = GTurboVisionFormatV2.artifactKind,
        versionMajor: Int = GTurboVisionFormatV2.versionMajor,
        versionMinor: Int = GTurboVisionFormatV2.versionMinor,
        family: GTurboModelFamilyV2, modelID: String,
        sourceRevision: String,
        processorProfile: GTurboQwenVisionProcessorProfileV2,
        processorConfigSHA256: String,
        compatibleTextManifestSHA256: String,
        visionPayloadSHA256: String,
        supportsStillImages: Bool, supportsVideo: Bool,
        files: [String: GTurboManifestFileV1],
        tensorRegions: [GTurboTensorRegionV2]
    ) {
        self.magic = magic
        self.artifactKind = artifactKind
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.family = family
        self.modelID = modelID
        self.sourceRevision = sourceRevision
        self.processorProfile = processorProfile
        self.processorConfigSHA256 = processorConfigSHA256
        self.compatibleTextManifestSHA256 = compatibleTextManifestSHA256
        self.visionPayloadSHA256 = visionPayloadSHA256
        self.supportsStillImages = supportsStillImages
        self.supportsVideo = supportsVideo
        self.files = files
        self.tensorRegions = tensorRegions
    }
}

/// A companion manifest accepted by the format validator. The initializer is
/// module-private so clients cannot turn display metadata into verified state.
package struct GTurboVerifiedVisionManifestV2: Sendable, Equatable {
    package let manifest: GTurboVisionManifestV2

    fileprivate init(manifest: GTurboVisionManifestV2) {
        self.manifest = manifest
    }
}

package enum GTurboVisionManifestDocument: Sendable, Equatable {
    case v1(GTurboVisionManifestV1)
    case v2(GTurboVerifiedVisionManifestV2)
}

package enum GTurboVisionManifestDocumentCodec {
    private struct Header: Decodable {
        let magic: String
        let versionMajor: Int
        let versionMinor: Int
    }

    package static func decode(_ data: Data) throws -> GTurboVisionManifestDocument {
        let header: Header
        do { header = try JSONDecoder().decode(Header.self, from: data) }
        catch {
            throw GTurboFormatError.invalid(
                field: GTurboVisionFormatV2.processorFile, reason: "\(error)")
        }
        guard header.magic == GTurboVisionFormatV2.magic,
              header.versionMajor >= 0, header.versionMinor >= 0 else {
            throw GTurboFormatError.invalid(field: "vision.manifest.identity", reason: "invalid header")
        }
        switch header.versionMajor {
        case GTurboVisionFormatV1.versionMajor:
            return .v1(try GTurboVisionManifestCodec.decode(data))
        case GTurboVisionFormatV2.versionMajor:
            return .v2(try GTurboVisionManifestV2Codec.decode(data))
        default:
            throw GTurboFormatError.invalid(field: "vision.manifest.version", reason: "unsupported major version")
        }
    }
}

package enum GTurboVisionManifestV2Codec {
    package static func decode(_ data: Data) throws -> GTurboVerifiedVisionManifestV2 {
        guard UInt64(data.count) <= GTurboVisionFormatV2.metadataMaxBytes else {
            throw GTurboFormatError.invalid(field: "vision.manifest.json", reason: "metadata exceeds 4 MiB")
        }
        let manifest: GTurboVisionManifestV2
        do { manifest = try JSONDecoder().decode(GTurboVisionManifestV2.self, from: data) }
        catch { throw GTurboFormatError.invalid(field: "vision.manifest.json", reason: "\(error)") }
        try GTurboVisionManifestV2Validator.validate(manifest)
        return GTurboVerifiedVisionManifestV2(manifest: manifest)
    }

    package static func encode(_ manifest: GTurboVisionManifestV2) throws -> Data {
        try GTurboVisionManifestV2Validator.validate(manifest)
        do {
            let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
            let data = try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
            guard UInt64(data.count) <= GTurboVisionFormatV2.metadataMaxBytes else {
                throw GTurboFormatError.invalid(field: "vision.manifest.json", reason: "metadata exceeds 4 MiB")
            }
            return data
        } catch let error as GTurboFormatError {
            throw error
        } catch {
            throw GTurboFormatError.invalid(field: "vision.manifest.json", reason: "\(error)")
        }
    }
}

package enum GTurboVisionManifestV2Validator {
    package static func validate(_ manifest: GTurboVisionManifestV2) throws {
        guard manifest.magic == GTurboVisionFormatV2.magic,
              manifest.artifactKind == GTurboVisionFormatV2.artifactKind,
              manifest.versionMajor == GTurboVisionFormatV2.versionMajor,
              manifest.versionMinor == GTurboVisionFormatV2.versionMinor else {
            throw GTurboFormatError.invalid(field: "vision.manifest.identity", reason: "unsupported artifact or version")
        }
        guard manifest.family == .qwen3_6,
              manifest.modelID == GTurboFormatV2.qwenRepository,
              manifest.sourceRevision == GTurboFormatV2.qwenRevision else {
            throw GTurboFormatError.invalid(field: "vision.manifest.family", reason: "not pinned Qwen")
        }
        guard manifest.processorProfile == GTurboQwenVisionProcessorProfileV2(
            processorClass: "Qwen3VLProcessor",
            imageProcessorType: "Qwen2VLImageProcessorFast",
            patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2),
              manifest.processorConfigSHA256
                == GTurboFormatV2.qwenSidecarSHA256["preprocessor_config.json"],
              manifest.supportsStillImages,
              !manifest.supportsVideo else {
            throw GTurboFormatError.invalid(field: "vision.manifest.processor", reason: "unsupported processor or media")
        }
        try gturboValidateSHA256(
            manifest.compatibleTextManifestSHA256,
            field: "vision.compatibleTextManifestSHA256")
        try gturboValidateSHA256(
            manifest.visionPayloadSHA256, field: "vision.visionPayloadSHA256")

        let expectedFiles = Set([
            GTurboVisionFormatV2.weightsFile,
            GTurboVisionFormatV2.processorFile,
        ])
        guard Set(manifest.files.keys) == expectedFiles,
              manifest.files[GTurboVisionFormatV2.weightsFile]?.sha256
                == manifest.visionPayloadSHA256,
              manifest.files[GTurboVisionFormatV2.processorFile]?.sha256
                == manifest.processorConfigSHA256 else {
            throw GTurboFormatError.invalid(field: "vision.manifest.files", reason: "payload binding mismatch")
        }
        for (path, file) in manifest.files {
            try GTurboPathValidator.validateBasename(path, field: "vision.manifest.files.\(path)")
            try gturboValidateSHA256(file.sha256, field: "vision.manifest.files.\(path).sha256")
        }
        guard !manifest.tensorRegions.isEmpty,
              Set(manifest.tensorRegions.map(\.name)).count == manifest.tensorRegions.count else {
            throw GTurboFormatError.invalid(field: "vision.tensorRegions", reason: "empty or duplicate tensor")
        }
        guard let weights = manifest.files[GTurboVisionFormatV2.weightsFile] else {
            throw GTurboFormatError.invalid(
                field: "vision.manifest.files", reason: "missing vision payload")
        }
        let weightsSize = weights.size
        var ranges: [(UInt64, UInt64)] = []
        for region in manifest.tensorRegions {
            guard !region.name.isEmpty, !region.name.contains("\0"),
                  region.file == GTurboVisionFormatV2.weightsFile,
                  region.offset % GTurboVisionFormatV2.alignmentBytes == 0,
                  region.size > 0,
                  !region.shape.isEmpty, region.shape.allSatisfy({ $0 > 0 }) else {
                throw GTurboFormatError.invalid(field: "vision.tensorRegions", reason: "invalid tensor region")
            }
            switch region.storage {
            case .affineInt4, .affineInt8:
                guard region.quantizationCategory != nil else {
                    throw GTurboFormatError.invalid(field: "vision.tensorRegions.quantization", reason: "missing category")
                }
            case .bf16, .fp32:
                guard region.quantizationCategory == nil else {
                    throw GTurboFormatError.invalid(field: "vision.tensorRegions.quantization", reason: "unexpected category")
                }
            }
            let end = try gturboCheckedAdd(region.offset, region.size, field: "vision.tensorRegions.range")
            guard end <= weightsSize else {
                throw GTurboFormatError.invalid(field: "vision.tensorRegions.range", reason: "region exceeds payload")
            }
            ranges.append((region.offset, end))
        }
        var previousEnd: UInt64 = 0
        for range in ranges.sorted(by: { $0.0 < $1.0 }) {
            guard range.0 >= previousEnd else {
                throw GTurboFormatError.invalid(field: "vision.tensorRegions.range", reason: "overlapping regions")
            }
            previousEnd = range.1
        }
    }
}
