import Foundation
import Metal
import TurboFieldfareFormat

/// Metadata-only family classification. This boundary intentionally does not
/// inspect payload files, create a Metal object, map weights, or construct a
/// runner.
enum ModelFamilyAdmission: Sendable, Equatable {
    case gemmaV1
    case qwenV2(LoadedModelManifest)

    static func classify(
        directoryURL: URL,
        maxManifestBytes: UInt64 = ManifestReader.defaultMaxBytes
    ) throws -> ModelFamilyAdmission {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxManifestBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let version: (major: Int, minor: Int)
        do {
            version = try GTurboManifestHeaderCodec.version(in: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        switch version.major {
        case 1:
            _ = try ManifestReader.decode(data: data, expecting: .gemma4_26B_A4B)
            return .gemmaV1
        case 2:
            let manifest = try ManifestReader.decodeVerified(data: data)
            guard manifest.descriptor.family == .qwen3_6,
                  case .qwen3_6 = manifest.architecture else {
                throw ModelError.indexCorrupt(
                    detail: "v2 runtime admission currently accepts only verified Qwen 3.6")
            }
            return .qwenV2(manifest)
        default:
            throw ModelError.unsupportedVersion(major: version.major, minor: version.minor)
        }
    }
}

public enum LoadedRuntimeFamily: String, Sendable, Equatable {
    case gemma4
    case qwen3_6
}

public struct LoadedRuntimeQuantizationIdentity: Sendable, Equatable {
    public let category: String
    public let storage: String
    public let groupSize: Int?
    public let scaleType: String?
    public let biasType: String?
}

public struct LoadedRuntimeVerifiedVisionIdentity: Sendable, Equatable {
    public let sourceRevision: String
    public let processorConfigSHA256: String
    public let compatibleTextManifestSHA256: String
    public let visionPayloadSHA256: String
    public let supportsStillImages: Bool
    public let supportsVideo: Bool
}

public enum LoadedRuntimeVisionIdentity: Sendable, Equatable {
    case unavailable
    case verified(LoadedRuntimeVerifiedVisionIdentity)
}

/// Lossless plain-module mirror of the format validator's installed identity.
public struct LoadedRuntimeIdentity: Sendable, Equatable {
    public let family: LoadedRuntimeFamily
    public let modelID: String
    public let sourceRevision: String
    public let formatMajor: Int
    public let formatMinor: Int
    public let sourceIndexSHA256: String
    public let quantizationPolicySHA256: String
    public let textManifestSHA256: String
    public let quantization: [LoadedRuntimeQuantizationIdentity]
    public let vision: LoadedRuntimeVisionIdentity

    init(descriptor: InstalledModelDescriptor) {
        family = switch descriptor.family {
        case .gemma4: .gemma4
        case .qwen3_6: .qwen3_6
        }
        modelID = descriptor.modelID
        sourceRevision = descriptor.sourceRevision
        formatMajor = descriptor.formatMajor
        formatMinor = descriptor.formatMinor
        sourceIndexSHA256 = descriptor.sourceIndexSHA256
        quantizationPolicySHA256 = descriptor.quantizationPolicySHA256
        textManifestSHA256 = descriptor.textManifestSHA256
        quantization = descriptor.quantization.map {
            LoadedRuntimeQuantizationIdentity(
                category: $0.category, storage: $0.storage,
                groupSize: $0.groupSize, scaleType: $0.scaleType,
                biasType: $0.biasType)
        }
        vision = switch descriptor.vision {
        case .unavailable:
            .unavailable
        case .verified(let value):
            .verified(LoadedRuntimeVerifiedVisionIdentity(
                sourceRevision: value.sourceRevision,
                processorConfigSHA256: value.processorConfigSHA256,
                compatibleTextManifestSHA256: value.compatibleTextManifestSHA256,
                visionPayloadSHA256: value.visionPayloadSHA256,
                supportsStillImages: value.supportsStillImages,
                supportsVideo: value.supportsVideo))
        }
    }
}

/// One admitted payload plus the family-specific codec needed to use it.
public struct LoadedModelFamilyBundle: Sendable {
    public let runtime: ModelFamilyRuntime
    public let family: LoadedRuntimeFamily
    public let verifiedIdentity: LoadedRuntimeIdentity?
    public let qwenCodec: QwenChatCodec?

    init(runtime: ModelFamilyRuntime, family: LoadedRuntimeFamily,
         verifiedIdentity: LoadedRuntimeIdentity?, qwenCodec: QwenChatCodec?) {
        self.runtime = runtime
        self.family = family
        self.verifiedIdentity = verifiedIdentity
        self.qwenCodec = qwenCodec
    }
}

/// Family-selected model payload. Family variation is resolved once here and
/// is not re-checked inside the Qwen layer/token loop.
public enum ModelFamilyRuntime: @unchecked Sendable {
    case gemma(Model)
    case qwen(QwenTextModel)

    public static func load(
        directoryURL: URL,
        device: MTLDevice,
        streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
        expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
        integrityPolicy: ModelIntegrityPolicy? = nil
    ) throws -> ModelFamilyRuntime {
        switch try ModelFamilyAdmission.classify(directoryURL: directoryURL) {
        case .gemmaV1:
            return .gemma(try Model.load(
                directoryURL: directoryURL,
                device: device,
                expecting: .gemma4_26B_A4B,
                streamingMode: streamingMode,
                expertCachePolicy: expertCachePolicy,
                integrityPolicy: integrityPolicy))
        case .qwenV2(let manifest):
            return .qwen(try QwenTextModel.loadOfficial(
                directoryURL: directoryURL, manifest: manifest, device: device))
        }
    }
}

public extension ModelFamilyRuntime {
    /// Production service composition. Unlike `load`, Qwen readiness requires
    /// both the verified payload and its pinned tokenizer/chat codec.
    static func loadBundle(
        directoryURL: URL,
        device: MTLDevice,
        streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
        expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
        integrityPolicy: ModelIntegrityPolicy? = nil
    ) throws -> LoadedModelFamilyBundle {
        switch try ModelFamilyAdmission.classify(directoryURL: directoryURL) {
        case .gemmaV1:
            let runtime = try Model.load(
                directoryURL: directoryURL,
                device: device,
                expecting: .gemma4_26B_A4B,
                streamingMode: streamingMode,
                expertCachePolicy: expertCachePolicy,
                integrityPolicy: integrityPolicy)
            return LoadedModelFamilyBundle(
                runtime: .gemma(runtime), family: .gemma4,
                verifiedIdentity: nil, qwenCodec: nil)
        case .qwenV2(let manifest):
            let model = try QwenTextModel.loadOfficial(
                directoryURL: directoryURL, manifest: manifest, device: device)
            let codec = QwenChatCodec(
                tokenizer: try QwenTokenizer.load(from: directoryURL))
            return LoadedModelFamilyBundle(
                runtime: .qwen(model), family: .qwen3_6,
                verifiedIdentity: LoadedRuntimeIdentity(descriptor: manifest.descriptor),
                qwenCodec: codec)
        }
    }
}
