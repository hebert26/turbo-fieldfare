import Foundation
import Metal
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

/// Metadata-only family classification. This boundary intentionally does not
/// inspect payload files, create a Metal object, map weights, or construct a
/// runner.
enum ModelFamilyAdmission: Sendable, Equatable {
    case gemmaV1
    case qwenV2(LoadedModelManifest)
    case qwenOfficialSource(OfficialSourceDescriptor)

    /// Inspect only the top-level keys of the already capped metadata. The
    /// packed decoders tolerate unrelated legacy extensions, but source-only
    /// descriptor fields must never be silently accepted as packed metadata.
    private struct PackedTopLevelKeys: Decodable {
        private struct Key: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }

        let names: Set<String>

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: Key.self)
            names = Set(values.allKeys.map(\.stringValue))
        }
    }

    private static let sourceOnlyManifestKeys: Set<String> = [
        "kind", "version", "repository", "revision", "storageProfile",
        "sidecarSHA256", "shards", "sourceRoot", "contentSHA256",
    ]

    static func classify(
        directoryURL: URL,
        maxManifestBytes: UInt64 = ManifestReader.defaultMaxBytes
    ) throws -> ModelFamilyAdmission {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let names = try directory.basenames()
        let hasSource = names.contains(OfficialSourceDescriptor.markerFilename)
        let hasManifest = names.contains("manifest.json")
        guard !hasSource || !hasManifest else {
            throw ModelError.indexCorrupt(detail: "ambiguous model directory: source marker and packed manifest")
        }
        if hasSource {
            do {
                let data = try directory.readMetadata(
                    OfficialSourceDescriptor.markerFilename,
                    maxBytes: min(maxManifestBytes, OfficialSourceDescriptor.maximumMarkerBytes))
                let descriptor = try OfficialSourceDescriptor.decodeStrict(data: data)
                try OfficialSourceDescriptorValidation.validate(descriptor)
                return .qwenOfficialSource(descriptor)
            } catch {
                throw ModelError.indexCorrupt(detail: "\(OfficialSourceDescriptor.markerFilename): \(error)")
            }
        }
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxManifestBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let packedKeys: PackedTopLevelKeys
        do {
            packedKeys = try JSONDecoder().decode(PackedTopLevelKeys.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        if let sourceKey = packedKeys.names.intersection(sourceOnlyManifestKeys).sorted().first {
            throw ModelError.indexCorrupt(
                detail: "manifest.json contains source-only field \(sourceKey)")
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
    public let sourceIdentity: LoadedRuntimeSourceIdentity?
    public let qwenCodec: QwenChatCodec?

    init(runtime: ModelFamilyRuntime, family: LoadedRuntimeFamily,
         verifiedIdentity: LoadedRuntimeIdentity?, qwenCodec: QwenChatCodec?,
         sourceIdentity: LoadedRuntimeSourceIdentity? = nil) {
        self.runtime = runtime
        self.family = family
        self.verifiedIdentity = verifiedIdentity
        self.sourceIdentity = sourceIdentity
        self.qwenCodec = qwenCodec
    }
}

/// Family-selected model payload. Family variation is resolved once here and
/// is not re-checked inside the Qwen layer/token loop.
public enum ModelFamilyRuntime: @unchecked Sendable {
    case gemma(Model)
    case qwen(QwenTextModel)
    case qwenOfficialSource(QwenOfficialSourceModel)

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
        case .qwenOfficialSource:
            let context = try MetalContext()
            guard context.device.registryID == device.registryID else {
                throw ModelError.indexCorrupt(detail: "source device changed during load")
            }
            let expertSlots: Int
            switch streamingMode {
            case .pread(let slotCount): expertSlots = slotCount
            }
            return .qwenOfficialSource(try QwenOfficialSourceModel.load(
                registrationURL: directoryURL, context: context,
                integrityPolicy: integrityPolicy ?? .fullSha256,
                expertCacheSlots: expertSlots,
                expertCachePolicy: expertCachePolicy,
                residencyBudgetBytes: 12 * 1024 * 1024 * 1024))
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
        case .qwenOfficialSource:
            let context = try MetalContext()
            guard context.device.registryID == device.registryID else {
                throw ModelError.indexCorrupt(detail: "source device changed during load")
            }
            let expertSlots: Int
            switch streamingMode {
            case .pread(let slotCount): expertSlots = slotCount
            }
            let model = try QwenOfficialSourceModel.load(
                registrationURL: directoryURL, context: context,
                integrityPolicy: integrityPolicy ?? .fullSha256,
                expertCacheSlots: expertSlots,
                expertCachePolicy: expertCachePolicy,
                residencyBudgetBytes: 12 * 1024 * 1024 * 1024)
            guard let identity = model.sourceIdentity else {
                throw ModelError.sourceBackingUnsupported
            }
            try model.revalidateSource()
            // The existing pinned sidecar codec grants no source admission by
            // itself. The full/trusted policy and retained-handle checks above
            // and below are mandatory before this codec enters a bundle.
            let codec = QwenChatCodec(tokenizer: try QwenTokenizer.loadVerifiedOfficialSource(
                from: model.source))
            try model.revalidateSource()
            return LoadedModelFamilyBundle(
                runtime: .qwenOfficialSource(model), family: .qwen3_6,
                verifiedIdentity: nil, qwenCodec: codec,
                sourceIdentity: identity)
        }
    }
}
