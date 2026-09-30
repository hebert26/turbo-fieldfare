import Foundation
import TurboFieldfare
import TurboFieldfareDecodeProtocol

public enum AppLoadedModelReadiness: Sendable, Equatable {
    case gemma(toolThinkingEnabled: Bool)
    case qwen(identity: DecodeModelIdentity)
    case qwenSource(identity: DecodeSourceIdentity)
}

extension DecodeModelIdentity {
    init(runtimeIdentity: LoadedRuntimeIdentity) {
        let family: DecodeModelFamily
        switch runtimeIdentity.family {
        case .gemma4: family = .gemma4
        case .qwen3_6: family = .qwen3_6
        }
        let vision: DecodeVisionIdentity
        switch runtimeIdentity.vision {
        case .unavailable:
            vision = .unavailable
        case .verified(let value):
            vision = .verified(DecodeVerifiedVisionIdentity(
                sourceRevision: value.sourceRevision,
                processorConfigSHA256: value.processorConfigSHA256,
                compatibleTextManifestSHA256: value.compatibleTextManifestSHA256,
                visionPayloadSHA256: value.visionPayloadSHA256,
                supportsStillImages: value.supportsStillImages,
                supportsVideo: value.supportsVideo))
        }
        self.init(
            family: family,
            modelID: runtimeIdentity.modelID,
            sourceRevision: runtimeIdentity.sourceRevision,
            formatMajor: runtimeIdentity.formatMajor,
            formatMinor: runtimeIdentity.formatMinor,
            sourceIndexSHA256: runtimeIdentity.sourceIndexSHA256,
            quantizationPolicySHA256: runtimeIdentity.quantizationPolicySHA256,
            textManifestSHA256: runtimeIdentity.textManifestSHA256,
            quantization: runtimeIdentity.quantization.map {
                DecodeQuantizationIdentity(
                    category: $0.category, storage: $0.storage,
                    groupSize: $0.groupSize, scaleType: $0.scaleType,
                    biasType: $0.biasType)
            },
            vision: vision)
    }
}

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    /// Joins the exact active load or bound teardown. Cancellation of one
    /// caller must not abandon shared cleanup, and return means the model owner
    /// has released its resources rather than merely closing local transport.
    func unload() async
    /// Loader-produced family readiness. The default is deliberately not ready.
    var loadedModelReadiness: AppLoadedModelReadiness? { get async }
    /// Releases this client's transport and any private service it launched.
    /// This must not wait for model teardown because app termination calls it
    /// from the main actor.
    func shutdownForTermination()
    /// Ends the current conversation and opens `epoch` as the only lineage the
    /// inference side will accept turns for. The model stays loaded.
    ///
    /// Required rather than defaulted: a client that silently did nothing here
    /// would keep appending a new chat's turns onto the previous chat's KV, and
    /// nothing downstream could detect it.
    func resetConversation(epoch: UUID) async throws
}

extension AppModelLifecycleClient {
    public var loadedModelReadiness: AppLoadedModelReadiness? { get async { nil } }
    public func shutdownForTermination() {}
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
    /// Resident bytes, which include the mapped weights the footprint omits.
    /// Defaulted so a reporter that cannot answer simply does not.
    var currentInferenceResidentBytes: UInt64? { get }
    /// Bytes of image tower the inference process holds mapped, or nil when it
    /// has no vision runtime. The only figure that separates the two image
    /// residency policies: both charge the process the same few MB.
    var currentInferenceTowerBytes: UInt64? { get }
    /// Logical bytes in the committed conversation state. This is distinct
    /// from process memory and allocated container capacity.
    var currentConversationLogicalStateBytes: UInt64? { get }
    /// Bytes of routed-expert cache buffers the loaded runtime currently owns.
    var currentExpertCacheBytes: UInt64? { get }
}

extension AppInferenceMemoryReporting {
    public var currentInferenceResidentBytes: UInt64? { nil }
    public var currentInferenceTowerBytes: UInt64? { nil }
    public var currentConversationLogicalStateBytes: UInt64? { nil }
    public var currentExpertCacheBytes: UInt64? { nil }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}
