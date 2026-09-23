import Foundation
import Metal

public enum ModelFamilyThinkingMode: Sendable, Equatable {
    case automatic
    case enabled
    case disabled
}

public enum ModelFamilyGenerationPrompt: Sendable, Equatable {
    /// Verbatim completion input. No chat template, tools, or media is applied.
    case raw(String)
    case chat(
        messages: [ModelChatMessage],
        tools: [ModelChatToolDefinition],
        thinking: ModelFamilyThinkingMode = .automatic)
}

public struct ModelFamilyGenerationRequest: Sendable {
    public var prompt: ModelFamilyGenerationPrompt
    /// Media IDs are the IDs carried by `ModelChatMedia`. Extra or missing IDs
    /// are rejected before any image is encoded.
    public var imagesByID: [String: URL]
    public var visionResidency: VisionResidencyPolicy
    public var config: GenerationConfig

    public init(
        prompt: ModelFamilyGenerationPrompt,
        imagesByID: [String: URL] = [:],
        visionResidency: VisionResidencyPolicy = .defaultPolicy,
        config: GenerationConfig
    ) {
        self.prompt = prompt
        self.imagesByID = imagesByID
        self.visionResidency = visionResidency
        self.config = config
    }
}

public enum ModelFamilyGenerationEvent: Sendable, Equatable {
    case prefill(done: Int, total: Int)
    case text(String)
    /// Published only after the complete model turn passes the family parser.
    case toolCall(ParsedToolCall)
}

public struct ModelFamilyGenerationResult: Sendable, Equatable {
    public let reason: StopReason
    public let promptTokens: Int
    public let newTokens: Int
    public let prefillSeconds: Double
    public let decodeSeconds: Double

    public init(
        reason: StopReason,
        promptTokens: Int,
        newTokens: Int,
        prefillSeconds: Double,
        decodeSeconds: Double
    ) {
        self.reason = reason
        self.promptTokens = promptTokens
        self.newTokens = newTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
    }
}

public enum ModelFamilyGenerationError: Error, Equatable, CustomStringConvertible {
    case emptyPrompt
    case unsupportedInput(String)
    case contextOverflow(prompt: Int, maxNew: Int, maximum: Int)
    case missingImage(String)
    case unexpectedImage(String)
    case duplicateImageID(String)
    case busy
    case modelIdentityChanged
    case verifiedVisionUnavailable
    case incompatibleVisionPack

    public var description: String {
        switch self {
        case .emptyPrompt: "prompt is empty"
        case .unsupportedInput(let detail): detail
        case .contextOverflow(let prompt, let maxNew, let maximum):
            "context overflow: prompt \(prompt) + maxNew \(maxNew) exceeds maxContext \(maximum)"
        case .missingImage(let id): "message image \(id) has no input file"
        case .unexpectedImage(let id): "image input \(id) is not referenced by the messages"
        case .duplicateImageID(let id): "message image ID \(id) is repeated"
        case .busy: "generation is already in progress"
        case .modelIdentityChanged:
            "model identity changed after the generation session loaded"
        case .verifiedVisionUnavailable:
            "this verified model installation does not support still images"
        case .incompatibleVisionPack:
            "the vision companion does not match the verified text model"
        }
    }
}

public struct ModelFamilyGenerationAdmission: Sendable, Equatable {
    public let family: LoadedRuntimeFamily
    public let verifiedIdentity: LoadedRuntimeIdentity?

    public init(
        family: LoadedRuntimeFamily,
        verifiedIdentity: LoadedRuntimeIdentity?
    ) {
        self.family = family
        self.verifiedIdentity = verifiedIdentity
    }
}

public struct ModelFamilyGenerationPreflight: Sendable, Equatable {
    public let promptTokens: Int
    public let imageCount: Int

    public init(promptTokens: Int, imageCount: Int) {
        self.promptTokens = promptTokens
        self.imageCount = imageCount
    }
}

public enum ModelFamilyVisionCompanionStatus: String, Sendable, Equatable {
    case ready
    case missing
    case invalid
    case unsupported
}

struct VerifiedQwenVisionCompanion {
    let textManifest: LoadedModelManifest
    let store: QwenVisionWeightStore
}

/// High-level family runtime used by products. It exposes family-neutral
/// requests and results while retaining Qwen runners, positions, and vision
/// allocations inside the TurboFieldfare module.
public actor ModelFamilyGenerationSession {
    public nonisolated let family: LoadedRuntimeFamily
    public nonisolated let verifiedIdentity: LoadedRuntimeIdentity?

    private let bundle: LoadedModelFamilyBundle
    private let context: MetalContext
    private let modelDirectoryURL: URL
    private let maxContext: Int
    private let runtimeConfiguration: RuntimeConfiguration
    private let visionPackURL: URL?
    private let fixturePromptTokenIDs: [Int32]?
    private let fixtureAfterBusyAcquired: (@Sendable () async -> Void)?
    private var generating = false

    /// Metadata-only family admission for product routing. This performs no
    /// Metal setup, weight mapping, model construction, or tokenizer load.
    public nonisolated static func inspect(
        directoryURL: URL
    ) throws -> ModelFamilyGenerationAdmission {
        switch try ModelFamilyAdmission.classify(directoryURL: directoryURL) {
        case .gemmaV1:
            return ModelFamilyGenerationAdmission(
                family: .gemma4, verifiedIdentity: nil)
        case .qwenV2(let manifest):
            return ModelFamilyGenerationAdmission(
                family: .qwen3_6,
                verifiedIdentity: LoadedRuntimeIdentity(
                    descriptor: manifest.descriptor))
        }
    }

    /// Opens the independently installed companion and binds it to the exact
    /// verified text artifact. A text-only identity may acquire a companion;
    /// an identity that already names one must name this exact companion.
    nonisolated static func openVerifiedQwenVisionCompanion(
        directoryURL: URL,
        loadedIdentity: LoadedRuntimeIdentity,
        visionPackURL: URL? = nil
    ) throws -> VerifiedQwenVisionCompanion {
        guard case .qwenV2(let textManifest) = try ModelFamilyAdmission.classify(
            directoryURL: directoryURL)
        else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        let currentIdentity = LoadedRuntimeIdentity(
            descriptor: textManifest.descriptor)
        guard qwenTextIdentityMatches(currentIdentity, loadedIdentity) else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        let location = try visionPackURL
            ?? VisionPackLocation.companionURL(forTextModel: directoryURL)
        let document = try VisionWeightStore.openCompanion(
            directoryURL: location,
            compatibleTextSourceSnapshotHash:
                loadedIdentity.sourceIndexSHA256,
            compatibleTextManifestSHA256:
                loadedIdentity.textManifestSHA256)
        guard case .qwen(let store) = document,
              store.manifest.modelID == loadedIdentity.modelID,
              store.manifest.sourceRevision == loadedIdentity.sourceRevision,
              store.manifest.supportsStillImages,
              !store.manifest.supportsVideo else {
            throw ModelFamilyGenerationError.incompatibleVisionPack
        }
        let actualVision = LoadedRuntimeVerifiedVisionIdentity(
            sourceRevision: store.manifest.sourceRevision,
            processorConfigSHA256: store.manifest.processorConfigSHA256,
            compatibleTextManifestSHA256:
                store.manifest.compatibleTextManifestSHA256,
            visionPayloadSHA256: store.manifest.visionPayloadSHA256,
            supportsStillImages: store.manifest.supportsStillImages,
            supportsVideo: store.manifest.supportsVideo)
        if case .verified(let expectedVision) = loadedIdentity.vision,
           expectedVision != actualVision {
            throw ModelFamilyGenerationError.incompatibleVisionPack
        }
        return VerifiedQwenVisionCompanion(
            textManifest: textManifest, store: store)
    }

    /// Verifies the Qwen companion against the identity of the already loaded
    /// text session without constructing the vision runtime or running GPU work.
    /// The caller can expose `ready` only after this exact check succeeds.
    public nonisolated static func inspectQwenVisionCompanion(
        directoryURL: URL,
        loadedIdentity: LoadedRuntimeIdentity,
        visionPackURL: URL? = nil
    ) throws -> ModelFamilyVisionCompanionStatus {
        guard case .qwenV2(let textManifest) = try ModelFamilyAdmission.classify(
            directoryURL: directoryURL),
              qwenTextIdentityMatches(
                LoadedRuntimeIdentity(descriptor: textManifest.descriptor),
                loadedIdentity) else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        let location = try visionPackURL
            ?? VisionPackLocation.companionURL(forTextModel: directoryURL)
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(
            atPath: location.standardizedFileURL.path,
            isDirectory: &isDirectory) && isDirectory.boolValue
        if visionPackURL != nil, !exists {
            throw VisionPackError.packNotFound(location.path)
        }
        guard exists else { return .missing }
        do {
            _ = try openVerifiedQwenVisionCompanion(
                directoryURL: directoryURL,
                loadedIdentity: loadedIdentity,
                visionPackURL: location)
            guard let device = MetalContext.makeSystemDefaultDevice(),
                  VisionRuntime.isSupported(on: device) else {
                return .unsupported
            }
            return .ready
        } catch ModelFamilyGenerationError.modelIdentityChanged {
            throw ModelFamilyGenerationError.modelIdentityChanged
        } catch {
            return .invalid
        }
    }

    /// Validates and sizes a Qwen request without constructing the text model.
    /// Tokenizer and image metadata are read only after verified admission.
    public nonisolated static func preflightQwen(
        directoryURL: URL,
        prompt: ModelFamilyGenerationPrompt,
        imagesByID: [String: URL],
        visionPackURL: URL? = nil,
        visionResidency: VisionResidencyPolicy = .defaultPolicy,
        maxContext: Int
    ) throws -> ModelFamilyGenerationPreflight {
        let admission = try ModelFamilyAdmission.classify(directoryURL: directoryURL)
        guard case .qwenV2(let manifest) = admission,
              case .qwen3_6(let architecture) = manifest.architecture else {
            throw ModelFamilyGenerationError.unsupportedInput(
                "Qwen preflight requires a verified Qwen installation")
        }
        let tokenizer = try QwenTokenizer.load(from: directoryURL)
        let codec = QwenChatCodec(tokenizer: tokenizer)
        let baseTokens: [Int32]
        let imageIDs: [String]
        switch prompt {
        case .raw(let text):
            guard imagesByID.isEmpty else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "raw completion does not accept images")
            }
            baseTokens = tokenizer.encode(text)
            imageIDs = []
        case .chat(let messages, let tools, let thinking):
            baseTokens = try codec.encodePrompt(
                messages: messages,
                tools: tools,
                options: .init(enableThinking: thinking != .disabled))
            imageIDs = try orderedImageIDs(in: messages)
        }
        guard !baseTokens.isEmpty else {
            throw ModelFamilyGenerationError.emptyPrompt
        }
        let expected = Set(imageIDs)
        if let missing = imageIDs.first(where: { imagesByID[$0] == nil }) {
            throw ModelFamilyGenerationError.missingImage(missing)
        }
        if let extra = imagesByID.keys.sorted().first(where: {
            !expected.contains($0)
        }) {
            throw ModelFamilyGenerationError.unexpectedImage(extra)
        }
        var promptTokens = baseTokens.count
        if !imageIDs.isEmpty {
            guard visionResidency == .onDemand else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "verified Qwen vision supports --vision-residency on-demand only")
            }
            let textIdentity = LoadedRuntimeIdentity(
                descriptor: manifest.descriptor)
            _ = try openVerifiedQwenVisionCompanion(
                directoryURL: directoryURL,
                loadedIdentity: textIdentity,
                visionPackURL: visionPackURL)
            guard let device = MetalContext.makeSystemDefaultDevice() else {
                throw ModelFamilyGenerationError.unsupportedInput("no Metal device")
            }
            try VisionRuntime.requireSupportedDevice(device)
            let preprocessor = QwenImagePreprocessor(device: device)
            let plans = try imageIDs.map {
                try preprocessor.plan(fileURL: imagesByID[$0]!)
            }
            try QwenImagePreprocessor.preflight(plans.map(\.geometry))
            let normalized = try normalizeQwenCodecImageFrames(
                baseTokens,
                architecture: architecture)
            // The codec's three-token frame is normalized to one placeholder;
            // the renderer then expands that placeholder to start + rows + end.
            promptTokens = try qwenExpandedPromptTokenCount(
                normalizedTokenIDs: normalized,
                imageMergedRows: plans.map(\.geometry.mergedRows),
                imageTokenID: Int32(architecture.imageTokenID))
        }
        guard promptTokens < maxContext else {
            throw ModelFamilyGenerationError.contextOverflow(
                prompt: promptTokens, maxNew: 0, maximum: maxContext)
        }
        return ModelFamilyGenerationPreflight(
            promptTokens: promptTokens, imageCount: imageIDs.count)
    }

    public static func load(
        directoryURL: URL,
        maxContext: Int,
        runtimeConfiguration: RuntimeConfiguration = .production,
        visionPackURL: URL? = nil
    ) throws -> ModelFamilyGenerationSession {
        let context = try MetalContext()
        let bundle = try ModelFamilyRuntime.loadBundle(
            directoryURL: directoryURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtimeConfiguration.expertCacheSlots),
            expertCachePolicy: runtimeConfiguration.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        return ModelFamilyGenerationSession(
            bundle: bundle,
            context: context,
            modelDirectoryURL: directoryURL,
            maxContext: maxContext,
            runtimeConfiguration: runtimeConfiguration,
            visionPackURL: visionPackURL,
            fixturePromptTokenIDs: nil,
            fixtureAfterBusyAcquired: nil)
    }

    /// Internal production-path fixture seam. Tests supply the existing tiny
    /// family model and a real codec, then run the same `generate` method.
    init(
        bundle: LoadedModelFamilyBundle,
        context: MetalContext,
        modelDirectoryURL: URL,
        maxContext: Int,
        runtimeConfiguration: RuntimeConfiguration,
        visionPackURL: URL? = nil,
        fixturePromptTokenIDs: [Int32]? = nil,
        fixtureAfterBusyAcquired: (@Sendable () async -> Void)? = nil
    ) {
        self.bundle = bundle
        self.context = context
        self.modelDirectoryURL = modelDirectoryURL
        self.maxContext = maxContext
        self.runtimeConfiguration = runtimeConfiguration
        self.visionPackURL = visionPackURL
        self.fixturePromptTokenIDs = fixturePromptTokenIDs
        self.fixtureAfterBusyAcquired = fixtureAfterBusyAcquired
        family = bundle.family
        verifiedIdentity = bundle.verifiedIdentity
    }

    public func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        if let fixtureAfterBusyAcquired { await fixtureAfterBusyAcquired() }
        try Task.checkCancellation()
        try request.config.validate()
        switch bundle.runtime {
        case .gemma(let model):
            return try await generateGemma(
                model: model, request: request, onEvent: onEvent)
        case .qwen(let model):
            guard let codec = bundle.qwenCodec,
                  (bundle.verifiedIdentity?.family == .qwen3_6
                    || fixturePromptTokenIDs != nil) else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "Qwen generation requires an admitted model and pinned codec")
            }
            return try await generateQwen(
                model: model, codec: codec, request: request, onEvent: onEvent)
        }
    }
}

private func orderedImageIDs(
    in messages: [ModelChatMessage]
) throws -> [String] {
    var result: [String] = []
    var seen: Set<String> = []
    for message in messages {
        guard case .parts(let parts)? = message.content else { continue }
        for part in parts {
            switch part {
            case .image(let media):
                guard let id = media.id, !id.isEmpty else {
                    throw ModelFamilyGenerationError.unsupportedInput(
                        "every chat image requires a non-empty media ID")
                }
                guard seen.insert(id).inserted else {
                    throw ModelFamilyGenerationError.duplicateImageID(id)
                }
                result.append(id)
            case .video:
                throw ModelFamilyGenerationError.unsupportedInput(
                    "video input is not supported")
            case .audio:
                throw ModelFamilyGenerationError.unsupportedInput(
                    "audio input is not supported")
            case .unsupported(let type):
                throw ModelFamilyGenerationError.unsupportedInput(
                    "unsupported media type: \(type)")
            case .text:
                break
            }
        }
    }
    return result
}

/// Compares only the text artifact. Vision is an independently installed
/// companion and is verified against the text manifest when it is opened.
private func qwenTextIdentityMatches(
    _ current: LoadedRuntimeIdentity,
    _ loaded: LoadedRuntimeIdentity
) -> Bool {
    current.family == .qwen3_6
        && loaded.family == .qwen3_6
        && current.modelID == loaded.modelID
        && current.sourceRevision == loaded.sourceRevision
        && current.formatMajor == loaded.formatMajor
        && current.formatMinor == loaded.formatMinor
        && current.sourceIndexSHA256 == loaded.sourceIndexSHA256
        && current.quantizationPolicySHA256
            == loaded.quantizationPolicySHA256
        && current.textManifestSHA256 == loaded.textManifestSHA256
        && current.quantization == loaded.quantization
}

/// Converts the pinned codec's
/// `vision_start, image_pad, vision_end` frame to the renderer's one-token
/// placeholder contract. Every vision marker must belong to one exact frame.
func normalizeQwenCodecImageFrames(
    _ tokenIDs: [Int32],
    architecture: QwenArchConfig
) throws -> [Int32] {
    let start = Int32(architecture.visionStartTokenID)
    let image = Int32(architecture.imageTokenID)
    let end = Int32(architecture.visionEndTokenID)
    var normalized: [Int32] = []
    normalized.reserveCapacity(tokenIDs.count)
    var index = 0
    while index < tokenIDs.count {
        let token = tokenIDs[index]
        if token == start {
            guard index + 2 < tokenIDs.count,
                  tokenIDs[index + 1] == image,
                  tokenIDs[index + 2] == end else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            normalized.append(image)
            index += 3
            continue
        }
        guard token != image, token != end else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        normalized.append(token)
        index += 1
    }
    return normalized
}

/// Sizes the exact normalized sequence consumed by the production renderer.
/// Replacing one image placeholder with start + rows + end adds `rows + 1`.
func qwenExpandedPromptTokenCount(
    normalizedTokenIDs: [Int32],
    imageMergedRows: [Int],
    imageTokenID: Int32
) throws -> Int {
    guard normalizedTokenIDs.filter({ $0 == imageTokenID }).count
            == imageMergedRows.count,
          imageMergedRows.allSatisfy({ $0 > 0 }) else {
        throw MultimodalPromptRendererError.placeholderMismatch
    }
    return try imageMergedRows.reduce(normalizedTokenIDs.count) { count, rows in
        let (withRows, rowsOverflow) = count.addingReportingOverflow(rows)
        let (withFrameEnd, endOverflow) = withRows.addingReportingOverflow(1)
        guard !rowsOverflow, !endOverflow else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        return withFrameEnd
    }
}

enum QwenGenerationTermination: Sendable, Equatable {
    case modelEOS
    case tokenStop
    case maxTokens
    case stopString
    case cancelled
}

/// The production terminal publication boundary. Tool calls can leave this
/// function only after actual model EOS, complete parser validation, and the
/// final cancellation check. Tests call this same boundary with a real decoder.
func finalizeQwenStructuredTurn(
    decoder: inout QwenStructuredAssistantDecoder,
    tokenizerTail: String,
    termination: QwenGenerationTermination,
    publish: (StructuredAssistantEvent) -> Void
) throws {
    let events: [StructuredAssistantEvent]
    switch termination {
    case .modelEOS:
        try Task.checkCancellation()
        try decoder.markEndOfStream()
        var completed = try decoder.consumeTerminalTail(tokenizerTail)
        completed += try decoder.finish()
        events = completed
    case .tokenStop, .maxTokens, .stopString:
        events = try decoder.consume(tokenizerTail).compactMap { event in
            if case .content = event { return event }
            return nil
        }
    case .cancelled:
        return
    }
    for event in events {
        try Task.checkCancellation()
        publish(event)
    }
}

private extension ModelFamilyGenerationSession {
    func generateGemma(
        model: Model,
        request: ModelFamilyGenerationRequest,
        onEvent: @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: modelDirectoryURL)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtimeConfiguration)
        let scratch = try RawCompletionScratch(
            context: context, vocab: model.config.vocabSize)

        let promptIDs: [Int32]
        let multimodal: MultimodalPrefillInput?
        switch request.prompt {
        case .raw(let text):
            guard request.imagesByID.isEmpty else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "raw completion does not accept images")
            }
            promptIDs = tokenizer.encode(text, addBOS: true)
            multimodal = nil

        case .chat(let messages, let tools, let thinking):
            guard thinking == .automatic else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "explicit --thinking is available only for verified Qwen models")
            }
            guard tools.isEmpty,
                  messages.allSatisfy({
                      $0.toolCalls.isEmpty && $0.reasoningContent == nil
                  }) else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "structured tools and reasoning are available only for verified Qwen models")
            }
            let orderedIDs = try orderedImageIDs(in: messages)
            if orderedIDs.isEmpty {
                if let extra = request.imagesByID.keys.sorted().first {
                    throw ModelFamilyGenerationError.unexpectedImage(extra)
                }
                promptIDs = try GemmaChatCodecAdapter(tokenizer: tokenizer)
                    .encodePrompt(messages: messages, tools: tools, options: .init())
                multimodal = nil
            } else {
                guard tools.isEmpty else {
                    throw ModelFamilyGenerationError.unsupportedInput(
                        "Gemma image chat and tools cannot be combined")
                }
                let converted = try gemmaMultimodalMessages(
                    messages, orderedImageIDs: orderedIDs)
                let expected = Set(orderedIDs)
                if let missing = orderedIDs.first(where: {
                    request.imagesByID[$0] == nil
                }) {
                    throw ModelFamilyGenerationError.missingImage(missing)
                }
                if let extra = request.imagesByID.keys.sorted().first(where: {
                    !expected.contains($0)
                }) {
                    throw ModelFamilyGenerationError.unexpectedImage(extra)
                }
                let vision = try VisionRuntime.open(
                    textModelURL: modelDirectoryURL,
                    context: context,
                    visionPackURL: visionPackURL)
                var features: [UUID: VisionFeatures] = [:]
                for id in orderedIDs {
                    let uuid = converted.ids[id]!
                    features[uuid] = try vision.encodeImage(
                        at: request.imagesByID[id]!,
                        languageModel: model,
                        residencyPolicy: request.visionResidency,
                        checkCancellation: { try Task.checkCancellation() })
                }
                let rendered = try MultimodalPromptRenderer.render(
                    messages: converted.messages,
                    featuresByID: features,
                    tokenizer: tokenizer)
                promptIDs = rendered.effectiveTokenIDs
                multimodal = rendered
            }
        }

        guard !promptIDs.isEmpty else {
            throw ModelFamilyGenerationError.emptyPrompt
        }
        guard promptIDs.count < maxContext else {
            throw ModelFamilyGenerationError.contextOverflow(
                prompt: promptIDs.count,
                maxNew: 0,
                maximum: maxContext)
        }
        var effectiveConfig = request.config
        effectiveConfig.maxNewTokens = min(
            request.config.maxNewTokens, maxContext - promptIDs.count)
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            multimodalInput: multimodal,
            config: effectiveConfig,
            context: context,
            scratch: scratch,
            prefillConfig: runtimeConfiguration.prefillConfig
        ) { progress in
            switch progress {
            case .prefill(let done, let total):
                onEvent(.prefill(done: done, total: total))
            case .token(_, _, let delta):
                if !delta.isEmpty { onEvent(.text(delta)) }
            case .tail(let text):
                if !text.isEmpty { onEvent(.text(text)) }
            }
        }
        return ModelFamilyGenerationResult(
            reason: stats.reason,
            promptTokens: stats.prefillTokens,
            newTokens: stats.newTokens,
            prefillSeconds: stats.prefillSeconds,
            decodeSeconds: stats.decodeSeconds)
    }

    func gemmaMultimodalMessages(
        _ messages: [ModelChatMessage],
        orderedImageIDs: [String]
    ) throws -> (messages: [MultimodalMessage], ids: [String: UUID]) {
        var ids: [String: UUID] = [:]
        for id in orderedImageIDs { ids[id] = UUID() }
        let converted = try messages.map { message -> MultimodalMessage in
            let role: GFTokenizer.Role = switch message.role {
            case .system: .system
            case .user: .user
            case .assistant: .assistant
            case .tool: .tool
            case .developer: .developer
            case .other(_):
                throw ModelFamilyGenerationError.unsupportedInput(
                    "Gemma image chat contains an unsupported role")
            }
            guard message.reasoningContent == nil, message.toolCalls.isEmpty else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "Gemma image chat does not accept reasoning or historical tool calls")
            }
            let parts: [MultimodalContentPart]
            switch message.content {
            case .none:
                parts = []
            case .text(let text):
                parts = [.text(text)]
            case .parts(let values):
                parts = try values.map { value in
                    switch value {
                    case .text(let text): return .text(text)
                    case .image(let media):
                        guard let id = media.id, let uuid = ids[id] else {
                            throw ModelFamilyGenerationError.unsupportedInput(
                                "every chat image requires a unique media ID")
                        }
                        return .image(id: uuid)
                    case .video:
                        throw ModelFamilyGenerationError.unsupportedInput(
                            "video input is not supported")
                    case .audio:
                        throw ModelFamilyGenerationError.unsupportedInput(
                            "audio input is not supported")
                    case .unsupported(let type):
                        throw ModelFamilyGenerationError.unsupportedInput(
                            "unsupported media type: \(type)")
                    }
                }
            }
            return MultimodalMessage(
                role: role, content: parts,
                toolCallID: message.toolCallID, name: message.name)
        }
        return (converted, ids)
    }

    struct QwenPrompt {
        let tokenIDs: [Int32]
        let prepared: QwenPreparedPrefill?
        let tokenizer: QwenTokenizer
        let tools: [ModelChatToolDefinition]
        let startsInThoughtChannel: Bool
        let isChat: Bool
        let textRoPEDelta: Int
    }

    func generateQwen(
        model: QwenTextModel,
        codec: QwenChatCodec,
        request: ModelFamilyGenerationRequest,
        onEvent: @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        let prompt = try await prepareQwenPrompt(
            model: model, codec: codec, request: request)
        guard !prompt.tokenIDs.isEmpty else {
            throw ModelFamilyGenerationError.emptyPrompt
        }
        guard prompt.tokenIDs.allSatisfy({
            $0 >= 0 && Int($0) < model.architecture.vocabularySize
        }) else {
            throw ModelFamilyGenerationError.unsupportedInput(
                "prompt token is outside the loaded Qwen vocabulary")
        }
        guard prompt.tokenIDs.count < maxContext else {
            throw ModelFamilyGenerationError.contextOverflow(
                prompt: prompt.tokenIDs.count,
                maxNew: 0,
                maximum: maxContext)
        }

        let producer = try model.makeLogitProducer(
            context: context,
            expertSlotCount: runtimeConfiguration.expertCacheSlots)
        let scratch = try RawCompletionScratch(
            context: context, vocab: model.architecture.vocabularySize)
        producer.reset()

        let prefillStarted = Date()
        if let prepared = prompt.prepared {
            try await producer.prefill(prepared: prepared, into: scratch.logits)
            onEvent(.prefill(done: prompt.tokenIDs.count, total: prompt.tokenIDs.count))
        } else {
            for (position, token) in prompt.tokenIDs.enumerated() {
                try Task.checkCancellation()
                try await producer.produce(
                    token: token, position: position, into: scratch.logits)
                onEvent(.prefill(done: position + 1, total: prompt.tokenIDs.count))
            }
        }
        let decodeStarted = Date()
        let prefillSeconds = decodeStarted.timeIntervalSince(prefillStarted)

        var config = request.config
        config.maxNewTokens = min(
            request.config.maxNewTokens, maxContext - prompt.tokenIDs.count)
        config.logitTransform = .raw
        config.extraStopTokens.insert(prompt.tokenizer.eosID)
        var detokenizer = prompt.tokenizer.makeIncrementalDecoder()
        var structured = QwenStructuredAssistantDecoder(
            tools: prompt.tools,
            startsInThoughtChannel: prompt.startsInThoughtChannel)
        var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
        var history = prompt.tokenIDs
        history.reserveCapacity(prompt.tokenIDs.count + config.maxNewTokens)
        var generated = 0
        var reason: StopReason = .maxTokens
        var position = prompt.tokenIDs.count
        var termination: QwenGenerationTermination = .maxTokens

        func publish(_ text: String) throws {
            guard !text.isEmpty else { return }
            if prompt.isChat {
                for event in try structured.consume(text) {
                    if case .content(let content) = event {
                        let visible = stopMatcher.push(content)
                        if !visible.isEmpty { onEvent(.text(visible)) }
                    }
                }
            } else {
                let visible = stopMatcher.push(text)
                if !visible.isEmpty { onEvent(.text(visible)) }
            }
        }

        do {
            while generated < config.maxNewTokens {
                try Task.checkCancellation()
                let token = try sampleQwen(
                    scratch: scratch, history: history, config: config,
                    samplePosition: generated)
                try Task.checkCancellation()
                generated += 1
                if config.extraStopTokens.contains(token) {
                    reason = .eos
                    termination = token == prompt.tokenizer.eosID
                        ? .modelEOS : .tokenStop
                    break
                }
                try publish(detokenizer.push(token))
                if stopMatcher.isStopped {
                    reason = .stopString
                    termination = .stopString
                    break
                }
                history.append(token)
                let rope = position + prompt.textRoPEDelta
                try await producer.produce(
                    token: token,
                    cachePosition: position,
                    ropePosition: try QwenMRoPEPosition(
                        temporal: rope, height: rope, width: rope),
                    into: scratch.logits)
                position += 1
            }
        } catch is CancellationError {
            try finalizeQwenStructuredTurn(
                decoder: &structured,
                tokenizerTail: "",
                termination: .cancelled,
                publish: { _ in })
            throw CancellationError()
        }

        let tokenizerTail = detokenizer.finish()
        if prompt.isChat {
            try finalizeQwenStructuredTurn(
                decoder: &structured,
                tokenizerTail: tokenizerTail,
                termination: termination
            ) { event in
                switch event {
                case .content(let content):
                    let visible = stopMatcher.push(content)
                    if !visible.isEmpty { onEvent(.text(visible)) }
                case .toolCall(let call):
                    onEvent(.toolCall(call))
                    reason = .toolCalls
                }
            }
        } else {
            let visible = stopMatcher.push(tokenizerTail)
            if !visible.isEmpty { onEvent(.text(visible)) }
        }
        let finalText = stopMatcher.finish()
        if !finalText.isEmpty { onEvent(.text(finalText)) }

        return ModelFamilyGenerationResult(
            reason: reason,
            promptTokens: prompt.tokenIDs.count,
            newTokens: generated,
            prefillSeconds: prefillSeconds,
            decodeSeconds: Date().timeIntervalSince(decodeStarted))
    }

    func sampleQwen(
        scratch: RawCompletionScratch,
        history: [Int32],
        config: GenerationConfig,
        samplePosition: Int
    ) throws -> Int32 {
        guard let command = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.execution(detail: "sampler command allocation failed")
        }
        scratch.sampler.sample(
            commandBuffer: command,
            logits: scratch.logits,
            probs: scratch.probs,
            history: history,
            config: config,
            position: samplePosition,
            outToken: scratch.outToken)
        command.commit()
        command.waitUntilCompleted()
        try checkCommandBufferError(command)
        return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
    }

    func prepareQwenPrompt(
        model: QwenTextModel,
        codec: QwenChatCodec,
        request: ModelFamilyGenerationRequest
    ) async throws -> QwenPrompt {
        switch request.prompt {
        case .raw(let text):
            guard request.imagesByID.isEmpty else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "raw completion does not accept images")
            }
            return QwenPrompt(
                tokenIDs: fixturePromptTokenIDs ?? codec.tokenizer.encode(text),
                prepared: nil,
                tokenizer: codec.tokenizer, tools: [],
                startsInThoughtChannel: false, isChat: false, textRoPEDelta: 0)

        case .chat(let messages, let tools, let thinking):
            let options = ModelChatRenderOptions(
                enableThinking: thinking != .disabled)
            let tokenIDs = try codec.encodePrompt(
                messages: messages, tools: tools, options: options)
            let effectiveTokenIDs = fixturePromptTokenIDs ?? tokenIDs
            let orderedIDs = try orderedImageIDs(in: messages)
            guard !orderedIDs.isEmpty else {
                if let extra = request.imagesByID.keys.sorted().first {
                    throw ModelFamilyGenerationError.unexpectedImage(extra)
                }
                return QwenPrompt(
                    tokenIDs: effectiveTokenIDs, prepared: nil,
                    tokenizer: codec.tokenizer, tools: tools,
                    startsInThoughtChannel: options.enableThinking,
                    isChat: true, textRoPEDelta: 0)
            }
            let expected = Set(orderedIDs)
            if let missing = orderedIDs.first(where: { request.imagesByID[$0] == nil }) {
                throw ModelFamilyGenerationError.missingImage(missing)
            }
            if let extra = request.imagesByID.keys.sorted().first(where: {
                !expected.contains($0)
            }) {
                throw ModelFamilyGenerationError.unexpectedImage(extra)
            }
            guard let identity = verifiedIdentity else {
                throw ModelFamilyGenerationError.verifiedVisionUnavailable
            }
            guard request.visionResidency == .onDemand else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "verified Qwen vision supports on-demand residency only")
            }
            guard fixturePromptTokenIDs == nil else {
                throw ModelFamilyGenerationError.unsupportedInput(
                    "tiny text fixture prompt overrides cannot carry images")
            }
            let companion = try Self.openVerifiedQwenVisionCompanion(
                directoryURL: modelDirectoryURL,
                loadedIdentity: identity,
                visionPackURL: visionPackURL)
            guard case .qwen3_6(let wireArchitecture) =
                    companion.textManifest.architecture else {
                throw ModelFamilyGenerationError.modelIdentityChanged
            }
            let rendererTokens = try normalizeQwenCodecImageFrames(
                tokenIDs,
                architecture: wireArchitecture)
            let preprocessor = QwenImagePreprocessor(device: context.device)
            let plans = try orderedIDs.map {
                try preprocessor.plan(fileURL: request.imagesByID[$0]!)
            }
            try QwenImagePreprocessor.preflight(plans.map(\.geometry))
            let pixels = try plans.map(preprocessor.preprocess)
            let vision = try QwenVisionRuntime(
                context: context, store: companion.store)
            let features = try await vision.process(pixels)
            let multimodal = try MultimodalPromptRenderer.expandingQwenImageTokens(
                rendererTokens,
                features: features,
                architecture: wireArchitecture)
            let prepared = try QwenPreparedPrefill(
                tokenIDs: multimodal.embeddingTokenIDs,
                featureOverrides: multimodal.imageSpans.map {
                    QwenPreparedFeatureOverride(
                        tokenRange: $0.tokenRange, owner: $0.features.owner)
                },
                positions: multimodal.positionPlan.positions,
                textRoPEDelta: multimodal.positionPlan.textRoPEDelta)
            return QwenPrompt(
                tokenIDs: multimodal.effectiveTokenIDs,
                prepared: prepared, tokenizer: codec.tokenizer, tools: tools,
                startsInThoughtChannel: options.enableThinking,
                isChat: true,
                textRoPEDelta: multimodal.positionPlan.textRoPEDelta)
        }
    }

}
