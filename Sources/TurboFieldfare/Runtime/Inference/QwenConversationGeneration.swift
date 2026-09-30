import Foundation
import Metal

public enum QwenConversationTurn: Sendable, Equatable {
    case user(ModelChatMessage)
    case toolResults([ModelChatMessage])
    /// Resumes generation from a committed two-phase checkpoint without
    /// appending another prompt suffix.
    case checkpoint(UUID)
}

public struct QwenConversationGenerationRequest: Sendable {
    public var turn: QwenConversationTurn
    /// Standing host instruction rendered in Qwen's system channel on the
    /// first retained turn only.
    public var systemPrompt: String?
    public var tools: [ModelChatToolDefinition]
    public var imagesByID: [String: URL]
    public var thinking: ModelFamilyThinkingMode
    public var visionResidency: VisionResidencyPolicy
    public var config: GenerationConfig

    public init(
        turn: QwenConversationTurn,
        systemPrompt: String? = nil,
        tools: [ModelChatToolDefinition] = [],
        imagesByID: [String: URL] = [:],
        thinking: ModelFamilyThinkingMode = .automatic,
        visionResidency: VisionResidencyPolicy = .defaultPolicy,
        config: GenerationConfig
    ) {
        self.turn = turn
        self.systemPrompt = systemPrompt
        self.tools = tools
        self.imagesByID = imagesByID
        self.thinking = thinking
        self.visionResidency = visionResidency
        self.config = config
    }
}

public enum QwenConversationGenerationEvent: Sendable, Equatable {
    case prefill(done: Int, total: Int)
    case structuredProgress(StructuredAssistantProgress)
    case text(String)
    /// Published only after model EOS and complete schema validation.
    case toolCall(ParsedToolCall)
}

public struct QwenConversationGenerationResult: Sendable, Equatable {
    public let reason: StopReason
    public let promptTokens: Int
    public let newTokens: Int
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    public let metrics: ConversationStateMetrics
    public let acceptedGeneratedTokenIDs: [Int32]
    /// Present only for a committed, trust-gated original-source turn. This
    /// local source identity is not a whole-shard payload-authenticity claim.
    public let sourceIdentity: LoadedRuntimeSourceIdentity?

    public init(
        reason: StopReason,
        promptTokens: Int,
        newTokens: Int,
        prefillSeconds: Double,
        decodeSeconds: Double,
        metrics: ConversationStateMetrics,
        acceptedGeneratedTokenIDs: [Int32],
        sourceIdentity: LoadedRuntimeSourceIdentity? = nil
    ) {
        self.reason = reason
        self.promptTokens = promptTokens
        self.newTokens = newTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.metrics = metrics
        self.acceptedGeneratedTokenIDs = acceptedGeneratedTokenIDs
        self.sourceIdentity = sourceIdentity
    }
}

public struct QwenConversationCheckpointRequest: Sendable {
    public var checkpointID: UUID
    public var messages: [ModelChatMessage]
    public var tools: [ModelChatToolDefinition]
    public var imagesByID: [String: URL]
    public var thinking: ModelFamilyThinkingMode
    public var visionResidency: VisionResidencyPolicy
    public var reason: QwenCheckpointReason
    public var commit: Bool

    public init(
        checkpointID: UUID,
        messages: [ModelChatMessage],
        tools: [ModelChatToolDefinition] = [],
        imagesByID: [String: URL] = [:],
        thinking: ModelFamilyThinkingMode = .automatic,
        visionResidency: VisionResidencyPolicy = .defaultPolicy,
        reason: QwenCheckpointReason,
        commit: Bool
    ) {
        self.checkpointID = checkpointID
        self.messages = messages
        self.tools = tools
        self.imagesByID = imagesByID
        self.thinking = thinking
        self.visionResidency = visionResidency
        self.reason = reason
        self.commit = commit
    }
}

public enum QwenConversationCheckpointEvent: Sendable, Equatable {
    case progress(done: Int, total: Int)
}

public struct QwenConversationImagePreflight: Sendable, Equatable {
    public let retainedImageCount: Int
    public let retainedImageRows: Int
    public let retainedFeatureBytes: Int

    public init(
        retainedImageCount: Int,
        retainedImageRows: Int,
        retainedFeatureBytes: Int
    ) {
        self.retainedImageCount = retainedImageCount
        self.retainedImageRows = retainedImageRows
        self.retainedFeatureBytes = retainedFeatureBytes
    }
}

public struct QwenConversationCheckpointResult: Sendable, Equatable {
    public let committed: Bool
    public let promptTokens: Int
    public let metrics: ConversationStateMetrics
    public let retainedImageCount: Int
    public let retainedImageRows: Int
    public let retainedFeatureBytes: Int

    public init(
        committed: Bool,
        promptTokens: Int,
        metrics: ConversationStateMetrics,
        retainedImageCount: Int,
        retainedImageRows: Int,
        retainedFeatureBytes: Int
    ) {
        self.committed = committed
        self.promptTokens = promptTokens
        self.metrics = metrics
        self.retainedImageCount = retainedImageCount
        self.retainedImageRows = retainedImageRows
        self.retainedFeatureBytes = retainedFeatureBytes
    }
}

public enum QwenConversationGenerationError: Error, Equatable, CustomStringConvertible {
    case invalidTurn(String)
    case toolsChanged
    case systemPromptChanged
    case fixtureImagesUnavailable
    case rollbackFailed(operation: String, rollback: String)

    public var description: String {
        switch self {
        case .invalidTurn(let detail): detail
        case .toolsChanged:
            "tool definitions changed inside a retained conversation"
        case .systemPromptChanged:
            "the system prompt changed inside a retained conversation"
        case .fixtureImagesUnavailable:
            "the text-only fixture session has no injected image preparation"
        case .rollbackFailed(let operation, let rollback):
            "operation failed with \(operation); rollback failed with \(rollback)"
        }
    }
}

/// Internal deterministic image seam. Row counts and requested bytes are
/// available before `prepare`, so tests exercise the same reserve-before-
/// allocation and retained lineage path as production.
struct QwenConversationFixtureImagePlan: Sendable {
    let architecture: QwenArchConfig
    let visionConfig: QwenVisionConfig
    let mergedRows: [Int]
    let requestedAllocationBytes: Int
    let prepare: @Sendable () async throws -> [QwenVisionFeatures]

    init(
        architecture: QwenArchConfig,
        visionConfig: QwenVisionConfig,
        mergedRows: [Int],
        requestedAllocationBytes: Int,
        prepare: @escaping @Sendable () async throws -> [QwenVisionFeatures]
    ) {
        self.architecture = architecture
        self.visionConfig = visionConfig
        self.mergedRows = mergedRows
        self.requestedAllocationBytes = requestedAllocationBytes
        self.prepare = prepare
    }
}

typealias QwenConversationFixtureImagePlanner = @Sendable (
    _ orderedImageIDs: [String],
    _ imagesByID: [String: URL]
) throws -> QwenConversationFixtureImagePlan

typealias QwenConversationFixtureTokenMapper = @Sendable (
    _ composedTokenIDs: [Int32]
) throws -> [Int32]

/// Stateful ordinary Qwen generation over an injected loaded model and the
/// exact `QwenConversationState` also used by the prepared-token boundary.
/// This type never constructs or loads another text model.
public actor QwenConversationGenerationSession {
    private struct PendingCheckpoint: Sendable, Equatable {
        let id: UUID
        let thinking: ModelFamilyThinkingMode
    }

    private struct PreparedImages {
        let prepared: QwenPreparedPrefill
        let lineage: QwenImageLineage
        let reservation: QwenLineageReservation
        let effectiveTokenCount: Int
    }

    private let model: QwenTextModel
    private let state: QwenConversationState
    private let codec: QwenChatCodec
    private let verifiedIdentity: LoadedRuntimeIdentity?
    private let modelDirectoryURL: URL?
    private let context: MetalContext
    private let maxContext: Int
    private let visionPackURL: URL?
    private let fixtureImagePlanner: QwenConversationFixtureImagePlanner?
    private let fixtureTokenMapper: QwenConversationFixtureTokenMapper?
    private let fixtureGeneratedTokenIDs: [Int32]?
    private let fixtureEndTokenID: Int32?
    private let scratch: RawCompletionScratch
    private var retainedTools: [ModelChatToolDefinition]?
    private var retainedSystemPrompt: String?
    private var retainedMetadataInitialized = false
    private var retainedContinuationBoundary: QwenChatContinuationBoundary?
    private var observedRetainedTokenIDs: [Int32]?
    private var pendingCheckpoint: PendingCheckpoint?
    private var generating = false

    public init(
        model: QwenTextModel,
        state: QwenConversationState,
        codec: QwenChatCodec,
        verifiedIdentity: LoadedRuntimeIdentity,
        modelDirectoryURL: URL,
        context: MetalContext,
        maxContext: Int,
        visionPackURL: URL? = nil
    ) throws {
        guard maxContext > 0,
              verifiedIdentity.family == .qwen3_6,
              verifiedIdentity.textManifestSHA256 == model.mappingIdentity,
              state.modelIdentity == model.mappingIdentity,
              state.contextLimit == maxContext,
              case .qwenV2(let manifest) = try ModelFamilyAdmission.classify(
                directoryURL: modelDirectoryURL),
              Self.sameVerifiedTextIdentity(
                LoadedRuntimeIdentity(descriptor: manifest.descriptor),
                verifiedIdentity) else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        self.model = model
        self.state = state
        self.codec = codec
        self.verifiedIdentity = verifiedIdentity
        self.modelDirectoryURL = modelDirectoryURL
        self.context = context
        self.maxContext = maxContext
        self.visionPackURL = visionPackURL
        fixtureImagePlanner = nil
        fixtureTokenMapper = nil
        fixtureGeneratedTokenIDs = nil
        fixtureEndTokenID = nil
        scratch = try RawCompletionScratch(
            context: context, vocab: model.architecture.vocabularySize)
    }

    init(
        fixtureModel model: QwenTextModel,
        state: QwenConversationState,
        codec: QwenChatCodec,
        context: MetalContext,
        maxContext: Int,
        imagePlanner: QwenConversationFixtureImagePlanner? = nil,
        fixtureTokenMapper: QwenConversationFixtureTokenMapper? = nil,
        fixtureGeneratedTokenIDs: [Int32]? = nil,
        fixtureEndTokenID: Int32? = nil
    ) throws {
        guard maxContext > 0,
              state.modelIdentity == model.mappingIdentity,
              state.contextLimit == maxContext else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        let vocabularySize = model.architecture.vocabularySize
        for tokens in [fixtureGeneratedTokenIDs].compactMap({ $0 }) {
            guard !tokens.isEmpty,
                  tokens.allSatisfy({ $0 >= 0 && Int($0) < vocabularySize }) else {
                throw QwenConversationGenerationError.invalidTurn(
                    "fixture token override is outside the tiny model vocabulary")
            }
        }
        if let fixtureEndTokenID {
            guard fixtureEndTokenID >= 0,
                  Int(fixtureEndTokenID) < vocabularySize else {
                throw QwenConversationGenerationError.invalidTurn(
                    "fixture terminal token is outside the tiny model vocabulary")
            }
        }
        self.model = model
        self.state = state
        self.codec = codec
        verifiedIdentity = nil
        modelDirectoryURL = nil
        self.context = context
        self.maxContext = maxContext
        visionPackURL = nil
        fixtureImagePlanner = imagePlanner
        self.fixtureTokenMapper = fixtureTokenMapper
        self.fixtureGeneratedTokenIDs = fixtureGeneratedTokenIDs
        self.fixtureEndTokenID = fixtureEndTokenID
        scratch = try RawCompletionScratch(
            context: context, vocab: model.architecture.vocabularySize)
    }

    public func generate(
        _ request: QwenConversationGenerationRequest,
        shouldStop: @escaping @Sendable () -> Bool = { false },
        onEvent: @escaping @Sendable (QwenConversationGenerationEvent) -> Void = { _ in }
    ) async throws -> QwenConversationGenerationResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }

        try request.config.validate()
        try validateCurrentIdentity()
        let statusBefore = await state.status()
        if let observedRetainedTokenIDs,
           observedRetainedTokenIDs != statusBefore.committed.retainedTokenIDs {
            retainedTools = nil
            retainedSystemPrompt = nil
            retainedMetadataInitialized = false
            retainedContinuationBoundary = nil
            pendingCheckpoint = nil
        }
        let firstTurn = statusBefore.committed.retainedTokenIDs.isEmpty
        try validateTurn(
            request.turn,
            firstTurn: firstTurn,
            systemPrompt: request.systemPrompt,
            imagesByID: request.imagesByID,
            thinking: request.thinking)
        if let retainedTools, retainedTools != request.tools {
            throw QwenConversationGenerationError.toolsChanged
        }
        if !firstTurn, let systemPrompt = request.systemPrompt {
            guard retainedMetadataInitialized,
                  retainedSystemPrompt == systemPrompt else {
                throw QwenConversationGenerationError.systemPromptChanged
            }
        }

        let turnMessages: [ModelChatMessage]
        let resumesCheckpoint: Bool
        switch request.turn {
        case .user(let message):
            turnMessages = [message]
            resumesCheckpoint = false
        case .toolResults(let values):
            turnMessages = values
            resumesCheckpoint = false
        case .checkpoint:
            turnMessages = []
            resumesCheckpoint = true
        }
        var messages = turnMessages
        if firstTurn, let systemPrompt = request.systemPrompt {
            messages.insert(
                ModelChatMessage(role: .system, content: systemPrompt), at: 0)
        }
        let options = ModelChatRenderOptions(
            enableThinking: request.thinking != .disabled)
        let encodedSuffixTokens: [Int32]
        if resumesCheckpoint {
            encodedSuffixTokens = []
        } else if firstTurn {
            encodedSuffixTokens = try codec.encodePrompt(
                messages: messages, tools: request.tools, options: options)
        } else {
            guard let boundary = retainedContinuationBoundary else {
                throw QwenConversationGenerationError.invalidTurn(
                    "retained lineage has no verified assistant turn boundary")
            }
            encodedSuffixTokens = try codec.encodeContinuation(
                messages: messages, boundary: boundary, options: options)
        }
        let suffixTokens = resumesCheckpoint
            ? [] : try mappedFixtureTokens(encodedSuffixTokens)
        if !resumesCheckpoint {
            guard !suffixTokens.isEmpty else {
                throw ModelFamilyGenerationError.emptyPrompt
            }
        }
        let orderedIDs = try orderedImageIDs(in: turnMessages)
        try validateImages(orderedIDs: orderedIDs, imagesByID: request.imagesByID)

        let transaction = try await state.begin()
        let prefillStarted = Date()
        var accepted: [Int32] = []
        do {
            try Task.checkCancellation()
            let promptTokens: Int
            if resumesCheckpoint {
                onEvent(.prefill(done: 0, total: 0))
                promptTokens = 0
            } else if orderedIDs.isEmpty {
                try await state.prefill(
                    suffixTokens,
                    transaction: transaction,
                    onProgress: { done, total in
                        onEvent(.prefill(done: done, total: total))
                    })
                promptTokens = suffixTokens.count
            } else {
                let base = try await state.diagnosticSnapshot()
                let images = try await prepareImages(
                    tokenIDs: suffixTokens,
                    composedTokenIDs: encodedSuffixTokens,
                    orderedIDs: orderedIDs,
                    imagesByID: request.imagesByID,
                    visionResidency: request.visionResidency,
                    baseTokenCount: base.retainedTokenIDs.count,
                    baseTextRoPEDelta: base.textRoPEDelta,
                    transaction: transaction)
                try await state.prefillMultimodal(
                    images.prepared,
                    lineage: images.lineage,
                    reservation: images.reservation,
                    transaction: transaction)
                onEvent(.prefill(
                    done: images.effectiveTokenCount,
                    total: images.effectiveTokenCount))
                promptTokens = images.effectiveTokenCount
            }

            let decodeStarted = Date()
            let prefillSeconds = decodeStarted.timeIntervalSince(prefillStarted)
            let statusAfterPrompt = await state.status()
            let afterPrompt = statusAfterPrompt.working
            guard let afterPrompt else {
                throw ConversationStateTransactionError.staleTransaction
            }
            let available = maxContext - afterPrompt.retainedTokenIDs.count
            guard available > 0 else {
                throw ModelFamilyGenerationError.contextOverflow(
                    prompt: afterPrompt.retainedTokenIDs.count,
                    maxNew: 1,
                    maximum: maxContext)
            }
            var config = request.config
            config.maxNewTokens = min(config.maxNewTokens, available)
            config.logitTransform = .raw
            let endTokenID = fixtureEndTokenID ?? codec.tokenizer.eosID
            config.extraStopTokens.insert(endTokenID)

            var detokenizer = codec.tokenizer.makeIncrementalDecoder()
            var structured = QwenStructuredAssistantDecoder(
                tools: request.tools,
                startsInThoughtChannel: options.enableThinking)
            var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
            var reason: StopReason = .maxTokens
            var termination: QwenGenerationTermination = .maxTokens
            var trailingStopTokenCount = 0
            var previousProgress = structured.progress
            var terminalToolCalls: [ParsedToolCall] = []

            while accepted.count < config.maxNewTokens {
                try Task.checkCancellation()
                if shouldStop() {
                    reason = .cancelled
                    termination = .cancelled
                    break
                }
                let input = try await state.prepareSampling(transaction: transaction)
                let token: Int32
                if let fixtureGeneratedTokenIDs {
                    guard accepted.count < fixtureGeneratedTokenIDs.count else {
                        throw QwenConversationGenerationError.invalidTurn(
                            "fixture generated-token sequence was exhausted")
                    }
                    token = fixtureGeneratedTokenIDs[accepted.count]
                } else {
                    token = try sample(
                        input: input,
                        config: config,
                        samplePosition: accepted.count)
                }
                try Task.checkCancellation()
                try await state.advance(token, transaction: transaction)
                accepted.append(token)

                if config.extraStopTokens.contains(token) {
                    reason = .eos
                    termination = token == endTokenID
                        ? .modelEOS : .tokenStop
                    break
                }

                let delta = detokenizer.push(token)
                let events = try structured.consumeToken(delta)
                var offeredVisibleText = false
                var publishedVisibleText = false
                for event in events {
                    guard case .content(let text) = event else { continue }
                    offeredVisibleText = offeredVisibleText || !text.isEmpty
                    let visible = stopMatcher.push(text)
                    if !visible.isEmpty {
                        publishedVisibleText = true
                        onEvent(.text(visible))
                    }
                }
                if structured.progress != previousProgress {
                    previousProgress = structured.progress
                    onEvent(.structuredProgress(previousProgress))
                }
                if offeredVisibleText {
                    trailingStopTokenCount = publishedVisibleText
                        ? 0 : trailingStopTokenCount + 1
                }
                if stopMatcher.isStopped {
                    reason = .stopString
                    termination = .stopString
                    break
                }
            }

            let tail = detokenizer.finish()
            try finalizeQwenStructuredTurn(
                decoder: &structured,
                tokenizerTail: tail,
                termination: termination
            ) { event in
                switch event {
                case .content(let text):
                    let visible = stopMatcher.push(text)
                    if !visible.isEmpty { onEvent(.text(visible)) }
                case .toolCall(let call):
                    terminalToolCalls.append(call)
                    reason = .toolCalls
                }
            }
            if structured.progress != previousProgress {
                onEvent(.structuredProgress(structured.progress))
            }
            if stopMatcher.isStopped, reason != .stopString {
                reason = .stopString
                trailingStopTokenCount = max(trailingStopTokenCount, 1)
            }
            let finalText = stopMatcher.finish()
            if !finalText.isEmpty { onEvent(.text(finalText)) }
            let sampledTokenCount = accepted.count
            if reason == .stopString, trailingStopTokenCount > 0 {
                let removable = min(trailingStopTokenCount, accepted.count)
                if removable > 0 {
                    try await state.removeSuffix(
                        tokenCount: removable, transaction: transaction)
                    accepted.removeLast(removable)
                }
            }

            try Task.checkCancellation()
            let metrics = try await state.commit(transaction: transaction)
            retainedTools = request.tools
            if firstTurn { retainedSystemPrompt = request.systemPrompt }
            retainedMetadataInitialized = true
            retainedContinuationBoundary = accepted.last == endTokenID
                ? .endedWithEndToken : .openAssistant
            observedRetainedTokenIDs = metrics.retainedTokenIDs
            pendingCheckpoint = nil
            for call in terminalToolCalls { onEvent(.toolCall(call)) }
            return QwenConversationGenerationResult(
                reason: reason,
                promptTokens: promptTokens,
                newTokens: sampledTokenCount,
                prefillSeconds: prefillSeconds,
                decodeSeconds: Date().timeIntervalSince(decodeStarted),
                metrics: metrics,
                acceptedGeneratedTokenIDs: accepted)
        } catch let operationError {
            let failureStatus = await state.status()
            if failureStatus.activeTransaction == transaction {
                do {
                    try await state.rollback(transaction: transaction)
                } catch {
                    throw QwenConversationGenerationError.rollbackFailed(
                        operation: String(describing: operationError),
                        rollback: String(describing: error))
                }
            }
            throw normalizedCancellation(operationError)
        }
    }

    public func rebuildCheckpoint(
        _ request: QwenConversationCheckpointRequest,
        onEvent: @escaping @Sendable (QwenConversationCheckpointEvent) -> Void = { _ in }
    ) async throws -> QwenConversationCheckpointResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        try validateCurrentIdentity()

        let options = ModelChatRenderOptions(
            enableThinking: request.thinking != .disabled,
            preserveThinking: true)
        let encodedTokenIDs = try codec.encodePrompt(
            messages: request.messages, tools: request.tools, options: options)
        let tokenIDs = try mappedFixtureTokens(encodedTokenIDs)
        guard !tokenIDs.isEmpty else { throw ModelFamilyGenerationError.emptyPrompt }
        let orderedIDs = try orderedImageIDs(in: request.messages)
        try validateImages(orderedIDs: orderedIDs, imagesByID: request.imagesByID)

        let transaction = try await state.begin()
        do {
            try Task.checkCancellation()
            let promptTokens: Int
            let retainedImageCount: Int
            let retainedImageRows: Int
            let retainedFeatureBytes: Int
            if orderedIDs.isEmpty {
                try await state.rebuildCheckpoint(
                    retaining: tokenIDs,
                    transaction: transaction,
                    onProgress: { done, total in
                        onEvent(.progress(done: done, total: total))
                    })
                promptTokens = tokenIDs.count
                retainedImageCount = 0
                retainedImageRows = 0
                retainedFeatureBytes = 0
            } else {
                let images = try await prepareImages(
                    tokenIDs: tokenIDs,
                    composedTokenIDs: encodedTokenIDs,
                    orderedIDs: orderedIDs,
                    imagesByID: request.imagesByID,
                    visionResidency: request.visionResidency,
                    baseTokenCount: 0,
                    baseTextRoPEDelta: 0,
                    transaction: transaction)
                try await state.rebuildMultimodalCheckpoint(
                    prepared: images.prepared,
                    lineage: images.lineage,
                    reservation: images.reservation,
                    transaction: transaction)
                onEvent(.progress(
                    done: images.effectiveTokenCount,
                    total: images.effectiveTokenCount))
                promptTokens = images.effectiveTokenCount
                retainedImageCount = images.lineage.owners.count
                retainedImageRows = images.lineage.rowCount
                retainedFeatureBytes = images.lineage.ownedRequestedBytes
            }
            try Task.checkCancellation()
            if request.commit {
                let metrics = try await state.commit(transaction: transaction)
                retainedTools = request.tools
                retainedSystemPrompt = checkpointSystemPrompt(request.messages)
                retainedMetadataInitialized = true
                retainedContinuationBoundary = .openAssistant
                observedRetainedTokenIDs = metrics.retainedTokenIDs
                pendingCheckpoint = PendingCheckpoint(
                    id: request.checkpointID,
                    thinking: request.thinking)
                return QwenConversationCheckpointResult(
                    committed: true,
                    promptTokens: promptTokens,
                    metrics: metrics,
                    retainedImageCount: retainedImageCount,
                    retainedImageRows: retainedImageRows,
                    retainedFeatureBytes: retainedFeatureBytes)
            }
            try await state.rollback(transaction: transaction)
            let metrics = await state.status().committed
            return QwenConversationCheckpointResult(
                committed: false,
                promptTokens: promptTokens,
                metrics: metrics,
                retainedImageCount: retainedImageCount,
                retainedImageRows: retainedImageRows,
                retainedFeatureBytes: retainedFeatureBytes)
        } catch let operationError {
            let failureStatus = await state.status()
            if failureStatus.activeTransaction == transaction {
                do {
                    try await state.rollback(transaction: transaction)
                } catch {
                    throw QwenConversationGenerationError.rollbackFailed(
                        operation: String(describing: operationError),
                        rollback: String(describing: error))
                }
            }
            throw normalizedCancellation(operationError)
        }
    }

    private func normalizedCancellation(_ error: Error) -> Error {
        guard let runnerError = error as? QwenTextRunnerError,
              runnerError == .cancelled else { return error }
        return CancellationError()
    }

    /// Metadata and geometry only. No pixels, vision inference, feature owner,
    /// or conversation transaction is created by this assessment.
    public func preflightCheckpointImages(
        orderedImageIDs: [String],
        imagesByID: [String: URL],
        visionResidency: VisionResidencyPolicy = .defaultPolicy
    ) throws -> QwenConversationImagePreflight {
        try validateCurrentIdentity()
        guard visionResidency == .onDemand else {
            throw ModelFamilyGenerationError.unsupportedInput(
                "verified Qwen vision supports on-demand residency only")
        }
        guard !orderedImageIDs.isEmpty,
              Set(orderedImageIDs).count == orderedImageIDs.count else {
            throw ModelFamilyGenerationError.unsupportedInput(
                "checkpoint image IDs must be nonempty and unique")
        }
        try validateImages(
            orderedIDs: orderedImageIDs, imagesByID: imagesByID)

        let rows: [Int]
        let bytes: Int
        if let fixtureImagePlanner {
            let plan = try fixtureImagePlanner(orderedImageIDs, imagesByID)
            rows = plan.mergedRows
            bytes = plan.requestedAllocationBytes
            guard bytes == (try retainedOwnerBytes(
                mergedRows: rows, visionConfig: plan.visionConfig)) else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
        } else {
            guard let verifiedIdentity, let modelDirectoryURL else {
                throw ModelFamilyGenerationError.verifiedVisionUnavailable
            }
            _ = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
                directoryURL: modelDirectoryURL,
                loadedIdentity: verifiedIdentity,
                visionPackURL: visionPackURL)
            let preprocessor = QwenImagePreprocessor(device: context.device)
            let plans = try orderedImageIDs.map {
                try preprocessor.plan(fileURL: imagesByID[$0]!)
            }
            try QwenImagePreprocessor.preflight(plans.map(\.geometry))
            rows = plans.map(\.geometry.mergedRows)
            bytes = try retainedOwnerBytes(
                mergedRows: rows, visionConfig: .official)
        }
        guard rows.count == orderedImageIDs.count,
              rows.allSatisfy({ $0 > 0 }), bytes >= 0 else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        return QwenConversationImagePreflight(
            retainedImageCount: rows.count,
            retainedImageRows: rows.reduce(0, +),
            retainedFeatureBytes: bytes)
    }

    public func reset() async throws {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        try await state.reset()
        retainedTools = nil
        retainedSystemPrompt = nil
        retainedMetadataInitialized = false
        retainedContinuationBoundary = nil
        observedRetainedTokenIDs = []
        pendingCheckpoint = nil
    }

    private func validateTurn(
        _ turn: QwenConversationTurn,
        firstTurn: Bool,
        systemPrompt: String?,
        imagesByID: [String: URL],
        thinking: ModelFamilyThinkingMode
    ) throws {
        switch turn {
        case .user(let message):
            guard pendingCheckpoint == nil,
                  firstTurn || (retainedMetadataInitialized
                    && retainedContinuationBoundary != nil),
                  message.role == .user,
                  message.reasoningContent == nil,
                  message.toolCalls.isEmpty else {
                throw QwenConversationGenerationError.invalidTurn(
                    "a user turn must contain one plain user message")
            }
        case .toolResults(let messages):
            guard pendingCheckpoint == nil,
                  !firstTurn,
                  retainedMetadataInitialized,
                  retainedContinuationBoundary != nil,
                  !messages.isEmpty,
                  messages.allSatisfy({
                      $0.role == .tool
                        && $0.reasoningContent == nil
                        && $0.toolCalls.isEmpty
                  }) else {
                throw QwenConversationGenerationError.invalidTurn(
                    "tool results require an existing conversation and tool-role messages")
            }
        case .checkpoint(let id):
            guard !firstTurn,
                  pendingCheckpoint == PendingCheckpoint(id: id, thinking: thinking),
                  retainedContinuationBoundary == .openAssistant,
                  systemPrompt == nil,
                  imagesByID.isEmpty else {
                throw QwenConversationGenerationError.invalidTurn(
                    "checkpoint generation requires the matching committed checkpoint with no new prompt or images")
            }
        }
    }

    private func validateCurrentIdentity() throws {
        guard let verifiedIdentity, let modelDirectoryURL else { return }
        guard model.mappingIdentity == verifiedIdentity.textManifestSHA256,
              state.modelIdentity == model.mappingIdentity,
              case .qwenV2(let manifest) = try ModelFamilyAdmission.classify(
                directoryURL: modelDirectoryURL),
              Self.sameVerifiedTextIdentity(
                LoadedRuntimeIdentity(descriptor: manifest.descriptor),
                verifiedIdentity) else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
    }

    private func checkpointSystemPrompt(
        _ messages: [ModelChatMessage]
    ) -> String? {
        guard messages.first?.role == .system else { return nil }
        guard case .text(let value) = messages.first?.content else { return nil }
        return value
    }

    private func sample(
        input: QwenConversationSamplingInput,
        config: GenerationConfig,
        samplePosition: Int
    ) throws -> Int32 {
        let requiredBytes = input.logits.count * MemoryLayout<Float16>.stride
        guard requiredBytes == scratch.logits.length else {
            throw QwenTextRunnerError.logitsBufferTooSmall(
                expected: scratch.logits.length, actual: requiredBytes)
        }
        _ = input.logits.withUnsafeBytes { source in
            memcpy(scratch.logits.contents(), source.baseAddress!, requiredBytes)
        }
        guard let command = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.execution(
                detail: "sampler command allocation failed")
        }
        scratch.sampler.sample(
            commandBuffer: command,
            logits: scratch.logits,
            probs: scratch.probs,
            history: input.retainedTokenIDs,
            config: config,
            position: samplePosition,
            outToken: scratch.outToken)
        command.commit()
        command.waitUntilCompleted()
        try checkCommandBufferError(command)
        return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
    }

    private func mappedFixtureTokens(
        _ composed: [Int32]
    ) throws -> [Int32] {
        guard let fixtureTokenMapper else { return composed }
        let mapped = try fixtureTokenMapper(composed)
        guard mapped.count == composed.count,
              mapped.allSatisfy({
                  $0 >= 0 && Int($0) < model.architecture.vocabularySize
              }) else {
            throw QwenConversationGenerationError.invalidTurn(
                "fixture token mapping must preserve count and stay inside the tiny vocabulary")
        }
        return mapped
    }

    private func validateMappedImageMarkers(
        composed: [Int32],
        mapped: [Int32],
        architecture: QwenArchConfig
    ) throws {
        guard composed.count == mapped.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        let sourceMarkers = [
            ("<|vision_start|>", Int32(architecture.visionStartTokenID)),
            ("<|image_pad|>", Int32(architecture.imageTokenID)),
            ("<|vision_end|>", Int32(architecture.visionEndTokenID)),
        ]
        for (content, target) in sourceMarkers {
            guard let source = codec.tokenizer.addedTokens.first(where: {
                $0.content == content
            })?.id else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            for index in composed.indices where composed[index] == source {
                guard mapped[index] == target else {
                    throw MultimodalPromptRendererError.placeholderMismatch
                }
            }
        }
    }

    private func prepareImages(
        tokenIDs: [Int32],
        composedTokenIDs: [Int32],
        orderedIDs: [String],
        imagesByID: [String: URL],
        visionResidency: VisionResidencyPolicy,
        baseTokenCount: Int,
        baseTextRoPEDelta: Int,
        transaction: ConversationTransactionID
    ) async throws -> PreparedImages {
        guard visionResidency == .onDemand else {
            throw ModelFamilyGenerationError.unsupportedInput(
                "verified Qwen vision supports on-demand residency only")
        }
        let fixturePlan = try fixtureImagePlanner?(orderedIDs, imagesByID)
        let architecture: QwenArchConfig
        let visionConfig: QwenVisionConfig
        let mergedRows: [Int]
        let requestedAllocationBytes: Int
        let normalized: [Int32]
        let reservation: QwenLineageReservation
        let features: [QwenVisionFeatures]
        if let fixturePlan {
            architecture = fixturePlan.architecture
            visionConfig = fixturePlan.visionConfig
            mergedRows = fixturePlan.mergedRows
            requestedAllocationBytes = fixturePlan.requestedAllocationBytes
            guard requestedAllocationBytes == (try retainedOwnerBytes(
                mergedRows: mergedRows, visionConfig: visionConfig)) else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            (normalized, reservation) = try await reserveImageLineage(
                tokenIDs: tokenIDs,
                composedTokenIDs: composedTokenIDs,
                architecture: architecture,
                mergedRows: mergedRows,
                requestedAllocationBytes: requestedAllocationBytes,
                orderedImageCount: orderedIDs.count,
                transaction: transaction)
            features = try await fixturePlan.prepare()
        } else {
            architecture = try currentArchitecture()
            visionConfig = .official
            guard let verifiedIdentity, let modelDirectoryURL else {
                throw ModelFamilyGenerationError.verifiedVisionUnavailable
            }
            let store = try ModelFamilyGenerationSession
                .openVerifiedQwenVisionCompanion(
                    directoryURL: modelDirectoryURL,
                    loadedIdentity: verifiedIdentity,
                    visionPackURL: visionPackURL).store
            let preprocessor = QwenImagePreprocessor(device: context.device)
            let plans = try orderedIDs.map {
                try preprocessor.plan(fileURL: imagesByID[$0]!)
            }
            try QwenImagePreprocessor.preflight(plans.map(\.geometry))
            mergedRows = plans.map(\.geometry.mergedRows)
            requestedAllocationBytes = try retainedOwnerBytes(
                mergedRows: mergedRows, visionConfig: .official)
            (normalized, reservation) = try await reserveImageLineage(
                tokenIDs: tokenIDs,
                composedTokenIDs: composedTokenIDs,
                architecture: architecture,
                mergedRows: mergedRows,
                requestedAllocationBytes: requestedAllocationBytes,
                orderedImageCount: orderedIDs.count,
                transaction: transaction)
            let pixels = try plans.map(preprocessor.preprocess)
            let runtime = try QwenVisionRuntime(context: context, store: store)
            features = try await runtime.process(pixels)
        }
        guard features.count == orderedIDs.count,
              features.map(\.tokenCount) == mergedRows,
              features.allSatisfy({
                  $0.hiddenSize == visionConfig.outputHiddenSize
              }) else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        let rendered = try MultimodalPromptRenderer.expandingQwenImageTokens(
            normalized,
            features: features,
            architecture: architecture,
            config: visionConfig)
        let positionOffset = try checkedAdd(
            baseTokenCount, baseTextRoPEDelta, detail: "multimodal position offset")
        let absolutePositions = try rendered.positionPlan.positions.map { position in
            try QwenMRoPEPosition(
                temporal: checkedAdd(
                    Int(position.temporal), positionOffset,
                    detail: "temporal multimodal position"),
                height: checkedAdd(
                    Int(position.height), positionOffset,
                    detail: "height multimodal position"),
                width: checkedAdd(
                    Int(position.width), positionOffset,
                    detail: "width multimodal position"))
        }
        let combinedDelta = try checkedAdd(
            baseTextRoPEDelta,
            rendered.positionPlan.textRoPEDelta,
            detail: "retained multimodal text delta")
        let prepared = try QwenPreparedPrefill(
            tokenIDs: rendered.embeddingTokenIDs,
            featureOverrides: rendered.imageSpans.map {
                QwenPreparedFeatureOverride(
                    tokenRange: $0.tokenRange, owner: $0.features.owner)
            },
            positions: absolutePositions,
            textRoPEDelta: combinedDelta)
        return PreparedImages(
            prepared: prepared,
            lineage: QwenImageLineage(
                owners: features.map(\.owner), textRoPEDelta: combinedDelta),
            reservation: reservation,
            effectiveTokenCount: rendered.effectiveTokenIDs.count)
    }

    private func reserveImageLineage(
        tokenIDs: [Int32],
        composedTokenIDs: [Int32],
        architecture: QwenArchConfig,
        mergedRows: [Int],
        requestedAllocationBytes: Int,
        orderedImageCount: Int,
        transaction: ConversationTransactionID
    ) async throws -> ([Int32], QwenLineageReservation) {
        guard mergedRows.count == orderedImageCount,
              mergedRows.allSatisfy({ $0 > 0 }) else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        try validateMappedImageMarkers(
            composed: composedTokenIDs,
            mapped: tokenIDs,
            architecture: architecture)
        let normalized = try normalizeQwenCodecImageFrames(
            tokenIDs, architecture: architecture)
        let expandedCount = try qwenExpandedPromptTokenCount(
            normalizedTokenIDs: normalized,
            imageMergedRows: mergedRows,
            imageTokenID: Int32(architecture.imageTokenID))
        let provenance = try QwenLineageProvenancePlan(
            tokenCount: expandedCount,
            positionCount: expandedCount,
            featureOverrideCount: orderedImageCount)
        let reservation = try await state.reserveLineage(
            plannedRows: mergedRows.reduce(0, +),
            requestedAllocationBytes: requestedAllocationBytes,
            provenancePlan: provenance,
            transaction: transaction)
        return (normalized, reservation)
    }

    private func currentArchitecture() throws -> QwenArchConfig {
        if let verifiedIdentity, let modelDirectoryURL {
            guard case .qwenV2(let manifest) = try ModelFamilyAdmission.classify(
                directoryURL: modelDirectoryURL),
                  case .qwen3_6(let architecture) = manifest.architecture,
                  Self.sameVerifiedTextIdentity(
                    LoadedRuntimeIdentity(descriptor: manifest.descriptor),
                    verifiedIdentity) else {
                throw ModelFamilyGenerationError.modelIdentityChanged
            }
            return architecture
        }
        throw QwenConversationGenerationError.fixtureImagesUnavailable
    }

    private func retainedOwnerBytes(
        mergedRows: [Int],
        visionConfig: QwenVisionConfig
    ) throws -> Int {
        let bytesPerRow = try checkedAdd(
            visionConfig.outputHiddenSize * MemoryLayout<Float>.stride,
            3 * MemoryLayout<Int32>.stride,
            detail: "retained image row bytes")
        return try mergedRows.reduce(0) { total, rows in
            let (bytes, productOverflow) = rows.multipliedReportingOverflow(by: bytesPerRow)
            let (next, sumOverflow) = total.addingReportingOverflow(bytes)
            guard !productOverflow, !sumOverflow else {
                throw QwenVisionError.arithmeticOverflow("retained image owner bytes")
            }
            return next
        }
    }

    private func orderedImageIDs(in messages: [ModelChatMessage]) throws -> [String] {
        var ordered: [String] = []
        var seen: Set<String> = []
        for message in messages {
            guard case .parts(let parts) = message.content else { continue }
            for part in parts {
                switch part {
                case .image(let media):
                    guard let id = media.id, !id.isEmpty else {
                        throw ModelFamilyGenerationError.unsupportedInput(
                            "every chat image requires a media ID")
                    }
                    guard seen.insert(id).inserted else {
                        throw ModelFamilyGenerationError.duplicateImageID(id)
                    }
                    ordered.append(id)
                case .video:
                    throw ModelFamilyGenerationError.unsupportedInput(
                        "video input is not supported")
                case .audio:
                    throw ModelFamilyGenerationError.unsupportedInput(
                        "audio input is not supported")
                case .text(_), .unsupported(_):
                    continue
                }
            }
        }
        return ordered
    }

    private func validateImages(
        orderedIDs: [String],
        imagesByID: [String: URL]
    ) throws {
        let expected = Set(orderedIDs)
        if let missing = orderedIDs.first(where: { imagesByID[$0] == nil }) {
            throw ModelFamilyGenerationError.missingImage(missing)
        }
        if let extra = imagesByID.keys.sorted().first(where: {
            !expected.contains($0)
        }) {
            throw ModelFamilyGenerationError.unexpectedImage(extra)
        }
    }

    private func checkedAdd(
        _ lhs: Int,
        _ rhs: Int,
        detail: String
    ) throws -> Int {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow, value >= 0 else {
            throw QwenVisionError.arithmeticOverflow(detail)
        }
        return value
    }

    /// Text admission and the fully verified adjacent companion are separate
    /// artifacts. Text identity comparison deliberately excludes the display
    /// vision field; image capability comes only from the opened companion.
    private nonisolated static func sameVerifiedTextIdentity(
        _ lhs: LoadedRuntimeIdentity,
        _ rhs: LoadedRuntimeIdentity
    ) -> Bool {
        lhs.family == rhs.family
            && lhs.modelID == rhs.modelID
            && lhs.sourceRevision == rhs.sourceRevision
            && lhs.formatMajor == rhs.formatMajor
            && lhs.formatMinor == rhs.formatMinor
            && lhs.sourceIndexSHA256 == rhs.sourceIndexSHA256
            && lhs.quantizationPolicySHA256 == rhs.quantizationPolicySHA256
            && lhs.textManifestSHA256 == rhs.textManifestSHA256
            && lhs.quantization == rhs.quantization
    }
}
