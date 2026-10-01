import Foundation
import Metal

/// One rendered source prompt, its exact UTF-8 bytes and token IDs, bound to
/// the retained source identity. This value does not admit a codec or source;
/// only the trust-gated production session can commit it.
struct QwenOfficialSourcePromptBinding: Sendable, Equatable {
    let promptBytes: Data
    let tokenIDs: [Int32]
    let sourceIdentity: LoadedRuntimeSourceIdentity

    static func prepare(
        codec: QwenChatCodec, messages: [ModelChatMessage],
        boundary: QwenChatContinuationBoundary?,
        options: ModelChatRenderOptions,
        sourceIdentity: LoadedRuntimeSourceIdentity,
        tools: [ModelChatToolDefinition] = []
    ) throws -> Self {
        let rendered: String
        if let boundary {
            rendered = try codec.renderContinuation(
                messages: messages, boundary: boundary, options: options)
        } else {
            rendered = try codec.renderPrompt(
                messages: messages, tools: tools, options: options)
        }
        return Self(promptBytes: Data(rendered.utf8),
                    tokenIDs: codec.tokenizer.encode(rendered),
                    sourceIdentity: sourceIdentity)
    }

    func validateIdentity(current: LoadedRuntimeSourceIdentity?) throws {
        guard current == sourceIdentity else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
    }

    func validateForCommit(model: QwenOfficialSourceModel) throws {
        try validateIdentity(current: model.sourceIdentity)
        try model.revalidateSource()
    }
}

/// Bounded synthetic decoded steps; these are not claims of model output.
/// Only the codec-free tiny fixture may supply them to the real transaction.
enum QwenOfficialSourcePreparedToolStep: Sendable {
    case token(id: Int32, decoded: String)
    case modelEOS(id: Int32, tokenizerTail: String)

    var id: Int32 {
        switch self {
        case .token(let id, _), .modelEOS(let id, _): id
        }
    }

    var text: String {
        switch self {
        case .token(_, let text), .modelEOS(_, let text): text
        }
    }
}

/// Text/tool original-BF16 Qwen conversation. The protected two-layer fixture
/// enters the identical transaction/sampler loop with prepared token IDs;
/// that internal entry neither admits a public source nor forges a codec.
public actor QwenOfficialSourceConversationGenerationSession {
    private let model: QwenOfficialSourceModel
    private let state: QwenOfficialSourceConversationState
    private let codec: QwenChatCodec?
    /// Captured with the protected tokenizer at production bundle admission;
    /// nil only for the internal codec-free tiny fixture.
    private let codecSourceIdentity: LoadedRuntimeSourceIdentity?
    private let context: MetalContext
    private let scratch: RawCompletionScratch
    private let hooks: QwenOfficialSourceTransactionHooks
    /// Internal proof capture replaces only prefill observations. Decode keeps hooks.
    private let groupedCapture: QwenSourceGroupedPrefillCapture?
    private let maxContext: Int
    private let modelDirectoryURL: URL?
    private let visionPackURL: URL?
    private var generating = false
    private var retainedSystemPrompt: String?
    private var retainedContinuationBoundary: QwenChatContinuationBoundary?
    private var retainedMetadataInitialized = false
    private var retainedTools: [ModelChatToolDefinition]?
    private var outstandingToolCalls: [String: String] = [:]

    init(model: QwenOfficialSourceModel, codec: QwenChatCodec,
         sourceIdentity: LoadedRuntimeSourceIdentity,
         context: MetalContext, maxContext: Int,
         expertSlotCount: Int, modelDirectoryURL: URL? = nil,
         visionPackURL: URL? = nil,
         prefillConfig: PrefillRuntimeConfig = .defaultChunked,
         hooks: QwenOfficialSourceTransactionHooks = .none,
         groupedCapture: QwenSourceGroupedPrefillCapture? = nil) async throws {
        guard model.sourceIdentity == sourceIdentity,
              context.device === model.context.device,
              context.queue === model.context.queue else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        try model.revalidateSource()
        let state = try await QwenOfficialSourceConversationState(
            model: model, context: context, maxContext: maxContext,
            expertSlotCount: expertSlotCount,
            groupedPrefillEnabled: prefillConfig.mode != .off, hooks: hooks)
        self.model = model
        self.state = state
        self.codec = codec
        self.codecSourceIdentity = sourceIdentity
        self.context = context
        self.maxContext = maxContext
        self.modelDirectoryURL = modelDirectoryURL
        self.visionPackURL = visionPackURL
        scratch = try RawCompletionScratch(
            context: context, vocab: model.architecture.vocabularySize)
        self.hooks = hooks
        self.groupedCapture = groupedCapture
    }

    /// Luna's protected tiny fixture: real BF16 reads and GPU execution.
    /// Prepared text turns sample normally; tool tests may simulate decoding.
    init(fixtureModel: QwenOfficialSourceModel, context: MetalContext,
         maxContext: Int, expertSlotCount: Int,
         hooks: QwenOfficialSourceTransactionHooks = .none,
         groupedCapture: QwenSourceGroupedPrefillCapture? = nil) async throws {
        guard fixtureModel.sourceIdentity == nil,
              context.device === fixtureModel.context.device,
              context.queue === fixtureModel.context.queue else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        let state = try await QwenOfficialSourceConversationState(
            model: fixtureModel, context: context, maxContext: maxContext,
            expertSlotCount: expertSlotCount, hooks: hooks)
        model = fixtureModel
        self.state = state
        codec = nil
        codecSourceIdentity = nil
        self.context = context
        self.maxContext = maxContext
        modelDirectoryURL = nil
        visionPackURL = nil
        scratch = try RawCompletionScratch(
            context: context, vocab: fixtureModel.architecture.vocabularySize)
        self.hooks = hooks
        self.groupedCapture = groupedCapture
    }

    func groupedPrefillDiagnostics() async -> QwenSourceGroupedPrefillDiagnostics? {
        await state.groupedPrefillDiagnostics()
    }

    public func generate(
        _ request: QwenConversationGenerationRequest,
        shouldStop: @escaping @Sendable () -> Bool = { false },
        onEvent: @escaping @Sendable (QwenConversationGenerationEvent) -> Void = { _ in }
    ) async throws -> QwenConversationGenerationResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        guard let codec, let codecSourceIdentity,
              model.sourceIdentity == codecSourceIdentity else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        try request.config.validate()
        try model.revalidateSource()
        let status = await state.status()
        let firstTurn = status.committed.retainedTokenIDs.isEmpty
        if !firstTurn, retainedTools != request.tools {
            throw QwenConversationGenerationError.invalidTurn(
                "retained source tool definitions changed")
        }
        let messages: [ModelChatMessage]
        let imageIDs: [String]
        switch request.turn {
        case .user(let user):
            guard outstandingToolCalls.isEmpty, user.role == .user,
                  user.content != nil,
                  user.reasoningContent == nil, user.toolCalls.isEmpty else {
                throw QwenConversationGenerationError.invalidTurn(
                    "source conversation requires a user turn and no outstanding calls")
            }
            imageIDs = try orderedImageIDs(in: [user])
            messages = [user]
        case .toolResults(let results):
            guard !firstTurn, !outstandingToolCalls.isEmpty,
                  results.count == outstandingToolCalls.count else {
                throw QwenConversationGenerationError.invalidTurn(
                    "source tool results do not match committed calls")
            }
            var remaining = outstandingToolCalls
            for message in results {
                guard message.role == .tool, case .text? = message.content,
                      message.reasoningContent == nil, message.toolCalls.isEmpty,
                      let id = message.toolCallID, let name = message.name,
                      remaining.removeValue(forKey: id) == name else {
                    throw QwenConversationGenerationError.invalidTurn(
                        "source tool result identity or content is invalid")
                }
            }
            imageIDs = []
            messages = results
        case .checkpoint:
            throw ConversationStateTransactionError.unsupportedFamily
        }
        if let missing = imageIDs.first(where: { request.imagesByID[$0] == nil }) {
            throw ModelFamilyGenerationError.missingImage(missing)
        }
        if let extra = request.imagesByID.keys.sorted().first(where: { !imageIDs.contains($0) }) {
            throw ModelFamilyGenerationError.unexpectedImage(extra)
        }
        if !firstTurn, let systemPrompt = request.systemPrompt,
           (!retainedMetadataInitialized || retainedSystemPrompt != systemPrompt) {
            throw QwenConversationGenerationError.systemPromptChanged
        }
        var promptMessages = messages
        if firstTurn, let systemPrompt = request.systemPrompt {
            promptMessages.insert(ModelChatMessage(role: .system,
                                             content: systemPrompt), at: 0)
        }
        let options = ModelChatRenderOptions(
            enableThinking: request.thinking != .disabled)
        if !firstTurn, retainedContinuationBoundary == nil {
            throw QwenConversationGenerationError.invalidTurn(
                "retained source conversation has no committed assistant boundary")
        }
        let prompt = try QwenOfficialSourcePromptBinding.prepare(
            codec: codec, messages: promptMessages,
            boundary: firstTurn ? nil : retainedContinuationBoundary,
            options: options, sourceIdentity: codecSourceIdentity,
            tools: firstTurn ? request.tools : [])
        // Production encodes and validates the same bound value exposed to
        // independent host-side prompt-byte and identity tests.
        try prompt.validateForCommit(model: model)
        let prepared: QwenPreparedPrefill?
        var visionStore: QwenOfficialSourceVisionWeightStore?
        var producedVisionFeatureRows: Int? = nil
        if imageIDs.isEmpty { prepared = nil }
        else {
            guard request.visionResidency == .onDemand,
                  let modelDirectoryURL else {
                throw ModelFamilyGenerationError.verifiedVisionUnavailable
            }
            let companion = try visionPackURL
                ?? VisionPackLocation.companionURL(forTextModel: modelDirectoryURL)
            let store = try QwenOfficialSourceVisionWeightStore.open(
                directoryURL: companion, model: model)
            visionStore = store
            try VisionRuntime.requireSupportedDevice(context.device)
            let preprocessor = QwenImagePreprocessor(device: context.device)
            let plans = try imageIDs.map { id -> QwenImagePlan in
                guard let url = request.imagesByID[id] else {
                    throw ModelFamilyGenerationError.missingImage(id)
                }
                return try preprocessor.plan(fileURL: url)
            }
            try QwenImagePreprocessor.preflight(plans.map(\.geometry))
            let pixels = try plans.map(preprocessor.preprocess)
            let vision = try QwenVisionRuntime(context: context, sourceStore: store)
            let features = try await vision.process(pixels)
            prepared = try await prepareSourceImages(
                templateTokens: prompt.tokenIDs, features: features,
                visionConfig: .official)
            producedVisionFeatureRows = features.reduce(0) { $0 + $1.tokenCount }
        }
        let result = try await generateCore(
            promptTokenIDs: prepared?.tokenIDs ?? prompt.tokenIDs,
            config: request.config, preparedPrompt: prepared,
            boundTemplateTokenIDs: prompt.tokenIDs, visionStore: visionStore,
            codec: codec, binding: prompt,
            thinking: options.enableThinking, tools: request.tools,
            firstSourceTurn: firstTurn, sourceSystemPrompt: request.systemPrompt,
            producedVisionFeatureRows: producedVisionFeatureRows,
            shouldStop: shouldStop, onEvent: onEvent)
        return result
    }

    func generatePreparedTurn(
        promptTokenIDs: [Int32], config: GenerationConfig,
        shouldStop: @escaping @Sendable () -> Bool = { false },
        onEvent: @escaping @Sendable (QwenConversationGenerationEvent) -> Void = { _ in }
    ) async throws -> QwenConversationGenerationResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        guard codec == nil, model.sourceIdentity == nil else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        return try await generateCore(
            promptTokenIDs: promptTokenIDs, config: config,
            codec: nil, binding: nil, thinking: false,
            shouldStop: shouldStop, onEvent: onEvent)
    }

    /// Internal audit of literal tokenizer IDs on an already admitted source.
    /// It uses the production state, BF16 runner, and GPU sampler unchanged.
    func generateAuditedSourceTokenTurn(
        promptTokenIDs: [Int32], config: GenerationConfig
    ) async throws -> QwenConversationGenerationResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        guard codec != nil, codecSourceIdentity != nil,
              model.sourceIdentity == codecSourceIdentity,
              promptTokenIDs == codec?.tokenizer.encode("Hello") else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        try model.revalidateSource()
        return try await generateCore(
            promptTokenIDs: promptTokenIDs, config: config,
            codec: nil, binding: nil, thinking: false,
            shouldStop: { false }, onEvent: { _ in })
    }

    /// Joins no in-flight turn implicitly: the owning session waits for its
    /// generation barrier first. A failed reset leaves the lineage unusable;
    /// retained metadata is cleared only after runner reset succeeds.
    public func reset() async throws {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        try model.revalidateSource()
        try await state.reset()
        retainedSystemPrompt = nil
        retainedContinuationBoundary = nil
        retainedMetadataInitialized = false
        retainedTools = nil
        outstandingToolCalls.removeAll(keepingCapacity: true)
        // The runner reset can suspend. Do not acknowledge a lineage reset
        // after the registered source changed during that suspension.
        try model.revalidateSource()
    }

    public func status() async -> ConversationStateStatus { await state.status() }

    /// Simulated decoding on the protected tiny BF16 fixture, not model output.
    /// Sampling alone is replaced; every token still advances the real source
    /// transaction and uses the same structured parser and commit boundary.
    func generatePreparedToolTurn(
        promptTokenIDs: [Int32],
        steps: [QwenOfficialSourcePreparedToolStep],
        tools: [ModelChatToolDefinition],
        config: GenerationConfig,
        thinking: Bool = false,
        shouldStop: @escaping @Sendable () -> Bool = { false },
        onEvent: @escaping @Sendable (QwenConversationGenerationEvent) -> Void = { _ in }
    ) async throws -> QwenConversationGenerationResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        guard codec == nil, model.sourceIdentity == nil else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        try config.validate()
        // Bounded host data and fixture-vocabulary IDs only. The actual
        // transaction also checks each ID and its retained-context limit.
        guard !tools.isEmpty, !steps.isEmpty,
              steps.count <= 256, steps.count <= config.maxNewTokens,
              steps.allSatisfy({ $0.id >= 0 &&
                  Int($0.id) < model.architecture.vocabularySize &&
                  $0.text.utf8.count <= 4_096 }),
              steps.reduce(0, { $0 + $1.text.utf8.count }) <= 65_536 else {
            throw QwenConversationGenerationError.invalidTurn(
                "prepared source tool steps exceed fixture bounds")
        }
        return try await generateCore(
            promptTokenIDs: promptTokenIDs, config: config,
            codec: nil, binding: nil, thinking: thinking,
            tools: tools, fixtureSteps: steps,
            shouldStop: shouldStop, onEvent: onEvent)
    }

    /// Internal nil-trust fixture exercises the same protected source group
    /// reader, image renderer, M-RoPE and text transaction as production.
    /// The caller finishes any writes (including GPU commands) before handing
    /// over uniquely owned pixel buffers and does not access them afterwards.
    /// This transfer does not declare their mutable Metal storage Sendable.
    func generatePreparedImageTurn(
        templateTokenIDs: [Int32], pixels: sending [QwenVisionPixelBuffer],
        companionURL: URL, visionConfig: QwenVisionConfig,
        config: GenerationConfig,
        shouldStop: @escaping @Sendable () -> Bool = { false },
        onEvent: @escaping @Sendable (QwenConversationGenerationEvent) -> Void = { _ in }
    ) async throws -> QwenConversationGenerationResult {
        guard !generating else { throw ModelFamilyGenerationError.busy }
        generating = true
        defer { generating = false }
        guard codec == nil, model.sourceIdentity == nil,
              !pixels.isEmpty, pixels.count <= 8,
              visionConfig.allowsFixtureGeometry,
              visionConfig.outputHiddenSize == model.architecture.hiddenSize else {
            throw ModelFamilyGenerationError.verifiedVisionUnavailable
        }
        let store = try QwenOfficialSourceVisionWeightStore.open(
            directoryURL: companionURL, model: model, config: visionConfig)
        try QwenImagePreprocessor.preflight(pixels.map(\.geometry))
        let vision = try QwenVisionRuntime(context: context, sourceStore: store)
        let features = try await vision.process(pixels)
        let prepared = try await prepareSourceImages(
            templateTokens: templateTokenIDs, features: features,
            visionConfig: visionConfig, alreadyNormalized: true)
        return try await generateCore(
            promptTokenIDs: prepared.tokenIDs, config: config,
            preparedPrompt: prepared, visionStore: store,
            codec: nil, binding: nil, thinking: false,
            shouldStop: shouldStop, onEvent: onEvent)
    }

    private func prepareSourceImages(
        templateTokens: [Int32], features: [QwenVisionFeatures],
        visionConfig: QwenVisionConfig,
        alreadyNormalized: Bool = false
    ) async throws -> QwenPreparedPrefill {
        guard !features.isEmpty, features.count <= 8,
              features.allSatisfy({ $0.hiddenSize == model.architecture.hiddenSize
                  && $0.hiddenSize == visionConfig.outputHiddenSize }) else {
            throw QwenVisionError.invalidFeatureShape
        }
        let normalized: [Int32]
        if alreadyNormalized { normalized = templateTokens }
        else {
            normalized = try normalizeQwenCodecImageFrames(
                templateTokens, architecture: model.visionArchitecture)
        }
        let rendered = try MultimodalPromptRenderer.expandingQwenImageTokens(
            normalized, features: features, architecture: model.visionArchitecture,
            config: visionConfig)
        let status = await state.status()
        let existing = status.committed.retainedTokenIDs.count
        guard rendered.embeddingTokenIDs.count < maxContext - existing else {
            throw ModelFamilyGenerationError.contextOverflow(
                prompt: existing + rendered.embeddingTokenIDs.count,
                maxNew: 1, maximum: maxContext)
        }
        let delta = await state.committedTextRoPEDelta()
        let (offset, overflow) = existing.addingReportingOverflow(delta)
        guard !overflow, offset >= 0 else { throw QwenVisionError.invalidPositions }
        let positions = try rendered.positionPlan.positions.map { position in
            let components = position.values.map { Int64($0) + Int64(offset) }
            guard components.allSatisfy({ $0 >= 0 && $0 <= Int64(Int32.max) }) else {
                throw QwenVisionError.invalidPositions
            }
            return try QwenMRoPEPosition(
                temporal: Int(components[0]), height: Int(components[1]),
                width: Int(components[2]))
        }
        return try QwenPreparedPrefill(
            tokenIDs: rendered.embeddingTokenIDs,
            featureOverrides: rendered.imageSpans.map {
                QwenPreparedFeatureOverride(tokenRange: $0.tokenRange, owner: $0.features.owner)
            }, positions: positions,
            textRoPEDelta: rendered.positionPlan.textRoPEDelta)
    }

    func diagnosticSnapshot() async throws
        -> QwenOfficialSourceConversationDiagnosticSnapshot {
        try await state.diagnosticSnapshot()
    }

    /// Owned expert-cache allocation and counters, excluding process physical memory.
    /// This synchronous snapshot does not wait for decode or validate source files.
    public nonisolated var currentRoutedExpertCacheSummary: RoutedExpertCacheSummary? {
        state.currentRoutedExpertCacheSummary
    }

    func cacheDiagnostics() async -> QwenOfficialSourceCacheDiagnostics {
        await state.cacheDiagnostics()
    }

    func committedJournalSnapshot() async -> QwenOfficialSourceCommittedJournalSnapshot {
        await state.committedJournalSnapshot()
    }

    /// One real BF16 transaction and one parser/publication path. Only the
    /// internal tiny fixture substitutes bounded decoded steps for sampling.
    private func generateCore(
        promptTokenIDs: [Int32], config requested: GenerationConfig,
        preparedPrompt: QwenPreparedPrefill? = nil,
        boundTemplateTokenIDs: [Int32]? = nil,
        visionStore: QwenOfficialSourceVisionWeightStore? = nil,
        codec: QwenChatCodec?,
        binding: QwenOfficialSourcePromptBinding?, thinking: Bool,
        tools: [ModelChatToolDefinition] = [],
        fixtureSteps: [QwenOfficialSourcePreparedToolStep]? = nil,
        firstSourceTurn: Bool = false,
        sourceSystemPrompt: String? = nil,
        producedVisionFeatureRows: Int? = nil,
        shouldStop: @escaping @Sendable () -> Bool,
        onEvent: @escaping @Sendable (QwenConversationGenerationEvent) -> Void
    ) async throws -> QwenConversationGenerationResult {
        try requested.validate()
        guard (codec == nil) == (binding == nil),
              fixtureSteps == nil || (codec == nil && model.sourceIdentity == nil),
              binding.map({ $0.tokenIDs == (boundTemplateTokenIDs ?? promptTokenIDs) }) ?? true,
              preparedPrompt.map({ $0.tokenIDs == promptTokenIDs }) ?? true else {
            throw ModelFamilyGenerationError.modelIdentityChanged
        }
        if let binding { try binding.validateForCommit(model: model) }
        else { try model.revalidateSource() }
        try visionStore?.revalidate()
        guard !promptTokenIDs.isEmpty else {
            throw ModelFamilyGenerationError.emptyPrompt
        }
        guard codec != nil || requested.stopStrings.isEmpty else {
            throw QwenConversationGenerationError.invalidTurn(
                "prepared token turns cannot decode stop strings without a codec")
        }
        let transaction = try await state.begin()
        let started = Date()
        var accepted: [Int32] = []
        do {
            if let preparedPrompt {
                try await state.prefillPrepared(preparedPrompt, transaction: transaction,
                    groupedCapture: groupedCapture,
                    onProgress: { done, total in
                        onEvent(.prefill(done: done, total: total))
                    })
            } else {
                try await state.prefill(
                    promptTokenIDs, transaction: transaction,
                    groupedCapture: groupedCapture,
                    onProgress: { done, total in
                        onEvent(.prefill(done: done, total: total))
                    })
            }
            let decodeStarted = Date()
            let afterPrompt = await state.status()
            guard let working = afterPrompt.working else {
                throw ConversationStateTransactionError.staleTransaction
            }
            let available = maxContext - working.retainedTokenIDs.count
            guard available > 0 else {
                throw ModelFamilyGenerationError.contextOverflow(
                    prompt: working.retainedTokenIDs.count,
                    maxNew: 1, maximum: maxContext)
            }
            var config = requested
            config.maxNewTokens = min(config.maxNewTokens, available)
            config.logitTransform = .raw
            if let codec { config.extraStopTokens.insert(codec.tokenizer.eosID) }
            var decoder = codec?.tokenizer.makeIncrementalDecoder()
            var structured = QwenStructuredAssistantDecoder(
                tools: tools, startsInThoughtChannel: thinking)
            var matcher = StreamingStopMatcher(stops: config.stopStrings)
            var lastProgress = structured.progress
            var trailingStopTokenCount = 0
            var reason: StopReason = .maxTokens
            var termination: QwenGenerationTermination = .maxTokens
            var fixtureTail: String?
            var sawVisibleText = false
            var terminalToolCalls: [ParsedToolCall] = []
            while accepted.count < config.maxNewTokens {
                try Task.checkCancellation()
                // Source conversations abort the entire provisional turn on
                // host stop; unlike packed generation, partial turns do not
                // become accepted conversation history.
                if shouldStop() { throw CancellationError() }
                let input = try await state.prepareSampling(transaction: transaction)
                let step = fixtureSteps.flatMap { steps in
                    accepted.count < steps.count ? steps[accepted.count] : nil
                }
                if fixtureSteps != nil, step == nil {
                    throw QwenConversationGenerationError.invalidTurn(
                        "prepared tool steps ended without model EOS")
                }
                let token: Int32
                if let step { token = step.id }
                else {
                    token = try await sample(input: input, config: config,
                                             samplePosition: accepted.count)
                }
                try Task.checkCancellation()
                try await state.advance(token, transaction: transaction)
                accepted.append(token)
                if let step, case .modelEOS(_, let tail) = step {
                    guard accepted.count == fixtureSteps?.count else {
                        throw QwenConversationGenerationError.invalidTurn(
                            "prepared tool steps continue after model EOS")
                    }
                    fixtureTail = tail
                    reason = .eos
                    termination = .modelEOS
                    break
                }
                if config.extraStopTokens.contains(token) {
                    reason = .eos
                    termination = token == codec?.tokenizer.eosID
                        ? .modelEOS : .tokenStop
                    break
                }
                if let delta = step?.text ?? decoder?.push(token) {
                    let events = try structured.consumeToken(delta)
                    var offered = false
                    var published = false
                    for event in events {
                        guard case .content(let text) = event else { continue }
                        offered = offered || !text.isEmpty
                        let visible = matcher.push(text)
                        if !visible.isEmpty {
                            published = true
                            sawVisibleText = true
                            onEvent(.text(visible))
                        }
                    }
                    if structured.progress != lastProgress {
                        lastProgress = structured.progress
                        onEvent(.structuredProgress(lastProgress))
                    }
                    if offered {
                        trailingStopTokenCount = published
                            ? 0 : trailingStopTokenCount + 1
                    }
                    if matcher.isStopped {
                        reason = .stopString
                        termination = .stopString
                        break
                    }
                }
            }
            // A tool-enabled turn cannot commit a truncated or host-stopped
            // frame, even if the parser has already seen its closing marker.
            if !tools.isEmpty, termination != .modelEOS {
                throw QwenConversationGenerationError.invalidTurn(
                    "source tool turn requires complete model EOS")
            }
            if decoder != nil || fixtureSteps != nil {
                let tail = fixtureTail ?? decoder?.finish() ?? ""
                try finalizeQwenStructuredTurn(
                    decoder: &structured, tokenizerTail: tail,
                    termination: termination
                ) { event in
                    switch event {
                    case .content(let text):
                        let visible = matcher.push(text)
                        if !visible.isEmpty {
                            sawVisibleText = true
                            onEvent(.text(visible))
                        }
                    case .toolCall(let call):
                        // Never publish or invoke a handler from provisional
                        // parser output. Keep all validated calls until commit.
                        terminalToolCalls.append(call)
                    }
                }
                if structured.progress != lastProgress {
                    onEvent(.structuredProgress(structured.progress))
                }
                if matcher.isStopped, reason != .stopString {
                    reason = .stopString
                    trailingStopTokenCount = max(trailingStopTokenCount, 1)
                }
                let matcherTail = matcher.finish()
                if !matcherTail.isEmpty {
                    sawVisibleText = true
                    onEvent(.text(matcherTail))
                }
            }
            if !tools.isEmpty, matcher.isStopped ||
                (!sawVisibleText && terminalToolCalls.isEmpty) {
                throw QwenConversationGenerationError.invalidTurn(
                    "source tool output is empty or interrupted")
            }
            if !terminalToolCalls.isEmpty { reason = .toolCalls }
            var nextOutstandingCalls: [String: String] = [:]
            for call in terminalToolCalls {
                guard nextOutstandingCalls.updateValue(call.name, forKey: call.id) == nil else {
                    throw QwenConversationGenerationError.invalidTurn(
                        "source tool call IDs must be unique")
                }
            }
            let sampledCount = accepted.count
            if reason == .stopString, trailingStopTokenCount > 0 {
                let removable = min(trailingStopTokenCount, accepted.count)
                if removable > 0 {
                    try await state.removeSuffix(
                        tokenCount: removable, transaction: transaction)
                    accepted.removeLast(removable)
                }
            }
            try Task.checkCancellation()
            if shouldStop() { throw CancellationError() }
            // state.commit repeats the retained receipt/file binding after its
            // final async checkpoint, before the irreversible journal swap.
            if let binding { try binding.validateForCommit(model: model) }
            else { try model.revalidateSource() }
            try visionStore?.revalidate()
            let metrics = try await state.commit(
                transaction: transaction, shouldStop: shouldStop,
                validateCompanion: { try visionStore?.revalidate() })
            // No suspension or throwing operation follows the journal swap.
            retainedTools = tools
            outstandingToolCalls = nextOutstandingCalls
            if binding != nil {
                if firstSourceTurn { retainedSystemPrompt = sourceSystemPrompt }
                retainedMetadataInitialized = true
                retainedContinuationBoundary = accepted.last == codec?.tokenizer.eosID
                    ? .endedWithEndToken : .openAssistant
            }
            for call in terminalToolCalls { onEvent(.toolCall(call)) }
            return QwenConversationGenerationResult(
                reason: reason, promptTokens: promptTokenIDs.count,
                newTokens: sampledCount,
                prefillSeconds: decodeStarted.timeIntervalSince(started),
                decodeSeconds: Date().timeIntervalSince(decodeStarted),
                metrics: metrics, acceptedGeneratedTokenIDs: accepted,
                sourceIdentity: binding?.sourceIdentity,
                producedVisionFeatureRows: producedVisionFeatureRows)
        } catch let operationError {
            let status = await state.status()
            if status.activeTransaction == transaction {
                do {
                    try await state.rollback(transaction: transaction)
                } catch {
                    throw QwenConversationGenerationError.rollbackFailed(
                        operation: String(describing: operationError),
                        rollback: String(describing: error))
                }
            }
            throw operationError
        }
    }

    private func sample(input: QwenOfficialSourceSamplingInput,
                        config: GenerationConfig,
                        samplePosition: Int) async throws -> Int32 {
        try QwenSourceSamplerBoundary.publishFP32Logits(
            input.logits, into: scratch.logits,
            vocabularySize: model.architecture.vocabularySize)
        guard let command = context.queue.makeCommandBuffer() else {
            throw QwenTextRunnerError.execution(
                detail: "source sampler command unavailable")
        }
        scratch.sampler.sample(
            commandBuffer: command, logits: scratch.logits, probs: scratch.probs,
            history: input.retainedTokenIDs, config: config,
            position: samplePosition, outToken: scratch.outToken)
        try Task.checkCancellation()
        command.commit()
        var hookError: Error?
        do { try await hooks.afterActualGPUSubmission("source.sampler") }
        catch { hookError = error }
        await withTaskCancellationHandler {
            await command.completed()
        } onCancel: {
            // The actor holds all scratch until actual Metal completion.
        }
        if let hookError { throw hookError }
        guard command.status == .completed, command.error == nil else {
            throw QwenTextRunnerError.gpuExecution(
                stage: "source.sampler", detail: command.error?.localizedDescription
                    ?? "status \(command.status.rawValue)")
        }
        try await hooks.afterActualGPUCompletion("source.sampler")
        try Task.checkCancellation()
        let raw = scratch.outToken.contents().load(as: UInt32.self)
        guard let token = Int32(exactly: raw),
              Int(token) < model.architecture.vocabularySize else {
            throw QwenTextRunnerError.invalidState(detail: "source sampled token outside vocabulary")
        }
        let publicBits = Array(UnsafeBufferPointer(
            start: scratch.logits.contents().bindMemory(
                to: UInt16.self, capacity: model.architecture.vocabularySize),
            count: model.architecture.vocabularySize))
        hooks.observePublicLogitsAndSample(samplePosition, publicBits, token)
        return token
    }
}
