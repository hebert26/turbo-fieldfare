import Foundation
import Metal
import TurboFieldfare
import TurboFieldfareDecodeProtocol
import Synchronization

/// A bounded request-owned latch bridges service admission and producer startup.
/// Finishing it synchronizes with any in-flight Stop before another request runs.
public final class AppGenerationStop: Sendable {
    private struct State {
        var requested = false
        var finished = false
        var stop: (@Sendable () -> Void)?
    }
    private let state = Mutex(State())

    public init() {}

    public func requestStop() {
        state.withLock { value in
            guard !value.finished, !value.requested else { return }
            value.requested = true
            value.stop?()
        }
    }

    public func activate(_ stop: @escaping @Sendable () -> Void) {
        state.withLock { value in
            guard !value.finished else { return }
            value.stop = stop
            if value.requested { stop() }
        }
    }

    public func finish() {
        state.withLock { value in
            value.finished = true
            value.stop = nil
        }
    }

    public var isRequested: Bool { state.withLock { $0.requested } }
}

final class GenerationTaskRegistry: Sendable {
    private struct Entry: Sendable {
        let id: UUID
        var task: Task<Void, Never>?
    }

    private let state = Mutex<Entry?>(nil)

    func reserve(_ id: UUID) -> Bool {
        state.withLock { entry in
            guard entry == nil else { return false }
            entry = Entry(id: id, task: nil)
            return true
        }
    }

    func attach(_ task: Task<Void, Never>, to id: UUID) {
        let shouldCancel = state.withLock { entry -> Bool in
            guard entry?.id == id else { return true }
            entry?.task = task
            return false
        }
        if shouldCancel { task.cancel() }
    }

    func take(_ id: UUID) -> Task<Void, Never>? {
        state.withLock { entry in
            guard entry?.id == id else { return nil }
            defer { entry = nil }
            return entry?.task
        }
    }

    func takeCurrent() -> Task<Void, Never>? {
        state.withLock { entry in
            defer { entry = nil }
            return entry?.task
        }
    }

    func clear(_ id: UUID) {
        state.withLock { entry in
            if entry?.id == id { entry = nil }
        }
    }

}

/// Real-model inference client for the Mac app. Wraps the same raw-completion
/// loop the CLI uses (`runRawCompletion`, BOS + verbatim encode, no chat
/// template) behind the `AppInferenceClient` event stream, with an explicit
/// load lifecycle so the resident weights stay warm across generations.
public final class RealInferenceClient: AppModelLifecycleClient, @unchecked Sendable {
    private let session: RealInferenceSession
    /// Bytes of image tower held mapped, readable without awaiting the session.
    public var currentVisionTowerBytes: UInt64? {
        session.towerBytes.withLock { $0 }
    }

    private let memorySampler: AppMemorySampler
    private let generationTasks = GenerationTaskRegistry()

    public init(memorySampler: AppMemorySampler = AppMemorySampler()) {
        self.memorySampler = memorySampler
        self.session = RealInferenceSession()
    }

    public func ensureLoaded(modelDirectory: URL,
                             maxContextTokens: Int,
                             options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        try await session.ensureLoaded(
            key: SessionLoadKey(directory: modelDirectory.standardizedFileURL,
                                maxContext: maxContextTokens,
                                options: options,
                                forceLogitsHead: forceLogitsHead),
            onState: onState)
    }

    public func unload() async {
        await session.unload()
    }

    public var loadedToolThinkingEnabled: Bool? {
        get async { await session.loadedToolThinkingEnabled }
    }

    /// Drops the KV so the next turn starts a fresh lineage. The model stays
    /// loaded: this ends a conversation, it does not unload ~1.6 GB.
    public func resetConversation() async {
        await session.resetConversation()
    }

    public func contextCheckpoint(_ request: DecodeContextCheckpointRequest,
                                  stop: AppGenerationStop? = nil) async throws
        -> DecodeContextCheckpointReceipt {
        try await session.contextCheckpoint(request, stop: stop)
    }

    /// In-process, so there is no stale-epoch window to guard: the caller is
    /// the only writer. The epoch is accepted and ignored; the decode service
    /// holds the gate that uses it.
    public func resetConversation(epoch: UUID) async throws {
        await session.resetConversation()
    }

    /// Whether a conversation is still open on the session.
    ///
    /// Exists so a test can assert that `unload()` released it. It cannot be
    /// inferred from the outside, and not releasing it means unload freed
    /// nothing at all — the conversation holds the model, the runner, the
    /// scratch and the tower.
    public var hasOpenConversation: Bool {
        get async { await session.hasConversation }
    }

    /// Tokens the open conversation's KV holds, or zero when none is open.
    public var conversationTokenCount: Int {
        get async { await session.currentConversationTokens }
    }

    /// The same figure without awaiting the session, for the decode service's
    /// writer thread.
    public var currentConversationTokens: Int {
        session.conversationTokens.withLock { $0 }
    }

    public func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        generate(request, measurementCapture: nil)
    }

    /// The service owns this collector and drains it through its existing writer.
    /// Ordinary callers never allocate one or attach request measurements.
    public func generate(
        _ request: AppGenerationRequest, measurementCapture: RuntimeMeasurementCapture?,
        generationStop: AppGenerationStop? = nil
    ) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream<AppInferenceEvent, Error> { continuation in
            let generationID = UUID()
            guard generationTasks.reserve(generationID) else {
                continuation.yield(.failed(.generationInFlight, partial: nil))
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task { [self] in
                defer { generationStop?.finish() }
                let activateStop: (@Sendable () -> Void)?
                if let generationStop {
                    let stop: @Sendable () -> Void = { [weak self] in
                        guard let self else { return }
                        self.stop()
                    }
                    activateStop = { generationStop.activate(stop) }
                } else {
                    activateStop = nil
                }
                await session.run(request: request,
                                  measurementCapture: measurementCapture,
                                  activateStop: activateStop,
                                  memorySampler: memorySampler,
                                  continuation: continuation)
                generationTasks.clear(generationID)
            }
            generationTasks.attach(task, to: generationID)

            continuation.onTermination = { [generationTasks] termination in
                let task = generationTasks.take(generationID)
                if case .cancelled = termination { task?.cancel() }
            }
        }
    }

    public func cancel() {
        generationTasks.takeCurrent()?.cancel()
    }

    /// Ends the turn at the next token boundary and keeps what it produced.
    ///
    /// Only decode has token boundaries. A stop during prefill or an image
    /// encode has nothing to stop at, so it cancels instead and the
    /// conversation rewinds the turn — otherwise Stop did nothing at all until
    /// the first token, which on a long prompt is a long time.
    public func stop() {
        session.stopRequested.withLock { $0 = true }
        guard session.decodeBegan.withLock({ $0 }) else {
            generationTasks.takeCurrent()?.cancel()
            return
        }
    }

}

struct SessionLoadKey: Equatable, Sendable {
    var directory: URL
    var maxContext: Int
    var options: AppRuntimeOptions
    var forceLogitsHead: Bool

    init(directory: URL,
         maxContext: Int,
         options: AppRuntimeOptions,
         forceLogitsHead: Bool = false) {
        self.directory = directory.standardizedFileURL
        self.maxContext = maxContext
        self.options = options
        self.forceLogitsHead = forceLogitsHead
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.directory == rhs.directory
            && lhs.maxContext == rhs.maxContext
            && lhs.loadTimeOptions == rhs.loadTimeOptions
            && lhs.forceLogitsHead == rhs.forceLogitsHead
    }

    /// Prefill is selected for each request. The runner allocates for the
    /// largest supported chunk, so neither prefill field changes loaded state.
    private var loadTimeOptions: AppRuntimeOptions {
        var value = options
        value.prefillEnabled = false
        value.prefillChunkTokens = 128
        return value
    }
}

struct TokenizerDirectoryCache: Equatable, Sendable {
    private(set) var directory: URL?

    func shouldReload(for modelDirectory: URL) -> Bool {
        directory != modelDirectory.standardizedFileURL
    }

    mutating func markLoaded(for modelDirectory: URL) {
        directory = modelDirectory.standardizedFileURL
    }

    mutating func clear() {
        directory = nil
    }
}

/// Owns the loaded model and serializes load / unload / generate. All Metal
/// command-buffer waits happen inside this actor, off the main actor; one
/// cooperative-pool thread is occupied for the duration of a generation,
/// which is acceptable for the app's single session. The 8 GB rule lives
/// here: a reload releases the loaded model, runner, and scratch before constructing
/// replacements, so two models are never alive at once.
actor RealInferenceSession {
    private var loadedKey: SessionLoadKey?
    private var ctx: MetalContext?
    private var tokenizer: GFTokenizer?
    private var tokenizerDirectoryCache = TokenizerDirectoryCache()
    private var runner: RealForwardRunner?
    private var scratch: RawCompletionScratch?
    private var model: Model?
    /// The open conversation, when the app is in chat mode. Nil for the
    /// one-shot path, and dropped by any load, unload, or explicit reset.
    private var conversation: MultimodalConversation?
    /// Immutable identity/provenance in the same order as retained projected features.
    private var conversationImageProvenance: [String] = []
    /// Bytes of image tower held mapped right now, published outside the actor
    /// so a reader does not have to await it mid-decode.
    nonisolated let towerBytes = Mutex<UInt64?>(nil)
    /// Tokens the open conversation's KV holds, published outside the actor for
    /// the same reason `towerBytes` is: the decode service's writer thread
    /// stamps it on the turn's terminal event and cannot await this actor
    /// mid-decode.
    nonisolated let conversationTokens = Mutex<Int>(0)
    /// Set by Stop, read by the decode loop at each token boundary.
    ///
    /// Cancelling the task instead throws out of `runRawCompletion`'s loop
    /// before its `shouldStop` is consulted, and the conversation then rewinds
    /// the whole turn — measured: a stop after 13 tokens left the KV at zero.
    /// Stopping cooperatively keeps what was produced and lets the next turn
    /// continue from it.
    nonisolated let stopRequested = Mutex<Bool>(false)
    /// Whether the run has reached decode. Before that there are no token
    /// boundaries to stop at, and half a prefilled prompt is not a turn worth
    /// keeping — so a stop there has to cancel and let the conversation rewind.
    nonisolated let decodeBegan = Mutex<Bool>(false)

    private var visionRuntime: VisionRuntime? {
        didSet { publishTowerBytes() }
    }
    private var visionRuntimeError: Error?

    /// Called after anything that maps or releases tower regions.
    private func publishTowerBytes() {
        let value = visionRuntime.map { UInt64($0.retainedWeightBytes) }
        towerBytes.withLock { $0 = value }
    }

    /// A reader the decode loop can call from wherever it runs. The `Mutex` is
    /// a non-copyable stored property, so it is reached through `self` rather
    /// than captured.
    private nonisolated func stopFlagReader() -> @Sendable () -> Bool {
        { [weak self] in self?.stopRequested.withLock { $0 } ?? false }
    }

    func resetConversation() async {
        // Same hazard as `unload()`: `runner.reset()` under a live decode either
        // aborts that turn mid-stream or lets two turns drive one runner.
        if let conversation { await conversation.invalidate() }
        conversation = nil
        conversationImageProvenance.removeAll()
        runner?.reset()
        conversationTokens.withLock { $0 = 0 }
    }

    var hasConversation: Bool { conversation != nil }

    func contextCheckpoint(_ request: DecodeContextCheckpointRequest,
                           stop: AppGenerationStop?) async throws -> DecodeContextCheckpointReceipt {
        guard let conversation else { throw AppInferenceError.modelNotLoaded }
        let start = ContinuousClock.now
        let attachments = (request.result.imageAttachments ?? []).map {
            AppImageAttachment(id: $0.id, fileURL: URL(fileURLWithPath: $0.path),
                displayName: $0.displayName, encodedBytes: $0.encodedBytes, sha256: $0.sha256)
        }
        let check: @Sendable () throws -> Void = {
            try Task.checkCancellation()
            if stop?.isRequested == true { throw CancellationError() }
        }
        for attachment in attachments {
            try check()
            guard try Sha256Verifier.hashFile(at: attachment.fileURL, chunkBytes: 256 * 1_024) == attachment.sha256 else {
                throw AppInferenceError.invalidRequest("Checkpoint image changed after observation.")
            }
        }
        let addedProvenance = Self.imageProvenance(attachments, source: "tool result \(request.result.callID)")
        let arguments = try JSONDecoder().decode(JSONValue.self, from: Data(request.pendingCall.argumentsJSON.utf8))
        let receipt = try await conversation.checkpoint(id: request.checkpointID,
            pendingCall: GFTokenizer.HistoricalToolCall(id: request.pendingCall.id,
                name: request.pendingCall.name, arguments: arguments),
            result: ConversationToolResult(callID: request.result.callID, name: request.result.name,
                content: request.result.content, images: attachments.map(\.fileURL)),
            record: request.record, imageProvenance: conversationImageProvenance + addedProvenance,
            generationAllowance: request.generationAllowance,
            finalAnswerAllowance: request.finalAnswerAllowance, permitsScreenshot: request.permitsScreenshot,
            force: request.force,
            performanceRequested: request.trigger == .sustainedSlowDecode,
            commit: request.commit, checkCancellation: check)
        if request.commit {
            conversationImageProvenance.append(contentsOf: addedProvenance)
            conversationTokens.withLock { $0 = 0 }
        }
        publishTowerBytes()
        let duration = start.duration(to: .now).components
        return DecodeContextCheckpointReceipt(checkpointID: request.checkpointID,
            replacementEpoch: request.replacementEpoch, committed: request.commit,
            needed: receipt.needed, existingPromptTokens: receipt.existingPromptTokens,
            replacementPromptTokens: receipt.replacementPromptTokens, reserveTokens: receipt.reserveTokens,
            resultAllowanceTokens: receipt.resultAllowanceTokens,
            retainedImageCount: receipt.retainedImageCount, retainedImageRows: receipt.retainedImageRows,
            retainedFeatureBytes: receipt.retainedFeatureBytes,
            performanceMinimumSavingsTokens: receipt.performanceMinimumSavingsTokens,
            preparationSeconds: Double(duration.seconds) + Double(duration.attoseconds) / 1e18)
    }

    private static func imageProvenance(_ attachments: [AppImageAttachment], source: String) -> [String] {
        attachments.map {
            "Source: \(source). Attachment \($0.id.uuidString), SHA-256 \($0.sha256), \($0.displayName). Historical observation only, never an executable target."
        }
    }

    var loadedToolThinkingEnabled: Bool? {
        loadedKey == nil ? nil : tokenizer?.enableToolThinking
    }

    var currentConversationTokens: Int {
        get async {
            let count = await conversation?.kvTokenCount ?? 0
            conversationTokens.withLock { $0 = count }
            return count
        }
    }

    func ensureLoaded(key: SessionLoadKey,
                      onState: @Sendable (AppModelLoadState) -> Void) async throws {
        if loadedKey == key, runner != nil { return }

        conversation = nil
        conversationImageProvenance.removeAll()
        conversationTokens.withLock { $0 = 0 }
        runner = nil
        scratch = nil
        model = nil
        loadedKey = nil
        visionRuntime = nil
        visionRuntimeError = nil

        let start = Date()
        do {
            onState(.loading(.validatingDirectory))
            let manifest = key.directory.appendingPathComponent("manifest.json")
            guard FileManager.default.fileExists(atPath: manifest.path) else {
                throw AppInferenceError.modelNotFound(key.directory.path)
            }

            onState(.loading(.tokenizer))
            if tokenizer == nil || tokenizerDirectoryCache.shouldReload(for: key.directory) {
                do {
                    tokenizer = try await Self.loadTokenizer(for: key.directory)
                    tokenizerDirectoryCache.markLoaded(for: key.directory)
                } catch {
                    throw AppInferenceError.tokenizerUnavailable("\(error)")
                }
            }
            tokenizer = tokenizer?.withToolThinking(enabled: key.options.toolThinkingEnabled)
            try Task.checkCancellation()

            onState(.loading(.verifyingWeights))
            let runtimeConfiguration = try key.options.resolvedRuntimeConfiguration(
                forceLogitsHead: key.forceLogitsHead)
            let context: MetalContext
            if let ctx {
                context = ctx
            } else {
                context = try MetalContext()
                ctx = context
            }
            let loadedModel = try Model.load(
                directoryURL: key.directory,
                device: context.device,
                streamingMode: .pread(slotCount: runtimeConfiguration.expertCacheSlots),
                expertCachePolicy: runtimeConfiguration.modelExpertCachePolicy,
                integrityPolicy: key.options.modelVerification.runtimeValue)
            try Task.checkCancellation()

            onState(.loading(.preparingRunner))
            let loadedRunner = try RealForwardRunner(
                model: loadedModel,
                context: context,
                maxContext: key.maxContext,
                runtimeConfiguration: runtimeConfiguration)
            let loadedScratch = try RawCompletionScratch(context: context,
                                                         vocab: loadedModel.config.vocabSize)
            try Task.checkCancellation()

            let loadedVisionRuntime: VisionRuntime?
            let loadedVisionRuntimeError: Error?
            do {
                let runtime = try VisionRuntime.open(
                    textModelURL: key.directory,
                    context: context)
                // Keep Ready maps the tower during the load rather than on the
                // first image, so the wait is where the user asked for it.
                if key.options.visionResidencyPolicy == .keepReady {
                    onState(.loading(.mappingImageTower))
                    try runtime.prewarmWeightRegions()
                }
                loadedVisionRuntime = runtime
                loadedVisionRuntimeError = nil
            } catch {
                // A missing or invalid pack only means images are unavailable;
                // the reason travels to the first image turn.
                loadedVisionRuntime = nil
                loadedVisionRuntimeError = error
            }
            try Task.checkCancellation()

            runner = loadedRunner
            scratch = loadedScratch
            model = loadedModel
            loadedKey = key
            visionRuntime = loadedVisionRuntime
            visionRuntimeError = loadedVisionRuntimeError
            onState(.ready(modelDirectory: key.directory,
                           loadSeconds: Date().timeIntervalSince(start)))
        } catch is CancellationError {
            throw CancellationError()
        } catch let appError as AppInferenceError {
            onState(.failed(appError))
            throw appError
        } catch {
            let appError = AppInferenceError.modelLoadFailed("\(error)")
            onState(.failed(appError))
            throw appError
        }
    }

    private static func loadTokenizer(for modelDirectory: URL) async throws -> GFTokenizer {
        try await GFTokenizer.load(forModelDirectory: modelDirectory)
    }

    static func forceLogitsHead(for request: AppGenerationRequest) -> Bool {
        !request.isPureGreedy
    }

    static func generationConfig(for request: AppGenerationRequest,
                                 maxNewTokens: Int? = nil) -> GenerationConfig {
        GenerationConfig(maxNewTokens: maxNewTokens ?? request.maxNewTokens,
                         temperature: request.temperature,
                         topK: request.topK,
                         topP: request.topP,
                         repetitionPenalty: request.repetitionPenalty)
    }

    static func effectiveMaxNewTokens(requested: Int,
                                      promptTokenCount: Int,
                                      maxContext: Int) -> Int {
        min(requested, max(0, maxContext - promptTokenCount))
    }

    func unload() async {
        // Awaited, not just dropped. `MultimodalConversation` holds the model,
        // runner, scratch and vision runtime as strong `let`s, and a turn in
        // flight holds its own reference — so nilling this while a decode runs
        // frees nothing and the next load builds a second set beside the live
        // one. `invalidate()` is what waits for that decode.
        if let conversation { await conversation.invalidate() }
        conversation = nil
        conversationImageProvenance.removeAll()
        conversationTokens.withLock { $0 = 0 }
        visionRuntime = nil
        visionRuntimeError = nil
        model = nil
        runner = nil
        scratch = nil
        tokenizer = nil
        tokenizerDirectoryCache.clear()
        loadedKey = nil
    }

    /// The single-prompt path: reset the KV and prefill the whole rendered
    /// prompt. Unchanged behaviour, moved out of `run` so the conversational
    /// path sits beside it rather than inside it.
    private func runOneShot(
        request: AppGenerationRequest,
        runner: RealForwardRunner,
        tokenizer: GFTokenizer,
        ctx: MetalContext,
        scratch: RawCompletionScratch,
        prefillConfig: PrefillRuntimeConfig,
        progress: ProgressState,
        memorySampler: AppMemorySampler,
        report: @escaping @Sendable (RawDecodeProgress) -> Void
    ) async throws -> TurnOutcome {
            let promptIds: [Int32]
            let multimodalInput: MultimodalPrefillInput?
            if request.imageAttachments.isEmpty {
                let renderedPrompt = try tokenizer.applyChatTemplate([
                    GFTokenizer.Message(role: .user, content: request.prompt)
                ])
                promptIds = tokenizer.encode(renderedPrompt, addBOS: false)
                multimodalInput = nil
            } else {
                guard let visionRuntime, let model else {
                    throw AppInferenceError.invalidRequest(
                        "Image support is unavailable: "
                            + (visionRuntimeError.map(String.init(describing:))
                                ?? "the image companion pack is not installed"))
                }
                var features: [UUID: VisionFeatures] = [:]
                features.reserveCapacity(request.imageAttachments.count)
                for attachment in request.imageAttachments {
                    try Task.checkCancellation()
                    // The file may have changed between selection and send.
                    let actualDigest = try Sha256Verifier.hashFile(
                        at: attachment.fileURL, chunkBytes: 256 * 1_024)
                    guard actualDigest == attachment.sha256 else {
                        throw AppInferenceError.invalidRequest(
                            "Image \(attachment.displayName) changed after selection.")
                    }
                    defer { publishTowerBytes() }
                    features[attachment.id] = try visionRuntime.encodeImage(
                        at: attachment.fileURL,
                        languageModel: model,
                        residencyPolicy: request.runtimeOptions.visionResidencyPolicy,
                        checkCancellation: { try Task.checkCancellation() })
                }
                var content = request.imageAttachments.map {
                    MultimodalContentPart.image(id: $0.id)
                }
                if !request.prompt.isEmpty { content.append(.text(request.prompt)) }
                let input = try MultimodalPromptRenderer.render(
                    messages: [MultimodalMessage(role: .user, content: content)],
                    featuresByID: features,
                    tokenizer: tokenizer)
                promptIds = input.effectiveTokenIDs
                multimodalInput = input
            }
            progress.promptTokenCount = promptIds.count
            guard promptIds.count < runner.maxContext else {
                throw AppInferenceError.contextOverflow(prompt: promptIds.count,
                                                        maxNew: request.maxNewTokens,
                                                        maxContext: runner.maxContext)
            }
            memorySampler.resetPeak()
            _ = memorySampler.sample()
            let config = Self.generationConfig(
                for: request,
                maxNewTokens: Self.effectiveMaxNewTokens(
                    requested: request.maxNewTokens,
                    promptTokenCount: promptIds.count,
                    maxContext: runner.maxContext))
            runner.reset()
            progress.prefillStart = Date()


        let result = try await runRawCompletion(
            producer: runner, tokenizer: tokenizer, promptIds: promptIds,
            multimodalInput: multimodalInput,
            config: config, context: ctx, scratch: scratch,
            prefillConfig: prefillConfig,
            shouldStop: stopFlagReader(),
            onProgress: report)
        return TurnOutcome(
            reason: result.reason, prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds, newTokens: result.newTokens,
            computedPrefillTokens: result.computedPrefillTokens)
    }

    /// What both turn paths report back, so the terminal diagnostics do not
    /// have to know which one ran.
    struct TurnOutcome {
        let reason: StopReason
        let prefillSeconds: Double
        let decodeSeconds: Double
        let newTokens: Int
        /// Nil on the single-prompt path: nothing was retained to reuse.
        var cachedTokens: Int?
        var computedPrefillTokens: Int?
        var conversationTokens: Int?
        var toolCalls: [ParsedToolCall] = []
    }

    /// The open conversation, or a new one on the same runner.
    ///
    /// Built lazily rather than at load: a load that is never followed by a
    /// conversational turn should not reset the runner, and the residency
    /// policy belongs to the turn that asks for it.
    private func conversationForTurn(
        _ request: AppGenerationRequest
    ) throws -> MultimodalConversation {
        if let conversation { return conversation }
        guard let model, let ctx, let tokenizer, let runner, let scratch else {
            throw AppInferenceError.modelLoadFailed("session lost its loaded state")
        }
        // A conversation starts from an empty KV. The runner may be carrying a
        // one-shot generation's tokens, and resuming a first turn onto those
        // would put a prompt the user never sent in front of this one.
        runner.reset()
        // The same waiver `TurboFieldfareModelSession` takes, and it holds for
        // the same reason: `Model` and `VisionRuntime` are not Sendable because
        // two conversations encoding at once would race the tower's per-encode
        // scratch. What prevents that here is this session's own invariant —
        // one conversation, and one generation at a time behind
        // `GenerationTaskRegistry` and the decode service's serial command loop
        // — not the type system. Anything that lets two turns run at once has
        // to make `VisionRuntime` an actor first.
        nonisolated(unsafe) let sharedModel = model
        nonisolated(unsafe) let sharedVision = visionRuntime
        let created = MultimodalConversation(
            model: sharedModel, context: ctx, tokenizer: tokenizer, runner: runner,
            scratch: scratch, visionRuntime: sharedVision,
            visionRuntimeError: visionRuntimeError,
            visionResidency: request.runtimeOptions.visionResidencyPolicy,
            maxContext: runner.maxContext)
        conversation = created
        return created
    }

    private func runConversationTurn(
        request: AppGenerationRequest,
        prefillConfig: PrefillRuntimeConfig,
        progress: ProgressState,
        memorySampler: AppMemorySampler,
        report: @escaping @Sendable (RawDecodeProgress) -> Void
    ) async throws -> TurnOutcome {
        let conversation = try conversationForTurn(request)
        var parts: [MultimodalContinuationPart] = []
        var imageURLs: [URL] = []
        for attachment in request.imageAttachments {
            try Task.checkCancellation()
            // Same check the one-shot path makes: the file was hashed when it
            // was staged, and a file rewritten since then is a different image
            // than the one the user attached.
            let actualDigest = try Sha256Verifier.hashFile(
                at: attachment.fileURL, chunkBytes: 256 * 1_024)
            guard actualDigest == attachment.sha256 else {
                throw AppInferenceError.invalidRequest(
                    "Image \(attachment.displayName) changed after selection.")
            }
            parts.append(.image)
            imageURLs.append(attachment.fileURL)
        }
        if !request.prompt.isEmpty { parts.append(.text(request.prompt)) }

        memorySampler.resetPeak()
        _ = memorySampler.sample()
        progress.prefillStart = Date()
        defer { publishTowerBytes() }
        do {
            // `maxNewTokens` is clamped inside the conversation against what the
            // KV has room for, so it is passed through unmodified here.
            let config = Self.generationConfig(
                for: request, maxNewTokens: request.maxNewTokens)
            // This is the immutable mode configured on the loaded tokenizer.
            let invisibleTokenLimit: Int? = loadedToolThinkingEnabled == true
                ? nil : VisionCaptureAgentProfile.maximumConsecutiveInvisibleTokens
            let turn: MultimodalTurnResult
            let toolCalls: [ParsedToolCall]
            switch request.toolTurn {
            case .checkpoint(let id):
                let completion = try await conversation.resumeCheckpoint(id: id,
                    config: config, prefillConfig: prefillConfig,
                    checkCancellation: { try Task.checkCancellation() }, shouldStop: stopFlagReader(),
                    captureToolFailureEvidence: request.captureToolFailureEvidence,
                    maximumConsecutiveInvisibleTokens: invisibleTokenLimit, captureThoughtPreview: true,
                    detectThoughtRepetition: true,
                    onProgress: report,
                    onStructuredProgress: { progress.updateStructuredProgress($0) })
                turn = completion.turn
                toolCalls = completion.toolCalls
            case .user(let developerPrompt, let tools):
                let completion = try await conversation.sendToolUser(
                    parts: parts, images: imageURLs,
                    developerPrompt: developerPrompt,
                    tools: tools.map(\.tokenizerDefinition),
                    config: config,
                    prefillConfig: prefillConfig,
                    checkCancellation: { try Task.checkCancellation() },
                    shouldStop: stopFlagReader(),
                    acceptsUnknownToolNames: true,
                    captureToolFailureEvidence: request.captureToolFailureEvidence,
                    maximumConsecutiveInvisibleTokens: invisibleTokenLimit,
                    captureThoughtPreview: true,
                    detectThoughtRepetition: true,
                    onProgress: report,
                    onStructuredProgress: { progress.updateStructuredProgress($0) })
                turn = completion.turn
                toolCalls = completion.toolCalls
            case .results(let results):
                let completion = try await conversation.sendToolResults(
                    results.map {
                        ConversationToolResult(
                            callID: $0.callID,
                            name: $0.name,
                            content: $0.content,
                            images: $0.imageAttachments.map(\.fileURL))
                    },
                    config: config,
                    prefillConfig: prefillConfig,
                    checkCancellation: { try Task.checkCancellation() },
                    shouldStop: stopFlagReader(),
                    acceptsUnknownToolNames: true,
                    captureToolFailureEvidence: request.captureToolFailureEvidence,
                    maximumConsecutiveInvisibleTokens: invisibleTokenLimit,
                    captureThoughtPreview: true,
                    detectThoughtRepetition: true,
                    onProgress: report,
                    onStructuredProgress: { progress.updateStructuredProgress($0) })
                turn = completion.turn
                toolCalls = completion.toolCalls
            case nil:
                turn = try await conversation.send(
                    parts: parts, images: imageURLs,
                    config: config,
                    prefillConfig: prefillConfig,
                    checkCancellation: { try Task.checkCancellation() },
                    shouldStop: stopFlagReader(),
                    onProgress: report)
                toolCalls = []
            }
            if !request.imageAttachments.isEmpty {
                let source: String
                if case .results(let results) = request.toolTurn {
                    source = "tool result " + results.map(\.callID).joined(separator: ", ")
                } else { source = "original user message" }
                conversationImageProvenance.append(contentsOf: Self.imageProvenance(request.imageAttachments, source: source))
            }
            progress.promptTokenCount = turn.promptTokens
            // The conversation's own count. `promptTokens + completionTokens`
            // is one too many whenever a run stops on max tokens or is
            // cancelled: that final token is held outside the KV for the next
            // turn to replay.
            conversationTokens.withLock { $0 = turn.kvTokens }
            return TurnOutcome(
                reason: turn.reason, prefillSeconds: turn.prefillSeconds,
                decodeSeconds: turn.decodeSeconds, newTokens: turn.completionTokens,
                cachedTokens: turn.cachedTokens,
                computedPrefillTokens: turn.computedPrefillTokens,
                conversationTokens: turn.kvTokens,
                toolCalls: toolCalls)
        } catch let error as MultimodalConversationError {
            // Mapped rather than flattened: a lineage that broke can only be
            // cleared, an exhausted context is the user's to act on, and an
            // unavailable image names the pack that failed to open. Reporting
            // all three as one generic failure is how a chat becomes
            // undiagnosable.
            switch error {
            case .lineageBroken, .lineageRecoveryFailed:
                throw AppInferenceError.conversationLineageLost("\(error)")
            case .contextExhausted(let prompt, let maxContext):
                throw AppInferenceError.contextOverflow(
                    prompt: prompt, maxNew: request.maxNewTokens,
                    maxContext: maxContext)
            case .imageUnavailable:
                throw AppInferenceError.invalidRequest("\(error)")
            case .noObservableToolProgress(let limit):
                throw AppInferenceError.invalidRequest(
                    "Agent Mode reached its invisible-output token limit (\(limit) tokens) "
                        + "without visible answer text or a completed tool call. "
                        + "No new VisionCapture request was dispatched from this model step.")
            case .closed, .busy, .emptyTurn,
                    .toolModeRequiresNewConversation, .invalidToolContinuation:
                throw AppInferenceError.unknown("\(error)")
            }
        }
    }

    func run(request: AppGenerationRequest,
             measurementCapture: RuntimeMeasurementCapture? = nil,
             activateStop: (@Sendable () -> Void)? = nil,
             memorySampler: AppMemorySampler,
             continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation) async {
        // Also covers an unexpected early exit. Normal terminal paths detach
        // before publishing completion so the writer cannot freeze the collector
        // before the final cache snapshot has been appended.
        defer { clearMeasurementCapture(measurementCapture) }
        // Cleared here, before either branch and before anything a stop could
        // race. Resetting them inside the conversational branch left the
        // single-prompt path with a flag nothing ever lowered — after one Stop
        // every later generation returned exactly one token — and left a stale
        // `decodeBegan` that swallowed a Stop pressed while the next turn was
        // still hashing its images.
        stopRequested.withLock { $0 = false }
        decodeBegan.withLock { $0 = false }
        activateStop?()
        var prefillConfig = request.runtimeOptions.prefillConfig
        // Image spans only run under chunked prefill, and the app's prefill
        // toggle can select `.off`. Coerce rather than fail after the encodes:
        // whether images work is not a performance preference.
        if !request.imageAttachments.isEmpty,
           let coerced = prefillConfig.coercedForImagePrompt() {
            prefillConfig = coerced
        }
        let progress = ProgressState()
        do {
            try request.validate()
            let requestKey = SessionLoadKey(
                directory: request.modelDirectory.standardizedFileURL,
                maxContext: request.maxContextTokens,
                options: request.runtimeOptions,
                forceLogitsHead: Self.forceLogitsHead(for: request))
            guard let loadedKey else { throw AppInferenceError.modelNotLoaded }
            guard loadedKey == requestKey else { throw AppInferenceError.reloadRequired }
            guard let runner, let tokenizer, let ctx, let scratch else {
                throw AppInferenceError.modelLoadFailed("session lost its loaded state")
            }
            runner.captureGPUCompletionTiming = request.captureGPUCompletionTiming
            defer { runner.captureGPUCompletionTiming = false }
            if let measurementCapture {
                model?.measurementCapture = measurementCapture
                model?.recordMeasurementCacheSnapshot(
                    reason: 0, position: request.continuesConversation ? request.conversationTokens : -1)
            }
            let executedPrefillMode: PrefillExecutedMode =
                prefillConfig.mode == .chunked ? .chunked : .off
            let prefillDiagnostics = PrefillExecutionDiagnostics(config: prefillConfig,
                                                                 executedMode: executedPrefillMode,
                                                                 kvStorageMode: .fp16)

            let report: @Sendable (RawDecodeProgress) -> Void = { event in
                switch event {
                case .prefill(let done, let total):
                    if done == total {
                        // From here there are token boundaries to stop at.
                        self.decodeBegan.withLock { $0 = true }
                        progress.decodeStart = Date()
                        progress.countersAtDecodeStart = RunnerCounterSnapshot(runner)
                    }
                    continuation.yield(.prefillProgress(done: done, total: total))
                case .token(let index, _, let delta):
                    if progress.firstTokenDate == nil { progress.firstTokenDate = Date() }
                    progress.generated = index + 1
                    if index % 8 == 0 { _ = memorySampler.sample() }
                    continuation.yield(.token(AppTokenEvent(
                        index: index,
                        textDelta: delta,
                        elapsedDecodeSeconds: progress.elapsedDecodeSeconds,
                        structuredProgress: progress.structuredProgress,
                        thinkingPreview: progress.thinkingPreview,
                        toolCallPreview: progress.toolCallPreview)))
                case .tail(let text):
                    continuation.yield(.token(AppTokenEvent(
                        index: max(progress.generated - 1, 0),
                        textDelta: text,
                        elapsedDecodeSeconds: progress.elapsedDecodeSeconds,
                        structuredProgress: progress.structuredProgress,
                        thinkingPreview: progress.thinkingPreview,
                        toolCallPreview: progress.toolCallPreview)))
                }
            }

            let outcome: TurnOutcome
            if request.continuesConversation {
                outcome = try await runConversationTurn(
                    request: request, prefillConfig: prefillConfig,
                    progress: progress, memorySampler: memorySampler,
                    report: report)
            } else {
                outcome = try await runOneShot(
                    request: request, runner: runner, tokenizer: tokenizer,
                    ctx: ctx, scratch: scratch, prefillConfig: prefillConfig,
                    progress: progress, memorySampler: memorySampler,
                    report: report)
            }
            let result = outcome

            for call in result.toolCalls {
                continuation.yield(.toolCall(AppToolCall(
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments)))
            }

            let diagnostics = makeDiagnostics(request: request,
                                              memorySampler: memorySampler,
                                              progress: progress,
                                              stopReason: Self.stopReason(result.reason),
                                              prefillSeconds: result.prefillSeconds,
                                              decodeSeconds: result.decodeSeconds,
                                              generated: result.newTokens,
                                              cachedTokens: result.cachedTokens,
                                              computedPrefillTokens: result.computedPrefillTokens,
                                              conversationTokens: result.conversationTokens,
                                              prefill: prefillDiagnostics)
            clearMeasurementCapture(measurementCapture, position: result.conversationTokens ?? -1)
            continuation.yield(.finished(diagnostics))
            continuation.finish()
        } catch is CancellationError {
            clearMeasurementCapture(measurementCapture)
            if progress.generated > 0,
               progress.thinkingPreview != nil || progress.toolCallPreview != nil {
                continuation.yield(.token(AppTokenEvent(
                    index: progress.generated - 1, textDelta: "",
                    elapsedDecodeSeconds: progress.elapsedDecodeSeconds,
                    structuredProgress: progress.structuredProgress,
                    thinkingPreview: progress.thinkingPreview,
                    toolCallPreview: progress.toolCallPreview)))
            }
            let diagnostics = makeDiagnostics(request: request,
                                              memorySampler: memorySampler,
                                              progress: progress,
                                              stopReason: .cancelled,
                                              prefillSeconds: progress.elapsedPrefillSeconds,
                                              decodeSeconds: progress.elapsedDecodeSeconds,
                                              generated: progress.generated,
                                              prefill: PrefillExecutionDiagnostics(
                                                config: prefillConfig,
                                                executedMode: prefillConfig.mode == .chunked
                                                    ? PrefillExecutedMode.chunked : .off,
                                                kvStorageMode: .fp16))
            continuation.yield(.cancelled(diagnostics))
            continuation.finish(throwing: AppInferenceError.cancelled)
        } catch let prefillError as PrefillError {
            clearMeasurementCapture(measurementCapture)
            let diagnostics = Self.prefillFailureDiagnostics(config: prefillConfig,
                                                             kvStorageMode: .fp16,
                                                             reason: prefillError.description)
            failGeneration(.unknown(prefillError.description),
                           request: request,
                           memorySampler: memorySampler,
                           progress: progress,
                           continuation: continuation,
                           prefill: diagnostics,
                           forcePartialDiagnostics: true)
        } catch let recovery as ThoughtRepetitionRecovery {
            clearMeasurementCapture(measurementCapture)
            failGeneration(.repeatedThought(recovery),
                           request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        } catch let failure as StructuredToolFailure {
            clearMeasurementCapture(measurementCapture)
            failGeneration(.structuredToolFailure(
                message: failure.description,
                canRegenerateToolResult: failure.canRegenerateToolResult,
                evidence: failure.evidence),
                request: request, memorySampler: memorySampler,
                progress: progress, continuation: continuation)
        } catch let appError as AppInferenceError {
            clearMeasurementCapture(measurementCapture)
            failGeneration(appError, request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        } catch {
            clearMeasurementCapture(measurementCapture)
            failGeneration(.unknown("\(error)"), request: request, memorySampler: memorySampler,
                           progress: progress, continuation: continuation)
        }
    }

    private func clearMeasurementCapture(
        _ capture: RuntimeMeasurementCapture?, position: Int = -1
    ) {
        guard let capture, let model, model.measurementCapture === capture else { return }
        model.recordMeasurementCacheSnapshot(reason: 6, position: position)
        model.measurementCapture = nil
    }

    private func failGeneration(_ error: AppInferenceError,
                                request: AppGenerationRequest,
                                memorySampler: AppMemorySampler,
                                progress: ProgressState,
                                continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
                                prefill: PrefillExecutionDiagnostics? = nil,
                                forcePartialDiagnostics: Bool = false) {
        // A parsing error can follow held-back thought bytes on a control token.
        // Flush their display window without counting another sampled token.
        if progress.generated > 0,
           progress.thinkingPreview != nil || progress.toolCallPreview != nil {
            continuation.yield(.token(AppTokenEvent(
                index: progress.generated - 1, textDelta: "",
                elapsedDecodeSeconds: progress.elapsedDecodeSeconds,
                structuredProgress: progress.structuredProgress,
                thinkingPreview: progress.thinkingPreview,
                toolCallPreview: progress.toolCallPreview)))
        }
        let partial = progress.generated > 0 || forcePartialDiagnostics
            ? makeDiagnostics(request: request, memorySampler: memorySampler,
                              progress: progress, stopReason: .failed,
                              prefillSeconds: progress.elapsedPrefillSeconds,
                              decodeSeconds: progress.elapsedDecodeSeconds,
                              generated: progress.generated,
                              prefill: prefill)
            : nil
        continuation.yield(.failed(error, partial: partial))
        continuation.finish(throwing: error)
    }

    private func makeDiagnostics(request: AppGenerationRequest,
                                 memorySampler: AppMemorySampler,
                                 progress: ProgressState,
                                 stopReason: AppStopReason,
                                 prefillSeconds: Double? = nil,
                                 decodeSeconds: Double,
                                 generated: Int,
                                 cachedTokens: Int? = nil,
                                 computedPrefillTokens: Int? = nil,
                                 conversationTokens: Int? = nil,
                                 prefill: PrefillExecutionDiagnostics? = nil) -> AppDiagnostics {
        _ = memorySampler.sample()
        let ttft: Double?
        if let first = progress.firstTokenDate, let start = progress.decodeStart {
            ttft = first.timeIntervalSince(start)
        } else {
            ttft = nil
        }
        // A failure can follow a completed forward or interrupt one. Generated
        // tokens do not establish its forward count, so omit normalized buckets.
        let runnerTiming = stopReason == .failed
            ? nil : runnerDiagnostics(progress: progress, generated: generated)
        return AppDiagnostics(
            generatedTokens: generated,
            stopReason: stopReason,
            promptTokenCount: progress.promptTokenCount,
            cachedPromptTokens: cachedTokens,
            computedPrefillTokens: computedPrefillTokens,
            conversationTokens: conversationTokens,
            prefillSeconds: prefillSeconds,
            timeToFirstTokenSeconds: ttft,
            decodeSeconds: decodeSeconds,
            tokensPerSecond: decodeSeconds > 0 ? Double(generated) / decodeSeconds : 0,
            peakMemoryBytes: memorySampler.peakBytes,
            visionTowerMappedBytes: visionRuntime.map { UInt64($0.retainedWeightBytes) },
            runtimeOptions: request.runtimeOptions,
            prefill: prefill,
            runner: runnerTiming,
            structuredProgress: progress.structuredProgress)
    }

    /// Per-token buckets as diffs of the runner's cumulative counters from the
    /// decode start (excludes prefill), divided by the decode forward count.
    /// The forward count is `generated - 1`: each loop iteration that continues
    /// ends with one `produce`; the final sampled token never runs a forward.
    private func runnerDiagnostics(progress: ProgressState, generated: Int) -> AppRunnerDiagnostics? {
        guard let runner, let base = progress.countersAtDecodeStart, generated > 1 else { return nil }
        let now = RunnerCounterSnapshot(runner)
        let forwards = Double(generated - 1)
        func ms(_ end: UInt64, _ start: UInt64) -> Double {
            Double(end &- start) / 1_000_000 / forwards
        }
        func gpu(_ end: RealForwardRunner.GPUCompletionTiming,
                 _ start: RealForwardRunner.GPUCompletionTiming) -> DecodeGPUCompletionTiming {
            let valid = end.validCount &- start.validCount
            let expected = end.expectedCount &- start.expectedCount
            let elapsed = end.seconds - start.seconds
            let complete = expected > 0 && valid == expected && elapsed.isFinite && elapsed >= 0
            return DecodeGPUCompletionTiming(
                millisecondsPerForward: complete ? elapsed * 1_000 / forwards : nil,
                validCount: valid, expectedCount: expected)
        }
        let gpuTiming: [String: DecodeGPUCompletionTiming]? = base.gpuCompletionEnabled ? [
            "attention_router": gpu(now.attentionRouterGPU, base.attentionRouterGPU),
            "full_attention_router": gpu(now.fullAttentionRouterGPU, base.fullAttentionRouterGPU),
            "sliding_attention_router": gpu(now.slidingAttentionRouterGPU, base.slidingAttentionRouterGPU),
            "shared_experts": gpu(now.sharedExpertsGPU, base.sharedExpertsGPU),
            "routed_experts": gpu(now.routedExpertsGPU, base.routedExpertsGPU),
        ] : nil
        return AppRunnerDiagnostics(
            cb1MillisecondsPerToken: ms(now.cb1, base.cb1),
            routerWaitMillisecondsPerToken: ms(now.routerWait, base.routerWait),
            gpuCompletionTiming: gpuTiming,
            ioMillisecondsPerToken: ms(now.io, base.io),
            cb2MillisecondsPerToken: ms(now.cb2, base.cb2),
            headMillisecondsPerToken: ms(now.head, base.head),
            rdadviseMillisecondsPerToken: ms(now.rdadvise, base.rdadvise),
            rdadviseCallsPerToken: Double(now.rdadviseCalls &- base.rdadviseCalls) / forwards,
            rdadviseMegabytesPerToken: Double(now.rdadviseBytes &- base.rdadviseBytes) / 1_048_576.0 / forwards,
            rdadviseSkippedPerToken: Double(now.rdadviseSkipped &- base.rdadviseSkipped) / forwards,
            rdadviseFailures: now.rdadviseFailures &- base.rdadviseFailures)
    }

    private static func stopReason(_ reason: StopReason) -> AppStopReason {
        switch reason {
        case .eos: return .eos
        case .endOfTurn: return .endOfTurn
        case .maxTokens: return .maxTokens
        case .stopString: return .stopString
        case .cancelled: return .cancelled
        case .toolCalls: return .toolCalls
        }
    }

    internal static func prefillFailureDiagnostics(config: PrefillRuntimeConfig,
                                                   kvStorageMode: PrefillKVStorageMode,
                                                   reason: String) -> PrefillExecutionDiagnostics {
        PrefillExecutionDiagnostics.unsupported(config: config,
                                                kvStorageMode: kvStorageMode,
                                                reason: reason)
    }
}

/// Mutable per-generation state shared between the progress callback and the
/// surrounding actor method. Single-threaded: the callback runs synchronously
/// inside `runRawCompletion` on the session actor's task.
private final class ProgressState: @unchecked Sendable {
    var structuredProgress: DecodeStructuredProgress?
    var thinkingPreview: DecodeThinkingPreview?
    var toolCallPreview: DecodeToolCallPreview?

    func updateStructuredProgress(_ value: StructuredAssistantProgress) {
        structuredProgress = DecodeStructuredProgress(value)
        thinkingPreview = value.recentThoughtText.isEmpty ? nil : DecodeThinkingPreview(
            text: value.recentThoughtText, earlierTextOmitted: value.earlierThoughtTextOmitted)
        toolCallPreview = value.toolCallPreview.map {
            DecodeToolCallPreview(text: $0.text, middleTextOmitted: $0.middleTextOmitted)
        }
    }
    var generated = 0
    var promptTokenCount: Int?
    var prefillStart: Date?
    var decodeStart: Date?
    var firstTokenDate: Date?
    var countersAtDecodeStart: RunnerCounterSnapshot?

    var elapsedDecodeSeconds: Double {
        guard let decodeStart else { return 0 }
        return Date().timeIntervalSince(decodeStart)
    }

    var elapsedPrefillSeconds: Double? {
        guard let prefillStart else { return nil }
        let end = decodeStart ?? Date()
        return max(end.timeIntervalSince(prefillStart), 0)
    }
}

private struct RunnerCounterSnapshot {
    let gpuCompletionEnabled: Bool
    let attentionRouterGPU: RealForwardRunner.GPUCompletionTiming
    let fullAttentionRouterGPU: RealForwardRunner.GPUCompletionTiming
    let slidingAttentionRouterGPU: RealForwardRunner.GPUCompletionTiming
    let sharedExpertsGPU: RealForwardRunner.GPUCompletionTiming
    let routedExpertsGPU: RealForwardRunner.GPUCompletionTiming
    let cb1: UInt64
    let routerWait: UInt64
    let io: UInt64
    let cb2: UInt64
    let head: UInt64
    let rdadvise: UInt64
    let rdadviseCalls: UInt64
    let rdadviseBytes: UInt64
    let rdadviseFailures: UInt64
    let rdadviseSkipped: UInt64

    init(_ runner: RealForwardRunner) {
        gpuCompletionEnabled = runner.captureGPUCompletionTiming
        attentionRouterGPU = runner.attentionRouterGPUCompletion
        fullAttentionRouterGPU = runner.fullAttentionRouterGPUCompletion
        slidingAttentionRouterGPU = runner.slidingAttentionRouterGPUCompletion
        sharedExpertsGPU = runner.sharedExpertsGPUCompletion
        routedExpertsGPU = runner.routedExpertsGPUCompletion
        cb1 = runner.totalCb1Nanos
        routerWait = runner.totalRouterWaitNanos
        io = runner.totalIoNanos
        cb2 = runner.totalCb2Nanos
        head = runner.totalHeadNanos &+ runner.totalHeadFusedNanos
        rdadvise = runner.totalRDAdviseNanos
        rdadviseCalls = runner.totalRDAdviseCalls
        rdadviseBytes = runner.totalRDAdviseBytes
        rdadviseFailures = runner.totalRDAdviseFailures
        rdadviseSkipped = runner.totalRDAdviseSkipped
    }
}
