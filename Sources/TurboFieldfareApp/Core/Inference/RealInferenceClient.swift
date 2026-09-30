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
    private let qwenProgressObserver: @Sendable (TokenizedConversationProgress) async -> Void
    private let generationTasks = GenerationTaskRegistry()

    public convenience init(memorySampler: AppMemorySampler = AppMemorySampler()) {
        self.init(session: RealInferenceSession(), memorySampler: memorySampler)
    }

    init(
        session: RealInferenceSession,
        memorySampler: AppMemorySampler = AppMemorySampler(),
        qwenProgressObserver: @escaping @Sendable (TokenizedConversationProgress) async -> Void = { _ in }
    ) {
        self.memorySampler = memorySampler
        self.qwenProgressObserver = qwenProgressObserver
        self.session = session
    }

    public func ensureLoaded(modelDirectory: URL,
                             maxContextTokens: Int,
                             options: AppRuntimeOptions,
                             forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        try await session.ensureLoaded(
            key: SessionLoadKey(directory: modelDirectory,
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

    public var loadedModelReadiness: AppLoadedModelReadiness? {
        get async { await session.loadedModelReadiness }
    }

    /// Called by the service at its ready-publication boundary. The only
    /// authority is the runtime's retained source model and trust receipt;
    /// matching wire metadata alone does not establish source availability.
    public func validateSourceReadiness(_ identity: DecodeSourceIdentity) async throws {
        try await session.validateSourceReadiness(identity)
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
        try await session.resetConversationThrowing()
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

    /// Committed Qwen conversation state bytes. This is a logical element-byte
    /// count, never process RSS or allocated Array capacity.
    public var currentConversationLogicalStateBytes: UInt64? {
        session.conversationLogicalStateBytes.withLock { $0 }
    }

    /// Bytes allocated by the loaded family's routed-expert caches. Zero is a
    /// valid loaded value; nil means there is no installed model to report.
    public var currentExpertCacheBytes: UInt64? {
        session.expertCacheBytes.withLock { $0 }
    }

    func generatePreparedQwen(
        _ request: TokenizedConversationTurn,
        generationStop: AppGenerationStop? = nil
    ) -> AsyncThrowingStream<TokenizedConversationEvent, Error> {
        AsyncThrowingStream<TokenizedConversationEvent, Error> { continuation in
            let generationID = UUID()
            guard generationTasks.reserve(generationID) else {
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task { [self] in
                defer { generationStop?.finish() }
                let activateStop: (@Sendable () -> Void)?
                if let generationStop {
                    activateStop = { [weak self] in
                        generationStop.activate { [weak self] in self?.stop() }
                    }
                } else {
                    activateStop = nil
                }
                do {
                    let result = try await session.applyQwenTokenizedTurn(
                        request,
                        activateStop: activateStop,
                        onProgress: {
                            await self.qwenProgressObserver($0)
                            continuation.yield(.progress($0))
                        })
                    continuation.yield(.finished(result))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                generationTasks.clear(generationID)
            }
            generationTasks.attach(task, to: generationID)
            continuation.onTermination = { [generationTasks] termination in
                let task = generationTasks.take(generationID)
                if case .cancelled = termination { task?.cancel() }
            }
        }
    }

    func rebuildPreparedQwenCheckpoint(
        _ request: TokenizedCheckpointRequest,
        generationStop: AppGenerationStop? = nil
    ) -> AsyncThrowingStream<TokenizedCheckpointEvent, Error> {
        AsyncThrowingStream<TokenizedCheckpointEvent, Error> { continuation in
            let generationID = UUID()
            guard generationTasks.reserve(generationID) else {
                continuation.finish(throwing: AppInferenceError.generationInFlight)
                return
            }
            let task = Task { [self] in
                defer { generationStop?.finish() }
                let activateStop: (@Sendable () -> Void)?
                if let generationStop {
                    activateStop = { [weak self] in
                        generationStop.activate { [weak self] in self?.stop() }
                    }
                } else {
                    activateStop = nil
                }
                do {
                    let result = try await session.rebuildQwenTokenCheckpoint(
                        request,
                        activateStop: activateStop,
                        onProgress: {
                            await self.qwenProgressObserver($0)
                            continuation.yield(.progress($0))
                        })
                    continuation.yield(.finished(result))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                generationTasks.clear(generationID)
            }
            generationTasks.attach(task, to: generationID)
            continuation.onTermination = { [generationTasks] termination in
                let task = generationTasks.take(generationID)
                if case .cancelled = termination { task?.cancel() }
            }
        }
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
        let marker = directory.appendingPathComponent("official-source.json")
        self.directory = FileManager.default.fileExists(atPath: marker.path)
            ? AppModelLocation.qwenRegistrationResolved(logicalURL: directory).textModelURL
            : directory.standardizedFileURL
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

enum RealInferenceLifecycleError: Error, Equatable {
    case lifecycleInProgress
}

struct RealInferenceLifecycleCheckpoints: Sendable {
    var afterRetirement: @Sendable () async -> Void
    var afterPreparation: @Sendable () async -> Void

    init(
        afterRetirement: @escaping @Sendable () async -> Void = {},
        afterPreparation: @escaping @Sendable () async -> Void = {}
    ) {
        self.afterRetirement = afterRetirement
        self.afterPreparation = afterPreparation
    }
}

private final class RealInferenceLifecycleCompletion: Sendable {
    private struct State: Sendable {
        var finished = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state = Mutex(State())

    func wait() async {
        await withCheckedContinuation { continuation in
            let finishNow = state.withLock { value in
                if value.finished { return true }
                value.waiters.append(continuation)
                return false
            }
            if finishNow { continuation.resume() }
        }
    }

    func finish() {
        let waiters = state.withLock { value -> [CheckedContinuation<Void, Never>] in
            guard !value.finished else { return [] }
            value.finished = true
            defer { value.waiters.removeAll() }
            return value.waiters
        }
        waiters.forEach { $0.resume() }
    }
}

/// Owns the loaded model and serializes load / unload / generate. All Metal
/// command-buffer waits happen inside this actor, off the main actor. A
/// lifecycle transition retains every session-owned old/prepared holder until
/// its real GPU drain completes, and a second load is refused before it can
/// detach or construct anything.
actor RealInferenceSession {
    private enum LoadedFamily: @unchecked Sendable {
        case gemma
        case qwen(QwenTextModel)
        case qwenSource(QwenOfficialSourceModel)
    }

    private enum QwenTurnGenerator: Sendable {
        case packed(QwenConversationGenerationSession)
        case source(QwenOfficialSourceConversationGenerationSession)

        var supportsCheckpoint: Bool {
            if case .packed = self { return true }
            return false
        }

        func reset() async throws {
            switch self {
            case .packed(let session): try await session.reset()
            case .source(let session): try await session.reset()
            }
        }

        func generate(
            _ request: QwenConversationGenerationRequest,
            shouldStop: @escaping @Sendable () -> Bool,
            onEvent: @escaping @Sendable (QwenConversationGenerationEvent) -> Void
        ) async throws -> QwenConversationGenerationResult {
            switch self {
            case .packed(let session):
                return try await session.generate(
                    request, shouldStop: shouldStop, onEvent: onEvent)
            case .source(let session):
                return try await session.generate(
                    request, shouldStop: shouldStop, onEvent: onEvent)
            }
        }
    }

    private final class RetiredInstallation {
        var family: LoadedFamily?
        var identity: LoadedRuntimeIdentity?
        var codec: QwenChatCodec?
        var qwenGeneration: QwenConversationGenerationSession?
        var sourceGeneration: QwenOfficialSourceConversationGenerationSession?
        var runner: RealForwardRunner?
        var scratch: RawCompletionScratch?
        var model: Model?
        var conversation: MultimodalConversation?
        var visionRuntime: VisionRuntime?
        var visionError: Error?

        init(family: LoadedFamily?, identity: LoadedRuntimeIdentity?, codec: QwenChatCodec?,
             qwenGeneration: QwenConversationGenerationSession?,
             sourceGeneration: QwenOfficialSourceConversationGenerationSession?,
             runner: RealForwardRunner?, scratch: RawCompletionScratch?, model: Model?,
             conversation: MultimodalConversation?, visionRuntime: VisionRuntime?,
             visionError: Error?) {
            self.family = family
            self.identity = identity
            self.codec = codec
            self.qwenGeneration = qwenGeneration
            self.sourceGeneration = sourceGeneration
            self.runner = runner
            self.scratch = scratch
            self.model = model
            self.conversation = conversation
            self.visionRuntime = visionRuntime
            self.visionError = visionError
        }

        var hasResources: Bool {
            family != nil || identity != nil || codec != nil || qwenGeneration != nil
                || sourceGeneration != nil
                || runner != nil
                || scratch != nil || model != nil || conversation != nil
                || visionRuntime != nil || visionError != nil
        }

        func releaseAll() {
            family = nil
            identity = nil
            codec = nil
            qwenGeneration = nil
            sourceGeneration = nil
            runner = nil
            scratch = nil
            model = nil
            conversation = nil
            visionRuntime = nil
            visionError = nil
        }
    }

    private final class PreparedInstallation {
        let key: SessionLoadKey
        let context: MetalContext
        let family: LoadedFamily
        let identity: LoadedRuntimeIdentity?
        let codec: QwenChatCodec?
        let qwenGeneration: QwenConversationGenerationSession?
        let sourceGeneration: QwenOfficialSourceConversationGenerationSession?
        let sourceIdentity: DecodeSourceIdentity?
        let tokenizer: GFTokenizer?
        let runner: RealForwardRunner?
        let scratch: RawCompletionScratch?
        let model: Model?
        let conversation: MultimodalConversation?
        let visionRuntime: VisionRuntime?
        let visionError: Error?
        let logicalStateBytes: UInt64?
        let expertCacheBytes: UInt64

        init(key: SessionLoadKey, context: MetalContext, family: LoadedFamily,
             identity: LoadedRuntimeIdentity?, codec: QwenChatCodec?,
             qwenGeneration: QwenConversationGenerationSession?, tokenizer: GFTokenizer?,
             sourceGeneration: QwenOfficialSourceConversationGenerationSession? = nil,
             sourceIdentity: DecodeSourceIdentity? = nil,
             runner: RealForwardRunner?, scratch: RawCompletionScratch?, model: Model?,
             conversation: MultimodalConversation?, visionRuntime: VisionRuntime?,
             visionError: Error?, logicalStateBytes: UInt64?, expertCacheBytes: UInt64) {
            self.key = key
            self.context = context
            self.family = family
            self.identity = identity
            self.codec = codec
            self.qwenGeneration = qwenGeneration
            self.sourceGeneration = sourceGeneration
            self.tokenizer = tokenizer
            self.sourceIdentity = sourceIdentity
            self.runner = runner
            self.scratch = scratch
            self.model = model
            self.conversation = conversation
            self.visionRuntime = visionRuntime
            self.visionError = visionError
            self.logicalStateBytes = logicalStateBytes
            self.expertCacheBytes = expertCacheBytes
        }
    }

    private final class LifecycleTransition {
        let id = UUID()
        let completion = RealInferenceLifecycleCompletion()
        var retired: RetiredInstallation?
        var prepared: PreparedInstallation?
        var teardownRequested = false
    }

    private let lifecycleCheckpoints: RealInferenceLifecycleCheckpoints
    private var activeLifecycle: LifecycleTransition?
    private var lifecycleUnloadWaiters = 0
    private var loadedKey: SessionLoadKey?
    private var loadedFamily: LoadedFamily?
    private var verifiedIdentity: LoadedRuntimeIdentity?
    private var qwenCodec: QwenChatCodec?
    private var qwenGeneration: QwenConversationGenerationSession?
    private var sourceGeneration: QwenOfficialSourceConversationGenerationSession?
    private var loadedSourceIdentity: DecodeSourceIdentity?
    private var qwenOrdinaryGenerationInFlight = false
    private var qwenOrdinaryGenerationWaiters: [CheckedContinuation<Void, Never>] = []
    private var qwenTools: [ModelChatToolDefinition] = []
    private var qwenSystemPrompt: String?
    private var qwenPendingToolCalls: [ParsedToolCall] = []
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
    /// Logical committed Qwen state bytes, distinct from process RSS.
    nonisolated let conversationLogicalStateBytes = Mutex<UInt64?>(nil)
    /// Actual allocated routed-expert cache bytes for the installed family.
    nonisolated let expertCacheBytes = Mutex<UInt64?>(nil)
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
    private var conversationLifecycleError: Error?

    init(lifecycleCheckpoints: RealInferenceLifecycleCheckpoints = .init()) {
        self.lifecycleCheckpoints = lifecycleCheckpoints
    }

    /// Called after anything that maps or releases tower regions.
    private func publishTowerBytes() {
        let value = visionRuntime.map { UInt64($0.retainedWeightBytes) }
        towerBytes.withLock { $0 = value }
    }

    private func publishGemmaExpertCacheBytes() {
        guard case .gemma = loadedFamily, let model else { return }
        expertCacheBytes.withLock { $0 = model.routedExpertCacheAllocatedBytes }
    }

    /// A reader the decode loop can call from wherever it runs. The `Mutex` is
    /// a non-copyable stored property, so it is reached through `self` rather
    /// than captured.
    private nonisolated func stopFlagReader() -> @Sendable () -> Bool {
        { [weak self] in self?.stopRequested.withLock { $0 } ?? false }
    }

    /// Ordinary Qwen generation runs in a separate actor over the same state as
    /// the prepared-token boundary. RealInferenceSession is reentrant while it
    /// awaits that actor, so retirement needs its own barrier before it releases
    /// the old installation or permits another model mapping.
    private func beginQwenOrdinaryGeneration() throws
        -> QwenConversationGenerationSession {
        guard !qwenOrdinaryGenerationInFlight else {
            throw AppInferenceError.generationInFlight
        }
        guard let qwenGeneration else { throw AppInferenceError.modelNotLoaded }
        qwenOrdinaryGenerationInFlight = true
        return qwenGeneration
    }

    private func beginSourceOrdinaryGeneration() throws
        -> QwenOfficialSourceConversationGenerationSession {
        guard !qwenOrdinaryGenerationInFlight else {
            throw AppInferenceError.generationInFlight
        }
        guard let sourceGeneration else { throw AppInferenceError.modelNotLoaded }
        qwenOrdinaryGenerationInFlight = true
        return sourceGeneration
    }

    private func finishQwenOrdinaryGeneration() {
        guard qwenOrdinaryGenerationInFlight else { return }
        qwenOrdinaryGenerationInFlight = false
        let waiters = qwenOrdinaryGenerationWaiters
        qwenOrdinaryGenerationWaiters.removeAll(keepingCapacity: true)
        for waiter in waiters { waiter.resume() }
    }

    private func waitForQwenOrdinaryGeneration() async {
        guard qwenOrdinaryGenerationInFlight else { return }
        await withCheckedContinuation { continuation in
            qwenOrdinaryGenerationWaiters.append(continuation)
        }
    }

    private func publishQwenConversationStatus() async {
        let status: ConversationStateStatus?
        if let sourceGeneration { status = await sourceGeneration.status() }
        else { status = try? await conversation?.conversationStateStatus() }
        conversationTokens.withLock {
            $0 = status?.committed.retainedTokenIDs.count ?? 0
        }
        conversationLogicalStateBytes.withLock {
            $0 = status?.committed.logicalStateBytes
        }
    }

    func resetConversation() async {
        do {
            try await resetConversationThrowing()
        } catch {
            // The legacy no-throw entry cannot surface a Qwen restore failure.
            // Preserve it so every later operation fails explicitly instead of
            // treating an unknown lineage as reusable.
            conversationLifecycleError = error
            conversationLogicalStateBytes.withLock { $0 = nil }
        }
    }

    func resetConversationThrowing() async throws {
        switch loadedFamily {
        case .qwenSource:
            await waitForQwenOrdinaryGeneration()
            let generation = try beginSourceOrdinaryGeneration()
            defer { finishQwenOrdinaryGeneration() }
            try await generation.reset()
            qwenTools.removeAll(keepingCapacity: true)
            qwenSystemPrompt = nil
            qwenPendingToolCalls.removeAll(keepingCapacity: true)
            await publishQwenConversationStatus()
        case .qwen:
            await waitForQwenOrdinaryGeneration()
            if let qwenGeneration {
                try await qwenGeneration.reset()
            } else {
                guard let conversation else { throw AppInferenceError.modelNotLoaded }
                try await conversation.reset()
            }
            qwenTools.removeAll(keepingCapacity: true)
            qwenSystemPrompt = nil
            qwenPendingToolCalls.removeAll(keepingCapacity: true)
            await publishQwenConversationStatus()
        case .gemma:
            // Same hazard as `unload()`: reset only after the conversation has
            // settled its in-flight decode.
            if let conversation { await conversation.invalidate() }
            conversation = nil
            conversationImageProvenance.removeAll()
            runner?.reset()
            conversationTokens.withLock { $0 = 0 }
            conversationLogicalStateBytes.withLock { $0 = nil }
        case nil:
            conversationTokens.withLock { $0 = 0 }
            conversationLogicalStateBytes.withLock { $0 = nil }
        }
        conversationLifecycleError = nil
    }

    var hasConversation: Bool { conversation != nil || sourceGeneration != nil }

    var lifecycleOwnedInstallationCount: Int {
        var count = loadedFamily != nil || conversation != nil ? 1 : 0
        if activeLifecycle?.retired?.hasResources == true { count += 1 }
        if activeLifecycle?.prepared != nil { count += 1 }
        return count
    }

    var lifecycleTeardownIsRequested: Bool {
        activeLifecycle?.teardownRequested == true
    }

    var lifecycleUnloadWaiterCount: Int { lifecycleUnloadWaiters }

    func contextCheckpoint(_ request: DecodeContextCheckpointRequest,
                           stop: AppGenerationStop?) async throws -> DecodeContextCheckpointReceipt {
        if case .qwenSource = loadedFamily {
            throw ConversationStateTransactionError.unsupportedFamily
        }
        if case .qwen = loadedFamily {
            guard qwenGeneration != nil else {
                throw AppInferenceError.invalidRequest(
                    "Qwen string/tool checkpoints require the Phase 14 chat codec; use the prepared-token checkpoint boundary.")
            }
            return try await qwenContextCheckpoint(request, stop: stop)
        }
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
            conversationImageProvenance = addedProvenance
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

    private func qwenContextCheckpoint(
        _ request: DecodeContextCheckpointRequest,
        stop: AppGenerationStop?
    ) async throws -> DecodeContextCheckpointReceipt {
        guard let loadedKey, let qwenCodec else {
            throw AppInferenceError.modelNotLoaded
        }
        if let conversationLifecycleError { throw conversationLifecycleError }
        let generation = try beginQwenOrdinaryGeneration()
        defer { finishQwenOrdinaryGeneration() }
        let started = ContinuousClock.now
        let check: @Sendable () throws -> Void = {
            try Task.checkCancellation()
            if stop?.isRequested == true { throw CancellationError() }
        }
        try check()

        let attachments = (request.result.imageAttachments ?? []).map {
            AppImageAttachment(
                id: $0.id, fileURL: URL(fileURLWithPath: $0.path),
                displayName: $0.displayName, encodedBytes: $0.encodedBytes,
                sha256: $0.sha256)
        }
        let imagesByID = try Self.qwenImagesByID(attachments)
        let pendingArguments = try JSONDecoder().decode(
            JSONValue.self, from: Data(request.pendingCall.argumentsJSON.utf8))
        let suppliedArgumentsJSON = try pendingArguments.encoded()
        guard qwenPendingToolCalls.count == 1,
              let pending = qwenPendingToolCalls.first,
              pending.id == request.pendingCall.id,
              pending.name == request.pendingCall.name,
              try pending.arguments.encoded() == suppliedArgumentsJSON,
              request.result.callID == pending.id,
              request.result.name == pending.name else {
            throw AppInferenceError.invalidRequest(
                "Checkpoint result does not match the pending Qwen tool call.")
        }
        guard request.generationAllowance > 0, request.finalAnswerAllowance > 0,
              request.generationAllowance <= max(8_192, loadedKey.maxContext),
              request.finalAnswerAllowance <= max(2_048, loadedKey.maxContext) else {
            throw AppInferenceError.invalidRequest(
                "Checkpoint token allowances are invalid.")
        }
        guard !request.record.contains(MultimodalPromptRenderer.placeholder),
              !request.result.content.contains(MultimodalPromptRenderer.placeholder) else {
            throw AppInferenceError.invalidRequest(
                "Checkpoint text contains the reserved image marker.")
        }

        let resultMessage = Self.qwenToolResultMessages([
            AppToolResult(
                callID: request.result.callID,
                name: request.result.name,
                content: request.result.content,
                imageAttachments: attachments)
        ])[0]
        let thinking = loadedKey.options.toolThinkingEnabled
            ? ModelFamilyThinkingMode.enabled : .disabled
        let continuationTokens = try qwenCodec.encodeContinuation(
            messages: [resultMessage],
            options: .init(enableThinking: thinking != .disabled))
        let imagePreflight: QwenConversationImagePreflight
        if attachments.isEmpty {
            imagePreflight = QwenConversationImagePreflight(
                retainedImageCount: 0, retainedImageRows: 0,
                retainedFeatureBytes: 0)
        } else {
            do {
                imagePreflight = try await generation.preflightCheckpointImages(
                    orderedImageIDs: attachments.map { $0.id.uuidString },
                    imagesByID: imagesByID,
                    visionResidency: loadedKey.options.visionResidencyPolicy)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw Self.mapQwenGenerationError(error)
            }
        }
        let existingBase = conversationTokens.withLock { $0 }
        let existingCount = Self.qwenExpandedTokenCount(
            encodedCount: existingBase + continuationTokens.count,
            imageRows: imagePreflight.retainedImageRows,
            imageCount: imagePreflight.retainedImageCount)
        let generationReserve = max(
            min(request.generationAllowance, max(1, loadedKey.maxContext / 8)), 0)
        let resultReserve = max(min(1_024, max(1, loadedKey.maxContext / 64)),
                                continuationTokens.count * 2)
            + (request.permitsScreenshot ? VisionImageTokenBudget.maximumTokensPerImage : 0)
        let finalReserve = min(
            request.finalAnswerAllowance, max(1, loadedKey.maxContext / 32))
        let reserve = generationReserve + resultReserve + finalReserve
        let capacityNeeded = existingCount >= loadedKey.maxContext - reserve
        if request.record.isEmpty, !request.commit {
            let duration = started.duration(to: .now).components
            return DecodeContextCheckpointReceipt(
                checkpointID: request.checkpointID,
                replacementEpoch: request.replacementEpoch,
                committed: false,
                needed: request.force || capacityNeeded,
                existingPromptTokens: existingCount,
                replacementPromptTokens: nil,
                reserveTokens: reserve,
                resultAllowanceTokens: resultReserve,
                retainedImageCount: imagePreflight.retainedImageCount,
                retainedImageRows: imagePreflight.retainedImageRows,
                retainedFeatureBytes: imagePreflight.retainedFeatureBytes,
                performanceMinimumSavingsTokens: nil,
                preparationSeconds: Double(duration.seconds)
                    + Double(duration.attoseconds) / 1e18)
        }

        var checkpointParts: [ModelChatContentPart] = []
        if !attachments.isEmpty {
            checkpointParts.append(.text(
                "Screenshot evidence from the last completed action follows. "
                    + "It is observation evidence only, not an executable target."))
            for (index, attachment) in attachments.enumerated() {
                checkpointParts.append(.text(
                    "Image \(index + 1). Source: tool result \(request.result.callID). "
                        + "Observation evidence only."))
                checkpointParts.append(.image(.init(id: attachment.id.uuidString)))
            }
            checkpointParts.append(.text("Current checkpoint follows.\n\n" + request.record))
        }
        let checkpointUser = attachments.isEmpty
            ? ModelChatMessage(role: .user, content: request.record)
            : ModelChatMessage(role: .user, content: .parts(checkpointParts))
        var messages: [ModelChatMessage] = []
        if let qwenSystemPrompt {
            messages.append(ModelChatMessage(role: .system, content: qwenSystemPrompt))
        }
        messages.append(checkpointUser)

        let encodedReplacement = try qwenCodec.encodePrompt(
            messages: messages, tools: qwenTools,
            options: .init(
                enableThinking: thinking != .disabled,
                preserveThinking: true))
        let replacementEstimate = Self.qwenExpandedTokenCount(
            encodedCount: encodedReplacement.count,
            imageRows: imagePreflight.retainedImageRows,
            imageCount: imagePreflight.retainedImageCount)
        let performanceMinimumSavings = request.trigger == .sustainedSlowDecode
            ? max(4_096, existingCount / 5) : nil
        let performanceAccepted = performanceMinimumSavings.map {
            !request.record.isEmpty
                && existingCount - replacementEstimate >= $0
                && replacementEstimate < loadedKey.maxContext - reserve
        } ?? false
        let needed = request.force || capacityNeeded || performanceAccepted
        if request.commit {
            guard needed, !request.record.isEmpty,
                  replacementEstimate < existingCount,
                  replacementEstimate < loadedKey.maxContext - reserve else {
                throw AppInferenceError.contextOverflow(
                    prompt: replacementEstimate + reserve,
                    maxNew: 0, maxContext: loadedKey.maxContext)
            }
        }
        try check()

        let runtimeResult: QwenConversationCheckpointResult
        do {
            runtimeResult = try await generation.rebuildCheckpoint(
                QwenConversationCheckpointRequest(
                    checkpointID: request.checkpointID,
                    messages: messages,
                    tools: qwenTools,
                    imagesByID: imagesByID,
                    thinking: thinking,
                    visionResidency: loadedKey.options.visionResidencyPolicy,
                    reason: request.trigger == .sustainedSlowDecode
                        ? .sustainedSlowDecode : .capacity,
                    commit: request.commit),
                onEvent: { _ in })
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw Self.mapQwenGenerationError(error)
        }
        if request.commit {
            qwenPendingToolCalls.removeAll(keepingCapacity: true)
            conversationTokens.withLock {
                $0 = runtimeResult.metrics.retainedTokenIDs.count
            }
            conversationLogicalStateBytes.withLock {
                $0 = runtimeResult.metrics.logicalStateBytes
            }
        }
        let duration = started.duration(to: .now).components
        return DecodeContextCheckpointReceipt(
            checkpointID: request.checkpointID,
            replacementEpoch: request.replacementEpoch,
            committed: runtimeResult.committed,
            needed: needed,
            existingPromptTokens: existingCount,
            replacementPromptTokens: request.commit
                || request.trigger == .sustainedSlowDecode
                ? runtimeResult.promptTokens : nil,
            reserveTokens: reserve,
            resultAllowanceTokens: resultReserve,
            retainedImageCount: runtimeResult.retainedImageCount,
            retainedImageRows: runtimeResult.retainedImageRows,
            retainedFeatureBytes: runtimeResult.retainedFeatureBytes,
            performanceMinimumSavingsTokens: performanceMinimumSavings,
            preparationSeconds: Double(duration.seconds)
                + Double(duration.attoseconds) / 1e18)
    }

    private static func qwenExpandedTokenCount(
        encodedCount: Int,
        imageRows: Int,
        imageCount: Int
    ) -> Int {
        // The codec contributes one image-pad token inside each start/end
        // frame. Production expansion replaces that pad with `imageRows`, so
        // the effective sequence grows by rows minus one per image.
        let delta = max(imageRows - imageCount, 0)
        let (total, overflow) = encodedCount.addingReportingOverflow(delta)
        return overflow ? Int.max : total
    }

    private static func imageProvenance(_ attachments: [AppImageAttachment], source: String) -> [String] {
        attachments.map { _ in "Source: \(source). Observation evidence only." }
    }

    var loadedToolThinkingEnabled: Bool? {
        loadedKey == nil ? nil : tokenizer?.enableToolThinking
    }

    var loadedModelReadiness: AppLoadedModelReadiness? {
        guard loadedKey != nil else { return nil }
        switch loadedFamily {
        case .gemma:
            guard let toolThinkingEnabled = tokenizer?.enableToolThinking else { return nil }
            return .gemma(toolThinkingEnabled: toolThinkingEnabled)
        case .qwen:
            guard qwenCodec != nil, let verifiedIdentity else { return nil }
            return .qwen(identity: DecodeModelIdentity(runtimeIdentity: verifiedIdentity))
        case .qwenSource:
            guard qwenCodec != nil, sourceGeneration != nil,
                  let loadedSourceIdentity else { return nil }
            return .qwenSource(identity: loadedSourceIdentity)
        case nil:
            return nil
        }
    }

    func validateSourceReadiness(_ identity: DecodeSourceIdentity) throws {
        guard identity.kind == .officialSafetensorsBF16V1,
              activeLifecycle == nil, loadedKey != nil,
              case .qwenSource(let sourceModel)? = loadedFamily,
              sourceGeneration != nil, qwenCodec != nil,
              loadedSourceIdentity == identity else {
            throw AppInferenceError.modelLoadFailed(
                "loaded BF16 source no longer matches service readiness")
        }
        try sourceModel.revalidateLoadedSource(contentDigest: identity.contentDigest)
    }

    var currentConversationTokens: Int {
        get async {
            let count: Int
            if let sourceGeneration {
                let status = await sourceGeneration.status()
                count = status.committed.retainedTokenIDs.count
            } else {
                count = await conversation?.kvTokenCount ?? 0
            }
            conversationTokens.withLock { $0 = count }
            return count
        }
    }

    func applyQwenTokenizedTurn(
        _ turn: TokenizedConversationTurn,
        activateStop: (@Sendable () -> Void)?,
        onProgress: @escaping @Sendable (TokenizedConversationProgress) async -> Void
    ) async throws -> TokenizedConversationResult {
        guard case .qwen = loadedFamily else {
            throw ConversationStateTransactionError.unsupportedFamily
        }
        if let conversationLifecycleError { throw conversationLifecycleError }
        guard let conversation else { throw AppInferenceError.modelNotLoaded }
        stopRequested.withLock { $0 = false }
        decodeBegan.withLock { $0 = false }
        activateStop?()
        let result = try await conversation.applyTokenizedTurn(
            turn,
            checkCancellation: { try Task.checkCancellation() },
            shouldStop: stopFlagReader(),
            onProgress: { progress in
                if case .prefill(let done, let total) = progress, done == total {
                    self.decodeBegan.withLock { $0 = true }
                }
                await onProgress(progress)
            })
        qwenTools.removeAll(keepingCapacity: true)
        qwenSystemPrompt = nil
        qwenPendingToolCalls.removeAll(keepingCapacity: true)
        conversationTokens.withLock { $0 = result.metrics.retainedTokenIDs.count }
        conversationLogicalStateBytes.withLock { $0 = result.metrics.logicalStateBytes }
        return result
    }

    func rebuildQwenTokenCheckpoint(
        _ request: TokenizedCheckpointRequest,
        activateStop: (@Sendable () -> Void)?,
        onProgress: @escaping @Sendable (TokenizedConversationProgress) async -> Void
    ) async throws -> TokenizedCheckpointResult {
        guard case .qwen = loadedFamily else {
            throw ConversationStateTransactionError.unsupportedFamily
        }
        if let conversationLifecycleError { throw conversationLifecycleError }
        guard let conversation else { throw AppInferenceError.modelNotLoaded }
        stopRequested.withLock { $0 = false }
        decodeBegan.withLock { $0 = false }
        activateStop?()
        let result = try await conversation.rebuildTokenCheckpoint(
            request,
            checkCancellation: { try Task.checkCancellation() },
            onProgress: onProgress)
        qwenTools.removeAll(keepingCapacity: true)
        qwenSystemPrompt = nil
        qwenPendingToolCalls.removeAll(keepingCapacity: true)
        conversationTokens.withLock { $0 = result.metrics.retainedTokenIDs.count }
        conversationLogicalStateBytes.withLock { $0 = result.metrics.logicalStateBytes }
        return result
    }

    func installLoadedFamily(
        _ runtime: ModelFamilyRuntime,
        key: SessionLoadKey,
        context: MetalContext,
        onState: @Sendable (AppModelLoadState) -> Void = { _ in }
    ) async throws {
        try await performInstall(
            runtime: runtime, qwenCodec: nil, verifiedIdentity: nil,
            admittedBundle: nil, key: key, context: context, onState: onState)
    }

    func installLoadedFamily(
        _ bundle: LoadedModelFamilyBundle,
        key: SessionLoadKey,
        context: MetalContext,
        onState: @Sendable (AppModelLoadState) -> Void = { _ in }
    ) async throws {
        try Self.validate(bundle)
        try await performInstall(
            runtime: bundle.runtime, qwenCodec: bundle.qwenCodec,
            verifiedIdentity: bundle.verifiedIdentity,
            admittedBundle: bundle, key: key, context: context, onState: onState)
    }

    /// Test seam for AppCore's ordinary Qwen routing. The test constructs the
    /// runtime fixture facade with `@testable import TurboFieldfare`, then this
    /// installs that facade and the exact same state without a disk admission.
    /// Production loading always uses the verified bundle path above.
    func installQwenFixture(
        model: QwenTextModel,
        state: QwenConversationState,
        codec: QwenChatCodec,
        generation: QwenConversationGenerationSession,
        key: SessionLoadKey,
        context: MetalContext
    ) async throws {
        let transition = try beginLifecycleTransition()
        do {
            try await drainRetired(transition)
            try requireActive(transition)
            let conversation = MultimodalConversation(
                qwenState: state, maxContext: key.maxContext)
            let status = await state.status()
            let prepared = PreparedInstallation(
                key: key, context: context, family: .qwen(model),
                identity: nil, codec: codec, qwenGeneration: generation,
                tokenizer: nil, runner: nil, scratch: nil, model: nil,
                conversation: conversation, visionRuntime: nil,
                visionError: nil,
                logicalStateBytes: status.committed.logicalStateBytes,
                expertCacheBytes: await state.expertCacheAllocatedBytes)
            transition.prepared = prepared
            try installPrepared(prepared, transition: transition)
            finishInstall(transition)
        } catch {
            await cleanup(transition, explicitUnload: false)
            throw error
        }
    }

    private static func validate(_ bundle: LoadedModelFamilyBundle) throws {
        switch (bundle.family, bundle.runtime, bundle.verifiedIdentity,
                bundle.qwenCodec, bundle.sourceIdentity) {
        case (.gemma4, .gemma, nil, nil, nil):
            return
        case (.qwen3_6, .qwen, .some(let identity), .some(_), nil)
            where identity.family == .qwen3_6:
            return
        case (.qwen3_6, .qwenOfficialSource, nil, .some(_), .some(_)):
            return
        default:
            throw AppInferenceError.modelLoadFailed("loaded family bundle is inconsistent")
        }
    }

    private func beginLifecycleTransition() throws -> LifecycleTransition {
        guard activeLifecycle == nil else {
            throw RealInferenceLifecycleError.lifecycleInProgress
        }
        let transition = LifecycleTransition()
        transition.retired = RetiredInstallation(
            family: loadedFamily, identity: verifiedIdentity, codec: qwenCodec,
            qwenGeneration: qwenGeneration, sourceGeneration: sourceGeneration, runner: runner, scratch: scratch, model: model,
            conversation: conversation, visionRuntime: visionRuntime,
            visionError: visionRuntimeError)
        activeLifecycle = transition

        // Retire visibility synchronously. The transition remains the sole
        // session owner of old resources until invalidate and explicit release.
        conversation = nil
        conversationImageProvenance.removeAll()
        conversationTokens.withLock { $0 = 0 }
        conversationLogicalStateBytes.withLock { $0 = nil }
        expertCacheBytes.withLock { $0 = nil }
        loadedFamily = nil
        qwenCodec = nil
        qwenGeneration = nil
        sourceGeneration = nil
        loadedSourceIdentity = nil
        qwenTools.removeAll(keepingCapacity: true)
        qwenSystemPrompt = nil
        qwenPendingToolCalls.removeAll(keepingCapacity: true)
        verifiedIdentity = nil
        runner = nil
        scratch = nil
        model = nil
        loadedKey = nil
        visionRuntime = nil
        visionRuntimeError = nil
        conversationLifecycleError = nil
        return transition
    }

    private func requireActive(
        _ transition: LifecycleTransition,
        permitTeardown: Bool = false
    ) throws {
        guard activeLifecycle === transition else { throw CancellationError() }
        if !permitTeardown, transition.teardownRequested { throw CancellationError() }
        try Task.checkCancellation()
    }

    private func drainRetired(_ transition: LifecycleTransition) async throws {
        await waitForQwenOrdinaryGeneration()
        qwenTools.removeAll(keepingCapacity: true)
        qwenSystemPrompt = nil
        qwenPendingToolCalls.removeAll(keepingCapacity: true)
        conversationTokens.withLock { $0 = 0 }
        conversationLogicalStateBytes.withLock { $0 = nil }
        expertCacheBytes.withLock { $0 = nil }
        var capturedConversation = transition.retired?.conversation
        if let capturedConversation { await capturedConversation.invalidate() }
        capturedConversation = nil
        transition.retired?.releaseAll()
        transition.retired = nil
        await lifecycleCheckpoints.afterRetirement()
        try requireActive(transition)
    }

    private func prepare(
        runtime: ModelFamilyRuntime,
        qwenCodec admittedQwenCodec: QwenChatCodec?,
        verifiedIdentity admittedIdentity: LoadedRuntimeIdentity?,
        admittedBundle: LoadedModelFamilyBundle?,
        key: SessionLoadKey,
        context: MetalContext,
        transition: LifecycleTransition,
        onState: @Sendable (AppModelLoadState) -> Void
    ) async throws -> PreparedInstallation {
        let runtimeConfiguration = try key.options.resolvedRuntimeConfiguration(
            forceLogitsHead: key.forceLogitsHead)
        switch runtime {
        case .gemma(let loadedModel):
            try requireActive(transition)
            onState(.loading(.tokenizer))
            let loadedTokenizer: GFTokenizer
            if let tokenizer,
               !tokenizerDirectoryCache.shouldReload(for: key.directory) {
                loadedTokenizer = tokenizer.withToolThinking(
                    enabled: key.options.toolThinkingEnabled)
            } else {
                do {
                    loadedTokenizer = try await Self.loadTokenizer(for: key.directory)
                        .withToolThinking(enabled: key.options.toolThinkingEnabled)
                } catch {
                    throw AppInferenceError.tokenizerUnavailable("\(error)")
                }
                try requireActive(transition)
            }
            try requireActive(transition)
            onState(.loading(.preparingRunner))
            let loadedRunner = try RealForwardRunner(
                model: loadedModel, context: context,
                maxContext: key.maxContext,
                runtimeConfiguration: runtimeConfiguration)
            let loadedScratch = try RawCompletionScratch(
                context: context, vocab: loadedModel.config.vocabSize)
            let loadedVisionRuntime: VisionRuntime?
            let loadedVisionRuntimeError: Error?
            do {
                let runtime = try VisionRuntime.open(
                    textModelURL: key.directory, context: context)
                if key.options.visionResidencyPolicy == .keepReady {
                    try requireActive(transition)
                    onState(.loading(.mappingImageTower))
                    try runtime.prewarmWeightRegions()
                }
                loadedVisionRuntime = runtime
                loadedVisionRuntimeError = nil
            } catch {
                loadedVisionRuntime = nil
                loadedVisionRuntimeError = error
            }
            return PreparedInstallation(
                key: key, context: context, family: .gemma,
                identity: nil, codec: nil, qwenGeneration: nil, tokenizer: loadedTokenizer,
                runner: loadedRunner, scratch: loadedScratch, model: loadedModel,
                conversation: nil, visionRuntime: loadedVisionRuntime,
                visionError: loadedVisionRuntimeError, logicalStateBytes: nil,
                expertCacheBytes: loadedModel.routedExpertCacheAllocatedBytes)
        case .qwen(let qwenModel):
            try requireActive(transition)
            onState(.loading(.preparingRunner))
            let state = try await QwenConversationState(
                model: qwenModel, context: context, maxContext: key.maxContext)
            try requireActive(transition)
            let qwenConversation = MultimodalConversation(
                qwenState: state, maxContext: key.maxContext)
            let qwenGeneration: QwenConversationGenerationSession?
            switch (admittedIdentity, admittedQwenCodec) {
            case (.some(let identity), .some(let codec)):
                qwenGeneration = try QwenConversationGenerationSession(
                    model: qwenModel, state: state, codec: codec,
                    verifiedIdentity: identity, modelDirectoryURL: key.directory,
                    context: context, maxContext: key.maxContext)
            case (nil, nil):
                // Existing prepared-token fixture installs deliberately have no
                // admitted codec or disk identity. Keep that boundary available;
                // ordinary Qwen generation remains unavailable on such installs.
                qwenGeneration = nil
            default:
                throw AppInferenceError.modelLoadFailed(
                    "verified Qwen identity and chat codec are inconsistent")
            }
            let status = await state.status()
            try requireActive(transition)
            return PreparedInstallation(
                key: key, context: context, family: .qwen(qwenModel),
                identity: admittedIdentity, codec: admittedQwenCodec,
                qwenGeneration: qwenGeneration, tokenizer: nil, runner: nil,
                scratch: nil, model: nil, conversation: qwenConversation,
                visionRuntime: nil, visionError: nil,
                logicalStateBytes: status.committed.logicalStateBytes,
                expertCacheBytes: await state.expertCacheAllocatedBytes)
        case .qwenOfficialSource(let sourceModel):
            guard let admittedBundle,
                  case .qwenOfficialSource(let admittedModel) = admittedBundle.runtime,
                  admittedModel === sourceModel,
                  let sourceIdentity = admittedBundle.sourceIdentity,
                  admittedIdentity == nil, admittedQwenCodec != nil else {
                throw AppInferenceError.modelLoadFailed(
                    "source model requires its verified bundle and codec")
            }
            try requireActive(transition)
            onState(.loading(.preparingRunner))
            let sourceGeneration = try await ModelFamilyGenerationSession
                .makeSourceConversationSession(
                    admittedBundle: admittedBundle, proposedContext: context,
                    directoryURL: key.directory, maxContext: key.maxContext,
                    runtimeConfiguration: runtimeConfiguration)
            try requireActive(transition)
            let wireIdentity = try DecodeSourceIdentity(
                kind: .officialSafetensorsBF16V1,
                contentDigest: sourceIdentity.descriptorContentSHA256)
            let status = await sourceGeneration.status()
            return PreparedInstallation(
                key: key, context: ModelFamilyGenerationSession.contextForLoadedRuntime(
                    runtime, proposedContext: context), family: .qwenSource(sourceModel),
                identity: nil, codec: admittedQwenCodec, qwenGeneration: nil,
                tokenizer: nil, sourceGeneration: sourceGeneration,
                sourceIdentity: wireIdentity, runner: nil, scratch: nil,
                model: nil, conversation: nil, visionRuntime: nil,
                visionError: nil,
                logicalStateBytes: status.committed.logicalStateBytes,
                expertCacheBytes: 0)
        }
    }

    private func installPrepared(
        _ prepared: PreparedInstallation,
        transition: LifecycleTransition
    ) throws {
        try requireActive(transition)
        loadedKey = prepared.key
        loadedFamily = prepared.family
        verifiedIdentity = prepared.identity
        qwenCodec = prepared.codec
        qwenGeneration = prepared.qwenGeneration
        sourceGeneration = prepared.sourceGeneration
        loadedSourceIdentity = prepared.sourceIdentity
        qwenTools.removeAll(keepingCapacity: true)
        qwenSystemPrompt = nil
        qwenPendingToolCalls.removeAll(keepingCapacity: true)
        ctx = prepared.context
        tokenizer = prepared.tokenizer
        runner = prepared.runner
        scratch = prepared.scratch
        model = prepared.model
        conversation = prepared.conversation
        visionRuntime = prepared.visionRuntime
        visionRuntimeError = prepared.visionError
        conversationLifecycleError = nil
        conversationTokens.withLock { $0 = 0 }
        conversationLogicalStateBytes.withLock { $0 = prepared.logicalStateBytes }
        expertCacheBytes.withLock { $0 = prepared.expertCacheBytes }
        if prepared.tokenizer != nil {
            tokenizerDirectoryCache.markLoaded(for: prepared.key.directory)
        }
        transition.prepared = nil
    }

    private func finishInstall(_ transition: LifecycleTransition) {
        activeLifecycle = nil
        transition.completion.finish()
    }

    private func cleanup(
        _ transition: LifecycleTransition,
        explicitUnload: Bool
    ) async {
        var preparedConversation = transition.prepared?.conversation
        if let preparedConversation { await preparedConversation.invalidate() }
        preparedConversation = nil
        transition.prepared = nil
        transition.retired?.releaseAll()
        transition.retired = nil
        conversationLogicalStateBytes.withLock { $0 = nil }
        expertCacheBytes.withLock { $0 = nil }
        if explicitUnload || transition.teardownRequested {
            tokenizer = nil
            tokenizerDirectoryCache.clear()
            ctx = nil
        }
        if activeLifecycle === transition { activeLifecycle = nil }
        transition.completion.finish()
    }

    private func performInstall(
        runtime: ModelFamilyRuntime,
        qwenCodec: QwenChatCodec?,
        verifiedIdentity: LoadedRuntimeIdentity?,
        admittedBundle: LoadedModelFamilyBundle?,
        key: SessionLoadKey,
        context: MetalContext,
        onState: @Sendable (AppModelLoadState) -> Void
    ) async throws {
        let transition = try beginLifecycleTransition()
        var prepared: PreparedInstallation?
        do {
            try await drainRetired(transition)
            prepared = try await prepare(
                runtime: runtime, qwenCodec: qwenCodec,
                verifiedIdentity: verifiedIdentity, admittedBundle: admittedBundle,
                key: key, context: context, transition: transition, onState: onState)
            transition.prepared = prepared
            await lifecycleCheckpoints.afterPreparation()
            try requireActive(transition)
            try installPrepared(prepared!, transition: transition)
            prepared = nil
            finishInstall(transition)
        } catch {
            prepared = nil
            await cleanup(transition, explicitUnload: false)
            if error is CancellationError || transition.teardownRequested {
                throw CancellationError()
            }
            throw error
        }
    }

    func ensureLoaded(key: SessionLoadKey,
                      onState: @Sendable (AppModelLoadState) -> Void) async throws {
        if activeLifecycle != nil { throw RealInferenceLifecycleError.lifecycleInProgress }
        if loadedKey == key, loadedModelReadiness != nil {
            // An explicit source load must reopen the registration and trust
            // payload rather than accept a stale same-path session.
            guard case .qwenSource = loadedFamily else { return }
        }

        let start = Date()
        let transition: LifecycleTransition
        do {
            transition = try beginLifecycleTransition()
        } catch {
            throw error
        }
        var bundle: LoadedModelFamilyBundle?
        var prepared: PreparedInstallation?
        do {
            try await drainRetired(transition)
            try requireActive(transition)
            onState(.loading(.validatingDirectory))
            let manifest = key.directory.appendingPathComponent("manifest.json")
            let sourceMarker = key.directory.appendingPathComponent("official-source.json")
            guard FileManager.default.fileExists(atPath: manifest.path)
                    || FileManager.default.fileExists(atPath: sourceMarker.path) else {
                throw AppInferenceError.modelNotFound(key.directory.path)
            }
            let context = try (ctx ?? MetalContext())
            let runtimeConfiguration = try key.options.resolvedRuntimeConfiguration(
                forceLogitsHead: key.forceLogitsHead)
            try requireActive(transition)
            onState(.loading(.verifyingWeights))
            bundle = try ModelFamilyRuntime.loadBundle(
                directoryURL: key.directory, device: context.device,
                streamingMode: .pread(slotCount: runtimeConfiguration.expertCacheSlots),
                expertCachePolicy: runtimeConfiguration.modelExpertCachePolicy,
                integrityPolicy: key.options.modelVerification.runtimeValue)
            try requireActive(transition)
            try Self.validate(bundle!)
            prepared = try await prepare(
                runtime: bundle!.runtime, qwenCodec: bundle!.qwenCodec,
                verifiedIdentity: bundle!.verifiedIdentity, admittedBundle: bundle,
                key: key, context: context, transition: transition, onState: onState)
            transition.prepared = prepared
            bundle = nil
            await lifecycleCheckpoints.afterPreparation()
            try requireActive(transition)
            try installPrepared(prepared!, transition: transition)
            prepared = nil
            finishInstall(transition)
            onState(.ready(
                modelDirectory: key.directory,
                loadSeconds: Date().timeIntervalSince(start)))
        } catch {
            bundle = nil
            prepared = nil
            await cleanup(transition, explicitUnload: false)
            if error is CancellationError || transition.teardownRequested {
                throw CancellationError()
            }
            if let appError = error as? AppInferenceError {
                onState(.failed(appError))
                throw appError
            }
            let appError = AppInferenceError.modelLoadFailed("\(error)")
            onState(.failed(appError))
            throw appError
        }
    }

    private static func loadTokenizer(for modelDirectory: URL) async throws -> GFTokenizer {
        try await GFTokenizer.load(forModelDirectory: modelDirectory)
    }

    private static func qwenThinkingMode(
        for request: AppGenerationRequest
    ) -> ModelFamilyThinkingMode {
        request.runtimeOptions.toolThinkingEnabled ? .enabled : .disabled
    }

    private static func qwenTools(
        _ definitions: [AppToolDefinition]
    ) throws -> [ModelChatToolDefinition] {
        try definitions.map { definition in
            ModelChatToolDefinition(function: .init(
                name: definition.name,
                description: definition.description,
                parameters: try qwenJSON(definition.parameters)))
        }
    }

    /// App JSON objects predate the ordered Qwen boundary and therefore cannot
    /// promise insertion order. Sort their keys once at the family adapter so
    /// the model sees stable bytes while tool and result arrays retain the
    /// exact host order supplied by Agent Mode.
    private static func qwenJSON(_ value: JSONValue) throws -> ModelChatJSONValue {
        switch value {
        case .object(let object):
            return .object(try object.keys.sorted().map {
                ModelChatJSONMember($0, try qwenJSON(object[$0]!))
            })
        case .array(let values): return .array(try values.map(qwenJSON))
        case .string(let value): return .string(value)
        case .integer(let value): return .integer(value)
        case .unsignedInteger(let value): return .unsignedInteger(value)
        case .decimal(let value):
            let text = NSDecimalNumber(decimal: value).stringValue
            guard let number = Double(text), number.isFinite else {
                throw AppInferenceError.invalidRequest(
                    "Qwen tool JSON contains a number it cannot represent.")
            }
            return .number(number)
        case .number(let value):
            guard value.isFinite else {
                throw AppInferenceError.invalidRequest(
                    "Qwen tool JSON contains a non-finite number.")
            }
            return .number(value)
        case .bool(let value): return .bool(value)
        case .null: return .null
        }
    }

    private static func qwenUserMessage(
        prompt: String,
        attachments: [AppImageAttachment]
    ) -> ModelChatMessage {
        guard !attachments.isEmpty else {
            return ModelChatMessage(role: .user, content: prompt)
        }
        var parts = attachments.map {
            ModelChatContentPart.image(.init(id: $0.id.uuidString))
        }
        if !prompt.isEmpty { parts.append(.text(prompt)) }
        return ModelChatMessage(role: .user, content: .parts(parts))
    }

    private static func qwenToolResultMessages(
        _ results: [AppToolResult]
    ) -> [ModelChatMessage] {
        results.map { result in
            var parts: [ModelChatContentPart] = [.text(result.content)]
            parts.append(contentsOf: result.imageAttachments.map {
                .image(.init(id: $0.id.uuidString))
            })
            let content: ModelChatContent = result.imageAttachments.isEmpty
                ? .text(result.content) : .parts(parts)
            return ModelChatMessage(
                role: .tool, content: content,
                toolCallID: result.callID, name: result.name)
        }
    }

    private static func qwenImagesByID(
        _ attachments: [AppImageAttachment]
    ) throws -> [String: URL] {
        var result: [String: URL] = [:]
        result.reserveCapacity(attachments.count)
        for attachment in attachments {
            try Task.checkCancellation()
            let actualDigest = try Sha256Verifier.hashFile(
                at: attachment.fileURL, chunkBytes: 256 * 1_024)
            guard actualDigest == attachment.sha256 else {
                throw AppInferenceError.invalidRequest(
                    "Image \(attachment.displayName) changed after selection.")
            }
            guard result.updateValue(
                attachment.fileURL, forKey: attachment.id.uuidString) == nil else {
                throw AppInferenceError.invalidRequest("Images must be distinct.")
            }
        }
        return result
    }

    private static func mapQwenGenerationError(_ error: Error) -> AppInferenceError {
        if let appError = error as? AppInferenceError { return appError }
        if let error = error as? ModelFamilyGenerationError {
            switch error {
            case .contextOverflow(let prompt, let maxNew, let maximum):
                return .contextOverflow(prompt: prompt, maxNew: maxNew, maxContext: maximum)
            case .busy:
                return .generationInFlight
            case .modelIdentityChanged:
                return .reloadRequired
            case .emptyPrompt, .unsupportedInput, .missingImage, .unexpectedImage,
                    .duplicateImageID, .verifiedVisionUnavailable,
                    .sourceVisionUnavailable, .incompatibleVisionPack:
                return .invalidRequest(error.description)
            }
        }
        if let error = error as? ConversationStateTransactionError {
            switch error {
            case .restoreFailed:
                return .conversationLineageLost(String(describing: error))
            case .contextExceeded(let requested, let maximum):
                return .contextOverflow(prompt: requested, maxNew: 0, maxContext: maximum)
            case .busy:
                return .generationInFlight
            case .staleTransaction, .invalidBoundary, .unsupportedFamily,
                    .tokenCodecUnavailable:
                return .invalidRequest(String(describing: error))
            }
        }
        if error is QwenChatCodecError
            || error is QwenConversationGenerationError
            || error is MultimodalPromptRendererError {
            return .invalidRequest(String(describing: error))
        }
        return .unknown(String(describing: error))
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
        if let activeLifecycle {
            activeLifecycle.teardownRequested = true
            lifecycleUnloadWaiters += 1
            defer { lifecycleUnloadWaiters -= 1 }
            await activeLifecycle.completion.wait()
            return
        }
        guard loadedKey != nil || loadedFamily != nil || conversation != nil else {
            tokenizer = nil
            tokenizerDirectoryCache.clear()
            ctx = nil
            conversationLogicalStateBytes.withLock { $0 = nil }
            expertCacheBytes.withLock { $0 = nil }
            return
        }
        guard let transition = try? beginLifecycleTransition() else { return }
        transition.teardownRequested = true
        await waitForQwenOrdinaryGeneration()
        qwenTools.removeAll(keepingCapacity: true)
        qwenSystemPrompt = nil
        qwenPendingToolCalls.removeAll(keepingCapacity: true)
        conversationTokens.withLock { $0 = 0 }
        conversationLogicalStateBytes.withLock { $0 = nil }
        expertCacheBytes.withLock { $0 = nil }
        var capturedConversation = transition.retired?.conversation
        if let capturedConversation { await capturedConversation.invalidate() }
        capturedConversation = nil
        transition.retired?.releaseAll()
        transition.retired = nil
        tokenizer = nil
        tokenizerDirectoryCache.clear()
        ctx = nil
        if activeLifecycle === transition { activeLifecycle = nil }
        transition.completion.finish()
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
            case .unsupportedFamily, .tokenCodecUnavailable:
                throw AppInferenceError.invalidRequest("\(error)")
            case .closed, .busy, .emptyTurn,
                    .toolModeRequiresNewConversation, .invalidToolContinuation:
                throw AppInferenceError.unknown("\(error)")
            }
        }
    }

    private func runQwenTurn(
        request: AppGenerationRequest,
        loadedKey: SessionLoadKey,
        progress: ProgressState,
        memorySampler: AppMemorySampler,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation
    ) async throws -> TurnOutcome {
        let expectedKey = SessionLoadKey(
            directory: request.modelDirectory,
            maxContext: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: Self.forceLogitsHead(for: request))
        guard loadedKey == expectedKey else { throw AppInferenceError.reloadRequired }
        if let conversationLifecycleError { throw conversationLifecycleError }
        let generation: QwenTurnGenerator
        switch loadedFamily {
        case .qwen:
            generation = .packed(try beginQwenOrdinaryGeneration())
        case .qwenSource:
            generation = .source(try beginSourceOrdinaryGeneration())
        default:
            throw AppInferenceError.modelNotLoaded
        }
        defer { finishQwenOrdinaryGeneration() }

        if !request.continuesConversation {
            try await generation.reset()
            qwenTools.removeAll(keepingCapacity: true)
            qwenSystemPrompt = nil
            qwenPendingToolCalls.removeAll(keepingCapacity: true)
            await publishQwenConversationStatus()
        }

        let turn: QwenConversationTurn
        let tools: [ModelChatToolDefinition]
        let systemPrompt: String?
        switch request.toolTurn {
        case .user(let developerPrompt, let definitions):
            guard qwenPendingToolCalls.isEmpty else {
                throw AppInferenceError.invalidRequest(
                    "Qwen is waiting for the exact results of its pending tool calls.")
            }
            tools = try Self.qwenTools(definitions)
            systemPrompt = developerPrompt
            turn = .user(Self.qwenUserMessage(
                prompt: request.prompt, attachments: request.imageAttachments))
        case .results(let results):
            guard results.count == qwenPendingToolCalls.count,
                  zip(results, qwenPendingToolCalls).allSatisfy({ result, call in
                      result.callID == call.id && result.name == call.name
                  }) else {
                throw AppInferenceError.invalidRequest(
                    "Tool results do not match the pending Qwen tool calls.")
            }
            tools = qwenTools
            systemPrompt = nil
            turn = .toolResults(Self.qwenToolResultMessages(results))
        case .checkpoint(let id):
            guard generation.supportsCheckpoint else {
                throw ConversationStateTransactionError.unsupportedFamily
            }
            guard qwenPendingToolCalls.isEmpty, request.imageAttachments.isEmpty else {
                throw AppInferenceError.invalidRequest(
                    "Qwen checkpoint resume does not accept another image.")
            }
            tools = qwenTools
            systemPrompt = nil
            turn = .checkpoint(id)
        case nil:
            guard qwenPendingToolCalls.isEmpty else {
                throw AppInferenceError.invalidRequest(
                    "Qwen is waiting for the exact results of its pending tool calls.")
            }
            tools = []
            systemPrompt = nil
            turn = .user(Self.qwenUserMessage(
                prompt: request.prompt, attachments: request.imageAttachments))
        }

        let imagesByID = try Self.qwenImagesByID(request.imageAttachments)
        let previousTokens = conversationTokens.withLock { $0 }
        memorySampler.resetPeak()
        _ = memorySampler.sample()
        progress.prefillStart = Date()
        let terminalCalls = Mutex<[ParsedToolCall]>([])
        let config = Self.generationConfig(
            for: request, maxNewTokens: request.maxNewTokens)
        let result: QwenConversationGenerationResult
        do {
            result = try await generation.generate(
                QwenConversationGenerationRequest(
                    turn: turn,
                    systemPrompt: systemPrompt,
                    tools: tools,
                    imagesByID: imagesByID,
                    thinking: Self.qwenThinkingMode(for: request),
                    visionResidency: request.runtimeOptions.visionResidencyPolicy,
                    config: config),
                shouldStop: stopFlagReader(),
                onEvent: { event in
                    switch event {
                    case .prefill(let done, let total):
                        if done == total {
                            self.decodeBegan.withLock { $0 = true }
                            progress.decodeStart = Date()
                        }
                        continuation.yield(.prefillProgress(done: done, total: total))
                    case .structuredProgress(let value):
                        progress.updateStructuredProgress(value)
                        progress.generated = max(progress.generated,
                            value.thinkingTokens + value.toolCallTokens
                                + value.visibleResponseTokens + value.channelLabelTokens
                                + value.unknownHiddenChannelTokens)
                    case .text(let text):
                        if progress.firstTokenDate == nil { progress.firstTokenDate = Date() }
                        let index = progress.nextVisibleEventIndex
                        continuation.yield(.token(AppTokenEvent(
                            index: index,
                            textDelta: text,
                            elapsedDecodeSeconds: progress.elapsedDecodeSeconds,
                            structuredProgress: progress.structuredProgress,
                            thinkingPreview: progress.thinkingPreview,
                            toolCallPreview: progress.toolCallPreview)))
                    case .toolCall(let call):
                        terminalCalls.withLock { $0.append(call) }
                    }
                })
        } catch is CancellationError {
            await publishQwenConversationStatus()
            throw CancellationError()
        } catch {
            await publishQwenConversationStatus()
            throw Self.mapQwenGenerationError(error)
        }

        progress.generated = result.newTokens
        progress.promptTokenCount = result.promptTokens
        let calls = terminalCalls.withLock { $0 }
        qwenTools = tools
        if let systemPrompt { qwenSystemPrompt = systemPrompt }
        qwenPendingToolCalls = calls
        conversationTokens.withLock { $0 = result.metrics.retainedTokenIDs.count }
        conversationLogicalStateBytes.withLock { $0 = result.metrics.logicalStateBytes }
        return TurnOutcome(
            reason: result.reason,
            prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds,
            newTokens: result.newTokens,
            cachedTokens: request.continuesConversation ? previousTokens : nil,
            computedPrefillTokens: result.promptTokens,
            conversationTokens: result.metrics.retainedTokenIDs.count,
            toolCalls: calls)
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
                directory: request.modelDirectory,
                maxContext: request.maxContextTokens,
                options: request.runtimeOptions,
                forceLogitsHead: Self.forceLogitsHead(for: request))
            guard let loadedKey else { throw AppInferenceError.modelNotLoaded }
            switch loadedFamily {
            case .qwen, .qwenSource:
                guard qwenGeneration != nil || sourceGeneration != nil else {
                    throw AppInferenceError.invalidRequest(
                        "Qwen string, tool, and image generation requires the Phase 14 chat codec; use the prepared-token boundary.")
                }
                let outcome = try await runQwenTurn(
                    request: request, loadedKey: loadedKey,
                    progress: progress, memorySampler: memorySampler,
                    continuation: continuation)
                for call in outcome.toolCalls {
                    continuation.yield(.toolCall(AppToolCall(
                        id: call.id, name: call.name, arguments: call.arguments)))
                }
                let diagnostics = makeDiagnostics(
                    request: request, memorySampler: memorySampler, progress: progress,
                    stopReason: Self.stopReason(outcome.reason),
                    prefillSeconds: outcome.prefillSeconds,
                    decodeSeconds: outcome.decodeSeconds, generated: outcome.newTokens,
                    cachedTokens: outcome.cachedTokens,
                    computedPrefillTokens: outcome.computedPrefillTokens,
                    conversationTokens: outcome.conversationTokens,
                    prefill: PrefillExecutionDiagnostics(
                        config: prefillConfig,
                        executedMode: prefillConfig.mode == .chunked ? .chunked : .off,
                        kvStorageMode: .fp16))
                continuation.yield(.finished(diagnostics))
                continuation.finish()
                return
            case .gemma:
                guard loadedKey == requestKey else { throw AppInferenceError.reloadRequired }
            case nil:
                throw AppInferenceError.modelNotLoaded
            }
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
            // `Model` owns mutable streamer state behind its serial queue. The
            // generation registry keeps this installation alive and unique for
            // the callback while the queue-backed accessor reads real buffers.
            nonisolated(unsafe) let reportingModel = model

            let report: @Sendable (RawDecodeProgress) -> Void = { event in
                if let reportingModel {
                    self.expertCacheBytes.withLock {
                        $0 = reportingModel.routedExpertCacheAllocatedBytes
                    }
                }
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
        publishGemmaExpertCacheBytes()
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
        publishGemmaExpertCacheBytes()
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
            conversationLogicalStateBytes: conversationLogicalStateBytes.withLock { $0 },
            expertCacheBytes: expertCacheBytes.withLock { $0 },
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
    private var visibleEventCount = 0
    var nextVisibleEventIndex: Int {
        defer { visibleEventCount += 1 }
        return visibleEventCount
    }
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
