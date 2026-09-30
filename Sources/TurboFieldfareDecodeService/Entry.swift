import Darwin
import TurboFieldfare
import Foundation
import Synchronization
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

enum DecodeServiceError: Error, CustomStringConvertible {
    case attachmentOutsideStore(path: String)

    var description: String {
        switch self {
        case .attachmentOutsideStore(let path):
            "image attachment is not a staged attachment: \(path)"
        }
    }
}

protocol DecodeServiceModelRuntime: AnyObject, Sendable {
    func ensureLoaded(
        modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws
    func resetConversation(epoch: UUID) async throws
    var loadedModelReadiness: AppLoadedModelReadiness? { get async }
    func unload() async
}

extension RealInferenceClient: DecodeServiceModelRuntime {}

/// Only a runtime retaining an admitted BF16 model can validate its source
/// receipt after the service's await points. Synthetic/packed runtimes cannot
/// turn a digest received over the wire into an inference ticket.
protocol DecodeServiceSourceRuntime: DecodeServiceModelRuntime {
    func validateSourceReadiness(_ identity: DecodeSourceIdentity) async throws
}

extension RealInferenceClient: DecodeServiceSourceRuntime {}

/// Owns the service-side model transaction through ready-frame completion.
/// Session leases are synchronous; model work never runs while their mutex is held.
final class DecodeServiceLoadCoordinator: Sendable {
    enum Rejection: Error, Equatable {
        case lifecycleInProgress, invalidAttempt, sourceRegistrationChanged
    }

    struct Attempt: Sendable, Equatable {
        fileprivate let id: UUID
        fileprivate let completion: Completion
        let lease: DecodeServiceSession.LoadLease

        static func == (lhs: Self, rhs: Self) -> Bool { lhs.id == rhs.id }
    }

    fileprivate final class Completion: Sendable {
        private struct State: Sendable {
            var finished = false
            var waiters: [CheckedContinuation<Void, Never>] = []
        }
        private let state = Mutex(State())

        func wait() async {
            await withCheckedContinuation { continuation in
                let resume = state.withLock { value -> Bool in
                    if value.finished { return true }
                    value.waiters.append(continuation)
                    return false
                }
                if resume { continuation.resume() }
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

    private struct State: Sendable {
        var active: Attempt?
        var sourceProbe: OfficialSourceRegistrationProbe?
        var aborting = false
    }
    private let state = Mutex(State())
    private let session: DecodeServiceSession
    private let runtime: any DecodeServiceModelRuntime

    init(session: DecodeServiceSession, runtime: any DecodeServiceModelRuntime) {
        self.session = session
        self.runtime = runtime
    }

    func beginReplacement(_ lease: DecodeServiceSession.LoadLease) throws -> Attempt {
        try state.withLock { value in
            guard value.active == nil else { throw Rejection.lifecycleInProgress }
            let attempt = Attempt(id: UUID(), completion: Completion(), lease: lease)
            value.active = attempt
            value.sourceProbe = nil
            return attempt
        }
    }

    private func require(_ attempt: Attempt) throws {
        guard state.withLock({ $0.active == attempt }) else {
            throw Rejection.invalidAttempt
        }
        try session.requireActive(attempt.lease)
    }

    func prepare(
        _ attempt: Attempt, directory: URL, maxContextTokens: Int,
        options: AppRuntimeOptions, forceLogitsHead: Bool
    ) async throws -> AppLoadedModelReadiness {
        try require(attempt)
        // Retain the actual protected marker/root handles across loader awaits.
        // Metadata checks never stand in for ModelFamilyRuntime.loadBundle trust.
        let probe = try OfficialSourceRegistrationProbe.openIfSource(
            directoryURL: directory)
        try require(attempt)
        try state.withLock { value in
            guard value.active == attempt else { throw Rejection.invalidAttempt }
            value.sourceProbe = probe
        }
        try await runtime.ensureLoaded(
            modelDirectory: directory, maxContextTokens: maxContextTokens,
            options: options, forceLogitsHead: forceLogitsHead) { _ in }
        try require(attempt)
        try await runtime.resetConversation(epoch: UUID())
        try require(attempt)
        guard let readiness = await runtime.loadedModelReadiness else {
            throw AppInferenceError.modelLoadFailed(
                "loaded runtime did not publish verified family readiness")
        }
        try require(attempt)
        try validateReadiness(readiness, attempt: attempt)
        if case .qwenSource(let identity) = readiness {
            try await validateAdmittedSource(identity, attempt: attempt)
        }
        return readiness
    }

    private func validateAdmittedSource(
        _ identity: DecodeSourceIdentity, attempt: Attempt
    ) async throws {
        try require(attempt)
        guard let source = runtime as? any DecodeServiceSourceRuntime else {
            throw Rejection.sourceRegistrationChanged
        }
        try await source.validateSourceReadiness(identity)
        try require(attempt)
        guard let probe = try sourceProbe(attempt),
              probe.contentDigest == identity.contentDigest else {
            throw Rejection.sourceRegistrationChanged
        }
        try probe.revalidate()
        try require(attempt)
    }

    private func sourceProbe(_ attempt: Attempt) throws
        -> OfficialSourceRegistrationProbe? {
        try require(attempt)
        return try state.withLock { value in
            guard value.active == attempt else { throw Rejection.invalidAttempt }
            return value.sourceProbe
        }
    }

    private func validateReadiness(
        _ readiness: AppLoadedModelReadiness, attempt: Attempt
    ) throws {
        let probe = try sourceProbe(attempt)
        try probe?.revalidate()
        switch (probe, readiness) {
        case (nil, .qwenSource), (.some, .gemma), (.some, .qwen):
            throw Rejection.sourceRegistrationChanged
        case (.some(let probe), .qwenSource(let identity)):
            guard identity.kind == .officialSafetensorsBF16V1,
                  identity.contentDigest == probe.contentDigest else {
                throw Rejection.sourceRegistrationChanged
            }
        case (nil, .gemma), (nil, .qwen):
            break
        }
    }

    /// Called after encoding, before ready I/O. The source model rechecks its
    /// retained trust receipt and named payloads, then the probe rechecks its
    /// retained marker/root; no model work occurs under coordinator locks.
    func validatePublication(
        _ attempt: Attempt, event: DecodeServiceEvent
    ) async throws {
        try require(attempt)
        try event.validateBackingIdentity()
        guard event.kind == .ready,
              let binding = session.binding(loadID: event.loadID),
              binding.family == event.loadedFamily,
              binding.modelIdentity == event.modelIdentity,
              binding.sourceIdentity == event.sourceIdentity else {
            throw Rejection.sourceRegistrationChanged
        }
        let probe = try sourceProbe(attempt)
        try probe?.revalidate()
        switch (probe, event.sourceIdentity) {
        case (nil, nil): break
        case (.some(let held), .some(let identity))
            where held.contentDigest == identity.contentDigest
                && identity.kind == .officialSafetensorsBF16V1:
            break
        default: throw Rejection.sourceRegistrationChanged
        }
        if let identity = event.sourceIdentity {
            try await validateAdmittedSource(identity, attempt: attempt)
        }
        try require(attempt)
    }

    func reservePublication(
        _ attempt: Attempt, readiness: AppLoadedModelReadiness
    ) throws -> DecodeServiceSession.Binding {
        try require(attempt)
        try validateReadiness(readiness, attempt: attempt)
        return try session.reserveAndPublish(readiness, lease: attempt.lease)
    }

    func finishCommit(_ attempt: Attempt) throws -> DecodeServiceSession.CommitResult {
        try require(attempt)
        let result = try session.finishCommit(attempt.lease)
        state.withLock { value in
            if value.active == attempt {
                value.active = nil
                value.sourceProbe = nil
                value.aborting = false
            }
        }
        attempt.completion.finish()
        return result
    }

    func abortActive() async {
        guard let attempt = state.withLock({ $0.active }) else { return }
        await abort(attempt)
    }

    func cancel(_ request: DecodeCancelLoadRequest) async {
        let attempt = state.withLock { value -> Attempt? in
            guard let active = value.active,
                  active.lease.requestID == request.requestID,
                  active.lease.attemptID == request.attemptID else { return nil }
            return active
        }
        guard let attempt else { return }
        await abort(attempt)
    }

    func abort(_ attempt: Attempt) async {
        let owns = state.withLock { value -> Bool in
            guard value.active == attempt else { return false }
            guard !value.aborting else { return false }
            value.aborting = true
            return true
        }
        guard owns else {
            await attempt.completion.wait()
            return
        }
        session.beginLoadFailure(attempt.lease)
        await runtime.unload()
        session.finishLoadFailure(attempt.lease)
        state.withLock { value in
            if value.active == attempt {
                value.active = nil
                value.sourceProbe = nil
                value.aborting = false
            }
        }
        attempt.completion.finish()
    }
}

/// Publishes the ready frame and commits its exact load attempt as one
/// fail-closed boundary. The writer runs synchronously without coordinator locks.
struct DecodeServiceReadyPublisher {
    typealias WriteFrame = @Sendable (Data, FileHandle) throws -> Void

    enum Outcome: Sendable, Equatable {
        case committed(DecodeServiceSession.CommitResult)
        case deliveryFailed
    }

    static func publish(
        event: DecodeServiceEvent,
        to output: FileHandle,
        attempt: DecodeServiceLoadCoordinator.Attempt,
        coordinator: DecodeServiceLoadCoordinator,
        writeFrame: WriteFrame = { frame, output in
            try output.write(contentsOf: frame)
        }
    ) async -> Outcome {
        do {
            let frame = try DecodeFrameCodec.encode(event)
            try await coordinator.validatePublication(attempt, event: event)
            try writeFrame(frame, output)
            return .committed(try coordinator.finishCommit(attempt))
        } catch {
            // The ready bytes may be partly or fully visible. Join cleanup before
            // closing and never permit another frame on this transport.
            await coordinator.abort(attempt)
            try? output.close()
            return .deliveryFailed
        }
    }
}

@main enum TurboFieldfareDecodeServiceMain {
    static func main() async {
        let socketPath = argument(after: "--socket")
        let launchLabel = argument(after: "--launch-label")
        let handles: (input: FileHandle, output: FileHandle)
        do {
            handles = if let socketPath {
                try DecodeUnixSocket.listenAndAccept(path: socketPath)
            } else {
                (.standardInput, .standardOutput)
            }
        } catch {
            FileHandle.standardError.write(Data("Decode service transport failed: \(error)\n".utf8))
            Foundation.exit(1)
        }
        defer {
            if let socketPath { unlink(socketPath) }
            if let launchLabel { retireLaunchJob(launchLabel) }
        }

        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
        let client = RealInferenceClient()
        let commands = DecodeCommandQueue()
        let session = DecodeServiceSession()
        let loadCoordinator = DecodeServiceLoadCoordinator(
            session: session, runtime: client)
        let input = Thread {
            do {
                while true {
                    let command = try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handles.input)
                    let queued = DecodeQueuedCommand.admitting(command, using: session)
                    if case .cancelLoad(let request) = command {
                        if session.markCancel(request) {
                            Task { await loadCoordinator.cancel(request) }
                        }
                        continue
                    }
                    if case .generate(let request) = command {
                        // Arm before enqueue: a bound stop may arrive before the
                        // main loop starts the producer.
                        _ = session.prepare(
                            loadID: request.loadID, operationID: request.generationID)
                    }
                    if case .contextCheckpoint(let request) = command {
                        _ = session.prepare(
                            loadID: request.loadID, operationID: request.requestID)
                    }
                    if case .cancelGenerationBound(let request) = command {
                        _ = session.requestStop(
                            loadID: request.loadID, operationID: request.operationID)
                        continue
                    }
                    if case .cancelGeneration(_) = command { continue }
                    if case .cancel = command { continue }
                    if case .shutdown = command {
                        // Shutdown is read on this dedicated thread while the
                        // main loop may still be awaiting generation or load.
                        client.stop()
                        Task { await loadCoordinator.abortActive() }
                    }
                    commands.append(queued)
                    if case .shutdown = command { break }
                }
            } catch {
                // EOF is the normal app-close path. There may be no shutdown
                // frame if the process exited between close and write.
                client.stop()
                Task { await loadCoordinator.abortActive() }
                commands.close()
            }
        }
        input.name = "TurboFieldfare.DecodeService.Input"
        input.qualityOfService = .userInitiated
        input.start()

        var modelDirectory: URL?
        var loadedOptions: DecodeRuntimeOptions?
        var conversation = DecodeConversationGate()
        var pendingToolAdmission: DecodeConversationGate.Admission?
        while let queued = await nextCommand(commands) {
            let command = queued.command
            switch command {
            case .lifetimeProbe(let probe):
                do {
                    try write(DecodeServiceEvent(
                        kind: .lifetimeAcknowledged,
                        generationID: probe.nonce,
                        lifetimeNonce: probe.nonce), to: handles.output)
                } catch {
                    try? handles.output.close()
                    return
                }
            case .load(let request):
                guard case .load(let admittedLease) = queued.admission,
                      let lease = admittedLease else {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.requestID,
                        loadAttemptID: request.attemptID,
                        error: "a model lifecycle transition is already in progress"),
                        to: handles.output)
                    continue
                }
                let attempt: DecodeServiceLoadCoordinator.Attempt
                do {
                    attempt = try loadCoordinator.beginReplacement(lease)
                } catch {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.requestID,
                        loadAttemptID: request.attemptID,
                        error: "\(error)"), to: handles.output)
                    continue
                }
                // Registration already retired the visible service binding.
                // Keep transaction ownership through conversion and ready I/O.
                modelDirectory = nil
                loadedOptions = nil
                conversation.endLineage()
                pendingToolAdmission = nil
                let directory = URL(fileURLWithPath: request.modelPath)
                do {
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    let readiness = try await loadCoordinator.prepare(
                        attempt, directory: directory,
                        maxContextTokens: request.maxContextTokens,
                        options: options,
                        forceLogitsHead: request.forceLogitsHead)
                    if case .gemma(let thinkingEnabled) = readiness,
                       thinkingEnabled != options.toolThinkingEnabled {
                        throw AppInferenceError.modelLoadFailed(
                            "loaded tokenizer thinking mode does not match the requested setting")
                    }
                    let binding = try loadCoordinator.reservePublication(
                        attempt, readiness: readiness)
                    modelDirectory = directory
                    loadedOptions = request.runtimeOptions
                    let memory = AppMemorySampler().sample()
                    let publication = await DecodeServiceReadyPublisher.publish(
                        event: DecodeServiceEvent(
                            kind: .ready, generationID: request.requestID,
                            loadAttemptID: request.attemptID,
                            loadedFamily: binding.family, loadID: binding.loadID,
                            modelIdentity: binding.modelIdentity,
                            sourceIdentity: binding.sourceIdentity,
                            currentMemoryBytes: memory, peakMemoryBytes: memory,
                            conversationLogicalStateBytes:
                                client.currentConversationLogicalStateBytes,
                            expertCacheBytes: client.currentExpertCacheBytes,
                            toolThinkingEnabled: binding.toolThinkingEnabled),
                        to: handles.output,
                        attempt: attempt,
                        coordinator: loadCoordinator)
                    switch publication {
                    case .committed:
                        break
                    case .deliveryFailed:
                        modelDirectory = nil
                        loadedOptions = nil
                        return
                    }
                } catch {
                    let cancelled = session.wasCancellationRequested(attempt.lease)
                        || error is CancellationError
                        || (error as? DecodeServiceSession.Rejection) == .loadCancelled
                    await loadCoordinator.abort(attempt)
                    modelDirectory = nil
                    loadedOptions = nil
                    let kind: DecodeServiceEventKind = cancelled ? .loadCancelled : .failed
                    do {
                        try write(DecodeServiceEvent(
                            kind: kind, generationID: request.requestID,
                            loadAttemptID: request.attemptID,
                            error: cancelled ? nil : "\(error)"), to: handles.output)
                    } catch {
                        try? handles.output.close()
                        return
                    }
                }
            case .cancelLoad:
                // Consumed synchronously by the input-thread registry.
                break
            case .resetConversation(let request):
                guard let binding = session.binding(loadID: request.loadID) else {
                    try? write(Self.event(
                        .failed, id: request.requestID,
                        binding: session.currentBinding,
                        error: "reset does not match the loaded session",
                        epoch: conversation.openEpoch), to: handles.output)
                    continue
                }
                do {
                    try await client.resetConversation(epoch: request.epoch)
                    conversation.reset(to: request.epoch)
                    pendingToolAdmission = nil
                    // Not `try?`. The gate has already reset; if the app never
                    // hears so it waits out the whole timeout for a reply that
                    // cannot come, and then cannot tell that from a slow service.
                    // Closing the stream makes it an EOF the client rebuilds from.
                    var event = Self.event(
                        .conversationReset, id: request.requestID, binding: binding,
                        conversationTokenCount: 0, epoch: request.epoch)
                    event.conversationLogicalStateBytes =
                        client.currentConversationLogicalStateBytes
                    event.expertCacheBytes = client.currentExpertCacheBytes
                    try write(event, to: handles.output)
                } catch {
                    let message = "Decode service closing after a lost "
                        + "conversation reset: \(error)\n"
                    FileHandle.standardError.write(Data(message.utf8))
                    _ = session.retire(loadID: binding.loadID)
                    await client.unload()
                    try? handles.output.close()
                    return
                }
            case .contextCheckpoint(let request):
                defer { session.retire(loadID: request.loadID, operationID: request.requestID) }
                guard let binding = session.binding(loadID: request.loadID) else {
                    try? write(Self.event(
                        .failed, id: request.requestID,
                        binding: session.currentBinding,
                        error: "checkpoint does not match the loaded session",
                        epoch: conversation.openEpoch), to: handles.output)
                    continue
                }
                var checkpointWasCommitted = false
                do {
                    if let previous = try conversation.previousCheckpoint(request) {
                        checkpointWasCommitted = previous.committed
                        var event = Self.event(
                            .contextCheckpoint, id: request.requestID, binding: binding,
                            conversationTokenCount: previous.committed ? 0 : client.currentConversationTokens,
                            epoch: previous.committed ? previous.replacementEpoch : request.sourceEpoch)
                        event.contextCheckpoint = previous
                        event.conversationLogicalStateBytes =
                            client.currentConversationLogicalStateBytes
                        event.expertCacheBytes = client.currentExpertCacheBytes
                        try write(event, to: handles.output)
                        continue
                    }
                    try conversation.validateCheckpoint(request)
                    guard let checkpointAdmission = pendingToolAdmission,
                          Self.matches(checkpointAdmission, epoch: request.sourceEpoch,
                                       index: request.sourceTurnIndex),
                          let stop = session.stop(
                            loadID: request.loadID, operationID: request.requestID) else {
                        throw DecodeConversationGate.Rejection.checkpointMismatch
                    }
                    for attachment in request.result.imageAttachments ?? [] {
                        guard AppImageAttachmentStore.contains(URL(fileURLWithPath: attachment.path)) else {
                            throw DecodeServiceError.attachmentOutsideStore(path: attachment.path)
                        }
                    }
                    let receipt = try await client.contextCheckpoint(request, stop: stop)
                    checkpointWasCommitted = receipt.committed
                    try conversation.recordCheckpoint(request, receipt: receipt)
                    if receipt.committed { pendingToolAdmission = nil }
                    var event = Self.event(
                        .contextCheckpoint, id: request.requestID, binding: binding,
                        conversationTokenCount: receipt.committed ? 0 : client.currentConversationTokens,
                        epoch: receipt.committed ? receipt.replacementEpoch : request.sourceEpoch)
                    event.contextCheckpoint = receipt
                    event.conversationLogicalStateBytes =
                        client.currentConversationLogicalStateBytes
                    event.expertCacheBytes = client.currentExpertCacheBytes
                    // If this acknowledgement is lost, the app stops on EOF.
                    // Retrying this identity returns the same receipt, never resets twice.
                    try write(event, to: handles.output)
                } catch {
                    if checkpointWasCommitted {
                        // The replacement may already be the only valid KV
                        // lineage. A failed acknowledgement must become EOF,
                        // never a silent wait or another admitted command.
                        FileHandle.standardError.write(Data(
                            "Decode service closing after a lost checkpoint acknowledgement: \(error)\n".utf8))
                        try? handles.output.close()
                        return
                    }
                    try? write(Self.event(
                        .failed, id: request.requestID, binding: binding,
                        error: "Context checkpoint failed: \(error)",
                        epoch: conversation.openEpoch), to: handles.output)
                }
            case .generate(let request):
                defer { session.retire(loadID: request.loadID, operationID: request.generationID) }
                guard let binding = session.binding(loadID: request.loadID) else {
                    try? write(Self.event(
                        .failed, id: request.generationID,
                        binding: session.currentBinding,
                        error: "generation does not match the loaded session",
                        epoch: conversation.openEpoch), to: handles.output)
                    continue
                }
                let generationStop = session.stop(
                    loadID: request.loadID, operationID: request.generationID)
                guard generationStop != nil else {
                    try? write(Self.event(
                        .failed, id: request.generationID, binding: binding,
                        error: "generation operation limit reached",
                        epoch: conversation.openEpoch), to: handles.output)
                    continue
                }
                guard let modelDirectory else {
                    try? write(Self.event(
                        .failed, id: request.generationID, binding: binding,
                        error: "model is not loaded", epoch: conversation.openEpoch),
                        to: handles.output)
                    continue
                }
                // Prefill is chosen per request, not at load, so it must not be
                // part of this comparison: toggling it and pressing Generate was
                // refused as a mismatched session.
                var comparable = request.runtimeOptions
                comparable.prefillEnabled = loadedOptions?.prefillEnabled
                    ?? comparable.prefillEnabled
                comparable.prefillChunkTokens = loadedOptions?.prefillChunkTokens
                    ?? comparable.prefillChunkTokens
                guard comparable == loadedOptions else {
                    try? write(Self.event(
                        .failed, id: request.generationID, binding: binding,
                        error: "generation runtime options do not match the loaded session",
                        epoch: conversation.openEpoch),
                        to: handles.output)
                    continue
                }
                // Fail closed before the model is touched. The decision
                // lives in `DecodeConversationGate`, where its boundary cases
                // are tested without a socket or a model.
                let admission: DecodeConversationGate.Admission
                if case .results = request.toolTurn {
                    guard let pendingToolAdmission,
                          Self.matches(
                            pendingToolAdmission,
                            epoch: request.conversationEpoch,
                            index: request.turnIndex) else {
                        try? write(Self.event(
                            .failed, id: request.generationID, binding: binding,
                            error: "tool results do not match the pending app turn",
                            epoch: conversation.openEpoch), to: handles.output)
                        continue
                    }
                    admission = pendingToolAdmission
                } else {
                    guard pendingToolAdmission == nil else {
                        try? write(Self.event(
                            .failed, id: request.generationID, binding: binding,
                            error: "the pending tool turn needs results before another user turn",
                            epoch: conversation.openEpoch), to: handles.output)
                        continue
                    }
                    switch conversation.admit(request) {
                    case .success(let value):
                        admission = value
                    case .failure(let rejection):
                        try? write(Self.event(
                            .failed, id: request.generationID, binding: binding,
                            error: rejection.message,
                            epoch: conversation.openEpoch), to: handles.output)
                        continue
                    }
                }
                let isConversationTurn: Bool
                if case .turn = admission { isConversationTurn = true }
                else { isConversationTurn = false }
                // Measurement labels are defined for the agreed chunked-128
                // path. Unsupported settings run normally with an explicit
                // capture status and no collector allocation.
                let measurementSupported = request.runtimeOptions.prefillEnabled
                    && request.runtimeOptions.prefillChunkTokens == 128
                let measurementCapture = request.runtimeMeasurementCapture != nil
                    && measurementSupported ? RuntimeMeasurementCapture() : nil
                let outbox = DecodeServiceOutbox(
                    generationID: request.generationID,
                    loadedFamily: binding.family,
                    loadID: binding.loadID,
                    modelIdentity: binding.modelIdentity,
                    sourceIdentity: binding.sourceIdentity,
                    conversationEpoch: request.conversationEpoch,
                    towerBytes: { client.currentVisionTowerBytes },
                    conversationTokens: {
                        isConversationTurn ? client.currentConversationTokens : nil
                    },
                    conversationLogicalStateBytes: {
                        client.currentConversationLogicalStateBytes
                    },
                    expertCacheBytes: { client.currentExpertCacheBytes },
                    measurementRequest: request.runtimeMeasurementCapture,
                    measurementCapture: measurementCapture)
                let writerFinished = DispatchSemaphore(value: 0)
                let writer = Thread {
                    defer { writerFinished.signal() }
                    do { try outbox.runWriter(to: handles.output) }
                    catch {
                        FileHandle.standardError.write(Data("IPC writer failed: \(error)\n".utf8))
                    }
                }
                writer.name = "TurboFieldfare.DecodeService.Writer"
                writer.qualityOfService = .userInitiated
                writer.start()

                do {
                    // The trust boundary: these paths arrive over a socket, and
                    // this process opens and hashes whatever they name. Without
                    // this, a peer could use the service to report on any file
                    // the user can read.
                    if let outside = (request.imageAttachments ?? []).first(where: {
                        !AppImageAttachmentStore.contains(URL(fileURLWithPath: $0.path))
                    }) {
                        throw DecodeServiceError.attachmentOutsideStore(
                            path: outside.path)
                    }
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    // The conversation already in the KV is what the image
                    // budget has to fit around; reserving zero admits an image
                    // that only fits an empty context.
                    let carried = await client.conversationTokenCount
                    let continues = isConversationTurn
                    var generation = AppGenerationRequest(
                        modelDirectory: modelDirectory, prompt: request.prompt,
                        imageAttachments: (request.imageAttachments ?? []).map {
                            AppImageAttachment(
                                id: $0.id,
                                fileURL: URL(fileURLWithPath: $0.path),
                                displayName: $0.displayName,
                                encodedBytes: $0.encodedBytes,
                                sha256: $0.sha256)
                        },
                        maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        topK: request.topK,
                        topP: request.topP,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: options,
                        continuesConversation: continues,
                        conversationTokens: continues ? carried : 0,
                        conversationEpoch: request.conversationEpoch,
                        turnIndex: request.turnIndex,
                        toolTurn: try appToolTurn(request.toolTurn))
                    generation.captureToolFailureEvidence = request.captureToolFailureEvidence == true
                    generation.captureGPUCompletionTiming = request.captureGPUCompletionTiming == true
                    generation.runtimeMeasurementCapture = request.runtimeMeasurementCapture
                    var terminalStopReason: AppStopReason?
                    for try await event in client.generate(
                        generation, measurementCapture: measurementCapture,
                        generationStop: generationStop) {
                        if case .finished(let diagnostics) = event {
                            terminalStopReason = diagnostics.stopReason
                        }
                        outbox.publish(event)
                    }
                    // Reached only when the stream completed. A turn that threw
                    // was rewound by the conversation (or broke its lineage), so
                    // its tokens are not in the KV and it must not advance the
                    // order the next turn has to match — the app does not count
                    // it either, and a one-sided count rejects every later turn.
                    // A turn stopped by the user does reach here: it ends at a
                    // token boundary with its partial reply committed.
                    if case .checkpoint = request.toolTurn { conversation.checkpointResumed() }
                    if request.toolTurn != nil,
                       terminalStopReason == .toolCalls {
                        pendingToolAdmission = admission
                    } else {
                        conversation.commit(admission)
                        pendingToolAdmission = nil
                    }
                    outbox.finish()
                } catch {
                    outbox.finish(error: error)
                }
                await withCheckedContinuation { continuation in
                    DispatchQueue.global(qos: .userInitiated).async {
                        writerFinished.wait()
                        continuation.resume()
                    }
                }
            case .cancel, .cancelGeneration(_), .cancelGenerationBound(_):
                break
            case .unload(let requestID):
                // Legacy unbound control is intentionally inert once P17 owns
                // the session; it cannot name the incarnation to tear down.
                try? write(Self.event(
                    .failed, id: requestID, binding: session.currentBinding,
                    error: "unload is missing the loaded session incarnation",
                    epoch: conversation.openEpoch), to: handles.output)
            case .unloadBound(let request):
                guard case .unloadBound(let admittedLease) = queued.admission,
                      let lease = admittedLease,
                      let binding = session.binding(for: lease) else {
                    try? write(Self.event(
                        .failed, id: request.requestID,
                        binding: session.currentBinding,
                        error: "unload does not match the loaded session",
                        epoch: conversation.openEpoch), to: handles.output)
                    continue
                }
                await client.unload()
                modelDirectory = nil
                loadedOptions = nil
                conversation.endLineage()
                pendingToolAdmission = nil
                do {
                    try write(Self.event(
                        .unloaded, id: request.requestID, binding: binding),
                        to: handles.output)
                    _ = session.finishTeardown(lease)
                } catch {
                    // Runtime is clean but acknowledgement delivery is ambiguous.
                    // Closing makes the client retain cleanup until process death.
                    try? handles.output.close()
                    return
                }
            case .shutdown:
                _ = session.retireCurrent()
                await client.unload()
                return
            }
        }
    }

    private static func nextCommand(_ commands: DecodeCommandQueue)
        async -> DecodeQueuedCommand? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: commands.next())
            }
        }
    }

    private static func event(
        _ kind: DecodeServiceEventKind,
        id: UUID,
        binding: DecodeServiceSession.Binding?,
        error: String? = nil,
        conversationTokenCount: Int? = nil,
        epoch: UUID? = nil
    ) -> DecodeServiceEvent {
        DecodeServiceEvent(
            kind: kind, generationID: id,
            loadedFamily: binding?.family, loadID: binding?.loadID,
            modelIdentity: binding?.modelIdentity,
            sourceIdentity: binding?.sourceIdentity,
            error: error, conversationTokenCount: conversationTokenCount,
            conversationEpoch: epoch)
    }

    private static func write(_ event: DecodeServiceEvent,
                              to handle: FileHandle) throws {
        try handle.write(contentsOf: DecodeFrameCodec.encode(event))
    }

    private static func appRuntimeOptions(_ options: DecodeRuntimeOptions) throws
        -> AppRuntimeOptions {
        guard let cachePolicy = AppExpertCachePolicy(
            rawValue: options.expertCachePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown expert cache policy \(options.expertCachePolicy)")
        }
        guard let rdadvisePolicy = AppRDAdvicePolicy(
            rawValue: options.rdadvisePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown RDADVISE policy \(options.rdadvisePolicy)")
        }
        guard let modelVerification = AppModelVerification(
            rawValue: options.modelVerification) else {
            throw AppInferenceError.invalidRequest(
                "unknown model verification \(options.modelVerification)")
        }
        // An unknown policy is a request for behaviour that does not exist;
        // absent means the shipped default.
        guard let visionResidencyPolicy = VisionResidencyPolicy(
            rawValue: options.visionResidencyPolicy ?? VisionResidencyPolicy.onDemand.rawValue)
        else {
            throw AppInferenceError.invalidRequest(
                "unknown vision residency policy \(options.visionResidencyPolicy ?? "")")
        }
        let resolved = AppRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: cachePolicy,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: rdadvisePolicy,
            modelVerification: modelVerification,
            visionResidencyPolicy: visionResidencyPolicy,
            toolThinkingEnabled: options.toolThinkingEnabled ?? GFTokenizer.toolThinkingEnabled)
        try resolved.validate()
        return resolved
    }

    private static func appToolTurn(_ turn: DecodeToolTurn?) throws
        -> AppToolTurn? {
        switch turn {
        case .checkpoint(let id):
            return .checkpoint(id)
        case .user(let developerPrompt, let tools):
            return .user(
                developerPrompt: developerPrompt,
                tools: try tools.map { tool in
                    guard let data = tool.parametersJSON.data(using: .utf8) else {
                        throw AppInferenceError.invalidRequest(
                            "tool parameters are not UTF-8 JSON")
                    }
                    return AppToolDefinition(
                        name: tool.name,
                        description: tool.description,
                        parameters: try JSONDecoder().decode(
                            JSONValue.self, from: data))
                })
        case .results(let results):
            return .results(results.map {
                AppToolResult(
                    callID: $0.callID,
                    name: $0.name,
                    content: $0.content,
                    imageAttachments: ($0.imageAttachments ?? []).map {
                        AppImageAttachment(
                            id: $0.id, fileURL: URL(fileURLWithPath: $0.path),
                            displayName: $0.displayName, encodedBytes: $0.encodedBytes, sha256: $0.sha256)
                    })
            })
        case nil:
            return nil
        }
    }

    private static func matches(
        _ admission: DecodeConversationGate.Admission,
        epoch: UUID?,
        index: Int?
    ) -> Bool {
        guard case .turn(let admittedEpoch, let admittedIndex) = admission else {
            return false
        }
        return admittedEpoch == epoch && admittedIndex == index
    }

    private static func argument(after name: String) -> String? {
        let arguments = CommandLine.arguments
        guard let index = arguments.firstIndex(of: name),
              arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }

    private static func retireLaunchJob(_ label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
    }
}
