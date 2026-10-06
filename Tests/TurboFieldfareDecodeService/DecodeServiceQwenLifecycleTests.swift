import Foundation
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareDecodeService
import TurboFieldfareDecodeProtocol

@Suite struct DecodeServiceQwenLifecycleTests {
    @Test func gemmaReadinessBindsFamilyWithoutQwenIdentity() throws {
        let session = DecodeServiceSession()
        let loadID = UUID()
        let binding = try session.publish(
            .gemma(toolThinkingEnabled: true), loadID: loadID)

        #expect(binding.loadID == loadID)
        #expect(binding.family == .gemma4)
        #expect(binding.modelIdentity == nil)
        #expect(binding.toolThinkingEnabled == true)
        #expect(session.currentBinding == binding)
    }

    @Test func successfulLoadsMintFreshIncarnationsIndependentOfRequestIDs() throws {
        let session = DecodeServiceSession()
        let requestID = UUID()
        let readiness = AppLoadedModelReadiness.gemma(toolThinkingEnabled: true)
        let first = try session.publish(readiness)
        let second = try session.publish(readiness)

        #expect(first.loadID != requestID)
        #expect(second.loadID != requestID)
        #expect(second.loadID != first.loadID)
    }

    @Test func qwenReadinessRetainsIdentityAndRejectsInconsistentFamily() throws {
        let session = DecodeServiceSession()
        let identity = modelIdentity()
        let loadID = UUID()
        let binding = try session.publish(
            .qwen(identity: identity), loadID: loadID)

        #expect(binding.family == .qwen3_6)
        #expect(binding.modelIdentity == identity)
        #expect(binding.toolThinkingEnabled == nil)
        #expect(session.binding(loadID: loadID) == binding)

        var inconsistent = identity
        inconsistent.family = .gemma4
        #expect(throws: DecodeServiceSession.Rejection.inconsistentReadiness) {
            try session.publish(.qwen(identity: inconsistent), loadID: UUID())
        }
        #expect(session.currentBinding == binding)
    }

    @Test func operationsCannotCrossAReplacedLoadIncarnation() throws {
        let session = DecodeServiceSession()
        let firstLoad = UUID()
        let firstOperation = UUID()
        _ = try session.publish(.gemma(toolThinkingEnabled: false), loadID: firstLoad)
        let firstStop = try #require(
            session.prepare(loadID: firstLoad, operationID: firstOperation))

        let secondLoad = UUID()
        _ = try session.publish(.gemma(toolThinkingEnabled: true), loadID: secondLoad)
        #expect(session.binding(loadID: firstLoad) == nil)
        #expect(session.prepare(loadID: firstLoad, operationID: firstOperation) == nil)
        #expect(!session.requestStop(loadID: firstLoad, operationID: firstOperation))
        #expect(session.stop(loadID: firstLoad, operationID: firstOperation) == nil)
        #expect(!firstStop.isRequested)

        let secondOperation = UUID()
        let secondStop = try #require(
            session.prepare(loadID: secondLoad, operationID: secondOperation))
        #expect(session.requestStop(
            loadID: secondLoad, operationID: secondOperation))
        #expect(secondStop.isRequested)
    }

    @Test func legacyNilAttemptReceivesOpaqueLeaseWithoutCancellationAuthority() throws {
        let session = DecodeServiceSession()
        let requestID = UUID()
        let request = DecodeLoadRequest(
            modelPath: "/tmp/legacy-gemma", maxContextTokens: 128,
            requestID: requestID, attemptID: nil)
        let lease = try #require(session.registerLoad(request))

        #expect(lease.requestID == requestID)
        #expect(lease.attemptID == nil)
        #expect(session.loadLease(for: request) == lease)
        #expect(!session.markCancel(DecodeCancelLoadRequest(
            requestID: requestID, attemptID: UUID())))
        #expect(!session.wasCancellationRequested(lease))
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/next", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) == nil)
    }

    @Test func coordinatorCancellationWhilePreparationAwaitsJoinsRuntimeUnload() async throws {
        let session = DecodeServiceSession()
        let request = DecodeLoadRequest(
            modelPath: "/tmp/cancel-prepare", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())
        let lease = try #require(session.registerLoad(request))
        let entered = AwaitOnce()
        let release = AwaitOnce()
        let runtime = LifecycleFakeRuntime(
            readiness: .gemma(toolThinkingEnabled: false),
            ensureEntered: entered, ensureRelease: release)
        let coordinator = DecodeServiceLoadCoordinator(
            session: session, runtime: runtime)
        let attempt = try coordinator.beginReplacement(lease)
        let preparation = Task {
            try await coordinator.prepare(
                attempt, directory: URL(fileURLWithPath: request.modelPath),
                maxContextTokens: request.maxContextTokens,
                options: AppRuntimeOptions(), forceLogitsHead: false)
        }
        await entered.wait()

        let attemptID = try #require(request.attemptID)
        #expect(session.markCancel(DecodeCancelLoadRequest(
            requestID: request.requestID, attemptID: attemptID)))
        #expect(session.wasCancellationRequested(lease))
        let cancellation = Task {
            await coordinator.cancel(DecodeCancelLoadRequest(
                requestID: request.requestID, attemptID: attemptID))
        }
        await Task.yield()
        release.signal()
        await cancellation.value
        await #expect(throws: (any Error).self) { try await preparation.value }
        #expect(runtime.unloadCount == 1)
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/after-cancel", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) != nil)
    }

    @Test func readyPublicationTransfersPendingTeardownAndBlocksSuccessorUntilFinish() async throws {
        let session = DecodeServiceSession()
        let request = DecodeLoadRequest(
            modelPath: "/tmp/ready-teardown", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())
        let lease = try #require(session.registerLoad(request))
        let runtime = LifecycleFakeRuntime(
            readiness: .qwen(identity: modelIdentity()))
        let coordinator = DecodeServiceLoadCoordinator(
            session: session, runtime: runtime)
        let attempt = try coordinator.beginReplacement(lease)
        let readiness = try await coordinator.prepare(
            attempt, directory: URL(fileURLWithPath: request.modelPath),
            maxContextTokens: request.maxContextTokens,
            options: AppRuntimeOptions(), forceLogitsHead: false)
        let binding = try coordinator.reservePublication(attempt, readiness: readiness)
        let teardown = try #require(session.registerBoundTeardown(
            requestID: UUID(), loadID: binding.loadID))
        let result = try coordinator.finishCommit(attempt)
        guard case let .teardown(committedTeardown, committedBinding) = result else {
            Issue.record("ready publication did not transfer pending teardown")
            return
        }
        #expect(committedTeardown == teardown)
        #expect(committedBinding == binding)
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/successor", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) == nil)
        #expect(session.finishTeardown(teardown) == binding)
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/successor", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) != nil)
    }

    @Test func cancellationAfterPublicationCannotRevokeReadyBinding() async throws {
        let session = DecodeServiceSession()
        let request = DecodeLoadRequest(
            modelPath: "/tmp/post-reserve", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())
        let lease = try #require(session.registerLoad(request))
        let runtime = LifecycleFakeRuntime(
            readiness: .gemma(toolThinkingEnabled: true))
        let coordinator = DecodeServiceLoadCoordinator(
            session: session, runtime: runtime)
        let attempt = try coordinator.beginReplacement(lease)
        let readiness = try await coordinator.prepare(
            attempt, directory: URL(fileURLWithPath: request.modelPath),
            maxContextTokens: request.maxContextTokens,
            options: AppRuntimeOptions(), forceLogitsHead: false)
        let binding = try coordinator.reservePublication(attempt, readiness: readiness)
        let attemptID = try #require(request.attemptID)
        #expect(!session.markCancel(DecodeCancelLoadRequest(
            requestID: request.requestID, attemptID: attemptID)))
        guard case let .bound(committed) = try coordinator.finishCommit(attempt) else {
            Issue.record("ready publication was not committed")
            return
        }
        #expect(committed == binding)
        #expect(session.currentBinding == binding)
        #expect(runtime.unloadCount == 0)
    }

    @Test func bareOrLegacyOperationControlsCannotMutateBoundSession() throws {
        let session = DecodeServiceSession()
        let loadID = UUID()
        let operationID = UUID()
        _ = try session.publish(.gemma(toolThinkingEnabled: false), loadID: loadID)

        #expect(session.prepare(loadID: nil, operationID: operationID) == nil)
        #expect(!session.requestStop(loadID: nil, operationID: operationID))
        #expect(session.stop(loadID: nil, operationID: operationID) == nil)
        #expect(session.binding(loadID: nil) == nil)
    }

    @Test func queuedRejectedLoadRetainsRefusalAcrossPairReuse() throws {
        let session = DecodeServiceSession()
        let queue = DecodeCommandQueue()
        let requestID = UUID()
        let attemptID = UUID()
        let requestA = DecodeLoadRequest(
            modelPath: "/tmp/p17-queue-a", maxContextTokens: 128,
            requestID: requestID, attemptID: attemptID)
        let requestB = DecodeLoadRequest(
            modelPath: "/tmp/p17-queue-b", maxContextTokens: 128,
            requestID: requestID, attemptID: attemptID)
        let requestC = DecodeLoadRequest(
            modelPath: "/tmp/p17-queue-c", maxContextTokens: 128,
            requestID: requestID, attemptID: attemptID)

        let admittedA = DecodeQueuedCommand.admitting(.load(requestA), using: session)
        queue.append(admittedA)
        let refusedB = DecodeQueuedCommand.admitting(.load(requestB), using: session)
        queue.append(refusedB)

        guard case let .load(leaseA?) = admittedA.admission else {
            Issue.record("first load was not admitted")
            return
        }
        guard case .load(nil) = refusedB.admission else {
            Issue.record("duplicate load was not retained as a refusal")
            return
        }

        // Consume A as Entry does, then complete its lifecycle while B remains
        // queued. C is admitted with the same caller pair before B is consumed.
        guard let consumedA = queue.next() else {
            Issue.record("first queued load was missing")
            return
        }
        guard case let .load(leaseFromQueue?) = consumedA.admission else {
            Issue.record("consumed first load lost its lease")
            return
        }
        #expect(leaseFromQueue == leaseA)
        try session.requireActive(leaseFromQueue)
        _ = try session.reserveAndPublish(
            .gemma(toolThinkingEnabled: false), lease: leaseFromQueue)
        _ = try session.finishCommit(leaseFromQueue)

        let admittedC = DecodeQueuedCommand.admitting(.load(requestC), using: session)
        queue.append(admittedC)
        guard case let .load(leaseC?) = admittedC.admission else {
            Issue.record("successor load was not admitted")
            return
        }

        var admittedPaths: [String] = []
        var refusedPaths: [String] = []
        for _ in 0..<2 {
            guard let queued = queue.next() else {
                Issue.record("queued successor items were missing")
                continue
            }
            guard case let .load(lease) = queued.admission,
                  case let .load(request) = queued.command else {
                Issue.record("load queue item had an unexpected shape")
                continue
            }
            guard let lease else {
                refusedPaths.append(request.modelPath)
                continue
            }
            try session.requireActive(lease)
            admittedPaths.append(request.modelPath)
            #expect(lease == leaseC)
            _ = try session.reserveAndPublish(
                .gemma(toolThinkingEnabled: true), lease: lease)
            _ = try session.finishCommit(lease)
        }

        #expect(admittedPaths == [requestC.modelPath])
        #expect(refusedPaths == [requestB.modelPath])
        #expect(!admittedPaths.contains(requestB.modelPath))
    }

    @Test func queuedRejectedBoundUnloadCannotRunAfterBindingPairReuse() throws {
        let session = DecodeServiceSession()
        let queue = DecodeCommandQueue()
        let loadID = UUID()
        let requestID = UUID()
        _ = try session.publish(.gemma(toolThinkingEnabled: false), loadID: loadID)
        let request = DecodeUnloadRequest(requestID: requestID, loadID: loadID)

        let admittedA = DecodeQueuedCommand.admitting(.unloadBound(request), using: session)
        queue.append(admittedA)
        let refusedB = DecodeQueuedCommand.admitting(.unloadBound(request), using: session)
        queue.append(refusedB)
        guard case let .unloadBound(leaseA?) = admittedA.admission,
              case .unloadBound(nil) = refusedB.admission else {
            Issue.record("bound unload admission did not preserve A versus B")
            return
        }

        guard let consumedA = queue.next(),
              case let .unloadBound(leaseFromQueue?) = consumedA.admission else {
            Issue.record("first bound unload was missing its lease")
            return
        }
        #expect(leaseFromQueue == leaseA)
        guard session.binding(for: leaseFromQueue) != nil else {
            Issue.record("first bound unload lost its binding")
            return
        }
        #expect(session.finishTeardown(leaseFromQueue) != nil)

        // Reusing the same request/load pair makes a later pair lookup return C.
        // The queued B envelope must remain refused instead of acquiring C.
        _ = try session.publish(.gemma(toolThinkingEnabled: true), loadID: loadID)
        let admittedC = DecodeQueuedCommand.admitting(.unloadBound(request), using: session)
        queue.append(admittedC)
        guard case let .unloadBound(leaseC?) = admittedC.admission else {
            Issue.record("successor bound unload was not admitted")
            return
        }
        #expect(session.teardownLease(for: request) == leaseC)

        var executed: [DecodeServiceSession.TeardownLease] = []
        var refusedCount = 0
        for _ in 0..<2 {
            guard let queued = queue.next() else {
                Issue.record("queued successor unloads were missing")
                continue
            }
            guard case let .unloadBound(lease) = queued.admission else {
                Issue.record("bound unload queue item had an unexpected shape")
                continue
            }
            guard let lease else {
                refusedCount += 1
                continue
            }
            executed.append(lease)
            #expect(lease == leaseC)
            #expect(session.binding(for: lease) != nil)
            #expect(session.finishTeardown(lease) != nil)
        }

        #expect(executed == [leaseC])
        #expect(refusedCount == 1)
    }

    @Test func readyBytesPrecedeCommitAndTransferPendingBoundUnload() async throws {
        let session = DecodeServiceSession()
        let request = DecodeLoadRequest(
            modelPath: "/tmp/p17-ready-commit", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())
        let lease = try #require(session.registerLoad(request))
        let runtime = LifecycleFakeRuntime(
            readiness: .gemma(toolThinkingEnabled: false))
        let coordinator = DecodeServiceLoadCoordinator(
            session: session, runtime: runtime)
        let attempt = try coordinator.beginReplacement(lease)
        let binding = try coordinator.reservePublication(
            attempt, readiness: .gemma(toolThinkingEnabled: false))
        let event = DecodeServiceEvent(
            kind: .ready, generationID: request.requestID,
            loadAttemptID: request.attemptID, loadedFamily: binding.family,
            loadID: binding.loadID, toolThinkingEnabled: false)
        let output = Pipe()
        let visible = AwaitOnce()
        let received = ReadyFrameCapture()
        let peer = Thread {
            do {
                let decoded = try DecodeFrameCodec.read(
                    DecodeServiceEvent.self, from: output.fileHandleForReading)
                received.record(decoded)
            } catch {
                received.record(error)
            }
            visible.signal()
        }
        peer.name = "TurboFieldfare.Tests.ReadyCommitPeer"
        peer.start()

        let releaseWrite = DispatchSemaphore(value: 0)
        let publication = Task {
            await DecodeServiceReadyPublisher.publish(
                event: event,
                to: output.fileHandleForWriting,
                attempt: attempt,
                coordinator: coordinator,
                writeFrame: { frame, output in
                    try output.write(contentsOf: frame)
                    _ = releaseWrite.wait()
                })
        }
        defer { releaseWrite.signal() }

        await visible.wait()
        guard case let .event(decoded) = received.value else {
            Issue.record("ready peer did not decode the framed event")
            releaseWrite.signal()
            _ = await publication.value
            return
        }
        #expect(decoded.kind == event.kind)
        #expect(decoded.generationID == event.generationID)
        #expect(decoded.loadAttemptID == event.loadAttemptID)
        #expect(decoded.loadedFamily == event.loadedFamily)
        #expect(decoded.loadID == event.loadID)
        #expect(decoded.toolThinkingEnabled == event.toolThinkingEnabled)
        let attemptID = try #require(request.attemptID)
        #expect(!session.markCancel(DecodeCancelLoadRequest(
            requestID: request.requestID, attemptID: attemptID)))
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/p17-competing-load", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) == nil)

        let unloadRequest = DecodeUnloadRequest(
            requestID: UUID(), loadID: binding.loadID)
        let pendingTeardown = try #require(session.registerBoundTeardown(
            requestID: unloadRequest.requestID, loadID: unloadRequest.loadID))
        #expect(session.teardownLease(for: unloadRequest) == pendingTeardown)
        releaseWrite.signal()
        let outcome = await publication.value
        guard case let .committed(commit) = outcome,
              case let .teardown(transferred, transferredBinding) = commit else {
            Issue.record("ready commit did not transfer the pending teardown")
            return
        }
        #expect(transferred == pendingTeardown)
        #expect(transferredBinding == binding)
        #expect(session.finishTeardown(transferred) == binding)
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/p17-after-teardown", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) != nil)
        try? output.fileHandleForReading.close()
        try? output.fileHandleForWriting.close()
    }

    @Test func readyWriteFailuresJoinCleanupBeforeClosingAndConsumePendingTeardown() async throws {
        for partial in [true, false] {
            try await exerciseReadyWriteFailure(partial: partial)
        }
    }

    @Test func absentOrStaleBoundUnloadCannotReserveOrBlockAValidLoad() throws {
        let invalidRequests = [
            DecodeUnloadRequest(requestID: UUID(), loadID: nil),
            DecodeUnloadRequest(requestID: UUID(), loadID: UUID()),
        ]
        for request in invalidRequests {
            let session = DecodeServiceSession()
            _ = try session.publish(.gemma(toolThinkingEnabled: false), loadID: UUID())
            let queued = DecodeQueuedCommand.admitting(
                .unloadBound(request), using: session)
            guard case .unloadBound(nil) = queued.admission else {
                Issue.record("invalid bound unload unexpectedly reserved a lease")
                continue
            }
            #expect(session.teardownLease(for: request) == nil)
            #expect(session.registerLoad(DecodeLoadRequest(
                modelPath: "/tmp/p17-after-invalid-unload", maxContextTokens: 128,
                requestID: UUID(), attemptID: UUID())) != nil)
        }
    }

    @Test func retiringOnlyTheBoundLoadReleasesItsOperations() throws {
        let session = DecodeServiceSession()
        let loadID = UUID()
        let operationID = UUID()
        _ = try session.publish(.gemma(toolThinkingEnabled: false), loadID: loadID)
        let stop = try #require(session.prepare(
            loadID: loadID, operationID: operationID))

        #expect(session.retire(loadID: UUID()) == nil)
        #expect(session.currentBinding != nil)
        #expect(session.retire(loadID: loadID)?.loadID == loadID)
        #expect(session.currentBinding == nil)
        #expect(!session.requestStop(loadID: loadID, operationID: operationID))
        #expect(!stop.isRequested)
    }

    private func modelIdentity() -> DecodeModelIdentity {
        DecodeModelIdentity(
            family: .qwen3_6,
            modelID: "Qwen/Qwen3.6-35B-A3B",
            sourceRevision: "revision",
            formatMajor: 2,
            formatMinor: 0,
            sourceIndexSHA256: String(repeating: "a", count: 64),
            quantizationPolicySHA256: String(repeating: "b", count: 64),
            textManifestSHA256: String(repeating: "c", count: 64),
            quantization: [DecodeQuantizationIdentity(
                category: "recurrent_state", storage: "fp32")],
            vision: .unavailable)
    }

    private func exerciseReadyWriteFailure(partial: Bool) async throws {
        let session = DecodeServiceSession()
        let request = DecodeLoadRequest(
            modelPath: "/tmp/p17-ready-failure", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())
        let lease = try #require(session.registerLoad(request))
        let unloadEntered = AwaitOnce()
        let unloadRelease = AwaitOnce()
        let runtime = LifecycleFakeRuntime(
            readiness: .gemma(toolThinkingEnabled: false),
            unloadEntered: unloadEntered, unloadRelease: unloadRelease)
        let coordinator = DecodeServiceLoadCoordinator(
            session: session, runtime: runtime)
        let attempt = try coordinator.beginReplacement(lease)
        let binding = try coordinator.reservePublication(
            attempt, readiness: .gemma(toolThinkingEnabled: false))
        let unloadRequest = DecodeUnloadRequest(
            requestID: UUID(), loadID: binding.loadID)
        let pendingTeardown = try #require(session.registerBoundTeardown(
            requestID: unloadRequest.requestID, loadID: unloadRequest.loadID))
        #expect(session.teardownLease(for: unloadRequest) == pendingTeardown)
        let event = DecodeServiceEvent(
            kind: .ready, generationID: request.requestID,
            loadAttemptID: request.attemptID, loadedFamily: binding.family,
            loadID: binding.loadID, toolThinkingEnabled: false)
        let output = Pipe()
        let bytesVisible = AwaitOnce()
        let readerFinished = AwaitOnce()
        let received = ReadyRawFrameCapture()
        let submitted = ReadyWriteCapture()
        let reader = Thread {
            do {
                let first = try output.fileHandleForReading.read(upToCount: 1)
                received.append(first ?? Data())
                bytesVisible.signal()
                received.append(try output.fileHandleForReading.readToEnd() ?? Data())
            } catch {
                bytesVisible.signal()
                received.record(error)
            }
            readerFinished.signal()
        }
        reader.name = "TurboFieldfare.Tests.ReadyFailurePeer"
        reader.start()

        let releaseWrite = DispatchSemaphore(value: 0)
        let publication = Task {
            await DecodeServiceReadyPublisher.publish(
                event: event,
                to: output.fileHandleForWriting,
                attempt: attempt,
                coordinator: coordinator,
                writeFrame: { bytes, output in
                    submitted.record(bytes)
                    let count = partial ? max(1, bytes.count / 2) : bytes.count
                    try output.write(contentsOf: bytes.prefix(count))
                    _ = releaseWrite.wait()
                    throw ReadyWriteFailure.injected
                })
        }
        defer { releaseWrite.signal() }

        await bytesVisible.wait()
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/p17-competing-during-ready-failure", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) == nil)
        releaseWrite.signal()
        await unloadEntered.wait()
        #expect(!readerFinished.isSignaled)
        unloadRelease.signal()
        let outcome = await publication.value
        #expect(outcome == .deliveryFailed)
        await readerFinished.wait()
        #expect(received.error == nil)
        let submittedFrame = try #require(submitted.frame)
        let expectedBytes = partial
            ? Data(submittedFrame.prefix(max(1, submittedFrame.count / 2)))
            : submittedFrame
        #expect(received.data == expectedBytes)
        if !partial {
            let decoded = try JSONDecoder().decode(
                DecodeServiceEvent.self,
                from: Data(received.data.dropFirst(4)))
            #expect(decoded.kind == event.kind)
            #expect(decoded.generationID == event.generationID)
            #expect(decoded.loadAttemptID == event.loadAttemptID)
            #expect(decoded.loadID == event.loadID)
        }
        #expect(runtime.unloadCount == 1)
        #expect(session.teardownLease(for: unloadRequest) == nil)
        #expect(session.registerLoad(DecodeLoadRequest(
            modelPath: "/tmp/p17-after-ready-failure", maxContextTokens: 128,
            requestID: UUID(), attemptID: UUID())) != nil)
        try? output.fileHandleForReading.close()
        try? output.fileHandleForWriting.close()
    }
}

private final class LifecycleFakeRuntime: DecodeServiceModelRuntime, @unchecked Sendable {
    let readiness: AppLoadedModelReadiness
    let ensureEntered: AwaitOnce?
    let ensureRelease: AwaitOnce?
    let unloadEntered: AwaitOnce?
    let unloadRelease: AwaitOnce?
    private let lock = NSLock()
    private var unloads = 0

    init(
        readiness: AppLoadedModelReadiness,
        ensureEntered: AwaitOnce? = nil,
        ensureRelease: AwaitOnce? = nil,
        unloadEntered: AwaitOnce? = nil,
        unloadRelease: AwaitOnce? = nil
    ) {
        self.readiness = readiness
        self.ensureEntered = ensureEntered
        self.ensureRelease = ensureRelease
        self.unloadEntered = unloadEntered
        self.unloadRelease = unloadRelease
    }

    var unloadCount: Int {
        lock.withLock { unloads }
    }

    func ensureLoaded(
        modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {
        ensureEntered?.signal()
        if let ensureRelease { await ensureRelease.wait() }
    }

    func resetConversation(epoch: UUID) async throws {}

    var loadedModelReadiness: AppLoadedModelReadiness? {
        get async { readiness }
    }

    func unload() async {
        unloadEntered?.signal()
        if let unloadRelease { await unloadRelease.wait() }
        lock.withLock { unloads += 1 }
    }
}

private final class ReadyFrameCapture: @unchecked Sendable {
    enum Value {
        case event(DecodeServiceEvent)
        case error
    }

    private let lock = NSLock()
    private var stored: Value?

    var value: Value {
        lock.withLock { stored ?? .error }
    }

    func record(_ event: DecodeServiceEvent) {
        lock.withLock { stored = .event(event) }
    }

    func record(_: Error) {
        lock.withLock { stored = .error }
    }
}

private final class ReadyRawFrameCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var bytes = Data()
    private(set) var error: Error?

    var data: Data { lock.withLock { bytes } }

    func append(_ data: Data) {
        lock.withLock { bytes.append(data) }
    }

    func record(_ error: Error) {
        lock.withLock { self.error = error }
    }
}

private final class ReadyWriteCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var storedFrame: Data?

    var frame: Data? { lock.withLock { storedFrame } }

    func record(_ frame: Data) {
        lock.withLock { storedFrame = frame }
    }
}

private enum ReadyWriteFailure: Error, Equatable {
    case injected
}

private final class AwaitOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var signaled = false

    var isSignaled: Bool {
        lock.withLock { signaled }
    }

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if signaled {
                lock.unlock()
                continuation.resume()
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }
    }

    func signal() {
        lock.lock()
        signaled = true
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume()
    }
}
