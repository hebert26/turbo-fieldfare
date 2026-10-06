import Darwin
import Foundation
import Synchronization
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

@Suite struct DecodeServiceConnectionInvalidationTests {
    @Test func terminationSendsShutdownAndDropsConnectionImmediately() throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })

        client.shutdownForTermination()

        #expect(!client.connectionIsInstalled)
        let command = try DecodeFrameCodec.read(
            DecodeServiceCommand.self,
            from: commands.fileHandleForReading)
        guard case .shutdown = command else {
            Issue.record("termination did not send the shutdown command")
            return
        }
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func responsiveTerminationDeliversCompleteShutdownAfterWriterGate() async throws {
        let sockets = try makeDupSocketPair()
        let writerEntered = AwaitOnce()
        let releaseWriter = DispatchSemaphore(value: 0)
        let client = DecodeServiceInferenceClient(
            testInput: sockets.clientInput,
            responseOutput: sockets.clientOutput,
            sendFrame: { frame, output in
                writerEntered.signal()
                _ = releaseWriter.wait(timeout: .now() + 5)
                try output.write(contentsOf: frame)
            },
            scheduleTerminationFallback: { _ in })
        let router = try #require(client.installedRouter)
        let waiter = Task { try await router.next(matching: UUID()) }

        client.shutdownForTermination()
        #expect(!client.connectionIsInstalled)
        await #expect(throws: (any Error).self) { try await waiter.value }
        await writerEntered.wait()
        releaseWriter.signal()

        let command = try await BlockingCommandReader(sockets.peer).next()
        guard case .shutdown = command else {
            Issue.record("peer did not receive a complete shutdown frame")
            return
        }
        #expect(await awaitEndOfFile(sockets.peer))
        try? sockets.peer.close()
        await awaitRouterClosure(router)
    }

    @Test func writerCapacityIncludesInFlightBytesAndSettlesReceiptOnce() throws {
        let pipe = Pipe()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let frameBytes = try DecodeFrameCodec.encode(DecodeServiceCommand.shutdown).count
        let writer = DecodeServiceCommandWriter(
            output: pipe.fileHandleForWriting,
            capacity: 4,
            maximumQueuedBytes: frameBytes,
            sendFrame: { frame, output in
                entered.signal()
                _ = release.wait(timeout: .now() + 5)
                try output.write(contentsOf: frame)
            })
        let receipt = try writer.enqueue(.shutdown)
        #expect(entered.wait(timeout: .now() + 2) == .success)
        #expect(throws: DecodeServiceCommandWriterError.queueFull) {
            try writer.enqueue(.shutdown)
        }
        release.signal()
        try receipt.waitBlocking()
        #expect(receipt.settlementCount == 1)
        writer.close()
        writer.waitUntilFinished()
        try? pipe.fileHandleForReading.close()
    }

    @Test func completedFrameReleasesCapacityBeforeReplacementEnqueue() throws {
        let pipe = Pipe()
        let firstEntered = DispatchSemaphore(value: 0)
        let releaseFirst = DispatchSemaphore(value: 0)
        let probe = ReplacementEnqueueProbe()
        let sendCount = Mutex(0)
        let frameBytes = try DecodeFrameCodec.encode(DecodeServiceCommand.shutdown).count
        let writer = DecodeServiceCommandWriter(
            output: pipe.fileHandleForWriting,
            capacity: 1,
            maximumQueuedBytes: frameBytes,
            sendFrame: { frame, output in
                let isFirst = sendCount.withLock { count in
                    defer { count += 1 }
                    return count == 0
                }
                if isFirst {
                    firstEntered.signal()
                    _ = releaseFirst.wait(timeout: .now() + 5)
                }
                try output.write(contentsOf: frame)
            },
            beforeReceiptPublication: { probe.observe() })
        probe.attach(writer)
        let first = try writer.enqueue(.shutdown)
        #expect(firstEntered.wait(timeout: .now() + 2) == .success)
        releaseFirst.signal()
        #expect(probe.invoked.wait(timeout: .now() + 2) == .success)
        let replacement = try #require(probe.replacementReceipt())
        try first.waitBlocking()
        try replacement.waitBlocking()
        #expect(first.settlementCount == 1)
        #expect(replacement.settlementCount == 1)
        #expect(probe.failure() == nil)
        writer.close()
        writer.waitUntilFinished()
        try? pipe.fileHandleForReading.close()
    }

    @Test func controlledWriterAbortSettlesReceiptAndWaiterExactlyOnce() throws {
        let pipe = Pipe()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let aborted = AbortMarker()
        let writer = DecodeServiceCommandWriter(
            output: pipe.fileHandleForWriting,
            sendFrame: { _, _ in
                entered.signal()
                _ = release.wait(timeout: .now() + 5)
                if aborted.value { throw TestAbortError.aborted }
            })
        let receipt = try writer.enqueue(.shutdown)
        #expect(entered.wait(timeout: .now() + 2) == .success)
        aborted.set()
        writer.requestTransportShutdown()
        release.signal()
        #expect(throws: TestAbortError.aborted) {
            try receipt.waitBlocking()
        }
        #expect(receipt.settlementCount == 1)
        writer.waitUntilFinished()
        try? pipe.fileHandleForReading.close()
    }

    @Test func controlledAbortSettlesThroughManualFallbackWithoutClaimingPeerBytes() async throws {
        let sockets = try makeDupSocketPair()
        let writerEntered = AwaitOnce()
        let releaseWriter = DispatchSemaphore(value: 0)
        let aborted = AbortMarker()
        let fallback = ActionBox()
        let client = DecodeServiceInferenceClient(
            testInput: sockets.clientInput,
            responseOutput: sockets.clientOutput,
            sendFrame: { frame, output in
                writerEntered.signal()
                _ = releaseWriter.wait(timeout: .now() + 5)
                if aborted.value { throw TestAbortError.aborted }
                try output.write(contentsOf: frame)
            },
            scheduleTerminationFallback: { action in fallback.store(action) })

        client.shutdownForTermination()
        await writerEntered.wait()
        let peerRead = Task {
            try await BlockingCommandReader(sockets.peer).next()
        }
        await fallback.invoke()
        await #expect(throws: (any Error).self) { try await peerRead.value }
        aborted.set()
        releaseWriter.signal()
        #expect(!client.connectionIsInstalled)
        try? sockets.peer.close()
    }

    @Test func preExistingQuarantineRefusesFreshLoadWithoutOrphaningIt() async throws {
        let initialCommands = Pipe()
        let initialResponses = Pipe()
        let attemptedFrames = Mutex(0)
        let sendFrame: DecodeServiceCommandWriter.SendFrame = { frame, output in
            attemptedFrames.withLock { $0 += 1 }
            try output.write(contentsOf: frame)
        }
        let client = DecodeServiceInferenceClient(
            testInput: initialCommands.fileHandleForWriting,
            responseOutput: initialResponses.fileHandleForReading,
            sendFrame: sendFrame,
            scheduleTerminationFallback: { _ in })
        let initialRouter = try #require(client.installedRouter)
        try? initialResponses.fileHandleForWriting.close()
        await awaitRouterClosure(initialRouter)
        await client.unload()
        let lifetime = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(), peer: DecodeServicePeerIdentity(pid: 101, pidVersion: 1))
        let commands = Pipe()
        let responses = Pipe()
        client.installTransport(
            input: commands.fileHandleForWriting,
            output: responses.fileHandleForReading,
            lifetime: lifetime,
            sendFrame: sendFrame)
        _ = try lifetime.admitFirstModelFrame { true }
        client.handleLifetimeEvent(lifetimeID: lifetime.id, event: .executionChanged)
        #expect(client.lifetimeCleanupIsQuarantined)
        try? responses.fileHandleForWriting.close()
        await client.unload()

        await #expect(throws: (any Error).self) {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-quarantine-successor"),
                maxContextTokens: 128, options: AppRuntimeOptions(),
                forceLogitsHead: false) { _ in }
        }
        #expect(attemptedFrames.withLock { $0 } == 0)

        client.handleLifetimeEvent(lifetimeID: lifetime.id, event: .exited)
        #expect(!client.lifetimeCleanupIsQuarantined)
        let successorCommands = Pipe()
        let successorResponses = Pipe()
        client.installTransport(
            input: successorCommands.fileHandleForWriting,
            output: successorResponses.fileHandleForReading,
            sendFrame: sendFrame)
        let successorReader = BlockingCommandReader(successorCommands.fileHandleForReading)
        let successorOptions = AppRuntimeOptions()
        let successor = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-quarantine-successor"),
                maxContextTokens: 128, options: successorOptions,
                forceLogitsHead: false) { _ in }
        }
        let successorCommand = try await successorReader.next()
        guard case let .load(request) = successorCommand else {
            Issue.record("successor did not receive a load command")
            return
        }
        try successorResponses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: request.requestID,
                loadAttemptID: request.attemptID, loadedFamily: .gemma4,
                loadID: UUID(),
                toolThinkingEnabled: successorOptions.toolThinkingEnabled)))
        try await successor.value
        #expect(attemptedFrames.withLock { $0 } == 1)
        client.shutdownForTermination()
        try? initialCommands.fileHandleForReading.close()
        try? initialResponses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
        try? responses.fileHandleForWriting.close()
        try? successorCommands.fileHandleForReading.close()
        try? successorResponses.fileHandleForWriting.close()
    }

    @Test func retirementReservationBlocksUnloadAndSuccessorSetupUntilCompletion() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let attemptedFrames = Mutex(0)
        let entered = AwaitOnce()
        let release = DispatchSemaphore(value: 0)
        let checkpoints = DecodeServiceRetirementCheckpoints(
            afterReservationBeforeRetirement: {
                entered.signal()
                _ = release.wait(timeout: .now() + 5)
            })
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            sendFrame: { frame, output in
                attemptedFrames.withLock { $0 += 1 }
                try output.write(contentsOf: frame)
            },
            scheduleTerminationFallback: { _ in },
            retirementCheckpoints: checkpoints)
        let shutdown = Task { client.shutdownForTermination() }
        await entered.wait()
        #expect(!client.connectionIsInstalled)

        let unloadStarted = AwaitOnce()
        let unloadFinished = Mutex(false)
        let unload = Task {
            unloadStarted.signal()
            await client.unload()
            unloadFinished.withLock { $0 = true }
        }
        await unloadStarted.wait()
        await #expect(throws: (any Error).self) {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-retirement-successor"),
                maxContextTokens: 128, options: AppRuntimeOptions(),
                forceLogitsHead: false) { _ in }
        }
        #expect(attemptedFrames.withLock { $0 } == 0)
        #expect(!unloadFinished.withLock { $0 })
        release.signal()
        await shutdown.value
        try? responses.fileHandleForWriting.close()
        await unload.value
        #expect(unloadFinished.withLock { $0 })
        #expect(attemptedFrames.withLock { $0 } == 1)
        try? commands.fileHandleForReading.close()
    }

    @Test func lostResetAcknowledgementDiscardsTheDeadConnection() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        #expect(client.connectionIsInstalled)
        weak var releasedRouter: DecodeServiceResponseRouter?
        releasedRouter = client.installedRouter
        var retainedRouter = client.installedRouter
        let commandReader = BlockingCommandReader(commands.fileHandleForReading)
        let options = AppRuntimeOptions()

        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-test-model"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let loadCommand = try await commandReader.next()
        guard case let .load(request) = loadCommand else {
            Issue.record("load handshake did not send a load command")
            return
        }
        let loadID = UUID()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: request.requestID,
                loadAttemptID: request.attemptID,
                loadedFamily: .gemma4, loadID: loadID,
                modelIdentity: nil,
                toolThinkingEnabled: options.toolThinkingEnabled)))
        try await load.value

        let epoch = UUID()
        let reset = Task { try await client.resetConversation(epoch: epoch) }
        let resetCommand = try await commandReader.next()
        guard case let .resetConversation(resetRequest) = resetCommand else {
            Issue.record("reset handshake did not send a reset command")
            return
        }
        #expect(resetRequest.loadID == loadID)
        try? responses.fileHandleForWriting.close()

        await #expect(throws: (any Error).self) { try await reset.value }
        if let retainedRouter {
            await awaitRouterClosure(retainedRouter)
        }
        retainedRouter = nil
        let releaseDeadline = ContinuousClock.now + .seconds(2)
        while releasedRouter != nil, ContinuousClock.now < releaseDeadline {
            await Task.yield()
        }
        #expect(!client.connectionIsInstalled,
                "the router ended but its handles stayed installed")
        #expect(releasedRouter == nil,
                "the invalidated connection retained its response router")
        try? commands.fileHandleForReading.close()
    }

    @Test func ownerTerminationDominatesAQueuedStaleEvent() async throws {
        let responses = Pipe()
        let queued = AwaitOnce()
        let router = DecodeServiceResponseRouter(
            output: responses.fileHandleForReading,
            onEventQueued: { _ in queued.signal() })
        let generationID = UUID()
        let event = DecodeServiceEvent(kind: .snapshot, generationID: generationID)
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(event))
        await queued.wait()
        router.terminateWaiters(with: TestAbortError.aborted)

        await #expect(throws: TestAbortError.aborted) {
            try await router.next(matching: generationID)
        }
        try? responses.fileHandleForWriting.close()
        await awaitRouterClosure(router)
    }

    @Test func nativeObserverReceivesExitAndClosesItsOwnedDescriptor() async throws {
        let childInput = Pipe()
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/cat")
        child.standardInput = childInput
        child.standardOutput = FileHandle.nullDevice
        try child.run()

        let events = ObserverEventRecorder()
        let eventReceived = AwaitOnce()
        let phases = ObserverPhaseRecorder()
        let observer = ServiceProcessObserver(
            pid: child.processIdentifier,
            checkpoints: ServiceProcessObserverCheckpoints(
                beforeCreate: { phases.append("create") },
                beforeUserRegistration: { phases.append("user") },
                beforeProcessRegistration: { phases.append("process") },
                beforeWaitPublication: { phases.append("wait") },
                beforeClose: { phases.append("close") }),
            onEvent: { event in
                events.append(event)
                eventReceived.signal()
            })
        observer.start()
        #expect(observer.waitForRegistration() == .success)
        #expect(phases.snapshot() == ["create", "user", "process", "wait"])

        try childInput.fileHandleForWriting.close()
        await eventReceived.wait()
        await awaitObserverClosed(observer)
        await awaitProcessExit(child)
        #expect(events.snapshot() == [.exited])
        #expect(phases.snapshot().last == "close")
    }

    @Test func observerCancellationUsesUserWakeWithoutReportingProcessDeath() async throws {
        let childInput = Pipe()
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/cat")
        child.standardInput = childInput
        child.standardOutput = FileHandle.nullDevice
        try child.run()

        let events = ObserverEventRecorder()
        let observer = ServiceProcessObserver(
            pid: child.processIdentifier,
            onEvent: { event in events.append(event) })
        observer.start()
        #expect(observer.waitForRegistration() == .success)
        observer.requestCancellation()
        await awaitObserverClosed(observer)
        #expect(events.snapshot().isEmpty)
        try childInput.fileHandleForWriting.close()
        await awaitProcessExit(child)
    }

    private func makeDupSocketPair() throws -> (
        clientInput: FileHandle, clientOutput: FileHandle, peer: FileHandle
    ) {
        var fds = [Int32](repeating: -1, count: 2)
        guard socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0 else {
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }
        let inputFD = dup(fds[0])
        let outputFD = dup(fds[0])
        guard inputFD >= 0, outputFD >= 0 else {
            Darwin.close(fds[0])
            Darwin.close(fds[1])
            if inputFD >= 0 { Darwin.close(inputFD) }
            if outputFD >= 0 { Darwin.close(outputFD) }
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }
        Darwin.close(fds[0])
        return (
            FileHandle(fileDescriptor: inputFD, closeOnDealloc: true),
            FileHandle(fileDescriptor: outputFD, closeOnDealloc: true),
            FileHandle(fileDescriptor: fds[1], closeOnDealloc: true))
    }
}

private final class ObserverEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [ServiceProcessObserver.Event] = []

    func append(_ event: ServiceProcessObserver.Event) {
        lock.withLock { events.append(event) }
    }

    func snapshot() -> [ServiceProcessObserver.Event] {
        lock.withLock { events }
    }
}

private final class ObserverPhaseRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var phases: [String] = []

    func append(_ phase: String) {
        lock.withLock { phases.append(phase) }
    }

    func snapshot() -> [String] {
        lock.withLock { phases }
    }
}

private enum TestAbortError: Error, Equatable {
    case aborted
}

private final class ReplacementEnqueueProbe: @unchecked Sendable {
    private let lock = NSLock()
    private weak var writer: DecodeServiceCommandWriter?
    private var attempted = false
    private(set) var replacement: DecodeServiceWriteReceipt?
    private(set) var error: Error?
    let invoked = DispatchSemaphore(value: 0)

    func attach(_ writer: DecodeServiceCommandWriter) {
        lock.lock()
        self.writer = writer
        lock.unlock()
    }

    func replacementReceipt() -> DecodeServiceWriteReceipt? {
        lock.lock()
        defer { lock.unlock() }
        return replacement
    }

    func failure() -> Error? {
        lock.lock()
        defer { lock.unlock() }
        return error
    }

    func observe() {
        lock.lock()
        guard !attempted else {
            lock.unlock()
            return
        }
        attempted = true
        let writer = self.writer
        lock.unlock()
        guard let writer else {
            invoked.signal()
            return
        }
        do {
            let replacement = try writer.enqueue(.shutdown)
            lock.lock()
            self.replacement = replacement
            lock.unlock()
        } catch {
            lock.lock()
            self.error = error
            lock.unlock()
        }
        invoked.signal()
    }
}

private final class AbortMarker: @unchecked Sendable {
    private let lock = NSLock()
    private var aborted = false

    var value: Bool {
        lock.lock()
        defer { lock.unlock() }
        return aborted
    }

    func set() {
        lock.lock()
        aborted = true
        lock.unlock()
    }
}

private final class ActionBox: @unchecked Sendable {
    private let lock = NSLock()
    private var action: (@Sendable () -> Void)?
    private let available = AwaitOnce()

    func store(_ action: @escaping @Sendable () -> Void) {
        lock.lock()
        self.action = action
        lock.unlock()
        available.signal()
    }

    func invoke() async {
        await available.wait()
        let action = lock.withLock {
            defer { self.action = nil }
            return self.action
        }
        action?()
    }
}

private final class AwaitOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var signaled = false

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

private func awaitEndOfFile(_ handle: FileHandle) async -> Bool {
    let finished = DispatchSemaphore(value: 0)
    let result = EOFReadResult()
    let reader = Thread {
        do {
            var hasTrailingBytes = false
            while true {
                guard let data = try handle.read(upToCount: 1), !data.isEmpty else {
                    result.set(!hasTrailingBytes)
                    break
                }
                hasTrailingBytes = true
            }
        } catch {
            result.set(false)
        }
        finished.signal()
    }
    reader.name = "TurboFieldfare.Tests.PeerEOFReader"
    reader.start()
    return await withCheckedContinuation { continuation in
        let waiter = Thread {
            let completed = finished.wait(timeout: .now() + 2)
            continuation.resume(returning: completed == .success && result.value)
        }
        waiter.name = "TurboFieldfare.Tests.PeerEOFWaiter"
        waiter.start()
    }
}

private final class EOFReadResult: @unchecked Sendable {
    private let lock = NSLock()
    private var reachedEOF = false

    var value: Bool { lock.withLock { reachedEOF } }

    func set(_ value: Bool) {
        lock.withLock { reachedEOF = value }
    }
}

private func awaitObserverClosed(_ observer: ServiceProcessObserver) async {
    await withCheckedContinuation { continuation in
        let thread = Thread {
            observer.waitUntilClosed()
            continuation.resume()
        }
        thread.name = "TurboFieldfare.Tests.ObserverClosureWaiter"
        thread.start()
    }
}

private func awaitProcessExit(_ process: Process) async {
    await withCheckedContinuation { continuation in
        let thread = Thread {
            process.waitUntilExit()
            continuation.resume()
        }
        thread.name = "TurboFieldfare.Tests.ProcessExitWaiter"
        thread.start()
    }
}

private func awaitRouterClosure(_ router: DecodeServiceResponseRouter) async {
    await withCheckedContinuation { continuation in
        let thread = Thread {
            router.waitUntilTransportClosed()
            continuation.resume()
        }
        thread.name = "TurboFieldfare.Tests.RouterClosureWaiter"
        thread.start()
    }
}

private final class BlockingCommandReader: @unchecked Sendable {
    private let handle: FileHandle

    init(_ handle: FileHandle) { self.handle = handle }

    func next() async throws -> DecodeServiceCommand {
        try await withCheckedThrowingContinuation { continuation in
            let handle = self.handle
            let thread = Thread {
                do {
                    continuation.resume(returning: try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handle))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            thread.name = "TurboFieldfare.Tests.BlockingCommandReader"
            thread.start()
        }
    }
}
