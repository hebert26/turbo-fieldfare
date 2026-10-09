import Darwin
import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareDecodeProtocol

enum DecodeServiceCommandWriterError: Error, Equatable {
    case closed
    case queueFull
}

final class DecodeServiceWriteReceipt: @unchecked Sendable {
    private let condition = NSCondition()
    private var result: Result<Void, Error>?
    private var waiters: [CheckedContinuation<Void, Error>] = []
    private var settlements = 0

    var settlementCount: Int {
        condition.lock()
        defer { condition.unlock() }
        return settlements
    }

    func settle(_ result: Result<Void, Error>) {
        condition.lock()
        guard self.result == nil else {
            condition.unlock()
            return
        }
        self.result = result
        settlements += 1
        let waiters = self.waiters
        self.waiters.removeAll()
        condition.broadcast()
        condition.unlock()
        for waiter in waiters {
            switch result {
            case .success: waiter.resume()
            case .failure(let error): waiter.resume(throwing: error)
            }
        }
    }

    func wait() async throws {
        try await withCheckedThrowingContinuation { continuation in
            condition.lock()
            if let result {
                condition.unlock()
                switch result {
                case .success: continuation.resume()
                case .failure(let error): continuation.resume(throwing: error)
                }
            } else {
                waiters.append(continuation)
                condition.unlock()
            }
        }
    }

    func waitBlocking() throws {
        condition.lock()
        while result == nil { condition.wait() }
        let result = result
        condition.unlock()
        try result?.get()
    }
}

/// The short guard that pins shutdown against descriptor retirement. Blocking
/// writes and FileHandle.close both occur outside this guard.
final class DecodeServiceWriteFDLifetime: Sendable {
    private let state: Mutex<FileHandle?>

    init(_ handle: FileHandle) { state = Mutex(handle) }

    func borrow() throws -> FileHandle {
        try state.withLock { value in
            guard let value else { throw DecodeServiceCommandWriterError.closed }
            return value
        }
    }

    func shutdownWrite() {
        state.withLock { value in
            if let value { _ = Darwin.shutdown(value.fileDescriptor, SHUT_WR) }
        }
    }

    func interrupt() {
        state.withLock { value in
            if let value { _ = Darwin.shutdown(value.fileDescriptor, SHUT_RDWR) }
        }
    }

    func retire() {
        let retired = state.withLock { value -> FileHandle? in
            defer { value = nil }
            return value
        }
        try? retired?.close()
    }
}

/// Per-connection serial command owner. Queue admission never waits; both the
/// frame and byte limits include the item currently blocked in the kernel.
final class DecodeServiceCommandWriter: @unchecked Sendable {
    typealias SendFrame = @Sendable (Data, FileHandle) throws -> Void

    private struct Item: Sendable {
        let frame: Data
        let receipt: DecodeServiceWriteReceipt
    }

    private struct State: Sendable {
        var queue: [Item] = []
        var queuedBytes = 0
        var inFlightBytes = 0
        var accepting = true
        var closedError: Error?
        var finished = false
    }

    private let condition = NSCondition()
    private var state = State()
    private let lifetime: DecodeServiceWriteFDLifetime
    private let capacity: Int
    private let maximumQueuedBytes: Int
    private let sendFrame: SendFrame
    private let beforeReceiptPublication: @Sendable () -> Void

    init(
        output: FileHandle,
        capacity: Int = 8,
        maximumQueuedBytes: Int = 8 * (DecodeFrameCodec.maximumPayloadBytes + 4),
        sendFrame: @escaping SendFrame = { frame, output in
            try output.write(contentsOf: frame)
        },
        beforeReceiptPublication: @escaping @Sendable () -> Void = {}
    ) {
        lifetime = DecodeServiceWriteFDLifetime(output)
        self.capacity = capacity
        self.maximumQueuedBytes = maximumQueuedBytes
        self.sendFrame = sendFrame
        self.beforeReceiptPublication = beforeReceiptPublication
        let thread = Thread { [self] in run() }
        thread.name = "TurboFieldfare.DecodeService.CommandWriter"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func enqueue(_ command: DecodeServiceCommand) throws -> DecodeServiceWriteReceipt {
        try enqueue(encodedFrame: DecodeFrameCodec.encode(command))
    }

    /// Admission callers encode before their session critical section; only
    /// this bounded append is allowed while that section is held.
    func enqueue(encodedFrame frame: Data) throws -> DecodeServiceWriteReceipt {
        let receipt = DecodeServiceWriteReceipt()
        condition.lock()
        if let error = state.closedError {
            condition.unlock()
            throw error
        }
        guard state.accepting else {
            condition.unlock()
            throw DecodeServiceCommandWriterError.closed
        }
        let usedFrames = state.queue.count + (state.inFlightBytes == 0 ? 0 : 1)
        let usedBytes = state.queuedBytes + state.inFlightBytes
        guard usedFrames < capacity,
              usedBytes <= maximumQueuedBytes - frame.count else {
            condition.unlock()
            throw DecodeServiceCommandWriterError.queueFull
        }
        state.queue.append(Item(frame: frame, receipt: receipt))
        state.queuedBytes += frame.count
        condition.signal()
        condition.unlock()
        return receipt
    }

    func shutdownWrite() { lifetime.shutdownWrite() }
    func requestTransportShutdown() { lifetime.interrupt() }

    /// Refuses new admission while allowing already queued terminal work to drain.
    func seal() {
        condition.lock()
        state.accepting = false
        condition.unlock()
    }

    func close(error: Error = DecodeServiceCommandWriterError.closed) {
        let receipts = finish(error: error)
        receipts.forEach { $0.settle(.failure(error)) }
    }

    func waitUntilFinished() {
        condition.lock()
        while !state.finished { condition.wait() }
        condition.unlock()
    }

    private func run() {
        defer {
            lifetime.retire()
            condition.lock()
            state.finished = true
            condition.broadcast()
            condition.unlock()
        }
        while true {
            condition.lock()
            while state.queue.isEmpty, state.closedError == nil { condition.wait() }
            if state.queue.isEmpty, state.closedError != nil {
                condition.unlock()
                return
            }
            let item = state.queue.removeFirst()
            state.queuedBytes -= item.frame.count
            state.inFlightBytes = item.frame.count
            condition.unlock()

            do {
                let output = try lifetime.borrow()
                try sendFrame(item.frame, output)
                condition.lock()
                state.inFlightBytes = 0
                condition.broadcast()
                condition.unlock()
                beforeReceiptPublication()
                item.receipt.settle(.success(()))
            } catch {
                let receipts = finish(error: error, clearInFlight: true)
                beforeReceiptPublication()
                item.receipt.settle(.failure(error))
                receipts.forEach { $0.settle(.failure(error)) }
                return
            }
        }
    }

    private func finish(
        error: Error, clearInFlight: Bool = false
    ) -> [DecodeServiceWriteReceipt] {
        condition.lock()
        if clearInFlight { state.inFlightBytes = 0 }
        if state.closedError == nil { state.closedError = error }
        state.accepting = false
        let receipts = state.queue.map(\.receipt)
        state.queue.removeAll()
        state.queuedBytes = 0
        condition.broadcast()
        condition.unlock()
        return receipts
    }
}

private final class SessionEffectSerial: @unchecked Sendable {
    private let queue = DispatchQueue(label: "TurboFieldfare.DecodeService.SessionEffects")
    private let key = DispatchSpecificKey<UInt8>()

    init() { queue.setSpecific(key: key, value: 1) }

    func sync<T>(_ effect: () throws -> T) rethrows -> T {
        if DispatchQueue.getSpecific(key: key) == 1 { return try effect() }
        return try queue.sync(execute: effect)
    }
}

private final class DecodeServiceTransport: Sendable {
    let writer: DecodeServiceCommandWriter
    let router: DecodeServiceResponseRouter

    init(
        input: FileHandle,
        output: FileHandle,
        sendFrame: @escaping DecodeServiceCommandWriter.SendFrame = { frame, output in
            try output.write(contentsOf: frame)
        },
        onTerminate: @escaping @Sendable (DecodeServiceResponseRouter, Error) -> Void
    ) {
        writer = DecodeServiceCommandWriter(output: input, sendFrame: sendFrame)
        router = DecodeServiceResponseRouter(output: output, onTerminate: onTerminate)
    }
}

struct DecodeServicePeerIdentity: Sendable, Equatable {
    let pid: pid_t
    let pidVersion: Int32

    static func capture(from descriptor: Int32) throws -> Self {
        var token = audit_token_t()
        var length = socklen_t(MemoryLayout<audit_token_t>.size)
        let result = withUnsafeMutablePointer(to: &token) { pointer in
            getsockopt(descriptor, SOL_LOCAL, LOCAL_PEERTOKEN, pointer, &length)
        }
        guard result == 0, length == MemoryLayout<audit_token_t>.size else {
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }
        return Self(
            pid: audit_token_to_pid(token),
            pidVersion: Int32(audit_token_to_pidversion(token)))
    }
}

struct ServiceProcessObserverCheckpoints: Sendable {
    var beforeCreate: @Sendable () -> Void = {}
    var beforeUserRegistration: @Sendable () -> Void = {}
    var beforeProcessRegistration: @Sendable () -> Void = {}
    var beforeWaitPublication: @Sendable () -> Void = {}
    var beforeClose: @Sendable () -> Void = {}
}

/// Public-kqueue process observation. Its dedicated OS thread is the sole FD
/// owner from creation through both registration receipts, wait, last use, and
/// close. External callers can only request the serialized USER wake.
final class ServiceProcessObserver: @unchecked Sendable {
    enum RegistrationResult: Sendable, Equatable {
        case success
        case cancelled
        case failure(Int32)
    }

    enum Event: Sendable, Equatable {
        case executionChanged
        case exited
        case observationFailed(Int32)
    }
    private enum Phase: Equatable { case notStarted, registeringUser, registeringProcess, waiting, closing, closed }

    private let condition = NSCondition()
    private var phase: Phase = .notStarted
    private var cancelRequested = false
    private var registrationResult: RegistrationResult?
    private var descriptor: Int32 = -1
    private let pid: pid_t
    private let userIdentifier: UInt
    private let checkpoints: ServiceProcessObserverCheckpoints
    private let onEvent: @Sendable (Event) -> Void

    init(
        pid: pid_t,
        checkpoints: ServiceProcessObserverCheckpoints = .init(),
        onEvent: @escaping @Sendable (Event) -> Void
    ) {
        self.pid = pid
        self.userIdentifier = UInt.random(in: 1...UInt.max)
        self.checkpoints = checkpoints
        self.onEvent = onEvent
    }

    func start() {
        condition.lock()
        guard phase == .notStarted else { condition.unlock(); return }
        phase = .registeringUser
        condition.unlock()
        let thread = Thread { [self] in run() }
        thread.name = "TurboFieldfare.DecodeService.ProcessObserver"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func waitForRegistration() -> RegistrationResult {
        condition.lock()
        while registrationResult == nil { condition.wait() }
        let result = registrationResult!
        condition.unlock()
        return result
    }

    func waitUntilClosed() {
        condition.lock()
        while phase != .closed { condition.wait() }
        condition.unlock()
    }

    func requestCancellation() {
        condition.lock()
        guard phase != .closing, phase != .closed else {
            condition.unlock()
            return
        }
        guard !cancelRequested else { condition.unlock(); return }
        cancelRequested = true
        guard phase == .waiting, descriptor >= 0 else {
            condition.broadcast()
            condition.unlock()
            return
        }
        var trigger = kevent(
            ident: userIdentifier, filter: Int16(EVFILT_USER),
            flags: 0, fflags: UInt32(NOTE_TRIGGER), data: 0, udata: nil)
        _ = kevent(descriptor, &trigger, 1, nil, 0, nil)
        condition.unlock()
    }

    private func publishRegistration(_ result: RegistrationResult) {
        condition.lock()
        if registrationResult == nil { registrationResult = result }
        condition.broadcast()
        condition.unlock()
    }

    private func cancelled() -> Bool {
        condition.lock()
        defer { condition.unlock() }
        return cancelRequested
    }

    private func run() {
        checkpoints.beforeCreate()
        let fd = kqueue()
        guard fd >= 0 else {
            publishRegistration(.failure(errno))
            markClosed()
            return
        }
        condition.lock()
        descriptor = fd
        let cancelledBeforeRegistration = cancelRequested
        condition.unlock()
        guard !cancelledBeforeRegistration else {
            publishRegistration(.cancelled)
            closeOwned(fd)
            return
        }
        do {
            checkpoints.beforeUserRegistration()
            try register(
                fd: fd, ident: userIdentifier, filter: Int16(EVFILT_USER),
                fflags: 0)
            condition.lock()
            phase = .registeringProcess
            let cancelledAfterUser = cancelRequested
            condition.unlock()
            guard !cancelledAfterUser else {
                publishRegistration(.cancelled)
                closeOwned(fd)
                return
            }
            checkpoints.beforeProcessRegistration()
            try register(
                fd: fd, ident: UInt(pid), filter: Int16(EVFILT_PROC),
                fflags: UInt32(NOTE_EXIT) | UInt32(NOTE_EXEC))
            checkpoints.beforeWaitPublication()
            condition.lock()
            if cancelRequested {
                condition.unlock()
                publishRegistration(.cancelled)
                closeOwned(fd)
                return
            }
            phase = .waiting
            registrationResult = .success
            condition.broadcast()
            condition.unlock()

            while true {
                var event = kevent()
                let count = kevent(fd, nil, 0, &event, 1, nil)
                if count < 0, errno == EINTR { continue }
                if count <= 0 {
                    onEvent(.observationFailed(count < 0 ? errno : EIO))
                    break
                }
                if event.filter == Int16(EVFILT_USER), event.ident == userIdentifier {
                    break
                }
                guard event.filter == Int16(EVFILT_PROC),
                      event.ident == UInt(pid) else { continue }
                if event.fflags & UInt32(NOTE_EXIT) != 0 {
                    onEvent(.exited)
                    break
                }
                if event.fflags & UInt32(NOTE_EXEC) != 0 {
                    onEvent(.executionChanged)
                }
            }
        } catch let error as POSIXError {
            publishRegistration(.failure(error.code.rawValue))
        } catch {
            publishRegistration(.failure(EIO))
        }
        closeOwned(fd)
    }

    private func register(
        fd: Int32, ident: UInt, filter: Int16, fflags: UInt32
    ) throws {
        var change = kevent(
            ident: ident, filter: filter,
            flags: UInt16(EV_ADD | EV_CLEAR | EV_RECEIPT),
            fflags: fflags, data: 0, udata: nil)
        var receipt = kevent()
        let count = kevent(fd, &change, 1, &receipt, 1, nil)
        guard count == 1 else {
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }
        guard receipt.ident == ident, receipt.filter == filter,
              receipt.flags & UInt16(EV_ERROR) != 0, receipt.data == 0 else {
            let code = receipt.data == 0 ? EINVAL : Int32(receipt.data)
            throw POSIXError(.init(rawValue: code) ?? .EIO)
        }
    }

    private func closeOwned(_ fd: Int32) {
        checkpoints.beforeClose()
        condition.lock()
        phase = .closing
        descriptor = -1
        condition.unlock()
        Darwin.close(fd)
        markClosed()
    }

    private func markClosed() {
        condition.lock()
        phase = .closed
        if registrationResult == nil {
            registrationResult = cancelRequested ? .cancelled : .failure(EIO)
        }
        condition.broadcast()
        condition.unlock()
    }
}

struct ServiceSetupWorkerCheckpoints: Sendable {
    var beforeProcessRun: @Sendable () -> Void = {}
    var afterProcessRun: @Sendable () -> Void = {}
    var beforeConnectAttempt: @Sendable () -> Void = {}
    var beforeProbeHandoff: @Sendable () -> Void = {}
}

struct DecodeServiceRetirementCheckpoints: Sendable {
    /// Runs after removal has an exact joinable reservation and before any
    /// transport retirement I/O. Production uses the no-op default.
    var afterReservationBeforeRetirement: @Sendable () -> Void = {}
}

final class ServiceSetupWorker: @unchecked Sendable {
    private enum ProcessPhase: Equatable { case none, starting, running }
    private let condition = NSCondition()
    private let checkpoints: ServiceSetupWorkerCheckpoints
    private var cancelled = false
    private var cancellationActionsInFlight = false
    private var process: Process?
    private var phase: ProcessPhase = .none
    private var probeHandles: (FileHandle, FileHandle)?

    init(checkpoints: ServiceSetupWorkerCheckpoints = .init()) {
        self.checkpoints = checkpoints
    }

    func requestCancellation() {
        condition.lock()
        cancelled = true
        guard !cancellationActionsInFlight else {
            condition.broadcast()
            condition.unlock()
            return
        }
        let runningProcess = phase == .running ? process : nil
        let probeDescriptors = probeHandles.map {
            ($0.0.fileDescriptor, $0.1.fileDescriptor)
        }
        cancellationActionsInFlight = runningProcess != nil || probeDescriptors != nil
        condition.broadcast()
        condition.unlock()

        // Process and socket operations may block or call into Foundation.
        // The condition protects ownership only; no external I/O crosses it.
        runningProcess?.terminate()
        if let probeDescriptors {
            _ = Darwin.shutdown(probeDescriptors.0, SHUT_RDWR)
            _ = Darwin.shutdown(probeDescriptors.1, SHUT_RDWR)
        }

        if runningProcess != nil || probeDescriptors != nil {
            condition.lock()
            cancellationActionsInFlight = false
            condition.broadcast()
            condition.unlock()
        }
    }

    func checkCancellation() throws {
        condition.lock()
        let cancelled = cancelled
        condition.unlock()
        if cancelled { throw CancellationError() }
    }

    func runAndWait(_ process: Process) throws {
        condition.lock()
        guard !cancelled else { condition.unlock(); throw CancellationError() }
        self.process = process
        phase = .starting
        condition.unlock()
        checkpoints.beforeProcessRun()
        condition.lock()
        if cancelled {
            self.process = nil
            phase = .none
            condition.unlock()
            throw CancellationError()
        }
        condition.unlock()
        do { try process.run() }
        catch {
            condition.lock()
            self.process = nil
            phase = .none
            condition.unlock()
            throw error
        }
        condition.lock()
        phase = .running
        condition.unlock()
        checkpoints.afterProcessRun()
        condition.lock()
        let terminate = cancelled
        condition.unlock()
        if terminate { process.terminate() }
        process.waitUntilExit()
        condition.lock()
        self.process = nil
        phase = .none
        let wasCancelled = cancelled
        condition.unlock()
        if wasCancelled { throw CancellationError() }
    }

    func beforeConnectAttempt() throws {
        checkpoints.beforeConnectAttempt()
        try checkCancellation()
    }

    func beginProbe(input: FileHandle, output: FileHandle) throws {
        condition.lock()
        guard !cancelled else { condition.unlock(); throw CancellationError() }
        probeHandles = (input, output)
        condition.unlock()
    }

    func endProbe() {
        condition.lock()
        while cancellationActionsInFlight { condition.wait() }
        probeHandles = nil
        condition.unlock()
    }

    func completeProbeHandoff() throws {
        checkpoints.beforeProbeHandoff()
        condition.lock()
        while cancellationActionsInFlight { condition.wait() }
        probeHandles = nil
        let wasCancelled = cancelled
        condition.unlock()
        if wasCancelled { throw CancellationError() }
    }

    func waitBeforeRetry(milliseconds: Int) throws {
        condition.lock()
        if !cancelled {
            _ = condition.wait(until: Date().addingTimeInterval(
                Double(milliseconds) / 1_000))
        }
        let wasCancelled = cancelled
        condition.unlock()
        if wasCancelled { throw CancellationError() }
    }
}

private final class DecodeServiceLifecycleCompletion: Sendable {
    private struct State: Sendable {
        var finished = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state = Mutex(State())

    func wait() async {
        await withCheckedContinuation { continuation in
            let immediate = state.withLock { value in
                if value.finished { return true }
                value.waiters.append(continuation)
                return false
            }
            if immediate { continuation.resume() }
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

public final class DecodeServiceInferenceClient: AppModelLifecycleClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, AppContextCheckpointClient, @unchecked Sendable {
    typealias TerminationFallbackScheduler = @Sendable (
        @escaping @Sendable () -> Void
    ) -> Void
    typealias RuntimeMeasurementEvent = @Sendable (
        DecodeServiceEvent, DecodeRuntimeMeasurementRequest, UUID?, Int?
    ) async -> Void
    typealias RuntimeMeasurementReceptionEnded = @Sendable (
        DecodeRuntimeMeasurementRequest, UUID?, Int?
    ) async -> Void
    typealias StreamCancellationScheduler = @Sendable (
        @escaping @Sendable () -> Void
    ) -> Void
    typealias ServiceSetupWorkerFactory = @Sendable () -> ServiceSetupWorker
    private struct LoadedBinding: Sendable, Equatable {
        let loadID: UUID
        let family: DecodeModelFamily
        let modelIdentity: DecodeModelIdentity?
        let sourceIdentity: DecodeSourceIdentity?
        let toolThinkingEnabled: Bool?

        var readiness: AppLoadedModelReadiness? {
            switch family {
            case .gemma4:
                guard let toolThinkingEnabled, modelIdentity == nil,
                      sourceIdentity == nil else { return nil }
                return .gemma(toolThinkingEnabled: toolThinkingEnabled)
            case .qwen3_6:
                guard toolThinkingEnabled == nil else { return nil }
                if let modelIdentity, sourceIdentity == nil,
                   modelIdentity.family == .qwen3_6 {
                    return .qwen(identity: modelIdentity)
                }
                if let sourceIdentity, modelIdentity == nil {
                    return .qwenSource(identity: sourceIdentity)
                }
                return nil
            }
        }
    }

    private final class LoadAttemptToken: Sendable {
        private struct State: Sendable {
            var connectionID: UUID?
            var revision: UInt64?
            var lifetimeID: UUID?
            var teardownRequested = false
            var commandAdmitted = false
            var cancellationSent = false
        }

        let requestID: UUID
        let attemptID: UUID
        let priorBinding: LoadedBinding?
        let completion = DecodeServiceLifecycleCompletion()
        private let state = Mutex(State())

        init(
            requestID: UUID, attemptID: UUID, priorBinding: LoadedBinding?,
            priorLifetimeID: UUID?
        ) {
            self.requestID = requestID
            self.attemptID = attemptID
            self.priorBinding = priorBinding
            state.withLock { $0.lifetimeID = priorLifetimeID }
        }

        func bind(connectionID: UUID, revision: UInt64, lifetimeID: UUID?) -> Bool {
            state.withLock { value in
                guard value.connectionID == nil else { return false }
                value.connectionID = connectionID
                value.revision = revision
                value.lifetimeID = lifetimeID
                return true
            }
        }

        func bindPendingLifetime(_ lifetimeID: UUID) {
            state.withLock { value in
                if value.connectionID == nil { value.lifetimeID = lifetimeID }
            }
        }

        func belongs(to lifetimeID: UUID) -> Bool {
            state.withLock { $0.lifetimeID == lifetimeID }
        }

        var connectionIdentity: (UUID, UInt64)? {
            state.withLock { value in
                guard let connectionID = value.connectionID,
                      let revision = value.revision else { return nil }
                return (connectionID, revision)
            }
        }

        func requestTeardown() { state.withLock { $0.teardownRequested = true } }
        var teardownRequested: Bool { state.withLock { $0.teardownRequested } }
        func markCommandAdmitted() { state.withLock { $0.commandAdmitted = true } }
        var commandAdmitted: Bool { state.withLock { $0.commandAdmitted } }
        func reserveCancellation() -> Bool {
            state.withLock { value in
                guard !value.cancellationSent else { return false }
                value.cancellationSent = true
                return true
            }
        }
    }

    private final class BoundTeardownOwner: Sendable {
        let id = UUID()
        let binding: LoadedBinding
        let lifetimeID: UUID?
        let completion = DecodeServiceLifecycleCompletion()

        init(binding: LoadedBinding, lifetimeID: UUID?) {
            self.binding = binding
            self.lifetimeID = lifetimeID
        }
    }

    private enum ClientLifecycle: Sendable {
        case loading(LoadAttemptToken)
        case unloading(BoundTeardownOwner)
    }

    private final class TransportRetirementOwner: Sendable {
        let id = UUID()
        let completion = DecodeServiceLifecycleCompletion()
    }

    final class ServiceLifetime: Sendable {
        enum State: Sendable {
            case authoritativeIdle, active, executionChangedIdle
            case executionChangedAfterAdmission, exited, cleanupUnproven
        }
        let id: UUID
        let peer: DecodeServicePeerIdentity
        private let observerStorage = Mutex<ServiceProcessObserver?>(nil)
        private let state = Mutex<State>(.authoritativeIdle)

        init(id: UUID, peer: DecodeServicePeerIdentity) {
            self.id = id
            self.peer = peer
        }

        func attach(observer: ServiceProcessObserver) {
            observerStorage.withLock { $0 = observer }
        }

        func cancelObservation() {
            observerStorage.withLock { $0 }?.requestCancellation()
        }

        func cancelObservationAndWait() {
            guard let observer = observerStorage.withLock({ $0 }) else { return }
            observer.requestCancellation()
            observer.waitUntilClosed()
        }

        func admitFirstModelFrame<T>(_ append: () throws -> T) throws -> T {
            try state.withLock { value in
                switch value {
                case .authoritativeIdle, .active:
                    break
                default:
                    throw CancellationError()
                }
                let result = try append()
                value = .active
                return result
            }
        }

        func noteExecutionChanged() -> Bool {
            state.withLock { value in
                let hadPossibleAdmission: Bool
                switch value {
                case .active, .executionChangedAfterAdmission, .cleanupUnproven:
                    hadPossibleAdmission = true
                default:
                    hadPossibleAdmission = false
                }
                value = hadPossibleAdmission
                    ? .executionChangedAfterAdmission : .executionChangedIdle
                return hadPossibleAdmission
            }
        }
        func noteExit() { state.withLock { $0 = .exited } }
        func noteObservationFailure() { state.withLock { $0 = .cleanupUnproven } }
        func noteProtocolCleanup() {
            state.withLock { value in
                guard case .active = value else { return }
                value = .authoritativeIdle
            }
        }
        var isExited: Bool {
            state.withLock { value in
                if case .exited = value { return true }
                return false
            }
        }
        var cleanupProofRequired: Bool {
            state.withLock { value in
                switch value {
                case .active, .executionChangedAfterAdmission, .cleanupUnproven:
                    return true
                default:
                    return false
                }
            }
        }
        var isAuthoritativeIdle: Bool {
            state.withLock { value in
                if case .authoritativeIdle = value { return true }
                return false
            }
        }
    }

    private struct SessionToken: Sendable, Equatable {
        let connectionID: UUID
        let revision: UInt64
        let binding: LoadedBinding
        let openEpoch: UUID?
    }

    private final class ResponseEffectLease: Sendable {
        private struct State: Sendable { var active = true }

        let session: SessionToken
        let conversationEpoch: UUID?
        let operationID: UUID
        private let state = Mutex(State())

        init(session: SessionToken, conversationEpoch: UUID?, operationID: UUID) {
            self.session = session
            self.conversationEpoch = conversationEpoch
            self.operationID = operationID
        }

        /// Executes a synchronous observable effect in the same critical section
        /// used by revocation. Callers must not await, block on I/O, or invoke a
        /// cancellation callback from `body`.
        func admit<T>(_ body: (inout Bool) throws -> T) rethrows -> T? {
            try state.withLock { state in
                guard state.active else { return nil }
                var reserveTerminal = false
                let result = try body(&reserveTerminal)
                if reserveTerminal { state.active = false }
                return result
            }
        }

        func revoke() { state.withLock { $0.active = false } }
        var isActive: Bool { state.withLock { $0.active } }
    }

    private struct Connection: Sendable {
        var connectionID: UUID?
        var revision: UInt64 = 0
        var transport: DecodeServiceTransport?
        var binding: LoadedBinding?
        var openEpoch: UUID?
        var currentLoadAttempt: LoadAttemptToken?
        var currentOperation: ResponseEffectLease?
        var loadedDirectory: URL?
        var launchLabel: String?
        var socketPath: String?
        var lifetime: ServiceLifetime?
    }

    private struct ReservedRetirement: Sendable {
        let dead: Connection
        let owner: TransportRetirementOwner
        let observationsToCancel: [ServiceLifetime]
    }

    private struct Handles: Sendable {
        let writer: DecodeServiceCommandWriter
        let responses: DecodeServiceResponseRouter
        let connectionID: UUID
        let revision: UInt64
        let binding: LoadedBinding?
        let lifetime: ServiceLifetime?
    }

    /// Only opted-in requests need to outlive their visible stream. Protect the
    /// shared display state from an old receiver while its numeric footer drains.
    private final class MeasurementConsumer: Sendable {
        private struct State: Sendable {
            var active = true
            var stopRequested = false
            var finished = false
        }
        let generationID = UUID()
        private let state = Mutex(State())
        private let commands = DispatchQueue(label: "TurboFieldfare.MeasurementCommands")
        // Accessed only on commands. There is at most one queued Stop.
        private let dispatch = Mutex<(@Sendable () -> Void)?>(nil)

        func cancel() {
            state.withLock { $0.active = false }
            requestStop()
        }

        var isCancelled: Bool { state.withLock { !$0.active } }

        func requestStop() {
            let first = state.withLock { value in
                guard !value.stopRequested, !value.finished else { return false }
                value.stopRequested = true
                return true
            }
            guard first else { return }
            commands.async { [self] in
                let stop = dispatch.withLock { $0 }
                stop?()
            }
        }

        func enqueueGeneration(
            _ send: () throws -> DecodeServiceWriteReceipt,
            stop: @escaping @Sendable () -> Void
        ) throws -> DecodeServiceWriteReceipt {
            try commands.sync {
                guard !state.withLock({ $0.stopRequested }) else { throw CancellationError() }
                let receipt = try send()
                dispatch.withLock { $0 = stop }
                return receipt
            }
        }

        func finish() {
            state.withLock { $0.finished = true }
            commands.sync { dispatch.withLock { $0 = nil } }
        }

        func performIfActive(_ update: () -> Void) -> Bool {
            state.withLock { value in
                guard value.active else { return false }
                update()
                return true
            }
        }
    }

    private let connection = Mutex(Connection())
    private let pendingLifetime = Mutex<ServiceLifetime?>(nil)
    private let retiredLifetime = Mutex<ServiceLifetime?>(nil)
    private let quarantinedLifetime = Mutex<ServiceLifetime?>(nil)
    private let lifecycle = Mutex<ClientLifecycle?>(nil)
    private let retirementOwners = Mutex<[UUID: TransportRetirementOwner]>([:])
    private let activeSetupWorker = Mutex<ServiceSetupWorker?>(nil)
    private let sessionEffects = SessionEffectSerial()
    private let currentMeasurementConsumer = Mutex<MeasurementConsumer?>(nil)
    private let serviceURL: URL
    private let scheduleTerminationFallback: TerminationFallbackScheduler
    private let runtimeMeasurementEvent: RuntimeMeasurementEvent
    private let runtimeMeasurementReceptionEnded: RuntimeMeasurementReceptionEnded
    private let scheduleStreamCancellation: StreamCancellationScheduler
    private let makeSetupWorker: ServiceSetupWorkerFactory
    private let retirementCheckpoints: DecodeServiceRetirementCheckpoints
    private let inferenceMemory = Mutex<UInt64?>(nil)
    private let inferenceTowerMemory = Mutex<UInt64?>(nil)
    private let conversationLogicalStateMemory = Mutex<UInt64?>(nil)
    private let expertCacheMemory = Mutex<UInt64?>(nil)
    public let generationTranscriptMailbox = GenerationTranscriptMailbox()

    public var currentInferenceMemoryBytes: UInt64? {
        inferenceMemory.withLock { $0 }
    }

    public var currentInferenceTowerBytes: UInt64? {
        inferenceTowerMemory.withLock { $0 }
    }

    public var currentConversationLogicalStateBytes: UInt64? {
        conversationLogicalStateMemory.withLock { $0 }
    }

    public var currentExpertCacheBytes: UInt64? {
        expertCacheMemory.withLock { $0 }
    }

    private func publishRuntimeBytes(_ event: DecodeServiceEvent) {
        if let bytes = event.conversationLogicalStateBytes {
            conversationLogicalStateMemory.withLock { $0 = bytes }
        }
        if let bytes = event.expertCacheBytes {
            expertCacheMemory.withLock { $0 = bytes }
        }
    }

    public init(serviceURL: URL? = nil) {
        self.serviceURL = serviceURL ?? Self.defaultServiceURL()
        self.scheduleTerminationFallback = { action in
            DispatchQueue.global(qos: .userInitiated).asyncAfter(
                deadline: .now() + 1, execute: action)
        }
        self.runtimeMeasurementEvent = { event, capture, conversation, turn in
            await AgentInferenceTrace.shared?.runtimeMeasurementEvent(
                event, capture: capture, conversation: conversation, turn: turn)
        }
        self.runtimeMeasurementReceptionEnded = { capture, conversation, turn in
            await AgentInferenceTrace.shared?.runtimeMeasurementReceptionEnded(
                capture: capture, conversation: conversation, turn: turn)
        }
        // Always leave the continuation callback before touching admission.
        self.scheduleStreamCancellation = { action in
            DispatchQueue.global(qos: .userInitiated).async(execute: action)
        }
        self.makeSetupWorker = { ServiceSetupWorker() }
        self.retirementCheckpoints = .init()
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
    }

    /// Internal deterministic seams only. Production composition always uses
    /// the real trace actor and asynchronous cancellation scheduling above.
    init(
        testInput: FileHandle,
        responseOutput: FileHandle,
        sendFrame: @escaping DecodeServiceCommandWriter.SendFrame = { frame, output in
            try output.write(contentsOf: frame)
        },
        scheduleTerminationFallback: @escaping TerminationFallbackScheduler = { action in
            DispatchQueue.global(qos: .userInitiated).asyncAfter(
                deadline: .now() + 1, execute: action)
        },
        runtimeMeasurementEvent: @escaping RuntimeMeasurementEvent = {
            event, capture, conversation, turn in
            AgentInferenceTrace.shared?.runtimeMeasurementEvent(
                event, capture: capture, conversation: conversation, turn: turn)
        },
        runtimeMeasurementReceptionEnded: @escaping RuntimeMeasurementReceptionEnded = {
            capture, conversation, turn in
            AgentInferenceTrace.shared?.runtimeMeasurementReceptionEnded(
                capture: capture, conversation: conversation, turn: turn)
        },
        scheduleStreamCancellation: @escaping StreamCancellationScheduler = { action in
            DispatchQueue.global(qos: .userInitiated).async(execute: action)
        },
        makeSetupWorker: @escaping ServiceSetupWorkerFactory = {
            ServiceSetupWorker()
        },
        retirementCheckpoints: DecodeServiceRetirementCheckpoints = .init()
    ) {
        self.serviceURL = Self.defaultServiceURL()
        self.scheduleTerminationFallback = scheduleTerminationFallback
        self.runtimeMeasurementEvent = runtimeMeasurementEvent
        self.runtimeMeasurementReceptionEnded = runtimeMeasurementReceptionEnded
        self.scheduleStreamCancellation = scheduleStreamCancellation
        self.makeSetupWorker = makeSetupWorker
        self.retirementCheckpoints = retirementCheckpoints
        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
        installTransport(input: testInput, output: responseOutput, sendFrame: sendFrame)
    }

    var connectionIsInstalled: Bool {
        connection.withLock { $0.transport != nil }
    }

    var lifetimeCleanupIsQuarantined: Bool {
        quarantinedLifetime.withLock { $0 != nil }
    }

    var retiredLifetimeID: UUID? {
        retiredLifetime.withLock { $0?.id }
    }

    var installedRouter: DecodeServiceResponseRouter? {
        connection.withLock { $0.transport?.router }
    }

    private static func revokeOperationLocked(_ state: inout Connection) {
        state.currentOperation?.revoke()
        state.currentOperation = nil
    }

    private static func revokeLoadAttemptLocked(_ state: inout Connection) {
        state.currentLoadAttempt = nil
    }

    /// Caller holds `sessionEffects` and still owns the installed connection.
    /// Publish every state that blocks unload/successor admission before making
    /// that connection absent; transport I/O starts later in `retire`.
    private func reserveRetirementLocked(_ dead: Connection) -> ReservedRetirement? {
        guard dead.transport != nil else { return nil }
        let owner = TransportRetirementOwner()
        retirementOwners.withLock { $0[owner.id] = owner }
        if let lifetime = dead.lifetime {
            if lifetime.cleanupProofRequired, let binding = dead.binding {
                var displacedAttempt: LoadAttemptToken?
                lifecycle.withLock { value in
                    switch value {
                    case nil:
                        value = .unloading(BoundTeardownOwner(
                            binding: binding, lifetimeID: lifetime.id))
                    case .loading(let attempt)
                        where attempt.commandAdmitted && attempt.belongs(to: lifetime.id):
                        // This exact load already owns its possible admission.
                        break
                    case .loading(let attempt):
                        // A successor that had not admitted a model frame cannot
                        // inherit the prior lifetime's unresolved cleanup.
                        displacedAttempt = attempt
                        value = .unloading(BoundTeardownOwner(
                            binding: binding, lifetimeID: lifetime.id))
                    case .unloading:
                        // Admission rules prevent two production teardowns.
                        break
                    }
                }
                displacedAttempt?.completion.finish()
            }
            if lifetime.cleanupProofRequired {
                let quarantined = quarantinedLifetime.withLock { quarantine -> Bool in
                    if quarantine?.id == lifetime.id { return true }
                    guard quarantine == nil else { return false }
                    quarantine = lifetime
                    return true
                }
                if !quarantined {
                    retiredLifetime.withLock { retired in
                        if retired == nil { retired = lifetime }
                    }
                }
            } else if !lifetime.isExited {
                let quarantined = quarantinedLifetime.withLock { $0?.id == lifetime.id }
                if !quarantined {
                    retiredLifetime.withLock { retired in
                        if retired == nil { retired = lifetime }
                    }
                }
            }
        }
        return ReservedRetirement(
            dead: dead, owner: owner, observationsToCancel: [])
    }

    private func finishRetirement(_ owner: TransportRetirementOwner) {
        retirementOwners.withLock { owners in
            if owners[owner.id] === owner { owners.removeValue(forKey: owner.id) }
        }
        owner.completion.finish()
    }

    public var loadedModelReadiness: AppLoadedModelReadiness? {
        get async {
            guard lifecycle.withLock({ $0 == nil }) else { return nil }
            return connection.withLock { $0.binding?.readiness }
        }
    }

    func installTransport(
        input: FileHandle,
        output: FileHandle,
        lifetime: ServiceLifetime? = nil,
        sendFrame: @escaping DecodeServiceCommandWriter.SendFrame = { frame, output in
            try output.write(contentsOf: frame)
        }
    ) {
        let transport = DecodeServiceTransport(
            input: input, output: output, sendFrame: sendFrame) {
            [weak self] router, error in
            self?.invalidateConnection(
                expecting: router, error: error, dominatesPending: false)
        }
        let replaced = sessionEffects.sync {
            connection.withLock { state -> ReservedRetirement? in
                Self.revokeLoadAttemptLocked(&state)
                Self.revokeOperationLocked(&state)
                let replaced = state.transport == nil
                    ? nil : reserveRetirementLocked(state)
                state = Connection()
                state.connectionID = UUID()
                state.transport = transport
                state.lifetime = lifetime
                return replaced
            }
        }
        if let replaced { retire(replaced) }
    }

    func handleLifetimeEvent(
        lifetimeID: UUID, event: ServiceProcessObserver.Event
    ) {
        var hadPossibleAdmission = true
        let retirement = sessionEffects.sync {
            connection.withLock { state -> ReservedRetirement? in
                guard let lifetime = state.lifetime,
                      lifetime.id == lifetimeID else { return nil }
                switch event {
                case .executionChanged:
                    hadPossibleAdmission = lifetime.noteExecutionChanged()
                    if hadPossibleAdmission {
                        quarantinedLifetime.withLock { quarantine in
                            if quarantine == nil { quarantine = lifetime }
                        }
                    } else {
                        let retained = retiredLifetime.withLock { retired -> Bool in
                            guard retired == nil else { return false }
                            retired = lifetime
                            return true
                        }
                        if !retained {
                            quarantinedLifetime.withLock { quarantine in
                                if quarantine == nil { quarantine = lifetime }
                            }
                            hadPossibleAdmission = true
                        }
                    }
                case .exited:
                    lifetime.noteExit()
                case .observationFailed:
                    lifetime.noteObservationFailure()
                    quarantinedLifetime.withLock { quarantine in
                        if quarantine == nil { quarantine = lifetime }
                    }
                    hadPossibleAdmission = true
                }
                Self.revokeOperationLocked(&state)
                let retirement = reserveRetirementLocked(state)
                state = Connection()
                inferenceMemory.withLock { $0 = nil }
                inferenceTowerMemory.withLock { $0 = nil }
                conversationLogicalStateMemory.withLock { $0 = nil }
                expertCacheMemory.withLock { $0 = nil }
                return retirement
            }
        }
        if retirement == nil {
            var settlePendingLifecycle = false
            let pendingMatched = pendingLifetime.withLock { pending -> Bool in
                guard let pending, pending.id == lifetimeID else { return false }
                switch event {
                case .executionChanged:
                    let hadPossibleAdmission = pending.noteExecutionChanged()
                    let retained = retiredLifetime.withLock { retired -> Bool in
                        guard retired == nil else { return false }
                        retired = pending
                        return true
                    }
                    if hadPossibleAdmission || !retained {
                        quarantinedLifetime.withLock { quarantine in
                            if quarantine == nil { quarantine = pending }
                        }
                    } else {
                        settlePendingLifecycle = true
                    }
                case .exited:
                    pending.noteExit()
                    settlePendingLifecycle = true
                case .observationFailed:
                    pending.noteObservationFailure()
                    quarantinedLifetime.withLock { quarantine in
                        if quarantine == nil { quarantine = pending }
                    }
                }
                return true
            }
            if pendingMatched {
                if settlePendingLifecycle {
                    let completed = takeLifecycle(belongingTo: lifetimeID)
                    switch completed {
                    case .loading(let attempt): attempt.completion.finish()
                    case .unloading(let owner): owner.completion.finish()
                    case nil: break
                    }
                }
                return
            }
        }
        if retirement == nil, case .observationFailed = event {
            let retiredMatched = retiredLifetime.withLock { value -> Bool in
                guard let retired = value, retired.id == lifetimeID else {
                    return false
                }
                retired.noteObservationFailure()
                let moved = quarantinedLifetime.withLock { quarantine -> Bool in
                    guard quarantine == nil else { return false }
                    quarantine = retired
                    return true
                }
                if moved { value = nil }
                return true
            }
            if retiredMatched { return }
            let matchedQuarantine = quarantinedLifetime.withLock { quarantine -> Bool in
                guard quarantine?.id == lifetimeID else { return false }
                quarantine?.noteObservationFailure()
                return true
            }
            if matchedQuarantine { return }
        }
        if retirement == nil, event == .exited {
            let retiredMatch = retiredLifetime.withLock { retired -> Bool in
                guard retired?.id == lifetimeID else { return false }
                retired?.noteExit()
                retired = nil
                return true
            }
            let quarantineMatch = quarantinedLifetime.withLock { quarantine -> Bool in
                guard quarantine?.id == lifetimeID else { return false }
                quarantine?.noteExit()
                quarantine = nil
                return true
            }
            guard retiredMatch || quarantineMatch else { return }
            if quarantineMatch {
                retiredLifetime.withLock { retired in
                    guard let candidate = retired,
                          candidate.cleanupProofRequired else { return }
                    let promoted = quarantinedLifetime.withLock { quarantine -> Bool in
                        guard quarantine == nil else { return false }
                        quarantine = candidate
                        return true
                    }
                    if promoted { retired = nil }
                }
            }
            let completed = takeLifecycle(belongingTo: lifetimeID)
            switch completed {
            case .loading(let attempt): attempt.completion.finish()
            case .unloading(let owner): owner.completion.finish()
            case nil: break
            }
            return
        }
        guard let retirement else { return }
        retire(retirement)
        switch event {
        case .executionChanged where !hadPossibleAdmission:
            let loading = lifecycle.withLock { value -> LoadAttemptToken? in
                guard case .loading(let attempt) = value else { return nil }
                value = nil
                return attempt
            }
            loading?.completion.finish()
        case .exited:
            retiredLifetime.withLock { retired in
                if retired?.id == lifetimeID { retired = nil }
            }
            quarantinedLifetime.withLock { quarantine in
                if quarantine?.id == lifetimeID { quarantine = nil }
            }
            let completed = takeLifecycle(belongingTo: lifetimeID)
            switch completed {
            case .loading(let attempt): attempt.completion.finish()
            case .unloading(let owner): owner.completion.finish()
            case nil: break
            }
        case .observationFailed:
            // The watcher is gone without a death receipt. Cleanup remains
            // unproven and lifecycle admission stays blocked.
            break
        default:
            // Possible model admission remains quarantined until NOTE_EXIT.
            break
        }
    }

    private func takeLifecycle(belongingTo lifetimeID: UUID) -> ClientLifecycle? {
        lifecycle.withLock { value in
            let matches: Bool
            switch value {
            case .loading(let attempt): matches = attempt.belongs(to: lifetimeID)
            case .unloading(let owner): matches = owner.lifetimeID == lifetimeID
            case nil: matches = false
            }
            guard matches else { return nil }
            defer { value = nil }
            return value
        }
    }

    private func invalidateConnection(
        expecting router: DecodeServiceResponseRouter? = nil,
        error: Error = DecodeFrameError.unexpectedEOF,
        dominatesPending: Bool = true
    ) {
        let retirement = sessionEffects.sync {
            connection.withLock { state -> ReservedRetirement? in
                if let router, state.transport?.router !== router { return nil }
                guard state.transport != nil else { return nil }
                Self.revokeLoadAttemptLocked(&state)
                Self.revokeOperationLocked(&state)
                inferenceMemory.withLock { $0 = nil }
                inferenceTowerMemory.withLock { $0 = nil }
                conversationLogicalStateMemory.withLock { $0 = nil }
                expertCacheMemory.withLock { $0 = nil }
                let retirement = reserveRetirementLocked(state)
                state = Connection()
                return retirement
            }
        }
        guard let retirement else { return }
        retire(
            retirement, terminateWaiters: dominatesPending,
            terminationError: error)
    }

    private func invalidateConnection(
        admittedBy token: LoadAttemptToken,
        responses: DecodeServiceResponseRouter,
        error: Error
    ) {
        guard let identity = token.connectionIdentity else { return }
        invalidateConnectionIfCurrent(responses: responses, error: error) { state in
            state.currentLoadAttempt === token
                && state.connectionID == identity.0
                && state.revision == identity.1
                && state.binding == nil
                && state.openEpoch == nil
        }
    }

    private func invalidateConnection(
        admittedBy token: SessionToken,
        responses: DecodeServiceResponseRouter,
        error: Error
    ) {
        invalidateConnectionIfCurrent(responses: responses, error: error) { state in
            state.connectionID == token.connectionID
                && state.revision == token.revision
                && state.binding == token.binding
                && state.openEpoch == token.openEpoch
        }
    }

    private func invalidateConnection(
        admittedBy lease: ResponseEffectLease,
        responses: DecodeServiceResponseRouter,
        error: Error
    ) {
        invalidateConnectionIfCurrent(responses: responses, error: error) { state in
            state.currentOperation === lease
                && state.connectionID == lease.session.connectionID
                && state.revision == lease.session.revision
                && state.binding == lease.session.binding
                && state.openEpoch == lease.session.openEpoch
                && lease.conversationEpoch == lease.session.openEpoch
                && lease.isActive
        }
    }

    private func invalidateConnectionIfCurrent(
        responses: DecodeServiceResponseRouter,
        error: Error,
        admission: (Connection) -> Bool
    ) {
        let retirement = sessionEffects.sync {
            connection.withLock { state -> ReservedRetirement? in
                guard state.transport?.router === responses,
                      admission(state) else { return nil }
                Self.revokeLoadAttemptLocked(&state)
                Self.revokeOperationLocked(&state)
                inferenceMemory.withLock { $0 = nil }
                inferenceTowerMemory.withLock { $0 = nil }
                conversationLogicalStateMemory.withLock { $0 = nil }
                expertCacheMemory.withLock { $0 = nil }
                let retirement = reserveRetirementLocked(state)
                state = Connection()
                return retirement
            }
        }
        guard let retirement else { return }
        retire(retirement, terminationError: error)
    }

    private func sessionToken(
        for handles: Handles, binding: LoadedBinding
    ) -> SessionToken? {
        connection.withLock { state in
            guard state.connectionID == handles.connectionID,
                  state.revision == handles.revision,
                  state.transport?.router === handles.responses,
                  state.binding == binding else { return nil }
            return SessionToken(
                connectionID: handles.connectionID,
                revision: handles.revision, binding: binding,
                openEpoch: state.openEpoch)
        }
    }

    private func beginSessionMutation(
        handles: Handles,
        binding: LoadedBinding,
        requiringOpenEpoch: UUID? = nil,
        permittingLifecycle: Bool = false
    ) -> SessionToken? {
        sessionEffects.sync {
            if !permittingLifecycle,
               lifecycle.withLock({ $0 != nil }) { return nil }
            return connection.withLock { state in
                guard state.connectionID == handles.connectionID,
                      state.revision == handles.revision,
                      state.transport?.writer === handles.writer,
                      state.transport?.router === handles.responses,
                      state.binding == binding,
                      requiringOpenEpoch == nil
                        || state.openEpoch == requiringOpenEpoch else { return nil }
                Self.revokeOperationLocked(&state)
                state.revision &+= 1
                return SessionToken(
                    connectionID: handles.connectionID,
                    revision: state.revision, binding: binding,
                    openEpoch: state.openEpoch)
            }
        }
    }

    private func isCurrent(
        _ token: LoadAttemptToken, responses: DecodeServiceResponseRouter
    ) -> Bool {
        guard let identity = token.connectionIdentity else { return false }
        return connection.withLock { state in
            state.currentLoadAttempt === token
                && state.connectionID == identity.0
                && state.revision == identity.1
                && state.transport?.router === responses
                && state.binding == nil
                && state.openEpoch == nil
        }
    }

    private func finishLoadAttempt(_ token: LoadAttemptToken) {
        sessionEffects.sync {
            connection.withLock { state in
                if state.currentLoadAttempt === token {
                    state.currentLoadAttempt = nil
                }
            }
        }
    }

    private func isCurrent(
        _ token: SessionToken, responses: DecodeServiceResponseRouter
    ) -> Bool {
        connection.withLock { state in
            state.connectionID == token.connectionID
                && state.revision == token.revision
                && state.transport?.router === responses
                && state.binding == token.binding
                && state.openEpoch == token.openEpoch
        }
    }

    private func enqueue(
        _ command: DecodeServiceCommand,
        admittedBy token: LoadAttemptToken,
        with writer: DecodeServiceCommandWriter,
        expecting responses: DecodeServiceResponseRouter
    ) throws -> DecodeServiceWriteReceipt {
        let frame = try DecodeFrameCodec.encode(command)
        let isFirstModelFrame: Bool
        if case .load = command { isFirstModelFrame = true }
        else { isFirstModelFrame = false }
        guard let identity = token.connectionIdentity else { throw CancellationError() }
        return try connection.withLock { state in
            guard state.currentLoadAttempt === token,
                  state.connectionID == identity.0,
                  state.revision == identity.1,
                  state.transport?.writer === writer,
                  state.transport?.router === responses else {
                throw CancellationError()
            }
            let receipt: DecodeServiceWriteReceipt
            if isFirstModelFrame {
                guard state.binding == token.priorBinding else {
                    throw CancellationError()
                }
                if let lifetime = state.lifetime {
                    receipt = try quarantinedLifetime.withLock { quarantine in
                        guard quarantine == nil else { throw CancellationError() }
                        return try lifetime.admitFirstModelFrame {
                            try writer.enqueue(encodedFrame: frame)
                        }
                    }
                } else {
                    receipt = try writer.enqueue(encodedFrame: frame)
                }
                state.binding = nil
                state.openEpoch = nil
                state.loadedDirectory = nil
                inferenceMemory.withLock { $0 = nil }
                inferenceTowerMemory.withLock { $0 = nil }
                conversationLogicalStateMemory.withLock { $0 = nil }
                expertCacheMemory.withLock { $0 = nil }
            } else {
                guard state.binding == nil, state.openEpoch == nil else {
                    throw CancellationError()
                }
                receipt = try writer.enqueue(encodedFrame: frame)
            }
            token.markCommandAdmitted()
            return receipt
        }
    }

    private func enqueue(
        _ command: DecodeServiceCommand,
        admittedBy token: SessionToken,
        with writer: DecodeServiceCommandWriter,
        expecting responses: DecodeServiceResponseRouter
    ) throws -> DecodeServiceWriteReceipt {
        let frame = try DecodeFrameCodec.encode(command)
        return try connection.withLock { state in
            guard state.connectionID == token.connectionID,
                  state.revision == token.revision,
                  state.transport?.writer === writer,
                  state.transport?.router === responses,
                  state.binding == token.binding,
                  state.openEpoch == token.openEpoch else { throw CancellationError() }
            return try writer.enqueue(encodedFrame: frame)
        }
    }

    private func enqueue(
        _ command: DecodeServiceCommand,
        admittedBy lease: ResponseEffectLease,
        with writer: DecodeServiceCommandWriter,
        expecting responses: DecodeServiceResponseRouter
    ) throws -> DecodeServiceWriteReceipt {
        let frame = try DecodeFrameCodec.encode(command)
        return try connection.withLock { state in
            guard state.currentOperation === lease,
                  state.connectionID == lease.session.connectionID,
                  state.revision == lease.session.revision,
                  state.transport?.writer === writer,
                  state.transport?.router === responses,
                  state.binding == lease.session.binding,
                  state.openEpoch == lease.session.openEpoch,
                  lease.conversationEpoch == lease.session.openEpoch,
                  lease.isActive else { throw CancellationError() }
            return try writer.enqueue(encodedFrame: frame)
        }
    }

    private func waitForSend(
        _ receipt: DecodeServiceWriteReceipt,
        expecting responses: DecodeServiceResponseRouter
    ) async throws {
        do { try await receipt.wait() }
        catch {
            invalidateConnection(expecting: responses, error: error)
            throw AppInferenceError.unknown(
                "could not reach the decode service: \(error)")
        }
    }

    private func admit<T>(
        _ lease: ResponseEffectLease,
        responses: DecodeServiceResponseRouter,
        _ effect: (inout Bool) throws -> T
    ) rethrows -> T? {
        let current = connection.withLock { state in
            state.currentOperation === lease
                && state.connectionID == lease.session.connectionID
                && state.revision == lease.session.revision
                && state.transport?.router === responses
                && state.binding == lease.session.binding
                && state.openEpoch == lease.session.openEpoch
                && lease.conversationEpoch == lease.session.openEpoch
        }
        guard current else { return nil }
        return try lease.admit(effect)
    }

    private func cancel(
        operationID: UUID,
        session: SessionToken,
        writer: DecodeServiceCommandWriter,
        responses: DecodeServiceResponseRouter
    ) {
        do {
            let receipt = try enqueue(
                .cancelGenerationBound(DecodeCancelGenerationRequest(
                    operationID: operationID,
                    loadID: session.binding.loadID)),
                admittedBy: session, with: writer, expecting: responses)
            Task { [weak self] in
                do { try await receipt.wait() }
                catch { self?.invalidateConnection(expecting: responses, error: error) }
            }
        } catch {
            // The exact control was already superseded; never target its successor.
        }
    }

    private func retireOperation(_ lease: ResponseEffectLease) {
        sessionEffects.sync {
            connection.withLock { state in
                guard state.currentOperation === lease else { return }
                lease.revoke()
                state.currentOperation = nil
            }
        }
    }

    private func retireOperation(
        storedIn leaseBox: borrowing Mutex<ResponseEffectLease?>
    ) {
        sessionEffects.sync {
            guard let lease = leaseBox.withLock({ $0 }) else { return }
            connection.withLock { state in
                guard state.currentOperation === lease else { return }
                lease.revoke()
                state.currentOperation = nil
            }
        }
    }

    private func cancel(
        operation lease: ResponseEffectLease,
        writer: DecodeServiceCommandWriter,
        responses: DecodeServiceResponseRouter
    ) {
        do {
            let receipt = try enqueue(
                .cancelGenerationBound(DecodeCancelGenerationRequest(
                    operationID: lease.operationID,
                    loadID: lease.session.binding.loadID)),
                admittedBy: lease.session, with: writer, expecting: responses)
            Task { [weak self] in
                do { try await receipt.wait() }
                catch { self?.invalidateConnection(expecting: responses, error: error) }
            }
        } catch {
            // A replaced connection or load must not be invalidated by an old callback.
        }
    }

    private func retire(
        _ retirement: ReservedRetirement,
        shutdownFrame: Data? = nil,
        terminateWaiters: Bool = true,
        terminationError: Error? = nil
    ) {
        let dead = retirement.dead
        let owner = retirement.owner
        retirementCheckpoints.afterReservationBeforeRetirement()
        retirement.observationsToCancel.forEach { $0.cancelObservation() }
        guard let transport = dead.transport else {
            finishRetirement(owner)
            return
        }
        if terminateWaiters {
            if let terminationError {
                transport.router.terminateWaiters(with: terminationError)
            } else {
                transport.router.terminateWaiters()
            }
        }
        let shutdownReceipt = shutdownFrame.flatMap { frame in
            try? transport.writer.enqueue(encodedFrame: frame)
        }
        if shutdownFrame != nil { transport.writer.seal() }
        if shutdownReceipt != nil {
            scheduleTerminationFallback {
                transport.writer.requestTransportShutdown()
                transport.router.requestTransportShutdown()
            }
        } else {
            transport.writer.close()
            transport.writer.requestTransportShutdown()
            transport.router.requestTransportShutdown()
        }
        let label = dead.launchLabel
        let socketPath = dead.socketPath
        Thread.detachNewThread { [weak self] in
            if let shutdownReceipt {
                do {
                    try shutdownReceipt.waitBlocking()
                    transport.writer.shutdownWrite()
                } catch {
                    transport.writer.requestTransportShutdown()
                    transport.router.requestTransportShutdown()
                }
                transport.writer.close()
            }
            transport.writer.waitUntilFinished()
            transport.router.waitUntilTransportClosed()
            if let label { Self.removeLaunchJob(label: label) }
            if let socketPath { unlink(socketPath) }
            if let self { self.finishRetirement(owner) }
            else { owner.completion.finish() }
        }
    }

    private func settleFailedLoading(_ attempt: LoadAttemptToken) async {
        let ownsUnresolvedAdmission = attempt.commandAdmitted
            && (quarantinedLifetime.withLock { quarantine in
                guard let quarantine else { return false }
                return attempt.belongs(to: quarantine.id)
            } || retiredLifetime.withLock { retired in
                guard let retired, retired.cleanupProofRequired else { return false }
                return attempt.belongs(to: retired.id)
            })
        if ownsUnresolvedAdmission {
            await attempt.completion.wait()
        } else {
            finishLoading(attempt)
        }
    }

    private func finishLoading(_ attempt: LoadAttemptToken) {
        sessionEffects.sync {
            connection.withLock { state in
                if state.currentLoadAttempt === attempt {
                    state.currentLoadAttempt = nil
                }
            }
            lifecycle.withLock { value in
                if case .loading(let current) = value, current === attempt {
                    value = nil
                }
            }
        }
        attempt.completion.finish()
    }

    private func sendLoadCancellation(
        _ attempt: LoadAttemptToken,
        handles: Handles
    ) async {
        guard attempt.commandAdmitted, attempt.reserveCancellation() else { return }
        do {
            let receipt = try enqueue(
                .cancelLoad(DecodeCancelLoadRequest(
                    requestID: attempt.requestID, attemptID: attempt.attemptID)),
                admittedBy: attempt, with: handles.writer,
                expecting: handles.responses)
            try await waitForSend(receipt, expecting: handles.responses)
        } catch {
            invalidateConnection(expecting: handles.responses, error: error)
        }
    }

    private func performBoundUnload(
        handles: Handles, binding: LoadedBinding
    ) async -> Bool {
        guard let token = beginSessionMutation(
            handles: handles, binding: binding, permittingLifecycle: true
        ) else { return false }
        let request = DecodeUnloadRequest(loadID: binding.loadID)
        do {
            let receipt = try enqueue(
                .unloadBound(request), admittedBy: token,
                with: handles.writer, expecting: handles.responses)
            try await waitForSend(receipt, expecting: handles.responses)
            guard isCurrent(token, responses: handles.responses) else { return false }
            let event = try await handles.responses.next(matching: request.requestID)
            guard isCurrent(token, responses: handles.responses),
                  Self.matches(event, binding: binding, expectedEpoch: nil),
                  event.kind == .unloaded else {
                invalidateConnection(
                    admittedBy: token, responses: handles.responses,
                    error: AppInferenceError.unknown(
                        "decode service unload acknowledgement changed identity or epoch"))
                return false
            }
            let completed = sessionEffects.sync {
                connection.withLock { state -> Bool in
                    guard state.connectionID == token.connectionID,
                          state.revision == token.revision,
                          state.transport?.router === handles.responses,
                          state.binding == token.binding,
                          state.openEpoch == token.openEpoch else { return false }
                    Self.revokeOperationLocked(&state)
                    state.revision &+= 1
                    state.loadedDirectory = nil
                    state.binding = nil
                    state.openEpoch = nil
                    inferenceMemory.withLock { $0 = nil }
                    inferenceTowerMemory.withLock { $0 = nil }
                    conversationLogicalStateMemory.withLock { $0 = nil }
                    expertCacheMemory.withLock { $0 = nil }
                    return true
                }
            }
            if completed { handles.lifetime?.noteProtocolCleanup() }
            return completed
        } catch {
            invalidateConnection(
                admittedBy: token, responses: handles.responses, error: error)
            return false
        }
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        enum Admission {
            case admitted(LoadAttemptToken)
            case cleanupUnproven
            case lifecycleInProgress
        }
        let requestID = UUID()
        let attemptID = UUID()
        let admission = sessionEffects.sync { () -> Admission in
            guard retirementOwners.withLock({ $0.isEmpty }),
                  quarantinedLifetime.withLock({ $0 == nil }) else {
                return .cleanupUnproven
            }
            return connection.withLock { state in
                guard state.currentOperation == nil else {
                    return .lifecycleInProgress
                }
                return lifecycle.withLock { value in
                    guard value == nil else { return .lifecycleInProgress }
                    let attempt = LoadAttemptToken(
                        requestID: requestID, attemptID: attemptID,
                        priorBinding: state.binding,
                        priorLifetimeID: state.lifetime?.id)
                    value = .loading(attempt)
                    return .admitted(attempt)
                }
            }
        }
        let attempt: LoadAttemptToken
        switch admission {
        case .admitted(let admitted):
            attempt = admitted
        case .cleanupUnproven:
            throw AppInferenceError.modelLoadFailed(
                "decode service cleanup could not be proven")
        case .lifecycleInProgress:
            throw RealInferenceLifecycleError.lifecycleInProgress
        }
        let priorBinding = attempt.priorBinding

        onState(.loading(.validatingDirectory))
        let transport: (writer: DecodeServiceCommandWriter,
                        responses: DecodeServiceResponseRouter)
        do {
            transport = try await ensureProcessAsync()
        } catch {
            await settleFailedLoading(attempt)
            throw error
        }
        if Task.isCancelled { attempt.requestTeardown() }
        guard let handles = currentHandles(),
              handles.writer === transport.writer,
              handles.responses === transport.responses else {
            await settleFailedLoading(attempt)
            throw AppInferenceError.modelLoadFailed(
                "the decode service connection changed before loading began")
        }
        if attempt.teardownRequested, !attempt.commandAdmitted {
            if let priorBinding {
                guard await performBoundUnload(handles: handles, binding: priorBinding) else {
                    await attempt.completion.wait()
                    throw CancellationError()
                }
            }
            finishLoading(attempt)
            throw CancellationError()
        }
        let request = DecodeLoadRequest(
            modelPath: modelDirectory.path, maxContextTokens: maxContextTokens,
            runtimeOptions: Self.decodeRuntimeOptions(options),
            forceLogitsHead: forceLogitsHead, requestID: requestID,
            attemptID: attemptID)
        let attached = sessionEffects.sync {
            connection.withLock { state -> Bool in
                guard state.connectionID == handles.connectionID,
                      state.revision == handles.revision,
                      state.transport?.writer === handles.writer,
                      state.transport?.router === handles.responses else { return false }
                Self.revokeOperationLocked(&state)
                state.revision &+= 1
                guard attempt.bind(
                    connectionID: handles.connectionID, revision: state.revision,
                    lifetimeID: handles.lifetime?.id) else {
                    return false
                }
                state.currentLoadAttempt = attempt
                return true
            }
        }
        guard attached else {
            await settleFailedLoading(attempt)
            throw CancellationError()
        }
        do {
            let receipt = try enqueue(
                .load(request), admittedBy: attempt,
                with: handles.writer, expecting: handles.responses)
            if Task.isCancelled { attempt.requestTeardown() }
            if attempt.teardownRequested {
                await sendLoadCancellation(attempt, handles: handles)
            }
            try await waitForSend(receipt, expecting: handles.responses)
            guard isCurrent(attempt, responses: handles.responses) else {
                throw CancellationError()
            }
            if Task.isCancelled { attempt.requestTeardown() }
            if attempt.teardownRequested {
                await sendLoadCancellation(attempt, handles: handles)
            }
            let event = try await handles.responses.next(matching: request.requestID)
            guard isCurrent(attempt, responses: handles.responses),
                  event.loadAttemptID == attemptID else {
                throw CancellationError()
            }
            switch event.kind {
            case .loadCancelled:
                guard event.loadedFamily == nil, event.loadID == nil,
                      event.modelIdentity == nil, event.sourceIdentity == nil,
                      event.conversationEpoch == nil else {
                    invalidateConnection(
                        admittedBy: attempt, responses: handles.responses,
                        error: AppInferenceError.modelLoadFailed(
                            "decode service cancellation carried session identity"))
                    throw AppInferenceError.modelLoadFailed(
                        "decode service cancellation carried session identity")
                }
                handles.lifetime?.noteProtocolCleanup()
                finishLoading(attempt)
                throw CancellationError()
            case .failed:
                guard event.loadedFamily == nil, event.loadID == nil,
                      event.modelIdentity == nil, event.sourceIdentity == nil,
                      event.conversationEpoch == nil else {
                    invalidateConnection(
                        admittedBy: attempt, responses: handles.responses,
                        error: AppInferenceError.modelLoadFailed(
                            "decode service load failure carried an unadmitted session"))
                    throw AppInferenceError.modelLoadFailed(
                        "decode service load failure carried an unadmitted session")
                }
                handles.lifetime?.noteProtocolCleanup()
                finishLoading(attempt)
                throw AppInferenceError.modelLoadFailed(
                    event.error ?? "decode service load failed")
            case .ready:
                break
            default:
                let error = AppInferenceError.modelLoadFailed(
                    "decode service returned \(event.kind.rawValue) for a load request")
                invalidateConnection(
                    admittedBy: attempt, responses: handles.responses, error: error)
                throw error
            }
            let binding: LoadedBinding
            do {
                binding = try Self.validatedBinding(
                    event, requestedToolThinking: options.toolThinkingEnabled)
            } catch {
                invalidateConnection(
                    admittedBy: attempt, responses: handles.responses, error: error)
                throw error
            }
            let published = sessionEffects.sync { () -> Bool in
                guard let identity = attempt.connectionIdentity else { return false }
                return connection.withLock { state in
                    guard state.currentLoadAttempt === attempt,
                          state.connectionID == identity.0,
                          state.revision == identity.1,
                          state.transport?.router === handles.responses,
                          state.binding == nil else { return false }
                    state.binding = binding
                    state.loadedDirectory = modelDirectory.standardizedFileURL
                    inferenceMemory.withLock { $0 = event.currentMemoryBytes }
                    inferenceTowerMemory.withLock { $0 = event.visionTowerMappedBytes }
                    conversationLogicalStateMemory.withLock {
                        $0 = event.conversationLogicalStateBytes
                    }
                    expertCacheMemory.withLock { $0 = event.expertCacheBytes }
                    return true
                }
            }
            guard published else { throw CancellationError() }
            if Task.isCancelled { attempt.requestTeardown() }
            if attempt.teardownRequested {
                finishLoadAttempt(attempt)
                guard await performBoundUnload(handles: handles, binding: binding) else {
                    await attempt.completion.wait()
                    throw CancellationError()
                }
                finishLoading(attempt)
                throw CancellationError()
            }
            sessionEffects.sync {
                onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
            }
            finishLoading(attempt)
        } catch {
            if attempt.commandAdmitted {
                attempt.requestTeardown()
                await sendLoadCancellation(attempt, handles: handles)
                // A protocol cleanup terminal is the only normal completion.
                if isCurrent(attempt, responses: handles.responses),
                   let terminal = try? await handles.responses.next(matching: request.requestID),
                   terminal.loadAttemptID == attemptID,
                   terminal.kind == .loadCancelled || terminal.kind == .failed,
                   terminal.loadedFamily == nil, terminal.loadID == nil,
                   terminal.modelIdentity == nil, terminal.sourceIdentity == nil,
                   terminal.conversationEpoch == nil {
                    handles.lifetime?.noteProtocolCleanup()
                    finishLoading(attempt)
                } else if isCurrent(attempt, responses: handles.responses) {
                    invalidateConnection(
                        admittedBy: attempt, responses: handles.responses,
                        error: AppInferenceError.modelLoadFailed(
                            "decode service load cleanup could not be proven"))
                }
                await settleFailedLoading(attempt)
            } else {
                finishLoading(attempt)
            }
            throw error
        }
    }

    public func unload() async {
        enum Work {
            case joinRetirements([TransportRetirementOwner])
            case joinLoad(LoadAttemptToken)
            case joinUnload(BoundTeardownOwner)
            case start(BoundTeardownOwner, Handles)
            case none
        }
        let work: Work = sessionEffects.sync {
            let retirements = retirementOwners.withLock { Array($0.values) }
            if !retirements.isEmpty { return .joinRetirements(retirements) }
            return lifecycle.withLock { value in
                if let value {
                    switch value {
                    case .loading(let attempt):
                        attempt.requestTeardown()
                        return .joinLoad(attempt)
                    case .unloading(let owner):
                        return .joinUnload(owner)
                    }
                }
                guard let handles = currentHandles(), let binding = handles.binding else {
                    return .none
                }
                let owner = BoundTeardownOwner(
                    binding: binding, lifetimeID: handles.lifetime?.id)
                value = .unloading(owner)
                return .start(owner, handles)
            }
        }
        switch work {
        case .none:
            return
        case .joinRetirements(let owners):
            for owner in owners { await owner.completion.wait() }
            await unload()
        case .joinLoad(let attempt):
            if let handles = currentHandles() {
                await sendLoadCancellation(attempt, handles: handles)
            }
            await attempt.completion.wait()
            // The setup owner may have settled before admitting a model frame
            // while preserving its prior binding. Join that exact binding too.
            await unload()
        case .joinUnload(let owner):
            await owner.completion.wait()
        case .start(let owner, let handles):
            guard await performBoundUnload(handles: handles, binding: owner.binding) else {
                // Cleanup remains unproven. Keep the owner unresolved so callers
                // cannot report notLoaded or admit a successor.
                await owner.completion.wait()
                return
            }
            lifecycle.withLock { value in
                if case .unloading(let current) = value, current === owner {
                    value = nil
                }
            }
            owner.completion.finish()
        }
    }

    private enum ResponseDisposition {
        case continueReceiving
        case stopReceiving
        case finish
        case finishThrowing(AppInferenceError)
    }

    private func admitResponse(
        _ event: DecodeServiceEvent,
        request: AppGenerationRequest,
        lease: ResponseEffectLease,
        responses: DecodeServiceResponseRouter,
        measurementConsumer: MeasurementConsumer?,
        continuation: AsyncThrowingStream<AppInferenceEvent, Error>.Continuation,
        expectedSequence: inout UInt64,
        lastMetricYield: inout Date,
        hasYieldedVisibleText: inout Bool,
        pendingThinkingToken: inout AppTokenEvent?,
        recoveryTerminal: inout (error: AppInferenceError, diagnostics: AppDiagnostics)?
    ) throws -> ResponseDisposition? {
        try admit(lease, responses: responses) { reserveTerminal in
            // This yield is intentionally inside the short lease section so
            // revocation and publication have one atomic order. Stream finish,
            // cleanup, trace calls, and cancellation callbacks stay outside.
            guard event.generationID == lease.operationID else {
                return .continueReceiving
            }
            if let measurementConsumer {
                guard measurementConsumer.performIfActive({
                    if let bytes = event.currentMemoryBytes {
                        inferenceMemory.withLock { $0 = bytes }
                    }
                    if let tower = event.visionTowerMappedBytes {
                        inferenceTowerMemory.withLock { $0 = tower }
                    }
                    publishRuntimeBytes(event)
                }) else {
                    if Self.isTerminal(event.kind) {
                        reserveTerminal = true
                        return .stopReceiving
                    }
                    return .continueReceiving
                }
            } else {
                if let bytes = event.currentMemoryBytes {
                    inferenceMemory.withLock { $0 = bytes }
                }
                if let tower = event.visionTowerMappedBytes {
                    inferenceTowerMemory.withLock { $0 = tower }
                }
                publishRuntimeBytes(event)
            }
            if event.kind == .measurement { return .continueReceiving }

            if event.kind == .toolCall {
                guard let call = event.toolCall,
                      let data = call.argumentsJSON.data(using: .utf8),
                      let arguments = try? JSONDecoder().decode(
                        JSONValue.self, from: data) else {
                    throw AppInferenceError.unknown(
                        "decode service returned an invalid tool call")
                }
                continuation.yield(.toolCall(AppToolCall(
                    id: call.id, name: call.name, arguments: arguments)))
                return .continueReceiving
            }
            if event.kind == .memory {
                continuation.yield(.memorySample)
                return .continueReceiving
            }
            if event.kind == .prefill || event.kind == .snapshot {
                guard event.sequence == expectedSequence else {
                    throw AppInferenceError.unknown(
                        "decode service event sequence changed from \(expectedSequence) to \(event.sequence)")
                }
                expectedSequence &+= 1
            }
            if event.kind == .prefill,
               let done = event.prefillDone,
               let total = event.prefillTotal {
                continuation.yield(.prefillProgress(done: done, total: total))
                return .continueReceiving
            }
            if event.kind == .snapshot {
                if let measurementConsumer {
                    guard measurementConsumer.performIfActive({
                        generationTranscriptMailbox.append(event.textDelta)
                    }) else { return .continueReceiving }
                } else {
                    generationTranscriptMailbox.append(event.textDelta)
                }
                pendingThinkingToken = AppTokenEvent(
                    index: max(0, event.tokenCount - 1), textDelta: "",
                    elapsedDecodeSeconds: event.decodeSeconds,
                    structuredProgress: event.structuredProgress,
                    thinkingPreview: event.thinkingPreview,
                    toolCallPreview: event.toolCallPreview)
                let now = Date()
                let beginsVisibleText = !hasYieldedVisibleText
                    && event.textDelta.contains { !$0.isWhitespace }
                if beginsVisibleText || now.timeIntervalSince(lastMetricYield) >= 0.5 {
                    lastMetricYield = now
                    hasYieldedVisibleText = hasYieldedVisibleText || beginsVisibleText
                    continuation.yield(.token(AppTokenEvent(
                        index: max(0, event.tokenCount - 1),
                        textDelta: beginsVisibleText ? event.textDelta : "",
                        elapsedDecodeSeconds: event.decodeSeconds,
                        structuredProgress: event.structuredProgress,
                        thinkingPreview: event.thinkingPreview,
                        toolCallPreview: event.toolCallPreview)))
                    pendingThinkingToken = nil
                }
                return .continueReceiving
            }

            if let pendingThinkingToken { continuation.yield(.token(pendingThinkingToken)) }
            pendingThinkingToken = nil
            let diagnostics = Self.diagnostics(event, options: request.runtimeOptions)
            switch event.kind {
            case .finished:
                continuation.yield(.finished(diagnostics))
                reserveTerminal = true
                return .finish
            case .cancelled:
                continuation.yield(.cancelled(diagnostics))
                reserveTerminal = true
                return .finish
            case .failed:
                let message = event.error ?? "decode service failed"
                let error: AppInferenceError
                var evidence: StructuredToolFailureEvidence?
                if request.captureToolFailureEvidence,
                   let json = event.parserFailureJSON,
                   let data = json.data(using: .utf8) {
                    evidence = try? JSONDecoder().decode(
                        StructuredToolFailureEvidence.self, from: data)
                }
                if let json = event.thoughtRepetitionRecoveryJSON {
                    if json.utf8.count <= 4_096,
                       event.parserFailureCanRegenerateToolResult == nil,
                       event.parserFailureJSON == nil,
                       event.conversationEpoch == request.conversationEpoch,
                       let data = json.data(using: .utf8),
                       let recovery = try? JSONDecoder().decode(
                        ThoughtRepetitionRecovery.self, from: data),
                       Self.admitsThoughtRepetitionRecovery(
                        recovery, maxContextTokens: request.maxContextTokens) {
                        error = .repeatedThought(recovery)
                    } else {
                        error = .invalidRequest(
                            "The repeated-generation recovery receipt was invalid. No retry was admitted.")
                    }
                } else if let canRegenerate = event.parserFailureCanRegenerateToolResult {
                    error = .structuredToolFailure(
                        message: message,
                        canRegenerateToolResult: canRegenerate,
                        evidence: evidence)
                } else {
                    error = .unknown(message)
                }
                if case .repeatedThought = error {
                    recoveryTerminal = (error, diagnostics)
                    return .stopReceiving
                }
                continuation.yield(.failed(error, partial: diagnostics))
                reserveTerminal = true
                return .finishThrowing(error)
            case .lineageLost:
                let error = AppInferenceError.conversationLineageLost(
                    event.error ?? "the conversation's KV no longer matches it")
                continuation.yield(.failed(error, partial: diagnostics))
                reserveTerminal = true
                return .finishThrowing(error)
            default:
                return .continueReceiving
            }
        }
    }

    /// The receipt numbers fit a detector rule: a block of 1 to 512 tokens, at
    /// least 4 copies, and thinking tokens for every reported copy.
    static func admitsThoughtRepetitionRecovery(
        _ recovery: ThoughtRepetitionRecovery, maxContextTokens: Int
    ) -> Bool {
        let copies = recovery.blockTokens.multipliedReportingOverflow(by: recovery.repetitions)
        return recovery.restoredTokenCount >= 0
            && recovery.restoredTokenCount <= maxContextTokens
            && (1...512).contains(recovery.blockTokens)
            && recovery.repetitions >= 4
            && recovery.generatedTokens > 0
            && recovery.generatedTokens <= maxContextTokens
            && recovery.generatedTokens >= recovery.thinkingTokens
            && !copies.overflow
            && recovery.thinkingTokens >= copies.partialValue
    }

    private static func isTerminal(_ kind: DecodeServiceEventKind) -> Bool {
        kind == .finished || kind == .cancelled || kind == .failed || kind == .lineageLost
    }

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let measurementConsumer = request.runtimeMeasurementCapture.map { _ in
                MeasurementConsumer()
            }
            currentMeasurementConsumer.withLock { $0 = measurementConsumer }
            let streamTerminationRequested = Mutex(false)
            let leaseBox = Mutex<ResponseEffectLease?>(nil)
            let operationTransport = Mutex<(
                writer: DecodeServiceCommandWriter,
                responses: DecodeServiceResponseRouter
            )?>(nil)
            let task = Task.detached(priority: .userInitiated) { [self] in
                var pendingThinkingToken: AppTokenEvent?
                var measurementReceiver: (id: UUID, responses: DecodeServiceResponseRouter)?
                var admittedOperation: ResponseEffectLease?
                var recoveryTerminal: (error: AppInferenceError, diagnostics: AppDiagnostics)?
                do {
                    guard !streamTerminationRequested.withLock({ $0 }) else {
                        throw CancellationError()
                    }
                    try request.validate()
                    guard lifecycle.withLock({ $0 == nil }) else {
                        throw AppInferenceError.reloadRequired
                    }
                    guard let handles = currentHandles(), let binding = handles.binding else {
                        throw AppInferenceError.modelNotLoaded
                    }
                    operationTransport.withLock {
                        $0 = (handles.writer, handles.responses)
                    }
                    let generationID = measurementConsumer?.generationID ?? UUID()
                    guard let session = sessionToken(for: handles, binding: binding) else {
                        throw CancellationError()
                    }
                    let operation = ResponseEffectLease(
                        session: session,
                        conversationEpoch: request.conversationEpoch,
                        operationID: generationID)
                    let installed = sessionEffects.sync {
                        guard !streamTerminationRequested.withLock({ $0 }),
                              lifecycle.withLock({ $0 == nil }) else { return false }
                        return connection.withLock { state -> Bool in
                            guard state.currentOperation == nil,
                                  state.connectionID == session.connectionID,
                                  state.revision == session.revision,
                                  state.transport?.router === handles.responses,
                                  state.binding == session.binding,
                                  state.openEpoch == request.conversationEpoch else {
                                return false
                            }
                            leaseBox.withLock { $0 = operation }
                            state.currentOperation = operation
                            return true
                        }
                    }
                    guard installed else { throw CancellationError() }
                    admittedOperation = operation
                    guard admit(operation, responses: handles.responses, { _ in
                        if let measurementConsumer {
                            return measurementConsumer.performIfActive {
                                generationTranscriptMailbox.reset()
                            }
                        }
                        generationTranscriptMailbox.reset()
                        return true
                    }) == true else { throw CancellationError() }
                    var command = DecodeGenerationRequest(
                        prompt: request.prompt,
                        imageAttachments: request.imageAttachments.map {
                            DecodeImageAttachment(
                                id: $0.id,
                                path: $0.fileURL.path,
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
                        runtimeOptions: Self.decodeRuntimeOptions(request.runtimeOptions),
                        generationID: generationID,
                        conversationEpoch: request.conversationEpoch,
                        turnIndex: request.turnIndex,
                        toolTurn: try Self.decodeToolTurn(request.toolTurn),
                        loadID: binding.loadID)
                    command.captureToolFailureEvidence = request.captureToolFailureEvidence ? true : nil
                    command.captureGPUCompletionTiming = request.captureGPUCompletionTiming ? true : nil
                    command.runtimeMeasurementCapture = request.runtimeMeasurementCapture
                    command.scopedCancellation = measurementConsumer == nil ? nil : true
                    let receivesMeasurements = command.runtimeMeasurementCapture != nil
                        && handles.responses.beginMeasurementReception(generationID)
                    if receivesMeasurements {
                        measurementReceiver = (generationID, handles.responses)
                    }
                    if let capture = command.runtimeMeasurementCapture, !receivesMeasurements {
                        command.runtimeMeasurementCapture = nil
                        var status = DecodeServiceEvent(
                            kind: .measurement, generationID: generationID,
                            loadedFamily: binding.family, loadID: binding.loadID,
                            modelIdentity: binding.modelIdentity,
                            sourceIdentity: binding.sourceIdentity,
                            conversationEpoch: request.conversationEpoch)
                        status.measurementCaptureID = capture.stepID
                        status.measurementBatchJSON = "{\"receiver_admission_refused\":1}"
                        status.measurementFinal = true
                        await runtimeMeasurementEvent(
                            status, capture, request.conversationEpoch, request.turnIndex)
                        guard admit(operation, responses: handles.responses, { _ in () }) != nil else {
                            throw CancellationError()
                        }
                    }
                    let receipt: DecodeServiceWriteReceipt
                    if let measurementConsumer {
                        receipt = try measurementConsumer.enqueueGeneration({
                            try sessionEffects.sync {
                                guard !streamTerminationRequested.withLock({ $0 }) else {
                                    throw CancellationError()
                                }
                                return try enqueue(
                                    .generate(command), admittedBy: operation,
                                    with: handles.writer, expecting: handles.responses)
                            }
                        }, stop: { [weak self] in
                            self?.cancel(
                                operation: operation, writer: handles.writer,
                                responses: handles.responses)
                        })
                    } else {
                        receipt = try sessionEffects.sync {
                            guard !streamTerminationRequested.withLock({ $0 }) else {
                                throw CancellationError()
                            }
                            return try enqueue(
                                .generate(command), admittedBy: operation,
                                with: handles.writer, expecting: handles.responses)
                        }
                    }
                    try await waitForSend(receipt, expecting: handles.responses)
                    guard admit(operation, responses: handles.responses, { _ in () }) != nil else {
                        throw CancellationError()
                    }

                    var expectedSequence: UInt64 = 1
                    var lastMetricYield = Date.distantPast
                    var hasYieldedVisibleText = false
                    receiveLoop: while true {
                        let event = try await handles.responses.next(matching: generationID)
                        guard admit(operation, responses: handles.responses, { _ in () }) != nil else {
                            throw CancellationError()
                        }
                        guard Self.matches(
                            event, binding: binding,
                            expectedEpoch: request.conversationEpoch) else {
                            let error = AppInferenceError.unknown(
                                "decode service event changed loaded identity or conversation epoch")
                            invalidateConnection(
                                admittedBy: operation, responses: handles.responses,
                                error: error)
                            throw error
                        }
                        if let capture = request.runtimeMeasurementCapture {
                            guard event.generationID == generationID else { continue }
                            await runtimeMeasurementEvent(
                                event, capture, request.conversationEpoch,
                                request.turnIndex)
                        }
                        guard let disposition = try admitResponse(
                            event, request: request, lease: operation,
                            responses: handles.responses,
                            measurementConsumer: measurementConsumer,
                            continuation: continuation,
                            expectedSequence: &expectedSequence,
                            lastMetricYield: &lastMetricYield,
                            hasYieldedVisibleText: &hasYieldedVisibleText,
                            pendingThinkingToken: &pendingThinkingToken,
                            recoveryTerminal: &recoveryTerminal) else {
                            throw CancellationError()
                        }
                        switch disposition {
                        case .continueReceiving:
                            continue receiveLoop
                        case .stopReceiving:
                            break receiveLoop
                        case .finish:
                            continuation.finish()
                            break receiveLoop
                        case .finishThrowing(let error):
                            continuation.finish(throwing: error)
                            break receiveLoop
                        }
                    }
                } catch {
                    if let admittedOperation,
                       let responses = operationTransport.withLock({ $0?.responses }) {
                        _ = admit(admittedOperation, responses: responses) { reserveTerminal in
                            if measurementConsumer?.isCancelled != true,
                               let pendingThinkingToken {
                                continuation.yield(.token(pendingThinkingToken))
                            }
                            reserveTerminal = true
                        }
                    }
                    continuation.finish(throwing: error)
                }
                // Cleanup is unconditional and exactly once. It is never
                // suppressed by a revoked response-effect lease.
                if let measurementConsumer {
                    measurementConsumer.finish()
                    currentMeasurementConsumer.withLock {
                        if $0 === measurementConsumer { $0 = nil }
                    }
                }
                if let measurementReceiver {
                    measurementReceiver.responses.abandonMeasurementReception(measurementReceiver.id)
                }
                if let capture = request.runtimeMeasurementCapture {
                    await runtimeMeasurementReceptionEnded(
                        capture, request.conversationEpoch, request.turnIndex)
                }
                if let recoveryTerminal {
                    let published: Bool
                    if let admittedOperation,
                       let responses = operationTransport.withLock({ $0?.responses }) {
                        published = admit(
                            admittedOperation, responses: responses
                        ) { reserveTerminal in
                            continuation.yield(.failed(
                                recoveryTerminal.error,
                                partial: recoveryTerminal.diagnostics))
                            reserveTerminal = true
                        } != nil
                    } else {
                        published = false
                    }
                    continuation.finish(throwing: published
                        ? recoveryTerminal.error : CancellationError())
                }
                if let admittedOperation { retireOperation(admittedOperation) }
            }
            continuation.onTermination = { [weak self] termination in
                guard case .cancelled = termination else { return }
                streamTerminationRequested.withLock { $0 = true }
                self?.retireOperation(storedIn: leaseBox)
                self?.scheduleStreamCancellation { [weak self] in
                    guard let self else { return }
                    if let measurementConsumer {
                        measurementConsumer.cancel()
                    } else if let lease = leaseBox.withLock({ $0 }),
                              let transport = operationTransport.withLock({ $0 }) {
                        self.cancel(
                            operation: lease, writer: transport.writer,
                            responses: transport.responses)
                    }
                    task.cancel()
                }
            }
        }
    }

    /// Opens `epoch` on the service and waits for it to say so.
    ///
    /// Not fire-and-forget: the app numbers the next turn from zero the moment
    /// this returns, and a turn numbered against a reset the service never
    /// applied is refused by its gate — which is the safe outcome, but the user
    /// sees a rejected message instead of a new chat.
    public func resetConversation(epoch: UUID) async throws {
        guard let handles = currentHandles(), let binding = handles.binding,
              let token = beginSessionMutation(
                handles: handles, binding: binding) else {
            throw AppInferenceError.unknown(
                "the decode service connection is gone; the new chat was not opened")
        }
        let requestID = UUID()
        let receipt = try enqueue(
            .resetConversation(DecodeResetConversationRequest(
                epoch: epoch, requestID: requestID, loadID: binding.loadID)),
            admittedBy: token, with: handles.writer,
            expecting: handles.responses)
        try await waitForSend(receipt, expecting: handles.responses)
        guard isCurrent(token, responses: handles.responses) else {
            throw CancellationError()
        }
        let event = try await handles.responses.next(matching: requestID)
        guard isCurrent(token, responses: handles.responses) else {
            throw CancellationError()
        }
        let expectedEpoch = event.kind == .conversationReset
            ? epoch : token.openEpoch
        guard Self.matches(
            event, binding: binding, expectedEpoch: expectedEpoch) else {
            let error = AppInferenceError.unknown(
                event.error ?? "decode service reset acknowledgement changed identity or epoch")
            invalidateConnection(
                admittedBy: token, responses: handles.responses, error: error)
            throw error
        }
        guard event.kind == .conversationReset else {
            throw AppInferenceError.unknown(
                event.error ?? "decode service refused to start a new conversation")
        }
        let applied = sessionEffects.sync {
            connection.withLock { state -> Bool in
                guard state.connectionID == token.connectionID,
                      state.revision == token.revision,
                      state.transport?.router === handles.responses,
                      state.binding == token.binding,
                      state.openEpoch == token.openEpoch else { return false }
                state.openEpoch = epoch
                return true
            }
        }
        guard applied else { throw CancellationError() }
        publishRuntimeBytes(event)
    }

    public func contextCheckpoint(_ request: DecodeContextCheckpointRequest) async throws
        -> DecodeContextCheckpointReceipt {
        try Task.checkCancellation()
        guard let handles = currentHandles(), let binding = handles.binding,
              let snapshot = sessionToken(for: handles, binding: binding) else {
            throw AppInferenceError.modelNotLoaded
        }
        guard snapshot.openEpoch == request.sourceEpoch else {
            throw AppInferenceError.conversationLineageLost(
                "Checkpoint source epoch is no longer current. The task stopped without replaying its action.")
        }
        guard let token = beginSessionMutation(
            handles: handles, binding: binding,
            requiringOpenEpoch: request.sourceEpoch) else {
            throw CancellationError()
        }
        var boundRequest = request
        boundRequest.loadID = binding.loadID
        let cancellation = AppGenerationStop()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            let sendReceipt = try enqueue(
                .contextCheckpoint(boundRequest), admittedBy: token,
                with: handles.writer, expecting: handles.responses)
            try await waitForSend(sendReceipt, expecting: handles.responses)
            guard isCurrent(token, responses: handles.responses) else {
                throw CancellationError()
            }
            cancellation.activate { [weak self] in
                self?.cancel(
                    operationID: request.requestID, session: token,
                    writer: handles.writer, responses: handles.responses)
            }
            defer { cancellation.finish() }
            let event = try await handles.responses.next(matching: request.requestID)
            guard isCurrent(token, responses: handles.responses) else {
                throw CancellationError()
            }
            let expectedEpoch = event.kind == .contextCheckpoint && request.commit
                ? request.replacementEpoch : token.openEpoch
            guard Self.matches(
                event, binding: binding, expectedEpoch: expectedEpoch) else {
                let error = AppInferenceError.conversationLineageLost(
                    "Checkpoint acknowledgement changed loaded identity or epoch. The task stopped without replaying its action.")
                invalidateConnection(
                    admittedBy: token, responses: handles.responses, error: error)
                throw error
            }
            guard event.kind == .contextCheckpoint, let receipt = event.contextCheckpoint,
                  receipt.checkpointID == request.checkpointID,
                  receipt.replacementEpoch == request.replacementEpoch,
                  receipt.committed == request.commit else {
                throw AppInferenceError.conversationLineageLost(event.error
                    ?? "Checkpoint acknowledgement did not match. The task stopped without replaying its action.")
            }
            if receipt.committed {
                let applied = sessionEffects.sync {
                    connection.withLock { state -> Bool in
                        guard state.connectionID == token.connectionID,
                              state.revision == token.revision,
                              state.transport?.router === handles.responses,
                              state.binding == token.binding,
                              state.openEpoch == token.openEpoch else { return false }
                        state.openEpoch = receipt.replacementEpoch
                        return true
                    }
                }
                guard applied else { throw CancellationError() }
            }
            publishRuntimeBytes(event)
            // Return a committed acknowledgement even if Stop raced it. The
            // app must record the new epoch before reporting interruption.
            return receipt
        } onCancel: {
            cancellation.requestStop()
        }
    }

    public func cancel() {
        if let measurementConsumer = currentMeasurementConsumer.withLock({ $0 }) {
            measurementConsumer.requestStop()
            return
        }
        guard let snapshot = connection.withLock({ state
            -> (ResponseEffectLease, DecodeServiceCommandWriter,
                DecodeServiceResponseRouter)? in
            guard let operation = state.currentOperation,
                  operation.isActive,
                  let transport = state.transport else { return nil }
            return (operation, transport.writer, transport.router)
        }) else { return }
        cancel(
            operation: snapshot.0, writer: snapshot.1, responses: snapshot.2)
    }

    /// Disconnects synchronously but only requests launchd cleanup. Waiting for
    /// `bootout` here froze the main actor when the window closed during a run.
    public func shutdownForTermination() {
        let shutdownFrame = try? DecodeFrameCodec.encode(DecodeServiceCommand.shutdown)
        if let retirement = takeConnectionForTermination() {
            retire(retirement, shutdownFrame: shutdownFrame)
        }
    }

    deinit {
        let shutdownFrame = try? DecodeFrameCodec.encode(DecodeServiceCommand.shutdown)
        if let retirement = takeConnectionForTermination() {
            retire(retirement, shutdownFrame: shutdownFrame)
        }
    }

    private func takeConnectionForTermination() -> ReservedRetirement? {
        let setupWorker = activeSetupWorker.withLock { value -> ServiceSetupWorker? in
            defer { value = nil }
            return value
        }
        setupWorker?.requestCancellation()
        let taken = sessionEffects.sync {
            connection.withLock { value
                -> (ReservedRetirement?, [ServiceLifetime]) in
                Self.revokeLoadAttemptLocked(&value)
                Self.revokeOperationLocked(&value)
                let dead = value
                let reserved = reserveRetirementLocked(dead)
                var observations = [ServiceLifetime]()
                if let lifetime = dead.lifetime { observations.append(lifetime) }
                pendingLifetime.withLock { pending in
                    if let pending { observations.append(pending) }
                    pending = nil
                }
                retiredLifetime.withLock { retired in
                    if let retired { observations.append(retired) }
                    retired = nil
                }
                quarantinedLifetime.withLock { quarantine in
                    if let quarantine { observations.append(quarantine) }
                    quarantine = nil
                }
                let retirement = reserved.map {
                    ReservedRetirement(
                        dead: $0.dead, owner: $0.owner,
                        observationsToCancel: observations)
                }
                inferenceMemory.withLock { $0 = nil }
                inferenceTowerMemory.withLock { $0 = nil }
                conversationLogicalStateMemory.withLock { $0 = nil }
                expertCacheMemory.withLock { $0 = nil }
                value = Connection()
                return (retirement, observations)
            }
        }
        if taken.0 == nil {
            taken.1.forEach { $0.cancelObservation() }
        }
        return taken.0
    }

    private func ensureProcessAsync() async throws
        -> (writer: DecodeServiceCommandWriter, responses: DecodeServiceResponseRouter) {
        guard retirementOwners.withLock({ $0.isEmpty }),
              quarantinedLifetime.withLock({ $0 == nil }) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service cleanup could not be proven")
        }
        if let handles = currentHandles() {
            return (handles.writer, handles.responses)
        }
        guard retirementOwners.withLock({ $0.isEmpty }),
              quarantinedLifetime.withLock({ $0 == nil }) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service cleanup could not be proven")
        }
        let worker = makeSetupWorker()
        let registered = activeSetupWorker.withLock { value -> Bool in
            guard value == nil else { return false }
            value = worker
            return true
        }
        guard registered else { throw RealInferenceLifecycleError.lifecycleInProgress }
        defer {
            activeSetupWorker.withLock { value in
                if value === worker { value = nil }
            }
        }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let thread = Thread { [self] in
                    do {
                        continuation.resume(returning: try launchIndependentService(
                            worker: worker))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
                thread.name = "TurboFieldfare.DecodeService.Setup"
                thread.qualityOfService = .userInitiated
                thread.start()
            }
        } onCancel: {
            worker.requestCancellation()
        }
    }

    private func launchIndependentService(worker: ServiceSetupWorker) throws
        -> (writer: DecodeServiceCommandWriter, responses: DecodeServiceResponseRouter) {
        guard FileManager.default.isExecutableFile(atPath: serviceURL.path) else {
            throw AppInferenceError.modelLoadFailed(
                "decode service executable is missing at \(serviceURL.path); run swift build -c release before launching the app")
        }
        let identifier = "\(getuid()).\(getpid()).\(UUID().uuidString.lowercased())"
        let label = "com.turbofieldfare.decode.\(identifier)"
        let socketPath = "/private/tmp/turbofieldfare-decode-\(identifier).sock"
        var serviceHandedOff = false
        defer {
            if !serviceHandedOff { Self.removeLaunchJob(label: label) }
        }
        let propertyListURL = URL(
            fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("\(label).plist")
        let propertyList: [String: Any] = [
            "Label": label,
            "ProgramArguments": [
                serviceURL.path,
                "--socket", socketPath,
                "--launch-label", label,
            ],
            "RunAtLoad": true,
            "KeepAlive": false,
            "ProcessType": "Interactive",
            // Preserve the launch default for callers without explicit load
            // settings. The app sends its saved thinking choice with each load.
            "EnvironmentVariables": Self.launchEnvironment(
                environment: ProcessInfo.processInfo.environment,
                thinking: GFTokenizer.toolThinkingEnabled),
        ]
        let propertyListData = try PropertyListSerialization.data(
            fromPropertyList: propertyList, format: .xml, options: 0)
        try propertyListData.write(to: propertyListURL, options: .atomic)
        defer { try? FileManager.default.removeItem(at: propertyListURL) }

        let launcher = Process()
        let errors = Pipe()
        launcher.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        launcher.arguments = [
            "bootstrap", "gui/\(getuid())", propertyListURL.path,
        ]
        launcher.standardOutput = FileHandle.nullDevice
        launcher.standardError = errors
        try worker.runAndWait(launcher)
        guard launcher.terminationStatus == 0 else {
            let data = try? errors.fileHandleForReading.readToEnd()
            let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let message = detail.flatMap { $0.isEmpty ? nil : $0 }
                ?? "launchd could not start the decode service"
            throw AppInferenceError.modelLoadFailed(message)
        }

        // Demand the job without restarting a RunAtLoad winner; the socket is
        // authoritative. `bootstrap` alone does not always leave the job
        // running, and the failure then looked like a socket timeout with no
        // indication that launchd was the cause.
        var kickstartError: String?
        do {
            let starter = Process()
            let starterErrors = Pipe()
            starter.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            starter.arguments = Self.kickstartArguments(uid: getuid(), label: label)
            starter.standardOutput = FileHandle.nullDevice
            starter.standardError = starterErrors
            try worker.runAndWait(starter)
            if starter.terminationStatus != 0 {
                let data = try? starterErrors.fileHandleForReading.readToEnd()
                let detail = data.flatMap { String(data: $0, encoding: .utf8) }?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                kickstartError = detail.flatMap { $0.isEmpty ? nil : $0 }
                    ?? "exit status \(starter.terminationStatus)"
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            kickstartError = String(describing: error)
        }

        var lastError: Error?
        for _ in 0..<200 {
            try worker.beforeConnectAttempt()
            do {
                let handles = try DecodeUnixSocket.connect(path: socketPath)
                let peer = try DecodeServicePeerIdentity.capture(
                    from: handles.input.fileDescriptor)
                let lifetimeID = UUID()
                let lifetime = ServiceLifetime(id: lifetimeID, peer: peer)
                lifecycle.withLock { value in
                    if case .loading(let attempt) = value {
                        attempt.bindPendingLifetime(lifetimeID)
                    }
                }
                var lifetimeInstalled = false
                var handshakeOwnsFailureCleanup = false
                defer {
                    if !lifetimeInstalled && !handshakeOwnsFailureCleanup {
                        cleanupFailedStartupConnection(
                            input: handles.input, output: handles.output,
                            lifetime: lifetime)
                    }
                }
                let observer = ServiceProcessObserver(pid: peer.pid) { [weak self] event in
                    worker.requestCancellation()
                    self?.handleLifetimeEvent(lifetimeID: lifetimeID, event: event)
                }
                lifetime.attach(observer: observer)
                pendingLifetime.withLock { $0 = lifetime }
                observer.start()
                guard observer.waitForRegistration() == .success,
                      lifetime.isAuthoritativeIdle else {
                    pendingLifetime.withLock { value in
                        if value?.id == lifetimeID { value = nil }
                    }
                    lifetime.cancelObservation()
                    try? handles.input.close()
                    try? handles.output.close()
                    throw AppInferenceError.modelLoadFailed(
                        "decode service process observation could not be established")
                }
                handshakeOwnsFailureCleanup = true
                try performStartupHandshakeAndInstall(
                    input: handles.input, output: handles.output,
                    worker: worker, lifetime: lifetime)
                lifetimeInstalled = true
                pendingLifetime.withLock { value in
                    if value?.id == lifetimeID { value = nil }
                }
                guard let installed = connection.withLock({ state
                    -> (DecodeServiceCommandWriter, DecodeServiceResponseRouter)? in
                    state.launchLabel = label
                    state.socketPath = socketPath
                    guard state.lifetime?.id == lifetimeID,
                          lifetime.isAuthoritativeIdle,
                          quarantinedLifetime.withLock({ $0 == nil }),
                          let transport = state.transport else { return nil }
                    return (transport.writer, transport.router)
                }) else {
                    invalidateConnection(error: DecodeServiceCommandWriterError.closed)
                    throw DecodeServiceCommandWriterError.closed
                }
                serviceHandedOff = true
                return installed
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                if quarantinedLifetime.withLock({ $0 != nil }) { throw error }
                lastError = error
                try worker.waitBeforeRetry(milliseconds: 10)
            }
        }
        throw AppInferenceError.modelLoadFailed(
            Self.socketFailureMessage(
                socketError: lastError.map(String.init(describing:)),
                kickstartError: kickstartError))
    }

    /// Performs the production startup exchange and publishes its transport only
    /// after the model-independent acknowledgement is exact. This method owns
    /// local failure cleanup so tests exercise the same fail-closed path.
    func performStartupHandshakeAndInstall(
        input: FileHandle,
        output: FileHandle,
        worker: ServiceSetupWorker,
        lifetime: ServiceLifetime,
        nonce: UUID = UUID()
    ) throws {
        do {
            try worker.checkCancellation()
            try worker.beginProbe(input: input, output: output)
            defer { worker.endProbe() }
            try input.write(contentsOf: DecodeFrameCodec.encode(
                DecodeServiceCommand.lifetimeProbe(
                    DecodeLifetimeProbe(nonce: nonce))))
            let acknowledgement = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: output)
            guard acknowledgement.kind == .lifetimeAcknowledged,
                  acknowledgement.generationID == nonce,
                  acknowledgement.loadAttemptID == nil,
                  acknowledgement.lifetimeNonce == nonce,
                  acknowledgement.loadedFamily == nil,
                  acknowledgement.loadID == nil,
                  acknowledgement.modelIdentity == nil,
                  acknowledgement.sourceIdentity == nil,
                  acknowledgement.conversationEpoch == nil,
                  lifetime.isAuthoritativeIdle else {
                throw AppInferenceError.modelLoadFailed(
                    "decode service lifetime acknowledgement was invalid")
            }
            try worker.completeProbeHandoff()
            installTransport(input: input, output: output, lifetime: lifetime)
        } catch {
            cleanupFailedStartupConnection(
                input: input, output: output, lifetime: lifetime)
            throw error
        }
    }

    private func cleanupFailedStartupConnection(
        input: FileHandle,
        output: FileHandle,
        lifetime: ServiceLifetime
    ) {
        pendingLifetime.withLock { value in
            if value?.id == lifetime.id { value = nil }
        }
        let retained = retiredLifetime.withLock {
            $0?.id == lifetime.id
        } || quarantinedLifetime.withLock {
            $0?.id == lifetime.id
        }
        if !retained { lifetime.cancelObservationAndWait() }
        try? input.close()
        try? output.close()
    }

    /// `kickstart` demands the job; it deliberately omits `-k`, which would
    /// restart a job that a RunAtLoad bootstrap already won. The socket is what
    /// decides readiness, so restarting a healthy service would only lose it.
    /// Per-job trial opt-in only. No process-wide or launchd environment writes.
    static func launchEnvironment(environment: [String: String], thinking: Bool) -> [String: String] {
        var result = ["TURBOFIELDFARE_AGENT_THINKING": thinking ? "1" : "0"]
        if environment["TURBO_QWEN_GPU_LINEAR_PREPARATION"] == "1" {
            result["TURBO_QWEN_GPU_LINEAR_PREPARATION"] = "1"
        }
        if environment["TURBO_QWEN_EXPERT_CACHE_RESIDENCY"] == "1" {
            result["TURBO_QWEN_EXPERT_CACHE_RESIDENCY"] = "1"
        }
        if environment["TURBO_QWEN_GROUPED_LINEAR_PREFILL"] == "1" {
            result["TURBO_QWEN_GROUPED_LINEAR_PREFILL"] = "1"
        }
        if environment["TURBO_QWEN_EXPERT_PROJECTION_BATCH"] == "1" {
            result["TURBO_QWEN_EXPERT_PROJECTION_BATCH"] = "1"
        }
        if environment["TURBO_QWEN_SOURCE_VALIDATION_FAST"] == "1" {
            result["TURBO_QWEN_SOURCE_VALIDATION_FAST"] = "1"
        }
        if environment["TURBO_QWEN_SOURCE_MEMBERSHIP_SCAN"] == "1" {
            result["TURBO_QWEN_SOURCE_MEMBERSHIP_SCAN"] = "1"
        }
        if environment["TURBO_QWEN_EXACT_TOKEN_CAPTURE"] == "1" {
            result["TURBO_QWEN_EXACT_TOKEN_CAPTURE"] = "1"
            if let directory = environment["TURBO_QWEN_EXACT_TOKEN_CAPTURE_DIRECTORY"] {
                result["TURBO_QWEN_EXACT_TOKEN_CAPTURE_DIRECTORY"] = directory
            }
        }
        return result
    }

    static func kickstartArguments(uid: uid_t, label: String) -> [String] {
        ["kickstart", "gui/\(uid)/\(label)"]
    }

    /// The socket timeout stays the primary diagnostic, because that is what the
    /// caller actually observed; a launchctl failure is appended when there was
    /// one, since it usually explains the timeout.
    static func socketFailureMessage(socketError: String?, kickstartError: String?)
        -> String {
        let kickstartDetail = kickstartError.map {
            "; launchctl kickstart failed: \($0)"
        } ?? ""
        return "decode service socket did not become ready: \(socketError ?? "unknown error")\(kickstartDetail)"
    }

    private func currentHandles() -> Handles? {
        let handles = connection.withLock { state -> Handles? in
            guard let transport = state.transport,
                  let connectionID = state.connectionID else { return nil }
            return Handles(
                writer: transport.writer, responses: transport.router,
                connectionID: connectionID, revision: state.revision,
                binding: state.binding, lifetime: state.lifetime)
        }
        guard let handles else { return nil }
        if handles.responses.isTerminated {
            invalidateConnection(expecting: handles.responses)
            return nil
        }
        return handles
    }

    private static func validatedBinding(
        _ event: DecodeServiceEvent,
        requestedToolThinking: Bool
    ) throws -> LoadedBinding {
        guard let family = event.loadedFamily, let loadID = event.loadID,
              event.conversationEpoch == nil else {
            throw AppInferenceError.modelLoadFailed(
                "decode service readiness omitted its loaded family or incarnation")
        }
        try event.validateBackingIdentity()
        let binding = LoadedBinding(
            loadID: loadID, family: family, modelIdentity: event.modelIdentity,
            sourceIdentity: event.sourceIdentity,
            toolThinkingEnabled: event.toolThinkingEnabled)
        switch family {
        case .gemma4:
            guard event.modelIdentity == nil, event.sourceIdentity == nil,
                  event.toolThinkingEnabled == requestedToolThinking else {
                throw AppInferenceError.modelLoadFailed(
                    "Gemma readiness identity or thinking mode was inconsistent")
            }
        case .qwen3_6:
            guard event.toolThinkingEnabled == nil,
                  binding.readiness != nil else {
                throw AppInferenceError.modelLoadFailed(
                    "Qwen readiness omitted or mismatched its verified identity")
            }
        }
        return binding
    }

    private static func matches(
        _ event: DecodeServiceEvent,
        binding: LoadedBinding,
        expectedEpoch: UUID?
    ) -> Bool {
        matches(event, family: binding.family, loadID: binding.loadID,
                modelIdentity: binding.modelIdentity,
                sourceIdentity: binding.sourceIdentity,
                expectedEpoch: expectedEpoch)
    }

    /// The receiver uses this exact identity comparison for every bound frame.
    /// Kept internal so tests can check stale source frames without a process.
    static func matches(
        _ event: DecodeServiceEvent,
        family: DecodeModelFamily,
        loadID: UUID,
        modelIdentity: DecodeModelIdentity?,
        sourceIdentity: DecodeSourceIdentity?,
        expectedEpoch: UUID?
    ) -> Bool {
        event.loadedFamily == family
            && event.loadID == loadID
            && event.modelIdentity == modelIdentity
            && event.sourceIdentity == sourceIdentity
            && event.conversationEpoch == expectedEpoch
    }

    private static func diagnostics(_ event: DecodeServiceEvent,
                                    options: AppRuntimeOptions) -> AppDiagnostics {
        let stop = AppStopReason(rawValue: event.stopReason ?? "")
            ?? (event.kind == .cancelled ? .cancelled
                : (event.kind == .failed || event.kind == .lineageLost) ? .failed
                : .maxTokens)
        return AppDiagnostics(
            generatedTokens: event.tokenCount,
            stopReason: stop,
            promptTokenCount: event.promptTokenCount,
            cachedPromptTokens: event.cachedPromptTokens,
            computedPrefillTokens: event.computedPrefillTokens,
            conversationTokens: event.conversationTokenCount,
            prefillSeconds: event.prefillSeconds,
            timeToFirstTokenSeconds: event.timeToFirstTokenSeconds,
            decodeSeconds: event.decodeSeconds,
            tokensPerSecond: event.tokensPerSecond,
            peakMemoryBytes: event.peakMemoryBytes,
            visionTowerMappedBytes: event.visionTowerMappedBytes,
            conversationLogicalStateBytes: event.conversationLogicalStateBytes,
            expertCacheBytes: event.expertCacheBytes,
            runtimeOptions: options,
            prefill: prefillDiagnostics(event.prefill, options: options),
            runner: event.runner.map(runnerDiagnostics),
            structuredProgress: event.structuredProgress)
    }

    private static func prefillDiagnostics(
        _ value: DecodePrefillDiagnostics?, options: AppRuntimeOptions
    ) -> PrefillExecutionDiagnostics? {
        guard let value,
              let executedMode = PrefillExecutedMode(rawValue: value.executedMode),
              let completeness = PrefillChunkCompleteness(
                rawValue: value.chunkCompleteness) else { return nil }
        let kvStorage = value.kvStorageMode.flatMap(PrefillKVStorageMode.init(rawValue:))
        return PrefillExecutionDiagnostics(
            config: options.prefillConfig,
            executedMode: executedMode,
            kvStorageMode: kvStorage,
            chunkCompleteness: completeness,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: DecodeRunnerDiagnostics)
        -> AppRunnerDiagnostics {
        AppRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            routerWaitMillisecondsPerToken: value.routerWaitMillisecondsPerToken,
            gpuCompletionTiming: value.gpuCompletionTiming,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures,
            gpuExpertCacheEligibleForwards: value.gpuExpertCacheEligibleForwards ?? 0,
            gpuExpertCacheBatches: value.gpuExpertCacheBatches ?? 0,
            gpuExpertCacheHitExperts: value.gpuExpertCacheHitExperts ?? 0,
            gpuExpertCacheFirstMisses: value.gpuExpertCacheFirstMisses ?? 0,
            gpuExpertCacheCPUFallbackLayers: value.gpuExpertCacheCPUFallbackLayers ?? 0)
    }

    private static func decodeRuntimeOptions(_ options: AppRuntimeOptions)
        -> DecodeRuntimeOptions {
        DecodeRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: options.expertCachePolicy.rawValue,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: options.rdadvisePolicy.rawValue,
            modelVerification: options.modelVerification.rawValue,
            visionResidencyPolicy: options.visionResidencyPolicy.rawValue,
            toolThinkingEnabled: options.toolThinkingEnabled)
    }

    private static func decodeToolTurn(_ turn: AppToolTurn?) throws
        -> DecodeToolTurn? {
        switch turn {
        case .checkpoint(let id):
            return .checkpoint(id)
        case .user(let developerPrompt, let tools):
            return .user(
                developerPrompt: developerPrompt,
                tools: try tools.map {
                    DecodeToolDefinition(
                        name: $0.name,
                        description: $0.description,
                        parametersJSON: try $0.parameters.encoded())
                })
        case .results(let results):
            return .results(results.map {
                    DecodeToolResult(
                        callID: $0.callID,
                        name: $0.name,
                        content: $0.content,
                        imageAttachments: $0.imageAttachments.isEmpty ? nil : $0.imageAttachments.map {
                            DecodeImageAttachment(
                                id: $0.id, path: $0.fileURL.path, displayName: $0.displayName,
                                encodedBytes: $0.encodedBytes, sha256: $0.sha256)
                        })
            })
        case nil:
            return nil
        }
    }

    private static func removeLaunchJob(label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        // Do not wait here. This path is used by app termination and transport
        // failure handling; launchd owns the process cleanup once the request
        // has been accepted.
        try? process.run()
    }

    private static func defaultServiceURL() -> URL {
        return Bundle.main.executableURL!
            .deletingLastPathComponent()
            .appendingPathComponent("TurboFieldfareDecodeService")
    }
}
