import Foundation
import Synchronization
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

/// Synchronous admission state shared by the command reader and service loop.
/// Binding, operation, and lifecycle decisions use one mutex. No lock is held
/// while callbacks, continuations, frame I/O, or model work run.
final class DecodeServiceSession: Sendable {
    enum Rejection: Error, Equatable {
        case inconsistentReadiness
        case lifecycleInProgress
        case invalidLifecycleLease
        case loadCancelled
    }

    struct LoadAttemptPair: Hashable, Sendable {
        let requestID: UUID
        let attemptID: UUID?

        init(_ request: DecodeLoadRequest) {
            requestID = request.requestID
            attemptID = request.attemptID
        }
    }

    struct LoadLease: Hashable, Sendable {
        fileprivate let id: UUID
        let pair: LoadAttemptPair
        var requestID: UUID { pair.requestID }
        var attemptID: UUID? { pair.attemptID }
    }

    struct TeardownLease: Hashable, Sendable {
        fileprivate let id: UUID
        let requestID: UUID
        let loadID: UUID
    }

    struct Binding: Sendable, Equatable {
        let loadID: UUID
        let family: DecodeModelFamily
        let modelIdentity: DecodeModelIdentity?
        let sourceIdentity: DecodeSourceIdentity?
        let toolThinkingEnabled: Bool?
    }

    struct OperationKey: Hashable, Sendable {
        let loadID: UUID
        let operationID: UUID
    }

    enum CommitResult: Sendable, Equatable {
        case bound(Binding)
        case teardown(TeardownLease, Binding)
    }

    private struct Loading: Sendable {
        let lease: LoadLease
        var cancellationRequested = false
    }

    private struct ReadyWrite: Sendable {
        let lease: LoadLease
        let binding: Binding
        var pendingTeardown: TeardownLease?
    }

    private struct BoundTeardown: Sendable {
        let lease: TeardownLease
        let binding: Binding
    }

    private enum Lifecycle: Sendable {
        case available
        case loading(Loading)
        case readyWrite(ReadyWrite)
        case boundTeardown(BoundTeardown)
        case failureTeardown(LoadLease)
    }

    private struct State: Sendable {
        var binding: Binding?
        var stops: [OperationKey: AppGenerationStop] = [:]
        var lifecycle: Lifecycle = .available
        var lastCancelledLease: LoadLease?
    }

    private let state = Mutex(State())
    private let maximumOperations: Int

    init(maximumOperations: Int = 8) {
        self.maximumOperations = maximumOperations
    }

    /// Input-thread admission. A competing load never replaces or queues
    /// behind the exact active lifecycle owner.
    func registerLoad(_ request: DecodeLoadRequest) -> LoadLease? {
        // A legacy load still receives an opaque internal lease. Because its
        // attemptID is nil, no modern cancel pair can gain authority over it.
        let result = state.withLock { value -> (LoadLease?, [AppGenerationStop]) in
            guard case .available = value.lifecycle else { return (nil, []) }
            let lease = LoadLease(id: UUID(), pair: LoadAttemptPair(request))
            value.lastCancelledLease = nil
            let retired = Array(value.stops.values)
            value.stops.removeAll(keepingCapacity: true)
            value.binding = nil
            value.lifecycle = .loading(Loading(lease: lease))
            return (lease, retired)
        }
        result.1.forEach { $0.finish() }
        return result.0
    }

    /// Input-thread cancellation interception. Unknown, legacy, stale, or
    /// publication-won controls allocate no state and have no successor effect.
    @discardableResult
    func markCancel(_ request: DecodeCancelLoadRequest) -> Bool {
        state.withLock { value in
            guard case .loading(var loading) = value.lifecycle,
                  loading.lease.pair.requestID == request.requestID,
                  loading.lease.pair.attemptID == request.attemptID else {
                return false
            }
            loading.cancellationRequested = true
            value.lastCancelledLease = loading.lease
            value.lifecycle = .loading(loading)
            return true
        }
    }

    func loadLease(for request: DecodeLoadRequest) -> LoadLease? {
        state.withLock { value in
            switch value.lifecycle {
            case .loading(let loading) where loading.lease.pair == LoadAttemptPair(request):
                return loading.lease
            case .readyWrite(let ready) where ready.lease.pair == LoadAttemptPair(request):
                return ready.lease
            case .failureTeardown(let lease) where lease.pair == LoadAttemptPair(request):
                return lease
            default:
                return nil
            }
        }
    }

    func requireActive(_ lease: LoadLease) throws {
        try state.withLock { value in
            switch value.lifecycle {
            case .loading(let loading) where loading.lease == lease:
                if loading.cancellationRequested { throw Rejection.loadCancelled }
            case .readyWrite(let ready) where ready.lease == lease:
                break
            case .failureTeardown(let active) where active == lease:
                throw Rejection.invalidLifecycleLease
            default:
                throw Rejection.invalidLifecycleLease
            }
        }
    }

    func wasCancellationRequested(_ lease: LoadLease) -> Bool {
        state.withLock { $0.lastCancelledLease == lease }
    }

    /// Publication reservation and binding installation are one synchronous
    /// mutation. Once this wins, a later load cancellation is inert.
    func reserveAndPublish(
        _ readiness: AppLoadedModelReadiness,
        lease: LoadLease,
        loadID: UUID = UUID()
    ) throws -> Binding {
        let binding = try Self.binding(readiness, loadID: loadID)
        return try state.withLock { value in
            guard case .loading(let loading) = value.lifecycle,
                  loading.lease == lease else {
                throw Rejection.invalidLifecycleLease
            }
            guard !loading.cancellationRequested else {
                throw Rejection.loadCancelled
            }
            value.binding = binding
            value.lifecycle = .readyWrite(ReadyWrite(
                lease: lease, binding: binding, pendingTeardown: nil))
            return binding
        }
    }

    /// Validates the exact nonnil binding before allocating teardown state.
    /// A just-published ready joins the retained load lease and transfers only
    /// when finishCommit executes.
    func registerBoundTeardown(
        requestID: UUID, loadID: UUID?
    ) -> TeardownLease? {
        guard let loadID else { return nil }
        let result = state.withLock { value -> (TeardownLease?, [AppGenerationStop]) in
            guard let binding = value.binding, binding.loadID == loadID else {
                return (nil, [])
            }
            let lease = TeardownLease(id: UUID(), requestID: requestID, loadID: loadID)
            switch value.lifecycle {
            case .available:
                let stops = Array(value.stops.values)
                value.stops.removeAll(keepingCapacity: true)
                value.binding = nil
                value.lifecycle = .boundTeardown(BoundTeardown(
                    lease: lease, binding: binding))
                return (lease, stops)
            case .readyWrite(var ready) where ready.binding == binding
                    && ready.pendingTeardown == nil:
                ready.pendingTeardown = lease
                value.lifecycle = .readyWrite(ready)
                return (lease, [])
            default:
                return (nil, [])
            }
        }
        result.1.forEach { $0.finish() }
        return result.0
    }

    func teardownLease(for request: DecodeUnloadRequest) -> TeardownLease? {
        guard let loadID = request.loadID else { return nil }
        return state.withLock { value in
            switch value.lifecycle {
            case .boundTeardown(let teardown)
                where teardown.lease.requestID == request.requestID
                    && teardown.lease.loadID == loadID:
                return teardown.lease
            case .readyWrite(let ready)
                where ready.pendingTeardown?.requestID == request.requestID
                    && ready.pendingTeardown?.loadID == loadID:
                return ready.pendingTeardown
            default:
                return nil
            }
        }
    }

    func finishCommit(_ lease: LoadLease) throws -> CommitResult {
        try state.withLock { value in
            guard case .readyWrite(let ready) = value.lifecycle,
                  ready.lease == lease else {
                throw Rejection.invalidLifecycleLease
            }
            if let teardown = ready.pendingTeardown {
                value.binding = nil
                value.lifecycle = .boundTeardown(BoundTeardown(
                    lease: teardown, binding: ready.binding))
                return .teardown(teardown, ready.binding)
            }
            value.lifecycle = .available
            return .bound(ready.binding)
        }
    }

    /// Retains exact lifecycle ownership while runtime cleanup is in progress.
    func beginLoadFailure(_ lease: LoadLease) {
        state.withLock { value in
            switch value.lifecycle {
            case .loading(let loading) where loading.lease == lease:
                value.binding = nil
                value.lifecycle = .failureTeardown(lease)
            case .readyWrite(let ready) where ready.lease == lease:
                value.binding = nil
                value.lifecycle = .failureTeardown(lease)
            default:
                break
            }
        }
    }

    func finishLoadFailure(_ lease: LoadLease) {
        state.withLock { value in
            guard case .failureTeardown(let active) = value.lifecycle,
                  active == lease else { return }
            value.binding = nil
            value.lifecycle = .available
        }
    }

    func finishTeardown(_ lease: TeardownLease) -> Binding? {
        state.withLock { value in
            guard case .boundTeardown(let active) = value.lifecycle,
                  active.lease == lease else { return nil }
            value.lifecycle = .available
            return active.binding
        }
    }

    func binding(for lease: TeardownLease) -> Binding? {
        state.withLock { value in
            guard case .boundTeardown(let active) = value.lifecycle,
                  active.lease == lease else { return nil }
            return active.binding
        }
    }

    @discardableResult
    func publish(_ readiness: AppLoadedModelReadiness, loadID: UUID = UUID()) throws -> Binding {
        let binding = try Self.binding(readiness, loadID: loadID)
        let retired = try state.withLock { value -> [AppGenerationStop] in
            guard case .available = value.lifecycle else {
                throw Rejection.lifecycleInProgress
            }
            let retired = Array(value.stops.values)
            value.stops.removeAll(keepingCapacity: true)
            value.binding = binding
            return retired
        }
        retired.forEach { $0.finish() }
        return binding
    }

    private static func binding(
        _ readiness: AppLoadedModelReadiness, loadID: UUID
    ) throws -> Binding {
        switch readiness {
        case .gemma(let toolThinkingEnabled):
            return Binding(
                loadID: loadID, family: .gemma4, modelIdentity: nil,
                sourceIdentity: nil, toolThinkingEnabled: toolThinkingEnabled)
        case .qwen(let identity):
            guard identity.family == .qwen3_6 else {
                throw Rejection.inconsistentReadiness
            }
            return Binding(
                loadID: loadID, family: .qwen3_6, modelIdentity: identity,
                sourceIdentity: nil, toolThinkingEnabled: nil)
        case .qwenSource(let identity):
            return Binding(
                loadID: loadID, family: .qwen3_6, modelIdentity: nil,
                sourceIdentity: identity, toolThinkingEnabled: nil)
        }
    }

    var currentBinding: Binding? { state.withLock { $0.binding } }

    func binding(loadID: UUID?) -> Binding? {
        guard let loadID else { return nil }
        return state.withLock { value in
            guard value.binding?.loadID == loadID else { return nil }
            return value.binding
        }
    }

    func prepare(loadID: UUID?, operationID: UUID) -> AppGenerationStop? {
        guard let loadID else { return nil }
        return state.withLock { value in
            guard case .available = value.lifecycle,
                  value.binding?.loadID == loadID else { return nil }
            let key = OperationKey(loadID: loadID, operationID: operationID)
            if let existing = value.stops[key] { return existing }
            guard value.stops.count < maximumOperations else { return nil }
            let stop = AppGenerationStop()
            value.stops[key] = stop
            return stop
        }
    }

    @discardableResult
    func requestStop(loadID: UUID?, operationID: UUID) -> Bool {
        guard let loadID else { return false }
        let stop = state.withLock { value -> AppGenerationStop? in
            guard value.binding?.loadID == loadID else { return nil }
            return value.stops[OperationKey(loadID: loadID, operationID: operationID)]
        }
        stop?.requestStop()
        return stop != nil
    }

    func stop(loadID: UUID?, operationID: UUID) -> AppGenerationStop? {
        guard let loadID else { return nil }
        return state.withLock { value in
            guard value.binding?.loadID == loadID else { return nil }
            return value.stops[OperationKey(loadID: loadID, operationID: operationID)]
        }
    }

    func retire(loadID: UUID?, operationID: UUID) {
        guard let loadID else { return }
        let stop = state.withLock { value in
            value.stops.removeValue(forKey: OperationKey(
                loadID: loadID, operationID: operationID))
        }
        stop?.finish()
    }

    @discardableResult
    func retire(loadID: UUID?) -> Binding? {
        guard let loadID else { return nil }
        let result = state.withLock { value -> (Binding?, [AppGenerationStop]) in
            guard case .available = value.lifecycle,
                  value.binding?.loadID == loadID else { return (nil, []) }
            let binding = value.binding
            let stops = Array(value.stops.values)
            value.binding = nil
            value.stops.removeAll(keepingCapacity: true)
            return (binding, stops)
        }
        result.1.forEach { $0.finish() }
        return result.0
    }

    @discardableResult
    func retireCurrent() -> Binding? {
        let result = state.withLock { value -> (Binding?, [AppGenerationStop]) in
            guard case .available = value.lifecycle else { return (nil, []) }
            let binding = value.binding
            let stops = Array(value.stops.values)
            value.binding = nil
            value.stops.removeAll(keepingCapacity: true)
            return (binding, stops)
        }
        result.1.forEach { $0.finish() }
        return result.0
    }
}
