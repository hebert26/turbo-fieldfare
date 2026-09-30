import Foundation
import Metal

enum QwenLinearAttentionStateError: Error, Equatable, Sendable {
    case invalidLayerMaskCount(Int)
    case invalidLayerMaskValue(layer: Int, value: UInt8)
    case invalidLinearLayerCount(expected: Int, actual: Int)
    case invalidGeometry(field: String, value: Int)
    case arithmeticOverflow(operation: String)
    case allocationFailed(layer: Int, kind: String, bytes: Int)
    case notLinearAttentionLayer(Int)
    case updateAlreadyPending(layer: Int)
    case unknownUpdate
    case updateAlreadySubmitted
    case activeGPUUse
    case commandBufferAlreadySubmitted
    case invalidSnapshotGeometry
    case invalidSnapshotLayers
    case invalidPosition
    case deferredGPUUseActive
}

struct QwenLinearAttentionGeometry: Equatable, Sendable {
    let convolutionWidth: Int
    let convolutionChannelCount: Int
    let valueHeadCount: Int
    let keyHeadDimension: Int
    let valueHeadDimension: Int

    var convolutionElementCount: Int { convolutionChannelCount * convolutionWidth }
    var recurrentElementCount: Int { valueHeadCount * keyHeadDimension * valueHeadDimension }

    init(
        convolutionWidth: Int,
        convolutionChannelCount: Int,
        valueHeadCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int
    ) throws {
        guard convolutionWidth == 4 else {
            throw QwenLinearAttentionStateError.invalidGeometry(
                field: "convolutionWidth", value: convolutionWidth)
        }
        for (field, value) in [
            ("convolutionChannelCount", convolutionChannelCount),
            ("valueHeadCount", valueHeadCount),
            ("keyHeadDimension", keyHeadDimension),
            ("valueHeadDimension", valueHeadDimension),
        ] where value <= 0 || UInt32(exactly: value) == nil {
            throw QwenLinearAttentionStateError.invalidGeometry(field: field, value: value)
        }
        _ = try Self.checkedMultiply(
            convolutionChannelCount, convolutionWidth, operation: "convolution elements")
        _ = try Self.checkedMultiply(
            try Self.checkedMultiply(valueHeadCount, keyHeadDimension, operation: "recurrent heads"),
            valueHeadDimension,
            operation: "recurrent elements")
        self.convolutionWidth = convolutionWidth
        self.convolutionChannelCount = convolutionChannelCount
        self.valueHeadCount = valueHeadCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
    }

    static func official() throws -> Self {
        try Self(
            convolutionWidth: 4,
            convolutionChannelCount: 8_192,
            valueHeadCount: 32,
            keyHeadDimension: 128,
            valueHeadDimension: 128)
    }

    fileprivate static func checkedMultiply(
        _ lhs: Int, _ rhs: Int, operation: String
    ) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw QwenLinearAttentionStateError.arithmeticOverflow(operation: operation)
        }
        return result
    }
}

struct QwenLinearAttentionLayerState: Equatable, Sendable {
    let convolutionHistory: [Float]
    let recurrentMatrix: [Float]
}

struct QwenLinearAttentionSnapshot: Equatable, Sendable {
    let geometry: QwenLinearAttentionGeometry
    let layers: [Int: QwenLinearAttentionLayerState]
    /// Positions travel with their histories and recurrent matrices on clone/restore.
    let positions: [Int: Int]

    /// Legacy callers supplying only geometry and layers retain zero positions.
    init(geometry: QwenLinearAttentionGeometry,
         layers: [Int: QwenLinearAttentionLayerState],
         positions: [Int: Int]? = nil) {
        self.geometry = geometry
        self.layers = layers
        self.positions = positions ?? Dictionary(
            uniqueKeysWithValues: layers.keys.map { ($0, 0) })
    }
}

struct QwenBF16LinearStepSnapshot: Equatable, Sendable {
    let position: Int
    let state: QwenLinearAttentionLayerState
}

/// Staging buffers for one transactional layer update. These buffers are never
/// committed until the command buffer completes successfully.
struct QwenLinearAttentionUpdate: @unchecked Sendable {
    let layer: Int
    let convolutionHistory: MTLBuffer
    let recurrentMatrix: MTLBuffer
    fileprivate let identifier: UInt64
}

/// Owns committed linear-attention state and serializes staged GPU replacement.
/// Cancellation never mutates committed buffers. Submitted staging remains
/// retained until Metal reports actual completion.
final class QwenLinearAttentionState: @unchecked Sendable {
    private struct LayerStorage {
        var convolutionHistory: MTLBuffer
        var recurrentMatrix: MTLBuffer
        var position: Int
        var pending: PendingUpdate?
    }

    private struct PendingUpdate {
        let identifier: UInt64
        let convolutionHistory: MTLBuffer
        let recurrentMatrix: MTLBuffer
        var submitted: Bool
        var deferred: Bool
        var activeCommands: Int
        var gpuSucceeded: Bool
        var discardRequested: Bool
        var stageWaiters: [CheckedContinuation<Void, Never>]
    }

    let linearLayerIndices: [Int]
    let geometry: QwenLinearAttentionGeometry

    private let device: MTLDevice
    private let lock = NSLock()
    private var layers: [Int: LayerStorage]
    private var nextIdentifier: UInt64 = 1
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []

    init(
        device: MTLDevice,
        linearAttentionLayerMask: [UInt8],
        geometry: QwenLinearAttentionGeometry,
        expectedLinearLayerCount: Int? = nil
    ) throws {
        guard !linearAttentionLayerMask.isEmpty else {
            throw QwenLinearAttentionStateError.invalidLayerMaskCount(0)
        }
        for (layer, value) in linearAttentionLayerMask.enumerated() where value != 0 && value != 1 {
            throw QwenLinearAttentionStateError.invalidLayerMaskValue(layer: layer, value: value)
        }
        let selected = linearAttentionLayerMask.indices.filter { linearAttentionLayerMask[$0] == 1 }
        if let expectedLinearLayerCount, selected.count != expectedLinearLayerCount {
            throw QwenLinearAttentionStateError.invalidLinearLayerCount(
                expected: expectedLinearLayerCount, actual: selected.count)
        }
        guard !selected.isEmpty else {
            throw QwenLinearAttentionStateError.invalidLinearLayerCount(expected: 1, actual: 0)
        }

        let historyBytes = try Self.byteCount(
            geometry.convolutionElementCount, operation: "history bytes")
        let recurrentBytes = try Self.byteCount(
            geometry.recurrentElementCount, operation: "recurrent bytes")
        guard historyBytes <= device.maxBufferLength,
              recurrentBytes <= device.maxBufferLength else {
            throw QwenLinearAttentionStateError.arithmeticOverflow(
                operation: "state exceeds Metal maxBufferLength")
        }
        var storage: [Int: LayerStorage] = [:]
        storage.reserveCapacity(selected.count)
        for layer in selected {
            let history = try Self.makeZeroedBuffer(
                device: device, bytes: historyBytes, layer: layer, kind: "history")
            let recurrent = try Self.makeZeroedBuffer(
                device: device, bytes: recurrentBytes, layer: layer, kind: "recurrent")
            history.label = "qwen.linear.history.layer\(layer).committed"
            recurrent.label = "qwen.linear.recurrent.layer\(layer).committed"
            storage[layer] = LayerStorage(
                convolutionHistory: history, recurrentMatrix: recurrent,
                position: 0, pending: nil)
        }
        self.device = device
        self.geometry = geometry
        linearLayerIndices = selected
        layers = storage
    }

    static func official(
        device: MTLDevice,
        linearAttentionLayerMask: [UInt8]
    ) throws -> QwenLinearAttentionState {
        guard linearAttentionLayerMask.count == 40 else {
            throw QwenLinearAttentionStateError.invalidLayerMaskCount(
                linearAttentionLayerMask.count)
        }
        return try QwenLinearAttentionState(
            device: device,
            linearAttentionLayerMask: linearAttentionLayerMask,
            geometry: .official(),
            expectedLinearLayerCount: 30)
    }

    func layerState(_ layer: Int) throws -> QwenLinearAttentionLayerState {
        try withLock {
            let storage = try requireLayer(layer)
            return QwenLinearAttentionLayerState(
                convolutionHistory: Self.readFloats(
                    storage.convolutionHistory, count: geometry.convolutionElementCount),
                recurrentMatrix: Self.readFloats(
                    storage.recurrentMatrix, count: geometry.recurrentElementCount))
        }
    }

    func layerSnapshot(_ layer: Int) throws -> QwenBF16LinearStepSnapshot {
        try withLock {
            let storage = try requireLayer(layer)
            guard storage.pending == nil else { throw QwenLinearAttentionStateError.activeGPUUse }
            return QwenBF16LinearStepSnapshot(
                position: storage.position,
                state: QwenLinearAttentionLayerState(
                    convolutionHistory: Self.readFloats(
                        storage.convolutionHistory, count: geometry.convolutionElementCount),
                    recurrentMatrix: Self.readFloats(
                        storage.recurrentMatrix, count: geometry.recurrentElementCount)))
        }
    }

    func committedPosition(layer: Int) throws -> Int {
        try withLock { try requireLayer(layer).position }
    }

    func isBound(to device: MTLDevice) -> Bool { self.device === device }

    func clone() throws -> QwenLinearAttentionSnapshot {
        try withLock {
            try requireNoActiveUse()
            var result: [Int: QwenLinearAttentionLayerState] = [:]
            var positions: [Int: Int] = [:]
            result.reserveCapacity(linearLayerIndices.count)
            positions.reserveCapacity(linearLayerIndices.count)
            for layer in linearLayerIndices {
                let storage = try requireLayer(layer)
                result[layer] = QwenLinearAttentionLayerState(
                    convolutionHistory: Self.readFloats(
                        storage.convolutionHistory, count: geometry.convolutionElementCount),
                    recurrentMatrix: Self.readFloats(
                        storage.recurrentMatrix, count: geometry.recurrentElementCount))
                positions[layer] = storage.position
            }
            return QwenLinearAttentionSnapshot(
                geometry: geometry, layers: result, positions: positions)
        }
    }

    func restore(_ snapshot: QwenLinearAttentionSnapshot) throws {
        try withLock {
            try requireNoActiveUse()
            guard snapshot.geometry == geometry else {
                throw QwenLinearAttentionStateError.invalidSnapshotGeometry
            }
            guard Set(snapshot.layers.keys) == Set(linearLayerIndices),
                  Set(snapshot.positions.keys) == Set(linearLayerIndices),
                  snapshot.positions.values.allSatisfy({ $0 >= 0 }) else {
                throw QwenLinearAttentionStateError.invalidSnapshotLayers
            }
            for layer in linearLayerIndices {
                guard let state = snapshot.layers[layer],
                      state.convolutionHistory.count == geometry.convolutionElementCount,
                      state.recurrentMatrix.count == geometry.recurrentElementCount else {
                    throw QwenLinearAttentionStateError.invalidSnapshotGeometry
                }
            }
            for layer in linearLayerIndices {
                guard let state = snapshot.layers[layer], var storage = layers[layer] else { continue }
                Self.write(state.convolutionHistory, to: storage.convolutionHistory)
                Self.write(state.recurrentMatrix, to: storage.recurrentMatrix)
                storage.position = snapshot.positions[layer] ?? 0
                storage.pending = nil
                layers[layer] = storage
            }
        }
    }

    func reset() throws {
        try withLock {
            try requireNoActiveUse()
            for layer in linearLayerIndices {
                guard let storage = layers[layer] else { continue }
                memset(storage.convolutionHistory.contents(), 0, storage.convolutionHistory.length)
                memset(storage.recurrentMatrix.contents(), 0, storage.recurrentMatrix.length)
                layers[layer]?.position = 0
            }
        }
    }

    func reserveUpdate(layer: Int) throws -> QwenLinearAttentionUpdate {
        try withLock {
            var storage = try requireLayer(layer)
            guard storage.pending == nil else {
                throw QwenLinearAttentionStateError.updateAlreadyPending(layer: layer)
            }
            let history = try Self.makeBufferCopy(
                device: device, source: storage.convolutionHistory, layer: layer, kind: "history staging")
            let recurrent = try Self.makeBufferCopy(
                device: device, source: storage.recurrentMatrix, layer: layer, kind: "recurrent staging")
            history.label = "qwen.linear.history.layer\(layer).staging"
            recurrent.label = "qwen.linear.recurrent.layer\(layer).staging"
            let identifier = allocateIdentifier()
            storage.pending = PendingUpdate(
                identifier: identifier,
                convolutionHistory: history,
                recurrentMatrix: recurrent,
                submitted: false,
                deferred: false,
                activeCommands: 0,
                gpuSucceeded: true,
                discardRequested: false,
                stageWaiters: [])
            layers[layer] = storage
            return QwenLinearAttentionUpdate(
                layer: layer,
                convolutionHistory: history,
                recurrentMatrix: recurrent,
                identifier: identifier)
        }
    }

    /// Releases a reservation whose command buffer was never submitted.
    func abort(_ update: QwenLinearAttentionUpdate) throws {
        let waiters = try withLock { () -> [CheckedContinuation<Void, Never>] in
            var storage = try requireUpdate(update)
            guard let pending = storage.pending, !pending.submitted else {
                throw QwenLinearAttentionStateError.updateAlreadySubmitted
            }
            storage.pending = nil
            layers[update.layer] = storage
            return takeIdleWaitersIfIdle()
        }
        Self.resume(waiters)
    }

    /// Cancels before submit by releasing immediately, or after submit by
    /// marking the staged result for discard at actual GPU completion.
    func cancel(_ update: QwenLinearAttentionUpdate) throws {
        let waiters = try withLock { () -> [CheckedContinuation<Void, Never>] in
            var storage = try requireUpdate(update)
            guard var pending = storage.pending else {
                throw QwenLinearAttentionStateError.unknownUpdate
            }
            if pending.submitted && pending.activeCommands > 0 {
                pending.discardRequested = true
                storage.pending = pending
            } else {
                storage.pending = nil
            }
            layers[update.layer] = storage
            return takeIdleWaitersIfIdle()
        }
        Self.resume(waiters)
    }

    /// Suspends without blocking a thread until this owner has processed every
    /// pending update completion. Task cancellation cannot release GPU-owned
    /// staging or make state reusable before that owner transition.
    func waitUntilIdle() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if layers.values.allSatisfy({ $0.pending == nil }) {
                lock.unlock()
                continuation.resume()
            } else {
                idleWaiters.append(continuation)
                lock.unlock()
            }
        }
    }

    /// Installs completion ownership and commits as one lock-linearized action.
    /// The caller must encode all work before invoking this method.
    func submit(_ update: QwenLinearAttentionUpdate, on commandBuffer: MTLCommandBuffer) throws {
        try withLock {
            guard commandBuffer.status == .notEnqueued else {
                throw QwenLinearAttentionStateError.commandBufferAlreadySubmitted
            }
            var storage = try requireUpdate(update)
            guard var pending = storage.pending, !pending.submitted else {
                throw QwenLinearAttentionStateError.updateAlreadySubmitted
            }
            pending.submitted = true
            pending.activeCommands = 1
            storage.pending = pending
            layers[update.layer] = storage
            commandBuffer.addCompletedHandler { [self] completed in
                complete(
                    layer: update.layer,
                    identifier: update.identifier,
                    succeeded: completed.status == .completed && completed.error == nil)
            }
            commandBuffer.commit()
        }
    }

    /// Register a submitted BF16 stage without publishing staged state. The
    /// completion callback retains this owner and scratch until Metal settles;
    /// only `commitDeferred` can publish history, recurrence and position.
    func submitDeferred(_ update: QwenLinearAttentionUpdate,
                        on commandBuffer: MTLCommandBuffer) throws {
        try withLock {
            guard commandBuffer.status == .notEnqueued,
                  commandBuffer.device === device else {
                throw QwenLinearAttentionStateError.commandBufferAlreadySubmitted
            }
            var storage = try requireUpdate(update)
            guard var pending = storage.pending,
                  !pending.discardRequested,
                  !pending.submitted || pending.deferred && pending.activeCommands == 0 else {
                throw QwenLinearAttentionStateError.updateAlreadySubmitted
            }
            pending.submitted = true
            pending.deferred = true
            pending.activeCommands = 1
            storage.pending = pending
            layers[update.layer] = storage
            commandBuffer.addCompletedHandler { [self] completed in
                completeDeferred(layer: update.layer, identifier: update.identifier,
                                 succeeded: completed.status == .completed && completed.error == nil)
            }
            commandBuffer.commit()
        }
    }

    /// Wait for our completion callback as well as Metal's completion, so a
    /// subsequent stage cannot observe callback scheduling as unfinished use.
    func waitForDeferredStage(_ update: QwenLinearAttentionUpdate) async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if var storage = layers[update.layer], var pending = storage.pending,
               pending.identifier == update.identifier, pending.activeCommands > 0 {
                pending.stageWaiters.append(continuation)
                storage.pending = pending
                layers[update.layer] = storage
                lock.unlock()
            } else {
                lock.unlock()
                continuation.resume()
            }
        }
    }

    /// Non-suspending publication after all stages settle, output is checked,
    /// and the actor's final cancellation/hook check has passed.
    func commitDeferred(_ update: QwenLinearAttentionUpdate,
                        tokenCount: Int) throws {
        let waiters = try withLock { () -> [CheckedContinuation<Void, Never>] in
            var storage = try requireUpdate(update)
            guard let pending = storage.pending, pending.deferred,
                  pending.submitted, pending.activeCommands == 0,
                  !pending.discardRequested, pending.gpuSucceeded else {
                throw QwenLinearAttentionStateError.deferredGPUUseActive
            }
            let (position, overflow) = storage.position.addingReportingOverflow(tokenCount)
            guard tokenCount > 0, UInt32(exactly: tokenCount) != nil, !overflow,
                  position >= 0,
                  Self.allFinite(pending.convolutionHistory),
                  Self.allFinite(pending.recurrentMatrix) else {
                throw QwenLinearAttentionStateError.invalidPosition
            }
            storage.convolutionHistory = pending.convolutionHistory
            storage.recurrentMatrix = pending.recurrentMatrix
            storage.position = position
            storage.pending = nil
            layers[update.layer] = storage
            return takeIdleWaitersIfIdle()
        }
        Self.resume(waiters)
    }

    private static func allFinite(_ buffer: MTLBuffer) -> Bool {
        let values = UnsafeBufferPointer(
            start: buffer.contents().assumingMemoryBound(to: Float.self),
            count: buffer.length / MemoryLayout<Float>.stride)
        return values.allSatisfy(\.isFinite)
    }

    private func completeDeferred(layer: Int, identifier: UInt64, succeeded: Bool) {
        lock.lock()
        guard var storage = layers[layer], var pending = storage.pending,
              pending.identifier == identifier, pending.deferred,
              pending.activeCommands == 1 else { lock.unlock(); return }
        pending.activeCommands = 0
        pending.gpuSucceeded = pending.gpuSucceeded && succeeded
        let stageWaiters = pending.stageWaiters
        pending.stageWaiters = []
        storage.pending = pending.discardRequested ? nil : pending
        layers[layer] = storage
        let waiters = takeIdleWaitersIfIdle()
        lock.unlock()
        Self.resume(stageWaiters)
        Self.resume(waiters)
    }

    /// Guarantees abort for every thrown pre-submit encoding path.
    func encodeAndSubmit(
        layer: Int,
        commandBuffer: MTLCommandBuffer,
        encode: (QwenLinearAttentionUpdate) throws -> Void
    ) throws {
        let update = try reserveUpdate(layer: layer)
        do {
            try encode(update)
            try submit(update, on: commandBuffer)
        } catch {
            try? abort(update)
            throw error
        }
    }

    private func complete(layer: Int, identifier: UInt64, succeeded: Bool) {
        lock.lock()
        guard var storage = layers[layer],
              let pending = storage.pending,
              pending.identifier == identifier else {
            lock.unlock()
            return
        }
        if succeeded && !pending.discardRequested {
            storage.convolutionHistory = pending.convolutionHistory
            storage.recurrentMatrix = pending.recurrentMatrix
        }
        storage.pending = nil
        layers[layer] = storage
        let waiters = takeIdleWaitersIfIdle()
        lock.unlock()
        Self.resume(waiters)
    }

    /// Must be called only while `lock` is held. Removing under that same lock
    /// makes the idle transition and waiter ownership one atomic operation.
    private func takeIdleWaitersIfIdle() -> [CheckedContinuation<Void, Never>] {
        guard layers.values.allSatisfy({ $0.pending == nil }) else { return [] }
        let waiters = idleWaiters
        idleWaiters.removeAll(keepingCapacity: true)
        return waiters
    }

    private static func resume(_ waiters: [CheckedContinuation<Void, Never>]) {
        for waiter in waiters { waiter.resume() }
    }

    private func requireUpdate(_ update: QwenLinearAttentionUpdate) throws -> LayerStorage {
        let storage = try requireLayer(update.layer)
        guard let pending = storage.pending,
              pending.identifier == update.identifier,
              pending.convolutionHistory === update.convolutionHistory,
              pending.recurrentMatrix === update.recurrentMatrix else {
            throw QwenLinearAttentionStateError.unknownUpdate
        }
        return storage
    }

    private func requireLayer(_ layer: Int) throws -> LayerStorage {
        guard let storage = layers[layer] else {
            throw QwenLinearAttentionStateError.notLinearAttentionLayer(layer)
        }
        return storage
    }

    private func requireNoActiveUse() throws {
        for layer in linearLayerIndices where layers[layer]?.pending != nil {
            throw QwenLinearAttentionStateError.activeGPUUse
        }
    }

    private func allocateIdentifier() -> UInt64 {
        let result = nextIdentifier
        nextIdentifier &+= 1
        if nextIdentifier == 0 { nextIdentifier = 1 }
        return result
    }

    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private static func byteCount(_ elements: Int, operation: String) throws -> Int {
        try QwenLinearAttentionGeometry.checkedMultiply(
            elements, MemoryLayout<Float>.stride, operation: operation)
    }

    private static func makeZeroedBuffer(
        device: MTLDevice, bytes: Int, layer: Int, kind: String
    ) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw QwenLinearAttentionStateError.allocationFailed(
                layer: layer, kind: kind, bytes: bytes)
        }
        memset(buffer.contents(), 0, bytes)
        return buffer
    }

    private static func makeBufferCopy(
        device: MTLDevice, source: MTLBuffer, layer: Int, kind: String
    ) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: source.length, options: .storageModeShared) else {
            throw QwenLinearAttentionStateError.allocationFailed(
                layer: layer, kind: kind, bytes: source.length)
        }
        memcpy(buffer.contents(), source.contents(), source.length)
        return buffer
    }

    private static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static func write(_ values: [Float], to buffer: MTLBuffer) {
        values.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                memcpy(buffer.contents(), baseAddress, bytes.count)
            }
        }
    }
}
