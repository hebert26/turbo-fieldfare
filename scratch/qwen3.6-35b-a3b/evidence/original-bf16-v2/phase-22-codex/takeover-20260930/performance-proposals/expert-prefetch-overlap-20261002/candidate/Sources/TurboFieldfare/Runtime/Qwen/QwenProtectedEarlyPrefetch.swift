import Foundation
import Metal
import Synchronization

enum QwenQuarantineError: Error {
    case identity, busy, incomplete
}
struct QwenQuarantineReadFailure: Error, CustomStringConvertible {
    let errors: [any Error]
    var description: String { "quarantine read failures: " + errors.map { String(describing: $0) }.joined(separator: " | ") }
}
struct QwenProductionAndQuarantineFailure: Error, CustomStringConvertible {
    let primary: any Error
    let speculative: any Error
    var description: String { "production: \(primary); quarantine: \(speculative)" }
}
final class QwenQuarantineEpoch: Sendable {
    let position: Int
    let layer: Int
    let ordinal: UInt64
    private struct State { var valid = true; var reusable = false; var canceled = false }
    private let state = Mutex(State())
    let importFinished = DispatchGroup()
    init(position: Int, layer: Int, ordinal: UInt64) {
        self.position = position; self.layer = layer; self.ordinal = ordinal
        importFinished.enter()
    }
    func cancel() { state.withLock { $0.canceled = true } }
    func isCanceled() -> Bool { state.withLock { $0.canceled } }
    func requireValid(position: Int, layer: Int) throws {
        guard self.position == position, self.layer == layer,
              state.withLock({ $0.valid && !$0.canceled && !$0.reusable }) else {
            throw QwenQuarantineError.identity
        }
    }
    func finish() {
        let first = state.withLock { value -> Bool in
            if value.reusable { return false }
            value.reusable = true
            return true
        }
        if first { importFinished.leave() }
    }
    func canReuse() -> Bool { state.withLock { $0.reusable } }
    func invalidate() { state.withLock { $0.valid = false } }
}

// Buffer references are fixed. Streams have disjoint writers until the join.
// A sealed batch is read-only until its import lease finishes and epoch expires.
struct QwenQuarantineBuffers: @unchecked Sendable {
    let gateUp: MTLBuffer
    let down: MTLBuffer
}
final class QwenSettledQuarantineBatch: @unchecked Sendable {
    let owner: QwenBF16EarlyReadOwner
    let epoch: QwenQuarantineEpoch
    private let experts: [Int]
    private let pairs: [QwenQuarantineBuffers]
    private let quota: QwenOfficialSourceModel.ExpertCacheReservation
    private let capture: QwenEarlyPrefetchCapture
    private struct Usage { var copied = Set<Int>(); var finished = false }
    private let usage = Mutex(Usage())
    fileprivate init(owner: QwenBF16EarlyReadOwner, epoch: QwenQuarantineEpoch,
                     experts: [Int], pairs: [QwenQuarantineBuffers],
                     quota: QwenOfficialSourceModel.ExpertCacheReservation,
                     capture: QwenEarlyPrefetchCapture) {
        self.owner = owner; self.epoch = epoch; self.experts = experts; self.pairs = pairs
        self.quota = quota; self.capture = capture
    }
    func contains(_ expert: Int, position: Int, layer: Int) throws -> Bool {
        try epoch.requireValid(position: position, layer: layer)
        return experts.contains(expert)
    }
    func copy(expert: Int, stream: Int, into destination: MTLBuffer, position: Int, layer: Int) throws {
        try epoch.requireValid(position: position, layer: layer)
        guard let index = experts.firstIndex(of: expert), stream == 0 || stream == 1 else {
            throw QwenQuarantineError.identity
        }
        let source = stream == 0 ? pairs[index].gateUp : pairs[index].down
        let bytes = stream == 0 ? owner.gateBytes : owner.downBytes
        guard source.length == bytes, destination.length == bytes,
              source.storageMode == .shared, destination.storageMode == .shared,
              source.device === destination.device, source !== destination else {
            throw QwenQuarantineError.identity
        }
        capture.measure(.copy, bytes: UInt64(bytes)) {
            _ = memcpy(destination.contents(), source.contents(), bytes)
        }
    }
    func published(expert: Int) { usage.withLock { $0.copied.insert(expert) } }
    func finish() {
        let imported = usage.withLock { value -> Int? in
            if value.finished { return nil }
            value.finished = true
            return value.copied.count
        }
        if let imported { capture.completedBatch(predicted: experts.count, imported: imported); epoch.finish() }
    }
    deinit { finish() }
}

private final class QwenQuarantineReadBatch: @unchecked Sendable {
    let group = DispatchGroup()
    let settlement = DispatchGroup()
    let owner: QwenBF16EarlyReadOwner
    let epoch: QwenQuarantineEpoch
    private let experts: [Int]
    private let pairs: [QwenQuarantineBuffers]
    private let quota: QwenOfficialSourceModel.ExpertCacheReservation
    private let capture: QwenEarlyPrefetchCapture
    private let results: Mutex<[Result<Void, any Error>?]>
    init(owner: QwenBF16EarlyReadOwner, epoch: QwenQuarantineEpoch, experts: [Int],
         pairs: [QwenQuarantineBuffers], quota: QwenOfficialSourceModel.ExpertCacheReservation,
         capture: QwenEarlyPrefetchCapture) {
        self.owner = owner; self.epoch = epoch; self.experts = experts; self.pairs = pairs
        self.quota = quota; self.capture = capture
        results = Mutex(Array(repeating: nil, count: experts.count * 2))
        // Account for launch before the pool can publish this batch.
        group.enter()
    }
    func launch() {
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            defer { group.leave() }
            // Pairs are serial. The two original streams settle together.
            for index in experts.indices {
                if epoch.isCanceled() { break }
                DispatchQueue.concurrentPerform(iterations: 2) { stream in
                    let outcome = Result {
                        let pair = pairs[index]
                        try owner.read(expert: experts[index], stream: stream,
                            into: stream == 0 ? pair.gateUp : pair.down,
                            capture: capture, canceled: epoch.isCanceled)
                    }
                    results.withLock { $0[index * 2 + stream] = outcome }
                }
            }
        }
    }
    func readFailure() -> (any Error)? {
        let errors = results.withLock { values in
            values.compactMap { value -> (any Error)? in
                guard let value else { return nil }
                if case let .failure(error) = value { return error }
                return nil
            }
        }
        guard !errors.isEmpty else { return nil }
        if errors.allSatisfy({ $0 is CancellationError }) { return CancellationError() }
        return QwenQuarantineReadFailure(errors: errors)
    }
    // Caller joins first. Every launched stream settles even after a sibling error.
    func settled() throws -> QwenSettledQuarantineBatch {
        let values = results.withLock { $0 }
        if let error = readFailure() { epoch.finish(); throw error }
        if epoch.isCanceled() { epoch.finish(); throw CancellationError() }
        guard values.allSatisfy({ $0 != nil }) else { epoch.finish(); throw QwenQuarantineError.incomplete }
        return QwenSettledQuarantineBatch(owner: owner, epoch: epoch, experts: experts,
            pairs: pairs, quota: quota, capture: capture)
    }
}

// One pool per isolated request. No buffer enters a GPU or actual cache plan.
final class QwenProtectedEarlyPrefetch: @unchecked Sendable {
    let enabled: Bool
    let allocatedBytes: UInt64
    private let pairs: [QwenQuarantineBuffers]
    private let quota: QwenOfficialSourceModel.ExpertCacheReservation
    private let capture: QwenEarlyPrefetchCapture
    private struct State {
        var batch: QwenQuarantineReadBatch?
        var draining = false
        var epoch: QwenQuarantineEpoch?
        var ordinal: UInt64 = 0
    }
    private let state = Mutex(State())
    init(model: QwenOfficialSourceModel, enabled: Bool, capture: QwenEarlyPrefetchCapture) throws {
        let quota = try model.reserveEarlyReadQuarantine()
        let width = model.architecture.hiddenSize
        let intermediate = model.architecture.routedIntermediateSize
        guard width == 2048, intermediate == 512, model.architecture.layers == 40 else {
            throw QwenQuarantineError.identity
        }
        let gateBytes = width * intermediate * 4
        let downBytes = width * intermediate * 2
        guard quota.bytes == UInt64((gateBytes + downBytes) * 8) else { throw QwenQuarantineError.identity }
        var pairs: [QwenQuarantineBuffers] = []
        for _ in 0..<8 {
            guard let gate = model.context.device.makeBuffer(length: gateBytes, options: .storageModeShared),
                  let down = model.context.device.makeBuffer(length: downBytes, options: .storageModeShared) else {
                throw QwenBF16ExpertCacheError.allocationFailed
            }
            memset(gate.contents(), 0, gateBytes)
            memset(down.contents(), 0, downBytes)
            pairs.append(QwenQuarantineBuffers(gateUp: gate, down: down))
        }
        self.quota = quota; self.pairs = pairs; self.enabled = enabled
        self.capture = capture; allocatedBytes = quota.bytes
    }
    func start(owner: QwenBF16EarlyReadOwner, experts: [Int], position: Int, layer: Int) throws {
        guard enabled, owner.layer == layer, (0..<40).contains(layer), position >= 0,
              experts.count <= 8, Set(experts).count == experts.count,
              experts.allSatisfy({ (0..<owner.expertCount).contains($0) }) else { throw QwenQuarantineError.identity }
        let batch = try state.withLock { value -> QwenQuarantineReadBatch in
            guard value.batch == nil, value.epoch?.canReuse() ?? true, value.ordinal < UInt64.max else {
                throw QwenQuarantineError.busy
            }
            value.epoch?.invalidate()
            value.ordinal += 1
            let epoch = QwenQuarantineEpoch(position: position, layer: layer, ordinal: value.ordinal)
            let batch = QwenQuarantineReadBatch(owner: owner, epoch: epoch, experts: experts,
                pairs: Array(pairs.prefix(experts.count)), quota: quota, capture: capture)
            value.epoch = epoch; value.batch = batch
            return batch
        }
        batch.launch()
    }
    private func join(_ group: DispatchGroup) async {
        await withCheckedContinuation { continuation in
            group.notify(queue: .global(qos: .userInitiated)) { continuation.resume() }
        }
    }
    func drain(position: Int, layer: Int) async throws -> QwenSettledQuarantineBatch? {
        let batch = try state.withLock { value -> QwenQuarantineReadBatch? in
            guard !value.draining else { throw QwenQuarantineError.busy }
            guard let batch = value.batch else { return nil }
            batch.settlement.enter()
            value.draining = true
            return batch
        }
        guard let batch else { return nil }
        let start = QwenProductionTimingMeasurement.uptime()
        defer {
            state.withLock { value in
                if value.batch === batch { value.batch = nil; value.draining = false }
            }
            batch.settlement.leave()
            capture.duration(.drain, start: start)
        }
        return try await withTaskCancellationHandler {
            await join(batch.group)
            let settled = try batch.settled()
            do {
                try Task.checkCancellation()
                try settled.epoch.requireValid(position: position, layer: layer)
                return settled
            } catch { settled.finish(); throw error }
        } onCancel: { batch.epoch.cancel() }
    }
    func cancelAndDrain() async throws {
        let snapshot = state.withLock { ($0.batch, $0.epoch) }
        guard let epoch = snapshot.1 else { return }
        epoch.cancel()
        var failure: (any Error)?
        if let pending = snapshot.0 {
            await join(pending.group)
            failure = pending.readFailure()
            let drainOwns = state.withLock { value -> Bool in
                guard value.batch === pending else { return false }
                if value.draining { return true }
                value.batch = nil
                epoch.finish()
                return false
            }
            if drainOwns { await join(pending.settlement) }
        }
        // A returned import lease still owns these buffers until its jobs settle.
        await join(epoch.importFinished)
        epoch.invalidate()
        if let failure, !(failure is CancellationError) { throw failure }
    }
    func idle() -> Bool { state.withLock { $0.batch == nil && ($0.epoch?.canReuse() ?? true) } }
    deinit {
        let pending = state.withLock { $0.batch }
        pending?.epoch.cancel()
        pending?.group.wait()
        pending?.epoch.invalidate()
    }
}
