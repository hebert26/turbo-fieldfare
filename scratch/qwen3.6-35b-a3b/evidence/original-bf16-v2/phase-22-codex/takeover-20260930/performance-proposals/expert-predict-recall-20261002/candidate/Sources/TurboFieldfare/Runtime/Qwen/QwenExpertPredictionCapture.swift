import Foundation
import Synchronization

// Fixed diagnostic storage. It cannot supply routes, bytes or callbacks.
final class QwenExpertPredictionCapture: Sendable {
    struct Key: Hashable, Sendable {
        let position: Int
        let layer: Int
    }
    struct Map: Sendable {
        let residentSlots: [Int]
        let experts: [Int]
        let missIndices: [Int]
        let assignedSlots: [Int]
    }
    struct Row: Sendable {
        var duplicate = false
        var early: [Float]?
        var late: [Float]?
        var lateTime: UInt64 = 0
        var mapStart: UInt64 = 0
        var mapEnd: UInt64 = 0
        var map: Map?
    }
    struct State: Sendable {
        var enabled = true
        var rows: [Key: Row] = [:]
        var failure: String?
        var callbackNanoseconds: UInt64 = 0
    }
    let firstPosition: Int
    let width: Int
    let layers: Int
    private let storage = Mutex(State())
    init(firstPosition: Int, width: Int, layers: Int) {
        precondition(firstPosition >= 0 && width > 0 && width <= 4096 && layers == 40)
        self.firstPosition = firstPosition
        self.width = width
        self.layers = layers
    }
    func includes(_ position: Int) -> Bool {
        position >= firstPosition && position < firstPosition + 32
    }
    private func update(position: Int, layer: Int, _ body: (inout Row) -> Void) {
        guard includes(position), (0..<layers).contains(layer) else { return }
        let start = QwenProductionTimingMeasurement.uptime()
        storage.withLock { state in
            guard state.enabled else { return }
            body(&state.rows[Key(position: position, layer: layer), default: Row()])
            state.callbackNanoseconds += QwenProductionTimingMeasurement.uptime() - start
        }
    }
    func residual(position: Int, layer: Int, early: Bool, hidden: [Float]) {
        guard includes(position), layer < layers - 1 else { return }
        guard hidden.count == width else {
            storage.withLock { $0.failure = "residual width mismatch" }
            return
        }
        update(position: position, layer: layer) { row in
            // Swift arrays retain value storage. No mutable scratch pointer is kept.
            if early {
                if row.early != nil { row.duplicate = true }
                row.early = hidden
            }
            else {
                if row.late != nil { row.duplicate = true }
                row.late = hidden
                row.lateTime = QwenProductionTimingMeasurement.uptime()
            }
        }
    }
    func mapStart(position: Int, layer: Int) {
        guard includes(position) else { return }
        let observed = QwenProductionTimingMeasurement.uptime()
        update(position: position, layer: layer) {
            if $0.mapStart != 0 { $0.duplicate = true }
            $0.mapStart = observed
        }
    }
    func mapEnd(position: Int, layer: Int) {
        guard includes(position) else { return }
        let observed = QwenProductionTimingMeasurement.uptime()
        update(position: position, layer: layer) {
            if $0.mapEnd != 0 { $0.duplicate = true }
            $0.mapEnd = observed
        }
    }
    func plan(position: Int, layer: Int, residents: [Int], plan: ExpertCachePlan) {
        guard includes(position) else { return }
        guard residents.count == 16 && plan.experts.count == 8
            && plan.assignedSlots.count == 8 else {
            storage.withLock { $0.failure = "map geometry mismatch" }
            return
        }
        update(position: position, layer: layer) { row in
            if row.map != nil { row.duplicate = true }
            row.map = Map(residentSlots: residents, experts: plan.experts,
                          missIndices: plan.misses, assignedSlots: plan.assignedSlots)
        }
    }
    func isDrained() -> Bool { storage.withLock { !$0.enabled } }
    func drain() -> State {
        storage.withLock { state in
            let result = state
            state = State(enabled: false)
            return result
        }
    }
}

struct QwenExpertPredictionMapContext: Sendable {
    let capture: QwenExpertPredictionCapture
    let position: Int
    let layer: Int
}
