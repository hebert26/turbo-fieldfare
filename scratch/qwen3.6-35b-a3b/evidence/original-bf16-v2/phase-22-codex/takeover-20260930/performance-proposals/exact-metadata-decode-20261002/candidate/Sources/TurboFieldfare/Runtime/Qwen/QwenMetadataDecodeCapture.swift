import Foundation
import Synchronization

// Fixed decode evidence only. No activation or transaction callback is installed.
final class QwenMetadataDecodeCapture: Sendable {
    struct Key: Hashable, Sendable { let position: Int; let layer: Int }
    struct Route: Codable, Sendable {
        let position: Int
        let layer: Int
        let experts: [Int]
        let weightBits: [UInt32]
    }
    struct Plan: Codable, Sendable {
        let position: Int
        let layer: Int
        let experts: [Int]
        let assignedSlots: [Int]
        let missIndices: [Int]
        let residents: [Int]
    }
    struct State: Sendable {
        var active = false
        var routes: [Key: Route] = [:]
        var plans: [Key: Plan] = [:]
        var duplicate = false
    }
    private let storage = Mutex(State())
    func begin() { storage.withLock { $0.active = true } }
    func snapshot() -> State { storage.withLock { $0 } }
    func route(position: Int, layer: Int, experts: [Int], weights: [Float]) {
        storage.withLock { value in
            guard value.active, (1175..<1203).contains(position), (0..<40).contains(layer) else { return }
            let key = Key(position: position, layer: layer)
            if value.routes[key] != nil { value.duplicate = true }
            value.routes[key] = Route(position: position, layer: layer, experts: experts, weightBits: weights.map(\.bitPattern))
        }
    }
    func plan(position: Int, layer: Int, plan: ExpertCachePlan, residents: [Int]) {
        storage.withLock { value in
            guard value.active, (1175..<1203).contains(position), (0..<40).contains(layer) else { return }
            let key = Key(position: position, layer: layer)
            if value.plans[key] != nil { value.duplicate = true }
            value.plans[key] = Plan(position: position, layer: layer, experts: plan.experts,
                assignedSlots: plan.assignedSlots, missIndices: plan.misses, residents: residents)
        }
    }
}
struct QwenMetadataDecodeMapContext: Sendable {
    let capture: QwenMetadataDecodeCapture
    let position: Int
    let layer: Int
}
