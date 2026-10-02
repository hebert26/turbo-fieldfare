import Foundation
import Synchronization

final class QwenEarlyPrefetchCapture: Sendable {
    enum Site: String, Codable, CaseIterable, Sendable {
        case fullSource, hitValidation, beforeCopyValidation, copy, publicationValidation
        case demandProtectedRead, speculativeProtectedRead, drain, hint, map
    }
    struct Counter: Codable, Sendable {
        var count: UInt64 = 0
        var wallNanoseconds: UInt64 = 0
        var requestedBytes: UInt64 = 0
        var failures: UInt64 = 0
    }
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
        var counters: [Site: Counter] = [:]
        var duplicate = false
        var speculativePairs: UInt64 = 0
        var importedPairs: UInt64 = 0
        var unusedPairs: UInt64 = 0
    }
    private let storage = Mutex(State())
    func begin() { storage.withLock { $0.active = true } }
    func snapshot() -> State { storage.withLock { $0 } }
    func measure<T>(_ site: Site, bytes: UInt64 = 0, _ body: () throws -> T) rethrows -> T {
        guard storage.withLock({ $0.active }) else { return try body() }
        let start = QwenProductionTimingMeasurement.uptime()
        var succeeded = false
        defer {
            let elapsed = QwenProductionTimingMeasurement.uptime() - start
            storage.withLock { value in
                value.counters[site, default: Counter()].count += 1
                value.counters[site, default: Counter()].wallNanoseconds += elapsed
                value.counters[site, default: Counter()].requestedBytes += bytes
                if !succeeded { value.counters[site, default: Counter()].failures += 1 }
            }
        }
        let result = try body()
        succeeded = true
        return result
    }
    func duration(_ site: Site, start: UInt64) {
        let elapsed = QwenProductionTimingMeasurement.uptime() - start
        storage.withLock { value in
            guard value.active else { return }
            value.counters[site, default: Counter()].count += 1
            value.counters[site, default: Counter()].wallNanoseconds += elapsed
        }
    }
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
    func completedBatch(predicted: Int, imported: Int) {
        storage.withLock { value in
            value.speculativePairs += UInt64(predicted)
            value.importedPairs += UInt64(imported)
            value.unusedPairs += UInt64(predicted - imported)
        }
    }
}
struct QwenEarlyPrefetchMapContext: Sendable {
    let capture: QwenEarlyPrefetchCapture
    let position: Int
    let layer: Int
}
