import Foundation
import Synchronization

// One produce call owns this latch and retains it until all entry lanes join.
package final class OfficialSourceDecodeCancellation: Sendable {
    private let canceled = Mutex(false)
    package init() {}
    package func cancel() { canceled.withLock { $0 = true } }
    package func check() throws {
        if canceled.withLock({ $0 }) { throw CancellationError() }
    }
}

// Counters cover decode only. Locks never cover filesystem work or worker joins.
package final class OfficialSourceDecodeMeasurement: Sendable {
    package struct Snapshot: Codable, Sendable {
        package var count: UInt64 = 0
        package var wallNanoseconds: UInt64 = 0
        package var failures: UInt64 = 0
        package var parallelCount: UInt64 = 0
        package var parallelWallNanoseconds: UInt64 = 0
        package var entryChecks: UInt64 = 0
        package var maximumEntryFDs: Int = 0
    }
    private struct State: Sendable { var snapshot = Snapshot(); var activeFDs = 0 }
    private let state = Mutex(State())
    package init() {}
    package func snapshot() -> Snapshot { state.withLock { $0.snapshot } }
    package func measure<T>(parallel: Bool, _ body: () throws -> T) rethrows -> T {
        let start = DispatchTime.now().uptimeNanoseconds
        var success = false
        defer {
            let wall = DispatchTime.now().uptimeNanoseconds - start
            state.withLock { value in
                value.snapshot.count += 1
                value.snapshot.wallNanoseconds += wall
                if parallel {
                    value.snapshot.parallelCount += 1
                    value.snapshot.parallelWallNanoseconds += wall
                }
                if !success { value.snapshot.failures += 1 }
            }
        }
        let result = try body()
        success = true
        return result
    }
    package func entry() { state.withLock { $0.snapshot.entryChecks += 1 } }
    package func opened() {
        state.withLock { value in
            value.activeFDs += 1
            value.snapshot.maximumEntryFDs = max(value.snapshot.maximumEntryFDs, value.activeFDs)
        }
    }
    package func closed() { state.withLock { $0.activeFDs -= 1 } }
    package func drained() -> Bool { state.withLock { $0.activeFDs == 0 } }
}
