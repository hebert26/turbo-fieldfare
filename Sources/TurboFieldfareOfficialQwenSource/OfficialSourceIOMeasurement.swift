import Darwin
import Foundation

/// One request's fixed numeric totals. Worker wall sums can overlap and are
/// not elapsed request time. No paths, identifiers, file contents or logs.
package final class OfficialSourceIOMeasurement: @unchecked Sendable {
    package enum Phase: Int, Sendable { case prefill = 0, decode = 1 }
    package enum Site: Int, Sendable { case initial = 0, beforeRead, afterRead, beforeReturn }
    package struct Counter: Sendable {
        package var count: UInt64 = 0
        package var wallNanoseconds: UInt64 = 0
        package var threadCPUNanoseconds: UInt64 = 0
        package var errors: UInt64 = 0
        package var unavailableCPU: UInt64 = 0
        package var bytes: UInt64 = 0
        package var interruptions: UInt64 = 0
        mutating func merge(_ other: Self) {
            count &+= other.count; wallNanoseconds &+= other.wallNanoseconds
            threadCPUNanoseconds &+= other.threadCPUNanoseconds; errors &+= other.errors
            unavailableCPU &+= other.unavailableCPU; bytes &+= other.bytes
            interruptions &+= other.interruptions
        }
    }
    struct LocalRead {
        let phase: Phase
        let zeroByte: Bool
        var initial = Counter(), beforeRead = Counter(), afterRead = Counter(), beforeReturn = Counter()
        var pread = Counter()
        var failed = true
    }
    package struct Snapshot: Sendable {
        /// Index: phase*8 + byteClass*4 + site; byteClass0=zero,1=payload.
        package let validations: [Counter]
        /// Index: phase*2 + byteClass.
        package let preads: [Counter]
        package let reads: [UInt64]
        package let failedReads: [UInt64]
    }
    private struct Clock {
        let wall: UInt64
        let cpu: UInt64?
        static func read() -> Self {
            var value = timespec()
            let cpu = clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0
                ? UInt64(value.tv_sec) * 1_000_000_000 + UInt64(value.tv_nsec) : nil
            return Self(wall: DispatchTime.now().uptimeNanoseconds, cpu: cpu)
        }
    }
    private let lock = NSLock()
    private var phase: Phase = .prefill
    private var validations = [Counter](repeating: Counter(), count: 16)
    private var preads = [Counter](repeating: Counter(), count: 4)
    private var reads = [UInt64](repeating: 0, count: 4)
    private var failedReads = [UInt64](repeating: 0, count: 4)

    package init() {}
    package func setPhase(_ value: Phase) { lock.lock(); phase = value; lock.unlock() }
    func beginRead(byteCount: UInt64) -> LocalRead {
        lock.lock(); defer { lock.unlock() }
        return LocalRead(phase: phase, zeroByte: byteCount == 0)
    }
    func merge(_ read: LocalRead) {
        lock.lock(); defer { lock.unlock() }
        let index = read.phase.rawValue * 2 + (read.zeroByte ? 0 : 1)
        validations[index * 4].merge(read.initial)
        validations[index * 4 + 1].merge(read.beforeRead)
        validations[index * 4 + 2].merge(read.afterRead)
        validations[index * 4 + 3].merge(read.beforeReturn)
        preads[index].merge(read.pread)
        reads[index] &+= 1
        if read.failed { failedReads[index] &+= 1 }
    }
    package func snapshot() -> Snapshot {
        lock.lock(); defer { lock.unlock() }
        return Snapshot(validations: validations, preads: preads, reads: reads, failedReads: failedReads)
    }
    private static func duration(_ start: Clock, _ end: Clock, failed: Bool) -> Counter {
        var result = Counter()
        result.count = 1
        result.wallNanoseconds = end.wall >= start.wall ? end.wall - start.wall : 0
        if let first = start.cpu, let last = end.cpu, last >= first {
            result.threadCPUNanoseconds = last - first
        } else { result.unavailableCPU = 1 }
        result.errors = failed ? 1 : 0
        return result
    }
    static func validate(_ local: inout LocalRead?, site: Site, _ body: () throws -> Void) rethrows {
        guard local != nil else { try body(); return }
        let start = Clock.read()
        var failed = true
        defer {
            let counter = duration(start, Clock.read(), failed: failed)
            switch site {
            case .initial: local!.initial.merge(counter)
            case .beforeRead: local!.beforeRead.merge(counter)
            case .afterRead: local!.afterRead.merge(counter)
            case .beforeReturn: local!.beforeReturn.merge(counter)
            }
        }
        try body()
        failed = false
    }
    static func pread(_ local: inout LocalRead?, requested: Int,
                      _ body: () -> OfficialSourceHandle.TensorPreadResult)
        -> OfficialSourceHandle.TensorPreadResult {
        guard local != nil else { return body() }
        let start = Clock.read()
        let outcome = body()
        var counter = duration(start, Clock.read(), failed: false)
        switch outcome {
        case .interrupted: counter.interruptions = 1
        case .failure(let code):
            if code == EINTR { counter.interruptions = 1 } else { counter.errors = 1 }
        case .bytes(let count):
            if count > 0, count <= requested { counter.bytes = UInt64(count) }
            else { counter.errors = 1 }
        }
        local!.pread.merge(counter)
        return outcome
    }
}
