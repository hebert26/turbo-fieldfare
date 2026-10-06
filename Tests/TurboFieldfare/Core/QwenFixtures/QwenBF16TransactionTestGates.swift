import Foundation

/// One-shot synchronous barrier for the protected-read hook, which runs on a
/// bounded worker and cannot suspend asynchronously.
final class QwenBF16TransactionSyncGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var entered = false
    private var released = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []

    func suspend() {
        condition.lock()
        entered = true
        let waiters = entryWaiters
        entryWaiters.removeAll()
        condition.broadcast()
        condition.unlock()
        for waiter in waiters { waiter.resume() }

        condition.lock()
        while !released { condition.wait() }
        condition.unlock()
    }

    func waitUntilEntered() async {
        await withCheckedContinuation { continuation in
            condition.lock()
            if entered {
                condition.unlock()
                continuation.resume()
            } else {
                entryWaiters.append(continuation)
                condition.unlock()
            }
        }
    }

    func release() {
        condition.lock()
        released = true
        condition.broadcast()
        condition.unlock()
    }
}

/// One-shot async barrier for transaction/GPU hooks. Tests can hold an actual
/// pipeline boundary and release it explicitly without timing sleeps.
actor QwenBF16TransactionAsyncGate {
    private var entered = false
    private var released = false
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiter: CheckedContinuation<Void, Never>?

    func suspend() async {
        entered = true
        let waiters = entryWaiters
        entryWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
        guard !released else { return }
        await withCheckedContinuation { continuation in
            releaseWaiter = continuation
        }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { continuation in
            entryWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        let waiter = releaseWaiter
        releaseWaiter = nil
        waiter?.resume()
    }
}

/// Thread-safe one-shot switch for failures armed after a committed baseline.
final class QwenBF16TransactionTestSwitch: @unchecked Sendable {
    private let lock = NSLock()
    private var armed = false

    func arm() {
        lock.lock()
        armed = true
        lock.unlock()
    }

    func consume() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard armed else { return false }
        armed = false
        return true
    }

    var isArmed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return armed
    }
}

/// Small synchronized event journal for Sendable GPU/source hook closures.
final class QwenBF16TransactionTestRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [String] = []

    func append(_ entry: String) {
        lock.lock()
        entries.append(entry)
        lock.unlock()
    }

    /// Atomically records an event and returns the current count of entries
    /// matching a prefix. This lets tests identify a specific repeated stage.
    func appendAndCount(_ entry: String, matchingPrefix prefix: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        entries.append(entry)
        return entries.reduce(into: 0) { count, value in
            if value.hasPrefix(prefix) { count += 1 }
        }
    }

    func removeAll() {
        lock.lock()
        entries.removeAll()
        lock.unlock()
    }

    var snapshot: [String] {
        lock.lock()
        defer { lock.unlock() }
        return entries
    }
}
