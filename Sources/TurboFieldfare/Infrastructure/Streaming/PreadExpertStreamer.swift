import Darwin
import Foundation
import Metal
import TurboFieldfareOfficialQwenSource

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
    }
}

public enum ExpertCachePolicy: String, Sendable, Equatable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    private var useClock = 0
    private var successfulPlanHits: UInt64 = 0
    private var successfulPlanMisses: UInt64 = 0
    private let cacheLock = NSLock()

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout,
                      device: device,
                      slotCount: slotCount,
                      cachePolicy: cachePolicy,
                      fileDescriptor: nil)
    }

    package init(layout: StreamLayout,
                 device: MTLDevice,
                 slotCount: Int,
                 cachePolicy: ExpertCachePolicy = .lfu,
                 fileDescriptor: Int32?) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        let pageSize = Int(getpagesize())

        let openedFD = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(layout.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD
        var closeFDOnFailure = true
        defer { if closeFDOnFailure { close(openedFD) } }

        var fileStats = stat()
        guard fstat(openedFD, &fileStats) == 0,
              (fileStats.st_mode & S_IFMT) == S_IFREG,
              fileStats.st_size >= 0 else {
            throw StreamerError.openFailed(
                path: layout.path, errno: errno == 0 ? EINVAL : errno)
        }
        let (required, requiredOverflow) = layout.streamOffset
            .addingReportingOverflow(layout.streamSize)
        guard !requiredOverflow, UInt64(fileStats.st_size) >= required else {
            throw StreamerError.sizeMismatch(
                expected: requiredOverflow ? UInt64.max : required,
                actual: UInt64(fileStats.st_size))
        }
        guard layout.expertStride > 0,
              layout.expertStride <= UInt64(Int.max - (pageSize - 1)) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "expertStride \(layout.expertStride) is not addressable")
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
        }

        for _ in 0..<slotCount {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: allocationSize,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        closeFDOnFailure = false
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        let slot = nextSlot
        nextSlot = (nextSlot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride))
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }
        guard misses.count <= evictable.count else { return nil }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
            slotExpert[slot] = -1
            slotLastUse[slot] = clock
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan,
                                       measurement: RuntimeMeasurementCapture? = nil,
                                       measurementPlanID: UInt64 = 0,
                                       measurementLayer: Int = 0) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: plan.misses.count) { missOffset in
            let index = plan.misses[missOffset]
            do {
                _ = try self.loadExpert(
                    layer: 0,
                    expert: plan.experts[index],
                    slot: plan.assignedSlots[index])
                if let measurement {
                    // Count each fully successful logical read even when another
                    // miss in this plan fails. Partial failed reads remain unknown.
                    measurement.recordSuccessfulMiss(
                        layer: measurementLayer, planID: measurementPlanID,
                        expert: plan.experts[index], slot: plan.assignedSlots[index],
                        bytes: self.layout.expertStride)
                }
            } catch {
                errorLock.lock()
                if firstError == nil { firstError = error }
                errorLock.unlock()
            }
        }
        if let firstError { throw firstError }

        cacheLock.lock()
        for index in plan.misses {
            slotExpert[plan.assignedSlots[index]] = plan.experts[index]
        }
        successfulPlanHits += UInt64(plan.hits)
        successfulPlanMisses += UInt64(plan.misses.count)
        cacheLock.unlock()

        return expertCachePlanBuffers(plan)
    }

    /// The model's all-hit fast path serves a successful plan without running
    /// the executor. Raw buffer previews and planning alone do not count.
    func recordSuccessfulCachePlan(_ plan: ExpertCachePlan) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        successfulPlanHits += UInt64(plan.hits)
        successfulPlanMisses += UInt64(plan.misses.count)
    }

    /// Lifetime counts include only plans whose complete fetch succeeded.
    /// A failed plan contributes neither hits nor partially completed misses.
    var successfulCachePlanCounts: (hits: UInt64, misses: UInt64) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return (successfulPlanHits, successfulPlanMisses)
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], UInt64(0), layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }

    /// Bytes of the actual Metal cache buffers this streamer owns. Each slot
    /// buffer is counted once; configured slot geometry is not used as a
    /// substitute for the buffers that were successfully created.
    public var allocatedCacheBytes: UInt64 {
        slotBuffers.reduce(UInt64(0)) { total, buffer in
            total + UInt64(buffer.length)
        }
    }

    /// CPU-side scratch owned by this streamer. Diagnostic metadata for
    /// residency reporting, not an ownership or lifetime API.
    public var diagnosticSlotScratchBytes: UInt64 {
        allocatedCacheBytes
    }

    func recordMeasurementSnapshot(_ capture: RuntimeMeasurementCapture, layer: Int, reason: UInt64) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        capture.recordCacheLayer(layer: layer, reason: reason, slots: slotExpert,
                                 lastUse: slotLastUse, frequencies: expertUseCount,
                                 useClock: useClock, scratchBytes: diagnosticSlotScratchBytes)
    }
}


/// Names alone cannot create an admissible range; each stream is separately
/// resolved through the protected handle and may belong to a different shard.
struct QwenBF16RoutedSourceNames: Sendable {
    let gateUpShardName: String
    let gateUpTensorName: String
    let downShardName: String
    let downTensorName: String
}

/// @testable-only deterministic gates around the REAL protected reads. No
/// callback supplies bytes, file descriptors, offsets, identities or a token.
struct QwenBF16ExpertReadHooks: Sendable {
    enum Stream: Sendable { case gateUp, down }
    enum Checkpoint: Sendable {
        case beforeProtectedRead(expert: Int, stream: Stream)
        case afterProtectedRead(expert: Int, stream: Stream)
        case beforePairPublish(expert: Int)
    }

    let checkpoint: @Sendable (Checkpoint) throws -> Void
    init(checkpoint: @escaping @Sendable (Checkpoint) throws -> Void = { _ in }) {
        self.checkpoint = checkpoint
    }
    static let none = Self()
}

enum QwenBF16ExpertCacheError: Error, Equatable, Sendable {
    case invalidGeometry
    case invalidHeader(String)
    case budgetExceeded
    case allocationFailed
    case invalidPlan
}

/// Private-to-runtime paired cache. The coordinator is the only production
/// caller of plan/fetch; it serializes them against GPU slot reservations.
/// No slot is valid unless BOTH independently admitted slices were read.
final class QwenBF16PairedExpertCache: @unchecked Sendable {
    struct MeasurementSnapshot: Sendable {
        let expertIDs: [Int]
        let useCounts: [UInt64]
        let lastUse: [UInt64]
        let clock: UInt64
        let policy: ExpertCachePolicy
    }
    struct Buffers: @unchecked Sendable {
        let gateUp: MTLBuffer
        let down: MTLBuffer
    }

    /// Errors are ordered by requested pair, then gate/up before down, rather
    /// than whichever protected read happens to finish first.
    private final class ReadFailures: @unchecked Sendable {
        private let lock = NSLock()
        private var errors: [Error?]
        init(count: Int) { errors = [Error?](repeating: nil, count: count) }
        func record(_ error: Error, index: Int) {
            lock.lock(); errors[index] = error; lock.unlock()
        }
        func first() -> Error? {
            lock.lock(); defer { lock.unlock() }
            return errors.first(where: { $0 != nil }) ?? nil
        }
    }

    private static let maximumConcurrentMissPairs = 8

    let slotCount: Int
    let expertCount: Int
    let gateUpBytes: Int
    let downBytes: Int
    let allocatedCacheBytes: UInt64

    private let source: OfficialSourceHandle
    private let gateUpRange: OfficialSourceHandle.TensorRange
    private let downRange: OfficialSourceHandle.TensorRange
    private let buffers: [Buffers]
    /// Accessed only under the owning coordinator's serialized I/O lock.
    private var slotExpert: [Int]
    private var lastUse: [UInt64]
    private var useCount: [UInt64]
    private let cachePolicy: ExpertCachePolicy
    private var clock: UInt64 = 0

    /// Caller must hold the owning coordinator's I/O lock. Array values are
    /// consumed synchronously before planning can mutate their backing storage.
    func measurementSnapshot() -> MeasurementSnapshot {
        MeasurementSnapshot(expertIDs: slotExpert, useCounts: useCount,
                            lastUse: lastUse, clock: clock, policy: cachePolicy)
    }

    var measurementClock: UInt64 { clock }

    /// Immutable slot allocations only. Payload/validity and replacement policy
    /// remain private; residency registration never reads source or buffer data.
    var residencyAllocations: [MTLBuffer] {
        buffers.flatMap { [$0.gateUp, $0.down] }
    }

    init(source: OfficialSourceHandle, names: QwenBF16RoutedSourceNames,
         expertCount: Int, hiddenSize: Int, intermediateSize: Int,
         device: MTLDevice, slotCount: Int, residencyBudget: UInt64,
         cachePolicy: ExpertCachePolicy = .lru) throws {
        guard expertCount > 0, UInt32(exactly: expertCount) != nil,
              hiddenSize > 0, UInt32(exactly: hiddenSize) != nil,
              intermediateSize > 0, UInt32(exactly: intermediateSize) != nil,
              slotCount > 0, slotCount <= expertCount else {
            throw QwenBF16ExpertCacheError.invalidGeometry
        }
        func multiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else { throw QwenBF16ExpertCacheError.invalidGeometry }
            return value
        }
        let doubleIntermediate = try multiply(UInt64(intermediateSize), 2)
        guard doubleIntermediate <= UInt64(UInt32.max) else {
            throw QwenBF16ExpertCacheError.invalidGeometry
        }
        let gateBytes = try multiply(
            try multiply(doubleIntermediate, UInt64(hiddenSize)), 2)
        let downBytes = try multiply(
            try multiply(UInt64(hiddenSize), UInt64(intermediateSize)), 2)
        let (pairBytes, sumOverflow) = gateBytes.addingReportingOverflow(downBytes)
        guard !sumOverflow else { throw QwenBF16ExpertCacheError.invalidGeometry }
        let gateTensorBytes = try multiply(gateBytes, UInt64(expertCount))
        let downTensorBytes = try multiply(downBytes, UInt64(expertCount))
        let total = try multiply(pairBytes, UInt64(slotCount))
        guard gateBytes > 0, downBytes > 0,
              gateTensorBytes <= UInt64(Int64.max), downTensorBytes <= UInt64(Int64.max),
              gateBytes / 2 <= UInt64(UInt32.max),
              downBytes / 2 <= UInt64(UInt32.max),
              gateBytes <= UInt64(device.maxBufferLength),
              downBytes <= UInt64(device.maxBufferLength),
              gateBytes <= UInt64(Int.max), downBytes <= UInt64(Int.max) else {
            throw QwenBF16ExpertCacheError.invalidGeometry
        }
        guard total <= residencyBudget else { throw QwenBF16ExpertCacheError.budgetExceeded }
        let gate = try source.admitTensor(
            shardName: names.gateUpShardName, tensorName: names.gateUpTensorName)
        let down = try source.admitTensor(
            shardName: names.downShardName, tensorName: names.downTensorName)
        guard gate.shape == [UInt64(expertCount), doubleIntermediate, UInt64(hiddenSize)] else {
            throw QwenBF16ExpertCacheError.invalidHeader(names.gateUpTensorName)
        }
        guard down.shape == [UInt64(expertCount), UInt64(hiddenSize), UInt64(intermediateSize)] else {
            throw QwenBF16ExpertCacheError.invalidHeader(names.downTensorName)
        }
        // Both tokens validate their issuer, root, marker and retained shard
        // identity before the first GPU allocation. No payload is read here.
        try Self.validate(source: source, range: gate)
        try Self.validate(source: source, range: down)
        var slots: [Buffers] = []
        slots.reserveCapacity(slotCount)
        for _ in 0..<slotCount {
            try Task.checkCancellation()
            guard let gateBuffer = device.makeBuffer(length: Int(gateBytes),
                                                      options: .storageModeShared),
                  let downBuffer = device.makeBuffer(length: Int(downBytes),
                                                      options: .storageModeShared) else {
                throw QwenBF16ExpertCacheError.allocationFailed
            }
            slots.append(Buffers(gateUp: gateBuffer, down: downBuffer))
        }
        self.source = source
        gateUpRange = gate
        downRange = down
        buffers = slots
        self.slotCount = slotCount
        self.expertCount = expertCount
        gateUpBytes = Int(gateBytes)
        self.downBytes = Int(downBytes)
        allocatedCacheBytes = total
        slotExpert = [Int](repeating: -1, count: slotCount)
        lastUse = [UInt64](repeating: 0, count: slotCount)
        useCount = [UInt64](repeating: 0, count: slotCount)
        self.cachePolicy = cachePolicy
    }

    /// Pure plan over ONE validity array. Active slots remain eligible only
    /// when they already contain the requested expert; misses cannot evict one.
    func plan(expertIDs: [Int], avoidingSlots: Set<Int>) -> ExpertCachePlan? {
        guard expertIDs.count <= slotCount,
              Set(expertIDs).count == expertIDs.count,
              expertIDs.allSatisfy({ $0 >= 0 && $0 < expertCount }) else { return nil }
        var assigned = [Int](repeating: -1, count: expertIDs.count)
        var reserved = avoidingSlots
        for (index, expert) in expertIDs.enumerated() {
            if let hit = slotExpert.indices.first(where: {
                slotExpert[$0] == expert && !assigned.contains($0)
            }) {
                assigned[index] = hit
                reserved.insert(hit)
            }
        }
        let misses = expertIDs.indices.filter { assigned[$0] < 0 }
        let victims = slotExpert.indices.filter { !reserved.contains($0) }
            .sorted { lhs, rhs in
                if cachePolicy == .lfu, useCount[lhs] != useCount[rhs] {
                    return useCount[lhs] < useCount[rhs]
                }
                return lastUse[lhs] == lastUse[rhs] ? lhs < rhs : lastUse[lhs] < lastUse[rhs]
            }
        guard misses.count <= victims.count else { return nil }
        clock &+= 1
        for (offset, index) in misses.enumerated() {
            assigned[index] = victims[offset]
        }
        for (index, slot) in assigned.enumerated() {
            if misses.contains(index) { useCount[slot] = 0 }
            useCount[slot] &+= 1
            lastUse[slot] = clock
        }
        return ExpertCachePlan(experts: expertIDs, assignedSlots: assigned,
                               misses: misses, hits: expertIDs.count - misses.count)
    }

    func load(_ plan: ExpertCachePlan, hooks: QwenBF16ExpertReadHooks,
              canceled: @Sendable () -> Bool) throws -> [Buffers] {
        guard plan.experts.count == plan.assignedSlots.count,
              Set(plan.assignedSlots).count == plan.assignedSlots.count,
              Set(plan.misses).count == plan.misses.count,
              plan.assignedSlots.allSatisfy({ buffers.indices.contains($0) }),
              plan.misses.allSatisfy({ plan.experts.indices.contains($0) }) else {
            throw QwenBF16ExpertCacheError.invalidPlan
        }
        let missSet = Set(plan.misses)
        for index in plan.experts.indices where !missSet.contains(index) {
            let slot = plan.assignedSlots[index]
            guard slotExpert[slot] == plan.experts[index] else {
                throw QwenBF16ExpertCacheError.invalidPlan
            }
            do {
                try validateBoth()
            } catch {
                slotExpert[slot] = -1
                throw error
            }
        }
        var missOffset = 0
        while missOffset < plan.misses.count {
            if canceled() { throw CancellationError() }
            let pairCount = min(Self.maximumConcurrentMissPairs, plan.misses.count - missOffset)
            let batch = Array(plan.misses[missOffset..<(missOffset + pairCount)])
            for index in batch {
                // All destinations are distinct reserved victims. Invalidate
                // them serially before any worker can overwrite either stream.
                slotExpert[plan.assignedSlots[index]] = -1
            }
            let failures = ReadFailures(count: pairCount * 2)
            DispatchQueue.concurrentPerform(iterations: pairCount * 2) { worker in
                let index = batch[worker / 2]
                let expert = plan.experts[index]
                let pair = self.buffers[plan.assignedSlots[index]]
                let stream = worker % 2
                do {
                    let kind: QwenBF16ExpertReadHooks.Stream = stream == 0 ? .gateUp : .down
                    try hooks.checkpoint(.beforeProtectedRead(expert: expert, stream: kind))
                    if stream == 0 {
                        try self.read(expert: expert, range: self.gateUpRange,
                                      into: pair.gateUp, sliceBytes: self.gateUpBytes)
                    } else {
                        try self.read(expert: expert, range: self.downRange,
                                      into: pair.down, sliceBytes: self.downBytes)
                    }
                    try hooks.checkpoint(.afterProtectedRead(expert: expert, stream: kind))
                } catch { failures.record(error, index: worker) }
            }
            // Join every launched stream even on failure or cancellation.
            // Failed/partial victims remain invalid; no pair is published yet.
            if let error = failures.first() { throw error }
            if canceled() { throw CancellationError() }
            missOffset += pairCount
        }
        // All requested miss reads have settled before publication. Keep hooks,
        // identity checks and paired validity updates serial in request order.
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            let expert = plan.experts[index]
            if canceled() { throw CancellationError() }
            try hooks.checkpoint(.beforePairPublish(expert: expert))
            if canceled() { throw CancellationError() }
            // The hook may mutate a retained shard or marker. Revalidate both
            // tokens after it, immediately before making this pair a cache hit.
            try validateBoth()
            if canceled() { throw CancellationError() }
            slotExpert[slot] = expert
        }
        if canceled() { throw CancellationError() }
        return plan.assignedSlots.map { buffers[$0] }
    }

    private func validateBoth() throws {
        try Self.validate(source: source, range: gateUpRange)
        try Self.validate(source: source, range: downRange)
    }

    private static func validate(source: OfficialSourceHandle,
                                 range: OfficialSourceHandle.TensorRange) throws {
        try source.preadTensorRange(
            range, byteOffset: 0, byteCount: 0, expectedByteCount: 0,
            into: UnsafeMutableRawBufferPointer(start: nil, count: 0))
    }

    private func read(expert: Int, range: OfficialSourceHandle.TensorRange,
                      into buffer: MTLBuffer, sliceBytes: Int) throws {
        let base = UInt64(expert) * UInt64(sliceBytes) // admitted total shape checked
        var copied = 0
        while copied < sliceBytes {
            let count = min(sliceBytes - copied,
                            Int(OfficialSourceHandle.maximumTensorReadBytes))
            let destination = UnsafeMutableRawBufferPointer(
                start: buffer.contents().advanced(by: copied), count: count)
            try source.preadTensorRange(
                range, byteOffset: base + UInt64(copied), byteCount: UInt64(count),
                expectedByteCount: UInt64(count), into: destination)
            copied += count
        }
    }
}
