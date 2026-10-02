import Darwin
import Foundation
import TurboFieldfareOfficialQwenSource

/// Request-scoped inheritance crosses actor calls, but not dispatch workers.
/// The runner passes an explicit context to each serialized cache map worker.
enum QwenCacheMapMeasurement {
    @TaskLocal static var capture: RuntimeMeasurementCapture?
}

struct QwenCacheMapContext: Sendable {
    let capture: RuntimeMeasurementCapture
    let layer: Int
    let position: Int
    let tokenCount: Int
}

/// Request-owned numeric measurement. It never retains model, image or GPU resources.
/// Callers must test their optional capture before constructing payloads or reading clocks.
/// Storage is allocated once. Filling the ring drops records rather than growing it.
public final class RuntimeMeasurementCapture: @unchecked Sendable {
    public enum Phase: UInt64, Sendable { case decode = 1, prefill = 2 }
    public enum QwenCacheCaptureMode: UInt64, Sendable {
        case decode = 1, prefillAndDecode = 3
    }

    /// Wraps the existing Data without copying its payload. Transport must protect
    /// every footer-bearing batch, including a batch that also contains tail detail.
    public struct DrainedBatch: Sendable {
        public let data: Data
        public let containsFooter: Bool
    }

    /// Each row is [kind, a, b, c, d, e]. IDs, positions and byte counts are exact integers.
    public enum Event: UInt64, Sendable {
        case configuration = 1, window = 2, route = 3, plan = 4, planMembers = 5
        case successfulMiss = 6, fetchQueue = 7, fetchCompleted = 8, fetchFailed = 9
        case cacheBoundary = 10, cacheLayer = 11, cacheSlot = 12, cacheFrequency = 13
        case streamerOpened = 14, residencyRelease = 15, memory = 16
        case prefillChunk = 100, prefillTile = 101, prefillFetch = 102, prefillComplete = 103
        case decodeFetch = 104
        case visionBegin = 110, visionEnd = 111, visionFailure = 112
        case fullAttentionTimingConfiguration = 120, fullAttentionTimingStorage = 121
        case fullAttentionTimingCalibration = 122, fullAttentionTimingSamples = 123
        case fullAttentionStageDurations = 124, fullAttentionHostTiming = 125
        case fullAttentionTimingForward = 126
        case sourceIOValidation = 130, sourceIOValidationOutcomes = 131
        case sourceIOPread = 132, sourceIOPreadOutcomes = 133, sourceIOReads = 134
        case sourceIOConfiguration = 135
        case qwenCacheConfiguration = 140, qwenCacheInitialLayer = 141
        case qwenCacheInitialSlot = 142, qwenCacheMap = 143, qwenCachePlan = 144
        case qwenCacheConstraints = 145, qwenCacheAvoidingSlot = 146
        case qwenCacheMember = 147, qwenCacheOutcome = 148, qwenCacheCompletion = 149
        case qwenCacheScope = 150
        case qwenProductionScope = 160, qwenProductionCommand = 161
        case qwenProductionSubmitWait = 162, qwenProductionGPU = 163
        case qwenProductionSpan = 164, qwenProductionHostClocks = 165
        case qwenProductionCompletion = 166, qwenProductionPhase = 167
        case qwenProductionClockCorrelation = 168
        case qwenProducerPriority = 169, qwenProductionResumption = 170
        case qwenProductionDriver = 171
        case summary = 200, layerSummary = 201, expertSummary = 202, omittedRoutes = 203
        case schedulingSummary = 204, serializationSummary = 205
        case observedCoverage = 206, droppedRange = 207
    }

    private struct Record {
        var kind: UInt64 = 0
        var a: UInt64 = 0
        var b: UInt64 = 0
        var c: UInt64 = 0
        var d: UInt64 = 0
        var e: UInt64 = 0
    }

    private struct Layer {
        var phase: UInt64 = 0
        var position: UInt64 = 0
        var detailed: Bool = false
        var decodeRoutes: UInt64 = 0
        var prefillRoutes: UInt64 = 0
        var firstDecodePosition: UInt64 = .max
        var lastDecodePosition: UInt64 = .max
        var firstPrefillPosition: UInt64 = .max
        var lastPrefillPosition: UInt64 = .max
        var omittedDecode: UInt64 = 0
        var omittedPrefill: UInt64 = 0
        var firstOmittedDecode: UInt64 = .max
        var lastOmittedDecode: UInt64 = 0
        var firstOmittedPrefill: UInt64 = .max
        var lastOmittedPrefill: UInt64 = 0
        var plans: UInt64 = 0
        var hits: UInt64 = 0
        var misses: UInt64 = 0
        var successfulMisses: UInt64 = 0
        var logicalBytes: UInt64 = 0
        var failedFetches: UInt64 = 0
        var fetches: UInt64 = 0
        var allHitFetches: UInt64 = 0
        var queueNanos: UInt64 = 0
        var workNanos: UInt64 = 0
        var resumeNanos: UInt64 = 0
        var allHitQueueNanos: UInt64 = 0
        var allHitWorkNanos: UInt64 = 0
        var allHitResumeNanos: UInt64 = 0
    }

    public static let recordCapacity = 65_536
    public static let decodePositionLimit = 128
    public static let prefillPositionLimit = 512
    private static let layerCount = 30
    private static let expertCount = 128
    private static let batchRecordLimit = 256
    public static let schemaVersion = 1
    public static let maximumJSONBatchBytes = 48 * 1_024
    /// Exact Qwen rows stop at this request bound even if transport drains the ring.
    static let maximumQwenCacheRecords = 48_000
    private static let maximumSerializedRowBytes = 128
    private static let maximumBatchEnvelopeBytes = 32
    private static let footerLayerRows = layerCount * 9
    private static let footerExpertRows = layerCount * expertCount
    public static let footerRecordCount = 3 + footerLayerRows + footerExpertRows

    /// Schema-v1 raw JSON bound: 4,113 footer rows, at most 255 mixed detail rows,
    /// and at most one envelope per footer row for every supported drain budget.
    /// This excludes transport/disk wrapper fields. Update this derivation if the
    /// row schema, footer population, envelope or batch limits change.
    public static let maximumSerializedFooterBytes =
        (footerRecordCount + batchRecordLimit - 1) * maximumSerializedRowBytes
        + footerRecordCount * maximumBatchEnvelopeBytes

    /// Aggregate-bearing batches when the caller keeps this same drain budget
    /// throughout finalization. A 48 KiB budget needs at most 18 such batches,
    /// including one mixed tail batch. Transport reserves its wrapper bytes too.
    public static func maximumFooterBatchCount(maximumBytes: Int) -> Int {
        let budget = min(maximumBytes, maximumJSONBatchBytes)
        guard budget >= 256 else { return 0 }
        let limit = min(batchRecordLimit,
                        (budget - maximumBatchEnvelopeBytes) / maximumSerializedRowBytes)
        let rowsIncludingMixedTail = footerRecordCount + limit - 1
        return (rowsIncludingMixedTail + limit - 1) / limit
    }

    private let lock = NSLock()
    private let drainLock = NSLock()
    private let storage: UnsafeMutableRawPointer
    private let storageLength: Int
    private let records: UnsafeMutablePointer<Record>
    private let batch: UnsafeMutablePointer<Record>
    private let layers: UnsafeMutablePointer<Layer>
    private let selections: UnsafeMutablePointer<UInt64>
    /// Page-rounded mapped backing bytes plus a conservative 4 KiB allowance
    /// for this object and its locks. Allocator metadata and process RSS are not measured.
    public let allocatedStorageBytes: Int
    private var readIndex = 0
    private var pending = 0
    private var recorded: UInt64 = 0
    private var dropped: UInt64 = 0
    private var firstDroppedOrdinal: UInt64 = .max
    private var lastDroppedOrdinal: UInt64 = .max
    private var nextPlanID: UInt64 = 0
    private var qwenProductionActive = false
    private var qwenProductionID: UInt64 = 0
    private var qwenProductionRows: UInt64 = 0
    private var qwenProductionDrops: UInt64 = 0
    private var qwenProductionForwards: UInt64 = 0
    private var qwenProductionOmittedForwards: UInt64 = 0
    private var qwenProductionCommands: UInt64 = 0
    static let maximumQwenProductionForwards: UInt64 = 32
    // Four-row commands and two-row worker handoffs fit the unchanged32-forward
    // diagnostic bound. The total transport ring stays fixed at65,536 rows.
    static let maximumQwenProductionRows: UInt64 = 32_768
    private var decodeStart: Int?
    private var prefillStart: Int?
    private var decodeClosed = false
    private var prefillClosed = false
    private var postImagePending = false
    private var preReleaseCaptured = false
    private var finished = false
    private var finishStatus: UInt64 = 0
    private var footerIndex = 0
    private var drainedBatches: UInt64 = 0
    private var drainedBytes: UInt64 = 0
    private var serializationNanos: UInt64 = 0
    private var qwenCacheActive = false
    private var qwenCachePhase: Phase = .prefill
    private var qwenCacheInitialLayers: UInt64 = 0
    private var qwenCacheRecords = 0
    private var qwenCacheAttempts: UInt64 = 0
    private var qwenCacheSuccesses: UInt64 = 0
    private var qwenCacheFailures: UInt64 = 0
    private var qwenCacheDrops: UInt64 = 0
    private let qwenCacheCaptureMode: QwenCacheCaptureMode

    public convenience init() {
        self.init(qwenCacheCaptureMode:
            ProcessInfo.processInfo.environment["TURBOFIELDFARE_QWEN_CACHE_CAPTURE_PHASE"] == "all"
                ? .prefillAndDecode : .decode)
    }

    public init(qwenCacheCaptureMode: QwenCacheCaptureMode) {
        self.qwenCacheCaptureMode = qwenCacheCaptureMode
        precondition(MemoryLayout<Record>.stride == 48)
        func aligned(_ offset: Int, to alignment: Int) -> Int {
            let remainder = offset % alignment
            return remainder == 0 ? offset : offset + alignment - remainder
        }
        let recordBytes = MemoryLayout<Record>.stride * Self.recordCapacity
        let batchOffset = aligned(recordBytes, to: MemoryLayout<Record>.alignment)
        let layerOffset = aligned(batchOffset + MemoryLayout<Record>.stride * Self.batchRecordLimit,
                                  to: MemoryLayout<Layer>.alignment)
        let selectionOffset = aligned(layerOffset + MemoryLayout<Layer>.stride * Self.layerCount,
                                      to: MemoryLayout<UInt64>.alignment)
        let end = selectionOffset + MemoryLayout<UInt64>.stride * Self.layerCount * Self.expertCount
        let pageSize = Int(getpagesize())
        precondition(pageSize > 0 && pageSize % MemoryLayout<Record>.alignment == 0
                     && pageSize % MemoryLayout<Layer>.alignment == 0
                     && pageSize % MemoryLayout<UInt64>.alignment == 0)
        let mappedBytes = aligned(end, to: pageSize)
        precondition(mappedBytes + 4_096 < 4 * 1_024 * 1_024)
        // malloc may grant a larger cached block than requested. A fixed mapping
        // makes the backing-storage bound independent of allocator size classes.
        guard let mapping = mmap(nil, mappedBytes, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANON, -1, 0), mapping != MAP_FAILED else {
            preconditionFailure("Could not allocate runtime measurement storage")
        }
        storage = mapping
        storageLength = mappedBytes
        records = mapping.bindMemory(to: Record.self, capacity: Self.recordCapacity)
        records.initialize(repeating: Record(), count: Self.recordCapacity)
        batch = mapping.advanced(by: batchOffset).bindMemory(to: Record.self,
                                                           capacity: Self.batchRecordLimit)
        batch.initialize(repeating: Record(), count: Self.batchRecordLimit)
        layers = mapping.advanced(by: layerOffset).bindMemory(to: Layer.self, capacity: Self.layerCount)
        layers.initialize(repeating: Layer(), count: Self.layerCount)
        selections = mapping.advanced(by: selectionOffset).bindMemory(to: UInt64.self,
                                                                     capacity: Self.layerCount * Self.expertCount)
        selections.initialize(repeating: 0, count: Self.layerCount * Self.expertCount)
        allocatedStorageBytes = mappedBytes + 4_096
        record(.configuration, UInt64(Self.schemaVersion), UInt64(allocatedStorageBytes), UInt64(Self.recordCapacity),
               UInt64(Self.decodePositionLimit), UInt64(Self.prefillPositionLimit))
    }

    deinit {
        records.deinitialize(count: Self.recordCapacity)
        batch.deinitialize(count: Self.batchRecordLimit)
        layers.deinitialize(count: Self.layerCount)
        selections.deinitialize(count: Self.layerCount * Self.expertCount)
        _ = munmap(storage, storageLength)
    }

    public func record(_ kind: Event, _ a: UInt64 = 0, _ b: UInt64 = 0,
                       _ c: UInt64 = 0, _ d: UInt64 = 0, _ e: UInt64 = 0) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        appendLocked(kind, a, b, c, d, e)
    }

    private func appendLocked(_ kind: Event, _ a: UInt64 = 0, _ b: UInt64 = 0,
                              _ c: UInt64 = 0, _ d: UInt64 = 0, _ e: UInt64 = 0) {
        guard pending < Self.recordCapacity else {
            let ordinal = recorded &+ dropped
            firstDroppedOrdinal = min(firstDroppedOrdinal, ordinal)
            lastDroppedOrdinal = ordinal
            dropped &+= 1
            return
        }
        records[(readIndex + pending) % Self.recordCapacity] = Record(
            kind: kind.rawValue, a: a, b: b, c: c, d: d, e: e)
        pending += 1
        recorded &+= 1
    }

    @discardableResult
    func beginQwenCacheMaps(layerCount: Int, expertCount: Int, slotCount: Int,
                            pairBytes: UInt64) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !finished, !qwenCacheActive,
              (1...64).contains(layerCount), (1...256).contains(expertCount),
              (1...expertCount).contains(slotCount) else { return false }
        qwenCacheActive = true
        qwenCachePhase = .prefill
        qwenCacheInitialLayers = 0
        qwenCacheRecords = 0
        qwenCacheAttempts = 0
        qwenCacheSuccesses = 0
        qwenCacheFailures = 0
        qwenCacheDrops = 0
        appendQwenLocked(.qwenCacheConfiguration, 1, UInt64(layerCount),
                         UInt64(expertCount), UInt64(slotCount), pairBytes)
        appendQwenLocked(.qwenCacheScope, qwenCacheCaptureMode.rawValue,
                         UInt64(Self.maximumQwenCacheRecords))
        return true
    }

    func setQwenCacheMapPhase(_ phase: Phase) {
        lock.lock(); defer { lock.unlock() }
        if qwenCacheActive { qwenCachePhase = phase }
    }

    func endQwenCacheMaps(completed: Bool) {
        lock.lock(); defer { lock.unlock() }
        guard qwenCacheActive else { return }
        // Completion is outside the detail limit. Existing ring-drop/footer
        // counters still detect a completion row that cannot enter the ring.
        if !finished {
            appendLocked(.qwenCacheCompletion, qwenCacheAttempts, qwenCacheSuccesses,
                         qwenCacheFailures, qwenCacheDrops, completed ? 1 : 0)
        }
        qwenCacheActive = false
    }

    func recordQwenCacheInitial(layer: Int, expertCount: Int,
                               snapshot: () -> QwenBF16PairedExpertCache.MeasurementSnapshot) {
        lock.lock(); defer { lock.unlock() }
        guard qwenCacheActive, !finished, capturesQwenPhaseLocked,
              (0..<64).contains(layer) else { return }
        let bit = UInt64(1) << layer
        guard qwenCacheInitialLayers & bit == 0 else { return }
        qwenCacheInitialLayers |= bit
        let value = snapshot()
        appendQwenLocked(.qwenCacheInitialLayer, UInt64(layer), value.policy == .lfu ? 1 : 0,
                         value.clock, UInt64(value.expertIDs.count), UInt64(expertCount))
        for slot in value.expertIDs.indices {
            appendQwenLocked(.qwenCacheInitialSlot, UInt64(layer), UInt64(slot),
                             UInt64(value.expertIDs[slot] + 1), value.useCounts[slot], value.lastUse[slot])
        }
    }

    func beginQwenCacheMap(_ context: QwenCacheMapContext) -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        guard qwenCacheActive, !finished, capturesQwenPhaseLocked else { return 0 }
        guard (0..<64).contains(context.layer), context.position >= 0, context.tokenCount > 0 else {
            qwenCacheDrops &+= 1
            return 0
        }
        qwenCacheAttempts &+= 1
        let identifier = qwenCacheAttempts
        appendQwenLocked(.qwenCacheMap, identifier, qwenCachePhase.rawValue,
                         UInt64(context.layer), UInt64(context.position), UInt64(context.tokenCount))
        return identifier
    }

    func recordQwenCachePlan(_ identifier: UInt64, plan: ExpertCachePlan,
                             clock: UInt64, avoidingSlots: Set<Int>) {
        lock.lock(); defer { lock.unlock() }
        guard qwenCacheActive, identifier != 0 else { return }
        appendQwenLocked(.qwenCachePlan, identifier, clock, UInt64(plan.experts.count),
                         UInt64(plan.hits), UInt64(plan.misses.count))
        appendQwenLocked(.qwenCacheConstraints, identifier, UInt64(avoidingSlots.count))
        for slot in avoidingSlots.sorted() {
            appendQwenLocked(.qwenCacheAvoidingSlot, identifier, UInt64(slot))
        }
        for index in plan.experts.indices {
            appendQwenLocked(.qwenCacheMember, identifier, UInt64(index), UInt64(plan.experts[index]),
                             UInt64(plan.assignedSlots[index] + 1), plan.misses.contains(index) ? 1 : 0)
        }
    }

    func recordQwenCacheUnplanned(_ identifier: UInt64, experts: [Int]) {
        lock.lock(); defer { lock.unlock() }
        guard qwenCacheActive, identifier != 0 else { return }
        for (index, expert) in experts.enumerated() {
            // Invalid request IDs are encoded without trapping measurement.
            appendQwenLocked(.qwenCacheMember, identifier, UInt64(index),
                             UInt64(clamping: expert), 0, 2)
        }
    }

    func recordQwenCacheOutcome(_ identifier: UInt64, status: UInt64, stage: UInt64) {
        lock.lock(); defer { lock.unlock() }
        guard qwenCacheActive, identifier != 0 else { return }
        if status == 0 { qwenCacheSuccesses &+= 1 } else { qwenCacheFailures &+= 1 }
        appendQwenLocked(.qwenCacheOutcome, identifier, status, stage)
    }

    private func appendQwenLocked(_ kind: Event, _ a: UInt64 = 0, _ b: UInt64 = 0,
                                  _ c: UInt64 = 0, _ d: UInt64 = 0, _ e: UInt64 = 0) {
        guard !finished else { qwenCacheDrops &+= 1; return }
        guard qwenCacheRecords < Self.maximumQwenCacheRecords else {
            qwenCacheDrops &+= 1
            return
        }
        qwenCacheRecords += 1
        if pending >= Self.recordCapacity { qwenCacheDrops &+= 1 }
        appendLocked(kind, a, b, c, d, e)
    }

    private var capturesQwenPhaseLocked: Bool {
        qwenCacheCaptureMode == .prefillAndDecode || qwenCachePhase == .decode
    }

    @discardableResult
    func beginQwenProductionTiming() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !finished, qwenCacheActive, !qwenProductionActive else { return false }
        qwenProductionActive = true
        appendLocked(.qwenProductionScope, 3, Phase.decode.rawValue,
                     Self.maximumQwenProductionForwards, Self.maximumQwenProductionRows, 0)
        let uptimeBefore = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let absolute = mach_absolute_time()
        var timebase = mach_timebase_info_data_t()
        let result = mach_timebase_info(&timebase)
        let uptimeAfter = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        appendLocked(.qwenProductionClockCorrelation, uptimeBefore, absolute,
                     result == KERN_SUCCESS ? UInt64(timebase.numer) : 0,
                     result == KERN_SUCCESS ? UInt64(timebase.denom) : 0, uptimeAfter)
        return true
    }

    func endQwenProductionTiming() {
        lock.lock(); defer { lock.unlock() }
        guard qwenProductionActive else { return }
        appendLocked(.qwenProductionCompletion, qwenProductionRows, qwenProductionDrops,
                     qwenProductionCommands, qwenProductionForwards, qwenProductionOmittedForwards)
        qwenProductionActive = false
    }

    func qwenProductionLocation(position: Int, tokenCount: Int, forward: Bool) -> QwenProductionLocation? {
        lock.lock(); defer { lock.unlock() }
        guard qwenProductionActive, !finished, qwenCachePhase == .decode,
              position >= 0, tokenCount >= 0 else { return nil }
        if forward {
            guard qwenProductionForwards < Self.maximumQwenProductionForwards else {
                qwenProductionOmittedForwards &+= 1
                return nil
            }
            qwenProductionForwards &+= 1
        } else if qwenProductionForwards >= Self.maximumQwenProductionForwards {
            return nil
        }
        return QwenProductionLocation(capture: self, position: position, tokenCount: tokenCount, layer: -1)
    }

    private func reserveQwenProductionRowsLocked(_ count: UInt64) -> UInt64? {
        guard qwenProductionActive, !finished else { return nil }
        guard qwenProductionRows + count <= Self.maximumQwenProductionRows,
              pending + Int(count) <= Self.recordCapacity else {
            qwenProductionDrops &+= count
            return nil
        }
        qwenProductionRows += count
        qwenProductionID &+= 1
        return qwenProductionID
    }

    func recordQwenProductionCommand(_ location: QwenProductionLocation, stage: QwenProductionStage,
        submitBefore: UInt64, submitAfter: UInt64, waitBefore: UInt64, resumed: UInt64,
        gpuStartBits: UInt64, gpuEndBits: UInt64, flags: UInt64,
        completionCallback: UInt64? = nil, kernelStartBits: UInt64 = 0,
        kernelEndBits: UInt64 = 0, kernelFlags: UInt64 = 0) {
        lock.lock(); defer { lock.unlock() }
        guard let id = reserveQwenProductionRowsLocked(5) else { return }
        qwenProductionCommands &+= 1
        appendLocked(.qwenProductionCommand, id, stage.rawValue, UInt64(location.layer + 1),
                     UInt64(location.position), UInt64(location.tokenCount))
        appendLocked(.qwenProductionSubmitWait, id, submitBefore, submitAfter, waitBefore, resumed)
        appendLocked(.qwenProductionGPU, id, gpuStartBits, gpuEndBits, flags, Phase.decode.rawValue)
        do {
            let completionCallback = completionCallback ?? 0
            // bit1: callback observed. bit2: ordered for nonnegative interval.
            // bit4: no after-await sample. bit8: callback sample is later than
            // the sampled after-await clock. Zero means explicitly unavailable.
            var observationFlags: UInt64 = completionCallback > 0 ? 1 : 0
            if resumed == 0 { observationFlags |= 4 }
            else if completionCallback > 0 {
                observationFlags |= completionCallback <= resumed ? 2 : 8
            }
            appendLocked(.qwenProductionResumption, id, completionCallback, resumed, observationFlags, 0)
        }
        // Scope 3 emits this row even for unawaited/unavailable clocks.
        // Raw IEEE-754 bits are preserved; consumers must inspect flags.
        appendLocked(.qwenProductionDriver, id, kernelStartBits, kernelEndBits, kernelFlags, 0)
    }

    func recordQwenProductionSpan(_ location: QwenProductionLocation, stage: QwenProductionStage,
        started: UInt64, ended: UInt64, cpuStarted: UInt64, cpuEnded: UInt64) {
        lock.lock(); defer { lock.unlock() }
        guard let id = reserveQwenProductionRowsLocked(2) else { return }
        appendLocked(.qwenProductionSpan, id, stage.rawValue, UInt64(location.layer + 1),
                     UInt64(location.position), UInt64(location.tokenCount))
        appendLocked(.qwenProductionHostClocks, id, started, ended, cpuStarted, cpuEnded)
    }

    /// Bounded per-phase aggregates only. Qwen never enters the Gemma-specific
    /// layer/expert arrays. Worker wall sums overlap; these are not turn latency.
    func recordSourceIO(_ snapshot: OfficialSourceIOMeasurement.Snapshot) {
        for phase in 0..<2 {
            // Use the existing public phase IDs: prefill2,decode1.
            let phaseID: UInt64 = phase == 0 ? Phase.prefill.rawValue : Phase.decode.rawValue
            for byteClass in 0..<2 {
                let readIndex = phase * 2 + byteClass
                for site in 0..<4 {
                    let value = snapshot.validations[readIndex * 4 + site]
                    // phase, byteClass*4+site, count, worker-wall ns, thread-CPU ns
                    let code = UInt64(byteClass * 4 + site)
                    record(.sourceIOValidation, phaseID, code, value.count,
                           value.wallNanoseconds, value.threadCPUNanoseconds)
                    // phase, site code, thrown validations, unavailable CPU intervals
                    record(.sourceIOValidationOutcomes, phaseID, code, value.errors, value.unavailableCPU)
                }
                let pread = snapshot.preads[readIndex]
                record(.sourceIOPread, phaseID, UInt64(byteClass), pread.count,
                       pread.wallNanoseconds, pread.threadCPUNanoseconds)
                // Bytes are valid completed bytes, not requested bytes. EOF and
                // impossible byte counts are errors; EINTR is an interruption.
                record(.sourceIOPreadOutcomes, phaseID, UInt64(byteClass), pread.bytes,
                       pread.errors, pread.interruptions)
                record(.sourceIOReads, phaseID, UInt64(byteClass), snapshot.reads[readIndex],
                       snapshot.failedReads[readIndex], pread.unavailableCPU)
            }
        }
    }

    /// Returns true only on a window boundary, so the caller can snapshot the
    /// existing cache before planning. No expert-array slices or copies are made.
    public func recordDecodeRoutes(layer: Int, position: Int, expertIDs: [Int]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !finished, (0..<Self.layerCount).contains(layer), position >= 0 else { return false }
        let boundary = updateWindowLocked(.decode, position: position)
        let detailed = inWindowLocked(.decode, position: position)
        setContextLocked(layer: layer, phase: .decode, position: position, detailed: detailed)
        recordRouteLocked(layer: layer, phase: .decode, position: position, detailed: detailed,
                          count: expertIDs.count) { expertIDs[$0] }
        return boundary
    }

    public func recordPrefillRoutes(layer: Int, startPosition: Int, tokenCount: Int,
                                    topK: Int, expertIDs: [UInt32]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !finished, (0..<Self.layerCount).contains(layer), startPosition >= 0,
              tokenCount >= 0, topK > 0, tokenCount <= expertIDs.count / topK else { return false }
        let boundary = updateWindowLocked(.prefill, position: startPosition)
        // Image chunks may cross the configured route window. Their exact grouped
        // plans still cover the whole chunk (prefillChunk records its token count).
        // A retained route prefix alone is insufficient for policy reconstruction.
        setContextLocked(layer: layer, phase: .prefill, position: startPosition,
                         detailed: inWindowLocked(.prefill, position: startPosition))
        for token in 0..<tokenCount {
            let position = startPosition + token
            recordRouteLocked(layer: layer, phase: .prefill, position: position,
                              detailed: inWindowLocked(.prefill, position: position), count: topK) {
                Int(expertIDs[token * topK + $0])
            }
        }
        return boundary
    }

    private func updateWindowLocked(_ phase: Phase, position: Int) -> Bool {
        let start = phase == .decode ? decodeStart : prefillStart
        if start == nil {
            if phase == .decode { decodeStart = position } else { prefillStart = position }
            // phase, open=0, first position, exclusive end, timestamp
            let limit = phase == .decode ? Self.decodePositionLimit : Self.prefillPositionLimit
            appendLocked(.window, phase.rawValue, 0, UInt64(position), UInt64(position + limit),
                         DispatchTime.now().uptimeNanoseconds)
            return true
        }
        let closed = phase == .decode ? decodeClosed : prefillClosed
        if !closed && !inWindowLocked(phase, position: position) {
            if phase == .decode { decodeClosed = true } else { prefillClosed = true }
            appendLocked(.window, phase.rawValue, 1, UInt64(start!), UInt64(position),
                         DispatchTime.now().uptimeNanoseconds)
            return true
        }
        return false
    }

    private func inWindowLocked(_ phase: Phase, position: Int) -> Bool {
        guard let start = phase == .decode ? decodeStart : prefillStart else { return false }
        let limit = phase == .decode ? Self.decodePositionLimit : Self.prefillPositionLimit
        return position >= start && position - start < limit
    }

    private func setContextLocked(layer: Int, phase: Phase, position: Int, detailed: Bool) {
        layers[layer].phase = phase.rawValue
        layers[layer].position = UInt64(position)
        layers[layer].detailed = detailed
    }

    private func recordRouteLocked(layer: Int, phase: Phase, position: Int, detailed: Bool,
                                   count: Int, expertAt: (Int) -> Int) {
        if phase == .decode {
            layers[layer].decodeRoutes &+= 1
            layers[layer].firstDecodePosition = min(layers[layer].firstDecodePosition, UInt64(position))
            layers[layer].lastDecodePosition = UInt64(position)
        } else {
            layers[layer].prefillRoutes &+= 1
            layers[layer].firstPrefillPosition = min(layers[layer].firstPrefillPosition, UInt64(position))
            layers[layer].lastPrefillPosition = UInt64(position)
        }
        if !detailed {
            if phase == .decode {
                layers[layer].omittedDecode &+= 1
                layers[layer].firstOmittedDecode = min(layers[layer].firstOmittedDecode, UInt64(position))
                layers[layer].lastOmittedDecode = UInt64(position)
            } else {
                layers[layer].omittedPrefill &+= 1
                layers[layer].firstOmittedPrefill = min(layers[layer].firstOmittedPrefill, UInt64(position))
                layers[layer].lastOmittedPrefill = UInt64(position)
            }
        }
        var packed: UInt64 = 0
        var complete = count == 8
        for index in 0..<count {
            let expert = expertAt(index)
            guard (0..<Self.expertCount).contains(expert) else { complete = false; continue }
            selections[layer * Self.expertCount + expert] &+= 1
            if index < 8 { packed |= UInt64(expert) << (index * 8) }
        }
        if detailed {
            // layer | phase<<32, position, topK, eight UInt8 IDs, incomplete flag.
            appendLocked(.route, UInt64(layer) | phase.rawValue << 32, UInt64(position),
                         UInt64(count), packed, complete ? 0 : 1)
        }
    }

    /// Called exactly once for the already-produced cache plan. Misses are indices
    /// into its ordered expert list, not expert IDs. Zero means no detailed record.
    func recordPlan(layer: Int, plan: ExpertCachePlan) -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        guard !finished, (0..<Self.layerCount).contains(layer) else { return 0 }
        layers[layer].plans &+= 1
        layers[layer].hits &+= UInt64(plan.hits)
        layers[layer].misses &+= UInt64(plan.misses.count)
        guard layers[layer].detailed else { return 0 }
        nextPlanID &+= 1
        let id = nextPlanID
        var missMask: UInt64 = 0
        for index in plan.misses where index < 64 { missMask |= 1 << index }
        // planID, layer|phase<<32, position, count|hits<<32, miss-index mask
        appendLocked(.plan, id, UInt64(layer) | layers[layer].phase << 32,
                     layers[layer].position, UInt64(plan.experts.count) | UInt64(plan.hits) << 32, missMask)
        // Up to four exact UInt16 expert/slot pairs per row. The active capture
        // profile is 128 experts/32 slots. Larger values are marked unknown.
        for start in stride(from: 0, to: plan.experts.count, by: 4) {
            var low: UInt64 = 0
            var high: UInt64 = 0
            for offset in 0..<min(4, plan.experts.count - start) {
                let index = start + offset
                let expert = UInt64(UInt16(exactly: plan.experts[index]) ?? .max)
                let slot = UInt64(UInt16(exactly: plan.assignedSlots[index]) ?? .max)
                let pair = expert | slot << 16
                if offset < 2 { low |= pair << (offset * 32) }
                else { high |= pair << ((offset - 2) * 32) }
            }
            appendLocked(.planMembers, id, UInt64(start), low, high,
                         UInt64(min(4, plan.experts.count - start)))
        }
        return id
    }

    func recordSuccessfulMiss(layer: Int, planID: UInt64, expert: Int, slot: Int, bytes: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished, (0..<Self.layerCount).contains(layer) else { return }
        layers[layer].successfulMisses &+= 1
        layers[layer].logicalBytes &+= bytes
        if planID != 0 {
            appendLocked(.successfulMiss, planID, UInt64(layer), UInt64(expert), UInt64(slot), bytes)
        }
    }

    func recordFetch(layer: Int, planID: UInt64, allHit: Bool, enqueued: UInt64,
                     entered: UInt64, completed: UInt64, resumed: UInt64, failed: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished, (0..<Self.layerCount).contains(layer) else { return }
        let queue = entered >= enqueued ? entered - enqueued : 0
        let work = completed >= entered ? completed - entered : 0
        let resume = resumed >= completed ? resumed - completed : 0
        layers[layer].fetches &+= 1
        layers[layer].queueNanos &+= queue
        layers[layer].workNanos &+= work
        layers[layer].resumeNanos &+= resume
        if failed { layers[layer].failedFetches &+= 1 }
        if allHit {
            layers[layer].allHitFetches &+= 1
            layers[layer].allHitQueueNanos &+= queue
            layers[layer].allHitWorkNanos &+= work
            layers[layer].allHitResumeNanos &+= resume
        }
        if planID != 0 {
            appendLocked(.fetchQueue, planID, enqueued, entered, allHit ? 1 : 0, UInt64(layer))
            appendLocked(failed ? .fetchFailed : .fetchCompleted, planID, completed, resumed, work, resume)
        }
    }

    public var needsPostImagePrefillSnapshot: Bool {
        lock.lock()
        defer { lock.unlock() }
        return postImagePending
    }

    public func beginFreshImage() {
        lock.lock()
        defer { lock.unlock() }
        if !postImagePending { preReleaseCaptured = false }
        postImagePending = true
    }

    func shouldCapturePreRelease(openLayers: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard openLayers > 0, !preReleaseCaptured else { return false }
        preReleaseCaptured = true
        return true
    }

    public func finishPostImagePrefill() {
        lock.lock()
        defer { lock.unlock() }
        postImagePending = false
        preReleaseCaptured = false
    }

    /// Synchronously reads the streamer's arrays under its existing lock. It does
    /// not retain those arrays, and never mutates the LFU/recency state.
    func recordCacheLayer(layer: Int, reason: UInt64, slots: [Int], lastUse: [Int],
                          frequencies: [Int], useClock: Int, scratchBytes: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        appendLocked(.cacheLayer, UInt64(layer), reason, UInt64(slots.count),
                     UInt64(bitPattern: Int64(useClock)), scratchBytes)
        for index in slots.indices {
            appendLocked(.cacheSlot, UInt64(layer), reason, UInt64(index),
                         UInt64(bitPattern: Int64(slots[index])), UInt64(bitPattern: Int64(lastUse[index])))
        }
        for index in frequencies.indices {
            appendLocked(.cacheFrequency, UInt64(layer), reason, UInt64(index),
                         UInt64(bitPattern: Int64(frequencies[index])), 0)
        }
    }

    public func recordMemorySample(currentBytes: UInt64?, peakBytes: UInt64?, towerBytes: UInt64?) {
        let presence: UInt64 = (currentBytes == nil ? 0 : 1)
            | (peakBytes == nil ? 0 : 2) | (towerBytes == nil ? 0 : 4)
        record(.memory, DispatchTime.now().uptimeNanoseconds, currentBytes ?? 0,
               peakBytes ?? 0, towerBytes ?? 0, presence)
    }

    /// The service calls this only after the generation stream has ended. The
    /// remaining ring and a fixed aggregate footer are then drained before terminal.
    public func finish(status: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        finished = true
        finishStatus = status
    }

    /// Bounded serialization happens on the service writer, never on a forward
    /// path. Only one 12 KiB fixed scratch batch and one <=48 KiB Data exist here.
    /// Each row is <=128 bytes even with six UInt64.max values. No record arrays grow.
    public func drainJSONBatch(maximumBytes: Int) -> DrainedBatch? {
        let budget = min(maximumBytes, Self.maximumJSONBatchBytes)
        guard budget >= 256 else { return nil }
        drainLock.lock()
        defer { drainLock.unlock() }
        let started = DispatchTime.now().uptimeNanoseconds
        let limit = min(Self.batchRecordLimit,
                        (budget - Self.maximumBatchEnvelopeBytes) / Self.maximumSerializedRowBytes)
        lock.lock()
        var count = 0
        var containsFooter = false
        while count < limit && pending > 0 {
            batch[count] = records[readIndex]
            readIndex = (readIndex + 1) % Self.recordCapacity
            pending -= 1
            count += 1
        }
        if finished && pending == 0 {
            while count < limit, let footer = nextFooterLocked() {
                batch[count] = footer
                containsFooter = true
                count += 1
            }
        }
        lock.unlock()
        guard count > 0 else { return nil }
        var data = Data()
        data.reserveCapacity(budget)
        data.append(contentsOf: "{\"version\":\(Self.schemaVersion),\"records\":[".utf8)
        for index in 0..<count {
            let row = batch[index]
            if index > 0 { data.append(44) }
            let text = "[\(row.kind),\(row.a),\(row.b),\(row.c),\(row.d),\(row.e)]"
            data.append(contentsOf: text.utf8)
        }
        data.append(contentsOf: "]}".utf8)
        precondition(data.count <= budget)
        // The footer reports completed earlier batches. Its own final serialization
        // is excluded explicitly, avoiding a recursive measurement record.
        serializationNanos &+= DispatchTime.now().uptimeNanoseconds - started
        drainedBatches &+= 1
        drainedBytes &+= UInt64(data.count)
        return DrainedBatch(data: data, containsFooter: containsFooter)
    }

    private func nextFooterLocked() -> Record? {
        // Fixed layer summaries and all 128 selection totals per layer.
        // These totals survive route-window closure and ring overflow.
        let layerRows = Self.footerLayerRows
        let expertRows = Self.footerExpertRows
        let index = footerIndex
        guard index < Self.footerRecordCount else { return nil }
        footerIndex += 1
        if index == 0 {
            // status, accepted rows, dropped rows, backing+object budget, pending image
            return Record(kind: Event.summary.rawValue, a: finishStatus, b: recorded,
                          c: dropped, d: UInt64(allocatedStorageBytes), e: postImagePending ? 1 : 0)
        }
        if index <= layerRows {
            let layer = (index - 1) / 9
            let subtype = (index - 1) % 9
            let state = layers[layer]
            let identity = UInt64(layer) | UInt64(subtype) << 32
            switch subtype {
            case 0: return Record(kind: Event.layerSummary.rawValue, a: identity,
                                  b: state.plans, c: state.hits, d: state.misses, e: state.failedFetches)
            case 1: return Record(kind: Event.layerSummary.rawValue, a: identity,
                                  b: state.successfulMisses, c: state.logicalBytes,
                                  d: state.decodeRoutes, e: state.prefillRoutes)
            case 2: return Record(kind: Event.omittedRoutes.rawValue, a: identity,
                                  b: Phase.decode.rawValue, c: state.omittedDecode,
                                  d: state.firstOmittedDecode, e: state.lastOmittedDecode)
            case 3: return Record(kind: Event.omittedRoutes.rawValue, a: identity,
                                  b: Phase.prefill.rawValue, c: state.omittedPrefill,
                                  d: state.firstOmittedPrefill, e: state.lastOmittedPrefill)
            case 4: return Record(kind: Event.schedulingSummary.rawValue, a: identity,
                                  b: state.fetches, c: state.queueNanos, d: state.workNanos, e: state.resumeNanos)
            case 5: return Record(kind: Event.schedulingSummary.rawValue, a: identity,
                                  b: state.allHitFetches, c: state.allHitQueueNanos,
                                  d: state.allHitWorkNanos, e: state.allHitResumeNanos)
            case 6: return Record(kind: Event.layerSummary.rawValue, a: identity,
                                  b: state.phase, c: state.position, d: state.detailed ? 1 : 0, e: 0)
            case 7: return Record(kind: Event.observedCoverage.rawValue, a: UInt64(layer),
                                  b: Phase.decode.rawValue, c: state.firstDecodePosition,
                                  d: state.lastDecodePosition, e: state.decodeRoutes)
            default: return Record(kind: Event.observedCoverage.rawValue, a: UInt64(layer),
                                   b: Phase.prefill.rawValue, c: state.firstPrefillPosition,
                                   d: state.lastPrefillPosition, e: state.prefillRoutes)
            }
        }
        if index <= layerRows + expertRows {
            let expertIndex = index - layerRows - 1
            return Record(kind: Event.expertSummary.rawValue,
                          a: UInt64(expertIndex / Self.expertCount), b: UInt64(expertIndex % Self.expertCount),
                          c: selections[expertIndex], d: 0, e: 0)
        }
        if index == 1 + layerRows + expertRows {
            // A bounding range of attempted row ordinals, not a claim that every
            // ordinal inside it was dropped. Any drop invalidates exact replay.
            return Record(kind: Event.droppedRange.rawValue, a: dropped,
                          b: firstDroppedOrdinal, c: lastDroppedOrdinal, d: 0, e: 0)
        }
        return Record(kind: Event.serializationSummary.rawValue,
                      a: drainedBatches, b: drainedBytes, c: serializationNanos,
                      d: UInt64(MemoryLayout<Record>.stride), e: 1)
    }
}
