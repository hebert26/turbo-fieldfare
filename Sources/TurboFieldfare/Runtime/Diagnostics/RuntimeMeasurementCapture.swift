import Darwin
import Foundation

/// Request-owned numeric measurement. It never retains model, image or GPU resources.
/// Callers must test their optional capture before constructing payloads or reading clocks.
/// Storage is allocated once. Filling the ring drops records rather than growing it.
public final class RuntimeMeasurementCapture: @unchecked Sendable {
    public enum Phase: UInt64, Sendable { case decode = 1, prefill = 2 }

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
    private let records: UnsafeMutablePointer<Record>
    private let batch: UnsafeMutablePointer<Record>
    private let layers: UnsafeMutablePointer<Layer>
    private let selections: UnsafeMutablePointer<UInt64>
    /// Allocator-reported bytes for every fixed backing allocation. Object/locks are
    /// separately budgeted with a conservative 4 KiB allowance in this total.
    public let allocatedStorageBytes: Int
    private var readIndex = 0
    private var pending = 0
    private var recorded: UInt64 = 0
    private var dropped: UInt64 = 0
    private var firstDroppedOrdinal: UInt64 = .max
    private var lastDroppedOrdinal: UInt64 = .max
    private var nextPlanID: UInt64 = 0
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

    public init() {
        precondition(MemoryLayout<Record>.stride == 48)
        records = .allocate(capacity: Self.recordCapacity)
        records.initialize(repeating: Record(), count: Self.recordCapacity)
        batch = .allocate(capacity: Self.batchRecordLimit)
        batch.initialize(repeating: Record(), count: Self.batchRecordLimit)
        layers = .allocate(capacity: Self.layerCount)
        layers.initialize(repeating: Layer(), count: Self.layerCount)
        selections = .allocate(capacity: Self.layerCount * Self.expertCount)
        selections.initialize(repeating: 0, count: Self.layerCount * Self.expertCount)
        allocatedStorageBytes = malloc_size(records) + malloc_size(batch)
            + malloc_size(layers) + malloc_size(selections) + 4_096
        precondition(allocatedStorageBytes < 4 * 1_024 * 1_024)
        record(.configuration, UInt64(Self.schemaVersion), UInt64(allocatedStorageBytes), UInt64(Self.recordCapacity),
               UInt64(Self.decodePositionLimit), UInt64(Self.prefillPositionLimit))
    }

    deinit {
        records.deinitialize(count: Self.recordCapacity)
        records.deallocate()
        batch.deinitialize(count: Self.batchRecordLimit)
        batch.deallocate()
        layers.deinitialize(count: Self.layerCount)
        layers.deallocate()
        selections.deinitialize(count: Self.layerCount * Self.expertCount)
        selections.deallocate()
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
