import Foundation
import Metal
import TurboFieldfareFormat

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan
    /// nil means the plan was created without capture. Zero means its exact
    /// details were outside the bounded window, while aggregate counts remain.
    public let measurementPlanID: UInt64?

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan, measurementPlanID: UInt64? = nil) {
        self.layer = layer
        self.cachePlan = cachePlan
        self.measurementPlanID = measurementPlanID
    }
}

enum QwenExpertMappingError: Error, Equatable, Sendable {
    case invalidLayer(Int)
    case missingAffineDescriptor(String)
    case invalidExpert(Int)
    case duplicateExpertWithinToken(Int)
    case invalidSlotCount(Int)
    case layerFileNotRegular(String)
    case layerFileSizeMismatch(expected: UInt64, actual: UInt64)
    case insufficientUnpinnedSlots
    case unknownLease
    case leaseAlreadySubmitted
    case commandBufferAlreadySubmitted
    case commandBufferUnavailable
    case canceled
}

struct QwenMappedExpert: @unchecked Sendable {
    let expertID: Int
    let slot: Int
    let buffer: MTLBuffer
    let length: UInt64
    let gateUp: PackedAffineDescriptor
    let down: PackedAffineDescriptor
}

struct QwenExpertMappingDiagnostics: Sendable, Equatable {
    let requestedExpertIDs: [Int]
    let assignedSlots: [Int]
    let hits: Int
    let misses: Int
}

struct QwenExpertLeaseSnapshot: Sendable, Equatable {
    let submitted: Bool
    let canceled: Bool
    let completed: Bool
    let succeeded: Bool?
}

/// Owns the Qwen-specific safety layer above `PreadExpertStreamer`. All Qwen
/// cache planning for this layer must pass through this coordinator so a slot
/// cannot be overwritten while a submitted command buffer still reads it.
final class QwenExpertMappingCoordinator: @unchecked Sendable {
    private struct ActiveSlot {
        var expertID: Int
        var referenceCount: Int
    }

    private struct Reservation {
        let slots: [Int]
        var submitted: Bool
        var canceled: Bool
    }

    private struct FetchResult {
        let identifier: UInt64
        let plan: ExpertCachePlan
        let buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)]
    }

    let layer: LayerLayout
    let expertStride: UInt64
    let slotCount: Int

    private let streamer: PreadExpertStreamer
    private let stateLock = NSLock()
    /// Serializes plan+pread publication. This is a blocking file-I/O lock used
    /// only on the dedicated dispatch worker, never on a cooperative executor.
    private let ioLock = NSLock()
    private var activeSlots: [Int: ActiveSlot] = [:]
    private var reservations: [UInt64: Reservation] = [:]
    private var nextIdentifier: UInt64 = 1

    /// Bytes of this layer's actual routed-expert cache buffers.
    var allocatedCacheBytes: UInt64 { streamer.allocatedCacheBytes }

    init(directoryURL: URL,
         layout: PackedExpertsLayout,
         layer layerIndex: Int,
         device: MTLDevice,
         slotCount: Int,
         cachePolicy: ExpertCachePolicy = .lfu) throws {
        guard slotCount > 0 else { throw QwenExpertMappingError.invalidSlotCount(slotCount) }
        guard layerIndex >= 0, layerIndex < layout.layers.count,
              layout.layers[layerIndex].layer == layerIndex else {
            throw QwenExpertMappingError.invalidLayer(layerIndex)
        }
        let layer = layout.layers[layerIndex]
        guard layer.experts.count == layout.expertsPerLayer,
              layout.expertStride > 0 else {
            throw QwenExpertMappingError.invalidLayer(layerIndex)
        }
        for (expertID, expert) in layer.experts.enumerated() {
            let expectedOffset = try Self.checkedMultiply(
                UInt64(expertID), layout.expertStride)
            guard expert.expert == expertID, expert.physicalRank == expertID,
                  expert.offset == expectedOffset,
                  expert.size == layout.expertStride else {
                throw QwenExpertMappingError.invalidExpert(expertID)
            }
        }
        guard layer.affineDescriptors["gate_up"] != nil else {
            throw QwenExpertMappingError.missingAffineDescriptor("gate_up")
        }
        guard layer.affineDescriptors["down"] != nil else {
            throw QwenExpertMappingError.missingAffineDescriptor("down")
        }
        let fileURL = directoryURL
            .appendingPathComponent("packed_experts", isDirectory: true)
            .appendingPathComponent(layer.file, isDirectory: false)
        let values = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true else {
            throw QwenExpertMappingError.layerFileNotRegular(fileURL.path)
        }
        let expectedSize = try Self.checkedMultiply(
            UInt64(layout.expertsPerLayer), layout.expertStride)
        guard let fileSize = values.fileSize, fileSize >= 0 else {
            throw QwenExpertMappingError.layerFileNotRegular(fileURL.path)
        }
        let actualSize = UInt64(fileSize)
        guard actualSize == expectedSize else {
            throw QwenExpertMappingError.layerFileSizeMismatch(
                expected: expectedSize, actual: actualSize)
        }
        let streamLayout = StreamLayout(
            path: fileURL.path,
            streamOffset: 0,
            streamSize: expectedSize,
            expertsPerLayer: layout.expertsPerLayer,
            expertStride: layout.expertStride,
            expertOffsets: layer.experts.map(\.offset))
        self.layer = layer
        self.expertStride = layout.expertStride
        self.slotCount = slotCount
        self.streamer = try PreadExpertStreamer(
            layout: streamLayout, device: device, slotCount: slotCount,
            cachePolicy: cachePolicy)
    }

    /// Maps one token's unique selected experts. Different tokens and
    /// submissions may select the same expert; those uses share/refcount a hit.
    func map(expertIDs: [Int]) async throws -> QwenMappedExpertLease {
        guard expertIDs.count <= slotCount else {
            throw QwenExpertMappingError.insufficientUnpinnedSlots
        }
        var seen = Set<Int>()
        for expert in expertIDs {
            guard expert >= 0, expert < layer.experts.count else {
                throw QwenExpertMappingError.invalidExpert(expert)
            }
            guard seen.insert(expert).inserted else {
                throw QwenExpertMappingError.duplicateExpertWithinToken(expert)
            }
        }
        let result: Result<FetchResult, Error> = await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                continuation.resume(returning: Result { try performFetch(expertIDs: expertIDs) })
            }
        }
        let fetched = try result.get()
        if Task.isCancelled {
            releaseUnsubmitted(identifier: fetched.identifier)
            throw CancellationError()
        }
        guard let gateUp = layer.affineDescriptors["gate_up"],
              let down = layer.affineDescriptors["down"] else {
            releaseUnsubmitted(identifier: fetched.identifier)
            throw QwenExpertMappingError.missingAffineDescriptor("gate_up/down")
        }
        let mapped = zip(fetched.plan.experts.indices, fetched.buffers).map { index, entry in
            QwenMappedExpert(
                expertID: fetched.plan.experts[index],
                slot: fetched.plan.assignedSlots[index],
                buffer: entry.buffer,
                length: entry.size,
                gateUp: gateUp,
                down: down)
        }
        return QwenMappedExpertLease(
            coordinator: self,
            identifier: fetched.identifier,
            experts: mapped,
            diagnostics: QwenExpertMappingDiagnostics(
                requestedExpertIDs: fetched.plan.experts,
                assignedSlots: fetched.plan.assignedSlots,
                hits: fetched.plan.hits,
                misses: fetched.plan.misses.count))
    }

    private func performFetch(expertIDs: [Int]) throws -> FetchResult {
        ioLock.lock()
        defer { ioLock.unlock() }

        let reserved: (UInt64, ExpertCachePlan) = try withStateLock {
            let avoiding = Set(activeSlots.keys)
            guard let plan = streamer.planExpertsCachedIfPossible(
                experts: expertIDs, avoidingSlots: avoiding) else {
                throw QwenExpertMappingError.insufficientUnpinnedSlots
            }
            // A hit on an active slot is valid only for the same expert. Misses
            // cannot use active slots because they were passed as avoiding.
            for index in plan.experts.indices {
                let slot = plan.assignedSlots[index]
                if let active = activeSlots[slot], active.expertID != plan.experts[index] {
                    throw QwenExpertMappingError.insufficientUnpinnedSlots
                }
            }
            let identifier = allocateIdentifierLocked()
            for index in plan.experts.indices {
                let slot = plan.assignedSlots[index]
                if var active = activeSlots[slot] {
                    active.referenceCount += 1
                    activeSlots[slot] = active
                } else {
                    activeSlots[slot] = ActiveSlot(
                        expertID: plan.experts[index], referenceCount: 1)
                }
            }
            reservations[identifier] = Reservation(
                slots: plan.assignedSlots, submitted: false, canceled: false)
            return (identifier, plan)
        }
        do {
            let buffers = try streamer.executeExpertCachePlan(reserved.1)
            return FetchResult(identifier: reserved.0, plan: reserved.1, buffers: buffers)
        } catch {
            releaseUnsubmitted(identifier: reserved.0)
            throw error
        }
    }

    fileprivate func trackLastUse(identifier: UInt64, on commandBuffer: MTLCommandBuffer) throws {
        try withStateLock {
            guard commandBuffer.status == .notEnqueued else {
                throw QwenExpertMappingError.commandBufferAlreadySubmitted
            }
            guard var reservation = reservations[identifier] else {
                throw QwenExpertMappingError.unknownLease
            }
            guard !reservation.canceled else { throw QwenExpertMappingError.canceled }
            guard !reservation.submitted else {
                throw QwenExpertMappingError.leaseAlreadySubmitted
            }
            reservation.submitted = true
            reservations[identifier] = reservation
        }
    }

    fileprivate func cancel(identifier: UInt64) throws -> Bool {
        try withStateLock {
            guard var reservation = reservations[identifier] else {
                throw QwenExpertMappingError.unknownLease
            }
            if reservation.submitted {
                reservation.canceled = true
                reservations[identifier] = reservation
                return false
            }
            releaseLocked(identifier: identifier)
            return true
        }
    }

    fileprivate func complete(identifier: UInt64) -> Bool {
        withStateLock {
            guard let reservation = reservations[identifier] else { return true }
            let discarded = reservation.canceled
            releaseLocked(identifier: identifier)
            return discarded
        }
    }

    fileprivate func isUsable(identifier: UInt64) -> Bool {
        withStateLock {
            guard let reservation = reservations[identifier] else { return false }
            return !reservation.canceled
        }
    }

    fileprivate func releaseUnsubmitted(identifier: UInt64) {
        withStateLock {
            guard let reservation = reservations[identifier], !reservation.submitted else { return }
            releaseLocked(identifier: identifier)
        }
    }

    private func releaseLocked(identifier: UInt64) {
        guard let reservation = reservations.removeValue(forKey: identifier) else { return }
        for slot in reservation.slots {
            guard var active = activeSlots[slot] else { continue }
            active.referenceCount -= 1
            if active.referenceCount == 0 {
                activeSlots[slot] = nil
            } else {
                activeSlots[slot] = active
            }
        }
    }

    private func allocateIdentifierLocked() -> UInt64 {
        let identifier = nextIdentifier
        nextIdentifier &+= 1
        if nextIdentifier == 0 { nextIdentifier = 1 }
        return identifier
    }

    private func withStateLock<T>(_ body: () throws -> T) rethrows -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return try body()
    }

    private static func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw QwenExpertMappingError.layerFileSizeMismatch(
                expected: UInt64.max, actual: 0)
        }
        return result
    }
}

final class QwenMappedExpertLease: @unchecked Sendable {
    let experts: [QwenMappedExpert]
    let diagnostics: QwenExpertMappingDiagnostics

    private let coordinator: QwenExpertMappingCoordinator
    private let identifier: UInt64
    private let lock = NSLock()
    private var submitted = false
    private var canceled = false
    private var completed = false
    private var succeeded: Bool?

    fileprivate init(
        coordinator: QwenExpertMappingCoordinator,
        identifier: UInt64,
        experts: [QwenMappedExpert],
        diagnostics: QwenExpertMappingDiagnostics
    ) {
        self.coordinator = coordinator
        self.identifier = identifier
        self.experts = experts
        self.diagnostics = diagnostics
    }

    deinit {
        coordinator.releaseUnsubmitted(identifier: identifier)
    }

    func requireUsable() throws {
        guard coordinator.isUsable(identifier: identifier) else {
            throw QwenExpertMappingError.canceled
        }
    }

    /// Creates, completion-owns, encodes, and commits one GPU use as an
    /// inseparable operation. Validation must happen before calling this API.
    /// Once ownership transfers, both success and failure paths commit so no
    /// encoded command buffer can outlive or leak its pinned cache admission.
    @discardableResult
    func submit(
        on queue: MTLCommandQueue,
        encode: (MTLCommandBuffer) throws -> Void
    ) throws -> MTLCommandBuffer {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw QwenExpertMappingError.commandBufferUnavailable
        }
        try coordinator.trackLastUse(identifier: identifier, on: commandBuffer)
        lock.lock()
        submitted = true
        lock.unlock()
        commandBuffer.addCompletedHandler { [self] commandBuffer in
            let discarded = coordinator.complete(identifier: identifier)
            lock.lock()
            canceled = canceled || discarded
            completed = true
            succeeded = !canceled && commandBuffer.status == .completed
                && commandBuffer.error == nil
            lock.unlock()
        }
        do {
            try encode(commandBuffer)
            commandBuffer.commit()
            return commandBuffer
        } catch {
            try? cancel()
            if commandBuffer.status == .notEnqueued {
                commandBuffer.commit()
            }
            throw error
        }
    }

    /// Before submission this releases immediately. After submission the lease
    /// remains pinned and only the eventual result is discarded.
    func cancel() throws {
        let released = try coordinator.cancel(identifier: identifier)
        lock.lock()
        canceled = true
        if released { completed = true; succeeded = false }
        lock.unlock()
    }

    func snapshot() -> QwenExpertLeaseSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return QwenExpertLeaseSnapshot(
            submitted: submitted, canceled: canceled,
            completed: completed, succeeded: succeeded)
    }
}

extension Model {
    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            guard let tensor = expert.subTensors[role],
                  let offset = UInt32(exactly: tensor.offset) else {
                preconditionFailure("invalid routed expert metadata for role \(role)")
            }
            return offset
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        let cachePlan = streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots)
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan,
                                     measurementPlanID: measurementCapture?.recordPlan(layer: layer, plan: cachePlan))
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan,
                                     measurementPlanID: measurementCapture?.recordPlan(layer: layer, plan: cachePlan))
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        if let capture = measurementCapture {
            return try await Self.fetchMeasuredExperts(streamer: streamer, layer: plan.layer,
                                                       existingPlan: plan, experts: plan.experts,
                                                       capture: capture)
        }
        if plan.cachePlan.misses.isEmpty {
            return Self.makeExpertViews(
                streamer.expertCachePlanBuffers(plan.cachePlan),
                layer: plan.layer,
                experts: plan.experts)
        }
        // Cache misses keep the existing dispatch, continuation and buffer views.
        // No measurement object, timestamp or cache snapshot is constructed.
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: plan.layer,
                        experts: plan.experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        if let capture = measurementCapture {
            return try await Self.fetchMeasuredExperts(streamer: streamer, layer: layer,
                                                       existingPlan: nil, experts: experts,
                                                       capture: capture)
        }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.loadExpertsCached(experts: experts)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private struct MeasuredExpertFetch: @unchecked Sendable {
        let result: Result<[TensorView], Error>
        let planID: UInt64
        let allHit: Bool
        let enqueued: UInt64
        let entered: UInt64
        let completed: UInt64
    }

    /// Keeps the existing global-queue boundary, including all-hit plans. Timing
    /// distinguishes queue delay, closure work and continuation resume delay.
    private static func fetchMeasuredExperts(
        streamer: PreadExpertStreamer, layer: Int, existingPlan: RoutedExpertFetchPlan?,
        experts: [Int], capture: RuntimeMeasurementCapture
    ) async throws -> [TensorView] {
        let fetched: MeasuredExpertFetch = await withCheckedContinuation { continuation in
            let enqueued = DispatchTime.now().uptimeNanoseconds
            DispatchQueue.global(qos: .userInitiated).async {
                let entered = DispatchTime.now().uptimeNanoseconds
                // The unplanned path still plans inside this same worker, exactly
                // where loadExpertsCached did. Never run the stateful planner twice.
                let cachePlan = existingPlan?.cachePlan ?? streamer.planExpertsCached(experts: experts)
                let planID = existingPlan?.measurementPlanID
                    ?? capture.recordPlan(layer: layer, plan: cachePlan)
                let result: Result<[TensorView], Error>
                do {
                    let buffers = try streamer.executeExpertCachePlan(
                        cachePlan, measurement: capture, measurementPlanID: planID,
                        measurementLayer: layer)
                    result = .success(Self.makeExpertViews(buffers, layer: layer, experts: experts))
                } catch {
                    result = .failure(error)
                }
                let completed = DispatchTime.now().uptimeNanoseconds
                continuation.resume(returning: MeasuredExpertFetch(
                    result: result, planID: planID, allHit: cachePlan.misses.isEmpty,
                    enqueued: enqueued, entered: entered, completed: completed))
            }
        }
        let resumed = DispatchTime.now().uptimeNanoseconds
        let failed: Bool
        switch fetched.result {
        case .success: failed = false
        case .failure: failed = true
        }
        capture.recordFetch(layer: layer, planID: fetched.planID, allHit: fetched.allHit,
                            enqueued: fetched.enqueued, entered: fetched.entered,
                            completed: fetched.completed, resumed: resumed, failed: failed)
        return try fetched.result.get()
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: GTurboFormatV1.DType.u32.rawValue)
        }
    }
}
