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
