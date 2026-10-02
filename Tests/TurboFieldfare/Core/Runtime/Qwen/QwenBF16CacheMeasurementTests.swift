import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized)
struct QwenBF16CacheMeasurementTests {
    @Test
    func qwenCacheCapturePreservesSnapshotOrderingAndPhaseMetadata() throws {
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        #expect(capture.beginQwenCacheMaps(
            layerCount: 2, expertCount: 9, slotCount: 8, pairBytes: 96))

        let initial = QwenBF16PairedExpertCache.MeasurementSnapshot(
            expertIDs: Array(0..<8),
            useCounts: [4, 3, 2, 1, 0, 7, 6, 5],
            lastUse: [11, 12, 13, 14, 15, 16, 17, 18],
            clock: 18,
            policy: .lfu)
        capture.recordQwenCacheInitial(layer: 0, expertCount: 9) { initial }
        capture.recordQwenCacheInitial(layer: 0, expertCount: 9) {
            QwenBF16PairedExpertCache.MeasurementSnapshot(
                expertIDs: Array(repeating: 8, count: 8),
                useCounts: Array(repeating: 99, count: 8),
                lastUse: Array(repeating: 99, count: 8),
                clock: 99,
                policy: .lru)
        }

        let prefill = capture.beginQwenCacheMap(
            QwenCacheMapContext(capture: capture, layer: 0, position: 3, tokenCount: 1))
        #expect(prefill == 1)
        let prefillPlan = ExpertCachePlan(
            experts: [7, 2, 5, 0, 4, 1, 6, 3],
            assignedSlots: [0, 1, 2, 3, 4, 5, 6, 7],
            misses: [0, 2, 4], hits: 5)
        capture.recordQwenCachePlan(
            prefill, plan: prefillPlan, clock: 19, avoidingSlots: [1, 4])
        capture.recordQwenCacheOutcome(prefill, status: 0, stage: 3)

        capture.setQwenCacheMapPhase(.decode)
        let decode = capture.beginQwenCacheMap(
            QwenCacheMapContext(capture: capture, layer: 1, position: 4, tokenCount: 2))
        #expect(decode == 2)
        let decodePlan = ExpertCachePlan(
            experts: [8, 6, 4, 2, 0, 1, 3, 5],
            assignedSlots: [7, 6, 5, 4, 3, 2, 1, 0],
            misses: [1, 6], hits: 6)
        capture.recordQwenCachePlan(
            decode, plan: decodePlan, clock: 20, avoidingSlots: [])
        capture.recordQwenCacheOutcome(decode, status: 0, stage: 0)
        capture.endQwenCacheMaps(completed: true)

        let rows = try drainQwenCacheRows(capture)
        #expect(rows.contains([140, 1, 2, 9, 8, 96]))

        let initialLayers = rows.filter { $0[0] == 141 }
        #expect(initialLayers == [[141, 0, 1, 18, 8, 9]])
        let initialSlots = rows.filter { $0[0] == 142 }
        #expect(initialSlots.count == 8)
        for slot in 0..<8 {
            #expect(initialSlots.contains([
                142, 0, UInt64(slot), UInt64(slot + 1), initial.useCounts[slot], initial.lastUse[slot]
            ]))
        }

        #expect(rows.contains([143, 1, 2, 0, 3, 1]))
        #expect(rows.contains([143, 2, 1, 1, 4, 2]))
        #expect(rows.contains([144, 1, 19, 8, 5, 3]))
        #expect(rows.contains([144, 2, 20, 8, 6, 2]))
        #expect(rows.contains([145, 1, 2, 0, 0, 0]))
        #expect(rows.contains([146, 1, 1, 0, 0, 0]))
        #expect(rows.contains([146, 1, 4, 0, 0, 0]))
        #expect(rows.contains([147, 1, 0, 7, 1, 1]))
        #expect(rows.contains([147, 1, 1, 2, 2, 0]))
        #expect(rows.contains([147, 2, 1, 6, 7, 1]))
        #expect(rows.contains([148, 1, 0, 3, 0, 0]))
        #expect(rows.contains([148, 2, 0, 0, 0, 0]))
        #expect(rows.contains([149, 2, 2, 0, 0, 1]))
    }

    @Test
    func qwenCacheCaptureRecordsFailureCancellationAndBoundedOverflow() throws {
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        #expect(capture.beginQwenCacheMaps(
            layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
        let first = capture.beginQwenCacheMap(
            QwenCacheMapContext(capture: capture, layer: 0, position: 0, tokenCount: 1))
        capture.recordQwenCacheOutcome(first, status: 1, stage: 1)
        let second = capture.beginQwenCacheMap(
            QwenCacheMapContext(capture: capture, layer: 0, position: 1, tokenCount: 1))
        capture.recordQwenCacheOutcome(second, status: 2, stage: 2)
        capture.endQwenCacheMaps(completed: false)

        let rows = try drainQwenCacheRows(capture)
        #expect(rows.contains([148, first, 1, 1, 0, 0]))
        #expect(rows.contains([148, second, 2, 2, 0, 0]))
        let completion = try #require(rows.first { $0[0] == 149 })
        #expect(completion[1] == 2)
        #expect(completion[2] == 0)
        #expect(completion[3] == 2)
        #expect(completion[5] == 0)

        let overflow = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        #expect(overflow.beginQwenCacheMaps(
            layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
        let overflowID = overflow.beginQwenCacheMap(
            QwenCacheMapContext(capture: overflow, layer: 0, position: 2, tokenCount: 1))
        overflow.recordQwenCacheUnplanned(
            overflowID, experts: Array(repeating: 8, count: 50_000))
        overflow.endQwenCacheMaps(completed: false)
        let overflowRows = try drainQwenCacheRows(overflow)
        let overflowCompletion = try #require(overflowRows.first { $0[0] == 149 })
        #expect(overflowCompletion[1] == 1)
        #expect(overflowCompletion[2] == 0)
        #expect(overflowCompletion[3] == 0)
        #expect(overflowCompletion[4] > 0)
        #expect(overflowCompletion[5] == 0)
    }

    @Test
    func qwenCacheCaptureIsQuietWhenCacheMapsAreDisabled() throws {
        let capture = RuntimeMeasurementCapture()
        let snapshot = QwenBF16PairedExpertCache.MeasurementSnapshot(
            expertIDs: Array(0..<8), useCounts: Array(repeating: 0, count: 8),
            lastUse: Array(repeating: 0, count: 8), clock: 0, policy: .lru)
        capture.recordQwenCacheInitial(layer: 0, expertCount: 9) { snapshot }
        let map = capture.beginQwenCacheMap(
            QwenCacheMapContext(capture: capture, layer: 0, position: 0, tokenCount: 1))
        capture.recordQwenCacheOutcome(map, status: 0, stage: 0)
        capture.endQwenCacheMaps(completed: true)

        let rows = try drainQwenCacheRows(capture)
        #expect(rows.filter { (140...149).contains($0[0]) }.isEmpty)
    }

    @Test
    func qwenCacheCaptureDefaultsToDecodeAndKeepsWarmSlotState() throws {
        let capture = RuntimeMeasurementCapture()
        #expect(capture.beginQwenCacheMaps(
            layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
        let initial = QwenBF16PairedExpertCache.MeasurementSnapshot(
            expertIDs: Array(0..<8), useCounts: Array(repeating: 1, count: 8),
            lastUse: (1...8).map(UInt64.init), clock: 8, policy: .lfu)
        capture.recordQwenCacheInitial(layer: 0, expertCount: 9) { initial }
        let prefill = capture.beginQwenCacheMap(
            QwenCacheMapContext(capture: capture, layer: 0, position: 0, tokenCount: 1))
        #expect(prefill == 0)
        capture.setQwenCacheMapPhase(.decode)
        capture.recordQwenCacheInitial(layer: 0, expertCount: 9) { initial }
        let decode = capture.beginQwenCacheMap(
            QwenCacheMapContext(capture: capture, layer: 0, position: 1, tokenCount: 1))
        #expect(decode == 1)
        capture.recordQwenCachePlan(
            decode,
            plan: ExpertCachePlan(
                experts: [0, 1], assignedSlots: [0, 1], misses: [], hits: 2),
            clock: 9, avoidingSlots: [2])
        capture.recordQwenCacheOutcome(decode, status: 0, stage: 3)
        capture.endQwenCacheMaps(completed: true)

        let rows = try drainQwenCacheRows(capture)
        #expect(rows.contains([150, 1, 48_000, 0, 0, 0]))
        #expect(!rows.contains { $0[0] == 143 && $0[2] == 2 })
        #expect(rows.contains([143, 1, 1, 0, 1, 1]))
        #expect(rows.filter { $0[0] == 141 }.count == 1)
        #expect(rows.filter { $0[0] == 142 }.count == 8)
    }

    @Test
    func sourceTurnForwardsQwenCacheMapsWithActualSyntheticFixture() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget(
                expertSlotCount: QwenBF16TextRunnerFixture.topK),
            expertCacheSlots: QwenBF16TextRunnerFixture.topK,
            expertCachePolicy: .lfu)
        let session = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        let result = try await session.generatePreparedTurn(
            promptTokenIDs: QwenBF16TextRunnerFixture.inputTokens,
            config: qwenCacheMeasurementGreedyConfig(maxNewTokens: 2),
            measurementCapture: capture)
        #expect(result.newTokens > 0)

        let rows = try drainQwenCacheRows(capture)
        #expect(rows.contains([140, 1, 2, 9, 8, 96]))

        let initialLayers = rows.filter { $0[0] == 141 }
        #expect(initialLayers.count == QwenBF16TextRunnerFixture.layerCount)
        #expect(Set(initialLayers.map { $0[1] }) == Set((0..<2).map(UInt64.init)))
        #expect(initialLayers.allSatisfy {
            $0[2] == 1 && $0[4] == 8 && $0[5] == 9
        })
        #expect(rows.filter { $0[0] == 142 }.count == 16)

        let maps = rows.filter { $0[0] == 143 }
        #expect(!maps.isEmpty)
        #expect(Set(maps.map { $0[1] }) == Set((1...max(1, maps.count)).map(UInt64.init)))
        #expect(Set(maps.map { $0[2] }) == Set([1, 2]))
        #expect(maps.allSatisfy { $0[3] < 2 && $0[5] > 0 })

        let mapIDs = Set(maps.map { $0[1] })
        let plans = rows.filter { $0[0] == 144 }
        let outcomes = rows.filter { $0[0] == 148 }
        #expect(Set(plans.map { $0[1] }) == mapIDs)
        #expect(Set(outcomes.map { $0[1] }) == mapIDs)
        #expect(outcomes.allSatisfy { $0[2] == 0 && $0[3] == 3 })

        let completion = try #require(rows.first { $0[0] == 149 })
        #expect(completion[1] == completion[2])
        #expect(completion[1] > 0)
        #expect(completion[3] == 0)
        #expect(completion[5] == 1)
    }

    @Test
    func actualTinyCoordinatorReportsColdWarmAssignmentsAndValidationFailure() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let slotCount = QwenBF16TextRunnerFixture.topK
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget(expertSlotCount: slotCount),
            expertCacheSlots: slotCount,
            expertCachePolicy: .lfu)
        let configuration = try QwenMoEConfiguration(
            hiddenSize: model.architecture.hiddenSize,
            expertCount: model.architecture.experts,
            topK: model.architecture.expertsPerToken,
            routedIntermediateSize: model.architecture.routedIntermediateSize,
            sharedIntermediateSize: model.architecture.sharedIntermediateSize)
        let coordinator = try QwenBF16ExpertMappingCoordinator(
            source: model.source, names: model.layers[0].routedNames, layer: 0,
            configuration: configuration, device: context.device,
            slotCount: slotCount, residencyBudget: model.expertCacheBytesPerLayer,
            cachePolicy: .lfu)
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        #expect(capture.beginQwenCacheMaps(
            layerCount: 2, expertCount: 9, slotCount: slotCount, pairBytes: 96))
        capture.setQwenCacheMapPhase(.decode)

        let cold = try await coordinator.map(
            expertIDs: Array(0..<8),
            measurement: QwenCacheMapContext(
                capture: capture, layer: 0, position: 0, tokenCount: 1))
        #expect(cold.diagnostics.requestedExpertIDs == Array(0..<8))
        #expect(cold.diagnostics.assignedSlots == Array(0..<8))
        #expect(cold.diagnostics.hits == 0)
        #expect(cold.diagnostics.misses == 8)
        try cold.cancel()

        let warm = try await coordinator.map(
            expertIDs: [0, 1, 2, 3, 4, 5, 6, 8],
            measurement: QwenCacheMapContext(
                capture: capture, layer: 0, position: 1, tokenCount: 1))
        #expect(warm.diagnostics.requestedExpertIDs == [0, 1, 2, 3, 4, 5, 6, 8])
        #expect(warm.diagnostics.assignedSlots == [0, 1, 2, 3, 4, 5, 6, 7])
        #expect(warm.diagnostics.hits == 7)
        #expect(warm.diagnostics.misses == 1)
        try warm.cancel()

        var rejected = false
        do {
            _ = try await coordinator.map(
                expertIDs: [0, 0],
                measurement: QwenCacheMapContext(
                    capture: capture, layer: 0, position: 2, tokenCount: 1))
        } catch {
            rejected = true
        }
        #expect(rejected)
        capture.endQwenCacheMaps(completed: false)

        let rows = try drainQwenCacheRows(capture)
        #expect(rows.contains([141, 0, 1, 0, 8, 9]))
        #expect(rows.filter { $0[0] == 142 }.count == 8)
        #expect(rows.contains([144, 1, 1, 8, 0, 8]))
        #expect(rows.contains([144, 2, 2, 8, 7, 1]))
        #expect(rows.contains([148, 1, 0, 3, 0, 0]))
        #expect(rows.contains([148, 2, 0, 3, 0, 0]))
        #expect(rows.contains([148, 3, 1, 0, 0, 0]))
        #expect(rows.contains([147, 3, 0, 0, 0, 2]))
        #expect(rows.contains([147, 3, 1, 0, 0, 2]))
        #expect(rows.contains([149, 3, 2, 1, 0, 0]))
    }
}

private func drainQwenCacheRows(_ capture: RuntimeMeasurementCapture) throws -> [[UInt64]] {
    struct Batch: Decodable { let records: [[UInt64]] }
    capture.finish(status: 0)
    var rows: [[UInt64]] = []
    while let batch = capture.drainJSONBatch(
        maximumBytes: RuntimeMeasurementCapture.maximumJSONBatchBytes) {
        let decoded = try JSONDecoder().decode(Batch.self, from: batch.data)
        rows += decoded.records.filter { $0.count == 6 && (140...150).contains($0[0]) }
    }
    return rows
}

private func qwenCacheMeasurementGreedyConfig(maxNewTokens: Int) -> GenerationConfig {
    var config = GenerationConfig(
        maxNewTokens: maxNewTokens, temperature: 0, topK: nil, topP: nil,
        repetitionPenalty: 1, seed: 0, stopStrings: [], extraStopTokens: [])
    config.logitTransform = .raw
    return config
}
