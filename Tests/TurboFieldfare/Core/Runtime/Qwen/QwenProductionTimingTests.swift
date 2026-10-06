import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Tests for the staged Qwen production timing collector. These tests use the
/// tiny source fixture or literal collector rows and never read model shards.
@Suite(.serialized)
struct QwenProductionTimingTests {
    @Test
    func disabledCaptureProducesNoProductionRows() throws {
        let capture = RuntimeMeasurementCapture()
        #expect(!capture.beginQwenProductionTiming(),
                "normal capture configuration must not enable timing without cache scope")
        #expect(QwenProductionTimingMeasurement.command(.sampler) == nil)
        #expect(QwenProductionTimingMeasurement.span(.forward) == nil)
        let rows = try drainProductionRows(capture)
        #expect(rows.isEmpty)
    }

    @Test
    func productionCaptureIsDecodeOnlyAndCapsForwards() throws {
        let capture = try makeProductionCapture()
        #expect(capture.beginQwenProductionTiming())
        #expect(capture.qwenProductionLocation(position: 0, tokenCount: 4, forward: true) == nil,
                "prefill must not acquire a production timing location")

        capture.setQwenCacheMapPhase(.decode)
        for position in 0..<32 {
            #expect(capture.qwenProductionLocation(
                position: position, tokenCount: 1, forward: true) != nil)
        }
        #expect(capture.qwenProductionLocation(position: 32, tokenCount: 1, forward: true) == nil,
                "the forward bound must omit later forwards rather than grow storage")
        capture.endQwenProductionTiming()

        let rows = try drainProductionRows(capture)
        let scope = try #require(rows.first { $0[0] == 160 })
        #expect(scope[1] == 3)
        #expect(scope[2] == RuntimeMeasurementCapture.Phase.decode.rawValue)
        #expect(scope[3] == 32)
        #expect(scope[4] == RuntimeMeasurementCapture.maximumQwenProductionRows)
        let completion = try #require(rows.first { $0[0] == 166 })
        #expect(completion[4] == 32)
        #expect(completion[5] == 1)
        let correlation = try #require(rows.first { $0[0] == 168 })
        #expect(correlation[1] <= correlation[5])
        #expect(correlation[3] > 0)
        #expect(correlation[4] > 0)
    }

    @Test
    func commandAndSpanReservationsAreAtomicAtRowLimit() throws {
        let capture = try makeProductionCapture()
        #expect(capture.beginQwenProductionTiming())
        capture.setQwenCacheMapPhase(.decode)
        let location = try #require(capture.qwenProductionLocation(
            position: 1, tokenCount: 1, forward: true))

        // Keep the last two rows for one complete span. The next command
        // must be dropped atomically under either bounded scope version.
        let maximumCommands = (RuntimeMeasurementCapture.maximumQwenProductionRows - 2) / 5
        for _ in 0..<maximumCommands {
            capture.recordQwenProductionCommand(
                location, stage: .forward,
                submitBefore: 10, submitAfter: 20, waitBefore: 30, resumed: 40,
                gpuStartBits: 50, gpuEndBits: 60, flags: 3)
        }
        capture.recordQwenProductionSpan(
            location, stage: .forward,
            started: 100, ended: 110, cpuStarted: 200, cpuEnded: 210)
        capture.recordQwenProductionCommand(
            location, stage: .forward,
            submitBefore: 70, submitAfter: 80, waitBefore: 90, resumed: 100,
            gpuStartBits: 0, gpuEndBits: 0, flags: 4)
        capture.endQwenProductionTiming()

        let rows = try drainProductionRows(capture)
        let commands = rows.filter { $0[0] == 161 }
        let waits = rows.filter { $0[0] == 162 }
        let gpu = rows.filter { $0[0] == 163 }
        let spans = rows.filter { $0[0] == 164 }
        let clocks = rows.filter { $0[0] == 165 }
        #expect(UInt64(commands.count) == maximumCommands)
        #expect(waits.count == commands.count)
        #expect(gpu.count == commands.count)
        #expect(spans.count == 1)
        #expect(clocks.count == 1)
        #expect(Set(commands.map { $0[1] }) == Set(waits.map { $0[1] }))
        #expect(Set(commands.map { $0[1] }) == Set(gpu.map { $0[1] }))

        let completion = try #require(rows.first { $0[0] == 166 })
        #expect(completion[1] == maximumCommands * 5 + 2)
        #expect(completion[2] == 5)
        #expect(completion[3] == maximumCommands)
        #expect(!commands.contains { $0[1] == maximumCommands + 1 },
                "a failed five-row reservation must not leave a partial command")
    }

    @Test
    func invalidGPUClockRowPreservesInvalidFlagWithoutClockSubtraction() throws {
        let capture = try makeProductionCapture()
        #expect(capture.beginQwenProductionTiming())
        capture.setQwenCacheMapPhase(.decode)
        let location = try #require(capture.qwenProductionLocation(
            position: 0, tokenCount: 1, forward: true))
        capture.recordQwenProductionCommand(
            location, stage: .sampler,
            submitBefore: 10, submitAfter: 20, waitBefore: 30, resumed: 40,
            gpuStartBits: 0, gpuEndBits: 0, flags: 2)
        capture.endQwenProductionTiming()

        let rows = try drainProductionRows(capture)
        let gpu = try #require(rows.first { $0[0] == 163 })
        #expect(gpu[2] == 0)
        #expect(gpu[3] == 0)
        #expect((gpu[4] & 1) == 0,
                "completed status alone must not mark an invalid GPU interval valid")
    }

    @Test
    func instrumentationPreservesTinyTurnAndSettlesCancellation() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let modelWithoutCapture = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget(
                expertSlotCount: QwenBF16TextRunnerFixture.topK),
            expertCacheSlots: QwenBF16TextRunnerFixture.topK,
            expertCachePolicy: .lfu)
        let modelWithCapture = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget(
                expertSlotCount: QwenBF16TextRunnerFixture.topK),
            expertCacheSlots: QwenBF16TextRunnerFixture.topK,
            expertCachePolicy: .lfu)
        let plain = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: modelWithoutCapture, context: context, maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let instrumented = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: modelWithCapture, context: context, maxContext: 16,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        var config = GenerationConfig(
            maxNewTokens: 3, temperature: 0, topK: nil, topP: nil,
            repetitionPenalty: 1, seed: 0, stopStrings: [], extraStopTokens: [])
        config.logitTransform = .raw
        let prompt = QwenBF16TextRunnerFixture.inputTokens
        let plainResult = try await plain.generatePreparedTurn(
            promptTokenIDs: prompt, config: config)
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        let instrumentedResult = try await instrumented.generatePreparedTurn(
            promptTokenIDs: prompt, config: config, measurementCapture: capture)
        #expect(instrumentedResult.reason == plainResult.reason)
        #expect(instrumentedResult.promptTokens == plainResult.promptTokens)
        #expect(instrumentedResult.newTokens == plainResult.newTokens)
        #expect(instrumentedResult.acceptedGeneratedTokenIDs == plainResult.acceptedGeneratedTokenIDs)

        let rows = try drainProductionRows(capture)
        let commandRows = rows.filter { $0[0] == 161 }
        let waitRows = rows.filter { $0[0] == 162 }
        let gpuRows = rows.filter { $0[0] == 163 }
        #expect(!commandRows.isEmpty)
        #expect(commandRows.count == waitRows.count)
        #expect(commandRows.count == gpuRows.count)
        let resumptionRows = rows.filter { $0[0] == 170 }
        #expect(resumptionRows.count == commandRows.count)
        #expect(Set(commandRows.map { $0[1] }) == Set(resumptionRows.map { $0[1] }))
        for row in resumptionRows {
            let wait = try #require(waitRows.first { $0[1] == row[1] })
            #expect(row[3] == wait[5])
            if (row[4] & 2) != 0 { #expect(row[2] > 0 && row[2] <= row[3]) }
            if (row[4] & 8) != 0 { #expect(row[2] > row[3]) }
        }
        let workerRows = rows.filter { $0[0] == 164 && $0[2] == 22 }
        #expect(!workerRows.isEmpty)
        #expect(workerRows.count == rows.filter { $0[0] == 164 && $0[2] == 17 }.count)
        for row in workerRows {
            let clocks = try #require(rows.first { $0[0] == 165 && $0[1] == row[1] })
            #expect(clocks[2] > 0 && clocks[3] >= clocks[2])
            #expect(clocks[4] == 0 && clocks[5] == 0)
        }
        #expect(Set(commandRows.map { $0[1] }) == Set(waitRows.map { $0[1] }))
        #expect(Set(commandRows.map { $0[1] }) == Set(gpuRows.map { $0[1] }))
        let stageValues = Set(commandRows.map { $0[2] } + rows.filter { $0[0] == 164 }.map { $0[2] })
        for required in [
            QwenProductionStage.embedding.rawValue,
            QwenProductionStage.router.rawValue,
            QwenProductionStage.head.rawValue,
            QwenProductionStage.moe.rawValue,
            QwenProductionStage.cpuAttention.rawValue,
            QwenProductionStage.sampler.rawValue
        ] {
            #expect(stageValues.contains(required), "missing staged timing value \(required)")
        }
        #expect(!stageValues.contains(255), "unknown stages must not appear in a normal fixture turn")
        let phases = rows.filter { $0[0] == 167 }.map { $0[1] }
        #expect(phases.contains(2))
        #expect(phases.contains(1))
        #expect(phases.contains(3))

        let failedCapture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        var cancellationError: Error?
        do {
            _ = try await instrumented.generatePreparedTurn(
                promptTokenIDs: prompt, config: config,
                measurementCapture: failedCapture, shouldStop: { true })
            Issue.record("the cancellation fixture unexpectedly completed")
        } catch {
            cancellationError = error
        }
        #expect(cancellationError is CancellationError)
        let failedRows = try drainProductionRows(failedCapture)
        #expect(failedRows.contains { $0[0] == 166 },
                "cancellation must settle and close the timing collector")
    }

    private func makeProductionCapture() throws -> RuntimeMeasurementCapture {
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        #expect(capture.beginQwenCacheMaps(
            layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
        return capture
    }
}

private func drainProductionRows(_ capture: RuntimeMeasurementCapture) throws -> [[UInt64]] {
    struct Batch: Decodable { let records: [[UInt64]] }
    capture.finish(status: 0)
    var rows: [[UInt64]] = []
    while let batch = capture.drainJSONBatch(
        maximumBytes: RuntimeMeasurementCapture.maximumJSONBatchBytes) {
        let decoded = try JSONDecoder().decode(Batch.self, from: batch.data)
        rows += decoded.records.filter {
            $0.count == 6 && (160...171).contains($0[0])
        }
    }
    return rows
}
