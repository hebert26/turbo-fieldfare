import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized)
struct QwenCompletionResumptionTimingTests {
    @Test func completionRowsDistinguishOrderedMissingLateAndUnawaitedSamples() throws {
        let (capture, location) = try collector()
        for (completed, resumed) in [(UInt64(10), UInt64(20)), (0, 20), (30, 20), (10, 0)] {
            capture.recordQwenProductionCommand(location, stage: .moe,
                submitBefore: 1, submitAfter: 2, waitBefore: 3, resumed: resumed,
                gpuStartBits: 0, gpuEndBits: 0, flags: 0, completionCallback: completed)
        }
        capture.endQwenProductionTiming()
        let rows = try drainCompletionRows(capture)
        let observations = rows.filter { $0[0] == 170 }
        #expect(observations.map { $0[2] } == [10, 0, 30, 10])
        #expect(observations.map { $0[3] } == [20, 20, 20, 0])
        #expect(observations.map { $0[4] } == [3, 0, 9, 5])
        #expect(Set(observations.map { $0[1] }) == Set(rows.filter { $0[0] == 161 }.map { $0[1] }))
        #expect(rows.filter { $0[0] == 163 }.allSatisfy { $0[4] == 0 },
                "host observation flags must not mark an invalid GPU clock valid")
    }

    @Test func lateCallbackDoesNotMutateAnUnavailableSnapshotOrClosedCollector() throws {
        let (capture, location) = try collector()
        let clock = QwenProductionCompletionClock()
        #expect(clock.snapshot() == 0)
        capture.recordQwenProductionCommand(location, stage: .moe,
            submitBefore: 1, submitAfter: 2, waitBefore: 3, resumed: 4,
            gpuStartBits: 0, gpuEndBits: 0, flags: 0, completionCallback: clock.snapshot())
        capture.endQwenProductionTiming()
        clock.completed()
        let first = clock.snapshot()
        #expect(first > 0)
        clock.completed()
        #expect(clock.snapshot() == first, "a duplicate observation must not replace the first clock")
        let rows = try drainCompletionRows(capture)
        let observation = try #require(rows.first { $0[0] == 170 })
        #expect(observation[2] == 0 && observation[4] == 0,
                "missing means unavailable, never zero delay or a retroactive observation")
        #expect(rows.filter { $0[0] == 170 }.count == 1)
    }

    @Test func workerHandoffUsesHostSpanWithNoCallingThreadCPUSample() throws {
        let (capture, location) = try collector()
        let handoff = QwenProductionWorkerHandoff(location: location)
        handoff.completed()
        handoff.resumed()
        capture.endQwenProductionTiming()
        let rows = try drainCompletionRows(capture)
        let metadata = try #require(rows.first { $0[0] == 164 && $0[2] == 22 })
        #expect(metadata[3] == UInt64(location.layer + 1))
        #expect(metadata[4] == UInt64(location.position))
        let clocks = try #require(rows.first { $0[0] == 165 && $0[1] == metadata[1] })
        #expect(clocks[2] > 0 && clocks[3] >= clocks[2])
        #expect(clocks[4] == 0 && clocks[5] == 0,
                "a worker/continuation handoff is not one thread's CPU interval")
        #expect(QwenProductionTimingMeasurement.workerHandoff() == nil)
    }

    @Test func fiveRowCommandReservationDoesNotLeavePartialRecordsAtBound() throws {
        let (capture, location) = try collector()
        let count = RuntimeMeasurementCapture.maximumQwenProductionRows / 5
        for _ in 0..<count {
            capture.recordQwenProductionCommand(location, stage: .moe,
                submitBefore: 1, submitAfter: 2, waitBefore: 3, resumed: 20,
                gpuStartBits: 0, gpuEndBits: 0, flags: 0, completionCallback: 10)
        }
        capture.recordQwenProductionCommand(location, stage: .moe,
            submitBefore: 1, submitAfter: 2, waitBefore: 3, resumed: 20,
            gpuStartBits: 0, gpuEndBits: 0, flags: 0, completionCallback: 10)
        capture.recordQwenProductionSpan(location, stage: .expertMapWorkerResume,
            started: 10, ended: 20, cpuStarted: 0, cpuEnded: 0)
        capture.endQwenProductionTiming()
        let rows = try drainCompletionRows(capture)
        for kind in [UInt64(161), 162, 163, 170, 171] {
            #expect(UInt64(rows.filter { $0[0] == kind }.count) == count)
        }
        #expect(rows.filter { $0[0] == 164 }.count == 1)
        let footer = try #require(rows.first { $0[0] == 166 })
        #expect(footer[1] == count * 5 + 2 && footer[2] == 5 && footer[3] == count)
    }

    @Test func existingMetalCompletionAwaitAndCommandCountRemainUnchanged() async throws {
        let (capture, location) = try collector()
        let context = try MetalContext()
        let command = try #require(context.queue.makeCommandBuffer())
        let timing = QwenProductionCommand(location: location, stage: .moe)
        timing.willCommit(command)
        command.commit()
        timing.didCommit()
        timing.willWait()
        await command.completed()
        timing.resumed(command)
        capture.endQwenProductionTiming()
        let rows = try drainCompletionRows(capture)
        #expect(rows.filter { $0[0] == 161 }.count == 1)
        let driver = try #require(rows.first { $0[0] == 171 })
        #expect(driver[2] == command.kernelStartTime.bitPattern)
        #expect(driver[3] == command.kernelEndTime.bitPattern)
        #expect(driver[4] == QwenProductionCommand.driverClockFlags(
            start: command.kernelStartTime, end: command.kernelEndTime,
            completedSuccessfully: command.status == .completed && command.error == nil))
        let wait = try #require(rows.first { $0[0] == 162 })
        let observed = try #require(rows.first { $0[0] == 170 })
        #expect(observed[1] == wait[1] && observed[3] == wait[5])
        if (observed[4] & 2) != 0 { #expect(observed[2] > 0 && observed[2] <= observed[3]) }
        else if (observed[4] & 8) != 0 { #expect(observed[2] > observed[3]) }
        else { #expect(observed[2] == 0 && observed[4] == 0) }
    }

    private func collector() throws -> (RuntimeMeasurementCapture, QwenProductionLocation) {
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        try #require(capture.beginQwenCacheMaps(layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
        try #require(capture.beginQwenProductionTiming())
        capture.setQwenCacheMapPhase(.decode)
        let location = try #require(capture.qwenProductionLocation(position: 7, tokenCount: 1, forward: true))
        return (capture, location.atLayer(0))
    }
}

private func drainCompletionRows(_ capture: RuntimeMeasurementCapture) throws -> [[UInt64]] {
    struct Batch: Decodable { let records: [[UInt64]] }
    capture.finish(status: 0)
    var rows: [[UInt64]] = []
    while let batch = capture.drainJSONBatch(maximumBytes: RuntimeMeasurementCapture.maximumJSONBatchBytes) {
        rows += try JSONDecoder().decode(Batch.self, from: batch.data).records.filter {
            $0.count == 6 && (160...171).contains($0[0])
        }
    }
    return rows
}
