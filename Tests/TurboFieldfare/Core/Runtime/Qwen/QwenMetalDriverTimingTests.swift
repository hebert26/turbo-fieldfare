import Foundation
import Testing
@testable import TurboFieldfare

struct QwenMetalDriverTimingTests {
    @Test func invalidClocksNeverBecomeZeroDurationSuccess() {
        for (start, end) in [(0.0, 0.0), (-1.0, 2.0), (2.0, 1.0),
                             (Double.nan, 2.0), (1.0, Double.infinity)] {
            #expect(QwenProductionCommand.driverClockFlags(
                start: start, end: end, completedSuccessfully: true) == 1)
        }
        #expect(QwenProductionCommand.driverClockFlags(
            start: 1, end: 1, completedSuccessfully: true) == 3)
        #expect(QwenProductionCommand.driverClockFlags(
            start: 1, end: 2, completedSuccessfully: false) == 1)
    }

    @Test func rawInvalidClocksAndUnavailableCommandsHaveCompleteGroups() throws {
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        try #require(capture.beginQwenCacheMaps(layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
        try #require(capture.beginQwenProductionTiming())
        capture.setQwenCacheMapPhase(.decode)
        let location = try #require(capture.qwenProductionLocation(position: 7, tokenCount: 1, forward: true))
        let invalidStart = UInt64(0x7ff8000000000042)
        capture.recordQwenProductionCommand(location, stage: .moe,
            submitBefore: 1, submitAfter: 2, waitBefore: 3, resumed: 4,
            gpuStartBits: 0, gpuEndBits: 0, flags: 0, completionCallback: 0,
            kernelStartBits: invalidStart, kernelEndBits: Double.infinity.bitPattern, kernelFlags: 1)
        capture.recordQwenProductionCommand(location, stage: .moe,
            submitBefore: 5, submitAfter: 6, waitBefore: 0, resumed: 0,
            gpuStartBits: 0, gpuEndBits: 0, flags: 4)
        capture.finish(status: 0)
        struct Batch: Decodable { let records: [[UInt64]] }
        var rows: [[UInt64]] = []
        while let batch = capture.drainJSONBatch(maximumBytes: RuntimeMeasurementCapture.maximumJSONBatchBytes) {
            rows += try JSONDecoder().decode(Batch.self, from: batch.data).records
        }
        let drivers = rows.filter { $0[0] == 171 }
        #expect(drivers.count == 2)
        #expect(drivers[0][2] == invalidStart)
        #expect(drivers[0][3] == Double.infinity.bitPattern && drivers[0][4] == 1)
        #expect(Array(drivers[1][2...5]) == [0, 0, 0, 0])
        let ids = Set(drivers.map { $0[1] })
        for kind in [UInt64(161), 162, 163, 170] {
            #expect(Set(rows.filter { $0[0] == kind }.map { $0[1] }) == ids)
        }
    }
}
