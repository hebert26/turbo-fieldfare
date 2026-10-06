import Foundation
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

@Suite(.serialized)
struct DecodeServiceMeasurementQueueTests {
    @Test func thirtyTwoTinyBatchesRemainFIFOAndFooterPrecedesTerminal() async throws {
        let generationID = UUID()
        let pipe = Pipe()
        let queued = DispatchSemaphore(value: 0)
        let router = DecodeServiceResponseRouter(
            output: pipe.fileHandleForReading,
            onEventQueued: { _ in queued.signal() })
        #expect(router.beginMeasurementReception(generationID))

        var events = (0..<32).map { index in
            measurementEvent(
                generationID: generationID,
                sequence: UInt64(index),
                payload: "tiny-\(index)")
        }
        events.append(measurementEvent(
            generationID: generationID,
            sequence: 32,
            payload: "footer",
            footer: true))
        events.append(DecodeServiceEvent(
            kind: .finished, generationID: generationID, sequence: 33))
        let writer = write(events, to: pipe.fileHandleForWriting)
        try waitForSignals(32, queued)

        var sequences: [UInt64] = []
        for _ in 0..<32 {
            let event = try await router.next(matching: generationID)
            #expect(event.kind == .measurement)
            #expect(event.measurementDroppedBatches == nil)
            #expect(event.measurementDroppedBytes == nil)
            sequences.append(event.sequence)
        }
        let footer = try await router.next(matching: generationID)
        #expect(footer.kind == .measurement)
        #expect(footer.measurementContainsFooter == true)
        #expect(footer.sequence == 32)
        let terminal = try await router.next(matching: generationID)
        #expect(terminal.kind == .finished)
        #expect(terminal.sequence == 33)
        #expect(sequences == Array(0..<32).map(UInt64.init))
        try writer.wait()
    }

    @Test func countOverflowReportsExact33rdBatchLossBeforeTerminal() async throws {
        let generationID = UUID()
        let pipe = Pipe()
        let queued = DispatchSemaphore(value: 0)
        let router = DecodeServiceResponseRouter(
            output: pipe.fileHandleForReading,
            onEventQueued: { _ in queued.signal() })
        #expect(router.beginMeasurementReception(generationID))

        let events = (0..<33).map { index in
            measurementEvent(
                generationID: generationID,
                sequence: UInt64(index),
                payload: "count-\(index)")
        } + [DecodeServiceEvent(
            kind: .finished, generationID: generationID, sequence: 33)]
        let writer = write(events, to: pipe.fileHandleForWriting)
        try waitForSignals(33, queued)
        try writer.wait()

        for expected in 0..<32 {
            let event = try await router.next(matching: generationID)
            #expect(event.kind == .measurement)
            #expect(event.sequence == UInt64(expected))
            #expect(event.measurementDroppedBatches == nil)
        }
        let terminal = try await router.next(matching: generationID)
        #expect(terminal.kind == .finished)
        #expect(terminal.measurementDroppedBatches == 1)
        #expect(terminal.measurementDroppedBytes == UInt64("count-32".utf8.count))
    }

    @Test func byteCeilingStillReportsThirdFortyEightKiBBatchLoss() async throws {
        let generationID = UUID()
        let pipe = Pipe()
        let queued = DispatchSemaphore(value: 0)
        let router = DecodeServiceResponseRouter(
            output: pipe.fileHandleForReading,
            onEventQueued: { _ in queued.signal() })
        #expect(router.beginMeasurementReception(generationID))

        let payload = String(repeating: "x", count: 48 * 1_024)
        let events = (0..<3).map { index in
            measurementEvent(
                generationID: generationID,
                sequence: UInt64(index),
                payload: payload)
        } + [DecodeServiceEvent(
            kind: .finished, generationID: generationID, sequence: 3)]
        let writer = write(events, to: pipe.fileHandleForWriting)
        try waitForSignals(3, queued)
        try writer.wait()

        for expected in 0..<2 {
            let event = try await router.next(matching: generationID)
            #expect(event.kind == .measurement)
            #expect(event.sequence == UInt64(expected))
            #expect(event.measurementBatchJSON?.utf8.count == 48 * 1_024)
            #expect(event.measurementDroppedBatches == nil)
        }
        let terminal = try await router.next(matching: generationID)
        #expect(terminal.kind == .finished)
        #expect(terminal.measurementDroppedBatches == 1)
        #expect(terminal.measurementDroppedBytes == UInt64(48 * 1_024))
    }

    @Test func abandonmentRemovesQueuedMeasurementsAndAttachesExactLossToTerminal() async throws {
        let generationID = UUID()
        let pipe = Pipe()
        let queued = DispatchSemaphore(value: 0)
        let router = DecodeServiceResponseRouter(
            output: pipe.fileHandleForReading,
            onEventQueued: { _ in queued.signal() })
        #expect(router.beginMeasurementReception(generationID))

        let payloads = (0..<4).map { "abandon-\($0)" }
        var events = payloads.enumerated().map { index, payload in
            measurementEvent(
                generationID: generationID,
                sequence: UInt64(index),
                payload: payload,
                footer: true)
        }
        events.append(DecodeServiceEvent(
            kind: .cancelled, generationID: generationID, sequence: 4))
        let writer = write(events, to: pipe.fileHandleForWriting)
        try waitForSignals(5, queued)

        router.abandonMeasurementReception(generationID)
        let terminal = try await router.next(matching: generationID)
        #expect(terminal.kind == .cancelled)
        #expect(terminal.measurementDroppedBatches == 4)
        #expect(terminal.measurementDroppedBytes == UInt64(payloads.reduce(0) {
            $0 + $1.utf8.count
        }))
        try writer.wait()
    }

    @Test func receiverAdmissionRemainsEightAndTerminationRejectsNewReceiver() throws {
        let pipe = Pipe()
        let router = DecodeServiceResponseRouter(output: pipe.fileHandleForReading)
        let generationIDs = (0..<9).map { _ in UUID() }
        for index in 0..<8 {
            #expect(router.beginMeasurementReception(generationIDs[index]))
        }
        #expect(!router.beginMeasurementReception(generationIDs[8]))

        router.terminateWaiters()
        #expect(!router.beginMeasurementReception(UUID()))
        try? pipe.fileHandleForWriting.close()
    }

    private func measurementEvent(
        generationID: UUID,
        sequence: UInt64,
        payload: String,
        footer: Bool = false) -> DecodeServiceEvent {
        var event = DecodeServiceEvent(
            kind: .measurement, generationID: generationID, sequence: sequence)
        event.measurementBatchJSON = payload
        event.measurementContainsFooter = footer
        return event
    }

    private func write(
        _ events: [DecodeServiceEvent],
        to output: FileHandle) -> ThreadCompletion {
        let completion = ThreadCompletion()
        Thread {
            defer {
                try? output.close()
                completion.signal()
            }
            do {
                for event in events {
                    let frame = try DecodeFrameCodec.encode(event)
                    try output.write(contentsOf: frame)
                }
            } catch {
                completion.record(error)
            }
        }.start()
        return completion
    }

    private func waitForSignals(
        _ count: Int,
        _ semaphore: DispatchSemaphore) throws {
        for _ in 0..<count {
            guard semaphore.wait(timeout: .now() + 5) == .success else {
                throw MeasurementQueueTestError.timeout
            }
        }
    }
}

private final class ThreadCompletion: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var error: Error?

    func record(_ error: Error) {
        lock.lock()
        self.error = error
        lock.unlock()
    }

    func signal() { semaphore.signal() }

    func wait() throws {
        guard semaphore.wait(timeout: .now() + 5) == .success else {
            throw MeasurementQueueTestError.timeout
        }
        lock.lock()
        let error = self.error
        lock.unlock()
        if let error { throw error }
    }
}

private enum MeasurementQueueTestError: Error {
    case timeout
}
