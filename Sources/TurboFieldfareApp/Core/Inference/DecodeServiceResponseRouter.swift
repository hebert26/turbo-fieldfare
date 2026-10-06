import Darwin
import Foundation
import TurboFieldfareDecodeProtocol

final class DecodeServiceResponseRouter: @unchecked Sendable {
    private struct State {
        var pending: [UUID: [DecodeServiceEvent]] = [:]
        var terminalError: Error?
        var measurementBytes = 0
        var measurementBatches = 0
        var droppedGenerationID: UUID?
        var droppedMeasurementBatches: UInt64 = 0
        var droppedMeasurementBytes: UInt64 = 0
        var measurementReceivers: Set<UUID> = []
        var streamClosed = false
        var terminalDominatesPending = false
        var transportShutdownRequested = false
        var transportClosed = false
        var output: FileHandle?
    }

    private let condition = NSCondition()
    private var state = State()
    private let onTerminate: @Sendable (DecodeServiceResponseRouter, Error) -> Void
    private let onEventQueued: @Sendable (UUID) -> Void

    init(
        output: FileHandle,
        onTerminate: @escaping @Sendable (DecodeServiceResponseRouter, Error) -> Void
            = { _, _ in },
        onEventQueued: @escaping @Sendable (UUID) -> Void = { _ in }
    ) {
        self.onTerminate = onTerminate
        self.onEventQueued = onEventQueued
        state.output = output
        let reader = Thread { [weak self] in
            self?.readFrames(from: output)
        }
        reader.name = "TurboFieldfare.DecodeService.ResponseRouter"
        reader.qualityOfService = .userInitiated
        reader.start()
    }

    var isTerminated: Bool {
        condition.lock()
        defer { condition.unlock() }
        return state.terminalError != nil
    }

    /// Logically terminates this owner without touching its descriptor.
    /// Owner termination dominates events queued before retirement.
    func terminateWaiters(with error: Error = DecodeFrameError.unexpectedEOF) {
        condition.lock()
        guard !state.streamClosed else {
            condition.unlock()
            return
        }
        state.streamClosed = true
        state.terminalError = error
        state.terminalDominatesPending = true
        state.pending.removeAll()
        state.measurementBytes = 0
        state.measurementBatches = 0
        state.droppedGenerationID = nil
        state.droppedMeasurementBatches = 0
        state.droppedMeasurementBytes = 0
        state.measurementReceivers.removeAll()
        condition.broadcast()
        condition.unlock()
    }

    /// Physical abort, guarded against reader retirement and descriptor reuse.
    func requestTransportShutdown() {
        condition.lock()
        guard !state.transportClosed, !state.transportShutdownRequested,
              let output = state.output else {
            condition.unlock()
            return
        }
        state.transportShutdownRequested = true
        _ = Darwin.shutdown(output.fileDescriptor, SHUT_RDWR)
        condition.unlock()
    }

    func waitUntilTransportClosed() {
        condition.lock()
        while !state.transportClosed { condition.wait() }
        condition.unlock()
    }

    func beginMeasurementReception(_ generationID: UUID) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        guard !state.streamClosed, state.terminalError == nil,
              state.measurementReceivers.count < 8 else { return false }
        state.measurementReceivers.insert(generationID)
        return true
    }

    /// A failed/cancelled receive loop cannot keep the socket reader waiting
    /// behind footer batches that nobody will consume. Ordinary events stay intact.
    func abandonMeasurementReception(_ generationID: UUID) {
        condition.lock()
        defer { condition.broadcast(); condition.unlock() }
        state.measurementReceivers.remove(generationID)
        guard var events = state.pending[generationID] else { return }
        var removedBatches: UInt64 = 0
        var removedBytes: UInt64 = 0
        events.removeAll { event in
            guard let json = event.measurementBatchJSON else { return false }
            let bytes = json.utf8.count
            state.measurementBytes -= bytes
            state.measurementBatches -= 1
            removedBatches &+= 1 &+ (event.measurementDroppedBatches ?? 0)
            removedBytes &+= UInt64(bytes) &+ (event.measurementDroppedBytes ?? 0)
            return true
        }
        if removedBatches > 0 {
            if let terminalIndex = events.lastIndex(where: {
                $0.kind == .finished || $0.kind == .cancelled
                    || $0.kind == .failed || $0.kind == .lineageLost
            }) {
                // The reader may already be on a newer generation. Attach this
                // abandoned queue's loss to its own queued terminal, never to
                // the current socket generation's pending loss counters.
                events[terminalIndex].measurementDroppedBatches =
                    (events[terminalIndex].measurementDroppedBatches ?? 0) &+ removedBatches
                events[terminalIndex].measurementDroppedBytes =
                    (events[terminalIndex].measurementDroppedBytes ?? 0) &+ removedBytes
            } else {
                state.droppedGenerationID = generationID
                state.droppedMeasurementBatches &+= removedBatches
                state.droppedMeasurementBytes &+= removedBytes
            }
        }
        if events.isEmpty { state.pending.removeValue(forKey: generationID) }
        else { state.pending[generationID] = events }
    }

    private func recordMeasurementDropLocked(generationID: UUID, bytes: Int) {
        state.droppedGenerationID = generationID
        state.droppedMeasurementBatches &+= 1
        state.droppedMeasurementBytes &+= UInt64(bytes)
    }

    func next(matching requestID: UUID) async throws -> DecodeServiceEvent {
        try await Task.detached(priority: .userInitiated) { [self] in
            try waitForEvent(matching: requestID)
        }.value
    }

    private func waitForEvent(matching requestID: UUID) throws -> DecodeServiceEvent {
        condition.lock()
        defer { condition.unlock() }
        while state.pending[requestID]?.isEmpty != false,
              state.terminalError == nil {
            condition.wait()
        }
        if state.terminalDominatesPending {
            throw state.terminalError ?? DecodeFrameError.unexpectedEOF
        }
        if var events = state.pending[requestID], !events.isEmpty {
            let event = events.removeFirst()
            if let json = event.measurementBatchJSON {
                state.measurementBytes -= json.utf8.count
                state.measurementBatches -= 1
                condition.broadcast()
            }
            if events.isEmpty {
                state.pending.removeValue(forKey: requestID)
            } else {
                state.pending[requestID] = events
            }
            return event
        }
        throw state.terminalError ?? DecodeFrameError.unexpectedEOF
    }

    private func readFrames(from output: FileHandle) {
        defer {
            condition.lock()
            state.transportClosed = true
            let retired = state.output
            state.output = nil
            condition.broadcast()
            condition.unlock()
            try? retired?.close()
        }
        do {
            while true {
                // This thread lives for the service connection. Drain Foundation's
                // temporary read/decode objects after each frame, not at thread exit.
                try autoreleasepool {
                    var event = try DecodeFrameCodec.read(
                        DecodeServiceEvent.self, from: output)
                    condition.lock()
                    guard !state.streamClosed, state.terminalError == nil else {
                        condition.unlock()
                        throw DecodeFrameError.unexpectedEOF
                    }
                    // Only measurement payloads have this extra queue budget.
                    // Inference text, progress, terminal order and sequence stay intact.
                    if let json = event.measurementBatchJSON {
                        let bytes = json.utf8.count
                        let requiresDelivery = event.measurementContainsFooter == true
                            || event.measurementFinal == true
                        if requiresDelivery,
                           bytes <= DecodeRuntimeMeasurementLimits.maximumBatchBytes {
                            while state.measurementReceivers.contains(event.generationID),
                                  !state.streamClosed, state.terminalError == nil,
                                  state.measurementBytes + bytes > DecodeRuntimeMeasurementLimits.maximumQueuedBytes
                                    || state.measurementBatches >= DecodeRuntimeMeasurementLimits.maximumQueuedBatches {
                                condition.wait()
                            }
                        }
                        if state.streamClosed || state.terminalError != nil {
                            condition.unlock()
                            throw DecodeFrameError.unexpectedEOF
                        }
                        let fits = bytes <= DecodeRuntimeMeasurementLimits.maximumBatchBytes
                            && state.measurementBytes + bytes <= DecodeRuntimeMeasurementLimits.maximumQueuedBytes
                            && state.measurementBatches < DecodeRuntimeMeasurementLimits.maximumQueuedBatches
                            && state.measurementReceivers.contains(event.generationID)
                        if !fits {
                            // The service finishes one writer before starting the
                            // next request. Its next ordinary event (at latest the
                            // terminal) carries this fixed-size loss summary.
                            recordMeasurementDropLocked(generationID: event.generationID, bytes: bytes)
                            condition.unlock()
                            return
                        }
                        state.measurementBytes += bytes
                        state.measurementBatches += 1
                    }
                    if state.droppedGenerationID == event.generationID {
                        event.measurementDroppedBatches =
                            (event.measurementDroppedBatches ?? 0) &+ state.droppedMeasurementBatches
                        event.measurementDroppedBytes =
                            (event.measurementDroppedBytes ?? 0) &+ state.droppedMeasurementBytes
                        state.droppedGenerationID = nil
                        state.droppedMeasurementBatches = 0
                        state.droppedMeasurementBytes = 0
                    }
                    state.pending[event.generationID, default: []].append(event)
                    condition.broadcast()
                    condition.unlock()
                    onEventQueued(event.generationID)
                }
            }
        } catch {
            condition.lock()
            let shouldNotify = state.terminalError == nil
            if shouldNotify { state.terminalError = error }
            condition.broadcast()
            condition.unlock()
            if shouldNotify { onTerminate(self, error) }
        }
    }
}
