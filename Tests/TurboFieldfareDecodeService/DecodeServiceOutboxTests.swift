import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareDecodeService
import TurboFieldfareDecodeProtocol

@Suite struct DecodeServiceOutboxTests {
    private let loadedFamily = DecodeModelFamily.qwen3_6
    private let loadID = UUID()
    private let modelIdentity = DecodeModelIdentity(
        family: .qwen3_6,
        modelID: "Qwen/Qwen3.6-35B-A3B",
        sourceRevision: "revision",
        formatMajor: 2,
        formatMinor: 0,
        sourceIndexSHA256: String(repeating: "a", count: 64),
        quantizationPolicySHA256: String(repeating: "b", count: 64),
        textManifestSHA256: String(repeating: "c", count: 64),
        quantization: [DecodeQuantizationIdentity(
            category: "recurrent_state", storage: "fp32")],
        vision: .unavailable)

    private func makeOutbox(
        generationID: UUID = UUID(),
        conversationEpoch: UUID? = nil,
        towerBytes: @escaping @Sendable () -> UInt64? = { nil },
        conversationLogicalStateBytes: @escaping @Sendable () -> UInt64? = { nil },
        expertCacheBytes: @escaping @Sendable () -> UInt64? = { nil },
        measurementRequest: DecodeRuntimeMeasurementRequest? = nil,
        measurementCapture: RuntimeMeasurementCapture? = nil
    ) -> DecodeServiceOutbox {
        DecodeServiceOutbox(
            generationID: generationID,
            loadedFamily: loadedFamily,
            loadID: loadID,
            modelIdentity: modelIdentity,
            conversationEpoch: conversationEpoch,
            towerBytes: towerBytes,
            conversationLogicalStateBytes: conversationLogicalStateBytes,
            expertCacheBytes: expertCacheBytes,
            measurementRequest: measurementRequest,
            measurementCapture: measurementCapture)
    }

    private func expectBindingStamps(
        _ event: DecodeServiceEvent,
        generationID: UUID? = nil,
        conversationEpoch: UUID? = nil,
        measurementStepID: UUID? = nil
    ) {
        if let generationID {
            #expect(event.generationID == generationID)
        }
        #expect(event.loadedFamily == loadedFamily)
        #expect(event.loadID == loadID)
        #expect(event.modelIdentity == modelIdentity)
        if let conversationEpoch {
            #expect(event.conversationEpoch == conversationEpoch)
        }
        if let measurementStepID {
            #expect(event.measurementCaptureID == measurementStepID)
        }
    }

    @Test func cancellationFollowedByThrownCancellationWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = makeOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .cancelled(diagnostics(stopReason: .cancelled)),
            finishError: AppInferenceError.cancelled)

        #expect(event.kind == .cancelled)
        #expect(event.generationID == generationID)
        #expect(event.computedPrefillTokens == 7)
    }

    @Test func failureFollowedByThrownErrorWritesOneTerminal() throws {
        let generationID = UUID()
        let outbox = makeOutbox(generationID: generationID)

        let event = try firstTerminal(
            from: outbox,
            published: .failed(.unknown("first"), partial: nil),
            finishError: AppInferenceError.unknown("second"))

        #expect(event.kind == .failed)
        #expect(event.generationID == generationID)
        #expect(event.error == "first")
    }

    @Test func abrokenLineageIsItsOwnTerminalKind() throws {
        let outbox = makeOutbox()
        let event = try firstTerminal(
            from: outbox,
            published: .failed(.conversationLineageLost("prefill cursor mismatch"),
                               partial: nil),
            finishError: AppInferenceError.cancelled)

        #expect(event.kind == .lineageLost)
        #expect(event.error?.contains("prefill cursor mismatch") == true)
        #expect(event.error?.contains("Start a new chat") == true)
    }

    @Test func anordinaryFailureStaysFailed() throws {
        let outbox = makeOutbox()
        let event = try firstTerminal(
            from: outbox,
            published: .failed(.unknown("something else"), partial: nil),
            finishError: AppInferenceError.cancelled)
        #expect(event.kind == .failed)
    }

    /// Image encoding produces no progress or tokens, so the outbox must emit
    /// memory-only events during that otherwise silent interval.
    @Test func aSilentGenerationStillReportsMemory() throws {
        let generationID = UUID()
        let outbox = makeOutbox(generationID: generationID)
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let first = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(first.kind == .memory)
        expectBindingStamps(first, generationID: generationID)
        #expect(try #require(first.currentMemoryBytes) > 0)

        let second = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(second.kind == .memory)
        expectBindingStamps(second, generationID: generationID)

        outbox.publish(.prefillProgress(done: 4, total: 8))
        var event = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while event.kind == .memory {
            event = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(event.kind == .prefill)
        expectBindingStamps(event, generationID: generationID)
        #expect(event.prefillDone == 4)
        #expect(event.currentMemoryBytes != nil)

        outbox.finish(error: AppInferenceError.cancelled)
        #expect(writerFinished.wait(timeout: .now() + 5) == .success)
    }

    @Test func liveEventsCarryTheImageTowerFigure() throws {
        let generationID = UUID()
        let outbox = makeOutbox(
            generationID: generationID,
            towerBytes: { 1_144_373_248 })
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let idle = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(idle.kind == .memory)
        expectBindingStamps(idle, generationID: generationID)
        #expect(idle.visionTowerMappedBytes == 1_144_373_248)

        outbox.publish(.prefillProgress(done: 1, total: 2))
        var event = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while event.kind == .memory {
            event = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(event.kind == .prefill)
        expectBindingStamps(event, generationID: generationID)
        #expect(event.visionTowerMappedBytes == 1_144_373_248)

        outbox.finish(error: AppInferenceError.cancelled)
        #expect(writerFinished.wait(timeout: .now() + 5) == .success)
    }

    @Test func liveAndTerminalEventsCarryLogicalStateAndExpertCacheBytes() throws {
        let generationID = UUID()
        let outbox = makeOutbox(
            generationID: generationID,
            conversationLogicalStateBytes: { 4_444 },
            expertCacheBytes: { 5_555 })
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        let memory = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        #expect(memory.kind == .memory)
        #expect(memory.conversationLogicalStateBytes == 4_444)
        #expect(memory.expertCacheBytes == 5_555)

        outbox.publish(.prefillProgress(done: 1, total: 2))
        var prefill = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while prefill.kind == .memory {
            prefill = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(prefill.kind == .prefill)
        #expect(prefill.conversationLogicalStateBytes == 4_444)
        #expect(prefill.expertCacheBytes == 5_555)

        outbox.publish(.finished(diagnostics(
            stopReason: .eos,
            conversationLogicalStateBytes: 0,
            expertCacheBytes: 0)))
        outbox.finish()
        var terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        while terminal.kind == .memory || terminal.kind == .prefill {
            terminal = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        }
        #expect(terminal.kind == .finished)
        #expect(terminal.conversationLogicalStateBytes == 0)
        #expect(terminal.expertCacheBytes == 0)
        #expect(writerFinished.wait(timeout: .now() + 2) == .success)
    }

    @Test func allEmittedKindsCarryTheBoundSessionStamps() throws {
        let generationID = UUID()
        let stepID = UUID()
        let conversationEpoch = UUID()
        let capture = RuntimeMeasurementCapture()
        let outbox = makeOutbox(
            generationID: generationID,
            conversationEpoch: conversationEpoch,
            measurementRequest: DecodeRuntimeMeasurementRequest(
                stepID: stepID, stepIndex: 0,
                requestStartRetainedTokens: 0, contextBucket: 0),
            measurementCapture: capture)
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()
        var sawMemory = false
        var sawPrefill = false
        var sawSnapshot = false
        var sawToolCall = false
        var sawMeasurement = false
        var sawFinished = false
        func account(_ event: DecodeServiceEvent) {
            switch event.kind {
            case .memory: sawMemory = true
            case .prefill: sawPrefill = true
            case .snapshot: sawSnapshot = true
            case .toolCall: sawToolCall = true
            case .measurement: sawMeasurement = true
            case .finished: sawFinished = true
            default: break
            }
            expectBindingStamps(
                event, generationID: generationID,
                conversationEpoch: conversationEpoch,
                measurementStepID: event.kind == .measurement ? stepID : nil)
        }
        account(try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading))
        capture.record(.memory, 1, 2, 3, 4, 5)
        outbox.publish(.prefillProgress(done: 1, total: 2))
        outbox.publish(.token(AppTokenEvent(
            index: 0, textDelta: "hello", elapsedDecodeSeconds: 0.1)))
        outbox.publish(.toolCall(AppToolCall(
            id: "call-1", name: "lookup",
            arguments: .object(["ok": .bool(true)]))))
        outbox.publish(.finished(diagnostics(stopReason: .maxTokens)))
        outbox.finish()

        while !sawFinished {
            account(try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading))
        }
        #expect(sawMemory)
        #expect(sawPrefill)
        #expect(sawSnapshot)
        #expect(sawToolCall)
        #expect(sawMeasurement)
        #expect(sawFinished)
        #expect(writerFinished.wait(timeout: .now() + 5) == .success)
    }

    @Test func measurementTransportSummaryIsOneFinalBoundedBatch() throws {
        let stepID = UUID()
        let outbox = makeOutbox(
            measurementRequest: DecodeRuntimeMeasurementRequest(
                stepID: stepID, stepIndex: 0,
                requestStartRetainedTokens: 0, contextBucket: 0))
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()
        outbox.finish(error: AppInferenceError.cancelled)

        var finalCount = 0
        var terminal: DecodeServiceEvent?
        while terminal == nil {
            let event = try DecodeFrameCodec.read(
                DecodeServiceEvent.self, from: pipe.fileHandleForReading)
            if event.kind == .measurement, event.measurementFinal == true {
                finalCount += 1
                #expect(event.measurementCaptureID == stepID)
                #expect(event.measurementBatchJSON != nil)
            }
            if event.kind == .failed || event.kind == .cancelled
                || event.kind == .finished || event.kind == .lineageLost {
                terminal = event
            }
        }
        #expect(finalCount == 1)
        #expect(terminal?.kind == .failed)
        #expect(writerFinished.wait(timeout: .now() + 2) == .success)
    }

    private func firstTerminal(
        from outbox: DecodeServiceOutbox,
        published event: AppInferenceEvent,
        finishError: Error
    ) throws -> DecodeServiceEvent {
        let pipe = Pipe()
        let writerFinished = DispatchSemaphore(value: 0)
        let writer = Thread {
            defer {
                try? pipe.fileHandleForWriting.close()
                writerFinished.signal()
            }
            try? outbox.runWriter(to: pipe.fileHandleForWriting)
        }
        writer.start()

        outbox.publish(event)
        let terminal = try DecodeFrameCodec.read(
            DecodeServiceEvent.self, from: pipe.fileHandleForReading)
        expectBindingStamps(terminal)
        outbox.finish(error: finishError)

        #expect(writerFinished.wait(timeout: .now() + 2) == .success)
        #expect(pipe.fileHandleForReading.readDataToEndOfFile().isEmpty)
        return terminal
    }

    private func diagnostics(
        stopReason: AppStopReason,
        conversationLogicalStateBytes: UInt64? = nil,
        expertCacheBytes: UInt64? = nil
    ) -> AppDiagnostics {
        AppDiagnostics(
            generatedTokens: 0,
            stopReason: stopReason,
            computedPrefillTokens: 7,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            conversationLogicalStateBytes: conversationLogicalStateBytes,
            expertCacheBytes: expertCacheBytes,
            runtimeOptions: AppRuntimeOptions())
    }
}
