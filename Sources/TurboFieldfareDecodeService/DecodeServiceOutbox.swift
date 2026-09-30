import Foundation
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

final class DecodeServiceOutbox: @unchecked Sendable {
    private struct PrefillProgress {
        var done: Int
        var total: Int
    }

    private struct State {
        var pendingText = ""
        var latestPrefill: PrefillProgress?
        var latestToken: AppTokenEvent?
        var pendingToolCalls: [AppToolCall] = []
        var terminal: DecodeServiceEvent?
        var terminalCommitted = false
        var finished = false
        var sequence: UInt64 = 0
    }

    private let condition = NSCondition()
    private var state = State()
    private let generationID: UUID
    private let loadedFamily: DecodeModelFamily
    private let loadID: UUID
    private let modelIdentity: DecodeModelIdentity?
    private let sourceIdentity: DecodeSourceIdentity?
    private let conversationEpoch: UUID?
    private let memorySampler = AppMemorySampler()
    private let measurementRequest: DecodeRuntimeMeasurementRequest?
    private let measurementCapture: RuntimeMeasurementCapture?
    // Writer-owned fixed-size transport totals. No batches are queued here.
    private var measurementBatches: UInt64 = 0
    private var measurementBytes: UInt64 = 0
    private var measurementEncodeNanoseconds: UInt64 = 0
    private var measurementWriteNanoseconds: UInt64 = 0
    private var measurementDroppedBatches: UInt64 = 0
    private var measurementDroppedBytes: UInt64 = 0

    /// Bytes of image tower currently held mapped, or nil when there is no
    /// vision runtime. Sampled per event so Keep Ready is visible while a run
    /// is happening, not only in its final diagnostics.
    private let towerBytes: @Sendable () -> UInt64?
    /// Nil outside conversation mode, so a one-shot turn reports no gauge
    /// rather than a misleading zero.
    private let conversationTokens: @Sendable () -> Int?
    /// Logical bytes in the committed conversation state, when available.
    private let conversationLogicalStateBytes: @Sendable () -> UInt64?
    /// Bytes of actual routed-expert cache buffers currently owned.
    private let expertCacheBytes: @Sendable () -> UInt64?

    init(generationID: UUID,
         loadedFamily: DecodeModelFamily,
         loadID: UUID,
         modelIdentity: DecodeModelIdentity?,
         sourceIdentity: DecodeSourceIdentity? = nil,
         conversationEpoch: UUID? = nil,
         towerBytes: @escaping @Sendable () -> UInt64? = { nil },
         conversationTokens: @escaping @Sendable () -> Int? = { nil },
         conversationLogicalStateBytes: @escaping @Sendable () -> UInt64? = { nil },
         expertCacheBytes: @escaping @Sendable () -> UInt64? = { nil },
         measurementRequest: DecodeRuntimeMeasurementRequest? = nil,
         measurementCapture: RuntimeMeasurementCapture? = nil) {
        self.conversationTokens = conversationTokens
        self.conversationLogicalStateBytes = conversationLogicalStateBytes
        self.expertCacheBytes = expertCacheBytes
        self.generationID = generationID
        self.loadedFamily = loadedFamily
        self.loadID = loadID
        self.modelIdentity = modelIdentity
        self.sourceIdentity = sourceIdentity
        self.conversationEpoch = conversationEpoch
        self.towerBytes = towerBytes
        self.measurementRequest = measurementRequest
        self.measurementCapture = measurementCapture
        memorySampler.resetPeak()
    }

    func publish(_ event: AppInferenceEvent) {
        condition.lock()
        switch event {
        case .memorySample:
            // The writer samples on its own schedule; an inbound reading is the
            // runtime's, and there is nothing to queue.
            break
        case .prefillProgress(let done, let total):
            state.latestPrefill = PrefillProgress(done: done, total: total)
            condition.signal()
        case .token(let token):
            state.pendingText += token.textDelta
            state.latestToken = token
        case .toolCall(let call):
            state.pendingToolCalls.append(call)
            condition.signal()
        case .finished(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.finished, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .cancelled(let diagnostics):
            if !state.terminalCommitted {
                state.terminal = terminal(.cancelled, diagnostics: diagnostics)
                state.terminalCommitted = true
            }
        case .failed(let error, let diagnostics):
            if !state.terminalCommitted {
                // A broken lineage is not the same terminal state as a failed
                // turn. The turn can be retried; the conversation cannot, and
                // the app has to clear it rather than offer a retry that would
                // fail identically forever.
                let kind: DecodeServiceEventKind
                if case .conversationLineageLost = error { kind = .lineageLost }
                else { kind = .failed }
                state.terminal = terminal(
                    kind, diagnostics: diagnostics, error: error.userMessage)
                if case .structuredToolFailure(_, let canRegenerate, let evidence) = error {
                    state.terminal?.parserFailureCanRegenerateToolResult = canRegenerate
                    if let evidence, let data = try? JSONEncoder().encode(evidence) {
                        state.terminal?.parserFailureJSON = String(data: data, encoding: .utf8)
                    }
                }
                if case .repeatedThought(let recovery) = error,
                   let data = try? JSONEncoder().encode(recovery) {
                    state.terminal?.thoughtRepetitionRecoveryJSON = String(data: data, encoding: .utf8)
                }
                state.terminalCommitted = true
            }
        }
        if state.terminal != nil { condition.signal() }
        condition.unlock()
    }

    func finish(error: Error? = nil) {
        condition.lock()
        if !state.terminalCommitted, let error {
            state.terminal = DecodeServiceEvent(
                kind: .failed, generationID: generationID,
                loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
                error: "\(error)",
                conversationLogicalStateBytes: conversationLogicalStateBytes(),
                expertCacheBytes: expertCacheBytes(),
                conversationEpoch: conversationEpoch)
            state.terminalCommitted = true
        }
        state.finished = true
        condition.broadcast()
        condition.unlock()
    }

    func runWriter(to handle: FileHandle) throws {
        while true {
            condition.lock()
            if !state.finished, state.terminal == nil || measurementRequest != nil {
                _ = condition.wait(until: Date().addingTimeInterval(0.1))
            }
            let prefill = state.latestPrefill
            let text = state.pendingText
            let token = state.latestToken
            let toolCalls = state.pendingToolCalls
            // Keep the collector alive through producer cleanup and drain it
            // before the terminal event, which ends the app's receive loop.
            // A recovery receipt also waits for Entry to finish the failed
            // stream without committing its pending tool admission.
            let waitsForProducer = measurementRequest != nil
                || state.terminal?.thoughtRepetitionRecoveryJSON != nil
            let terminal = !waitsForProducer || state.finished ? state.terminal : nil
            let done = state.finished
            state.latestPrefill = nil
            state.pendingText = ""
            state.latestToken = nil
            state.pendingToolCalls = []
            if terminal != nil { state.terminal = nil }
            var prefillSequence: UInt64?
            if prefill != nil {
                state.sequence &+= 1
                prefillSequence = state.sequence
            }
            var tokenSequence: UInt64?
            if !text.isEmpty || token != nil {
                state.sequence &+= 1
                tokenSequence = state.sequence
            }
            condition.unlock()

            let currentMemoryBytes = memorySampler.sample()
            if let measurementCapture {
                measurementCapture.recordMemorySample(
                    currentBytes: currentMemoryBytes,
                    peakBytes: memorySampler.peakBytes,
                    towerBytes: towerBytes())
                if terminal != nil || done {
                    let status: UInt64
                    if terminal?.kind == .cancelled || terminal?.stopReason == "cancelled" {
                        status = 2
                    } else if terminal?.kind == .failed || terminal?.kind == .lineageLost {
                        status = 1
                    } else {
                        status = 0
                    }
                    measurementCapture.finish(status: status)
                    while let batch = measurementCapture.drainJSONBatch(
                        maximumBytes: DecodeRuntimeMeasurementLimits.maximumBatchBytes) {
                        try writeMeasurementBatch(
                            batch.data, containsFooter: batch.containsFooter, to: handle)
                    }
                } else if let batch = measurementCapture.drainJSONBatch(
                    maximumBytes: DecodeRuntimeMeasurementLimits.maximumBatchBytes) {
                    try writeMeasurementBatch(
                        batch.data, containsFooter: batch.containsFooter, to: handle)
                }
            }

            if prefill == nil, text.isEmpty, token == nil, toolCalls.isEmpty,
               terminal == nil, !done {
                let snapshot = DecodeServiceEvent(
                    kind: .memory, generationID: generationID,
                    loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes(),
                    conversationLogicalStateBytes: conversationLogicalStateBytes(),
                    expertCacheBytes: expertCacheBytes(),
                    conversationEpoch: conversationEpoch)
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
                continue
            }
            if let prefill, let prefillSequence {
                let snapshot = DecodeServiceEvent(
                    kind: .prefill, generationID: generationID,
                    loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
                    sequence: prefillSequence,
                    prefillDone: prefill.done, prefillTotal: prefill.total,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes(),
                    conversationLogicalStateBytes: conversationLogicalStateBytes(),
                    expertCacheBytes: expertCacheBytes(),
                    conversationEpoch: conversationEpoch)
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
            }
            if !text.isEmpty || token != nil {
                let elapsed = token?.elapsedDecodeSeconds ?? 0
                let count = (token?.index ?? -1) + 1
                let snapshot = DecodeServiceEvent(
                    kind: .snapshot, generationID: generationID,
                    loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
                    sequence: tokenSequence ?? 0, textDelta: text, tokenCount: count,
                    decodeSeconds: elapsed,
                    tokensPerSecond: elapsed > 0 ? Double(count) / elapsed : 0,
                    currentMemoryBytes: memorySampler.sample(),
                    peakMemoryBytes: memorySampler.peakBytes,
                    visionTowerMappedBytes: towerBytes(),
                    conversationLogicalStateBytes: conversationLogicalStateBytes(),
                    expertCacheBytes: expertCacheBytes(),
                    conversationEpoch: conversationEpoch,
                    structuredProgress: token?.structuredProgress,
                    thinkingPreview: token?.thinkingPreview,
                    toolCallPreview: token?.toolCallPreview)
                try handle.write(contentsOf: DecodeFrameCodec.encode(snapshot))
            }
            for call in toolCalls {
                let event = DecodeServiceEvent(
                    kind: .toolCall,
                    generationID: generationID,
                    loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
                    conversationEpoch: conversationEpoch,
                    toolCall: DecodeToolCall(
                        id: call.id,
                        name: call.name,
                        argumentsJSON: (try? call.arguments.encoded()) ?? "{}"))
                try handle.write(contentsOf: DecodeFrameCodec.encode(event))
            }
            if var terminal {
                if measurementRequest != nil {
                    terminal.currentMemoryBytes = currentMemoryBytes
                    terminal.peakMemoryBytes = memorySampler.peakBytes
                }
                if measurementRequest != nil { try writeMeasurementTransportSummary(to: handle) }
                try handle.write(contentsOf: DecodeFrameCodec.encode(terminal))
            }
            if terminal != nil { return }
            if done {
                if measurementRequest != nil { try writeMeasurementTransportSummary(to: handle) }
                return
            }
        }
    }

    private func writeMeasurementBatch(
        _ data: Data, containsFooter: Bool, to handle: FileHandle
    ) throws {
        guard let measurementRequest else { return }
        guard data.count <= DecodeRuntimeMeasurementLimits.maximumBatchBytes,
              let json = String(data: data, encoding: .utf8) else {
            measurementDroppedBatches &+= 1
            measurementDroppedBytes &+= UInt64(data.count)
            return
        }
        try autoreleasepool {
            let encodeStart = DispatchTime.now().uptimeNanoseconds
            var event = DecodeServiceEvent(
                kind: .measurement, generationID: generationID,
                loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
                conversationEpoch: conversationEpoch)
            event.measurementCaptureID = measurementRequest.stepID
            event.measurementBatchJSON = json
            event.measurementContainsFooter = containsFooter ? true : nil
            let frame = try DecodeFrameCodec.encode(event)
            let writeStart = DispatchTime.now().uptimeNanoseconds
            measurementEncodeNanoseconds &+= writeStart &- encodeStart
            try handle.write(contentsOf: frame)
            measurementWriteNanoseconds &+= DispatchTime.now().uptimeNanoseconds &- writeStart
            measurementBatches &+= 1
            measurementBytes &+= UInt64(data.count)
        }
    }

    private func writeMeasurementTransportSummary(to handle: FileHandle) throws {
        guard let measurementRequest else { return }
        let totals: [String: UInt64] = [
            "transport_summary": 1,
            "unsupported_prefill_configuration": measurementCapture == nil ? 1 : 0,
            "collector_allocated_storage_bytes": UInt64(measurementCapture?.allocatedStorageBytes ?? 0),
            "batches": measurementBatches,
            "numeric_json_bytes": measurementBytes,
            "frame_encode_wall_nanoseconds": measurementEncodeNanoseconds,
            "socket_write_wall_nanoseconds": measurementWriteNanoseconds,
            "dropped_batches": measurementDroppedBatches,
            "dropped_bytes": measurementDroppedBytes,
            "app_queue_byte_limit": UInt64(DecodeRuntimeMeasurementLimits.maximumQueuedBytes),
        ]
        let data = try JSONEncoder().encode(totals)
        var event = DecodeServiceEvent(
            kind: .measurement, generationID: generationID,
            loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
            conversationEpoch: conversationEpoch)
        event.measurementCaptureID = measurementRequest.stepID
        event.measurementBatchJSON = String(decoding: data, as: UTF8.self)
        event.measurementFinal = true
        try handle.write(contentsOf: DecodeFrameCodec.encode(event))
    }

    private func terminal(_ kind: DecodeServiceEventKind,
                          diagnostics: AppDiagnostics?,
                          error: String? = nil) -> DecodeServiceEvent {
        DecodeServiceEvent(
            kind: kind, generationID: generationID,
            loadedFamily: loadedFamily, loadID: loadID, modelIdentity: modelIdentity, sourceIdentity: sourceIdentity,
            tokenCount: diagnostics?.generatedTokens ?? 0,
            promptTokenCount: diagnostics?.promptTokenCount,
            computedPrefillTokens: diagnostics?.computedPrefillTokens,
            prefillSeconds: diagnostics?.prefillSeconds,
            timeToFirstTokenSeconds: diagnostics?.timeToFirstTokenSeconds,
            decodeSeconds: diagnostics?.decodeSeconds ?? 0,
            tokensPerSecond: diagnostics?.tokensPerSecond ?? 0,
            stopReason: diagnostics?.stopReason.rawValue,
            error: error,
            currentMemoryBytes: memorySampler.sample(),
            peakMemoryBytes: memorySampler.peakBytes,
            visionTowerMappedBytes: diagnostics?.visionTowerMappedBytes,
            conversationLogicalStateBytes: diagnostics?.conversationLogicalStateBytes
                ?? conversationLogicalStateBytes(),
            expertCacheBytes: diagnostics?.expertCacheBytes ?? expertCacheBytes(),
            cachedPromptTokens: diagnostics?.cachedPromptTokens,
            conversationTokenCount: conversationTokens(),
            conversationEpoch: conversationEpoch,
            prefill: diagnostics?.prefill.map(Self.prefillDiagnostics),
            runner: diagnostics?.runner.map(Self.runnerDiagnostics),
            structuredProgress: diagnostics?.structuredProgress)
    }

    private static func prefillDiagnostics(_ value: PrefillExecutionDiagnostics)
        -> DecodePrefillDiagnostics {
        DecodePrefillDiagnostics(
            requestedMode: value.requestedMode.rawValue,
            executedMode: value.executedMode.rawValue,
            kvStorageMode: value.kvStorageMode?.rawValue,
            chunkCompleteness: value.chunkCompleteness.rawValue,
            unsupportedReason: value.unsupportedReason)
    }

    private static func runnerDiagnostics(_ value: AppRunnerDiagnostics)
        -> DecodeRunnerDiagnostics {
        DecodeRunnerDiagnostics(
            cb1MillisecondsPerToken: value.cb1MillisecondsPerToken,
            routerWaitMillisecondsPerToken: value.routerWaitMillisecondsPerToken,
            gpuCompletionTiming: value.gpuCompletionTiming,
            ioMillisecondsPerToken: value.ioMillisecondsPerToken,
            cb2MillisecondsPerToken: value.cb2MillisecondsPerToken,
            headMillisecondsPerToken: value.headMillisecondsPerToken,
            rdadviseMillisecondsPerToken: value.rdadviseMillisecondsPerToken,
            rdadviseCallsPerToken: value.rdadviseCallsPerToken,
            rdadviseMegabytesPerToken: value.rdadviseMegabytesPerToken,
            rdadviseSkippedPerToken: value.rdadviseSkippedPerToken,
            rdadviseFailures: value.rdadviseFailures)
    }
}
