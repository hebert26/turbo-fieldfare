import Darwin
import CryptoKit
import Foundation
import TurboFieldfare
import TurboFieldfareDecodeProtocol

/// Opt-in logical input/output capture for disposable local Agent Mode runs.
/// No raw MCP responses, image contents, or continuous token streams are collected.
/// Bounded unfinished tool drafts may be attached to periodic progress and errors.
actor AgentInferenceTrace {
    static let shared: AgentInferenceTrace? = {
        guard let path = ProcessInfo.processInfo.environment[
            "TURBOFIELDFARE_AGENT_TRACE_PATH"], path.hasPrefix("/") else {
            return nil
        }
        return AgentInferenceTrace(path: path)
    }()

    struct Step: Sendable {
        let id: UUID
        let index: Int
        let conversation: UUID?
        let turn: Int?
        let startedAt: ContinuousClock.Instant
        var measurementRequest: DecodeRuntimeMeasurementRequest?
    }

    private let file: FileHandle
    private let checkpointRequestPath: String
    private let measurementFile: FileHandle?
    private let measurementPausePath: String?
    private var measurementFileBytes: Int
    private var measurementStopped = false
    private var measurementDroppedBatches: UInt64 = 0
    private var measurementDroppedBytes: UInt64 = 0
    private var measurementBucketBytes: [Int] = []
    private var measurementBucketStopped: [Bool] = []
    private struct MeasurementReservation {
        let stepID: UUID
        var remainingFooterBytes: Int
        var detailStopped = false
        var generationID: UUID?
    }
    private var measurementReservation: MeasurementReservation?
    private static let maximumMeasurementWrapperBytes = 2_048
    private static var footerReservationBytes: Int {
        // Includes the collector's one possible mixed tail batch, all aggregate
        // rows, their wrappers, the small transport footer and terminal record.
        RuntimeMeasurementCapture.maximumSerializedFooterBytes
            + (RuntimeMeasurementCapture.maximumFooterBatchCount(
                maximumBytes: DecodeRuntimeMeasurementLimits.maximumBatchBytes) + 2)
                * maximumMeasurementWrapperBytes
    }
    private var confirmedConversation: UUID?
    private var confirmedRetainedTokens: Int?
    private var nextStepIndex = 0
    private var latestStep: Step?

    private init?(path: String) {
        let descriptor = Darwin.open(
            path, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC,
            S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { return nil }
        var info = stat()
        guard fstat(descriptor, &info) == 0,
              info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == geteuid(),
              fchmod(descriptor, S_IRUSR | S_IWUSR) == 0 else {
            Darwin.close(descriptor)
            return nil
        }
        file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        checkpointRequestPath = path + ".checkpoint.request"
        var captureFile: FileHandle?
        var captureBytes = 0
        if ProcessInfo.processInfo.environment[
            "TURBOFIELDFARE_RUNTIME_MEASUREMENT_CAPTURE"] == "1" {
            let captureDescriptor = Darwin.open(
                path + ".runtime-measurements.jsonl",
                O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW | O_CLOEXEC,
                S_IRUSR | S_IWUSR)
            if captureDescriptor >= 0 {
                var captureInfo = stat()
                if fstat(captureDescriptor, &captureInfo) == 0,
                   captureInfo.st_mode & S_IFMT == S_IFREG,
                   captureInfo.st_uid == geteuid(),
                   fchmod(captureDescriptor, S_IRUSR | S_IWUSR) == 0 {
                    captureFile = FileHandle(
                        fileDescriptor: captureDescriptor, closeOnDealloc: true)
                    captureBytes = Int(clamping: captureInfo.st_size)
                } else {
                    Darwin.close(captureDescriptor)
                }
            }
        }
        measurementFile = captureFile
        measurementPausePath = captureFile == nil ? nil : path + ".runtime-measurements.paused"
        measurementFileBytes = captureBytes
        measurementStopped = captureBytes >= DecodeRuntimeMeasurementLimits.maximumArtifactBytes - 4_096
        if captureFile != nil {
            measurementBucketBytes = [Int](repeating: 0, count: 4)
            measurementBucketStopped = [Bool](repeating: false, count: 4)
        }
        if captureFile != nil, captureBytes > 0, !measurementStopped {
            // Restore reservations when an existing trace path is reused. Each
            // read and line is bounded, and no complete capture is materialized.
            if let recovered = Self.recoverMeasurementBuckets(
                path: path + ".runtime-measurements.jsonl") {
                measurementBucketBytes = recovered.bytes
                measurementBucketStopped = recovered.stopped
            } else {
                measurementStopped = true
            }
        }
    }

    func runtimeMeasurementRequest(for step: Step?) -> DecodeRuntimeMeasurementRequest? {
        step?.measurementRequest
    }

    /// The socket receive owner calls this on actual exit, including failures
    /// before dispatch. The visible stream may have been cancelled much earlier.
    func runtimeMeasurementReceptionEnded(
        capture: DecodeRuntimeMeasurementRequest, conversation: UUID?, turn: Int?
    ) {
        guard measurementReservation?.stepID == capture.stepID else { return }
        appendMeasurementStop(
            capture: capture, conversation: conversation, turn: turn,
            generationID: measurementReservation?.generationID,
            reason: "receive_ended_without_measurement_terminal")
        measurementReservation = nil
    }

    private struct MeasurementEnvelope: Decodable {
        let event: String
        let context_bucket: Int?
        let context_bucket_capture_disabled: Bool?
    }

    private static func recoverMeasurementBuckets(path: String)
        -> (bytes: [Int], stopped: [Bool])? {
        let descriptor = Darwin.open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { return nil }
        let reader = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        var counts = [Int](repeating: 0, count: 4)
        var stopped = [Bool](repeating: false, count: 4)
        var buffer = Data()
        var readBytes = 0
        let maximumLine = DecodeRuntimeMeasurementLimits.maximumBatchBytes + 4_096
        do {
            while let chunk = try reader.read(upToCount: 32 * 1_024), !chunk.isEmpty {
                readBytes += chunk.count
                guard readBytes <= DecodeRuntimeMeasurementLimits.maximumArtifactBytes else { return nil }
                buffer.append(chunk)
                while let newline = buffer.firstIndex(of: 0x0a) {
                    let bytes = buffer.distance(from: buffer.startIndex, to: newline) + 1
                    guard bytes <= maximumLine else { return nil }
                    let envelope: MeasurementEnvelope? = autoreleasepool {
                        try? JSONDecoder().decode(
                            MeasurementEnvelope.self, from: Data(buffer[..<newline]))
                    }
                    guard let envelope else { return nil }
                    // Small stop notices use the shared reserved space. Numeric
                    // batches always carry one of the four exact bucket indices.
                    if let bucket = envelope.context_bucket {
                        guard (0..<4).contains(bucket) else { return nil }
                        if envelope.event == "capture_stopped" {
                            if envelope.context_bucket_capture_disabled == true { stopped[bucket] = true }
                        } else {
                            counts[bucket] += bytes
                            if counts[bucket] >= DecodeRuntimeMeasurementLimits.maximumContextBucketBytes {
                                stopped[bucket] = true
                            }
                        }
                    } else if envelope.event != "capture_stopped" { return nil }
                    buffer.removeSubrange(...newline)
                }
                guard buffer.count <= maximumLine else { return nil }
            }
            return buffer.isEmpty ? (counts, stopped) : nil
        } catch { return nil }
    }

    /// Called after the IPC generation check. The identity comes from the request,
    /// never `latestStep`, which may already refer to a later cancelled/retried turn.
    func runtimeMeasurementEvent(
        _ event: DecodeServiceEvent, capture: DecodeRuntimeMeasurementRequest,
        conversation: UUID?, turn: Int?
    ) {
        autoreleasepool {
            persistRuntimeMeasurementEvent(
                event, capture: capture, conversation: conversation, turn: turn)
        }
    }

    private func persistRuntimeMeasurementEvent(
        _ event: DecodeServiceEvent, capture: DecodeRuntimeMeasurementRequest,
        conversation: UUID?, turn: Int?
    ) {
        guard measurementFile != nil else { return }
        guard (0..<4).contains(capture.contextBucket) else { return }
        let terminal = event.kind == .finished || event.kind == .cancelled
            || event.kind == .failed || event.kind == .lineageLost
        guard event.kind == .measurement || terminal
            || event.measurementDroppedBatches != nil else { return }
        guard measurementReservation?.stepID == capture.stepID else { return }
        measurementReservation?.generationID = event.generationID
        defer {
            if terminal { measurementReservation = nil }
        }
        let requiresReservation = event.measurementContainsFooter == true
            || event.measurementFinal == true || terminal
        let batchBytes = event.measurementBatchJSON?.utf8.count ?? 0
        guard !measurementStopped else {
            measurementDroppedBatches &+= event.measurementDroppedBatches ?? 0
            measurementDroppedBytes &+= event.measurementDroppedBytes ?? 0
            if event.measurementBatchJSON != nil {
                measurementDroppedBatches &+= 1
                measurementDroppedBytes &+= UInt64(batchBytes)
            }
            if terminal {
                appendMeasurementStop(
                    capture: capture, conversation: conversation, turn: turn,
                    generationID: event.generationID,
                    reason: measurementStopped ? "artifact_limit_or_write_failure" : "context_bucket_byte_limit")
            }
            return
        }
        if measurementReservation?.detailStopped == true, !requiresReservation {
            measurementDroppedBatches &+= event.measurementDroppedBatches ?? 0
            measurementDroppedBytes &+= event.measurementDroppedBytes ?? 0
            if event.measurementBatchJSON != nil {
                measurementDroppedBatches &+= 1
                measurementDroppedBytes &+= UInt64(batchBytes)
            }
            return
        }
        var values: [String: JSONValue] = [
            "schema_version": .integer(1),
            "event": .string(terminal ? "terminal" : "runtime_measurements"),
            "step_id": .string(capture.stepID.uuidString),
            "step_index": Self.integer(capture.stepIndex),
            "generation_id": .string(event.generationID.uuidString),
            "context_bucket": Self.integer(capture.contextBucket),
            "request_start_retained_tokens": Self.integer(capture.requestStartRetainedTokens),
            "conversation_id": conversation.map { .string($0.uuidString) } ?? .null,
            "turn_index": Self.integer(turn),
            "timestamp_unix_seconds": .number(Date().timeIntervalSince1970),
        ]
        if let dropped = event.measurementDroppedBatches {
            values["transport_dropped_batches"] = .unsignedInteger(dropped)
            values["transport_dropped_bytes"] = .unsignedInteger(event.measurementDroppedBytes ?? 0)
        }
        if terminal {
            values["terminal_kind"] = .string(event.kind.rawValue)
            values["stop_reason"] = event.stopReason.map(JSONValue.string) ?? .null
            values["current_memory_bytes"] = event.currentMemoryBytes.map(JSONValue.unsignedInteger) ?? .null
            values["sampled_peak_memory_bytes"] = event.peakMemoryBytes.map(JSONValue.unsignedInteger) ?? .null
            if measurementReservation?.detailStopped == true {
                values["detail_capture_stopped"] = .bool(true)
                values["capture_dropped_batches_total"] = .unsignedInteger(measurementDroppedBatches)
                values["capture_dropped_bytes_total"] = .unsignedInteger(measurementDroppedBytes)
            }
        }
        let validBatch = event.kind == .measurement
            && event.measurementCaptureID == capture.stepID
            && batchBytes <= DecodeRuntimeMeasurementLimits.maximumBatchBytes
        if event.measurementBatchJSON != nil, !validBatch {
            values["rejected_batch_bytes"] = .unsignedInteger(UInt64(batchBytes))
        }
        if let final = event.measurementFinal { values["final_batch"] = .bool(final) }
        if let footer = event.measurementContainsFooter { values["contains_footer"] = .bool(footer) }
        guard var data = try? JSONEncoder().encode(JSONValue.object(values)) else { return }
        if validBatch, let batch = event.measurementBatchJSON {
            // The service's bounded numeric JSON is embedded directly. Avoid
            // decoding thousands of numeric fields into duplicate object trees.
            data.removeLast()
            data.append(contentsOf: ",\"batch\":".utf8)
            data.append(contentsOf: batch.utf8)
            data.append(0x7d)
        }
        data.append(0x0a)
        let remainingFooter = measurementReservation?.remainingFooterBytes ?? 0
        let cap = DecodeRuntimeMeasurementLimits.maximumArtifactBytes
        let wrapperBytes = data.count - (validBatch ? batchBytes : 0)
        let boundedEnvelopeBytes = event.measurementContainsFooter == true ? wrapperBytes : data.count
        if requiresReservation,
           data.count > remainingFooter || boundedEnvelopeBytes > Self.maximumMeasurementWrapperBytes {
            measurementStopped = true
            appendMeasurementStop(
                capture: capture, conversation: conversation, turn: turn,
                generationID: event.generationID, reason: "footer_reservation_exceeded")
            return
        }
        let protectedRemainder = requiresReservation ? 0 : remainingFooter
        if !requiresReservation,
           measurementBucketBytes[capture.contextBucket] + data.count + protectedRemainder
                > DecodeRuntimeMeasurementLimits.maximumContextBucketBytes
                || measurementFileBytes + data.count + protectedRemainder > cap {
            measurementBucketStopped[capture.contextBucket] = true
            measurementReservation?.detailStopped = true
            measurementDroppedBatches &+= 1
            measurementDroppedBytes &+= UInt64(batchBytes)
            appendMeasurementStop(
                capture: capture, conversation: conversation, turn: turn,
                generationID: event.generationID, reason: "context_bucket_byte_limit")
            return
        }
        guard measurementFileBytes + data.count <= cap,
              measurementBucketBytes[capture.contextBucket] + data.count
                <= DecodeRuntimeMeasurementLimits.maximumContextBucketBytes else {
            measurementStopped = true
            measurementDroppedBatches &+= 1
            measurementDroppedBytes &+= UInt64(batchBytes)
            appendMeasurementStop(
                capture: capture, conversation: conversation, turn: turn,
                generationID: event.generationID, reason: "artifact_byte_limit")
            return
        }
        do {
            try measurementFile?.write(contentsOf: data)
            measurementFileBytes += data.count
            measurementBucketBytes[capture.contextBucket] += data.count
            if requiresReservation {
                measurementReservation?.remainingFooterBytes -= data.count
            }
        } catch {
            measurementStopped = true
            appendMeasurementStop(
                capture: capture, conversation: conversation, turn: turn,
                generationID: event.generationID, reason: "artifact_write_failure")
        }
    }

    private func appendMeasurementStop(
        capture: DecodeRuntimeMeasurementRequest, conversation: UUID?, turn: Int?,
        generationID: UUID?, reason: String
    ) {
        let body: [String: JSONValue] = [
            "generation_id": generationID.map { .string($0.uuidString) } ?? .null,
            "reason": .string(reason),
            "artifact_bytes": Self.integer(measurementFileBytes),
            "context_bucket": Self.integer(capture.contextBucket),
            "context_bucket_bytes": Self.integer(measurementBucketBytes[capture.contextBucket]),
            "context_bucket_capture_disabled": .bool(measurementBucketStopped[capture.contextBucket]),
            "dropped_batches": .unsignedInteger(measurementDroppedBatches),
            "dropped_bytes": .unsignedInteger(measurementDroppedBytes),
            "subsequent_requests_capture_disabled": .bool(measurementStopped),
        ]
        let step = Step(id: capture.stepID, index: capture.stepIndex,
                        conversation: conversation, turn: turn, startedAt: .now)
        append(step: step, event: "runtime_measurement_capture_stopped", body: body,
               maximumBytes: 2_048)
        // Reserve enough artifact space to make truncation visible in the
        // numeric artifact as well as the existing trace, without exceeding cap.
        var record = body
        record["event"] = .string("capture_stopped")
        record["step_id"] = .string(capture.stepID.uuidString)
        if var data = try? JSONEncoder().encode(JSONValue.object(record)) {
            data.append(0x0a)
            let noticeBytes = measurementFileBytes - measurementBucketBytes.reduce(0, +)
            if noticeBytes + data.count <= 4_096,
               measurementFileBytes + data.count + (measurementReservation?.remainingFooterBytes ?? 0)
                <= DecodeRuntimeMeasurementLimits.maximumArtifactBytes {
                do {
                    try measurementFile?.write(contentsOf: data)
                    measurementFileBytes += data.count
                } catch { /* A failed private trace must not fail inference. */ }
            }
        }
    }

    func begin(_ request: AppGenerationRequest) -> Step {
        var step = Step(
            id: UUID(), index: nextStepIndex,
            conversation: request.conversationEpoch, turn: request.turnIndex,
            startedAt: .now)
        nextStepIndex += 1
        latestStep = step
        var input: [String: JSONValue] = [
            "prompt": .string(request.prompt),
            "image_attachment_count": Self.integer(request.imageAttachments.count),
            "model_directory": .string(request.modelDirectory.path),
            "settings": .object([
                "max_new_tokens": Self.integer(request.maxNewTokens),
                "max_context_tokens": Self.integer(request.maxContextTokens),
                "temperature": .number(Double(request.temperature)),
                "top_k": Self.integer(request.topK),
                "top_p": Self.number(request.topP.map(Double.init)),
                "repetition_penalty": .number(Double(request.repetitionPenalty)),
                "enable_thinking": .bool(request.runtimeOptions.toolThinkingEnabled),
                "expert_cache_slots": Self.integer(request.runtimeOptions.expertCacheSlots),
                "expert_cache_policy": .string(request.runtimeOptions.expertCachePolicy.rawValue),
                "prefill_enabled": .bool(request.runtimeOptions.prefillEnabled),
                "prefill_chunk_tokens": Self.integer(request.runtimeOptions.prefillChunkTokens),
                "rdadvise_policy": .string(request.runtimeOptions.rdadvisePolicy.rawValue),
                "model_verification": .string(request.runtimeOptions.modelVerification.rawValue),
                "vision_residency_policy": .string(request.runtimeOptions.visionResidencyPolicy.rawValue),
            ]),
        ]
        if measurementFile != nil {
            var retained: Int?
            if let conversation = step.conversation, conversation == confirmedConversation {
                retained = confirmedRetainedTokens
            } else if step.conversation != nil, step.turn == 0,
                      request.continuesConversation, case .user = request.toolTurn {
                // The app awaits the service's new-epoch reset before this
                // opening request. A tool-result continuation never enters here.
                retained = 0
            }
            let bucket: Int? = retained.flatMap { tokens in
                guard (0...65_536).contains(tokens) else { return nil }
                if tokens < 8_192 { return 0 }
                if tokens < 32_768 { return 1 }
                if tokens < 49_152 { return 2 }
                return 3
            }
            let disposition: String
            // Exactly one pause-file check per opted-in request. It does not
            // change any inference input or runtime setting.
            if measurementStopped { disposition = "artifact_unavailable_or_full" }
            else if measurementReservation != nil { disposition = "capture_request_in_flight" }
            else if let measurementPausePath,
                    FileManager.default.fileExists(atPath: measurementPausePath) {
                disposition = "paused"
            } else if let bucket, let retained {
                if measurementBucketStopped[bucket] { disposition = "context_bucket_full" }
                else if measurementBucketBytes[bucket] + Self.footerReservationBytes
                            > DecodeRuntimeMeasurementLimits.maximumContextBucketBytes
                    || measurementFileBytes + Self.footerReservationBytes
                            > DecodeRuntimeMeasurementLimits.maximumArtifactBytes {
                    disposition = "insufficient_footer_reservation"
                }
                else {
                    disposition = "enabled"
                    step.measurementRequest = DecodeRuntimeMeasurementRequest(
                        stepID: step.id, stepIndex: step.index,
                        requestStartRetainedTokens: retained, contextBucket: bucket)
                    measurementReservation = MeasurementReservation(
                        stepID: step.id, remainingFooterBytes: Self.footerReservationBytes)
                }
            } else { disposition = "unknown_request_start_context" }
            input["runtime_measurement_capture"] = .object([
                "disposition": .string(disposition),
                "request_start_retained_tokens": Self.integer(retained),
                "context_bucket": Self.integer(bucket),
                "context_bucket_byte_limit": Self.integer(DecodeRuntimeMeasurementLimits.maximumContextBucketBytes),
                "footer_reserved_bytes": Self.integer(step.measurementRequest == nil ? 0 : Self.footerReservationBytes),
            ])
        }
        switch request.toolTurn {
        case .checkpoint(let id):
            input["kind"] = .string("checkpoint_resume")
            input["checkpoint_id"] = .string(id.uuidString)
        case .user(let developerPrompt, let tools):
            input["kind"] = .string("user")
            input["developer_prompt"] = developerPrompt.map(JSONValue.string) ?? .null
            input["tools"] = .array(tools.map { tool in
                .object([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": tool.parameters,
                ])
            })
        case .results(let results):
            input["kind"] = .string("tool_results")
            input["results"] = .array(results.map { result in
                var value: [String: JSONValue] = [
                    "call_id": .string(result.callID),
                    "name": .string(result.name),
                    "content": .string(result.content),
                ]
                if !result.imageAttachments.isEmpty {
                    value["image_attachments"] = .array(result.imageAttachments.map {
                        .object([
                            "sha256": .string($0.sha256),
                            "encoded_bytes": Self.integer($0.encodedBytes),
                        ])
                    })
                }
                return .object(value)
            })
        case nil:
            input["kind"] = .string("user")
        }
        append(step: step, event: "input", body: ["input": .object(input)])
        return step
    }

    func finish(
        _ step: Step?, content: String, calls: [AppToolCall],
        diagnostics: AppDiagnostics?, error: String? = nil,
        cancelled: Bool = false,
        parserFailure: StructuredToolFailureEvidence? = nil,
        thoughtRepetitionRecovery: ThoughtRepetitionRecovery? = nil,
        structuredProgress: DecodeStructuredProgress? = nil,
        toolCallPreview: DecodeToolCallPreview? = nil
    ) {
        guard let step else { return }
        // Logical completion does not end the socket receiver. In particular,
        // Stop can finish this trace before its protected numeric footer arrives.
        if measurementFile != nil {
            confirmedConversation = step.conversation
            // Only a successful service result confirms retained positions.
            // A rollback, missing result or lost lineage is recorded as unknown.
            confirmedRetainedTokens = error == nil && !cancelled
                ? diagnostics?.conversationTokens : nil
        }
        let elapsed = step.startedAt.duration(to: .now).components
        var output: [String: JSONValue] = [
            "content": .string(content),
            "tool_calls": .array(calls.map { call in
                .object([
                    "id": .string(call.id),
                    "name": .string(call.name),
                    "arguments": call.arguments,
                ])
            }),
            "elapsed_seconds": .number(
                Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18),
        ]
        if let diagnostics {
            var values: [String: JSONValue] = [
                "stop_reason": .string(diagnostics.stopReason.rawValue),
                "generated_tokens": Self.integer(diagnostics.generatedTokens),
                "prompt_tokens": Self.integer(diagnostics.promptTokenCount),
                "cached_prompt_tokens": Self.integer(diagnostics.cachedPromptTokens),
                "computed_prefill_tokens": Self.integer(diagnostics.computedPrefillTokens),
                "conversation_tokens": Self.integer(diagnostics.conversationTokens),
                "prefill_seconds": Self.number(diagnostics.prefillSeconds),
                "time_to_first_token_seconds": Self.number(diagnostics.timeToFirstTokenSeconds),
                "decode_seconds": Self.number(diagnostics.decodeSeconds),
                "tokens_per_second": Self.number(diagnostics.tokensPerSecond),
            ]
            if let runner = diagnostics.runner, diagnostics.generatedTokens > 1 {
                var runnerValues: [String: JSONValue] = [
                    "decode_forward_count": Self.integer(diagnostics.generatedTokens - 1),
                    "cb1_milliseconds_per_forward": Self.number(runner.cb1MillisecondsPerToken),
                    "router_wait_milliseconds_per_forward": Self.number(runner.routerWaitMillisecondsPerToken),
                    "io_milliseconds_per_forward": Self.number(runner.ioMillisecondsPerToken),
                    "cb2_milliseconds_per_forward": Self.number(runner.cb2MillisecondsPerToken),
                    "head_milliseconds_per_forward": Self.number(runner.headMillisecondsPerToken),
                ]
                if let timing = runner.gpuCompletionTiming {
                    runnerValues["gpu_completion_timing"] = .object(timing.mapValues { group in
                        .object([
                            "milliseconds_per_forward": Self.number(group.millisecondsPerForward),
                            "valid_buffer_count": .unsignedInteger(group.validCount),
                            "expected_buffer_count": .unsignedInteger(group.expectedCount),
                        ])
                    })
                }
                values["runner"] = .object(runnerValues)
            }
            output["diagnostics"] = .object(values)
        }
        if let progress = diagnostics?.structuredProgress ?? structuredProgress {
            output["structured_progress"] = Self.progressSummary(progress)
        }
        if let error { output["error"] = .string(error) }
        if error != nil || cancelled, let toolCallPreview {
            output["raw_tool_call_draft"] = Self.toolCallDraft(toolCallPreview)
        }
        if error != nil, let parserFailure,
           let data = try? JSONEncoder().encode(parserFailure),
           let value = try? JSONDecoder().decode(JSONValue.self, from: data) {
            output["parser_failure"] = value
        }
        if let thoughtRepetitionRecovery,
           let data = try? JSONEncoder().encode(thoughtRepetitionRecovery),
           let value = try? JSONDecoder().decode(JSONValue.self, from: data) {
            output["thought_repetition_recovery"] = value
        }
        append(
            step: step, event: cancelled ? "cancelled" : error == nil ? "output" : "error",
            body: ["output": .object(output)])
    }

    func generationProgress(
        _ step: Step?, progress: DecodeStructuredProgress?, tokens: Int,
        elapsedSeconds: Double, toolCallPreview: DecodeToolCallPreview? = nil
    ) {
        guard let step else { return }
        var body: [String: JSONValue] = [
            "generated_tokens": Self.integer(tokens),
            "decode_seconds": Self.number(elapsedSeconds),
        ]
        if let progress { body["structured_progress"] = Self.progressSummary(progress) }
        if let toolCallPreview {
            body["raw_tool_call_draft"] = Self.toolCallDraft(toolCallPreview)
        }
        append(step: step, event: "generation_progress", body: body)
    }

    private static func toolCallDraft(_ value: DecodeToolCallPreview) -> JSONValue {
        .object([
            "text": .string(value.text),
            "middle_text_omitted": .bool(value.middleTextOmitted),
        ])
    }

    private static func progressSummary(_ value: DecodeStructuredProgress) -> JSONValue {
        .object([
            "stage": .string(value.stage),
            "thinking_tokens": Self.integer(value.thinkingTokens),
            "tool_call_tokens": Self.integer(value.toolCallTokens),
            "visible_response_tokens": Self.integer(value.visibleResponseTokens),
            "channel_label_tokens": Self.integer(value.channelLabelTokens),
            "unknown_hidden_channel_tokens": Self.integer(value.unknownHiddenChannelTokens),
        ])
    }

    func currentStep() -> Step? { latestStep }

    /// Local one-use diagnostic, available only when an explicit trace opened.
    /// Claim the pathname first, then validate and read that exact descriptor.
    /// Invalid files are restored where possible, never silently discarded.
    func consumeCheckpointRequest() -> Bool {
        let claimed = checkpointRequestPath + ".claimed." + UUID().uuidString
        guard renameatx_np(AT_FDCWD, checkpointRequestPath, AT_FDCWD, claimed, UInt32(RENAME_EXCL)) == 0 else {
            return false
        }
        var consumed = false
        defer {
            if !consumed {
                _ = renameatx_np(AT_FDCWD, claimed, AT_FDCWD, checkpointRequestPath, UInt32(RENAME_EXCL))
            }
        }
        let descriptor = Darwin.open(claimed, O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { return false }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var info = stat()
        guard fstat(descriptor, &info) == 0,
              info.st_mode & S_IFMT == S_IFREG, info.st_uid == geteuid(),
              info.st_mode & 0o777 == 0o600, info.st_size == 11,
              let data = try? handle.read(upToCount: 12),
              data == Data("checkpoint\n".utf8) else { return false }
        var pathInfo = stat()
        guard lstat(claimed, &pathInfo) == 0, pathInfo.st_dev == info.st_dev,
              pathInfo.st_ino == info.st_ino, unlink(claimed) == 0 else { return false }
        consumed = true
        if let step = latestStep {
            append(step: step, event: "forced_context_checkpoint_requested", body: ["one_use": .bool(true)])
        }
        return true
    }

    func checkpoint(_ receipt: DecodeContextCheckpointReceipt, callID: String,
                    sourceEpoch: UUID, trigger: DecodeContextCheckpointTrigger,
                    performanceEvidence: DecodePerformanceCheckpointEvidence?) {
        if receipt.committed {
            confirmedConversation = receipt.replacementEpoch
            confirmedRetainedTokens = 0
        }
        guard let step = latestStep else { return }
        var body: [String: JSONValue] = [
            "checkpoint_id": .string(receipt.checkpointID.uuidString),
            "settled_call_id": .string(callID),
            "source_epoch": .string(sourceEpoch.uuidString),
            "replacement_epoch": .string(receipt.replacementEpoch.uuidString),
            "committed": .bool(receipt.committed), "needed": .bool(receipt.needed),
            "trigger": .string(trigger.rawValue),
            "existing_prompt_tokens": Self.integer(receipt.existingPromptTokens),
            "replacement_prompt_tokens": Self.integer(receipt.replacementPromptTokens),
            "forecast_reserve_tokens": Self.integer(receipt.reserveTokens),
            "next_result_allowance_tokens": Self.integer(receipt.resultAllowanceTokens),
            "retained_image_count": Self.integer(receipt.retainedImageCount),
            "retained_image_rows": Self.integer(receipt.retainedImageRows),
            "retained_feature_bytes": Self.integer(receipt.retainedFeatureBytes),
            "performance_minimum_savings_tokens": Self.integer(
                receipt.performanceMinimumSavingsTokens),
            "released_feature_bytes": .integer(0),
            "preparation_seconds": .number(receipt.preparationSeconds),
            "rebuild_timing": .string("reported by the following checkpoint_resume prefill"),
            "latency_prediction": .null,
        ]
        if let performanceEvidence {
            body["performance_window_decisions"] = Self.integer(
                performanceEvidence.completedDecisions)
            body["performance_window_generated_tokens"] = Self.integer(
                performanceEvidence.generatedTokens)
            body["performance_window_decode_seconds"] = .number(
                performanceEvidence.decodeSeconds)
            body["performance_window_context_tokens"] = Self.integer(
                performanceEvidence.conversationTokens)
            body["performance_window_weighted_tokens_per_second"] = .number(
                performanceEvidence.weightedTokensPerSecond)
        }
        append(step: step, event: "context_checkpoint", body: body)
    }

    func checkpointCompleted(
        id: UUID, before: DecodePerformanceCheckpointEvidence?,
        replacementPromptTokens: Int, after: AppDiagnostics
    ) {
        guard let step = latestStep else { return }
        let rate = after.generatedTokens > 0 && after.decodeSeconds.isFinite
            && after.decodeSeconds > 0
            ? Double(after.generatedTokens) / after.decodeSeconds : nil
        append(step: step, event: "performance_context_checkpoint_completed", body: [
            "checkpoint_id": .string(id.uuidString),
            "trigger": .string(DecodeContextCheckpointTrigger.sustainedSlowDecode.rawValue),
            "before_context_tokens": Self.integer(before?.conversationTokens),
            "before_weighted_tokens_per_second": before.map {
                .number($0.weightedTokensPerSecond)
            } ?? .null,
            "replacement_prompt_tokens": Self.integer(replacementPromptTokens),
            "after_context_tokens": Self.integer(after.conversationTokens),
            "after_generated_tokens": Self.integer(after.generatedTokens),
            "after_decode_seconds": .number(after.decodeSeconds),
            "after_tokens_per_second": rate.map(JSONValue.number) ?? .null,
            "measurement_scope": .string("first completed decision after rebuild"),
        ])
    }

    func checkpointRead(id: UUID, seconds: Double) {
        guard let step = latestStep else { return }
        append(step: step, event: "context_checkpoint_read", body: [
            "checkpoint_id": .string(id.uuidString), "read_seconds": .number(seconds)])
    }

    func checkpointFailure(id: UUID, commit: Bool, error: String) {
        guard let step = latestStep else { return }
        append(step: step, event: "context_checkpoint_failed", body: [
            "checkpoint_id": .string(id.uuidString), "commit_requested": .bool(commit),
            "error": .string(error)], maximumBytes: 8_192)
    }

    /// Diagnostic-only projection of an already-returned failure. It is captured
    /// before identity validation, so none of these fields grant execution rights.
    func mcpFailure(
        step: Step?, requestID: UUID, operation: String?, result: VisionCaptureMCPResult
    ) {
        guard let step, result.isError else { return }
        var capture = MCPFailureCapture()
        capture.collect(result.value)
        var failure = capture.fields
        failure["host_request_id"] = .string(requestID.uuidString)
        failure["response_identity_validated"] = .bool(false)
        if let operation {
            let fixedOperations = ["launch app", "take a screenshot", "go back", "inspect cache",
                                   "describe screen", "describe system alert", "press system alert button",
                                   "tap cached action", "execute cached action", "revalidate cached action",
                                   "execute observed action"]
            // Cold tap/type requests embed their argument in the request prose.
            // Only a fixed host operation name may cross this trace boundary.
            if fixedOperations.contains(operation) { failure["operation"] = .string(operation) }
            else if operation.hasPrefix("type ") { failure["operation"] = .string("type") }
            else if operation.hasPrefix("tap ") { failure["operation"] = .string("tap") }
            else { failure["operation"] = .string("other") }
        }
        if let code = result.refusalCode,
           let value = FailureField.code.project(.string(code)) {
            failure["refusal_code"] = value
        }
        failure["dispatch_attempted"] = result.dispatchAttempted.map(JSONValue.bool) ?? .null
        failure["conflicting_dispatch_evidence"] = .bool(result.hasConflictingDispatchAttemptEvidence)
        failure["source_layout_recovery_signal_validated"] = .bool(result.isSourceLayoutChangedBeforeRevalidation)
        failure["delivered_transition_continuation_signal_validated"] = .bool(result.isDeliveredTransitionContinuation)
        failure["capture_limited"] = .bool(capture.limited)
        if let data = try? JSONEncoder().encode(JSONValue.object(failure)), data.count > 12 * 1_024 {
            failure["interaction_evidence"] = .array([])
            failure["capture_limited"] = .bool(true)
        }
        append(step: step, event: "mcp_failure", body: ["mcp_failure": .object(failure)],
               maximumBytes: 16 * 1_024)
    }

    private func append(
        step: Step, event: String, body: [String: JSONValue], maximumBytes: Int? = nil
    ) {
        var record = body
        record["schema_version"] = .integer(1)
        record["event"] = .string(event)
        record["process_id"] = .integer(Int64(ProcessInfo.processInfo.processIdentifier))
        record["step_id"] = .string(step.id.uuidString)
        record["step_index"] = Self.integer(step.index)
        record["conversation_id"] = step.conversation.map { .string($0.uuidString) } ?? .null
        record["turn_index"] = Self.integer(step.turn)
        record["timestamp_unix_seconds"] = .number(Date().timeIntervalSince1970)
        // Trace failures must not affect inference or put captured text in logs.
        guard var data = try? JSONEncoder().encode(JSONValue.object(record)) else { return }
        data.append(0x0a)
        if let maximumBytes, data.count > maximumBytes { return }
        try? file.write(contentsOf: data)
    }

    private static func integer(_ value: Int?) -> JSONValue {
        value.map { .integer(Int64($0)) } ?? .null
    }

    private static func number(_ value: Double?) -> JSONValue {
        guard let value, value.isFinite else { return .null }
        return .number(value)
    }
}

/// Only the public proof's structural comparison inputs are admitted. No target
/// labels, field values, prose, assertion values, images, selectors or capabilities.
private indirect enum FailureField: Sendable {
    case code, digest, number, flag, identityHash
    case object([String: FailureField])
    case array(FailureField)

    func project(_ value: JSONValue) -> JSONValue? {
        if value == .null { return .null }
        switch (self, value) {
        case (.code, .string(let text)):
            guard !text.isEmpty, text.utf8.count <= 96,
                  text.utf8.allSatisfy({ (48...57).contains($0) || (65...90).contains($0)
                      || [45, 46, 95].contains($0) || (97...122).contains($0) }) else { return nil }
            return value
        case (.digest, .string(let text)):
            guard text.utf8.count == 64,
                  text.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { return nil }
            return value
        case (.number, .integer), (.number, .unsignedInteger): return value
        case (.number, .number(let number)): return number.isFinite ? value : nil
        case (.flag, .bool): return value
        case (.identityHash, .string(let text)):
            guard !text.isEmpty, text.utf8.count <= 4_096 else { return nil }
            return .string(SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined())
        case (.object(let fields), .object(let object)):
            var projected: [String: JSONValue] = [:]
            for (key, rule) in fields {
                guard let raw = object[key], let safe = rule.project(raw) else { continue }
                let outputKey: String
                if case .identityHash = rule { outputKey = key + "_sha256" } else { outputKey = key }
                projected[outputKey] = safe
            }
            return projected.isEmpty ? nil : .object(projected)
        case (.array(let element), .array(let values)):
            return .array(values.prefix(2).compactMap { element.project($0) })
        default: return nil
        }
    }
}

private struct MCPFailureCapture {
    var fields: [String: JSONValue] = [:]
    var limited = false
    private var visited = 0
    private var remainingTextBytes = 256 * 1_024

    private static let timing: FailureField = .object([
        "started": .number, "completed": .number, "fresh_through": .number,
        "submission_started": .number, "dispatch_receipt_completed": .number,
    ])
    private static let observation: FailureField = .object([
        "structural_algorithm": .code, "structural_version": .number,
        "topology_digest": .digest, "layout_digest": .digest,
        "provider": .object(["source": .code, "version": .code, "root_identity": .identityHash]),
        "timing_monotonic_ns": timing,
        "accessibility_state": .object([
            "algorithm": .code, "version": .number, "state_digest": .digest,
            "coverage_digest": .digest, "disclosed_semantic_field_count": .number,
        ]),
    ])
    private static let expected: FailureField = .object([
        "kind": .code, "schema_version": .number,
        "structural_algorithm": .code, "structural_version": .number,
        "terminal_topology_digest": .digest, "terminal_layout_digest": .digest,
        "state_algorithm": .code, "state_version": .number,
        "terminal_state_digest": .digest, "terminal_coverage_digest": .digest,
        "terminal_disclosed_semantic_field_count": .number,
    ])
    private static let evidence: FailureField = .object([
        "schema_version": .number, "truth_contract_version": .number, "lane": .code,
        "action_id": .identityHash,
        "binding": .object([
            "udid": .identityHash, "requested_bundle_id": .identityHash,
            "observed_bundle_id": .identityHash, "observed_pid": .number,
            "lane_owner_id": .identityHash, "coordinate_mapping_revision": .identityHash,
            "process_epoch": .number, "session_epoch": .number, "evidence_generation": .number,
            "timestamps_monotonic_ns": timing,
        ]),
        "target": .object(["status": .code, "actual_event_recipient_observed": .flag,
                           "reason_code": .code]),
        "dispatch": .object(["status": .code, "delivery_acknowledged": .flag,
                             "submission_started": .flag]),
        "outcome": .object([
            "status": .code, "scope": .code, "reason_code": .code, "claim_status": .code,
            "assertion": expected,
            "observations": .object([
                "before": observation, "after": .array(observation),
                "after_attempted_read_count": .number,
            ]),
        ]),
    ])
    private static let metadata: FailureField = .object([
        "error": .code, "error_code": .code, "cache_used": .flag,
        "cache_revalidation_used": .flag, "revalidation_verified": .flag,
        "fresh_authority_recorded": .flag, "dispatch_attempted": .flag,
        "fresh_authority_error_code": .code, "revalidation_evidence_code": .code,
    ])

    mutating func collect(_ value: JSONValue, depth: Int = 0) {
        guard depth <= 12, visited < 128 else { limited = true; return }
        visited += 1
        switch value {
        case .object(let object):
            if object["type"] == .string("image") { return }
            add("metadata", Self.metadata.project(value))
            for (name, rule) in [
                ("interaction_evidence", Self.evidence),
                ("proof", FailureField.object(["verdict": .code, "verdict_source": .code, "reason_code": .code])),
                ("cache", FailureField.object(["state": .code, "cache_used": .flag])),
                ("expected_outcome", Self.expected),
            ] {
                if let raw = object[name] { add(name, rule.project(raw)) }
            }
            // Walk only transport envelopes, not arbitrary application objects.
            for key in ["result", "data", "payload", "structuredContent", "content"] {
                if let child = object[key] { collect(child, depth: depth + 1) }
            }
            if object["type"] == .string("text"), case .string(let text)? = object["text"] {
                collectText(text, depth: depth + 1)
            }
        case .array(let values):
            if values.count > 16 { limited = true }
            for child in values.prefix(16) { collect(child, depth: depth + 1) }
        default: break
        }
    }

    private mutating func add(_ name: String, _ value: JSONValue?) {
        guard let value, value != .null else { return }
        var values: [JSONValue] = []
        if case .array(let existing)? = fields[name] { values = existing }
        guard !values.contains(value) else { return }
        guard values.count < 4 else { limited = true; return }
        values.append(value)
        fields[name] = .array(values)
    }

    private mutating func collectText(_ text: String, depth: Int) {
        guard text.utf8.count <= remainingTextBytes else { limited = true; return }
        remainingTextBytes -= text.utf8.count
        let bytes = Array(text.utf8)
        var index = 0
        // MCP text can contain prose around multiple JSON objects. Scan balanced
        // objects without retaining or emitting the surrounding text.
        while index < bytes.count, visited < 128 {
            guard bytes[index] == 123 else { index += 1; continue }
            let start = index
            var nesting = 1
            var quoted = false
            var escaped = false
            index += 1
            while index < bytes.count, nesting > 0 {
                let byte = bytes[index]
                if quoted {
                    if escaped { escaped = false }
                    else if byte == 92 { escaped = true }
                    else if byte == 34 { quoted = false }
                } else if byte == 34 { quoted = true }
                else if byte == 123 { nesting += 1 }
                else if byte == 125 { nesting -= 1 }
                index += 1
            }
            guard nesting == 0 else { limited = true; return }
            if let decoded = try? JSONDecoder().decode(JSONValue.self, from: Data(bytes[start..<index])) {
                collect(decoded, depth: depth)
            }
        }
        if index < bytes.count { limited = true }
    }
}
