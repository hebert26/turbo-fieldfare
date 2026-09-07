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
    }

    private let file: FileHandle
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
    }

    func begin(_ request: AppGenerationRequest) -> Step {
        let step = Step(
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
        switch request.toolTurn {
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
        structuredProgress: DecodeStructuredProgress? = nil,
        toolCallPreview: DecodeToolCallPreview? = nil
    ) {
        guard let step else { return }
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
