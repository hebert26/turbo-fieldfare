import CryptoKit
import Foundation
import TurboFieldfare
import TurboFieldfareDecodeProtocol

struct AgentContextCheckpointProposal: Sendable {
    let id: UUID
    let replacementEpoch: UUID
    let call: AppToolCall
    let result: AppToolResult
    let record: String
    let commit: Bool
    let force: Bool
    let trigger: DecodeContextCheckpointTrigger
    let performanceEvidence: DecodePerformanceCheckpointEvidence?
    let permitsScreenshot: Bool
}

/// Conversation-owned facts, never a model-written completion certificate.
/// Exact user instructions and settled execution records survive host-side.
/// The replacement prompt receives only work status and the current screen.
struct AgentTaskCheckpoint: Sendable {
    static let maximumHistoryReplyBytes = 4_096
    static let maximumEvidenceLedgerBytes = 4_096
    static let maximumEvidenceGroups = 32
    /// The existing model-only format correction appends short host feedback.
    /// Reserve its space so even that corrected history result stays bounded.
    static let historyFeedbackReserveBytes = 512

    let taskID = UUID()
    private(set) var revision = 0
    private var userInstructions: [JSONValue] = []
    private var events: [JSONValue] = []
    /// Full host-only record for audit and exact handoff binding. Never rendered
    /// into the replacement prompt. Strings/attachment metadata share storage.
    private var auditHandoffs: [(call: AppToolCall, result: AppToolResult, outcome: String,
        target: JSONValue?, requestIDs: [UUID], session: String?)] = []
    private var observations: [String: JSONValue] = [:]
    private var observationIDs: [String: String] = [:]
    /// Local evidence reads are auditable but are not app execution or new
    /// observations. Results share their existing strings, not a copied history.
    private var historyReads: [(call: AppToolCall, result: AppToolResult)] = []
    private var interruptedGenerations: [ThoughtRepetitionRecovery] = []
    private var recoveryObservations: [(afterCallID: String, result: AppToolResult,
        outcome: String, requestIDs: [UUID], session: String?)] = []
    /// An inconclusive action can later be proved by a current screen. Keep
    /// that resolution after the agent moves to another screen.
    private var resolvedFollowUpEvents: Set<Int64> = []
    private struct HistoryCursor: Hashable, Sendable {
        let observationID: String
        let digest: String
        let offset: Int
    }
    private var issuedHistoryCursors: Set<HistoryCursor> = []

    private enum CheckpointActionStatus {
        case completed
        case failed
        case unresolved
    }

    mutating func appendUser(_ text: String, images: [AppImageAttachment]) {
        userInstructions.append(.object([
            "text_verbatim": .string(text),
            "images": .array(images.map(Self.imageReference)),
        ]))
    }

    mutating func appendHistoryRead(call: AppToolCall, result: AppToolResult) {
        historyReads.append((call, result))
    }

    /// Pages one existing sanitized projection. No private selectors, old
    /// choice IDs, raw transport payloads or live app reads enter this route.
    mutating func historyReply(arguments: JSONValue) throws -> (content: String, invalidReason: String?) {
        try Task.checkCancellation()
        guard let arguments = arguments.objectValue,
              (Set(arguments.keys) == ["observation_id"]
                || Set(arguments.keys) == ["observation_id", "cursor"]),
              case .string(let observationID)? = arguments["observation_id"],
              let observation = observations[observationID] else {
            return try historyFailure("Use an existing execution_records observation reference as observation_id, with only an optional returned cursor.")
        }
        let serialized = try Self.modelObservation(observation).encoded()
        let bytes = Array(serialized.utf8)
        let digest = Self.digest(serialized)
        var offset = 0
        if let supplied = arguments["cursor"] {
            guard let cursor = supplied.objectValue,
                  Set(cursor.keys) == ["task_id", "observation_id", "sha256", "offset_utf8"],
                  cursor["task_id"] == .string(taskID.uuidString),
                  cursor["observation_id"] == .string(observationID),
                  cursor["sha256"] == .string(digest),
                  case .integer(let value)? = cursor["offset_utf8"],
                  let position = Int(exactly: value), position > 0, position < bytes.count,
                  bytes[position] & 0xC0 != 0x80,
                  issuedHistoryCursors.contains(HistoryCursor(
                    observationID: observationID, digest: digest, offset: position)) else {
                return try historyFailure("The cursor does not identify a UTF-8 boundary in this task's exact immutable observation. Copy its returned next_cursor unchanged, or omit cursor for the first page.")
            }
            offset = position
        }
        let sourceEvent = events.first {
            $0.objectValue?["observation"] == .string(observationID)
        }?.objectValue?["event"] ?? .null
        func page(endingAt end: Int) throws -> String {
            let next: JSONValue = end == bytes.count ? .null : .object([
                "task_id": .string(taskID.uuidString), "observation_id": .string(observationID),
                "sha256": .string(digest), "offset_utf8": .integer(Int64(end)),
            ])
            return try JSONValue.object([
                "origin": .string("host_local_history"), "historical_only": .bool(true),
                "task_id": .string(taskID.uuidString), "observation_id": .string(observationID),
                "source_event": sourceEvent, "projection_sha256": .string(digest),
                "total_utf8_bytes": .integer(Int64(bytes.count)),
                "start_utf8": .integer(Int64(offset)), "end_utf8": .integer(Int64(end)),
                "record_complete_in_this_reply": .bool(offset == 0 && end == bytes.count),
                "page_is_final": .bool(end == bytes.count), "next_cursor": next,
                "content_json_fragment": .string(String(decoding: bytes[offset..<end], as: UTF8.self)),
                "instruction": .string("Historical evidence only, not a new screen or action result. Incomplete fragments must not be treated as a full record. Read remaining pages needed for a claim. Use only choices from the latest current decision packet. No app input was sent or replayed."),
            ]).encoded()
        }
        // Bound the encoded reply, including escaped fragment/cursor metadata,
        // rather than assuming a byte of source text is a byte of JSON output.
        // A final page replaces its cursor with null, so it can fit even when
        // a shorter partial page cannot. Check that bounded remainder first.
        if bytes.count - offset <= Self.maximumHistoryReplyBytes {
            let final = try page(endingAt: bytes.count)
            if final.utf8.count <= Self.maximumHistoryReplyBytes - Self.historyFeedbackReserveBytes {
                return (final, nil)
            }
        }
        var lower = 1
        var upper = min(Self.maximumHistoryReplyBytes, bytes.count - offset)
        var fitting: String?
        var fittingEnd = offset
        while lower <= upper {
            try Task.checkCancellation()
            let count = lower + (upper - lower) / 2
            var end = offset + count
            while end < bytes.count, end > offset, bytes[end] & 0xC0 == 0x80 { end -= 1 }
            guard end > offset else { lower = count + 1; continue }
            let candidate = try page(endingAt: end)
            if candidate.utf8.count <= Self.maximumHistoryReplyBytes - Self.historyFeedbackReserveBytes {
                fitting = candidate
                fittingEnd = end
                lower = count + 1
            } else {
                upper = count - 1
            }
        }
        guard let fitting else {
            return try historyFailure("This historical page could not fit the reply bound. Its evidence remains in the host audit and has not been provided. Leave claims requiring it unverified.", invalidRequest: false)
        }
        if fittingEnd < bytes.count {
            issuedHistoryCursors.insert(HistoryCursor(
                observationID: observationID, digest: digest, offset: fittingEnd))
        }
        return (fitting, nil)
    }

    private func historyFailure(_ reason: String, invalidRequest: Bool = true) throws -> (content: String, invalidReason: String?) {
        let content = try JSONValue.object(["origin": .string("host_local_history"),
            "historical_only": .bool(true),
            "status": .string(invalidRequest ? "invalid_history_request" : "history_page_unavailable"),
            "instruction": .string(reason),
            "app_input_sent": .bool(false)]).encoded()
        return (content, invalidRequest ? reason : nil)
    }

    mutating func appendSettled(call: AppToolCall, result: AppToolResult,
                               outcome: String, target: JSONValue?, requestIDs: [UUID],
                               session: String?, origin: String = "model_selected") throws {
        auditHandoffs.append((call, result, outcome, target, requestIDs, session))
        let decodedPacket = try Self.decodePacket(result.content)
        let decodedOutcome = try Self.decodePacket(outcome)
        let packet = decodedPacket.packet
        let body = decodedOutcome.packet
        let observation = Self.historicalObservation(packet)
        let canonical = try observation.encoded()
        let digest = Self.digest(canonical)
        let observationID: String
        if let previous = observationIDs[digest] { observationID = previous }
        else {
            observationID = "history_\(observations.count + 1)"
            observationIDs[digest] = observationID
            observations[observationID] = observation
        }
        // outcome is already the host's sanitized result. The displayed
        // packet supplies the readable facts formerly held by these wrappers.
        var execution = body.objectValue?.filter {
            !["available_actions", "available_text_fields", "screen_summary", "operation", "image"].contains($0.key)
        } ?? [:]
        if execution["instruction"] == .string("Observation is complete.") {
            execution.removeValue(forKey: "instruction")
        }
        var arguments = call.arguments.objectValue ?? [:]
        arguments.removeValue(forKey: "target")
        var event: [String: JSONValue] = [
            "event": .integer(Int64(events.count + 1)),
            "request": .object(arguments), "result": .object(execution),
            "observation": .string(observationID),
        ]
        if let target = Self.readableTarget(target) { event["target"] = target }
        if origin != "model_selected" { event["origin"] = .string(origin) }
        if let guidance = packet.objectValue?["guidance"], guidance != execution["instruction"] {
            event["guidance"] = guidance
        }
        if !result.imageAttachments.isEmpty {
            // Links the event to its actual retained image input. Other call
            // and transport identities remain solely in auditHandoffs.
            event["image_source_call"] = .string(call.id)
        }
        let feedback = [decodedPacket.feedback, decodedOutcome.feedback].compactMap { $0 }
        if !feedback.isEmpty { event["host_feedback"] = .array(Array(Set(feedback)).sorted().map(JSONValue.string)) }
        events.append(.object(event))
    }

    mutating func nextRevision() { revision += 1 }

    mutating func recordInterruptedGeneration(_ receipt: ThoughtRepetitionRecovery) {
        interruptedGenerations.append(receipt)
    }

    /// Additional host observation, not another execution of the preceding call.
    mutating func appendRecoveryObservation(afterCallID: String, result: AppToolResult,
        outcome: String, requestIDs: [UUID], session: String?) throws {
        recoveryObservations.append((afterCallID, result, outcome, requestIDs, session))
        let packet = try Self.decodePacket(result.content).packet
        let observation = Self.historicalObservation(packet)
        let digest = Self.digest(try observation.encoded())
        let id = observationIDs[digest] ?? "history_\(observations.count + 1)"
        observationIDs[digest] = id
        observations[id] = observation
        events.append(.object([
            "event": .integer(Int64(events.count + 1)),
            "origin": .string("host_read_only_generation_recovery"),
            "result": .object(["instruction": .string("The host obtained additional observation evidence after discarding an unfinished repeated model response. No application action was replayed. The earlier action retains its original verdict and delivery scope.")]),
            "observation": .string(id),
            "guidance": packet.objectValue?["guidance"] ?? .null,
            "image_source_call": .string(afterCallID),
        ]))
    }

    mutating func render(currentPacket: String, safety: JSONValue,
                pendingHistoryResult: AppToolResult? = nil,
                currentPacketWasRefreshed: Bool = true) throws -> String {
        try Task.checkCancellation()
        let decoded = try Self.decodePacket(currentPacket)
        let pendingHistory: JSONValue
        let pendingHistoryFeedback: JSONValue
        if let pendingHistoryResult {
            guard pendingHistoryResult.name == VisionCaptureToolDefinitions.historyReadName,
                  pendingHistoryResult.imageAttachments.isEmpty,
                  pendingHistoryResult.content.utf8.count <= Self.maximumHistoryReplyBytes else {
                throw VisionCaptureAgentError.malformedCall("The pending local history result was invalid.")
            }
            // The real call/result binding remains in the checkpoint proposal.
            // Keep the complete local page, including its required cursor and
            // fragment metadata, without repeating that transport wrapper.
            let pending = try Self.decodePacket(pendingHistoryResult.content)
            pendingHistory = pending.packet
            pendingHistoryFeedback = pending.feedback.map(JSONValue.string) ?? .null
        } else {
            pendingHistory = .null
            pendingHistoryFeedback = .null
        }
        var checkpointActions = try events.compactMap { event -> (CheckpointActionStatus, JSONValue)? in
            try Task.checkCancellation()
            return Self.checkpointAction(event)
        }
        checkpointActions = checkpointActions.map { status, action in
            guard status == .unresolved,
                  case .integer(let event)? = action.objectValue?["event"] else {
                return (status, action)
            }
            if resolvedFollowUpEvents.contains(event)
                || Self.currentScreenProves(action, packet: decoded.packet) {
                resolvedFollowUpEvents.insert(event)
                var resolved = action.objectValue ?? [:]
                resolved["verified_by"] = .string("later_current_screen")
                return (.completed, .object(resolved))
            }
            return (status, action)
        }
        var seenFollowUps: Set<String> = []
        let followUpAttempts = try checkpointActions.compactMap { status, action -> JSONValue? in
            guard status == .unresolved else { return nil }
            var semantic = action.objectValue ?? [:]
            semantic.removeValue(forKey: "event")
            guard seenFollowUps.insert(try JSONValue.object(semantic).encoded()).inserted else {
                return nil
            }
            return action
        }
        var seenFailures: Set<String> = []
        let failedActions = try checkpointActions.compactMap { status, action -> JSONValue? in
            guard status == .failed else { return nil }
            var semantic = action.objectValue ?? [:]
            semantic.removeValue(forKey: "event")
            guard seenFailures.insert(try JSONValue.object(semantic).encoded()).inserted else {
                return nil
            }
            return action
        }
        let renderedInstructions: [JSONValue]
        let supersededInstructionCount: Int
        if userInstructions.count > 2,
           let first = userInstructions.first,
           let last = userInstructions.last {
            // The original task and newest steering are the only instructions
            // needed to continue. Settled actions already carry the effect of
            // intermediate steering, so replaying every correction only adds
            // stale and sometimes conflicting text to the rebuilt context.
            renderedInstructions = [first, last]
            supersededInstructionCount = userInstructions.count - 2
        } else {
            renderedInstructions = userInstructions
            supersededInstructionCount = 0
        }
        var fields: [String: JSONValue] = [
            "schema_version": .integer(4),
            "user_instructions": .array(renderedInstructions),
            "completed_actions": .array(checkpointActions.compactMap {
                $0.0 == .completed ? Self.completedActionSummary($0.1) : nil
            }),
            "failed_actions": .array(failedActions),
            "follow_up_attempts": .array(followUpAttempts),
            "safety": Self.modelSafety(
                safety, currentPacketWasRefreshed: currentPacketWasRefreshed),
            "current_screen": Self.checkpointCurrentScreen(decoded.packet),
        ]
        if supersededInstructionCount > 0 {
            fields["superseded_instruction_count"] = .integer(
                Int64(supersededInstructionCount))
        }
        if pendingHistory != .null { fields["pending_history_result"] = pendingHistory }
        if pendingHistoryFeedback != .null { fields["pending_history_feedback"] = pendingHistoryFeedback }
        if let feedback = decoded.feedback { fields["current_screen_feedback"] = .string(feedback) }
        let value = JSONValue.object(fields)
        try Task.checkCancellation()
        return """
        Resume the user's QA goal from this host checkpoint.
        user_instructions contain the exact original goal and newest steering. Earlier steering is superseded; its settled effects remain in completed_actions and safety. completed_actions are verified. failed_actions are terminal evidence for the final report. follow_up_attempts still need current evidence. current_screen is the only current state and source of executable choices. Apply every safety restriction, including no replay.
        \(try value.encoded())

        Compare the goal with this record. Continue any reachable check with one permitted tool call. Answer only when every check has a terminal result or factual blocker.
        """
    }

    /// Keep user-visible actions and exceptional observations. Routine reads
    /// remain in the host audit; only the freshly read current screen is rendered.
    private static func checkpointAction(
        _ event: JSONValue
    ) -> (CheckpointActionStatus, JSONValue)? {
        let record = modelExecutionRecord(event)
        guard var body = record.objectValue,
              let request = body["request"]?.objectValue,
              case .string(let action)? = request["action"] else { return nil }
        if body["origin"] == .string("host_read_only_checkpoint_refresh") { return nil }
        let result = body["result"]?.objectValue ?? [:]
        // A malformed model proposal rejected by the host changed no app state,
        // consumes no retry budget, and cannot help continuation after compaction.
        if result["outcome"] == .string("not_sent") { return nil }
        let exceptionalRead = result["outcome"] != .string("succeeded")
            || result["refusal"] != nil || result["observation_refusal"] != nil
            || result["delivery_unknown"] != nil || result["system_alert"] != nil
        guard (action != "observe" && action != "screenshot") || exceptionalRead else {
            return nil
        }
        body.removeValue(forKey: "guidance")
        body.removeValue(forKey: "host_feedback")
        body.removeValue(forKey: "image_source_call")
        body.removeValue(forKey: "observation")
        let completed: Bool
        if action == "launch" {
            completed = result["launch"]?.objectValue?["verdict"] == .string("foreground_ready")
        } else if action == "observe" || action == "screenshot" {
            completed = result["outcome"] == .string("succeeded")
                && result["observation_refusal"] == nil
        } else {
            completed = result["proof"]?.objectValue?["verdict"] == .string("verified")
        }
        let failed = !completed && result["proof"]?.objectValue?["verdict"] == .string("failed")
        if completed || failed {
            body["result"] = .object(result.filter {
                ["outcome", "proof", "launch", "observation_outcome", "image_observation"].contains($0.key)
            })
        }
        return (completed ? .completed : failed ? .failed : .unresolved, .object(body))
    }

    /// Completion is already the array contract. Keep only the action, its
    /// readable target, and input needed to resume the user's work.
    private static func completedActionSummary(
        _ action: JSONValue, verifiedBy: String? = nil
    ) -> JSONValue {
        let source = action.objectValue ?? [:]
        var summary: [String: JSONValue] = [:]
        if let event = source["event"] { summary["event"] = event }
        if let request = source["request"]?.objectValue {
            summary["request"] = .object(request.filter {
                ["action", "text", "direction", "desired_state"].contains($0.key)
            })
        }
        if let target = source["target"]?.objectValue {
            let readable = target.filter {
                ["label", "role", "position"].contains($0.key)
            }
            if !readable.isEmpty { summary["target"] = .object(readable) }
        }
        if let verifiedBy {
            summary["verified_by"] = .string(verifiedBy)
        } else if let existing = source["verified_by"] {
            summary["verified_by"] = existing
        }
        return .object(summary)
    }

    /// Resolve only facts that directly establish the attempted result. A
    /// visible unselected target is insufficient evidence.
    private static func currentScreenProves(
        _ action: JSONValue, packet: JSONValue
    ) -> Bool {
        guard let body = action.objectValue,
              let request = body["request"]?.objectValue,
              case .string(let operation)? = request["action"],
              case .string(let label)? = body["target"]?.objectValue?["label"],
              let screen = packet.objectValue else { return false }
        let normalizedLabel = normalizedEvidence(label)
        let facts = values(screen["facts"])
        let choices = values(screen["choices"])

        func matchingObject(_ value: JSONValue) -> [String: JSONValue]? {
            guard let object = value.objectValue,
                  case .string(let currentLabel)? = object["label"],
                  normalizedEvidence(currentLabel) == normalizedLabel else { return nil }
            return object
        }

        switch operation {
        case "type":
            guard case .string(let inserted)? = request["text"] else { return false }
            return choices.compactMap(matchingObject).contains { object in
                guard case .string(let value)? = object["value"] else { return false }
                return value.contains(inserted)
            }
        case "set_boolean":
            guard case .bool(let desired)? = request["desired_state"] else { return false }
            return (choices + facts).compactMap(matchingObject).contains { object in
                object["current_state"] == .bool(desired)
                    || object["selected"] == .bool(desired)
            }
        case "tap":
            if (choices + facts).compactMap(matchingObject).contains(where: { object in
                object["selected"] == .bool(true)
                    || object["availability"] == .string("already_selected")
            }) { return true }
            if facts.contains(where: { fact in
                guard case .string(let text) = fact else { return false }
                let normalized = normalizedEvidence(plainFact(text))
                return normalized == "selected tab: \(normalizedLabel)"
                    || normalized == "selected segment: \(normalizedLabel)"
                    || normalized == "\(normalizedLabel) screen"
            }) { return true }
            // Screen headings and summaries often include the destination
            // name inside a longer sentence, for example "Your remaining
            // todos" after tapping Todos. Use whole words and exclude common
            // action labels so a generic Add or Back control cannot settle a
            // navigation result.
            let genericLabels: Set<String> = [
                "add", "back", "cancel", "close", "continue", "delete",
                "done", "edit", "more", "next", "remove", "save", "search",
                "submit",
            ]
            guard normalizedLabel.count >= 4,
                  !genericLabels.contains(normalizedLabel) else { return false }
            let labelWords = normalizedWords(normalizedLabel)
            guard !labelWords.isEmpty else { return false }
            return facts.contains { fact in
                guard case .string(let text) = fact else { return false }
                let factWords = normalizedWords(plainFact(text))
                return labelWords.allSatisfy(factWords.contains)
            }
        default:
            return false
        }
    }

    private static func normalizedEvidence(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private static func normalizedWords(_ text: String) -> Set<String> {
        Set(normalizedEvidence(text).split { !$0.isLetter && !$0.isNumber }
            .map(String.init))
    }

    private static func values(_ value: JSONValue?) -> [JSONValue] {
        guard case .array(let values) = value else { return [] }
        return values
    }

    /// Retain distinct observation groups from oldest to newest. Oversized
    /// groups are skipped so later compact facts can still be admitted. Overflow
    /// is explicit, and groups are admitted whole so facts are never truncated.
    static func evidenceLedger(
        events: [JSONValue], observations: [String: JSONValue],
        excluding: JSONValue
    ) throws -> JSONValue {
        let excludedDigest = try JSONValue.array(
            evidenceFacts(modelObservationSummary(excluding))
                .sorted().map(JSONValue.string)
        ).encoded()
        var retainedDigests: Set<String> = []
        var retained: [JSONValue] = []
        var omitted = 0

        func ledger(
            observations: [JSONValue], coverage: String, omitted: Int
        ) -> JSONValue {
            .object([
                "observations": .array(observations),
                "coverage": .string(coverage),
                "omitted_observations": .integer(Int64(omitted)),
            ])
        }

        // Reserve enough space for the largest possible omission count and the
        // longer coverage value so later skips cannot push an admitted ledger
        // over its byte limit.
        let reservedOmissionCount = max(events.count, 1)
        for event in events {
            guard case .integer(let eventNumber)? = event.objectValue?["event"],
                  case .string(let id)? = event.objectValue?["observation"],
                  let observation = observations[id] else { continue }
            let facts = evidenceFacts(modelObservationSummary(observation)).sorted()
            guard !facts.isEmpty else { continue }
            let digest = try JSONValue.array(facts.map(JSONValue.string)).encoded()
            guard digest != excludedDigest,
                  !retainedDigests.contains(digest) else { continue }
            let group = JSONValue.object([
                "event": .integer(eventNumber),
                "facts": .array(facts.map(JSONValue.string)),
            ])
            guard retained.count < maximumEvidenceGroups else {
                omitted += 1
                continue
            }
            let candidate = ledger(
                observations: retained + [group],
                coverage: "complete",
                omitted: reservedOmissionCount)
            guard try candidate.encoded().utf8.count <= maximumEvidenceLedgerBytes else {
                omitted += 1
                continue
            }
            retained.append(group)
            retainedDigests.insert(digest)
        }
        return ledger(
            observations: retained,
            coverage: omitted == 0 ? "complete" : "partial",
            omitted: omitted)
    }

    private static func modelObservationSummary(_ observation: JSONValue) -> JSONValue {
        let original = modelObservation(observation).objectValue ?? [:]
        var summary: [String: JSONValue] = [:]
        if case .array(let facts)? = original["facts"] {
            let useful = facts.compactMap { fact -> JSONValue? in
                guard case .string(let text) = fact,
                      !text.contains("label unavailable at ("),
                      !text.contains(" at ("),
                      !text.contains("[REDACTED]"),
                      text != "No app-owned accessibility action or editable field is currently published."
                else { return nil }
                return fact
            }
            if !useful.isEmpty { summary["facts"] = .array(useful) }
        }
        if case .array(let controls)? = original["displayed_controls"] {
            let labels = controls.compactMap { control -> JSONValue? in
                guard let values = control.objectValue,
                      case .string(let label)? = values["label"] else { return nil }
                var text = label
                if case .string(let value)? = values["value"], value != label {
                    text += " = \(value)"
                }
                if values["selected"] == .bool(true) { text += " [selected]" }
                return .string(text)
            }
            if !labels.isEmpty { summary["displayed"] = .array(labels) }
        }
        return .object(summary)
    }

    private static func evidenceFacts(_ summary: JSONValue) -> [String] {
        let body = summary.objectValue ?? [:]
        var items: [String] = []
        if case .array(let displayed)? = body["displayed"] {
            items.append(contentsOf: displayed.compactMap {
                guard case .string(let text) = $0 else { return nil }
                return text
            })
        }
        if case .array(let facts)? = body["facts"] {
            items.append(contentsOf: facts.compactMap {
                guard case .string(let text) = $0 else { return nil }
                return plainFact(text)
            })
        }
        var seen: Set<String> = []
        return items.filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    private static func plainFact(_ text: String) -> String {
        guard text.first == "[", let close = text.firstIndex(of: "]") else { return text }
        return text[text.index(after: close)...].trimmingCharacters(in: .whitespaces)
    }

    private static func checkpointCurrentScreen(_ packet: JSONValue) -> JSONValue {
        guard var body = packet.objectValue else { return packet }
        body.removeValue(forKey: "schema_version")
        if case .array(let facts)? = body["facts"] {
            body["facts"] = .array(facts.compactMap { fact -> JSONValue? in
                if case .object(var object) = fact {
                    object.removeValue(forKey: "position")
                    return .object(object)
                }
                guard case .string(let text) = fact,
                      !text.contains("label unavailable at ("),
                      !text.contains(" at ("),
                      !text.contains("[REDACTED]") else { return nil }
                return .string(plainFact(text))
            })
        }
        return .object(body)
    }

    /// Replace only exact repeated subtrees, keeping a readable, lossless and
    /// deterministic event stream. A table is used only when it is smaller.
    private static func compactExecutionRecords(
        _ original: [JSONValue]
    ) throws -> (records: [JSONValue], shared: [String: JSONValue]) {
        var records = original
        var shared: [String: JSONValue] = [:]
        let fields = [
            (field: "request", reference: "request_ref", table: "requests"),
            (field: "result", reference: "result_ref", table: "results"),
            (field: "target", reference: "target_ref", table: "targets"),
        ]

        func encodedBytes(_ candidateRecords: [JSONValue],
                          _ candidateShared: [String: JSONValue]) throws -> Int {
            try JSONValue.object([
                "execution_records": .array(candidateRecords),
                "execution_shared": .object(candidateShared),
            ]).encoded().utf8.count
        }

        for names in fields {
            try Task.checkCancellation()
            var counts: [String: Int] = [:]
            var values: [String: JSONValue] = [:]
            for record in records {
                guard let value = record.objectValue?[names.field] else { continue }
                let key = try value.encoded()
                counts[key, default: 0] += 1
                values[key] = value
            }
            let repeated = Set(counts.compactMap { $0.value > 1 ? $0.key : nil })
            guard !repeated.isEmpty else { continue }

            var indexes: [String: Int] = [:]
            var table: [JSONValue] = []
            var candidateRecords: [JSONValue] = []
            candidateRecords.reserveCapacity(records.count)
            for record in records {
                var body = record.objectValue ?? [:]
                if let value = body[names.field] {
                    let key = try value.encoded()
                    if repeated.contains(key) {
                        let index: Int
                        if let existing = indexes[key] {
                            index = existing
                        } else {
                            index = table.count
                            indexes[key] = index
                            table.append(values[key] ?? value)
                        }
                        body.removeValue(forKey: names.field)
                        body[names.reference] = .integer(Int64(index))
                    }
                }
                candidateRecords.append(.object(body))
            }
            var candidateShared = shared
            candidateShared[names.table] = .array(table)
            if try encodedBytes(candidateRecords, candidateShared)
                    < encodedBytes(records, shared) {
                records = candidateRecords
                shared = candidateShared
            }
        }
        return (records, shared)
    }

    /// Render decision facts only. Stored events and exact audit handoffs stay
    /// unchanged, including the private evidence used by the live tool loop.
    private static func modelExecutionRecord(_ event: JSONValue) -> JSONValue {
        let source = event.objectValue ?? [:]
        var record = source.filter {
            ["event", "request", "target", "observation", "origin", "guidance",
             "host_feedback", "image_source_call"].contains($0.key)
        }
        let original = source["result"]?.objectValue ?? [:]
        var result = original.filter {
            ["outcome", "observation_outcome", "outcome_note", "instruction", "guidance",
             "navigation_fact", "direction", "dispatch_attempted", "submission_started",
             "delivery_acknowledged", "delivery_unknown", "is_error", "recoverable",
             "message", "screen_changed"].contains($0.key)
        }
        // Retain action-level proof and readable explanations. Provider names
        // and diagnostic reason codes are not additional execution verdicts.
        result["proof"] = modelFields(original["proof"], keys: ["verdict", "action", "reason"])
        result["launch"] = modelFields(original["launch"], keys: ["verdict", "disposition",
            "mutation_sent", "device_readiness", "failed_proof_stage", "process_state", "foreground_state"])
        result["current_observation"] = modelFields(original["current_observation"],
            keys: ["outcome", "requested_foreground_app_proven"])
        result["image_observation"] = modelFields(original["image_observation"],
            keys: ["state", "order", "visual_agreement", "image_source", "read_source"])
        result["system_alert"] = original["system_alert"]
        if let refusal = original["refusal"]?.objectValue {
            // Each currently produced refusal also carries its plain instruction.
            // Keep its specific meaning without exposing cache or routing codes.
            let fact: String
            switch refusal["code"] {
            case .string("GUARDED_TARGET_REJECTED")?:
                fact = "The exact target was rejected before submission. It must not be sent again."
            case .string("CACHE_ACTION_CAPABILITY_INVALID")?:
                fact = "The action's permission expired before input was sent."
            case .string("OBSERVED_TARGET_NOT_PUBLISHED")?:
                fact = "The observed target was not offered as a current executable action."
            default:
                fact = "A refusal was recorded. Its detailed cause is not represented here. Retain the accompanying instructions and do not infer successful execution."
            }
            result["refusal"] = .object(["fact": .string(fact)])
        }
        if let refusal = original["observation_refusal"]?.objectValue {
            var facts = refusal.filter {
                ["dispatch_attempted", "submission_started", "delivery_acknowledged", "delivery_unknown"].contains($0.key)
            }
            facts["fact"] = .string(refusal["recovery_reason"] == .string("observation_topology_changed_before_completion")
                ? "The screen changed before the read could complete. These delivery facts belong to that read, not to the preceding input."
                : "A read was refused. Its detailed cause is not represented here. These delivery facts do not describe the preceding input.")
            result["observation_refusal"] = .object(facts)
        }
        if let recovery = original["stale_recovery"]?.objectValue {
            result["stale_recovery"] = .object(recovery.filter {
                ["dispatch_attempted", "screen_changed"].contains($0.key)
            })
            if recovery["screen_changed"] == .bool(false) {
                let instruction = JSONValue.string("The action was not sent. A follow-up read found unchanged screen content. The refused input was not retried. Any historical confirmation offer is expired. Use only the current choices.")
                result["instruction"] = instruction
                if record["guidance"] == original["instruction"] { record["guidance"] = instruction }
            }
        }
        record["result"] = .object(result)
        return .object(record)
    }

    private static func modelSafety(
        _ safety: JSONValue, currentPacketWasRefreshed: Bool
    ) -> JSONValue {
        let original = safety.objectValue ?? [:]
        var facts: [String: JSONValue] = [:]
        if original["read_only_recovery_required"] == .bool(true) {
            facts["read_only_recovery_required"] = .bool(true)
        }
        if let alert = modelFields(original["uncertain_alert_press"], keys: ["button"]),
           alert != .null {
            facts["uncertain_alert_press"] = alert
        }
        let targetRestriction = currentPacketWasRefreshed
            ? "Earlier targets and confirmation offers are expired."
            : "Historical targets and confirmation offers are expired. Targets in current_screen remain choices only for their listed operations and still require normal host capability checks."
        var restrictions = targetRestriction
            + " Keep prior refusals and retry limits. Never replay delivery-unknown input."
        if original["read_only_recovery_required"] == .bool(true) {
            restrictions += " Use only permitted current read-only recovery."
        }
        facts["restrictions"] = .string(restrictions)
        return .object(facts)
    }

    /// Preserve explicit null and missing as distinct facts. Never manufacture
    /// a new value/status when the original did not supply it.
    private static func modelFields(_ value: JSONValue?, keys: Set<String>) -> JSONValue? {
        guard let value else { return nil }
        if value == .null { return .null }
        guard let object = value.objectValue else { return nil }
        return .object(object.filter { keys.contains($0.key) })
    }

    private static func historicalObservation(_ packet: JSONValue) -> JSONValue {
        var value = packet.objectValue ?? [:]
        value.removeValue(forKey: "last_action")
        value.removeValue(forKey: "journey_hint")
        value.removeValue(forKey: "guidance")
        value.removeValue(forKey: "allowed_next")
        value.removeValue(forKey: "can_swipe")
        if var observation = value["observation"]?.objectValue {
            observation.removeValue(forKey: "id")
            value["observation"] = .object(observation)
        }
        if case .array(let choices)? = value["choices"] {
            value["readable_controls_at_observation"] = .array(choices.map {
                var choice = $0.objectValue ?? [:]
                choice.removeValue(forKey: "id")
                choice.removeValue(forKey: "operations")
                choice.removeValue(forKey: "confirmation_attempts")
                return .object(choice)
            })
            value.removeValue(forKey: "choices")
        }
        return .object(value)
    }

    /// Some model-only correction paths append plain host feedback after the
    /// JSON packet. Preserve it separately. Never trim an incomplete JSON body
    /// into a successful record or discard an unparsed trailing instruction.
    private static func decodePacket(_ content: String) throws -> (packet: JSONValue, feedback: String?) {
        let data = Data(content.utf8)
        if let packet = try? JSONDecoder().decode(JSONValue.self, from: data),
           case .object = packet { return (packet, nil) }
        let bytes = Array(content.utf8)
        guard let start = bytes.firstIndex(where: { ![UInt8(9), 10, 13, 32].contains($0) }),
              bytes[start] == 123 else {
            throw VisionCaptureAgentError.malformedCall("Checkpoint result did not contain a JSON object.")
        }
        var depth = 0
        var inString = false
        var escaped = false
        for index in start..<bytes.count {
            let byte = bytes[index]
            if inString {
                if escaped { escaped = false }
                else if byte == 92 { escaped = true }
                else if byte == 34 { inString = false }
            } else if byte == 34 { inString = true }
            else if byte == 123 || byte == 91 { depth += 1 }
            else if byte == 125 || byte == 93 {
                depth -= 1
                if depth == 0 {
                    let packet = try JSONDecoder().decode(JSONValue.self, from: Data(bytes[start...index]))
                    guard case .object = packet else {
                        throw VisionCaptureAgentError.malformedCall("Checkpoint result was not a JSON object.")
                    }
                    let feedback = String(decoding: bytes[(index + 1)...], as: UTF8.self)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    return (packet, feedback.isEmpty ? nil : feedback)
                }
            }
        }
        throw VisionCaptureAgentError.malformedCall("Checkpoint result JSON was incomplete.")
    }

    private static func readableTarget(_ target: JSONValue?) -> JSONValue? {
        guard let target = target?.objectValue else { return nil }
        var result: [String: JSONValue] = [:]
        for key in ["label", "role", "selected", "enabled", "current_state"] {
            if let value = target[key], value != .null { result[key] = value }
        }
        // An unlabeled control can only be identified after compaction by its
        // visible position. Keep that small semantic fact without retaining
        // the expired selector or choice ID.
        if result["label"] == nil, let position = target["position"] {
            result["position"] = position
        }
        // Already-sanitized field facts: preserve supplied status and explicit
        // null without turning an absent value into empty or available text.
        for key in ["value", "value_status"] {
            if let value = target[key] { result[key] = value }
        }
        if result["label"] == nil, target["selector_kind"] == .string("placeholder") {
            result["label"] = target["selector_at_that_time"]
        }
        return result.isEmpty ? nil : .object(result)
    }

    private static func modelObservation(_ observation: JSONValue) -> JSONValue {
        let original = observation.objectValue ?? [:]
        var result: [String: JSONValue] = [:]
        if let facts = original["facts"], facts != .array([]) { result["facts"] = facts }
        if let metadata = original["observation"]?.objectValue {
            if metadata["state"] != .string("current") { result["state"] = metadata["state"] }
            for key in ["image_relationship", "system_alert", "requested_app_present"] {
                result[key] = metadata[key]
            }
        }
        if case .array(let controls)? = original["readable_controls_at_observation"] {
            // Preserve displayed text, values and selection evidence without
            // publishing any historical action list or executable target ID.
            let readable = controls.compactMap { Self.readableTarget($0) }
            if !readable.isEmpty { result["displayed_controls"] = .array(readable) }
        }
        return .object(result)
    }

    static func imageReference(_ image: AppImageAttachment) -> JSONValue {
        .object(["id": .string(image.id.uuidString), "sha256": .string(image.sha256),
                 "name": .string(image.displayName)])
    }

    static func digest(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
