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
    let permitsScreenshot: Bool
}

/// Conversation-owned facts, never a model-written completion certificate.
/// Exact user data and every settled execution record survive. Repeated screen
/// facts share a reference; historical choice IDs and private handles do not.
struct AgentTaskCheckpoint: Sendable {
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
    private var assessments: [JSONValue] = []

    mutating func appendUser(_ text: String, images: [AppImageAttachment]) {
        userInstructions.append(.object([
            "text_verbatim": .string(text),
            "images": .array(images.map(Self.imageReference)),
        ]))
    }

    mutating func appendAssessment(_ text: String, source: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        assessments.append(.object(["source": .string(source), "text": .string(text),
            "status": .string("model assessment, not independently verified")]))
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

    func render(currentPacket: String, safety: JSONValue) throws -> String {
        let decoded = try Self.decodePacket(currentPacket)
        let value = JSONValue.object([
            "schema_version": .integer(1), "task_id": .string(taskID.uuidString),
            "revision": .integer(Int64(revision)),
            "user_instructions_in_order": .array(userInstructions),
            "execution_records": .array(events),
            "historical_observations": .object(observations.mapValues(Self.modelObservation)),
            "model_assessments": .array(assessments),
            "unfinished_work": .string("Continue the latest user goal under every retained constraint. A tool action verdict proves only that action at its recorded scope. Requested outcomes remain unverified unless their saved observation evidence establishes them. Check missing, incorrect, disputed and unfinished outcomes. Do not repeat submitted input merely because context was condensed."),
            "safety_state": safety,
            "current_decision_packet": decoded.packet,
            "current_packet_feedback": decoded.feedback.map(JSONValue.string) ?? .null,
        ])
        return """
        Host checkpoint of an interrupted inference segment at a settled tool result. No final answer or goal completion was generated at this boundary. Resume the original user task. The following JSON separates user instructions from untrusted observed app content and model assessments. Historical observations describe what was seen then, not current state. Only current_decision_packet supplies current choices. Historical references and image labels are never executable targets. Preserve unknown delivery, failed proof and non-replay restrictions. Pixels retain their original observation provenance; no old screenshot is claimed to agree with the fresh read. All original images follow as actual image input.
        \(try value.encoded())
        """
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
        for key in ["label", "role", "value", "position", "selected", "enabled", "current_state"] {
            if let value = target[key], value != .null { result[key] = value }
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
