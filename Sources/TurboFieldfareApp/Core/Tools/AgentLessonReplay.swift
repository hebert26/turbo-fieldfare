import Foundation

/// Reads JSONL locally. Raw labels, values and coordinates exist only while
/// binding an episode; they never leave this adapter in records or queries.
/// Unknown layouts and missing results cause abstention, not guessed success.
struct AgentLessonReplay {
    struct Write: Sendable {
        let trace: String
        let step: Int
        let lesson: AgentLesson
    }
    struct Retrieval: Sendable {
        let trace: String
        let step: Int
        let result: AgentLessonRetrieval
    }
    struct Report: Sendable {
        var writes: [Write] = []
        var retrievals: [Retrieval] = []
        var lessons: [AgentLesson] = []
        var settledSteps = 0
        var unpairedCalls = 0
    }
    enum ReplayError: Error { case invalidLine(Int), unsupportedSchema(Int), missingRunReference }
    private struct Pending {
        let step: Int
        let callID: String
        let arguments: [String: Any]
        let before: [String: Any]
        let subgoal: UUID
        let state: UUID
        let versions: AgentLessonVersions
    }

    /// Input order is the learning order. Retrieval precedes generation; a
    /// future repair cannot be retrieved at an earlier step. No persistent
    /// default memory is opened by replay.
    static func run(traces: [URL]) throws -> Report {
        var policy = AgentLessonPolicy()
        var report = Report()
        for url in traces {
            let rows = try read(url)
            guard let sourceRun = rows.first?["conversation_id"] as? String,
                  let run = UUID(uuidString: sourceRun) else { throw ReplayError.missingRunReference }
            let trace = url.deletingPathExtension().lastPathComponent
            var hosts: [Int: [[String: Any]]] = [:]
            for row in rows where row["event"] as? String == "mcp_request" {
                if let step = row["step_index"] as? Int, let request = row["mcp_request"] as? [String: Any] {
                    hosts[step, default: []].append(request)
                }
            }
            var pending: [String: Pending] = [:]
            var packet: [String: Any] = [:]
            var state = UUID(), subgoal = UUID()
            var desiredText: String?
            var previousError: AgentLessonError?
            var currentEpisode: AgentLessonReference?
            policy.discardPendingEpisodes()
            for row in rows {
                guard let step = row["step_index"] as? Int, step >= 0 else { continue }
                let now = Date(timeIntervalSince1970: row["timestamp_unix_seconds"] as? Double ?? 0)
                if row["event"] as? String == "input", let input = row["input"] as? [String: Any] {
                    if input["kind"] as? String == "user" {
                        policy.discardPendingEpisodes()
                        pending.removeAll()
                        packet = [:]
                        state = UUID(); subgoal = UUID(); desiredText = nil
                        previousError = nil; currentEpisode = nil
                    }
                    for result in input["results"] as? [[String: Any]] ?? [] {
                        guard let callID = result["call_id"] as? String,
                              let content = result["content"] as? String,
                              let next = try? JSONSerialization.jsonObject(with: Data(content.utf8))
                                as? [String: Any] else { continue }
                        if let proposal = pending.removeValue(forKey: callID) {
                            let requests = hosts[proposal.step] ?? []
                            let settled = abstract(proposal, after: next, hosts: requests,
                                                   run: run, resultStep: step, time: now)
                            let writes = policy.settle(settled)
                            report.settledSteps += 1
                            for lesson in writes {
                                report.writes.append(.init(trace: trace, step: proposal.step, lesson: lesson))
                            }
                            if settled.before.contains(.deliveryUnknown) || settled.after.contains(.deliveryUnknown) {
                                state = UUID(); subgoal = UUID(); desiredText = nil
                                previousError = nil; currentEpisode = nil
                            } else if settled.outcome == .rejected {
                                previousError = settled.shape == .mixedTargetAndPoint ? .callShape : .unavailableTarget
                                if currentEpisode == nil { currentEpisode = settled.reference }
                            } else if settled.action == .refresh {
                                previousError = nil
                            } else {
                                previousError = settled.outcome == .inconclusive ? .inconclusive : nil
                            }
                            // A non-field navigation ends the scope, even if its effect is weak.
                            // A bound field focus stays in the same form.
                            let action = proposal.arguments["action"] as? String ?? ""
                            if settled.outcome != .rejected,
                               !["type", "observe", "screenshot"].contains(action),
                               !isFieldFocus(proposal.arguments, packet: proposal.before) {
                                state = UUID(); subgoal = UUID(); desiredText = nil
                                currentEpisode = nil
                                policy.discardPendingEpisodes()
                            }
                        }
                        packet = next
                    }
                    guard !packet.isEmpty else { continue }
                    var situation = conditions(packet)
                    if previousError == .callShape { situation.insert(.callShapeRejected) }
                    let query = AgentLessonQuery(
                        controlType: situation.contains(.nativeAlertPresent) ? .systemAlert : .editableField,
                        situation: situation, error: previousError, versions: versions(packet),
                        permitted: situation.intersection(
                            [.freshReadPermitted, .pointTypingPermitted, .typingPermitted]),
                        evidenceKinds: [.hostProof, .loopRejection, .observation],
                        currentEpisode: currentEpisode, now: now)
                    let retrieval = TextLessonRetriever().retrieve(query, from: policy.lessons)
                    if !retrieval.lessons.isEmpty {
                        report.retrievals.append(.init(trace: trace, step: step, result: retrieval))
                    }
                }
                if row["event"] as? String == "output", let output = row["output"] as? [String: Any],
                   let calls = output["tool_calls"] as? [[String: Any]], calls.count == 1,
                   let call = calls.first, let callID = call["id"] as? String,
                   let arguments = call["arguments"] as? [String: Any] {
                    if arguments["action"] as? String == "type", let text = arguments["text"] as? String {
                        if let previous = desiredText, previous != text {
                            subgoal = UUID(); currentEpisode = nil; previousError = nil
                            policy.discardPendingEpisodes()
                        }
                        desiredText = text
                    }
                    pending[callID] = .init(step: step, callID: callID, arguments: arguments, before: packet,
                                           subgoal: subgoal, state: state, versions: versions(packet))
                }
            }
            report.unpairedCalls += pending.count
        }
        report.lessons = policy.lessons
        return report
    }

    private static func read(_ url: URL) throws -> [[String: Any]] {
        let content = try String(contentsOf: url, encoding: .utf8)
        return try content.split(separator: "\n").enumerated().map { index, line in
            guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else {
                throw ReplayError.invalidLine(index + 1)
            }
            guard row["schema_version"] as? Int == 1 else { throw ReplayError.unsupportedSchema(index + 1) }
            return row
        }
    }

    static func versions(_ packet: [String: Any]) -> AgentLessonVersions {
        .init(record: 1, tool: packet["schema_version"] as? Int ?? 1, proof: 1)
    }
    static func choices(_ packet: [String: Any]) -> [[String: Any]] {
        packet["choices"] as? [[String: Any]] ?? []
    }
    static func isEditable(_ choice: [String: Any]) -> Bool {
        ["text_field", "text_view", "secure_text_field"].contains(choice["role"] as? String ?? "")
            && choice["enabled"] as? Bool == true
            && (choice["operations"] as? [String] ?? []).contains("type")
    }
    private static func ocrFields(_ packet: [String: Any]) -> [String] {
        (packet["facts"] as? [Any] ?? []).compactMap { $0 as? String }.filter {
            $0.hasPrefix("[text_field] ") && $0.contains("ocr-confirmed") && !$0.contains(" disabled")
                && ocrPosition($0) != nil
        }
    }

    private static func uniqueRange(of delimiter: String, in text: String) -> Range<String.Index>? {
        guard let range = text.range(of: delimiter),
              text.range(of: delimiter, range: range.upperBound..<text.endIndex) == nil else { return nil }
        return range
    }

    private static func ocrPosition(_ fact: String) -> (start: String.Index, x: Double, y: Double)? {
        guard let start = uniqueRange(of: " at (", in: fact),
              let end = uniqueRange(of: ") ocr-confirmed", in: fact),
              start.upperBound <= end.lowerBound else { return nil }
        let coordinates = fact[start.upperBound..<end.lowerBound]
            .split(separator: ",", omittingEmptySubsequences: false)
        guard coordinates.count == 2,
              let x = Double(coordinates[0].trimmingCharacters(in: .whitespaces)), x.isFinite,
              let y = Double(coordinates[1].trimmingCharacters(in: .whitespaces)), y.isFinite else { return nil }
        return (start.lowerBound, x, y)
    }

    static func deliveryIsUnknown(_ packet: [String: Any]) -> Bool {
        let last = packet["last_action"] as? [String: Any] ?? [:]
        return last["delivery"] as? String == "unknown" || last["verdict"] as? String == "delivery_unknown"
    }

    static func conditions(_ packet: [String: Any]) -> Set<AgentLessonCondition> {
        var conditions: Set<AgentLessonCondition> = []
        if deliveryIsUnknown(packet) { conditions.insert(.deliveryUnknown) }
        let allowed = packet["allowed_next"] as? [String] ?? []
        let observation = packet["observation"] as? [String: Any] ?? [:]
        if observation["state"] as? String != "current" { return conditions }
        if choices(packet).contains(where: isEditable) { conditions.insert(.editableChoicePresent) }
        if !ocrFields(packet).isEmpty { conditions.insert(.ocrFieldPresent) }
        if allowed.contains("type") { conditions.insert(.typingPermitted) }
        if allowed.contains("type"), !ocrFields(packet).isEmpty { conditions.insert(.pointTypingPermitted) }
        if allowed.contains("screenshot") || allowed.contains("observe") { conditions.insert(.freshReadPermitted) }
        // Only the explicit screenshot recovery state is admitted as a missing
        // composer field, rather than treating every screen without input as a form.
        if allowed == ["screenshot"], !conditions.contains(.editableChoicePresent),
           !conditions.contains(.ocrFieldPresent) { conditions.insert(.fieldAbsent) }
        if (observation["system_alert"] as? [String: Any])?["present"] as? Bool == true {
            conditions.insert(.nativeAlertPresent)
        }
        return conditions
    }

    static func pointMatchesField(_ arguments: [String: Any], packet: [String: Any]) -> Bool {
        guard let x = arguments["x_norm"] as? NSNumber, let y = arguments["y_norm"] as? NSNumber,
              x.doubleValue.isFinite, y.doubleValue.isFinite else { return false }
        // Compare the current OCR position locally; never preserve it in a lesson.
        return ocrFields(packet).filter { fact in
            guard let position = ocrPosition(fact) else { return false }
            return position.x == x.doubleValue && position.y == y.doubleValue
        }.count == 1
    }
    static func isFieldFocus(_ arguments: [String: Any], packet: [String: Any]) -> Bool {
        guard arguments["action"] as? String == "tap_coordinates" else { return false }
        return pointMatchesField(arguments, packet: packet)
    }

    private static func abstract(_ pending: Pending, after: [String: Any], hosts: [[String: Any]],
                                 run: UUID, resultStep: Int, time: Date) -> AgentLessonStep {
        let args = pending.arguments
        let rawAction = args["action"] as? String ?? ""
        let action: AgentLessonStep.Action = rawAction == "type" ? .type
            : (["observe", "screenshot"].contains(rawAction) ? .refresh : .other)
        let target = args["target"] as? String
        let hasPoint = args["x_norm"] != nil || args["y_norm"] != nil
        let chosen = choices(pending.before).filter { $0["id"] as? String == target && target != nil }
        let choiceBound = chosen.count == 1 && chosen.first.map(isEditable) == true
        let pointBound = target == nil && pointMatchesField(args, packet: pending.before)
        let shape: AgentLessonStep.Shape = action != .type ? .other
            : (target != nil && hasPoint ? .mixedTargetAndPoint
                : (pointBound ? .pointOnly : (choiceBound ? .currentChoice : .unavailable)))
        let last = after["last_action"] as? [String: Any] ?? [:]
        let verdict = last["verdict"] as? String ?? ""
        let typeRequests = hosts.filter { $0["operation"] as? String == "type" }
        let typeSent = !typeRequests.isEmpty
        let outcome: AgentLessonStep.Outcome = verdict == "not_sent" ? .rejected
            : (action == .refresh && verdict == "observed" ? .observed
                : (action == .type && typeSent ? .admitted
                    : (verdict == "failed" ? .failed : .inconclusive)))
        let positiveHost = typeRequests.count == 1
            && typeRequests.first?["is_error"] as? Bool == false
            && typeRequests.first?["refusal_code"] as? String == "LEGACY_FIELD_VALUE_CHANGED"
        let exactValue = valueMatches(args, before: pending.before, after: after, chosen: chosen.first)
        let conflict = verdict == "verified" && typeSent && !positiveHost
        let obs = after["observation"] as? [String: Any] ?? [:]
        let fresh = obs["state"] as? String == "current" && obs["read_result"] as? String == "succeeded"
            && hosts.contains { $0["operation"] as? String == "describe screen"
                && $0["is_error"] as? Bool == false && $0["refusal_code"] == nil }
        return .init(reference: .init(run: run, step: pending.step),
            observationReference: .init(run: run, step: resultStep), subgoal: pending.subgoal,
            controlState: pending.state, versions: pending.versions, time: time,
            action: action, shape: shape, outcome: outcome, before: conditions(pending.before),
            after: conditions(after), observationIsFresh: fresh, targetBound: choiceBound || pointBound,
            hostVerifiedValue: positiveHost && exactValue && verdict == "verified", verdictConflict: conflict)
    }

    static func valueMatches(_ args: [String: Any], before: [String: Any],
                                     after: [String: Any], chosen: [String: Any]?) -> Bool {
        guard let expected = args["text"] as? String, !expected.isEmpty else { return false }
        if let chosen, let label = chosen["label"] as? String {
            let fields = choices(after).filter { isEditable($0) && $0["label"] as? String == label }
            return fields.count == 1 && fields.first?["value_status"] as? String == "available"
                && fields.first?["value"] as? String == expected
        }
        guard pointMatchesField(args, packet: before) else { return false }
        let boundFacts = ocrFields(before).filter { fact in
            pointMatchesField(args, packet: ["facts": [fact]])
        }
        guard boundFacts.count == 1, let bound = boundFacts.first,
              let position = ocrPosition(bound) else { return false }
        let label = String(bound[..<position.start])
        let matches = ocrFields(after).filter {
            pointMatchesField(args, packet: ["facts": [$0]])
                && ($0.hasPrefix(label + " at (") || $0.hasPrefix("[text_field] " + expected + " (ocr): "))
        }
        guard matches.count == 1, let fact = matches.first else { return false }
        if let valueStart = fact.range(of: " value=") {
            return String(fact[valueStart.upperBound...]) == expected
        }
        // Historical OCR changed the empty-field label into its observed value.
        // Require the same current point, unique editable role and exact OCR value.
        guard let marker = uniqueRange(of: " (ocr): ", in: fact), let position = ocrPosition(fact),
              marker.upperBound <= position.start else { return false }
        return String(fact[marker.upperBound..<position.start]) == expected
    }
}
