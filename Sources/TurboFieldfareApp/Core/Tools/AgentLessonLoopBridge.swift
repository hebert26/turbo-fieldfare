#if !AGENT_LESSON_REPLAY
import Foundation

/// Experimental, deterministic memory at the settled-result boundary. This
/// actor owns one live scope; only closed-vocabulary lessons reach disk or prompts.
actor AgentLessonLoopBridge {
    static let heading = "Lessons from earlier runs:"
    static func contextReplacementFields(environment: [String: String] = ProcessInfo.processInfo.environment)
        -> Set<String> {
        Mode(environment["TURBOFIELDFARE_AGENT_LESSONS"]) == nil ? [] : ["lessons"]
    }
    static let shared: AgentLessonLoopBridge? = {
        let environment = ProcessInfo.processInfo.environment
        guard Mode(environment["TURBOFIELDFARE_AGENT_LESSONS"]) != nil else { return nil }
        return AgentLessonLoopBridge(environment: environment, trace: AgentInferenceTrace.shared)
    }()

    enum Mode {
        case text, embedding
        init?(_ flag: String?) {
            switch flag {
            case "text": self = .text
            case "embedding": self = .embedding
            default: return nil
            }
        }
    }

    private struct Scope: Equatable {
        let owner: ObjectIdentifier?
        let run: UUID?
        let instructionDigest: String
        let targetDigest: String
        let sessionDigest: String?
    }
    private let mode: Mode?
    private let store: AgentLessonStore?
    private let trace: AgentInferenceTrace?
    private let encoder: (any LessonTextEncoding)?
    private var reportedMissingEncoder = false
    private var scope: Scope?
    private var runID = UUID()
    private var sequence = 0
    private var subgoal = UUID()
    private var controlState = UUID()
    private var desiredTextDigest: String?
    private var packet: [String: Any] = [:]
    private var policy = AgentLessonPolicy()
    private var previousError: AgentLessonError?
    private var currentEpisode: AgentLessonReference?

    init(environment: [String: String], trace: AgentInferenceTrace? = nil,
         encoder: (any LessonTextEncoding)? = nil) {
        mode = Mode(environment["TURBOFIELDFARE_AGENT_LESSONS"])
        if mode != nil {
            let url = environment["TURBOFIELDFARE_AGENT_LESSONS_PATH"].map {
                URL(fileURLWithPath: $0)
            } ?? AgentLessonStore.defaultURL
            store = AgentLessonStore(url: url)
            self.trace = trace
            self.encoder = encoder
        } else {
            store = nil
            self.trace = nil
            self.encoder = nil
        }
    }

    /// Called only after the ordinary settled-result checkpoint was accepted.
    /// The host outcome is trusted loop evidence, never model-authored reflection.
    func settle(call: AppToolCall, result: AppToolResult, outcome: String,
                owner: ObjectIdentifier? = nil, run: UUID?, instruction: String = "",
                target: String = "", session: String? = nil, now: Date = Date()) async {
        guard mode != nil, call.name == VisionCaptureToolDefinitions.navigateName,
              result.callID == call.id, result.name == call.name else { return }
        await reportMissingEncoder()
        let nextScope = Scope(owner: owner, run: run,
            instructionDigest: AgentTaskCheckpoint.digest(instruction),
            targetDigest: AgentTaskCheckpoint.digest(target),
            sessionDigest: session.map(AgentTaskCheckpoint.digest))
        if scope != nextScope {
            scope = nextScope
            runID = run ?? UUID()
            sequence = 0
            packet = [:]
            policy = AgentLessonPolicy()
            discardEpisode()
        }
        guard let after = Self.decode(result.content), let body = Self.decode(outcome),
              let arguments = Self.decode((try? call.arguments.encoded()) ?? ""),
              after["schema_version"] as? Int == 1 else {
            packet = [:]
            discardEpisode()
            return
        }
        let versions = AgentLessonReplay.versions(after)
        if !packet.isEmpty, AgentLessonReplay.versions(packet) != versions {
            packet = [:]
            discardEpisode()
        }
        if arguments["action"] as? String == "type", let text = arguments["text"] as? String {
            let digest = AgentTaskCheckpoint.digest(text)
            if let previous = desiredTextDigest, previous != digest { discardEpisode() }
            desiredTextDigest = digest
        }
        sequence += 1
        let traceStep = await trace?.currentStep()
        let stepIndex = traceStep?.index ?? sequence
        let reference = AgentLessonReference(run: runID, step: stepIndex)
        let step = Self.abstract(arguments: arguments, before: packet, after: after,
            outcome: body, reference: reference, subgoal: subgoal, state: controlState,
            versions: versions, now: now)
        if step.before.contains(.deliveryUnknown) || step.after.contains(.deliveryUnknown)
            || body["delivery_unknown"] as? Bool == true {
            discardEpisode()
        } else {
            let additions = policy.settle(step)
            if !additions.isEmpty, let store {
                do {
                    try await store.merge(additions, now: now, compatibleVersions: versions)
                    let persisted = await store.load(now: now, compatibleVersions: versions)
                    for addition in additions {
                        if let lesson = persisted.first(where: { $0.family == addition.family }) {
                            await trace?.lessonsWritten(family: lesson.family,
                                status: lesson.status, count: lesson.count)
                        }
                    }
                } catch {
                    await trace?.lessonsStoreError()
                }
            }
            if step.outcome == .rejected, step.action == .type {
                previousError = step.shape == .mixedTargetAndPoint ? .callShape : .unavailableTarget
                if currentEpisode == nil { currentEpisode = reference }
            } else if step.action == .refresh {
                previousError = nil
                if step.before.contains(.fieldAbsent), currentEpisode == nil { currentEpisode = reference }
            } else {
                previousError = step.outcome == .inconclusive ? .inconclusive : nil
            }
            let action = arguments["action"] as? String ?? ""
            if step.outcome != .rejected,
               !["type", "observe", "screenshot"].contains(action),
               !AgentLessonReplay.isFieldFocus(arguments, packet: packet) {
                discardEpisode()
            }
        }
        packet = after
    }

    /// Replaces one optional field in this packet. No text is accumulated in
    /// later packets, and choices, allowed operations and guidance are untouched.
    func advising(_ result: AppToolResult, now: Date = Date()) async -> AppToolResult {
        guard mode != nil, let store, result.name == VisionCaptureToolDefinitions.navigateName,
              var body = Self.decode(result.content), body["schema_version"] as? Int == 1 else { return result }
        await reportMissingEncoder()
        let hadBlock = body.removeValue(forKey: "lessons") != nil
        var situation = AgentLessonReplay.conditions(body)
        if previousError == .callShape { situation.insert(.callShapeRejected) }
        let query = AgentLessonQuery(
            controlType: situation.contains(.nativeAlertPresent) ? .systemAlert : .editableField,
            situation: situation, error: situation.contains(.fieldAbsent) ? .missingField : previousError,
            versions: AgentLessonReplay.versions(body),
            permitted: situation.intersection([.freshReadPermitted, .pointTypingPermitted, .typingPermitted]),
            evidenceKinds: [.hostProof, .loopRejection, .observation],
            currentEpisode: currentEpisode, now: now)
        let lessons = await store.load(now: now, compatibleVersions: query.versions)
        let retrieval: AgentLessonRetrieval
        if mode == .embedding, let encoder {
            retrieval = (try? EmbeddingLessonRetriever(encoder: encoder).retrieve(query, from: lessons))
                ?? .bounded([])
        } else {
            retrieval = TextLessonRetriever().retrieve(query, from: lessons)
        }
        // Include the heading and delimiters in the conservative token bound.
        var selected: [AgentLesson] = []
        var block = Self.heading
        for lesson in retrieval.lessons.prefix(2) {
            let proposed = block + "\n" + lesson.family.kind.rawValue + ": " + lesson.renderedText
            guard proposed.utf8.count <= 192 else { continue }
            selected.append(lesson)
            block = proposed
        }
        if !selected.isEmpty { body["lessons"] = block }
        await trace?.lessonsRetrieved(families: selected.map(\.family),
            tokenCount: selected.isEmpty ? 0 : block.utf8.count)
        guard hadBlock || !selected.isEmpty,
              let data = try? JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]),
              let content = String(data: data, encoding: .utf8) else { return result }
        return AppToolResult(callID: result.callID, name: result.name,
            content: content, imageAttachments: result.imageAttachments)
    }

    private func reportMissingEncoder() async {
        guard mode == .embedding, encoder == nil, !reportedMissingEncoder else { return }
        reportedMissingEncoder = true
        await trace?.lessonsEncoderMissing()
    }

    private func discardEpisode() {
        policy.discardPendingEpisodes()
        subgoal = UUID()
        controlState = UUID()
        desiredTextDigest = nil
        previousError = nil
        currentEpisode = nil
    }

    private static func decode(_ text: String) -> [String: Any]? {
        try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
    }

    /// Shares the replay's field bindings, exact-value checks and closed query
    /// vocabulary. Only explicit settled host evidence admits a live repair.
    private static func abstract(arguments: [String: Any], before: [String: Any],
                                 after: [String: Any], outcome: [String: Any],
                                 reference: AgentLessonReference, subgoal: UUID, state: UUID,
                                 versions: AgentLessonVersions, now: Date) -> AgentLessonStep {
        let rawAction = arguments["action"] as? String ?? ""
        let action: AgentLessonStep.Action = rawAction == "type" ? .type
            : (["observe", "screenshot"].contains(rawAction) ? .refresh : .other)
        let target = arguments["target"] as? String
        let chosen = AgentLessonReplay.choices(before).filter { $0["id"] as? String == target && target != nil }
        let choiceBound = chosen.count == 1 && chosen.first.map(AgentLessonReplay.isEditable) == true
        let pointBound = target == nil && AgentLessonReplay.pointMatchesField(arguments, packet: before)
        let shape: AgentLessonStep.Shape = action != .type ? .other
            : (target != nil && (arguments["x_norm"] != nil || arguments["y_norm"] != nil) ? .mixedTargetAndPoint
                : (pointBound ? .pointOnly : (choiceBound ? .currentChoice : .unavailable)))
        let last = after["last_action"] as? [String: Any] ?? [:]
        let verdict = last["verdict"] as? String ?? ""
        let settledOutcome = outcome["outcome"] as? String ?? ""
        let typeSent = action == .type && outcome["operation"] as? String == "type"
            && ["succeeded", "failed", "inconclusive", "unverified"].contains(settledOutcome)
            && outcome["dispatch_attempted"] as? Bool != false
        let proof = outcome["proof"] as? [String: Any] ?? [:]
        let positiveHost = typeSent && outcome["is_error"] as? Bool != true
            && proof["action"] as? String == "type" && proof["verdict"] as? String == "verified"
            && proof["reason_code"] as? String == "LEGACY_FIELD_VALUE_CHANGED"
        let observation = after["observation"] as? [String: Any] ?? [:]
        let fresh = observation["state"] as? String == "current"
            && observation["read_result"] as? String == "succeeded"
            && outcome["observation_outcome"] as? String == "succeeded"
            && outcome["is_error"] as? Bool != true
        return AgentLessonStep(reference: reference, observationReference: reference,
            subgoal: subgoal, controlState: state, versions: versions, time: now,
            action: action, shape: shape,
            outcome: verdict == "not_sent" ? .rejected
                : (action == .refresh && verdict == "observed" ? .observed
                    : (typeSent ? .admitted : (verdict == "failed" ? .failed : .inconclusive))),
            before: AgentLessonReplay.conditions(before), after: AgentLessonReplay.conditions(after),
            observationIsFresh: fresh,
            targetBound: choiceBound || pointBound,
            hostVerifiedValue: positiveHost && verdict == "verified"
                && AgentLessonReplay.valueMatches(arguments, before: before, after: after, chosen: chosen.first),
            verdictConflict: verdict == "verified" && typeSent && !positiveHost)
    }
}
#endif
