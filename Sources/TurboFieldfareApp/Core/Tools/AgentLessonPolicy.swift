import Foundation

/// Produced by a trusted settled-result adapter, never by a model reflection.
/// Subgoal/control-state tokens are ephemeral equality keys, not selectors.
struct AgentLessonStep: Sendable {
    enum Action: String, Sendable { case type, refresh, other }
    enum Shape: Sendable { case mixedTargetAndPoint, pointOnly, currentChoice, unavailable, other }
    enum Outcome: Sendable { case rejected, failed, inconclusive, admitted, observed }

    let reference: AgentLessonReference
    let observationReference: AgentLessonReference
    let subgoal: UUID
    let controlState: UUID
    let versions: AgentLessonVersions
    let time: Date
    let action: Action
    let shape: Shape
    let outcome: Outcome
    let before: Set<AgentLessonCondition>
    let after: Set<AgentLessonCondition>
    let observationIsFresh: Bool
    let targetBound: Bool
    let hostVerifiedValue: Bool
    let verdictConflict: Bool
}

/// Deliberately narrow: unknown causes, delayed effects, missing/redacted values,
/// changed builds and uncertain delivery never establish an action lesson.
struct AgentLessonPolicy: Sendable {
    private struct Scope: Hashable, Sendable {
        let run: UUID
        let subgoal: UUID
        let state: UUID
        let versions: AgentLessonVersions
    }
    private var failures: [Scope: AgentLessonStep] = [:]
    private var episodeAnchors: [Scope: AgentLessonReference] = [:]
    private var recoveredFields: [Scope: AgentLessonStep] = [:]
    private(set) var lessons: [AgentLesson] = []

    /// Returns new run support after deduplication, including promotions.
    /// A settled observation is independently useful; it need not prove typing.
    mutating func settle(_ step: AgentLessonStep) -> [AgentLesson] {
        guard !step.before.contains(.deliveryUnknown), !step.after.contains(.deliveryUnknown) else {
            discardPendingEpisodes()
            return []
        }
        let key = Scope(run: step.reference.run, subgoal: step.subgoal,
                        state: step.controlState, versions: step.versions)
        var written: [AgentLesson] = []
        if episodeAnchors[key] == nil,
           (step.action == .type && step.outcome == .rejected
            && (step.shape == .mixedTargetAndPoint || step.shape == .unavailable))
            || (step.action == .refresh && step.before.isSuperset(of: [.fieldAbsent, .freshReadPermitted])) {
            episodeAnchors[key] = step.reference
        }
        // A permitted fresh read with an absent-to-present transition proves only discovery.
        if step.action == .refresh, step.outcome == .observed, step.observationIsFresh,
           step.before.isSuperset(of: [.fieldAbsent, .freshReadPermitted]),
           step.after.contains(.editableChoicePresent), !step.before.contains(.deliveryUnknown) {
            let failure = failures[key] ?? step
            append(.restoreEditableControls, failure: failure, repair: step, into: &written)
            recoveredFields[key] = failure
            failures[key] = nil
        }
        if step.action == .type, step.outcome == .rejected,
           step.shape == .mixedTargetAndPoint || step.shape == .unavailable {
            // The first rejection starts an episode; retries cannot add support.
            if failures[key] == nil { failures[key] = step }
            return written
        }
        guard step.action == .type, step.outcome == .admitted,
              !step.before.contains(.deliveryUnknown), step.targetBound else { return written }

        if let failure = failures[key], failure.shape == .mixedTargetAndPoint {
            if step.shape == .pointOnly,
               step.before.isSuperset(of: [.ocrFieldPresent, .pointTypingPermitted]) {
                append(.separatePointTyping, failure: failure, repair: step, into: &written)
                if step.hostVerifiedValue, !step.verdictConflict {
                    append(.verifiedPointTyping, failure: failure, repair: step, into: &written)
                }
                failures[key] = nil
            } else if step.shape == .currentChoice,
                      step.before.isSuperset(of: [.editableChoicePresent, .typingPermitted]) {
                append(.currentChoiceTyping, failure: failure, repair: step, into: &written)
                failures[key] = nil
            }
        }
        if let failure = recoveredFields[key], step.shape == .currentChoice,
           step.before.isSuperset(of: [.editableChoicePresent, .typingPermitted]),
           step.hostVerifiedValue, !step.verdictConflict {
            append(.freshFieldTyping, failure: failure, repair: step, into: &written)
            recoveredFields[key] = nil
        }
        return written
    }

    mutating func quarantine(family: AgentLessonFamily, versions: AgentLessonVersions,
                             contradiction: AgentLessonReference) {
        guard let index = lessons.firstIndex(where: { $0.family == family && $0.versions == versions }) else { return }
        lessons[index].contradict(at: contradiction)
    }

    /// Call when a user instruction, task, owner or host build changes.
    mutating func discardPendingEpisodes() {
        failures.removeAll()
        recoveredFields.removeAll()
        episodeAnchors.removeAll()
    }

    private mutating func append(_ family: AgentLessonFamily, failure: AgentLessonStep,
                                 repair: AgentLessonStep, into writes: inout [AgentLesson]) {
        let scope = Scope(run: failure.reference.run, subgoal: failure.subgoal,
                          state: failure.controlState, versions: failure.versions)
        let evidence = AgentLessonEvidence(
            failure: episodeAnchors[scope] ?? failure.reference, repair: repair.reference,
            observation: repair.observationReference, kind: family.kind, predicate: family.predicate,
            verdictConflict: repair.verdictConflict)
        let addition = AgentLesson(family: family, versions: repair.versions,
                                  evidence: evidence, lastSeen: repair.time)
        guard addition.isValid else { return }
        if let index = lessons.firstIndex(where: { $0.family == family && $0.versions == repair.versions }) {
            let previousCount = lessons[index].count
            lessons[index].merge(addition)
            if lessons[index].count > previousCount { writes.append(lessons[index]) }
        } else {
            lessons.append(addition)
            writes.append(addition)
        }
    }
}
