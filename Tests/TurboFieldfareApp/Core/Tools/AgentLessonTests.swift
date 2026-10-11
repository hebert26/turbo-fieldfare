import Foundation
import Testing
@testable import TurboFieldfareAppCore

// MARK: - Helpers

private let now = Date(timeIntervalSince1970: 1_800_000_000)
private let day: TimeInterval = 86_400
private let pointSituation: Set<AgentLessonCondition> =
    [.callShapeRejected, .ocrFieldPresent, .pointTypingPermitted]
private let choiceSituation: Set<AgentLessonCondition> =
    [.callShapeRejected, .editableChoicePresent, .typingPermitted]
private let freshSituation: Set<AgentLessonCondition> = [.fieldAbsent, .freshReadPermitted]

private func ref(_ run: UUID, _ step: Int) -> AgentLessonReference {
    AgentLessonReference(run: run, step: step)
}

private func evidence(_ family: AgentLessonFamily, run: UUID = UUID(), step: Int = 1,
                      conflict: Bool = false) -> AgentLessonEvidence {
    AgentLessonEvidence(failure: ref(run, step), repair: ref(run, step + 1),
                        observation: ref(run, step + 2), kind: family.kind,
                        predicate: family.predicate, verdictConflict: conflict)
}

private func lesson(_ family: AgentLessonFamily, run: UUID = UUID(), step: Int = 1,
                    versions: AgentLessonVersions = .init(), at date: Date = now) -> AgentLesson {
    AgentLesson(family: family, versions: versions,
                evidence: evidence(family, run: run, step: step), lastSeen: date)
}

/// Two independent episodes: active.
private func active(_ family: AgentLessonFamily, versions: AgentLessonVersions = .init(),
                    at date: Date = now, extra: Int = 0) -> AgentLesson {
    var result = lesson(family, versions: versions, at: date)
    for _ in 0...extra { result.merge(lesson(family, versions: versions, at: date)) }
    return result
}

private func query(_ situation: Set<AgentLessonCondition>, error: AgentLessonError? = .callShape,
                   control: AgentLessonControl = .editableField,
                   permitted: Set<AgentLessonCondition>? = nil,
                   kinds: Set<AgentLessonEvidenceKind> = [.hostProof, .loopRejection, .observation],
                   episode: AgentLessonReference? = nil,
                   versions: AgentLessonVersions = .init(),
                   at date: Date = now) -> AgentLessonQuery {
    AgentLessonQuery(
        controlType: control, situation: situation, error: error, versions: versions,
        permitted: permitted ?? situation.intersection(
            [.freshReadPermitted, .pointTypingPermitted, .typingPermitted]),
        evidenceKinds: kinds, currentEpisode: episode, now: date)
}

private struct Scope {
    var run = UUID()
    var subgoal = UUID()
    var state = UUID()
    var versions = AgentLessonVersions()
}

private func step(_ scope: Scope, _ n: Int, _ action: AgentLessonStep.Action,
                  _ shape: AgentLessonStep.Shape, _ outcome: AgentLessonStep.Outcome,
                  before: Set<AgentLessonCondition> = [], after: Set<AgentLessonCondition> = [],
                  fresh: Bool = false, bound: Bool = true, hostValue: Bool = false,
                  conflict: Bool = false) -> AgentLessonStep {
    AgentLessonStep(
        reference: ref(scope.run, n), observationReference: ref(scope.run, n + 1),
        subgoal: scope.subgoal, controlState: scope.state, versions: scope.versions,
        time: now, action: action, shape: shape, outcome: outcome, before: before, after: after,
        observationIsFresh: fresh, targetBound: bound, hostVerifiedValue: hostValue,
        verdictConflict: conflict)
}

private func mixedRejection(_ s: Scope, _ n: Int,
                            before: Set<AgentLessonCondition> = pointSituation) -> AgentLessonStep {
    step(s, n, .type, .mixedTargetAndPoint, .rejected, before: before)
}

private func pointRepair(_ s: Scope, _ n: Int, hostValue: Bool = false, conflict: Bool = false,
                         bound: Bool = true,
                         before: Set<AgentLessonCondition> = [.ocrFieldPresent, .pointTypingPermitted])
    -> AgentLessonStep {
    step(s, n, .type, .pointOnly, .admitted, before: before, bound: bound,
         hostValue: hostValue, conflict: conflict)
}

private func choiceRepair(_ s: Scope, _ n: Int, hostValue: Bool = false) -> AgentLessonStep {
    step(s, n, .type, .currentChoice, .admitted,
         before: [.editableChoicePresent, .typingPermitted], hostValue: hostValue)
}

private func freshRead(_ s: Scope, _ n: Int, fresh: Bool = true,
                       before: Set<AgentLessonCondition> = freshSituation,
                       after: Set<AgentLessonCondition> = [.editableChoicePresent, .typingPermitted])
    -> AgentLessonStep {
    step(s, n, .refresh, .other, .observed, before: before, after: after, fresh: fresh)
}

private func families(_ written: [AgentLesson]) -> [AgentLessonFamily] { written.map(\.family) }

private func withTempURL<T>(_ body: (URL) async throws -> T) async throws -> T {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("agent-lesson-tests-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    return try await body(directory.appendingPathComponent("agent-lessons.json"))
}

private final class FakeEncoder: LessonTextEncoding, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [String] = []
    var documentVector: @Sendable (String) -> [Float] = { _ in FakeEncoder.basis(0) }
    var queryVector: [Float] = FakeEncoder.basis(0)
    var failure: (any Error)?

    struct Failure: Error {}

    var inputs: [String] { lock.lock(); defer { lock.unlock() }; return recorded }

    func encode(_ prefixedText: String) throws -> [Float] {
        lock.lock(); recorded.append(prefixedText); lock.unlock()
        if let failure { throw failure }
        return prefixedText.hasPrefix("task: search result | query: ")
            ? queryVector : documentVector(prefixedText)
    }

    static func basis(_ index: Int, scale: Float = 1) -> [Float] {
        var values = [Float](repeating: 0, count: 256)
        values[index] = scale
        return values
    }

    /// One orthogonal axis per family, so each lesson has its own direction.
    static func perFamily(_ text: String) -> [Float] {
        for (index, family) in AgentLessonFamily.allCases.enumerated()
        where text.hasPrefix("title: \(family.rawValue) |") {
            return basis(index, scale: 3)
        }
        return basis(200)
    }

    /// Unit vector with exactly the requested cosine to axis `axis`.
    static func cosine(_ value: Double, axis: Int) -> [Float] {
        var values = [Float](repeating: 0, count: 256)
        values[axis] = Float(value)
        values[100] = Float((1 - value * value).squareRoot())
        return values
    }
}

private func axis(_ family: AgentLessonFamily) -> Int {
    AgentLessonFamily.allCases.firstIndex(of: family)!
}

// MARK: - Record

@Suite struct AgentLessonRecordTests {
    @Test func everyTemplateIsShortAndMatchesItsDeclaredFacts() {
        for family in AgentLessonFamily.allCases {
            let record = lesson(family)
            #expect(record.tokenUpperBound <= 80)
            #expect(record.renderedText == "\(family.instruction) \(family.avoidance)")
            #expect(record.controlType == .editableField)
            #expect(record.family.kind == family.kind)
            #expect(record.status == .candidate)
            #expect(record.count == 1)
            #expect(record.contradictionCount == 0)
            #expect(record.isValid)
        }
    }

    @Test func evidenceKindDeterminesThePredicate() {
        #expect(AgentLessonFamily.separatePointTyping.kind == .loopRejection)
        #expect(AgentLessonFamily.separatePointTyping.predicate == .callShapeAdmitted)
        #expect(AgentLessonFamily.currentChoiceTyping.kind == .loopRejection)
        #expect(AgentLessonFamily.verifiedPointTyping.kind == .hostProof)
        #expect(AgentLessonFamily.freshFieldTyping.predicate == .fieldValueVerified)
        #expect(AgentLessonFamily.restoreEditableControls.kind == .observation)
        #expect(AgentLessonFamily.restoreEditableControls.predicate == .editableControlsRestored)
    }

    @Test func promptBearingTextHoldsOnlyGenericVocabulary() {
        let allowed = CharacterSet.letters.union(CharacterSet(charactersIn: " ;.:|-,"))
        var texts: [String] = []
        for family in AgentLessonFamily.allCases {
            let record = lesson(family)
            texts += [record.renderedText, record.embeddingDocument, family.rawValue]
        }
        texts += AgentLessonCondition.allCases.map(\.rawValue)
        texts += AgentLessonControl.allCases.map(\.rawValue)
        for text in texts {
            #expect(text.unicodeScalars.allSatisfy(allowed.contains), "unexpected character in \(text)")
            #expect(!text.contains("http"))
            #expect(text.rangeOfCharacter(from: .decimalDigits) == nil)
        }
        let every = query(Set(AgentLessonCondition.allCases))
        #expect(every.sanitizedText.rangeOfCharacter(from: .decimalDigits) == nil)
        #expect(every.embeddingQuery.hasPrefix("task: search result | query: "))
    }

    @Test func persistedRecordContainsOnlyOpaqueReferencesAndTypedFields() throws {
        let data = try JSONEncoder().encode(lesson(.separatePointTyping))
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(Set(object.keys) == [
            "family", "versions", "evidence", "last_seen", "status", "contradictions",
            "control_type", "situation", "do", "avoid", "count", "contradiction_count",
        ])
        let items = try #require(object["evidence"] as? [[String: Any]])
        #expect(Set(items[0].keys) == [
            "failure", "repair", "observation", "kind", "predicate", "verdictConflict",
        ])
        let failure = try #require(items[0]["failure"] as? [String: Any])
        #expect(Set(failure.keys) == ["run", "step"])
    }

    @Test func recordRoundTripsThroughJSON() throws {
        var original = active(.currentChoiceTyping)
        original.contradict(at: ref(UUID(), 4))
        let decoded = try JSONDecoder().decode(AgentLesson.self,
                                               from: try JSONEncoder().encode(original))
        #expect(decoded == original)
        #expect(decoded.status == .quarantined)
    }

    @Test func editedProseOrCountsFailToDecode() throws {
        let data = try JSONEncoder().encode(lesson(.separatePointTyping))
        let json = try #require(String(data: data, encoding: .utf8))
        let edits = [
            json.replacingOccurrences(of: "Use permitted point typing.", with: "Tap the Save button."),
            json.replacingOccurrences(of: "\"count\":1", with: "\"count\":9"),
            json.replacingOccurrences(of: "\"control_type\":\"editable field\"",
                                      with: "\"control_type\":\"system alert\""),
            json.replacingOccurrences(of: "\"status\":\"candidate\"", with: "\"status\":\"active\""),
        ]
        for edited in edits {
            #expect(edited != json)
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(AgentLesson.self, from: Data(edited.utf8))
            }
        }
    }

    @Test func recordWithMismatchedEvidenceKindIsInvalid() {
        let wrong = AgentLessonEvidence(
            failure: ref(UUID(), 1), repair: ref(UUID(), 2), observation: ref(UUID(), 3),
            kind: .hostProof, predicate: .fieldValueVerified, verdictConflict: false)
        #expect(!AgentLesson(family: .separatePointTyping, evidence: wrong, lastSeen: now).isValid)
    }

    @Test func hostProofWithAVerdictConflictIsInvalid() {
        let record = AgentLesson(family: .verifiedPointTyping,
                                 evidence: evidence(.verifiedPointTyping, conflict: true),
                                 lastSeen: now)
        #expect(!record.isValid)
    }

    @Test func loopRejectionKeepsAVerdictConflictWithoutBecomingHostProof() {
        let record = AgentLesson(family: .separatePointTyping,
                                 evidence: evidence(.separatePointTyping, conflict: true),
                                 lastSeen: now)
        #expect(record.isValid)
        #expect(record.family.kind == .loopRejection)
    }

    @Test func orderedReferencesAreRequired() {
        let run = UUID()
        let backwards = AgentLessonEvidence(
            failure: ref(run, 9), repair: ref(run, 2), observation: ref(run, 10),
            kind: .loopRejection, predicate: .callShapeAdmitted, verdictConflict: false)
        #expect(!AgentLesson(family: .separatePointTyping, evidence: backwards, lastSeen: now).isValid)
        let crossRun = AgentLessonEvidence(
            failure: ref(run, 1), repair: ref(UUID(), 2), observation: ref(run, 3),
            kind: .loopRejection, predicate: .callShapeAdmitted, verdictConflict: false)
        #expect(!AgentLesson(family: .separatePointTyping, evidence: crossRun, lastSeen: now).isValid)
    }

    @Test func mergingTheSameEpisodeCountsOnce() {
        let run = UUID()
        var record = lesson(.separatePointTyping, run: run, step: 5)
        record.merge(lesson(.separatePointTyping, run: run, step: 5))
        #expect(record.count == 1)
        #expect(record.status == .candidate)
    }

    @Test func twoDistinctEpisodesPromoteToActive() {
        var record = lesson(.separatePointTyping)
        record.merge(lesson(.separatePointTyping))
        #expect(record.count == 2)
        #expect(record.status == .active)
        #expect(record.isValid)
    }

    @Test func mergeIgnoresADifferentFamilyOrVersion() {
        var record = lesson(.separatePointTyping)
        record.merge(lesson(.currentChoiceTyping))
        record.merge(lesson(.separatePointTyping, versions: .init(record: 1, tool: 2, proof: 1)))
        #expect(record.count == 1)
    }

    @Test func aContradictionQuarantinesAndLaterSupportDoesNotReactivate() {
        var record = active(.separatePointTyping)
        record.contradict(at: ref(UUID(), 7))
        #expect(record.status == .quarantined)
        #expect(record.contradictionCount == 1)
        record.merge(lesson(.separatePointTyping))
        record.merge(lesson(.separatePointTyping))
        #expect(record.count == 4)
        #expect(record.status == .quarantined)
        #expect(record.isValid)
    }

    @Test func contradictionsSurviveMergingIntoACleanCopy() {
        var clean = lesson(.separatePointTyping)
        var contradicted = lesson(.separatePointTyping)
        contradicted.contradict(at: ref(UUID(), 3))
        clean.merge(contradicted)
        #expect(clean.status == .quarantined)
    }

    @Test func rankingWeightHalvesEveryThirtyDaysWithoutSupport() {
        let record = active(.separatePointTyping)
        let base = record.rankingWeight(at: now)
        #expect(base == 2)
        #expect(record.rankingWeight(at: now.addingTimeInterval(29 * day)) == base)
        #expect(record.rankingWeight(at: now.addingTimeInterval(30 * day)) == base / 2)
        #expect(record.rankingWeight(at: now.addingTimeInterval(60 * day)) == base / 4)
        #expect(record.rankingWeight(at: now.addingTimeInterval(-5 * day)) == base)
    }

    @Test func vectorsAreNormalizedAndFixedAt256Dimensions() throws {
        let vector = try #require(AgentLessonVector(normalizing: FakeEncoder.basis(3, scale: 7)))
        #expect(vector.values.count == 256)
        #expect(vector.isValid)
        #expect(abs(vector.values[3] - 1) < 1e-6)
        #expect(vector.documentPrefix == "title: … | text: …")
        #expect(vector.queryPrefix == "task: search result | query: …")
        #expect(AgentLessonVector(normalizing: [Float](repeating: 1, count: 255)) == nil)
        #expect(AgentLessonVector(normalizing: [Float](repeating: 0, count: 256)) == nil)
        var bad = FakeEncoder.basis(0)
        bad[5] = .nan
        #expect(AgentLessonVector(normalizing: bad) == nil)
        bad[5] = .infinity
        #expect(AgentLessonVector(normalizing: bad) == nil)
    }
}

// MARK: - Policy

@Suite struct AgentLessonPolicyTests {
    @Test func mixedRejectionThenPointRepairWritesACallShapeLessonOnly() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        #expect(policy.settle(mixedRejection(s, 1)).isEmpty)
        let written = policy.settle(pointRepair(s, 2))
        #expect(families(written) == [.separatePointTyping])
        #expect(written[0].family.kind == .loopRejection)
        #expect(written[0].status == .candidate)
        #expect(written[0].count == 1)
    }

    @Test func hostVerifiedValueAddsAHostProofLesson() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        let written = policy.settle(pointRepair(s, 2, hostValue: true))
        #expect(families(written) == [.separatePointTyping, .verifiedPointTyping])
        #expect(written[1].family.kind == .hostProof)
    }

    @Test func aVerdictConflictBlocksHostProofButKeepsCallShape() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        let written = policy.settle(pointRepair(s, 2, hostValue: true, conflict: true))
        #expect(families(written) == [.separatePointTyping])
        #expect(written[0].evidence[0].verdictConflict)
    }

    @Test func noHostValueNeverProducesAHostProofLesson() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        _ = policy.settle(pointRepair(s, 2, hostValue: false))
        #expect(!policy.lessons.contains { $0.family.kind == .hostProof })
    }

    @Test func currentChoiceRepairWritesTheCurrentChoiceLesson() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1, before: choiceSituation))
        let written = policy.settle(choiceRepair(s, 2))
        #expect(families(written) == [.currentChoiceTyping])
        #expect(written[0].family.kind == .loopRejection)
    }

    @Test func repairWithoutBoundTargetIsNotAdmitted() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        #expect(policy.settle(pointRepair(s, 2, hostValue: true, bound: false)).isEmpty)
        #expect(policy.lessons.isEmpty)
    }

    @Test func uncertainDeliveryIsNeverAdmitted() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        let uncertain = pointRepair(s, 2, hostValue: true,
                                    before: [.ocrFieldPresent, .pointTypingPermitted, .deliveryUnknown])
        #expect(policy.settle(uncertain).isEmpty)
    }

    @Test func repairMissingItsPermissionPreconditionIsNotAdmitted() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        #expect(policy.settle(pointRepair(s, 2, before: [.ocrFieldPresent])).isEmpty)
        #expect(policy.settle(pointRepair(s, 3, before: [.pointTypingPermitted])).isEmpty)
    }

    @Test func aRepairWithoutAnEarlierFailureWritesNothing() {
        var policy = AgentLessonPolicy()
        #expect(policy.settle(pointRepair(Scope(), 2, hostValue: true)).isEmpty)
        #expect(policy.settle(choiceRepair(Scope(), 3, hostValue: true)).isEmpty)
    }

    @Test func aRejectionAloneAndAnUnavailableTargetNeverWriteALesson() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        #expect(policy.settle(mixedRejection(s, 1)).isEmpty)
        #expect(policy.settle(step(s, 2, .type, .unavailable, .rejected)).isEmpty)
        #expect(policy.settle(step(s, 3, .type, .other, .failed)).isEmpty)
        #expect(policy.settle(step(s, 4, .type, .other, .inconclusive)).isEmpty)
        #expect(policy.lessons.isEmpty)
    }

    @Test func aFailedOrInconclusiveRepairIsNotASuccess() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        for outcome in [AgentLessonStep.Outcome.failed, .inconclusive, .rejected] {
            #expect(policy.settle(step(s, 2, .type, .pointOnly, outcome,
                                       before: [.ocrFieldPresent, .pointTypingPermitted],
                                       hostValue: true)).isEmpty)
        }
        #expect(policy.lessons.isEmpty)
    }

    @Test func nonTypingActionsNeverRepairAnEpisode() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        #expect(policy.settle(step(s, 2, .other, .other, .admitted, hostValue: true)).isEmpty)
        #expect(policy.lessons.isEmpty)
    }

    @Test func differentScopesNeverPair() {
        let base = Scope()
        let others = [
            Scope(run: UUID(), subgoal: base.subgoal, state: base.state, versions: base.versions),
            Scope(run: base.run, subgoal: UUID(), state: base.state, versions: base.versions),
            Scope(run: base.run, subgoal: base.subgoal, state: UUID(), versions: base.versions),
            Scope(run: base.run, subgoal: base.subgoal, state: base.state,
                  versions: .init(record: 1, tool: 2, proof: 1)),
            Scope(run: base.run, subgoal: base.subgoal, state: base.state,
                  versions: .init(record: 1, tool: 1, proof: 2)),
        ]
        for other in others {
            var policy = AgentLessonPolicy()
            _ = policy.settle(mixedRejection(base, 1))
            #expect(policy.settle(pointRepair(other, 2)).isEmpty)
            #expect(policy.lessons.isEmpty)
        }
    }

    @Test func discardingPendingEpisodesDropsThemButKeepsAdmittedLessons() {
        var policy = AgentLessonPolicy()
        let kept = Scope()
        _ = policy.settle(mixedRejection(kept, 1))
        _ = policy.settle(pointRepair(kept, 2))
        let pending = Scope()
        _ = policy.settle(mixedRejection(pending, 10))
        policy.discardPendingEpisodes()
        #expect(policy.settle(pointRepair(pending, 11)).isEmpty)
        #expect(policy.lessons.count == 1)
        #expect(policy.lessons[0].count == 1)
    }

    @Test func retriedRejectionsInOneEpisodeCountOnce() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        _ = policy.settle(mixedRejection(s, 2))
        _ = policy.settle(mixedRejection(s, 3))
        let written = policy.settle(pointRepair(s, 4))
        #expect(written.count == 1)
        #expect(written[0].count == 1)
        #expect(written[0].evidence[0].failure == ref(s.run, 1))
    }

    @Test func repeatedRepairInsideOneScopeAddsNoSupport() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        _ = policy.settle(pointRepair(s, 2))
        _ = policy.settle(mixedRejection(s, 3))
        let again = policy.settle(pointRepair(s, 4))
        #expect(again.isEmpty)
        #expect(policy.lessons[0].count == 1)
        #expect(policy.lessons[0].status == .candidate)
    }

    @Test func aSecondIndependentEpisodePromotesTheLesson() {
        var policy = AgentLessonPolicy()
        let first = Scope()
        _ = policy.settle(mixedRejection(first, 1))
        #expect(policy.settle(pointRepair(first, 2))[0].status == .candidate)
        let second = Scope()
        _ = policy.settle(mixedRejection(second, 1))
        let promoted = policy.settle(pointRepair(second, 2))
        #expect(promoted.count == 1)
        #expect(promoted[0].count == 2)
        #expect(promoted[0].status == .active)
        #expect(policy.lessons.count == 1)
    }

    @Test func familiesStaySeparateAndVersionsStaySeparate() {
        var policy = AgentLessonPolicy()
        let a = Scope()
        _ = policy.settle(mixedRejection(a, 1))
        _ = policy.settle(pointRepair(a, 2, hostValue: true))
        let b = Scope(versions: .init(record: 1, tool: 2, proof: 1))
        _ = policy.settle(mixedRejection(b, 1))
        _ = policy.settle(pointRepair(b, 2))
        #expect(policy.lessons.count == 3)
        #expect(policy.lessons.filter { $0.family == .separatePointTyping }.count == 2)
        #expect(policy.lessons.allSatisfy { $0.count == 1 })
    }

    @Test func aFreshReadThatRestoresControlsWritesAnObservationLesson() {
        var policy = AgentLessonPolicy()
        let written = policy.settle(freshRead(Scope(), 5))
        #expect(families(written) == [.restoreEditableControls])
        #expect(written[0].family.kind == .observation)
        #expect(written[0].family.predicate == .editableControlsRestored)
    }

    @Test func aFreshReadWithoutProofOfRestoredControlsWritesNothing() {
        var policy = AgentLessonPolicy()
        #expect(policy.settle(freshRead(Scope(), 1, fresh: false)).isEmpty)
        #expect(policy.settle(freshRead(Scope(), 2, before: [.fieldAbsent])).isEmpty)
        #expect(policy.settle(freshRead(Scope(), 3, before: [.freshReadPermitted])).isEmpty)
        #expect(policy.settle(freshRead(Scope(), 4, after: [.typingPermitted])).isEmpty)
        #expect(policy.settle(freshRead(Scope(), 5, after: [])).isEmpty)
        #expect(policy.settle(freshRead(
            Scope(), 6, before: [.fieldAbsent, .freshReadPermitted, .deliveryUnknown])).isEmpty)
        #expect(policy.settle(step(Scope(), 7, .refresh, .other, .inconclusive,
                                   before: freshSituation, after: [.editableChoicePresent],
                                   fresh: true)).isEmpty)
        #expect(policy.lessons.isEmpty)
    }

    @Test func observationLessonDoesNotClaimThatTypingFollowed() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(freshRead(s, 1))
        #expect(!policy.lessons.contains { $0.family.kind == .hostProof })
        #expect(policy.lessons.count == 1)
    }

    @Test func aFreshReadFollowedByVerifiedTypingAddsAHostProofCandidate() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(freshRead(s, 1))
        let written = policy.settle(choiceRepair(s, 3, hostValue: true))
        #expect(families(written) == [.freshFieldTyping])
        #expect(written[0].family.kind == .hostProof)
        #expect(written[0].status == .candidate)
    }

    @Test func typingAfterAFreshReadWithoutAHostValueIsNotHostProof() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(freshRead(s, 1))
        #expect(policy.settle(choiceRepair(s, 3, hostValue: false)).isEmpty)
        #expect(policy.settle(step(s, 4, .type, .currentChoice, .admitted,
                                   before: [.editableChoicePresent, .typingPermitted],
                                   hostValue: true, conflict: true)).isEmpty)
        #expect(policy.lessons.map(\.family) == [.restoreEditableControls])
    }

    @Test func aRejectedProposalFollowedByAFreshReadIsOneObservationEpisode() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(step(s, 1, .type, .unavailable, .rejected, before: freshSituation))
        let written = policy.settle(freshRead(s, 2))
        #expect(families(written) == [.restoreEditableControls])
        #expect(written[0].evidence[0].failure == ref(s.run, 1))
    }

    @Test func quarantineMarksALessonAndLaterSupportCannotReactivateIt() {
        var policy = AgentLessonPolicy()
        let first = Scope()
        _ = policy.settle(mixedRejection(first, 1))
        _ = policy.settle(pointRepair(first, 2))
        policy.quarantine(family: .separatePointTyping, versions: .init(), contradiction: ref(UUID(), 9))
        #expect(policy.lessons[0].status == .quarantined)
        let second = Scope()
        _ = policy.settle(mixedRejection(second, 1))
        let after = policy.settle(pointRepair(second, 2))
        #expect(after[0].count == 2)
        #expect(after[0].status == .quarantined)
    }

    @Test func quarantineOfAnUnknownFamilyIsIgnored() {
        var policy = AgentLessonPolicy()
        policy.quarantine(family: .separatePointTyping, versions: .init(), contradiction: ref(UUID(), 1))
        #expect(policy.lessons.isEmpty)
    }

    @Test func everyWrittenLessonIsValidAndLeaksNothingFromTheStep() {
        var policy = AgentLessonPolicy()
        let s = Scope()
        _ = policy.settle(mixedRejection(s, 1))
        _ = policy.settle(pointRepair(s, 2, hostValue: true))
        _ = policy.settle(freshRead(s, 5))
        for record in policy.lessons {
            #expect(record.isValid)
            #expect(record.renderedText.rangeOfCharacter(from: .decimalDigits) == nil)
            #expect(record.tokenUpperBound <= 80)
        }
    }
}

// MARK: - Store

@Suite struct AgentLessonStoreTests {
    @Test func missingFileMeansNoMemory() async throws {
        try await withTempURL { url in
            #expect(await AgentLessonStore(url: url).load(now: now).isEmpty)
        }
    }

    @Test func mergeThenLoadRoundTripsAtAnInjectedPath() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            let written = [lesson(.separatePointTyping), active(.restoreEditableControls)]
            try await store.merge(written, now: now)
            let loaded = await store.load(now: now)
            #expect(loaded.count == 2)
            #expect(Set(loaded.map(\.family)) == [.separatePointTyping, .restoreEditableControls])
            #expect(loaded.first { $0.family == .restoreEditableControls }?.status == .active)
            #expect(FileManager.default.fileExists(atPath: url.path))
        }
    }

    @Test func writeIsAnAtomicReplaceLeavingOnlyTheStoreFile() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            for _ in 0..<5 { try await store.merge([lesson(.separatePointTyping)], now: now) }
            let names = try FileManager.default.contentsOfDirectory(atPath: url.deletingLastPathComponent().path)
            #expect(names == ["agent-lessons.json"])
            #expect(await store.load(now: now).first?.count == 5)
        }
    }

    @Test func twoStoreInstancesAtOnePathAccumulateSupport() async throws {
        try await withTempURL { url in
            try await AgentLessonStore(url: url).merge([lesson(.separatePointTyping)], now: now)
            try await AgentLessonStore(url: url).merge([lesson(.separatePointTyping)], now: now)
            let loaded = await AgentLessonStore(url: url).load(now: now)
            #expect(loaded.count == 1)
            #expect(loaded[0].count == 2)
            #expect(loaded[0].status == .active)
        }
    }

    @Test func concurrentWritersDoNotLoseSupport() async throws {
        try await withTempURL { url in
            await withTaskGroup(of: Void.self) { group in
                for _ in 0..<8 {
                    group.addTask {
                        try? await AgentLessonStore(url: url).merge([lesson(.currentChoiceTyping)], now: now)
                    }
                }
            }
            let loaded = await AgentLessonStore(url: url).load(now: now)
            #expect(loaded.count == 1)
            #expect(loaded[0].count == 8)
        }
    }

    @Test func mergingTheSameEpisodeTwiceCountsOnce() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            let record = lesson(.separatePointTyping)
            try await store.merge([record], now: now)
            try await store.merge([record], now: now)
            #expect(await store.load(now: now)[0].count == 1)
        }
    }

    @Test func corruptFileMeansNoMemoryAndIsNotRepairedByARead() async throws {
        try await withTempURL { url in
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try Data("{not json".utf8).write(to: url)
            let store = AgentLessonStore(url: url)
            #expect(await store.load(now: now).isEmpty)
            #expect(try Data(contentsOf: url) == Data("{not json".utf8))
            try await store.merge([lesson(.separatePointTyping)], now: now)
            #expect(await store.load(now: now).count == 1)
        }
    }

    @Test func wrongSchemaVersionIsIgnored() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            try await store.merge([lesson(.separatePointTyping)], now: now)
            let json = try #require(String(data: Data(contentsOf: url), encoding: .utf8))
            #expect(json.contains("\"schemaVersion\" : 1"))
            try Data(json.replacingOccurrences(of: "\"schemaVersion\" : 1", with: "\"schemaVersion\" : 2").utf8)
                .write(to: url)
            #expect(await store.load(now: now).isEmpty)
        }
    }

    @Test func oversizedFileIsIgnored() async throws {
        try await withTempURL { url in
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try Data(repeating: 0x20, count: 9 * 1_024 * 1_024).write(to: url)
            #expect(await AgentLessonStore(url: url).load(now: now).isEmpty)
        }
    }

    @Test func editedProseInTheFileIsRejected() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            try await store.merge([lesson(.separatePointTyping)], now: now)
            let json = try #require(String(data: Data(contentsOf: url), encoding: .utf8))
            let edited = json.replacingOccurrences(of: "Use permitted point typing.",
                                                   with: "Press the delete button.")
            #expect(edited != json)
            try Data(edited.utf8).write(to: url)
            #expect(await store.load(now: now).isEmpty)
        }
    }

    @Test func duplicateRecordKeysInTheFileAreRejected() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            try await store.merge([lesson(.separatePointTyping)], now: now)
            let object = try #require(try JSONSerialization.jsonObject(
                with: Data(contentsOf: url)) as? [String: Any])
            var duplicated = object
            let records = try #require(object["lessons"] as? [Any])
            duplicated["lessons"] = records + records
            try JSONSerialization.data(withJSONObject: duplicated).write(to: url)
            #expect(await store.load(now: now).isEmpty)
        }
    }

    @Test func aDisabledStoreReadsAndWritesNothing() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url, enabled: false)
            try await store.merge([lesson(.separatePointTyping)], now: now)
            #expect(await store.load(now: now).isEmpty)
            #expect(!FileManager.default.fileExists(atPath: url.path))
            try await AgentLessonStore(url: url).merge([lesson(.separatePointTyping)], now: now)
            #expect(await store.load(now: now).isEmpty)
        }
    }

    @Test func resetEmptiesTheStore() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            try await store.merge([lesson(.separatePointTyping)], now: now)
            try await store.reset()
            #expect(await store.load(now: now).isEmpty)
        }
    }

    @Test func invalidRecordsAreRefusedOnWrite() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            let wrong = AgentLessonEvidence(
                failure: ref(UUID(), 1), repair: ref(UUID(), 2), observation: ref(UUID(), 3),
                kind: .hostProof, predicate: .fieldValueVerified, verdictConflict: false)
            let invalid = AgentLesson(family: .separatePointTyping, evidence: wrong, lastSeen: now)
            await #expect(throws: AgentLessonStore.StoreError.self) {
                try await store.merge([invalid], now: now)
            }
            #expect(!FileManager.default.fileExists(atPath: url.path))
        }
    }

    @Test func candidateExpiresAtExactlySevenDays() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            let seen = now
            try await store.merge([lesson(.separatePointTyping, at: seen)], now: seen)
            #expect(await store.load(now: seen.addingTimeInterval(7 * day - 1)).count == 1)
            #expect(await store.load(now: seen.addingTimeInterval(7 * day)).isEmpty)
        }
    }

    @Test func activeLessonsDecayInsteadOfExpiring() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            try await store.merge([active(.separatePointTyping)], now: now)
            let later = now.addingTimeInterval(90 * day)
            let loaded = await store.load(now: later)
            #expect(loaded.count == 1)
            #expect(loaded[0].rankingWeight(at: later) == 0.25)
        }
    }

    @Test func loadingNeverRefreshesLastSeen() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            try await store.merge([active(.separatePointTyping)], now: now)
            let later = now.addingTimeInterval(40 * day)
            _ = await store.load(now: later)
            _ = TextLessonRetriever().retrieve(query(pointSituation, at: later),
                                               from: await store.load(now: later))
            #expect(await store.load(now: later)[0].lastSeen == now)
        }
    }

    @Test func compatibleVersionsPruneObsoleteRecords() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            let old = AgentLessonVersions(record: 1, tool: 1, proof: 1)
            let new = AgentLessonVersions(record: 1, tool: 2, proof: 1)
            try await store.merge([lesson(.separatePointTyping, versions: old),
                                   lesson(.separatePointTyping, versions: new)], now: now)
            #expect(await store.load(now: now).count == 2)
            #expect(await store.load(now: now, compatibleVersions: new).map(\.versions) == [new])
            try await store.merge([], now: now, compatibleVersions: new)
            #expect(await store.load(now: now).map(\.versions) == [new])
        }
    }

    @Test func capKeepsAtMost256RecordsAndEvictsTheLeastSupported() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            var records: [AgentLesson] = []
            for index in 0..<260 {
                let versions = AgentLessonVersions(record: 1, tool: index / 5 + 1, proof: 1)
                records.append(lesson(AgentLessonFamily.allCases[index % 5], versions: versions,
                                      at: now.addingTimeInterval(-Double(index) * 60)))
            }
            // The oldest record has the most support, so it must survive the cap.
            let oldest = records[259]
            var supported = oldest
            supported.merge(lesson(oldest.family, versions: oldest.versions, at: oldest.lastSeen))
            records[259] = supported
            try await store.merge(records, now: now)
            let loaded = await store.load(now: now)
            #expect(loaded.count == AgentLessonStore.maximumRecords)
            #expect(loaded.contains { $0.family == oldest.family && $0.versions == oldest.versions })
            #expect(loaded.contains { $0.versions.tool == 1 && $0.family == .separatePointTyping })
            #expect(!loaded.contains { $0.family == records[258].family
                && $0.versions == records[258].versions })
        }
    }

    @Test func quarantinedRecordsAreEvictedFirst() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            var records: [AgentLesson] = []
            for index in 0..<256 {
                records.append(lesson(AgentLessonFamily.allCases[index % 5],
                                      versions: .init(record: 1, tool: index / 5 + 1, proof: 1),
                                      at: now.addingTimeInterval(-3_600)))
            }
            var quarantined = active(.separatePointTyping, versions: .init(record: 1, tool: 999, proof: 1))
            quarantined.contradict(at: ref(UUID(), 1))
            records.append(quarantined)
            try await store.merge(records, now: now)
            let loaded = await store.load(now: now)
            #expect(loaded.count == 256)
            #expect(!loaded.contains { $0.status == .quarantined })
        }
    }

    @Test func prunedOrdersByStatusThenWeightAndIsDeterministic() {
        let healthyCandidate = lesson(.separatePointTyping)
        let strong = active(.currentChoiceTyping, extra: 2)
        var bad = active(.restoreEditableControls)
        bad.contradict(at: ref(UUID(), 1))
        let sorted = AgentLessonStore.pruned([bad, healthyCandidate, strong], now: now)
        #expect(sorted.map(\.family) == [.currentChoiceTyping, .separatePointTyping, .restoreEditableControls])
        #expect(AgentLessonStore.pruned([strong, healthyCandidate, bad], now: now).map(\.family)
            == sorted.map(\.family))
    }

    @Test func mergingCleanSupportIntoAStoredQuarantineKeepsItQuarantined() async throws {
        try await withTempURL { url in
            let store = AgentLessonStore(url: url)
            var bad = active(.separatePointTyping)
            bad.contradict(at: ref(UUID(), 1))
            try await store.merge([bad], now: now)
            try await store.merge([lesson(.separatePointTyping), lesson(.separatePointTyping)], now: now)
            let loaded = await store.load(now: now)
            #expect(loaded.count == 1)
            #expect(loaded[0].status == .quarantined)
        }
    }
}

// MARK: - Text retriever

@Suite struct AgentLessonTextRetrieverTests {
    private let retriever = TextLessonRetriever()

    @Test func anActiveMatchingLessonIsReturnedWithItsEvidenceKind() {
        let result = retriever.retrieve(query(pointSituation), from: [active(.separatePointTyping)])
        #expect(result.lessons.map(\.family) == [.separatePointTyping])
        #expect(result.block == "loop-rejection: Use permitted point typing. Avoid mixing target and point arguments.")
    }

    @Test func noLessonsOrNoMatchReturnsNothing() {
        #expect(retriever.retrieve(query(pointSituation), from: []).block.isEmpty)
        let none = retriever.retrieve(query([]), from: [active(.separatePointTyping)])
        #expect(none.lessons.isEmpty)
        #expect(none.block.isEmpty)
        #expect(none.tokenUpperBound == 0)
        #expect(retriever.retrieve(query([.nativeAlertPresent]), from: [active(.separatePointTyping)])
            .lessons.isEmpty)
    }

    @Test func aCandidateHelpsOnlyItsOwnEpisode() {
        let run = UUID()
        let candidate = lesson(.separatePointTyping, run: run, step: 18)
        #expect(retriever.retrieve(query(pointSituation), from: [candidate]).lessons.isEmpty)
        #expect(retriever.retrieve(query(pointSituation, episode: ref(UUID(), 18)),
                                   from: [candidate]).lessons.isEmpty)
        #expect(retriever.retrieve(query(pointSituation, episode: ref(run, 18)),
                                   from: [candidate]).lessons.count == 1)
    }

    @Test func anExpiredCandidateIsNotReturnedEvenForItsEpisode() {
        let run = UUID()
        let candidate = lesson(.separatePointTyping, run: run, step: 3, at: now)
        let later = now.addingTimeInterval(8 * day)
        #expect(retriever.retrieve(query(pointSituation, episode: ref(run, 3), at: later),
                                   from: [candidate]).lessons.isEmpty)
    }

    @Test func controlTypeIsAHardFilter() {
        #expect(retriever.retrieve(query(pointSituation, control: .systemAlert),
                                   from: [active(.separatePointTyping)]).lessons.isEmpty)
    }

    @Test func situationMustContainEveryLessonCondition() {
        let partial: Set<AgentLessonCondition> = [.callShapeRejected, .ocrFieldPresent]
        #expect(retriever.retrieve(query(partial, permitted: [.pointTypingPermitted]),
                                   from: [active(.separatePointTyping)]).lessons.isEmpty)
    }

    @Test func toolAndProofVersionsMustMatch() {
        let record = active(.separatePointTyping)
        #expect(retriever.retrieve(query(pointSituation, versions: .init(record: 1, tool: 2, proof: 1)),
                                   from: [record]).lessons.isEmpty)
        #expect(retriever.retrieve(query(pointSituation, versions: .init(record: 1, tool: 1, proof: 2)),
                                   from: [record]).lessons.isEmpty)
    }

    @Test func evidenceKindMustBeAllowed() {
        let record = active(.separatePointTyping)
        #expect(retriever.retrieve(query(pointSituation, kinds: [.hostProof, .observation]),
                                   from: [record]).lessons.isEmpty)
        #expect(retriever.retrieve(query(pointSituation, kinds: [.loopRejection]),
                                   from: [record]).lessons.count == 1)
    }

    @Test func aMissingPermissionBlocksTheLesson() {
        #expect(retriever.retrieve(query(pointSituation, permitted: []),
                                   from: [active(.separatePointTyping)]).lessons.isEmpty)
        #expect(retriever.retrieve(query(freshSituation, error: .missingField, permitted: []),
                                   from: [active(.restoreEditableControls)]).lessons.isEmpty)
    }

    @Test func uncertainDeliveryAbstains() {
        #expect(retriever.retrieve(query(pointSituation.union([.deliveryUnknown])),
                                   from: [active(.separatePointTyping)]).lessons.isEmpty)
    }

    @Test func quarantinedAndContradictedLessonsAreNeverReturned() {
        var bad = active(.separatePointTyping)
        bad.contradict(at: ref(UUID(), 1))
        #expect(retriever.retrieve(query(pointSituation), from: [bad]).lessons.isEmpty)
    }

    @Test func atMostTwoLessonsAreReturnedWithinTheTokenBlock() {
        let situation = pointSituation.union(choiceSituation)
        let pool = [active(.separatePointTyping), active(.verifiedPointTyping),
                    active(.currentChoiceTyping)]
        let result = retriever.retrieve(query(situation), from: pool)
        #expect(result.lessons.count == 2)
        #expect(result.tokenUpperBound <= 192)
        #expect(result.lessons.allSatisfy { $0.tokenUpperBound <= 80 })
        #expect(result.block.split(separator: "\n").count == 2)
    }

    @Test func everyPossiblePairStaysUnderTheBlockBudget() {
        for first in AgentLessonFamily.allCases {
            for second in AgentLessonFamily.allCases where second != first {
                let block = AgentLessonRetrieval.bounded([active(first), active(second)])
                #expect(block.tokenUpperBound <= 192)
                #expect(block.lessons.count <= 2)
            }
        }
        #expect(AgentLessonRetrieval.bounded(AgentLessonFamily.allCases.map { active($0) })
            .lessons.count == 2)
        #expect(AgentLessonRetrieval.bounded([]).block.isEmpty)
    }

    @Test func higherSupportRanksFirstAmongEqualMatches() {
        let weak = active(.separatePointTyping)
        let strong = active(.verifiedPointTyping, extra: 2)
        let result = retriever.retrieve(query(pointSituation), from: [weak, strong])
        #expect(result.lessons.map(\.family) == [.verifiedPointTyping, .separatePointTyping])
    }

    @Test func decayedSupportRanksBelowFreshSupport() {
        let old = active(.separatePointTyping, at: now.addingTimeInterval(-90 * day), extra: 2)
        let fresh = active(.verifiedPointTyping)
        let result = retriever.retrieve(query(pointSituation), from: [old, fresh])
        #expect(result.lessons.map(\.family) == [.verifiedPointTyping, .separatePointTyping])
    }

    @Test func ordinaryTiesBreakByFamilyNameDeterministically() {
        let pool = [active(.verifiedPointTyping), active(.separatePointTyping)]
        let a = retriever.retrieve(query(pointSituation), from: pool)
        let b = retriever.retrieve(query(pointSituation), from: pool.reversed())
        #expect(a.lessons.map(\.family) == [.separatePointTyping, .verifiedPointTyping])
        #expect(b.lessons.map(\.family) == a.lessons.map(\.family))
    }

    @Test func retrievalDoesNotChangeTheStoredRecords() {
        let record = active(.separatePointTyping)
        _ = retriever.retrieve(query(pointSituation), from: [record])
        #expect(record.lastSeen == now)
        #expect(record.count == 2)
    }

    @Test func queryTextIsBuiltFromTypedValuesOnly() {
        let built = query(pointSituation)
        #expect(built.sanitizedText == "editable field; call shape rejected; ocr field present; point typing permitted; callShape")
        #expect(built.embeddingQuery == "task: search result | query: \(built.sanitizedText)")
    }
}

// MARK: - Embedding retriever

@Suite struct AgentLessonEmbeddingRetrieverTests {
    private func makeEncoder() -> FakeEncoder {
        let encoder = FakeEncoder()
        encoder.documentVector = { FakeEncoder.perFamily($0) }
        return encoder
    }

    private func embedded(_ record: AgentLesson, with retriever: EmbeddingLessonRetriever) throws
        -> AgentLesson {
        try #require(try retriever.encoding(record))
    }

    @Test func documentsUseTheDocumentFormatAndStoreANormalizedVector() throws {
        let encoder = makeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = lesson(.separatePointTyping)
        let result = try embedded(record, with: retriever)
        #expect(encoder.inputs == [record.embeddingDocument])
        #expect(record.embeddingDocument ==
            "title: separate point typing | text: Use permitted point typing. Avoid mixing target and point arguments.")
        let vector = try #require(result.vector)
        #expect(vector.values.count == 256)
        #expect(vector.isValid)
        #expect(abs(vector.values[axis(.separatePointTyping)] - 1) < 1e-6)
        #expect(result.renderedText == record.renderedText)
    }

    @Test func badEncoderOutputIsNotStored() throws {
        let encoder = FakeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = lesson(.separatePointTyping)
        encoder.documentVector = { _ in [Float](repeating: 1, count: 255) }
        #expect(try retriever.encoding(record) == nil)
        encoder.documentVector = { _ in [Float](repeating: 0, count: 256) }
        #expect(try retriever.encoding(record) == nil)
        encoder.documentVector = { _ in var v = FakeEncoder.basis(0); v[1] = .nan; return v }
        #expect(try retriever.encoding(record) == nil)
    }

    @Test func anInvalidLessonIsNeverEncoded() throws {
        let encoder = makeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let wrong = AgentLessonEvidence(
            failure: ref(UUID(), 1), repair: ref(UUID(), 2), observation: ref(UUID(), 3),
            kind: .hostProof, predicate: .fieldValueVerified, verdictConflict: false)
        let invalid = AgentLesson(family: .separatePointTyping, evidence: wrong, lastSeen: now)
        #expect(try retriever.encoding(invalid) == nil)
        #expect(encoder.inputs.isEmpty)
    }

    @Test func theQueryUsesThePrefixAndCarriesNoRawInput() throws {
        let encoder = makeEncoder()
        encoder.queryVector = FakeEncoder.basis(axis(.separatePointTyping))
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = try embedded(active(.separatePointTyping), with: retriever)
        let before = encoder.inputs.count
        _ = try retriever.retrieve(query(pointSituation), from: [record])
        let sent = Array(encoder.inputs.dropFirst(before))
        #expect(sent.count == 1)
        #expect(sent[0].hasPrefix("task: search result | query: "))
        #expect(sent[0] == query(pointSituation).embeddingQuery)
        for text in encoder.inputs {
            #expect(text.rangeOfCharacter(from: .decimalDigits) == nil)
            #expect(text.hasPrefix("title: ") || text.hasPrefix("task: search result | query: "))
        }
    }

    @Test func aMatchingVectorIsReturnedAndAnOrthogonalOneIsNot() throws {
        let encoder = makeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = try embedded(active(.separatePointTyping), with: retriever)
        encoder.queryVector = FakeEncoder.basis(axis(.separatePointTyping))
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.count == 1)
        encoder.queryVector = FakeEncoder.basis(axis(.currentChoiceTyping))
        let none = try retriever.retrieve(query(pointSituation), from: [record])
        #expect(none.lessons.isEmpty)
        #expect(none.block.isEmpty)
    }

    @Test func theCosineThresholdIsRespectedAtTheEdge() throws {
        let encoder = makeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = try embedded(active(.separatePointTyping), with: retriever)
        let axisIndex = axis(.separatePointTyping)
        encoder.queryVector = FakeEncoder.cosine(0.79, axis: axisIndex)
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.isEmpty)
        encoder.queryVector = FakeEncoder.cosine(0.81, axis: axisIndex)
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.count == 1)
    }

    @Test func aStricterThresholdCanExcludeAMatch() throws {
        let encoder = makeEncoder()
        var retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = try embedded(active(.separatePointTyping), with: retriever)
        encoder.queryVector = FakeEncoder.cosine(0.9, axis: axis(.separatePointTyping))
        retriever.minimumCosine = 0.95
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.isEmpty)
        retriever.minimumCosine = .nan
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.isEmpty)
        retriever.minimumCosine = 1.5
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.isEmpty)
    }

    @Test func hardFiltersApplyBeforeAnySimilarity() throws {
        let encoder = makeEncoder()
        encoder.queryVector = FakeEncoder.basis(axis(.separatePointTyping))
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let activeRecord = try embedded(active(.separatePointTyping), with: retriever)
        let run = UUID()
        let candidate = try embedded(lesson(.separatePointTyping, run: run, step: 4), with: retriever)

        #expect(try retriever.retrieve(query(pointSituation, control: .systemAlert),
                                       from: [activeRecord]).lessons.isEmpty)
        #expect(try retriever.retrieve(query(pointSituation, versions: .init(record: 1, tool: 2, proof: 1)),
                                       from: [activeRecord]).lessons.isEmpty)
        #expect(try retriever.retrieve(query(pointSituation, kinds: [.observation]),
                                       from: [activeRecord]).lessons.isEmpty)
        #expect(try retriever.retrieve(query(pointSituation, permitted: []),
                                       from: [activeRecord]).lessons.isEmpty)
        #expect(try retriever.retrieve(query(pointSituation.union([.deliveryUnknown])),
                                       from: [activeRecord]).lessons.isEmpty)
        #expect(try retriever.retrieve(query(pointSituation), from: [candidate]).lessons.isEmpty)
        #expect(try retriever.retrieve(query(pointSituation, episode: ref(run, 4)),
                                       from: [candidate]).lessons.count == 1)
        var bad = activeRecord
        bad.contradict(at: ref(UUID(), 1))
        #expect(try retriever.retrieve(query(pointSituation), from: [bad]).lessons.isEmpty)
    }

    @Test func noEncodingHappensWhenNoCompatibleVectorExists() throws {
        let encoder = makeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        // No stored vector.
        _ = try retriever.retrieve(query(pointSituation), from: [active(.separatePointTyping)])
        // Filtered out before ranking.
        let record = try embedded(active(.separatePointTyping), with: retriever)
        let baseline = encoder.inputs.count
        _ = try retriever.retrieve(query(pointSituation, control: .systemAlert), from: [record])
        _ = try retriever.retrieve(query(pointSituation), from: [])
        #expect(encoder.inputs.count == baseline)
    }

    @Test func aRecordWithAnInvalidStoredVectorIsSkipped() throws {
        let encoder = makeEncoder()
        encoder.queryVector = FakeEncoder.basis(axis(.separatePointTyping))
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        var record = try embedded(active(.separatePointTyping), with: retriever)
        // A hand-built, unnormalized vector must fail validation and be ignored.
        let data = try JSONEncoder().encode(record.vector)
        var json = try #require(String(data: data, encoding: .utf8))
        json = json.replacingOccurrences(of: "\"EmbeddingGemma 2\"", with: "\"Other Model\"")
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(AgentLessonVector.self, from: Data(json.utf8))
        }
        record.vector = nil
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.isEmpty)
    }

    @Test func atMostTwoLessonsAndTheBlockBudgetHold() throws {
        let encoder = FakeEncoder()
        encoder.documentVector = { _ in FakeEncoder.basis(7) }
        encoder.queryVector = FakeEncoder.basis(7)
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let pool = try [AgentLessonFamily.separatePointTyping, .verifiedPointTyping, .currentChoiceTyping]
            .map { try embedded(active($0), with: retriever) }
        let result = try retriever.retrieve(query(pointSituation.union(choiceSituation)), from: pool)
        #expect(result.lessons.count == 2)
        #expect(result.tokenUpperBound <= 192)
    }

    @Test func higherCosineRanksFirstAndTiesAreDeterministic() throws {
        let encoder = FakeEncoder()
        // Both lessons share one direction; the query is closer to the first axis.
        encoder.documentVector = { text in
            text.hasPrefix("title: verified") ? FakeEncoder.cosine(0.9, axis: 7) : FakeEncoder.basis(7)
        }
        encoder.queryVector = FakeEncoder.basis(7)
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let separate = try embedded(active(.separatePointTyping), with: retriever)
        let verified = try embedded(active(.verifiedPointTyping), with: retriever)
        let result = try retriever.retrieve(query(pointSituation), from: [verified, separate])
        #expect(result.lessons.map(\.family) == [.separatePointTyping, .verifiedPointTyping])
        encoder.documentVector = { _ in FakeEncoder.basis(7) }
        let tieA = try embedded(active(.separatePointTyping), with: retriever)
        let tieB = try embedded(active(.verifiedPointTyping), with: retriever)
        let tie = try retriever.retrieve(query(pointSituation), from: [tieB, tieA])
        #expect(tie.lessons.map(\.family) == [.separatePointTyping, .verifiedPointTyping])
    }

    @Test func anEncoderFailureIsReportedNotSwallowed() throws {
        let encoder = makeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = try embedded(active(.separatePointTyping), with: retriever)
        encoder.failure = FakeEncoder.Failure()
        #expect(throws: FakeEncoder.Failure.self) {
            try retriever.retrieve(query(pointSituation), from: [record])
        }
        #expect(throws: FakeEncoder.Failure.self) {
            try retriever.encoding(lesson(.separatePointTyping))
        }
    }

    @Test func aBadQueryVectorReturnsNothing() throws {
        let encoder = makeEncoder()
        let retriever = EmbeddingLessonRetriever(encoder: encoder)
        let record = try embedded(active(.separatePointTyping), with: retriever)
        encoder.queryVector = [Float](repeating: 1, count: 12)
        #expect(try retriever.retrieve(query(pointSituation), from: [record]).lessons.isEmpty)
    }

    @Test func vectorsSurviveStorageAndStillRetrieve() async throws {
        try await withTempURL { url in
            let encoder = makeEncoder()
            encoder.queryVector = FakeEncoder.basis(axis(.separatePointTyping))
            let retriever = EmbeddingLessonRetriever(encoder: encoder)
            let store = AgentLessonStore(url: url)
            try await store.merge([try embedded(active(.separatePointTyping), with: retriever)], now: now)
            let loaded = await store.load(now: now)
            #expect(loaded.first?.vector?.isValid == true)
            #expect(try retriever.retrieve(query(pointSituation), from: loaded).lessons.count == 1)
        }
    }
}
