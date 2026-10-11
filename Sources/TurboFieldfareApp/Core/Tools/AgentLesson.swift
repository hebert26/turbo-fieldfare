import Foundation

/// All prompt-bearing content is generated from this closed vocabulary.
/// Evidence references resolve into an external audit, never into prompt text.
enum AgentLessonControl: String, Codable, CaseIterable, Sendable {
    case editableField = "editable field"
    case systemAlert = "system alert"
}

enum AgentLessonCondition: String, Codable, CaseIterable, Sendable {
    case fieldAbsent = "field absent"
    case editableChoicePresent = "editable choice present"
    case ocrFieldPresent = "ocr field present"
    case nativeAlertPresent = "native alert present"
    case callShapeRejected = "call shape rejected"
    case freshReadPermitted = "fresh read permitted"
    case pointTypingPermitted = "point typing permitted"
    case typingPermitted = "typing permitted"
    case deliveryUnknown = "delivery unknown"
}

enum AgentLessonError: String, Codable, Sendable {
    case callShape, unavailableTarget, inconclusive, missingField
}

enum AgentLessonEvidenceKind: String, Codable, Sendable {
    case hostProof = "host-proof"
    case loopRejection = "loop-rejection"
    case observation
}

enum AgentLessonPredicate: String, Codable, Sendable {
    case callShapeAdmitted = "call shape admitted"
    case fieldValueVerified = "field value verified"
    case editableControlsRestored = "editable controls restored"
}

enum AgentLessonStatus: String, Codable, Sendable {
    case candidate, active, quarantined
}

enum AgentLessonFamily: String, Codable, CaseIterable, Sendable {
    case separatePointTyping = "separate point typing"
    case currentChoiceTyping = "current choice typing"
    case verifiedPointTyping = "verified point typing"
    case freshFieldTyping = "fresh field typing"
    case restoreEditableControls = "restore editable controls"

    var control: AgentLessonControl { .editableField }
    var kind: AgentLessonEvidenceKind {
        switch self {
        case .separatePointTyping, .currentChoiceTyping: .loopRejection
        case .verifiedPointTyping, .freshFieldTyping: .hostProof
        case .restoreEditableControls: .observation
        }
    }
    var predicate: AgentLessonPredicate {
        switch kind {
        case .loopRejection: .callShapeAdmitted
        case .hostProof: .fieldValueVerified
        case .observation: .editableControlsRestored
        }
    }
    var situation: Set<AgentLessonCondition> {
        switch self {
        case .separatePointTyping:
            [.callShapeRejected, .ocrFieldPresent, .pointTypingPermitted]
        case .currentChoiceTyping:
            [.callShapeRejected, .editableChoicePresent, .typingPermitted]
        case .verifiedPointTyping:
            [.callShapeRejected, .ocrFieldPresent, .pointTypingPermitted]
        case .freshFieldTyping, .restoreEditableControls:
            [.fieldAbsent, .freshReadPermitted]
        }
    }
    var error: AgentLessonError {
        switch self {
        case .separatePointTyping, .currentChoiceTyping, .verifiedPointTyping: .callShape
        case .freshFieldTyping, .restoreEditableControls: .missingField
        }
    }
    var instruction: String {
        switch self {
        case .separatePointTyping: "Use permitted point typing."
        case .currentChoiceTyping: "Use the current editable choice for typing."
        case .verifiedPointTyping: "Use permitted point typing; verify the field value."
        case .freshFieldTyping: "Type into a freshly offered field and verify its value."
        case .restoreEditableControls: "Read fresh controls before typing."
        }
    }
    var avoidance: String {
        switch self {
        case .separatePointTyping: "Avoid mixing target and point arguments."
        case .currentChoiceTyping: "Avoid labels and points."
        case .verifiedPointTyping: "Avoid prose targets."
        case .freshFieldTyping: "Avoid old targets."
        case .restoreEditableControls: "Avoid earlier field targets."
        }
    }
}

struct AgentLessonVersions: Codable, Hashable, Sendable {
    var record = 1
    var tool = 1
    var proof = 1
}

struct AgentLessonReference: Codable, Hashable, Sendable {
    let run: UUID
    let step: Int
}

struct AgentLessonEvidence: Codable, Hashable, Sendable {
    let failure: AgentLessonReference
    let repair: AgentLessonReference
    let observation: AgentLessonReference
    let kind: AgentLessonEvidenceKind
    let predicate: AgentLessonPredicate
    /// Preserves a loop/host disagreement without promoting it to host proof.
    let verdictConflict: Bool
}

struct AgentLessonVector: Codable, Equatable, Sendable {
    static let dimensions = 256
    static let encodingVersion = 1
    let model: Model
    let encodingVersion: Int
    let documentPrefix: String
    let queryPrefix: String
    let values: [Float]
    enum Model: String, Codable, Sendable { case embeddingGemma2 = "EmbeddingGemma 2" }

    init?(normalizing values: [Float]) {
        guard values.count == Self.dimensions, values.allSatisfy({ $0.isFinite }) else { return nil }
        let norm = sqrt(values.reduce(0.0) { $0 + Double($1) * Double($1) })
        guard norm > 0, norm.isFinite else { return nil }
        self.model = .embeddingGemma2
        self.encodingVersion = Self.encodingVersion
        self.documentPrefix = "title: … | text: …"
        self.queryPrefix = "task: search result | query: …"
        self.values = values.map { Float(Double($0) / norm) }
    }
    var isValid: Bool {
        guard model == .embeddingGemma2, encodingVersion == Self.encodingVersion,
              documentPrefix == "title: … | text: …", queryPrefix == "task: search result | query: …",
              values.count == Self.dimensions, values.allSatisfy({ $0.isFinite }) else { return false }
        return abs(values.reduce(0.0) { $0 + Double($1) * Double($1) } - 1) < 0.001
    }
}

struct AgentLesson: Codable, Equatable, Sendable {
    let family: AgentLessonFamily
    var versions: AgentLessonVersions
    private(set) var evidence: [AgentLessonEvidence]
    private(set) var lastSeen: Date
    private(set) var status: AgentLessonStatus
    private(set) var contradictions: Set<AgentLessonReference> = []
    var vector: AgentLessonVector?

    var controlType: AgentLessonControl { family.control }
    var situation: Set<AgentLessonCondition> { family.situation }
    var `do`: String { family.instruction }
    var avoid: String { family.avoidance }
    var renderedText: String { "\(self.do) \(avoid)" }
    var contradictionCount: Int { contradictions.count }
    /// Support counts distinct runs, regardless of repairs or scopes within a run.
    var count: Int { Set(evidence.map { $0.failure.run }).count }
    var tokenUpperBound: Int { renderedText.utf8.count }
    var embeddingDocument: String { "title: \(family.rawValue) | text: \(renderedText)" }

    init(family: AgentLessonFamily, versions: AgentLessonVersions = .init(),
         evidence: AgentLessonEvidence, lastSeen: Date) {
        self.family = family
        self.versions = versions
        self.evidence = [evidence]
        self.lastSeen = lastSeen
        self.status = .candidate
    }

    mutating func merge(_ other: AgentLesson) {
        guard family == other.family, versions == other.versions else { return }
        var seen = Set(evidence.map(\.failure))
        for item in other.evidence where seen.insert(item.failure).inserted { evidence.append(item) }
        contradictions.formUnion(other.contradictions)
        lastSeen = max(lastSeen, other.lastSeen)
        status = contradictions.isEmpty ? (count >= 2 ? .active : .candidate) : .quarantined
        // A vector cannot be supplied for arbitrary text; its input is the fixed document above.
        if vector == nil { vector = other.vector }
    }

    mutating func contradict(at reference: AgentLessonReference) {
        contradictions.insert(reference)
        status = .quarantined
    }

    func rankingWeight(at now: Date) -> Double {
        let periods = floor(max(0, now.timeIntervalSince(lastSeen)) / (30 * 86_400))
        return Double(count) * pow(0.5, periods)
    }

    var isValid: Bool {
        guard versions.record == 1, versions.tool > 0, versions.proof > 0,
              !evidence.isEmpty, tokenUpperBound <= 80, lastSeen.timeIntervalSince1970.isFinite,
              Set(evidence.map(\.failure)).count == evidence.count,
              vector == nil || vector?.isValid == true else { return false }
        guard evidence.allSatisfy({ item in
            item.kind == family.kind && item.predicate == family.predicate
                && item.failure.run == item.repair.run && item.repair.run == item.observation.run
                && item.failure.step >= 0 && item.failure.step <= item.repair.step
                && item.repair.step <= item.observation.step
                && (item.kind != .hostProof || !item.verdictConflict)
        }) else { return false }
        return contradictions.isEmpty
            ? status == (count >= 2 ? .active : .candidate)
            : status == .quarantined
    }
}

extension AgentLesson {
    private enum CodingKeys: String, CodingKey {
        case family, versions, evidence, status, vector, contradictions
        case controlType = "control_type"
        case situation, `do`, avoid, count
        case lastSeen = "last_seen"
        case contradictionCount = "contradiction_count"
    }

    /// Persist the complete record, validating redundant fields on load so an
    /// edited file cannot supply arbitrary lesson prose or retrieval keys.
    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        family = try container.decode(AgentLessonFamily.self, forKey: .family)
        versions = try container.decode(AgentLessonVersions.self, forKey: .versions)
        evidence = try container.decode([AgentLessonEvidence].self, forKey: .evidence)
        lastSeen = try container.decode(Date.self, forKey: .lastSeen)
        status = try container.decode(AgentLessonStatus.self, forKey: .status)
        contradictions = try container.decode(Set<AgentLessonReference>.self, forKey: .contradictions)
        vector = try container.decodeIfPresent(AgentLessonVector.self, forKey: .vector)
        guard try container.decode(AgentLessonControl.self, forKey: .controlType) == controlType,
              try container.decode(Set<AgentLessonCondition>.self, forKey: .situation) == situation,
              try container.decode(String.self, forKey: .do) == self.do,
              try container.decode(String.self, forKey: .avoid) == avoid,
              try container.decode(Int.self, forKey: .count) == count,
              try container.decode(Int.self, forKey: .contradictionCount) == contradictionCount,
              isValid else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath,
                                                    debugDescription: "Invalid lesson record"))
        }
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(family, forKey: .family)
        try container.encode(versions, forKey: .versions)
        try container.encode(evidence, forKey: .evidence)
        try container.encode(lastSeen, forKey: .lastSeen)
        try container.encode(status, forKey: .status)
        try container.encode(contradictions.sorted {
            if $0.run != $1.run { return $0.run.uuidString < $1.run.uuidString }
            return $0.step < $1.step
        }, forKey: .contradictions)
        try container.encodeIfPresent(vector, forKey: .vector)
        try container.encode(controlType, forKey: .controlType)
        try container.encode(situation.sorted { $0.rawValue < $1.rawValue }, forKey: .situation)
        try container.encode(self.do, forKey: .do)
        try container.encode(avoid, forKey: .avoid)
        try container.encode(count, forKey: .count)
        try container.encode(contradictionCount, forKey: .contradictionCount)
    }
}
