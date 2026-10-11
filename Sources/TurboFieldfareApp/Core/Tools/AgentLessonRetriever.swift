import Foundation

struct AgentLessonQuery: Sendable {
    let controlType: AgentLessonControl
    let situation: Set<AgentLessonCondition>
    let error: AgentLessonError?
    let versions: AgentLessonVersions
    let permitted: Set<AgentLessonCondition>
    let evidenceKinds: Set<AgentLessonEvidenceKind>
    /// Candidates can help only the episode that produced their support.
    let currentEpisode: AgentLessonReference?
    let now: Date

    var sanitizedText: String {
        ([controlType.rawValue] + situation.map(\.rawValue).sorted() + [error?.rawValue ?? ""])
            .filter { !$0.isEmpty }.joined(separator: "; ")
    }
    var embeddingQuery: String { "task: search result | query: \(sanitizedText)" }
}

struct AgentLessonRetrieval: Sendable {
    let lessons: [AgentLesson]
    let block: String
    var tokenUpperBound: Int { block.utf8.count }

    static func bounded(_ ranked: [AgentLesson]) -> Self {
        var selected: [AgentLesson] = []
        var block = ""
        for lesson in ranked where selected.count < 2 {
            let line = "\(lesson.family.kind.rawValue): \(lesson.renderedText)"
            let proposed = block.isEmpty ? line : block + "\n" + line
            guard lesson.tokenUpperBound <= 80, proposed.utf8.count <= 192 else { continue }
            selected.append(lesson)
            block = proposed
        }
        return .init(lessons: selected, block: block)
    }
}

protocol AgentLessonRetrieving: Sendable {
    func retrieve(_ query: AgentLessonQuery, from lessons: [AgentLesson]) throws -> AgentLessonRetrieval
}

/// Implementations receive only fixed template documents and typed queries.
/// No encoder implementation or model loading belongs to this phase.
protocol LessonTextEncoding: Sendable {
    /// Return 256 finite dimensions; the retriever normalizes the result.
    func encode(_ prefixedText: String) throws -> [Float]
}

struct TextLessonRetriever: AgentLessonRetrieving {
    func retrieve(_ query: AgentLessonQuery, from lessons: [AgentLesson]) -> AgentLessonRetrieval {
        let matches = Self.filtered(query, lessons: lessons).map { lesson in
            let exact = lesson.situation == query.situation && lesson.family.error == query.error
            let words = Self.words(lesson.controlType.rawValue + "; "
                + lesson.situation.map(\.rawValue).sorted().joined(separator: "; ")
                + "; " + lesson.family.error.rawValue)
            let queryWords = Self.words(query.sanitizedText)
            let overlap = Double(words.intersection(queryWords).count) / Double(max(1, words.union(queryWords).count))
            return (lesson, exact, overlap * lesson.rankingWeight(at: query.now))
        }.sorted { lhs, rhs in
            if lhs.1 != rhs.1 { return lhs.1 }
            if lhs.2 != rhs.2 { return lhs.2 > rhs.2 }
            return lhs.0.family.rawValue < rhs.0.family.rawValue
        }
        return .bounded(matches.filter { $0.2 > 0 }.map { $0.0 })
    }

    static func filtered(_ query: AgentLessonQuery, lessons: [AgentLesson]) -> [AgentLesson] {
        guard !query.situation.contains(.deliveryUnknown) else { return [] }
        return AgentLessonStore.pruned(lessons, now: query.now).filter { lesson in
            guard lesson.controlType == query.controlType, lesson.versions == query.versions,
                  lesson.situation.isSubset(of: query.situation),
                  query.evidenceKinds.contains(lesson.family.kind), lesson.contradictionCount == 0,
                  lesson.status != .quarantined else { return false }
            let permissions = lesson.situation.intersection(
                [.freshReadPermitted, .pointTypingPermitted, .typingPermitted])
            guard permissions.isSubset(of: query.permitted) else { return false }
            if lesson.status == .candidate {
                guard let episode = query.currentEpisode,
                      lesson.evidence.contains(where: { $0.failure == episode }) else { return false }
            }
            return true
        }
    }

    private static func words(_ text: String) -> Set<String> {
        Set(text.lowercased().split(whereSeparator: { !$0.isLetter }).map(String.init))
    }
}

struct EmbeddingLessonRetriever: AgentLessonRetrieving {
    let encoder: any LessonTextEncoding
    var minimumCosine: Double = 0.80

    /// Encodes only the canonical document; callers persist the returned record.
    func encoding(_ lesson: AgentLesson) throws -> AgentLesson? {
        guard lesson.isValid,
              let vector = AgentLessonVector(normalizing: try encoder.encode(lesson.embeddingDocument)) else {
            return nil
        }
        var result = lesson
        result.vector = vector
        return result
    }

    func retrieve(_ query: AgentLessonQuery, from lessons: [AgentLesson]) throws -> AgentLessonRetrieval {
        let candidates = TextLessonRetriever.filtered(query, lessons: lessons)
            .filter { $0.vector?.isValid == true }
        guard !candidates.isEmpty, minimumCosine.isFinite, minimumCosine >= 0, minimumCosine <= 1,
              let queryVector = AgentLessonVector(normalizing: try encoder.encode(query.embeddingQuery)) else {
            return .bounded([])
        }
        let matches = candidates.compactMap { lesson -> (AgentLesson, Double)? in
            guard let vector = lesson.vector else { return nil }
            let cosine = zip(vector.values, queryVector.values).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
            return cosine >= minimumCosine ? (lesson, cosine) : nil
        }.sorted { lhs, rhs in
            if lhs.1 != rhs.1 { return lhs.1 > rhs.1 }
            let left = lhs.0.rankingWeight(at: query.now), right = rhs.0.rankingWeight(at: query.now)
            if left != right { return left > right }
            return lhs.0.family.rawValue < rhs.0.family.rawValue
        }
        return .bounded(matches.map { $0.0 })
    }
}
