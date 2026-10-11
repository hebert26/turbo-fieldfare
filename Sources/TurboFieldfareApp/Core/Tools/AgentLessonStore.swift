import Foundation

/// Serializes read/merge/atomic-replace across store instances in this process.
/// Phase two should create one owner; this store does not coordinate other processes.
actor AgentLessonStore {
    static let maximumRecords = 256
    static let defaultURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/TurboFieldfare/agent-lessons.json")
    private static let fileLock = NSLock()
    private struct File: Codable {
        let schemaVersion: Int
        var lessons: [AgentLesson]
    }
    enum StoreError: Error { case invalidSchema, invalidRecord, oversizedFile }
    private let url: URL
    private let enabled: Bool

    init(url: URL = AgentLessonStore.defaultURL, enabled: Bool = true) {
        self.url = url
        self.enabled = enabled
    }

    /// Missing/corrupt/incompatible memory means ordinary operation. The file is
    /// not repaired or rewritten by a read; failed writes still report errors.
    func load(now: Date = Date(), compatibleVersions: AgentLessonVersions? = nil) -> [AgentLesson] {
        guard enabled else { return [] }
        return Self.fileLock.withLock {
            guard let file = try? read() else { return [] }
            return Self.pruned(file.lessons, now: now, compatibleVersions: compatibleVersions)
        }
    }

    func merge(_ additions: [AgentLesson], now: Date = Date(),
               compatibleVersions: AgentLessonVersions? = nil) throws {
        guard enabled else { return }
        try Self.fileLock.withLock {
            guard additions.allSatisfy(\.isValid) else { throw StoreError.invalidRecord }
            // Corrupt memory is discarded only when the caller explicitly writes support.
            var records = Self.pruned((try? read().lessons) ?? [], now: now,
                                      compatibleVersions: compatibleVersions)
            for addition in additions {
                if let index = records.firstIndex(where: {
                    $0.family == addition.family && $0.versions == addition.versions
                }) {
                    records[index].merge(addition)
                } else { records.append(addition) }
            }
            records = Self.pruned(records, now: now, compatibleVersions: compatibleVersions)
            try write(records)
        }
    }

    func reset() throws {
        guard enabled else { return }
        try Self.fileLock.withLock { try write([]) }
    }

    private func read() throws -> File {
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size <= 8 * 1_024 * 1_024 else { throw StoreError.oversizedFile }
        let decoded = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        guard decoded.schemaVersion == 1 else { throw StoreError.invalidSchema }
        guard decoded.lessons.count <= Self.maximumRecords,
              decoded.lessons.allSatisfy(\.isValid) else { throw StoreError.invalidRecord }
        let keys = decoded.lessons.map { "\($0.family.rawValue)|\($0.versions.tool)|\($0.versions.proof)" }
        guard Set(keys).count == keys.count else { throw StoreError.invalidRecord }
        return decoded
    }

    private func write(_ records: [AgentLesson]) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(File(schemaVersion: 1, lessons: records))
        guard data.count <= 8 * 1_024 * 1_024 else { throw StoreError.oversizedFile }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    static func pruned(_ records: [AgentLesson], now: Date,
                       compatibleVersions: AgentLessonVersions? = nil) -> [AgentLesson] {
        records.filter { lesson in
            lesson.isValid
                && (compatibleVersions == nil || lesson.versions == compatibleVersions)
                && !(lesson.status == .candidate && now.timeIntervalSince(lesson.lastSeen) >= 7 * 86_400)
        }.sorted { left, right in
            if (left.status == .quarantined) != (right.status == .quarantined) {
                return left.status != .quarantined
            }
            let lhs = left.rankingWeight(at: now), rhs = right.rankingWeight(at: now)
            if lhs != rhs { return lhs > rhs }
            if left.lastSeen != right.lastSeen { return left.lastSeen > right.lastSeen }
            if left.family != right.family { return left.family.rawValue < right.family.rawValue }
            if left.versions.tool != right.versions.tool { return left.versions.tool < right.versions.tool }
            return left.versions.proof < right.versions.proof
        }.prefix(Self.maximumRecords).map { $0 }
    }
}
