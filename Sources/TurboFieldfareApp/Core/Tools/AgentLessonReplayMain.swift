#if AGENT_LESSON_REPLAY
import Foundation

/// Compile these six files with swiftc -D AGENT_LESSON_REPLAY. This entry point
/// is absent from the app target and imports no inference or host client code.
@main
struct AgentLessonReplayMain {
    static func main() async throws {
        let paths = Array(CommandLine.arguments.dropFirst())
        guard !paths.isEmpty else {
            print("usage: agent-lesson-replay trace.jsonl [trace.jsonl ...]")
            return
        }
        let report = try AgentLessonReplay.run(traces: paths.map { URL(fileURLWithPath: $0) })
        let clock = report.lessons.map(\.lastSeen).max() ?? Date(timeIntervalSince1970: 0)
        // Exercise atomic persistence at an explicit replay-only path.
        let storeURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("agent-lesson-replay-\(UUID().uuidString)/agent-lessons.json")
        let store = AgentLessonStore(url: storeURL)
        try await store.merge(report.lessons, now: clock)
        let stored = await store.load(now: clock)
        guard stored.count == report.lessons.count else {
            throw AgentLessonStore.StoreError.invalidRecord
        }
        try FileManager.default.removeItem(at: storeURL.deletingLastPathComponent())
        for write in report.writes {
            print("WRITE \(write.trace) step \(write.step) | \(write.lesson.family.rawValue) | "
                + "\(write.lesson.family.kind.rawValue) | \(write.lesson.status.rawValue) | "
                + "count \(write.lesson.count) | conflict \(write.lesson.evidence.last?.verdictConflict ?? false) | "
                + write.lesson.renderedText)
        }
        for retrieval in report.retrievals {
            let block = retrieval.result.block.replacingOccurrences(of: "\n", with: " | ")
            print("RETRIEVE \(retrieval.trace) step \(retrieval.step) | \(block)")
        }
        for lesson in report.lessons {
            print("FINAL \(lesson.family.rawValue) | \(lesson.family.kind.rawValue) | \(lesson.status.rawValue) | "
                + "count \(lesson.count) | token ceiling \(lesson.tokenUpperBound) | \(lesson.renderedText)")
        }
        print("SUMMARY traces \(paths.count) settled \(report.settledSteps) unpaired \(report.unpairedCalls) "
            + "writes \(report.writes.count) families \(report.lessons.count) "
            + "retrieval steps \(report.retrievals.count)")
    }
}
#endif
