import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

private let bridgeNow = Date(timeIntervalSince1970: 1_800_000_000)
private let bridgeTool = VisionCaptureToolDefinitions.navigateName

private func bridgeDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("agent-lesson-bridge-\(UUID())")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func bridgeEnvironment(_ flag: String?, at url: URL) -> [String: String] {
    var result = ["TURBOFIELDFARE_AGENT_LESSONS_PATH": url.path]
    result["TURBOFIELDFARE_AGENT_LESSONS"] = flag
    return result
}

private func bridgePacket(id: String = "c1", verdict: String = "observed", value: String = "",
                          missing: Bool = false, unknown: Bool = false) throws -> String {
    var last: [String: Any] = ["verdict": verdict]
    if unknown { last["delivery"] = "unknown" }
    let body: [String: Any] = [
        "schema_version": 1,
        "observation": ["state": "current", "read_result": "succeeded"],
        "allowed_next": missing ? ["screenshot"] : ["observe", "type"],
        "choices": missing ? [] : [["id": id, "role": "text_field", "enabled": true,
            "operations": ["type"], "label": "Private field", "value_status": "available", "value": value]],
        "facts": [], "last_action": last, "guidance": "Keep current instructions.",
    ]
    return String(decoding: try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]), as: UTF8.self)
}

private func bridgeResult(_ id: String, content: String) -> AppToolResult {
    AppToolResult(callID: id, name: bridgeTool, content: content)
}

private func bridgeCall(_ id: String, action: String, target: String? = nil,
                        mixed: Bool = false, text: String = "private input") -> AppToolCall {
    var args: [String: JSONValue] = ["action": .string(action)]
    if action == "type" { args["text"] = .string(text) }
    if let target { args["target"] = .string(target) }
    if mixed { args["x_norm"] = .integer(20); args["y_norm"] = .integer(30) }
    return AppToolCall(id: id, name: bridgeTool, arguments: .object(args))
}

private func bridgeLesson(_ family: AgentLessonFamily, active: Bool = true) -> AgentLesson {
    func candidate() -> AgentLesson {
        let run = UUID()
        return AgentLesson(family: family, evidence: .init(
            failure: .init(run: run, step: 1), repair: .init(run: run, step: 2),
            observation: .init(run: run, step: 2), kind: family.kind, predicate: family.predicate,
            verdictConflict: false), lastSeen: bridgeNow)
    }
    var result = candidate()
    if active { result.merge(candidate()) }
    return result
}

private func bridgeRows(_ url: URL) throws -> [[String: Any]] {
    try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map {
        try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
    }
}

private func bridgeLearn(_ bridge: AgentLessonLoopBridge, run: UUID, instruction: String = "task",
                         repairText: String = "private input", unknown: Bool = false) async throws {
    let observe = bridgeCall("read", action: "observe")
    await bridge.settle(call: observe, result: bridgeResult(observe.id, content: try bridgePacket()),
        outcome: "{\"operation\":\"observe\",\"outcome\":\"succeeded\",\"observation_outcome\":\"succeeded\"}",
        run: run, instruction: instruction, now: bridgeNow)
    let rejection = bridgeCall("rejected", action: "type", target: "c1", mixed: true)
    await bridge.settle(call: rejection,
        result: bridgeResult(rejection.id, content: try bridgePacket(verdict: "not_sent")),
        outcome: "{\"outcome\":\"not_sent\"}", run: run, instruction: instruction, now: bridgeNow)
    let repair = bridgeCall("repair", action: "type", target: "c1", text: repairText)
    await bridge.settle(call: repair,
        result: bridgeResult(repair.id, content: try bridgePacket(verdict: "verified", value: repairText, unknown: unknown)),
        outcome: "{\"operation\":\"type\",\"outcome\":\"succeeded\",\"dispatch_attempted\":true}",
        run: run, instruction: instruction, now: bridgeNow)
}

@Suite("AgentLesson loop bridge")
struct AgentLessonLoopBridgeTests {
    @Test(arguments: [nil, "", "off", "TEXT", "1", " text "] as [String?])
    func disabledIsInert(flag: String?) async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        let traceURL = directory.appendingPathComponent("trace.jsonl")
        let trace = try #require(AgentInferenceTrace.forTesting(path: traceURL.path))
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment(flag, at: url), trace: trace)
        try await bridgeLearn(bridge, run: UUID())
        let original = bridgeResult("packet", content: try bridgePacket(missing: true))
        #expect(await bridge.advising(original, now: bridgeNow) == original)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        #expect(try Data(contentsOf: traceURL).isEmpty)
    }

    @Test func disabledLeavesExistingStoreAndPacketUntouched() async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        let bytes = Data("unreadable lesson memory".utf8)
        try bytes.write(to: url)
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment(nil, at: url))
        let original = bridgeResult("packet", content: "{\"lessons\":\"existing bytes\"}")
        try await bridgeLearn(bridge, run: UUID())
        #expect(await bridge.advising(original, now: bridgeNow) == original)
        #expect(try Data(contentsOf: url) == bytes)
    }

    @Test(arguments: ["text", "embedding"])
    func adviceIsBoundedReplacedAndGeneric(flag: String) async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        let traceURL = directory.appendingPathComponent("trace.jsonl")
        let trace = try #require(AgentInferenceTrace.forTesting(path: traceURL.path))
        let store = AgentLessonStore(url: url)
        try await store.merge([bridgeLesson(.restoreEditableControls), bridgeLesson(.freshFieldTyping)], now: bridgeNow)
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment(flag, at: url), trace: trace)
        let original = bridgeResult("packet", content: try bridgePacket(missing: true))
        let advised = await bridge.advising(original, now: bridgeNow)
        let repeated = await bridge.advising(advised, now: bridgeNow)
        #expect(repeated == advised)
        var body = try #require(JSONSerialization.jsonObject(with: Data(advised.content.utf8)) as? [String: Any])
        let block = try #require(body.removeValue(forKey: "lessons") as? String)
        #expect(block.components(separatedBy: AgentLessonLoopBridge.heading).count == 2)
        #expect(block.split(separator: "\n").count == 3)
        #expect(block.utf8.count <= 192)
        #expect(!block.contains("Private field") && !block.contains("c1") && !block.contains("private input"))
        let unchanged = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        #expect(String(decoding: unchanged, as: UTF8.self) == original.content)
        let next = bridgeResult("next", content: try bridgePacket())
        #expect(await bridge.advising(next, now: bridgeNow) == next)
        var stale = try #require(JSONSerialization.jsonObject(with: Data(next.content.utf8)) as? [String: Any])
        stale["lessons"] = block
        let staleResult = bridgeResult("next", content: String(decoding:
            try JSONSerialization.data(withJSONObject: stale, options: [.sortedKeys]), as: UTF8.self))
        #expect(await bridge.advising(staleResult, now: bridgeNow) == next)
        let rows = try bridgeRows(traceURL)
        let missing = rows.filter { $0["event"] as? String == "lessons_encoder_missing" }
        #expect(missing.count == (flag == "embedding" ? 1 : 0))
        let reads = rows.filter { $0["event"] as? String == "lessons_retrieved" }
        #expect(reads.count == 4)
        #expect(reads.first?["token_count"] as? Int == block.utf8.count)
        #expect((reads.first?["families"] as? [String])?.count == 2)
    }

    @Test func embeddingFallsBackExactlyToText() async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        try await AgentLessonStore(url: url).merge([bridgeLesson(.restoreEditableControls)], now: bridgeNow)
        let text = AgentLessonLoopBridge(environment: bridgeEnvironment("text", at: url))
        let embedding = AgentLessonLoopBridge(environment: bridgeEnvironment("embedding", at: url))
        let input = bridgeResult("packet", content: try bridgePacket(missing: true))
        #expect(await text.advising(input, now: bridgeNow) == embedding.advising(input, now: bridgeNow))
    }

    @Test func settledSequenceStoresCandidateAndDistinctRunsPromote() async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        let traceURL = directory.appendingPathComponent("trace.jsonl")
        let trace = try #require(AgentInferenceTrace.forTesting(path: traceURL.path))
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment("text", at: url), trace: trace)
        let run = UUID()
        try await bridgeLearn(bridge, run: run)
        let store = AgentLessonStore(url: url)
        let candidate = try #require(await store.load(now: bridgeNow).first)
        #expect(candidate.family == .currentChoiceTyping && candidate.status == .candidate && candidate.count == 1)
        #expect(candidate.evidence.first?.failure.run == run)
        #expect(candidate.evidence.first?.verdictConflict == true)
        let raw = try String(contentsOf: url, encoding: .utf8)
        #expect(!raw.contains("private input") && !raw.contains("Private field") && !raw.contains("c1"))
        try await bridgeLearn(bridge, run: run)
        #expect(await store.load(now: bridgeNow).first?.count == 1)
        try await bridgeLearn(bridge, run: UUID())
        let active = try #require(await store.load(now: bridgeNow).first)
        #expect(active.count == 2 && active.status == .active)
        let writes = try bridgeRows(traceURL).filter { $0["event"] as? String == "lessons_written" }
        #expect(writes.count == 2)
        #expect(writes.first?["status"] as? String == "candidate")
        #expect(writes.last?["status"] as? String == "active")
        #expect(writes.last?["count"] as? Int == 2)
    }

    @Test(arguments: ["changed text", "unknown delivery"])
    func uncertainOrChangedRepairsDoNotLearn(reason: String) async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment("text", at: url))
        try await bridgeLearn(bridge, run: UUID(),
            repairText: reason == "changed text" ? "different input" : "private input",
            unknown: reason == "unknown delivery")
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func candidateFromAnotherEpisodeIsNotRetrieved() async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        try await AgentLessonStore(url: url).merge([bridgeLesson(.restoreEditableControls, active: false)], now: bridgeNow)
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment("text", at: url))
        let input = bridgeResult("packet", content: try bridgePacket(missing: true))
        #expect(await bridge.advising(input, now: bridgeNow) == input)
    }

    @Test func unknownDeliverySuppressesActiveAdvice() async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        try await AgentLessonStore(url: url).merge([bridgeLesson(.restoreEditableControls)], now: bridgeNow)
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment("text", at: url))
        let input = bridgeResult("packet", content: try bridgePacket(missing: true, unknown: true))
        #expect(await bridge.advising(input, now: bridgeNow) == input)
    }

    @Test func unavailableStoreDoesNotChangeSettledResult() async throws {
        let directory = try bridgeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let parent = directory.appendingPathComponent("not-a-directory")
        try Data().write(to: parent)
        let bridge = AgentLessonLoopBridge(environment: bridgeEnvironment("text", at: parent.appendingPathComponent("memory.json")))
        try await bridgeLearn(bridge, run: UUID())
        let input = bridgeResult("packet", content: try bridgePacket())
        #expect(await bridge.advising(input, now: bridgeNow) == input)
    }
}
