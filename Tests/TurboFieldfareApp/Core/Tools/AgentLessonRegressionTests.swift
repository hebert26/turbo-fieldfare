import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareAppCore

private let regressionNow = Date(timeIntervalSince1970: 1_800_000_000)
private let regressionTool = VisionCaptureToolDefinitions.navigateName

private func regressionDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("lesson-regression-\(UUID())")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func regressionLesson(run: UUID = UUID(), at time: Date = regressionNow) -> AgentLesson {
    AgentLesson(family: .currentChoiceTyping, evidence: .init(
        failure: .init(run: run, step: 1), repair: .init(run: run, step: 2),
        observation: .init(run: run, step: 2), kind: .loopRejection,
        predicate: .callShapeAdmitted, verdictConflict: false), lastSeen: time)
}

private func regressionStep(run: UUID, scope: UUID, index: Int,
                            action: AgentLessonStep.Action = .type,
                            shape: AgentLessonStep.Shape, outcome: AgentLessonStep.Outcome,
                            before: Set<AgentLessonCondition> = [.editableChoicePresent, .typingPermitted],
                            after: Set<AgentLessonCondition> = [], hostValue: Bool = false) -> AgentLessonStep {
    AgentLessonStep(reference: .init(run: run, step: index), observationReference: .init(run: run, step: index),
        subgoal: scope, controlState: scope, versions: .init(), time: regressionNow,
        action: action, shape: shape, outcome: outcome, before: before, after: after,
        observationIsFresh: true, targetBound: true, hostVerifiedValue: hostValue, verdictConflict: false)
}

private func regressionPacket(verdict: String = "observed", delivery: String? = nil,
                              current: Bool = true) -> [String: Any] {
    var last: [String: Any] = ["verdict": verdict]
    last["delivery"] = delivery
    return ["schema_version": 1, "observation": ["state": current ? "current" : "stale",
        "read_result": "succeeded"], "allowed_next": ["observe", "type"],
        "choices": [["id": "field", "label": "Field", "role": "text_field", "enabled": true,
            "operations": ["type"], "value_status": "available", "value": "input"]],
        "facts": [], "last_action": last, "guidance": "Current guidance"]
}

private func regressionJSON(_ body: [String: Any]) throws -> String {
    String(decoding: try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys]), as: UTF8.self)
}

private func regressionResult(_ id: String, _ packet: [String: Any]) throws -> AppToolResult {
    AppToolResult(callID: id, name: regressionTool, content: try regressionJSON(packet))
}

private func regressionLearn(_ bridge: AgentLessonLoopBridge, run: UUID) async throws {
    let read = AppToolCall(id: "read", name: regressionTool, arguments: .object(["action": .string("observe")]))
    await bridge.settle(call: read, result: try regressionResult(read.id, regressionPacket()),
        outcome: "{\"outcome\":\"succeeded\"}", run: run, now: regressionNow)
    let reject = AppToolCall(id: "reject", name: regressionTool, arguments: .object([
        "action": .string("type"), "target": .string("field"), "text": .string("input"),
        "x_norm": .integer(20), "y_norm": .integer(30)]))
    await bridge.settle(call: reject,
        result: try regressionResult(reject.id, regressionPacket(verdict: "not_sent")),
        outcome: "{\"outcome\":\"not_sent\"}", run: run, now: regressionNow)
    let repair = AppToolCall(id: "repair", name: regressionTool, arguments: .object([
        "action": .string("type"), "target": .string("field"), "text": .string("input")]))
    await bridge.settle(call: repair,
        result: try regressionResult(repair.id, regressionPacket(verdict: "verified")),
        outcome: "{\"operation\":\"type\",\"outcome\":\"succeeded\",\"dispatch_attempted\":true}",
        run: run, now: regressionNow)
}

private func regressionTrace(at url: URL, unknownAt: String? = nil) throws {
    let run = UUID().uuidString
    var rows: [[String: Any]] = []
    func append(_ event: String, _ step: Int, _ key: String, _ body: [String: Any]) {
        rows.append(["schema_version": 1, "conversation_id": run, "event": event,
            "step_index": step, "timestamp_unix_seconds": regressionNow.timeIntervalSince1970, key: body])
    }
    func input(_ step: Int, _ id: String, _ packet: [String: Any]) throws {
        append("input", step, "input", ["results": [["call_id": id, "content": try regressionJSON(packet)]]])
    }
    func output(_ step: Int, _ id: String, mixed: Bool = false) {
        var args: [String: Any] = ["action": "type", "target": "field", "text": "input"]
        if mixed { args["x_norm"] = 20; args["y_norm"] = 30 }
        append("output", step, "output", ["tool_calls": [["id": id, "arguments": args]]])
    }
    try input(1, "opening", regressionPacket(delivery: unknownAt == "before" ? "unknown" : nil))
    output(1, "rejection", mixed: true)
    try input(2, "rejection", regressionPacket(verdict: "not_sent",
        delivery: unknownAt == "rejection" ? "unknown" : nil))
    output(2, "repair")
    append("mcp_request", 2, "mcp_request", ["operation": "type", "is_error": false])
    try input(3, "repair", regressionPacket(verdict: "verified",
        delivery: unknownAt == "repair" ? "unknown" : nil))
    // A later known result must not complete a discarded rejection episode.
    output(3, "later")
    append("mcp_request", 3, "mcp_request", ["operation": "type", "is_error": false])
    try input(4, "later", regressionPacket(verdict: "verified"))
    try rows.map(regressionJSON).joined(separator: "\n").appending("\n").write(to: url, atomically: true,
        encoding: .utf8)
}

/// CPU storage implementing the buffer surface used by prompt rendering.
/// Any attempted GPU use fails immediately; the seam must never execute a model.
private final class RegressionFeatureBuffer: NSObject, MTLBuffer {
    let length: Int
    private let storage: UnsafeMutableRawPointer
    init(length: Int) {
        self.length = length
        storage = .allocate(byteCount: length, alignment: 16)
        super.init()
    }
    deinit { storage.deallocate() }
    func contents() -> UnsafeMutableRawPointer { storage }
    var device: any MTLDevice { fatalError("Regression buffer has no GPU device") }
    var label: String?
    var cpuCacheMode: MTLCPUCacheMode { .defaultCache }
    var storageMode: MTLStorageMode { .shared }
    var hazardTrackingMode: MTLHazardTrackingMode { .default }
    var resourceOptions: MTLResourceOptions { .storageModeShared }
    var heap: (any MTLHeap)? { nil }
    var heapOffset: Int { 0 }
    var allocatedSize: Int { length }
    func isAliasable() -> Bool { false }
    var remoteStorageBuffer: (any MTLBuffer)? { nil }
    var gpuAddress: UInt64 { 0 }
    var sparseBufferTier: MTLBufferSparseTier { .tierNone }
    func setPurgeableState(_ state: MTLPurgeableState) -> MTLPurgeableState { .nonVolatile }
    func makeAliasable() {}
    func __setOwnerWithIdentity(_ identity: task_id_token_t) -> kern_return_t { KERN_SUCCESS }
    func __didModifyRange(_ range: NSRange) {}
    func makeTexture(descriptor: MTLTextureDescriptor, offset: Int, bytesPerRow: Int) -> (any MTLTexture)? { nil }
    func makeTensor(descriptor: MTLTensorDescriptor, offset: Int) throws -> any MTLTensor {
        fatalError("Regression buffer cannot make tensors")
    }
    func __addDebugMarker(_ marker: String, range: NSRange) {}
    func removeAllDebugMarkers() {}
    func makeRemoteBufferView(_ device: any MTLDevice) -> (any MTLBuffer)? { nil }
}

/// No weights, kernels, image files or model execution. Production code still
/// tokenizes the prompt, carries feature buffers and parses fake output.
private final class RegressionCompletion: Sendable {
    struct Attempt: Sendable {
        let prompt: [Int32]
        let input: MultimodalPrefillInput?
        let cached: Int
    }
    private struct State: Sendable {
        var tokens: [Int32]
        var failure: String?
        var attempts: [Attempt] = []
    }
    private let state: Mutex<State>
    private let tokenizer: GFTokenizer

    init(tokenizer: GFTokenizer, tokens: [Int32]) {
        self.tokenizer = tokenizer
        state = Mutex(State(tokens: tokens))
    }

    func failNext(_ failure: String) { state.withLock { $0.failure = failure } }
    var attempts: [Attempt] { state.withLock { $0.attempts } }

    var hooks: GemmaMultimodalConversation.TestCompletion {
        .init(position: { self.state.withLock { $0.tokens.count } },
            rewind: { position in
                try self.state.withLock {
                    guard position <= $0.tokens.count else { throw MultimodalConversationError.lineageBroken }
                    $0.tokens = Array($0.tokens.prefix(position))
                }
            },
            reset: { self.state.withLock { $0.tokens = [] } },
            run: { prompt, input, cached, progress in
                let failure = try self.state.withLock { value in
                    guard value.tokens.count == cached,
                          Array(prompt.prefix(cached)) == value.tokens else {
                        throw MultimodalConversationError.lineageBroken
                    }
                    value.attempts.append(Attempt(prompt: prompt, input: input, cached: cached))
                    value.tokens = prompt
                    defer { value.failure = nil }
                    return value.failure
                }
                progress(.prefill(done: prompt.count, total: prompt.count))
                let output: [Int32]
                let reason: StopReason
                switch failure {
                case "malformed":
                    output = [self.tokenizer.toolCallStartID]
                    reason = .endOfTurn
                case "repetition":
                    output = self.tokenizer.encode("<|channel>thought\n", addBOS: false)
                        + Array(repeating: self.tokenizer.encode(" repeat", addBOS: false)[0], count: 160)
                    reason = .maxTokens
                default:
                    output = self.tokenizer.encode(
                        "<|tool_call>call:\(regressionTool){}<tool_call|>", addBOS: false)
                    reason = .toolCalls
                }
                for (index, id) in output.enumerated() {
                    self.state.withLock { $0.tokens.append(id) }
                    progress(.token(index: index, id: id,
                        delta: self.tokenizer.decode([id], skipSpecialTokens: false)))
                }
                let tokens = prompt + output
                return RawDecodeResult(prefillTokens: prompt.count, cachedPromptTokens: cached,
                    computedPrefillTokens: prompt.count - cached, prefillSeconds: 0,
                    newTokens: output.count, decodeSeconds: 0, reason: reason,
                    kvPosition: tokens.count, kvBackedTokenIDs: tokens, uncommittedBoundaryTokenIDs: [])
            })
    }
}

private let regressionTokenizerFolder = Bundle.module.resourceURL!
    .appendingPathComponent("lesson-tokenizer", isDirectory: true)

private enum RegressionTokenizerError: Error, CustomStringConvertible {
    case missingFixture(URL)

    var description: String {
        switch self {
        case .missingFixture(let file): "missing fixture: \(file.path)"
        }
    }
}

private func regressionTokenizer(from folder: URL = regressionTokenizerFolder) async throws -> GFTokenizer {
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
        let file = folder.appendingPathComponent(name)
        guard FileManager.default.isReadableFile(atPath: file.path) else {
            throw RegressionTokenizerError.missingFixture(file)
        }
    }
    // This entry point only reads local files; it never chooses the pretrained loader or an override.
    return try await GFTokenizer.load(from: folder).withToolThinking(enabled: false)
}

private func regressionPending(_ id: String) -> GFTokenizer.Message {
    GFTokenizer.Message(role: .assistant, content: nil, toolCalls: [
        .init(id: id, name: regressionTool, arguments: .object([:]))])
}

private let regressionDefinition = GFTokenizer.FunctionDefinition(
    name: regressionTool, description: "Inspect", parameters: .object(["type": .string("object")]))

private func regressionLessonBlocks(_ text: String) -> Int {
    text.components(separatedBy: "\"lessons\"").count - 1
}

@Suite("AgentLesson review regressions")
struct AgentLessonRegressionTests {
    @Test(arguments: ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])
    func missingTokenizerFixtureFailsLocally(missing: String) async throws {
        let directory = try regressionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] where name != missing {
            try FileManager.default.copyItem(at: regressionTokenizerFolder.appendingPathComponent(name),
                to: directory.appendingPathComponent(name))
        }
        do {
            _ = try await regressionTokenizer(from: directory)
            Issue.record("Expected missing fixture error")
        } catch let error as RegressionTokenizerError {
            #expect(error.description == "missing fixture: \(directory.appendingPathComponent(missing).path)")
        }
    }

    @Test func tokenizerFixturePreservesTextAndMarkers() async throws {
        let tokenizer = try await regressionTokenizer()
        let text = "{\"lessons\":[\"advice é 🐦\"]}\n<|tool_call>call:tool{}<tool_call|><|image|>"
        let tokens = tokenizer.encode(text, addBOS: false)
        #expect(tokenizer.decode(tokens, skipSpecialTokens: false) == text)
        #expect(tokens.contains(tokenizer.toolCallStartID) && tokens.contains(tokenizer.toolCallEndID))
        #expect(tokens.contains(MultimodalPromptRenderer.imageTokenID))
        #expect(tokenizer.encode(" repeat", addBOS: false).count == 1)
    }

    @Test(arguments: [
        "[text_field] note ) ocr-confirmed at (20, 30) ocr-confirmed",
        "[text_field] note ) ocr-confirmed at (20, 30)",
        "[text_field] note at (1, 2) at (20, 30) ocr-confirmed",
        "[text_field] note at (20, 30) ocr-confirmed ) ocr-confirmed",
        "[text_field] note at (nan, 30) ocr-confirmed",
        "[text_field] note at (, 20, 30) ocr-confirmed",
    ])
    func malformedOCRAbstains(fact: String) {
        let args: [String: Any] = ["x_norm": 20, "y_norm": 30, "text": "input"]
        let packet: [String: Any] = ["facts": [fact]]
        #expect(!AgentLessonReplay.pointMatchesField(args, packet: packet))
        #expect(!AgentLessonReplay.valueMatches(args, before: packet, after: packet, chosen: nil))
    }

    @Test func valueMarkerOrderingAndUniquenessAbstain() {
        let args: [String: Any] = ["x_norm": 20, "y_norm": 30, "text": "input"]
        let before: [String: Any] = ["facts": ["[text_field] empty at (20, 30) ocr-confirmed"]]
        for fact in [
            "[text_field] input (ocr): input at (20, 30) ocr-confirmed (ocr): input",
            "[text_field] input (ocr): input at (20, 30) ocr-confirmed at (20, 30)",
        ] {
            #expect(!AgentLessonReplay.valueMatches(args, before: before, after: ["facts": [fact]], chosen: nil))
        }
        #expect(AgentLessonReplay.valueMatches(args, before: before,
            after: ["facts": ["[text_field] input (ocr): input at (20, 30) ocr-confirmed"]], chosen: nil))
    }

    @Test func differentScopesWithinOneRunStayCandidateThroughPersistence() async throws {
        let directory = try regressionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AgentLessonStore(url: directory.appendingPathComponent("memory.json"))
        let run = UUID()
        var policy = AgentLessonPolicy()
        for (index, scope) in [UUID(), UUID()].enumerated() {
            #expect(policy.settle(regressionStep(run: run, scope: scope, index: 10 * index + 1,
                shape: .mixedTargetAndPoint, outcome: .rejected)).isEmpty)
            let writes = policy.settle(regressionStep(run: run, scope: scope, index: 10 * index + 2,
                shape: .currentChoice, outcome: .admitted))
            #expect(writes.count == (index == 0 ? 1 : 0))
            try await store.merge(writes, now: regressionNow)
        }
        let record = try #require(policy.lessons.first)
        #expect(record.evidence.count == 2)
        #expect(Set(record.evidence.map(\.failure.step)) == [1, 11])
        #expect(record.count == 1 && record.status == .candidate)
        // Persist the combined record too, protecting the store's distinct-run count.
        try await store.merge([record], now: regressionNow)
        let stored = try #require(await store.load(now: regressionNow).first)
        #expect(stored.evidence.count == 2 && stored.count == 1 && stored.status == .candidate)
    }

    @Test(arguments: [Set<AgentLessonCondition>([.editableChoicePresent]), [.typingPermitted], []])
    func freshFieldAdmissionRequiresChoiceAndPermission(before: Set<AgentLessonCondition>) {
        var policy = AgentLessonPolicy()
        let run = UUID(), scope = UUID()
        #expect(policy.settle(regressionStep(run: run, scope: scope, index: 1, action: .refresh,
            shape: .other, outcome: .observed, before: [.fieldAbsent, .freshReadPermitted],
            after: [.editableChoicePresent, .typingPermitted])).map(\.family) == [.restoreEditableControls])
        #expect(policy.settle(regressionStep(run: run, scope: scope, index: 2,
            shape: .currentChoice, outcome: .admitted, before: before, hostValue: true)).isEmpty)
        #expect(!policy.lessons.contains { $0.family == .freshFieldTyping })
        #expect(policy.settle(regressionStep(run: run, scope: scope, index: 3,
            shape: .currentChoice, outcome: .admitted, hostValue: true)).map(\.family) == [.freshFieldTyping])
    }

    @Test func disabledResetPreservesPopulatedFile() async throws {
        let directory = try regressionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        try await AgentLessonStore(url: url).merge([regressionLesson()], now: regressionNow)
        let original = try Data(contentsOf: url)
        try await AgentLessonStore(url: url, enabled: false).reset()
        #expect(try Data(contentsOf: url) == original)
        #expect(await AgentLessonStore(url: url).load(now: regressionNow).count == 1)
    }

    @Test func dayEightSupportDoesNotResurrectExpiredCandidate() async throws {
        let directory = try regressionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        let store = AgentLessonStore(url: url)
        let expired = regressionLesson()
        try await store.merge([expired], now: regressionNow)
        let later = regressionNow.addingTimeInterval(8 * 86_400)
        let fresh = regressionLesson(at: later)
        try await store.merge([fresh], now: later)
        let record = try #require(await store.load(now: later).first)
        #expect(record.count == 1 && record.status == .candidate)
        #expect(record.evidence == fresh.evidence)
        #expect(record.lastSeen == later)
    }

    @Test(arguments: ["before", "rejection", "repair"])
    func replayUnknownDeliveryDiscardsSyntaxPairingAndAdvice(unknownAt: String) throws {
        let directory = try regressionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let clean = directory.appendingPathComponent("clean.jsonl")
        let clean2 = directory.appendingPathComponent("clean2.jsonl")
        let clean3 = directory.appendingPathComponent("clean3.jsonl")
        let uncertain = directory.appendingPathComponent("uncertain.jsonl")
        try regressionTrace(at: clean)
        try regressionTrace(at: clean2)
        try regressionTrace(at: clean3)
        try regressionTrace(at: uncertain, unknownAt: unknownAt)
        let alone = try AgentLessonReplay.run(traces: [uncertain])
        #expect(alone.settledSteps == 3)
        #expect(alone.writes.isEmpty && alone.lessons.isEmpty && alone.retrievals.isEmpty)
        let report = try AgentLessonReplay.run(traces: [clean, clean2, clean3, uncertain])
        #expect(report.writes.count == 3)
        #expect(!report.writes.contains { $0.trace == "uncertain" })
        #expect(report.lessons.count == 1 && report.lessons.first?.status == .active)
        #expect(report.lessons.first?.count == 3)
        let unknownStep = unknownAt == "before" ? 1 : (unknownAt == "rejection" ? 2 : 3)
        #expect(!report.retrievals.contains { $0.trace == "uncertain" && $0.step == unknownStep })
        #expect(report.retrievals.contains { $0.trace == "clean3" && $0.step == 2 })
    }

    @Test(arguments: [true, false])
    func sharedUnknownExtractionIncludesStalePackets(current: Bool) {
        for packet in [regressionPacket(delivery: "unknown", current: current),
                       regressionPacket(verdict: "delivery_unknown", current: current)] {
            #expect(AgentLessonReplay.conditions(packet).contains(.deliveryUnknown))
        }
    }

    @Test func retainedContextReplacesTwoAdvisedTurnsThenRemovesAdvice() async throws {
        let directory = try regressionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("memory.json")
        var active = regressionLesson()
        active.merge(regressionLesson())
        try await AgentLessonStore(url: url).merge([active], now: regressionNow)
        let bridge = AgentLessonLoopBridge(environment: ["TURBOFIELDFARE_AGENT_LESSONS": "text",
            "TURBOFIELDFARE_AGENT_LESSONS_PATH": url.path])
        let run = UUID()
        let read = AppToolCall(id: "read", name: regressionTool,
            arguments: .object(["action": .string("observe")]))
        await bridge.settle(call: read, result: try regressionResult(read.id, regressionPacket()),
            outcome: "{\"outcome\":\"succeeded\"}", run: run, now: regressionNow)
        let reject = AppToolCall(id: "reject", name: regressionTool, arguments: .object([
            "action": .string("type"), "target": .string("field"), "text": .string("input"),
            "x_norm": .integer(20), "y_norm": .integer(30)]))
        await bridge.settle(call: reject,
            result: try regressionResult(reject.id, regressionPacket(verdict: "not_sent")),
            outcome: "{\"outcome\":\"not_sent\"}", run: run, now: regressionNow)
        let tokenizer = try await regressionTokenizer()
        let initial = [GFTokenizer.Message(role: .user, content: "Task"), regressionPending("call1")]
        let initialIDs = try tokenizer.encodeToolChat(messages: initial, tools: [regressionDefinition])
        let fake = RegressionCompletion(tokenizer: tokenizer, tokens: initialIDs)
        let runtime = GemmaMultimodalConversation(tokenizer: tokenizer, testCompletion: fake.hooks,
            messages: initial, tools: [regressionDefinition], tokenIDs: initialIDs, maxContext: 16_384)
        let conversation = MultimodalConversation(gemmaConversation: runtime)
        var callID = "call1"
        for index in 1...3 {
            let packet = regressionPacket(verdict: "not_sent", delivery: index == 3 ? "unknown" : nil)
            let sent = await bridge.advising(try regressionResult(callID, packet), now: regressionNow)
            let before = await runtime.kvTokenCount
            let completion = try await RealInferenceSession.sendToolResults(
                [sent], conversation: conversation, config: GenerationConfig(maxNewTokens: 256),
                environment: ["TURBOFIELDFARE_AGENT_LESSONS": "text"])
            callID = try #require(completion.toolCalls.first).id
            let history = await runtime.messagesForTesting()
            let bodies = try history.filter { $0.role == .tool }.map {
                try #require(JSONSerialization.jsonObject(with: Data(($0.content ?? "").utf8)) as? [String: Any])
            }
            let expectedBlocks = index < 3 ? 1 : 0
            #expect(bodies.filter { $0["lessons"] != nil }.count == expectedBlocks)
            #expect(bodies.allSatisfy { $0["guidance"] as? String == "Current guidance" })
            let attempt = try #require(fake.attempts.last)
            // Checking the actual decode prompt catches stale cache content,
            // even if only the logical message transformation remains wired.
            #expect(regressionLessonBlocks(tokenizer.decode(attempt.prompt, skipSpecialTokens: false))
                == expectedBlocks)
            let committed = await runtime.committedStateForTesting()
            #expect(regressionLessonBlocks(tokenizer.decode(committed.tokenIDs, skipSpecialTokens: false))
                == expectedBlocks)
            if index > 1 { #expect(completion.turn.cachedTokens < before) }
            else { #expect(completion.turn.cachedTokens == before) }
        }
    }

    @Test(arguments: ["malformed", "repetition"])
    func retainedAdvisedImageSurvivesFailedReplacementAndRetry(failure: String) async throws {
        let tokenizer = try await regressionTokenizer()
        let bytes = VisionConfig().textHiddenSize * MemoryLayout<Float16>.stride
        let buffer = RegressionFeatureBuffer(length: bytes)
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0x5a, count: bytes)
        let originalBytes = Data(bytes: buffer.contents(), count: bytes)
        let features = VisionFeatures(buffer: buffer, tokenCount: 1,
            hiddenSize: VisionConfig().textHiddenSize, gpuNanoseconds: 0, scratchBytes: bytes,
            attentionVariant: .native72Q16, projectorPath: .affineThreadgroupF16,
            expertResidencyTransition: nil, preprocessing: nil)
        let messages = [GFTokenizer.Message(role: .user, content: "Task"), regressionPending("old"),
            GFTokenizer.Message(role: .tool,
                content: "{\"guidance\":\"Current guidance\",\"lessons\":[\"old advice\"]}",
                toolCallID: "old", name: regressionTool, toolImageCount: 1), regressionPending("next")]
        let template = try tokenizer.encodeToolChat(messages: messages, tools: [regressionDefinition])
        let input = try MultimodalPromptRenderer.expandingImageTokens(template, features: [features])
        let fake = RegressionCompletion(tokenizer: tokenizer, tokens: input.effectiveTokenIDs)
        let runtime = GemmaMultimodalConversation(tokenizer: tokenizer, testCompletion: fake.hooks,
            messages: messages, tools: [regressionDefinition], tokenIDs: input.effectiveTokenIDs,
            imageSpans: input.imageSpans, maxContext: 16_384)
        let conversation = MultimodalConversation(gemmaConversation: runtime)
        let sent = AppToolResult(callID: "next", name: regressionTool,
            content: "{\"guidance\":\"Current guidance\",\"lessons\":[\"new advice\"]}")
        let environment = ["TURBOFIELDFARE_AGENT_LESSONS": "text"]
        fake.failNext(failure)
        do {
            _ = try await RealInferenceSession.sendToolResults([sent], conversation: conversation,
                config: GenerationConfig(maxNewTokens: 256), environment: environment,
                detectThoughtRepetition: true)
            Issue.record("Expected failed generation")
        } catch let error as StructuredToolFailure {
            #expect(failure == "malformed" && error.underlying == .malformed)
            #expect(error.canRegenerateToolResult)
        } catch let error as ThoughtRepetitionRecovery {
            #expect(failure == "repetition" && error.canRetryToolResult)
            #expect(error.restoredTokenCount == input.effectiveTokenIDs.count && error.requiresRebuild)
        }
        #expect(await runtime.messagesForTesting() == messages)
        let restored = await runtime.committedStateForTesting()
        #expect(restored.tokenIDs == input.effectiveTokenIDs)
        #expect(restored.imageSpans.map(\.tokenRange) == input.imageSpans.map(\.tokenRange))
        let restoredFeature = try #require(restored.imageSpans.first).features
        #expect(restoredFeature.buffer === buffer)
        #expect(Data(bytes: restoredFeature.buffer.contents(), count: bytes) == originalBytes)
        let failedAttempt = try #require(fake.attempts.first)
        #expect(failedAttempt.cached < input.imageSpans[0].tokenRange.lowerBound)
        #expect(try #require(failedAttempt.input?.imageSpans.first).features.buffer === buffer)
        let completion = try await RealInferenceSession.sendToolResults([sent], conversation: conversation,
            config: GenerationConfig(maxNewTokens: 256), environment: environment)
        #expect(completion.toolCalls.count == 1)
        let retry = try #require(fake.attempts.last)
        #expect(retry.cached == 0)
        #expect(regressionLessonBlocks(tokenizer.decode(retry.prompt, skipSpecialTokens: false)) == 1)
        let retained = await runtime.committedStateForTesting()
        let retainedSpan = try #require(retained.imageSpans.first)
        #expect(retained.imageSpans.count == 1 && retainedSpan.features.buffer === buffer)
        #expect(retainedSpan.features.tokenCount == features.tokenCount)
        #expect(Data(bytes: retainedSpan.features.buffer.contents(), count: bytes) == originalBytes)
        #expect(retained.tokenIDs[retainedSpan.tokenRange].allSatisfy {
            $0 == MultimodalPromptRenderer.imageTokenID
        })
        #expect(try #require(retry.input?.imageSpans.first).features.buffer === buffer)
    }

    @Test func replacementIsGenericAndDisabledPreservesExactBytes() throws {
        let original = GFTokenizer.Message(role: .tool,
            content: "{ \"transient\": \"old\", \"keep\": 1 }", toolCallID: "call", name: "tool", toolImageCount: 2)
        let user = GFTokenizer.Message(role: .user, content: original.content ?? "")
        let history = [user, original]
        #expect(try GemmaMultimodalConversation.replacingToolResultFields([], in: history) == history)
        let changed = try GemmaMultimodalConversation.replacingToolResultFields(["transient"], in: history)
        #expect(changed[0] == user)
        #expect(changed[1].content == "{\"keep\":1}")
        #expect(changed[1].toolCallID == "call" && changed[1].toolImageCount == 2)
        #expect(try GemmaMultimodalConversation.replacingToolResultFields(["absent"], in: history) == history)
    }

    @Test(arguments: ["off", "TEXT", ""])
    func disabledProductionHandoffPreservesHistoryAndPromptBytes(flag: String) async throws {
        let tokenizer = try await regressionTokenizer()
        let original = "{ \"lessons\": [\"old\"], \"keep\": 1 }"
        let messages = [GFTokenizer.Message(role: .user, content: "Task"), regressionPending("old"),
            GFTokenizer.Message(role: .tool, content: original, toolCallID: "old", name: regressionTool),
            regressionPending("next")]
        let ids = try tokenizer.encodeToolChat(messages: messages, tools: [regressionDefinition])
        let fake = RegressionCompletion(tokenizer: tokenizer, tokens: ids)
        let runtime = GemmaMultimodalConversation(tokenizer: tokenizer, testCompletion: fake.hooks,
            messages: messages, tools: [regressionDefinition], tokenIDs: ids, maxContext: 16_384)
        let result = AppToolResult(callID: "next", name: regressionTool, content: "{ \"keep\": 2 }")
        let incoming = messages + [GFTokenizer.Message(role: .tool, content: result.content,
            toolCallID: result.callID, name: result.name)]
        let suffix = try tokenizer.encodeToolResultContinuation(cachedMessages: Array(messages.dropLast()),
            assistant: messages.last!, incomingMessages: incoming, tools: [regressionDefinition])
        let completion = try await RealInferenceSession.sendToolResults([result],
            conversation: MultimodalConversation(gemmaConversation: runtime),
            config: GenerationConfig(maxNewTokens: 256), environment: ["TURBOFIELDFARE_AGENT_LESSONS": flag])
        #expect(completion.turn.cachedTokens == ids.count)
        #expect(try #require(fake.attempts.last).prompt == ids + suffix)
        #expect(Array(await runtime.messagesForTesting().prefix(messages.count)) == messages)
    }

    @Test(arguments: ["text", "embedding", "off", "TEXT", ""])
    func serviceAndRuntimeReplacementFollowExactFlag(flag: String) {
        let environment = ["TURBOFIELDFARE_AGENT_LESSONS": flag]
        let launched = DecodeServiceInferenceClient.launchEnvironment(environment: environment, thinking: false)
        let enabled = ["text", "embedding"].contains(flag)
        #expect(launched["TURBOFIELDFARE_AGENT_LESSONS"] == (enabled ? flag : nil))
        #expect(AgentLessonLoopBridge.contextReplacementFields(environment: launched) == (enabled ? ["lessons"] : []))
    }

    @Test(arguments: ["text", "off"])
    func storeFailureIsBoundedTracedOnlyWhenEnabled(flag: String) async throws {
        let directory = try regressionDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let blocked = directory.appendingPathComponent("private-store-path")
        try Data().write(to: blocked)
        let traceURL = directory.appendingPathComponent("trace.jsonl")
        let trace = try #require(AgentInferenceTrace.forTesting(path: traceURL.path))
        let bridge = AgentLessonLoopBridge(environment: ["TURBOFIELDFARE_AGENT_LESSONS": flag,
            "TURBOFIELDFARE_AGENT_LESSONS_PATH": blocked.appendingPathComponent("memory.json").path], trace: trace)
        try await regressionLearn(bridge, run: UUID())
        let lines = try String(contentsOf: traceURL, encoding: .utf8).split(separator: "\n")
        let rows = try lines.map {
            try #require(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
        }
        let errors = rows.filter { $0["event"] as? String == "lessons_store_error" }
        #expect(errors.count == (flag == "text" ? 1 : 0))
        for (line, row) in zip(lines, rows) where row["event"] as? String == "lessons_store_error" {
            #expect(line.utf8.count + 1 <= 512)
        }
        #expect(!rows.contains { $0["event"] as? String == "lessons_written" })
        #expect(!String(decoding: try Data(contentsOf: traceURL), as: UTF8.self).contains("private-store-path"))
        let input = try regressionResult("packet", regressionPacket(delivery: "unknown"))
        #expect(await bridge.advising(input, now: regressionNow) == input)
    }

    @Test func lessonSourcesHaveNoLinesOver120Columns() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/TurboFieldfareApp/Core/Tools")
        for url in try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        where url.lastPathComponent.hasPrefix("AgentLesson") && url.pathExtension == "swift" {
            let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: false)
            for (index, line) in lines.enumerated() {
                #expect(line.count <= 120, "\(url.lastPathComponent):\(index + 1) has \(line.count) columns")
            }
        }
    }
}
