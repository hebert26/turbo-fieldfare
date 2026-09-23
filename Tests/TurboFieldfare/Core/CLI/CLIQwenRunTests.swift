import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

@Suite struct CLIQwenRunTests {
    @Test func routingPreservesMessagesToolsThinkingAndOutputOrder() async throws {
        let directory = try makeTemporaryDirectory(named: "cli-qwen-routing")
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesURL = directory.appendingPathComponent("messages.json")
        let toolsURL = directory.appendingPathComponent("tools.json")
        try Data(#"""
        [
          {"role":"system","content":"follow the policy"},
          {"role":"assistant","content":"answer","reasoning_content":"private reason",
           "tool_calls":[{"type":"function","function":{"name":"lookup","arguments":{"z":1,"a":"x"}}}]},
          {"role":"tool","content":"tool result","tool_call_id":"call-1","name":"lookup"},
          {"role":"user","content":[{"type":"text","text":"ask now"}]}
        ]
        """#.utf8).write(to: messagesURL)
        try Data(#"""
        [
          {"type":"function","function":{"name":"lookup","description":"look something up",
           "parameters":{"type":"object","properties":{"z":{"type":"integer"},"a":{"type":"string"}},"required":["z"]}}}
        ]
        """#.utf8).write(to: toolsURL)

        let session = RecordingCLISession(identity: try QwenP18FixtureSupport.identity())
        let counters = CLITestCounters()
        let dependencies = CLIRunDependencies(
            inspect: { _ in
                counters.inspected()
                return .init(
                    family: .qwen3_6,
                    verifiedIdentity: session.verifiedIdentity)
            },
            preflightQwen: { _, prompt, images, _, residency, maximum in
                counters.preflighted()
                #expect(images.isEmpty)
                #expect(residency == .defaultPolicy)
                #expect(maximum == 8)
                guard case .chat(let messages, let tools, let thinking) = prompt else {
                    Issue.record("expected a chat request")
                    return .init(promptTokens: 3, imageCount: 0)
                }
                #expect(messages.count == 4)
                #expect(tools.count == 1)
                #expect(thinking == .disabled)
                guard case .object(let members) = tools[0].function.parameters else {
                    Issue.record("tool parameters lost their object shape")
                    return .init(promptTokens: 3, imageCount: 0)
                }
                #expect(members.map(\.name) == ["type", "properties", "required"])
                return .init(promptTokens: 3, imageCount: 0)
            },
            loadSession: { _, maxContext, runtime, _ in
                counters.loaded()
                #expect(maxContext == 8)
                #expect(runtime.headPath == .logits)
                return session
            })
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--messages-file", messagesURL.path,
            "--tools-file", toolsURL.path, "--thinking", "off",
            "--show-model-identity", "--max-new", "100", "--max-context", "8", "--quiet",
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)

        #expect(output.result.exitCode == 0)
        #expect(output.stdout == "firstsecond")
        #expect(output.stderr.contains("family=qwen3_6"))
        #expect(output.stderr.contains("format=2.0"))
        #expect(counters.inspectedCount == 1)
        #expect(counters.preflightCount == 1)
        #expect(counters.loadedCount == 1)
        let request = try #require(session.request)
        #expect(request.config.maxNewTokens == 5)
        guard case .chat(let messages, let tools, let thinking) = request.prompt else {
            Issue.record("recorded request was not chat")
            return
        }
        #expect(thinking == .disabled)
        #expect(messages[1].reasoningContent == "private reason")
        #expect(messages[1].toolCalls.count == 1)
        #expect(messages[1].toolCalls[0].name == "lookup")
        #expect(messages[1].toolCalls[0].id == nil)
        guard case .object(let arguments) = messages[1].toolCalls[0].arguments else {
            Issue.record("historical tool arguments lost their object shape")
            return
        }
        #expect(arguments.map(\.name) == ["z", "a"])
        #expect(tools[0].function.name == "lookup")
    }

    @Test func streamedTextAndToolEventsRemainInPublishedOrder() async throws {
        let session = RecordingCLISession(identity: nil, events: [
            .text("A"),
            .text("B"),
            .toolCall(.init(
                id: "call-1", name: "lookup", arguments: .object(["z": .integer(1)]),
                argumentsJSON: "{\"z\":1}")),
        ])
        let counters = CLITestCounters()
        let dependencies = qwenDependencies(session: session, counters: counters)
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--chat-prompt", "hello", "--quiet",
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 0)
        #expect(output.stdout == "AB{\"tool_call\":{\"id\":\"call-1\",\"name\":\"lookup\",\"arguments\":{\"z\":1}}}\n")
    }

    @Test(.enabled(if: VisionRuntime.isSupportedOnDefaultDevice,
                   "requires the verified Qwen image hardware gate"))
    func messagesFileImagesRetainPromptOrderAndIDs() async throws {
        let directory = try makeTemporaryDirectory(named: "cli-qwen-images")
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesURL = directory.appendingPathComponent("messages.json")
        try Data(#"""
        [{"role":"user","content":[
          {"type":"image_file","path":"first.png"},
          {"type":"text","text":"compare"},
          {"type":"image_file","path":"second.png"}
        ]}]
        """#.utf8).write(to: messagesURL)

        let session = RecordingCLISession(identity: nil)
        let counters = CLITestCounters()
        let dependencies = CLIRunDependencies(
            inspect: { _ in
                counters.inspected()
                return .init(family: .qwen3_6, verifiedIdentity: nil)
            },
            preflightQwen: { _, prompt, images, _, residency, _ in
                counters.preflighted()
                #expect(residency == .defaultPolicy)
                guard case .chat(let messages, _, _) = prompt,
                      case .parts(let parts)? = messages[0].content else {
                    Issue.record("image messages were not routed as structured content")
                    return .init(promptTokens: 3, imageCount: 2)
                }
                let ids = parts.compactMap { part -> String? in
                    guard case .image(let media) = part else { return nil }
                    return media.id
                }
                #expect(ids == ["messages-image-1", "messages-image-2"])
                #expect(images["messages-image-1"] == directory.appendingPathComponent("first.png"))
                #expect(images["messages-image-2"] == directory.appendingPathComponent("second.png"))
                return .init(promptTokens: 3, imageCount: 2)
            },
            loadSession: { _, _, _, _ in
                counters.loaded()
                return session
            })
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--messages-file", messagesURL.path, "--quiet",
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 0)
        #expect(counters.inspectedCount == 1)
        #expect(counters.preflightCount == 1)
        #expect(counters.loadedCount == 1)
        let request = try #require(session.request)
        #expect(request.visionResidency == .defaultPolicy)
        #expect(request.imagesByID["messages-image-1"] == directory.appendingPathComponent("first.png"))
        #expect(request.imagesByID["messages-image-2"] == directory.appendingPathComponent("second.png"))
    }

    @Test(.enabled(if: VisionRuntime.isSupportedOnDefaultDevice,
                   "requires the verified Qwen image hardware gate"))
    func chatImagesRetainRepeatedFlagOrderResidencyAndRequestMapping() async throws {
        let first = URL(fileURLWithPath: "/tmp/first.png")
        let second = URL(fileURLWithPath: "/tmp/second.png")
        let session = RecordingCLISession(identity: nil)
        let counters = CLITestCounters()
        let dependencies = CLIRunDependencies(
            inspect: { _ in
                counters.inspected()
                return .init(family: .qwen3_6, verifiedIdentity: nil)
            },
            preflightQwen: { _, prompt, images, _, residency, maximum in
                counters.preflighted()
                #expect(residency == .onDemand)
                #expect(maximum == 12)
                #expect(images["cli-image-1"] == first)
                #expect(images["cli-image-2"] == second)
                guard case .chat(let messages, _, _) = prompt,
                      case .parts(let parts) = messages[0].content else {
                    Issue.record("flag images were not routed as message parts")
                    return .init(promptTokens: 3, imageCount: 2)
                }
                let ids = parts.compactMap { part -> String? in
                    guard case .image(let media) = part else { return nil }
                    return media.id
                }
                #expect(ids == ["cli-image-1", "cli-image-2"])
                return .init(promptTokens: 3, imageCount: 2)
            },
            loadSession: { _, maxContext, _, _ in
                counters.loaded()
                #expect(maxContext == 12)
                return session
            })
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--chat-prompt", "compare",
            "--image", first.path, "--image", second.path,
            "--vision-residency", "on-demand", "--max-context", "12", "--quiet",
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 0)
        #expect(counters.inspectedCount == 1)
        #expect(counters.preflightCount == 1)
        #expect(counters.loadedCount == 1)
        let request = try #require(session.request)
        #expect(request.visionResidency == .onDemand)
        #expect(request.imagesByID["cli-image-1"] == first)
        #expect(request.imagesByID["cli-image-2"] == second)
    }

    @Test func malformedMessagesFailBeforeInspectionPreflightAndLoad() async throws {
        let directory = try makeTemporaryDirectory(named: "cli-qwen-malformed")
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesURL = directory.appendingPathComponent("messages.json")
        try Data(#"[{"role":"user","content":[{"type":"video_file","path":"x.mp4"}]}]"#.utf8)
            .write(to: messagesURL)
        let counters = CLITestCounters()
        let session = RecordingCLISession(identity: nil)
        let dependencies = CLIRunDependencies(
            inspect: { _ in
                counters.inspected()
                throw UnexpectedCLITestCall.inspect
            },
            preflightQwen: { _, _, _, _, _, _ in
                counters.preflighted()
                throw UnexpectedCLITestCall.preflight
            },
            loadSession: { _, _, _, _ in
                counters.loaded()
                throw UnexpectedCLITestCall.load
            })
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--messages-file", messagesURL.path,
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 2)
        #expect(output.stderr.contains("video input is not supported"))
        #expect(counters.inspectedCount == 0)
        #expect(counters.preflightCount == 0)
        #expect(counters.loadedCount == 0)
        _ = session
    }

    @Test func malformedJSONFailsBeforeInspectionPreflightAndLoad() async throws {
        let directory = try makeTemporaryDirectory(named: "cli-qwen-invalid-json")
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesURL = directory.appendingPathComponent("messages.json")
        try Data(#"[{"#.utf8).write(to: messagesURL)
        let counters = CLITestCounters()
        let dependencies = rejectingDependencies(counters: counters)
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--messages-file", messagesURL.path,
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 1)
        #expect(output.stderr.contains("invalid JSON"))
        #expect(counters.inspectedCount == 0)
        #expect(counters.preflightCount == 0)
        #expect(counters.loadedCount == 0)
    }

    @Test func oversizedMessagesFileFailsBeforeInspectionPreflightAndLoad() async throws {
        let directory = try makeTemporaryDirectory(named: "cli-qwen-large-messages")
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesURL = directory.appendingPathComponent("messages.json")
        try Data(repeating: 0x20, count: 4 * 1_024 * 1_024 + 1).write(to: messagesURL)
        let counters = CLITestCounters()
        let dependencies = rejectingDependencies(counters: counters)
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--messages-file", messagesURL.path,
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 2)
        #expect(output.stderr.contains("file exceeds 4194304 bytes"))
        #expect(counters.inspectedCount == 0)
        #expect(counters.preflightCount == 0)
        #expect(counters.loadedCount == 0)
    }

    @Test func oversizedToolsFileFailsBeforeInspectionPreflightAndLoad() async throws {
        let directory = try makeTemporaryDirectory(named: "cli-qwen-large-tools")
        defer { try? FileManager.default.removeItem(at: directory) }
        let messagesURL = directory.appendingPathComponent("messages.json")
        let toolsURL = directory.appendingPathComponent("tools.json")
        try Data(#"[{"role":"user","content":"hello"}]"#.utf8)
            .write(to: messagesURL)
        try Data(repeating: 0x20, count: 4 * 1_024 * 1_024 + 1).write(to: toolsURL)
        let counters = CLITestCounters()
        let dependencies = rejectingDependencies(counters: counters)
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--messages-file", messagesURL.path,
            "--tools-file", toolsURL.path,
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 2)
        #expect(output.stderr.contains("file exceeds 4194304 bytes"))
        #expect(counters.inspectedCount == 0)
        #expect(counters.preflightCount == 0)
        #expect(counters.loadedCount == 0)
    }

    @Test func preflightContextOverflowFailsBeforeSessionLoad() async throws {
        let counters = CLITestCounters()
        let session = RecordingCLISession(identity: nil)
        let dependencies = CLIRunDependencies(
            inspect: { _ in
                counters.inspected()
                return .init(family: .qwen3_6, verifiedIdentity: nil)
            },
            preflightQwen: { _, _, _, _, _, _ in
                counters.preflighted()
                throw ModelFamilyGenerationError.contextOverflow(
                    prompt: 8, maxNew: 100, maximum: 8)
            },
            loadSession: { _, _, _, _ in
                counters.loaded()
                return session
            })
        let args = try Args.parse([
            "--model", "qwen.gturbo", "--chat-prompt", "hello", "--max-new", "100",
        ])
        let output = try await runAndCapture(args: args, dependencies: dependencies)
        #expect(output.result.exitCode == 1)
        #expect(output.stderr.contains("context overflow"))
        #expect(counters.inspectedCount == 1)
        #expect(counters.preflightCount == 1)
        #expect(counters.loadedCount == 0)
    }

    private func qwenDependencies(
        session: RecordingCLISession,
        counters: CLITestCounters
    ) -> CLIRunDependencies {
        CLIRunDependencies(
            inspect: { _ in
                counters.inspected()
                return .init(family: .qwen3_6, verifiedIdentity: session.verifiedIdentity)
            },
            preflightQwen: { _, _, images, _, _, maximum in
                counters.preflighted()
                #expect(images.isEmpty)
                return .init(promptTokens: min(3, maximum - 1), imageCount: 0)
            },
            loadSession: { _, _, _, _ in
                counters.loaded()
                return session
            })
    }

    private func rejectingDependencies(counters: CLITestCounters) -> CLIRunDependencies {
        CLIRunDependencies(
            inspect: { _ in
                counters.inspected()
                throw UnexpectedCLITestCall.inspect
            },
            preflightQwen: { _, _, _, _, _, _ in
                counters.preflighted()
                throw UnexpectedCLITestCall.preflight
            },
            loadSession: { _, _, _, _ in
                counters.loaded()
                throw UnexpectedCLITestCall.load
            })
    }

    private func makeTemporaryDirectory(named name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func runAndCapture(
        args: Args,
        dependencies: CLIRunDependencies
    ) async throws -> CapturedCLIOutput {
        let stdout = Pipe()
        let stderr = Pipe()
        let result = await run(
            args: args,
            dependencies: dependencies,
            stdout: stdout.fileHandleForWriting,
            stderr: stderr.fileHandleForWriting)
        stdout.fileHandleForWriting.closeFile()
        stderr.fileHandleForWriting.closeFile()
        return CapturedCLIOutput(
            result: result,
            stdout: String(
                decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self),
            stderr: String(
                decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
    }

    private struct CapturedCLIOutput {
        let result: RunResult
        let stdout: String
        let stderr: String
    }
}

private final class CLITestCounters: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var inspectedCount = 0
    private(set) var preflightCount = 0
    private(set) var loadedCount = 0

    func inspected() { lock.withLock { inspectedCount += 1 } }
    func preflighted() { lock.withLock { preflightCount += 1 } }
    func loaded() { lock.withLock { loadedCount += 1 } }
}

private final class RecordingCLISession: CLIGenerationSession, @unchecked Sendable {
    let family: LoadedRuntimeFamily = .qwen3_6
    let verifiedIdentity: LoadedRuntimeIdentity?
    private let lock = NSLock()
    private(set) var request: ModelFamilyGenerationRequest?
    private let events: [ModelFamilyGenerationEvent]

    init(
        identity: LoadedRuntimeIdentity?,
        events: [ModelFamilyGenerationEvent] = [.text("first"), .text("second")]
    ) {
        verifiedIdentity = identity
        self.events = events
    }

    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        lock.withLock { self.request = request }
        for event in events { onEvent(event) }
        return .init(
            reason: .maxTokens,
            promptTokens: 3,
            newTokens: events.count,
            prefillSeconds: 0,
            decodeSeconds: 0)
    }
}

private enum UnexpectedCLITestCall: Error {
    case inspect
    case preflight
    case load
}
