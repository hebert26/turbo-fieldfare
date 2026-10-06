import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Opt-in P22 tests that run the real Qwen runner through the pinned tokenizer
/// and structured assistant decoder. No parsed call is dispatched to a host.
@Suite(.serialized)
struct QwenRealToolCodecTests {
    @Test(.enabled(if: RealQwenToolArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact tool checks"))
    func realThinkingOnAndOffCaptureRawTokensAndStructuredOutput() async throws {
        let artifact = try RealQwenToolArtifact.load()
        let prompt = "Answer with one short word: blue."
        let thinkingOn = try await artifact.capture(
            userPrompt: prompt, tools: [], thinking: true, maxNewTokens: 64)
        let thinkingOff = try await artifact.capture(
            userPrompt: prompt, tools: [], thinking: false, maxNewTokens: 64)

        #expect(!thinkingOn.generatedTokenIDs.isEmpty)
        #expect(!thinkingOff.generatedTokenIDs.isEmpty)
        #expect(thinkingOn.rawText.contains("</think>"))
        #expect(!thinkingOff.rawText.contains("<think>"))
        #expect(!thinkingOff.rawText.contains("</think>"))
        #expect(!thinkingOn.events.isEmpty)
        #expect(!thinkingOff.events.isEmpty)
        #expect(thinkingOn.parserError == nil)
        #expect(thinkingOff.parserError == nil)
        #expect(thinkingOn.events.allSatisfy { event in
            if case .toolCall = event { return false }
            return true
        })
        #expect(thinkingOff.events.allSatisfy { event in
            if case .toolCall = event { return false }
            return true
        })
        #expect(thinkingOn.termination == .modelEOS)
        #expect(thinkingOff.termination == .modelEOS)
        RealQwenToolEvidence.report(
            suite: "thinking-modes", artifact: artifact,
            captures: [thinkingOn, thinkingOff])
    }

    @Test(.enabled(if: RealQwenToolArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact tool checks"))
    func realAllowedToolOutputPublishesOnlyOneCompleteCallWithoutDispatch() async throws {
        let artifact = try RealQwenToolArtifact.load()
        let capture = try await artifact.capture(
            userPrompt: "Call lookup with query qwen real route.",
            tools: [RealQwenToolArtifact.lookupTool()],
            thinking: false,
            maxNewTokens: 64)
        let calls = capture.events.compactMap { event -> ParsedToolCall? in
            guard case .toolCall(let call) = event else { return nil }
            return call
        }

        #expect(capture.rawText.contains("<tool_call>"))
        #expect(capture.rawText.contains("</tool_call>"))
        #expect(capture.termination == .modelEOS)
        #expect(capture.parserError == nil)
        #expect(calls.count == 1)
        guard let call = calls.first else {
            Issue.record("real allowed-tool output produced no parsed call")
            RealQwenToolEvidence.report(
                suite: "allowed-tool", artifact: artifact,
                captures: [capture], note: "no parsed call was available for field assertions")
            return
        }
        #expect(call.id == "real-call-1")
        #expect(call.name == "lookup")
        #expect(call.arguments != .null)
        RealQwenToolEvidence.report(
            suite: "allowed-tool", artifact: artifact,
            captures: [capture])
    }

    @Test(.enabled(if: RealQwenToolArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact tool checks"))
    func truncatedRealToolOutputRemainsNonExecutable() async throws {
        let artifact = try RealQwenToolArtifact.load()
        let capture = try await artifact.capture(
            userPrompt: "Call lookup with query qwen real route.",
            tools: [RealQwenToolArtifact.lookupTool()],
            thinking: false,
            maxNewTokens: 64)
        guard let close = capture.rawText.range(of: "</tool_call>") else {
            Issue.record("real tool capture did not contain a complete closing marker")
            RealQwenToolEvidence.report(
                suite: "malformed-derived-truncation", artifact: artifact,
                captures: [capture],
                note: "capture had no closing marker for deterministic truncation")
            return
        }
        let truncated = String(capture.rawText[..<close.lowerBound])
        var decoder = QwenStructuredAssistantDecoder(
            tools: [RealQwenToolArtifact.lookupTool()],
            startsInThoughtChannel: false,
            idGenerator: { "derived-call-should-not-release" })
        var observed: [StructuredAssistantEvent] = []
        do {
            observed += try decoder.consume(truncated)
            try decoder.markEndOfStream()
            observed += try decoder.consumeTerminalTail("")
            observed += try decoder.finish()
            Issue.record("truncated real tool output unexpectedly finalized")
        } catch {
            // This is a deterministic negative derived from the captured real
            // output. Failure before finish is the required non-dispatch path.
        }
        let calls = observed.compactMap { event -> ParsedToolCall? in
            guard case .toolCall(let call) = event else { return nil }
            return call
        }
        #expect(calls.isEmpty)
        RealQwenToolEvidence.report(
            suite: "malformed-derived-truncation", artifact: artifact,
            captures: [capture],
            note: "negative derived by truncating captured raw output before </tool_call>")
    }
}

private struct RealQwenToolCapture: Sendable {
    enum Termination: String, Sendable {
        case modelEOS
        case maxTokens
    }

    let promptTokenIDs: [Int32]
    let generatedTokenIDs: [Int32]
    let rawText: String
    let events: [StructuredAssistantEvent]
    let termination: Termination
    let parserError: String?
}

private struct RealQwenToolArtifact {
    static let environmentKey = "TURBO_FIELDFARE_REAL_QWEN_ARTIFACT"

    static var isOptedIn: Bool {
        guard let value = ProcessInfo.processInfo.environment[environmentKey]
        else { return false }
        return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    let directory: URL
    let manifest: LoadedModelManifest
    let identity: LoadedRuntimeIdentity

    static func load() throws -> Self {
        guard let raw = ProcessInfo.processInfo.environment[environmentKey],
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RealQwenToolArtifactError.notOptedIn
        }
        let directory = URL(fileURLWithPath: raw, isDirectory: true).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw RealQwenToolArtifactError.missingDirectory(directory.path)
        }
        guard case .qwenV2(let manifest) = try ModelFamilyAdmission.classify(
            directoryURL: directory) else {
            throw RealQwenToolArtifactError.wrongFamily
        }
        let identity = LoadedRuntimeIdentity(descriptor: manifest.descriptor)
        guard identity.family == .qwen3_6,
              identity.modelID == "Qwen/Qwen3.6-35B-A3B",
              identity.sourceRevision == "995ad96eacd98c81ed38be0c5b274b04031597b0" else {
            throw RealQwenToolArtifactError.identityMismatch(identity.modelID)
        }
        return Self(directory: directory, manifest: manifest, identity: identity)
    }

    static func lookupTool() -> ModelChatToolDefinition {
        ModelChatToolDefinition(function: .init(
            name: "lookup",
            description: "Return the supplied query without external action.",
            parameters: .object([
                .init("type", .string("object")),
                .init("properties", .object([
                    .init("query", .object([
                        .init("type", .string("string")),
                    ])),
                ])),
                .init("required", .array([.string("query")])),
                .init("additionalProperties", .bool(false)),
            ])))
    }

    func capture(
        userPrompt: String,
        tools: [ModelChatToolDefinition],
        thinking: Bool,
        maxNewTokens: Int
    ) async throws -> RealQwenToolCapture {
        let context = try MetalContext()
        let model = try QwenTextModel.loadOfficial(
            directoryURL: directory, manifest: manifest, device: context.device)
        let tokenizer = try QwenTokenizer.load(from: directory)
        let codec = QwenChatCodec(tokenizer: tokenizer)
        let promptTokenIDs = try codec.encodePrompt(
            messages: [.init(role: .user, content: userPrompt)],
            tools: tools,
            options: .init(enableThinking: thinking))
        guard !promptTokenIDs.isEmpty else {
            throw RealQwenToolArtifactError.emptyPrompt
        }

        let producer = try model.makeLogitProducer(context: context, expertSlotCount: 8)
        let scratch = try RawCompletionScratch(
            context: context, vocab: model.architecture.vocabularySize)
        let config = GenerationConfig.qwenRaw(
            maxNewTokens: maxNewTokens,
            temperature: 0,
            topK: nil,
            topP: nil,
            repetitionPenalty: 1,
            seed: 0,
            stopStrings: [],
            stopTokenIDs: [tokenizer.eosID])
        producer.reset()
        for (position, token) in promptTokenIDs.enumerated() {
            try await producer.produce(
                token: token, position: position, into: scratch.logits)
        }

        var history = promptTokenIDs
        var generated: [Int32] = []
        generated.reserveCapacity(maxNewTokens)
        var termination: RealQwenToolCapture.Termination = .maxTokens
        for step in 0..<maxNewTokens {
            try Task.checkCancellation()
            guard let command = context.queue.makeCommandBuffer() else {
                throw RealQwenToolArtifactError.commandBuffer
            }
            _ = scratch.sampler.sample(
                commandBuffer: command,
                logits: scratch.logits,
                probs: scratch.probs,
                history: history,
                config: config,
                position: step,
                outToken: scratch.outToken)
            command.commit()
            _ = await command.completed()
            try checkCommandBufferError(command)
            let token = Int32(
                bitPattern: scratch.outToken.contents().load(as: UInt32.self))
            generated.append(token)
            if token == tokenizer.eosID {
                termination = .modelEOS
                break
            }
            history.append(token)
            try await producer.produce(
                token: token,
                position: promptTokenIDs.count + step,
                into: scratch.logits)
        }

        let rawText = tokenizer.decode(
            generated.filter { $0 != tokenizer.eosID }, skipSpecialTokens: false)
        var decoder = QwenStructuredAssistantDecoder(
            tools: tools,
            startsInThoughtChannel: thinking,
            idGenerator: { "real-call-1" })
        var events: [StructuredAssistantEvent] = []
        var parserError: String?
        for character in rawText {
            guard parserError == nil else { break }
            do {
                events += try decoder.consume(String(character))
            } catch {
                parserError = String(describing: error)
            }
        }
        if parserError == nil, termination == .modelEOS {
            do {
                try decoder.markEndOfStream()
                events += try decoder.consumeTerminalTail("")
                events += try decoder.finish()
            } catch {
                parserError = String(describing: error)
            }
        } else if parserError == nil {
            parserError = RealQwenToolArtifactError.missingEOS(
                rawText.prefix(160).description).description
        }
        return RealQwenToolCapture(
            promptTokenIDs: promptTokenIDs,
            generatedTokenIDs: generated,
            rawText: rawText,
            events: events,
            termination: termination,
            parserError: parserError)
    }
}

private enum RealQwenToolArtifactError: Error, CustomStringConvertible {
    case notOptedIn
    case missingDirectory(String)
    case wrongFamily
    case identityMismatch(String)
    case emptyPrompt
    case commandBuffer
    case missingEOS(String)

    var description: String {
        switch self {
        case .notOptedIn: "real Qwen artifact opt-in is missing"
        case .missingDirectory(let path): "real Qwen artifact is not a directory: \(path)"
        case .wrongFamily: "selected artifact was not admitted as Qwen3.6"
        case .identityMismatch(let modelID): "unexpected Qwen identity: \(modelID)"
        case .emptyPrompt: "real Qwen chat prompt encoded no tokens"
        case .commandBuffer: "real Qwen sampler command buffer was unavailable"
        case .missingEOS(let prefix): "real Qwen output hit its bound without EOS: \(prefix)"
        }
    }
}

private struct RealQwenToolEvidence {
    static func report(
        suite: String,
        artifact: RealQwenToolArtifact,
        captures: [RealQwenToolCapture],
        note: String? = nil
    ) {
        let rendered = captures.map { capture in
            "prompt=\(capture.promptTokenIDs) raw=\(capture.generatedTokenIDs) "
                + "text=\(capture.rawText.debugDescription) "
                + "events=\(capture.events.debugDescription) "
                + "stop=\(capture.termination.rawValue) "
                + "parserError=\(String(describing: capture.parserError))"
        }.joined(separator: " | ")
        print(
            "P22 case=\(suite) tokenSource=actual-greedy-sampler "
                + "modelID=\(artifact.identity.modelID) "
                + "revision=\(artifact.identity.sourceRevision) "
                + "format=\(artifact.identity.formatMajor).\(artifact.identity.formatMinor) "
                + "manifest=\(artifact.identity.textManifestSHA256) "
                + "sourceIndex=\(artifact.identity.sourceIndexSHA256) "
                + "policy=\(artifact.identity.quantizationPolicySHA256) "
                + rendered
                + (note.map { " note=\($0)" } ?? ""))
    }
}
