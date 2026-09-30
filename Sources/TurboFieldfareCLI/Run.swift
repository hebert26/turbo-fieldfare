import Foundation
import Metal
import TurboFieldfare

/// One row of the `--messages-file` JSON.
private struct MessageJSON: Decodable {
    let role: String
    let content: MessageContentJSON
}

private enum MessageContentJSON: Decodable {
    case text(String)
    case parts([MessagePartJSON])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([MessagePartJSON].self))
        }
    }
}

private struct MessagePartJSON: Decodable {
    let type: String
    let text: String?
    let path: String?
}

enum PromptInput {
    case raw(String)
    case messages([GFTokenizer.Message])
    case multimodal(messages: [MultimodalMessage], images: [UUID: URL])

    /// Whether this prompt carries images. `--messages-file` reaches the image
    /// path with `args.images` empty, so the runtime checks that depend on
    /// images ask the parsed input, not the flag list.
    var hasImages: Bool {
        if case .multimodal = self { return true }
        return false
    }
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

func routedExpertCacheFooter(_ summary: RoutedExpertCacheSummary) -> String {
    "\n[expert-cache scope=lifetime configured-slots=\(summary.configuredSlots) "
        + "effective-slots=\(summary.effectiveSlots) policy=\(summary.policy) "
        + "allocated-bytes=\(summary.allocatedBytes) "
        + "peak-allocated-bytes=\(summary.peakAllocatedBytes) "
        + "hits=\(summary.hits) misses=\(summary.misses)]\n"
}

protocol CLIGenerationSession: Sendable {
    var family: LoadedRuntimeFamily { get }
    var verifiedIdentity: LoadedRuntimeIdentity? { get }
    var sourceIdentity: LoadedRuntimeSourceIdentity? { get }
    func preflightLoadedSource(
        prompt: ModelFamilyGenerationPrompt,
        imagesByID: [String: URL],
        visionResidency: VisionResidencyPolicy
    ) async throws -> ModelFamilyGenerationPreflight
    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @escaping @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult
}

extension ModelFamilyGenerationSession: CLIGenerationSession {}

struct CLIRunDependencies: Sendable {
    var inspect: @Sendable (URL) throws -> ModelFamilyGenerationAdmission
    var preflightQwen: @Sendable (
        URL, ModelFamilyGenerationPrompt, [String: URL], URL?,
        VisionResidencyPolicy, Int
    ) throws -> ModelFamilyGenerationPreflight
    var loadSession: @Sendable (
        URL, Int, RuntimeConfiguration, URL?, ModelIntegrityPolicy
    ) throws -> any CLIGenerationSession

    static let live = CLIRunDependencies(
        inspect: ModelFamilyGenerationSession.inspect,
        preflightQwen: { directory, prompt, images, visionPack, residency, maximum in
            try ModelFamilyGenerationSession.preflightQwen(
                directoryURL: directory,
                prompt: prompt,
                imagesByID: images,
                visionPackURL: visionPack,
                visionResidency: residency,
                maxContext: maximum)
        },
        loadSession: { directory, maxContext, runtime, visionPack, integrity in
            try ModelFamilyGenerationSession.load(
                directoryURL: directory,
                maxContext: maxContext,
                runtimeConfiguration: runtime,
                visionPackURL: visionPack,
                integrityPolicy: integrity)
        })
}

public func run(
    args: Args,
    stdout: FileHandle = .standardOutput,
    stderr: FileHandle = .standardError
) async -> RunResult {
    await run(args: args, dependencies: .live, stdout: stdout, stderr: stderr)
}

func run(
    args: Args,
    dependencies: CLIRunDependencies,
    stdout: FileHandle,
    stderr: FileHandle
) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let parsed = try parseFamilyRequest(args: args)
        if !parsed.imagesByID.isEmpty {
            guard let device = MetalContext.makeSystemDefaultDevice() else {
                return errored(stderr, "no Metal device", 1)
            }
            try VisionRuntime.requireSupportedDevice(device)
        }
        let admission = try dependencies.inspect(modelURL)
        if admission.family == .gemma4 {
            guard args.sourceIntegrity == nil else {
                return errored(stderr, "--source-integrity requires an original BF16 source", 2)
            }
            // Preserve the legacy parser's exact role/content acceptance before
            // its tokenizer or model is opened.
            _ = try parseInput(args: args)
            guard args.thinking == .auto else {
                return errored(
                    stderr,
                    "explicit --thinking is available only for verified Qwen models",
                    2)
            }
            guard args.toolsFile == nil else {
                return errored(
                    stderr,
                    "--tools-file is available only for verified Qwen models",
                    2)
            }
            return await legacyRun(
                args: args,
                showModelIdentity: args.showModelIdentity,
                stdout: stdout,
                stderr: stderr)
        }

        var effectiveArgs = args
        if args.prefillChunkTokensAuto {
            // Qwen's current runner uses its native scalar/prepared prefill.
            // Resolve the CLI's runtime value deterministically without
            // changing Gemma's prompt-sized auto path above.
            effectiveArgs.prefillChunkTokens =
                PrefillRuntimeConfig.autoChunkTokens(promptTokens: args.maxContext)
        }
        let visionPackURL = args.visionPack.map { URL(fileURLWithPath: $0) }
        let sourceBacking = admission.verifiedIdentity == nil
        guard sourceBacking || args.sourceIntegrity == nil else {
            return errored(stderr, "--source-integrity requires an original BF16 source", 2)
        }
        let runtime = try effectiveArgs.resolvedRuntimeConfiguration(
            forceLogitsHead: true,
            imagePrompt: !parsed.imagesByID.isEmpty)
        let preflight: ModelFamilyGenerationPreflight
        let session: any CLIGenerationSession
        if sourceBacking {
            // Obvious missing companions fail before source payload verification.
            // A present companion is admitted against the loaded source below.
            if !parsed.imagesByID.isEmpty {
                let companion = try visionPackURL
                    ?? VisionPackLocation.companionURL(forTextModel: modelURL)
                guard FileManager.default.fileExists(atPath: companion.path) else {
                    throw ModelFamilyGenerationError.sourceVisionUnavailable
                }
            }
            session = try dependencies.loadSession(
                modelURL, args.maxContext, runtime, visionPackURL,
                (args.sourceIntegrity ?? .fullSHA256).policy)
            guard session.family == .qwen3_6,
                  session.verifiedIdentity == nil,
                  session.sourceIdentity != nil else {
                throw ModelFamilyGenerationError.modelIdentityChanged
            }
            preflight = try await session.preflightLoadedSource(
                prompt: parsed.prompt, imagesByID: parsed.imagesByID,
                visionResidency: args.visionResidency)
        } else {
            preflight = try dependencies.preflightQwen(
                modelURL, parsed.prompt, parsed.imagesByID,
                visionPackURL, args.visionResidency, args.maxContext)
            session = try dependencies.loadSession(
                modelURL, args.maxContext, runtime, visionPackURL, .fullSha256)
            guard session.family == .qwen3_6,
                  session.verifiedIdentity != nil,
                  session.sourceIdentity == nil else {
                throw ModelFamilyGenerationError.modelIdentityChanged
            }
        }
        if args.prefillChunkTokensAuto, !args.quiet {
            stderr.write(Data(
                "[prefill chunk auto: verified Qwen uses native prefill]\n".utf8))
        }
        let config = GenerationConfig(
            maxNewTokens: min(args.maxNew, args.maxContext - preflight.promptTokens),
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        try config.validate()
        if args.showModelIdentity {
            if let identity = session.sourceIdentity {
                let line = sourceIdentityLine(
                    contentSHA256: identity.descriptorContentSHA256,
                    integrity: args.sourceIntegrity ?? .fullSHA256)
                stderr.write(Data(line.utf8))
            } else if let identity = session.verifiedIdentity {
                let line = "[model family=\(identity.family.rawValue) id=\(identity.modelID) "
                    + "revision=\(identity.sourceRevision) "
                    + "format=\(identity.formatMajor).\(identity.formatMinor)]\n"
                stderr.write(Data(line.utf8))
            } else {
                throw ModelFamilyGenerationError.modelIdentityChanged
            }
        }
        let result = try await session.generate(
            ModelFamilyGenerationRequest(
                prompt: parsed.prompt,
                imagesByID: parsed.imagesByID,
                visionResidency: args.visionResidency,
                config: config)
        ) { event in
            switch event {
            case .prefill:
                break
            case .text(let text):
                if !text.isEmpty { stdout.write(Data(text.utf8)) }
            case .toolCall(let call):
                stdout.write(Data((renderToolCall(call) + "\n").utf8))
            }
        }
        if !args.quiet {
            if let summary = result.cacheSummary {
                stderr.write(Data(routedExpertCacheFooter(summary).utf8))
            }
            let rate = result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            let footer = "\n[stop=\(String(describing: result.reason)) "
                + "prefill=\(result.promptTokens)tok new=\(result.newTokens)tok "
                + "decode=\(String(format: "%.2f", result.decodeSeconds))s "
                + "tok/s=\(String(format: "%.3f", rate))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch let error as ArgsError {
        return errored(stderr, "\(error)", 2)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

func sourceIdentityLine(contentSHA256: String, integrity: CLISourceIntegrityMode) -> String {
    "[model family=qwen3_6 backing=official-safetensors-bf16-v1 "
        + "content-sha256=\(contentSHA256) verification=\(integrity.rawValue)]\n"
}

private struct ParsedFamilyRequest {
    let prompt: ModelFamilyGenerationPrompt
    let imagesByID: [String: URL]
}

private func parseFamilyRequest(args: Args) throws -> ParsedFamilyRequest {
    if let raw = args.prompt {
        return ParsedFamilyRequest(prompt: .raw(raw), imagesByID: [:])
    }
    let tools = try args.toolsFile.map(parseToolsFile) ?? []
    let thinking: ModelFamilyThinkingMode = switch args.thinking {
    case .auto: .automatic
    case .on: .enabled
    case .off: .disabled
    }
    if let text = args.chatPrompt {
        var images: [String: URL] = [:]
        var parts: [ModelChatContentPart] = []
        for (offset, path) in args.images.enumerated() {
            let id = "cli-image-\(offset + 1)"
            images[id] = URL(fileURLWithPath: path)
            parts.append(.image(.init(id: id)))
        }
        if !text.isEmpty { parts.append(.text(text)) }
        let content: ModelChatContent = parts.isEmpty ? .text("") : .parts(parts)
        return ParsedFamilyRequest(
            prompt: .chat(
                messages: [.init(role: .user, content: content)],
                tools: tools,
                thinking: thinking),
            imagesByID: images)
    }
    guard let path = args.messagesFile else { throw ArgsError.modeMissing }
    let documentURL = URL(fileURLWithPath: path)
    let root = try OrderedModelJSON.parse(
        readCLIJSON(documentURL, flag: "--messages-file", maximumBytes: 4 * 1_024 * 1_024))
    guard case .array(let rows) = root else {
        throw invalidMessages("top level must be an array")
    }
    let base = documentURL.deletingLastPathComponent()
    var images: [String: URL] = [:]
    var nextImage = 1
    let messages = try rows.map { row -> ModelChatMessage in
        let members = try objectMembers(row, label: "message")
        let roleText = try requiredString("role", in: members)
        let role: ModelChatRole = switch roleText {
        case "system": .system
        case "developer": .developer
        case "user": .user
        case "assistant": .assistant
        case "tool": .tool
        default: throw invalidMessages("unknown role: \(roleText)")
        }
        let content: ModelChatContent?
        if let value = try optionalUnique("content", in: members) {
            switch value {
            case .null:
                content = nil
            case .string(let text):
                content = .text(text)
            case .array(let values):
                content = .parts(try values.map { value in
                    let part = try objectMembers(value, label: "content part")
                    let type = try requiredString("type", in: part)
                    switch type {
                    case "text":
                        return .text(try requiredString("text", in: part))
                    case "image_file":
                        guard role == .user else {
                            throw invalidMessages("image_file requires a user role")
                        }
                        let path = try requiredString("path", in: part)
                        let id = "messages-image-\(nextImage)"
                        nextImage += 1
                        images[id] = path.hasPrefix("/")
                            ? URL(fileURLWithPath: path)
                            : base.appendingPathComponent(path)
                        return .image(.init(id: id))
                    case "video_file":
                        throw invalidMessages("video input is not supported")
                    case "audio_file":
                        throw invalidMessages("audio input is not supported")
                    default:
                        throw invalidMessages("unknown content type: \(type)")
                    }
                })
            default:
                throw invalidMessages("content must be a string, array, or null")
            }
        } else {
            content = nil
        }
        let reasoning = try optionalString("reasoning_content", in: members)
        let toolCallID = try optionalString("tool_call_id", in: members)
        let name = try optionalString("name", in: members)
        let calls: [ModelChatToolCall]
        if let value = try optionalUnique("tool_calls", in: members) {
            guard case .array(let rows) = value else {
                throw invalidMessages("tool_calls must be an array")
            }
            calls = try rows.map(parseHistoricalToolCall)
        } else {
            calls = []
        }
        return ModelChatMessage(
            role: role, content: content,
            reasoningContent: reasoning, toolCalls: calls,
            toolCallID: toolCallID, name: name)
    }
    return ParsedFamilyRequest(
        prompt: .chat(messages: messages, tools: tools, thinking: thinking),
        imagesByID: images)
}

private func parseToolsFile(_ path: String) throws -> [ModelChatToolDefinition] {
    let value = try OrderedModelJSON.parse(readCLIJSON(
        URL(fileURLWithPath: path),
        flag: "--tools-file",
        maximumBytes: 4 * 1_024 * 1_024))
    guard case .array(let rows) = value else {
        throw ArgsError.invalidValue(flag: "--tools-file", value: "top level must be an array")
    }
    return try rows.map { row in
        let members = try objectMembers(
            row, label: "tool", flag: "--tools-file")
        let type = try requiredString(
            "type", in: members, flag: "--tools-file")
        guard type == "function" else {
            throw ArgsError.invalidValue(flag: "--tools-file", value: "tool type must be function")
        }
        let functionValue = try requiredUnique(
            "function", in: members, flag: "--tools-file")
        let function = try objectMembers(
            functionValue, label: "function", flag: "--tools-file")
        let name = try requiredString(
            "name", in: function, flag: "--tools-file")
        let description = try optionalString(
            "description", in: function, flag: "--tools-file") ?? ""
        let parameters = try requiredUnique(
            "parameters", in: function, flag: "--tools-file")
        guard case .object = parameters else {
            throw ArgsError.invalidValue(flag: "--tools-file", value: "parameters must be an object")
        }
        return ModelChatToolDefinition(
            type: type,
            function: .init(
                name: name, description: description, parameters: parameters))
    }
}

private func parseHistoricalToolCall(_ value: ModelChatJSONValue) throws -> ModelChatToolCall {
    let members = try objectMembers(value, label: "tool call")
    if let type = try optionalString("type", in: members), type != "function" {
        throw invalidMessages("tool call type must be function")
    }
    let function = try objectMembers(
        try requiredUnique("function", in: members), label: "tool call function")
    let rawArguments = try requiredUnique("arguments", in: function)
    let arguments: ModelChatJSONValue
    if case .string(let encoded) = rawArguments {
        arguments = try OrderedModelJSON.parse(Data(encoded.utf8))
    } else {
        arguments = rawArguments
    }
    guard case .object = arguments else {
        throw invalidMessages("tool call arguments must be an object")
    }
    return ModelChatToolCall(
        id: try optionalString("id", in: members),
        name: try requiredString("name", in: function),
        arguments: arguments)
}

private func renderToolCall(_ call: ParsedToolCall) -> String {
    "{\"tool_call\":{\"id\":\(jsonString(call.id)),"
        + "\"name\":\(jsonString(call.name)),"
        + "\"arguments\":\(call.argumentsJSON)}}"
}

private func jsonString(_ value: String) -> String {
    var output = "\""
    for scalar in value.unicodeScalars {
        switch scalar.value {
        case 0x08: output += "\\b"
        case 0x09: output += "\\t"
        case 0x0A: output += "\\n"
        case 0x0C: output += "\\f"
        case 0x0D: output += "\\r"
        case 0x22: output += "\\\""
        case 0x5C: output += "\\\\"
        case 0x00...0x1F:
            output += String(format: "\\u%04x", scalar.value)
        default:
            output.unicodeScalars.append(scalar)
        }
    }
    return output + "\""
}

private func invalidMessages(_ detail: String) -> ArgsError {
    invalidInput(flag: "--messages-file", detail)
}

private func invalidInput(flag: String, _ detail: String) -> ArgsError {
    .invalidValue(flag: flag, value: detail)
}

private func readCLIJSON(
    _ url: URL,
    flag: String,
    maximumBytes: UInt64
) throws -> Data {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
    guard size <= maximumBytes else {
        throw ArgsError.invalidValue(
            flag: flag,
            value: "file exceeds \(maximumBytes) bytes")
    }
    return try Data(contentsOf: url)
}

private func objectMembers(
    _ value: ModelChatJSONValue,
    label: String,
    flag: String = "--messages-file"
) throws -> [ModelChatJSONMember] {
    guard case .object(let members) = value else {
        throw invalidInput(flag: flag, "\(label) must be an object")
    }
    return members
}

private func optionalUnique(
    _ name: String,
    in members: [ModelChatJSONMember],
    flag: String = "--messages-file"
) throws -> ModelChatJSONValue? {
    let values = members.filter { $0.name == name }
    guard values.count <= 1 else {
        throw invalidInput(flag: flag, "duplicate key: \(name)")
    }
    return values.first?.value
}

private func requiredUnique(
    _ name: String,
    in members: [ModelChatJSONMember],
    flag: String = "--messages-file"
) throws -> ModelChatJSONValue {
    guard let value = try optionalUnique(name, in: members, flag: flag) else {
        throw invalidInput(flag: flag, "missing field: \(name)")
    }
    return value
}

private func requiredString(
    _ name: String,
    in members: [ModelChatJSONMember],
    flag: String = "--messages-file"
) throws -> String {
    guard case .string(let value) = try requiredUnique(
        name, in: members, flag: flag) else {
        throw invalidInput(flag: flag, "\(name) must be a string")
    }
    return value
}

private func optionalString(
    _ name: String,
    in members: [ModelChatJSONMember],
    flag: String = "--messages-file"
) throws -> String? {
    guard let raw = try optionalUnique(name, in: members, flag: flag) else { return nil }
    if case .null = raw { return nil }
    guard case .string(let value) = raw else {
        throw invalidInput(flag: flag, "\(name) must be a string or null")
    }
    return value
}

private enum OrderedModelJSON {
    enum ParseError: Error, CustomStringConvertible {
        case invalid(String)
        var description: String {
            switch self { case .invalid(let detail): "invalid JSON: \(detail)" }
        }
    }

    static func parse(_ data: Data) throws -> ModelChatJSONValue {
        do {
            _ = try JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed])
        } catch {
            throw ParseError.invalid("\(error)")
        }
        var parser = Parser(bytes: Array(data))
        let value = try parser.value()
        parser.whitespace()
        guard parser.index == parser.bytes.count else {
            throw ParseError.invalid("trailing bytes")
        }
        return value
    }

    struct Parser {
        let bytes: [UInt8]
        var index = 0

        mutating func value() throws -> ModelChatJSONValue {
            whitespace()
            guard index < bytes.count else { throw ParseError.invalid("unexpected end") }
            switch bytes[index] {
            case 0x7B: return try object()
            case 0x5B: return try array()
            case 0x22: return .string(try string())
            case 0x74: try literal("true"); return .bool(true)
            case 0x66: try literal("false"); return .bool(false)
            case 0x6E: try literal("null"); return .null
            case 0x2D, 0x30...0x39: return try number()
            default: throw ParseError.invalid("unexpected byte at \(index)")
            }
        }

        mutating func object() throws -> ModelChatJSONValue {
            index += 1
            whitespace()
            var members: [ModelChatJSONMember] = []
            if take(0x7D) { return .object(members) }
            while true {
                whitespace()
                guard index < bytes.count, bytes[index] == 0x22 else {
                    throw ParseError.invalid("object key is not a string")
                }
                let key = try string()
                whitespace()
                guard take(0x3A) else { throw ParseError.invalid("missing colon") }
                members.append(.init(key, try value()))
                whitespace()
                if take(0x7D) { return .object(members) }
                guard take(0x2C) else { throw ParseError.invalid("missing comma") }
            }
        }

        mutating func array() throws -> ModelChatJSONValue {
            index += 1
            whitespace()
            var values: [ModelChatJSONValue] = []
            if take(0x5D) { return .array(values) }
            while true {
                values.append(try value())
                whitespace()
                if take(0x5D) { return .array(values) }
                guard take(0x2C) else { throw ParseError.invalid("missing comma") }
            }
        }

        mutating func string() throws -> String {
            let start = index
            index += 1
            var escaped = false
            while index < bytes.count {
                let byte = bytes[index]
                index += 1
                if escaped { escaped = false; continue }
                if byte == 0x5C { escaped = true; continue }
                if byte == 0x22 {
                    let data = Data(bytes[start..<index])
                    do { return try JSONDecoder().decode(String.self, from: data) }
                    catch { throw ParseError.invalid("invalid string escape") }
                }
                guard byte >= 0x20 else {
                    throw ParseError.invalid("control byte in string")
                }
            }
            throw ParseError.invalid("unterminated string")
        }

        mutating func number() throws -> ModelChatJSONValue {
            let start = index
            while index < bytes.count,
                  "-+0123456789.eE".utf8.contains(bytes[index]) { index += 1 }
            let text = String(decoding: bytes[start..<index], as: UTF8.self)
            if !text.contains(".") && !text.contains("e") && !text.contains("E") {
                if let signed = Int64(text) { return .integer(signed) }
                if let unsigned = UInt64(text) { return .unsignedInteger(unsigned) }
            }
            guard let value = Double(text), value.isFinite else {
                throw ParseError.invalid("invalid number")
            }
            return .number(value)
        }

        mutating func literal(_ text: String) throws {
            let expected = Array(text.utf8)
            guard index + expected.count <= bytes.count,
                  Array(bytes[index..<(index + expected.count)]) == expected else {
                throw ParseError.invalid("invalid literal")
            }
            index += expected.count
        }

        mutating func whitespace() {
            while index < bytes.count,
                  bytes[index] == 0x20 || bytes[index] == 0x0A
                    || bytes[index] == 0x0D || bytes[index] == 0x09 { index += 1 }
        }

        mutating func take(_ byte: UInt8) -> Bool {
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
            return true
        }
    }
}

private func legacyRun(args: Args,
                showModelIdentity: Bool,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let input = try parseInput(args: args)
        var selectedDevice: MTLDevice?
        if input.hasImages {
            guard let device = MetalContext.makeSystemDefaultDevice() else {
                return errored(stderr, "no Metal device", 1)
            }
            try VisionRuntime.requireSupportedDevice(device)
            selectedDevice = device
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        var promptIds: [Int32]
        var multimodalMessages: [MultimodalMessage]?
        var imageURLs: [UUID: URL] = [:]
        switch input {
        case .raw(let text):
            multimodalMessages = nil
            promptIds = tokenizer.encode(text, addBOS: true)
        case .messages(let messages):
            multimodalMessages = nil
            promptIds = tokenizer.encode(
                try tokenizer.applyChatTemplate(messages), addBOS: false)
        case .multimodal(let messages, let images):
            multimodalMessages = messages
            imageURLs = images
            // Filled in after the tower runs; the renderer needs the projected
            // features to lay out the placeholder spans.
            promptIds = []
        }
        if multimodalMessages == nil {
            guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        }
        guard multimodalMessages != nil || promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        func makeConfig(maxNewTokens: Int) -> GenerationConfig {
            GenerationConfig(
                maxNewTokens: maxNewTokens,
                temperature: args.temperature,
                topK: args.topK,
                topP: args.topP,
                repetitionPenalty: args.repetitionPenalty,
                seed: args.seed,
                stopStrings: args.stops,
                extraStopTokens: [])
        }
        // Hoisted above the `auto` estimate, which needs a device to read image
        // geometry. Planning never touches the GPU, but building the plan does
        // need the device the run will use.
        guard let device = selectedDevice ?? MetalContext.makeSystemDefaultDevice() else {
            return errored(stderr, "no Metal device", 1)
        }

        var effectiveArgs = args
        if args.prefillChunkTokensAuto {
            let promptTokens = try estimatedPromptTokens(
                input: input, tokenizer: tokenizer, device: device)
            effectiveArgs.prefillChunkTokens =
                PrefillRuntimeConfig.autoChunkTokens(promptTokens: promptTokens)
            if !args.quiet {
                let line = "[prefill chunk auto: "
                    + "\(effectiveArgs.prefillChunkTokens) tokens for a "
                    + "\(promptTokens)-token prompt]\n"
                stderr.write(Data(line.utf8))
            }
        }
        let args = effectiveArgs
        let runtime = try args.resolvedRuntimeConfiguration(
            forceLogitsHead: !makeConfig(maxNewTokens: args.maxNew).isPureGreedy,
            imagePrompt: input.hasImages)

        if !args.quiet,
           let notice = prefillCoercionNotice(
            hasImages: input.hasImages, config: runtime.prefillConfig) {
            stderr.write(Data((notice + "\n").utf8))
        }

        // An image prompt that cannot fit is decided by geometry, so decide it
        // before spending a model load and a GPU encode on it. The plan reads
        // metadata only, and the same plan is reused at encode time so each
        // image is opened and parsed once.
        var imagePlans: [UUID: VisionImagePlan] = [:]
        if let multimodalMessages {
            // Dictionary order depends on a per-process hash seed, so planning
            // straight from `imageURLs` named a different image on each run when
            // two were bad. The prompt's own part order is the stable one.
            let ordered = orderedImageIDs(messages: multimodalMessages, images: imageURLs)
            // The device from the guard above, not a fresh `MetalContext`:
            // building one compiles every shader module, and planning never
            // touches the GPU.
            let preprocessor = Gemma4ImagePreprocessor(device: device, config: VisionConfig())
            var projected = 0
            for id in ordered {
                guard let url = imageURLs[id] else { continue }
                let plan = try preprocessor.plan(fileURL: url)
                imagePlans[id] = plan
                projected += plan.geometry.softTokenCount
                    + VisionImageTokenBudget.markerTokensPerImage
            }
            // The text counts too. The tokenizer is already loaded, so this is
            // free and it closes the common case: a prompt whose text overflows
            // used to be refused only after the pack opened and every image had
            // been encoded. A lower bound - the template's framing tokens are
            // not counted - so it can only refuse what could never fit.
            for message in multimodalMessages {
                for part in message.content {
                    if case .text(let text) = part {
                        projected += tokenizer.encode(text, addBOS: false).count
                    }
                }
            }
            guard projected < args.maxContext else {
                return errored(
                    stderr,
                    "context overflow: prompt needs at least \(projected) tokens, "
                        + "which reaches maxContext \(args.maxContext)",
                    2)
            }
        }

        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: .fullSha256)
        if showModelIdentity {
            stderr.write(Data(
                "[model family=gemma4 format=1 legacy-unverified]\n".utf8))
        }
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        let multimodalInput: MultimodalPrefillInput?
        if let multimodalMessages {
            let vision = try VisionRuntime.open(
                textModelURL: modelURL,
                context: context,
                visionPackURL: args.visionPack.map { URL(fileURLWithPath: $0) })
            var features: [UUID: VisionFeatures] = [:]
            features.reserveCapacity(imageURLs.count)
            let visionStarted = ContinuousClock.now
            for id in orderedImageIDs(messages: multimodalMessages, images: imageURLs) {
                guard let url = imageURLs[id] else { continue }
                try Task.checkCancellation()
                if let plan = imagePlans[id] {
                    features[id] = try vision.encodeImage(
                        plan: plan,
                        languageModel: model,
                        residencyPolicy: args.visionResidency,
                        checkCancellation: { try Task.checkCancellation() })
                } else {
                    features[id] = try vision.encodeImage(
                        at: url,
                        languageModel: model,
                        residencyPolicy: args.visionResidency,
                        checkCancellation: { try Task.checkCancellation() })
                }
            }
            if !args.quiet {
                let duration = visionStarted.duration(to: .now)
                let seconds = Double(duration.components.seconds)
                    + Double(duration.components.attoseconds) / 1e18
                stderr.write(Data(
                    "[vision images=\(imageURLs.count) encode=\(String(format: "%.3f", seconds))s]\n".utf8))
            }
            let rendered = try MultimodalPromptRenderer.render(
                messages: multimodalMessages,
                featuresByID: features,
                tokenizer: tokenizer)
            promptIds = rendered.effectiveTokenIDs
            multimodalInput = rendered
            guard promptIds.count < args.maxContext else {
                return errored(
                    stderr,
                    "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                    2)
            }
        } else {
            multimodalInput = nil
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = makeConfig(maxNewTokens: effectiveMaxNew)

        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            multimodalInput: multimodalInput,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if !args.quiet {
            stderr.write(Data(routedExpertCacheFooter(model.routedExpertCacheSummary).utf8))
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch let error as ArgsError {
        return errored(stderr, "\(error)", 2)
    } catch let error as VisionImageError {
        return errored(stderr, "\(error)", 3)
    } catch let error as VisionPackError {
        return errored(stderr, "\(error)", 4)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}


/// The order the messages reference their images, so a repeated command plans
/// and encodes them the same way every run.
func orderedImageIDs(messages: [MultimodalMessage],
                     images: [UUID: URL]) -> [UUID] {
    var ordered: [UUID] = []
    var seen: Set<UUID> = []
    for message in messages {
        for part in message.content {
            if case .image(let id) = part, images[id] != nil, seen.insert(id).inserted {
                ordered.append(id)
            }
        }
    }
    for id in images.keys.sorted(by: { $0.uuidString < $1.uuidString })
    where !seen.contains(id) {
        ordered.append(id)
    }
    return ordered
}

/// The line to print when an image prompt overrides the requested prefill mode.
///
/// The override itself happens deep in `RawCompletion`, silently, while
/// `RUNTIME_CONTROLS.md` says `--prefill off` disables that path. A flag that is
/// ignored without a word is worse than one that is refused, so the CLI says it
/// before the model loads.

/// A prompt-length estimate good enough to choose a chunk size.
///
/// Deliberately cheap: it tokenises text and reads image geometry, and never
/// loads the model. Being a little low only costs one extra chunk.
func estimatedPromptTokens(
    input: PromptInput,
    tokenizer: GFTokenizer,
    device: MTLDevice
) throws -> Int {
    switch input {
    case .raw(let text):
        return tokenizer.encode(text, addBOS: true).count
    case .messages(let messages):
        return tokenizer.encode(try tokenizer.applyChatTemplate(messages),
                                addBOS: false).count
    case .multimodal(let messages, let images):
        var total = 0
        for message in messages {
            for part in message.content {
                if case .text(let text) = part {
                    total += tokenizer.encode(text, addBOS: false).count
                }
            }
        }
        let preprocessor = Gemma4ImagePreprocessor(
            device: device, config: VisionConfig())
        for url in images.values {
            // Not `try?`: an image that cannot be planned would count as zero
            // soft tokens, so `auto` would size the chunk for the text alone and
            // put the multimodal prompt back on the many-chunk path this flag
            // exists to avoid. An unplannable image fails the run anyway, and
            // failing here says why, before the load.
            total += try preprocessor.plan(fileURL: url).geometry.softTokenCount
            total += VisionImageTokenBudget.markerTokensPerImage
        }
        return total
    }
}

func prefillCoercionNotice(
    hasImages: Bool, config: PrefillRuntimeConfig
) -> String? {
    guard hasImages, config.coercedForImagePrompt() != nil else { return nil }
    return "[prefill coerced to chunked: image prompts require it]"
}

func parseInput(args: Args) throws -> PromptInput {
    if let prompt = args.prompt {
        return .raw(prompt)
    }
    if let chatPrompt = args.chatPrompt {
        guard !args.images.isEmpty else {
            return .messages([GFTokenizer.Message(role: .user, content: chatPrompt)])
        }
        var images: [UUID: URL] = [:]
        var content: [MultimodalContentPart] = []
        for path in args.images {
            let id = UUID()
            images[id] = URL(fileURLWithPath: path)
            content.append(.image(id: id))
        }
        if !chatPrompt.isEmpty { content.append(.text(chatPrompt)) }
        return .multimodal(
            messages: [MultimodalMessage(role: .user, content: content)],
            images: images)
    }
    guard let path = args.messagesFile else {
        throw ArgsError.modeMissing
    }
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
    let base = url.deletingLastPathComponent()
    var images: [UUID: URL] = [:]
    let messages: [MultimodalMessage] = try rows.map { row in
        guard let role = GFTokenizer.Role(rawValue: row.role) else {
            throw ArgsError.invalidValue(flag: "--messages-file", value: "unknown role: \(row.role)")
        }
        let parts: [MultimodalContentPart]
        switch row.content {
        case .text(let text):
            parts = [.text(text)]
        case .parts(let rows):
            parts = try rows.map { part in
                switch part.type {
                case "text":
                    guard let text = part.text else {
                        throw ArgsError.invalidValue(
                            flag: "--messages-file", value: "text part is missing text")
                    }
                    return .text(text)
                case "image_file":
                    guard role == .user, let path = part.path else {
                        throw ArgsError.invalidValue(
                            flag: "--messages-file", value: "image_file requires a user role and path")
                    }
                    let id = UUID()
                    // Test the string, not a URL: `URL(fileURLWithPath:)`
                    // resolves against the process directory, so its `path` is
                    // always absolute and the relative branch never ran. A
                    // relative path belongs to the messages file's directory.
                    images[id] = path.hasPrefix("/")
                        ? URL(fileURLWithPath: path)
                        : base.appendingPathComponent(path)
                    return .image(id: id)
                default:
                    throw ArgsError.invalidValue(
                        flag: "--messages-file", value: "unknown content type: \(part.type)")
                }
            }
        }
        return MultimodalMessage(role: role, content: parts)
    }
    if images.isEmpty {
        return .messages(messages.map { message in
            GFTokenizer.Message(
                role: message.role,
                content: message.content.compactMap {
                    if case .text(let text) = $0 { text } else { nil }
                }.joined())
        })
    }
    return .multimodal(messages: messages, images: images)
}

/// An image prompt's token count follows from geometry, so an oversized one can
/// be refused in milliseconds instead of after a model load and a GPU encode.
func validatePromptSize(
    tokens: Int,
    args: Args,
    stderr: FileHandle
) -> RunResult? {
    guard tokens + args.maxNew <= args.maxContext else {
        return errored(
            stderr,
            "context overflow: prompt \(tokens) + maxNew \(args.maxNew) exceeds maxContext \(args.maxContext)",
            2)
    }
    return nil
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
