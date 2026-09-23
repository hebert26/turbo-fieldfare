import Foundation

public enum QwenChatCodecError: Error, Equatable, CustomStringConvertible {
    case noMessages
    case noUserQuery
    case invalidContinuation
    case systemMessageMustBeFirst
    case systemMessageContainsImage
    case unexpectedRole(ModelChatRole)
    case unexpectedContentPart(String)
    case unsupportedAudio
    case unsupportedVideo
    case toolArgumentsMustBeObject
    case duplicateJSONKey(String)
    case nonFiniteJSONNumber

    public var description: String {
        switch self {
        case .noMessages:
            return "Cannot apply chat template to an empty conversation. Provide at least one message."
        case .noUserQuery: return "No user query found in messages."
        case .invalidContinuation:
            return "A continuation must be one user message or one or more tool-result messages."
        case .systemMessageMustBeFirst: return "System message must be at the beginning."
        case .systemMessageContainsImage: return "System message cannot contain images."
        case .unexpectedRole: return "Unexpected message role."
        case .unexpectedContentPart: return "Unexpected item type in content."
        case .unsupportedAudio: return "Structured audio content is unsupported in Phase 14."
        case .unsupportedVideo: return "Structured video content is unsupported in Phase 14."
        case .toolArgumentsMustBeObject: return "Can only get item pairs from a mapping."
        case .duplicateJSONKey(let key): return "Duplicate JSON object key: \(key)"
        case .nonFiniteJSONNumber: return "JSON numbers must be finite."
        }
    }
}

/// Exact retained assistant boundary immediately before a new user or grouped
/// tool-response turn.
public enum QwenChatContinuationBoundary: Sendable, Equatable {
    /// The retained final token is `<|im_end|>`; only the template newline is
    /// missing.
    case endedWithEndToken
    /// A max-budget, Stop, or non-EOS token stop left assistant content open.
    case openAssistant
}

/// Compiled Swift implementation of the chat template embedded in the pinned
/// Qwen tokenizer config. This type never evaluates Jinja at runtime.
public struct QwenChatCodec: ModelChatCodec, Sendable {
    public let tokenizer: QwenTokenizer

    public init(tokenizer: QwenTokenizer) {
        self.tokenizer = tokenizer
    }

    public func encodePrompt(
        messages: [ModelChatMessage],
        tools: [ModelChatToolDefinition],
        options: ModelChatRenderOptions
    ) throws -> [Int32] {
        tokenizer.encode(try renderPrompt(messages: messages, tools: tools, options: options))
    }

    /// Encodes only the next retained conversation suffix. The preceding
    /// assistant turn, including its exact generated tokens and EOS, already
    /// lives in the conversation state and must not be rendered again.
    public func encodeContinuation(
        messages: [ModelChatMessage],
        boundary: QwenChatContinuationBoundary = .endedWithEndToken,
        options: ModelChatRenderOptions = .init()
    ) throws -> [Int32] {
        tokenizer.encode(try renderContinuation(
            messages: messages, boundary: boundary, options: options))
    }

    public func renderContinuation(
        messages: [ModelChatMessage],
        boundary: QwenChatContinuationBoundary = .endedWithEndToken,
        options: ModelChatRenderOptions = .init()
    ) throws -> String {
        guard !messages.isEmpty else { throw QwenChatCodecError.noMessages }
        guard messages.allSatisfy({
            $0.reasoningContent == nil && $0.toolCalls.isEmpty
        }) else {
            throw QwenChatCodecError.invalidContinuation
        }

        var state = RenderState(options: options)
        var output = switch boundary {
        case .endedWithEndToken: "\n"
        case .openAssistant: "<|im_end|>\n"
        }
        if messages.count == 1, messages[0].role == .user {
            let content = try state.renderContent(
                messages[0].content, countVision: true, isSystemContent: false)
                .qwenTrimmed
            output += "<|im_start|>user\n" + content + "<|im_end|>\n"
        } else if messages.allSatisfy({ $0.role == .tool }) {
            output += "<|im_start|>user"
            for message in messages {
                let content = try state.renderContent(
                    message.content, countVision: true, isSystemContent: false)
                    .qwenTrimmed
                output += "\n<tool_response>\n" + content + "\n</tool_response>"
            }
            output += "<|im_end|>\n"
        } else {
            throw QwenChatCodecError.invalidContinuation
        }

        if options.addGenerationPrompt {
            output += "<|im_start|>assistant\n"
            output += options.enableThinking
                ? "<think>\n"
                : "<think>\n\n</think>\n\n"
        }
        return output
    }

    public func renderPrompt(
        messages: [ModelChatMessage],
        tools: [ModelChatToolDefinition] = [],
        options: ModelChatRenderOptions = .init()
    ) throws -> String {
        guard !messages.isEmpty else { throw QwenChatCodecError.noMessages }

        var state = RenderState(options: options)
        var output = ""
        if !tools.isEmpty {
            output += "<|im_start|>system\n"
            output += Self.toolIntroduction
            for tool in tools {
                output += "\n" + (try Self.renderToolJSON(tool))
            }
            output += "\n</tools>"
            output += Self.toolInstructions
            if messages[0].role == .system {
                let system = try state.renderContent(
                    messages[0].content, countVision: false, isSystemContent: true)
                    .qwenTrimmed
                if !system.isEmpty { output += "\n\n" + system }
            }
            output += "<|im_end|>\n"
        } else if messages[0].role == .system {
            let system = try state.renderContent(
                messages[0].content, countVision: false, isSystemContent: true)
                .qwenTrimmed
            output += "<|im_start|>system\n" + system + "<|im_end|>\n"
        }

        var lastQueryIndex = messages.count - 1
        var needsRealQuery = true
        for index in messages.indices.reversed() where needsRealQuery {
            guard messages[index].role == .user else { continue }
            let content = try state.renderContent(
                messages[index].content, countVision: false, isSystemContent: false)
                .qwenTrimmed
            if !(content.hasPrefix("<tool_response>")
                && content.hasSuffix("</tool_response>")) {
                needsRealQuery = false
                lastQueryIndex = index
            }
        }
        guard !needsRealQuery else { throw QwenChatCodecError.noUserQuery }

        for index in messages.indices {
            let message = messages[index]
            var content = try state.renderContent(
                message.content, countVision: true, isSystemContent: false)
                .qwenTrimmed

            switch message.role {
            case .system:
                guard index == messages.startIndex else {
                    throw QwenChatCodecError.systemMessageMustBeFirst
                }
                // The leading system branch rendered this message already.
            case .user:
                output += "<|im_start|>user\n" + content + "<|im_end|>\n"
            case .assistant:
                let reasoning: String
                if let explicitReasoning = message.reasoningContent {
                    reasoning = explicitReasoning.qwenTrimmed
                } else if content.contains("</think>") {
                    let beforeFirstEnd = content.components(separatedBy: "</think>").first ?? ""
                    reasoning = (beforeFirstEnd.components(separatedBy: "<think>").last ?? "")
                        .trimmingOnlyNewlinesAtStartAndEnd
                        .qwenTrimmed
                    content = (content.components(separatedBy: "</think>").last ?? "")
                        .trimmingOnlyNewlinesAtStart
                } else {
                    reasoning = ""
                }

                if options.preserveThinking || index > lastQueryIndex {
                    output += "<|im_start|>assistant\n<think>\n"
                        + reasoning + "\n</think>\n\n" + content
                } else {
                    output += "<|im_start|>assistant\n" + content
                }
                output += try Self.renderToolCalls(message.toolCalls, content: content)
                output += "<|im_end|>\n"
            case .tool:
                if index > messages.startIndex, messages[index - 1].role != .tool {
                    output += "<|im_start|>user"
                }
                output += "\n<tool_response>\n" + content + "\n</tool_response>"
                if index == messages.index(before: messages.endIndex)
                    || messages[index + 1].role != .tool {
                    output += "<|im_end|>\n"
                }
            case .developer, .other:
                throw QwenChatCodecError.unexpectedRole(message.role)
            }
        }

        if options.addGenerationPrompt {
            output += "<|im_start|>assistant\n"
            output += options.enableThinking
                ? "<think>\n"
                : "<think>\n\n</think>\n\n"
        }
        return output
    }

    private static func renderToolJSON(_ tool: ModelChatToolDefinition) throws -> String {
        try renderJSON(.object([
            .init("type", .string(tool.type)),
            .init("function", .object([
                .init("name", .string(tool.function.name)),
                .init("description", .string(tool.function.description)),
                .init("parameters", tool.function.parameters),
            ])),
        ]))
    }

    private static func renderToolCalls(
        _ calls: [ModelChatToolCall],
        content: String
    ) throws -> String {
        var output = ""
        for (index, call) in calls.enumerated() {
            if index == 0 {
                if !content.qwenTrimmed.isEmpty { output += "\n\n" }
            } else {
                output += "\n"
            }
            output += "<tool_call>\n<function=\(call.name)>\n"
            guard case .object(let arguments) = call.arguments else {
                throw QwenChatCodecError.toolArgumentsMustBeObject
            }
            try validateUnique(arguments)
            for argument in arguments {
                output += "<parameter=\(argument.name)>\n"
                if case .string(let raw) = argument.value {
                    output += raw
                } else {
                    output += try renderJSON(argument.value)
                }
                output += "\n</parameter>\n"
            }
            output += "</function>\n</tool_call>"
        }
        return output
    }

    private static func renderJSON(_ value: ModelChatJSONValue) throws -> String {
        switch value {
        case .object(let members):
            try validateUnique(members)
            return "{" + (try members.map {
                try renderJSONString($0.name) + ": " + renderJSON($0.value)
            }.joined(separator: ", ")) + "}"
        case .array(let values):
            return "[" + (try values.map(renderJSON).joined(separator: ", ")) + "]"
        case .string(let value): return renderJSONString(value)
        case .integer(let value): return String(value)
        case .unsignedInteger(let value): return String(value)
        case .number(let value):
            guard value.isFinite else { throw QwenChatCodecError.nonFiniteJSONNumber }
            return renderPythonJSONNumber(value)
        case .bool(let value): return value ? "true" : "false"
        case .null: return "null"
        }
    }

    /// Matches Python's shortest-round-trip float spelling and its fixed-vs-
    /// scientific threshold used by `json.dumps`. Swift already supplies the
    /// same shortest significant digits, but can select scientific notation
    /// for decimal exponent 15 where Python keeps fixed notation.
    private static func renderPythonJSONNumber(_ value: Double) -> String {
        let rendered = String(value)
        guard let exponentMarker = rendered.firstIndex(of: "e"),
              let exponent = Int(rendered[rendered.index(after: exponentMarker)...]),
              (-4..<16).contains(exponent) else {
            return rendered
        }

        let isNegative = rendered.first == "-"
        let coefficientStart = isNegative ? rendered.index(after: rendered.startIndex) : rendered.startIndex
        let digits = String(rendered[coefficientStart..<exponentMarker].filter { $0 != "." })
        let decimalOffset = exponent + 1
        let sign = isNegative ? "-" : ""
        if decimalOffset <= 0 {
            return sign + "0." + String(repeating: "0", count: -decimalOffset) + digits
        }
        if decimalOffset >= digits.count {
            return sign + digits
                + String(repeating: "0", count: decimalOffset - digits.count) + ".0"
        }
        let decimalIndex = digits.index(digits.startIndex, offsetBy: decimalOffset)
        return sign + String(digits[..<decimalIndex]) + "." + String(digits[decimalIndex...])
    }

    private static func validateUnique(_ members: [ModelChatJSONMember]) throws {
        var keys: Set<String> = []
        for member in members where !keys.insert(member.name).inserted {
            throw QwenChatCodecError.duplicateJSONKey(member.name)
        }
    }

    /// Python `json.dumps(..., ensure_ascii=False)` string escaping used by
    /// Transformers' pinned `tojson` override. It deliberately does not HTML-
    /// escape `<`, `>`, `&`, or apostrophes.
    private static func renderJSONString(_ value: String) -> String {
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

    private struct RenderState {
        let options: ModelChatRenderOptions
        var imageCount = 0

        mutating func renderContent(
            _ content: ModelChatContent?,
            countVision: Bool,
            isSystemContent: Bool
        ) throws -> String {
            guard let content else { return "" }
            switch content {
            case .text(let text):
                return text
            case .parts(let parts):
                var output = ""
                for part in parts {
                    switch part {
                    case .text(let text):
                        output += text
                    case .image:
                        if isSystemContent {
                            throw QwenChatCodecError.systemMessageContainsImage
                        }
                        if countVision { imageCount += 1 }
                        if options.addVisionID { output += "Picture \(imageCount): " }
                        output += "<|vision_start|><|image_pad|><|vision_end|>"
                    case .video:
                        throw QwenChatCodecError.unsupportedVideo
                    case .audio:
                        throw QwenChatCodecError.unsupportedAudio
                    case .unsupported(let type):
                        throw QwenChatCodecError.unexpectedContentPart(type)
                    }
                }
                return output
            }
        }
    }

    private static let toolIntroduction = """
    # Tools

    You have access to the following functions:

    <tools>
    """

    private static let toolInstructions = """


    If you choose to call a function ONLY reply in the following format with NO suffix:

    <tool_call>
    <function=example_function_name>
    <parameter=example_parameter_1>
    value_1
    </parameter>
    <parameter=example_parameter_2>
    This is the value for the second parameter
    that can span
    multiple lines
    </parameter>
    </function>
    </tool_call>

    <IMPORTANT>
    Reminder:
    - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
    - Required parameters MUST be specified
    - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
    - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
    </IMPORTANT>
    """
}

private extension String {
    var qwenTrimmed: String {
        let scalars = unicodeScalars
        var lowerBound = scalars.startIndex
        while lowerBound != scalars.endIndex,
              scalars[lowerBound].isPythonStripScalar {
            scalars.formIndex(after: &lowerBound)
        }

        var upperBound = scalars.endIndex
        while upperBound != lowerBound {
            let candidate = scalars.index(before: upperBound)
            guard scalars[candidate].isPythonStripScalar else { break }
            upperBound = candidate
        }
        return String(scalars[lowerBound..<upperBound])
    }

    var trimmingOnlyNewlinesAtStart: String {
        var value = self
        while value.first == "\n" { value.removeFirst() }
        return value
    }

    var trimmingOnlyNewlinesAtStartAndEnd: String {
        var value = trimmingOnlyNewlinesAtStart
        while value.last == "\n" { value.removeLast() }
        return value
    }
}

private extension Unicode.Scalar {
    var isPythonStripScalar: Bool {
        switch value {
        case 0x0009...0x000D,
             0x001C...0x001F,
             0x0020,
             0x0085,
             0x00A0,
             0x1680,
             0x2000...0x200A,
             0x2028...0x2029,
             0x202F,
             0x205F,
             0x3000:
            true
        default:
            false
        }
    }
}
