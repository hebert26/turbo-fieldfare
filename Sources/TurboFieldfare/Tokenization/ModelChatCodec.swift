import Foundation

/// Model-neutral input accepted by chat prompt codecs.
///
/// Object members are stored in an array rather than a dictionary because the
/// pinned Qwen template preserves host insertion order when serializing tool
/// schemas and arguments.
public indirect enum ModelChatJSONValue: Equatable, Sendable {
    case object([ModelChatJSONMember])
    case array([ModelChatJSONValue])
    case string(String)
    case integer(Int64)
    case unsignedInteger(UInt64)
    case number(Double)
    case bool(Bool)
    case null
}

public struct ModelChatJSONMember: Equatable, Sendable {
    public let name: String
    public let value: ModelChatJSONValue

    public init(_ name: String, _ value: ModelChatJSONValue) {
        self.name = name
        self.value = value
    }
}

public enum ModelChatRole: Equatable, Sendable {
    case system
    case developer
    case user
    case assistant
    case tool
    /// Retains an unrecognized host role so a codec can reject it at the same
    /// precedence point as its authoritative template.
    case other(String)
}

public struct ModelChatMedia: Equatable, Sendable {
    /// Host identity retained for application bookkeeping. Prompt renderers do
    /// not serialize it; model-visible media is represented by pinned markers.
    public let id: String?

    public init(id: String? = nil) {
        self.id = id
    }
}

public enum ModelChatContentPart: Equatable, Sendable {
    case text(String)
    case image(ModelChatMedia = .init())
    case video(ModelChatMedia = .init())
    case audio(ModelChatMedia = .init())
    case unsupported(type: String)
}

public enum ModelChatContent: Equatable, Sendable {
    case text(String)
    case parts([ModelChatContentPart])
}

public struct ModelChatToolCall: Equatable, Sendable {
    /// Host call identity is retained but deliberately absent from the pinned
    /// Qwen serialization, which writes only function name and arguments.
    public let id: String?
    public let name: String
    public let arguments: ModelChatJSONValue

    public init(id: String? = nil, name: String, arguments: ModelChatJSONValue) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

public struct ModelChatMessage: Equatable, Sendable {
    public let role: ModelChatRole
    public let content: ModelChatContent?
    public let reasoningContent: String?
    public let toolCalls: [ModelChatToolCall]
    /// Retained host metadata. The pinned Qwen template ignores both fields.
    public let toolCallID: String?
    public let name: String?

    public init(
        role: ModelChatRole,
        content: ModelChatContent?,
        reasoningContent: String? = nil,
        toolCalls: [ModelChatToolCall] = [],
        toolCallID: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }

    public init(
        role: ModelChatRole,
        content: String,
        reasoningContent: String? = nil,
        toolCalls: [ModelChatToolCall] = [],
        toolCallID: String? = nil,
        name: String? = nil
    ) {
        self.init(
            role: role,
            content: .text(content),
            reasoningContent: reasoningContent,
            toolCalls: toolCalls,
            toolCallID: toolCallID,
            name: name)
    }
}

public struct ModelChatFunctionDefinition: Equatable, Sendable {
    public let name: String
    public let description: String
    public let parameters: ModelChatJSONValue

    public init(name: String, description: String, parameters: ModelChatJSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public struct ModelChatToolDefinition: Equatable, Sendable {
    public let type: String
    public let function: ModelChatFunctionDefinition

    public init(type: String = "function", function: ModelChatFunctionDefinition) {
        self.type = type
        self.function = function
    }
}

public struct ModelChatRenderOptions: Equatable, Sendable {
    public var addGenerationPrompt: Bool
    public var enableThinking: Bool
    public var preserveThinking: Bool
    public var addVisionID: Bool

    public init(
        addGenerationPrompt: Bool = true,
        enableThinking: Bool = true,
        preserveThinking: Bool = false,
        addVisionID: Bool = false
    ) {
        self.addGenerationPrompt = addGenerationPrompt
        self.enableThinking = enableThinking
        self.preserveThinking = preserveThinking
        self.addVisionID = addVisionID
    }
}

/// The narrow model-neutral boundary used to produce prompt token IDs.
/// Concrete codecs may also expose a rendered-text method for parity testing.
public protocol ModelChatCodec: Sendable {
    func encodePrompt(
        messages: [ModelChatMessage],
        tools: [ModelChatToolDefinition],
        options: ModelChatRenderOptions
    ) throws -> [Int32]
}

public extension ModelChatCodec {
    func encodePrompt(messages: [ModelChatMessage]) throws -> [Int32] {
        try encodePrompt(messages: messages, tools: [], options: .init())
    }

    func encodePrompt(
        messages: [ModelChatMessage],
        tools: [ModelChatToolDefinition]
    ) throws -> [Int32] {
        try encodePrompt(messages: messages, tools: tools, options: .init())
    }
}

public enum GemmaChatCodecAdapterError: Error, Equatable, CustomStringConvertible {
    case unsupportedStructuredContent
    case unsupportedRole(ModelChatRole)
    case invalidToolArguments

    public var description: String {
        switch self {
        case .unsupportedStructuredContent:
            return "Gemma chat adapter requires text-only logical history"
        case .unsupportedRole(let role):
            return "Gemma chat adapter does not support role: \(role)"
        case .invalidToolArguments:
            return "Gemma tool-call arguments must be a JSON object"
        }
    }
}

/// Additive adapter around `GFTokenizer`. It converts model-neutral values and
/// delegates to the existing Gemma render/encode methods without changing
/// Gemma defaults, prompt bytes, or tokenizer behavior.
public struct GemmaChatCodecAdapter: ModelChatCodec {
    public let tokenizer: GFTokenizer

    public init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer
    }

    public func encodePrompt(
        messages: [ModelChatMessage],
        tools: [ModelChatToolDefinition],
        options: ModelChatRenderOptions
    ) throws -> [Int32] {
        let gemmaMessages = try messages.map(Self.gemmaMessage)
        if tools.isEmpty {
            return tokenizer.encode(
                try tokenizer.applyChatTemplate(gemmaMessages), addBOS: false)
        }

        let gemmaTools = try tools.map { tool in
            GFTokenizer.FunctionDefinition(
                name: tool.function.name,
                description: tool.function.description,
                parameters: try Self.jsonValue(tool.function.parameters))
        }
        return try tokenizer
            .withToolThinking(enabled: options.enableThinking)
            .encodeToolChat(messages: gemmaMessages, tools: gemmaTools)
    }

    private static func gemmaMessage(_ message: ModelChatMessage) throws -> GFTokenizer.Message {
        let role: GFTokenizer.Role
        switch message.role {
        case .system: role = .system
        case .developer: role = .developer
        case .user: role = .user
        case .assistant: role = .assistant
        case .tool: role = .tool
        case .other: throw GemmaChatCodecAdapterError.unsupportedRole(message.role)
        }

        let content: String?
        switch message.content {
        case .none:
            content = nil
        case .text(let text):
            content = text
        case .parts:
            throw GemmaChatCodecAdapterError.unsupportedStructuredContent
        }

        let calls = try message.toolCalls.map { call in
            guard case .object = call.arguments else {
                throw GemmaChatCodecAdapterError.invalidToolArguments
            }
            return GFTokenizer.HistoricalToolCall(
                id: call.id ?? "",
                name: call.name,
                arguments: try jsonValue(call.arguments))
        }
        return GFTokenizer.Message(
            role: role,
            content: content,
            toolCalls: calls,
            toolCallID: message.toolCallID,
            name: message.name)
    }

    private static func jsonValue(_ value: ModelChatJSONValue) throws -> JSONValue {
        switch value {
        case .object(let members):
            var object: [String: JSONValue] = [:]
            object.reserveCapacity(members.count)
            for member in members {
                guard object[member.name] == nil else {
                    throw GemmaChatCodecAdapterError.invalidToolArguments
                }
                object[member.name] = try jsonValue(member.value)
            }
            return .object(object)
        case .array(let values): return .array(try values.map(jsonValue))
        case .string(let value): return .string(value)
        case .integer(let value): return .integer(value)
        case .unsignedInteger(let value): return .unsignedInteger(value)
        case .number(let value): return .number(value)
        case .bool(let value): return .bool(value)
        case .null: return .null
        }
    }
}
