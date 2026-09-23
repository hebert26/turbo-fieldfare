import Foundation
import TurboFieldfare

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(message: String, param: String? = nil, code: String,
                type: String = "invalid_request_error") {
        error = Detail(message: message,
                       type: type,
                       param: param,
                       code: code)
    }
}

public struct OpenAIImageURL: Codable, Equatable, Sendable {
    public let url: String
    public let detail: String?
}

public struct OpenAIContentPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
    public let imageURL: OpenAIImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }
}

public typealias OpenAITextPart = OpenAIContentPart

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAIContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAIContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?
    public let reasoningContent: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
        case reasoningContent = "reasoning_content"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    /// Family-neutral switch used by verified Qwen. Gemma accepts only auto.
    public let thinking: String?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case thinking
        case streamOptions = "stream_options"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
    }

    public init(promptTokens: Int,
                completionTokens: Int,
                totalTokens: Int,
                cachedTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
    }
}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        public let created: Int
        public let ownedBy: String
        public let capabilities: [String]?
        public let family: String?
        public let revision: String?

        enum CodingKeys: String, CodingKey {
            case id, object, created, capabilities, family, revision
            case ownedBy = "owned_by"
        }

        public init(id: String, object: String, created: Int,
                    ownedBy: String, capabilities: [String]? = nil,
                    family: String? = nil, revision: String? = nil) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
            self.capabilities = capabilities
            self.family = family
            self.revision = revision
        }
    }

    public let object: String
    public let data: [Model]
}

public enum ServerRequestError: Error, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    case unknownModel
    case queueFull

    public var envelope: OpenAIErrorEnvelope {
        switch self {
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(message: "requested model is not available",
                                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(message: "generation queue is full",
                                code: "queue_full")
        }
    }
}

public struct ValidatedChatRequest: Sendable {
    public let messages: [GFTokenizer.Message]
    public let multimodalMessages: [MultimodalMessage]?
    public let imageFiles: [UUID: URL]
    /// Content SHA-256 of each message's images, in order, aligned with
    /// `messages`. Staged image UUIDs are fresh per request, so they cannot
    /// identify an image across turns; the content hash can. Empty for
    /// text-only requests.
    public let imageIdentities: [[String]]
    public let tools: [GFTokenizer.FunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
    /// Present only for the verified Qwen backend. The ordered values come
    /// from the sanitized request bytes rather than Dictionary iteration.
    public let qwenPrompt: ModelFamilyGenerationPrompt?
    public let qwenImagesByID: [String: URL]
    /// Every staging directory this request's image files live in. The parser
    /// and the validator's store each stage under their own lease, and a
    /// request may carry files from both, so dropping either would delete
    /// files the other path staged before `generate` reads them.
    fileprivate let attachmentLeases: [ServerAttachmentLease]

    public init(
        messages: [GFTokenizer.Message],
        multimodalMessages: [MultimodalMessage]? = nil,
        imageFiles: [UUID: URL] = [:],
        imageIdentities: [[String]] = [],
        tools: [GFTokenizer.FunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        qwenPrompt: ModelFamilyGenerationPrompt? = nil,
        qwenImagesByID: [String: URL] = [:]
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.qwenPrompt = qwenPrompt
        self.qwenImagesByID = qwenImagesByID
        self.attachmentLeases = []
    }

    fileprivate init(
        messages: [GFTokenizer.Message],
        multimodalMessages: [MultimodalMessage]? = nil,
        imageFiles: [UUID: URL] = [:],
        imageIdentities: [[String]] = [],
        tools: [GFTokenizer.FunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        attachmentLeases: [ServerAttachmentLease],
        qwenPrompt: ModelFamilyGenerationPrompt? = nil,
        qwenImagesByID: [String: URL] = [:]
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.attachmentLeases = attachmentLeases
        self.qwenPrompt = qwenPrompt
        self.qwenImagesByID = qwenImagesByID
    }
}

private enum OpenAIToolName {
    static let maximumLength = 64

    static func isValid(_ name: String) -> Bool {
        let bytes = name.utf8
        guard !bytes.isEmpty, bytes.count <= maximumLength else { return false }
        return bytes.allSatisfy { byte in
            switch byte {
            case 45, 48...57, 65...90, 95, 97...122:
                true
            default:
                false
            }
        }
    }

    static func validationMessage(for name: String) -> String {
        let prefix = name.prefix(maximumLength + 1)
        let displayed = String(prefix.prefix(maximumLength))
            + (prefix.count > maximumLength ? "..." : "")
        return "tool name \(String(reflecting: displayed)) must contain 1 to 64 ASCII letters, numbers, underscores, or hyphens"
    }
}

public enum OpenAIRequestValidator {
    enum Family: Sendable, Equatable { case gemma, qwen }

    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String) throws -> ValidatedChatRequest {
        try validate(
            request,
            modelID: modelID,
            preStagedImages: [:],
            attachmentLease: nil,
            family: .gemma,
            sanitizedJSON: nil)
    }

    static func validate(_ request: OpenAIChatRequest,
                         modelID: String,
                         preStagedImages: [String: ServerStagedImage],
                         attachmentLease: ServerAttachmentLease?,
                         family: Family = .gemma,
                         sanitizedJSON: Data? = nil) throws -> ValidatedChatRequest {
        guard request.model == modelID else { throw ServerRequestError.unknownModel }
        guard request.n == nil || request.n == 1 else {
            throw invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        guard request.presencePenalty == nil || request.presencePenalty == 0 else {
            throw invalid("presence_penalty must be zero", "presence_penalty", "unsupported_value")
        }
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid("frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        guard request.parallelToolCalls != false else {
            throw invalid("parallel_tool_calls=false is not supported",
                          "parallel_tool_calls", "unsupported_value")
        }

        let temperature = request.temperature ?? 0.2
        guard temperature.isFinite, temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2",
                          "temperature", "invalid_value")
        }
        let topP = request.topP ?? 0.95
        guard topP.isFinite, topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1",
                          "top_p", "invalid_value")
        }
        let topK = request.topK ?? 64
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty.isFinite, repetitionPenalty > 0 else {
            throw invalid("repetition_penalty must be positive",
                          "repetition_penalty", "invalid_value")
        }
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? 4096
        guard maximum > 0 else {
            throw invalid("maximum completion tokens must be positive",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "invalid_value")
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid("tool_choice=required is not supported",
                          "tool_choice", "unsupported_value")
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try family == .gemma
            ? (includeTools ? request.tools ?? [] : []).map(validateTool)
            : []
        if family == .gemma,
           (request.thinking != nil && request.thinking != "auto")
            || request.messages.contains(where: { $0.reasoningContent != nil }) {
            throw invalid("thinking fields require a verified Qwen model",
                          "thinking", "unsupported_value")
        }
        if family == .qwen {
            guard let sanitizedJSON else {
                throw invalid("ordered Qwen request bytes are unavailable",
                              nil, "invalid_request_body")
            }
            let qwen = try validateOrderedQwen(
                sanitizedJSON,
                request: request,
                includeTools: includeTools,
                preStagedImages: preStagedImages,
                attachmentLease: attachmentLease)
            let config = GenerationConfig(
                maxNewTokens: maximum, temperature: temperature,
                topK: topK, topP: topP,
                repetitionPenalty: repetitionPenalty,
                seed: request.seed, stopStrings: request.stop?.values ?? [])
            return ValidatedChatRequest(
                messages: [], tools: [], stream: request.stream ?? false,
                includeUsage: request.streamOptions?.includeUsage ?? false,
                generationConfig: config,
                maximumCompletionTokens: maximum,
                attachmentLeases: qwen.leases,
                qwenPrompt: .chat(
                    messages: qwen.messages,
                    tools: qwen.tools,
                    thinking: qwen.thinking),
                qwenImagesByID: qwen.imagesByID)
        }
        let validatedMessages = try validateMessages(
            request.messages,
            preStagedImages: preStagedImages,
            attachmentLease: attachmentLease)
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: request.stop?.values ?? [])
        return ValidatedChatRequest(messages: validatedMessages.messages,
                                    multimodalMessages: validatedMessages.multimodal,
                                    imageFiles: validatedMessages.imageFiles,
                                    imageIdentities: validatedMessages.imageIdentities,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    attachmentLeases: validatedMessages.leases)
    }

    private static func validateTool(_ tool: OpenAITool) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard OpenAIToolName.isValid(name) else {
            throw invalid(OpenAIToolName.validationMessage(for: name),
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters)
        let parameters = try GemmaToolSchema.adapted(
            tool.function.parameters, toolName: name)
        do {
            guard (try? parameters.jinjaSendableValue()) != nil else {
                throw invalid("tool schema contains a number that cannot be represented exactly",
                              "tools", "invalid_tool_schema")
            }
        } catch {
            // Carry the underlying cause: "cannot be represented exactly" alone
            // does not say which value, and this surfaces to a remote client.
            throw invalid(
                "tool schema cannot be represented exactly: \(error)",
                "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (key, definition) in definitions {
                        guard GemmaToolCallParser.isRepresentableObjectKey(key) else {
                            throw invalid(
                                "tool parameter names may contain only letters, numbers, _, -, ., and $",
                                "tools",
                                "invalid_tool_schema")
                        }
                        try validateSchemaKeys(definition)
                    }
                } else {
                    try validateSchemaKeys(value)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value)
            }
        default:
            break
        }
    }

    private struct ValidatedMessages {
        let messages: [GFTokenizer.Message]
        let multimodal: [MultimodalMessage]?
        let imageFiles: [UUID: URL]
        let imageIdentities: [[String]]
        let leases: [ServerAttachmentLease]
    }

    private static func validateMessages(
        _ input: [OpenAIChatMessage],
        preStagedImages: [String: ServerStagedImage],
        attachmentLease: ServerAttachmentLease?
    ) throws -> ValidatedMessages {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var multimodal: [MultimodalMessage] = []
        var imageFiles: [UUID: URL] = [:]
        var imageIdentities: [[String]] = []
        var messageIdentities: [String] = []
        var store: ServerAttachmentStore?
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid("system or developer guidance must precede the conversation",
                                  "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            var orderedContent: [MultimodalContentPart] = []
            let content: String?
            switch message.content {
            case nil:
                content = nil
            case .text(let text):
                content = text
                orderedContent = [.text(text)]
            case .parts(let parts):
                var joined = ""
                for part in parts {
                    switch part.type {
                    case "text":
                        guard let text = part.text else {
                            throw invalid("text content part requires text",
                                          "messages", "invalid_message")
                        }
                        joined += text
                        orderedContent.append(.text(text))
                    case "image_url":
                        guard role == .user else {
                            throw invalid("image_url is supported only in user messages",
                                          "messages", "unsupported_content")
                        }
                        guard let image = part.imageURL else {
                            throw invalid("image_url content part requires an image_url object",
                                          "messages", "invalid_message")
                        }
                        guard image.detail == nil || image.detail == "auto" else {
                            throw invalid("image detail must be absent or auto",
                                          "messages", "unsupported_value")
                        }
                        let staged: ServerStagedImage
                        let prefix = "turbofieldfare-attachment:"
                        if image.url.hasPrefix(prefix) {
                            let token = String(image.url.dropFirst(prefix.count))
                            guard let existing = preStagedImages[token] else {
                                throw invalid("image attachment lease is missing",
                                              "messages", "invalid_image")
                            }
                            staged = existing
                        } else {
                            if store == nil { store = try ServerAttachmentStore() }
                            staged = try store!.stage(dataURL: image.url)
                        }
                        imageFiles[staged.id] = staged.fileURL
                        messageIdentities.append(staged.sha256)
                        orderedContent.append(.image(id: staged.id))
                    default:
                        throw invalid("unsupported content part \(part.type)",
                                      "messages", "unsupported_content")
                    }
                }
                content = joined
            }
            // The semantic bound on images is the context budget, checked
            // against the model's actual context in `ServerModelSession.prepare`
            // where the per-image token cost is known. Staging enforces its own
            // per-file, per-request, and count resource caps; no further count
            // check belongs here.

            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                guard OpenAIToolName.isValid(call.function.name) else {
                    throw invalid(OpenAIToolName.validationMessage(for: call.function.name),
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                do {
                    guard (try? arguments.gemmaToolArgumentBody()) != nil,
                          (try? arguments.jinjaSendableValue()) != nil else {
                        throw invalid(
                            "historical tool arguments cannot be represented exactly",
                            "messages", "invalid_message")
                    }
                } catch {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly: "
                            + "\(error)",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                      let call = knownCalls[id], !call.resolved else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            result.append(GFTokenizer.Message(role: role,
                                              content: content,
                                              toolCalls: calls,
                                              toolCallID: message.toolCallID,
                                              name: message.name))
            if orderedContent.isEmpty, let content {
                orderedContent = [.text(content)]
            }
            multimodal.append(MultimodalMessage(
                role: role,
                content: orderedContent,
                toolCalls: calls,
                toolCallID: message.toolCallID,
                name: message.name))
            imageIdentities.append(messageIdentities)
            messageIdentities = []
        }
        return ValidatedMessages(
            messages: result,
            multimodal: imageFiles.isEmpty ? nil : multimodal,
            imageFiles: imageFiles,
            imageIdentities: imageFiles.isEmpty ? [] : imageIdentities,
            leases: [attachmentLease, store?.lease].compactMap { $0 })
    }

    private struct OrderedQwenRequest {
        let messages: [ModelChatMessage]
        let tools: [ModelChatToolDefinition]
        let thinking: ModelFamilyThinkingMode
        let imagesByID: [String: URL]
        let leases: [ServerAttachmentLease]
    }

    /// Converts the exact sanitized wire JSON into the order-preserving values
    /// consumed by the pinned Qwen template. The streaming parser has already
    /// enforced the body, nesting, attachment-count, and attachment-byte caps.
    private static func validateOrderedQwen(
        _ data: Data,
        request: OpenAIChatRequest,
        includeTools: Bool,
        preStagedImages: [String: ServerStagedImage],
        attachmentLease: ServerAttachmentLease?
    ) throws -> OrderedQwenRequest {
        let root = try OrderedServerJSON.parse(data)
        let rootMembers = try orderedObject(root, "request")
        guard case .array(let messageRows) = try orderedRequired(
            "messages", in: rootMembers), !messageRows.isEmpty else {
            throw invalid("messages must be a non-empty array",
                          "messages", "invalid_message")
        }
        let thinking: ModelFamilyThinkingMode
        switch request.thinking ?? "auto" {
        case "auto": thinking = .automatic
        case "on": thinking = .enabled
        case "off": thinking = .disabled
        default:
            throw invalid("thinking must be auto, on, or off",
                          "thinking", "invalid_value")
        }

        var store: ServerAttachmentStore?
        var imagesByID: [String: URL] = [:]
        var nextImage = 1
        var knownCalls: [String: Bool] = [:]
        var sawConversation = false
        let messages = try messageRows.map { row -> ModelChatMessage in
            let members = try orderedObject(row, "message")
            let roleText = try orderedString("role", in: members)
            let role: ModelChatRole
            switch roleText {
            case "system": role = .system
            case "developer": role = .developer
            case "user": role = .user
            case "assistant": role = .assistant
            case "tool": role = .tool
            default:
                throw invalid("unsupported message role \(roleText)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversation else {
                    throw invalid(
                        "system or developer guidance must precede the conversation",
                        "messages", "invalid_message")
                }
            } else {
                sawConversation = true
            }
            let content: ModelChatContent?
            if let raw = try orderedOptional("content", in: members) {
                switch raw {
                case .null:
                    content = nil
                case .string(let text):
                    content = .text(text)
                case .array(let parts):
                    content = .parts(try parts.map { part in
                        let partMembers = try orderedObject(part, "content part")
                        switch try orderedString("type", in: partMembers) {
                        case "text":
                            return .text(try orderedString("text", in: partMembers))
                        case "image_url":
                            guard role == .user else {
                                throw invalid(
                                    "image_url is supported only in user messages",
                                    "messages", "unsupported_content")
                            }
                            let image = try orderedObject(
                                try orderedRequired("image_url", in: partMembers),
                                "image_url")
                            if let detail = try orderedOptionalString(
                                "detail", in: image), detail != "auto" {
                                throw invalid("image detail must be absent or auto",
                                              "messages", "unsupported_value")
                            }
                            let url = try orderedString("url", in: image)
                            let staged: ServerStagedImage
                            let prefix = "turbofieldfare-attachment:"
                            if url.hasPrefix(prefix) {
                                let token = String(url.dropFirst(prefix.count))
                                guard let value = preStagedImages[token] else {
                                    throw invalid("image attachment lease is missing",
                                                  "messages", "invalid_image")
                                }
                                staged = value
                            } else {
                                if store == nil { store = try ServerAttachmentStore() }
                                staged = try store!.stage(dataURL: url)
                            }
                            let id = "server-image-\(nextImage)"
                            nextImage += 1
                            imagesByID[id] = staged.fileURL
                            return .image(.init(id: id))
                        case "video_url", "input_video", "video":
                            throw invalid("video input is not supported",
                                          "messages", "unsupported_content")
                        case "audio_url", "input_audio", "audio":
                            throw invalid("audio input is not supported",
                                          "messages", "unsupported_content")
                        case let type:
                            throw invalid("unsupported content part \(type)",
                                          "messages", "unsupported_content")
                        }
                    })
                default:
                    throw invalid("content must be a string, array, or null",
                                  "messages", "invalid_message")
                }
            } else {
                content = nil
            }

            let calls: [ModelChatToolCall]
            if let rawCalls = try orderedOptional("tool_calls", in: members) {
                guard role == .assistant, case .array(let rows) = rawCalls else {
                    throw invalid("tool_calls require an assistant message",
                                  "messages", "invalid_tool_call")
                }
                calls = try rows.map { rawCall in
                    let call = try orderedObject(rawCall, "tool call")
                    let id = try orderedString("id", in: call)
                    guard !id.isEmpty, knownCalls[id] == nil else {
                        throw invalid("invalid or duplicate historical tool call",
                                      "messages", "invalid_tool_call")
                    }
                    if let type = try orderedOptionalString("type", in: call),
                       type != "function" {
                        throw invalid("only function tool calls are supported",
                                      "messages", "invalid_tool_call")
                    }
                    let function = try orderedObject(
                        try orderedRequired("function", in: call), "tool call function")
                    let name = try orderedString("name", in: function)
                    guard OpenAIToolName.isValid(name) else {
                        throw invalid(OpenAIToolName.validationMessage(for: name),
                                      "messages", "invalid_tool_call")
                    }
                    var arguments = try orderedRequired("arguments", in: function)
                    if case .string(let encoded) = arguments {
                        arguments = try OrderedServerJSON.parse(Data(encoded.utf8))
                    }
                    guard case .object = arguments else {
                        throw invalid("historical tool arguments must be a JSON object",
                                      "messages", "invalid_tool_arguments")
                    }
                    knownCalls[id] = false
                    return .init(id: id, name: name, arguments: arguments)
                }
            } else {
                calls = []
            }
            let toolCallID = try orderedOptionalString("tool_call_id", in: members)
            if role == .tool {
                guard let toolCallID, knownCalls[toolCallID] == false else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[toolCallID] = true
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            return ModelChatMessage(
                role: role,
                content: content,
                reasoningContent: try orderedOptionalString(
                    "reasoning_content", in: members),
                toolCalls: calls,
                toolCallID: toolCallID,
                name: try orderedOptionalString("name", in: members))
        }

        var tools: [ModelChatToolDefinition] = []
        if includeTools, let rawTools = try orderedOptional("tools", in: rootMembers) {
            guard case .array(let rows) = rawTools else {
                throw invalid("tools must be an array", "tools", "invalid_tool_schema")
            }
            tools = try rows.map { row in
                let tool = try orderedObject(row, "tool")
                guard try orderedString("type", in: tool) == "function" else {
                    throw invalid("only function tools are supported",
                                  "tools", "unsupported_tool")
                }
                let function = try orderedObject(
                    try orderedRequired("function", in: tool), "function")
                let name = try orderedString("name", in: function)
                guard OpenAIToolName.isValid(name) else {
                    throw invalid(OpenAIToolName.validationMessage(for: name),
                                  "tools", "invalid_tool_name")
                }
                let parameters = try orderedRequired("parameters", in: function)
                guard case .object = parameters else {
                    throw invalid("tool parameters must be an object schema",
                                  "tools", "invalid_tool_schema")
                }
                return ModelChatToolDefinition(
                    function: .init(
                        name: name,
                        description: try orderedOptionalString(
                            "description", in: function) ?? "",
                        parameters: parameters))
            }
        }
        return OrderedQwenRequest(
            messages: messages, tools: tools, thinking: thinking,
            imagesByID: imagesByID,
            leases: [attachmentLease, store?.lease].compactMap { $0 })
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}

private func orderedObject(
    _ value: ModelChatJSONValue, _ label: String
) throws -> [ModelChatJSONMember] {
    guard case .object(let members) = value else {
        throw ServerRequestError.invalid(
            message: "\(label) must be an object", param: nil,
            code: "invalid_request_body")
    }
    return members
}

private func orderedOptional(
    _ name: String, in members: [ModelChatJSONMember]
) throws -> ModelChatJSONValue? {
    let values = members.filter { $0.name == name }
    guard values.count <= 1 else {
        throw ServerRequestError.invalid(
            message: "duplicate key: \(name)", param: name,
            code: "invalid_request_body")
    }
    return values.first?.value
}

private func orderedRequired(
    _ name: String, in members: [ModelChatJSONMember]
) throws -> ModelChatJSONValue {
    guard let value = try orderedOptional(name, in: members) else {
        throw ServerRequestError.invalid(
            message: "missing field: \(name)", param: name,
            code: "invalid_request_body")
    }
    return value
}

private func orderedString(
    _ name: String, in members: [ModelChatJSONMember]
) throws -> String {
    guard case .string(let value) = try orderedRequired(name, in: members) else {
        throw ServerRequestError.invalid(
            message: "\(name) must be a string", param: name,
            code: "invalid_request_body")
    }
    return value
}

private func orderedOptionalString(
    _ name: String, in members: [ModelChatJSONMember]
) throws -> String? {
    guard let value = try orderedOptional(name, in: members) else { return nil }
    if case .null = value { return nil }
    guard case .string(let string) = value else {
        throw ServerRequestError.invalid(
            message: "\(name) must be a string or null", param: name,
            code: "invalid_request_body")
    }
    return string
}

/// Bounded by `StreamingChatRequestBody.maximumSanitizedBytes` before entry.
/// Objects retain their source member order for Qwen template parity.
private enum OrderedServerJSON {
    struct ParseError: Error, CustomStringConvertible {
        let description: String
    }

    static func parse(_ data: Data) throws -> ModelChatJSONValue {
        guard data.count <= StreamingChatRequestBody.maximumSanitizedBytes else {
            throw ServerRequestError.invalid(
                message: "non-image request content is too large", param: nil,
                code: "request_too_large")
        }
        do {
            _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        } catch {
            throw ParseError(description: "invalid JSON: \(error)")
        }
        var parser = Parser(bytes: Array(data))
        let result = try parser.value()
        parser.whitespace()
        guard parser.index == parser.bytes.count else {
            throw ParseError(description: "invalid JSON: trailing bytes")
        }
        return result
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        mutating func value() throws -> ModelChatJSONValue {
            whitespace()
            guard index < bytes.count else { throw error("unexpected end") }
            switch bytes[index] {
            case 0x7B: return try object()
            case 0x5B: return try array()
            case 0x22: return .string(try string())
            case 0x74: try literal("true"); return .bool(true)
            case 0x66: try literal("false"); return .bool(false)
            case 0x6E: try literal("null"); return .null
            case 0x2D, 0x30...0x39: return try number()
            default: throw error("unexpected byte at \(index)")
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
                    throw error("object key is not a string")
                }
                let key = try string()
                whitespace()
                guard take(0x3A) else { throw error("missing colon") }
                members.append(.init(key, try value()))
                whitespace()
                if take(0x7D) { return .object(members) }
                guard take(0x2C) else { throw error("missing comma") }
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
                guard take(0x2C) else { throw error("missing comma") }
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
                    do {
                        return try JSONDecoder().decode(
                            String.self, from: Data(bytes[start..<index]))
                    } catch {
                        throw self.error("invalid string escape")
                    }
                }
                guard byte >= 0x20 else { throw error("control byte in string") }
            }
            throw error("unterminated string")
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
                throw error("invalid number")
            }
            return .number(value)
        }

        mutating func literal(_ text: String) throws {
            let expected = Array(text.utf8)
            guard index + expected.count <= bytes.count,
                  bytes[index..<(index + expected.count)].elementsEqual(expected) else {
                throw error("invalid literal")
            }
            index += expected.count
        }

        mutating func whitespace() {
            while index < bytes.count,
                  bytes[index] == 0x20 || bytes[index] == 0x09
                    || bytes[index] == 0x0A || bytes[index] == 0x0D { index += 1 }
        }

        mutating func take(_ byte: UInt8) -> Bool {
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
            return true
        }

        func error(_ detail: String) -> ParseError {
            ParseError(description: "invalid JSON: \(detail)")
        }
    }
}
