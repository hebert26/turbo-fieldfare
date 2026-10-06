import Foundation

public enum MultimodalContentPart: Sendable, Equatable {
    case text(String)
    case image(id: UUID)
}

/// A continuation turn's content, before image token counts are known.
public enum MultimodalContinuationPart: Sendable, Equatable {
    case text(String)
    case image
}

/// Tokens for a continuation turn. Ranges are relative to this turn, which is
/// also the suffix handed to the runner, so they need no rebasing.
public struct MultimodalContinuationTokens: Sendable, Equatable {
    public let effectiveTokenIDs: [Int32]
    public let embeddingTokenIDs: [Int32]
    public let imageTokenRanges: [Range<Int>]
}

public struct MultimodalMessage: Sendable, Equatable {
    public let role: GFTokenizer.Role
    public let content: [MultimodalContentPart]
    public let toolCalls: [GFTokenizer.HistoricalToolCall]
    public let toolCallID: String?
    public let name: String?

    public init(role: GFTokenizer.Role,
                content: [MultimodalContentPart],
                toolCalls: [GFTokenizer.HistoricalToolCall] = [],
                toolCallID: String? = nil,
                name: String? = nil) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.name = name
    }
}

public enum MultimodalPromptRendererError: Error, Equatable {
    case emptyMessages
    case emptyContent
    case reservedImageMarker
    case missingImage(UUID)
    case unexpectedImage(UUID)
    case placeholderMismatch
}

enum MultimodalTokenLayout: Sendable, Equatable {
    case gemma
    case qwen(imageTokenID: Int32, visionStartTokenID: Int32, visionEndTokenID: Int32)

    static func qwen(_ config: QwenArchConfig) -> Self {
        .qwen(
            imageTokenID: Int32(config.imageTokenID),
            visionStartTokenID: Int32(config.visionStartTokenID),
            visionEndTokenID: Int32(config.visionEndTokenID))
    }
}

public enum MultimodalPromptRenderer {
    public static let placeholder = "<|image|>"
    public static let imageTokenID = Int32(258_880)
    public static let beginImageTokenID = Int32(255_999)
    public static let endImageTokenID = Int32(258_882)

    public static func render(
        messages: [MultimodalMessage],
        featuresByID: [UUID: VisionFeatures],
        tokenizer: GFTokenizer,
        tools: [GFTokenizer.FunctionDefinition] = []
    ) throws -> MultimodalPrefillInput {
        guard !messages.isEmpty else { throw MultimodalPromptRendererError.emptyMessages }
        var orderedImages: [(UUID, VisionFeatures)] = []
        let tokenizerMessages = try messages.map { message in
            guard !message.content.isEmpty || !message.toolCalls.isEmpty else {
                throw MultimodalPromptRendererError.emptyContent
            }
            var text = ""
            for part in message.content {
                switch part {
                case .text(let value):
                    guard !value.contains(placeholder) else {
                        throw MultimodalPromptRendererError.reservedImageMarker
                    }
                    text += value
                case .image(let id):
                    guard let features = featuresByID[id] else {
                        throw MultimodalPromptRendererError.missingImage(id)
                    }
                    text += placeholder
                    orderedImages.append((id, features))
                }
            }
            return GFTokenizer.Message(
                role: message.role,
                content: text,
                toolCalls: message.toolCalls,
                toolCallID: message.toolCallID,
                name: message.name)
        }
        guard Set(orderedImages.map(\.0)) == Set(featuresByID.keys) else {
            let used = Set(orderedImages.map(\.0))
            throw MultimodalPromptRendererError.unexpectedImage(
                featuresByID.keys.first { !used.contains($0) }!)
        }

        let usesToolTemplate = !tools.isEmpty || tokenizerMessages.contains {
            $0.role == .system || $0.role == .developer || $0.role == .tool
                || !$0.toolCalls.isEmpty
        }
        let templateTokens: [Int32]
        if usesToolTemplate {
            templateTokens = try tokenizer.encodeToolChat(
                messages: tokenizerMessages,
                tools: tools)
        } else {
            let rendered = try tokenizer.applyChatTemplate(tokenizerMessages)
            templateTokens = tokenizer.encode(rendered, addBOS: false)
        }
        return try expandingImageTokens(
            templateTokens, features: orderedImages.map(\.1))
    }

    /// Expands Qwen image-pad markers without borrowing Gemma token IDs or its
    /// fixed row count. Product-facing Qwen chat routing remains P18/P20; this
    /// prepared boundary is directly consumable by that routing.
    static func expandingQwenImageTokens(
        _ templateTokens: [Int32],
        features: [QwenVisionFeatures],
        architecture: QwenArchConfig,
        config: QwenVisionConfig = .official,
        limits: QwenVisionResourceLimits = .provisional
    ) throws -> QwenMultimodalPrefillInput {
        let image = Int32(architecture.imageTokenID)
        let placeholders = templateTokens.indices.filter {
            templateTokens[$0] == image
        }
        guard placeholders.count == features.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        var effective: [Int32] = []
        var embedding: [Int32] = []
        var spans: [QwenMultimodalImageSpan] = []
        var grids: [QwenVisionGrid] = []
        let rows = features.reduce(0) { $0 + $1.tokenCount }
        guard rows <= limits.maximumVisibleHistoryRows else {
            throw QwenVisionError.mergedRowQuotaExceeded(
                requested: rows, maximum: limits.maximumVisibleHistoryRows)
        }
        effective.reserveCapacity(templateTokens.count + rows + features.count)
        embedding.reserveCapacity(effective.capacity)
        var featureIndex = 0
        for token in templateTokens {
            guard token == image else {
                effective.append(token)
                embedding.append(token)
                continue
            }
            let value = features[featureIndex]
            let start = Int32(architecture.visionStartTokenID)
            let end = Int32(architecture.visionEndTokenID)
            effective.append(start)
            embedding.append(start)
            let lower = effective.count
            effective.append(contentsOf: repeatElement(image, count: value.tokenCount))
            embedding.append(contentsOf: repeatElement(image, count: value.tokenCount))
            let range = lower..<effective.count
            spans.append(QwenMultimodalImageSpan(tokenRange: range, features: value))
            grids.append(value.grid)
            effective.append(end)
            embedding.append(end)
            featureIndex += 1
        }
        let positions = try QwenMultimodalPositions.make(
            tokenCount: effective.count,
            imageRanges: spans.map(\.tokenRange), grids: grids,
            maximumRows: limits.maximumVisibleHistoryRows)
        return try QwenMultimodalPrefillInput(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageSpans: spans,
            positionPlan: positions,
            config: config,
            limits: limits)
    }

    /// Expands only the supplied native prompt or continuation. Callers retain
    /// past image tokens in their KV without retaining or re-encoding pixels.
    static func expandingImageTokens(
        _ templateTokens: [Int32], features: [VisionFeatures]
    ) throws -> MultimodalPrefillInput {
        let placeholders = templateTokens.indices.filter {
            templateTokens[$0] == imageTokenID
        }
        guard placeholders.count == features.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        var effective: [Int32] = []
        var embedding: [Int32] = []
        var spans: [MultimodalImageSpan] = []
        effective.reserveCapacity(
            templateTokens.count + features.reduce(0) { $0 + $1.tokenCount + 1 })
        embedding.reserveCapacity(effective.capacity)
        var imageIndex = 0
        for token in templateTokens {
            guard token == imageTokenID else {
                effective.append(token)
                embedding.append(token)
                continue
            }
            let imageFeatures = features[imageIndex]
            effective.append(beginImageTokenID)
            embedding.append(beginImageTokenID)
            let lower = effective.count
            effective.append(contentsOf: repeatElement(imageTokenID, count: imageFeatures.tokenCount))
            embedding.append(contentsOf: repeatElement(Int32(0), count: imageFeatures.tokenCount))
            spans.append(MultimodalImageSpan(
                tokenRange: lower..<effective.count,
                features: imageFeatures))
            effective.append(endImageTokenID)
            embedding.append(endImageTokenID)
            imageIndex += 1
        }
        return try MultimodalPrefillInput(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageSpans: spans)
    }
}
