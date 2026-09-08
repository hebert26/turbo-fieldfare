import Foundation

/// A failed attempt, never a completed assistant turn or an instruction to replay input.
public struct ThoughtRepetitionRecovery: Error, Codable, Equatable, Sendable {
    public let canRetryToolResult: Bool
    public let pendingCallID: String?
    public let pendingToolName: String?
    public let restoredTokenCount: Int
    public let requiresRebuild: Bool
    public let generatedTokens: Int
    public let thinkingTokens: Int
    public let blockTokens: Int
    public let repetitions: Int
}

struct RepeatedThoughtDetected: Error {
    let blockTokens: Int
}

/// Exact periodic token matching, independent of words, app names or elapsed time.
/// At most 1,024 token IDs. Check every four tokens, with early mismatches.
struct ThoughtRepetitionDetector {
    static let capacity = 1_024
    static let repetitions = 8
    static let minimumBlock = 16
    static let maximumBlock = 128
    private var ring = [Int32](repeating: 0, count: capacity)
    private var count = 0
    private var next = 0

    mutating func reset() { count = 0; next = 0 }

    mutating func append(_ token: Int32) -> Int? {
        ring[next] = token
        next = (next + 1) % Self.capacity
        count = min(count + 1, Self.capacity)
        guard count >= Self.minimumBlock * Self.repetitions, next.isMultiple(of: 4) else { return nil }
        func previous(_ distance: Int) -> Int32 {
            ring[(next - 1 - distance + Self.capacity) % Self.capacity]
        }
        for width in Self.minimumBlock...min(Self.maximumBlock, count / Self.repetitions) {
            var matches = true
            for offset in width..<(width * Self.repetitions) {
                if previous(offset) != previous(offset % width) { matches = false; break }
            }
            guard matches else { continue }
            // A long block made entirely of a short common phrase is not eligible.
            guard Set((0..<width).map(previous)).count >= 8 else { continue }
            let shortPeriod = (1..<Self.minimumBlock).contains { period in
                width.isMultiple(of: period) && (period..<width).allSatisfy {
                    previous($0) == previous($0 % period)
                }
            }
            if !shortPeriod { return width }
        }
        return nil
    }
}

/// Diagnostic-only rejected call span. Never includes surrounding channel text.
public struct StructuredToolFailureEvidence: Codable, Equatable, Sendable {
    public let phase: String
    public let payloadTokenIDs: [Int32]?
    public let payloadText: String?
    public let triggerTokenID: Int32?
    public var originalStopReason: String?
}

public struct StructuredToolFailure: Error, CustomStringConvertible, Sendable {
    public let underlying: GemmaToolCallParserError
    /// True only for malformed output after successful KV restoration, with no
    /// captured calls and no cancellation request. Independent of raw tracing.
    public let canRegenerateToolResult: Bool
    public let evidence: StructuredToolFailureEvidence?
    public var description: String { String(describing: underlying) }
}

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

/// Bounded raw draft for display and opt-in diagnostics, never an executable call.
public struct StructuredToolCallPreview: Equatable, Sendable {
    public let text: String
    public let middleTextOmitted: Bool
}

/// Decoder counters plus a bounded, display-only window of recognised thought text.
public struct StructuredAssistantProgress: Equatable, Sendable {
    public static let maximumThoughtPreviewBytes = 8 * 1_024
    public enum Stage: String, Sendable {
        case thinking
        case toolCall = "tool_call"
        case visibleResponse = "visible_response"
        case channelLabel = "channel_label"
        case unknownHiddenChannel = "unknown_hidden_channel"
    }
    public var stage: Stage = .visibleResponse
    public var thinkingTokens = 0
    public var toolCallTokens = 0
    public var visibleResponseTokens = 0
    public var channelLabelTokens = 0
    public var unknownHiddenChannelTokens = 0
    public var recentThoughtText = ""
    public var earlierThoughtTextOmitted = false
    public var toolCallPreview: StructuredToolCallPreview?
}

public final class StructuredAssistantDecoder: @unchecked Sendable {
    private enum Channel {
        case thought
        case visible
        case label
    }

    private let tokenizer: GFTokenizer
    private let allowedTools: Set<String>
    private let acceptsUnknownToolNames: Bool
    private let idGenerator: @Sendable () -> String
    private var channel: Channel = .visible
    private var isKnownThoughtChannel = false
    public private(set) var progress = StructuredAssistantProgress()
    private var label = ""
    private var toolTokens: [Int32]?
    private var emittedCalls = 0
    private var failed = false
    private let captureFailureEvidence: Bool
    private let captureThoughtPreview: Bool
    private var thoughtRepetition: ThoughtRepetitionDetector?
    private var previewTokenCount = 0
    private var openingDetokenizer: GFDetokenizer?
    private var openingText = ""
    private(set) var failureEvidence: StructuredToolFailureEvidence?

    public init(tokenizer: GFTokenizer,
                allowedTools: Set<String>,
                startsInThoughtChannel: Bool = false,
                acceptsUnknownToolNames: Bool = false,
                captureFailureEvidence: Bool = false,
                captureThoughtPreview: Bool = false,
                detectThoughtRepetition: Bool = false,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.acceptsUnknownToolNames = acceptsUnknownToolNames
        self.captureFailureEvidence = captureFailureEvidence
        self.captureThoughtPreview = captureThoughtPreview
        self.thoughtRepetition = detectThoughtRepetition ? ThoughtRepetitionDetector() : nil
        self.idGenerator = idGenerator
        // The opener belongs to the prompt, not generated output. Seed state
        // directly so its tokens cannot inflate generated progress counters.
        if startsInThoughtChannel {
            channel = .thought
            isKnownThoughtChannel = true
            progress.stage = .thinking
        }
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw GemmaToolCallParserError.malformed }
        defer { recordProgress() }

        // A non-empty delta on a control token is text the detokenizer held
        // back from BEFORE the token (a skipped special contributes nothing of
        // its own), so it belongs to the channel state in effect now — route
        // it before the token changes that state. Inside a tool call the held
        // bytes are part of the payload, which is re-decoded from its IDs at
        // toolCallEnd, so nothing is lost by not routing there.
        let isControl = tokenID == tokenizer.channelStartID
            || tokenID == tokenizer.channelEndID
            || tokenID == tokenizer.toolCallStartID
            || tokenID == tokenizer.toolCallEndID
            || tokenID == tokenizer.toolResponseID
            || tokenID == tokenizer.toolResponseEndID
        if isControl || channel != .thought || !isKnownThoughtChannel || toolTokens != nil {
            thoughtRepetition?.reset()
        } else if emittedCalls == 0, let width = thoughtRepetition?.append(tokenID) {
            throw RepeatedThoughtDetected(blockTokens: width)
        }
        var events: [StructuredAssistantEvent] = []
        if isControl, !delta.isEmpty, toolTokens == nil {
            events = routeText(delta)
        }

        if tokenID == tokenizer.channelStartID {
            label = ""
            channel = .label
            return events
        }
        if tokenID == tokenizer.channelEndID {
            channel = .visible
            return events
        }
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                captureFailure(phase: "nested_tool_start", tokens: toolTokens, trigger: tokenID)
                failed = true
                throw GemmaToolCallParserError.malformed
            }
            toolTokens = []
            previewTokenCount = 0
            openingDetokenizer = GFDetokenizer(tokenizer: tokenizer, skipSpecialTokens: false)
            openingText = ""
            if captureThoughtPreview {
                progress.toolCallPreview = StructuredToolCallPreview(text: "", middleTextOmitted: false)
            }
            return events
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                captureFailure(phase: "tool_end_without_start", trigger: tokenID)
                failed = true
                throw GemmaToolCallParserError.malformed
            }
            refreshToolCallPreview(tokens: tokens)
            toolTokens = nil
            openingDetokenizer = nil
            openingText = ""
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try GemmaToolCallParser().parse(
                    text,
                    allowedTools: allowedTools,
                    id: idGenerator(),
                    acceptsUnknownToolNames: acceptsUnknownToolNames)
                emittedCalls += 1
                progress.toolCallPreview = nil
                return events + [.toolCall(call)]
            } catch {
                captureFailure(phase: "tool_payload_parse", tokens: tokens, text: text, trigger: tokenID)
                failed = true
                throw error
            }
        }
        if tokenID == tokenizer.toolResponseID || tokenID == tokenizer.toolResponseEndID {
            guard emittedCalls > 0, toolTokens == nil else {
                captureFailure(phase: "unexpected_tool_response", tokens: toolTokens, trigger: tokenID)
                failed = true
                throw GemmaToolCallParserError.malformed
            }
            return events
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= GemmaToolCallParser.maximumBytes else {
                captureFailure(phase: "tool_payload_oversized", tokens: tokens, trigger: tokenID)
                failed = true
                throw GemmaToolCallParserError.oversized
            }
            toolTokens = tokens
            if previewTokenCount == 0 || tokens.count - previewTokenCount >= 32 {
                refreshToolCallPreview(tokens: tokens)
            }
            try inspectOpening(tokenID: tokenID, tokens: tokens)
            return []
        }
        return routeText(delta)
    }

    /// Route text flushed at a stop boundary through the current channel
    /// state. The flush tail is not tied to a token ID, so it cannot go
    /// through `consume`; without this a generation cut off inside the
    /// thought channel would leak its held-back bytes into visible content.
    public func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw GemmaToolCallParserError.malformed }
        guard toolTokens == nil, !text.isEmpty else { return [] }
        return routeText(text)
    }

    private func routeText(_ delta: String) -> [StructuredAssistantEvent] {
        switch channel {
        case .thought:
            if isKnownThoughtChannel { appendThoughtPreview(delta) }
            return []
        case .visible:
            return delta.isEmpty ? [] : [.content(delta)]
        case .label:
            label += delta
            guard let newline = label.firstIndex(of: "\n") else { return [] }
            let name = label[..<newline].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let contentStart = label.index(after: newline)
            let content = String(label[contentStart...])
            channel = name == "final" || name == "answer" ? .visible : .thought
            isKnownThoughtChannel = name == "thought"
            label = ""
            if channel == .visible, !content.isEmpty {
                return [.content(content)]
            }
            if isKnownThoughtChannel { appendThoughtPreview(content) }
            return []
        }
    }

    private func appendThoughtPreview(_ text: String) {
        guard captureThoughtPreview, !text.isEmpty else { return }
        progress.recentThoughtText += text
        let bytes = progress.recentThoughtText.utf8
        guard bytes.count > StructuredAssistantProgress.maximumThoughtPreviewBytes else { return }
        var start = bytes.index(bytes.endIndex,
                                offsetBy: -StructuredAssistantProgress.maximumThoughtPreviewBytes)
        // Retain whole UTF-8 scalars, never introduce a replacement character.
        while bytes[start] & 0xC0 == 0x80 { start = bytes.index(after: start) }
        progress.recentThoughtText = String(decoding: bytes[start...], as: UTF8.self)
        progress.earlierThoughtTextOmitted = true
    }

    public func finish() throws {
        guard !failed, toolTokens == nil else {
            captureFailure(phase: "unfinished_tool_span", tokens: toolTokens)
            throw GemmaToolCallParserError.malformed
        }
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }

    /// Inspect only the opening 256 tokens / 8 KiB. Reaching either bound
    /// disables this check, never generation. Complete parsing is unchanged.
    private func inspectOpening(tokenID: Int32, tokens: [Int32]) throws {
        // Only committed fragments are safe to reject. Flushing a byte-fallback
        // run here could manufacture replacement text before its final byte.
        guard let fragment = openingDetokenizer?.push(tokenID) else { return }
        guard openingText.utf8.count + fragment.utf8.count <= 8 * 1_024 else {
            openingDetokenizer = nil
            openingText = ""
            return
        }
        openingText += fragment
        if tokens.count == 1 || tokens.count % 8 == 0 || tokens.count == 256 {
            // The last grapheme may gain combining scalars on the next push.
            // Excluding it also keeps an unfinished delimiter pending.
            let stableOpening = String(openingText.dropLast())
            if GemmaToolCallParser().hasInvalidOpeningPrefix(stableOpening) {
                captureFailure(phase: "tool_payload_prefix", tokens: tokens, trigger: tokenID)
                failed = true
                throw GemmaToolCallParserError.malformed
            }
        }
        if tokens.count >= 256 {
            openingDetokenizer = nil
            openingText = ""
        }
    }

    /// Decode bounded windows only, never the growing payload on each token.
    /// A cut byte-fallback sequence may be incomplete at an omitted boundary.
    func refreshToolCallPreview() {
        if let toolTokens { refreshToolCallPreview(tokens: toolTokens) }
    }

    private func refreshToolCallPreview(tokens: [Int32]) {
        guard captureThoughtPreview else { return }
        previewTokenCount = tokens.count
        let prefixLimit = 2 * 1_024
        let tailLimit = 6 * 1_024
        func prefix(_ text: String, limit: Int) -> String {
            let bytes = text.utf8
            guard bytes.count > limit else { return text }
            var end = bytes.index(bytes.startIndex, offsetBy: limit)
            while bytes[end] & 0xC0 == 0x80 { end = bytes.index(before: end) }
            return String(decoding: bytes[..<end], as: UTF8.self)
        }
        func tail(_ text: String, limit: Int) -> String {
            let bytes = text.utf8
            guard bytes.count > limit else { return text }
            var start = bytes.index(bytes.endIndex, offsetBy: -limit)
            while bytes[start] & 0xC0 == 0x80 { start = bytes.index(after: start) }
            return String(decoding: bytes[start...], as: UTF8.self)
        }
        let text: String
        let omitted: Bool
        if tokens.count <= 1_792 {
            let decoded = tokenizer.decode(tokens, skipSpecialTokens: false)
            omitted = decoded.utf8.count > prefixLimit + tailLimit
            text = omitted
                ? prefix(decoded, limit: prefixLimit) + tail(decoded, limit: tailLimit)
                : decoded
        } else {
            text = prefix(tokenizer.decode(Array(tokens.prefix(256)), skipSpecialTokens: false), limit: prefixLimit)
                + tail(tokenizer.decode(Array(tokens.suffix(1_536)), skipSpecialTokens: false), limit: tailLimit)
            omitted = true
        }
        progress.toolCallPreview = StructuredToolCallPreview(text: text, middleTextOmitted: omitted)
    }

    private func recordProgress() {
        if toolTokens != nil {
            progress.stage = .toolCall
            progress.toolCallTokens += 1
            return
        }
        switch channel {
        case .visible:
            progress.stage = .visibleResponse
            progress.visibleResponseTokens += 1
        case .label:
            progress.stage = .channelLabel
            progress.channelLabelTokens += 1
        case .thought:
            if isKnownThoughtChannel {
                progress.stage = .thinking
                progress.thinkingTokens += 1
            } else {
                progress.stage = .unknownHiddenChannel
                progress.unknownHiddenChannelTokens += 1
            }
        }
    }

    func captureFailure(
        phase: String, tokens: [Int32]? = nil, text: String? = nil, trigger: Int32? = nil
    ) {
        if let tokens { refreshToolCallPreview(tokens: tokens) }
        guard captureFailureEvidence, failureEvidence == nil else { return }
        failureEvidence = StructuredToolFailureEvidence(
            phase: phase, payloadTokenIDs: tokens,
            payloadText: text ?? tokens.map { tokenizer.decode($0, skipSpecialTokens: false) },
            triggerTokenID: trigger, originalStopReason: nil)
    }
}
