import Foundation

/// Incrementally separates Qwen thought, visible, and structured-call text.
/// This value owns one stream. Calls remain non-executable until `finish()`
/// validates the entire turn and assigns host-owned IDs.
public struct QwenStructuredAssistantDecoder: Sendable {
    public static let maximumPendingToolBytes = GemmaToolCallParser.maximumBytes

    private enum State: Sendable, Equatable {
        case thought
        case visible
        case toolCall
    }

    private static let thoughtStart = "<think>"
    private static let thoughtEnd = "</think>"
    private static let toolStart = "<tool_call>"
    private static let toolEnd = "</tool_call>"
    private static let toolResponseStart = "<tool_response>"
    private static let toolResponseEnd = "</tool_response>"
    private static let markers = [
        thoughtStart, thoughtEnd, toolStart, toolEnd,
        toolResponseStart, toolResponseEnd,
    ]

    private let tools: [ModelChatToolDefinition]
    private let idGenerator: @Sendable () -> String
    private var state: State
    private var pending = ""
    private var activeFrame = ""
    private var frames: [String] = []
    private var completedFrameBytes = 0
    private var sawToolCall = false
    private var releasedCallCount = 0
    private var terminalSeen = false
    private var terminalTailConsumed = false
    private var finalized = false
    private var failed = false
    public private(set) var progress = StructuredAssistantProgress()

    public init(
        tools: [ModelChatToolDefinition],
        startsInThoughtChannel: Bool,
        idGenerator: @escaping @Sendable () -> String = {
            "call_" + (0..<24).map { _ in
                String(format: "%x", UInt8.random(in: 0...15))
            }.joined()
        }
    ) {
        self.tools = tools
        self.idGenerator = idGenerator
        state = startsInThoughtChannel ? .thought : .visible
        progress.stage = startsInThoughtChannel ? .thinking : .visibleResponse
    }

    public mutating func consume(_ text: String) throws -> [StructuredAssistantEvent] {
        try consume(text, countsToken: !text.isEmpty)
    }

    /// One sampled token, including tokens for which the incremental decoder
    /// has not released text yet. This keeps stage counters token-accurate.
    mutating func consumeToken(_ text: String) throws -> [StructuredAssistantEvent] {
        try consume(text, countsToken: true)
    }

    private mutating func consume(
        _ text: String,
        countsToken: Bool
    ) throws -> [StructuredAssistantEvent] {
        guard !failed, !finalized else { throw GemmaToolCallParserError.malformed }
        guard !terminalSeen else {
            fail()
            throw GemmaToolCallParserError.malformed
        }
        let startingState = state
        let events = try consumeText(text)
        recordProgress(text, startingState: startingState, countsToken: countsToken)
        return events
    }

    /// The only publication point. A second successful finish is a no-op.
    @discardableResult
    public mutating func finish() throws -> [StructuredAssistantEvent] {
        if finalized { return [] }
        guard !failed else { throw GemmaToolCallParserError.malformed }
        do {
            let trailingEvents = try drain(final: true)
            guard state == .visible, activeFrame.isEmpty else {
                throw GemmaToolCallParserError.malformed
            }

            let parser = QwenToolCallParser()
            let validated = try frames.map {
                try parser.parse($0, tools: tools, id: "pending_host_id")
            }
            let calls = validated.map { call in
                ParsedToolCall(
                    id: idGenerator(),
                    name: call.name,
                    arguments: call.arguments,
                    argumentsJSON: call.argumentsJSON)
            }
            releasedCallCount = calls.count
            finalized = true
            frames.removeAll(keepingCapacity: false)
            completedFrameBytes = 0
            return trailingEvents + calls.map(StructuredAssistantEvent.toolCall)
        } catch {
            fail()
            throw error
        }
    }

    public var hasToolCalls: Bool { releasedCallCount > 0 }

    /// Marks model EOS without publishing. The pinned, SHA-verified Qwen
    /// tokenizer declares `<|im_end|>` as ID 248046 (`QwenTokenizer.eosID`).
    mutating func markEndOfStream() throws {
        guard !failed, !finalized else { throw GemmaToolCallParserError.malformed }
        guard !terminalSeen else {
            fail()
            throw GemmaToolCallParserError.malformed
        }
        terminalSeen = true
    }

    /// Routes the detokenizer's one held suffix after EOS, still without
    /// publishing. A second tail is malformed.
    mutating func consumeTerminalTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed, !finalized else { throw GemmaToolCallParserError.malformed }
        guard terminalSeen, !terminalTailConsumed else {
            fail()
            throw GemmaToolCallParserError.malformed
        }
        terminalTailConsumed = true
        let startingState = state
        let events = try consumeText(text)
        recordProgress(text, startingState: startingState, countsToken: false)
        return events
    }

    var hasSeenEndOfStream: Bool { terminalSeen }

    private mutating func consumeText(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !text.isEmpty else { return [] }
        pending += text
        do {
            if state == .toolCall || sawToolCall { try enforcePendingByteLimit() }
            return try drain(final: false)
        } catch {
            fail()
            throw error
        }
    }

    private mutating func drain(final: Bool) throws -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        while !pending.isEmpty {
            switch state {
            case .visible:
                if let match = firstMarker(in: pending) {
                    let prefix = String(pending[..<match.range.lowerBound])
                    events += try routeVisible(prefix)
                    pending.removeSubrange(..<match.range.upperBound)
                    switch match.marker {
                    case Self.thoughtStart:
                        guard !sawToolCall else { throw GemmaToolCallParserError.malformed }
                        state = .thought
                    case Self.toolStart:
                        activeFrame = Self.toolStart
                        state = .toolCall
                        try enforcePendingByteLimit()
                    default:
                        throw GemmaToolCallParserError.malformed
                    }
                    continue
                }
                let held = final ? partialMarkerLength(in: pending) : markerSuffixLength(in: pending)
                if final, held > 0 { throw GemmaToolCallParserError.malformed }
                let split = pending.index(pending.endIndex, offsetBy: -held)
                let safe = String(pending[..<split])
                pending = String(pending[split...])
                events += try routeVisible(safe)
                return events

            case .thought:
                if let match = firstMarker(in: pending) {
                    pending.removeSubrange(..<match.range.upperBound)
                    guard match.marker == Self.thoughtEnd else {
                        throw GemmaToolCallParserError.malformed
                    }
                    state = .visible
                    continue
                }
                let held = final ? partialMarkerLength(in: pending) : markerSuffixLength(in: pending)
                if final { throw GemmaToolCallParserError.malformed }
                let split = pending.index(pending.endIndex, offsetBy: -held)
                pending = String(pending[split...])
                return events

            case .toolCall:
                if let match = firstMarker(in: pending) {
                    let prefix = String(pending[..<match.range.lowerBound])
                    activeFrame += prefix
                    pending.removeSubrange(..<match.range.upperBound)
                    guard match.marker == Self.toolEnd else {
                        throw GemmaToolCallParserError.malformed
                    }
                    activeFrame += Self.toolEnd
                    try enforcePendingByteLimit()
                    completedFrameBytes += activeFrame.utf8.count
                    frames.append(activeFrame)
                    activeFrame = ""
                    sawToolCall = true
                    state = .visible
                    continue
                }
                let held = final ? partialMarkerLength(in: pending) : markerSuffixLength(in: pending)
                if final { throw GemmaToolCallParserError.malformed }
                let split = pending.index(pending.endIndex, offsetBy: -held)
                activeFrame += String(pending[..<split])
                pending = String(pending[split...])
                try enforcePendingByteLimit()
                return events
            }
        }
        if final, state != .visible { throw GemmaToolCallParserError.malformed }
        return events
    }

    private mutating func routeVisible(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !text.isEmpty else { return [] }
        if sawToolCall {
            guard text.allSatisfy(\.isWhitespace) else {
                throw GemmaToolCallParserError.malformed
            }
            return []
        }
        return [.content(text)]
    }

    private func enforcePendingByteLimit() throws {
        let retained = completedFrameBytes + activeFrame.utf8.count + pending.utf8.count
        guard retained <= Self.maximumPendingToolBytes else {
            throw GemmaToolCallParserError.oversized
        }
    }

    private mutating func fail() {
        failed = true
        pending = ""
        activeFrame = ""
        frames.removeAll(keepingCapacity: false)
        completedFrameBytes = 0
        releasedCallCount = 0
    }

    private mutating func recordProgress(
        _ text: String,
        startingState: State,
        countsToken: Bool
    ) {
        let tokenStage: StructuredAssistantProgress.Stage
        switch startingState {
        case .thought:
            tokenStage = .thinking
        case .toolCall:
            tokenStage = .toolCall
        case .visible:
            if text.contains(Self.thoughtStart) || state == .thought {
                tokenStage = .thinking
            } else if text.contains(Self.toolStart) || state == .toolCall {
                tokenStage = .toolCall
            } else {
                tokenStage = .visibleResponse
            }
        }
        if countsToken {
            switch tokenStage {
            case .thinking: progress.thinkingTokens += 1
            case .toolCall: progress.toolCallTokens += 1
            case .visibleResponse: progress.visibleResponseTokens += 1
            case .channelLabel: progress.channelLabelTokens += 1
            case .unknownHiddenChannel: progress.unknownHiddenChannelTokens += 1
            }
        }

        if let thought = thoughtFragment(in: text, startingState: startingState),
           !thought.isEmpty {
            appendThoughtPreview(thought)
        }
        refreshToolPreview()
        progress.stage = switch state {
        case .thought: .thinking
        case .toolCall: .toolCall
        case .visible: sawToolCall ? .toolCall : .visibleResponse
        }
    }

    private func thoughtFragment(in text: String, startingState: State) -> String? {
        switch startingState {
        case .thought:
            if let end = text.range(of: Self.thoughtEnd) {
                return String(text[..<end.lowerBound])
            }
            return text
        case .visible, .toolCall:
            guard let start = text.range(of: Self.thoughtStart) else { return nil }
            let remainder = text[start.upperBound...]
            if let end = remainder.range(of: Self.thoughtEnd) {
                return String(remainder[..<end.lowerBound])
            }
            return String(remainder)
        }
    }

    private mutating func appendThoughtPreview(_ text: String) {
        progress.recentThoughtText += text
        let bytes = progress.recentThoughtText.utf8
        guard bytes.count > StructuredAssistantProgress.maximumThoughtPreviewBytes else { return }
        var start = bytes.index(
            bytes.endIndex,
            offsetBy: -StructuredAssistantProgress.maximumThoughtPreviewBytes)
        while bytes[start] & 0xC0 == 0x80 { start = bytes.index(after: start) }
        progress.recentThoughtText = String(decoding: bytes[start...], as: UTF8.self)
        progress.earlierThoughtTextOmitted = true
    }

    private mutating func refreshToolPreview() {
        let draft = frames.joined(separator: "\n") + activeFrame + pending
        guard sawToolCall || state == .toolCall || !activeFrame.isEmpty else {
            progress.toolCallPreview = nil
            return
        }
        let limit = StructuredAssistantProgress.maximumThoughtPreviewBytes
        guard draft.utf8.count > limit else {
            progress.toolCallPreview = StructuredToolCallPreview(
                text: draft, middleTextOmitted: false)
            return
        }
        let bytes = Array(draft.utf8)
        let half = limit / 2
        let head = String(decoding: bytes.prefix(half), as: UTF8.self)
        let tail = String(decoding: bytes.suffix(half), as: UTF8.self)
        progress.toolCallPreview = StructuredToolCallPreview(
            text: head + tail, middleTextOmitted: true)
    }

    private func firstMarker(in text: String) -> (marker: String, range: Range<String.Index>)? {
        Self.markers.compactMap { marker in
            text.range(of: marker).map { (marker, $0) }
        }.min { lhs, rhs in lhs.1.lowerBound < rhs.1.lowerBound }
    }

    private func markerSuffixLength(in text: String) -> Int {
        Self.markers.reduce(0) { result, marker in
            let maximum = min(text.count, marker.count - 1)
            guard maximum > result else { return result }
            for length in stride(from: maximum, through: result + 1, by: -1) {
                if text.suffix(length) == marker.prefix(length) { return length }
            }
            return result
        }
    }

    private func partialMarkerLength(in text: String) -> Int {
        markerSuffixLength(in: text)
    }
}
