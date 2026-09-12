import Foundation
import Testing

@testable import TurboFieldfare

@Suite("Multimodal conversation canonicalization")
struct MultimodalConversationCanonicalizationTests {
    @Test("Interleaved user parts become text and image placeholders in order")
    func interleavedUserPartsPreserveOrder() throws {
        let content = try MultimodalConversationCanonicalization.userContent([
            .text("before "),
            .image,
            .text("between "),
            .image,
            .text("after"),
        ])

        #expect(content == "before <|image|>between <|image|>after")
    }

    @Test("User text cannot reserve the image placeholder")
    func reservedImagePlaceholderIsRejected() {
        #expect(throws: MultimodalPromptRendererError.reservedImageMarker) {
            try MultimodalConversationCanonicalization.userContent([
                .text("typed <|image|> marker"),
            ])
        }
    }

    @Test("Reusable prefix stops at the first token mismatch")
    func longestReusablePrefixStopsAtMismatch() {
        let count = MultimodalConversationCanonicalization.longestReusablePrefix(
            cached: [10, 20, 90, 40],
            canonical: [10, 20, 30, 40],
            imageTokenRanges: [])

        #expect(count == 2)
    }

    @Test("Reusable prefix keeps the common prefix when there are no images")
    func longestReusablePrefixUsesCommonPrefix() {
        let count = MultimodalConversationCanonicalization.longestReusablePrefix(
            cached: [10, 20, 30],
            canonical: [10, 20, 30, 40],
            imageTokenRanges: [])

        #expect(count == 3)
    }

    @Test("Prefix ending in projected image rows backs to the image boundary")
    func prefixInsideImageRowsBacksToImageBoundary() {
        let canonical: [Int32] = [
            10, 20, 30,
            MultimodalPromptRenderer.beginImageTokenID,
            MultimodalPromptRenderer.imageTokenID,
            MultimodalPromptRenderer.imageTokenID,
            MultimodalPromptRenderer.imageTokenID,
            MultimodalPromptRenderer.endImageTokenID,
            40,
        ]
        let cached = Array(canonical.prefix(6))

        let count = MultimodalConversationCanonicalization.longestReusablePrefix(
            cached: cached,
            canonical: canonical,
            imageTokenRanges: [4..<7])

        // The begin-image marker at index 3 remains reusable. No projected
        // image row may be treated as ordinary text-cache state.
        #expect(count == 4)
    }

    @Test("Prefix after a complete image remains reusable")
    func prefixAfterCompleteImageRemainsReusable() {
        let canonical: [Int32] = [
            10, 20, 30,
            MultimodalPromptRenderer.beginImageTokenID,
            MultimodalPromptRenderer.imageTokenID,
            MultimodalPromptRenderer.imageTokenID,
            MultimodalPromptRenderer.endImageTokenID,
            40, 50,
        ]

        let count = MultimodalConversationCanonicalization.longestReusablePrefix(
            cached: canonical,
            canonical: canonical,
            imageTokenRanges: [4..<6])

        #expect(count == canonical.count)
    }

    @Test("Assistant history keeps visible text alongside tool calls")
    func assistantMessagePreservesVisibleContentWithToolCall() {
        let call = ParsedToolCall(
            id: "call_1",
            name: "inspect",
            arguments: .object(["topic": .string("bookmarks")]),
            argumentsJSON: "{\"topic\":\"bookmarks\"}")
        let turn = MultimodalTurnResult(
            text: "visible answer",
            promptTokens: 12,
            cachedTokens: 0,
            computedPrefillTokens: 12,
            kvTokens: 12,
            completionTokens: 4,
            reason: .toolCalls,
            prefillSeconds: 0,
            decodeSeconds: 0)
        let message = MultimodalConversation.assistantMessage(
            for: StructuredConversationTurnResult(turn: turn, toolCalls: [call]))

        #expect(message.role == .assistant)
        #expect(message.content == "visible answer")
        #expect(message.toolCalls.count == 1)
        #expect(message.toolCalls[0].id == call.id)
        #expect(message.toolCalls[0].name == call.name)
    }

    @Test("Full tool history keeps visible fields and strips historical thought text")
    func fullToolHistoryCanonicalizesVisibleContent() async throws {
        let tokenizer = try await GFTokenizer.load()
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_1",
            name: "inspect",
            arguments: .object(["topic": .string("bookmarks")]))
        let tool = GFTokenizer.FunctionDefinition(
            name: "inspect",
            description: "Inspect the current app screen.",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "topic": .object(["type": .string("string")]),
                ]),
            ]))
        let messages = [
            GFTokenizer.Message(role: .system, content: "system marker"),
            GFTokenizer.Message(role: .user, content: "user marker"),
            GFTokenizer.Message(
                role: .assistant,
                content: "<|channel>thought\nhistorical private thought<channel|>visible tool answer",
                toolCalls: [call]),
            GFTokenizer.Message(
                role: .tool,
                content: "tool result marker",
                toolCallID: call.id,
                name: call.name),
            GFTokenizer.Message(role: .assistant, content: "final visible marker"),
        ]

        let ids = try tokenizer.encodeToolChat(messages: messages, tools: [tool])
        let rendered = tokenizer.decode(ids, skipSpecialTokens: false)

        #expect(rendered.contains("system marker"))
        #expect(rendered.contains("user marker"))
        #expect(rendered.contains("call:inspect"))
        #expect(rendered.contains("tool result marker"))
        #expect(rendered.contains("visible tool answer"))
        #expect(rendered.contains("final visible marker"))
        #expect(!rendered.contains("historical private thought"))
    }
}
