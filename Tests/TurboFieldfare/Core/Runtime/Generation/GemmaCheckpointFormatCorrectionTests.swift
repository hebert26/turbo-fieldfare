import Testing
@testable import TurboFieldfare

@Suite struct GemmaCheckpointFormatCorrectionTests {
    @Test func correctionPreservesCheckpointAndImageMarkers() throws {
        let original = [
            GFTokenizer.Message(role: .system, content: "Tool contract"),
            GFTokenizer.Message(role: .user, content: "Completed actions and current choices.\n<|image|>"),
        ]
        let corrected = try GemmaMultimodalConversation.correctingCheckpointToolFormat(original)
        #expect(corrected.count == original.count)
        #expect(corrected[0] == original[0])
        #expect(corrected[1].content?.hasPrefix(original[1].content!) == true)
        #expect(corrected[1].content?.contains("action:<|\"|>observe<|\"|>") == true)
        #expect(corrected[1].toolImageCount == original[1].toolImageCount)
        #expect(try GemmaMultimodalConversation.correctingCheckpointToolFormat(corrected) == corrected)
    }

    @Test func correctionRejectsAnInvalidCheckpointBoundary() {
        #expect(throws: MultimodalConversationError.self) {
            try GemmaMultimodalConversation.correctingCheckpointToolFormat([
                GFTokenizer.Message(role: .assistant, content: "unfinished")])
        }
    }

    @Test func retainedMalformedDraftIsRejectedAndCorrectStringFormatParses() throws {
        let parser = GemmaToolCallParser()
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse("call:visioncapture_navigate{action:tap,target:c22",
                allowedTools: ["visioncapture_navigate"], id: "invalid")
        }
        let corrected = try parser.parse(
            "call:visioncapture_navigate{action:<|\"|>tap<|\"|>,target:<|\"|>c22<|\"|>}",
            allowedTools: ["visioncapture_navigate"], id: "corrected")
        #expect(corrected.arguments == .object(["action": .string("tap"), "target": .string("c22")]))
    }
}
