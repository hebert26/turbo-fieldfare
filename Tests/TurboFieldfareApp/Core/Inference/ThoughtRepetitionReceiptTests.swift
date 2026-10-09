import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// The app's check of a repeated-thought receipt from the decode service.
@Suite struct ThoughtRepetitionReceiptTests {
    private func receipt(block: Int, copies: Int, thinking: Int, generated: Int? = nil) throws -> ThoughtRepetitionRecovery {
        let json = """
            {"canRetryToolResult": true, "pendingCallID": "call_1", "pendingToolName": "visioncapture_navigate",
             "restoredTokenCount": 9000, "requiresRebuild": false, "generatedTokens": \(generated ?? thinking + 40),
             "thinkingTokens": \(thinking), "blockTokens": \(block), "repetitions": \(copies)}
            """
        return try JSONDecoder().decode(ThoughtRepetitionRecovery.self, from: Data(json.utf8))
    }

    private func admits(_ receipt: ThoughtRepetitionRecovery) -> Bool {
        DecodeServiceInferenceClient.admitsThoughtRepetitionRecovery(receipt, maxContextTokens: 32_768)
    }

    @Test
    func acceptsEveryDetectorRule() throws {
        // Short period (rerun step 10): 10 tokens, 16 copies in 160 thinking tokens.
        #expect(admits(try receipt(block: 10, copies: 16, thinking: 160)))
        #expect(admits(try receipt(block: 10, copies: 12, thinking: 120)))
        // Long blocks: run 32 step 39 (138 x 4) and MuckTasks step 15 (308 x 4).
        #expect(admits(try receipt(block: 138, copies: 4, thinking: 556)))
        #expect(admits(try receipt(block: 308, copies: 4, thinking: 1_236)))
        // Today's rule: 20 tokens, 8 copies.
        #expect(admits(try receipt(block: 20, copies: 8, thinking: 160)))
        // Rule edges: 1 and 512 tokens.
        #expect(admits(try receipt(block: 1, copies: 160, thinking: 160)))
        #expect(admits(try receipt(block: 512, copies: 4, thinking: 2_048)))
    }

    @Test
    func refusesAnImpossibleReceipt() throws {
        // Fewer thinking tokens than the reported copies hold.
        #expect(!admits(try receipt(block: 308, copies: 4, thinking: 1_231)))
        #expect(!admits(try receipt(block: 10, copies: 16, thinking: 159)))
        // Outside every rule.
        #expect(!admits(try receipt(block: 0, copies: 8, thinking: 160)))
        #expect(!admits(try receipt(block: 513, copies: 4, thinking: 2_052)))
        #expect(!admits(try receipt(block: 138, copies: 3, thinking: 556)))
        // A copy count that overflows the product is refused, not trapped.
        #expect(!admits(try receipt(block: 512, copies: Int.max, thinking: 30_000)))
        // The other checks stay: more thinking than generated tokens.
        #expect(!admits(try receipt(block: 10, copies: 16, thinking: 160, generated: 159)))
    }
}
