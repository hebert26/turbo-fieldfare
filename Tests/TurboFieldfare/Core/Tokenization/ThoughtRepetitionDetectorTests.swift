import Foundation
import Testing
@testable import TurboFieldfare

/// The three real thinking loops of 8 October 2026 and seven texts that must not
/// trigger, as token IDs from the model's tokenizer (ThoughtRepetitionFixtures).
@Suite struct ThoughtRepetitionDetectorTests {
    /// The token count at which the detector first reports, the reported width and copies.
    private func firstDetection(_ tokens: [Int32]) -> (tokens: Int, width: Int, copies: Int)? {
        var detector = ThoughtRepetitionDetector()
        for (index, token) in tokens.enumerated() {
            if let hit = detector.append(token) { return (index + 1, hit.blockTokens, hit.repetitions) }
        }
        return nil
    }

    @Test
    func rerunStep10LoopIsCaughtAsAShortPeriod() {
        // "The header is `c119`." plus a newline: 10 tokens, below today's 16-token minimum.
        let hit = firstDetection(ThoughtRepetitionFixtures.loopShortStep10)
        #expect(hit?.tokens == 160)
        #expect(hit?.width == 10)
        // The exact run is all 160 tokens: 16 copies.
        #expect(hit?.copies == 16)
    }

    @Test
    func run32Step39LoopIsCaughtAsALongBlock() {
        let hit = firstDetection(ThoughtRepetitionFixtures.loopRun32Step39)
        #expect(hit?.tokens == 556)
        #expect(hit?.width == 138)
        #expect(hit?.copies == 4)
    }

    @Test
    func muckTasksStep15LoopIsCaughtAsALongBlock() {
        let hit = firstDetection(ThoughtRepetitionFixtures.loopMuckTasksStep15)
        #expect(hit?.tokens == 1_236)
        #expect(hit?.width == 308)
        #expect(hit?.copies == 4)
    }

    @Test
    func normalTextDoesNotTrigger() {
        let negatives: [(String, [Int32])] = [
            ("probe 2 step 7 thinking", ThoughtRepetitionFixtures.normalProbe2Step7Thinking),
            ("run 4 final answer", ThoughtRepetitionFixtures.normalRun4FinalAnswer),
            ("probe 2 choice list", ThoughtRepetitionFixtures.builtChoiceListProbe2),
            ("probe 2 packet JSON", ThoughtRepetitionFixtures.realPacketJSONProbe2),
            ("system prompt", ThoughtRepetitionFixtures.realSystemPrompt),
            ("probe 2 chat transcript", ThoughtRepetitionFixtures.stressProbe2Chat),
            ("MuckTasks chat transcript", ThoughtRepetitionFixtures.stressMuckTasksChat),
        ]
        for (name, tokens) in negatives {
            #expect(firstDetection(tokens) == nil, "\(name) (\(tokens.count) tokens) triggered")
        }
    }

    @Test
    func todaysRuleStillReportsAMediumBlock() {
        // A 20-token block with 20 distinct tokens, eight copies: today's rule, width 20.
        let tokens = (0..<200).map { Int32($0 % 20 + 1_000) }
        let hit = firstDetection(tokens)
        #expect(hit?.tokens == 160)
        #expect(hit?.width == 20)
        #expect(hit?.copies == 8)
    }

    @Test
    func costPerTokenComparedWithTodaysDetector() {
        // Worst case: one token repeated, broken every 159 tokens by a new token. Every
        // short period and every block width matches up to the break; nothing is reported.
        let worst = (0..<20_000).map { $0 % 159 == 158 ? Int32(100_000 + $0) : 1 }
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        let random = (0..<20_000).map { _ -> Int32 in
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int32(truncatingIfNeeded: (seed >> 33) % 262_144)
        }
        func nanosecondsPerToken(_ tokens: [Int32], _ append: (Int32) -> Bool) -> Double {
            let clock = ContinuousClock()
            var reported = 0
            let elapsed = clock.measure { for token in tokens where append(token) { reported += 1 } }
            #expect(reported == 0)
            let (seconds, attoseconds) = elapsed.components
            return (Double(seconds) * 1e9 + Double(attoseconds) / 1e9) / Double(tokens.count)
        }
        var results: [String: Double] = [:]
        for (name, tokens) in [("worst", worst), ("random", random)] {
            var current = ThoughtRepetitionDetector()
            var today = TodaysThoughtRepetitionDetector()
            results["new " + name] = nanosecondsPerToken(tokens) { current.append($0) != nil }
            results["today " + name] = nanosecondsPerToken(tokens) { today.append($0) != nil }
        }
        print("ThoughtRepetitionDetector cost, ns per appended token:",
              results.sorted { $0.key < $1.key }.map { "\($0.key) \(Int($0.value))" }.joined(separator: ", "))
        // A debug build only catches a blow-up here. The cost proof is the optimized
        // benchmark: at most 50 µs per token in the worst case; decode takes about 30 ms.
        #expect(results["new worst"]! < 5_000_000)
    }
}

/// Today's detector before the short-period and long-block rules (commit 6aae129),
/// kept only to compare the cost per token.
private struct TodaysThoughtRepetitionDetector {
    static let capacity = 1_024
    static let repetitions = 8
    static let minimumBlock = 16
    static let maximumBlock = 128
    private var ring = [Int32](repeating: 0, count: capacity)
    private var count = 0
    private var next = 0

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
