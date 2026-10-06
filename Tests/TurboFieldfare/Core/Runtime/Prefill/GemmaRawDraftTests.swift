import Foundation
import Testing
@testable import TurboFieldfare

@Suite struct GemmaRawDraftTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_RAW_DRAFT_MODEL"] != nil))
    func preservesOutputAndStopState() async throws {
        let environment = ProcessInfo.processInfo.environment
        let path = try #require(environment["TURBOFIELDFARE_RAW_DRAFT_MODEL"])
        let pack = try #require(environment["TURBOFIELDFARE_DRAFT_PACK"])
        let directory = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
            streamingMode: .pread(slotCount: 64), integrityPolicy: .sizeCheckTrustedReceipt)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        let text = try tokenizer.applyChatTemplate([
            .init(role: .user, content: "Explain why the sky is blue in clear, simple language.")
        ])
        let prompt = tokenizer.encode(text, addBOS: false)
        let runtime = RuntimeConfiguration(expertCacheSlots: 64, forceLogitsHead: true, gemmaDraftEnabled: false)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 4096,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context, vocab: 262144)
        var ordinaryTokens: [Int32] = []
        for scenario in 0..<8 {
            var config = GenerationConfig(maxNewTokens: [16, 1, 2, 3, 16, 16, 16, 16][scenario],
                temperature: 0.2, topK: 64, topP: 0.95, seed: 20260721)
            if scenario == 4 { config.stopStrings = [" "] }
            if scenario == 5 { config.extraStopTokens = [try #require(ordinaryTokens.dropFirst().first)] }
            if scenario == 7 { config.repetitionPenalty = 1.1 }
            var baseline: RawDecodeResult?
            var baselineTokens: [Int32] = []
            var baselineText = ""
            for enabled in [false, true] {
                try runner.configureGemmaDraft(directory: enabled ? URL(fileURLWithPath: pack) : nil)
                var tokens: [Int32] = []
                var output = ""
                let result = try await runRawCompletion(producer: runner, tokenizer: tokenizer,
                    promptIds: prompt, config: config, context: context, scratch: scratch,
                    prefillConfig: runtime.prefillConfig,
                    shouldStop: { scenario == 6 && tokens.count >= 2 },
                    onProgress: { event in
                        switch event {
                        case .token(_, let id, let delta): tokens.append(id); output += delta
                        case .tail(let tail): output += tail
                        case .prefill: break
                        }
                    })
                #expect(runner.continuationPosition == result.kvPosition)
                if let baseline {
                    #expect(tokens == baselineTokens)
                    #expect(output == baselineText)
                    #expect(result.reason == baseline.reason)
                    #expect(result.newTokens == baseline.newTokens)
                    #expect(result.kvPosition == baseline.kvPosition)
                    #expect(result.kvBackedTokenIDs == baseline.kvBackedTokenIDs)
                    #expect(result.uncommittedBoundaryTokenIDs == baseline.uncommittedBoundaryTokenIDs)
                    #expect(result.withheldTrailingKVTokens == baseline.withheldTrailingKVTokens)
                } else {
                    baseline = result
                    baselineTokens = tokens
                    baselineText = output
                    if scenario == 0 { ordinaryTokens = tokens }
                }
                if scenario == 4 { #expect(result.reason == .stopString) }
                if scenario == 5 { #expect(result.reason == .eos) }
                if scenario == 6 { #expect(result.reason == .cancelled) }
                print("GEMMA_RAW_DRAFT scenario=\(scenario) enabled=\(enabled) tokens=\(result.newTokens) seconds=\(result.decodeSeconds)")
            }
        }

        var resumedReference: [Int32]?
        var resumedState: RawDecodeResult?
        for enabled in [false, true] {
            try runner.configureGemmaDraft(directory: enabled ? URL(fileURLWithPath: pack) : nil)
            var output: [Int32] = []
            let first = try await runRawCompletion(producer: runner, tokenizer: tokenizer,
                promptIds: prompt,
                config: GenerationConfig(maxNewTokens: 3, temperature: 0.2, topK: 64, topP: 0.95, seed: 20260721),
                context: context, scratch: scratch, prefillConfig: runtime.prefillConfig,
                onProgress: { if case .token(_, let id, _) = $0 { output.append(id) } })
            let continuedPrompt = first.kvBackedTokenIDs + first.uncommittedBoundaryTokenIDs
                + tokenizer.encode("\nContinue.", addBOS: false)
            let result = try await runRawCompletion(producer: runner, tokenizer: tokenizer,
                promptIds: continuedPrompt,
                config: GenerationConfig(maxNewTokens: 16, temperature: 0.2, topK: 64, topP: 0.95, seed: 20260721),
                context: context, scratch: scratch, prefillConfig: runtime.prefillConfig,
                start: .resume(cachedPromptTokens: first.kvPosition),
                onProgress: { if case .token(_, let id, _) = $0 { output.append(id) } })
            #expect(runner.continuationPosition == result.kvPosition)
            if let resumedReference, let resumedState {
                #expect(output == resumedReference)
                #expect(result.kvBackedTokenIDs == resumedState.kvBackedTokenIDs)
                #expect(result.uncommittedBoundaryTokenIDs == resumedState.uncommittedBoundaryTokenIDs)
                #expect(result.reason == resumedState.reason)
                #expect(result.kvPosition == resumedState.kvPosition)
            } else {
                resumedReference = output
                resumedState = result
            }
            print("GEMMA_RAW_DRAFT continuation enabled=\(enabled) tokens=\(result.newTokens)")
        }
    }
}
