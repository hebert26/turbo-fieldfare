import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite struct GemmaDraftRunnerTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_COMBINED_MODEL"] != nil))
    func combinedDraftKeepsSampledTokens() async throws {
        let environment = ProcessInfo.processInfo.environment
        let path = try #require(environment["TURBOFIELDFARE_COMBINED_MODEL"])
        let pack = try #require(environment["TURBOFIELDFARE_DRAFT_PACK"])
        let directory = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   streamingMode: .pread(slotCount: 64),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        let prompt = try tokenizer.applyChatTemplate([
            .init(role: .user, content: "Explain why the sky is blue in clear, simple language.")
        ])
        let tokens = tokenizer.encode(prompt, addBOS: false)
        let configuration = RuntimeConfiguration(expertCacheSlots: 64, forceLogitsHead: true)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 4096,
                                            runtimeConfiguration: configuration)
        let draftWeights = try GemmaDraftWeights(directory: URL(fileURLWithPath: pack), device: context.device)
        let draft = try GemmaDraftRunner(context: context, weights: draftWeights)
        let sampler = try Sampler(context: context)
        let logits = try #require(context.device.makeBuffer(length: 262144 * 2, options: .storageModeShared))
        let probabilities = try #require(context.device.makeBuffer(length: 262144 * 2, options: .storageModeShared))
        let sampled = try #require(context.device.makeBuffer(length: 4, options: .storageModeShared))
        let config = GenerationConfig(maxNewTokens: 128, temperature: 0.2, topK: 64, topP: 0.95, seed: 20260721)
        func sample(_ scores: MTLBuffer, history: [Int32], position: Int) throws -> Int32 {
            let command = try #require(context.queue.makeCommandBuffer())
            sampler.sample(commandBuffer: command, logits: scores, probs: probabilities,
                history: history, config: config, position: position, outToken: sampled)
            command.commit()
            command.waitUntilCompleted()
            try checkCommandBufferError(command)
            return Int32(sampled.contents().load(as: UInt32.self))
        }
        var reference: [Int32] = []
        var reports: [[String: Any]] = []
        for mode in [false, true] {
            runner.reset()
            _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0,
                outputMode: .logits, config: configuration.prefillConfig, into: logits, onProgress: { _ in })
            var history = tokens
            let start = DispatchTime.now().uptimeNanoseconds
            var next = try sample(logits, history: history, position: 0)
            var result = [next]
            var rounds = 0
            var accepted = 0
            var draftSeconds = 0.0
            var verifySeconds = 0.0
            let limit = mode ? reference.count : config.maxNewTokens
            while result.count < limit && !tokenizer.stopTokenIDs.contains(next) {
                history.append(next)
                let count = min(4, limit - result.count - 1)
                if mode && count > 0 {
                    let draftStart = DispatchTime.now().uptimeNanoseconds
                    let proposed = try draft.draft(after: next, count: count, using: runner.makeDraftContext())
                    draftSeconds += Double(DispatchTime.now().uptimeNanoseconds - draftStart) / 1e9
                    let verifyStart = DispatchTime.now().uptimeNanoseconds
                    let scores = try await runner.verifyDraft(tokens: [next] + proposed)
                    verifySeconds += Double(DispatchTime.now().uptimeNanoseconds - verifyStart) / 1e9
                    var keep = 0
                    for row in scores.indices {
                        next = try sample(scores[row], history: history, position: result.count)
                        result.append(next)
                        keep = row + 1
                        if result.count == limit || tokenizer.stopTokenIDs.contains(next) { break }
                        guard row < proposed.count, next == proposed[row] else { break }
                        accepted += 1
                        history.append(next)
                    }
                    try runner.acceptVerifiedPrefix(count: keep)
                    rounds += 1
                } else {
                    try await runner.produce(token: next, position: runner.continuationPosition, into: logits)
                    next = try sample(logits, history: history, position: result.count)
                    result.append(next)
                }
            }
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            #expect(runner.continuationPosition == tokens.count + result.count - 1)
            if mode { #expect(result == reference) } else { reference = result }
            let contentCount = result.filter { !tokenizer.stopTokenIDs.contains($0) }.count
            reports.append(["draft_enabled": mode, "content_tokens": contentCount,
                            "seconds": seconds, "tokens_per_second": Double(contentCount) / seconds,
                            "rounds": rounds, "accepted_proposals": accepted,
                            "draft_seconds": draftSeconds, "verify_seconds": verifySeconds])
        }
        print("GEMMA_COMBINED_PROBE " + String(decoding: try JSONSerialization.data(
            withJSONObject: reports, options: [.prettyPrinted, .sortedKeys]), as: UTF8.self))
    }


    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_VERIFY_MODEL"] != nil))
    func verificationMatchesSequential() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["TURBOFIELDFARE_VERIFY_MODEL"])
        let directory = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   streamingMode: .pread(slotCount: 64),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        let text = try tokenizer.applyChatTemplate([
            .init(role: .user, content: "Explain why the sky is blue in clear, simple language.")
        ])
        let tokens = tokenizer.encode(text, addBOS: false)
        let configuration = RuntimeConfiguration(expertCacheSlots: 64, forceLogitsHead: true)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 4096,
                                            runtimeConfiguration: configuration)
        let logits = try #require(context.device.makeBuffer(length: 262144 * 2, options: .storageModeShared))
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .logits, config: configuration.prefillConfig, into: logits, onProgress: { _ in })
        func argmax(_ buffer: MTLBuffer) -> Int32 {
            let values = buffer.contents().assumingMemoryBound(to: Float16.self)
            var best = 0
            for index in 1..<262144 where values[index] > values[best] { best = index }
            return Int32(best)
        }
        var next = argmax(logits)
        for round in 0..<3 {
            let start = runner.continuationPosition
            var inputs: [Int32] = []
            var expected: [Data] = []
            let referenceStart = DispatchTime.now().uptimeNanoseconds
            for _ in 0..<5 {
                inputs.append(next)
                try await runner.produce(token: next, position: runner.continuationPosition, into: logits)
                expected.append(Data(bytes: logits.contents(), count: logits.length))
                next = argmax(logits)
            }
            let referenceSeconds = Double(DispatchTime.now().uptimeNanoseconds - referenceStart) / 1e9
            try runner.rewind(to: start)
            let batchStart = DispatchTime.now().uptimeNanoseconds
            let actual = try await runner.verifyDraft(tokens: inputs)
            let batchSeconds = Double(DispatchTime.now().uptimeNanoseconds - batchStart) / 1e9
            for index in 0..<5 {
                #expect(Data(bytes: actual[index].contents(), count: actual[index].length) == expected[index])
            }
            let kept = round + 1
            try runner.acceptVerifiedPrefix(count: kept)
            #expect(runner.continuationPosition == start + kept)
            _ = try runner.makeDraftContext()
            try await runner.produce(token: inputs[kept], position: runner.continuationPosition, into: logits)
            #expect(Data(bytes: logits.contents(), count: logits.length) == expected[kept])
            next = argmax(logits)
            print("GEMMA_VERIFY_PROBE round=\(round) sequential_seconds=\(referenceSeconds) batch_seconds=\(batchSeconds) kept=\(kept)")
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_DRAFT_MODEL"] != nil))
    func comparesDraftWithTarget() async throws {
        let environment = ProcessInfo.processInfo.environment
        let modelPath = try #require(environment["TURBOFIELDFARE_DRAFT_MODEL"])
        let packPath = try #require(environment["TURBOFIELDFARE_DRAFT_PACK"])
        let directory = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let context = try MetalContext()
        let weights = try GemmaDraftWeights(directory: URL(fileURLWithPath: packPath), device: context.device)
        let draft = try GemmaDraftRunner(context: context, weights: weights)
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   streamingMode: .pread(slotCount: 64),
                                   integrityPolicy: .sizeCheckTrustedReceipt)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: directory)
        let prompt = try tokenizer.applyChatTemplate([
            .init(role: .user, content: "Explain why the sky is blue in clear, simple language.")
        ])
        let tokens = tokenizer.encode(prompt, addBOS: false)
        let configuration = RuntimeConfiguration(expertCacheSlots: 64)
        let runner = try RealForwardRunner(model: model, context: context, maxContext: 4096,
                                            runtimeConfiguration: configuration)
        let logits = try #require(context.device.makeBuffer(length: 262144 * 2, options: .storageModeShared))
        _ = try await runner.prefillChunked(tokens: tokens[...], startPosition: 0,
            outputMode: .greedyIfAvailable, config: configuration.prefillConfig,
            into: logits, onProgress: { _ in })
        var token = Int32(runner.lastGreedyToken)
        var generated = [token]
        var records: [[String: Any]] = []
        for round in 0..<8 {
            let state = try runner.makeDraftContext()
            func cacheHash() -> String {
                var hash = SHA256()
                for buffer in [state.sliding.keys, state.sliding.values, state.full.keys, state.full.values] {
                    hash.update(data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none))
                }
                return hash.finalize().map { String(format: "%02x", $0) }.joined()
            }
            let before = cacheHash()
            let start = DispatchTime.now().uptimeNanoseconds
            let proposals = try draft.draft(after: token, count: 4, using: state)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            #expect(cacheHash() == before)
            var target: [Int32] = []
            for _ in 0..<4 {
                try await runner.produce(token: token, position: runner.continuationPosition, into: logits)
                token = Int32(runner.lastGreedyToken)
                target.append(token)
                generated.append(token)
            }
            let matches = zip(proposals, target).prefix(while: { $0 == $1 }).count
            records.append(["round": round, "draft": proposals, "target": target,
                            "prefix_matches": matches, "draft_seconds": elapsed,
                            "draft_gpu_seconds": draft.lastGPUSeconds,
                            "draft_text": tokenizer.decode(proposals), "target_text": tokenizer.decode(target)])
        }
        let report: [String: Any] = ["rounds": records, "target_text": tokenizer.decode(generated)]
        let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print("GEMMA_DRAFT_PROBE " + String(decoding: json, as: UTF8.self))
    }
}
