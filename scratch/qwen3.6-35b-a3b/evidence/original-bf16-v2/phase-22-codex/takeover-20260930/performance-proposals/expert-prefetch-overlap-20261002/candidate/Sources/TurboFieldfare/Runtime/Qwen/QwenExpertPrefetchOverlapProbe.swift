import Foundation
import Metal
import CryptoKit
import TurboFieldfareOfficialQwenSource

private struct EarlyPrefetchProbeError: Error { let detail: String }
private func requireEarlyPrefetch(_ condition: Bool, _ detail: String) throws {
    if !condition { throw EarlyPrefetchProbeError(detail: detail) }
}

// Source-copy command only. It publishes no parser, tools or conversation state.
public enum QwenExpertPrefetchOverlapProbe {
    @MainActor public static func run(registrationURL: URL, requestURL: URL, enabled: Bool) async throws -> Data {
        let raw = try Data(contentsOf: requestURL)
        let digest = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        try requireEarlyPrefetch(digest == "841f885f3e7e30c56656176e2cb41b4616f01d88db16f30f03ab058f125e1285", "capture hash")
        guard let request = try JSONSerialization.jsonObject(with: raw) as? [String: Any],
              let prefixInts = request["inputTokenIDs"] as? [Int],
              let recordedInts = request["sampledTokenIDs"] as? [Int],
              let settings = request["settings"] as? [String: Any],
              let stops = settings["extraStopTokens"] as? [Int],
              let strings = settings["stopStrings"] as? [String],
              let temperature = settings["temperature"] as? NSNumber,
              let penalty = settings["repetitionPenalty"] as? NSNumber,
              let maxNew = settings["maxNewTokens"] as? Int,
              let maxContext = request["maximumContext"] as? Int,
              let eosInt = request["codecEOS"] as? Int else {
            throw EarlyPrefetchProbeError(detail: "capture fields")
        }
        try requireEarlyPrefetch(prefixInts.count == 1175 && recordedInts.count == 29
            && request["complete"] as? Bool == true && request["terminal"] as? String == "committed"
            && request["fixtureSamples"] as? Bool == false && request["checkpointResume"] as? Bool == false
            && request["preparedPromptPresent"] as? Bool == false && request["visionStorePresent"] as? Bool == false
            && maxContext == 8192 && prefixInts.allSatisfy { Int32(exactly: $0) != nil }
            && recordedInts.allSatisfy { Int32(exactly: $0) != nil }, "frozen capture geometry")
        let prefix = prefixInts.map { Int32($0) }
        let recorded = recordedInts.map { Int32($0) }
        let eos = Int32(eosInt)
        var config = GenerationConfig(maxNewTokens: maxNew,
            temperature: Float(temperature.doubleValue), topK: settings["topK"] as? Int,
            topP: (settings["topP"] as? NSNumber).map { Float($0.doubleValue) },
            repetitionPenalty: Float(penalty.doubleValue), seed: nil,
            stopStrings: strings, extraStopTokens: Set(stops.map { Int32($0) }))
        config.logitTransform = .raw
        try config.validate()
        try requireEarlyPrefetch(config.temperature == 0 && config.extraStopTokens.contains(eos)
            && recorded.last == eos, "greedy terminal capture")
        var minimumFree = try memoryGuard()
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.load(registrationURL: registrationURL, context: context,
            integrityPolicy: .sizeCheckTrustedReceipt, expertCacheSlots: 16,
            expertCachePolicy: .lfu, residencyBudgetBytes: 12 * 1024 * 1024 * 1024)
        guard let source = model.sourceIdentity else { throw EarlyPrefetchProbeError(detail: "source identity") }
        try requireEarlyPrefetch(source.descriptorContentSHA256 == (request["sourceDescriptorSHA256"] as? String)
            && source.markerSHA256 == (request["sourceMarkerSHA256"] as? String)
            && source.checksumManifestSHA256 == (request["sourceManifestSHA256"] as? String)
            && source.shardSetSHA256 == (request["sourceShardSetSHA256"] as? String), "official source mismatch")
        try requireEarlyPrefetch(model.architecture.layers == 40 && model.architecture.hiddenSize == 2048 && model.architecture.experts == 256,
                              "bounded architecture")
        let capture = QwenEarlyPrefetchCapture()
        let allocationStart = QwenProductionTimingMeasurement.uptime()
        // Both modes allocate and touch the same bounded quarantine footprint.
        let pool = try QwenProtectedEarlyPrefetch(model: model, enabled: enabled, capture: capture)
        let allocationNanoseconds = QwenProductionTimingMeasurement.uptime() - allocationStart
        minimumFree = min(minimumFree, try memoryGuard())
        let runner = try QwenOfficialSourceRunner(model: model, maxContext: maxContext,
            expertSlotCount: 16, hooks: .none, earlyPrefetch: pool, earlyCapture: capture)
        let scratch = try RawCompletionScratch(context: context, vocab: model.architecture.vocabularySize)
        let prefillStart = QwenProductionTimingMeasurement.uptime()
        var logits = try await runner.prefill(tokenIDs: prefix, position: 0)
        let prefillEnd = QwenProductionTimingMeasurement.uptime()
        minimumFree = min(minimumFree, try memoryGuard())
        capture.begin()
        var history = prefix
        var generated: [Int32] = []
        var forwardTimes: [UInt64] = []
        var foundStop = false
        let decodeStart = QwenProductionTimingMeasurement.uptime()
        do {
            // Recorded IDs only check outputs. They never supply decode inputs.
            for index in 0..<min(config.maxNewTokens, 64) {
                try Task.checkCancellation()
                try QwenSourceSamplerBoundary.publishFP32Logits(logits, into: scratch.logits,
                    vocabularySize: model.architecture.vocabularySize)
                guard let command = context.queue.makeCommandBuffer() else {
                    throw EarlyPrefetchProbeError(detail: "sampler command")
                }
                scratch.sampler.sample(commandBuffer: command, logits: scratch.logits, probs: scratch.probs,
                    history: history, config: config, position: index, outToken: scratch.outToken)
                try Task.checkCancellation()
                command.commit()
                await command.completed()
                try requireEarlyPrefetch(command.status == .completed && command.error == nil, "sampler GPU error")
                let sampled = scratch.outToken.contents().load(as: UInt32.self)
                try requireEarlyPrefetch(sampled < UInt32(model.architecture.vocabularySize), "invalid sampled ID")
                let token = Int32(sampled)
                try requireEarlyPrefetch(index < recorded.count && token == recorded[index], "sample differs at index \(index)")
                generated.append(token)
                if config.extraStopTokens.contains(token) { foundStop = true; break }
                let start = QwenProductionTimingMeasurement.uptime()
                logits = try await runner.produce(token: token, position: history.count)
                forwardTimes.append(QwenProductionTimingMeasurement.uptime() - start)
                history.append(token)
            }
            try await pool.cancelAndDrain()
        } catch {
            var failure: any Error = error
            do { try await pool.cancelAndDrain() }
            catch { failure = QwenProductionAndQuarantineFailure(primary: failure, speculative: error) }
            throw failure
        }
        let decodeEnd = QwenProductionTimingMeasurement.uptime()
        try requireEarlyPrefetch(foundStop && generated == recorded && forwardTimes.count == 28,
                                 "incomplete frozen generation")
        let finalPosition = await runner.position
        let usable = !(await runner.isUnusable)
        let state = capture.snapshot()
        try requireEarlyPrefetch(finalPosition == 1203 && usable && pool.idle(), "terminal runner/quarantine state")
        try requireEarlyPrefetch(!state.duplicate && state.routes.count == 1120 && state.plans.count == 1120,
                                 "missing or duplicate decode evidence")
        for position in 1175..<1203 {
            for layer in 0..<40 {
                let key = QwenEarlyPrefetchCapture.Key(position: position, layer: layer)
                guard let route = state.routes[key], let plan = state.plans[key] else {
                    throw EarlyPrefetchProbeError(detail: "missing route/plan")
                }
                let residents = Set(plan.residents.filter { $0 >= 0 })
                try requireEarlyPrefetch(route.experts == plan.experts && route.experts.count == 8
                    && route.weightBits.count == 8 && Set(route.experts).count == 8
                    && plan.residents.count == 16 && plan.assignedSlots.count == 8
                    && Set(plan.assignedSlots).count == 8 && plan.assignedSlots.allSatisfy { (0..<16).contains($0) }
                    && plan.residents.allSatisfy { $0 == -1 || (0..<256).contains($0) }
                    && residents.count == plan.residents.filter { $0 >= 0 }.count
                    && Set(plan.missIndices).count == plan.missIndices.count
                    && plan.missIndices.allSatisfy { plan.experts.indices.contains($0) }, "route/plan shape")
                try requireEarlyPrefetch(Set(plan.missIndices.map { plan.experts[$0] })
                    == Set(plan.experts).subtracting(residents), "actual miss membership")
            }
        }
        let order: (QwenEarlyPrefetchCapture.Key, QwenEarlyPrefetchCapture.Key) -> Bool = {
            $0.position == $1.position ? $0.layer < $1.layer : $0.position < $1.position
        }
        let routes = state.routes.keys.sorted(by: order).map { state.routes[$0]! }
        let plans = state.plans.keys.sorted(by: order).map { state.plans[$0]! }
        var counters: [String: Any] = [:]
        for site in QwenEarlyPrefetchCapture.Site.allCases {
            let counter = state.counters[site] ?? .init()
            try requireEarlyPrefetch(counter.failures == 0, "failed safety/read counter")
            counters[site.rawValue] = try json(counter)
        }
        let hints = state.counters[.hint]?.count ?? 0
        try requireEarlyPrefetch(hints == (enabled ? 1092 : 0), "fixed early hint count")
        minimumFree = min(minimumFree, try memoryGuard())
        let result: [String: Any] = ["schema": 1, "mode": enabled ? "on" : "off",
            "captureSHA256": digest, "outputIDs": generated.map(Int.init), "expectedOutputIDsMatched": true,
            "promptWallNanoseconds": prefillEnd - prefillStart,
            "decodeWallNanoseconds": decodeEnd - decodeStart, "forwardWallNanoseconds": forwardTimes,
            "routes": try json(routes), "plans": plans.map { plan -> [String: Any] in
                ["position": plan.position, "layer": plan.layer, "experts": plan.experts,
                 "assignedSlots": plan.assignedSlots, "missIndices": plan.missIndices,
                 "residents": plan.residents.map { value -> Any in value < 0 ? NSNull() : NSNumber(value: value) }]
            }, "counters": counters,
            "quarantineBytes": pool.allocatedBytes, "quarantineAllocationTouchNanoseconds": allocationNanoseconds,
            "speculativePairs": state.speculativePairs, "importedPairs": state.importedPairs,
            "unusedPairs": state.unusedPairs, "unusedBytes": state.unusedPairs * 6 * 1024 * 1024,
            "minimumFreePercent": minimumFree,
            "safety": ["captureComplete": true, "quarantineIdle": pool.idle(), "runnerUsable": usable,
                       "position": finalPosition, "pendingTokens": 1],
            "rollbackExercised": false, "terminalEOSLeftPending": true,
            "timingLimits": "Decode includes all29 samples and28 forwards. Per-forward timings are secondary. Map includes drain, hits, demand reads and imports. Protected read timings include their source checks. Worker sums overlap and are not critical-path elapsed. Full-source scans are separate. Before-copy and original publication checks are retained. Counters are requested logical bytes, not physical disk bytes.",
            "scope": "Isolated source-copy experiment only. Fixed EARLY Top8 hints use existing owned BF16 router buffers. One serial speculative pair queue preserves two streams per pair. Only actual LFU16 misses import owner/epoch-matching bytes. Both modes touch48MiB before comparison. Hooks.none, normal grouped prefill, original demand routing and arithmetic remain. No app gain qualification."]
        return try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    }
    private static func json<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
    private static func memoryGuard() throws -> Int {
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/memory_pressure")
        process.arguments = ["-Q"]
        let pipe = Pipe(); process.standardOutput = pipe; process.standardError = pipe
        try process.run(); process.waitUntilExit()
        let text = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        let regex = try NSRegularExpression(pattern: "free percentage: ([0-9]+)%")
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard process.terminationStatus == 0, let match = regex.firstMatch(in: text, range: range),
              let valueRange = Range(match.range(at: 1), in: text), let value = Int(text[valueRange]), value >= 30 else {
            throw EarlyPrefetchProbeError(detail: "system free memory below30% or guard unavailable")
        }
        return value
    }
}
