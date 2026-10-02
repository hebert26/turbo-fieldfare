import Foundation
import Darwin
import Metal
import CryptoKit
import TurboFieldfareOfficialQwenSource

private struct MetadataDecodeProbeError: Error { let detail: String }
private func requireMetadataDecode(_ condition: Bool, _ detail: String) throws {
    if !condition { throw MetadataDecodeProbeError(detail: detail) }
}

// Source-copy command only. It publishes no parser, tools or conversation state.
public enum QwenExactMetadataDecodeProbe {
    @MainActor public static func run(registrationURL: URL, requestURL: URL, enabled: Bool) async throws -> Data {
        let raw = try Data(contentsOf: requestURL)
        let digest = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        try requireMetadataDecode(digest == "841f885f3e7e30c56656176e2cb41b4616f01d88db16f30f03ab058f125e1285", "capture hash")
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
            throw MetadataDecodeProbeError(detail: "capture fields")
        }
        try requireMetadataDecode(prefixInts.count == 1175 && recordedInts.count == 29
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
        try requireMetadataDecode(config.temperature == 0 && config.extraStopTokens.contains(eos)
            && recorded.last == eos, "greedy terminal capture")
        var minimumFree = try memoryGuard()
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.load(registrationURL: registrationURL, context: context,
            integrityPolicy: .sizeCheckTrustedReceipt, expertCacheSlots: 16,
            expertCachePolicy: .lfu, residencyBudgetBytes: 12 * 1024 * 1024 * 1024)
        guard let source = model.sourceIdentity else { throw MetadataDecodeProbeError(detail: "source identity") }
        try requireMetadataDecode(source.descriptorContentSHA256 == (request["sourceDescriptorSHA256"] as? String)
            && source.markerSHA256 == (request["sourceMarkerSHA256"] as? String)
            && source.checksumManifestSHA256 == (request["sourceManifestSHA256"] as? String)
            && source.shardSetSHA256 == (request["sourceShardSetSHA256"] as? String), "official source mismatch")
        try requireMetadataDecode(model.architecture.layers == 40 && model.architecture.hiddenSize == 2048 && model.architecture.experts == 256,
                              "bounded architecture")
        let capture = QwenMetadataDecodeCapture()
        let measurement = OfficialSourceDecodeMeasurement()
        guard let validator = try model.makeDecodeReceiptValidator() else {
            throw MetadataDecodeProbeError(detail: "trusted receipt required")
        }
        let runner = try QwenOfficialSourceRunner(model: model, maxContext: maxContext,
            expertSlotCount: 16, hooks: .none, metadataCapture: capture,
            decodeReceiptValidator: validator, decodeMeasurement: measurement)
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
        let cpuBefore = try cpuUsage()
        let decodeStart = QwenProductionTimingMeasurement.uptime()
        // Recorded IDs only check outputs. They never supply decode inputs.
        for index in 0..<min(config.maxNewTokens, 64) {
            try Task.checkCancellation()
            try QwenSourceSamplerBoundary.publishFP32Logits(logits, into: scratch.logits,
                vocabularySize: model.architecture.vocabularySize)
            guard let command = context.queue.makeCommandBuffer() else {
                throw MetadataDecodeProbeError(detail: "sampler command")
            }
            scratch.sampler.sample(commandBuffer: command, logits: scratch.logits, probs: scratch.probs,
                history: history, config: config, position: index, outToken: scratch.outToken)
            try Task.checkCancellation()
            command.commit()
            await command.completed()
            try requireMetadataDecode(command.status == .completed && command.error == nil, "sampler GPU error")
            let sampled = scratch.outToken.contents().load(as: UInt32.self)
            try requireMetadataDecode(sampled < UInt32(model.architecture.vocabularySize), "invalid sampled ID")
            let token = Int32(sampled)
            try requireMetadataDecode(index < recorded.count && token == recorded[index], "sample differs at index \(index)")
            generated.append(token)
            if config.extraStopTokens.contains(token) { foundStop = true; break }
            let start = QwenProductionTimingMeasurement.uptime()
            logits = try await runner.produce(token: token, position: history.count)
            forwardTimes.append(QwenProductionTimingMeasurement.uptime() - start)
            history.append(token)
        }
        let decodeEnd = QwenProductionTimingMeasurement.uptime()
        let cpuAfter = try cpuUsage()
        try requireMetadataDecode(foundStop && generated == recorded && forwardTimes.count == 28,
                                 "incomplete frozen generation")
        let finalPosition = await runner.position
        let usable = !(await runner.isUnusable)
        let state = capture.snapshot()
        try requireMetadataDecode(finalPosition == 1203 && usable && measurement.drained(), "terminal runner/entry-lane state")
        try requireMetadataDecode(!state.duplicate && state.routes.count == 1120 && state.plans.count == 1120,
                                 "missing or duplicate decode evidence")
        for position in 1175..<1203 {
            for layer in 0..<40 {
                let key = QwenMetadataDecodeCapture.Key(position: position, layer: layer)
                guard let route = state.routes[key], let plan = state.plans[key] else {
                    throw MetadataDecodeProbeError(detail: "missing route/plan")
                }
                let residents = Set(plan.residents.filter { $0 >= 0 })
                try requireMetadataDecode(route.experts == plan.experts && route.experts.count == 8
                    && route.weightBits.count == 8 && Set(route.experts).count == 8
                    && plan.residents.count == 16 && plan.assignedSlots.count == 8
                    && Set(plan.assignedSlots).count == 8 && plan.assignedSlots.allSatisfy { (0..<16).contains($0) }
                    && plan.residents.allSatisfy { $0 == -1 || (0..<256).contains($0) }
                    && residents.count == plan.residents.filter { $0 >= 0 }.count
                    && Set(plan.missIndices).count == plan.missIndices.count
                    && plan.missIndices.allSatisfy { plan.experts.indices.contains($0) }, "route/plan shape")
                try requireMetadataDecode(Set(plan.missIndices.map { plan.experts[$0] })
                    == Set(plan.experts).subtracting(residents), "actual miss membership")
            }
        }
        let order: (QwenMetadataDecodeCapture.Key, QwenMetadataDecodeCapture.Key) -> Bool = {
            $0.position == $1.position ? $0.layer < $1.layer : $0.position < $1.position
        }
        let routes = state.routes.keys.sorted(by: order).map { state.routes[$0]! }
        let plans = state.plans.keys.sorted(by: order).map { state.plans[$0]! }
        let sourceValidation = measurement.snapshot()
        try requireMetadataDecode(sourceValidation.count == 2296 && sourceValidation.entryChecks == 2296 * 78
            && sourceValidation.failures == 0 && sourceValidation.maximumEntryFDs > 0
            && sourceValidation.maximumEntryFDs <= (enabled ? 4 : 1)
            && sourceValidation.parallelCount == (enabled ? sourceValidation.count : 0), "source counters")
        minimumFree = min(minimumFree, try memoryGuard())
        let cpuProcess = try cpuUsage()
        let result: [String: Any] = ["schema": 1, "mode": enabled ? "on" : "off",
            "captureSHA256": digest, "outputIDs": generated.map(Int.init), "expectedOutputIDsMatched": true,
            "promptWallNanoseconds": prefillEnd - prefillStart,
            "decodeWallNanoseconds": decodeEnd - decodeStart, "forwardWallNanoseconds": forwardTimes,
            "routes": try json(routes), "plans": plans.map { plan -> [String: Any] in
                ["position": plan.position, "layer": plan.layer, "experts": plan.experts,
                 "assignedSlots": plan.assignedSlots, "missIndices": plan.missIndices,
                 "residents": plan.residents.map { value -> Any in value < 0 ? NSNull() : NSNumber(value: value) }]
            }, "sourceValidation": try json(sourceValidation), "minimumFreePercent": minimumFree,
            "safety": ["captureComplete": true, "runnerUsable": usable,
                       "position": finalPosition, "pendingTokens": 1],
            "processCPUUserMicroseconds": cpuProcess.user, "processCPUSystemMicroseconds": cpuProcess.system,
            "decodeCPUUserMicroseconds": cpuAfter.user - cpuBefore.user,
            "decodeCPUSystemMicroseconds": cpuAfter.system - cpuBefore.system,
            "rollbackExercised": false, "terminalEOSLeftPending": true,
            "timingLimits": "Decode includes all29 samples and28 forwards. Per-forward timings are secondary. Source wall is synchronous whole validator wall, including owned dispatch/join and counter overhead. Process CPU is separate and not additive proof for intervals. Prefill and all non-produce validations remain serial.",
            "scope": "Source-copy isolated diagnostic. Same fresh receipt entries and two complete scans with unchanged binding/root/list boundaries. Four contiguous ascending lanes each stop at first indexed error and join before lowest observed indexed error returns. One produce-owned cancellation latch bridges caller cancellation. No new await between post-map source validation and GPU submission. Up to three extra concurrent entry FDs can introduce EMFILE/ENFILE failure. Inter-entry visitation and temporal error precedence differ. No serial retry, prefetch or arithmetic change."]
        return try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    }
    private static func json<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
    private static func cpuUsage() throws -> (user: UInt64, system: UInt64) {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0,
              usage.ru_utime.tv_sec >= 0, usage.ru_utime.tv_usec >= 0,
              usage.ru_stime.tv_sec >= 0, usage.ru_stime.tv_usec >= 0 else {
            throw MetadataDecodeProbeError(detail: "process CPU unavailable")
        }
        return (UInt64(usage.ru_utime.tv_sec) * 1_000_000 + UInt64(usage.ru_utime.tv_usec),
                UInt64(usage.ru_stime.tv_sec) * 1_000_000 + UInt64(usage.ru_stime.tv_usec))
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
            throw MetadataDecodeProbeError(detail: "system free memory below30% or guard unavailable")
        }
        return value
    }
}
