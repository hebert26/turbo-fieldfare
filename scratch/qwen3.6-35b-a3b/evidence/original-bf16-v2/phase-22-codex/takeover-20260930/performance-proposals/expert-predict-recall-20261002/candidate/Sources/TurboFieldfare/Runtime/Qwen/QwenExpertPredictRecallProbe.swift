import Foundation
import Metal
import CryptoKit
import TurboFieldfareOfficialQwenSource

private struct PredictionProbeError: Error { let detail: String }
private func requirePrediction(_ condition: Bool, _ detail: String) throws {
    if !condition { throw PredictionProbeError(detail: detail) }
}

// Source-copy command only. It publishes no parser, tools or conversation state.
public enum QwenExpertPredictRecallProbe {
    @MainActor public static func run(registrationURL: URL, requestURL: URL) async throws -> Data {
        let raw = try Data(contentsOf: requestURL)
        let digest = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        try requirePrediction(digest == "841f885f3e7e30c56656176e2cb41b4616f01d88db16f30f03ab058f125e1285", "capture hash")
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
            throw PredictionProbeError(detail: "capture fields")
        }
        try requirePrediction(prefixInts.count == 1175 && recordedInts.count == 29
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
        try requirePrediction(config.temperature == 0 && config.extraStopTokens.contains(eos)
            && recorded.last == eos, "greedy terminal capture")
        try memoryGuard()
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.load(registrationURL: registrationURL, context: context,
            integrityPolicy: .sizeCheckTrustedReceipt, expertCacheSlots: 16,
            expertCachePolicy: .lfu, residencyBudgetBytes: 12 * 1024 * 1024 * 1024)
        guard let source = model.sourceIdentity else { throw PredictionProbeError(detail: "source identity") }
        try requirePrediction(source.descriptorContentSHA256 == (request["sourceDescriptorSHA256"] as? String)
            && source.markerSHA256 == (request["sourceMarkerSHA256"] as? String)
            && source.checksumManifestSHA256 == (request["sourceManifestSHA256"] as? String)
            && source.shardSetSHA256 == (request["sourceShardSetSHA256"] as? String), "official source mismatch")
        try requirePrediction(model.architecture.layers == 40 && model.architecture.hiddenSize <= 4096,
                              "bounded architecture")
        let capture = QwenExpertPredictionCapture(firstPosition: prefix.count,
            width: model.architecture.hiddenSize, layers: model.architecture.layers)
        let runner = try QwenOfficialSourceRunner(model: model, maxContext: maxContext,
            expertSlotCount: 16, hooks: .none, predictionCapture: capture)
        let scratch = try RawCompletionScratch(context: context, vocab: model.architecture.vocabularySize)
        let prefillStart = QwenProductionTimingMeasurement.uptime()
        var logits = try await runner.prefill(tokenIDs: prefix, position: 0)
        let prefillEnd = QwenProductionTimingMeasurement.uptime()
        try memoryGuard()
        var history = prefix
        var generated: [Int32] = []
        var consumedForwards = 0
        var foundStop = false
        let decodeStart = QwenProductionTimingMeasurement.uptime()
        // The model supplies every input. Recorded IDs are used only to check results.
        for index in 0..<min(config.maxNewTokens, 64) {
            try Task.checkCancellation()
            try QwenSourceSamplerBoundary.publishFP32Logits(logits, into: scratch.logits,
                vocabularySize: model.architecture.vocabularySize)
            guard let command = context.queue.makeCommandBuffer() else {
                throw PredictionProbeError(detail: "sampler command")
            }
            scratch.sampler.sample(commandBuffer: command, logits: scratch.logits, probs: scratch.probs,
                history: history, config: config, position: index, outToken: scratch.outToken)
            try Task.checkCancellation()
            command.commit()
            await command.completed()
            try requirePrediction(command.status == .completed && command.error == nil, "sampler GPU error")
            let sampled = scratch.outToken.contents().load(as: UInt32.self)
            try requirePrediction(sampled < UInt32(model.architecture.vocabularySize), "invalid sampled ID")
            let token = Int32(sampled)
            try requirePrediction(index < recorded.count && token == recorded[index], "sample differs at index \(index)")
            generated.append(token)
            if config.extraStopTokens.contains(token) { foundStop = true; break }
            let position = history.count
            logits = try await runner.produce(token: token, position: position)
            consumedForwards += 1
            history.append(token)
        }
        let decodeEnd = QwenProductionTimingMeasurement.uptime()
        try requirePrediction(foundStop && generated == recorded && consumedForwards == 28, "incomplete frozen generation")
        let finalPosition = await runner.position
        try requirePrediction(finalPosition == 1203, "consumed/pending alignment")
        let state = capture.drain()
        try requirePrediction(state.failure == nil && state.rows.count == consumedForwards * 40,
                              state.failure ?? "missing decode rows")
        // Serial caches are private to each layer. The next layer cannot change
        // between this layer's launch point and its next map's entry snapshot.
        for position in 1175..<1203 {
            for layer in 0..<40 {
                let key = QwenExpertPredictionCapture.Key(position: position, layer: layer)
                guard let row = state.rows[key], let map = row.map else {
                    throw PredictionProbeError(detail: "missing actual map")
                }
                try requirePrediction(!row.duplicate && row.mapStart > 0 && row.mapEnd > row.mapStart,
                                      "duplicate, failed or incomplete map")
                try requirePrediction(map.experts.count == 8 && Set(map.experts).count == 8
                    && map.residentSlots.count == 16 && map.assignedSlots.count == 8
                    && Set(map.assignedSlots).count == 8 && map.assignedSlots.allSatisfy { (0..<16).contains($0) }
                    && Set(map.missIndices).count == map.missIndices.count
                    && map.missIndices.allSatisfy { map.experts.indices.contains($0) }, "actual plan shape")
                let residents = Set(map.residentSlots.filter { $0 >= 0 })
                try requirePrediction(map.residentSlots.allSatisfy { $0 == -1 || (0..<model.architecture.experts).contains($0) }
                    && residents.count == map.residentSlots.filter { $0 >= 0 }.count, "actual resident inventory shape")
                let misses = Set(map.missIndices.map { map.experts[$0] })
                try requirePrediction(misses == Set(map.experts).subtracting(residents), "actual miss/resident mismatch")
                if layer < 39 {
                    try requirePrediction(row.early?.count == model.architecture.hiddenSize
                        && row.late?.count == model.architecture.hiddenSize && row.lateTime > row.mapEnd,
                        "missing residual or layer boundary")
                }
            }
        }
        try memoryGuard()
        var rows: [[String: Any]] = []
        var totals: [String: (routeHits: Int, misses: Int, covered: Int, nonresident: Int, waste: Int,
                              predictor: UInt64, valid: Int, absent: Int, window: UInt64, ceiling: UInt64)] = [:]
        var scenarioTotals: [String: Double] = [:]
        let readCosts: [Double] = [630_000, 1_260_000, 2_520_000]
        let validationCosts: [Double] = [0, 1_000_000, 2_000_000]
        let pairBytes = UInt64(model.architecture.hiddenSize) * UInt64(model.architecture.routedIntermediateSize) * 6
        let posthocStart = QwenProductionTimingMeasurement.uptime()
        for position in 1175..<1203 {
            try memoryGuard()
            for layer in 0..<39 {
                let current = state.rows[.init(position: position, layer: layer)]!
                let target = state.rows[.init(position: position, layer: layer + 1)]!
                let map = target.map!
                let actual = Set(map.experts)
                let residents = Set(map.residentSlots.filter { $0 >= 0 })
                let actualMisses = Set(map.missIndices.map { map.experts[$0] })
                for predictor in ["early", "late"] {
                    let hidden = predictor == "early" ? current.early! : current.late!
                    let start = QwenProductionTimingMeasurement.uptime()
                    let predicted = try await runner.diagnosticNextLayerPrediction(hidden: hidden, layer: layer + 1)
                    let cost = QwenProductionTimingMeasurement.uptime() - start
                    let predictedSet = Set(predicted)
                    try requirePrediction(predicted.count == 8 && predictedSet.count == 8, "prediction route shape")
                    let nonresident = predictedSet.subtracting(residents)
                    let covered = nonresident.intersection(actualMisses)
                    let waste = nonresident.subtracting(actual)
                    try requirePrediction(nonresident.intersection(actual) == covered, "launch inventory mismatch")
                    let launch = predictor == "early" ? current.mapEnd : current.lateTime
                    let validWindow = launch > 0 && target.mapStart >= launch
                    let window = validWindow ? target.mapStart - launch : 0
                    let actualMapWall = target.mapEnd - target.mapStart
                    var record: [String: Any] = ["position": position, "fromLayer": layer,
                        "targetLayer": layer + 1, "predictor": predictor, "predictedExperts": predicted,
                        "actualExperts": map.experts, "actualMissExperts": actualMisses.sorted(),
                        "actualEntrySlots": map.residentSlots, "actualAssignedSlots": map.assignedSlots,
                        "routeMatches": predictedSet.intersection(actual).count,
                        "coveredMisses": covered.count, "nonresidentPredictions": nonresident.count,
                        "wastedNonresidentPredictions": waste.count, "wastedReadBytesScenario": UInt64(waste.count) * pairBytes,
                        "predictorNanoseconds": cost, "windowPresentAndCausal": validWindow,
                        "actualMapWallNanoseconds": actualMapWall]
                    if validWindow { record["windowNanoseconds"] = window }
                    var aggregate = totals[predictor] ?? (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                    aggregate.routeHits += predictedSet.intersection(actual).count
                    aggregate.misses += actualMisses.count
                    aggregate.covered += covered.count
                    aggregate.nonresident += nonresident.count
                    aggregate.waste += waste.count
                    aggregate.predictor += cost
                    if validWindow {
                        aggregate.valid += 1
                        aggregate.window += window
                        aggregate.ceiling += min(actualMapWall, window)
                    } else { aggregate.absent += 1 }
                    totals[predictor] = aggregate
                    if validWindow {
                        var scenarios: [[String: Any]] = []
                        for q in readCosts {
                            for a in validationCosts {
                                let score = min(Double(actualMapWall), Double(covered.count) * q)
                                    - max(0, Double(nonresident.count) * q - Double(window)) - Double(cost) - a
                                let name = "\(predictor)/q\(Int(q))/a\(Int(a))"
                                scenarioTotals[name, default: 0] += score
                                scenarios.append(["qNanosecondsPerPair": q, "additionalValidationNanoseconds": a,
                                    "netNanosecondsScenario": score, "wastedReadNanosecondsScenario": Double(waste.count) * q])
                            }
                        }
                        record["scenarios"] = scenarios
                    }
                    rows.append(record)
                }
            }
        }
        let posthocEnd = QwenProductionTimingMeasurement.uptime()
        var summaries: [[String: Any]] = []
        for predictor in ["early", "late"] {
            let value = totals[predictor]!
            try requirePrediction(value.valid + value.absent == 1092, "prediction coverage count")
            let score = scenarioTotals["\(predictor)/q1260000/a0", default: 0] / Double(generated.count)
            let priority = value.absent != 0 ? "incomplete windows" : score < 65_000_000 ? "drop"
                : score >= 130_000_000 ? "supports next diagnostic review" : "review"
            summaries.append(["predictor": predictor, "cases": value.valid + value.absent,
                "routeRecall": Double(value.routeHits) / Double(1092 * 8),
                "actualMissCount": value.misses, "coveredActualMisses": value.covered,
                "actualMissRecall": Double(value.covered) / Double(max(1, value.misses)),
                "actualMissRecallDefined": value.misses != 0,
                "nonresidentPredictions": value.nonresident, "wastedNonresidentPredictions": value.waste,
                "wastedReadBytesScenario": UInt64(value.waste) * pairBytes,
                "predictorTotalNanoseconds": value.predictor, "causalWindows": value.valid,
                "excludedAbsentOrNoncausalWindows": value.absent,
                "totalCausalWindowNanoseconds": value.window,
                "idealOverlapWindowCeilingNanosecondsPerOutput": Double(value.ceiling) / Double(generated.count),
                "optimisticQ1260usA0NetNanosecondsPerOutputScenario": score,
                "optimisticQ1260usA0NetNanosecondsPerForwardScenario": score * Double(generated.count) / Double(consumedForwards),
                "prioritizationScreenOnly": priority])
        }
        let result: [String: Any] = ["passed": true, "captureSHA256": digest,
            "sampledTokenIDs": generated, "sampledIDMatch": true, "generatedOutputsIncludingEOS": generated.count,
            "consumedDecodeForwards": consumedForwards, "terminalEOSLeftPending": true,
            "actualDecodeMaps": state.rows.count, "predictionCasesPerPredictor": 1092,
            "prefillNanoseconds": prefillEnd - prefillStart, "observedDecodeNanoseconds": decodeEnd - decodeStart,
            "posthocNanoseconds": posthocEnd - posthocStart, "sourceBF16PairBytesFromGeometry": pairBytes, "collectionCallbackNanoseconds": state.callbackNanoseconds,
            "boundedResidualBytesMaximum": 32 * 39 * 2 * model.architecture.hiddenSize * MemoryLayout<Float>.stride,
            "actualResidualBytes": 28 * 39 * 2 * model.architecture.hiddenSize * MemoryLayout<Float>.stride,
            "summaries": summaries, "scenarioTotalNanoseconds": scenarioTotals, "rows": rows,
            "scenarioFormula": "min(actualMapWall,covered*q)-max(0,nonresident*q-window)-predictorCost-additionalValidationCost",
            "scenarioLimits": "q is an old cross-capture slope with0.5x/1x/2x sensitivity. a=0 is optimistic unknown extra prefetch validation cost;1ms/2ms are declared sensitivity assumptions. Reads serialize in a hypothetical queue and ALL speculative reads drain before demand work. Negative scores retained. Predictor timing includes norm, real router projection, routing and two full source validations. No prefetch runs or app gain is measured.",
            "windowLimits": "Early starts after current map. Late starts at residual-after-MoE. Both end at next map start. Existing phase boundaries use the production monotonic clock. Collection can inflate windows. Callback measurement includes lock wait/body but excludes unlock and later allocation effects. No calibration replay was added.",
            "residentStateProof": "Each layer has one private coordinator/cache. Serial decode makes one map per layer per position. There is no prefetch or other cache writer. Therefore target-layer preplan inventory equals its inventory at both earlier-layer launch points. No LFU replay is used.",
            "scope": "Posthoc recall probe only. Same normal grouped prefill and serial decode scheduling with hooks.none. No candidate routes or cache reads are substituted. Global early and late totals are separate; no per-case best-predictor oracle."]
        return try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    }
    private static func memoryGuard() throws {
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/memory_pressure")
        process.arguments = ["-Q"]
        let pipe = Pipe(); process.standardOutput = pipe; process.standardError = pipe
        try process.run(); process.waitUntilExit()
        let text = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        let regex = try NSRegularExpression(pattern: "free percentage: ([0-9]+)%")
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard process.terminationStatus == 0, let match = regex.firstMatch(in: text, range: range),
              let valueRange = Range(match.range(at: 1), in: text), let value = Int(text[valueRange]), value >= 30 else {
            throw PredictionProbeError(detail: "system free memory below30% or guard unavailable")
        }
    }
}
