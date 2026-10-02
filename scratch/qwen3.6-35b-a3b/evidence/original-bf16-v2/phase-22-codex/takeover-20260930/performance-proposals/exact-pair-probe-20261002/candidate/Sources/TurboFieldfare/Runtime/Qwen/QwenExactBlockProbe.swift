import Foundation
import Metal
import CryptoKit
import TurboFieldfareOfficialQwenSource

private struct ExactProbeError: Error { let detail: String }
private func probeRequire(_ condition: Bool, _ detail: String) throws {
    if !condition { throw ExactProbeError(detail: detail) }
}
private func probeBitsEqual(_ a: [Float], _ b: [Float]) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
}
private func probeSHA(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
private func probeRowSHA(_ rows: [[Float]]) -> String {
    var hash = SHA256()
    for row in rows {
        var offset = 0
        while offset < row.count {
            let end = min(offset + 1024, row.count)
            let bits = row[offset..<end].map(\.bitPattern)
            bits.withUnsafeBytes { hash.update(data: Data($0)) }
            offset = end
        }
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
private func probeStateEqual(_ a: QwenOfficialSourceRunnerDiagnosticSnapshot,
                             _ b: QwenOfficialSourceRunnerDiagnosticSnapshot) -> Bool {
    guard a.position == b.position, a.fullKVPositions == b.fullKVPositions,
          a.linear.geometry == b.linear.geometry, a.linear.positions == b.linear.positions,
          Set(a.linear.layers.keys) == Set(b.linear.layers.keys),
          Set(a.committedKeys.keys) == Set(b.committedKeys.keys),
          Set(a.committedValues.keys) == Set(b.committedValues.keys) else { return false }
    for key in a.linear.layers.keys {
        guard let x = a.linear.layers[key], let y = b.linear.layers[key],
              probeBitsEqual(x.convolutionHistory, y.convolutionHistory),
              probeBitsEqual(x.recurrentMatrix, y.recurrentMatrix) else { return false }
    }
    for key in a.committedKeys.keys {
        guard probeBitsEqual(a.committedKeys[key]!, b.committedKeys[key]!),
              probeBitsEqual(a.committedValues[key]!, b.committedValues[key]!) else { return false }
    }
    return true
}
private func probeRoutes(_ rows: [QwenExactBlockRoute]) -> [QwenExactBlockRoute] {
    rows.sorted { $0.position == $1.position ? $0.layer < $1.layer : $0.position < $1.position }
}
private func probeProposal(_ history: [Int32], maximum: Int, stops: Set<Int32>) -> [Int32] {
    guard maximum > 0, history.count > 2 else { return [] }
    for width in stride(from: min(12, history.count), through: 2, by: -1) {
        let suffix = Array(history.suffix(width))
        let upper = history.count - width - 1
        if upper < 0 { continue }
        for start in stride(from: upper, through: 0, by: -1) {
            if Array(history[start..<(start + width)]) == suffix {
                var proposal: [Int32] = []
                for token in history[(start + width)..<min(start + width + maximum, history.count)] {
                    proposal.append(token)
                    if stops.contains(token) { break }
                }
                return proposal
            }
        }
    }
    return []
}

/// Staged diagnostic entry only. Loads through the unchanged pinned source loader.
/// It publishes no text, parser state, tool calls or conversation journal.
public enum QwenExactBlockProbe {
    @MainActor public static func run(registrationURL: URL, requestURL: URL) async throws -> Data {
        let raw = try Data(contentsOf: requestURL)
        guard raw.count < 128 * 1024,
              let request = try JSONSerialization.jsonObject(with: raw) as? [String: Any],
              let prefixInts = request["inputTokenIDs"] as? [Int],
              let firstInts = request["firstInputs"] as? [Int],
              let settings = request["settings"] as? [String: Any],
              let eosInt = request["codecEOS"] as? Int,
              let maxContext = request["maximumContext"] as? Int,
              let maxNew = settings["maxNewTokens"] as? Int,
              let temperature = settings["temperature"] as? NSNumber,
              let penalty = settings["repetitionPenalty"] as? NSNumber,
              let strings = settings["stopStrings"] as? [String],
              let stops = settings["extraStopTokens"] as? [Int],
              (prefixInts + firstInts + stops + [eosInt]).allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }) else {
            throw ExactProbeError(detail: "invalid bounded request")
        }
        try probeRequire(request["captureSHA256"] as? String == "841f885f3e7e30c56656176e2cb41b4616f01d88db16f30f03ab058f125e1285", "capture identity")
        let prefix = prefixInts.map { Int32($0) }
        let firstInputs = firstInts.map { Int32($0) }
        let eos = Int32(eosInt)
        try probeRequire(prefix.count == 1175 && firstInputs.count == 4 && maxContext == 8192,
                         "bounded frozen first block geometry")
        var config = GenerationConfig(maxNewTokens: maxNew,
            temperature: Float(temperature.doubleValue),
            topK: settings["topK"] as? Int,
            topP: (settings["topP"] as? NSNumber).map { Float($0.doubleValue) },
            repetitionPenalty: Float(penalty.doubleValue),
            seed: nil, stopStrings: strings,
            extraStopTokens: Set(stops.map { Int32($0) }))
        config.logitTransform = .raw
        try config.validate()
        try probeRequire(config.temperature == 0 && config.extraStopTokens.contains(eos), "greedy EOS config")
        try probeRequire(Array(firstInputs.dropFirst()) == probeProposal(prefix + [firstInputs[0]], maximum: 3, stops: config.extraStopTokens), "first drafts were not causal frozen policy")
        try memoryGuard()
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.load(registrationURL: registrationURL, context: context,
            integrityPolicy: .sizeCheckTrustedReceipt, expertCacheSlots: 16,
            expertCachePolicy: .lfu, residencyBudgetBytes: 12 * 1024 * 1024 * 1024)
        guard let source = model.sourceIdentity else { throw ExactProbeError(detail: "missing official identity") }
        try probeRequire(source.descriptorContentSHA256 == (request["sourceDescriptorSHA256"] as? String)
            && source.markerSHA256 == (request["sourceMarkerSHA256"] as? String)
            && source.checksumManifestSHA256 == (request["sourceManifestSHA256"] as? String)
            && source.shardSetSHA256 == (request["sourceShardSetSHA256"] as? String), "loaded source/capture identity mismatch")
        let runner = try QwenOfficialSourceRunner(model: model, maxContext: maxContext,
            expertSlotCount: 16, hooks: .none)
        let scratch = try RawCompletionScratch(context: context, vocab: model.architecture.vocabularySize)
        var activeTurn = false
        func rollback() async throws {
            if activeTurn { try await runner.rollbackTurn(); activeTurn = false }
        }
        do {
        var baselineLogits = try await runner.prefill(tokenIDs: prefix, position: 0)
        try memoryGuard()

        func sample(_ logits: [Float], history: [Int32], position: Int) async throws -> Int32 {
            try Task.checkCancellation()
            try model.revalidateSource()
            try QwenSourceSamplerBoundary.publishFP32Logits(logits, into: scratch.logits,
                vocabularySize: model.architecture.vocabularySize)
            guard let command = context.queue.makeCommandBuffer() else { throw ExactProbeError(detail: "sampler command") }
            scratch.sampler.sample(commandBuffer: command, logits: scratch.logits, probs: scratch.probs,
                history: history, config: config, position: position, outToken: scratch.outToken)
            try Task.checkCancellation()
            command.commit()
            await command.completed()
            try probeRequire(command.status == .completed && command.error == nil, "sampler GPU failure")
            try Task.checkCancellation()
            let value = scratch.outToken.contents().load(as: UInt32.self)
            try probeRequire(value < UInt32(model.architecture.vocabularySize), "sample out of vocabulary")
            return Int32(value)
        }
        let initialPending = try await sample(baselineLogits, history: prefix, position: 0)
        try probeRequire(initialPending == firstInputs[0], "actual pending token differs from captured observed token")

        struct Captured: Sendable {
            let logits: [[Float]]
            let routes: [QwenExactBlockRoute]
            let tokenAdmissions: Int
            let layerAdmissions: Int
            let reusedPairChecks: Int
            let preGPUChecks: Int
            let pairMaps: Int
            let pairMappedExperts: Int
            let pairMoECommands: Int
            let separateSharedCommands: Int
            let sharedChecks: Int
            let finalChecks: Int
            init(_ capture: QwenExactBlockCapture) {
                logits = capture.logits
                routes = capture.routes
                tokenAdmissions = capture.tokenAdmissions
                layerAdmissions = capture.layerAdmissions
                reusedPairChecks = capture.reusedPairChecks
                preGPUChecks = capture.preGPUChecks
                pairMaps = capture.pairMaps
                pairMappedExperts = capture.pairMappedExperts
                pairMoECommands = capture.pairMoECommands
                separateSharedCommands = capture.separateSharedCommands
                sharedChecks = capture.sharedChecks
                finalChecks = capture.finalChecks
            }
        }
        struct Arm: Sendable {
            let capture: Captured
            let nextLogits: [Float]
            let samples: [Int32]
            let matched: Int
            let wall: Double
            let recovery: Double
            let hits: UInt64
            let misses: UInt64
            let io: OfficialSourceIOMeasurement.Snapshot
        }
        func ioJSON(_ snapshot: OfficialSourceIOMeasurement.Snapshot) -> [String: Any] {
            ["reads": snapshot.reads, "failedReads": snapshot.failedReads,
             "preads": snapshot.preads.map { ["count": $0.count, "bytes": $0.bytes, "errors": $0.errors] },
             "validations": snapshot.validations.map { ["count": $0.count, "errors": $0.errors] }]
        }
        func arm(grouped: Bool, inputs: [Int32], history: [Int32], sampleBase: Int,
                 keep: Int = 4, serialPrefixOnly: Bool = false) async throws -> Arm {
            try memoryGuard()
            let capture = QwenExactBlockCapture()
            let io = OfficialSourceIOMeasurement(); io.setPhase(.decode)
            try probeRequire(model.source.installIOMeasurement(io), "IO measurement already installed")
            defer { model.source.removeIOMeasurement(io) }
            let before = await runner.routedExpertCacheSummary()
            let start = ProcessInfo.processInfo.systemUptime
            try await runner.beginTurn()
            activeTurn = true
            do {
                var next = baselineLogits
                if grouped {
                    _ = try await runner.prefill(tokenIDs: inputs, position: history.count, exactBlock: capture)
                } else {
                    for index in 0..<(serialPrefixOnly ? keep : inputs.count) {
                        next = try await runner.produce(token: inputs[index], position: history.count + index, exactBlock: capture)
                    }
                }
                var samples: [Int32] = []
                var matched = 0
                if !serialPrefixOnly {
                    for index in capture.logits.indices {
                        let token = try await sample(capture.logits[index],
                            history: history + Array(inputs.prefix(index + 1)), position: sampleBase + index)
                        samples.append(token)
                        if config.extraStopTokens.contains(token) { break }
                        if index < inputs.count - 1 {
                            if token != inputs[index + 1] { break }
                            matched += 1
                        }
                    }
                    next = capture.logits.last ?? baselineLogits
                }
                let replayStart = ProcessInfo.processInfo.systemUptime
                if grouped && keep < inputs.count {
                    try await runner.restoreTurnBaseline()
                    next = baselineLogits
                    for index in 0..<keep {
                        next = try await runner.produce(token: inputs[index], position: history.count + index)
                    }
                }
                let end = ProcessInfo.processInfo.systemUptime
                let after = await runner.routedExpertCacheSummary()
                return Arm(capture: Captured(capture), nextLogits: next, samples: samples, matched: matched,
                    wall: end - start, recovery: grouped && keep < inputs.count ? end - replayStart : 0,
                    hits: after.hits - before.hits, misses: after.misses - before.misses, io: io.snapshot())
            } catch let operation {
                do { try await rollback() }
                catch { throw ExactProbeError(detail: "operation \(operation); rollback \(error)") }
                throw operation
            }
        }
        func metrics(_ value: Arm, label: String) throws -> [String: Any] {
            ["label": label, "wallSeconds": value.wall, "restoreReplaySeconds": value.recovery,
             "hits": value.hits, "misses": value.misses, "io": ioJSON(value.io),
             "matchedDrafts": value.matched, "targetSamples": value.samples,
             "routeSHA256": probeSHA(try JSONEncoder().encode(probeRoutes(value.capture.routes))),
             "rawRowSHA256": probeRowSHA(value.capture.logits),
             "commands": ["pairMaps": value.capture.pairMaps,
                          "pairMappedExperts": value.capture.pairMappedExperts,
                          "pairMoE": value.capture.pairMoECommands,
                          "separateShared": value.capture.separateSharedCommands],
             "sourceChecks": ["tokenAdmissions": value.capture.tokenAdmissions,
                              "layerAdmissions": value.capture.layerAdmissions,
                              "reusedPairChecks": value.capture.reusedPairChecks,
                              "preGPUChecks": value.capture.preGPUChecks,
                              "sharedChecks": value.capture.sharedChecks,
                              "finalChecks": value.capture.finalChecks]]
        }
        func numerical(inputs: [Int32], history: [Int32], sampleBase: Int) async throws -> [String: Any] {
            let serial = try await arm(grouped: false, inputs: inputs, history: history, sampleBase: sampleBase)
            let expectedState = try await runner.diagnosticSnapshot()
            try await rollback()
            let grouped = try await arm(grouped: true, inputs: inputs, history: history, sampleBase: sampleBase)
            try probeRequire(grouped.capture.pairMaps == 80 && grouped.capture.pairMoECommands == 80
                && grouped.capture.pairMappedExperts + grouped.capture.reusedPairChecks == 1280
                && grouped.capture.separateSharedCommands == 0 && grouped.capture.sharedChecks == 0
                && grouped.capture.tokenAdmissions == 4 && grouped.capture.finalChecks == 4
                && grouped.capture.layerAdmissions == 160 && grouped.capture.preGPUChecks == 160, "pair scheduling/source check counts")
            try probeRequire(serial.capture.logits.count == 4 && grouped.capture.logits.count == 4,
                             "missing all-row logits")
            for row in 0..<4 {
                try probeRequire(probeBitsEqual(serial.capture.logits[row], grouped.capture.logits[row]), "raw row bit mismatch \(row)")
            }
            try probeRequire(probeRoutes(serial.capture.routes) == probeRoutes(grouped.capture.routes), "route ID/weight bits mismatch")
            try probeRequire(serial.samples == grouped.samples && serial.matched == grouped.matched, "target sampler mismatch")
            let actualState = try await runner.diagnosticSnapshot()
            try probeRequire(probeStateEqual(expectedState, actualState), "full block state bit mismatch")
            try await rollback()
            return ["serial": try metrics(serial, label: "serial numerical"),
                    "grouped": try metrics(grouped, label: "grouped numerical"), "allRowsRoutesStateExact": true]
        }
        func boundaries(inputs: [Int32], history: [Int32], sampleBase: Int) async throws -> [[String: Any]] {
            var checks: [[String: Any]] = []
            for keep in [0, 1, 3, 4] {
                let serial = try await arm(grouped: false, inputs: inputs, history: history,
                    sampleBase: sampleBase, keep: keep, serialPrefixOnly: true)
                let expected = try await runner.diagnosticSnapshot()
                let serialPending = try await sample(serial.nextLogits, history: history + Array(inputs.prefix(keep)),
                    position: sampleBase - 1 + keep)
                try await rollback()
                let grouped = try await arm(grouped: true, inputs: inputs, history: history,
                    sampleBase: sampleBase, keep: keep)
                let actual = try await runner.diagnosticSnapshot()
                try probeRequire(probeStateEqual(expected, actual), "forced consumed boundary state bits \(keep)")
                try probeRequire(probeBitsEqual(serial.nextLogits, grouped.nextLogits), "recovered next logits bits \(keep)")
                let pending = try await sample(grouped.nextLogits, history: history + Array(inputs.prefix(keep)),
                    position: sampleBase - 1 + keep)
                try probeRequire(pending == serialPending && actual.position == history.count + keep, "pending token consumed or sampler mismatch")
                checks.append(["forcedConsumedInputs": keep, "diagnosticTruncationNotModelRejection": true,
                               "nextPending": pending, "EOSBehaviorExercised": false,
                               "grouped": try metrics(grouped, label: "forced boundary")])
                try await rollback()
            }
            return checks
        }
        var report: [String: Any] = ["diagnosticOnly": true, "requestSHA256": probeSHA(raw),
            "captureSHA256": request["captureSHA256"]!, "expertSlots": 16,
            "prefixTokens": prefix.count, "firstInputs": firstInputs,
            "expertCachePolicy": "lfu", "EOSBehavior": "not exercised by this diagnostic",
            "sourceDescriptorSHA256": source.descriptorContentSHA256]
        report["firstNumerical"] = try await numerical(inputs: firstInputs, history: prefix, sampleBase: 1)
        report["firstForcedBoundaries"] = try await boundaries(inputs: firstInputs, history: prefix, sampleBase: 1)
        // Warm both arms, then opposite-order ABBA/BAAB. Cache never rolls back.
        for grouped in [false, true] {
            _ = try await arm(grouped: grouped, inputs: firstInputs, history: prefix, sampleBase: 1)
            try await rollback()
        }
        var times: [[String: Any]] = [], serialTimes: [Double] = [], groupedTimes: [Double] = []
        for grouped in [false, true, true, false, true, false, false, true] {
            let value = try await arm(grouped: grouped, inputs: firstInputs, history: prefix, sampleBase: 1)
            try probeRequire(value.matched == 3 && value.samples.count == 4, "full accept cost gate input changed")
            if grouped { groupedTimes.append(value.wall) } else { serialTimes.append(value.wall) }
            times.append(try metrics(value, label: grouped ? "grouped" : "serial"))
            try await rollback()
        }
        let serialSorted = serialTimes.sorted(), groupedSorted = groupedTimes.sorted()
        let serialMedian = (serialSorted[1] + serialSorted[2]) / 2
        let groupedMedian = (groupedSorted[1] + groupedSorted[2]) / 2
        let ratio = groupedMedian / serialMedian
        report["alternatingTiming"] = times
        report["groupedToSerialMedian"] = ratio
        report["workloadBreakEvenRatioBeforeRecovery"] = 29.0 / 43.0
        report["cacheCaveat"] = "One runner; cache contents are not rolled back. Actual hits/misses/reads recorded per arm; opposite order after warming. No throughput qualification."
        if ratio >= 1 {
            report["decision"] = "reject and abandon: full-accept verifier did not beat serial; second block not executed; full rejection correctness not qualified"
            return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        }
        // Consume only the first block's verified inputs; sampled bonus stays pending.
        try await runner.beginTurn()
            activeTurn = true
        var next = baselineLogits
        for index in firstInputs.indices { next = try await runner.produce(token: firstInputs[index], position: prefix.count + index) }
        try await runner.finishTurn()
        activeTurn = false
        let history = prefix + firstInputs
        let pending = try await sample(next, history: history, position: 4)
        let drafts = probeProposal(history + [pending], maximum: 3, stops: config.extraStopTokens)
        let secondInputs = [pending] + drafts
        try probeRequire(secondInputs.count == 4 && pending == 28 && drafts == [8422, 8901, 1224], "adjacent causal cycle alignment")
        baselineLogits = next
        report["secondInputs"] = secondInputs
        report["secondNumerical"] = try await numerical(inputs: secondInputs, history: history, sampleBase: 5)
        let rejected = try await arm(grouped: true, inputs: secondInputs, history: history, sampleBase: 5, keep: 1)
        try probeRequire(rejected.matched == 0 && rejected.samples.count == 1, "selected adjacent rejection changed")
        let rejectedState = try await runner.diagnosticSnapshot()
        let correction = rejected.samples[0] // Actual sampler, never saved answer.
        try probeRequire(rejectedState.position == history.count + 1, "correction was consumed")
        try await rollback()
        let reference = try await arm(grouped: false, inputs: secondInputs, history: history,
            sampleBase: 5, keep: 1, serialPrefixOnly: true)
        let referenceState = try await runner.diagnosticSnapshot()
        try probeRequire(probeStateEqual(rejectedState, referenceState)
            && probeBitsEqual(rejected.nextLogits, reference.nextLogits), "actual rejection restore/replay bits")
        let referenceCorrection = try await sample(reference.nextLogits, history: history + [pending], position: 5)
        try probeRequire(referenceCorrection == correction, "actual correction sampler mismatch")
        try await rollback()
        report["actualRejection"] = ["correctionPending": correction, "consumedAcceptedInputs": 1,
            "recovery": try metrics(rejected, label: "actual rejection"), "reference": try metrics(reference, label: "accepted prefix reference")]
        report["decision"] = "exact isolated probe passed; not speculative generation or speed qualification"
        return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        } catch let operation {
            do { try await rollback() }
            catch { throw ExactProbeError(detail: "operation \(operation); rollback \(error)") }
            throw operation
        }
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
            throw ExactProbeError(detail: "system free memory below30% or guard unavailable")
        }
    }
}
