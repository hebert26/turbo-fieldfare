import Foundation
import Darwin
import Metal
import CryptoKit
import TurboFieldfareOfficialQwenSource

private struct ExactProbeError: Error { let detail: String }
private func probeRequire(_ condition: Bool, _ detail: String) throws {
    if !condition { throw ExactProbeError(detail: detail) }
}
private func probeSHA(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
private func probeUpdateRows(_ rows: [[Float]], hash: inout SHA256) {
    for row in rows {
        var offset = 0
        while offset < row.count {
            let end = min(offset + 1024, row.count)
            let bits = row[offset..<end].map(\.bitPattern)
            bits.withUnsafeBytes { hash.update(data: Data($0)) }
            offset = end
        }
    }
}
private func probeRowSHA(_ rows: [[Float]]) -> String {
    var hash = SHA256()
    probeUpdateRows(rows, hash: &hash)
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
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

/// Isolated causal lookup comparison. Saved output IDs are checked only at the end.
public enum QwenExactBlockProbe {
    @MainActor public static func run(registrationURL: URL, requestURL: URL, mode: String) async throws -> Data {
        try probeRequire(["serial", "legacy", "new"].contains(mode), "invalid mode")
        let raw = try Data(contentsOf: requestURL)
        try probeRequire(raw.count < 128 * 1024 && probeSHA(raw) == "841f885f3e7e30c56656176e2cb41b4616f01d88db16f30f03ab058f125e1285", "capture identity")
        guard let request = try JSONSerialization.jsonObject(with: raw) as? [String: Any],
              let prefixInts = request["inputTokenIDs"] as? [Int],
              let settings = request["settings"] as? [String: Any],
              let eosInt = request["codecEOS"] as? Int,
              let maxContext = request["maximumContext"] as? Int,
              let maxNew = settings["maxNewTokens"] as? Int,
              let temperature = settings["temperature"] as? NSNumber,
              let penalty = settings["repetitionPenalty"] as? NSNumber,
              let strings = settings["stopStrings"] as? [String],
              let stops = settings["extraStopTokens"] as? [Int],
              (prefixInts + stops + [eosInt]).allSatisfy({ $0 >= 0 && $0 <= Int(Int32.max) }) else {
            throw ExactProbeError(detail: "invalid bounded request")
        }
        let prefix = prefixInts.map { Int32($0) }
        let eos = Int32(eosInt)
        let maximumOutputs = 29
        try probeRequire(prefix.count == 1175 && maxContext == 8192 && strings.isEmpty, "frozen geometry")
        var config = GenerationConfig(maxNewTokens: maxNew, temperature: Float(temperature.doubleValue),
            topK: settings["topK"] as? Int, topP: (settings["topP"] as? NSNumber).map { Float($0.doubleValue) },
            repetitionPenalty: Float(penalty.doubleValue), seed: nil, stopStrings: strings,
            extraStopTokens: Set(stops.map { Int32($0) }))
        config.logitTransform = .raw
        try config.validate()
        try probeRequire(config.temperature == 0 && config.extraStopTokens == [eos], "greedy terminal configuration")
        var minimumFree = try memoryGuard()
        try probeRequire(minimumFree >= 50, "launch free memory below50%")
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.load(registrationURL: registrationURL, context: context,
            integrityPolicy: .sizeCheckTrustedReceipt, expertCacheSlots: 16,
            expertCachePolicy: .lfu, residencyBudgetBytes: 12 * 1024 * 1024 * 1024)
        guard let source = model.sourceIdentity else { throw ExactProbeError(detail: "missing source identity") }
        try probeRequire(source.descriptorContentSHA256 == (request["sourceDescriptorSHA256"] as? String)
            && source.markerSHA256 == (request["sourceMarkerSHA256"] as? String)
            && source.checksumManifestSHA256 == (request["sourceManifestSHA256"] as? String)
            && source.shardSetSHA256 == (request["sourceShardSetSHA256"] as? String), "source identity mismatch")
        let runner = try QwenOfficialSourceRunner(model: model, maxContext: maxContext, expertSlotCount: 16, hooks: .none)
        let scratch = try RawCompletionScratch(context: context, vocab: model.architecture.vocabularySize)
        var activeTurn = false
        func clock() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func cpu() throws -> (Int64, Int64) {
            var value = rusage()
            try probeRequire(getrusage(RUSAGE_SELF, &value) == 0, "process CPU observation")
            return (Int64(value.ru_utime.tv_sec) * 1_000_000 + Int64(value.ru_utime.tv_usec),
                    Int64(value.ru_stime.tv_sec) * 1_000_000 + Int64(value.ru_stime.tv_usec))
        }
        func rollback() async throws {
            if activeTurn { try await runner.rollbackTurn(); activeTurn = false }
        }
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
            try probeRequire(value < UInt32(model.architecture.vocabularySize), "sample vocabulary")
            return Int32(value)
        }
        func ioJSON(_ snapshot: OfficialSourceIOMeasurement.Snapshot) -> [String: Any] {
            ["reads": snapshot.reads, "failedReads": snapshot.failedReads,
             "preads": snapshot.preads.map { ["count": $0.count, "bytes": $0.bytes, "errors": $0.errors] },
             "validations": snapshot.validations.map { ["count": $0.count, "errors": $0.errors] }]
        }
        do {
            let promptStart = clock()
            let promptLogits = try await runner.prefill(tokenIDs: prefix, position: 0)
            let promptWall = clock() - promptStart
            minimumFree = min(minimumFree, try memoryGuard())
            let cpuBefore = try cpu()
            let decodeStart = clock()
            var turnBegins = 0
            var turnFinishes = 0
            var turnRestores = 0
            var settlementWall: UInt64 = 0
            var rejectedCycles = 0
            var replayedInputs = 0
            var totalRecurrentAppendCalls = 0
            var totalRecurrentKeptInputs = 0
            var maximumSavedInputBytes = 0
            if mode == "serial" {
                try await runner.beginTurn(); activeTurn = true; turnBegins += 1
            }
            var pending = try await sample(promptLogits, history: prefix, position: 0)
            var outputs = [pending]
            var history = prefix
            var cycles: [[String: Any]] = []
            var acceptedRouteHash = SHA256()
            var acceptedRawRowHash = SHA256()
            var acceptedRouteCount = 0
            var acceptedRawRowCount = 0
            let routeEncoder = JSONEncoder()
            routeEncoder.outputFormatting = [.sortedKeys]
            while !config.extraStopTokens.contains(pending) && outputs.count < maximumOutputs {
                try Task.checkCancellation()
                // The budget depends on emitted outputs. It does not inspect saved termination length.
                let remaining = maximumOutputs - outputs.count
                let drafts = mode != "serial" ? probeProposal(history + [pending],
                    maximum: min(3, remaining - 1), stops: config.extraStopTokens) : []
                let inputs = [pending] + drafts
                let capture = QwenExactBlockCapture(enableRecurrentRecovery: mode == "new" && inputs.count == 4)
                let io = OfficialSourceIOMeasurement(); io.setPhase(.decode)
                try probeRequire(model.source.installIOMeasurement(io), "IO measurement already installed")
                let before = await runner.routedExpertCacheSummary()
                let cycleStart = clock()
                var replayWall: UInt64 = 0
                var recoveryKind = "none"
                var recurrentAppendCalls = 0
                var recurrentKeptInputs = 0
                var normalizedInputBytes = 0
                var samples: [Int32] = []
                var matched = 0
                var keep = 0
                do {
                    if mode != "serial" {
                        try await runner.beginTurn(); activeTurn = true; turnBegins += 1
                    }
                    if mode != "serial" && inputs.count == 4 {
                        _ = try await runner.prefill(tokenIDs: inputs, position: history.count, exactBlock: capture)
                    } else {
                        // Short proposals retain the same policy. Verify their real inputs serially.
                        for index in inputs.indices {
                            _ = try await runner.produce(token: inputs[index], position: history.count + index, exactBlock: capture)
                        }
                    }
                    try probeRequire(capture.logits.count == inputs.count, "missing verification rows")
                    for index in inputs.indices {
                        let token = try await sample(capture.logits[index],
                            history: history + Array(inputs.prefix(index + 1)), position: outputs.count + index)
                        samples.append(token)
                        keep = index + 1
                        if config.extraStopTokens.contains(token) { break }
                        if index < drafts.count {
                            if token != inputs[index + 1] { break }
                            matched += 1
                        }
                    }
                    try probeRequire(keep == 1 + matched && !samples.isEmpty, "accepted prefix alignment")
                    normalizedInputBytes = capture.savedInputBytes
                    maximumSavedInputBytes = max(maximumSavedInputBytes, normalizedInputBytes)
                    if keep < inputs.count {
                        let replayStart = clock()
                        if mode == "new" && inputs.count == 4 {
                            let settlement = try await runner.settleRecurrentPrefix(capture, keep: keep)
                            recoveryKind = "recurrentOnly"
                            recurrentAppendCalls = settlement.appendCalls
                            recurrentKeptInputs = settlement.keptInputs
                            try probeRequire(settlement.savedInputBytes == normalizedInputBytes,
                                "saved recurrent input accounting")
                        } else {
                            recoveryKind = "fullReplay"
                            try await runner.restoreTurnBaseline(); turnRestores += 1
                            for index in 0..<keep {
                                _ = try await runner.produce(token: inputs[index], position: history.count + index)
                            }
                            replayedInputs += keep
                        }
                        replayWall = clock() - replayStart
                        settlementWall += replayWall
                        rejectedCycles += 1
                        totalRecurrentAppendCalls += recurrentAppendCalls
                        totalRecurrentKeptInputs += recurrentKeptInputs
                    }
                    try Task.checkCancellation()
                    if mode != "serial" {
                        try await runner.finishTurn(); activeTurn = false; turnFinishes += 1
                    }
                } catch {
                    model.source.removeIOMeasurement(io)
                    throw error
                }
                let cycleWall = clock() - cycleStart
                let after = await runner.routedExpertCacheSummary()
                model.source.removeIOMeasurement(io)
                let snapshot = io.snapshot()
                let startPosition = history.count
                history.append(contentsOf: inputs.prefix(keep))
                outputs.append(contentsOf: samples)
                pending = samples.last!
                let position = await runner.position
                try probeRequire(position == history.count && outputs.count <= maximumOutputs, "consumed cursor mismatch")
                let acceptedRoutes = probeRoutes(capture.routes).filter { $0.position < startPosition + keep }
                try probeRequire(acceptedRoutes.count == keep * 40, "accepted route completeness")
                for route in acceptedRoutes {
                    acceptedRouteHash.update(data: try routeEncoder.encode(route))
                    acceptedRouteHash.update(data: Data([10]))
                }
                acceptedRouteCount += acceptedRoutes.count
                probeUpdateRows(Array(capture.logits.prefix(keep)), hash: &acceptedRawRowHash)
                acceptedRawRowCount += keep
                cycles.append(["index": cycles.count, "position": startPosition,
                    "pendingInput": inputs[0], "proposalIDs": drafts, "provisionalIDs": samples,
                    "matchedDrafts": matched, "consumedInputs": keep, "verifiedInputs": inputs.count,
                    "wallNanoseconds": cycleWall, "restoreReplayNanoseconds": replayWall,
                    "recoveryKind": recoveryKind, "recurrentAppendCalls": recurrentAppendCalls,
                    "recurrentKeptInputs": recurrentKeptInputs, "normalizedInputBytes": normalizedInputBytes,
                    "hits": after.hits - before.hits, "misses": after.misses - before.misses,
                    "routeSHA256": probeSHA(try routeEncoder.encode(probeRoutes(capture.routes))),
                    "rawRowSHA256": probeRowSHA(capture.logits),
                    "sourceChecks": ["tokenAdmissions": capture.tokenAdmissions,
                        "layerAdmissions": capture.layerAdmissions, "reusedPairChecks": capture.reusedPairChecks,
                        "preGPUChecks": capture.preGPUChecks, "sharedChecks": capture.sharedChecks,
                        "finalChecks": capture.finalChecks], "io": ioJSON(snapshot)])
            }
            if mode == "serial" {
                try await runner.finishTurn(); activeTurn = false; turnFinishes += 1
            }
            let decodeEnd = clock()
            let decodeWall = decodeEnd - decodeStart
            let requestWall = decodeEnd - promptStart
            let cpuAfter = try cpu()
            minimumFree = min(minimumFree, try memoryGuard())
            // Golden output is unavailable to every proposal and sampling decision above.
            guard let expected = request["sampledTokenIDs"] as? [Int] else { throw ExactProbeError(detail: "missing final assertion") }
            try probeRequire(outputs.map(Int.init) == expected && outputs.count == maximumOutputs
                && pending == eos, "generated outputs or terminal mismatch")
            try probeRequire(acceptedRouteCount == 1120 && acceptedRawRowCount == 28, "accepted chain completeness")
            let liveSavedInputBytesAtEnd = await runner.diagnosticRecurrentInputBytes()
            try probeRequire(liveSavedInputBytesAtEnd == 0, "recurrent input cleanup")
            let state = try await runner.diagnosticSnapshot()
            try probeRequire(state.position == 1203 && state.position == history.count
                && state.linear.positions.values.allSatisfy({ $0 == state.position })
                && state.fullKVPositions.values.allSatisfy({ $0 == state.position }), "final state cursor")
            let linearRows = state.linear.layers.keys.sorted().flatMap { key in
                [state.linear.layers[key]!.convolutionHistory, state.linear.layers[key]!.recurrentMatrix]
            }
            let final: [String: Any] = ["position": state.position, "pendingTokens": 1, "runnerUsable": true,
                "linearPositions": Dictionary(uniqueKeysWithValues: state.linear.positions.map { (String($0.key), $0.value) }),
                "fullKVPositions": Dictionary(uniqueKeysWithValues: state.fullKVPositions.map { (String($0.key), $0.value) }),
                "linearStateSHA256": probeRowSHA(linearRows),
                "committedKeysSHA256": probeRowSHA(state.committedKeys.keys.sorted().map { state.committedKeys[$0]! }),
                "committedValuesSHA256": probeRowSHA(state.committedValues.keys.sorted().map { state.committedValues[$0]! })]
            let report: [String: Any] = ["schema": 1, "mode": mode, "diagnosticOnly": true,
                "requestSHA256": probeSHA(raw), "sourceDescriptorSHA256": source.descriptorContentSHA256,
                "prefixTokens": prefix.count, "maximumContext": maxContext, "maximumOutputs": maximumOutputs,
                "expertSlots": 16, "expertCachePolicy": "lfu", "outputIDs": outputs,
                "expectedOutputIDsMatched": true, "acceptedRouteCount": acceptedRouteCount,
                "acceptedRawRowCount": acceptedRawRowCount,
                "acceptedRouteSHA256": acceptedRouteHash.finalize().map { String(format: "%02x", $0) }.joined(),
                "acceptedRawRowSHA256": acceptedRawRowHash.finalize().map { String(format: "%02x", $0) }.joined(),
                "promptWallNanoseconds": promptWall,
                "decodeWallNanoseconds": decodeWall, "requestWallNanoseconds": requestWall,
                "cycles": cycles, "finalState": final,
                "settlement": ["wallNanoseconds": settlementWall, "rejectedCycles": rejectedCycles,
                    "replayedInputs": replayedInputs, "recurrentAppendCalls": totalRecurrentAppendCalls,
                    "recurrentKeptInputs": totalRecurrentKeptInputs,
                    "maximumSavedInputBytes": maximumSavedInputBytes,
                    "liveSavedInputBytesAtEnd": liveSavedInputBytesAtEnd],
                "transactions": ["begins": turnBegins, "finishes": turnFinishes, "restores": turnRestores],
                "minimumFreePercent": minimumFree, "processCPUUserBeforeMicroseconds": cpuBefore.0,
                "processCPUSystemBeforeMicroseconds": cpuBefore.1, "processCPUUserAfterMicroseconds": cpuAfter.0,
                "processCPUSystemAfterMicroseconds": cpuAfter.1,
                "terminal": "sampled EOS remains pending and unconsumed",
                "timingLimits": "Isolated schedule; decode includes checkpoints, lookup, sampling, recovery and cycle diagnostics. Final state readback is excluded. Serial owns one turn; legacy/new own per-cycle turns. New replaces only exact4 rejection settlement; short proposals retain charged full replay. Outer conversation integration is not implemented. Not app or native MTP qualification."]
            return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        } catch let operation {
            do { try await rollback() }
            catch { throw ExactProbeError(detail: "operation \(operation); rollback \(error)") }
            throw operation
        }
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
            throw ExactProbeError(detail: "system free memory below30% or guard unavailable")
        }
        return value
    }
}
