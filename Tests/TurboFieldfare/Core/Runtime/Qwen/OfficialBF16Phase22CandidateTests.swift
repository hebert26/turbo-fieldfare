import CryptoKit
import Foundation
import Testing
@testable import TurboFieldfare

private struct P22Route: Codable {
    let layer: Int
    let routerLogitsFP32: [Float]
    let top8ExpertIds: [Int]
    let top8WeightsFP32: [Float]
    let selectedVsUnselectedLogitMarginFP32: Float
}

private struct P22Step: Codable {
    let stepIndex: Int
    let position: Int
    let inputTokenId: Int32
    let rawLogitsFile: String
    let rawLogitsSHA256: String
    let publicFloat16LogitsFile: String
    let publicFloat16LogitsSHA256: String
    let rawArgmaxTokenId: Int
    let publicArgmaxTokenId: Int
    let sampledTokenId: Int32
    let argmaxMarginFP32: Float
    let publicArgmaxMarginFP16: Float
    let activations: [P22Activation]
    let routes: [P22Route]
}

private struct P22Activation: Codable {
    let layer: Int
    let stage: String
    let file: String
    let count: Int
    let sha256: String
}

private struct P22ActivationKey: Hashable {
    let layer: Int
    let stage: String
}

private struct P22Receipt: Codable {
    let schemaVersion: Int
    let modelFamily: String
    let sourceDescriptorContentSHA256: String
    let sourceMarkerSHA256: String
    let sourceShardSetSHA256: String
    let sourceChecksumManifestSHA256: String
    let sourceRoot: String
    let sourceKind: String
    let registrationPath: String
    let integrityPolicy: String
    let promptText: String
    let promptTokenIds: [Int32]
    let temperature: Float
    let maxNewTokens: Int
    let expertCacheSlots: Int
    let expertCachePolicy: String
    let steps: [P22Step]
}

private final class P22Capture: @unchecked Sendable {
    private let lock = NSLock()
    private let linearCaptureLayer: Int
    private var routes: [Int: [P22Route]] = [:]
    private var raw: [Int: (Int32, [Float])] = [:]
    private var sampled: [Int: ([UInt16], Int32)] = [:]
    private var activations: [Int: [P22ActivationKey: [Float]]] = [:]
    private var duplicate = false

    init(linearCaptureLayer: Int = 0) {
        self.linearCaptureLayer = linearCaptureLayer
    }

    func route(position: Int, layer: Int, logits: [Float], ids: [Int],
               weights: [Float], margin: Float) {
        lock.lock()
        routes[position, default: []].append(P22Route(
            layer: layer, routerLogitsFP32: logits, top8ExpertIds: ids,
            top8WeightsFP32: weights,
            selectedVsUnselectedLogitMarginFP32: margin))
        lock.unlock()
    }

    func logits(position: Int, token: Int32, values: [Float]) {
        lock.lock()
        if raw[position] != nil { duplicate = true }
        raw[position] = (token, values)
        lock.unlock()
    }

    func sample(step: Int, bits: [UInt16], token: Int32) {
        lock.lock()
        if sampled[step] != nil { duplicate = true }
        sampled[step] = (bits, token)
        lock.unlock()
    }

    func activation(position: Int, layer: Int, stage: String, values: [Float]) {
        // Keep the first linear-attention layer's actual inputs and outputs,
        // plus the final state and exact LM-head input.
        let isLayerBoundary = layer >= 0 && [
            "input-norm", "mixer", "residual-after-mixer", "post-norm", "residual-after-moe",
        ].contains(stage)
        let isLinearInternals = layer == linearCaptureLayer && stage.hasPrefix("linear.")
        let isFinalActivation = layer == -1 &&
            (stage == "final-hidden" || stage == "final-norm")
        guard isLayerBoundary || isLinearInternals || isFinalActivation else {
            return
        }
        lock.lock()
        let key = P22ActivationKey(layer: layer, stage: stage)
        if activations[position]?[key] != nil { duplicate = true }
        activations[position, default: [:]][key] = values
        lock.unlock()
    }

    func snapshot() -> (routes: [Int: [P22Route]],
                        raw: [Int: (Int32, [Float])],
                        sampled: [Int: ([UInt16], Int32)],
                        activations: [Int: [P22ActivationKey: [Float]]], duplicate: Bool) {
        lock.lock()
        defer { lock.unlock() }
        return (routes, raw, sampled, activations, duplicate)
    }

    var hooks: QwenOfficialSourceTransactionHooks {
        QwenOfficialSourceTransactionHooks(
            observeRoute: { [self] position, layer, logits, ids, weights, margin in
                route(position: position, layer: layer, logits: logits,
                      ids: ids, weights: weights, margin: margin)
            },
            observeRawLogits: { [self] position, token, values in
                logits(position: position, token: token, values: values)
            },
            observePublicLogitsAndSample: { [self] step, bits, token in
                sample(step: step, bits: bits, token: token)
            },
            observeActivation: { [self] position, layer, stage, values in
                activation(position: position, layer: layer, stage: stage, values: values)
            })
    }
}

private enum P22CandidateError: Error {
    case incompleteCapture(String)
    case wrongRegistration
}

private func p22Data<T: FixedWidthInteger>(_ values: [T]) -> Data {
    var bytes = Data(capacity: values.count * MemoryLayout<T>.size)
    for value in values {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { bytes.append(contentsOf: $0) }
    }
    return bytes
}

private func p22Argmax(_ values: [Float]) -> (Int, Float) {
    let ranked = values.indices.sorted {
        values[$0] == values[$1] ? $0 < $1 : values[$0] > values[$1]
    }
    return (ranked[0], values[ranked[0]] - values[ranked[1]])
}

private func p22ActivationCounts(
    _ architecture: QwenTextArchitecture, linearCaptureLayer: Int
) -> [P22ActivationKey: Int] {
    let hidden = architecture.hiddenSize
    let keyWidth = architecture.linearKeyHeads * architecture.linearKeyDimension
    let valueWidth = architecture.linearValueHeads * architecture.linearValueDimension
    let channels = 2 * keyWidth + valueWidth
    let queryWidth = architecture.linearValueHeads * architecture.linearKeyDimension
    let heads = architecture.linearValueHeads
    let linear: [String: Int] = [
        "linear.input": hidden, "linear.qkv": channels, "linear.z": valueWidth,
        "linear.b": heads, "linear.a": heads, "linear.convolved": channels,
        "linear.query-raw": queryWidth, "linear.key-raw": queryWidth,
        "linear.value": valueWidth, "linear.beta": heads,
        "linear.log-decay": heads, "linear.core": valueWidth,
        "linear.gated": valueWidth, "linear.output": hidden,
    ]
    var result: [P22ActivationKey: Int] = [:]
    let boundaryStages = [
        "input-norm", "mixer", "residual-after-mixer", "post-norm", "residual-after-moe",
    ]
    for layer in architecture.layerKinds.indices {
        for stage in boundaryStages {
            result[P22ActivationKey(layer: layer, stage: stage)] = hidden
        }
    }
    for (stage, count) in linear {
        result[P22ActivationKey(layer: linearCaptureLayer, stage: stage)] = count
    }
    result[P22ActivationKey(layer: -1, stage: "final-hidden")] = hidden
    result[P22ActivationKey(layer: -1, stage: "final-norm")] = hidden
    return result
}

@Suite(.serialized) struct OfficialBF16Phase22CandidateTests {
    @Test func captureHooksObserveActualTinyRunnerAndSampler() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.expectedResidentBytes)
        let linearCaptureLayer = try #require(model.architecture.layerKinds.firstIndex(
            of: .linearAttention))
        let capture = P22Capture(linearCaptureLayer: linearCaptureLayer)
        let session = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 8,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: capture.hooks)
        let result = try await session.generatePreparedTurn(
            promptTokenIDs: [1], config: .qwenRaw(
                maxNewTokens: 2, temperature: 0, stopTokenIDs: []))
        let observed = capture.snapshot()
        #expect(result.acceptedGeneratedTokenIDs.count == 2)
        #expect(observed.raw.count == 2)
        #expect(!observed.duplicate)
        #expect(observed.sampled.count == 2)
        #expect(observed.activations.count == 2)
        #expect(observed.routes.count == 2)
        #expect(observed.routes[0]?.count == 2)
        #expect(observed.routes[1]?.count == 2)
        #expect(observed.raw[0]?.0 == 1)
        #expect(observed.raw[1]?.0 == result.acceptedGeneratedTokenIDs[0])
        #expect(observed.sampled[0]?.1 == result.acceptedGeneratedTokenIDs[0])
        #expect(observed.sampled[1]?.1 == result.acceptedGeneratedTokenIDs[1])
        for position in 0..<2 {
            let stages = try #require(observed.activations[position])
            let expectedCounts = p22ActivationCounts(
                model.architecture, linearCaptureLayer: linearCaptureLayer)
            #expect(Set(stages.keys) == Set(expectedCounts.keys))
            #expect(stages.allSatisfy { key, values in
                values.count == (expectedCounts[key] ?? -1)
                    && values.allSatisfy(\.isFinite)
            })
            #expect(stages[P22ActivationKey(layer: linearCaptureLayer, stage: "input-norm")]
                    == stages[P22ActivationKey(layer: linearCaptureLayer, stage: "linear.input")])
            let hidden = try #require(stages[P22ActivationKey(layer: -1, stage: "final-hidden")])
            // This verifies that the taps bracket the real final norm; the
            // frozen source RMSNorm tests cover its numerical behavior.
            let expectedFinalNormFromCapturedHidden = try qwenOfficialSourceRMSNorm(
                hidden, model.finalNorm)
            #expect(stages[P22ActivationKey(layer: -1, stage: "final-norm")]
                    == expectedFinalNormFromCapturedHidden)
        }

        let ordinary = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: model, context: context, maxContext: 8,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let ordinaryResult = try await ordinary.generatePreparedTurn(
            promptTokenIDs: [1], config: .qwenRaw(
                maxNewTokens: 2, temperature: 0, stopTokenIDs: []))
        #expect(ordinaryResult.acceptedGeneratedTokenIDs
                == result.acceptedGeneratedTokenIDs)
        let ordinaryState = try await ordinary.diagnosticSnapshot()
        #expect(ordinaryState.currentLogits == observed.raw[1]?.1)
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_P22_REAL_CANDIDATE"] == "1"))
    func captureRegisteredBF16TwoStepCandidate() async throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let registration = root.appendingPathComponent(
            "scratch/qwen3.6-35b-a3b.gturbo", isDirectory: true)
        guard ProcessInfo.processInfo.environment["TURBO_P22_REGISTRATION"]
                == registration.path else { throw P22CandidateError.wrongRegistration }
        guard let outputPath = ProcessInfo.processInfo.environment["TURBO_P22_OUTPUT"] else {
            throw P22CandidateError.incompleteCapture("missing output directory")
        }
        let output = URL(fileURLWithPath: outputPath, isDirectory: true)
        guard (try FileManager.default.contentsOfDirectory(
            atPath: output.path)).isEmpty else {
            throw P22CandidateError.incompleteCapture("output directory is not empty")
        }
        let context = try MetalContext()
        let bundle = try ModelFamilyRuntime.loadBundle(
            directoryURL: registration, device: context.device,
            streamingMode: .pread(slotCount: 16), expertCachePolicy: .lfu,
            integrityPolicy: .sizeCheckTrustedReceipt)
        guard case .qwenOfficialSource(let model) = bundle.runtime,
              let codec = bundle.qwenCodec,
              let identity = bundle.sourceIdentity,
              codec.tokenizer.encode("Hello") == [9419] else {
            throw P22CandidateError.incompleteCapture("trusted source or frozen token mismatch")
        }
        guard model.architecture.layerKinds.first == .linearAttention else {
            throw P22CandidateError.incompleteCapture("official layer 0 is not linear attention")
        }
        let capture = P22Capture(linearCaptureLayer: 0)
        let session = try await QwenOfficialSourceConversationGenerationSession(
            model: model, codec: codec, sourceIdentity: identity,
            context: model.context, maxContext: 4, expertSlotCount: 16,
            modelDirectoryURL: registration, hooks: capture.hooks)
        let result = try await session.generateAuditedSourceTokenTurn(
            promptTokenIDs: [9419], config: .qwenRaw(
                maxNewTokens: 2, temperature: 0, stopTokenIDs: []))
        let observed = capture.snapshot()
        guard !observed.duplicate, result.acceptedGeneratedTokenIDs.count == 2,
              observed.raw.count == 2, observed.sampled.count == 2,
              observed.activations.count == 2,
              observed.routes.count == 2 else {
            throw P22CandidateError.incompleteCapture("expected exactly two complete steps")
        }
        var steps: [P22Step] = []
        let expectedActivationCounts = p22ActivationCounts(
            model.architecture, linearCaptureLayer: 0)
        for index in 0..<2 {
            guard let (input, raw) = observed.raw[index],
                  let (publicBits, sampled) = observed.sampled[index],
                  let activations = observed.activations[index],
                  let routes = observed.routes[index],
                  raw.count == 248_320, publicBits.count == 248_320,
                  Set(activations.keys) == Set(expectedActivationCounts.keys),
                  activations.allSatisfy({ key, values in
                      values.count == (expectedActivationCounts[key] ?? -1)
                          && values.allSatisfy(\.isFinite)
                  }),
                  routes.count == 40,
                  routes.map(\.layer).sorted() == Array(0..<40),
                  routes.allSatisfy({ $0.routerLogitsFP32.count == 256
                      && $0.top8ExpertIds.count == 8
                      && $0.top8WeightsFP32.count == 8 }),
                  input == (index == 0 ? 9419 : result.acceptedGeneratedTokenIDs[0]),
                  sampled == result.acceptedGeneratedTokenIDs[index] else {
                throw P22CandidateError.incompleteCapture("step \(index) geometry or lineage")
            }
            let rawData = p22Data(raw.map(\.bitPattern))
            let publicData = p22Data(publicBits)
            let rawName = "step-\(index).raw-fp32-le.bin"
            let publicName = "step-\(index).public-fp16-le.bin"
            try rawData.write(to: output.appendingPathComponent(rawName), options: .atomic)
            try publicData.write(to: output.appendingPathComponent(publicName), options: .atomic)
            var activationRecords: [P22Activation] = []
            for (key, values) in activations.sorted(by: {
                $0.key.layer == $1.key.layer
                    ? $0.key.stage < $1.key.stage : $0.key.layer < $1.key.layer
            }) {
                let layer = key.layer
                let stage = key.stage
                let bits = values.map(\.bitPattern)
                let bytes = p22Data(bits)
                let fileStage = stage.replacingOccurrences(of: ".", with: "-")
                let layerPrefix = layer < 0 ? "" : "layer-\(layer)."
                let file = "step-\(index).\(layerPrefix)\(fileStage).fp32-le.bin"
                try bytes.write(to: output.appendingPathComponent(file), options: .atomic)
                activationRecords.append(P22Activation(
                    layer: layer, stage: stage, file: file, count: values.count,
                    sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()))
            }
            let rawBest = p22Argmax(raw)
            let publicValues = publicBits.map { Float(Float16(bitPattern: $0)) }
            let publicBest = p22Argmax(publicValues)
            steps.append(P22Step(
                stepIndex: index, position: index, inputTokenId: input,
                rawLogitsFile: rawName,
                rawLogitsSHA256: SHA256.hash(data: rawData).map {
                    String(format: "%02x", $0) }.joined(),
                publicFloat16LogitsFile: publicName,
                publicFloat16LogitsSHA256: SHA256.hash(data: publicData).map {
                    String(format: "%02x", $0) }.joined(),
                rawArgmaxTokenId: rawBest.0,
                publicArgmaxTokenId: publicBest.0,
                sampledTokenId: sampled,
                argmaxMarginFP32: rawBest.1,
                publicArgmaxMarginFP16: publicBest.1,
                activations: activationRecords,
                routes: routes.sorted { $0.layer < $1.layer }))
        }
        guard steps[0].sampledTokenId == steps[1].inputTokenId,
              steps.allSatisfy({ $0.sampledTokenId == $0.publicArgmaxTokenId }) else {
            throw P22CandidateError.incompleteCapture("production sample/public argmax mismatch")
        }
        let receipt = P22Receipt(
            schemaVersion: 1, modelFamily: "Qwen/Qwen3.6-35B-A3B original BF16",
            sourceDescriptorContentSHA256: identity.descriptorContentSHA256,
            sourceMarkerSHA256: identity.markerSHA256,
            sourceShardSetSHA256: identity.shardSetSHA256,
            sourceChecksumManifestSHA256: identity.checksumManifestSHA256,
            sourceRoot: identity.sourceRoot,
            sourceKind: "official-safetensors-bf16-v1",
            registrationPath: registration.path,
            integrityPolicy: "sizeCheckTrustedReceipt",
            promptText: "Hello", promptTokenIds: [9419],
            temperature: 0, maxNewTokens: 2,
            expertCacheSlots: 16, expertCachePolicy: "lfu", steps: steps)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(
            to: output.appendingPathComponent("receipt.json"), options: .atomic)
    }
}
