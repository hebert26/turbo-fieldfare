import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Main must first pass Phase 22, finish the independent CPU oracle, and run
/// AGENTS.md preflight. This single opt-in consumer owns one source model.
/// It proves patches/structure/features only, not image-conditioned logits.
@Suite(.serialized) struct QwenOfficialSourceRealVisionParityTests {
    private static let keys = [
        "TURBO_FIELDFARE_P23_REGISTRATION", "TURBO_FIELDFARE_P23_VISION",
        "TURBO_FIELDFARE_P23_REFERENCE", "TURBO_FIELDFARE_P23_REFERENCE_SHA256",
        "TURBO_FIELDFARE_P23_REQUEST_SHA256", "TURBO_FIELDFARE_P23_OUTPUT",
    ]

    @Test(.enabled(if: keys.allSatisfy {
        !(ProcessInfo.processInfo.environment[$0] ?? "").isEmpty
    }, "set all explicit original-BF16 P23 inputs after Phase22 acceptance"))
    func authenticSourcePatchesStructureAndFeaturesMatchOfficialCPU() async throws {
        let environment = ProcessInfo.processInfo.environment
        let mergerDiagnosticRow: Int?
        if let value = environment["TURBO_FIELDFARE_P23_MERGER_DIAGNOSTIC_ROW"] {
            guard let row = Int(value), row >= 0,
                  environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_DIAGNOSTICS"] != "1",
                  environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_ARRAY_DIAGNOSTICS"] != "1" else {
                throw P23SourceVisionError.invalid("invalid or conflicting merger diagnostic row")
            }
            mergerDiagnosticRow = row
        } else {
            mergerDiagnosticRow = nil
        }
        let towerDiagnostics = environment["TURBO_FIELDFARE_P23_TOWER_DIAGNOSTICS"] == "1"
        guard !towerDiagnostics || (mergerDiagnosticRow == nil
            && environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_DIAGNOSTICS"] != "1"
            && environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_ARRAY_DIAGNOSTICS"] != "1") else {
            throw P23SourceVisionError.invalid("conflicting tower diagnostic options")
        }
        let initialStageDiagnostics = environment["TURBO_FIELDFARE_P23_INITIAL_STAGE_DIAGNOSTICS"] == "1"
        guard !initialStageDiagnostics || (!towerDiagnostics && mergerDiagnosticRow == nil
            && environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_DIAGNOSTICS"] != "1"
            && environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_ARRAY_DIAGNOSTICS"] != "1") else {
            throw P23SourceVisionError.invalid("conflicting initial-stage diagnostic options")
        }
        let firstBlockArrayDiagnostics = environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_ARRAY_DIAGNOSTICS"] == "1"
        guard !firstBlockArrayDiagnostics || (!towerDiagnostics && mergerDiagnosticRow == nil
            && environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_DIAGNOSTICS"] != "1"
            && !initialStageDiagnostics) else {
            throw P23SourceVisionError.invalid("conflicting first-block array diagnostic options")
        }
        func required(_ key: String) throws -> String {
            guard let value = environment[key], !value.isEmpty else {
                throw P23SourceVisionError.invalid("missing \(key)")
            }
            return value
        }
        let registration = try p23Canonical(URL(fileURLWithPath: required(Self.keys[0])))
        let companion = try p23Canonical(URL(fileURLWithPath: required(Self.keys[1])))
        let referenceURL = try p23Canonical(URL(fileURLWithPath: required(Self.keys[2])))
        let referenceSHA = try required(Self.keys[3])
        let requestSHA = try required(Self.keys[4])
        let output = try p23Canonical(URL(fileURLWithPath: required(Self.keys[5])))
        let referenceBytes = try p23Read(referenceURL, maximum: 4 * 1024 * 1024)
        guard p23SHA(referenceBytes) == referenceSHA else {
            throw P23SourceVisionError.invalid("independent reference receipt digest changed")
        }
        let reference = try JSONDecoder().decode(P23SourceVisionReference.self, from: referenceBytes)
        try reference.validate(registration: registration, companion: companion, requestSHA: requestSHA)
        try OfficialQwenSourceVisionMetadataProbe.verify(textModelURL: registration, visionURL: companion)
        guard p23SHA(try p23Read(companion.appendingPathComponent("manifest.json"), maximum: 1024 * 1024))
                == reference.visionManifestSHA256 else {
            throw P23SourceVisionError.invalid("current source vision manifest differs")
        }
        let context = try MetalContext()
        try VisionRuntime.requireSupportedDevice(context.device)
        let preprocessor = QwenImagePreprocessor(device: context.device)
        var prepared: [P23SourceVisionCase] = []
        // Preprocess every accepted image and gate exact patch identity before
        // admitting even one text-source model or reading tower weights.
        for item in reference.results {
            let image = try p23Canonical(URL(fileURLWithPath: item.imagePath))
            let plan = try preprocessor.plan(fileURL: image)
            let pixels = try preprocessor.preprocess(plan)
            let geometry = pixels.geometry
            let patchByteCount = geometry.patchRows * 1536 * 2
            let patchBytes = Data(bytes: pixels.patchesBF16.contents(), count: patchByteCount)
            let actualHash = p23SHA(patchBytes)
            try patchBytes.write(to: output.appendingPathComponent(item.id + ".candidate-patches-bf16-le.bin"),
                                 options: .withoutOverwriting)
            let patchMatch = pixels.imageDigest == item.imageSHA256
                && pixels.metadata.encodedBytes == item.imageBytes
                && [pixels.metadata.encodedWidth, pixels.metadata.encodedHeight] == item.encodedSize
                && pixels.metadata.orientation == item.orientation
                && [pixels.metadata.orientedWidth, pixels.metadata.orientedHeight] == item.orientedSize
                && [geometry.processedWidth, geometry.processedHeight] == item.processedSize
                && [geometry.gridT, geometry.gridH, geometry.gridW] == item.gridTHW
                && item.patches.shape == [geometry.patchRows, 1536]
                && item.patches.byteCount == patchByteCount && item.patches.sha256 == actualHash
            try p23WriteReceipt([
                "kind": "qwen36-original-bf16-vision-candidate-preprocessing-v1",
                "id": item.id, "referenceSHA256": referenceSHA, "requestSHA256": requestSHA,
                "patchByteCount": patchByteCount, "patchSHA256": actualHash,
                "exactPreprocessingMatch": patchMatch,
                "gridTHW": [geometry.gridT, geometry.gridH, geometry.gridW],
            ], to: output.appendingPathComponent(item.id + ".preprocessing.json"))
            guard patchMatch else {
                throw P23SourceVisionError.invalid("exact preprocessing gate failed for \(item.id); no tower ran")
            }
            let referencePatches = try p23Read(referenceURL.deletingLastPathComponent()
                .appendingPathComponent(item.patches.file), maximum: 16 * 1024 * 1024)
            guard referencePatches.count == patchByteCount, p23SHA(referencePatches) == item.patches.sha256 else {
                throw P23SourceVisionError.invalid("reference patch artifact changed")
            }
            prepared.append(item)
        }
        let started = ContinuousClock.now
        let bundle = try ModelFamilyRuntime.loadBundle(
            directoryURL: registration, device: context.device,
            streamingMode: .pread(slotCount: 16), expertCachePolicy: .lfu,
            integrityPolicy: .sizeCheckTrustedReceipt)
        guard case .qwenOfficialSource(let model) = bundle.runtime,
              let identity = bundle.sourceIdentity,
              identity.descriptorContentSHA256 == reference.sourceDescriptorContentSHA256,
              let codec = bundle.qwenCodec else {
            throw P23SourceVisionError.invalid("loaded model is not the verified original-BF16 source")
        }
        let store = try QwenOfficialSourceVisionWeightStore.open(directoryURL: companion, model: model)
        let stageCapture = P23SourceVisionStageCapture()
        let mergerCapture = P23SourceVisionMergerCapture()
        let towerCapture = P23SourceVisionTowerCapture()
        let initialStageCapture = P23SourceVisionTowerCapture()
        let firstBlockArrayCapture = P23SourceVisionTowerCapture()
        var hooks = QwenVisionExecutionHooks.none
        if environment["TURBO_FIELDFARE_P23_FIRST_BLOCK_DIAGNOSTICS"] == "1" {
            hooks.firstBlockDiagnostics = { stageCapture.append($0) }
        }
        if let row = mergerDiagnosticRow {
            hooks.mergerDiagnosticRow = row
            hooks.mergerDiagnostics = { mergerCapture.append($0) }
        }
        if towerDiagnostics {
            hooks.towerDiagnostics = { towerCapture.append($0) }
        }
        if initialStageDiagnostics {
            hooks.initialStageDiagnostics = { initialStageCapture.append($0) }
        }
        if firstBlockArrayDiagnostics {
            hooks.firstBlockStageDiagnostics = { firstBlockArrayCapture.append($0) }
        }
        let runtime = try QwenVisionRuntime(context: model.context, sourceStore: store, executionHooks: hooks)
        var failedCases: [String] = []
        for item in prepared {
            let image = try p23Canonical(URL(fileURLWithPath: item.imagePath))
            let featureStarted = ContinuousClock.now
            func saveMergerDiagnostics(stopReason: String) throws {
                try p23WriteReceipt([
                    "kind": "qwen36-original-bf16-vision-merger-diagnostics-v1", "id": item.id,
                    "acceptanceScope": "diagnostics-only", "qualificationPassed": false,
                    "referenceSHA256": referenceSHA, "requestSHA256": requestSHA,
                    "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
                    "row": mergerDiagnosticRow ?? -1,
                    "stages": try mergerCapture.writeArtifacts(id: item.id, output: output),
                    "stopReason": stopReason,
                ], to: output.appendingPathComponent(item.id + ".merger-diagnostics.json"))
            }
            func saveTowerDiagnostics(stopReason: String) throws {
                try p23WriteReceipt([
                    "kind": "qwen36-original-bf16-vision-tower-diagnostics-v1", "id": item.id,
                    "acceptanceScope": "diagnostics-only", "qualificationPassed": false,
                    "referenceSHA256": referenceSHA, "requestSHA256": requestSHA,
                    "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
                    "stages": try towerCapture.writeArtifacts(id: item.id, output: output),
                    "stopReason": stopReason,
                ], to: output.appendingPathComponent(item.id + ".tower-diagnostics.json"))
            }
            func saveInitialStageDiagnostics(stopReason: String) throws {
                try p23WriteReceipt([
                    "kind": "qwen36-original-bf16-vision-initial-stage-diagnostics-v1", "id": item.id,
                    "acceptanceScope": "diagnostics-only", "qualificationPassed": false,
                    "referenceSHA256": referenceSHA, "requestSHA256": requestSHA,
                    "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
                    "stages": try initialStageCapture.writeArtifacts(id: item.id, output: output),
                    "stopReason": stopReason,
                ], to: output.appendingPathComponent(item.id + ".initial-stage-diagnostics.json"))
            }
            func saveFirstBlockArrayDiagnostics(stopReason: String) throws {
                try p23WriteReceipt([
                    "kind": "qwen36-original-bf16-vision-first-block-array-diagnostics-v1", "id": item.id,
                    "acceptanceScope": "diagnostics-only", "qualificationPassed": false,
                    "referenceSHA256": referenceSHA, "requestSHA256": requestSHA,
                    "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
                    "imageSHA256": item.imageSHA256, "gridTHW": item.gridTHW,
                    "stages": try firstBlockArrayCapture.writeArtifacts(
                        id: item.id, output: output,
                        expectedStages: [
                            ("block-0.input", 1152),
                            ("block-0.norm1", 1152),
                            ("block-0.qkv", 3456),
                            ("block-0.rotate-qk", 3456),
                            ("block-0.attention-context", 1152),
                            ("block-0.attention-residual", 1152),
                            ("block-0.norm2", 1152),
                            ("block-0.fc1-pre-gelu", 4304),
                            ("block-0.fc1-gelu", 4304),
                            ("block-0.output", 1152),
                        ]),
                    "stopReason": stopReason,
                ], to: output.appendingPathComponent(item.id + ".first-block-array-diagnostics.json"))
            }
            let actual: QwenVisionFeatures
            do {
                actual = try await runtime.processReferenceImage(
                    fileURL: image, device: model.context.device,
                    expectedPatchSHA: item.patches.sha256)
            } catch {
                if firstBlockArrayDiagnostics {
                    try saveFirstBlockArrayDiagnostics(stopReason: String(describing: error))
                }
                if initialStageDiagnostics {
                    try saveInitialStageDiagnostics(stopReason: String(describing: error))
                }
                if towerDiagnostics {
                    try saveTowerDiagnostics(stopReason: String(describing: error))
                }
                if mergerDiagnosticRow != nil {
                    try saveMergerDiagnostics(stopReason: String(describing: error))
                }
                if hooks.firstBlockDiagnostics != nil {
                    try p23WriteReceipt([
                        "kind": "qwen36-original-bf16-vision-first-block-diagnostics-v1", "id": item.id,
                        "acceptanceScope": "diagnostics-only", "qualificationPassed": false,
                        "referenceSHA256": referenceSHA, "requestSHA256": requestSHA,
                        "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
                        "stages": stageCapture.receipts(), "stopReason": String(describing: error),
                    ], to: output.appendingPathComponent(item.id + ".first-block-diagnostics.json"))
                }
                throw error
            }
            if firstBlockArrayDiagnostics {
                try saveFirstBlockArrayDiagnostics(stopReason: "consumer stopped after block-0 array capture")
                throw P23SourceVisionError.invalid("source vision first-block array diagnostic completed without qualification")
            }
            if initialStageDiagnostics {
                try saveInitialStageDiagnostics(stopReason: "consumer stopped after initial-stage capture")
                throw P23SourceVisionError.invalid("source vision initial-stage diagnostic completed without qualification")
            }
            if towerDiagnostics {
                try saveTowerDiagnostics(stopReason: "consumer stopped after tower capture")
                throw P23SourceVisionError.invalid("source vision tower diagnostic completed without qualification")
            }
            if mergerDiagnosticRow != nil {
                try saveMergerDiagnostics(stopReason: "consumer stopped after merger capture")
                throw P23SourceVisionError.invalid("source vision merger diagnostic completed without qualification")
            }
            let actualValues = actual.owner.features()
            var raw = Data(capacity: actualValues.count * 4)
            for value in actualValues {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { raw.append(contentsOf: $0) }
            }
            let rawFile = item.id + ".candidate-features-fp32-le.bin"
            try raw.write(to: output.appendingPathComponent(rawFile), options: .withoutOverwriting)
            let nonfiniteIndices = actualValues.indices.filter { !actualValues[$0].isFinite }
            try p23WriteReceipt([
                "kind": "qwen36-original-bf16-vision-feature-diagnostics-v1", "id": item.id,
                "referenceSHA256": referenceSHA, "requestSHA256": requestSHA,
                "shape": [actual.tokenCount, actual.hiddenSize],
                "expectedFeatureCount": item.features.byteCount / 4,
                "actualFeatureCount": actualValues.count, "featureByteCount": raw.count,
                "featureFile": rawFile, "featureSHA256": p23SHA(raw),
                "finiteCount": actualValues.count - nonfiniteIndices.count,
                "nonfiniteCount": nonfiniteIndices.count,
                "nanCount": actualValues.filter(\.isNaN).count,
                "positiveInfinityCount": actualValues.filter { $0 == .infinity }.count,
                "negativeInfinityCount": actualValues.filter { $0 == -.infinity }.count,
                "firstNonfiniteIndex": nonfiniteIndices.first ?? -1,
                "nonfiniteSamples": nonfiniteIndices.prefix(16).map {
                    ["index": $0, "bitPattern": UInt64(actualValues[$0].bitPattern)] as [String: Any]
                },
                "orderedLayerIntermediatesBitPatterns": actual.diagnostics.orderedLayerIntermediates.map {
                    $0.map { UInt64($0.bitPattern) }
                },
            ], to: output.appendingPathComponent(item.id + ".feature-diagnostics.json"))
            let referenceData = try p23Read(referenceURL.deletingLastPathComponent()
                .appendingPathComponent(item.features.file), maximum: 16 * 1024 * 1024)
            guard p23SHA(referenceData) == item.features.sha256,
                  referenceData.count == item.features.byteCount,
                  item.features.shape == [actual.tokenCount, actual.hiddenSize],
                  referenceData.count == actualValues.count * 4 else {
                throw P23SourceVisionError.invalid("reference merger artifact shape or digest changed")
            }
            let expectedValues = stride(from: 0, to: referenceData.count, by: 4).map { offset in
                referenceData.withUnsafeBytes { bytes in
                    Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)))
                }
            }
            let message = ModelChatMessage(role: .user, content: .parts([
                .image(.init(id: item.id)), .text(item.promptText),
            ]))
            let template = try codec.encodePrompt(messages: [message], tools: [], options: .init(enableThinking: false))
            let normalized = try normalizeQwenCodecImageFrames(template, architecture: model.visionArchitecture)
            let prompt = try MultimodalPromptRenderer.expandingQwenImageTokens(
                normalized, features: [actual], architecture: model.visionArchitecture)
            let pads = prompt.imageSpans.flatMap { Array($0.tokenRange) }
            let modality = prompt.effectiveTokenIDs.indices.map { pads.contains($0) ? 1 : 0 }
            let structureMatch = prompt.effectiveTokenIDs == item.tokenIds
                && modality == item.modalityIds && pads == item.padRows
                && prompt.positionPlan.positions.map(\.values) == item.positions
                && prompt.positionPlan.textRoPEDelta == item.textRoPEDelta
                && actual.owner.positions().map(\.values) == item.featurePositions
            let metrics = try P23SourceVisionFeatureComparison.compare(actualValues, expectedValues)
            let featureDuration = featureStarted.duration(to: .now)
            let totalDuration = started.duration(to: .now)
            try p23WriteReceipt([
                "kind": "qwen36-original-bf16-vision-candidate-v1", "id": item.id,
                "acceptanceScope": "patches-structure-features-only",
                "sourceDescriptorContentSHA256": identity.descriptorContentSHA256,
                "sourceVerification": "trusted-receipt", "referenceSHA256": referenceSHA,
                "requestSHA256": requestSHA, "visionManifestSHA256": reference.visionManifestSHA256,
                "shape": [actual.tokenCount, actual.hiddenSize], "featureSHA256": p23SHA(raw),
                "exactStructureMatch": structureMatch, "tokenIds": prompt.effectiveTokenIDs,
                "modalityIds": modality, "padRows": pads,
                "positions": prompt.positionPlan.positions.map(\.values),
                "textRoPEDelta": prompt.positionPlan.textRoPEDelta,
                "featureFailureCount": metrics.failureCount, "maximumAbsoluteError": metrics.maximumAbsoluteError,
                "maximumFailureIndex": metrics.maximumFailureIndex ?? -1,
                "firstFailureIndex": metrics.firstFailureIndex ?? -1,
                "firstFailureRow": metrics.firstFailureIndex.map { $0 / actual.hiddenSize } ?? -1,
                "firstFailureColumn": metrics.firstFailureIndex.map { $0 % actual.hiddenSize } ?? -1,
                "featureSeconds": p23Seconds(featureDuration), "elapsedSinceLoadSeconds": p23Seconds(totalDuration),
                "passed": structureMatch && metrics.failureCount == 0,
            ], to: output.appendingPathComponent(item.id + ".candidate.json"))
            if !structureMatch || metrics.failureCount != 0 {
                failedCases.append(item.id)
            }
            try model.revalidateSource()
            print("P23_ORIGINAL_BF16_VISION image=\(item.id) rows=\(actual.tokenCount) structure=\(structureMatch ? "exact" : "mismatch") featureFailures=\(metrics.failureCount) maxAbs=\(metrics.maximumAbsoluteError)")
        }
        guard failedCases.isEmpty else {
            throw P23SourceVisionError.invalid("source image structure/feature qualification failed for \(failedCases.joined(separator: ", "))")
        }
    }
}

private final class P23SourceVisionTowerCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stages: [QwenVisionTowerDiagnostic] = []

    func append(_ stage: QwenVisionTowerDiagnostic) {
        lock.lock()
        defer { lock.unlock() }
        stages.append(stage)
    }

    func writeArtifacts(
        id: String,
        output: URL,
        expectedStages: [(name: String, width: Int)]? = nil
    ) throws -> [[String: Any]] {
        lock.lock()
        let captured = stages
        lock.unlock()
        if let expectedStages {
            guard captured.count == expectedStages.count,
                  zip(captured, expectedStages).allSatisfy({ stage, expected in
                      stage.stage == expected.name && stage.rows == 308
                          && stage.width == expected.width
                          && stage.values.count == stage.rows * stage.width
                  }) else {
                throw P23SourceVisionError.invalid("first block array stages differ from the frozen layout")
            }
        }
        return try captured.map { stage in
            guard stage.rows > 0, stage.width > 0,
                  stage.values.count == stage.rows * stage.width else {
                throw P23SourceVisionError.invalid("invalid tower diagnostic shape")
            }
            var raw = Data(capacity: stage.values.count * 4)
            for value in stage.values {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { raw.append(contentsOf: $0) }
            }
            let file = id + "." + stage.stage + ".fp32-le.bin"
            try raw.write(to: output.appendingPathComponent(file), options: .withoutOverwriting)
            let nonfinite = stage.values.indices.filter { !stage.values[$0].isFinite }
            return [
                "stage": stage.stage, "shape": [stage.rows, stage.width], "count": stage.values.count,
                "file": file, "byteCount": raw.count, "sha256": p23SHA(raw),
                "finiteCount": stage.values.count - nonfinite.count,
                "nonfiniteCount": nonfinite.count, "firstNonfiniteIndex": nonfinite.first ?? -1,
            ]
        }
    }
}

private final class P23SourceVisionMergerCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stages: [QwenVisionMergerDiagnostic] = []

    func append(_ stage: QwenVisionMergerDiagnostic) {
        lock.lock()
        defer { lock.unlock() }
        stages.append(stage)
    }

    func writeArtifacts(id: String, output: URL) throws -> [[String: Any]] {
        lock.lock()
        let captured = stages
        lock.unlock()
        return try captured.map { stage in
            var raw = Data(capacity: stage.values.count * 4)
            for value in stage.values {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { raw.append(contentsOf: $0) }
            }
            let file = id + "." + stage.stage + ".row-\(stage.row).fp32-le.bin"
            try raw.write(to: output.appendingPathComponent(file), options: .withoutOverwriting)
            let nonfinite = stage.values.indices.filter { !stage.values[$0].isFinite }
            return [
                "stage": stage.stage, "row": stage.row, "count": stage.values.count,
                "file": file, "byteCount": raw.count, "sha256": p23SHA(raw),
                "finiteCount": stage.values.count - nonfinite.count,
                "nonfiniteCount": nonfinite.count, "firstNonfiniteIndex": nonfinite.first ?? -1,
                "bitPatterns": stage.values.map { UInt64($0.bitPattern) },
            ]
        }
    }
}

private final class P23SourceVisionStageCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stages: [QwenVisionStageDiagnostic] = []

    func append(_ stage: QwenVisionStageDiagnostic) {
        lock.lock()
        defer { lock.unlock() }
        stages.append(stage)
    }

    func receipts() -> [[String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        return stages.map { stage in
            [
                "stage": stage.stage, "count": stage.count, "finiteCount": stage.finiteCount,
                "nanCount": stage.nanCount, "positiveInfinityCount": stage.positiveInfinityCount,
                "negativeInfinityCount": stage.negativeInfinityCount,
                "firstNonfiniteIndex": stage.firstNonfiniteIndex ?? -1,
                "firstNonfiniteIndices": stage.firstNonfiniteIndices,
                "sampleBits": stage.sampleBits.map { UInt64($0) },
                "finiteMinimum": stage.finiteMinimum.map { Double($0) as Any } ?? NSNull(),
                "finiteMaximum": stage.finiteMaximum.map { Double($0) as Any } ?? NSNull(),
            ]
        }
    }
}

struct P23SourceVisionFeatureComparison {
    let failureCount: Int
    let maximumAbsoluteError: Double
    let firstFailureIndex: Int?
    let maximumFailureIndex: Int?

    static func compare(_ actual: [Float], _ reference: [Float]) throws -> Self {
        guard !actual.isEmpty, actual.count == reference.count,
              actual.allSatisfy(\.isFinite), reference.allSatisfy(\.isFinite) else {
            throw P23SourceVisionError.invalid("non-finite or incomplete source image features")
        }
        var failures = 0
        var maximum = 0.0
        var first: Int?
        var maximumFailure: Int?
        var maximumFailedError = 0.0
        for index in actual.indices {
            let error = abs(Double(actual[index]) - Double(reference[index]))
            maximum = max(maximum, error)
            if error > 1e-5 + 1e-5 * abs(Double(reference[index])) {
                failures += 1
                if first == nil { first = index }
                if maximumFailure == nil || error > maximumFailedError {
                    maximumFailedError = error
                    maximumFailure = index
                }
            }
        }
        return Self(failureCount: failures, maximumAbsoluteError: maximum,
                    firstFailureIndex: first, maximumFailureIndex: maximumFailure)
    }
}

private struct P23SourceVisionTensor: Decodable {
    let file: String
    let shape: [Int]
    let byteCount: Int
    let sha256: String
}

private struct P23SourceVisionCase: Decodable {
    let id: String
    let imagePath: String
    let imageBytes: Int
    let imageSHA256: String
    let promptText: String
    let encodedSize: [Int]
    let orientation: Int
    let orientedSize: [Int]
    let processedSize: [Int]
    let gridTHW: [Int]
    let patches: P23SourceVisionTensor
    let features: P23SourceVisionTensor
    let tokenIds: [Int32]
    let modalityIds: [Int]
    let padRows: [Int]
    let positions: [[Int32]]
    let featurePositions: [[Int32]]
    let textRoPEDelta: Int
}

private struct P23SourceVisionEnvironment: Decodable {
    let python: String
    let torch: String
    let numpy: String
    let pillow: String
    let transformers: String
    let transformersCommit: String
    let transformersTree: String
    let device: String
    let dtype: String
    let attention: String
    let threads: Int
    let deterministic: Bool
}

private struct P23SourceVisionReference: Decodable {
    let kind: String
    let complete: Bool
    let acceptanceScope: String
    let sourceDescriptorContentSHA256: String
    let registrationPath: String
    let visionCompanionPath: String
    let visionManifestSHA256: String
    let processorConfigSHA256: String
    let requestSHA256: String
    let referenceAppliesPassFail: Bool
    let featureAbsoluteTolerance: Double
    let featureRelativeTolerance: Double
    let environment: P23SourceVisionEnvironment
    let visionSourceBytes: Int
    let maximumReadBytes: Int
    let sourceTensorSHA256: [String: String]
    let results: [P23SourceVisionCase]

    func validate(registration: URL, companion: URL, requestSHA: String) throws {
        guard kind == "qwen36-original-bf16-vision-reference-v1", complete,
              acceptanceScope == "patches-structure-features-only", !referenceAppliesPassFail,
              sourceDescriptorContentSHA256 == "c1ac463726b716e7db3a9fbe5db4cb690f95be7e5779aea02e49c2020bca7a7a",
              processorConfigSHA256 == "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
              registrationPath == registration.path, visionCompanionPath == companion.path,
              requestSHA256 == requestSHA, requestSHA.count == 64,
              featureAbsoluteTolerance == 1e-5, featureRelativeTolerance == 1e-5,
              environment.python == "3.12.3", environment.torch == "2.10.0",
              environment.numpy == "2.4.3", environment.pillow == "10.3.0",
              environment.transformers == "5.18.0.dev0",
              environment.transformersCommit == "bd15bc95a89e728bbc1224084eb3b5829428c353",
              environment.transformersTree == "80eb369e589827bc7ed45b3a1f0ead5457097535",
              environment.device == "cpu", environment.dtype == "float32",
              environment.attention == "eager", environment.threads == 1, environment.deterministic,
              visionSourceBytes == 893_142_496, maximumReadBytes == 8 * 1024 * 1024,
              sourceTensorSHA256.count == 333, (1...3).contains(results.count),
              Set(results.map(\.id)).count == results.count else {
            throw P23SourceVisionError.invalid("independent source vision reference contract changed")
        }
        for item in results {
            guard !item.id.isEmpty, item.id.utf8.count <= 80,
                  item.id.unicodeScalars.allSatisfy({ CharacterSet.alphanumerics.contains($0) || $0 == "-" }),
                  item.patches.file == item.id + ".patches-bf16-le.bin",
                  item.features.file == item.id + ".features-fp32-le.bin",
                  (1...20 * 1024 * 1024).contains(item.imageBytes),
                  !item.promptText.isEmpty, item.promptText.utf8.count <= 4096,
                  item.gridTHW.count == 3, item.gridTHW[0] == 1,
                  item.tokenIds.count <= 4096, item.positions.count == item.tokenIds.count,
                  item.featurePositions.count <= 630 else {
                throw P23SourceVisionError.invalid("invalid or excessive reference image metadata")
            }
        }
    }
}

private enum P23SourceVisionError: Error {
    case invalid(String)
}

private func p23Canonical(_ url: URL) throws -> URL {
    let canonical = url.standardizedFileURL
    guard canonical.path == canonical.resolvingSymlinksInPath().path,
          FileManager.default.fileExists(atPath: canonical.path) else {
        throw P23SourceVisionError.invalid("noncanonical or missing evidence path: \(url.path)")
    }
    return canonical
}

private func p23Read(_ url: URL, maximum: Int) throws -> Data {
    let path = try p23Canonical(url)
    let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
    guard attributes[.type] as? FileAttributeType == .typeRegular,
          let size = attributes[.size] as? NSNumber, size.intValue > 0, size.intValue <= maximum else {
        throw P23SourceVisionError.invalid("nonregular or oversized evidence: \(url.path)")
    }
    let bytes = try Data(contentsOf: path)
    guard bytes.count == size.intValue else {
        throw P23SourceVisionError.invalid("evidence changed during read: \(url.path)")
    }
    return bytes
}

private func p23SHA(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func p23WriteReceipt(_ object: [String: Any], to url: URL) throws {
    try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        .write(to: url, options: .withoutOverwriting)
}

private func p23Seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
}

private extension QwenVisionRuntime {
    func processReferenceImage(
        fileURL: URL, device: MTLDevice, expectedPatchSHA: String
    ) async throws -> QwenVisionFeatures {
        let preprocessor = QwenImagePreprocessor(device: device)
        let plan = try preprocessor.plan(fileURL: fileURL)
        let pixels = try preprocessor.preprocess(plan)
        let patchByteCount = pixels.geometry.patchRows * 1536 * 2
        let patchBytes = Data(bytes: pixels.patchesBF16.contents(), count: patchByteCount)
        guard p23SHA(patchBytes) == expectedPatchSHA else {
            throw P23SourceVisionError.invalid("fresh patch buffer differs from the early gate")
        }
        return try await process(pixels)
    }
}
