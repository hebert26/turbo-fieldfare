import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare

@Suite struct QwenVisionRuntimeTests {
    private enum VisionFixtureError: Error {
        case missingResource
        case invalidJSON
        case invalidTensor(String)
        case missingPath(String)
    }

    @Test func frozenP3FixtureMatchesRegisteredKernelsAtFP32Tolerance() async throws {
        let fixture = try loadVisionFixture()
        let context = try MetalContext()
        let config = QwenVisionConfig(
            depth: 1,
            hiddenSize: 16,
            intermediateSize: 32,
            numHeads: 2,
            numPositionEmbeddings: 64,
            inputChannels: 3,
            outputHiddenSize: 2_048,
            patchSize: 2,
            temporalPatchSize: 1,
            spatialMergeSize: 2,
            minimumProcessedPixels: 96,
            publisherMaximumProcessedPixels: 96,
            maximumPatchRows: 24,
            maximumMergedRows: 6,
            allowsFixtureGeometry: true)
        let runtime = try QwenVisionRuntime(context: context, fixtureConfig: config)
        let result = try await runtime.executeFixture(
            patches: fixture.patches,
            positions: fixture.positions,
            gridHeight: 4,
            gridWidth: 6,
            weights: fixture.weights)
        assertElementwiseClose(result.output, fixture.expectedOutput,
                                label: "P3 merger output")
        #expect(result.orderedLayerIntermediates.count == 1)
        let intermediate = try #require(result.orderedLayerIntermediates.first)
        assertElementwiseClose(intermediate, fixture.expectedBlockOutput,
                               label: "P3 block intermediate")
        #expect(result.allocationPlan.totalBytes <= 200_000_000)
    }

    @Test func depth27OrderSensitiveFixtureMatchesIndependentPerLayerRecurrence() async throws {
        let fixture = try loadVisionFixture()
        let context = try MetalContext()
        let config = tinyConfig(depth: 27)
        let weights = orderFixtureWeights(depth: config.depth)
        let runtime = try QwenVisionRuntime(context: context, fixtureConfig: config)
        let result = try await runtime.executeFixture(
            patches: fixture.patches, positions: fixture.positions,
            gridHeight: 4, gridWidth: 6, weights: weights)
        #expect(result.orderedLayerIntermediates.count == 27)
        var expectedValue: Float = 0.25
        for layer in 0..<27 {
            expectedValue += Float(layer + 1) * 0.01
            let expected = [Float](
                repeating: expectedValue, count: fixture.positions.count * config.hiddenSize)
            guard result.orderedLayerIntermediates.indices.contains(layer) else {
                Issue.record("missing ordered block \(layer) intermediate")
                return
            }
            let actual = result.orderedLayerIntermediates[layer]
            assertElementwiseClose(actual, expected,
                                   label: "ordered block \(layer)")
        }
    }

    @Test func swappingDepth27LayerBiasesChangesOrderedIntermediate() async throws {
        let fixture = try loadVisionFixture()
        let context = try MetalContext()
        let config = tinyConfig(depth: 27)
        let weights = orderFixtureWeights(depth: config.depth)
        var swapped = weights
        let first = try #require(swapped["blocks.0.attn.proj.bias"])
        let second = try #require(swapped["blocks.1.attn.proj.bias"])
        swapped["blocks.0.attn.proj.bias"] = second
        swapped["blocks.1.attn.proj.bias"] = first
        let runtime = try QwenVisionRuntime(context: context, fixtureConfig: config)
        let result = try await runtime.executeFixture(
            patches: fixture.patches, positions: fixture.positions,
            gridHeight: 4, gridWidth: 6, weights: swapped)
        let originalFirst = [Float](
            repeating: 0.25 + 0.01, count: fixture.positions.count * config.hiddenSize)
        let swappedFirst = try #require(result.orderedLayerIntermediates.first)
        #expect(maxDifference(swappedFirst, originalFirst) > 1e-5)
    }

    @Test func cancelledFixtureAfterSubmissionDiscardsCommandAndReleasesScratch() async throws {
        let fixture = try loadVisionFixture()
        let context = try MetalContext()
        let config = tinyConfig(depth: 1)
        let events = Mutex<[QwenVisionExecutionEvent]>([])
        let gate = FixtureSubmissionGate(target: "fixture patch")
        let runtime = try QwenVisionRuntime(
            context: context,
            fixtureConfig: config,
            executionHooks: QwenVisionExecutionHooks(
                afterCommandSubmission: { stage in
                    await gate.hold(stage: stage)
                },
                observe: { event in
                    events.withLock { $0.append(event) }
                }))
        let operation = Task { () -> FixtureOperationOutcome in
            do {
                _ = try await runtime.executeFixture(
                    patches: fixture.patches, positions: fixture.positions,
                    gridHeight: 4, gridWidth: 6, weights: fixture.weights)
                return .completed
            } catch is CancellationError {
                return .cancelled
            } catch {
                return .failed
            }
        }
        await gate.waitUntilSubmitted()
        operation.cancel()
        await gate.release()
        let outcome = await operation.value
        #expect(outcome == .cancelled)
        let recorded = events.withLock { $0 }
        let scratchBytes = recorded.compactMap { event -> Int? in
            guard case let .scratchAllocated(bytes) = event else { return nil }
            return bytes
        }
        #expect(scratchBytes.count == 1)
        #expect((scratchBytes.first ?? 0) > 0)
        #expect(recorded.contains(.commandSubmitted("fixture patch")))
        #expect(recorded.contains(
            .commandCompleted("fixture patch", discarded: true)))
        #expect(recorded.contains(.scratchReleased))
        let submittedStages = recorded.compactMap { event -> String? in
            guard case let .commandSubmitted(stage) = event else { return nil }
            return stage
        }
        #expect(submittedStages == ["fixture patch"])
    }

    @Test func productionRuntimeMapsAndReleasesOneGroupAtATime() async throws {
        let companion = try QwenVisionWeightStoreTests().makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: companion) }
        let context = try MetalContext()
        let store = try QwenVisionWeightStore.open(
            directoryURL: companion,
            compatibleTextManifestSHA256: String(repeating: "a", count: 64))
        let config = QwenVisionConfig(
            minimumProcessedPixels: 1_024,
            publisherMaximumProcessedPixels: 1_024,
            maximumPatchRows: 4,
            maximumMergedRows: 1)
        let events = Mutex<[QwenVisionExecutionEvent]>([])
        let runtime = try QwenVisionRuntime(
            context: context, store: store, config: config,
            executionHooks: QwenVisionExecutionHooks(
                afterCommandSubmission: { _ in },
                observe: { event in events.withLock { $0.append(event) } }))
        let pixels = try makeTinyProductionPixels(config: config, device: context.device)
        let features = try await runtime.process(pixels)
        let recorded = events.withLock { $0 }
        let expectedGroups = [QwenVisionWeightGroup.patchAndPosition]
            + (0..<27).map(QwenVisionWeightGroup.block)
            + [.merger]
        let mapped = recorded.compactMap { event -> QwenVisionWeightGroup? in
            guard case let .groupMapped(group, _) = event else { return nil }
            return group
        }
        let released = recorded.compactMap { event -> QwenVisionWeightGroup? in
            guard case let .groupReleased(group) = event else { return nil }
            return group
        }
        #expect(mapped == expectedGroups)
        #expect(released == expectedGroups)
        var liveGroups = 0
        var maximumLiveGroups = 0
        for event in recorded {
            switch event {
            case .groupMapped:
                #expect(liveGroups == 0)
                liveGroups += 1
                maximumLiveGroups = max(maximumLiveGroups, liveGroups)
            case .groupReleased:
                #expect(liveGroups == 1)
                liveGroups -= 1
            default:
                break
            }
        }
        #expect(liveGroups == 0)
        #expect(maximumLiveGroups == 1)
        let submitted = recorded.compactMap { event -> String? in
            guard case let .commandSubmitted(stage) = event else { return nil }
            return stage
        }
        let completed = recorded.compactMap { event -> Bool? in
            guard case let .commandCompleted(_, discarded) = event else { return nil }
            return discarded
        }
        #expect(submitted.count == 29)
        #expect(completed == [Bool](repeating: false, count: 29))
        #expect(recorded.contains { event in
            if case .scratchAllocated = event { return true }
            return false
        })
        #expect(recorded.contains(.scratchReleased))
        #expect(features.tokenCount == 1)
        #expect(features.hiddenSize == 2_048)
    }

    @Test func storeBackedRuntimeRejectsMismatchedArchitectureBeforeMapping() throws {
        let companion = try QwenVisionWeightStoreTests().makeSyntheticCompanion()
        defer { try? FileManager.default.removeItem(at: companion) }
        let context = try MetalContext()
        let store = try QwenVisionWeightStore.open(
            directoryURL: companion,
            compatibleTextManifestSHA256: String(repeating: "a", count: 64))
        let mismatched = QwenVisionConfig(
            depth: 1, hiddenSize: 16, intermediateSize: 32, numHeads: 2,
            numPositionEmbeddings: 64, inputChannels: 3, outputHiddenSize: 2_048,
            patchSize: 2, temporalPatchSize: 2, spatialMergeSize: 2,
            minimumProcessedPixels: 16, publisherMaximumProcessedPixels: 16,
            maximumPatchRows: 4, maximumMergedRows: 1,
            allowsFixtureGeometry: true)
        do {
            _ = try QwenVisionRuntime(
                context: context, store: store, config: mismatched)
            Issue.record("mismatched store/runtime architecture unexpectedly passed")
        } catch let error as QwenVisionError {
            #expect(error == .invalidConfiguration)
        } catch {
            Issue.record("unexpected mismatch error: \(error)")
        }
    }

    @Test func frozenP3WeightMutationCannotMatchReferenceOutput() async throws {
        let fixture = try loadVisionFixture()
        var mutated = fixture.weights
        guard var bias = mutated["merger.linear_fc2.bias"] else {
            Issue.record("missing merger output bias")
            return
        }
        bias[0] += 0.25
        mutated["merger.linear_fc2.bias"] = bias
        let context = try MetalContext()
        let config = QwenVisionConfig(
            depth: 1, hiddenSize: 16, intermediateSize: 32, numHeads: 2,
            numPositionEmbeddings: 64, inputChannels: 3, outputHiddenSize: 2_048,
            patchSize: 2, temporalPatchSize: 1, spatialMergeSize: 2,
            minimumProcessedPixels: 96, publisherMaximumProcessedPixels: 96,
            maximumPatchRows: 24, maximumMergedRows: 6,
            allowsFixtureGeometry: true)
        let runtime = try QwenVisionRuntime(context: context, fixtureConfig: config)
        let result = try await runtime.executeFixture(
            patches: fixture.patches, positions: fixture.positions,
            gridHeight: 4, gridWidth: 6, weights: mutated)
        #expect(maxDifference(result.output, fixture.expectedOutput) > 1e-5)
    }

    @Test func scratchPlanUsesPaddedTemporalTwoAndMergerWidths() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let config = QwenVisionConfig.official
        let scratch = try QwenVisionScratch(
            device: device, rows: config.maximumPatchRows, config: config)
        let entries = Dictionary(
            uniqueKeysWithValues: scratch.allocationPlan.entries.map { ($0.name, $0.bytes) })
        let paddedRows = 2_560
        let mergedRows = 630
        let bf16 = MemoryLayout<UInt16>.stride
        let float32 = MemoryLayout<Float>.stride
        #expect(scratch.paddedRows == paddedRows)
        #expect(scratch.mergedRows == mergedRows)
        let fp32Bytes = paddedRows * config.hiddenSize * float32
        #expect(entries["hidden A"] == fp32Bytes)
        #expect(entries["hidden B"] == fp32Bytes)
        #expect(entries["QKV workspace"]
            == paddedRows * 3 * config.hiddenSize * float32)
        #expect(entries["attention workspace"] == fp32Bytes)
        #expect(entries["MLP workspace"]
            == paddedRows * config.intermediateSize * float32)
        #expect(entries["retained Float32 output staging"]
            == mergedRows * config.outputHiddenSize * float32)
        #expect(scratch.normalizedPatches.length
            >= paddedRows * config.patchWidth * bf16)
        #expect(scratch.normalizedPatches.length
            > paddedRows * 768 * bf16)
        #expect(scratch.mergerPacked.length
            >= mergedRows * config.mergedHiddenSize * bf16)
        #expect(scratch.outputFloat32.length
            == entries["retained Float32 output staging"])
        #expect(scratch.outputBF16 === scratch.outputFloat32)
        // These aliases are intentional physical reuse points; each prior
        // command completes before the next logical stage uses the storage.
        #expect(scratch.q === scratch.k)
        #expect(scratch.k === scratch.v)
        #expect(scratch.mergerPacked === scratch.q)
        #expect(scratch.mergerNormalized === scratch.attention)
        #expect(scratch.mergerHidden === scratch.gate)
        #expect(scratch.allocationPlan.totalBytes <= 200_000_000)
        // Every physical activation is FP32; aliases only reuse storage after
        // command completion and do not round the P3 oracle path.
        #expect(entries.values.allSatisfy { $0.isMultiple(of: float32) })
    }

    @Test func scratchRetainedLineageBytesAreIncludedInCheckedPlan() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let retained = 5_256_720
        let scratch = try QwenVisionScratch(
            device: device,
            rows: 2_520,
            config: .official,
            retainedRequestedBytes: retained)
        let entry = try #require(
            scratch.allocationPlan.entries.first { $0.name == "retained lineage" })
        #expect(entry.bytes == retained)
        #expect(scratch.allocationPlan.totalBytes
            >= scratch.outputFloat32.length + retained)
    }

    @Test func scratchRejectsRowsPastPatchLimitBeforeAllocation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: QwenVisionError.patchRowQuotaExceeded(
            requested: 2_521, maximum: 2_520)) {
            try QwenVisionScratch(device: device, rows: 2_521, config: .official)
        }
        #expect(throws: QwenVisionError.invalidFeatureShape) {
            try QwenVisionScratch(device: device, rows: 2, config: .official)
        }
    }

    @Test func scratchRejectsCheckedMutableByteBudgetBeforeBufferCreation() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let limits = QwenVisionResourceLimits(
            maximumVisibleHistoryRows: 630,
            maximumLiveOwnerRows: 1_260,
            maximumMutablePreparedBytes: 1)
        do {
            _ = try QwenVisionScratch(
                device: device, rows: 2_520, config: .official, limits: limits)
            Issue.record("scratch unexpectedly passed a one-byte budget")
        } catch let error as QwenVisionError {
            guard case let .mutableByteQuotaExceeded(requested, maximum) = error else {
                Issue.record("unexpected Qwen scratch error: \(error)")
                return
            }
            #expect(requested > maximum)
            #expect(maximum == 1)
        }
    }

    private func makeTinyProductionPixels(
        config: QwenVisionConfig, device: MTLDevice
    ) throws -> QwenVisionPixelBuffer {
        let geometry = try QwenImageGeometry(
            sourceWidth: 32, sourceHeight: 32, config: config)
        let patchRowCount = geometry.patchRows
        let patchValues = [UInt16](repeating: 0, count: patchRowCount * config.patchWidth)
        let positions = [Int32](repeating: 0, count: patchRowCount * 2)
        let patchBytes = patchValues.count * MemoryLayout<UInt16>.stride
        let positionBytes = positions.count * MemoryLayout<Int32>.stride
        guard let patchBuffer = device.makeBuffer(length: patchBytes, options: .storageModeShared),
              let positionBuffer = device.makeBuffer(
                  length: positionBytes, options: .storageModeShared) else {
            throw QwenVisionError.allocationFailed(name: "tiny production pixels")
        }
        patchValues.withUnsafeBytes { source in
            if let address = source.baseAddress {
                memcpy(patchBuffer.contents(), address, source.count)
            }
        }
        positions.withUnsafeBytes { source in
            if let address = source.baseAddress {
                memcpy(positionBuffer.contents(), address, source.count)
            }
        }
        let metadata = VisionImageMetadata(
            encodedBytes: 1, encodedWidth: 32, encodedHeight: 32,
            orientedWidth: 32, orientedHeight: 32, orientation: 1,
            bitsPerComponent: 8, colorModel: "RGB", typeIdentifier: "fixture")
        return QwenVisionPixelBuffer(
            patchesBF16: patchBuffer, positionsInt32x2: positionBuffer,
            metadata: metadata, geometry: geometry, imageDigest: String(repeating: "0", count: 64),
            wallNanoseconds: 0, allocatedBytes: patchBytes + positionBytes)
    }

    private enum FixtureOperationOutcome: Sendable, Equatable {
        case completed
        case cancelled
        case failed
    }

    private actor FixtureSubmissionGate {
        private let target: String
        private var submitted = false
        private var released = false
        private var submittedWaiters: [CheckedContinuation<Void, Never>] = []
        private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

        init(target: String) {
            self.target = target
        }

        func hold(stage: String) async {
            guard stage == target else { return }
            submitted = true
            let waiters = submittedWaiters
            submittedWaiters.removeAll(keepingCapacity: false)
            for waiter in waiters { waiter.resume() }
            guard !released else { return }
            await withCheckedContinuation { continuation in
                if released { continuation.resume() }
                else { releaseWaiters.append(continuation) }
            }
        }

        func waitUntilSubmitted() async {
            if submitted { return }
            await withCheckedContinuation { continuation in
                if submitted { continuation.resume() }
                else { submittedWaiters.append(continuation) }
            }
        }

        func release() {
            released = true
            let waiters = releaseWaiters
            releaseWaiters.removeAll(keepingCapacity: false)
            for waiter in waiters { waiter.resume() }
        }
    }

    private struct VisionFixture {
        let patches: [Float]
        let positions: [SIMD2<Int32>]
        let weights: [String: [Float]]
        let expectedBlockOutput: [Float]
        let expectedOutput: [Float]
    }

    private func tinyConfig(depth: Int) -> QwenVisionConfig {
        QwenVisionConfig(
            depth: depth, hiddenSize: 16, intermediateSize: 32, numHeads: 2,
            numPositionEmbeddings: 64, inputChannels: 3, outputHiddenSize: 2_048,
            patchSize: 2, temporalPatchSize: 1, spatialMergeSize: 2,
            minimumProcessedPixels: 96, publisherMaximumProcessedPixels: 96,
            maximumPatchRows: 24, maximumMergedRows: 6,
            allowsFixtureGeometry: true)
    }

    private func orderFixtureWeights(depth: Int) -> [String: [Float]] {
        var weights: [String: [Float]] = [
            "patch_embed.proj.weight": [Float](repeating: 0, count: 16 * 12),
            "patch_embed.proj.bias": [Float](repeating: 0.25, count: 16),
            "pos_embed.weight": [Float](repeating: 0, count: 64 * 16),
            "merger.norm.weight": [Float](repeating: 1, count: 16),
            "merger.norm.bias": [Float](repeating: 0, count: 16),
            "merger.linear_fc1.weight": [Float](repeating: 0, count: 64 * 64),
            "merger.linear_fc1.bias": [Float](repeating: 0, count: 64),
            "merger.linear_fc2.weight": [Float](repeating: 0, count: 2_048 * 64),
            "merger.linear_fc2.bias": [Float](repeating: 0, count: 2_048),
        ]
        for layer in 0..<depth {
            let prefix = "blocks.\(layer)."
            weights[prefix + "attn.proj.weight"] = [Float](repeating: 0, count: 16 * 16)
            weights[prefix + "attn.proj.bias"] = [Float](
                repeating: Float(layer + 1) * 0.01, count: 16)
            weights[prefix + "attn.qkv.weight"] = [Float](repeating: 0, count: 48 * 16)
            weights[prefix + "attn.qkv.bias"] = [Float](repeating: 0, count: 48)
            weights[prefix + "norm1.weight"] = [Float](repeating: 1, count: 16)
            weights[prefix + "norm1.bias"] = [Float](repeating: 0, count: 16)
            weights[prefix + "mlp.linear_fc1.weight"] = [Float](repeating: 0, count: 32 * 16)
            weights[prefix + "mlp.linear_fc1.bias"] = [Float](repeating: 0, count: 32)
            weights[prefix + "norm2.weight"] = [Float](repeating: 1, count: 16)
            weights[prefix + "norm2.bias"] = [Float](repeating: 0, count: 16)
            weights[prefix + "mlp.linear_fc2.weight"] = [Float](repeating: 0, count: 16 * 32)
            weights[prefix + "mlp.linear_fc2.bias"] = [Float](repeating: 0, count: 16)
        }
        return weights
    }

    private func loadVisionFixture() throws -> VisionFixture {
        guard let url = Bundle.module.url(
            forResource: "qwen36-tiny-fixtures", withExtension: "json") else {
            throw VisionFixtureError.missingResource
        }
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let vision = root["vision"] as? [String: Any],
              let weights = vision["weights"] as? [String: Any] else {
            throw VisionFixtureError.invalidJSON
        }
        let patches = try tensorValues(vision, key: "rawPatchInput")
        let positionNumbers = try tensorNumbers(vision, key: "visionPositionIDs")
        guard positionNumbers.count.isMultiple(of: 2) else {
            throw VisionFixtureError.invalidTensor("visionPositionIDs")
        }
        let positions = stride(from: 0, to: positionNumbers.count, by: 2).map {
            SIMD2<Int32>(positionNumbers[$0].int32Value, positionNumbers[$0 + 1].int32Value)
        }
        var decodedWeights: [String: [Float]] = [:]
        for (name, value) in weights {
            decodedWeights[name] = try tensorValues(value, label: name)
        }
        return VisionFixture(
            patches: patches,
            positions: positions,
            weights: decodedWeights,
            expectedBlockOutput: try tensorValues(
                vision, path: ["block", "output"]),
            expectedOutput: try tensorValues(
                vision, path: ["merger", "output"]))
    }

    private func tensorValues(_ object: Any, key: String) throws -> [Float] {
        try tensorValues(object, path: [key])
    }

    private func tensorValues(_ object: Any, path: [String]) throws -> [Float] {
        var current = object
        for component in path {
            guard let dictionary = current as? [String: Any],
                  let next = dictionary[component] else {
                throw VisionFixtureError.missingPath(path.joined(separator: "."))
            }
            current = next
        }
        return try tensorValues(current, label: path.joined(separator: "."))
    }

    private func tensorValues(_ object: Any, label: String) throws -> [Float] {
        guard let dictionary = object as? [String: Any],
              let values = dictionary["values"] as? [NSNumber] else {
            throw VisionFixtureError.invalidTensor(label)
        }
        return values.map(\.floatValue)
    }

    private func tensorNumbers(_ object: Any, key: String) throws -> [NSNumber] {
        guard let dictionary = object as? [String: Any],
              let tensor = dictionary[key] as? [String: Any],
              let values = tensor["values"] as? [NSNumber] else {
            throw VisionFixtureError.invalidTensor(key)
        }
        return values
    }

    private func maxDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        return zip(lhs, rhs).reduce(Float.zero) { max($0, abs($1.0 - $1.1)) }
    }

    private struct ErrorSummary {
        let maxAbsolute: Float
        let maxRelative: Float
        let failures: Int
    }

    private func assertElementwiseClose(
        _ actual: [Float], _ expected: [Float], label: String
    ) {
        #expect(actual.count == expected.count, "\(label) count")
        guard actual.count == expected.count else { return }
        let summary = summarize(actual, expected)
        print("P16_SEAM \(label) maxAbs=\(summary.maxAbsolute) "
            + "maxRel=\(summary.maxRelative) failures=\(summary.failures)")
        #expect(summary.failures == 0,
                "\(label) elementwise abs+rel exceeded: maxAbs=\(summary.maxAbsolute), maxRel=\(summary.maxRelative)")
    }

    private func summarize(_ actual: [Float], _ expected: [Float]) -> ErrorSummary {
        var maxAbsolute: Float = 0
        var maxRelative: Float = 0
        var failures = 0
        for (lhs, rhs) in zip(actual, expected) {
            let absolute = abs(lhs - rhs)
            let relative = absolute / max(abs(rhs), 1e-12)
            maxAbsolute = max(maxAbsolute, absolute)
            maxRelative = max(maxRelative, relative)
            if absolute > 1e-5 + 1e-5 * abs(rhs) { failures += 1 }
        }
        return ErrorSummary(
            maxAbsolute: maxAbsolute, maxRelative: maxRelative, failures: failures)
    }
}
