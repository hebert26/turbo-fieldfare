import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite(.serialized) struct QwenMoETests {
    @Test func configurationsKeepOfficialAndTinyIdentitySeparate() throws {
        let official = try QwenMoEConfiguration.official()
        #expect(official.hiddenSize == 2_048)
        #expect(official.expertCount == 256)
        #expect(official.topK == 8)
        #expect(official.routedIntermediateSize == 512)
        #expect(official.sharedIntermediateSize == 512)

        let tiny = try QwenMoEConfiguration(
            hiddenSize: 32, expertCount: 10, routedIntermediateSize: 16,
            sharedIntermediateSize: 16)
        #expect(tiny != official)
        #expect(throws: QwenMoEError.invalidConfiguration(field: "topK", value: 4)) {
            try QwenMoEConfiguration(
                hiddenSize: 32, expertCount: 10, topK: 4,
                routedIntermediateSize: 16, sharedIntermediateSize: 16)
        }
        #expect(throws: QwenMoEError.invalidConfiguration(field: "topK", value: 8)) {
            try QwenMoEConfiguration(
                hiddenSize: 32, expertCount: 7,
                routedIntermediateSize: 16, sharedIntermediateSize: 16)
        }
    }

    @Test func routeReconstructsPinnedFixtureWithIndependentFP32Oracle() throws {
        let fixture = try qwenFixtureSection()
        let logits = try tensorFloats(fixture, key: "routerLogits")
        let expectedIDs = try tensorInts(fixture, key: "top8Indices")
        let expectedWeights = try tensorFloats(fixture, key: "top8NormalizedWeights")
        let configuration = try QwenMoEConfiguration(
            hiddenSize: 32, expertCount: 10, routedIntermediateSize: 16,
            sharedIntermediateSize: 16)

        // This reference is reconstructed from the pinned generator's operation
        // order, rather than from the implementation under test.
        let reference = independentRoute(logits: logits, expertCount: 10, topK: 8)
        let actual = try QwenMoE.route(logits: logits, configuration: configuration)
        #expect(actual.selectedExpertIDs.flatMap { $0 } == expectedIDs)
        #expect(actual.selectedExpertIDs.flatMap { $0 } == reference.ids.flatMap { $0 })
        expectClose(actual.normalizedWeights.flatMap { $0 }, expectedWeights, tolerance: 1e-5)
        expectClose(actual.normalizedWeights.flatMap { $0 }, reference.weights.flatMap { $0 }, tolerance: 1e-5)
        expectClose(actual.probabilities.flatMap { $0 },
                    reference.probabilities.flatMap { $0 }, tolerance: 1e-5)

        let tieIDs = actual.selectedExpertIDs[0]
        #expect(tieIDs == Array(0..<8))
        #expect(actual.normalizedWeights[0].allSatisfy { abs($0 - 0.125) < 1e-6 })
    }

    @Test func evaluateReconstructsPinnedQwen36MoEWithIndependentGeneratorOracle() throws {
        let fixture = try qwenFixtureSection()
        let inputShape = try tensorShape(fixture, key: "input")
        let logitsShape = try tensorShape(fixture, key: "routerLogits")
        let idsShape = try tensorShape(fixture, key: "top8Indices")
        let weightsShape = try tensorShape(fixture, key: "top8NormalizedWeights")
        let gateShape = try tensorShape(fixture, key: "sharedExpertGate")
        let outputShape = try tensorShape(fixture, key: "output")
        #expect(inputShape == [1, 3, 32])
        #expect(logitsShape == [3, 10])
        #expect(idsShape == [3, 8])
        #expect(weightsShape == [3, 8])
        #expect(gateShape == [3, 1])
        #expect(outputShape == [1, 3, 32])

        let configuration = try QwenMoEConfiguration(
            hiddenSize: 32, expertCount: 10, routedIntermediateSize: 7,
            sharedIntermediateSize: 6)
        let expectedInput = try tensorFloats(fixture, key: "input")
        let expectedRouterLogits = try tensorFloats(fixture, key: "routerLogits")
        let expectedIDs = try tensorInts(fixture, key: "top8Indices")
        let expectedWeights = try tensorFloats(fixture, key: "top8NormalizedWeights")
        let expectedGates = try tensorFloats(fixture, key: "sharedExpertGate")
        let expectedOutput = try tensorFloats(fixture, key: "output")

        // Reconstruct the fixture input independently from synthetic(), then apply
        // the pinned generator's explicit zeroing of the first token.
        var input = pinnedSynthetic(count: 3 * configuration.hiddenSize,
                                    phase: 1.7, scale: 0.2)
        input.replaceSubrange(
            0..<configuration.hiddenSize,
            with: Array(repeating: Float.zero, count: configuration.hiddenSize))
        expectClose(input, expectedInput, tolerance: 1e-5)

        // initialize_module enumerates these parameters in registration order.
        // Every parameter is at least rank two, so all use scale 0.18.
        let routerWeight = pinnedSynthetic(
            count: configuration.expertCount * configuration.hiddenSize,
            phase: 3.0, scale: 0.18)
        let routedGateUp = pinnedSynthetic(
            count: configuration.expertCount * 2 * configuration.routedIntermediateSize
                * configuration.hiddenSize,
            phase: 4.0, scale: 0.18)
        let routedDown = pinnedSynthetic(
            count: configuration.expertCount * configuration.hiddenSize
                * configuration.routedIntermediateSize,
            phase: 5.0, scale: 0.18)
        let sharedGate = pinnedSynthetic(
            count: configuration.sharedIntermediateSize * configuration.hiddenSize,
            phase: 6.0, scale: 0.18)
        let sharedUp = pinnedSynthetic(
            count: configuration.sharedIntermediateSize * configuration.hiddenSize,
            phase: 7.0, scale: 0.18)
        let sharedDown = pinnedSynthetic(
            count: configuration.hiddenSize * configuration.sharedIntermediateSize,
            phase: 8.0, scale: 0.18)
        let sharedOutputGate = pinnedSynthetic(
            count: configuration.hiddenSize, phase: 9.0, scale: 0.18)

        // Qwen3.5's F.linear computes input @ weight.T. Keep the same row-major
        // operation order while deriving logits instead of consuming frozen logits.
        var routerLogits: [Float] = []
        for token in 0..<(input.count / configuration.hiddenSize) {
            let hidden = Array(input[
                token * configuration.hiddenSize..<(token + 1) * configuration.hiddenSize])
            routerLogits.append(contentsOf: project(
                hidden, matrix: routerWeight, rows: configuration.expertCount,
                columns: configuration.hiddenSize))
        }
        expectClose(routerLogits, expectedRouterLogits, tolerance: 1e-5)

        let gateUpCount = 2 * configuration.routedIntermediateSize * configuration.hiddenSize
        let downCount = configuration.hiddenSize * configuration.routedIntermediateSize
        var routedExperts: [Int: QwenMoEDenseExpert] = [:]
        for expertID in 0..<configuration.expertCount {
            let gateUpStart = expertID * gateUpCount
            let downStart = expertID * downCount
            routedExperts[expertID] = QwenMoEDenseExpert(
                gateUp: Array(routedGateUp[gateUpStart..<(gateUpStart + gateUpCount)]),
                down: Array(routedDown[downStart..<(downStart + downCount)]))
        }
        let sharedExpert = QwenMoEDenseSharedExpert(
            gate: sharedGate, up: sharedUp, down: sharedDown,
            outputGate: sharedOutputGate)

        let actual = try QwenMoE.evaluate(
            input: input, routerLogits: routerLogits, configuration: configuration,
            routedExperts: routedExperts, sharedExpert: sharedExpert)
        #expect(actual.routing.selectedExpertIDs.flatMap { $0 } == expectedIDs)
        expectClose(actual.routing.normalizedWeights.flatMap { $0 }, expectedWeights,
                    tolerance: 1e-5)
        expectClose(actual.sharedGate, expectedGates, tolerance: 1e-5)
        expectClose(actual.output, expectedOutput, tolerance: 1e-5)
    }

    @Test func routeRejectsNonfiniteValuesAndMalformedCounts() throws {
        let configuration = try QwenMoEConfiguration(
            hiddenSize: 3, expertCount: 8, routedIntermediateSize: 2,
            sharedIntermediateSize: 2)
        #expect(throws: QwenMoEError.nonfiniteRouterValue(token: 0, expert: 3)) {
            try QwenMoE.route(
                logits: [0, 0, 0, .nan, 0, 0, 0, 0], configuration: configuration)
        }
        #expect(throws: QwenMoEError.invalidCount(field: "routerLogits", expected: 8, actual: 7)) {
            try QwenMoE.route(logits: Array(repeating: 0, count: 7), configuration: configuration)
        }
        #expect(throws: QwenMoEError.invalidCount(field: "input", expected: 3, actual: 2)) {
            try QwenMoE.evaluate(
                input: [0, 1], routerLogits: Array(repeating: 0, count: 8),
                configuration: configuration, routedExperts: [:],
                sharedExpert: denseShared(hidden: 3, intermediate: 2, seed: 0.1))
        }
    }

    @Test func evaluateUsesOriginalHiddenForBothBranchesAndRejectsMissingExperts() throws {
        let configuration = try QwenMoEConfiguration(
            hiddenSize: 3, expertCount: 10, routedIntermediateSize: 2,
            sharedIntermediateSize: 2)
        let input: [Float] = [0.25, -0.5, 0.75]
        let logits: [Float] = [2.0, 1.5, 1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]
        let experts = Dictionary(uniqueKeysWithValues: (0..<10).map {
            ($0, denseExpert(hidden: 3, intermediate: 2, seed: Float($0) * 0.17 + 0.2))
        })
        let shared = denseShared(hidden: 3, intermediate: 2, seed: -0.2)
        let result = try QwenMoE.evaluate(
            input: input, routerLogits: logits, configuration: configuration,
            routedExperts: experts, sharedExpert: shared)
        let expected = independentEvaluate(
            input: input, logits: logits, configuration: configuration,
            routedExperts: experts, sharedExpert: shared)
        expectClose(result.routedOutput, expected.routed, tolerance: 1e-5)
        expectClose(result.sharedOutput, expected.shared, tolerance: 1e-5)
        expectClose(result.output, expected.output, tolerance: 1e-5)
        expectClose(result.sharedGate, expected.gates, tolerance: 1e-5)

        // Required strong negative controls: omitting selected renormalization or
        // omitting the MoE sigmoid shared gate must be observably rejected.
        let wrongNormalization = independentEvaluate(
            input: input, logits: logits, configuration: configuration,
            routedExperts: experts, sharedExpert: shared, normalizeSelected: false)
        #expect(maxDifference(result.output, wrongNormalization.output) > 1e-5)
        let missingSharedGate = zip(expected.routed, expected.sharedRaw).map(+)
        #expect(maxDifference(result.output, missingSharedGate) > 1e-4)

        var missing = experts
        missing.removeValue(forKey: result.routing.selectedExpertIDs[0][0])
        #expect(throws: QwenMoEError.missingExpert(result.routing.selectedExpertIDs[0][0])) {
            try QwenMoE.evaluate(
                input: input, routerLogits: logits, configuration: configuration,
                routedExperts: missing, sharedExpert: shared)
        }
    }

    @Test func qwenV2WireIsStrictAndProducesPerExpertAffineOffsets() throws {
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: 1, expertCount: 10, hiddenSize: 32,
            intermediateSize: 16, expertStride: 16_384)
        let sizes = try qwenV2SourceData(configuration: configuration)
        let data = try qwenV2Data(configuration: configuration, sizes: sizes)
        let files = ["packed_experts/layer_00.bin": UInt64(configuration.expertCount) * configuration.expertStride]
        let layout = try PackedExpertsLayoutReader.decodeQwenV2(
            data: data, configuration: configuration, manifestFileSizes: files)
        #expect(layout.numLayers == 1)
        #expect(layout.expertsPerLayer == 10)
        #expect(layout.layers[0].file == "layer_00.bin")
        #expect(layout.layers[0].experts.count == 10)
        #expect(layout.expert(layer: 0, expert: 9).offset == 9 * 16_384)
        #expect(layout.layers[0].affineDescriptors["gate_up"]?.valuesSize == sizes.gateUp.values)
        #expect(layout.layers[0].affineDescriptors["down"]?.valuesSize == sizes.down.values)

        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["unexpected"] = true
        let unknownField = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: ModelError.self) {
            try PackedExpertsLayoutReader.decodeQwenV2(
                data: unknownField, configuration: configuration, manifestFileSizes: files)
        }
        root.removeValue(forKey: "unexpected")
        root["version"] = 3
        let unknownVersion = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: ModelError.self) {
            try PackedExpertsLayoutReader.decodeQwenV2(
                data: unknownVersion, configuration: configuration, manifestFileSizes: files)
        }

        var wrongPair = root
        wrongPair["version"] = 2
        wrongPair["layers"] = [[
            "layer": 0, "path": "packed_experts/layer_00.bin", "experts": 10,
            "stride": 16_384, "sources": [],
        ]]
        let malformedPair = try JSONSerialization.data(withJSONObject: wrongPair)
        #expect(throws: ModelError.self) {
            try PackedExpertsLayoutReader.decodeQwenV2(
                data: malformedPair, configuration: configuration, manifestFileSizes: files)
        }
    }

    @Test func officialVerifiedManifestUsesDistinctFortyBy256QwenV2Path() throws {
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: 40, expertCount: 256, hiddenSize: 2_048,
            intermediateSize: 512, expertStride: 2_097_152)
        let sizes = try qwenV2SourceData(configuration: configuration)
        let manifest = try officialQwenManifest()
        let layout = try PackedExpertsLayoutReader.decodeQwenV2(
            data: qwenV2Data(configuration: configuration, sizes: sizes), manifest: manifest)
        #expect(layout.numLayers == 40)
        #expect(layout.expertsPerLayer == 256)
        #expect(layout.layers.count == 40)
        #expect(layout.expert(layer: 39, expert: 255).offset == 255 * 2_097_152)

        var root = try #require(JSONSerialization.jsonObject(
            with: qwenV2Data(configuration: configuration, sizes: sizes)) as? [String: Any])
        root["unknown"] = true
        #expect(throws: ModelError.self) {
            try PackedExpertsLayoutReader.decodeQwenV2(
                data: try JSONSerialization.data(withJSONObject: root), manifest: manifest)
        }
        let tinyConfiguration = try QwenPackedLayoutConfiguration(
            layerCount: 1, expertCount: 8, hiddenSize: 4,
            intermediateSize: 2, expertStride: 16_384)
        let tinySizes = try qwenV2SourceData(configuration: tinyConfiguration)
        #expect(throws: ModelError.self) {
            try PackedExpertsLayoutReader.decodeQwenV2(
                data: qwenV2Data(configuration: tinyConfiguration, sizes: tinySizes),
                manifest: manifest)
        }
    }

    @Test func legacyV1LayoutKeepsGemmaOffsetsAndByteCalculations() throws {
        let root: [String: Any] = [
            "expertStride": 16_384, "numLayers": 1, "expertsPerLayer": 2,
            "layers": [[
                "layer": 0, "file": "layer_00.bin",
                "experts": [[
                    "expert": 0, "offset": 0, "size": 16_384,
                    "tensors": [
                        "gate": ["offset": 0, "size": 4_096, "dtype": "U32", "shape": [64, 64], "bits": 4],
                        "gate_scales": ["offset": 4_096, "size": 256, "dtype": "BF16", "shape": [64, 1]],
                    ],
                ], [
                    "expert": 1, "physicalRank": 1, "offset": 16_384, "size": 16_384,
                    "tensors": [:],
                ]],
            ]],
        ]
        let layout = try PackedExpertsLayoutReader.decode(
            data: JSONSerialization.data(withJSONObject: root), manifest: nil)
        #expect(layout.expertStride == 16_384)
        #expect(layout.expert(layer: 0, expert: 0).subTensors["gate"]?.size == 4_096)
        #expect(layout.expert(layer: 0, expert: 0).subTensors["gate_scales"]?.offset == 4_096)
        #expect(layout.expert(layer: 0, expert: 1).offset == 16_384)
    }

    @Test func tinyMappingUsesRealFilesAndPinsCrossSubmissionReuse() async throws {
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: 1, expertCount: 8, hiddenSize: 4,
            intermediateSize: 2, expertStride: 16_384)
        let sizes = try qwenV2SourceData(configuration: configuration)
        let directory = try makeTinyExpertDirectory(configuration: configuration, sizes: sizes)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let coordinator = try QwenExpertMappingCoordinator(
            directoryURL: directory, layout: try decodeTinyLayout(configuration: configuration, sizes: sizes),
            layer: 0, device: context.device, slotCount: 2, cachePolicy: .lfu)

        let first = try await coordinator.map(expertIDs: [0, 1])
        #expect(first.diagnostics.misses == 2)
        #expect(first.diagnostics.hits == 0)
        let second = try await coordinator.map(expertIDs: [0])
        #expect(second.diagnostics.hits == 1)
        #expect(second.diagnostics.misses == 0)
        #expect(first.experts[0].buffer === second.experts[0].buffer)

        try first.cancel()
        #expect(first.snapshot().completed)
        do {
            _ = try await coordinator.map(expertIDs: [2, 3])
            Issue.record("Expected pinned active slot capacity failure")
        } catch let error as QwenExpertMappingError {
            #expect(error == .insufficientUnpinnedSlots)
        }
        try second.cancel()
        let evicted = try await coordinator.map(expertIDs: [2, 3])
        #expect(evicted.diagnostics.misses == 2)
        try evicted.cancel()

        do {
            _ = try await coordinator.map(expertIDs: [4, 4])
            Issue.record("Expected duplicate expert rejection")
        } catch let error as QwenExpertMappingError {
            #expect(error == .duplicateExpertWithinToken(4))
        }
    }

    @Test func shortReadAfterCoordinatorConstructionFailsAndDoesNotPublishLease() async throws {
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: 1, expertCount: 8, hiddenSize: 4,
            intermediateSize: 2, expertStride: 16_384)
        let sizes = try qwenV2SourceData(configuration: configuration)
        let directory = try makeTinyExpertDirectory(configuration: configuration, sizes: sizes)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let layout = try decodeTinyLayout(configuration: configuration, sizes: sizes)
        let coordinator = try QwenExpertMappingCoordinator(
            directoryURL: directory, layout: layout, layer: 0,
            device: context.device, slotCount: 1, cachePolicy: .lfu)
        let file = directory.appendingPathComponent("packed_experts/layer_00.bin")
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(configuration.expertCount) * configuration.expertStride - 1)
        try handle.close()
        var sawShortRead = false
        do {
            _ = try await coordinator.map(expertIDs: [7])
            Issue.record("Expected a short read after construction")
        } catch let error as StreamerError {
            guard case .sizeMismatch = error else {
                Issue.record("Unexpected streamer failure: \(error)")
                return
            }
            sawShortRead = true
        } catch {
            Issue.record("Unexpected mapping failure: \(error)")
        }
        #expect(sawShortRead)

        let restore = try FileHandle(forWritingTo: file)
        try restore.truncate(atOffset: UInt64(configuration.expertCount) * configuration.expertStride)
        try restore.close()
        let recovered = try await coordinator.map(expertIDs: [0])
        #expect(recovered.diagnostics.misses == 1)
        try recovered.cancel()
    }

    @Test func evictionRefreshesMappedContentInsteadOfReusingStaleBytes() async throws {
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: 1, expertCount: 8, hiddenSize: 4,
            intermediateSize: 2, expertStride: 16_384)
        let sizes = try qwenV2SourceData(configuration: configuration)
        let directory = try makeTinyExpertDirectory(configuration: configuration, sizes: sizes)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeMarker(0x11, expert: 0, configuration: configuration, directory: directory)
        try writeMarker(0x22, expert: 1, configuration: configuration, directory: directory)
        let context = try MetalContext()
        let coordinator = try QwenExpertMappingCoordinator(
            directoryURL: directory, layout: try decodeTinyLayout(configuration: configuration, sizes: sizes),
            layer: 0, device: context.device, slotCount: 1, cachePolicy: .lfu)
        let first = try await coordinator.map(expertIDs: [0])
        let old = first.experts[0].buffer.contents().assumingMemoryBound(to: UInt8.self).pointee
        try first.cancel()
        let second = try await coordinator.map(expertIDs: [1])
        let refreshed = second.experts[0].buffer.contents().assumingMemoryBound(to: UInt8.self).pointee
        #expect(old == 0x11)
        #expect(refreshed == 0x22)
        #expect(old != refreshed)
        try second.cancel()
    }

    @Test func realGpuRoutingMatchesIndependentReferenceAndReportsStatus() throws {
        let configuration = try QwenMoEConfiguration(
            hiddenSize: 3, expertCount: 10, routedIntermediateSize: 2,
            sharedIntermediateSize: 2)
        let logits: [Float] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               1.0, 0.5, 0.0, -0.5, -1.0, 2.0, 1.5, 0.25, -0.25, -0.75]
        let expected = independentRoute(logits: logits, expertCount: 10, topK: 8)
        let context = try MetalContext()
        let qwen = try QwenMoE(context: context, configuration: configuration)
        let logitsBuffer = try #require(floatBuffer(logits, device: context.device))
        let idsBuffer = try #require(context.device.makeBuffer(
            length: 2 * configuration.topK * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let weightsBuffer = try #require(context.device.makeBuffer(
            length: 2 * configuration.topK * MemoryLayout<Float>.stride, options: .storageModeShared))
        let statusBuffer = try #require(context.device.makeBuffer(
            length: 2 * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try qwen.encodeRouting(commandBuffer: commandBuffer, logits: logitsBuffer,
                               tokenCount: 2, selectedExpertIDs: idsBuffer,
                               normalizedWeights: weightsBuffer, status: statusBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        #expect(commandBuffer.error == nil)
        let actual = try QwenMoE.readRoutingDiagnostics(
            selectedExpertIDs: idsBuffer, normalizedWeights: weightsBuffer,
            status: statusBuffer, tokenCount: 2, configuration: configuration)
        #expect(actual.selectedExpertIDs == expected.ids)
        expectClose(actual.normalizedWeights.flatMap { $0 },
                    expected.weights.flatMap { $0 }, tolerance: 2e-6)

        let nanLogits = logits.enumerated().map { index, value in index == 3 ? .nan : value }
        let nanCommand = try #require(context.queue.makeCommandBuffer())
        let nanLogitsBuffer = try #require(floatBuffer(nanLogits, device: context.device))
        try qwen.encodeRouting(commandBuffer: nanCommand, logits: nanLogitsBuffer,
                               tokenCount: 2, selectedExpertIDs: idsBuffer,
                               normalizedWeights: weightsBuffer, status: statusBuffer)
        nanCommand.commit()
        nanCommand.waitUntilCompleted()
        #expect(throws: QwenMoEError.routingKernelRejectedInput(token: 0)) {
            try QwenMoE.readRoutingDiagnostics(
                selectedExpertIDs: idsBuffer, normalizedWeights: weightsBuffer,
                status: statusBuffer, tokenCount: 2, configuration: configuration)
        }
    }

    @Test func submittedLeaseRemainsPinnedUntilGpuCompletion() async throws {
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: 1, expertCount: 9, hiddenSize: 4,
            intermediateSize: 2, expertStride: 16_384)
        let sizes = try qwenV2SourceData(configuration: configuration)
        let directory = try makeTinyExpertDirectory(configuration: configuration, sizes: sizes)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeUnitPackedWeights(directory: directory, configuration: configuration, sizes: sizes)
        let context = try MetalContext()
        let layout = try decodeTinyLayout(configuration: configuration, sizes: sizes)
        let coordinator = try QwenExpertMappingCoordinator(
            directoryURL: directory, layout: layout, layer: 0,
            device: context.device, slotCount: 8, cachePolicy: .lfu)
        let lease = try await coordinator.map(expertIDs: Array(0..<8))
        let output = try #require(context.device.makeBuffer(
            length: 4 * MemoryLayout<Float>.stride, options: .storageModeShared))
        let qwen = try QwenMoE(context: context, configuration: try QwenMoEConfiguration(
            hiddenSize: 4, expertCount: 9, routedIntermediateSize: 2,
            sharedIntermediateSize: 2))
        let hidden = try #require(floatBuffer([1, 2, 3, 4], device: context.device))
        let weights = try #require(floatBuffer([1, 0, 0, 0, 0, 0, 0, 0], device: context.device))
        let scratch = try qwen.makeScratch()
        let sharedBindings = QwenMoESharedAffineBindings(
            gate: try patternedAffineBinding(device: context.device, rows: 2, columns: 4,
                                              valueByte: 0x31, scaleBits: 0x3f80, biasBits: 0),
            up: try patternedAffineBinding(device: context.device, rows: 2, columns: 4,
                                            valueByte: 0x42, scaleBits: 0x3f00, biasBits: 0x3e80),
            down: try patternedAffineBinding(device: context.device, rows: 4, columns: 2,
                                              valueByte: 0x54, scaleBits: 0x3e80, biasBits: 0xbe80),
            outputGate: try patternedAffineBinding(device: context.device, rows: 1, columns: 4,
                                                    valueByte: 0x21, scaleBits: 0x3f80, biasBits: 0))
        let event = try #require(context.device.makeSharedEvent())
        let blocker = try #require(context.queue.makeCommandBuffer())
        blocker.encodeWaitForEvent(event, value: 1)
        blocker.commit()
        let submitted = try qwen.submitExperts(
            hidden: hidden, lease: lease, routingWeights: weights,
            sharedBindings: sharedBindings, scratch: scratch, output: output)
        #expect(submitted.status != .notEnqueued)
        try lease.cancel()
        #expect(lease.snapshot().submitted)
        #expect(lease.snapshot().canceled)
        do {
            _ = try await coordinator.map(expertIDs: [8])
            Issue.record("Expected submitted lease to pin its slot")
        } catch let error as QwenExpertMappingError {
            #expect(error == .insufficientUnpinnedSlots)
        }
        #expect(!lease.snapshot().completed)
        event.signaledValue = 1
        _ = await blocker.completed()
        _ = await submitted.completed()
        #expect(lease.snapshot().completed)
        #expect(lease.snapshot().succeeded == false)
        let actualOutput = readFloats(output, count: 4)
        let expectedOutput = packedReferenceOutput(hidden: [1, 2, 3, 4])
        expectClose(actualOutput, expectedOutput, tolerance: 2e-4)
        #expect(actualOutput.allSatisfy { $0.isFinite })
        #expect(actualOutput.contains { $0 != 0 })
        let reused = try await coordinator.map(expertIDs: [8])
        #expect(reused.diagnostics.misses == 1)
        try reused.cancel()
    }

    @Test func prevalidationFailureLeavesLeaseUnsubmittedAndCancellationReleasesSlots() async throws {
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: 1, expertCount: 9, hiddenSize: 4,
            intermediateSize: 2, expertStride: 16_384)
        let sizes = try qwenV2SourceData(configuration: configuration)
        let directory = try makeTinyExpertDirectory(configuration: configuration, sizes: sizes)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeUnitPackedWeights(directory: directory, configuration: configuration, sizes: sizes)
        let context = try MetalContext()
        let layout = try decodeTinyLayout(configuration: configuration, sizes: sizes)
        let coordinator = try QwenExpertMappingCoordinator(
            directoryURL: directory, layout: layout, layer: 0,
            device: context.device, slotCount: 8, cachePolicy: .lfu)
        let lease = try await coordinator.map(expertIDs: Array(0..<8))
        let qwen = try QwenMoE(context: context, configuration: try QwenMoEConfiguration(
            hiddenSize: 4, expertCount: 9, routedIntermediateSize: 2,
            sharedIntermediateSize: 2))
        let shortHidden = try #require(floatBuffer([1, 2, 3], device: context.device))
        let weights = try #require(floatBuffer([1, 0, 0, 0, 0, 0, 0, 0], device: context.device))
        let output = try #require(context.device.makeBuffer(
            length: 4 * MemoryLayout<Float>.stride, options: .storageModeShared))
        let scratch = try qwen.makeScratch()
        let sharedBindings = QwenMoESharedAffineBindings(
            gate: try patternedAffineBinding(device: context.device, rows: 2, columns: 4,
                                              valueByte: 0x31, scaleBits: 0x3f80, biasBits: 0),
            up: try patternedAffineBinding(device: context.device, rows: 2, columns: 4,
                                            valueByte: 0x42, scaleBits: 0x3f00, biasBits: 0x3e80),
            down: try patternedAffineBinding(device: context.device, rows: 4, columns: 2,
                                              valueByte: 0x54, scaleBits: 0x3e80, biasBits: 0xbe80),
            outputGate: try patternedAffineBinding(device: context.device, rows: 1, columns: 4,
                                                    valueByte: 0x21, scaleBits: 0x3f80, biasBits: 0))
        #expect(throws: QwenMoEError.bufferTooSmall(
            name: "hidden", required: 16, actual: shortHidden.length)) {
            try qwen.submitExperts(
                hidden: shortHidden, lease: lease, routingWeights: weights,
                sharedBindings: sharedBindings, scratch: scratch, output: output)
        }
        #expect(!lease.snapshot().submitted)
        try lease.cancel()
        #expect(lease.snapshot().completed)
        let reused = try await coordinator.map(expertIDs: [8])
        #expect(reused.diagnostics.misses == 1)
        try reused.cancel()
    }
}

private enum QwenTestError: Error { case malformedFixture; case malformedTensor(String) }

private func qwenFixtureSection() throws -> [String: Any] {
    guard let url = Bundle.module.url(forResource: "qwen36-tiny-fixtures", withExtension: "json") else {
        throw QwenTestError.malformedFixture
    }
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
    guard let root = object as? [String: Any], let moe = root["moe"] as? [String: Any] else {
        throw QwenTestError.malformedFixture
    }
    return moe
}

private func tensorFloats(_ section: [String: Any], key: String) throws -> [Float] {
    guard let tensor = section[key] as? [String: Any], let values = tensor["values"] as? [NSNumber] else {
        throw QwenTestError.malformedTensor(key)
    }
    return values.map(\.floatValue)
}

private func tensorShape(_ section: [String: Any], key: String) throws -> [Int] {
    guard let tensor = section[key] as? [String: Any], let shape = tensor["shape"] as? [NSNumber] else {
        throw QwenTestError.malformedTensor(key)
    }
    return shape.map(\.intValue)
}

private func tensorInts(_ section: [String: Any], key: String) throws -> [Int] {
    try tensorFloats(section, key: key).map(Int.init)
}

private func pinnedSynthetic(count: Int, phase: Float, scale: Float) -> [Float] {
    (0..<count).map { index in
        sin(Float(index) * 0.37 + phase) * scale
    }
}

private struct RouteReference {
    let ids: [[Int]]
    let weights: [[Float]]
    let probabilities: [[Float]]
}

private func independentRoute(logits: [Float], expertCount: Int, topK: Int) -> RouteReference {
    let tokenCount = logits.count / expertCount
    var ids: [[Int]] = []
    var weights: [[Float]] = []
    var probabilities: [[Float]] = []
    for token in 0..<tokenCount {
        let row = Array(logits[(token * expertCount)..<((token + 1) * expertCount)])
        let maximum = row.max()!
        let exponentials = row.map { Foundation.exp(Double($0 - maximum)) }
        let denominator = exponentials.reduce(0, +)
        let probability = exponentials.map { Float($0 / denominator) }
        let selected = probability.indices.sorted {
            probability[$0] == probability[$1] ? $0 < $1 : probability[$0] > probability[$1]
        }.prefix(topK)
        let selectedIDs = Array(selected)
        let selectedSum = selectedIDs.reduce(Float.zero) { $0 + probability[$1] }
        ids.append(selectedIDs)
        weights.append(selectedIDs.map { probability[$0] / selectedSum })
        probabilities.append(probability)
    }
    return RouteReference(ids: ids, weights: weights, probabilities: probabilities)
}

private func denseExpert(hidden: Int, intermediate: Int, seed: Float) -> QwenMoEDenseExpert {
    let gateUp = (0..<(2 * intermediate * hidden)).map {
        sin(Float($0) * 0.31 + seed) * 0.3
    }
    let down = (0..<(hidden * intermediate)).map {
        cos(Float($0) * 0.19 + seed) * 0.25
    }
    return QwenMoEDenseExpert(gateUp: gateUp, down: down)
}

private func denseShared(hidden: Int, intermediate: Int, seed: Float) -> QwenMoEDenseSharedExpert {
    QwenMoEDenseSharedExpert(
        gate: (0..<(intermediate * hidden)).map { sin(Float($0) * 0.23 + seed) * 0.3 },
        up: (0..<(intermediate * hidden)).map { cos(Float($0) * 0.17 + seed) * 0.3 },
        down: (0..<(hidden * intermediate)).map { sin(Float($0) * 0.11 + seed) * 0.25 },
        outputGate: (0..<hidden).map { cos(Float($0) * 0.27 + seed) * 0.2 })
}

private struct EvaluationReference {
    let routed: [Float]
    let shared: [Float]
    let sharedRaw: [Float]
    let gates: [Float]
    let output: [Float]
}

private func independentEvaluate(
    input: [Float], logits: [Float], configuration: QwenMoEConfiguration,
    routedExperts: [Int: QwenMoEDenseExpert], sharedExpert: QwenMoEDenseSharedExpert,
    normalizeSelected: Bool = true
) -> EvaluationReference {
    let route = independentRoute(logits: logits, expertCount: configuration.expertCount, topK: configuration.topK)
    let hidden = input
    var routed = [Float](repeating: 0, count: configuration.hiddenSize)
    for rank in 0..<configuration.topK {
        let id = route.ids[0][rank]
        let expert = routedExperts[id]!
        let projected = project(hidden, matrix: expert.gateUp,
                                rows: 2 * configuration.routedIntermediateSize,
                                columns: configuration.hiddenSize)
        let activation = (0..<configuration.routedIntermediateSize).map {
            let gate = projected[$0]
            let up = projected[configuration.routedIntermediateSize + $0]
            return gate / (1 + Foundation.exp(-gate)) * up
        }
        let down = project(activation, matrix: expert.down,
                           rows: configuration.hiddenSize,
                           columns: configuration.routedIntermediateSize)
        let weight = normalizeSelected ? route.weights[0][rank] : route.probabilities[0][id]
        for index in routed.indices { routed[index] += down[index] * weight }
    }
    let gateProjection = project(hidden, matrix: sharedExpert.gate,
                                 rows: configuration.sharedIntermediateSize,
                                 columns: configuration.hiddenSize)
    let upProjection = project(hidden, matrix: sharedExpert.up,
                               rows: configuration.sharedIntermediateSize,
                               columns: configuration.hiddenSize)
    let activation = (0..<configuration.sharedIntermediateSize).map {
        let value = gateProjection[$0]
        return value / (1 + Foundation.exp(-value)) * upProjection[$0]
    }
    let down = project(activation, matrix: sharedExpert.down,
                       rows: configuration.hiddenSize,
                       columns: configuration.sharedIntermediateSize)
    let rawGate = (0..<configuration.hiddenSize).reduce(Float.zero) {
        $0 + hidden[$1] * sharedExpert.outputGate[$1]
    }
    let gate = 1 / (1 + Foundation.exp(-rawGate))
    let shared = down.map { $0 * gate }
    return EvaluationReference(
        routed: routed, shared: shared, sharedRaw: down,
        gates: [gate], output: zip(routed, shared).map(+))
}

private func project(_ input: [Float], matrix: [Float], rows: Int, columns: Int) -> [Float] {
    (0..<rows).map { row in
        (0..<columns).reduce(Float.zero) { sum, column in
            sum + matrix[row * columns + column] * input[column]
        }
    }
}

private func qwenV2SourceData(configuration: QwenPackedLayoutConfiguration) throws -> (
    gateUp: (values: UInt64, metadata: UInt64), down: (values: UInt64, metadata: UInt64)
) {
    func sizes(rows: Int, columns: Int) -> (UInt64, UInt64) {
        let groups = (columns + 63) / 64
        return (
            UInt64(rows * ((columns * 4 + 7) / 8)),
            UInt64(rows * groups * 2))
    }
    return (
        sizes(rows: 2 * configuration.intermediateSize, columns: configuration.hiddenSize),
        sizes(rows: configuration.hiddenSize, columns: configuration.intermediateSize))
}

private func qwenV2Data(
    configuration: QwenPackedLayoutConfiguration,
    sizes: (gateUp: (values: UInt64, metadata: UInt64), down: (values: UInt64, metadata: UInt64))
) throws -> Data {
    let gateUpValues = UInt64(0)
    let gateUpScales = gateUpValues + sizes.gateUp.values
    let gateUpBiases = gateUpScales + sizes.gateUp.metadata
    let downValues = gateUpBiases + sizes.gateUp.metadata
    let downScales = downValues + sizes.down.values
    let downBiases = downScales + sizes.down.metadata
    let layers: [[String: Any]] = (0..<configuration.layerCount).map { layerIndex in
        [
            "layer": layerIndex,
            "path": "packed_experts/layer_\(String(format: "%02d", layerIndex)).bin",
            "experts": configuration.expertCount, "stride": configuration.expertStride,
            "sources": [
                ["name": "experts.gate_up_proj", "role": "gate_up",
                 "shape": [2 * configuration.intermediateSize, configuration.hiddenSize],
                 "valuesOffset": gateUpValues, "valuesSize": sizes.gateUp.values,
                 "scalesOffset": gateUpScales, "scalesSize": sizes.gateUp.metadata,
                 "biasesOffset": gateUpBiases, "biasesSize": sizes.gateUp.metadata],
                ["name": "experts.down_proj", "role": "down",
                 "shape": [configuration.hiddenSize, configuration.intermediateSize],
                 "valuesOffset": downValues, "valuesSize": sizes.down.values,
                 "scalesOffset": downScales, "scalesSize": sizes.down.metadata,
                 "biasesOffset": downBiases, "biasesSize": sizes.down.metadata],
            ],
        ]
    }
    let root: [String: Any] = ["version": 2, "layers": layers]
    return try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
}

private func officialQwenManifest() throws -> LoadedModelManifest {
    let digest = String(repeating: "a", count: 64)
    let layerTypes: [GTurboQwenLayerTypeV2] = (0..<40).map {
        ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
    }
    let architecture = GTurboQwenArchitectureV2(
        hiddenSize: 2_048, numLayers: 40, layerTypes: layerTypes,
        numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256,
        attentionOutputGate: true, linearConvolutionKernel: 4,
        linearKeyHeads: 16, linearKeyHeadDimension: 128,
        linearValueHeads: 32, linearValueHeadDimension: 128,
        recurrentStateType: .fp32, partialRotaryFactor: 0.25,
        ropeTheta: 10_000_000, mropeInterleaved: true, mropeSections: [11, 11, 10],
        numberOfExperts: 256, expertsPerToken: 8,
        routedExpertIntermediateSize: 512, sharedExpertIntermediateSize: 512,
        vocabularySize: 248_320, tiedWordEmbeddings: false,
        hiddenActivation: "silu", bosTokenID: 248_044, eosTokenID: 248_044,
        imageTokenID: 248_056, videoTokenID: 248_057,
        visionStartTokenID: 248_053, visionEndTokenID: 248_054)
    let quantization = GTurboQuantizationCategoryV2.allCases.map { category in
        category == .recurrentState
            ? GTurboQuantizationGroupV2(category: category, storage: .fp32)
            : GTurboQuantizationGroupV2(
                category: category, storage: .affineInt4, groupSize: 64,
                scaleType: "bf16", biasType: "bf16")
    }
    let omissions = [
        "fc.weight", "layers.0.input_layernorm.weight", "layers.0.mlp.experts.down_proj",
        "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.gate.weight",
        "layers.0.mlp.shared_expert.down_proj.weight",
        "layers.0.mlp.shared_expert.gate_proj.weight",
        "layers.0.mlp.shared_expert.up_proj.weight",
        "layers.0.mlp.shared_expert_gate.weight",
        "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight",
        "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
        "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.v_proj.weight", "norm.weight",
        "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
    ].map { GTurboIgnoredTensorV2(name: "mtp.\($0)", reason: .unsupportedMTP) }
    let expertFileSize = UInt64(256 * 2_097_152)
    var files: [String: GTurboManifestFileV1] = [
        "model_weights.bin": .init(size: 32_768, sha256: digest),
        "packed_experts/layout.json": .init(size: 1, sha256: digest),
    ]
    for layer in 0..<40 {
        files["packed_experts/layer_\(String(format: "%02d", layer)).bin"] =
            .init(size: expertFileSize, sha256: digest)
    }
    let manifest = GTurboManifestV2(
        family: .qwen3_6,
        requiredFeatures: [.familyDispatch, .verifiedIdentity, .qwenHybridAttention, .qwenMTPExcluded],
        modelID: GTurboFormatV2.qwenRepository,
        architecture: .qwen3_6(architecture),
        provenance: .init(
            sourceRepository: GTurboFormatV2.qwenRepository,
            sourceRevision: GTurboFormatV2.qwenRevision,
            sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
            sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
            quantizationPolicySHA256: digest),
        quantization: quantization, ignoredTensors: omissions, files: files,
        tensorRegions: [.init(
            name: "embed", file: "model_weights.bin", offset: 0, size: 16,
            shape: [1], storage: .affineInt4,
            quantizationCategory: .embedding)],
        expertsPerLayer: 256, numLayers: 40, expertStride: 2_097_152)
    return try ManifestReader.decodeVerified(data: GTurboManifestV2Codec.encode(manifest))
}

private func decodeTinyLayout(
    configuration: QwenPackedLayoutConfiguration,
    sizes: (gateUp: (values: UInt64, metadata: UInt64), down: (values: UInt64, metadata: UInt64))
) throws -> PackedExpertsLayout {
    try PackedExpertsLayoutReader.decodeQwenV2(
        data: qwenV2Data(configuration: configuration, sizes: sizes),
        configuration: configuration,
        manifestFileSizes: [
            "packed_experts/layer_00.bin": UInt64(configuration.expertCount) * configuration.expertStride,
        ])
}

private func makeTinyExpertDirectory(
    configuration: QwenPackedLayoutConfiguration,
    sizes: (gateUp: (values: UInt64, metadata: UInt64), down: (values: UInt64, metadata: UInt64))
) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("qwen-moe-")
        .appendingPathComponent(UUID().uuidString)
    let packed = directory.appendingPathComponent("packed_experts")
    try FileManager.default.createDirectory(at: packed, withIntermediateDirectories: true)
    let file = packed.appendingPathComponent("layer_00.bin")
    FileManager.default.createFile(atPath: file.path, contents: nil)
    let handle = try FileHandle(forWritingTo: file)
    try handle.truncate(atOffset: UInt64(configuration.expertCount) * configuration.expertStride)
    try handle.close()
    try qwenV2Data(configuration: configuration, sizes: sizes)
        .write(to: packed.appendingPathComponent("layout.json"))
    return directory
}

private func writeUnitPackedWeights(
    directory: URL, configuration: QwenPackedLayoutConfiguration,
    sizes: (gateUp: (values: UInt64, metadata: UInt64), down: (values: UInt64, metadata: UInt64))
) throws {
    let file = directory.appendingPathComponent("packed_experts/layer_00.bin")
    let layout = try decodeTinyLayout(configuration: configuration, sizes: sizes)
    let handle = try FileHandle(forWritingTo: file)
    defer { try? handle.close() }
    let metadataCount = { (size: UInt64) -> Int in Int(size / 2) }
    let patterns: [String: (value: UInt8, scale: UInt16, bias: UInt16)] = [
        "gate_up": (0x21, 0x3f00, 0xbe80),
        "down": (0x43, 0x3e80, 0xbe80),
    ]
    for expert in 0..<configuration.expertCount {
        let base = UInt64(expert) * configuration.expertStride
        for role in ["gate_up", "down"] {
            guard let descriptor = layout.layers[0].affineDescriptors[role],
                  let pattern = patterns[role] else {
                throw QwenTestError.malformedFixture
            }
            try handle.seek(toOffset: base + descriptor.valuesOffset)
            try handle.write(contentsOf: Data(
                repeating: pattern.value, count: Int(descriptor.valuesSize)))
            try handle.seek(toOffset: base + descriptor.scalesOffset)
            try handle.write(contentsOf: Data(
                bytes: [UInt16](repeating: pattern.scale, count: metadataCount(descriptor.scalesSize)),
                count: metadataCount(descriptor.scalesSize) * 2))
            try handle.seek(toOffset: base + descriptor.biasesOffset)
            try handle.write(contentsOf: Data(
                bytes: [UInt16](repeating: pattern.bias, count: metadataCount(descriptor.biasesSize)),
                count: metadataCount(descriptor.biasesSize) * 2))
        }
    }
}

private func packedReferenceOutput(hidden: [Float]) -> [Float] {
    let routedGateUp = packedProjection(
        hidden: hidden, rows: 4, columns: 4,
        valueByte: 0x21, scaleBits: 0x3f00, biasBits: 0xbe80)
    let routedActivation = (0..<2).map { index in
        let gate = routedGateUp[index]
        let up = routedGateUp[2 + index]
        return gate / (1 + Foundation.exp(-gate)) * up
    }
    let routed = packedProjection(
        hidden: routedActivation, rows: 4, columns: 2,
        valueByte: 0x43, scaleBits: 0x3e80, biasBits: 0xbe80)
    let sharedGate = packedProjection(
        hidden: hidden, rows: 2, columns: 4,
        valueByte: 0x31, scaleBits: 0x3f80, biasBits: 0)
    let sharedUp = packedProjection(
        hidden: hidden, rows: 2, columns: 4,
        valueByte: 0x42, scaleBits: 0x3f00, biasBits: 0x3e80)
    let sharedActivation = (0..<2).map { index in
        let gate = sharedGate[index]
        return gate / (1 + Foundation.exp(-gate)) * sharedUp[index]
    }
    let sharedDown = packedProjection(
        hidden: sharedActivation, rows: 4, columns: 2,
        valueByte: 0x54, scaleBits: 0x3e80, biasBits: 0xbe80)
    let gateProjection = packedProjection(
        hidden: hidden, rows: 1, columns: 4,
        valueByte: 0x21, scaleBits: 0x3f80, biasBits: 0)[0]
    let gate = 1 / (1 + Foundation.exp(-gateProjection))
    return zip(routed, sharedDown).map { $0 + $1 * gate }
}

private func packedProjection(
    hidden: [Float], rows: Int, columns: Int,
    valueByte: UInt8, scaleBits: UInt16, biasBits: UInt16
) -> [Float] {
    let scale = Float(bitPattern: UInt32(scaleBits) << 16)
    let bias = Float(bitPattern: UInt32(biasBits) << 16)
    return (0..<rows).map { row in
        (0..<columns).reduce(Float.zero) { sum, column in
            let quantized = Float((column.isMultiple(of: 2) ? valueByte & 0x0F : valueByte >> 4))
            return sum + (scale * quantized + bias) * hidden[column]
        }
    }
}

private func writeMarker(
    _ marker: UInt8, expert: Int, configuration: QwenPackedLayoutConfiguration,
    directory: URL
) throws {
    let file = directory.appendingPathComponent("packed_experts/layer_00.bin")
    let handle = try FileHandle(forWritingTo: file)
    try handle.seek(toOffset: UInt64(expert) * configuration.expertStride)
    try handle.write(contentsOf: Data([marker]))
    try handle.close()
}

private func floatBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer? {
    device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Float>.stride,
                      options: .storageModeShared)
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
}

private func patternedAffineBinding(
    device: MTLDevice, rows: Int, columns: Int, valueByte: UInt8,
    scaleBits: UInt16, biasBits: UInt16
) throws -> QwenMoEAffineBinding {
    let layout = try QwenMetalAffineLayout(rows: rows, columns: columns, bitWidth: 4)
    let valueCount = Int(layout.rowCount) * Int(layout.valuesRowStrideBytes)
    let metadataCount = Int(layout.rowCount) * Int(layout.metadataRowStrideBytes) / MemoryLayout<UInt16>.stride
    let values = [UInt8](repeating: valueByte, count: valueCount)
    let scales = [UInt16](repeating: scaleBits, count: metadataCount)
    let biases = [UInt16](repeating: biasBits, count: metadataCount)
    guard let valueBuffer = device.makeBuffer(bytes: values, length: values.count,
                                               options: .storageModeShared),
          let scaleBuffer = device.makeBuffer(bytes: scales,
                                               length: scales.count * MemoryLayout<UInt16>.stride,
                                               options: .storageModeShared),
          let biasBuffer = device.makeBuffer(bytes: biases,
                                              length: biases.count * MemoryLayout<UInt16>.stride,
                                              options: .storageModeShared) else {
        throw QwenMoEError.bufferTooSmall(name: "patternedAffine", required: 1, actual: 0)
    }
    return QwenMoEAffineBinding(values: valueBuffer, scales: scaleBuffer, biases: biasBuffer,
                                layout: layout)
}

private func expectClose(
    _ actual: [Float], _ expected: [Float], tolerance: Float,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(actual.count == expected.count, sourceLocation: sourceLocation)
    for (index, pair) in zip(actual, expected).enumerated() {
        #expect(abs(pair.0 - pair.1) <= tolerance + tolerance * abs(pair.1),
                "index \(index): \(pair.0) != \(pair.1)", sourceLocation: sourceLocation)
    }
}

private func maxDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
    zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
}
