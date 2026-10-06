import Darwin
import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Qualifies the full 512-wide original-BF16 routed branch against a pinned
/// CPU Torch oracle. The only dense tensors are generated in memory.
@Suite(.serialized) struct QwenBF16LargeMoEParityTests {
    private let hiddenSize = 512
    private let intermediateSize = 512
    private let expertIDsInLeaseOrder = Array((0..<8).reversed())
    private let routeWeightsInLeaseOrder: [Float] = [
        0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.25, 0.125,
    ]

    @Test func largeBF16RoutedBranchMatchesPinnedTorchAndRejectsLeaseRankOrder() async throws {
        let manifest = try loadManifest()
        #expect(manifest.formatVersion == 1)
        #expect(manifest.torchVersion == "2.10.0")
        #expect(manifest.torchGit == "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0")
        #expect(manifest.torchThreads == 1)
        #expect(manifest.torchInteropThreads == 1)
        #expect(manifest.device == "cpu")
        #expect(manifest.geometry == .init(
            hiddenSize: hiddenSize, routedIntermediateSize: intermediateSize,
            sharedIntermediateSize: 1, expertCount: 8, topK: 8))
        #expect(manifest.leaseExpertIDs == expertIDsInLeaseOrder)
        #expect(manifest.accumulationExpertIDs == Array(0..<8))
        #expect(manifest.routeWeightFloat32Bits == routeWeightsInLeaseOrder.map(\.bitPattern))

        let fixture = LargeMoEParityWords()
        expectFixtureHashes(fixture, manifest: manifest)
        let expected = try loadOutput(
            named: manifest.expectedOutputFile, sha256: manifest.expectedOutputSHA256)
        let rankOrderControl = try loadOutput(
            named: manifest.leaseRankControlFile, sha256: manifest.leaseRankControlSHA256)
        #expect(expected.count == hiddenSize)
        #expect(rankOrderControl.count == hiddenSize)
        #expect(expected.allSatisfy { $0.isFinite })
        #expect(rankOrderControl.allSatisfy { $0.isFinite })
        #expect(abs(expected[0] - rankOrderControl[0]) > 0.05,
                "the pinned fixture must expose the selected-rank accumulation error")

        let actual = try await runProductionMoE(fixture)
        #expect(actual.count == hiddenSize)
        #expect(actual.allSatisfy { $0.isFinite })
        for index in expected.indices {
            let limit = 1e-7 + 1e-6 * abs(expected[index])
            #expect(abs(actual[index] - expected[index]) <= limit,
                    "output \(index): actual \(actual[index]), pinned Torch \(expected[index]), limit \(limit)")
        }
        #expect(abs(actual[0] - rankOrderControl[0]) > 0.05,
                "production output must reject lease-rank accumulation")
    }

    @Test func denseBF16RoutedBranchMatchesIndependentSource64ActivationOracle() async throws {
        let hiddenSize = 512
        let intermediateSize = 512
        let expertIDs = Array((0..<8).reversed())
        let routeWeights: [Float] = [0.01, 0.02, 0.04, 0.08, 0.16, 0.20, 0.21, 0.28]
        let hidden: [Float] = (0..<hiddenSize).map { index -> Float in
            let magnitude: Float = 0.25 + Float((index * 19) % 97) / 128
            let sign: Float = (index & 1) == 0 ? 1.0 : -1.0
            return magnitude * sign
        }
        let gateUp = denseBF16Words(count: 8 * 2 * intermediateSize * hiddenSize,
                                    salt: 17)
        let down = denseBF16Words(count: 8 * hiddenSize * intermediateSize, salt: 43)

        let actual = try await runDenseProductionMoE(
            hidden: hidden, expertIDs: expertIDs, routeWeights: routeWeights,
            gateUp: gateUp, down: down,
            hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        let expected = denseRoutedReference(
            hidden: hidden, expertIDs: expertIDs, routeWeights: routeWeights,
            gateUp: gateUp, down: down,
            hiddenSize: hiddenSize, intermediateSize: intermediateSize)

        #expect(actual.count == expected.count)
        #expect(actual.allSatisfy { $0.isFinite })
        #expect(expected.allSatisfy { $0.isFinite })
        for index in expected.indices {
            #expect(actual[index].bitPattern == expected[index].bitPattern,
                    "output \(index): cooperative \(actual[index]), source64 oracle \(expected[index])")
        }
    }

    @Test func denseBF16RoutedBranchPreservesNonfiniteClassification() async throws {
        let hiddenSize = 512
        let intermediateSize = 512
        let expertIDs = Array((0..<8).reversed())
        let routeWeights: [Float] = [0.01, 0.02, 0.04, 0.08, 0.16, 0.20, 0.21, 0.28]
        let hidden = [Float](repeating: 1, count: hiddenSize)
        var gateUp = denseBF16Words(count: 8 * 2 * intermediateSize * hiddenSize,
                                    salt: 71)
        gateUp[0] = 0x7fc0 // NaN in expert 0's first gate row.
        let down = denseBF16Words(count: 8 * hiddenSize * intermediateSize, salt: 89)

        let actual = try await runDenseProductionMoE(
            hidden: hidden, expertIDs: expertIDs, routeWeights: routeWeights,
            gateUp: gateUp, down: down,
            hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        #expect(actual.count == hiddenSize)
        #expect(actual.allSatisfy { $0.isNaN },
                "a nonfinite cooperative gate must remain nonfinite through routed reduction")
    }

    private func runProductionMoE(_ fixture: LargeMoEParityWords) async throws -> [Float] {
        let gateUpName = "large_moe.gate_up"
        let downName = "large_moe.down"
        let sharedGateName = "large_moe.shared_gate"
        let sharedUpName = "large_moe.shared_up"
        let sharedDownName = "large_moe.shared_down"
        let sharedOutputGateName = "large_moe.shared_output_gate"
        let source = try QwenBF16ExpertCacheSourceFixture.make(
            firstShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: gateUpName, shape: [8, 2 * intermediateSize, hiddenSize],
                    words: fixture.gateUp),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedGateName, shape: [1, hiddenSize],
                    words: fixture.sharedGate),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedUpName, shape: [1, hiddenSize],
                    words: fixture.sharedUp),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedOutputGateName, shape: [1, hiddenSize],
                    words: fixture.sharedOutputGate),
            ],
            secondShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: downName, shape: [8, hiddenSize, intermediateSize],
                    words: fixture.down),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedDownName, shape: [hiddenSize, 1],
                    words: fixture.sharedDown),
            ])
        defer { source.remove() }

        let context = try MetalContext()
        let configuration = try QwenMoEConfiguration(
            hiddenSize: hiddenSize, expertCount: 8, topK: 8,
            routedIntermediateSize: intermediateSize, sharedIntermediateSize: 1)
        let sharedWeights = try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [
                QwenBF16TensorSpec(name: sharedGateName, shardName: source.gateUpShardName,
                                   role: .sharedGate, rows: 1, columns: hiddenSize),
                QwenBF16TensorSpec(name: sharedUpName, shardName: source.gateUpShardName,
                                   role: .sharedUp, rows: 1, columns: hiddenSize),
                QwenBF16TensorSpec(name: sharedDownName, shardName: source.downShardName,
                                   role: .sharedDown, rows: hiddenSize, columns: 1),
                QwenBF16TensorSpec(name: sharedOutputGateName,
                                   shardName: source.gateUpShardName,
                                   role: .sharedOutputGate, rows: 1, columns: hiddenSize),
            ], residencyBudget: UInt64(4 * hiddenSize * MemoryLayout<UInt16>.stride))
        let pairBytes = (2 * intermediateSize * hiddenSize
            + hiddenSize * intermediateSize) * MemoryLayout<UInt16>.stride
        let coordinator = try QwenBF16ExpertMappingCoordinator(
            source: source.handle,
            names: QwenBF16RoutedSourceNames(
                gateUpShardName: source.gateUpShardName,
                gateUpTensorName: gateUpName,
                downShardName: source.downShardName,
                downTensorName: downName),
            layer: 0, configuration: configuration, device: context.device,
            slotCount: 8, residencyBudget: UInt64(pairBytes * 8))
        let lease = try await coordinator.map(expertIDs: expertIDsInLeaseOrder)
        #expect(lease.experts.map(\.expertID) == expertIDsInLeaseOrder)

        var hidden = [Float](repeating: 0, count: hiddenSize)
        hidden[0] = 1
        let hiddenBuffer = try floatBuffer(hidden, device: context.device)
        let routeWeights = try floatBuffer(routeWeightsInLeaseOrder, device: context.device)
        let output = try floatBuffer([Float](repeating: 0, count: hiddenSize),
                                     device: context.device)
        let moe = try QwenMoE(context: context, configuration: configuration)
        let scratch = try moe.makeScratch()
        let command = try moe.submitExpertsBF16(
            hidden: hiddenBuffer, lease: lease, routingWeights: routeWeights,
            sharedWeights: sharedWeights,
            sharedNames: QwenBF16SharedNames(
                gate: sharedGateName, up: sharedUpName,
                down: sharedDownName, outputGate: sharedOutputGateName),
            scratch: scratch, output: output)
        _ = await command.completed()
        #expect(command.status == .completed)
        #expect(command.error == nil)
        #expect(lease.snapshot().completed)
        #expect(lease.snapshot().succeeded == true)
        return readFloats(output, count: hiddenSize)
    }

    private func runDenseProductionMoE(
        hidden: [Float], expertIDs: [Int], routeWeights: [Float],
        gateUp: [UInt16], down: [UInt16], hiddenSize: Int, intermediateSize: Int
    ) async throws -> [Float] {
        let gateUpName = "dense_moe.gate_up"
        let downName = "dense_moe.down"
        let sharedGateName = "dense_moe.shared_gate"
        let sharedUpName = "dense_moe.shared_up"
        let sharedDownName = "dense_moe.shared_down"
        let sharedOutputGateName = "dense_moe.shared_output_gate"
        let source = try QwenBF16ExpertCacheSourceFixture.make(
            firstShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: gateUpName,
                    shape: [8, 2 * intermediateSize, hiddenSize], words: gateUp),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedGateName, shape: [1, hiddenSize],
                    words: [UInt16](repeating: 0, count: hiddenSize)),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedUpName, shape: [1, hiddenSize],
                    words: [UInt16](repeating: 0, count: hiddenSize)),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedOutputGateName, shape: [1, hiddenSize],
                    words: [UInt16](repeating: 0, count: hiddenSize)),
            ],
            secondShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: downName,
                    shape: [8, hiddenSize, intermediateSize], words: down),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedDownName, shape: [hiddenSize, 1],
                    words: [UInt16](repeating: 0, count: hiddenSize)),
            ])
        defer { source.remove() }

        let context = try MetalContext()
        let configuration = try QwenMoEConfiguration(
            hiddenSize: hiddenSize, expertCount: 8, topK: 8,
            routedIntermediateSize: intermediateSize, sharedIntermediateSize: 1)
        let sharedWeights = try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [
                QwenBF16TensorSpec(name: sharedGateName,
                                   shardName: source.gateUpShardName,
                                   role: .sharedGate, rows: 1, columns: hiddenSize),
                QwenBF16TensorSpec(name: sharedUpName,
                                   shardName: source.gateUpShardName,
                                   role: .sharedUp, rows: 1, columns: hiddenSize),
                QwenBF16TensorSpec(name: sharedDownName,
                                   shardName: source.downShardName,
                                   role: .sharedDown, rows: hiddenSize, columns: 1),
                QwenBF16TensorSpec(name: sharedOutputGateName,
                                   shardName: source.gateUpShardName,
                                   role: .sharedOutputGate, rows: 1, columns: hiddenSize),
            ], residencyBudget: UInt64((3 * hiddenSize + hiddenSize) * 2))
        let pairBytes = (2 * intermediateSize * hiddenSize
            + hiddenSize * intermediateSize) * MemoryLayout<UInt16>.stride
        let coordinator = try QwenBF16ExpertMappingCoordinator(
            source: source.handle,
            names: QwenBF16RoutedSourceNames(
                gateUpShardName: source.gateUpShardName,
                gateUpTensorName: gateUpName,
                downShardName: source.downShardName,
                downTensorName: downName),
            layer: 0, configuration: configuration, device: context.device,
            slotCount: 8, residencyBudget: UInt64(pairBytes * 8))
        let lease = try await coordinator.map(expertIDs: expertIDs)
        #expect(lease.experts.map(\.expertID) == expertIDs)

        let hiddenBuffer = try floatBuffer(hidden, device: context.device)
        let routeBuffer = try floatBuffer(routeWeights, device: context.device)
        let output = try floatBuffer([Float](repeating: 0, count: hiddenSize),
                                     device: context.device)
        let moe = try QwenMoE(context: context, configuration: configuration)
        let scratch = try moe.makeScratch()
        let command = try moe.submitExpertsBF16(
            hidden: hiddenBuffer, lease: lease, routingWeights: routeBuffer,
            sharedWeights: sharedWeights,
            sharedNames: QwenBF16SharedNames(
                gate: sharedGateName, up: sharedUpName,
                down: sharedDownName, outputGate: sharedOutputGateName),
            scratch: scratch, output: output)
        _ = await command.completed()
        #expect(command.status == .completed)
        #expect(command.error == nil)
        #expect(lease.snapshot().completed)
        #expect(lease.snapshot().succeeded == true)
        return readFloats(output, count: hiddenSize)
    }

    private func denseRoutedReference(
        hidden: [Float], expertIDs: [Int], routeWeights: [Float],
        gateUp: [UInt16], down: [UInt16], hiddenSize: Int, intermediateSize: Int
    ) -> [Float] {
        let weightByExpert = Dictionary(uniqueKeysWithValues:
            zip(expertIDs, routeWeights).map { ($0.0, $0.1) })
        var result = [Float](repeating: 0, count: hiddenSize)
        for expert in expertIDs.sorted() {
            let gateOffset = expert * 2 * intermediateSize * hiddenSize
            let downOffset = expert * hiddenSize * intermediateSize
            var activation = [Float](repeating: 0, count: intermediateSize)
            for row in 0..<intermediateSize {
                let gate = source64Dot(
                    vector: hidden, weights: gateUp,
                    rowOffset: gateOffset + row * hiddenSize, columns: hiddenSize)
                let up = source64Dot(
                    vector: hidden, weights: gateUp,
                    rowOffset: gateOffset + (intermediateSize + row) * hiddenSize,
                    columns: hiddenSize)
                activation[row] = gate / (1 + Darwin.expf(-gate)) * up
            }
            for row in 0..<hiddenSize {
                let projected = source64Dot(
                    vector: activation, weights: down,
                    rowOffset: downOffset + row * intermediateSize,
                    columns: intermediateSize)
                let contribution = projected * weightByExpert[expert, default: 0]
                result[row] += contribution
            }
        }
        return result
    }

    private func source64Dot(
        vector: [Float], weights: [UInt16], rowOffset: Int, columns: Int
    ) -> Float {
        var partial = [Float](repeating: 0, count: 64)
        for column in 0..<columns {
            let weight = Float(bitPattern: UInt32(weights[rowOffset + column]) << 16)
            partial[column & 63] = partial[column & 63].addingProduct(
                weight, vector[column])
        }
        var collapsed = [Float](repeating: 0, count: 16)
        for index in 0..<16 {
            let first = partial[index] + partial[index + 16]
            let second = first + partial[index + 32]
            collapsed[index] = second + partial[index + 48]
        }
        var groups = [Float](repeating: 0, count: 4)
        for index in 0..<4 {
            let base = index * 4
            let first = collapsed[base] + collapsed[base + 1]
            let second = first + collapsed[base + 2]
            groups[index] = second + collapsed[base + 3]
        }
        let first = groups[0] + groups[1]
        let second = first + groups[2]
        return second + groups[3]
    }

    private func denseBF16Words(count: Int, salt: Int) -> [UInt16] {
        (0..<count).map { index in
            let magnitude = UInt16(0x3c00 + ((index * 37 + salt) % 0x180))
            let sign: UInt16 = ((index + salt) & 1) == 0 ? 0 : 0x8000
            return magnitude | sign
        }
    }

    private func loadManifest() throws -> LargeMoEParityManifest {
        guard let url = Bundle.module.url(
            forResource: "manifest", withExtension: "json",
            subdirectory: "large-moe-parity") else {
            throw LargeMoEParityFixtureError.missingManifest
        }
        return try JSONDecoder().decode(
            LargeMoEParityManifest.self, from: Data(contentsOf: url))
    }

    private func loadOutput(named name: String, sha256 expectedHash: String) throws -> [Float] {
        guard let url = Bundle.module.url(
            forResource: name, withExtension: nil,
            subdirectory: "large-moe-parity") else {
            throw LargeMoEParityFixtureError.missingOutput(name)
        }
        let data = try Data(contentsOf: url)
        #expect(hash(data) == expectedHash)
        #expect(data.count == hiddenSize * MemoryLayout<UInt32>.stride)
        return stride(from: 0, to: data.count, by: MemoryLayout<UInt32>.stride).map { offset in
            let bits = UInt32(littleEndian: data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            })
            return Float(bitPattern: bits)
        }
    }

    private func expectFixtureHashes(
        _ fixture: LargeMoEParityWords, manifest: LargeMoEParityManifest
    ) {
        var hidden = [Float](repeating: 0, count: hiddenSize)
        hidden[0] = 1
        #expect(hash(floatBytes(hidden)) == manifest.inputSHA256)
        #expect(hash(wordBytes(fixture.gateUp)) == manifest.gateUpBF16SHA256)
        #expect(hash(wordBytes(fixture.down)) == manifest.downBF16SHA256)
        #expect(hash(wordBytes(fixture.sharedGate) + wordBytes(fixture.sharedUp)
                     + wordBytes(fixture.sharedDown) + wordBytes(fixture.sharedOutputGate))
                == manifest.sharedWeightsBF16SHA256)
        #expect(hash(floatBytes(routeWeightsInLeaseOrder))
                == manifest.routingWeightsFloat32SHA256)
    }

    private func floatBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
        try #require(values.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                              options: .storageModeShared)
        })
    }

    private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count))
    }

    private func wordBytes(_ words: [UInt16]) -> Data {
        var bytes = Data(capacity: words.count * MemoryLayout<UInt16>.stride)
        for word in words {
            var value = word.littleEndian
            withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    private func floatBytes(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * MemoryLayout<UInt32>.stride)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    private func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private struct LargeMoEParityWords {
    let gateUp: [UInt16]
    let down: [UInt16]
    let sharedGate: [UInt16]
    let sharedUp: [UInt16]
    let sharedDown: [UInt16]
    let sharedOutputGate: [UInt16]

    init() {
        let hidden = 512
        let intermediate = 512
        var gateUp = [UInt16](repeating: 0, count: 8 * 2 * intermediate * hidden)
        let expertGateUpCount = 2 * intermediate * hidden
        for expert in 0..<8 {
            let start = expert * expertGateUpCount
            gateUp[start] = 0x3f80
            gateUp[start + intermediate * hidden] = 0x3f80
        }
        var down = [UInt16](repeating: 0, count: 8 * hidden * intermediate)
        let expertDownCount = hidden * intermediate
        down[0] = 0x4c80
        down[expertDownCount] = 0xcc00
        down[2 * expertDownCount] = 0x3f80
        var sharedGate = [UInt16](repeating: 0, count: hidden)
        sharedGate[0] = 0x3f80
        let sharedUp = sharedGate
        var sharedDown = [UInt16](repeating: 0, count: hidden)
        sharedDown[0] = 0x3f80
        let sharedOutputGate = [UInt16](repeating: 0, count: hidden)

        self.gateUp = gateUp
        self.down = down
        self.sharedGate = sharedGate
        self.sharedUp = sharedUp
        self.sharedDown = sharedDown
        self.sharedOutputGate = sharedOutputGate
    }
}

private struct LargeMoEParityManifest: Decodable {
    struct Geometry: Decodable, Equatable {
        let hiddenSize: Int
        let routedIntermediateSize: Int
        let sharedIntermediateSize: Int
        let expertCount: Int
        let topK: Int
    }

    let formatVersion: Int
    let torchVersion: String
    let torchGit: String
    let torchThreads: Int
    let torchInteropThreads: Int
    let device: String
    let geometry: Geometry
    let leaseExpertIDs: [Int]
    let accumulationExpertIDs: [Int]
    let routeWeightFloat32Bits: [UInt32]
    let inputSHA256: String
    let gateUpBF16SHA256: String
    let downBF16SHA256: String
    let sharedWeightsBF16SHA256: String
    let routingWeightsFloat32SHA256: String
    let expectedOutputFile: String
    let expectedOutputSHA256: String
    let leaseRankControlFile: String
    let leaseRankControlSHA256: String
}

private enum LargeMoEParityFixtureError: Error {
    case missingManifest
    case missingOutput(String)
}
