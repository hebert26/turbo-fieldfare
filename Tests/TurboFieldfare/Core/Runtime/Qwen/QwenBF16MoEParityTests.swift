import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Small actual-source submissions that distinguish the pinned Qwen expert
/// accumulation order and its separately rounded route weighting.
@Suite(.serialized) struct QwenBF16MoEParityTests {
    @Test func actualBF16SubmissionAccumulatesByExpertIDAndAddsSharedBranchLast() async throws {
        // Lease/rank order is deliberately descending. Experts 0 and 1 cancel
        // at a large magnitude, so expert 2's smaller contribution survives
        // only when routed outputs are added in ascending expert-ID order.
        let expertIDs = [7, 6, 5, 4, 3, 2, 1, 0]
        let rankWeights: [Float] = [0.115, 0.115, 0.115, 0.115, 0.115, 0.025, 0.2, 0.2]
        var down = [UInt16](repeating: 0, count: 8)
        down[0] = 0x4c00 // +2^25
        down[1] = 0xcc00 // -2^25
        down[2] = 0x3f80 // +1

        let actual = try await runProductionMoE(
            expertIDs: expertIDs, rankWeights: rankWeights,
            routedDownWords: down, sharedDownWord: 0x3d80)
        let expected = pinnedFloatReference(
            expertIDs: expertIDs, rankWeights: rankWeights,
            routedDownWords: down, sharedDownWord: 0x3d80)
        let wrongRankOrder = pinnedFloatReference(
            expertIDs: expertIDs, rankWeights: rankWeights,
            routedDownWords: down, sharedDownWord: 0x3d80,
            accumulationOrder: expertIDs)

        expectFrozenClose(actual, expected)
        #expect(abs(expected - wrongRankOrder) > 0.01,
                "fixture must reject accumulation in selected-rank order")
        #expect(abs(expected - pinnedFloatReference(
            expertIDs: expertIDs, rankWeights: rankWeights,
            routedDownWords: down, sharedDownWord: 0)) > 0.01,
                "shared branch must contribute after routed experts")
    }

    @Test func actualBF16SubmissionRoundsRouteProductBeforeAccumulation() async throws {
        // Opposite routed projections use the same unequal-to-one route weight.
        // Their separately rounded products cancel exactly. An FMA in the
        // second weighted add retains the product's discarded low bits.
        let expertIDs = Array(0..<8)
        let routeWeight: Float = 0.3
        let rankWeights: [Float] = [routeWeight, routeWeight, 0.4, 0, 0, 0, 0, 0]
        var down = [UInt16](repeating: 0, count: 8)
        down[0] = 0xcc00 // -2^25
        down[1] = 0x4c00 // +2^25

        let actual = try await runProductionMoE(
            expertIDs: expertIDs, rankWeights: rankWeights,
            routedDownWords: down, sharedDownWord: 0)
        let separatelyRounded = pinnedFloatReference(
            expertIDs: expertIDs, rankWeights: rankWeights,
            routedDownWords: down, sharedDownWord: 0)
        let fusedControl = pinnedFloatReference(
            expertIDs: expertIDs, rankWeights: rankWeights,
            routedDownWords: down, sharedDownWord: 0,
            useFusedRouteWeighting: true)

        expectFrozenClose(actual, separatelyRounded)
        #expect(separatelyRounded == 0)
        #expect(abs(fusedControl - separatelyRounded) > 1e-4,
                "fixture must distinguish separate multiply/add from fused weighting")
    }

}

private func runProductionMoE(
    expertIDs: [Int], rankWeights: [Float],
    routedDownWords: [UInt16], sharedDownWord: UInt16
) async throws -> [Float] {
    let fixture = try QwenBF16ExpertCacheSourceFixture.make(
        firstShard: [
            QwenBF16ExpertCacheLiteralTensor(
                name: "parity.gate_up", shape: [8, 2, 1],
                words: [UInt16](repeating: 0x3f80, count: 16)),
            QwenBF16ExpertCacheLiteralTensor(
                name: "parity.shared_gate", shape: [1, 1], words: [0x3f80]),
            QwenBF16ExpertCacheLiteralTensor(
                name: "parity.shared_up", shape: [1, 1], words: [0x3f80]),
            QwenBF16ExpertCacheLiteralTensor(
                name: "parity.shared_output_gate", shape: [1, 1], words: [0]),
        ],
        secondShard: [
            QwenBF16ExpertCacheLiteralTensor(
                name: "parity.down", shape: [8, 1, 1], words: routedDownWords),
            QwenBF16ExpertCacheLiteralTensor(
                name: "parity.shared_down", shape: [1, 1], words: [sharedDownWord]),
        ])
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let context = try MetalContext()
    let configuration = try QwenMoEConfiguration(
        hiddenSize: 1, expertCount: 8, topK: 8,
        routedIntermediateSize: 1, sharedIntermediateSize: 1)
    let moe = try QwenMoE(context: context, configuration: configuration)
    let sharedWeights = try QwenBF16Weights(
        context: context, source: fixture.handle,
        specifications: [
            QwenBF16TensorSpec(name: "parity.shared_gate", shardName: fixture.gateUpShardName,
                               role: .sharedGate, rows: 1, columns: 1),
            QwenBF16TensorSpec(name: "parity.shared_up", shardName: fixture.gateUpShardName,
                               role: .sharedUp, rows: 1, columns: 1),
            QwenBF16TensorSpec(name: "parity.shared_down", shardName: fixture.downShardName,
                               role: .sharedDown, rows: 1, columns: 1),
            QwenBF16TensorSpec(name: "parity.shared_output_gate",
                               shardName: fixture.gateUpShardName,
                               role: .sharedOutputGate, rows: 1, columns: 1),
        ], residencyBudget: 8)
    let coordinator = try QwenBF16ExpertMappingCoordinator(
        source: fixture.handle,
        names: QwenBF16RoutedSourceNames(
            gateUpShardName: fixture.gateUpShardName,
            gateUpTensorName: "parity.gate_up",
            downShardName: fixture.downShardName,
            downTensorName: "parity.down"),
        layer: 0, configuration: configuration, device: context.device,
        slotCount: 8, residencyBudget: 48)
    let lease = try await coordinator.map(expertIDs: expertIDs)
    #expect(lease.experts.map(\.expertID) == expertIDs)

    let sharedNames = QwenBF16SharedNames(
        gate: "parity.shared_gate", up: "parity.shared_up",
        down: "parity.shared_down", outputGate: "parity.shared_output_gate")
    let hidden = try parityBuffer([Float(1)], device: context.device)
    let routeBuffer = try parityBuffer(rankWeights, device: context.device)
    let output = try parityBuffer([Float(0)], device: context.device)
    let scratch = try moe.makeScratch()
    let command = try moe.submitExpertsBF16(
        hidden: hidden, lease: lease, routingWeights: routeBuffer,
        sharedWeights: sharedWeights, sharedNames: sharedNames,
        scratch: scratch, output: output)
    _ = await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    #expect(lease.snapshot().completed)
    #expect(lease.snapshot().succeeded == true)
    return parityRead(output, count: 1)
}

private func pinnedFloatReference(
    expertIDs: [Int], rankWeights: [Float], routedDownWords: [UInt16],
    sharedDownWord: UInt16, accumulationOrder: [Int]? = nil,
    useFusedRouteWeighting: Bool = false
) -> Float {
    let hidden: Float = 1
    let gate = parityMultiply(parityBF16(0x3f80), hidden)
    let up = parityMultiply(parityBF16(0x3f80), hidden)
    let activation = parityMultiply(paritySiLU(gate), up)
    let weightByExpert = Dictionary(uniqueKeysWithValues:
        zip(expertIDs, rankWeights).map { ($0.0, $0.1) })
    var routed: Float = 0
    for expert in accumulationOrder ?? expertIDs.sorted() {
        let projected = parityMultiply(activation, parityBF16(routedDownWords[expert]))
        let weight = weightByExpert[expert, default: 0]
        if useFusedRouteWeighting {
            routed = routed.addingProduct(projected, weight)
        } else {
            // Double exactly represents the product of two FP32 operands;
            // converting back pins the separate FP32 product rounding.
            let contribution = parityMultiply(projected, weight)
            routed = parityAdd(routed, contribution)
        }
    }

    let sharedActivation = parityMultiply(paritySiLU(gate), up)
    let sharedProjection = parityMultiply(sharedActivation, parityBF16(sharedDownWord))
    let outputGate = parityMultiply(parityBF16(0), hidden)
    let sharedScale = 1 / (1 + Float(Foundation.exp(Double(-outputGate))))
    let sharedContribution = parityMultiply(sharedProjection, sharedScale)
    return parityAdd(routed, sharedContribution)
}

private func expectFrozenClose(_ actual: [Float], _ expected: Float) {
    #expect(actual.count == 1)
    #expect(actual[0].isFinite)
    let limit = 1e-7 + 1e-6 * abs(expected)
    #expect(abs(actual[0] - expected) <= limit,
            "actual \(actual[0]) differs from pinned FP32 reference \(expected), limit \(limit)")
}

private func parityBF16(_ bits: UInt16) -> Float {
    Float(bitPattern: UInt32(bits) << 16)
}

private func paritySiLU(_ value: Float) -> Float {
    value / (1 + Float(Foundation.exp(Double(-value))))
}

private func parityMultiply(_ lhs: Float, _ rhs: Float) -> Float {
    Float(Double(lhs) * Double(rhs))
}

private func parityAdd(_ lhs: Float, _ rhs: Float) -> Float {
    Float(Double(lhs) + Double(rhs))
}

private func parityBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
    try #require(values.withUnsafeBytes { bytes in
        device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                          options: .storageModeShared)
    })
}

private func parityRead(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count))
}
