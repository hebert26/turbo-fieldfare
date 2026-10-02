import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Production-path completion coverage missing from the timing discriminator:
/// a lease encode failure and one settled grouped-MoE command. The source
/// bytes are synthetic literals and never come from an original shard.
@Suite(.serialized)
struct QwenCompletionResumptionTimingSupplementTests {
    @Test func leaseEncodeFailureEmitsOneUnawaitedCompletionRow() async throws {
        let fixture = try makeTimingFixture()
        defer { fixture.source.remove() }
        let coordinator = try makeCoordinator(fixture)
        let lease = try await coordinator.map(expertIDs: [0])
        let (capture, location) = try makeCollector()
        var capturedCommand: MTLCommandBuffer?
        var sawFailure = false
        do {
            try await QwenProductionTimingMeasurement.$location.withValue(location) {
                let timing = try #require(QwenProductionTimingMeasurement.command(.groupedMoE))
                _ = try lease.submit(on: fixture.context.queue, timing: timing) { command in
                    capturedCommand = command
                    throw TimingSupplementEncodeFailure.injected
                }
            }
            Issue.record("the injected lease encoder failure must be thrown")
        } catch is TimingSupplementEncodeFailure {
            sawFailure = true
        }
        #expect(sawFailure)
        let command = try #require(capturedCommand)
        _ = await command.completed()
        capture.finish(status: 0)
        let rows = try drainTimingRows(capture)
        let commands = rows.filter { $0[0] == 161 }
        let completions = rows.filter { $0[0] == 170 }
        #expect(commands.count == 1)
        #expect(completions.count == 1,
                "an encode failure must settle exactly one completion observation")
        let completion = try #require(completions.first)
        #expect(completion[1] == commands[0][1])
        #expect(completion[3] == 0, "failed submission has no after-await sample")
        #expect((completion[4] & 4) == 4, "failed submission must be marked unawaited")
        let driver = try #require(rows.first { $0[0] == 171 })
        #expect(Array(driver[2...5]) == [0, 0, 0, 0],
                "later completion must not invent an after-await driver sample")
    }

    @Test func groupedMoECompletionEmitsOneResumptionRow() async throws {
        let fixture = try makeTimingFixture()
        defer { fixture.source.remove() }
        let coordinator = try makeCoordinator(fixture)
        let lease = try await coordinator.map(expertIDs: Array(0..<8))
        let moe = try QwenMoE(context: fixture.context, configuration: fixture.configuration)
        let shared = try makeSharedWeights(fixture)
        let scratch = try moe.makeScratch()
        let hidden = try sharedBuffer([0.25, 0.5, 0.75], device: fixture.context.device)
        let routes = try sharedBuffer([Float](repeating: 0.125, count: 8), device: fixture.context.device)
        let output = try sharedBuffer([Float](repeating: 0, count: 3), device: fixture.context.device)
        let work = Array(0..<8).map {
            QwenBF16GroupedExpertWork(expertID: $0, tokenIndex: 0, routeRank: $0)
        }
        let (capture, location) = try makeCollector()
        try await QwenProductionTimingMeasurement.$location.withValue(location) {
            try await moe.submitGroupedExpertsBF16(
                hiddenRows: hidden, routingExpertIDs: Array(0..<8), routingWeights: routes,
                outputRows: output, tokenCount: 1, lease: lease, work: work,
                sharedWeights: shared, scratch: scratch, initializeOutput: true)
        }
        capture.finish(status: 0)
        let rows = try drainTimingRows(capture)
        let commands = rows.filter { $0[0] == 161 }
        let completions = rows.filter { $0[0] == 170 }
        #expect(commands.count == 1)
        #expect(completions.count == 1,
                "one grouped command must emit one completion observation")
        let completion = try #require(completions.first)
        #expect(completion[1] == commands[0][1])
        let wait = try #require(rows.first {
            $0[0] == 162 && $0[1] == commands[0][1]
        })
        #expect(completion[3] == wait[5])
        #expect((completion[4] & 4) == 0)
        if (completion[4] & 2) != 0 {
            #expect(completion[2] > 0 && completion[2] <= completion[3])
        } else if (completion[4] & 8) != 0 {
            #expect(completion[2] > completion[3])
        } else {
            #expect(completion[2] == 0 && completion[4] == 0)
        }
    }

    private struct Fixture {
        let source: QwenBF16ExpertCacheSourceFixture
        let context: MetalContext
        let configuration: QwenMoEConfiguration
    }

    private func makeTimingFixture() throws -> Fixture {
        let gateName = "timing.gate_up"
        let downName = "timing.down"
        let sharedGateName = "timing.shared_gate"
        let sharedUpName = "timing.shared_up"
        let sharedDownName = "timing.shared_down"
        let sharedOutputName = "timing.shared_output_gate"
        let source = try QwenBF16ExpertCacheSourceFixture.make(
            firstShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: gateName, shape: [9, 4, 3],
                    words: [UInt16](repeating: 0x3f80, count: 9 * 4 * 3)),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedGateName, shape: [2, 3],
                    words: [UInt16](repeating: 0x3f80, count: 6)),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedUpName, shape: [2, 3],
                    words: [UInt16](repeating: 0x3f80, count: 6)),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedOutputName, shape: [1, 3],
                    words: [UInt16](repeating: 0x3f80, count: 3)),
            ],
            secondShard: [
                QwenBF16ExpertCacheLiteralTensor(
                    name: downName, shape: [9, 3, 2],
                    words: [UInt16](repeating: 0x3f80, count: 9 * 3 * 2)),
                QwenBF16ExpertCacheLiteralTensor(
                    name: sharedDownName, shape: [3, 2],
                    words: [UInt16](repeating: 0x3f80, count: 6)),
            ])
        let context = try MetalContext()
        let configuration = try QwenMoEConfiguration(
            hiddenSize: 3, expertCount: 9, topK: 8,
            routedIntermediateSize: 2, sharedIntermediateSize: 2)
        return Fixture(source: source, context: context, configuration: configuration)
    }

    private func makeCoordinator(_ fixture: Fixture) throws
        -> QwenBF16ExpertMappingCoordinator {
        try QwenBF16ExpertMappingCoordinator(
            source: fixture.source.handle,
            names: QwenBF16RoutedSourceNames(
                gateUpShardName: fixture.source.gateUpShardName,
                gateUpTensorName: "timing.gate_up",
                downShardName: fixture.source.downShardName,
                downTensorName: "timing.down"),
            layer: 0, configuration: fixture.configuration, device: fixture.context.device,
            slotCount: 8, residencyBudget: 8 * (2 * 2 * 2 * 3 + 2 * 3 * 2))
    }

    private func makeSharedWeights(_ fixture: Fixture) throws -> QwenBF16Weights {
        try QwenBF16Weights(
            context: fixture.context, source: fixture.source.handle,
            specifications: [
                QwenBF16TensorSpec(name: "timing.shared_gate",
                                   shardName: fixture.source.gateUpShardName,
                                   role: .sharedGate, rows: 2, columns: 3),
                QwenBF16TensorSpec(name: "timing.shared_up",
                                   shardName: fixture.source.gateUpShardName,
                                   role: .sharedUp, rows: 2, columns: 3),
                QwenBF16TensorSpec(name: "timing.shared_down",
                                   shardName: fixture.source.downShardName,
                                   role: .sharedDown, rows: 3, columns: 2),
                QwenBF16TensorSpec(name: "timing.shared_output_gate",
                                   shardName: fixture.source.gateUpShardName,
                                   role: .sharedOutputGate, rows: 1, columns: 3),
            ], residencyBudget: 42)
    }

    private func makeCollector() throws
        -> (RuntimeMeasurementCapture, QwenProductionLocation) {
        let capture = RuntimeMeasurementCapture(qwenCacheCaptureMode: .prefillAndDecode)
        try #require(capture.beginQwenCacheMaps(
            layerCount: 1, expertCount: 9, slotCount: 8, pairBytes: 96))
        try #require(capture.beginQwenProductionTiming())
        capture.setQwenCacheMapPhase(.decode)
        let location = try #require(capture.qwenProductionLocation(
            position: 0, tokenCount: 1, forward: true))
        return (capture, location.atLayer(0))
    }
}

private enum TimingSupplementEncodeFailure: Error {
    case injected
}

private func sharedBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
    try values.withUnsafeBytes { bytes in
        try #require(device.makeBuffer(
            bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared))
    }
}

private func drainTimingRows(_ capture: RuntimeMeasurementCapture) throws -> [[UInt64]] {
    struct Batch: Decodable { let records: [[UInt64]] }
    var rows: [[UInt64]] = []
    while let batch = capture.drainJSONBatch(
        maximumBytes: RuntimeMeasurementCapture.maximumJSONBatchBytes) {
        rows += try JSONDecoder().decode(Batch.self, from: batch.data).records.filter {
            $0.count == 6 && (160...171).contains($0[0])
        }
    }
    return rows
}
