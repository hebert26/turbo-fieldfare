import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized)
struct QwenGroupedLinearPrototypeTests {
    @Test func eightRowsMatchSinglesAtInitialAndCachedPositionsExactly() async throws {
        let fixture = try GroupedLinearFixture()
        for start in [0, 3] {
            let baseline = try fixture.makeLane(grouped: false)
            let candidate = try fixture.makeLane(grouped: true)
            try await fixture.warm(baseline, count: start)
            try await fixture.warm(candidate, count: start)
            let expected = try await fixture.singles(baseline, start: start, count: 8)
            let actual = try await fixture.group(candidate, start: start, count: 8)
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
            try await fixture.expectSameState(baseline, candidate)
            #expect(try candidate.owner.committedPosition(layer: 0) == start + 8)
        }
    }

    @Test func variedGroupSizesMatchSinglesWithCPUAndGPUPreparation() async throws {
        let fixture = try GroupedLinearFixture()
        for count in [2, 7, 16] {
            for start in [0, 3] {
                for gpu in [false, true] {
                    let baseline = try fixture.makeLane(grouped: false)
                    let candidate = try fixture.makeLane(grouped: true)
                    try await fixture.warm(baseline, count: start)
                    try await fixture.warm(candidate, count: start)
                    let expected = try await fixture.singles(baseline, start: start,
                        count: count, useGPUPreparation: gpu)
                    let actual = try await fixture.group(candidate, start: start,
                        count: count, useGPUPreparation: gpu)
                    #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
                    try await fixture.expectSameState(baseline, candidate)
                    #expect(try candidate.owner.committedPosition(layer: 0) == start + count)
                }
            }
        }
    }

    @Test func finiteProjectionOverflowRejectsAndRestoresWholeGroup() async throws {
        let fixture = try GroupedLinearFixture()
        for start in [0, 3] {
            for gpu in [false, true] {
                let lane = try fixture.makeLane(grouped: true)
                try await fixture.warm(lane, count: start)
                let before = try await lane.step.snapshot()
                var rejected = false
                do {
                    _ = try await fixture.group(lane, start: start, count: 8,
                        useGPUPreparation: gpu, overflowLastRow: true)
                } catch QwenTextRunnerError.invalidState(let detail) {
                    #expect(detail == "BF16 linear nonfinite projections")
                    rejected = true
                }
                #expect(rejected)
                GroupedLinearFixture.expectBitsEqual(before, try await lane.step.snapshot())
                let retried = try await fixture.group(lane, start: start, count: 8,
                    useGPUPreparation: gpu)
                let baseline = try fixture.makeLane(grouped: false)
                try await fixture.warm(baseline, count: start)
                let expected = try await fixture.singles(baseline, start: start,
                    count: 8, useGPUPreparation: gpu)
                #expect(retried.map(\.bitPattern) == expected.map(\.bitPattern))
                try await fixture.expectSameState(baseline, lane)
            }
        }
    }

    @Test func injectedFailuresRestoreWholeGroupIncludingInitialToken() async throws {
        let fixture = try GroupedLinearFixture()
        for start in [0, 3] {
            for target in 0..<3 {
                let lane = try fixture.makeLane(grouped: true)
                try await fixture.warm(lane, count: start)
                let before = try await lane.step.snapshot()
                let hooks = QwenBF16LinearPreparationHooks { checkpoint in
                    if GroupedLinearFixture.matches(checkpoint, target: target) {
                        throw GroupedLinearFailure.injected
                    }
                }
                var rejected = false
                do { _ = try await fixture.group(lane, start: start, count: 8, hooks: hooks) }
                catch GroupedLinearFailure.injected { rejected = true }
                #expect(rejected)
                let after = try await lane.step.snapshot()
                GroupedLinearFixture.expectBitsEqual(before, after)
                // A retry proves the deferred reservation was released.
                let retried = try await fixture.group(lane, start: start, count: 8)
                #expect(retried.count == 8 * 2048)
            }
        }
    }

    @Test func actualCancellationDrainsAndRestoresBeforeAndAfterSubmissionAndCommitGate() async throws {
        let fixture = try GroupedLinearFixture()
        for start in [0, 3] {
            for target in 0..<3 {
                let lane = try fixture.makeLane(grouped: true)
                try await fixture.warm(lane, count: start)
                let before = try await lane.step.snapshot()
                let gate = GroupedLinearGate()
                let hooks = QwenBF16LinearPreparationHooks { checkpoint in
                    if GroupedLinearFixture.matches(checkpoint, target: target) { await gate.hold() }
                }
                let operation = Task {
                    try await fixture.group(lane, start: start, count: 8, hooks: hooks)
                }
                await gate.waitForEntry()
                operation.cancel()
                await gate.release()
                var cancelled = false
                do { _ = try await operation.value } catch is CancellationError { cancelled = true }
                #expect(cancelled)
                let after = try await lane.step.snapshot()
                GroupedLinearFixture.expectBitsEqual(before, after)
                _ = try await fixture.group(lane, start: start, count: 8)
            }
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_GROUPED_LINEAR_TIMING"] == "1"))
    func warmedWholeEightTokenBlockABBA() async throws {
        let fixture = try GroupedLinearFixture()
        let serial = try fixture.makeLane(grouped: false)
        let grouped = try fixture.makeLane(grouped: true)
        try await fixture.warm(serial, count: 3)
        try await fixture.warm(grouped, count: 3)
        let baseline = try serial.owner.retainCheckpoint()
        let candidate = try grouped.owner.retainCheckpoint()
        for pair in 0..<16 {
            var results: [[Float]] = []
            for mode in (pair.isMultiple(of: 2) ? [false, true] : [true, false]) {
                let lane = mode ? grouped : serial
                try lane.owner.restore(mode ? candidate : baseline)
                let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let output: [Float]
                if mode { output = try await fixture.group(lane, start: 3, count: 8) }
                else { output = try await fixture.singles(lane, start: 3, count: 8) }
                let elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
                results.append(output)
                if pair >= 6 {
                    print("QWEN_GROUPED_LINEAR_BLOCK pair=\(pair - 6) grouped=\(mode) tokens=8 wall_ns=\(elapsed)")
                }
            }
            #expect(results[0].map(\.bitPattern) == results[1].map(\.bitPattern))
            try await fixture.expectSameState(serial, grouped)
        }
    }
}

private enum GroupedLinearFailure: Error { case injected }

private actor GroupedLinearGate {
    private var entered = false
    private var observers: [CheckedContinuation<Void, Never>] = []
    private var continuation: CheckedContinuation<Void, Never>?
    func hold() async {
        await withCheckedContinuation { continuation in
            self.continuation = continuation
            entered = true
            observers.forEach { $0.resume() }
            observers.removeAll()
        }
    }
    func waitForEntry() async {
        if entered { return }
        await withCheckedContinuation { observers.append($0) }
    }
    func release() { continuation?.resume(); continuation = nil }
}

// Immutable resident tensors shared by serial test calls. All mutable attention
// state belongs to each Lane's actor and synchronized state owner.
private final class GroupedLinearFixture: @unchecked Sendable {
    struct Lane: Sendable {
        let step: QwenBF16LinearStep
        let owner: QwenLinearAttentionState
    }
    let context: MetalContext
    let configuration: QwenGatedDeltaNetConfiguration
    let source: QwenBF16SyntheticSource
    let weights: QwenBF16Weights
    let names = QwenBF16LinearNames(qkv: "qkv", z: "z", b: "b", a: "a", output: "out")

    init() throws {
        context = try MetalContext()
        configuration = try .official()
        let library = try MetalContext.privateLibrary(device: context.device,
            module: "qwen_linear_attention", mathMode: .safe,
            mathFloatingPointFunctions: .precise, includeQwenSourceMath: true)
        let function = try #require(library.makeFunction(name: "qwen_source_linear_recurrence_cached_128_grouped"))
        let pipeline = try context.device.makeComputePipelineState(function: function)
        try #require(pipeline.maxTotalThreadsPerThreadgroup >= 128,
                     "Discriminator must exercise grouped cached128, not its serial fallback")
        let shapes = [("qkv", 8192, 2048), ("z", 4096, 2048),
                      ("b", 32, 2048), ("a", 32, 2048), ("out", 2048, 4096)]
        let palette: [UInt16] = [0x3b80, 0xbb80, 0x3c00, 0xbc00, 0]
        let tensors = shapes.enumerated().map { matrix, shape in
            QwenBF16LiteralTensor(name: shape.0, rows: shape.1, columns: shape.2,
                bits: (0..<(shape.1 * shape.2)).map {
                    palette[($0 * 7 + $0 / shape.2 + matrix) % palette.count]
                })
        }
        let created = try QwenBF16SyntheticSource.make(tensors: tensors)
        source = created
        do {
            weights = try QwenBF16Weights(context: context, source: created.handle,
                specifications: shapes.map {
                    QwenBF16TensorSpec(name: $0.0, shardName: created.shardName,
                        role: .dense, rows: $0.1, columns: $0.2)
                }, residencyBudget: shapes.reduce(UInt64(0)) { $0 + UInt64($1.1 * $1.2 * 2) })
        } catch { created.remove(); throw error }
    }
    deinit { source.remove() }

    func makeLane(grouped: Bool) throws -> Lane {
        let geometry = try QwenLinearAttentionGeometry(convolutionWidth: 4,
            convolutionChannelCount: 8192, valueHeadCount: 32,
            keyHeadDimension: 128, valueHeadDimension: 128)
        let owner = try QwenLinearAttentionState(device: context.device,
            linearAttentionLayerMask: [1], geometry: geometry)
        let vectors = QwenBF16LinearVectors(
            convolution: (0..<(8192 * 4)).map { Float($0 % 7 - 3) / 16 },
            normalization: (0..<128).map { Float($0 % 7 + 1) / 4 },
            aLog: (0..<32).map { -Float($0 % 5) / 8 },
            timeStepBias: (0..<32).map { Float($0 % 7 - 3) / 16 })
        let step = try QwenBF16LinearStep(context: context, weights: weights, names: names,
            configuration: configuration, vectors: vectors, layer: 0, state: owner,
            useGroupedCachedRecurrence: grouped)
        return Lane(step: step, owner: owner)
    }
    func row(_ index: Int) -> [Float] {
        (0..<2048).map { Float(($0 * 11 + index * 13) % 127 - 63) / 64 }
    }
    func warm(_ lane: Lane, count: Int) async throws {
        _ = try await singles(lane, start: 0, count: count)
    }
    func singles(_ lane: Lane, start: Int, count: Int,
                 useGPUPreparation: Bool = true) async throws -> [Float] {
        var result: [Float] = []
        for token in start..<(start + count) {
            result += try await lane.step.append(normalizedHidden: row(token), tokenCount: 1,
                                                  useGPUPreparation: useGPUPreparation)
        }
        return result
    }
    // Test-owned group transaction, deliberately not production runner wiring.
    // Initial position executes the existing special first-token path alone.
    func group(_ lane: Lane, start: Int, count: Int,
               hooks: QwenBF16LinearPreparationHooks = .none,
               useGPUPreparation: Bool = true,
               overflowLastRow: Bool = false) async throws -> [Float] {
        let checkpoint = try lane.owner.retainCheckpoint()
        do {
            var first = start
            var result: [Float] = []
            if start == 0 {
                result += try await lane.step.append(normalizedHidden: row(0), tokenCount: 1,
                                                     useGPUPreparation: useGPUPreparation)
                first += 1
            }
            if first < start + count {
                var input = (first..<(start + count)).flatMap { row($0) }
                if overflowLastRow {
                    // Match signs of the first QKV row so finite products add
                    // beyond FP32 range. This reaches preparation rejection,
                    // rather than the normalized-input finite-value guard.
                    let offset = input.count - 2048
                    for column in 0..<2048 {
                        let paletteIndex = (column * 7) % 5
                        let negative = paletteIndex == 1 || paletteIndex == 3
                        input[offset + column] = negative
                            ? -Float.greatestFiniteMagnitude : Float.greatestFiniteMagnitude
                    }
                }
                result += try await lane.step.append(normalizedHidden: input,
                    tokenCount: start + count - first, useGPUPreparation: useGPUPreparation,
                    preparationHooks: hooks)
            }
            return result
        } catch {
            await lane.owner.waitUntilIdle()
            try lane.owner.restore(checkpoint)
            throw error
        }
    }
    static func matches(_ checkpoint: QwenBF16LinearHooks.Checkpoint, target: Int) -> Bool {
        switch checkpoint {
        case .beforeSubmission: return target == 0
        case .afterSubmission: return target == 1
        case .beforeCommit: return target == 2
        }
    }
    func expectSameState(_ a: Lane, _ b: Lane) async throws {
        let first = try await a.step.snapshot()
        let second = try await b.step.snapshot()
        Self.expectBitsEqual(first, second)
    }
    static func expectBitsEqual(_ a: QwenBF16LinearStepSnapshot, _ b: QwenBF16LinearStepSnapshot) {
        #expect(a.position == b.position)
        #expect(a.state.convolutionHistory.map(\.bitPattern) == b.state.convolutionHistory.map(\.bitPattern))
        #expect(a.state.recurrentMatrix.map(\.bitPattern) == b.state.recurrentMatrix.map(\.bitPattern))
    }
}
