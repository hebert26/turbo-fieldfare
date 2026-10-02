import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized)
struct QwenSourceGatedNormLanesTests {
    @Test func actual32Heads128ValuesMatchSerialIncludingFiniteExtremesAndSignedZero() async throws {
        try await NormLaneFixture.compareNorm(dimension: 128, heads: 32, tokens: 1, source: true)
    }

    @Test func multipleTokenHeadsMatchSerial() async throws {
        try await NormLaneFixture.compareNorm(dimension: 128, heads: 32, tokens: 2, source: true)
    }

    @Test func otherDimensionsAndPackedArithmeticRetainOriginalKernel() async throws {
        for dimension in [3, 64, 129, 192] {
            try await NormLaneFixture.compareNorm(dimension: dimension, heads: 3, tokens: 2, source: true)
        }
        try await NormLaneFixture.compareNorm(dimension: 128, heads: 32, tokens: 1, source: false)
    }

    /// Opt-in diagnostic. GPU timestamps cover recurrence + norm + BF16 output
    /// together. There is no claimed whole-stage result from a norm-only timer.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_GATED_NORM_STAGE_TIMING"] == "1"))
    func warmedResidentInterleavedRecurrenceNormOutputComparison() async throws {
        let fixture = try NormLaneResidentStage()
        for _ in 0..<8 {
            let serial = try await fixture.run(mode: .serial)
            let lanes = try await fixture.run(mode: .lanes128)
            #expect(serial.outputBits == lanes.outputBits)
            #expect(serial.normBits == lanes.normBits)
            #expect(serial.stateBits == lanes.stateBits)
        }
        for pair in 0..<20 {
            let order: [QwenSourceGatedNormMode] = pair.isMultiple(of: 2)
                ? [.serial, .lanes128] : [.lanes128, .serial]
            let first = try await fixture.run(mode: order[0])
            let second = try await fixture.run(mode: order[1])
            #expect(first.outputBits == second.outputBits)
            #expect(first.normBits == second.normBits)
            #expect(first.stateBits == second.stateBits)
            for (ordinal, result) in [first, second].enumerated() {
                let row: [String: Any] = [
                    "stage": "resident_recurrence_norm_output", "pair": pair, "order": ordinal,
                    "mode": result.mode == .serial ? "serial" : "lanes128",
                    "heads": 32, "value_dimension": 128, "output_rows": 2048,
                    "commands": 1, "gpu_start_seconds": result.gpuStart,
                    "gpu_end_seconds": result.gpuEnd,
                    "gpu_interval_seconds": result.gpuEnd - result.gpuStart,
                    "cpu_submit_to_resume_nanoseconds": result.cpuNanos,
                    "cpu_encode_through_settlement_nanoseconds": result.encodeThroughSettlementNanos,
                    "clocks": "separate; no CPU/GPU absolute-time translation",
                ]
                let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
                print("QWEN_GATED_NORM_STAGE " + String(decoding: data, as: UTF8.self))
            }
        }
    }
}

private enum NormLaneFixture {
    static func compareNorm(dimension: Int, heads: Int, tokens: Int, source: Bool) async throws {
        let context = try MetalContext()
        let configuration = try QwenGatedDeltaNetConfiguration(
            hiddenSize: heads * dimension, keyHeadCount: heads, valueHeadCount: heads,
            keyHeadDimension: 2, valueHeadDimension: dimension)
        let serial = try QwenGatedDeltaNet(context: context, configuration: configuration,
            useOfficialSourceMath: source)
        let lanes = try QwenGatedDeltaNet(context: context, configuration: configuration,
            useOfficialSourceMath: source, sourceGatedNormMode: .lanes128)
        let count = tokens * heads * dimension
        let input: [Float] = (0..<count).map { index in
            let sign: Float = index.isMultiple(of: 2) ? 1 : -1
            switch (index / dimension) % 4 {
            case 0: return index.isMultiple(of: 2) ? Float.zero : -Float.zero
            case 1: return sign * 1e-30
            case 2: return sign * 1e18
            default: return sign * Float(index % 29 + 1) / 32
            }
        }
        let gate: [Float] = (0..<count).map { index in
            if (index / dimension) % 4 == 0 { return Float.zero }
            return [Float(64), -64, Float.zero, -Float.zero, 0.125, -0.5,
                    Float.leastNormalMagnitude, -Float.leastNormalMagnitude][index % 8]
        }
        let weights = (0..<dimension).map { Float($0 % 7 + 1) / 4 }
        let inputBuffer = try buffer(input, device: context.device)
        let gateBuffer = try buffer(gate, device: context.device)
        let weightBuffer = try buffer(weights, device: context.device)
        var outputs: [[Float]] = []
        for runtime in [serial, lanes] {
            // Extra initialized elements check for writes beyond the declared region.
            let output = try buffer([Float](repeating: 123.5, count: count + 16), device: context.device)
            let command = try #require(context.queue.makeCommandBuffer())
            try runtime.encodeGatedRMSNorm(commandBuffer: command, input: inputBuffer,
                gate: gateBuffer, weights: weightBuffer, output: output, tokenCount: tokens)
            command.commit()
            await command.completed()
            try checkCommandBufferError(command)
            let values = read(output, count: count + 16)
            #expect(values.suffix(16).allSatisfy { $0 == 123.5 })
            #expect(values.prefix(count).allSatisfy { $0.isFinite })
            outputs.append(Array(values.prefix(count)))
        }
        #expect(outputs[0].map(\.bitPattern) == outputs[1].map(\.bitPattern))
        if source {
            #expect(outputs[0].contains { $0.bitPattern == (-Float.zero).bitPattern })
        }
    }

    static func buffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            try #require(device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                           options: .storageModeShared))
        }
    }

    static func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    static func write(_ values: [Float], to buffer: MTLBuffer) {
        values.withUnsafeBytes { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
    }
}

private final class NormLaneResidentStage {
    struct Result {
        let mode: QwenSourceGatedNormMode
        let outputBits: [UInt32]
        let normBits: [UInt32]
        let stateBits: [UInt32]
        let gpuStart: Double
        let gpuEnd: Double
        let cpuNanos: UInt64
        let encodeThroughSettlementNanos: UInt64
    }
    let context: MetalContext
    let serial: QwenGatedDeltaNet
    let lanes: QwenGatedDeltaNet
    let owner: QwenLinearAttentionState
    let source: QwenBF16SyntheticSource
    let weights: QwenBF16Weights
    let initialState: [Float]
    let query: MTLBuffer, key: MTLBuffer, value: MTLBuffer, decay: MTLBuffer, beta: MTLBuffer
    let gate: MTLBuffer, normWeights: MTLBuffer, recurrent: MTLBuffer, norm: MTLBuffer, output: MTLBuffer

    init() throws {
        let context = try MetalContext()
        self.context = context
        // Require the actual lane path for timing, not silently time its fallback.
        let library = try MetalContext.privateLibrary(device: context.device,
            module: "qwen_linear_attention", mathMode: .safe,
            mathFloatingPointFunctions: .precise, includeQwenSourceMath: true)
        let function = try #require(library.makeFunction(name: "qwen_source_linear_gated_rmsnorm_128_lanes"))
        let lanePipeline = try context.device.makeComputePipelineState(function: function)
        try #require(lanePipeline.maxTotalThreadsPerThreadgroup >= 128)
        let configuration = try QwenGatedDeltaNetConfiguration.official()
        serial = try QwenGatedDeltaNet(context: context, configuration: configuration, useOfficialSourceMath: true)
        lanes = try QwenGatedDeltaNet(context: context, configuration: configuration,
            useOfficialSourceMath: true, sourceGatedNormMode: .lanes128)
        let geometry = try QwenLinearAttentionGeometry(convolutionWidth: 4,
            convolutionChannelCount: configuration.convolutionChannelCount,
            valueHeadCount: 32, keyHeadDimension: 128, valueHeadDimension: 128)
        owner = try QwenLinearAttentionState(device: context.device, linearAttentionLayerMask: [1], geometry: geometry)
        initialState = (0..<(32 * 128 * 128)).map { Float($0 % 31 - 15) / 4096 }
        func vector(_ seed: Int) -> [Float] {
            (0..<4096).map { Float(($0 * 7 + seed) % 37 - 18) / 64 }
        }
        query = try NormLaneFixture.buffer(vector(3), device: context.device)
        key = try NormLaneFixture.buffer(vector(11), device: context.device)
        value = try NormLaneFixture.buffer(vector(23), device: context.device)
        decay = try NormLaneFixture.buffer([Float](repeating: -0.125, count: 32), device: context.device)
        beta = try NormLaneFixture.buffer([Float](repeating: 0.375, count: 32), device: context.device)
        gate = try NormLaneFixture.buffer(vector(17), device: context.device)
        normWeights = try NormLaneFixture.buffer((0..<128).map { Float($0 % 7 + 1) / 4 }, device: context.device)
        recurrent = try NormLaneFixture.buffer([Float](repeating: 0, count: 4096), device: context.device)
        norm = try NormLaneFixture.buffer([Float](repeating: 0, count: 4096), device: context.device)
        output = try NormLaneFixture.buffer([Float](repeating: 0, count: 2048), device: context.device)
        // Test-owned resident BF16 output projection, never an original shard.
        let palette: [UInt16] = [0x3b80, 0xbb80, 0x3c00, 0xbc00, 0x0000]
        let bits = (0..<(2048 * 4096)).map { palette[($0 * 7 + $0 / 4096) % palette.count] }
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "synthetic.norm.output", rows: 2048, columns: 4096, bits: bits),
        ])
        self.source = source
        do {
            weights = try QwenBF16Weights(context: context, source: source.handle, specifications: [
                QwenBF16TensorSpec(name: "synthetic.norm.output", shardName: source.shardName,
                                   role: .dense, rows: 2048, columns: 4096),
            ], residencyBudget: UInt64(2048 * 4096 * 2))
        } catch { source.remove(); throw error }
    }

    deinit { source.remove() }

    func run(mode: QwenSourceGatedNormMode) async throws -> Result {
        let runtime = mode == .serial ? serial : lanes
        let update = try owner.reserveUpdate(layer: 0)
        NormLaneFixture.write(initialState, to: update.recurrentMatrix)
        let encodingStarted = DispatchTime.now().uptimeNanoseconds
        let command = try #require(context.queue.makeCommandBuffer())
        do {
            try runtime.encodeRecurrence(commandBuffer: command, query: query, key: key,
                value: value, logDecay: decay, beta: beta, update: update,
                output: recurrent, tokenCount: 1, initialToken: false)
            try runtime.encodeGatedRMSNorm(commandBuffer: command, input: recurrent, gate: gate,
                weights: normWeights, output: norm, tokenCount: 1)
            try weights.encodeProjection(commandBuffer: command, tensorName: "synthetic.norm.output",
                input: norm, tokenCount: 1, output: output)
        } catch { try? owner.abort(update); throw error }
        let started = DispatchTime.now().uptimeNanoseconds
        do { try owner.submit(update, on: command) }
        catch { try? owner.abort(update); throw error }
        await command.completed()
        let resumed = DispatchTime.now().uptimeNanoseconds
        await owner.waitUntilIdle()
        let settled = DispatchTime.now().uptimeNanoseconds
        try checkCommandBufferError(command)
        let start = command.gpuStartTime, end = command.gpuEndTime
        try #require(start.isFinite && end.isFinite && start > 0 && end >= start)
        let values = NormLaneFixture.read(output, count: 2048)
        #expect(values.allSatisfy { $0.isFinite })
        return Result(mode: mode, outputBits: values.map(\.bitPattern),
            normBits: NormLaneFixture.read(norm, count: 4096).map(\.bitPattern),
            stateBits: NormLaneFixture.read(update.recurrentMatrix, count: initialState.count).map(\.bitPattern),
            gpuStart: start, gpuEnd: end, cpuNanos: resumed - started, encodeThroughSettlementNanos: settled - encodingStarted)
    }
}
