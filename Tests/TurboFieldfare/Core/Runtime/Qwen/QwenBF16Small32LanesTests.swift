import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized)
struct QwenBF16Small32LanesTests {
    @Test func exactSmall32StreamsAcrossGlobalRowBoundariesAndTokens() async throws {
        let fixture = try Small32ResidentFixture(stage: false)
        for seed in 0..<8 {
            for tokens in [1, 3] {
                try fixture.replaceInput(tokens: tokens, seed: seed)
                let serial = try await fixture.run(lanes: false, tokens: tokens, chunkRows: 6)
                let parallel = try await fixture.run(lanes: true, tokens: tokens, chunkRows: 6)
                #expect(serial.bits == parallel.bits)
                #expect(serial.guardsIntact && parallel.guardsIntact)
            }
        }
    }

    @Test func actualFourProjectionStageMatchesEveryOutputBit() async throws {
        let fixture = try Small32ResidentFixture(stage: true)
        let serial = try await fixture.run(lanes: false)
        let parallel = try await fixture.run(lanes: true)
        #expect(serial.bits == parallel.bits)
        #expect(serial.guardsIntact && parallel.guardsIntact)
    }

    // Explicitly opt-in. Four real-shaped projections together, not a dot
    // helper timer. Excludes convolution, preparation, recurrence and disk.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBOFIELDFARE_SMALL32_STAGE_TIMING"] == "1"))
    func warmedFourProjectionStageABBA() async throws {
        let fixture = try Small32ResidentFixture(stage: true)
        for _ in 0..<8 {
            let a = try await fixture.run(lanes: false)
            let b = try await fixture.run(lanes: true)
            #expect(a.bits == b.bits)
        }
        for pair in 0..<20 {
            let modes = pair.isMultiple(of: 2) ? [false, true] : [true, false]
            let first = try await fixture.run(lanes: modes[0])
            let second = try await fixture.run(lanes: modes[1])
            #expect(first.bits == second.bits)
            #expect(first.guardsIntact && second.guardsIntact)
            for (ordinal, result) in [first, second].enumerated() {
                let row: [String: Any] = [
                    "stage": "resident_four_linear_projections", "pair": pair,
                    "order": ordinal, "small32_lanes": modes[ordinal],
                    "rows": [8192, 4096, 32, 32], "columns": 2048,
                    "commands": 1, "gpu_seconds": result.gpuSeconds,
                    "encode_through_settlement_ns": result.cpuNanos,
                    "limitations": "resident synthetic; excludes convolution and all later stages"
                ]
                let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
                print("QWEN_SMALL32_STAGE " + String(decoding: data, as: UTF8.self))
            }
        }
    }
}

private final class Small32ResidentFixture {
    struct Parameters {
        var rows: UInt32, columns: UInt32, firstRow: UInt32
        var rowsInChunk: UInt32, tokenCount: UInt32
    }
    struct Result {
        let bits: [[UInt32]]
        let guardsIntact: Bool
        let gpuSeconds: Double
        let cpuNanos: UInt64
    }
    let context: MetalContext
    let serial: MTLComputePipelineState
    let large: MTLComputePipelineState
    let lanes: MTLComputePipelineState
    let rows: [Int]
    let weights: [MTLBuffer]
    let outputs: [MTLBuffer]
    let input: MTLBuffer
    let sentinel: UInt32 = 0x42f70000

    init(stage: Bool) throws {
        context = try MetalContext()
        // Production library loader locates bundled Metal sources. The
        // candidate resource is added only when Main applies this patch.
        let library = try MetalContext.privateLibrary(device: context.device,
            module: "qwen_bf16", mathMode: .safe)
        let laneLibrary = try MetalContext.privateLibrary(device: context.device,
            module: "qwen_bf16_small32_lanes", mathMode: .safe)
        serial = try context.device.makeComputePipelineState(function:
            #require(library.makeFunction(name: "qwen_bf16_project_fp32")))
        large = try context.device.makeComputePipelineState(function:
            #require(library.makeFunction(name: "qwen_bf16_project_source64_fp32")))
        lanes = try context.device.makeComputePipelineState(function:
            #require(laneLibrary.makeFunction(name: "qwen_bf16_project_small32_lanes")))
        try #require(lanes.threadExecutionWidth == 32 && lanes.maxTotalThreadsPerThreadgroup >= 32)
        try #require(large.maxTotalThreadsPerThreadgroup >= 64)
        rows = stage ? [8192, 4096, 32, 32] : [32]
        var stagedWeights: [MTLBuffer] = [], stagedOutputs: [MTLBuffer] = []
        for (matrix, count) in rows.enumerated() {
            let buffer = try #require(context.device.makeBuffer(length: count * 2048 * 2, options: .storageModeShared))
            let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
            // Signed BF16 values with varied exponents and cancellation.
            for index in 0..<(count * 2048) {
                let value = Float((index * 17 + matrix * 23) % 251 - 125) / 128
                pointer[index] = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
            }
            stagedWeights.append(buffer)
            stagedOutputs.append(try #require(context.device.makeBuffer(length: (count * 3 + 16) * 4,
                                                                          options: .storageModeShared)))
        }
        weights = stagedWeights
        outputs = stagedOutputs
        input = try #require(context.device.makeBuffer(length: 3 * 2048 * 4, options: .storageModeShared))
        try replaceInput(tokens: 1, seed: 0)
    }

    func replaceInput(tokens: Int, seed: Int) throws {
        try #require((1...3).contains(tokens))
        let pointer = input.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<(tokens * 2048) {
            let sign: Float = index.isMultiple(of: 2) ? 1 : -1
            switch seed {
            case 1: pointer[index] = sign * 1e-30
            case 2: pointer[index] = sign * 1e18
            case 3: pointer[index] = index.isMultiple(of: 2) ? Float.zero : -Float.zero
            case 4: pointer[index] = sign * Float.leastNormalMagnitude
            case 5: pointer[index] = sign * Float.leastNonzeroMagnitude
            default: pointer[index] = Float((index * (seed + 3)) % 127 - 63) / 32
            }
        }
    }

    func run(lanes useLanes: Bool, tokens: Int = 1, chunkRows: Int? = nil) async throws -> Result {
        for (index, output) in outputs.enumerated() {
            output.contents().assumingMemoryBound(to: UInt32.self)
                .update(repeating: sentinel, count: rows[index] * 3 + 16)
        }
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let command = try #require(context.queue.makeCommandBuffer())
        for matrix in rows.indices {
            let rowCount = rows[matrix]
            let width = chunkRows ?? rowCount
            let pipeline = rowCount == 32 ? (useLanes ? lanes : serial) : large
            for first in stride(from: 0, to: rowCount, by: width) {
                let count = min(width, rowCount - first)
                var params = Parameters(rows: UInt32(rowCount), columns: 2048,
                    firstRow: UInt32(first), rowsInChunk: UInt32(count), tokenCount: UInt32(tokens))
                let encoder = try #require(command.makeComputeCommandEncoder())
                encoder.setComputePipelineState(pipeline)
                encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 0)
                encoder.setBuffer(input, offset: 0, index: 1)
                encoder.setBuffer(weights[matrix], offset: first * 2048 * 2, index: 2)
                encoder.setBuffer(outputs[matrix], offset: 0, index: 5)
                if rowCount != 32 || useLanes {
                    encoder.dispatchThreadgroups(MTLSize(width: count, height: tokens, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: rowCount == 32 ? 32 : 64, height: 1, depth: 1))
                } else {
                    encoder.dispatchThreads(MTLSize(width: count, height: tokens, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(count, pipeline.threadExecutionWidth), height: 1, depth: 1))
                }
                encoder.endEncoding()
            }
        }
        command.commit()
        await command.completed()
        let elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
        try checkCommandBufferError(command)
        var bits: [[UInt32]] = []
        var intact = true
        for matrix in rows.indices {
            let pointer = outputs[matrix].contents().assumingMemoryBound(to: UInt32.self)
            bits.append(Array(UnsafeBufferPointer(start: pointer, count: rows[matrix] * tokens)))
            for index in (rows[matrix] * tokens)..<(rows[matrix] * 3 + 16) {
                intact = intact && pointer[index] == sentinel
            }
        }
        return Result(bits: bits, guardsIntact: intact,
                      gpuSeconds: command.gpuEndTime - command.gpuStartTime, cpuNanos: elapsed)
    }
}
