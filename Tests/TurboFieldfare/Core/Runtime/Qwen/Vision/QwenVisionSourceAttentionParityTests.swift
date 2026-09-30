import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenVisionSourceAttentionParityTests {
    @Test func sourceAttentionMatchesPinnedCPUOnSevenRowsAndZeroesRuntimePadding() async throws {
        let fixture = try loadAttentionFixture()
        #expect(fixture.kind == "qwen-vision-synthetic-attention-shape-oracle-v1")
        #expect(fixture.complete)
        #expect(fixture.heads == 16)
        #expect(fixture.dimension == 72)
        #expect(fixture.rows.contains(7))
        let item = try #require(fixture.cases.first { $0.rows == 7 })
        #expect(item.rows == 7)
        #expect(item.heads == fixture.heads)
        #expect(item.dimension == fixture.dimension)
        #expect(item.scaleFP32Bits == fixture.scaleFP32Bits)
        #expect(fixture.scaleFP32Bits == 1_039_227_887)
        #expect(item.seed == 0x2325A007)
        #expect(item.qkv.shape == [7, 3 * fixture.heads * fixture.dimension])
        #expect(item.qkv.count == 7 * 3 * fixture.heads * fixture.dimension)
        #expect(item.expectedContext.shape == [7, fixture.heads * fixture.dimension])
        #expect(item.expectedContext.count == 7 * fixture.heads * fixture.dimension)

        let qkvBits = try loadAttentionBits(item.qkv)
        let expectedBits = try loadAttentionBits(item.expectedContext)
        let qkv = qkvBits.map(Float.init(bitPattern:))
        let expected = expectedBits.map(Float.init(bitPattern:))
        #expect(qkv.allSatisfy { $0.isFinite })
        #expect(expected.allSatisfy { $0.isFinite })

        let context = try MetalContext()
        let inputBuffer = makeAttentionBuffer(qkv, device: context.device)
        let rows = item.rows
        let paddedRows = ((rows + 63) / 64) * 64
        let outputCount = paddedRows * fixture.heads * fixture.dimension
        let outputBuffer = try #require(context.device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared))
        let library = try MetalContext.privateLibrary(
            device: context.device, module: "qwen_vision", mathMode: .safe,
            mathFloatingPointFunctions: .precise, includeQwenSourceMath: true)
        let function = try #require(library.makeFunction(name: "qwen_source_vision_attention"))
        let pipeline = try await context.device.makeComputePipelineState(function: function)
        var parameters = VisionAttentionParameters(
            rows: UInt32(rows), paddedRows: UInt32(paddedRows),
            inputWidth: UInt32(fixture.heads * fixture.dimension),
            outputWidth: UInt32(fixture.heads * fixture.dimension),
            intermediateWidth: 0, heads: UInt32(fixture.heads),
            gridHeight: 0, gridWidth: 0, mergeSize: 0, positionCount: 0,
            epsilon: 1e-6, weightScalarBytes: 4, patchScalarBytes: 4, reserved: 0)
        var scale = Float(bitPattern: fixture.scaleFP32Bits)
        #expect(scale.bitPattern == fixture.scaleFP32Bits)
        #expect(scale.bitPattern == Float(pow(Double(fixture.dimension), -0.5)).bitPattern)

        let command = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&parameters, length: MemoryLayout<VisionAttentionParameters>.stride, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 3)
        let workCount = paddedRows * fixture.heads
        let groupWidth = min(workCount, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: workCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        await command.completed()
        #expect(command.status == .completed)
        #expect(command.error == nil)

        let actual = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().assumingMemoryBound(to: Float.self), count: outputCount))
        for index in expected.indices {
            let value = actual[index]
            let reference = expected[index]
            let limit = 1e-7 + 1e-6 * abs(reference)
            #expect(value.isFinite, "non-finite context at scalar \(index)")
            #expect(abs(Double(value) - Double(reference)) <= Double(limit),
                    "row \(index / (fixture.heads * fixture.dimension)), head/dimension \(index % (fixture.heads * fixture.dimension)): actual=\(value) expected=\(reference)")
        }
        #expect(actual[(rows * fixture.heads * fixture.dimension)..<outputCount]
            .allSatisfy { $0 == 0 }, "runtime-padded rows must be zero")
    }
}

private struct AttentionSweep: Decodable {
    let kind: String
    let complete: Bool
    let rows: [Int]
    let heads: Int
    let dimension: Int
    let scaleFP32Bits: UInt32
    let cases: [AttentionCase]
}

private struct AttentionCase: Decodable {
    let rows: Int
    let heads: Int
    let dimension: Int
    let seed: UInt32
    let scaleFP32Bits: UInt32
    let qkv: AttentionArrayRecord
    let expectedContext: AttentionArrayRecord
}

private struct AttentionArrayRecord: Decodable {
    let file: String
    let shape: [Int]
    let count: Int
    let byteCount: Int
    let sha256: String
}

private struct VisionAttentionParameters {
    var rows: UInt32
    var paddedRows: UInt32
    var inputWidth: UInt32
    var outputWidth: UInt32
    var intermediateWidth: UInt32
    var heads: UInt32
    var gridHeight: UInt32
    var gridWidth: UInt32
    var mergeSize: UInt32
    var positionCount: UInt32
    var epsilon: Float
    var weightScalarBytes: UInt32
    var patchScalarBytes: UInt32
    var reserved: UInt32
}

private func loadAttentionFixture() throws -> AttentionSweep {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-source-attention", isDirectory: true))
    return try JSONDecoder().decode(
        AttentionSweep.self,
        from: Data(contentsOf: directory.appendingPathComponent("sweep.json")))
}

private func loadAttentionBits(_ record: AttentionArrayRecord) throws -> [UInt32] {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-source-attention", isDirectory: true))
    let data = try Data(contentsOf: directory.appendingPathComponent(record.file))
    #expect(data.count == record.byteCount)
    #expect(data.count == record.count * MemoryLayout<UInt32>.stride)
    #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == record.sha256)
    return stride(from: 0, to: data.count, by: 4).map { offset -> UInt32 in
        let byte0 = UInt32(data[offset])
        let byte1 = UInt32(data[offset + 1])
        let byte2 = UInt32(data[offset + 2])
        let byte3 = UInt32(data[offset + 3])
        return byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    }
}

private func makeAttentionBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(
        bytes: values, length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared)!
}
