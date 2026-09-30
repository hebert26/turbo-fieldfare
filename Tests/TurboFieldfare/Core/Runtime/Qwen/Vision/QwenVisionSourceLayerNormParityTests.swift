import Foundation
import CryptoKit
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenVisionSourceLayerNormParityTests {
    @Test func sourceLayerNormMatchesPinnedTorchUnitAffineRow54() async throws {
        let fixture = try loadVisionLayerNormRow54Fixture()
        #expect(fixture.kind == "qwen-source-vision-layernorm-row-fixture-v1")
        #expect(fixture.row == 54)
        #expect(fixture.width == 1152)
        #expect(fixture.epsilon == 1e-6)
        #expect(fixture.affine == "explicit unit gamma, zero beta")
        #expect(fixture.expectedNativeMeanBits == 3_147_055_095)
        #expect(fixture.expectedNativeRstdBits == 1_077_235_360)

        let inputBits = try loadLittleEndianFloatBits(
            fixture.inputFile, expectedSHA256: fixture.inputSHA256,
            expectedCount: fixture.width)
        let expectedBits = try loadLittleEndianFloatBits(
            fixture.expectedFile, expectedSHA256: fixture.expectedSHA256,
            expectedCount: fixture.width)
        let inputs = inputBits.map(Float.init(bitPattern:))
        let weight = Array(repeating: Float(1), count: fixture.width)
        let bias = Array(repeating: Float(0), count: fixture.width)
        #expect(inputs.allSatisfy { $0.isFinite })
        #expect(expectedBits.allSatisfy { Float(bitPattern: $0).isFinite })

        let context = try MetalContext()
        let actual = try await runSourceNorm(
            context: context, name: "qwen_source_vision_layernorm",
            input: makeLayerNormBuffer(inputs, device: context.device),
            weight: makeLayerNormBuffer(weight, device: context.device),
            bias: makeLayerNormBuffer(bias, device: context.device),
            rows: 1, paddedRows: 4, width: fixture.width,
            outputWidth: fixture.width, epsilon: fixture.epsilon)

        #expect(actual.count >= fixture.width)
        for column in 0..<fixture.width {
            #expect(actual[column].bitPattern == expectedBits[column],
                    "row 54 column \(column): actual=0x\(String(actual[column].bitPattern, radix: 16)) expected=0x\(String(expectedBits[column], radix: 16))")
        }
    }

    @Test func sourceLayerNormAndMergerPackMatchPinnedTorchRows() async throws {
        let fixture = try loadVisionLayerNormFixture()
        #expect(fixture.kind == "qwen-source-vision-layernorm-cpu-fixture-v1")
        #expect(fixture.width == 1152)
        #expect(fixture.epsilon == 1e-6)
        #expect(fixture.weightBF16Bits.count == fixture.width)
        #expect(fixture.biasBF16Bits.count == fixture.width)
        #expect(!fixture.cases.isEmpty)

        let width = fixture.width
        let rowCount = fixture.cases.count
        let paddedRows = (rowCount / 4 + 1) * 4
        #expect(fixture.cases.allSatisfy {
            $0.inputFP32Bits.count == width && $0.outputFP32Bits.count == width
        })

        let inputs = fixture.cases.flatMap { $0.inputFP32Bits.map(Float.init(bitPattern:)) }
        let expected = fixture.cases.flatMap { $0.outputFP32Bits.map(Float.init(bitPattern:)) }
        let weight = fixture.weightBF16Bits.map { Float(bitPattern: UInt32($0) << 16) }
        let bias = fixture.biasBF16Bits.map { Float(bitPattern: UInt32($0) << 16) }
        #expect(inputs.allSatisfy { $0.isFinite })
        #expect(expected.allSatisfy { $0.isFinite })

        let context = try MetalContext()
        let inputBuffer = makeLayerNormBuffer(inputs, device: context.device)
        let weightBuffer = makeLayerNormBuffer(weight, device: context.device)
        let biasBuffer = makeLayerNormBuffer(bias, device: context.device)

        let layerNorm = try await runSourceNorm(
            context: context, name: "qwen_source_vision_layernorm",
            input: inputBuffer, weight: weightBuffer, bias: biasBuffer,
            rows: rowCount, paddedRows: paddedRows, width: width,
            outputWidth: width, epsilon: fixture.epsilon)
        let mergerPack = try await runSourceNorm(
            context: context, name: "qwen_source_vision_merger_norm_pack",
            input: inputBuffer, weight: weightBuffer, bias: biasBuffer,
            rows: rowCount, paddedRows: rowCount, width: width,
            outputWidth: width * 4, epsilon: fixture.epsilon)

        for (name, actual) in [("layernorm", layerNorm), ("merger pack", mergerPack)] {
            #expect(actual.count >= expected.count)
            for index in expected.indices {
                let reference = expected[index]
                let value = actual[index]
                let limit = 1e-7 + 1e-6 * abs(reference)
                #expect(value.isFinite, "\(name) produced non-finite output at scalar \(index)")
                #expect(abs(Double(value) - Double(reference)) <= Double(limit),
                        "\(name) case \(fixture.cases[index / width].name), column \(index % width): actual=\(value) expected=\(reference)")
            }
        }

        for row in rowCount..<paddedRows {
            let start = row * width
            #expect(layerNorm[start..<(start + width)].allSatisfy { $0 == 0 })
        }
    }
}

private struct VisionLayerNormFixture: Decodable {
    let kind: String
    let width: Int
    let epsilon: Float
    let weightBF16Bits: [UInt16]
    let biasBF16Bits: [UInt16]
    let cases: [VisionLayerNormCase]
}

private struct VisionLayerNormCase: Decodable {
    let name: String
    let inputFP32Bits: [UInt32]
    let outputFP32Bits: [UInt32]
    let meanFP32Bits: UInt32
    let rstdFP32Bits: UInt32
}

private struct VisionLayerNormRow54Fixture: Decodable {
    let kind: String
    let row: Int
    let width: Int
    let epsilon: Float
    let affine: String
    let inputFile: String
    let inputSHA256: String
    let expectedFile: String
    let expectedSHA256: String
    let expectedNativeMeanBits: UInt32
    let expectedNativeRstdBits: UInt32
}

private struct VisionLayerNormParameters {
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

private func loadVisionLayerNormFixture() throws -> VisionLayerNormFixture {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-source-layernorm", isDirectory: true))
    let url = directory.appendingPathComponent("fixture.json")
    return try JSONDecoder().decode(VisionLayerNormFixture.self, from: Data(contentsOf: url))
}

private func loadVisionLayerNormRow54Fixture() throws -> VisionLayerNormRow54Fixture {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-source-layernorm", isDirectory: true))
    let url = directory.appendingPathComponent("row54-fixture.json")
    return try JSONDecoder().decode(VisionLayerNormRow54Fixture.self, from: Data(contentsOf: url))
}

private func loadLittleEndianFloatBits(
    _ filename: String,
    expectedSHA256: String,
    expectedCount: Int
) throws -> [UInt32] {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-source-layernorm", isDirectory: true))
    let data = try Data(contentsOf: directory.appendingPathComponent(filename))
    guard data.count == expectedCount * MemoryLayout<UInt32>.stride else {
        throw CocoaError(.fileReadCorruptFile)
    }
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    #expect(digest == expectedSHA256)
    return stride(from: 0, to: data.count, by: MemoryLayout<UInt32>.stride).map { offset in
        let value = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
        return UInt32(littleEndian: value)
    }
}

private func runSourceNorm(
    context: MetalContext,
    name: String,
    input: MTLBuffer,
    weight: MTLBuffer,
    bias: MTLBuffer,
    rows: Int,
    paddedRows: Int,
    width: Int,
    outputWidth: Int,
    epsilon: Float
) async throws -> [Float] {
    let library = try MetalContext.privateLibrary(
        device: context.device, module: "qwen_vision", mathMode: .safe,
        mathFloatingPointFunctions: .precise)
    let function = try #require(library.makeFunction(name: name))
    let pipeline = try await context.device.makeComputePipelineState(function: function)
    let outputCount = paddedRows * width
    let output = try #require(context.device.makeBuffer(
        length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared))
    var parameters = VisionLayerNormParameters(
        rows: UInt32(rows), paddedRows: UInt32(paddedRows),
        inputWidth: UInt32(width), outputWidth: UInt32(outputWidth),
        intermediateWidth: 0, heads: 0, gridHeight: 0, gridWidth: 0,
        mergeSize: 0, positionCount: 0, epsilon: epsilon,
        weightScalarBytes: 4, patchScalarBytes: 4, reserved: 0)
    let command = try #require(context.queue.makeCommandBuffer())
    let encoder = try #require(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBytes(&parameters, length: MemoryLayout<VisionLayerNormParameters>.stride, index: 0)
    encoder.setBuffer(input, offset: 0, index: 1)
    encoder.setBuffer(weight, offset: 0, index: 2)
    encoder.setBuffer(bias, offset: 0, index: 3)
    encoder.setBuffer(output, offset: 0, index: 4)
    let threadCount = name == "qwen_source_vision_layernorm" ? paddedRows : rows
    let groupWidth = min(threadCount, pipeline.maxTotalThreadsPerThreadgroup)
    encoder.dispatchThreads(
        MTLSize(width: threadCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    return Array(UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: Float.self), count: outputCount))
}

private func makeLayerNormBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(
        bytes: values, length: values.count * MemoryLayout<Float>.stride,
        options: .storageModeShared)!
}
