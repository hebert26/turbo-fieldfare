import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenVisionSourceRotaryParityTests {
    @Test func sourceRotationMatchesPinnedCPUAcrossZeroAndMixedAxes() async throws {
        let fixture = try loadVisionRotaryFixture()
        #expect(fixture.kind == "qwen-source-vision-rotary-cpu-fixture-v1")
        #expect(fixture.rows == 3)
        #expect(fixture.hiddenSize == 1152)
        #expect(fixture.heads == 16)
        #expect(fixture.headDimension == 72)
        #expect(fixture.frequencyCount == 18)
        #expect(fixture.positions == [[0, 0], [6, 10], [13, 21]])
        #expect(fixture.crossAxisSIMD4Dimensions == [16, 17, 18, 19])
        #expect(fixture.negativeExponentFrequencyDifferenceCount > 0)
        #expect(fixture.inputQKVFP32Bits.count == fixture.rows * 3 * fixture.hiddenSize)
        #expect(fixture.expectedQKFP32Bits.count == fixture.rows * 2 * fixture.hiddenSize)
        #expect(fixture.inverseFrequenciesFP32Bits.count == fixture.frequencyCount)

        let frequencies = try QwenVisionSourceRotaryArithmetic.inverseFrequencies(
            headDimension: fixture.headDimension)
        #expect(frequencies.count == fixture.frequencyCount)
        for index in frequencies.indices {
            #expect(frequencies[index].bitPattern == fixture.inverseFrequenciesFP32Bits[index],
                    "inverse frequency \(index) differs from pinned official FP32")
        }

        let input = fixture.inputQKVFP32Bits.map(Float.init(bitPattern:))
        #expect(input.allSatisfy { $0.isFinite })
        let positions = fixture.positions.map { SIMD2<Int32>(Int32($0[0]), Int32($0[1])) }
        let context = try MetalContext()
        let inputOutputBuffer = makeRotaryBuffer(input, device: context.device)
        let positionsBuffer = makeRotaryBuffer(positions, device: context.device)
        let frequencyBuffer = makeRotaryBuffer(frequencies, device: context.device)

        let library = try MetalContext.privateLibrary(
            device: context.device, module: "qwen_vision", mathMode: .safe,
            mathFloatingPointFunctions: .precise)
        let function = try #require(library.makeFunction(name: "qwen_source_vision_rotate_qk"))
        let pipeline = try await context.device.makeComputePipelineState(function: function)
        let parameters = VisionRotaryParameters(
            rows: UInt32(fixture.rows), paddedRows: UInt32(fixture.rows),
            inputWidth: UInt32(fixture.hiddenSize), outputWidth: UInt32(fixture.hiddenSize),
            intermediateWidth: 4304, heads: UInt32(fixture.heads),
            gridHeight: 14, gridWidth: 22, mergeSize: 2, positionCount: 2304,
            epsilon: 1e-6, weightScalarBytes: 4, patchScalarBytes: 4, reserved: 0)
        let command = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        var mutableParameters = parameters
        encoder.setBytes(&mutableParameters, length: MemoryLayout<VisionRotaryParameters>.stride, index: 0)
        encoder.setBuffer(positionsBuffer, offset: 0, index: 1)
        encoder.setBuffer(inputOutputBuffer, offset: 0, index: 2)
        encoder.setBuffer(frequencyBuffer, offset: 0, index: 3)
        let workCount = fixture.rows * fixture.heads * fixture.headDimension
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
            start: inputOutputBuffer.contents().assumingMemoryBound(to: Float.self),
            count: input.count))
        let hidden = fixture.hiddenSize
        for row in 0..<fixture.rows {
            for projection in 0..<2 {
                for head in 0..<fixture.heads {
                    for dimension in 0..<fixture.headDimension {
                        let actualIndex = row * 3 * hidden + projection * hidden
                            + head * fixture.headDimension + dimension
                        let expectedIndex = (row * 2 + projection) * hidden
                            + head * fixture.headDimension + dimension
                        let expectedBits = fixture.expectedQKFP32Bits[expectedIndex]
                        #expect(actual[actualIndex].isFinite)
                        #expect(actual[actualIndex].bitPattern == expectedBits,
                                "row \(row), projection \(projection), head \(head), dimension \(dimension): actual=\(actual[actualIndex]) expected=\(Float(bitPattern: expectedBits))")
                    }
                }
            }
            let keyValueStart = row * 3 * hidden + 2 * hidden
            let inputKeyValueStart = keyValueStart
            for column in 0..<hidden {
                #expect(actual[keyValueStart + column].bitPattern
                    == fixture.inputQKVFP32Bits[inputKeyValueStart + column],
                    "value projection changed at row \(row), column \(column)")
            }
        }
    }
}

private struct VisionRotaryFixture: Decodable {
    let kind: String
    let rows: Int
    let hiddenSize: Int
    let heads: Int
    let headDimension: Int
    let frequencyCount: Int
    let positions: [[Int32]]
    let crossAxisSIMD4Dimensions: [Int]
    let inputQKVFP32Bits: [UInt32]
    let inverseFrequenciesFP32Bits: [UInt32]
    let expectedQKFP32Bits: [UInt32]
    let negativeExponentFrequencyDifferenceCount: Int
}

private struct VisionRotaryParameters {
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

private func loadVisionRotaryFixture() throws -> VisionRotaryFixture {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-source-rotary", isDirectory: true))
    return try JSONDecoder().decode(
        VisionRotaryFixture.self,
        from: Data(contentsOf: directory.appendingPathComponent("fixture.json")))
}

private func makeRotaryBuffer<T>(_ values: [T], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(
        bytes: values, length: values.count * MemoryLayout<T>.stride,
        options: .storageModeShared)!
}
