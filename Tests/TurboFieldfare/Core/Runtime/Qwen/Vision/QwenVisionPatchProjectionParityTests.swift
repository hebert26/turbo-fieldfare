import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenVisionPatchProjectionParityTests {
    @Test func sourcePatchProjectionPositionAndCombinedOutputMatchPinnedCPU() async throws {
        let fixture = try loadPatchFixture()
        #expect(fixture.kind == "qwen-source-vision-patch-cpu-fixture-v1")
        #expect(fixture.inputWidth == 1536)
        #expect(fixture.outputWidth == 1152)
        #expect(fixture.rows == 16)
        #expect(fixture.gridTHW == [1, 4, 4])
        #expect(fixture.spatialMergeSize == 2)
        #expect(fixture.positionCount == 2304)
        #expect(fixture.coordinates.count == fixture.rows)
        #expect(fixture.paddedRowsExpectedZero)
        #expect(fixture.inputBF16Bits.count == fixture.rows * fixture.inputWidth)
        #expect(fixture.biasBF16Bits.count == fixture.outputWidth)

        let weight = makeFixtureBF16(fixture.weightRecipe, count: fixture.outputWidth * fixture.inputWidth,
                                     overrides: fixture.weightOverrides)
        let position = makeFixtureBF16(fixture.positionRecipe, count: fixture.positionCount * fixture.outputWidth,
                                       overrides: [])
        #expect(sha256BF16(weight) == fixture.weightBF16SHA256)
        #expect(sha256BF16(position) == fixture.positionBF16SHA256)
        #expect(fixture.projectionFP32Bits.count == fixture.rows * fixture.outputWidth)
        #expect(fixture.positionFP32Bits.count == fixture.rows * fixture.outputWidth)
        #expect(fixture.outputFP32Bits.count == fixture.rows * fixture.outputWidth)

        let positions = fixture.coordinates.map { SIMD2<Int32>($0[0], $0[1]) }
        let context = try MetalContext()
        let buffers = PatchKernelBuffers(
            input: makePatchBuffer(fixture.inputBF16Bits, device: context.device),
            positions: makePositionBuffer(positions, device: context.device),
            weight: makePatchBuffer(weight, device: context.device),
            bias: makePatchBuffer(fixture.biasBF16Bits, device: context.device),
            positionTable: makePatchBuffer(position, device: context.device))

        let paddedRows = fixture.rows + 4
        for (component, expected) in [
            (UInt32(1), fixture.projectionFP32Bits),
            (UInt32(2), fixture.positionFP32Bits),
            (UInt32(0), fixture.outputFP32Bits),
        ] {
            let actual = try await runSourcePatchKernel(
                context: context, buffers: buffers, fixture: fixture,
                paddedRows: paddedRows, component: component)
            for index in expected.indices {
                #expect(actual[index].isFinite, "component \(component), scalar \(index) is non-finite")
                #expect(actual[index].bitPattern == expected[index],
                        "component \(component), row \(index / fixture.outputWidth), column \(index % fixture.outputWidth): actual=\(actual[index]) expected=\(Float(bitPattern: expected[index]))")
            }
            for row in fixture.rows..<paddedRows {
                let start = row * fixture.outputWidth
                #expect(actual[start..<(start + fixture.outputWidth)].allSatisfy { $0 == 0 })
            }
        }
    }
}

private struct PatchFixture: Decodable {
    let kind: String
    let inputWidth: Int
    let outputWidth: Int
    let rows: Int
    let gridTHW: [Int]
    let spatialMergeSize: Int
    let positionCount: Int
    let coordinates: [[Int32]]
    let inputBF16Bits: [UInt16]
    let biasBF16Bits: [UInt16]
    let weightRecipe: BF16Recipe
    let positionRecipe: BF16Recipe
    let weightOverrides: [BF16Override]
    let weightBF16SHA256: String
    let positionBF16SHA256: String
    let paddedRowsExpectedZero: Bool
    let projectionFP32Bits: [UInt32]
    let positionFP32Bits: [UInt32]
    let outputFP32Bits: [UInt32]
}

private struct BF16Recipe: Decodable {
    let kind: String
    let seed: UInt32
    let exponentBase: Int
    let exponentSpan: Int
}

private struct BF16Override: Decodable {
    let start: Int
    let count: Int
    let bits: UInt16
}

private struct PatchKernelParameters {
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

private struct PatchKernelBuffers {
    let input: MTLBuffer
    let positions: MTLBuffer
    let weight: MTLBuffer
    let bias: MTLBuffer
    let positionTable: MTLBuffer
}

private func loadPatchFixture() throws -> PatchFixture {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-patch-projection", isDirectory: true))
    return try JSONDecoder().decode(
        PatchFixture.self,
        from: Data(contentsOf: directory.appendingPathComponent("fixture.json")))
}

private func runSourcePatchKernel(
    context: MetalContext,
    buffers: PatchKernelBuffers,
    fixture: PatchFixture,
    paddedRows: Int,
    component: UInt32
) async throws -> [Float] {
    let library = try MetalContext.privateLibrary(
        device: context.device, module: "qwen_vision", mathMode: .safe,
        mathFloatingPointFunctions: .precise)
    let function = try #require(library.makeFunction(name: "qwen_source_vision_patch_position"))
    let pipeline = try await context.device.makeComputePipelineState(function: function)
    let outputCount = paddedRows * fixture.outputWidth
    let output = try #require(context.device.makeBuffer(
        length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared))
    var parameters = PatchKernelParameters(
        rows: UInt32(fixture.rows), paddedRows: UInt32(paddedRows),
        inputWidth: UInt32(fixture.inputWidth), outputWidth: UInt32(fixture.outputWidth),
        intermediateWidth: 0, heads: 0,
        gridHeight: UInt32(fixture.gridTHW[1]), gridWidth: UInt32(fixture.gridTHW[2]),
        mergeSize: UInt32(fixture.spatialMergeSize), positionCount: UInt32(fixture.positionCount),
        epsilon: 1e-6, weightScalarBytes: 2, patchScalarBytes: 2, reserved: component)
    let command = try #require(context.queue.makeCommandBuffer())
    let encoder = try #require(command.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBytes(&parameters, length: MemoryLayout<PatchKernelParameters>.stride, index: 0)
    encoder.setBuffer(buffers.input, offset: 0, index: 1)
    encoder.setBuffer(buffers.positions, offset: 0, index: 2)
    encoder.setBuffer(buffers.weight, offset: 0, index: 3)
    encoder.setBuffer(buffers.bias, offset: 0, index: 4)
    encoder.setBuffer(buffers.positionTable, offset: 0, index: 5)
    encoder.setBuffer(output, offset: 0, index: 6)
    let count = paddedRows * fixture.outputWidth
    let groupWidth = min(count, pipeline.maxTotalThreadsPerThreadgroup)
    encoder.dispatchThreads(
        MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: groupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    return Array(UnsafeBufferPointer(
        start: output.contents().assumingMemoryBound(to: Float.self), count: outputCount))
}

private func makeFixtureBF16(
    _ recipe: BF16Recipe, count: Int, overrides: [BF16Override]
) -> [UInt16] {
    precondition(recipe.kind == "wrapping-uint32-mix-bf16-v1" && recipe.exponentSpan > 0)
    var result = (0..<count).map { index -> UInt16 in
        var value = UInt32(truncatingIfNeeded: index) &+ recipe.seed
        value ^= value >> 16
        value &*= 0x7FEB352D
        value ^= value >> 15
        value &*= 0x846CA68B
        value ^= value >> 16
        let sign = (value >> 16) & 0x8000
        let exponent = UInt32(recipe.exponentBase + Int((value >> 8) % UInt32(recipe.exponentSpan)))
        return UInt16(sign | (exponent << 7) | (value & 0x7F))
    }
    for item in overrides {
        precondition(item.start >= 0 && item.count >= 0 && item.start + item.count <= result.count)
        result.replaceSubrange(item.start..<(item.start + item.count),
                                with: repeatElement(item.bits, count: item.count))
    }
    return result
}

private func sha256BF16(_ values: [UInt16]) -> String {
    var data = Data(capacity: values.count * MemoryLayout<UInt16>.stride)
    for value in values {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
    }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func makePatchBuffer(_ values: [UInt16], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(
        bytes: values, length: values.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared)!
}

private func makePositionBuffer(_ values: [SIMD2<Int32>], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(
        bytes: values, length: values.count * MemoryLayout<SIMD2<Int32>>.stride,
        options: .storageModeShared)!
}
