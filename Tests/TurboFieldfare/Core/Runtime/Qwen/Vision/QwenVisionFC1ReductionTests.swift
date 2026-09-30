import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenVisionFC1ReductionTests {
    @Test func sourceFC1AccelerateProjectMatchesPinnedBitsAtFullVisionShape() async throws {
        let fixture = try loadFC1Fixture()
        #expect(fixture.kind == "qwen-source-vision-fc1-reduction-synthetic-v1")
        #expect(fixture.shape == [308, 4304, 1152])
        #expect(fixture.inputShape == [308, 1152])
        #expect(fixture.rows == [0, 154, 307])
        #expect(fixture.columns.contains(2143))
        #expect(fixture.columns.contains(2144))
        #expect(fixture.columns.contains(2159))
        #expect(fixture.columns.contains(2160))

        let input = try readFloats(fixture.inputFile, count: fixture.inputShape.reduce(1, *))
        let selectedWeights = try readBF16(fixture.weightFile, count: fixture.weightShape.reduce(1, *))
        let selectedBias = try readBF16(fixture.biasFile, count: fixture.biasShape.reduce(1, *))
        let expected = try readBits(fixture.expectedFile, count: fixture.expectedShape.reduce(1, *))
        #expect(input.allSatisfy { $0.isFinite })
        #expect(expected.count == fixture.rows.count * fixture.columns.count)

        var weights = [UInt16](repeating: 0, count: 4304 * 1152)
        for (selected, column) in fixture.columns.enumerated() {
            let source = selected * 1152
            let destination = column * 1152
            weights.replaceSubrange(destination..<(destination + 1152),
                                     with: selectedWeights[source..<(source + 1152)])
        }
        var bias = [UInt16](repeating: 0, count: 4304)
        for (selected, column) in fixture.columns.enumerated() {
            bias[column] = selectedBias[selected]
        }

        let context = try MetalContext()
        let inputBuffer = try #require(context.device.makeBuffer(
            bytes: input, length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let weightBuffer = try #require(context.device.makeBuffer(
            bytes: weights, length: weights.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let biasBuffer = try #require(context.device.makeBuffer(
            bytes: bias, length: bias.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let outputCount = fixture.shape[0] * fixture.shape[1]
        let outputBuffer = try #require(context.device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared))
        let stagingBytes = try QwenSourceVisionLinear.stagingByteCount(
            inputWidth: fixture.shape[2], outputWidth: fixture.shape[1])
        #expect(stagingBytes == 19_850_048)
        #expect(stagingBytes < 200_000_000)
        let stagingBuffer = try #require(context.device.makeBuffer(
            length: stagingBytes, options: .storageModeShared))
        try QwenSourceVisionLinear.project(
            rows: fixture.shape[0], inputWidth: fixture.shape[2], outputWidth: fixture.shape[1],
            input: inputBuffer, weight: weightBuffer, weightOffset: 0,
            bias: biasBuffer, biasOffset: 0, output: outputBuffer, staging: stagingBuffer)

        let actual = Array(UnsafeBufferPointer(
            start: outputBuffer.contents().assumingMemoryBound(to: Float.self), count: outputCount))
        for (rowIndex, row) in fixture.rows.enumerated() {
            for (columnIndex, column) in fixture.columns.enumerated() {
                let actualIndex = row * fixture.shape[1] + column
                let expectedIndex = rowIndex * fixture.columns.count + columnIndex
                #expect(actual[actualIndex].bitPattern == expected[expectedIndex],
                        "row \(row), column \(column): actual=\(actual[actualIndex]) expected=\(Float(bitPattern: expected[expectedIndex]))")
            }
        }
    }
}

private struct FC1Fixture: Decodable {
    let kind: String
    let shape: [Int]
    let rows: [Int]
    let columns: [Int]
    let inputFile: String
    let weightFile: String
    let biasFile: String
    let expectedFile: String
    let inputShape: [Int]
    let weightShape: [Int]
    let biasShape: [Int]
    let expectedShape: [Int]
}

private func loadFC1Fixture() throws -> FC1Fixture {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-patch-projection/fc1-reduction", isDirectory: true))
    return try JSONDecoder().decode(
        FC1Fixture.self,
        from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
}

private func fixtureURL(_ name: String) throws -> URL {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "vision-patch-projection/fc1-reduction", isDirectory: true))
    return directory.appendingPathComponent(name)
}

private func readFloats(_ name: String, count: Int) throws -> [Float] {
    let data = try Data(contentsOf: try fixtureURL(name))
    #expect(data.count == count * MemoryLayout<Float>.stride)
    return data.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: Float.self))
    }
}

private func readBF16(_ name: String, count: Int) throws -> [UInt16] {
    let data = try Data(contentsOf: try fixtureURL(name))
    #expect(data.count == count * MemoryLayout<UInt16>.stride)
    return data.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: UInt16.self))
    }
}

private func readBits(_ name: String, count: Int) throws -> [UInt32] {
    let data = try Data(contentsOf: try fixtureURL(name))
    #expect(data.count == count * MemoryLayout<UInt32>.stride)
    return data.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: UInt32.self))
    }
}
