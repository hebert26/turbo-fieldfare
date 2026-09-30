import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenBF16StableProjectionTests {
    @Test func fullWidthMixedSignsMatchIndependentDoubleDots() throws {
        let columns = 2048
        let rows = 32
        var input: [Float] = []
        for index in 0..<columns {
            let magnitude = UInt16(0x3c00 + index * 37 % 1024)
            let sign: UInt16 = index % 3 == 0 ? 0x8000 : 0
            input.append(Float(bitPattern: UInt32(magnitude | sign) << 16))
        }
        var weights: [UInt16] = []
        for row in 0..<rows {
            for index in 0..<columns {
                let magnitude = UInt16(0x3b00 + (index * 17 + row * 29) % 1536)
                let sign: UInt16 = (index + row) % 5 < 2 ? 0x8000 : 0
                weights.append(magnitude | sign)
            }
        }
        let expected = doubleDots(input: input, weights: weights,
                                  rows: rows, columns: columns)
        let actual = try project(input: input, weights: weights,
                                 rows: rows, columns: columns)
        #expect(actual.count == rows)
        for (value, reference) in zip(actual, expected) {
            #expect(value.isFinite)
            #expect(abs(value - reference) <= 1e-7 + 1e-6 * abs(reference))
        }
        let old = orderedDots(input: input, weights: weights,
                              rows: rows, columns: columns)
        let rejected = zip(old, expected).filter {
            abs($0.0 - $0.1) > 1e-7 + 1e-6 * abs($0.1)
        }
        #expect(!rejected.isEmpty)
    }

    @Test func fullWidthCancellationRetainsSmallTerms() throws {
        let pattern: [Float] = [16_777_216, 1, -16_777_216, 1]
        let input = Array(repeating: pattern, count: 512).flatMap { $0 }
        let weights = [UInt16](repeating: 0x3f80, count: 2048)
        let expected = doubleDots(input: input, weights: weights,
                                  rows: 1, columns: 2048)
        #expect(expected == [1024])
        let actual = try project(input: input, weights: weights,
                                 rows: 1, columns: 2048)
        #expect(actual == expected)
        #expect(orderedDots(input: input, weights: weights,
                           rows: 1, columns: 2048) == [1])
    }

    @Test func recoversRoundedProductBitsAndKeepsExactSmallDots() throws {
        // The first product has more significant bits than Float can store.
        // After cancellation, its discarded bits are part of the result.
        let input: [Float] = [Float(1).nextUp, -1.0078125, 0]
        let weights: [UInt16] = [
            0x3f81, 0x3f80, 0, // 1.0078125 * nextUp(1) - 1.0078125
            0x3f80, 0, 0, // exact nextUp(1)
            0, 0, 0, // exact zero
        ]
        let expected = doubleDots(input: input, weights: weights, rows: 3, columns: 3)
        let actual = try project(input: input, weights: weights, rows: 3, columns: 3)
        #expect(actual == expected)
        #expect(expected[0] == Float(1.0078125 * Double(Float(1).nextUp) - 1.0078125))
        #expect(orderedDots(input: input, weights: weights,
                           rows: 3, columns: 3)[0] != expected[0])
        #expect(actual[1] == Float(1).nextUp)
        #expect(actual[2] == 0)
    }

    @Test func nonfiniteAndOverflowKeepOrderedFMAClasses() throws {
        let actual = try project(input: [1, 1], weights: [
            0x7f80, 0x3f80,
            0xff80, 0x3f80,
            0x7f80, 0xff80,
            0x7fc1, 0x3f80,
        ], rows: 4, columns: 2)
        #expect(actual[0].isInfinite && actual[0].sign == .plus)
        #expect(actual[1].isInfinite && actual[1].sign == .minus)
        #expect(actual[2].isNaN)
        #expect(actual[3].isNaN)

        let productOverflow = try project(
            input: [Float.greatestFiniteMagnitude, 1], weights: [0x4000, 0x3f80],
            rows: 1, columns: 2)
        #expect(productOverflow[0].isInfinite && productOverflow[0].sign == .plus)
        let sumOverflow = try project(
            input: [Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude],
            weights: [0x3f80, 0x3f80], rows: 1, columns: 2)
        #expect(sumOverflow[0].isInfinite && sumOverflow[0].sign == .plus)

        // A separate product can overflow while fused multiplication and
        // addition cancel to a finite result. Replay must preserve that result.
        let fusedCancellation = try project(
            input: [-Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude],
            weights: [0x3f80, 0x4000], rows: 1, columns: 2)
        #expect(fusedCancellation == [Float.greatestFiniteMagnitude])
    }

    private func doubleDots(input: [Float], weights: [UInt16],
                            rows: Int, columns: Int) -> [Float] {
        (0..<rows).map { row in
            var sum = Double(0)
            for column in 0..<columns {
                let weight = Float(bitPattern: UInt32(weights[row * columns + column]) << 16)
                sum += Double(weight) * Double(input[column])
            }
            return Float(sum)
        }
    }

    private func orderedDots(input: [Float], weights: [UInt16],
                            rows: Int, columns: Int) -> [Float] {
        (0..<rows).map { row in
            var sum = Float(0)
            for column in 0..<columns {
                let weight = Float(bitPattern: UInt32(weights[row * columns + column]) << 16)
                sum = sum.addingProduct(weight, input[column])
            }
            return sum
        }
    }

    private func project(input: [Float], weights: [UInt16],
                         rows: Int, columns: Int) throws -> [Float] {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "projection", rows: rows,
                                  columns: columns, bits: weights),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        let resident = try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [QwenBF16TensorSpec(
                name: "projection", shardName: source.shardName,
                role: .dense, rows: rows, columns: columns)],
            residencyBudget: UInt64(weights.count * 2))
        let inputBuffer = try #require(input.withUnsafeBytes { bytes in
            context.device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                      options: .storageModeShared)
        })
        let output = try #require(context.device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride, options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        try resident.encodeProjection(commandBuffer: command, tensorName: "projection",
                                      input: inputBuffer, tokenCount: 1, output: output)
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        return Array(UnsafeBufferPointer(
            start: output.contents().bindMemory(to: Float.self, capacity: rows), count: rows))
    }
}
