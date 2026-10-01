import Foundation
import Metal
import Testing
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfare

/// The staged head encoder may place all row chunks in one concurrent Metal
/// encoder. These cases compare that path with the unchanged serial `.dense`
/// encoder using identical BF16 data and an independent operation-order oracle.
@Suite(.serialized) struct QwenBF16HeadConcurrentChunkTests {
    @Test func multichunkHeadMatchesSerialDenseForScalarAndCooperativeFamilies() throws {
        let cases = [
            HeadProjectionCase(rows: 5, columns: 5, chunkRows: 2, tokenCount: 3),
            HeadProjectionCase(rows: 512, columns: 2048, chunkRows: 64, tokenCount: 3),
        ]

        for fixture in cases {
            let source = try QwenBF16SyntheticSource.make(tensors: [
                QwenBF16LiteralTensor(name: "head", rows: fixture.rows,
                                      columns: fixture.columns, bits: fixture.weights),
                QwenBF16LiteralTensor(name: "dense", rows: fixture.rows,
                                      columns: fixture.columns, bits: fixture.weights),
            ])
            defer { source.remove() }
            let context = try MetalContext()
            let weights = try QwenBF16Weights(
                context: context,
                source: source.handle,
                specifications: [
                    QwenBF16TensorSpec(name: "head", shardName: source.shardName,
                                       role: .head, rows: fixture.rows,
                                       columns: fixture.columns),
                    QwenBF16TensorSpec(name: "dense", shardName: source.shardName,
                                       role: .dense, rows: fixture.rows,
                                       columns: fixture.columns),
                ],
                residencyBudget: UInt64(fixture.weights.count * 2
                                         * MemoryLayout<UInt16>.stride),
                maximumChunkBytes: UInt64(fixture.chunkRows * fixture.columns
                                          * MemoryLayout<UInt16>.stride),
                checkpoint: { _ in })

            let headChunks = weights.inspectedChunks.filter { $0.name == "head" }
            let denseChunks = weights.inspectedChunks.filter { $0.name == "dense" }
            let expectedChunkRows = stride(from: 0, to: fixture.rows,
                                            by: fixture.chunkRows).map {
                min(fixture.chunkRows, fixture.rows - $0)
            }
            #expect(headChunks.map(\.rowCount) == expectedChunkRows)
            #expect(denseChunks.map(\.rowCount) == expectedChunkRows)
            #expect(headChunks.count > 1)

            let head = try project(weights: weights, context: context,
                                   tensorName: "head", input: fixture.input,
                                   rows: fixture.rows, columns: fixture.columns,
                                   tokenCount: fixture.tokenCount)
            let dense = try project(weights: weights, context: context,
                                    tensorName: "dense", input: fixture.input,
                                    rows: fixture.rows, columns: fixture.columns,
                                    tokenCount: fixture.tokenCount)
            let expected = fixture.independentOutputBits()
            #expect(head == dense,
                    "head concurrent chunks changed the serial dense output")
            #expect(head == expected,
                    "head output differs from the independent ordered oracle")
        }
    }
}

private struct HeadProjectionCase {
    let rows: Int
    let columns: Int
    let chunkRows: Int
    let tokenCount: Int
    let weights: [UInt16]
    let input: [Float]

    init(rows: Int, columns: Int, chunkRows: Int, tokenCount: Int) {
        self.rows = rows
        self.columns = columns
        self.chunkRows = chunkRows
        self.tokenCount = tokenCount
        self.weights = Self.makeWeights(rows: rows, columns: columns)
        self.input = Self.makeInput(tokenCount: tokenCount, columns: columns)
    }

    func independentOutputBits() -> [UInt32] {
        let values: [Float]
        if rows >= 512 && columns >= 512 && columns.isMultiple(of: 64) {
            values = sourceLargeOrderedDots(input: input, weights: weights,
                                             rows: rows, columns: columns,
                                             tokenCount: tokenCount)
        } else {
            values = stableDots(input: input, weights: weights,
                                rows: rows, columns: columns,
                                tokenCount: tokenCount)
        }
        return values.map(\.bitPattern)
    }

    private static func makeWeights(rows: Int, columns: Int) -> [UInt16] {
        (0..<(rows * columns)).map { index in
            let row = index / columns
            let column = index % columns
            let low = UInt16((row &* 17 &+ column &* 3) & 0x1f)
            let sign: UInt16 = (row + column).isMultiple(of: 2) ? 0 : 0x8000
            return sign | 0x3f80 | low
        }
    }

    private static func makeInput(tokenCount: Int, columns: Int) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(tokenCount * columns)
        for token in 0..<tokenCount {
            for column in 0..<columns {
                let magnitude = 0.25 + Float((token * 11 + column * 7) % 31) * 0.015625
                result.append((token + column).isMultiple(of: 2)
                    ? magnitude : -magnitude)
            }
        }
        return result
    }
}

private func stableDots(input: [Float], weights: [UInt16], rows: Int,
                       columns: Int, tokenCount: Int) -> [Float] {
    var result = [Float](repeating: 0, count: tokenCount * rows)
    for token in 0..<tokenCount {
        let inputBase = token * columns
        for row in 0..<rows {
            let rowBase = row * columns
            var sum: Float = 0
            var correction: Float = 0
            for column in 0..<columns {
                let weight = Float(bitPattern: UInt32(weights[rowBase + column]) << 16)
                let product = weight * input[inputBase + column]
                let next = sum + product
                let productError = (-product).addingProduct(
                    weight, input[inputBase + column])
                let sumError = abs(sum) >= abs(product)
                    ? (sum - next) + product
                    : (product - next) + sum
                correction += productError + sumError
                sum = next
            }
            result[token * rows + row] = sum + correction
        }
    }
    return result
}

private func sourceLargeOrderedDots(input: [Float], weights: [UInt16], rows: Int,
                                    columns: Int, tokenCount: Int) -> [Float] {
    var result = [Float](repeating: 0, count: tokenCount * rows)
    for token in 0..<tokenCount {
        let inputBase = token * columns
        for row in 0..<rows {
            let rowBase = row * columns
            var partial = [Float](repeating: 0, count: 64)
            for column in 0..<columns {
                let weight = Float(bitPattern: UInt32(weights[rowBase + column]) << 16)
                let stream = column & 63
                partial[stream] = partial[stream].addingProduct(
                    weight, input[inputBase + column])
            }
            var collapsed = [Float](repeating: 0, count: 16)
            for index in 0..<16 {
                let first = partial[index] + partial[index + 16]
                let second = first + partial[index + 32]
                collapsed[index] = second + partial[index + 48]
            }
            var groups = [Float](repeating: 0, count: 4)
            for index in 0..<4 {
                let base = index * 4
                let first = collapsed[base] + collapsed[base + 1]
                let second = first + collapsed[base + 2]
                groups[index] = second + collapsed[base + 3]
            }
            let first = groups[0] + groups[1]
            let second = first + groups[2]
            result[token * rows + row] = second + groups[3]
        }
    }
    return result
}

private func project(weights: QwenBF16Weights, context: MetalContext,
                     tensorName: String, input: [Float], rows: Int,
                     columns: Int, tokenCount: Int) throws -> [UInt32] {
    let inputBuffer = try sharedBuffer(input, device: context.device)
    let output = try sharedBuffer([Float](repeating: -73,
                                          count: tokenCount * rows),
                                   device: context.device)
    let command = try #require(context.queue.makeCommandBuffer())
    try weights.encodeProjection(commandBuffer: command, tensorName: tensorName,
                                 input: inputBuffer, tokenCount: tokenCount,
                                 output: output)
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, command.error == nil else {
        throw HeadConcurrentChunkTestError.commandFailed
    }
    let values = output.contents().assumingMemoryBound(to: Float.self)
    return UnsafeBufferPointer(start: values, count: tokenCount * rows)
        .map(\.bitPattern)
}

private func sharedBuffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
    guard !values.isEmpty,
          let buffer = device.makeBuffer(bytes: values,
                                         length: values.count * MemoryLayout<T>.stride,
                                         options: .storageModeShared) else {
        throw HeadConcurrentChunkTestError.bufferAllocation
    }
    return buffer
}

private enum HeadConcurrentChunkTestError: Error {
    case bufferAllocation
    case commandFailed
}
