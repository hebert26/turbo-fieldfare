import Foundation
import Metal
import Testing
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfare

/// Regression coverage for the source-runner embedding shortcut. The fixture
/// uses the same complete-row chunking with a deliberately small injected cap
/// so the test remains tiny while comparing the two actual Metal entry points
/// without reading model weights.
@Suite(.serialized) struct QwenBF16EmbeddingChunkSelectionTests {
    @Test func immutableEmbeddingMatchesGeneralAtChunkEdgesAndDuplicateIDs() throws {
        let fixture = try EmbeddingChunkFixture.make()
        defer { fixture.source.remove() }
        let context = try MetalContext()
        let weights = try makeWeights(context: context, fixture: fixture)

        let chunks = weights.inspectedChunks.filter { $0.name == fixture.tensorName }
        #expect(chunks.map(\.firstRow) == [0, fixture.rowsPerChunk,
                                            fixture.rowsPerChunk * 2])
        #expect(chunks.map(\.rowCount) == [fixture.rowsPerChunk,
                                           fixture.rowsPerChunk, 1])
        #expect(chunks.map { $0.buffer.length }
            == [fixture.chunkBytes, fixture.chunkBytes, fixture.rowBytes])

        // The sparse case selects the first and final chunks while leaving the
        // middle chunk unbound. The second case touches every edge and keeps a
        // duplicate ID in both a boundary and final row.
        let cases: [[UInt32]] = [
            [0, UInt32(fixture.lastRow), UInt32(fixture.lastRow)],
            [UInt32(fixture.rowsPerChunk - 1), UInt32(fixture.rowsPerChunk),
             UInt32(fixture.rowsPerChunk),
             UInt32(fixture.rowsPerChunk * 2 - 1), UInt32(fixture.lastRow)],
        ]
        for ids in cases {
            let general = try encodeEmbedding(weights: weights, context: context,
                                               ids: ids, immutable: false)
            let immutable = try encodeEmbedding(weights: weights, context: context,
                                                 ids: ids, immutable: true)
            let expected = independentEmbeddingBits(ids: ids, bits: fixture.bits,
                                                    columns: fixture.columns)
            #expect(general == expected)
            #expect(immutable == expected)
            #expect(general == immutable,
                    "immutable source path changed BF16 embedding bits")
        }
    }

    @Test func generalEmbeddingReadsAValidIDMutationBeforeGPUExecution() throws {
        let fixture = try EmbeddingChunkFixture.make()
        defer { fixture.source.remove() }
        let context = try MetalContext()
        let weights = try makeWeights(context: context, fixture: fixture)
        let initialIDs: [UInt32] = [0, UInt32(fixture.rowsPerChunk - 1)]
        let mutatedIDs: [UInt32] = [UInt32(fixture.rowsPerChunk),
                                    UInt32(fixture.lastRow)]
        let tokenBuffer = try sharedBuffer(initialIDs, device: context.device)
        let output = try sharedBuffer(
            [Float](repeating: -17, count: initialIDs.count * fixture.columns),
            device: context.device)
        let command = try #require(context.queue.makeCommandBuffer())
        try weights.encodeEmbedding(commandBuffer: command,
                                    tensorName: fixture.tensorName,
                                    tokenIDs: tokenBuffer,
                                    tokenCount: initialIDs.count,
                                    output: output)

        overwrite(tokenBuffer, with: mutatedIDs)
        try finish(command)
        let actual = readFloatBits(output,
                                   count: mutatedIDs.count * fixture.columns)
        let expected = independentEmbeddingBits(ids: mutatedIDs, bits: fixture.bits,
                                                columns: fixture.columns)
        #expect(actual == expected,
                "general callers must retain pre-submission valid-ID mutation")
    }

    @Test func generalEmbeddingStillRejectsNegativeIDBeforeEncoding() throws {
        let fixture = try EmbeddingChunkFixture.make()
        defer { fixture.source.remove() }
        let context = try MetalContext()
        let weights = try makeWeights(context: context, fixture: fixture)
        // The public Metal buffer carries UInt32 IDs. A signed -1 from a
        // caller reaches this boundary as all-one bits and must stay rejected.
        let tokenBuffer = try sharedBuffer([Int32(-1)], device: context.device)
        let output = try sharedBuffer([Float](repeating: 23, count: fixture.columns),
                                       device: context.device)
        let command = try #require(context.queue.makeCommandBuffer())
        do {
            try weights.encodeEmbedding(commandBuffer: command,
                                        tensorName: fixture.tensorName,
                                        tokenIDs: tokenBuffer,
                                        tokenCount: 1,
                                        output: output)
            Issue.record("negative embedding ID must be rejected")
        } catch let error as QwenBF16WeightError {
            guard case .invalidGeometry = error else {
                Issue.record("unexpected embedding error: \(error)")
                return
            }
        }
        #expect(command.status == .notEnqueued)
        #expect(readFloatBits(output, count: fixture.columns)
            == [Float(23).bitPattern, Float(23).bitPattern])
    }
}

private struct EmbeddingChunkFixture {
    // Production defaults to 8 MiB. A test-only 64-byte cap exercises the
    // identical row-boundary and selected-chunk arithmetic without a massive
    // synthetic payload.
    private static let chunkByteLimit = 64
    private static let columnCount = 2
    private static let rowByteCount = columnCount * MemoryLayout<UInt16>.stride
    private static let fullChunkRows = chunkByteLimit / rowByteCount
    private static let totalRows = fullChunkRows * 2 + 1
    static let publicName = "embedding"

    let source: QwenBF16SyntheticSource
    let bits: [UInt16]

    var lastRow: Int { Self.totalRows - 1 }
    var chunkBytes: Int { Self.chunkByteLimit }
    var columns: Int { Self.columnCount }
    var rowBytes: Int { Self.rowByteCount }
    var rowsPerChunk: Int { Self.fullChunkRows }
    var rows: Int { Self.totalRows }
    var tensorName: String { Self.publicName }

    static func make() throws -> Self {
        let bits = makeBits()
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: publicName, rows: totalRows,
                                  columns: columnCount, bits: bits),
        ])
        return Self(source: source, bits: bits)
    }

    private static func makeBits() -> [UInt16] {
        var bits: [UInt16] = []
        bits.reserveCapacity(totalRows * columnCount)
        for row in 0..<totalRows {
            let low = UInt16(row & 0x7f)
            bits.append(0x3f80 | low)
            bits.append(0xbf80 | low)
        }
        return bits
    }
}

private func makeWeights(context: MetalContext,
                         fixture: EmbeddingChunkFixture) throws -> QwenBF16Weights {
    try QwenBF16Weights(
        context: context,
        source: fixture.source.handle,
        specifications: [QwenBF16TensorSpec(
            name: fixture.tensorName,
            shardName: fixture.source.shardName,
            role: .embedding,
            rows: fixture.rows,
            columns: fixture.columns)],
        residencyBudget: UInt64(fixture.bits.count * MemoryLayout<UInt16>.stride),
        maximumChunkBytes: UInt64(fixture.chunkBytes),
        checkpoint: { _ in })
}

private func encodeEmbedding(weights: QwenBF16Weights,
                             context: MetalContext,
                             ids: [UInt32],
                             immutable: Bool) throws -> [UInt32] {
    let tokenBuffer = try sharedBuffer(ids, device: context.device)
    let output = try sharedBuffer([Float](repeating: -29, count: ids.count * 2),
                                   device: context.device)
    let command = try #require(context.queue.makeCommandBuffer())
    if immutable {
        try weights.encodeEmbeddingFromImmutableIDs(
            commandBuffer: command, tensorName: EmbeddingChunkFixture.publicName,
            tokenIDs: tokenBuffer, tokenCount: ids.count, output: output)
    } else {
        try weights.encodeEmbedding(
            commandBuffer: command, tensorName: EmbeddingChunkFixture.publicName,
            tokenIDs: tokenBuffer, tokenCount: ids.count, output: output)
    }
    try finish(command)
    return readFloatBits(output, count: ids.count * 2)
}

private func independentEmbeddingBits(ids: [UInt32], bits: [UInt16], columns: Int)
    -> [UInt32] {
    ids.flatMap { id in
        let start = Int(id) * columns
        return bits[start..<(start + columns)].map { UInt32($0) << 16 }
    }
}

private func sharedBuffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
    guard !values.isEmpty,
          let buffer = device.makeBuffer(bytes: values,
                                         length: values.count * MemoryLayout<T>.stride,
                                         options: .storageModeShared) else {
        throw EmbeddingChunkTestError.bufferAllocation
    }
    return buffer
}

private func overwrite<T>(_ buffer: MTLBuffer, with values: [T]) {
    values.withUnsafeBytes { raw in
        guard let baseAddress = raw.baseAddress else { return }
        buffer.contents().copyMemory(from: baseAddress, byteCount: raw.count)
    }
}

private func readFloatBits(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
    let values = buffer.contents().assumingMemoryBound(to: Float.self)
    return UnsafeBufferPointer(start: values, count: count).map(\.bitPattern)
}

private func finish(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, command.error == nil else {
        throw EmbeddingChunkTestError.commandFailed
    }
}

private enum EmbeddingChunkTestError: Error {
    case bufferAllocation
    case commandFailed
}
