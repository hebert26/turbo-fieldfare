import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct BF16AffineTransformReaderTests {
    @Test func discontiguousRangesAndLegalTilesProduceIdenticalCompleteGroups() throws {
        let path = try writeBF16File(repeating: 1, count: 65)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let ranges = [
            BF16AffineSourceRange(path: path, offset: 0, length: 3),
            BF16AffineSourceRange(path: path, offset: 3, length: 127),
        ]
        let small = try collect(ranges: ranges, elements: 65, rowWidth: 65, tileBytes: 5)
        let large = try collect(ranges: ranges, elements: 65, rowWidth: 65, tileBytes: 128)
        #expect(small.groups.map(\.packedValues) == large.groups.map(\.packedValues))
        #expect(small.groups.map(\.elementCount) == [64, 1])
        #expect(small.completion.checkpoint.committedElements == 65)
        #expect(small.completion.maximumScratchBytes == 5 + 128 + 640)
        #expect(large.completion.maximumScratchBytes == 128 + 128 + 640)
    }

    @Test func rowTailsNeverMergeAcrossRowsAndTilesAreInvariant() throws {
        // Two 65-element rows. A flattened grouping bug would emit [64, 64, 2].
        // Different constants make each row's retained BF16 bias independently observable.
        let path = try writeBF16Values(Array(repeating: 0x3f80, count: 65) + Array(repeating: 0x4000, count: 65))
        defer { try? FileManager.default.removeItem(atPath: path) }
        let ranges = [
            BF16AffineSourceRange(path: path, offset: 0, length: 127),
            BF16AffineSourceRange(path: path, offset: 127, length: 133),
        ]
        let small = try collect(ranges: ranges, elements: 130, rowWidth: 65, tileBytes: 7)
        let large = try collect(ranges: ranges, elements: 130, rowWidth: 65, tileBytes: 128)
        #expect(small.groups.map(\.elementCount) == [64, 1, 64, 1])
        #expect(small.groups.map(\.biasBF16) == [0x3f80, 0x3f80, 0x4000, 0x4000])
        #expect(small.groups.map(\.scaleBF16) == [0x3f80, 0x3f80, 0x3f80, 0x3f80])
        #expect(small.groups.map(\.packedValues) == large.groups.map(\.packedValues))
        #expect(small.groups.map(\.biasBF16) == large.groups.map(\.biasBF16))

        var cleanEmissions: [BF16AffineTransformEmission] = []
        _ = try BF16AffineTransformReader.transform(
            ranges: ranges, elementCount: 130, elementsPerRow: 65, sourceIdentity: "row-resume",
            as: .int8, tileBytes: 11, emit: { cleanEmissions.append($0) })
        #expect(cleanEmissions.map { $0.checkpointAfterEmission.committedElements } == [64, 65, 129, 130])
        for boundary in [64, 65, 129, 130] {
            let checkpoint = try #require(cleanEmissions.first {
                $0.checkpointAfterEmission.committedElements == boundary
            }?.checkpointAfterEmission)
            var resumed: [BF16AffineQuantizedGroup] = []
            _ = try BF16AffineTransformReader.transform(
                ranges: ranges, elementCount: 130, elementsPerRow: 65, sourceIdentity: "row-resume",
                as: .int8, tileBytes: 3, resumeFrom: checkpoint, emit: { resumed.append($0.quantized) })
            let combined = cleanEmissions.prefix { $0.checkpointAfterEmission.committedElements <= boundary }
                .map(\.quantized) + resumed
            #expect(combined.map(\.packedValues) == small.groups.map(\.packedValues))
            #expect(combined.map(\.scaleBF16) == small.groups.map(\.scaleBF16))
            #expect(combined.map(\.biasBF16) == small.groups.map(\.biasBF16))
        }
    }

    @Test func cancellationCommitsOnlyPriorGroupAndResumeMatchesCleanOutput() throws {
        let path = try writeBF16File(repeating: 1, count: 65)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let ranges = [BF16AffineSourceRange(path: path, offset: 0, length: 130)]
        let clean = try collect(ranges: ranges, elements: 65, rowWidth: 65, tileBytes: 128)
        var emitted: [BF16AffineTransformEmission] = []
        var checks = 0
        var cancelledCheckpoint: BF16AffineTransformCheckpoint?
        do {
            _ = try BF16AffineTransformReader.transform(
                ranges: ranges, elementCount: 65, elementsPerRow: 65, sourceIdentity: "same-input",
                as: .int8, tileBytes: 128, cancellationCheck: {
                    checks += 1
                    if checks == 3 { throw CancellationError() }
                }, emit: { emitted.append($0) })
            Issue.record("expected cancellation")
        } catch let error as BF16AffineTransformReaderError {
            guard case let .cancelled(checkpoint) = error else { Issue.record("unexpected: \(error)"); return }
            cancelledCheckpoint = checkpoint
        }
        let checkpoint = try #require(cancelledCheckpoint)
        #expect(emitted.count == 1)
        #expect(checkpoint.committedElements == 64)
        #expect(checkpoint.elementsPerRow == 65)
        let resumed = try collect(ranges: ranges, elements: 65, rowWidth: 65, tileBytes: 7, resume: checkpoint)
        #expect((emitted.map(\.quantized) + resumed.groups).map(\.packedValues) == clean.groups.map(\.packedValues))
        #expect(resumed.completion.checkpoint == clean.completion.checkpoint)
    }

    @Test func shortReadSinkFailureAndInvalidBoundariesDoNotReportCompletion() throws {
        let path = try writeRawFile([0x80, 0x3f])
        defer { try? FileManager.default.removeItem(atPath: path) }
        #expect(throws: BF16AffineTransformReaderError.self) {
            _ = try BF16AffineTransformReader.transform(ranges: [.init(path: path, offset: 0, length: 4)],
                elementCount: 2, elementsPerRow: 2, sourceIdentity: "short", as: .int4, tileBytes: 4, emit: { _ in })
        }
        #expect(throws: BF16AffineTransformReaderError.self) {
            _ = try BF16AffineTransformReader.transform(ranges: [.init(path: path, offset: 0, length: 2)],
                elementCount: 1, elementsPerRow: 1, sourceIdentity: "sink", as: .int4, tileBytes: 2,
                emit: { _ in throw TestSinkError.rejected })
        }
        #expect(throws: BF16AffineTransformReaderError.self) {
            _ = try BF16AffineTransformReader.transform(ranges: [.init(path: path, offset: 0, length: 1)],
                elementCount: 1, elementsPerRow: 1, sourceIdentity: "odd", as: .int8, tileBytes: 1, emit: { _ in })
        }
        #expect(throws: BF16AffineTransformReaderError.self) {
            _ = try BF16AffineTransformReader.transform(ranges: [.init(path: path, offset: 0, length: 2)],
                elementCount: 1, elementsPerRow: 0, sourceIdentity: "row", as: .int8, tileBytes: 1, emit: { _ in })
        }
    }

    @Test func everyByteSplitAndMismatchedCheckpointOrOffsetFailExplicitly() throws {
        let path = try writeBF16File(repeating: 1, count: 4)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let clean = try collect(ranges: [.init(path: path, offset: 0, length: 8)], elements: 4, rowWidth: 4, tileBytes: 8)
        for split in 0...8 {
            let output = try collect(ranges: [.init(path: path, offset: 0, length: UInt64(split)), .init(path: path, offset: UInt64(split), length: UInt64(8 - split))], elements: 4, rowWidth: 4, tileBytes: 3)
            #expect(output.groups.map(\.packedValues) == clean.groups.map(\.packedValues))
        }
        #expect(throws: BF16AffineTransformReaderError.self) {
            _ = try BF16AffineTransformReader.transform(ranges: [.init(path: path, offset: .max, length: 1)],
                elementCount: 1, elementsPerRow: 1, sourceIdentity: "overflow", as: .int8, tileBytes: 1, emit: { _ in })
        }
        let mismatched = BF16AffineTransformCheckpoint(sourceIdentity: "same-input", elementsPerRow: 2,
            bitWidth: .int4, committedSourceBytes: 0, committedElements: 0, committedGroups: 0)
        #expect(throws: BF16AffineTransformReaderError.self) {
            _ = try BF16AffineTransformReader.transform(ranges: [.init(path: path, offset: 0, length: 8)],
                elementCount: 4, elementsPerRow: 4, sourceIdentity: "same-input", as: .int8, tileBytes: 2,
                resumeFrom: mismatched, emit: { _ in })
        }
        var emissions = 0
        #expect(throws: BF16AffineTransformReaderError.self) {
            _ = try BF16AffineTransformReader.transform(ranges: [.init(path: path, offset: 0, length: 8)],
                elementCount: 4, elementsPerRow: 4, sourceIdentity: "cancel-mid-group", as: .int8, tileBytes: 2,
                cancellationCheck: { throw CancellationError() }, emit: { _ in emissions += 1 })
        }
        #expect(emissions == 0)
    }

    private enum TestSinkError: Error { case rejected }

    private func collect(ranges: [BF16AffineSourceRange], elements: Int, rowWidth: Int, tileBytes: Int,
                         resume: BF16AffineTransformCheckpoint? = nil) throws -> (groups: [BF16AffineQuantizedGroup], completion: BF16AffineTransformCompletion) {
        var groups: [BF16AffineQuantizedGroup] = []
        let completion = try BF16AffineTransformReader.transform(ranges: ranges, elementCount: elements,
            elementsPerRow: rowWidth, sourceIdentity: "same-input", as: .int8, tileBytes: tileBytes,
            resumeFrom: resume, emit: { groups.append($0.quantized) })
        return (groups, completion)
    }

    private func writeBF16File(repeating value: UInt16, count: Int) throws -> String {
        try writeBF16Values(Array(repeating: value, count: count))
    }

    private func writeBF16Values(_ values: [UInt16]) throws -> String {
        try writeRawFile(values.flatMap { [UInt8($0 & 0xff), UInt8($0 >> 8)] })
    }

    private func writeRawFile(_ bytes: [UInt8]) throws -> String {
        let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("turbofieldfare-bf16-reader-\(UUID().uuidString)")
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }
}
