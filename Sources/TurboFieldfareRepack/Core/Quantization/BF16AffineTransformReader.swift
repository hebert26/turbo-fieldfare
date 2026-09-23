import Darwin
import Foundation

struct BF16AffineSourceRange: Sendable, Equatable {
    let path: String
    let offset: UInt64
    let length: UInt64

    init(path: String, offset: UInt64, length: UInt64) {
        self.path = path
        self.offset = offset
        self.length = length
    }
}

struct BF16AffineTransformCheckpoint: Sendable, Equatable {
    let sourceIdentity: String
    let elementsPerRow: Int
    let bitWidth: AffineBitWidth
    let committedSourceBytes: UInt64
    let committedElements: Int
    let committedGroups: Int

    static func start(
        sourceIdentity: String,
        elementsPerRow: Int,
        bitWidth: AffineBitWidth
    ) -> Self {
        .init(sourceIdentity: sourceIdentity, elementsPerRow: elementsPerRow,
              bitWidth: bitWidth, committedSourceBytes: 0,
              committedElements: 0, committedGroups: 0)
    }
}

struct BF16AffineTransformEmission: Sendable, Equatable {
    let groupIndex: Int
    let sourceElementRange: Range<Int>
    let quantized: BF16AffineQuantizedGroup
    let checkpointAfterEmission: BF16AffineTransformCheckpoint
}

struct BF16AffineTransformCompletion: Sendable, Equatable {
    let checkpoint: BF16AffineTransformCheckpoint
    let maximumScratchBytes: Int
}

enum BF16AffineTransformReaderError: Error, Sendable, Equatable {
    case invalidSourceIdentity
    case invalidElementCount(Int)
    case invalidElementsPerRow(Int)
    case invalidTileBytes(Int)
    case misalignedSourceBytes(UInt64)
    case sourceLengthMismatch(expected: UInt64, actual: UInt64)
    case arithmeticOverflow
    case invalidCheckpoint(BF16AffineTransformCheckpoint)
    case openFailed(path: String, errno: Int32,
                    checkpoint: BF16AffineTransformCheckpoint)
    case readFailed(path: String, errno: Int32,
                    checkpoint: BF16AffineTransformCheckpoint)
    case shortRead(path: String, offset: UInt64,
                   checkpoint: BF16AffineTransformCheckpoint)
    case cancelled(checkpoint: BF16AffineTransformCheckpoint)
    case cancellationCheckFailed(checkpoint: BF16AffineTransformCheckpoint)
    case quantizationFailed(checkpoint: BF16AffineTransformCheckpoint)
    case outputRejected(checkpoint: BF16AffineTransformCheckpoint)
}

/// Joins ordered, discontiguous BF16 source ranges into complete affine groups.
/// Only one configured tile and one group are resident, independent of tensor
/// length. The output sink must commit an emission atomically before returning.
enum BF16AffineTransformReader {
    static let maximumTileBytes = 1024 * 1024
    static let maximumGroupBytes = BF16AffineQuantizationPolicy.affineGroupSize * 2

    static func transform(
        ranges: [BF16AffineSourceRange],
        elementCount: Int,
        elementsPerRow: Int,
        sourceIdentity: String,
        as bitWidth: AffineBitWidth,
        tileBytes: Int,
        resumeFrom suppliedCheckpoint: BF16AffineTransformCheckpoint? = nil,
        cancellationCheck: () throws -> Void = {
            try Task.checkCancellation()
        },
        emit: (BF16AffineTransformEmission) throws -> Void
    ) throws -> BF16AffineTransformCompletion {
        guard !sourceIdentity.isEmpty else {
            throw BF16AffineTransformReaderError.invalidSourceIdentity
        }
        guard elementCount > 0 else {
            throw BF16AffineTransformReaderError.invalidElementCount(elementCount)
        }
        guard elementsPerRow > 0, elementsPerRow <= elementCount,
              elementCount.isMultiple(of: elementsPerRow) else {
            throw BF16AffineTransformReaderError.invalidElementsPerRow(elementsPerRow)
        }
        guard (1...maximumTileBytes).contains(tileBytes) else {
            throw BF16AffineTransformReaderError.invalidTileBytes(tileBytes)
        }

        let expectedBytes = try checkedMultiply(UInt64(elementCount), 2)
        var totalBytes: UInt64 = 0
        for range in ranges {
            _ = try checkedAdd(range.offset, range.length)
            totalBytes = try checkedAdd(totalBytes, range.length)
        }
        guard totalBytes.isMultiple(of: 2) else {
            throw BF16AffineTransformReaderError.misalignedSourceBytes(totalBytes)
        }
        guard totalBytes == expectedBytes else {
            throw BF16AffineTransformReaderError.sourceLengthMismatch(
                expected: expectedBytes, actual: totalBytes)
        }

        var checkpoint = suppliedCheckpoint
            ?? BF16AffineTransformCheckpoint.start(
                sourceIdentity: sourceIdentity,
                elementsPerRow: elementsPerRow,
                bitWidth: bitWidth)
        try validate(checkpoint: checkpoint,
                     sourceIdentity: sourceIdentity,
                     elementCount: elementCount,
                     elementsPerRow: elementsPerRow,
                     bitWidth: bitWidth)

        var bytesToSkip = checkpoint.committedSourceBytes
        var groupBytes: [UInt8] = []
        groupBytes.reserveCapacity(maximumGroupBytes)
        var tile = [UInt8](repeating: 0, count: tileBytes)

        for range in ranges {
            let skippedHere = min(bytesToSkip, range.length)
            bytesToSkip -= skippedHere
            var position = skippedHere
            guard position < range.length else { continue }

            let descriptor = open(range.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            guard descriptor >= 0 else {
                throw BF16AffineTransformReaderError.openFailed(
                    path: range.path, errno: errno, checkpoint: checkpoint)
            }
            defer { close(descriptor) }

            while position < range.length {
                try checkCancellation(cancellationCheck, checkpoint: checkpoint)
                let remaining = range.length - position
                let requested = remaining > UInt64(tileBytes)
                    ? tileBytes : Int(remaining)
                let absoluteOffset = try checkedAdd(range.offset, position)
                guard absoluteOffset <= UInt64(Int64.max) else {
                    throw BF16AffineTransformReaderError.arithmeticOverflow
                }

                let readCount: Int = tile.withUnsafeMutableBytes { buffer in
                    while true {
                        let result = Darwin.pread(
                            descriptor, buffer.baseAddress, requested,
                            off_t(absoluteOffset))
                        if result < 0, errno == EINTR { continue }
                        return result
                    }
                }
                guard readCount >= 0 else {
                    throw BF16AffineTransformReaderError.readFailed(
                        path: range.path, errno: errno, checkpoint: checkpoint)
                }
                guard readCount > 0 else {
                    throw BF16AffineTransformReaderError.shortRead(
                        path: range.path, offset: absoluteOffset,
                        checkpoint: checkpoint)
                }

                for byte in tile.prefix(readCount) {
                    groupBytes.append(byte)
                    let elementsIntoRow = checkpoint.committedElements % elementsPerRow
                    let remainingElementsInRow = elementsPerRow - elementsIntoRow
                    let groupElements = min(
                        BF16AffineQuantizationPolicy.affineGroupSize,
                        remainingElementsInRow)
                    if groupBytes.count == groupElements * 2 {
                        try checkCancellation(cancellationCheck, checkpoint: checkpoint)
                        let quantized: BF16AffineQuantizedGroup
                        do {
                            quantized = try StreamingBF16AffineQuantizer.quantize(
                                bf16LittleEndian: groupBytes, as: bitWidth)
                        } catch {
                            throw BF16AffineTransformReaderError.quantizationFailed(
                                checkpoint: checkpoint)
                        }
                        let nextElements = checkpoint.committedElements + groupElements
                        let nextBytes = try checkedMultiply(UInt64(nextElements), 2)
                        let nextCheckpoint = BF16AffineTransformCheckpoint(
                            sourceIdentity: sourceIdentity,
                            elementsPerRow: elementsPerRow,
                            bitWidth: bitWidth,
                            committedSourceBytes: nextBytes,
                            committedElements: nextElements,
                            committedGroups: checkpoint.committedGroups + 1)
                        let emission = BF16AffineTransformEmission(
                            groupIndex: checkpoint.committedGroups,
                            sourceElementRange: checkpoint.committedElements..<nextElements,
                            quantized: quantized,
                            checkpointAfterEmission: nextCheckpoint)
                        do {
                            try emit(emission)
                        } catch {
                            throw BF16AffineTransformReaderError.outputRejected(
                                checkpoint: checkpoint)
                        }
                        // Advancing after sink success is the commit boundary.
                        checkpoint = nextCheckpoint
                        groupBytes.removeAll(keepingCapacity: true)
                    }
                }
                position = try checkedAdd(position, UInt64(readCount))
            }
        }

        guard bytesToSkip == 0,
              groupBytes.isEmpty,
              checkpoint.committedElements == elementCount,
              checkpoint.committedSourceBytes == expectedBytes else {
            throw BF16AffineTransformReaderError.shortRead(
                path: ranges.last?.path ?? "<no source range>",
                offset: checkpoint.committedSourceBytes,
                checkpoint: checkpoint)
        }
        return BF16AffineTransformCompletion(
            checkpoint: checkpoint,
            maximumScratchBytes: tileBytes + maximumGroupBytes
                + StreamingBF16AffineQuantizer.maximumScratchPayloadBytes)
    }

    private static func validate(
        checkpoint: BF16AffineTransformCheckpoint,
        sourceIdentity: String,
        elementCount: Int,
        elementsPerRow: Int,
        bitWidth: AffineBitWidth
    ) throws {
        guard checkpoint.sourceIdentity == sourceIdentity,
              checkpoint.elementsPerRow == elementsPerRow,
              checkpoint.bitWidth == bitWidth,
              checkpoint.committedElements >= 0,
              checkpoint.committedElements <= elementCount,
              checkpoint.committedGroups >= 0 else {
            throw BF16AffineTransformReaderError.invalidCheckpoint(checkpoint)
        }
        let expectedBytes = try checkedMultiply(
            UInt64(checkpoint.committedElements), 2)
        let completeRows = checkpoint.committedElements / elementsPerRow
        let elementsIntoRow = checkpoint.committedElements % elementsPerRow
        guard elementsIntoRow == 0 || elementsIntoRow.isMultiple(
            of: BF16AffineQuantizationPolicy.affineGroupSize) else {
            throw BF16AffineTransformReaderError.invalidCheckpoint(checkpoint)
        }
        let groupsPerRow = elementsPerRow / BF16AffineQuantizationPolicy.affineGroupSize
            + (elementsPerRow.isMultiple(
                of: BF16AffineQuantizationPolicy.affineGroupSize) ? 0 : 1)
        let (completeRowGroups, overflow) = completeRows.multipliedReportingOverflow(
            by: groupsPerRow)
        guard !overflow else {
            throw BF16AffineTransformReaderError.arithmeticOverflow
        }
        let expectedGroups = completeRowGroups
            + elementsIntoRow / BF16AffineQuantizationPolicy.affineGroupSize
        guard checkpoint.committedSourceBytes == expectedBytes,
              checkpoint.committedGroups == expectedGroups else {
            throw BF16AffineTransformReaderError.invalidCheckpoint(checkpoint)
        }
    }

    private static func checkCancellation(
        _ check: () throws -> Void,
        checkpoint: BF16AffineTransformCheckpoint
    ) throws {
        do {
            try check()
        } catch is CancellationError {
            throw BF16AffineTransformReaderError.cancelled(checkpoint: checkpoint)
        } catch {
            throw BF16AffineTransformReaderError.cancellationCheckFailed(
                checkpoint: checkpoint)
        }
    }

    private static func checkedAdd(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw BF16AffineTransformReaderError.arithmeticOverflow }
        return value
    }

    private static func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw BF16AffineTransformReaderError.arithmeticOverflow }
        return value
    }
}
