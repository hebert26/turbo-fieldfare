import Darwin
import Foundation
import TurboFieldfareFormat

struct TransformedTensorWriteRequest: Sendable {
    let requestIndex: UInt64
    let requestID: String
    let tensorName: String
    let source: SourceTensor
    /// Present only when the initial full-source proof is also this request's
    /// exact range. Routed slices rely on the consumed-byte chain instead.
    let sourcePayloadSHA256: String?
    let destinationPath: String
    let destinationFileSize: UInt64
    let fileOffset: UInt64
    let storage: BF16AffineTensorStorage
    let affineComponents: QwenAffineComponentLayout?
}

struct TransformedTensorWriterOperations {
    var read: (_ fd: Int32, _ path: String, _ offset: UInt64, _ count: Int) throws -> Data
    var write: WriterCore.PositionedWrite
    var sync: (_ fd: Int32, _ path: String) throws -> Void
    var cancellationCheck: () throws -> Void

    static var production: Self { Self(
        read: { fd, path, offset, count in
            var data = Data(count: count)
            try data.withUnsafeMutableBytes { raw in
                guard let base = raw.baseAddress else { return }
                try Posix.preadAll(fd: fd, path: path, buf: base, count: count, offset: offset)
            }
            return data
        },
        write: { fd, path, bytes, offset in
            guard let base = bytes.baseAddress else { return 0 }
            try Posix.pwriteAll(fd: fd, path: path, buf: base, count: bytes.count, offset: offset)
            return bytes.count
        },
        sync: { try Posix.fsync($0, path: $1) },
        cancellationCheck: { try Task.checkCancellation() }) }
}

struct TransformedTensorWriteResult: Sendable, Equatable {
    let progress: RemoteTransformProgress
    let maximumScratchBytes: Int

    var completedUnitCount: UInt64 { progress.totalCompletedUnitCount }
}

enum TransformedTensorWriter {
    static let converterVersion = "TurboFieldfareRepack/QwenBF16Affine/v2"
    private static let proofDomain = "TurboFieldfareRepack/transform-progress/v2"

    static func sourcePayloadSHA256(
        source: SourceTensor,
        operations: TransformedTensorWriterOperations = .production
    ) throws -> String {
        let fd = try Posix.openReadNoFollow(source.shardPath)
        defer { close(fd) }
        let end = source.absoluteOffset.addingReportingOverflow(source.sizeBytes)
        guard !end.overflow,
              try Posix.fileSize(fd: fd, path: source.shardPath) >= end.partialValue else {
            throw RepackError.configurationInvalid(detail: "source tensor range exceeds file")
        }
        var hasher = Sha256Stream()
        var offset = source.absoluteOffset
        var remaining = source.sizeBytes
        while remaining > 0 {
            try operations.cancellationCheck()
            let count = Int(min(UInt64(WriterCore.tileBytes), remaining))
            let data = try operations.read(fd, source.shardPath, offset, count)
            guard data.count == count else {
                throw RepackError.preadShort(
                    path: source.shardPath, expected: count, got: data.count, errno: 0)
            }
            data.withUnsafeBytes { hasher.update($0) }
            offset += UInt64(count)
            remaining -= UInt64(count)
        }
        return hasher.finalizeHexString()
    }

    static func write(
        _ request: TransformedTensorWriteRequest,
        progress initialProgress: RemoteTransformProgress,
        resumeUnitCount: UInt64 = 0,
        audit: RepackAudit,
        affineDurabilityInterval: UInt64 = 1,
        operations: TransformedTensorWriterOperations = .production,
        commit: (RemoteTransformProgress) throws -> Void = { _ in }
    ) throws -> TransformedTensorWriteResult {
        guard request.source.dtype == .bf16,
              request.source.sizeBytes.isMultiple(of: 2),
              !request.tensorName.isEmpty else {
            throw RepackError.configurationInvalid(detail: "invalid BF16 transform request")
        }
        if let expectedSourceDigest = request.sourcePayloadSHA256 {
            let payloadDigest = try sourcePayloadSHA256(
                source: request.source, operations: operations)
            guard payloadDigest == expectedSourceDigest.lowercased() else {
                throw RepackError.installStateIncompatible(
                    detail: "transformed source payload changed")
            }
        }
        let sourceByteCount = try checkedInt(request.source.sizeBytes)
        let elementCount = sourceByteCount / 2
        guard let elementsPerRow64 = request.source.shape.last,
              elementsPerRow64 > 0,
              elementCount.isMultiple(of: try checkedInt(elementsPerRow64)) else {
            throw RepackError.configurationInvalid(detail: "invalid transform row shape")
        }
        let elementsPerRow = try checkedInt(elementsPerRow64)
        try validate(request: request, elementCount: elementCount,
                     elementsPerRow: elementsPerRow)
        let completedElements = try completedElements(
            unitCount: resumeUnitCount, request: request,
            elementCount: elementCount, elementsPerRow: elementsPerRow)
        var progress = initialProgress

        let sourceFD = try Posix.openReadNoFollow(request.source.shardPath)
        defer { close(sourceFD) }
        let destinationFD = try Posix.openExistingRW(request.destinationPath)
        defer { close(destinationFD) }
        guard try Posix.fileSize(fd: destinationFD, path: request.destinationPath)
            == request.destinationFileSize else {
            throw RepackError.installStateIncompatible(
                detail: "transformed destination size changed")
        }
        switch request.storage {
        case .retainedBF16:
            guard resumeUnitCount <= (try unitCount(request: request)) else {
                throw RepackError.installStateIncompatible(
                    detail: "transformed progress exceeds its request")
            }
            var position = try checkedInt(
                try multiply(resumeUnitCount, UInt64(WriterCore.tileBytes)))
            while position < sourceByteCount {
                try operations.cancellationCheck()
                let count = min(WriterCore.tileBytes, sourceByteCount - position)
                let sourceOffset = try add(
                    request.source.absoluteOffset, UInt64(position))
                let destinationOffset = try add(request.fileOffset, UInt64(position))
                let data = try operations.read(
                    sourceFD, request.source.shardPath, sourceOffset, count)
                guard data.count == count else {
                    throw RepackError.preadShort(
                        path: request.source.shardPath,
                        expected: count, got: data.count, errno: 0)
                }
                try WriterCore.pwriteTransformedComponent(
                    data, dstFd: destinationFD, dstPath: request.destinationPath,
                    dstOffset: destinationOffset,
                    audit: audit, write: operations.write)
                audit.recordRead(bytes: count)
                try operations.sync(destinationFD, request.destinationPath)
                progress = try advanced(
                    progress, request: request,
                    unitIndex: UInt64(position / WriterCore.tileBytes),
                    sourceOffset: sourceOffset, sourceData: data,
                    destinationPieces: [(destinationOffset, data)])
                try commit(progress)
                try operations.cancellationCheck()
                position += count
            }
            return .init(progress: progress, maximumScratchBytes: WriterCore.tileBytes)
        case .affineInt4, .affineInt8:
            guard let components = request.affineComponents,
                  let bitWidth = request.storage.affineBitWidth else {
                throw RepackError.configurationInvalid(
                    detail: "affine transform has no component layout")
            }
            guard affineDurabilityInterval > 0 else {
                throw RepackError.configurationInvalid(
                    detail: "affine durability interval must be positive")
            }
            let totalUnitCount = try unitCount(request: request)
            var unitIndex = resumeUnitCount
            var element = completedElements
            while element < elementCount {
                try operations.cancellationCheck()
                let rowPosition = element % elementsPerRow
                let groupElements = min(
                    BF16AffineQuantizationPolicy.affineGroupSize,
                    elementsPerRow - rowPosition)
                let sourceCount = groupElements * 2
                let sourceOffset = try add(
                    request.source.absoluteOffset,
                    try multiply(UInt64(element), 2))
                let data = try operations.read(
                    sourceFD, request.source.shardPath, sourceOffset, sourceCount)
                guard data.count == sourceCount else {
                    throw RepackError.preadShort(
                        path: request.source.shardPath,
                        expected: sourceCount, got: data.count, errno: 0)
                }
                audit.recordRead(bytes: sourceCount)
                let quantized = try StreamingBF16AffineQuantizer.quantize(
                    bf16LittleEndian: Array(data), as: bitWidth)
                let groupIndex = try checkedInt(unitIndex)
                let relativeValueOffset = try packedValueOffset(
                    element: element, elementsPerRow: elementsPerRow,
                    bitWidth: bitWidth)
                let valuesOffset = try add(
                    try add(request.fileOffset, components.valuesOffset),
                    relativeValueOffset)
                let groupParameterOffset = try multiply(UInt64(groupIndex), 2)
                let scalesOffset = try add(
                    try add(request.fileOffset, components.scalesOffset),
                    groupParameterOffset)
                let biasesOffset = try add(
                    try add(request.fileOffset, components.biasesOffset),
                    groupParameterOffset)
                let values = Data(quantized.packedValues)
                var scale = quantized.scaleBF16.littleEndian
                var bias = quantized.biasBF16.littleEndian
                try WriterCore.pwriteTransformedComponent(
                    values, dstFd: destinationFD, dstPath: request.destinationPath,
                    dstOffset: valuesOffset, audit: audit, write: operations.write)
                try withUnsafeBytes(of: &scale) {
                    try WriterCore.pwriteTransformedComponent(
                        Data($0), dstFd: destinationFD, dstPath: request.destinationPath,
                        dstOffset: scalesOffset, audit: audit, write: operations.write)
                }
                try withUnsafeBytes(of: &bias) {
                    try WriterCore.pwriteTransformedComponent(
                        Data($0), dstFd: destinationFD, dstPath: request.destinationPath,
                        dstOffset: biasesOffset, audit: audit, write: operations.write)
                }
                let scaleData = withUnsafeBytes(of: &scale) { Data($0) }
                let biasData = withUnsafeBytes(of: &bias) { Data($0) }
                progress = try advanced(
                    progress, request: request, unitIndex: unitIndex,
                    sourceOffset: sourceOffset, sourceData: data,
                    destinationPieces: [
                        (valuesOffset, values),
                        (scalesOffset, scaleData),
                        (biasesOffset, biasData),
                    ])
                let completedUnitCount = progress.completedUnitCount
                if completedUnitCount.isMultiple(of: affineDurabilityInterval)
                    || completedUnitCount == totalUnitCount {
                    try operations.sync(destinationFD, request.destinationPath)
                    try commit(progress)
                }
                try operations.cancellationCheck()
                unitIndex += 1
                element += groupElements
            }
            return .init(
                progress: progress,
                maximumScratchBytes: BF16AffineTransformReader.maximumGroupBytes
                    + StreamingBF16AffineQuantizer.maximumScratchPayloadBytes)
        case .omittedMTP:
            throw RepackError.configurationInvalid(
                detail: "omitted MTP tensor reached transformed writer")
        }
    }

    private static func validate(
        request: TransformedTensorWriteRequest,
        elementCount: Int,
        elementsPerRow: Int
    ) throws {
        let sourceProduct = try request.source.shape.reduce(UInt64(1)) { partial, extent in
            let product = partial.multipliedReportingOverflow(by: extent)
            guard !product.overflow else {
                throw RepackError.configurationInvalid(detail: "source shape overflows")
            }
            return product.partialValue
        }
        guard sourceProduct == UInt64(elementCount) else {
            throw RepackError.configurationInvalid(detail: "source shape byte count disagrees")
        }
        let end = request.fileOffset.addingReportingOverflow(
            request.storage == .retainedBF16
                ? request.source.sizeBytes : request.affineComponents?.totalSize ?? UInt64.max)
        guard !end.overflow, end.partialValue <= request.destinationFileSize else {
            throw RepackError.configurationInvalid(detail: "transformed destination range exceeds file")
        }
        if let bitWidth = request.storage.affineBitWidth {
            let expected = try QwenRepackPlanner.affineComponentSizes(
                shape: request.source.shape, bitWidth: bitWidth)
            guard expected == request.affineComponents else {
                throw RepackError.configurationInvalid(detail: "affine component layout disagrees")
            }
        } else if request.affineComponents != nil || elementsPerRow <= 0 {
            throw RepackError.configurationInvalid(detail: "retained tensor has affine components")
        }
    }

    struct ResumePosition: Sendable, Equatable {
        let requestIndex: Int
        let completedUnitCount: UInt64
    }

    static func initialProgress(
        binding: RemoteTransformBinding,
        firstRequestID: String
    ) -> RemoteTransformProgress {
        RemoteTransformProgress(
            requestIndex: 0, requestID: firstRequestID,
            completedUnitCount: 0, totalCompletedUnitCount: 0,
            sourceChainSHA256: seed(domain: "source", binding: binding),
            destinationChainSHA256: seed(domain: "destination", binding: binding))
    }

    /// Replays exactly the saved prefix with read-only descriptors. Callers
    /// must not open a destination read-write until this succeeds.
    static func validateProgress(
        _ saved: RemoteTransformProgress,
        binding: RemoteTransformBinding,
        requests: [TransformedTensorWriteRequest],
        operations: TransformedTensorWriterOperations = .production
    ) throws -> ResumePosition {
        guard saved.version == RemoteTransformProgress.currentVersion,
              saved.requestIndex < UInt64(requests.count) else {
            throw RepackError.installStateIncompatible(
                detail: "transformed progress exhausts its request schedule")
        }
        let savedRequestIndex = try checkedInt(saved.requestIndex)
        guard requests[savedRequestIndex].requestID == saved.requestID else {
            throw RepackError.installStateIncompatible(
                detail: "transformed progress request changed")
        }
        var replayed = initialProgress(
            binding: binding, firstRequestID: requests[0].requestID)
        for requestIndex in 0...savedRequestIndex {
            let request = requests[requestIndex]
            let available = try unitCount(request: request)
            let required = requestIndex == savedRequestIndex
                ? saved.completedUnitCount : available
            guard required <= available else {
                throw RepackError.installStateIncompatible(
                    detail: "transformed progress exceeds its request")
            }
            guard required > 0 else { continue }
            try withReadDescriptors(request: request) { sourceFD, destinationFD in
                for unitIndex in 0..<required {
                    try operations.cancellationCheck()
                    let geometry = try unitGeometry(
                        request: request, unitIndex: unitIndex)
                    let sourceData = try readExact(
                        fd: sourceFD, path: request.source.shardPath,
                        offset: geometry.sourceOffset, count: geometry.sourceBytes,
                        operations: operations)
                    var destinationPieces: [(UInt64, Data)] = []
                    for range in geometry.destinationRanges {
                        destinationPieces.append((range.0, try readExact(
                            fd: destinationFD, path: request.destinationPath,
                            offset: range.0, count: range.1,
                            operations: operations)))
                    }
                    replayed = try advanced(
                        replayed, request: request, unitIndex: unitIndex,
                        sourceOffset: geometry.sourceOffset, sourceData: sourceData,
                        destinationPieces: destinationPieces)
                }
            }
        }
        guard replayed.totalCompletedUnitCount == saved.totalCompletedUnitCount,
              replayed.sourceChainSHA256 == saved.sourceChainSHA256,
              replayed.destinationChainSHA256 == saved.destinationChainSHA256 else {
            throw RepackError.installStateIncompatible(
                detail: "transformed committed source or destination prefix changed")
        }
        let currentUnits = try unitCount(request: requests[savedRequestIndex])
        if saved.completedUnitCount == currentUnits {
            return .init(requestIndex: savedRequestIndex + 1, completedUnitCount: 0)
        }
        return .init(
            requestIndex: savedRequestIndex,
            completedUnitCount: saved.completedUnitCount)
    }

    static func unitCount(request: TransformedTensorWriteRequest) throws -> UInt64 {
        switch request.storage {
        case .retainedBF16:
            let tile = UInt64(WriterCore.tileBytes)
            return request.source.sizeBytes / tile
                + (request.source.sizeBytes.isMultiple(of: tile) ? 0 : 1)
        case .affineInt4, .affineInt8:
            guard let elementsPerRow = request.source.shape.last,
                  elementsPerRow > 0 else {
                throw RepackError.configurationInvalid(
                    detail: "invalid transform row shape")
            }
            guard request.source.sizeBytes.isMultiple(of: 2) else {
                throw RepackError.configurationInvalid(
                    detail: "invalid BF16 transform request")
            }
            let elementCount = request.source.sizeBytes / 2
            guard elementCount.isMultiple(of: elementsPerRow) else {
                throw RepackError.configurationInvalid(
                    detail: "invalid transform row shape")
            }
            let rows = elementCount / elementsPerRow
            let groupSize = UInt64(BF16AffineQuantizationPolicy.affineGroupSize)
            let groupsPerRow = elementsPerRow / groupSize
                + (elementsPerRow.isMultiple(of: groupSize) ? 0 : 1)
            return try multiply(rows, groupsPerRow)
        case .omittedMTP:
            throw RepackError.configurationInvalid(
                detail: "omitted MTP tensor reached transformed writer")
        }
    }

    private static func completedElements(
        unitCount completedUnitCount: UInt64,
        request: TransformedTensorWriteRequest,
        elementCount: Int,
        elementsPerRow: Int
    ) throws -> Int {
        guard request.storage.affineBitWidth != nil else { return 0 }
        let available = try unitCount(request: request)
        guard completedUnitCount <= available else {
            throw RepackError.installStateIncompatible(
                detail: "transformed progress is not a complete prefix")
        }
        if completedUnitCount == available { return elementCount }
        let geometry = try unitGeometry(
            request: request, unitIndex: completedUnitCount)
        guard geometry.sourceOffset >= request.source.absoluteOffset else {
            throw RepackError.configurationInvalid(
                detail: "transformed source offset is invalid")
        }
        return try checkedInt(
            (geometry.sourceOffset - request.source.absoluteOffset) / 2)
    }

    private struct UnitGeometry {
        let sourceOffset: UInt64
        let sourceBytes: Int
        let destinationRanges: [(UInt64, Int)]
    }

    private static func unitGeometry(
        request: TransformedTensorWriteRequest,
        unitIndex: UInt64
    ) throws -> UnitGeometry {
        guard unitIndex < (try unitCount(request: request)) else {
            throw RepackError.installStateIncompatible(
                detail: "transformed unit is outside its request")
        }
        switch request.storage {
        case .retainedBF16:
            let relative = try multiply(unitIndex, UInt64(WriterCore.tileBytes))
            let count = Int(min(
                UInt64(WriterCore.tileBytes), request.source.sizeBytes - relative))
            return .init(
                sourceOffset: try add(request.source.absoluteOffset, relative),
                sourceBytes: count,
                destinationRanges: [(try add(request.fileOffset, relative), count)])
        case .affineInt4, .affineInt8:
            guard let bitWidth = request.storage.affineBitWidth,
                  let components = request.affineComponents,
                  let elementsPerRow64 = request.source.shape.last else {
                throw RepackError.configurationInvalid(
                    detail: "affine transform has no component layout")
            }
            let elementsPerRow = try checkedInt(elementsPerRow64)
            let groupSize = UInt64(BF16AffineQuantizationPolicy.affineGroupSize)
            let groupsPerRow = elementsPerRow64 / groupSize
                + (elementsPerRow64.isMultiple(of: groupSize) ? 0 : 1)
            let rowStart = try multiply(
                unitIndex / groupsPerRow, elementsPerRow64)
            let groupStart = try multiply(unitIndex % groupsPerRow, groupSize)
            let element = try checkedInt(try add(rowStart, groupStart))
            let count = min(
                BF16AffineQuantizationPolicy.affineGroupSize,
                elementsPerRow - element % elementsPerRow)
            let valuesBytes = (count * bitWidth.rawValue + 7) / 8
            let valuesOffset = try add(
                try add(request.fileOffset, components.valuesOffset),
                try packedValueOffset(
                    element: element, elementsPerRow: elementsPerRow,
                    bitWidth: bitWidth))
            let parameterOffset = try multiply(unitIndex, 2)
            return .init(
                sourceOffset: try add(
                    request.source.absoluteOffset, try multiply(UInt64(element), 2)),
                sourceBytes: count * 2,
                destinationRanges: [
                    (valuesOffset, valuesBytes),
                    (try add(try add(request.fileOffset, components.scalesOffset),
                             parameterOffset), 2),
                    (try add(try add(request.fileOffset, components.biasesOffset),
                             parameterOffset), 2),
                ])
        case .omittedMTP:
            throw RepackError.configurationInvalid(
                detail: "omitted MTP tensor reached transformed writer")
        }
    }

    private static func withReadDescriptors<T>(
        request: TransformedTensorWriteRequest,
        body: (Int32, Int32) throws -> T
    ) throws -> T {
        let sourceFD = try Posix.openReadNoFollow(request.source.shardPath)
        defer { close(sourceFD) }
        let destinationFD = try Posix.openReadNoFollow(request.destinationPath)
        defer { close(destinationFD) }
        guard try Posix.fileSize(fd: destinationFD, path: request.destinationPath)
                == request.destinationFileSize else {
            throw RepackError.installStateIncompatible(
                detail: "transformed destination size changed")
        }
        return try body(sourceFD, destinationFD)
    }

    private static func readExact(
        fd: Int32, path: String, offset: UInt64, count: Int,
        operations: TransformedTensorWriterOperations
    ) throws -> Data {
        let data = try operations.read(fd, path, offset, count)
        guard data.count == count else {
            throw RepackError.preadShort(
                path: path, expected: count, got: data.count, errno: 0)
        }
        return data
    }

    private static func advanced(
        _ progress: RemoteTransformProgress,
        request: TransformedTensorWriteRequest,
        unitIndex: UInt64,
        sourceOffset: UInt64,
        sourceData: Data,
        destinationPieces: [(UInt64, Data)]
    ) throws -> RemoteTransformProgress {
        let nextTotal = progress.totalCompletedUnitCount.addingReportingOverflow(1)
        guard !nextTotal.overflow else {
            throw RepackError.configurationInvalid(
                detail: "transformed unit count overflows")
        }
        return RemoteTransformProgress(
            requestIndex: request.requestIndex, requestID: request.requestID,
            completedUnitCount: unitIndex + 1,
            totalCompletedUnitCount: nextTotal.partialValue,
            sourceChainSHA256: chain(
                domain: "source", previous: progress.sourceChainSHA256,
                request: request, unitIndex: unitIndex,
                pieces: [(sourceOffset, sourceData)]),
            destinationChainSHA256: chain(
                domain: "destination", previous: progress.destinationChainSHA256,
                request: request, unitIndex: unitIndex,
                pieces: destinationPieces))
    }

    private static func seed(
        domain: String, binding: RemoteTransformBinding
    ) -> String {
        var stream = Sha256Stream()
        update(Data(proofDomain.utf8), in: &stream)
        update(Data(domain.utf8), in: &stream)
        for value in [
            binding.sourcePayloadSHA256, binding.converterVersion,
            binding.quantizationPolicySHA256, binding.planFingerprint,
            binding.destinationIdentity, String(binding.destinationBytes),
        ] {
            update(Data(value.utf8), in: &stream)
        }
        return stream.finalizeHexString()
    }

    private static func chain(
        domain: String,
        previous: String,
        request: TransformedTensorWriteRequest,
        unitIndex: UInt64,
        pieces: [(UInt64, Data)]
    ) -> String {
        var stream = Sha256Stream()
        update(Data(proofDomain.utf8), in: &stream)
        update(Data(domain.utf8), in: &stream)
        update(Data(previous.utf8), in: &stream)
        update(request.requestIndex, in: &stream)
        update(Data(request.requestID.utf8), in: &stream)
        update(unitIndex, in: &stream)
        update(UInt64(pieces.count), in: &stream)
        for (offset, data) in pieces {
            update(offset, in: &stream)
            update(data, in: &stream)
        }
        return stream.finalizeHexString()
    }

    private static func update(_ data: Data, in stream: inout Sha256Stream) {
        update(UInt64(data.count), in: &stream)
        data.withUnsafeBytes { stream.update($0) }
    }

    private static func update(_ value: UInt64, in stream: inout Sha256Stream) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { stream.update($0) }
    }

    private static func packedValueOffset(
        element: Int,
        elementsPerRow: Int,
        bitWidth: AffineBitWidth
    ) throws -> UInt64 {
        let groupSize = BF16AffineQuantizationPolicy.affineGroupSize
        let completeBytes = (groupSize * bitWidth.rawValue + 7) / 8
        let completeGroups = elementsPerRow / groupSize
        let remainder = elementsPerRow % groupSize
        let rowBytes = completeGroups * completeBytes
            + (remainder == 0 ? 0 : (remainder * bitWidth.rawValue + 7) / 8)
        let row = element / elementsPerRow
        let inRow = element % elementsPerRow
        let group = inRow / groupSize
        let value = row.multipliedReportingOverflow(by: rowBytes)
        let local = group.multipliedReportingOverflow(by: completeBytes)
        guard !value.overflow, !local.overflow else {
            throw RepackError.configurationInvalid(detail: "packed value offset overflows")
        }
        let total = value.partialValue.addingReportingOverflow(local.partialValue)
        guard !total.overflow else {
            throw RepackError.configurationInvalid(detail: "packed value offset overflows")
        }
        return UInt64(total.partialValue)
    }

    private static func add(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let value = lhs.addingReportingOverflow(rhs)
        guard !value.overflow else {
            throw RepackError.configurationInvalid(detail: "writer offset addition overflows")
        }
        return value.partialValue
    }

    private static func multiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let value = lhs.multipliedReportingOverflow(by: rhs)
        guard !value.overflow else {
            throw RepackError.configurationInvalid(detail: "writer offset multiplication overflows")
        }
        return value.partialValue
    }

    private static func checkedInt(_ value: UInt64) throws -> Int {
        guard value <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(detail: "integer conversion overflows")
        }
        return Int(value)
    }
}

struct SyntheticTransformedPackInput: Sendable {
    let source: SourceTensor
    let outputDirectory: String
    let resume: Bool
    let reserveBytes: UInt64

    init(source: SourceTensor, outputDirectory: String,
         resume: Bool = false, reserveBytes: UInt64 = 0) {
        self.source = source
        self.outputDirectory = outputDirectory
        self.resume = resume
        self.reserveBytes = reserveBytes
    }
}

struct SyntheticTransformedPackResult: Sendable, Equatable {
    let outputDirectory: String
    let manifestSHA256: String
    let completedGroupCount: Int
    let structuralFixtureOnly: Bool
}

enum SyntheticTransformedPackWriter {
    static func write(
        input: SyntheticTransformedPackInput,
        audit: RepackAudit = RepackAudit(),
        operations: TransformedTensorWriterOperations = .production,
        afterGroup: (Int) throws -> Void = { _ in }
    ) throws -> SyntheticTransformedPackResult {
        let sourceDigest = try TransformedTensorWriter.sourcePayloadSHA256(
            source: input.source, operations: operations)
        let components = try QwenRepackPlanner.affineComponentSizes(
            shape: input.source.shape, bitWidth: .int4)
        let weightsSize = GTurboFormatV2.alignmentBytes
        let layoutData = try JSONSerialization.data(
            withJSONObject: ["layers": [], "version": 2], options: [.sortedKeys])
        let planFingerprint = digest(strings: [
            "synthetic-v2-gemma", sourceDigest, input.source.name,
            (input.source.shardPath as NSString).lastPathComponent,
            String(input.source.absoluteOffset), String(input.source.sizeBytes),
            input.source.shape.map(String.init).joined(separator: ","),
            String(weightsSize), String(layoutData.count),
            String(components.totalSize),
        ])
        let destinationIdentity = digest(strings: [
            URL(fileURLWithPath: input.outputDirectory).standardizedFileURL.path,
            String(weightsSize), String(layoutData.count),
        ])
        let binding = RemoteTransformBinding(
            sourcePayloadSHA256: sourceDigest,
            converterVersion: TransformedTensorWriter.converterVersion,
            quantizationPolicySHA256: planFingerprint,
            planFingerprint: planFingerprint,
            destinationIdentity: destinationIdentity,
            destinationBytes: weightsSize + UInt64(layoutData.count))
        let requestID = digest(strings: [
            "synthetic-transform-request", planFingerprint, input.source.name,
        ])
        let freshProgress = TransformedTensorWriter.initialProgress(
            binding: binding, firstRequestID: requestID)
        let freshCheckpoint = RemoteInstallCheckpoint(
            repoID: "fixture/transformed-v2",
            requestedRevision: "fixture-revision",
            resolvedCommit: String(repeating: "f", count: 40),
            sourceIndexSHA256: sourceDigest,
            planFingerprint: planFingerprint,
            totalSourceBytes: input.source.sizeBytes,
            transformBinding: binding,
            transformProgress: freshProgress)
        if !input.resume {
            let checkpointData = try JSONEncoder().encode(freshCheckpoint)
            let provisional = try fixtureManifest(
                weights: .init(size: weightsSize, sha256: String(repeating: "0", count: 64)),
                layout: .init(size: UInt64(layoutData.count),
                              sha256: String(repeating: "0", count: 64)),
                components: components)
            let manifestBytes = try GTurboManifestV2Codec.encode(provisional).count
            _ = try DiskSpaceChecker.requireAvailableBeforeCreating(
                path: input.outputDirectory,
                budget: .init(
                    artifactBytes: weightsSize + UInt64(layoutData.count),
                    manifestBytes: UInt64(manifestBytes),
                    checkpointBytes: UInt64(checkpointData.count),
                    temporaryFileBytes: UInt64(manifestBytes + checkpointData.count)),
                reserveBytes: input.reserveBytes)
        }
        let lock = try InstallLock.acquire(outputDirectory: input.outputDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = lock.paths
        guard try Posix.entryKind(paths.finalDirectory) == .absent else {
            throw RepackError.installStateIncompatible(
                detail: "completed synthetic v2 destination is protected")
        }
        var checkpoint: RemoteInstallCheckpoint
        if input.resume {
            checkpoint = try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
            guard checkpoint.matchesTransform(
                repoID: "fixture/transformed-v2",
                requestedRevision: "fixture-revision",
                resolvedCommit: String(repeating: "f", count: 40),
                binding: binding) else {
                throw RepackError.installStateIncompatible(
                    detail: "synthetic transformed resume binding changed")
            }
            _ = try checkpoint.requireTransformBinding(path: paths.checkpointFile)
            guard try Posix.entryKind(paths.partialDirectory) == .directory else {
                throw RepackError.installStateMissing(path: paths.partialDirectory)
            }
        } else {
            guard try Posix.entryKind(paths.partialDirectory) == .absent,
                  try Posix.entryKind(paths.checkpointFile) == .absent else {
                throw RepackError.installStateIncompatible(
                    detail: "owned partial exists; resume it explicitly")
            }
            checkpoint = freshCheckpoint
            try Posix.mkdirP((paths.partialDirectory as NSString)
                .appendingPathComponent("packed_experts"))
            let weightsPath = (paths.partialDirectory as NSString)
                .appendingPathComponent("model_weights.bin")
            let fd = try Posix.openCreateRW(weightsPath)
            try Posix.ftruncate(fd, path: weightsPath, size: weightsSize)
            try Posix.fsync(fd, path: weightsPath)
            close(fd)
            try Posix.atomicWrite(
                layoutData,
                to: (paths.partialDirectory as NSString)
                    .appendingPathComponent("packed_experts/layout.json"),
                durableIn: (paths.partialDirectory as NSString)
                    .appendingPathComponent("packed_experts"))
            try checkpoint.write(
                to: paths.checkpointFile, parentDirectory: paths.parentDirectory)
        }

        let weightsPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("model_weights.bin")
        let request = TransformedTensorWriteRequest(
            requestIndex: 0, requestID: requestID,
            tensorName: "fixture.tensor", source: input.source,
            sourcePayloadSHA256: sourceDigest,
            destinationPath: weightsPath, destinationFileSize: weightsSize,
            fileOffset: 0, storage: .affineInt4, affineComponents: components)
        let savedProgress = try checkpoint.validatedTransformProgress(
            path: paths.checkpointFile)
        let resumePosition = try TransformedTensorWriter.validateProgress(
            savedProgress, binding: binding, requests: [request], operations: operations)
        var progress = savedProgress
        if resumePosition.requestIndex == 0 {
            let result = try TransformedTensorWriter.write(
                request, progress: savedProgress,
                resumeUnitCount: resumePosition.completedUnitCount,
                audit: audit, operations: operations) { updated in
                    checkpoint.transformProgress = updated
                    try checkpoint.write(
                        to: paths.checkpointFile, parentDirectory: paths.parentDirectory)
                    try afterGroup(try checkedInt(updated.totalCompletedUnitCount))
                }
            progress = result.progress
        }

        let finalSourceDigest = try TransformedTensorWriter.sourcePayloadSHA256(
            source: input.source, operations: operations)
        guard finalSourceDigest == sourceDigest else {
            throw RepackError.installStateIncompatible(
                detail: "synthetic source changed during transformation")
        }
        let finalPosition = try TransformedTensorWriter.validateProgress(
            progress, binding: binding, requests: [request], operations: operations)
        guard finalPosition.requestIndex == 1 else {
            throw RepackError.installStateIncompatible(
                detail: "synthetic transformed output is incomplete")
        }
        let weightsSHA = try WriterCore.hashEntireFile(
            path: weightsPath, size: weightsSize, audit: audit,
            cancellationCheck: operations.cancellationCheck)
        let layoutPath = (paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts/layout.json")
        let layoutSHA = try Sha256Stream.hashFile(
            path: layoutPath, noCache: true, noFollow: true)
        let manifest = try fixtureManifest(
            weights: .init(size: weightsSize, sha256: weightsSHA),
            layout: .init(size: UInt64(layoutData.count), sha256: layoutSHA),
            components: components)
        let manifestData = try GTurboManifestV2Codec.encode(manifest)
        guard case let .v2(verified) = try GTurboManifestDocumentCodec.decode(manifestData),
              verified.manifest.family == .gemma4,
              verified.manifest.modelID == "fixture/transformed-v2" else {
            throw RepackError.configurationInvalid(
                detail: "synthetic structural fixture did not decode as Gemma v2")
        }
        try Posix.atomicWrite(
            manifestData,
            to: (paths.partialDirectory as NSString).appendingPathComponent("manifest.json"),
            durableIn: paths.partialDirectory)
        try audit.verifyTransformedPack(
            rootDirectory: paths.partialDirectory, manifest: verified)
        try Posix.fsyncDirectory(paths.partialDirectory)
        try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
        try Posix.fsyncDirectory(paths.parentDirectory)
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)
        try audit.verifyTransformedPack(
            rootDirectory: paths.finalDirectory, manifest: verified)
        return .init(
            outputDirectory: paths.finalDirectory,
            manifestSHA256: verified.manifestSHA256,
            completedGroupCount: try checkedInt(progress.totalCompletedUnitCount),
            structuralFixtureOnly: true)
    }

    private static func fixtureManifest(
        weights: GTurboManifestFileV1,
        layout: GTurboManifestFileV1,
        components: QwenAffineComponentLayout
    ) throws -> GTurboManifestV2 {
        let arch = GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 32, vocabSize: 1_024,
            slidingWindow: 128, finalLogitSoftcap: 30,
            ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1,
            numExperts: 2, topKExperts: 1, tieWordEmbeddings: true,
            attentionKEqV: true, hiddenActivation: "gelu_pytorch_tanh",
            fullAttentionLayerMask: [0])
        let digest = String(repeating: "0", count: 64)
        let categories: [GTurboQuantizationCategoryV2] = [
            .embedding, .attention, .router, .sharedExpert, .routedExpert,
        ]
        return GTurboManifestV2(
            family: .gemma4,
            requiredFeatures: [.familyDispatch, .verifiedIdentity],
            modelID: "fixture/transformed-v2", architecture: .gemma4(arch),
            provenance: .init(
                sourceRepository: "fixture/transformed-v2",
                sourceRevision: "fixture-revision",
                sourceIndexSHA256: digest,
                sidecarSHA256: ["config.json": digest],
                quantizationPolicySHA256: digest),
            quantization: categories.map {
                .init(category: $0, storage: .affineInt4,
                      groupSize: 64, scaleType: "bf16", biasType: "bf16")
            },
            ignoredTensors: [],
            files: ["model_weights.bin": weights,
                    "packed_experts/layout.json": layout],
            tensorRegions: [.init(
                name: "fixture.tensor", file: "model_weights.bin",
                offset: 0, size: components.totalSize, shape: [2, 65],
                storage: .affineInt4, quantizationCategory: .embedding)],
            expertsPerLayer: 2, numLayers: 1,
            expertStride: GTurboFormatV2.alignmentBytes)
    }

    private static func digest(strings: [String]) -> String {
        var stream = Sha256Stream()
        for string in strings {
            Data((string + "\n").utf8).withUnsafeBytes { stream.update($0) }
        }
        return stream.finalizeHexString()
    }

    private static func checkedInt(_ value: UInt64) throws -> Int {
        guard value <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(
                detail: "transformed unit count exceeds Int")
        }
        return Int(value)
    }
}

struct QwenTransformedPackWriteOptions: Sendable {
    let outputDirectory: String
    let visionOutputDirectory: String?
    let visionProcessorData: Data?
    let resume: Bool
    let reserveBytes: UInt64
    let officialPayloadIdentity: LocalOfficialQwenPayloadIdentity?
    let durableProgress: @Sendable (UInt64) -> Void
    let revalidateOfficialPayload: (@Sendable () throws -> LocalOfficialQwenPayloadIdentity)?

    init(outputDirectory: String,
         visionOutputDirectory: String? = nil,
         visionProcessorData: Data? = nil,
         resume: Bool = false,
         reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
         officialPayloadIdentity: LocalOfficialQwenPayloadIdentity? = nil,
         durableProgress: @escaping @Sendable (UInt64) -> Void = { _ in },
         revalidateOfficialPayload: (@Sendable () throws -> LocalOfficialQwenPayloadIdentity)? = nil) {
        self.outputDirectory = outputDirectory
        self.visionOutputDirectory = visionOutputDirectory
        self.visionProcessorData = visionProcessorData
        self.resume = resume
        self.reserveBytes = reserveBytes
        self.officialPayloadIdentity = officialPayloadIdentity
        self.durableProgress = durableProgress
        self.revalidateOfficialPayload = revalidateOfficialPayload
    }
}

struct QwenTransformedPackWriteResult: Sendable {
    let textManifestSHA256: String
    let textReceiptSHA256: String
    let textReceiptPath: String
    let visionManifestSHA256: String?
    let visionReceiptSHA256: String?
    let visionReceiptPath: String?
    let visionManifestWritten: Bool
    let completedGroupCount: Int
    let maximumObservedTransformScratchBytes: Int
}

struct QwenResumeDiskUsage: Sendable, Equatable {
    let textRemainingArtifactBytes: UInt64
    let visionRemainingArtifactBytes: UInt64
    let existingTextMetadataBytes: UInt64
    let existingVisionMetadataBytes: UInt64
    let existingVisionProcessorBytes: UInt64
}

enum QwenPublicationAuditTarget: Sendable {
    case text
    case vision
}

struct QwenPublicationOperations {
    var rename: (_ source: String, _ destination: String) throws -> Void
    var fsyncDirectory: (_ path: String) throws -> Void
    var beforeFinalAudit: (_ target: QwenPublicationAuditTarget) throws -> Void

    static var production: Self { Self(
        rename: { source, destination in
            try Posix.rename(from: source, to: destination)
        },
        fsyncDirectory: { try Posix.fsyncDirectory($0) },
        beforeFinalAudit: { _ in }) }
}

/// Qwen-only manifest factory and production entry boundary. Synthetic fixture
/// input has no overload here and cannot be promoted as official Qwen.
enum QwenTransformedPackWriter {
    /// Reads only the compact checkpoint and plan geometry. The writer repeats
    /// the full source/destination chain replay under its install locks before
    /// opening any partial read-write, so this method is only the exact
    /// remaining-write input to the no-create disk preflight.
    static func resumeDiskUsage(
        plan: QwenRepackPlan,
        outputDirectory: String,
        visionOutputDirectory: String?
    ) throws -> QwenResumeDiskUsage {
        let textPaths = try RemoteInstallPaths(outputDirectory: outputDirectory)
        let visionPaths = try visionOutputDirectory.map(RemoteInstallPaths.init)
        let checkpoint = try RemoteInstallCheckpoint.load(from: textPaths.checkpointFile)
        _ = try validatedBinding(
            checkpoint: checkpoint, plan: plan,
            textPaths: textPaths, visionPaths: visionPaths)
        let requests = try transformRequests(
            plan: plan, textPartialDirectory: textPaths.partialDirectory,
            visionPartialDirectory: visionPaths?.partialDirectory,
            sourceDigests: placeholderSourceDigests(plan: plan))
        let progress = try checkpoint.validatedTransformProgress(
            path: textPaths.checkpointFile)
        let position = try validatedResumePosition(
            progress: progress, requests: requests)
        try validateCanonicalLayout(plan: plan, textPaths: textPaths)

        let textMetadataBytes = try existingRegularFileBytes(
            names: ["manifest.json", VerifiedInstallReceiptWriter.fileName],
            root: textPaths.partialDirectory)
        let visionMetadataBytes = try visionPaths.map {
            try existingRegularFileBytes(
                names: ["manifest.json", VerifiedInstallReceiptWriter.fileName],
                root: $0.partialDirectory)
        } ?? 0
        let visionProcessorBytes = try visionPaths.map {
            try existingRegularFileBytes(
                names: [GTurboVisionFormatV2.processorFile],
                root: $0.partialDirectory)
        } ?? 0

        var textBytes: UInt64 = 0
        var visionBytes: UInt64 = 0
        if position.requestIndex < requests.count {
            for index in position.requestIndex..<requests.count {
                let request = requests[index]
                let completed = index == position.requestIndex
                    ? position.completedUnitCount : 0
                let remaining = try remainingDestinationWriteBytes(
                    request: request, completedUnitCount: completed)
                if request.destinationPath.hasPrefix(textPaths.partialDirectory + "/") {
                    textBytes = try sum([textBytes, remaining])
                } else if let visionPaths,
                          request.destinationPath.hasPrefix(
                            visionPaths.partialDirectory + "/") {
                    visionBytes = try sum([visionBytes, remaining])
                } else {
                    throw RepackError.installStateIncompatible(
                        detail: "Qwen resume request is outside its owned partials")
                }
            }
        }
        return .init(
            textRemainingArtifactBytes: textBytes,
            visionRemainingArtifactBytes: visionBytes,
            existingTextMetadataBytes: textMetadataBytes,
            existingVisionMetadataBytes: visionMetadataBytes,
            existingVisionProcessorBytes: visionProcessorBytes)
    }

    /// Proves that discard is targeting the checkpoint-bound partials rather
    /// than recursively deleting whatever currently occupies those names.
    static func validateOwnedPartialState(
        plan: QwenRepackPlan,
        textPaths: RemoteInstallPaths,
        visionPaths: RemoteInstallPaths?,
        processorData: Data?,
        checkpoint: RemoteInstallCheckpoint,
        operations: TransformedTensorWriterOperations = .production
    ) throws {
        let binding = try validatedBinding(
            checkpoint: checkpoint, plan: plan,
            textPaths: textPaths, visionPaths: visionPaths)
        let requests = try transformRequests(
            plan: plan, textPartialDirectory: textPaths.partialDirectory,
            visionPartialDirectory: visionPaths?.partialDirectory,
            sourceDigests: placeholderSourceDigests(plan: plan))
        let progress = try checkpoint.validatedTransformProgress(
            path: textPaths.checkpointFile)
        _ = try TransformedTensorWriter.validateProgress(
            progress, binding: binding, requests: requests, operations: operations)
        try validateCanonicalLayout(plan: plan, textPaths: textPaths)
        try validateOwnedInventories(
            plan: plan, textPaths: textPaths, visionPaths: visionPaths,
            processorData: processorData)
    }

    static func write(
        plan: QwenRepackPlan,
        options: QwenTransformedPackWriteOptions,
        audit: RepackAudit = RepackAudit(),
        operations: TransformedTensorWriterOperations = .production,
        publicationOperations: QwenPublicationOperations = .production
    ) throws -> QwenTransformedPackWriteResult {
        if plan.visionCompanion != nil {
            guard options.visionOutputDirectory != nil,
                  options.visionProcessorData != nil else {
                throw RepackError.configurationInvalid(
                    detail: "Qwen vision plan requires companion output and processor data")
            }
        } else {
            guard options.visionOutputDirectory == nil,
                  options.visionProcessorData == nil else {
                throw RepackError.configurationInvalid(
                    detail: "vision output supplied for a text-only plan")
            }
        }
        let layout = try layoutData(plan: plan)
        let textArtifacts = plan.artifacts.filter {
            $0.relativePath != plan.visionCompanion?.artifact.relativePath
        }
        let textArtifactBytes = try sum(textArtifacts.map(\.size))
        if !options.resume {
            let placeholderDigest = String(repeating: "0", count: 64)
            let placeholderFiles = Dictionary(uniqueKeysWithValues: textArtifacts.map {
                ($0.relativePath, RepackAudit.OutputFile(
                    relativePath: $0.relativePath, size: $0.size,
                    sha256: placeholderDigest))
            })
            let placeholderText = try makeTextManifest(
                plan: plan, outputFiles: placeholderFiles)
            let textManifestBytes = try GTurboManifestV2Codec.encode(placeholderText.manifest).count
            let placeholderPayload = options.officialPayloadIdentity?.shardSetSHA256
                ?? placeholderDigest
            let textReceipt = try VerifiedInstallReceiptWriter.encode(
                outputDir: options.outputDirectory,
                manifestSha256: placeholderDigest,
                manifestSize: UInt64(textManifestBytes),
                sourceRepoID: plan.provenance.repository,
                sourceRevision: plan.provenance.revision,
                toolVersion: TransformedTensorWriter.converterVersion,
                verificationTimestamp: VerifiedInstallReceiptWriter.deterministicV2Timestamp,
                conversionProvenance: .init(
                    sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                    sourcePayloadSHA256: placeholderPayload,
                    planFingerprint: plan.canonicalFingerprint,
                    quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256,
                    converterVersion: TransformedTensorWriter.converterVersion,
                    compatibleTextManifestSHA256: nil),
                files: Array(placeholderFiles.values))
            var destinations = [TransformedDiskSpaceDestination(
                path: options.outputDirectory,
                budget: .init(
                    artifactBytes: textArtifactBytes,
                    manifestBytes: try sum([
                        UInt64(textManifestBytes), UInt64(textReceipt.count),
                    ]),
                    checkpointBytes: RemoteInstallCheckpoint.maximumBytes,
                    temporaryFileBytes: UInt64(max(
                        layout.count, textManifestBytes, textReceipt.count,
                        Int(RemoteInstallCheckpoint.maximumBytes)))))]
            if let vision = plan.visionCompanion,
               let visionOutput = options.visionOutputDirectory,
               let processor = options.visionProcessorData {
                let processorSHA = digest(data: processor)
                guard processorSHA == plan.provenance.validatedSidecarSHA256[
                    GTurboVisionFormatV2.processorFile] else {
                    throw RepackError.installStateIncompatible(
                        detail: "Qwen vision processor sidecar changed")
                }
                let placeholderVision = makeVisionManifest(
                    plan: plan, vision: vision,
                    processorBytes: UInt64(processor.count),
                    processorSHA256: processorSHA,
                    textManifestSHA256: placeholderDigest,
                    weightsSHA256: placeholderDigest)
                let visionManifestBytes = try GTurboVisionManifestV2Codec.encode(
                    placeholderVision).count
                let visionReceipt = try VerifiedInstallReceiptWriter.encode(
                    outputDir: visionOutput,
                    manifestSha256: placeholderDigest,
                    manifestSize: UInt64(visionManifestBytes),
                    sourceRepoID: plan.provenance.repository,
                    sourceRevision: plan.provenance.revision,
                    toolVersion: TransformedTensorWriter.converterVersion,
                    verificationTimestamp: VerifiedInstallReceiptWriter.deterministicV2Timestamp,
                    conversionProvenance: .init(
                        sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                        sourcePayloadSHA256: placeholderPayload,
                        planFingerprint: plan.canonicalFingerprint,
                        quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256,
                        converterVersion: TransformedTensorWriter.converterVersion,
                        compatibleTextManifestSHA256: placeholderDigest),
                    files: [
                        .init(relativePath: vision.artifact.relativePath,
                              size: vision.artifact.size, sha256: placeholderDigest),
                        .init(relativePath: GTurboVisionFormatV2.processorFile,
                              size: UInt64(processor.count), sha256: processorSHA),
                    ])
                let visionArtifactBytes = try sum([
                    vision.artifact.size, UInt64(processor.count),
                ])
                destinations.append(.init(
                    path: visionOutput,
                    budget: .init(
                        artifactBytes: visionArtifactBytes,
                        manifestBytes: try sum([
                            UInt64(visionManifestBytes), UInt64(visionReceipt.count),
                        ]),
                        checkpointBytes: 0,
                        temporaryFileBytes: UInt64(max(
                            processor.count, visionManifestBytes, visionReceipt.count)))))
            }
            _ = try DiskSpaceChecker.requireCombinedAvailableBeforeCreating(
                destinations: destinations, reserveBytes: options.reserveBytes)
        }

        let textLock = try InstallLock.acquire(outputDirectory: options.outputDirectory)
        defer { withExtendedLifetime(textLock) {} }
        let textPaths = textLock.paths
        guard try Posix.entryKind(textPaths.finalDirectory) == .absent else {
            throw RepackError.installStateIncompatible(
                detail: "completed Qwen destination is protected")
        }
        let visionLock: InstallLock?
        if let output = options.visionOutputDirectory {
            visionLock = try InstallLock.acquire(outputDirectory: output)
            guard try Posix.entryKind(visionLock!.paths.finalDirectory) == .absent else {
                throw RepackError.installStateIncompatible(
                    detail: "completed Qwen vision destination is protected")
            }
        } else {
            visionLock = nil
        }
        defer { withExtendedLifetime(visionLock) {} }

        var sourceDigests: [SourceTensor: String] = [:]
        var orderedSources = plan.textTensors.map(\.source)
        orderedSources += plan.expertLayers.flatMap { $0.sources.map(\.source) }
        orderedSources += plan.visionCompanion?.tensors.map(\.source) ?? []
        var aggregate = Sha256Stream()
        var totalSourceBytes: UInt64 = 0
        for source in orderedSources {
            let digest: String
            if let existing = sourceDigests[source] {
                digest = existing
            } else {
                digest = try TransformedTensorWriter.sourcePayloadSHA256(
                    source: source, operations: operations)
                sourceDigests[source] = digest
            }
            Data((source.name + "\t" + digest + "\n").utf8)
                .withUnsafeBytes { aggregate.update($0) }
            let sum = totalSourceBytes.addingReportingOverflow(source.sizeBytes)
            guard !sum.overflow else {
                throw RepackError.configurationInvalid(detail: "Qwen source byte total overflows")
            }
            totalSourceBytes = sum.partialValue
        }
        let payloadDigest = aggregate.finalizeHexString()
        let destinationBytes = try sum(plan.artifacts.map(\.size))
        let destinationIdentity = digest(strings: [
            URL(fileURLWithPath: textPaths.finalDirectory).standardizedFileURL.path,
            options.visionOutputDirectory ?? "none",
            String(destinationBytes), plan.canonicalFingerprint,
        ])
        let binding = RemoteTransformBinding(
            sourcePayloadSHA256: payloadDigest,
            converterVersion: TransformedTensorWriter.converterVersion,
            quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256,
            planFingerprint: plan.canonicalFingerprint,
            destinationIdentity: destinationIdentity,
            destinationBytes: destinationBytes)
        let requests = try transformRequests(
            plan: plan, textPartialDirectory: textPaths.partialDirectory,
            visionPartialDirectory: visionLock?.paths.partialDirectory,
            sourceDigests: sourceDigests)
        guard let firstRequest = requests.first else {
            throw RepackError.configurationInvalid(
                detail: "Qwen transform schedule is empty")
        }
        let initialProgress = TransformedTensorWriter.initialProgress(
            binding: binding, firstRequestID: firstRequest.requestID)
        var checkpoint: RemoteInstallCheckpoint
        if options.resume {
            checkpoint = try RemoteInstallCheckpoint.load(from: textPaths.checkpointFile)
            _ = try validatedBinding(
                checkpoint: checkpoint, plan: plan,
                textPaths: textPaths, visionPaths: visionLock?.paths)
            guard checkpoint.matchesTransform(
                repoID: plan.provenance.repository,
                requestedRevision: plan.provenance.revision,
                resolvedCommit: plan.provenance.revision,
                binding: binding),
                  try Posix.entryKind(textPaths.partialDirectory) == .directory else {
                throw RepackError.installStateIncompatible(
                    detail: "Qwen transformed resume binding changed")
            }
            _ = try checkpoint.validatedTransformProgress(
                path: textPaths.checkpointFile)
            try validateCanonicalLayout(plan: plan, textPaths: textPaths)
        } else {
            let visionPartialKind = try visionLock.map {
                try Posix.entryKind($0.paths.partialDirectory)
            } ?? .absent
            guard try Posix.entryKind(textPaths.partialDirectory) == .absent,
                  try Posix.entryKind(textPaths.checkpointFile) == .absent,
                  visionPartialKind == .absent else {
                throw RepackError.installStateIncompatible(
                    detail: "owned Qwen partial exists; resume it explicitly")
            }
            checkpoint = RemoteInstallCheckpoint(
                repoID: plan.provenance.repository,
                requestedRevision: plan.provenance.revision,
                resolvedCommit: plan.provenance.revision,
                sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                planFingerprint: plan.canonicalFingerprint,
                totalSourceBytes: totalSourceBytes,
                transformBinding: binding,
                transformProgress: initialProgress)
            try createArtifacts(
                textArtifacts, root: textPaths.partialDirectory, layout: layout)
            if let vision = plan.visionCompanion, let visionLock {
                try createArtifacts(
                    [vision.artifact], root: visionLock.paths.partialDirectory,
                    layout: nil)
            }
            try checkpoint.write(
                to: textPaths.checkpointFile, parentDirectory: textPaths.parentDirectory)
        }

        var progress = try checkpoint.validatedTransformProgress(
            path: textPaths.checkpointFile)
        var maximumObservedTransformScratchBytes = 0
        let resumePosition = try TransformedTensorWriter.validateProgress(
            progress, binding: binding, requests: requests, operations: operations)
        options.durableProgress(progress.totalCompletedUnitCount)
        if resumePosition.requestIndex < requests.count {
            for requestIndex in resumePosition.requestIndex..<requests.count {
                try operations.cancellationCheck()
                let request = requests[requestIndex]
                let resumeUnits = requestIndex == resumePosition.requestIndex
                    ? resumePosition.completedUnitCount : 0
                let result = try TransformedTensorWriter.write(
                    request, progress: progress, resumeUnitCount: resumeUnits,
                    audit: audit, affineDurabilityInterval: 65_536,
                    operations: operations) { updated in
                        checkpoint.transformProgress = updated
                        try checkpoint.write(
                            to: textPaths.checkpointFile,
                            parentDirectory: textPaths.parentDirectory)
                        options.durableProgress(updated.totalCompletedUnitCount)
                    }
                progress = result.progress
                maximumObservedTransformScratchBytes = max(
                    maximumObservedTransformScratchBytes,
                    result.maximumScratchBytes)
            }
        }

        var finalAggregate = Sha256Stream()
        for source in orderedSources {
            let current = try TransformedTensorWriter.sourcePayloadSHA256(
                source: source, operations: operations)
            Data((source.name + "\t" + current + "\n").utf8)
                .withUnsafeBytes { finalAggregate.update($0) }
        }
        guard finalAggregate.finalizeHexString() == payloadDigest else {
            throw RepackError.installStateIncompatible(
                detail: "Qwen source changed during transformation")
        }
        if let expectedOfficial = options.officialPayloadIdentity {
            guard let revalidate = options.revalidateOfficialPayload else {
                throw RepackError.configurationInvalid(
                    detail: "official Qwen payload has no publication revalidation")
            }
            let observedOfficial = try revalidate()
            guard observedOfficial == expectedOfficial else {
                throw RepackError.installStateIncompatible(
                    detail: "official Qwen payload identity changed before publication")
            }
        } else if options.revalidateOfficialPayload != nil {
            throw RepackError.configurationInvalid(
                detail: "Qwen payload revalidation has no initial identity")
        }
        let finalPosition = try TransformedTensorWriter.validateProgress(
            progress, binding: binding, requests: requests, operations: operations)
        guard finalPosition.requestIndex == requests.count else {
            throw RepackError.installStateIncompatible(
                detail: "Qwen transformed output is incomplete")
        }

        let textFiles = try hashArtifacts(
            textArtifacts, root: textPaths.partialDirectory, audit: audit,
            cancellationCheck: operations.cancellationCheck)
        let textManifest = try makeTextManifest(plan: plan, outputFiles: textFiles)
        let textManifestData = try GTurboManifestV2Codec.encode(textManifest.manifest)
        let placeholderTextFiles = textFiles.mapValues {
            RepackAudit.OutputFile(
                relativePath: $0.relativePath, size: $0.size,
                sha256: String(repeating: "0", count: 64))
        }
        let placeholderTextManifest = try makeTextManifest(
            plan: plan, outputFiles: placeholderTextFiles)
        guard try GTurboManifestV2Codec.encode(placeholderTextManifest.manifest).count
            == textManifestData.count else {
            throw RepackError.configurationInvalid(
                detail: "Qwen text manifest size changed after hashing")
        }
        try Posix.atomicWrite(
            textManifestData,
            to: (textPaths.partialDirectory as NSString).appendingPathComponent("manifest.json"),
            durableIn: textPaths.partialDirectory)
        let publishedSourcePayloadSHA256 = options.officialPayloadIdentity?.shardSetSHA256
            ?? payloadDigest
        let textReceiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: textPaths.finalDirectory,
            manifestSha256: textManifest.manifestSHA256,
            manifestSize: UInt64(textManifestData.count),
            sourceRepoID: plan.provenance.repository,
            sourceRevision: plan.provenance.revision,
            toolVersion: TransformedTensorWriter.converterVersion,
            verificationTimestamp: VerifiedInstallReceiptWriter.deterministicV2Timestamp,
            conversionProvenance: .init(
                sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                sourcePayloadSHA256: publishedSourcePayloadSHA256,
                planFingerprint: plan.canonicalFingerprint,
                quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256,
                converterVersion: TransformedTensorWriter.converterVersion,
                compatibleTextManifestSHA256: nil),
            files: Array(textFiles.values))
        let textReceiptPath = (textPaths.partialDirectory as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try Posix.atomicWrite(
            textReceiptData, to: textReceiptPath,
            durableIn: textPaths.partialDirectory)
        try audit.verifyTransformedPack(
            rootDirectory: textPaths.partialDirectory,
            manifest: textManifest,
            receiptData: textReceiptData)
        try Posix.fsyncDirectory(textPaths.partialDirectory)

        var preparedVision: (paths: RemoteInstallPaths,
                             manifest: GTurboVerifiedVisionManifestV2,
                             manifestSHA256: String,
                             receiptData: Data)?
        if let vision = plan.visionCompanion, let visionLock,
           let processorData = options.visionProcessorData {
            let processorPath = (visionLock.paths.partialDirectory as NSString)
                .appendingPathComponent(GTurboVisionFormatV2.processorFile)
            try Posix.atomicWrite(
                processorData, to: processorPath,
                durableIn: visionLock.paths.partialDirectory)
            let processorSHA = digest(data: processorData)
            guard processorSHA == plan.provenance.validatedSidecarSHA256[
                GTurboVisionFormatV2.processorFile] else {
                throw RepackError.installStateIncompatible(
                    detail: "Qwen vision processor sidecar changed")
            }
            let visionFiles = try hashArtifacts(
                [vision.artifact], root: visionLock.paths.partialDirectory,
                audit: audit, cancellationCheck: operations.cancellationCheck)
            guard let weights = visionFiles[vision.artifact.relativePath] else {
                throw RepackError.configurationInvalid(detail: "missing Qwen vision output")
            }
            let manifest = makeVisionManifest(
                plan: plan, vision: vision,
                processorBytes: UInt64(processorData.count),
                processorSHA256: processorSHA,
                textManifestSHA256: textManifest.manifestSHA256,
                weightsSHA256: weights.sha256)
            let manifestData = try GTurboVisionManifestV2Codec.encode(manifest)
            let placeholderVisionManifest = makeVisionManifest(
                plan: plan, vision: vision,
                processorBytes: UInt64(processorData.count),
                processorSHA256: processorSHA,
                textManifestSHA256: String(repeating: "0", count: 64),
                weightsSHA256: String(repeating: "0", count: 64))
            guard try GTurboVisionManifestV2Codec.encode(placeholderVisionManifest).count
                == manifestData.count else {
                throw RepackError.configurationInvalid(
                    detail: "Qwen vision manifest size changed after hashing")
            }
            let verified = try GTurboVisionManifestV2Codec.decode(manifestData)
            let visionManifestSHA256 = digest(data: manifestData)
            try Posix.atomicWrite(
                manifestData,
                to: (visionLock.paths.partialDirectory as NSString)
                    .appendingPathComponent("manifest.json"),
                durableIn: visionLock.paths.partialDirectory)
            let visionReceiptData = try VerifiedInstallReceiptWriter.encode(
                outputDir: visionLock.paths.finalDirectory,
                manifestSha256: visionManifestSHA256,
                manifestSize: UInt64(manifestData.count),
                sourceRepoID: plan.provenance.repository,
                sourceRevision: plan.provenance.revision,
                toolVersion: TransformedTensorWriter.converterVersion,
                verificationTimestamp: VerifiedInstallReceiptWriter.deterministicV2Timestamp,
                conversionProvenance: .init(
                    sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                    sourcePayloadSHA256: publishedSourcePayloadSHA256,
                    planFingerprint: plan.canonicalFingerprint,
                    quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256,
                    converterVersion: TransformedTensorWriter.converterVersion,
                    compatibleTextManifestSHA256: textManifest.manifestSHA256),
                files: [
                    weights,
                    .init(
                        relativePath: GTurboVisionFormatV2.processorFile,
                        size: UInt64(processorData.count),
                        sha256: processorSHA),
                ])
            try Posix.atomicWrite(
                visionReceiptData,
                to: (visionLock.paths.partialDirectory as NSString)
                    .appendingPathComponent(VerifiedInstallReceiptWriter.fileName),
                durableIn: visionLock.paths.partialDirectory)
            try audit.verifyTransformedVisionPack(
                rootDirectory: visionLock.paths.partialDirectory,
                manifest: verified,
                receiptData: visionReceiptData)
            try Posix.fsyncDirectory(visionLock.paths.partialDirectory)
            preparedVision = (
                visionLock.paths, verified, visionManifestSHA256, visionReceiptData)
        }

        var textPublished = false
        var visionPublished = false
        do {
            try publicationOperations.rename(
                textPaths.partialDirectory, textPaths.finalDirectory)
            textPublished = true
            try publicationOperations.fsyncDirectory(textPaths.parentDirectory)
            try publicationOperations.beforeFinalAudit(.text)
            try audit.verifyTransformedPack(
                rootDirectory: textPaths.finalDirectory,
                manifest: textManifest,
                receiptData: textReceiptData)
            if let preparedVision {
                try publicationOperations.rename(
                    preparedVision.paths.partialDirectory,
                    preparedVision.paths.finalDirectory)
                visionPublished = true
                try publicationOperations.fsyncDirectory(
                    preparedVision.paths.parentDirectory)
                try publicationOperations.beforeFinalAudit(.vision)
                try audit.verifyTransformedVisionPack(
                    rootDirectory: preparedVision.paths.finalDirectory,
                    manifest: preparedVision.manifest,
                    receiptData: preparedVision.receiptData)
            }
        } catch {
            let publicationError = error
            do {
                if visionPublished, let preparedVision {
                    try publicationOperations.rename(
                        preparedVision.paths.finalDirectory,
                        preparedVision.paths.partialDirectory)
                    try publicationOperations.fsyncDirectory(
                        preparedVision.paths.parentDirectory)
                }
                if textPublished {
                    try publicationOperations.rename(
                        textPaths.finalDirectory, textPaths.partialDirectory)
                    try publicationOperations.fsyncDirectory(textPaths.parentDirectory)
                }
                try checkpoint.write(
                    to: textPaths.checkpointFile,
                    parentDirectory: textPaths.parentDirectory)
            } catch {
                throw RepackError.installStateIncompatible(
                    detail: "Qwen publication failed (\(publicationError)); "
                        + "rollback failed (\(error)); owned state requires inspection")
            }
            throw publicationError
        }
        do {
            try FileManager.default.removeItem(atPath: textPaths.checkpointFile)
            try publicationOperations.fsyncDirectory(textPaths.parentDirectory)
        } catch {
            throw RepackError.installStateIncompatible(
                detail: "Qwen packs were published and audited, but checkpoint cleanup "
                    + "failed (\(error)); published outputs remain authoritative")
        }
        let textReceiptSHA256 = digest(data: textReceiptData)
        return .init(
            textManifestSHA256: textManifest.manifestSHA256,
            textReceiptSHA256: textReceiptSHA256,
            textReceiptPath: (textPaths.finalDirectory as NSString)
                .appendingPathComponent(VerifiedInstallReceiptWriter.fileName),
            visionManifestSHA256: preparedVision?.manifestSHA256,
            visionReceiptSHA256: preparedVision.map { digest(data: $0.receiptData) },
            visionReceiptPath: preparedVision.map {
                ($0.paths.finalDirectory as NSString)
                    .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
            },
            visionManifestWritten: preparedVision != nil,
            completedGroupCount: try checkedInt(progress.totalCompletedUnitCount),
            maximumObservedTransformScratchBytes:
                maximumObservedTransformScratchBytes)
    }

    static func makeTextManifest(
        plan: QwenRepackPlan,
        outputFiles: [String: RepackAudit.OutputFile]
    ) throws -> GTurboVerifiedManifestV2 {
        let expectedArtifacts = plan.artifacts.filter {
            $0.relativePath != plan.visionCompanion?.artifact.relativePath
        }
        let expectedByPath = Dictionary(uniqueKeysWithValues: expectedArtifacts.map {
            ($0.relativePath, $0.size)
        })
        guard Set(outputFiles.keys) == Set(expectedByPath.keys),
              outputFiles.allSatisfy({ key, file in
                  key == file.relativePath && expectedByPath[key] == file.size
              }) else {
            throw RepackError.configurationInvalid(
                detail: "Qwen manifest files disagree with the P6 plan")
        }
        let files = Dictionary(uniqueKeysWithValues: outputFiles.values.map {
            ($0.relativePath, GTurboManifestFileV1(size: $0.size, sha256: $0.sha256))
        })
        let textRegions = plan.textTensors.map { tensor in
            GTurboTensorRegionV2(
                name: tensor.source.name, file: tensor.relativeFile,
                offset: tensor.fileOffset, size: tensor.regionSize,
                shape: tensor.source.shape, storage: storage(tensor.storage),
                quantizationCategory: tensor.quantizationCategory)
        }
        let expertRegions = plan.expertLayers.flatMap { layer in
            (0..<layer.expertsPerLayer).map { expert in
                let used = layer.sources.map(\.components.totalSize).max() ?? 0
                return GTurboTensorRegionV2(
                    name: "qwen.layer.\(layer.layerIndex).expert.\(expert)",
                    file: layer.relativePath,
                    offset: UInt64(expert) * layer.expertStride,
                    size: used,
                    shape: [used], storage: .affineInt4,
                    quantizationCategory: .routedExpert)
            }
        }
        let manifest = GTurboManifestV2(
            family: .qwen3_6,
            requiredFeatures: [
                .familyDispatch, .verifiedIdentity,
                .qwenHybridAttention, .qwenMTPExcluded,
            ],
            modelID: plan.provenance.repository,
            architecture: .qwen3_6(plan.architecture.text),
            provenance: .init(
                sourceRepository: plan.provenance.repository,
                sourceRevision: plan.provenance.revision,
                sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                sidecarSHA256: plan.provenance.validatedSidecarSHA256,
                quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256),
            quantization: plan.quantizationGroups,
            ignoredTensors: plan.omittedMTPNames.map {
                .init(name: $0, reason: .unsupportedMTP)
            },
            files: files,
            tensorRegions: textRegions + expertRegions,
            expertsPerLayer: plan.architecture.text.numberOfExperts,
            numLayers: plan.architecture.text.numLayers,
            expertStride: plan.expertLayers.first?.expertStride ?? 0)
        return try GTurboManifestV2Codec.decode(
            GTurboManifestV2Codec.encode(manifest))
    }

    static func layoutData(plan: QwenRepackPlan) throws -> Data {
        let layers: [[String: Any]] = plan.expertLayers.map { layer in
            [
                "layer": layer.layerIndex,
                "path": layer.relativePath,
                "experts": layer.expertsPerLayer,
                "stride": layer.expertStride,
                "sources": layer.sources.map { source in
                    [
                        "name": source.source.name,
                        "role": source.role,
                        "shape": source.perExpertShape,
                        "valuesOffset": source.components.valuesOffset,
                        "valuesSize": source.components.valuesSize,
                        "scalesOffset": source.components.scalesOffset,
                        "scalesSize": source.components.scalesSize,
                        "biasesOffset": source.components.biasesOffset,
                        "biasesSize": source.components.biasesSize,
                    ] as [String: Any]
                },
            ]
        }
        let data = try JSONSerialization.data(
            withJSONObject: ["version": 2, "layers": layers], options: [.sortedKeys])
        guard plan.artifacts.first(where: {
            $0.relativePath == "packed_experts/layout.json"
        })?.size == UInt64(data.count) else {
            throw RepackError.configurationInvalid(
                detail: "Qwen layout bytes disagree with P6 plan")
        }
        return data
    }

    static func makeVisionManifest(
        plan: QwenRepackPlan,
        vision: QwenVisionCompanionPlan,
        processorBytes: UInt64,
        processorSHA256: String,
        textManifestSHA256: String,
        weightsSHA256: String
    ) -> GTurboVisionManifestV2 {
        GTurboVisionManifestV2(
            family: .qwen3_6,
            modelID: plan.provenance.repository,
            sourceRevision: plan.provenance.revision,
            processorProfile: .init(
                processorClass: "Qwen3VLProcessor",
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2),
            processorConfigSHA256: processorSHA256,
            compatibleTextManifestSHA256: textManifestSHA256,
            visionPayloadSHA256: weightsSHA256,
            supportsStillImages: true, supportsVideo: false,
            files: [
                vision.artifact.relativePath: .init(
                    size: vision.artifact.size, sha256: weightsSHA256),
                GTurboVisionFormatV2.processorFile: .init(
                    size: processorBytes, sha256: processorSHA256),
            ],
            tensorRegions: vision.tensors.map {
                .init(name: $0.source.name, file: $0.relativeFile,
                      offset: $0.fileOffset, size: $0.regionSize,
                      shape: $0.source.shape, storage: storage($0.storage))
            })
    }

    private static func transformRequests(
        plan: QwenRepackPlan,
        textPartialDirectory: String,
        visionPartialDirectory: String?,
        sourceDigests: [SourceTensor: String]
    ) throws -> [TransformedTensorWriteRequest] {
        let artifactSizes = Dictionary(uniqueKeysWithValues: plan.artifacts.map {
            ($0.relativePath, $0.size)
        })
        var requests: [TransformedTensorWriteRequest] = []

        func append(
            tensorName: String,
            source: SourceTensor,
            sourceDigest: String?,
            destinationPath: String,
            destinationSize: UInt64,
            fileOffset: UInt64,
            storage: BF16AffineTensorStorage,
            components: QwenAffineComponentLayout?
        ) throws {
            let index = UInt64(requests.count)
            let requestID = digest(strings: [
                "qwen-transform-request-v2", plan.canonicalFingerprint,
                String(index), tensorName, source.name, source.shardPath,
                String(source.absoluteOffset), String(source.sizeBytes),
                source.shape.map(String.init).joined(separator: ","),
                destinationPath, String(destinationSize), String(fileOffset),
                storageIdentity(storage),
            ])
            requests.append(.init(
                requestIndex: index, requestID: requestID,
                tensorName: tensorName, source: source,
                sourcePayloadSHA256: sourceDigest,
                destinationPath: destinationPath,
                destinationFileSize: destinationSize,
                fileOffset: fileOffset, storage: storage,
                affineComponents: components))
        }

        for tensor in plan.textTensors {
            guard let sourceDigest = sourceDigests[tensor.source],
                  let destinationSize = artifactSizes[tensor.relativeFile] else {
                throw RepackError.configurationInvalid(
                    detail: "Qwen text transform proof has no planned artifact")
            }
            try append(
                tensorName: tensor.source.name, source: tensor.source,
                sourceDigest: sourceDigest,
                destinationPath: (textPartialDirectory as NSString)
                    .appendingPathComponent(tensor.relativeFile),
                destinationSize: destinationSize,
                fileOffset: tensor.fileOffset, storage: tensor.storage,
                components: tensor.affineComponents)
        }
        for layer in plan.expertLayers {
            let destination = (textPartialDirectory as NSString)
                .appendingPathComponent(layer.relativePath)
            for expert in 0..<layer.expertsPerLayer {
                for sourcePlan in layer.sources {
                    let source = sourcePlan.source
                    guard source.sizeBytes.isMultiple(
                        of: UInt64(layer.expertsPerLayer)) else {
                        throw RepackError.configurationInvalid(
                            detail: "Qwen expert source is not evenly strided")
                    }
                    let sliceBytes = source.sizeBytes / UInt64(layer.expertsPerLayer)
                    let expertOffset = try checkedMultiply(UInt64(expert), sliceBytes)
                    let slice = SourceTensor(
                        name: source.name, shardPath: source.shardPath,
                        dtype: source.dtype, shape: sourcePlan.perExpertShape,
                        absoluteOffset: try checkedAdd(source.absoluteOffset, expertOffset),
                        sizeBytes: sliceBytes)
                    // A slice digest captured after the full-source binding
                    // would authenticate mutable bytes to themselves. Exact
                    // routed bytes are instead bound by progress.sourceChain.
                    let sourceDigest: String? = nil
                    let base = sourcePlan.components.valuesOffset
                    let components = QwenAffineComponentLayout(
                        valuesOffset: 0,
                        valuesSize: sourcePlan.components.valuesSize,
                        scalesOffset: sourcePlan.components.scalesOffset - base,
                        scalesSize: sourcePlan.components.scalesSize,
                        biasesOffset: sourcePlan.components.biasesOffset - base,
                        biasesSize: sourcePlan.components.biasesSize)
                    try append(
                        tensorName: source.name + "#expert-" + String(expert),
                        source: slice, sourceDigest: sourceDigest,
                        destinationPath: destination,
                        destinationSize: layer.fileSize,
                        fileOffset: try checkedAdd(
                            try checkedMultiply(UInt64(expert), layer.expertStride), base),
                        storage: .affineInt4, components: components)
                }
            }
        }
        if let vision = plan.visionCompanion {
            guard let visionPartialDirectory else {
                throw RepackError.configurationInvalid(
                    detail: "Qwen vision transform has no destination")
            }
            for tensor in vision.tensors {
                guard let sourceDigest = sourceDigests[tensor.source] else {
                    throw RepackError.configurationInvalid(
                        detail: "Qwen vision transform has no source proof")
                }
                try append(
                    tensorName: tensor.source.name, source: tensor.source,
                    sourceDigest: sourceDigest,
                    destinationPath: (visionPartialDirectory as NSString)
                        .appendingPathComponent(tensor.relativeFile),
                    destinationSize: vision.artifact.size,
                    fileOffset: tensor.fileOffset, storage: tensor.storage,
                    components: tensor.affineComponents)
            }
        }
        return requests
    }

    private static func createArtifacts(
        _ artifacts: [QwenPlannedArtifact],
        root: String,
        layout: Data?
    ) throws {
        try Posix.mkdirP(root)
        var parentDirectories = Set<String>()
        for artifact in artifacts {
            let path = (root as NSString).appendingPathComponent(artifact.relativePath)
            let parent = (path as NSString).deletingLastPathComponent
            try Posix.mkdirP(parent)
            parentDirectories.insert(parent)
            if artifact.relativePath == "packed_experts/layout.json", let layout {
                guard UInt64(layout.count) == artifact.size else {
                    throw RepackError.configurationInvalid(
                        detail: "Qwen layout size changed after planning")
                }
                try Posix.atomicWrite(
                    layout, to: path,
                    durableIn: (path as NSString).deletingLastPathComponent)
            } else {
                let fd = try Posix.openCreateRW(path)
                do {
                    try Posix.ftruncate(fd, path: path, size: artifact.size)
                    try Posix.fsync(fd, path: path)
                    close(fd)
                } catch {
                    close(fd)
                    throw error
                }
            }
        }
        for directory in parentDirectories.sorted() {
            try Posix.fsyncDirectory(directory)
        }
        try Posix.fsyncDirectory(root)
    }

    private static func hashArtifacts(
        _ artifacts: [QwenPlannedArtifact],
        root: String,
        audit: RepackAudit,
        cancellationCheck: () throws -> Void
    ) throws -> [String: RepackAudit.OutputFile] {
        var result: [String: RepackAudit.OutputFile] = [:]
        for artifact in artifacts {
            let path = (root as NSString).appendingPathComponent(artifact.relativePath)
            let digest = try WriterCore.hashEntireFile(
                path: path, size: artifact.size, audit: audit,
                cancellationCheck: cancellationCheck)
            result[artifact.relativePath] = .init(
                relativePath: artifact.relativePath,
                size: artifact.size, sha256: digest)
        }
        return result
    }

    private static func validatedBinding(
        checkpoint: RemoteInstallCheckpoint,
        plan: QwenRepackPlan,
        textPaths: RemoteInstallPaths,
        visionPaths: RemoteInstallPaths?
    ) throws -> RemoteTransformBinding {
        guard try Posix.entryKind(textPaths.partialDirectory) == .directory,
              try Posix.entryKind(textPaths.checkpointFile) == .regular else {
            throw RepackError.installStateMissing(path: textPaths.checkpointFile)
        }
        if let visionPaths {
            guard try Posix.entryKind(visionPaths.partialDirectory) == .directory else {
                throw RepackError.installStateMissing(path: visionPaths.partialDirectory)
            }
        }
        let binding = try checkpoint.requireTransformBinding(
            path: textPaths.checkpointFile)
        let destinationBytes = try sum(plan.artifacts.map(\.size))
        let destinationIdentity = digest(strings: [
            textPaths.finalDirectory,
            visionPaths?.finalDirectory ?? "none",
            String(destinationBytes), plan.canonicalFingerprint,
        ])
        guard checkpoint.repoID == plan.provenance.repository,
              checkpoint.requestedRevision == plan.provenance.revision,
              checkpoint.resolvedCommit == plan.provenance.revision,
              checkpoint.sourceIndexSHA256 == plan.provenance.observedIndexSHA256,
              checkpoint.planFingerprint == plan.canonicalFingerprint,
              binding.planFingerprint == plan.canonicalFingerprint,
              binding.quantizationPolicySHA256
                == plan.provenance.quantizationPolicySHA256,
              binding.converterVersion == TransformedTensorWriter.converterVersion,
              binding.destinationIdentity == destinationIdentity,
              binding.destinationBytes == destinationBytes else {
            throw RepackError.installStateIncompatible(
                detail: "saved partial is not owned by this official Qwen plan")
        }
        return binding
    }

    private static func placeholderSourceDigests(
        plan: QwenRepackPlan
    ) -> [SourceTensor: String] {
        var result: [SourceTensor: String] = [:]
        for source in plan.textTensors.map(\.source)
            + plan.expertLayers.flatMap({ $0.sources.map(\.source) })
            + (plan.visionCompanion?.tensors.map(\.source) ?? []) {
            result[source] = String(repeating: "0", count: 64)
        }
        return result
    }

    private static func validatedResumePosition(
        progress: RemoteTransformProgress,
        requests: [TransformedTensorWriteRequest]
    ) throws -> TransformedTensorWriter.ResumePosition {
        guard progress.requestIndex < UInt64(requests.count) else {
            throw RepackError.installStateIncompatible(
                detail: "transformed progress exhausts its request schedule")
        }
        let index = try checkedInt(progress.requestIndex)
        let current = requests[index]
        let currentUnits = try TransformedTensorWriter.unitCount(request: current)
        guard progress.requestID == current.requestID,
              progress.completedUnitCount <= currentUnits else {
            throw RepackError.installStateIncompatible(
                detail: "transformed progress request changed")
        }
        var expectedTotal = progress.completedUnitCount
        if index > 0 {
            for prior in requests[..<index] {
                expectedTotal = try checkedAdd(
                    expectedTotal,
                    try TransformedTensorWriter.unitCount(request: prior))
            }
        }
        guard expectedTotal == progress.totalCompletedUnitCount else {
            throw RepackError.installStateIncompatible(
                detail: "transformed progress unit total changed")
        }
        if progress.completedUnitCount == currentUnits {
            return .init(requestIndex: index + 1, completedUnitCount: 0)
        }
        return .init(
            requestIndex: index,
            completedUnitCount: progress.completedUnitCount)
    }

    private static func remainingDestinationWriteBytes(
        request: TransformedTensorWriteRequest,
        completedUnitCount: UInt64
    ) throws -> UInt64 {
        let available = try TransformedTensorWriter.unitCount(request: request)
        guard completedUnitCount <= available else {
            throw RepackError.installStateIncompatible(
                detail: "transformed progress exceeds its request")
        }
        let total: UInt64
        let completed: UInt64
        switch request.storage {
        case .retainedBF16:
            total = request.source.sizeBytes
            completed = min(
                try checkedMultiply(
                    completedUnitCount, UInt64(WriterCore.tileBytes)), total)
        case .affineInt4, .affineInt8:
            guard let components = request.affineComponents,
                  let bitWidth = request.storage.affineBitWidth,
                  let elementsPerRow = request.source.shape.last else {
                throw RepackError.configurationInvalid(
                    detail: "affine transform has no component layout")
            }
            total = try sum([
                components.valuesSize, components.scalesSize, components.biasesSize,
            ])
            let groupSize = UInt64(BF16AffineQuantizationPolicy.affineGroupSize)
            let completeGroups = elementsPerRow / groupSize
            let remainder = elementsPerRow % groupSize
            let groupsPerRow = completeGroups + (remainder == 0 ? 0 : 1)
            let completeValueBytes = try checkedDivideRoundingUp(
                try checkedMultiply(groupSize, UInt64(bitWidth.rawValue)), by: 8)
            let remainderValueBytes = remainder == 0 ? 0
                : try checkedDivideRoundingUp(
                    try checkedMultiply(remainder, UInt64(bitWidth.rawValue)), by: 8)
            let valuesPerRow = try checkedAdd(
                try checkedMultiply(completeGroups, completeValueBytes),
                remainderValueBytes)
            let completeRows = completedUnitCount / groupsPerRow
            let localGroups = completedUnitCount % groupsPerRow
            var completedValues = try checkedMultiply(completeRows, valuesPerRow)
            completedValues = try checkedAdd(
                completedValues,
                try checkedMultiply(min(localGroups, completeGroups), completeValueBytes))
            if localGroups > completeGroups {
                completedValues = try checkedAdd(completedValues, remainderValueBytes)
            }
            completed = try checkedAdd(
                completedValues, try checkedMultiply(completedUnitCount, 4))
        case .omittedMTP:
            throw RepackError.configurationInvalid(
                detail: "omitted MTP tensor reached transformed writer")
        }
        guard completed <= total else {
            throw RepackError.installStateIncompatible(
                detail: "transformed completed bytes exceed request output")
        }
        return total - completed
    }

    private static func validateCanonicalLayout(
        plan: QwenRepackPlan,
        textPaths: RemoteInstallPaths
    ) throws {
        let expected = try layoutData(plan: plan)
        let path = (textPaths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts/layout.json")
        let observed = try Posix.readBoundedData(
            path, maximumBytes: UInt64(expected.count))
        guard observed == expected else {
            throw RepackError.installStateIncompatible(
                detail: "Qwen partial layout differs from the canonical plan")
        }
    }

    private static func validateOwnedInventories(
        plan: QwenRepackPlan,
        textPaths: RemoteInstallPaths,
        visionPaths: RemoteInstallPaths?,
        processorData: Data?
    ) throws {
        let textArtifacts = plan.artifacts.filter {
            $0.relativePath != plan.visionCompanion?.artifact.relativePath
        }
        try validateOwnedInventory(
            root: textPaths.partialDirectory,
            artifacts: textArtifacts,
            optionalPrefix: ["manifest.json", VerifiedInstallReceiptWriter.fileName])
        if let vision = plan.visionCompanion {
            guard let visionPaths, let processorData else {
                throw RepackError.installStateIncompatible(
                    detail: "owned Qwen vision partial is missing its plan binding")
            }
            try validateOwnedInventory(
                root: visionPaths.partialDirectory,
                artifacts: [vision.artifact],
                optionalPrefix: [
                    GTurboVisionFormatV2.processorFile,
                    "manifest.json", VerifiedInstallReceiptWriter.fileName,
                ])
            let processorPath = (visionPaths.partialDirectory as NSString)
                .appendingPathComponent(GTurboVisionFormatV2.processorFile)
            if try Posix.entryKind(processorPath) == .regular {
                let observed = try Posix.readBoundedData(
                    processorPath, maximumBytes: UInt64(processorData.count))
                guard observed == processorData else {
                    throw RepackError.installStateIncompatible(
                        detail: "owned Qwen vision processor differs from the plan")
                }
            }
        } else if visionPaths != nil {
            throw RepackError.installStateIncompatible(
                detail: "text-only Qwen partial has an unexpected vision destination")
        }
    }

    private static func validateOwnedInventory(
        root: String,
        artifacts: [QwenPlannedArtifact],
        optionalPrefix: [String]
    ) throws {
        let inventory = try partialInventory(root: root)
        let required = Set(artifacts.map(\.relativePath))
        let extras = inventory.files.subtracting(required)
        guard extras.isSubset(of: Set(optionalPrefix)) else {
            throw RepackError.installStateIncompatible(
                detail: "Qwen partial contains files outside its owned plan")
        }
        let prefixCount = optionalPrefix.prefix { extras.contains($0) }.count
        guard extras == Set(optionalPrefix.prefix(prefixCount)),
              inventory.files == required.union(extras) else {
            throw RepackError.installStateIncompatible(
                detail: "Qwen partial metadata is not a valid production prefix")
        }
        let expectedDirectories = Set(
            inventory.files.flatMap(parentDirectories))
        guard inventory.directories == expectedDirectories else {
            throw RepackError.installStateIncompatible(
                detail: "Qwen partial contains directories outside its owned plan")
        }
        for artifact in artifacts {
            let path = (root as NSString).appendingPathComponent(artifact.relativePath)
            let fd = try Posix.openReadNoFollow(path)
            defer { close(fd) }
            guard try Posix.fileSize(fd: fd, path: path) == artifact.size,
                  try Posix.descriptorMatchesPath(fd, path: path) else {
                throw RepackError.installStateIncompatible(
                    detail: "Qwen partial artifact changed: \(artifact.relativePath)")
            }
        }
        for name in extras {
            let path = (root as NSString).appendingPathComponent(name)
            _ = try Posix.readBoundedData(path, maximumBytes: 32 * 1024 * 1024)
        }
    }

    private static func partialInventory(
        root: String
    ) throws -> (files: Set<String>, directories: Set<String>) {
        var files = Set<String>()
        var directories = Set<String>()
        func visit(_ relativeDirectory: String) throws {
            let absolute = relativeDirectory.isEmpty ? root
                : (root as NSString).appendingPathComponent(relativeDirectory)
            for name in try FileManager.default.contentsOfDirectory(atPath: absolute) {
                guard name != ".", name != "..", !name.contains("/"),
                      !name.contains("\0") else {
                    throw RepackError.installStateIncompatible(
                        detail: "Qwen partial contains an unsafe entry")
                }
                let relative = relativeDirectory.isEmpty
                    ? name : relativeDirectory + "/" + name
                let path = (root as NSString).appendingPathComponent(relative)
                switch try Posix.entryKind(path) {
                case .regular: files.insert(relative)
                case .directory:
                    directories.insert(relative)
                    try visit(relative)
                case .absent, .symlink, .other:
                    throw RepackError.installStateIncompatible(
                        detail: "Qwen partial contains a non-regular entry")
                }
            }
        }
        try visit("")
        return (files, directories)
    }

    private static func parentDirectories(_ path: String) -> [String] {
        var parts = path.split(separator: "/").map(String.init)
        var result: [String] = []
        while parts.count > 1 {
            _ = parts.removeLast()
            result.append(parts.joined(separator: "/"))
        }
        return result
    }

    private static func checkedDivideRoundingUp(
        _ value: UInt64, by divisor: UInt64
    ) throws -> UInt64 {
        guard divisor > 0 else {
            throw RepackError.configurationInvalid(detail: "division by zero")
        }
        return value / divisor + (value.isMultiple(of: divisor) ? 0 : 1)
    }

    private static func existingRegularFileBytes(
        names: [String], root: String
    ) throws -> UInt64 {
        var total: UInt64 = 0
        for name in names {
            let path = (root as NSString).appendingPathComponent(name)
            switch try Posix.entryKind(path) {
            case .absent:
                continue
            case .regular:
                let fd = try Posix.openReadNoFollow(path)
                do {
                    let size = try Posix.fileSize(fd: fd, path: path)
                    guard try Posix.descriptorMatchesPath(fd, path: path) else {
                        throw RepackError.installStateIncompatible(
                            detail: "Qwen resume metadata changed during disk preflight")
                    }
                    close(fd)
                    total = try sum([total, size])
                } catch {
                    close(fd)
                    throw error
                }
            case .directory, .symlink, .other:
                throw RepackError.installStateIncompatible(
                    detail: "Qwen resume metadata is not a regular file: \(name)")
            }
        }
        return total
    }

    private static func sum(_ values: [UInt64]) throws -> UInt64 {
        try values.reduce(UInt64(0)) { partial, value in
            let addition = partial.addingReportingOverflow(value)
            guard !addition.overflow else {
                throw RepackError.configurationInvalid(detail: "Qwen byte total overflows")
            }
            return addition.partialValue
        }
    }

    private static func storageIdentity(_ storage: BF16AffineTensorStorage) -> String {
        switch storage {
        case .retainedBF16: "retained-bf16"
        case .affineInt4: "affine-int4"
        case .affineInt8: "affine-int8"
        case .omittedMTP: "omitted-mtp"
        }
    }

    private static func checkedAdd(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let result = lhs.addingReportingOverflow(rhs)
        guard !result.overflow else {
            throw RepackError.configurationInvalid(
                detail: "Qwen transform offset addition overflows")
        }
        return result.partialValue
    }

    private static func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow else {
            throw RepackError.configurationInvalid(
                detail: "Qwen transform offset multiplication overflows")
        }
        return result.partialValue
    }

    private static func checkedInt(_ value: UInt64) throws -> Int {
        guard value <= UInt64(Int.max) else {
            throw RepackError.configurationInvalid(
                detail: "Qwen transformed unit count exceeds Int")
        }
        return Int(value)
    }

    private static func digest(strings: [String]) -> String {
        digest(data: Data(strings.joined(separator: "\n").utf8))
    }

    private static func digest(data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private static func storage(_ storage: BF16AffineTensorStorage) -> GTurboStorageTypeV2 {
        switch storage {
        case .retainedBF16: .bf16
        case .affineInt4: .affineInt4
        case .affineInt8: .affineInt8
        case .omittedMTP: .bf16
        }
    }
}
