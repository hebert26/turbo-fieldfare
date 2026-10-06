import Foundation
import Metal
import TurboFieldfareFormat

struct QwenVisionGrid: Equatable, Sendable {
    let temporal: Int
    let height: Int
    let width: Int

    init(temporal: Int, height: Int, width: Int) throws {
        guard temporal > 0, height > 0, width > 0 else {
            throw QwenVisionError.invalidPositions
        }
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    init(_ geometry: QwenImageGeometry) throws {
        try self.init(
            temporal: geometry.gridT,
            height: geometry.gridH,
            width: geometry.gridW)
    }

    func mergedRows(merge: Int) throws -> Int {
        guard merge > 0, height.isMultiple(of: merge), width.isMultiple(of: merge) else {
            throw QwenVisionError.invalidPositions
        }
        let (spatial, firstOverflow) = (height / merge).multipliedReportingOverflow(
            by: width / merge)
        let (rows, secondOverflow) = temporal.multipliedReportingOverflow(by: spatial)
        guard !firstOverflow, !secondOverflow else {
            throw QwenVisionError.arithmeticOverflow("merged grid rows")
        }
        return rows
    }
}

struct QwenMRoPEPosition: Equatable, Sendable {
    let temporal: Int32
    let height: Int32
    let width: Int32

    init(temporal: Int, height: Int, width: Int) throws {
        guard let temporal = Int32(exactly: temporal),
              let height = Int32(exactly: height),
              let width = Int32(exactly: width),
              temporal >= 0, height >= 0, width >= 0 else {
            throw QwenVisionError.invalidPositions
        }
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    private init(validatedTemporal: Int32, height: Int32, width: Int32) {
        temporal = validatedTemporal
        self.height = height
        self.width = width
    }

    static func read(from pointer: UnsafePointer<Int32>, row: Int) -> Self {
        Self(
            validatedTemporal: pointer[row * 3],
            height: pointer[row * 3 + 1],
            width: pointer[row * 3 + 2])
    }

    var values: [Int32] { [temporal, height, width] }

    func value(axis: Int) -> Int32 {
        switch axis {
        case 0: temporal
        case 1: height
        default: width
        }
    }
}

struct QwenMultimodalPositionPlan: Equatable, Sendable {
    let positions: [QwenMRoPEPosition]
    let textRoPEDelta: Int
    let imageRanges: [Range<Int>]
    let grids: [QwenVisionGrid]

    var flattenedInt32x3: [Int32] { positions.flatMap(\.values) }
}

enum QwenMultimodalPositions {
    static func make(
        tokenCount: Int,
        imageRanges: [Range<Int>],
        grids: [QwenVisionGrid],
        merge: Int = QwenVisionConfig.official.spatialMergeSize,
        maximumRows: Int = QwenVisionResourceLimits.provisional.maximumVisibleHistoryRows
    ) throws -> QwenMultimodalPositionPlan {
        guard tokenCount > 0, imageRanges.count == grids.count else {
            throw QwenVisionError.invalidPositions
        }
        var previousUpper = 0
        var totalRows = 0
        for (range, grid) in zip(imageRanges, grids) {
            let expected = try grid.mergedRows(merge: merge)
            guard !range.isEmpty, range.lowerBound >= previousUpper,
                  range.upperBound <= tokenCount, range.count == expected else {
                throw QwenVisionError.invalidPositions
            }
            let (next, overflow) = totalRows.addingReportingOverflow(expected)
            guard !overflow else {
                throw QwenVisionError.arithmeticOverflow("position rows")
            }
            totalRows = next
            previousUpper = range.upperBound
        }
        guard totalRows <= maximumRows else {
            throw QwenVisionError.mergedRowQuotaExceeded(
                requested: totalRows, maximum: maximumRows)
        }

        var result = [QwenMRoPEPosition]()
        result.reserveCapacity(tokenCount)
        var token = 0
        var scalar = 0
        for (range, grid) in zip(imageRanges, grids) {
            while token < range.lowerBound {
                result.append(try QwenMRoPEPosition(
                    temporal: scalar, height: scalar, width: scalar))
                scalar += 1
                token += 1
            }
            let base = scalar
            let mergedH = grid.height / merge
            let mergedW = grid.width / merge
            for temporal in 0..<grid.temporal {
                for height in 0..<mergedH {
                    for width in 0..<mergedW {
                        result.append(try QwenMRoPEPosition(
                            temporal: base + temporal,
                            height: base + height,
                            width: base + width))
                        token += 1
                    }
                }
            }
            scalar = base + max(grid.temporal, max(mergedH, mergedW))
        }
        while token < tokenCount {
            result.append(try QwenMRoPEPosition(
                temporal: scalar, height: scalar, width: scalar))
            scalar += 1
            token += 1
        }
        guard result.count == tokenCount else { throw QwenVisionError.invalidPositions }
        return QwenMultimodalPositionPlan(
            positions: result,
            textRoPEDelta: scalar - tokenCount,
            imageRanges: imageRanges,
            grids: grids)
    }

    /// Axis selected for one frequency in the first half of partial RoPE.
    static func axis(frequencyIndex: Int, sections: [Int]) throws -> Int {
        guard sections.count == 3, sections.allSatisfy({ $0 > 0 }),
              frequencyIndex >= 0, frequencyIndex < sections.reduce(0, +) else {
            throw QwenVisionError.invalidPositions
        }
        if frequencyIndex < sections[0] { return 0 }
        if frequencyIndex < sections[0] + sections[1] { return 1 }
        return 2
    }
}

/// Immutable retained rows. Neither Metal buffers nor mutable storage escape.
final class QwenRetainedFeatureOwner: @unchecked Sendable {
    let allocationID: UUID
    let rowCount: Int
    let hiddenSize: Int
    let featureBuffer: MTLBuffer
    let positionBuffer: MTLBuffer
    let imageDigest: String
    let processorDigest: String
    let profile: GTurboQwenVisionProcessorProfileV2
    let grid: QwenVisionGrid

    init(
        device: MTLDevice,
        features: [Float],
        positions: [QwenMRoPEPosition],
        imageDigest: String,
        processorDigest: String,
        profile: GTurboQwenVisionProcessorProfileV2,
        grid: QwenVisionGrid,
        hiddenSize: Int = QwenVisionConfig.official.outputHiddenSize
    ) throws {
        guard imageDigest.utf8.count == 64, processorDigest.utf8.count == 64,
              !positions.isEmpty,
              features.count == positions.count * hiddenSize,
              try grid.mergedRows(merge: profile.spatialMergeSize) == positions.count else {
            throw QwenVisionError.invalidFeatureShape
        }
        let featureBytes = features.count * MemoryLayout<Float>.stride
        let flattened = positions.flatMap(\.values)
        let positionBytes = flattened.count * MemoryLayout<Int32>.stride
        guard featureBytes <= device.maxBufferLength,
              positionBytes <= device.maxBufferLength,
              let featureBuffer = device.makeBuffer(
                length: featureBytes, options: .storageModeShared),
              let positionBuffer = device.makeBuffer(
                length: positionBytes, options: .storageModeShared) else {
            throw QwenVisionError.allocationFailed(name: "retained feature owner")
        }
        features.withUnsafeBytes { source in
            if let baseAddress = source.baseAddress {
                memcpy(featureBuffer.contents(), baseAddress, source.count)
            }
        }
        flattened.withUnsafeBytes { source in
            if let baseAddress = source.baseAddress {
                memcpy(positionBuffer.contents(), baseAddress, source.count)
            }
        }
        allocationID = UUID()
        rowCount = positions.count
        self.hiddenSize = hiddenSize
        self.featureBuffer = featureBuffer
        self.positionBuffer = positionBuffer
        self.imageDigest = imageDigest
        self.processorDigest = processorDigest
        self.profile = profile
        self.grid = grid
    }

    var requestedAllocationBytes: Int { featureBuffer.length + positionBuffer.length }

    var logicalBytes: Int {
        requestedAllocationBytes
            + imageDigest.utf8.count + processorDigest.utf8.count
            + 3 * MemoryLayout<Int>.stride
            + MemoryLayout<Int>.stride
    }

    func features() -> [Float] {
        let count = rowCount * hiddenSize
        return Array(UnsafeBufferPointer(
            start: featureBuffer.contents().assumingMemoryBound(to: Float.self),
            count: count))
    }

    func positions() -> [QwenMRoPEPosition] {
        let pointer = positionBuffer.contents().assumingMemoryBound(to: Int32.self)
        return (0..<rowCount).map {
            QwenMRoPEPosition.read(from: pointer, row: $0)
        }
    }
}

struct QwenImageLineage: Sendable {
    let owners: [QwenRetainedFeatureOwner]
    let textRoPEDelta: Int

    init(owners: [QwenRetainedFeatureOwner], textRoPEDelta: Int = 0) {
        var seen: Set<UUID> = []
        self.owners = owners.filter { seen.insert($0.allocationID).inserted }
        self.textRoPEDelta = textRoPEDelta
    }

    static let empty = QwenImageLineage(owners: [])

    var rowCount: Int { owners.reduce(0) { $0 + $1.rowCount } }
    var logicalBytes: Int { owners.reduce(0) { $0 + $1.logicalBytes } }
    var ownedRequestedBytes: Int {
        owners.reduce(0) { $0 + $1.requestedAllocationBytes }
    }
    var allocationIDs: Set<UUID> { Set(owners.map(\.allocationID)) }

    static func union(_ lineages: [QwenImageLineage]) -> QwenImageLineage {
        QwenImageLineage(owners: lineages.flatMap(\.owners))
    }
}
