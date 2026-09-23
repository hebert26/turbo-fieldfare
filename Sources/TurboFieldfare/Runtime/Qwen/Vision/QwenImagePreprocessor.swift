import CoreGraphics
import Foundation
import ImageIO
import Metal

struct QwenImageGeometry: Equatable, Sendable {
    let processedWidth: Int
    let processedHeight: Int
    let gridT: Int
    let gridH: Int
    let gridW: Int
    let patchRows: Int
    let mergedRows: Int

    init(
        sourceWidth: Int,
        sourceHeight: Int,
        media: QwenVisionMedia = .stillImage,
        config: QwenVisionConfig = .official
    ) throws {
        guard media == .stillImage else { throw QwenVisionError.unsupportedVideo }
        try config.validate()
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw QwenVisionError.invalidDimensions(width: sourceWidth, height: sourceHeight)
        }
        let sourcePixels = try Self.multiply(
            sourceWidth, sourceHeight, operation: "source pixels")
        let factor = config.resizeFactor
        func rounded(_ value: Int) -> Int {
            max(factor, Int((Double(value) / Double(factor)).rounded()) * factor)
        }
        var width = rounded(sourceWidth)
        var height = rounded(sourceHeight)
        var pixels = try Self.multiply(width, height, operation: "rounded pixels")
        if pixels > config.publisherMaximumProcessedPixels {
            let beta = sqrt(
                Double(sourcePixels) / Double(config.publisherMaximumProcessedPixels))
            width = max(factor, Int(floor(
                Double(sourceWidth) / beta / Double(factor))) * factor)
            height = max(factor, Int(floor(
                Double(sourceHeight) / beta / Double(factor))) * factor)
        } else if pixels < config.minimumProcessedPixels {
            let beta = sqrt(Double(config.minimumProcessedPixels) / Double(sourcePixels))
            width = max(factor, Int(ceil(
                Double(sourceWidth) * beta / Double(factor))) * factor)
            height = max(factor, Int(ceil(
                Double(sourceHeight) * beta / Double(factor))) * factor)
        }
        pixels = try Self.multiply(width, height, operation: "processed pixels")
        guard pixels >= config.minimumProcessedPixels,
              pixels <= config.publisherMaximumProcessedPixels,
              width.isMultiple(of: factor), height.isMultiple(of: factor) else {
            throw QwenVisionError.invalidDimensions(width: width, height: height)
        }
        let gridH = height / config.patchSize
        let gridW = width / config.patchSize
        let patchRows = try Self.multiply(gridH, gridW, operation: "patch rows")
        guard patchRows <= config.maximumPatchRows else {
            throw QwenVisionError.patchRowQuotaExceeded(
                requested: patchRows, maximum: config.maximumPatchRows)
        }
        let mergeArea = try Self.multiply(
            config.spatialMergeSize, config.spatialMergeSize,
            operation: "merge area")
        guard gridH.isMultiple(of: config.spatialMergeSize),
              gridW.isMultiple(of: config.spatialMergeSize),
              patchRows.isMultiple(of: mergeArea) else {
            throw QwenVisionError.invalidDimensions(width: width, height: height)
        }
        let mergedRows = patchRows / mergeArea
        guard mergedRows <= config.maximumMergedRows else {
            throw QwenVisionError.mergedRowQuotaExceeded(
                requested: mergedRows, maximum: config.maximumMergedRows)
        }
        processedWidth = width
        processedHeight = height
        gridT = 1
        self.gridH = gridH
        self.gridW = gridW
        self.patchRows = patchRows
        self.mergedRows = mergedRows
    }

    private static func multiply(
        _ lhs: Int, _ rhs: Int, operation: String
    ) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw QwenVisionError.arithmeticOverflow(operation) }
        return result
    }
}

struct QwenImagePlan {
    let metadata: VisionImageMetadata
    let geometry: QwenImageGeometry
    let opened: OpenedVisionImage
    let imageDigest: String
    let started: ContinuousClock.Instant
}

struct QwenVisionPixelBuffer {
    let patchesBF16: MTLBuffer
    /// Block-major raw-patch positions `[row, height, width]`.
    let positionsInt32x2: MTLBuffer
    let metadata: VisionImageMetadata
    let geometry: QwenImageGeometry
    let imageDigest: String
    let wallNanoseconds: UInt64
    /// Requested bytes owned by decode/staging/output buffers. This is not a
    /// process-footprint measurement.
    let allocatedBytes: Int
}

final class QwenImagePreprocessor {
    private static let normalizedBF16 = (0...255).map {
        Quantization.bf16Bits(2 * (Float($0) / 255) - 1)
    }

    private let device: MTLDevice
    private let config: QwenVisionConfig
    private let metadataReader: ImageMetadataReader

    init(
        device: MTLDevice,
        config: QwenVisionConfig = .official,
        limits: VisionImageLimits = VisionImageLimits()
    ) {
        self.device = device
        self.config = config
        metadataReader = ImageMetadataReader(limits: limits)
    }

    func geometry(
        width: Int, height: Int, media: QwenVisionMedia = .stillImage
    ) throws -> QwenImageGeometry {
        try QwenImageGeometry(
            sourceWidth: width, sourceHeight: height,
            media: media, config: config)
    }

    func plan(
        _ image: VisionImageSource,
        media: QwenVisionMedia = .stillImage
    ) throws -> QwenImagePlan {
        guard media == .stillImage else { throw QwenVisionError.unsupportedVideo }
        let started = ContinuousClock.now
        let opened = try image.open(
            maximumEncodedBytes: metadataReader.limits.maximumEncodedBytes)
        let metadata = try metadataReader.read(opened: opened)
        let geometry = try geometry(
            width: metadata.orientedWidth, height: metadata.orientedHeight,
            media: media)
        let digest = try Sha256Verifier.hashFile(at: image.fileURL)
        return QwenImagePlan(
            metadata: metadata, geometry: geometry,
            opened: opened, imageDigest: digest, started: started)
    }

    func plan(
        fileURL: URL, media: QwenVisionMedia = .stillImage
    ) throws -> QwenImagePlan {
        try plan(VisionImageSource(fileURL: fileURL), media: media)
    }

    /// Checks the whole request before Qwen scratch or feature allocation.
    static func preflight(
        _ geometries: [QwenImageGeometry],
        limits: QwenVisionResourceLimits = .provisional
    ) throws {
        var rows = 0
        for geometry in geometries {
            let (next, overflow) = rows.addingReportingOverflow(geometry.mergedRows)
            guard !overflow else {
                throw QwenVisionError.arithmeticOverflow("aggregate merged rows")
            }
            rows = next
        }
        guard rows <= limits.maximumVisibleHistoryRows else {
            throw QwenVisionError.mergedRowQuotaExceeded(
                requested: rows, maximum: limits.maximumVisibleHistoryRows)
        }
    }

    func preprocess(_ plan: QwenImagePlan) throws -> QwenVisionPixelBuffer {
        let metadata = plan.metadata
        let geometry = plan.geometry
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize:
                max(metadata.orientedWidth, metadata.orientedHeight),
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceShouldAllowFloat: false,
        ]
        guard let decoded = CGImageSourceCreateThumbnailAtIndex(
            plan.opened.source, 0, options as CFDictionary) else {
            throw VisionImageError.decodeFailed
        }
        let sourceRowBytes = try multiply(decoded.width, 4, "source row bytes")
        let sourceBytes = try multiply(sourceRowBytes, decoded.height, "source bytes")
        let sourceRGBA = UnsafeMutableRawPointer.allocate(
            byteCount: sourceBytes, alignment: 64)
        defer { sourceRGBA.deallocate() }
        sourceRGBA.initializeMemory(as: UInt8.self, repeating: 255, count: sourceBytes)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: sourceRGBA, width: decoded.width, height: decoded.height,
                bitsPerComponent: 8, bytesPerRow: sourceRowBytes,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw VisionImageError.allocationFailed
        }
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
        context.draw(decoded, in: CGRect(
            x: 0, y: 0, width: decoded.width, height: decoded.height))
        withExtendedLifetime(context) {}

        let targetRowBytes = try multiply(geometry.processedWidth, 4, "target row bytes")
        let targetBytes = try multiply(
            targetRowBytes, geometry.processedHeight, "target bytes")
        let targetRGBA = UnsafeMutableRawPointer.allocate(
            byteCount: targetBytes, alignment: 64)
        defer { targetRGBA.deallocate() }
        let resizeScratchBytes = TorchBicubicResize.resize(
            source: sourceRGBA.assumingMemoryBound(to: UInt8.self),
            sourceWidth: decoded.width, sourceHeight: decoded.height,
            sourceRowBytes: sourceRowBytes,
            destination: targetRGBA.assumingMemoryBound(to: UInt8.self),
            destinationWidth: geometry.processedWidth,
            destinationHeight: geometry.processedHeight,
            destinationRowBytes: targetRowBytes)

        let patchElements = try multiply(
            geometry.patchRows, config.patchWidth, "patch elements")
        let patchBytes = try multiply(
            patchElements, MemoryLayout<UInt16>.stride, "patch bytes")
        let positionElements = try multiply(
            geometry.patchRows, 2, "position elements")
        let positionBytes = try multiply(
            positionElements, MemoryLayout<Int32>.stride, "position bytes")
        guard patchBytes <= device.maxBufferLength,
              positionBytes <= device.maxBufferLength else {
            throw QwenVisionError.bufferExceedsDevice(
                name: "preprocessed image", requested: max(patchBytes, positionBytes),
                maximum: device.maxBufferLength)
        }
        guard let patches = device.makeBuffer(
                length: patchBytes, options: .storageModeShared),
              let positions = device.makeBuffer(
                length: positionBytes, options: .storageModeShared) else {
            throw QwenVisionError.allocationFailed(name: "preprocessed image")
        }
        patchify(
            rgba: targetRGBA.assumingMemoryBound(to: UInt8.self),
            rowBytes: targetRowBytes, geometry: geometry,
            patches: patches, positions: positions)
        let allocated = try sum([
            sourceBytes, targetBytes, resizeScratchBytes, patchBytes, positionBytes,
        ], "preprocessor allocation bytes")
        return QwenVisionPixelBuffer(
            patchesBF16: patches, positionsInt32x2: positions,
            metadata: metadata, geometry: geometry,
            imageDigest: plan.imageDigest,
            wallNanoseconds: nanoseconds(plan.started.duration(to: .now)),
            allocatedBytes: allocated)
    }

    private func patchify(
        rgba: UnsafePointer<UInt8>, rowBytes: Int,
        geometry: QwenImageGeometry,
        patches: MTLBuffer, positions: MTLBuffer
    ) {
        let patchPointer = patches.contents().bindMemory(
            to: UInt16.self, capacity: geometry.patchRows * config.patchWidth)
        let positionPointer = positions.contents().bindMemory(
            to: Int32.self, capacity: geometry.patchRows * 2)
        let merge = config.spatialMergeSize
        var row = 0
        for blockY in 0..<(geometry.gridH / merge) {
            for blockX in 0..<(geometry.gridW / merge) {
                for innerY in 0..<merge {
                    for innerX in 0..<merge {
                        let patchY = blockY * merge + innerY
                        let patchX = blockX * merge + innerX
                        positionPointer[row * 2] = Int32(patchY)
                        positionPointer[row * 2 + 1] = Int32(patchX)
                        var output = row * config.patchWidth
                        // Official flattening order is C,T,H,W. A still image
                        // supplies identical values at both temporal indices.
                        for channel in 0..<config.inputChannels {
                            for _ in 0..<config.temporalPatchSize {
                                for pixelY in 0..<config.patchSize {
                                    let inputRow = (patchY * config.patchSize + pixelY)
                                        * rowBytes
                                    for pixelX in 0..<config.patchSize {
                                        let input = inputRow
                                            + (patchX * config.patchSize + pixelX) * 4
                                            + channel
                                        patchPointer[output] = Self.normalizedBF16[
                                            Int(rgba[input])]
                                        output += 1
                                    }
                                }
                            }
                        }
                        row += 1
                    }
                }
            }
        }
    }

    private func multiply(_ lhs: Int, _ rhs: Int, _ operation: String) throws -> Int {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw QwenVisionError.arithmeticOverflow(operation) }
        return result
    }

    private func sum(_ values: [Int], _ operation: String) throws -> Int {
        try values.reduce(0) { partial, value in
            let (result, overflow) = partial.addingReportingOverflow(value)
            guard !overflow else { throw QwenVisionError.arithmeticOverflow(operation) }
            return result
        }
    }

    private func nanoseconds(_ duration: Duration) -> UInt64 {
        let components = duration.components
        let seconds = UInt64(max(0, components.seconds))
        let fractional = UInt64(max(0, components.attoseconds / 1_000_000_000))
        return seconds * 1_000_000_000 + fractional
    }
}
