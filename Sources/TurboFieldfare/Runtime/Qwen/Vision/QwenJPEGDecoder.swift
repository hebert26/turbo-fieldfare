import Foundation
import TurboFieldfareJPEGBridge

/// Byte-oriented CPU output. JPEG decoding and EXIF permutation do not apply
/// an ICC transform, matching the pinned processor's Pillow RGB input.
struct QwenJPEGDecodedImage: Sendable {
    let width: Int
    let height: Int
    let rgba: Data
    /// Peak explicit buffers: encoded bytes, native RGB and oriented RGBA.
    let allocatedBytes: Int
}

enum QwenJPEGDecoder {
    static func decode(
        encoded: Data, metadata: VisionImageMetadata,
        limits: VisionImageLimits = VisionImageLimits()
    ) throws -> QwenJPEGDecodedImage {
        guard !encoded.isEmpty, encoded.count <= limits.maximumEncodedBytes else {
            throw VisionImageError.sourceTooLarge(bytes: encoded.count, limit: limits.maximumEncodedBytes)
        }
        guard encoded.count == metadata.encodedBytes else {
            throw VisionImageError.invalidSource("JPEG encoded byte count differs from admitted metadata")
        }
        let width = metadata.encodedWidth
        let height = metadata.encodedHeight
        guard width > 0, height > 0 else {
            throw VisionImageError.invalidMetadata("JPEG dimensions must be positive")
        }
        guard width <= limits.maximumSourceDimension, height <= limits.maximumSourceDimension else {
            throw VisionImageError.sideTooLarge(width: width, height: height, sideLimit: limits.maximumSourceDimension)
        }
        let (pixels, pixelOverflow) = width.multipliedReportingOverflow(by: height)
        guard !pixelOverflow, pixels <= limits.maximumSourcePixels else {
            throw VisionImageError.dimensionsTooLarge(width: width, height: height, pixelLimit: limits.maximumSourcePixels)
        }
        let (rgbaBytes, byteOverflow) = pixels.multipliedReportingOverflow(by: 4)
        guard !byteOverflow, rgbaBytes <= limits.maximumDecodedBytes else {
            throw VisionImageError.decodedBytesTooLarge(width: width, height: height, byteLimit: limits.maximumDecodedBytes)
        }
        guard (1...8).contains(metadata.orientation) else {
            throw VisionImageError.invalidMetadata("unsupported EXIF orientation \(metadata.orientation)")
        }
        let swapsAxes = metadata.orientation >= 5
        let orientedWidth = swapsAxes ? height : width
        let orientedHeight = swapsAxes ? width : height
        guard metadata.orientedWidth == orientedWidth, metadata.orientedHeight == orientedHeight else {
            throw VisionImageError.invalidMetadata("JPEG oriented dimensions differ from admitted metadata")
        }
        var decoded = tf_jpeg_rgb()
        defer { tf_jpeg_rgb_free(&decoded) }
        var error = [CChar](repeating: 0, count: 512)
        let status = encoded.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            error.withUnsafeMutableBufferPointer { message in
                tf_jpeg_decode_rgb(
                    bytes.baseAddress!.assumingMemoryBound(to: UInt8.self), bytes.count,
                    limits.maximumSourcePixels, limits.maximumDecodedBytes,
                    &decoded, message.baseAddress!, message.count)
            }
        }
        guard status == 0 else {
            let message = String(cString: error)
            throw VisionImageError.invalidSource("Qwen JPEG decode failed: \(message)")
        }
        guard Int(decoded.width) == width, Int(decoded.height) == height,
              decoded.byte_count == pixels * 3, let rgb = decoded.pixels else {
            throw VisionImageError.invalidMetadata("JPEG decoder dimensions differ from admitted metadata")
        }
        var rgba = Data(count: rgbaBytes)
        rgba.withUnsafeMutableBytes { (buffer: UnsafeMutableRawBufferPointer) in
            let output = buffer.bindMemory(to: UInt8.self)
            for y in 0..<height {
                for x in 0..<width {
                    let destination: (Int, Int)
                    switch metadata.orientation {
                    case 1: destination = (x, y)
                    case 2: destination = (width - 1 - x, y)
                    case 3: destination = (width - 1 - x, height - 1 - y)
                    case 4: destination = (x, height - 1 - y)
                    case 5: destination = (y, x)
                    case 6: destination = (height - 1 - y, x)
                    case 7: destination = (height - 1 - y, width - 1 - x)
                    default: destination = (y, width - 1 - x)
                    }
                    let from = (y * width + x) * 3
                    let to = (destination.1 * orientedWidth + destination.0) * 4
                    output[to] = rgb[from]
                    output[to + 1] = rgb[from + 1]
                    output[to + 2] = rgb[from + 2]
                    output[to + 3] = 255
                }
            }
        }
        let (first, firstOverflow) = encoded.count.addingReportingOverflow(decoded.byte_count)
        let (allocated, totalOverflow) = first.addingReportingOverflow(rgbaBytes)
        guard !firstOverflow, !totalOverflow else {
            throw VisionImageError.invalidMetadata("JPEG allocation byte count overflow")
        }
        return QwenJPEGDecodedImage(width: orientedWidth, height: orientedHeight,
                                    rgba: rgba, allocatedBytes: allocated)
    }
}
