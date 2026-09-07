import Foundation
import ImageIO
import TurboFieldfare

/// Only an explicit MCP PNG image block supplies pixels. No URL, file path,
/// OCR result, or prose description is accepted as an image substitute.
enum VisionCaptureScreenshot {
    static let maximumImageBytes = 16 * 1_024 * 1_024
    static let maximumResponseBytes = 24 * 1_024 * 1_024

    /// A separate, bounded display copy. It never replaces the original model
    /// attachment and is made off the main actor before that file is released.
    static func stagePreview(
        of image: AppImageAttachment, in store: AppImageAttachmentStore
    ) throws -> AppImageAttachment {
        guard let source = CGImageSourceCreateWithURL(
            image.fileURL as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
              let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 512,
                kCGImageSourceShouldCacheImmediately: true,
              ] as CFDictionary) else {
            throw VisionImageError.invalidSource("Screenshot preview could not be decoded.")
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else {
            throw VisionImageError.invalidSource("Screenshot preview could not be encoded.")
        }
        CGImageDestinationAddImage(destination, thumbnail, nil)
        guard CGImageDestinationFinalize(destination), data.length <= 1_100_000 else {
            throw VisionImageError.invalidSource("Screenshot preview exceeded its display budget.")
        }
        return try store.stage(data: data as Data, displayName: "Screenshot preview.png")
    }

    static func stage(
        _ result: VisionCaptureMCPResult, in store: AppImageAttachmentStore
    ) throws -> AppImageAttachment {
        guard !result.isError,
              case .object(let root) = result.value,
              case .array(let content)? = root["content"] else {
            throw VisionImageError.invalidSource("Screenshot response has no MCP content.")
        }
        let images = content.compactMap { block -> [String: JSONValue]? in
            guard case .object(let value) = block, value["type"] == .string("image") else { return nil }
            return value
        }
        guard images.count == 1,
              images[0]["mimeType"] == .string("image/png"),
              case .string(let encoded)? = images[0]["data"],
              encoded.utf8.count <= ((maximumImageBytes + 2) / 3) * 4,
              let data = Data(base64Encoded: encoded), !data.isEmpty,
              data.count <= maximumImageBytes else {
            throw VisionImageError.invalidSource("Screenshot requires exactly one bounded PNG image block.")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              CGImageSourceGetType(source) as String? == "public.png",
              CGImageSourceGetCount(source) == 1,
              CGImageSourceGetStatus(source) == .statusComplete,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
              width.intValue > 0, height.intValue > 0,
              width.intValue <= 8_192, height.intValue <= 8_192,
              width.intValue * height.intValue <= 16_777_216 else {
            throw VisionImageError.invalidSource("Screenshot PNG dimensions or image data are invalid.")
        }
        // Existing attachment storage seals and hashes the file. The existing
        // vision preprocessor also verifies the complete PNG before encoding.
        return try store.stage(data: data, displayName: "VisionCapture screenshot.png")
    }
}
