import Foundation
import ImageIO
import TurboFieldfare

/// Only an explicit MCP PNG image block supplies pixels. No URL, file path,
/// OCR result, or prose description is accepted as an image substitute.
enum VisionCaptureScreenshot {
    static let maximumImageBytes = 16 * 1_024 * 1_024
    static let maximumResponseBytes = 24 * 1_024 * 1_024

    /// Cache provenance is not pixel identity. Even equal generations cannot
    /// establish that a later accessibility tree matches this image.
    struct ObservationMetadata: Sendable, Equatable {
        var deviceID: String?
        var source: String?
        var cacheAgeMilliseconds: Int?
        var invalidationGeneration: Int?
    }

    /// Inspect structured metadata and its explicit text copies. Missing values
    /// stay unknown; contradictory or malformed supplied evidence fails closed.
    static func observationMetadata(in result: VisionCaptureMCPResult) throws -> ObservationMetadata {
        try observationMetadata(in: result.value)
    }

    static func observationMetadata(in value: JSONValue) throws -> ObservationMetadata {
        var metadata = ObservationMetadata()
        var inspectedNodes = 0
        var metadataTextBytes = 0

        func invalid(_ name: String) -> VisionImageError {
            .invalidSource("Screenshot observation metadata is invalid or conflicting: \(name).")
        }
        func merge<T: Equatable>(_ value: T, into stored: inout T?, name: String) throws {
            if let stored, stored != value { throw invalid(name) }
            stored = value
        }
        func string(_ value: JSONValue, name: String) throws -> String {
            guard case .string(let text) = value,
                  !text.isEmpty, text.utf8.count <= 256,
                  text == text.trimmingCharacters(in: .whitespacesAndNewlines) else {
                throw invalid(name)
            }
            return text
        }
        func integer(_ value: JSONValue, name: String) throws -> Int {
            let number: Int?
            switch value {
            case .integer(let value): number = Int(exactly: value)
            case .unsignedInteger(let value): number = Int(exactly: value)
            case .decimal(let value): number = Int(NSDecimalNumber(decimal: value).stringValue)
            case .number(let value): number = value.isFinite ? Int(exactly: value) : nil
            default: number = nil
            }
            guard let number, number >= 0 else { throw invalid(name) }
            return number
        }
        func observe(_ object: [String: JSONValue]) throws {
            if let value = object["source"] {
                let source = try string(value, name: "source")
                guard ["live", "cache", "coalesced"].contains(source) else { throw invalid("source") }
                try merge(source, into: &metadata.source, name: "source")
            }
            if let value = object["cache_age_ms"] {
                try merge(integer(value, name: "cache_age_ms"),
                          into: &metadata.cacheAgeMilliseconds, name: "cache_age_ms")
            }
            if let value = object["invalidation_generation"] {
                try merge(integer(value, name: "invalidation_generation"),
                          into: &metadata.invalidationGeneration, name: "invalidation_generation")
            }
        }
        func visit(_ value: JSONValue, depth: Int) throws {
            inspectedNodes += 1
            guard depth <= 32, inspectedNodes <= 100_000 else { throw invalid("metadata size") }
            switch value {
            case .object(let object):
                // Never traverse or copy encoded image/audio data.
                if object["type"] == .string("image") || object["type"] == .string("audio") { return }
                if let value = object["udid"] {
                    try merge(string(value, name: "udid").lowercased(),
                              into: &metadata.deviceID, name: "udid")
                }
                if let value = object["observe"] {
                    guard case .object(let fields) = value else { throw invalid("observe") }
                    try observe(fields)
                }
                if object["type"] == .string("text"), case .string(let text)? = object["text"] {
                    // Bound every text block before trimming, decoding or line
                    // parsing, including ordinary UI prose and nested copies.
                    let byteCount = text.utf8.count
                    guard byteCount <= 1_048_576 - metadataTextBytes else { throw invalid("metadata text size") }
                    metadataTextBytes += byteCount
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmed.first == "{" || trimmed.first == "[" {
                        guard let embedded = try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)) else {
                            throw invalid("embedded JSON")
                        }
                        try visit(embedded, depth: depth + 1)
                    } else {
                        // The server emits one exact five/six-line metadata
                        // block, alone for captures or appended after screen
                        // prose. Inspect only that bounded suffix, never split
                        // all screen text into an unbounded array of lines.
                        let block: Substring
                        if text.hasPrefix("observe_source: ") {
                            block = text[...]
                        } else if let marker = text.range(of: "\nobserve_source: ", options: .backwards) {
                            block = text[text.index(after: marker.lowerBound)...]
                        } else { block = "" }
                        if !block.isEmpty {
                            guard block.utf8.count <= 1_024 else { throw invalid("observe text size") }
                            let lines = block.split(separator: "\n", maxSplits: 6, omittingEmptySubsequences: false)
                            let names = ["source", "cached", "coalesced", "cache_ttl_ms", "invalidation_generation", "cache_age_ms"]
                            guard lines.count == 5 || lines.count == 6 else { throw invalid("observe text shape") }
                            var fields: [String: JSONValue] = [:]
                            for (index, line) in lines.enumerated() {
                                let name = names[index]
                                let prefix = "observe_\(name): "
                                guard line.hasPrefix(prefix) else { throw invalid("observe text shape") }
                                let raw = String(line.dropFirst(prefix.count))
                                if name == "source" { fields[name] = .string(raw) }
                                else if name == "cached" || name == "coalesced" {
                                    guard raw == "true" || raw == "false" else { throw invalid(name) }
                                } else {
                                    guard let number = Int64(raw), number >= 0 else { throw invalid(name) }
                                    fields[name] = .integer(number)
                                }
                            }
                            try observe(fields)
                        }
                    }
                }
                for child in object.values { try visit(child, depth: depth + 1) }
            case .array(let values):
                for child in values { try visit(child, depth: depth + 1) }
            default: break
            }
        }
        try visit(value, depth: 0)
        return metadata
    }

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
        _ result: VisionCaptureMCPResult, in store: AppImageAttachmentStore,
        expectedDeviceID: String? = nil
    ) throws -> AppImageAttachment {
        let metadata = try observationMetadata(in: result)
        if let expectedDeviceID, let deviceID = metadata.deviceID,
           deviceID != expectedDeviceID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            throw VisionImageError.invalidSource("Screenshot belongs to a different simulator device.")
        }
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
