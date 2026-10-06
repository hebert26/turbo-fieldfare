import Foundation

/// Metadata-only adjacent companion. It never names a second weights file:
/// each required tensor is resolved through the already admitted source index.
package struct OfficialSourceVisionDescriptor: Codable, Equatable, Sendable {
    package static let filename = "manifest.json"
    package static let kind = "official-source-vision-bf16-v1"
    package static let maximumBytes = 256 * 1024

    package struct Tensor: Codable, Equatable, Sendable {
        package let name: String
        package let shape: [UInt64]
        package init(name: String, shape: [UInt64]) {
            self.name = name
            self.shape = shape
        }
    }

    package let artifactKind: String
    package let textContentSHA256: String
    package let processorConfigSHA256: String
    package let processorProfile: GTurboQwenVisionProcessorProfileV2
    package let tensors: [Tensor]

    package init(textContentSHA256: String, processorConfigSHA256: String,
                 processorProfile: GTurboQwenVisionProcessorProfileV2,
                 tensors: [Tensor]) {
        artifactKind = Self.kind
        self.textContentSHA256 = textContentSHA256
        self.processorConfigSHA256 = processorConfigSHA256
        self.processorProfile = processorProfile
        self.tensors = tensors
    }

    /// Strict duplicate-aware metadata admission; callers must additionally
    /// match the current loaded text identity, processor and tensor contract.
    package static func decodeStrict(_ data: Data) throws -> Self {
        guard data.count <= maximumBytes else {
            throw OfficialSourceDescriptorError.invalid(field: "vision", reason: "metadata cap exceeded")
        }
        var scanner = SourceJSONDuplicateScanner(data: data)
        try scanner.validate()
        let value = try JSONDecoder().decode(Self.self, from: data)
        guard value.artifactKind == kind,
              [value.textContentSHA256, value.processorConfigSHA256].allSatisfy({ hash in
                  hash.utf8.count == 64 && hash.utf8.allSatisfy {
                      (48...57).contains($0) || (97...102).contains($0)
                  }
              }),
              !value.tensors.isEmpty, value.tensors.count <= 333,
              value.tensors.map(\.name) == value.tensors.map(\.name).sorted(),
              Set(value.tensors.map(\.name)).count == value.tensors.count,
              value.tensors.allSatisfy({ !$0.name.isEmpty && $0.name.utf8.count <= 160
                  && !$0.shape.isEmpty && $0.shape.count <= 5
                  && $0.shape.allSatisfy { $0 > 0 } }) else {
            throw OfficialSourceDescriptorError.invalid(field: "vision", reason: "invalid source companion")
        }
        // Reject unknown fields even when JSONDecoder would silently ignore them.
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == Set(["artifactKind", "textContentSHA256",
                                       "processorConfigSHA256", "processorProfile", "tensors"]),
              let profile = object["processorProfile"] as? [String: Any],
              Set(profile.keys) == Set(["processorClass", "imageProcessorType",
                                       "patchSize", "temporalPatchSize", "spatialMergeSize"]),
              let tensors = object["tensors"] as? [[String: Any]],
              tensors.allSatisfy({ Set($0.keys) == Set(["name", "shape"]) }) else {
            throw OfficialSourceDescriptorError.invalid(field: "vision", reason: "unknown metadata field")
        }
        return value
    }
}
