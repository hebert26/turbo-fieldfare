import Foundation
import Testing
@testable import TurboFieldfareFormat

@Suite struct InstalledModelDescriptorTests {
    @Test func descriptorRoundTripsWithoutChangingValidatedIdentity() throws {
        let verified = try GTurboManifestV2Codec.decode(
            GTurboManifestV2Codec.encode(V2TestFixture.manifest()))
        let data = try JSONEncoder().encode(verified.descriptor)
        let decoded = try JSONDecoder().decode(InstalledModelDescriptor.self, from: data)
        #expect(decoded == verified.descriptor)
        #expect(decoded.quantization.count == GTurboQuantizationCategoryV2.allCases.count)
        #expect(decoded.vision == .unavailable)
    }

    @Test func descriptorRejectsForgedPinnedIdentityOnDecode() throws {
        let verified = try GTurboManifestV2Codec.decode(
            GTurboManifestV2Codec.encode(V2TestFixture.manifest()))
        var root = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(verified.descriptor)) as? [String: Any])
        root["modelID"] = "display-label-is-not-identity"
        let forged = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: GTurboFormatError.self) {
            try JSONDecoder().decode(InstalledModelDescriptor.self, from: forged)
        }
    }

    @Test func verifiedVisionRequiresBoundMatchingCompanion() throws {
        let text = try GTurboManifestV2Codec.decode(
            GTurboManifestV2Codec.encode(V2TestFixture.manifest()))
        let processorDigest = try #require(
            GTurboFormatV2.qwenSidecarSHA256["preprocessor_config.json"])
        func vision(textDigest: String) -> GTurboVisionManifestV2 {
            GTurboVisionManifestV2(
                family: .qwen3_6, modelID: GTurboFormatV2.qwenRepository,
                sourceRevision: GTurboFormatV2.qwenRevision,
                processorProfile: .init(processorClass: "Qwen3VLProcessor", imageProcessorType: "Qwen2VLImageProcessorFast", patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2),
                processorConfigSHA256: processorDigest,
                compatibleTextManifestSHA256: textDigest, visionPayloadSHA256: V2TestFixture.digest,
                supportsStillImages: true, supportsVideo: false,
                files: ["vision_weights.bin": .init(size: 16_384, sha256: V2TestFixture.digest),
                        "preprocessor_config.json": .init(size: 1, sha256: processorDigest)],
                tensorRegions: [.init(name: "vision", file: "vision_weights.bin", offset: 0, size: 16, shape: [1], storage: .bf16)])
        }
        let mismatched = try GTurboVisionManifestV2Codec.decode(
            GTurboVisionManifestV2Codec.encode(vision(textDigest: V2TestFixture.digest)))
        #expect(throws: GTurboFormatError.self) { try text.descriptor(binding: mismatched) }
        let bound = try GTurboVisionManifestV2Codec.decode(
            GTurboVisionManifestV2Codec.encode(vision(textDigest: text.manifestSHA256)))
        let descriptor = try text.descriptor(binding: bound)
        guard case .verified = descriptor.vision else { Issue.record("bound companion was lost"); return }

        func forgeCompanionDigest(_ value: Any) -> Any {
            if var object = value as? [String: Any] {
                if object["compatibleTextManifestSHA256"] != nil {
                    object["compatibleTextManifestSHA256"] = String(repeating: "b", count: 64)
                }
                for (key, nested) in object { object[key] = forgeCompanionDigest(nested) }
                return object
            }
            if let values = value as? [Any] { return values.map(forgeCompanionDigest) }
            return value
        }
        let root = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(descriptor)) as? [String: Any])
        let forged = try JSONSerialization.data(withJSONObject: forgeCompanionDigest(root))
        #expect(throws: GTurboFormatError.self) {
            try JSONDecoder().decode(InstalledModelDescriptor.self, from: forged)
        }
    }
}
