import Foundation
import Testing
@testable import TurboFieldfareFormat

@Suite struct GTurboVisionFormatV2Tests {
    private let processorDigest = "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"

    private func manifest() -> GTurboVisionManifestV2 {
        GTurboVisionManifestV2(
            family: .qwen3_6, modelID: GTurboFormatV2.qwenRepository,
            sourceRevision: GTurboFormatV2.qwenRevision,
            processorProfile: .init(processorClass: "Qwen3VLProcessor",
                                    imageProcessorType: "Qwen2VLImageProcessorFast",
                                    patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2),
            processorConfigSHA256: processorDigest,
            compatibleTextManifestSHA256: V2TestFixture.digest,
            visionPayloadSHA256: V2TestFixture.digest,
            supportsStillImages: true, supportsVideo: false,
            files: ["vision_weights.bin": .init(size: 16_384, sha256: V2TestFixture.digest),
                    "preprocessor_config.json": .init(size: 1, sha256: processorDigest)],
            tensorRegions: [.init(name: "vision.embed", file: "vision_weights.bin", offset: 0,
                                  size: 16, shape: [1], storage: .bf16)])
    }

    @Test func validQwenVisionDocumentUsesSeparateV2Dispatch() throws {
        let data = try GTurboVisionManifestV2Codec.encode(manifest())
        guard case .v2 = try GTurboVisionManifestDocumentCodec.decode(data) else {
            Issue.record("expected Qwen v2 vision document"); return
        }
    }

    @Test(arguments: ["family", "sourceRevision", "processorConfigSHA256", "visionPayloadSHA256"])
    func rejectsChangedBindingIdentity(_ key: String) throws {
        let data = try GTurboVisionManifestV2Codec.encode(manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root[key] = key == "family" ? "gemma4" : String(repeating: "b", count: 64)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
    }

    @Test func rejectsWrongProcessorProfile() throws {
        let data = try GTurboVisionManifestV2Codec.encode(manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var profile = try #require(root["processorProfile"] as? [String: Any])
        profile["patchSize"] = 15
        root["processorProfile"] = profile
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
    }

    @Test func rejectsVideoAndPayloadBindingMismatch() throws {
        let data = try GTurboVisionManifestV2Codec.encode(manifest())
        var root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["supportsVideo"] = true
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
        root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        root["visionPayloadSHA256"] = String(repeating: "b", count: 64)
        #expect(throws: GTurboFormatError.self) {
            try GTurboVisionManifestV2Codec.decode(JSONSerialization.data(withJSONObject: root))
        }
    }
}
