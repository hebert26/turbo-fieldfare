import Foundation
import Testing
@testable import TurboFieldfareFormat

@Suite struct OfficialSourceVisionDescriptorTests {
    private let profile = GTurboQwenVisionProcessorProfileV2(
        processorClass: "Qwen3VLProcessor",
        imageProcessorType: "Qwen2VLImageProcessorFast",
        patchSize: 2,
        temporalPatchSize: 1,
        spatialMergeSize: 2)

    private func descriptor(
        tensors: [OfficialSourceVisionDescriptor.Tensor]? = nil,
        textDigest: String = String(repeating: "a", count: 64),
        processorDigest: String = String(repeating: "b", count: 64)
    ) -> OfficialSourceVisionDescriptor {
        OfficialSourceVisionDescriptor(
            textContentSHA256: textDigest,
            processorConfigSHA256: processorDigest,
            processorProfile: profile,
            tensors: tensors ?? [
                .init(name: "model.visual.blocks.0.attn.qkv.bias", shape: [48]),
                .init(name: "model.visual.blocks.0.attn.qkv.weight", shape: [48, 16]),
            ])
    }

    private func encoded(_ value: OfficialSourceVisionDescriptor) throws -> Data {
        try JSONEncoder().encode(value)
    }

    private func expectRejected(_ data: Data, sourceLocation: SourceLocation = #_sourceLocation) {
        #expect(throws: OfficialSourceDescriptorError.self,
                sourceLocation: sourceLocation) {
            try OfficialSourceVisionDescriptor.decodeStrict(data)
        }
    }

    @Test func canonicalMetadataRoundTripsWithoutPayloadFields() throws {
        let value = descriptor()
        let data = try encoded(value)
        let decoded = try OfficialSourceVisionDescriptor.decodeStrict(data)
        #expect(value.artifactKind == "official-source-vision-bf16-v1")
        #expect(OfficialSourceVisionDescriptor.filename == "manifest.json")
        #expect(decoded == value)
        #expect(data.count <= OfficialSourceVisionDescriptor.maximumBytes)

        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(Set(object.keys) == Set([
            "artifactKind", "textContentSHA256", "processorConfigSHA256",
            "processorProfile", "tensors",
        ]))
        #expect(object["files"] == nil)
        #expect(object["weightsFile"] == nil)
        #expect((object["tensors"] as? [[String: Any]])?.allSatisfy {
            Set($0.keys) == Set(["name", "shape"])
        } == true)
    }

    @Test func acceptsMetadataExactlyAtBoundAndRejectsOneByteOver() throws {
        var atBound = try encoded(descriptor())
        #expect(atBound.count < OfficialSourceVisionDescriptor.maximumBytes)
        atBound.append(Data(repeating: 0x20,
                           count: OfficialSourceVisionDescriptor.maximumBytes - atBound.count))
        #expect(try OfficialSourceVisionDescriptor.decodeStrict(atBound) == descriptor())
        atBound.append(0x20)
        expectRejected(atBound)
    }

    @Test(arguments: ["artifactKind", "textContentSHA256", "processorConfigSHA256", "processorProfile", "tensors"])
    func rejectsUnknownRootFields(_ field: String) throws {
        var object = try #require(JSONSerialization.jsonObject(
            with: encoded(descriptor())) as? [String: Any])
        object["unexpected_\(field)"] = NSNull()
        expectRejected(try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
    }

    @Test func rejectsUnknownProfileAndTensorFields() throws {
        var object = try #require(JSONSerialization.jsonObject(
            with: encoded(descriptor())) as? [String: Any])
        var profileObject = try #require(object["processorProfile"] as? [String: Any])
        profileObject["unexpected"] = true
        object["processorProfile"] = profileObject
        expectRejected(try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))

        object = try #require(JSONSerialization.jsonObject(
            with: encoded(descriptor())) as? [String: Any])
        var tensors = try #require(object["tensors"] as? [[String: Any]])
        tensors[0]["dtype"] = "BF16"
        object["tensors"] = tensors
        expectRejected(try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
    }

    @Test func rejectsDuplicateRootKey() throws {
        let hash = String(repeating: "a", count: 64)
        let json = """
        {"artifactKind":"official-source-vision-bf16-v1","artifactKind":"official-source-vision-bf16-v1",
         "textContentSHA256":"\(hash)","processorConfigSHA256":"\(hash)",
         "processorProfile":{"processorClass":"Qwen3VLProcessor","imageProcessorType":"Qwen2VLImageProcessorFast","patchSize":2,"temporalPatchSize":1,"spatialMergeSize":2},
         "tensors":[{"name":"model.visual.blocks.0.attn.qkv.bias","shape":[48]},{"name":"model.visual.blocks.0.attn.qkv.weight","shape":[48,16]}]}
        """
        expectRejected(Data(json.utf8))
    }

    @Test func rejectsDuplicateProfileKey() throws {
        let hash = String(repeating: "a", count: 64)
        let json = """
        {"artifactKind":"official-source-vision-bf16-v1",
         "textContentSHA256":"\(hash)","processorConfigSHA256":"\(hash)",
         "processorProfile":{"processorClass":"Qwen3VLProcessor","imageProcessorType":"Qwen2VLImageProcessorFast","patchSize":2,"temporalPatchSize":1,"spatialMergeSize":2,"patchSize":2},
         "tensors":[{"name":"model.visual.blocks.0.attn.qkv.bias","shape":[48]},{"name":"model.visual.blocks.0.attn.qkv.weight","shape":[48,16]}]}
        """
        expectRejected(Data(json.utf8))
    }

    @Test func rejectsDuplicateTensorKey() throws {
        let hash = String(repeating: "a", count: 64)
        let json = """
        {"artifactKind":"official-source-vision-bf16-v1",
         "textContentSHA256":"\(hash)","processorConfigSHA256":"\(hash)",
         "processorProfile":{"processorClass":"Qwen3VLProcessor","imageProcessorType":"Qwen2VLImageProcessorFast","patchSize":2,"temporalPatchSize":1,"spatialMergeSize":2},
         "tensors":[{"name":"model.visual.blocks.0.attn.qkv.bias","name":"model.visual.blocks.0.attn.qkv.bias","shape":[48]},{"name":"model.visual.blocks.0.attn.qkv.weight","shape":[48,16]}]}
        """
        expectRejected(Data(json.utf8))
    }

    @Test func rejectsInvalidDigestsAndArtifactKind() throws {
        expectRejected(try encoded(descriptor(textDigest: String(repeating: "A", count: 64))))
        expectRejected(try encoded(descriptor(processorDigest: String(repeating: "g", count: 64))))

        var object = try #require(JSONSerialization.jsonObject(
            with: encoded(descriptor())) as? [String: Any])
        object["artifactKind"] = "other-vision-kind"
        expectRejected(try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
    }

    @Test func rejectsUnsortedDuplicateOrMalformedTensorContracts() throws {
        let a = OfficialSourceVisionDescriptor.Tensor(name: "a", shape: [1])
        let b = OfficialSourceVisionDescriptor.Tensor(name: "b", shape: [1])
        expectRejected(try encoded(descriptor(tensors: [b, a])))
        expectRejected(try encoded(descriptor(tensors: [a, a])))
        expectRejected(try encoded(descriptor(tensors: [.init(name: "", shape: [1])])))
        expectRejected(try encoded(descriptor(tensors: [.init(name: "a", shape: [])])))
        expectRejected(try encoded(descriptor(tensors: [.init(name: "a", shape: [1, 0])])))
        expectRejected(try encoded(descriptor(tensors: [.init(
            name: String(repeating: "x", count: 161), shape: [1])])))
        expectRejected(try encoded(descriptor(tensors: [.init(
            name: "a", shape: [1, 1, 1, 1, 1, 1])])))
    }

    @Test func rejectsEmptyAndExcessiveTensorInventories() throws {
        expectRejected(try encoded(descriptor(tensors: [])))
        let tooMany = (0..<334).map { index in
            OfficialSourceVisionDescriptor.Tensor(
                name: String(format: "tensor.%03d", index), shape: [1])
        }
        expectRejected(try encoded(descriptor(tensors: tooMany)))
    }

    @Test func rejectsMalformedJSONAndMetadataBeyondByteLimit() throws {
        expectRejected(Data(#"{"artifactKind":}"#.utf8))
        expectRejected(Data(repeating: 0x20,
                            count: OfficialSourceVisionDescriptor.maximumBytes + 1))
    }
}
