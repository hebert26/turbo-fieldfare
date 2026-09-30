import Foundation
import Testing
@testable import TurboFieldfareFormat

@Suite struct OfficialSourceDescriptorTests {
    private static let expectedFixtureDigest =
        "c6c3f178fa71b1807853746aa679ed0726ff186141dd681d0738451e5b965394"

    @Test func canonicalDigestUsesSortedUTF8LinesAndOmitsSourceRoot() throws {
        let first = try Self.fixtureDescriptor(sourceRoot: "/fixture/source-one")
        let reordered = try OfficialSourceDescriptor(
            repository: "fixture/Qwen",
            revision: "fixture-revision",
            storageProfile: "original-bf16",
            sidecarSHA256: [
                "config.json": String(repeating: "b", count: 64),
                "tokenizer.json": String(repeating: "a", count: 64),
            ],
            shards: [
                .init(filename: "model-00001.safetensors", sha256: String(repeating: "1", count: 64)),
                .init(filename: "model-00002.safetensors", sha256: String(repeating: "2", count: 64)),
            ],
            sourceRoot: "/fixture/source-two")

        #expect(first.kind == "official-safetensors-bf16-v1")
        #expect(first.version == 1)
        #expect(first.contentSHA256 == Self.expectedFixtureDigest)
        #expect(reordered.contentSHA256 == Self.expectedFixtureDigest)
        #expect(first.sourceRoot != reordered.sourceRoot)

        let encoded = try JSONEncoder().encode(first)
        let decoded = try OfficialSourceDescriptor.decodeStrict(data: encoded)
        #expect(decoded == first)
        #expect(decoded.contentSHA256 == Self.expectedFixtureDigest)
    }

    @Test func independentlyPinnedV2DecoderCannotAcceptSourceDescriptor() throws {
        let encoded = try JSONEncoder().encode(Self.fixtureDescriptor())
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestV2Codec.decode(encoded)
        }
    }

    @Test(arguments: ["versionMajor", "versionMinor", "formatMajor", "magic", "files", "tensorRegions"])
    func decoderRejectsPackedOrUnknownFields(_ key: String) throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(Self.fixtureDescriptor())) as? [String: Any])
        root[key] = [:]
        let forged = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: OfficialSourceDescriptorError.self) {
            try OfficialSourceDescriptor.decodeStrict(data: forged)
        }
    }

    @Test func decoderRejectsDuplicateSidecarKeysInsteadOfCollapsingTheObject() throws {
        let raw = """
        {"kind":"official-safetensors-bf16-v1","version":1,
        "repository":"fixture/Qwen","revision":"fixture-revision",
        "storageProfile":"original-bf16",
        "sidecarSHA256":{"config.json":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "config.json":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "tokenizer.json":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
        "shards":[
        {"filename":"model-00001.safetensors","sha256":"1111111111111111111111111111111111111111111111111111111111111111"},
        {"filename":"model-00002.safetensors","sha256":"2222222222222222222222222222222222222222222222222222222222222222"}],
        "sourceRoot":"/fixture/source",
        "contentSHA256":"c6c3f178fa71b1807853746aa679ed0726ff186141dd681d0738451e5b965394"}
        """
        #expect(throws: OfficialSourceDescriptorError.self) {
            try OfficialSourceDescriptor.decodeStrict(data: Data(raw.utf8))
        }
    }

    @Test func strictDecoderRejectsNestingBeyondSixtyFourWithStructuralError() throws {
        var raw = String(decoding: try JSONEncoder().encode(Self.fixtureDescriptor()), as: UTF8.self)
        let closingBrace = try #require(raw.lastIndex(of: "}"))
        let depth = 65
        let tooDeep = String(repeating: "[", count: depth) + "0"
            + String(repeating: "]", count: depth)
        raw.insert(contentsOf: ",\"extra\":\(tooDeep)", at: closingBrace)
        expectStructuralJSONError(Data(raw.utf8))
    }

    @Test func strictDecoderRejectsMalformedJSONWithStructuralError() {
        let malformed = [
            #"{"kind":"fixture"} trailing"#,
            #"{"kind":"fixture",}"#,
            #"{"kind":"fixture""#,
            #"{"kind":"bad\q"}"#,
            #"{"version":01}"#,
        ]
        for json in malformed {
            expectStructuralJSONError(Data(json.utf8))
        }
    }

    @Test func decoderRejectsMismatchedContentDigest() throws {
        var root = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(Self.fixtureDescriptor())) as? [String: Any])
        root["contentSHA256"] = String(repeating: "0", count: 64)
        let forged = try JSONSerialization.data(withJSONObject: root)
        #expect(throws: OfficialSourceDescriptorError.self) {
            try OfficialSourceDescriptor.decodeStrict(data: forged)
        }
    }

    @Test func constructorRejectsWrongKindAndVersion() throws {
        #expect(throws: OfficialSourceDescriptorError.self) {
            try OfficialSourceDescriptor(
                kind: "gturbo-v2", version: 2,
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16", sidecarSHA256: Self.sidecars,
                shards: Self.shards, sourceRoot: "/fixture/source",
                contentSHA256: Self.expectedFixtureDigest)
        }
        #expect(throws: OfficialSourceDescriptorError.self) {
            try OfficialSourceDescriptor(
                kind: "official-safetensors-bf16-v1", version: 2,
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16", sidecarSHA256: Self.sidecars,
                shards: Self.shards, sourceRoot: "/fixture/source",
                contentSHA256: Self.expectedFixtureDigest)
        }
    }

    @Test func rejectsUnsafeScalarIdentityFieldsBeforeDigesting() throws {
        for invalid in ["fixture=value", "fixture\nvalue", "fixture\rvalue"] {
            for field in ["repository", "revision", "storageProfile"] {
                #expect(throws: OfficialSourceDescriptorError.self) {
                    _ = try OfficialSourceDescriptor.digest(
                        repository: field == "repository" ? invalid : "fixture/Qwen",
                        revision: field == "revision" ? invalid : "fixture-revision",
                        storageProfile: field == "storageProfile" ? invalid : "original-bf16",
                        sidecarSHA256: Self.sidecars, shards: Self.shards)
                }
            }
        }
    }

    @Test(arguments: ["", ".", "..", "../escape.json", "nested/file.json", "bad=name.json", "bad\nname.json", "bad\rname.json", "bad\0name.json"])
    func rejectsUnsafeSidecarAndShardNames(_ name: String) {
        #expect(throws: OfficialSourceDescriptorError.self) {
            _ = try OfficialSourceDescriptor.digest(
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16",
                sidecarSHA256: [name: String(repeating: "a", count: 64)],
                shards: Self.shards)
        }
        #expect(throws: OfficialSourceDescriptorError.self) {
            _ = try OfficialSourceDescriptor.digest(
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16", sidecarSHA256: Self.sidecars,
                shards: [.init(filename: name, sha256: String(repeating: "a", count: 64))])
        }
    }

    @Test func rejectsDuplicateShardNamesBeforeCanonicalSorting() {
        let duplicate = [
            OfficialSourceDescriptor.Shard(filename: "model-00002.safetensors", sha256: String(repeating: "2", count: 64)),
            OfficialSourceDescriptor.Shard(filename: "model-00001.safetensors", sha256: String(repeating: "1", count: 64)),
            OfficialSourceDescriptor.Shard(filename: "model-00001.safetensors", sha256: String(repeating: "3", count: 64)),
        ]
        #expect(throws: OfficialSourceDescriptorError.self) {
            _ = try OfficialSourceDescriptor.digest(
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16", sidecarSHA256: Self.sidecars,
                shards: duplicate)
        }
    }

    @Test func rejectsAppleFilesystemEquivalentSidecarNames() {
        #expect(throws: OfficialSourceDescriptorError.self) {
            _ = try OfficialSourceDescriptor.digest(
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16",
                sidecarSHA256: [
                    "config.json": String(repeating: "a", count: 64),
                    "Config.json": String(repeating: "b", count: 64),
                ],
                shards: Self.shards)
        }
    }

    @Test func rejectsSidecarShardNameCollision() {
        #expect(throws: OfficialSourceDescriptorError.self) {
            _ = try OfficialSourceDescriptor.digest(
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16",
                sidecarSHA256: ["model.safetensors": String(repeating: "a", count: 64)],
                shards: [.init(filename: "model.safetensors", sha256: String(repeating: "b", count: 64))])
        }
    }

    @Test func rejectsNonLowercaseOrMalformedSHA256Values() {
        let invalidDigests = [
            String(repeating: "A", count: 64),
            String(repeating: "a", count: 63),
            String(repeating: "g", count: 64),
        ]
        for digest in invalidDigests {
            #expect(throws: OfficialSourceDescriptorError.self) {
                _ = try OfficialSourceDescriptor.digest(
                    repository: "fixture/Qwen", revision: "fixture-revision",
                    storageProfile: "original-bf16", sidecarSHA256: ["config.json": digest],
                    shards: Self.shards)
            }
            #expect(throws: OfficialSourceDescriptorError.self) {
                _ = try OfficialSourceDescriptor.digest(
                    repository: "fixture/Qwen", revision: "fixture-revision",
                    storageProfile: "original-bf16", sidecarSHA256: Self.sidecars,
                    shards: [.init(filename: "model.safetensors", sha256: digest)])
            }
        }
    }

    @Test(arguments: ["relative/source", "/tmp//source", "/tmp/./source", "/tmp/../source", "/tmp/source/", "/tmp/source\0tail"])
    func rejectsMalformedSourceRootSyntax(_ path: String) throws {
        #expect(throws: OfficialSourceDescriptorError.self) {
            try OfficialSourceDescriptor(
                repository: "fixture/Qwen", revision: "fixture-revision",
                storageProfile: "original-bf16", sidecarSHA256: Self.sidecars,
                shards: Self.shards, sourceRoot: path)
        }
    }

    @Test func nonexistentAbsoluteSourceRootIsValidMetadata() throws {
        let path = "/tmp/turbo-source-descriptor-\(UUID().uuidString)/not-created"
        #expect(!FileManager.default.fileExists(atPath: path))
        let descriptor = try Self.fixtureDescriptor(sourceRoot: path)
        #expect(descriptor.sourceRoot == path)
        #expect(descriptor.contentSHA256 == Self.expectedFixtureDigest)
    }

    private static let sidecars = [
        "tokenizer.json": String(repeating: "a", count: 64),
        "config.json": String(repeating: "b", count: 64),
    ]

    private static let shards = [
        OfficialSourceDescriptor.Shard(filename: "model-00002.safetensors", sha256: String(repeating: "2", count: 64)),
        OfficialSourceDescriptor.Shard(filename: "model-00001.safetensors", sha256: String(repeating: "1", count: 64)),
    ]

    private static func fixtureDescriptor(sourceRoot: String = "/fixture/source") throws
        -> OfficialSourceDescriptor {
        try OfficialSourceDescriptor(
            repository: "fixture/Qwen", revision: "fixture-revision",
            storageProfile: "original-bf16", sidecarSHA256: sidecars,
            shards: shards, sourceRoot: sourceRoot)
    }

    private func expectStructuralJSONError(_ data: Data) {
        #expect {
            _ = try OfficialSourceDescriptor.decodeStrict(data: data)
        } throws: { error in
            guard let error = error as? OfficialSourceDescriptorError else { return false }
            return error == .invalid(
                field: "descriptor.json", reason: "invalid JSON structure or nesting")
        }
    }
}
