import Testing
import TurboFieldfareOfficialQwenSource

/// Metadata-only tests for the publisher-pinned original BF16 identity.
/// Expected digests below are transcribed independently from the prepared
/// snapshot's SHA256SUMS; these tests never open official weight files.
@Suite struct OfficialBF16IdentityTests {
    @Test func exactPinnedIdentityIsAcceptedWithoutReadingWeights() throws {
        #expect(OfficialBF16Fixture.shardMetadata.count == 26)
        #expect(Set(OfficialBF16Fixture.shardMetadata.map(\.filename)).count == 26)
        #expect(OfficialBF16Fixture.shardMetadata.allSatisfy { entry in
            entry.sha256.count == 64
                && entry.sha256.allSatisfy { $0.isNumber || ("a"..."f").contains($0) }
        })

        try OfficialQwenIdentity.validate(OfficialBF16Fixture.identity())
    }

    @Test func repositoryMutationIsRejected() {
        #expect(rejection(for: OfficialBF16Fixture.identity(
            repository: "Qwen/Qwen3.6-35B-A3B-fork")) == .invalidRepository)
    }

    @Test func revisionMutationIsRejected() {
        #expect(rejection(for: OfficialBF16Fixture.identity(
            revision: "995ad96eacd98c81ed38be0c5b274b04031597b1")) == .invalidRevision)
    }

    @Test(arguments: [
        "mlx-affine-4bit-group64",
        "int4-experts-int8-router",
        "qwen3.6-35b-a3b-quantized",
    ])
    func quantizedStorageProfilesAreRejected(profile: String) {
        #expect(rejection(for: OfficialBF16Fixture.identity(storageProfile: profile))
            == .invalidStorageProfile)
    }

    @Test(arguments: [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ])
    func oneByteChangeToEachPinnedSidecarDigestIsRejected(filename: String) throws {
        var digests = OfficialBF16Fixture.sidecarSHA256
        let original = try #require(digests[filename])
        digests[filename] = String(original.dropLast()) + (original.hasSuffix("0") ? "1" : "0")

        #expect(digests[filename] != original)
        #expect(rejection(for: OfficialBF16Fixture.identity(sidecarSHA256: digests))
            == .invalidSidecarDigests)
    }

    @Test func missingAndAdditionalSidecarIdentitiesAreRejected() {
        var missing = OfficialBF16Fixture.sidecarSHA256
        missing.removeValue(forKey: "tokenizer.json")
        #expect(rejection(for: OfficialBF16Fixture.identity(sidecarSHA256: missing))
            == .invalidSidecarDigests)

        var additional = OfficialBF16Fixture.sidecarSHA256
        additional["chat_template.jinja"] = String(repeating: "a", count: 64)
        #expect(rejection(for: OfficialBF16Fixture.identity(sidecarSHA256: additional))
            == .invalidSidecarDigests)
    }

    @Test func everyPinnedShardNameIsBoundIndependently() {
        for index in OfficialBF16Fixture.shards.indices {
            var shards = OfficialBF16Fixture.shards
            let original = shards[index]
            shards[index] = OfficialQwenShard(
                filename: "altered-\(original.filename)", sha256: original.sha256)
            #expect(rejection(for: OfficialBF16Fixture.identity(shards: shards)) == .invalidShards,
                    "accepted changed shard name at index \(index)")
        }
    }

    @Test func shardCountMutationIsRejected() {
        var shards = OfficialBF16Fixture.shards
        shards.removeLast()
        #expect(shards.count == 25)
        #expect(rejection(for: OfficialBF16Fixture.identity(shards: shards)) == .invalidShards)
    }

    @Test func everyPinnedShardHashIsBoundIndependently() {
        for index in OfficialBF16Fixture.shards.indices {
            var shards = OfficialBF16Fixture.shards
            let original = shards[index]
            let changedHash = String(original.sha256.dropLast())
                + (original.sha256.hasSuffix("0") ? "1" : "0")
            shards[index] = OfficialQwenShard(
                filename: original.filename, sha256: changedHash)
            #expect(rejection(for: OfficialBF16Fixture.identity(shards: shards)) == .invalidShards,
                    "accepted changed shard hash at index \(index)")
        }
    }

    @Test func missingShardEntryIsRejected() {
        var shards = OfficialBF16Fixture.shards
        shards.remove(at: 12)
        #expect(rejection(for: OfficialBF16Fixture.identity(shards: shards)) == .invalidShards)
    }

    @Test func duplicateShardEntryIsRejected() {
        var shards = OfficialBF16Fixture.shards
        shards[1] = shards[0]
        #expect(shards.count == 26)
        #expect(rejection(for: OfficialBF16Fixture.identity(shards: shards)) == .invalidShards)
    }

    @Test func additionalShardEntryIsRejected() {
        var shards = OfficialBF16Fixture.shards
        shards.append(OfficialQwenShard(
            filename: "model-00027-of-00026.safetensors",
            sha256: String(repeating: "f", count: 64)))
        #expect(rejection(for: OfficialBF16Fixture.identity(shards: shards)) == .invalidShards)
    }

    private func rejection(
        for identity: OfficialQwenSourceIdentity
    ) -> OfficialQwenIdentityError? {
        do {
            try OfficialQwenIdentity.validate(identity)
            return nil
        } catch let error as OfficialQwenIdentityError {
            return error
        } catch {
            Issue.record("validate threw an unexpected error: \(error)")
            return nil
        }
    }
}

private enum OfficialBF16Fixture {
    static let repository = "Qwen/Qwen3.6-35B-A3B"
    static let revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    static let storageProfile = "original-bf16"

    /// The seven behavior-defining sidecars pinned by the existing Qwen identity contract.
    static let sidecarSHA256: [String: String] = [
        "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "model.safetensors.index.json": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    ]

    /// Ordered name/hash pairs copied from the pinned SHA256SUMS metadata.
    static let shardMetadata: [(filename: String, sha256: String)] = [
        ("model-00001-of-00026.safetensors", "adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478"),
        ("model-00002-of-00026.safetensors", "88f2dfd2b9e73e4b70be533dbf61bcfa3c9a0003758900fcbc9d9b96f5751d4b"),
        ("model-00003-of-00026.safetensors", "8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb"),
        ("model-00004-of-00026.safetensors", "12d7db38689ba3c8af74b23ef8523eca41e0cd95db870583d0663a3ee8a6bd60"),
        ("model-00005-of-00026.safetensors", "a836047305d0f7a7b50f0815d09d5c03ec03d59ec2c763fcdc4bf7e9936bf902"),
        ("model-00006-of-00026.safetensors", "c9080d718e9c5f9e337443225aa417d4c24d00ae7995d76ee3f1cc296b557d15"),
        ("model-00007-of-00026.safetensors", "e8c05e23131b1dd45a455ec38cfac7db14667358268623c3938d00cf3e959a68"),
        ("model-00008-of-00026.safetensors", "4b6a6d495053089f4a80e7cbc82e848fba44e2c0c60122233d8fdff79fa7b296"),
        ("model-00009-of-00026.safetensors", "a31a954bb72d1c714e751bf0aabf2ff533f5a509693ebf7dd22ad6e90be46f67"),
        ("model-00010-of-00026.safetensors", "246560e66570fe746653b8443e245dc334c9b8b831ea43d2d9f1b7d98623994e"),
        ("model-00011-of-00026.safetensors", "7180392817fe3ecb3a27a1da43b7ff22c1a94806bac49975f9f122c3126df675"),
        ("model-00012-of-00026.safetensors", "043fb525f6625c2f2acb75e65a9959ee3fa7b6e3fdd2034b5cfe1859b01d3cfb"),
        ("model-00013-of-00026.safetensors", "33a20fb20a21379bf43c84a43105f9c0cc35bd50d740b1c302dcbe4b700f5425"),
        ("model-00014-of-00026.safetensors", "be823e33c5cb6120ad3769d081f34a2449dc2358041fca7c29d636c1ba19130d"),
        ("model-00015-of-00026.safetensors", "a89d547c6f9d0b535ee5ea2f2478f163089539f3f0dd330cb23d278a19d76123"),
        ("model-00016-of-00026.safetensors", "69fc3ae0316482288afdcdd0b9eb7d626703ae26f7567e89aa3fc8d1ffd4ff5b"),
        ("model-00017-of-00026.safetensors", "e356e3943cf3852b76bb8992e674f3256013e27d54b78e8250514151cdc29637"),
        ("model-00018-of-00026.safetensors", "9e5e63fd1cc7d6848330c1fa363dfcb661bbc2ac87e672d0e28b71c9cb7f3c7f"),
        ("model-00019-of-00026.safetensors", "708644ad34f1de727bf484f396944d8ec628645d52c183e9a992e65671685e21"),
        ("model-00020-of-00026.safetensors", "ca083a1d1aa64f8e8a785998f543a43374f13436dc85d396eee4e72c7a84e1ae"),
        ("model-00021-of-00026.safetensors", "ada4ae48f3d48fe01b4c53f2f82bce25e798a9631fd33959c881156fef2ccbce"),
        ("model-00022-of-00026.safetensors", "def207fb42d7db31efb512755557763c23233c6e4d4c433027cb5102a7bce2f7"),
        ("model-00023-of-00026.safetensors", "864d52ca7768a36f514069222e8de8626264ae124097ba8fcce5b5da2c6e2ed7"),
        ("model-00024-of-00026.safetensors", "391acd27420cdce5935ff18152423c70620d19dac3c39a5ef1a81d369f82d737"),
        ("model-00025-of-00026.safetensors", "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154"),
        ("model-00026-of-00026.safetensors", "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613"),
    ]

    static var shards: [OfficialQwenShard] {
        shardMetadata.map { entry in
            OfficialQwenShard(filename: entry.filename, sha256: entry.sha256)
        }
    }

    static func identity(
        repository: String? = nil,
        revision: String? = nil,
        storageProfile: String? = nil,
        sidecarSHA256: [String: String]? = nil,
        shards: [OfficialQwenShard]? = nil
    ) -> OfficialQwenSourceIdentity {
        OfficialQwenSourceIdentity(
            repository: repository ?? Self.repository,
            revision: revision ?? Self.revision,
            storageProfile: storageProfile ?? Self.storageProfile,
            sidecarSHA256: sidecarSHA256 ?? Self.sidecarSHA256,
            shards: shards ?? Self.shards)
    }
}
