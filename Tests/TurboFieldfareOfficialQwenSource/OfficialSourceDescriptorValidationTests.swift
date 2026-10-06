import Foundation
import Testing
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

/// The literal identity fixture below is transcribed from the pinned
/// `scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0/SHA256SUMS`,
/// not from OfficialQwenSourceIdentity.pinned. Its expected digest
/// was independently calculated with Python 3 hashlib.sha256 by sorting the
/// seven sidecars and 26 shards within category, concatenating the exact
/// UTF-8/LF-terminated canonical records, and hashing those bytes. The expected
/// value does not come from the production descriptor digest API.
@Suite struct OfficialSourceDescriptorValidationTests {
    private static let independentlyCalculatedPinnedDigest =
        "c1ac463726b716e7db3a9fbe5db4cb690f95be7e5779aea02e49c2020bca7a7a"

    @Test func literalPinnedDescriptorMatchesIndependentDigestAndSharedBridge() throws {
        let descriptor = try makeDescriptor()
        #expect(descriptor.contentSHA256 == Self.independentlyCalculatedPinnedDigest)
        #expect(descriptor.kind == "official-safetensors-bf16-v1")
        #expect(descriptor.version == 1)
        try OfficialSourceDescriptorValidation.validate(descriptor)
    }

    @Test func serializationAndShardOrderDoNotChangeLiteralPinnedIdentity() throws {
        let original = try makeDescriptor(sourceRoot: "/tmp/pinned-source-a")
        let reordered = try makeDescriptor(
            sourceRoot: "/tmp/pinned-source-b",
            shards: Array(IndependentOfficialQwenMetadata.shards.reversed()))

        #expect(original.contentSHA256 == Self.independentlyCalculatedPinnedDigest)
        #expect(reordered.contentSHA256 == Self.independentlyCalculatedPinnedDigest)
        #expect(original.contentSHA256 == reordered.contentSHA256)
        try OfficialSourceDescriptorValidation.validate(reordered)

        let encoded = try JSONEncoder().encode(reordered)
        let decoded = try OfficialSourceDescriptor.decodeStrict(data: encoded)
        #expect(decoded.contentSHA256 == original.contentSHA256)
        try OfficialSourceDescriptorValidation.validate(decoded)
    }

    @Test func localSourceRootIsNotPartOfLiteralPinnedContentIdentity() throws {
        let first = try makeDescriptor(sourceRoot: "/tmp/source-location-one")
        let second = try makeDescriptor(sourceRoot: "/nonexistent/source-location-two")
        #expect(first.sourceRoot != second.sourceRoot)
        #expect(first.contentSHA256 == second.contentSHA256)
        try OfficialSourceDescriptorValidation.validate(second)
    }

    @Test func rejectsRepositoryRevisionAndStorageProfileOutsideSharedPin() throws {
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(
                repository: IndependentOfficialQwenMetadata.repository + "-fork"))
        }
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(
                revision: String(repeating: "0", count: 40)))
        }
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(
                storageProfile: "quantized-int4"))
        }
    }

    @Test func rejectsChangedOrIncompleteLiteralSidecarInventory() throws {
        var changed = IndependentOfficialQwenMetadata.sidecars
        let original = try #require(changed["config.json"])
        changed["config.json"] = Self.changedDigest(original)
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(sidecars: changed))
        }

        var missing = IndependentOfficialQwenMetadata.sidecars
        missing.removeValue(forKey: "tokenizer.json")
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(sidecars: missing))
        }

        var additional = IndependentOfficialQwenMetadata.sidecars
        additional["extra.json"] = String(repeating: "0", count: 64)
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(sidecars: additional))
        }
    }

    @Test func rejectsChangedMissingOrAdditionalLiteralShardMetadata() throws {
        var changedHash = IndependentOfficialQwenMetadata.shards
        let first = changedHash[0]
        changedHash[0] = OfficialQwenShard(
            filename: first.filename, sha256: Self.changedDigest(first.sha256))
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(shards: changedHash))
        }

        var changedName = IndependentOfficialQwenMetadata.shards
        let second = changedName[1]
        changedName[1] = OfficialQwenShard(
            filename: "renamed-\(second.filename)", sha256: second.sha256)
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(shards: changedName))
        }

        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(
                shards: Array(IndependentOfficialQwenMetadata.shards.dropLast())))
        }

        var additional = IndependentOfficialQwenMetadata.shards
        additional.append(OfficialQwenShard(
            filename: "model-00027-of-00026.safetensors",
            sha256: String(repeating: "0", count: 64)))
        #expect(throws: OfficialQwenIdentityError.self) {
            try OfficialSourceDescriptorValidation.validate(try makeDescriptor(shards: additional))
        }
    }

    private func makeDescriptor(
        sourceRoot: String = "/tmp/qwen-source-never-opened",
        repository: String? = nil,
        revision: String? = nil,
        storageProfile: String? = nil,
        sidecars: [String: String]? = nil,
        shards: [OfficialQwenShard]? = nil
    ) throws -> OfficialSourceDescriptor {
        let shardList = shards ?? IndependentOfficialQwenMetadata.shards
        return try OfficialSourceDescriptor(
            repository: repository ?? IndependentOfficialQwenMetadata.repository,
            revision: revision ?? IndependentOfficialQwenMetadata.revision,
            storageProfile: storageProfile ?? IndependentOfficialQwenMetadata.storageProfile,
            sidecarSHA256: sidecars ?? IndependentOfficialQwenMetadata.sidecars,
            shards: shardList.map {
                OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
            },
            sourceRoot: sourceRoot)
    }

    private static func changedDigest(_ digest: String) -> String {
        String(digest.dropLast()) + (digest.hasSuffix("0") ? "1" : "0")
    }
}

private enum IndependentOfficialQwenMetadata {
    static let repository = "Qwen/Qwen3.6-35B-A3B"
    static let revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    static let storageProfile = "original-bf16"

    static let sidecars: [String: String] = [
        "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "model.safetensors.index.json": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    ]

    static let shards: [OfficialQwenShard] = [
        .init(filename: "model-00001-of-00026.safetensors", sha256: "adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478"),
        .init(filename: "model-00002-of-00026.safetensors", sha256: "88f2dfd2b9e73e4b70be533dbf61bcfa3c9a0003758900fcbc9d9b96f5751d4b"),
        .init(filename: "model-00003-of-00026.safetensors", sha256: "8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb"),
        .init(filename: "model-00004-of-00026.safetensors", sha256: "12d7db38689ba3c8af74b23ef8523eca41e0cd95db870583d0663a3ee8a6bd60"),
        .init(filename: "model-00005-of-00026.safetensors", sha256: "a836047305d0f7a7b50f0815d09d5c03ec03d59ec2c763fcdc4bf7e9936bf902"),
        .init(filename: "model-00006-of-00026.safetensors", sha256: "c9080d718e9c5f9e337443225aa417d4c24d00ae7995d76ee3f1cc296b557d15"),
        .init(filename: "model-00007-of-00026.safetensors", sha256: "e8c05e23131b1dd45a455ec38cfac7db14667358268623c3938d00cf3e959a68"),
        .init(filename: "model-00008-of-00026.safetensors", sha256: "4b6a6d495053089f4a80e7cbc82e848fba44e2c0c60122233d8fdff79fa7b296"),
        .init(filename: "model-00009-of-00026.safetensors", sha256: "a31a954bb72d1c714e751bf0aabf2ff533f5a509693ebf7dd22ad6e90be46f67"),
        .init(filename: "model-00010-of-00026.safetensors", sha256: "246560e66570fe746653b8443e245dc334c9b8b831ea43d2d9f1b7d98623994e"),
        .init(filename: "model-00011-of-00026.safetensors", sha256: "7180392817fe3ecb3a27a1da43b7ff22c1a94806bac49975f9f122c3126df675"),
        .init(filename: "model-00012-of-00026.safetensors", sha256: "043fb525f6625c2f2acb75e65a9959ee3fa7b6e3fdd2034b5cfe1859b01d3cfb"),
        .init(filename: "model-00013-of-00026.safetensors", sha256: "33a20fb20a21379bf43c84a43105f9c0cc35bd50d740b1c302dcbe4b700f5425"),
        .init(filename: "model-00014-of-00026.safetensors", sha256: "be823e33c5cb6120ad3769d081f34a2449dc2358041fca7c29d636c1ba19130d"),
        .init(filename: "model-00015-of-00026.safetensors", sha256: "a89d547c6f9d0b535ee5ea2f2478f163089539f3f0dd330cb23d278a19d76123"),
        .init(filename: "model-00016-of-00026.safetensors", sha256: "69fc3ae0316482288afdcdd0b9eb7d626703ae26f7567e89aa3fc8d1ffd4ff5b"),
        .init(filename: "model-00017-of-00026.safetensors", sha256: "e356e3943cf3852b76bb8992e674f3256013e27d54b78e8250514151cdc29637"),
        .init(filename: "model-00018-of-00026.safetensors", sha256: "9e5e63fd1cc7d6848330c1fa363dfcb661bbc2ac87e672d0e28b71c9cb7f3c7f"),
        .init(filename: "model-00019-of-00026.safetensors", sha256: "708644ad34f1de727bf484f396944d8ec628645d52c183e9a992e65671685e21"),
        .init(filename: "model-00020-of-00026.safetensors", sha256: "ca083a1d1aa64f8e8a785998f543a43374f13436dc85d396eee4e72c7a84e1ae"),
        .init(filename: "model-00021-of-00026.safetensors", sha256: "ada4ae48f3d48fe01b4c53f2f82bce25e798a9631fd33959c881156fef2ccbce"),
        .init(filename: "model-00022-of-00026.safetensors", sha256: "def207fb42d7db31efb512755557763c23233c6e4d4c433027cb5102a7bce2f7"),
        .init(filename: "model-00023-of-00026.safetensors", sha256: "864d52ca7768a36f514069222e8de8626264ae124097ba8fcce5b5da2c6e2ed7"),
        .init(filename: "model-00024-of-00026.safetensors", sha256: "391acd27420cdce5935ff18152423c70620d19dac3c39a5ef1a81d369f82d737"),
        .init(filename: "model-00025-of-00026.safetensors", sha256: "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154"),
        .init(filename: "model-00026-of-00026.safetensors", sha256: "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613"),
    ]
}
