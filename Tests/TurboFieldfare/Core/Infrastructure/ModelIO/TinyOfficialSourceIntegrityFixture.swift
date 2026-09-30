import CryptoKit
import Darwin
import Foundation
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Independent tiny source tree for policy/admission tests. The pinned checksum
/// manifest is duplicated from the independent Task 6.8 test fixture; every
/// shard written here contains only a few synthetic bytes.
struct TinyOfficialSourceIntegrityFixture {
    static let checksumManifest = """
    50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32  LICENSE
    c4ddaa065649ff6352648f64747a16eda31726f3e34add94ce04abb461c77b75  README.md
    e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259  chat_template.jinja
    93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99  config.json
    c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb  configuration.json
    e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e  generation_config.json
    a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d  merges.txt
    41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83  model.safetensors.index.json
    27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516  preprocessor_config.json
    5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42  tokenizer.json
    5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b  tokenizer_config.json
    ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003  vocab.json
    adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478  model-00001-of-00026.safetensors
    88f2dfd2b9e73e4b70be533dbf61bcfa3c9a0003758900fcbc9d9b96f5751d4b  model-00002-of-00026.safetensors
    8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb  model-00003-of-00026.safetensors
    12d7db38689ba3c8af74b23ef8523eca41e0cd95db870583d0663a3ee8a6bd60  model-00004-of-00026.safetensors
    a836047305d0f7a7b50f0815d09d5c03ec03d59ec2c763fcdc4bf7e9936bf902  model-00005-of-00026.safetensors
    c9080d718e9c5f9e337443225aa417d4c24d00ae7995d76ee3f1cc296b557d15  model-00006-of-00026.safetensors
    e8c05e23131b1dd45a455ec38cfac7db14667358268623c3938d00cf3e959a68  model-00007-of-00026.safetensors
    4b6a6d495053089f4a80e7cbc82e848fba44e2c0c60122233d8fdff79fa7b296  model-00008-of-00026.safetensors
    a31a954bb72d1c714e751bf0aabf2ff533f5a509693ebf7dd22ad6e90be46f67  model-00009-of-00026.safetensors
    246560e66570fe746653b8443e245dc334c9b8b831ea43d2d9f1b7d98623994e  model-00010-of-00026.safetensors
    7180392817fe3ecb3a27a1da43b7ff22c1a94806bac49975f9f122c3126df675  model-00011-of-00026.safetensors
    043fb525f6625c2f2acb75e65a9959ee3fa7b6e3fdd2034b5cfe1859b01d3cfb  model-00012-of-00026.safetensors
    33a20fb20a21379bf43c84a43105f9c0cc35bd50d740b1c302dcbe4b700f5425  model-00013-of-00026.safetensors
    be823e33c5cb6120ad3769d081f34a2449dc2358041fca7c29d636c1ba19130d  model-00014-of-00026.safetensors
    a89d547c6f9d0b535ee5ea2f2478f163089539f3f0dd330cb23d278a19d76123  model-00015-of-00026.safetensors
    69fc3ae0316482288afdcdd0b9eb7d626703ae26f7567e89aa3fc8d1ffd4ff5b  model-00016-of-00026.safetensors
    e356e3943cf3852b76bb8992e674f3256013e27d54b78e8250514151cdc29637  model-00017-of-00026.safetensors
    9e5e63fd1cc7d6848330c1fa363dfcb661bbc2ac87e672d0e28b71c9cb7f3c7f  model-00018-of-00026.safetensors
    708644ad34f1de727bf484f396944d8ec628645d52c183e9a992e65671685e21  model-00019-of-00026.safetensors
    ca083a1d1aa64f8e8a785998f543a43374f13436dc85d396eee4e72c7a84e1ae  model-00020-of-00026.safetensors
    ada4ae48f3d48fe01b4c53f2f82bce25e798a9631fd33959c881156fef2ccbce  model-00021-of-00026.safetensors
    def207fb42d7db31efb512755557763c23233c6e4d4c433027cb5102a7bce2f7  model-00022-of-00026.safetensors
    864d52ca7768a36f514069222e8de8626264ae124097ba8fcce5b5da2c6e2ed7  model-00023-of-00026.safetensors
    391acd27420cdce5935ff18152423c70620d19dac3c39a5ef1a81d369f82d737  model-00024-of-00026.safetensors
    778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154  model-00025-of-00026.safetensors
    1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613  model-00026-of-00026.safetensors
    """

    static let checksumManifestSHA256 =
        "1378d1bb153694b13c20641fa5d2485dede401d1ce4c267953ba4a855fc59e7e"

    let root: URL
    let sourceRoot: URL
    let modelDirectory: URL
    let shardBytes: UInt64
    let checksumManifestURL: URL
    let firstShardURL: URL

    static func make(includeReceipt: Bool) throws -> Self {
        let fileManager = FileManager.default
        let temporaryRoot = fileManager.temporaryDirectory.appendingPathComponent(
            "tiny-official-source-policy-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: temporaryRoot, withIntermediateDirectories: false)
        var canonicalPath = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(temporaryRoot.path, &canonicalPath) != nil else {
            try? fileManager.removeItem(at: temporaryRoot)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let root = URL(fileURLWithPath: String(cString: canonicalPath), isDirectory: true)
        var transfersRootOwnership = false
        defer {
            if !transfersRootOwnership { try? fileManager.removeItem(at: root) }
        }
        let sourceRoot = root.appendingPathComponent("synthetic-source", isDirectory: true)
        let models = root.appendingPathComponent("models", isDirectory: true)
        let modelDirectory = models.appendingPathComponent("source.gturbo", isDirectory: true)
        try fileManager.createDirectory(at: sourceRoot, withIntermediateDirectories: false)
        try fileManager.createDirectory(at: models, withIntermediateDirectories: false)

        let manifestData = Data((checksumManifest + "\n").utf8)
        guard manifestData.count == 3_571,
              SHA256.hash(data: manifestData).map({ String(format: "%02x", $0) }).joined()
                == checksumManifestSHA256 else {
            try? fileManager.removeItem(at: root)
            throw FixtureError.invalidChecksumManifest
        }
        let manifestURL = sourceRoot.appendingPathComponent("SHA256SUMS")
        try manifestData.write(to: manifestURL)

        var shardBytes: UInt64 = 0
        let firstShard = "model-00001-of-00026.safetensors"
        for line in checksumManifest.split(separator: "\n") {
            let filename = String(line.dropFirst(66))
            let data: Data
            if filename == firstShard {
                data = Data("abc".utf8)
            } else if filename.hasSuffix(".safetensors") {
                data = Data("tiny synthetic payload for \(filename)".utf8)
                shardBytes += UInt64(data.count)
            } else {
                data = Data("synthetic sidecar \(filename)".utf8)
            }
            if filename == firstShard { shardBytes += UInt64(data.count) }
            try data.write(to: sourceRoot.appendingPathComponent(filename))
        }
        let marker = try IndependentOfficialSourceFixture.markerData(sourceRoot: sourceRoot.path)
        _ = try OfficialSourceRegistration.register(markerData: marker, at: modelDirectory)

        if includeReceipt {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: shardBytes,
                fullVerification: {})
        }
        transfersRootOwnership = true
        return Self(
            root: root,
            sourceRoot: sourceRoot,
            modelDirectory: modelDirectory,
            shardBytes: shardBytes,
            checksumManifestURL: manifestURL,
            firstShardURL: sourceRoot.appendingPathComponent(firstShard))
    }

    var receiptURL: URL {
        modelDirectory.appendingPathComponent("official-source-receipt.json")
    }

    func verifyTrustedReopen() throws -> OfficialSourceTrustReceipt {
        try OfficialSourceTrust.verifySynthetic(
            at: modelDirectory,
            policy: .sizeCheckTrustedReceipt,
            expectedShardBytes: shardBytes,
            fullVerification: {})
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    enum FixtureError: Error {
        case invalidChecksumManifest
    }
}
