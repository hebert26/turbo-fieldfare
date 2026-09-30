import Darwin
import Foundation
import Testing
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfareRepackCore

/// The checksum document is metadata only. The shard callbacks below return
/// the pinned digest and synthetic byte counts without opening an official
/// shard payload, so these tests remain tiny and offline.
@Suite(.serialized)
struct LocalOfficialQwenPayloadVerifierTests {
    @Test func canonicalChecksumInventoryAcceptsSyntheticShardCallbacks() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        let identity = try LocalOfficialQwenPayloadVerifier.verify(
            snapshotDirectory: fixture.root.path,
            shardFilenames: fixture.shards,
            operations: fixture.operations())

        #expect(identity.checksumManifestSHA256 ==
                LocalOfficialQwenPayloadVerifier.checksumManifestSHA256)
        #expect(identity.shardBytes == LocalOfficialQwenPayloadVerifier.expectedShardBytes)
        #expect(identity.shardCount == LocalOfficialQwenPayloadVerifier.expectedShardCount)
        // Independently calculated with Python hashlib from the literal pinned
        // names/digests and 26 synthetic sizes of 2,765,529,876 bytes.
        #expect(identity.shardSetSHA256 ==
                "0563ff248ca173aed6fa2c8a730dbebec9d2d5eedbc87cdbd0932c0da76af4d1")
    }

    @Test func checksumDocumentMutationFailsBeforeAnyShardCallback() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        var inspected = false
        var operations = fixture.operations()
        operations.readChecksumManifest = { _, _ in
            var data = fixture.checksumData
            data[0] ^= 0x01
            return data
        }
        operations.inspectAndHashFile = { _, _ in
            inspected = true
            return (0, String(repeating: "0", count: 64))
        }

        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: operations)
        }
        #expect(!inspected)
    }

    @Test func malformedChecksumDocumentIsRejectedBeforeAnyShardCallback() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        var inspected = false
        var operations = fixture.operations()
        operations.readChecksumManifest = { _, _ in Data("malformed checksum metadata".utf8) }
        operations.inspectAndHashFile = { path, _ in
            inspected = true
            return (fixture.size(for: path), fixture.expectedDigest(for: path))
        }

        do {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: operations)
            Issue.record("expected malformed checksum bytes to fail the pinned fingerprint")
        } catch let error as RepackError {
            guard case .sourceFingerprintRejected(let path, _) = error else {
                Issue.record("unexpected adapter error for malformed checksum metadata")
                return
            }
            #expect(path == fixture.root.appendingPathComponent("SHA256SUMS").path)
        } catch {
            Issue.record("unexpected non-RepackError for malformed checksum metadata")
        }
        #expect(!inspected)
    }

    @Test func missingExtraDuplicateAndUnsafeShardNamesAreRejected() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }

        var missing = fixture.shards
        missing.removeLast()
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: missing,
                operations: fixture.operations())
        }

        var extra = fixture.shards
        extra.append("model-00027-of-00026.safetensors")
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: extra,
                operations: fixture.operations())
        }

        var duplicate = fixture.shards
        duplicate[1] = duplicate[0]
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: duplicate,
                operations: fixture.operations())
        }

        var unsafe = fixture.shards
        unsafe[0] = "../model-00001-of-00026.safetensors"
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: unsafe,
                operations: fixture.operations())
        }
    }

    @Test func wrongShardDigestAndAggregateSizeAreRejected() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }

        var wrongDigest = fixture.operations()
        wrongDigest.inspectAndHashFile = { path, _ in
            let expected = fixture.expectedDigest(for: path)
            return (fixture.size(for: path), expected == fixture.expectedDigest(for: path)
                ? String(repeating: "0", count: 64) : expected)
        }
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: wrongDigest)
        }

        var wrongSize = fixture.operations()
        wrongSize.inspectAndHashFile = { path, _ in
            (fixture.size(for: path) + (path.hasSuffix("00026.safetensors") ? 1 : 0),
             fixture.expectedDigest(for: path))
        }
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: wrongSize)
        }
    }

    @Test func symlinkAndShortReadFailuresPropagateFromNoFollowCallback() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }

        try FileManager.default.createSymbolicLink(
            atPath: fixture.root.appendingPathComponent(fixture.shards[0]).path,
            withDestinationPath: "not-a-real-payload")
        var symlink = fixture.operations()
        symlink.inspectAndHashFile = { path, _ in
            #expect(try Posix.entryKind(path) == .symlink)
            throw RepackError.installStateIncompatible(
                detail: "no-follow rejected symlink (path)")
        }
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: symlink)
        }

        var shortRead = fixture.operations()
        shortRead.inspectAndHashFile = { path, _ in
            throw RepackError.preadShort(path: path, expected: 64 * 1024, got: 7, errno: 0)
        }
        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: shortRead)
        }
    }

    @Test func sharedShortReadErrorMapsToExactRepackErrorValues() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        let expectedPath = fixture.root.appendingPathComponent(fixture.shards[0]).path
        var inspected = 0
        var operations = fixture.operations()
        operations.inspectAndHashFile = { _, _ in
            inspected += 1
            throw OfficialQwenPayloadVerificationError.preadShort(
                path: expectedPath, expected: 512 * 1024, got: 7, errno: EIO)
        }

        do {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: operations)
            Issue.record("expected shared short-read error to be mapped")
        } catch let error as RepackError {
            guard case .preadShort(let path, let expected, let got, let code) = error else {
                Issue.record("shared short-read error mapped to the wrong RepackError case")
                return
            }
            #expect(path == expectedPath)
            #expect(expected == 512 * 1024)
            #expect(got == 7)
            #expect(code == EIO)
        } catch {
            Issue.record("shared short-read error escaped as a non-RepackError")
        }
        #expect(inspected == 1)
    }

    @Test func cancellationStopsAtEverySyntheticShardBoundary() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        for stopAfterShard in 1...fixture.shards.count {
            var completedShards = 0
            var inspectedShards = 0
            var operations = fixture.operations()
            operations.inspectAndHashFile = { path, progress in
                inspectedShards += 1
                let size = fixture.size(for: path)
                try progress(size)
                completedShards += 1
                return (size, fixture.expectedDigest(for: path))
            }
            operations.cancellationCheck = {
                if completedShards >= stopAfterShard { throw CancellationError() }
            }
            #expect(throws: CancellationError.self) {
                _ = try LocalOfficialQwenPayloadVerifier.verify(
                    snapshotDirectory: fixture.root.path,
                    shardFilenames: fixture.shards,
                    operations: operations)
            }
            #expect(inspectedShards == stopAfterShard)
            #expect(completedShards == stopAfterShard)
        }
    }

    @Test func syntheticProgressAdapterReportsMonotonicAggregateBytes() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        var progress: [UInt64] = []

        _ = try LocalOfficialQwenPayloadVerifier.verify(
            snapshotDirectory: fixture.root.path,
            shardFilenames: fixture.shards,
            operations: fixture.operations(),
            progress: { completed, total in
                #expect(total == LocalOfficialQwenPayloadVerifier.expectedShardBytes)
                progress.append(completed)
            })

        #expect(progress.count == 2 * fixture.shards.count)
        #expect(progress.first == PayloadVerifierFixture.syntheticShardSize)
        #expect(progress.last == LocalOfficialQwenPayloadVerifier.expectedShardBytes)
        #expect(zip(progress, progress.dropFirst()).allSatisfy { pair in pair.0 <= pair.1 })
        #expect(progress.allSatisfy { $0 <= LocalOfficialQwenPayloadVerifier.expectedShardBytes })
    }

    @Test func overflowingSyntheticProgressMapsToConfigurationInvalid() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        var inspected = 0
        var operations = fixture.operations()
        operations.inspectAndHashFile = { path, progress in
            inspected += 1
            if inspected == 1 {
                let size = fixture.size(for: path)
                try progress(size)
                return (size, fixture.expectedDigest(for: path))
            }
            try progress(UInt64.max)
            return (fixture.size(for: path), fixture.expectedDigest(for: path))
        }

        do {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: operations)
            Issue.record("expected overflowing progress arithmetic to be rejected")
        } catch let error as RepackError {
            guard case .configurationInvalid = error else {
                Issue.record("unexpected adapter error for overflowing progress")
                return
            }
        } catch {
            Issue.record("unexpected non-RepackError for overflowing progress")
        }
        #expect(inspected == 2)
    }

    @Test func sourceMutationBetweenValidationAndPublicationFailsIdentityCheck() throws {
        let fixture = try PayloadVerifierFixture.make()
        defer { fixture.remove() }
        var inspected = 0
        var operations = fixture.operations()
        operations.inspectAndHashFile = { path, progress in
            inspected += 1
            try progress(0)
            if inspected == 2 {
                throw RepackError.installStateIncompatible(
                    detail: "official Qwen shard changed while it was authenticated")
            }
            return (fixture.size(for: path), fixture.expectedDigest(for: path))
        }

        #expect(throws: RepackError.self) {
            _ = try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.root.path,
                shardFilenames: fixture.shards,
                operations: operations)
        }
        #expect(inspected == 2)
    }
}

private struct PayloadVerifierFixture {
    static let syntheticShardSize: UInt64 = 2_765_529_876

    let root: URL
    let checksumData: Data
    let shards: [String]
    let entries: [String: String]

    static func make() throws -> Self {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-qwen-payload-\(UUID().uuidString)",
                                   isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let checksum = Data((canonicalChecksumManifest + "\n").utf8)
        try checksum.write(to: root.appendingPathComponent("SHA256SUMS"))
        let text = try #require(String(data: checksum, encoding: .utf8))
        let parsed = text.split(separator: "\n")
            .reduce(into: [String: String]()) { result, line in
                let pieces = line.split(separator: "  ", maxSplits: 1)
                if pieces.count == 2 { result[String(pieces[1])] = String(pieces[0]) }
            }
        let shards = (1...26).map {
            String(format: "model-%05d-of-00026.safetensors", $0)
        }
        return .init(root: root, checksumData: checksum, shards: shards, entries: parsed)
    }

    func operations() -> LocalOfficialQwenPayloadVerifierOperations {
        .init(
            readChecksumManifest: { _, _ in checksumData },
            inspectAndHashFile: { path, progress in
                try progress(size(for: path))
                return (size(for: path), expectedDigest(for: path))
            },
            cancellationCheck: {})
    }

    func expectedDigest(for path: String) -> String {
        entries[(path as NSString).lastPathComponent] ?? String(repeating: "0", count: 64)
    }

    func size(for path: String) -> UInt64 {
        _ = path
        return Self.syntheticShardSize
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

private let canonicalChecksumManifest = """
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
