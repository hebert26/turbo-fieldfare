import CryptoKit
import Darwin
import Foundation
import Testing
@testable import TurboFieldfareOfficialQwenSource

/// Independent SHA-256 literals were calculated with Python hashlib over the
/// explicit fixture bytes, not through the production verifier.
private enum IndependentPayloadVerifierVectors {
    static let emptySHA256 =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    static let abcSHA256 =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    static let abdSHA256 =
        "a52d159f262b2c6ddb724a61840befc36eb30c88877a4030b65cbe86298449c9"
    static let tileBytes = 512 * 1024
    // hashlib.sha256(b"a" * (512 * 1024 + 1))
    static let repeatedAAtTileBoundarySHA256 =
        "8d666ffa0196841cce7c504d43bf27e311775220d2490a23a2f984a43d901015"
}

/// Literal pinned SHA256SUMS fixture copied from independent source metadata.
/// It is test input only; tiny callbacks below never read original shards.
private let syntheticPinnedChecksumManifest = """
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

@Suite(.serialized)
struct OfficialPayloadVerifierTests {
    @Test func hashesIndependentEmptyAndABCVectors() throws {
        let empty = try TinyPayloadFileFixture.make(bytes: Data())
        defer { empty.remove() }
        let emptyResult = try OfficialQwenPayloadVerifier.hashRegularFile(path: empty.fileURL.path)
        #expect(emptyResult.size == 0)
        #expect(emptyResult.sha256 == IndependentPayloadVerifierVectors.emptySHA256)

        let abc = try TinyPayloadFileFixture.make(bytes: Data("abc".utf8))
        defer { abc.remove() }
        let result = try OfficialQwenPayloadVerifier.hashRegularFile(path: abc.fileURL.path)
        #expect(result.size == 3)
        #expect(result.sha256 == IndependentPayloadVerifierVectors.abcSHA256)
    }

    @Test func detectsActualABDMutationThroughProductionHashPath() throws {
        let fixture = try TinyPayloadFileFixture.make(bytes: Data("abc".utf8))
        defer { fixture.remove() }

        let before = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path)
        try fixture.overwrite(with: Data("abd".utf8))
        let after = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path)

        #expect(before.size == 3)
        #expect(before.sha256 == IndependentPayloadVerifierVectors.abcSHA256)
        #expect(after.size == 3)
        #expect(after.sha256 == IndependentPayloadVerifierVectors.abdSHA256)
        #expect(before.sha256 != after.sha256)
    }

    @Test func readChecksumManifestAcceptsAtCapAndRejectsOneByteOver() throws {
        let manifest = Data((syntheticPinnedChecksumManifest + "\n").utf8)
        #expect(manifest.count == 3_571)

        let atCapFixture = try TinyPayloadFileFixture.make(bytes: manifest)
        defer { atCapFixture.remove() }
        let atCapPath = try atCapFixture.moveTinyFileToChecksumManifest()
        let read = try OfficialQwenPayloadVerifier.readChecksumManifest(
            path: atCapPath.path, maximumBytes: 3_571)
        #expect(read == manifest)

        var overCapBytes = manifest
        overCapBytes.append(0x20)
        let overCapFixture = try TinyPayloadFileFixture.make(bytes: overCapBytes)
        defer { overCapFixture.remove() }
        let overCapPath = try overCapFixture.moveTinyFileToChecksumManifest()
        do {
            _ = try OfficialQwenPayloadVerifier.readChecksumManifest(
                path: overCapPath.path, maximumBytes: 3_571)
            Issue.record("expected manifest larger than the byte cap to be rejected")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .installStateCorrupt(let path, let detail) = error else {
                Issue.record("unexpected verifier error for an over-cap manifest")
                return
            }
            #expect(path == overCapPath.path)
            #expect(detail.contains("3571-byte cap"))
        } catch {
            Issue.record("unexpected non-verifier error for an over-cap manifest")
        }
    }

    @Test func mutatedTinyShardDigestIsRejectedBeforeLaterShardCallbacks() throws {
        let fixture = try TinyPayloadFileFixture.make(bytes: Data("abc".utf8))
        defer { fixture.remove() }
        let before = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path)
        #expect(before.sha256 == IndependentPayloadVerifierVectors.abcSHA256)

        let manifest = Data((syntheticPinnedChecksumManifest + "\n").utf8)
        let manifestSHA256 = SHA256.hash(data: manifest)
            .map { String(format: "%02x", $0) }.joined()
        #expect(manifest.count == 3_571)
        #expect(manifestSHA256 ==
                "1378d1bb153694b13c20641fa5d2485dede401d1ce4c267953ba4a855fc59e7e")
        var inspectedPaths: [String] = []
        let operations = OfficialQwenPayloadVerifierOperations(
            readChecksumManifest: { _, _ in manifest },
            inspectAndHashFile: { path, progress in
                inspectedPaths.append(path)
                try fixture.overwrite(with: Data("abd".utf8))
                return try OfficialQwenPayloadVerifier.hashRegularFile(
                    path: fixture.fileURL.path, progress: progress)
            },
            cancellationCheck: {})
        let shardNames = (1...26).map {
            String(format: "model-%05d-of-00026.safetensors", $0)
        }
        let firstShardPath = fixture.directory
            .appendingPathComponent("model-00001-of-00026.safetensors").path

        do {
            _ = try OfficialQwenPayloadVerifier.verify(
                snapshotDirectory: fixture.directory.path,
                shardFilenames: shardNames,
                operations: operations)
            Issue.record("expected the changed tiny shard bytes to fail the pinned digest")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .sourceFingerprintRejected(let path, let sha256) = error else {
                Issue.record("unexpected verifier error for changed shard bytes")
                return
            }
            #expect(path == firstShardPath)
            #expect(sha256 == IndependentPayloadVerifierVectors.abdSHA256)
        } catch {
            Issue.record("unexpected non-verifier error for changed shard bytes")
        }
        #expect(inspectedPaths == [firstShardPath])
    }

    @Test func packageVerifyPropagatesFinalRecheckCancellationAfterAllSyntheticShards() throws {
        let manifest = Data((syntheticPinnedChecksumManifest + "\n").utf8)
        let entries = independentPinnedManifestEntries()
        let shardNames = (1...26).map {
            String(format: "model-%05d-of-00026.safetensors", $0)
        }
        var inspectedShards = 0
        var finalRechecks = 0
        let operations = OfficialQwenPayloadVerifierOperations(
            readChecksumManifest: { _, maximumBytes in
                #expect(maximumBytes == 3_571)
                return manifest
            },
            inspectAndHashFile: { path, progress in
                inspectedShards += 1
                let size: UInt64 = 2_765_529_876
                try progress(size)
                let name = (path as NSString).lastPathComponent
                let digest = try #require(entries[name])
                return (size, digest)
            },
            cancellationCheck: {},
            finalRecheck: {
                finalRechecks += 1
                throw CancellationError()
            })

        do {
            _ = try OfficialQwenPayloadVerifier.verify(
                snapshotDirectory: "/synthetic-snapshot",
                shardFilenames: shardNames,
                operations: operations)
            Issue.record("expected cancellation from the final set-level recheck")
        } catch is CancellationError {
            #expect(inspectedShards == 26)
            #expect(finalRechecks == 1)
        } catch {
            Issue.record("unexpected error from final set-level recheck")
        }
    }

    @Test func reportsProgressAtBoundedHashTileBoundary() throws {
        let tileBytes = IndependentPayloadVerifierVectors.tileBytes
        let fixture = try TinyPayloadFileFixture.make(
            bytes: Data(repeating: 0x61, count: tileBytes + 1))
        defer { fixture.remove() }
        var progress: [UInt64] = []

        let result = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path) {
            progress.append($0)
        }

        #expect(progress == [UInt64(tileBytes), UInt64(tileBytes + 1)])
        #expect(result.size == UInt64(tileBytes + 1))
        #expect(result.sha256 == IndependentPayloadVerifierVectors.repeatedAAtTileBoundarySHA256)
    }

    @Test func progressCancellationStopsFurtherHashing() throws {
        let tileBytes = IndependentPayloadVerifierVectors.tileBytes
        let fixture = try TinyPayloadFileFixture.make(
            bytes: Data(repeating: 0x61, count: tileBytes + 1))
        defer { fixture.remove() }
        var progress: [UInt64] = []

        do {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path) {
                progress.append($0)
                throw CancellationError()
            }
            Issue.record("expected progress cancellation to escape the hash operation")
        } catch is CancellationError {
            #expect(progress == [UInt64(tileBytes)])
        } catch {
            Issue.record("unexpected cancellation error: \(error)")
        }
    }

    @Test func rejectsSymlinkLeafWithoutFollowingIt() throws {
        let fixture = try TinyPayloadFileFixture.make(bytes: Data("abc".utf8))
        defer { fixture.remove() }
        let link = fixture.directory.appendingPathComponent("linked-payload.bin")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: fixture.fileURL)

        do {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: link.path)
            Issue.record("expected no-follow hashing to reject a symlink leaf")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .fileOpenFailed(let path, let code) = error else {
                Issue.record("unexpected verifier error for symlink: \(error)")
                return
            }
            #expect(path == link.path)
            #expect(code == ELOOP)
        } catch {
            Issue.record("unexpected non-verifier error for symlink: \(error)")
        }
    }

    @Test func rejectsDirectoryAsNonRegularFile() throws {
        let fixture = try TinyPayloadFileFixture.make(bytes: Data("unused".utf8))
        defer { fixture.remove() }

        do {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.directory.path)
            Issue.record("expected directory leaf to be rejected as nonregular")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .fileStatFailed(let path, let code) = error else {
                Issue.record("unexpected verifier error for directory leaf")
                return
            }
            #expect(path == fixture.directory.path)
            #expect(code == EINVAL)
        } catch {
            Issue.record("unexpected non-verifier error for directory leaf")
        }
    }

    @Test func rejectsFIFOWithoutBlocking() throws {
        let fixture = try TinyPayloadFileFixture.make(bytes: Data("unused".utf8))
        defer { fixture.remove() }
        let fifo = fixture.directory.appendingPathComponent("payload.pipe")
        guard mkfifo(fifo.path, mode_t(S_IRUSR | S_IWUSR)) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }

        do {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fifo.path)
            Issue.record("expected non-regular FIFO to be rejected")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .fileStatFailed(let path, let code) = error else {
                Issue.record("unexpected verifier error for FIFO: \(error)")
                return
            }
            #expect(path == fifo.path)
            #expect(code == EINVAL)
        } catch {
            Issue.record("unexpected non-verifier error for FIFO: \(error)")
        }
    }

    @Test func rejectsTruncationDuringFirstHashTile() throws {
        let tileBytes = IndependentPayloadVerifierVectors.tileBytes
        let fixture = try TinyPayloadFileFixture.make(
            bytes: Data(repeating: 0x61, count: tileBytes + 1))
        defer { fixture.remove() }
        var truncated = false

        do {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path) {
                completed in
                if completed == UInt64(tileBytes), !truncated {
                    try fixture.truncateInPlace(to: 1)
                    truncated = true
                }
            }
            Issue.record("expected truncation during hashing to fail before returning a digest")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .preadShort(let path, let expected, let got, _) = error else {
                Issue.record("unexpected verifier error after truncation")
                return
            }
            #expect(path == fixture.fileURL.path)
            #expect(expected == 1)
            #expect(got == 0)
        } catch {
            Issue.record("unexpected non-verifier error after truncation")
        }
        #expect(truncated)
    }

    @Test func rejectsGrowthDuringFirstHashTile() throws {
        let tileBytes = IndependentPayloadVerifierVectors.tileBytes
        let fixture = try TinyPayloadFileFixture.make(
            bytes: Data(repeating: 0x61, count: tileBytes + 1))
        defer { fixture.remove() }
        var grew = false

        do {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path) {
                completed in
                if completed == UInt64(tileBytes), !grew {
                    try fixture.appendInPlace(0x62)
                    grew = true
                }
            }
            Issue.record("expected growth during hashing to fail before returning a digest")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .installStateIncompatible(let detail) = error else {
                Issue.record("unexpected verifier error after growth")
                return
            }
            #expect(detail.contains("file grew while it was authenticated"))
        } catch {
            Issue.record("unexpected non-verifier error after growth")
        }
        #expect(grew)
    }

    @Test func rejectsRootEntryMutationAtFinalSetRecheck() throws {
        let tileBytes = IndependentPayloadVerifierVectors.tileBytes
        let fixture = try TinyPayloadFileFixture.make(
            bytes: Data(repeating: 0x61, count: tileBytes + 1))
        defer { fixture.remove() }
        var rootEntryChanged = false

        do {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path) {
                completed in
                if completed == UInt64(tileBytes), !rootEntryChanged {
                    try fixture.createAdditionalRootEntry()
                    rootEntryChanged = true
                }
            }
            Issue.record("expected final set recheck to detect a changed root directory")
        } catch let error as OfficialQwenPayloadVerificationError {
            guard case .installStateIncompatible(let detail) = error else {
                Issue.record("unexpected verifier error after root-entry mutation")
                return
            }
            #expect(detail.contains("source directory changed while it was authenticated"))
        } catch {
            Issue.record("unexpected non-verifier error after root-entry mutation")
        }
        #expect(rootEntryChanged)
    }

    @Test func rejectsSamePathMutationDuringHashBeforeReturningDigest() throws {
        let tileBytes = IndependentPayloadVerifierVectors.tileBytes
        let fixture = try TinyPayloadFileFixture.make(
            bytes: Data(repeating: 0x61, count: tileBytes + 1))
        defer { fixture.remove() }
        var mutated = false

        expectMutationRejection {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path) {
                completed in
                if completed == UInt64(tileBytes), !mutated {
                    mutated = true
                    try fixture.overwriteFirstByteInPlace(with: 0x62)
                }
            }
        }
        #expect(mutated)
    }

    @Test func rejectsLeafReplacementDuringHashBeforeReturningDigest() throws {
        let tileBytes = IndependentPayloadVerifierVectors.tileBytes
        let fixture = try TinyPayloadFileFixture.make(
            bytes: Data(repeating: 0x61, count: tileBytes + 1))
        defer { fixture.remove() }
        let replacement = fixture.directory.appendingPathComponent("replacement.bin")
        try Data(repeating: 0x62, count: tileBytes + 1).write(to: replacement)
        var replaced = false

        expectMutationRejection {
            _ = try OfficialQwenPayloadVerifier.hashRegularFile(path: fixture.fileURL.path) {
                completed in
                if completed == UInt64(tileBytes), !replaced {
                    let result = replacement.path.withCString { source in
                        fixture.fileURL.path.withCString { destination in
                            Darwin.rename(source, destination)
                        }
                    }
                    guard result == 0 else {
                        throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
                    }
                    replaced = true
                }
            }
        }
        #expect(replaced)
    }
}

private struct TinyPayloadFileFixture {
    let directory: URL
    let fileURL: URL

    static func make(bytes: Data) throws -> Self {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-payload-hash-\(UUID().uuidString)",
                                   isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let fileURL = directory.appendingPathComponent("tiny-payload.bin")
        try bytes.write(to: fileURL)
        return Self(directory: directory, fileURL: fileURL)
    }

    func overwrite(with bytes: Data) throws {
        try bytes.write(to: fileURL)
    }

    func moveTinyFileToChecksumManifest() throws -> URL {
        let manifestURL = directory.appendingPathComponent("SHA256SUMS")
        try FileManager.default.moveItem(at: fileURL, to: manifestURL)
        return manifestURL
    }

    func createAdditionalRootEntry() throws {
        let entry = directory.appendingPathComponent("late-entry.bin")
        try Data("new root entry".utf8).write(to: entry)
    }

    func overwriteFirstByteInPlace(with byte: UInt8) throws {
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data([byte]))
        try handle.synchronize()
    }

    func truncateInPlace(to size: UInt64) throws {
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.truncate(atOffset: size)
        try handle.synchronize()
    }

    func appendInPlace(_ byte: UInt8) throws {
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data([byte]))
        try handle.synchronize()
    }

    func remove() {
        try? FileManager.default.removeItem(at: directory)
    }
}

private func independentPinnedManifestEntries() -> [String: String] {
    var result: [String: String] = [:]
    for line in syntheticPinnedChecksumManifest.split(separator: "\n") {
        result[String(line.dropFirst(66))] = String(line.prefix(64))
    }
    return result
}

private func expectMutationRejection(
    _ operation: () throws -> Void,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    do {
        try operation()
        Issue.record("expected a detected source mutation or replacement", sourceLocation: sourceLocation)
    } catch let error as OfficialQwenPayloadVerificationError {
        guard case .installStateIncompatible(let detail) = error else {
            Issue.record("unexpected verifier error for source mutation: \(error)",
                         sourceLocation: sourceLocation)
            return
        }
        #expect(detail.contains("source file changed while it was authenticated"),
                sourceLocation: sourceLocation)
    } catch {
        Issue.record("unexpected error for source mutation: \(error)", sourceLocation: sourceLocation)
    }
}
