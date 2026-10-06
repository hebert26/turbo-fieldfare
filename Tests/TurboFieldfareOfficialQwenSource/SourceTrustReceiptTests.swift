import CryptoKit
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

private enum SyntheticCheckpointFailure: Error {
    case injected
}

private enum TrustFailureExpectation {
    case stale
    case publishedStale
}

private func expectTrustFailure(
    _ operation: () throws -> Void,
    expected: TrustFailureExpectation
) {
    do {
        try operation()
        Issue.record("Expected trust failure: \(expected)")
    } catch let error as OfficialSourceTrust.TrustError {
        switch (expected, error) {
        case (.stale, .stale(_)), (.publishedStale, .publishedStale(_)):
            break
        default:
            Issue.record("Unexpected trust error: \(error)")
        }
    } catch {
        Issue.record("Unexpected error type: \(error)")
    }
}

private func seedFullSyntheticReceipt(
    for fixture: Task68SyntheticTrustFixture
) throws -> (shardBytes: UInt64, receipt: OfficialSourceTrustReceipt, bytes: Data) {
    let shardBytes = try fixture.totalShardBytes()
    let receipt = try OfficialSourceTrust.verifySynthetic(
        at: fixture.registration.modelDirectory,
        policy: .fullSha256,
        expectedShardBytes: shardBytes,
        fullVerification: {})
    return (shardBytes, receipt, try Data(contentsOf: fixture.receiptURL))
}

private func overwriteInPlace(_ url: URL, with bytes: Data) throws {
    let file = try FileHandle(forWritingTo: url)
    defer { try? file.close() }
    try file.seek(toOffset: 0)
    try file.write(contentsOf: bytes)
    try file.truncate(atOffset: UInt64(bytes.count))
    try file.synchronize()
}

private func mutateFirstByteInPlace(_ url: URL) throws -> Data {
    var bytes = try Data(contentsOf: url)
    guard !bytes.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
    bytes[bytes.startIndex] ^= 0x01
    try overwriteInPlace(url, with: bytes)
    return bytes
}

private final class Task68LockedValue<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Value

    init(_ value: Value) { storage = value }

    var value: Value {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func set(_ value: Value) {
        lock.lock()
        defer { lock.unlock() }
        storage = value
    }
}

@Suite struct SourceTrustReceiptTests {
    @Test func probeAndTrustedReopenObserveMetadataWithoutOpeningOrReadingShards() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let expectedShardBytes = try fixture.totalShardBytes()
        var fullVerificationCalls = 0

        // Seed the receipt through the existing private synthetic full-verification
        // lifecycle. It cannot mint an official-size receipt or pass the public pin.
        let fullReceipt = try OfficialSourceTrust.verifySynthetic(
            at: fixture.registration.modelDirectory,
            policy: .fullSha256,
            expectedShardBytes: expectedShardBytes,
            fullVerification: { fullVerificationCalls += 1 })
        #expect(fullVerificationCalls == 1)

        let shardPaths = Set(fixture.registration.descriptor.shards.map {
            fixture.sourceRootURL.appendingPathComponent($0.filename).path
        })
        #expect(shardPaths.count == OfficialQwenPayloadVerifier.expectedShardCount)
        let markerPath = fixture.logicalModelURL
            .appendingPathComponent(OfficialSourceDescriptor.markerFilename).path

        var probeEvents: [OfficialSourceTrustIOEvent] = []
        let probe = try OfficialSourceTrust.probeObserved(
            at: fixture.registration.modelDirectory,
            observeIO: { probeEvents.append($0) })
        // A synthetic-byte receipt is not valid for the official-size public
        // probe. This policy result is separate from the observed I/O claim.
        #expect(probe.receipt == .invalid)

        var trustedEvents: [OfficialSourceTrustIOEvent] = []
        var trustedCheckpointCalls = 0
        let reopened = try OfficialSourceTrust.verifySynthetic(
            at: fixture.registration.modelDirectory,
            policy: .sizeCheckTrustedReceipt,
            expectedShardBytes: expectedShardBytes,
            fullVerification: { fullVerificationCalls += 1 },
            observeIO: { trustedEvents.append($0) },
            trustedCheckpoint: { trustedCheckpointCalls += 1 })
        #expect(reopened == fullReceipt)
        #expect(fullVerificationCalls == 1)
        #expect(trustedCheckpointCalls == 1)

        func openedOrReadPaths(in events: [OfficialSourceTrustIOEvent]) -> Set<String> {
            Set(events.compactMap { event in
                switch event {
                case .statFile:
                    nil
                case let .openFile(path):
                    path
                case let .readFile(path, _, _):
                    path
                }
            })
        }
        func statPaths(in events: [OfficialSourceTrustIOEvent]) -> Set<String> {
            Set(events.compactMap { event in
                if case let .statFile(path) = event { return path }
                return nil
            })
        }

        let probeMetadataPaths = openedOrReadPaths(in: probeEvents)
        let trustedMetadataPaths = openedOrReadPaths(in: trustedEvents)
        let trustedShardStatPaths = statPaths(in: trustedEvents).intersection(shardPaths)
        #expect(!probeEvents.isEmpty)
        #expect(!trustedEvents.isEmpty)
        #expect(trustedShardStatPaths.count == OfficialQwenPayloadVerifier.expectedShardCount)
        #expect(probeMetadataPaths.contains(markerPath))
        #expect(probeMetadataPaths.contains(fixture.receiptURL.path))
        #expect(trustedMetadataPaths.contains(markerPath))
        #expect(trustedMetadataPaths.contains(fixture.receiptURL.path))
        #expect(shardPaths.isSubset(of: statPaths(in: trustedEvents)))
        #expect(probeMetadataPaths.isDisjoint(with: shardPaths))
        #expect(trustedMetadataPaths.isDisjoint(with: shardPaths))
    }

    @Test func decodingSyntheticReceiptDoesNotMakeProbeTrusted() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let bytes = try fixture.receiptData()
        let decoded = try OfficialSourceTrust.decodeReceiptStrict(data: bytes)
        let totalShardBytes = try fixture.totalShardBytes()
        #expect(decoded.shardBytes == totalShardBytes)
        try fixture.installReceipt(bytes)

        let probe = try OfficialSourceTrust.probe(at: fixture.registration.modelDirectory)

        #expect(probe.receipt == .invalid)
    }

    @Test func syntheticFullVerificationPublishesAndTrustedReopenMatches() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        var fullVerificationCalls = 0

        let fullReceipt = try OfficialSourceTrust.verifySynthetic(
            at: fixture.registration.modelDirectory,
            policy: .fullSha256,
            expectedShardBytes: totalShardBytes,
            fullVerification: { fullVerificationCalls += 1 })

        #expect(fullVerificationCalls == 1)
        #expect(fullReceipt.shardBytes == totalShardBytes)
        #expect(FileManager.default.fileExists(atPath: fixture.receiptURL.path))
        let probe = try OfficialSourceTrust.probe(at: fixture.registration.modelDirectory)
        #expect(probe.receipt == .invalid)

        let reopened = try OfficialSourceTrust.verifySynthetic(
            at: fixture.registration.modelDirectory,
            policy: .sizeCheckTrustedReceipt,
            expectedShardBytes: totalShardBytes,
            fullVerification: {})
        #expect(reopened == fullReceipt)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verify(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt)
        }
    }

    @Test func probeReportsMissingReceipt() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }

        let probe = try OfficialSourceTrust.probe(at: fixture.registration.modelDirectory)

        #expect(probe.receipt == .missing)
    }

    @Test func trustedReopenFailsWithoutReceipt() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verify(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt)
        }
    }

    @Test func fullVerificationCheckpointFailurePreservesExistingReceipt() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        let originalReceipt = try fixture.installReceipt()

        #expect(throws: SyntheticCheckpointFailure.self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: totalShardBytes,
                fullVerification: { throw SyntheticCheckpointFailure.injected })
        }

        #expect(try Data(contentsOf: fixture.receiptURL) == originalReceipt)
    }

    @Test func trustedReopenRejectsSameSizeShardMutation() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        try fixture.installReceipt()

        let shard = try FileHandle(forWritingTo: fixture.firstShardURL)
        try shard.seek(toOffset: 0)
        try shard.write(contentsOf: Data("abd".utf8))
        try shard.close()
        #expect(try OfficialQwenPayloadVerifier.hashRegularFile(
            path: fixture.firstShardURL.path).size == 3)
        #expect(try fixture.totalShardBytes() == totalShardBytes)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: totalShardBytes,
                fullVerification: {})
        }
    }

    @Test func checkpointFailuresPreserveReceipt() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        let originalReceipt = try fixture.installReceipt()

        #expect(throws: SyntheticCheckpointFailure.self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: totalShardBytes,
                fullVerification: {},
                checkpoint: { _ in throw SyntheticCheckpointFailure.injected })
        }

        #expect(try Data(contentsOf: fixture.receiptURL) == originalReceipt)
    }

    @Test func syncFailureReportsPublishedUncertainty() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: totalShardBytes,
                fullVerification: {},
                syncDirectory: { _ in -1 })
        }

        #expect(FileManager.default.fileExists(atPath: fixture.receiptURL.path))
    }

    @Test func trustedReopenRejectsTruncatedShard() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        try fixture.installReceipt()
        try Data("a".utf8).write(to: fixture.firstShardURL)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: totalShardBytes,
                fullVerification: {})
        }
    }

    @Test func trustedReopenRejectsChangedSidecar() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        try fixture.installReceipt()
        try Data("changed sidecar".utf8).write(to: fixture.sidecarURL)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: totalShardBytes,
                fullVerification: {})
        }
    }

    @Test func decodeRejectsUnknownFieldAndUnsupportedVersion() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let receiptData = try fixture.receiptData()

        var withUnknownField = try JSONSerialization.jsonObject(with: receiptData) as! [String: Any]
        withUnknownField["unknown"] = 1
        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(
                data: JSONSerialization.data(withJSONObject: withUnknownField))
        }

        var unsupportedVersion = try JSONSerialization.jsonObject(with: receiptData) as! [String: Any]
        unsupportedVersion["version"] = 2
        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(
                data: JSONSerialization.data(withJSONObject: unsupportedVersion))
        }
    }

    @Test func decodeRejectsInvalidFingerprint() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        var receipt = try JSONSerialization.jsonObject(with: fixture.receiptData()) as! [String: Any]
        var files = receipt["files"] as! [[String: Any]]
        var firstFile = files[0]
        var fingerprint = firstFile["fingerprint"] as! [String: Any]
        fingerprint["size"] = -1
        firstFile["fingerprint"] = fingerprint
        files[0] = firstFile
        receipt["files"] = files

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(
                data: JSONSerialization.data(withJSONObject: receipt))
        }
    }

    @Test func receiptBindingMismatchIsInvalid() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        var receipt = try JSONSerialization.jsonObject(with: fixture.receiptData()) as! [String: Any]
        receipt["logicalModelPath"] = "different-model"
        try fixture.installReceipt(JSONSerialization.data(withJSONObject: receipt))

        let probe = try OfficialSourceTrust.probe(at: fixture.registration.modelDirectory)
        #expect(probe.receipt == .invalid)
    }

    @Test func beforePublishCheckpointFailurePreservesReceipt() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        let originalReceipt = try fixture.installReceipt()

        #expect(throws: SyntheticCheckpointFailure.self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: totalShardBytes,
                fullVerification: {},
                checkpoint: { checkpoint in
                    if case .beforePublish = checkpoint {
                        throw SyntheticCheckpointFailure.injected
                    }
                })
        }

        #expect(try Data(contentsOf: fixture.receiptURL) == originalReceipt)
    }

    @Test func syncDirectoryFailureReportsPublishedUncertainty() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        var syncCalls = 0

        do {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: totalShardBytes,
                fullVerification: {},
                syncDirectory: { _ in
                    syncCalls += 1
                    return syncCalls == 1 ? 0 : -1
                })
            Issue.record("Expected published durability uncertainty")
        } catch OfficialSourceTrust.TrustError.publishedDurabilityUnknown {
            // Expected: the receipt was published, but directory sync failed.
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        #expect(FileManager.default.fileExists(atPath: fixture.receiptURL.path))
    }

    @Test func trustedReopenRejectsReceiptRemoval() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        try fixture.installReceipt(fixture.receiptData())
        try FileManager.default.removeItem(at: fixture.receiptURL)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: totalShardBytes,
                fullVerification: {})
        }
    }

    @Test func trustedReopenRejectsMarkerMutation() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        try fixture.installReceipt(fixture.receiptData())
        let markerURL = fixture.logicalModelURL.appendingPathComponent("official-source.json")
        var markerData = try Data(contentsOf: markerURL)
        markerData[markerData.startIndex] ^= 0x01
        try markerData.write(to: markerURL)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: totalShardBytes,
                fullVerification: {})
        }
    }

    @Test func trustedReopenRejectsSourceRootReplacement() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let totalShardBytes = try fixture.totalShardBytes()
        try fixture.installReceipt(fixture.receiptData())
        let fileManager = FileManager.default
        let backupURL = fixture.sourceRootURL.deletingLastPathComponent()
            .appendingPathComponent(".source-root-backup-\(UUID().uuidString)", isDirectory: true)
        try fileManager.moveItem(at: fixture.sourceRootURL, to: backupURL)
        defer {
            do {
                if fileManager.fileExists(atPath: fixture.sourceRootURL.path) {
                    try fileManager.removeItem(at: fixture.sourceRootURL)
                }
                try fileManager.moveItem(at: backupURL, to: fixture.sourceRootURL)
            } catch {
                Issue.record("Failed to restore source root: \(error)")
            }
        }
        try fileManager.createDirectory(
            at: fixture.sourceRootURL,
            withIntermediateDirectories: false)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: totalShardBytes,
                fullVerification: {})
        }
    }

    @Test func decodeRejectsEscapedDuplicateAndNonIntegerNumbers() throws {
        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(
                data: Data(#"{"version":1,"vers\u0069on":1}"#.utf8))
        }
        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(data: Data(#"{"version":1.5}"#.utf8))
        }
        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(data: Data(#"{"version":1e0}"#.utf8))
        }
        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(
                data: Data(#"{"version":999999999999999999999999999999999999}"#.utf8))
        }
    }

    @Test func decodeRejectsEmptyBytes() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(data: Data())
        }
    }

    @Test func decodeRejectsInvalidUTF8() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(data: Data([0xff]))
        }
    }

    @Test func decodeRejectsMalformedJSON() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(data: Data("{".utf8))
        }
    }

    @Test func decodeRejectsDuplicateKeys() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(
                data: Data(#"{"version":1,"version":1}"#.utf8))
        }
    }

    @Test func invalidReceiptBytesFailDecodeAndProbe() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let bytes = Data("not a receipt".utf8)
        try fixture.installReceipt(bytes)

        #expect(throws: (any Error).self) {
            _ = try OfficialSourceTrust.decodeReceiptStrict(data: bytes)
        }
        let probe = try OfficialSourceTrust.probe(at: fixture.registration.modelDirectory)
        #expect(probe.receipt == .invalid)
    }

    @Test func hashesSyntheticABCAndMutatedABDWithProductionVerifier() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }

        let abc = try OfficialQwenPayloadVerifier.hashRegularFile(
            path: fixture.firstShardURL.path)
        #expect(abc.size == 3)
        #expect(abc.sha256 == Task68SyntheticTrustFixture.tinyABCExpectedSHA256)

        try Data("abd".utf8).write(to: fixture.firstShardURL)
        let abd = try OfficialQwenPayloadVerifier.hashRegularFile(
            path: fixture.firstShardURL.path)
        #expect(abd.size == 3)
        #expect(abd.sha256 == Task68SyntheticTrustFixture.tinyABDAfterMutationExpectedSHA256)
        #expect(abc.sha256 != abd.sha256)
    }

    @Test func trustedNoOpCheckpointAllowsReopen() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        var fullVerificationCalls = 0
        var trustedCheckpointCalls = 0

        let reopened = try OfficialSourceTrust.verifySynthetic(
            at: fixture.registration.modelDirectory,
            policy: .sizeCheckTrustedReceipt,
            expectedShardBytes: seeded.shardBytes,
            fullVerification: { fullVerificationCalls += 1 },
            trustedCheckpoint: { trustedCheckpointCalls += 1 })

        #expect(reopened == seeded.receipt)
        #expect(fullVerificationCalls == 0)
        #expect(trustedCheckpointCalls == 1)
    }

    @Test func trustedReopenRejectsSameSizeShardMutationAtCheckpoint() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        var mutated = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                trustedCheckpoint: {
                    _ = try mutateFirstByteInPlace(fixture.firstShardURL)
                    mutated = true
                })
        }, expected: .stale)

        #expect(mutated)
        #expect((try Data(contentsOf: fixture.firstShardURL)).count == 3)
        #expect(try Data(contentsOf: fixture.receiptURL) == seeded.bytes)
    }

    @Test func trustedReopenRejectsSidecarMutationAtCheckpoint() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let originalSidecar = try Data(contentsOf: fixture.sidecarURL)
        var mutated = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                trustedCheckpoint: {
                    _ = try mutateFirstByteInPlace(fixture.sidecarURL)
                    mutated = true
                })
        }, expected: .stale)

        #expect(mutated)
        #expect(try Data(contentsOf: fixture.sidecarURL) != originalSidecar)
        #expect(try Data(contentsOf: fixture.receiptURL) == seeded.bytes)
    }

    @Test func trustedReopenRejectsSourceRootRenameAndReplacementAtCheckpoint() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let fileManager = FileManager.default
        let sourceRoot = fixture.sourceRootURL
        let backupURL = fixture.registration.root.appendingPathComponent(
            "task68-original-source-\(UUID().uuidString)", isDirectory: true)
        let replacementURL = fixture.registration.root.appendingPathComponent(
            "task68-replacement-source-\(UUID().uuidString)", isDirectory: true)
        var originalMoved = false
        defer {
            if originalMoved {
                if fileManager.fileExists(atPath: sourceRoot.path) {
                    try? fileManager.removeItem(at: sourceRoot)
                }
                if fileManager.fileExists(atPath: backupURL.path) {
                    try? fileManager.moveItem(at: backupURL, to: sourceRoot)
                }
            } else if fileManager.fileExists(atPath: replacementURL.path) {
                try? fileManager.removeItem(at: replacementURL)
            }
        }

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                trustedCheckpoint: {
                    try fileManager.copyItem(at: sourceRoot, to: replacementURL)
                    try fileManager.moveItem(at: sourceRoot, to: backupURL)
                    originalMoved = true
                    try fileManager.moveItem(at: replacementURL, to: sourceRoot)
                })
        }, expected: .stale)

        #expect(originalMoved)
        #expect(fileManager.fileExists(atPath: sourceRoot.path))
        #expect(fileManager.fileExists(atPath: backupURL.path))
        #expect(try Data(contentsOf: fixture.receiptURL) == seeded.bytes)
    }

    @Test func trustedReopenRejectsReceiptReplacementAtCheckpoint() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let fileManager = FileManager.default
        let receiptURL = fixture.receiptURL
        let backupURL = fixture.registration.root.appendingPathComponent(
            "task68-original-receipt-\(UUID().uuidString)")
        let replacementURL = fixture.registration.root.appendingPathComponent(
            "task68-replacement-receipt-\(UUID().uuidString)")
        let replacement = Data("external receipt replacement".utf8)
        var replaced = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                trustedCheckpoint: {
                    try replacement.write(to: replacementURL)
                    try fileManager.moveItem(at: receiptURL, to: backupURL)
                    try fileManager.moveItem(at: replacementURL, to: receiptURL)
                    replaced = true
                })
        }, expected: .stale)

        #expect(replaced)
        #expect(try Data(contentsOf: receiptURL) == replacement)
    }

    @Test func trustedReopenRejectsLogicalRegistrationRelocationAtCheckpoint() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let fileManager = FileManager.default
        let registrationURL = fixture.registration.modelDirectory
        let relocatedURL = registrationURL.deletingLastPathComponent()
            .appendingPathComponent("relocated-registration-\(UUID().uuidString)",
                                    isDirectory: true)
        var relocated = false
        defer {
            if fileManager.fileExists(atPath: relocatedURL.path) {
                if fileManager.fileExists(atPath: registrationURL.path) {
                    try? fileManager.removeItem(at: registrationURL)
                }
                try? fileManager.moveItem(at: relocatedURL, to: registrationURL)
            }
        }

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: registrationURL,
                policy: .sizeCheckTrustedReceipt,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                trustedCheckpoint: {
                    try fileManager.moveItem(at: registrationURL, to: relocatedURL)
                    relocated = true
                })
        }, expected: .stale)

        #expect(relocated)
        #expect(fileManager.fileExists(atPath: relocatedURL.path))
        #expect(!fileManager.fileExists(atPath: registrationURL.path))
    }

    @Test func fullVerificationRejectsSourceMutationBeforePublishAndPreservesReceipt() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let replacement = Data("abd".utf8)
        var mutated = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                checkpoint: { checkpoint in
                    if case .beforePublish = checkpoint {
                        try overwriteInPlace(fixture.firstShardURL, with: replacement)
                        mutated = true
                    }
                })
        }, expected: .stale)

        #expect(mutated)
        #expect(try Data(contentsOf: fixture.firstShardURL) == replacement)
        #expect(try Data(contentsOf: fixture.receiptURL) == seeded.bytes)
    }

    @Test func fullVerificationRejectsLockReplacementBeforePublish() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let fileManager = FileManager.default
        let lockURL = fixture.registration.modelDirectory.deletingLastPathComponent()
            .appendingPathComponent(
                fixture.registration.modelDirectory.lastPathComponent + ".install.lock")
        let parkedURL = fixture.registration.root.appendingPathComponent(
            "task68-parked-install-lock-\(UUID().uuidString)")
        let replacement = Data("external lock replacement".utf8)
        var replaced = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                checkpoint: { checkpoint in
                    if case .beforePublish = checkpoint {
                        try fileManager.moveItem(at: lockURL, to: parkedURL)
                        try replacement.write(to: lockURL)
                        replaced = true
                    }
                })
        }, expected: .stale)

        #expect(replaced)
        #expect(try Data(contentsOf: lockURL) == replacement)
        #expect(fileManager.fileExists(atPath: parkedURL.path))
        #expect(try Data(contentsOf: fixture.receiptURL) == seeded.bytes)
    }

    @Test func fullVerificationRejectsPriorReceiptMutationBeforePublish() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let externalReceipt = Data(repeating: 0x5a, count: seeded.bytes.count)
        var mutated = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                checkpoint: { checkpoint in
                    if case .beforePublish = checkpoint {
                        try overwriteInPlace(fixture.receiptURL, with: externalReceipt)
                        mutated = true
                    }
                })
        }, expected: .stale)

        #expect(mutated)
        #expect(try Data(contentsOf: fixture.receiptURL) == externalReceipt)
    }

    @Test func fullVerificationRejectsConcurrentReceiptCreationBeforePublish() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let shardBytes = try fixture.totalShardBytes()
        let externalReceipt = Data("concurrent external receipt".utf8)
        var created = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: shardBytes,
                fullVerification: {},
                checkpoint: { checkpoint in
                    if case .beforePublish = checkpoint {
                        _ = try fixture.installReceipt(externalReceipt)
                        created = true
                    }
                })
        }, expected: .stale)

        #expect(created)
        #expect(try Data(contentsOf: fixture.receiptURL) == externalReceipt)
    }

    @Test func fullVerificationRejectsInPlaceStageMutationAndCleansOwnedStage() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        var stageURL: URL?
        var mutated = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                stagedReceiptCheckpoint: { url in
                    stageURL = url
                    _ = try mutateFirstByteInPlace(url)
                    mutated = true
                })
        }, expected: .stale)

        #expect(mutated)
        #expect(try Data(contentsOf: fixture.receiptURL) == seeded.bytes)
        if let stageURL {
            #expect(!FileManager.default.fileExists(atPath: stageURL.path))
        } else {
            Issue.record("Expected the writer to expose its owned stage")
        }
    }

    @Test func cancellationAtBeforePublishPreservesReceiptAndCleansOwnedStage() async throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let seeded = try seedFullSyntheticReceipt(for: fixture)
        let logicalModelURL = fixture.registration.modelDirectory
        let stageURL = Task68LockedValue<URL?>(nil)
        let operation = Task {
            try OfficialSourceTrust.verifySynthetic(
                at: logicalModelURL,
                policy: .fullSha256,
                expectedShardBytes: seeded.shardBytes,
                fullVerification: {},
                checkpoint: { checkpoint in
                    if case .beforePublish = checkpoint {
                        withUnsafeCurrentTask { $0?.cancel() }
                    }
                },
                stagedReceiptCheckpoint: { stageURL.set($0) })
        }

        do {
            _ = try await operation.value
            Issue.record("Expected cancellation at the precommit checkpoint")
        } catch is CancellationError {
            // Expected: cancellation occurred before the receipt commit point.
        } catch {
            Issue.record("Expected CancellationError, got \(error)")
        }

        #expect(try Data(contentsOf: fixture.receiptURL) == seeded.bytes)
        if let ownedStage = stageURL.value {
            #expect(!FileManager.default.fileExists(atPath: ownedStage.path))
        } else {
            Issue.record("Expected the writer to expose its owned stage")
        }
    }

    @Test func probeObservedPropagatesTaskCancellation() async throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        _ = try seedFullSyntheticReceipt(for: fixture)
        let logicalModelURL = fixture.registration.modelDirectory
        let receiptPath = fixture.receiptURL.path
        let observedIO = Task68LockedValue(false)
        let operation = Task {
            try OfficialSourceTrust.probeObserved(
                at: logicalModelURL,
                observeIO: { event in
                    guard case let .openFile(path) = event, path == receiptPath else { return }
                    observedIO.set(true)
                    withUnsafeCurrentTask { $0?.cancel() }
                })
        }

        do {
            _ = try await operation.value
            Issue.record("Expected probe cancellation to propagate")
        } catch is CancellationError {
            // Expected: the observer cancels after registration inspection, at receipt I/O.
        } catch {
            Issue.record("Expected CancellationError, got \(error)")
        }
        #expect(observedIO.value)
    }

    @Test func probeObservedPropagatesReceiptPermissionIOFailure() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        _ = try seedFullSyntheticReceipt(for: fixture)
        let receiptURL = fixture.receiptURL
        var original = stat()
        guard lstat(receiptURL.path, &original) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let originalMode = original.st_mode & mode_t(0o777)
        defer { _ = chmod(receiptURL.path, originalMode) }
        guard chmod(receiptURL.path, 0) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        var denied = stat()
        guard lstat(receiptURL.path, &denied) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        #expect((denied.st_mode & mode_t(0o777)) == 0)

        var observedReceiptOpen = false
        do {
            _ = try OfficialSourceTrust.probeObserved(
                at: fixture.registration.modelDirectory,
                observeIO: { event in
                    if case let .openFile(path) = event, path == receiptURL.path {
                        observedReceiptOpen = true
                    }
                })
            Issue.record("Expected receipt permission failure to propagate")
        } catch let error as OfficialSourceTrust.TrustError {
            guard case let .io(path, code) = error else {
                Issue.record("Expected receipt I/O error, got \(error)")
                return
            }
            #expect(path == OfficialSourceTrust.receiptFilename)
            #expect(code == EACCES)
        } catch {
            Issue.record("Expected TrustError.io, got \(error)")
        }
        #expect(observedReceiptOpen)
    }

    @Test func fullVerificationReportsPublishedStaleAfterSourceDeletionDuringSync() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let shardBytes = try fixture.totalShardBytes()
        var deletionAttempted = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: shardBytes,
                fullVerification: {},
                syncDirectory: { _ in
                    if !deletionAttempted {
                        deletionAttempted = true
                        try? FileManager.default.removeItem(at: fixture.sidecarURL)
                    }
                    return 0
                })
        }, expected: .publishedStale)

        #expect(deletionAttempted)
        #expect(!FileManager.default.fileExists(atPath: fixture.sidecarURL.path))
        #expect(FileManager.default.fileExists(atPath: fixture.receiptURL.path))
    }

    @Test func fullVerificationReportsPublishedStaleAfterSourceMutationDuringSync() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let shardBytes = try fixture.totalShardBytes()
        let originalSidecar = try Data(contentsOf: fixture.sidecarURL)
        var mutationAttempted = false

        expectTrustFailure({
            _ = try OfficialSourceTrust.verifySynthetic(
                at: fixture.registration.modelDirectory,
                policy: .fullSha256,
                expectedShardBytes: shardBytes,
                fullVerification: {},
                syncDirectory: { _ in
                    if !mutationAttempted {
                        mutationAttempted = true
                        _ = try? mutateFirstByteInPlace(fixture.sidecarURL)
                    }
                    return 0
                })
        }, expected: .publishedStale)

        #expect(mutationAttempted)
        #expect(try Data(contentsOf: fixture.sidecarURL) != originalSidecar)
        #expect(FileManager.default.fileExists(atPath: fixture.receiptURL.path))
    }

    @Test func metadataReceiptWithOfficialAggregateRemainsUntrustedAtProbe() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let shards = fixture.registration.descriptor.shards.sorted {
            $0.filename < $1.filename
        }
        let officialShardBytes = OfficialQwenPayloadVerifier.expectedShardBytes
        let shardCount = UInt64(shards.count)
        let baseSize = officialShardBytes / shardCount
        let remainder = officialShardBytes % shardCount
        var sizes: [String: UInt64] = [:]
        for (index, shard) in shards.enumerated() {
            sizes[shard.filename] = baseSize + (UInt64(index) < remainder ? 1 : 0)
        }

        var receipt = try fixture.receiptObject()
        var files = receipt["files"] as! [[String: Any]]
        for index in files.indices {
            guard let filename = files[index]["filename"] as? String,
                  let size = sizes[filename] else { continue }
            var fingerprint = files[index]["fingerprint"] as! [String: Any]
            fingerprint["size"] = NSNumber(value: size)
            files[index]["fingerprint"] = fingerprint
        }
        receipt["files"] = files
        receipt["shardBytes"] = officialShardBytes

        // Independently hash the documented manifest-hash prefix and sorted
        // filename<TAB>size<TAB>digest<LF> records for the adjusted metadata.
        var aggregate = SHA256()
        aggregate.update(data: Data(
            (OfficialQwenPayloadVerifier.checksumManifestSHA256 + "\n").utf8))
        for shard in shards {
            let file = files.first { ($0["filename"] as? String) == shard.filename }!
            let digest = file["sha256"] as! String
            let fingerprint = file["fingerprint"] as! [String: Any]
            let size = (fingerprint["size"] as! NSNumber).uint64Value
            aggregate.update(data: Data("\(shard.filename)\t\(size)\t\(digest)\n".utf8))
        }
        receipt["shardSetSHA256"] = aggregate.finalize()
            .map { String(format: "%02x", $0) }.joined()

        let bytes = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        #expect(UInt64(bytes.count) <= OfficialSourceTrust.maximumReceiptBytes)
        try fixture.installReceipt(bytes)

        let probe = try OfficialSourceTrust.probe(at: fixture.registration.modelDirectory)
        #expect(probe.receipt == .presentUntrusted)
        #expect(try fixture.totalShardBytes() < officialShardBytes)
        expectTrustFailure({
            _ = try OfficialSourceTrust.verify(
                at: fixture.registration.modelDirectory,
                policy: .sizeCheckTrustedReceipt)
        }, expected: .stale)
    }
}
