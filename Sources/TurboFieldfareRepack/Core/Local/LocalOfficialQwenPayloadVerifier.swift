import Foundation
import TurboFieldfareOfficialQwenSource

struct LocalOfficialQwenPayloadIdentity: Sendable, Equatable {
    let checksumManifestSHA256: String
    let shardSetSHA256: String
    let shardBytes: UInt64
    let shardCount: Int

    init(_ shared: OfficialQwenPayloadIdentity) {
        checksumManifestSHA256 = shared.checksumManifestSHA256
        shardSetSHA256 = shared.shardSetSHA256
        shardBytes = shared.shardBytes
        shardCount = shared.shardCount
    }

    // Preserve existing synthetic workflow fixture construction.
    init(checksumManifestSHA256: String, shardSetSHA256: String,
         shardBytes: UInt64, shardCount: Int) {
        self.checksumManifestSHA256 = checksumManifestSHA256
        self.shardSetSHA256 = shardSetSHA256
        self.shardBytes = shardBytes
        self.shardCount = shardCount
    }
}

/// Compatibility seam for existing synthetic callbacks. Normal RepackCore
/// calls use the shared public production verifier, never callback digests.
struct LocalOfficialQwenPayloadVerifierOperations {
    var readChecksumManifest: (_ path: String, _ maximumBytes: UInt64) throws -> Data {
        didSet { isProduction = false }
    }
    var inspectAndHashFile: (
        _ path: String,
        _ progress: @escaping (UInt64) throws -> Void
    ) throws -> (size: UInt64, sha256: String) {
        didSet { isProduction = false }
    }
    var cancellationCheck: () throws -> Void {
        didSet { isProduction = false }
    }
    private(set) var isProduction = false

    init(
        readChecksumManifest: @escaping (String, UInt64) throws -> Data,
        inspectAndHashFile: @escaping (
            String, @escaping (UInt64) throws -> Void
        ) throws -> (size: UInt64, sha256: String),
        cancellationCheck: @escaping () throws -> Void
    ) {
        self.readChecksumManifest = readChecksumManifest
        self.inspectAndHashFile = inspectAndHashFile
        self.cancellationCheck = cancellationCheck
    }

    static var production: Self {
        var result = Self(
            readChecksumManifest: { path, maximumBytes in
                try OfficialQwenPayloadVerifier.readChecksumManifest(
                    path: path, maximumBytes: maximumBytes)
            },
            inspectAndHashFile: { path, progress in
                try OfficialQwenPayloadVerifier.hashRegularFile(path: path, progress: progress)
            },
            cancellationCheck: { try Task.checkCancellation() })
        result.isProduction = true
        return result
    }
}

enum LocalOfficialQwenPayloadVerifier {
    static let checksumManifestFile = OfficialQwenPayloadVerifier.checksumManifestFile
    static let checksumManifestBytes = OfficialQwenPayloadVerifier.checksumManifestBytes
    static let checksumManifestSHA256 = OfficialQwenPayloadVerifier.checksumManifestSHA256
    static let expectedShardBytes = OfficialQwenPayloadVerifier.expectedShardBytes
    static let expectedShardCount = OfficialQwenPayloadVerifier.expectedShardCount
    static let expectedLineCount = OfficialQwenPayloadVerifier.expectedLineCount

    static func verify(
        snapshotDirectory: String,
        shardFilenames: [String],
        operations: LocalOfficialQwenPayloadVerifierOperations = .production,
        progress: @escaping (_ completedBytes: UInt64, _ totalBytes: UInt64) -> Void = { _, _ in }
    ) throws -> LocalOfficialQwenPayloadIdentity {
        do {
            let result: OfficialQwenPayloadIdentity
            if operations.isProduction {
                result = try OfficialQwenPayloadVerifier.verify(
                    snapshotDirectory: snapshotDirectory,
                    shardFilenames: shardFilenames,
                    progress: progress)
            } else {
                let sharedOperations = OfficialQwenPayloadVerifierOperations(
                    readChecksumManifest: operations.readChecksumManifest,
                    inspectAndHashFile: operations.inspectAndHashFile,
                    cancellationCheck: operations.cancellationCheck)
                result = try OfficialQwenPayloadVerifier.verify(
                    snapshotDirectory: snapshotDirectory,
                    shardFilenames: shardFilenames,
                    operations: sharedOperations,
                    progress: progress)
            }
            return LocalOfficialQwenPayloadIdentity(result)
        } catch let error as OfficialQwenPayloadVerificationError {
            throw mapError(error)
        }
    }

    private static func mapError(_ error: OfficialQwenPayloadVerificationError) -> RepackError {
        switch error {
        case .fileOpenFailed(let path, let code):
            .fileOpenFailed(path: path, errno: code)
        case .fileStatFailed(let path, let code):
            .fileStatFailed(path: path, errno: code)
        case .preadShort(let path, let expected, let got, let code):
            .preadShort(path: path, expected: expected, got: got, errno: code)
        case .sourceFingerprintRejected(let path, let sha256):
            .sourceFingerprintRejected(path: path, sha256: sha256)
        case .installStateCorrupt(let path, let detail):
            .installStateCorrupt(path: path, detail: detail)
        case .installStateIncompatible(let detail):
            .installStateIncompatible(detail: detail)
        case .configurationInvalid(let detail):
            .configurationInvalid(detail: detail)
        }
    }
}
