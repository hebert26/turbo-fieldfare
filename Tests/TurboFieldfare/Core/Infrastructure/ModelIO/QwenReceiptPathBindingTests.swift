import CryptoKit
import Darwin
import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareRepackCore

private let qwenRepository = "Qwen/Qwen3.6-35B-A3B"
private let qwenRevision = "995ad96eacd98c81ed38be0c5b274b04031597b0"

private struct ReceiptPathFixture {
    let root: URL

    init(_ name: String) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-receipt-\(name)-\(UUID().uuidString)",
                                   isDirectory: true)
        try FileManager.default.createDirectory(at: root,
                                                withIntermediateDirectories: true)
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}

private func realpath(_ path: String) throws -> String {
    guard let resolved = path.withCString({ Darwin.realpath($0, nil) }) else {
        throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: path])
    }
    defer { free(resolved) }
    return String(cString: resolved)
}

private func expectedPhysicalPath(for directory: URL) throws -> String {
    let path = directory.path
    if FileManager.default.fileExists(atPath: path) {
        return try realpath(path)
    }
    let parent = directory.deletingLastPathComponent()
    return (try realpath(parent.path)) + "/" + directory.lastPathComponent
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func conversionProvenance() -> VerifiedInstallReceiptWriter.ConversionProvenance {
    VerifiedInstallReceiptWriter.ConversionProvenance(
        sourceIndexSHA256: String(repeating: "1", count: 64),
        sourcePayloadSHA256: String(repeating: "2", count: 64),
        planFingerprint: String(repeating: "3", count: 64),
        quantizationPolicySHA256: String(repeating: "4", count: 64),
        converterVersion: "fixture-converter",
        compatibleTextManifestSHA256: nil)
}

private func outputFiles(_ payload: Data) -> [RepackAudit.OutputFile] {
    [RepackAudit.OutputFile(relativePath: "model_weights.bin",
                            size: UInt64(payload.count),
                            sha256: sha256(payload))]
}

@Suite
struct QwenReceiptPathBindingTests {
@Test("Qwen receipt binds physical path across partial rename and aliases")
func qwenReceiptBindsPhysicalPathAcrossPartialRenameAndAliases() throws {
    let fixture = try ReceiptPathFixture("rename")
    defer { fixture.remove() }

    let requestedFinal = fixture.root.appendingPathComponent("qwen.gturbo",
                                                             isDirectory: true)
    let partial = fixture.root.appendingPathComponent("qwen.gturbo.partial",
                                                     isDirectory: true)
    try FileManager.default.createDirectory(at: partial,
                                            withIntermediateDirectories: true)

    let manifestData = Data("qwen-manifest-fixture".utf8)
    let payload = Data("qwen-payload-fixture".utf8)
    let manifestSHA = sha256(manifestData)
    let expectedPath = try expectedPhysicalPath(for: requestedFinal)
    let receiptData = try VerifiedInstallReceiptWriter.encode(
        outputDir: requestedFinal.path,
        manifestSha256: manifestSHA,
        manifestSize: UInt64(manifestData.count),
        sourceRepoID: qwenRepository,
        sourceRevision: qwenRevision,
        verificationTimestamp: "fixture-time",
        conversionProvenance: conversionProvenance(),
        files: outputFiles(payload))

    #expect(!FileManager.default.fileExists(atPath: requestedFinal.path))
    try manifestData.write(to: partial.appendingPathComponent("manifest.json"))
    try payload.write(to: partial.appendingPathComponent("model_weights.bin"))
    try receiptData.write(to: partial.appendingPathComponent(
        VerifiedInstallReceiptReader.fileName))

    try FileManager.default.moveItem(at: partial, to: requestedFinal)
    let receipt = try VerifiedInstallReceiptReader.load(directoryURL: requestedFinal)
    #expect(receipt.sourceRepoID == qwenRepository)
    #expect(receipt.sourceRevision == qwenRevision)
    #expect(receipt.modelDirectoryPath == expectedPath)

    let physicalAlias = URL(fileURLWithPath: try realpath(requestedFinal.path),
                            isDirectory: true)
    try VerifiedInstallReceiptReader.validateManifestBinding(
        receipt,
        directoryURL: requestedFinal,
        manifestSha256: manifestSHA)
    try VerifiedInstallReceiptReader.validateManifestBinding(
        receipt,
        directoryURL: physicalAlias,
        manifestSha256: manifestSHA)
}

@Test("Qwen receipt rejects wrong, missing, and digest-mismatched bindings")
func qwenReceiptRejectsWrongMissingAndDigestMismatchedBindings() throws {
    let fixture = try ReceiptPathFixture("reject")
    defer { fixture.remove() }

    let requestedFinal = fixture.root.appendingPathComponent("qwen.gturbo",
                                                             isDirectory: true)
    try FileManager.default.createDirectory(at: requestedFinal,
                                            withIntermediateDirectories: true)
    let manifestData = Data("qwen-manifest-reject-fixture".utf8)
    let payload = Data("qwen-payload-reject-fixture".utf8)
    let manifestSHA = sha256(manifestData)
    let receiptData = try VerifiedInstallReceiptWriter.encode(
        outputDir: requestedFinal.path,
        manifestSha256: manifestSHA,
        manifestSize: UInt64(manifestData.count),
        sourceRepoID: qwenRepository,
        sourceRevision: qwenRevision,
        verificationTimestamp: "fixture-time",
        conversionProvenance: conversionProvenance(),
        files: outputFiles(payload))
    try receiptData.write(to: requestedFinal.appendingPathComponent(
        VerifiedInstallReceiptReader.fileName))
    let receipt = try VerifiedInstallReceiptReader.load(directoryURL: requestedFinal)

    let wrongExisting = fixture.root.appendingPathComponent("other.gturbo",
                                                             isDirectory: true)
    try FileManager.default.createDirectory(at: wrongExisting,
                                            withIntermediateDirectories: true)
    #expect(throws: ModelError.self) {
        try VerifiedInstallReceiptReader.validateManifestBinding(
            receipt,
            directoryURL: wrongExisting,
            manifestSha256: manifestSHA)
    }

    let missingStoredPath = fixture.root.appendingPathComponent("never-published.gturbo",
                                                                 isDirectory: true)
    let missingReceipt = VerifiedInstallReceipt(
        manifestSha256: receipt.manifestSha256,
        modelDirectoryPath: missingStoredPath.path,
        sourceRepoID: receipt.sourceRepoID,
        sourceRevision: receipt.sourceRevision,
        verificationTimestamp: receipt.verificationTimestamp,
        toolVersion: receipt.toolVersion,
        files: receipt.files)
    #expect(throws: ModelError.self) {
        try VerifiedInstallReceiptReader.validateManifestBinding(
            missingReceipt,
            directoryURL: requestedFinal,
            manifestSha256: manifestSHA)
    }

    #expect(throws: ModelError.self) {
        try VerifiedInstallReceiptReader.validateManifestBinding(
            receipt,
            directoryURL: requestedFinal,
            manifestSha256: String(repeating: "f", count: 64))
    }
}

@Test("Qwen-provenance receipt roundtrip binds a physical path and hashes tiny opaque files")
func qwenProvenanceReceiptRoundTripBindsPhysicalPathAndHashesTinyOpaqueFiles() throws {
    let fixture = try ReceiptPathFixture("tiny-opaque-qwen")
    defer { fixture.remove() }

    let output = fixture.root.appendingPathComponent("qwen-receipt-roundtrip-toy",
                                                      isDirectory: true)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    // These opaque bytes are not a valid packed/Gemma model; this covers only
    // receipt roundtrip, path binding and hashes of tiny files.
    let manifestData = Data("opaque tiny manifest bytes".utf8)
    let payload = Data("opaque tiny payload bytes".utf8)
    let manifestURL = output.appendingPathComponent("manifest.json")
    let payloadURL = output.appendingPathComponent("model_weights.bin")
    try manifestData.write(to: manifestURL)
    try payload.write(to: payloadURL)

    let manifestSHA = sha256(manifestData)
    let payloadSHA = sha256(payload)
    #expect(try Sha256Verifier.hashFile(at: manifestURL) == manifestSHA)
    #expect(try Sha256Verifier.hashFile(at: payloadURL) == payloadSHA)
    try Sha256Verifier.verifyFile(at: payloadURL,
                                  named: "model_weights.bin",
                                  expectedHex: payloadSHA)

    let receiptData = try VerifiedInstallReceiptWriter.encode(
        outputDir: output.path,
        manifestSha256: manifestSHA,
        manifestSize: UInt64(manifestData.count),
        sourceRepoID: qwenRepository,
        sourceRevision: qwenRevision,
        verificationTimestamp: "tiny-fixture-time",
        conversionProvenance: conversionProvenance(),
        files: outputFiles(payload))
    try receiptData.write(to: output.appendingPathComponent(
        VerifiedInstallReceiptReader.fileName))

    let receipt = try VerifiedInstallReceiptReader.load(directoryURL: output)
    #expect(receipt.sourceRepoID == qwenRepository)
    #expect(receipt.sourceRevision == qwenRevision)
    let payloadEntry = try #require(receipt.files["model_weights.bin"])
    #expect(payloadEntry.size == UInt64(payload.count))
    #expect(payloadEntry.sha256 == payloadSHA)
    try Sha256Verifier.verifyFile(at: payloadURL,
                                  named: "model_weights.bin",
                                  expectedHex: payloadEntry.sha256)
    try VerifiedInstallReceiptReader.validateManifestBinding(
        receipt,
        directoryURL: output,
        manifestSha256: manifestSHA)
}

@Test("Legacy Gemma receipt keeps standardized path and deterministic bytes")
func legacyGemmaReceiptKeepsStandardizedPathAndDeterministicBytes() throws {
    let fixture = try ReceiptPathFixture("legacy")
    defer { fixture.remove() }

    let output = fixture.root.appendingPathComponent("gemma.gturbo", isDirectory: true)
    let manifestData = Data("legacy-manifest-fixture".utf8)
    let payload = Data("legacy-payload-fixture".utf8)
    let manifestSHA = sha256(manifestData)
    let files = outputFiles(payload)

    let first = try VerifiedInstallReceiptWriter.encode(
        outputDir: output.path,
        manifestSha256: manifestSHA,
        manifestSize: UInt64(manifestData.count),
        sourceRepoID: "google/gemma-4",
        sourceRevision: "legacy-revision",
        verificationTimestamp: "fixture-time",
        files: files)
    let second = try VerifiedInstallReceiptWriter.encode(
        outputDir: output.path,
        manifestSha256: manifestSHA,
        manifestSize: UInt64(manifestData.count),
        sourceRepoID: "google/gemma-4",
        sourceRevision: "legacy-revision",
        verificationTimestamp: "fixture-time",
        files: files)

    #expect(first == second)
    let receipt = try JSONDecoder().decode(VerifiedInstallReceipt.self, from: first)
    #expect(receipt.modelDirectoryPath == output.standardizedFileURL.path)
    try VerifiedInstallReceiptReader.validateManifestBinding(
        receipt,
        directoryURL: output,
        manifestSha256: manifestSHA)
}
}
