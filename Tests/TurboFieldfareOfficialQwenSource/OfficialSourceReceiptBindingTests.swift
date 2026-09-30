import Foundation
import Testing
@testable import TurboFieldfareOfficialQwenSource

@Suite struct OfficialSourceReceiptBindingTests {
    @Test func unchangedSyntheticReceiptBindsToProtectedHandle() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let receipt = try verifiedSyntheticReceipt(for: fixture)
        let handle = try OfficialSourceHandle(registrationURL: fixture.logicalModelURL)

        try handle.validateTrustedReceipt(receipt)

        #expect(receipt.files.count == 39)
    }

    @Test func rejectsShardMutationBetweenReceiptVerificationAndHandleAcquisition() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let receipt = try verifiedSyntheticReceipt(for: fixture)
        try mutateFirstByteInPlace(at: fixture.firstShardURL)
        let handle = try OfficialSourceHandle(registrationURL: fixture.logicalModelURL)

        #expect(throws: OfficialSourceHandleError.self) {
            try handle.validateTrustedReceipt(receipt)
        }
    }

    @Test func rejectsSidecarMutationBetweenReceiptVerificationAndHandleAcquisition() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let receipt = try verifiedSyntheticReceipt(for: fixture)
        try mutateFirstByteInPlace(at: fixture.sidecarURL)
        let handle = try OfficialSourceHandle(registrationURL: fixture.logicalModelURL)

        #expect(throws: OfficialSourceHandleError.self) {
            try handle.validateTrustedReceipt(receipt)
        }
    }

    @Test func rejectsSourceRootReplacementBetweenReceiptVerificationAndHandleAcquisition() throws {
        let fixture = try Task68SyntheticTrustFixture.make()
        defer { fixture.remove() }
        let receipt = try verifiedSyntheticReceipt(for: fixture)
        let fileManager = FileManager.default
        let sourceRoot = fixture.sourceRootURL
        let originalRoot = fixture.registration.root.appendingPathComponent(
            "receipt-binding-original-\(UUID().uuidString)", isDirectory: true)
        try fileManager.moveItem(at: sourceRoot, to: originalRoot)
        defer {
            if fileManager.fileExists(atPath: sourceRoot.path) {
                try? fileManager.removeItem(at: sourceRoot)
            }
            if fileManager.fileExists(atPath: originalRoot.path) {
                try? fileManager.moveItem(at: originalRoot, to: sourceRoot)
            }
        }
        try fileManager.createDirectory(at: sourceRoot, withIntermediateDirectories: false)
        let handle = try OfficialSourceHandle(registrationURL: fixture.logicalModelURL)

        #expect(throws: OfficialSourceHandleError.self) {
            try handle.validateTrustedReceipt(receipt)
        }
    }
}

private func verifiedSyntheticReceipt(
    for fixture: Task68SyntheticTrustFixture
) throws -> OfficialSourceTrustReceipt {
    try OfficialSourceTrust.verifySynthetic(
        at: fixture.logicalModelURL,
        policy: .fullSha256,
        expectedShardBytes: fixture.totalShardBytes(),
        fullVerification: {})
}

private func mutateFirstByteInPlace(at url: URL) throws {
    var bytes = try Data(contentsOf: url)
    guard !bytes.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
    bytes[bytes.startIndex] ^= 0x01
    let file = try FileHandle(forWritingTo: url)
    defer { try? file.close() }
    try file.seek(toOffset: 0)
    try file.write(contentsOf: bytes)
    try file.truncate(atOffset: UInt64(bytes.count))
    try file.synchronize()
}
