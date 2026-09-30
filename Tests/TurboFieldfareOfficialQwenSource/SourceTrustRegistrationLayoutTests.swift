import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

@Suite struct SourceTrustRegistrationLayoutTests {
    @Test func inspectionAcceptsOnlyTheMarkerAndBoundedReceiptLayout() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let receipt = fixture.modelDirectory.appendingPathComponent(
            OfficialSourceTrust.receiptFilename)
        let deliberatelyUntrustedBytes = Data("not a verified receipt".utf8)
        try deliberatelyUntrustedBytes.write(to: receipt)

        let inspected = try OfficialSourceRegistration.inspect(at: fixture.modelDirectory)

        #expect(inspected == fixture.descriptor)
        #expect(try FileManager.default.contentsOfDirectory(
            atPath: fixture.modelDirectory.path).sorted() == [
                OfficialSourceDescriptor.markerFilename,
                OfficialSourceTrust.receiptFilename,
            ].sorted())
        #expect(try Data(contentsOf: receipt) == deliberatelyUntrustedBytes)
    }

    @Test func inspectionRejectsArbitraryEntriesBesideMarkerAndReceipt() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        try Data("untrusted receipt bytes".utf8).write(to: fixture.modelDirectory
            .appendingPathComponent(OfficialSourceTrust.receiptFilename))
        let unexpected = fixture.modelDirectory.appendingPathComponent("extra.json")
        let unexpectedBytes = Data("must be preserved".utf8)
        try unexpectedBytes.write(to: unexpected)

        expectInvalidLayout(at: fixture.modelDirectory)

        #expect(try Data(contentsOf: unexpected) == unexpectedBytes)
    }

    @Test func inspectionRejectsSymlinkReceiptWithoutFollowingIt() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let target = fixture.root.appendingPathComponent("receipt-target.json")
        try Data("outside the logical registration".utf8).write(to: target)
        let link = fixture.modelDirectory.appendingPathComponent(
            OfficialSourceTrust.receiptFilename)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        expectInvalidLayout(at: fixture.modelDirectory)

        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: link.path)
            == target.path)
        #expect(try Data(contentsOf: target) == Data("outside the logical registration".utf8))
    }

    @Test func inspectionRejectsFIFOReceiptWithoutBlocking() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let fifo = fixture.modelDirectory.appendingPathComponent(
            OfficialSourceTrust.receiptFilename)
        guard mkfifo(fifo.path, mode_t(S_IRUSR | S_IWUSR)) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }

        expectInvalidLayout(at: fixture.modelDirectory)
    }

    @Test func inspectionRejectsReceiptAboveTheMetadataBound() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let overLimit = try #require(Int(exactly: OfficialSourceTrust.maximumReceiptBytes + 1))
        let receipt = fixture.modelDirectory.appendingPathComponent(
            OfficialSourceTrust.receiptFilename)
        try Data(repeating: 0x41, count: overLimit).write(to: receipt)

        expectInvalidLayout(at: fixture.modelDirectory)
    }
}

private func expectInvalidLayout(
    at modelDirectory: URL,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    do {
        _ = try OfficialSourceRegistration.inspect(at: modelDirectory)
        Issue.record("expected invalid registration layout", sourceLocation: sourceLocation)
    } catch let error as OfficialSourceRegistration.RegistrationError {
        guard case .invalidLayout = error else {
            Issue.record("expected invalidLayout, got \(error)", sourceLocation: sourceLocation)
            return
        }
    } catch {
        Issue.record("unexpected inspection error: \(error)", sourceLocation: sourceLocation)
    }
}
