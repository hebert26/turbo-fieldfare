import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Adapter-only checks over the existing tiny synthetic registration fixture.
/// The fixture is marker-only (no receipt), and these tests do not load a model.
@Suite(.serialized)
struct GTurboModelDirectoryOfficialSourceTests {
    @Test func protectedModeUsesRegisteredNamesAndReadsOnlySyntheticMetadata() throws {
        let fixture = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: false)
        defer { fixture.remove() }
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let directory = try GTurboModelDirectory(protectedOfficialSource: handle)
        let configURL = fixture.sourceRoot.appendingPathComponent("config.json")
        let configBytes = try Data(contentsOf: configURL)
        let identity = OfficialQwenSourceIdentity.pinned
        let allowedNames = Set(identity.sidecarSHA256.keys)
            .union(identity.shards.map(\.filename))

        #expect(directory.rootURL.path == fixture.sourceRoot.path)
        #expect(try directory.fileSize("config.json") == UInt64(configBytes.count))
        #expect(try directory.readMetadata("config.json", maxBytes: UInt64(configBytes.count))
            == configBytes)
        #expect(try directory.basenames() == allowedNames)
        #expect(!FileManager.default.fileExists(atPath: fixture.modelDirectory
            .appendingPathComponent(OfficialSourceTrust.receiptFilename).path))
        #expect(throws: ModelError.self) {
            _ = try directory.fileSize("not-registered.txt")
        }
    }

    @Test func protectedMetadataRejectsAFileDescriptorForAnotherFile() throws {
        let fixture = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: false)
        defer { fixture.remove() }
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let directory = try GTurboModelDirectory(protectedOfficialSource: handle)
        let configURL = fixture.sourceRoot.appendingPathComponent("config.json")
        let configBytes = try Data(contentsOf: configURL)
        let unrelatedURL = fixture.root.appendingPathComponent("unrelated-config.json")
        try configBytes.write(to: unrelatedURL)
        let unrelatedFD = open(unrelatedURL.path, O_RDONLY | O_CLOEXEC)
        #expect(unrelatedFD >= 0)
        guard unrelatedFD >= 0 else { return }
        defer { close(unrelatedFD) }

        #expect(throws: ModelError.self) {
            _ = try directory.readMetadata(fileDescriptor: unrelatedFD,
                                           relativePath: "config.json",
                                           maxBytes: UInt64(configBytes.count))
        }
    }

    @Test func protectedAdapterRechecksRegistrationBinding() throws {
        let fixture = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: false)
        defer { fixture.remove() }
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let directory = try GTurboModelDirectory(protectedOfficialSource: handle)
        let markerURL = fixture.modelDirectory.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename)
        try Data("replaced synthetic marker".utf8).write(to: markerURL, options: .atomic)

        #expect(throws: ModelError.self) {
            _ = try directory.fileSize("config.json")
        }
        #expect(throws: ModelError.self) {
            _ = try directory.basenames()
        }
    }

    @Test func legacyRootModeRemainsCompatibleForSyntheticPackedFiles() throws {
        let fixture = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: false)
        defer { fixture.remove() }
        let legacyRoot = fixture.root.appendingPathComponent("legacy-packed.gturbo",
                                                              isDirectory: true)
        try FileManager.default.createDirectory(at: legacyRoot, withIntermediateDirectories: false)
        let marker = Data("synthetic legacy manifest".utf8)
        try marker.write(to: legacyRoot.appendingPathComponent("manifest.json"))
        let directory = try GTurboModelDirectory(rootURL: legacyRoot)

        #expect(try directory.fileSize("manifest.json") == UInt64(marker.count))
        #expect(try directory.readMetadata("manifest.json", maxBytes: UInt64(marker.count)) == marker)
        #expect(try directory.basenames() == Set(["manifest.json"]))
    }
}
