import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Exercises descriptor-protected path admission only. All source entries are
/// tiny synthetic files; these tests neither verify receipt trust nor load a model.
@Suite(.serialized)
struct OfficialSourceHandleTests {
    @Test func opensPinnedSyntheticAllowlistThroughProtectedRoot() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        try writeSyntheticFile("config.json", bytes: "synthetic-config", under: fixture.sourceRoot)
        try writeSyntheticFile("not-in-descriptor.txt", bytes: "synthetic-unlisted", under: fixture.sourceRoot)

        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        #expect(handle.sourceRootURL.path == fixture.sourceRoot.path)
        try handle.validateBinding()
        #expect(try handle.basenames() == Set(["config.json"]))

        let fd = try handle.openFile("config.json")
        var info = stat()
        #expect(fstat(fd, &info) == 0)
        #expect((info.st_mode & S_IFMT) == S_IFREG)
        #expect(info.st_size == off_t("synthetic-config".utf8.count))
        #expect(close(fd) == 0)
    }

    @Test func rejectsEscapingAbsoluteMalformedAndUnlistedNames() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let unsafeNames = [
            "", ".", "..", "../config.json", "../../outside", "/tmp/config.json",
            "nested/config.json", "config.json/../outside", "config\\name.json",
            "config.json\u{0}suffix", "unlisted-file.txt",
        ]

        for name in unsafeNames {
            expectNotAllowed {
                let fd = try handle.openFile(name)
                close(fd)
            }
        }
    }

    @Test func rejectsSymlinkedSourceRoot() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let realRoot = fixture.root.appendingPathComponent("real-source-root", isDirectory: true)
        try FileManager.default.createDirectory(at: realRoot, withIntermediateDirectories: false)
        let rootLink = fixture.root.appendingPathComponent("source-root-link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: rootLink, withDestinationURL: realRoot)
        let registration = try makeSyntheticRegistration(
            basedOn: fixture.descriptor, sourceRoot: rootLink, under: fixture.root,
            name: "symlinked-root")

        expectInvalidPath { _ = try OfficialSourceHandle(registrationURL: registration) }
    }

    @Test func rejectsSymlinkedIntermediateSourceAncestor() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let realParent = fixture.root.appendingPathComponent("real-parent", isDirectory: true)
        try FileManager.default.createDirectory(at: realParent, withIntermediateDirectories: false)
        let realRoot = realParent.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: realRoot, withIntermediateDirectories: false)
        let parentLink = fixture.root.appendingPathComponent("parent-link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: parentLink, withDestinationURL: realParent)
        let linkedRoot = parentLink.appendingPathComponent("source", isDirectory: true)
        let registration = try makeSyntheticRegistration(
            basedOn: fixture.descriptor, sourceRoot: linkedRoot, under: fixture.root,
            name: "symlinked-intermediate")

        expectInvalidPath { _ = try OfficialSourceHandle(registrationURL: registration) }
    }

    @Test func rejectsSymlinkedAllowlistedLeaf() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let outside = fixture.root.appendingPathComponent("outside-config")
        try Data("synthetic-outside".utf8).write(to: outside)
        let leaf = fixture.sourceRoot.appendingPathComponent("config.json")
        try FileManager.default.createSymbolicLink(at: leaf, withDestinationURL: outside)
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)

        expectInvalidPath {
            let fd = try handle.openFile("config.json")
            close(fd)
        }
    }

    @Test func rejectsFIFOsAndDirectoriesWithoutBlocking() throws {
        let fifoFixture = try Task68SyntheticRegistrationFixture.make()
        defer { fifoFixture.remove() }
        let fifo = fifoFixture.sourceRoot.appendingPathComponent("config.json")
        #expect(mkfifo(fifo.path, mode_t(0o600)) == 0)
        let fifoHandle = try OfficialSourceHandle(registrationURL: fifoFixture.modelDirectory)
        expectNotRegular { try fifoHandle.openFile("config.json") }

        let directoryFixture = try Task68SyntheticRegistrationFixture.make()
        defer { directoryFixture.remove() }
        let directoryLeaf = directoryFixture.sourceRoot.appendingPathComponent("config.json",
                                                                                isDirectory: true)
        try FileManager.default.createDirectory(at: directoryLeaf, withIntermediateDirectories: false)
        let directoryHandle = try OfficialSourceHandle(registrationURL: directoryFixture.modelDirectory)
        expectNotRegular { try directoryHandle.openFile("config.json") }
    }

    @Test func rejectsRootReplacementAfterConstruction() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let movedRoot = fixture.root.appendingPathComponent("retained-source-root", isDirectory: true)
        try FileManager.default.moveItem(at: fixture.sourceRoot, to: movedRoot)
        try FileManager.default.createDirectory(at: fixture.sourceRoot, withIntermediateDirectories: false)

        expectReplaced { try handle.validateBinding() }
    }

    @Test func rejectsSameNameFileReplacementOnRepeatedOpen() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let leaf = try writeSyntheticFile("config.json", bytes: "synthetic-A", under: fixture.sourceRoot)
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let firstFD = try handle.openFile("config.json")
        #expect(close(firstFD) == 0)

        let movedLeaf = fixture.sourceRoot.appendingPathComponent("previous-config")
        try FileManager.default.moveItem(at: leaf, to: movedLeaf)
        try Data("synthetic-B".utf8).write(to: leaf)

        expectReplaced {
            let fd = try handle.openFile("config.json")
            close(fd)
        }
    }

    @Test func rejectsFileReplacementDuringOpenAndClosesTransientDescriptor() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let leaf = try writeSyntheticFile("config.json", bytes: "synthetic-A", under: fixture.sourceRoot)
        let movedLeaf = fixture.sourceRoot.appendingPathComponent("previous-config")
        let handle = try OfficialSourceHandle(
            registrationURL: fixture.modelDirectory,
            checkpoint: { point in
                guard case .fileOpened = point else { return }
                try FileManager.default.moveItem(at: leaf, to: movedLeaf)
                try Data("synthetic-B".utf8).write(to: leaf)
            })
        let descriptorsWithHandle = openDescriptorCount()

        expectReplaced {
            let fd = try handle.openFile("config.json")
            close(fd)
        }
        #expect(openDescriptorCount() == descriptorsWithHandle)
    }

    @Test func rejectsSameNameFileReplacementDuringEnumeration() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let leaf = try writeSyntheticFile("config.json", bytes: "synthetic-A", under: fixture.sourceRoot)
        let movedLeaf = fixture.sourceRoot.appendingPathComponent("previous-config")
        let handle = try OfficialSourceHandle(
            registrationURL: fixture.modelDirectory,
            checkpoint: { point in
                guard case .beforeEnumerationReturn = point else { return }
                try FileManager.default.moveItem(at: leaf, to: movedLeaf)
                try Data("synthetic-B".utf8).write(to: leaf)
            })

        expectReplaced { _ = try handle.basenames() }
    }

    @Test func rejectsRootReplacementDuringInitializationAndClosesDescriptors() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let movedRoot = fixture.root.appendingPathComponent("retained-source-root", isDirectory: true)
        let descriptorsBefore = openDescriptorCount()

        expectReplaced {
            _ = try OfficialSourceHandle(
                registrationURL: fixture.modelDirectory,
                checkpoint: { point in
                    guard case .sourceOpened = point else { return }
                    try FileManager.default.moveItem(at: fixture.sourceRoot, to: movedRoot)
                    try FileManager.default.createDirectory(at: fixture.sourceRoot,
                                                            withIntermediateDirectories: false)
                })
        }
        #expect(openDescriptorCount() == descriptorsBefore)
    }

    @Test func rejectsRegisteredMarkerReplacementAfterConstruction() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let alternateRoot = fixture.root.appendingPathComponent("alternate-source", isDirectory: true)
        try FileManager.default.createDirectory(at: alternateRoot, withIntermediateDirectories: false)
        let changedDescriptor = try descriptor(fixture.descriptor, sourceRoot: alternateRoot)
        let changedMarker = try JSONEncoder().encode(changedDescriptor)
        let markerURL = fixture.modelDirectory.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename)
        try changedMarker.write(to: markerURL, options: .atomic)

        expectReplaced { try handle.validateBinding() }
    }

    @Test func rejectsRegisteredDirectoryReplacementAfterConstruction() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
        let markerURL = fixture.modelDirectory.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename)
        let markerBytes = try Data(contentsOf: markerURL)
        let displaced = fixture.root.appendingPathComponent("displaced-registration",
                                                              isDirectory: true)
        try FileManager.default.moveItem(at: fixture.modelDirectory, to: displaced)
        try FileManager.default.createDirectory(at: fixture.modelDirectory,
                                                withIntermediateDirectories: false)
        try markerBytes.write(to: fixture.modelDirectory.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename))

        expectReplaced { try handle.validateBinding() }
    }

    @Test func closesCallerOwnedDescriptorsOnSuccessAndAllRetainedDescriptorsOnDeinit() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        try writeSyntheticFile("config.json", bytes: "synthetic-config", under: fixture.sourceRoot)
        let descriptorsBefore = openDescriptorCount()
        do {
            let handle = try OfficialSourceHandle(registrationURL: fixture.modelDirectory)
            let descriptorsWithHandle = openDescriptorCount()
            #expect(descriptorsWithHandle == descriptorsBefore + 2)
            let fd = try handle.openFile("config.json")
            #expect(openDescriptorCount() == descriptorsWithHandle + 1)
            #expect(close(fd) == 0)
            #expect(openDescriptorCount() == descriptorsWithHandle)
            withExtendedLifetime(handle) {}
        }
        #expect(openDescriptorCount() == descriptorsBefore)
    }

    @Test func closesDescriptorsWhenInitializationFailsAfterOpeningBothRoots() throws {
        let fixture = try Task68SyntheticRegistrationFixture.make()
        defer { fixture.remove() }
        let descriptorsBefore = openDescriptorCount()
        expectInjectedInterruption {
            _ = try OfficialSourceHandle(
                registrationURL: fixture.modelDirectory,
                checkpoint: { point in
                    if case .sourceOpened = point { throw HandleTestInterruption.stop }
                })
        }
        #expect(openDescriptorCount() == descriptorsBefore)
    }
}

private enum HandleTestInterruption: Error {
    case stop
}

private func makeSyntheticRegistration(
    basedOn original: OfficialSourceDescriptor,
    sourceRoot: URL,
    under root: URL,
    name: String
) throws -> URL {
    let registrationParent = root.appendingPathComponent("synthetic-registrations", isDirectory: true)
    try FileManager.default.createDirectory(at: registrationParent, withIntermediateDirectories: true)
    let registration = registrationParent.appendingPathComponent(name, isDirectory: true)
    try FileManager.default.createDirectory(at: registration, withIntermediateDirectories: false)
    let marker = try JSONEncoder().encode(descriptor(original, sourceRoot: sourceRoot))
    try marker.write(to: registration.appendingPathComponent(OfficialSourceDescriptor.markerFilename))
    return registration
}

private func descriptor(_ original: OfficialSourceDescriptor,
                        sourceRoot: URL) throws -> OfficialSourceDescriptor {
    try OfficialSourceDescriptor(
        repository: original.repository,
        revision: original.revision,
        storageProfile: original.storageProfile,
        sidecarSHA256: original.sidecarSHA256,
        shards: original.shards,
        sourceRoot: sourceRoot.path)
}

@discardableResult
private func writeSyntheticFile(_ name: String, bytes: String, under root: URL) throws -> URL {
    let url = root.appendingPathComponent(name)
    try Data(bytes.utf8).write(to: url)
    return url
}

private func openDescriptorCount() -> Int {
    (0..<Int(getdtablesize())).reduce(into: 0) { count, index in
        if fcntl(Int32(index), F_GETFD) != -1 { count += 1 }
    }
}

private func expectNotAllowed(_ operation: () throws -> Void) {
    do {
        try operation()
        Issue.record("Expected an unsafe or unallowlisted path to be rejected")
    } catch let error as OfficialSourceHandleError {
        if case .notAllowed = error { return }
        Issue.record("Expected .notAllowed, got \(error)")
    } catch {
        Issue.record("Expected OfficialSourceHandleError.notAllowed, got \(error)")
    }
}

private func expectInvalidPath(_ operation: () throws -> Void) {
    do {
        try operation()
        Issue.record("Expected a symlinked path to be rejected")
    } catch let error as OfficialSourceHandleError {
        if case .invalidPath = error { return }
        Issue.record("Expected .invalidPath, got \(error)")
    } catch {
        Issue.record("Expected OfficialSourceHandleError.invalidPath, got \(error)")
    }
}

private func expectNotRegular(_ operation: () throws -> Int32) {
    do {
        let fd = try operation()
        close(fd)
        Issue.record("Expected a nonregular source entry to be rejected")
    } catch let error as OfficialSourceHandleError {
        if case .notRegular = error { return }
        Issue.record("Expected .notRegular, got \(error)")
    } catch {
        Issue.record("Expected OfficialSourceHandleError.notRegular, got \(error)")
    }
}

private func expectReplaced(_ operation: () throws -> Void) {
    do {
        try operation()
        Issue.record("Expected a replaced source or registration entry to be rejected")
    } catch let error as OfficialSourceHandleError {
        if case .replaced = error { return }
        Issue.record("Expected .replaced, got \(error)")
    } catch {
        Issue.record("Expected OfficialSourceHandleError.replaced, got \(error)")
    }
}

private func expectInjectedInterruption(_ operation: () throws -> Void) {
    do {
        try operation()
        Issue.record("Expected the injected initializer interruption")
    } catch is HandleTestInterruption {
        // The injected checkpoint runs after both local root descriptors are open.
    } catch {
        Issue.record("Expected HandleTestInterruption, got \(error)")
    }
}
