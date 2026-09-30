import CryptoKit
import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Metadata-only registration tests. Every source tree is tiny and temporary;
/// the literal identity values below are copied from the pinned SHA256SUMS
/// metadata, not obtained from the production pin API. No official shard is read.
@Suite struct SourceRegistrationTests {
    @Test func registersPinnedMetadataAndLeavesSourcePayloadsUntouched() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let logicalParent = try RegistrationTestTree.makeDirectory("models", under: root)
        let logicalURL = logicalParent.appendingPathComponent("qwen.gturbo", isDirectory: true)
        let syntheticShard = sourceRoot.appendingPathComponent(
            RegistrationFixture.shards[0].filename)
        try Data("synthetic-not-weight-data".utf8).write(to: syntheticShard)
        let sourceBefore = try RegistrationTestTree.regularFiles(under: sourceRoot)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        #expect(RegistrationFixture.independentDigest() == RegistrationFixture.contentDigest)
        let registered = try OfficialSourceRegistration.register(
            markerData: marker, at: logicalURL)
        let inspected = try OfficialSourceRegistration.inspect(at: logicalURL)

        #expect(registered == inspected)
        #expect(registered.sourceRoot == sourceRoot.path)
        #expect(registered.contentSHA256 == RegistrationFixture.contentDigest)
        #expect(try RegistrationTestTree.regularFiles(under: sourceRoot) == sourceBefore)
        #expect(try FileManager.default.contentsOfDirectory(atPath: logicalURL.path)
            == [OfficialSourceDescriptor.markerFilename])
        let persistedMarker = try Data(contentsOf: logicalURL.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename))
        #expect(try OfficialSourceDescriptor.decodeStrict(data: persistedMarker) == registered)

        let persisted = try JSONSerialization.jsonObject(with: persistedMarker)
        let persistedObject = try #require(persisted as? [String: Any])
        #expect(Set(persistedObject.keys) == Set([
            "kind", "version", "repository", "revision", "storageProfile",
            "sidecarSHA256", "shards", "sourceRoot", "contentSHA256",
        ]))
        #expect(!FileManager.default.fileExists(atPath:
            logicalURL.appendingPathComponent("manifest.json").path))
    }

    @Test func sourceRootChangesDoNotChangePinnedContentIdentity() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceA = try RegistrationTestTree.makeDirectory("source-a", under: root)
        let sourceB = try RegistrationTestTree.makeDirectory("source-b", under: root)
        let logicalParent = try RegistrationTestTree.makeDirectory("models", under: root)
        let logicalA = logicalParent.appendingPathComponent("a.gturbo", isDirectory: true)
        let logicalB = logicalParent.appendingPathComponent("b.gturbo", isDirectory: true)

        let first = try OfficialSourceRegistration.register(
            markerData: RegistrationFixture.markerData(sourceRoot: sourceA.path),
            at: logicalA)
        let second = try OfficialSourceRegistration.register(
            markerData: RegistrationFixture.markerData(sourceRoot: sourceB.path),
            at: logicalB)

        #expect(first.sourceRoot != second.sourceRoot)
        #expect(first.contentSHA256 == RegistrationFixture.contentDigest)
        #expect(second.contentSHA256 == RegistrationFixture.contentDigest)
        #expect(first.contentSHA256 == second.contentSHA256)
    }

    @Test func rejectsStrictJSONFailuresBeforeCreatingDestination() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let logicalParent = try RegistrationTestTree.makeDirectory("models", under: root)
        let valid = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)
        let validText = try #require(String(data: valid, encoding: .utf8))
        let escapedRootKey = "\"source" + String(UnicodeScalar(92)!) + "u0052oot\""
        let escapedDuplicateAfter = String(validText.dropLast())
            + ",\(escapedRootKey):\(RegistrationFixture.jsonString(sourceRoot.path))}"
        let escapedDuplicateBefore = "{\(escapedRootKey):\(RegistrationFixture.jsonString(sourceRoot.path)),"
            + String(validText.dropFirst())
        var unknownObject = try #require(
            JSONSerialization.jsonObject(with: valid) as? [String: Any])
        unknownObject["verifiedReceipt"] = true
        let unknownField = try JSONSerialization.data(
            withJSONObject: unknownObject, options: [.sortedKeys])
        let candidates = [
            Data("{not-json".utf8),
            unknownField,
            Data(escapedDuplicateAfter.utf8),
            Data(escapedDuplicateBefore.utf8),
        ]

        for (index, marker) in candidates.enumerated() {
            let logicalURL = logicalParent.appendingPathComponent(
                "invalid-\(index).gturbo", isDirectory: true)
            #expect(RegistrationTestTree.registrationFails(marker, at: logicalURL))
            #expect(!FileManager.default.fileExists(atPath: logicalURL.path))
        }
    }

    @Test func rejectsChangedPinnedIdentityWithOtherwiseValidContentDigests() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let logicalParent = try RegistrationTestTree.makeDirectory("models", under: root)
        var variants: [Data] = []

        variants.append(try RegistrationFixture.markerData(
            sourceRoot: sourceRoot.path,
            repository: RegistrationFixture.repository + "-fork"))
        variants.append(try RegistrationFixture.markerData(
            sourceRoot: sourceRoot.path,
            revision: String(repeating: "0", count: 40)))
        variants.append(try RegistrationFixture.markerData(
            sourceRoot: sourceRoot.path, storageProfile: "original-fp16"))

        var changedSidecars = RegistrationFixture.sidecars
        changedSidecars["config.json"] = RegistrationFixture.changedSHA(
            try #require(changedSidecars["config.json"]))
        variants.append(try RegistrationFixture.markerData(
            sourceRoot: sourceRoot.path, sidecars: changedSidecars))

        var changedShardHash = RegistrationFixture.shards
        changedShardHash[0].sha256 = RegistrationFixture.changedSHA(changedShardHash[0].sha256)
        variants.append(try RegistrationFixture.markerData(
            sourceRoot: sourceRoot.path, shards: changedShardHash))

        var changedShardName = RegistrationFixture.shards
        changedShardName[0].filename = "renamed-\(changedShardName[0].filename)"
        variants.append(try RegistrationFixture.markerData(
            sourceRoot: sourceRoot.path, shards: changedShardName))

        for (index, marker) in variants.enumerated() {
            let logicalURL = logicalParent.appendingPathComponent(
                "mutated-\(index).gturbo", isDirectory: true)
            #expect(RegistrationTestTree.registrationFails(marker, at: logicalURL))
            #expect(!FileManager.default.fileExists(atPath: logicalURL.path))
        }
    }

    @Test func rejectsMissingSymlinkedAndNonDirectorySourceRoots() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try RegistrationTestTree.makeDirectory("source", under: root)
        let regularFile = root.appendingPathComponent("not-a-directory")
        try Data("ordinary file".utf8).write(to: regularFile)
        let sourceLink = root.appendingPathComponent("source-link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: sourceLink, withDestinationURL: source)
        let logicalParent = try RegistrationTestTree.makeDirectory("models", under: root)
        let invalidRoots = [
            root.appendingPathComponent("does-not-exist", isDirectory: true).path,
            regularFile.path,
            sourceLink.path,
            "relative/source",
            "/tmp/source/../escape",
        ]

        for (index, sourcePath) in invalidRoots.enumerated() {
            let marker = try RegistrationFixture.markerData(sourceRoot: sourcePath)
            let logicalURL = logicalParent.appendingPathComponent(
                "bad-root-\(index).gturbo", isDirectory: true)
            #expect(RegistrationTestTree.registrationFails(marker, at: logicalURL))
            #expect(!FileManager.default.fileExists(atPath: logicalURL.path))
        }
    }

    @Test func rejectsSourceAndDestinationOverlapWithoutChangingSourceInventory() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let existingChild = sourceRoot.appendingPathComponent("payload.safetensors")
        try Data("synthetic payload sentinel".utf8).write(to: existingChild)
        let before = try RegistrationTestTree.regularFiles(under: sourceRoot)
        let nestedDestination = sourceRoot.appendingPathComponent("registered.gturbo", isDirectory: true)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        #expect(RegistrationTestTree.registrationFails(marker, at: nestedDestination))
        #expect(!FileManager.default.fileExists(atPath: nestedDestination.path))
        #expect(try RegistrationTestTree.regularFiles(under: sourceRoot) == before)
    }

    @Test func rejectsEveryExistingDestinationAndPreservesItsContents() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let parent = try RegistrationTestTree.makeDirectory("models", under: root)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        let emptyDirectory = parent.appendingPathComponent("empty.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: emptyDirectory, withIntermediateDirectories: false)
        let packedDirectory = parent.appendingPathComponent("packed.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: packedDirectory, withIntermediateDirectories: false)
        let oldManifest = packedDirectory.appendingPathComponent("manifest.json")
        try Data("preserve packed install".utf8).write(to: oldManifest)
        let regularFile = parent.appendingPathComponent("regular-file.gturbo")
        try Data("preserve non-directory destination".utf8).write(to: regularFile)
        let symlinkTarget = parent.appendingPathComponent("symlink-target", isDirectory: true)
        try FileManager.default.createDirectory(at: symlinkTarget, withIntermediateDirectories: false)
        let symlink = parent.appendingPathComponent("symlink.gturbo", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: symlinkTarget)

        let oldManifestBytes = try Data(contentsOf: oldManifest)
        let regularFileBytes = try Data(contentsOf: regularFile)
        for logicalURL in [emptyDirectory, packedDirectory, regularFile, symlink] {
            #expect(RegistrationTestTree.registrationFails(marker, at: logicalURL))
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: emptyDirectory.path).isEmpty)
        #expect(try Data(contentsOf: oldManifest) == oldManifestBytes)
        #expect(try Data(contentsOf: regularFile) == regularFileBytes)
        #expect(try FileManager.default.destinationOfSymbolicLink(atPath: symlink.path)
            == symlinkTarget.path)
    }

    @Test func rejectsSymlinkedOrNonDirectoryParentsAndDoesNotFollowMarkerSymlinks() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let targetParent = try RegistrationTestTree.makeDirectory("real-parent", under: root)
        let parentLink = root.appendingPathComponent("parent-link", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: parentLink, withDestinationURL: targetParent)
        let regularParent = root.appendingPathComponent("regular-parent")
        try Data("not a directory".utf8).write(to: regularParent)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        let throughSymlink = parentLink.appendingPathComponent("new.gturbo", isDirectory: true)
        let throughFile = regularParent.appendingPathComponent("new.gturbo", isDirectory: true)
        #expect(RegistrationTestTree.registrationFails(marker, at: throughSymlink))
        #expect(RegistrationTestTree.registrationFails(marker, at: throughFile))
        #expect(try FileManager.default.contentsOfDirectory(atPath: targetParent.path).isEmpty)

        let registeredElsewhere = try RegistrationTestTree.makeDirectory("other", under: root)
        let validMarker = registeredElsewhere.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename)
        try marker.write(to: validMarker)
        let logical = try RegistrationTestTree.makeDirectory("logical", under: root)
        try FileManager.default.createSymbolicLink(
            at: logical.appendingPathComponent(OfficialSourceDescriptor.markerFilename),
            withDestinationURL: validMarker)
        #expect(RegistrationTestTree.inspectFails(at: logical))
    }

    @Test func concurrentRegistrationNeverReplacesTheWinningDescriptor() async throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceA = try RegistrationTestTree.makeDirectory("source-a", under: root)
        let sourceB = try RegistrationTestTree.makeDirectory("source-b", under: root)
        let parent = try RegistrationTestTree.makeDirectory("models", under: root)
        let logical = parent.appendingPathComponent("raced.gturbo", isDirectory: true)
        let markerA = try RegistrationFixture.markerData(sourceRoot: sourceA.path)
        let markerB = try RegistrationFixture.markerData(sourceRoot: sourceB.path)

        let successes = await withTaskGroup(of: Bool.self, returning: Int.self) { group in
            group.addTask {
                do {
                    _ = try OfficialSourceRegistration.register(markerData: markerA, at: logical)
                    return true
                } catch { return false }
            }
            group.addTask {
                do {
                    _ = try OfficialSourceRegistration.register(markerData: markerB, at: logical)
                    return true
                } catch { return false }
            }
            var count = 0
            for await succeeded in group where succeeded { count += 1 }
            return count
        }

        #expect(successes == 1)
        let winner = try OfficialSourceRegistration.inspect(at: logical)
        #expect(winner.sourceRoot == sourceA.path || winner.sourceRoot == sourceB.path)
        #expect(try FileManager.default.contentsOfDirectory(atPath: parent.path).sorted()
            == [logical.lastPathComponent, logical.lastPathComponent + ".install.lock"].sorted())
        #expect(try FileManager.default.contentsOfDirectory(atPath: logical.path)
            == [OfficialSourceDescriptor.markerFilename])
    }

    @Test func registersAtCanonicalScratchURLAndRejectsDirectSymlinkComponents() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let canonicalParent = try RegistrationTestTree.makeDirectory("canonical-models", under: root)
        let checkout = try RegistrationTestTree.makeDirectory("checkout", under: root)
        let scratchLink = checkout.appendingPathComponent("scratch", isDirectory: true)
        try FileManager.default.createSymbolicLink(
            at: scratchLink, withDestinationURL: canonicalParent)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        let unresolvedURL = scratchLink.appendingPathComponent("qwen.gturbo", isDirectory: true)
        #expect(RegistrationTestTree.registrationFails(marker, at: unresolvedURL))
        #expect(try FileManager.default.contentsOfDirectory(atPath: canonicalParent.path).isEmpty)

        // AppModelLocation resolves the checkout scratch symlink and passes this canonical URL.
        let canonicalURL = canonicalParent.appendingPathComponent("qwen.gturbo", isDirectory: true)
        let registered = try OfficialSourceRegistration.register(markerData: marker, at: canonicalURL)
        #expect(try OfficialSourceRegistration.inspect(at: canonicalURL) == registered)

        let symlinkTarget = try RegistrationTestTree.makeDirectory("target", under: root)
        let finalSymlink = canonicalParent.appendingPathComponent("linked.gturbo", isDirectory: true)
        try FileManager.default.createSymbolicLink(
            at: finalSymlink, withDestinationURL: symlinkTarget)
        #expect(RegistrationTestTree.registrationFails(marker, at: finalSymlink))
        #expect(try FileManager.default.contentsOfDirectory(atPath: symlinkTarget.path).isEmpty)
    }

    @Test func rejectsCaseAndUnicodeAliasesThatMakeSourceAndDestinationOverlap() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let aliases: [(name: String, actual: String, alternate: String)] = [
            ("case", "CaseAlias", "casealias"),
            ("unicode", "Caf\u{00e9}", "Cafe\u{0301}"),
        ]

        for alias in aliases {
            let actualParent = try RegistrationTestTree.makeDirectory(alias.actual, under: root)
            let actualSource = try RegistrationTestTree.makeDirectory("source", under: actualParent)
            let alternateParent = root.appendingPathComponent(alias.alternate, isDirectory: true)
            let alternateSource = alternateParent.appendingPathComponent("source", isDirectory: true)
            // Case-sensitive or normalization-sensitive volumes do not provide this alias.
            guard FileManager.default.fileExists(atPath: alternateSource.path) else { continue }
            let actualID = try FileManager.default.attributesOfItem(atPath: actualSource.path)[.systemFileNumber]
            let alternateID = try FileManager.default.attributesOfItem(atPath: alternateSource.path)[.systemFileNumber]
            #expect(actualID as? NSNumber == alternateID as? NSNumber)

            let destination = alternateSource.appendingPathComponent(
                "nested.gturbo", isDirectory: true)
            let marker = try RegistrationFixture.markerData(sourceRoot: actualSource.path)
            #expect(RegistrationTestTree.registrationFails(marker, at: destination))
            #expect(try FileManager.default.contentsOfDirectory(atPath: actualSource.path).isEmpty)
        }
    }

    @Test func registrationDoesNotOpenAnInaccessibleShardSentinel() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let shard = sourceRoot.appendingPathComponent(RegistrationFixture.shards[0].filename)
        let sentinelBytes = Data("synthetic inaccessible payload sentinel".utf8)
        try sentinelBytes.write(to: shard)
        let metadataBefore = try FileManager.default.attributesOfItem(atPath: shard.path)
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: shard.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: shard.path) }

        let probe = open(shard.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        let probeError = errno
        if probe >= 0 { close(probe) }
        #expect(probe == -1)
        #expect(probeError == EACCES)
        let sourceInventoryBefore = try RegistrationTestTree.regularFileNames(under: sourceRoot)
        let logicalParent = try RegistrationTestTree.makeDirectory("models", under: root)
        let logical = logicalParent.appendingPathComponent("qwen.gturbo", isDirectory: true)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        let registered = try OfficialSourceRegistration.register(markerData: marker, at: logical)
        let expected = try OfficialSourceDescriptor.decodeStrict(data: marker)
        #expect(registered == expected)
        #expect(try OfficialSourceRegistration.inspect(at: logical) == expected)

        let metadataAfter = try FileManager.default.attributesOfItem(atPath: shard.path)
        #expect(try RegistrationTestTree.regularFileNames(under: sourceRoot)
            == sourceInventoryBefore)
        #expect(try RegistrationTestTree.regularFileNames(under: logical)
            == [OfficialSourceDescriptor.markerFilename])
        #expect(metadataAfter[.systemFileNumber] as? NSNumber
            == metadataBefore[.systemFileNumber] as? NSNumber)
        #expect(metadataAfter[.size] as? NSNumber == metadataBefore[.size] as? NSNumber)
        #expect(metadataAfter[.modificationDate] as? Date
            == metadataBefore[.modificationDate] as? Date)

        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: shard.path)
        #expect(try Data(contentsOf: shard) == sentinelBytes)
    }

    @Test func cancellationAtEachPublicationCheckpointCleansOnlyOwnedStaging() throws {
        let targets = ["stageCreated", "markerSynced", "beforePublish"]
        for target in targets {
            let root = try RegistrationTestTree.makeRoot()
            defer { try? FileManager.default.removeItem(at: root) }
            let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
            let parent = try RegistrationTestTree.makeDirectory("models", under: root)
            let logical = parent.appendingPathComponent("cancelled.gturbo", isDirectory: true)
            let lockName = logical.lastPathComponent + ".install.lock"
            let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)
            var visited: [String] = []

            do {
                _ = try OfficialSourceRegistration.register(
                    markerData: marker,
                    at: logical,
                    checkpoint: { checkpoint in
                        let name: String
                        switch checkpoint {
                        case .stageCreated: name = "stageCreated"
                        case .markerSynced: name = "markerSynced"
                        case .beforePublish: name = "beforePublish"
                        }
                        visited.append(name)
                        if name == target { throw CancellationError() }
                    })
                Issue.record("registration did not stop at checkpoint \(target)")
            } catch is CancellationError {
                // The deterministic seam reports cancellation at the chosen boundary.
            } catch {
                Issue.record("unexpected error at checkpoint \(target): \(error)")
            }

            let targetIndex = try #require(targets.firstIndex(of: target))
            #expect(visited == Array(targets.prefix(targetIndex + 1)))
            #expect(!FileManager.default.fileExists(atPath: logical.path))
            #expect(try FileManager.default.contentsOfDirectory(atPath: parent.path)
                == [lockName])
        }
    }

    @Test func postCommitParentSyncFailureReportsVisibleDurabilityUnknownRegistration() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let parent = try RegistrationTestTree.makeDirectory("models", under: root)
        let logical = parent.appendingPathComponent("committed.gturbo", isDirectory: true)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        do {
            _ = try OfficialSourceRegistration.register(
                markerData: marker,
                at: logical,
                checkpoint: { _ in },
                syncParent: { _ in
                    errno = EIO
                    return -1
                })
            Issue.record("parent-sync failure was not reported")
        } catch let error as OfficialSourceRegistration.RegistrationError {
            #expect(error == .publishedDurabilityUnknown(path: logical.path, errno: EIO))
        } catch {
            Issue.record("unexpected post-commit error: \(error)")
        }

        let inspected = try OfficialSourceRegistration.inspect(at: logical)
        let expected = try OfficialSourceDescriptor.decodeStrict(data: marker)
        #expect(inspected == expected)
        let persisted = try Data(contentsOf: logical.appendingPathComponent(
            OfficialSourceDescriptor.markerFilename))
        #expect(try OfficialSourceDescriptor.decodeStrict(data: persisted) == expected)
        #expect(try FileManager.default.contentsOfDirectory(atPath: logical.path)
            == [OfficialSourceDescriptor.markerFilename])
    }

    @Test func registrationCreatesFreshApplicationSupportModelParent() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let support = try RegistrationTestTree.makeDirectory("Application Support", under: root)
        let modelParent = support.appendingPathComponent("TurboFieldfare", isDirectory: true)
        let logical = modelParent.appendingPathComponent(
            "qwen3.6-35b-a3b.gturbo", isDirectory: true)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)
        #expect(!FileManager.default.fileExists(atPath: modelParent.path))

        let registered = try OfficialSourceRegistration.register(
            markerData: marker,
            at: logical,
            checkpoint: { _ in },
            allowedApplicationSupportURL: support)

        #expect(try OfficialSourceRegistration.inspect(at: logical) == registered)
        #expect(FileManager.default.fileExists(atPath: modelParent.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: logical.path)
            == [OfficialSourceDescriptor.markerFilename])
    }

    @Test func rejectsPartialAndResumeSiblingsAndPreservesTheirContents() throws {
        for suffix in [".partial", ".resume.json"] {
            let root = try RegistrationTestTree.makeRoot()
            defer { try? FileManager.default.removeItem(at: root) }
            let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
            let parent = try RegistrationTestTree.makeDirectory("models", under: root)
            let logical = parent.appendingPathComponent("in-progress.gturbo", isDirectory: true)
            let conflict = parent.appendingPathComponent(logical.lastPathComponent + suffix)
            let sentinel = Data("preserve existing install state".utf8)
            try sentinel.write(to: conflict)
            let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

            do {
                _ = try OfficialSourceRegistration.register(markerData: marker, at: logical)
                Issue.record("registration ignored existing sibling \(suffix)")
            } catch let error as OfficialSourceRegistration.RegistrationError {
                #expect(error == .conflict(conflict.lastPathComponent))
            } catch {
                Issue.record("unexpected sibling-conflict error: \(error)")
            }

            #expect(try Data(contentsOf: conflict) == sentinel)
            #expect(!FileManager.default.fileExists(atPath: logical.path))
            #expect(try FileManager.default.contentsOfDirectory(atPath: parent.path).sorted()
                == [conflict.lastPathComponent, logical.lastPathComponent + ".install.lock"].sorted())
        }
    }

    @Test func rejectsRegistrationWhilePackedInstallerHoldsSharedSiblingLock() throws {
        let root = try RegistrationTestTree.makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceRoot = try RegistrationTestTree.makeDirectory("source", under: root)
        let parent = try RegistrationTestTree.makeDirectory("models", under: root)
        let logical = parent.appendingPathComponent("busy.gturbo", isDirectory: true)
        let lock = parent.appendingPathComponent("busy.gturbo.install.lock")
        let lockFD = open(lock.path, O_RDWR | O_CREAT | O_CLOEXEC, mode_t(0o600))
        #expect(lockFD >= 0)
        guard lockFD >= 0 else { return }
        defer { _ = flock(lockFD, LOCK_UN); close(lockFD) }
        #expect(flock(lockFD, LOCK_EX | LOCK_NB) == 0)
        let marker = try RegistrationFixture.markerData(sourceRoot: sourceRoot.path)

        do {
            _ = try OfficialSourceRegistration.register(markerData: marker, at: logical)
            Issue.record("registration ignored the packed install lock")
        } catch let error as OfficialSourceRegistration.RegistrationError {
            #expect(error == .conflict("busy.gturbo.install.lock"))
        } catch {
            Issue.record("unexpected shared-lock error: \(error)")
        }
        #expect(!FileManager.default.fileExists(atPath: logical.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: parent.path)
            == [lock.lastPathComponent])
    }

}

private enum RegistrationTestTree {
    static func makeRoot() throws -> URL {
        let temporaryPath = FileManager.default.temporaryDirectory.path
        guard let canonicalBuffer = temporaryPath.withCString({ realpath($0, nil) }) else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer { free(canonicalBuffer) }
        let canonicalTemporaryDirectory = URL(
            fileURLWithPath: String(cString: canonicalBuffer), isDirectory: true)
        let root = canonicalTemporaryDirectory.appendingPathComponent(
            "qwen-source-registration-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }

    static func makeDirectory(_ name: String, under parent: URL) throws -> URL {
        let result = parent.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: result, withIntermediateDirectories: false)
        return result
    }

    static func regularFiles(under root: URL) throws -> [String: Data] {
        let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        var files: [String: Data] = [:]
        while let url = enumerator?.nextObject() as? URL {
            let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            if values.isDirectory == true || values.isSymbolicLink == true { continue }
            let path = String(url.path.dropFirst(root.path.count + 1))
            files[path] = try Data(contentsOf: url)
        }
        return files
    }

    static func regularFileNames(under root: URL) throws -> [String] {
        let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey])
        var names: [String] = []
        while let url = enumerator?.nextObject() as? URL {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { continue }
            names.append(String(url.path.dropFirst(root.path.count + 1)))
        }
        return names.sorted()
    }

    static func registrationFails(_ marker: Data, at logicalURL: URL) -> Bool {
        do {
            _ = try OfficialSourceRegistration.register(markerData: marker, at: logicalURL)
            return false
        } catch { return true }
    }

    static func inspectFails(at logicalURL: URL) -> Bool {
        do {
            _ = try OfficialSourceRegistration.inspect(at: logicalURL)
            return false
        } catch { return true }
    }
}

private enum RegistrationFixture {
    static let repository = "Qwen/Qwen3.6-35B-A3B"
    static let revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"
    static let storageProfile = "original-bf16"
    static let contentDigest = "c1ac463726b716e7db3a9fbe5db4cb690f95be7e5779aea02e49c2020bca7a7a"

    // Independent literal fixture transcribed from the pinned SHA256SUMS file.
    static let sidecars: [String: String] = [
        "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "configuration.json": "c1b09db419119513247e9b8b912c4b9897106c9b20c6cada7e107d993c5435eb",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "model.safetensors.index.json": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    ]

    static let shards: [(filename: String, sha256: String)] = [
        ("model-00001-of-00026.safetensors", "adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478"),
        ("model-00002-of-00026.safetensors", "88f2dfd2b9e73e4b70be533dbf61bcfa3c9a0003758900fcbc9d9b96f5751d4b"),
        ("model-00003-of-00026.safetensors", "8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb"),
        ("model-00004-of-00026.safetensors", "12d7db38689ba3c8af74b23ef8523eca41e0cd95db870583d0663a3ee8a6bd60"),
        ("model-00005-of-00026.safetensors", "a836047305d0f7a7b50f0815d09d5c03ec03d59ec2c763fcdc4bf7e9936bf902"),
        ("model-00006-of-00026.safetensors", "c9080d718e9c5f9e337443225aa417d4c24d00ae7995d76ee3f1cc296b557d15"),
        ("model-00007-of-00026.safetensors", "e8c05e23131b1dd45a455ec38cfac7db14667358268623c3938d00cf3e959a68"),
        ("model-00008-of-00026.safetensors", "4b6a6d495053089f4a80e7cbc82e848fba44e2c0c60122233d8fdff79fa7b296"),
        ("model-00009-of-00026.safetensors", "a31a954bb72d1c714e751bf0aabf2ff533f5a509693ebf7dd22ad6e90be46f67"),
        ("model-00010-of-00026.safetensors", "246560e66570fe746653b8443e245dc334c9b8b831ea43d2d9f1b7d98623994e"),
        ("model-00011-of-00026.safetensors", "7180392817fe3ecb3a27a1da43b7ff22c1a94806bac49975f9f122c3126df675"),
        ("model-00012-of-00026.safetensors", "043fb525f6625c2f2acb75e65a9959ee3fa7b6e3fdd2034b5cfe1859b01d3cfb"),
        ("model-00013-of-00026.safetensors", "33a20fb20a21379bf43c84a43105f9c0cc35bd50d740b1c302dcbe4b700f5425"),
        ("model-00014-of-00026.safetensors", "be823e33c5cb6120ad3769d081f34a2449dc2358041fca7c29d636c1ba19130d"),
        ("model-00015-of-00026.safetensors", "a89d547c6f9d0b535ee5ea2f2478f163089539f3f0dd330cb23d278a19d76123"),
        ("model-00016-of-00026.safetensors", "69fc3ae0316482288afdcdd0b9eb7d626703ae26f7567e89aa3fc8d1ffd4ff5b"),
        ("model-00017-of-00026.safetensors", "e356e3943cf3852b76bb8992e674f3256013e27d54b78e8250514151cdc29637"),
        ("model-00018-of-00026.safetensors", "9e5e63fd1cc7d6848330c1fa363dfcb661bbc2ac87e672d0e28b71c9cb7f3c7f"),
        ("model-00019-of-00026.safetensors", "708644ad34f1de727bf484f396944d8ec628645d52c183e9a992e65671685e21"),
        ("model-00020-of-00026.safetensors", "ca083a1d1aa64f8e8a785998f543a43374f13436dc85d396eee4e72c7a84e1ae"),
        ("model-00021-of-00026.safetensors", "ada4ae48f3d48fe01b4c53f2f82bce25e798a9631fd33959c881156fef2ccbce"),
        ("model-00022-of-00026.safetensors", "def207fb42d7db31efb512755557763c23233c6e4d4c433027cb5102a7bce2f7"),
        ("model-00023-of-00026.safetensors", "864d52ca7768a36f514069222e8de8626264ae124097ba8fcce5b5da2c6e2ed7"),
        ("model-00024-of-00026.safetensors", "391acd27420cdce5935ff18152423c70620d19dac3c39a5ef1a81d369f82d737"),
        ("model-00025-of-00026.safetensors", "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154"),
        ("model-00026-of-00026.safetensors", "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613"),
    ]

    static func markerData(
        sourceRoot: String,
        repository: String = repository,
        revision: String = revision,
        storageProfile: String = storageProfile,
        sidecars: [String: String]? = nil,
        shards: [(filename: String, sha256: String)]? = nil,
        contentSHA256: String? = nil
    ) throws -> Data {
        let activeSidecars = sidecars ?? Self.sidecars
        let activeShards = shards ?? Self.shards
        let object: [String: Any] = [
            "kind": "official-safetensors-bf16-v1",
            "version": 1,
            "repository": repository,
            "revision": revision,
            "storageProfile": storageProfile,
            "sidecarSHA256": activeSidecars,
            "shards": activeShards.map { ["filename": $0.filename, "sha256": $0.sha256] },
            "sourceRoot": sourceRoot,
            "contentSHA256": contentSHA256 ?? independentDigest(
                repository: repository,
                revision: revision,
                storageProfile: storageProfile,
                sidecars: activeSidecars,
                shards: activeShards),
        ]
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    static func independentDigest(
        repository: String = repository,
        revision: String = revision,
        storageProfile: String = storageProfile,
        sidecars: [String: String] = sidecars,
        shards: [(filename: String, sha256: String)] = shards
    ) -> String {
        var canonical = "official-safetensors-bf16-v1\n"
        canonical += "repository=\(repository)\nrevision=\(revision)\nstorageProfile=\(storageProfile)\n"
        for name in sidecars.keys.sorted() {
            canonical += "sidecar:\(name)=\(sidecars[name]!)\n"
        }
        for shard in shards.sorted(by: { $0.filename < $1.filename }) {
            canonical += "shard:\(shard.filename)=\(shard.sha256)\n"
        }
        return SHA256.hash(data: Data(canonical.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    static func changedSHA(_ value: String) -> String {
        String(value.dropLast()) + (value.hasSuffix("0") ? "1" : "0")
    }

    static func jsonString(_ value: String) -> String {
        let data = try! JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed])
        return String(decoding: data, as: UTF8.self)
    }
}
