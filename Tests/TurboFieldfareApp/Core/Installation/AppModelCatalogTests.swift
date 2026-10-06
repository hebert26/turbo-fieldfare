import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite("App model catalog")
struct AppModelCatalogTests {
    @Test func GemmaRemainsTheDefaultAndCatalogOrderIsStable() {
        #expect(AppModelCatalog.defaultID == .gemma4)
        #expect(AppModelCatalog.all.map(\.id) == [.gemma4, .qwen3_6])
        #expect(AppModelCatalog.all.map(\.family) == [.gemma4, .qwen3_6])
    }

    @Test func GemmaDescriptorAndRemoteRouteRemainUnchanged() {
        let entry = AppModelCatalog.entry(for: .gemma4)
        let descriptor = AppModelInstallDescriptor.default

        #expect(entry.id == .gemma4)
        #expect(entry.displayName == descriptor.displayName)
        #expect(entry.family == .gemma4)
        #expect(entry.sourceIdentity == descriptor.sourceIdentity)
        #expect(entry.isInstallable)
        guard case let .remoteRepack(text, vision) = entry.installRoute else {
            Issue.record("Gemma must retain its remote repack route")
            return
        }
        #expect(text == .default)
        #expect(vision == .visionCompanion)
    }

    @Test func QwenHasItsOwnVerifiedIdentityAndLocalOnlyRoute() {
        let entry = AppModelCatalog.entry(for: .qwen3_6)

        #expect(entry.id == .qwen3_6)
        #expect(entry.displayName == "Qwen3.6 35B-A3B")
        #expect(entry.family == .qwen3_6)
        #expect(entry.sourceIdentity.repoID == "Qwen/Qwen3.6-35B-A3B")
        #expect(entry.sourceIdentity.revision
            == "995ad96eacd98c81ed38be0c5b274b04031597b0")
        #expect(entry.sourceIdentity.sourceIndexSHA256
            == "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83")
        #expect(!entry.isInstallable)
        guard case let .localQwenConversion(local) = entry.installRoute else {
            Issue.record("Qwen must not use the Gemma remote route")
            return
        }
        #expect(local.displayName == entry.displayName)
        #expect(local.sourceIdentity == entry.sourceIdentity)
    }

    @Test func CatalogResolvesSeparateTextAndVisionDestinations() {
        let gemma = AppModelCatalog.entry(for: .gemma4)
        let qwen = AppModelCatalog.entry(for: .qwen3_6)

        #expect(gemma.location.textModelURL.lastPathComponent == "gemma4.gturbo")
        #expect(gemma.location.visionModelURL.lastPathComponent
            == "gemma4.vision.gturbo")
        #expect(qwen.location.textModelURL.lastPathComponent
            == "qwen3.6-35b-a3b.gturbo")
        #expect(qwen.location.visionModelURL.lastPathComponent
            == "qwen3.6-35b-a3b.vision.gturbo")
        #expect(gemma.location.textModelURL != qwen.location.textModelURL)
        #expect(gemma.location.visionModelURL != qwen.location.visionModelURL)
    }

    @Test func LocationResolverKeepsExplicitFamilyPathsSeparate() {
        let root = URL(fileURLWithPath: "/tmp/catalog-location")
        let gemma = AppModelLocation.resolve(
            modelID: .gemma4,
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: root,
            applicationSupportURL: root.appendingPathComponent("support"),
            fileExists: { _ in false })
        let qwen = AppModelLocation.resolve(
            modelID: .qwen3_6,
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: root,
            applicationSupportURL: root.appendingPathComponent("support"),
            fileExists: { _ in false })

        #expect(gemma.textModelURL.path
            == "/tmp/catalog-location/support/TurboFieldfare/gemma4.gturbo")
        #expect(qwen.textModelURL.path
            == "/tmp/catalog-location/support/TurboFieldfare/qwen3.6-35b-a3b.gturbo")
        #expect(gemma != qwen)
    }

    @Test func CatalogAcceptsOnlyTheExpectedQwenIdentity() {
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        let identity = entry.sourceIdentity

        #expect(entry.accepts(
            family: .qwen3_6,
            modelID: identity.repoID,
            revision: identity.revision,
            sourceIndexSHA256: identity.sourceIndexSHA256))
        #expect(entry.accepts(
            family: .qwen3_6,
            modelID: identity.repoID,
            revision: identity.revision,
            sourceIndexSHA256: identity.sourceIndexSHA256.uppercased()))
        #expect(!entry.accepts(
            family: .gemma4,
            modelID: identity.repoID,
            revision: identity.revision,
            sourceIndexSHA256: identity.sourceIndexSHA256))
        #expect(!entry.accepts(
            family: .qwen3_6,
            modelID: "Qwen/Qwen3.6-35B-A3B-replacement",
            revision: identity.revision,
            sourceIndexSHA256: identity.sourceIndexSHA256))
    }
}
