import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfareAppCore

@Suite("Official BF16 app registration")
struct OfficialSourceInstallationTests {
    @Test func markerOnlyIsNotReadyAndMovedSourceBecomesUnavailable() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        let logical = root.appendingPathComponent("qwen3.6-35b-a3b.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let pin = OfficialQwenSourceIdentity.pinned
        let descriptor = try OfficialSourceDescriptor(
            repository: pin.repository, revision: pin.revision,
            storageProfile: pin.storageProfile,
            sidecarSHA256: pin.sidecarSHA256,
            shards: pin.shards.map {
                .init(filename: $0.filename, sha256: $0.sha256)
            }, sourceRoot: source.path)
        _ = try OfficialSourceRegistration.register(
            markerData: JSONEncoder().encode(descriptor), at: logical)
        let entry = AppModelCatalog.entry(
            for: .qwen3_6,
            location: .init(textModelURL: logical,
                            visionModelURL: root.appendingPathComponent(
                                "qwen3.6-35b-a3b.vision.gturbo", isDirectory: true),
                            preservePhysicalPath: true))
        #expect(logical.path.hasPrefix("/private/var/"))
        guard case let .partial(reason) = AppModelInstallationProbe.status(at: logical, entry: entry) else {
            Issue.record("descriptor-only source was marked ready or missing")
            return
        }
        #expect(reason == "\(OfficialSourceTrust.TrustError.missing)")
        try FileManager.default.moveItem(
            at: source, to: root.appendingPathComponent("moved-source", isDirectory: true))
        guard case .partial = AppModelInstallationProbe.status(at: logical, entry: entry) else {
            Issue.record("moved source was marked ready")
            return
        }
    }

    @Test func sourceAndLogicalPathsSurviveSettingsReopen() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let gemma = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let logical = root.appendingPathComponent("qwen3.6-35b-a3b.gturbo", isDirectory: true)
        let settings = MacAppSettings(qwenSourceRoot: source.path,
                                      qwenRegistrationPath: logical.path)
        try MacAppSettingsFileStore.save(settings, forModelDirectory: gemma)
        let reopened = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: gemma)
        #expect(reopened.qwenSourceRoot == source.path)
        #expect(reopened.qwenRegistrationPath == logical.path)
    }

    @MainActor
    @Test func movedSourceGetsNewLogicalTargetAndReopensIt() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let gemma = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let gemmaSentinel = gemma.appendingPathComponent("keep.bin")
        try FileManager.default.createDirectory(at: gemma, withIntermediateDirectories: true)
        try Data("Gemma stays".utf8).write(to: gemmaSentinel)
        let original = root.appendingPathComponent("original-source", isDirectory: true)
        let moved = root.appendingPathComponent("moved-source", isDirectory: true)
        try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
        let shard = original.appendingPathComponent("model-00001-of-00026.safetensors")
        try Data("original weights stay".utf8).write(to: shard)
        let logical = root.appendingPathComponent("qwen3.6-35b-a3b.gturbo", isDirectory: true)
        let pin = OfficialQwenSourceIdentity.pinned
        let descriptor = try OfficialSourceDescriptor(
            repository: pin.repository, revision: pin.revision,
            storageProfile: pin.storageProfile, sidecarSHA256: pin.sidecarSHA256,
            shards: pin.shards.map { .init(filename: $0.filename, sha256: $0.sha256) },
            sourceRoot: original.path)
        _ = try OfficialSourceRegistration.register(
            markerData: JSONEncoder().encode(descriptor), at: logical)
        try MacAppSettingsFileStore.save(
            MacAppSettings(selectedModelID: .qwen3_6,
                           qwenSourceRoot: original.path,
                           qwenRegistrationPath: logical.path),
            forModelDirectory: gemma)
        try FileManager.default.moveItem(at: original, to: moved)

        let model = AppModel(modelDirectory: gemma, settingsPersistenceEnabled: true)
        #expect(model.selectedModelEntry.location.textModelURL.path == logical.path)
        #expect(model.modelPathText == logical.path)
        guard case .partial = model.installationStatus else {
            Issue.record("moved source appeared ready")
            return
        }
        model.configureQwenSourceDirectory(moved)
        for _ in 0..<200 {
            if case .ready = model.qwenSourceConfigurationState { break }
            await Task.yield()
        }
        guard case .ready = model.qwenSourceConfigurationState else {
            Issue.record("moved source could not be selected")
            return
        }
        let newLogical = model.selectedModelEntry.location.textModelURL
        #expect(newLogical.path != logical.path)
        #expect(newLogical.lastPathComponent.hasSuffix(".gturbo"))
        #expect(FileManager.default.fileExists(
            atPath: logical.appendingPathComponent("official-source.json").path))
        let reopened = AppModel(modelDirectory: gemma, settingsPersistenceEnabled: true)
        #expect(reopened.selectedModelEntry.location.textModelURL.path == newLogical.path)
        #expect(reopened.modelPathText == newLogical.path)
        #expect(reopened.qwenSourceDirectory?.path == moved.path)
        #expect(reopened.installationStatus == .missing)

        model.clearQwenSourceDirectory()
        #expect(try Data(contentsOf: moved.appendingPathComponent(
            "model-00001-of-00026.safetensors")) == Data("original weights stay".utf8))
        #expect(try Data(contentsOf: gemmaSentinel) == Data("Gemma stays".utf8))
    }

    private func temporaryRoot() throws -> URL {
        let temporaryPath = FileManager.default.temporaryDirectory.path
        guard let physical = temporaryPath.withCString({ realpath($0, nil) }) else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer { free(physical) }
        let root = URL(fileURLWithPath: String(cString: physical), isDirectory: true)
            .appendingPathComponent("app-source-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
