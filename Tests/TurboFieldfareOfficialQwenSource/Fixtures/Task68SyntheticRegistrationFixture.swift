import Darwin
import Foundation
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

/// Small temporary marker-only registrations for Task 6.8 layout tests.
/// This fixture contains no checkpoint bytes and makes no payload-trust claim.
struct Task68SyntheticRegistrationFixture {
    let root: URL
    let sourceRoot: URL
    let modelDirectory: URL
    let descriptor: OfficialSourceDescriptor

    static func make() throws -> Self {
        var root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "task-6-8-source-registration-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: false)
        var canonicalRoot = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(root.path, &canonicalRoot) != nil else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        root = URL(fileURLWithPath: String(cString: canonicalRoot), isDirectory: true)
        let sourceRoot = root.appendingPathComponent("synthetic-source", isDirectory: true)
        try FileManager.default.createDirectory(
            at: sourceRoot, withIntermediateDirectories: false)
        let modelParent = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelParent, withIntermediateDirectories: false)
        let modelDirectory = modelParent.appendingPathComponent(
            "source.gturbo", isDirectory: true)

        let identity = OfficialQwenSourceIdentity.pinned
        let descriptor = try OfficialSourceDescriptor(
            repository: identity.repository,
            revision: identity.revision,
            storageProfile: identity.storageProfile,
            sidecarSHA256: identity.sidecarSHA256,
            shards: identity.shards.map {
                OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
            },
            sourceRoot: sourceRoot.path)
        let marker = try JSONEncoder().encode(descriptor)
        _ = try OfficialSourceRegistration.register(markerData: marker, at: modelDirectory)
        return Self(root: root, sourceRoot: sourceRoot,
                    modelDirectory: modelDirectory, descriptor: descriptor)
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}
