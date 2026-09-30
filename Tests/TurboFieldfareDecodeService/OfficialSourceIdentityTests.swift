import Darwin
import Foundation
import Testing
import TurboFieldfareDecodeProtocol
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareDecodeService

/// Marker-only registrations exercise service binding without claiming that
/// synthetic files are a trusted or runnable BF16 checkpoint.
@Suite(.serialized) struct OfficialSourceIdentityTests {
    @Test func sourceWireIdentityCannotBePackedIdentityWithSameIndex() throws {
        let index = OfficialQwenSourceIdentity.pinned.sidecarSHA256["model.safetensors.index.json"]!
        let source = try DecodeSourceIdentity(
            kind: .officialSafetensorsBF16V1,
            contentDigest: String(repeating: "a", count: 64))
        let packed = DecodeModelIdentity(
            family: .qwen3_6, modelID: "Qwen/Qwen3.6-35B-A3B",
            sourceRevision: "fixture", formatMajor: 2, formatMinor: 0,
            sourceIndexSHA256: index,
            quantizationPolicySHA256: String(repeating: "b", count: 64),
            textManifestSHA256: String(repeating: "c", count: 64),
            quantization: [], vision: .unavailable)
        let session = DecodeServiceSession()
        let sourceBinding = try session.publish(.qwenSource(identity: source))
        #expect(sourceBinding.sourceIdentity == source)
        #expect(sourceBinding.modelIdentity == nil)
        let packedBinding = try session.publish(.qwen(identity: packed))
        #expect(packedBinding.modelIdentity == packed)
        #expect(packedBinding.sourceIdentity == nil)
        #expect(sourceBinding.loadID != packedBinding.loadID)

        let valid = DecodeServiceEvent(kind: .ready, generationID: UUID(),
            loadedFamily: .qwen3_6, loadID: sourceBinding.loadID,
            sourceIdentity: source)
        try valid.validateBackingIdentity()
        let decoded = try DecodeFrameCodec.read(
            DecodeServiceEvent.self,
            from: frameInput(try DecodeFrameCodec.encode(valid)))
        #expect(decoded.sourceIdentity == source)
        #expect(decoded.modelIdentity == nil)
        let contradictory = DecodeServiceEvent(kind: .ready,
            generationID: UUID(), loadedFamily: .qwen3_6,
            loadID: sourceBinding.loadID, modelIdentity: packed,
            sourceIdentity: source)
        #expect(throws: DecodeServiceEvent.BackingError.self) {
            try contradictory.validateBackingIdentity()
        }
    }

    @Test func sourceLoadReopensAndMissingMarkerIsUnavailable() async throws {
        let fixture = try MarkerFixture.make()
        defer { fixture.remove() }
        let identity = try fixture.wireIdentity()
        func load(_ session: DecodeServiceSession,
                  _ coordinator: DecodeServiceLoadCoordinator) async throws
            -> DecodeServiceSession.Binding {
            let request = DecodeLoadRequest(modelPath: fixture.registration.path,
                maxContextTokens: 64, requestID: UUID(), attemptID: UUID())
            let lease = try #require(session.registerLoad(request))
            let attempt = try coordinator.beginReplacement(lease)
            let readiness = try await coordinator.prepare(attempt,
                directory: fixture.registration, maxContextTokens: 64,
                options: AppRuntimeOptions(), forceLogitsHead: false)
            let binding = try coordinator.reservePublication(attempt, readiness: readiness)
            let event = DecodeServiceEvent(kind: .ready,
                generationID: request.requestID, loadAttemptID: request.attemptID,
                loadedFamily: binding.family, loadID: binding.loadID,
                sourceIdentity: binding.sourceIdentity)
            try await coordinator.validatePublication(attempt, event: event)
            _ = try coordinator.finishCommit(attempt)
            return binding
        }

        let firstSession = DecodeServiceSession()
        let firstRuntime = SourceFakeRuntime(readiness: .qwenSource(identity: identity))
        let firstCoordinator = DecodeServiceLoadCoordinator(
            session: firstSession, runtime: firstRuntime)
        let first = try await load(firstSession, firstCoordinator)
        // A new owner models a service restart: no source handle or load
        // incarnation is reused from the first coordinator.
        let session = DecodeServiceSession()
        let runtime = SourceFakeRuntime(readiness: .qwenSource(identity: identity))
        let coordinator = DecodeServiceLoadCoordinator(session: session, runtime: runtime)
        let second = try await load(session, coordinator)
        #expect(first.loadID != second.loadID)
        #expect(second.sourceIdentity == identity)
        try FileManager.default.removeItem(at: fixture.marker)
        let request = DecodeLoadRequest(modelPath: fixture.registration.path,
            maxContextTokens: 64, requestID: UUID(), attemptID: UUID())
        let lease = try #require(session.registerLoad(request))
        let attempt = try coordinator.beginReplacement(lease)
        await #expect(throws: (any Error).self) {
            _ = try await coordinator.prepare(attempt, directory: fixture.registration,
                maxContextTokens: 64, options: AppRuntimeOptions(),
                forceLogitsHead: false)
        }
        await coordinator.abort(attempt)
        #expect(session.currentBinding == nil)
    }

    @Test func markerReplacementDuringLoadAndBeforeReadyIsRejected() async throws {
        for replaceDuringLoad in [true, false] {
            let fixture = try MarkerFixture.make()
            defer { fixture.remove() }
            let identity = try fixture.wireIdentity()
            let runtime = SourceFakeRuntime(readiness: .qwenSource(identity: identity))
            let session = DecodeServiceSession()
            let coordinator = DecodeServiceLoadCoordinator(session: session, runtime: runtime)
            let request = DecodeLoadRequest(modelPath: fixture.registration.path,
                maxContextTokens: 64, requestID: UUID(), attemptID: UUID())
            let lease = try #require(session.registerLoad(request))
            let attempt = try coordinator.beginReplacement(lease)
            if replaceDuringLoad {
                runtime.onEnsure = { try fixture.replaceMarker() }
                await #expect(throws: (any Error).self) {
                    _ = try await coordinator.prepare(attempt,
                        directory: fixture.registration, maxContextTokens: 64,
                        options: AppRuntimeOptions(), forceLogitsHead: false)
                }
            } else {
                let readiness = try await coordinator.prepare(attempt,
                    directory: fixture.registration, maxContextTokens: 64,
                    options: AppRuntimeOptions(), forceLogitsHead: false)
                let binding = try coordinator.reservePublication(attempt, readiness: readiness)
                try fixture.replaceMarker()
                let event = DecodeServiceEvent(kind: .ready,
                    generationID: request.requestID, loadAttemptID: request.attemptID,
                    loadedFamily: binding.family, loadID: binding.loadID,
                    sourceIdentity: binding.sourceIdentity)
                await #expect(throws: (any Error).self) {
                    try await coordinator.validatePublication(attempt, event: event)
                }
            }
            await coordinator.abort(attempt)
            #expect(session.currentBinding == nil)
        }
    }

    @Test func removedOrReplacedSourceRootCannotPublishReady() async throws {
        for removeDuringLoad in [true, false] {
            let fixture = try MarkerFixture.make()
            defer { fixture.remove() }
            let identity = try fixture.wireIdentity()
            let runtime = SourceFakeRuntime(readiness: .qwenSource(identity: identity))
            let session = DecodeServiceSession()
            let coordinator = DecodeServiceLoadCoordinator(session: session, runtime: runtime)
            let request = DecodeLoadRequest(modelPath: fixture.registration.path,
                maxContextTokens: 64, requestID: UUID(), attemptID: UUID())
            let lease = try #require(session.registerLoad(request))
            let attempt = try coordinator.beginReplacement(lease)
            if removeDuringLoad {
                runtime.onEnsure = { try fixture.removeSourceRoot() }
                await #expect(throws: (any Error).self) {
                    _ = try await coordinator.prepare(attempt,
                        directory: fixture.registration, maxContextTokens: 64,
                        options: AppRuntimeOptions(), forceLogitsHead: false)
                }
            } else {
                let readiness = try await coordinator.prepare(attempt,
                    directory: fixture.registration, maxContextTokens: 64,
                    options: AppRuntimeOptions(), forceLogitsHead: false)
                let binding = try coordinator.reservePublication(attempt, readiness: readiness)
                try fixture.replaceSourceRoot()
                let event = DecodeServiceEvent(kind: .ready,
                    generationID: request.requestID, loadAttemptID: request.attemptID,
                    loadedFamily: binding.family, loadID: binding.loadID,
                    sourceIdentity: binding.sourceIdentity)
                await #expect(throws: (any Error).self) {
                    try await coordinator.validatePublication(attempt, event: event)
                }
            }
            await coordinator.abort(attempt)
            #expect(session.currentBinding == nil)
            // The marker was never edited; failure came from the source root.
            #expect(try OfficialSourceDescriptor.decodeStrict(
                data: Data(contentsOf: fixture.marker)) == fixture.descriptor)
        }
    }
}

private func frameInput(_ frame: Data) throws -> FileHandle {
    let pipe = Pipe()
    try pipe.fileHandleForWriting.write(contentsOf: frame)
    try pipe.fileHandleForWriting.close()
    return pipe.fileHandleForReading
}

private struct MarkerFixture: Sendable {
    let root: URL
    let source: URL
    let registration: URL
    let marker: URL
    let descriptor: OfficialSourceDescriptor

    static func make() throws -> Self {
        var root = FileManager.default.temporaryDirectory
            .appendingPathComponent("phase17-source-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        var canonical = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(root.path, &canonical) != nil else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        root = URL(fileURLWithPath: String(decoding:
            canonical.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) },
            as: UTF8.self), isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        let registration = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: registration, withIntermediateDirectories: false)
        let pinned = OfficialQwenSourceIdentity.pinned
        let descriptor = try OfficialSourceDescriptor(
            repository: pinned.repository, revision: pinned.revision,
            storageProfile: pinned.storageProfile,
            sidecarSHA256: pinned.sidecarSHA256,
            shards: pinned.shards.map { .init(filename: $0.filename, sha256: $0.sha256) },
            sourceRoot: source.path)
        let marker = registration.appendingPathComponent(OfficialSourceDescriptor.markerFilename)
        try JSONEncoder().encode(descriptor).write(to: marker)
        return Self(root: root, source: source,
                    registration: registration, marker: marker,
                    descriptor: descriptor)
    }

    func wireIdentity() throws -> DecodeSourceIdentity {
        try DecodeSourceIdentity(kind: .officialSafetensorsBF16V1,
                                 contentDigest: descriptor.contentSHA256)
    }

    func replaceMarker() throws {
        let replacement = registration.appendingPathComponent("replacement")
        try JSONEncoder().encode(descriptor).write(to: replacement)
        _ = try FileManager.default.replaceItemAt(marker, withItemAt: replacement)
    }

    func removeSourceRoot() throws {
        try FileManager.default.removeItem(at: source)
    }

    func replaceSourceRoot() throws {
        let moved = root.appendingPathComponent("old-source", isDirectory: true)
        try FileManager.default.moveItem(at: source, to: moved)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

private final class SourceFakeRuntime: DecodeServiceSourceRuntime, @unchecked Sendable {
    let readiness: AppLoadedModelReadiness
    var onEnsure: (@Sendable () throws -> Void)?
    init(readiness: AppLoadedModelReadiness) { self.readiness = readiness }
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        try onEnsure?()
        await Task.yield()
    }
    func resetConversation(epoch: UUID) async throws {}
    var loadedModelReadiness: AppLoadedModelReadiness? { get async { readiness } }
    func validateSourceReadiness(_ identity: DecodeSourceIdentity) async throws {
        guard readiness == .qwenSource(identity: identity) else {
            throw DecodeServiceLoadCoordinator.Rejection.sourceRegistrationChanged
        }
    }
    func unload() async {}
}
