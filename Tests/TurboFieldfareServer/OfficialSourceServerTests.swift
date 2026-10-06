import Darwin
import CryptoKit
import Foundation
import Testing
@testable import TurboFieldfare
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfareServerCore

@Suite("Official source server", .serialized)
struct OfficialSourceServerTests {
    @Test func markerOnlySourceCannotStartServerModel() async throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-source-marker-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        var canonical = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(temporary.path, &canonical) != nil else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let root = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let registration = root.appendingPathComponent("source.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let pin = OfficialQwenSourceIdentity.pinned
        let descriptor = try OfficialSourceDescriptor(
            repository: pin.repository, revision: pin.revision,
            storageProfile: pin.storageProfile,
            sidecarSHA256: pin.sidecarSHA256,
            shards: pin.shards.map { OfficialSourceDescriptor.Shard(
                filename: $0.filename, sha256: $0.sha256) },
            sourceRoot: source.path)
        _ = try OfficialSourceRegistration.register(
            markerData: JSONEncoder().encode(descriptor), at: registration)
        let admission = try ModelFamilyGenerationSession.inspect(directoryURL: registration)
        #expect(admission.family == .qwen3_6)
        #expect(admission.verifiedIdentity == nil)

        do {
            _ = try await ServerModelLoader.load(
                modelDirectory: registration, assertedModelID: pin.repository,
                maxContext: 4_096, visionPackURL: nil,
                visionResidencyPolicy: .onDemand, promptCacheMode: .off,
                runtimeConfiguration: .production)
            Issue.record("metadata-only marker unexpectedly started a BF16 server model")
        } catch {
            // The production loader must require source payload and codec
            // admission; the marker alone is never enough to serve requests.
            #expect(!FileManager.default.fileExists(
                atPath: source.appendingPathComponent("model-00001-of-00026.safetensors").path))
        }
    }

    @Test func sourceErrorPathsKeepChangedSourceAndVisionFailuresVisible() async throws {
        let fixture = try SourceServerMetadataFixture.make()
        defer { fixture.remove() }
        let identity = LoadedRuntimeSourceIdentity(
            receipt: try fixture.verifyTrustedReopen())
        let request = ValidatedChatRequest(
            messages: [], tools: [], stream: false, includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 1),
            maximumCompletionTokens: 1,
            qwenPrompt: .chat(
                messages: [.init(role: .user, content: .parts([
                    .text("describe"), .image(.init(id: "image-1")),
                ]))], tools: [], thinking: .automatic),
            qwenImagesByID: ["image-1": URL(fileURLWithPath: "/tmp/image.png")])
        for (failure, expectedCode, expectedMessage) in [
            (ModelFamilyGenerationError.sourceVisionUnavailable,
             "vision_unavailable", "image support is unavailable"),
            (.modelIdentityChanged,
             "invalid_request", "model identity changed"),
        ] {
            if failure == .modelIdentityChanged {
                try FileManager.default.removeItem(at: fixture.sourceRoot)
                #expect(throws: (any Error).self) {
                    _ = try fixture.verifyTrustedReopen()
                }
            }
            let driver = SourceFailureDriver(error: failure)
            let loaded = try ServerModelLoader.finishLoadedSource(
                identity: identity, assertedModelID: nil,
                visionCapability: .invalid, driver: driver,
                visionResidencyPolicy: .onDemand)
            do {
                _ = try await loaded.backend.generate(request) { _ in
                    Issue.record("failed source request published an event")
                }
                Issue.record("failed source request returned a completion")
            } catch let error as ServerRequestError {
                #expect(error.envelope.error.code == expectedCode)
                #expect(error.envelope.error.message.contains(expectedMessage))
            }
            #expect(await driver.generateCount == 0)
        }
    }

    @Test func loopbackChatUsesOneQwenBackendAndNeverBindsRemotely() async throws {
        // A synthetic verifier receipt exercises production server composition.
        // The scripted driver tests HTTP routing, not official weight inference.
        let fixture = try SourceServerMetadataFixture.make()
        defer { fixture.remove() }
        let identity = LoadedRuntimeSourceIdentity(
            receipt: try fixture.verifyTrustedReopen())
        let driver = SourceTextDriver()
        let pin = OfficialQwenSourceIdentity.pinned
        let loaded = try ServerModelLoader.finishLoadedSource(
            identity: identity, assertedModelID: pin.repository,
            visionCapability: .missing, driver: driver,
            visionResidencyPolicy: .onDemand)
        #expect(loaded.identity.sourceIdentity == identity)
        #expect(loaded.identity.verifiedIdentity == nil)
        #expect(loaded.identity.apiModelID == pin.repository)
        #expect(loaded.identity.sourceRevision == pin.revision)
        #expect(throws: ServerArgumentError.self) {
            _ = try ServerModelLoader.finishLoadedSource(
                identity: identity, assertedModelID: "different-model",
                visionCapability: .missing, driver: driver,
                visionResidencyPolicy: .onDemand)
        }
        let server = TurboFieldfareHTTPServer(
            modelID: loaded.identity.apiModelID, queueLimit: 1,
            backend: loaded.backend,
            visionCapability: loaded.visionCapability,
            modelFamily: loaded.identity.family,
            modelRevision: loaded.identity.sourceRevision)
        let channel = try await server.start(port: 0)
        do {
            let port = try #require(channel.localAddress?.port)
            #expect(channel.localAddress?.ipAddress == "127.0.0.1")
            var request = URLRequest(url: URL(
                string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            request.httpBody = Data("""
                {"model":"\(pin.repository)","messages":[{"role":"user","content":"hi"}]}
                """.utf8)
            let (body, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            let object = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
            #expect(object["model"] as? String == pin.repository)
            let choices = try #require(object["choices"] as? [[String: Any]])
            let firstChoice = try #require(choices.first)
            let message = try #require(firstChoice["message"] as? [String: Any])
            #expect(message["content"] as? String == "tiny-source-chat")
            #expect(await driver.preflightCount == 1)
            #expect(await driver.generateCount == 1)
            try await server.shutdown()
        } catch {
            try? await server.shutdown()
            throw error
        }
    }
}

private struct SourceServerMetadataFixture {
    let root: URL
    let sourceRoot: URL
    let modelDirectory: URL
    let shardBytes: UInt64

    static func make() throws -> Self {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("server-source-trust-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        var canonical = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(temporary.path, &canonical) != nil else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let root = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
        let source = root.appendingPathComponent("source", isDirectory: true)
        let registration = root.appendingPathComponent("source.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let pin = OfficialQwenSourceIdentity.pinned
        let extras: [String: String] = [
            "LICENSE": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
            "README.md": "c4ddaa065649ff6352648f64747a16eda31726f3e34add94ce04abb461c77b75",
            "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
            "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        ]
        let names = [
            "LICENSE", "README.md", "chat_template.jinja", "config.json",
            "configuration.json", "generation_config.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
        ]
        let sidecarLines = try names.map { name -> String in
            guard let digest = pin.sidecarSHA256[name] ?? extras[name] else {
                throw SourceServerFixtureError.missingPin(name)
            }
            return "\(digest)  \(name)"
        }
        let shardLines = pin.shards.map { "\($0.sha256)  \($0.filename)" }
        let manifest = (sidecarLines + shardLines).joined(separator: "\n") + "\n"
        let manifestData = Data(manifest.utf8)
        let manifestSHA = SHA256.hash(data: manifestData)
            .map { String(format: "%02x", $0) }.joined()
        guard manifestData.count == 3_571,
              manifestSHA == "1378d1bb153694b13c20641fa5d2485dede401d1ce4c267953ba4a855fc59e7e" else {
            throw SourceServerFixtureError.invalidManifest
        }
        try manifestData.write(to: source.appendingPathComponent("SHA256SUMS"))
        for name in names {
            try Data("tiny \(name)".utf8).write(to: source.appendingPathComponent(name))
        }
        var shardBytes: UInt64 = 0
        for shard in pin.shards {
            let data = Data("tiny \(shard.filename)".utf8)
            shardBytes += UInt64(data.count)
            try data.write(to: source.appendingPathComponent(shard.filename))
        }
        let descriptor = try OfficialSourceDescriptor(
            repository: pin.repository, revision: pin.revision,
            storageProfile: pin.storageProfile,
            sidecarSHA256: pin.sidecarSHA256,
            shards: pin.shards.map { .init(filename: $0.filename, sha256: $0.sha256) },
            sourceRoot: source.path)
        _ = try OfficialSourceRegistration.register(
            markerData: JSONEncoder().encode(descriptor), at: registration)
        _ = try OfficialSourceTrust.verifySynthetic(
            at: registration, policy: .fullSha256,
            expectedShardBytes: shardBytes, fullVerification: {})
        return .init(root: root, sourceRoot: source,
                     modelDirectory: registration, shardBytes: shardBytes)
    }

    func verifyTrustedReopen() throws -> OfficialSourceTrustReceipt {
        try OfficialSourceTrust.verifySynthetic(
            at: modelDirectory, policy: .sizeCheckTrustedReceipt,
            expectedShardBytes: shardBytes, fullVerification: {})
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

private enum SourceServerFixtureError: Error {
    case missingPin(String)
    case invalidManifest
}

private actor SourceFailureDriver: ServerQwenGenerationDriver {
    let error: ModelFamilyGenerationError
    private(set) var generateCount = 0

    init(error: ModelFamilyGenerationError) { self.error = error }

    func preflight(_ request: ModelFamilyGenerationRequest) async throws
        -> ModelFamilyGenerationPreflight {
        throw error
    }

    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @escaping @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        generateCount += 1
        throw error
    }
}

private actor SourceTextDriver: ServerQwenGenerationDriver {
    private(set) var preflightCount = 0
    private(set) var generateCount = 0

    func preflight(_ request: ModelFamilyGenerationRequest) async throws
        -> ModelFamilyGenerationPreflight {
        preflightCount += 1
        return .init(promptTokens: 2, imageCount: 0)
    }

    func generate(
        _ request: ModelFamilyGenerationRequest,
        onEvent: @escaping @Sendable (ModelFamilyGenerationEvent) -> Void
    ) async throws -> ModelFamilyGenerationResult {
        generateCount += 1
        onEvent(.text("tiny-source-chat"))
        return .init(reason: .eos, promptTokens: 2, newTokens: 1,
                     prefillSeconds: 0, decodeSeconds: 0)
    }
}
