import CryptoKit
import Foundation
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

@Suite("Qwen installation routing")
struct AppQwenInstallationTests {
    @Test func QwenInstallRouteIsLocalAndCannotUseGemmaRemoteDestination() {
        let qwen = AppModelCatalog.entry(for: .qwen3_6)
        let gemma = AppModelCatalog.entry(for: .gemma4)

        guard case .localQwenConversion = qwen.installRoute else {
            Issue.record("Qwen was advertised through a remote Gemma route")
            return
        }
        guard case .remoteRepack = gemma.installRoute else {
            Issue.record("Gemma lost its existing remote route")
            return
        }
        #expect(qwen.location.textModelURL != gemma.location.textModelURL)
        #expect(qwen.location.visionModelURL != gemma.location.visionModelURL)
        #expect(!qwen.isInstallable)
    }

    @Test func MissingOrMalformedQwenArtifactsAreNotAdvertisedAsComplete() throws {
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        let root = try Self.makeTemporaryRoot("qwen-probe-missing")
        defer { try? FileManager.default.removeItem(at: root) }
        let missing = root.appendingPathComponent("missing.gturbo", isDirectory: true)
        #expect(AppModelInstallationProbe.status(at: missing, entry: entry) == .missing)

        let malformed = root.appendingPathComponent("malformed.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: malformed, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: malformed.appendingPathComponent("manifest.json"))
        guard case .partial = AppModelInstallationProbe.status(at: malformed, entry: entry) else {
            Issue.record("malformed Qwen metadata was not reported partial")
            return
        }
    }

    @Test func OldPackedQwenWithMatchingOfficialIndexCannotSatisfyBF16Selection() throws {
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        let directory = try Self.makeQwenInstall("complete")
        defer { try? FileManager.default.removeItem(at: directory) }

        guard case .partial = AppModelInstallationProbe.status(at: directory, entry: entry) else {
            Issue.record("old packed Qwen was accepted as the original BF16 source")
            return
        }

        let receiptURL = directory.appendingPathComponent(
            VerifiedInstallReceiptReader.fileName)
        var receipt = try JSONSerialization.jsonObject(
            with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/different/qwen.gturbo"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: receiptURL, options: .atomic)
        guard case .partial = AppModelInstallationProbe.status(at: directory, entry: entry) else {
            Issue.record("receipt bound to a different directory was accepted")
            return
        }

        let restored = try Self.makeQwenInstall("manifest-binding")
        defer { try? FileManager.default.removeItem(at: restored) }
        let restoredReceiptURL = restored.appendingPathComponent(
            VerifiedInstallReceiptReader.fileName)
        var badManifestReceipt = try JSONSerialization.jsonObject(
            with: Data(contentsOf: restoredReceiptURL)) as! [String: Any]
        badManifestReceipt["manifestSha256"] = String(repeating: "0", count: 64)
        try JSONSerialization.data(
            withJSONObject: badManifestReceipt, options: [.sortedKeys])
            .write(to: restoredReceiptURL, options: .atomic)
        guard case .partial = AppModelInstallationProbe.status(
            at: restored, entry: entry) else {
            Issue.record("receipt with a different manifest hash was accepted")
            return
        }
    }

    @Test func QwenTextCanBeReadyWhileItsVisionCompanionIsMissing() throws {
        let directory = try Self.makeQwenInstall("text-only")
        defer { try? FileManager.default.removeItem(at: directory) }

        guard case .partial = AppModelInstallationProbe.status(
            at: directory, entry: AppModelCatalog.entry(for: .qwen3_6)) else {
            Issue.record("old packed Qwen was accepted as the original BF16 source")
            return
        }
        #expect(AppVisionPackInstallationProbe.status(
            at: directory, entry: AppModelCatalog.entry(for: .qwen3_6)) == .missing)
    }

    @Test func QwenVisionProbeRejectsAnUnboundCompanion() throws {
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        let directory = try Self.makeQwenInstall("invalid-vision")
        defer { try? FileManager.default.removeItem(at: directory) }
        let companion = try VisionPackLocation.companionURL(
            forTextModel: directory)
        try FileManager.default.createDirectory(
            at: companion, withIntermediateDirectories: true)
        try Data("{}".utf8).write(
            to: companion.appendingPathComponent("manifest.json"))

        guard case .partial = AppVisionPackInstallationProbe.status(
            at: directory, entry: entry) else {
            Issue.record("unbound Qwen vision companion was reported ready")
            return
        }
    }

    @Test func RegistrationDiscardDoesNotTouchOldPackedPartialFiles() async throws {
        let root = try Self.makeTemporaryRoot("qwen-local-installer")
        defer { try? FileManager.default.removeItem(at: root) }
        let location = AppModelLocation.Resolved(
            textModelURL: root.appendingPathComponent("qwen.gturbo", isDirectory: true),
            visionModelURL: root.appendingPathComponent("qwen.vision.gturbo", isDirectory: true))
        let entry = AppModelCatalog.entry(for: .qwen3_6, location: location)
        let source = root.appendingPathComponent("official-source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let sourceShard = source.appendingPathComponent("model-00001-of-00026.safetensors")
        try Data("original source".utf8).write(to: sourceShard)
        let gemma = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: gemma, withIntermediateDirectories: true)
        let gemmaSentinel = gemma.appendingPathComponent("keep.bin")
        try Data("Gemma stays".utf8).write(to: gemmaSentinel)
        let installer = try LocalQwenModelInstallerClient(
            entry: entry, sourceDirectory: source, includesVision: false)
        let partial = URL(fileURLWithPath: location.textModelURL.path + ".partial",
                          isDirectory: true)
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
        let resume = URL(fileURLWithPath: location.textModelURL.path + ".resume.json")
        try Data("{}".utf8).write(to: resume, options: .atomic)

        #expect(installer.sourceDirectory == source.standardizedFileURL)
        #expect(!installer.hasPartialInstall)
        #expect(!installer.canResume)
        installer.cancel()
        #expect(FileManager.default.fileExists(atPath: partial.path),
                "cancellation must leave a resumable partial install")
        #expect(!installer.canResume)
        try await installer.discardPartialInstall()
        #expect(FileManager.default.fileExists(atPath: partial.path))
        #expect(FileManager.default.fileExists(atPath: resume.path))
        #expect(try Data(contentsOf: sourceShard) == Data("original source".utf8))
        #expect(try Data(contentsOf: gemmaSentinel) == Data("Gemma stays".utf8))
    }

    @Test func LocalQwenInstallerRejectsGemmaRoute() throws {
        let root = try Self.makeTemporaryRoot("qwen-route-reject")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("official-source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        #expect(throws: AppModelInstallerRoutingError.self) {
            try LocalQwenModelInstallerClient(
                entry: AppModelCatalog.entry(for: .gemma4),
                sourceDirectory: source)
        }
    }

    @Test func staleStreamTerminationCannotCancelNewRegistration() async throws {
        let root = try Self.makeTemporaryRoot("qwen-stream-owner")
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let location = AppModelLocation.Resolved(
            textModelURL: root.appendingPathComponent("qwen.gturbo", isDirectory: true),
            visionModelURL: root.appendingPathComponent("qwen.vision.gturbo", isDirectory: true))
        let installer = try LocalQwenModelInstallerClient(
            entry: AppModelCatalog.entry(for: .qwen3_6, location: location),
            sourceDirectory: source, includesVision: false,
            beforeRegistration: { try await Task.sleep(for: .seconds(30)) })
        let first = installer.install(resume: false)
        let firstConsumer = Task {
            do {
                for try await _ in first {}
                return false
            } catch is CancellationError {
                return true
            } catch {
                return false
            }
        }
        #expect(installer.activeInstallID != nil)
        let second = installer.install(resume: false)
        let secondConsumer = Task {
            do {
                for try await _ in second {}
                return false
            } catch is CancellationError {
                return true
            } catch {
                return false
            }
        }
        let secondID = try #require(installer.activeInstallID)
        #expect(await firstConsumer.value)
        #expect(installer.activeInstallID == secondID)
        installer.cancel()
        #expect(await secondConsumer.value)
    }

    @Test func QwenProgressBurstDoesNotReplayStaleUpdatesBeforeInstallation() async throws {
        let output = URL(fileURLWithPath: "/tmp/qwen-burst.gturbo", isDirectory: true)
        let stream = AsyncThrowingStream<AppModelInstallEvent, Error>(
            bufferingPolicy: LocalQwenModelInstallerClient.eventBufferingPolicy
        ) { continuation in
            continuation.yield(.checking)
            for completed in 0..<8_192 {
                continuation.yield(.copyingPayload(
                    reusedBytes: 0,
                    downloadedThisRunBytes: UInt64(completed),
                    totalBytes: 8_192))
            }
            continuation.yield(.installed(output))
            continuation.finish()
        }

        // The producer has already completed its burst when the consumer starts.
        var events: [AppModelInstallEvent] = []
        for try await event in stream {
            events.append(event)
        }

        #expect(events == [.installed(output)])
    }

    @Test func QwenProgressBurstPreservesTerminalErrorForLateConsumer() async throws {
        let stream = AsyncThrowingStream<AppModelInstallEvent, Error>(
            bufferingPolicy: LocalQwenModelInstallerClient.eventBufferingPolicy
        ) { continuation in
            for completed in 0..<8_192 {
                continuation.yield(.copyingPayload(
                    reusedBytes: 0,
                    downloadedThisRunBytes: UInt64(completed),
                    totalBytes: 8_192))
            }
            continuation.finish(throwing: SyntheticInstallationError())
        }

        var events: [AppModelInstallEvent] = []
        var terminalError: Error?
        do {
            for try await event in stream {
                events.append(event)
            }
        } catch {
            terminalError = error
        }

        #expect(terminalError is SyntheticInstallationError)
        #expect(!events.contains {
            if case .installed = $0 { return true }
            return false
        })
    }

    @Test func QwenProgressBurstPreservesCancellationForLateConsumer() async throws {
        let stream = AsyncThrowingStream<AppModelInstallEvent, Error>(
            bufferingPolicy: LocalQwenModelInstallerClient.eventBufferingPolicy
        ) { continuation in
            for completed in 0..<8_192 {
                continuation.yield(.copyingPayload(
                    reusedBytes: 0,
                    downloadedThisRunBytes: UInt64(completed),
                    totalBytes: 8_192))
            }
            continuation.finish(throwing: CancellationError())
        }

        var events: [AppModelInstallEvent] = []
        var wasCancelled = false
        do {
            for try await event in stream {
                events.append(event)
            }
        } catch is CancellationError {
            wasCancelled = true
        }

        #expect(wasCancelled)
        #expect(!events.contains {
            if case .installed = $0 { return true }
            return false
        })
    }

    @Test func VisionReadinessDistinguishesUnsupportedLayoutFromMissingPack() throws {
        let root = try Self.makeTemporaryRoot("qwen-vision")
        defer { try? FileManager.default.removeItem(at: root) }
        let unsupported = root.appendingPathComponent("qwen-source", isDirectory: true)
        try FileManager.default.createDirectory(at: unsupported, withIntermediateDirectories: true)
        let missing = root.appendingPathComponent("qwen-source.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: missing, withIntermediateDirectories: true)

        #expect(AppVisionPackInstallationProbe.status(at: unsupported)
            == .unsupportedLayout)
        #expect(AppVisionPackInstallationProbe.status(at: missing) == .missing)
    }

    private static func makeQwenInstall(_ tag: String) throws -> URL {
        let directory = try makeTemporaryRoot("qwen-\(tag)")
        let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
        try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)

        let weights = Data(repeating: UInt8(tag.utf8.first ?? 0), count: 32_768)
        let layout = Data("{}".utf8)
        let weightsURL = directory.appendingPathComponent("model_weights.bin")
        let layoutURL = experts.appendingPathComponent("layout.json")
        try weights.write(to: weightsURL, options: .atomic)
        try layout.write(to: layoutURL, options: .atomic)
        let weightsHash = Self.sha256(weights)
        let layoutHash = Self.sha256(layout)

        let layers: [GTurboQwenLayerTypeV2] = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
        }
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2_048, numLayers: 40, layerTypes: layers,
            numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256,
            attentionOutputGate: true, linearConvolutionKernel: 4,
            linearKeyHeads: 16, linearKeyHeadDimension: 128,
            linearValueHeads: 32, linearValueHeadDimension: 128,
            recurrentStateType: .fp32, partialRotaryFactor: 0.25,
            ropeTheta: 10_000_000, mropeInterleaved: true,
            mropeSections: [11, 11, 10], numberOfExperts: 256,
            expertsPerToken: 8, routedExpertIntermediateSize: 512,
            sharedExpertIntermediateSize: 512, vocabularySize: 248_320,
            tiedWordEmbeddings: false, hiddenActivation: "silu",
            bosTokenID: 248_044, eosTokenID: 248_044,
            imageTokenID: 248_056, videoTokenID: 248_057,
            visionStartTokenID: 248_053, visionEndTokenID: 248_054)
        let quantization = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState
                ? GTurboQuantizationGroupV2(category: category, storage: .fp32)
                : GTurboQuantizationGroupV2(
                    category: category, storage: .affineInt4,
                    groupSize: 64, scaleType: "bf16", biasType: "bf16")
        }
        let unsupportedMTPNames = [
            "fc.weight", "layers.0.input_layernorm.weight",
            "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.gate.weight",
            "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight",
            "norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
        ].map {
            GTurboIgnoredTensorV2(name: "mtp.\($0)", reason: .unsupportedMTP)
        }
        let manifest = GTurboManifestV2(
            family: .qwen3_6,
            requiredFeatures: [.familyDispatch, .verifiedIdentity,
                               .qwenHybridAttention, .qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository,
            architecture: .qwen3_6(architecture),
            provenance: .init(
                sourceRepository: GTurboFormatV2.qwenRepository,
                sourceRevision: GTurboFormatV2.qwenRevision,
                sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
                sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256,
                quantizationPolicySHA256: String(repeating: "a", count: 64)),
            quantization: quantization,
            ignoredTensors: unsupportedMTPNames,
            files: [
                "model_weights.bin": .init(
                    size: UInt64(weights.count), sha256: weightsHash),
                "packed_experts/layout.json": .init(
                    size: UInt64(layout.count), sha256: layoutHash),
            ],
            tensorRegions: [.init(
                name: "embed", file: "model_weights.bin", offset: 0,
                size: 16, shape: [1], storage: .affineInt4,
                quantizationCategory: .embedding)],
            expertsPerLayer: 256, numLayers: 40, expertStride: 16_384)
        let manifestData = try GTurboManifestV2Codec.encode(manifest)
        let manifestURL = directory.appendingPathComponent("manifest.json")
        try manifestData.write(to: manifestURL, options: .atomic)
        let manifestHash = Self.sha256(manifestData)
        let receipt = VerifiedInstallReceipt(
            manifestSha256: manifestHash,
            modelDirectoryPath: directory.standardizedFileURL.path,
            sourceRepoID: GTurboFormatV2.qwenRepository,
            sourceRevision: GTurboFormatV2.qwenRevision,
            verificationTimestamp: "fixture",
            toolVersion: "TurboFieldfareAppCoreTests",
            files: [
                "manifest.json": .init(
                    size: UInt64(manifestData.count), sha256: manifestHash),
                "model_weights.bin": .init(
                    size: UInt64(weights.count), sha256: weightsHash),
                "packed_experts/layout.json": .init(
                    size: UInt64(layout.count), sha256: layoutHash),
            ])
        try JSONEncoder().encode(receipt).write(
            to: directory.appendingPathComponent(VerifiedInstallReceiptReader.fileName),
            options: .atomic)
        return directory
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func makeTemporaryRoot(_ tag: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("app-qwen-\(tag)-\(UUID().uuidString).gturbo",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }
}

private struct SyntheticInstallationError: Error, Equatable, Sendable {}
