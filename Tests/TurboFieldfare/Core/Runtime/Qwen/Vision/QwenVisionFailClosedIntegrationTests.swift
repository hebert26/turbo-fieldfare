import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat

/// Opt-in P23 admission and fail-closed checks for the installed Qwen text
/// artifact and its explicitly selected vision companion. These tests stop at metadata,
/// binding, or media admission boundaries and never construct a model or run
/// vision kernels.
@Suite("Qwen real vision fail-closed", .serialized)
struct QwenVisionFailClosedIntegrationTests {
    @Test(.enabled(if: RealQwenVisionArtifact.isOptedIn,
                   "set the explicit text and vision artifact paths to run P23 real-artifact checks"))
    func selectedCompanionStatusAndExplicitMissingPathAreRecorded() throws {
        let artifact = try RealQwenVisionArtifact.load()
        let selectedStatus = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: artifact.directory,
            loadedIdentity: artifact.identity,
            visionPackURL: artifact.visionDirectory)
        #expect(selectedStatus == .ready || selectedStatus == .unsupported)
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "selected-companion-status",
            category: selectedStatus.rawValue,
            detail: "path=\(artifact.visionDirectory.path)")

        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("p23-missing-qwen-vision-\(UUID().uuidString)",
                                   isDirectory: true)
            .appendingPathComponent("missing.vision.gturbo", isDirectory: true)
        var observed: VisionPackError?
        do {
            _ = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
                directoryURL: artifact.directory,
                loadedIdentity: artifact.identity,
                visionPackURL: missing)
            Issue.record("explicit missing companion was accepted")
        } catch let error as VisionPackError {
            observed = error
        }
        #expect(observed == .packNotFound(missing.standardizedFileURL.path))
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "explicit-missing-companion",
            category: "packNotFound",
            detail: observed.map { String(describing: $0) } ?? "no error")
    }

    @Test(.enabled(if: RealQwenVisionArtifact.isOptedIn,
                   "set the explicit text and vision artifact paths to run P23 real-artifact checks"))
    func wrongTextBindingIsInvalidBeforeHardwareStatus() throws {
        let artifact = try RealQwenVisionArtifact.load()
        guard let companion = try artifact.validatedCompanionForMetadataMutation() else {
            return
        }
        let staleDigest = String(repeating: "f", count: 64)
        let malformed = try Self.makeMetadataOnlyCompanion(
            from: companion,
            replacing: "compatibleTextManifestSHA256",
            with: staleDigest)
        defer { Self.removeOwned(malformed, label: "wrong-text-binding") }

        var observed: VisionPackError?
        do {
            _ = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
                directoryURL: artifact.directory,
                loadedIdentity: artifact.identity,
                visionPackURL: malformed)
            Issue.record("companion with stale text binding was accepted")
        } catch let error as VisionPackError {
            observed = error
        }
        #expect(observed == .incompatibleTextArtifact)
        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: artifact.directory,
            loadedIdentity: artifact.identity,
            visionPackURL: malformed)
        #expect(status == .invalid)
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "wrong-text-binding",
            category: status.rawValue,
            detail: observed.map { String(describing: $0) } ?? "no error")
    }

    @Test(.enabled(if: RealQwenVisionArtifact.isOptedIn,
                   "set the explicit text and vision artifact paths to run P23 real-artifact checks"))
    func wrongProcessorIdentityIsInvalidBeforeHardwareStatus() throws {
        let artifact = try RealQwenVisionArtifact.load()
        guard let companion = try artifact.validatedCompanionForMetadataMutation() else {
            return
        }
        let malformed = try Self.makeMetadataOnlyCompanion(
            from: companion,
            replacing: "processorConfigSHA256",
            with: String(repeating: "e", count: 64))
        defer { Self.removeOwned(malformed, label: "wrong-processor-identity") }

        var observed: VisionPackError?
        do {
            _ = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
                directoryURL: artifact.directory,
                loadedIdentity: artifact.identity,
                visionPackURL: malformed)
            Issue.record("companion with stale processor identity was accepted")
        } catch let error as VisionPackError {
            observed = error
        }
        guard case .some(.invalidMetadata(let detail)) = observed else {
            Issue.record("unexpected processor-identity error: \(String(describing: observed))")
            return
        }
        #expect(detail.contains("vision.manifest.processor"))
        #expect(detail.contains("unsupported processor or media"))
        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: artifact.directory,
            loadedIdentity: artifact.identity,
            visionPackURL: malformed)
        #expect(status == .invalid)
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "wrong-processor-identity",
            category: status.rawValue,
            detail: observed.map { String(describing: $0) } ?? "no error")
    }

    @Test(.enabled(if: RealQwenVisionArtifact.isOptedIn,
                   "set the explicit text and vision artifact paths to run P23 real-artifact checks"))
    func wrongRevisionAndProcessorProfileAreRejectedBeforeFiles() throws {
        let artifact = try RealQwenVisionArtifact.load()
        let revision = try Self.makeMetadataOnlyCompanion(
            from: artifact.visionDirectory,
            replacing: "sourceRevision",
            with: "not-the-pinned-qwen-revision")
        defer { Self.removeOwned(revision, label: "wrong-revision") }
        var revisionError: VisionPackError?
        do {
            _ = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
                directoryURL: artifact.directory,
                loadedIdentity: artifact.identity,
                visionPackURL: revision)
            Issue.record("wrong revision companion was accepted")
        } catch let error as VisionPackError {
            revisionError = error
        }
        guard case .some(.invalidMetadata(let revisionDetail)) = revisionError else {
            Issue.record("unexpected wrong-revision error: \(String(describing: revisionError))")
            return
        }
        #expect(revisionDetail.contains("vision.manifest.family"))
        #expect(revisionDetail.contains("not pinned Qwen"))
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "wrong-revision",
            category: "invalidMetadata",
            detail: revisionDetail)

        let profile = try Self.makeMetadataOnlyCompanion(
            from: artifact.visionDirectory,
            mutate: { object in
                var changed = object
                var processor = changed["processorProfile"] as? [String: Any] ?? [:]
                processor["patchSize"] = 2
                changed["processorProfile"] = processor
                return changed
            })
        defer { Self.removeOwned(profile, label: "wrong-processor-profile") }
        var profileError: VisionPackError?
        do {
            _ = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
                directoryURL: artifact.directory,
                loadedIdentity: artifact.identity,
                visionPackURL: profile)
            Issue.record("wrong processor profile companion was accepted")
        } catch let error as VisionPackError {
            profileError = error
        }
        guard case .some(.invalidMetadata(let profileDetail)) = profileError else {
            Issue.record("unexpected wrong-profile error: \(String(describing: profileError))")
            return
        }
        #expect(profileDetail.contains("vision.manifest.processor"))
        #expect(profileDetail.contains("unsupported processor or media"))
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "wrong-processor-profile",
            category: "invalidMetadata",
            detail: profileDetail)
    }

    @Test(.enabled(if: RealQwenVisionArtifact.isOptedIn,
                   "set the explicit text and vision artifact paths to run P23 real-artifact checks"))
    func corruptReceiptAndRegionMetadataFailBeforeFiles() throws {
        let artifact = try RealQwenVisionArtifact.load()
        let receiptFixture = try Self.makeMinimalCompanion(
            textManifestSHA256: artifact.identity.textManifestSHA256,
            regionMode: .valid,
            receiptMode: .wrongManifestBinding)
        defer { Self.removeOwned(receiptFixture, label: "corrupt-receipt") }
        var receiptError: VisionPackError?
        do {
            _ = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
                directoryURL: artifact.directory,
                loadedIdentity: artifact.identity,
                visionPackURL: receiptFixture)
            Issue.record("wrongly bound receipt was accepted")
        } catch let error as VisionPackError {
            receiptError = error
        }
        guard case .some(.invalidReceipt(let receiptDetail)) = receiptError else {
            Issue.record("unexpected corrupt-receipt error: \(String(describing: receiptError))")
            return
        }
        #expect(receiptDetail.contains("manifest SHA mismatch"))
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "corrupt-receipt-binding",
            category: "invalidReceipt",
            detail: receiptError.map { String(describing: $0) } ?? "no error")

        let regionFixture = try Self.makeMinimalCompanion(
            textManifestSHA256: artifact.identity.textManifestSHA256,
            regionMode: .overlapping,
            receiptMode: .none)
        defer { Self.removeOwned(regionFixture, label: "corrupt-region-metadata") }
        var regionError: VisionPackError?
        do {
            _ = try ModelFamilyGenerationSession.openVerifiedQwenVisionCompanion(
                directoryURL: artifact.directory,
                loadedIdentity: artifact.identity,
                visionPackURL: regionFixture)
            Issue.record("overlapping region metadata was accepted")
        } catch let error as VisionPackError {
            regionError = error
        }
        guard case .some(.invalidMetadata(let regionDetail)) = regionError else {
            Issue.record("unexpected corrupt-region error: \(String(describing: regionError))")
            return
        }
        #expect(regionDetail.contains("vision.tensorRegions.range"))
        #expect(regionDetail.contains("overlapping regions"))
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "corrupt-region-metadata",
            category: "invalidMetadata",
            detail: regionDetail)
    }

    @Test(.enabled(if: RealQwenVisionArtifact.isOptedIn,
                   "set the explicit text and vision artifact paths to run P23 real-artifact checks"))
    func preprocessorVideoInputIsRejectedBeforeImageDecode() throws {
        let artifact = try RealQwenVisionArtifact.load()
        let device = try #require(MTLCreateSystemDefaultDevice())
        let preprocessor = QwenImagePreprocessor(device: device)
        let unopenedVideo = FileManager.default.temporaryDirectory
            .appendingPathComponent("p23-unopened-video-\(UUID().uuidString).mp4")
        var observed: QwenVisionError?
        do {
            _ = try preprocessor.plan(fileURL: unopenedVideo, media: .video)
            Issue.record("video input was accepted")
        } catch let error as QwenVisionError {
            observed = error
        }
        #expect(observed == .unsupportedVideo)
        RealQwenVisionEvidence.record(
            artifact: artifact,
            caseID: "preprocessor-video-rejected-before-decode",
            category: "unsupportedVideo-preprocessor-only",
            detail: "direct preprocessor boundary; CLI/server video admission is covered by their existing selectors. "
                + (observed.map { String(describing: $0) } ?? "no error"))
    }

    private static func makeMetadataOnlyCompanion(
        from source: URL,
        replacing key: String,
        with value: Any
    ) throws -> URL {
        let sourceManifest = source.appendingPathComponent("manifest.json")
        guard let sourceObject = try JSONSerialization.jsonObject(
            with: Data(contentsOf: sourceManifest)) as? [String: Any] else {
            throw RealQwenVisionTestError.sourceManifestNotObject
        }
        guard sourceObject[key] != nil else {
            throw RealQwenVisionTestError.missingManifestField(key)
        }
        return try makeMetadataOnlyCompanion(from: source) { object in
            var changed = object
            changed[key] = value
            return changed
        }
    }

    private static func makeMetadataOnlyCompanion(
        from source: URL,
        mutate: ([String: Any]) -> [String: Any]
    ) throws -> URL {
        let sourceManifest = source.appendingPathComponent("manifest.json")
        guard var object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: sourceManifest)) as? [String: Any] else {
            throw RealQwenVisionTestError.sourceManifestNotObject
        }
        let originalKeys = Set(object.keys)
        object = mutate(object)
        guard Set(object.keys) == originalKeys else {
            throw RealQwenVisionTestError.missingManifestField("mutation changed manifest keys")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("p23-malformed-qwen-vision-\(UUID().uuidString)",
                                   isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        do {
            let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            try data.write(to: root.appendingPathComponent("manifest.json"), options: .atomic)
        } catch {
            removeOwned(root, label: "metadata-only-companion-write-failure")
            throw error
        }
        return root
    }

    private enum MinimalRegionMode: Equatable {
        case valid
        case overlapping
    }

    private enum MinimalReceiptMode {
        case none
        case wrongManifestBinding
    }

    private static func makeMinimalCompanion(
        textManifestSHA256: String,
        regionMode: MinimalRegionMode,
        receiptMode: MinimalReceiptMode
    ) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("p23-minimal-qwen-vision-\(UUID().uuidString)",
                                   isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let zeroSHA = String(repeating: "0", count: 64)
        let processorSHA = "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
        let weightsSize = 16_386
        var regions: [[String: Any]] = [[
            "name": "model.visual.patch_embed.proj.weight",
            "file": "vision_weights.bin",
            "offset": 16_384,
            "size": 2,
            "shape": [1],
            "storage": "bf16",
            "quantizationCategory": NSNull(),
        ]]
        if regionMode == .overlapping {
            regions.append([
                "name": "model.visual.patch_embed.proj.bias",
                "file": "vision_weights.bin",
                "offset": 16_384,
                "size": 2,
                "shape": [1],
                "storage": "bf16",
                "quantizationCategory": NSNull(),
            ])
        }
        let manifest: [String: Any] = [
            "magic": "GTURBO-VISION",
            "artifactKind": "qwen3_6_vision_companion",
            "versionMajor": 2,
            "versionMinor": 0,
            "family": "qwen3_6",
            "modelID": RealQwenVisionArtifact.modelID,
            "sourceRevision": RealQwenVisionArtifact.revision,
            "processorProfile": [
                "processorClass": "Qwen3VLProcessor",
                "imageProcessorType": "Qwen2VLImageProcessorFast",
                "patchSize": 16,
                "temporalPatchSize": 2,
                "spatialMergeSize": 2,
            ],
            "processorConfigSHA256": processorSHA,
            "compatibleTextManifestSHA256": textManifestSHA256,
            "visionPayloadSHA256": zeroSHA,
            "supportsStillImages": true,
            "supportsVideo": false,
            "files": [
                "vision_weights.bin": ["size": weightsSize, "sha256": zeroSHA],
                "preprocessor_config.json": ["size": 2, "sha256": processorSHA],
            ],
            "tensorRegions": regions,
        ]
        do {
            try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
                .write(to: root.appendingPathComponent("manifest.json"), options: .atomic)
            if case .wrongManifestBinding = receiptMode {
                let receipt: [String: Any] = [
                    "schemaVersion": 1,
                    "manifestSha256": String(repeating: "f", count: 64),
                    "modelDirectoryPath": root.path,
                    "sourceRepoID": RealQwenVisionArtifact.modelID,
                    "sourceRevision": RealQwenVisionArtifact.revision,
                    "verificationTimestamp": "2026-09-22T00:00:00Z",
                    "toolVersion": "p23-fixture",
                    "files": [
                        "manifest.json": ["size": 1, "sha256": zeroSHA],
                        "vision_weights.bin": ["size": weightsSize, "sha256": zeroSHA],
                        "preprocessor_config.json": ["size": 2, "sha256": processorSHA],
                    ],
                ]
                try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
                    .write(to: root.appendingPathComponent("verified-install.json"), options: .atomic)
            }
        } catch {
            removeOwned(root, label: "minimal-companion-write-failure")
            throw error
        }
        return root
    }

    private static func removeOwned(_ url: URL, label: String) {
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
        } catch {
            Issue.record("owned fixture cleanup failed (\(label)): \(error)")
        }
    }
}

private struct RealQwenVisionArtifact {
    static let environmentKey = "TURBO_FIELDFARE_REAL_QWEN_ARTIFACT"
    static let visionEnvironmentKey = "TURBO_FIELDFARE_REAL_QWEN_VISION_PACK"
    static let modelID = "Qwen/Qwen3.6-35B-A3B"
    static let revision = "995ad96eacd98c81ed38be0c5b274b04031597b0"

    static var isOptedIn: Bool {
        let environment = ProcessInfo.processInfo.environment
        return [environmentKey, visionEnvironmentKey].allSatisfy { key in
            guard let value = environment[key] else { return false }
            return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    let directory: URL
    let visionDirectory: URL
    let identity: LoadedRuntimeIdentity

    static func load() throws -> Self {
        guard let raw = ProcessInfo.processInfo.environment[environmentKey],
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RealQwenVisionTestError.notOptedIn
        }
        guard let visionRaw = ProcessInfo.processInfo.environment[visionEnvironmentKey],
              !visionRaw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RealQwenVisionTestError.notOptedIn
        }
        let directory = URL(fileURLWithPath: raw, isDirectory: true)
            .standardizedFileURL
            .resolvingSymlinksInPath().standardizedFileURL
        let visionDirectory = URL(fileURLWithPath: visionRaw, isDirectory: true)
            .standardizedFileURL
            .resolvingSymlinksInPath().standardizedFileURL
        guard directory != visionDirectory else {
            throw RealQwenVisionTestError.identityMismatch
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw RealQwenVisionTestError.missingTextArtifact(directory.path)
        }
        var isVisionDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: visionDirectory.path, isDirectory: &isVisionDirectory),
              isVisionDirectory.boolValue else {
            throw RealQwenVisionTestError.missingVisionArtifact(visionDirectory.path)
        }
        let admission = try ModelFamilyGenerationSession.inspect(directoryURL: directory)
        guard let identity = admission.verifiedIdentity,
              identity.family == .qwen3_6,
              identity.modelID == modelID,
              identity.sourceRevision == revision else {
            throw RealQwenVisionTestError.identityMismatch
        }
        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: directory,
            loadedIdentity: identity,
            visionPackURL: visionDirectory)
        guard status == .ready || status == .unsupported else {
            throw RealQwenVisionTestError.invalidVisionArtifact(status.rawValue)
        }
        return Self(directory: directory, visionDirectory: visionDirectory, identity: identity)
    }

    func validatedCompanionForMetadataMutation() throws -> URL? {
        let companion = visionDirectory
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: companion.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            RealQwenVisionEvidence.record(
                artifact: self,
                caseID: "metadata-mutation-prerequisite",
                category: "gap",
                detail: "selected explicit vision pack is absent at \(companion.path)")
            Issue.record("selected explicit vision pack is absent for metadata-only mutation")
            return nil
        }
        let status = try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
            directoryURL: directory,
            loadedIdentity: identity,
            visionPackURL: companion)
        guard status == .ready || status == .unsupported else {
            RealQwenVisionEvidence.record(
                artifact: self,
                caseID: "metadata-mutation-prerequisite",
                category: "gap",
                detail: "selected explicit companion status=\(status.rawValue)")
            Issue.record("selected real companion is not valid: \(status.rawValue)")
            return nil
        }
        return companion
    }
}

private enum RealQwenVisionTestError: Error, CustomStringConvertible {
    case notOptedIn
    case missingTextArtifact(String)
    case missingVisionArtifact(String)
    case invalidVisionArtifact(String)
    case identityMismatch
    case sourceManifestNotObject
    case missingManifestField(String)

    var description: String {
        switch self {
        case .notOptedIn: "real Qwen vision opt-in is missing"
        case .missingTextArtifact(let path): "text artifact is not a directory: \(path)"
        case .missingVisionArtifact(let path): "vision artifact is not a directory: \(path)"
        case .invalidVisionArtifact(let status): "explicit vision artifact is not valid: \(status)"
        case .identityMismatch: "selected artifact is not the pinned Qwen3.6 identity"
        case .sourceManifestNotObject: "companion manifest is not a JSON object"
        case .missingManifestField(let field): "companion manifest lacks \(field)"
        }
    }
}

private enum RealQwenVisionEvidence {
    static func record(
        artifact: RealQwenVisionArtifact,
        caseID: String,
        category: String,
        detail: String
    ) {
        let object: [String: Any] = [
            "schema": "p23-real-vision-fail-closed-v1",
            "case": caseID,
            "category": category,
            "detail": detail,
            "modelID": artifact.identity.modelID,
            "sourceRevision": artifact.identity.sourceRevision,
            "textManifestSHA256": artifact.identity.textManifestSHA256,
            "textArtifact": artifact.directory.path,
            "visionArtifact": artifact.visionDirectory.path,
            "visionManifestSHA256": manifestSHA256(artifact.visionDirectory) ?? "unknown",
            "visionProcessorConfigSHA256": visionField(
                artifact.visionDirectory, key: "processorConfigSHA256") ?? "unknown",
            "visionPayloadSHA256": visionField(
                artifact.visionDirectory, key: "visionPayloadSHA256") ?? "unknown",
            "visionReceiptSHA256": receiptSHA256(artifact.visionDirectory) ?? "unknown",
        ]
        let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        let line = data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        print("P23_REAL_VISION_FAIL_CLOSED \(line)")
    }

    private static func manifestSHA256(_ directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("manifest.json")) else {
            return nil
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func visionField(_ directory: URL, key: String) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return object[key] as? String
    }

    private static func receiptSHA256(_ directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("verified-install.json")) else {
            return nil
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
