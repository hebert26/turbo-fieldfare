import Foundation
import TurboFieldfare
import TurboFieldfareOfficialQwenSource

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    public static func status(
        at directory: URL,
        entry: AppModelCatalogEntry
    ) -> AppModelInstallationStatus {
        do {
            switch entry.family {
            case .gemma4:
                let directory = directory.standardizedFileURL
                guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path) else {
                    return .missing
                }
                return try gemmaStatus(
                    at: directory,
                    descriptor: remoteTextDescriptor(for: entry))
            case .qwen3_6:
                // The registration binds to its physical path. Standardizing
                // /private/var to /var would invalidate a valid receipt.
                return try qwenStatus(at: directory, entry: entry)
            }
        } catch {
            return .partial("\(error)")
        }
    }

    /// Source-compatible Gemma probe retained for existing app and test call
    /// sites. The catalog-aware overload is required for family selection.
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor = .default
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }
        do {
            return try gemmaStatus(at: directory, descriptor: descriptor)
        } catch {
            return .partial("\(error)")
        }
    }

    private static func gemmaStatus(
        at directory: URL,
        descriptor: AppModelInstallDescriptor
    ) throws -> AppModelInstallationStatus {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let manifest = try ManifestReader.load(
            directoryURL: directory,
            expecting: .gemma4_26B_A4B)
        let expectedSource = "sha256:" + descriptor.sourceIndexSHA256
        guard manifest.sourceSnapshotHash == expectedSource else {
            return .partial("installed checkpoint does not match \(descriptor.displayName)")
        }
        let layout = directory.appendingPathComponent("packed_experts/layout.json")
        guard FileManager.default.fileExists(atPath: layout.path) else {
            return .partial("packed_experts/layout.json is missing")
        }
        let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
        let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
        try VerifiedInstallReceiptReader.validateManifestBinding(
            receipt,
            directoryURL: directory,
            manifestSha256: manifestHash)
        return .complete
    }

    private static func qwenStatus(
        at directory: URL,
        entry: AppModelCatalogEntry
    ) throws -> AppModelInstallationStatus {
        guard FileManager.default.fileExists(atPath: directory.path) else { return .missing }
        let descriptor = try OfficialSourceRegistration.inspect(at: directory)
        guard descriptor.repository == entry.sourceIdentity.repoID,
              descriptor.revision == entry.sourceIdentity.revision,
              descriptor.storageProfile == OfficialQwenSourceIdentity.pinned.storageProfile,
              descriptor.sidecarSHA256["model.safetensors.index.json"]
                == entry.sourceIdentity.sourceIndexSHA256 else {
            return .partial("registered BF16 source does not match \(entry.displayName)")
        }
        // A marker or syntactically valid receipt alone proves no payload.
        // Trusted reopen binds the pinned receipt to current source files.
        _ = try OfficialSourceTrust.verify(
            at: directory, policy: .sizeCheckTrustedReceipt)
        return .complete
    }

    private static func remoteTextDescriptor(
        for entry: AppModelCatalogEntry
    ) throws -> AppModelInstallDescriptor {
        guard case let .remoteRepack(text, _) = entry.installRoute else {
            throw ProbeError.invalidRoute
        }
        return text
    }

    private enum ProbeError: Error, CustomStringConvertible {
        case invalidRoute

        var description: String {
            "catalog entry has no compatible text installation route"
        }
    }
}
