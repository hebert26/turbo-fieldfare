import Foundation
import TurboFieldfare

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
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            switch entry.family {
            case .gemma4:
                return try gemmaStatus(
                    at: directory,
                    descriptor: remoteTextDescriptor(for: entry))
            case .qwen3_6:
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
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let loaded = try ManifestReader.loadVerified(directoryURL: directory)
        guard loaded.descriptor.family == .qwen3_6,
              entry.accepts(
                family: .qwen3_6,
                modelID: loaded.descriptor.modelID,
                revision: loaded.descriptor.sourceRevision,
                sourceIndexSHA256: loaded.descriptor.sourceIndexSHA256) else {
            return .partial("installed checkpoint does not match \(entry.displayName)")
        }

        let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
        let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
        try VerifiedInstallReceiptReader.validateManifestBinding(
            receipt,
            directoryURL: directory,
            manifestSha256: manifestHash)
        guard receipt.sourceRepoID == entry.sourceIdentity.repoID,
              receipt.sourceRevision == entry.sourceIdentity.revision else {
            return .partial("verified install receipt does not match \(entry.displayName)")
        }

        let expectedFiles = Set(loaded.files.keys).union(["manifest.json"])
        guard Set(receipt.files.keys) == expectedFiles else {
            return .partial("verified install receipt file set does not match manifest")
        }
        let manifestAttributes = try FileManager.default.attributesOfItem(
            atPath: manifestURL.path)
        guard let manifestReceipt = receipt.files["manifest.json"],
              manifestReceipt.sha256.lowercased() == manifestHash.lowercased(),
              let manifestSize = manifestAttributes[.size] as? NSNumber,
              manifestReceipt.size == manifestSize.uint64Value else {
            return .partial("verified install receipt manifest entry is invalid")
        }
        for (path, file) in loaded.files {
            guard let receiptFile = receipt.files[path],
                  receiptFile.size == file.size,
                  receiptFile.sha256.lowercased() == file.sha256.lowercased() else {
                return .partial("verified install receipt does not match \(path)")
            }
            let attributes = try FileManager.default.attributesOfItem(
                atPath: directory.appendingPathComponent(path).path)
            guard let size = attributes[.size] as? NSNumber,
                  size.uint64Value == file.size else {
                return .partial("installed file size does not match \(path)")
            }
        }
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
