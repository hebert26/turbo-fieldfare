import Foundation

enum VerifiedInstallReceiptWriter {
    static let fileName = "verified-install.json"
    /// Reproducible v2 conversion receipts deliberately carry an epoch marker
    /// instead of pretending that a wall-clock value is part of model identity.
    /// The real verification time belongs in the operator evidence log.
    static let deterministicV2Timestamp = "1970-01-01T00:00:00Z"

    struct ConversionProvenance: Sendable, Equatable {
        let sourceIndexSHA256: String
        let sourcePayloadSHA256: String
        let planFingerprint: String
        let quantizationPolicySHA256: String
        let converterVersion: String
        let compatibleTextManifestSHA256: String?
    }

    static func encode(outputDir: String,
                              manifestSha256: String,
                              manifestSize: UInt64,
                              sourceRepoID: String?,
                              sourceRevision: String?,
                              toolVersion: String = "TurboFieldfareRepack",
                              verificationTimestamp: String? = nil,
                              conversionProvenance: ConversionProvenance? = nil,
                              files: [RepackAudit.OutputFile]) throws -> Data {
        let modelDirectoryPath: String
        if conversionProvenance != nil {
            modelDirectoryPath = try canonicalPhysicalModelDirectoryPath(outputDir)
        } else {
            modelDirectoryPath = URL(fileURLWithPath: outputDir).standardizedFileURL.path
        }
        var filesDict: [String: Any] = [:]
        for file in files {
            filesDict[file.relativePath] = [
                "size": file.size,
                "sha256": file.sha256
            ]
        }
        filesDict["manifest.json"] = [
            "size": manifestSize,
            "sha256": manifestSha256
        ]

        var receipt: [String: Any] = [
            "schemaVersion": 1,
            "manifestSha256": manifestSha256,
            "modelDirectoryPath": modelDirectoryPath,
            "verificationTimestamp": verificationTimestamp
                ?? ISO8601DateFormatter().string(from: Date()),
            "toolVersion": toolVersion,
            "files": filesDict
        ]
        if let sourceRepoID {
            receipt["sourceRepoID"] = sourceRepoID
        }
        if let sourceRevision {
            receipt["sourceRevision"] = sourceRevision
        }
        if let conversionProvenance {
            receipt["sourceIndexSHA256"] = conversionProvenance.sourceIndexSHA256
            receipt["sourcePayloadSHA256"] = conversionProvenance.sourcePayloadSHA256
            receipt["planFingerprint"] = conversionProvenance.planFingerprint
            receipt["quantizationPolicySHA256"] =
                conversionProvenance.quantizationPolicySHA256
            receipt["converterVersion"] = conversionProvenance.converterVersion
            receipt["verificationTimePolicy"] = "deterministic-epoch; see operational evidence"
            if let compatible = conversionProvenance.compatibleTextManifestSHA256 {
                receipt["compatibleTextManifestSHA256"] = compatible
            }
        }
        return try JSONSerialization.data(withJSONObject: receipt,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    static func canonicalPhysicalModelDirectoryPath(_ path: String) throws -> String {
        let standardized = URL(fileURLWithPath: path).standardizedFileURL
        let basename = standardized.lastPathComponent.precomposedStringWithCanonicalMapping
        guard !basename.isEmpty,
              basename != ".",
              basename != "..",
              !basename.contains("/"),
              !basename.contains("\0") else {
            throw RepackError.installPathUnsafe(
                path: path, detail: "invalid receipt directory basename")
        }
        switch try Posix.entryKind(standardized.path) {
        case .directory, .symlink:
            return try Posix.physicalPath(standardized.path)
        case .absent:
            let parent = try Posix.physicalPath(
                standardized.deletingLastPathComponent().path)
            return (parent as NSString).appendingPathComponent(basename)
        case .regular, .other:
            throw RepackError.installPathUnsafe(
                path: path, detail: "receipt directory has the wrong type")
        }
    }
}
