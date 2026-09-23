import Foundation
import TurboFieldfare
import TurboFieldfareRepackCore

public enum AppVisionPackInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
    /// The text path cannot have an adjacent companion directory.
    case unsupportedLayout
    /// The companion is verified, but this Mac cannot run Qwen vision. This is
    /// distinct from both a usable installation and a damaged pack.
    case verificationUnavailable(String)
}

public enum AppVisionPackInstallationProbe {
    /// Source-compatible Gemma probe retained for existing callers.
    public static func status(at textModelDirectory: URL) -> AppVisionPackInstallationStatus {
        gemmaStatus(at: textModelDirectory)
    }

    public static func status(
        at textModelDirectory: URL,
        entry: AppModelCatalogEntry
    ) -> AppVisionPackInstallationStatus {
        switch entry.family {
        case .gemma4:
            return gemmaStatus(at: textModelDirectory)
        case .qwen3_6:
            return qwenStatus(at: textModelDirectory, entry: entry)
        }
    }

    private static func gemmaStatus(
        at textModelDirectory: URL
    ) -> AppVisionPackInstallationStatus {
        let textModelDirectory = textModelDirectory.standardizedFileURL
        let companion: URL
        do {
            companion = try VisionPackLocation.companionURL(forTextModel: textModelDirectory)
        } catch {
            return .unsupportedLayout
        }
        guard FileManager.default.fileExists(atPath: companion.path) else {
            return .missing
        }
        do {
            _ = try VisionPackVerifier.verify(
                directory: companion,
                textModelDirectory: textModelDirectory,
                verifyWeights: false)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }

    private static func qwenStatus(
        at textModelDirectory: URL,
        entry: AppModelCatalogEntry
    ) -> AppVisionPackInstallationStatus {
        let textModelDirectory = textModelDirectory.standardizedFileURL
        let companion: URL
        do {
            companion = try VisionPackLocation.companionURL(
                forTextModel: textModelDirectory)
        } catch {
            return .unsupportedLayout
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: companion.path,
            isDirectory: &isDirectory), isDirectory.boolValue else {
            return .missing
        }
        do {
            let admission = try ModelFamilyGenerationSession.inspect(
                directoryURL: textModelDirectory)
            guard admission.family == .qwen3_6,
                  let identity = admission.verifiedIdentity,
                  entry.accepts(
                    family: .qwen3_6,
                    modelID: identity.modelID,
                    revision: identity.sourceRevision,
                    sourceIndexSHA256: identity.sourceIndexSHA256) else {
                return .partial("text model identity does not match \(entry.displayName)")
            }
            switch try ModelFamilyGenerationSession.inspectQwenVisionCompanion(
                directoryURL: textModelDirectory,
                loadedIdentity: identity,
                visionPackURL: companion) {
            case .ready:
                return .complete
            case .missing:
                return .missing
            case .invalid:
                return .partial("vision companion is not bound to the selected Qwen model")
            case .unsupported:
                // The runtime helper has fully verified the companion before
                // checking hardware. Keep unsupported distinct from usable.
                return .verificationUnavailable(
                    "vision companion is verified but unsupported on this Mac")
            }
        } catch {
            return .partial("\(error)")
        }
    }
}
