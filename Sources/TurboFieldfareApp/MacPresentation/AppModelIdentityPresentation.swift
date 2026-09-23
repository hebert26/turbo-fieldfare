import Foundation
import TurboFieldfareAppCore

public struct AppModelIdentityPresentation: Equatable, Sendable {
    public let selectedName: String
    public let selectedState: String
    public let loadedName: String?
    public let loadedDetail: String?
    public let accessibilityValue: String

    public init(
        selectedName: String,
        selectedState: String,
        loadedName: String?,
        loadedDetail: String?,
        accessibilityValue: String
    ) {
        self.selectedName = selectedName
        self.selectedState = selectedState
        self.loadedName = loadedName
        self.loadedDetail = loadedDetail
        self.accessibilityValue = accessibilityValue
    }

    public static func resolve(
        selected: AppModelCatalogEntry,
        installationStatus: AppModelInstallationStatus,
        loadState: AppModelLoadState,
        transition: AppModel.ModelSelectionTransition,
        readiness: AppLoadedModelReadiness?
    ) -> Self {
        let loaded = loadedIdentity(
            selected: selected,
            loadState: loadState,
            transition: transition,
            readiness: readiness)
        let selectedState = selectedState(
            installationStatus: installationStatus,
            loadState: loadState,
            transition: transition,
            hasVerifiedLoadedIdentity: loaded != nil)
        let loadedValue = loaded.map { "Loaded and verified: \($0.name)" }
            ?? "No verified model loaded"
        return Self(
            selectedName: selected.displayName,
            selectedState: selectedState,
            loadedName: loaded?.name,
            loadedDetail: loaded?.detail,
            accessibilityValue: "Selected: \(selected.displayName). \(selectedState). \(loadedValue).")
    }

    private static func selectedState(
        installationStatus: AppModelInstallationStatus,
        loadState: AppModelLoadState,
        transition: AppModel.ModelSelectionTransition,
        hasVerifiedLoadedIdentity: Bool
    ) -> String {
        switch transition {
        case .unloading(_, let destination):
            return "Switching to \(AppModelCatalog.entry(for: destination).displayName)"
        case .loading(let modelID):
            return "Loading \(AppModelCatalog.entry(for: modelID).displayName)"
        case .failed(let modelID, let message):
            return "Could not load \(AppModelCatalog.entry(for: modelID).displayName): \(message)"
        case .idle:
            break
        }

        switch loadState {
        case .loading(let phase): return phase.label
        case .cancelling: return "Cancelling model load"
        case .unloading: return "Unloading model"
        case .failed(let error): return "Load failed: \(error.localizedDescription)"
        case .ready:
            return hasVerifiedLoadedIdentity
                ? "Loaded and verified"
                : "Waiting for verified runtime identity"
        case .notLoaded:
            break
        }

        switch installationStatus {
        case .missing: return "Not installed"
        case .partial(let message): return "Needs repair: \(message)"
        case .complete: return "Installed, not loaded"
        }
    }

    private static func loadedIdentity(
        selected: AppModelCatalogEntry,
        loadState: AppModelLoadState,
        transition: AppModel.ModelSelectionTransition,
        readiness: AppLoadedModelReadiness?
    ) -> (name: String, detail: String)? {
        guard transition == .idle, loadState.isReady, let readiness else { return nil }
        switch readiness {
        case .gemma:
            guard selected.family == .gemma4 else { return nil }
            return (
                name: selected.displayName,
                detail: "Verified Gemma family · the legacy runtime does not report a model ID or source revision")
        case .qwen(let identity):
            guard identity.family.rawValue == selected.family.rawValue,
                  selected.accepts(
                    family: selected.family,
                    modelID: identity.modelID,
                    revision: identity.sourceRevision,
                    sourceIndexSHA256: identity.sourceIndexSHA256) else {
                return nil
            }
            let quantization = identity.quantization.isEmpty
                ? "quantization unavailable"
                : identity.quantization.map { value in
                    var parts = [value.category, value.storage]
                    if let groupSize = value.groupSize {
                        parts.append("group \(groupSize)")
                    }
                    if let scaleType = value.scaleType {
                        parts.append("scales \(scaleType)")
                    }
                    if let biasType = value.biasType {
                        parts.append("bias \(biasType)")
                    }
                    return parts.joined(separator: " ")
                }.joined(separator: ", ")
            return (
                name: identity.modelID,
                detail: "Source revision \(identity.sourceRevision) · format v\(identity.formatMajor).\(identity.formatMinor) · \(quantization)")
        }
    }

}
