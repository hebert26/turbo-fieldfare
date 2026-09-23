import Foundation
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfareMacPresentation
import TurboFieldfareDecodeProtocol

@Suite("App model picker presentation")
struct AppModelPickerPresentationTests {
    @Test func verifiedGemmaFamilyIdentityDoesNotInventSourceRevision() {
        let entry = AppModelCatalog.entry(for: .gemma4)
        let presentation = AppModelIdentityPresentation.resolve(
            selected: entry,
            installationStatus: .complete,
            loadState: Self.readyState(for: entry),
            transition: .idle,
            readiness: .gemma(toolThinkingEnabled: true))

        #expect(presentation.selectedName == entry.displayName)
        #expect(presentation.selectedState == "Loaded and verified")
        #expect(presentation.loadedName == entry.displayName)
        #expect(presentation.loadedDetail
            == "Verified Gemma family · the legacy runtime does not report a model ID or source revision")
        #expect(!(presentation.loadedDetail?.contains(entry.sourceIdentity.revision) ?? false))
        #expect(presentation.accessibilityValue.contains("Selected: \(entry.displayName)"))
        #expect(presentation.accessibilityValue.contains("Loaded and verified: \(entry.displayName)"))
    }

    @Test func verifiedQwenIdentityKeepsSourceAndQuantizationDetails() {
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        let presentation = AppModelIdentityPresentation.resolve(
            selected: entry,
            installationStatus: .complete,
            loadState: Self.readyState(for: entry),
            transition: .idle,
            readiness: Self.qwenReadiness(for: entry))

        #expect(presentation.selectedName == "Qwen3.6 35B-A3B")
        #expect(presentation.selectedState == "Loaded and verified")
        #expect(presentation.loadedName == entry.sourceIdentity.repoID)
        #expect(presentation.loadedDetail
            == "Source revision \(entry.sourceIdentity.revision) · format v2.0 · MTP int4 group 32 scales fp8")
        #expect(presentation.accessibilityValue.contains("Loaded and verified: \(entry.sourceIdentity.repoID)"))
    }

    @Test func failedUnloadedOrMismatchedRuntimeNeverShowsAStaleLoadedIdentity() {
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        let readiness = Self.qwenReadiness(for: entry)
        let states: [AppModelLoadState] = [
            .notLoaded,
            .unloading,
            .failed(.modelLoadFailed("replacement rejected")),
        ]

        for state in states {
            let presentation = AppModelIdentityPresentation.resolve(
                selected: entry,
                installationStatus: .complete,
                loadState: state,
                transition: .idle,
                readiness: readiness)

            #expect(presentation.loadedName == nil)
            #expect(presentation.loadedDetail == nil)
            #expect(presentation.accessibilityValue.contains("No verified model loaded"))
        }

        let mismatched = AppModelIdentityPresentation.resolve(
            selected: entry,
            installationStatus: .complete,
            loadState: Self.readyState(for: entry),
            transition: .idle,
            readiness: Self.qwenReadiness(
                for: entry,
                sourceRevision: "different-source-revision"))

        #expect(mismatched.loadedName == nil)
        #expect(mismatched.loadedDetail == nil)
        #expect(mismatched.selectedState == "Waiting for verified runtime identity")
        #expect(mismatched.accessibilityValue.contains("No verified model loaded"))
    }

    @Test func pickerTransitionsExplainLoadingSwitchingAndGemmaRollback() {
        let qwen = AppModelCatalog.entry(for: .qwen3_6)
        let gemma = AppModelCatalog.entry(for: .gemma4)

        let loading = AppModelIdentityPresentation.resolve(
            selected: qwen,
            installationStatus: .complete,
            loadState: Self.readyState(for: qwen),
            transition: .loading(.qwen3_6),
            readiness: Self.qwenReadiness(for: qwen))
        #expect(loading.selectedState == "Loading \(qwen.displayName)")
        #expect(loading.accessibilityValue.contains("Loading \(qwen.displayName)"))
        #expect(loading.loadedName == nil)
        #expect(loading.loadedDetail == nil)
        #expect(loading.accessibilityValue.contains("No verified model loaded"))

        let switching = AppModelIdentityPresentation.resolve(
            selected: qwen,
            installationStatus: .complete,
            loadState: Self.readyState(for: qwen),
            transition: .unloading(from: .gemma4, to: .qwen3_6),
            readiness: Self.qwenReadiness(for: qwen))
        #expect(switching.selectedState == "Switching to \(qwen.displayName)")
        #expect(switching.accessibilityValue.contains("Switching to \(qwen.displayName)"))
        #expect(switching.loadedName == nil)
        #expect(switching.loadedDetail == nil)
        #expect(switching.accessibilityValue.contains("No verified model loaded"))

        let rollback = AppModelIdentityPresentation.resolve(
            selected: gemma,
            installationStatus: .complete,
            loadState: Self.readyState(for: gemma),
            transition: .unloading(from: .qwen3_6, to: .gemma4),
            readiness: .gemma(toolThinkingEnabled: true))
        #expect(rollback.selectedState == "Switching to \(gemma.displayName)")
        #expect(rollback.accessibilityValue.contains("Switching to \(gemma.displayName)"))
        #expect(rollback.loadedName == nil)
        #expect(rollback.loadedDetail == nil)
        #expect(rollback.accessibilityValue.contains("No verified model loaded"))

        let failed = AppModelIdentityPresentation.resolve(
            selected: qwen,
            installationStatus: .complete,
            loadState: Self.readyState(for: qwen),
            transition: .failed(.qwen3_6, "verification unavailable"),
            readiness: Self.qwenReadiness(for: qwen))
        #expect(failed.selectedState == "Could not load \(qwen.displayName): verification unavailable")
        #expect(failed.accessibilityValue.contains("Could not load \(qwen.displayName)"))
        #expect(failed.loadedName == nil)
        #expect(failed.loadedDetail == nil)
        #expect(failed.accessibilityValue.contains("No verified model loaded"))
    }

    @Test func missingSelectionExplainsThatThePickerCannotLoadIt() {
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        let presentation = AppModelIdentityPresentation.resolve(
            selected: entry,
            installationStatus: .missing,
            loadState: .notLoaded,
            transition: .idle,
            readiness: nil)

        #expect(presentation.selectedState == "Not installed")
        #expect(presentation.loadedName == nil)
        #expect(presentation.accessibilityValue
            == "Selected: \(entry.displayName). Not installed. No verified model loaded.")
    }

    @Test func GemmaThinkingChoiceDoesNotInventASecondLoadedIdentity() {
        let entry = AppModelCatalog.entry(for: .gemma4)
        let enabled = AppModelIdentityPresentation.resolve(
            selected: entry,
            installationStatus: .complete,
            loadState: Self.readyState(for: entry),
            transition: .idle,
            readiness: .gemma(toolThinkingEnabled: true))
        let disabled = AppModelIdentityPresentation.resolve(
            selected: entry,
            installationStatus: .complete,
            loadState: Self.readyState(for: entry),
            transition: .idle,
            readiness: .gemma(toolThinkingEnabled: false))

        #expect(enabled.loadedName == disabled.loadedName)
        #expect(enabled.loadedDetail == disabled.loadedDetail)
        #expect(enabled.selectedState == disabled.selectedState)
    }

    private static func readyState(for entry: AppModelCatalogEntry) -> AppModelLoadState {
        .ready(modelDirectory: entry.location.textModelURL, loadSeconds: 1)
    }

    private static func qwenReadiness(
        for entry: AppModelCatalogEntry,
        sourceRevision: String? = nil
    ) -> AppLoadedModelReadiness {
        .qwen(identity: DecodeModelIdentity(
            family: .qwen3_6,
            modelID: entry.sourceIdentity.repoID,
            sourceRevision: sourceRevision ?? entry.sourceIdentity.revision,
            formatMajor: 2,
            formatMinor: 0,
            sourceIndexSHA256: entry.sourceIdentity.sourceIndexSHA256,
            quantizationPolicySHA256: "fixture-policy-sha256",
            textManifestSHA256: "fixture-manifest-sha256",
            quantization: [DecodeQuantizationIdentity(
                category: "MTP",
                storage: "int4",
                groupSize: 32,
                scaleType: "fp8")],
            vision: .unavailable))
    }
}
