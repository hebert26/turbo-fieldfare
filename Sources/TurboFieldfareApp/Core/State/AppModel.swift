import Foundation
import Synchronization
import TurboFieldfare
import TurboFieldfareRepackCore
import TurboFieldfare
import Observation
import TurboFieldfareDecodeProtocol

@MainActor
@Observable
public final class AppModel {
    public enum RunState: Equatable {
        case idle
        case running
    }

    public var modelPathText: String
    public var promptText: String = ""
    public private(set) var imageAttachments: [AppImageAttachment] = []
    public private(set) var imageAttachmentError: String?
    /// A count, not a flag. The picker and a drop can both be staging at once,
    /// and whichever finished first cleared a shared Bool — reopening `canRun`
    /// while the other was still copying, so Generate ran against a partial set
    /// and the remaining images were appended after the run had snapshotted its
    /// own, where `removeImage` and `clearImages` are no-ops.
    private var addingImagesCount = 0
    public var isAddingImages: Bool { addingImagesCount > 0 }
    /// Set by the Model menu's Remove Image Support item; the window presents
    /// the confirmation.
    public var isConfirmingVisionPackRemoval = false
    public private(set) var outputPromptText: String = ""
    public private(set) var outputImageAttachments: [AppImageAttachment] = []
    /// The open chat. The transcript renders it, and its turn order is what the
    /// decode service's gate checks every turn against.
    public private(set) var conversation = AppConversation()
    /// Turns from conversations whose KV no longer exists — a reload or an
    /// unload took it. They stay on screen because the app deliberately keeps a
    /// transcript across lifecycle actions, but they are not in the model's
    /// context any more, and the transcript draws a break to say so. Keeping
    /// them here rather than in `conversation` is what preserves that type's
    /// invariant: its turns are exactly the model's context.
    public private(set) var archivedPairs: [(user: AppChatTurn, assistant: AppChatTurn)] = []
    /// Host-side Agent Mode activity keyed to the visible user turn. This is
    /// transcript-only state and never enters `AppConversation` or the model KV.
    private var agentActivitiesByTurnID: [UUID: [AppAgentActivity]] = [:]
    private var retainedScreenshotPreviewCount = 0
    public private(set) var outputAgentActivities: [AppAgentActivity] = []
    /// The epoch the inference side has actually been told to open. Nil after a
    /// load or unload, both of which rebuild or release the KV; the next turn
    /// opens the conversation again before it sends anything.
    private var serviceEpoch: UUID?
    public var outputText: String = ""
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens: Int = 4096
    public var temperature: Double = 0.2
    public var topKEnabled: Bool = true
    public var topK: Int = 64
    public var topPEnabled: Bool = true
    public var topP: Double = 0.95
    public private(set) var newlineShortcut: AppNewlineShortcut = .return
    public private(set) var showPromptExamples: Bool = true
    /// Whether launching the app should load the model straight away. Off by
    /// default, because loading takes minutes and holds gigabytes.
    public private(set) var loadModelOnLaunch: Bool = false
    public private(set) var agentModeEnabled: Bool = false
    public private(set) var agentBundleIdentifier = "com.hebertgo.nestmind.debug"
    public private(set) var agentSimulatorUDID = "7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382"
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var installState: AppModelInstallState = .idle
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    /// The companion download is 1.5 GB and deserves the same answer to "how
    /// long is this going to take" as the model download. Kept separate because
    /// both can be in flight in principle and an estimator holds per-download
    /// rate state.
    public private(set) var visionInstallETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var visionInstallETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus
    public var visionInstallState: AppModelInstallState = .idle
    /// How far activation's hash of the companion weights has got, 0 to 1.
    /// Activation reads about 1.5 GB, which was a bare spinner with no way to
    /// tell a slow verify from a stuck one.
    public private(set) var visionActivationProgress: Double?
    public private(set) var visionInstallReadiness: AppModelInstallReadiness = .checking
    public private(set) var visionInstallationStatus: AppVisionPackInstallationStatus

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public private(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    private var liveStructuredProgress: DecodeStructuredProgress?
    /// Display only. Never appended to the answer, conversation, or tool results.
    public private(set) var thinkingPreview: DecodeThinkingPreview?
    /// Raw unfinished model output for display only, never an executable request.
    public private(set) var toolCallPreview: DecodeToolCallPreview?
    private var agentWaitingForMCP = false
    private var agentModelStepActive = false

    public var generationStatusText: String? {
        guard isRunning else { return nil }
        if isCancellationPending { return "Stopping generation" }
        if let compaction = activeAgentCompaction {
            guard compaction.replacementPromptTokens != nil else { return "Compacting history…" }
            return phase == .prefill && livePrefillTotal > 0
                ? "Compacting history… · Rebuilding \(livePrefillDone) / \(livePrefillTotal) tokens"
                : "Compacting history… · Rebuilding model context"
        }
        if agentWaitingForMCP { return "Waiting for VisionCapture" }
        if phase == .prefill {
            return livePrefillTotal > 0
                ? "Reading prompt · \(livePrefillDone) / \(livePrefillTotal) tokens"
                : "Processing your prompt"
        }
        guard phase == .decode else { return "Preparing next step" }
        let stage: String
        switch liveStructuredProgress?.stage {
        case "thinking": stage = "Thinking"
        case "tool_call": stage = "Preparing tool call"
        case "visible_response": stage = "Writing response"
        case "channel_label": stage = "Reading channel label"
        default: stage = agentModeEnabled ? "Unknown output stage" : "Writing response"
        }
        let seconds = max(0, Int(liveElapsedDecodeSeconds))
        return "\(stage) · \(seconds / 60):\(String(format: "%02d", seconds % 60)) elapsed · \(liveTokenCount) tokens"
    }
    public private(set) var livePrefillDone: Int = 0
    public private(set) var livePrefillTotal: Int = 0
    public private(set) var liveMemoryBytes: UInt64?
    /// Resident bytes of the inference process. The footprint above is what
    /// the system counts against the process; this is what it actually holds,
    /// including the mapped weights the footprint omits. A 26B model reports
    /// about 160 MB of footprint right after loading, which is true and reads
    /// as nonsense without this beside it.
    public private(set) var liveResidentBytes: UInt64?
    /// Tower weights the inference process is holding mapped, reported
    /// separately because no per-process counter attributes them.
    public private(set) var visionTowerMappedBytes: UInt64?
    public private(set) var isCancellationPending: Bool = false
    /// Increments when a generation starts. The transcript watches it to put
    /// the newest turn on screen: with several images attached, the prompt and
    /// its thumbnails are tall enough to push the answer out of view, so
    /// scrolling only when the reader was already at the bottom left them
    /// looking at their own attachments while the model worked.
    public private(set) var runIdentity: Int = 0

    private let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private let visionInstaller: any AppVisionPackInstallerClient
    private var runTask: Task<Void, Never>?
    /// One exact user instruction waiting for the current Agent Mode step to
    /// stop at a safe model boundary. It becomes a normal visible user turn,
    /// so the model and later context checkpoints keep its real role.
    private var pendingAgentInstruction: String?
    /// One automatic correction is permitted for a model answer that labels
    /// its own QA checklist unfinished. A second such answer stays visibly
    /// incomplete instead of looping.
    private var agentChecklistContinuationUsed = false
    /// Agent Mode Stop must reach the decode service after decode begins so
    /// the service can settle and commit the turn at a token boundary.
    private var agentCancellationIssued = false
    private var loadTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var visionInstallTask: Task<Void, Never>?
    private var unloadTask: Task<Void, Never>?
    private var agentToolLoop = VisionCaptureToolLoop()
    private var agentCheckpointRequested = false
    private var agentCheckpointRecord: String?
    private var agentCheckpointPrepared = false
    /// One display row spans the refresh, accepted reset and actual KV rebuild.
    /// A reset receipt alone is not a completed rebuild.
    private struct AgentCompactionProgress {
        let id: UUID
        let trigger: DecodeContextCheckpointTrigger
        let performanceEvidence: DecodePerformanceCheckpointEvidence?
        var existingPromptTokens: Int
        var replacementPromptTokens: Int?
        var rebuildConfirmed = false
    }
    private var activeAgentCompaction: AgentCompactionProgress?
    /// Small pinned chat caption. Updated only at compaction transitions, with
    /// the same text retained in the transcript activity row.
    public private(set) var contextCompactionStatusText: String?
    private var activeAgentActivityTurnID: UUID?
    private var loadGeneration: UInt64 = 0
    /// The highest load-phase sequence already applied. Each `onState` callback
    /// hops to the main actor in its own task, and ordering between separately
    /// created tasks is not guaranteed, so `.ready` could be applied before the
    /// `.loading(.preparingRunner)` that preceded it and leave the UI showing a
    /// phase the runtime had already left.
    private var appliedLoadSequence: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var visionInstallGeneration: UInt64 = 0
    private var visionInstallCancellationRequested = false
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    private var hasHandledTerminalEvent = false
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()
    private var visionInstallETAEstimator = DownloadETAEstimator()
    private let attachmentStore: AppImageAttachmentStore
    public let isVisionRuntimeSupported: Bool

    public static var currentDeviceSupportsVisionRuntime: Bool {
        VisionRuntime.isSupportedOnDefaultDevice
    }

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(),
                visionInstaller: any AppVisionPackInstallerClient = RepackVisionPackInstallerClient(),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                attachmentStore: AppImageAttachmentStore = AppImageAttachmentStore(),
                visionRuntimeSupported: Bool = true,
                settingsPersistenceEnabled: Bool = false) {
        let directory = (modelDirectory ?? AppModelLocation.defaultURL()).standardizedFileURL
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
            : MacAppSettings()
        self.modelPathText = directory.path
        // The app always releases the image tower after each image. Keeping it
        // resident saves a few hundred milliseconds on a run of images and
        // holds about 1 GB of page cache to do it — a trade worth exposing to
        // a CLI or server operator, not to someone using the app, where it was
        // one more setting whose effect no figure on screen could show.
        // `keepReady` remains available through AppRuntimeOptions for those.
        self.runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled,
            rdadvisePolicy: settings.rdadvisePolicy,
            visionResidencyPolicy: .onDemand,
            toolThinkingEnabled: settings.toolThinkingEnabled)
        self.maxContextTokens = settings.contextTokens
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.newlineShortcut = settings.newlineShortcut
        self.showPromptExamples = settings.showPromptExamples
        self.loadModelOnLaunch = settings.loadModelOnLaunch
        self.agentModeEnabled = settings.agentModeEnabled
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.visionInstallationStatus = AppVisionPackInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.visionInstaller = visionInstaller
        self.memorySampler = memorySampler
        self.attachmentStore = attachmentStore
        self.isVisionRuntimeSupported = visionRuntimeSupported
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        // Staged images of runs that were killed before they could clean up;
        // nothing else ever removes them.
        AppImageAttachmentStore.sweepAbandoned()
        refreshInstallReadiness()
        refreshVisionInstallReadiness()
    }

    public var isRunning: Bool { runState == .running }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && !isVisionCompanionOperationInProgress
            && loadState.isReady
    }

    public var isModelInstalled: Bool { installationStatus == .complete }

    public var requiresModelInstallation: Bool { !isModelInstalled }

    public var installDescriptor: AppModelInstallDescriptor { installer.descriptor }

    public var installRequirement: AppModelInstallRequirement? {
        installReadiness.requirement
    }

    public var isInstallingModel: Bool { installState.isInstalling }

    public var canInstallModel: Bool {
        guard case .ready = installReadiness else { return false }
        return !isRunning && !loadState.isLoading && !isInstallingModel
            && !isVisionCompanionOperationInProgress
            && requiresModelInstallation
    }

    public var canCancelInstall: Bool { installState.canCancel }

    public var isVisionPackInstalled: Bool { visionInstallationStatus == .complete }

    public var isInstallingVisionPack: Bool { visionInstallState.isInstalling }

    public var visionInstallDescriptor: AppModelInstallDescriptor {
        visionInstaller.descriptor
    }

    /// Every companion Download, Resume, Verify, Activate, Repair, and Remove
    /// operation is one app-blocking state: model actions stay disabled until it
    /// reaches a resting state, so a companion transaction never overlaps a
    /// loaded session or another companion operation.
    public var isVisionCompanionOperationInProgress: Bool {
        visionInstallState.isInstalling
    }

    /// A companion operation may only begin against an unloaded model session
    /// with no other transfer in flight; the draft, transcript, and attachments
    /// are untouched by the gate.
    public var canBeginVisionCompanionOperation: Bool {
        !isRunning && !loadState.isLoading && !loadState.isReady
            && !isInstallingModel && !isVisionCompanionOperationInProgress
    }

    public var canInstallVisionPack: Bool {
        guard isVisionRuntimeSupported else { return false }
        // A layout with nowhere to put a companion cannot be repaired by
        // downloading one, so do not offer to.
        guard visionInstallationStatus != .unsupportedLayout else { return false }
        guard isModelInstalled, !isVisionPackInstalled,
              case .ready = visionInstallReadiness else { return false }
        if case .readyToActivate = visionInstallState { return false }
        return canBeginVisionCompanionOperation
    }

    public var canActivateVisionPack: Bool {
        guard isVisionRuntimeSupported else { return false }
        guard case .readyToActivate = visionInstallState else { return false }
        return canBeginVisionCompanionOperation
    }

    public var canCancelVisionInstall: Bool { visionInstallState.canCancel }

    public var visionInstallProgressFraction: Double? {
        // Activation hashes about 1.5 GB, so it gets a bar of its own rather
        // than an indeterminate spinner for minutes.
        if case .activating = visionInstallState { return visionActivationProgress }
        guard case .copyingPayload(let reused, let downloaded, let total) = visionInstallState,
              total > 0 else { return nil }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var visionInstallPhaseLabel: String {
        switch visionInstallState {
        case .idle: return isVisionPackInstalled ? "Installed" : "Not installed"
        case .checking: return "Checking image support"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning image support"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading image support"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing download"
        case .activating:
            guard let fraction = visionActivationProgress else {
                return "Activating image support"
            }
            return "Verifying image support \(Int(fraction * 100))%"
        case .cancelling: return "Cancelling"
        case .discarding: return "Cleaning up"
        case .cancelled: return "Download paused"
        case .readyToActivate: return "Ready to activate"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Installed"
        case .failed: return "Installation failed"
        }
    }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        return min(reused.addingReportingOverflow(downloaded).partialValue, total)
    }

    public var installTotalBytes: UInt64? {
        guard case .copyingPayload(_, _, let total) = installState else {
            return nil
        }
        return total
    }

    public var installReusedBytes: UInt64? {
        guard case .copyingPayload(let reused, _, _) = installState else {
            return nil
        }
        return reused
    }

    public var installDownloadedThisRunBytes: UInt64? {
        guard case .copyingPayload(_, let downloaded, _) = installState else {
            return nil
        }
        return downloaded
    }

    public var installProgressFraction: Double? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState,
              total > 0 else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var installPhaseLabel: String {
        switch installState {
        case .idle: return "Model required"
        case .checking: return "Checking installation"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning installation"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading model"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing installation"
        case .activating:
            guard let fraction = visionActivationProgress else {
                return "Activating image support"
            }
            return "Verifying image support \(Int(fraction * 100))%"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding download"
        case .cancelled: return "Download paused"
        case .readyToActivate: return "Ready to activate"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Model installed"
        case .failed: return "Installation failed"
        }
    }

    public var canRun: Bool {
        // Staging copies the files a request will carry. Starting a run while
        // it is in flight sent a request without those images and then landed
        // them on the next message instead.
        !isRunning && !isAddingImages && isModelAvailable && !loadState.isLoading
            && !isVisionCompanionOperationInProgress
            && !hasStaleLoadedRuntime
            // A conversation whose KV no longer matches it cannot take another
            // turn; only New chat clears that.
            && conversation.canSend
            && (!promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !imageAttachments.isEmpty)
    }

    public var canSendAgentInstruction: Bool {
        agentModeEnabled && isRunning && !isCancellationPending
            && pendingAgentInstruction == nil
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canSubmitPrompt: Bool { canRun || canSendAgentInstruction }

    public var isAgentInstructionPending: Bool {
        pendingAgentInstruction != nil
    }

    public var canCancel: Bool {
        isRunning && (!isCancellationPending || pendingAgentInstruction != nil)
    }

    public var hasOutputTranscript: Bool {
        !archivedPairs.isEmpty
            || !conversation.isEmpty
            || !outputPromptText.isEmpty || !outputImageAttachments.isEmpty
            || !outputText.isEmpty || !outputAgentActivities.isEmpty
    }

    public var shouldShowPromptExamples: Bool {
        showPromptExamples
            && promptText.isEmpty
            && !isRunning
            && !hasOutputTranscript
    }

    public var outputResponsePlainText: String {
        generationTranscriptMailbox?.completeText ?? outputText
    }

    /// Completed turns the transcript draws *above* the live one.
    ///
    /// The live fields keep holding the last finished turn between runs — that
    /// is what leaves an answer on screen after it ends — so while nothing is
    /// decoding the newest pair is drawn live and must not also appear here.
    /// One property, used by both the transcript and Copy Conversation, so the
    /// two cannot disagree about which turn is which.
    public var transcriptHistory: [(user: AppChatTurn, assistant: AppChatTurn)] {
        let pairs = conversation.completedPairs
        let live = conversation.hasTurnInFlight ? pairs
            : (pairs.isEmpty ? pairs : Array(pairs.dropLast()))
        return archivedPairs + live
    }

    public var transcriptAgentActivityHistory: [[AppAgentActivity]] {
        transcriptHistory.map { pair in
            agentActivitiesByTurnID[pair.user.id] ?? []
        }
    }

    /// Where the transcript draws "earlier turns are no longer in context",
    /// counted in pairs from the top. Nil when everything on screen is still in
    /// the model's context.
    public var transcriptContextBreak: Int? {
        archivedPairs.isEmpty ? nil : archivedPairs.count
    }

    public var outputConversationPlainText: String {
        var history: [String] = []
        for pair in transcriptHistory {
            var prompt = pair.user.text
            if !pair.user.images.isEmpty {
                let names = pair.user.images.map(\.displayName).joined(separator: ", ")
                prompt = prompt.isEmpty ? "[images: \(names)]" : "[images: \(names)]\n\(prompt)"
            }
            history.append("You:\n\(prompt)")
            history.append("Answer:\n\(pair.assistant.text)")
        }
        let live = liveConversationPlainText
        if history.isEmpty { return live }
        if live.isEmpty { return history.joined(separator: "\n\n") }
        return history.joined(separator: "\n\n") + "\n\n" + live
    }

    private var liveConversationPlainText: String {
        let response = outputResponsePlainText
        switch (outputPromptText.isEmpty, response.isEmpty) {
        case (true, true):
            return ""
        case (false, true):
            return "You:\n\(outputPromptText)"
        case (true, false):
            return "Answer:\n\(response)"
        case (false, false):
            return "You:\n\(outputPromptText)\n\nAnswer:\n\(response)"
        }
    }

    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    public var presentation: AppPresentationState {
        AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: requiresModelInstallation,
            installState: installState,
            installReadiness: installReadiness,
            loadState: loadState,
            hasStaleRuntime: hasStaleLoadedRuntime,
            isRunning: isRunning,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            lastStopReason: diagnostics?.stopReason,
            isVisionCompanionOperationInProgress: isVisionCompanionOperationInProgress,
            terminalError: hasHandledTerminalEvent ? error : nil))
    }

    public var currentProcessMemoryBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        // `liveMemoryBytes` first because it is a tracked property: reading the
        // reporter alone told the truth but was invisible to observation, so
        // the figure only refreshed when something else — a generated token —
        // happened to redraw the view. Through prefill, nothing did.
        if let liveMemoryBytes { return liveMemoryBytes }
        // When inference runs in another process, its memory is the only
        // memory worth showing. Falling back to this app's own sampler put the
        // UI's footprint in a row labelled as the model's.
        if let reporter = client as? any AppInferenceMemoryReporting {
            return reporter.currentInferenceMemoryBytes
        }
        return memorySampler.sample()
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox? {
        (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
    }

    private var currentRuntimeKey: AppLoadedRuntimeKey {
        AppLoadedRuntimeKey(modelDirectory: URL(fileURLWithPath: modelPathText),
                            maxContextTokens: maxContextTokens,
                            options: runtimeOptions,
                            forceLogitsHead: currentForceLogitsHead)
    }

    private var currentForceLogitsHead: Bool {
        temperature != 0
    }

    public func setModelURL(_ url: URL) {
        guard !isRunning else { return }
        let path = url.standardizedFileURL.path
        guard path != modelPathText else { return }

        modelPathText = path
        clearImages()
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        installGeneration &+= 1
        installTask?.cancel()
        installer.cancel()
        installTask = nil
        visionInstallGeneration &+= 1
        visionInstallCancellationRequested = false
        visionInstallTask?.cancel()
        visionInstaller.cancel()
        visionInstallTask = nil
        resetInstallETA()
        installState = .idle
        visionInstallState = .idle
        pendingExplicitLoadRuntimeKey = nil
        activeRunRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        endConversationForReleasedKV()
        diagnostics = nil
        error = nil
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(at: URL(fileURLWithPath: path))
        visionInstallationStatus = AppVisionPackInstallationProbe.status(
            at: URL(fileURLWithPath: path))
        refreshInstallReadiness()
        refreshVisionInstallReadiness()

        if let lifecycle = client as? AppModelLifecycleClient {
            unloadGeneration &+= 1
            let generation = unloadGeneration
            let task = Task { [weak self, lifecycle] in
                await lifecycle.unload()
                self?.clearUnloadTask(generation: generation)
            }
            unloadTask = task
        }
    }

    public func loadModel() {
        guard canLoadModel else { return }
        beginLoad()
    }

    public func perform(_ action: AppModelAction) {
        switch action {
        case .install: installModel()
        case .cancelInstall: cancelInstall()
        case .load, .retryLoad: loadModel()
        case .cancelLoad: cancelLoad()
        case .reload: reloadModel()
        case .unload: unloadModel()
        }
    }

    public func setNewlineShortcut(_ shortcut: AppNewlineShortcut) {
        guard newlineShortcut != shortcut else { return }
        newlineShortcut = shortcut
        persistSettings()
    }

    public func setShowPromptExamples(_ show: Bool) {
        guard showPromptExamples != show else { return }
        showPromptExamples = show
        persistSettings()
    }


    public func setLoadModelOnLaunch(_ enabled: Bool) {
        guard loadModelOnLaunch != enabled else { return }
        loadModelOnLaunch = enabled
        persistSettings()
    }

    public var canChangeAgentMode: Bool {
        !isRunning && !loadState.isLoading && !isInstallingModel
            && !isVisionCompanionOperationInProgress
            && conversation.isEmpty && conversation.canSend
    }

    public var toolThinkingEnabled: Bool { runtimeOptions.toolThinkingEnabled }

    public var canChangeToolThinking: Bool {
        !isRunning && !loadState.isLoading && !isInstallingModel
            && !isVisionCompanionOperationInProgress
    }

    public var toolThinkingStatus: String {
        guard loadState.isReady, let loadedRuntimeKey else {
            return "Used with Agent Mode. Applies when the model loads."
        }
        let current = loadedRuntimeKey.toolThinkingEnabled ? "On" : "Off"
        if loadedRuntimeKey.toolThinkingEnabled != toolThinkingEnabled {
            return "Current: \(current). Reload Model to apply. This starts a new chat."
        }
        return "Current: \(current). Used with Agent Mode."
    }

    public func setToolThinkingEnabled(_ enabled: Bool) {
        guard canChangeToolThinking, toolThinkingEnabled != enabled else { return }
        runtimeOptions.toolThinkingEnabled = enabled
        persistSettings()
    }

    public func setAgentModeEnabled(_ enabled: Bool) {
        guard canChangeAgentMode, agentModeEnabled != enabled else { return }
        agentModeEnabled = enabled
        agentToolLoop = VisionCaptureToolLoop()
        persistSettings()
    }

    public func setAgentBundleIdentifier(_ value: String) {
        guard agentModeEnabled, !isRunning,
              agentBundleIdentifier != value else { return }
        agentBundleIdentifier = value
    }

    public func setAgentSimulatorUDID(_ value: String) {
        guard agentModeEnabled, !isRunning,
              agentSimulatorUDID != value else { return }
        agentSimulatorUDID = value
    }

    private func makeAgentConfiguration() -> VisionCaptureAgentConfiguration {
        VisionCaptureAgentConfiguration(
            bundleIdentifier: agentBundleIdentifier.trimmingCharacters(
                in: .whitespacesAndNewlines),
            simulatorUDID: agentSimulatorUDID.trimmingCharacters(
                in: .whitespacesAndNewlines),
            modelDirectory: URL(fileURLWithPath: modelPathText, isDirectory: true))
    }

    /// Starts the launch load if it is switched on and the model can be loaded.
    /// Called once, when the window first appears; a model that is missing,
    /// already loading, or busy with a companion operation is left alone.
    public func loadModelAtLaunchIfEnabled() {
        guard loadModelOnLaunch, canLoadModel else { return }
        loadModel()
    }

    /// Whether an image can be attached at all.
    ///
    /// The runtime flag only says this build *can* use images; the companion
    /// pack is what makes it possible for this model. Gating on the flag alone
    /// offered an Add-images button with no tower behind it, and the failure
    /// only surfaced when the user pressed Generate.
    public var isImageInputAvailable: Bool {
        isVisionRuntimeSupported && isVisionPackInstalled
    }

    /// Image support is part of this build. Hardware support and companion-pack
    /// availability are separate so the inspector can explain either absence.
    public var visionRuntimeEnabled: Bool { true }

    /// Room left for the prompt when working out how many images fit. The
    /// runtime still rejects a combination that does not fit, so this only has
    /// to be a defensible reserve rather than an exact prompt measurement.
    nonisolated static let reservedPromptTokens = 1_024

    /// How many images this conversation can hold, derived from the context
    /// exactly as the server derives its budget. It used to be a fixed four,
    /// which meant the same set of images was accepted over the API and refused
    /// in the app.
    public var maximumImageAttachments: Int {
        // The context a run will actually use, which is the loaded session's
        // until it is reloaded. Capping on the pending setting instead let the
        // composer accept images the request then refused.
        //
        // The conversation counts against the same window, and the request
        // reserves it (`AppGenerationRequest.validate`). Reserving only a fixed
        // 1,024 here meant that past that point the composer kept offering
        // images Send would refuse — accepted on attach, rejected on the button.
        Self.imageAttachmentCapacity(
            maxContextTokens: effectiveMaxContextTokens,
            conversationTokens: conversation.kvTokens)
    }

    nonisolated static func imageAttachmentCapacity(
        maxContextTokens: Int,
        conversationTokens: Int?
    ) -> Int {
        guard let conversationTokens else { return 0 }
        return VisionImageTokenBudget.capacity(
            maxContext: maxContextTokens,
            reservedTextTokens: max(reservedPromptTokens, conversationTokens))
    }

    /// The context a generation would run with right now.
    public var effectiveMaxContextTokens: Int {
        (loadedRuntimeKey ?? currentRuntimeKey).maxContextTokens
    }

    /// `discardingSourceDirectory` is the temp directory a file-promise drop
    /// wrote into. It is ours, it holds nothing but those copies, and staging
    /// takes its own copy — so it must not outlive the staging that consumed
    /// it, which is exactly how it leaked.
    public func addImages(_ urls: [URL], discardingSourceDirectory: URL? = nil) {
        // Every early return has to discard the promise directory itself. The
        // staging task's `defer` below owns it only once that task exists, so a
        // return above it strands the full-size copies with nothing left to
        // delete them: the attachment sweep only covers the staging root.
        func discardSource() {
            if let discardingSourceDirectory {
                try? FileManager.default.removeItem(at: discardingSourceDirectory)
            }
        }
        guard isImageInputAvailable, !urls.isEmpty else {
            discardSource()
            return
        }
        guard !isRunning else {
            // A promise drop admitted before the run started can be delivered
            // after it. Returning silently made the images look as though they
            // had simply vanished.
            imageAttachmentError =
                "Wait for the current run to finish before attaching images."
            discardSource()
            return
        }
        let capacity = maximumImageAttachments
        let available = max(0, capacity - imageAttachments.count)
        guard available > 0 else {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
            discardSource()
            return
        }
        addingImagesCount += 1
        imageAttachmentError = nil
        // Dropping the rest silently left the user believing every image they
        // chose was attached.
        let selected = Array(urls.prefix(available))
        if selected.count < urls.count {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
        }
        let store = attachmentStore
        Task.detached(priority: .userInitiated) { [weak self] in
            var staged: [AppImageAttachment] = []
            defer {
                if let discardingSourceDirectory {
                    try? FileManager.default.removeItem(at: discardingSourceDirectory)
                }
            }
            do {
                for url in selected {
                    staged.append(try store.stage(url))
                }
                await self?.finishAddingImages(staged)
            } catch {
                // The batch is all-or-nothing, so the copies made before the
                // failure are referenced by nothing and would never be deleted.
                for attachment in staged { store.remove(attachment) }
                await self?.finishAddingImages(error: error)
            }
        }
    }

    /// Attaches image bytes that have no file behind them: an image copied out
    /// of another app arrives on the pasteboard as data, and a drag from an app
    /// that has not written the file yet arrives as a promise.
    public func addImageData(_ data: Data, displayName: String) {
        guard isImageInputAvailable, !isRunning else { return }
        let capacity = maximumImageAttachments
        guard imageAttachments.count < capacity else {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
            return
        }
        addingImagesCount += 1
        imageAttachmentError = nil
        let store = attachmentStore
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let staged = try store.stage(data: data, displayName: displayName)
                await self?.finishAddingImages([staged])
            } catch {
                await self?.finishAddingImages(error: error)
            }
        }
    }

    static func imageCapacityMessage(capacity: Int, context: Int) -> String {
        "At most \(capacity) image\(capacity == 1 ? "" : "s") fit in the "
            + "\(context / 1_024)K context this session is running with. Raise "
            + "Context in Memory and reload the model to send more."
    }

    public func reportImageAttachmentError(_ error: Error) {
        imageAttachmentError = String(describing: error)
    }

    public func reportImageAttachmentError(_ message: String) {
        imageAttachmentError = message
    }

    public func removeImage(id: UUID) {
        guard !isRunning,
              let index = imageAttachments.firstIndex(where: { $0.id == id }) else { return }
        let attachment = imageAttachments.remove(at: index)
        attachmentStore.remove(attachment)
        imageAttachmentError = nil
    }

    public func clearImages() {
        guard !isRunning else { return }
        for attachment in imageAttachments { attachmentStore.remove(attachment) }
        imageAttachments.removeAll()
        imageAttachmentError = nil
    }

    private func finishAddingImages(_ staged: [AppImageAttachment]) {
        // Two adds can be in flight at once — the picker and a drop — and each
        // sized itself against the count it saw at admission, so the second to
        // land can push past the cap. Re-check against the real count here and
        // delete what does not fit, rather than leaving staged copies that
        // nothing references.
        defer { addingImagesCount = max(0, addingImagesCount - 1) }
        // The counter keeps `canRun` closed until every batch lands, so a run
        // should not be able to start underneath one. If it ever does, the run
        // has already snapshotted its images: appending here would attach them
        // to the *next* message with no way to take them off, which is worse
        // than saying so. `addImages` refuses a mid-run drop the same way.
        guard !isRunning else {
            for attachment in staged { attachmentStore.remove(attachment) }
            imageAttachmentError =
                "Wait for the current run to finish before attaching images."
            return
        }
        let capacity = maximumImageAttachments
        let available = max(0, capacity - imageAttachments.count)
        let accepted = staged.prefix(available)
        for attachment in staged.dropFirst(accepted.count) {
            attachmentStore.remove(attachment)
        }
        imageAttachments.append(contentsOf: accepted)
        if accepted.count < staged.count {
            imageAttachmentError = Self.imageCapacityMessage(
                capacity: capacity, context: effectiveMaxContextTokens)
        }
    }

    private func finishAddingImages(error: Error) {
        addingImagesCount = max(0, addingImagesCount - 1)
        imageAttachmentError = String(describing: error)
    }

    public func reloadModel() {
        guard canReloadModel else { return }
        beginLoad()
    }

    private func beginLoad() {
        activeAgentCompaction = nil
        contextCompactionStatusText = nil
        liveStructuredProgress = nil
        thinkingPreview = nil
        toolCallPreview = nil
        agentWaitingForMCP = false
        guard let lifecycle = client as? AppModelLifecycleClient else {
            loadState = .failed(.modelLoadFailed("This client has no model load lifecycle."))
            return
        }
        let directory = URL(fileURLWithPath: modelPathText)
        let maxContext = maxContextTokens
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: runtimeOptions,
                                             forceLogitsHead: forceLogitsHead)
        // The session is loaded with the same normalized options a run sends.
        // Loading with the raw settings instead meant a control that is off but
        // still carries a non-default value — RDADVISE off with its policy left
        // on `bounded` — produced a loaded session no run could match, and the
        // staleness check compares two normalized keys, so nothing ever offered
        // the reload that would have cleared it.
        let options = runtimeKey.options(prefillEnabled: runtimeOptions.prefillEnabled,
                                        prefillChunkTokens: runtimeOptions.prefillChunkTokens)
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        error = nil
        appliedLoadSequence = 0
        loadState = .loading(.validatingDirectory)
        let emitted = Mutex<UInt64>(0)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    // Stamped where the phase is emitted, in order; checked
                    // where it is applied, which is not.
                    let sequence = emitted.withLock { value -> UInt64 in
                        value += 1
                        return value
                    }
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation,
                                             sequence: sequence)
                    }
                }
            } catch is CancellationError {
            } catch let appError as AppInferenceError {
                await self?.applyLoadState(.failed(appError), generation: generation)
            } catch {
                await self?.applyLoadState(
                    .failed(.modelLoadFailed("\(error)")),
                    generation: generation)
            }
            await self?.clearLoadTask(generation: generation)
        }
    }

    public func cancelLoad() {
        guard canCancelLoad, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .cancelling
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.loadState = .notLoaded
            endConversationForReleasedKV()
            self.clearUnloadTask(generation: generation)
        }
    }

    /// Ends the conversation because the KV behind it is gone.
    ///
    /// `applyLoadState`'s `.notLoaded` branch does this too, but nothing reaches
    /// it: every production transition to `.notLoaded` assigns `loadState`
    /// directly. Calling this from those sites is what actually runs it.
    private func endConversationForReleasedKV() {
        serviceEpoch = nil
        archiveConversationContext()
    }

    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            endConversationForReleasedKV()
            self.clearUnloadTask(generation: generation)
        }
    }

    public func installModel() {
        guard !isRunning, !loadState.isLoading, !isInstallingModel,
              requiresModelInstallation else {
            return
        }
        refreshInstallReadiness()
        guard canInstallModel else { return }
        installTask?.cancel()
        installer.cancel()
        resetInstallETA()
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .checking
        installTask = Task { [weak self, installer] in
            do {
                for try await event in installer.installDefaultModel(outputDirectory: outputDirectory) {
                    guard let self else { return }
                    self.applyInstallEvent(event, generation: generation)
                }
                self?.finishInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishInstallCancellation(generation: generation)
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelInstall() {
        guard canCancelInstall else { return }
        installState = .cancelling
        installer.cancel()
    }

    public var hasPartialModelDownload: Bool {
        guard let paths = try? RemoteInstallPaths(outputDirectory: modelPathText) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardModelDownload: Bool {
        hasPartialModelDownload && !isInstallingModel && !isRunning
    }

    public func discardModelDownload() {
        guard canDiscardModelDownload else { return }
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .discarding
        installTask = Task { [weak self, installer] in
            do {
                try await installer.discardPartialInstall(
                    outputDirectory: outputDirectory)
                guard let self, generation == self.installGeneration else { return }
                self.installTask = nil
                self.installState = .idle
                self.refreshInstallReadiness()
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public var hasPartialVisionPackDownload: Bool {
        guard let output = try? VisionPackLocation.companionURL(
            forTextModel: URL(fileURLWithPath: modelPathText, isDirectory: true)),
              let paths = try? RemoteInstallPaths(outputDirectory: output.path) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardVisionPackDownload: Bool {
        hasPartialVisionPackDownload && canBeginVisionCompanionOperation
    }

    public var canRemoveVisionPack: Bool {
        hasVisionPackDirectory && canBeginVisionCompanionOperation
    }

    public var hasVisionPackDirectory: Bool {
        guard let output = try? VisionPackLocation.companionURL(
            forTextModel: URL(fileURLWithPath: modelPathText, isDirectory: true)) else {
            return false
        }
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: output.path,
            isDirectory: &isDirectory) && isDirectory.boolValue
    }

    public func installVisionPack() {
        guard canInstallVisionPack else { return }
        visionInstallCancellationRequested = false
        visionInstallTask?.cancel()
        visionInstaller.cancel()
        let textModelDirectory = URL(
            fileURLWithPath: modelPathText,
            isDirectory: true).standardizedFileURL
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .checking
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                for try await event in visionInstaller.install(
                    textModelDirectory: textModelDirectory) {
                    guard let self else { return }
                    self.applyVisionInstallEvent(event, generation: generation)
                }
                self?.finishVisionInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishVisionInstallCancellation(generation: generation)
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelVisionInstall() {
        guard canCancelVisionInstall else { return }
        visionInstallCancellationRequested = true
        visionInstallState = .cancelling
        visionInstaller.cancel()
        // A cancel raised before the stream registers its own task would find no
        // active install; cancelling the consumer terminates the stream, which
        // routes through the same cooperative drain-to-checkpoint path.
        visionInstallTask?.cancel()
    }

    public func activateVisionPack() {
        guard canActivateVisionPack else { return }
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        resetVisionInstallETA()
        visionInstallState = .activating
        visionActivationProgress = 0
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                let output = try await visionInstaller.activatePreparedInstall(
                    textModelDirectory: directory,
                    onVerifyProgress: { [weak self] fraction in
                        Task { @MainActor in
                            guard let self,
                                  generation == self.visionInstallGeneration,
                                  case .activating = self.visionInstallState else { return }
                            // Each hop is its own task, and tasks are not
                            // ordered against each other, so a late one must
                            // not walk the bar backwards.
                            guard fraction >= (self.visionActivationProgress ?? 0)
                            else { return }
                            self.visionActivationProgress = fraction
                        }
                    })
                // Applied first: a progress hop still in flight is dropped
                // once the state is no longer `.activating`, so clearing before
                // this could be undone by a late update.
                self?.applyVisionInstallEvent(
                    .installed(output), generation: generation)
                self?.visionActivationProgress = nil
            } catch is CancellationError {
                // Cancellation can only land during verification, before
                // anything is renamed, so the prepared pack is untouched and
                // still activatable.
                self?.finishVisionActivationCancelled(generation: generation)
                self?.visionActivationProgress = nil
            } catch {
                self?.finishVisionInstallFailure(
                    error, generation: generation, phase: .activation)
                self?.visionActivationProgress = nil
            }
        }
    }

    private func finishVisionActivationCancelled(generation: UInt64) {
        guard generation == visionInstallGeneration else { return }
        resetVisionInstallETA()
        visionInstallTask = nil
        visionInstallCancellationRequested = false
        visionInstallState = .idle
        refreshVisionInstallReadiness()
    }

    public func discardVisionPackDownload() {
        guard canDiscardVisionPackDownload else { return }
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .discarding
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                try await visionInstaller.discardPartialInstall(
                    textModelDirectory: directory)
                guard let self, generation == self.visionInstallGeneration else { return }
                self.visionInstallTask = nil
                self.visionInstallState = .idle
                self.refreshVisionInstallReadiness()
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    /// Drives the confirmation the Model menu puts in front of `removeVisionPack`.
    ///
    /// The Inspector hides its own Remove button once the pack is installed, so
    /// the menu item is the only reachable way to delete 1.14 GB — and it called
    /// straight through, with the dialog sitting on an unreachable branch.
    public func requestVisionPackRemoval() {
        guard canRemoveVisionPack else { return }
        isConfirmingVisionPackRemoval = true
    }

    public func removeVisionPack() {
        isConfirmingVisionPackRemoval = false
        guard canRemoveVisionPack else { return }
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        visionInstallCancellationRequested = false
        visionInstallGeneration &+= 1
        let generation = visionInstallGeneration
        visionInstallState = .discarding
        visionInstallTask = Task { [weak self, visionInstaller] in
            do {
                try await visionInstaller.removeInstalled(
                    textModelDirectory: directory)
                guard let self, generation == self.visionInstallGeneration else { return }
                self.visionInstallTask = nil
                self.visionInstallState = .idle
                self.visionInstallationStatus = .missing
                self.refreshVisionInstallReadiness()
            } catch {
                self?.finishVisionInstallFailure(error, generation: generation)
            }
        }
    }

    public func refreshInstallReadiness() {
        refreshInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true).standardizedFileURL)
    }

    public func recheckModelAtCurrentLocation() {
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        modelPathText = directory.path
        refreshInstallReadiness(at: directory)
        refreshVisionInstallReadiness(at: directory)
    }

    private func refreshInstallReadiness(at outputDirectory: URL) {
        installationStatus = AppModelInstallationProbe.status(
            at: outputDirectory,
            descriptor: installer.descriptor)
        guard !isModelInstalled else { return }
        installReadiness = .checking
        do {
            let requirement = try installer.checkInstallRequirement(
                outputDirectory: outputDirectory)
            installReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            installReadiness = .failed("\(error)")
        }
    }

    public func refreshVisionInstallReadiness() {
        refreshVisionInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true)
                .standardizedFileURL)
    }

    private func refreshVisionInstallReadiness(at textModelDirectory: URL) {
        visionInstallationStatus = AppVisionPackInstallationProbe.status(
            at: textModelDirectory)
        // Removing the companion leaves any attached image unsendable, and the
        // composer would keep offering it with nothing able to encode it.
        // Only once the dust has settled: the probe verifies the pack on disk,
        // and a companion operation renames that directory underneath it, so
        // refreshing mid-operation can briefly report no image support. Acting
        // on that would delete images the user had staged.
        if !isImageInputAvailable, !isVisionCompanionOperationInProgress,
           !imageAttachments.isEmpty {
            for attachment in imageAttachments { attachmentStore.remove(attachment) }
            imageAttachments.removeAll()
            // Say so. Clearing the error alongside the images removed them and
            // the only explanation for their absence in one step, so the
            // composer just quietly emptied itself.
            imageAttachmentError =
                "Image support is unavailable, so the attached images were removed."
        }
        guard isModelInstalled else {
            visionInstallReadiness = .failed("Install the text model first")
            return
        }
        guard !isVisionPackInstalled else { return }
        if visionInstaller.preparedInstallIsValid(
            textModelDirectory: textModelDirectory) {
            let output = try? VisionPackLocation.companionURL(
                forTextModel: textModelDirectory)
            // A pack that failed to activate must not be re-offered for
            // activation: `preparedInstallIsValid` does not hash the weights,
            // so a corrupt pack still looks ready and the user would loop
            // between Activate and the same failure.
            let reportedBroken: Bool
            switch visionInstallState {
            case .recoverable, .failed: reportedBroken = true
            default: reportedBroken = false
            }
            if let output, !isInstallingVisionPack, !reportedBroken {
                visionInstallState = .readyToActivate(output)
            }
        }
        visionInstallReadiness = .checking
        do {
            let requirement = try visionInstaller.checkInstallRequirement(
                textModelDirectory: textModelDirectory)
            visionInstallReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            visionInstallReadiness = .failed("\(error)")
        }
    }

    private func applyVisionInstallEvent(
        _ event: AppModelInstallEvent,
        generation: UInt64
    ) {
        guard generation == visionInstallGeneration else { return }
        if visionInstallCancellationRequested {
            switch event {
            case .readyToActivate, .installed:
                // Work that finished before the cancel landed is reported as it
                // actually ended, not as a pause.
                visionInstallCancellationRequested = false
            default:
                return
            }
        }
        switch event {
        case .checking:
            resetVisionInstallETA()
            visionInstallState = .checking
        case .downloadingMetadata:
            resetVisionInstallETA()
            visionInstallState = .downloadingMetadata
        case .planning:
            resetVisionInstallETA()
            visionInstallState = .planning
        case .reservingOutput:
            resetVisionInstallETA()
            visionInstallState = .reservingOutput
        case .copyingPayload(let reused, let downloaded, let total):
            visionInstallState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloaded,
                totalBytes: total)
            updateVisionInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloaded,
                totalBytes: total)
        case .hashingOutput(let file):
            resetVisionInstallETA()
            visionInstallState = .hashingOutput(file)
        case .finalizing:
            resetVisionInstallETA()
            visionInstallState = .finalizing
        case .readyToActivate(let directory):
            resetVisionInstallETA()
            visionInstallState = .readyToActivate(directory)
            visionInstallTask = nil
        case .installed:
            resetVisionInstallETA()
            let textModelDirectory = URL(
                fileURLWithPath: modelPathText,
                isDirectory: true).standardizedFileURL
            visionInstallationStatus = AppVisionPackInstallationProbe.status(
                at: textModelDirectory)
            guard isVisionPackInstalled else {
                finishVisionInstallFailure(
                    RepackError.configurationInvalid(
                        detail: "completed vision install failed verification"),
                    generation: generation)
                return
            }
            visionInstallState = .installed(modelDirectory: textModelDirectory)
            visionInstallTask = nil
        }
    }

    private func finishVisionInstallStream(generation: UInt64) {
        guard generation == visionInstallGeneration,
              visionInstallTask != nil else { return }
        if visionInstallCancellationRequested || visionInstallState == .cancelling {
            finishVisionInstallCancellation(generation: generation)
        } else if !isVisionPackInstalled {
            finishVisionInstallFailure(
                RepackError.configurationInvalid(
                    detail: "vision installer ended before completion"),
                generation: generation)
        }
    }

    private func finishVisionInstallCancellation(generation: UInt64) {
        guard generation == visionInstallGeneration else { return }
        resetVisionInstallETA()
        visionInstallCancellationRequested = false
        visionInstallTask = nil
        visionInstallState = .cancelled
        refreshVisionInstallReadiness()
    }

    /// Which phase failed. Only a download failure may leave a prepared pack
    /// that is worth activating; a verification failure must never send the
    /// user back to Activate, or the same corrupt pack is offered forever.
    enum VisionFailurePhase { case download, activation }

    func finishVisionInstallFailure(
        _ error: Error, generation: UInt64,
        phase: VisionFailurePhase = .download
    ) {
        guard generation == visionInstallGeneration else { return }
        // An error raised because the user cancelled is a pause with saved
        // progress, not an installation failure.
        guard !visionInstallCancellationRequested else {
            finishVisionInstallCancellation(generation: generation)
            return
        }
        resetVisionInstallETA()
        visionInstallTask = nil
        let hasSavedDownload = hasPartialVisionPackDownload
        let textModelDirectory = URL(
            fileURLWithPath: modelPathText,
            isDirectory: true).standardizedFileURL
        // A download that finished and verifies is activatable whatever went
        // wrong afterwards. Reporting it as "needs attention" hid the Activate
        // button behind a Resume that only repeats work already done. The one
        // failure that must not come back here is verification itself, or the
        // same corrupt pack is offered forever — but a lock held by another
        // process is contention, not corruption.
        let isContention: Bool
        if let repackError = error as? RepackError, case .installBusy = repackError {
            isContention = true
        } else {
            isContention = false
        }
        if phase == .download || isContention, hasSavedDownload,
           let output = try? VisionPackLocation.companionURL(
            forTextModel: textModelDirectory),
           visionInstaller.preparedInstallIsValid(
            textModelDirectory: textModelDirectory) {
            visionInstallState = .readyToActivate(output)
            refreshVisionInstallReadiness(at: textModelDirectory)
            return
        }
        visionInstallState = hasSavedDownload
            ? .recoverable("\(error)")
            : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            visionInstallReadiness = .insufficientSpace(AppModelInstallRequirement(
                probePath: path,
                requiredBytes: required,
                availableBytes: available))
        } else {
            refreshVisionInstallReadiness()
            if hasSavedDownload {
                visionInstallState = .recoverable("\(error)")
            }
        }
    }

    private func applyInstallEvent(_ event: AppModelInstallEvent, generation: UInt64) {
        guard generation == installGeneration else { return }
        switch event {
        case .checking:
            resetInstallETA()
            installState = .checking
        case .downloadingMetadata:
            resetInstallETA()
            installState = .downloadingMetadata
        case .planning:
            resetInstallETA()
            installState = .planning
        case .reservingOutput:
            resetInstallETA()
            installState = .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            installState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
            updateInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            resetInstallETA()
            installState = .hashingOutput(file)
        case .finalizing:
            resetInstallETA()
            installState = .finalizing
        case .readyToActivate:
            finishInstallFailure(
                RepackError.configurationInvalid(
                    detail: "text installer returned a vision-only activation event"),
                generation: generation)
        case .installed(let directory):
            resetInstallETA()
            let directory = directory.standardizedFileURL
            installationStatus = AppModelInstallationProbe.status(
                at: directory,
                descriptor: installer.descriptor)
            guard installationStatus == .complete else {
                finishInstallFailure(
                    RepackError.configurationInvalid(detail: "completed install did not pass metadata validation"),
                    generation: generation)
                return
            }
            installState = .installed(modelDirectory: directory)
            installTask = nil
            modelPathText = directory.path
            loadState = .notLoaded
            endConversationForReleasedKV()
            refreshVisionInstallReadiness(at: directory)
        }
    }

    private func finishInstallStream(generation: UInt64) {
        guard generation == installGeneration, installTask != nil else { return }
        if installState == .cancelling {
            finishInstallCancellation(generation: generation)
        } else if !isModelInstalled {
            finishInstallFailure(
                RepackError.configurationInvalid(detail: "installer ended before completion"),
                generation: generation)
        }
    }

    private func finishInstallCancellation(generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        installState = .cancelled
        resetInstallETA()
        refreshInstallReadiness()
    }

    private func updateInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let timestamp = installETATimestamp
        setInstallETAPresentation(
            installETAEstimator.update(observation, timestamp: timestamp))
    }

    private var installETATimestamp: Double {
        let components = installETAOrigin.duration(to: installETAClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func resetInstallETA() {
        installETAEstimator.reset()
        installETAPresentation = .hidden
        installETAText = nil
    }

    private func updateVisionInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let presentation = visionInstallETAEstimator.update(
            observation, timestamp: installETATimestamp)
        visionInstallETAPresentation = presentation
        visionInstallETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func resetVisionInstallETA() {
        visionInstallETAEstimator.reset()
        visionInstallETAPresentation = .hidden
        visionInstallETAText = nil
    }

    private func setInstallETAPresentation(
        _ presentation: DownloadETAPresentation
    ) {
        installETAPresentation = presentation
        installETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func applyPersistedSettings(forModelDirectory modelDirectory: URL) {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled,
            rdadvisePolicy: settings.rdadvisePolicy,
            // Pinned for the same reason as `init`: the app always releases the
            // image tower. Reading the persisted value here would let a
            // `keepReady` written by an older build resurrect ~1 GB of resident
            // tower on a machine with no control that shows or clears it.
            visionResidencyPolicy: .onDemand,
            toolThinkingEnabled: settings.toolThinkingEnabled)
        maxContextTokens = settings.contextTokens
        temperature = settings.temperature
        topKEnabled = settings.topKEnabled
        topK = settings.topK
        topPEnabled = settings.topPEnabled
        topP = settings.topP
        newlineShortcut = settings.newlineShortcut
        showPromptExamples = settings.showPromptExamples
        loadModelOnLaunch = settings.loadModelOnLaunch
        agentModeEnabled = settings.agentModeEnabled
        agentToolLoop = VisionCaptureToolLoop()
    }

    private func persistSettings() {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled,
            newlineShortcut: newlineShortcut,
            showPromptExamples: showPromptExamples,
            visionResidencyPolicy: runtimeOptions.visionResidencyPolicy,
            rdadvisePolicy: runtimeOptions.rdadvisePolicy,
            loadModelOnLaunch: loadModelOnLaunch,
            agentModeEnabled: agentModeEnabled,
            toolThinkingEnabled: runtimeOptions.toolThinkingEnabled)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        try? MacAppSettingsFileStore.save(
            settings,
            forModelDirectory: modelDirectory)
    }

    private func finishInstallFailure(_ error: Error, generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        resetInstallETA()
        let hasSavedDownload = hasPartialModelDownload
        installState = hasSavedDownload ? .recoverable("\(error)") : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            let requirement = AppModelInstallRequirement(probePath: path,
                                                          requiredBytes: required,
                                                          availableBytes: available)
            installReadiness = .insufficientSpace(requirement)
        } else {
            refreshInstallReadiness()
            if hasSavedDownload {
                installState = .recoverable("\(error)")
            }
        }
    }

    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    /// `sequence` orders the phases a load emits. It is 0 for states this model
    /// raises itself, which bypass the ordering check.
    func applyLoadState(_ state: AppModelLoadState, generation: UInt64,
                        sequence: UInt64 = 0) {
        guard generation == loadGeneration else { return }
        if sequence > 0 {
            guard sequence > appliedLoadSequence else { return }
            appliedLoadSequence = sequence
        }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        // An outcome closes the load. Phases emitted before it but delivered
        // after it must not reopen one that has already finished: `.failed` is
        // raised here at sequence 0, so it never advanced the counter, and a
        // late `.loading` hop could put the UI back into a load with no task
        // left to cancel and no way to start another. `beginLoad` resets the
        // counter, so the seal lasts exactly one load.
        switch state {
        case .notLoaded, .ready, .failed:
            appliedLoadSequence = .max
        case .loading, .cancelling, .unloading:
            break
        }
        switch state {
        case .notLoaded:
            loadedRuntimeKey = nil
            // Unloading released the runner and the KV, so whatever lineage was
            // open no longer has tokens behind it.
            serviceEpoch = nil
            archiveConversationContext()
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            // A load builds a new runner and an empty KV. The service ends the
            // lineage on its side for the same reason; leaving the app's epoch
            // in place would have the next turn claim to resume onto a cache
            // that had just been rebuilt.
            serviceEpoch = nil
            // And the conversation itself is gone with that KV. Keeping the
            // turn list would leave the app numbering turns from where it left
            // off while the service, having just ended the lineage, expects
            // zero — so the gate would refuse the next turn and every turn
            // after it, for the rest of the session.
            archiveConversationContext()
            loadedRuntimeKey = pendingExplicitLoadRuntimeKey
                ?? activeRunRuntimeKey
                ?? currentRuntimeKey
            pendingExplicitLoadRuntimeKey = nil
            // The freshly loaded model's footprint, so the figure is right
            // before the first generation rather than after it.
            sampleLiveMemory()
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            error = loadError
        }
    }

    /// Releases the transcript's own references to the images it was showing.
    /// They are separate files from the composer's, so nothing else frees them.
    /// Releases every image the conversation is holding — each turn's, the
    /// archived turns', and the newest turn's.
    private func releaseConversationImages() {
        for activity in outputAgentActivities + agentActivitiesByTurnID.values.flatMap({ $0 }) {
            if let preview = activity.screenshotPreview { attachmentStore.remove(preview) }
        }
        retainedScreenshotPreviewCount = 0
        for pair in archivedPairs {
            for attachment in pair.user.images { attachmentStore.remove(attachment) }
        }
        for turn in conversation.turns {
            for attachment in turn.images { attachmentStore.remove(attachment) }
        }
        releaseTranscriptImages()
    }

    private func releaseTranscriptImages() {
        for attachment in outputImageAttachments { attachmentStore.remove(attachment) }
        outputImageAttachments = []
    }

    /// Deletes every file this session staged. Called when the app is quitting,
    /// which is the only moment they are all certainly unwanted.
    public func releaseAllAttachments() {
        releaseConversationImages()
        for attachment in imageAttachments { attachmentStore.remove(attachment) }
        imageAttachments.removeAll()
        attachmentStore.removeAll()
    }

    /// Memory comes from the process doing the work: the decode service when
    /// there is one, this process otherwise.
    private func sampleLiveMemory() {
        if let reporter = client as? any AppInferenceMemoryReporting {
            if let bytes = reporter.currentInferenceMemoryBytes {
                liveMemoryBytes = bytes
            }
            if let resident = reporter.currentInferenceResidentBytes {
                liveResidentBytes = resident
            }
            // Refreshed on every sample, so the tower figure tracks a run
            // instead of appearing only in its final diagnostics.
            if let tower = reporter.currentInferenceTowerBytes {
                visionTowerMappedBytes = tower
            }
        } else {
            liveMemoryBytes = memorySampler.sample()
            // The occupied figure, not the footprint again: this row is the one
            // that includes the mapped weights, and feeding it the footprint
            // made both numbers report the same thing.
            liveResidentBytes = memorySampler.occupiedSample()
        }
    }

    /// Resident bytes for display, on the same terms as
    /// `currentProcessMemoryBytes`.
    public var currentProcessResidentBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        if let liveResidentBytes { return liveResidentBytes }
        if let reporter = client as? any AppInferenceMemoryReporting {
            return reporter.currentInferenceResidentBytes
        }
        return memorySampler.occupiedSample()
    }

    /// Ends the conversation and starts an empty one.
    ///
    /// There is no history to recover it from, so the window confirms before
    /// calling this when the transcript is not empty.
    public func newChat() {
        guard !isRunning else { return }
        activeAgentCompaction = nil
        contextCompactionStatusText = nil
        liveStructuredProgress = nil
        thinkingPreview = nil
        toolCallPreview = nil
        agentWaitingForMCP = false
        // Every turn holds its own hard links. Dropping the turn list without
        // releasing them leaked one staged file per image per turn until quit:
        // `releaseTranscriptImages` only ever covered the newest turn, which is
        // why a single-turn test passed.
        releaseConversationImages()
        archivedPairs.removeAll()
        agentActivitiesByTurnID.removeAll()
        outputAgentActivities = []
        activeAgentActivityTurnID = nil
        conversation.startNew()
        outputPromptText = ""
        releaseTranscriptImages()
        outputText = ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
        pendingAgentInstruction = nil
        agentChecklistContinuationUsed = false
        agentToolLoop = VisionCaptureToolLoop()
        // Only the intent is recorded here; the next turn opens the new lineage
        // on the inference side. Resetting eagerly as well raced that opening
        // and sent two resets for one new chat, and it buys nothing: the KV
        // allocation is fixed, so an unsent new chat holds no extra memory.
        serviceEpoch = nil
    }

    /// Starts an empty conversation because the KV behind the old one is gone.
    ///
    /// Distinct from `newChat()`: the user did not ask for this. The transcript
    /// stays — lifecycle actions are not supposed to discard it — but its turns
    /// move to `archivedPairs`, because the model can no longer see them. The
    /// alternative, letting `conversation` keep counting, desynchronises the app
    /// from the service's gate, which has just gone back to expecting turn zero;
    /// the gate would then refuse every turn for the rest of the session.
    private func archiveConversationContext() {
        activeAgentCompaction = nil
        contextCompactionStatusText = nil
        liveStructuredProgress = nil
        thinkingPreview = nil
        toolCallPreview = nil
        agentWaitingForMCP = false
        var carried = conversation.completedPairs
        // The live fields hold the newest finished turn between runs, and it is
        // already in `completedPairs`; nothing extra to carry.
        if carried.isEmpty, !outputPromptText.isEmpty {
            carried = [(user: AppChatTurn(role: .user, text: outputPromptText,
                                          images: outputImageAttachments),
                        assistant: AppChatTurn(role: .assistant, text: outputText))]
        }
        archivedPairs.append(contentsOf: carried)
        conversation.startNew()
        // The archived pairs are what the transcript draws now. Leaving the
        // live fields holding the newest of them would draw that turn twice,
        // once as history and once as the turn still on screen.
        if !carried.isEmpty {
            outputPromptText = ""
            outputText = ""
            outputAgentActivities = []
            activeAgentActivityTurnID = nil
            outputImageAttachments = []
            generationTranscriptMailbox?.reset()
        }
    }

    /// Opens the conversation on the inference side if it has not been opened
    /// yet. Called before every turn: a load or an unload ends the lineage
    /// there without the app being asked, and the next turn has to re-open it
    /// rather than resume onto a KV that was rebuilt empty.
    private func openConversationIfNeeded() async throws {
        guard serviceEpoch != conversation.epoch else { return }
        guard let lifecycle = client as? AppModelLifecycleClient else { return }
        try await lifecycle.resetConversation(epoch: conversation.epoch)
        serviceEpoch = conversation.epoch
    }

    public func run() {
        guard canRun else { return }
        agentChecklistContinuationUsed = false
        _ = startRun(
            prompt: promptText,
            attachments: imageAttachments,
            clearsComposer: true)
    }

    @discardableResult
    private func startRun(
        prompt: String,
        attachments: [AppImageAttachment],
        clearsComposer: Bool
    ) -> Bool {
        guard !isRunning && !isAddingImages && isModelAvailable
                && !loadState.isLoading && !isVisionCompanionOperationInProgress
                && !hasStaleLoadedRuntime && conversation.canSend
                && (!prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || !attachments.isEmpty) else { return false }
        let agentConfiguration = agentModeEnabled ? makeAgentConfiguration() : nil
        // Reserved before the request is built, so the position the service
        // will check is the position the transcript shows.
        guard let ticket = conversation.beginTurn(text: prompt, images: []) else {
            return false
        }
        var request: AppGenerationRequest
        do {
            request = try makeRequest(
                ticket: ticket,
                prompt: prompt,
                imageAttachments: attachments)
        } catch let appError as AppInferenceError {
            conversation.abandonTurn()
            error = appError
            return false
        } catch {
            let appError = AppInferenceError.unknown("\(error)")
            conversation.abandonTurn()
            self.error = appError
            return false
        }

        // The run reads the transcript's own hard links rather than the
        // composer's files, so clearing the composer below cannot delete an
        // image this request has not opened yet. One set of files, one owner.
        // A failed retain leaves no reference that is guaranteed to outlive
        // the composer, so the run is refused instead of started against files
        // that are about to be removed.
        var retained: [AppImageAttachment] = []
        do {
            for attachment in request.imageAttachments {
                retained.append(try attachmentStore.retain(attachment))
            }
        } catch {
            for attachment in retained { attachmentStore.remove(attachment) }
            conversation.abandonTurn()
            imageAttachmentError = String(describing: error)
            self.error = .invalidRequest(
                "Could not prepare the attached images for this run: \(error)")
            return false
        }
        request.imageAttachments = retained

        persistSettings()

        generationTranscriptMailbox?.reset()
        runIdentity &+= 1
        let generation = runIdentity
        agentCheckpointRequested = false
        agentCheckpointRecord = nil
        agentCheckpointPrepared = false
        activeAgentCompaction = nil
        contextCompactionStatusText = nil
        outputPromptText = request.prompt
        outputAgentActivities = []
        activeAgentActivityTurnID = agentConfiguration == nil
            ? nil : conversation.turns.last?.id
        if let activeAgentActivityTurnID {
            agentActivitiesByTurnID[activeAgentActivityTurnID] = []
        }
        // Not released: every turn of a conversation keeps its own images for
        // as long as the conversation shows them. They are hard links to files
        // that already exist, so holding them costs no additional bytes.
        conversation.attachImagesToPendingTurn(retained)
        outputImageAttachments = retained
        outputText = ""
        diagnostics = nil
        error = nil
        hasHandledTerminalEvent = false
        thinkingPreview = nil
        toolCallPreview = nil
        activeRunRuntimeKey = AppLoadedRuntimeKey(
            modelDirectory: request.modelDirectory,
            maxContextTokens: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: !request.isPureGreedy)
        isCancellationPending = false
        agentCancellationIssued = false
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
        sampleLiveMemory()
        phase = .prefill
        runState = .running
        // A chat composer always clears. Keeping the sent turn in the box means
        // the next message starts as a copy of the last one, and the images
        // silently re-attach to a different message. Dropping them is safe
        // because the request above was repointed at the retained links:
        // removing these files cannot pull the ground out from under a run that
        // has not opened its images yet. A refused turn puts both back through
        // `restoreComposer`.
        if clearsComposer {
            promptText = ""
            for attachment in attachments { attachmentStore.remove(attachment) }
            imageAttachments.removeAll()
            imageAttachmentError = nil
        }

        if let agentConfiguration {
            let loop = agentToolLoop
            runTask = Task.detached {
                [weak self, client, request, generation, loop, agentConfiguration] in
                guard let self else { return }
                do {
                    try await self.openConversationIfNeeded()
                    let checkpoint: VisionCaptureToolLoop.Checkpoint?
                    if client is any AppContextCheckpointClient {
                        checkpoint = { @Sendable (proposal: AgentContextCheckpointProposal) async throws -> DecodeContextCheckpointReceipt in
                            try await self.applyAgentCheckpoint(proposal, client: client, generation: generation)
                        }
                    } else {
                        checkpoint = nil
                    }
                    let result = try await loop.run(
                        configuration: agentConfiguration,
                        activity: { event in
                            await self.applyAgentActivity(
                                event,
                                generation: generation)
                        },
                        userPrompt: request.prompt,
                        userImages: request.imageAttachments,
                        conversationEpoch: request.conversationEpoch,
                        maxContextTokens: request.maxContextTokens,
                        checkpoint: checkpoint,
                        forceCheckpoint: { await self.shouldForceAgentCheckpoint() },
                        hasPendingUserInstruction: {
                            await self.hasPendingAgentInstruction(
                                generation: generation)
                        }
                    ) { toolTurn in
                        try await self.generateAgentStep(
                            client: client,
                            baseRequest: request,
                            toolTurn: toolTurn,
                            generation: generation)
                    }
                    try Task.checkCancellation()
                    await self.finishAgentSuccessfully(
                        result,
                        generation: generation)
                } catch is CancellationError {
                    await self.finishAgentFailure(.cancelled, generation: generation)
                } catch let error as VisionCaptureAgentError {
                    await self.finishAgentFailure(
                        .conversationLineageLost(error.description),
                        generation: generation)
                } catch let appError as AppInferenceError {
                    await self.finishAgentFailure(appError, generation: generation)
                } catch {
                    await self.finishAgentFailure(
                        .conversationLineageLost(String(describing: error)),
                        generation: generation)
                }
            }
        } else {
            runTask = Task.detached { [weak self, client, request, generation] in
                guard let self else { return }
                do {
                    try await self.openConversationIfNeeded()
                    for try await event in client.generate(request) {
                        await self.apply(event, generation: generation)
                    }
                } catch let appError as AppInferenceError {
                    await self.finishStreamFailure(appError, generation: generation)
                } catch {
                    await self.finishStreamFailure(.unknown("\(error)"), generation: generation)
                }
            }
        }
        return true
    }

    /// Sends the composer text normally when idle, or queues it as a live
    /// Agent Mode instruction while a QA task is running.
    public func submitPrompt() {
        if isRunning {
            sendAgentInstruction()
        } else {
            run()
        }
    }

    public func sendAgentInstruction() {
        guard canSendAgentInstruction else { return }
        agentChecklistContinuationUsed = false
        pendingAgentInstruction = promptText
        promptText = ""
        liveStructuredProgress = nil
        isCancellationPending = true
        issueAgentCancellationAtTokenBoundaryIfReady()
    }

    public func cancel() {
        guard canCancel else { return }
        if let pendingAgentInstruction {
            self.pendingAgentInstruction = nil
            restoreUnsentAgentInstruction(pendingAgentInstruction)
        }
        liveStructuredProgress = nil
        isCancellationPending = true
        if agentModeEnabled {
            issueAgentCancellationAtTokenBoundaryIfReady()
        } else {
            client.cancel()
        }
    }

    private func issueAgentCancellationAtTokenBoundaryIfReady() {
        guard isCancellationPending, !agentCancellationIssued,
              agentModelStepActive, phase == .decode,
              liveTokenCount > 0 else { return }
        agentCancellationIssued = true
        client.cancel()
    }

    /// Callable by coordinated local comparisons. The same capacity, evidence,
    /// image and transaction checks apply. It never interrupts a live action.
    public func requestAgentContextCheckpoint() {
        guard agentModeEnabled, isRunning else { return }
        agentCheckpointRequested = true
    }

    private func shouldForceAgentCheckpoint() async -> Bool {
        if await AgentInferenceTrace.shared?.consumeCheckpointRequest() == true {
            agentCheckpointRequested = true
        }
        return agentCheckpointRequested
    }

    private func hasPendingAgentInstruction(generation: Int) -> Bool {
        generation == runIdentity && isRunning
            && pendingAgentInstruction != nil
    }

    private func applyAgentCheckpoint(_ proposal: AgentContextCheckpointProposal,
        client: any AppInferenceClient, generation: Int
    ) async throws -> DecodeContextCheckpointReceipt {
        guard generation == runIdentity, isRunning,
              let ticket = conversation.pendingTicket,
              let checkpointClient = client as? any AppContextCheckpointClient else {
            throw AppInferenceError.conversationLineageLost("The active task changed before its checkpoint.")
        }
        if proposal.trigger == .sustainedSlowDecode, !proposal.commit,
           let evidence = proposal.performanceEvidence {
            activeAgentCompaction = AgentCompactionProgress(
                id: proposal.id, trigger: proposal.trigger,
                performanceEvidence: evidence,
                existingPromptTokens: evidence.conversationTokens,
                replacementPromptTokens: nil)
            updateAgentCompactionActivity(
                "Performance compaction checking at \(evidence.conversationTokens) input tokens after \(evidence.completedDecisions) decisions at \(Self.compactionRateText(evidence.weightedTokensPerSecond)) tokens/second.",
                status: .dispatching)
        } else if proposal.trigger == .sustainedSlowDecode, proposal.commit {
            updateAgentCompactionActivity(
                "Performance compaction starting. Replacing the settled task history without replaying app input.",
                status: .dispatching)
        }
        let request = DecodeContextCheckpointRequest(checkpointID: proposal.id,
            sourceEpoch: ticket.epoch, sourceTurnIndex: ticket.index,
            replacementEpoch: proposal.replacementEpoch,
            pendingCall: DecodeToolCall(id: proposal.call.id, name: proposal.call.name,
                argumentsJSON: try proposal.call.arguments.encoded()),
            result: DecodeToolResult(callID: proposal.result.callID, name: proposal.result.name,
                content: proposal.result.content, imageAttachments: proposal.result.imageAttachments.map {
                    DecodeImageAttachment(id: $0.id, path: $0.fileURL.path, displayName: $0.displayName,
                        encodedBytes: $0.encodedBytes, sha256: $0.sha256)
                }),
            record: proposal.record, commit: proposal.commit, force: proposal.force,
            trigger: proposal.trigger, performanceEvidence: proposal.performanceEvidence,
            permitsScreenshot: proposal.permitsScreenshot)
        let receipt: DecodeContextCheckpointReceipt
        if proposal.commit { agentCheckpointRecord = proposal.record }
        do {
            receipt = try await checkpointClient.contextCheckpoint(request)
        } catch {
            await AgentInferenceTrace.shared?.checkpointFailure(id: proposal.id, commit: proposal.commit,
                error: String(describing: error))
            if proposal.trigger == .sustainedSlowDecode {
                let cancelled = error is CancellationError
                    || (error as? AppInferenceError) == .cancelled
                updateAgentCompactionActivity(
                    cancelled
                        ? "Performance compaction cancelled. The rebuild was not confirmed."
                        : "Performance compaction failed before a rebuild was confirmed: \(error)",
                    status: cancelled ? .cancelled
                        : .localFailure(reason: String(describing: error)))
                activeAgentCompaction = nil
            }
            throw error
        }
        guard generation == runIdentity, isRunning else {
            throw AppInferenceError.conversationLineageLost("The task changed while the checkpoint was being acknowledged.")
        }
        await AgentInferenceTrace.shared?.checkpoint(receipt, callID: proposal.call.id,
            sourceEpoch: ticket.epoch, trigger: proposal.trigger,
            performanceEvidence: proposal.performanceEvidence)
        if receipt.committed {
            guard conversation.acceptCheckpoint(epoch: receipt.replacementEpoch, source: ticket) else {
                throw AppInferenceError.conversationLineageLost("The service accepted a checkpoint for a different visible task.")
            }
            serviceEpoch = receipt.replacementEpoch
            agentCheckpointRecord = proposal.record
            agentCheckpointRequested = false
            guard let replacementCount = receipt.replacementPromptTokens else {
                throw AppInferenceError.conversationLineageLost("The accepted checkpoint omitted its rendered token count.")
            }
            activeAgentCompaction = AgentCompactionProgress(id: proposal.id,
                trigger: proposal.trigger,
                performanceEvidence: proposal.performanceEvidence,
                existingPromptTokens: receipt.existingPromptTokens,
                replacementPromptTokens: replacementCount)
            let prefix = proposal.trigger == .sustainedSlowDecode
                ? "Performance compaction" : "Compacting history"
            updateAgentCompactionActivity(
                "\(prefix): \(receipt.existingPromptTokens) → \(replacementCount) input tokens. Rebuilding model context.",
                status: .dispatching)
        } else if proposal.trigger == .sustainedSlowDecode, proposal.commit {
            updateAgentCompactionActivity(
                "Performance compaction failed because the service did not acknowledge the rebuild.",
                status: .localFailure(reason: "The checkpoint commit was not acknowledged."))
            activeAgentCompaction = nil
        } else if receipt.needed {
            if proposal.trigger != .sustainedSlowDecode {
                agentCheckpointPrepared = true
            }
            activeAgentCompaction = AgentCompactionProgress(id: proposal.id,
                trigger: proposal.trigger,
                performanceEvidence: proposal.performanceEvidence,
                existingPromptTokens: receipt.existingPromptTokens)
            let text = proposal.trigger == .sustainedSlowDecode
                ? "Performance compaction starting at \(receipt.existingPromptTokens) input tokens."
                : "Compacting history…"
            updateAgentCompactionActivity(text, status: .dispatching)
        } else if proposal.trigger == .sustainedSlowDecode {
            let candidate = receipt.replacementPromptTokens.map(String.init) ?? "unavailable"
            let minimum = receipt.performanceMinimumSavingsTokens.map(String.init) ?? "unavailable"
            updateAgentCompactionActivity(
                "Performance compaction skipped. Candidate: \(candidate) input tokens. Required minimum saving: \(minimum) tokens. The current conversation continues.",
                status: .succeeded)
            activeAgentCompaction = nil
        }
        return receipt
    }

    private static func compactionRateText(_ rate: Double) -> String {
        String(format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), rate)
    }

    private func updateAgentCompactionActivity(_ text: String, status: AppAgentActivity.Status) {
        guard let compaction = activeAgentCompaction else { return }
        contextCompactionStatusText = text
        let activity = AppAgentActivity(id: compaction.id, kind: .contextCompaction,
            body: text, status: status)
        if let index = outputAgentActivities.firstIndex(where: { $0.id == compaction.id }) {
            outputAgentActivities[index] = activity
        } else {
            outputAgentActivities.append(activity)
        }
    }

    private func completeAgentCompactionRebuild(checkpointID: UUID?, generation: Int) {
        guard generation == runIdentity, isRunning,
              var compaction = activeAgentCompaction, compaction.id == checkpointID,
              let replacementCount = compaction.replacementPromptTokens else { return }
        if compaction.trigger == .sustainedSlowDecode {
            guard !compaction.rebuildConfirmed else { return }
            compaction.rebuildConfirmed = true
            activeAgentCompaction = compaction
            updateAgentCompactionActivity(
                "Performance compaction rebuilt model context: \(compaction.existingPromptTokens) → \(replacementCount) input tokens. Measuring the first completed decision.",
                status: .dispatching)
            return
        }
        updateAgentCompactionActivity(
            "History compacted: \(compaction.existingPromptTokens) → \(replacementCount) input tokens. Model context rebuilt.",
            status: .succeeded)
        activeAgentCompaction = nil
    }

    private func completePerformanceCompactionMeasurement(
        checkpointID: UUID?, diagnostics: AppDiagnostics, generation: Int
    ) async {
        guard generation == runIdentity, isRunning,
              let compaction = activeAgentCompaction,
              compaction.id == checkpointID,
              compaction.trigger == .sustainedSlowDecode,
              compaction.rebuildConfirmed,
              let replacementCount = compaction.replacementPromptTokens else { return }
        let afterContext = diagnostics.conversationTokens
        let afterRate = diagnostics.generatedTokens > 0
            && diagnostics.decodeSeconds.isFinite && diagnostics.decodeSeconds > 0
            ? Double(diagnostics.generatedTokens) / diagnostics.decodeSeconds : nil
        let afterContextText = afterContext.map(String.init) ?? "unavailable"
        let afterRateText = afterRate.map(Self.compactionRateText) ?? "unavailable"
        updateAgentCompactionActivity(
            "Performance compaction completed: \(compaction.existingPromptTokens) → \(replacementCount) input tokens. First completed decision: \(afterContextText) context tokens at \(afterRateText) tokens/second.",
            status: .succeeded)
        await AgentInferenceTrace.shared?.checkpointCompleted(
            id: compaction.id, before: compaction.performanceEvidence,
            replacementPromptTokens: replacementCount, after: diagnostics)
        activeAgentCompaction = nil
    }

    private func interruptAgentCompaction(_ appError: AppInferenceError) {
        guard let compaction = activeAgentCompaction else { return }
        let cancelled = appError == .cancelled
        let outcome = cancelled ? "cancelled" : "failed"
        let detail: String
        if compaction.rebuildConfirmed, let replacement = compaction.replacementPromptTokens {
            detail = " Model context rebuilt from \(compaction.existingPromptTokens) to \(replacement) input tokens, but the resumed decision did not complete."
        } else if let replacement = compaction.replacementPromptTokens {
            detail = " \(compaction.existingPromptTokens) → \(replacement) input tokens were accepted, but the rebuild was not confirmed complete."
        } else {
            detail = " No completed rebuild was confirmed."
        }
        updateAgentCompactionActivity("History compaction \(outcome)." + detail,
            status: cancelled ? .cancelled : .localFailure(reason: appError.description))
        activeAgentCompaction = nil
    }

    private func agentRequest(_ base: AppGenerationRequest, toolTurn: AppToolTurn,
                              generation: Int) throws -> AppGenerationRequest {
        guard generation == runIdentity, isRunning, let ticket = conversation.pendingTicket else {
            throw CancellationError()
        }
        var request = base
        request.conversationEpoch = ticket.epoch
        request.turnIndex = ticket.index
        request.toolTurn = toolTurn
        if case .checkpoint = toolTurn {
            request.prompt = agentCheckpointRecord ?? ""
            request.imageAttachments = []
            request.conversationTokens = 0
            request.runtimeOptions.prefillChunkTokens = 256
        } else if case .results(let results) = toolTurn {
            request.prompt = ""
            request.imageAttachments = results.flatMap(\.imageAttachments)
        }
        return request
    }

    private nonisolated func generateAgentStep(
        client: any AppInferenceClient,
        baseRequest: AppGenerationRequest,
        toolTurn: AppToolTurn,
        generation: Int,
        allowsMalformedRegeneration: Bool = true
    ) async throws -> VisionCaptureModelCompletion {
        await beginAgentStep(generation: generation)
        var request = try await agentRequest(baseRequest, toolTurn: toolTurn, generation: generation)
        let trace = AgentInferenceTrace.shared
        let traceStep = await trace?.begin(request)
        request.captureToolFailureEvidence = traceStep != nil
        request.captureGPUCompletionTiming = traceStep != nil
        request.runtimeMeasurementCapture = await trace?.runtimeMeasurementRequest(for: traceStep)
        var content = ""
        var calls: [AppToolCall] = []
        var terminal: AppDiagnostics?
        var streamFailure: AppInferenceError?
        var previousStepElapsed = 0.0
        var previousStepTokenCount = 0
        var latestStructuredProgress: DecodeStructuredProgress?
        var latestToolCallPreview: DecodeToolCallPreview?
        var lastProgressTrace = ContinuousClock.now
        let checkpointID: UUID?
        if case .checkpoint(let id) = toolTurn { checkpointID = id } else { checkpointID = nil }
        do {
            await recordAgentModelInput(
                request, isFormatCorrection: !allowsMalformedRegeneration,
                generation: generation)
            generationEvents: for try await event in client.generate(request) {
                switch event {
                case .token(let token):
                    // The first decode event proves this checkpoint's prefill
                    // completed. Keep the indicator up throughout the rebuild.
                    if checkpointID != nil, previousStepTokenCount == 0 {
                        await completeAgentCompactionRebuild(checkpointID: checkpointID, generation: generation)
                    }
                    content += token.textDelta
                    latestStructuredProgress = token.structuredProgress
                    latestToolCallPreview = token.toolCallPreview
                    let observedCount = max(0, token.index + 1)
                    let tokenCountDelta = max(0, observedCount - previousStepTokenCount)
                    let elapsedDelta = max(
                        0, token.elapsedDecodeSeconds - previousStepElapsed)
                    previousStepTokenCount = observedCount
                    previousStepElapsed = token.elapsedDecodeSeconds
                    await applyAgentToken(
                        text: token.textDelta,
                        tokenCountDelta: tokenCountDelta,
                        elapsedDelta: elapsedDelta,
                        structuredProgress: token.structuredProgress,
                        thinkingPreview: token.thinkingPreview,
                        toolCallPreview: token.toolCallPreview,
                        generation: generation)
                    if traceStep != nil,
                       lastProgressTrace.duration(to: .now) >= .seconds(30) {
                        lastProgressTrace = .now
                        await trace?.generationProgress(
                            traceStep, progress: token.structuredProgress,
                            tokens: observedCount, elapsedSeconds: token.elapsedDecodeSeconds,
                            toolCallPreview: latestToolCallPreview)
                    }
                case .toolCall(let call):
                    calls.append(call)
                case .finished(let diagnostics):
                    terminal = diagnostics
                    await applyAgentStep(event, generation: generation)
                case .cancelled(let diagnostics):
                    terminal = diagnostics
                    await endAgentStep(generation: generation)
                    break generationEvents
                case .failed(let error, let partial):
                    terminal = partial
                    await endAgentStep(generation: generation)
                    // Drain through stream termination before considering a
                    // retry. Abandoning this iterator can cancel the next run.
                    streamFailure = error
                case .memorySample, .prefillProgress:
                    await applyAgentStep(event, generation: generation)
                }
            }
            if let streamFailure { throw streamFailure }
            // Cancelling stream iteration can end it without a terminal event.
            // Preserve the owner's Stop before classifying a missing answer.
            try Task.checkCancellation()
            guard let terminal else {
                throw VisionCaptureAgentError.incompleteAnswer
            }
            if checkpointID != nil {
                await completeAgentCompactionRebuild(checkpointID: checkpointID, generation: generation)
                await completePerformanceCompactionMeasurement(
                    checkpointID: checkpointID, diagnostics: terminal,
                    generation: generation)
            }
            if let reporter = client as? any AppInferenceTranscriptReporting {
                content = reporter.generationTranscriptMailbox.completeText
            }
            await trace?.finish(
                traceStep, content: content, calls: calls, diagnostics: terminal,
                structuredProgress: latestStructuredProgress)
            return VisionCaptureModelCompletion(
                content: content,
                toolCalls: calls,
                diagnostics: terminal)
        } catch {
            await endAgentStep(generation: generation)
            await trace?.finish(
                traceStep, content: content, calls: calls, diagnostics: terminal,
                error: String(describing: error), cancelled: error is CancellationError,
                parserFailure: (error as? AppInferenceError).flatMap {
                    if case .structuredToolFailure(_, _, let evidence) = $0 { return evidence }
                    return nil
                }, thoughtRepetitionRecovery: (error as? AppInferenceError).flatMap {
                    if case .repeatedThought(let receipt) = $0 { return receipt }
                    return nil
                }, structuredProgress: latestStructuredProgress,
                toolCallPreview: latestToolCallPreview)
            if let failure = error as? AppInferenceError,
               case .repeatedThought = failure, !calls.isEmpty {
                throw AppInferenceError.invalidRequest(
                    "The repetition receipt conflicted with completed tool-call output. No recovery or action was admitted.")
            }
            if allowsMalformedRegeneration, calls.isEmpty,
               case .results(let results) = toolTurn,
               let failure = error as? AppInferenceError,
               case .structuredToolFailure(_, true, _) = failure {
                try Task.checkCancellation()
                // The pending tool result was rolled back, not its preceding
                // app action. Reprocess that result once without invoking MCP.
                let feedback = """


                Host format correction: Your previous response was malformed and no proposed action was executed. Make one visioncapture_navigate call using its schema and the latest permitted choices. Use Gemma's native string delimiters. Do not replay earlier input.
                """
                let corrected = results.map { result in
                    return AppToolResult(callID: result.callID, name: result.name,
                        content: result.content + feedback, imageAttachments: result.imageAttachments)
                }
                return try await generateAgentStep(
                    client: client, baseRequest: baseRequest,
                    toolTurn: .results(corrected), generation: generation,
                    allowsMalformedRegeneration: false)
            }
            if !allowsMalformedRegeneration, calls.isEmpty,
               case .results = toolTurn,
               let failure = error as? AppInferenceError,
               case .structuredToolFailure(_, true, let evidence) = failure {
                throw AppInferenceError.structuredToolFailure(
                    message: "The model still produced an invalid tool request after one format correction. This invalid request was not sent. Generation stopped.",
                    canRegenerateToolResult: true,
                    evidence: evidence)
            }
            throw error
        }
    }

    /// Live decode counters describe the current model step. A tool result
    /// starts a new prompt whose absolute prefill total already contains every
    /// earlier step, so carrying their generated-token count forward would
    /// double-count context growth in the HUD.
    private func beginAgentStep(generation: Int) {
        guard generation == runIdentity, agentModeEnabled, isRunning else { return }
        agentModelStepActive = true
        // Cancellation belongs to one decode step. A completed tool call can
        // race a queued instruction; its required not-sent tool result then
        // opens another model step which must receive a fresh cancellation.
        agentCancellationIssued = false
        liveStructuredProgress = nil
        thinkingPreview = nil
        toolCallPreview = nil
        agentWaitingForMCP = false
        phase = .prefill
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
    }

    private func endAgentStep(generation: Int) {
        guard generation == runIdentity else { return }
        agentModelStepActive = false
    }

    private func recordAgentModelInput(
        _ request: AppGenerationRequest, isFormatCorrection: Bool, generation: Int
    ) {
        guard generation == runIdentity, isRunning else { return }
        var body = ""
        func appendImages(_ images: [AppImageAttachment]) {
            body += "\nImage attachments: \(images.count) (image bytes supplied separately)\n"
            for image in images {
                body += "\(image.displayName)\nReference: \(image.fileURL.path)\nSHA-256: \(image.sha256)\n"
            }
        }
        switch request.toolTurn {
        case .checkpoint(let id):
            body = "Explicit checkpoint resume: \(id.uuidString). Original system/tool configuration and retained image features are reconstructed by the service.\n\(request.prompt)"
        case .user(let developerPrompt, let tools):
            if let developerPrompt {
                body += "System instructions added to this model context:\n\(developerPrompt)\n\n"
            }
            body += "User message:\n\(request.prompt)\n"
            appendImages(request.imageAttachments)
            // Tool definitions are encoded only when tool mode opens. Later
            // user turns carry the same definitions as a consistency check,
            // while the model reuses the definitions already in its KV prefix.
            if developerPrompt != nil {
                for tool in tools {
                    body += "\nTool definition added to this model context: \(tool.name)\n\(tool.description)\nParameters (JSON formatted for display):\n"
                    body += Self.prettyAgentActivityJSON(tool.parameters) + "\n"
                }
            }
        case .results(let results):
            for result in results {
                body += "Tool result: \(result.name)\nCall: \(result.callID)\n\(result.content)\n"
                appendImages(result.imageAttachments)
            }
        case nil:
            body += "User message:\n\(request.prompt)\n"
            appendImages(request.imageAttachments)
        }
        let attempt = outputAgentActivities.reduce(1) { count, activity in
            if case .modelInput = activity.kind { return count + 1 }
            return count
        }
        outputAgentActivities.append(AppAgentActivity(
            id: UUID(), kind: .modelInput(attempt: attempt, isFormatCorrection: isFormatCorrection),
            body: body, status: .succeeded))
    }

    private func applyAgentToken(
        text: String,
        tokenCountDelta: Int,
        elapsedDelta: Double,
        structuredProgress: DecodeStructuredProgress?,
        thinkingPreview: DecodeThinkingPreview?,
        toolCallPreview: DecodeToolCallPreview?,
        generation: Int
    ) {
        guard generation == runIdentity, agentModeEnabled, isRunning else { return }
        phase = .decode
        liveStructuredProgress = structuredProgress
        if let thinkingPreview { self.thinkingPreview = thinkingPreview }
        self.toolCallPreview = toolCallPreview
        liveTokenCount += tokenCountDelta
        liveElapsedDecodeSeconds += elapsedDelta
        sampleLiveMemory()
        if !text.isEmpty {
            outputText += text
        }
        issueAgentCancellationAtTokenBoundaryIfReady()
    }

    private func applyAgentStep(_ event: AppInferenceEvent, generation: Int) {
        guard generation == runIdentity, agentModeEnabled, isRunning else { return }
        switch event {
        case .memorySample:
            sampleLiveMemory()
        case .prefillProgress(let done, let total):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
            sampleLiveMemory()
        case .token:
            break
        case .finished(let diagnostics):
            self.diagnostics = diagnostics
            visionTowerMappedBytes = diagnostics.visionTowerMappedBytes
            liveStructuredProgress = nil
            agentModelStepActive = false
            phase = .idle
        case .toolCall, .cancelled, .failed:
            break
        }
    }

    private func applyAgentActivity(
        _ event: VisionCaptureActivityEvent,
        generation: Int
    ) async {
        guard generation == runIdentity,
              agentModeEnabled,
              isRunning,
              activeAgentActivityTurnID != nil else { return }
        switch event {
        case .outgoingRequest(let id, let arguments):
            agentWaitingForMCP = true
            liveStructuredProgress = nil
            outputAgentActivities.append(AppAgentActivity(
                id: id,
                kind: .mcpRequest,
                body: Self.prettyAgentActivityJSON(arguments),
                status: .dispatching))
        case .generationRecovery(let id, let text, let status):
            let activity = AppAgentActivity(id: id, kind: .generationRecovery,
                body: text, status: status)
            if let index = outputAgentActivities.firstIndex(where: { $0.id == id }) {
                outputAgentActivities[index] = activity
            } else {
                outputAgentActivities.append(activity)
            }
        case .requestStatus(let id, let status, let elapsedSeconds):
            agentWaitingForMCP = false
            guard let index = outputAgentActivities.firstIndex(
                where: { $0.id == id }) else { return }
            outputAgentActivities[index].status = status
            outputAgentActivities[index].elapsedSeconds = elapsedSeconds
        case .incomingResponse(let id, let excerpt):
            guard let index = outputAgentActivities.firstIndex(
                where: { $0.id == id && $0.kind == .mcpRequest }) else { return }
            outputAgentActivities[index].responseBody = excerpt
        case .modelResult(let callID, let toolName, let excerpt, let imageCount):
            outputAgentActivities.append(AppAgentActivity(
                id: UUID(),
                kind: .modelResult(callID: callID, toolName: toolName, imageCount: imageCount),
                body: excerpt,
                status: .succeeded))
        case .screenshot(let id, let image):
            guard let index = outputAgentActivities.firstIndex(where: { $0.id == id }) else { return }
            outputAgentActivities[index].screenshotPreviewUnavailable = "Screenshot preview unavailable."
            guard retainedScreenshotPreviewCount < 32 else {
                outputAgentActivities[index].screenshotPreviewUnavailable =
                    "Preview not retained: this chat has reached its 32-screenshot display limit."
                return
            }
            let store = attachmentStore
            let preview = await Task.detached(priority: .userInitiated) {
                try? VisionCaptureScreenshot.stagePreview(of: image, in: store)
            }.value
            guard generation == runIdentity, isRunning,
                  let currentIndex = outputAgentActivities.firstIndex(where: { $0.id == id }) else {
                if let preview { store.remove(preview) }
                return
            }
            if let preview {
                outputAgentActivities[currentIndex].screenshotPreview = preview
                outputAgentActivities[currentIndex].screenshotPreviewUnavailable = nil
                retainedScreenshotPreviewCount += 1
            } else {
                outputAgentActivities[currentIndex].screenshotPreviewUnavailable = "Screenshot preview unavailable."
            }
        case .localRejection(let id, let call, let reason):
            let proposal = JSONValue.object([
                "name": .string(call.name),
                "arguments": call.arguments,
            ])
            outputAgentActivities.append(AppAgentActivity(
                id: id,
                kind: .localProposal,
                body: Self.prettyAgentActivityJSON(proposal),
                status: .notSent(reason: reason)))
        }
    }

    private static func prettyAgentActivityJSON(_ value: JSONValue) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(value) else {
            return "Unable to display this tool call."
        }
        return String(decoding: data, as: UTF8.self)
    }

    private func finishAgentSuccessfully(
        _ result: VisionCaptureAgentRunResult,
        generation: Int
    ) async {
        guard generation == runIdentity, !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        let exhaustedChecklistCorrection = result.followUpPrompt != nil
            && agentChecklistContinuationUsed
        let answer = exhaustedChecklistCorrection
            ? "Incomplete QA report: Gemma stopped again with pending or in-progress checks.\n\n"
                + result.answer
            : result.answer
        outputText = answer
        diagnostics = result.diagnostics
        conversation.completeTurn(text: answer, diagnostics: result.diagnostics)
        finishTerminalRun()
        if let instruction = pendingAgentInstruction {
            await continueAgentTask(with: instruction)
            return
        }
        guard let followUp = result.followUpPrompt,
              !agentChecklistContinuationUsed else { return }
        agentChecklistContinuationUsed = true
        await continueAgentTask(with: "[Agent Mode continuation]\n" + followUp)
    }

    private func continueAgentTask(with instruction: String) async {
        do {
            try await agentToolLoop.prepareForUserInstruction(
                configuration: makeAgentConfiguration())
        } catch {
            if pendingAgentInstruction == instruction {
                pendingAgentInstruction = nil
            }
            restoreUnsentAgentInstruction(instruction)
            self.error = .conversationLineageLost(String(describing: error))
            return
        }
        if startRun(prompt: instruction, attachments: [], clearsComposer: false) {
            if pendingAgentInstruction == instruction {
                pendingAgentInstruction = nil
            }
        } else {
            if pendingAgentInstruction == instruction {
                pendingAgentInstruction = nil
            }
            restoreUnsentAgentInstruction(instruction)
        }
    }

    private func finishAgentFailure(_ appError: AppInferenceError, generation: Int) async {
        guard generation == runIdentity, !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true

        // A live instruction stops the current model step at its next token
        // boundary. If that boundary falls inside an unfinished structured
        // tool span, the parser correctly refuses to commit the partial call.
        // That refusal is an interruption, not a failed user instruction: keep
        // the completed QA evidence, move the uncommitted turn out of the old
        // KV lineage, and immediately send the exact queued instruction in a
        // fresh lineage reconstructed from the host checkpoint.
        if let instruction = pendingAgentInstruction,
           agentCancellationIssued,
           Self.isRecoverableAgentInstructionInterruption(appError) {
            pendingAgentInstruction = nil
            interruptAgentCompaction(appError)
            let interruption = "Interrupted by a new user instruction before an unfinished tool call was sent."
            if !outputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                outputText += "\n\n" + interruption
            } else {
                outputText = interruption
            }
            conversation.interruptUncommittedTurn(
                text: outputText,
                stopReason: .cancelled)
            error = nil
            finishTerminalRun()
            archiveConversationContext()
            await continueAgentTask(with: instruction)
            return
        }

        error = appError
        interruptAgentCompaction(appError)
        if conversation.checkpointCount > 0 || agentCheckpointPrepared {
            let interruption = "Task interrupted during context compaction or its resumed work. Completed actions remain recorded. \(appError)"
            if !outputText.isEmpty { outputText += "\n\n" }
            outputText += interruption
            conversation.interruptAfterCheckpoint(text: outputText)
        } else if let abandoned = conversation.markLineageLost() {
            restoreComposer(from: abandoned)
        }
        if let instruction = pendingAgentInstruction {
            pendingAgentInstruction = nil
            restoreUnsentAgentInstruction(instruction)
        }
        finishTerminalRun()
    }

    private static func isRecoverableAgentInstructionInterruption(
        _ error: AppInferenceError
    ) -> Bool {
        switch error {
        case .cancelled, .structuredToolFailure:
            return true
        default:
            return false
        }
    }

    private func restoreUnsentAgentInstruction(_ instruction: String) {
        guard promptText != instruction else { return }
        if promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            promptText = instruction
        } else {
            promptText = instruction + "\n\n" + promptText
        }
    }

    public func makeRequest(
        ticket: AppConversation.Ticket? = nil,
        prompt promptOverride: String? = nil,
        imageAttachments attachmentOverride: [AppImageAttachment]? = nil
    ) throws -> AppGenerationRequest {
        // A run executes against the session that is actually loaded. Sending
        // the current settings instead meant that changing Context, Slots or
        // image residency and pressing Generate — without reloading first —
        // was refused outright with "generation runtime options do not match
        // the loaded session". The settings still apply on reload, which is
        // what the Memory section promises; they simply no longer break the
        // run in the meantime.
        let effective = loadedRuntimeKey ?? currentRuntimeKey
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            prompt: promptOverride ?? promptText,
            imageAttachments: attachmentOverride ?? imageAttachments,
            maxNewTokens: maxNewTokensOverride ?? effective.maxContextTokens,
            maxContextTokens: effective.maxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: effective.options(
                prefillEnabled: runtimeOptions.prefillEnabled,
                prefillChunkTokens: runtimeOptions.prefillChunkTokens),
            // Carried whether or not a ticket exists: the image budget has to
            // fit around the conversation even while the composer is only being
            // validated.
            // The decode service overrides this from its own gate, but the
            // in-process client reads it directly — and without it that client
            // ran every turn through the single-prompt path while the app drew a
            // growing transcript, so the model saw only the newest message.
            continuesConversation: ticket != nil,
            // Unknown means the runtime committed a turn but did not report its
            // position. Reserve the whole window: text can still continue on
            // the service's own exact state, while every image fails closed.
            conversationTokens: conversation.kvTokens ?? effective.maxContextTokens,
            conversationEpoch: ticket?.epoch,
            turnIndex: ticket?.index)
        try request.validate(requireModelDirectory: true)
        return request
    }

    func apply(_ event: AppInferenceEvent, generation: Int? = nil) {
        guard generation == nil || generation == runIdentity else { return }
        switch event {
        case .memorySample:
            sampleLiveMemory()
        case .prefillProgress(let done, let total):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
            sampleLiveMemory()
        case .token(let token):
            phase = .decode
            liveTokenCount = token.index + 1
            liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
            sampleLiveMemory()
            if !token.textDelta.isEmpty {
                outputText += token.textDelta
            }
        case .toolCall:
            finishWithError(.conversationLineageLost(
                "The model returned a tool call while Agent Mode was off."))
        case .finished(let diagnostics):
            visionTowerMappedBytes = diagnostics.visionTowerMappedBytes
            finishSuccessfully(diagnostics)
        case .cancelled(let diagnostics):
            finishCancelled(diagnostics)
        case .failed(let appError, let partial):
            diagnostics = partial
            materializeServiceTranscript()
            finishWithError(appError)
        }
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        conversation.completeTurn(text: outputText, diagnostics: diagnostics)
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        error = .cancelled
        // A `.cancelled` event means the run threw `CancellationError`, and the
        // conversation rewound the turn: its tokens are not in the KV, and the
        // service did not count it either. Counting it here put the app one
        // ahead for the rest of the conversation, so the next turn carried an
        // index the gate refused — and that refusal reached the user as
        // "decode service runtime profile changed during generation".
        //
        // A stop that lands at a token boundary is a different event: the run
        // returns normally with `.cancelled` as its *stop reason*, arrives as
        // `.finished`, and is committed by `finishSuccessfully`.
        // Not counted. The live fields are not part of `conversation.turns`, so
        // what stays on screen is the stopped turn as the current one, exactly
        // as the single-prompt path left it — the next run replaces it, and it
        // never enters the history the transcript freezes.
        //
        // The turn itself is handed back rather than dropped: discarding it lost
        // the user's message and stranded its retained image links, which the
        // next run overwrote without releasing — one staged file per image,
        // until quit.
        if let abandoned = conversation.abandonTurn() {
            restoreComposer(from: abandoned)
        }
        finishTerminalRun()
    }

    private func materializeServiceTranscript() {
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        outputText = reporter.generationTranscriptMailbox.completeText
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        error = appError
        let abandoned: AppChatTurn?
        if case .conversationLineageLost = appError {
            // The transcript stays readable; nothing further can be sent until
            // New chat.
            abandoned = conversation.markLineageLost()
        } else {
            // The runtime rewound this turn, so it is in neither the KV nor the
            // transcript. Give the user their message back instead of making
            // them retype it.
            abandoned = conversation.abandonTurn()
        }
        if let abandoned {
            restoreComposer(from: abandoned)
        }
        finishTerminalRun()
    }

    /// Puts a turn that never reached the model back in the composer.
    ///
    /// The images move with it: they are this turn's retained links, and the
    /// turn is gone from the transcript, so nothing else refers to them. Losing
    /// them here would silently drop attachments the user had picked.
    private func restoreComposer(from turn: AppChatTurn) {
        if promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            promptText = turn.text
        }
        if imageAttachments.isEmpty, !turn.images.isEmpty {
            imageAttachments = turn.images
        } else {
            for attachment in turn.images { attachmentStore.remove(attachment) }
        }
        outputImageAttachments = []
    }

    private func finishStreamFailure(_ appError: AppInferenceError, generation: Int) {
        guard generation == runIdentity else { return }
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        liveStructuredProgress = nil
        agentWaitingForMCP = false
        agentModelStepActive = false
        for index in outputAgentActivities.indices
            where outputAgentActivities[index].kind == .generationRecovery
                && outputAgentActivities[index].status == .dispatching {
            let cancelled = error == .cancelled
            outputAgentActivities[index] = AppAgentActivity(
                id: outputAgentActivities[index].id, kind: .generationRecovery,
                body: cancelled ? "Thinking recovery cancelled."
                    : "Generation ended before thinking recovery completed.",
                status: cancelled ? .cancelled : .localFailure(reason: "Recovery incomplete"))
        }
        if let activeAgentActivityTurnID {
            // The live array drives in-flight rendering. Snapshot it once when
            // the turn ends instead of copying an ever-growing audit trail into
            // history after every request and status event.
            agentActivitiesByTurnID[activeAgentActivityTurnID] = outputAgentActivities
        }
        phase = .idle
        runState = .idle
        isCancellationPending = false
        agentCancellationIssued = false
        activeRunRuntimeKey = nil
        activeAgentActivityTurnID = nil
        runTask = nil
    }

    private func clearLoadTask(generation: UInt64) {
        guard generation == loadGeneration else { return }
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
    }

    private func clearUnloadTask(generation: UInt64) {
        guard generation == unloadGeneration else { return }
        unloadTask = nil
    }
}
