import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI
import UniformTypeIdentifiers

struct InspectorView: View {
    @Bindable var model: AppModel
    @State private var showingQwenSourceImporter = false
    @State private var qwenSourceIncludesVision = true
    @State private var qwenSourcePickerError: String?

    var body: some View {
        Form {
            agentModeSection
            thinkingSection
            modelSection
            qwenSourceSection
            // Second, beside Model: image support is an install concern, not a
            // diagnostic. Last put it under the runner diagnostics and below the
            // fold, where the one screen that must mention it - the empty state
            // before any model exists - could not.
            if showsVisionSection {
                visionSection
            }
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(nsColor: .windowBackgroundColor))
        .fileImporter(
            isPresented: $showingQwenSourceImporter,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let URLs):
                guard let directory = URLs.first else { return }
                qwenSourcePickerError = nil
                model.configureQwenSourceDirectory(
                    directory,
                    includesVision: qwenSourceIncludesVision)
            case .failure(let error):
                if (error as NSError).code != NSUserCancelledError {
                    qwenSourcePickerError = error.localizedDescription
                }
            }
        }
        .onAppear {
            if model.qwenSourceDirectory != nil {
                qwenSourceIncludesVision = model.qwenSourceIncludesVision
            }
        }
    }

    private var agentModeSection: some View {
        Section("Agent Mode") {
            Toggle("Enable Agent Mode", isOn: agentModeBinding)
                .toggleStyle(.switch)
                .disabled(!model.canChangeAgentMode)

            if model.agentModeEnabled {
                Text("Adds VisionCapture tools to the existing loaded model and chat. VisionCapture must be running on its default local endpoint.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                TextField("Application bundle identifier", text: agentBundleBinding)
                    .textFieldStyle(.roundedBorder)
                    .disabled(model.isRunning)
                TextField("Simulator UDID", text: agentUDIDBinding)
                    .textFieldStyle(.roundedBorder)
                    .disabled(model.isRunning)
            }
        }
    }

    private var agentModeBinding: Binding<Bool> {
        Binding {
            model.agentModeEnabled
        } set: { enabled in
            model.setAgentModeEnabled(enabled)
        }
    }

    private var thinkingSection: some View {
        Section("Thinking") {
            Toggle("Enable \(model.selectedModelEntry.displayName) Thinking", isOn: Binding {
                model.toolThinkingEnabled
            } set: { enabled in
                model.setToolThinkingEnabled(enabled)
            })
            .toggleStyle(.switch)
            .disabled(!model.canChangeToolThinking)

            Text(model.toolThinkingStatus)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var agentBundleBinding: Binding<String> {
        Binding {
            model.agentBundleIdentifier
        } set: { value in
            model.setAgentBundleIdentifier(value)
        }
    }

    private var agentUDIDBinding: Binding<String> {
        Binding {
            model.agentSimulatorUDID
        } set: { value in
            model.setAgentSimulatorUDID(value)
        }
    }

    /// Shown while it has something to say, hidden once it does not. Unlike
    /// root this does not require an installed text model: the empty state is
    /// exactly where someone decides whether this app does what they need, and
    /// hiding image support until after a 14.62 GB download meant nobody found
    /// out it existed. It still collapses once the pack is installed and healthy.
     private var showsVisionSection: Bool {
        VisionSectionVisibility.shows(
            visionRuntimeEnabled: model.visionRuntimeEnabled,
            visionRuntimeSupported: model.isVisionRuntimeSupported,
            isModelInstalled: model.isModelInstalled,
            isVisionPackInstalled: model.isVisionPackInstalled,
            isCompanionOperationInProgress: model.isVisionCompanionOperationInProgress,
            installState: model.visionInstallState)
    }

    private var visionSection: some View {
        Section("Image Support") {
            LabeledContent("State") {
                Text(visionStatusLabel)
                    .font(.caption)
                    .foregroundStyle(visionStatusColor)
            }
            if model.isVisionRuntimeSupported && !model.isVisionPackInstalled,
               let descriptor = model.selectedVisionInstallDescriptor {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(
                        descriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if let fraction = model.visionInstallProgressFraction {
                ProgressView(value: fraction)
                    .accessibilityValue(visionAccessibleProgress(fraction: fraction))
                HStack(alignment: .firstTextBaseline) {
                    Text(MetricFormat.percent(fraction * 100))
                    Spacer(minLength: 8)
                    if let eta = model.visionInstallETAText {
                        Text(eta)
                    }
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            } else if model.isInstallingVisionPack {
                ProgressView()
                    .controlSize(.small)
                if let eta = model.visionInstallETAText {
                    Text(eta)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if !model.isVisionRuntimeSupported {
                Text("Image support requires an M2 or newer Mac. "
                    + "Text generation remains available on this Mac.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if case .failed(let message) = model.visionInstallState {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            } else if case .recoverable(let message) = model.visionInstallState {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .partial(let message) = model.visionInstallationStatus,
                      !model.isInstallingVisionPack {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .unsupportedLayout = model.visionInstallationStatus {
                Text("Image support needs a model directory named "
                    + "“<name>.gturbo”, which is where the companion pack lives.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if case .verificationUnavailable(let message) =
                        model.visionInstallationStatus {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .failed(let message) = model.visionInstallReadiness,
                      !model.isInstallingVisionPack {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .insufficientSpace(let requirement) = model.visionInstallReadiness,
                      !model.isInstallingVisionPack {
                Text("Free \(MetricFormat.storage(requirement.shortfallBytes)) more storage.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if model.isVisionCompanionOperationInProgress {
                Text("Model actions stay unavailable until this finishes. "
                    + "Your prompt, images, and transcript are kept.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if !model.isVisionPackInstalled && model.loadState.isReady {
                Text("Unload the model before preparing image support.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if model.isVisionPackInstalled && model.loadState.isReady {
                Text("Unload the model before removing image support.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if model.selectedModelID == .qwen3_6 {
                Text("Qwen image support is prepared with the selected local source during model conversion.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                visionActions
            }
        }
    }

    private var visionActions: some View {
        HStack {
                if model.isInstallingVisionPack {
                    Button("Cancel", action: model.cancelVisionInstall)
                        .disabled(!model.canCancelVisionInstall)
                } else if case .readyToActivate = model.visionInstallState {
                    Button("Discard", role: .destructive) {
                        model.discardVisionPackDownload()
                    }
                    .disabled(!model.canDiscardVisionPackDownload)
                    if model.isVisionRuntimeSupported {
                        Button("Activate", action: model.activateVisionPack)
                            .buttonStyle(.borderedProminent)
                            .disabled(!model.canActivateVisionPack)
                    }
                } else if model.isVisionPackInstalled {
                    Button("Remove", role: .destructive) {
                        model.requestVisionPackRemoval()
                    }
                    .disabled(!model.canRemoveVisionPack)
                } else {
                    if model.hasVisionPackDirectory {
                        Button("Remove", role: .destructive) {
                            model.requestVisionPackRemoval()
                        }
                        .disabled(!model.canRemoveVisionPack)
                    }
                    if model.hasPartialVisionPackDownload {
                        Button("Discard", role: .destructive) {
                            model.discardVisionPackDownload()
                        }
                        .disabled(!model.canDiscardVisionPackDownload)
                    }
                    if model.isVisionRuntimeSupported {
                        Button(visionInstallButtonLabel) {
                            model.installVisionPack()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canInstallVisionPack)
                    }
                }
            }
    }

    private var visionInstallButtonLabel: String {
        if model.hasPartialVisionPackDownload { return "Resume" }
        if model.hasVisionPackDirectory { return "Repair" }
        return "Download"
    }

    private func visionAccessibleProgress(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = model.visionInstallETAText else { return percent }
        return "\(percent), \(eta)"
    }

    private var visionStatusLabel: String {
        guard model.isVisionRuntimeSupported else { return "Requires M2 or newer" }
        if model.visionInstallState != .idle {
            return model.visionInstallPhaseLabel
        }
        switch model.visionInstallationStatus {
        case .missing: return "Not installed"
        case .partial: return "Needs repair"
        case .complete: return "Installed"
        case .unsupportedLayout: return "Not available for this model"
        case .verificationUnavailable(let message): return message
        }
    }

    private var visionStatusColor: Color {
        guard model.isVisionRuntimeSupported else { return .secondary }
        switch model.visionInstallationStatus {
        case .partial: return .orange
        case .verificationUnavailable: return .orange
        case .missing, .complete, .unsupportedLayout: return .secondary
        }
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Selected") {
                Text(modelIdentity.selectedName)
                    .font(.caption)
                    .multilineTextAlignment(.trailing)
            }
            LabeledContent("Selection state") {
                Text(modelIdentity.selectedState)
                    .font(.caption)
                    .foregroundStyle(modelIdentityStateColor)
                    .multilineTextAlignment(.trailing)
            }
            if let loadedName = modelIdentity.loadedName {
                LabeledContent("Loaded identity") {
                    Text(loadedName)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .multilineTextAlignment(.trailing)
                }
                if let detail = modelIdentity.loadedDetail {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            } else {
                LabeledContent("Loaded identity") {
                    Text("None")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            LabeledContent("Text") {
                Text(textInstallationStatusLabel)
                    .font(.caption)
                    .foregroundStyle(textInstallationStatusColor)
            }
            LabeledContent("Image support") {
                Text(visionStatusLabel)
                    .font(.caption)
                    .foregroundStyle(visionStatusColor)
                    .multilineTextAlignment(.trailing)
            }
            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .font(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
            }
            if model.requiresModelInstallation,
               let descriptor = model.selectedTextInstallDescriptor {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(descriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(descriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel
            || model.isVisionCompanionOperationInProgress)
    }

    private var modelIdentity: AppModelIdentityPresentation {
        .resolve(
            selected: model.selectedModelEntry,
            installationStatus: model.installationStatus,
            loadState: model.loadState,
            transition: model.modelSelectionTransition,
            readiness: model.loadedModelReadiness)
    }

    private var modelIdentityStateColor: Color {
        if case .failed = model.modelSelectionTransition { return .red }
        if model.loadState.isFailed { return .red }
        if model.loadState.isReady, modelIdentity.loadedName == nil { return .orange }
        switch model.installationStatus {
        case .partial: return .orange
        case .missing, .complete: return .secondary
        }
    }

    private var textInstallationStatusLabel: String {
        switch model.installationStatus {
        case .missing: return "Not installed"
        case .partial: return "Needs repair"
        case .complete: return "Installed and verified"
        }
    }

    private var textInstallationStatusColor: Color {
        if case .partial = model.installationStatus { return .orange }
        return .secondary
    }

    private var qwenSourceSection: some View {
        Section("Local Qwen Source") {
            Toggle("Include image support", isOn: Binding {
                qwenSourceIncludesVision
            } set: { includesVision in
                qwenSourceIncludesVision = includesVision
                guard let source = model.qwenSourceDirectory,
                      model.canSelectModel else { return }
                model.configureQwenSourceDirectory(
                    source,
                    includesVision: includesVision)
                qwenSourceIncludesVision = model.qwenSourceIncludesVision
            })
            .toggleStyle(.switch)
            .disabled(!model.canSelectModel)

            Button("Choose Qwen Source Folder…") {
                showingQwenSourceImporter = true
            }
            .disabled(!model.canSelectModel)
            .keyboardShortcut("q", modifiers: [.command, .option])
            .accessibilityIdentifier("qwen-source-folder-chooser")
            .accessibilityHint("Choose the pinned official Qwen source folder for local verification and conversion")

            qwenSourceStatus

            if let source = model.qwenSourceDirectory {
                HStack(spacing: 6) {
                    Text(source.path)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                        .help(source.path)
                    Spacer(minLength: 6)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(source.path, forType: .string)
                    } label: {
                        Label("Copy Qwen source path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                }
                Button("Clear Source", role: .destructive) {
                    model.clearQwenSourceDirectory()
                }
                .disabled(!model.canSelectModel)
            }
        }
    }

    @ViewBuilder
    private var qwenSourceStatus: some View {
        if let qwenSourcePickerError {
            Label(qwenSourcePickerError, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
        }
        switch model.qwenSourceConfigurationState {
        case .notConfigured:
            Text("Choose an existing official source folder. Conversion starts only when you use the Convert button.")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .checking:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Verifying source files and available space")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        case .ready:
            Label("Source verified. Qwen is available in the model picker.",
                  systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
        }
    }

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Conversation state") {
                Text(runtimeByteText(model.conversationLogicalStateBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Expert cache") {
                Text(runtimeByteText(model.expertCacheBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.menuLabel).tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            Text("More slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 8K context and 16 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

    private func runtimeByteText(_ bytes: UInt64?) -> String {
        guard let bytes else { return "Unavailable" }
        return MetricFormat.storage(bytes)
    }

    private var generationSection: some View {
        Section("Generation") {
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

}
