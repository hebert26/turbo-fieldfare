import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ModelInstallView: View {
    let model: AppModel
    @State private var showingDiscardConfirmation = false

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                identity
                storageCard
                progressArea
                actions
            }
            .frame(maxWidth: 560)
            .padding(.horizontal, 28)
            .padding(.vertical, 48)
            .frame(maxWidth: .infinity)
        }
        .confirmationDialog(
            discardConfirmationTitle,
            isPresented: $showingDiscardConfirmation,
            titleVisibility: .visible
        ) {
            Button(discardButtonLabel, role: .destructive) {
                model.discardModelDownload()
            }
            Button(keepButtonLabel, role: .cancel) {}
        } message: {
            Text(discardConfirmationMessage)
        }
    }

    private var identity: some View {
        VStack(spacing: 10) {
            Image(systemName: "externaldrive.badge.plus")
                .font(.system(.largeTitle, design: .rounded))
                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                .accessibilityHidden(true)
            Text("Model required")
                .font(.title.bold())
                .accessibilityHeading(.h1)
            Text("TurboFieldfare needs \(model.selectedModelEntry.displayName) before it can generate text.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if isLocalQwenInstall {
                Text(qwenSourceStatusText)
                    .font(.caption)
                    .foregroundStyle(qwenSourceStatusColor)
                    .multilineTextAlignment(.center)
            }
        }
    }

    private var storageCard: some View {
        VStack(spacing: 12) {
            if let requirement = model.installRequirement {
                StorageRow(label: "Space required",
                           value: MetricFormat.storage(requirement.requiredBytes))
                StorageRow(label: "Available on this Mac",
                           value: MetricFormat.storage(requirement.availableBytes))
                capacityStatus(requirement)
            } else if case .failed(let message) = model.installReadiness {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(message).foregroundStyle(.secondary)
                    Spacer()
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(readinessLabel)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
            Text(model.modelPathText)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(2)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(18)
        .background {
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    @ViewBuilder
    private func capacityStatus(_ requirement: AppModelInstallRequirement) -> some View {
        if requirement.canInstall {
            Label("Enough space to install", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Label("\(MetricFormat.storage(requirement.shortfallBytes)) more is required",
                  systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private var progressArea: some View {
        if model.isInstallingModel {
            VStack(alignment: .leading, spacing: 8) {
                if let fraction = model.installProgressFraction,
                   let downloaded = model.installDownloadedBytes,
                   let total = model.installTotalBytes {
                    ProgressView(value: fraction)
                        .accessibilityLabel(progressAccessibilityLabel)
                        .accessibilityValue(Text(accessibleProgressValue(fraction: fraction)))
                    HStack {
                        Text(progressAmountText(completed: downloaded, total: total))
                        Spacer()
                        Text(MetricFormat.percent(fraction * 100))
                    }
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    HStack(alignment: .firstTextBaseline, spacing: 16) {
                        if let reused = model.installReusedBytes, reused > 0 {
                            Text(reusedAmountText(reused))
                                .font(.caption)
                        }
                        Spacer(minLength: 16)
                        if let eta = model.installETAText {
                            Text(eta)
                                .font(.caption.monospacedDigit())
                        }
                    }
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                } else {
                    Text(model.presentation.label)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
        } else if case .cancelled = model.installState {
            Label(pausedLabel, systemImage: "pause.circle")
                .foregroundStyle(.secondary)
        } else if case .failed(let message) = model.installState {
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.red)
                .multilineTextAlignment(.center)
        } else if case .recoverable(let message) = model.installState {
            Label(message, systemImage: "exclamationmark.triangle")
                .font(.callout)
                .foregroundStyle(.orange)
                .multilineTextAlignment(.center)
        }
    }

    private var actions: some View {
        HStack(spacing: 12) {
            if model.isInstallingModel {
                Button("Cancel", action: model.cancelInstall)
                    .buttonStyle(.bordered)
                    .keyboardShortcut(.cancelAction)
                    .disabled(!model.canCancelInstall)
            } else {
                if model.hasPartialModelDownload {
                    Button(discardButtonLabel, role: .destructive) {
                        showingDiscardConfirmation = true
                    }
                    .buttonStyle(.bordered)
                    .disabled(!model.canDiscardModelDownload)
                }

                Button("Check Again", action: model.recheckModelAtCurrentLocation)
                .buttonStyle(.bordered)
                .disabled(model.isInstallingModel)

                Button(installButtonLabel,
                       action: model.installModel)
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canInstallModel)
            }
        }
        .controlSize(.large)
    }

    private var readinessLabel: String {
        switch model.installReadiness {
        case .checking:
            return isLocalQwenInstall
                ? "Verifying the local source and available space"
                : "Checking available space"
        case .failed(let message):
            return message
        case .ready, .insufficientSpace:
            return "Checking available space"
        }
    }

    private var isLocalQwenInstall: Bool {
        model.selectedModelID == .qwen3_6
    }

    private var qwenSourceStatusText: String {
        switch model.qwenSourceConfigurationState {
        case .notConfigured:
            return "Choose the official local Qwen source folder in the Model inspector."
        case .checking:
            return "Verifying the selected Qwen source folder."
        case .ready(let url):
            return "Verified local source: \(url.path)"
        case .failed(let message):
            return message
        }
    }

    private var qwenSourceStatusColor: Color {
        if case .failed = model.qwenSourceConfigurationState { return .red }
        return .secondary
    }

    private var progressAccessibilityLabel: String {
        isLocalQwenInstall ? "Model conversion" : "Model download"
    }

    private func progressAmountText(completed: UInt64, total: UInt64) -> String {
        let verb = isLocalQwenInstall ? "Converted" : "Downloaded"
        return "\(verb) \(MetricFormat.storage(completed)) of \(MetricFormat.storage(total))"
    }

    private func reusedAmountText(_ bytes: UInt64) -> String {
        let source = isLocalQwenInstall ? "saved conversion" : "saved download"
        return "Reused \(MetricFormat.storage(bytes)) from the \(source)"
    }

    private var pausedLabel: String {
        isLocalQwenInstall ? "Conversion paused" : "Download paused"
    }

    private var installButtonLabel: String {
        if isLocalQwenInstall {
            return model.hasPartialModelDownload ? "Resume Conversion" : "Convert"
        }
        return model.hasPartialModelDownload ? "Resume" : "Download"
    }

    private var discardButtonLabel: String {
        isLocalQwenInstall ? "Discard Conversion" : "Discard Download"
    }

    private var keepButtonLabel: String {
        isLocalQwenInstall ? "Keep Conversion" : "Keep Download"
    }

    private var discardConfirmationTitle: String {
        isLocalQwenInstall
            ? "Discard the saved model conversion?"
            : "Discard the saved model download?"
    }

    private var discardConfirmationMessage: String {
        if isLocalQwenInstall {
            return "Converted partial files and the conversion checkpoint will be removed. The verified installed model, if any, is preserved."
        }
        return "Downloaded ranges will be removed. The installed model, if any, is preserved."
    }

    private func accessibleProgressValue(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = model.installETAText else { return percent }
        return "\(percent), \(eta)"
    }
}

private struct StorageRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
        }
    }
}
