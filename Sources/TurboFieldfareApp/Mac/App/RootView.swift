import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0

    var body: some View {
        HStack(spacing: 0) {
            primaryContent
                .frame(minWidth: 720, maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            InspectorView(model: model)
                .frame(width: 320)
                .frame(maxHeight: .infinity)
                .background(Color(nsColor: .windowBackgroundColor))
        }
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: TurboFieldfareMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(TurboFieldfareMacTheme.accentColor)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .animation(.smooth(duration: 0.2), value: model.presentation.conversationAction)
        .transaction { transaction in
            if model.isRunning {
                transaction.animation = nil
            }
        }
    }

    private var primaryContent: some View {
        Group {
            if model.requiresModelInstallation {
                ModelInstallView(model: model)
            } else {
                conversationView
            }
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(spacing: 0) {
                StatusHUDView(model: model)
                modelSelectionBar
            }
        }
    }

    private var modelSelectionBar: some View {
        let presentation = AppModelIdentityPresentation.resolve(
            selected: model.selectedModelEntry,
            installationStatus: model.installationStatus,
            loadState: model.loadState,
            transition: model.modelSelectionTransition,
            readiness: model.loadedModelReadiness)
        return HStack(spacing: 10) {
            Picker("Model", selection: modelSelectionBinding) {
                ForEach(model.modelCatalogEntries) { entry in
                    Text(entry.displayName)
                        .tag(entry.id)
                        .disabled(!model.isModelSelectable(entry.id))
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
            .disabled(!model.canSelectModel)
            .accessibilityIdentifier("model-picker")
            .accessibilityLabel("Selected model")
            .accessibilityValue(presentation.accessibilityValue)
            .accessibilityHint("Choose an installed model or one ready to install")

            Text(presentation.selectedState)
                .font(.caption)
                .foregroundStyle(selectionStateColor)
                .lineLimit(1)
                .accessibilityIdentifier("model-selection-state")

            Spacer(minLength: 8)

            if case .failed = model.modelSelectionTransition,
               model.selectedModelID != AppModelCatalog.defaultID {
                Button("Use Gemma") {
                    model.selectModel(AppModelCatalog.defaultID)
                }
                .buttonStyle(.borderless)
                .disabled(!model.canSelectModel)
                .accessibilityHint("Returns to the default Gemma model")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 7)
        .background(Color(nsColor: .controlBackgroundColor))
        .overlay(alignment: .bottom) { Divider() }
    }

    private var modelSelectionBinding: Binding<AppModelID> {
        Binding {
            model.selectedModelID
        } set: { modelID in
            model.selectModel(modelID)
        }
    }

    private var selectionStateColor: Color {
        if case .failed = model.modelSelectionTransition { return .red }
        if model.loadState.isFailed { return .red }
        if model.loadState.isReady, model.loadedModelReadiness == nil { return .orange }
        switch model.installationStatus {
        case .partial: return .orange
        case .missing, .complete: return .secondary
        }
    }

    private var conversationView: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                if model.hasOutputTranscript {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(
                            height: max(
                                0,
                                geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                }

                conversationChrome
                    .background {
                        GeometryReader { chromeGeometry in
                            Color.clear.preference(
                                key: ConversationChromeHeightKey.self,
                                value: chromeGeometry.size.height)
                        }
                    }
            }
            .onPreferenceChange(ConversationChromeHeightKey.self) { height in
                guard height > 0 else { return }
                var transaction = Transaction()
                transaction.disablesAnimations = true
                withTransaction(transaction) {
                    conversationChromeHeight = height
                }
            }
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ErrorBanner(model: model)
            if model.shouldShowPromptExamples {
                PromptExamplesView { preset in
                    model.promptText = preset.prompt
                }
            }
            ModelActionBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .animation(.smooth(duration: 0.2), value: model.promptText.isEmpty)
        .animation(.smooth(duration: 0.2), value: model.showPromptExamples)
    }

}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
