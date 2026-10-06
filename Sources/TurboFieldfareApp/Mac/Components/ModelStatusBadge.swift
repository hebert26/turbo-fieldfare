import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            statusDot
            Text(identity.selectedName)
                .font(.callout.weight(.semibold))
                .lineLimit(1)
                .help(identity.loadedDetail ?? identity.selectedState)
                .accessibilityLabel("Model status")
                .accessibilityValue(identity.accessibilityValue)
        }
    }

    private var identity: AppModelIdentityPresentation {
        .resolve(
            selected: model.selectedModelEntry,
            installationStatus: model.installationStatus,
            loadState: model.loadState,
            transition: model.modelSelectionTransition,
            readiness: model.loadedModelReadiness)
    }

    @ViewBuilder
    private var statusDot: some View {
        switch model.presentation.severity {
        case .neutral: dot(.gray)
        case .active, .warning: dot(.orange)
        case .success: dot(.green)
        case .error: dot(.red)
        }
    }

    private func dot(_ color: Color) -> some View {
        Circle().fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
    }
}
