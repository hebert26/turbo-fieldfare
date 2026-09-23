import Foundation

public enum AppModelFamily: String, Codable, Equatable, Sendable {
    case gemma4
    case qwen3_6
}

public struct AppModelCatalogEntry: Equatable, Sendable, Identifiable {
    public let id: AppModelID
    public let displayName: String
    public let family: AppModelFamily
    public let sourceIdentity: AppModelSourceIdentity
    public let location: AppModelLocation.Resolved
    public let installRoute: AppModelInstallRoute

    init(
        id: AppModelID,
        displayName: String,
        family: AppModelFamily,
        sourceIdentity: AppModelSourceIdentity,
        location: AppModelLocation.Resolved,
        installRoute: AppModelInstallRoute
    ) {
        self.id = id
        self.displayName = displayName
        self.family = family
        self.sourceIdentity = sourceIdentity
        self.location = location
        self.installRoute = installRoute
    }

    /// True only when the app can begin installation without first receiving a
    /// local source. Qwen deliberately returns false until a verified local
    /// source is supplied to the P21 conversion client.
    public var isInstallable: Bool {
        installRoute.isInstallableWithoutLocalSource
    }

    public func accepts(
        family actualFamily: AppModelFamily,
        modelID: String,
        revision: String,
        sourceIndexSHA256: String
    ) -> Bool {
        family == actualFamily
            && sourceIdentity.repoID == modelID
            && sourceIdentity.revision == revision
            && sourceIdentity.sourceIndexSHA256.lowercased()
                == sourceIndexSHA256.lowercased()
    }
}

public enum AppModelCatalog {
    public static let defaultID: AppModelID = .gemma4

    public static var all: [AppModelCatalogEntry] {
        AppModelID.allCases.map { entry(for: $0) }
    }

    public static func entry(for id: AppModelID) -> AppModelCatalogEntry {
        entry(for: id, location: AppModelLocation.resolved(modelID: id))
    }

    static func entry(
        for id: AppModelID,
        location: AppModelLocation.Resolved
    ) -> AppModelCatalogEntry {
        switch id {
        case .gemma4:
            let text = AppModelInstallDescriptor.default
            return AppModelCatalogEntry(
                id: .gemma4,
                displayName: text.displayName,
                family: .gemma4,
                sourceIdentity: text.sourceIdentity,
                location: location,
                installRoute: .remoteRepack(
                    text: text,
                    vision: .visionCompanion))
        case .qwen3_6:
            let source = AppModelSourceIdentity(
                repoID: "Qwen/Qwen3.6-35B-A3B",
                revision: "995ad96eacd98c81ed38be0c5b274b04031597b0",
                sourceIndexSHA256:
                    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83")
            return AppModelCatalogEntry(
                id: .qwen3_6,
                displayName: "Qwen3.6 35B-A3B",
                family: .qwen3_6,
                sourceIdentity: source,
                location: location,
                installRoute: .localQwenConversion(AppLocalQwenInstallDescriptor(
                    displayName: "Qwen3.6 35B-A3B",
                    sourceIdentity: source)))
        }
    }
}
