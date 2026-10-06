import Darwin
import Foundation

public enum AppModelID: String, CaseIterable, Codable, Equatable, Hashable, Sendable {
    case gemma4 = "gemma4-26b-a4b-it"
    case qwen3_6 = "qwen3.6-35b-a3b"
}

public enum AppModelLocation {
    public struct Resolved: Equatable, Sendable {
        public let textModelURL: URL
        public let visionModelURL: URL

        public init(textModelURL: URL, visionModelURL: URL,
                    preservePhysicalPath: Bool = false) {
            self.textModelURL = preservePhysicalPath ? textModelURL : textModelURL.standardizedFileURL
            self.visionModelURL = preservePhysicalPath ? visionModelURL : visionModelURL.standardizedFileURL
        }
    }

    /// Preserves the existing Gemma lookup and fallback behavior.
    public static func defaultURL() -> URL {
        defaultResolved().textModelURL
    }

    public static func defaultResolved() -> Resolved {
        resolve(modelID: .gemma4)
    }

    public static func resolved(modelID: AppModelID) -> Resolved {
        resolve(modelID: modelID)
    }

    public static func qwenRegistrationResolved(logicalURL: URL) -> Resolved {
        let text = physicalParentKeepingLeaf(logicalURL)
        return Resolved(
            textModelURL: text,
            visionModelURL: adjacentVisionURL(
                forTextModel: text, preservePhysicalPath: true),
            preservePhysicalPath: true)
    }

    /// Logical Qwen registration destination. Canonicalize an existing checkout
    /// parent, but retain the final leaf so registration rejects its symlink.
    /// The physical source root is stored separately in official-source.json.
    public static func qwenRegistrationDestination(explicitURL: URL? = nil) -> URL {
        qwenRegistrationDestination(
            explicitURL: explicitURL,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true),
            applicationSupportURL: nil,
            fileExists: FileManager.default.fileExists(atPath:))
    }

    static func qwenRegistrationDestination(
        explicitURL: URL?, executableURL: URL?, currentDirectoryURL: URL,
        applicationSupportURL: URL?, fileExists: (String) -> Bool
    ) -> URL {
        let manager = FileManager.default
        let support = applicationSupportURL ?? ((try? manager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil, create: false)) ?? manager.homeDirectoryForCurrentUser)
        return resolveTextModel(
            modelID: .qwen3_6, explicitURL: explicitURL,
            executableURL: executableURL, currentDirectoryURL: currentDirectoryURL,
            applicationSupportURL: support, fileExists: fileExists,
            resolveCheckoutSymlinks: false)
    }

    static func resolve(
        modelID: AppModelID,
        explicitURL: URL? = nil,
        executableURL: URL? = Bundle.main.executableURL,
        currentDirectoryURL: URL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true),
        applicationSupportURL: URL? = nil,
        fileExists: (String) -> Bool = {
            FileManager.default.fileExists(atPath: $0)
        }
    ) -> Resolved {
        let fileManager = FileManager.default
        let support = applicationSupportURL ?? ((try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser)
        let text = resolveTextModel(
            modelID: modelID,
            explicitURL: explicitURL,
            executableURL: executableURL,
            currentDirectoryURL: currentDirectoryURL,
            applicationSupportURL: support,
            fileExists: fileExists,
            resolveCheckoutSymlinks: modelID != .qwen3_6)
        return Resolved(
            textModelURL: text,
            visionModelURL: adjacentVisionURL(
                forTextModel: text, preservePhysicalPath: modelID == .qwen3_6),
            preservePhysicalPath: modelID == .qwen3_6)
    }

    /// Source-compatible Gemma-only seam retained for existing call sites and
    /// tests. Its behavior remains the same as before the catalog existed.
    static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool) -> URL {
        resolveTextModel(
            modelID: .gemma4,
            explicitURL: explicitURL,
            executableURL: executableURL,
            currentDirectoryURL: currentDirectoryURL,
            applicationSupportURL: applicationSupportURL,
            fileExists: fileExists)
    }

    private static func resolveTextModel(
        modelID: AppModelID,
        explicitURL: URL?,
        executableURL: URL?,
        currentDirectoryURL: URL,
        applicationSupportURL: URL,
        fileExists: (String) -> Bool,
        resolveCheckoutSymlinks: Bool = true
    ) -> URL {
        if let explicitURL {
            let absolute = absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
            return resolveCheckoutSymlinks ? absolute : physicalParentKeepingLeaf(absolute)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return checkoutModelURL(packageRoot: root, modelID: modelID,
                                    resolveSymlinks: resolveCheckoutSymlinks)
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return checkoutModelURL(packageRoot: root, modelID: modelID,
                                    resolveSymlinks: resolveCheckoutSymlinks)
        }
        let fallback = applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .appendingPathComponent(fileName(for: modelID), isDirectory: true)
            .standardizedFileURL
        return resolveCheckoutSymlinks ? fallback : physicalParentKeepingLeaf(fallback)
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func checkoutModelURL(
        packageRoot: URL, modelID: AppModelID, resolveSymlinks: Bool
    ) -> URL {
        // Existing lookups resolve the whole path for receipt-bound installs.
        // Registration canonicalizes only the existing parent through POSIX:
        // Foundation may leave an ancestor such as /var as a symlink alias.
        // Never follow the final Qwen leaf; the registrar must reject its link.
        let parent = packageRoot.appendingPathComponent("scratch", isDirectory: true)
        let location = parent.appendingPathComponent(fileName(for: modelID), isDirectory: true)
        if resolveSymlinks { return location.resolvingSymlinksInPath().standardizedFileURL }
        guard let physicalParent = realpath(parent.path, nil) else {
            return location.standardizedFileURL
        }
        defer { free(physicalParent) }
        return URL(fileURLWithPath: String(cString: physicalParent), isDirectory: true)
            .appendingPathComponent(fileName(for: modelID), isDirectory: true)
    }

    private static func physicalParentKeepingLeaf(_ url: URL) -> URL {
        let parent = url.deletingLastPathComponent()
        guard let physical = realpath(parent.path, nil) else { return url }
        defer { free(physical) }
        return URL(fileURLWithPath: String(cString: physical), isDirectory: true)
            .appendingPathComponent(url.lastPathComponent, isDirectory: true)
    }

    private static func fileName(for modelID: AppModelID) -> String {
        switch modelID {
        case .gemma4: "gemma4.gturbo"
        case .qwen3_6: "qwen3.6-35b-a3b.gturbo"
        }
    }

    private static func adjacentVisionURL(
        forTextModel textModelURL: URL, preservePhysicalPath: Bool = false
    ) -> URL {
        let name = textModelURL.lastPathComponent
        let suffix = ".gturbo"
        precondition(name.hasSuffix(suffix) && name.count > suffix.count)
        let adjacent = textModelURL.deletingLastPathComponent()
            .appendingPathComponent(
                "\(name.dropLast(suffix.count)).vision.gturbo",
                isDirectory: true)
        return preservePhysicalPath ? adjacent : adjacent.standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/TurboFieldfareApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
