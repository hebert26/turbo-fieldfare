import Foundation

public enum AppModelID: String, CaseIterable, Codable, Equatable, Hashable, Sendable {
    case gemma4 = "gemma4-26b-a4b-it"
    case qwen3_6 = "qwen3.6-35b-a3b"
}

public enum AppModelLocation {
    public struct Resolved: Equatable, Sendable {
        public let textModelURL: URL
        public let visionModelURL: URL

        public init(textModelURL: URL, visionModelURL: URL) {
            self.textModelURL = textModelURL.standardizedFileURL
            self.visionModelURL = visionModelURL.standardizedFileURL
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
            fileExists: fileExists)
        return Resolved(
            textModelURL: text,
            visionModelURL: adjacentVisionURL(forTextModel: text))
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
        fileExists: (String) -> Bool
    ) -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return checkoutModelURL(packageRoot: root, modelID: modelID)
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return checkoutModelURL(packageRoot: root, modelID: modelID)
        }
        return applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .appendingPathComponent(fileName(for: modelID), isDirectory: true)
            .standardizedFileURL
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func checkoutModelURL(packageRoot: URL, modelID: AppModelID) -> URL {
        // Development checkouts commonly link scratch to the canonical install
        // in Application Support. Use the real text-model location so the
        // adjacent vision companion is checked against its receipt-bound path,
        // rather than checking that same directory through a symlink alias.
        packageRoot.appendingPathComponent("scratch", isDirectory: true)
            .appendingPathComponent(fileName(for: modelID), isDirectory: true)
            .resolvingSymlinksInPath()
            .standardizedFileURL
    }

    private static func fileName(for modelID: AppModelID) -> String {
        switch modelID {
        case .gemma4: "gemma4.gturbo"
        case .qwen3_6: "qwen3.6-35b-a3b.gturbo"
        }
    }

    private static func adjacentVisionURL(forTextModel textModelURL: URL) -> URL {
        let name = textModelURL.lastPathComponent
        let suffix = ".gturbo"
        precondition(name.hasSuffix(suffix) && name.count > suffix.count)
        return textModelURL.deletingLastPathComponent()
            .appendingPathComponent(
                "\(name.dropLast(suffix.count)).vision.gturbo",
                isDirectory: true)
            .standardizedFileURL
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
