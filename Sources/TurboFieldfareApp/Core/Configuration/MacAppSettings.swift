import Darwin
import Foundation
import TurboFieldfare

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 6

    var version: Int = currentVersion
    var contextTokens: Int = AppContextLengthOption.eightK.tokens
    var expertCacheSlots: Int = 16
    var temperature: Double = 0.2
    var topKEnabled: Bool = true
    var topK: Int = 64
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var newlineShortcut: AppNewlineShortcut = .return
    var showPromptExamples: Bool = true
    var visionResidencyPolicy: VisionResidencyPolicy = .onDemand
    var rdadvisePolicy: AppRDAdvicePolicy = .off
    var loadModelOnLaunch: Bool = false
    var agentModeEnabled: Bool = false
    /// The original settings key remains the Gemma choice so old files and old
    /// call sites retain exactly the same default and encoding.
    var toolThinkingEnabled: Bool = GFTokenizer.toolThinkingEnabled
    var selectedModelID: AppModelID = AppModelCatalog.defaultID
    var qwenToolThinkingEnabled: Bool = true
    var qwenSourceRoot: String? = nil
    var qwenRegistrationPath: String? = nil

    var gemmaToolThinkingEnabled: Bool {
        get { toolThinkingEnabled }
        set { toolThinkingEnabled = newValue }
    }

    private enum CodingKeys: String, CodingKey {
        case version
        case contextTokens
        case expertCacheSlots
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case newlineShortcut
        case showPromptExamples
        case visionResidencyPolicy
        case rdadvisePolicy
        case loadModelOnLaunch
        case agentModeEnabled
        case toolThinkingEnabled
        case selectedModelID
        case qwenToolThinkingEnabled
        case qwenSourceRoot
        case qwenRegistrationPath
    }

    init(version: Int = currentVersion,
         contextTokens: Int = AppContextLengthOption.eightK.tokens,
         expertCacheSlots: Int = 16,
         temperature: Double = 0.2,
         topKEnabled: Bool = true,
         topK: Int = 64,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         newlineShortcut: AppNewlineShortcut = .return,
         showPromptExamples: Bool = true,
         visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
         rdadvisePolicy: AppRDAdvicePolicy = .off,
         loadModelOnLaunch: Bool = false,
         agentModeEnabled: Bool = false,
         toolThinkingEnabled: Bool = GFTokenizer.toolThinkingEnabled,
         selectedModelID: AppModelID = AppModelCatalog.defaultID,
         qwenToolThinkingEnabled: Bool = true,
         qwenSourceRoot: String? = nil,
         qwenRegistrationPath: String? = nil) {
        self.version = version
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.newlineShortcut = newlineShortcut
        self.showPromptExamples = showPromptExamples
        self.visionResidencyPolicy = visionResidencyPolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.loadModelOnLaunch = loadModelOnLaunch
        self.agentModeEnabled = agentModeEnabled
        self.toolThinkingEnabled = toolThinkingEnabled
        self.selectedModelID = selectedModelID
        self.qwenToolThinkingEnabled = qwenToolThinkingEnabled
        self.qwenSourceRoot = qwenSourceRoot
        self.qwenRegistrationPath = qwenRegistrationPath
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        newlineShortcut = try container.decodeIfPresent(
            AppNewlineShortcut.self,
            forKey: .newlineShortcut) ?? .return
        showPromptExamples = try container.decodeIfPresent(
            Bool.self,
            forKey: .showPromptExamples) ?? true
        visionResidencyPolicy = try container.decodeIfPresent(
            VisionResidencyPolicy.self,
            forKey: .visionResidencyPolicy) ?? .onDemand
        rdadvisePolicy = try container.decodeIfPresent(
            AppRDAdvicePolicy.self,
            forKey: .rdadvisePolicy) ?? .off
        loadModelOnLaunch = try container.decodeIfPresent(
            Bool.self,
            forKey: .loadModelOnLaunch) ?? false
        agentModeEnabled = try container.decodeIfPresent(
            Bool.self,
            forKey: .agentModeEnabled) ?? false
        toolThinkingEnabled = try container.decodeIfPresent(
            Bool.self,
            forKey: .toolThinkingEnabled) ?? GFTokenizer.toolThinkingEnabled
        let selectedRaw = try container.decodeIfPresent(
            String.self,
            forKey: .selectedModelID)
        selectedModelID = selectedRaw.flatMap(AppModelID.init(rawValue:))
            ?? AppModelCatalog.defaultID
        qwenToolThinkingEnabled = try container.decodeIfPresent(
            Bool.self,
            forKey: .qwenToolThinkingEnabled) ?? true
        qwenSourceRoot = try container.decodeIfPresent(String.self, forKey: .qwenSourceRoot)
        qwenRegistrationPath = try container.decodeIfPresent(String.self, forKey: .qwenRegistrationPath)
    }

    func toolThinkingEnabled(for modelID: AppModelID) -> Bool {
        switch modelID {
        case .gemma4: toolThinkingEnabled
        case .qwen3_6: qwenToolThinkingEnabled
        }
    }

    mutating func setToolThinkingEnabled(_ enabled: Bool, for modelID: AppModelID) {
        switch modelID {
        case .gemma4: toolThinkingEnabled = enabled
        case .qwen3_6: qwenToolThinkingEnabled = enabled
        }
    }

    func isValid() -> Bool {
        AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
            && Self.validDirectoryPath(qwenSourceRoot)
            && Self.validDirectoryPath(qwenRegistrationPath)
    }

    private static func validDirectoryPath(_ path: String?) -> Bool {
        guard let path else { return true }
        return path.hasPrefix("/") && !path.contains("\0")
    }
}

/// Just enough of the file to route on, so a newer build's schema cannot throw
/// before its version has been read.
private struct VersionStamp: Decodable {
    let version: Int
}

/// Validate an explicit settings location before the app creates its model.
public enum AppSettingsLaunchConfiguration {
    public static func validate(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        if let fileURL = try MacAppSettingsFileStore.explicitFileURL(environment: environment) {
            _ = try MacAppSettingsFileStore.loadOrCreate(at: fileURL)
        }
    }
}

enum MacAppSettingsFileStore {
    static let settingsPathEnvironmentKey = "TURBOFIELDFARE_SETTINGS_PATH"

    // The chosen settings file stays fixed if the model path changes later.
    private static let launchOverride: Result<URL?, Error> = Result {
        try explicitFileURL(environment: ProcessInfo.processInfo.environment)
    }

    static func explicitFileURL(environment: [String: String]) throws -> URL? {
        guard let path = environment[settingsPathEnvironmentKey] else { return nil }
        guard path.hasPrefix("/"), !path.contains("\0"), !path.hasSuffix("/") else {
            throw SettingsOverrideError.invalidPath
        }
        let fileURL = URL(fileURLWithPath: path, isDirectory: false).standardizedFileURL
        try validateExplicitFileURL(fileURL)
        return fileURL
    }

    static func fileURL(forModelDirectory modelDirectory: URL,
                        environment: [String: String]) throws -> URL {
        try explicitFileURL(environment: environment)
            ?? defaultFileURL(forModelDirectory: modelDirectory)
    }

    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        requireLaunchOverride() ?? defaultFileURL(forModelDirectory: modelDirectory)
    }

    private static func defaultFileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(MacAppSettings.fileName, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> MacAppSettings {
        if let override = requireLaunchOverride() {
            do {
                return try loadOrCreate(at: override, fileManager: fileManager)
            } catch {
                stopForInvalidOverride(error)
            }
        }
        let fileURL = defaultFileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                // Read the version before the full schema. A newer build may
                // have renamed required keys, and trying the full decode first
                // would reach the catch below and delete that newer file.
                let stamp = try JSONDecoder().decode(VersionStamp.self, from: data)
                // This build cannot interpret later versions. Use defaults in
                // memory while leaving the newer owner's file untouched.
                guard stamp.version <= MacAppSettings.currentVersion else {
                    return MacAppSettings()
                }
                var settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
                let needsMigration = settings.version < MacAppSettings.currentVersion
                if needsMigration {
                    // Existing values stay deliberate choices; only the schema
                    // version and newly defaulted fields are added on rewrite.
                    settings.version = MacAppSettings.currentVersion
                }
                guard settings.isValid() else { throw InvalidSettings() }
                if needsMigration {
                    try? save(settings, forModelDirectory: modelDirectory,
                              fileManager: fileManager)
                }
                return settings
            } catch {
                try? fileManager.removeItem(at: fileURL)
            }
        }

        let settings = MacAppSettings()
        try? save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
        return settings
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        if let override = requireLaunchOverride() {
            do {
                try save(settings, at: override, fileManager: fileManager)
                return
            } catch {
                stopForInvalidOverride(error)
            }
        }
        try write(settings, at: defaultFileURL(forModelDirectory: modelDirectory),
                  fileManager: fileManager)
    }

    /// Explicit files are never deleted or replaced with defaults on read failure.
    static func loadOrCreate(at fileURL: URL,
                             fileManager: FileManager = .default) throws -> MacAppSettings {
        try validateExplicitFileURL(fileURL, fileManager: fileManager)
        guard fileManager.fileExists(atPath: fileURL.path) else {
            let settings = MacAppSettings()
            try save(settings, at: fileURL, fileManager: fileManager)
            return settings
        }
        let data = try Data(contentsOf: fileURL)
        let stamp = try JSONDecoder().decode(VersionStamp.self, from: data)
        guard stamp.version <= MacAppSettings.currentVersion else {
            throw SettingsOverrideError.newerVersion(stamp.version)
        }
        var settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
        guard settings.isValid() else { throw InvalidSettings() }
        if settings.version < MacAppSettings.currentVersion {
            settings.version = MacAppSettings.currentVersion
            try save(settings, at: fileURL, fileManager: fileManager)
        }
        return settings
    }

    static func save(_ settings: MacAppSettings,
                     at fileURL: URL,
                     fileManager: FileManager = .default) throws {
        try validateExplicitFileURL(fileURL, fileManager: fileManager)
        try write(settings, at: fileURL, fileManager: fileManager)
    }

    private static func write(_ settings: MacAppSettings,
                              at fileURL: URL,
                              fileManager: FileManager) throws {
        guard settings.isValid() else { throw InvalidSettings() }
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private static func validateExplicitFileURL(
        _ fileURL: URL, fileManager: FileManager = .default
    ) throws {
        guard fileURL.isFileURL, fileURL.path.hasPrefix("/"),
              !fileURL.path.contains("\0"), fileURL.path != "/" else {
            throw SettingsOverrideError.invalidPath
        }
        // Also reject dangling links, which fileExists reports as absent.
        if (try? fileManager.destinationOfSymbolicLink(atPath: fileURL.path)) != nil {
            throw SettingsOverrideError.invalidFileType
        }
        if fileManager.fileExists(atPath: fileURL.path) {
            let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular else {
                throw SettingsOverrideError.invalidFileType
            }
        }
    }

    private static func requireLaunchOverride() -> URL? {
        do { return try launchOverride.get() }
        catch { stopForInvalidOverride(error) }
    }

    private static func stopForInvalidOverride(_ error: Error) -> Never {
        FileHandle.standardError.write(Data(
            "Invalid \(settingsPathEnvironmentKey): \(error.localizedDescription)\n".utf8))
        exit(EXIT_FAILURE)
    }

    private enum SettingsOverrideError: LocalizedError {
        case invalidPath
        case invalidFileType
        case newerVersion(Int)

        var errorDescription: String? {
            switch self {
            case .invalidPath:
                "The settings path must be an absolute file path without a trailing slash or a NUL character."
            case .invalidFileType:
                "The settings path must name a regular file, not a directory or symbolic link."
            case .newerVersion(let version):
                "Settings version \(version) is newer than this app supports."
            }
        }
    }

    private struct InvalidSettings: LocalizedError {
        var errorDescription: String? { "The settings file contains invalid values." }
    }
}
