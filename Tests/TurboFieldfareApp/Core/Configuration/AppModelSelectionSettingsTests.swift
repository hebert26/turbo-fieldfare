import Foundation
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfare

@Suite("App model selection settings")
struct AppModelSelectionSettingsTests {
    @Test func LegacySettingsSelectGemmaAndKeepItsOriginalDefaults() throws {
        let data = Data("""
        {
          "version": 1,
          "contextTokens": 8192,
          "expertCacheSlots": 24,
          "temperature": 0.4,
          "topKEnabled": false,
          "topK": 32,
          "topPEnabled": false,
          "topP": 0.8,
          "prefillEnabled": false
        }
        """.utf8)

        let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)

        #expect(settings.selectedModelID == .gemma4)
        #expect(settings.toolThinkingEnabled(for: .gemma4)
            == GFTokenizer.toolThinkingEnabled)
        #expect(settings.toolThinkingEnabled(for: .qwen3_6))
        #expect(settings.contextTokens == 8_192)
        #expect(settings.contextTokens(for: .gemma4) == 8_192)
        #expect(settings.contextTokens(for: .qwen3_6) == 8_192)
        #expect(settings.expertCacheSlots == 24)
        #expect(settings.temperature == 0.4)
        #expect(!settings.topKEnabled)
        #expect(settings.topK == 32)
        #expect(!settings.topPEnabled)
        #expect(settings.topP == 0.8)
        #expect(!settings.prefillEnabled)
    }

    @Test func LegacyQwenSelectionFallsBackToTheSharedContextValue() throws {
        let data = Data("""
        {
          "version": 1,
          "contextTokens": 32768,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "selectedModelID": "qwen3.6-35b-a3b"
        }
        """.utf8)
        let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)

        #expect(settings.selectedModelID == .qwen3_6)
        #expect(settings.contextTokens(for: .qwen3_6) == 32768)
        #expect(settings.contextTokens(for: .gemma4) == 32768)
    }

    @Test func UnknownSelectionFallsBackToGemmaWithoutChangingTheFile() throws {
        let root = try Self.makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let original = try Self.encodedSettings(
            MacAppSettings(
                temperature: 0.37,
                topKEnabled: false,
                topK: 31,
                topPEnabled: true,
                topP: 0.87,
                selectedModelID: .qwen3_6,
                qwenToolThinkingEnabled: false),
            replacingSelectionWith: "future-family")
        try original.write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: model)

        #expect(settings.selectedModelID == .gemma4)
        #expect(settings.temperature == 0.37)
        #expect(settings.topK == 31)
        #expect(!settings.qwenToolThinkingEnabled)
        #expect(FileManager.default.fileExists(atPath: fileURL.path))
        #expect(try Data(contentsOf: fileURL) == original)
    }

    @Test func GemmaAndQwenThinkingChoicesAreStoredIndependently() throws {
        var settings = MacAppSettings(
            toolThinkingEnabled: false,
            selectedModelID: .qwen3_6,
            qwenToolThinkingEnabled: true)

        #expect(!settings.toolThinkingEnabled(for: .gemma4))
        #expect(settings.toolThinkingEnabled(for: .qwen3_6))
        settings.setToolThinkingEnabled(false, for: .qwen3_6)
        #expect(!settings.toolThinkingEnabled(for: .qwen3_6))
        #expect(!settings.toolThinkingEnabled(for: .gemma4))

        let roundTrip = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(settings))
        #expect(roundTrip.selectedModelID == .qwen3_6)
        #expect(!roundTrip.toolThinkingEnabled(for: .gemma4))
        #expect(!roundTrip.toolThinkingEnabled(for: .qwen3_6))
    }

    @Test func SelectedQwenPreservesGemmaSamplingValuesAcrossPersistence() throws {
        let initial = MacAppSettings(
            contextTokens: 16_384,
            expertCacheSlots: 24,
            temperature: 0.41,
            topKEnabled: false,
            topK: 23,
            topPEnabled: true,
            topP: 0.71,
            prefillEnabled: false,
            selectedModelID: .qwen3_6,
            qwenToolThinkingEnabled: false)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
        #expect(decoded.temperature == 0.41)
        #expect(decoded.topK == 23)
        #expect(decoded.topP == 0.71)
        #expect(!decoded.prefillEnabled)
        #expect(decoded.selectedModelID == .qwen3_6)
    }

    @Test func QwenContextRoundTripsIndependentlyFromGemmaContext() throws {
        var settings = MacAppSettings(
            contextTokens: AppContextLengthOption.eightK.tokens,
            selectedModelID: .qwen3_6)
        settings.setContextTokens(AppContextLengthOption.sixtyFourK.tokens, for: .qwen3_6)
        settings.setContextTokens(AppContextLengthOption.sixteenK.tokens, for: .gemma4)

        #expect(settings.contextTokens == AppContextLengthOption.sixteenK.tokens)
        #expect(settings.contextTokens(for: .gemma4) == AppContextLengthOption.sixteenK.tokens)
        #expect(settings.contextTokens(for: .qwen3_6) == AppContextLengthOption.sixtyFourK.tokens)

        let roundTrip = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))
        #expect(roundTrip.contextTokens(for: .gemma4) == AppContextLengthOption.sixteenK.tokens)
        #expect(roundTrip.contextTokens(for: .qwen3_6) == AppContextLengthOption.sixtyFourK.tokens)
    }

    @Test func GemmaFirstChangePreservesInheritedQwenChoiceAcrossRoundTrip() throws {
        var settings = MacAppSettings(contextTokens: AppContextLengthOption.thirtyTwoK.tokens)
        settings.setContextTokens(AppContextLengthOption.sixteenK.tokens, for: .gemma4)

        #expect(settings.contextTokens(for: .gemma4) == AppContextLengthOption.sixteenK.tokens)
        #expect(settings.contextTokens(for: .qwen3_6) == AppContextLengthOption.thirtyTwoK.tokens)

        let roundTrip = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))
        #expect(roundTrip.contextTokens(for: .gemma4) == AppContextLengthOption.sixteenK.tokens)
        #expect(roundTrip.contextTokens(for: .qwen3_6) == AppContextLengthOption.thirtyTwoK.tokens)
    }

    @MainActor
    @Test func PersistedQwenContextPropagatesToAppModelAndSurvivesGemmaSave() throws {
        let root = try Self.makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("qwen3.6-35b-a3b.gturbo", isDirectory: true)
        var settings = MacAppSettings(
            contextTokens: AppContextLengthOption.eightK.tokens,
            selectedModelID: .qwen3_6)
        settings.setContextTokens(AppContextLengthOption.sixtyFourK.tokens, for: .qwen3_6)
        try MacAppSettingsFileStore.save(settings, forModelDirectory: model)

        let app = AppModel(
            modelDirectory: model,
            settingsPersistenceEnabled: true,
            installationStatusProvider: { _, _ in .missing },
            catalogEntryProvider: { AppModelCatalog.entry(for: $0) })
        #expect(app.selectedModelID == .qwen3_6)
        #expect(app.maxContextTokens == AppContextLengthOption.sixtyFourK.tokens)

        settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        settings.setContextTokens(AppContextLengthOption.sixteenK.tokens, for: .gemma4)
        try MacAppSettingsFileStore.save(settings, forModelDirectory: model)
        let saved = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        #expect(saved.contextTokens(for: .gemma4) == AppContextLengthOption.sixteenK.tokens)
        #expect(saved.contextTokens(for: .qwen3_6) == AppContextLengthOption.sixtyFourK.tokens)

        settings = saved
        settings.selectedModelID = .gemma4
        try MacAppSettingsFileStore.save(settings, forModelDirectory: model)
        let gemmaApp = AppModel(
            modelDirectory: model,
            settingsPersistenceEnabled: true,
            installationStatusProvider: { _, _ in .missing },
            catalogEntryProvider: { AppModelCatalog.entry(for: $0) })
        #expect(gemmaApp.selectedModelID == .gemma4)
        #expect(gemmaApp.maxContextTokens == AppContextLengthOption.sixteenK.tokens)

        settings.selectedModelID = .qwen3_6
        try MacAppSettingsFileStore.save(settings, forModelDirectory: model)
        let qwenApp = AppModel(
            modelDirectory: model,
            settingsPersistenceEnabled: true,
            installationStatusProvider: { _, _ in .missing },
            catalogEntryProvider: { AppModelCatalog.entry(for: $0) })
        #expect(qwenApp.selectedModelID == .qwen3_6)
        #expect(qwenApp.maxContextTokens == AppContextLengthOption.sixtyFourK.tokens)
    }

    @Test func InvalidQwenContextFallsBackAndRewritesDefaultSettings() throws {
        let root = try Self.makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let invalid = Data("""
        {
          "version": 7,
          "contextTokens": 8192,
          "qwenContextTokens": 123,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "selectedModelID": "qwen3.6-35b-a3b"
        }
        """.utf8)
        try invalid.write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        #expect(settings.contextTokens(for: .qwen3_6) == AppContextLengthOption.eightK.tokens)
        #expect(try Data(contentsOf: fileURL) != invalid)
    }

    @Test func InvalidQwenContextInExplicitSettingsFileIsRejectedAndPreserved() throws {
        let root = try Self.makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let fileURL = root.appendingPathComponent("explicit-settings.json")
        let invalid = Data("""
        {
          "version": 7,
          "contextTokens": 8192,
          "qwenContextTokens": 123,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "selectedModelID": "qwen3.6-35b-a3b"
        }
        """.utf8)
        try invalid.write(to: fileURL)

        #expect(throws: (any Error).self) {
            try MacAppSettingsFileStore.loadOrCreate(at: fileURL)
        }
        #expect(try Data(contentsOf: fileURL) == invalid)
    }

    @Test func FutureSettingsVersionIsPreservedWhileUsingLegacyDefaultsInMemory() throws {
        let root = try Self.makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("qwen3.6-35b-a3b.gturbo", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let future = Data("""
        {
          "version": 999,
          "contextTokens": 8192,
          "qwenContextTokens": 65536,
          "selectedModelID": "qwen3.6-35b-a3b"
        }
        """.utf8)
        try future.write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        #expect(settings.contextTokens(for: .qwen3_6) == AppContextLengthOption.eightK.tokens)
        #expect(try Data(contentsOf: fileURL) == future)
    }

    private static func encodedSettings(
        _ settings: MacAppSettings,
        replacingSelectionWith value: String
    ) throws -> Data {
        var object = try #require(JSONSerialization.jsonObject(
            with: JSONEncoder().encode(settings)) as? [String: Any])
        object["selectedModelID"] = value
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .appendingNewline()
    }

    private static func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("app-selection-settings-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }
}

private extension Data {
    func appendingNewline() -> Data {
        var value = self
        value.append(0x0A)
        return value
    }
}
