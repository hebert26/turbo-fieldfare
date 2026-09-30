import Foundation
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfare

/// Opt-in metadata audit of the exact Phase 21 registration. This never loads
/// source weights or starts a model; ordinary package test runs skip it.
@Suite struct OfficialRegistrationAppAuditTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_P21_AUDIT"] == "1"))
    func registeredSourceAndCompanionAreReadyButTamperedMetadataIsNot() throws {
        let text = URL(fileURLWithPath:
            "/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b.gturbo",
            isDirectory: true)
        let vision = URL(fileURLWithPath:
            "/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b.vision.gturbo",
            isDirectory: true)
        let entry = AppModelCatalog.entry(for: .qwen3_6)
        #expect(AppModelInstallationProbe.status(at: text, entry: entry) == .complete)
        #expect(AppVisionPackInstallationProbe.status(at: text, entry: entry) == .complete)
        try OfficialQwenSourceVisionMetadataProbe.verify(
            textModelURL: text, visionURL: vision)

        let temporary = URL(fileURLWithPath:
            "/Users/dev-machine/dev/turbo-fieldfare-personal/scratch/qwen3.6-35b-a3b/evidence/original-bf16-v2/phase-21-codex",
            isDirectory: true)
            .appendingPathComponent("negative-vision-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let sourceBytes = try Data(contentsOf: vision.appendingPathComponent("manifest.json"))
        var object = try #require(JSONSerialization.jsonObject(with: sourceBytes) as? [String: Any])
        object["textContentSHA256"] = String(repeating: "0", count: 64)
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .write(to: temporary.appendingPathComponent("manifest.json"), options: .withoutOverwriting)
        #expect(throws: Error.self) {
            try OfficialQwenSourceVisionMetadataProbe.verify(
                textModelURL: text, visionURL: temporary)
        }
    }
}
