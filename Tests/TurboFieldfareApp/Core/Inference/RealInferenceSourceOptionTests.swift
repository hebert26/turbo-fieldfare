import Foundation
import Testing
@testable import TurboFieldfareAppCore
@testable import TurboFieldfare

struct RealInferenceSourceOptionTests {
    @Test func selectedSourceOptionsReachSessionKeyAndRuntimeConfiguration() throws {
        // Eight slots are selectable when chunked prefill is off. The tiny
        // source fixture exercises the separate allocation boundary at nine.
        let options = AppRuntimeOptions(
            expertCacheSlots: 8, expertCachePolicy: .lru,
            prefillEnabled: false, modelVerification: .trustedInstall)
        let key = SessionLoadKey(
            directory: URL(fileURLWithPath: "/tmp/packed.gturbo"),
            maxContext: 32, options: options)
        let configuration = try key.options.resolvedRuntimeConfiguration(
            forceLogitsHead: key.forceLogitsHead)
        #expect(configuration.expertCacheSlots == 8)
        #expect(configuration.modelExpertCachePolicy == .lru)
        #expect(key.options.modelVerification.runtimeValue == .sizeCheckTrustedReceipt)
    }

    @Test func sourceSessionKeyKeepsPhysicalParentAndFinalLeaf() throws {
        let fileManager = FileManager.default
        let root = URL(fileURLWithPath: "/private/var/tmp", isDirectory: true)
            .appendingPathComponent("source-key-\(UUID().uuidString)", isDirectory: true)
        defer { try? fileManager.removeItem(at: root) }
        let registration = root.appendingPathComponent("source.gturbo", isDirectory: true)
        try fileManager.createDirectory(at: registration, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: registration.appendingPathComponent("official-source.json"))
        let key = SessionLoadKey(directory: registration, maxContext: 32,
                                 options: AppRuntimeOptions())
        #expect(key.directory.path == registration.path)

        let alias = root.appendingPathComponent("alias.gturbo", isDirectory: true)
        try fileManager.createSymbolicLink(at: alias, withDestinationURL: registration)
        let aliasKey = SessionLoadKey(directory: alias, maxContext: 32,
                                      options: AppRuntimeOptions())
        #expect(aliasKey.directory.path == alias.path,
                "the key cannot resolve a final-leaf symlink into an admitted source")
    }
}
