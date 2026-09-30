import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Selected literals already accepted in QwenTokenizerTests and
/// QwenChatTemplateTests. They are independent oracle values, not candidate output.
private enum QwenOfficialSourceTokenizerOracle {
    static let asciiText = "Hello, world!"
    static let asciiIDs: [Int32] = [9419, 11, 1814, 0]
    static let startTokenIDs: [Int32] = [248045]
    static let endTokenIDs: [Int32] = [248046]

    static let promptText =
        "<|im_start|>user\nExplain NFC briefly.<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n"
    static let promptIDs: [Int32] = [
        248045, 846, 198, 814, 20139, 45629, 25899, 13, 248046,
        198, 248045, 74455, 198, 248068, 198,
    ]
}

/// Isolated pinned logical registration with only the two authentic tokenizer
/// sidecars in its source root. The descriptor names official shards, but no
/// shard file is copied or opened by these tests.
private struct QwenOfficialSourceTokenizerFixture {
    let root: URL
    let sourceRoot: URL
    let modelDirectory: URL

    static var officialSidecars: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0",
                isDirectory: true)
    }

    static func make() throws -> Self {
        let fileManager = FileManager.default
        let unresolvedRoot = fileManager.temporaryDirectory.appendingPathComponent(
            "qwen-official-source-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(
            at: unresolvedRoot, withIntermediateDirectories: false)
        var canonicalPath = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(unresolvedRoot.path, &canonicalPath) != nil else {
            try? fileManager.removeItem(at: unresolvedRoot)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let root = URL(
            fileURLWithPath: String(cString: canonicalPath), isDirectory: true)
        let sourceRoot = root.appendingPathComponent("source", isDirectory: true)
        let modelsRoot = root.appendingPathComponent("models", isDirectory: true)
        let modelDirectory = modelsRoot.appendingPathComponent(
            "official.gturbo", isDirectory: true)
        do {
            try fileManager.createDirectory(
                at: sourceRoot, withIntermediateDirectories: false)
            try fileManager.createDirectory(
                at: modelDirectory, withIntermediateDirectories: true)
            for name in ["tokenizer.json", "tokenizer_config.json"] {
                try fileManager.copyItem(
                    at: officialSidecars.appendingPathComponent(name),
                    to: sourceRoot.appendingPathComponent(name))
            }
            let marker = try IndependentOfficialSourceFixture.markerData(
                sourceRoot: sourceRoot.path)
            try marker.write(
                to: modelDirectory.appendingPathComponent(
                    OfficialSourceDescriptor.markerFilename))
            return Self(root: root, sourceRoot: sourceRoot,
                        modelDirectory: modelDirectory)
        } catch {
            try? fileManager.removeItem(at: root)
            throw error
        }
    }

    func makeHandle() throws -> OfficialSourceHandle {
        try OfficialSourceHandle(registrationURL: modelDirectory)
    }

    func makeCodec() throws -> QwenChatCodec {
        let handle = try makeHandle()
        return QwenChatCodec(tokenizer: try QwenTokenizer.loadVerifiedOfficialSource(
            from: handle))
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}

@Suite("Qwen official-source tokenizer")
struct QwenOfficialSourceTokenizerTests {
    @Test("Protected prompt binding matches accepted bytes, IDs and source identity")
    func promptBinding() throws {
        let fixture = try QwenOfficialSourceTokenizerFixture.make()
        defer { fixture.remove() }
        #expect(!FileManager.default.fileExists(atPath: fixture.sourceRoot
            .appendingPathComponent("model-00001-of-00026.safetensors").path))

        let codec = try fixture.makeCodec()
        #expect(codec.tokenizer.encode(QwenOfficialSourceTokenizerOracle.asciiText)
            == QwenOfficialSourceTokenizerOracle.asciiIDs)
        #expect(codec.tokenizer.encode("<|im_start|>")
            == QwenOfficialSourceTokenizerOracle.startTokenIDs)
        #expect(codec.tokenizer.encode("<|im_end|>")
            == QwenOfficialSourceTokenizerOracle.endTokenIDs)

        let messages = [ModelChatMessage(
            role: .user, content: "Explain NFC briefly.")]
        let options = ModelChatRenderOptions(
            addGenerationPrompt: true, enableThinking: true)
        let sourceIdentity = try Self.syntheticSourceIdentity()
        let binding = try QwenOfficialSourcePromptBinding.prepare(
            codec: codec, messages: messages, boundary: nil,
            options: options, sourceIdentity: sourceIdentity)
        #expect(binding.promptBytes
            == Data(QwenOfficialSourceTokenizerOracle.promptText.utf8))
        #expect(binding.tokenIDs == QwenOfficialSourceTokenizerOracle.promptIDs)
        #expect(binding.sourceIdentity == sourceIdentity)
    }

    @Test("Prompt binding accepts matching identity and rejects nil or changed identity")
    func promptBindingIdentityValidation() throws {
        let fixture = try QwenOfficialSourceTokenizerFixture.make()
        defer { fixture.remove() }
        let codec = try fixture.makeCodec()
        let sourceIdentity = try Self.syntheticSourceIdentity()
        let changedIdentity = try Self.syntheticSourceIdentity()
        let binding = try QwenOfficialSourcePromptBinding.prepare(
            codec: codec,
            messages: [ModelChatMessage(
                role: .user, content: "Explain NFC briefly.")],
            boundary: nil,
            options: ModelChatRenderOptions(
                addGenerationPrompt: true, enableThinking: true),
            sourceIdentity: sourceIdentity)

        #expect(!Self.didThrow {
            try binding.validateIdentity(current: sourceIdentity)
        })
        #expect(Self.isIdentityMismatch(Self.captureError {
            try binding.validateIdentity(current: changedIdentity)
        }))
        #expect(Self.isIdentityMismatch(Self.captureError {
            try binding.validateIdentity(current: nil)
        }))
    }

    @Test("Verified source tokenizer rejects either missing pinned sidecar")
    func missingSidecars() throws {
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let fixture = try QwenOfficialSourceTokenizerFixture.make()
            defer { fixture.remove() }
            try FileManager.default.removeItem(
                at: fixture.sourceRoot.appendingPathComponent(name))
            let handle = try fixture.makeHandle()
            let error = Self.captureError {
                _ = try QwenTokenizer.loadVerifiedOfficialSource(from: handle)
            }
            #expect(Self.isMissingSidecar(error, name: name))
        }
    }

    @Test("Verified source tokenizer rejects changed bytes in either pinned sidecar")
    func changedSidecars() throws {
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let fixture = try QwenOfficialSourceTokenizerFixture.make()
            defer { fixture.remove() }
            let url = fixture.sourceRoot.appendingPathComponent(name)
            let file = try FileHandle(forWritingTo: url)
            try file.seekToEnd()
            try file.write(contentsOf: Data([0x20]))
            try file.close()

            let handle = try fixture.makeHandle()
            let error = Self.captureError {
                _ = try QwenTokenizer.loadVerifiedOfficialSource(from: handle)
            }
            #expect(Self.isDigestMismatch(error, file: name))
        }
    }

    @Test("Protected sidecar read rejects shard and traversal names and enforces caller cap")
    func protectedNamesAndCallerCap() throws {
        let fixture = try QwenOfficialSourceTokenizerFixture.make()
        defer { fixture.remove() }
        let handle = try fixture.makeHandle()
        let shardName = "model-00001-of-00026.safetensors"
        #expect(Self.isNotAllowed(
            Self.captureError {
                _ = try handle.readBoundedSidecar(shardName, maximumBytes: 1024)
            }, name: shardName))
        #expect(Self.isNotAllowed(
            Self.captureError {
                _ = try handle.readBoundedSidecar("../tokenizer.json", maximumBytes: 1024)
            }, name: "../tokenizer.json"))
        #expect(Self.isTooLarge(
            Self.captureError {
                _ = try handle.readBoundedSidecar("tokenizer.json", maximumBytes: 1)
            }, name: "tokenizer.json"))
        #expect(!FileManager.default.fileExists(atPath: fixture.sourceRoot
            .appendingPathComponent(shardName).path))
    }

    @Test("Oversized config sidecar is rejected at its fixed cap")
    func oversizedConfigSidecar() throws {
        let fixture = try QwenOfficialSourceTokenizerFixture.make()
        defer { fixture.remove() }
        try Data(repeating: 0, count: 256 * 1024 + 1).write(
            to: fixture.sourceRoot.appendingPathComponent("tokenizer_config.json"))
        let handle = try fixture.makeHandle()
        let error = Self.captureError {
            _ = try QwenTokenizer.loadVerifiedOfficialSource(from: handle)
        }
        #expect(Self.isTooLarge(error, name: "tokenizer_config.json"))
    }

    @Test("Protected source handle rejects a tokenizer sidecar symlink")
    func symlinkSidecar() throws {
        let fixture = try QwenOfficialSourceTokenizerFixture.make()
        defer { fixture.remove() }
        let sidecar = fixture.sourceRoot.appendingPathComponent("tokenizer.json")
        let movedSidecar = fixture.root.appendingPathComponent("copied-tokenizer.json")
        try FileManager.default.moveItem(at: sidecar, to: movedSidecar)
        try FileManager.default.createSymbolicLink(
            atPath: sidecar.path,
            withDestinationPath: Self.officialSidecars.appendingPathComponent(
                "tokenizer.json").path)
        let handle = try fixture.makeHandle()
        #expect(Self.isHandleError(Self.captureError {
            _ = try handle.readBoundedSidecar(
                "tokenizer.json", maximumBytes: 16 * 1024 * 1024)
        }))
    }

    @Test("Retained handle rejects a previously accepted sidecar after replacement")
    func replacedAcceptedSidecar() throws {
        let fixture = try QwenOfficialSourceTokenizerFixture.make()
        defer { fixture.remove() }
        let handle = try fixture.makeHandle()
        _ = try handle.readBoundedSidecar(
            "tokenizer_config.json", maximumBytes: 256 * 1024)

        let sidecar = fixture.sourceRoot.appendingPathComponent("tokenizer_config.json")
        let saved = fixture.root.appendingPathComponent("accepted-config.json")
        try FileManager.default.moveItem(at: sidecar, to: saved)
        try FileManager.default.copyItem(
            at: Self.officialSidecars.appendingPathComponent("tokenizer_config.json"),
            to: sidecar)
        let error = Self.captureError {
            _ = try handle.readBoundedSidecar(
                "tokenizer_config.json", maximumBytes: 256 * 1024)
        }
        #expect(Self.isChangedFileIdentity(error))
    }

    private static var officialSidecars: URL {
        QwenOfficialSourceTokenizerFixture.officialSidecars
    }

    private static func captureError(_ body: () throws -> Void) -> Error? {
        do {
            try body()
            return nil
        } catch {
            return error
        }
    }

    private static func didThrow(_ body: () throws -> Void) -> Bool {
        captureError(body) != nil
    }

    private static func syntheticSourceIdentity() throws -> LoadedRuntimeSourceIdentity {
        let fixture = try TinyOfficialSourceIntegrityFixture.make(includeReceipt: true)
        defer { fixture.remove() }
        return LoadedRuntimeSourceIdentity(receipt: try fixture.verifyTrustedReopen())
    }

    private static func isIdentityMismatch(_ error: Error?) -> Bool {
        guard let error = error as? ModelFamilyGenerationError else { return false }
        return error == .modelIdentityChanged
    }

    private static func isMissingSidecar(_ error: Error?, name: String) -> Bool {
        guard let error = error as? OfficialSourceHandleError,
              case .replaced(let detail) = error else { return false }
        return detail.contains("allowlisted name is absent") && detail.contains(name)
    }

    private static func isDigestMismatch(_ error: Error?, file: String) -> Bool {
        guard let error = error as? QwenTokenizerError,
              case .checksumMismatch(let actualFile, _, _) = error else { return false }
        return actualFile == file
    }

    private static func isNotAllowed(_ error: Error?, name: String) -> Bool {
        guard let error = error as? OfficialSourceHandleError,
              case .notAllowed(let actualName) = error else { return false }
        return actualName == name
    }

    private static func isTooLarge(_ error: Error?, name: String) -> Bool {
        guard let error = error as? OfficialSourceHandleError,
              case .range(let detail) = error else { return false }
        return detail.contains("sidecar exceeds bounded size") && detail.contains(name)
    }

    private static func isHandleError(_ error: Error?) -> Bool {
        error is OfficialSourceHandleError
    }

    private static func isChangedFileIdentity(_ error: Error?) -> Bool {
        guard let error = error as? OfficialSourceHandleError,
              case .replaced(let detail) = error else { return false }
        return detail.contains("previously accepted source file changed")
            && detail.contains("tokenizer_config.json")
    }
}
