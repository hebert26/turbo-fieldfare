import Darwin
import Foundation
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare

/// Independent literals captured from the pinned Qwen fast-tokenizer oracle.
/// These values are deliberately not derived from the production codec.
private enum QwenOracleVectors {
    static let tokenizerBaseVocabulary = 248_044
    static let tokenizerLengthWithAddedTokens = 248_077
    static let modelLogitsVocabulary = 248_320
    static let embeddedMergeCount = 247_587

    static let tokenizerJSONSpecialIDs: Set<Int32> = Set(248_044...248_057)
    static let configDeclaredSpecialIDs: Set<Int32> = Set(248_044...248_057)
        .union(Set(248_070...248_076))
    /// The nine marker IDs reported by the pinned Python probe. The Swift
    /// decoder intentionally uses the broader 21-ID config special set.
    static let pythonSpecialIDs: Set<Int32> = [
        248_044, 248_046, 248_053, 248_054, 248_056, 248_057,
        248_070, 248_071, 248_076,
    ]

    static let byteTokenIDs: [UInt8: Int32] = [
        0x28: 7, 0x80: 222, 0x8C: 234, 0x98: 246,
        0x9F: 253, 0xBC: 120, 0xC3: 127, 0xA9: 102,
        0xF0: 172, 0xFF: 187,
    ]

    static let ascii = (text: "Hello, world!", ids: [Int32](arrayLiteral: 9419, 11, 1814, 0))
    static let unicode = (
        text: "雪 😀 café é",
        ids: [Int32](arrayLiteral: 97055, 87209, 50203, 3825))
    static let whitespaceAndPunctuation = (
        text: "  one\n\nTwo... 42!\t",
        ids: [Int32](arrayLiteral: 220, 799, 271, 11280, 1076, 220, 19, 17, 0, 197))
    static let nfcComposed = (text: "café", ids: [Int32](arrayLiteral: 895, 56868))
    static let nfcDecomposed = (text: "café", ids: [Int32](arrayLiteral: 895, 56868))
    static let markerText = "<tool_call><think><|vision_start|><|image_pad|><|vision_end|>"
    static let markerIDs: [Int32] = [248058, 248068, 248053, 248056, 248054]

    static let validEmojiIDs: [Int32] = [172, 253, 246, 222]
    static let incompleteEmojiIDs: [Int32] = [172, 253]
    static let genuinelyInvalidUTF8IDs: [Int32] = [172, 7, 234, 120]
    static let specialBetweenValidHalvesIDs: [Int32] = [172, 253, 248045, 246, 222]
    static let nonSpecialBetweenValidHalvesIDs: [Int32] = [172, 253, 248068, 246, 222]
    static let paddedUnknownBetweenValidHalvesIDs: [Int32] = [172, 253, 248100, 246, 222]
    static let beyondVocabularyUnknownBetweenValidHalvesIDs: [Int32] = [172, 253, 249077, 246, 222]

    static let invalidUTF8Text = "�(��"
    static let incompleteUTF8FlushText = "�"
    static let validEmojiText = "😀"
    static let specialKeepText = "�<|im_start|>��"
    static let nonSpecialBoundaryText = "�<think>��"

    /// Every config-declared special token is skipped by the pinned decoder;
    /// special:false added tokens remain literal text in skip mode.
    static let addedTokenDeclarations: [(id: Int32, content: String, special: Bool)] = [
        (248044, "<|endoftext|>", true),
        (248045, "<|im_start|>", true),
        (248046, "<|im_end|>", true),
        (248047, "<|object_ref_start|>", true),
        (248048, "<|object_ref_end|>", true),
        (248049, "<|box_start|>", true),
        (248050, "<|box_end|>", true),
        (248051, "<|quad_start|>", true),
        (248052, "<|quad_end|>", true),
        (248053, "<|vision_start|>", true),
        (248054, "<|vision_end|>", true),
        (248055, "<|vision_pad|>", true),
        (248056, "<|image_pad|>", true),
        (248057, "<|video_pad|>", true),
        (248058, "<tool_call>", false),
        (248059, "</tool_call>", false),
        (248060, "<|fim_prefix|>", false),
        (248061, "<|fim_middle|>", false),
        (248062, "<|fim_suffix|>", false),
        (248063, "<|fim_pad|>", false),
        (248064, "<|repo_name|>", false),
        (248065, "<|file_sep|>", false),
        (248066, "<tool_response>", false),
        (248067, "</tool_response>", false),
        (248068, "<think>", false),
        (248069, "</think>", false),
        (248070, "<|audio_start|>", true),
        (248071, "<|audio_end|>", true),
        (248072, "<tts_pad>", true),
        (248073, "<tts_text_bos>", true),
        (248074, "<tts_text_eod>", true),
        (248075, "<tts_text_bos_single>", true),
        (248076, "<|audio_pad|>", true),
    ]
}

@Suite("Qwen tokenizer")
struct QwenTokenizerTests {
    let tokenizer: QwenTokenizer

    init() throws {
        let directory = try Self.makeCopiedSidecarDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        tokenizer = try QwenTokenizer.loadOfficialSidecar(from: directory)
    }

    private static var fixtureDirectory: URL {
        // The pinned offline sidecar is read only as the source for disposable
        // regular-file copies. The loader never opens these source inodes.
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0", isDirectory: true)
    }

    @Test("Pinned metadata retains model and tokenizer vocabulary geometry")
    func metadata() {
        #expect(QwenTokenizer.baseVocabularySize == QwenOracleVectors.tokenizerBaseVocabulary)
        #expect(QwenTokenizer.tokenizerLength == QwenOracleVectors.tokenizerLengthWithAddedTokens)
        #expect(QwenTokenizer.modelVocabularySize == QwenOracleVectors.modelLogitsVocabulary)
        #expect(QwenTokenizer.mergeCount == QwenOracleVectors.embeddedMergeCount)
    }

    @Test("Effective special IDs include every config-declared special token")
    func effectiveSpecialIDs() {
        #expect(tokenizer.effectiveSpecialTokenIDs == QwenOracleVectors.configDeclaredSpecialIDs)
    }

    @Test("Pinned BPE vectors match independent oracle")
    func encodeVectors() {
        #expect(tokenizer.encode(QwenOracleVectors.ascii.text) == QwenOracleVectors.ascii.ids)
        #expect(tokenizer.encode(QwenOracleVectors.unicode.text) == QwenOracleVectors.unicode.ids)
        #expect(tokenizer.encode(QwenOracleVectors.whitespaceAndPunctuation.text) == QwenOracleVectors.whitespaceAndPunctuation.ids)
        #expect(tokenizer.encode(QwenOracleVectors.nfcComposed.text) == QwenOracleVectors.nfcComposed.ids)
        #expect(tokenizer.encode(QwenOracleVectors.nfcDecomposed.text) == QwenOracleVectors.nfcDecomposed.ids)
        #expect(tokenizer.encode(QwenOracleVectors.markerText) == QwenOracleVectors.markerIDs)
    }

    @Test("NFC composed and decomposed input share the pinned normalized encoding")
    func nfcNormalization() {
        #expect(tokenizer.encode(QwenOracleVectors.nfcComposed.text) == tokenizer.encode(QwenOracleVectors.nfcDecomposed.text))
    }

    @Test("All 33 added tokens retain IDs, spelling, specialness, and decode behavior")
    func allAddedTokens() {
        #expect(tokenizer.addedTokens.count == QwenOracleVectors.addedTokenDeclarations.count)
        for (actual, expected) in zip(tokenizer.addedTokens, QwenOracleVectors.addedTokenDeclarations) {
            #expect(actual.id == expected.id)
            #expect(actual.content == expected.content)
            #expect(actual.isSpecial == expected.special)
            #expect(tokenizer.encode(expected.content) == [expected.id])
            #expect(tokenizer.decode([expected.id], skipSpecialTokens: false) == expected.content)
            let skipped = tokenizer.decode([expected.id])
            #expect(skipped == (expected.special ? "" : expected.content))
        }
    }

    @Test("Effective special set distinguishes config-only declarations")
    func specialDeclarationSets() {
        #expect(QwenOracleVectors.configDeclaredSpecialIDs.count == 21)
        #expect(QwenOracleVectors.tokenizerJSONSpecialIDs.count == 14)
        #expect(tokenizer.effectiveSpecialTokenIDs.count == 21)
        #expect(tokenizer.effectiveSpecialTokenIDs == QwenOracleVectors.configDeclaredSpecialIDs)
        #expect(QwenOracleVectors.pythonSpecialIDs.count == 9)
        #expect(QwenOracleVectors.pythonSpecialIDs.isSubset(of: tokenizer.effectiveSpecialTokenIDs))
        #expect(QwenOracleVectors.pythonSpecialIDs != tokenizer.effectiveSpecialTokenIDs)
    }

    @Test("Batch decode follows pinned byte and special-token boundaries")
    func decodeVectors() {
        #expect(tokenizer.decode(QwenOracleVectors.validEmojiIDs) == QwenOracleVectors.validEmojiText)
        #expect(tokenizer.decode(QwenOracleVectors.genuinelyInvalidUTF8IDs) == QwenOracleVectors.invalidUTF8Text)
        #expect(tokenizer.decode(QwenOracleVectors.specialBetweenValidHalvesIDs) == QwenOracleVectors.validEmojiText)
        #expect(tokenizer.decode(QwenOracleVectors.paddedUnknownBetweenValidHalvesIDs) == QwenOracleVectors.validEmojiText)
        #expect(tokenizer.decode(QwenOracleVectors.beyondVocabularyUnknownBetweenValidHalvesIDs) == QwenOracleVectors.validEmojiText)
        #expect(tokenizer.decode(QwenOracleVectors.specialBetweenValidHalvesIDs, skipSpecialTokens: false) == QwenOracleVectors.specialKeepText)
        #expect(tokenizer.decode(QwenOracleVectors.nonSpecialBetweenValidHalvesIDs) == QwenOracleVectors.nonSpecialBoundaryText)
    }

    @Test("Incremental decoder holds incomplete bytes through skipped and unknown IDs")
    func incrementalValidUTF8AndBoundaries() {
        var decoder = tokenizer.makeIncrementalDecoder()
        var deltas: [String] = []
        for id in QwenOracleVectors.specialBetweenValidHalvesIDs {
            deltas.append(decoder.push(id))
        }
        let finish = decoder.finish()
        #expect(deltas == ["", "", "", "", QwenOracleVectors.validEmojiText])
        #expect(finish == "")
        #expect(deltas.joined() + finish == tokenizer.decode(QwenOracleVectors.specialBetweenValidHalvesIDs))

        decoder = tokenizer.makeIncrementalDecoder()
        deltas.removeAll(keepingCapacity: true)
        for id in QwenOracleVectors.paddedUnknownBetweenValidHalvesIDs {
            deltas.append(decoder.push(id))
        }
        let unknownFinish = decoder.finish()
        #expect(deltas == ["", "", "", "", QwenOracleVectors.validEmojiText])
        #expect(unknownFinish == "")
        #expect(deltas.joined() + unknownFinish == QwenOracleVectors.validEmojiText)
        #expect(deltas.joined() + unknownFinish == tokenizer.decode(QwenOracleVectors.paddedUnknownBetweenValidHalvesIDs))
    }

    @Test("Batch and incremental decode flush an incomplete UTF-8 prefix")
    func incrementalIncompleteUTF8() {
        #expect(tokenizer.decode(QwenOracleVectors.incompleteEmojiIDs) == QwenOracleVectors.incompleteUTF8FlushText)
        var decoder = tokenizer.makeIncrementalDecoder()
        #expect(decoder.push(QwenOracleVectors.incompleteEmojiIDs[0]) == "")
        #expect(decoder.push(QwenOracleVectors.incompleteEmojiIDs[1]) == "")
        #expect(decoder.finish() == QwenOracleVectors.incompleteUTF8FlushText)
    }

    @Test("Incremental keep mode treats special tokens as boundaries")
    func incrementalKeepModeBoundaries() {
        var decoder = tokenizer.makeIncrementalDecoder(skipSpecialTokens: false)
        var output = ""
        for id in QwenOracleVectors.specialBetweenValidHalvesIDs {
            output += decoder.push(id)
        }
        output += decoder.finish()
        #expect(output == QwenOracleVectors.specialKeepText)
        #expect(output == tokenizer.decode(QwenOracleVectors.specialBetweenValidHalvesIDs, skipSpecialTokens: false))

        decoder = tokenizer.makeIncrementalDecoder()
        output = ""
        for id in QwenOracleVectors.nonSpecialBetweenValidHalvesIDs {
            output += decoder.push(id)
        }
        output += decoder.finish()
        #expect(output == QwenOracleVectors.nonSpecialBoundaryText)
        #expect(output == tokenizer.decode(QwenOracleVectors.nonSpecialBetweenValidHalvesIDs))
    }

    @Test("Push concatenation plus finish equals batch decode in both modes")
    func incrementalConcatenationInvariant() {
        let sequences = [
            QwenOracleVectors.validEmojiIDs,
            QwenOracleVectors.incompleteEmojiIDs,
            QwenOracleVectors.genuinelyInvalidUTF8IDs,
            QwenOracleVectors.specialBetweenValidHalvesIDs,
            QwenOracleVectors.nonSpecialBetweenValidHalvesIDs,
            QwenOracleVectors.paddedUnknownBetweenValidHalvesIDs,
        ]
        for ids in sequences {
            for skipSpecialTokens in [true, false] {
                var decoder = tokenizer.makeIncrementalDecoder(skipSpecialTokens: skipSpecialTokens)
                var output = ""
                for id in ids { output += decoder.push(id) }
                output += decoder.finish()
                #expect(output == tokenizer.decode(ids, skipSpecialTokens: skipSpecialTokens))
            }
        }
    }

    @Test("Incremental invalid-byte output concatenates to the pinned decode")
    func incrementalInvalidUTF8() {
        var decoder = tokenizer.makeIncrementalDecoder()
        var output = ""
        for id in QwenOracleVectors.genuinelyInvalidUTF8IDs {
            output += decoder.push(id)
        }
        output += decoder.finish()
        #expect(output == QwenOracleVectors.invalidUTF8Text)
    }

    @Test("Unknown IDs are ignored without flushing a pending byte sequence")
    func unknownIDsDoNotFlush() {
        let sequences = [
            QwenOracleVectors.paddedUnknownBetweenValidHalvesIDs,
            QwenOracleVectors.beyondVocabularyUnknownBetweenValidHalvesIDs,
        ]
        for ids in sequences {
            var decoder = tokenizer.makeIncrementalDecoder()
            var output = ""
            for id in ids { output += decoder.push(id) }
            output += decoder.finish()
            #expect(output == tokenizer.decode(ids))
            #expect(output == QwenOracleVectors.validEmojiText)
        }
    }

    @Test("Official-sidecar seam rejects missing required JSON files")
    func loaderRejectsMissingFiles() throws {
        let root = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(qwenError(at: root) == .missingFile("tokenizer.json"))

        try Self.copyOfficial("tokenizer_config.json", into: root)
        #expect(qwenError(at: root) == .missingFile("tokenizer.json"))

        try Self.copyOfficial("tokenizer.json", into: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("tokenizer_config.json"))
        #expect(qwenError(at: root) == .missingFile("tokenizer_config.json"))
    }

    @Test("Official-sidecar seam rejects malformed or digest-tampered required JSON")
    func loaderRejectsMalformedRequiredJSON() throws {
        let root = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("{not-json".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        try Self.copyOfficial("tokenizer_config.json", into: root)
        guard case .checksumMismatch(file: "tokenizer.json", _, _) = qwenError(at: root) else {
            Issue.record("tampered tokenizer.json was not rejected by its pinned digest")
            return
        }
    }

    @Test("Official-sidecar seam enforces the fixed byte caps")
    func loaderRejectsOversizedSidecars() throws {
        let cases: [(String, UInt64, String)] = [
            ("tokenizer.json", 16_777_216, "tokenizer.json"),
            ("tokenizer_config.json", 262_144, "tokenizer_config.json"),
        ]
        for (name, maximumBytes, expectedName) in cases {
            let root = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let bytes = Data(repeating: 0x20, count: Int(maximumBytes) + 1)
            try bytes.write(to: root.appendingPathComponent(name))
            let other = name == "tokenizer.json" ? "tokenizer_config.json" : "tokenizer.json"
            try Self.copyOfficial(other, into: root)
            guard case let .fileTooLarge(file, limit) = qwenError(at: root) else {
                Issue.record("oversized \(name) was not rejected with fileTooLarge")
                continue
            }
            #expect(file == expectedName)
            #expect(limit == maximumBytes)
        }
    }

    @Test("Installed admission accepts a copied sidecar pair with a verified Qwen manifest")
    func installedAdmission() throws {
        let root = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let loaded = try QwenTokenizer.load(from: root)
        #expect(loaded.addedTokens.count == QwenOracleVectors.addedTokenDeclarations.count)
        #expect(loaded.effectiveSpecialTokenIDs == QwenOracleVectors.configDeclaredSpecialIDs)
    }

    @Test("Public installed loading rejects a validator-accepted wrong-family Gemma manifest")
    func publicLoaderRejectsWrongFamilyManifest() throws {
        let root = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let gemmaManifest = try Self.gemmaManifestData()
        let decodedGemma = try ManifestReader.decodeVerified(data: gemmaManifest)
        #expect(decodedGemma.descriptor.family == .gemma4)
        try gemmaManifest
            .write(to: root.appendingPathComponent("manifest.json"), options: [.atomic])
        guard case let .model(error) = loadOutcome(at: root) else {
            Issue.record("wrong-family manifest did not remain in ModelError")
            return
        }
        guard case .indexCorrupt(detail: _) = error else {
            Issue.record("wrong-family manifest escaped as the wrong ModelError case")
            return
        }
    }

    @Test("Public installed loading rejects a bare copied sidecar root")
    func publicLoaderRejectsBareSidecarRoot() throws {
        let root = try Self.makeCopiedSidecarDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        guard case let .model(error) = loadOutcome(at: root) else {
            Issue.record("bare sidecar root did not preserve a manifest-domain error")
            return
        }
        #expect(error == .partialInstall(path: root.path))
    }

    @Test("Installed asset reads translate an ENOTDIR intermediate path")
    func installedAssetENOTDIR() throws {
        let root = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.removeItem(at: tokenizerDirectory)
        try Data("not a directory".utf8).write(to: tokenizerDirectory)
        guard case let .qwen(error) = loadOutcome(at: root) else {
            Issue.record("ENOTDIR asset path did not translate to a Qwen tokenizer error")
            return
        }
        #expect(error == .unreadableFile("tokenizer/tokenizer.json"))
    }

    @Test("Manifest sidecar digest disagreement remains an admission error")
    func manifestDigestDisagreement() throws {
        let root = try Self.makeInstalledDirectory(tamperManifestSidecar: true)
        defer { try? FileManager.default.removeItem(at: root) }
        guard case let .model(error) = loadOutcome(at: root) else {
            Issue.record("manifest digest disagreement did not remain in ModelError domain")
            return
        }
        guard case .indexCorrupt(detail: _) = error else {
            Issue.record("manifest digest disagreement escaped as the wrong ModelError case")
            return
        }
    }

    @Test("Public manifest admission rejects malformed, oversized, and non-regular manifests")
    func publicManifestAdmissionHostiles() throws {
        let malformedRoot = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: malformedRoot) }
        try Data("{not-json".utf8)
            .write(to: malformedRoot.appendingPathComponent("manifest.json"), options: [.atomic])
        guard case let .model(malformed) = loadOutcome(at: malformedRoot) else {
            Issue.record("malformed manifest did not remain in ModelError")
            return
        }
        guard case .indexCorrupt(detail: _) = malformed else {
            Issue.record("malformed manifest escaped as the wrong ModelError case")
            return
        }

        let oversizedRoot = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: oversizedRoot) }
        let manifestCap = ManifestReader.defaultMaxBytes
        try Data(repeating: 0, count: Int(manifestCap) + 1)
            .write(to: oversizedRoot.appendingPathComponent("manifest.json"), options: [.atomic])
        guard case let .model(oversized) = loadOutcome(at: oversizedRoot) else {
            Issue.record("oversized manifest did not remain in ModelError")
            return
        }
        guard case .indexCorrupt(detail: _) = oversized else {
            Issue.record("oversized manifest escaped as the wrong ModelError case")
            return
        }

        let symlinkRoot = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: symlinkRoot) }
        let symlinkTarget = symlinkRoot.appendingPathComponent("manifest-copy.json")
        try Self.copyFile(
            from: symlinkRoot.appendingPathComponent("manifest.json"), to: symlinkTarget)
        try FileManager.default.removeItem(at: symlinkRoot.appendingPathComponent("manifest.json"))
        try FileManager.default.createSymbolicLink(
            at: symlinkRoot.appendingPathComponent("manifest.json"),
            withDestinationURL: symlinkTarget)
        guard case let .model(symlink) = loadOutcome(at: symlinkRoot) else {
            Issue.record("manifest symlink did not remain in ModelError")
            return
        }
        guard case .posixFailed(call: _, errno: _) = symlink else {
            Issue.record("manifest symlink did not take the no-follow posix path")
            return
        }

        let directoryRoot = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: directoryRoot) }
        try FileManager.default.removeItem(at: directoryRoot.appendingPathComponent("manifest.json"))
        try FileManager.default.createDirectory(
            at: directoryRoot.appendingPathComponent("manifest.json"), withIntermediateDirectories: false)
        guard case let .model(directory) = loadOutcome(at: directoryRoot) else {
            Issue.record("manifest directory did not remain in ModelError")
            return
        }
        guard case .posixFailed(call: _, errno: _) = directory else {
            Issue.record("manifest directory did not take the non-regular-file path")
            return
        }

        let fifoRoot = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: fifoRoot) }
        try FileManager.default.removeItem(at: fifoRoot.appendingPathComponent("manifest.json"))
        try Self.makeFIFO(at: fifoRoot.appendingPathComponent("manifest.json"))
        guard case let .model(fifo) = loadOutcome(at: fifoRoot) else {
            Issue.record("manifest FIFO did not remain in ModelError")
            return
        }
        guard case .posixFailed(call: _, errno: _) = fifo else {
            Issue.record("manifest FIFO did not take the non-regular-file path")
            return
        }
    }

    @Test("Public installed loading translates missing, caps, and tampered assets")
    func publicAssetAdmissionFailures() throws {
        let missingRoot = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: missingRoot) }
        try FileManager.default.removeItem(
            at: missingRoot.appendingPathComponent("tokenizer/tokenizer.json"))
        guard case let .qwen(missing) = loadOutcome(at: missingRoot) else {
            Issue.record("missing installed tokenizer asset did not become QwenTokenizerError")
            return
        }
        #expect(missing == .missingFile("tokenizer/tokenizer.json"))

        let capCases: [(String, UInt64)] = [
            ("tokenizer/tokenizer.json", 16_777_216),
            ("tokenizer/tokenizer_config.json", 262_144),
        ]
        for (relativePath, maximumBytes) in capCases {
            let root = try Self.makeInstalledDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            try Data(repeating: 0x20, count: Int(maximumBytes) + 1)
                .write(to: root.appendingPathComponent(relativePath), options: [.atomic])
            guard case let .qwen(error) = loadOutcome(at: root) else {
                Issue.record("public asset cap was not translated for \(relativePath)")
                continue
            }
            #expect(error == .fileTooLarge(file: relativePath, maximumBytes: maximumBytes))
        }

        let tamperedCases = [
            ("tokenizer/tokenizer.json", QwenTokenizer.tokenizerJSONSHA256),
            ("tokenizer/tokenizer_config.json", QwenTokenizer.tokenizerConfigJSONSHA256),
        ]
        for (relativePath, expectedDigest) in tamperedCases {
            let root = try Self.makeInstalledDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            try Data("tampered installed sidecar".utf8)
                .write(to: root.appendingPathComponent(relativePath), options: [.atomic])
            guard case let .qwen(error) = loadOutcome(at: root) else {
                Issue.record("tampered public asset was not translated for \(relativePath)")
                continue
            }
            guard case let .checksumMismatch(file, expected, actual) = error else {
                Issue.record("tampered public asset escaped as the wrong Qwen error")
                continue
            }
            #expect(file == relativePath)
            #expect(expected == expectedDigest)
            #expect(actual != expectedDigest)
        }
    }

    @Test("Public installed loading rejects asset symlink, directory, FIFO, and intermediate symlink")
    func publicAssetNonRegularFailures() throws {
        let leafCases = [
            ("tokenizer/tokenizer.json", "tokenizer-leaf-target.json"),
            ("tokenizer/tokenizer_config.json", "config-leaf-target.json"),
        ]
        for (relativePath, targetName) in leafCases {
            let root = try Self.makeInstalledDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let leaf = root.appendingPathComponent(relativePath)
            let target = root.appendingPathComponent(targetName)
            try Self.copyFile(from: leaf, to: target)
            try FileManager.default.removeItem(at: leaf)
            try FileManager.default.createSymbolicLink(at: leaf, withDestinationURL: target)
            guard case let .qwen(error) = loadOutcome(at: root) else {
                Issue.record("asset symlink was not translated for \(relativePath)")
                continue
            }
            #expect(error == .unreadableFile(relativePath))
        }

        for relativePath in ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            let root = try Self.makeInstalledDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let leaf = root.appendingPathComponent(relativePath)
            try FileManager.default.removeItem(at: leaf)
            try FileManager.default.createDirectory(at: leaf, withIntermediateDirectories: false)
            guard case let .qwen(error) = loadOutcome(at: root) else {
                Issue.record("asset directory was not translated for \(relativePath)")
                continue
            }
            #expect(error == .unreadableFile(relativePath))
        }

        for relativePath in ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            let root = try Self.makeInstalledDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let leaf = root.appendingPathComponent(relativePath)
            try FileManager.default.removeItem(at: leaf)
            try Self.makeFIFO(at: leaf)
            guard case let .qwen(error) = loadOutcome(at: root) else {
                Issue.record("asset FIFO was not translated for \(relativePath)")
                continue
            }
            #expect(error == .unreadableFile(relativePath))
        }

        let root = try Self.makeInstalledDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceDirectory = root.appendingPathComponent("tokenizer")
        let targetDirectory = root.appendingPathComponent("tokenizer-target", isDirectory: true)
        try FileManager.default.createDirectory(at: targetDirectory, withIntermediateDirectories: false)
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            try Self.copyFile(
                from: sourceDirectory.appendingPathComponent(name),
                to: targetDirectory.appendingPathComponent(name))
        }
        try FileManager.default.removeItem(at: sourceDirectory)
        try FileManager.default.createSymbolicLink(at: sourceDirectory, withDestinationURL: targetDirectory)
        guard case let .qwen(error) = loadOutcome(at: root) else {
            Issue.record("intermediate tokenizer symlink was not translated")
            return
        }
        #expect(error == .unreadableFile("tokenizer/tokenizer.json"))
    }

    @Test("Loader ignores malformed external vocab, merges, template, and generation sidecars")
    func loaderIgnoresExternalSidecars() throws {
        let root = try Self.makeCopiedSidecarDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"] {
            try Data("poison and malformed external sidecar".utf8)
                .write(to: root.appendingPathComponent(name))
        }
        let loaded = try QwenTokenizer.loadOfficialSidecar(from: root)
        #expect(loaded.addedTokens.count == 33)
        #expect(loaded.effectiveSpecialTokenIDs == QwenOracleVectors.configDeclaredSpecialIDs)
    }

    private enum LoadOutcome {
        case success
        case qwen(QwenTokenizerError)
        case model(ModelError)
        case other(String)
    }

    private static func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-tokenizer-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func makeCopiedSidecarDirectory() throws -> URL {
        let directory = try temporaryDirectory()
        do {
            try copyOfficial("tokenizer.json", into: directory)
            try copyOfficial("tokenizer_config.json", into: directory)
            return directory
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private static func copyOfficial(_ name: String, into directory: URL) throws {
        let source = fixtureDirectory.appendingPathComponent(name)
        let destination = directory.appendingPathComponent(name)
        let bytes = try Data(contentsOf: source, options: [.mappedIfSafe])
        try bytes.write(to: destination, options: [.atomic])
    }

    private static func copyFile(from source: URL, to destination: URL) throws {
        let bytes = try Data(contentsOf: source, options: [.mappedIfSafe])
        try bytes.write(to: destination, options: [.atomic])
    }

    private static func makeFIFO(at path: URL) throws {
        guard Darwin.mkfifo(path.path, 0o600) == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
    }

    private static func makeInstalledDirectory(
        tamperManifestSidecar: Bool = false
    ) throws -> URL {
        let root = try temporaryDirectory()
        do {
            let tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
            try FileManager.default.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
            try copyOfficial("tokenizer.json", into: tokenizerDirectory)
            try copyOfficial("tokenizer_config.json", into: tokenizerDirectory)
            var manifest = try manifestData(sidecarSHA256: GTurboFormatV2.qwenSidecarSHA256)
            if tamperManifestSidecar {
                let expected = Data(QwenTokenizer.tokenizerJSONSHA256.utf8)
                let replacement = Data(repeating: 0x30, count: expected.count)
                guard let range = manifest.range(of: expected) else {
                    throw NSError(domain: "QwenTokenizerTests", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "manifest digest was not serialized"])
                }
                manifest.replaceSubrange(range, with: replacement)
            }
            try manifest.write(to: root.appendingPathComponent("manifest.json"), options: [.atomic])
            return root
        } catch {
            try? FileManager.default.removeItem(at: root)
            throw error
        }
    }

    private static func gemmaManifestData() throws -> Data {
        let digest = String(repeating: "a", count: 64)
        let architecture = GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1, headDim: 16,
            fullHeadDim: 32, vocabSize: 1_024, slidingWindow: 128,
            finalLogitSoftcap: 30, ropeTheta: 10_000,
            fullRopeTheta: 1_000_000, partialRotaryFactor: 0.25,
            numLayers: 1, numExperts: 2, topKExperts: 1,
            tieWordEmbeddings: true, attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0])
        let categories: [GTurboQuantizationCategoryV2] = [
            .embedding, .attention, .router, .sharedExpert, .routedExpert,
        ]
        let groups = categories.map {
            GTurboQuantizationGroupV2(
                category: $0, storage: .affineInt4, groupSize: 64,
                scaleType: "bf16", biasType: "bf16")
        }
        return try GTurboManifestV2Codec.encode(.init(
            family: .gemma4, requiredFeatures: [.familyDispatch, .verifiedIdentity],
            modelID: "fixture/gemma", architecture: .gemma4(architecture),
            provenance: .init(
                sourceRepository: "fixture/gemma", sourceRevision: "fixture-revision",
                sourceIndexSHA256: digest, sidecarSHA256: ["config.json": digest],
                quantizationPolicySHA256: digest),
            quantization: groups, ignoredTensors: [],
            files: ["model_weights.bin": .init(size: 16_384, sha256: digest),
                    "packed_experts/layout.json": .init(size: 1, sha256: digest)],
            tensorRegions: [.init(name: "embed", file: "model_weights.bin", offset: 0,
                                  size: 16, shape: [1], storage: .affineInt4,
                                  quantizationCategory: .embedding)],
            expertsPerLayer: 2, numLayers: 1, expertStride: 16_384))
    }

    private static func manifestData(sidecarSHA256: [String: String]) throws -> Data {
        let digest = String(repeating: "a", count: 64)
        let layers: [GTurboQwenLayerTypeV2] = (0..<40).map {
            ($0 + 1).isMultiple(of: 4) ? .fullAttention : .linearAttention
        }
        let architecture = GTurboQwenArchitectureV2(
            hiddenSize: 2048, numLayers: 40, layerTypes: layers,
            numAttentionHeads: 16, numKeyValueHeads: 2, headDimension: 256,
            attentionOutputGate: true, linearConvolutionKernel: 4,
            linearKeyHeads: 16, linearKeyHeadDimension: 128,
            linearValueHeads: 32, linearValueHeadDimension: 128,
            recurrentStateType: .fp32, partialRotaryFactor: 0.25,
            ropeTheta: 10_000_000, mropeInterleaved: true,
            mropeSections: [11, 11, 10], numberOfExperts: 256,
            expertsPerToken: 8, routedExpertIntermediateSize: 512,
            sharedExpertIntermediateSize: 512, vocabularySize: 248_320,
            tiedWordEmbeddings: false, hiddenActivation: "silu",
            bosTokenID: 248_044, eosTokenID: 248_044, imageTokenID: 248_056,
            videoTokenID: 248_057, visionStartTokenID: 248_053,
            visionEndTokenID: 248_054)
        let groups = GTurboQuantizationCategoryV2.allCases.map { category in
            category == .recurrentState
                ? GTurboQuantizationGroupV2(category: category, storage: .fp32)
                : GTurboQuantizationGroupV2(
                    category: category, storage: .affineInt4, groupSize: 64,
                    scaleType: "bf16", biasType: "bf16")
        }
        let omissions = ["fc.weight", "layers.0.input_layernorm.weight",
            "layers.0.mlp.experts.down_proj", "layers.0.mlp.experts.gate_up_proj",
            "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert.down_proj.weight",
            "layers.0.mlp.shared_expert.gate_proj.weight",
            "layers.0.mlp.shared_expert.up_proj.weight",
            "layers.0.mlp.shared_expert_gate.weight",
            "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight",
            "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.v_proj.weight", "norm.weight",
            "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"].map {
                GTurboIgnoredTensorV2(name: "mtp." + $0, reason: .unsupportedMTP)
            }
        return try GTurboManifestV2Codec.encode(.init(
            family: .qwen3_6,
            requiredFeatures: [.familyDispatch, .verifiedIdentity,
                               .qwenHybridAttention, .qwenMTPExcluded],
            modelID: GTurboFormatV2.qwenRepository,
            architecture: .qwen3_6(architecture),
            provenance: .init(
                sourceRepository: GTurboFormatV2.qwenRepository,
                sourceRevision: GTurboFormatV2.qwenRevision,
                sourceIndexSHA256: GTurboFormatV2.qwenSourceIndexSHA256,
                sidecarSHA256: sidecarSHA256,
                quantizationPolicySHA256: digest),
            quantization: groups, ignoredTensors: omissions,
            files: ["model_weights.bin": .init(size: 32_768, sha256: digest),
                    "packed_experts/layout.json": .init(size: 1, sha256: digest)],
            tensorRegions: [.init(name: "embed", file: "model_weights.bin", offset: 0,
                                  size: 16, shape: [1], storage: .affineInt4,
                                  quantizationCategory: .embedding)],
            expertsPerLayer: 256, numLayers: 40, expertStride: 16_384))
    }

    private func loadOutcome(at directory: URL) -> LoadOutcome {
        do {
            _ = try QwenTokenizer.load(from: directory)
            return .success
        } catch let error as QwenTokenizerError {
            return .qwen(error)
        } catch let error as ModelError {
            return .model(error)
        } catch {
            return .other(String(describing: error))
        }
    }

    private func qwenError(at directory: URL) -> QwenTokenizerError? {
        do {
            _ = try QwenTokenizer.loadOfficialSidecar(from: directory)
            return nil
        } catch let error as QwenTokenizerError {
            return error
        } catch {
            return nil
        }
    }
}
