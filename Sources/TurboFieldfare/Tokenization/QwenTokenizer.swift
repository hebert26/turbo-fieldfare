import CryptoKit
import Darwin
import Foundation
import Hub
import Tokenizers

public enum QwenTokenizerError: Error, Equatable, CustomStringConvertible {
    case missingFile(String)
    case unreadableFile(String)
    case fileTooLarge(file: String, maximumBytes: UInt64)
    case checksumMismatch(file: String, expected: String, actual: String)
    case malformedJSON(String)
    case invalidStructure(String)
    case tokenMismatch(token: String, expected: Int32, actual: Int?)

    public var description: String {
        switch self {
        case .missingFile(let file):
            return "missing required Qwen tokenizer file: \(file)"
        case .unreadableFile(let file):
            return "cannot read required Qwen tokenizer file: \(file)"
        case .fileTooLarge(let file, let maximumBytes):
            return "required Qwen tokenizer file \(file) exceeds \(maximumBytes) bytes"
        case .checksumMismatch(let file, let expected, let actual):
            return "Qwen tokenizer checksum mismatch for \(file): expected \(expected), got \(actual)"
        case .malformedJSON(let file):
            return "malformed JSON in required Qwen tokenizer file: \(file)"
        case .invalidStructure(let detail):
            return "invalid pinned Qwen tokenizer structure: \(detail)"
        case .tokenMismatch(let token, let expected, let actual):
            return "Qwen token \(token) must resolve to \(expected), got \(String(describing: actual))"
        }
    }
}

public struct QwenAddedToken: Equatable, Sendable {
    public let id: Int32
    public let content: String
    public let isSpecial: Bool

    public init(id: Int32, content: String, isSpecial: Bool) {
        self.id = id
        self.content = content
        self.isSpecial = isSpecial
    }
}

/// Exact tokenizer for the pinned Qwen3.6-35B-A3B revision.
///
/// Installed loading first admits a verified Qwen v2 `.gturbo` manifest, then
/// reads exactly `tokenizer/tokenizer.json` and
/// `tokenizer/tokenizer_config.json` through the retained model-directory file
/// descriptor. Both byte digests and their behavior-defining structure are
/// checked before constructing swift-transformers' BPE engine. External vocab,
/// merges, Jinja, model, and generation sidecars are inert.
public struct QwenTokenizer: Sendable {
    public static let baseVocabularySize = 248_044
    public static let tokenizerLength = 248_077
    public static let modelVocabularySize = 248_320
    public static let mergeCount = 247_587

    public static let tokenizerJSONSHA256 =
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    public static let tokenizerConfigJSONSHA256 =
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    public static let embeddedChatTemplateSHA256 =
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"

    private static let tokenizerJSONMaximumBytes: UInt64 = 16 * 1024 * 1024
    private static let tokenizerConfigJSONMaximumBytes: UInt64 = 256 * 1024

    public let eosID: Int32 = 248_046
    public let padID: Int32 = 248_044
    public let addedTokens: [QwenAddedToken]
    /// All 21 `special:true` declarations used by decode filtering. This is
    /// deliberately broader than Transformers' nine-entry `all_special_ids`
    /// convenience property, which is not the decoder's complete skip set.
    public let effectiveSpecialTokenIDs: Set<Int32>

    let tokenizer: any Tokenizer
    private let addedTokenIDs: Set<Int32>

    /// Performs blocking local file I/O and one-time BPE construction. Callers
    /// must invoke it away from UI isolation, then freely share the immutable,
    /// `Sendable` result for deterministic concurrent encode/decode operations.
    /// Manifest admission failures remain `ModelError`; failures while reading
    /// the two fixed tokenizer assets are translated to `QwenTokenizerError`.
    public static func load(from folder: URL) throws -> Self {
        let directory = try GTurboModelDirectory(rootURL: folder)
        let manifestData: Data
        do {
            manifestData = try directory.readMetadata(
                "manifest.json", maxBytes: ManifestReader.defaultMaxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: folder.path)
        }
        let manifest = try ManifestReader.decodeVerified(data: manifestData)
        guard manifest.descriptor.family == .qwen3_6,
              case .qwen3_6 = manifest.architecture else {
            throw ModelError.indexCorrupt(
                detail: "v2 tokenizer admission accepts only verified Qwen 3.6")
        }
        return try load(
            from: directory,
            tokenizerPath: "tokenizer/tokenizer.json",
            configPath: "tokenizer/tokenizer_config.json")
    }

    /// Test-only seam for authentic copied tokenizer sidecars. Unlike public
    /// installed loading, it deliberately performs no `.gturbo` admission.
    static func loadOfficialSidecar(from folder: URL) throws -> Self {
        let directory: GTurboModelDirectory
        do {
            directory = try GTurboModelDirectory(rootURL: folder)
        } catch {
            throw QwenTokenizerError.unreadableFile(folder.path)
        }
        return try load(
            from: directory,
            tokenizerPath: "tokenizer.json",
            configPath: "tokenizer_config.json")
    }

    private static func load(
        from directory: GTurboModelDirectory,
        tokenizerPath: String,
        configPath: String
    ) throws -> Self {
        let tokenizerBytes = try readRequired(
            from: directory,
            relativePath: tokenizerPath,
            maximumBytes: tokenizerJSONMaximumBytes)
        let configBytes = try readRequired(
            from: directory,
            relativePath: configPath,
            maximumBytes: tokenizerConfigJSONMaximumBytes)
        try verifyDigest(
            tokenizerBytes, name: tokenizerPath, expected: tokenizerJSONSHA256)
        try verifyDigest(
            configBytes, name: configPath, expected: tokenizerConfigJSONSHA256)

        let tokenizerData = try parseConfig(tokenizerBytes, name: tokenizerPath)
        let tokenizerConfig = try parseConfig(configBytes, name: configPath)
        let declarations = try validateAndMerge(
            tokenizerData: tokenizerData, tokenizerConfig: tokenizerConfig)
        let underlying = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: declarations.mergedTokenizerData,
            strict: true)

        for token in declarations.tokens {
            let actual = underlying.convertTokenToId(token.content)
            guard actual == Int(token.id),
                  underlying.convertIdToToken(Int(token.id)) == token.content else {
                throw QwenTokenizerError.tokenMismatch(
                    token: token.content, expected: token.id, actual: actual)
            }
        }

        return Self(
            tokenizer: underlying,
            addedTokens: declarations.tokens,
            effectiveSpecialTokenIDs: Set(
                declarations.tokens.lazy.filter(\.isSpecial).map(\.id)))
    }

    init(
        tokenizer: any Tokenizer,
        addedTokens: [QwenAddedToken],
        effectiveSpecialTokenIDs: Set<Int32>
    ) {
        self.tokenizer = tokenizer
        self.addedTokens = addedTokens
        self.effectiveSpecialTokenIDs = effectiveSpecialTokenIDs
        self.addedTokenIDs = Set(addedTokens.map(\.id))
    }

    /// Encodes plain text without adding BOS/EOS tokens. Added-token literals,
    /// including all seven config-only audio declarations, remain available.
    public func encode(_ text: String) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// ByteLevel decode with pinned Rust-tokenizers semantics for unknown IDs,
    /// special-token filtering, invalid UTF-8 replacement, and added-token
    /// boundaries. Batch decode is the same push/finish path used by streaming.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var decoder = makeIncrementalDecoder(skipSpecialTokens: skipSpecialTokens)
        var output = ""
        for id in ids {
            output += decoder.push(id)
        }
        return output + decoder.finish()
    }

    public func makeIncrementalDecoder(
        skipSpecialTokens: Bool = true
    ) -> QwenIncrementalDecoder {
        QwenIncrementalDecoder(
            tokenizer: tokenizer,
            addedTokenIDs: addedTokenIDs,
            specialTokenIDs: effectiveSpecialTokenIDs,
            skipSpecialTokens: skipSpecialTokens)
    }

    private struct ValidatedDeclarations {
        let tokens: [QwenAddedToken]
        let mergedTokenizerData: Config
    }

    private static func readRequired(
        from directory: GTurboModelDirectory,
        relativePath: String,
        maximumBytes: UInt64
    ) throws -> Data {
        let fileDescriptor: Int32
        do {
            fileDescriptor = try directory.openFile(relativePath)
        } catch ModelError.missingFile {
            throw QwenTokenizerError.missingFile(relativePath)
        } catch {
            throw QwenTokenizerError.unreadableFile(relativePath)
        }
        defer { Darwin.close(fileDescriptor) }

        let size: UInt64
        do {
            size = try directory.fileSize(
                fileDescriptor: fileDescriptor, relativePath: relativePath)
        } catch {
            throw QwenTokenizerError.unreadableFile(relativePath)
        }
        guard size <= maximumBytes else {
            throw QwenTokenizerError.fileTooLarge(
                file: relativePath, maximumBytes: maximumBytes)
        }

        do {
            return try directory.readMetadata(
                fileDescriptor: fileDescriptor,
                relativePath: relativePath,
                maxBytes: maximumBytes)
        } catch {
            throw QwenTokenizerError.unreadableFile(relativePath)
        }
    }

    private static func verifyDigest(_ data: Data, name: String, expected: String) throws {
        let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard actual == expected else {
            throw QwenTokenizerError.checksumMismatch(
                file: name, expected: expected, actual: actual)
        }
    }

    private static func parseConfig(_ data: Data, name: String) throws -> Config {
        do {
            guard let dictionary = try JSONSerialization.jsonObject(with: data)
                as? [NSString: Any] else {
                throw QwenTokenizerError.malformedJSON(name)
            }
            return Config(dictionary)
        } catch let error as QwenTokenizerError {
            throw error
        } catch {
            throw QwenTokenizerError.malformedJSON(name)
        }
    }

    private static func validateAndMerge(
        tokenizerData: Config,
        tokenizerConfig: Config
    ) throws -> ValidatedDeclarations {
        guard tokenizerConfig["tokenizerClass"].string() == "Qwen2Tokenizer" else {
            throw QwenTokenizerError.invalidStructure("tokenizer_class")
        }
        let eosToken = tokenizerConfig["eosToken"]
        let padToken = tokenizerConfig["padToken"]
        guard tokenizerConfig["modelMaxLength"].integer() == 262_144,
              tokenizerConfig["cleanUpTokenizationSpaces"].boolean() == false,
              (eosToken.string() ?? eosToken["content"].string()) == "<|im_end|>",
              (padToken.string() ?? padToken["content"].string()) == "<|endoftext|>" else {
            throw QwenTokenizerError.invalidStructure("tokenizer special-token metadata")
        }
        guard let template = tokenizerConfig["chatTemplate"].string(),
              SHA256.hash(data: Data(template.utf8))
                .map({ String(format: "%02x", $0) }).joined()
                == embeddedChatTemplateSHA256 else {
            throw QwenTokenizerError.invalidStructure("embedded chat template identity")
        }

        let model: Config = tokenizerData["model"]
        guard model["type"].string() == "BPE",
              model["vocab"].dictionary(or: [:]).count == baseVocabularySize,
              model["merges"].array(or: []).count == mergeCount,
              model["byteFallback"].boolean() == false,
              model["ignoreMerges"].boolean() == false,
              tokenizerData["normalizer"]["type"].string() == "NFC",
              tokenizerData["decoder"]["type"].string() == "ByteLevel",
              tokenizerData["postProcessor"]["type"].string() == "ByteLevel",
              tokenizerData["preTokenizer"]["type"].string() == "Sequence" else {
            throw QwenTokenizerError.invalidStructure("BPE/ByteLevel geometry")
        }

        let expected = expectedAddedTokens
        let configDecoder = tokenizerConfig["addedTokensDecoder"].dictionary(or: [:])
        guard configDecoder.count == expected.count else {
            throw QwenTokenizerError.invalidStructure("added_tokens_decoder count")
        }

        var configRows: [Int32: Config] = [:]
        for (key, declaration) in configDecoder {
            guard let rawID = Int(key.string), let id = Int32(exactly: rawID),
                  let expectedToken = expected.first(where: { $0.id == id }),
                  declaration["content"].string() == expectedToken.content,
                  declaration["special"].boolean() == expectedToken.isSpecial,
                  declaration["singleWord"].boolean() == false,
                  declaration["lstrip"].boolean() == false,
                  declaration["rstrip"].boolean() == false,
                  declaration["normalized"].boolean() == false else {
                throw QwenTokenizerError.invalidStructure(
                    "added_tokens_decoder declaration \(key.string)")
            }
            configRows[id] = declaration
        }

        let tokenizerRows = tokenizerData["addedTokens"].array(or: [])
        guard tokenizerRows.count == 26 else {
            throw QwenTokenizerError.invalidStructure("tokenizer.json added_tokens count")
        }
        var presentIDs: Set<Int32> = []
        for row in tokenizerRows {
            guard let rawID = row["id"].integer(), let id = Int32(exactly: rawID),
                  let expectedToken = expected.first(where: { $0.id == id }),
                  id <= 248_069,
                  row["content"].string() == expectedToken.content,
                  row["special"].boolean() == expectedToken.isSpecial else {
                throw QwenTokenizerError.invalidStructure("tokenizer.json added token")
            }
            guard presentIDs.insert(id).inserted else {
                throw QwenTokenizerError.invalidStructure("duplicate tokenizer.json added ID")
            }
        }
        guard presentIDs == Set(Int32(248_044)...Int32(248_069)) else {
            throw QwenTokenizerError.invalidStructure("tokenizer.json added ID range")
        }

        var mergedRows = tokenizerRows
        for id in Int32(248_070)...Int32(248_076) {
            guard let declaration = configRows[id], var row = declaration.dictionary() else {
                throw QwenTokenizerError.invalidStructure("missing config-only token \(id)")
            }
            row["id"] = Config(Int(id))
            mergedRows.append(Config(row))
        }
        guard mergedRows.count + baseVocabularySize == tokenizerLength else {
            throw QwenTokenizerError.invalidStructure("total tokenizer length")
        }

        guard var tokenizerDictionary = tokenizerData.dictionary() else {
            throw QwenTokenizerError.invalidStructure("tokenizer.json root")
        }
        tokenizerDictionary["added_tokens"] = Config(mergedRows)
        return ValidatedDeclarations(
            tokens: expected, mergedTokenizerData: Config(tokenizerDictionary))
    }

    private static let expectedAddedTokens: [QwenAddedToken] = {
        let contents = [
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
            "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
            "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
            "<tool_response>", "</tool_response>", "<think>", "</think>",
            "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
            "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
            "<|audio_pad|>",
        ]
        return contents.enumerated().map { offset, content in
            let id = Int32(248_044 + offset)
            return QwenAddedToken(
                id: id,
                content: content,
                isSpecial: id <= 248_057 || id >= 248_070)
        }
    }()
}

/// Append-only ByteLevel decoder. A valid but incomplete UTF-8 suffix is held
/// across skipped special and unknown IDs, then emitted only when completed or
/// replaced at `finish()`. Added tokens kept in output are decode boundaries.
/// Each value owns one stream's pending UTF-8 suffix. Do not share a mutable
/// instance between streams; make one decoder per generated sequence.
public struct QwenIncrementalDecoder: Sendable {
    private let tokenizer: any Tokenizer
    private let addedTokenIDs: Set<Int32>
    private let specialTokenIDs: Set<Int32>
    private let skipSpecialTokens: Bool
    private var pendingBytes: [UInt8] = []

    init(
        tokenizer: any Tokenizer,
        addedTokenIDs: Set<Int32>,
        specialTokenIDs: Set<Int32>,
        skipSpecialTokens: Bool
    ) {
        self.tokenizer = tokenizer
        self.addedTokenIDs = addedTokenIDs
        self.specialTokenIDs = specialTokenIDs
        self.skipSpecialTokens = skipSpecialTokens
    }

    public mutating func push(_ id: Int32) -> String {
        // Reference decode drops unresolved IDs without closing the current
        // ByteLevel subtext. This includes padded logits IDs 248077...248319.
        guard let token = tokenizer.convertIdToToken(Int(id)) else { return "" }
        if skipSpecialTokens, specialTokenIDs.contains(id) { return "" }
        if addedTokenIDs.contains(id) {
            return commitPending() + token
        }

        for scalar in token.unicodeScalars {
            guard let byte = Self.byteDecoder[scalar] else {
                return commitPending() + token
            }
            pendingBytes.append(byte)
        }
        return emitCompletePrefix()
    }

    public mutating func finish() -> String {
        commitPending()
    }

    private mutating func emitCompletePrefix() -> String {
        let held = trailingIncompleteByteCount(pendingBytes)
        let emittedCount = pendingBytes.count - held
        guard emittedCount > 0 else { return "" }
        let output = String(decoding: pendingBytes.prefix(emittedCount), as: UTF8.self)
        pendingBytes.removeFirst(emittedCount)
        return output
    }

    private mutating func commitPending() -> String {
        defer { pendingBytes.removeAll(keepingCapacity: true) }
        return String(decoding: pendingBytes, as: UTF8.self)
    }

    private func trailingIncompleteByteCount(_ bytes: [UInt8]) -> Int {
        guard !bytes.isEmpty else { return 0 }
        let lower = max(0, bytes.count - 3)
        for start in stride(from: bytes.count - 1, through: lower, by: -1) {
            let lead = bytes[start]
            let required: Int
            switch lead {
            case 0xC2...0xDF: required = 2
            case 0xE0...0xEF: required = 3
            case 0xF0...0xF4: required = 4
            default: continue
            }
            let count = bytes.count - start
            guard count < required else { continue }
            let suffix = bytes[start...]
            guard validIncompletePrefix(Array(suffix), required: required) else { continue }
            return count
        }
        return 0
    }

    private func validIncompletePrefix(_ bytes: [UInt8], required: Int) -> Bool {
        guard let lead = bytes.first, bytes.count < required else { return false }
        for (offset, byte) in bytes.dropFirst().enumerated() {
            let position = offset + 1
            let allowed: ClosedRange<UInt8>
            if position == 1 {
                switch lead {
                case 0xE0: allowed = 0xA0...0xBF
                case 0xED: allowed = 0x80...0x9F
                case 0xF0: allowed = 0x90...0xBF
                case 0xF4: allowed = 0x80...0x8F
                default: allowed = 0x80...0xBF
                }
            } else {
                allowed = 0x80...0xBF
            }
            guard allowed.contains(byte) else { return false }
        }
        return true
    }

    private static let byteDecoder: [Unicode.Scalar: UInt8] = {
        var bytes = Array(UInt8(33)...UInt8(126))
        bytes += Array(UInt8(161)...UInt8(172))
        bytes += Array(UInt8(174)...UInt8(255))
        var codepoints = bytes.map(Int.init)
        var extra = 0
        for byte in UInt8.min...UInt8.max where !bytes.contains(byte) {
            bytes.append(byte)
            codepoints.append(256 + extra)
            extra += 1
        }
        return Dictionary(uniqueKeysWithValues: zip(codepoints, bytes).compactMap { pair in
            Unicode.Scalar(pair.0).map { ($0, pair.1) }
        })
    }()
}
