import CryptoKit
import Foundation

/// Metadata identity for the original BF16 Safetensors source. This is not a
/// packed `.gturbo` manifest and does not attest to the presence or contents
/// of files at `sourceRoot`. Payload verification belongs to the source reader.
public struct OfficialSourceDescriptor: Encodable, Equatable, Sendable {
    public static let markerFilename = "official-source.json"
    public static let descriptorKind = "official-safetensors-bf16-v1"
    public static let descriptorVersion = 1
    public static let maximumMarkerBytes: UInt64 = 1024 * 1024

    public struct Shard: Encodable, Equatable, Sendable {
        public let filename: String
        public let sha256: String

        public init(filename: String, sha256: String) {
            self.filename = filename
            self.sha256 = sha256
        }
    }

    public let kind: String
    public let version: Int
    public let repository: String
    public let revision: String
    public let storageProfile: String
    public let sidecarSHA256: [String: String]
    public let shards: [Shard]
    public let sourceRoot: String
    public let contentSHA256: String

    private enum CodingKeys: String, CodingKey {
        case kind, version, repository, revision, storageProfile, sidecarSHA256
        case shards, sourceRoot, contentSHA256
    }

    /// Decoding is private to the duplicate-aware entry point. Neither wire
    /// type can be decoded by a caller to fabricate an admitted descriptor.
    private struct Wire: Decodable {
        let kind: String
        let version: Int
        let repository: String
        let revision: String
        let storageProfile: String
        let sidecarSHA256: [String: String]
        let shards: [WireShard]
        let sourceRoot: String
        let contentSHA256: String

        init(from decoder: Decoder) throws {
            let keys = try decoder.container(keyedBy: SourceJSONKey.self)
            guard Set(keys.allKeys.map(\.stringValue)) == Set([
                "kind", "version", "repository", "revision", "storageProfile",
                "sidecarSHA256", "shards", "sourceRoot", "contentSHA256",
            ]) else {
                throw OfficialSourceDescriptorError.invalid(field: "descriptor", reason: "unknown or missing field")
            }
            let values = try decoder.container(keyedBy: OfficialSourceDescriptor.CodingKeys.self)
            kind = try values.decode(String.self, forKey: .kind)
            version = try values.decode(Int.self, forKey: .version)
            repository = try values.decode(String.self, forKey: .repository)
            revision = try values.decode(String.self, forKey: .revision)
            storageProfile = try values.decode(String.self, forKey: .storageProfile)
            sidecarSHA256 = try values.decode([String: String].self, forKey: .sidecarSHA256)
            shards = try values.decode([WireShard].self, forKey: .shards)
            sourceRoot = try values.decode(String.self, forKey: .sourceRoot)
            contentSHA256 = try values.decode(String.self, forKey: .contentSHA256)
        }
    }

    private struct WireShard: Decodable {
        let filename: String
        let sha256: String

        private enum CodingKeys: String, CodingKey { case filename, sha256 }

        init(from decoder: Decoder) throws {
            let keys = try decoder.container(keyedBy: SourceJSONKey.self)
            guard Set(keys.allKeys.map(\.stringValue)) == Set(["filename", "sha256"]) else {
                throw OfficialSourceDescriptorError.invalid(field: "shards", reason: "unknown or missing field")
            }
            let values = try decoder.container(keyedBy: CodingKeys.self)
            filename = try values.decode(String.self, forKey: .filename)
            sha256 = try values.decode(String.self, forKey: .sha256)
        }
    }

    /// Sole public decode entry point. Ordinary `JSONDecoder` cannot report
    /// duplicate object members; this type intentionally does not conform to Decodable.
    /// The raw scan is O(bytes + total key bytes), with a 1 MiB input cap and
    /// bounded nesting; it never reads source files or changes the digest.
    public static func decodeStrict(data: Data) throws -> Self {
        guard UInt64(data.count) <= maximumMarkerBytes else {
            throw OfficialSourceDescriptorError.invalid(field: "descriptor.json", reason: "metadata cap exceeded")
        }
        var scanner = SourceJSONDuplicateScanner(data: data)
        try scanner.validate()
        let wire = try JSONDecoder().decode(Wire.self, from: data)
        return try Self(
            kind: wire.kind, version: wire.version,
            repository: wire.repository, revision: wire.revision,
            storageProfile: wire.storageProfile, sidecarSHA256: wire.sidecarSHA256,
            shards: wire.shards.map { Shard(filename: $0.filename, sha256: $0.sha256) },
            sourceRoot: wire.sourceRoot, contentSHA256: wire.contentSHA256)
    }

    /// Construct a descriptor with the canonical digest computed from content
    /// identity, independent of JSON, shard order, and physical source root.
    public init(repository: String, revision: String, storageProfile: String,
                sidecarSHA256: [String: String], shards: [Shard], sourceRoot: String) throws {
        let digest = try Self.digest(repository: repository, revision: revision,
                                     storageProfile: storageProfile,
                                     sidecarSHA256: sidecarSHA256, shards: shards)
        try self.init(kind: Self.descriptorKind, version: Self.descriptorVersion, repository: repository,
                      revision: revision, storageProfile: storageProfile,
                      sidecarSHA256: sidecarSHA256, shards: shards,
                      sourceRoot: sourceRoot, contentSHA256: digest)
    }

    public init(kind: String, version: Int, repository: String, revision: String,
                storageProfile: String, sidecarSHA256: [String: String],
                shards: [Shard], sourceRoot: String, contentSHA256: String) throws {
        guard kind == Self.descriptorKind, version == Self.descriptorVersion else {
            throw OfficialSourceDescriptorError.invalid(field: "kind/version", reason: "unsupported source descriptor")
        }
        let expected = try Self.digest(repository: repository, revision: revision,
                                        storageProfile: storageProfile,
                                        sidecarSHA256: sidecarSHA256, shards: shards)
        try Self.checkSHA(contentSHA256, field: "contentSHA256")
        guard contentSHA256 == expected else {
            throw OfficialSourceDescriptorError.invalid(field: "contentSHA256", reason: "content identity mismatch")
        }
        try Self.checkRoot(sourceRoot)
        self.kind = kind
        self.version = version
        self.repository = repository
        self.revision = revision
        self.storageProfile = storageProfile
        self.sidecarSHA256 = sidecarSHA256
        self.shards = shards
        self.sourceRoot = sourceRoot
        self.contentSHA256 = contentSHA256
    }

    /// SHA-256 of UTF-8 lines, sorted by filename within each file category.
    /// Validation precedes sorting, so duplicate shards cannot disappear.
    public static func digest(repository: String, revision: String,
                              storageProfile: String, sidecarSHA256: [String: String],
                              shards: [Shard]) throws -> String {
        try checkScalar(repository, field: "repository")
        try checkScalar(revision, field: "revision")
        try checkScalar(storageProfile, field: "storageProfile")
        guard !sidecarSHA256.isEmpty, !shards.isEmpty else {
            throw OfficialSourceDescriptorError.invalid(field: "files", reason: "empty inventory")
        }
        var names = Set<String>()
        for (name, hash) in sidecarSHA256 {
            try checkName(name, field: "sidecarSHA256")
            try checkSHA(hash, field: "sidecarSHA256.\(name)")
            guard names.insert(GTurboPathValidator.appleFilesystemKey(name)).inserted else {
                throw OfficialSourceDescriptorError.invalid(field: "sidecarSHA256", reason: "duplicate filename")
            }
        }
        for shard in shards {
            try checkName(shard.filename, field: "shards.filename")
            try checkSHA(shard.sha256, field: "shards.sha256")
            guard names.insert(GTurboPathValidator.appleFilesystemKey(shard.filename)).inserted else {
                throw OfficialSourceDescriptorError.invalid(field: "shards", reason: "duplicate filename")
            }
        }
        var lines = "\(Self.descriptorKind)\nrepository=\(repository)\nrevision=\(revision)\nstorageProfile=\(storageProfile)\n"
        for name in sidecarSHA256.keys.sorted() {
            if let hash = sidecarSHA256[name] {
                lines += "sidecar:\(name)=\(hash)\n"
            }
        }
        for shard in shards.sorted(by: { $0.filename < $1.filename }) {
            lines += "shard:\(shard.filename)=\(shard.sha256)\n"
        }
        let hash = SHA256.hash(data: Data(lines.utf8))
        return hash.map { String(format: "%02x", $0) }.joined()
    }

    private static func checkScalar(_ value: String, field: String) throws {
        guard !value.isEmpty, !value.contains("\r"), !value.contains("\n"),
              !value.contains("="), !value.contains("\0") else {
            throw OfficialSourceDescriptorError.invalid(field: field, reason: "unsafe digest field")
        }
    }

    private static func checkName(_ value: String, field: String) throws {
        try checkScalar(value, field: field)
        do { try GTurboPathValidator.validateBasename(value, field: field) }
        catch { throw OfficialSourceDescriptorError.invalid(field: field, reason: "unsafe filename") }
    }

    private static func checkSHA(_ value: String, field: String) throws {
        guard value.utf8.count == 64,
              value.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw OfficialSourceDescriptorError.invalid(field: field, reason: "expected lowercase SHA-256")
        }
    }

    private static func checkRoot(_ path: String) throws {
        guard path.hasPrefix("/"), path != "/", !path.contains("\0"),
              path.dropFirst().split(separator: "/", omittingEmptySubsequences: false)
                .allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw OfficialSourceDescriptorError.invalid(field: "sourceRoot", reason: "unsafe absolute path")
        }
    }
}

private struct SourceJSONKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

public enum OfficialSourceDescriptorError: Error, Equatable, Sendable {
    case invalid(field: String, reason: String)
}

/// A bounded structural pass over raw JSON, before JSONDecoder collapses object
/// keys. Decode each raw key token so `"kind"` and `"\\u006bind"` compare equal.
package struct SourceJSONDuplicateScanner {
    private let bytes: [UInt8]
    private var offset = 0
    private let keyDecoder = JSONDecoder()
    private static let maximumDepth = 64

    package init(data: Data) { bytes = Array(data) }

    package mutating func validate() throws {
        skipWhitespace()
        try parseValue(depth: 0)
        skipWhitespace()
        guard offset == bytes.count else { throw invalidJSON() }
    }

    private mutating func parseValue(depth: Int) throws {
        guard depth <= Self.maximumDepth, offset < bytes.count else { throw invalidJSON() }
        switch bytes[offset] {
        case 0x7b: try parseObject(depth: depth)
        case 0x5b: try parseArray(depth: depth)
        case 0x22: _ = try scanString()
        case 0x74: try consume([0x74, 0x72, 0x75, 0x65]) // true
        case 0x66: try consume([0x66, 0x61, 0x6c, 0x73, 0x65]) // false
        case 0x6e: try consume([0x6e, 0x75, 0x6c, 0x6c]) // null
        case 0x2d, 0x30...0x39: try parseNumber()
        default: throw invalidJSON()
        }
    }

    private mutating func parseObject(depth: Int) throws {
        offset += 1 // {
        skipWhitespace()
        if take(0x7d) { return }
        var seen = Set<String>()
        while true {
            let keyRange = try scanString()
            let key = try keyDecoder.decode(String.self, from: Data(bytes[keyRange]))
            guard seen.insert(key).inserted else {
                throw OfficialSourceDescriptorError.invalid(
                    field: "descriptor.json", reason: "duplicate JSON object key")
            }
            skipWhitespace()
            guard take(0x3a) else { throw invalidJSON() } // :
            skipWhitespace()
            try parseValue(depth: depth + 1)
            skipWhitespace()
            if take(0x7d) { return }
            guard take(0x2c) else { throw invalidJSON() } // ,
            skipWhitespace()
        }
    }

    private mutating func parseArray(depth: Int) throws {
        offset += 1 // [
        skipWhitespace()
        if take(0x5d) { return }
        while true {
            try parseValue(depth: depth + 1)
            skipWhitespace()
            if take(0x5d) { return }
            guard take(0x2c) else { throw invalidJSON() } // ,
            skipWhitespace()
        }
    }

    /// Returns the complete quoted token, retaining escapes for JSONDecoder.
    private mutating func scanString() throws -> Range<Int> {
        guard take(0x22) else { throw invalidJSON() }
        let start = offset - 1
        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            if byte == 0x22 { return start..<offset }
            guard byte >= 0x20 else { throw invalidJSON() }
            if byte == 0x5c { // backslash
                guard offset < bytes.count else { throw invalidJSON() }
                let escape = bytes[offset]
                offset += 1
                if escape == 0x75 { // u + exactly four hexadecimal digits
                    for _ in 0..<4 {
                        guard offset < bytes.count, Self.isHex(bytes[offset]) else {
                            throw invalidJSON()
                        }
                        offset += 1
                    }
                } else if ![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].contains(escape) {
                    throw invalidJSON()
                }
            }
        }
        throw invalidJSON()
    }

    private mutating func parseNumber() throws {
        _ = take(0x2d) // optional -
        guard offset < bytes.count else { throw invalidJSON() }
        if !take(0x30) {
            guard (0x31...0x39).contains(bytes[offset]) else { throw invalidJSON() }
            consumeDigits()
        }
        if take(0x2e) { // .
            let start = offset
            consumeDigits()
            guard offset > start else { throw invalidJSON() }
        }
        if take(0x65) || take(0x45) { // e or E
            if !take(0x2b) { _ = take(0x2d) } // optional sign
            let start = offset
            consumeDigits()
            guard offset > start else { throw invalidJSON() }
        }
    }

    private mutating func consumeDigits() {
        while offset < bytes.count, (0x30...0x39).contains(bytes[offset]) { offset += 1 }
    }

    private mutating func consume(_ literal: [UInt8]) throws {
        for byte in literal {
            guard take(byte) else { throw invalidJSON() }
        }
    }

    private static func isHex(_ byte: UInt8) -> Bool {
        (0x30...0x39).contains(byte) || (0x41...0x46).contains(byte)
            || (0x61...0x66).contains(byte)
    }

    private mutating func skipWhitespace() {
        while offset < bytes.count,
              [0x20, 0x09, 0x0a, 0x0d].contains(bytes[offset]) { offset += 1 }
    }

    private mutating func take(_ byte: UInt8) -> Bool {
        guard offset < bytes.count, bytes[offset] == byte else { return false }
        offset += 1
        return true
    }

    private func invalidJSON() -> OfficialSourceDescriptorError {
        .invalid(field: "descriptor.json", reason: "invalid JSON structure or nesting")
    }
}
