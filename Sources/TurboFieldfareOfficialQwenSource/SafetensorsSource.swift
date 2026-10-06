import Foundation

/// Safetensors metadata only. Neither parsing nor layout validation reads or
/// authenticates tensor payloads. The caller owns the file descriptor and its
/// lifetime; `readAt` is called for only the prefix and bounded header.
/// Parsing uses O(metadata bytes + tensor count) space; range validation sorts
/// tensor coordinates in O(tensor count log tensor count) time.
public enum OfficialSafetensorsSource {
    public static let maximumHeaderBytes: UInt64 = 16 * 1024 * 1024
    public static let maximumIndexBytes: UInt64 = 4 * 1024 * 1024

    public struct Tensor: Sendable, Equatable {
        public let name: String
        public let dtype: String
        public let shape: [UInt64]
        public let absoluteOffset: UInt64
        public let sizeBytes: UInt64
    }

    public struct Header: Sendable, Equatable {
        public let path: String
        public let payloadBaseOffset: UInt64
        public let tensors: [Tensor]
    }

    public enum ValidationError: Error, Sendable, Equatable {
        case headerTooLarge(size: UInt64)
        case invalidHeader(detail: String)
        case unknownDtype(String)
        case tensorOutOfRange(name: String, end: UInt64, fileSize: UInt64)
        case shapeMismatch(name: String, detail: String)
        case invalidIndex(detail: String)
        case shortRead(expected: Int, got: Int)
    }

    /// Reads exactly two bounded metadata ranges from a caller-owned retained
    /// descriptor. The caller closes that descriptor after this synchronous
    /// call. Short reads fail; an I/O error from the closure propagates.
    public static func readHeader(
        path: String, fileSize: UInt64,
        readAt: (_ offset: UInt64, _ count: Int) throws -> Data
    ) throws -> Header {
        guard fileSize >= 8 else {
            throw ValidationError.invalidHeader(detail: "file too short")
        }
        let prefix = try readAt(0, 8)
        guard prefix.count == 8 else {
            throw ValidationError.shortRead(expected: 8, got: prefix.count)
        }
        let length = prefix.enumerated().reduce(UInt64(0)) { result, entry in
            result | (UInt64(entry.element) << (entry.offset * 8))
        }
        guard length <= maximumHeaderBytes, length <= fileSize - 8 else {
            throw ValidationError.headerTooLarge(size: length)
        }
        let bytes = try readAt(8, Int(length))
        guard bytes.count == Int(length) else {
            throw ValidationError.shortRead(expected: Int(length), got: bytes.count)
        }
        return try parseHeader(path: path, fileSize: fileSize, headerBytes: bytes)
    }

    /// Parses an already bounded remote or local header. All coordinates in
    /// the result are absolute file offsets, suitable for later range planning.
    public static func parseHeader(path: String, fileSize: UInt64,
                                   headerBytes: Data) throws -> Header {
        guard fileSize >= 8 else {
            throw ValidationError.invalidHeader(detail: "file too short")
        }
        let length = UInt64(headerBytes.count)
        guard length <= maximumHeaderBytes, length <= fileSize - 8 else {
            throw ValidationError.headerTooLarge(size: length)
        }
        let root: JSONValue
        do {
            var scanner = StrictMetadataJSON(data: headerBytes)
            root = try scanner.parse()
        }
        catch { throw ValidationError.invalidHeader(detail: "invalid or duplicate JSON: \(error)") }
        guard case .object(let entries) = root else {
            throw ValidationError.invalidHeader(detail: "header is not a JSON object")
        }
        let payloadBase = 8 + length
        var tensors: [Tensor] = []
        tensors.reserveCapacity(entries.count)
        for (name, value) in entries.sorted(by: { $0.key < $1.key }) {
            if name == "__metadata__" { continue }
            guard !name.isEmpty, case .object(let entry) = value else {
                throw ValidationError.invalidHeader(detail: "entry for \(name) is not a dict")
            }
            guard let dtypeValue = entry["dtype"], case .string(let dtype) = dtypeValue else {
                throw ValidationError.invalidHeader(detail: "entry for \(name) has no dtype")
            }
            let elementBytes: UInt64
            switch dtype {
            case "U32", "F32": elementBytes = 4
            case "BF16", "F16": elementBytes = 2
            default: throw ValidationError.unknownDtype(dtype)
            }
            guard let shapeValue = entry["shape"], case .array(let shapeValues) = shapeValue else {
                throw ValidationError.invalidHeader(detail: "entry for \(name) has no shape")
            }
            var shape: [UInt64] = []
            shape.reserveCapacity(shapeValues.count)
            for value in shapeValues {
                guard let extent = value.unsignedInteger else {
                    throw ValidationError.invalidHeader(detail: "entry for \(name) has non-integer shape entry")
                }
                shape.append(extent)
            }
            guard let offsetValue = entry["data_offsets"],
                  case .array(let offsets) = offsetValue, offsets.count == 2,
                  let begin = offsets[0].unsignedInteger,
                  let end = offsets[1].unsignedInteger else {
                throw ValidationError.invalidHeader(detail: "entry for \(name) has bad data_offsets")
            }
            guard end >= begin else {
                throw ValidationError.invalidHeader(
                    detail: "entry for \(name) has data_offsets end \(end) before begin \(begin)")
            }
            let size = end - begin
            let absolute = try checked(payloadBase.addingReportingOverflow(begin),
                                       name: name, field: "absolute offset")
            let absoluteEnd = try checked(absolute.addingReportingOverflow(size),
                                          name: name, field: "absolute end")
            guard absoluteEnd <= fileSize else {
                throw ValidationError.tensorOutOfRange(name: name, end: absoluteEnd, fileSize: fileSize)
            }
            var elements: UInt64 = 1
            for extent in shape {
                elements = try checked(elements.multipliedReportingOverflow(by: extent),
                                       name: name, field: "shape product")
            }
            let declared = try checked(elements.multipliedReportingOverflow(by: elementBytes),
                                       name: name, field: "element bytes")
            guard declared == size else {
                throw ValidationError.shapeMismatch(
                    name: name, detail: "shape product \(elements)*\(elementBytes) != size \(size)")
            }
            tensors.append(Tensor(name: name, dtype: dtype, shape: shape,
                                  absoluteOffset: absolute, sizeBytes: size))
        }
        let header = Header(path: path, payloadBaseOffset: payloadBase, tensors: tensors)
        try validateContiguousLayout(header, fileSize: fileSize)
        return header
    }

    /// Strict, bounded index parsing. The map's values must be safe shard leaf
    /// names; no shard is opened here. Other index metadata is not a trust pin.
    public static func parseIndex(_ data: Data) throws -> [String: String] {
        guard UInt64(data.count) <= maximumIndexBytes else {
            throw ValidationError.invalidIndex(detail: "index exceeds metadata cap")
        }
        let root: JSONValue
        do {
            var scanner = StrictMetadataJSON(data: data)
            root = try scanner.parse()
        }
        catch { throw ValidationError.invalidIndex(detail: "invalid or duplicate JSON: \(error)") }
        guard case .object(let fields) = root,
              let mapValue = fields["weight_map"],
              case .object(let weightMap) = mapValue, !weightMap.isEmpty else {
            throw ValidationError.invalidIndex(detail: "no weight_map")
        }
        var result: [String: String] = [:]
        result.reserveCapacity(weightMap.count)
        for (name, value) in weightMap {
            guard !name.isEmpty, case .string(let shard) = value,
                  isSafeShardName(shard) else {
                throw ValidationError.invalidIndex(detail: "unsafe shard name or tensor entry \(name)")
            }
            result[name] = shard
        }
        return result
    }

    /// Requires the exact tensor names and assignment for this one shard.
    /// Generic dtype support is intentional; a BF16-only policy belongs to
    /// the local Qwen adapter, not this shared parser.
    public static func validateShard(_ header: Header, weightMap: [String: String],
                                     shardName: String) throws {
        let expected = Set(weightMap.compactMap { $0.value == shardName ? $0.key : nil })
        let actual = Set(header.tensors.map(\.name))
        guard expected == actual, actual.count == header.tensors.count else {
            throw ValidationError.invalidIndex(detail: "tensor names disagree for \(shardName)")
        }
    }

    private static func isSafeShardName(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".." && value.hasSuffix(".safetensors")
            && !value.contains("/") && !value.contains("\\") && !value.contains("\0")
    }

    /// Safetensors offsets must partition the entire payload, not merely stay
    /// in bounds. Empty tensors are valid only at the current boundary: sorting
    /// them before nonempty tensors at the same offset makes this deterministic.
    /// An empty tensor set is valid only when the file has no payload bytes.
    private static func validateContiguousLayout(_ header: Header, fileSize: UInt64) throws {
        let sorted = header.tensors.sorted {
            if $0.absoluteOffset != $1.absoluteOffset {
                return $0.absoluteOffset < $1.absoluteOffset
            }
            if $0.sizeBytes != $1.sizeBytes { return $0.sizeBytes < $1.sizeBytes }
            return $0.name < $1.name
        }
        var cursor = header.payloadBaseOffset
        var previousName: String?
        for tensor in sorted {
            guard tensor.absoluteOffset == cursor else {
                if tensor.absoluteOffset < cursor {
                    throw ValidationError.invalidHeader(
                        detail: "tensor ranges overlap: \(previousName ?? tensor.name), \(tensor.name)")
                }
                throw ValidationError.invalidHeader(detail: "payload gap before \(tensor.name)")
            }
            let end = cursor.addingReportingOverflow(tensor.sizeBytes)
            guard !end.overflow, end.partialValue <= fileSize else {
                throw ValidationError.invalidHeader(detail: "entry for \(tensor.name) overflows layout")
            }
            cursor = end.partialValue
            previousName = tensor.name
        }
        guard cursor == fileSize else {
            throw ValidationError.invalidHeader(detail: "trailing payload gap")
        }
    }

    private static func checked(_ result: (partialValue: UInt64, overflow: Bool),
                                name: String, field: String) throws -> UInt64 {
        guard !result.overflow else {
            throw ValidationError.invalidHeader(detail: "entry for \(name) overflows \(field)")
        }
        return result.partialValue
    }
}

private enum JSONValue {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case number(String)
    case other

    /// Only decimal integer tokens in the exact UInt64 range are accepted;
    /// booleans, negative values, fractions and exponents cannot be coerced.
    var unsignedInteger: UInt64? {
        guard case .number(let token) = self, !token.isEmpty,
              token.utf8.allSatisfy({ (48...57).contains($0) }) else { return nil }
        return UInt64(token)
    }
}

/// A bounded structural JSON reader which decodes escaped keys before checking
/// duplicates. Keeping number tokens avoids NSNumber's boolean and rounding
/// conversions. Depth is bounded independently of the byte cap.
private struct StrictMetadataJSON {
    private let bytes: [UInt8]
    private var offset = 0
    private let decoder = JSONDecoder()
    private static let maximumDepth = 64

    init(data: Data) { bytes = Array(data) }

    mutating func parse() throws -> JSONValue {
        skipSpace()
        let value = try parseValue(depth: 0)
        skipSpace()
        guard offset == bytes.count else { throw ParseError.malformed }
        return value
    }

    private enum ParseError: Error { case malformed, duplicateKey }

    private mutating func parseValue(depth: Int) throws -> JSONValue {
        guard depth <= Self.maximumDepth, offset < bytes.count else { throw ParseError.malformed }
        switch bytes[offset] {
        case 0x7b: return try parseObject(depth: depth)
        case 0x5b: return try parseArray(depth: depth)
        case 0x22: return .string(try parseString())
        case 0x74: try consume([0x74, 0x72, 0x75, 0x65]); return .other
        case 0x66: try consume([0x66, 0x61, 0x6c, 0x73, 0x65]); return .other
        case 0x6e: try consume([0x6e, 0x75, 0x6c, 0x6c]); return .other
        case 0x2d, 0x30...0x39: return .number(try parseNumber())
        default: throw ParseError.malformed
        }
    }

    private mutating func parseObject(depth: Int) throws -> JSONValue {
        offset += 1
        skipSpace()
        if take(0x7d) { return .object([:]) }
        var fields: [String: JSONValue] = [:]
        while true {
            let key = try parseString()
            guard fields[key] == nil else { throw ParseError.duplicateKey }
            skipSpace()
            guard take(0x3a) else { throw ParseError.malformed }
            skipSpace()
            fields[key] = try parseValue(depth: depth + 1)
            skipSpace()
            if take(0x7d) { return .object(fields) }
            guard take(0x2c) else { throw ParseError.malformed }
            skipSpace()
        }
    }

    private mutating func parseArray(depth: Int) throws -> JSONValue {
        offset += 1
        skipSpace()
        if take(0x5d) { return .array([]) }
        var values: [JSONValue] = []
        while true {
            values.append(try parseValue(depth: depth + 1))
            skipSpace()
            if take(0x5d) { return .array(values) }
            guard take(0x2c) else { throw ParseError.malformed }
            skipSpace()
        }
    }

    private mutating func parseString() throws -> String {
        guard take(0x22) else { throw ParseError.malformed }
        let start = offset - 1
        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            if byte == 0x22 {
                return try decoder.decode(String.self, from: Data(bytes[start..<offset]))
            }
            guard byte >= 0x20 else { throw ParseError.malformed }
            if byte == 0x5c {
                guard offset < bytes.count else { throw ParseError.malformed }
                let escape = bytes[offset]
                offset += 1
                if escape == 0x75 {
                    for _ in 0..<4 {
                        guard offset < bytes.count, Self.isHex(bytes[offset]) else {
                            throw ParseError.malformed
                        }
                        offset += 1
                    }
                } else if ![0x22, 0x5c, 0x2f, 0x62, 0x66, 0x6e, 0x72, 0x74].contains(escape) {
                    throw ParseError.malformed
                }
            }
        }
        throw ParseError.malformed
    }

    private mutating func parseNumber() throws -> String {
        let start = offset
        _ = take(0x2d)
        guard offset < bytes.count else { throw ParseError.malformed }
        if !take(0x30) {
            guard (0x31...0x39).contains(bytes[offset]) else { throw ParseError.malformed }
            consumeDigits()
        }
        if take(0x2e) {
            let decimalStart = offset
            consumeDigits()
            guard offset > decimalStart else { throw ParseError.malformed }
        }
        if take(0x65) || take(0x45) {
            if !take(0x2b) { _ = take(0x2d) }
            let exponentStart = offset
            consumeDigits()
            guard offset > exponentStart else { throw ParseError.malformed }
        }
        return String(decoding: bytes[start..<offset], as: UTF8.self)
    }

    private mutating func consumeDigits() {
        while offset < bytes.count, (0x30...0x39).contains(bytes[offset]) { offset += 1 }
    }
    private mutating func skipSpace() {
        while offset < bytes.count, [0x20, 0x09, 0x0a, 0x0d].contains(bytes[offset]) { offset += 1 }
    }
    private mutating func take(_ value: UInt8) -> Bool {
        guard offset < bytes.count, bytes[offset] == value else { return false }
        offset += 1
        return true
    }
    private mutating func consume(_ literal: [UInt8]) throws {
        for byte in literal { guard take(byte) else { throw ParseError.malformed } }
    }
    private static func isHex(_ value: UInt8) -> Bool {
        (0x30...0x39).contains(value) || (0x41...0x46).contains(value)
            || (0x61...0x66).contains(value)
    }
}
