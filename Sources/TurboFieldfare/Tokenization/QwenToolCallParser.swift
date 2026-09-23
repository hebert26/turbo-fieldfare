import Foundation

/// Parses one complete Qwen tool-call frame against the host's tool definitions.
/// The supported schema vocabulary is intentionally smaller than JSON Schema;
/// unsupported constraints fail closed rather than widening host permissions.
/// Non-string JSON input is limited to 128 nested container levels.
public struct QwenToolCallParser: Sendable {
    public init() {}

    public func parse(
        _ text: String,
        tools: [ModelChatToolDefinition],
        id: String
    ) throws -> ParsedToolCall {
        guard text.utf8.count <= GemmaToolCallParser.maximumBytes else {
            throw GemmaToolCallParserError.oversized
        }

        let definitions = try Self.toolDefinitions(tools)
        var cursor = Cursor(text)
        try cursor.consume("<tool_call>\n<function=")
        let name = try cursor.name(terminatedBy: ">\n")
        guard let definition = definitions[name] else {
            throw GemmaToolCallParserError.unknownTool(name)
        }
        let schema = try Schema(definition.function.parameters)
        guard schema.types == [.object] else { throw GemmaToolCallParserError.malformed }

        var arguments: [String: JSONValue] = [:]
        while !cursor.starts(with: "</function>\n") {
            try cursor.consume("<parameter=")
            let parameterName = try cursor.name(terminatedBy: ">\n")
            guard arguments[parameterName] == nil,
                  let parameterSchema = schema.properties[parameterName] else {
                throw GemmaToolCallParserError.malformed
            }
            let body = try cursor.body(terminatedBy: "\n</parameter>\n")
            let value: JSONValue
            if parameterSchema.types == [.string] {
                value = .string(body)
            } else {
                guard let data = body.data(using: .utf8) else {
                    throw GemmaToolCallParserError.malformed
                }
                try JSONDuplicateKeyValidator.validate(data)
                guard let decoded = try? JSONDecoder().decode(JSONValue.self, from: data) else {
                    throw GemmaToolCallParserError.malformed
                }
                value = decoded
            }
            try parameterSchema.validate(value)
            arguments[parameterName] = value
        }
        try cursor.consume("</function>\n</tool_call>")
        guard cursor.isAtEnd,
              schema.required.isSubset(of: Set(arguments.keys)) else {
            throw GemmaToolCallParserError.malformed
        }

        let value = JSONValue.object(arguments)
        try schema.validate(value)
        return ParsedToolCall(
            id: id,
            name: name,
            arguments: value,
            argumentsJSON: try value.encoded())
    }

    private static func toolDefinitions(
        _ tools: [ModelChatToolDefinition]
    ) throws -> [String: ModelChatToolDefinition] {
        var result: [String: ModelChatToolDefinition] = [:]
        for tool in tools {
            let name = tool.function.name
            guard tool.type == "function",
                  isRepresentableName(name),
                  result.updateValue(tool, forKey: name) == nil else {
                throw GemmaToolCallParserError.malformed
            }
        }
        return result
    }

    private static func isRepresentableName(_ name: String) -> Bool {
        let bytes = name.utf8
        guard !bytes.isEmpty, bytes.count <= 64 else { return false }
        return bytes.allSatisfy {
            $0 == 45 || $0 == 95 || (48...57).contains($0)
                || (65...90).contains($0) || (97...122).contains($0)
        }
    }
}

private struct Cursor {
    private let text: String
    private var index: String.Index

    init(_ text: String) {
        self.text = text
        index = text.startIndex
    }

    var isAtEnd: Bool { index == text.endIndex }

    func starts(with literal: String) -> Bool {
        text[index...].hasPrefix(literal)
    }

    mutating func consume(_ literal: String) throws {
        guard starts(with: literal) else { throw GemmaToolCallParserError.malformed }
        index = text.index(index, offsetBy: literal.count)
    }

    mutating func name(terminatedBy delimiter: String) throws -> String {
        guard let end = text[index...].range(of: delimiter)?.lowerBound else {
            throw GemmaToolCallParserError.malformed
        }
        let value = String(text[index..<end])
        guard QwenToolCallParserName.isValid(value) else {
            throw GemmaToolCallParserError.malformed
        }
        index = text.index(end, offsetBy: delimiter.count)
        return value
    }

    mutating func body(terminatedBy delimiter: String) throws -> String {
        guard let range = text[index...].range(of: delimiter) else {
            throw GemmaToolCallParserError.malformed
        }
        let value = String(text[index..<range.lowerBound])
        index = range.upperBound
        return value
    }
}

private enum QwenToolCallParserName {
    static func isValid(_ name: String) -> Bool {
        let bytes = name.utf8
        guard !bytes.isEmpty, bytes.count <= 64 else { return false }
        return bytes.allSatisfy {
            $0 == 45 || $0 == 95 || (48...57).contains($0)
                || (65...90).contains($0) || (97...122).contains($0)
        }
    }
}

private final class Schema {
    enum ValueType: String, Hashable {
        case array, boolean, integer, null, number, object, string
    }

    static let annotations: Set<String> = [
        "$comment", "default", "deprecated", "description", "examples",
        "readOnly", "title", "writeOnly",
    ]
    static let constraints: Set<String> = [
        "type", "nullable", "properties", "required", "additionalProperties",
        "items", "enum", "const", "minLength", "maxLength", "minItems",
        "maxItems", "minProperties", "maxProperties",
    ]

    let types: Set<ValueType>
    let properties: [String: Schema]
    let required: Set<String>
    let items: Schema?
    let enumerated: [JSONValue]?
    let constant: JSONValue?
    let minimumLength: Int?
    let maximumLength: Int?
    let minimumItems: Int?
    let maximumItems: Int?
    let minimumProperties: Int?
    let maximumProperties: Int?

    init(_ value: ModelChatJSONValue) throws {
        let object = try Self.uniqueObject(value)
        guard Set(object.keys).isSubset(of: Self.annotations.union(Self.constraints)) else {
            throw GemmaToolCallParserError.malformed
        }

        types = try Self.types(object)
        if let nullable = object["nullable"] {
            guard nullable == .bool(true) else { throw GemmaToolCallParserError.malformed }
        }

        if let value = object["properties"] {
            guard types.contains(.object) else { throw GemmaToolCallParserError.malformed }
            let members = try Self.uniqueObject(value)
            var parsed: [String: Schema] = [:]
            for (name, schema) in members {
                guard parsed.updateValue(try Schema(schema), forKey: name) == nil else {
                    throw GemmaToolCallParserError.malformed
                }
            }
            properties = parsed
        } else {
            properties = [:]
        }

        if let value = object["required"] {
            guard types.contains(.object), case .array(let values) = value else {
                throw GemmaToolCallParserError.malformed
            }
            let names = try values.map { value -> String in
                guard case .string(let name) = value else {
                    throw GemmaToolCallParserError.malformed
                }
                return name
            }
            guard Set(names).count == names.count,
                  Set(names).isSubset(of: Set(properties.keys)) else {
                throw GemmaToolCallParserError.malformed
            }
            required = Set(names)
        } else {
            required = []
        }

        if let additional = object["additionalProperties"] {
            guard additional == .bool(false) else {
                throw GemmaToolCallParserError.malformed
            }
        }

        if let value = object["items"] {
            guard types.contains(.array) else { throw GemmaToolCallParserError.malformed }
            items = try Schema(value)
        } else {
            items = nil
        }

        if let value = object["enum"] {
            guard case .array(let values) = value, !values.isEmpty else {
                throw GemmaToolCallParserError.malformed
            }
            enumerated = try values.map(Self.jsonValue)
        } else {
            enumerated = nil
        }
        constant = try object["const"].map(Self.jsonValue)
        minimumLength = try Self.nonnegativeInteger(object["minLength"])
        maximumLength = try Self.nonnegativeInteger(object["maxLength"])
        minimumItems = try Self.nonnegativeInteger(object["minItems"])
        maximumItems = try Self.nonnegativeInteger(object["maxItems"])
        minimumProperties = try Self.nonnegativeInteger(object["minProperties"])
        maximumProperties = try Self.nonnegativeInteger(object["maxProperties"])
        guard Self.ordered(minimumLength, maximumLength),
              Self.ordered(minimumItems, maximumItems),
              Self.ordered(minimumProperties, maximumProperties),
              (minimumLength == nil && maximumLength == nil) || types.contains(.string),
              (minimumItems == nil && maximumItems == nil) || types.contains(.array),
              (minimumProperties == nil && maximumProperties == nil) || types.contains(.object) else {
            throw GemmaToolCallParserError.malformed
        }
        if let enumerated {
            for value in enumerated { try validate(value) }
        }
        if let constant { try validate(constant) }
    }

    func validate(_ value: JSONValue) throws {
        let valueType = Self.type(of: value)
        guard types.contains(valueType)
                || (valueType == .integer && types.contains(.number)) else {
            throw GemmaToolCallParserError.malformed
        }
        if let enumerated, !enumerated.contains(where: { Self.equivalent($0, value) }) {
            throw GemmaToolCallParserError.malformed
        }
        if let constant, !Self.equivalent(constant, value) {
            throw GemmaToolCallParserError.malformed
        }
        switch value {
        case .string(let text):
            try Self.validateCount(
                text.unicodeScalars.count, minimum: minimumLength, maximum: maximumLength)
        case .array(let values):
            try Self.validateCount(values.count, minimum: minimumItems, maximum: maximumItems)
            if let items {
                for value in values { try items.validate(value) }
            }
        case .object(let values):
            try Self.validateCount(
                values.count, minimum: minimumProperties, maximum: maximumProperties)
            guard required.isSubset(of: Set(values.keys)),
                  Set(values.keys).isSubset(of: Set(properties.keys)) else {
                throw GemmaToolCallParserError.malformed
            }
            for (name, value) in values {
                guard let schema = properties[name] else {
                    throw GemmaToolCallParserError.malformed
                }
                try schema.validate(value)
            }
        default:
            break
        }
    }

    private static func types(_ object: [String: ModelChatJSONValue]) throws -> Set<ValueType> {
        let parsed: [ValueType]
        switch object["type"] {
        case .string(let raw):
            guard let type = ValueType(rawValue: raw) else {
                throw GemmaToolCallParserError.malformed
            }
            parsed = [type]
        case .array(let values):
            parsed = try values.map {
                guard case .string(let raw) = $0, let type = ValueType(rawValue: raw) else {
                    throw GemmaToolCallParserError.malformed
                }
                return type
            }
            guard parsed.count == 2, parsed.contains(.null) else {
                throw GemmaToolCallParserError.malformed
            }
        default:
            throw GemmaToolCallParserError.malformed
        }
        var result = Set(parsed)
        if object["nullable"] == .bool(true) { result.insert(.null) }
        guard !result.isEmpty,
              !(result.contains(.string) && result.count > 1) else {
            throw GemmaToolCallParserError.malformed
        }
        return result
    }

    private static func uniqueObject(
        _ value: ModelChatJSONValue
    ) throws -> [String: ModelChatJSONValue] {
        guard case .object(let members) = value else {
            throw GemmaToolCallParserError.malformed
        }
        var result: [String: ModelChatJSONValue] = [:]
        for member in members {
            guard result.updateValue(member.value, forKey: member.name) == nil else {
                throw GemmaToolCallParserError.malformed
            }
        }
        return result
    }

    private static func jsonValue(_ value: ModelChatJSONValue) throws -> JSONValue {
        switch value {
        case .object(let members):
            var result: [String: JSONValue] = [:]
            for member in members {
                guard result.updateValue(try jsonValue(member.value), forKey: member.name) == nil else {
                    throw GemmaToolCallParserError.malformed
                }
            }
            return .object(result)
        case .array(let values): return .array(try values.map(jsonValue))
        case .string(let value): return .string(value)
        case .integer(let value): return .integer(value)
        case .unsignedInteger(let value): return .unsignedInteger(value)
        case .number(let value):
            guard value.isFinite else { throw GemmaToolCallParserError.malformed }
            return .number(value)
        case .bool(let value): return .bool(value)
        case .null: return .null
        }
    }

    private static func nonnegativeInteger(_ value: ModelChatJSONValue?) throws -> Int? {
        guard let value else { return nil }
        let integer: UInt64
        switch value {
        case .integer(let raw) where raw >= 0: integer = UInt64(raw)
        case .unsignedInteger(let raw): integer = raw
        default: throw GemmaToolCallParserError.malformed
        }
        guard let result = Int(exactly: integer) else {
            throw GemmaToolCallParserError.malformed
        }
        return result
    }

    private static func ordered(_ minimum: Int?, _ maximum: Int?) -> Bool {
        guard let minimum, let maximum else { return true }
        return minimum <= maximum
    }

    private static func validateCount(_ count: Int, minimum: Int?, maximum: Int?) throws {
        if let minimum, count < minimum { throw GemmaToolCallParserError.malformed }
        if let maximum, count > maximum { throw GemmaToolCallParserError.malformed }
    }

    private static func type(of value: JSONValue) -> ValueType {
        switch value {
        case .array: .array
        case .bool: .boolean
        case .integer, .unsignedInteger: .integer
        case .null: .null
        case .decimal, .number: .number
        case .object: .object
        case .string: .string
        }
    }

    private static func equivalent(_ lhs: JSONValue, _ rhs: JSONValue) -> Bool {
        if let lhsNumber = decimal(lhs), let rhsNumber = decimal(rhs) {
            return lhsNumber == rhsNumber
        }
        switch (lhs, rhs) {
        case let (.object(left), .object(right)):
            guard left.keys == right.keys else { return false }
            return left.allSatisfy { key, value in
                right[key].map { equivalent(value, $0) } == true
            }
        case let (.array(left), .array(right)):
            return left.count == right.count
                && zip(left, right).allSatisfy(equivalent)
        default:
            return lhs == rhs
        }
    }

    private static func decimal(_ value: JSONValue) -> Decimal? {
        switch value {
        case .integer(let value): return Decimal(value)
        case .unsignedInteger(let value):
            return Decimal(string: String(value), locale: Locale(identifier: "en_US_POSIX"))
        case .decimal(let value): return value
        case .number(let value):
            guard value.isFinite else { return nil }
            return Decimal(
                string: String(value), locale: Locale(identifier: "en_US_POSIX"))
        default:
            return nil
        }
    }
}

/// JSONDecoder intentionally keeps only one value for duplicate object keys.
/// This preflight scanner rejects duplicates at every nesting level before
/// decoding, comparing decoded key strings so escaped spellings cannot bypass it.
private struct JSONDuplicateKeyValidator {
    /// Keeps preflight recursion safely bounded before Foundation decoding.
    private static let maximumNestingDepth = 128

    private let bytes: [UInt8]
    private var index = 0

    static func validate(_ data: Data) throws {
        var validator = JSONDuplicateKeyValidator(bytes: Array(data))
        try validator.value(depth: 0)
        validator.whitespace()
        guard validator.index == validator.bytes.count else {
            throw GemmaToolCallParserError.malformed
        }
    }

    private mutating func value(depth: Int) throws {
        guard depth <= Self.maximumNestingDepth else {
            throw GemmaToolCallParserError.malformed
        }
        whitespace()
        guard index < bytes.count else { throw GemmaToolCallParserError.malformed }
        switch bytes[index] {
        case 0x7B: try object(depth: depth)
        case 0x5B: try array(depth: depth)
        case 0x22: _ = try string()
        case 0x74: try literal("true")
        case 0x66: try literal("false")
        case 0x6E: try literal("null")
        case 0x2D, 0x30...0x39: try number()
        default: throw GemmaToolCallParserError.malformed
        }
    }

    private mutating func object(depth: Int) throws {
        index += 1
        whitespace()
        if take(0x7D) { return }
        var keys: Set<String> = []
        while true {
            whitespace()
            let key = try string()
            guard keys.insert(key).inserted else {
                throw GemmaToolCallParserError.malformed
            }
            whitespace()
            guard take(0x3A) else { throw GemmaToolCallParserError.malformed }
            try value(depth: depth + 1)
            whitespace()
            if take(0x7D) { return }
            guard take(0x2C) else { throw GemmaToolCallParserError.malformed }
        }
    }

    private mutating func array(depth: Int) throws {
        index += 1
        whitespace()
        if take(0x5D) { return }
        while true {
            try value(depth: depth + 1)
            whitespace()
            if take(0x5D) { return }
            guard take(0x2C) else { throw GemmaToolCallParserError.malformed }
        }
    }

    private mutating func string() throws -> String {
        guard index < bytes.count, bytes[index] == 0x22 else {
            throw GemmaToolCallParserError.malformed
        }
        let start = index
        index += 1
        var escaped = false
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if escaped {
                escaped = false
            } else if byte == 0x5C {
                escaped = true
            } else if byte == 0x22 {
                let data = Data(bytes[start..<index])
                guard let decoded = try? JSONDecoder().decode(String.self, from: data) else {
                    throw GemmaToolCallParserError.malformed
                }
                return decoded
            } else if byte < 0x20 {
                throw GemmaToolCallParserError.malformed
            }
        }
        throw GemmaToolCallParserError.malformed
    }

    private mutating func literal(_ text: StaticString) throws {
        let expected = Array(String(describing: text).utf8)
        guard index + expected.count <= bytes.count,
              bytes[index..<(index + expected.count)].elementsEqual(expected) else {
            throw GemmaToolCallParserError.malformed
        }
        index += expected.count
    }

    private mutating func number() throws {
        let start = index
        while index < bytes.count,
              "-+0123456789.eE".utf8.contains(bytes[index]) {
            index += 1
        }
        let text = String(decoding: bytes[start..<index], as: UTF8.self)
        guard let source = CanonicalDecimal(text),
              let decimal = Decimal(
                string: text, locale: Locale(identifier: "en_US_POSIX")) else {
            throw GemmaToolCallParserError.malformed
        }
        let decodedText = NSDecimalNumber(decimal: decimal).stringValue
        guard let decoded = CanonicalDecimal(decodedText), source == decoded else {
            throw GemmaToolCallParserError.malformed
        }
    }

    private mutating func whitespace() {
        while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) {
            index += 1
        }
    }

    private mutating func take(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }
}

/// Exact, arbitrary-length normalization used only to verify that Foundation's
/// bounded-precision Decimal decoder did not round a model-supplied JSON number.
private struct CanonicalDecimal: Equatable {
    let isNegative: Bool
    let digits: Substring
    let exponent: Int

    init?(_ text: String) {
        var index = text.startIndex
        let end = text.endIndex
        var negative = false
        if index < end, text[index] == "-" {
            negative = true
            index = text.index(after: index)
        }

        let integerStart = index
        if index < end, text[index] == "0" {
            index = text.index(after: index)
            if index < end, text[index].isNumber { return nil }
        } else {
            guard index < end, ("1"..."9").contains(text[index]) else { return nil }
            while index < end, text[index].isNumber {
                index = text.index(after: index)
            }
        }
        let integer = text[integerStart..<index]

        var fraction: Substring = ""
        if index < end, text[index] == "." {
            index = text.index(after: index)
            let start = index
            while index < end, text[index].isNumber {
                index = text.index(after: index)
            }
            guard start < index else { return nil }
            fraction = text[start..<index]
        }

        var explicitExponent = 0
        if index < end, text[index] == "e" || text[index] == "E" {
            index = text.index(after: index)
            var exponentIsNegative = false
            if index < end, text[index] == "+" || text[index] == "-" {
                exponentIsNegative = text[index] == "-"
                index = text.index(after: index)
            }
            let start = index
            while index < end, text[index].isNumber {
                guard let digit = text[index].wholeNumberValue else { return nil }
                let (timesTen, overflow1) = explicitExponent.multipliedReportingOverflow(by: 10)
                let (next, overflow2) = timesTen.addingReportingOverflow(digit)
                guard !overflow1, !overflow2 else { return nil }
                explicitExponent = next
                index = text.index(after: index)
            }
            guard start < index else { return nil }
            if exponentIsNegative { explicitExponent = -explicitExponent }
        }
        guard index == end else { return nil }

        let combined = integer + fraction
        guard let firstNonzero = combined.firstIndex(where: { $0 != "0" }) else {
            isNegative = false
            digits = "0"
            exponent = 0
            return
        }
        var significant = combined[firstNonzero...]
        var removedTrailingZeros = 0
        while significant.last == "0" {
            significant = significant.dropLast()
            removedTrailingZeros += 1
        }
        let (withoutFraction, overflow1) = explicitExponent.subtractingReportingOverflow(
            fraction.count)
        let (normalizedExponent, overflow2) = withoutFraction.addingReportingOverflow(
            removedTrailingZeros)
        guard !overflow1, !overflow2 else { return nil }
        isNegative = negative
        digits = significant
        exponent = normalizedExponent
    }
}
