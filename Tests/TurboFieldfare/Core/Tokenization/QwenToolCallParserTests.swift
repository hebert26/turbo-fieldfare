import Foundation
import Testing
@testable import TurboFieldfare

@Suite("Qwen tool-call parser")
struct QwenToolCallParserTests {
    private let parser = QwenToolCallParser()

    private func schema(
        _ properties: [(String, ModelChatJSONValue)],
        required: [String] = [],
        additionalProperties: ModelChatJSONValue = .bool(false)
    ) -> ModelChatJSONValue {
        .object([
            .init("type", .string("object")),
            .init("properties", .object(properties.map { .init($0.0, $0.1) })),
            .init("required", .array(required.map { .string($0) })),
            .init("additionalProperties", additionalProperties),
        ])
    }

    private func stringSchema(
        enum values: [String]? = nil,
        minLength: Int? = nil,
        maxLength: Int? = nil
    ) -> ModelChatJSONValue {
        var members: [ModelChatJSONMember] = [.init("type", .string("string"))]
        if let values {
            members.append(.init("enum", .array(values.map { .string($0) })))
        }
        if let minLength {
            members.append(.init("minLength", .integer(Int64(minLength))))
        }
        if let maxLength {
            members.append(.init("maxLength", .integer(Int64(maxLength))))
        }
        return .object(members)
    }

    private func tool(
        _ name: String = "lookup",
        properties: [(String, ModelChatJSONValue)]? = nil,
        required: [String] = ["query"]
    ) -> ModelChatToolDefinition {
        let effectiveProperties = properties ?? [("query", stringSchema())]
        return ModelChatToolDefinition(function: ModelChatFunctionDefinition(
            name: name,
            description: "test tool",
            parameters: schema(effectiveProperties, required: required)))
    }

    private func frame(
        function: String = "lookup",
        parameters: [(String, String)] = [("query", "snow")]
    ) -> String {
        var text = "<tool_call>\n<function=\(function)>\n"
        for (name, body) in parameters {
            text += "<parameter=\(name)>\n\(body)\n</parameter>\n"
        }
        text += "</function>\n</tool_call>"
        return text
    }

    @Test func parsesRawUnicodeAndMultilineStringExactly() throws {
        let value = "雪<&>'\nsecond\r\n"
        let parsed = try parser.parse(
            frame(parameters: [("query", value)]),
            tools: [tool()],
            id: "host-42")

        #expect(parsed.id == "host-42")
        #expect(parsed.name == "lookup")
        #expect(parsed.arguments == .object(["query": .string(value)]))
        #expect(parsed.argumentsJSON.contains("雪"))
    }

    @Test func parsesNonStringParameterAsOneJSONValue() throws {
        let definition = tool(
            properties: [
                ("options", .object([
                    .init("type", .string("object")),
                    .init("properties", .object([
                        .init("limit", .object([.init("type", .string("integer"))])),
                    ])),
                    .init("required", .array([.string("limit")])),
                    .init("additionalProperties", .bool(false)),
                ])),
            ],
            required: ["options"])
        let parsed = try parser.parse(
            frame(parameters: [("options", "{\"limit\":3}")]),
            tools: [definition],
            id: "host-1")

        #expect(parsed.arguments == .object([
            "options": .object(["limit": .integer(3)]),
        ]))
    }

    @Test func acceptsAnEmptyArgumentListWhenSchemaHasNoRequiredProperties() throws {
        let empty = tool("ping", properties: [], required: [])
        let parsed = try parser.parse(
            frame(function: "ping", parameters: []),
            tools: [empty],
            id: "host-empty")
        #expect(parsed.arguments == .object([:]))
    }

    @Test func rejectsUnknownToolWithoutRelaxationEscapeHatch() {
        #expect(throws: GemmaToolCallParserError.unknownTool("write")) {
            try parser.parse(
                frame(function: "write"), tools: [tool()], id: "ignored")
        }
    }

    @Test func rejectsUnknownParameterAndMissingRequiredParameter() {
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("other", "snow")]),
                tools: [tool()],
                id: "ignored")
        }

        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: []),
                tools: [tool()],
                id: "ignored")
        }
    }

    @Test func rejectsDuplicateParameterAndDuplicateToolDefinitions() {
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("query", "one"), ("query", "two")]),
                tools: [tool()],
                id: "ignored")
        }

        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(),
                tools: [tool(), tool()],
                id: "ignored")
        }
    }

    @Test func rejectsMissingFramingLineFeedBeforeParameterClose() {
        let malformed = "<tool_call>\n<function=lookup>\n"
            + "<parameter=query>\nsnow</parameter>\n</function>\n</tool_call>"
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(malformed, tools: [tool()], id: "ignored")
        }
    }

    @Test func rejectsTrailingNestedAndAmbiguousFraming() {
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame() + "tail", tools: [tool()], id: "ignored")
        }
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                "<tool_call>\n<function=lookup>\n<parameter=query>\n"
                    + "<parameter=other>\nvalue\n</parameter>\n"
                    + "</parameter>\n</function>\n</tool_call>",
                tools: [tool()],
                id: "ignored")
        }
    }

    @Test func validatesRawStringEnumAndLengthConstraints() throws {
        let constrained = tool(
            properties: [
                ("query", stringSchema(enum: ["雪"], minLength: 1, maxLength: 2)),
            ])
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(parameters: [("query", "no")]), tools: [constrained], id: "x")
        }
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(parameters: [("query", "雪雪雪")]), tools: [constrained], id: "x")
        }
        #expect(try parser.parse(
            frame(parameters: [("query", "雪")]), tools: [constrained], id: "x").arguments
            == .object(["query": .string("雪")]))
    }

    @Test func rejectsInvalidJSONForNonStringParameters() throws {
        let numeric = tool(
            properties: [("count", .object([.init("type", .string("integer"))]))],
            required: ["count"])
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(parameters: [("count", "not-json")]), tools: [numeric], id: "x")
        }
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(parameters: [("count", "1.5")]), tools: [numeric], id: "x")
        }
    }

    @Test func validatesRawBoundsWithoutRelyingOnAnEnumMismatch() throws {
        let bounded = tool(
            properties: [("query", stringSchema(minLength: 2, maxLength: 4))])
        let valid = try parser.parse(
            frame(parameters: [("query", "雪雪")]), tools: [bounded], id: "x")
        #expect(valid.arguments == .object(["query": .string("雪雪")]))
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(parameters: [("query", "x")]), tools: [bounded], id: "x")
        }
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(parameters: [("query", "雪雪雪雪雪")]), tools: [bounded], id: "x")
        }
    }

    @Test func validatesRawStringConstIndependently() throws {
        let constrained = tool(properties: [("query", .object([
            .init("type", .string("string")),
            .init("const", .string("snow")),
        ]))])
        let valid = try parser.parse(
            frame(parameters: [("query", "snow")]), tools: [constrained], id: "x")
        #expect(valid.arguments == .object(["query": .string("snow")]))
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(parameters: [("query", "rain")]), tools: [constrained], id: "x")
        }
    }

    @Test func validatesRecursiveArrayObjectAndNullableIntegerSchemas() throws {
        let nested = tool(
            properties: [("options", .object([
                .init("type", .string("object")),
                .init("properties", .object([
                    .init("tags", .object([
                        .init("type", .string("array")),
                        .init("items", .object([.init("type", .string("string"))])),
                    ])),
                    .init("child", .object([
                        .init("type", .string("object")),
                        .init("properties", .object([
                            .init("enabled", .object([.init("type", .string("boolean"))])),
                        ])),
                        .init("required", .array([.string("enabled")])),
                        .init("additionalProperties", .bool(false)),
                    ])),
                ])),
                .init("required", .array([.string("tags"), .string("child")])),
                .init("additionalProperties", .bool(false)),
            ]))],
            required: ["options"])
        let body = "{\"tags\":[\"雪\",\"<&>\"],\"child\":{\"enabled\":true}}"
        let parsed = try parser.parse(
            frame(parameters: [("options", body)]), tools: [nested], id: "nested")
        #expect(parsed.arguments == .object([
            "options": .object([
                "tags": .array([.string("雪"), .string("<&>")]),
                "child": .object(["enabled": .bool(true)]),
            ]),
        ]))

        let nullable = tool(
            properties: [("count", .object([
                .init("type", .array([.string("integer"), .string("null")])),
            ]))],
            required: [])
        let nullValue = try parser.parse(
            frame(parameters: [("count", "null")]), tools: [nullable], id: "nullable")
        #expect(nullValue.arguments == .object(["count": .null]))
    }

    @Test func rejectsDuplicateNestedJSONKeysIncludingEscapedEquivalentKeys() throws {
        let nested = tool(
            properties: [("options", .object([
                .init("type", .string("object")),
                .init("properties", .object([
                    .init("limit", .object([.init("type", .string("integer"))])),
                ])),
                .init("required", .array([.string("limit")])),
                .init("additionalProperties", .bool(false)),
            ]))],
            required: ["options"])
        for body in [
            "{\"limit\":1,\"limit\":2}",
            "{\"\\u006cimit\":1,\"limit\":2}",
        ] {
            #expect(throws: GemmaToolCallParserError.malformed) {
                try parser.parse(
                    frame(parameters: [("options", body)]), tools: [nested], id: "x")
            }
        }
    }

    @Test func validatesRawLengthByUnicodeScalarRatherThanGraphemeCount() throws {
        let combining = "e\u{301}"
        let maxOne = tool(properties: [("query", stringSchema(maxLength: 1))])
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("query", combining)]), tools: [maxOne], id: "x")
        }
        let minTwo = tool(properties: [("query", stringSchema(minLength: 2))])
        _ = try parser.parse(
            frame(parameters: [("query", combining)]), tools: [minTwo], id: "x")

        let emojiSequence = "👩‍💻"
        let maxTwo = tool(properties: [("query", stringSchema(maxLength: 2))])
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("query", emojiSequence)]), tools: [maxTwo], id: "x")
        }
        let minThree = tool(properties: [("query", stringSchema(minLength: 3))])
        _ = try parser.parse(
            frame(parameters: [("query", emojiSequence)]), tools: [minThree], id: "x")
    }

    @Test func numericEnumAndConstUseExactValueEquivalence() throws {
        let halfEnum = tool(properties: [("value", .object([
            .init("type", .string("number")),
            .init("enum", .array([.number(0.5)])),
        ]))], required: ["value"])
        _ = try parser.parse(
            frame(parameters: [("value", "0.5")]), tools: [halfEnum], id: "x")
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("value", "0.5000000001")]), tools: [halfEnum], id: "x")
        }
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("value", "0.5000000000000000000000000000000000000001")]),
                tools: [halfEnum], id: "x")
        }

        let halfConst = tool(properties: [("value", .object([
            .init("type", .string("number")),
            .init("const", .number(0.5)),
        ]))], required: ["value"])
        _ = try parser.parse(
            frame(parameters: [("value", "0.5")]), tools: [halfConst], id: "x")
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("value", "0.5000000000000000000000000000000000000001")]),
                tools: [halfConst], id: "x")
        }

        let oneConst = tool(properties: [("value", .object([
            .init("type", .string("number")),
            .init("const", .number(1.0)),
        ]))], required: ["value"])
        _ = try parser.parse(
            frame(parameters: [("value", "1")]), tools: [oneConst], id: "x")
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("value", "1.1")]), tools: [oneConst], id: "x")
        }

        // A Double host schema cannot round a distinct integer into acceptance.
        let precision = tool(properties: [("value", .object([
            .init("type", .string("number")),
            .init("const", .number(9_007_199_254_740_992.0)),
        ]))], required: ["value"])
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(
                frame(parameters: [("value", "9007199254740993")]),
                tools: [precision], id: "x")
        }
    }

    @Test func rejectsDeepJSONArrayWithoutStackRecursion() throws {
        let depth = 16_384
        let arrayTool = tool(properties: [("value", .object([
            .init("type", .string("array")),
        ]))], required: ["value"])
        let arrayBody = String(repeating: "[", count: depth)
            + "0" + String(repeating: "]", count: depth)
        #expect(arrayBody.utf8.count < GemmaToolCallParser.maximumBytes)
        #expect(throws: GemmaToolCallParserError.self) {
            try parser.parse(
                frame(parameters: [("value", arrayBody)]), tools: [arrayTool], id: "x")
        }
    }

    @Test func rejectsDeepJSONObjectWithoutStackRecursion() throws {
        let depth = 16_384
        let objectTool = tool(properties: [("value", .object([
            .init("type", .string("object")),
            .init("additionalProperties", .bool(false)),
        ]))], required: ["value"])
        let objectBody = String(repeating: "{\"x\":", count: depth)
            + "null" + String(repeating: "}", count: depth)
        #expect(objectBody.utf8.count < GemmaToolCallParser.maximumBytes)
        #expect(throws: GemmaToolCallParserError.self) {
            try parser.parse(
                frame(parameters: [("value", objectBody)]), tools: [objectTool], id: "x")
        }
    }

    @Test func toleratesSchemaAnnotationsAndRejectsUnsupportedConstrainingKeywords() throws {
        let annotated = ModelChatToolDefinition(function: ModelChatFunctionDefinition(
            name: "lookup", description: "description", parameters: .object([
                .init("type", .string("object")),
                .init("description", .string("annotation")),
                .init("properties", .object([
                    .init("query", .object([
                        .init("type", .string("string")),
                        .init("title", .string("annotation")),
                        .init("default", .string("snow")),
                        .init("examples", .array([.string("snow")])),
                    ])),
                ])),
                .init("required", .array([.string("query")])),
                .init("additionalProperties", .bool(false)),
            ])))
        _ = try parser.parse(frame(), tools: [annotated], id: "x")

        let unsupported = tool(properties: [("query", .object([
            .init("type", .string("string")),
            .init("pattern", .string(".*")),
        ]))])
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(frame(), tools: [unsupported], id: "x")
        }
    }

    @Test func rejectsOversizedSingleFrameBeforeParsing() {
        let huge = String(repeating: "x", count: 256 * 1024)
        #expect(throws: GemmaToolCallParserError.oversized) {
            try parser.parse(frame(parameters: [("query", huge)]), tools: [tool()], id: "x")
        }
    }
}
