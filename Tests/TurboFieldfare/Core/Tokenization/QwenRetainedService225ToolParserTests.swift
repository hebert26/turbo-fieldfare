import Foundation
import Testing
@testable import TurboFieldfare

/// Regression vectors from retained service run-225. The complete model output
/// was visible in the final snapshot but was rejected as malformed before the
/// host could admit the call.
@Suite("Qwen retained service 225 tool parser")
struct QwenRetainedService225ToolParserTests {
    private let parser = QwenToolCallParser()

    @Test func retainedLaunchAppFrameDecodesThroughStructuredAssistantDecoder() throws {
        var decoder = QwenStructuredAssistantDecoder(
            tools: [Self.retainedVisionCaptureTool],
            startsInThoughtChannel: false,
            idGenerator: { "retained-225-call" })
        _ = try decoder.consume(Self.retainedLaunchAppFrame)
        let events = try decoder.finish()
        let calls = events.compactMap { event -> ParsedToolCall? in
            guard case .toolCall(let call) = event else { return nil }
            return call
        }

        #expect(calls.count == 1)
        guard let call = calls.first else { return }
        #expect(call.id == "retained-225-call")
        #expect(call.name == "visioncapture_execute")
        #expect(call.arguments == .object([
            "action": .string("launch_app"),
            "bundle_id": .string("com.hebertgo.nestmind.debug"),
            "parameters": .object([
                "udid": .string("7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382"),
            ]),
        ]))
    }

    @Test func retainedSchemaRejectsInvalidActionAndBundleConst() {
        for replacement in [
            ("launch_app", "delete_app"),
            ("com.hebertgo.nestmind.debug", "com.other.app"),
        ] {
            let malformed = Self.retainedLaunchAppFrame.replacingOccurrences(
                of: replacement.0, with: replacement.1)
            #expect(throws: GemmaToolCallParserError.malformed) {
                try parser.parse(malformed, tools: [Self.retainedVisionCaptureTool], id: "ignored")
            }
        }
    }

    @Test func retainedNestedParametersRejectDuplicateJSONKeys() {
        let duplicate = Self.retainedLaunchAppFrame.replacingOccurrences(
            of: #"{"udid": "7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382"}"#,
            with: #"{"udid": "7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382", "udid": "other"}"#)
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(duplicate, tools: [Self.retainedVisionCaptureTool], id: "ignored")
        }
    }

    @Test func infersFiniteNumericConstAndEnumDomains() throws {
        let constant = Self.inferredValueTool(.object([
            .init("const", .number(1.5)),
        ]))
        let constantResult = try parser.parse(
            Self.valueFrame("1.5"), tools: [constant], id: "numeric-const")
        #expect(constantResult.arguments == .object([
            "value": .decimal(Self.decimal("1.5")),
        ]))
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(Self.valueFrame("1.25"), tools: [constant], id: "ignored")
        }
        let integralConstant = Self.inferredValueTool(.object([
            .init("const", .number(1.0)),
        ]))
        let integralResult = try parser.parse(
            Self.valueFrame("1"), tools: [integralConstant], id: "numeric-const-integer")
        #expect(integralResult.arguments == .object(["value": .integer(1)]))

        let enumeration = Self.inferredValueTool(.object([
            .init("enum", .array([.integer(1), .number(2.5)])),
        ]))
        let integerResult = try parser.parse(
            Self.valueFrame("1"), tools: [enumeration], id: "numeric-enum-integer")
        #expect(integerResult.arguments == .object(["value": .integer(1)]))
        let decimalResult = try parser.parse(
            Self.valueFrame("2.5"), tools: [enumeration], id: "numeric-enum-decimal")
        #expect(decimalResult.arguments == .object([
            "value": .decimal(Self.decimal("2.5")),
        ]))
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(Self.valueFrame("3"), tools: [enumeration], id: "ignored")
        }
    }

    @Test func rejectsUnconstrainedOrAmbiguousMissingTypeDomains() {
        let unconstrained = Self.inferredValueTool(.object([
            .init("description", .string("no value domain")),
        ]))
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(Self.valueFrame("anything"), tools: [unconstrained], id: "ignored")
        }

        let ambiguous = Self.inferredValueTool(.object([
            .init("enum", .array([.string("text"), .number(1)])),
        ]))
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(Self.valueFrame("text"), tools: [ambiguous], id: "ignored")
        }

        let empty = Self.inferredValueTool(.object([
            .init("enum", .array([])),
        ]))
        #expect(throws: GemmaToolCallParserError.malformed) {
            try parser.parse(Self.valueFrame("1"), tools: [empty], id: "ignored")
        }
    }

    private static func inferredValueTool(_ valueSchema: ModelChatJSONValue)
        -> ModelChatToolDefinition {
        ModelChatToolDefinition(function: .init(
            name: "inferred_value",
            description: "Test a value inferred from a finite schema constraint.",
            parameters: .object([
                .init("type", .string("object")),
                .init("properties", .object([
                    .init("value", valueSchema),
                ])),
                .init("required", .array([.string("value")])),
                .init("additionalProperties", .bool(false)),
            ])))
    }

    private static func valueFrame(_ body: String) -> String {
        "<tool_call>\n"
            + "<function=inferred_value>\n"
            + "<parameter=value>\n"
            + body + "\n"
            + "</parameter>\n"
            + "</function>\n"
            + "</tool_call>"
    }

    private static func decimal(_ value: String) -> Decimal {
        Decimal(string: value, locale: Locale(identifier: "en_US_POSIX"))!
    }

    private static let retainedLaunchAppFrame =
        "<tool_call>\n"
        + "<function=visioncapture_execute>\n"
        + "<parameter=action>\n"
        + "launch_app\n"
        + "</parameter>\n"
        + "<parameter=bundle_id>\n"
        + "com.hebertgo.nestmind.debug\n"
        + "</parameter>\n"
        + "<parameter=parameters>\n"
        + "{\"udid\": \"7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382\"}\n"
        + "</parameter>\n"
        + "</function>\n"
        + "</tool_call>"

    private static let retainedVisionCaptureTool = ModelChatToolDefinition(
        function: .init(
            name: "visioncapture_execute",
            description: "Launch, then describe.",
            parameters: .object([
                .init("type", .string("object")),
                .init("properties", .object([
                    // The retained tool schema intentionally omits `type` for
                    // enum and const properties, matching effective-tools.json.
                    .init("action", .object([
                        .init("enum", .array([
                            .string("launch_app"), .string("describe_screen"),
                        ])),
                    ])),
                    .init("bundle_id", .object([
                        .init("const", .string("com.hebertgo.nestmind.debug")),
                    ])),
                    .init("parameters", .object([
                        .init("type", .string("object")),
                        .init("properties", .object([
                            .init("udid", .object([
                                .init("const", .string(
                                    "7BE1EC4B-8A9F-4C00-8A2C-D4321F9AB382")),
                            ])),
                        ])),
                        .init("required", .array([.string("udid")])),
                        .init("additionalProperties", .bool(false)),
                    ])),
                ])),
                .init("required", .array([
                    .string("action"), .string("bundle_id"), .string("parameters"),
                ])),
                .init("additionalProperties", .bool(false)),
            ])))
}
