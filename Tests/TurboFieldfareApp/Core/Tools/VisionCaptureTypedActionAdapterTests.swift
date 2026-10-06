import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureTypedActionAdapterTests {
    private let supportedActions: Set<String> = [
        "describe_screen", "tap", "type", "swipe",
    ]

    @Test func mapsEmbeddedTapTargetAndPreservesBoundRequestData() throws {
        let request: JSONValue = .object([
            "request": .string("tap profile-card"),
            "bundle_id": .string("com.example.nestmind"),
            "parameters": .object([
                "udid": .string("device-1"),
                "observation_grant": .string("grant-1"),
            ]),
            "session_id": .string("flow-1"),
            "session_kind": .string("flow"),
        ])

        let adapted = try VisionCaptureMCPClient.typedExecuteArguments(
            request, supportedActions: supportedActions)

        #expect(adapted == .object([
            "action": .string("tap"),
            "bundle_id": .string("com.example.nestmind"),
            "parameters": .object([
                "udid": .string("device-1"),
                "observation_grant": .string("grant-1"),
                "query": .string("profile-card"),
            ]),
            "session_id": .string("flow-1"),
        ]))
        #expect(request.objectValue?["request"] == .string("tap profile-card"))
        #expect(request.objectValue?["session_kind"] == .string("flow"))
    }

    @Test func mapsFixedReadRequestToAdvertisedActionAndKeepsDescribeOptions() throws {
        let request: JSONValue = .object([
            "request": .string("describe screen"),
            "bundle_id": .string("com.example.nestmind"),
            "parameters": .object([
                "udid": .string("device-1"),
                "describe": .object(["redaction": .string("balanced")]),
            ]),
            "mode": .string("preview"),
        ])

        let adapted = try VisionCaptureMCPClient.typedExecuteArguments(
            request, supportedActions: supportedActions)

        #expect(adapted == .object([
            "action": .string("describe_screen"),
            "bundle_id": .string("com.example.nestmind"),
            "parameters": .object([
                "udid": .string("device-1"),
                "describe": .object(["redaction": .string("balanced")]),
            ]),
            "mode": .string("preview"),
        ]))
    }

    @Test func mapsTypedTextIntoItsParameterWithoutChangingTheOriginal() throws {
        let request: JSONValue = .object([
            "request": .string("type Add milk to the list"),
            "bundle_id": .string("com.example.nestmind"),
            "parameters": .object(["udid": .string("device-1")]),
        ])

        let adapted = try VisionCaptureMCPClient.typedExecuteArguments(
            request, supportedActions: supportedActions)

        #expect(adapted == .object([
            "action": .string("type"),
            "bundle_id": .string("com.example.nestmind"),
            "parameters": .object([
                "udid": .string("device-1"),
                "text": .string("Add milk to the list"),
            ]),
        ]))
        #expect(request.objectValue?["request"] == .string("type Add milk to the list"))
    }

    @Test func mapsVisualActionsWithoutChangingTheirParameters() throws {
        let actions: [(String, String)] = [
            ("tap coordinates", "tap_coordinates"),
            ("activate computer use", "activate_computer_use"),
            ("click pointer", "click_pointer"),
            ("hide pointer", "hide_pointer"),
        ]
        for (requestName, actionName) in actions {
            let request: JSONValue = .object([
                "request": .string(requestName),
                "bundle_id": .string("com.example.nestmind"),
                "parameters": .object([
                    "udid": .string("device-1"),
                    "x_norm": .integer(500),
                    "y_norm": .integer(400),
                ]),
            ])
            let adapted = try VisionCaptureMCPClient.typedExecuteArguments(
                request,
                supportedActions: [actionName])
            #expect(adapted.objectValue?["action"] == .string(actionName))
            #expect(adapted.objectValue?["parameters"] == request.objectValue?["parameters"])
        }
    }

    @Test func refusesUnknownRequestsAndActionsTheServerDidNotAdvertise() {
        expectRefused(.object([
            "request": .string("open settings"),
            "parameters": .object([:]),
        ]), supportedActions: supportedActions)

        expectRefused(.object([
            "request": .string("tap profile-card"),
            "parameters": .object([:]),
        ]), supportedActions: ["describe_screen"])
    }

    @Test func refusesUnknownOrConflictingRequestFields() {
        let base: [String: JSONValue] = [
            "request": .string("tap profile-card"),
            "parameters": .object([:]),
        ]
        var unknownTopLevel = base
        unknownTopLevel["unrecognized"] = .bool(true)
        var alreadyTyped = base
        alreadyTyped["action"] = .string("tap")

        expectRefused(.object(unknownTopLevel), supportedActions: supportedActions)
        expectRefused(.object(alreadyTyped), supportedActions: supportedActions)

        // Even an identical query is ambiguous because the legacy request and
        // typed parameter would both claim the target.
        expectRefused(.object([
            "request": .string("tap profile-card"),
            "parameters": .object(["query": .string("profile-card")]),
        ]), supportedActions: supportedActions)
    }

    @Test func refusesUnknownSessionKindBeforeChangingTheRequest() {
        let request: JSONValue = .object([
            "request": .string("describe screen"),
            "parameters": .object([:]),
            "session_id": .string("flow-1"),
            "session_kind": .string("other"),
        ])

        expectRefused(request, supportedActions: supportedActions)
        #expect(request.objectValue?["session_kind"] == .string("other"))
    }

    private func expectRefused(
        _ request: JSONValue,
        supportedActions: Set<String>,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        do {
            _ = try VisionCaptureMCPClient.typedExecuteArguments(
                request, supportedActions: supportedActions)
            Issue.record("unsupported typed request was accepted", sourceLocation: sourceLocation)
        } catch let error as VisionCaptureMCPError {
            #expect(error.description.contains("No action was sent"), sourceLocation: sourceLocation)
        } catch {
            Issue.record("unexpected adapter error: \(error)", sourceLocation: sourceLocation)
        }
    }
}
