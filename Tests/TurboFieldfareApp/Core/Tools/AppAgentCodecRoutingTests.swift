import Foundation
import Testing
@testable import TurboFieldfare
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

@Suite("Agent family routing and host contract")
struct AppAgentCodecRoutingTests {
    @Test func QwenStructuredCodecPublishesOneHostOwnedToolID() throws {
        let tool = lookupTool()
        let frame = "check the current page</think>\n"
            + "<tool_call>\n<function=lookup>\n<parameter=query>\n"
            + "Qwen route\n</parameter>\n</function>\n</tool_call>"
        var decoder = QwenStructuredAssistantDecoder(
            tools: [tool], startsInThoughtChannel: true,
            idGenerator: { "host-call-1" })

        _ = try decoder.consume(frame)
        try decoder.markEndOfStream()
        _ = try decoder.consumeTerminalTail("")
        let events = try decoder.finish()

        #expect(events == [.toolCall(.init(
            id: "host-call-1", name: "lookup",
            arguments: .object(["query": .string("Qwen route")]),
            argumentsJSON: "{\"query\":\"Qwen route\"}"))])
    }

    @Test func QwenRouteCarriesItsPinnedIdentityAndCannotUseTheGemmaRoute() {
        let gemma = AppModelCatalog.entry(for: .gemma4)
        let qwen = AppModelCatalog.entry(for: .qwen3_6)

        guard case .remoteRepack = gemma.installRoute else {
            Issue.record("Gemma lost its remote repack route")
            return
        }
        guard case .localQwenConversion(let local) = qwen.installRoute else {
            Issue.record("Qwen was routed through a remote install")
            return
        }
        #expect(local.sourceIdentity == qwen.sourceIdentity)
        #expect(!qwen.isInstallable)
        #expect(qwen.location.textModelURL != gemma.location.textModelURL)
        #expect(qwen.location.visionModelURL != gemma.location.visionModelURL)
        #expect(qwen.accepts(
            family: .qwen3_6,
            modelID: qwen.sourceIdentity.repoID,
            revision: qwen.sourceIdentity.revision,
            sourceIndexSHA256: qwen.sourceIdentity.sourceIndexSHA256))
        #expect(!qwen.accepts(
            family: .gemma4,
            modelID: qwen.sourceIdentity.repoID,
            revision: qwen.sourceIdentity.revision,
            sourceIndexSHA256: qwen.sourceIdentity.sourceIndexSHA256))
    }

    @Test func hostConfigurationAndReturnedIdentityFailClosed() throws {
        let configuration = VisionCaptureAgentConfiguration(
            bundleIdentifier: "com.example.nestmind",
            simulatorUDID: "00000000-0000-0000-0000-000000000001",
            modelDirectory: URL(fileURLWithPath: "/tmp/qwen.gturbo"))
        try configuration.validate()

        #expect(throws: VisionCaptureAgentError.self) {
            try VisionCaptureAgentConfiguration(
                bundleIdentifier: "nestmind", simulatorUDID: configuration.simulatorUDID,
                modelDirectory: configuration.modelDirectory).validate()
        }
        #expect(throws: VisionCaptureAgentError.self) {
            try VisionCaptureAgentConfiguration(
                bundleIdentifier: configuration.bundleIdentifier,
                simulatorUDID: "not-a-udid",
                modelDirectory: configuration.modelDirectory).validate()
        }

        let valid = identityEvidence(observedBundle: .null)
        try VisionCaptureToolLoop.validateReturnedIdentity(
            in: valid, configuration: configuration, refusalCode: nil)

        var wrongBundle = identityEvidence(observedBundle: .string("com.example.other"))
        #expect(throws: VisionCaptureAgentError.self) {
            try VisionCaptureToolLoop.validateReturnedIdentity(
                in: wrongBundle, configuration: configuration, refusalCode: nil)
        }
        wrongBundle = identityEvidence(
            observedBundle: .null,
            binding: [
                "requested_bundle_id": .string(configuration.bundleIdentifier),
                "observed_bundle_id": .null,
                "observed_pid": .null,
                "udid": .string("10000000-0000-0000-0000-000000000001"),
            ])
        #expect(throws: VisionCaptureAgentError.self) {
            try VisionCaptureToolLoop.validateReturnedIdentity(
                in: wrongBundle, configuration: configuration, refusalCode: nil)
        }
    }

    @Test func allowlistTargetLockAndVerifiedVerdictPreventReplays() throws {
        var restrictions = AgentUserRestrictions()
        restrictions.apply("Do not test or use the Speak button. Continue other checks.")
        #expect(restrictions.prohibits(label: "Speak", selector: "Speak"))
        #expect(!restrictions.prohibits(label: "Speaker settings", selector: "Speaker settings"))

        let route: [String: JSONValue] = [
            "action": .string("tap"), "selector": .string("save-button"),
            "selector_kind": .string("identifier"), "role": .string("button"),
            "label": .string("Save"),
        ]
        #expect(VisionCaptureToolLoop.matchesCompletedCycleEntry(
            route, isField: false, operation: "tap", selector: "save-button",
            selectorKind: "identifier", role: "button", desiredState: nil))
        #expect(!VisionCaptureToolLoop.matchesCompletedCycleEntry(
            route, isField: false, operation: "tap", selector: "other-button",
            selectorKind: "identifier", role: "button", desiredState: nil))

        let selected: [String: JSONValue] = [
            "action": .string("tap"), "selector": .string("save-button"),
            "role": .string("button"), "selected": .bool(true),
        ]
        #expect(VisionCaptureToolLoop.isAlreadySelectedTap(selected, isField: false))

        var checkpoint = AgentTaskCheckpoint()
        checkpoint.appendUser("Verify the save action.", images: [])
        let call = AppToolCall(
            id: "tap-1", name: "visioncapture_navigate",
            arguments: .object([
                "action": .string("tap"), "target": .string("save-button"),
            ]))
        let outcome = try JSONValue.object([
            "outcome": .string("succeeded"),
            "proof": .object([
                "action": .string("tap"), "verdict": .string("verified"),
            ]),
        ]).encoded()
        try checkpoint.appendSettled(
            call: call,
            result: AppToolResult(
                callID: "tap-1", name: "visioncapture_navigate",
                content: try packet(facts: ["[button] Save"], choices: ["Save"])),
            outcome: outcome,
            target: .object([
                "label": .string("Save"), "role": .string("button"),
                "selector": .string("save-button"),
            ]), requestIDs: [], session: nil)

        let rendered = try checkpoint.render(
            currentPacket: try packet(
                facts: ["[static_text] Saved"], choices: []),
            safety: .object([:]))
        #expect(rendered.contains(#""verdict":"verified""#))
        #expect(rendered.contains(#""completed_actions":[{"#))
        #expect(rendered.contains("Apply every safety restriction, including no replay."))
    }

    private func lookupTool() -> ModelChatToolDefinition {
        let schema = ModelChatJSONValue.object([
            .init("type", .string("object")),
            .init("properties", .object([
                .init("query", .object([.init("type", .string("string"))])),
            ])),
            .init("required", .array([.string("query")])),
            .init("additionalProperties", .bool(false)),
        ])
        return ModelChatToolDefinition(function: .init(
            name: "lookup", description: "route test tool", parameters: schema))
    }

    private func identityEvidence(
        observedBundle: JSONValue,
        binding: [String: JSONValue]? = nil
    ) -> JSONValue {
        .object([
            "interaction_evidence": .object([
                "binding": .object(binding ?? [
                    "requested_bundle_id": .string("com.example.nestmind"),
                    "observed_bundle_id": observedBundle,
                    "observed_pid": .null,
                    "udid": .string("00000000-0000-0000-0000-000000000001"),
                ]),
                "dispatch": .object([
                    "status": .string("rejected_before_submission"),
                    "submission_started": .bool(false),
                    "delivery_acknowledged": .bool(false),
                ]),
                "outcome": .object([
                    "status": .string("failed"), "scope": .string("dispatch"),
                    "reason_code": .string("DISPATCH_REJECTED_BEFORE_SUBMISSION"),
                ]),
                "target": .object([
                    "actual_event_recipient_observed": .bool(false),
                    "status": .string("unavailable"),
                    "reason_code": .string("TARGET_UNAVAILABLE"),
                ]),
            ]),
        ])
    }

    private func packet(facts: [String], choices: [String]) throws -> String {
        try JSONValue.object([
            "schema_version": .integer(1),
            "allowed_next": .array([
                .string("observe"), .string("screenshot"), .string("tap"),
            ]),
            "choices": .array(choices.enumerated().map { index, label in
                .object([
                    "id": .string("choice-\(index + 1)"),
                    "label": .string(label),
                    "role": .string("button"),
                    "operations": .array([.string("tap")]),
                ])
            }),
            "facts": .array(facts.map(JSONValue.string)),
            "last_action": .object([
                "action": .string("tap"), "verdict": .string("verified"),
            ]),
            "observation": .object([
                "id": .string("current"), "state": .string("current"),
            ]),
        ]).encoded()
    }
}
