import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct AgentTaskCheckpointTests {
    @Test
    func checkpointKeepsWorkStatusWithoutHistoricalScreenEvidence() throws {
        var checkpoint = AgentTaskCheckpoint()
        checkpoint.appendUser("Find the saved Swift bookmark and verify Learning.", images: [])

        let searchPacket = try Self.packet(
            action: "type",
            facts: ["[static_text] Search Bookmarks"],
            choices: ["Swift Programming Language, swift.org, -15%"])
        try checkpoint.appendSettled(
            call: Self.call("type", id: "type-1"),
            result: AppToolResult(callID: "type-1", name: "visioncapture_navigate", content: searchPacket),
            outcome: try JSONValue.object([
                "outcome": .string("succeeded"),
                "proof": .object(["action": .string("type"), "verdict": .string("verified")]),
            ]).encoded(),
            target: .object([
                "label": .string("Search bookmarks by meaning"),
                "role": .string("text_field"),
                "position": .object(["x_norm": .integer(500), "y_norm": .integer(177)]),
            ]),
            requestIDs: [], session: nil)

        let webPacket = try Self.packet(
            action: "tap", facts: ["[other] Address: swift.org"], choices: ["Close"])
        try checkpoint.appendSettled(
            call: Self.call("tap", id: "tap-1"),
            result: AppToolResult(callID: "tap-1", name: "visioncapture_navigate", content: webPacket),
            outcome: try JSONValue.object([
                "outcome": .string("inconclusive"),
                "delivery_acknowledged": .bool(true),
                "proof": .object(["action": .string("tap"), "verdict": .string("inconclusive")]),
            ]).encoded(),
            target: .object([
                "label": .string("Swift Programming Language, swift.org, -15%"),
                "role": .string("button"),
                "position": .object(["x_norm": .integer(500), "y_norm": .integer(213)]),
            ]),
            requestIDs: [], session: nil)

        let rendered = try checkpoint.render(
            currentPacket: webPacket,
            safety: .object(["read_only_recovery_required": .bool(false)]))

        #expect(rendered.contains(#""completed_actions":[{"event":1"#))
        #expect(rendered.contains(#""follow_up_attempts":[{"event":2"#))
        #expect(rendered.contains("Swift Programming Language, swift.org, -15%"))
        #expect(!rendered.contains(#""evidence":"#))
        #expect(!rendered.contains("proven_visible_targets"))
        #expect(!rendered.contains("x_norm"))
        #expect(!rendered.contains("pending_history_result\":null"))
        #expect(rendered.utf8.count < 6_000)
    }

    @Test
    func renderOmitsOlderScreenFactsAfterUnrelatedScreens() throws {
        var checkpoint = AgentTaskCheckpoint()
        checkpoint.appendUser("Verify the saved Learning bookmarks, then test Assistant.", images: [])

        try Self.appendObservation(
            "Swift bookmark under Learning", event: 1, checkpoint: &checkpoint)
        for event in 2...6 {
            try Self.appendObservation(
                "Unrelated screen \(event)", event: event, checkpoint: &checkpoint)
        }
        _ = try checkpoint.render(
            currentPacket: Self.packet(
                action: "observe", facts: ["[static_text] First checkpoint"], choices: []),
            safety: .object([:]))
        try Self.appendObservation(
            "Bookmark count 2", event: 7, checkpoint: &checkpoint)
        for event in 8...12 {
            try Self.appendObservation(
                "Later screen \(event)", event: event, checkpoint: &checkpoint)
        }
        try Self.appendObservation(
            "Bookmark count 3", event: 13, checkpoint: &checkpoint)

        let rendered = try checkpoint.render(
            currentPacket: Self.packet(
                action: "observe", facts: ["[static_text] Assistant"], choices: []),
            safety: .object([:]))

        #expect(!rendered.contains("Swift bookmark under Learning"))
        #expect(!rendered.contains("Bookmark count 2"))
        #expect(!rendered.contains("Bookmark count 3"))
        #expect(rendered.contains("Assistant"))
    }

    @Test
    func evidenceLedgerBoundsWholeGroupsAndReportsOverflow() throws {
        var events: [JSONValue] = []
        var observations: [String: JSONValue] = [:]
        for event in 1...40 {
            let id = "observation-\(event)"
            let fact = String(repeating: "fact-\(event)-", count: 35) + "END"
            events.append(.object([
                "event": .integer(Int64(event)),
                "observation": .string(id),
            ]))
            observations[id] = .object([
                "facts": .array([.string("[static_text] \(fact)")]),
            ])
        }

        let ledger = try AgentTaskCheckpoint.evidenceLedger(
            events: events, observations: observations, excluding: .object([:]))
        let body = try #require(ledger.objectValue)
        guard case .array(let groups)? = body["observations"] else {
            Issue.record("The evidence ledger did not contain observation groups.")
            return
        }

        #expect(try ledger.encoded().utf8.count <= AgentTaskCheckpoint.maximumEvidenceLedgerBytes)
        #expect(groups.count <= AgentTaskCheckpoint.maximumEvidenceGroups)
        #expect(body["coverage"] == .string("partial"))
        #expect(body["omitted_observations"] != .integer(0))
        let containsOnlyWholeFacts = groups.allSatisfy { group in
            guard case .array(let facts)? = group.objectValue?["facts"] else { return false }
            return facts.allSatisfy {
                guard case .string(let fact) = $0 else { return false }
                return fact.hasSuffix("END")
            }
        }
        #expect(containsOnlyWholeFacts)
    }

    @Test
    func evidenceLedgerSkipsOversizedFirstGroupAndKeepsLaterFacts() throws {
        let events: [JSONValue] = (1...4).map { event in
            .object([
                "event": .integer(Int64(event)),
                "observation": .string("observation-\(event)"),
            ])
        }
        let observations: [String: JSONValue] = [
            "observation-1": .object([
                "facts": .array([
                    .string("[static_text] \(String(repeating: "oversized ", count: 600))")
                ]),
            ]),
            "observation-2": .object([
                "facts": .array([.string("[static_text] Swift bookmark saved")]),
            ]),
            "observation-3": .object([
                "facts": .array([.string("[static_text] Blender video saved")]),
            ]),
            "observation-4": .object([
                "facts": .array([.string("[static_text] Assistant found related content")]),
            ]),
        ]

        let ledger = try AgentTaskCheckpoint.evidenceLedger(
            events: events, observations: observations, excluding: .object([:]))
        let encoded = try ledger.encoded()

        #expect(encoded.utf8.count <= AgentTaskCheckpoint.maximumEvidenceLedgerBytes)
        #expect(!encoded.contains("oversized"))
        #expect(encoded.contains("Swift bookmark saved"))
        #expect(encoded.contains("Blender video saved"))
        #expect(encoded.contains("Assistant found related content"))
        #expect(encoded.contains(#""coverage":"partial""#))
        #expect(encoded.contains(#""omitted_observations":1"#))
    }

    @Test
    func evidenceLedgerDeduplicatesReorderedFacts() throws {
        let events: [JSONValue] = [1, 2].map { event in
            .object([
                "event": .integer(Int64(event)),
                "observation": .string("observation-\(event)"),
            ])
        }
        let observations: [String: JSONValue] = [
            "observation-1": .object([
                "facts": .array([
                    .string("[static_text] Swift bookmark"),
                    .string("[static_text] Learning"),
                ]),
            ]),
            "observation-2": .object([
                "facts": .array([
                    .string("[static_text] Learning"),
                    .string("[static_text] Swift bookmark"),
                ]),
            ]),
        ]

        let ledger = try AgentTaskCheckpoint.evidenceLedger(
            events: events, observations: observations, excluding: .object([:]))
        guard case .array(let groups)? = ledger.objectValue?["observations"] else {
            Issue.record("The evidence ledger did not contain observation groups.")
            return
        }

        #expect(groups.count == 1)
        #expect(ledger.objectValue?["omitted_observations"] == .integer(0))
    }

    @Test
    func successfulHistoricalScreenshotIsNotRenderedAsCompletedWork() throws {
        var checkpoint = AgentTaskCheckpoint()
        checkpoint.appendUser("Verify the saved bookmark.", images: [])
        let screenshot = try Self.packet(
            action: "screenshot", facts: ["[static_text] Swift bookmark saved"], choices: [])
        try checkpoint.appendSettled(
            call: Self.call("screenshot", id: "screenshot-1"),
            result: AppToolResult(
                callID: "screenshot-1", name: "visioncapture_navigate", content: screenshot),
            outcome: try JSONValue.object([
                "outcome": .string("succeeded"),
                "observation_outcome": .string("succeeded"),
            ]).encoded(),
            target: nil, requestIDs: [], session: nil)

        let rendered = try checkpoint.render(
            currentPacket: Self.packet(
                action: "observe", facts: ["[static_text] Assistant"], choices: []),
            safety: .object([:]))

        #expect(rendered.contains(#""completed_actions":[]"#))
        #expect(!rendered.contains("Swift bookmark saved"))
        #expect(rendered.contains("Assistant"))
    }

    @Test
    func checkpointDropsUnsentProposalsAndDeduplicatesFailedAttempts() throws {
        var checkpoint = AgentTaskCheckpoint()
        checkpoint.appendUser("Continue the QA journey.", images: [])
        let packet = try Self.packet(
            action: "tap", facts: ["[button] Clear text"], choices: ["Sheet Grabber"])
        let failedOutcome = try JSONValue.object([
            "outcome": .string("not_dispatched_reobserved"),
            "proof": .object([
                "action": .string("tap"),
                "reason": .string("The current screen could not be validated."),
                "verdict": .string("failed"),
            ]),
            "refusal": .object([
                "fact": .string("The action permission expired before input was sent.")
            ]),
        ]).encoded()
        let target = JSONValue.object([
            "label": .string("Clear text"), "role": .string("button"),
            "enabled": .bool(true), "selected": .bool(false),
        ])
        for event in 1...2 {
            try checkpoint.appendSettled(
                call: Self.call("tap", id: "failed-\(event)"),
                result: AppToolResult(
                    callID: "failed-\(event)", name: "visioncapture_navigate", content: packet),
                outcome: failedOutcome, target: target, requestIDs: [], session: nil)
        }
        try checkpoint.appendSettled(
            call: Self.call("tap", id: "malformed"),
            result: AppToolResult(
                callID: "malformed", name: "visioncapture_navigate", content: packet),
            outcome: try JSONValue.object([
                "outcome": .string("not_sent"),
                "instruction": .string("The proposed target supports type."),
            ]).encoded(),
            target: .object([
                "label": .string("Search bookmarks by meaning"),
                "role": .string("text_field"),
            ]), requestIDs: [], session: nil)

        let rendered = try checkpoint.render(
            currentPacket: packet, safety: .object([:]))

        #expect(rendered.contains(#""failed_actions":[{"event":1"#))
        #expect(rendered.contains(#""follow_up_attempts":[]"#))
        #expect(!rendered.contains(#""event":2,"request"#))
        #expect(!rendered.contains(#""event":3,"request"#))
        #expect(!rendered.contains("The proposed target supports type."))
        #expect(rendered.components(separatedBy: #""request":{"action":"tap"}"#).count == 2)
    }

    @Test
    func laterCurrentScreenResolvesAndRemembersInconclusiveNavigation() throws {
        var checkpoint = AgentTaskCheckpoint()
        checkpoint.appendUser("Open Todos and create one item.", images: [])
        let unavailable = try Self.packet(
            action: "tap",
            facts: ["No app-owned accessibility action or editable field is currently published."],
            choices: [])
        try checkpoint.appendSettled(
            call: Self.call("tap", id: "todos-tap"),
            result: AppToolResult(
                callID: "todos-tap", name: "visioncapture_navigate", content: unavailable),
            outcome: try JSONValue.object([
                "outcome": .string("inconclusive"),
                "delivery_acknowledged": .bool(true),
                "proof": .object([
                    "action": .string("tap"), "verdict": .string("inconclusive")
                ]),
            ]).encoded(),
            target: .object([
                "label": .string("Todos"), "role": .string("button"),
            ]), requestIDs: [], session: nil)

        let proved = try checkpoint.render(
            currentPacket: Self.packet(
                action: "observe", facts: ["[static_text] Your remaining todos: 1"], choices: ["Add"]),
            safety: .object([:]))
        #expect(proved.contains(#""follow_up_attempts":[]"#))
        #expect(proved.contains(#""verified_by":"later_current_screen""#))
        #expect(!proved.contains(#""delivery_acknowledged":true"#))

        let remembered = try checkpoint.render(
            currentPacket: Self.packet(
                action: "observe", facts: ["[static_text] Assistant"], choices: []),
            safety: .object([:]))
        #expect(remembered.contains(#""follow_up_attempts":[]"#))
        #expect(remembered.contains(#""verified_by":"later_current_screen""#))
    }

    private static func call(_ action: String, id: String) -> AppToolCall {
        AppToolCall(id: id, name: "visioncapture_navigate",
            arguments: .object(["action": .string(action), "target": .string("expired")]))
    }

    private static func appendObservation(
        _ fact: String,
        event: Int,
        checkpoint: inout AgentTaskCheckpoint
    ) throws {
        let id = "observe-\(event)"
        let packet = try Self.packet(
            action: "observe", facts: ["[static_text] \(fact)"], choices: [])
        try checkpoint.appendSettled(
            call: Self.call("observe", id: id),
            result: AppToolResult(
                callID: id, name: "visioncapture_navigate", content: packet),
            outcome: try JSONValue.object(["outcome": .string("succeeded")]).encoded(),
            target: nil, requestIDs: [], session: nil)
    }

    private static func packet(
        action: String, facts: [String], choices: [String]
    ) throws -> String {
        try JSONValue.object([
            "schema_version": .integer(1),
            "allowed_next": .array([.string("observe"), .string("screenshot"), .string("tap")]),
            "choices": .array(choices.enumerated().map { index, label in
                .object([
                    "id": .string("c\(index + 1)"), "label": .string(label),
                    "role": .string("button"), "operations": .array([.string("tap")]),
                ])
            }),
            "facts": .array(facts.map(JSONValue.string)),
            "last_action": .object(["action": .string(action), "verdict": .string("observed")]),
            "observation": .object(["id": .string("current"), "state": .string("current")]),
        ]).encoded()
    }
}
