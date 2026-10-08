import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureDecisionChoiceTests {
    @Test
    func proposalCorrectionPreservesChoicesAndListsOnlyPermittedActions() throws {
        let packet: JSONValue = .object([
            "allowed_next": .array([.string("back"), .string("swipe"), .string("tap")]),
            "choices": .array([.object(["id": .string("c33"), "label": .string("return")])]),
            "last_action": .object(["action": .string("observe"), "verdict": .string("not_sent")]),
            "guidance": .string("Observation is not permitted."),
        ])
        let corrected = try VisionCaptureToolLoop.addingProposalCorrection(to: packet.encoded())
        let result = try JSONDecoder().decode(JSONValue.self, from: Data(corrected.utf8))
        #expect(result.objectValue?["choices"] == packet.objectValue?["choices"])
        #expect(result.objectValue?["allowed_next"] == packet.objectValue?["allowed_next"])
        #expect(result.objectValue?["last_action"] == packet.objectValue?["last_action"])
        #expect(corrected.contains("Allowed actions now: [back, swipe, tap]"))
        #expect(corrected.contains("without a tool call"))
    }

    @Test
    func editableFieldAliasesPublishOneIdentifierChoicePerElement() {
        let root: JSONValue = .object([
            "elements": .array([
                Self.field(
                    id: "title-field", label: "Bookmark title",
                    identifier: "bookmarkTitle", x: 547, y: 205),
                Self.field(
                    id: "url-field", label: "Bookmark URL",
                    identifier: "bookmarkURL", x: 547, y: 285),
            ]),
            "editable_fields": .array([
                Self.editableMetadata(id: "title-field", placeholder: "Bookmark title"),
                Self.editableMetadata(id: "url-field", placeholder: "Bookmark URL"),
            ]),
        ])

        let fields = VisionCaptureToolLoop.returnedEditableFields(from: root)

        #expect(fields.count == 2)
        #expect(fields.map(\.selectorKind) == ["identifier", "identifier"])
        #expect(Set(fields.map(\.selector)) == ["bookmarkTitle", "bookmarkURL"])
        #expect(Set(fields.compactMap(\.elementID)) == ["title-field", "url-field"])
    }

    @Test
    func fieldsWithoutElementIdentityRemainSeparate() {
        let root: JSONValue = .object([
            "elements": .array([
                Self.field(id: nil, label: "First", identifier: "firstField", x: 200, y: 200),
                Self.field(id: nil, label: "Second", identifier: "secondField", x: 200, y: 300),
            ]),
        ])

        let fields = VisionCaptureToolLoop.returnedEditableFields(from: root)

        #expect(fields.count == 4)
        #expect(Set(fields.map(\.selector)) == ["First", "Second", "firstField", "secondField"])
    }

    @Test
    func placeholderLabelsAnIdentifierFieldWhenAccessibilityLabelIsMissing() {
        let facts = VisionCaptureScreenFacts(elements: [
            .object([
                "element_id": .string("todo-title"),
                "identifier": .string("ui.todos.titleField"),
                "role": .string("text_field"),
                "type": .string("XCUIElementTypeTextField"),
                "visible": .bool(true),
                "enabled": .bool(true),
                "editable_field_metadata": .object([
                    "enabled": .bool(true),
                    "visible": .bool(true),
                    "type": .string("XCUIElementTypeTextField"),
                    "placeholder": .object([
                        "status": .string("present"),
                        "text": .string("What do you want to do?"),
                    ]),
                ]),
            ]),
        ])

        #expect(facts.readableLabel(
            selector: "ui.todos.titleField",
            role: "text_field",
            selectorKind: "identifier") == "What do you want to do?")
    }

    @Test
    func semanticActionDedupKeepsDistinctPositionsAndAlertButtons() {
        let firstPosition: JSONValue = .object([
            "x_norm": .integer(200), "y_norm": .integer(300),
        ])
        let secondPosition: JSONValue = .object([
            "x_norm": .integer(800), "y_norm": .integer(300),
        ])
        let facts = VisionCaptureScreenFacts(elements: [
            Self.button(id: "first", label: "Open", type: "XCUIElementTypeButton", x: 200, y: 300),
            Self.button(id: "second", label: "Open", type: "XCUIElementTypeButton", x: 800, y: 300),
        ])
        let common: [String: JSONValue] = [
            "action": .string("tap"), "role": .string("button"),
            "label": .string("Open"), "enabled": .bool(true), "selected": .bool(false),
        ]
        var firstManifest = common
        firstManifest["selector"] = .string("__vc_button_occurrence_v1_first")
        firstManifest["position"] = firstPosition
        var firstCandidate = common
        firstCandidate["selector"] = .string("first")
        firstCandidate["position"] = firstPosition
        firstCandidate["requires_validation"] = .bool(true)
        var secondManifest = common
        secondManifest["selector"] = .string("__vc_button_occurrence_v1_second")
        secondManifest["position"] = secondPosition
        let allow: JSONValue = .object([
            "action": .string("tap"), "selector": .string("Allow"),
            "role": .string("system_alert_button"),
        ])
        let deny: JSONValue = .object([
            "action": .string("tap"), "selector": .string("Don’t Allow"),
            "role": .string("system_alert_button"),
        ])

        let deduplicated = VisionCaptureToolLoop.deduplicatedDecisionActions(
            [.object(firstManifest), .object(firstCandidate), .object(secondManifest), allow, deny],
            facts: facts)

        #expect(deduplicated.count == 4)
        #expect(deduplicated[0] == .object(firstManifest))
        #expect(deduplicated[1] == .object(secondManifest))
        #expect(deduplicated[2] == allow)
        #expect(deduplicated[3] == deny)
    }

    @Test
    func semanticActionDedupDoesNotHideSelectionStateChanges() {
        let position: JSONValue = .object([
            "x_norm": .integer(500), "y_norm": .integer(300),
        ])
        let facts = VisionCaptureScreenFacts(elements: [
            Self.button(id: "choice", label: "Learning", type: "XCUIElementTypeButton", x: 500, y: 300),
        ])
        let unselected: JSONValue = .object([
            "action": .string("tap"), "selector": .string("choice"),
            "role": .string("button"), "label": .string("Learning"),
            "position": position, "selected": .bool(false),
        ])
        let selected: JSONValue = .object([
            "action": .string("tap"), "selector": .string("choice"),
            "role": .string("button"), "label": .string("Learning"),
            "position": position, "selected": .bool(true),
        ])

        let deduplicated = VisionCaptureToolLoop.deduplicatedDecisionActions(
            [unselected, selected], facts: facts)

        #expect(deduplicated == [unselected, selected])
    }

    @Test
    func selectedButtonIsEvidenceInsteadOfAnExecutableTap() {
        let selected: [String: JSONValue] = [
            "action": .string("tap"), "selector": .string("active-profile"),
            "role": .string("button"), "label": .string("Gemma Explorer, active"),
            "selected": .bool(true),
        ]
        var unselected = selected
        unselected["selected"] = .bool(false)

        #expect(VisionCaptureToolLoop.isAlreadySelectedTap(selected, isField: false))
        #expect(!VisionCaptureToolLoop.isAlreadySelectedTap(unselected, isField: false))
        #expect(!VisionCaptureToolLoop.isAlreadySelectedTap(selected, isField: true))
    }

    @Test
    func completedCycleEntryMatchesOnlyTheExactTapRoute() {
        let route: [String: JSONValue] = [
            "action": .string("tap"), "selector": .string("profile-card"),
            "selector_kind": .string("identifier"), "role": .string("button"),
            "label": .string("Test Profile, Profile, Bookmarks & Data"),
        ]

        #expect(VisionCaptureToolLoop.matchesCompletedCycleEntry(
            route, isField: false, operation: "tap", selector: "profile-card",
            selectorKind: "identifier", role: "button", desiredState: nil))
        #expect(!VisionCaptureToolLoop.matchesCompletedCycleEntry(
            route, isField: true, operation: "tap", selector: "profile-card",
            selectorKind: "identifier", role: "button", desiredState: nil))
        #expect(!VisionCaptureToolLoop.matchesCompletedCycleEntry(
            route, isField: false, operation: "tap", selector: "other-card",
            selectorKind: "identifier", role: "button", desiredState: nil))
        #expect(!VisionCaptureToolLoop.matchesCompletedCycleEntry(
            route, isField: false, operation: "set_boolean", selector: "profile-card",
            selectorKind: "identifier", role: "button", desiredState: nil))
    }

    @Test
    func softwareKeyboardSuppressesOnlyEmojiAndShift() {
        let facts = VisionCaptureScreenFacts(elements: [
            Self.button(id: "space", label: "space", type: "XCUIElementTypeKey", x: 500, y: 891),
            Self.button(id: "emoji", label: "Emoji", type: "XCUIElementTypeButton", x: 106, y: 961),
            Self.button(id: "shift", label: "shift", type: "XCUIElementTypeButton", x: 75, y: 830),
            Self.button(id: "return", label: "return", type: "XCUIElementTypeKey", x: 871, y: 891),
            Self.button(id: "app-emoji", label: "Emoji", type: "XCUIElementTypeButton", x: 500, y: 200),
        ])

        #expect(facts.isLowValueSoftwareKeyboardControl(
            selector: "__vc_button_occurrence_v1_emoji", role: "button",
            displayLabel: "Emoji", position: Self.position(x: 106, y: 961)))
        #expect(facts.isLowValueSoftwareKeyboardControl(
            selector: "shift", role: "button"))
        #expect(!facts.isLowValueSoftwareKeyboardControl(
            selector: "return", role: "button"))
        #expect(!facts.isLowValueSoftwareKeyboardControl(
            selector: "app-emoji", role: "button"))
        let summary = facts.summary()
        #expect(summary?.contains("Emoji") == true)
        #expect(summary?.contains("shift") == false)
        #expect(summary?.components(separatedBy: "[button] Emoji").count == 2)

        let appWithoutKeyboard = VisionCaptureScreenFacts(elements: [
            Self.button(id: "bottom-emoji", label: "Emoji", type: "XCUIElementTypeButton", x: 500, y: 900),
        ])
        #expect(!appWithoutKeyboard.isLowValueSoftwareKeyboardControl(
            selector: "bottom-emoji", role: "button"))
    }

    @Test
    func repeatedThinkingCanReuseOneCurrentScreenshot() throws {
        let packet: JSONValue = .object([
            "last_action": .object([
                "action": .string("screenshot"),
                "verdict": .string("observed"),
            ]),
            "observation": .object([
                "current_image_evidence": .bool(true),
                "state": .string("current"),
            ]),
        ])

        let retry = try VisionCaptureToolLoop.repeatedThinkingRetryContent(
            from: packet.encoded(), imageCount: 1)
        let content = try #require(retry)
        let recovered = try JSONDecoder().decode(
            JSONValue.self, from: Data(content.utf8)).objectValue
        #expect(recovered?["generation_recovery"] != nil)
        #expect(recovered?["image_attachment_order"] != nil)
        let missingImage = try VisionCaptureToolLoop.repeatedThinkingRetryContent(
            from: packet.encoded(), imageCount: 0)
        #expect(missingImage == nil)
    }

    @Test
    func visualRecoveryCorrectionListsTheNewAllowedActions() {
        let old = "Not sent. Return a corrected response. Allowed actions now: [observe, screenshot, swipe, tap]. Use an action from this list."
        let updated = VisionCaptureToolLoop.correctionListingCurrentActions(old,
            allowedNext: .array([.string("swipe"), .string("tap"), .string("tap_coordinates")]))
        #expect(updated.contains("Allowed actions now: [swipe, tap, tap_coordinates]."))
        #expect(!updated.contains("observe, screenshot"))
        #expect(VisionCaptureToolLoop.correctionListingCurrentActions(old, allowedNext: nil) == old)
    }

    @Test
    func repeatedThinkingReadNoteClaimsNoEffectOnlyWhenTheScreenDidNotChange() {
        #expect(VisionCaptureToolLoop.repeatedThinkingReadNote(
            outcome: ["screen_changed": .bool(false)]).contains("no visible effect"))
        #expect(!VisionCaptureToolLoop.repeatedThinkingReadNote(
            outcome: ["screen_changed": .bool(true)]).contains("no visible effect"))
        #expect(!VisionCaptureToolLoop.repeatedThinkingReadNote(outcome: [:]).contains("no visible effect"))
    }

    @Test
    func coordinateImageSupportReplacesTheScreenshotRequest() {
        let rejection = "Not sent. Coordinate actions need current screenshot evidence. Choose screenshot now, then use positions from that screenshot in the next step. Use only the current choices."
        let supported = VisionCaptureToolLoop.coordinateImageSupportCorrection(rejection)
        #expect(!supported.contains("Choose screenshot now"))
        #expect(supported.contains("use positions from the attached image"))
        #expect(supported.hasSuffix("Use only the current choices."))
    }

    @Test
    func actionCycleNotesThreeRepeatsAndPausesOnTheFourth() throws {
        typealias Tracker = VisionCaptureToolLoop.ActionCycleTracker
        let sequence = ["tap Edit", "tap Tags", "tap Title", "tap Save"]
        func steps(_ repeats: Int, facts: (Int) -> String = { "f\($0 % 4)" }) -> [Tracker.Step] {
            (0..<(sequence.count * repeats)).map {
                Tracker.Step(action: sequence[$0 % sequence.count], facts: facts($0))
            }
        }
        #expect(Tracker.cycle(in: steps(2)) == nil)
        #expect(Tracker.cycle(in: steps(3)) == Tracker.Cycle(sequence: sequence, repeats: 3))
        #expect(Tracker.cycle(in: steps(4))?.repeats == 4)
        // New app facts in the latest repetition are progress.
        #expect(Tracker.cycle(in: steps(3, facts: { "f\($0)" })) == nil)
        // One action repeated alone is left to the identical-call rules.
        #expect(Tracker.cycle(in: Array(repeating: Tracker.Step(action: "tap A", facts: "f"), count: 9)) == nil)

        var tracker = Tracker()
        let cycles = steps(4).map { tracker.record(action: $0.action, facts: $0.facts) }
        #expect(cycles.firstIndex { $0 != nil } == 11)
        #expect(cycles.last??.repeats == 4)

        let noted = try VisionCaptureToolLoop.addingCycleNote(
            to: JSONValue.object(["guidance": .string("Choose again.")]).encoded(), sequence: sequence)
        #expect(noted.contains("You repeated the same 4 actions 3 times without new app facts"))
        #expect(noted.contains("Choose again."))
    }

    @Test
    func validationMessageIsTheFirstAppearedTextThatReadsLikeOne() {
        #expect(VisionCaptureToolLoop.validationMessage(
            in: ["Step 2 of 6", "Not a YouTube link", "Invalid URL"]) == "Not a YouTube link")
        #expect(VisionCaptureToolLoop.validationMessage(in: ["Title is REQUIRED"]) == "Title is REQUIRED")
        #expect(VisionCaptureToolLoop.validationMessage(
            in: ["Password must be 8 characters"]) == "Password must be 8 characters")
        #expect(VisionCaptureToolLoop.validationMessage(in: ["Upload failed"]) == "Upload failed")
        #expect(VisionCaptureToolLoop.validationMessage(in: ["Network error"]) == "Network error")
        #expect(VisionCaptureToolLoop.validationMessage(in: ["Bookmark saved", "Step 3 of 6"]) == nil)
        #expect(VisionCaptureToolLoop.validationMessage(in: []) == nil)
    }

    @Test
    func ambiguousTargetIsRecoverableOnlyBeforeSubmission() {
        func result(
            code: String, dispatchAttempted: Bool? = nil, dispatch: [String: JSONValue]?
        ) -> VisionCaptureMCPResult {
            var evidence: [String: JSONValue] = [
                "target": .object(["status": .string("ambiguous"), "reason_code": .string(code)]),
            ]
            if let dispatch { evidence["dispatch"] = .object(dispatch) }
            return VisionCaptureMCPResult(
                value: .object(["isError": .bool(true),
                    "payload": .object(["interaction_evidence": .object(evidence)])]),
                isError: true, refusalCode: code, dispatchAttempted: dispatchAttempted,
                hasConflictingDispatchAttemptEvidence: false,
                isGuardedTargetRejectedBeforeSubmission: false,
                isStaleActionCapabilityBeforeDispatch: false,
                isSourceLayoutChangedBeforeRevalidation: false,
                isActionAuthorizationExpiredBeforeDispatch: false,
                isObservedTapTargetUnavailableBeforeDispatch: false,
                isDeliveredTransitionContinuation: false,
                isPointerPreCaptureFailureBeforeSubmission: false)
        }
        let rejected: [String: JSONValue] = [
            "status": .string("rejected_before_submission"),
            "submission_started": .bool(false), "delivery_acknowledged": .bool(false),
        ]
        let submitted: [String: JSONValue] = [
            "status": .string("submitted"),
            "submission_started": .bool(true), "delivery_acknowledged": .bool(true),
        ]
        #expect(VisionCaptureToolLoop.isAmbiguousTargetBeforeSubmission(
            result(code: "TARGET_AMBIGUOUS", dispatch: rejected)))
        #expect(!VisionCaptureToolLoop.isAmbiguousTargetBeforeSubmission(
            result(code: "TARGET_AMBIGUOUS", dispatch: submitted)))
        #expect(!VisionCaptureToolLoop.isAmbiguousTargetBeforeSubmission(
            result(code: "TARGET_AMBIGUOUS", dispatchAttempted: true, dispatch: rejected)))
        #expect(!VisionCaptureToolLoop.isAmbiguousTargetBeforeSubmission(
            result(code: "TARGET_AMBIGUOUS", dispatch: nil)))
        #expect(!VisionCaptureToolLoop.isAmbiguousTargetBeforeSubmission(
            result(code: "TARGET_UNAVAILABLE", dispatch: rejected)))
    }

    private static func field(
        id: String?, label: String, identifier: String, x: Int, y: Int
    ) -> JSONValue {
        var object: [String: JSONValue] = [
            "role": .string("text_field"), "type": .string("XCUIElementTypeTextField"),
            "label": .string(label), "identifier": .string(identifier),
            "visible": .bool(true), "enabled": .bool(true),
            "x_norm": .integer(Int64(x)), "y_norm": .integer(Int64(y)),
        ]
        if let id { object["element_id"] = .string(id) }
        return .object(object)
    }

    private static func editableMetadata(id: String, placeholder: String) -> JSONValue {
        .object([
            "element_id": .string(id), "type": .string("XCUIElementTypeTextField"),
            "role": .string("text_field"), "visible": .bool(true), "enabled": .bool(true),
            "placeholder": .object([
                "status": .string("present"), "text": .string(placeholder),
            ]),
        ])
    }

    private static func button(
        id: String, label: String, type: String, x: Int, y: Int
    ) -> JSONValue {
        .object([
            "element_id": .string(id), "identifier": .string(id),
            "role": .string("button"), "type": .string(type),
            "label": .string(label), "visible": .bool(true), "enabled": .bool(true),
            "selected": .bool(false),
            "x_norm": .integer(Int64(x)), "y_norm": .integer(Int64(y)),
        ])
    }

    private static func position(x: Int, y: Int) -> JSONValue {
        .object(["x_norm": .integer(Int64(x)), "y_norm": .integer(Int64(y))])
    }
}
