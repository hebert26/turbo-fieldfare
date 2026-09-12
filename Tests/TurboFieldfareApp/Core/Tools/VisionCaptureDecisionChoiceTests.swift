import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureDecisionChoiceTests {
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
