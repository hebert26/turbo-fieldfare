import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureScreenFactsEmptyFieldTests {
    @Test
    func emptyEditableValuePrintsValueEmpty() throws {
        let facts = try Self.facts(fieldValue: "\"\"", valueStatus: "\"available\"")
        let summary = try #require(facts.summary())
        #expect(summary.contains("[text_field] Quick add value=empty"))
        #expect(facts.properties(selector: "Quick add", role: "text_field")["value"] == .string(""))
        #expect(facts.properties(selector: "Quick add", role: "text_field")["value_status"] == .string("available"))
    }

    @Test
    func placeholderAmbiguousValuePrintsPlaceholderNote() throws {
        let facts = try Self.facts(fieldValue: "null", valueStatus: "\"placeholder_ambiguous\"")
        let summary = try #require(facts.summary())
        #expect(summary.contains("[text_field] Quick add shows-placeholder-only (probably empty)"))
    }

    @Test
    func keyboardHintDependsOnWhetherTheFieldIsEmpty() throws {
        let empty = try #require(Self.facts(fieldValue: "\"\"", valueStatus: "\"available\"").summary())
        #expect(empty.contains("[keyboard] open: the focused field is empty; type its text first, then the return key submits it"))
        #expect(!empty.contains("the return key submits the focused one-line field"))

        let ambiguous = try #require(
            Self.facts(fieldValue: "null", valueStatus: "\"placeholder_ambiguous\"").summary())
        #expect(ambiguous.contains("the focused field is empty"))

        let filled = try #require(Self.facts(fieldValue: "\"Buy milk\"", valueStatus: "\"available\"").summary())
        #expect(filled.contains("[keyboard] open: the return key submits the focused one-line field when no submit button can be reached (choose the return choice)"))
        #expect(!filled.contains("the focused field is empty"))
    }

    @Test
    func coveringControlKeepsItsWarningUnlessEveryCoveredControlIsDisabled() {
        func facts(fieldEnabled: Bool, covers: Int64) -> VisionCaptureScreenFacts {
            VisionCaptureScreenFacts(elements: [
                .object([
                    "element_id": .string("chip"), "label": .string("New item"), "role": .string("button"),
                    "type": .string("XCUIElementTypeButton"), "visible": .bool(true), "enabled": .bool(true),
                    "frame_covers_controls": .integer(covers),
                    "frame": .object(["x": .integer(16), "y": .integer(440), "width": .integer(120), "height": .integer(36)]),
                    "center": .object(["x": .integer(76), "y": .integer(458)]),
                ]),
                .object([
                    "element_id": .string("name"), "role": .string("text_field"),
                    "type": .string("XCUIElementTypeTextField"), "visible": .bool(true), "enabled": .bool(fieldEnabled),
                    "frame": .object(["x": .integer(20), "y": .integer(444), "width": .integer(40), "height": .integer(28)]),
                    "center": .object(["x": .integer(40), "y": .integer(458)]),
                ]),
            ])
        }
        #expect(facts(fieldEnabled: false, covers: 1).coverageWarning(selector: "New item", role: "button") == nil)
        #expect(facts(fieldEnabled: true, covers: 1).coverageWarning(selector: "New item", role: "button") != nil)
        #expect(facts(fieldEnabled: false, covers: 2).coverageWarning(selector: "New item", role: "button") != nil)
    }

    /// A quick-add text field with the software keyboard open, decoded from
    /// JSON the way the loop receives returned elements.
    private static func facts(fieldValue: String, valueStatus: String) throws -> VisionCaptureScreenFacts {
        let json = """
        [
          {"element_id": "quick-add", "role": "text_field", "type": "XCUIElementTypeTextField",
           "label": "Quick add", "value": \(fieldValue), "value_status": \(valueStatus),
           "visible": true, "enabled": true, "x_norm": 500, "y_norm": 300},
          {"role": "button", "type": "XCUIElementTypeKey", "label": "q", "visible": true, "enabled": true},
          {"role": "button", "type": "XCUIElementTypeKey", "label": "return", "visible": true, "enabled": true}
        ]
        """
        let elements = try JSONDecoder().decode([JSONValue].self, from: Data(json.utf8))
        return VisionCaptureScreenFacts(elements: elements)
    }
}
