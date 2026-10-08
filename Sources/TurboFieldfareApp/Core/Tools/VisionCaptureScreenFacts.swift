import Foundation
import TurboFieldfare

/// Readable facts from one structured accessibility observation. Execution
/// selectors still come exclusively from the host's published action list.
struct VisionCaptureScreenFacts {
    struct TapCandidate {
        let selector: String
        let role: String
    }

    private struct Element {
        let elementID: String?
        let role: String
        let type: String?
        let identifier: String?
        let label: String?
        let placeholder: String?
        let value: String?
        let valueStatus: String?
        let enabled: Bool?
        let selected: Bool?
        let position: JSONValue?
        /// XCTest marked the control not visible; VisionCapture confirmed its
        /// text by OCR inside its accessibility frame.
        let ocrConfirmed: Bool
        /// Other controls whose centre lies inside this control's frame.
        let coversControls: Int
        /// Frame and centre in device points, when returned.
        let frame: (x: Double, y: Double, width: Double, height: Double)?
        let center: (x: Double, y: Double)?
    }

    private let elements: [Element]
    private let navigationFacts: [String]

    var hasSoftwareKeyboard: Bool {
        elements.contains { $0.type == "XCUIElementTypeKey" }
    }

    init(elements: [JSONValue], navigation: [[String: JSONValue]] = []) {
        self.elements = elements.compactMap { value in
            guard case .object(let object) = value,
                  object["visible"] == .bool(true)
                      || object["visibility"] == .string("ocr_confirmed")
                      || object["visibility"] == .string("descendant_visible"),
                  let role = Self.text(object["role"]) else { return nil }
            let ocrConfirmed = object["visibility"] == .string("ocr_confirmed")
            let coversControls: Int
            if case .integer(let count)? = object["frame_covers_controls"] { coversControls = Int(count) } else { coversControls = 0 }
            let label = Self.rawText(object["label"])
            let type = Self.text(object["type"])
            let metadata = object["editable_field_metadata"]?.objectValue
            let secure = role == "secure_text_field" || type == "XCUIElementTypeSecureTextField"
                || metadata?["type"] == .string("XCUIElementTypeSecureTextField")
            let returnedStatus = Self.text(object["value_status"])
            let rawValue = Self.rawText(object["value"])
            let valueStatus: String?
            if secure { valueStatus = "secure" }
            else if let returnedStatus {
                valueStatus = ["available", "redacted", "omitted", "unavailable", "placeholder_ambiguous"]
                    .contains(returnedStatus)
                    ? returnedStatus : "unavailable"
            } else if object["value_hash"] != nil || rawValue?.contains("[REDACTED]") == true {
                valueStatus = "redacted"
            } else { valueStatus = nil }
            return Element(
                elementID: Self.rawText(object["element_id"]),
                role: role,
                type: type,
                identifier: Self.rawText(object["identifier"]),
                label: label == Self.rawText(object["element_id"]) ? nil : label,
                placeholder: Self.observedPlaceholder(
                    metadata: metadata, type: type, ocrConfirmed: ocrConfirmed),
                value: valueStatus == nil || valueStatus == "available" ? rawValue : nil,
                valueStatus: valueStatus == "available" && rawValue == nil ? "unavailable" : valueStatus,
                enabled: Self.boolean(object["enabled"]),
                selected: Self.boolean(object["selected"]),
                position: Self.position(in: object),
                ocrConfirmed: ocrConfirmed,
                coversControls: coversControls,
                frame: Self.frame(in: object["frame"]),
                center: Self.point(in: object["center"]))
        }
        navigationFacts = navigation.flatMap { navigation in
            [("tab_bars", "Selected tab"), ("segmented_controls", "Selected segment")]
                .flatMap { key, name -> [String] in
                    guard case .array(let controls) = navigation[key] else { return [] }
                    return controls.compactMap { control in
                        guard case .object(let object) = control,
                              let label = Self.text(object["selected_label"]) else { return nil }
                        return "\(name): \(label)"
                    }
                }
        }
    }

    func properties(
        selector: String, role: String, selectorKind: String? = nil, elementID: String? = nil
    ) -> [String: JSONValue] {
        guard let index = matchingIndex(
            selector: selector, role: role, selectorKind: selectorKind, elementID: elementID) else { return [:] }
        let element = elements[index]
        var properties: [String: JSONValue] = [:]
        if let label = element.label.flatMap(Self.displayText), label != selector {
            properties["label"] = .string(label)
        }
        if let selected = element.selected { properties["selected"] = .bool(selected) }
        if let enabled = element.enabled { properties["enabled"] = .bool(enabled) }
        if let value = element.value { properties["value"] = .string(value) }
        if let status = element.valueStatus { properties["value_status"] = .string(status) }
        if element.value?.isEmpty == true { properties["value_status"] = .string("available") }
        if let position = element.position { properties["position"] = position }
        return properties
    }

    func readableLabel(selector: String, role: String, selectorKind: String? = nil) -> String? {
        guard let index = matchingIndex(selector: selector, role: role, selectorKind: selectorKind)
        else { return nil }
        let element = elements[index]
        let readable = element.label ?? (
            ["text_field", "secure_text_field"].contains(element.role)
                ? element.placeholder : nil)
        guard let label = readable.flatMap(Self.displayText),
              !label.hasPrefix("__vc") else { return nil }
        return label
    }

    /// Correlates two private descriptions of one observed element. The index
    /// is meaningful only inside this observation and is never model-facing.
    func semanticTargetIdentity(
        selector: String,
        role: String,
        selectorKind: String? = nil,
        displayLabel: String? = nil,
        position: JSONValue? = nil
    ) -> Int? {
        if let index = matchingIndex(
            selector: selector, role: role, selectorKind: selectorKind) {
            return index
        }
        guard let displayLabel, let position else { return nil }
        let matches = elements.indices.filter { index in
            let element = elements[index]
            return element.role == role
                && element.label?.utf8.elementsEqual(displayLabel.utf8) == true
                && element.position == position
        }
        return matches.count == 1 ? matches[0] : nil
    }

    /// Letter entry uses the high-level type action. Keep keyboard controls
    /// such as return that can submit a form, and suppress only the observed
    /// mode keys that do not advance an app QA journey.
    func isLowValueSoftwareKeyboardControl(
        selector: String,
        role: String,
        displayLabel: String? = nil,
        position: JSONValue? = nil
    ) -> Bool {
        let softwareKeyboardIsVisible = elements.contains { $0.type == "XCUIElementTypeKey" }
        guard softwareKeyboardIsVisible else { return false }
        let labels = elements.compactMap { element -> String? in
            guard ["XCUIElementTypeKey", "XCUIElementTypeButton"].contains(element.type),
                  element.role == role else {
                return nil
            }
            let exactSelector = element.identifier?.utf8.elementsEqual(selector.utf8) == true
                || element.label?.utf8.elementsEqual(selector.utf8) == true
            let positionedDisplay: Bool
            if let displayLabel, let position {
                positionedDisplay = element.label?.utf8.elementsEqual(displayLabel.utf8) == true
                    && element.position == position
            } else {
                positionedDisplay = false
            }
            guard exactSelector || positionedDisplay else { return nil }
            if element.type == "XCUIElementTypeButton" {
                guard case .object(let point)? = element.position,
                      case .integer(let y)? = point["y_norm"], y >= 650 else { return nil }
            }
            return element.label ?? (exactSelector ? selector : displayLabel)
        }
        return labels.contains { label in
            switch label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "emoji", "shift": true
            default: false
            }
        }
    }

    /// Roles a pointer tap at the element's position can act on.
    private static let tapCandidateRoles: Set<String> = [
        "button", "link", "cell", "switch", "tab", "segmented_item", "menu_item",
    ]

    /// Observation suggests a target to validate, never permission to tap it.
    /// Taps go through the pointer at the element's position, so a candidate
    /// needs a position and a name that is unique within its role.
    func tapCandidates(
        excluding controls: [(selector: String, role: String)],
        excludingLabels published: [(label: String, role: String)] = []
    ) -> [TapCandidate] {
        let represented = Set(controls.compactMap {
            matchingIndex(selector: $0.selector, role: $0.role)
        })
        return elements.indices.compactMap { index in
            let element = elements[index]
            guard !represented.contains(index), element.enabled != false, element.position != nil,
                  Self.tapCandidateRoles.contains(element.role) else { return nil }
            if let label = element.label, published.contains(where: {
                $0.role == element.role && $0.label.utf8.elementsEqual(label.utf8)
            }) { return nil }
            // Prefer an exact identifier, then an exact label. Do not normalize
            // a name into the contract or use a synthetic element ID.
            for selector in [element.identifier, element.label].compactMap({ $0 }) {
                guard !selector.isEmpty, selector.utf8.count <= 160,
                      selector != "[REDACTED]",
                      selector.utf8.elementsEqual(
                        selector.trimmingCharacters(in: .whitespacesAndNewlines).utf8),
                      matchingIndex(selector: selector, role: element.role) == index else { continue }
                return TapCandidate(selector: selector, role: element.role)
            }
            return nil
        }
    }

    func summary(
        excluding controls: [(selector: String, role: String)] = [],
        availableActions: [JSONValue] = [],
        editableFields: [JSONValue] = []
    ) -> String? {
        var excluded = Set(controls.compactMap { matchingIndex(selector: $0.selector, role: $0.role) })
        var lines: [String] = []
        for fact in navigationFacts where !lines.contains(fact) { lines.append(fact) }
        // Choices already carry their facts in the arrays. Exclude only the
        // exact element they represent, never another control with a shared label.
        for choice in availableActions + editableFields {
            guard case .object(let object) = choice,
                  let role = Self.text(object["role"]),
                  let selector = Self.rawText(object["selector"]) else { continue }
            let selectorKind = Self.text(object["selector_kind"])
            if let index = matchingIndex(
                selector: selector, role: role, selectorKind: selectorKind,
                elementID: Self.rawText(object["element_id"])) {
                excluded.insert(index)
            }
        }
        for (index, element) in elements.enumerated() {
            guard !excluded.contains(index), element.type != "XCUIElementTypeKey" else { continue }
            if let selector = element.identifier ?? element.label,
               isLowValueSoftwareKeyboardControl(selector: selector, role: element.role) {
                continue
            }
            // A field without a label is named by its placeholder ("Tag name"
            // against "What do you want to do?"), so the model can tell them apart.
            let content = [element.label ?? element.placeholder, element.value]
                .compactMap { $0.flatMap(Self.displayText) }
                .reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
                .joined(separator: ": ")
            let isControl = ["button", "cell", "tab", "link", "switch", "text_field",
                             "secure_text_field", "interactive", "segmented_item"].contains(element.role)
            guard !content.isEmpty || isControl else { continue }
            var line = "[\(element.role)] \(content.isEmpty ? "label unavailable" : content)"
            if content.isEmpty || element.role == "segmented_item", case .object(let position)? = element.position,
               case .integer(let x)? = position["x_norm"], case .integer(let y)? = position["y_norm"] {
                line += " at (\(x), \(y))"
            }
            if element.ocrConfirmed, !line.contains(" at ("),
               case .object(let position)? = element.position,
               case .integer(let x)? = position["x_norm"], case .integer(let y)? = position["y_norm"] {
                line += " at (\(x), \(y))"
            }
            if element.ocrConfirmed {
                line += " ocr-confirmed: type or tap_coordinates at this position is allowed without a screenshot; for type, pass this line as target"
            }
            if element.coversControls > 0 {
                line += " frame-covers-\(element.coversControls)-controls: an element tap may hit one of them"
            }
            if element.role == "segmented_item" {
                // Preserve independent observed state without joining rounded
                // positions to a private action selector.
                if let selected = element.selected { line += " selected=\(selected)" }
                if let enabled = element.enabled { line += " enabled=\(enabled)" }
            } else {
                if element.selected == true { line += " selected" }
                if element.enabled == false { line += " disabled" }
            }
            if Self.isEditableRole(element.role) {
                if element.value?.isEmpty == true { line += " value=empty" }
                if element.valueStatus == "placeholder_ambiguous" { line += " shows-placeholder-only (probably empty)" }
            }
            if !lines.contains(line) { lines.append(line) }
        }
        let editable = elements.filter { Self.isEditableRole($0.role) }
        if hasSoftwareKeyboard, !editable.isEmpty {
            // The facts carry no focus flag, so the focused field is known to be
            // empty only when every editable field on screen is empty.
            if editable.allSatisfy({ $0.value?.isEmpty == true || $0.valueStatus == "placeholder_ambiguous" }) {
                lines.append("[keyboard] open: the focused field is empty; type its text first, then the return key submits it")
            } else {
                // Generic iOS behaviour: the keyboard's return key submits a one-line
                // field. It is the reliable exit when a submit button cannot be reached.
                lines.append("[keyboard] open: the return key submits the focused one-line field when no submit button can be reached (choose the return choice)")
            }
        }
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }

    private func matchingIndex(
        selector: String, role: String, selectorKind: String? = nil, elementID: String? = nil
    ) -> Int? {
        func exactMatch(_ candidate: String?) -> Bool {
            candidate?.utf8.elementsEqual(selector.utf8) == true
        }
        // A placeholder alias must bind to one returned element, not a rounded
        // position or another field with a coincidentally matching label.
        if selectorKind == "placeholder", elementID == nil { return nil }
        if let elementID {
            guard !elementID.isEmpty,
                  elements.filter({ $0.elementID?.utf8.elementsEqual(elementID.utf8) == true }).count == 1
            else { return nil }
        }
        let matches = elements.indices.filter { index in
            let element = elements[index]
            guard element.role == role else { return false }
            if let elementID, element.elementID?.utf8.elementsEqual(elementID.utf8) != true { return false }
            switch selectorKind {
            case "identifier": return exactMatch(element.identifier)
            case "label": return exactMatch(element.label)
            case "placeholder": return exactMatch(element.placeholder)
            case nil: return exactMatch(element.identifier) || exactMatch(element.label)
            default: return false
            }
        }
        return matches.count == 1 ? matches.first : nil
    }

    private static func observedPlaceholder(
        metadata: [String: JSONValue]?, type: String?, ocrConfirmed: Bool = false
    ) -> String? {
        guard let metadata, let type, metadata["type"] == .string(type),
              metadata["enabled"] == .bool(true),
              metadata["visible"] == .bool(true) || ocrConfirmed,
              let placeholder = metadata["placeholder"]?.objectValue,
              placeholder["status"] == .string("present"),
              let raw = rawText(placeholder["text"]), !raw.isEmpty, raw.utf8.count <= 256,
              raw != "[REDACTED]" else { return nil }
        return raw
    }

    private static func text(_ value: JSONValue?) -> String? {
        rawText(value).flatMap(displayText)
    }

    private static func rawText(_ value: JSONValue?) -> String? {
        guard case .string(let raw) = value else { return nil }
        return raw
    }

    private static func displayText(_ raw: String) -> String? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    private static func double(_ value: JSONValue?) -> Double? {
        switch value {
        case .integer(let number)?: Double(number)
        case .unsignedInteger(let number)?: Double(number)
        case .number(let number)?: number
        case .decimal(let number)?: NSDecimalNumber(decimal: number).doubleValue
        default: nil
        }
    }

    private static func frame(in value: JSONValue?) -> (x: Double, y: Double, width: Double, height: Double)? {
        guard case .object(let object)? = value,
              let x = double(object["x"]), let y = double(object["y"]),
              let width = double(object["width"]), let height = double(object["height"]) else { return nil }
        return (x, y, width, height)
    }

    private static func point(in value: JSONValue?) -> (x: Double, y: Double)? {
        guard case .object(let object)? = value,
              let x = double(object["x"]), let y = double(object["y"]) else { return nil }
        return (x, y)
    }

    private static func boolean(_ value: JSONValue?) -> Bool? {
        guard case .bool(let boolean) = value else { return nil }
        return boolean
    }

    /// Controls hidden from XCTest but confirmed by OCR carry accessibility
    /// positions, so coordinate taps on them need no screenshot evidence.
    /// Position of an OCR-confirmed editable field for a type action. The target may be
    /// the fact line itself (it carries "at (x, y)") or any text when one such field exists.
    func ocrConfirmedTypingPosition(for target: String?) -> (x: Int64, y: Int64)? {
        let fields = elements.filter { $0.ocrConfirmed && Self.isEditableRole($0.role) }
        func point(_ element: Element) -> (x: Int64, y: Int64)? {
            guard case .object(let position)? = element.position,
                  case .integer(let x)? = position["x_norm"],
                  case .integer(let y)? = position["y_norm"] else { return nil }
            return (x, y)
        }
        if let target, let range = target.range(of: #"\((\d+),\s*(\d+)\)"#, options: .regularExpression) {
            let numbers = target[range].split(whereSeparator: { !$0.isNumber }).compactMap { Int64($0) }
            if numbers.count == 2, isOCRConfirmedPosition(x: numbers[0], y: numbers[1]) {
                return (numbers[0], numbers[1])
            }
        }
        if fields.count == 1, let point = point(fields[0]) { return point }
        return nil
    }

    private static func isEditableRole(_ role: String) -> Bool {
        ["text_field", "textfield", "secure_text_field", "search_field", "text_view", "textview"]
            .contains(role.lowercased())
    }

    /// Normalized position of a control, for a tap by position when a tap by name is refused.
    func normalizedPosition(selector: String, role: String, selectorKind: String? = nil) -> (x: Int64, y: Int64)? {
        guard let index = matchingIndex(selector: selector, role: role, selectorKind: selectorKind),
              case .object(let position)? = elements[index].position,
              case .integer(let x)? = position["x_norm"],
              case .integer(let y)? = position["y_norm"] else { return nil }
        return (x, y)
    }

    /// Warning text for a control whose frame covers other controls, or nil.
    /// The element's enabled flag, or nil when unknown or not on screen.
    func isEnabled(selector: String, role: String, selectorKind: String? = nil) -> Bool? {
        guard let index = matchingIndex(selector: selector, role: role, selectorKind: selectorKind) else { return nil }
        return elements[index].enabled
    }

    func coverageWarning(selector: String, role: String, selectorKind: String? = nil) -> String? {
        guard let index = matchingIndex(selector: selector, role: role, selectorKind: selectorKind),
              elements[index].coversControls > 0,
              !coversOnlyDisabledControls(index) else { return nil }
        return "its frame covers \(elements[index].coversControls) other controls, so an element tap may hit one of them; if the tap has no effect, take a screenshot and use tap_coordinates on the visible control"
    }

    /// True when every control centred inside this frame is disabled, so a tap
    /// cannot reach another enabled control. Unknown frames or a count that
    /// differs from the returned coverage keep the warning.
    private func coversOnlyDisabledControls(_ index: Int) -> Bool {
        guard let frame = elements[index].frame else { return false }
        let covered = elements.indices.filter { other in
            guard other != index, !["static_text", "image", "other"].contains(elements[other].role),
                  let center = elements[other].center else { return false }
            return center.x >= frame.x && center.x <= frame.x + frame.width
                && center.y >= frame.y && center.y <= frame.y + frame.height
        }
        return covered.count == elements[index].coversControls
            && covered.allSatisfy { elements[$0].enabled == false }
    }

    /// True when an editable field on screen still contains the text.
    func editableFieldShows(text: String) -> Bool {
        let needle = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return false }
        return elements.contains { element in
            Self.isEditableRole(element.role)
                && [element.value, element.label].contains { $0?.contains(needle) == true }
        }
    }

    /// Labels and static texts on screen, trimmed and de-duplicated.
    var readableTexts: [String] {
        elements.compactMap { element -> String? in
            guard element.type != "XCUIElementTypeKey",
                  let text = element.label.flatMap(Self.displayText),
                  !text.hasPrefix("__vc") else { return nil }
            return text
        }
        .reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
    }

    var hasOCRConfirmedEditableField: Bool {
        elements.contains { $0.ocrConfirmed && Self.isEditableRole($0.role) }
    }

    var hasOCRConfirmedControls: Bool {
        elements.contains { $0.ocrConfirmed && $0.position != nil }
    }

    func isOCRConfirmedPosition(x: Int64, y: Int64) -> Bool {
        elements.contains { element in
            guard element.ocrConfirmed, case .object(let position)? = element.position,
                  case .integer(let px)? = position["x_norm"],
                  case .integer(let py)? = position["y_norm"] else { return false }
            return abs(px - x) <= 30 && abs(py - y) <= 30
        }
    }

    private static func position(in object: [String: JSONValue]) -> JSONValue? {
        guard case .integer(let x)? = object["x_norm"],
              case .integer(let y)? = object["y_norm"],
              (0...1000).contains(x), (0...1000).contains(y) else { return nil }
        return .object(["x_norm": .integer(x), "y_norm": .integer(y)])
    }
}
