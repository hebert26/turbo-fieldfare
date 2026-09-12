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
    }

    private let elements: [Element]
    private let navigationFacts: [String]

    init(elements: [JSONValue], navigation: [[String: JSONValue]] = []) {
        self.elements = elements.compactMap { value in
            guard case .object(let object) = value,
                  object["visible"] == .bool(true),
                  let role = Self.text(object["role"]) else { return nil }
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
                placeholder: Self.observedPlaceholder(metadata: metadata, type: type),
                value: valueStatus == nil || valueStatus == "available" ? rawValue : nil,
                valueStatus: valueStatus == "available" && rawValue == nil ? "unavailable" : valueStatus,
                enabled: Self.boolean(object["enabled"]),
                selected: Self.boolean(object["selected"]),
                position: Self.position(in: object))
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

    /// Observation suggests a target to validate, never permission to tap it.
    func tapCandidates(excluding controls: [(selector: String, role: String)]) -> [TapCandidate] {
        let represented = Set(controls.compactMap {
            matchingIndex(selector: $0.selector, role: $0.role)
        })
        return elements.indices.compactMap { index in
            let element = elements[index]
            guard !represented.contains(index), element.enabled == true,
                  ["button", "cell", "tab", "link"].contains(element.role) else { return nil }
            // Prefer an exact identifier, then an exact label. Do not normalize
            // a name into the contract or use a synthetic element ID.
            for selector in [element.identifier, element.label].compactMap({ $0 }) {
                guard !selector.isEmpty, selector.utf8.count <= 160,
                      selector != "[REDACTED]",
                      selector.utf8.elementsEqual(
                        selector.trimmingCharacters(in: .whitespacesAndNewlines).utf8),
                      matchingIndex(selector: selector, role: element.role) == index else { continue }
                // Cold name dispatch has no selector-kind argument. Reject a
                // name shared by any other returned element, even another role.
                let namedElements = elements.filter {
                    $0.identifier?.utf8.elementsEqual(selector.utf8) == true
                        || $0.label?.utf8.elementsEqual(selector.utf8) == true
                }
                guard namedElements.count == 1 else { continue }
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
            let content = [element.label, element.value]
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
            if element.role == "segmented_item" {
                // Preserve independent observed state without joining rounded
                // positions to a private action selector.
                if let selected = element.selected { line += " selected=\(selected)" }
                if let enabled = element.enabled { line += " enabled=\(enabled)" }
            } else {
                if element.selected == true { line += " selected" }
                if element.enabled == false { line += " disabled" }
            }
            if !lines.contains(line) { lines.append(line) }
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

    private static func observedPlaceholder(metadata: [String: JSONValue]?, type: String?) -> String? {
        guard let metadata, let type, metadata["type"] == .string(type),
              metadata["enabled"] == .bool(true), metadata["visible"] == .bool(true),
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

    private static func boolean(_ value: JSONValue?) -> Bool? {
        guard case .bool(let boolean) = value else { return nil }
        return boolean
    }

    private static func position(in object: [String: JSONValue]) -> JSONValue? {
        guard case .integer(let x)? = object["x_norm"],
              case .integer(let y)? = object["y_norm"],
              (0...1000).contains(x), (0...1000).contains(y) else { return nil }
        return .object(["x_norm": .integer(x), "y_norm": .integer(y)])
    }
}
