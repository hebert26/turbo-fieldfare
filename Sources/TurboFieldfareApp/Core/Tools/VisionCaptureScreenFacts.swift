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
        let role: String
        let type: String?
        let identifier: String?
        let label: String?
        let value: String?
        let enabled: Bool?
        let selected: Bool?
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
            return Element(
                role: role,
                type: type,
                identifier: Self.rawText(object["identifier"]),
                label: label == Self.rawText(object["element_id"]) ? nil : label,
                value: role == "secure_text_field" || type == "XCUIElementTypeSecureTextField"
                    ? nil : Self.rawText(object["value"]),
                enabled: Self.boolean(object["enabled"]),
                selected: Self.boolean(object["selected"]))
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
        selector: String, role: String, selectorKind: String? = nil
    ) -> [String: JSONValue] {
        guard let index = matchingIndex(
            selector: selector, role: role, selectorKind: selectorKind) else { return [:] }
        let element = elements[index]
        var properties: [String: JSONValue] = [:]
        if let label = element.label.flatMap(Self.displayText), label != selector {
            properties["label"] = .string(label)
        }
        if let selected = element.selected { properties["selected"] = .bool(selected) }
        if let enabled = element.enabled { properties["enabled"] = .bool(enabled) }
        if let value = element.value { properties["value"] = .string(value) }
        return properties
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
        // Include the exact current choices in the human summary without changing their arrays.
        for choice in availableActions + editableFields {
            guard case .object(let object) = choice,
                  let role = Self.text(object["role"]),
                  let selector = Self.rawText(object["selector"]),
                  let name = Self.text(object["label"]) ?? Self.displayText(selector) else { continue }
            let selectorKind = Self.text(object["selector_kind"])
            if selectorKind != "placeholder",
               let index = matchingIndex(selector: selector, role: role, selectorKind: selectorKind) {
                excluded.insert(index)
            }
            let value = role == "secure_text_field" ? nil : Self.text(object["value"])
            let content = [name, value].compactMap { $0 }
                .reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
                .joined(separator: ": ")
            var line = "[\(role)] \(content)"
            if object["selected"] == .bool(true) { line += " selected" }
            if object["enabled"] == .bool(false) { line += " disabled" }
            if !lines.contains(line) { lines.append(line) }
        }
        for (index, element) in elements.enumerated() {
            guard !excluded.contains(index), element.type != "XCUIElementTypeKey" else { continue }
            let content = [element.label, element.value]
                .compactMap { $0.flatMap(Self.displayText) }
                .reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
                .joined(separator: ": ")
            guard !content.isEmpty else { continue }
            var line = "[\(element.role)] \(content)"
            if element.selected == true { line += " selected" }
            if element.enabled == false { line += " disabled" }
            if !lines.contains(line) { lines.append(line) }
        }
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }

    private func matchingIndex(
        selector: String, role: String, selectorKind: String? = nil
    ) -> Int? {
        func exactMatch(_ candidate: String?) -> Bool {
            candidate?.utf8.elementsEqual(selector.utf8) == true
        }
        let matches = elements.indices.filter { index in
            let element = elements[index]
            guard element.role == role else { return false }
            switch selectorKind {
            case "identifier": return exactMatch(element.identifier)
            case "label": return exactMatch(element.label)
            default: return exactMatch(element.identifier) || exactMatch(element.label)
            }
        }
        return matches.count == 1 ? matches.first : nil
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
}
