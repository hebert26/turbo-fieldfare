import Foundation
import TurboFieldfare

public enum VisionCaptureAgentProfile {
    /// VisionCapture's checked-in default MCP endpoint is
    /// http://127.0.0.1:8766/mcp. The proof remains loopback-only.
    public static let mcpPort = 8_766
    /// Bounds consecutive invisible output only when tool thinking is disabled.
    /// Thinking-enabled turns use Stop, cancellation, and the existing generation
    /// and remaining-context limits. This token budget does not detect loops.
    public static let maximumConsecutiveInvisibleTokens = 2_048
}

enum VisionCaptureToolDefinitions {
    static let navigateName = "visioncapture_navigate"

    static let all: [AppToolDefinition] = [
        AppToolDefinition(
            name: navigateName,
            description: """
            Choose one simple app-navigation action. The host privately handles the configured
            target, observation, sessions, execution evidence, and the exact VisionCapture request.
            Use selectors and roles only from the latest sanitized available_actions. For typing,
            use the exact selector and selector_kind from available_text_fields.
            Observe once to obtain current facts, then choose an available action or answer; do
            not repeat observe while the sanitized screen and choices are unchanged. Use screenshot
            when a complex form, selection state, completion, or navigation is unclear from text. It is read-only, requires installed
            image support, and retires old cache handles. Screenshot pixels may help match a control to a latest
            offered position; submit only that action's exact selector and role. Mutations remain accessibility-only:
            no pointer actions, Computer Use, recordings, or visual bypass. Native iOS system-alert buttons remain simple tap
            choices with role system_alert_button; the host privately uses VisionCapture's
            existing guarded system-alert route.
            """,
            parameters: .object([
                "type": .string("object"),
                "additionalProperties": .bool(false),
                "properties": .object([
                    "action": .object([
                        "type": .string("string"),
                        "enum": .array([
                            .string("launch"),
                            .string("observe"),
                            .string("screenshot"),
                            .string("tap"),
                            .string("set_boolean"),
                            .string("type"),
                            .string("back"),
                        ]),
                        "description": .string(
                            "One high-level navigation choice. Observe is a read-only refresh, not a waiting loop: after it returns unchanged current choices, choose an action or answer instead of observing again."),
                    ]),
                    "selector": .object([
                        "type": .string("string"),
                        "minLength": .integer(1),
                        "description": .string(
                            "For tap or set_boolean, copy the exact selector from available_actions. For type, copy it from available_text_fields. The current sanitized result does not publish focused-field proof, so do not omit a type selector."),
                    ]),
                    "selector_kind": .object([
                        "type": .string("string"),
                        "enum": .array([.string("label"), .string("identifier"), .string("placeholder")]),
                        "description": .string(
                            "For a named type action, copy the exact selector_kind paired with selector in available_text_fields."),
                    ]),
                    "role": .object([
                        "type": .string("string"),
                        "minLength": .integer(1),
                        "description": .string(
                            "Required for set_boolean. For tap, copy the exact role or omit it only when the exact selector identifies one action in the latest available_actions or its still-permitted confirming_action; the host then uses that offered role. After a correction, include the explicit role. A native alert button uses system_alert_button."),
                    ]),
                    "desired_state": .object([
                        "type": .string("boolean"),
                        "description": .string(
                            "Required only for set_boolean."),
                    ]),
                    "text": .object([
                        "type": .string("string"),
                        "minLength": .integer(1),
                        "description": .string(
                            "Required only for type. Inserts text at the field's current cursor or selection without clearing existing content. Repeating text may duplicate it. A missing reported value does not mean the field is empty."),
                    ]),
                ]),
                "required": .array([.string("action")]),
            ])),
    ]
}
