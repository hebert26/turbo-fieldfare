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
            Choose an operation from allowed_next. For tap, set_boolean, or type, copy the target ID
            from a choice in the latest packet. IDs expire when a new packet arrives.
            For swipe, choose a direction from can_swipe. Direction is finger movement. The host
            reads the screen after one gesture; a submitted gesture does not prove its intended effect.
            Observe once to obtain current facts, then choose an available action or answer; do
            not repeat observe while the sanitized screen and choices are unchanged. Use screenshot
            when a complex form, selection state, completion, or navigation is unclear from text. It is read-only, requires installed
            image support, and retires old choices. The host then reads accessibility facts and returns current
            target IDs when permitted. Image and read are sequential, with visual agreement unverified.
            Offered positions belong to the later read; submit only a current target ID. Mutations remain accessibility-only:
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
                            .string("swipe"),
                        ]),
                        "description": .string(
                            "One high-level navigation choice. Observe is a read-only refresh, not a waiting loop: after it returns unchanged current choices, choose an action or answer instead of observing again."),
                    ]),
                    "direction": .object([
                        "type": .string("string"),
                        "enum": .array([.string("up"), .string("down"), .string("left"), .string("right")]),
                        "description": .string(
                            "Required only for swipe. Copy a direction from can_swipe. This is finger movement, with no target, distance, or coordinates."),
                    ]),
                    "target": .object([
                        "type": .string("string"),
                        "minLength": .integer(1),
                        "description": .string(
                            "Required for tap, set_boolean, and type. Copy the exact id from the latest choices. Never reuse an ID from an older packet or substitute a label."),
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
