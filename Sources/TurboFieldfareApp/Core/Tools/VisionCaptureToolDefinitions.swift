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
    static let historyReadName = "task_history_read"
    /// How guidance names a coordinate action: the exact call the parser accepts.
    static let coordinateTapCall =
        "visioncapture_navigate with action \"tap_coordinates\", x_norm, y_norm and intent"
    static let visualClickCall =
        "visioncapture_navigate with action \"computer_use_click\", x_norm, y_norm and intent"
    static let typeAtPositionCall =
        "visioncapture_navigate with action \"type\", text, x_norm and y_norm"

    static let all: [AppToolDefinition] = [
        AppToolDefinition(
            name: navigateName,
            description: "Perform one observable QA step in the configured simulator app.",
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
                            .string("tap_coordinates"),
                            .string("computer_use_click"),
                            .string("set_boolean"),
                            .string("type"),
                            .string("back"),
                            .string("swipe"),
                        ]),
                        "description": .string(
                            "Before the first result, use launch, observe, or screenshot. Later copy from allowed_next. Screenshot is read-only; do not repeat an unchanged observe."),
                    ]),
                    "direction": .object([
                        "type": .string("string"),
                        "enum": .array([.string("up"), .string("down"), .string("left"), .string("right")]),
                        "description": .string(
                            "For swipe only. Copy a direction from can_swipe."),
                    ]),
                    "target": .object([
                        "type": .string("string"),
                        "minLength": .integer(1),
                        "description": .string(
                            "For tap and set_boolean. For type, use a current field ID when one exists. Omit it only when a current screenshot shows that the intended field is already focused."),
                    ]),
                    "x_norm": .object([
                        "type": .string("integer"),
                        "minimum": .integer(0),
                        "maximum": .integer(1000),
                        "description": .string(
                            "For tap_coordinates, computer_use_click, or type at an ocr-confirmed field. X position on the screen, from 0 at the left to 1000 at the right."),
                    ]),
                    "y_norm": .object([
                        "type": .string("integer"),
                        "minimum": .integer(0),
                        "maximum": .integer(1000),
                        "description": .string(
                            "For tap_coordinates, computer_use_click, or type at an ocr-confirmed field. Y position on the screen, from 0 at the top to 1000 at the bottom."),
                    ]),
                    "intent": .object([
                        "type": .string("string"),
                        "minLength": .integer(1),
                        "description": .string(
                            "Briefly name the purpose. Required for tap_coordinates and computer_use_click; optional for other actions."),
                    ]),
                    "desired_state": .object([
                        "type": .string("boolean"),
                        "description": .string(
                            "For set_boolean only. Copy an allowed_desired_states value."),
                    ]),
                    "text": .object([
                        "type": .string("string"),
                        "minLength": .integer(1),
                        "description": .string(
                            "For type only. Text is appended. A target is optional only for a field already focused in the current screenshot."),
                    ]),
                ]),
                "required": .array([.string("action")]),
            ])),
    ]
}
