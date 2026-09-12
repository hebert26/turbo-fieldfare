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
                            "For tap, set_boolean, and type only. Copy a current choice ID and use its listed operation. A new packet expires it; request screenshot first if required."),
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
                            "For type only. Text is appended. To replace a nonempty value, tap an offered clear control first."),
                    ]),
                ]),
                "required": .array([.string("action")]),
            ])),
    ]
}
