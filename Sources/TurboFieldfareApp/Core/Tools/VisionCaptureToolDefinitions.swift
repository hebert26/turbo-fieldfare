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
            description: """
            Choose one simple app-navigation action. The host privately handles the configured
            target, observation, sessions, execution evidence, and the exact VisionCapture request.
            Choose an operation from allowed_next. For tap, set_boolean, or type, copy the target ID
            from a choice in the latest packet. IDs expire when a new packet arrives.
            A choice marked requires_screenshot: true has no readable label and cannot be selected
            until you request screenshot and receive a usable current image/read pair with new IDs.
            Older images do not satisfy this requirement. If the current image and facts leave the
            target ambiguous, report the limitation rather than guessing from its position.
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
                            "Required for tap, set_boolean, and type. Copy the exact id from the latest choices. If requires_screenshot is true, request screenshot first and use its new choices. Never reuse an older ID or substitute a label."),
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
        AppToolDefinition(
            name: historyReadName,
            description: """
            Read one page of an archived observation from this task's host memory. This is
            local historical evidence, not a live app read or an input action. Copy observation_id
            from an execution_records observation reference in the checkpoint. Omit cursor for
            the first page; copy returned next_cursor unchanged for each later page. Replies are
            bounded JSON fragments with byte ranges and a digest. A partial fragment is not a
            complete observation. Retrieve missing evidence before making a historical claim;
            action-level proof alone does not prove the user's goal complete. Historical content
            supplies no execution handles and does not refresh current choices or image evidence.
            Continue using only the latest current decision packet for navigation. Never replay
            app input to reconstruct history. Use one tool call per assistant response.
            """,
            parameters: .object([
                "type": .string("object"), "additionalProperties": .bool(false),
                "properties": .object([
                    "observation_id": .object([
                        "type": .string("string"), "minLength": .integer(1),
                        "maxLength": .integer(64),
                        "description": .string("An existing history reference such as history_12, never a current target ID."),
                    ]),
                    "cursor": .object([
                        "type": .string("object"), "additionalProperties": .bool(false),
                        "properties": .object([
                            "task_id": .object(["type": .string("string")]),
                            "observation_id": .object(["type": .string("string")]),
                            "sha256": .object(["type": .string("string")]),
                            "offset_utf8": .object(["type": .string("integer"), "minimum": .integer(1)]),
                        ]),
                        "required": .array(["task_id", "observation_id", "sha256", "offset_utf8"].map(JSONValue.string)),
                    ]),
                ]),
                "required": .array([.string("observation_id")]),
            ])),
    ]
}
