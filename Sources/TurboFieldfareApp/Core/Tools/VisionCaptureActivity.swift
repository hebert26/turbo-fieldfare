import Foundation
import TurboFieldfare

/// Display-only projection of VisionCapture's public proof. It never grants
/// recovery or changes whether the host stops, observes, or submits an action.
public struct VisionCaptureServerOutcome: Equatable, Sendable {
    public let verdict: String?
    public let reason: String?
    public let reasonCode: String?
    public let dispatchAttempted: Bool?

    public var description: String {
        var text = "VisionCapture: \(verdict ?? "error")"
        if let reasonCode { text += " (\(reasonCode))" }
        if let reason { text += ". \(reason)" }
        if let dispatchAttempted {
            text += dispatchAttempted
                ? " · Dispatch attempted" : " · No dispatch attempted"
        }
        return text
    }
}

public struct AppAgentActivity: Identifiable, Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case mcpRequest
        case localProposal
        case modelInput(attempt: Int, isFormatCorrection: Bool)
        case modelResult(callID: String, toolName: String, imageCount: Int)
        /// Host display status, never a tool result or model instruction.
        case contextCompaction
        case generationRecovery
    }

    public enum Status: Equatable, Sendable {
        case dispatching
        case succeeded
        case recoverablePreDispatchRefusal(VisionCaptureServerOutcome)
        case serverOutcome(VisionCaptureServerOutcome)
        case localFailure(reason: String)
        case cancelled
        case notSent(reason: String)
    }

    public let id: UUID
    public let kind: Kind
    public let body: String
    public var status: Status
    /// Host request elapsed time for display and measurement only. It is never
    /// copied into a tool result or the model conversation.
    public var elapsedSeconds: Double?
    /// Bounded display excerpt of this exact MCP response, not model context.
    public var responseBody: String?
    /// Display-only thumbnail, never included in a model tool result.
    public var screenshotPreview: AppImageAttachment?
    public var screenshotPreviewUnavailable: String?

    init(
        id: UUID,
        kind: Kind,
        body: String,
        status: Status,
        elapsedSeconds: Double? = nil
    ) {
        self.id = id
        self.kind = kind
        self.body = body
        self.status = status
        self.elapsedSeconds = elapsedSeconds
    }
}

enum VisionCaptureActivityEvent: Sendable {
    case outgoingRequest(id: UUID, arguments: JSONValue)
    case incomingResponse(id: UUID, excerpt: String)
    case modelResult(callID: String, toolName: String, excerpt: String, imageCount: Int)
    case requestStatus(
        id: UUID,
        status: AppAgentActivity.Status,
        elapsedSeconds: Double)
    case localRejection(id: UUID, call: AppToolCall, reason: String)
    case screenshot(id: UUID, image: AppImageAttachment)
    case generationRecovery(id: UUID, text: String, status: AppAgentActivity.Status)
}

extension AppAgentActivity {
    /// A byte-bounded prefix must not end inside a UTF-8 scalar. At most three
    /// continuation bytes can precede the cut; the original string stays shared.
    static func displayPrefix(_ content: String, maximumUTF8Bytes: Int) -> String {
        let bytes = content.utf8
        guard bytes.count > maximumUTF8Bytes else { return content }
        var end = bytes.index(bytes.startIndex, offsetBy: maximumUTF8Bytes)
        for _ in 0..<3 {
            guard end > bytes.startIndex, bytes[end] & 0xC0 == 0x80 else { break }
            bytes.formIndex(before: &end)
        }
        return String(decoding: bytes[..<end], as: UTF8.self)
    }

    /// The inference result stays untouched. Only this display copy is bounded.
    static func modelResultExcerpt(_ content: String) -> String {
        let maximumBytes = 16 * 1_024
        var display = content
        // Parse only the complete, bounded input, never its truncated prefix.
        if content.utf8.count <= maximumBytes,
           let value = try? JSONDecoder().decode(JSONValue.self, from: Data(content.utf8)) {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            if let data = try? encoder.encode(value) {
                display = String(decoding: data, as: UTF8.self)
            }
        }
        guard display.utf8.count > maximumBytes else { return display }
        let marker = "\n[Display truncated at 16 KiB. Original model input is unchanged.]"
        return displayPrefix(display, maximumUTF8Bytes: maximumBytes - marker.utf8.count) + marker
    }
}
