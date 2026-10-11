import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// The loop's alert press asks VisionCapture for the screen after the press
/// (ADR 0054 amendment of 9 October). The reply's after-screen replaces the
/// loop's own alert read only when its alert fact proves the alert state.
@Suite struct VisionCaptureAlertPressAfterScreenTests {
    private let pressed = "digest-1"

    private func alertFact(digest: String, extra: [String: JSONValue] = [:]) -> JSONValue {
        var fact: [String: JSONValue] = [
            "present": .bool(true),
            "title": .string("Allow notifications?"),
            "content_digest": .string(digest),
            "buttons": .array([
                .object(["label": .string("Allow"), "enabled": .bool(true), "visible": .bool(true)]),
                .object(["label": .string("Don't Allow"), "enabled": .bool(true), "visible": .bool(true)]),
            ]),
        ]
        fact.merge(extra) { _, new in new }
        return .object(fact)
    }

    /// A press reply: the payload's after_screen (and, like the host, the same
    /// after_screen again as a text block).
    private func reply(afterScreen: [String: JSONValue]?, textCopy: Bool = true) throws -> JSONValue {
        var payload: [String: JSONValue] = [
            "system_alert_tap": .object(["submission": .string("submitted_once")]),
        ]
        var content: [JSONValue] = [.object(["type": .string("text"), "text": .string("{\"system_alert_tap\":{}}")])]
        if let afterScreen {
            payload["after_screen"] = .object(afterScreen)
            if textCopy {
                content.append(.object(["type": .string("text"),
                                        "text": .string(try JSONValue.object(["after_screen": .object(afterScreen)]).encoded())]))
            }
        }
        return .object(["content": .array(content), "payload": .object(payload)])
    }

    private func proof(_ reply: JSONValue, deliveryUnknown: Bool = false) -> VisionCaptureToolLoop.AfterScreenAlertProof? {
        VisionCaptureToolLoop.afterScreenAlertProof(in: reply, pressedDigest: pressed, deliveryUnknown: deliveryUnknown)
    }

    @Test
    func alertPressAsksForTheScreenAfter() {
        let parameters = VisionCaptureAlertPress.parameters(button: "Allow", digest: pressed)
        #expect(parameters["include_after_screen"] == .bool(true))
        #expect(parameters["system_alert_button"] == .string("Allow"))
        #expect(parameters["system_alert_digest"] == .string(pressed))
        #expect(parameters.count == 3)
    }

    @Test
    func theSameAlertInTheAfterScreenReplacesTheRead() throws {
        let screen: [String: JSONValue] = ["total_elements": .integer(2), "system_alert": alertFact(digest: pressed)]
        #expect(proof(try reply(afterScreen: screen)) == .sameAlert)
        // The text copy of the after-screen is the same fact, not a conflicting one.
        #expect(proof(try reply(afterScreen: screen, textCopy: false)) == .sameAlert)
    }

    @Test
    func aGoneAlertFactReplacesTheRead() throws {
        let gone: [String: JSONValue] = ["system_alert": .object([
            "present": .bool(false), "reason_code": .string("SYSTEM_ALERT_NOT_PRESENT")])]
        #expect(proof(try reply(afterScreen: gone)) == .alertGone)
    }

    @Test
    func anAfterScreenWithoutAnAlertFactKeepsTheRead() throws {
        // The host leaves the fact out both when there is no alert and when its
        // alert probe failed, so the loop cannot tell them apart.
        #expect(proof(try reply(afterScreen: ["total_elements": .integer(5)])) == nil)
        #expect(proof(try reply(afterScreen: nil)) == nil)
    }

    @Test
    func anotherOrUnreadableAlertKeepsTheRead() throws {
        #expect(proof(try reply(afterScreen: ["system_alert": alertFact(digest: "digest-2")])) == nil)
        #expect(proof(try reply(afterScreen: ["system_alert": alertFact(
            digest: pressed, extra: ["owner_is_app_under_test": .bool(true)])])) == nil)
        #expect(proof(try reply(afterScreen: ["system_alert": alertFact(
            digest: pressed, extra: ["buttons": .array([])])])) == nil)
        for code in ["SYSTEM_ALERT_AMBIGUOUS", "SYSTEM_ALERT_UNREADABLE", "SYSTEM_ALERT_NOT_CHECKED"] {
            #expect(proof(try reply(afterScreen: ["system_alert": .object([
                "present": .bool(false), "reason_code": .string(code)])])) == nil, Comment(rawValue: code))
        }
        // Two different alert facts in one reply prove nothing.
        var two = try reply(afterScreen: ["system_alert": alertFact(digest: pressed)], textCopy: false).objectValue!
        two["content"] = .array([.object(["type": .string("text"), "text": .string(
            try JSONValue.object(["after_screen": .object(["system_alert": alertFact(digest: "digest-2")])]).encoded())])])
        #expect(proof(.object(two)) == nil)
    }

    @Test
    func unknownDeliveryKeepsTheRead() throws {
        // The alert-safety rule needs the loop's own read-only alert read.
        let screen: [String: JSONValue] = ["system_alert": alertFact(digest: pressed)]
        #expect(proof(try reply(afterScreen: screen), deliveryUnknown: true) == nil)
        let gone: [String: JSONValue] = ["system_alert": .object([
            "present": .bool(false), "reason_code": .string("SYSTEM_ALERT_NOT_PRESENT")])]
        #expect(proof(try reply(afterScreen: gone), deliveryUnknown: true) == nil)
    }
}
