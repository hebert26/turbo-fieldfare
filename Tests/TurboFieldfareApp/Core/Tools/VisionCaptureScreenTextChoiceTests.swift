import CryptoKit
import CoreGraphics
import Foundation
import ImageIO
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureScreenTextChoiceTests {
    /// A plain screenshot reply: the PNG block, then the metadata text with OCR.
    private func reply(_ ocr: JSONValue, width: Int64 = 1206, height: Int64 = 2622,
                       png: Data? = nil, udid: String = "DE8B571C-2234-498F-9FAC-71C96B614792") throws -> JSONValue {
        let metadata = try JSONValue.object([
            "udid": .string(udid),
            "width": .integer(width), "height": .integer(height), "ocr": ocr,
        ]).encoded()
        return .object(["content": .array([
            .object(["type": .string("image"), "mimeType": .string("image/png"),
                     "data": .string(png?.base64EncodedString() ?? "AAAA")]),
            .object(["type": .string("text"), "text": .string(metadata)]),
        ])])
    }

    private func block(_ text: String, _ x: Double, _ y: Double, _ w: Double, _ h: Double) -> JSONValue {
        .object(["text": .string(text), "confidence": .number(1),
                 "frame": .object(["x": .number(x), "y": .number(y), "width": .number(w), "height": .number(h)])])
    }

    @Test
    func ocrBlockBecomesATapChoiceAtItsCentre() throws {
        let read = VisionCaptureToolLoop.screenTextBlocks(in: try reply(.object([
            "blocks": .array([block("March", 342, 1600, 164, 39), block("  ", 10, 10, 5, 5)]),
        ])))
        #expect(read.unavailableReason == nil)
        #expect(read.blocks == [.init(text: "March", xNorm: 352, yNorm: 618)])
        let choice = try #require(VisionCaptureToolLoop.screenTextChoice(read.blocks[0], id: "c7").objectValue)
        #expect(choice["id"] == .string("c7"))
        #expect(choice["role"] == .string("text"))
        #expect(choice["label"] == .string("March (screen text)"))
        #expect(choice["operations"] == .array([.string("tap")]))
        #expect(choice["position"] == .object(["x_norm": .integer(352), "y_norm": .integer(618)]))
    }

    @Test
    func screenTextAlreadyNamedNearbyIsNotOfferedTwice() {
        let existing: [JSONValue] = [
            .object(["id": .string("c1"), "label": .string("Add"),
                     "position": .object(["x_norm": .integer(881), "y_norm": .integer(114)])]),
            .object(["id": .string("c2"), "label": .string("Cancel (ocr)"),
                     "position": .object(["x_norm": .integer(146), "y_norm": .integer(114)])]),
        ]
        let blocks: [VisionCaptureToolLoop.ScreenTextBlock] = [
            .init(text: "Add", xNorm: 879, yNorm: 115),
            .init(text: "Cancel", xNorm: 147, yNorm: 116),
            .init(text: "Add", xNorm: 500, yNorm: 500),
            .init(text: "March", xNorm: 352, yNorm: 618),
        ]
        #expect(VisionCaptureToolLoop.screenTextBlocksToOffer(blocks, knownElements: existing) == [
            .init(text: "Add", xNorm: 500, yNorm: 500),
            .init(text: "March", xNorm: 352, yNorm: 618),
        ])
    }

    @Test
    func tapOnScreenTextIsAPointerClickAtItsPositionWithoutANewScreenshot() {
        let click = VisionCaptureToolLoop.screenTextPointerClick(
            selectorKind: VisionCaptureToolLoop.screenTextSelectorKind, selector: "March", xNorm: 352, yNorm: 618)
        #expect(click?.x == 352 && click?.y == 618)
        #expect(click?.intent == "tap March")
        // Other choices keep their accessibility position path.
        #expect(VisionCaptureToolLoop.screenTextPointerClick(
            selectorKind: nil, selector: "March", xNorm: 352, yNorm: 618) == nil)
    }

    @Test
    func everyTextBlockIsOfferedWithNoLimit() throws {
        // 40 blocks, one of them already named by a choice nearby: 39 text choices.
        let blocks = (0..<40).map { i in block("Row \(i)", 100, Double(100 + i * 60), 200, 40) }
        let read = VisionCaptureToolLoop.screenTextBlocks(in: try reply(.object(["blocks": .array(blocks)])))
        #expect(read.blocks.count == 40)
        let existing: [JSONValue] = [.object([
            "label": .string("Row 0"),
            "position": .object(["x_norm": .integer(read.blocks[0].xNorm), "y_norm": .integer(read.blocks[0].yNorm)]),
        ])]
        let offered = VisionCaptureToolLoop.screenTextBlocksToOffer(read.blocks, knownElements: existing)
        #expect(offered.count == 39)
        #expect(offered.map(\.text) == (1..<40).map { "Row \($0)" })
    }

    @Test
    func ocrFailureGivesOneNoteAndNoChoices() throws {
        let failed = VisionCaptureToolLoop.screenTextBlocks(in: try reply(.object(["error": .string("Text recognition failed.")])))
        #expect(failed.blocks.isEmpty)
        #expect(VisionCaptureToolLoop.screenTextUnavailableNote(try #require(failed.unavailableReason))
            == "Screen text positions unavailable: Text recognition failed.")
        let empty = VisionCaptureToolLoop.screenTextBlocks(in: try reply(.object(["blocks": .array([])])))
        #expect(empty.blocks.isEmpty && empty.unavailableReason == "no text was found on the screen")
        let noOCR = VisionCaptureToolLoop.screenTextBlocks(in: .object(["content": .array([
            .object(["type": .string("text"), "text": .string("{\"width\":1206}")]),
        ])]))
        #expect(noOCR.unavailableReason == "the screenshot reply had no OCR result")
        #expect(VisionCaptureToolLoop.screenTextBlocks(in: .object([:])).unavailableReason != nil)
        let long = String(repeating: "x", count: 400)
        #expect(VisionCaptureToolLoop.screenTextBlocks(in: try reply(.object(["error": .string(long)])))
            .unavailableReason?.count == 160)
        #expect(VisionCaptureToolLoop.screenTextRefusalReason(code: long, reason: nil).count == 160)
    }

    @Test
    func refusedOrFailedOCRReadGivesOnlyTheHostsPlainReason() {
        let refused = VisionCaptureToolLoop.screenTextFailureReason(VisionCaptureAgentError.mcpOutcome(
            VisionCaptureServerOutcome(verdict: "failed",
                reason: "Device is quarantined after a failed dispatch. Retry after recovery.",
                reasonCode: "DEVICE_DISPATCH_QUARANTINED", dispatchAttempted: false)))
        #expect(refused == "DEVICE_DISPATCH_QUARANTINED: Device is quarantined after a failed dispatch")
        let thrown = VisionCaptureToolLoop.screenTextFailureReason(VisionCaptureAgentError.noProgress(
            "The screenshot was captured. Agent Mode stopped this unsupported observation path. No action was replayed."))
        #expect(thrown == "the host did not return screen text")
        #expect(VisionCaptureToolLoop.screenTextFailureReason(VisionCaptureAgentError.mcpUnavailable("socket closed"))
            == "the host did not return screen text")
        for note in [refused, thrown].map(VisionCaptureToolLoop.screenTextUnavailableNote) {
            #expect(note.hasPrefix("Screen text positions unavailable: "))
            #expect(!note.contains("Agent Mode stopped") && !note.contains("No action was replayed"))
        }
    }

    @Test
    func identityMismatchFromTheOCRReadIsNotTurnedIntoANote() {
        #expect(VisionCaptureToolLoop.isFatalScreenTextReadError(.returnedIdentityMismatch(fieldPath: "udid", refusalCode: nil)))
        #expect(VisionCaptureToolLoop.isFatalScreenTextReadError(.sessionIdentityMismatch))
        #expect(VisionCaptureToolLoop.isFatalScreenTextReadError(.unsupportedSystemInteraction("SYSTEM_ALERT_PRESENT")))
        #expect(!VisionCaptureToolLoop.isFatalScreenTextReadError(.mcpOutcome(VisionCaptureServerOutcome(
            verdict: "failed", reason: nil, reasonCode: "DEVICE_DISPATCH_QUARANTINED", dispatchAttempted: false))))
        #expect(!VisionCaptureToolLoop.isFatalScreenTextReadError(.mcpUnavailable("socket closed")))
    }

    @Test
    func prohibitedTextIsNotOffered() {
        var restrictions = AgentUserRestrictions()
        restrictions.apply("Never press the Delete button.")
        let blocks: [VisionCaptureToolLoop.ScreenTextBlock] = [
            .init(text: "Delete", xNorm: 689, yNorm: 486), .init(text: "Reopen", xNorm: 888, yNorm: 486),
        ]
        #expect(VisionCaptureToolLoop.screenTextBlocksToOffer(blocks, knownElements: [], restrictions: restrictions)
            == [.init(text: "Reopen", xNorm: 888, yNorm: 486)])
    }

    /// One accessibility read, as the loop's own observation code takes it.
    private func read(_ elements: [JSONValue], signature: String = "screen-text-test") -> VisionCaptureMCPResult {
        VisionCaptureMCPResult(
            value: .object([
                "view": .object(["signature_fine": .string(signature)]),
                "elements": .array(elements),
            ]),
            isError: false, refusalCode: nil, dispatchAttempted: nil,
            hasConflictingDispatchAttemptEvidence: false,
            isGuardedTargetRejectedBeforeSubmission: false,
            isStaleActionCapabilityBeforeDispatch: false,
            isSourceLayoutChangedBeforeRevalidation: false,
            isActionAuthorizationExpiredBeforeDispatch: false,
            isObservedTapTargetUnavailableBeforeDispatch: false,
            isDeliveredTransitionContinuation: false,
            isPointerPreCaptureFailureBeforeSubmission: false)
    }

    private func element(_ id: String, _ label: String, _ x: Int64, _ y: Int64, role: String = "button",
                         enabled: Bool = true, covers: Int64 = 0) -> JSONValue {
        .object([
            "element_id": .string(id), "role": .string(role), "type": .string("XCUIElementTypeButton"),
            "label": .string(label), "visible": .bool(true), "enabled": .bool(enabled),
            "x_norm": .integer(x), "y_norm": .integer(y), "frame_covers_controls": .integer(covers),
        ])
    }

    /// The screen-text labels of the screenshot packet the loop builds.
    private func screenTextLabels(
        _ loop: VisionCaptureToolLoop, read: VisionCaptureMCPResult, ocr: JSONValue,
        rejectedScreenText: [VisionCaptureToolLoop.ScreenTextBlock] = [],
        usedUpSwitches: [(selector: String, desiredState: Bool)] = []
    ) async throws -> [String] {
        let store = AppImageAttachmentStore(directoryURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("screen-text-test-\(UUID().uuidString)"))
        let image = try store.stage(data: Data("png".utf8), displayName: "screenshot.png")
        let packet = try await loop.screenshotPacketForTesting(
            read: read, screenTextReply: ocr, configuration: Self.configuration, images: [image],
            rejectedScreenText: rejectedScreenText, usedUpSwitches: usedUpSwitches)
        guard case .array(let choices)? = try JSONDecoder().decode(JSONValue.self, from: Data(packet.utf8))
            .objectValue?["choices"] else { return [] }
        return choices.compactMap { choice -> String? in
            guard case .string(let label)? = choice.objectValue?["label"],
                  label.hasSuffix(" (screen text)") else { return nil }
            return label
        }
    }

    @Test
    func withheldControlsTextIsNotOfferedInTheScreenshotPacket() async throws {
        // A disabled Save and a control that covers others, both also found by OCR.
        let screen = read([
            element("save", "Save", 872, 114, enabled: false),
            element("states", "Fixture states", 501, 952, covers: 3),
            element("agenda", "Agenda", 393, 941),
        ])
        // OCR boxes centred on (872, 114), (501, 952) and (352, 618) on a 1206 x 2622 image.
        let ocr = try reply(.object(["blocks": .array([
            block("Save", 1026, 278, 51, 42), block("Fixture states", 504, 2475, 200, 46),
            block("March", 342, 1600, 164, 39),
        ])]))
        let labels = try await screenTextLabels(VisionCaptureToolLoop(), read: screen, ocr: ocr)
        #expect(labels.contains("March (screen text)"))
        #expect(!labels.contains("Save (screen text)"))
        #expect(!labels.contains("Fixture states (screen text)"))
    }

    @Test
    func screenTextRejectedAtItsPositionIsNotOfferedAgain() async throws {
        let loop = VisionCaptureToolLoop()
        let screen = read([element("agenda", "Agenda", 393, 941)])
        let ocr = try reply(.object(["blocks": .array([
            block("March", 342, 1600, 164, 39), block("April", 350, 1662, 118, 34),
        ])]))
        #expect(try await screenTextLabels(loop, read: screen, ocr: ocr)
            == ["March (screen text)", "April (screen text)"])
        // The tap on "March" at (352, 618) was rejected before submission on this screen.
        let again = try await screenTextLabels(loop, read: screen, ocr: ocr,
            rejectedScreenText: [.init(text: "March", xNorm: 352, yNorm: 618)])
        #expect(again == ["April (screen text)"])
    }

    @Test
    func switchUsedUpThroughSetBooleanWithholdsItsScreenText() async throws {
        let loop = VisionCaptureToolLoop()
        // The leading space keeps the loop from offering a tap for the switch, as
        // for a switch published only as set_boolean: its own choice is not there.
        let screen = read([element("completed", " Completed", 500, 589, role: "switch")])
        // OCR box centred on (500, 589).
        let ocr = try reply(.object(["blocks": .array([block("Completed", 503, 1524, 200, 40)])]))
        #expect(try await screenTextLabels(loop, read: screen, ocr: ocr) == ["Completed (screen text)"])
        let usedUp = try await screenTextLabels(loop, read: screen, ocr: ocr,
            usedUpSwitches: [(selector: " Completed", desiredState: true)])
        #expect(usedUp.isEmpty)
    }

    private static let configuration = VisionCaptureAgentConfiguration(
        bundleIdentifier: "com.example.app",
        simulatorUDID: "00000000-0000-0000-0000-000000000001",
        modelDirectory: URL(fileURLWithPath: "/tmp/visioncapture-test-model.gturbo"))

    @Test
    func packetCarriesTheNoteWhenScreenTextIsUnavailable() throws {
        let packet = try JSONValue.object(["guidance": .string("Use the image."), "choices": .array([])]).encoded()
        let noted = try VisionCaptureToolLoop.addingScreenTextNote(
            to: packet, body: ["screen_text_unavailable": .string("DEVICE_DISPATCH_QUARANTINED: Device is quarantined")])
        #expect(noted.contains("Screen text positions unavailable: DEVICE_DISPATCH_QUARANTINED: Device is quarantined."))
        #expect(noted.contains("Use the image."))
        #expect(try VisionCaptureToolLoop.addingScreenTextNote(to: packet, body: ["screen_text": .array([])]) == packet)
    }

    // MARK: Screen text after actions

    private func textChoices(_ packet: String) throws -> [String] {
        guard case .array(let choices)? = try JSONDecoder().decode(JSONValue.self, from: Data(packet.utf8))
            .objectValue?["choices"] else { return [] }
        return choices.compactMap { choice -> String? in
            guard case .string(let label)? = choice.objectValue?["label"],
                  label.hasSuffix(" (screen text)") else { return nil }
            return label
        }
    }

    private func packetObject(_ packet: String) throws -> [String: JSONValue] {
        try JSONDecoder().decode(JSONValue.self, from: Data(packet.utf8)).objectValue ?? [:]
    }

    @Test
    func changedScreenAfterATapGetsScreenTextWithoutAnImage() async throws {
        let ocr = try reply(.object(["blocks": .array([block("2025", 768, 1509, 152, 53)])]))
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "screen-text-test",
            body: ["screen_changed": .bool(true)], screenTextReply: ocr, configuration: Self.configuration)
        #expect(result.readScreenText)
        #expect(try textChoices(result.packet) == ["2025 (screen text)"])
        let packet = try packetObject(result.packet)
        #expect(packet["image"] == nil)
        #expect(packet["observation"]?.objectValue?["current_image_evidence"] == .bool(false))
    }

    @Test
    func unchangedScreenGetsNoOCRRead() async throws {
        let ocr = try reply(.object(["blocks": .array([block("2025", 768, 1509, 152, 53)])]))
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "screen-text-test",
            body: ["screen_changed": .bool(false), "effect": .string("Same elements as before the action.")],
            screenTextReply: ocr, configuration: Self.configuration)
        #expect(!result.readScreenText)
        #expect(try textChoices(result.packet).isEmpty)
    }

    @Test
    func effectListOrNewSignatureTriggersTheReadWithoutScreenChanged() async throws {
        // Exam trace line 16: "New event" opened a sheet; the result had no screen_changed.
        let ocr = try reply(.object(["blocks": .array([block("2025", 768, 1509, 152, 53)])]))
        let byEffect = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "screen-text-test",
            body: ["effect": .string("appeared: Cancel, Add, New Event; disappeared: Calendar")],
            screenTextReply: ocr, configuration: Self.configuration)
        #expect(byEffect.readScreenText)
        #expect(try textChoices(byEffect.packet) == ["2025 (screen text)"])
        let bySignature = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: [:], screenTextReply: ocr, configuration: Self.configuration)
        #expect(bySignature.readScreenText)
    }

    @Test
    func backGetsTheReadOnlyWithACurrentFollowUpRead() async throws {
        let ocr = try reply(.object(["blocks": .array([block("2025", 768, 1509, 152, 53)])]))
        let withRead = try await VisionCaptureToolLoop().actionPacketForTesting(
            operation: "back", read: read([element("agenda", "Agenda", 393, 941)]),
            signatureBefore: "an-earlier-screen", body: [:], screenTextReply: ocr, configuration: Self.configuration)
        #expect(withRead.readScreenText)
        #expect(try textChoices(withRead.packet) == ["2025 (screen text)"])
        // Today's back makes no follow-up read: nothing current to bind words to.
        let withoutRead = try await VisionCaptureToolLoop().actionPacketForTesting(
            operation: "back", read: nil, signatureBefore: "an-earlier-screen", body: [:],
            screenTextReply: ocr, configuration: Self.configuration)
        #expect(!withoutRead.readScreenText)
    }

    @Test
    func unknownDeliveryGetsNoOCRRead() async throws {
        let ocr = try reply(.object(["blocks": .array([block("2025", 768, 1509, 152, 53)])]))
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true), "delivery_unknown": .bool(true)],
            screenTextReply: ocr, configuration: Self.configuration)
        #expect(!result.readScreenText)
        #expect(try textChoices(result.packet).isEmpty)
    }

    @Test
    func staticTextIsSkippedAndDuplicateNamedRowsKeepTheirScreenText() async throws {
        // "Date" is a host static text at (126, 308); two "Edit" rows share one name,
        // so the host offers neither: both keep their screen text.
        let date: JSONValue = .object([
            "element_id": .string("date"), "role": .string("static_text"), "label": .string("Date"),
            "visible": .bool(true), "x_norm": .integer(126), "y_norm": .integer(308),
        ])
        let ocr = try reply(.object(["blocks": .array([
            block("Date", 95, 785, 114, 46), block("2025", 768, 1509, 152, 53),
            block("Edit", 1050, 1000, 80, 40), block("Edit", 1050, 1200, 80, 40),
        ])]))
        let screen = read([date, element("edit-1", "Edit", 904, 389), element("edit-2", "Edit", 904, 465)])
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: screen, signatureBefore: "an-earlier-screen", body: ["screen_changed": .bool(true)],
            screenTextReply: ocr, configuration: Self.configuration)
        #expect(try textChoices(result.packet) == ["2025 (screen text)", "Edit (screen text)", "Edit (screen text)"])
        // The screenshot path follows the same rule.
        #expect(try await screenTextLabels(VisionCaptureToolLoop(), read: screen, ocr: ocr)
            == ["2025 (screen text)", "Edit (screen text)", "Edit (screen text)"])
    }

    @Test
    func failedOCRReadAfterAnActionGivesTheNote() async throws {
        let failed = try reply(.object(["error": .string("Text recognition failed.")]))
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: failed, configuration: Self.configuration)
        #expect(result.readScreenText)
        #expect(try textChoices(result.packet).isEmpty)
        guard case .string(let guidance)? = try packetObject(result.packet)["guidance"] else {
            Issue.record("the packet has no guidance"); return
        }
        #expect(guidance.contains("Screen text positions unavailable: Text recognition failed."))
    }

    @Test
    func alertRefusalOnTheAfterActionReadGivesTheNoteButStopsTheScreenshot() async throws {
        let alert = VisionCaptureAgentError.unsupportedSystemInteraction("SYSTEM_ALERT_PRESENT",
            outcome: VisionCaptureServerOutcome(verdict: "failed", reason: "A system alert is shown. Handle it first.",
                reasonCode: "SYSTEM_ALERT_PRESENT", dispatchAttempted: false))
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextError: alert, configuration: Self.configuration)
        guard case .string(let guidance)? = try packetObject(result.packet)["guidance"] else {
            Issue.record("the packet has no guidance"); return
        }
        #expect(guidance.contains("Screen text positions unavailable: SYSTEM_ALERT_PRESENT: A system alert is shown."))
        // An identity mismatch still stops the run after an action; the screenshot keeps the alert fatal.
        #expect(throws: VisionCaptureAgentError.self) {
            try VisionCaptureToolLoop.screenTextReadFailure(VisionCaptureAgentError.sessionIdentityMismatch, afterAction: true)
        }
        #expect(throws: VisionCaptureAgentError.self) {
            try VisionCaptureToolLoop.screenTextReadFailure(alert, afterAction: false)
        }
    }

    @Test
    func statusBarTextIsNotOffered() async throws {
        // The clock at y_norm 39 is system UI; a word at y_norm 60 belongs to the app.
        #expect(VisionCaptureToolLoop.screenTextBlocksToOffer([
            .init(text: "15:36", xNorm: 133, yNorm: 39), .init(text: "Inbox", xNorm: 500, yNorm: 60),
            .init(text: "Edge", xNorm: 500, yNorm: 50), .init(text: "Below", xNorm: 500, yNorm: 51),
        ], knownElements: []) == [.init(text: "Inbox", xNorm: 500, yNorm: 60), .init(text: "Below", xNorm: 500, yNorm: 51)])
        // The same through the after-action packet.
        let ocr = try reply(.object(["blocks": .array([block("15:36", 100, 79, 120, 46), block("Inbox", 440, 134, 120, 46)])]))
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: ocr, configuration: Self.configuration)
        #expect(try textChoices(result.packet) == ["Inbox (screen text)"])
    }

    @Test
    func newWordsAreOfferedAgainAfterAScreenTextTapChangedTheScreen() async throws {
        let loop = VisionCaptureToolLoop()
        let screen = read([element("agenda", "Agenda", 393, 941)])
        // The screenshot offers "March"; the tap on it changes the wheels.
        let before = try reply(.object(["blocks": .array([block("March", 342, 1600, 164, 39)])]))
        #expect(try await screenTextLabels(loop, read: screen, ocr: before) == ["March (screen text)"])
        let after = try reply(.object(["blocks": .array([
            block("March", 319, 1412, 195, 60), block("2025", 768, 1509, 152, 53),
        ])]))
        let result = try await loop.actionPacketForTesting(
            read: screen, signatureBefore: "screen-text-test", body: ["screen_changed": .bool(true)],
            screenTextReply: after, configuration: Self.configuration)
        #expect(try textChoices(result.packet) == ["March (screen text)", "2025 (screen text)"])
    }

    // MARK: Picker wheel hint

    /// Every OCR block (pixel frames, 1206 x 2622) of an open month and year wheel picker: a
    /// calendar form (hand check, 9 Oct 06:24:42, wheel/shots/062442-check1-wheels-open.json).
    private static let calendarWheelScreen: [(String, Double, Double, Double, Double)] = [
        ("06:24", 148.37, 72.41, 152.18, 53.35),
        ("Cancel", 94.95, 281.37, 163.92, 43.22),
        ("Add", 1008.17, 278.21, 102.72, 45.73),
        ("New Event", 49.32, 429.84, 502.45, 85.47),
        ("Event title", 94.79, 611.77, 228.91, 45.55),
        ("Date", 95.11, 785.08, 114.13, 45.73),
        ("15 Jan 2024", 566.86, 785.0, 285.33, 46.0),
        ("12:00", 935.47, 783.91, 137.79, 51.88),
        ("January 2024 v", 216.85, 968.0, 384.25, 57.17),
        ("October", 350.01, 1196.67, 209.24, 26.68),
        ("November", 338.59, 1250.02, 281.53, 38.11),
        ("December", 334.62, 1321.52, 289.47, 55.18),
        ("January", 317.98, 1418.74, 249.81, 59.1),
        ("February", 334.6, 1512.17, 247.66, 58.8),
        ("March", 342.27, 1600.09, 163.85, 39.22),
        ("April", 346.2, 1661.62, 121.74, 34.3),
        ("2021", 772.3, 1192.86, 144.57, 34.3),
        ("2022", 768.43, 1245.98, 152.3, 42.38),
        ("2023", 771.97, 1321.52, 149.02, 55.17),
        ("2024", 757.01, 1409.9, 171.34, 65.17),
        ("2025", 768.49, 1509.17, 152.18, 53.35),
        ("2026", 768.49, 1593.02, 152.18, 45.73),
        ("2027", 764.69, 1661.62, 140.76, 34.3)
    ]
    /// The same in a task form's due-date picker (walk, 9 Oct 06:47:35, second-app/walk/shots/064735-w1-wheels-open.json).
    private static let taskWheelScreen: [(String, Double, Double, Double, Double)] = [
        ("06:47", 148.37, 49.54, 148.37, 49.54),
        ("‹ MuckCalendar", 34.24, 102.9, 273.92, 34.3),
        ("Cancel", 94.98, 281.52, 163.85, 42.93),
        ("Save", 992.95, 282.02, 114.13, 45.73),
        ("Create Task", 49.46, 430.65, 559.25, 80.35),
        ("New Task", 98.54, 588.9, 232.82, 53.17),
        ("Title", 95.11, 727.91, 102.72, 49.54),
        ("Product", 94.66, 882.4, 187.33, 53.07),
        ("Priority", 94.93, 1039.86, 167.75, 54.47),
        ("Medium", 874.63, 1038.4, 232.83, 49.77),
        ("Due", 98.91, 1223.35, 91.31, 41.92),
        ("10 Oct 2026", 563.05, 1215.72, 289.14, 49.54),
        ("06:46", 932.03, 1215.57, 140.87, 49.85),
        ("October 2026 v", 216.85, 1402.0, 388.05, 46.2),
        ("No", 98.91, 1429.14, 64.68, 45.73),
        ("July", 346.2, 1612.07, 110.33, 49.54),
        ("August", 327.18, 1680.67, 205.44, 45.73),
        ("September", 330.7, 1751.61, 312.53, 63.92),
        ("October", 323.24, 1843.96, 251.36, 58.34),
        ("November", 334.79, 1943.63, 289.14, 49.54),
        ("December", 342.4, 2031.29, 277.72, 41.92),
        ("January", 345.82, 2093.94, 206.19, 42.39),
        ("2023", 764.41, 1626.07, 152.72, 36.8),
        ("2024", 768.49, 1676.86, 152.18, 45.73),
        ("2025", 768.49, 1753.08, 152.18, 53.35),
        ("2026", 756.99, 1844.29, 175.18, 61.5),
        ("2027", 768.37, 1943.25, 152.43, 50.31),
        ("2028", 768.24, 2026.61, 152.69, 47.46),
        ("2029", 764.69, 2092.26, 152.18, 34.3)
    ]
    /// The words of an open calendar day grid with their positions (rerun 2, 9 Oct, trace line 38).
    private static let dayGridScreen: [(String, Int64, Int64)] = [
        ("New Event", 249, 180),
        ("Dentist visit", 191, 241),
        ("Date", 126, 308),
        ("12:00", 833, 309),
        ("January 2024 >", 338, 380),
        ("MON TUE WED THU FRI", 418, 428),
        ("1", 208, 465),
        ("8", 210, 515),
        ("15", 210, 568),
        ("22", 210, 621),
        ("29", 211, 672),
        ("2", 315, 464),
        ("9", 315, 516),
        ("16", 317, 568),
        ("23", 317, 621),
        ("30", 317, 672),
        ("3", 421, 464),
        ("10", 423, 515),
        ("17", 420, 569),
        ("24", 423, 621),
        ("31", 423, 672),
        ("4", 528, 464),
        ("11", 528, 516),
        ("18", 527, 568),
        ("25", 528, 621),
        ("5", 632, 464),
        ("12", 634, 516),
        ("19", 634, 568),
        ("26", 634, 620),
        ("<", 754, 379),
        ("SAT SUN", 789, 428),
        ("13", 738, 516),
        ("20", 741, 568),
        ("27", 738, 621),
        ("7", 845, 464),
        ("14", 845, 516),
        ("21", 845, 569),
        ("28", 845, 620)
    ]
    private static let monthNames = ["January", "February", "March", "April", "May", "June", "July", "August",
                                     "September", "October", "November", "December"]

    private func ocrReply(_ rows: [(String, Double, Double, Double, Double)]) throws -> JSONValue {
        try reply(.object(["blocks": .array(rows.map { block($0.0, $0.1, $0.2, $0.3, $0.4) })]))
    }

    /// Wheel lines in a packet: the one-wheel and the several-wheel line both end so.
    private func wheelNoteCount(_ packet: String) throws -> Int {
        guard case .string(let text)? = try packetObject(packet)["guidance"] else { return 0 }
        return text.components(separatedBy: "do not swipe these rows.").count - 1
    }

    private func afterTap(ocr: JSONValue, elements: [JSONValue]) async throws -> String {
        try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read(elements), signatureBefore: "an-earlier-screen", body: ["screen_changed": .bool(true)],
            screenTextReply: ocr, configuration: Self.configuration).packet
    }

    @Test
    func realWheelColumnsFireOnce() async throws {
        #expect(VisionCaptureToolLoop.pickerWheelNote == "These look like picker wheel rows: tap the row you want; if it is not visible, tap the end row nearest to it; do not swipe these rows.")
        for screen in [Self.calendarWheelScreen, Self.taskWheelScreen] {
            let words = VisionCaptureToolLoop.screenTextBlocks(in: try ocrReply(screen)).blocks
            let months = words.filter { Self.monthNames.contains($0.text) }
            let years = words.filter { $0.text.count == 4 && Int($0.text) != nil }
            #expect(months.count == 7 && years.count == 7)
            #expect(!VisionCaptureToolLoop.looksLikePickerWheel(months).isEmpty)
            #expect(!VisionCaptureToolLoop.looksLikePickerWheel(years).isEmpty)
            let packet = try await afterTap(ocr: try ocrReply(screen), elements: [element("agenda", "Agenda", 393, 941)])
            #expect(try wheelNoteCount(packet) == 1)
        }
    }

    /// The widest real month columns: the calendar after August (hand check 06:25:37,
    /// wheel/shots/062537-check3c-after-tap-august.json; its month column alone needs a band of 69)
    /// and the task form after the upward tap (walk 06:48:16, second-app/walk/shots/064816-w4-after-tap-july-top.json;
    /// month centres spread 90).
    private static let calendarAugustScreen: [(String, Double, Double, Double, Double)] = [
        ("06:25", 148.29, 72.19, 148.53, 53.8),
        ("....", 859.8, 102.9, 79.89, 19.06),
        ("Cancel", 94.95, 281.37, 163.92, 43.22),
        ("Add", 1008.17, 278.21, 102.72, 45.73),
        ("New Event", 49.35, 430.02, 502.39, 85.09),
        ("Event title", 94.73, 611.43, 229.02, 46.22),
        ("Date", 95.11, 785.08, 114.13, 45.73),
        ("15 Aug 2029", 555.27, 784.08, 297.1, 55.34),
        ("12:00", 935.66, 784.45, 137.41, 50.79),
        ("August 2029 v", 216.85, 968.0, 361.42, 57.17),
        ("May", 346.2, 1177.61, 117.94, 49.54),
        ("June", 334.45, 1248.8, 133.84, 40.56),
        ("July", 330.98, 1326.24, 117.94, 57.17),
        ("August", 315.66, 1413.55, 224.67, 69.29),
        ("September", 334.4, 1507.15, 308.94, 65.02),
        ("October", 338.45, 1596.09, 220.93, 43.41),
        ("November", 346.2, 1661.62, 270.11, 38.11),
        ("2026", 764.64, 1196.45, 152.26, 30.93),
        ("2027", 768.25, 1245.33, 152.65, 43.68),
        ("2028", 772.01, 1321.63, 148.95, 54.97),
        ("2029", 756.94, 1409.72, 175.27, 65.52),
        ("2030", 768.22, 1508.4, 152.71, 54.9),
        ("2031", 768.01, 1591.36, 153.14, 49.05),
        ("2032", 764.45, 1660.37, 152.66, 32.98)
    ]
    private static let taskJulyScreen: [(String, Double, Double, Double, Double)] = [
        ("06:48", 148.28, 49.27, 152.36, 50.1),
        ("‹ MuckCalendar", 34.24, 102.9, 273.92, 34.3),
        ("Cancel", 95.11, 282.02, 163.59, 41.92),
        ("Save", 992.95, 282.02, 114.13, 45.73),
        ("Create Task", 49.46, 430.65, 559.25, 80.35),
        ("New Task", 98.54, 588.9, 232.82, 53.17),
        ("Title", 95.11, 727.91, 102.72, 49.54),
        ("Product", 94.61, 882.23, 187.41, 53.41),
        ("Priority", 94.93, 1039.86, 167.75, 54.47),
        ("Medium", 874.63, 1038.4, 232.83, 49.77),
        ("Due", 98.91, 1223.35, 91.31, 41.92),
        ("10 Jul 2030", 578.17, 1215.16, 277.92, 50.67),
        ("06:46", 931.91, 1215.24, 141.1, 50.5),
        ("No", 98.91, 1429.14, 64.68, 45.73),
        ("July 2030 v", 216.69, 1401.55, 300.87, 55.19),
        ("April", 346.2, 1627.32, 121.74, 34.3),
        ("May", 342.4, 1676.86, 117.94, 49.54),
        ("June", 330.61, 1755.8, 141.52, 51.73),
        ("July", 311.96, 1848.36, 136.96, 68.6),
        ("August", 334.44, 1942.42, 198.52, 59.59),
        ("September", 338.59, 2031.29, 300.55, 45.73),
        ("October", 346.2, 2092.26, 216.85, 34.3),
        ("2027", 764.62, 1619.46, 152.31, 42.4),
        ("2028", 768.14, 1675.66, 152.88, 48.13),
        ("2029", 771.48, 1750.74, 150.0, 58.04),
        ("2030", 757.0, 1844.31, 175.17, 61.45),
        ("2031", 772.14, 1943.15, 148.69, 50.51),
        ("2032", 768.21, 2026.51, 152.75, 47.66),
        ("2033", 764.62, 2095.74, 152.31, 31.15)
    ]

    @Test
    func widestRealWheelColumnsFire() async throws {
        for screen in [Self.calendarAugustScreen, Self.taskJulyScreen] {
            let words = VisionCaptureToolLoop.screenTextBlocks(in: try ocrReply(screen)).blocks
            let months = words.filter { Self.monthNames.contains($0.text) }
            #expect(months.count == 7)
            #expect(!VisionCaptureToolLoop.looksLikePickerWheel(months).isEmpty)
            let packet = try await afterTap(ocr: try ocrReply(screen), elements: [element("agenda", "Agenda", 393, 941)])
            #expect(try wheelNoteCount(packet) == 1)
        }
    }

    // MARK: Hint v3: each wheel by its visible rows

    private func guidanceText(_ packet: String) throws -> String {
        guard case .string(let text)? = try packetObject(packet)["guidance"] else { return "" }
        return text
    }

    @Test
    func twoWheelsAreNamedByTheirVisibleRows() async throws {
        let packet = try await afterTap(ocr: try ocrReply(Self.taskWheelScreen), elements: [element("agenda", "Agenda", 393, 941)])
        let note = "These rows are 2 picker wheels side by side: July to January, and 2023 to 2029. Set each wheel by tapping its own rows; if the value is not visible, tap that wheel's end row nearest to it; do not swipe these rows."
        let guidance = try guidanceText(packet)
        #expect(guidance.components(separatedBy: note).count == 2)
        #expect(!guidance.contains(VisionCaptureToolLoop.pickerWheelNote))
        let wheels = VisionCaptureToolLoop.looksLikePickerWheel(VisionCaptureToolLoop.screenTextBlocks(in: try ocrReply(Self.taskWheelScreen)).blocks)
        #expect(wheels.map { $0.map(\.text) } == [
            ["July", "August", "September", "October", "November", "December", "January"],
            ["2023", "2024", "2025", "2026", "2027", "2028", "2029"],
        ])
    }

    @Test
    func wheelsAreListedLeftToRight() async throws {
        // In this read the year column's top row sits higher than the month column's.
        let packet = try await afterTap(ocr: try ocrReply(Self.taskJulyScreen), elements: [element("agenda", "Agenda", 393, 941)])
        #expect(try guidanceText(packet).contains("These rows are 2 picker wheels side by side: April to October, and 2027 to 2033."))
    }

    @Test
    func oneWheelKeepsTheOneWheelNote() async throws {
        let monthsOnly = Self.taskWheelScreen.filter { !($0.0.count == 4 && Int($0.0) != nil) }
        let packet = try await afterTap(ocr: try ocrReply(monthsOnly), elements: [element("agenda", "Agenda", 393, 941)])
        let guidance = try guidanceText(packet)
        #expect(guidance.components(separatedBy: VisionCaptureToolLoop.pickerWheelNote).count == 2)
        #expect(!guidance.contains("picker wheels side by side"))
        #expect(VisionCaptureToolLoop.pickerWheelNote == "These look like picker wheel rows: tap the row you want; if it is not visible, tap the end row nearest to it; do not swipe these rows.")
    }

    @Test
    func hostKnownRowsDoNotFire() async throws {
        // The same rows, each also in the host's element list: the lean rule skips them.
        let words = VisionCaptureToolLoop.screenTextBlocks(in: try ocrReply(Self.calendarWheelScreen)).blocks
        let known = words.enumerated().map { index, word in element("e\(index)", word.text, word.xNorm, word.yNorm) }
        let packet = try await afterTap(ocr: try ocrReply(Self.calendarWheelScreen), elements: known)
        #expect(try textChoices(packet).isEmpty)
        #expect(try wheelNoteCount(packet) == 0)
    }

    @Test
    func aColumnOfFourDoesNotFire() async throws {
        let words = VisionCaptureToolLoop.screenTextBlocks(in: try ocrReply(Self.calendarWheelScreen)).blocks
        let months = words.filter { Self.monthNames.contains($0.text) }.sorted { $0.yNorm < $1.yNorm }
        #expect(VisionCaptureToolLoop.looksLikePickerWheel(Array(months.prefix(4))).isEmpty)
        #expect(!VisionCaptureToolLoop.looksLikePickerWheel(Array(months.prefix(5))).isEmpty)
        let four = Self.calendarWheelScreen.filter { ["October", "November", "December", "January"].contains($0.0) }
        #expect(try wheelNoteCount(try await afterTap(ocr: try ocrReply(four), elements: [element("agenda", "Agenda", 393, 941)])) == 0)
    }

    @Test
    func calendarDayGridDoesNotFire() async throws {
        let grid = Self.dayGridScreen.map { VisionCaptureToolLoop.ScreenTextBlock(text: $0.0, xNorm: $0.1, yNorm: $0.2) }
        #expect(VisionCaptureToolLoop.looksLikePickerWheel(grid).isEmpty)
        // Day numbers stack in columns 50 to 53 apart: "1", "8", "15", "22", "29".
        #expect(VisionCaptureToolLoop.looksLikePickerWheel(grid.filter { ["1", "8", "15", "22", "29"].contains($0.text) }).isEmpty)
        let rows = Self.dayGridScreen.map { word -> (String, Double, Double, Double, Double) in
            let width = 30.0 * Double(word.0.count), height = 46.0
            return (word.0, Double(word.1) * 1.206 - width / 2, Double(word.2) * 2.622 - height / 2, width, height)
        }
        let packet = try await afterTap(ocr: try ocrReply(rows), elements: [element("agenda", "Agenda", 393, 941)])
        #expect(try textChoices(packet).count == 38)
        #expect(try wheelNoteCount(packet) == 0)
    }

    // MARK: Refusal packets and stable screen-text IDs

    /// Screen-text choices of a packet: label -> ID.
    private func screenTextIDs(_ packet: String) throws -> [String: String] {
        guard case .array(let choices)? = try packetObject(packet)["choices"] else { return [:] }
        var ids: [String: String] = [:]
        for choice in choices {
            guard case .string(let label)? = choice.objectValue?["label"], label.hasSuffix(" (screen text)"),
                  case .string(let id)? = choice.objectValue?["id"], ids[label] == nil else { continue }
            ids[label] = id
        }
        return ids
    }

    private func wheelRead(_ loop: VisionCaptureToolLoop, _ screen: [(String, Double, Double, Double, Double)],
                           signatureBefore: String = "an-earlier-screen") async throws -> String {
        try await loop.actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: signatureBefore,
            body: ["screen_changed": .bool(true)], screenTextReply: try ocrReply(screen), visionPackComplete: true,
            configuration: Self.configuration).packet
    }

    @Test
    func refusedProposalOffersTheScreenTextAgain() async throws {
        let loop = VisionCaptureToolLoop()
        let first = try await wheelRead(loop, Self.taskWheelScreen)
        let offered = try screenTextIDs(first)
        // 29 OCR blocks; the clock and the back link sit in the status-bar strip.
        #expect(offered.count == 27)
        // An expired ID, then a coordinate tap with no screenshot: nothing is sent or read.
        for proposal: JSONValue in [
            .object(["action": .string("tap"), "target": .string("c9999")]),
            .object(["action": .string("tap_coordinates"), "x_norm": .integer(350), "y_norm": .integer(600),
                     "intent": .string("tap July")]),
        ] {
            let refusal = try await loop.refusalPacketForTesting(proposal, configuration: Self.configuration)
            guard case .string(let guidance)? = try packetObject(refusal)["guidance"] else {
                Issue.record("the refusal has no guidance"); return
            }
            #expect(guidance.contains("Not sent."))
            #expect(try screenTextIDs(refusal) == offered)
            #expect(try wheelNoteCount(refusal) == 1)
        }
    }

    @Test
    func sameWordAtTheSamePlaceKeepsItsID() async throws {
        let loop = VisionCaptureToolLoop()
        let first = try screenTextIDs(try await wheelRead(loop, Self.taskWheelScreen))
        // The next read: every word in place except "July", 30 units lower.
        let moved = Self.taskWheelScreen.map { row in row.0 == "July" ? (row.0, row.1, row.2 + 30 * 2.622, row.3, row.4) : row }
        let second = try screenTextIDs(try await wheelRead(loop, moved, signatureBefore: "screen-text-earlier"))
        #expect(second.count == first.count)
        for (label, id) in first where label != "July (screen text)" { #expect(second[label] == id, "\(label)") }
        let july = try #require(second["July (screen text)"])
        #expect(july != first["July (screen text)"])
        #expect(!first.values.contains(july))
        // A kept ID is a current choice: the tap is accepted.
        let kept = try #require(second["August (screen text)"])
        #expect(kept == first["August (screen text)"])
        #expect(await loop.proposalRefusalForTesting(
            .object(["action": .string("tap"), "target": .string(kept)]), configuration: Self.configuration) == nil)
    }

    /// The task form's wheels after the tap on 2029 (walk 06:47:50, second-app/walk/shots/064750-w2-after-tap-2029.json):
    /// the year column moved 3 rows, so other years now sit where 2023 to 2029 were.
    private static let taskWheelAfter2029: [(String, Double, Double, Double, Double)] = [
        ("06:47", 148.37, 49.54, 148.37, 49.54),
        ("‹ MuckCalendar", 34.24, 102.9, 273.92, 34.3),
        ("Cancel", 94.98, 281.52, 163.85, 42.93),
        ("Save", 992.95, 282.02, 114.13, 45.73),
        ("Create Task", 49.46, 430.65, 559.25, 80.35),
        ("New Task", 98.54, 588.9, 232.82, 53.17),
        ("Title", 95.11, 727.91, 102.72, 49.54),
        ("Product", 94.66, 882.4, 187.33, 53.07),
        ("Priority", 94.77, 1039.33, 168.08, 55.54),
        ("Medium", 874.67, 1038.57, 232.77, 49.42),
        ("Due", 98.91, 1223.35, 91.31, 41.92),
        ("10 Oct 2029", 563.05, 1215.72, 289.14, 49.54),
        ("06:46", 931.85, 1215.05, 141.24, 50.9),
        ("October 2029 v", 216.85, 1402.0, 388.05, 46.2),
        ("No", 98.91, 1429.14, 64.68, 45.73),
        ("July", 346.2, 1612.07, 110.33, 49.54),
        ("August", 327.18, 1680.67, 205.44, 45.73),
        ("September", 330.7, 1751.61, 312.53, 63.92),
        ("October", 323.24, 1843.96, 251.36, 58.34),
        ("November", 334.79, 1943.63, 289.14, 49.54),
        ("December", 342.4, 2031.29, 277.72, 41.92),
        ("January", 345.83, 2093.94, 206.19, 42.38),
        ("2026", 764.08, 1617.37, 153.4, 46.57),
        ("2027", 768.29, 1676.19, 152.58, 47.08),
        ("2028", 771.85, 1755.54, 149.26, 52.25),
        ("2029", 757.0, 1844.32, 175.16, 61.42),
        ("2030", 768.28, 1942.98, 152.6, 50.86),
        ("2031", 772.23, 2027.27, 148.5, 46.15),
        ("2032", 764.69, 2092.26, 152.18, 34.3)
    ]

    @Test
    func aDifferentWordAtAnOldPlaceGetsANewID() async throws {
        let loop = VisionCaptureToolLoop()
        let before = try screenTextIDs(try await wheelRead(loop, Self.taskWheelScreen))
        let after = try screenTextIDs(try await wheelRead(loop, Self.taskWheelAfter2029, signatureBefore: "screen-text-earlier"))
        // The months did not move: same IDs.
        for month in ["July", "August", "September", "October", "November", "December", "January"] {
            #expect(after["\(month) (screen text)"] == before["\(month) (screen text)"], "\(month)")
        }
        // Every year moved or is new: no year takes any earlier year's ID.
        let earlierYearIDs = Set(before.filter { $0.key.first?.isNumber == true && $0.key.count == 18 }.values)
        for year in 2026...2032 {
            let id = try #require(after["\(year) (screen text)"])
            #expect(!earlierYearIDs.contains(id), "\(year)")
        }
    }

    // MARK: Guards for kept screen-text words and IDs

    private func refusalWords(_ loop: VisionCaptureToolLoop, target: String = "c9999") async throws -> [String: String] {
        try screenTextIDs(try await loop.refusalPacketForTesting(
            .object(["action": .string("tap"), "target": .string(target)]), configuration: Self.configuration))
    }

    /// An after-action packet with no OCR read: the screen did not change.
    private func unchangedRead(_ loop: VisionCaptureToolLoop, signature: String = "screen-text-test") async throws {
        _ = try await loop.actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)], signature: signature), signatureBefore: signature,
            body: ["screen_changed": .bool(false), "effect": .string("Same elements as before the action.")],
            visionPackComplete: true, configuration: Self.configuration)
    }

    @Test
    func failedReadClearsTheKeptWords() async throws {
        let loop = VisionCaptureToolLoop()
        #expect(try screenTextIDs(try await wheelRead(loop, Self.taskWheelScreen)).count == 27)
        // The next read changed the screen but its OCR failed; the signature is the same.
        _ = try await loop.actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "screen-text-earlier",
            body: ["screen_changed": .bool(true)], screenTextReply: try reply(.object(["error": .string("Text recognition failed.")])),
            visionPackComplete: true, configuration: Self.configuration)
        #expect(try await refusalWords(loop).isEmpty)
    }

    @Test
    func anotherScreenGetsNoKeptWords() async throws {
        let loop = VisionCaptureToolLoop()
        #expect(try screenTextIDs(try await wheelRead(loop, Self.taskWheelScreen)).count == 27)
        try await unchangedRead(loop, signature: "another-screen")
        #expect(try await refusalWords(loop).isEmpty)
    }

    @Test
    func theRefusedIDIsNotKept() async throws {
        let loop = VisionCaptureToolLoop()
        let first = try screenTextIDs(try await wheelRead(loop, Self.taskWheelScreen))
        let july = try #require(first["July (screen text)"])
        // A packet with no words, then a tap on July's old ID: refused as expired.
        try await unchangedRead(loop)
        let repair = try await refusalWords(loop, target: july)
        #expect(repair.count == first.count)
        #expect(repair["July (screen text)"] != july)
        for (label, id) in first where label != "July (screen text)" { #expect(repair[label] == id, "\(label)") }
    }

    @Test
    func twoMatchesForOneKeptIDGetDistinctIDs() async throws {
        let loop = VisionCaptureToolLoop()
        let one = [("Alpha", 400.0, 700.0, 120.0, 46.0)]
        let first = try screenTextIDs(try await wheelRead(loop, one))
        let kept = try #require(first["Alpha (screen text)"])
        // The same text found twice near its old place (2 units apart).
        let twice = [("Alpha", 400.0, 700.0, 120.0, 46.0), ("Alpha", 402.4, 705.2, 120.0, 46.0)]
        let packet = try await wheelRead(loop, twice, signatureBefore: "screen-text-earlier")
        guard case .array(let choices)? = try packetObject(packet)["choices"] else { Issue.record("no choices"); return }
        let ids = choices.compactMap { choice -> String? in
            guard case .string(let label)? = choice.objectValue?["label"], label == "Alpha (screen text)",
                  case .string(let id)? = choice.objectValue?["id"] else { return nil }
            return id
        }
        #expect(ids.count == 2)
        #expect(Set(ids).count == 2)
        #expect(ids.contains(kept))
    }

    @Test
    func aNewModelContextForgetsTheKeptWords() async throws {
        // The run calls the same function when a new model context starts (no test can drive the run itself).
        let loop = VisionCaptureToolLoop()
        let first = try screenTextIDs(try await wheelRead(loop, Self.taskWheelScreen))
        await loop.forgetKeptScreenTextForTesting()
        #expect(try await refusalWords(loop).isEmpty)
        let again = try screenTextIDs(try await wheelRead(loop, Self.taskWheelScreen, signatureBefore: "screen-text-earlier"))
        #expect(Set(again.values).isDisjoint(with: Set(first.values)))
    }

    // MARK: Automatic image after actions (fix c)

    /// A real 2 x 2 PNG, so the capture stages like a screenshot.
    private func smallPNG() throws -> Data {
        let context = try #require(CGContext(data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        let image = try #require(context.makeImage())
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        return data as Data
    }

    private func capture(_ blocks: [JSONValue]) throws -> (reply: JSONValue, png: Data) {
        let png = try smallPNG()
        return (try reply(.object(["blocks": .array(blocks)]), png: png, udid: Self.configuration.simulatorUDID), png)
    }

    @Test
    func unknownWordAfterAnActionAttachesThatCapturesImage() async throws {
        let ocr = try capture([block("2025", 768, 1509, 152, 53)])
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: ocr.reply, visionPackComplete: true, configuration: Self.configuration)
        let image = try #require(result.image)
        #expect(image.sha256 == SHA256.hash(data: ocr.png).map { String(format: "%02x", $0) }.joined())
        #expect(try textChoices(result.packet) == ["2025 (screen text)"])
    }

    @Test
    func onlyKnownWordsAttachNoImage() async throws {
        // "Date" is a host static text: the lean rule skips it, so nothing new is shown.
        let date: JSONValue = .object([
            "element_id": .string("date"), "role": .string("static_text"), "label": .string("Date"),
            "visible": .bool(true), "x_norm": .integer(126), "y_norm": .integer(308),
        ])
        let ocr = try capture([block("Date", 95, 785, 114, 46)])
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([date]), signatureBefore: "an-earlier-screen", body: ["screen_changed": .bool(true)],
            screenTextReply: ocr.reply, visionPackComplete: true, configuration: Self.configuration)
        #expect(result.readScreenText)
        #expect(try textChoices(result.packet).isEmpty)
        #expect(result.image == nil)
    }

    @Test
    func failedReadAttachesNoImageAndGivesTheNote() async throws {
        let png = try smallPNG()
        let failed = try reply(.object(["error": .string("Text recognition failed.")]), png: png,
                               udid: Self.configuration.simulatorUDID)
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: failed, visionPackComplete: true, configuration: Self.configuration)
        #expect(result.image == nil)
        guard case .string(let guidance)? = try packetObject(result.packet)["guidance"] else {
            Issue.record("the packet has no guidance"); return
        }
        #expect(guidance.contains("Screen text positions unavailable: Text recognition failed."))
    }

    @Test
    func settingOffAttachesNoImage() async throws {
        let off = VisionCaptureAgentConfiguration(
            bundleIdentifier: Self.configuration.bundleIdentifier, simulatorUDID: Self.configuration.simulatorUDID,
            modelDirectory: Self.configuration.modelDirectory, autoScreenImage: false)
        let ocr = try capture([block("2025", 768, 1509, 152, 53)])
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: ocr.reply, visionPackComplete: true, configuration: off)
        #expect(result.image == nil)
        #expect(try textChoices(result.packet) == ["2025 (screen text)"])
    }

    @Test
    func autoImageIsNotCoordinateEvidence() async throws {
        let ocr = try capture([block("2025", 768, 1509, 152, 53)])
        let loop = VisionCaptureToolLoop()
        let result = try await loop.actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: ocr.reply, visionPackComplete: true, configuration: Self.configuration)
        #expect(result.image != nil)
        let packet = try packetObject(result.packet)
        #expect(packet["observation"]?.objectValue?["current_image_evidence"] == .bool(false))
        guard case .array(let allowed)? = packet["allowed_next"] else { Issue.record("no allowed_next"); return }
        #expect(!allowed.contains(.string("tap_coordinates")))
        #expect(allowed.contains(.string("screenshot")))
        let refusal = await loop.proposalRefusalForTesting(.object([
            "action": .string("tap_coordinates"), "x_norm": .integer(700), "y_norm": .integer(586),
            "intent": .string("tap 2025"),
        ]), configuration: Self.configuration)
        #expect(refusal?.contains("Coordinate actions need current screenshot evidence. Choose screenshot now") == true)
    }

    @Test
    func noAfterActionReadAttachesNoImage() async throws {
        let ocr = try capture([block("2025", 768, 1509, 152, 53)])
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "screen-text-test",
            body: ["screen_changed": .bool(false), "effect": .string("Same elements as before the action.")],
            screenTextReply: ocr.reply, visionPackComplete: true, configuration: Self.configuration)
        #expect(!result.readScreenText)
        #expect(result.image == nil)
    }

    @Test
    func noVisionPackAttachesNoImage() async throws {
        let ocr = try capture([block("2025", 768, 1509, 152, 53)])
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941)]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: ocr.reply, visionPackComplete: false,
            configuration: Self.configuration)
        #expect(result.image == nil)
        #expect(try textChoices(result.packet) == ["2025 (screen text)"])
    }

    private func guidance(_ packet: String) throws -> String {
        guard case .string(let text)? = try packetObject(packet)["guidance"] else { return "" }
        return text
    }

    @Test
    func noteComesOnlyWithTheAutoImage() async throws {
        let note = VisionCaptureToolLoop.autoScreenImageNote
        #expect(note == "The attached image shows the screen after your action. Tap words by their screen-text choice IDs; for a tap by position, take a screenshot first.")
        let ocr = try capture([block("2025", 768, 1509, 152, 53)])
        let screen = read([element("agenda", "Agenda", 393, 941)])
        let withImage = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: screen, signatureBefore: "an-earlier-screen", body: ["screen_changed": .bool(true)],
            screenTextReply: ocr.reply, visionPackComplete: true, configuration: Self.configuration)
        #expect(withImage.image != nil)
        #expect(try guidance(withImage.packet).contains(note))
        let off = VisionCaptureAgentConfiguration(
            bundleIdentifier: Self.configuration.bundleIdentifier, simulatorUDID: Self.configuration.simulatorUDID,
            modelDirectory: Self.configuration.modelDirectory, autoScreenImage: false)
        let settingOff = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: screen, signatureBefore: "an-earlier-screen", body: ["screen_changed": .bool(true)],
            screenTextReply: ocr.reply, visionPackComplete: true, configuration: off)
        let noVision = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: screen, signatureBefore: "an-earlier-screen", body: ["screen_changed": .bool(true)],
            screenTextReply: ocr.reply, visionPackComplete: false, configuration: Self.configuration)
        let noRead = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: screen, signatureBefore: "screen-text-test",
            body: ["screen_changed": .bool(false), "effect": .string("Same elements as before the action.")],
            screenTextReply: ocr.reply, visionPackComplete: true, configuration: Self.configuration)
        for result in [settingOff, noVision, noRead] {
            #expect(result.image == nil)
            #expect(try !guidance(result.packet).contains(note))
        }
    }

    @Test
    func choicesWipedForVisualDisambiguationAttachNoImage() async throws {
        // With a user restriction, a control with no readable label (only a symbol)
        // asks for the model's own screenshot; the loop clears every choice,
        // screen text included.
        let unlabeled = element("more", "›", 500, 800)
        let ocr = try capture([block("2025", 768, 1509, 152, 53)])
        let result = try await VisionCaptureToolLoop().actionPacketForTesting(
            read: read([element("agenda", "Agenda", 393, 941), unlabeled]), signatureBefore: "an-earlier-screen",
            body: ["screen_changed": .bool(true)], screenTextReply: ocr.reply, visionPackComplete: true,
            userInstruction: "Never tap Delete.", configuration: Self.configuration)
        #expect(result.readScreenText)
        #expect(try packetObject(result.packet)["allowed_next"] == .array([.string("screenshot")]))
        #expect(try textChoices(result.packet).isEmpty)
        #expect(result.image == nil)
        #expect(try !guidance(result.packet).contains(VisionCaptureToolLoop.autoScreenImageNote))
    }

    @Test
    func autoScreenImageSettingDefaultsToOnAndReadsTheKey() throws {
        #expect(MacAppSettings().agentAutoScreenImage)
        var file = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(MacAppSettings())) as? [String: Any])
        #expect(file["agentAutoScreenImage"] as? Bool == true)
        file.removeValue(forKey: "agentAutoScreenImage")  // a file written before the setting existed
        #expect(try JSONDecoder().decode(MacAppSettings.self, from: JSONSerialization.data(withJSONObject: file)).agentAutoScreenImage)
        file["agentAutoScreenImage"] = false
        #expect(try !JSONDecoder().decode(MacAppSettings.self, from: JSONSerialization.data(withJSONObject: file)).agentAutoScreenImage)
    }
}
