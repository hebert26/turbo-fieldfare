import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureScreenTextChoiceTests {
    /// A plain screenshot reply: the PNG block, then the metadata text with OCR.
    private func reply(_ ocr: JSONValue, width: Int64 = 1206, height: Int64 = 2622) throws -> JSONValue {
        let metadata = try JSONValue.object([
            "udid": .string("DE8B571C-2234-498F-9FAC-71C96B614792"),
            "width": .integer(width), "height": .integer(height), "ocr": ocr,
        ]).encoded()
        return .object(["content": .array([
            .object(["type": .string("image"), "mimeType": .string("image/png"), "data": .string("AAAA")]),
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
    private func read(_ elements: [JSONValue]) -> VisionCaptureMCPResult {
        VisionCaptureMCPResult(
            value: .object([
                "view": .object(["signature_fine": .string("screen-text-test")]),
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
}
