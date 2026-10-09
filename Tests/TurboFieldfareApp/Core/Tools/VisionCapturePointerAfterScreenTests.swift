import Foundation
import Testing
@testable import TurboFieldfareAppCore

/// The loop's pointer click asks VisionCapture for the screen after the click
/// (ADR 0054 amendment of 9 October, Codex task 7).
@Suite struct VisionCapturePointerAfterScreenTests {
    private let parameters = VisionCapturePointerClick.parameters(
        taskID: "00000000-0000-0000-0000-000000000002", generation: 3, x: 500, y: 400, intent: "Save")

    @Test
    func pointerClickAsksForTheScreenAfter() {
        #expect(parameters["include_after_screen"] == .bool(true))
        #expect(parameters["cache_policy"] == .string("visual_bypass"))
        #expect(parameters["x_norm"] == .integer(500))
        #expect(parameters["y_norm"] == .integer(400))
        #expect(parameters["intent"] == .string("Save"))
    }

    @Test
    func sentClickKeysAreRecognized() {
        var sent = parameters
        sent["udid"] = .string("00000000-0000-0000-0000-000000000001")
        #expect(VisionCapturePointerClick.hasKeys(sent))
        // A click recorded before the after-screen request still counts.
        sent.removeValue(forKey: "include_after_screen")
        #expect(VisionCapturePointerClick.hasKeys(sent))
    }

    @Test
    func otherClickKeysAreNotRecognized() {
        var sent = parameters
        sent["udid"] = .string("00000000-0000-0000-0000-000000000001")
        var off = sent
        off["include_after_screen"] = .bool(false)
        #expect(!VisionCapturePointerClick.hasKeys(off))
        var extra = sent
        extra["mode"] = .string("fast")
        #expect(!VisionCapturePointerClick.hasKeys(extra))
        var missing = sent
        missing.removeValue(forKey: "intent")
        #expect(!VisionCapturePointerClick.hasKeys(missing))
    }
}
