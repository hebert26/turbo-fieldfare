import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct VisionCaptureSystemPromptTests {
    private let prompt = VisionCaptureToolLoop.instructions(configuration: VisionCaptureAgentConfiguration(
        bundleIdentifier: "com.example.app", simulatorUDID: "00000000-0000-0000-0000-000000000001",
        modelDirectory: URL(fileURLWithPath: "/tmp/visioncapture-test-model.gturbo", isDirectory: true)))

    @Test
    func systemPromptTeachesWheelPickers() {
        let rule = "- Wheel pickers (columns of stacked values where the middle row is the current value, for example a date picker's month and year): set each column separately. To choose a value, tap its row when you can see it. If it is not visible, tap the row at the end of that column closest to it (the top or bottom row); the column moves and shows the next values. Check the screen again and repeat until the value is visible, then tap it. Do not swipe a picker wheel."
        #expect(prompt.split(separator: "\n").filter { $0 == Substring(rule) }.count == 1)
    }
}
