import Testing
@testable import TurboFieldfareAppCore

@Suite struct DecodeServiceLaunchPolicyTests {
    @Test func kickstartTargetsGUIJobWithoutRestartingIt() {
        let arguments = DecodeServiceInferenceClient.kickstartArguments(
            uid: 501, label: "com.turbofieldfare.decode.test")

        #expect(arguments == [
            "kickstart", "gui/501/com.turbofieldfare.decode.test",
        ])
        #expect(!arguments.contains("-k"))
    }

    @Test func socketFailureRemainsThePrimaryDiagnostic() {
        let message = DecodeServiceInferenceClient.socketFailureMessage(
            socketError: "No such file or directory", kickstartError: nil)

        #expect(message ==
            "decode service socket did not become ready: No such file or directory")
    }

    @Test func socketFailureIncludesKickstartDiagnosticWhenAvailable() {
        let message = DecodeServiceInferenceClient.socketFailureMessage(
            socketError: "No such file or directory",
            kickstartError: "Operation not permitted")

        #expect(message ==
            "decode service socket did not become ready: No such file or directory; launchctl kickstart failed: Operation not permitted")
    }
    @Test func linearGPUPreparationTrialForwardsOnlyExactOneAndPreservesThinking() {
        let flags: [String?] = [nil, "0", "1", "true", "2", "", " 1", "1 "]
        for thinking in [false, true] {
            for flag in flags {
                var environment = ["UNRELATED_TEST_ENV": "must-not-be-forwarded"]
                if let flag { environment["TURBO_QWEN_GPU_LINEAR_PREPARATION"] = flag }
                let forwarded = DecodeServiceInferenceClient.launchEnvironment(
                    environment: environment, thinking: thinking)
                var expected = ["TURBOFIELDFARE_AGENT_THINKING": thinking ? "1" : "0"]
                if flag == "1" { expected["TURBO_QWEN_GPU_LINEAR_PREPARATION"] = "1" }
                #expect(forwarded == expected)
            }
        }
    }

    @Test func groupedLinearPrefillTrialForwardsOnlyExactOneIndependently() {
        let flags: [String?] = [nil, "0", "1", "true", "2", "", " 1", "1 "]
        for flag in flags {
            for preparation in [false, true] {
                var environment: [String: String] = [:]
                if let flag { environment["TURBO_QWEN_GROUPED_LINEAR_PREFILL"] = flag }
                if preparation { environment["TURBO_QWEN_GPU_LINEAR_PREPARATION"] = "1" }
                let actual = DecodeServiceInferenceClient.launchEnvironment(environment: environment, thinking: true)
                var expected = ["TURBOFIELDFARE_AGENT_THINKING": "1"]
                if flag == "1" { expected["TURBO_QWEN_GROUPED_LINEAR_PREFILL"] = "1" }
                if preparation { expected["TURBO_QWEN_GPU_LINEAR_PREPARATION"] = "1" }
                #expect(actual == expected)
            }
        }
    }

}
