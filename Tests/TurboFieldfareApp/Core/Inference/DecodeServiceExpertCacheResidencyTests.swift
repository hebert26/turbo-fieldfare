import Testing
@testable import TurboFieldfareAppCore

@Suite struct DecodeServiceExpertCacheResidencyTests {
    @Test func forwardsOnlyExactOneAndPreservesExistingFlagsAndThinking() {
        let values: [String?] = [nil, "0", "1", "true", "2", "", " 1", "1 "]
        for thinking in [false, true] {
            for value in values {
                for priorFlags in [false, true] {
                    var input = ["UNRELATED_RESIDENCY_TEST": "not-forwarded"]
                    if let value { input["TURBO_QWEN_EXPERT_CACHE_RESIDENCY"] = value }
                    if priorFlags {
                        input["TURBO_QWEN_GPU_LINEAR_PREPARATION"] = "1"
                        input["TURBO_QWEN_GROUPED_LINEAR_PREFILL"] = "1"
                    }
                    var expected = ["TURBOFIELDFARE_AGENT_THINKING": thinking ? "1" : "0"]
                    if value == "1" { expected["TURBO_QWEN_EXPERT_CACHE_RESIDENCY"] = "1" }
                    if priorFlags {
                        expected["TURBO_QWEN_GPU_LINEAR_PREPARATION"] = "1"
                        expected["TURBO_QWEN_GROUPED_LINEAR_PREFILL"] = "1"
                    }
                    #expect(DecodeServiceInferenceClient.launchEnvironment(environment: input, thinking: thinking) == expected)
                }
            }
        }
    }
}
