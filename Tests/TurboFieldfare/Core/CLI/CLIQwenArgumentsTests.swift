import Testing
import TurboFieldfare
@testable import TurboFieldfareCLICore

@Suite struct CLIQwenArgumentsTests {
    @Test func helpDescribesTheVerifiedQwenSurface() {
        let usage = Args.usage
        #expect(usage.contains("Gemma and verified Qwen"))
        #expect(usage.contains("--thinking <auto|on|off>"))
        #expect(usage.contains("--tools-file <path>"))
        #expect(usage.contains("--show-model-identity"))
        #expect(usage.contains("--video <path>"))
        #expect(usage.contains("still-image") || usage.contains("still image"))
        #expect(throws: ArgsError.helpRequested) {
            _ = try Args.parse(["--help"])
        }
    }

    @Test func thinkingDefaultsToAutoAndAcceptsExplicitModes() throws {
        #expect(try Args.parse(["--model", "qwen.gturbo", "--chat-prompt", "hello"]).thinking == .auto)
        #expect(try Args.parse([
            "--model", "qwen.gturbo", "--chat-prompt", "hello", "--thinking", "on",
        ]).thinking == .on)
        #expect(try Args.parse([
            "--model", "qwen.gturbo", "--chat-prompt", "hello", "--thinking", "off",
        ]).thinking == .off)
        for value in ["default", "true", "ON"] {
            #expect(throws: ArgsError.invalidValue(flag: "--thinking", value: value)) {
                _ = try Args.parse([
                    "--model", "qwen.gturbo", "--chat-prompt", "hello",
                    "--thinking", value,
                ])
            }
        }
    }

    @Test func identityAndToolsOptionsAreRetainedWithoutChangingSamplingDefaults() throws {
        let arguments = try Args.parse([
            "--model", "qwen.gturbo", "--chat-prompt", "hello",
            "--tools-file", "tools.json", "--show-model-identity",
            "--max-new", "32", "--temperature", "0", "--top-k", "0",
            "--top-p", "1", "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "first", "--stop", "second",
        ])
        #expect(arguments.toolsFile == "tools.json")
        #expect(arguments.showModelIdentity)
        #expect(arguments.maxNew == 32)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == nil)
        #expect(arguments.topP == 1)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["first", "second"])
    }

    @Test func rawCompletionRejectsThinkingAndToolsBeforeRuntimeSelection() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--thinking")) {
            _ = try Args.parse([
                "--model", "qwen.gturbo", "--prompt", "raw", "--thinking", "on",
            ])
        }
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--tools-file")) {
            _ = try Args.parse([
                "--model", "qwen.gturbo", "--prompt", "raw",
                "--tools-file", "tools.json",
            ])
        }
    }

    @Test func videoIsRejectedBeforeAnyModelOrTokenizerLoad() {
        #expect(throws: ArgsError.unsupported("--video is not supported")) {
            _ = try Args.parse([
                "--model", "/does/not/exist.gturbo", "--chat-prompt", "hello",
                "--video", "/does/not/exist.mp4",
            ])
        }
    }
}
