import Foundation
import Testing
@testable import TurboFieldfareRepackCore

/// These tests exercise only the argument boundary. They intentionally use a
/// nonexistent source directory for branches that must reject before metadata
/// or payload access, so a test can never read the prepared official bundle.
@Suite(.serialized)
struct QwenLocalRepackCLITests {
    @Test func parserBuildsPairedAndTextOnlyLocalIntents() throws {
        let source = temporaryPath("parser-source")
        let output = temporaryOutput("parser-output")

        let paired = try LocalQwenRepackCommandParser.parse([
            "--local-source", source, "--output", output,
        ])
        guard case let .run(pairedOptions) = paired else {
            Issue.record("expected local run command")
            return
        }
        #expect(pairedOptions.sourceDirectory == source)
        #expect(pairedOptions.outputDirectory == output)
        let pairedVisionOutput = try LocalQwenRepackCommandParser.derivedVisionOutput(
            textOutput: output)
        #expect(pairedOptions.visionOutputDirectory ==
                pairedVisionOutput)
        #expect(!pairedOptions.resume)

        let textOnly = try LocalQwenRepackCommandParser.parse([
            "--local-source", source, "--output", output, "--text-only",
        ])
        guard case let .run(textOnlyOptions) = textOnly else {
            Issue.record("expected text-only local run command")
            return
        }
        #expect(textOnlyOptions.visionOutputDirectory == nil)
    }

    @Test func parserBuildsPreflightIntentWithAdjacentVisionByDefault() throws {
        let source = temporaryPath("parser-preflight-source")
        let output = temporaryOutput("parser-preflight-output")
        let command = try LocalQwenRepackCommandParser.parse([
            "--local-source", source, "--output", output, "--preflight-only",
        ])
        guard case let .preflight(options) = command else {
            Issue.record("expected local preflight command")
            return
        }

        let expectedVision = try LocalQwenRepackCommandParser.derivedVisionOutput(
            textOutput: output)
        #expect(options.sourceDirectory == source)
        #expect(options.outputDirectory == output)
        #expect(options.visionOutputDirectory == expectedVision)
        #expect(!options.resume)
    }

    @Test func parserBuildsTextOnlyPreflightIntent() throws {
        let source = temporaryPath("parser-preflight-text-source")
        let output = temporaryOutput("parser-preflight-text-output")
        let command = try LocalQwenRepackCommandParser.parse([
            "--local-source", source, "--output", output,
            "--preflight-only", "--text-only",
        ])
        guard case let .preflight(options) = command else {
            Issue.record("expected text-only local preflight command")
            return
        }

        #expect(options.sourceDirectory == source)
        #expect(options.outputDirectory == output)
        #expect(options.visionOutputDirectory == nil)
    }

    @Test func parserAllowsResumeForPreflightWithoutChangingTheIntent() throws {
        let source = temporaryPath("parser-preflight-resume-source")
        let output = temporaryOutput("parser-preflight-resume-output")
        let command = try LocalQwenRepackCommandParser.parse([
            "--local-source", source, "--output", output,
            "--preflight-only", "--resume",
        ])
        guard case let .preflight(options) = command else {
            Issue.record("expected resume preflight command")
            return
        }

        #expect(options.resume)
        #expect(options.sourceDirectory == source)
        #expect(options.outputDirectory == output)
    }

    @Test func parserRejectsPreflightAndDiscardTogether() throws {
        do {
            _ = try LocalQwenRepackCommandParser.parse([
                "--local-source", temporaryPath("parser-preflight-discard-source"),
                "--output", temporaryOutput("parser-preflight-discard-output"),
                "--preflight-only", "--discard-partial",
            ])
            Issue.record("preflight and discard should be incompatible")
        } catch let error as LocalQwenRepackCommandParseError {
            #expect(error == .invalid(
                "--preflight-only and --discard-partial are mutually exclusive"))
        } catch {
            Issue.record("unexpected parser error: \(error)")
        }
    }

    @Test func parserRejectsDuplicatePreflightFlag() throws {
        do {
            _ = try LocalQwenRepackCommandParser.parse([
                "--local-source", temporaryPath("parser-preflight-duplicate-source"),
                "--output", temporaryOutput("parser-preflight-duplicate-output"),
                "--preflight-only", "--preflight-only",
            ])
            Issue.record("duplicate preflight flag should be rejected")
        } catch let error as LocalQwenRepackCommandParseError {
            #expect(error == .duplicate("--preflight-only"))
        } catch {
            Issue.record("unexpected parser error: \(error)")
        }
    }

    @Test func parserBuildsDiscardIntentWithoutOpeningSource() throws {
        let source = temporaryPath("parser-discard-source")
        let output = temporaryOutput("parser-discard-output")
        let command = try LocalQwenRepackCommandParser.parse([
            "--local-source", source, "--output", output, "--discard-partial",
        ])
        guard case let .discard(options) = command else {
            Issue.record("expected local discard command")
            return
        }
        #expect(options.sourceDirectory == source)
        #expect(options.outputDirectory == output)
        #expect(!options.resume)
    }

    @Test func helpDescribesLocalModeAndProtection() throws {
        let result = try run(["--help"])

        #expect(result.status == 0)
        #expect(result.stdout.contains("--local-source"))
        #expect(result.stdout.contains("local"))
        #expect(result.stdout.contains("deterministic"))
        #expect(result.stdout.contains("--resume"))
        #expect(result.stdout.contains("--text-only"))
        #expect(result.stdout.contains("--preflight-only"))
        #expect(result.stdout.contains("never replaces a completed destination"))
    }

    @Test func localModeRequiresSourceAndOutputTogether() throws {
        let source = temporaryPath("source")
        let output = temporaryOutput("missing-flag")
        defer { cleanup([output]) }

        let missingSource = try run(["--local-source", source, "--text-only"])
        #expect(missingSource.status == 2)
        #expect(missingSource.stderr.contains("--output"))

        let missingOutput = try run(["--output", output, "--text-only"])
        #expect(missingOutput.status == 2)
        #expect(missingOutput.stderr.contains("--local-source"))
    }

    @Test func localModeRejectsRemoteAndVerificationModesBeforeSourceAccess() throws {
        let source = temporaryPath("missing-source")
        let output = temporaryOutput("mode-conflict")
        defer { cleanup([output]) }

        let cases: [[String]] = [
            ["--local-source", source, "--output", output, "--verify-install"],
            ["--local-source", source, "--output", output, "--input-gturbo", output],
            ["--local-source", source, "--output", output, "--overwrite"],
        ]
        for arguments in cases {
            let result = try run(arguments)
            #expect(result.status == 2, "accepted conflicting arguments: (arguments)")
            #expect(!FileManager.default.fileExists(atPath: output))
            #expect(result.stderr.contains("local Qwen conversion does not accept"))
        }
    }

    @Test func localResumeAndDiscardAreMutuallyExclusive() throws {
        let source = temporaryPath("missing-source")
        let output = temporaryOutput("resume-discard")
        defer { cleanup([output]) }

        let result = try run([
            "--local-source", source, "--output", output,
            "--resume", "--discard-partial",
        ])

        #expect(result.status == 2)
        #expect(result.stderr.contains("mutually exclusive"))
        #expect(!FileManager.default.fileExists(atPath: output))
    }

    @Test func localTextOnlyIsAcceptedAsExplicitVisionOptOut() throws {
        let source = temporaryPath("missing-source")
        let output = temporaryOutput("text-only")
        defer { cleanup([output]) }

        let result = try run([
            "--local-source", source, "--output", output, "--text-only",
        ])

        #expect(result.status == 1)
        #expect(!result.stderr.contains("unknown argument: --text-only"))
        #expect(!FileManager.default.fileExists(atPath: output))
    }

    @Test func localDefaultDoesNotRouteMissingSourceToRemoteInstaller() throws {
        let source = temporaryPath("missing-source")
        let output = temporaryOutput("offline")
        defer { cleanup([output]) }

        let result = try run(["--local-source", source, "--output", output])

        #expect(result.status == 1)
        #expect(!result.stderr.localizedCaseInsensitiveContains("hugging face"))
        #expect(!result.stderr.localizedCaseInsensitiveContains("remote install"))
        #expect(!FileManager.default.fileExists(atPath: output))
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))
    }

    @Test func localModeRejectsGemmaDestinationBeforeOpeningSource() throws {
        let source = temporaryDirectory("existing-source")
        let gemma = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("scratch/gemma4.gturbo").path
        defer { cleanup([source]) }
        try FileManager.default.createDirectory(atPath: source, withIntermediateDirectories: true)

        let result = try run(["--local-source", source, "--output", gemma])

        #expect(result.status == 1)
        #expect(result.stderr.localizedCaseInsensitiveContains("gemma"))
        #expect(!result.stderr.localizedCaseInsensitiveContains("no such file"))
    }

    @Test func localModeRejectsEqualAndNestedSourceOutputPaths() throws {
        let root = temporaryDirectory("path-boundary")
        defer { cleanup([root]) }
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        let source = (root as NSString).appendingPathComponent("source.gturbo")
        let nested = (source as NSString).appendingPathComponent("nested.gturbo")
        try FileManager.default.createDirectory(atPath: source, withIntermediateDirectories: true)

        let equal = try run(["--local-source", source, "--output", source])
        #expect(equal.status == 1)
        #expect(equal.stderr.localizedCaseInsensitiveContains("path"))

        let nestedResult = try run(["--local-source", source, "--output", nested])
        #expect(nestedResult.status == 1)
        #expect(nestedResult.stderr.localizedCaseInsensitiveContains("path"))
        #expect(!FileManager.default.fileExists(atPath: nested + ".partial"))
    }

    @Test func localMissingValuesAreParseErrors() throws {
        for flag in ["--local-source", "--output"] {
            let result = try run([flag])
            #expect(result.status == 2)
            #expect(result.stderr.contains("missing value"))
        }
    }

    @Test func localDiscardDoesNotDeleteAnUnrelatedPartial() throws {
        let output = temporaryOutput("owned-discard")
        let unrelated = output + ".partial"
        defer { cleanup([output]) }
        try FileManager.default.createDirectory(atPath: unrelated, withIntermediateDirectories: true)
        try Data("unrelated".utf8).write(
            to: URL(fileURLWithPath: (unrelated as NSString).appendingPathComponent("keep.txt")))

        let result = try run([
            "--local-source", temporaryPath("missing-source"),
            "--output", output, "--discard-partial",
        ])

        #expect(result.status == 1)
        #expect(FileManager.default.fileExists(atPath: unrelated))
        #expect(FileManager.default.fileExists(
            atPath: (unrelated as NSString).appendingPathComponent("keep.txt")))
    }

    private func run(_ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String) {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/TurboFieldfareRepack")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        return (
            process.terminationStatus,
            String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self),
            String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
    }

    private func temporaryOutput(_ tag: String) -> String {
        temporaryPath(tag) + ".gturbo"
    }

    private func temporaryPath(_ tag: String) -> String {
        temporaryDirectory(tag)
    }

    private func temporaryDirectory(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-qwen-local-\(tag)-\(UUID().uuidString)")
    }

    private func cleanup(_ paths: [String]) {
        for path in paths { try? FileManager.default.removeItem(atPath: path) }
    }
}
