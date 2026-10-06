import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --verify-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --activate-vision-install
                       --vision-output <model.vision.gturbo>
                       --text-model <model.gturbo>
  TurboFieldfareRepack --remove-vision-install
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --discard-partial
                       --vision-output <model.vision.gturbo>
  TurboFieldfareRepack --local-source <official-snapshot>
                       --output <model.gturbo> [--preflight-only]
                       [--resume] [--text-only]
  TurboFieldfareRepack --local-source <official-snapshot>
                       --output <model.gturbo> --discard-partial [--text-only]
  TurboFieldfareRepack --help

The installer streams the supported Gemma 4 checkpoint from Hugging Face and
repackages it without materializing the source checkpoint on disk. Set HF_TOKEN
only if Hugging Face requests authentication. A cancelled or interrupted
download can be continued with --resume or removed with --discard-partial.

The optional image companion pack installs beside an existing text model and
is bound to it. Without the pack the text runtime is unchanged; image input is
simply unavailable.

The local Qwen converter reads the pinned official snapshot in place and
publishes deterministic text and matching image packs together. It never
downloads source files and never replaces a completed destination. The image
pack defaults to <model>.vision.gturbo; use --text-only to omit it. Use
--preflight-only to authenticate the source and report exact capacity without
creating output paths.
"""

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputGTurbo: String?
    var visionOutput: String?
    var textModel: String?
    var verifyVisionInstall = false
    var activateVisionInstall = false
    var removeVisionInstall = false

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--verify-vision-install":
                parsed.verifyVisionInstall = true
                index += 1
            case "--activate-vision-install":
                parsed.activateVisionInstall = true
                index += 1
            case "--remove-vision-install":
                parsed.removeVisionInstall = true
                index += 1
            case "--vision-output":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.visionOutput = values[index + 1]
                index += 2
            case "--text-model":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                parsed.textModel = values[index + 1]
                index += 2
            case "--output", "--input-gturbo":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else {
                    parsed.inputGTurbo = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        let visionModes = [parsed.verifyVisionInstall,
                           parsed.activateVisionInstall,
                           parsed.removeVisionInstall].filter { $0 }.count
        guard visionModes <= 1 else {
            throw ParseError.invalidMode("vision install modes are mutually exclusive")
        }
        if visionModes == 1 || parsed.visionOutput != nil {
            guard parsed.visionOutput != nil else {
                throw ParseError.missingRequired("--vision-output")
            }
            guard parsed.output == nil, parsed.inputGTurbo == nil,
                  !parsed.verifyInstall else {
                throw ParseError.invalidMode(
                    "vision install operations do not accept text install arguments")
            }
            // Discard runs first below, so accepting it alongside another mode
            // would silently perform the discard and exit 0 without ever doing
            // what was asked.
            guard !(parsed.discardPartial && visionModes == 1) else {
                throw ParseError.invalidMode(
                    "--discard-partial is mutually exclusive with the other vision "
                        + "install operations")
            }
            if parsed.removeVisionInstall || parsed.discardPartial {
                guard parsed.textModel == nil, !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output")
                }
            } else if parsed.verifyVisionInstall || parsed.activateVisionInstall {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
                // Neither reads a download, so a transfer flag here is a
                // request this mode cannot honour rather than a no-op.
                guard !parsed.overwrite, !parsed.resume else {
                    throw ParseError.invalidMode(
                        "this vision operation only accepts --vision-output and "
                            + "--text-model")
                }
            } else {
                guard parsed.textModel != nil else {
                    throw ParseError.missingRequired("--text-model")
                }
            }
            return parsed
        }
        guard parsed.textModel == nil else {
            throw ParseError.invalidMode("--text-model requires --vision-output")
        }
        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil, !parsed.overwrite, !parsed.verifyInstall else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func printLocalQwenPreflight(
    _ preflight: LocalQwenStreamingRepackPreflight,
    reserveBytes: UInt64
) {
    var lines = [
        "Local Qwen preflight",
        "Source repository: \(preflight.sourceRepository)",
        "Source revision: \(preflight.sourceRevision)",
        "Source index SHA-256: \(preflight.sourceIndexSHA256)",
        "Source payload SHA-256: \(preflight.sourcePayloadSHA256)",
        "Plan fingerprint: \(preflight.planFingerprint)",
        "Quantization policy SHA-256: \(preflight.quantizationPolicySHA256)",
        "Converter version: \(preflight.converterVersion)",
        "Text output: \(preflight.textOutputDirectory)",
        "Image output: \(preflight.visionOutputDirectory ?? "none")",
        "Planned artifact bytes: \(preflight.artifactBytes)",
        "Required bytes: \(preflight.requiredBytes)",
        "Available bytes: \(preflight.availableBytes)",
        "Protected reserve bytes: \(reserveBytes)",
    ]
    for (index, requirement) in preflight.diskRequirements.enumerated() {
        let number = index + 1
        lines.append("Volume \(number) probe: \(requirement.probePath)")
        lines.append("Volume \(number) outputs: \(requirement.paths.joined(separator: ", "))")
        lines.append("Volume \(number) required bytes: \(requirement.requiredBytes)")
        lines.append("Volume \(number) available bytes: \(requirement.availableBytes)")
    }
    FileHandle.standardOutput.write(Data((lines.joined(separator: "\n") + "\n").utf8))
}

private func printLocalQwenProgress(_ progress: LocalQwenStreamingRepackProgress) {
    guard progress.stage == .converting,
          let completed = progress.durableCompletedUnitCount else { return }
    FileHandle.standardOutput.write(
        Data("Durable completed units: \(completed)\n".utf8))
}

private func runLocalQwen(_ command: LocalQwenRepackCommand) -> Int32 {
    do {
        switch command {
        case .preflight(let options):
            let preflight = try LocalQwenStreamingRepacker.preflight(options: options)
            printLocalQwenPreflight(preflight, reserveBytes: options.reserveBytes)
        case .discard(let options):
            try LocalQwenStreamingRepacker.discardPartial(options: options)
            print("Discarded owned Qwen partial for \(options.outputDirectory)")
        case .run(let options):
            let result = try LocalQwenStreamingRepacker.run(
                options: options,
                preflightReport: {
                    printLocalQwenPreflight($0, reserveBytes: options.reserveBytes)
                },
                progress: printLocalQwenProgress)
            print("Converted official local Qwen snapshot")
            print("Source revision: \(result.preflight.sourceRevision)")
            print("Completed units: \(result.completedUnitCount)")
            print("Maximum observed transform scratch bytes: "
                + "\(result.maximumObservedTransformScratchBytes)")
            print("Text model: \(result.textOutputDirectory)")
            print("Text receipt: \(result.textReceiptPath)")
            if let vision = result.visionOutputDirectory,
               let receipt = result.visionReceiptPath {
                print("Image model: \(vision)")
                print("Image receipt: \(receipt)")
            }
        }
        return 0
    } catch {
        printError("local Qwen operation failed: \(error)")
        return 1
    }
}

private func runVisionInstall(_ arguments: Arguments) async -> Int32? {
    guard let visionOutput = arguments.visionOutput else { return nil }

    if arguments.discardPartial {
        do {
            try RemoteVisionPackInstaller.discardPartial(outputDirectory: visionOutput)
            print("Discarded saved image-pack download for \(visionOutput)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.removeVisionInstall {
        do {
            try RemoteVisionPackInstaller.removeInstalled(outputDirectory: visionOutput)
            print("Removed image pack \(visionOutput)")
            return 0
        } catch {
            printError("remove-vision-install failed: \(error)")
            return 1
        }
    }

    guard let textModel = arguments.textModel else { return 2 }

    if arguments.verifyVisionInstall {
        do {
            let verification = try VisionPackVerifier.verify(
                directory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                installedDirectory: URL(fileURLWithPath: visionOutput, isDirectory: true),
                textModelDirectory: URL(fileURLWithPath: textModel, isDirectory: true),
                verifyWeights: true)
            print("Verified image pack \(visionOutput)")
            print("Bound to text model \(textModel)")
            print("Text manifest sha256 \(verification.compatibleTextManifestSha256)")
            return 0
        } catch {
            printError("verify-vision-install failed: \(error)")
            return 1
        }
    }

    if arguments.activateVisionInstall {
        do {
            try RemoteVisionPackInstaller.activatePrepared(
                outputDirectory: visionOutput,
                textModelDirectory: textModel,
                repoID: SupportedModelSource.repoID,
                requestedRevision: SupportedModelSource.revision)
            print("Activated image pack \(visionOutput)")
            return 0
        } catch {
            printError("activate-vision-install failed: \(error)")
            return 1
        }
    }

    let options = RemoteVisionPackInstallOptions(
        repoID: SupportedModelSource.repoID,
        revision: SupportedModelSource.revision,
        textModelDirectory: textModel,
        outputDirectory: visionOutput,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        overwrite: arguments.overwrite,
        resume: arguments.resume)
    do {
        let progress = InstallProgressReporter()
        try await RemoteVisionPackInstaller(options: options).run(
            progress: { progress($0) })
        print("Installed image pack \(visionOutput)")
        print("Text model: \(textModel)")
        return 0
    } catch {
        printError("vision install failed: \(error)")
        return 1
    }
}

private func run(_ values: [String]) async -> Int32 {
    let localArguments = Array(values.dropFirst())
    if LocalQwenRepackCommandParser.isLocalMode(localArguments) {
        do {
            return runLocalQwen(try LocalQwenRepackCommandParser.parse(localArguments))
        } catch {
            printError("error: \(error)\n\n\(usage)")
            return 2
        }
    }

    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if let code = await runVisionInstall(arguments) {
        return code
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let options = SupportedModelSource.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let progress = InstallProgressReporter()
        let result = try await RemoteStreamingRepacker(options: options).run(
            progress: { progress($0) })
        print("Installed \(SupportedModelSource.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
