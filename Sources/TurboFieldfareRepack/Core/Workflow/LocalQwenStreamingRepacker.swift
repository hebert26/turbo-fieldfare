import Foundation
import TurboFieldfareFormat

public struct LocalQwenStreamingRepackOptions: Sendable, Equatable {
    public let sourceDirectory: String
    public let outputDirectory: String
    public let visionOutputDirectory: String?
    public let resume: Bool
    public let reserveBytes: UInt64

    public init(
        sourceDirectory: String,
        outputDirectory: String,
        visionOutputDirectory: String? = nil,
        resume: Bool = false,
        reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024
    ) {
        self.sourceDirectory = sourceDirectory
        self.outputDirectory = outputDirectory
        self.visionOutputDirectory = visionOutputDirectory
        self.resume = resume
        self.reserveBytes = reserveBytes
    }
}

public struct LocalQwenStreamingRepackProgress: Sendable, Equatable {
    public enum Stage: String, Sendable {
        case validatingSource
        case planning
        case preflighting
        case converting
        case auditing
        case publishing
    }

    public let stage: Stage
    public let completedBytes: UInt64
    public let totalBytes: UInt64
    public let durableCompletedUnitCount: UInt64?

    public init(
        stage: Stage,
        completedBytes: UInt64 = 0,
        totalBytes: UInt64 = 0,
        durableCompletedUnitCount: UInt64? = nil
    ) {
        self.stage = stage
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
        self.durableCompletedUnitCount = durableCompletedUnitCount
    }
}

public struct LocalQwenStreamingRepackPreflight: Sendable, Equatable {
    public let sourceRepository: String
    public let sourceRevision: String
    public let sourceIndexSHA256: String
    public let sourcePayloadSHA256: String
    public let planFingerprint: String
    public let quantizationPolicySHA256: String
    public let converterVersion: String
    public let textOutputDirectory: String
    public let visionOutputDirectory: String?
    public let artifactBytes: UInt64
    public let requiredBytes: UInt64
    public let availableBytes: UInt64
    public let diskRequirements: [CombinedDiskSpaceRequirement]
}

public struct LocalQwenStreamingRepackResult: Sendable {
    public let preflight: LocalQwenStreamingRepackPreflight
    public let textOutputDirectory: String
    public let visionOutputDirectory: String?
    public let textManifestSHA256: String
    public let textReceiptSHA256: String
    public let textReceiptPath: String
    public let visionManifestSHA256: String?
    public let visionReceiptSHA256: String?
    public let visionReceiptPath: String?
    public let completedUnitCount: Int
    public let maximumObservedTransformScratchBytes: Int
}

public struct LocalQwenPublishedVerification: Sendable, Equatable {
    public let textOutputDirectory: String
    public let visionOutputDirectory: String?
    public let textManifestSHA256: String
    public let textReceiptSHA256: String
    public let visionManifestSHA256: String?
    public let visionReceiptSHA256: String?
}

public enum LocalQwenRepackCommand: Sendable, Equatable {
    case preflight(LocalQwenStreamingRepackOptions)
    case run(LocalQwenStreamingRepackOptions)
    case discard(LocalQwenStreamingRepackOptions)
}

public enum LocalQwenRepackCommandParseError: Error, CustomStringConvertible, Equatable {
    case missingValue(String)
    case missingRequired(String)
    case duplicate(String)
    case invalid(String)
    case unknown(String)

    public var description: String {
        switch self {
        case .missingValue(let flag): "missing value for \(flag)"
        case .missingRequired(let flag): "missing required argument: \(flag)"
        case .duplicate(let flag): "duplicate local Qwen argument: \(flag)"
        case .invalid(let detail): detail
        case .unknown(let flag): "local Qwen conversion does not accept \(flag)"
        }
    }
}

/// The executable and its tests share this parser. It accepts arguments after
/// the executable name and never falls through to the remote Gemma installer.
public enum LocalQwenRepackCommandParser {
    public static func isLocalMode(_ arguments: [String]) -> Bool {
        arguments.contains("--local-source")
    }

    public static func parse(_ arguments: [String]) throws -> LocalQwenRepackCommand {
        var source: String?
        var output: String?
        var textOnly = false
        var resume = false
        var discard = false
        var preflightOnly = false
        var seen = Set<String>()
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            guard seen.insert(flag).inserted else {
                throw LocalQwenRepackCommandParseError.duplicate(flag)
            }
            switch flag {
            case "--local-source", "--output":
                guard index + 1 < arguments.count else {
                    throw LocalQwenRepackCommandParseError.missingValue(flag)
                }
                let value = arguments[index + 1]
                guard !value.hasPrefix("--") else {
                    throw LocalQwenRepackCommandParseError.missingValue(flag)
                }
                if flag == "--local-source" { source = value } else { output = value }
                index += 2
            case "--text-only":
                textOnly = true
                index += 1
            case "--resume":
                resume = true
                index += 1
            case "--discard-partial":
                discard = true
                index += 1
            case "--preflight-only":
                preflightOnly = true
                index += 1
            default:
                throw LocalQwenRepackCommandParseError.unknown(flag)
            }
        }
        guard let source else {
            throw LocalQwenRepackCommandParseError.missingRequired("--local-source")
        }
        guard let output else {
            throw LocalQwenRepackCommandParseError.missingRequired("--output")
        }
        guard !(resume && discard) else {
            throw LocalQwenRepackCommandParseError.invalid(
                "--resume and --discard-partial are mutually exclusive")
        }
        guard !(preflightOnly && discard) else {
            throw LocalQwenRepackCommandParseError.invalid(
                "--preflight-only and --discard-partial are mutually exclusive")
        }
        let vision = textOnly ? nil : try derivedVisionOutput(textOutput: output)
        let options = LocalQwenStreamingRepackOptions(
            sourceDirectory: source,
            outputDirectory: output,
            visionOutputDirectory: vision,
            resume: resume)
        if preflightOnly { return .preflight(options) }
        return discard ? .discard(options) : .run(options)
    }

    public static func derivedVisionOutput(textOutput: String) throws -> String {
        guard textOutput.hasSuffix(".gturbo"),
              !textOutput.hasSuffix(".vision.gturbo") else {
            throw LocalQwenRepackCommandParseError.invalid(
                "local Qwen --output must end in .gturbo and name the text pack")
        }
        return String(textOutput.dropLast(".gturbo".count)) + ".vision.gturbo"
    }
}

/// Internal injection boundary for bounded fixtures. Production entry points
/// always use `.production`; tests can replace source inspection and writing
/// without routing through a second parser or touching the official payload.
struct LocalQwenStreamingRepackerOperations: Sendable {
    var loadSnapshot: @Sendable (String, LocalPinnedSnapshotIdentity) throws
        -> LocalPinnedSnapshot
    var makePlan: @Sendable (String, String, Bool) throws -> QwenRepackPlan
    var verifyPayload: @Sendable (
        String, [String],
        @escaping @Sendable (UInt64, UInt64) -> Void
    ) throws -> LocalOfficialQwenPayloadIdentity
    var writePack: @Sendable (
        QwenRepackPlan, QwenTransformedPackWriteOptions
    ) throws -> QwenTransformedPackWriteResult

    static let production = Self(
        loadSnapshot: { directory, identity in
            try LocalPinnedSnapshotLoader.load(
                snapshotDirectory: directory, expectedIdentity: identity)
        },
        makePlan: { directory, output, includeVision in
            try QwenRepackPlanner.plan(
                snapshotDirectory: directory,
                outputDirectory: output,
                includeVision: includeVision)
        },
        verifyPayload: { directory, shards, progress in
            try LocalOfficialQwenPayloadVerifier.verify(
                snapshotDirectory: directory,
                shardFilenames: shards,
                progress: progress)
        },
        writePack: { plan, options in
            try QwenTransformedPackWriter.write(plan: plan, options: options)
        })
}

public enum LocalQwenStreamingRepacker {
    private enum DestinationState {
        case absent
        case published
    }

    private struct Context {
        let options: LocalQwenStreamingRepackOptions
        let sourceDirectory: String
        let textPaths: RemoteInstallPaths
        let visionPaths: RemoteInstallPaths?
        let snapshot: LocalPinnedSnapshot
        let plan: QwenRepackPlan
        let processorData: Data?
    }

    private struct Prepared {
        let context: Context
        let payloadIdentity: LocalOfficialQwenPayloadIdentity
        let preflight: LocalQwenStreamingRepackPreflight
    }

    public static func preflight(
        options: LocalQwenStreamingRepackOptions,
        progress: @escaping @Sendable (LocalQwenStreamingRepackProgress) -> Void = { _ in }
    ) throws -> LocalQwenStreamingRepackPreflight {
        try preflight(options: options, operations: .production, progress: progress)
    }

    static func preflight(
        options: LocalQwenStreamingRepackOptions,
        operations: LocalQwenStreamingRepackerOperations,
        progress: @escaping @Sendable (LocalQwenStreamingRepackProgress) -> Void = { _ in }
    ) throws -> LocalQwenStreamingRepackPreflight {
        try prepare(
            options: options, operations: operations, progress: progress).preflight
    }

    public static func run(
        options: LocalQwenStreamingRepackOptions,
        progress: @escaping @Sendable (LocalQwenStreamingRepackProgress) -> Void = { _ in }
    ) throws -> LocalQwenStreamingRepackResult {
        try run(
            options: options, operations: .production,
            progress: progress)
    }

    public static func run(
        options: LocalQwenStreamingRepackOptions,
        preflightReport: @escaping @Sendable (LocalQwenStreamingRepackPreflight) -> Void,
        progress: @escaping @Sendable (LocalQwenStreamingRepackProgress) -> Void
    ) throws -> LocalQwenStreamingRepackResult {
        try run(
            options: options, operations: .production,
            preflightReport: preflightReport, progress: progress)
    }

    static func run(
        options: LocalQwenStreamingRepackOptions,
        operations: LocalQwenStreamingRepackerOperations,
        progress: @escaping @Sendable (LocalQwenStreamingRepackProgress) -> Void = { _ in }
    ) throws -> LocalQwenStreamingRepackResult {
        try run(
            options: options, operations: operations,
            preflightReport: { _ in }, progress: progress)
    }

    static func run(
        options: LocalQwenStreamingRepackOptions,
        operations: LocalQwenStreamingRepackerOperations,
        preflightReport: @escaping @Sendable (LocalQwenStreamingRepackPreflight) -> Void,
        progress: @escaping @Sendable (LocalQwenStreamingRepackProgress) -> Void
    ) throws -> LocalQwenStreamingRepackResult {
        let prepared = try prepare(
            options: options, operations: operations, progress: progress)
        let context = prepared.context
        preflightReport(prepared.preflight)
        progress(.init(stage: .converting))
        let shardNames = context.snapshot.shardFilenames
        let source = context.sourceDirectory
        let result = try operations.writePack(
            context.plan,
            .init(
                outputDirectory: context.textPaths.finalDirectory,
                visionOutputDirectory: context.visionPaths?.finalDirectory,
                visionProcessorData: context.processorData,
                resume: options.resume,
                reserveBytes: options.reserveBytes,
                officialPayloadIdentity: prepared.payloadIdentity,
                durableProgress: { completedUnitCount in
                    progress(.init(
                        stage: .converting,
                        durableCompletedUnitCount: completedUnitCount))
                },
                revalidateOfficialPayload: {
                    progress(.init(stage: .auditing))
                    return try operations.verifyPayload(
                        source, shardNames) { completed, total in
                            progress(.init(
                                stage: .auditing,
                                completedBytes: completed,
                                totalBytes: total))
                        }
                }))
        progress(.init(stage: .publishing))
        return LocalQwenStreamingRepackResult(
            preflight: prepared.preflight,
            textOutputDirectory: context.textPaths.finalDirectory,
            visionOutputDirectory: context.visionPaths?.finalDirectory,
            textManifestSHA256: result.textManifestSHA256,
            textReceiptSHA256: result.textReceiptSHA256,
            textReceiptPath: result.textReceiptPath,
            visionManifestSHA256: result.visionManifestSHA256,
            visionReceiptSHA256: result.visionReceiptSHA256,
            visionReceiptPath: result.visionReceiptPath,
            completedUnitCount: result.completedGroupCount,
            maximumObservedTransformScratchBytes:
                result.maximumObservedTransformScratchBytes)
    }

    public static func discardPartial(options: LocalQwenStreamingRepackOptions) throws {
        try discardPartial(options: options, operations: .production)
    }

    static func discardPartial(
        options: LocalQwenStreamingRepackOptions,
        operations: LocalQwenStreamingRepackerOperations
    ) throws {
        let context = try makeContext(
            options: options, destinationState: .absent, operations: operations)
        let textLock = try InstallLock.acquire(outputDirectory: context.textPaths.finalDirectory)
        defer { withExtendedLifetime(textLock) {} }
        let visionLock = try context.visionPaths.map {
            try InstallLock.acquire(outputDirectory: $0.finalDirectory)
        }
        defer { withExtendedLifetime(visionLock) {} }
        let paths = textLock.paths
        guard try Posix.entryKind(paths.partialDirectory) == .directory,
              try Posix.entryKind(paths.checkpointFile) == .regular else {
            throw RepackError.installStateMissing(path: paths.checkpointFile)
        }
        let checkpoint = try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
        let binding = try checkpoint.requireTransformBinding(path: paths.checkpointFile)
        let destinationBytes = try sum(context.plan.artifacts.map(\.size))
        let destinationIdentity = digest(strings: [
            context.textPaths.finalDirectory,
            context.visionPaths?.finalDirectory ?? "none",
            String(destinationBytes),
            context.plan.canonicalFingerprint,
        ])
        guard checkpoint.repoID == context.plan.provenance.repository,
              checkpoint.requestedRevision == context.plan.provenance.revision,
              checkpoint.resolvedCommit == context.plan.provenance.revision,
              checkpoint.sourceIndexSHA256 == context.plan.provenance.observedIndexSHA256,
              checkpoint.planFingerprint == context.plan.canonicalFingerprint,
              binding.planFingerprint == context.plan.canonicalFingerprint,
              binding.quantizationPolicySHA256
                == context.plan.provenance.quantizationPolicySHA256,
              binding.converterVersion == TransformedTensorWriter.converterVersion,
              binding.destinationIdentity == destinationIdentity,
              binding.destinationBytes == destinationBytes else {
            throw RepackError.installStateIncompatible(
                detail: "saved partial is not owned by this official Qwen plan")
        }
        try QwenTransformedPackWriter.validateOwnedPartialState(
            plan: context.plan,
            textPaths: paths,
            visionPaths: visionLock?.paths,
            processorData: context.processorData,
            checkpoint: checkpoint)
        if let visionPaths = context.visionPaths {
            guard try Posix.entryKind(visionPaths.partialDirectory) == .directory else {
                throw RepackError.installStateCorrupt(
                    path: visionPaths.partialDirectory,
                    detail: "owned text checkpoint has no matching vision partial")
            }
            try FileManager.default.removeItem(atPath: visionPaths.partialDirectory)
            try Posix.fsyncDirectory(visionPaths.parentDirectory)
        }
        try FileManager.default.removeItem(atPath: paths.partialDirectory)
        try FileManager.default.removeItem(atPath: paths.checkpointFile)
        try Posix.fsyncDirectory(paths.parentDirectory)
    }

    public static func verifyPublished(
        options: LocalQwenStreamingRepackOptions
    ) throws -> LocalQwenPublishedVerification {
        try verifyPublished(options: options, operations: .production)
    }

    static func verifyPublished(
        options: LocalQwenStreamingRepackOptions,
        operations: LocalQwenStreamingRepackerOperations
    ) throws -> LocalQwenPublishedVerification {
        let context = try makeContext(
            options: options, destinationState: .published, operations: operations)
        let payload = try operations.verifyPayload(
            context.sourceDirectory, context.snapshot.shardFilenames) { _, _ in }
        let audit = RepackAudit()
        let textManifestPath = (context.textPaths.finalDirectory as NSString)
            .appendingPathComponent("manifest.json")
        let textManifestData = try Posix.readBoundedData(
            textManifestPath, maximumBytes: 4 * 1024 * 1024)
        let textVerified = try GTurboManifestV2Codec.decode(textManifestData)
        let textOutputFiles = Dictionary(uniqueKeysWithValues:
            textVerified.manifest.files.map { path, file in
                (path, RepackAudit.OutputFile(
                    relativePath: path, size: file.size, sha256: file.sha256))
            })
        let expectedText = try QwenTransformedPackWriter.makeTextManifest(
            plan: context.plan, outputFiles: textOutputFiles)
        guard expectedText.manifest == textVerified.manifest else {
            throw RepackError.installStateIncompatible(
                detail: "published Qwen text manifest does not match the pinned plan")
        }
        let textManifestSHA = digest(data: textManifestData)
        let textReceiptData = try readReceipt(root: context.textPaths.finalDirectory)
        try verifyReceipt(
            textReceiptData,
            outputDirectory: context.textPaths.finalDirectory,
            manifestSHA256: textManifestSHA,
            manifestSize: UInt64(textManifestData.count),
            expectedFiles: textVerified.manifest.files,
            sourcePayloadSHA256: payload.shardSetSHA256,
            plan: context.plan,
            compatibleTextManifestSHA256: nil)
        try audit.verifyTransformedPack(
            rootDirectory: context.textPaths.finalDirectory,
            manifest: textVerified,
            receiptData: textReceiptData)

        var visionManifestSHA: String?
        var visionReceiptSHA: String?
        if let visionPaths = context.visionPaths {
            let manifestData = try Posix.readBoundedData(
                (visionPaths.finalDirectory as NSString).appendingPathComponent("manifest.json"),
                maximumBytes: GTurboVisionFormatV2.metadataMaxBytes)
            let verified = try GTurboVisionManifestV2Codec.decode(manifestData)
            let manifestSHA = digest(data: manifestData)
            guard verified.manifest.compatibleTextManifestSHA256 == textManifestSHA else {
                throw RepackError.installStateIncompatible(
                    detail: "published Qwen vision pack is bound to another text manifest")
            }
            guard let vision = context.plan.visionCompanion,
                  let processor = context.processorData,
                  let weights = verified.manifest.files[vision.artifact.relativePath] else {
                throw RepackError.installStateIncompatible(
                    detail: "published Qwen vision manifest has no pinned companion plan")
            }
            let expectedVision = QwenTransformedPackWriter.makeVisionManifest(
                plan: context.plan,
                vision: vision,
                processorBytes: UInt64(processor.count),
                processorSHA256: digest(data: processor),
                textManifestSHA256: textManifestSHA,
                weightsSHA256: weights.sha256)
            guard expectedVision == verified.manifest else {
                throw RepackError.installStateIncompatible(
                    detail: "published Qwen vision manifest does not match the pinned plan")
            }
            let receiptData = try readReceipt(root: visionPaths.finalDirectory)
            try verifyReceipt(
                receiptData,
                outputDirectory: visionPaths.finalDirectory,
                manifestSHA256: manifestSHA,
                manifestSize: UInt64(manifestData.count),
                expectedFiles: verified.manifest.files,
                sourcePayloadSHA256: payload.shardSetSHA256,
                plan: context.plan,
                compatibleTextManifestSHA256: textManifestSHA)
            try audit.verifyTransformedVisionPack(
                rootDirectory: visionPaths.finalDirectory,
                manifest: verified,
                receiptData: receiptData)
            visionManifestSHA = manifestSHA
            visionReceiptSHA = digest(data: receiptData)
        }
        return LocalQwenPublishedVerification(
            textOutputDirectory: context.textPaths.finalDirectory,
            visionOutputDirectory: context.visionPaths?.finalDirectory,
            textManifestSHA256: textManifestSHA,
            textReceiptSHA256: digest(data: textReceiptData),
            visionManifestSHA256: visionManifestSHA,
            visionReceiptSHA256: visionReceiptSHA)
    }

    private static func prepare(
        options: LocalQwenStreamingRepackOptions,
        operations: LocalQwenStreamingRepackerOperations,
        progress: @escaping @Sendable (LocalQwenStreamingRepackProgress) -> Void
    ) throws -> Prepared {
        progress(.init(stage: .planning))
        let context = try makeContext(
            options: options, destinationState: .absent, operations: operations)
        progress(.init(stage: .validatingSource))
        let payload = try operations.verifyPayload(
            context.sourceDirectory,
            context.snapshot.shardFilenames) { completed, total in
                progress(.init(
                    stage: .validatingSource,
                    completedBytes: completed,
                    totalBytes: total))
        }
        progress(.init(stage: .preflighting))
        let resumeUsage = options.resume
            ? try QwenTransformedPackWriter.resumeDiskUsage(
                plan: context.plan,
                outputDirectory: context.textPaths.finalDirectory,
                visionOutputDirectory: context.visionPaths?.finalDirectory)
            : nil
        let destinations = try diskDestinations(
            context: context, payload: payload, resumeUsage: resumeUsage)
        let requirements = try DiskSpaceChecker.requireCombinedAvailableBeforeCreating(
            destinations: destinations,
            reserveBytes: options.reserveBytes)
        let artifactBytes = try sum([
            try sum(context.plan.artifacts.map(\.size)),
            UInt64(context.processorData?.count ?? 0),
        ])
        let requiredBytes = try sum(requirements.map(\.requiredBytes))
        let availableBytes = try sum(requirements.map(\.availableBytes))
        let preflight = LocalQwenStreamingRepackPreflight(
            sourceRepository: context.plan.provenance.repository,
            sourceRevision: context.plan.provenance.revision,
            sourceIndexSHA256: context.plan.provenance.observedIndexSHA256,
            sourcePayloadSHA256: payload.shardSetSHA256,
            planFingerprint: context.plan.canonicalFingerprint,
            quantizationPolicySHA256: context.plan.provenance.quantizationPolicySHA256,
            converterVersion: TransformedTensorWriter.converterVersion,
            textOutputDirectory: context.textPaths.finalDirectory,
            visionOutputDirectory: context.visionPaths?.finalDirectory,
            artifactBytes: artifactBytes,
            requiredBytes: requiredBytes,
            availableBytes: availableBytes,
            diskRequirements: requirements)
        return Prepared(context: context, payloadIdentity: payload, preflight: preflight)
    }

    private static func makeContext(
        options: LocalQwenStreamingRepackOptions,
        destinationState: DestinationState,
        operations: LocalQwenStreamingRepackerOperations
    ) throws -> Context {
        guard options.reserveBytes > 0 else {
            throw RepackError.configurationInvalid(
                detail: "local Qwen protected reserve must be positive")
        }
        guard try Posix.entryKind(options.sourceDirectory) == .directory else {
            throw RepackError.installPathUnsafe(
                path: options.sourceDirectory,
                detail: "local Qwen source must be an existing directory")
        }
        let source = try Posix.physicalPath(options.sourceDirectory)
        let textPaths = try resolvedPaths(options.outputDirectory)
        let visionPaths: RemoteInstallPaths?
        if let visionOutputDirectory = options.visionOutputDirectory {
            visionPaths = try resolvedPaths(visionOutputDirectory)
        } else {
            visionPaths = nil
        }
        try validatePathIsolation(
            source: source, text: textPaths.finalDirectory,
            vision: visionPaths?.finalDirectory)
        for path in [textPaths.finalDirectory, visionPaths?.finalDirectory].compactMap({ $0 }) {
            let kind = try Posix.entryKind(path)
            switch destinationState {
            case .absent:
                guard kind == .absent else {
                    throw RepackError.installStateIncompatible(
                        detail: "completed Qwen destination is protected: \(path)")
                }
            case .published:
                guard kind == .directory else {
                    throw RepackError.installStateMissing(path: path)
                }
            }
        }
        let expectedIdentity = LocalPinnedSnapshotIdentity(source: ModelSourceCatalog.qwen)
        let snapshot = try operations.loadSnapshot(source, expectedIdentity)
        let plan = try operations.makePlan(
            source, textPaths.finalDirectory, visionPaths != nil)
        let processor: Data?
        if visionPaths != nil {
            processor = try Posix.readBoundedData(
                (source as NSString).appendingPathComponent(GTurboVisionFormatV2.processorFile),
                maximumBytes: 32 * 1024 * 1024)
        } else {
            processor = nil
        }
        return Context(
            options: options,
            sourceDirectory: source,
            textPaths: textPaths,
            visionPaths: visionPaths,
            snapshot: snapshot,
            plan: plan,
            processorData: processor)
    }

    private static func resolvedPaths(_ output: String) throws -> RemoteInstallPaths {
        let standardized = URL(fileURLWithPath: output).standardizedFileURL
        let parent = standardized.deletingLastPathComponent().path
        guard try Posix.entryKind(parent) == .directory else {
            throw RepackError.installPathUnsafe(
                path: output, detail: "destination parent must already exist")
        }
        return try RemoteInstallPaths(outputDirectory: standardized.path)
    }

    private static func validatePathIsolation(
        source: String,
        text: String,
        vision: String?
    ) throws {
        let protectedGemma = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("scratch/gemma4.gturbo")
            .standardizedFileURL.path
        let outputs = [text, vision].compactMap { $0 }
        guard Set(outputs).count == outputs.count else {
            throw RepackError.installPathUnsafe(
                path: text, detail: "text and vision destinations must differ")
        }
        for output in outputs {
            let targetsProtectedGemma = output == protectedGemma
                || output.hasSuffix("/scratch/gemma4.gturbo")
            guard !targetsProtectedGemma,
                  output != source,
                  !output.hasPrefix(source + "/") else {
                throw RepackError.installPathUnsafe(
                    path: output,
                    detail: "local Qwen output cannot target Gemma or the source snapshot")
            }
        }
    }

    private static func diskDestinations(
        context: Context,
        payload: LocalOfficialQwenPayloadIdentity,
        resumeUsage: QwenResumeDiskUsage?
    ) throws -> [TransformedDiskSpaceDestination] {
        let plan = context.plan
        let placeholder = String(repeating: "0", count: 64)
        let textArtifacts = plan.artifacts.filter {
            $0.relativePath != plan.visionCompanion?.artifact.relativePath
        }
        let placeholderFiles = Dictionary(uniqueKeysWithValues: textArtifacts.map {
            ($0.relativePath, RepackAudit.OutputFile(
                relativePath: $0.relativePath, size: $0.size, sha256: placeholder))
        })
        let textManifest = try QwenTransformedPackWriter.makeTextManifest(
            plan: plan, outputFiles: placeholderFiles)
        let textManifestData = try GTurboManifestV2Codec.encode(textManifest.manifest)
        let textReceipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: context.textPaths.finalDirectory,
            manifestSha256: placeholder,
            manifestSize: UInt64(textManifestData.count),
            sourceRepoID: plan.provenance.repository,
            sourceRevision: plan.provenance.revision,
            toolVersion: TransformedTensorWriter.converterVersion,
            verificationTimestamp: VerifiedInstallReceiptWriter.deterministicV2Timestamp,
            conversionProvenance: .init(
                sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                sourcePayloadSHA256: payload.shardSetSHA256,
                planFingerprint: plan.canonicalFingerprint,
                quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256,
                converterVersion: TransformedTensorWriter.converterVersion,
                compatibleTextManifestSHA256: nil),
            files: Array(placeholderFiles.values))
        let layout = try QwenTransformedPackWriter.layoutData(plan: plan)
        let textArtifactBytes: UInt64
        if let resumeUsage {
            textArtifactBytes = resumeUsage.textRemainingArtifactBytes
        } else {
            textArtifactBytes = try sum(textArtifacts.map(\.size))
        }
        let textMetadataBytes = try sum([
            UInt64(textManifestData.count), UInt64(textReceipt.count),
        ])
        var result = [TransformedDiskSpaceDestination(
            path: context.textPaths.finalDirectory,
            budget: .init(
                artifactBytes: textArtifactBytes,
                manifestBytes: remainingBytes(
                    total: textMetadataBytes,
                    existing: resumeUsage?.existingTextMetadataBytes ?? 0),
                checkpointBytes: resumeUsage == nil
                    ? RemoteInstallCheckpoint.maximumBytes : 0,
                temporaryFileBytes: UInt64(max(
                    layout.count, textManifestData.count, textReceipt.count,
                    Int(RemoteInstallCheckpoint.maximumBytes)))))]
        if let vision = plan.visionCompanion,
           let visionPaths = context.visionPaths,
           let processor = context.processorData {
            let processorSHA = digest(data: processor)
            let visionManifest = QwenTransformedPackWriter.makeVisionManifest(
                plan: plan, vision: vision,
                processorBytes: UInt64(processor.count),
                processorSHA256: processorSHA,
                textManifestSHA256: placeholder,
                weightsSHA256: placeholder)
            let visionManifestData = try GTurboVisionManifestV2Codec.encode(visionManifest)
            let visionReceipt = try VerifiedInstallReceiptWriter.encode(
                outputDir: visionPaths.finalDirectory,
                manifestSha256: placeholder,
                manifestSize: UInt64(visionManifestData.count),
                sourceRepoID: plan.provenance.repository,
                sourceRevision: plan.provenance.revision,
                toolVersion: TransformedTensorWriter.converterVersion,
                verificationTimestamp: VerifiedInstallReceiptWriter.deterministicV2Timestamp,
                conversionProvenance: .init(
                    sourceIndexSHA256: plan.provenance.observedIndexSHA256,
                    sourcePayloadSHA256: payload.shardSetSHA256,
                    planFingerprint: plan.canonicalFingerprint,
                    quantizationPolicySHA256: plan.provenance.quantizationPolicySHA256,
                    converterVersion: TransformedTensorWriter.converterVersion,
                    compatibleTextManifestSHA256: placeholder),
                files: [
                    .init(
                        relativePath: vision.artifact.relativePath,
                        size: vision.artifact.size,
                        sha256: placeholder),
                    .init(
                        relativePath: GTurboVisionFormatV2.processorFile,
                        size: UInt64(processor.count),
                        sha256: processorSHA),
                ])
            let visionArtifactBytes = resumeUsage?.visionRemainingArtifactBytes
                ?? vision.artifact.size
            let visionMetadataBytes = try sum([
                UInt64(visionManifestData.count), UInt64(visionReceipt.count),
            ])
            let processorBytes = UInt64(processor.count)
            result.append(.init(
                path: visionPaths.finalDirectory,
                budget: .init(
                    artifactBytes: try sum([
                        visionArtifactBytes,
                        remainingBytes(
                            total: processorBytes,
                            existing: resumeUsage?.existingVisionProcessorBytes ?? 0),
                    ]),
                    manifestBytes: remainingBytes(
                        total: visionMetadataBytes,
                        existing: resumeUsage?.existingVisionMetadataBytes ?? 0),
                    checkpointBytes: 0,
                    temporaryFileBytes: UInt64(max(
                        processor.count, visionManifestData.count, visionReceipt.count)))))
        }
        return result
    }

    private static func remainingBytes(total: UInt64, existing: UInt64) -> UInt64 {
        total > existing ? total - existing : 0
    }

    private static func readReceipt(root: String) throws -> Data {
        try Posix.readBoundedData(
            (root as NSString).appendingPathComponent(VerifiedInstallReceiptWriter.fileName),
            maximumBytes: 4 * 1024 * 1024)
    }

    private static func verifyReceipt(
        _ data: Data,
        outputDirectory: String,
        manifestSHA256: String,
        manifestSize: UInt64,
        expectedFiles: [String: GTurboManifestFileV1],
        sourcePayloadSHA256: String,
        plan: QwenRepackPlan,
        compatibleTextManifestSHA256: String?
    ) throws {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["schemaVersion"] as? Int == 1,
              object["manifestSha256"] as? String == manifestSHA256,
              let receiptDirectory = object["modelDirectoryPath"] as? String,
              object["sourceRepoID"] as? String == plan.provenance.repository,
              object["sourceRevision"] as? String == plan.provenance.revision,
              object["sourceIndexSHA256"] as? String == plan.provenance.observedIndexSHA256,
              object["sourcePayloadSHA256"] as? String == sourcePayloadSHA256,
              object["planFingerprint"] as? String == plan.canonicalFingerprint,
              object["quantizationPolicySHA256"] as? String
                == plan.provenance.quantizationPolicySHA256,
              object["converterVersion"] as? String
                == TransformedTensorWriter.converterVersion,
              object["toolVersion"] as? String
                == TransformedTensorWriter.converterVersion,
              object["verificationTimestamp"] as? String
                == VerifiedInstallReceiptWriter.deterministicV2Timestamp,
              object["verificationTimePolicy"] as? String
                == "deterministic-epoch; see operational evidence" else {
            throw RepackError.installStateIncompatible(
                detail: "published Qwen receipt provenance is invalid")
        }
        let canonicalReceiptDirectory = try VerifiedInstallReceiptWriter
            .canonicalPhysicalModelDirectoryPath(receiptDirectory)
        let canonicalOutputDirectory = try VerifiedInstallReceiptWriter
            .canonicalPhysicalModelDirectoryPath(outputDirectory)
        guard canonicalReceiptDirectory == canonicalOutputDirectory else {
            throw RepackError.installStateIncompatible(
                detail: "published Qwen receipt provenance is invalid")
        }
        let observedCompatible = object["compatibleTextManifestSHA256"] as? String
        guard observedCompatible == compatibleTextManifestSHA256 else {
            throw RepackError.installStateIncompatible(
                detail: "published Qwen receipt text binding is invalid")
        }
        try verifyReceiptFiles(
            object,
            expectedFiles: expectedFiles,
            manifestSHA256: manifestSHA256,
            manifestSize: manifestSize)
    }

    private static func verifyReceiptFiles(
        _ receipt: [String: Any],
        expectedFiles: [String: GTurboManifestFileV1],
        manifestSHA256: String,
        manifestSize: UInt64
    ) throws {
        guard let observedFiles = receipt["files"] as? [String: Any],
              observedFiles.count == expectedFiles.count + 1 else {
            throw RepackError.installStateIncompatible(
                detail: "published Qwen receipt file inventory is invalid")
        }
        var expected = expectedFiles
        expected["manifest.json"] = .init(
            size: manifestSize, sha256: manifestSHA256)
        for (path, file) in expected {
            guard let object = observedFiles[path] as? [String: Any],
                  let size = object["size"] as? NSNumber,
                  size.uint64Value == file.size,
                  object["sha256"] as? String == file.sha256 else {
                throw RepackError.installStateIncompatible(
                    detail: "published Qwen receipt entry is invalid: \(path)")
            }
        }
    }

    private static func sum(_ values: [UInt64]) throws -> UInt64 {
        try values.reduce(UInt64(0)) { partial, value in
            let addition = partial.addingReportingOverflow(value)
            guard !addition.overflow else {
                throw RepackError.configurationInvalid(
                    detail: "local Qwen byte total overflows UInt64")
            }
            return addition.partialValue
        }
    }

    private static func digest(strings: [String]) -> String {
        digest(data: Data(strings.joined(separator: "\n").utf8))
    }

    private static func digest(data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }
}
