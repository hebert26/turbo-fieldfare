import Foundation
import TurboFieldfareRepackCore
import Synchronization

public enum AppModelInstallerRoutingError: Error, Equatable, CustomStringConvertible, Sendable {
    case remoteInstallUnavailable(AppModelID)
    case localConversionUnavailable(AppModelID)
    case destinationMismatch(expected: String, actual: String)
    case publishedPathMismatch(expected: String, actual: String)

    public var description: String {
        switch self {
        case .remoteInstallUnavailable(let id):
            "remote installation is unavailable for \(id.rawValue)"
        case .localConversionUnavailable(let id):
            "local conversion is unavailable for \(id.rawValue)"
        case .destinationMismatch(let expected, let actual):
            "installation destination \(actual) does not match catalog path \(expected)"
        case .publishedPathMismatch(let expected, let actual):
            "published model path \(actual) does not match catalog path \(expected)"
        }
    }
}

public final class RepackModelInstallerClient: AppModelInstallerClient, Sendable {
    typealias InstallRunner = @Sendable (
        URL,
        @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> URL
    typealias DiscardRunner = @Sendable (URL) async throws -> Void

    private struct ActiveInstall: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }

    private final class InstallTaskState: Sendable {
        let value = Mutex<ActiveInstall?>(nil)
    }

    public let descriptor: AppModelInstallDescriptor
    public let expectedOutputDirectory: URL?
    private let runInstall: InstallRunner
    private let runDiscard: DiscardRunner
    private let taskState = InstallTaskState()

    public convenience init(entry: AppModelCatalogEntry) throws {
        guard entry.id == .gemma4, entry.family == .gemma4,
              case let .remoteRepack(text, _) = entry.installRoute,
              text == .default else {
            throw AppModelInstallerRoutingError.remoteInstallUnavailable(entry.id)
        }
        self.init(
            descriptor: text,
            expectedOutputDirectory: entry.location.textModelURL)
    }

    public convenience init(descriptor: AppModelInstallDescriptor = .default) {
        self.init(descriptor: descriptor, expectedOutputDirectory: nil)
    }

    private init(
        descriptor: AppModelInstallDescriptor,
        expectedOutputDirectory: URL?
    ) {
        self.descriptor = descriptor
        self.expectedOutputDirectory = expectedOutputDirectory?.standardizedFileURL
        self.runInstall = { outputDirectory, progress in
            let paths = try RemoteInstallPaths(outputDirectory: outputDirectory.path)
            let resume = FileManager.default.fileExists(atPath: paths.checkpointFile)
            let options = RemoteStreamingRepackOptions(
                repoID: descriptor.repoID,
                revision: descriptor.revision,
                outputDir: outputDirectory.path,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                requireKnownSource: true,
                minFreeReserveBytes: descriptor.reserveBytes,
                overwrite: true,
                resume: resume)
            let result = try await RemoteStreamingRepacker(options: options).run(progress: progress)
            return URL(fileURLWithPath: result.outputDir).standardizedFileURL
        }
        self.runDiscard = { outputDirectory in
            try RemoteStreamingRepacker.discardPartial(
                outputDirectory: outputDirectory.path)
        }
    }

    init(descriptor: AppModelInstallDescriptor = .default,
         runInstall: @escaping InstallRunner,
         runDiscard: @escaping DiscardRunner = { _ in }) {
        self.descriptor = descriptor
        self.expectedOutputDirectory = nil
        self.runInstall = runInstall
        self.runDiscard = runDiscard
    }

    public func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        let outputDirectory = try validated(outputDirectory)
        let saved = try RemoteStreamingRepacker.inspectPersistentInstall(
            outputDirectory: outputDirectory.path,
            repoID: descriptor.repoID,
            requestedRevision: descriptor.revision)
        let remainingBytes: UInt64
        if let saved {
            let checkpointPath = try RemoteInstallPaths(
                outputDirectory: outputDirectory.path).checkpointFile
            let reused = try saved.validatedDestinationBytes(
                maximum: descriptor.installedBytes,
                path: checkpointPath)
            remainingBytes = descriptor.installedBytes > reused
                ? descriptor.installedBytes - reused
                : 0
        } else {
            remainingBytes = descriptor.installedBytes
        }
        let requested = remainingBytes.addingReportingOverflow(
            descriptor.rangeStagingBytes)
        guard !requested.overflow else {
            throw RepackError.configurationInvalid(
                detail: "model install requirement overflows UInt64")
        }
        let requirement = try DiskSpaceChecker.assess(
            path: outputDirectory.path,
            bytes: requested.partialValue,
            reserveBytes: descriptor.reserveBytes)
        return AppModelInstallRequirement(probePath: requirement.path,
                                          requiredBytes: requirement.requiredBytes,
                                          availableBytes: requirement.availableBytes)
    }

    public func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task { [runInstall] in
                do {
                    let outputDirectory = try validated(outputDirectory)
                    continuation.yield(.checking)
                    let completedDirectory = try await runInstall(outputDirectory) { progress in
                        continuation.yield(Self.event(for: progress))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.installed(completedDirectory))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            let previous = taskState.value.withLock { active in
                let previous = active?.task
                active = ActiveInstall(id: id, task: task)
                return previous
            }
            previous?.cancel()

            continuation.onTermination = { [taskState] _ in
                let task = taskState.value.withLock { active -> Task<Void, Never>? in
                    guard active?.id == id else { return nil }
                    defer { active = nil }
                    return active?.task
                }
                task?.cancel()
            }
        }
    }

    public func cancel() {
        let task = taskState.value.withLock { active -> Task<Void, Never>? in
            defer { active = nil }
            return active?.task
        }
        task?.cancel()
    }

    public func discardPartialInstall(outputDirectory: URL) async throws {
        let directory = try validated(outputDirectory)
        try await Task.detached(priority: .utility) { [runDiscard] in
            try await runDiscard(directory)
        }.value
    }

    private func validated(_ outputDirectory: URL) throws -> URL {
        let output = outputDirectory.standardizedFileURL
        if let expectedOutputDirectory, output.path != expectedOutputDirectory.path {
            throw AppModelInstallerRoutingError.destinationMismatch(
                expected: expectedOutputDirectory.path,
                actual: output.path)
        }
        return output
    }

    static func event(for progress: ModelInstallProgress) -> AppModelInstallEvent {
        switch progress {
        case .downloadingMetadata:
            return .downloadingMetadata
        case .planning:
            return .planning
        case .checkingDisk:
            return .checking
        case .reservingOutput:
            return .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            return .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            return .hashingOutput(file)
        case .finalizing:
            return .finalizing
        }
    }
}

/// App-side owner of one pinned local Qwen conversion. The source and both
/// destinations are captured at construction so resume, cancellation, discard,
/// and verification cannot drift onto Gemma or another catalog entry.
public final class LocalQwenModelInstallerClient: Sendable {
    private struct ActiveInstall: Sendable {
        let id: UUID
        let task: Task<Void, Never>
    }

    private final class InstallTaskState: Sendable {
        let value = Mutex<ActiveInstall?>(nil)
    }

    public let entry: AppModelCatalogEntry
    public let sourceDirectory: URL
    public let includesVision: Bool
    private let taskState = InstallTaskState()

    public var hasPartialInstall: Bool {
        let locations = [entry.location.textModelURL]
            + (includesVision ? [entry.location.visionModelURL] : [])
        return locations.contains { location in
            let output = location.standardizedFileURL.path
            return FileManager.default.fileExists(atPath: output + ".partial")
                || FileManager.default.fileExists(atPath: output + ".resume.json")
        }
    }

    public var canResume: Bool {
        let text = entry.location.textModelURL.standardizedFileURL.path
        guard FileManager.default.fileExists(atPath: text + ".partial"),
              FileManager.default.fileExists(atPath: text + ".resume.json") else {
            return false
        }
        return !includesVision || FileManager.default.fileExists(
            atPath: entry.location.visionModelURL.standardizedFileURL.path + ".partial")
    }

    public init(
        entry: AppModelCatalogEntry,
        sourceDirectory: URL,
        includesVision: Bool = true
    ) throws {
        guard entry.id == .qwen3_6,
              entry.family == .qwen3_6,
              entry.sourceIdentity == AppModelCatalog.entry(for: .qwen3_6).sourceIdentity,
              case .localQwenConversion = entry.installRoute,
              Self.hasIsolatedQwenLocations(entry.location) else {
            throw AppModelInstallerRoutingError.localConversionUnavailable(entry.id)
        }
        self.entry = entry
        self.sourceDirectory = sourceDirectory.standardizedFileURL
        self.includesVision = includesVision
    }

    private static func hasIsolatedQwenLocations(
        _ location: AppModelLocation.Resolved
    ) -> Bool {
        let text = location.textModelURL.standardizedFileURL
        let vision = location.visionModelURL.standardizedFileURL
        let gemma = AppModelCatalog.entry(for: .gemma4).location
        let suffix = ".gturbo"
        let name = text.lastPathComponent
        guard name != "gemma4.gturbo",
              text.path != gemma.textModelURL.path,
              vision.path != gemma.visionModelURL.path,
              name.hasSuffix(suffix), name.count > suffix.count else { return false }
        let expectedVision = text.deletingLastPathComponent()
            .appendingPathComponent(
                "\(name.dropLast(suffix.count)).vision.gturbo",
                isDirectory: true)
            .standardizedFileURL
        return vision.path == expectedVision.path
    }

    public func checkInstallRequirement() async throws -> AppModelInstallRequirement {
        try await runCancellableDetached { [self] in
            let preflight = try LocalQwenStreamingRepacker.preflight(
                options: options(resume: canResume))
            let probePath = preflight.diskRequirements.first?.probePath
                ?? entry.location.textModelURL.deletingLastPathComponent().path
            return AppModelInstallRequirement(
                probePath: probePath,
                requiredBytes: preflight.requiredBytes,
                availableBytes: preflight.availableBytes)
        }
    }

    public func install(
        resume: Bool
    ) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { continuation in
            let id = UUID()
            let task = Task.detached(priority: .utility) { [self] in
                do {
                    continuation.yield(.checking)
                    let result = try LocalQwenStreamingRepacker.run(
                        options: options(resume: resume)) { progress in
                            continuation.yield(Self.event(for: progress))
                        }
                    try Task.checkCancellation()
                    let published = URL(
                        fileURLWithPath: result.textOutputDirectory,
                        isDirectory: true).standardizedFileURL
                    try validatePublishedPath(published)
                    if includesVision {
                        let actual = result.visionOutputDirectory.map {
                            URL(fileURLWithPath: $0, isDirectory: true)
                                .standardizedFileURL
                        }
                        guard actual?.path == entry.location.visionModelURL.path else {
                            throw AppModelInstallerRoutingError.publishedPathMismatch(
                                expected: entry.location.visionModelURL.path,
                                actual: actual?.path ?? "missing")
                        }
                    } else if let actual = result.visionOutputDirectory {
                        throw AppModelInstallerRoutingError.publishedPathMismatch(
                            expected: "no vision output",
                            actual: actual)
                    }
                    try Task.checkCancellation()
                    continuation.yield(.installed(entry.location.textModelURL))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            let previous = taskState.value.withLock { active in
                let previous = active?.task
                active = ActiveInstall(id: id, task: task)
                return previous
            }
            previous?.cancel()
            continuation.onTermination = { [taskState] _ in
                let task = taskState.value.withLock { active -> Task<Void, Never>? in
                    guard active?.id == id else { return nil }
                    defer { active = nil }
                    return active?.task
                }
                task?.cancel()
            }
        }
    }

    public func cancel() {
        let task = taskState.value.withLock { active -> Task<Void, Never>? in
            defer { active = nil }
            return active?.task
        }
        task?.cancel()
    }

    public func discardPartialInstall() async throws {
        try await runCancellableDetached { [self] in
            try LocalQwenStreamingRepacker.discardPartial(
                options: options(resume: true))
        }
    }

    public func verifyPublished() async throws -> AppModelLocation.Resolved {
        try await runCancellableDetached { [self] in
            let verified = try LocalQwenStreamingRepacker.verifyPublished(
                options: options(resume: true))
            return try validate(verified)
        }
    }

    private func runCancellableDetached<T: Sendable>(
        _ body: @escaping @Sendable () throws -> T
    ) async throws -> T {
        let work = Task.detached(priority: .utility, operation: body)
        return try await withTaskCancellationHandler {
            try await work.value
        } onCancel: {
            work.cancel()
        }
    }

    private func options(resume: Bool) -> LocalQwenStreamingRepackOptions {
        let descriptor: AppLocalQwenInstallDescriptor
        switch entry.installRoute {
        case .localQwenConversion(let value): descriptor = value
        case .remoteRepack: preconditionFailure("validated by initializer")
        }
        return LocalQwenStreamingRepackOptions(
            sourceDirectory: sourceDirectory.path,
            outputDirectory: entry.location.textModelURL.path,
            visionOutputDirectory: includesVision
                ? entry.location.visionModelURL.path : nil,
            resume: resume,
            reserveBytes: descriptor.reserveBytes)
    }

    private func validatePublishedPath(_ path: URL) throws {
        guard path.path == entry.location.textModelURL.path else {
            throw AppModelInstallerRoutingError.publishedPathMismatch(
                expected: entry.location.textModelURL.path,
                actual: path.path)
        }
    }

    private func validate(
        _ verified: LocalQwenPublishedVerification
    ) throws -> AppModelLocation.Resolved {
        let text = URL(
            fileURLWithPath: verified.textOutputDirectory,
            isDirectory: true).standardizedFileURL
        try validatePublishedPath(text)
        if includesVision {
            let actual = verified.visionOutputDirectory.map {
                URL(fileURLWithPath: $0, isDirectory: true).standardizedFileURL
            }
            guard actual?.path == entry.location.visionModelURL.path else {
                throw AppModelInstallerRoutingError.publishedPathMismatch(
                    expected: entry.location.visionModelURL.path,
                    actual: actual?.path ?? "missing")
            }
        } else if let actual = verified.visionOutputDirectory {
            throw AppModelInstallerRoutingError.publishedPathMismatch(
                expected: "no vision output",
                actual: actual)
        }
        return entry.location
    }

    private static func event(
        for progress: LocalQwenStreamingRepackProgress
    ) -> AppModelInstallEvent {
        switch progress.stage {
        case .validatingSource, .preflighting:
            .checking
        case .planning:
            .planning
        case .converting:
            .copyingPayload(
                reusedBytes: 0,
                downloadedThisRunBytes: progress.completedBytes,
                totalBytes: progress.totalBytes)
        case .auditing:
            .hashingOutput("Qwen artifact")
        case .publishing:
            .finalizing
        }
    }
}
