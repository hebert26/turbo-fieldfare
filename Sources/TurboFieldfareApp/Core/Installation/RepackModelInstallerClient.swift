import Foundation
import TurboFieldfareRepackCore
import Synchronization
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

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

/// Registers an existing pinned BF16 source without copying or changing its shards.
/// A descriptor is metadata only. Installation completes after source trust
/// authenticates every pinned shard and publishes its bound receipt.
public final class LocalQwenModelInstallerClient: Sendable {
    // Progress is replaceable state. Retain only the latest update so a slow
    // main actor cannot replay every 512 KiB hash update after verification.
    // Installed is the final yield and remains buffered until consumed.
    static let eventBufferingPolicy:
        AsyncThrowingStream<AppModelInstallEvent, Error>.Continuation.BufferingPolicy = .bufferingNewest(1)

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
    private let beforeRegistration: @Sendable () async throws -> Void

    var activeInstallID: UUID? {
        taskState.value.withLock { $0?.id }
    }

    public var hasPartialInstall: Bool { false }
    public var canResume: Bool { false }

    public convenience init(entry: AppModelCatalogEntry, sourceDirectory: URL,
                            includesVision: Bool = true) throws {
        try self.init(entry: entry, sourceDirectory: sourceDirectory,
                      includesVision: includesVision, beforeRegistration: {})
    }

    init(entry: AppModelCatalogEntry, sourceDirectory: URL,
         includesVision: Bool,
         beforeRegistration: @escaping @Sendable () async throws -> Void) throws {
        guard entry.id == .qwen3_6, entry.family == .qwen3_6,
              entry.sourceIdentity == AppModelCatalog.entry(for: .qwen3_6).sourceIdentity,
              case .localQwenConversion = entry.installRoute,
              entry.location.textModelURL.path != AppModelCatalog.entry(for: .gemma4).location.textModelURL.path else {
            throw AppModelInstallerRoutingError.localConversionUnavailable(entry.id)
        }
        self.entry = entry
        self.sourceDirectory = sourceDirectory
        self.includesVision = includesVision
        self.beforeRegistration = beforeRegistration
    }

    public func checkInstallRequirement() async throws -> AppModelInstallRequirement {
        let source = sourceDirectory
        let destination = entry.location.textModelURL
        return try await Task.detached(priority: .utility) {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw AppModelInstallerRoutingError.destinationMismatch(
                    expected: "an existing BF16 source directory", actual: source.path)
            }
            // Registration writes only bounded metadata. Payload size is not
            // a required-free-space estimate and total RAM fit is unmeasured.
            return AppModelInstallRequirement(
                probePath: destination.path, requiredBytes: 0, availableBytes: 0)
        }.value
    }

    public func install(resume: Bool) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream<AppModelInstallEvent, Error>(bufferingPolicy: Self.eventBufferingPolicy) {
            (continuation: AsyncThrowingStream<AppModelInstallEvent, Error>.Continuation) in
            let id = UUID()
            let task = Task.detached(priority: .utility) { [self] in
                do {
                    continuation.yield(.checking)
                    try await beforeRegistration()
                    let destination = entry.location.textModelURL
                    let pinned = OfficialQwenSourceIdentity.pinned
                    let descriptor = try OfficialSourceDescriptor(
                        repository: pinned.repository, revision: pinned.revision,
                        storageProfile: pinned.storageProfile,
                        sidecarSHA256: pinned.sidecarSHA256,
                        shards: pinned.shards.map {
                            OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
                        }, sourceRoot: sourceDirectory.path)
                    if FileManager.default.fileExists(atPath: destination.path) {
                        let existing = try OfficialSourceRegistration.inspect(at: destination)
                        guard existing == descriptor else {
                            throw AppModelInstallerRoutingError.destinationMismatch(
                                expected: descriptor.sourceRoot,
                                actual: existing.sourceRoot)
                        }
                    } else {
                        let data = try JSONEncoder().encode(descriptor)
                        _ = try OfficialSourceRegistration.register(
                            markerData: data, at: destination)
                    }
                    try Task.checkCancellation()
                    continuation.yield(.hashingOutput("original BF16 source"))
                    _ = try OfficialSourceTrust.verify(
                        at: destination, policy: .fullSha256) { completed, total in
                            continuation.yield(.copyingPayload(
                                reusedBytes: 0, downloadedThisRunBytes: completed,
                                totalBytes: total))
                        }
                    try Task.checkCancellation()
                    continuation.yield(.installed(destination))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            let previous = taskState.value.withLock { current in
                let previous = current?.task
                current = ActiveInstall(id: id, task: task)
                return previous
            }
            previous?.cancel()
            continuation.onTermination = { [taskState] _ in
                let current = taskState.value.withLock { task -> Task<Void, Never>? in
                    guard task?.id == id else { return nil }
                    let current = task?.task
                    task = nil
                    return current
                }
                current?.cancel()
            }
        }
    }

    public func cancel() {
        let current = taskState.value.withLock { task in
            let current = task?.task
            task = nil
            return current
        }
        current?.cancel()
    }

    /// Registration has no resumable payload. Never delete the source or an
    /// existing logical registration through a generic download discard.
    public func discardPartialInstall() async throws {}

    public func verifyPublished() async throws -> AppModelLocation.Resolved {
        guard AppModelInstallationProbe.status(
            at: entry.location.textModelURL, entry: entry) == .complete else {
            throw AppModelInstallerRoutingError.publishedPathMismatch(
                expected: entry.location.textModelURL.path,
                actual: "BF16 source is not verified")
        }
        return entry.location
    }
}
