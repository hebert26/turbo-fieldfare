import Foundation
import Darwin

public struct DiskSpaceRequirement: Equatable, Sendable {
    public let path: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(path: String, requiredBytes: UInt64, availableBytes: UInt64) {
        self.path = path
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public struct TransformedDiskSpaceBudget: Equatable, Sendable {
    public let artifactBytes: UInt64
    public let manifestBytes: UInt64
    public let checkpointBytes: UInt64
    public let temporaryFileBytes: UInt64

    public init(artifactBytes: UInt64,
                manifestBytes: UInt64,
                checkpointBytes: UInt64,
                temporaryFileBytes: UInt64) {
        self.artifactBytes = artifactBytes
        self.manifestBytes = manifestBytes
        self.checkpointBytes = checkpointBytes
        self.temporaryFileBytes = temporaryFileBytes
    }

    public func requiredBytes() throws -> UInt64 {
        var total: UInt64 = 0
        for value in [artifactBytes, manifestBytes, checkpointBytes, temporaryFileBytes] {
            let addition = total.addingReportingOverflow(value)
            guard !addition.overflow else {
                throw RepackError.configurationInvalid(
                    detail: "transformed disk-space budget overflows UInt64")
            }
            total = addition.partialValue
        }
        return total
    }
}

public struct TransformedDiskSpaceDestination: Equatable, Sendable {
    public let path: String
    public let budget: TransformedDiskSpaceBudget

    public init(path: String, budget: TransformedDiskSpaceBudget) {
        self.path = path
        self.budget = budget
    }
}

public struct CombinedDiskSpaceRequirement: Equatable, Sendable {
    public let probePath: String
    public let paths: [String]
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public var canInstall: Bool { availableBytes >= requiredBytes }
    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum DiskSpaceChecker {
    public static func assess(path: String,
                              bytes: UInt64,
                              reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) throws
        -> DiskSpaceRequirement {
        let requestedDirectory = directoryForProbe(path)
        let probeDirectory = nearestExistingDirectory(requestedDirectory)
        return try requirement(path: probeDirectory,
                               bytes: bytes,
                               reserveBytes: reserveBytes)
    }

    public static func requireAvailable(path: String,
                                        bytes: UInt64,
                                        reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) throws -> DiskSpaceRequirement {
        let dir = directoryForProbe(path)
        try Posix.mkdirP(dir)
        let result = try requirement(path: dir, bytes: bytes, reserveBytes: reserveBytes)
        guard result.canInstall else {
            throw RepackError.diskSpaceInsufficient(path: dir,
                                                    required: result.requiredBytes,
                                                    available: result.availableBytes)
        }
        return result
    }

    /// Authoritative transformed-pack preflight. Unlike the established
    /// installer API above, this never creates a directory, so an insufficient
    /// result cannot leave destination payload state behind.
    public static func requireAvailableBeforeCreating(
        path: String,
        budget: TransformedDiskSpaceBudget,
        reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024
    ) throws -> DiskSpaceRequirement {
        let requestedDirectory = directoryForProbe(path)
        let probeDirectory = nearestExistingDirectory(requestedDirectory)
        let result = try requirement(
            path: probeDirectory,
            bytes: budget.requiredBytes(),
            reserveBytes: reserveBytes)
        guard result.canInstall else {
            throw RepackError.diskSpaceInsufficient(
                path: probeDirectory,
                required: result.requiredBytes,
                available: result.availableBytes)
        }
        return result
    }

    /// Preflights all transformed destinations without creating directories.
    /// Budgets on the same filesystem are summed and the protected reserve is
    /// applied once. This prevents two individually valid sibling packs from
    /// overcommitting the same available bytes.
    public static func requireCombinedAvailableBeforeCreating(
        destinations: [TransformedDiskSpaceDestination],
        reserveBytes: UInt64 = 1 * 1024 * 1024 * 1024
    ) throws -> [CombinedDiskSpaceRequirement] {
        guard !destinations.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: "combined disk-space preflight has no destinations")
        }
        struct VolumeBudget {
            var probePath: String
            var paths: [String]
            var bytes: UInt64
        }
        var grouped: [UInt64: VolumeBudget] = [:]
        for destination in destinations {
            let requestedDirectory = directoryForProbe(destination.path)
            let probe = nearestExistingDirectory(requestedDirectory)
            let device = try deviceIdentity(path: probe)
            let bytes = try destination.budget.requiredBytes()
            var value = grouped[device] ?? VolumeBudget(
                probePath: probe, paths: [], bytes: 0)
            value.paths.append(destination.path)
            let addition = value.bytes.addingReportingOverflow(bytes)
            guard !addition.overflow else {
                throw RepackError.configurationInvalid(
                    detail: "combined disk-space budget overflows UInt64")
            }
            value.bytes = addition.partialValue
            grouped[device] = value
        }

        var results: [CombinedDiskSpaceRequirement] = []
        for value in grouped.values {
            let assessed = try requirement(
                path: value.probePath,
                bytes: value.bytes,
                reserveBytes: reserveBytes)
            let result = CombinedDiskSpaceRequirement(
                probePath: value.probePath,
                paths: value.paths.sorted(),
                requiredBytes: assessed.requiredBytes,
                availableBytes: assessed.availableBytes)
            guard result.canInstall else {
                throw RepackError.diskSpaceInsufficient(
                    path: value.probePath,
                    required: result.requiredBytes,
                    available: result.availableBytes)
            }
            results.append(result)
        }
        return results.sorted { $0.paths.lexicographicallyPrecedes($1.paths) }
    }

    private static func requirement(path: String,
                                    bytes: UInt64,
                                    reserveBytes: UInt64) throws -> DiskSpaceRequirement {
        var st = statfs()
        if statfs(path, &st) != 0 {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        let available = UInt64(st.f_bavail) * UInt64(st.f_bsize)
        let sum = bytes.addingReportingOverflow(reserveBytes)
        guard !sum.overflow else {
            throw RepackError.configurationInvalid(
                detail: "disk-space requirement overflows UInt64")
        }
        let required = sum.partialValue
        return DiskSpaceRequirement(path: path,
                                    requiredBytes: required,
                                    availableBytes: available)
    }

    private static func nearestExistingDirectory(_ path: String) -> String {
        var url = URL(fileURLWithPath: path, isDirectory: true).standardizedFileURL
        while !FileManager.default.fileExists(atPath: url.path) {
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path { return url.path }
            url = parent
        }
        return url.path
    }

    private static func directoryForProbe(_ path: String) -> String {
        let ns = path as NSString
        let ext = ns.pathExtension
        if ext.isEmpty {
            return path
        }
        return ns.deletingLastPathComponent
    }

    private static func deviceIdentity(path: String) throws -> UInt64 {
        var info = stat()
        guard stat(path, &info) == 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        return UInt64(info.st_dev)
    }
}
