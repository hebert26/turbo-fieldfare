import Foundation

public struct RemoteCompletedRange: Codable, Sendable, Equatable {
    public let id: String
    public let destinationDigest: String
    public let sourceBytes: UInt64
    public let destinationBytes: UInt64
}

public struct RemoteTransformBinding: Codable, Sendable, Equatable {
    private enum CodingKeys: String, CodingKey {
        case sourcePayloadSHA256 = "s"
        case converterVersion = "c"
        case quantizationPolicySHA256 = "q"
        case planFingerprint = "p"
        case destinationIdentity = "i"
        case destinationBytes = "b"
    }

    public let sourcePayloadSHA256: String
    public let converterVersion: String
    public let quantizationPolicySHA256: String
    public let planFingerprint: String
    public let destinationIdentity: String
    public let destinationBytes: UInt64

    public init(sourcePayloadSHA256: String,
                converterVersion: String,
                quantizationPolicySHA256: String,
                planFingerprint: String,
                destinationIdentity: String,
                destinationBytes: UInt64) {
        self.sourcePayloadSHA256 = sourcePayloadSHA256
        self.converterVersion = converterVersion
        self.quantizationPolicySHA256 = quantizationPolicySHA256
        self.planFingerprint = planFingerprint
        self.destinationIdentity = destinationIdentity
        self.destinationBytes = destinationBytes
    }

    fileprivate var isValid: Bool {
        Self.isSHA256(sourcePayloadSHA256)
            && !converterVersion.isEmpty
            && Self.isSHA256(quantizationPolicySHA256)
            && Self.isSHA256(planFingerprint)
            && Self.isSHA256(destinationIdentity)
            && destinationBytes > 0
    }

    fileprivate static func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }
}

public struct RemoteTransformProgress: Codable, Sendable, Equatable {
    public static let currentVersion = 2

    private enum CodingKeys: String, CodingKey {
        case version = "v"
        case requestIndex = "r"
        case requestID = "i"
        case completedUnitCount = "u"
        case totalCompletedUnitCount = "t"
        case sourceChainSHA256 = "s"
        case destinationChainSHA256 = "d"
    }

    public let version: Int
    public let requestIndex: UInt64
    public let requestID: String
    public let completedUnitCount: UInt64
    public let totalCompletedUnitCount: UInt64
    public let sourceChainSHA256: String
    public let destinationChainSHA256: String

    public init(requestIndex: UInt64,
                requestID: String,
                completedUnitCount: UInt64,
                totalCompletedUnitCount: UInt64,
                sourceChainSHA256: String,
                destinationChainSHA256: String) {
        self.version = Self.currentVersion
        self.requestIndex = requestIndex
        self.requestID = requestID
        self.completedUnitCount = completedUnitCount
        self.totalCompletedUnitCount = totalCompletedUnitCount
        self.sourceChainSHA256 = sourceChainSHA256
        self.destinationChainSHA256 = destinationChainSHA256
    }

    fileprivate var isValid: Bool {
        version == Self.currentVersion
            && RemoteTransformBinding.isSHA256(requestID)
            && completedUnitCount <= totalCompletedUnitCount
            && RemoteTransformBinding.isSHA256(sourceChainSHA256)
            && RemoteTransformBinding.isSHA256(destinationChainSHA256)
    }
}

public struct RemoteInstallCheckpoint: Codable, Sendable, Equatable {
    public static let schemaVersion = 1
    public static let maximumBytes: UInt64 = 8 * 1024 * 1024

    public let schema: Int
    public let repoID: String
    public let requestedRevision: String
    public let resolvedCommit: String
    public let sourceIndexSHA256: String
    public let planFingerprint: String
    public let totalSourceBytes: UInt64
    public var completedRanges: [RemoteCompletedRange]
    /// Absent in established Gemma schema-1 checkpoints. A transformed resume
    /// must require both this binding and current compact progress explicitly.
    public let transformBinding: RemoteTransformBinding?
    public var transformProgress: RemoteTransformProgress?

    public init(repoID: String,
                requestedRevision: String,
                resolvedCommit: String,
                sourceIndexSHA256: String,
                planFingerprint: String,
                totalSourceBytes: UInt64,
                completedRanges: [RemoteCompletedRange] = [],
                transformBinding: RemoteTransformBinding? = nil,
                transformProgress: RemoteTransformProgress? = nil) {
        self.schema = Self.schemaVersion
        self.repoID = repoID
        self.requestedRevision = requestedRevision
        self.resolvedCommit = resolvedCommit
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.planFingerprint = planFingerprint
        self.totalSourceBytes = totalSourceBytes
        self.completedRanges = completedRanges
        self.transformBinding = transformBinding
        self.transformProgress = transformProgress
    }

    public static func load(from path: String) throws -> Self {
        let data = try Posix.readBoundedData(path, maximumBytes: maximumBytes)
        do {
            let checkpoint = try JSONDecoder().decode(Self.self, from: data)
            try checkpoint.validate(path: path)
            return checkpoint
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.installStateCorrupt(path: path, detail: "\(error)")
        }
    }

    public func write(to path: String, parentDirectory: String) throws {
        try validate(path: path)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        guard data.count <= Self.maximumBytes else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "checkpoint exceeds \(Self.maximumBytes)-byte cap")
        }
        try Posix.atomicWrite(data, to: path, durableIn: parentDirectory)
    }

    public func matches(repoID: String,
                        requestedRevision: String,
                        sourceIndexSHA256: String,
                        planFingerprint: String) -> Bool {
        self.repoID == repoID
            && self.requestedRevision == requestedRevision
            && self.sourceIndexSHA256 == sourceIndexSHA256
            && self.planFingerprint == planFingerprint
    }

    /// Strict transformed-resume matching. The established `matches` method
    /// intentionally remains unchanged for Gemma range-copy callers.
    public func matchesTransform(
        repoID: String,
        requestedRevision: String,
        resolvedCommit: String,
        binding: RemoteTransformBinding
    ) -> Bool {
        self.repoID == repoID
            && self.requestedRevision == requestedRevision
            && self.resolvedCommit == resolvedCommit
            && self.planFingerprint == binding.planFingerprint
            && transformBinding == binding
    }

    public func requireTransformBinding(path: String) throws -> RemoteTransformBinding {
        try validate(path: path)
        guard let transformBinding else {
            throw RepackError.installStateIncompatible(
                detail: "saved install has no transformed-source binding")
        }
        return transformBinding
    }

    public func validatedTransformProgress(path: String) throws -> RemoteTransformProgress {
        try validate(path: path)
        guard transformBinding != nil, let transformProgress else {
            throw RepackError.installStateIncompatible(
                detail: "saved install has no current transformed progress")
        }
        return transformProgress
    }

    public func validatedDestinationBytes(maximum: UInt64, path: String) throws -> UInt64 {
        try validate(path: path)
        var total: UInt64 = 0
        for range in completedRanges {
            let sum = total.addingReportingOverflow(range.destinationBytes)
            guard !sum.overflow, sum.partialValue <= maximum else {
                throw RepackError.installStateCorrupt(
                    path: path,
                    detail: "completed destination bytes exceed the installed model")
            }
            total = sum.partialValue
        }
        return total
    }

    private func validate(path: String) throws {
        guard schema == Self.schemaVersion,
              !repoID.isEmpty,
              !requestedRevision.isEmpty,
              resolvedCommit.count == 40,
              sourceIndexSHA256.count == 64,
              planFingerprint.count == 64,
              totalSourceBytes > 0 else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "invalid checkpoint identity")
        }
        let ids = completedRanges.map(\.id)
        guard Set(ids).count == ids.count,
              completedRanges.allSatisfy({
                  !$0.id.isEmpty
                      && $0.destinationDigest.count == 64
                      && $0.sourceBytes > 0
                      && $0.destinationBytes > 0
              }) else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "invalid completed range")
        }
        if let transformBinding {
            guard transformBinding.isValid,
                  transformBinding.planFingerprint == planFingerprint,
                  transformProgress?.isValid == true else {
                throw RepackError.installStateCorrupt(
                    path: path, detail: "invalid transformed checkpoint binding")
            }
        } else if transformProgress != nil {
            throw RepackError.installStateCorrupt(
                path: path, detail: "transformed progress has no binding")
        }

        var sourceTotal: UInt64 = 0
        var destinationTotal: UInt64 = 0
        for range in completedRanges {
            let nextSource = sourceTotal.addingReportingOverflow(range.sourceBytes)
            let nextDestination = destinationTotal.addingReportingOverflow(
                range.destinationBytes)
            guard !nextSource.overflow,
                  nextSource.partialValue <= totalSourceBytes,
                  !nextDestination.overflow else {
                throw RepackError.installStateCorrupt(
                    path: path,
                    detail: "completed range byte totals are invalid")
            }
            sourceTotal = nextSource.partialValue
            destinationTotal = nextDestination.partialValue
        }
    }
}
