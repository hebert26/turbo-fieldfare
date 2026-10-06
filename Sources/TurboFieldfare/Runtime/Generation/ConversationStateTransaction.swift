import Foundation

public struct ConversationTransactionID: Hashable, Sendable {
    fileprivate let rawValue: UInt64

    init(rawValue: UInt64) {
        self.rawValue = rawValue
    }
}

public enum ConversationStateTransactionError: Error, Equatable, Sendable {
    case busy
    case staleTransaction
    case invalidBoundary(String)
    case contextExceeded(requested: Int, maximum: Int)
    case restoreFailed(String)
    case unsupportedFamily
    case tokenCodecUnavailable
}

public struct ConversationStateMetrics: Equatable, Sendable {
    public let retainedTokenIDs: [Int32]
    public let consumedTokenCount: Int
    public let pendingTokenCount: Int
    public let logicalStateBytes: UInt64

    public init(
        retainedTokenIDs: [Int32],
        consumedTokenCount: Int,
        pendingTokenCount: Int,
        logicalStateBytes: UInt64
    ) {
        self.retainedTokenIDs = retainedTokenIDs
        self.consumedTokenCount = consumedTokenCount
        self.pendingTokenCount = pendingTokenCount
        self.logicalStateBytes = logicalStateBytes
    }
}

public struct ConversationStateStatus: Equatable, Sendable {
    public let committed: ConversationStateMetrics
    public let working: ConversationStateMetrics?
    public let activeTransaction: ConversationTransactionID?

    public init(
        committed: ConversationStateMetrics,
        working: ConversationStateMetrics?,
        activeTransaction: ConversationTransactionID?
    ) {
        self.committed = committed
        self.working = working
        self.activeTransaction = activeTransaction
    }
}

public protocol ConversationStateTransaction: AnyObject, Sendable {
    func begin() async throws -> ConversationTransactionID
    func prefill(
        _ tokenIDs: [Int32],
        transaction: ConversationTransactionID,
        onProgress: @Sendable (Int, Int) async -> Void
    ) async throws
    /// Accepts a sampled token. Any previously accepted pending token is first
    /// consumed by the model exactly once.
    func advance(_ tokenID: Int32, transaction: ConversationTransactionID) async throws
    /// Removes an accepted suffix before commit. Pending tokens are removed
    /// before consumed tokens; consumed removal replays from the begin boundary.
    func removeSuffix(tokenCount: Int, transaction: ConversationTransactionID) async throws
    /// Validates before one irreversible synchronous state swap. It never
    /// suspends or throws after that swap.
    func commit(transaction: ConversationTransactionID) async throws -> ConversationStateMetrics
    func rollback(transaction: ConversationTransactionID) async throws
    func rebuildCheckpoint(
        retaining tokenIDs: [Int32],
        transaction: ConversationTransactionID,
        onProgress: @Sendable (Int, Int) async -> Void
    ) async throws
    func reset() async throws
    func status() async -> ConversationStateStatus
}

public enum TokenizedConversationProgress: Equatable, Sendable {
    case prefill(done: Int, total: Int)
    case accepted(index: Int, tokenID: Int32)
    case checkpoint(done: Int, total: Int)
}

public struct TokenizedConversationTurn: Equatable, Sendable {
    public let promptTokenIDs: [Int32]
    /// Generated IDs in acceptance order. The final accepted ID remains
    /// pending until the next turn unless suffix removal discards it.
    public let generatedTokenIDs: [Int32]
    /// Number of generated IDs hidden at the tail. It may include the pending
    /// final ID and consumed predecessors, but cannot cross into the prompt.
    public let hiddenSuffixTokenCount: Int

    public init(
        promptTokenIDs: [Int32],
        generatedTokenIDs: [Int32],
        hiddenSuffixTokenCount: Int = 0
    ) {
        self.promptTokenIDs = promptTokenIDs
        self.generatedTokenIDs = generatedTokenIDs
        self.hiddenSuffixTokenCount = hiddenSuffixTokenCount
    }
}

public enum TokenizedConversationStopReason: Equatable, Sendable {
    case complete
    case softStop
}

public struct TokenizedConversationResult: Equatable, Sendable {
    public let reason: TokenizedConversationStopReason
    public let metrics: ConversationStateMetrics
    public let acceptedGeneratedTokenIDs: [Int32]

    public init(
        reason: TokenizedConversationStopReason,
        metrics: ConversationStateMetrics,
        acceptedGeneratedTokenIDs: [Int32]
    ) {
        self.reason = reason
        self.metrics = metrics
        self.acceptedGeneratedTokenIDs = acceptedGeneratedTokenIDs
    }
}

public enum TokenizedConversationEvent: Equatable, Sendable {
    case progress(TokenizedConversationProgress)
    case finished(TokenizedConversationResult)
}

public enum QwenCheckpointReason: Equatable, Sendable {
    case capacity
    case sustainedSlowDecode
}

public struct TokenizedCheckpointRequest: Equatable, Sendable {
    public let retainedTokenIDs: [Int32]
    public let reason: QwenCheckpointReason

    public init(retainedTokenIDs: [Int32], reason: QwenCheckpointReason) {
        self.retainedTokenIDs = retainedTokenIDs
        self.reason = reason
    }
}

public struct TokenizedCheckpointResult: Equatable, Sendable {
    public let needed: Bool
    public let committed: Bool
    public let metrics: ConversationStateMetrics

    public init(needed: Bool, committed: Bool, metrics: ConversationStateMetrics) {
        self.needed = needed
        self.committed = committed
        self.metrics = metrics
    }
}

public enum TokenizedCheckpointEvent: Equatable, Sendable {
    case progress(TokenizedConversationProgress)
    case finished(TokenizedCheckpointResult)
}
