import Metal

/// Atomic conversation state for the concrete Qwen text runner.
///
/// The runner owns GPU/decoder state. This actor is the sole owner of the
/// conversation journal, accepted-but-unconsumed token, and transaction
/// boundaries. The two stores are validated together before publication.
private struct QwenConversationLogitsBuffer: @unchecked Sendable {
    let value: MTLBuffer
}

/// Internal settled-state observation for transaction recovery tests. Public
/// conversation status intentionally remains independent of KV layout.
struct QwenImageOccurrenceDiagnostics: Equatable, Sendable {
    let segmentID: UUID
    let ownerAllocationID: UUID
    let imageDigest: String
    let processorDigest: String
    let absoluteTokenRange: Range<Int>
    let exactPositions: [QwenMRoPEPosition]
}

struct QwenConversationLineageDiagnostics: Equatable, Sendable {
    let visibleRows: Int
    let logicalBytes: Int
    let allocationIDs: Set<UUID>
    let ownedRequestedBytes: Int
    let provenanceRequestedBytes: Int
    let occurrences: [QwenImageOccurrenceDiagnostics]
    let liveOwnerRows: Int
    let liveRequestedBytes: Int
    let highWaterOwnerRows: Int
    let highWaterRequestedBytes: Int
}

struct QwenLineageProvenancePlan: Hashable, Sendable {
    static let none = QwenLineageProvenancePlan(
        uncheckedTokenCount: 0, positionCount: 0, featureOverrideCount: 0,
        requestedPayloadBytes: 0)

    let tokenCount: Int
    let positionCount: Int
    let featureOverrideCount: Int
    let requestedPayloadBytes: Int

    private init(
        uncheckedTokenCount tokenCount: Int,
        positionCount: Int,
        featureOverrideCount: Int,
        requestedPayloadBytes: Int
    ) {
        self.tokenCount = tokenCount
        self.positionCount = positionCount
        self.featureOverrideCount = featureOverrideCount
        self.requestedPayloadBytes = requestedPayloadBytes
    }

    init(tokenCount: Int, positionCount: Int, featureOverrideCount: Int) throws {
        guard tokenCount >= 0, positionCount >= 0, featureOverrideCount >= 0 else {
            throw QwenConversationLineageError.invalidReservation
        }
        var total = 32 // UUID, base offset, and retained text delta.
        for (count, stride) in [
            (tokenCount, MemoryLayout<Int32>.stride),
            (positionCount, MemoryLayout<QwenMRoPEPosition>.stride),
            (featureOverrideCount, MemoryLayout<QwenPreparedFeatureOverride>.stride),
        ] {
            let (payload, productOverflow) = count.multipliedReportingOverflow(by: stride)
            let (next, sumOverflow) = total.addingReportingOverflow(payload)
            guard !productOverflow, !sumOverflow else {
                throw QwenConversationLineageError.requestedBytesExceeded(
                    requested: Int.max,
                    maximum: QwenVisionResourceLimits.provisional.maximumMutablePreparedBytes)
            }
            total = next
        }
        self.tokenCount = tokenCount
        self.positionCount = positionCount
        self.featureOverrideCount = featureOverrideCount
        requestedPayloadBytes = total
    }

    fileprivate func matches(_ prepared: QwenPreparedPrefill) -> Bool {
        tokenCount == prepared.tokenIDs.count
            && positionCount == prepared.positions.count
            && featureOverrideCount == prepared.featureOverrides.count
    }
}

struct QwenLineageReservation: Hashable, Sendable {
    fileprivate let id: UUID
    let plannedRows: Int
    let requestedAllocationBytes: Int
    let provenancePlan: QwenLineageProvenancePlan
}

enum QwenConversationLineageError: Error, Equatable {
    case visibleRowsExceeded(requested: Int, maximum: Int)
    case liveRowsExceeded(requested: Int, maximum: Int)
    case requestedBytesExceeded(requested: Int, maximum: Int)
    case invalidReservation
}

struct QwenConversationStateDiagnosticSnapshot: Equatable, Sendable {
    let runnerState: QwenTextRunnerState
    let retainedTokenIDs: [Int32]
    let consumedTokenIDs: [Int32]
    let pendingAcceptedToken: Int32?
    let currentLogits: [Float16]?
    let textRoPEDelta: Int
    let replayGeneration: UInt64
    let logicalStateBytes: UInt64
    let producerEpoch: UInt64
    let lineage: QwenConversationLineageDiagnostics
}

/// Atomic input for one retained-conversation sample. Preparing it consumes
/// the preceding accepted token exactly once before exposing its successor
/// logits and the matching repetition-penalty history.
struct QwenConversationSamplingInput: Sendable {
    let logits: [Float16]
    let retainedTokenIDs: [Int32]
}

public actor QwenConversationState: ConversationStateTransaction {
    nonisolated let modelIdentity: String
    nonisolated let contextLimit: Int
    /// Immutable shared backing. Aggregate and replay copies retain this exact
    /// object; requested payload counts array elements, not allocator overhead.
    private final class MultimodalProvenance: Sendable {
        let id = UUID()
        let baseTokenOffset: Int
        let prepared: QwenPreparedPrefill
        let requestedPayloadBytes: Int

        init(baseTokenOffset: Int, prepared: QwenPreparedPrefill, plan: QwenLineageProvenancePlan) {
            self.baseTokenOffset = baseTokenOffset
            self.prepared = prepared
            requestedPayloadBytes = plan.requestedPayloadBytes
        }
    }

    private enum PromptReplayStep: Sendable {
        case token(Int32)
        case multimodal(MultimodalProvenance)
    }

    private struct Aggregate {
        var runner: QwenTextRunnerState
        var consumedTokenIDs: [Int32]
        var pendingAcceptedToken: Int32?
        var currentLogits: [Float16]
        var logitsValid: Bool
        var textRoPEDelta: Int
        var replayGeneration: UInt64
        var logicalStateBytes: UInt64
        var imageLineage: QwenImageLineage
        /// nil is owner-only accounting and is deliberately non-publishable.
        var imageProvenance: [MultimodalProvenance]?
    }

    private struct ActiveTransaction {
        let id: ConversationTransactionID
        let begin: Aggregate
        var working: Aggregate
        var replayOrigin: Aggregate
        var promptSteps: [PromptReplayStep]
        var generatedTokenIDs: [Int32]
    }

    private let runner: QwenTextRunner
    private let producer: QwenTextLogitProducer
    private nonisolated let logits: QwenConversationLogitsBuffer
    private let vocabularySize: Int
    private let maxContext: Int
    private var committed: Aggregate
    private var active: ActiveTransaction?
    private var nextIdentifier: UInt64 = 1
    private let visionLimits = QwenVisionResourceLimits.provisional
    private var lineageReservations: [UUID: (ConversationTransactionID, QwenLineageReservation)] = [:]
    private var lineageHighWaterRows = 0
    private var lineageHighWaterBytes = 0
    private var mutationLease: ActiveTransaction?
    private var mutating = false
    private var lineageFailed = false

    public init(
        model: QwenTextModel,
        context: MetalContext,
        maxContext: Int
    ) async throws {
        try await self.init(
            model: model,
            context: context,
            maxContext: maxContext,
            executionHooks: .none,
            publicationHooks: .none)
    }

    init(
        model: QwenTextModel,
        context: MetalContext,
        maxContext: Int,
        executionHooks: QwenTextExecutionHooks,
        publicationHooks: QwenTextPublicationHooks
    ) async throws {
        guard maxContext > 0 else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: maxContext, maximum: maxContext)
        }
        let runner = try model.makeRunner(
            context: context,
            executionHooks: executionHooks)
        let producer = QwenTextLogitProducer(
            runner: runner,
            vocabularySize: model.architecture.vocabularySize,
            hooks: publicationHooks)
        guard let logits = context.device.makeBuffer(
            length: model.architecture.vocabularySize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw QwenTextRunnerError.execution(detail: "conversation logits allocation failed")
        }
        let state = try await runner.snapshot()
        let emptyLogits = [Float16](
            repeating: 0, count: model.architecture.vocabularySize)
        memset(logits.contents(), 0, logits.length)
        let bytes = try Self.logicalBytes(
            runner: state,
            consumedTokenIDs: [],
            pendingAcceptedToken: nil,
            currentLogits: emptyLogits)
        self.runner = runner
        self.producer = producer
        self.logits = QwenConversationLogitsBuffer(value: logits)
        modelIdentity = model.mappingIdentity
        contextLimit = maxContext
        vocabularySize = model.architecture.vocabularySize
        self.maxContext = maxContext
        committed = Aggregate(
            runner: state,
            consumedTokenIDs: [],
            pendingAcceptedToken: nil,
            currentLogits: emptyLogits,
            logitsValid: false,
            textRoPEDelta: 0,
            replayGeneration: 0,
            logicalStateBytes: bytes,
            imageLineage: .empty,
            imageProvenance: [])
    }

    public func begin() async throws -> ConversationTransactionID {
        try requireHealthyAndIdle()
        guard active == nil else { throw ConversationStateTransactionError.busy }
        let id = ConversationTransactionID(rawValue: nextIdentifier)
        nextIdentifier &+= 1
        active = ActiveTransaction(
            id: id,
            begin: committed,
            working: committed,
            replayOrigin: committed,
            promptSteps: [],
            generatedTokenIDs: [])

        mutating = true
        defer { mutating = false }
        do {
            if committed.pendingAcceptedToken != nil {
                try await consumePending(transaction: id)
            }
            return id
        } catch {
            try await restoreCommittedAfterFailure(original: error)
            throw error
        }
    }

    /// Reserves all replacement rows and requested mutable bytes before the
    /// caller allocates prepared image storage. The old committed/begin
    /// lineage stays live until transaction commit or rollback.
    func reserveLineage(
        plannedRows: Int,
        requestedAllocationBytes: Int,
        provenancePlan: QwenLineageProvenancePlan = .none,
        transaction: ConversationTransactionID
    ) throws -> QwenLineageReservation {
        try requireMutable(transaction)
        guard plannedRows >= 0,
              plannedRows <= visionLimits.maximumVisibleHistoryRows else {
            throw QwenConversationLineageError.visibleRowsExceeded(
                requested: max(0, plannedRows),
                maximum: visionLimits.maximumVisibleHistoryRows)
        }
        guard requestedAllocationBytes >= 0,
              requestedAllocationBytes <= visionLimits.maximumMutablePreparedBytes else {
            throw QwenConversationLineageError.requestedBytesExceeded(
                requested: requestedAllocationBytes,
                maximum: visionLimits.maximumMutablePreparedBytes)
        }
        let base = liveLineageUnion()
        let reservedRows = try lineageReservations.values.reduce(0) { total, entry in
            let (next, overflow) = total.addingReportingOverflow(entry.1.plannedRows)
            guard !overflow else { throw QwenConversationLineageError.liveRowsExceeded(requested: Int.max, maximum: visionLimits.maximumLiveOwnerRows) }
            return next
        }
        let reservedBytes = try lineageReservations.values.reduce(0) { total, entry in
            let (entryBytes, entryOverflow) = entry.1.requestedAllocationBytes
                .addingReportingOverflow(entry.1.provenancePlan.requestedPayloadBytes)
            let (next, overflow) = total.addingReportingOverflow(entryBytes)
            guard !entryOverflow else { throw QwenConversationLineageError.requestedBytesExceeded(requested: Int.max, maximum: visionLimits.maximumMutablePreparedBytes) }
            guard !overflow else { throw QwenConversationLineageError.requestedBytesExceeded(requested: Int.max, maximum: visionLimits.maximumMutablePreparedBytes) }
            return next
        }
        let (plannedLiveRows, plannedOverflow) = reservedRows.addingReportingOverflow(plannedRows)
        let (liveRows, rowOverflow) = base.rowCount.addingReportingOverflow(plannedLiveRows)
        guard !plannedOverflow, !rowOverflow, liveRows <= visionLimits.maximumLiveOwnerRows else {
            throw QwenConversationLineageError.liveRowsExceeded(
                requested: plannedOverflow || rowOverflow ? Int.max : liveRows,
                maximum: visionLimits.maximumLiveOwnerRows)
        }
        let (requestBytes, requestOverflow) = requestedAllocationBytes
            .addingReportingOverflow(provenancePlan.requestedPayloadBytes)
        let (plannedLiveBytes, plannedByteOverflow) = reservedBytes.addingReportingOverflow(requestBytes)
        let (baseBytes, baseOverflow) = base.ownedRequestedBytes
            .addingReportingOverflow(liveProvenanceRequestedBytes())
        let (liveBytes, byteOverflow) = baseBytes.addingReportingOverflow(plannedLiveBytes)
        let anyByteOverflow = requestOverflow || plannedByteOverflow || baseOverflow || byteOverflow
        guard !anyByteOverflow,
              liveBytes <= visionLimits.maximumMutablePreparedBytes else {
            throw QwenConversationLineageError.requestedBytesExceeded(
                requested: anyByteOverflow ? Int.max : liveBytes,
                maximum: visionLimits.maximumMutablePreparedBytes)
        }
        let reservation = QwenLineageReservation(
            id: UUID(), plannedRows: plannedRows,
            requestedAllocationBytes: requestedAllocationBytes,
            provenancePlan: provenancePlan)
        lineageReservations[reservation.id] = (transaction, reservation)
        lineageHighWaterRows = max(lineageHighWaterRows, liveRows)
        lineageHighWaterBytes = max(lineageHighWaterBytes, liveBytes)
        return reservation
    }

    func cancelLineage(
        _ reservation: QwenLineageReservation,
        transaction: ConversationTransactionID
    ) throws {
        try requireMutable(transaction)
        guard let stored = lineageReservations[reservation.id],
              stored.0 == transaction, stored.1 == reservation else {
            throw QwenConversationLineageError.invalidReservation
        }
        lineageReservations.removeValue(forKey: reservation.id)
    }

    /// Internal owner-allocation accounting only. It binds no token ranges or
    /// positions and is therefore non-publishable; callers must roll back or
    /// replace it with a complete multimodal rebuild.
    func consumeLineage(
        _ reservation: QwenLineageReservation,
        replacement: QwenImageLineage,
        transaction: ConversationTransactionID
    ) throws {
        try requireMutable(transaction)
        guard let stored = lineageReservations[reservation.id],
              stored.0 == transaction, stored.1 == reservation,
              reservation.provenancePlan == .none,
              replacement.rowCount == reservation.plannedRows,
              replacement.ownedRequestedBytes <= reservation.requestedAllocationBytes,
              let current = active,
              current.working.consumedTokenIDs.isEmpty,
              current.working.pendingAcceptedToken == nil,
              current.promptSteps.isEmpty,
              current.generatedTokenIDs.isEmpty,
              current.working.imageProvenance?.isEmpty != false else {
            throw QwenConversationLineageError.invalidReservation
        }
        let beginLineage: QwenImageLineage = active?.begin.imageLineage ?? .empty
        let workingLineage: QwenImageLineage = active?.working.imageLineage ?? .empty
        let base = QwenImageLineage.union([
            committed.imageLineage, beginLineage, workingLineage, replacement,
        ])
        guard base.rowCount <= visionLimits.maximumLiveOwnerRows else {
            throw QwenConversationLineageError.liveRowsExceeded(
                requested: base.rowCount,
                maximum: visionLimits.maximumLiveOwnerRows)
        }
        guard base.ownedRequestedBytes <= visionLimits.maximumMutablePreparedBytes else {
            throw QwenConversationLineageError.requestedBytesExceeded(
                requested: base.ownedRequestedBytes,
                maximum: visionLimits.maximumMutablePreparedBytes)
        }
        lineageHighWaterRows = max(lineageHighWaterRows, base.rowCount)
        lineageHighWaterBytes = max(lineageHighWaterBytes, base.ownedRequestedBytes)
        active!.working.imageLineage = replacement
        active!.working.imageProvenance = nil
        active!.working.textRoPEDelta = replacement.textRoPEDelta
        active!.working.logicalStateBytes = try Self.logicalBytes(
            runner: active!.working.runner,
            consumedTokenIDs: active!.working.consumedTokenIDs,
            pendingAcceptedToken: active!.working.pendingAcceptedToken,
            currentLogits: active!.working.currentLogits,
            imageLineage: replacement)
        lineageReservations.removeValue(forKey: reservation.id)
    }

    /// Appends one prepared image-bearing prompt step. Prepared positions and
    /// delta are already absolute for the complete history and are retained
    /// unchanged for suffix replay.
    func prefillMultimodal(
        _ prepared: QwenPreparedPrefill,
        lineage: QwenImageLineage,
        reservation: QwenLineageReservation,
        transaction: ConversationTransactionID
    ) async throws {
        try requireMutable(transaction)
        try requireBoundWorking()
        guard let stored = lineageReservations[reservation.id],
              stored.0 == transaction, stored.1 == reservation,
              reservation.provenancePlan.matches(prepared),
              lineage.rowCount == reservation.plannedRows,
              lineage.ownedRequestedBytes <= reservation.requestedAllocationBytes,
              lineage.textRoPEDelta == prepared.textRoPEDelta else {
            throw QwenConversationLineageError.invalidReservation
        }
        let overrideIDs = Set(prepared.featureOverrides.map { $0.owner.allocationID })
        guard overrideIDs == lineage.allocationIDs,
              Self.hasCommonProcessingMetadata(lineage.owners),
              Self.hasCompatibleProcessingMetadata(
                lineage.owners, with: active!.working.imageLineage.owners) else {
            throw QwenConversationLineageError.invalidReservation
        }
        let requested = try retainedCount(transaction) + prepared.tokenIDs.count
        guard requested <= maxContext else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: requested, maximum: maxContext)
        }
        let provenance = MultimodalProvenance(
            baseTokenOffset: active!.working.consumedTokenIDs.count,
            prepared: prepared, plan: reservation.provenancePlan)
        try validateProspectiveAppend(provenance, incomingLineage: lineage)
        let previous = active!
        mutating = true
        defer { mutating = false }
        do {
            try await applyMultimodal(provenance, transaction: transaction)
            active!.promptSteps.append(.multimodal(provenance))
            lineageReservations.removeValue(forKey: reservation.id)
        } catch {
            try await restoreTransaction(previous, operation: "multimodal prefill", original: error)
            throw error
        }
    }

    public func prefill(
        _ tokenIDs: [Int32],
        transaction: ConversationTransactionID,
        onProgress: @Sendable (Int, Int) async -> Void
    ) async throws {
        try requireMutable(transaction)
        try requireBoundWorking()
        let requested = try retainedCount(transaction) + tokenIDs.count
        guard requested <= maxContext else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: requested, maximum: maxContext)
        }
        mutating = true
        defer { mutating = false }
        for (index, tokenID) in tokenIDs.enumerated() {
            try await consume(tokenID, transaction: transaction)
            active!.promptSteps.append(.token(tokenID))
            await onProgress(index + 1, tokenIDs.count)
        }
    }

    public func advance(
        _ tokenID: Int32,
        transaction: ConversationTransactionID
    ) async throws {
        try requireMutable(transaction)
        try requireBoundWorking()
        try validate(tokenID)
        let requested = try retainedCount(transaction) + 1
        guard requested <= maxContext else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: requested, maximum: maxContext)
        }
        mutating = true
        defer { mutating = false }
        try await acceptGenerated(tokenID, transaction: transaction)
    }

    public func removeSuffix(
        tokenCount: Int,
        transaction: ConversationTransactionID
    ) async throws {
        try requireMutable(transaction)
        try requireBoundWorking()
        guard tokenCount > 0,
              let current = active,
              tokenCount <= current.generatedTokenIDs.count else {
            throw ConversationStateTransactionError.invalidBoundary(
                "hidden suffix crosses the generated-token boundary")
        }
        let retainedGenerated = Array(current.generatedTokenIDs.dropLast(tokenCount))
        mutationLease = current
        mutating = true
        defer {
            mutationLease = nil
            mutating = false
        }
        do {
            try await restore(current.replayOrigin)
            var replaying = current
            replaying.working = current.replayOrigin
            replaying.generatedTokenIDs = []
            active = replaying
            if current.replayOrigin.pendingAcceptedToken != nil {
                try await consumePending(transaction: transaction)
            }
            for step in current.promptSteps {
                switch step {
                case .token(let tokenID):
                    try await consume(tokenID, transaction: transaction)
                case .multimodal(let provenance):
                    try await applyMultimodal(provenance, transaction: transaction)
                }
            }
            active!.generatedTokenIDs = []
            for tokenID in retainedGenerated {
                try await acceptGenerated(tokenID, transaction: transaction)
            }
        } catch {
            try await restoreTransaction(current, operation: "suffix replay", original: error)
            throw error
        }
    }

    public func commit(
        transaction: ConversationTransactionID
    ) async throws -> ConversationStateMetrics {
        try requireMutable(transaction)
        guard let current = active else {
            throw ConversationStateTransactionError.staleTransaction
        }
        try Self.validate(current.working)
        guard current.working.currentLogits.count == vocabularySize else {
            throw ConversationStateTransactionError.invalidBoundary(
                "published logits count does not match vocabulary")
        }
        let metrics = Self.metrics(current.working)
        // Irreversible publication point. There is deliberately no await,
        // cancellation check, or throwing operation after this assignment.
        committed = current.working
        lineageReservations = lineageReservations.filter { $0.value.0 != transaction }
        active = nil
        return metrics
    }

    public func rollback(transaction: ConversationTransactionID) async throws {
        try requireMutable(transaction)
        guard let current = active else {
            throw ConversationStateTransactionError.staleTransaction
        }
        mutating = true
        defer { mutating = false }
        do {
            try await restore(current.begin)
            lineageReservations = lineageReservations.filter { $0.value.0 != transaction }
            active = nil
        } catch {
            lineageFailed = true
            active = nil
            throw ConversationStateTransactionError.restoreFailed("\(error)")
        }
    }

    public func rebuildCheckpoint(
        retaining tokenIDs: [Int32],
        transaction: ConversationTransactionID,
        onProgress: @Sendable (Int, Int) async -> Void
    ) async throws {
        try requireMutable(transaction)
        guard !tokenIDs.isEmpty, tokenIDs.count <= maxContext else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: tokenIDs.count, maximum: maxContext)
        }
        for tokenID in tokenIDs { try validate(tokenID) }
        guard let previous = active else {
            throw ConversationStateTransactionError.staleTransaction
        }
        try requireBoundWorking()
        guard previous.working.imageLineage.rowCount == 0 else {
            throw ConversationStateTransactionError.invalidBoundary(
                "multimodal checkpoints require prepared features and lineage")
        }
        mutationLease = previous
        mutating = true
        defer {
            mutationLease = nil
            mutating = false
        }
        do {
            try await runner.reset()
            let zero = try await makeZeroAggregate(
                replayGeneration: previous.working.replayGeneration &+ 1)
            var rebuilding = previous
            rebuilding.working = zero
            rebuilding.replayOrigin = zero
            rebuilding.promptSteps = []
            rebuilding.generatedTokenIDs = []
            active = rebuilding
            for (index, tokenID) in tokenIDs.enumerated() {
                try await consume(tokenID, transaction: transaction)
                active!.promptSteps.append(.token(tokenID))
                await onProgress(index + 1, tokenIDs.count)
            }
            active!.working.pendingAcceptedToken = nil
            try refreshLogicalBytes()
        } catch {
            try await restoreTransaction(previous, operation: "checkpoint", original: error)
            throw error
        }
    }

    func rebuildMultimodalCheckpoint(
        prepared: QwenPreparedPrefill,
        lineage: QwenImageLineage,
        reservation: QwenLineageReservation,
        transaction: ConversationTransactionID
    ) async throws {
        try requireMutable(transaction)
        guard !prepared.tokenIDs.isEmpty,
              prepared.tokenIDs.count <= maxContext,
              let stored = lineageReservations[reservation.id],
              stored.0 == transaction, stored.1 == reservation,
              reservation.provenancePlan.matches(prepared),
              lineage.rowCount == reservation.plannedRows,
              lineage.ownedRequestedBytes <= reservation.requestedAllocationBytes,
              lineage.textRoPEDelta == prepared.textRoPEDelta,
              Set(prepared.featureOverrides.map { $0.owner.allocationID })
                == lineage.allocationIDs,
              Self.hasCommonProcessingMetadata(lineage.owners),
              Self.hasCompatibleProcessingMetadata(
                lineage.owners, with: active!.begin.imageLineage.owners) else {
            throw QwenConversationLineageError.invalidReservation
        }
        let previous = active!
        let provenance = MultimodalProvenance(
            baseTokenOffset: 0, prepared: prepared, plan: reservation.provenancePlan)
        try validateBoundCandidate(lineage: lineage, provenance: [provenance])
        mutationLease = previous
        mutating = true
        defer {
            mutationLease = nil
            mutating = false
        }
        do {
            try await runner.reset()
            let zero = try await makeZeroAggregate(
                replayGeneration: previous.working.replayGeneration &+ 1)
            try await producer.prefill(prepared: prepared, into: logits.value)
            var rebuilt = zero
            rebuilt.consumedTokenIDs = prepared.tokenIDs
            rebuilt.runner = try await runner.snapshot()
            rebuilt.currentLogits = snapshotLogits()
            rebuilt.logitsValid = true
            rebuilt.textRoPEDelta = prepared.textRoPEDelta
            rebuilt.imageLineage = lineage
            rebuilt.imageProvenance = [provenance]
            rebuilt.logicalStateBytes = try Self.logicalBytes(
                runner: rebuilt.runner,
                consumedTokenIDs: rebuilt.consumedTokenIDs,
                pendingAcceptedToken: nil,
                currentLogits: rebuilt.currentLogits,
                imageLineage: lineage,
                provenanceRequestedBytes: provenance.requestedPayloadBytes)
            var current = previous
            current.working = rebuilt
            current.replayOrigin = zero
            current.promptSteps = [.multimodal(provenance)]
            current.generatedTokenIDs = []
            active = current
            lineageReservations.removeValue(forKey: reservation.id)
        } catch {
            try await restoreTransaction(previous, operation: "multimodal rebuild", original: error)
            throw error
        }
    }

    public func reset() async throws {
        guard !mutating, active == nil else {
            throw ConversationStateTransactionError.busy
        }
        mutating = true
        defer { mutating = false }
        producer.reset()
        try await runner.reset()
        committed = try await makeZeroAggregate(
            replayGeneration: committed.replayGeneration &+ 1)
        lineageReservations.removeAll(keepingCapacity: true)
        mutationLease = nil
        lineageHighWaterRows = 0
        lineageHighWaterBytes = 0
        lineageFailed = false
    }

    public func status() -> ConversationStateStatus {
        ConversationStateStatus(
            committed: Self.metrics(committed),
            working: active.map { Self.metrics($0.working) },
            activeTransaction: active?.id)
    }

    /// Exact live allocation owned by the runner's per-layer expert caches.
    /// The runner remains the sole accounting source.
    public var expertCacheAllocatedBytes: UInt64 {
        get async { await runner.expertCacheAllocatedBytes }
    }

    /// Advances an accepted pending token, when present, then snapshots the
    /// logits and history at the same transaction boundary. Generation must
    /// call this before sampling and `advance` only after the sampled token is
    /// accepted by the caller.
    func prepareSampling(
        transaction: ConversationTransactionID
    ) async throws -> QwenConversationSamplingInput {
        try requireMutable(transaction)
        try requireBoundWorking()
        if active?.working.pendingAcceptedToken != nil {
            try await consumePending(transaction: transaction)
        }
        guard let working = active?.working, working.logitsValid else {
            throw ConversationStateTransactionError.invalidBoundary(
                "generation requires a nonempty retained prompt")
        }
        return QwenConversationSamplingInput(
            logits: snapshotLogits(),
            retainedTokenIDs: working.consumedTokenIDs)
    }

    /// Returns settled real runner and producer state, not the stored rollback
    /// shadow. Callers must not invoke this while an operation is in GPU work.
    func diagnosticSnapshot() async throws -> QwenConversationStateDiagnosticSnapshot {
        let aggregate = active?.working ?? committed
        var retained = aggregate.consumedTokenIDs
        if let pending = aggregate.pendingAcceptedToken { retained.append(pending) }
        return QwenConversationStateDiagnosticSnapshot(
            runnerState: try await producer.snapshot(),
            retainedTokenIDs: retained,
            consumedTokenIDs: aggregate.consumedTokenIDs,
            pendingAcceptedToken: aggregate.pendingAcceptedToken,
            currentLogits: aggregate.logitsValid ? snapshotLogits() : nil,
            textRoPEDelta: aggregate.textRoPEDelta,
            replayGeneration: aggregate.replayGeneration,
            logicalStateBytes: aggregate.logicalStateBytes,
            producerEpoch: producer.generationEpoch,
            lineage: lineageDiagnostics())
    }

    private func acceptGenerated(
        _ tokenID: Int32,
        transaction: ConversationTransactionID
    ) async throws {
        if active?.working.pendingAcceptedToken != nil {
            try await consumePending(transaction: transaction)
        }
        try require(transaction)
        active!.working.pendingAcceptedToken = tokenID
        active!.generatedTokenIDs.append(tokenID)
        active!.working.logicalStateBytes = try Self.logicalBytes(
            runner: active!.working.runner,
            consumedTokenIDs: active!.working.consumedTokenIDs,
            pendingAcceptedToken: tokenID,
            currentLogits: active!.working.currentLogits,
            imageLineage: active!.working.imageLineage,
            provenanceRequestedBytes: Self.provenanceRequestedBytes(active!.working))
    }

    private func consumePending(transaction: ConversationTransactionID) async throws {
        try require(transaction)
        guard let tokenID = active?.working.pendingAcceptedToken else { return }
        try await consume(tokenID, transaction: transaction)
        active!.working.pendingAcceptedToken = nil
        active!.working.logicalStateBytes = try Self.logicalBytes(
            runner: active!.working.runner,
            consumedTokenIDs: active!.working.consumedTokenIDs,
            pendingAcceptedToken: nil,
            currentLogits: active!.working.currentLogits,
            imageLineage: active!.working.imageLineage,
            provenanceRequestedBytes: Self.provenanceRequestedBytes(active!.working))
    }

    private func consume(
        _ tokenID: Int32,
        transaction: ConversationTransactionID
    ) async throws {
        try require(transaction)
        try validate(tokenID)
        guard var current = active else {
            throw ConversationStateTransactionError.staleTransaction
        }
        let cachePosition = current.working.consumedTokenIDs.count
        let (ropePosition, overflow) = cachePosition.addingReportingOverflow(
            current.working.textRoPEDelta)
        guard !overflow, ropePosition >= 0,
              cachePosition == current.working.runner.sequenceLength else {
            throw ConversationStateTransactionError.invalidBoundary(
                "journal/cache or M-RoPE coordinate is invalid")
        }
        try await producer.produce(
            token: tokenID,
            cachePosition: cachePosition,
            ropePosition: try QwenMRoPEPosition(
                temporal: ropePosition, height: ropePosition, width: ropePosition),
            into: logits.value)
        current = active!
        current.working.consumedTokenIDs.append(tokenID)
        current.working.runner = try await runner.snapshot()
        current.working.currentLogits = snapshotLogits()
        current.working.logitsValid = true
        current.working.logicalStateBytes = try Self.logicalBytes(
            runner: current.working.runner,
            consumedTokenIDs: current.working.consumedTokenIDs,
            pendingAcceptedToken: current.working.pendingAcceptedToken,
            currentLogits: current.working.currentLogits,
            imageLineage: current.working.imageLineage,
            provenanceRequestedBytes: Self.provenanceRequestedBytes(current.working))
        active = current
    }

    func lineageDiagnostics() -> QwenConversationLineageDiagnostics {
        let aggregate = active?.working ?? committed
        let live = liveLineageUnion()
        let reservedRows = lineageReservations.values.reduce(0) {
            Self.saturatingAdd($0, $1.1.plannedRows)
        }
        let reservedBytes = lineageReservations.values.reduce(0) {
            let requested = Self.saturatingAdd(
                $1.1.requestedAllocationBytes,
                $1.1.provenancePlan.requestedPayloadBytes)
            return Self.saturatingAdd($0, requested)
        }
        let provenanceBytes = Self.provenanceRequestedBytes(aggregate)
        let visibleRows = aggregate.imageProvenance.map(Self.visibleRows) ?? 0
        return QwenConversationLineageDiagnostics(
            visibleRows: visibleRows,
            logicalBytes: aggregate.imageLineage.logicalBytes + provenanceBytes,
            allocationIDs: aggregate.imageLineage.allocationIDs,
            ownedRequestedBytes: aggregate.imageLineage.ownedRequestedBytes,
            provenanceRequestedBytes: provenanceBytes,
            occurrences: aggregate.imageProvenance.map(Self.occurrences) ?? [],
            liveOwnerRows: Self.saturatingAdd(live.rowCount, reservedRows),
            liveRequestedBytes: Self.saturatingAdd(
                Self.saturatingAdd(live.ownedRequestedBytes, liveProvenanceRequestedBytes()),
                reservedBytes),
            highWaterOwnerRows: lineageHighWaterRows,
            highWaterRequestedBytes: lineageHighWaterBytes)
    }

    private func liveLineageUnion() -> QwenImageLineage {
        QwenImageLineage.union(liveAggregates().map(\.imageLineage))
    }

    private func liveAggregates() -> [Aggregate] {
        var values = [committed]
        if let active { values += [active.begin, active.working, active.replayOrigin] }
        if let mutationLease {
            values += [mutationLease.begin, mutationLease.working, mutationLease.replayOrigin]
        }
        return values
    }

    private func liveProvenanceRequestedBytes() -> Int {
        var seen: Set<ObjectIdentifier> = []
        var total = 0
        func add(_ provenance: MultimodalProvenance) {
            guard seen.insert(ObjectIdentifier(provenance)).inserted else { return }
            total = Self.saturatingAdd(total, provenance.requestedPayloadBytes)
        }
        for aggregate in liveAggregates() { aggregate.imageProvenance?.forEach(add) }
        active?.promptSteps.forEach {
            if case .multimodal(let provenance) = $0 { add(provenance) }
        }
        mutationLease?.promptSteps.forEach {
            if case .multimodal(let provenance) = $0 { add(provenance) }
        }
        return total
    }

    private func applyMultimodal(
        _ provenance: MultimodalProvenance,
        transaction: ConversationTransactionID
    ) async throws {
        try require(transaction)
        guard let existing = active?.working.imageProvenance,
              provenance.baseTokenOffset == active?.working.consumedTokenIDs.count else {
            throw ConversationStateTransactionError.invalidBoundary(
                "multimodal replay origin does not match the journal")
        }
        let incoming = QwenImageLineage(
            owners: provenance.prepared.featureOverrides.map(\.owner),
            textRoPEDelta: provenance.prepared.textRoPEDelta)
        let combined = QwenImageLineage(
            owners: active!.working.imageLineage.owners + incoming.owners,
            textRoPEDelta: provenance.prepared.textRoPEDelta)
        let combinedProvenance = existing + [provenance]
        try validateBoundCandidate(lineage: combined, provenance: combinedProvenance)
        try await producer.prefill(prepared: provenance.prepared, into: logits.value)
        var current = active!
        current.working.consumedTokenIDs.append(contentsOf: provenance.prepared.tokenIDs)
        current.working.runner = try await runner.snapshot()
        current.working.currentLogits = snapshotLogits()
        current.working.logitsValid = true
        current.working.imageLineage = combined
        current.working.imageProvenance = combinedProvenance
        current.working.textRoPEDelta = provenance.prepared.textRoPEDelta
        current.working.logicalStateBytes = try Self.logicalBytes(
            runner: current.working.runner,
            consumedTokenIDs: current.working.consumedTokenIDs,
            pendingAcceptedToken: current.working.pendingAcceptedToken,
            currentLogits: current.working.currentLogits,
            imageLineage: combined,
            provenanceRequestedBytes: Self.provenanceRequestedBytes(current.working))
        active = current
    }

    private func validateProspectiveAppend(
        _ provenance: MultimodalProvenance,
        incomingLineage: QwenImageLineage
    ) throws {
        guard let existing = active?.working.imageProvenance else {
            throw ConversationStateTransactionError.invalidBoundary(
                "owner-only accounting cannot be appended")
        }
        let combined = QwenImageLineage(
            owners: active!.working.imageLineage.owners + incomingLineage.owners,
            textRoPEDelta: provenance.prepared.textRoPEDelta)
        try validateBoundCandidate(lineage: combined, provenance: existing + [provenance])
    }

    private func validateBoundCandidate(
        lineage: QwenImageLineage,
        provenance: [MultimodalProvenance]
    ) throws {
        let rows = Self.visibleRows(provenance)
        guard rows <= visionLimits.maximumVisibleHistoryRows else {
            throw QwenConversationLineageError.visibleRowsExceeded(
                requested: rows, maximum: visionLimits.maximumVisibleHistoryRows)
        }
        let total = Self.saturatingAdd(
            lineage.ownedRequestedBytes, Self.provenanceRequestedBytes(provenance))
        guard total <= visionLimits.maximumMutablePreparedBytes else {
            throw QwenConversationLineageError.requestedBytesExceeded(
                requested: total, maximum: visionLimits.maximumMutablePreparedBytes)
        }
    }

    private func restoreTransaction(
        _ transaction: ActiveTransaction,
        operation: String,
        original: Error
    ) async throws {
        do {
            try await restore(transaction.working)
            active = transaction
        } catch let restoreError {
            lineageFailed = true
            active = nil
            throw ConversationStateTransactionError.restoreFailed(
                "\(operation) error \(original); restore error \(restoreError)")
        }
    }

    private func requireBoundWorking() throws {
        guard active?.working.imageProvenance != nil else {
            throw ConversationStateTransactionError.invalidBoundary(
                "owner-only image accounting is not publishable")
        }
    }

    private func refreshLogicalBytes() throws {
        guard var current = active else {
            throw ConversationStateTransactionError.staleTransaction
        }
        current.working.logicalStateBytes = try Self.logicalBytes(
            runner: current.working.runner,
            consumedTokenIDs: current.working.consumedTokenIDs,
            pendingAcceptedToken: current.working.pendingAcceptedToken,
            currentLogits: current.working.currentLogits,
            imageLineage: current.working.imageLineage,
            provenanceRequestedBytes: Self.provenanceRequestedBytes(current.working))
        active = current
    }

    private func retainedCount(_ transaction: ConversationTransactionID) throws -> Int {
        try require(transaction)
        guard let working = active?.working else {
            throw ConversationStateTransactionError.staleTransaction
        }
        return working.consumedTokenIDs.count + (working.pendingAcceptedToken == nil ? 0 : 1)
    }

    private func requireHealthyAndIdle() throws {
        guard !lineageFailed else {
            throw ConversationStateTransactionError.restoreFailed("conversation lineage is closed")
        }
        guard !mutating else { throw ConversationStateTransactionError.busy }
    }

    private func requireMutable(_ transaction: ConversationTransactionID) throws {
        try requireHealthyAndIdle()
        try require(transaction)
    }

    private func require(_ transaction: ConversationTransactionID) throws {
        guard active?.id == transaction else {
            throw ConversationStateTransactionError.staleTransaction
        }
    }

    private func validate(_ tokenID: Int32) throws {
        guard tokenID >= 0, Int(tokenID) < vocabularySize else {
            throw QwenTextRunnerError.invalidToken(id: tokenID)
        }
    }

    private func restore(_ aggregate: Aggregate) async throws {
        await runner.waitUntilIdle()
        try await runner.restore(aggregate.runner)
        try restoreLogits(aggregate.currentLogits)
    }

    private func snapshotLogits() -> [Float16] {
        let pointer = logits.value.contents().bindMemory(
            to: Float16.self, capacity: vocabularySize)
        return Array(UnsafeBufferPointer(start: pointer, count: vocabularySize))
    }

    private func restoreLogits(_ values: [Float16]) throws {
        guard values.count == vocabularySize else {
            throw ConversationStateTransactionError.invalidBoundary(
                "restored logits count does not match vocabulary")
        }
        _ = values.withUnsafeBytes { source in
            memcpy(logits.value.contents(), source.baseAddress!, source.count)
        }
    }

    private func restoreCommittedAfterFailure(original: Error) async throws {
        do {
            try await restore(committed)
            lineageReservations.removeAll(keepingCapacity: true)
            active = nil
        } catch {
            lineageFailed = true
            active = nil
            throw ConversationStateTransactionError.restoreFailed(
                "operation \(original); restore \(error)")
        }
    }

    private static func hasCommonProcessingMetadata(
        _ owners: [QwenRetainedFeatureOwner]
    ) -> Bool {
        guard let first = owners.first else { return true }
        return owners.dropFirst().allSatisfy {
            $0.processorDigest == first.processorDigest && $0.profile == first.profile
        }
    }

    private static func hasCompatibleProcessingMetadata(
        _ candidate: [QwenRetainedFeatureOwner],
        with existing: [QwenRetainedFeatureOwner]
    ) -> Bool {
        guard let candidateFirst = candidate.first,
              let existingFirst = existing.first else { return true }
        return candidateFirst.processorDigest == existingFirst.processorDigest
            && candidateFirst.profile == existingFirst.profile
    }

    private func makeZeroAggregate(replayGeneration: UInt64) async throws -> Aggregate {
        let state = try await runner.snapshot()
        let emptyLogits = [Float16](repeating: 0, count: vocabularySize)
        try restoreLogits(emptyLogits)
        let bytes = try Self.logicalBytes(
            runner: state,
            consumedTokenIDs: [],
            pendingAcceptedToken: nil,
            currentLogits: emptyLogits)
        return Aggregate(
            runner: state,
            consumedTokenIDs: [],
            pendingAcceptedToken: nil,
            currentLogits: emptyLogits,
            logitsValid: false,
            textRoPEDelta: 0,
            replayGeneration: replayGeneration,
            logicalStateBytes: bytes,
            imageLineage: .empty,
            imageProvenance: [])
    }

    private static func validate(_ aggregate: Aggregate) throws {
        let (position, overflow) = aggregate.consumedTokenIDs.count.addingReportingOverflow(
            aggregate.textRoPEDelta)
        guard !overflow, position >= 0,
              aggregate.consumedTokenIDs.count == aggregate.runner.sequenceLength,
              let provenance = aggregate.imageProvenance else {
            throw ConversationStateTransactionError.invalidBoundary(
                "published journal, RoPE delta, and bound image provenance disagree")
        }
        let occurrences = occurrences(provenance)
        let ownerIDs = Set(occurrences.map(\.ownerAllocationID))
        let payload = provenanceRequestedBytes(provenance)
        let total = saturatingAdd(aggregate.imageLineage.ownedRequestedBytes, payload)
        guard visibleRows(provenance) <= QwenVisionResourceLimits.provisional.maximumVisibleHistoryRows,
              ownerIDs == aggregate.imageLineage.allocationIDs,
              total <= QwenVisionResourceLimits.provisional.maximumMutablePreparedBytes,
              aggregate.imageLineage.textRoPEDelta == aggregate.textRoPEDelta,
              provenance.allSatisfy({ segment in
                  let (end, endOverflow) = segment.baseTokenOffset.addingReportingOverflow(
                      segment.prepared.tokenIDs.count)
                  guard !endOverflow, segment.baseTokenOffset >= 0,
                        end <= aggregate.consumedTokenIDs.count else { return false }
                  return Array(aggregate.consumedTokenIDs[segment.baseTokenOffset..<end])
                      == segment.prepared.tokenIDs
              }) else {
            throw ConversationStateTransactionError.invalidBoundary(
                "published image occurrences, owners, or requested payload disagree")
        }
    }

    private static func occurrences(
        _ provenance: [MultimodalProvenance]
    ) -> [QwenImageOccurrenceDiagnostics] {
        provenance.flatMap { segment in
            segment.prepared.featureOverrides.map { override in
                QwenImageOccurrenceDiagnostics(
                    segmentID: segment.id,
                    ownerAllocationID: override.owner.allocationID,
                    imageDigest: override.owner.imageDigest,
                    processorDigest: override.owner.processorDigest,
                    absoluteTokenRange: Range(
                        uncheckedBounds: (
                            lower: segment.baseTokenOffset + override.tokenRange.lowerBound,
                            upper: segment.baseTokenOffset + override.tokenRange.upperBound)),
                    exactPositions: Array(segment.prepared.positions[override.tokenRange]))
            }
        }
    }

    private static func visibleRows(_ provenance: [MultimodalProvenance]) -> Int {
        provenance.reduce(0) { total, segment in
            segment.prepared.featureOverrides.reduce(total) {
                saturatingAdd($0, $1.tokenRange.count)
            }
        }
    }

    private static func provenanceRequestedBytes(_ aggregate: Aggregate) -> Int {
        aggregate.imageProvenance.map(provenanceRequestedBytes) ?? 0
    }

    private static func provenanceRequestedBytes(
        _ provenance: [MultimodalProvenance]
    ) -> Int {
        var seen: Set<ObjectIdentifier> = []
        return provenance.reduce(0) { total, segment in
            guard seen.insert(ObjectIdentifier(segment)).inserted else { return total }
            return saturatingAdd(total, segment.requestedPayloadBytes)
        }
    }

    private static func saturatingAdd(_ lhs: Int, _ rhs: Int) -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? Int.max : sum
    }

    private static func metrics(_ aggregate: Aggregate) -> ConversationStateMetrics {
        var retained = aggregate.consumedTokenIDs
        if let pending = aggregate.pendingAcceptedToken { retained.append(pending) }
        return ConversationStateMetrics(
            retainedTokenIDs: retained,
            consumedTokenCount: aggregate.consumedTokenIDs.count,
            pendingTokenCount: aggregate.pendingAcceptedToken == nil ? 0 : 1,
            logicalStateBytes: aggregate.logicalStateBytes)
    }

    private static func logicalBytes(
        runner: QwenTextRunnerState,
        consumedTokenIDs: [Int32],
        pendingAcceptedToken: Int32?,
        currentLogits: [Float16],
        imageLineage: QwenImageLineage = .empty,
        provenanceRequestedBytes: Int = 0
    ) throws -> UInt64 {
        guard provenanceRequestedBytes >= 0 else {
            throw ConversationStateTransactionError.invalidBoundary(
                "negative provenance byte count")
        }
        let (initial, initialOverflow) = UInt64(imageLineage.logicalBytes)
            .addingReportingOverflow(UInt64(provenanceRequestedBytes))
        guard !initialOverflow else {
            throw ConversationStateTransactionError.invalidBoundary(
                "logical state byte count overflow")
        }
        var bytes = initial
        func add(_ count: Int, stride: Int) throws {
            let (product, overflow) = UInt64(count).multipliedReportingOverflow(by: UInt64(stride))
            let (sum, sumOverflow) = bytes.addingReportingOverflow(product)
            guard !overflow, !sumOverflow else {
                throw ConversationStateTransactionError.invalidBoundary(
                    "logical state byte count overflow")
            }
            bytes = sum
        }
        try add(runner.architectureIdentity.utf8.count, stride: 1)
        try add(consumedTokenIDs.count, stride: MemoryLayout<Int32>.stride)
        if pendingAcceptedToken != nil {
            try add(1, stride: MemoryLayout<Int32>.stride)
        }
        try add(currentLogits.count, stride: MemoryLayout<Float16>.stride)
        for layer in runner.layers {
            switch layer {
            case .linear(let convolutionHistory, let recurrentMatrix):
                try add(convolutionHistory.count, stride: MemoryLayout<Float>.stride)
                try add(recurrentMatrix.count, stride: MemoryLayout<Float>.stride)
            case .full(let key, let value):
                try add(key.count, stride: MemoryLayout<Float>.stride)
                try add(value.count, stride: MemoryLayout<Float>.stride)
            }
        }
        // sequence length, text RoPE delta, max capacity and replay generation
        try add(3, stride: MemoryLayout<Int>.stride)
        try add(1, stride: MemoryLayout<UInt64>.stride)
        return bytes
    }
}
