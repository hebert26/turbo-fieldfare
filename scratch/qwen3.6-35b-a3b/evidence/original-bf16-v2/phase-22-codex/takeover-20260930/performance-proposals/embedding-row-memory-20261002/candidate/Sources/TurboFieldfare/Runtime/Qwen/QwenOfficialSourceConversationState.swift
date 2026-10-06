import Foundation
import Metal

/// Default-inert fault gates on the actual protected read and GPU completion
/// path. Synchronous read gates run inside bounded paired-read workers; async
/// gates never supply bytes, offsets, buffers, or an admission identity.
struct QwenOfficialSourceTransactionHooks: Sendable {
    let isKnownNone: Bool
    let beforeProtectedExpertRead:
        @Sendable (Int, Int, QwenBF16ExpertReadHooks.Stream) throws -> Void
    let afterActualGPUSubmission: @Sendable (String) async throws -> Void
    let afterActualGPUCompletion: @Sendable (String) async throws -> Void
    /// Explicit callbacks, including no-op callbacks, retain stage boundaries.
    let requiresSeparateGPUStages: Bool
    /// Explicit per-token observations retain their original publication timing.
    let requiresTokenMajorPrefill: Bool
    /// These boundaries cannot be replaced by an explicit grouped capture.
    let requiresOrderedPrefillStages: Bool
    let betweenConsumedTokens: @Sendable (Int) async throws -> Void
    let beforeTurnCommit: @Sendable () async throws -> Void
    let beforeRollbackRestore: @Sendable () throws -> Void
    /// Default-inert Phase 22 observations. They cannot change computation.
    let observeRoute: @Sendable (Int, Int, [Float], [Int], [Float], Float) -> Void
    let observeRawLogits: @Sendable (Int, Int32, [Float]) -> Void
    let observePublicLogitsAndSample: @Sendable (Int, [UInt16], Int32) -> Void
    /// Position, layer (-1 for final stages), stage, actual FP32 activations.
    /// Nil avoids diagnostic readbacks and copies in ordinary generation.
    let observeActivation: (@Sendable (Int, Int, String, [Float]) -> Void)?
    /// Successful consumed position, token ID, supplied image row and M-RoPE position.
    /// Nil retains no diagnostic rows and performs no additional copies or readbacks.
    let observeConsumedInput: (@Sendable (Int, Int32, [Float]?, QwenMRoPEPosition?) -> Void)?

    init(
        beforeProtectedExpertRead: @escaping @Sendable
            (Int, Int, QwenBF16ExpertReadHooks.Stream) throws -> Void = { _, _, _ in },
        afterActualGPUSubmission: (@Sendable (String) async throws -> Void)? = nil,
        afterActualGPUCompletion: (@Sendable (String) async throws -> Void)? = nil,
        betweenConsumedTokens: (@Sendable (Int) async throws -> Void)? = nil,
        beforeTurnCommit: @escaping @Sendable () async throws -> Void = {},
        beforeRollbackRestore: @escaping @Sendable () throws -> Void = {},
        observeRoute: @escaping @Sendable (Int, Int, [Float], [Int], [Float], Float) -> Void = { _, _, _, _, _, _ in },
        observeRawLogits: (@Sendable (Int, Int32, [Float]) -> Void)? = nil,
        observePublicLogitsAndSample: @escaping @Sendable (Int, [UInt16], Int32) -> Void = { _, _, _ in },
        observeActivation: (@Sendable (Int, Int, String, [Float]) -> Void)? = nil,
        observeConsumedInput: (@Sendable (Int, Int32, [Float]?, QwenMRoPEPosition?) -> Void)? = nil
    ) {
        isKnownNone = false
        self.beforeProtectedExpertRead = beforeProtectedExpertRead
        self.afterActualGPUSubmission = afterActualGPUSubmission ?? { _ in }
        self.afterActualGPUCompletion = afterActualGPUCompletion ?? { _ in }
        requiresSeparateGPUStages = afterActualGPUSubmission != nil
            || afterActualGPUCompletion != nil || observeActivation != nil
        requiresOrderedPrefillStages = requiresSeparateGPUStages || betweenConsumedTokens != nil
        requiresTokenMajorPrefill = requiresOrderedPrefillStages
            || observeRawLogits != nil || observeConsumedInput != nil
        self.betweenConsumedTokens = betweenConsumedTokens ?? { _ in }
        self.beforeTurnCommit = beforeTurnCommit
        self.beforeRollbackRestore = beforeRollbackRestore
        self.observeRoute = observeRoute
        self.observeRawLogits = observeRawLogits ?? { _, _, _ in }
        self.observePublicLogitsAndSample = observePublicLogitsAndSample
        self.observeActivation = observeActivation
        self.observeConsumedInput = observeConsumedInput
    }

    private init(knownNone: Void) {
        isKnownNone = true
        beforeProtectedExpertRead = { _, _, _ in }
        afterActualGPUSubmission = { _ in }
        afterActualGPUCompletion = { _ in }
        requiresSeparateGPUStages = false
        requiresTokenMajorPrefill = false
        requiresOrderedPrefillStages = false
        betweenConsumedTokens = { _ in }
        beforeTurnCommit = {}
        beforeRollbackRestore = {}
        observeRoute = { _, _, _, _, _, _ in }
        observeRawLogits = { _, _, _ in }
        observePublicLogitsAndSample = { _, _, _ in }
        observeActivation = nil
        observeConsumedInput = nil
    }
    static let none = Self(knownNone: ())
}

struct QwenOfficialSourceSamplingInput: Sendable {
    let logits: [Float]
    let retainedTokenIDs: [Int32]
}

/// Only committed KV prefixes are observable. Uncommitted tail bytes may be
/// overwritten on a new branch and are deliberately excluded from equality.
struct QwenOfficialSourceConversationDiagnosticSnapshot: Sendable, Equatable {
    let retainedTokenIDs: [Int32]
    let consumedTokenIDs: [Int32]
    let pendingAcceptedToken: Int32?
    let currentLogits: [Float]?
    let runner: QwenOfficialSourceRunnerDiagnosticSnapshot
    let sourceIdentity: LoadedRuntimeSourceIdentity?
    let activeTransaction: ConversationTransactionID?
    let unusable: Bool
}

/// A copy of the accepted journal only. An active transaction is reported but
/// none of its provisional tokens, logits, or live GPU state are included.
struct QwenOfficialSourceCommittedJournalSnapshot: Sendable, Equatable {
    let retainedTokenIDs: [Int32]
    let consumedTokenIDs: [Int32]
    let pendingAcceptedToken: Int32?
    let currentLogits: [Float]?
    let sourceIdentity: LoadedRuntimeSourceIdentity?
    let activeTransaction: ConversationTransactionID?
    let unusable: Bool
}

/// The same bounded real source runner underlies production and internal tiny
/// protected fixtures. A turn has exactly one rollback baseline. Accepted
/// tokens and logits remain provisional until one final no-suspension swap.
actor QwenOfficialSourceConversationState: ConversationStateTransaction {
    private struct Aggregate: Sendable {
        var consumedTokenIDs: [Int32] = []
        var pendingAcceptedToken: Int32?
        var currentLogits: [Float]?
        var textRoPEDelta = 0
        var usesMRoPE = false
        var retainedTokenIDs: [Int32] {
            consumedTokenIDs + (pendingAcceptedToken.map { [$0] } ?? [])
        }
    }

    private struct Transaction {
        let id: ConversationTransactionID
        let begin: Aggregate
        var working: Aggregate
        var promptTokenIDs: [Int32] = []
        var preparedPrompt: QwenPreparedPrefill?
        var generatedTokenIDs: [Int32] = []
    }

    nonisolated let modelIdentity: LoadedRuntimeSourceIdentity?
    nonisolated let contextLimit: Int
    private let model: QwenOfficialSourceModel
    private let runner: QwenOfficialSourceRunner
    private let hooks: QwenOfficialSourceTransactionHooks
    private let groupedPrefillEnabled: Bool
    private let vocabularySize: Int
    private let fixedStateBytes: UInt64
    private var committed = Aggregate()
    private var active: Transaction?
    private var nextID: UInt64 = 1
    private var mutating = false
    private var unusable = false
    private var legacyPrefillDiagnostics: QwenSourceGroupedPrefillDiagnostics?

    init(model: QwenOfficialSourceModel, context: MetalContext,
         maxContext: Int, expertSlotCount: Int,
         groupedPrefillEnabled: Bool = true,
         hooks: QwenOfficialSourceTransactionHooks = .none) async throws {
        guard maxContext > 0, model.architecture.vocabularySize > 0,
              expertSlotCount == model.expertCacheSlots else {
            throw QwenTextRunnerError.invalidState(
                detail: "source conversation context/vocabulary")
        }
        guard context.device === model.context.device,
              context.queue === model.context.queue else {
            throw QwenTextRunnerError.invalidState(detail: "source context binding changed")
        }
        let runner = try QwenOfficialSourceRunner(
            model: model, maxContext: maxContext,
            expertSlotCount: expertSlotCount, hooks: hooks)
        let snapshot = try await runner.diagnosticSnapshot()
        let linearElements = snapshot.linear.layers.values.reduce(UInt64(0)) {
            $0 + UInt64($1.convolutionHistory.count + $1.recurrentMatrix.count)
        }
        let kvBytes = UInt64(model.architecture.fullAttentionLayerMask.filter { $0 == 1 }.count)
            * UInt64(maxContext)
            * UInt64(model.architecture.keyValueHeads)
            * UInt64(model.architecture.headDimension) * 2 * 4
        let (bytes, overflow) = (linearElements * 4).addingReportingOverflow(kvBytes)
        guard !overflow else {
            throw QwenTextRunnerError.invalidState(detail: "source state byte count overflow")
        }
        self.model = model
        self.runner = runner
        self.hooks = hooks
        self.groupedPrefillEnabled = groupedPrefillEnabled
        vocabularySize = model.architecture.vocabularySize
        modelIdentity = model.sourceIdentity
        contextLimit = maxContext
        fixedStateBytes = bytes
    }

    func begin() async throws -> ConversationTransactionID {
        try requireIdle()
        try Task.checkCancellation()
        try model.revalidateSource()
        try await runner.beginTurn()
        let id = ConversationTransactionID(rawValue: nextID)
        nextID &+= 1
        active = Transaction(id: id, begin: committed, working: committed)
        return id
    }

    func prefill(_ tokenIDs: [Int32], transaction: ConversationTransactionID,
                 onProgress: @Sendable (Int, Int) async -> Void) async throws {
        try await prefill(tokenIDs, transaction: transaction,
                          groupedCapture: nil, onProgress: onProgress)
    }

    func prefill(_ tokenIDs: [Int32], transaction: ConversationTransactionID,
                 groupedCapture: QwenSourceGroupedPrefillCapture?,
                 onProgress: @Sendable (Int, Int) async -> Void) async throws {
        try requireMutable(transaction)
        guard !tokenIDs.isEmpty else {
            throw ConversationStateTransactionError.invalidBoundary(
                "source prefill requires at least one token")
        }
        guard tokenIDs.count <= contextLimit - retainedCount() else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: retainedCount() + tokenIDs.count, maximum: contextLimit)
        }
        mutating = true
        defer { mutating = false }
        let tokenMajor = !groupedPrefillEnabled || (groupedCapture == nil
            ? hooks.requiresTokenMajorPrefill : hooks.requiresOrderedPrefillStages)
        legacyPrefillDiagnostics = tokenMajor
            ? QwenSourceGroupedPrefillDiagnostics(mode: .tokenMajor, tokenCount: tokenIDs.count) : nil
        if tokenMajor {
            for (index, token) in tokenIDs.enumerated() {
                try Task.checkCancellation()
                try await consumePending(transaction: transaction)
                try await consume(token, transaction: transaction)
                active?.promptTokenIDs.append(token)
                await onProgress(index + 1, tokenIDs.count)
            }
        } else {
            try await consumePending(transaction: transaction)
            guard let working = active?.working else {
                throw ConversationStateTransactionError.staleTransaction
            }
            let position = working.consumedTokenIDs.count
            let positions: [QwenMRoPEPosition]?
            if working.usesMRoPE {
                positions = try tokenIDs.indices.map { index in
                    let absolute = Int64(position + index) + Int64(working.textRoPEDelta)
                    guard absolute >= 0, absolute <= Int64(Int32.max) else {
                        throw QwenVisionError.invalidPositions
                    }
                    return try QwenMRoPEPosition(temporal: Int(absolute),
                        height: Int(absolute), width: Int(absolute))
                }
            } else { positions = nil }
            let logits = try await runner.prefill(tokenIDs: tokenIDs, position: position,
                mropePositions: positions, groupedCapture: groupedCapture, onProgress: onProgress)
            try publishPrefill(tokenIDs, logits: logits, transaction: transaction)
        }
    }

    /// The prepared value and its feature owners are held by this transaction
    /// through prefill and suffix replay; only IDs and the text position delta
    /// survive commit. A failure rolls back the original runner baseline.
    func prefillPrepared(_ prepared: QwenPreparedPrefill,
                         transaction: ConversationTransactionID,
                         groupedCapture: QwenSourceGroupedPrefillCapture? = nil,
                         onProgress: @Sendable (Int, Int) async -> Void) async throws {
        try requireMutable(transaction)
        guard !prepared.tokenIDs.isEmpty,
              prepared.positions.count == prepared.tokenIDs.count,
              prepared.tokenIDs.count <= contextLimit - retainedCount(),
              active?.promptTokenIDs.isEmpty == true,
              let current = active?.working,
              let newDelta = Int(exactly: Int64(current.textRoPEDelta)
                + Int64(prepared.textRoPEDelta)),
              newDelta >= -contextLimit, newDelta <= contextLimit else {
            throw ConversationStateTransactionError.invalidBoundary("source image prefill boundary")
        }
        mutating = true
        defer { mutating = false }
        active?.preparedPrompt = prepared
        let tokenMajor = !groupedPrefillEnabled || (groupedCapture == nil
            ? hooks.requiresTokenMajorPrefill : hooks.requiresOrderedPrefillStages)
        legacyPrefillDiagnostics = tokenMajor
            ? QwenSourceGroupedPrefillDiagnostics(mode: .tokenMajor, tokenCount: prepared.tokenIDs.count) : nil
        if tokenMajor {
            for (index, token) in prepared.tokenIDs.enumerated() {
                try Task.checkCancellation()
                try await consumePending(transaction: transaction)
                try await consume(token, transaction: transaction,
                                  featureRow: try Self.featureRow(prepared, at: index),
                                  mropePosition: prepared.positions[index])
                active?.promptTokenIDs.append(token)
                await onProgress(index + 1, prepared.tokenIDs.count)
            }
        } else {
            try await consumePending(transaction: transaction)
            guard let position = active?.working.consumedTokenIDs.count else {
                throw ConversationStateTransactionError.staleTransaction
            }
            let logits = try await runner.prefill(tokenIDs: prepared.tokenIDs, position: position,
                featureRowAt: { index in try Self.featureRow(prepared, at: index) },
                mropePositions: prepared.positions, groupedCapture: groupedCapture,
                onProgress: onProgress)
            try publishPrefill(prepared.tokenIDs, logits: logits, transaction: transaction)
        }
        active?.working.textRoPEDelta = newDelta
        active?.working.usesMRoPE = true
    }

    func committedTextRoPEDelta() -> Int { committed.textRoPEDelta }

    func advance(_ tokenID: Int32, transaction: ConversationTransactionID) async throws {
        try requireMutable(transaction)
        try validate(tokenID)
        guard retainedCount() < contextLimit else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: retainedCount() + 1, maximum: contextLimit)
        }
        mutating = true
        defer { mutating = false }
        try Task.checkCancellation()
        try await consumePending(transaction: transaction)
        active?.working.pendingAcceptedToken = tokenID
        active?.generatedTokenIDs.append(tokenID)
    }

    func prepareSampling(transaction: ConversationTransactionID)
        async throws -> QwenOfficialSourceSamplingInput {
        try requireMutable(transaction)
        mutating = true
        defer { mutating = false }
        try Task.checkCancellation()
        try await consumePending(transaction: transaction)
        guard let working = active?.working, let logits = working.currentLogits,
              logits.count == vocabularySize else {
            throw ConversationStateTransactionError.invalidBoundary(
                "source sampling requires a nonempty consumed prompt")
        }
        return QwenOfficialSourceSamplingInput(
            logits: logits, retainedTokenIDs: working.consumedTokenIDs)
    }

    func removeSuffix(tokenCount: Int,
                      transaction: ConversationTransactionID) async throws {
        try requireMutable(transaction)
        guard let current = active, tokenCount > 0,
              tokenCount <= current.generatedTokenIDs.count else {
            throw ConversationStateTransactionError.invalidBoundary(
                "source suffix crosses the generated-token boundary")
        }
        let kept = Array(current.generatedTokenIDs.dropLast(tokenCount))
        mutating = true
        defer { mutating = false }
        do {
            try await runner.restoreTurnBaseline()
            active?.working = current.begin
            active?.generatedTokenIDs = []
            if !current.promptTokenIDs.isEmpty {
                try await consumePending(transaction: transaction)
            }
            for (index, token) in current.promptTokenIDs.enumerated() {
                let feature: [Float]?
                if let prepared = current.preparedPrompt {
                    feature = try Self.featureRow(prepared, at: index)
                } else { feature = nil }
                try await consume(token, transaction: transaction,
                    featureRow: feature,
                    mropePosition: current.preparedPrompt?.positions[index])
            }
            active?.working.textRoPEDelta = current.working.textRoPEDelta
            active?.working.usesMRoPE = current.working.usesMRoPE
            for token in kept {
                try Task.checkCancellation()
                try await consumePending(transaction: transaction)
                active?.working.pendingAcceptedToken = token
                active?.generatedTokenIDs.append(token)
            }
        } catch {
            // Leave this transaction rollback-capable; a failed restoration
            // itself closes the runner and makes all further use impossible.
            if await runner.isUnusable { unusable = true }
            throw error
        }
    }

    func commit(transaction: ConversationTransactionID)
        async throws -> ConversationStateMetrics {
        try await commit(transaction: transaction, shouldStop: { false })
    }

    /// The source-only host stop is sampled again after the precommit gate,
    /// immediately before the runner discards its rollback snapshot.
    func commit(transaction: ConversationTransactionID,
                shouldStop: @Sendable () -> Bool,
                validateCompanion: @Sendable () throws -> Void = {})
        async throws -> ConversationStateMetrics {
        try requireMutable(transaction)
        guard let current = active else {
            throw ConversationStateTransactionError.staleTransaction
        }
        mutating = true
        defer { mutating = false }
        try Task.checkCancellation()
        if shouldStop() { throw CancellationError() }
        try model.revalidateSource()
        try await hooks.beforeTurnCommit()
        try Task.checkCancellation()
        if shouldStop() { throw CancellationError() }
        try model.revalidateSource()
        let metrics = Self.metrics(current.working, fixedStateBytes: fixedStateBytes)
        let runnerPosition = await runner.position
        guard metrics.retainedTokenIDs.count <= contextLimit,
              current.working.currentLogits?.count == vocabularySize,
              runnerPosition == current.working.consumedTokenIDs.count else {
            throw ConversationStateTransactionError.invalidBoundary(
                "source token journal, logits or runner position changed")
        }
        // The last host-stop check precedes the runner's synchronous source
        // and companion validation. Those validations occur inside the runner
        // after this await and immediately before it releases its checkpoint;
        // a failure leaves rollback possible. Nothing suspends or throws after
        // finishTurn succeeds and before the accepted journal swap.
        if shouldStop() { throw CancellationError() }
        try await runner.finishTurn(validateCompanion: validateCompanion)
        committed = current.working
        active = nil
        return metrics
    }

    func rollback(transaction: ConversationTransactionID) async throws {
        try requireMutable(transaction)
        mutating = true
        defer { mutating = false }
        do {
            try await runner.rollbackTurn()
            active = nil
        } catch {
            unusable = true
            active = nil
            throw ConversationStateTransactionError.restoreFailed("\(error)")
        }
    }

    func rebuildCheckpoint(retaining tokenIDs: [Int32],
                           transaction: ConversationTransactionID,
                           onProgress: @Sendable (Int, Int) async -> Void) async throws {
        try await beginCheckpointReplacement(transaction: transaction)
        try await prefill(tokenIDs, transaction: transaction, onProgress: onProgress)
    }

    func rebuildPreparedCheckpoint(_ prepared: QwenPreparedPrefill,
        transaction: ConversationTransactionID,
        onProgress: @Sendable (Int, Int) async -> Void) async throws {
        try await beginCheckpointReplacement(transaction: transaction)
        try await prefillPrepared(prepared, transaction: transaction, onProgress: onProgress)
    }

    private func beginCheckpointReplacement(transaction: ConversationTransactionID) async throws {
        try requireMutable(transaction)
        guard active?.promptTokenIDs.isEmpty == true,
              active?.generatedTokenIDs.isEmpty == true else {
            throw ConversationStateTransactionError.invalidBoundary("checkpoint must begin an untouched turn")
        }
        mutating = true
        defer { mutating = false }
        try Task.checkCancellation()
        try await runner.beginCheckpointReplacement()
        active?.working = Aggregate()
    }

    func reset() async throws {
        try requireIdle()
        mutating = true
        defer { mutating = false }
        do {
            try await runner.resetConversation()
            committed = Aggregate()
        } catch {
            unusable = true
            throw ConversationStateTransactionError.restoreFailed("\(error)")
        }
    }

    func status() -> ConversationStateStatus {
        ConversationStateStatus(
            committed: Self.metrics(committed, fixedStateBytes: fixedStateBytes),
            working: active.map { Self.metrics($0.working, fixedStateBytes: fixedStateBytes) },
            activeTransaction: active?.id)
    }

    /// Safe during a suspended or poisoned turn: reads only the committed
    /// value journal and actor-owned status, never runner or Metal buffers.
    func committedJournalSnapshot() -> QwenOfficialSourceCommittedJournalSnapshot {
        QwenOfficialSourceCommittedJournalSnapshot(
            retainedTokenIDs: committed.retainedTokenIDs,
            consumedTokenIDs: committed.consumedTokenIDs,
            pendingAcceptedToken: committed.pendingAcceptedToken,
            currentLogits: committed.currentLogits,
            sourceIdentity: modelIdentity,
            activeTransaction: active?.id,
            unusable: unusable)
    }

    func diagnosticSnapshot() async throws
        -> QwenOfficialSourceConversationDiagnosticSnapshot {
        guard !mutating else { throw ConversationStateTransactionError.busy }
        let state = active?.working ?? committed
        return QwenOfficialSourceConversationDiagnosticSnapshot(
            retainedTokenIDs: state.retainedTokenIDs,
            consumedTokenIDs: state.consumedTokenIDs,
            pendingAcceptedToken: state.pendingAcceptedToken,
            currentLogits: state.currentLogits,
            runner: try await runner.diagnosticSnapshot(),
            sourceIdentity: modelIdentity,
            activeTransaction: active?.id,
            unusable: unusable)
    }

    nonisolated var currentRoutedExpertCacheSummary: RoutedExpertCacheSummary? {
        runner.currentRoutedExpertCacheSummary
    }

    func cacheDiagnostics() async -> QwenOfficialSourceCacheDiagnostics {
        await runner.cacheDiagnostics()
    }

    private func consumePending(transaction: ConversationTransactionID) async throws {
        guard let token = active?.working.pendingAcceptedToken else { return }
        try await consume(token, transaction: transaction)
        active?.working.pendingAcceptedToken = nil
    }

    private static func featureRow(_ prepared: QwenPreparedPrefill,
                                   at index: Int) throws -> [Float]? {
        for override in prepared.featureOverrides where override.tokenRange.contains(index) {
            let owner = override.owner
            let row = index - override.tokenRange.lowerBound
            guard row >= 0, row < owner.rowCount else {
                throw QwenTextRunnerError.invalidState(detail: "source image row outside owner")
            }
            let start = row * owner.hiddenSize
            let pointer = owner.featureBuffer.contents().assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: pointer.advanced(by: start),
                                             count: owner.hiddenSize))
        }
        return nil
    }

    private func consume(_ token: Int32,
                         transaction: ConversationTransactionID,
                         featureRow: [Float]? = nil,
                         mropePosition: QwenMRoPEPosition? = nil) async throws {
        try validate(token)
        guard let position = active?.working.consumedTokenIDs.count,
              position < contextLimit else {
            throw ConversationStateTransactionError.contextExceeded(
                requested: contextLimit + 1, maximum: contextLimit)
        }
        let positionValue: QwenMRoPEPosition?
        if let mropePosition { positionValue = mropePosition }
        else if let working = active?.working, working.usesMRoPE {
            let absolute = Int64(position) + Int64(working.textRoPEDelta)
            guard absolute >= 0, absolute <= Int64(Int32.max) else {
                throw QwenVisionError.invalidPositions
            }
            positionValue = try QwenMRoPEPosition(
                temporal: Int(absolute), height: Int(absolute), width: Int(absolute))
        } else { positionValue = nil }
        let logits = try await runner.produce(
            token: token, position: position,
            featureRow: featureRow, mropePosition: positionValue)
        guard logits.count == vocabularySize, logits.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.execution(detail: "source conversation logits invalid")
        }
        guard active?.id == transaction else {
            throw ConversationStateTransactionError.staleTransaction
        }
        active?.working.consumedTokenIDs.append(token)
        active?.working.currentLogits = logits
        try await hooks.betweenConsumedTokens(position + 1)
    }

    func groupedPrefillDiagnostics() async -> QwenSourceGroupedPrefillDiagnostics? {
        if let legacyPrefillDiagnostics { return legacyPrefillDiagnostics }
        return await runner.groupedPrefillDiagnostics()
    }

    /// Runner has validated and advanced every layer. Journal publication has
    /// no suspension and remains provisional until the existing turn commit.
    private func publishPrefill(_ tokenIDs: [Int32], logits: [Float],
                                transaction: ConversationTransactionID) throws {
        guard logits.count == vocabularySize, logits.allSatisfy(\.isFinite) else {
            throw QwenTextRunnerError.execution(detail: "source conversation logits invalid")
        }
        guard active?.id == transaction else {
            throw ConversationStateTransactionError.staleTransaction
        }
        active?.working.consumedTokenIDs.append(contentsOf: tokenIDs)
        active?.working.currentLogits = logits
        active?.promptTokenIDs.append(contentsOf: tokenIDs)
    }

    private func retainedCount() -> Int { active?.working.retainedTokenIDs.count ?? 0 }

    private func validate(_ token: Int32) throws {
        guard token >= 0, Int(token) < vocabularySize else {
            throw QwenTextRunnerError.invalidToken(id: token)
        }
    }

    private func requireIdle() throws {
        if unusable { throw ConversationStateTransactionError.restoreFailed("source session unusable") }
        guard !mutating, active == nil else { throw ConversationStateTransactionError.busy }
    }

    private func requireMutable(_ transaction: ConversationTransactionID) throws {
        if unusable { throw ConversationStateTransactionError.restoreFailed("source session unusable") }
        guard !mutating else { throw ConversationStateTransactionError.busy }
        guard active?.id == transaction else {
            throw ConversationStateTransactionError.staleTransaction
        }
    }

    private static func metrics(_ state: Aggregate,
                                fixedStateBytes: UInt64) -> ConversationStateMetrics {
        let retained = state.retainedTokenIDs
        return ConversationStateMetrics(
            retainedTokenIDs: retained,
            consumedTokenCount: state.consumedTokenIDs.count,
            pendingTokenCount: state.pendingAcceptedToken == nil ? 0 : 1,
            logicalStateBytes: fixedStateBytes
                + UInt64(retained.count) * UInt64(MemoryLayout<Int32>.stride)
                + UInt64(state.currentLogits?.count ?? 0) * UInt64(MemoryLayout<Float>.stride))
    }
}
