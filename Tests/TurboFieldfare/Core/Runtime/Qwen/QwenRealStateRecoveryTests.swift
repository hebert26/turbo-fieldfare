import Foundation
import Metal
import Synchronization
import Testing
@testable import TurboFieldfare

/// Opt-in P22 tests for an installed, verified Qwen artifact. They intentionally
/// do not fall back to the tiny fixture: an unset environment disables the
/// suite, while a selected but missing or invalid artifact fails the test.
@Suite(.serialized)
struct QwenRealStateRecoveryTests {
    @Test(.enabled(if: RealQwenArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact state recovery"))
    func realSoftStopCommitsAcceptedPendingToken() async throws {
        let artifact = try RealQwenArtifact.load()
        let state = try await artifact.makeState()
        let conversation = MultimodalConversation(
            qwenState: state, maxContext: RealQwenArtifact.maxContext)
        let stopRequested = Mutex(false)
        let prompt = RealQwenArtifact.promptTokenIDs
        let generated = RealQwenArtifact.generatedTokenIDs

        let result = try await conversation.applyTokenizedTurn(
            TokenizedConversationTurn(
                promptTokenIDs: prompt, generatedTokenIDs: generated),
            shouldStop: { stopRequested.withLock { $0 } },
            onProgress: { progress in
                if case .accepted(index: 0, tokenID: generated[0]) = progress {
                    stopRequested.withLock { $0 = true }
                }
            })

        #expect(result.reason == .softStop)
        #expect(result.acceptedGeneratedTokenIDs == [generated[0]])
        #expect(result.metrics.retainedTokenIDs == prompt + [generated[0]])
        #expect(result.metrics.consumedTokenCount == prompt.count)
        #expect(result.metrics.pendingTokenCount == 1)
        let recovered = try await state.diagnosticSnapshot()

        let actualNextConversation = MultimodalConversation(
            qwenState: state, maxContext: RealQwenArtifact.maxContext)
        let actualNext = try await actualNextConversation.applyTokenizedTurn(
            TokenizedConversationTurn(
                promptTokenIDs: [RealQwenArtifact.nextPromptToken],
                generatedTokenIDs: [generated[1]]))
        let actualNextSnapshot = try await state.diagnosticSnapshot()

        // The model is reused after reset for a clean sequential reference.
        try await state.reset()
        let cleanConversation = MultimodalConversation(
            qwenState: state, maxContext: RealQwenArtifact.maxContext)
        _ = try await cleanConversation.applyTokenizedTurn(
            TokenizedConversationTurn(
                promptTokenIDs: prompt, generatedTokenIDs: [generated[0]]))
        let clean = try await state.diagnosticSnapshot()
        var comparisons: [RealStateComparisonEvidence] = []
        comparisons.append(expectRealStateMatches(
            recovered, clean, label: "soft-stop clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        let cleanNext = try await cleanConversation.applyTokenizedTurn(
            TokenizedConversationTurn(
                promptTokenIDs: [RealQwenArtifact.nextPromptToken],
                generatedTokenIDs: [generated[1]]))
        let cleanNextSnapshot = try await state.diagnosticSnapshot()
        #expect(actualNext == cleanNext)
        comparisons.append(expectRealStateMatches(
            actualNextSnapshot, cleanNextSnapshot,
            label: "soft-stop continuation clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        QwenRealEvidence.report(
            suite: "state-soft-stop", artifact: artifact,
            tokens: actualNext.metrics.retainedTokenIDs,
            state: actualNextSnapshot, reference: cleanNextSnapshot,
            comparisons: comparisons,
            note: "stop=softStop "
                + "softStopSnapshot=\(realStateReferenceSummary(recovered))")
    }

    @Test(.enabled(if: RealQwenArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact state recovery"))
    func realHardCancellationRollsBackAndAllowsReuse() async throws {
        let artifact = try RealQwenArtifact.load()
        let context = try MetalContext()
        let model = try artifact.loadModel(device: context.device)
        let submitted = AsyncLatch()
        let release = AsyncLatch()
        let event = try #require(context.device.makeSharedEvent())
        let eventBox = SharedEventBox(event)
        let gate = CancellationBlockGate(
            event: eventBox, submitted: submitted, release: release)
        let state = try await QwenConversationState(
            model: model,
            context: context,
            maxContext: RealQwenArtifact.maxContext,
            executionHooks: QwenTextExecutionHooks(
                afterCommandSubmission: { stage in
                    await gate.observe(stage)
                }),
            publicationHooks: .none)

        // Establish a non-empty committed prefix before cancellation. The
        // gate is armed only for the next token so rollback is compared with a
        // genuine committed state, not merely an empty initial aggregate.
        let prefixTransaction = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.promptTokenIDs[0]], transaction: prefixTransaction) { _, _ in }
        _ = try await state.commit(transaction: prefixTransaction)
        let blocker = try #require(context.queue.makeCommandBuffer())
        blocker.encodeWaitForEvent(event, value: 1)
        blocker.commit()
        // The queue now owns a command waiting on this event. Keep the
        // cleanup active across every later throwing setup path, including a
        // failed transaction begin or a task construction failure.
        defer { eventBox.event.signaledValue = 1 }
        await gate.arm()

        let transaction = try await state.begin()
        let outcome = CancellationOutcome()
        let operation = Task {
            do {
                try await state.prefill(
                    [RealQwenArtifact.promptTokenIDs[1]], transaction: transaction) { _, _ in }
                await outcome.record(error: nil)
            } catch {
                await outcome.record(error: String(describing: error))
            }
        }

        let didSubmit = await submitted.wait(timeoutNanoseconds: 5_000_000_000)
        guard didSubmit else {
            operation.cancel()
            await release.signal()
            eventBox.event.signaledValue = 1
            _ = await operation.value
            if let timeoutSnapshot = try? await state.diagnosticSnapshot() {
                QwenRealEvidence.report(
                    suite: "state-hard-cancel-gate-timeout", artifact: artifact,
                    tokens: timeoutSnapshot.retainedTokenIDs,
                    state: timeoutSnapshot,
                    note: "cancellation gate was not reached within 5 seconds")
            }
            Issue.record("real cancellation gate was not reached within its bound")
            return
        }
        operation.cancel()
        #expect(await outcome.finished == false)
        #expect((await state.status()).activeTransaction == transaction)
        await release.signal()
        _ = await operation.value
        #expect(await outcome.finished)
        let cancellationError = await outcome.errorDescription
        #expect(cancellationError != nil)
        #expect(cancellationError?.localizedCaseInsensitiveContains("cancel") == true)
        try await state.rollback(transaction: transaction)

        let recovered = try await state.diagnosticSnapshot()
        var comparisons: [RealStateComparisonEvidence] = []
        let actualNext = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: actualNext) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: actualNext)
        let actualNextMetrics = try await state.commit(transaction: actualNext)
        let actualNextSnapshot = try await state.diagnosticSnapshot()

        try await state.reset()
        let clean = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.promptTokenIDs[0]], transaction: clean) { _, _ in }
        _ = try await state.commit(transaction: clean)
        let cleanSnapshot = try await state.diagnosticSnapshot()
        comparisons.append(expectRealStateMatches(
            recovered, cleanSnapshot, label: "hard-cancel clean prefix",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        let cleanNext = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: cleanNext) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: cleanNext)
        let cleanNextMetrics = try await state.commit(transaction: cleanNext)
        let cleanNextSnapshot = try await state.diagnosticSnapshot()
        #expect(actualNextMetrics == cleanNextMetrics)
        comparisons.append(expectRealStateMatches(
            actualNextSnapshot, cleanNextSnapshot,
            label: "hard-cancel continuation clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        QwenRealEvidence.report(
            suite: "state-hard-cancel", artifact: artifact,
            tokens: actualNextMetrics.retainedTokenIDs, state: actualNextSnapshot,
            reference: cleanNextSnapshot,
            comparisons: comparisons,
            note: "thrown=\(String(describing: cancellationError)) "
                + "rollbackSnapshot=\(realStateReferenceSummary(recovered))")
    }

    @Test(.enabled(if: RealQwenArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact state recovery"))
    func realInjectedErrorRollsBackToTheCommittedState() async throws {
        let artifact = try RealQwenArtifact.load()
        let context = try MetalContext()
        let model = try artifact.loadModel(device: context.device)
        let gate = LayerFailureGate(failAfterLayer: 1)
        let state = try await QwenConversationState(
            model: model,
            context: context,
            maxContext: RealQwenArtifact.maxContext,
            executionHooks: QwenTextExecutionHooks(
            beforeLayer: { layer in try await gate.check(layer) }),
            publicationHooks: .none)

        let prefixTransaction = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.promptTokenIDs[0]], transaction: prefixTransaction) { _, _ in }
        _ = try await state.commit(transaction: prefixTransaction)
        await gate.arm()

        let transaction = try await state.begin()
        await #expect(throws: InjectedStateFailure.self) {
            try await state.prefill(
                Array(RealQwenArtifact.promptTokenIDs.dropFirst()), transaction: transaction) { _, _ in }
        }
        #expect((await state.status()).activeTransaction == transaction)
        try await state.rollback(transaction: transaction)

        let recovered = try await state.diagnosticSnapshot()
        var comparisons: [RealStateComparisonEvidence] = []
        let actualNext = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: actualNext) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: actualNext)
        let actualNextMetrics = try await state.commit(transaction: actualNext)
        let actualNextSnapshot = try await state.diagnosticSnapshot()

        try await state.reset()
        let clean = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.promptTokenIDs[0]], transaction: clean) { _, _ in }
        _ = try await state.commit(transaction: clean)
        let cleanSnapshot = try await state.diagnosticSnapshot()
        comparisons.append(expectRealStateMatches(
            recovered, cleanSnapshot, label: "error-rollback clean prefix",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        let cleanNext = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: cleanNext) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: cleanNext)
        let cleanNextMetrics = try await state.commit(transaction: cleanNext)
        let cleanNextSnapshot = try await state.diagnosticSnapshot()
        #expect(actualNextMetrics == cleanNextMetrics)
        comparisons.append(expectRealStateMatches(
            actualNextSnapshot, cleanNextSnapshot,
            label: "error-rollback continuation clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        QwenRealEvidence.report(
            suite: "state-error-rollback", artifact: artifact,
            tokens: actualNextMetrics.retainedTokenIDs, state: actualNextSnapshot,
            reference: cleanNextSnapshot,
            comparisons: comparisons,
            note: "thrown=InjectedStateFailure "
                + "rollbackSnapshot=\(realStateReferenceSummary(recovered))")
    }

    @Test(.enabled(if: RealQwenArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact state recovery"))
    func realMatchedSuffixTrimRemovesOnlyTheHiddenGeneratedTail() async throws {
        let artifact = try RealQwenArtifact.load()
        let state = try await artifact.makeState()
        let transaction = try await state.begin()
        try await state.prefill(
            RealQwenArtifact.promptTokenIDs, transaction: transaction) { _, _ in }
        for token in RealQwenArtifact.generatedTokenIDs {
            try await state.advance(token, transaction: transaction)
        }

        try await state.removeSuffix(
            tokenCount: RealQwenArtifact.hiddenSuffixCount, transaction: transaction)
        let working = try #require((await state.status()).working)
        let expected = RealQwenArtifact.promptTokenIDs
            + Array(RealQwenArtifact.generatedTokenIDs.dropLast(
                RealQwenArtifact.hiddenSuffixCount))
        #expect(working.retainedTokenIDs == expected)
        #expect(working.consumedTokenCount == RealQwenArtifact.promptTokenIDs.count)
        #expect(working.pendingTokenCount == 1)
        let committed = try await state.commit(transaction: transaction)
        #expect(committed.retainedTokenIDs == expected)
        let recovered = try await state.diagnosticSnapshot()
        var comparisons: [RealStateComparisonEvidence] = []
        let actualNext = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: actualNext) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: actualNext)
        let actualNextMetrics = try await state.commit(transaction: actualNext)
        let actualNextSnapshot = try await state.diagnosticSnapshot()

        try await state.reset()
        let clean = try await state.begin()
        try await state.prefill(
            RealQwenArtifact.promptTokenIDs, transaction: clean) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[0], transaction: clean)
        _ = try await state.commit(transaction: clean)
        let cleanSnapshot = try await state.diagnosticSnapshot()
        comparisons.append(expectRealStateMatches(
            recovered, cleanSnapshot, label: "suffix-trim clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        let cleanNext = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: cleanNext) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: cleanNext)
        let cleanNextMetrics = try await state.commit(transaction: cleanNext)
        let cleanNextSnapshot = try await state.diagnosticSnapshot()
        #expect(actualNextMetrics == cleanNextMetrics)
        comparisons.append(expectRealStateMatches(
            actualNextSnapshot, cleanNextSnapshot,
            label: "suffix-trim continuation clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        QwenRealEvidence.report(
            suite: "state-suffix-trim", artifact: artifact,
            tokens: actualNextMetrics.retainedTokenIDs, state: actualNextSnapshot,
            reference: cleanNextSnapshot,
            comparisons: comparisons,
            note: "trimmedSnapshot=\(realStateReferenceSummary(recovered))")
    }

    @Test(.enabled(if: RealQwenArtifact.isOptedIn,
                   "set TURBO_FIELDFARE_REAL_QWEN_ARTIFACT to run real-artifact state recovery"))
    func realCheckpointRebuildAndMultiTurnContinuationMatchCleanState() async throws {
        let artifact = try RealQwenArtifact.load()
        let state = try await artifact.makeState()
        let first = try await state.begin()
        try await state.prefill(
            RealQwenArtifact.promptTokenIDs, transaction: first) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[0], transaction: first)
        let prefix = try await state.commit(transaction: first)
        let checkpoint = try await state.begin()
        try await state.rebuildCheckpoint(
            retaining: prefix.retainedTokenIDs, transaction: checkpoint) { _, _ in }
        let rebuilt = try await state.commit(transaction: checkpoint)

        let rebuiltSnapshot = try await state.diagnosticSnapshot()
        var comparisons: [RealStateComparisonEvidence] = []
        try await state.reset()

        // Reuse the same loaded real model after reset for the clean reference.
        let clean = try await state.begin()
        try await state.prefill(
            prefix.retainedTokenIDs, transaction: clean) { _, _ in }
        let cleanMetrics = try await state.commit(transaction: clean)
        #expect(rebuilt == cleanMetrics)
        #expect(cleanMetrics.retainedTokenIDs == prefix.retainedTokenIDs)
        #expect(cleanMetrics.pendingTokenCount == 0)
        let cleanSnapshot = try await state.diagnosticSnapshot()
        comparisons.append(expectRealStateMatches(
            rebuiltSnapshot, cleanSnapshot,
            label: "checkpoint clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))

        // Rebuild and continue again on the same model after reset. This keeps
        // the comparison sequential and never creates a second real cache.
        try await state.reset()
        let actualPrefix = try await state.begin()
        try await state.prefill(
            RealQwenArtifact.promptTokenIDs, transaction: actualPrefix) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[0], transaction: actualPrefix)
        let actualPrefixMetrics = try await state.commit(transaction: actualPrefix)
        let actualCheckpoint = try await state.begin()
        try await state.rebuildCheckpoint(
            retaining: actualPrefixMetrics.retainedTokenIDs,
            transaction: actualCheckpoint) { _, _ in }
        _ = try await state.commit(transaction: actualCheckpoint)
        let next = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: next) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: next)
        let actualContinuation = try await state.commit(transaction: next)
        let actualNextSnapshot = try await state.diagnosticSnapshot()

        try await state.reset()
        let cleanPrefix = try await state.begin()
        try await state.prefill(
            prefix.retainedTokenIDs, transaction: cleanPrefix) { _, _ in }
        _ = try await state.commit(transaction: cleanPrefix)
        let cleanNext = try await state.begin()
        try await state.prefill(
            [RealQwenArtifact.nextPromptToken], transaction: cleanNext) { _, _ in }
        try await state.advance(
            RealQwenArtifact.generatedTokenIDs[1], transaction: cleanNext)
        let expectedContinuation = try await state.commit(transaction: cleanNext)
        let expectedNextSnapshot = try await state.diagnosticSnapshot()
        #expect(actualContinuation == expectedContinuation)
        #expect(actualContinuation.retainedTokenIDs
            == prefix.retainedTokenIDs
                + [RealQwenArtifact.nextPromptToken,
                   RealQwenArtifact.generatedTokenIDs[1]])
        comparisons.append(expectRealStateMatches(
            actualNextSnapshot, expectedNextSnapshot,
            label: "checkpoint continuation clean reference",
            expectedReplayGenerationDelta: 1, expectedProducerEpochDelta: 1))
        QwenRealEvidence.report(
            suite: "state-checkpoint-multiturn", artifact: artifact,
            tokens: actualContinuation.retainedTokenIDs, state: actualNextSnapshot,
            reference: expectedNextSnapshot, comparisons: comparisons)
    }
}

private struct RealQwenArtifact {
    static let environmentKey = "TURBO_FIELDFARE_REAL_QWEN_ARTIFACT"
    static let maxContext = 32
    // These are fixed Phase 3 state-operation inputs. They exercise the real
    // Qwen weights and KV ledger, but are intentionally not claimed as greedy
    // model output from this recovery suite.
    static let promptTokenIDs: [Int32] = [1, 4, 7]
    static let generatedTokenIDs: [Int32] = [2, 14, 6]
    static let nextPromptToken: Int32 = 3
    static let hiddenSuffixCount = 2

    static var isOptedIn: Bool {
        guard let value = ProcessInfo.processInfo.environment[environmentKey]
        else { return false }
        return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    let directory: URL
    let manifest: LoadedModelManifest
    let identity: LoadedRuntimeIdentity

    static func load() throws -> Self {
        guard let raw = ProcessInfo.processInfo.environment[environmentKey],
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RealQwenArtifactError.notOptedIn
        }
        let directory = URL(fileURLWithPath: raw, isDirectory: true)
            .standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw RealQwenArtifactError.missingDirectory(directory.path)
        }
        guard case .qwenV2(let manifest) = try ModelFamilyAdmission.classify(
            directoryURL: directory) else {
            throw RealQwenArtifactError.wrongFamily
        }
        let identity = LoadedRuntimeIdentity(descriptor: manifest.descriptor)
        guard identity.family == .qwen3_6,
              identity.modelID == "Qwen/Qwen3.6-35B-A3B",
              identity.sourceRevision == "995ad96eacd98c81ed38be0c5b274b04031597b0" else {
            throw RealQwenArtifactError.identityMismatch(identity.modelID)
        }
        return Self(directory: directory, manifest: manifest, identity: identity)
    }

    func loadModel(device: MTLDevice) throws -> QwenTextModel {
        try QwenTextModel.loadOfficial(
            directoryURL: directory, manifest: manifest, device: device)
    }

    func makeState() async throws -> QwenConversationState {
        let context = try MetalContext()
        return try await QwenConversationState(
            model: loadModel(device: context.device),
            context: context,
            maxContext: Self.maxContext)
    }
}

private enum RealQwenArtifactError: Error, CustomStringConvertible {
    case notOptedIn
    case missingDirectory(String)
    case wrongFamily
    case identityMismatch(String)

    var description: String {
        switch self {
        case .notOptedIn: "real Qwen artifact opt-in is missing"
        case .missingDirectory(let path): "real Qwen artifact is not a directory: \(path)"
        case .wrongFamily: "selected artifact was not admitted as Qwen3.6"
        case .identityMismatch(let modelID): "unexpected Qwen identity: \(modelID)"
        }
    }
}

private struct QwenRealEvidence {
    static func report(
        suite: String,
        artifact: RealQwenArtifact,
        tokens: [Int32],
        state: QwenConversationStateDiagnosticSnapshot,
        reference: QwenConversationStateDiagnosticSnapshot? = nil,
        comparisons: [RealStateComparisonEvidence] = [],
        note: String? = nil
    ) {
        let currentLogitEvidence = comparisons
            .map { $0.evidenceDescription }
            .joined(separator: "|")
        print(
            "P22 case=\(suite) tokenSource=forced-prepared-state-ids "
                + "modelID=\(artifact.identity.modelID) "
                + "revision=\(artifact.identity.sourceRevision) "
                + "format=\(artifact.identity.formatMajor).\(artifact.identity.formatMinor) "
                + "manifest=\(artifact.identity.textManifestSHA256) "
                + "sourceIndex=\(artifact.identity.sourceIndexSHA256) "
                + "policy=\(artifact.identity.quantizationPolicySHA256) "
                + "mapping=\(state.runnerState.architectureIdentity) "
                + "retained=\(state.retainedTokenIDs) "
                + "consumed=\(state.consumedTokenIDs) "
                + "pending=\(String(describing: state.pendingAcceptedToken)) "
                + "tokens=\(tokens) sequence=\(state.runnerState.sequenceLength) "
                + "replayGeneration=\(state.replayGeneration) "
                + "producerEpoch=\(state.producerEpoch) "
                + "logicalStateBytes=\(state.logicalStateBytes) "
                + "runnerTensorComparison=all-layers+currentLogits "
                + "comparisonBound=2e-5+2e-5*scale "
                + "currentLogitEvidence=\(currentLogitEvidence) "
                + "reference=\(reference.map { realStateReferenceSummary($0) } ?? "none")"
                + (note.map { " note=\($0)" } ?? ""))
    }
}

private func realStateReferenceSummary(
    _ state: QwenConversationStateDiagnosticSnapshot
) -> String {
    "retained=\(state.retainedTokenIDs) consumed=\(state.consumedTokenIDs) "
        + "pending=\(String(describing: state.pendingAcceptedToken)) "
        + "sequence=\(state.runnerState.sequenceLength) "
        + "replayGeneration=\(state.replayGeneration) "
        + "producerEpoch=\(state.producerEpoch) "
        + "logicalStateBytes=\(state.logicalStateBytes)"
}

private actor AsyncLatch {
    private var signaled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        guard !signaled else { return }
        signaled = true
        let current = waiters
        waiters.removeAll(keepingCapacity: false)
        for waiter in current { waiter.resume() }
    }

    func wait() async {
        if signaled { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func wait(timeoutNanoseconds: UInt64) async -> Bool {
        var elapsed: UInt64 = 0
        while elapsed < timeoutNanoseconds {
            if signaled { return true }
            do {
                try await Task.sleep(nanoseconds: 10_000_000)
            } catch {
                return false
            }
            elapsed += 10_000_000
        }
        return signaled
    }
}

private actor CancellationOutcome {
    private(set) var finished = false
    private(set) var errorDescription: String?

    func record(error: String?) {
        finished = true
        errorDescription = error
    }
}

private actor CancellationBlockGate {
    let event: SharedEventBox
    let submitted: AsyncLatch
    let release: AsyncLatch
    private var armed = false

    init(event: SharedEventBox, submitted: AsyncLatch, release: AsyncLatch) {
        self.event = event
        self.submitted = submitted
        self.release = release
    }

    func arm() {
        armed = true
    }

    func observe(_ stage: String) async {
        guard armed, stage == "linearConvolution" else { return }
        armed = false
        await submitted.signal()
        await release.wait()
        event.event.signaledValue = 1
    }
}

private struct InjectedStateFailure: Error, Equatable {}

private final class SharedEventBox: @unchecked Sendable {
    let event: MTLSharedEvent

    init(_ event: MTLSharedEvent) {
        self.event = event
    }
}

private struct RealCurrentLogitComparison: Sendable {
    let actualFinite: Bool
    let expectedFinite: Bool
    let actualMaximumAbsoluteValue: Float?
    let expectedMaximumAbsoluteValue: Float?
    let comparisonScale: Float?
    let maximumAbsoluteDelta: Float?

    static let absent = Self(
        actualFinite: true, expectedFinite: true,
        actualMaximumAbsoluteValue: nil, expectedMaximumAbsoluteValue: nil,
        comparisonScale: nil, maximumAbsoluteDelta: nil)

    static let mismatchedPresence = Self(
        actualFinite: false, expectedFinite: false,
        actualMaximumAbsoluteValue: nil, expectedMaximumAbsoluteValue: nil,
        comparisonScale: nil, maximumAbsoluteDelta: nil)

    var evidenceDescription: String {
        if actualMaximumAbsoluteValue == nil,
           expectedMaximumAbsoluteValue == nil,
           comparisonScale == nil,
           maximumAbsoluteDelta == nil {
            return actualFinite && expectedFinite ? "present=false" : "present=mismatch finite=false"
        }
        guard actualFinite, expectedFinite,
              let actualMaximumAbsoluteValue,
              let expectedMaximumAbsoluteValue,
              let comparisonScale,
              let maximumAbsoluteDelta else {
            return "finite=false"
        }
        return "finite=true actualMax=\(actualMaximumAbsoluteValue) "
            + "referenceMax=\(expectedMaximumAbsoluteValue) "
            + "scale=\(comparisonScale) maxAbsDelta=\(maximumAbsoluteDelta)"
    }
}

private struct RealStateComparisonEvidence: Sendable {
    let label: String
    let currentLogits: RealCurrentLogitComparison

    var evidenceDescription: String {
        "\(label){\(currentLogits.evidenceDescription)}"
    }
}

private func expectRealStateMatches(
    _ actual: QwenConversationStateDiagnosticSnapshot,
    _ expected: QwenConversationStateDiagnosticSnapshot,
    label: String,
    expectedReplayGenerationDelta: UInt64 = 0,
    expectedProducerEpochDelta: UInt64 = 0
) -> RealStateComparisonEvidence {
    #expect(actual.retainedTokenIDs == expected.retainedTokenIDs, "\(label): retained ledger")
    #expect(actual.consumedTokenIDs == expected.consumedTokenIDs, "\(label): consumed ledger")
    #expect(actual.pendingAcceptedToken == expected.pendingAcceptedToken, "\(label): pending token")
    #expect(actual.textRoPEDelta == expected.textRoPEDelta, "\(label): RoPE delta")
    #expect(
        expected.replayGeneration == actual.replayGeneration &+ expectedReplayGenerationDelta,
        "\(label): replay generation transition")
    #expect(actual.logicalStateBytes == expected.logicalStateBytes, "\(label): logical bytes")
    #expect(
        expected.producerEpoch == actual.producerEpoch &+ expectedProducerEpochDelta,
        "\(label): producer epoch transition")
    #expect(actual.lineage == expected.lineage, "\(label): lineage")
    #expect(actual.runnerState.architectureIdentity == expected.runnerState.architectureIdentity,
            "\(label): architecture identity")
    #expect(actual.runnerState.sequenceLength == expected.runnerState.sequenceLength,
            "\(label): sequence length")
    #expect(actual.runnerState.layers.count == expected.runnerState.layers.count,
            "\(label): layer count")
    for (index, pair) in zip(actual.runnerState.layers, expected.runnerState.layers).enumerated() {
        switch pair {
        case let (.linear(actualHistory, actualRecurrent), .linear(expectedHistory, expectedRecurrent)):
            expectRealFloatArraysClose(
                actualHistory, expectedHistory,
                label: "\(label): layer \(index) convolution")
            expectRealFloatArraysClose(
                actualRecurrent, expectedRecurrent,
                label: "\(label): layer \(index) recurrent")
        case let (.full(actualKey, actualValue), .full(expectedKey, expectedValue)):
            expectRealFloatArraysClose(
                actualKey, expectedKey,
                label: "\(label): layer \(index) key")
            expectRealFloatArraysClose(
                actualValue, expectedValue,
                label: "\(label): layer \(index) value")
        default:
            Issue.record("\(label): layer \(index) changed state kind")
        }
    }
    let currentLogitEvidence: RealCurrentLogitComparison
    switch (actual.currentLogits, expected.currentLogits) {
    case let (.some(actualLogits), .some(expectedLogits)):
        currentLogitEvidence = expectRealFloat16ArraysClose(
            actualLogits, expectedLogits,
            label: "\(label): current logits")
    case (.none, .none):
        currentLogitEvidence = .absent
    default:
        Issue.record("\(label): current logits validity changed")
        currentLogitEvidence = .mismatchedPresence
    }
    return RealStateComparisonEvidence(label: label, currentLogits: currentLogitEvidence)
}

private func expectRealFloat16ArraysClose(
    _ actual: [Float16], _ expected: [Float16], label: String
) -> RealCurrentLogitComparison {
    #expect(actual.count == expected.count, "\(label): element count")
    guard actual.count == expected.count else { return .mismatchedPresence }
    let actualValues = actual.map(Float.init)
    let expectedValues = expected.map(Float.init)
    let actualFinite = actualValues.allSatisfy(\.isFinite)
    let expectedFinite = expectedValues.allSatisfy(\.isFinite)
    #expect(actualFinite, "\(label): actual logits contain a nonfinite value")
    #expect(expectedFinite, "\(label): reference logits contain a nonfinite value")
    guard actualFinite, expectedFinite else {
        return RealCurrentLogitComparison(
            actualFinite: actualFinite, expectedFinite: expectedFinite,
            actualMaximumAbsoluteValue: nil, expectedMaximumAbsoluteValue: nil,
            comparisonScale: nil, maximumAbsoluteDelta: nil)
    }
    let actualMaximum = actualValues.map { abs($0) }.max() ?? 0
    let expectedMaximum = expectedValues.map { abs($0) }.max() ?? 0
    let scale = max(actualMaximum, expectedMaximum)
    let maximumDelta = zip(actualValues, expectedValues)
        .map { abs($0 - $1) }.max() ?? 0
    let derivedMetricsFinite = actualMaximum.isFinite
        && expectedMaximum.isFinite
        && scale.isFinite
        && maximumDelta.isFinite
    #expect(derivedMetricsFinite, "\(label): derived logit metrics contain a nonfinite value")
    guard derivedMetricsFinite else {
        return RealCurrentLogitComparison(
            actualFinite: true, expectedFinite: true,
            actualMaximumAbsoluteValue: nil, expectedMaximumAbsoluteValue: nil,
            comparisonScale: nil, maximumAbsoluteDelta: nil)
    }
    let limit: Float = 2e-5 + 2e-5 * scale
    #expect(limit.isFinite, "\(label): comparison limit is nonfinite")
    guard limit.isFinite else {
        return RealCurrentLogitComparison(
            actualFinite: true, expectedFinite: true,
            actualMaximumAbsoluteValue: nil, expectedMaximumAbsoluteValue: nil,
            comparisonScale: nil, maximumAbsoluteDelta: nil)
    }
    #expect(maximumDelta <= limit,
            "\(label): max FP32 delta \(maximumDelta) > \(limit)")
    return RealCurrentLogitComparison(
        actualFinite: true, expectedFinite: true,
        actualMaximumAbsoluteValue: actualMaximum,
        expectedMaximumAbsoluteValue: expectedMaximum,
        comparisonScale: scale, maximumAbsoluteDelta: maximumDelta)
}

private func expectRealFloatArraysClose(
    _ actual: [Float], _ expected: [Float], label: String
) {
    #expect(actual.count == expected.count, "\(label): element count")
    guard actual.count == expected.count else { return }
    let actualFinite = actual.allSatisfy(\.isFinite)
    let expectedFinite = expected.allSatisfy(\.isFinite)
    #expect(actualFinite, "\(label): actual tensor contains a nonfinite value")
    #expect(expectedFinite, "\(label): reference tensor contains a nonfinite value")
    guard actualFinite, expectedFinite else { return }
    let scale = max(
        actual.map { abs($0) }.max() ?? 0,
        expected.map { abs($0) }.max() ?? 0)
    let maximum = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    let derivedMetricsFinite = scale.isFinite && maximum.isFinite
    #expect(derivedMetricsFinite, "\(label): derived tensor metrics contain a nonfinite value")
    guard derivedMetricsFinite else { return }
    let limit: Float = 2e-5 + 2e-5 * scale
    #expect(limit.isFinite, "\(label): comparison limit is nonfinite")
    guard limit.isFinite else { return }
    #expect(maximum <= limit, "\(label): max FP32 delta \(maximum) > \(limit)")
}

private actor LayerFailureGate {
    let failAfterLayer: Int
    private var armed = false
    private var didFail = false

    init(failAfterLayer: Int) {
        self.failAfterLayer = failAfterLayer
    }

    func arm() {
        armed = true
    }

    func check(_ layer: Int) throws {
        guard armed, !didFail, layer == failAfterLayer else { return }
        didFail = true
        throw InjectedStateFailure()
    }
}
