import Foundation
import Metal
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized)
struct QwenOfficialSourceCheckpointTests {
    @Test
    func successfulTextReplacementMatchesFreshStateAndSubsequentProducedStep() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let freshModel = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let replaced = try await makeCheckpointState(model: model, context: context)
        let fresh = try await makeCheckpointState(model: freshModel, context: context)

        try await commitText([1, 2], on: replaced)
        let replacement = try await replaced.begin()
        try await replaced.rebuildCheckpoint(
            retaining: [2, 1], transaction: replacement, onProgress: { _, _ in })
        _ = try await replaced.commit(transaction: replacement)
        try await commitText([2, 1], on: fresh)
        let replacedTextSnapshot = try await replaced.diagnosticSnapshot()
        let freshTextSnapshot = try await fresh.diagnosticSnapshot()
        #expect(replacedTextSnapshot == freshTextSnapshot)

        try await commitProducedStep(on: replaced)
        try await commitProducedStep(on: fresh)
        let replacedTextStepSnapshot = try await replaced.diagnosticSnapshot()
        let freshTextStepSnapshot = try await fresh.diagnosticSnapshot()
        #expect(replacedTextStepSnapshot == freshTextStepSnapshot)
    }

    @Test
    func successfulPreparedImageReplacementMatchesFreshStateAndSubsequentProducedStep() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let freshModel = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let replaced = try await makeCheckpointState(model: model, context: context)
        let fresh = try await makeCheckpointState(model: freshModel, context: context)
        let replacedPrepared = try makeCheckpointPrefill(
            owner: try makeCheckpointFeatureOwner(context: context))
        let freshPrepared = try makeCheckpointPrefill(
            owner: try makeCheckpointFeatureOwner(context: context))

        try await commitText([1, 2], on: replaced)
        let replacement = try await replaced.begin()
        try await replaced.rebuildPreparedCheckpoint(
            replacedPrepared, transaction: replacement, onProgress: { _, _ in })
        _ = try await replaced.commit(transaction: replacement)

        let initial = try await fresh.begin()
        try await fresh.prefillPrepared(
            freshPrepared, transaction: initial, onProgress: { _, _ in })
        _ = try await fresh.commit(transaction: initial)
        let replacedImageSnapshot = try await replaced.diagnosticSnapshot()
        let freshImageSnapshot = try await fresh.diagnosticSnapshot()
        #expect(replacedImageSnapshot == freshImageSnapshot)

        try await commitProducedStep(on: replaced)
        try await commitProducedStep(on: fresh)
        let replacedImageStepSnapshot = try await replaced.diagnosticSnapshot()
        let freshImageStepSnapshot = try await fresh.diagnosticSnapshot()
        #expect(replacedImageStepSnapshot == freshImageStepSnapshot)
    }

    @Test
    func cancellationAfterTextReplacementOverwriteRestoresExactPriorSnapshot() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let state = try await makeCheckpointState(model: model, context: context)
        try await commitText([1, 2], on: state)
        let baseline = try await state.diagnosticSnapshot()
        let owner = try makeCheckpointFeatureOwner(context: context)
        let prepared = try makeCheckpointPrefill(owner: owner)

        let dryText = try await state.begin()
        try await state.rebuildCheckpoint(
            retaining: [2, 1, 3], transaction: dryText, onProgress: { _, _ in })
        try await expectReplacementVisible(state, baseline: baseline)
        try await state.rollback(transaction: dryText)
        #expect(try await state.diagnosticSnapshot() == baseline)

        let dryImage = try await state.begin()
        try await state.rebuildPreparedCheckpoint(
            prepared, transaction: dryImage, onProgress: { _, _ in })
        try await expectReplacementVisible(state, baseline: baseline)
        try await state.rollback(transaction: dryImage)
        #expect(try await state.diagnosticSnapshot() == baseline)

        let replacement = try await state.begin()
        try await state.rebuildCheckpoint(
            retaining: [2, 1, 3], transaction: replacement, onProgress: { _, _ in })
        try await expectReplacementVisible(state, baseline: baseline)

        await #expect(throws: CancellationError.self) {
            _ = try await state.commit(
                transaction: replacement, shouldStop: { true })
        }
        try await state.rollback(transaction: replacement)
        #expect(try await state.diagnosticSnapshot() == baseline)
    }

    @Test
    func textAndPreparedImageReplacementFailuresRestoreExactDiagnosticSnapshot() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL,
            context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let failure = SourceCheckpointCommitFailureGate()
        let state = try await QwenOfficialSourceConversationState(
            model: model, context: context, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(beforeTurnCommit: {
                if failure.consume() { throw SourceCheckpointCommitFailure.injected }
            }))

        let initial = try await state.begin()
        try await state.prefill([1, 2], transaction: initial, onProgress: { _, _ in })
        _ = try await state.commit(transaction: initial)
        let baseline = try await state.diagnosticSnapshot()

        let textReplacement = try await state.begin()
        try await state.rebuildCheckpoint(
            retaining: [2, 1], transaction: textReplacement, onProgress: { _, _ in })
        failure.arm()
        await #expect(throws: SourceCheckpointCommitFailure.self) {
            _ = try await state.commit(transaction: textReplacement)
        }
        try await state.rollback(transaction: textReplacement)
        #expect(try await state.diagnosticSnapshot() == baseline)

        let prepared = try makeCheckpointPrefill(
            owner: try makeCheckpointFeatureOwner(context: context))
        let imageReplacement = try await state.begin()
        try await state.rebuildPreparedCheckpoint(
            prepared, transaction: imageReplacement, onProgress: { _, _ in })
        failure.arm()
        await #expect(throws: SourceCheckpointCommitFailure.self) {
            _ = try await state.commit(transaction: imageReplacement)
        }
        try await state.rollback(transaction: imageReplacement)
        #expect(try await state.diagnosticSnapshot() == baseline)
    }

    @Test
    func gpuCompletionFailureDuringTextAndImageReplacementRestoresCommittedBytesAndRetries() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let freshModel = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let failure = SourceCheckpointReplacementGPUFailureGate()
        let state = try await QwenOfficialSourceConversationState(
            model: model, context: context, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(afterActualGPUCompletion: { stage in
                if stage == "source.moe", failure.consume() {
                    throw SourceCheckpointReplacementGPUFailure.injected
                }
            }))
        let fresh = try await makeCheckpointState(
            model: freshModel, context: context,
            hooks: QwenOfficialSourceTransactionHooks(afterActualGPUCompletion: { _ in }))
        let imageModel = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let freshImageModel = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let imageState = try await makeCheckpointState(
            model: imageModel, context: context,
            hooks: QwenOfficialSourceTransactionHooks(afterActualGPUCompletion: { stage in
                if stage == "source.moe", failure.consume() {
                    throw SourceCheckpointReplacementGPUFailure.injected
                }
            }))
        let freshImage = try await makeCheckpointState(
            model: freshImageModel, context: context,
            hooks: QwenOfficialSourceTransactionHooks(afterActualGPUCompletion: { _ in }))
        // Two source MoE completions per token. Failure at the fourth
        // completion follows the image override at replacement index one.
        let replacementTokens: [Int32] = [2, 1, 3]

        try await commitText([1, 2], on: state)
        try await commitText([1, 2], on: fresh)
        let textBaseline = try await state.diagnosticSnapshot()

        let failedText = try await state.begin()
        failure.arm(completions: 4)
        await #expect(throws: SourceCheckpointReplacementGPUFailure.self) {
            try await state.rebuildCheckpoint(
                retaining: replacementTokens, transaction: failedText,
                onProgress: { _, _ in })
        }
        #expect(failure.wasConsumed, "text replacement must reach its second-token GPU completion")
        try await state.rollback(transaction: failedText)
        try await expectCommittedRunnerEqual(state, textBaseline)

        let retriedText = try await state.begin()
        try await state.rebuildCheckpoint(
            retaining: replacementTokens, transaction: retriedText,
            onProgress: { _, _ in })
        _ = try await state.commit(transaction: retriedText)
        let freshText = try await fresh.begin()
        try await fresh.rebuildCheckpoint(
            retaining: replacementTokens, transaction: freshText,
            onProgress: { _, _ in })
        _ = try await fresh.commit(transaction: freshText)
        let expectedFreshText = try await fresh.diagnosticSnapshot()
        try await expectCommittedRunnerEqual(state, expectedFreshText)

        let owner = try makeCheckpointFeatureOwner(context: context)
        let imageReplacementTokens: [Int32] = [
            2, Int32(imageModel.visionArchitecture.imageTokenID), 1,
        ]
        let prepared = try makeCheckpointPrefill(owner: owner, tokenIDs: imageReplacementTokens)
        try await commitText([1, 2], on: imageState)
        try await commitText([1, 2], on: freshImage)
        let imageBaseline = try await imageState.diagnosticSnapshot()
        let failedImage = try await imageState.begin()
        failure.arm(completions: 4)
        await #expect(throws: SourceCheckpointReplacementGPUFailure.self) {
            try await imageState.rebuildPreparedCheckpoint(
                prepared, transaction: failedImage, onProgress: { _, _ in })
        }
        #expect(failure.wasConsumed, "image replacement must consume its feature override before failure")
        try await imageState.rollback(transaction: failedImage)
        try await expectCommittedRunnerEqual(imageState, imageBaseline)

        let retriedImage = try await imageState.begin()
        try await imageState.rebuildPreparedCheckpoint(
            prepared, transaction: retriedImage, onProgress: { _, _ in })
        _ = try await imageState.commit(transaction: retriedImage)

        let freshImageTransaction = try await freshImage.begin()
        try await freshImage.rebuildPreparedCheckpoint(
            prepared, transaction: freshImageTransaction, onProgress: { _, _ in })
        _ = try await freshImage.commit(transaction: freshImageTransaction)
        let expectedFreshImage = try await freshImage.diagnosticSnapshot()
        try await expectCommittedRunnerEqual(imageState, expectedFreshImage)
    }

    @Test
    func taskCancellationAfterReplacementFullCommitRestoresAndRetriesAgainstFreshState() async throws {
        let source = try QwenBF16TextRunnerFixture.make()
        defer { source.remove() }
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let freshModel = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: source.registrationURL, context: context,
            residencyBudgetBytes: source.totalResidencyBudget())
        let gate = SourceCheckpointAsyncGate()
        let pause = SourceCheckpointCommitFailureGate()
        let state = try await QwenOfficialSourceConversationState(
            model: model, context: context, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(
                afterActualGPUCompletion: { stage in
                    if stage == "full.commit", pause.consume() {
                        await gate.holdUntilReleased()
                    }
                }))
        let fresh = try await makeCheckpointState(model: freshModel, context: context)

        try await commitText([1, 2], on: state)
        try await commitText([1, 2], on: fresh)
        let baseline = try await state.diagnosticSnapshot()
        let replacement = try await state.begin()
        pause.arm()
        let operation = Task {
            try await state.rebuildCheckpoint(
                retaining: [2, 1, 3], transaction: replacement, onProgress: { _, _ in })
        }
        await gate.waitUntilEntered()
        operation.cancel()
        await gate.release()
        await #expect(throws: CancellationError.self) {
            try await operation.value
        }

        try await state.rollback(transaction: replacement)
        try await expectCommittedRunnerEqual(state, baseline)

        let retried = try await state.begin()
        try await state.rebuildCheckpoint(
            retaining: [2, 1, 3], transaction: retried, onProgress: { _, _ in })
        _ = try await state.commit(transaction: retried)
        let freshTransaction = try await fresh.begin()
        try await fresh.rebuildCheckpoint(
            retaining: [2, 1, 3], transaction: freshTransaction, onProgress: { _, _ in })
        _ = try await fresh.commit(transaction: freshTransaction)
        let expected = try await fresh.diagnosticSnapshot()
        try await expectCommittedRunnerEqual(state, expected)
    }
}

private func expectReplacementVisible(
    _ state: QwenOfficialSourceConversationState,
    baseline: QwenOfficialSourceConversationDiagnosticSnapshot
) async throws {
    let preview = try await state.diagnosticSnapshot()
    #expect(preview.runner.position != baseline.runner.position
        || preview.runner.committedKeys != baseline.runner.committedKeys
        || preview.runner.committedValues != baseline.runner.committedValues)
}

private func expectCommittedRunnerEqual(
    _ state: QwenOfficialSourceConversationState,
    _ expected: QwenOfficialSourceConversationDiagnosticSnapshot
) async throws {
    let actual = try await state.diagnosticSnapshot()
    #expect(actual.retainedTokenIDs == expected.retainedTokenIDs)
    #expect(actual.consumedTokenIDs == expected.consumedTokenIDs)
    #expect(actual.pendingAcceptedToken == expected.pendingAcceptedToken)
    #expect(actual.currentLogits == expected.currentLogits)
    #expect(actual.runner.linear == expected.runner.linear)
    #expect(actual.runner.position == expected.runner.position)
    #expect(actual.runner.committedKeys == expected.runner.committedKeys)
    #expect(actual.runner.committedValues == expected.runner.committedValues)
}

private func makeCheckpointState(
    model: QwenOfficialSourceModel,
    context: MetalContext,
    hooks: QwenOfficialSourceTransactionHooks = .none
) async throws -> QwenOfficialSourceConversationState {
    try await QwenOfficialSourceConversationState(
        model: model, context: context, maxContext: 32,
        expertSlotCount: QwenBF16TextRunnerFixture.topK, hooks: hooks)
}

private func commitText(
    _ tokens: [Int32], on state: QwenOfficialSourceConversationState
) async throws {
    let transaction = try await state.begin()
    try await state.prefill(tokens, transaction: transaction, onProgress: { _, _ in })
    _ = try await state.commit(transaction: transaction)
}

private func commitProducedStep(
    on state: QwenOfficialSourceConversationState
) async throws {
    let transaction = try await state.begin()
    let input = try await state.prepareSampling(transaction: transaction)
    #expect(input.logits.count == QwenBF16TextRunnerFixture.vocabularySize)
    guard let selected = input.logits.enumerated().max(by: {
        $0.element < $1.element
    })?.offset else { throw SourceCheckpointTestFailure.emptyLogits }
    try await state.advance(Int32(selected), transaction: transaction)
    _ = try await state.prepareSampling(transaction: transaction)
    _ = try await state.commit(transaction: transaction)
}

private enum SourceCheckpointTestFailure: Error {
    case emptyLogits
}

private func makeCheckpointPrefill(
    owner: QwenRetainedFeatureOwner,
    tokenIDs: [Int32] = [1, 3, 2]
) throws -> QwenPreparedPrefill {
    precondition(tokenIDs.count == 3)
    let position = try QwenMRoPEPosition(temporal: 1, height: 1, width: 1)
    return try QwenPreparedPrefill(
        tokenIDs: tokenIDs,
        featureOverrides: [QwenPreparedFeatureOverride(tokenRange: 1..<2, owner: owner)],
        positions: [
            try QwenMRoPEPosition(temporal: 0, height: 0, width: 0),
            position,
            try QwenMRoPEPosition(temporal: 2, height: 2, width: 2),
        ], textRoPEDelta: 0)
}

private enum SourceCheckpointCommitFailure: Error {
    case injected
}

private enum SourceCheckpointReplacementGPUFailure: Error {
    case injected
}

private final class SourceCheckpointCommitFailureGate: @unchecked Sendable {
    private let lock = NSLock()
    private var armed = false

    func arm() {
        lock.lock()
        armed = true
        lock.unlock()
    }

    func consume() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard armed else { return false }
        armed = false
        return true
    }
}

private final class SourceCheckpointReplacementGPUFailureGate: @unchecked Sendable {
    private let lock = NSLock()
    private var remainingCompletions = 0
    private var consumed = false

    func arm(completions: Int) {
        lock.lock()
        remainingCompletions = completions
        consumed = false
        lock.unlock()
    }

    var wasConsumed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return consumed
    }

    func consume() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard remainingCompletions > 0 else { return false }
        remainingCompletions -= 1
        guard remainingCompletions == 0 else { return false }
        consumed = true
        return true
    }
}

private actor SourceCheckpointAsyncGate {
    private var entered = false
    private var released = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func holdUntilReleased() async {
        entered = true
        let waiters = enteredWaiters
        enteredWaiters.removeAll(keepingCapacity: false)
        for waiter in waiters { waiter.resume() }
        guard !released else { return }
        await withCheckedContinuation { continuation in
            releaseWaiters.append(continuation)
        }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll(keepingCapacity: false)
        for waiter in waiters { waiter.resume() }
    }
}

private func makeCheckpointFeatureOwner(
    context: MetalContext
) throws -> QwenRetainedFeatureOwner {
    let profile = GTurboQwenVisionProcessorProfileV2(
        processorClass: "synthetic",
        imageProcessorType: "synthetic",
        patchSize: 1, temporalPatchSize: 1, spatialMergeSize: 1)
    let grid = try QwenVisionGrid(temporal: 1, height: 1, width: 1)
    let position = try QwenMRoPEPosition(temporal: 1, height: 1, width: 1)
    return try QwenRetainedFeatureOwner(
        device: context.device,
        features: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
        positions: [position],
        imageDigest: String(repeating: "a", count: 64),
        processorDigest: String(repeating: "b", count: 64),
        profile: profile,
        grid: grid,
        hiddenSize: QwenBF16TextRunnerFixture.hiddenSize)
}
