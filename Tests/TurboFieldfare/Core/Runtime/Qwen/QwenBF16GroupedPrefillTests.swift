import Darwin
import Foundation
import Metal
import Testing
import TurboFieldfareFormat
@testable import TurboFieldfare
@testable import TurboFieldfareOfficialQwenSource

/// Staged coverage for the original-BF16 grouped source prefill proposal.
/// This file deliberately uses only the synthetic two-layer fixture. It never
/// opens the pinned model or reads an original shard.
@Suite(.serialized) struct QwenBF16GroupedPrefillTests {
    private let prompt: [Int32] = [1, 2, 1, 2, 1]

    @Test func groupedPrefillMatchesLegacyAndIndependentOracle() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }

        let grouped = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let legacy = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: QwenOfficialSourceTransactionHooks(
                observeConsumedInput: { _, _, _, _ in }))

        let groupedLogits = try await grouped.prefill(
            tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        let legacyLogits = try await legacyRun(legacy, tokens: prompt)
        let expected = sourceOracle(harness.source, tokens: prompt)
        let expectedLast = try #require(expected.last)

        #expect(groupedLogits == legacyLogits,
                "grouped and token-major source logits must be bit-identical")
        assertWithinTolerance(
            groupedLogits, expectedLast.logits,
            label: "grouped final prompt logits")
        let groupedState = try await grouped.diagnosticSnapshot()
        let legacyState = try await legacy.diagnosticSnapshot()
        #expect(groupedState == legacyState,
                "grouped source state must match token-major state exactly")

        let diagnostics = try #require(await grouped.groupedPrefillDiagnostics())
        #expect(String(describing: diagnostics.mode).lowercased() == "grouped")
        #expect(diagnostics.tokenCount == prompt.count)
        #expect(diagnostics.completedLayers == QwenBF16TextRunnerFixture.layerCount)
    }

    @Test func defaultGroupedPrefillReportsDistinctRoutesAndBoundedMisses() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let runner = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)

        _ = try await runner.prefill(
            tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        let diagnostics = try #require(await runner.groupedPrefillDiagnostics())

        #expect(String(describing: diagnostics.mode).lowercased() == "grouped")
        #expect(diagnostics.mappedUniqueExperts == [0: 8, 1: 8],
                "the tiny fixture routes eight distinct experts in each layer")
        #expect(diagnostics.mappingMisses <= diagnostics.mappedUniqueExperts.values.reduce(0, +),
                "grouped reads must not reread an expert already loaded for this layer")
    }

    @Test func groupedPrefillSplitsMoreThan128ContributionsAndMatchesLegacy() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let longPrompt = (0..<17).map { Int32([1, 2, 1, 2, 1][$0 % 5]) }
        let grouped = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let legacy = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)

        let groupedLogits = try await grouped.prefill(
            tokenIDs: longPrompt, position: 0, onProgress: { _, _ in })
        let legacyLogits = try await legacyRun(legacy, tokens: longPrompt)
        let groupedFinalState = try await grouped.diagnosticSnapshot()
        let legacyFinalState = try await legacy.diagnosticSnapshot()
        #expect(groupedLogits == legacyLogits,
                "the 136-route grouped operation must preserve final logits across its command split")
        #expect(groupedFinalState == legacyFinalState,
            "command splitting must preserve the complete linear/KV state")

        let expected = try #require(sourceOracle(harness.source, tokens: longPrompt).last)
        assertWithinTolerance(groupedLogits, expected.logits,
                              label: "long grouped prompt logits")
        let diagnostics = try #require(await grouped.groupedPrefillDiagnostics())
        #expect(diagnostics.mode == .grouped)
        #expect(diagnostics.tokenCount == longPrompt.count)
        #expect(diagnostics.mappingMisses <= 16,
                "two layers with eight distinct routed experts must not reread payloads while splitting 136 contributions")
    }

    @Test func groupedPrefillDisabledSelectsTokenMajorWithoutChangingResult() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let state = try await makeConversationState(
            model: harness.model, context: harness.context,
            groupedPrefillEnabled: false, hooks: .none)
        let transaction = try await state.begin()
        try await state.prefill(prompt, transaction: transaction) { _, _ in }

        let diagnostics = try #require(await state.groupedPrefillDiagnostics())
        #expect(diagnostics.mode == .tokenMajor)
        #expect(diagnostics.tokenCount == prompt.count)
        let snapshot = try await state.diagnosticSnapshot()
        let expected = try #require(sourceOracle(harness.source, tokens: prompt).last)
        let actualLogits = try #require(snapshot.currentLogits)
        assertWithinTolerance(actualLogits, expected.logits,
                              label: "token-major off-mode final logits")
        #expect(snapshot.runner.position == prompt.count)
    }

    @Test func groupedProtectedReadFailureRollsBackAndRetryMatchesOracle() async throws {
        let failure = QwenBF16TransactionTestSwitch()
        let harness = try makeRunnerHarness(
            hooks: QwenOfficialSourceTransactionHooks(
                beforeProtectedExpertRead: { layer, _, _ in
                    if layer == 1, failure.consume() {
                        throw GroupedPrefillInjectedFailure.protectedRead
                    }
                }))
        defer { harness.source.remove() }
        let runner = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: harness.hooks)
        let before = try await runner.diagnosticSnapshot()
        failure.arm()

        let failed = await captureGroupedError {
            _ = try await runner.prefill(
                tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        }
        #expect(failed != nil)
        #expect(try await runner.diagnosticSnapshot() == before,
                "a grouped protected-read failure must restore all source state")
        let failedDiagnostics = try #require(await runner.groupedPrefillDiagnostics())
        #expect(String(describing: failedDiagnostics.mode).lowercased() == "grouped")

        let retried = try await runner.prefill(
            tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        let expected = try #require(sourceOracle(harness.source, tokens: prompt).last)
        assertWithinTolerance(retried, expected.logits, label: "grouped retry logits")
        #expect((try await runner.diagnosticSnapshot()).position == prompt.count)
    }

    @Test func groupedCancellationRollsBackAndRetryMatchesOracle() async throws {
        let gate = QwenBF16TransactionSyncGate()
        let firstRead = QwenBF16TransactionTestSwitch()
        let harness = try makeRunnerHarness(
            hooks: QwenOfficialSourceTransactionHooks(
                beforeProtectedExpertRead: { _, _, _ in
                    if firstRead.consume() { gate.suspend() }
                }))
        defer { harness.source.remove() }
        let runner = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: harness.hooks)
        let before = try await runner.diagnosticSnapshot()
        firstRead.arm()
        let operation = Task {
            try await runner.prefill(
                tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        }
        await gate.waitUntilEntered()
        operation.cancel()
        gate.release()
        let cancelled = await captureTaskError(operation)
        #expect(cancelled != nil)
        #expect(try await runner.diagnosticSnapshot() == before,
                "cancellation must not publish a partial grouped prompt")
        let cancelledDiagnostics = try #require(await runner.groupedPrefillDiagnostics())
        #expect(String(describing: cancelledDiagnostics.mode).lowercased() == "grouped")

        let retried = try await runner.prefill(
            tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        let expected = try #require(sourceOracle(harness.source, tokens: prompt).last)
        assertWithinTolerance(retried, expected.logits, label: "grouped cancellation retry")
    }

    @Test func preparedImagePrefillPreservesFeatureRowsAndMRoPEPositions() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let grouped = try await makeConversationState(
            model: harness.model, context: harness.context, hooks: .none)
        let groupedInputRecorder = GroupedConsumedInputRecorder()
        let groupedCapture = QwenSourceGroupedPrefillCapture(
            observeConsumedInput: { position, token, row, mrope in
                groupedInputRecorder.record(position: position, token: token,
                                             row: row, mrope: mrope)
            })
        let legacyInputRecorder = GroupedConsumedInputRecorder()
        let legacyHooks = QwenOfficialSourceTransactionHooks(
            observeConsumedInput: { position, token, row, mrope in
                legacyInputRecorder.record(position: position, token: token,
                                           row: row, mrope: mrope)
            })
        let legacy = try await makeConversationState(
            model: harness.model, context: harness.context, hooks: legacyHooks)
        let prepared = try makePreparedImage(context: harness.context)

        let groupedTransaction = try await grouped.begin()
        try await grouped.prefillPrepared(
            prepared, transaction: groupedTransaction,
            groupedCapture: groupedCapture) { _, _ in }
        let groupedSnapshot = try await grouped.diagnosticSnapshot()
        let groupedDiagnostics = try #require(await grouped.groupedPrefillDiagnostics())
        #expect(String(describing: groupedDiagnostics.mode).lowercased() == "grouped")

        let legacyTransaction = try await legacy.begin()
        try await legacy.prefillPrepared(prepared, transaction: legacyTransaction) { _, _ in }
        let legacySnapshot = try await legacy.diagnosticSnapshot()
        let legacyDiagnostics = try #require(await legacy.groupedPrefillDiagnostics())
        #expect(String(describing: legacyDiagnostics.mode).lowercased() == "tokenmajor")
        #expect(groupedSnapshot.currentLogits == legacySnapshot.currentLogits,
                "grouped prepared-image logits must preserve image rows and positions")
        #expect(groupedSnapshot.runner == legacySnapshot.runner)

        let groupedImageObservation = try #require(
            groupedInputRecorder.snapshot().first(where: { $0.position == 1 }))
        #expect(groupedImageObservation.token == 3)
        #expect(groupedImageObservation.row == [0.125, 0.25, 0.375, 0.5,
                                                 0.625, 0.75, 0.875, 1])
        #expect(groupedImageObservation.mrope == prepared.positions[1])
        let imageObservation = try #require(
            legacyInputRecorder.snapshot().first(where: { $0.position == 1 }))
        #expect(imageObservation.token == 3)
        #expect(imageObservation.row == [0.125, 0.25, 0.375, 0.5,
                                         0.625, 0.75, 0.875, 1])
        #expect(imageObservation.mrope == prepared.positions[1])
    }

    @Test func groupedPrefillRejectsWrongStartOffsetWithoutMutation() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let runner = try QwenOfficialSourceRunner(
            model: harness.model, maxContext: 32,
            expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let before = try await runner.diagnosticSnapshot()

        let error = await captureGroupedError {
            _ = try await runner.prefill(
                tokenIDs: prompt, position: 1, onProgress: { _, _ in })
        }
        #expect(error != nil, "a nonzero start offset must be rejected at position zero")
        #expect(try await runner.diagnosticSnapshot() == before)
    }
    @Test func groupedLinearTrialPreservesRoutesStateAndRemaindersAtBothOffsets() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let tail = (0..<17).map { Int32($0.isMultiple(of: 2) ? 1 : 2) }
        for start in [0, 3] {
            for groupedEnabled in [false, true] {
                for gpuPreparationEnabled in [false, true] {
                    let baseRoutes = GroupedLinearRouteRecorder()
                    let trialRoutes = GroupedLinearRouteRecorder()
                    let baseline = try makeGroupedLinearTrialRunner(harness, enabled: false,
                        gpuPreparationEnabled: false,
                        hooks: QwenOfficialSourceTransactionHooks(observeRoute: baseRoutes.record))
                    let trial = try makeGroupedLinearTrialRunner(harness,
                        enabled: groupedEnabled,
                        gpuPreparationEnabled: gpuPreparationEnabled,
                        hooks: QwenOfficialSourceTransactionHooks(observeRoute: trialRoutes.record))
                    if start > 0 {
                        let prefix = Array(tail.prefix(start))
                        _ = try await legacyRun(baseline, tokens: prefix)
                        _ = try await legacyRun(trial, tokens: prefix)
                    }
                    let expected = try await baseline.prefill(
                        tokenIDs: tail, position: start, onProgress: { _, _ in })
                    let actual = try await trial.prefill(
                        tokenIDs: tail, position: start, onProgress: { _, _ in })
                    #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
                    let trialState = try await trial.diagnosticSnapshot()
                    let baselineState = try await baseline.diagnosticSnapshot()
                    #expect(trialState == baselineState)
                    #expect(trialRoutes.snapshot() == baseRoutes.snapshot())
                    let diagnostics = try #require(await trial.groupedPrefillDiagnostics())
                    if groupedEnabled {
                        #expect(diagnostics.mode == .grouped)
                        #expect(diagnostics.groupedLinearBatchSizes[1]
                            == (start == 0 ? [1, 15, 1] : [16, 1]))
                        #expect(diagnostics.groupedLinearBatchSizes[0] == nil,
                                "Full attention stays per-token")
                    } else {
                        #expect(diagnostics.mode == .grouped)
                        #expect(diagnostics.groupedLinearBatchSizes.isEmpty)
                    }
                    let nextExpected = try await baseline.produce(
                        token: 1, position: start + tail.count)
                    let nextActual = try await trial.produce(
                        token: 1, position: start + tail.count)
                    #expect(nextActual.map(\.bitPattern) == nextExpected.map(\.bitPattern),
                            "Single-token decode must continue from identical cached state")
                }
            }
        }
    }

    @Test func groupedLinearTrialProtectedReadFailureRestoresAndRetryMatchesBaseline() async throws {
        let failure = QwenBF16TransactionTestSwitch()
        let harness = try makeRunnerHarness(hooks: QwenOfficialSourceTransactionHooks(
            beforeProtectedExpertRead: { layer, _, _ in
                if layer == 1, failure.consume() { throw GroupedPrefillInjectedFailure.protectedRead }
            }))
        defer { harness.source.remove() }
        let trial = try makeGroupedLinearTrialRunner(harness, enabled: true, hooks: harness.hooks)
        let baseline = try makeGroupedLinearTrialRunner(harness, enabled: false)
        let before = try await trial.diagnosticSnapshot()
        let tail = (0..<17).map { Int32($0.isMultiple(of: 2) ? 1 : 2) }
        failure.arm()
        let error = await captureGroupedError {
            _ = try await trial.prefill(tokenIDs: tail, position: 0, onProgress: { _, _ in })
        }
        #expect(error != nil)
        #expect(try await trial.diagnosticSnapshot() == before)
        let failed = try #require(await trial.groupedPrefillDiagnostics())
        #expect(failed.groupedLinearBatchSizes[1] == [1, 15, 1],
                "Failure must occur after the layer-1 linear mixer batches committed")
        #expect(failed.groupedLinearBatchSizes[0] == nil,
                "Layer 0 is full attention in the two-layer fixture")
        let retried = try await trial.prefill(tokenIDs: tail, position: 0, onProgress: { _, _ in })
        let expected = try await baseline.prefill(tokenIDs: tail, position: 0, onProgress: { _, _ in })
        #expect(retried.map(\.bitPattern) == expected.map(\.bitPattern))
        let trialState = try await trial.diagnosticSnapshot()
        let baselineState = try await baseline.diagnosticSnapshot()
        #expect(trialState == baselineState)
        let oracle = try #require(sourceOracle(harness.source, tokens: tail).last)
        assertWithinTolerance(retried, oracle.logits, label: "grouped linear retry oracle")
    }

    @Test func groupedLinearCheckpointRebuildMatchesTokenMajorState() async throws {
        let keys = ["TURBO_QWEN_GROUPED_LINEAR_PREFILL", "TURBO_QWEN_GPU_LINEAR_PREPARATION"]
        let previous = keys.map { ProcessInfo.processInfo.environment[$0] }
        for key in keys { setenv(key, "1", 1) }
        defer {
            for (key, value) in zip(keys, previous) {
                if let value { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let grouped = try await makeConversationState(
            model: harness.model, context: harness.context,
            groupedPrefillEnabled: true, hooks: .none)
        let tokenMajor = try await makeConversationState(
            model: harness.model, context: harness.context,
            groupedPrefillEnabled: false, hooks: .none)
        let groupedTransaction = try await grouped.begin()
        let tokenMajorTransaction = try await tokenMajor.begin()
        let replacement = (0..<17).map { Int32($0.isMultiple(of: 2) ? 1 : 2) }
        try await grouped.rebuildCheckpoint(
            retaining: replacement, transaction: groupedTransaction) { _, _ in }
        try await tokenMajor.rebuildCheckpoint(
            retaining: replacement, transaction: tokenMajorTransaction) { _, _ in }

        let groupedDiagnostics = try #require(await grouped.groupedPrefillDiagnostics())
        #expect(groupedDiagnostics.mode == .grouped)
        #expect(groupedDiagnostics.groupedLinearBatchSizes[1] == [1, 15, 1])
        let tokenMajorDiagnostics = try #require(await tokenMajor.groupedPrefillDiagnostics())
        #expect(tokenMajorDiagnostics.mode == .tokenMajor)
        #expect(tokenMajorDiagnostics.groupedLinearBatchSizes.isEmpty)

        let groupedWorking = try await grouped.diagnosticSnapshot()
        let tokenMajorWorking = try await tokenMajor.diagnosticSnapshot()
        #expect(groupedWorking.runner == tokenMajorWorking.runner,
                "grouped checkpoint rebuild must preserve linear and full-attention state")
        #expect(groupedWorking.currentLogits?.map(\.bitPattern)
            == tokenMajorWorking.currentLogits?.map(\.bitPattern),
                "grouped checkpoint rebuild logits must be bit-identical")

        _ = try await grouped.commit(transaction: groupedTransaction)
        _ = try await tokenMajor.commit(transaction: tokenMajorTransaction)
        let groupedCommitted = try await grouped.diagnosticSnapshot()
        let tokenMajorCommitted = try await tokenMajor.diagnosticSnapshot()
        #expect(groupedCommitted.runner == tokenMajorCommitted.runner,
                "committed grouped checkpoint state must match token-major state")
        #expect(groupedCommitted.currentLogits?.map(\.bitPattern)
            == tokenMajorCommitted.currentLogits?.map(\.bitPattern))
    }

    @Test func groupedLinearTrialActivationObserverRetainsTokenMajorFallback() async throws {
        let harness = try makeRunnerHarness()
        defer { harness.source.remove() }
        let trial = try makeGroupedLinearTrialRunner(harness, enabled: true,
            hooks: QwenOfficialSourceTransactionHooks(observeActivation: { _, _, _, _ in }))
        let baseline = try makeGroupedLinearTrialRunner(harness, enabled: false)
        let actual = try await trial.prefill(tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        let expected = try await baseline.prefill(tokenIDs: prompt, position: 0, onProgress: { _, _ in })
        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
        let diagnostics = try #require(await trial.groupedPrefillDiagnostics())
        #expect(diagnostics.mode == .tokenMajor)
        #expect(diagnostics.groupedLinearBatchSizes.isEmpty)
    }

}

private struct GroupedRunnerHarness {
    let source: QwenBF16TextRunnerFixture.Source
    let context: MetalContext
    let model: QwenOfficialSourceModel
    let hooks: QwenOfficialSourceTransactionHooks
}

private func makeRunnerHarness(
    hooks: QwenOfficialSourceTransactionHooks = .none
) throws -> GroupedRunnerHarness {
    let source = try QwenBF16TextRunnerFixture.make()
    let context = try MetalContext()
    let model = try QwenOfficialSourceModel.loadSyntheticFixture(
        registrationURL: source.registrationURL, context: context,
        residencyBudgetBytes: source.totalResidencyBudget(
            expertSlotCount: 2 * QwenBF16TextRunnerFixture.topK))
    return GroupedRunnerHarness(source: source, context: context, model: model, hooks: hooks)
}

private func makeConversationState(
    model: QwenOfficialSourceModel,
    context: MetalContext,
    groupedPrefillEnabled: Bool = true,
    hooks: QwenOfficialSourceTransactionHooks
) async throws -> QwenOfficialSourceConversationState {
    try await QwenOfficialSourceConversationState(
        model: model, context: context, maxContext: 32,
        expertSlotCount: QwenBF16TextRunnerFixture.topK,
        groupedPrefillEnabled: groupedPrefillEnabled, hooks: hooks)
}

private func sourceOracle(
    _ source: QwenBF16TextRunnerFixture.Source,
    tokens: [Int32]
) -> [QwenBF16TextRunnerFixture.TokenResult] {
    var oracle = source.oracle()
    return oracle.run(tokens: tokens)
}

private func legacyRun(
    _ runner: QwenOfficialSourceRunner,
    tokens: [Int32]
) async throws -> [Float] {
    var logits: [Float] = []
    for (position, token) in tokens.enumerated() {
        logits = try await runner.produce(token: token, position: position)
    }
    return logits
}

private func assertWithinTolerance(
    _ actual: [Float], _ expected: [Float], label: String
) {
    let matches = actual.count == expected.count
        && zip(actual, expected).allSatisfy { actual, expected in
            actual.isFinite && expected.isFinite
                && abs(actual - expected) <= max(
                    QwenBF16TextRunnerFixture.absoluteTolerance,
                    QwenBF16TextRunnerFixture.relativeTolerance * abs(expected))
        }
    #expect(matches, "\(label)")
}

private func captureGroupedError(
    _ operation: () async throws -> Void
) async -> Error? {
    do {
        try await operation()
        return nil
    } catch {
        return error
    }
}

private func captureTaskError<Value>(_ task: Task<Value, Error>) async -> Error? {
    do {
        _ = try await task.value
        return nil
    } catch {
        return error
    }
}

private enum GroupedPrefillInjectedFailure: Error {
    case protectedRead
}

private final class GroupedConsumedInputRecorder: @unchecked Sendable {
    struct Record: Sendable {
        let position: Int
        let token: Int32
        let row: [Float]?
        let mrope: QwenMRoPEPosition?
    }

    private let lock = NSLock()
    private var records: [Record] = []

    func record(position: Int, token: Int32, row: [Float]?, mrope: QwenMRoPEPosition?) {
        lock.lock()
        records.append(Record(position: position, token: token, row: row, mrope: mrope))
        lock.unlock()
    }

    func snapshot() -> [Record] {
        lock.lock()
        defer { lock.unlock() }
        return records
    }
}

private func makePreparedImage(context: MetalContext) throws -> QwenPreparedPrefill {
    let profile = GTurboQwenVisionProcessorProfileV2(
        processorClass: "Qwen3VLProcessor",
        imageProcessorType: "Qwen2VLImageProcessorFast",
        patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
    let grid = try QwenVisionGrid(temporal: 1, height: 2, width: 2)
    let position = try QwenMRoPEPosition(temporal: 0, height: 0, width: 0)
    let owner = try QwenRetainedFeatureOwner(
        device: context.device,
        features: [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
        positions: [position],
        imageDigest: String(repeating: "a", count: 64),
        processorDigest: String(repeating: "b", count: 64),
        profile: profile, grid: grid, hiddenSize: QwenBF16TextRunnerFixture.hiddenSize)
    return try QwenPreparedPrefill(
        tokenIDs: [1, 3, 2],
        featureOverrides: [QwenPreparedFeatureOverride(tokenRange: 1..<2, owner: owner)],
        positions: [
            try QwenMRoPEPosition(temporal: 0, height: 0, width: 0),
            position,
            try QwenMRoPEPosition(temporal: 2, height: 2, width: 2),
        ], textRoPEDelta: 0)
}

// Changes environment only around synchronous runner construction. Scripts/test.sh
// and this serialized suite must remain nonparallel. Runner captures the flag once.
private func makeGroupedLinearTrialRunner(_ harness: GroupedRunnerHarness, enabled: Bool,
    gpuPreparationEnabled: Bool = false,
    hooks: QwenOfficialSourceTransactionHooks = .none) throws -> QwenOfficialSourceRunner {
    let groupedKey = "TURBO_QWEN_GROUPED_LINEAR_PREFILL"
    let gpuKey = "TURBO_QWEN_GPU_LINEAR_PREPARATION"
    let previousGrouped = ProcessInfo.processInfo.environment[groupedKey]
    let previousGPU = ProcessInfo.processInfo.environment[gpuKey]
    setenv(groupedKey, enabled ? "1" : "0", 1)
    setenv(gpuKey, gpuPreparationEnabled ? "1" : "0", 1)
    defer {
        if let previousGrouped { setenv(groupedKey, previousGrouped, 1) }
        else { unsetenv(groupedKey) }
        if let previousGPU { setenv(gpuKey, previousGPU, 1) }
        else { unsetenv(gpuKey) }
    }
    return try QwenOfficialSourceRunner(model: harness.model, maxContext: 32,
        expertSlotCount: QwenBF16TextRunnerFixture.topK, hooks: hooks)
}

private final class GroupedLinearRouteRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var rows: [[UInt32]] = []
    func record(_ position: Int, _ layer: Int, _ logits: [Float], _ ids: [Int],
                _ weights: [Float], _ cutoff: Float) {
        let row = [UInt32(position), UInt32(layer)] + logits.map(\.bitPattern)
            + ids.map { UInt32($0) } + weights.map(\.bitPattern) + [cutoff.bitPattern]
        lock.lock(); defer { lock.unlock() }
        rows.append(row)
    }
    func snapshot() -> [[UInt32]] { lock.lock(); defer { lock.unlock() }; return rows }
}
