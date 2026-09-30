import Foundation
import Metal
import Testing
@testable import TurboFieldfare
@testable import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

@Suite(.serialized) struct QwenOfficialSourceVisionTests {
    @Test func protectedSourceGroupsMatchIndependentNamedBF16Ranges() throws {
        let harness = try makeSourceVisionHarness()
        defer { harness.remove() }

        #expect(harness.model.sourceIdentity == nil,
                "the synthetic registration must remain untrusted")
        #expect(Set(try FileManager.default.contentsOfDirectory(
            atPath: harness.companionURL.path)) == Set(["manifest.json"]),
                "the source companion must contain metadata only")

        for fixtureGroup in QwenOfficialSourceVisionFixture.Group.allCases {
            let group: QwenVisionWeightGroup
            switch fixtureGroup {
            case .patchAndPosition: group = .patchAndPosition
            case .block0: group = .block(0)
            case .merger: group = .merger
            }
            let lease = try harness.store.mapGroup(group, device: harness.context.device)
            #expect(lease.group == group)

            // Independent packed-buffer layout: lexicographic tensor names,
            // 256-byte start alignment, and no expectation for padding bytes.
            var end = 0
            for name in fixtureGroup.tensorNames.sorted() {
                let expectedOffset = (end + 255) & ~255
                let expectedBytes = try harness.fixture.expectedTensorBytes(name)
                let actualOffset = try lease.offset(of: name)
                #expect(actualOffset == expectedOffset, "offset for \(name)")
                #expect(actualOffset + expectedBytes.count <= lease.buffer.length,
                        "named range for \(name) must fit its mapped group")
                let actualBytes = Data(
                    bytes: lease.buffer.contents().advanced(by: actualOffset),
                    count: expectedBytes.count)
                #expect(actualBytes == expectedBytes, "BF16 bytes for \(name)")
                end = expectedOffset + expectedBytes.count
            }
            #expect(lease.diagnostic.residentBytes == UInt64(end))
        }
    }

    @Test func sourceBackedTinyImageRowsAndPositionsMatchIndependentCPUReference() async throws {
        let harness = try makeSourceVisionHarness()
        defer { harness.remove() }

        let pixels = try harness.fixture.makePixelBuffer(device: harness.context.device)
        let expected = try harness.fixture.cpuReference()
        let expectedRowsAreFinite = expected.featureRows.allSatisfy { $0.isFinite }
        #expect(expectedRowsAreFinite)
        let runtime = try QwenVisionRuntime(
            context: harness.context, sourceStore: harness.store)
        let actual = try await runtime.process(pixels)

        #expect(actual.tokenCount == 6)
        #expect(actual.hiddenSize == 8)
        #expect(actual.grid.temporal == 1)
        #expect(actual.grid.height == 4)
        #expect(actual.grid.width == 6)
        #expect(QwenOfficialSourceVisionFixture.matchesFrozenTolerance(
            actual.owner.features(), expected.featureRows),
            "BF16 source vision rows must match the scalar CPU tower")
        #expect(actual.owner.positions().map(\.values)
            == QwenOfficialSourceVisionFixture.expectedFeaturePositions)

        // Keep full three-axis M-RoPE as literal acceptance data. The tiny
        // text fixture's rotary sections are temporal-only, so this is a
        // planner assertion, not a claim of tiny-model height/width rotation.
        let plan = try QwenMultimodalPositions.make(
            tokenCount: 10,
            imageRanges: [QwenOfficialSourceVisionFixture.expectedPromptImageRange],
            grids: [actual.grid], merge: 2, maximumRows: 6)
        #expect(plan.positions.map(\.values)
            == QwenOfficialSourceVisionFixture.expectedPromptPositions)
        #expect(plan.textRoPEDelta
            == QwenOfficialSourceVisionFixture.expectedTextRoPEDelta)
    }

    @Test func deliberateRowAndWeightCorruptionsSeparateTheCPUReference() throws {
        let fixture = try QwenOfficialSourceVisionFixture.make()
        defer { fixture.remove() }
        let baseline = try fixture.cpuReference()

        let swappedPatchWords = try fixture.patchWords(swappingRows: 0, and: 23)
        let swappedRows = try fixture.cpuReference(patchWords: swappedPatchWords)
        #expect(QwenOfficialSourceVisionFixture.maximumAbsoluteDifference(
            baseline.featureRows, swappedRows.featureRows)
            > QwenOfficialSourceVisionFixture.negativeControlMinimumDifference)

        let outputBias = "model.visual.merger.linear_fc2.bias"
        var corruptedBias = try fixture.tensor(named: outputBias).words
        corruptedBias[0] = 0x3F00
        let wrongWeights = try fixture.cpuReference(
            tensorWordOverrides: [outputBias: corruptedBias])
        #expect(QwenOfficialSourceVisionFixture.maximumAbsoluteDifference(
            baseline.featureRows, wrongWeights.featureRows)
            > QwenOfficialSourceVisionFixture.negativeControlMinimumDifference)
    }

    @Test func missingAndMismatchedCompanionsLeaveCommittedTurnUnchanged() async throws {
        let harness = try makeSourceVisionHarness()
        defer { harness.remove() }
        let session = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: harness.model, context: harness.context,
            maxContext: 64, expertSlotCount: QwenBF16TextRunnerFixture.topK)
        let textTurn = try await session.generatePreparedTurn(
            promptTokenIDs: [1, 2], config: sourceVisionGreedyConfig(maxNewTokens: 1))
        #expect(textTurn.acceptedGeneratedTokenIDs.count == 1)
        let committed = try await session.diagnosticSnapshot()
        let pixels = try harness.fixture.makePixelBuffer(device: harness.context.device)
        let events = SourceVisionConversationEventRecorder()
        let missingURL = harness.companionURL.deletingLastPathComponent()
            .appendingPathComponent("missing-\(UUID().uuidString)", isDirectory: true)

        let missing = await captureSourceVisionAsync {
            try await session.generatePreparedImageTurn(
                templateTokenIDs: [1, 3, 2], pixels: [pixels],
                companionURL: missingURL,
                visionConfig: QwenOfficialSourceVisionFixture.config,
                config: sourceVisionGreedyConfig(maxNewTokens: 1),
                onEvent: { events.append($0) })
        }
        #expect(missing.value == nil)
        #expect(missing.errorDescription != nil)
        #expect(try await session.diagnosticSnapshot() == committed)
        #expect(events.snapshot.isEmpty)

        let badCompanion = try makeSourceVisionCompanion(
            fixture: harness.fixture, model: harness.model,
            textDigest: sourceVisionChangedHex(harness.model.sourceContentSHA256))
        defer { try? FileManager.default.removeItem(at: badCompanion) }
        let mismatch = await captureSourceVisionAsync {
            try await session.generatePreparedImageTurn(
                templateTokenIDs: [1, 3, 2], pixels: [pixels],
                companionURL: badCompanion,
                visionConfig: QwenOfficialSourceVisionFixture.config,
                config: sourceVisionGreedyConfig(maxNewTokens: 1),
                onEvent: { events.append($0) })
        }
        #expect(mismatch.value == nil)
        #expect(mismatch.errorIsVisionPack)
        #expect(try await session.diagnosticSnapshot() == committed)
        #expect(events.snapshot.isEmpty)
    }

    @Test func malformedSourceTensorHeaderFailsBeforeReturningAnyGroupLease() throws {
        let name = "model.visual.patch_embed.proj.weight"
        let harness = try makeSourceVisionHarness(
            tensorHeaderShapeOverrides: [name: [3, 16, 1, 2, 2]])
        defer { harness.remove() }
        let outcome = captureSourceVisionSync {
            try harness.store.mapGroup(.patchAndPosition, device: harness.context.device)
        }
        #expect(outcome.value == nil)
        #expect(outcome.errorDescription != nil)
    }

    @Test func changedProtectedSourceCannotSupplyAGroupLease() throws {
        let harness = try makeSourceVisionHarness()
        defer { harness.remove() }
        try harness.fixture.textSource.replaceRoutedShardWithIdenticalBytes()
        let outcome = captureSourceVisionSync {
            try harness.store.mapGroup(.block(0), device: harness.context.device)
        }
        #expect(outcome.value == nil)
        #expect(outcome.errorDescription != nil)
    }

    @Test func originalTextOnlyFixtureStillLoadsWithoutProcessorMetadata() throws {
        let textOnly = try QwenBF16TextRunnerFixture.make()
        defer { textOnly.remove() }
        let processorURL = textOnly.sourceRoot.appendingPathComponent("preprocessor_config.json")
        #expect(!FileManager.default.fileExists(atPath: processorURL.path))
        let context = try MetalContext()
        let model = try QwenOfficialSourceModel.loadSyntheticFixture(
            registrationURL: textOnly.registrationURL,
            context: context,
            residencyBudgetBytes: textOnly.expectedResidentBytes)
        #expect(model.sourceIdentity == nil)
    }

    @Test func cancellationWaitsForMappedCommandThenAllowsReentryAndReuse() async throws {
        let harness = try makeSourceVisionHarness()
        defer { harness.remove() }
        let expected = try harness.fixture.cpuReference()
        let patchWords = harness.fixture.imageInput.patchWords
        let imagePositions = harness.fixture.imageInput.positions
        let device = harness.context.device
        let gate = QwenOfficialSourceVisionTestGate()
        let oneShot = SourceVisionOneShot()
        oneShot.arm()
        let events = QwenVisionExecutionEventRecorder()
        let hooks = QwenVisionExecutionHooks(
            afterCommandSubmission: { stage in
                if stage == "patch+position", oneShot.consume() {
                    await gate.suspend()
                }
            },
            observe: { events.append($0) })
        let runtime = try QwenVisionRuntime(
            context: harness.context, sourceStore: harness.store,
            executionHooks: hooks)

        let blocker = try SourceVisionMetalQueueBlocker(context: harness.context)
        defer {
            blocker.release()
            Task { await gate.release() }
        }
        let first = Task {
            let outcome = await captureSourceVisionAsync {
                try await runtime.processFixtureImageInput(
                    device: device, patchWords: patchWords, positions: imagePositions)
            }
            await gate.producerDidFinish()
            return outcome
        }
        guard await gate.waitUntilEntered() else {
            first.cancel()
            await gate.release()
            blocker.release()
            let early = await first.value
            #expect(Bool(false),
                    "producer ended before reporting its submitted GPU command: \(early.errorDescription ?? "<no error text>")")
            return
        }
        let eventsBeforeGPURelease = events.snapshot
        #expect(eventsBeforeGPURelease.contains {
            if case .commandSubmitted("patch+position") = $0 { return true }
            return false
        }, "the patch+position command must be submitted before the cancellation gate")
        #expect(!eventsBeforeGPURelease.contains {
            if case .commandCompleted("patch+position", discarded: _) = $0 { return true }
            return false
        }, "the runtime must not observe command completion before the cancellation gate is released")
        #expect(!eventsBeforeGPURelease.contains {
            if case .groupReleased(.patchAndPosition) = $0 { return true }
            return false
        }, "mapped weights must remain leased until the queued GPU command completes")
        let overlapping = await captureSourceVisionAsync {
            try await runtime.processFixtureImageInput(
                device: device, patchWords: patchWords, positions: imagePositions)
        }
        #expect(overlapping.value == nil)
        #expect(overlapping.errorDescription?.contains("vision process already active") == true)

        first.cancel()
        await gate.release()
        blocker.release()
        let cancelled = await first.value
        #expect(cancelled.value == nil)
        #expect(cancelled.errorIsCancellation)
        let cancelledEvents = events.snapshot
        let completion = cancelledEvents.firstIndex {
            if case .commandCompleted("patch+position", discarded: true) = $0 { return true }
            return false
        }
        let release = cancelledEvents.firstIndex {
            if case .groupReleased(.patchAndPosition) = $0 { return true }
            return false
        }
        #expect(completion != nil)
        #expect(release != nil)
        if let completion, let release { #expect(completion < release) }

        let retried = try await runtime.processFixtureImageInput(
            device: device, patchWords: patchWords, positions: imagePositions)
        #expect(QwenOfficialSourceVisionFixture.matchesFrozenTolerance(
            retried.features, expected.featureRows))
        #expect(retried.positions
            == QwenOfficialSourceVisionFixture.expectedFeaturePositions)
    }

    @Test func companionMutationAtPreparedImageCommitRollsBackAndRecovers() async throws {
        let harness = try makeSourceVisionHarness()
        defer { harness.remove() }
        let gate = QwenOfficialSourceVisionTestGate()
        let oneShot = SourceVisionOneShot()
        let events = SourceVisionConversationEventRecorder()
        let hooks = QwenOfficialSourceTransactionHooks(beforeTurnCommit: {
            if oneShot.consume() { await gate.suspend() }
        })
        let session = try await QwenOfficialSourceConversationGenerationSession(
            fixtureModel: harness.model, context: harness.context,
            maxContext: 64, expertSlotCount: QwenBF16TextRunnerFixture.topK,
            hooks: hooks)
        let firstPixels = try harness.fixture.makePixelBuffer(device: harness.context.device)
        let first = try await session.generatePreparedImageTurn(
            templateTokenIDs: [1, 3, 2], pixels: [firstPixels],
            companionURL: harness.companionURL,
            visionConfig: QwenOfficialSourceVisionFixture.config,
            config: sourceVisionGreedyConfig(maxNewTokens: 1),
            onEvent: { events.append($0) })
        #expect(first.acceptedGeneratedTokenIDs.count == 1)
        let committed = try await session.diagnosticSnapshot()
        let committedJournal = await session.committedJournalSnapshot()
        let companionURL = harness.companionURL
        let visionConfig = QwenOfficialSourceVisionFixture.config
        let turnConfig = sourceVisionGreedyConfig(maxNewTokens: 1)
        let manifestURL = companionURL
            .appendingPathComponent(OfficialSourceVisionDescriptor.filename)
        let originalManifest = try Data(contentsOf: manifestURL)
        defer { try? writeSourceVisionFileInPlace(originalManifest, to: manifestURL) }
        events.clear()
        oneShot.arm()

        let secondPixels = try harness.fixture.makePixelBuffer(device: harness.context.device)
        let failedTurn = Task {
            let outcome = await captureSourceVisionAsync {
                try await session.generatePreparedImageTurn(
                    templateTokenIDs: [1, 3, 2], pixels: [secondPixels],
                    companionURL: companionURL,
                    visionConfig: visionConfig,
                    config: turnConfig,
                    onEvent: { events.append($0) })
            }
            await gate.producerDidFinish()
            return outcome
        }
        defer { Task { await gate.release() } }
        guard await gate.waitUntilEntered() else {
            await gate.release()
            let early = await failedTurn.value
            #expect(Bool(false),
                    "prepared-image producer ended before commit gate: \(early.errorDescription ?? "<no error text>")")
            return
        }
        let suspendedJournal = await session.committedJournalSnapshot()
        #expect(suspendedJournal.retainedTokenIDs == committedJournal.retainedTokenIDs)
        #expect(suspendedJournal.consumedTokenIDs == committedJournal.consumedTokenIDs)
        #expect(suspendedJournal.pendingAcceptedToken == committedJournal.pendingAcceptedToken)
        #expect(suspendedJournal.currentLogits == committedJournal.currentLogits)
        #expect(suspendedJournal.sourceIdentity == committedJournal.sourceIdentity)
        #expect(suspendedJournal.unusable == committedJournal.unusable)
        #expect(suspendedJournal.activeTransaction != nil,
                "the image turn must remain active while companion validation is suspended")
        do {
            let mutated = try sourceVisionDescriptorWithChangedTextDigest(originalManifest)
            try writeSourceVisionFileInPlace(mutated, to: manifestURL)
        } catch {
            await gate.release()
            throw error
        }
        await gate.release()

        let rejected = await failedTurn.value
        #expect(rejected.value == nil)
        #expect(rejected.errorDescription != nil)
        #expect(try await session.diagnosticSnapshot() == committed)
        #expect(events.snapshot.allSatisfy {
            if case .prefill = $0 { return true }
            return false
        }, "failed commit may report provisional prefill progress, not accepted text/tool events")

        try writeSourceVisionFileInPlace(originalManifest, to: manifestURL)
        events.clear()
        let recoveryPixels = try harness.fixture.makePixelBuffer(device: harness.context.device)
        let recovered = try await session.generatePreparedImageTurn(
            templateTokenIDs: [1, 3, 2], pixels: [recoveryPixels],
            companionURL: companionURL,
            visionConfig: visionConfig,
            config: turnConfig,
            onEvent: { events.append($0) })
        #expect(recovered.acceptedGeneratedTokenIDs.count == 1)
        #expect(try await session.diagnosticSnapshot() != committed)
    }
}

private struct QwenOfficialSourceVisionHarness {
    let fixture: QwenOfficialSourceVisionFixture.Source
    let context: MetalContext
    let model: QwenOfficialSourceModel
    let companionURL: URL
    let store: QwenOfficialSourceVisionWeightStore

    func remove() {
        try? FileManager.default.removeItem(at: companionURL)
        fixture.remove()
    }
}

private func makeSourceVisionHarness(
    tensorWordOverrides: [String: [UInt16]] = [:],
    tensorHeaderShapeOverrides: [String: [Int]] = [:],
    companionTextDigest: String? = nil,
    companionProcessorDigest: String? = nil,
    companionTensors: [OfficialSourceVisionDescriptor.Tensor]? = nil
) throws -> QwenOfficialSourceVisionHarness {
    let fixture = try QwenOfficialSourceVisionFixture.make(
        tensorWordOverrides: tensorWordOverrides,
        tensorHeaderShapeOverrides: tensorHeaderShapeOverrides)
    do {
        // The Phase 15 text-only loader remains processor-independent. Only
        // this image fixture installs its tiny processor sidecar before opening
        // the nil-trust text model through the protected registration.
        try fixture.processorConfiguration.write(
            to: fixture.sourceRoot.appendingPathComponent("preprocessor_config.json"),
            options: .atomic)
        let context = try MetalContext()
        let model = try fixture.loadTinyTextModel(context: context)
        let companionURL = try makeSourceVisionCompanion(
            fixture: fixture, model: model,
            textDigest: companionTextDigest,
            processorDigest: companionProcessorDigest,
            tensors: companionTensors)
        let store = try QwenOfficialSourceVisionWeightStore.open(
            directoryURL: companionURL, model: model,
            config: QwenOfficialSourceVisionFixture.config)
        return QwenOfficialSourceVisionHarness(
            fixture: fixture, context: context, model: model,
            companionURL: companionURL, store: store)
    } catch {
        fixture.remove()
        throw error
    }
}

private func makeSourceVisionCompanion(
    fixture: QwenOfficialSourceVisionFixture.Source,
    model: QwenOfficialSourceModel,
    textDigest: String? = nil,
    processorDigest: String? = nil,
    tensors tensorOverride: [OfficialSourceVisionDescriptor.Tensor]? = nil
) throws -> URL {
    let phase16 = URL(fileURLWithPath: FileManager.default.currentDirectoryPath,
                      isDirectory: true)
        .appendingPathComponent("scratch/qwen3.6-35b-a3b/evidence/phase-16",
                                isDirectory: true)
    let url = phase16.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
        at: url, withIntermediateDirectories: true)
    let tensors = tensorOverride ?? fixture.visionTensors.map {
        OfficialSourceVisionDescriptor.Tensor(name: $0.name,
                                               shape: $0.shape.map(UInt64.init))
    }.sorted { $0.name < $1.name }
    let descriptor = OfficialSourceVisionDescriptor(
        textContentSHA256: textDigest ?? model.sourceContentSHA256,
        processorConfigSHA256: try processorDigest ?? model.visionProcessorSHA256(),
        processorProfile: QwenOfficialSourceVisionFixture.processorProfile,
        tensors: tensors)
    let data = try JSONEncoder().encode(descriptor)
    try data.write(
        to: url.appendingPathComponent(OfficialSourceVisionDescriptor.filename),
        options: .withoutOverwriting)
    return url
}

private func sourceVisionGreedyConfig(maxNewTokens: Int) -> GenerationConfig {
    var config = GenerationConfig(
        maxNewTokens: maxNewTokens, temperature: 0, topK: 1,
        topP: 1, repetitionPenalty: 1, seed: 0)
    config.logitTransform = .raw
    return config
}

private func sourceVisionChangedHex(_ value: String) -> String {
    guard let first = value.first else { return value }
    var changed = value
    changed.replaceSubrange(changed.startIndex...changed.startIndex,
                            with: first == "0" ? "1" : "0")
    return changed
}

private func sourceVisionDescriptorWithChangedTextDigest(_ data: Data) throws -> Data {
    let original = try OfficialSourceVisionDescriptor.decodeStrict(data)
    let changed = OfficialSourceVisionDescriptor(
        textContentSHA256: sourceVisionChangedHex(original.textContentSHA256),
        processorConfigSHA256: original.processorConfigSHA256,
        processorProfile: original.processorProfile,
        tensors: original.tensors)
    let encoded = try JSONEncoder().encode(changed)
    guard encoded.count == data.count else {
        throw QwenOfficialSourceVisionFixture.FixtureError
            .malformedVisionFixture("mutated companion must preserve byte length")
    }
    return encoded
}

private func writeSourceVisionFileInPlace(_ data: Data, to url: URL) throws {
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    try handle.seek(toOffset: 0)
    try handle.write(contentsOf: data)
    try handle.truncate(atOffset: UInt64(data.count))
    try handle.synchronize()
}

private final class SourceVisionMetalQueueBlocker {
    let event: MTLSharedEvent
    let waitCommandBuffer: MTLCommandBuffer

    init(context: MetalContext) throws {
        guard let event = context.device.makeSharedEvent(),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw QwenVisionError.allocationFailed(name: "vision test shared-event blocker")
        }
        self.event = event
        self.waitCommandBuffer = commandBuffer
        let scheduled = DispatchSemaphore(value: 0)
        commandBuffer.addScheduledHandler { _ in scheduled.signal() }
        commandBuffer.encodeWaitForEvent(event, value: 1)
        commandBuffer.commit()
        guard scheduled.wait(timeout: .now() + 10) == .success else {
            event.signaledValue = 1
            throw QwenVisionError.allocationFailed(name: "vision test queue scheduling timeout")
        }
    }

    func release() {
        event.signaledValue = 1
    }
}

private struct SourceVisionProcessOutput: Sendable {
    let features: [Float]
    let positions: [[Int32]]
}

private extension QwenVisionRuntime {
    func processFixtureImageInput(
        device: MTLDevice,
        patchWords: [UInt16],
        positions: [SIMD2<Int32>]
    ) async throws -> SourceVisionProcessOutput {
        let pixels = try QwenOfficialSourceVisionFixture.Source.makePixelBuffer(
            device: device, patchWords: patchWords, positions: positions)
        let output = try await process(pixels)
        return SourceVisionProcessOutput(
            features: output.owner.features(),
            positions: output.owner.positions().map(\.values))
    }
}

private struct SourceVisionAsyncOutcome<Value: Sendable>: Sendable {
    let value: Value?
    let errorDescription: String?
    let errorIsCancellation: Bool
    let errorIsVisionPack: Bool
}

private func captureSourceVisionAsync<Value: Sendable>(
    _ operation: () async throws -> Value
) async -> SourceVisionAsyncOutcome<Value> {
    do {
        return SourceVisionAsyncOutcome(
            value: try await operation(), errorDescription: nil,
            errorIsCancellation: false, errorIsVisionPack: false)
    } catch {
        return SourceVisionAsyncOutcome(
            value: nil, errorDescription: String(describing: error),
            errorIsCancellation: error is CancellationError,
            errorIsVisionPack: error is VisionPackError)
    }
}

private struct SourceVisionSyncOutcome<Value> {
    let value: Value?
    let errorDescription: String?
}

private func captureSourceVisionSync<Value>(
    _ operation: () throws -> Value
) -> SourceVisionSyncOutcome<Value> {
    do { return SourceVisionSyncOutcome(value: try operation(), errorDescription: nil) }
    catch { return SourceVisionSyncOutcome(value: nil, errorDescription: String(describing: error)) }
}

private final class SourceVisionOneShot: @unchecked Sendable {
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

private final class QwenVisionExecutionEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [QwenVisionExecutionEvent] = []

    func append(_ event: QwenVisionExecutionEvent) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }

    var snapshot: [QwenVisionExecutionEvent] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}

private final class SourceVisionConversationEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [QwenConversationGenerationEvent] = []

    func append(_ event: QwenConversationGenerationEvent) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }

    func clear() {
        lock.lock()
        events.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    var snapshot: [QwenConversationGenerationEvent] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}
