import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenTextRunnerTests {
    private let prompt: [Int32] = [1, 4, 7]
    private let stream: [Int32] = [1, 4, 7, 2, 14]
    private let expectedGreedy: [Int32] = [1, 12, 6]

    @Test func oneShotPrefillMatchesNormalizedPackedOracleIncludingLayerTraces() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let runner = try model.makeReferenceRunner()
        let output = try runner.prefill(tokens: stream)
        let root = try QwenTextFixtureSupport.object()
        let oneShot = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "streaming", "oneShot",
        ])
        let expectedHidden = try QwenTextFixtureSupport.tensorValues(oneShot, at: ["finalHidden"])
        let expectedLogits = try QwenTextFixtureSupport.tensorValues(oneShot, at: ["rawLogits"])

        #expect(output.tokenCount == stream.count)
        #expect(output.finalHidden.count == stream.count * 32)
        #expect(output.logits.count == stream.count * 19)
        assertClose(output.finalHidden, expectedHidden,
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "one-shot final hidden")
        assertClose(output.logits, expectedLogits,
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "one-shot raw logits")
        #expect(output.layers.count == 4)
        let traceRunner = try model.makeReferenceRunner()
        let traceOutput = try traceRunner.prefill(tokens: prompt)
        let expectedLayers = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "prefillTrace", "layers",
        ])
        guard let expectedLayerArray = expectedLayers as? [Any] else {
            Issue.record("one-shot layer traces are not an array")
            return
        }
        for layer in 0..<4 {
            guard let layerObject = expectedLayerArray[layer] as? [String: Any] else {
                Issue.record("missing expected layer trace \(layer)")
                continue
            }
            assertClose(traceOutput.layers[layer].inputNormalized,
                        try QwenTextFixtureSupport.tensorValues(
                            try QwenTextFixtureSupport.value(layerObject, at: ["inputNormalized"])),
                        absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                        relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                        label: "layer \(layer) input norm")
            assertClose(traceOutput.layers[layer].postAttentionNormalized,
                        try QwenTextFixtureSupport.tensorValues(
                            try QwenTextFixtureSupport.value(layerObject, at: ["postAttentionNormalized"])),
                        absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                        relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                        label: "layer \(layer) post-attention norm")
            assertClose(traceOutput.layers[layer].postResidualHidden,
                        try QwenTextFixtureSupport.tensorValues(
                            try QwenTextFixtureSupport.value(layerObject, at: ["postResidualHidden"])),
                        absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                        relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                        label: "layer \(layer) residual hidden")
        }
        #expect(runner.position == stream.count)
    }

    @Test func splitAndMultiChunkPrefillMatchTheirIndependentPackedReferences() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let root = try QwenTextFixtureSupport.object()
        let streaming = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "streaming",
        ])

        let splitRunner = try model.makeReferenceRunner()
        _ = try splitRunner.append(tokens: Array(stream.prefix(3)))
        let splitOutput = try splitRunner.append(tokens: Array(stream.suffix(2)))
        let splitExpected = try QwenTextFixtureSupport.value(streaming, at: ["boundary3Split"])
        assertSuffixMatches(splitOutput,
                            hidden: try QwenTextFixtureSupport.tensorValues(splitExpected, at: ["finalHidden"]),
                            logits: try QwenTextFixtureSupport.tensorValues(splitExpected, at: ["rawLogits"]),
                            tokenCount: 2,
                            label: "boundary-3 split")

        let chunkedRunner = try model.makeReferenceRunner()
        var chunkOutput: QwenTextRunnerOutput?
        for chunk in [[Int32(1), 4], [7], [2, 14]] {
            chunkOutput = try chunkedRunner.append(tokens: chunk)
        }
        let multiChunkOutput = try #require(chunkOutput)
        let multiExpected = try QwenTextFixtureSupport.value(streaming, at: ["splitMultichunk"])
        assertSuffixMatches(multiChunkOutput,
                            hidden: try QwenTextFixtureSupport.tensorValues(multiExpected, at: ["finalHidden"]),
                            logits: try QwenTextFixtureSupport.tensorValues(multiExpected, at: ["rawLogits"]),
                            tokenCount: 2,
                            label: "multi-chunk")
        #expect(splitRunner.position == stream.count)
        #expect(chunkedRunner.position == stream.count)
    }

    @Test func tokenAtATimeStateCarryMatchesReferenceRowsAndStateLengths() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let runner = try model.makeReferenceRunner()
        let root = try QwenTextFixtureSupport.object()
        let streaming = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "streaming",
        ])
        let tokenReference = try QwenTextFixtureSupport.value(streaming, at: ["tokenAtATime"])
        let expectedHidden = try QwenTextFixtureSupport.tensorValues(tokenReference, at: ["finalHidden"])
        let expectedLogits = try QwenTextFixtureSupport.tensorValues(tokenReference, at: ["rawLogits"])

        for (index, token) in stream.enumerated() {
            let output = try runner.append(tokens: [token])
            let expectedHiddenRow = Array(expectedHidden[(index * 32)..<(index * 32 + 32)])
            let expectedLogitRow = Array(expectedLogits[(index * 19)..<(index * 19 + 19)])
            #expect(output.tokenCount == 1)
            assertClose(output.finalHidden, expectedHiddenRow,
                        absolute: QwenTextFixtureSupport.stateAbsoluteTolerance,
                        relative: QwenTextFixtureSupport.stateRelativeTolerance,
                        label: "token-at-a-time hidden \(index)")
            assertClose(output.logits, expectedLogitRow,
                        absolute: QwenTextFixtureSupport.stateAbsoluteTolerance,
                        relative: QwenTextFixtureSupport.stateRelativeTolerance,
                        label: "token-at-a-time logits \(index)")
            #expect(output.state.sequenceLength == index + 1)
            #expect(output.state.layers.count == 4)
        }
        let state = runner.snapshot()
        for layer in state.layers {
            switch layer {
            case let .linear(history, recurrent):
                #expect(history.allSatisfy { $0.isFinite })
                #expect(recurrent.allSatisfy { $0.isFinite })
            case let .full(key, value):
                #expect(key.count == 5 * 16)
                #expect(value.count == 5 * 16)
                #expect(key.allSatisfy { $0.isFinite })
                #expect(value.allSatisfy { $0.isFinite })
            }
        }
    }

    @Test func cachedDecodeAndFullRecomputeProduceNewGreedyIDsAndMatchingLogits() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let root = try QwenTextFixtureSupport.object()
        let complete = try QwenTextFixtureSupport.value(root, at: ["completeTextModel"])
        let cached = try QwenTextFixtureSupport.value(complete, at: ["cachedGreedy"])
        let decisions = try QwenTextFixtureSupport.value(cached, at: ["decisions"])
        guard let decisionArray = decisions as? [Any], decisionArray.count == expectedGreedy.count else {
            Issue.record("cached greedy decisions are incomplete")
            return
        }

        let cachedRunner = try model.makeReferenceRunner()
        let prefillOutput = try cachedRunner.prefill(tokens: prompt)
        var cachedIDs: [Int32] = []
        let prefillDecision = decisionArray[0]
        assertClose(Array(prefillOutput.logits.suffix(19)),
                    try QwenTextFixtureSupport.tensorValues(prefillDecision, at: ["rawLogits"]),
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "cached prefill logits")
        cachedIDs.append(try #require(
            (try QwenTextFixtureSupport.value(prefillDecision, at: ["selectedToken"]) as? NSNumber)?.int32Value))
        for index in 1..<decisionArray.count {
            let output = try cachedRunner.decode(token: cachedIDs[index - 1])
            let decision = decisionArray[index]
            let expectedLogits = try QwenTextFixtureSupport.tensorValues(decision, at: ["rawLogits"])
            assertClose(output.logits, expectedLogits,
                        absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                        relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                        label: "cached decode logits \(index)")
            let selected = try QwenTextFixtureSupport.value(decision, at: ["selectedToken"])
            guard let selectedNumber = selected as? NSNumber else {
                Issue.record("missing cached selected token")
                continue
            }
            cachedIDs.append(selectedNumber.int32Value)
        }
        #expect(cachedIDs == expectedGreedy)
        #expect(cachedIDs != [2, 14, 0])

        var recomputedIDs: [Int32] = []
        for index in 0..<expectedGreedy.count {
            let recomputeRunner = try model.makeReferenceRunner()
            let input = prompt + Array(expectedGreedy.prefix(index))
            let output = try recomputeRunner.prefill(tokens: input)
            let expectedStep = try QwenTextFixtureSupport.arrayElement(
                complete, at: ["recomputeGreedy", "steps"], index: index)
            let expectedLogits = try QwenTextFixtureSupport.tensorValues(expectedStep, at: ["rawLogits"])
            let actualLast = Array(output.logits.suffix(expectedLogits.count))
            assertClose(actualLast, expectedLogits,
                        absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                        relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                        label: "full recompute logits \(index)")
            let selected = try QwenTextFixtureSupport.value(expectedStep, at: ["selectedToken"])
            recomputedIDs.append(try #require((selected as? NSNumber)?.int32Value))
        }
        #expect(recomputedIDs == expectedGreedy)
    }

    @Test func snapshotRestoreResetAndInvalidInputsPreserveStateContracts() throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let runner = try model.makeReferenceRunner()
        #expect(throws: QwenTextRunnerError.emptyInput) {
            _ = try runner.append(tokens: [])
        }
        #expect(throws: QwenTextRunnerError.invalidToken(id: -1)) {
            _ = try runner.append(tokens: [-1])
        }
        _ = try runner.prefill(tokens: prompt)
        let before = runner.snapshot()
        let first = try runner.decode(token: expectedGreedy[0])
        try runner.restore(before)
        let replay = try runner.decode(token: expectedGreedy[0])
        #expect(replay == first)
        #expect(runner.position == 4)
        #expect(throws: QwenTextRunnerError.alreadyPrefilled(position: 4)) {
            _ = try runner.prefill(tokens: [0])
        }

        let wrongIdentity = QwenTextRunnerState(
            architectureIdentity: "not-the-mapped-model",
            sequenceLength: before.sequenceLength,
            layers: before.layers)
        #expect(throws: QwenTextRunnerError.stateArchitectureMismatch) {
            try runner.restore(wrongIdentity)
        }
        #expect(runner.position == 4)
        runner.reset()
        #expect(runner.position == 0)
    }

    @Test func integratedHybridOneShotMatchesFrozenOracleAndReportsEveryGPUStage() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let runner = try model.makeRunner(context: context)
        let result = try await runner.prefill(tokens: stream)
        let output = result.output
        let root = try QwenTextFixtureSupport.object()
        let oneShot = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "streaming", "oneShot",
        ])
        let maximumHiddenDelta = assertClose(output.finalHidden,
                                              try QwenTextFixtureSupport.tensorValues(oneShot, at: ["finalHidden"]),
                                              absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                                              relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                                              label: "hybrid one-shot hidden")
        let maximumLogitDelta = assertClose(output.logits,
                                            try QwenTextFixtureSupport.tensorValues(oneShot, at: ["rawLogits"]),
                                            absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                                            relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                                            label: "hybrid one-shot logits")
        #expect(output.tokenCount == stream.count)
        assertHybridDiagnostics(result.diagnostics, label: "hybrid one-shot")
        let maximumStateDelta = try assertStateMatchesFixture(
            output.state,
            oneShot,
            path: ["finalState"],
            absolute: QwenTextFixtureSupport.stateAbsoluteTolerance,
            relative: QwenTextFixtureSupport.stateRelativeTolerance,
            label: "hybrid one-shot state")
        print("QWEN_HYBRID_MAX one-shot hidden_abs=\(maximumHiddenDelta) logits_abs=\(maximumLogitDelta) state_abs=\(maximumStateDelta)")
        #expect(await runner.position == stream.count)
    }

    @Test func integratedHybridSplitAndMultichunkPreserveFrozenOutputsAndState() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let root = try QwenTextFixtureSupport.object()
        let streaming = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "streaming",
        ])

        let splitRunner = try model.makeRunner(context: context)
        _ = try await splitRunner.append(tokens: Array(stream.prefix(3)))
        let splitResult = try await splitRunner.append(tokens: Array(stream.suffix(2)))
        let splitExpected = try QwenTextFixtureSupport.value(streaming, at: ["boundary3Split"])
        assertSuffixMatches(splitResult.output,
                            hidden: try QwenTextFixtureSupport.tensorValues(splitExpected, at: ["finalHidden"]),
                            logits: try QwenTextFixtureSupport.tensorValues(splitExpected, at: ["rawLogits"]),
                            tokenCount: 2,
                            label: "hybrid boundary-3 split")
        assertHybridDiagnostics(splitResult.diagnostics, label: "hybrid boundary-3 split")

        let chunkedRunner = try model.makeRunner(context: context)
        var chunkResult: QwenTextHybridResult?
        for chunk in [[Int32(1), 4], [7], [2, 14]] {
            chunkResult = try await chunkedRunner.append(tokens: chunk)
        }
        let multiResult = try #require(chunkResult)
        let multiExpected = try QwenTextFixtureSupport.value(streaming, at: ["splitMultichunk"])
        assertSuffixMatches(multiResult.output,
                            hidden: try QwenTextFixtureSupport.tensorValues(multiExpected, at: ["finalHidden"]),
                            logits: try QwenTextFixtureSupport.tensorValues(multiExpected, at: ["rawLogits"]),
                            tokenCount: 2,
                            label: "hybrid multi-chunk")
        assertHybridDiagnostics(multiResult.diagnostics, label: "hybrid multi-chunk")
        #expect(await splitRunner.position == stream.count)
        #expect(await chunkedRunner.position == stream.count)
    }

    @Test func integratedHybridTokenStateCarryAndCachedDecodeMatchFrozenRows() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let root = try QwenTextFixtureSupport.object()
        let streaming = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "streaming",
        ])
        let tokenReference = try QwenTextFixtureSupport.value(streaming, at: ["tokenAtATime"])
        let expectedHidden = try QwenTextFixtureSupport.tensorValues(tokenReference, at: ["finalHidden"])
        let expectedLogits = try QwenTextFixtureSupport.tensorValues(tokenReference, at: ["rawLogits"])
        let runner = try model.makeRunner(context: context)
        var maximumHiddenDelta: Float = 0
        var maximumLogitDelta: Float = 0

        for (index, token) in stream.enumerated() {
            let result = try await runner.append(tokens: [token])
            let output = result.output
            let expectedHiddenRow = Array(expectedHidden[(index * 32)..<(index * 32 + 32)])
            let expectedLogitRow = Array(expectedLogits[(index * 19)..<(index * 19 + 19)])
            #expect(output.tokenCount == 1)
            maximumHiddenDelta = max(maximumHiddenDelta, assertClose(
                output.finalHidden, expectedHiddenRow,
                absolute: QwenTextFixtureSupport.stateAbsoluteTolerance,
                relative: QwenTextFixtureSupport.stateRelativeTolerance,
                label: "hybrid token hidden \(index)"))
            maximumLogitDelta = max(maximumLogitDelta, assertClose(
                output.logits, expectedLogitRow,
                absolute: QwenTextFixtureSupport.stateAbsoluteTolerance,
                relative: QwenTextFixtureSupport.stateRelativeTolerance,
                label: "hybrid token logits \(index)"))
            #expect(output.state.sequenceLength == index + 1)
            #expect(output.state.layers.count == 4)
            assertHybridDiagnostics(result.diagnostics, label: "hybrid token \(index)")
        }
        let state = try await runner.snapshot()
        let maximumStateDelta = try assertStateMatchesFixture(
            state,
            tokenReference,
            path: ["finalState"],
            absolute: QwenTextFixtureSupport.stateAbsoluteTolerance,
            relative: QwenTextFixtureSupport.stateRelativeTolerance,
            label: "hybrid token state")
        print("QWEN_HYBRID_MAX token-at-a-time hidden_abs=\(maximumHiddenDelta) logits_abs=\(maximumLogitDelta) state_abs=\(maximumStateDelta)")
        #expect(state.sequenceLength == stream.count)
        for layer in state.layers {
            switch layer {
            case let .linear(history, recurrent):
                #expect(history.count == 24 * 4)
                #expect(recurrent.count == 2 * 4 * 4)
                #expect(history.allSatisfy { $0.isFinite })
                #expect(recurrent.allSatisfy { $0.isFinite })
            case let .full(key, value):
                #expect(key.count == 5 * 16)
                #expect(value.count == 5 * 16)
                #expect(key.allSatisfy { $0.isFinite })
                #expect(value.allSatisfy { $0.isFinite })
            }
        }

        let complete = try QwenTextFixtureSupport.value(root, at: ["completeTextModel"])
        let decisions = try QwenTextFixtureSupport.value(
            try QwenTextFixtureSupport.value(complete, at: ["cachedGreedy"]), at: ["decisions"])
        guard let decisionArray = decisions as? [Any], decisionArray.count == expectedGreedy.count else {
            Issue.record("hybrid cached greedy decisions are incomplete")
            return
        }
        let cachedRunner = try model.makeRunner(context: context)
        let prefill = try await cachedRunner.prefill(tokens: prompt)
        var ids: [Int32] = []
        let firstDecision = decisionArray[0]
        assertClose(Array(prefill.output.logits.suffix(19)),
                    try QwenTextFixtureSupport.tensorValues(firstDecision, at: ["rawLogits"]),
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "hybrid cached prefill logits")
        ids.append(try #require(
            (try QwenTextFixtureSupport.value(firstDecision, at: ["selectedToken"]) as? NSNumber)?.int32Value))
        assertHybridDiagnostics(prefill.diagnostics, label: "hybrid cached prefill")
        for index in 1..<decisionArray.count {
            let result = try await cachedRunner.decode(token: ids[index - 1])
            let decision = decisionArray[index]
            assertClose(result.output.logits,
                        try QwenTextFixtureSupport.tensorValues(decision, at: ["rawLogits"]),
                        absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                        relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                        label: "hybrid cached decode logits \(index)")
            ids.append(try #require(
                (try QwenTextFixtureSupport.value(decision, at: ["selectedToken"]) as? NSNumber)?.int32Value))
            assertHybridDiagnostics(result.diagnostics, label: "hybrid cached decode \(index)")
        }
        #expect(ids == expectedGreedy)
    }

    @Test func integratedHybridRejectsInvalidInputsAndBusyStateWithoutCorruption() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let runner = try model.makeRunner(context: context)
        await #expect(throws: QwenTextRunnerError.emptyInput) {
            _ = try await runner.append(tokens: [])
        }
        await #expect(throws: QwenTextRunnerError.invalidToken(id: -1)) {
            _ = try await runner.append(tokens: [-1])
        }
        let output = try await runner.prefill(tokens: prompt)
        #expect(output.output.state.sequenceLength == prompt.count)
        #expect(await runner.position == prompt.count)
        let snapshot = try await runner.snapshot()
        #expect(snapshot.sequenceLength == prompt.count)
        try await runner.restore(snapshot)
        #expect(await runner.position == prompt.count)
        await #expect(throws: QwenTextRunnerError.alreadyPrefilled(position: prompt.count)) {
            _ = try await runner.prefill(tokens: [0])
        }
        await #expect(throws: QwenTextRunnerError.stateArchitectureMismatch) {
            try await runner.restore(QwenTextRunnerState(
                architectureIdentity: "not-the-mapped-model",
                sequenceLength: snapshot.sequenceLength,
                layers: snapshot.layers))
        }
        #expect(await runner.position == prompt.count)
        try await runner.reset()
        #expect(await runner.position == 0)

        let producer = try model.makeLogitProducer(context: context)
        guard let logits = context.device.makeBuffer(
            length: 19 * MemoryLayout<UInt16>.stride, options: .storageModeShared),
              let smallLogits = context.device.makeBuffer(
                length: 18 * MemoryLayout<UInt16>.stride, options: .storageModeShared) else {
            Issue.record("unable to allocate validation logits buffers")
            return
        }
        await #expect(throws: QwenTextRunnerError.invalidPosition(expected: 0, actual: 1)) {
            try await producer.produce(token: 1, position: 1, into: logits)
        }
        await #expect(throws: QwenTextRunnerError.logitsBufferTooSmall(expected: 38, actual: 36)) {
            try await producer.produce(token: 1, position: 0, into: smallLogits)
        }
    }

    @Test func logitProducerPreWriteResetSuppressesLogitsAndRollsBack() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let entered = AsyncLatch()
        let release = AsyncLatch()
        let hooks = QwenTextPublicationHooks(beforeLockedWrite: {
            await entered.signal()
            await release.wait()
        })
        let producer = try model.makeLogitProducer(context: context, publicationHooks: hooks)
        guard let logits = context.device.makeBuffer(
            length: 19 * MemoryLayout<UInt16>.stride, options: .storageModeShared) else {
            Issue.record("unable to allocate shared logits buffer")
            return
        }
        let sentinel: UInt16 = 0x7BFF
        fillUInt16Buffer(logits, count: 19, with: sentinel)
        let holder = TestSendableBuffer(logits)

        let task = Task {
            try await producer.produce(token: 1, position: 0, into: holder.buffer)
        }
        await entered.wait()
        await #expect(throws: QwenTextRunnerError.operationInProgress) {
            _ = try await producer.snapshot()
        }
        producer.reset()
        await release.signal()
        await #expect(throws: QwenTextRunnerError.cancelled) {
            try await task.value
        }
        #expect(allUInt16BufferValuesEqual(holder.buffer, count: 19, to: sentinel))
        let state = try await producer.snapshot()
        #expect(state.sequenceLength == 0)
    }

    @Test func logitProducerPostWriteResetCommitsVisibleLogitsAndState() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let entered = AsyncLatch()
        let release = AsyncLatch()
        let hooks = QwenTextPublicationHooks(afterLockedWriteBeforeFinish: {
            await entered.signal()
            await release.wait()
        })
        let producer = try model.makeLogitProducer(context: context, publicationHooks: hooks)
        guard let logits = context.device.makeBuffer(
            length: 19 * MemoryLayout<UInt16>.stride, options: .storageModeShared) else {
            Issue.record("unable to allocate shared logits buffer")
            return
        }
        let sentinel: UInt16 = 0x7BFF
        fillUInt16Buffer(logits, count: 19, with: sentinel)
        let holder = TestSendableBuffer(logits)

        let task = Task {
            try await producer.produce(token: 1, position: 0, into: holder.buffer)
        }
        await entered.wait()
        producer.reset()
        task.cancel()
        await release.signal()
        try await task.value
        #expect(anyUInt16BufferValueDiffers(holder.buffer, count: 19, from: sentinel))
        let state = try await producer.snapshot()
        #expect(state.sequenceLength == 1)

        guard let nextLogits = context.device.makeBuffer(
            length: 19 * MemoryLayout<UInt16>.stride, options: .storageModeShared) else {
            Issue.record("unable to allocate second shared logits buffer")
            return
        }
        try await producer.produce(token: 1, position: 0, into: nextLogits)
        let reusedState = try await producer.snapshot()
        #expect(reusedState.sequenceLength == 1)
    }

    @Test func hybridCancellationAfterSubmittedConvolutionRollsBackAndReleasesResources() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let submitted = AsyncLatch()
        let release = AsyncLatch()
        let event = try #require(context.device.makeSharedEvent())
        let eventHolder = TestSendableSharedEvent(event)
        let blocker = try #require(context.queue.makeCommandBuffer())
        blocker.encodeWaitForEvent(event, value: 1)
        blocker.commit()
        let hooks = QwenTextExecutionHooks(afterCommandSubmission: { stage in
            guard stage == "linearConvolution" else { return }
            await submitted.signal()
            await release.wait()
            eventHolder.event.signaledValue = 1
        })
        let runner = try model.makeRunner(context: context, executionHooks: hooks)
        let task = Task {
            try await runner.append(tokens: prompt)
        }
        await submitted.wait()
        #expect(blocker.status != .completed)
        await #expect(throws: QwenTextRunnerError.operationInProgress) {
            _ = try await runner.snapshot()
        }
        task.cancel()
        await release.signal()
        await #expect(throws: QwenTextRunnerError.cancelled) {
            try await task.value
        }
        _ = await blocker.completed()
        #expect(blocker.status == .completed)
        let rolledBack = try await runner.snapshot()
        #expect(rolledBack.sequenceLength == 0)
        for layer in rolledBack.layers {
            switch layer {
            case let .linear(history, recurrent):
                #expect(history.allSatisfy { $0 == 0 })
                #expect(recurrent.allSatisfy { $0 == 0 })
            case let .full(key, value):
                #expect(key.isEmpty)
                #expect(value.isEmpty)
            }
        }
        let reused = try await runner.append(tokens: [1])
        #expect(reused.output.tokenCount == 1)
        #expect(await runner.position == 1)
    }

    @Test func hybridFailureAfterCompletedEarlierLayersRollsBackAndSupportsReuse() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let failureSwitch = FailureSwitch()
        let stages = StageRecorder()
        let hooks = QwenTextExecutionHooks(
            beforeLayer: { layer in
                if layer == 1, await failureSwitch.consume() {
                    throw InjectedHybridFailure()
                }
            },
            afterCommandSubmission: { stage in
                await stages.append(stage)
            })
        let runner = try model.makeRunner(context: context, executionHooks: hooks)
        await #expect(throws: InjectedHybridFailure.self) {
            _ = try await runner.append(tokens: [1])
        }
        let completedStages = await stages.values()
        #expect(completedStages.filter { $0 == "linearConvolution" }.count >= 1)
        #expect(completedStages.filter { $0 == "linearRecurrence" }.count >= 1)
        #expect(completedStages.filter { $0 == "mappedExperts" }.count >= 1)
        let rolledBack = try await runner.snapshot()
        #expect(rolledBack.sequenceLength == 0)
        for layer in rolledBack.layers {
            switch layer {
            case let .linear(history, recurrent):
                #expect(history.allSatisfy { $0 == 0 })
                #expect(recurrent.allSatisfy { $0 == 0 })
            case let .full(key, value):
                #expect(key.isEmpty)
                #expect(value.isEmpty)
            }
        }

        let root = try QwenTextFixtureSupport.object()
        let oneShot = try QwenTextFixtureSupport.value(root, at: [
            "completeTextModel", "streaming", "oneShot",
        ])
        let reused = try await runner.append(tokens: stream)
        assertClose(reused.output.finalHidden,
                    try QwenTextFixtureSupport.tensorValues(oneShot, at: ["finalHidden"]),
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "post-failure reuse hidden")
        assertClose(reused.output.logits,
                    try QwenTextFixtureSupport.tensorValues(oneShot, at: ["rawLogits"]),
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "post-failure reuse logits")
        #expect(reused.output.state.sequenceLength == stream.count)
        #expect(await runner.position == stream.count)
    }

    @Test func hybridCancellationAfterEarlierLayerCommitAndPendingUpdateRollsBackWholeRunner() async throws {
        let context = try MetalContext()
        let model = try QwenTextFixtureSupport.model(device: context.device)
        let layer = CurrentLayer()
        let commitOnce = OnceFlag()
        let pauseOnce = OnceFlag()
        let submitted = AsyncLatch()
        let release = AsyncLatch()
        let event = try #require(context.device.makeSharedEvent())
        let eventHolder = TestSendableSharedEvent(event)
        let blocker = try #require(context.queue.makeCommandBuffer())
        blocker.encodeWaitForEvent(event, value: 1)
        let blockerHolder = TestSendableCommandBuffer(blocker)
        let hooks = QwenTextExecutionHooks(
            beforeLayer: { index in
                await layer.set(index)
                if index == 1, await commitOnce.take() {
                    blockerHolder.buffer.commit()
                }
            },
            afterCommandSubmission: { stage in
                guard stage == "linearConvolution", await layer.isCurrent(1), await pauseOnce.take() else {
                    return
                }
                await submitted.signal()
                await release.wait()
                eventHolder.event.signaledValue = 1
            })
        let runner = try model.makeRunner(context: context, executionHooks: hooks)
        let task = Task {
            try await runner.append(tokens: prompt)
        }
        await submitted.wait()
        #expect(blocker.status != .completed)
        await #expect(throws: QwenTextRunnerError.operationInProgress) {
            _ = try await runner.snapshot()
        }
        task.cancel()
        await release.signal()
        await #expect(throws: QwenTextRunnerError.cancelled) {
            try await task.value
        }
        _ = await blocker.completed()
        #expect(blocker.status == .completed)
        let rolledBack = try await runner.snapshot()
        #expect(rolledBack.sequenceLength == 0)
        for state in rolledBack.layers {
            switch state {
            case let .linear(history, recurrent):
                #expect(history.allSatisfy { $0 == 0 })
                #expect(recurrent.allSatisfy { $0 == 0 })
            case let .full(key, value):
                #expect(key.isEmpty)
                #expect(value.isEmpty)
            }
        }
        let reused = try await runner.append(tokens: [1])
        #expect(reused.output.tokenCount == 1)
        #expect(await runner.position == 1)
    }

    @Test func qwenSamplingIsRawWhileGemmaDefaultRetainsSoftcapAndExplicitStops() throws {
        let gemma = GenerationConfig()
        #expect(gemma.logitTransform == .gemmaSoftcap(30))
        let qwen = GenerationConfig.qwenRaw(
            maxNewTokens: 3, temperature: 0, stopTokenIDs: [248_044, 7])
        #expect(qwen.logitTransform == .raw)
        #expect(qwen.extraStopTokens == [248_044, 7])
        #expect(qwen.maxNewTokens == 3)
    }

    @Test func frozenNegativeControlsExceedDeclaredAcceptanceBudgets() throws {
        let root = try QwenTextFixtureSupport.object()
        let controls = try QwenTextFixtureSupport.value(root, at: ["negativeControls"])
        guard let controlArray = controls as? [Any] else {
            Issue.record("negative controls are not an array")
            return
        }
        let required = Set([
            "bypassInputNorms", "bypassPostAttentionNorms",
            "bypassFinalNorm", "resetCacheBeforeDecode",
        ])
        #expect(Set(controlArray.compactMap {
            ($0 as? [String: Any])?["name"] as? String
        }) == required)
        for control in controlArray {
            guard let object = control as? [String: Any],
                  let name = object["name"] as? String,
                  let difference = (object["maxAbsoluteDifference"] as? NSNumber)?.floatValue,
                  let tolerance = (object["requiredToExceed"] as? NSNumber)?.floatValue else {
                Issue.record("malformed negative control")
                continue
            }
            #expect(difference > tolerance, "negative control \(name) does not exceed tolerance")
            #expect(object["detected"] as? Bool == true)
        }
    }

    private func assertSuffixMatches(
        _ output: QwenTextRunnerOutput,
        hidden: [Float], logits: [Float], tokenCount: Int, label: String
    ) {
        let expectedHidden = Array(hidden.suffix(tokenCount * 32))
        let expectedLogits = Array(logits.suffix(tokenCount * 19))
        #expect(output.tokenCount == tokenCount)
        assertClose(output.finalHidden, expectedHidden,
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "\(label) hidden")
        assertClose(output.logits, expectedLogits,
                    absolute: QwenTextFixtureSupport.fp32AbsoluteTolerance,
                    relative: QwenTextFixtureSupport.fp32RelativeTolerance,
                    label: "\(label) logits")
    }
    private func assertHybridDiagnostics(
        _ diagnostics: QwenTextExecutionDiagnostics, label: String
    ) {
        #expect(diagnostics.fullAttentionNormRoPESubmissions == 2,
                "\(label): P9 norm/RoPE submissions")
        #expect(diagnostics.fullAttentionGateSubmissions == 1,
                "\(label): P9 gate submissions")
        #expect(diagnostics.linearLayoutSubmissions == 3,
                "\(label): P10 layout submissions")
        #expect(diagnostics.linearConvolutionSubmissions == 3,
                "\(label): P10 convolution submissions")
        #expect(diagnostics.linearRecurrenceSubmissions == 3,
                "\(label): P10 recurrence submissions")
        #expect(diagnostics.linearGatedNormSubmissions == 3,
                "\(label): P10 gated-norm submissions")
        #expect(diagnostics.mappedExpertSubmissions > 0,
                "\(label): P11 mapped expert submissions")
        #expect(diagnostics.completedCommandBuffers > 0,
                "\(label): completed command buffers")
        #expect(diagnostics.discardedCommandBuffers == 0,
                "\(label): discarded command buffers")
        #expect(diagnostics.convolutionOutputsConsumed == diagnostics.linearConvolutionSubmissions,
                "\(label): every GPU convolution output must feed the next stage")
    }

    @discardableResult
    private func assertStateMatchesFixture(
        _ actual: QwenTextRunnerState,
        _ fixture: Any,
        path: [String],
        absolute: Float,
        relative: Float,
        label: String
    ) throws -> Float {
        let fixtureLayers = try QwenTextFixtureSupport.value(fixture, at: path + ["layers"])
        guard let expectedLayers = fixtureLayers as? [Any], expectedLayers.count == actual.layers.count else {
            Issue.record("\(label): state layer count mismatch")
            return 0
        }
        #expect(actual.sequenceLength == 5, "\(label): sequence length")
        var maximum: Float = 0
        for (index, layer) in actual.layers.enumerated() {
            let expectedLayer = expectedLayers[index]
            switch layer {
            case let .linear(history, recurrent):
                maximum = max(maximum, assertClose(
                    history,
                    try QwenTextFixtureSupport.tensorValues(expectedLayer, at: ["convolutionState"]),
                    absolute: absolute, relative: relative,
                    label: "\(label) layer \(index) convolution"))
                maximum = max(maximum, assertClose(
                    recurrent,
                    try QwenTextFixtureSupport.tensorValues(expectedLayer, at: ["recurrentState"]),
                    absolute: absolute, relative: relative,
                    label: "\(label) layer \(index) recurrent"))
            case let .full(key, value):
                maximum = max(maximum, assertClose(
                    key,
                    try QwenTextFixtureSupport.tensorValues(expectedLayer, at: ["keyState"]),
                    absolute: absolute, relative: relative,
                    label: "\(label) layer \(index) key"))
                maximum = max(maximum, assertClose(
                    value,
                    try QwenTextFixtureSupport.tensorValues(expectedLayer, at: ["valueState"]),
                    absolute: absolute, relative: relative,
                    label: "\(label) layer \(index) value"))
            }
        }
        return maximum
    }
}

/// Metal completion and latch ordering keep this buffer exclusively owned by
/// the producer task until task.value returns; tests read it only afterward.
private struct InjectedHybridFailure: Error, Equatable, Sendable {}

/// The event is signalled only after the cancellation test releases the hook;
/// this keeps the committed GPU work unfinished without sleeping.
private final class TestSendableCommandBuffer: @unchecked Sendable {
    let buffer: MTLCommandBuffer

    init(_ buffer: MTLCommandBuffer) {
        self.buffer = buffer
    }
}

private actor CurrentLayer {
    private var value = -1

    func set(_ layer: Int) {
        value = layer
    }

    func isCurrent(_ layer: Int) -> Bool {
        value == layer
    }
}

private actor OnceFlag {
    private var consumed = false

    func take() -> Bool {
        guard !consumed else { return false }
        consumed = true
        return true
    }
}

private final class TestSendableSharedEvent: @unchecked Sendable {
    let event: MTLSharedEvent

    init(_ event: MTLSharedEvent) {
        self.event = event
    }
}

private actor FailureSwitch {
    private var shouldFail = true

    func consume() -> Bool {
        guard shouldFail else { return false }
        shouldFail = false
        return true
    }
}

private actor StageRecorder {
    private var recorded: [String] = []

    func append(_ stage: String) {
        recorded.append(stage)
    }

    func values() -> [String] {
        recorded
    }
}

private final class TestSendableBuffer: @unchecked Sendable {
    let buffer: MTLBuffer

    init(_ buffer: MTLBuffer) {
        self.buffer = buffer
    }
}

private actor AsyncLatch {
    private var signaled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func signal() {
        guard !signaled else { return }
        signaled = true
        let pending = waiters
        waiters.removeAll(keepingCapacity: false)
        for waiter in pending { waiter.resume() }
    }

    func wait() async {
        if signaled { return }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }
}

private func fillUInt16Buffer(_ buffer: MTLBuffer, count: Int, with value: UInt16) {
    let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
    for index in 0..<count { pointer[index] = value }
}

private func allUInt16BufferValuesEqual(_ buffer: MTLBuffer, count: Int, to value: UInt16) -> Bool {
    let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
    return (0..<count).allSatisfy { pointer[$0] == value }
}

private func anyUInt16BufferValueDiffers(_ buffer: MTLBuffer, count: Int, from value: UInt16) -> Bool {
    let pointer = buffer.contents().assumingMemoryBound(to: UInt16.self)
    return (0..<count).contains { pointer[$0] != value }
}

@discardableResult
private func assertClose(
    _ actual: [Float], _ expected: [Float], absolute: Float, relative: Float, label: String
) -> Float {
    guard actual.count == expected.count else {
        Issue.record("\(label): count \(actual.count) != expected \(expected.count)")
        return 0
    }
    let maxExpected = expected.map { abs($0) }.max() ?? 0
    let maxActual = actual.map { abs($0) }.max() ?? 0
    let scale = max(maxExpected, maxActual)
    let limit = absolute + relative * scale
    let maximum = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    #expect(maximum <= limit, "\(label): max error \(maximum) > \(limit)")
    return maximum
}
