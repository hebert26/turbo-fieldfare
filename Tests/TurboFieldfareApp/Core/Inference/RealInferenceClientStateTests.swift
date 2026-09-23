import CryptoKit
import Foundation
import Testing
import Tokenizers
import TurboFieldfareDecodeProtocol
import TurboFieldfareFormat
@testable import TurboFieldfare
@testable import TurboFieldfareAppCore

/// Model-free state coverage for the real client: load failure surfaces
/// before any network or Metal work, idle cancel is a no-op, and a bad
/// request fails the stream with a typed error.
@Suite struct RealInferenceClientStateTests {
    @Test func generationRegistryScopesTerminationToOwningID() async {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first))
        registry.clear(first)
        #expect(registry.reserve(second))
        let secondTask = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(secondTask, to: second)

        #expect(registry.take(first) == nil)
        #expect(!secondTask.isCancelled)
        registry.take(second)?.cancel()
        #expect(secondTask.isCancelled)
    }

    @Test func generationRegistryRejectsConcurrentReservationAndClearsByOwner() {
        let registry = GenerationTaskRegistry()
        let first = UUID()
        let second = UUID()
        #expect(registry.reserve(first))
        #expect(!registry.reserve(second))
        registry.clear(second)
        #expect(!registry.reserve(second))
        registry.clear(first)
        #expect(registry.reserve(second))
        registry.clear(second)
    }

    @Test func generationRegistryCancelsTaskAttachedAfterReservationEnded() async {
        let registry = GenerationTaskRegistry()
        let id = UUID()
        #expect(registry.reserve(id))
        registry.clear(id)
        let task = Task<Void, Never> {
            do { try await Task.sleep(for: .seconds(10)) } catch {}
        }
        registry.attach(task, to: id)
        #expect(task.isCancelled)
        let next = UUID()
        #expect(registry.reserve(next))
        registry.clear(next)
    }

    @Test func generationRunnerPolicyKeepsFusionHeadForPureGreedyChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "hello",
            temperature: 0,
            repetitionPenalty: 1)

        #expect(!RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func generationRunnerPolicyForcesLogitsForSamplingChunkedPrefill() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "hello",
            temperature: 0.7,
            repetitionPenalty: 1)

        #expect(RealInferenceSession.forceLogitsHead(for: request))
    }

    @Test func generationConfigCarriesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "hello")

        let config = RealInferenceSession.generationConfig(for: request)
        #expect(config.temperature == 0.2)
        #expect(config.topK == 64)
        #expect(config.topP == 0.95)
        #expect(config.repetitionPenalty == 1)
    }

    @Test func tokenizerDirectoryCacheReloadsOnlyWhenModelDirectoryChanges() {
        var cache = TokenizerDirectoryCache()
        let first = URL(fileURLWithPath: "/tmp/first.gturbo")
        let second = URL(fileURLWithPath: "/tmp/second.gturbo")

        #expect(cache.shouldReload(for: first))
        cache.markLoaded(for: first)
        #expect(!cache.shouldReload(for: first))
        #expect(cache.shouldReload(for: second))
        cache.clear()
        #expect(cache.shouldReload(for: first))
    }

    @Test func generateWithoutLoadedModelFailsWithoutPartialDiagnostics() async throws {
        let client = RealInferenceClient()
        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-prefill-off-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDirectory,
                                                withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let request = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "hello",
            runtimeOptions: AppRuntimeOptions(prefillEnabled: false))

        var failure: AppInferenceError?
        var partial: AppDiagnostics?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, let diagnostics) = event {
                    failure = error
                    partial = diagnostics
                }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }

        #expect(failure != nil)
        #expect(partial == nil)
        #expect(client.currentExpertCacheBytes == nil)
    }

    @Test func ensureLoadedFailsFastForMissingDirectory() async {
        let client = RealInferenceClient()
        var states: [AppModelLoadState] = []
        let recorder = StateRecorder()

        await #expect(throws: AppInferenceError.self) {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
                maxContextTokens: 1024,
                options: AppRuntimeOptions(),
                forceLogitsHead: false,
                onState: { recorder.append($0) })
        }
        states = recorder.snapshot()
        #expect(states.first == .loading(.validatingDirectory))
        #expect(states.last?.isFailed == true)
        #expect(!states.contains(.loading(.tokenizer)))
        #expect(client.currentExpertCacheBytes == nil)
    }

    @Test func generateWithMissingDirectoryFailsStream() async {
        let client = RealInferenceClient()
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            prompt: "hello")

        var failure: AppInferenceError?
        do {
            for try await event in client.generate(request) {
                if case .failed(let error, _) = event { failure = error }
            }
        } catch let error as AppInferenceError {
            failure = failure ?? error
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(failure == .modelNotFound("/nonexistent/model.gturbo"))
        #expect(client.currentExpertCacheBytes == nil)
    }

    @Test func prefillFailureDiagnosticsMarksUnsupportedModeAndReason() {
        let config = PrefillRuntimeConfig.production(chunkTokens: 32)

        let diagnostics = RealInferenceSession.prefillFailureDiagnostics(
            config: config,
            kvStorageMode: .fp16,
            reason: "chunked prefill synthetic unsupported diagnostic")

        #expect(diagnostics.requestedMode == .chunked)
        #expect(diagnostics.executedMode == .unsupported)
        #expect(diagnostics.chunkCompleteness == .unsupported)
        #expect(diagnostics.kvStorageMode == .fp16)
        #expect(diagnostics.unsupportedReason?.contains("synthetic unsupported") == true)
    }

    @Test func cancelWhenIdleIsNoOp() {
        let client = RealInferenceClient()
        client.cancel()
        client.cancel()
    }

    @Test func unloadWhenIdleIsSafe() async {
        let client = RealInferenceClient()
        await client.unload()
    }
}

@Suite(.serialized) struct RealInferenceClientQwenRoutingTests {
    @Test func sourceRelativeFixtureHasPinnedFileAndVirtualAggregateIdentity() throws {
        let data = try AppQwenFixtureSupport.data()
        #expect(AppQwenFixtureSupport.sha256(data)
            == "e07372907e09f7deab72abd64f417b6f514953a098ca51d53845dc832ae6b1ec")
        let records = try QwenTextFixtureRecords.decode(jsonData: data)
        #expect(records.virtualFileAggregateSHA256
            == "d4edeb3c98db5c084b6cbf9cf38b56859c717c6d0c20c522f5e1382ed39db7c6")
    }

    @Test func preparedQwenRouteUsesStoredConversationAndPublishesLogicalBytes() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }

        #expect(await client.hasOpenConversation)
        // The tiny fixture has four routed layers, eight cache slots per
        // layer, and a 16 KiB expert stride. Assert the real Metal buffer
        // ownership reported by the installed runtime.
        #expect(client.currentExpertCacheBytes ==
                AppQwenFixtureSupport.expectedExpertCacheBytes)
        let initialBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(initialBytes > 0)

        let turn = TokenizedConversationTurn(
            promptTokenIDs: [1, 4], generatedTokenIDs: [7])
        var events: [TokenizedConversationEvent] = []
        for try await event in client.generatePreparedQwen(turn) {
            events.append(event)
        }
        guard case .finished(let result) = try #require(events.last) else {
            Issue.record("prepared Qwen route did not finish with a result")
            return
        }
        #expect(result.reason == .complete)
        #expect(result.acceptedGeneratedTokenIDs == [7])
        #expect(result.metrics.retainedTokenIDs == [1, 4, 7])
        #expect(result.metrics.consumedTokenCount == 2)
        #expect(result.metrics.pendingTokenCount == 1)
        #expect(result.metrics.logicalStateBytes > initialBytes)
        let committedBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(committedBytes == result.metrics.logicalStateBytes)
        #expect(await client.conversationTokenCount == 3)
        #expect(client.currentExpertCacheBytes ==
                AppQwenFixtureSupport.expectedExpertCacheBytes)

        await client.resetConversation()
        #expect(await client.conversationTokenCount == 0)
        #expect(client.currentConversationLogicalStateBytes != nil)
        await client.unload()
        #expect(client.currentConversationLogicalStateBytes == nil)
        #expect(client.currentExpertCacheBytes == nil)
        let hasConversation = await client.hasOpenConversation
        #expect(!hasConversation)
    }

    @Test func ordinaryQwenRouteReusesTheLoadedSessionWithoutDuplicatePrefill() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }

        let first = try await finishedDiagnostics(
            client: client,
            request: AppGenerationRequest(
                modelDirectory: modelDirectory,
                prompt: "ordinary first turn",
                maxNewTokens: 1,
                maxContextTokens: 4_096))
        let second = try await finishedDiagnostics(
            client: client,
            request: AppGenerationRequest(
                modelDirectory: modelDirectory,
                prompt: "ordinary retained continuation",
                maxNewTokens: 1,
                maxContextTokens: 4_096,
                continuesConversation: true))

        #expect(first.stopReason == .maxTokens)
        #expect(second.stopReason == .maxTokens)
        #expect(first.conversationTokens ?? 0 > 0)
        #expect(second.cachedPromptTokens ?? 0 > 0)
        #expect(second.computedPrefillTokens ?? 0 > 0)
        #expect(await client.hasOpenConversation)
    }

    @Test func ordinaryQwenHardCancellationRollsBackAndAllowsRetry() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let beforeBytes = try #require(client.currentConversationLogicalStateBytes)
        let request = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: String(repeating: "cancel this ordinary Qwen turn ", count: 24),
            maxNewTokens: 16,
            maxContextTokens: 4_096)

        let operation = Task { () -> Bool in
            var sawCancelled = false
            do {
                for try await event in client.generate(request) {
                    if case .prefillProgress = event { client.cancel() }
                    if case .cancelled = event { sawCancelled = true }
                }
            } catch let error as AppInferenceError {
                sawCancelled = sawCancelled || error == .cancelled
            } catch {
                sawCancelled = false
            }
            return sawCancelled
        }
        #expect(await operation.value)
        #expect(client.currentConversationLogicalStateBytes == beforeBytes)
        #expect(await client.conversationTokenCount == 0)

        let retry = try await finishedDiagnostics(
            client: client,
            request: AppGenerationRequest(
                modelDirectory: modelDirectory,
                prompt: "retry ordinary Qwen after cancellation",
                maxNewTokens: 1,
                maxContextTokens: 4_096))
        #expect(retry.stopReason == .maxTokens)
    }

    @Test func ordinaryQwenSoftStopCommitsAcceptedTokenState() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let stop = AppGenerationStop()
        var terminal: AppDiagnostics?
        for try await event in client.generate(
            AppGenerationRequest(
                modelDirectory: modelDirectory,
                prompt: "stop after an accepted ordinary token",
                maxNewTokens: 16,
                maxContextTokens: 4_096),
            measurementCapture: nil,
            generationStop: stop) {
            if case .token = event { stop.requestStop() }
            if case .finished(let diagnostics) = event { terminal = diagnostics }
        }
        let diagnostics = try #require(terminal)
        #expect(diagnostics.stopReason == .cancelled)
        #expect(diagnostics.conversationTokens ?? 0 > 0)
    }

    @Test func ordinaryQwenImageAttachmentKeepsIDAndURLBoundToTheTurn() async throws {
        let recorder = AppQwenImagePlannerRecorder()
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client(
            imagePlannerRecorder: recorder)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let imageURL = modelDirectory.appendingPathComponent("fixture-image.bin")
        let imageData = Data("fixture-image".utf8)
        try imageData.write(to: imageURL)
        let imageID = UUID()
        let attachment = AppImageAttachment(
            id: imageID, fileURL: imageURL, displayName: "fixture-image.bin",
            encodedBytes: imageData.count, sha256: AppQwenFixtureSupport.sha256(imageData))

        let diagnostics = try await finishedDiagnostics(
            client: client,
            request: AppGenerationRequest(
                modelDirectory: modelDirectory,
                prompt: "describe the attached fixture image",
                imageAttachments: [attachment],
                maxNewTokens: 1,
                maxContextTokens: 4_096))
        #expect(diagnostics.stopReason == .maxTokens)
        #expect(recorder.calls == [[imageID.uuidString]])
        #expect(recorder.boundURLs.first?[imageID.uuidString] == imageURL)
    }

    @Test func ordinaryQwenResultsAndCheckpointResumeAreSingleUseWithoutPrefillReplay() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client(
            decodeFixture: .toolCall)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let tool = AppToolDefinition(
            name: "lookup", description: "synthetic decode fixture tool",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "query": .object(["type": .string("string")]),
                ]),
                "required": .array([.string("query")]),
                "additionalProperties": .bool(false),
            ]))
        let userRequest = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "call lookup",
            maxNewTokens: 4,
            maxContextTokens: 4_096,
            continuesConversation: true,
            toolTurn: .user(
                developerPrompt: "Use the supplied lookup tool.", tools: [tool]))

        var userCall: AppToolCall?
        var userDiagnostics: AppDiagnostics?
        for try await event in client.generate(userRequest) {
            if case .toolCall(let call) = event { userCall = call }
            if case .finished(let diagnostics) = event { userDiagnostics = diagnostics }
        }
        let firstCall = try #require(userCall)
        #expect(firstCall.name == "lookup")
        #expect(userDiagnostics?.stopReason == .toolCalls)

        let resultRequest = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "",
            maxNewTokens: 4,
            maxContextTokens: 4_096,
            continuesConversation: true,
            toolTurn: .results([
                AppToolResult(
                    callID: firstCall.id, name: firstCall.name,
                    content: "synthetic lookup result")
            ]))
        var resultCall: AppToolCall?
        var resultDiagnostics: AppDiagnostics?
        for try await event in client.generate(resultRequest) {
            if case .toolCall(let call) = event { resultCall = call }
            if case .finished(let diagnostics) = event { resultDiagnostics = diagnostics }
        }
        let pendingCall = try #require(resultCall)
        #expect(resultDiagnostics?.stopReason == .toolCalls)
        #expect(resultDiagnostics?.cachedPromptTokens ?? 0 > 0)

        let checkpointID = UUID()
        let checkpoint = DecodeContextCheckpointRequest(
            checkpointID: checkpointID,
            sourceEpoch: UUID(),
            sourceTurnIndex: 1,
            replacementEpoch: UUID(),
            pendingCall: DecodeToolCall(
                id: pendingCall.id, name: pendingCall.name,
                argumentsJSON: try pendingCall.arguments.encoded()),
            result: DecodeToolResult(
                callID: pendingCall.id, name: pendingCall.name,
                content: "checkpoint result"),
            record: "checkpoint record",
            commit: true,
            force: true,
            generationAllowance: 128,
            finalAnswerAllowance: 128,
            permitsScreenshot: false)
        let receipt = try await client.contextCheckpoint(checkpoint)
        #expect(receipt.committed)
        #expect(receipt.replacementPromptTokens ?? 0 > 0)
        #expect(receipt.retainedImageCount == 0)

        let resumeRequest = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "",
            maxNewTokens: 4,
            maxContextTokens: 4_096,
            continuesConversation: true,
            toolTurn: .checkpoint(checkpointID))
        var resumePrefill: [(Int, Int)] = []
        var resumeDiagnostics: AppDiagnostics?
        for try await event in client.generate(resumeRequest) {
            if case .prefillProgress(let done, let total) = event {
                resumePrefill.append((done, total))
            }
            if case .finished(let diagnostics) = event { resumeDiagnostics = diagnostics }
        }
        #expect(resumeDiagnostics?.stopReason == .toolCalls)
        #expect(resumeDiagnostics?.computedPrefillTokens == 0)
        #expect(resumeDiagnostics?.cachedPromptTokens ?? 0 > 0)
        #expect(!resumePrefill.isEmpty)
        #expect(resumePrefill.allSatisfy { $0.0 == 0 && $0.1 == 0 })

        var secondResumeFailed = false
        do {
            for try await event in client.generate(resumeRequest) {
                if case .failed = event { secondResumeFailed = true }
            }
        } catch {
            secondResumeFailed = true
        }
        #expect(secondResumeFailed)
    }

    @Test func ordinaryQwenUnloadDropsConversationAfterTheGenerationBarrier() async throws {
        let entered = AppAsyncLatch()
        let prefillSeen = AppProgressBlocker()
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let operation = Task {
            var sawPrefill = false
            do {
                for try await event in client.generate(AppGenerationRequest(
                    modelDirectory: modelDirectory,
                    prompt: String(repeating: "lifecycle barrier ", count: 32),
                    maxNewTokens: 16,
                    maxContextTokens: 4_096)) {
                    if case .prefillProgress(let done, _) = event, done > 0 {
                        sawPrefill = true
                        await prefillSeen.claim()
                        await entered.signal()
                    }
                }
            } catch {
                // Unload may cancel the active request while it waits for the
                // same generation barrier. The assertion is on post-barrier state.
            }
            await entered.signal()
            return sawPrefill
        }
        await entered.wait()
        #expect(await prefillSeen.isClaimed,
                "ordinary Qwen generation failed before publishing prefill progress")
        guard await prefillSeen.isClaimed else {
            _ = await operation.value
            return
        }
        await client.unload()
        _ = await operation.value
        #expect(!(await client.hasOpenConversation))
        #expect(client.currentConversationLogicalStateBytes == nil)
        #expect(client.currentExpertCacheBytes == nil)
    }

    @Test func preparedQwenCancellationRollsBackAndReleasesTheSharedRegistry() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let initialBytes = try #require(client.currentConversationLogicalStateBytes)
        let stop = AppGenerationStop()
        stop.requestStop()
        var didThrow = false
        do {
            for try await _ in client.generatePreparedQwen(
                TokenizedConversationTurn(promptTokenIDs: [1], generatedTokenIDs: [4]),
                generationStop: stop) {}
        } catch {
            didThrow = true
        }
        #expect(didThrow)
        #expect(await client.conversationTokenCount == 0)
        let rolledBackBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(rolledBackBytes == initialBytes)

        // A cancelled prepared call must release the same registry slot used by
        // normal generation and leave the stored Qwen conversation reusable.
        var finished = false
        for try await event in client.generatePreparedQwen(
            TokenizedConversationTurn(promptTokenIDs: [1], generatedTokenIDs: [4])) {
            if case .finished = event { finished = true }
        }
        #expect(finished)
    }

    @Test func preparedQwenCheckpointStopRollsBackThroughRegistryAndAllowsCleanReuse() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let initialBytes = try #require(client.currentConversationLogicalStateBytes)
        let stop = AppGenerationStop()
        stop.requestStop()
        let request = TokenizedCheckpointRequest(
            retainedTokenIDs: [1, 4], reason: .capacity)
        var didThrow = false
        do {
            for try await _ in client.rebuildPreparedQwenCheckpoint(
                request, generationStop: stop) {}
        } catch {
            didThrow = true
        }
        #expect(didThrow)
        #expect(await client.conversationTokenCount == 0)
        let rolledBackBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(rolledBackBytes == initialBytes)

        var events: [TokenizedCheckpointEvent] = []
        for try await event in client.rebuildPreparedQwenCheckpoint(request) {
            events.append(event)
        }
        guard case .finished(let result) = try #require(events.last) else {
            Issue.record("clean checkpoint did not finish after Stop rollback")
            return
        }
        #expect(result.committed)
        #expect(result.metrics.retainedTokenIDs == [1, 4])
        #expect(await client.conversationTokenCount == 2)
    }

    @Test func preparedQwenCheckpointCancelAfterReplayMutationRestoresAndReusesTheSession() async throws {
        let clientReference = AppClientReference()
        let firstCheckpoint = AppProgressBlocker()
        let entered = AppAsyncLatch()
        let release = AppAsyncLatch()
        let observed = AppObservedBytes()
        let observer: @Sendable (TokenizedConversationProgress) async -> Void = { progress in
            guard case .checkpoint(let done, _) = progress, done == 1,
                  await firstCheckpoint.claim() else { return }
            observed.record(await clientReference.logicalBytes())
            await clientReference.cancel()
            await entered.signal()
            await release.wait()
        }
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client(
            qwenProgressObserver: observer)
        await clientReference.install(client)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        for try await _ in client.generatePreparedQwen(
            TokenizedConversationTurn(promptTokenIDs: [1], generatedTokenIDs: [4])) {}
        let committedBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(await client.conversationTokenCount == 2)

        let outcome = AppTaskOutcome()
        let request = TokenizedCheckpointRequest(
            retainedTokenIDs: [1, 4], reason: .capacity)
        let operation = Task {
            do {
                for try await _ in client.rebuildPreparedQwenCheckpoint(request) {}
                await outcome.finish(threw: false)
                await entered.signal()
            } catch {
                await outcome.finish(threw: true)
                await entered.signal()
            }
        }

        await entered.wait()
        let observedBytes = try #require(observed.value)
        #expect(observedBytes == committedBytes)
        let activeBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(activeBytes == committedBytes)
        #expect(await client.conversationTokenCount == 2)
        #expect(await outcome.finished == false)

        await release.signal()
        _ = await operation.value
        #expect(await outcome.finished)
        #expect(await outcome.threw)
        let restoredBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(restoredBytes == committedBytes)
        #expect(await client.conversationTokenCount == 2)

        var events: [TokenizedCheckpointEvent] = []
        for try await event in client.rebuildPreparedQwenCheckpoint(request) {
            events.append(event)
        }
        guard case .finished(let result) = try #require(events.last) else {
            Issue.record("checkpoint could not reuse the session after cancellation")
            return
        }
        #expect(result.committed)
        #expect(result.metrics.retainedTokenIDs == [1, 4])
        #expect(result.metrics.pendingTokenCount == 0)
    }

    @Test func preparedQwenLogicalBytesStayCommittedWhileActiveAndHardCancelRollsBack() async throws {
        let clientReference = AppClientReference()
        let entered = AppAsyncLatch()
        let release = AppAsyncLatch()
        let blocker = AppProgressBlocker()
        let observed = AppObservedBytes()
        let observer: @Sendable (TokenizedConversationProgress) async -> Void = { progress in
            guard case .prefill(let done, let total) = progress,
                  done == 1, total == 2,
                  await blocker.claim() else { return }
            observed.record(await clientReference.logicalBytes())
            await entered.signal()
            await release.wait()
        }
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client(
            qwenProgressObserver: observer)
        await clientReference.install(client)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let initialBytes = try #require(client.currentConversationLogicalStateBytes)
        let outcome = AppTaskOutcome()
        let operation = Task {
            do {
                for try await _ in client.generatePreparedQwen(
                    TokenizedConversationTurn(promptTokenIDs: [1, 4], generatedTokenIDs: [7])) {}
                await outcome.finish(threw: false)
                await entered.signal()
            } catch {
                await outcome.finish(threw: true)
                await entered.signal()
            }
        }

        await entered.wait()
        // The transaction's working aggregate is active, but the client
        // publishes only the last committed bytes/tokens.
        let observedBytes = try #require(observed.value)
        #expect(observedBytes == initialBytes)
        let activeBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(activeBytes == initialBytes)
        #expect(await client.conversationTokenCount == 0)
        #expect(await outcome.finished == false)

        await clientReference.cancel()
        #expect(await outcome.finished == false)
        await release.signal()
        _ = await operation.value
        #expect(await outcome.finished)
        #expect(await outcome.threw)
        let rolledBackBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(rolledBackBytes == initialBytes)
        #expect(await client.conversationTokenCount == 0)

        // Cancellation must release the shared client registry and leave the
        // same stored session reusable.
        var finished = false
        for try await event in client.generatePreparedQwen(
            TokenizedConversationTurn(promptTokenIDs: [1], generatedTokenIDs: [4])) {
            if case .finished = event { finished = true }
        }
        #expect(finished)
    }

    @Test func preparedQwenSoftStopAfterDecodeBeginsCommitsAcceptedPendingToken() async throws {
        let stop = AppGenerationStop()
        let observer: @Sendable (TokenizedConversationProgress) async -> Void = { progress in
            if case .accepted(index: 0, tokenID: 4) = progress {
                // This callback runs after prefill marked decodeBegan, so
                // Stop is cooperative and commits the accepted pending ID.
                stop.requestStop()
            }
        }
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client(
            qwenProgressObserver: observer)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let initialBytes = try #require(client.currentConversationLogicalStateBytes)
        var events: [TokenizedConversationEvent] = []
        for try await event in client.generatePreparedQwen(
            TokenizedConversationTurn(promptTokenIDs: [1], generatedTokenIDs: [4, 14]),
            generationStop: stop) {
            events.append(event)
        }
        guard case .finished(let result) = try #require(events.last) else {
            Issue.record("soft Stop did not finish with a prepared result")
            return
        }
        #expect(result.reason == .softStop)
        #expect(result.acceptedGeneratedTokenIDs == [4])
        #expect(result.metrics.retainedTokenIDs == [1, 4])
        #expect(result.metrics.consumedTokenCount == 1)
        #expect(result.metrics.pendingTokenCount == 1)
        #expect(result.metrics.logicalStateBytes > initialBytes)
        #expect(await client.conversationTokenCount == 2)
    }

    @Test func qwenCheckpointPolicyAndOrdinaryCodecRouting() async throws {
        let (client, modelDirectory) = try await AppQwenFixtureSupport.client()
        defer { try? FileManager.default.removeItem(at: modelDirectory) }

        let initialBytes = try #require(client.currentConversationLogicalStateBytes)
        let noOpRequest = TokenizedCheckpointRequest(
            retainedTokenIDs: [1], reason: .sustainedSlowDecode)
        var noOpEvents: [TokenizedCheckpointEvent] = []
        for try await event in client.rebuildPreparedQwenCheckpoint(noOpRequest) {
            noOpEvents.append(event)
        }
        guard case .finished(let noOp) = try #require(noOpEvents.last) else {
            Issue.record("sustained-slow checkpoint did not finish")
            return
        }
        #expect(!noOp.needed)
        #expect(!noOp.committed)
        #expect(noOp.metrics.retainedTokenIDs.isEmpty)
        #expect(noOp.metrics.logicalStateBytes == initialBytes)
        let noOpBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(noOpBytes == initialBytes)

        let generated = TokenizedConversationTurn(
            promptTokenIDs: [1, 4], generatedTokenIDs: [7])
        for try await _ in client.generatePreparedQwen(generated) {}
        let capacityRequest = TokenizedCheckpointRequest(
            retainedTokenIDs: [1, 4, 7], reason: .capacity)
        var capacityEvents: [TokenizedCheckpointEvent] = []
        for try await event in client.rebuildPreparedQwenCheckpoint(capacityRequest) {
            capacityEvents.append(event)
        }
        guard case .finished(let capacity) = try #require(capacityEvents.last) else {
            Issue.record("capacity checkpoint did not finish")
            return
        }
        #expect(capacity.needed)
        #expect(capacity.committed)
        #expect(capacity.metrics.retainedTokenIDs == [1, 4, 7])
        #expect(capacity.metrics.consumedTokenCount == 3)
        #expect(capacity.metrics.pendingTokenCount == 0)
        let committedBytes = try #require(client.currentConversationLogicalStateBytes)
        #expect(capacity.metrics.logicalStateBytes == committedBytes)

        let stringRequest = AppGenerationRequest(
            modelDirectory: modelDirectory, prompt: "ordinary Qwen text",
            maxNewTokens: 1, maxContextTokens: 4_096)
        let ordinary = try await finishedDiagnostics(
            client: client, request: stringRequest)
        #expect(ordinary.stopReason == .maxTokens)
        #expect(ordinary.computedPrefillTokens ?? 0 > 0)
        #expect(ordinary.conversationTokens ?? 0 > 0)
        let retainedBytesBeforeTools = try #require(
            client.currentConversationLogicalStateBytes)

        let orderedTools = [
            AppToolDefinition(
                name: "first", description: "first ordered schema",
                parameters: .object(["type": .string("object")])),
            AppToolDefinition(
                name: "second", description: "second ordered schema",
                parameters: .object(["type": .string("object")])),
        ]
        let changedToolRequest = AppGenerationRequest(
            modelDirectory: modelDirectory, prompt: "use the ordered tools",
            maxNewTokens: 1, maxContextTokens: 4_096,
            continuesConversation: true,
            toolTurn: .user(
                developerPrompt: "Follow the host tool policy.", tools: orderedTools))
        var changedToolError: AppInferenceError?
        do {
            for try await event in client.generate(changedToolRequest) {
                if case .failed(let error, _) = event { changedToolError = error }
            }
        } catch let error as AppInferenceError {
            changedToolError = changedToolError ?? error
        } catch {
            Issue.record("unexpected changed-tool error: \(error)")
        }
        #expect(changedToolError == .invalidRequest(
            "tool definitions changed inside a retained conversation"))
        #expect(client.currentConversationLogicalStateBytes == retainedBytesBeforeTools)
        #expect(await client.conversationTokenCount == (ordinary.conversationTokens ?? 0))

        // AppGenerationRequest requires the retained app-conversation gate for
        // a tool turn even when the underlying Qwen session is empty. Resetting
        // first proves the fresh-session route without weakening that policy.
        await client.resetConversation()
        let freshToolRequest = AppGenerationRequest(
            modelDirectory: modelDirectory, prompt: "use the ordered tools",
            maxNewTokens: 1, maxContextTokens: 4_096,
            continuesConversation: true,
            toolTurn: .user(
                developerPrompt: "Follow the host tool policy.", tools: orderedTools))
        let toolUserDiagnostics = try await finishedDiagnostics(
            client: client, request: freshToolRequest)
        #expect(toolUserDiagnostics.stopReason == .maxTokens)
        #expect((toolUserDiagnostics.cachedPromptTokens ?? 0) == 0)
        #expect(toolUserDiagnostics.computedPrefillTokens ?? 0 > 0)
    }
}

private func finishedDiagnostics(
    client: RealInferenceClient, request: AppGenerationRequest
) async throws -> AppDiagnostics {
    var finished: AppDiagnostics?
    for try await event in client.generate(request) {
        if case .finished(let diagnostics) = event { finished = diagnostics }
    }
    return try #require(finished)
}

/// A test-only decoder fixture. It delegates encoding and chat-template work to
/// the authentic Qwen tokenizer, then gives selected tiny fixture IDs complete
/// text or tool-call chunks. The runtime's own incremental decoder and parser
/// still consume those chunks through the ordinary AppCore route.
private struct AppQwenSyntheticDecodeFixture: Sendable {
    let generatedTokenIDs: [Int32]
    let endTokenID: Int32?
    let tokenOverrides: [Int: String]

    var addedTokens: [QwenAddedToken] {
        tokenOverrides.map { id, content in
            QwenAddedToken(id: Int32(id), content: content, isSpecial: false)
        }
    }

    static let ordinary = Self(
        generatedTokenIDs: [1, 12, 6],
        endTokenID: nil,
        tokenOverrides: [
            1: "fixture",
            12: " response",
            6: " text",
        ])

    static let toolCall = Self(
        generatedTokenIDs: [1, 12, 6, 18],
        endTokenID: 18,
        tokenOverrides: [
            1: "<tool_call>\n",
            12: "<function=lookup>\n<parameter=query>\nsnow\n</parameter>\n</function>\n",
            6: "</tool_call>",
            18: "<|fixture_eos|>",
        ])
}

private struct AppQwenTokenizerOverride: Tokenizer {
    let base: any Tokenizer
    let overrides: [Int: String]

    func tokenize(text: String) -> [String] { base.tokenize(text: text) }

    func encode(text: String) -> [Int] { base.encode(text: text) }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        base.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        overrides[id] ?? base.convertIdToToken(id)
    }

    var bosToken: String? { base.bosToken }
    var bosTokenId: Int? { base.bosTokenId }
    var eosToken: String? { base.eosToken }
    var eosTokenId: Int? { base.eosTokenId }
    var unknownToken: String? { base.unknownToken }
    var unknownTokenId: Int? { base.unknownTokenId }
    var hasChatTemplate: Bool { base.hasChatTemplate }

    func applyChatTemplate(messages: [Message]) throws -> [Int] {
        try base.applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        try base.applyChatTemplate(messages: messages, tools: tools)
    }

    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }

    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument
    ) throws -> [Int] {
        try base.applyChatTemplate(messages: messages, chatTemplate: chatTemplate)
    }

    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try base.applyChatTemplate(messages: messages, chatTemplate: chatTemplate)
    }

    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, chatTemplate: chatTemplate,
            addGenerationPrompt: addGenerationPrompt, truncation: truncation,
            maxLength: maxLength, tools: tools)
    }

    func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?,
        tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, chatTemplate: chatTemplate,
            addGenerationPrompt: addGenerationPrompt, truncation: truncation,
            maxLength: maxLength, tools: tools, additionalContext: additionalContext)
    }
}

private enum AppQwenFixtureSupport {
    static func data() throws -> Data {
        try Data(contentsOf: fixtureURL)
    }

    static let expectedExpertCacheBytes: UInt64 = 4 * 8 * 16_384

    static func client(
        qwenProgressObserver: @escaping @Sendable (TokenizedConversationProgress) async -> Void = { _ in },
        imagePlannerRecorder: AppQwenImagePlannerRecorder? = nil,
        decodeFixture: AppQwenSyntheticDecodeFixture = .ordinary
    ) async throws -> (RealInferenceClient, URL) {
        let data = try data()
        let records = try QwenTextFixtureRecords.decode(jsonData: data)
        let context = try MetalContext()
        let model = try QwenTextModel.loadFixtureRecords(records, device: context.device)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-app-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false)
        let session = RealInferenceSession()
        let fixtureRequest = AppGenerationRequest(
            modelDirectory: directory, prompt: "fixture ordinary request")
        let key = SessionLoadKey(
            directory: directory, maxContext: 4_096,
            options: fixtureRequest.runtimeOptions,
            forceLogitsHead: RealInferenceSession.forceLogitsHead(for: fixtureRequest))
        let authenticTokenizer = try QwenTokenizer.loadOfficialSidecar(
            from: fixtureSourceDirectory)
        let fixtureTokenizer = AppQwenTokenizerOverride(
            base: authenticTokenizer.tokenizer,
            overrides: decodeFixture.tokenOverrides)
        let codec = QwenChatCodec(tokenizer: QwenTokenizer(
            tokenizer: fixtureTokenizer,
            addedTokens: authenticTokenizer.addedTokens + decodeFixture.addedTokens,
            effectiveSpecialTokenIDs: authenticTokenizer.effectiveSpecialTokenIDs))
        let state = try await QwenConversationState(
            model: model, context: context, maxContext: 4_096)
        let generation = try QwenConversationGenerationSession(
            fixtureModel: model,
            state: state,
            codec: codec,
            context: context,
            maxContext: 4_096,
            imagePlanner: imagePlannerRecorder.map { recorder in
                let imageContext = context
                return { orderedIDs, imagesByID in
                    recorder.record(orderedIDs: orderedIDs, imagesByID: imagesByID)
                    return try Self.imagePlan(
                        context: imageContext, orderedIDs: orderedIDs)
                }
            },
            fixtureTokenMapper: fixtureTokenMapper(),
            fixtureGeneratedTokenIDs: decodeFixture.generatedTokenIDs,
            fixtureEndTokenID: decodeFixture.endTokenID)
        try await session.installQwenFixture(
            model: model, state: state, codec: codec, generation: generation,
            key: key, context: context)
        return (RealInferenceClient(
            session: session, qwenProgressObserver: qwenProgressObserver), directory)
    }

    private static func fixtureTokenMapper() -> QwenConversationFixtureTokenMapper {
        let safeTextTokens: [Int32] = [0, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        return { composed in
            composed.enumerated().map { index, token in
                switch token {
                case 248_053: 5 // <|vision_start|>
                case 248_056: 3 // <|image_pad|>
                case 248_054: 6 // <|vision_end|>
                case 248_057: 4 // <|video_pad|>
                case 248_046: 2 // tokenizer EOS
                default: safeTextTokens[index % safeTextTokens.count]
                }
            }
        }
    }

    private static func imagePlan(
        context: MetalContext, orderedIDs: [String]
    ) throws -> QwenConversationFixtureImagePlan {
        let requestedAllocationBytes = orderedIDs.count * (
            32 * MemoryLayout<Float>.stride + 3 * MemoryLayout<Int32>.stride)
        let wire = GTurboQwenArchitectureV2(
            hiddenSize: 32, numLayers: 4,
            layerTypes: [.linearAttention, .linearAttention,
                          .linearAttention, .fullAttention],
            numAttentionHeads: 2, numKeyValueHeads: 1, headDimension: 16,
            attentionOutputGate: true, linearConvolutionKernel: 4,
            linearKeyHeads: 2, linearKeyHeadDimension: 4,
            linearValueHeads: 2, linearValueHeadDimension: 4,
            recurrentStateType: .fp32, partialRotaryFactor: 0.75,
            ropeTheta: 10_000_000, mropeInterleaved: true,
            mropeSections: [2, 1, 3], numberOfExperts: 10,
            expertsPerToken: 8, routedExpertIntermediateSize: 7,
            sharedExpertIntermediateSize: 6, vocabularySize: 19,
            tiedWordEmbeddings: false, hiddenActivation: "silu",
            bosTokenID: 1, eosTokenID: 2, imageTokenID: 3,
            videoTokenID: 4, visionStartTokenID: 5, visionEndTokenID: 6)
        return QwenConversationFixtureImagePlan(
            architecture: QwenArchConfig(wire: wire),
            visionConfig: QwenVisionConfig(
                outputHiddenSize: 32, allowsFixtureGeometry: true),
            mergedRows: Array(repeating: 1, count: orderedIDs.count),
            requestedAllocationBytes: requestedAllocationBytes,
            prepare: {
                try orderedIDs.enumerated().map { index, _ in
                    let grid = try QwenVisionGrid(temporal: 1, height: 2, width: 2)
                    let position = try QwenMRoPEPosition(temporal: 0, height: 0, width: 0)
                    let profile = GTurboQwenVisionProcessorProfileV2(
                        processorClass: "Qwen3VLProcessor",
                        imageProcessorType: "Qwen2VLImageProcessorFast",
                        patchSize: 16, temporalPatchSize: 2, spatialMergeSize: 2)
                    return try QwenVisionFeatures(
                        device: context.device,
                        features: [Float](repeating: Float(index + 1), count: 32),
                        positions: [position],
                        imageDigest: String(repeating: index == 0 ? "a" : "b", count: 64),
                        processorDigest: String(repeating: "c", count: 64),
                        profile: profile, grid: grid, hiddenSize: 32)
                }
            })
    }

    private static var fixtureSourceDirectory: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                "scratch/qwen3.6-35b-a3b/official-995ad96eacd98c81ed38be0c5b274b04031597b0",
                isDirectory: true)
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static var fixtureURL: URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.appendingPathComponent(
            "TurboFieldfare/Core/QwenFixtures/qwen36-tiny-text-model-fixtures.json")
    }
}

private final class AppQwenImagePlannerRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var orderedIDCalls: [[String]] = []
    private var boundURLCalls: [[String: URL]] = []

    func record(orderedIDs: [String], imagesByID: [String: URL]) {
        lock.lock()
        orderedIDCalls.append(orderedIDs)
        boundURLCalls.append(imagesByID)
        lock.unlock()
    }

    var calls: [[String]] {
        lock.lock()
        defer { lock.unlock() }
        return orderedIDCalls
    }

    var boundURLs: [[String: URL]] {
        lock.lock()
        defer { lock.unlock() }
        return boundURLCalls
    }
}

private actor AppClientReference {
    private var client: RealInferenceClient?

    func install(_ client: RealInferenceClient) {
        self.client = client
    }

    func cancel() {
        client?.cancel()
    }

    func logicalBytes() -> UInt64? {
        client?.currentConversationLogicalStateBytes
    }
}

private actor AppAsyncLatch {
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

private actor AppProgressBlocker {
    private var claimed = false

    func claim() -> Bool {
        guard !claimed else { return false }
        claimed = true
        return true
    }

    var isClaimed: Bool { claimed }
}

private final class AppObservedBytes: @unchecked Sendable {
    private let lock = NSLock()
    private var storedValue: UInt64? = nil

    func record(_ value: UInt64?) {
        lock.lock()
        storedValue = value
        lock.unlock()
    }

    var value: UInt64? {
        lock.lock()
        defer { lock.unlock() }
        return storedValue
    }
}

private actor AppTaskOutcome {
    private(set) var finished = false
    private(set) var threw = false

    func finish(threw: Bool) {
        self.threw = threw
        finished = true
    }
}

private final class StateRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppModelLoadState] = []

    func append(_ state: AppModelLoadState) {
        lock.lock()
        states.append(state)
        lock.unlock()
    }

    func snapshot() -> [AppModelLoadState] {
        lock.lock()
        defer { lock.unlock() }
        return states
    }
}
