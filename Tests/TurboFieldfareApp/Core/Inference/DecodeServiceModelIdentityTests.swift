import Darwin
import Foundation
import Testing
@testable import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

@Suite struct DecodeServiceModelIdentityTests {
    @Test func setupCancellationBeforeProcessRunDoesNotStartChild() throws {
        let worker = ServiceSetupWorker()
        worker.requestCancellation()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/true")

        #expect(throws: CancellationError.self) {
            try worker.runAndWait(process)
        }
        #expect(!process.isRunning)
    }

    @Test func setupCancellationBlocksTheNextConnectionAttempt() throws {
        let worker = ServiceSetupWorker()
        worker.requestCancellation()

        #expect(throws: CancellationError.self) {
            try worker.beforeConnectAttempt()
        }
    }

    @Test func setupCancellationAfterProcessRunTerminatesTheHelper() async throws {
        let entered = AwaitOnce()
        let finished = AwaitOnce()
        let result = ProcessRunResult()
        let worker = ServiceSetupWorker(
            checkpoints: ServiceSetupWorkerCheckpoints(
                afterProcessRun: { entered.signal() }))
        let input = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/cat")
        process.standardInput = input
        process.standardOutput = FileHandle.nullDevice
        let thread = Thread {
            do {
                try worker.runAndWait(process)
                result.record(nil)
            } catch {
                result.record(error)
            }
            finished.signal()
        }
        thread.name = "TurboFieldfare.Tests.SetupWorkerProcess"
        thread.start()

        await entered.wait()
        worker.requestCancellation()
        await finished.wait()
        #expect(result.wasCancellation)
        #expect(result.completionCount == 1)
        #expect(!process.isRunning)
        try? input.fileHandleForWriting.close()
        try? input.fileHandleForReading.close()
    }

    @Test func setupCancellationAtProbeHandoffSettlesTheContinuation() async throws {
        let entered = AwaitOnce()
        let releaseCheckpoint = DispatchSemaphore(value: 0)
        let worker = ServiceSetupWorker(
            checkpoints: ServiceSetupWorkerCheckpoints(
                beforeProbeHandoff: {
                    entered.signal()
                    _ = releaseCheckpoint.wait()
                }))
        let input = Pipe()
        let output = Pipe()
        try worker.beginProbe(
            input: input.fileHandleForWriting,
            output: output.fileHandleForReading)
        let result = ProcessRunResult()
        let finished = AwaitOnce()
        let thread = Thread {
            do {
                try worker.completeProbeHandoff()
                result.record(nil)
            } catch {
                result.record(error)
            }
            finished.signal()
        }
        thread.name = "TurboFieldfare.Tests.SetupWorkerProbeHandoff"
        thread.start()

        await entered.wait()
        worker.requestCancellation()
        releaseCheckpoint.signal()
        await finished.wait()
        #expect(result.wasCancellation)
        #expect(result.completionCount == 1)
        try? input.fileHandleForWriting.close()
        try? input.fileHandleForReading.close()
        try? output.fileHandleForWriting.close()
        try? output.fileHandleForReading.close()
    }

    @Test func malformedStartupAcknowledgementsFailClosedBeforeTransportAdmission() async throws {
        let nonce = UUID()
        let wrongGeneration = DecodeServiceEvent(
            kind: .lifetimeAcknowledged, generationID: UUID(),
            lifetimeNonce: nonce)
        let wrongLifetimeNonce = DecodeServiceEvent(
            kind: .lifetimeAcknowledged, generationID: nonce,
            lifetimeNonce: UUID())
        let variants: [(String, DecodeServiceEvent)] = [
            ("wrong kind", DecodeServiceEvent(
                kind: .ready, generationID: nonce, lifetimeNonce: nonce)),
            ("wrong generation", wrongGeneration),
            ("wrong nonce", wrongLifetimeNonce),
            ("loaded family", DecodeServiceEvent(
                kind: .lifetimeAcknowledged, generationID: nonce,
                lifetimeNonce: nonce, loadedFamily: .qwen3_6)),
            ("load id", DecodeServiceEvent(
                kind: .lifetimeAcknowledged, generationID: nonce,
                lifetimeNonce: nonce, loadID: UUID())),
            ("model identity", DecodeServiceEvent(
                kind: .lifetimeAcknowledged, generationID: nonce,
                lifetimeNonce: nonce, modelIdentity: qwenIdentity())),
            ("conversation epoch", DecodeServiceEvent(
                kind: .lifetimeAcknowledged, generationID: nonce,
                lifetimeNonce: nonce, conversationEpoch: UUID())),
            ("load attempt", DecodeServiceEvent(
                kind: .lifetimeAcknowledged, generationID: nonce,
                loadAttemptID: UUID(), lifetimeNonce: nonce)),
        ]

        for (label, acknowledgement) in variants {
            let result = try await exerciseStartupHandshake(
                acknowledgement: acknowledgement, nonce: nonce,
                expectInstallation: false)
            #expect(result.probeWasTheOnlyCommand, "\(label) accepted another command")
            #expect(result.peerObservedCleanup, "\(label) did not close the probe handles")
            #expect(result.failureWasModelLoadFailed, "\(label) returned the wrong failure")
            #expect(!result.connectionWasInstalled, "\(label) installed transport")
        }
    }

    @Test func validStartupAcknowledgementInstallsTheTransport() async throws {
        let nonce = UUID()
        let acknowledgement = DecodeServiceEvent(
            kind: .lifetimeAcknowledged, generationID: nonce,
            lifetimeNonce: nonce)
        let result = try await exerciseStartupHandshake(
            acknowledgement: acknowledgement, nonce: nonce,
            expectInstallation: true)

        #expect(result.probeWasTheOnlyCommand)
        #expect(!result.peerObservedCleanup)
        #expect(!result.failureWasModelLoadFailed)
        #expect(result.connectionWasInstalled)
    }

    @Test func acceptedQwenReadyFrameAdmitsExactClientReadiness() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let options = AppRuntimeOptions()
        let expectedIdentity = qwenIdentity()
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-identity-model"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let request = try await BlockingCommandReader(
            commands.fileHandleForReading).nextLoad()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: request.requestID,
                loadAttemptID: request.attemptID,
                loadedFamily: .qwen3_6, loadID: UUID(),
                modelIdentity: expectedIdentity, toolThinkingEnabled: nil)))
        try await load.value
        let readiness = await client.loadedModelReadiness
        guard case let .qwen(identity) = readiness else {
            Issue.record("accepted Qwen readiness was not exposed by the client")
            return
        }
        #expect(identity == expectedIdentity)
        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func wrongLoadAttemptIDIsRejectedBeforeSuccessorPublishes() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let options = AppRuntimeOptions()
        let firstLoad = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-wrong-attempt"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let firstRequest = try await reader.nextLoad()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: firstRequest.requestID,
                loadAttemptID: UUID(), loadedFamily: .qwen3_6, loadID: UUID(),
                modelIdentity: qwenIdentity(), toolThinkingEnabled: nil)))
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .loadCancelled, generationID: firstRequest.requestID,
                loadAttemptID: firstRequest.attemptID)))
        await #expect(throws: (any Error).self) { try await firstLoad.value }
        let cancellationCommand = try await reader.next()
        guard case let .cancelLoad(cancellationRequest) = cancellationCommand else {
            Issue.record("wrong-attempt cleanup did not send a load cancellation")
            return
        }
        #expect(cancellationRequest.requestID == firstRequest.requestID)
        #expect(cancellationRequest.attemptID == firstRequest.attemptID)
        #expect(client.connectionIsInstalled)
        #expect(await client.loadedModelReadiness == nil)

        let successor = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-correct-successor"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let successorRequest = try await reader.nextLoad()
        let successorLoadID = UUID()
        let successorIdentity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: successorRequest.requestID,
                loadAttemptID: successorRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: successorLoadID,
                modelIdentity: successorIdentity, toolThinkingEnabled: nil)))
        try await successor.value
        guard case let .qwen(identity) = await client.loadedModelReadiness else {
            Issue.record("correct successor did not publish readiness")
            return
        }
        #expect(identity == successorIdentity)
        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func missingQwenIdentityFailsClosedAndInvalidatesTheClient() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let options = AppRuntimeOptions()
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-missing-identity"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let request = try await BlockingCommandReader(
            commands.fileHandleForReading).nextLoad()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: request.requestID,
                loadAttemptID: request.attemptID,
                loadedFamily: .qwen3_6, loadID: UUID(),
                modelIdentity: nil, toolThinkingEnabled: nil)))
        await #expect(throws: (any Error).self) { try await load.value }
        #expect(!client.connectionIsInstalled)
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func missingLoadedFamilyFailsClosedAndInvalidatesTheClient() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let options = AppRuntimeOptions()
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-missing-family"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let request = try await BlockingCommandReader(
            commands.fileHandleForReading).nextLoad()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: request.requestID,
                loadAttemptID: request.attemptID,
                loadedFamily: nil, loadID: UUID(),
                modelIdentity: qwenIdentity(), toolThinkingEnabled: nil)))
        await #expect(throws: (any Error).self) { try await load.value }
        #expect(!client.connectionIsInstalled)
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func mismatchedQwenFamilyFailsClosedAndInvalidatesTheClient() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let options = AppRuntimeOptions()
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-mismatched-family"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let request = try await BlockingCommandReader(
            commands.fileHandleForReading).nextLoad()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: request.requestID,
                loadAttemptID: request.attemptID,
                loadedFamily: .gemma4, loadID: UUID(),
                modelIdentity: qwenIdentity(), toolThinkingEnabled: nil)))
        await #expect(throws: (any Error).self) { try await load.value }
        #expect(!client.connectionIsInstalled)
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func postTraceReplacementDropsEveryStaleResponseEffect() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let modelDirectory = try makeTemporaryModelDirectory("p17-trace-replacement")
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let traceEntered = AwaitOnce()
        let releaseTrace = AwaitOnce()
        let cleanupCompleted = DispatchSemaphore(value: 0)
        let cleanupCount = LockedCounter()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in },
            runtimeMeasurementEvent: { _, _, _, _ in
                traceEntered.signal()
                await releaseTrace.wait()
            },
            runtimeMeasurementReceptionEnded: { _, _, _ in
                cleanupCount.increment()
                cleanupCompleted.signal()
            })
        let options = AppRuntimeOptions()
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: modelDirectory,
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let loadCommand = try await reader.next()
        guard case let .load(loadRequest) = loadCommand else {
            Issue.record("load handshake did not send a load command")
            return
        }
        let loadID = UUID()
        let identity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: loadRequest.requestID,
                loadAttemptID: loadRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: loadID,
                modelIdentity: identity, toolThinkingEnabled: nil)))
        try await load.value

        var request = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "hello", maxNewTokens: 2, maxContextTokens: 128)
        let capture = DecodeRuntimeMeasurementRequest(
            stepID: UUID(), stepIndex: 0, requestStartRetainedTokens: 0,
            contextBucket: 0)
        request.runtimeMeasurementCapture = capture
        let stream = client.generate(request)
        let consumer = Task { () -> [AppInferenceEvent] in
            var events: [AppInferenceEvent] = []
            do {
                for try await event in stream { events.append(event) }
            } catch { }
            return events
        }
        let generationCommand = try await reader.next()
        guard case let .generate(generationRequest) = generationCommand else {
            Issue.record("generation did not send a generate command")
            return
        }
        var snapshot = DecodeServiceEvent(
            kind: .snapshot, generationID: generationRequest.generationID,
            loadedFamily: .qwen3_6, loadID: loadID, modelIdentity: identity,
            sequence: 1, textDelta: "stale", tokenCount: 1,
            currentMemoryBytes: 123, visionTowerMappedBytes: 456)
        snapshot.measurementCaptureID = capture.stepID
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(snapshot))
        await traceEntered.wait()

        let unload = Task { await client.unload() }
        let unloadCommand = try await reader.next()
        guard case let .unloadBound(unloadRequest) = unloadCommand else {
            Issue.record("replacement did not send a bound unload command")
            return
        }
        #expect(unloadRequest.loadID == loadID)
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .unloaded, generationID: unloadRequest.requestID,
                loadedFamily: .qwen3_6, loadID: loadID,
                modelIdentity: identity)))
        await unload.value
        releaseTrace.signal()

        client.shutdownForTermination()
        let events = await consumer.value
        let cleanupObserved = await awaitSemaphore(
            cleanupCompleted, timeout: .now() + 5)
        #expect(cleanupObserved, "runtime measurement cleanup did not complete")
        #expect(client.currentInferenceMemoryBytes == nil)
        #expect(client.currentInferenceTowerBytes == nil)
        #expect(client.generationTranscriptMailbox.completeText.isEmpty)
        #expect(events.isEmpty)
        #expect(cleanupCount.value == 1)
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func delayedOldStreamCancellationCannotCancelReplacementOperation() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let modelDirectory = try makeTemporaryModelDirectory("p17-delayed-cancel")
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let scheduledCancellation = DeferredAction()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in },
            scheduleStreamCancellation: { action in scheduledCancellation.store(action) })
        let options = AppRuntimeOptions()
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: modelDirectory,
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        let loadCommand = try await reader.next()
        guard case let .load(loadRequest) = loadCommand else {
            Issue.record("load handshake did not send a load command")
            return
        }
        let loadID = UUID()
        let identity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: loadRequest.requestID,
                loadAttemptID: loadRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: loadID,
                modelIdentity: identity, toolThinkingEnabled: nil)))
        try await load.value

        func request() -> AppGenerationRequest {
            AppGenerationRequest(
                modelDirectory: modelDirectory,
                prompt: "hello", maxNewTokens: 2, maxContextTokens: 128)
        }
        let oldStream = client.generate(request())
        let oldConsumer = Task {
            do { for try await _ in oldStream {} } catch { }
        }
        let oldCommand = try await reader.next()
        guard case let .generate(oldGeneration) = oldCommand else {
            Issue.record("old generation did not send a generate command")
            return
        }

        oldConsumer.cancel()
        await scheduledCancellation.wait()

        let newStream = client.generate(request())
        let newConsumer = Task {
            do { for try await _ in newStream {} } catch { }
        }
        let newCommand = try await reader.next()
        guard case let .generate(newGeneration) = newCommand else {
            Issue.record("replacement generation did not send a generate command")
            return
        }
        #expect(oldGeneration.generationID != newGeneration.generationID)

        await scheduledCancellation.invoke()
        let oldCancelCommand = try await reader.next()
        guard case let .cancelGenerationBound(oldCancelRequest) = oldCancelCommand else {
            Issue.record("old cancellation did not send a bound cancel command")
            return
        }
        #expect(oldCancelRequest.operationID == oldGeneration.generationID)
        #expect(oldCancelRequest.loadID == loadID)

        client.cancel()
        let newCancelCommand = try await reader.next()
        guard case let .cancelGenerationBound(newCancelRequest) = newCancelCommand else {
            Issue.record("replacement cancellation did not send a bound cancel command")
            return
        }
        #expect(newCancelRequest.operationID == newGeneration.generationID)
        #expect(newCancelRequest.loadID == loadID)

        client.shutdownForTermination()
        _ = await oldConsumer.value
        _ = await newConsumer.value
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func resetAndCheckpointAdmitTheirDeclaredEpochs() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let options = AppRuntimeOptions()
        let directory = URL(fileURLWithPath: "/tmp/p17-epoch-admission")
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }
        let loadCommand = try await reader.next()
        guard case let .load(loadRequest) = loadCommand else {
            Issue.record("load did not send a load command")
            return
        }
        let loadID = UUID()
        let identity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: loadRequest.requestID,
                loadAttemptID: loadRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: loadID,
                modelIdentity: identity, toolThinkingEnabled: nil)))
        try await load.value

        let resetEpoch = UUID()
        let reset = Task { try await client.resetConversation(epoch: resetEpoch) }
        let resetCommand = try await reader.next()
        guard case let .resetConversation(resetRequest) = resetCommand else {
            Issue.record("reset did not send a reset command")
            return
        }
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .conversationReset, generationID: resetRequest.requestID,
                loadedFamily: .qwen3_6, loadID: loadID, modelIdentity: identity,
                conversationEpoch: resetEpoch)))
        try await reset.value

        func checkpoint(
            source: UUID, replacement: UUID, commit: Bool
        ) -> DecodeContextCheckpointRequest {
            DecodeContextCheckpointRequest(
                requestID: UUID(), loadID: loadID, checkpointID: UUID(),
                sourceEpoch: source, sourceTurnIndex: 0,
                replacementEpoch: replacement,
                pendingCall: DecodeToolCall(
                    id: "call", name: "tool", argumentsJSON: "{}"),
                result: DecodeToolResult(
                    callID: "call", name: "tool", content: "ok"),
                record: "record", commit: commit)
        }
        func receipt(for request: DecodeContextCheckpointRequest)
            -> DecodeContextCheckpointReceipt {
            DecodeContextCheckpointReceipt(
                checkpointID: request.checkpointID,
                replacementEpoch: request.replacementEpoch,
                committed: request.commit, needed: false,
                existingPromptTokens: 0, replacementPromptTokens: nil,
                reserveTokens: 0, resultAllowanceTokens: 0,
                retainedImageCount: 0, retainedImageRows: 0,
                retainedFeatureBytes: 0, preparationSeconds: 0)
        }
        let committedRequest = checkpoint(
            source: resetEpoch, replacement: UUID(), commit: true)
        let committed = Task {
            try await client.contextCheckpoint(committedRequest)
        }
        let committedCommand = try await reader.next()
        guard case let .contextCheckpoint(committedWire) = committedCommand else {
            Issue.record("committed checkpoint did not send a checkpoint command")
            return
        }
        var committedEvent = DecodeServiceEvent(
            kind: .contextCheckpoint, generationID: committedWire.requestID,
            loadedFamily: .qwen3_6, loadID: loadID, modelIdentity: identity,
            conversationEpoch: committedRequest.replacementEpoch)
        committedEvent.contextCheckpoint = receipt(for: committedRequest)
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            committedEvent))
        let committedReceipt = try await committed.value
        #expect(committedReceipt == receipt(for: committedRequest))

        let uncommittedRequest = checkpoint(
            source: committedRequest.replacementEpoch, replacement: UUID(), commit: false)
        let uncommitted = Task {
            try await client.contextCheckpoint(uncommittedRequest)
        }
        let uncommittedCommand = try await reader.next()
        guard case let .contextCheckpoint(uncommittedWire) = uncommittedCommand else {
            Issue.record("uncommitted checkpoint did not send a checkpoint command")
            return
        }
        var uncommittedEvent = DecodeServiceEvent(
            kind: .contextCheckpoint, generationID: uncommittedWire.requestID,
            loadedFamily: .qwen3_6, loadID: loadID, modelIdentity: identity,
            conversationEpoch: uncommittedRequest.sourceEpoch)
        uncommittedEvent.contextCheckpoint = receipt(for: uncommittedRequest)
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            uncommittedEvent))
        let uncommittedReceipt = try await uncommitted.value
        #expect(uncommittedReceipt == receipt(for: uncommittedRequest))

        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func wrongGenerationEpochInvalidatesOnlyTheCurrentOperation() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let options = AppRuntimeOptions()
        let directory = try makeTemporaryModelDirectory("p17-wrong-generation-epoch")
        defer { try? FileManager.default.removeItem(at: directory) }
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }
        let loadCommand = try await reader.next()
        guard case let .load(loadRequest) = loadCommand else {
            Issue.record("load did not send a load command")
            return
        }
        let loadID = UUID()
        let identity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: loadRequest.requestID,
                loadAttemptID: loadRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: loadID,
                modelIdentity: identity, toolThinkingEnabled: nil)))
        try await load.value

        let expectedEpoch = UUID()
        let reset = Task { try await client.resetConversation(epoch: expectedEpoch) }
        let resetCommand = try await reader.next()
        guard case let .resetConversation(resetRequest) = resetCommand else {
            Issue.record("reset did not send a reset command")
            return
        }
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .conversationReset, generationID: resetRequest.requestID,
                loadedFamily: .qwen3_6, loadID: loadID, modelIdentity: identity,
                conversationEpoch: expectedEpoch)))
        try await reset.value

        let controlRequest = AppGenerationRequest(
            modelDirectory: directory, prompt: "control", maxNewTokens: 2,
            maxContextTokens: 128, conversationEpoch: expectedEpoch)
        let controlStream = client.generate(controlRequest)
        let controlConsumer = Task {
            do { for try await _ in controlStream {} } catch { }
        }
        let controlCommand = try await reader.next()
        guard case let .generate(controlGeneration) = controlCommand else {
            Issue.record("control generation did not send a generate command")
            return
        }
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .failed, generationID: controlGeneration.generationID,
                loadedFamily: .qwen3_6, loadID: loadID, modelIdentity: identity,
                error: "control failure", conversationEpoch: expectedEpoch)))
        _ = await controlConsumer.value
        #expect(client.connectionIsInstalled)
        guard case .qwen = await client.loadedModelReadiness else {
            Issue.record("control failure incorrectly removed readiness")
            return
        }

        let request = AppGenerationRequest(
            modelDirectory: directory, prompt: "hello", maxNewTokens: 2,
            maxContextTokens: 128, conversationEpoch: expectedEpoch)
        let stream = client.generate(request)
        let consumer = Task {
            do { for try await _ in stream {} } catch { }
        }
        let command = try await reader.next()
        guard case let .generate(generation) = command else {
            Issue.record("generation did not send a generate command")
            return
        }
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .snapshot, generationID: generation.generationID,
                loadedFamily: .qwen3_6, loadID: loadID,
                modelIdentity: identity, sequence: 1, textDelta: "wrong",
                conversationEpoch: UUID())))
        _ = await consumer.value
        #expect(!client.connectionIsInstalled)
        #expect(client.currentInferenceMemoryBytes == nil)
        #expect(client.currentInferenceTowerBytes == nil)
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func overlappingLoadRefusesBeforePublishingReadyState() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let states = StateRecorder()
        let firstDirectory = URL(fileURLWithPath: "/tmp/p17-overlapping-load-a")
        let secondDirectory = URL(fileURLWithPath: "/tmp/p17-overlapping-load-b")
        let firstLoad = Task {
            try await client.ensureLoaded(
                modelDirectory: firstDirectory, maxContextTokens: 128,
                options: AppRuntimeOptions(), forceLogitsHead: false) { state in
                    states.append(state)
                }
        }
        let firstCommand = try await reader.next()
        guard case let .load(firstRequest) = firstCommand else {
            Issue.record("first load did not send a load command")
            return
        }

        await #expect(throws: RealInferenceLifecycleError.lifecycleInProgress) {
            try await client.ensureLoaded(
                modelDirectory: secondDirectory, maxContextTokens: 128,
                options: AppRuntimeOptions(), forceLogitsHead: false) { state in
                    states.append(state)
                }
        }

        let identity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: firstRequest.requestID,
                loadAttemptID: firstRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: UUID(), modelIdentity: identity,
                toolThinkingEnabled: nil)))
        try await firstLoad.value
        #expect(states.snapshot().contains { state in
            if case let .ready(directory, _) = state { return directory == firstDirectory }
            return false
        })
        let readiness = await client.loadedModelReadiness
        guard case .qwen = readiness else {
            Issue.record("first load readiness was not retained")
            return
        }
        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func postAdmissionLoadWriteFailureInvalidatesConnection() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let writerControl = PreAdmissionWriterControl()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            sendFrame: { frame, output in
                try writerControl.send(frame: frame, output: output)
            },
            scheduleTerminationFallback: { _ in })
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let options = AppRuntimeOptions()
        let directory = URL(fileURLWithPath: "/tmp/p17-pre-admission-a")
        let identity = qwenIdentity()
        let firstLoad = Task {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }
        let firstRequest = try await reader.nextLoad()
        let loadID = UUID()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: firstRequest.requestID,
                loadAttemptID: firstRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: loadID,
                modelIdentity: identity, toolThinkingEnabled: nil)))
        try await firstLoad.value

        writerControl.failNextLoad()
        await #expect(throws: (any Error).self) {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/p17-pre-admission-b"),
                maxContextTokens: 128, options: options, forceLogitsHead: false) { _ in }
        }
        #expect(writerControl.successfulFrameCount == 1)
        #expect(writerControl.attemptedFrameCount == 2)
        #expect(!client.connectionIsInstalled)
        let readiness = await client.loadedModelReadiness
        #expect(readiness == nil)
        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func supersededUnloadAcknowledgementCannotClearReplacementState() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let options = AppRuntimeOptions()
        let directory = URL(fileURLWithPath: "/tmp/p17-superseded-unload")
        let identityA = qwenIdentity()
        let identityB = DecodeModelIdentity(
            family: .qwen3_6, modelID: "Qwen/Qwen3.6-35B-A3B-replacement",
            sourceRevision: "replacement-text-revision", formatMajor: 2,
            formatMinor: 0, sourceIndexSHA256: String(repeating: "f", count: 64),
            quantizationPolicySHA256: String(repeating: "e", count: 64),
            textManifestSHA256: String(repeating: "d", count: 64),
            quantization: identityA.quantization, vision: identityA.vision)
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }
        let loadCommand = try await reader.next()
        guard case let .load(loadRequest) = loadCommand else {
            Issue.record("load did not send a load command")
            return
        }
        let firstLoadID = UUID()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: loadRequest.requestID,
                loadAttemptID: loadRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: firstLoadID,
                modelIdentity: identityA, currentMemoryBytes: 11,
                visionTowerMappedBytes: 12, toolThinkingEnabled: nil)))
        try await load.value

        let unload = Task { await client.unload() }
        let unloadCommand = try await reader.next()
        guard case let .unloadBound(unloadRequest) = unloadCommand else {
            Issue.record("unload did not send a bound command")
            return
        }
        let joiningUnload = Task { await client.unload() }
        await Task.yield()
        joiningUnload.cancel()
        await #expect(throws: RealInferenceLifecycleError.lifecycleInProgress) {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }

        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .unloaded, generationID: unloadRequest.requestID,
                loadedFamily: .qwen3_6, loadID: firstLoadID,
                modelIdentity: identityA)))
        await unload.value
        await joiningUnload.value

        let replacementLoad = Task {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }
        let replacementCommand = try await reader.next()
        guard case let .load(replacementRequest) = replacementCommand else {
            Issue.record("replacement load did not send a load command")
            return
        }
        let secondLoadID = UUID()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: replacementRequest.requestID,
                loadAttemptID: replacementRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: secondLoadID,
                modelIdentity: identityB, currentMemoryBytes: 22,
                visionTowerMappedBytes: 23, toolThinkingEnabled: nil)))
        try await replacementLoad.value
        #expect(client.currentInferenceMemoryBytes == 22)
        #expect(client.currentInferenceTowerBytes == 23)
        guard case let .qwen(identity) = await client.loadedModelReadiness else {
            Issue.record("replacement readiness was cleared by stale unload")
            return
        }
        #expect(identity == identityB)
        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func staleMismatchedFrameCannotTearDownSameRouterReplacement() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = BlockingCommandReader(commands.fileHandleForReading)
        let options = AppRuntimeOptions()
        let directory = try makeTemporaryModelDirectory("p17-same-router-replacement")
        defer { try? FileManager.default.removeItem(at: directory) }
        let identityA = qwenIdentity()
        let identityB = DecodeModelIdentity(
            family: .qwen3_6, modelID: "Qwen/Qwen3.6-35B-A3B-replacement",
            sourceRevision: "replacement-text-revision", formatMajor: 2,
            formatMinor: 0, sourceIndexSHA256: String(repeating: "f", count: 64),
            quantizationPolicySHA256: String(repeating: "e", count: 64),
            textManifestSHA256: String(repeating: "d", count: 64),
            quantization: identityA.quantization, vision: identityA.vision)

        let firstLoad = Task {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }
        let firstLoadCommand = try await reader.next()
        guard case let .load(firstRequest) = firstLoadCommand else {
            Issue.record("first load did not send a load command")
            return
        }
        let firstLoadID = UUID()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: firstRequest.requestID,
                loadAttemptID: firstRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: firstLoadID,
                modelIdentity: identityA, toolThinkingEnabled: nil)))
        try await firstLoad.value

        let oldStream = client.generate(AppGenerationRequest(
            modelDirectory: directory, prompt: "old", maxNewTokens: 2,
            maxContextTokens: 128))
        let oldConsumer = Task {
            do { for try await _ in oldStream {} } catch { }
        }
        let oldCommand = try await reader.next()
        guard case let .generate(oldGeneration) = oldCommand else {
            Issue.record("old generation did not send a generate command")
            return
        }

        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .finished, generationID: oldGeneration.generationID,
                loadedFamily: .qwen3_6, loadID: firstLoadID,
                modelIdentity: identityA, stopReason: "max_tokens")))
        _ = await oldConsumer.value

        let secondLoad = Task {
            try await client.ensureLoaded(
                modelDirectory: directory, maxContextTokens: 128,
                options: options, forceLogitsHead: false) { _ in }
        }
        let secondLoadCommand = try await reader.next()
        guard case let .load(secondRequest) = secondLoadCommand else {
            Issue.record("replacement load did not send a load command")
            return
        }
        let secondLoadID = UUID()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready, generationID: secondRequest.requestID,
                loadAttemptID: secondRequest.attemptID,
                loadedFamily: .qwen3_6, loadID: secondLoadID,
                modelIdentity: identityB, toolThinkingEnabled: nil)))
        try await secondLoad.value

        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .snapshot, generationID: oldGeneration.generationID,
                loadedFamily: .qwen3_6, loadID: secondLoadID,
                modelIdentity: identityB, sequence: 1, textDelta: "stale")))
        let readiness = await client.loadedModelReadiness
        guard case let .qwen(identity) = readiness else {
            Issue.record("replacement readiness was lost after stale frame")
            return
        }
        #expect(identity == identityB)
        #expect(client.connectionIsInstalled)
        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func staleLifetimeExitCannotClearAReplacementTransport() throws {
        let initialCommands = Pipe()
        let initialResponses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: initialCommands.fileHandleForWriting,
            responseOutput: initialResponses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let lifetimeA = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(), peer: DecodeServicePeerIdentity(pid: 1, pidVersion: 1))
        let commandsA = Pipe()
        let responsesA = Pipe()
        client.installTransport(
            input: commandsA.fileHandleForWriting,
            output: responsesA.fileHandleForReading,
            lifetime: lifetimeA)
        client.handleLifetimeEvent(
            lifetimeID: lifetimeA.id, event: .executionChanged)
        #expect(client.retiredLifetimeID == lifetimeA.id)

        let lifetimeB = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(), peer: DecodeServicePeerIdentity(pid: 2, pidVersion: 1))
        let commandsB = Pipe()
        let responsesB = Pipe()
        client.installTransport(
            input: commandsB.fileHandleForWriting,
            output: responsesB.fileHandleForReading,
            lifetime: lifetimeB)
        let admittedB = try lifetimeB.admitFirstModelFrame { true }
        #expect(admittedB)
        client.handleLifetimeEvent(
            lifetimeID: lifetimeA.id, event: .exited)

        #expect(client.connectionIsInstalled)
        #expect(client.retiredLifetimeID == nil)
        #expect(!client.lifetimeCleanupIsQuarantined)
        #expect(!lifetimeB.isExited)
        client.shutdownForTermination()
        try? initialCommands.fileHandleForReading.close()
        try? initialResponses.fileHandleForWriting.close()
        try? commandsA.fileHandleForReading.close()
        try? responsesA.fileHandleForWriting.close()
        try? commandsB.fileHandleForReading.close()
        try? responsesB.fileHandleForWriting.close()
    }

    @Test func repeatedEmptyExecutionUsesBoundedRetiredAndQuarantineSlots() throws {
        let initialCommands = Pipe()
        let initialResponses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: initialCommands.fileHandleForWriting,
            responseOutput: initialResponses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let lifetimeA = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(), peer: DecodeServicePeerIdentity(pid: 1, pidVersion: 1))
        let commandsA = Pipe()
        let responsesA = Pipe()
        client.installTransport(
            input: commandsA.fileHandleForWriting,
            output: responsesA.fileHandleForReading,
            lifetime: lifetimeA)
        client.handleLifetimeEvent(
            lifetimeID: lifetimeA.id, event: .executionChanged)
        #expect(client.retiredLifetimeID == lifetimeA.id)
        #expect(!client.lifetimeCleanupIsQuarantined)

        let lifetimeB = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(), peer: DecodeServicePeerIdentity(pid: 2, pidVersion: 1))
        let commandsB = Pipe()
        let responsesB = Pipe()
        client.installTransport(
            input: commandsB.fileHandleForWriting,
            output: responsesB.fileHandleForReading,
            lifetime: lifetimeB)
        client.handleLifetimeEvent(
            lifetimeID: lifetimeB.id, event: .executionChanged)
        #expect(client.retiredLifetimeID == lifetimeA.id)
        #expect(client.lifetimeCleanupIsQuarantined)

        client.handleLifetimeEvent(lifetimeID: lifetimeA.id, event: .exited)
        #expect(client.retiredLifetimeID == nil)
        #expect(client.lifetimeCleanupIsQuarantined)
        client.handleLifetimeEvent(lifetimeID: lifetimeB.id, event: .exited)
        #expect(!client.lifetimeCleanupIsQuarantined)
        client.shutdownForTermination()
        try? initialCommands.fileHandleForReading.close()
        try? initialResponses.fileHandleForWriting.close()
        try? commandsA.fileHandleForReading.close()
        try? responsesA.fileHandleForWriting.close()
        try? commandsB.fileHandleForReading.close()
        try? responsesB.fileHandleForWriting.close()
    }

    @Test func cleanupUnprovenReplacementIsPromotedAfterOldWatcherExit() throws {
        let initialCommands = Pipe()
        let initialResponses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: initialCommands.fileHandleForWriting,
            responseOutput: initialResponses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let lifetimeA = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(), peer: DecodeServicePeerIdentity(pid: 1, pidVersion: 1))
        let commandsA = Pipe()
        let responsesA = Pipe()
        client.installTransport(
            input: commandsA.fileHandleForWriting,
            output: responsesA.fileHandleForReading,
            lifetime: lifetimeA)
        client.handleLifetimeEvent(
            lifetimeID: lifetimeA.id, event: .executionChanged)
        #expect(client.retiredLifetimeID == lifetimeA.id)
        #expect(!client.lifetimeCleanupIsQuarantined)

        let lifetimeB = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(), peer: DecodeServicePeerIdentity(pid: 2, pidVersion: 1))
        let commandsB = Pipe()
        let responsesB = Pipe()
        client.installTransport(
            input: commandsB.fileHandleForWriting,
            output: responsesB.fileHandleForReading,
            lifetime: lifetimeB)
        let admittedB = try lifetimeB.admitFirstModelFrame { true }
        #expect(admittedB)
        client.handleLifetimeEvent(
            lifetimeID: lifetimeA.id, event: .observationFailed(5))
        #expect(client.lifetimeCleanupIsQuarantined)
        client.handleLifetimeEvent(
            lifetimeID: lifetimeB.id, event: .observationFailed(6))

        #expect(client.retiredLifetimeID == lifetimeB.id)
        #expect(client.lifetimeCleanupIsQuarantined)
        client.handleLifetimeEvent(lifetimeID: lifetimeA.id, event: .exited)
        #expect(client.retiredLifetimeID == nil)
        #expect(client.lifetimeCleanupIsQuarantined)
        client.handleLifetimeEvent(lifetimeID: lifetimeB.id, event: .exited)
        #expect(!client.lifetimeCleanupIsQuarantined)
        client.shutdownForTermination()
        try? initialCommands.fileHandleForReading.close()
        try? initialResponses.fileHandleForWriting.close()
        try? commandsA.fileHandleForReading.close()
        try? responsesA.fileHandleForWriting.close()
        try? commandsB.fileHandleForReading.close()
        try? responsesB.fileHandleForWriting.close()
    }

    @Test func readyEventRoundTripsOrderedQuantizationAndVerifiedVision() throws {
        let generationID = UUID()
        let loadID = UUID()
        let identity = qwenIdentity()
        let event = DecodeServiceEvent(
            kind: .ready,
            generationID: generationID,
            loadedFamily: .qwen3_6,
            loadID: loadID,
            modelIdentity: identity,
            toolThinkingEnabled: false)
        let decoded = try JSONDecoder().decode(
            DecodeServiceEvent.self,
            from: JSONEncoder().encode(event))

        #expect(decoded.loadedFamily == .qwen3_6)
        #expect(decoded.loadID == loadID)
        #expect(decoded.modelIdentity == identity)
        #expect(decoded.generationID == generationID)
        #expect(decoded.toolThinkingEnabled == false)
        #expect(decoded.modelIdentity?.quantization.map(\.category)
            == identity.quantization.map(\.category))
        guard case let .verified(vision) = decoded.modelIdentity?.vision else {
            Issue.record("verified vision identity was not retained")
            return
        }
        #expect(vision.sourceRevision == "vision-revision")
        #expect(vision.processorConfigSHA256 == String(repeating: "d", count: 64))
        #expect(vision.compatibleTextManifestSHA256 == identity.textManifestSHA256)
        #expect(vision.visionPayloadSHA256 == String(repeating: "e", count: 64))
        #expect(vision.supportsStillImages)
        #expect(!vision.supportsVideo)
    }

    @Test func loadBoundControlsRoundTripTheirIncarnationAndOperation() throws {
        let loadID = UUID()
        let requestID = UUID()
        let operationID = UUID()
        let unload = DecodeServiceCommand.unloadBound(
            DecodeUnloadRequest(requestID: requestID, loadID: loadID))
        let cancel = DecodeServiceCommand.cancelGenerationBound(
            DecodeCancelGenerationRequest(operationID: operationID, loadID: loadID))

        let decodedUnload = try JSONDecoder().decode(
            DecodeServiceCommand.self,
            from: JSONEncoder().encode(unload))
        let decodedCancel = try JSONDecoder().decode(
            DecodeServiceCommand.self,
            from: JSONEncoder().encode(cancel))
        guard case let .unloadBound(unloadRequest) = decodedUnload else {
            Issue.record("bound unload was not retained")
            return
        }
        guard case let .cancelGenerationBound(cancelRequest) = decodedCancel else {
            Issue.record("bound cancellation was not retained")
            return
        }
        #expect(unloadRequest.requestID == requestID)
        #expect(unloadRequest.loadID == loadID)
        #expect(cancelRequest.operationID == operationID)
        #expect(cancelRequest.loadID == loadID)
    }

    @Test func legacyControlsRemainDecodableButHaveNoLoadBinding() throws {
        let requestID = UUID()
        let operationID = UUID()
        let unload = DecodeServiceCommand.unload(requestID)
        let cancel = DecodeServiceCommand.cancelGeneration(operationID)
        let decodedUnload = try JSONDecoder().decode(
            DecodeServiceCommand.self,
            from: JSONEncoder().encode(unload))
        let decodedCancel = try JSONDecoder().decode(
            DecodeServiceCommand.self,
            from: JSONEncoder().encode(cancel))

        guard case let .unload(unloadID) = decodedUnload else {
            Issue.record("legacy unload no longer decodes")
            return
        }
        guard case let .cancelGeneration(cancelID) = decodedCancel else {
            Issue.record("legacy cancellation no longer decodes")
            return
        }
        #expect(unloadID == requestID)
        #expect(cancelID == operationID)
    }

    @Test func resetAndCheckpointCarryTheSameLoadIncarnation() throws {
        let loadID = UUID()
        let epoch = UUID()
        let checkpointID = UUID()
        let reset = DecodeResetConversationRequest(
            epoch: epoch, requestID: UUID(), loadID: loadID)
        let call = DecodeToolCall(id: "call", name: "lookup", argumentsJSON: "{}")
        let result = DecodeToolResult(callID: "call", name: "lookup", content: "ok")
        let checkpoint = DecodeContextCheckpointRequest(
            requestID: UUID(), loadID: loadID, checkpointID: checkpointID,
            sourceEpoch: epoch, sourceTurnIndex: 0,
            replacementEpoch: UUID(), pendingCall: call, result: result,
            record: "record", commit: true)

        let decodedReset = try JSONDecoder().decode(
            DecodeResetConversationRequest.self,
            from: JSONEncoder().encode(reset))
        let decodedCheckpoint = try JSONDecoder().decode(
            DecodeContextCheckpointRequest.self,
            from: JSONEncoder().encode(checkpoint))
        #expect(decodedReset.loadID == loadID)
        #expect(decodedReset.epoch == epoch)
        #expect(decodedCheckpoint.loadID == loadID)
        #expect(decodedCheckpoint.checkpointID == checkpointID)
    }

    private func qwenIdentity() -> DecodeModelIdentity {
        DecodeModelIdentity(
            family: .qwen3_6,
            modelID: "Qwen/Qwen3.6-35B-A3B",
            sourceRevision: "text-revision",
            formatMajor: 2,
            formatMinor: 0,
            sourceIndexSHA256: String(repeating: "a", count: 64),
            quantizationPolicySHA256: String(repeating: "b", count: 64),
            textManifestSHA256: String(repeating: "c", count: 64),
            quantization: [
                DecodeQuantizationIdentity(
                    category: "recurrent_state", storage: "fp32"),
                DecodeQuantizationIdentity(
                    category: "embedding", storage: "affine_int4",
                    groupSize: 64, scaleType: "bf16", biasType: "bf16"),
            ],
            vision: .verified(DecodeVerifiedVisionIdentity(
                sourceRevision: "vision-revision",
                processorConfigSHA256: String(repeating: "d", count: 64),
                compatibleTextManifestSHA256: String(repeating: "c", count: 64),
                visionPayloadSHA256: String(repeating: "e", count: 64),
                supportsStillImages: true, supportsVideo: false)))
    }

    private func exerciseStartupHandshake(
        acknowledgement: DecodeServiceEvent,
        nonce: UUID,
        expectInstallation: Bool
    ) async throws -> StartupHandshakeResult {
        let input = Pipe()
        let output = Pipe()
        let peerFinished = DispatchSemaphore(value: 0)
        let peerResult = StartupHandshakePeerResult()
        let peer = Thread {
            var keepResponseOpen = false
            defer {
                if !keepResponseOpen {
                    try? output.fileHandleForWriting.close()
                }
                peerFinished.signal()
            }
            do {
                let command = try DecodeFrameCodec.read(
                    DecodeServiceCommand.self, from: input.fileHandleForReading)
                guard case let .lifetimeProbe(probe) = command,
                      probe.nonce == nonce else {
                    peerResult.record(commandWasOnlyProbe: false, observedCleanup: false)
                    return
                }
                try output.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
                    acknowledgement))
                guard !expectInstallation else {
                    keepResponseOpen = true
                    peerResult.record(commandWasOnlyProbe: true, observedCleanup: false)
                    return
                }
                let eof = try input.fileHandleForReading.read(upToCount: 1)
                peerResult.record(
                    commandWasOnlyProbe: true,
                    observedCleanup: eof == nil || eof?.isEmpty == true)
            } catch {
                peerResult.record(commandWasOnlyProbe: false, observedCleanup: false)
            }
        }
        peer.name = "TurboFieldfare.Tests.StartupHandshakePeer"
        peer.start()

        let client = DecodeServiceInferenceClient(
            serviceURL: URL(fileURLWithPath: "/tmp/p17-no-launchd-service"))
        let worker = ServiceSetupWorker()
        let lifetime = DecodeServiceInferenceClient.ServiceLifetime(
            id: UUID(),
            peer: DecodeServicePeerIdentity(pid: getpid(), pidVersion: 1))
        var threw = false
        var failureWasModelLoadFailed = false
        do {
            try client.performStartupHandshakeAndInstall(
                input: input.fileHandleForWriting,
                output: output.fileHandleForReading,
                worker: worker,
                lifetime: lifetime,
                nonce: nonce)
        } catch let error as AppInferenceError {
            threw = true
            if case .modelLoadFailed = error { failureWasModelLoadFailed = true }
        } catch {
            threw = true
        }

        #expect(threw == !expectInstallation)
        let peerCompleted = await awaitSemaphore(
            peerFinished, timeout: .now() + 2)
        if !peerCompleted {
            // Release both ends if a cleanup regression left the peer blocked,
            // then wait for the peer thread before returning from the fixture.
            try? input.fileHandleForReading.close()
            try? output.fileHandleForWriting.close()
            if !expectInstallation {
                try? input.fileHandleForWriting.close()
                try? output.fileHandleForReading.close()
            }
            _ = await awaitSemaphore(peerFinished, timeout: .now() + 2)
        }
        #expect(peerCompleted)
        let result = StartupHandshakeResult(
            probeWasTheOnlyCommand: peerResult.commandWasOnlyProbe,
            peerObservedCleanup: peerResult.observedCleanup,
            connectionWasInstalled: client.connectionIsInstalled,
            failureWasModelLoadFailed: failureWasModelLoadFailed)
        if expectInstallation {
            client.shutdownForTermination()
            try? output.fileHandleForWriting.close()
            await client.unload()
            try? input.fileHandleForReading.close()
        } else {
            try? input.fileHandleForReading.close()
            try? output.fileHandleForWriting.close()
        }
        return result
    }
}

private final class BlockingCommandReader: @unchecked Sendable {
    private let handle: FileHandle

    init(_ handle: FileHandle) { self.handle = handle }

    func next() async throws -> DecodeServiceCommand {
        try await withCheckedThrowingContinuation { continuation in
            let handle = self.handle
            let thread = Thread {
                do {
                    var descriptor = pollfd(
                        fd: handle.fileDescriptor,
                        events: Int16(POLLIN),
                        revents: 0)
                    guard Darwin.poll(&descriptor, 1, 5_000) > 0 else {
                        throw BlockingCommandReaderError.timedOut
                    }
                    continuation.resume(returning: try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handle))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            thread.name = "TurboFieldfare.Tests.IdentityCommandReader"
            thread.start()
        }
    }

    func nextLoad() async throws -> DecodeLoadRequest {
        let command = try await next()
        guard case let .load(request) = command else {
            throw DecodeFrameError.unexpectedEOF
        }
        return request
    }
}

private enum BlockingCommandReaderError: Error, Equatable {
    case timedOut
}

private func makeTemporaryModelDirectory(_ label: String) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("TurboFieldfare-\(label)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
        at: directory, withIntermediateDirectories: false)
    return directory
}

private final class ProcessRunResult: @unchecked Sendable {
    private let lock = NSLock()
    private var error: Error?
    private var completions = 0

    func record(_ error: Error?) {
        lock.withLock {
            self.error = error
            completions += 1
        }
    }

    var wasCancellation: Bool {
        lock.withLock { error is CancellationError }
    }

    var completionCount: Int {
        lock.withLock { completions }
    }
}

private struct StartupHandshakeResult: Sendable {
    let probeWasTheOnlyCommand: Bool
    let peerObservedCleanup: Bool
    let connectionWasInstalled: Bool
    let failureWasModelLoadFailed: Bool
}

private final class StartupHandshakePeerResult: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var commandWasOnlyProbe = false
    private(set) var observedCleanup = false

    func record(commandWasOnlyProbe: Bool, observedCleanup: Bool) {
        lock.withLock {
            self.commandWasOnlyProbe = commandWasOnlyProbe
            self.observedCleanup = observedCleanup
        }
    }
}

private final class PreAdmissionWriterControl: @unchecked Sendable {
    private let lock = NSLock()
    private var shouldFailNextLoad = false
    private(set) var attemptedFrameCount = 0
    private(set) var successfulFrameCount = 0

    func failNextLoad() {
        lock.withLock { shouldFailNextLoad = true }
    }

    func send(frame: Data, output: FileHandle) throws {
        let command = try JSONDecoder().decode(
            DecodeServiceCommand.self, from: Data(frame.dropFirst(4)))
        let fail = lock.withLock { () -> Bool in
            attemptedFrameCount += 1
            guard shouldFailNextLoad else { return false }
            if case .load = command {
                shouldFailNextLoad = false
                return true
            }
            return false
        }
        if fail { throw PreAdmissionWriterError.failed }
        try output.write(contentsOf: frame)
        lock.withLock { successfulFrameCount += 1 }
    }
}

private enum PreAdmissionWriterError: Error, Equatable {
    case failed
}

private final class StateRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var states: [AppModelLoadState] = []

    func append(_ state: AppModelLoadState) {
        lock.withLock { states.append(state) }
    }

    func snapshot() -> [AppModelLoadState] {
        lock.withLock { states }
    }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    func increment() {
        lock.lock()
        count += 1
        lock.unlock()
    }
}

private final class DeferredAction: @unchecked Sendable {
    private let lock = NSLock()
    private var action: (@Sendable () -> Void)?
    private let available = AwaitOnce()

    func store(_ action: @escaping @Sendable () -> Void) {
        lock.lock()
        self.action = action
        lock.unlock()
        available.signal()
    }

    func wait() async {
        await available.wait()
    }

    func invoke() async {
        await available.wait()
        let action = lock.withLock {
            defer { self.action = nil }
            return self.action
        }
        action?()
    }
}

private func awaitSemaphore(
    _ semaphore: DispatchSemaphore,
    timeout: DispatchTime
) async -> Bool {
    await withCheckedContinuation { continuation in
        let waiter = Thread {
            continuation.resume(returning: semaphore.wait(timeout: timeout) == .success)
        }
        waiter.name = "TurboFieldfare.Tests.SemaphoreWaiter"
        waiter.start()
    }
}

private final class AwaitOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var signaled = false

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if signaled {
                lock.unlock()
                continuation.resume()
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }
    }

    func signal() {
        lock.lock()
        signaled = true
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume()
    }
}
