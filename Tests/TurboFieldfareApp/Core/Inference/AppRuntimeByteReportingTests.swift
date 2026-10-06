import Darwin
import Foundation
import Testing
import TurboFieldfareDecodeProtocol
@testable import TurboFieldfareAppCore

@Suite("Runtime byte reporting")
struct AppRuntimeByteReportingTests {
    @MainActor
    @Test func reporterDefaultsKeepNewByteValuesUnavailable() {
        let reporter = LegacyMemoryReporter(memoryBytes: 111)

        #expect(reporter.currentInferenceMemoryBytes == 111)
        #expect(reporter.currentInferenceResidentBytes == nil)
        #expect(reporter.currentInferenceTowerBytes == nil)
        #expect(reporter.currentConversationLogicalStateBytes == nil)
        #expect(reporter.currentExpertCacheBytes == nil)
    }

    @MainActor
    @Test func appModelKeepsMemoryResidentLogicalAndCacheBytesSeparate() {
        let reporter = ByteReportingClient(
            memoryBytes: 11,
            residentBytes: 22,
            towerBytes: 33,
            logicalStateBytes: 44,
            expertCacheBytes: 55)
        let model = AppModel(client: reporter)
        let directory = URL(fileURLWithPath: "/tmp/runtime-byte-reporting.gturbo")
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(model.currentProcessMemoryBytes == 11)
        #expect(model.currentProcessResidentBytes == 22)
        #expect(model.visionTowerMappedBytes == 33)
        #expect(model.conversationLogicalStateBytes == 44)
        #expect(model.expertCacheBytes == 55)
        #expect(model.conversationLogicalStateBytes != model.currentProcessResidentBytes)
        #expect(model.conversationLogicalStateBytes != model.currentProcessMemoryBytes)
        #expect(model.expertCacheBytes != model.currentProcessMemoryBytes)
        #expect(model.expertCacheBytes != model.currentProcessResidentBytes)

        reporter.currentInferenceMemoryBytes = 0
        reporter.currentInferenceResidentBytes = 0
        reporter.currentInferenceTowerBytes = 0
        reporter.currentConversationLogicalStateBytes = 0
        reporter.currentExpertCacheBytes = 0
        model.apply(.memorySample)

        #expect(model.currentProcessMemoryBytes == 0)
        #expect(model.currentProcessResidentBytes == 0)
        #expect(model.visionTowerMappedBytes == 0)
        #expect(model.conversationLogicalStateBytes == 0)
        #expect(model.expertCacheBytes == 0)

        model.applyLoadState(.failed(.modelLoadFailed("stale runtime")))
        #expect(model.conversationLogicalStateBytes == nil)
        #expect(model.expertCacheBytes == nil)
    }

    @Test func absentAndExplicitZeroByteFieldsDecodeDifferently() throws {
        let legacy = DecodeServiceEvent(
            kind: .snapshot,
            generationID: UUID(),
            textDelta: "legacy")
        let legacyData = try JSONEncoder().encode(legacy)
        let legacyObject = try #require(
            JSONSerialization.jsonObject(with: legacyData) as? [String: Any])
        #expect(legacyObject["conversationLogicalStateBytes"] == nil)
        #expect(legacyObject["expertCacheBytes"] == nil)
        let decodedLegacy = try JSONDecoder().decode(
            DecodeServiceEvent.self, from: legacyData)
        #expect(decodedLegacy.conversationLogicalStateBytes == nil)
        #expect(decodedLegacy.expertCacheBytes == nil)

        let zero = DecodeServiceEvent(
            kind: .snapshot,
            generationID: UUID(),
            conversationLogicalStateBytes: 0,
            expertCacheBytes: 0)
        let decodedZero = try JSONDecoder().decode(
            DecodeServiceEvent.self,
            from: JSONEncoder().encode(zero))
        #expect(decodedZero.conversationLogicalStateBytes == 0)
        #expect(decodedZero.expertCacheBytes == 0)
    }

    @Test func decodeServiceClientForwardsByteFieldsAndClearsThemAfterUnload() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = ByteCommandReader(commands.fileHandleForReading)
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/runtime-byte-client.gturbo"),
                maxContextTokens: 128,
                options: AppRuntimeOptions(),
                forceLogitsHead: false) { _ in }
        }
        let request = try await reader.nextLoad()
        let loadID = UUID()
        let identity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready,
                generationID: request.requestID,
                loadAttemptID: request.attemptID,
                loadedFamily: .qwen3_6,
                loadID: loadID,
                modelIdentity: identity,
                currentMemoryBytes: 11,
                visionTowerMappedBytes: 33,
                conversationLogicalStateBytes: 44,
                expertCacheBytes: 55,
                toolThinkingEnabled: nil)))
        try await load.value

        #expect(client.currentInferenceMemoryBytes == 11)
        #expect(client.currentInferenceTowerBytes == 33)
        #expect(client.currentConversationLogicalStateBytes == 44)
        #expect(client.currentExpertCacheBytes == 55)

        let resetEpoch = UUID()
        let reset = Task { try await client.resetConversation(epoch: resetEpoch) }
        let resetCommand = try await reader.next()
        guard case let .resetConversation(resetRequest) = resetCommand else {
            Issue.record("byte-reporting client did not send a reset command")
            return
        }
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .conversationReset,
                generationID: resetRequest.requestID,
                loadedFamily: .qwen3_6,
                loadID: loadID,
                modelIdentity: identity,
                conversationLogicalStateBytes: 66,
                expertCacheBytes: 77,
                conversationEpoch: resetEpoch)))
        try await reset.value
        #expect(client.currentInferenceMemoryBytes == 11)
        #expect(client.currentInferenceTowerBytes == 33)
        #expect(client.currentConversationLogicalStateBytes == 66)
        #expect(client.currentExpertCacheBytes == 77)

        let checkpointRequest = DecodeContextCheckpointRequest(
            requestID: UUID(),
            loadID: loadID,
            checkpointID: UUID(),
            sourceEpoch: resetEpoch,
            sourceTurnIndex: 0,
            replacementEpoch: UUID(),
            pendingCall: DecodeToolCall(
                id: "call", name: "tool", argumentsJSON: "{}"),
            result: DecodeToolResult(
                callID: "call", name: "tool", content: "ok"),
            record: "record",
            commit: true)
        let checkpoint = Task {
            try await client.contextCheckpoint(checkpointRequest)
        }
        let checkpointCommand = try await reader.next()
        guard case let .contextCheckpoint(checkpointWire) = checkpointCommand else {
            Issue.record("byte-reporting client did not send a checkpoint command")
            return
        }
        let checkpointReceipt = DecodeContextCheckpointReceipt(
            checkpointID: checkpointRequest.checkpointID,
            replacementEpoch: checkpointRequest.replacementEpoch,
            committed: true,
            needed: false,
            existingPromptTokens: 0,
            replacementPromptTokens: nil,
            reserveTokens: 0,
            resultAllowanceTokens: 0,
            retainedImageCount: 0,
            retainedImageRows: 0,
            retainedFeatureBytes: 0,
            preparationSeconds: 0)
        var checkpointEvent = DecodeServiceEvent(
            kind: .contextCheckpoint,
            generationID: checkpointWire.requestID,
            loadedFamily: .qwen3_6,
            loadID: loadID,
            modelIdentity: identity,
            conversationLogicalStateBytes: 88,
            expertCacheBytes: 99,
            conversationEpoch: checkpointRequest.replacementEpoch)
        checkpointEvent.contextCheckpoint = checkpointReceipt
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            checkpointEvent))
        let receivedCheckpointReceipt = try await checkpoint.value
        #expect(receivedCheckpointReceipt == checkpointReceipt)
        #expect(client.currentInferenceMemoryBytes == 11)
        #expect(client.currentInferenceTowerBytes == 33)
        #expect(client.currentConversationLogicalStateBytes == 88)
        #expect(client.currentExpertCacheBytes == 99)

        let unload = Task { await client.unload() }
        let unloadCommand = try await reader.next()
        guard case let .unloadBound(unloadRequest) = unloadCommand else {
            Issue.record("byte-reporting client did not send a bound unload")
            return
        }
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .unloaded,
                generationID: unloadRequest.requestID,
                loadedFamily: .qwen3_6,
                loadID: loadID,
                modelIdentity: identity)))
        await unload.value

        #expect(client.currentInferenceMemoryBytes == nil)
        #expect(client.currentInferenceTowerBytes == nil)
        #expect(client.currentConversationLogicalStateBytes == nil)
        #expect(client.currentExpertCacheBytes == nil)
        client.shutdownForTermination()
        try? responses.fileHandleForWriting.close()
        try? commands.fileHandleForReading.close()
    }

    @Test func mismatchedGenerationFrameCannotResurrectReportedBytes() async throws {
        let commands = Pipe()
        let responses = Pipe()
        let client = DecodeServiceInferenceClient(
            testInput: commands.fileHandleForWriting,
            responseOutput: responses.fileHandleForReading,
            scheduleTerminationFallback: { _ in })
        let reader = ByteCommandReader(commands.fileHandleForReading)
        let modelDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "turbofieldfare-runtime-byte-generation-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDirectory) }
        let load = Task {
            try await client.ensureLoaded(
                modelDirectory: modelDirectory,
                maxContextTokens: 128,
                options: AppRuntimeOptions(),
                forceLogitsHead: false) { _ in }
        }
        let loadRequest = try await reader.nextLoad()
        let loadID = UUID()
        let identity = qwenIdentity()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .ready,
                generationID: loadRequest.requestID,
                loadAttemptID: loadRequest.attemptID,
                loadedFamily: .qwen3_6,
                loadID: loadID,
                modelIdentity: identity,
                currentMemoryBytes: 11,
                visionTowerMappedBytes: 33,
                conversationLogicalStateBytes: 444,
                expertCacheBytes: 555,
                toolThinkingEnabled: nil)))
        try await load.value

        let events = ByteEventCapture()
        let request = AppGenerationRequest(
            modelDirectory: modelDirectory,
            prompt: "current",
            maxNewTokens: 2,
            maxContextTokens: 128)
        try request.validate()
        let stream = client.generate(request)
        let consumer = Task {
            do {
                for try await event in stream { events.append(event) }
                events.finish()
                return Optional<String>.none
            } catch {
                let description = String(describing: error)
                Issue.record("generation consumer failed: \(description)")
                events.finish(error: description)
                return Optional(description)
            }
        }
        defer {
            consumer.cancel()
            client.shutdownForTermination()
            try? responses.fileHandleForWriting.close()
            try? commands.fileHandleForReading.close()
        }

        let command = try await reader.next()
        guard case let .generate(generationRequest) = command else {
            Issue.record("generation did not send a generate command")
            return
        }
        let staleGenerationID = UUID()
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .snapshot,
                generationID: staleGenerationID,
                loadedFamily: .qwen3_6,
                loadID: loadID,
                modelIdentity: identity,
                sequence: 1,
                textDelta: "stale",
                tokenCount: 1,
                currentMemoryBytes: 9_001,
                visionTowerMappedBytes: 9_002,
                conversationLogicalStateBytes: 9_003,
                expertCacheBytes: 9_004)))
        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .snapshot,
                generationID: generationRequest.generationID,
                loadedFamily: .qwen3_6,
                loadID: loadID,
                modelIdentity: identity,
                sequence: 1,
                textDelta: "current",
                tokenCount: 1)))

        guard let event = await events.next(timeout: .now() + 2) else {
            Issue.record("current generation snapshot was not delivered")
            return
        }
        guard case .token = event else {
            Issue.record("current generation snapshot did not yield a token")
            return
        }
        #expect(client.currentInferenceMemoryBytes == 11)
        #expect(client.currentInferenceTowerBytes == 33)
        #expect(client.currentConversationLogicalStateBytes == 444)
        #expect(client.currentExpertCacheBytes == 555)

        try responses.fileHandleForWriting.write(contentsOf: DecodeFrameCodec.encode(
            DecodeServiceEvent(
                kind: .finished,
                generationID: generationRequest.generationID,
                loadedFamily: .qwen3_6,
                loadID: loadID,
                modelIdentity: identity,
                stopReason: "max_tokens")))
        guard await events.waitForCompletion(timeout: .now() + 2) else {
            let failure = events.failureDescription ?? "consumer did not finish"
            Issue.record("generation consumer did not finish: \(failure)")
            return
        }
        let consumerError = await consumer.value
        let consumerDescription = consumerError ?? "unknown"
        #expect(consumerError == nil, "generation consumer failed: \(consumerDescription)")
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
            quantization: [DecodeQuantizationIdentity(
                category: "recurrent_state", storage: "fp32")],
            vision: .unavailable)
    }
}

private final class LegacyMemoryReporter: AppInferenceClient,
    AppInferenceMemoryReporting, @unchecked Sendable {
    let currentInferenceMemoryBytes: UInt64?

    init(memoryBytes: UInt64?) {
        currentInferenceMemoryBytes = memoryBytes
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func cancel() {}
}

private final class ByteReportingClient: AppInferenceClient,
    AppInferenceMemoryReporting, @unchecked Sendable {
    var currentInferenceMemoryBytes: UInt64?
    var currentInferenceResidentBytes: UInt64?
    var currentInferenceTowerBytes: UInt64?
    var currentConversationLogicalStateBytes: UInt64?
    var currentExpertCacheBytes: UInt64?

    init(memoryBytes: UInt64?, residentBytes: UInt64?, towerBytes: UInt64?,
         logicalStateBytes: UInt64?, expertCacheBytes: UInt64?) {
        currentInferenceMemoryBytes = memoryBytes
        currentInferenceResidentBytes = residentBytes
        currentInferenceTowerBytes = towerBytes
        currentConversationLogicalStateBytes = logicalStateBytes
        currentExpertCacheBytes = expertCacheBytes
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in continuation.finish() }
    }

    func cancel() {}
}

private final class ByteCommandReader: @unchecked Sendable {
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
                        throw ByteCommandReaderError.timedOut
                    }
                    continuation.resume(returning: try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handle))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
            thread.name = "TurboFieldfare.Tests.ByteCommandReader"
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

private final class ByteEventCapture: @unchecked Sendable {
    private let lock = NSLock()
    private let available = DispatchSemaphore(value: 0)
    private let completed = DispatchSemaphore(value: 0)
    private var events: [AppInferenceEvent] = []
    private var failure: String?

    func append(_ event: AppInferenceEvent) {
        lock.lock()
        events.append(event)
        lock.unlock()
        available.signal()
    }

    func finish(error: String? = nil) {
        if let error {
            lock.lock()
            failure = error
            lock.unlock()
        }
        completed.signal()
    }

    var failureDescription: String? {
        lock.lock()
        defer { lock.unlock() }
        return failure
    }

    func next(timeout: DispatchTime) async -> AppInferenceEvent? {
        await withCheckedContinuation { continuation in
            let waiter = Thread {
                guard self.available.wait(timeout: timeout) == .success else {
                    continuation.resume(returning: nil)
                    return
                }
                self.lock.lock()
                let event = self.events.isEmpty ? nil : self.events.removeFirst()
                self.lock.unlock()
                continuation.resume(returning: event)
            }
            waiter.name = "TurboFieldfare.Tests.ByteEventCapture"
            waiter.start()
        }
    }

    func waitForCompletion(timeout: DispatchTime) async -> Bool {
        await withCheckedContinuation { continuation in
            let waiter = Thread {
                continuation.resume(
                    returning: self.completed.wait(timeout: timeout) == .success)
            }
            waiter.name = "TurboFieldfare.Tests.ByteEventCompletion"
            waiter.start()
        }
    }
}

private enum ByteCommandReaderError: Error, Equatable {
    case timedOut
}
