import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct AppModelAgentInstructionTests {
    @MainActor
    @Test
    func pendingInstructionStartsExactTextInFreshLineageAfterModelCompletionRace() async throws {
        let client = PendingInstructionClient()
        let directory = FileManager.default.temporaryDirectory
        let model = AppModel(modelDirectory: directory, client: client)
        model.modelPathText = directory.path
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        model.setAgentModeEnabled(true)
        model.promptText = "Start the original task"
        model.run()

        #expect(await client.waitForRequestCount(1))
        model.promptText = "Open the settings screen"
        model.submitPrompt()
        #expect(model.isAgentInstructionPending)
        await client.releaseFirstCompletion()

        #expect(await client.waitForRequestCount(2))
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while model.isRunning && clock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(!model.isRunning)
        #expect(!model.isAgentInstructionPending)
        #expect(model.error == nil)

        let requests = await client.requests()
        #expect(requests.count == 2)
        guard requests.count == 2 else { return }
        #expect(requests.map(\.prompt) == [
            "Start the original task",
            "Open the settings screen",
        ])
        #expect(requests[0].conversationEpoch != requests[1].conversationEpoch)
        #expect(requests.allSatisfy { request in
            if case .results = request.toolTurn { return false }
            return true
        })
        #expect(model.archivedPairs.contains { pair in
            pair.user.text == "Start the original task"
        })
    }

    private actor Gate {
        private var released = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func wait() async {
            if released { return }
            await withCheckedContinuation { continuation in
                waiters.append(continuation)
            }
        }

        func release() {
            released = true
            let waiting = waiters
            waiters.removeAll()
            for continuation in waiting { continuation.resume() }
        }
    }

    private final class PendingInstructionClient: AppModelLifecycleClient,
        @unchecked Sendable {
        private let gate = Gate()
        private let lock = NSLock()
        private var recordedRequests: [AppGenerationRequest] = []

        func ensureLoaded(
            modelDirectory: URL,
            maxContextTokens: Int,
            options: AppRuntimeOptions,
            forceLogitsHead: Bool,
            onState: @escaping @Sendable (AppModelLoadState) -> Void
        ) async throws {
            onState(.ready(modelDirectory: modelDirectory.standardizedFileURL,
                           loadSeconds: 0))
        }

        func unload() async {}
        func resetConversation(epoch: UUID) async throws {}
        func shutdownForTermination() {}
        func cancel() {}

        func generate(_ request: AppGenerationRequest)
            -> AsyncThrowingStream<AppInferenceEvent, Error> {
            let index = lock.withLock {
                recordedRequests.append(request)
                return recordedRequests.count - 1
            }
            return AsyncThrowingStream { continuation in
                Task {
                    if index == 0 {
                        continuation.yield(.toolCall(Self.pendingCall))
                        await gate.wait()
                        continuation.yield(.finished(Self.diagnostics(.toolCalls)))
                    } else {
                        continuation.yield(.token(AppTokenEvent(
                            index: 0,
                            textDelta: "Done",
                            elapsedDecodeSeconds: 0.1)))
                        continuation.yield(.finished(Self.diagnostics(.endOfTurn)))
                    }
                    continuation.finish()
                }
            }
        }

        func waitForRequestCount(_ count: Int) async -> Bool {
            let clock = ContinuousClock()
            let deadline = clock.now.advanced(by: .seconds(2))
            while clock.now < deadline {
                if lock.withLock({ recordedRequests.count >= count }) { return true }
                try? await Task.sleep(for: .milliseconds(1))
            }
            return false
        }

        func releaseFirstCompletion() async {
            await gate.release()
        }

        func requests() -> [AppGenerationRequest] {
            lock.withLock { recordedRequests }
        }

        private static let pendingCall = AppToolCall(
            id: "pending-call-1",
            name: "visioncapture_navigate",
            arguments: .object([
                "action": .string("tap"),
                "target": .string("c10"),
            ]))

        private static func diagnostics(_ reason: AppStopReason) -> AppDiagnostics {
            AppDiagnostics(
                generatedTokens: 1,
                stopReason: reason,
                promptTokenCount: 1,
                timeToFirstTokenSeconds: nil,
                decodeSeconds: 0.1,
                tokensPerSecond: 10,
                peakMemoryBytes: nil,
                runtimeOptions: AppRuntimeOptions())
        }
    }
}
