import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

@Suite(.serialized)
struct VisionCaptureToolLoopPendingInstructionTests {
    @Test
    func pendingInstructionRecordsLocalRejectionAndStopsAfterOneProposal() async throws {
        let recorder = Recorder()
        let call = Self.tapCall(id: "pending-1")
        let completion = Self.completion(toolCalls: [call], stopReason: .toolCalls)

        let loop = VisionCaptureToolLoop()
        await #expect(throws: CancellationError.self) {
            try await loop.run(
                configuration: Self.configuration,
                activity: { event in await recorder.record(event) },
                hasPendingUserInstruction: { true },
                inference: { turn in
                    await recorder.recordInference(turn)
                    return completion
                })
        }

        let snapshot = await recorder.snapshot()
        #expect(snapshot.inferenceCount == 1)
        #expect(snapshot.turns.count == 1)
        #expect(snapshot.resultTurnCount == 0)
        #expect(snapshot.notSentRejectionCount == 1)
        #expect(snapshot.rejectionReasons.allSatisfy {
            $0.contains("before this proposed action was sent")
        })
    }

    @Test
    func ordinaryModelCancellationStillReturnsStoppedResult() async throws {
        let recorder = Recorder()
        let loop = VisionCaptureToolLoop()
        let result = try await loop.run(
            configuration: Self.configuration,
            activity: { event in await recorder.record(event) },
            inference: { turn in
                await recorder.recordInference(turn)
                return Self.completion(toolCalls: [], stopReason: .cancelled)
            })

        #expect(result.answer == "Generation stopped.")
        #expect(result.diagnostics.stopReason == .cancelled)
        #expect(await recorder.snapshot().inferenceCount == 1)
    }

    @Test
    func unrelatedInferenceFailureStillStopsWithoutRetry() async throws {
        let recorder = Recorder()
        let loop = VisionCaptureToolLoop()
        await #expect(throws: VisionCaptureAgentError.self) {
            try await loop.run(
                configuration: Self.configuration,
                activity: { event in await recorder.record(event) },
                inference: { turn in
                    await recorder.recordInference(turn)
                    throw VisionCaptureAgentError.malformedCall("synthetic failure")
                })
        }

        #expect(await recorder.snapshot().inferenceCount == 1)
    }

    private static let configuration = VisionCaptureAgentConfiguration(
        bundleIdentifier: "com.example.nestmind",
        simulatorUDID: "00000000-0000-0000-0000-000000000001",
        modelDirectory: URL(fileURLWithPath: "/tmp/visioncapture-test-model.gturbo"))

    private static func tapCall(id: String) -> AppToolCall {
        AppToolCall(
            id: id,
            name: "visioncapture_navigate",
            arguments: .object([
                "action": .string("tap"),
                "target": .string("c10"),
            ]))
    }

    private static func completion(
        toolCalls: [AppToolCall],
        stopReason: AppStopReason
    ) -> VisionCaptureModelCompletion {
        VisionCaptureModelCompletion(
            content: "",
            toolCalls: toolCalls,
            diagnostics: AppDiagnostics(
                generatedTokens: 1,
                stopReason: stopReason,
                promptTokenCount: 1,
                timeToFirstTokenSeconds: nil,
                decodeSeconds: 0.1,
                tokensPerSecond: 10,
                peakMemoryBytes: nil,
                runtimeOptions: AppRuntimeOptions()))
    }

    private actor Recorder {
        struct Snapshot: Sendable {
            let inferenceCount: Int
            let turns: [AppToolTurn]
            let resultTurnCount: Int
            let notSentRejectionCount: Int
            let rejectionReasons: [String]
        }

        private var inferenceCount = 0
        private var turns: [AppToolTurn] = []
        private var resultTurnCount = 0
        private var notSentRejectionCount = 0
        private var rejectionReasons: [String] = []

        func recordInference(_ turn: AppToolTurn) {
            inferenceCount += 1
            turns.append(turn)
            if case .results = turn { resultTurnCount += 1 }
        }

        func record(_ event: VisionCaptureActivityEvent) {
            guard case .localRejection(_, _, let reason) = event else { return }
            rejectionReasons.append(reason)
            if reason.contains("before this proposed action was sent") {
                notSentRejectionCount += 1
            }
        }

        func snapshot() -> Snapshot {
            Snapshot(
                inferenceCount: inferenceCount,
                turns: turns,
                resultTurnCount: resultTurnCount,
                notSentRejectionCount: notSentRejectionCount,
                rejectionReasons: rejectionReasons)
        }
    }
}
