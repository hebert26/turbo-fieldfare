import Darwin
import TurboFieldfare
import Foundation
import Synchronization
import TurboFieldfareAppCore
import TurboFieldfareDecodeProtocol

private final class ScopedGenerationStops: Sendable {
    private let pending = Mutex<[UUID: AppGenerationStop]>([:])

    func prepare(_ id: UUID) {
        pending.withLock { values in
            guard values[id] == nil, values.count < 8 else { return }
            values[id] = AppGenerationStop()
        }
    }

    func get(_ id: UUID) -> AppGenerationStop? { pending.withLock { $0[id] } }

    func retire(_ id: UUID) {
        let latch = pending.withLock { $0.removeValue(forKey: id) }
        latch?.finish()
    }
}

enum DecodeServiceError: Error, CustomStringConvertible {
    case attachmentOutsideStore(path: String)

    var description: String {
        switch self {
        case .attachmentOutsideStore(let path):
            "image attachment is not a staged attachment: \(path)"
        }
    }
}

@main enum TurboFieldfareDecodeServiceMain {
    static func main() async {
        let socketPath = argument(after: "--socket")
        let launchLabel = argument(after: "--launch-label")
        let handles: (input: FileHandle, output: FileHandle)
        do {
            handles = if let socketPath {
                try DecodeUnixSocket.listenAndAccept(path: socketPath)
            } else {
                (.standardInput, .standardOutput)
            }
        } catch {
            FileHandle.standardError.write(Data("Decode service transport failed: \(error)\n".utf8))
            Foundation.exit(1)
        }
        defer {
            if let socketPath { unlink(socketPath) }
            if let launchLabel { retireLaunchJob(launchLabel) }
        }

        DecodeUnixSocket.ignoreSIGPIPEProcessWide()
        let client = RealInferenceClient()
        let commands = DecodeCommandQueue()
        let scopedStops = ScopedGenerationStops()
        let input = Thread {
            do {
                while true {
                    let command = try DecodeFrameCodec.read(
                        DecodeServiceCommand.self, from: handles.input)
                    if case .generate(let request) = command,
                       request.scopedCancellation == true {
                        // Arm before enqueue: Stop can arrive before the main
                        // loop admits this request or its producer resets flags.
                        scopedStops.prepare(request.generationID)
                    }
                    if case .contextCheckpoint(let request) = command {
                        scopedStops.prepare(request.requestID)
                    }
                    if case .cancelGeneration(let id) = command {
                        scopedStops.get(id)?.requestStop()
                        continue
                    }
                    if case .cancel = command {
                        // Cooperative: end the turn at the next token boundary
                        // and keep what it produced, so the conversation can
                        // continue from it. Cancelling the task instead throws
                        // out of the decode loop and the turn is rewound.
                        client.stop()
                    }
                    commands.append(command)
                    if case .shutdown = command { break }
                }
            } catch {
                commands.close()
            }
        }
        input.name = "TurboFieldfare.DecodeService.Input"
        input.qualityOfService = .userInitiated
        input.start()

        var modelDirectory: URL?
        var loadedOptions: DecodeRuntimeOptions?
        var conversation = DecodeConversationGate()
        var pendingToolAdmission: DecodeConversationGate.Admission?
        while let command = await nextCommand(commands) {
            switch command {
            case .load(let request):
                let directory = URL(fileURLWithPath: request.modelPath)
                do {
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    try await client.ensureLoaded(
                        modelDirectory: directory,
                        maxContextTokens: request.maxContextTokens,
                        options: options,
                        forceLogitsHead: request.forceLogitsHead) { _ in }
                    guard let thinkingEnabled = await client.loadedToolThinkingEnabled,
                          thinkingEnabled == options.toolThinkingEnabled else {
                        throw AppInferenceError.modelLoadFailed(
                            "loaded tokenizer thinking mode does not match the requested setting")
                    }
                    modelDirectory = directory
                    loadedOptions = request.runtimeOptions
                    // A load builds a new runner and a new KV, so whatever
                    // lineage was open no longer has tokens behind it.
                    conversation.endLineage()
                    pendingToolAdmission = nil
                    let memory = AppMemorySampler().sample()
                    try write(DecodeServiceEvent(
                        kind: .ready, generationID: request.requestID,
                        currentMemoryBytes: memory, peakMemoryBytes: memory,
                        toolThinkingEnabled: thinkingEnabled),
                        to: handles.output)
                } catch {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.requestID,
                        error: "\(error)"), to: handles.output)
                }
            case .resetConversation(let request):
                conversation.reset(to: request.epoch)
                pendingToolAdmission = nil
                await client.resetConversation()
                // Not `try?`. The gate has already reset; if the app never
                // hears so it waits out the whole timeout for a reply that
                // cannot come, and then cannot tell that from a slow service.
                // Closing the stream makes it an EOF the client rebuilds from.
                do { try write(DecodeServiceEvent(
                    kind: .conversationReset, generationID: request.requestID,
                    conversationTokenCount: 0,
                    conversationEpoch: request.epoch), to: handles.output)
                } catch {
                    let message = "Decode service closing after a lost "
                        + "conversation reset: \(error)\n"
                    FileHandle.standardError.write(Data(message.utf8))
                    await client.unload()
                    try? handles.output.close()
                    return
                }
            case .contextCheckpoint(let request):
                defer { scopedStops.retire(request.requestID) }
                var checkpointWasCommitted = false
                do {
                    if let previous = try conversation.previousCheckpoint(request) {
                        checkpointWasCommitted = previous.committed
                        var event = DecodeServiceEvent(kind: .contextCheckpoint, generationID: request.requestID)
                        event.contextCheckpoint = previous
                        try write(event, to: handles.output)
                        continue
                    }
                    try conversation.validateCheckpoint(request)
                    guard let checkpointAdmission = pendingToolAdmission,
                          Self.matches(checkpointAdmission, epoch: request.sourceEpoch,
                                       index: request.sourceTurnIndex),
                          let stop = scopedStops.get(request.requestID) else {
                        throw DecodeConversationGate.Rejection.checkpointMismatch
                    }
                    for attachment in request.result.imageAttachments ?? [] {
                        guard AppImageAttachmentStore.contains(URL(fileURLWithPath: attachment.path)) else {
                            throw DecodeServiceError.attachmentOutsideStore(path: attachment.path)
                        }
                    }
                    let receipt = try await client.contextCheckpoint(request, stop: stop)
                    checkpointWasCommitted = receipt.committed
                    try conversation.recordCheckpoint(request, receipt: receipt)
                    if receipt.committed { pendingToolAdmission = nil }
                    var event = DecodeServiceEvent(kind: .contextCheckpoint, generationID: request.requestID,
                        conversationTokenCount: receipt.committed ? 0 : client.currentConversationTokens,
                        conversationEpoch: conversation.openEpoch)
                    event.contextCheckpoint = receipt
                    // If this acknowledgement is lost, the app stops on EOF.
                    // Retrying this identity returns the same receipt, never resets twice.
                    try write(event, to: handles.output)
                } catch {
                    if checkpointWasCommitted {
                        // The replacement may already be the only valid KV
                        // lineage. A failed acknowledgement must become EOF,
                        // never a silent wait or another admitted command.
                        FileHandle.standardError.write(Data(
                            "Decode service closing after a lost checkpoint acknowledgement: \(error)\n".utf8))
                        try? handles.output.close()
                        return
                    }
                    try? write(DecodeServiceEvent(kind: .failed, generationID: request.requestID,
                        error: "Context checkpoint failed: \(error)",
                        conversationEpoch: conversation.openEpoch), to: handles.output)
                }
            case .generate(let request):
                let generationStop = scopedStops.get(request.generationID)
                defer { scopedStops.retire(request.generationID) }
                if request.scopedCancellation == true, generationStop == nil {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "scoped cancellation request limit reached"), to: handles.output)
                    continue
                }
                guard let modelDirectory else {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "model is not loaded"), to: handles.output)
                    continue
                }
                // Prefill is chosen per request, not at load, so it must not be
                // part of this comparison: toggling it and pressing Generate was
                // refused as a mismatched session.
                var comparable = request.runtimeOptions
                comparable.prefillEnabled = loadedOptions?.prefillEnabled
                    ?? comparable.prefillEnabled
                comparable.prefillChunkTokens = loadedOptions?.prefillChunkTokens
                    ?? comparable.prefillChunkTokens
                guard comparable == loadedOptions else {
                    try? write(DecodeServiceEvent(
                        kind: .failed, generationID: request.generationID,
                        error: "generation runtime options do not match the loaded session"),
                        to: handles.output)
                    continue
                }
                // Fail closed before the model is touched. The decision
                // lives in `DecodeConversationGate`, where its boundary cases
                // are tested without a socket or a model.
                let admission: DecodeConversationGate.Admission
                if case .results = request.toolTurn {
                    guard let pendingToolAdmission,
                          Self.matches(
                            pendingToolAdmission,
                            epoch: request.conversationEpoch,
                            index: request.turnIndex) else {
                        try? write(DecodeServiceEvent(
                            kind: .failed, generationID: request.generationID,
                            error: "tool results do not match the pending app turn",
                            conversationEpoch: conversation.openEpoch), to: handles.output)
                        continue
                    }
                    admission = pendingToolAdmission
                } else {
                    guard pendingToolAdmission == nil else {
                        try? write(DecodeServiceEvent(
                            kind: .failed, generationID: request.generationID,
                            error: "the pending tool turn needs results before another user turn",
                            conversationEpoch: conversation.openEpoch), to: handles.output)
                        continue
                    }
                    switch conversation.admit(request) {
                    case .success(let value):
                        admission = value
                    case .failure(let rejection):
                        try? write(DecodeServiceEvent(
                            kind: .failed, generationID: request.generationID,
                            error: rejection.message,
                            conversationEpoch: conversation.openEpoch), to: handles.output)
                        continue
                    }
                }
                let isConversationTurn: Bool
                if case .turn = admission { isConversationTurn = true }
                else { isConversationTurn = false }
                // Measurement labels are defined for the agreed chunked-128
                // path. Unsupported settings run normally with an explicit
                // capture status and no collector allocation.
                let measurementSupported = request.runtimeOptions.prefillEnabled
                    && request.runtimeOptions.prefillChunkTokens == 128
                let measurementCapture = request.runtimeMeasurementCapture != nil
                    && measurementSupported ? RuntimeMeasurementCapture() : nil
                let outbox = DecodeServiceOutbox(
                    generationID: request.generationID,
                    towerBytes: { client.currentVisionTowerBytes },
                    conversationTokens: {
                        isConversationTurn ? client.currentConversationTokens : nil
                    },
                    measurementRequest: request.runtimeMeasurementCapture,
                    measurementCapture: measurementCapture)
                let writerFinished = DispatchSemaphore(value: 0)
                let writer = Thread {
                    defer { writerFinished.signal() }
                    do { try outbox.runWriter(to: handles.output) }
                    catch {
                        FileHandle.standardError.write(Data("IPC writer failed: \(error)\n".utf8))
                    }
                }
                writer.name = "TurboFieldfare.DecodeService.Writer"
                writer.qualityOfService = .userInitiated
                writer.start()

                do {
                    // The trust boundary: these paths arrive over a socket, and
                    // this process opens and hashes whatever they name. Without
                    // this, a peer could use the service to report on any file
                    // the user can read.
                    if let outside = (request.imageAttachments ?? []).first(where: {
                        !AppImageAttachmentStore.contains(URL(fileURLWithPath: $0.path))
                    }) {
                        throw DecodeServiceError.attachmentOutsideStore(
                            path: outside.path)
                    }
                    let options = try appRuntimeOptions(request.runtimeOptions)
                    // The conversation already in the KV is what the image
                    // budget has to fit around; reserving zero admits an image
                    // that only fits an empty context.
                    let carried = await client.conversationTokenCount
                    let continues = isConversationTurn
                    var generation = AppGenerationRequest(
                        modelDirectory: modelDirectory, prompt: request.prompt,
                        imageAttachments: (request.imageAttachments ?? []).map {
                            AppImageAttachment(
                                id: $0.id,
                                fileURL: URL(fileURLWithPath: $0.path),
                                displayName: $0.displayName,
                                encodedBytes: $0.encodedBytes,
                                sha256: $0.sha256)
                        },
                        maxNewTokens: request.maxNewTokens,
                        maxContextTokens: request.maxContextTokens,
                        temperature: request.temperature,
                        topK: request.topK,
                        topP: request.topP,
                        repetitionPenalty: request.repetitionPenalty,
                        runtimeOptions: options,
                        continuesConversation: continues,
                        conversationTokens: continues ? carried : 0,
                        conversationEpoch: request.conversationEpoch,
                        turnIndex: request.turnIndex,
                        toolTurn: try appToolTurn(request.toolTurn))
                    generation.captureToolFailureEvidence = request.captureToolFailureEvidence == true
                    generation.captureGPUCompletionTiming = request.captureGPUCompletionTiming == true
                    generation.runtimeMeasurementCapture = request.runtimeMeasurementCapture
                    var terminalStopReason: AppStopReason?
                    for try await event in client.generate(
                        generation, measurementCapture: measurementCapture,
                        generationStop: generationStop) {
                        if case .finished(let diagnostics) = event {
                            terminalStopReason = diagnostics.stopReason
                        }
                        outbox.publish(event)
                    }
                    // Reached only when the stream completed. A turn that threw
                    // was rewound by the conversation (or broke its lineage), so
                    // its tokens are not in the KV and it must not advance the
                    // order the next turn has to match — the app does not count
                    // it either, and a one-sided count rejects every later turn.
                    // A turn stopped by the user does reach here: it ends at a
                    // token boundary with its partial reply committed.
                    if case .checkpoint = request.toolTurn { conversation.checkpointResumed() }
                    if request.toolTurn != nil,
                       terminalStopReason == .toolCalls {
                        pendingToolAdmission = admission
                    } else {
                        conversation.commit(admission)
                        pendingToolAdmission = nil
                    }
                    outbox.finish()
                } catch {
                    outbox.finish(error: error)
                }
                await withCheckedContinuation { continuation in
                    DispatchQueue.global(qos: .userInitiated).async {
                        writerFinished.wait()
                        continuation.resume()
                    }
                }
            case .cancel, .cancelGeneration(_):
                break
            case .unload(let requestID):
                await client.unload()
                modelDirectory = nil
                loadedOptions = nil
                conversation.endLineage()
                pendingToolAdmission = nil
                try? write(DecodeServiceEvent(
                    kind: .unloaded, generationID: requestID), to: handles.output)
            case .shutdown:
                await client.unload()
                return
            }
        }
    }

    private static func nextCommand(_ commands: DecodeCommandQueue)
        async -> DecodeServiceCommand? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: commands.next())
            }
        }
    }

    private static func write(_ event: DecodeServiceEvent,
                              to handle: FileHandle) throws {
        try handle.write(contentsOf: DecodeFrameCodec.encode(event))
    }

    private static func appRuntimeOptions(_ options: DecodeRuntimeOptions) throws
        -> AppRuntimeOptions {
        guard let cachePolicy = AppExpertCachePolicy(
            rawValue: options.expertCachePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown expert cache policy \(options.expertCachePolicy)")
        }
        guard let rdadvisePolicy = AppRDAdvicePolicy(
            rawValue: options.rdadvisePolicy) else {
            throw AppInferenceError.invalidRequest(
                "unknown RDADVISE policy \(options.rdadvisePolicy)")
        }
        guard let modelVerification = AppModelVerification(
            rawValue: options.modelVerification) else {
            throw AppInferenceError.invalidRequest(
                "unknown model verification \(options.modelVerification)")
        }
        // An unknown policy is a request for behaviour that does not exist;
        // absent means the shipped default.
        guard let visionResidencyPolicy = VisionResidencyPolicy(
            rawValue: options.visionResidencyPolicy ?? VisionResidencyPolicy.onDemand.rawValue)
        else {
            throw AppInferenceError.invalidRequest(
                "unknown vision residency policy \(options.visionResidencyPolicy ?? "")")
        }
        let resolved = AppRuntimeOptions(
            expertCacheSlots: options.expertCacheSlots,
            expertCachePolicy: cachePolicy,
            prefillEnabled: options.prefillEnabled,
            prefillChunkTokens: options.prefillChunkTokens,
            rdadvisePolicy: rdadvisePolicy,
            modelVerification: modelVerification,
            visionResidencyPolicy: visionResidencyPolicy,
            toolThinkingEnabled: options.toolThinkingEnabled ?? GFTokenizer.toolThinkingEnabled)
        try resolved.validate()
        return resolved
    }

    private static func appToolTurn(_ turn: DecodeToolTurn?) throws
        -> AppToolTurn? {
        switch turn {
        case .checkpoint(let id):
            return .checkpoint(id)
        case .user(let developerPrompt, let tools):
            return .user(
                developerPrompt: developerPrompt,
                tools: try tools.map { tool in
                    guard let data = tool.parametersJSON.data(using: .utf8) else {
                        throw AppInferenceError.invalidRequest(
                            "tool parameters are not UTF-8 JSON")
                    }
                    return AppToolDefinition(
                        name: tool.name,
                        description: tool.description,
                        parameters: try JSONDecoder().decode(
                            JSONValue.self, from: data))
                })
        case .results(let results):
            return .results(results.map {
                AppToolResult(
                    callID: $0.callID,
                    name: $0.name,
                    content: $0.content,
                    imageAttachments: ($0.imageAttachments ?? []).map {
                        AppImageAttachment(
                            id: $0.id, fileURL: URL(fileURLWithPath: $0.path),
                            displayName: $0.displayName, encodedBytes: $0.encodedBytes, sha256: $0.sha256)
                    })
            })
        case nil:
            return nil
        }
    }

    private static func matches(
        _ admission: DecodeConversationGate.Admission,
        epoch: UUID?,
        index: Int?
    ) -> Bool {
        guard case .turn(let admittedEpoch, let admittedIndex) = admission else {
            return false
        }
        return admittedEpoch == epoch && admittedIndex == index
    }

    private static func argument(after name: String) -> String? {
        let arguments = CommandLine.arguments
        guard let index = arguments.firstIndex(of: name),
              arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }

    private static func retireLaunchJob(_ label: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
    }
}
