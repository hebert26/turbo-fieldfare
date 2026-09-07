import Foundation

public struct DecodeRuntimeOptions: Codable, Sendable, Equatable {
    public var expertCacheSlots: Int
    public var expertCachePolicy: String
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: String
    public var modelVerification: String
    public var visionResidencyPolicy: String?
    public var toolThinkingEnabled: Bool?

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: String = "lfu",
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                rdadvisePolicy: String = "off",
                modelVerification: String = "full-sha256",
                visionResidencyPolicy: String? = nil,
                toolThinkingEnabled: Bool? = nil) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
        self.visionResidencyPolicy = visionResidencyPolicy
        self.toolThinkingEnabled = toolThinkingEnabled
    }
}

public struct DecodeImageAttachment: Codable, Sendable, Equatable {
    public var id: UUID
    public var path: String
    public var displayName: String
    public var encodedBytes: Int
    public var sha256: String

    public init(id: UUID, path: String, displayName: String,
                encodedBytes: Int, sha256: String) {
        self.id = id
        self.path = path
        self.displayName = displayName
        self.encodedBytes = encodedBytes
        self.sha256 = sha256
    }
}

public struct DecodeLoadRequest: Codable, Sendable {
    public var modelPath: String
    public var maxContextTokens: Int
    public var runtimeOptions: DecodeRuntimeOptions
    public var forceLogitsHead: Bool
    public var requestID: UUID

    public init(modelPath: String, maxContextTokens: Int,
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                forceLogitsHead: Bool = false,
                requestID: UUID = UUID()) {
        self.modelPath = modelPath
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.forceLogitsHead = forceLogitsHead
        self.requestID = requestID
    }
}

public struct DecodeGenerationRequest: Codable, Sendable {
    /// One user turn, never a rendered transcript. In conversation mode the
    /// service appends exactly this onto the retained KV; re-rendering history
    /// here would destabilise the token prefix the cache is built on.
    public var prompt: String
    public var imageAttachments: [DecodeImageAttachment]?
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var temperature: Float
    /// Carried explicitly, and optional because nil means "no cut". Leaving
    /// them off the wire did not fall back to the sender's settings: the
    /// service rebuilt the request from its own initializer defaults, so
    /// turning Top-K off, or setting any value other than 64 / 0.95, was
    /// silently ignored on the only client the app ships with.
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var runtimeOptions: DecodeRuntimeOptions
    public var generationID: UUID
    /// The conversation this turn belongs to, or nil for the one-shot path that
    /// resets the KV before prefilling.
    ///
    /// Checked before the model is touched. A generate carrying a stale epoch
    /// is a turn composed against a conversation the user has since replaced,
    /// and appending it to the new lineage would put a message the user never
    /// sent into the model's context.
    public var conversationEpoch: UUID?
    /// Position of this turn within `conversationEpoch`, zero-based. The
    /// service rejects a turn that does not match the number of turns it has
    /// committed, so a dropped or duplicated turn is a refusal rather than a
    /// silently reordered conversation.
    public var turnIndex: Int?
    public var toolTurn: DecodeToolTurn?
    /// Absent means off, including requests from older clients.
    public var captureToolFailureEvidence: Bool?
    /// Absent means off. Enabled only by the process-opt-in agent trace.
    public var captureGPUCompletionTiming: Bool?

    public init(prompt: String,
                imageAttachments: [DecodeImageAttachment]? = nil,
                maxNewTokens: Int, maxContextTokens: Int,
                temperature: Float, topK: Int? = nil, topP: Float? = nil,
                repetitionPenalty: Float = 1,
                runtimeOptions: DecodeRuntimeOptions = DecodeRuntimeOptions(),
                generationID: UUID = UUID(),
                conversationEpoch: UUID? = nil,
                turnIndex: Int? = nil,
                toolTurn: DecodeToolTurn? = nil) {
        self.conversationEpoch = conversationEpoch
        self.turnIndex = turnIndex
        self.toolTurn = toolTurn
        self.prompt = prompt
        self.imageAttachments = imageAttachments
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.runtimeOptions = runtimeOptions
        self.generationID = generationID
    }
}

public struct DecodeToolDefinition: Codable, Equatable, Sendable {
    public var name: String
    public var description: String
    public var parametersJSON: String

    public init(name: String, description: String, parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}

public struct DecodeToolResult: Codable, Equatable, Sendable {
    public var callID: String
    public var name: String
    public var content: String
    /// Absent in older clients and text-only results.
    public var imageAttachments: [DecodeImageAttachment]?

    public init(callID: String, name: String, content: String,
                imageAttachments: [DecodeImageAttachment]? = nil) {
        self.callID = callID
        self.name = name
        self.content = content
        self.imageAttachments = imageAttachments
    }
}

public enum DecodeToolTurn: Codable, Equatable, Sendable {
    case user(developerPrompt: String?, tools: [DecodeToolDefinition])
    case results([DecodeToolResult])
}

public struct DecodeToolCall: Codable, Equatable, Sendable {
    public var id: String
    public var name: String
    public var argumentsJSON: String

    public init(id: String, name: String, argumentsJSON: String) {
        self.id = id
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

/// Starts a new conversation lineage: the KV is dropped and `epoch` becomes the
/// only value the service will accept on a generate.
public struct DecodeResetConversationRequest: Codable, Sendable, Equatable {
    public var epoch: UUID
    public var requestID: UUID

    public init(epoch: UUID = UUID(), requestID: UUID = UUID()) {
        self.epoch = epoch
        self.requestID = requestID
    }
}

public enum DecodeServiceCommand: Codable, Sendable {
    case load(DecodeLoadRequest)
    case generate(DecodeGenerationRequest)
    case resetConversation(DecodeResetConversationRequest)
    case cancel
    case unload(UUID)
    case shutdown
}

public enum DecodeServiceEventKind: String, Codable, Sendable {
    case loading
    case ready
    case prefill
    case snapshot
    case toolCall
    /// Carries a live memory reading while image encoding or another silent
    /// phase has not produced progress or tokens yet.
    case memory
    case finished
    case cancelled
    case failed
    /// The KV no longer matches the recorded conversation, so this lineage is
    /// unusable and only a reset can recover it. Distinct from `failed`, which
    /// leaves the conversation resumable.
    case lineageLost
    case conversationReset
    case unloaded
}

/// Completed-buffer timing coverage. A duration is absent for incomplete coverage.
public struct DecodeGPUCompletionTiming: Codable, Sendable, Equatable {
    public var millisecondsPerForward: Double?
    public var validCount: UInt64
    public var expectedCount: UInt64

    public init(millisecondsPerForward: Double?, validCount: UInt64, expectedCount: UInt64) {
        self.millisecondsPerForward = millisecondsPerForward
        self.validCount = validCount
        self.expectedCount = expectedCount
    }
}

public struct DecodeRunnerDiagnostics: Codable, Sendable, Equatable {
    public var cb1MillisecondsPerToken: Double
    public var routerWaitMillisecondsPerToken: Double?
    public var gpuCompletionTiming: [String: DecodeGPUCompletionTiming]?
    public var ioMillisecondsPerToken: Double
    public var cb2MillisecondsPerToken: Double
    public var headMillisecondsPerToken: Double
    public var rdadviseMillisecondsPerToken: Double
    public var rdadviseCallsPerToken: Double
    public var rdadviseMegabytesPerToken: Double
    public var rdadviseSkippedPerToken: Double
    public var rdadviseFailures: UInt64

    public init(cb1MillisecondsPerToken: Double,
                routerWaitMillisecondsPerToken: Double? = nil,
                gpuCompletionTiming: [String: DecodeGPUCompletionTiming]? = nil,
                ioMillisecondsPerToken: Double,
                cb2MillisecondsPerToken: Double,
                headMillisecondsPerToken: Double,
                rdadviseMillisecondsPerToken: Double,
                rdadviseCallsPerToken: Double,
                rdadviseMegabytesPerToken: Double,
                rdadviseSkippedPerToken: Double,
                rdadviseFailures: UInt64) {
        self.cb1MillisecondsPerToken = cb1MillisecondsPerToken
        self.routerWaitMillisecondsPerToken = routerWaitMillisecondsPerToken
        self.gpuCompletionTiming = gpuCompletionTiming
        self.ioMillisecondsPerToken = ioMillisecondsPerToken
        self.cb2MillisecondsPerToken = cb2MillisecondsPerToken
        self.headMillisecondsPerToken = headMillisecondsPerToken
        self.rdadviseMillisecondsPerToken = rdadviseMillisecondsPerToken
        self.rdadviseCallsPerToken = rdadviseCallsPerToken
        self.rdadviseMegabytesPerToken = rdadviseMegabytesPerToken
        self.rdadviseSkippedPerToken = rdadviseSkippedPerToken
        self.rdadviseFailures = rdadviseFailures
    }
}

public struct DecodePrefillDiagnostics: Codable, Sendable, Equatable {
    public var requestedMode: String
    public var executedMode: String
    public var kvStorageMode: String?
    public var chunkCompleteness: String
    public var unsupportedReason: String?

    public init(requestedMode: String, executedMode: String,
                kvStorageMode: String?, chunkCompleteness: String,
                unsupportedReason: String?) {
        self.requestedMode = requestedMode
        self.executedMode = executedMode
        self.kvStorageMode = kvStorageMode
        self.chunkCompleteness = chunkCompleteness
        self.unsupportedReason = unsupportedReason
    }
}

/// Fixed-size decoder diagnostics. Contains no generated token or channel text.
public struct DecodeStructuredProgress: Codable, Equatable, Sendable {
    public var stage: String
    public var thinkingTokens: Int
    public var toolCallTokens: Int
    public var visibleResponseTokens: Int
    public var channelLabelTokens: Int
    public var unknownHiddenChannelTokens: Int

    public init(stage: String, thinkingTokens: Int, toolCallTokens: Int,
                visibleResponseTokens: Int, channelLabelTokens: Int,
                unknownHiddenChannelTokens: Int) {
        self.stage = stage
        self.thinkingTokens = thinkingTokens
        self.toolCallTokens = toolCallTokens
        self.visibleResponseTokens = visibleResponseTokens
        self.channelLabelTokens = channelLabelTokens
        self.unknownHiddenChannelTokens = unknownHiddenChannelTokens
    }
}

/// Latest bounded thought window for display only, excluded from diagnostics/history.
public struct DecodeThinkingPreview: Codable, Equatable, Sendable {
    public var text: String
    public var earlierTextOmitted: Bool

    public init(text: String, earlierTextOmitted: Bool) {
        self.text = text
        self.earlierTextOmitted = earlierTextOmitted
    }
}

/// Raw, unvalidated draft. Display/opt-in trace only, never a tool request.
public struct DecodeToolCallPreview: Codable, Equatable, Sendable {
    public var text: String
    public var middleTextOmitted: Bool

    public init(text: String, middleTextOmitted: Bool) {
        self.text = text
        self.middleTextOmitted = middleTextOmitted
    }
}

public struct DecodeServiceEvent: Codable, Sendable {
    public var kind: DecodeServiceEventKind
    public var generationID: UUID
    public var sequence: UInt64
    public var textDelta: String
    public var tokenCount: Int
    public var promptTokenCount: Int?
    public var computedPrefillTokens: Int?
    public var prefillDone: Int?
    public var prefillTotal: Int?
    public var prefillSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var decodeSeconds: Double
    public var tokensPerSecond: Double
    public var stopReason: String?
    public var error: String?
    public var currentMemoryBytes: UInt64?
    public var peakMemoryBytes: UInt64?
    /// Bytes of image tower the inference process holds mapped, or nil when
    /// it has no vision runtime.
    public var visionTowerMappedBytes: UInt64?
    /// Prompt tokens served from the retained KV instead of being prefilled
    /// again. Reported per turn because a cache nobody can see is a cache that
    /// can regress to nothing without a single bug report.
    public var cachedPromptTokens: Int?
    /// Tokens the conversation's KV holds after this turn, for the context
    /// gauge. Nil outside conversation mode.
    public var conversationTokenCount: Int?
    /// The lineage this event belongs to, so a late event from a replaced
    /// conversation can be dropped rather than shown under the new one.
    public var conversationEpoch: UUID?
    public var prefill: DecodePrefillDiagnostics?
    public var runner: DecodeRunnerDiagnostics?
    public var toolCall: DecodeToolCall?
    /// Diagnostic attachment only. Never part of user-visible error text.
    public var parserFailureJSON: String?
    /// Runtime-confirmed rollback and zero captured calls. Absent means no retry.
    public var parserFailureCanRegenerateToolResult: Bool?
    /// Process-captured tool-template choice, acknowledged on model readiness.
    public var toolThinkingEnabled: Bool?
    public var structuredProgress: DecodeStructuredProgress?
    public var thinkingPreview: DecodeThinkingPreview?
    public var toolCallPreview: DecodeToolCallPreview?

    public init(kind: DecodeServiceEventKind, generationID: UUID,
                sequence: UInt64 = 0, textDelta: String = "",
                tokenCount: Int = 0, promptTokenCount: Int? = nil,
                computedPrefillTokens: Int? = nil,
                prefillDone: Int? = nil, prefillTotal: Int? = nil,
                prefillSeconds: Double? = nil,
                timeToFirstTokenSeconds: Double? = nil,
                decodeSeconds: Double = 0, tokensPerSecond: Double = 0,
                stopReason: String? = nil, error: String? = nil,
                currentMemoryBytes: UInt64? = nil, peakMemoryBytes: UInt64? = nil,
                visionTowerMappedBytes: UInt64? = nil,
                cachedPromptTokens: Int? = nil,
                conversationTokenCount: Int? = nil,
                conversationEpoch: UUID? = nil,
                prefill: DecodePrefillDiagnostics? = nil,
                runner: DecodeRunnerDiagnostics? = nil,
                toolCall: DecodeToolCall? = nil,
                toolThinkingEnabled: Bool? = nil,
                structuredProgress: DecodeStructuredProgress? = nil,
                thinkingPreview: DecodeThinkingPreview? = nil,
                toolCallPreview: DecodeToolCallPreview? = nil) {
        self.kind = kind
        self.generationID = generationID
        self.sequence = sequence
        self.textDelta = textDelta
        self.tokenCount = tokenCount
        self.promptTokenCount = promptTokenCount
        self.computedPrefillTokens = computedPrefillTokens
        self.prefillDone = prefillDone
        self.prefillTotal = prefillTotal
        self.prefillSeconds = prefillSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds
        self.tokensPerSecond = tokensPerSecond
        self.stopReason = stopReason
        self.error = error
        self.currentMemoryBytes = currentMemoryBytes
        self.peakMemoryBytes = peakMemoryBytes
        self.visionTowerMappedBytes = visionTowerMappedBytes
        self.cachedPromptTokens = cachedPromptTokens
        self.conversationTokenCount = conversationTokenCount
        self.conversationEpoch = conversationEpoch
        self.prefill = prefill
        self.runner = runner
        self.toolCall = toolCall
        self.toolThinkingEnabled = toolThinkingEnabled
        self.structuredProgress = structuredProgress
        self.thinkingPreview = thinkingPreview
        self.toolCallPreview = toolCallPreview
    }
}

public enum DecodeFrameCodec {
    public static let maximumPayloadBytes = 4 * 1_024 * 1_024

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let payload = try JSONEncoder().encode(value)
        guard payload.count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        var length = UInt32(payload.count).littleEndian
        var frame = withUnsafeBytes(of: &length) { Data($0) }
        frame.append(payload)
        return frame
    }

    public static func read<T: Decodable>(_ type: T.Type, from handle: FileHandle) throws -> T {
        let header = try readExactly(4, from: handle)
        let count = header.withUnsafeBytes { raw -> UInt32 in
            raw.loadUnaligned(as: UInt32.self).littleEndian
        }
        guard count <= maximumPayloadBytes else { throw DecodeFrameError.oversized }
        let payload = try readExactly(Int(count), from: handle)
        return try JSONDecoder().decode(type, from: payload)
    }

    private static func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            guard let chunk = try handle.read(upToCount: count - result.count), !chunk.isEmpty else {
                throw DecodeFrameError.unexpectedEOF
            }
            result.append(chunk)
        }
        return result
    }
}

public enum DecodeFrameError: Error {
    case oversized
    case unexpectedEOF
}
