import Foundation
import Metal

public enum MultimodalConversationError: Error, CustomStringConvertible {
    case closed
    case busy
    case lineageBroken
    case lineageRecoveryFailed(reason: String)
    case emptyTurn
    case toolModeRequiresNewConversation
    case invalidToolContinuation
    case noObservableToolProgress(limit: Int)
    case contextExhausted(prompt: Int, maxContext: Int)
    case imageUnavailable(reason: String?)
    case unsupportedFamily
    case tokenCodecUnavailable

    public var description: String {
        switch self {
        case .closed: "conversation is closed"
        case .busy: "a turn is already generating on this conversation"
        case .lineageBroken:
            "generation failed partway, so the KV no longer matches this "
                + "conversation; call reset() to start over"
        case .lineageRecoveryFailed(let reason):
            "generation failed partway and the KV could not be restored; "
                + "call reset() to start over. \(reason)"
        case .emptyTurn: "a turn needs text or an image"
        case .toolModeRequiresNewConversation:
            "tool mode must start in a new conversation"
        case .invalidToolContinuation:
            "tool results do not match the pending tool calls"
        case .noObservableToolProgress(let limit):
            "structured generation produced \(limit) consecutive tokens without "
                + "visible answer text or a completed tool call"
        case .contextExhausted(let prompt, let maxContext):
            "conversation needs \(prompt) tokens, beyond the \(maxContext)-token context"
        case .imageUnavailable(let reason):
            reason.map { "image support is unavailable: \($0)" }
                ?? "image support is unavailable: no companion pack is installed"
        case .unsupportedFamily:
            "this conversation operation is unavailable for the loaded model family"
        case .tokenCodecUnavailable:
            "Qwen string, tool, and image conversation input is unavailable until its chat codec is installed"
        }
    }
}

enum MultimodalConversationKVRecovery {
    static func restoreAfterFailure(
        positionBefore: Int,
        generationError: Error,
        rewind: (Int) throws -> Void,
        reset: () -> Void
    ) throws {
        guard positionBefore > 0 else {
            reset()
            return
        }
        do {
            try rewind(positionBefore)
        } catch {
            throw MultimodalConversationError.lineageRecoveryFailed(
                reason: "The turn failed with \(generationError). Rewinding to "
                    + "token \(positionBefore) then failed with \(error).")
        }
    }

    static func trimHiddenStopTokens(
        _ tokenIDs: [Int32],
        withheld: Int,
        rewind: (Int) throws -> Void,
        reset: () -> Void
    ) throws -> [Int32] {
        guard withheld > 0, withheld <= tokenIDs.count else {
            throw MultimodalConversationError.lineageRecoveryFailed(
                reason: "Stop-string cleanup reported \(withheld) hidden tokens "
                    + "for a \(tokenIDs.count)-token KV.")
        }
        let target = tokenIDs.count - withheld
        if target == 0 {
            reset()
        } else {
            do {
                try rewind(target)
            } catch {
                throw MultimodalConversationError.lineageRecoveryFailed(
                    reason: "Removing stop-string tokens required rewinding to "
                        + "token \(target), which failed with \(error).")
            }
        }
        return Array(tokenIDs.prefix(target))
    }
}

enum MultimodalConversationCanonicalization {
    static func userContent(_ parts: [MultimodalContinuationPart]) throws -> String {
        var content = ""
        for part in parts {
            switch part {
            case .text(let value):
                guard !value.contains(MultimodalPromptRenderer.placeholder) else {
                    throw MultimodalPromptRendererError.reservedImageMarker
                }
                content += value
            case .image:
                content += MultimodalPromptRenderer.placeholder
            }
        }
        return content
    }

    static func longestReusablePrefix(
        cached: [Int32],
        canonical: [Int32],
        imageTokenRanges: [Range<Int>]
    ) -> Int {
        var count = 0
        for (old, new) in zip(cached, canonical) {
            guard old == new else { break }
            count += 1
        }
        for range in imageTokenRanges {
            if count > range.lowerBound, count < range.upperBound {
                return range.lowerBound
            }
        }
        return count
    }
}

public struct MultimodalTurnResult: Sendable {
    public let text: String
    public let promptTokens: Int
    /// Tokens served from the retained KV rather than prefilled again.
    public let cachedTokens: Int
    /// Tokens this turn put through prefill. `RawCompletion` computes it as
    /// `promptIds.count - cachedPromptTokens`, so it is a derivation of the two
    /// figures beside it, not an independent count: asserting that it equals
    /// `promptTokens - cachedTokens` cannot fail and proves nothing. What it is
    /// good for is reporting — the figure a reader wants when asking what a
    /// turn cost. The claim that the KV was reused rests on `cachedTokens`
    /// matching the previous turn's `kvTokens`, and on `prefillSeconds`.
    public let computedPrefillTokens: Int
    /// Tokens the KV holds after this turn. Not `promptTokens +
    /// completionTokens`: a run that stops on max tokens or is cancelled holds
    /// its final token outside the KV, so that sum is one too many.
    public let kvTokens: Int
    public let completionTokens: Int
    /// Why the turn ended; `.cancelled` when `checkCancellation` stopped the
    /// decode mid-turn, in which case `text` holds the partial reply and the
    /// conversation remains resumable.
    public let reason: StopReason
    /// Wall time spent prefilling this turn's own tokens. Reported so a caller
    /// can show what resuming actually saved rather than asserting that it did.
    public let prefillSeconds: Double
    public let decodeSeconds: Double
}

public struct StructuredConversationTurnResult: Sendable {
    public let turn: MultimodalTurnResult
    public let toolCalls: [ParsedToolCall]
}

public struct ConversationToolResult: Equatable, Sendable {
    public let callID: String
    public let name: String
    public let content: String
    public let images: [URL]

    public init(callID: String, name: String, content: String, images: [URL] = []) {
        self.callID = callID
        self.name = name
        self.content = content
        self.images = images
    }
}

public struct ConversationCheckpointReceipt: Sendable {
    public let needed: Bool
    public let existingPromptTokens: Int
    public let replacementPromptTokens: Int?
    public let reserveTokens: Int
    public let resultAllowanceTokens: Int
    public let retainedImageCount: Int
    public let retainedImageRows: Int
    public let retainedFeatureBytes: Int
    public let performanceMinimumSavingsTokens: Int?
}

/// A stateful multi-turn conversation that owns its own KV lineage.
///
/// The server has to *match* a stateless request against a cached prefix and
/// fail closed when it cannot. A conversation does not: it appended every token
/// in the KV itself, so it always knows the boundary and always resumes. Each
/// turn prefills only the new tokens, and an image is encoded once, when its
/// turn is appended.
public actor GemmaMultimodalConversation {
    private struct ToolState {
        var messages: [GFTokenizer.Message]
        let tools: [GFTokenizer.FunctionDefinition]
        var awaitingResults: Bool
    }

    private struct ProvisionalToolResultPrefix {
        let committedTokenCount: Int
        let calls: [GFTokenizer.HistoricalToolCall]
        // Shares the failed turn's existing buffer, bounded by maxContext.
        let suffix: [Int32]
    }

    private let model: Model
    private let context: MetalContext
    private let tokenizer: GFTokenizer
    private let runner: RealForwardRunner
    private let scratch: RawCompletionScratch
    private let visionRuntime: VisionRuntime?
    private let visionRuntimeError: Error?
    private let visionResidency: VisionResidencyPolicy
    private let maxContext: Int

    /// Exact committed tokens, also retained when the KV needs reconstruction.
    private var kvTokenIDs: [Int32] = []
    /// Original feature buffers, with ranges in the committed token history.
    private var committedImageSpans: [MultimodalImageSpan] = []
    private var kvNeedsRebuild = false
    private var provisionalToolResultPrefix: ProvisionalToolResultPrefix?
    private var pending: (parts: [MultimodalContinuationPart], images: [URL])?
    private var closed = false
    /// One generation at a time. `RawCompletionScratch` documents that its
    /// buffers and sampler belong to a single run and that the guard is the
    /// caller's job; the actor alone does not provide it, because `generate`
    /// suspends for the whole decode.
    private var generating = false
    /// Resumed by `finishGeneration()`; see `waitForGeneration()`.
    private var generationWaiters: [CheckedContinuation<Void, Never>] = []
    /// Barrier for `reset()`, set before it awaits the in-flight decode so a
    /// `generate()` racing the reset cannot start and have its KV wiped.
    private var resetting = false
    /// Set only when the committed record cannot be safely reconstructed.
    private var lineageBroken = false
    /// Tokens the model emitted that never entered the KV, which happens when
    /// a run stops on max tokens or is cancelled. The next turn must replay
    /// them, exactly as the server's prefix cache does.
    private var uncommittedBoundary: [Int32] = []
    private var boundaryNeedsReplay = false
    private var toolState: ToolState?
    private var checkpointOpening: (
        id: UUID, turn: EncodedTurn, state: ToolState, needsFormatCorrection: Bool)?
    private var assessedResultBridge: (result: ConversationToolResult,
        calls: [GFTokenizer.HistoricalToolCall], tokens: [Int32])?
    private var largestCheckpointResultTokens = 0
    private var largestCheckpointGenerationTokens = 0

    public init(model: Model,
                context: MetalContext,
                tokenizer: GFTokenizer,
                runner: RealForwardRunner,
                scratch: RawCompletionScratch,
                visionRuntime: VisionRuntime? = nil,
                visionRuntimeError: Error? = nil,
                visionResidency: VisionResidencyPolicy = .defaultPolicy,
                maxContext: Int) {
        self.model = model
        self.context = context
        self.tokenizer = tokenizer
        self.runner = runner
        self.scratch = scratch
        self.visionRuntime = visionRuntime
        self.visionRuntimeError = visionRuntimeError
        self.visionResidency = visionResidency
        self.maxContext = maxContext
    }

    public var kvTokenCount: Int { kvTokenIDs.count }
    public var hasStagedTurn: Bool { pending != nil }
    public var isClosed: Bool { closed }
    public var isUsable: Bool { !closed && !lineageBroken }

    /// Marks this conversation unusable without touching the shared runner or
    /// vision runtime. The session calls it when handing out a replacement, so a
    /// stale reference cannot reset or resume onto the live conversation's KV.
    public func invalidate() async {
        closed = true
        pending = nil
        toolState = nil
        // Wait for any run still decoding: the session resets the shared runner
        // straight after this, and resetting under a live decode either aborts
        // that turn mid-stream or lets two turns drive one runner at once.
        await waitForGeneration()
        checkpointOpening = nil
        assessedResultBridge = nil
        committedImageSpans.removeAll()
        kvNeedsRebuild = false
        provisionalToolResultPrefix = nil
    }

    /// Suspends until no turn is decoding.
    ///
    /// This was a `while generating { await Task.yield() }` spin, which hot-looped
    /// the cooperative executor for the whole remaining decode — minutes at
    /// single-digit tok/s — and re-entered the actor it was waiting on at every
    /// iteration. Waiters are now resumed by the run itself.
    private func waitForGeneration() async {
        guard generating else { return }
        await withCheckedContinuation { continuation in
            generationWaiters.append(continuation)
        }
    }

    private func finishGeneration() {
        generating = false
        let waiters = generationWaiters
        generationWaiters = []
        for waiter in waiters { waiter.resume() }
    }

    private func discardProvisionalToolResultPrefix() {
        guard provisionalToolResultPrefix != nil else { return }
        provisionalToolResultPrefix = nil
        runner.reset()
        kvNeedsRebuild = !kvTokenIDs.isEmpty
    }

    /// Whether a turn is decoding right now.
    public var isGenerating: Bool { generating }

    /// Inspect, then explicitly interrupt a settled tool handoff. This is not
    /// ordinary result continuation: no model final answer is manufactured.
    /// The old token/feature record is untouched until every count and image
    /// has validated. Replacement prefill uses the same runner on resume.
    public func checkpoint(
        id: UUID, pendingCall: GFTokenizer.HistoricalToolCall,
        result: ConversationToolResult, record: String, imageProvenance: [String],
        generationAllowance: Int, finalAnswerAllowance: Int, permitsScreenshot: Bool,
        force: Bool, performanceRequested: Bool, commit: Bool,
        checkCancellation: @escaping @Sendable () throws -> Void = {}
    ) throws -> ConversationCheckpointReceipt {
        guard !closed, !lineageBroken else { throw MultimodalConversationError.lineageBroken }
        guard !generating, !resetting, pending == nil, checkpointOpening == nil,
              provisionalToolResultPrefix == nil else { throw MultimodalConversationError.busy }
        guard let state = toolState, state.awaitingResults,
              let assistant = state.messages.last,
              assistant.toolCalls == [pendingCall],
              result.callID == pendingCall.id, result.name == pendingCall.name,
              generationAllowance > 0, finalAnswerAllowance > 0,
              generationAllowance <= max(8_192, maxContext),
              finalAnswerAllowance <= max(2_048, maxContext) else {
            throw MultimodalConversationError.invalidToolContinuation
        }
        try checkCancellation()
        guard !record.contains(MultimodalPromptRenderer.placeholder),
              !result.content.contains(MultimodalPromptRenderer.placeholder),
              !imageProvenance.contains(where: { $0.contains(MultimodalPromptRenderer.placeholder) }) else {
            throw MultimodalPromptRendererError.reservedImageMarker
        }
        let toolMessage = GFTokenizer.Message(role: .tool, content: result.content,
            toolCallID: result.callID, name: result.name, toolImageCount: result.images.count)
        let continuationAssistant = Self.toolCallBoundaryMessage(assistant)
        let bridge: [Int32]
        if let assessed = assessedResultBridge, assessed.result == result,
           assessed.calls == assistant.toolCalls {
            bridge = assessed.tokens
        } else {
            bridge = try tokenizer.encodeToolResultContinuation(
                cachedMessages: Array(state.messages.dropLast()),
                assistant: continuationAssistant,
                incomingMessages: Array(state.messages.dropLast())
                    + [continuationAssistant, toolMessage],
                tools: state.tools)
        }
        guard bridge.filter({ $0 == MultimodalPromptRenderer.imageTokenID }).count == result.images.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        if !result.images.isEmpty, visionRuntime == nil {
            throw MultimodalConversationError.imageUnavailable(reason: visionRuntimeError.map(String.init(describing:)))
        }
        let plans = try result.images.map { url in
            try Gemma4ImagePreprocessor(device: context.device, config: visionRuntime!.config).plan(fileURL: url)
        }
        let oldFeatures = committedImageSpans.map(\.features)
        let allRowCounts = oldFeatures.map(\.tokenCount) + plans.map(\.geometry.softTokenCount)
        guard imageProvenance.count == allRowCounts.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        // Retain only images from this pending result. Gemma has already seen
        // older screenshots, while their compact text evidence survives in the
        // checkpoint. This removes historical image tokens and feature buffers.
        let retainedProvenance = Array(imageProvenance.suffix(plans.count))
        let retainedRowCounts = plans.map(\.geometry.softTokenCount)
        let currentImageExpansion = plans.reduce(0) { $0 + $1.geometry.softTokenCount + 1 }
        let existingCount = kvTokenIDs.count + (boundaryNeedsReplay ? uncommittedBoundary.count : 0)
            + bridge.count + currentImageExpansion
        // Forecast only. No generation cap is changed. A future packet uses
        // twice the largest measured text bridge, plus one maximum screenshot.
        let resultTextCount = bridge.count - result.images.count
        largestCheckpointResultTokens = max(largestCheckpointResultTokens, resultTextCount)
        let resultReserve = max(min(1_024, max(1, maxContext / 64)), largestCheckpointResultTokens * 2)
            + (permitsScreenshot ? VisionImageTokenBudget.maximumTokensPerImage : 0)
        // Scale forecast floors for smaller selected contexts without changing
        // the 64K policy. These allowances never alter generation limits.
        let generationReserve = max(min(generationAllowance, max(1, maxContext / 8)),
                                    largestCheckpointGenerationTokens * 2)
        let finalReserve = min(finalAnswerAllowance, max(1, maxContext / 32))
        let reserve = generationReserve + resultReserve + finalReserve
        let capacityNeeded = existingCount >= maxContext - reserve
        var replacementCount: Int?
        var performanceMinimumSavingsTokens: Int?
        var performanceCandidateAccepted = false
        var preparedReplacement: (messages: [GFTokenizer.Message], template: [Int32])?
        if commit || performanceRequested {
            var openingText = ""
            if !retainedProvenance.isEmpty {
                openingText = "Screenshot evidence from the last completed action follows. It is not an executable target.\n"
                for (index, source) in retainedProvenance.enumerated() {
                    openingText += "\nImage \(index + 1). \(source)\n"
                        + MultimodalPromptRenderer.placeholder + "\n"
                }
                openingText += "\nCurrent checkpoint follows.\n\n"
            }
            openingText += record
            var messages = state.messages.filter { $0.role == .system || $0.role == .developer }
            messages.append(GFTokenizer.Message(role: .user, content: openingText))
            let template = try tokenizer.encodeToolChat(messages: messages, tools: state.tools)
            guard template.filter({ $0 == MultimodalPromptRenderer.imageTokenID }).count == retainedRowCounts.count else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            let count = template.count + retainedRowCounts.reduce(0) { $0 + $1 + 1 }
            preparedReplacement = (messages, template)
            replacementCount = count
            if performanceRequested {
                let minimumSavings = max(4_096, existingCount / 5)
                performanceMinimumSavingsTokens = minimumSavings
                performanceCandidateAccepted = !record.isEmpty
                    && existingCount - count >= minimumSavings
                    && count < maxContext - reserve
            }
        }
        let needed = force || capacityNeeded || performanceCandidateAccepted
        if commit {
            guard needed, !record.isEmpty, let count = replacementCount,
                  count < existingCount, count < maxContext - reserve,
                  let preparedReplacement else {
                throw MultimodalConversationError.contextExhausted(
                    prompt: (replacementCount ?? existingCount) + reserve,
                    maxContext: maxContext)
            }
            var features = oldFeatures
            features.removeAll(keepingCapacity: true)
            for plan in plans {
                try checkCancellation()
                let encoded = try visionRuntime!.encodeImage(
                    plan: plan, languageModel: model, residencyPolicy: visionResidency,
                    checkCancellation: checkCancellation)
                guard encoded.tokenCount == plan.geometry.softTokenCount else {
                    throw MultimodalPromptRendererError.placeholderMismatch
                }
                features.append(encoded)
            }
            let turn: EncodedTurn
            if features.isEmpty {
                turn = EncodedTurn(
                    effectiveTokenIDs: preparedReplacement.template,
                    prefillInput: nil)
            } else {
                let input = try MultimodalPromptRenderer.expandingImageTokens(
                    preparedReplacement.template, features: features)
                turn = EncodedTurn(
                    effectiveTokenIDs: input.effectiveTokenIDs,
                    prefillInput: input)
            }
            guard turn.effectiveTokenIDs.count == count else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            try checkCancellation()
            // No await after the final cancellation check or before the new
            // record is installed. Old GPU work is settled by the idle guard.
            runner.reset()
            kvTokenIDs.removeAll()
            committedImageSpans.removeAll()
            uncommittedBoundary.removeAll()
            boundaryNeedsReplay = false
            kvNeedsRebuild = false
            toolState = nil
            assessedResultBridge = nil
            checkpointOpening = (id, turn,
                ToolState(messages: preparedReplacement.messages, tools: state.tools,
                          awaitingResults: false), false)
        } else {
            // Reuse the exact suffix on ordinary continuation. Capacity assess
            // does not construct or tokenize the growing replacement ledger.
            assessedResultBridge = (result, assistant.toolCalls, bridge)
        }
        return ConversationCheckpointReceipt(needed: needed, existingPromptTokens: existingCount,
            replacementPromptTokens: replacementCount, reserveTokens: reserve,
            resultAllowanceTokens: resultReserve, retainedImageCount: retainedRowCounts.count,
            retainedImageRows: retainedRowCounts.reduce(0, +),
            retainedFeatureBytes: plans.reduce(0) {
                $0 + $1.geometry.softTokenCount * model.config.hiddenSize * MemoryLayout<Float16>.stride
            },
            performanceMinimumSavingsTokens: performanceMinimumSavingsTokens)
    }

    public func resumeCheckpoint(
        id: UUID, config: GenerationConfig, prefillConfig: PrefillRuntimeConfig,
        checkCancellation: @escaping @Sendable () throws -> Void,
        shouldStop: (@Sendable () -> Bool)?, captureToolFailureEvidence: Bool,
        maximumConsecutiveInvisibleTokens: Int?, captureThoughtPreview: Bool,
        detectThoughtRepetition: Bool = false,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)?,
        onStructuredProgress: (@Sendable (StructuredAssistantProgress) -> Void)?
    ) async throws -> StructuredConversationTurnResult {
        guard !closed, !lineageBroken else { throw MultimodalConversationError.lineageBroken }
        guard !generating, !resetting else { throw MultimodalConversationError.busy }
        guard var opening = checkpointOpening, opening.id == id, kvTokenIDs.isEmpty else {
            throw MultimodalConversationError.invalidToolContinuation
        }
        try checkCancellation()
        if opening.needsFormatCorrection {
            let messages = try Self.correctingCheckpointToolFormat(opening.state.messages)
            let template = try tokenizer.encodeToolChat(messages: messages, tools: opening.state.tools)
            let features = opening.turn.prefillInput?.imageSpans.map(\.features) ?? []
            let input = features.isEmpty ? nil
                : try MultimodalPromptRenderer.expandingImageTokens(template, features: features)
            opening.turn = EncodedTurn(
                effectiveTokenIDs: input?.effectiveTokenIDs ?? template, prefillInput: input)
            opening.state.messages = messages
            opening.needsFormatCorrection = false
            checkpointOpening = opening
        }
        generating = true
        defer { finishGeneration() }
        let completion: StructuredConversationTurnResult
        do {
            completion = try await completeEncodedTurn(opening.turn, config: config,
            prefillConfig: prefillConfig, checkCancellation: checkCancellation,
            shouldStop: shouldStop, allowedTools: Set(opening.state.tools.map(\.name)),
            acceptsUnknownToolNames: true, captureToolFailureEvidence: captureToolFailureEvidence,
            maximumConsecutiveInvisibleTokens: maximumConsecutiveInvisibleTokens,
            captureThoughtPreview: captureThoughtPreview, detectThoughtRepetition: detectThoughtRepetition,
            onProgress: onProgress,
            onStructuredProgress: onStructuredProgress)
        } catch {
            if let failure = error as? StructuredToolFailure,
               failure.canRegenerateToolResult {
                checkpointOpening?.needsFormatCorrection = true
            }
            throw error
        }
        opening.state.messages.append(Self.assistantMessage(for: completion))
        opening.state.awaitingResults = !completion.toolCalls.isEmpty
        toolState = opening.state
        checkpointOpening = nil
        return completion
    }

    static func correctingCheckpointToolFormat(
        _ messages: [GFTokenizer.Message]
    ) throws -> [GFTokenizer.Message] {
        guard let last = messages.last, last.role == .user else {
            throw MultimodalConversationError.invalidToolContinuation
        }
        let feedback = """


        Host format correction: Your previous tool request was malformed. No action from that response was sent. Use only the latest permitted choices. Do not repeat completed actions. Gemma string values must use <|"|> delimiters, for example action:<|"|>observe<|"|>. Close every object and finish the tool call.
        """
        let content = last.content ?? ""
        guard !content.hasSuffix(feedback) else { return messages }
        var corrected = messages
        corrected[corrected.count - 1] = GFTokenizer.Message(
            role: last.role, content: content + feedback, toolCalls: last.toolCalls,
            toolCallID: last.toolCallID, name: last.name, toolImageCount: last.toolImageCount)
        return corrected
    }

    /// Stages the next user turn without prefilling it.
    public func append(parts: [MultimodalContinuationPart], images: [URL] = []) throws {
        guard !closed else { throw MultimodalConversationError.closed }
        // A turn staged while another is decoding used to be accepted and then
        // dropped: `generate()` clears `pending` when it commits, so the newly
        // staged turn vanished and the next `generate()` reported an empty turn
        // with the user's message gone.
        guard !generating else { throw MultimodalConversationError.busy }
        discardProvisionalToolResultPrefix()
        let imageCount = parts.filter {
            if case .image = $0 { return true } else { return false }
        }.count
        guard !parts.isEmpty, imageCount == images.count else {
            throw MultimodalConversationError.emptyTurn
        }
        pending = (parts, images)
    }

    /// Drops a staged turn that has not been generated yet.
    ///
    /// It cannot undo a generated turn. The KV cache has no rewind — the runner
    /// only validates that a continuation resumes at the current position — so
    /// once a turn is prefilled it is part of the lineage. Use `reset()` to
    /// abandon it, at the cost of re-prefilling the conversation.
    public func clear() {
        pending = nil
        if !generating { discardProvisionalToolResultPrefix() }
    }

    /// Drops the KV and starts over. The model stays loaded.
    ///
    /// Waits for a decode in flight. The actor accepts calls while `generate()`
    /// is suspended mid-run, so resetting here used to wipe the KV under a live
    /// turn: it then attended against an empty cache and either produced garbage
    /// or died on the prefill cursor, leaving `kvTokenIDs` describing a KV that
    /// no longer existed.
    public func reset() async {
        guard !closed else { return }
        // The barrier mirrors `close()` and `invalidate()`, which set theirs
        // before awaiting. Without one, a `generate()` enqueued while this
        // waiter was suspended could win the actor, start decoding, and have
        // its KV wiped underneath it — leaving `kvTokenIDs` restored from the
        // run's own token array while the cursor sat at zero, and
        // `lineageBroken` false, so `isUsable` kept saying yes forever. It
        // also stops back-to-back turns from a higher-priority task starving
        // the reset: the loop alone re-raced each freshly enqueued turn.
        resetting = true
        defer { resetting = false }
        while generating {
            await waitForGeneration()
        }
        guard !closed else { return }
        runner.reset()
        kvTokenIDs.removeAll(keepingCapacity: true)
        committedImageSpans.removeAll()
        kvNeedsRebuild = false
        provisionalToolResultPrefix = nil
        pending = nil
        lineageBroken = false
        uncommittedBoundary = []
        boundaryNeedsReplay = false
        toolState = nil
        checkpointOpening = nil
        assessedResultBridge = nil
        largestCheckpointResultTokens = 0
        largestCheckpointGenerationTokens = 0
    }

    /// Ends this conversation and clears the KV it was using. It deliberately
    /// does not touch the session-owned vision runtime: those mappings are
    /// shared, and releasing them here would corrupt another conversation's
    /// state. `TurboFieldfareModelSession.close()` owns that.
    public func close() async {
        guard !closed else { return }
        closed = true
        // Same hazard as `reset()`: the runner is shared and the decode is still
        // using it.
        await waitForGeneration()
        runner.reset()
        kvTokenIDs.removeAll(keepingCapacity: false)
        committedImageSpans.removeAll()
        kvNeedsRebuild = false
        provisionalToolResultPrefix = nil
        pending = nil
        toolState = nil
        checkpointOpening = nil
        assessedResultBridge = nil
    }

    /// Appends one user turn and generates the reply, prefilling only the new
    /// tokens. `images` are file URLs in the order they appear among `parts`.
    /// Appends a turn, generates the reply, and commits it.
    public func send(
        parts: [MultimodalContinuationPart],
        images: [URL] = [],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil
    ) async throws -> MultimodalTurnResult {
        try append(parts: parts, images: images)
        return try await generate(
            config: config, prefillConfig: prefillConfig,
            checkCancellation: checkCancellation, shouldStop: shouldStop,
            onProgress: onProgress)
    }

    /// Generates a reply to the staged turn. The turn joins the lineage as soon
    /// as it is prefilled, because the KV cannot rewind.
    /// `onProgress` receives the same events `runRawCompletion` reports, so a
    /// caller can stream tokens and prefill progress. Without it a turn was
    /// only observable once it had finished, which is unusable for a UI: the
    /// text was accumulated internally and returned in one piece.
    /// `shouldStop` ends the turn at a token boundary and **keeps** what it
    /// produced: the run returns normally with `.cancelled` as its stop reason,
    /// the partial reply is in the KV, and the next turn resumes on it. That is
    /// a different thing from cancelling the task, which throws out of the
    /// decode loop and makes this rewind the whole turn. A Stop button wants
    /// the first; a teardown wants the second.
    public func generate(
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil
    ) async throws -> MultimodalTurnResult {
        guard !closed else { throw MultimodalConversationError.closed }
        guard !lineageBroken else { throw MultimodalConversationError.lineageBroken }
        guard !generating, !resetting else { throw MultimodalConversationError.busy }
        discardProvisionalToolResultPrefix()
        guard let staged = pending else {
            throw MultimodalConversationError.emptyTurn
        }
        generating = true
        defer { finishGeneration() }
        let parts = staged.parts
        let images = staged.images
        let imageCount = parts.filter {
            if case .image = $0 { return true } else { return false }
        }.count
        if imageCount > 0, visionRuntime == nil {
            throw MultimodalConversationError.imageUnavailable(
                reason: visionRuntimeError.map(String.init(describing:)))
        }

        let turn = try await encodeTurn(parts: parts, images: images,
                                        checkCancellation: checkCancellation)
        let completion = try await completeEncodedTurn(
            turn,
            config: config,
            prefillConfig: prefillConfig,
            checkCancellation: checkCancellation,
            shouldStop: shouldStop,
            allowedTools: nil,
            onProgress: onProgress)
        pending = nil
        return completion.turn
    }

    /// Runs one tool-aware user turn on this conversation's existing model and
    /// KV. Tool mode starts only on an empty lineage so its definitions are
    /// present in the first rendered prompt. A later standard user turn renders
    /// the settled logical history again, which applies Gemma's bundled thought
    /// stripping while reusing the longest safe prefix of the existing KV.
    public func sendToolUser(
        parts: [MultimodalContinuationPart],
        images: [URL] = [],
        developerPrompt: String?,
        tools: [GFTokenizer.FunctionDefinition],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        acceptsUnknownToolNames: Bool = false,
        captureToolFailureEvidence: Bool = false,
        maximumConsecutiveInvisibleTokens: Int? = nil,
        captureThoughtPreview: Bool = false,
        detectThoughtRepetition: Bool = false,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil,
        onStructuredProgress: (@Sendable (StructuredAssistantProgress) -> Void)? = nil
    ) async throws -> StructuredConversationTurnResult {
        guard !closed else { throw MultimodalConversationError.closed }
        guard !lineageBroken else { throw MultimodalConversationError.lineageBroken }
        guard !generating, !resetting else { throw MultimodalConversationError.busy }
        discardProvisionalToolResultPrefix()
        guard pending == nil else { throw MultimodalConversationError.busy }
        let imageCount = parts.reduce(into: 0) {
            if case .image = $1 { $0 += 1 }
        }
        guard !parts.isEmpty, imageCount == images.count else {
            throw MultimodalConversationError.emptyTurn
        }
        if imageCount > 0, visionRuntime == nil {
            throw MultimodalConversationError.imageUnavailable(
                reason: visionRuntimeError.map(String.init(describing:)))
        }

        let userContent = try MultimodalConversationCanonicalization.userContent(parts)
        var state: ToolState
        let turn: EncodedTurn
        if var existing = toolState {
            guard existing.tools == tools, !existing.awaitingResults else {
                throw MultimodalConversationError.invalidToolContinuation
            }
            existing.messages.append(GFTokenizer.Message(role: .user, content: userContent))
            state = existing
            turn = try encodeCanonicalToolUserTurn(
                messages: existing.messages,
                tools: existing.tools,
                images: images,
                checkCancellation: checkCancellation,
                shouldStop: shouldStop)
        } else {
            guard kvTokenIDs.isEmpty else {
                throw MultimodalConversationError.toolModeRequiresNewConversation
            }
            var messages: [GFTokenizer.Message] = []
            if let developerPrompt {
                messages.append(GFTokenizer.Message(
                    role: .system, content: developerPrompt))
            }
            messages.append(GFTokenizer.Message(role: .user, content: userContent))
            state = ToolState(messages: messages, tools: tools, awaitingResults: false)
            turn = try await encodeOpeningToolTurn(
                parts: parts, images: images,
                messages: messages, tools: tools,
                checkCancellation: checkCancellation)
        }

        generating = true
        defer { finishGeneration() }
        let completion = try await completeEncodedTurn(
            turn,
            config: config,
            prefillConfig: prefillConfig,
            checkCancellation: checkCancellation,
            shouldStop: shouldStop,
            allowedTools: Set(tools.map(\.name)),
            acceptsUnknownToolNames: acceptsUnknownToolNames,
            captureToolFailureEvidence: captureToolFailureEvidence,
            maximumConsecutiveInvisibleTokens: maximumConsecutiveInvisibleTokens,
            captureThoughtPreview: captureThoughtPreview,
            detectThoughtRepetition: detectThoughtRepetition,
            onProgress: onProgress,
            onStructuredProgress: onStructuredProgress)
        state.messages.append(Self.assistantMessage(for: completion))
        state.awaitingResults = !completion.toolCalls.isEmpty
        toolState = state
        return completion
    }

    /// Appends host-produced tool results at the exact pending tool boundary,
    /// then asks the same loaded model to continue from the retained KV.
    public func sendToolResults(
        _ results: [ConversationToolResult],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @escaping @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        acceptsUnknownToolNames: Bool = false,
        captureToolFailureEvidence: Bool = false,
        maximumConsecutiveInvisibleTokens: Int? = nil,
        captureThoughtPreview: Bool = false,
        detectThoughtRepetition: Bool = false,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil,
        onStructuredProgress: (@Sendable (StructuredAssistantProgress) -> Void)? = nil
    ) async throws -> StructuredConversationTurnResult {
        guard !closed else { throw MultimodalConversationError.closed }
        guard !lineageBroken else { throw MultimodalConversationError.lineageBroken }
        guard !generating, !resetting else { throw MultimodalConversationError.busy }
        var handedToCompletion = false
        defer {
            if !handedToCompletion { discardProvisionalToolResultPrefix() }
        }
        guard var state = toolState,
              state.awaitingResults,
              let assistant = state.messages.last,
              !assistant.toolCalls.isEmpty,
              results.count == assistant.toolCalls.count,
              zip(results, assistant.toolCalls).allSatisfy({ result, call in
                  result.callID == call.id && result.name == call.name
              }) else {
            throw MultimodalConversationError.invalidToolContinuation
        }
        let toolMessages = results.map {
            GFTokenizer.Message(
                role: .tool,
                content: $0.content,
                toolCallID: $0.callID,
                name: $0.name,
                toolImageCount: $0.images.count)
        }
        let cached = Array(state.messages.dropLast())
        let continuationAssistant = Self.toolCallBoundaryMessage(assistant)
        let incoming = cached + [continuationAssistant] + toolMessages
        let bridge: [Int32]
        if let assessed = assessedResultBridge, results == [assessed.result],
           assessed.calls == assistant.toolCalls {
            bridge = assessed.tokens
        } else {
            bridge = try tokenizer.encodeToolResultContinuation(
                cachedMessages: cached, assistant: continuationAssistant,
                incomingMessages: incoming, tools: state.tools)
        }
        assessedResultBridge = nil
        let images = results.flatMap(\.images)
        guard !results.contains(where: { $0.content.contains(MultimodalPromptRenderer.placeholder) }) else {
            throw MultimodalPromptRendererError.reservedImageMarker
        }
        // The bridge must contain only this step's images, not prior image
        // markers retained in the template history.
        guard bridge.filter({ $0 == MultimodalPromptRenderer.imageTokenID }).count == images.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        generating = true
        defer { finishGeneration() }
        let turn: EncodedTurn
        if images.isEmpty {
            turn = EncodedTurn(effectiveTokenIDs: bridge, prefillInput: nil)
        } else {
            guard let visionRuntime else {
                throw MultimodalConversationError.imageUnavailable(
                    reason: visionRuntimeError.map(String.init(describing:)))
            }
            let checkImageCancellation: @Sendable () throws -> Void = {
                try checkCancellation()
                if shouldStop?() == true { throw CancellationError() }
            }
            try checkImageCancellation()
            let preprocessor = Gemma4ImagePreprocessor(device: context.device, config: visionRuntime.config)
            let plans = try images.map { try preprocessor.plan(fileURL: $0) }
            let imageTokens = plans.reduce(0) { $0 + $1.geometry.softTokenCount + 1 }
            let boundaryCount = boundaryNeedsReplay ? uncommittedBoundary.count : 0
            guard kvTokenIDs.count + boundaryCount + bridge.count + imageTokens + 1 <= maxContext else {
                throw MultimodalConversationError.contextExhausted(
                    prompt: kvTokenIDs.count + boundaryCount + bridge.count + imageTokens,
                    maxContext: maxContext)
            }
            var features: [VisionFeatures] = []
            for plan in plans {
                try checkImageCancellation()
                let encoded = try visionRuntime.encodeImage(
                    plan: plan, languageModel: model, residencyPolicy: visionResidency,
                    checkCancellation: checkImageCancellation)
                guard encoded.tokenCount == plan.geometry.softTokenCount else {
                    throw MultimodalPromptRendererError.placeholderMismatch
                }
                features.append(encoded)
            }
            try checkImageCancellation()
            let input = try MultimodalPromptRenderer.expandingImageTokens(bridge, features: features)
            turn = EncodedTurn(effectiveTokenIDs: input.effectiveTokenIDs, prefillInput: input)
        }
        handedToCompletion = true
        let completion = try await completeEncodedTurn(
            turn,
            config: config,
            prefillConfig: prefillConfig,
            checkCancellation: checkCancellation,
            shouldStop: shouldStop,
            allowedTools: Set(state.tools.map(\.name)),
            pendingToolCalls: assistant.toolCalls,
            acceptsUnknownToolNames: acceptsUnknownToolNames,
            captureToolFailureEvidence: captureToolFailureEvidence,
            maximumConsecutiveInvisibleTokens: maximumConsecutiveInvisibleTokens,
            captureThoughtPreview: captureThoughtPreview,
            detectThoughtRepetition: detectThoughtRepetition,
            onProgress: onProgress,
            onStructuredProgress: onStructuredProgress)
        state.messages.append(contentsOf: toolMessages)
        state.messages.append(Self.assistantMessage(for: completion))
        state.awaitingResults = !completion.toolCalls.isEmpty
        toolState = state
        return completion
    }

    static func assistantMessage(
        for completion: StructuredConversationTurnResult
    ) -> GFTokenizer.Message {
        GFTokenizer.Message(
            role: .assistant,
            content: completion.turn.text,
            toolCalls: completion.toolCalls.map {
                GFTokenizer.HistoricalToolCall(
                    id: $0.id, name: $0.name, arguments: $0.arguments)
            })
    }

    private static func toolCallBoundaryMessage(
        _ assistant: GFTokenizer.Message
    ) -> GFTokenizer.Message {
        GFTokenizer.Message(
            role: .assistant,
            content: nil,
            toolCalls: assistant.toolCalls)
    }

    private func completeEncodedTurn(
        _ turn: EncodedTurn,
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig,
        checkCancellation: @Sendable () throws -> Void,
        shouldStop: (@Sendable () -> Bool)?,
        allowedTools: Set<String>?,
        pendingToolCalls: [GFTokenizer.HistoricalToolCall]? = nil,
        acceptsUnknownToolNames: Bool = false,
        captureToolFailureEvidence: Bool = false,
        maximumConsecutiveInvisibleTokens: Int? = nil,
        captureThoughtPreview: Bool = false,
        detectThoughtRepetition: Bool = false,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)?,
        onStructuredProgress: (@Sendable (StructuredAssistantProgress) -> Void)? = nil
    ) async throws -> StructuredConversationTurnResult {
        let provisional = provisionalToolResultPrefix
        provisionalToolResultPrefix = nil
        var provisionalNeedsCleanup = provisional != nil
        defer {
            // Errors before decoding, or a failed retry without a new saved
            // prefix, must not leave uncommitted rows as the next KV lineage.
            if provisionalNeedsCleanup, provisionalToolResultPrefix == nil {
                runner.reset()
                kvNeedsRebuild = !kvTokenIDs.isEmpty
            }
        }
        // A run that stopped on max tokens or was cancelled left its final
        // token outside the KV. Replay it ahead of this turn, exactly as the
        // server's prefix cache does, or the model's context is missing a
        // token it already emitted.
        let boundary = boundaryNeedsReplay ? uncommittedBoundary : []
        let promptIDs = kvTokenIDs + boundary + turn.effectiveTokenIDs
        guard promptIDs.count + 1 <= maxContext else {
            throw MultimodalConversationError.contextExhausted(
                prompt: promptIDs.count, maxContext: maxContext)
        }

        var cached = kvNeedsRebuild ? 0 : kvTokenIDs.count
        if let provisional {
            do {
                try Task.checkCancellation()
                try checkCancellation()
                guard !closed, !resetting, shouldStop?() != true,
                      !kvNeedsRebuild, boundary.isEmpty, turn.prefillInput == nil,
                      provisional.committedTokenCount == kvTokenIDs.count,
                      let pendingToolCalls,
                      provisional.calls == pendingToolCalls,
                      runner.continuationPosition == kvTokenIDs.count + provisional.suffix.count else {
                    throw MultimodalConversationError.invalidToolContinuation
                }
                var shared = 0
                for (old, new) in zip(provisional.suffix, turn.effectiveTokenIDs) {
                    if old != new { break }
                    if shared.isMultiple(of: PrefillRuntimeConfig.maxChunkTokens) {
                        try Task.checkCancellation()
                        try checkCancellation()
                    }
                    shared += 1
                }
                // Refeed at least the final token to obtain the sampling seed.
                let sharedPosition = min(kvTokenIDs.count + shared, promptIDs.count - 1)
                guard sharedPosition > kvTokenIDs.count else {
                    throw MultimodalConversationError.invalidToolContinuation
                }
                // Uses the original high-water mark, including the first
                // rewind. Two short rewinds cannot bypass ring overwrite.
                try runner.rewind(to: sharedPosition)
                cached = sharedPosition
            } catch {
                runner.reset()
                kvNeedsRebuild = !kvTokenIDs.isEmpty
                provisionalNeedsCleanup = false
                cached = 0
                try Task.checkCancellation()
                try checkCancellation()
                if shouldStop?() == true { throw CancellationError() }
            }
        }

        let suffixInput = try turn.prefillInput?.prepending(boundary)
        let turnImageSpans = (suffixInput?.imageSpans ?? []).map {
            MultimodalImageSpan(
                tokenRange: ($0.tokenRange.lowerBound + kvTokenIDs.count)
                    ..< ($0.tokenRange.upperBound + kvTokenIDs.count),
                features: $0.features)
        }
        let multimodalInput: MultimodalPrefillInput?
        if kvNeedsRebuild, !committedImageSpans.isEmpty || !turnImageSpans.isEmpty {
            let spans = committedImageSpans + turnImageSpans
            var embeddingIDs = promptIDs
            for span in spans {
                // Match the original renderer: image rows use the retained
                // projected features, with zero IDs for the embedding input.
                embeddingIDs.replaceSubrange(
                    span.tokenRange, with: repeatElement(Int32(0), count: span.tokenRange.count))
            }
            multimodalInput = try MultimodalPrefillInput(
                effectiveTokenIDs: promptIDs,
                embeddingTokenIDs: embeddingIDs,
                imageSpans: spans)
        } else if kvNeedsRebuild {
            multimodalInput = nil
        } else {
            multimodalInput = suffixInput
        }

        // Same coercion as the other entry points: a turn carrying image spans
        // cannot run under a non-chunked mode, and refusing it after the images
        // are encoded helps nobody.
        var effectivePrefill = prefillConfig
        if multimodalInput != nil,
           let coerced = effectivePrefill.coercedForImagePrompt() {
            effectivePrefill = coerced
        }
        var generation = config
        generation.maxNewTokens = min(config.maxNewTokens, maxContext - promptIDs.count)
        var text = ""
        var calls: [ParsedToolCall] = []
        var structuredError: Error?
        var consecutiveInvisibleTokens = 0
        var generatedTokenCount = 0
        let decoder = allowedTools.map {
            StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: $0,
                startsInThoughtChannel: tokenizer.promptEndsInThoughtChannel(turn.effectiveTokenIDs),
                acceptsUnknownToolNames: acceptsUnknownToolNames,
                captureFailureEvidence: captureToolFailureEvidence,
                captureThoughtPreview: captureThoughtPreview,
                detectThoughtRepetition: detectThoughtRepetition)
        }
        // Marked only once the run has actually written to the KV. Setting it
        // before the attempt condemned the conversation for failures that never
        // touched the cache — a cancellation caught by the first
        // `checkCancellation()`, a rejected prefill config, a resume-position
        // mismatch — where the recorded lineage still matched the KV exactly and
        // the turn could simply have been retried.
        // The runner's own cursor, not a progress callback. Inferring it from
        // the first `.prefill` report missed a chunk that committed and then
        // threw — the KV had moved, nothing said so, and every later turn
        // resumed at a position the runner no longer had, failing forever while
        // `isUsable` still said yes.
        let positionBefore = runner.continuationPosition
        var kvAdvanced = false
        let result: RawDecodeResult
        var originalStopReason: String?
        do {
            result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            multimodalInput: multimodalInput,
            config: generation,
            context: context,
            scratch: scratch,
            prefillConfig: effectivePrefill,
            start: cached == 0 ? .reset : .resume(cachedPromptTokens: cached),
                // Cancellation mid-decode ends the turn at a token boundary
                // and returns the partial result; throwing here instead would
                // condemn the lineage for a KV that is perfectly resumable.
                shouldStop: {
                    if structuredError != nil { return true }
                    if shouldStop?() == true { return true }
                    do { try checkCancellation(); return false } catch { return true }
                }) { progress in
                    do {
                        switch progress {
                        case .token(let index, let tokenID, let delta):
                            generatedTokenCount = index + 1
                            if let decoder {
                                var visible = ""
                                let events: [StructuredAssistantEvent]
                                do {
                                    events = try decoder.consume(tokenID: tokenID, delta: delta)
                                } catch {
                                    onStructuredProgress?(decoder.progress)
                                    if error is RepeatedThoughtDetected {
                                        onProgress?(.token(index: index, id: tokenID, delta: ""))
                                    }
                                    throw error
                                }
                                onStructuredProgress?(decoder.progress)
                                for event in events {
                                    switch event {
                                    case .content(let value): visible += value
                                    case .toolCall(let call): calls.append(call)
                                    }
                                }
                                let madeProgress = visible.contains {
                                    !$0.isWhitespace
                                }
                                    || events.contains {
                                        if case .toolCall = $0 { return true }
                                        return false
                                    }
                                if madeProgress {
                                    consecutiveInvisibleTokens = 0
                                } else if let limit = maximumConsecutiveInvisibleTokens {
                                    consecutiveInvisibleTokens += 1
                                    if consecutiveInvisibleTokens >= limit {
                                        throw MultimodalConversationError
                                            .noObservableToolProgress(limit: limit)
                                    }
                                }
                                text += visible
                                onProgress?(.token(index: index,
                                                   id: tokenID,
                                                   delta: visible))
                            } else {
                                text += delta
                                onProgress?(progress)
                            }
                        case .tail(let tail):
                            if let decoder {
                                var visible = ""
                                for event in try decoder.consumeTail(tail) {
                                    if case .content(let value) = event { visible += value }
                                }
                                onStructuredProgress?(decoder.progress)
                                text += visible
                                onProgress?(.tail(visible))
                            } else {
                                text += tail
                                onProgress?(progress)
                            }
                        // The detokenizer flush at the stop boundary. Dropping
                        // it returned turn text missing its final characters
                        // while the KV kept those very tokens.
                        case .prefill:
                        // The first progress report is the proof the KV moved.
                            kvAdvanced = true
                            onProgress?(progress)
                        }
                    } catch {
                        if structuredError == nil { structuredError = error }
                    }
                }
            if captureToolFailureEvidence { originalStopReason = String(describing: result.reason) }
            if let structuredError { throw structuredError }
            try Self.validateStructuredCompletion(
                decoder, stopReason: result.reason, hasToolCalls: !calls.isEmpty)
        } catch let generationError {
            decoder?.refreshToolCallPreview()
            if let decoder { onStructuredProgress?(decoder.progress) }
            if runner.continuationPosition != positionBefore { kvAdvanced = true }
            var preservedPrompt = false
            if let pendingToolCalls, !pendingToolCalls.isEmpty,
               turn.prefillInput == nil, boundary.isEmpty,
               generationError as? GemmaToolCallParserError == .malformed,
               calls.isEmpty, !closed, !resetting,
               !Task.isCancelled, shouldStop?() != true {
                do {
                    try checkCancellation()
                    // Only a fully fed prompt is reusable. This guarded
                    // rewind drops generated rows, never the pending result.
                    try runner.rewind(to: promptIDs.count)
                    provisionalToolResultPrefix = ProvisionalToolResultPrefix(
                        committedTokenCount: kvTokenIDs.count,
                        calls: pendingToolCalls, suffix: turn.effectiveTokenIDs)
                    kvNeedsRebuild = false
                    preservedPrompt = true
                } catch {
                    // The existing rollback/reconstruction path remains the
                    // fallback when generation already exceeded ring slack.
                    preservedPrompt = false
                }
            }
            // The token record, image features and pending tool results are
            // still committed at the old boundary. If a long failed attempt
            // overwrote the sliding window, rebuild that prefix on the next
            // request. Prefilling recorded tokens does not execute tools.
            if kvAdvanced, !preservedPrompt {
                do {
                    try MultimodalConversationKVRecovery.restoreAfterFailure(
                        positionBefore: positionBefore,
                        generationError: generationError,
                        rewind: runner.rewind(to:),
                        reset: runner.reset)
                } catch {
                    runner.reset()
                    kvNeedsRebuild = !kvTokenIDs.isEmpty
                }
            }
            if let repetition = generationError as? RepeatedThoughtDetected {
                // Stop/teardown wins even when it races detection. A recovery
                // receipt is never a cancellation acknowledgement.
                try Task.checkCancellation()
                try checkCancellation()
                if shouldStop?() == true { throw CancellationError() }
                guard !closed, !resetting, !lineageBroken, calls.isEmpty else {
                    throw MultimodalConversationError.invalidToolContinuation
                }
                // The reusable prefix is the committed record, not a failed
                // format retry's provisional prompt. Retain every image span.
                provisionalToolResultPrefix = nil
                if !kvNeedsRebuild, runner.continuationPosition != kvTokenIDs.count {
                    do { try runner.rewind(to: kvTokenIDs.count) }
                    catch { runner.reset(); kvNeedsRebuild = !kvTokenIDs.isEmpty }
                }
                guard kvNeedsRebuild || runner.continuationPosition == kvTokenIDs.count else {
                    throw MultimodalConversationError.lineageRecoveryFailed(
                        reason: "The repeated generation did not restore its committed prefix.")
                }
                provisionalNeedsCleanup = false
                try Task.checkCancellation()
                try checkCancellation()
                if shouldStop?() == true { throw CancellationError() }
                let pendingCall = pendingToolCalls?.count == 1 ? pendingToolCalls?.first : nil
                throw ThoughtRepetitionRecovery(
                    canRetryToolResult: pendingCall != nil,
                    pendingCallID: pendingCall?.id, pendingToolName: pendingCall?.name,
                    restoredTokenCount: kvTokenIDs.count, requiresRebuild: kvNeedsRebuild,
                    generatedTokens: generatedTokenCount,
                    thinkingTokens: decoder?.progress.thinkingTokens ?? 0,
                    blockTokens: repetition.blockTokens,
                    repetitions: ThoughtRepetitionDetector.repetitions)
            }
            if let parserError = generationError as? GemmaToolCallParserError {
                var evidence = decoder?.failureEvidence
                evidence?.originalStopReason = originalStopReason
                throw StructuredToolFailure(
                    underlying: parserError,
                    canRegenerateToolResult: parserError == .malformed && calls.isEmpty
                        && !Task.isCancelled && shouldStop?() != true,
                    evidence: evidence)
            }
            throw generationError
        }

        provisionalNeedsCleanup = false
        // The KV now holds exactly what the run reported, including the tokens
        // the model generated; keeping the model's own tokens rather than
        // re-tokenising its text is what makes the next turn resumable.
        kvTokenIDs = result.kvBackedTokenIDs
        kvNeedsRebuild = false
        // A stop-string match discards the text of the tokens that formed it,
        // but all except the final one were already committed to the KV. Left
        // there, every later turn resumes on a context holding assistant text
        // the caller never saw. Drop them from the record and rewind the
        // runner so KV and transcript agree again; a token that showed a
        // visible prefix before the match began stays, so at most a few
        // characters of one boundary token remain hidden.
        if result.reason == .stopString, result.withheldTrailingKVTokens > 0 {
            do {
                kvTokenIDs = try MultimodalConversationKVRecovery.trimHiddenStopTokens(
                    result.kvBackedTokenIDs,
                    withheld: result.withheldTrailingKVTokens,
                    rewind: runner.rewind(to:),
                    reset: runner.reset)
            } catch {
                let target = result.kvBackedTokenIDs.count - result.withheldTrailingKVTokens
                guard target >= promptIDs.count else {
                    lineageBroken = true
                    throw error
                }
                kvTokenIDs = Array(result.kvBackedTokenIDs.prefix(target))
                runner.reset()
                kvNeedsRebuild = !kvTokenIDs.isEmpty
            }
        }
        committedImageSpans.append(contentsOf: turnImageSpans)
        uncommittedBoundary = result.uncommittedBoundaryTokenIDs
        boundaryNeedsReplay = result.reason == .maxTokens || result.reason == .cancelled
        let turnResult = MultimodalTurnResult(
            text: text, promptTokens: promptIDs.count, cachedTokens: cached,
            // The runner's own count, not a reconstruction from KV lengths.
            // Reconstructing it made the figure depend on the stop reason —
            // end-of-turn, EOS, tool-call and stop-string stops each reported
            // one fewer — so the same run was counted differently here and in
            // `ServerInference`, which has always reported this value.
            computedPrefillTokens: result.computedPrefillTokens,
            kvTokens: kvTokenIDs.count,
            completionTokens: result.newTokens,
            reason: result.reason,
            prefillSeconds: result.prefillSeconds,
            decodeSeconds: result.decodeSeconds)
        largestCheckpointGenerationTokens = max(largestCheckpointGenerationTokens, result.newTokens)
        return StructuredConversationTurnResult(
            turn: turnResult,
            toolCalls: calls)
    }

    /// A user stop can land inside a tool span that is valid so far but not
    /// complete. Treating that intentional boundary as malformed replaced the
    /// Stop result with a parser error. Natural model endings still require a
    /// complete structure and matching tool-call stop.
    static func validateStructuredCompletion(
        _ decoder: StructuredAssistantDecoder?,
        stopReason: StopReason,
        hasToolCalls: Bool
    ) throws {
        guard let decoder, stopReason != .cancelled else { return }
        try decoder.finish()
        guard (stopReason == .toolCalls) == hasToolCalls else {
            decoder.captureFailure(phase: "stop_call_mismatch")
            throw GemmaToolCallParserError.malformed
        }
    }

    private struct EncodedTurn {
        let effectiveTokenIDs: [Int32]
        let prefillInput: MultimodalPrefillInput?
    }

    private func encodeCanonicalToolUserTurn(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        images: [URL],
        checkCancellation: @Sendable () throws -> Void,
        shouldStop: (@Sendable () -> Bool)?
    ) throws -> EncodedTurn {
        try checkCancellation()
        if shouldStop?() == true { throw CancellationError() }

        let template = try tokenizer.encodeToolChat(messages: messages, tools: tools)
        let oldFeatures = committedImageSpans.map(\.features)
        let placeholderCount = template.reduce(into: 0) {
            if $1 == MultimodalPromptRenderer.imageTokenID { $0 += 1 }
        }

        var plans: [VisionImagePlan] = []
        if !images.isEmpty {
            guard let visionRuntime else {
                throw MultimodalConversationError.imageUnavailable(
                    reason: visionRuntimeError.map(String.init(describing:)))
            }
            let preprocessor = Gemma4ImagePreprocessor(
                device: context.device, config: visionRuntime.config)
            plans = try images.map { try preprocessor.plan(fileURL: $0) }
        }
        guard placeholderCount == oldFeatures.count + plans.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        let historicalImageTokens = oldFeatures.reduce(0) {
            $0 + $1.tokenCount + 1
        }
        let incomingImageTokens = plans.reduce(0) {
            $0 + $1.geometry.softTokenCount + 1
        }
        let expandedCount = template.count + historicalImageTokens + incomingImageTokens
        guard expandedCount + 1 <= maxContext else {
            throw MultimodalConversationError.contextExhausted(
                prompt: expandedCount, maxContext: maxContext)
        }

        var features = oldFeatures
        if let visionRuntime {
            for plan in plans {
                try checkCancellation()
                if shouldStop?() == true { throw CancellationError() }
                let encoded = try visionRuntime.encodeImage(
                    plan: plan,
                    languageModel: model,
                    residencyPolicy: visionResidency,
                    checkCancellation: checkCancellation)
                guard encoded.tokenCount == plan.geometry.softTokenCount else {
                    throw MultimodalPromptRendererError.placeholderMismatch
                }
                features.append(encoded)
            }
        }
        try checkCancellation()
        if shouldStop?() == true { throw CancellationError() }

        let fullInput = features.isEmpty
            ? nil
            : try MultimodalPromptRenderer.expandingImageTokens(template, features: features)
        let canonicalIDs = fullInput?.effectiveTokenIDs ?? template
        let imageRanges = fullInput?.imageSpans.map(\.tokenRange) ?? []
        var cached = 0
        if !kvNeedsRebuild, runner.continuationPosition == kvTokenIDs.count {
            cached = MultimodalConversationCanonicalization.longestReusablePrefix(
                cached: kvTokenIDs,
                canonical: canonicalIDs,
                imageTokenRanges: imageRanges)
        }

        func suffix(at cachedCount: Int) throws -> EncodedTurn {
            let remainingImages = fullInput?.imageSpans.contains {
                $0.tokenRange.upperBound > cachedCount
            } == true
            let input = remainingImages
                ? try fullInput?.suffix(dropping: cachedCount)
                : nil
            return EncodedTurn(
                effectiveTokenIDs: Array(canonicalIDs.dropFirst(cachedCount)),
                prefillInput: input)
        }

        var turn = try suffix(at: cached)
        if cached == 0 {
            runner.reset()
        } else if cached < kvTokenIDs.count {
            do {
                try runner.rewind(to: cached)
            } catch {
                runner.reset()
                cached = 0
                turn = try suffix(at: 0)
            }
        }

        kvTokenIDs = Array(canonicalIDs.prefix(cached))
        committedImageSpans = (fullInput?.imageSpans ?? []).filter {
            $0.tokenRange.upperBound <= cached
        }
        kvNeedsRebuild = false
        uncommittedBoundary.removeAll(keepingCapacity: true)
        boundaryNeedsReplay = false
        return turn
    }

    private func encodeTurn(
        parts: [MultimodalContinuationPart],
        images: [URL],
        checkCancellation: @Sendable () throws -> Void
    ) async throws -> EncodedTurn {
        guard !images.isEmpty, let visionRuntime else {
            var text = ""
            for part in parts { if case .text(let value) = part { text += value } }
            let ids = kvTokenIDs.isEmpty
                ? tokenizer.encode(
                    try tokenizer.applyChatTemplate(
                        [GFTokenizer.Message(role: .user, content: text)]),
                    addBOS: false)
                : tokenizer.encodeTextContinuation(userContent: text)
            return EncodedTurn(effectiveTokenIDs: ids, prefillInput: nil)
        }

        // Token counts come from geometry, so the turn's shape is known before
        // any image is encoded.
        let preprocessor = Gemma4ImagePreprocessor(
            device: context.device, config: visionRuntime.config)
        // Kept, not discarded. Planning for the token count and then letting
        // `encodeImage(at:)` re-plan internally opened, sniffed and parsed every
        // image twice per turn — and a file rewritten between the two opens gave
        // a count that no longer matched the span the tokenizer had laid out,
        // failing the turn with a placeholder mismatch that names nothing the
        // user did.
        let plans = try images.map { try preprocessor.plan(fileURL: $0) }
        let counts = plans.map(\.geometry.softTokenCount)
        let bridge = try tokenizer.encodeMultimodalUserContinuation(
            textAndImages: parts, imageTokenCounts: counts,
            // An empty KV means this is the conversation's first turn, which
            // needs the chat template's opening rather than a continuation's.
            openingConversation: kvTokenIDs.isEmpty)
        var spans: [MultimodalImageSpan] = []
        for (range, plan) in zip(bridge.imageTokenRanges, plans) {
            try checkCancellation()
            let features = try visionRuntime.encodeImage(
                plan: plan, languageModel: model,
                residencyPolicy: visionResidency,
                checkCancellation: checkCancellation)
            guard features.tokenCount == range.count else {
                throw MultimodalPromptRendererError.placeholderMismatch
            }
            spans.append(MultimodalImageSpan(tokenRange: range, features: features))
        }
        return EncodedTurn(
            effectiveTokenIDs: bridge.effectiveTokenIDs,
            prefillInput: try MultimodalPrefillInput(
                effectiveTokenIDs: bridge.effectiveTokenIDs,
                embeddingTokenIDs: bridge.embeddingTokenIDs,
                imageSpans: spans))
    }

    private func encodeOpeningToolTurn(
        parts: [MultimodalContinuationPart],
        images: [URL],
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        checkCancellation: @Sendable () throws -> Void
    ) async throws -> EncodedTurn {
        guard !images.isEmpty, let visionRuntime else {
            return EncodedTurn(
                effectiveTokenIDs: try tokenizer.encodeToolChat(
                    messages: messages, tools: tools),
                prefillInput: nil)
        }
        let preprocessor = Gemma4ImagePreprocessor(
            device: context.device, config: visionRuntime.config)
        let plans = try images.map { try preprocessor.plan(fileURL: $0) }
        let imageIDs = images.map { _ in UUID() }
        var imageIndex = 0
        let userContent: [MultimodalContentPart] = parts.map { part in
            switch part {
            case .text(let value):
                return .text(value)
            case .image:
                defer { imageIndex += 1 }
                return .image(id: imageIDs[imageIndex])
            }
        }
        var renderedMessages: [MultimodalMessage] = []
        if let systemInstructions = messages.first,
           systemInstructions.role == .system || systemInstructions.role == .developer {
            renderedMessages.append(MultimodalMessage(
                role: systemInstructions.role,
                content: [.text(systemInstructions.content ?? "")]))
        }
        renderedMessages.append(MultimodalMessage(
            role: .user, content: userContent))
        var features: [UUID: VisionFeatures] = [:]
        for (id, plan) in zip(imageIDs, plans) {
            try checkCancellation()
            features[id] = try visionRuntime.encodeImage(
                plan: plan, languageModel: model,
                residencyPolicy: visionResidency,
                checkCancellation: checkCancellation)
        }
        let input = try MultimodalPromptRenderer.render(
            messages: renderedMessages,
            featuresByID: features,
            tokenizer: tokenizer,
            tools: tools)
        return EncodedTurn(
            effectiveTokenIDs: input.effectiveTokenIDs,
            prefillInput: input)
    }
}


/// Family-resolved conversation facade. Gemma keeps its established actor and
/// APIs; Qwen exposes only the prepared-token boundary until Phase 14 supplies
/// a chat codec.
public actor MultimodalConversation {
    private enum Backing {
        case gemma(GemmaMultimodalConversation)
        case qwen(QwenConversationState, maxContext: Int)
    }

    private let backing: Backing
    private var qwenGenerating = false
    private var qwenResetting = false
    private var qwenClosed = false
    private var qwenWaiters: [CheckedContinuation<Void, Never>] = []

    public init(
        model: Model,
        context: MetalContext,
        tokenizer: GFTokenizer,
        runner: RealForwardRunner,
        scratch: RawCompletionScratch,
        visionRuntime: VisionRuntime? = nil,
        visionRuntimeError: Error? = nil,
        visionResidency: VisionResidencyPolicy = .defaultPolicy,
        maxContext: Int
    ) {
        // The session is the sole owner and serial admission boundary for both
        // objects. They lack Sendable annotations because their internal Metal
        // scratch is not safe for concurrent use; this is the same invariant
        // that guarded the original single Gemma conversation actor.
        nonisolated(unsafe) let sharedModel = model
        nonisolated(unsafe) let sharedVisionRuntime = visionRuntime
        backing = .gemma(GemmaMultimodalConversation(
            model: sharedModel,
            context: context,
            tokenizer: tokenizer,
            runner: runner,
            scratch: scratch,
            visionRuntime: sharedVisionRuntime,
            visionRuntimeError: visionRuntimeError,
            visionResidency: visionResidency,
            maxContext: maxContext))
    }

    public init(qwenState: QwenConversationState, maxContext: Int) {
        backing = .qwen(qwenState, maxContext: maxContext)
    }

    public var kvTokenCount: Int {
        get async {
            switch backing {
            case .gemma(let conversation): await conversation.kvTokenCount
            case .qwen(let state, _): await state.status().committed.retainedTokenIDs.count
            }
        }
    }

    public var hasStagedTurn: Bool {
        get async {
            switch backing {
            case .gemma(let conversation): await conversation.hasStagedTurn
            case .qwen: false
            }
        }
    }

    public var isClosed: Bool {
        get async {
            switch backing {
            case .gemma(let conversation): await conversation.isClosed
            case .qwen: qwenClosed
            }
        }
    }

    public var isUsable: Bool {
        get async {
            switch backing {
            case .gemma(let conversation): await conversation.isUsable
            case .qwen: !qwenClosed
            }
        }
    }

    public var isGenerating: Bool {
        get async {
            switch backing {
            case .gemma(let conversation): await conversation.isGenerating
            case .qwen: qwenGenerating
            }
        }
    }

    public func invalidate() async {
        switch backing {
        case .gemma(let conversation):
            await conversation.invalidate()
        case .qwen:
            qwenClosed = true
            await waitForQwenGeneration()
        }
    }

    public func checkpoint(
        id: UUID,
        pendingCall: GFTokenizer.HistoricalToolCall,
        result: ConversationToolResult,
        record: String,
        imageProvenance: [String],
        generationAllowance: Int,
        finalAnswerAllowance: Int,
        permitsScreenshot: Bool,
        force: Bool,
        performanceRequested: Bool,
        commit: Bool,
        checkCancellation: @escaping @Sendable () throws -> Void = {}
    ) async throws -> ConversationCheckpointReceipt {
        switch backing {
        case .gemma(let conversation):
            try await conversation.checkpoint(
                id: id, pendingCall: pendingCall, result: result, record: record,
                imageProvenance: imageProvenance,
                generationAllowance: generationAllowance,
                finalAnswerAllowance: finalAnswerAllowance,
                permitsScreenshot: permitsScreenshot, force: force,
                performanceRequested: performanceRequested, commit: commit,
                checkCancellation: checkCancellation)
        case .qwen:
            throw MultimodalConversationError.tokenCodecUnavailable
        }
    }

    public func resumeCheckpoint(
        id: UUID,
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig,
        checkCancellation: @escaping @Sendable () throws -> Void,
        shouldStop: (@Sendable () -> Bool)?,
        captureToolFailureEvidence: Bool,
        maximumConsecutiveInvisibleTokens: Int?,
        captureThoughtPreview: Bool,
        detectThoughtRepetition: Bool = false,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)?,
        onStructuredProgress: (@Sendable (StructuredAssistantProgress) -> Void)?
    ) async throws -> StructuredConversationTurnResult {
        switch backing {
        case .gemma(let conversation):
            try await conversation.resumeCheckpoint(
                id: id, config: config, prefillConfig: prefillConfig,
                checkCancellation: checkCancellation, shouldStop: shouldStop,
                captureToolFailureEvidence: captureToolFailureEvidence,
                maximumConsecutiveInvisibleTokens: maximumConsecutiveInvisibleTokens,
                captureThoughtPreview: captureThoughtPreview,
                detectThoughtRepetition: detectThoughtRepetition,
                onProgress: onProgress,
                onStructuredProgress: onStructuredProgress)
        case .qwen:
            throw MultimodalConversationError.tokenCodecUnavailable
        }
    }

    public func append(parts: [MultimodalContinuationPart], images: [URL] = []) async throws {
        switch backing {
        case .gemma(let conversation): try await conversation.append(parts: parts, images: images)
        case .qwen: throw MultimodalConversationError.tokenCodecUnavailable
        }
    }

    public func clear() async {
        if case .gemma(let conversation) = backing { await conversation.clear() }
    }

    public func reset() async throws {
        switch backing {
        case .gemma(let conversation):
            await conversation.reset()
        case .qwen(let state, _):
            guard !qwenClosed else { return }
            qwenResetting = true
            defer { qwenResetting = false }
            await waitForQwenGeneration()
            guard !qwenClosed else { return }
            try await state.reset()
        }
    }

    public func close() async {
        switch backing {
        case .gemma(let conversation): await conversation.close()
        case .qwen:
            qwenClosed = true
            await waitForQwenGeneration()
        }
    }

    public func send(
        parts: [MultimodalContinuationPart],
        images: [URL] = [],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil
    ) async throws -> MultimodalTurnResult {
        switch backing {
        case .gemma(let conversation):
            try await conversation.send(
                parts: parts, images: images, config: config,
                prefillConfig: prefillConfig,
                checkCancellation: checkCancellation,
                shouldStop: shouldStop, onProgress: onProgress)
        case .qwen:
            throw MultimodalConversationError.tokenCodecUnavailable
        }
    }

    public func generate(
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil
    ) async throws -> MultimodalTurnResult {
        switch backing {
        case .gemma(let conversation):
            try await conversation.generate(
                config: config, prefillConfig: prefillConfig,
                checkCancellation: checkCancellation,
                shouldStop: shouldStop, onProgress: onProgress)
        case .qwen:
            throw MultimodalConversationError.tokenCodecUnavailable
        }
    }

    public func sendToolUser(
        parts: [MultimodalContinuationPart],
        images: [URL] = [],
        developerPrompt: String?,
        tools: [GFTokenizer.FunctionDefinition],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        acceptsUnknownToolNames: Bool = false,
        captureToolFailureEvidence: Bool = false,
        maximumConsecutiveInvisibleTokens: Int? = nil,
        captureThoughtPreview: Bool = false,
        detectThoughtRepetition: Bool = false,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil,
        onStructuredProgress: (@Sendable (StructuredAssistantProgress) -> Void)? = nil
    ) async throws -> StructuredConversationTurnResult {
        switch backing {
        case .gemma(let conversation):
            try await conversation.sendToolUser(
                parts: parts, images: images, developerPrompt: developerPrompt,
                tools: tools, config: config, prefillConfig: prefillConfig,
                checkCancellation: checkCancellation, shouldStop: shouldStop,
                acceptsUnknownToolNames: acceptsUnknownToolNames,
                captureToolFailureEvidence: captureToolFailureEvidence,
                maximumConsecutiveInvisibleTokens: maximumConsecutiveInvisibleTokens,
                captureThoughtPreview: captureThoughtPreview,
                detectThoughtRepetition: detectThoughtRepetition,
                onProgress: onProgress,
                onStructuredProgress: onStructuredProgress)
        case .qwen:
            throw MultimodalConversationError.tokenCodecUnavailable
        }
    }

    public func sendToolResults(
        _ results: [ConversationToolResult],
        config: GenerationConfig,
        prefillConfig: PrefillRuntimeConfig = .defaultChunked,
        checkCancellation: @escaping @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        acceptsUnknownToolNames: Bool = false,
        captureToolFailureEvidence: Bool = false,
        maximumConsecutiveInvisibleTokens: Int? = nil,
        captureThoughtPreview: Bool = false,
        detectThoughtRepetition: Bool = false,
        onProgress: (@Sendable (RawDecodeProgress) -> Void)? = nil,
        onStructuredProgress: (@Sendable (StructuredAssistantProgress) -> Void)? = nil
    ) async throws -> StructuredConversationTurnResult {
        switch backing {
        case .gemma(let conversation):
            try await conversation.sendToolResults(
                results, config: config, prefillConfig: prefillConfig,
                checkCancellation: checkCancellation, shouldStop: shouldStop,
                acceptsUnknownToolNames: acceptsUnknownToolNames,
                captureToolFailureEvidence: captureToolFailureEvidence,
                maximumConsecutiveInvisibleTokens: maximumConsecutiveInvisibleTokens,
                captureThoughtPreview: captureThoughtPreview,
                detectThoughtRepetition: detectThoughtRepetition,
                onProgress: onProgress,
                onStructuredProgress: onStructuredProgress)
        case .qwen:
            throw MultimodalConversationError.tokenCodecUnavailable
        }
    }

    public func applyTokenizedTurn(
        _ turn: TokenizedConversationTurn,
        checkCancellation: @escaping @Sendable () throws -> Void = {},
        shouldStop: (@Sendable () -> Bool)? = nil,
        onProgress: (@Sendable (TokenizedConversationProgress) async -> Void)? = nil
    ) async throws -> TokenizedConversationResult {
        guard case .qwen(let state, let maxContext) = backing else {
            throw ConversationStateTransactionError.unsupportedFamily
        }
        guard !qwenClosed else { throw MultimodalConversationError.closed }
        guard !qwenGenerating, !qwenResetting else { throw MultimodalConversationError.busy }
        guard !turn.promptTokenIDs.isEmpty,
              turn.promptTokenIDs.count + turn.generatedTokenIDs.count <= maxContext,
              turn.hiddenSuffixTokenCount >= 0,
              turn.hiddenSuffixTokenCount <= turn.generatedTokenIDs.count else {
            throw ConversationStateTransactionError.invalidBoundary("invalid prepared token turn")
        }
        qwenGenerating = true
        defer { finishQwenGeneration() }
        let transaction = try await state.begin()
        do {
            try Task.checkCancellation()
            try checkCancellation()
            try await state.prefill(
                turn.promptTokenIDs,
                transaction: transaction,
                onProgress: { done, total in
                    await onProgress?(.prefill(done: done, total: total))
                })
            try Task.checkCancellation()
            try checkCancellation()
            if shouldStop?() == true { throw CancellationError() }

            var accepted: [Int32] = []
            var reason = TokenizedConversationStopReason.complete
            for tokenID in turn.generatedTokenIDs {
                try Task.checkCancellation()
                try checkCancellation()
                try await state.advance(tokenID, transaction: transaction)
                accepted.append(tokenID)
                await onProgress?(.accepted(index: accepted.count - 1, tokenID: tokenID))
                if shouldStop?() == true {
                    reason = .softStop
                    break
                }
            }
            if reason == .complete, turn.hiddenSuffixTokenCount > 0 {
                try await state.removeSuffix(
                    tokenCount: turn.hiddenSuffixTokenCount,
                    transaction: transaction)
                accepted.removeLast(turn.hiddenSuffixTokenCount)
            }
            let metrics = try await state.commit(transaction: transaction)
            return TokenizedConversationResult(
                reason: reason,
                metrics: metrics,
                acceptedGeneratedTokenIDs: accepted)
        } catch let operationError {
            do {
                try await state.rollback(transaction: transaction)
            } catch {
                throw MultimodalConversationError.lineageRecoveryFailed(
                    reason: "Prepared Qwen turn failed with \(operationError); rollback failed with \(error).")
            }
            throw operationError
        }
    }

    public func rebuildTokenCheckpoint(
        _ request: TokenizedCheckpointRequest,
        checkCancellation: @escaping @Sendable () throws -> Void = {},
        onProgress: (@Sendable (TokenizedConversationProgress) async -> Void)? = nil
    ) async throws -> TokenizedCheckpointResult {
        guard case .qwen(let state, _) = backing else {
            throw ConversationStateTransactionError.unsupportedFamily
        }
        guard !qwenClosed else { throw MultimodalConversationError.closed }
        guard !qwenGenerating, !qwenResetting else { throw MultimodalConversationError.busy }
        if request.reason == .sustainedSlowDecode {
            let metrics = await state.status().committed
            return TokenizedCheckpointResult(needed: false, committed: false, metrics: metrics)
        }
        qwenGenerating = true
        defer { finishQwenGeneration() }
        let transaction = try await state.begin()
        do {
            try Task.checkCancellation()
            try checkCancellation()
            try await state.rebuildCheckpoint(
                retaining: request.retainedTokenIDs,
                transaction: transaction,
                onProgress: { done, total in
                    await onProgress?(.checkpoint(done: done, total: total))
                })
            try Task.checkCancellation()
            try checkCancellation()
            let metrics = try await state.commit(transaction: transaction)
            return TokenizedCheckpointResult(needed: true, committed: true, metrics: metrics)
        } catch let operationError {
            do {
                try await state.rollback(transaction: transaction)
            } catch {
                throw MultimodalConversationError.lineageRecoveryFailed(
                    reason: "Qwen checkpoint failed with \(operationError); rollback failed with \(error).")
            }
            throw operationError
        }
    }

    public func conversationStateStatus() async throws -> ConversationStateStatus {
        guard case .qwen(let state, _) = backing else {
            throw ConversationStateTransactionError.unsupportedFamily
        }
        return await state.status()
    }

    static func assistantMessage(
        for completion: StructuredConversationTurnResult
    ) -> GFTokenizer.Message {
        GemmaMultimodalConversation.assistantMessage(for: completion)
    }

    static func validateStructuredCompletion(
        _ decoder: StructuredAssistantDecoder?,
        stopReason: StopReason,
        hasToolCalls: Bool
    ) throws {
        try GemmaMultimodalConversation.validateStructuredCompletion(
            decoder, stopReason: stopReason, hasToolCalls: hasToolCalls)
    }

    private func waitForQwenGeneration() async {
        guard qwenGenerating else { return }
        await withCheckedContinuation { continuation in
            qwenWaiters.append(continuation)
        }
    }

    private func finishQwenGeneration() {
        qwenGenerating = false
        let waiters = qwenWaiters
        qwenWaiters = []
        for waiter in waiters { waiter.resume() }
    }
}
