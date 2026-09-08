import CryptoKit
import Foundation
import os
import TurboFieldfare
import TurboFieldfareDecodeProtocol

public struct VisionCaptureAgentConfiguration: Equatable, Sendable {
    public let bundleIdentifier: String
    public let simulatorUDID: String
    public let modelDirectory: URL

    public init(
        bundleIdentifier: String,
        simulatorUDID: String,
        modelDirectory: URL
    ) {
        self.bundleIdentifier = bundleIdentifier
        self.simulatorUDID = simulatorUDID
        self.modelDirectory = modelDirectory.standardizedFileURL
    }

    public func validate() throws {
        let bundle = bundleIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        let allowed = CharacterSet(
            charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        guard !bundle.isEmpty, bundle.contains("."),
              bundle.unicodeScalars.allSatisfy(allowed.contains) else {
            throw VisionCaptureAgentError.invalidConfiguration(
                "Enter the exact application bundle identifier.")
        }
        guard UUID(uuidString: simulatorUDID.trimmingCharacters(
            in: .whitespacesAndNewlines)) != nil else {
            throw VisionCaptureAgentError.invalidConfiguration(
                "Enter the exact Simulator UDID.")
        }
    }

    fileprivate var targetKey: String {
        "\(bundleIdentifier)\u{1f}\(simulatorUDID)"
    }
}

public enum VisionCaptureAgentError: Error, Equatable, Sendable,
    CustomStringConvertible {
    case invalidConfiguration(String)
    case mcpUnavailable(String)
    case skillReadFailed(String)
    case malformedCall(String)
    case navigationUnavailable(String)
    case identityMismatch
    case returnedIdentityMismatch(fieldPath: String, refusalCode: String?)
    case launchOutcomeUnproven(String)
    case sessionIdentityMismatch
    case unsupportedVisualRequest
    case unsupportedSystemInteraction(String, outcome: VisionCaptureServerOutcome? = nil)
    case mcpRefused(String)
    case mcpOutcome(VisionCaptureServerOutcome)
    case noProgress(String)
    case incompleteAnswer

    public var description: String {
        switch self {
        case .invalidConfiguration(let message): message
        case .mcpUnavailable(let message):
            "VisionCapture is unavailable at 127.0.0.1:8766: \(message)"
        case .skillReadFailed(let message):
            "VisionCapture host instructions are unavailable: \(message)"
        case .malformedCall(let message):
            "Gemma returned an invalid navigation proposal: \(message)"
        case .navigationUnavailable(let message): message
        case .identityMismatch:
            "The configured application or Simulator differs from the locked target."
        case .returnedIdentityMismatch(let fieldPath, let refusalCode):
            "The MCP request was sent, but VisionCapture's returned identity failed validation at \(fieldPath). The requested operation's outcome cannot be established from that result. Agent Mode stopped and discarded retained execution evidence. No action was replayed."
                + (refusalCode.map { " Original refusal code: \($0)." } ?? "")
        case .launchOutcomeUnproven(let reason):
            "Launch returned SIMULATOR_UNRESPONSIVE with unknown submission and outcome. Its one read-only recovery did not establish the requested foreground app: \(reason) No launch or action was replayed."
        case .sessionIdentityMismatch:
            "VisionCapture returned an unpermitted session change."
        case .unsupportedVisualRequest:
            "Agent Mode supports read-only screenshots, but mutations remain accessibility-only. Computer Use, pointer actions, preview, recording, and visual bypass are unavailable."
        case .unsupportedSystemInteraction(let code, let outcome):
            outcome.map { $0.description + ". Agent Mode stopped. No action was replayed." }
                ?? "This first proof stopped because VisionCapture reported a system-owned interaction (\(code)). System permission prompts are outside the app-owned accessibility route."
        case .mcpRefused(let code):
            "VisionCapture reported an error (\(code)). No action was replayed."
        case .mcpOutcome(let outcome):
            outcome.description + ". Agent Mode stopped. No action was replayed."
        case .noProgress(let message): message
        case .incompleteAnswer:
            "Gemma stopped without a final answer."
        }
    }
}

struct VisionCaptureModelCompletion: Sendable {
    let content: String
    let toolCalls: [AppToolCall]
    let diagnostics: AppDiagnostics
}

struct VisionCaptureAgentRunResult: Sendable {
    let answer: String
    let diagnostics: AppDiagnostics
}

actor VisionCaptureToolLoop {
    typealias Inference = @Sendable (AppToolTurn) async throws
        -> VisionCaptureModelCompletion
    typealias Activity = @Sendable (VisionCaptureActivityEvent) async -> Void
    typealias Checkpoint = @Sendable (AgentContextCheckpointProposal) async throws
        -> DecodeContextCheckpointReceipt

    private enum NavigationOperation: String, Hashable {
        case launch
        case observe
        case screenshot
        case tap
        case setBoolean = "set_boolean"
        case type
        case back
        case swipe
    }

    private enum SwipeDirection: String, CaseIterable, Hashable {
        case up, down, left, right
    }

    private struct NavigationIntent: Hashable {
        let operation: NavigationOperation
        let selector: String?
        let selectorKind: String?
        let role: String?
        let desiredState: Bool?
        let text: String?
        let direction: SwipeDirection?

        init(operation: NavigationOperation, selector: String?, selectorKind: String?,
             role: String?, desiredState: Bool?, text: String?, direction: SwipeDirection? = nil) {
            self.operation = operation
            self.selector = selector
            self.selectorKind = selectorKind
            self.role = role
            self.desiredState = desiredState
            self.text = text
            self.direction = direction
        }
    }

    /// Keeps only the navigation fact needed to describe a revisited route. Typed
    /// text is deliberately excluded from both memory and model feedback.
    private struct JourneyAction: Hashable {
        let operation: NavigationOperation
        let selector: String?
        let selectorKind: String?
        let role: String?
        let desiredState: Bool?
        let readableLabel: String?
        let direction: SwipeDirection?

        init(_ intent: NavigationIntent, readableLabel: String? = nil) {
            operation = intent.operation
            selector = intent.selector.map { String($0.prefix(160)) }
            selectorKind = intent.selectorKind
            role = intent.role.map { String($0.prefix(80)) }
            desiredState = intent.desiredState
            self.readableLabel = readableLabel
            direction = intent.direction
        }
    }

    /// A stable, compact identity for no-progress checks. The coarse signature
    /// captures structural selection while the digest captures the exact
    /// canonical sanitized facts Gemma can use for navigation.
    private struct ScreenContentIdentity: Hashable {
        let coarseSignature: String
        let factsDigest: String
    }

    private enum NavigationScreenScope: Hashable {
        case content(ScreenContentIdentity)
        case fine(String)
    }

    private struct RejectedNavigationAction: Hashable {
        let operation: NavigationOperation
        let selector: String?
        let selectorKind: String?
        let role: String?
        let desiredState: Bool?
        let textDigest: String?
        let direction: SwipeDirection?

        init(_ intent: NavigationIntent) {
            operation = intent.operation
            selector = intent.selector.map { String($0.prefix(160)) }
            selectorKind = intent.selectorKind
            role = intent.role.map { String($0.prefix(80)) }
            desiredState = intent.desiredState
            direction = intent.direction
            textDigest = intent.text.map { text in
                SHA256.hash(data: Data(text.utf8))
                    .map { String(format: "%02x", $0) }
                    .joined()
            }
        }
    }

    private struct RejectedBeforeSubmissionProposal: Hashable {
        let screen: NavigationScreenScope
        let action: RejectedNavigationAction
    }

    private enum JourneyEvent {
        case screen(ScreenContentIdentity)
        case action(JourneyAction)
    }

    private struct StaleActionConfirmation: Hashable {
        let intent: NavigationIntent
        let screenSignature: String
    }

    private struct LocalProposalSignature: Equatable {
        let digest: String

        init(call: AppToolCall, reason: String) {
            var material = Data(call.name.utf8)
            material.append(0)
            if let arguments = try? call.arguments.encoded() {
                material.append(contentsOf: arguments.utf8)
            }
            material.append(0)
            material.append(contentsOf: reason.utf8)
            digest = SHA256.hash(data: material)
                .map { String(format: "%02x", $0) }
                .joined()
        }
    }

    private struct LocalProposalTracker {
        private static let repeatLimit = 3
        private static let trackedSignatureLimit = 24

        private struct ScreenRejection {
            let signature: LocalProposalSignature
            var count: Int
        }

        private var previous: LocalProposalSignature?
        private var repeatCount = 0
        private var screen: ScreenContentIdentity?
        private var screenRejections: [ScreenRejection] = []

        mutating func record(
            call: AppToolCall,
            reason: String,
            screen currentScreen: ScreenContentIdentity?
        ) -> Bool {
            let signature = LocalProposalSignature(call: call, reason: reason)
            if let currentScreen {
                previous = nil
                repeatCount = 0
                if screen != currentScreen {
                    screen = currentScreen
                    screenRejections.removeAll(keepingCapacity: true)
                }
                if let index = screenRejections.firstIndex(where: {
                    $0.signature == signature
                }) {
                    screenRejections[index].count += 1
                    return screenRejections[index].count >= Self.repeatLimit
                }
                screenRejections.append(ScreenRejection(
                    signature: signature,
                    count: 1))
                if screenRejections.count > Self.trackedSignatureLimit {
                    screenRejections.removeFirst(
                        screenRejections.count - Self.trackedSignatureLimit)
                }
                return false
            }

            screen = nil
            screenRejections.removeAll(keepingCapacity: true)
            if signature == previous {
                repeatCount += 1
            } else {
                previous = signature
                repeatCount = 1
            }
            return repeatCount >= Self.repeatLimit
        }

        mutating func reset() {
            previous = nil
            repeatCount = 0
            screen = nil
            screenRejections.removeAll(keepingCapacity: true)
        }
    }

    private struct SessionIdentity: Hashable {
        let id: String
        let kind: String
    }

    private struct PublishedAction: Equatable {
        let action: String
        let selector: String
        let role: String
        let desiredState: Bool?
        let currentState: Bool?
        let actionCapability: String?
        let revalidationCapability: String?
        var displayLabel: String? = nil
        var displayPosition: JSONValue? = nil
        var displaySelected: Bool? = nil
    }

    /// One exact, public accessibility selector for a currently visible and
    /// enabled editable element. The selector kind stays explicit so the host
    /// never relabels an identifier as a label, or the reverse.
    private struct PublishedEditableField: Equatable {
        let selector: String
        let selectorKind: String
        let role: String
    }

    private enum ChoiceRoute {
        case published(PublishedAction, grant: String?)
        case candidate(VisionCaptureScreenFacts.TapCandidate)
        case editable(PublishedEditableField)
        case alert(label: String, digest: String)
        case confirmation(StaleActionConfirmation)
    }

    private struct ChoiceBinding {
        let observation: UInt64
        let targetKey: String
        let session: SessionIdentity?
        let screenSignature: String?
        let operation: NavigationOperation
        let selector: String
        let selectorKind: String?
        let role: String
        let displayLabel: String?
        let allowedStates: [Bool]
        let route: ChoiceRoute
    }

    private struct DecisionPacket {
        let content: String
        let comparison: JSONValue
    }

    private struct AuthorityManifest: Equatable {
        var state = "unavailable"
        var observationGrant: String?
        var actions: [PublishedAction] = []
    }

    /// One READY observation offered to the immediately following model decision.
    /// This is run-local execution metadata, never a retained MCP response.
    private struct ReadyActionOffer {
        let manifest: AuthorityManifest
        let screenSignature: String
        let targetKey: String
        let session: SessionIdentity?
    }

    private struct SystemAlertButton: Equatable {
        let label: String
        let enabled: Bool
        let visible: Bool
    }

    private struct SystemAlertObservation: Equatable {
        let title: String
        let contentDigest: String
        let buttons: [SystemAlertButton]
    }

    private struct PreparedNavigation {
        let result: VisionCaptureMCPResult
        let arguments: JSONValue
        let manifest: AuthorityManifest
        let systemAlert: SystemAlertObservation?
        var screenFacts: VisionCaptureScreenFacts? = nil
        var observationRefreshed = false
    }

    private struct NavigationOutcome {
        let content: String
        let recoverableColdMissArguments: JSONValue?
        let progressed: Bool
        let successfulReadOnlyObservation: Bool
        var imageAttachments: [AppImageAttachment] = []
    }

    private struct RecoverableColdMissTracker {
        private static let repeatLimit = 3
        private var previousActionShape: JSONValue?
        private var repeatCount = 0

        mutating func record(arguments: JSONValue) -> Bool {
            let shape = VisionCaptureToolLoop.coldMissActionShape(arguments)
            if shape == previousActionShape {
                repeatCount += 1
            } else {
                previousActionShape = shape
                repeatCount = 1
            }
            return repeatCount >= Self.repeatLimit
        }

        mutating func reset() {
            previousActionShape = nil
            repeatCount = 0
        }
    }

    /// Bounds only repeated successful read-only choices on the same
    /// sanitized facts. It retains one digest and a count, never an MCP result.
    private struct ReadOnlyNoProgressTracker {
        private static let correctionThreshold = 3

        enum Decision {
            case continueObserving
            case correct(repetitionCount: Int)
            case stop(repetitionCount: Int)
        }

        private var previousDigest: String?
        private var repetitionCount = 0
        private var correctionSent = false

        mutating func record(content: String) -> Decision {
            let source = Data(content.utf8)
            let canonical: Data
            if let value = try? JSONDecoder().decode(
                JSONValue.self,
                from: source) {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                canonical = (try? encoder.encode(value)) ?? source
            } else {
                canonical = source
            }
            let digest = SHA256.hash(data: canonical)
                .map { String(format: "%02x", $0) }
                .joined()
            if digest != previousDigest {
                previousDigest = digest
                repetitionCount = 1
                correctionSent = false
                return .continueObserving
            }
            repetitionCount += 1
            if !correctionSent,
               repetitionCount >= Self.correctionThreshold {
                correctionSent = true
                return .correct(repetitionCount: repetitionCount)
            }
            if correctionSent {
                return .stop(repetitionCount: repetitionCount)
            }
            return .continueObserving
        }

        mutating func reset() {
            previousDigest = nil
            repetitionCount = 0
            correctionSent = false
        }
    }

    private static let logger = Logger(
        subsystem: "TurboFieldfare",
        category: "visioncapture-agent")
    private static let journeyEventLimit = 24
    private static let journeyHintActionLimit = 6
    private static let rejectedProposalLimit = 24
    private static let staleActionBlockLimit = 24

    private var hasInjectedPrompt = false
    private var hasValidatedHostContract = false
    private var committedTargetKey: String?
    private var committedSessionIdentity: SessionIdentity?
    private var currentManifest = AuthorityManifest()
    private var currentSystemAlert: SystemAlertObservation?
    private var uncertainAlertPress: (digest: String, button: String)?
    private var rejectedBeforeSubmissionProposals:
        [RejectedBeforeSubmissionProposal] = []
    private var currentScreenSignature: String?
    private var currentScreenObservation: (signature: String, facts: VisionCaptureScreenFacts)?
    private var currentScreenObservationMetadata: VisionCaptureScreenshot.ObservationMetadata?
    private var currentScreenObservationMetadataInvalid = false
    private var offeredTapCandidates:
        (signature: String, candidates: [VisionCaptureScreenFacts.TapCandidate])?
    private var currentScreenContentIdentity: ScreenContentIdentity?
    private var currentEditableFields: [PublishedEditableField] = []
    /// Armed only during local proposal checks, never during MCP I/O or result parsing.
    private var checkingLocalProposal = false
    private var staleActionConfirmation: StaleActionConfirmation?
    private var blockedStaleActions: Set<StaleActionConfirmation> = []
    private var journeyEvents: [JourneyEvent] = []
    private var currentJourneyHint: String?
    // Conversation-owned: AppModel replaces this actor when starting a new
    // conversation. A new run, launch, or observation does not erase emission history.
    private var lastEmittedJourneyHint: String?
    private var observationGeneration: UInt64 = 0
    private var nextChoiceNumber: UInt64 = 0
    private var currentChoiceBindings: [String: ChoiceBinding] = [:]
    private var permittedNextOperations: Set<NavigationOperation>?
    private var requiresReadOnlyRecovery = false
    private var resolvedJourneyLabel: String?
    private var mcpClient: VisionCaptureMCPClient?
    private let screenshotStore = AppImageAttachmentStore()
    private var taskCheckpoint = AgentTaskCheckpoint()
    private var checkpointRequestIDs: [UUID] = []

    private var currentScreenFacts: VisionCaptureScreenFacts? {
        guard let observation = currentScreenObservation,
              observation.signature == currentScreenSignature else { return nil }
        return observation.facts
    }

    func run(
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity,
        userPrompt: String = "",
        userImages: [AppImageAttachment] = [],
        checkpoint: Checkpoint? = nil,
        forceCheckpoint: @escaping @Sendable () async -> Bool = { false },
        inference: @escaping Inference
    ) async throws -> VisionCaptureAgentRunResult {
        try Task.checkCancellation()
        taskCheckpoint.appendUser(userPrompt, images: userImages)
        defer { screenshotStore.removeAll() }
        rejectedBeforeSubmissionProposals.removeAll(keepingCapacity: true)
        invalidateScreenObservation()
        staleActionConfirmation = nil
        blockedStaleActions.removeAll(keepingCapacity: true)
        journeyEvents.removeAll(keepingCapacity: true)
        currentJourneyHint = nil
        let developerPrompt: String?
        if hasInjectedPrompt {
            developerPrompt = nil
        } else {
            developerPrompt = Self.instructions(configuration: configuration)
            hasInjectedPrompt = true
        }
        var next = AppToolTurn.user(
            developerPrompt: developerPrompt,
            tools: VisionCaptureToolDefinitions.all)
        var knownCallIDs: Set<String> = []
        var localProposals = LocalProposalTracker()
        var recoverableColdMisses = RecoverableColdMissTracker()
        var readOnlyNoProgress = ReadOnlyNoProgressTracker()
        var nextReadyOffer: ReadyActionOffer?

        while true {
            try Task.checkCancellation()
            let readyOffer = nextReadyOffer
            nextReadyOffer = nil
            let completion = try await inference(next)
            // Inference (including its bounded model-only regeneration) has
            // consumed this step. Past pixels remain only in the model KV.
            screenshotStore.removeAll()
            for call in completion.toolCalls {
                guard !call.id.isEmpty, knownCallIDs.insert(call.id).inserted else {
                    let reason = "the tool call ID was empty or repeated"
                    Self.logLocalRejection(call, reason: reason)
                    await activity(.localRejection(
                        id: UUID(), call: call, reason: reason))
                    throw VisionCaptureAgentError.malformedCall(reason)
                }
            }

            if completion.toolCalls.isEmpty {
                guard completion.diagnostics.stopReason != .toolCalls else {
                    throw VisionCaptureAgentError.malformedCall(
                        "the model stopped for a tool call without returning one")
                }
                let answer = completion.content.trimmingCharacters(
                    in: .whitespacesAndNewlines)
                guard !answer.isEmpty else {
                    throw VisionCaptureAgentError.incompleteAnswer
                }
                taskCheckpoint.appendAssessment(answer, source: "final visible model reply")
                return VisionCaptureAgentRunResult(
                    answer: answer,
                    diagnostics: completion.diagnostics)
            }

            guard completion.diagnostics.stopReason == .toolCalls else {
                let reason = "the model returned a tool call without a tool-call stop"
                for call in completion.toolCalls {
                    Self.logLocalRejection(call, reason: reason)
                    await activity(.localRejection(
                        id: UUID(), call: call, reason: reason))
                }
                throw VisionCaptureAgentError.malformedCall(reason)
            }

            let call: AppToolCall
            do {
                call = try Self.preflight(completion.toolCalls)
            } catch let error as VisionCaptureAgentError {
                for proposed in completion.toolCalls {
                    Self.logLocalRejection(proposed, reason: error.description)
                    await activity(.localRejection(
                        id: UUID(), call: proposed, reason: error.description))
                }
                throw error
            }

            var content: String
            var executionOutcome: String
            var images: [AppImageAttachment] = []
            checkpointRequestIDs.removeAll(keepingCapacity: true)
            let historicalTarget = checkpointTarget(for: call)
            taskCheckpoint.appendAssessment(completion.content, source: "model text before call \(call.id)")
            do {
                checkingLocalProposal = true
                let intent = try navigationIntent(from: call, configuration: configuration)
                try configuration.validate()
                if let committedTargetKey,
                   committedTargetKey != configuration.targetKey {
                    throw VisionCaptureAgentError.identityMismatch
                }
                try ensureHostContract(configuration: configuration)
                let outcome = try await perform(
                    intent,
                    readyOffer: readyOffer,
                    configuration: configuration,
                    activity: activity)
                checkingLocalProposal = false
                images = outcome.imageAttachments
                executionOutcome = outcome.content
                let packet = try decisionPacket(
                    from: outcome.content, call: call, configuration: configuration,
                    images: images)
                content = packet.content
                if outcome.progressed {
                    localProposals.reset()
                    readOnlyNoProgress.reset()
                }
                if outcome.successfulReadOnlyObservation {
                    switch readOnlyNoProgress.record(content: try packet.comparison.encoded()) {
                    case .continueObserving:
                        break
                    case .correct(let repetitionCount):
                        content = try Self.addingReadOnlyNoProgressCorrection(
                            to: content,
                            repetitionCount: repetitionCount)
                    case .stop(let repetitionCount):
                        throw VisionCaptureAgentError.noProgress(
                            "Agent Mode stopped after \(repetitionCount) equivalent successful read-only observations returned the same sanitized screen and navigation choices without an intervening successful mutation. Gemma repeated observe after one explicit correction. No action was dispatched by those reads.")
                    }
                }
                if let arguments = outcome.recoverableColdMissArguments {
                    if recoverableColdMisses.record(arguments: arguments) {
                        throw VisionCaptureAgentError.noProgress(
                            "Agent Mode stopped after three equivalent safe pre-dispatch cache refusals for the same intended action. No refused action was replayed.")
                    }
                } else if outcome.progressed {
                    recoverableColdMisses.reset()
                }
                if outcome.successfulReadOnlyObservation,
                   currentManifest.state == "ready", currentSystemAlert == nil,
                   let signature = currentScreenSignature {
                    nextReadyOffer = ReadyActionOffer(
                        manifest: currentManifest, screenSignature: signature,
                        targetKey: configuration.targetKey, session: committedSessionIdentity)
                }
            } catch let error as VisionCaptureAgentError
                where checkingLocalProposal && Self.isRecoverableProposalError(error) {
                checkingLocalProposal = false
                // A correction declines the one confirming offer. Retiring only
                // its public ID would let a later observation revive the attempt.
                try retirePendingConfirmation()
                let reason = Self.localRejectionReason(error)
                Self.logLocalRejection(call, reason: reason)
                await activity(.localRejection(
                    id: UUID(), call: call, reason: reason))
                if localProposals.record(
                    call: semanticProposalForRepeatCheck(call),
                    reason: reason,
                    screen: currentScreenContentIdentity
                ) {
                    throw VisionCaptureAgentError.malformedCall(reason)
                }
                let failure = try Self.proposalFailureResult(
                    error,
                    facts: currentProposalRepairFacts(configuration: configuration))
                executionOutcome = failure
                content = try decisionPacket(
                    from: failure, call: call, configuration: configuration, images: []).content
            }

            let settledResult = AppToolResult(
                    callID: call.id,
                    name: call.name,
                    content: content,
                    imageAttachments: images)
            try taskCheckpoint.appendSettled(call: call, result: settledResult,
                outcome: executionOutcome, target: historicalTarget, requestIDs: checkpointRequestIDs,
                session: checkpointSessionReference)
            next = .results([settledResult])
            if let checkpoint {
                let id = UUID()
                let replacementEpoch = UUID()
                let force = await forceCheckpoint()
                let permitsScreenshot = AppVisionPackInstallationProbe.status(at: configuration.modelDirectory) == .complete
                let assessed = try await checkpoint(AgentContextCheckpointProposal(
                    id: id, replacementEpoch: replacementEpoch, call: call, result: settledResult,
                    record: "",
                    commit: false, force: force, permitsScreenshot: permitsScreenshot))
                if assessed.needed {
                    try Task.checkCancellation()
                    nextReadyOffer = nil
                    try retirePendingConfirmation()
                    let hadAlert = currentSystemAlert != nil || uncertainAlertPress != nil
                    currentManifest = AuthorityManifest()
                    invalidateScreenObservation()
                    checkpointRequestIDs.removeAll(keepingCapacity: true)
                    let refreshStarted = ContinuousClock.now
                    let refreshed = try await checkpointObservation(
                        hadAlert: hadAlert, configuration: configuration, activity: activity)
                    await AgentInferenceTrace.shared?.checkpointRead(id: id,
                        seconds: Self.elapsedSeconds(since: refreshStarted))
                    let refreshCall = AppToolCall(id: "checkpoint-read-\(id.uuidString)",
                        name: call.name, arguments: .object(["action": .string("observe")]))
                    let freshPacket = try decisionPacket(from: refreshed.content, call: refreshCall,
                        configuration: configuration, images: []).content
                    try taskCheckpoint.appendSettled(call: refreshCall,
                        result: AppToolResult(callID: refreshCall.id, name: refreshCall.name, content: freshPacket),
                        outcome: refreshed.content, target: nil, requestIDs: checkpointRequestIDs,
                        session: checkpointSessionReference, origin: "host_read_only_checkpoint_refresh")
                    taskCheckpoint.nextRevision()
                    let receipt = try await checkpoint(AgentContextCheckpointProposal(
                        id: id, replacementEpoch: replacementEpoch, call: call, result: settledResult,
                        record: try taskCheckpoint.render(currentPacket: freshPacket, safety: checkpointSafety(configuration)),
                        commit: true, force: force, permitsScreenshot: permitsScreenshot))
                    guard receipt.committed else {
                        throw VisionCaptureAgentError.noProgress("The context checkpoint was not acknowledged. No action was replayed.")
                    }
                    try Task.checkCancellation()
                    next = .checkpoint(id)
                }
            }
        }
    }

    private var checkpointSessionReference: String? {
        committedSessionIdentity.map { "\($0.kind):sha256:\(AgentTaskCheckpoint.digest($0.id))" }
    }

    private func checkpointTarget(for call: AppToolCall) -> JSONValue? {
        guard case .string(let id)? = call.arguments.objectValue?["target"],
              let binding = currentChoiceBindings[id] else { return nil }
        var facts: [String: JSONValue] = ["observation": .string("o\(binding.observation)"),
            "label": binding.displayLabel.map(JSONValue.string) ?? .null,
            "selector_at_that_time": .string(binding.selector),
            "selector_kind": binding.selectorKind.map(JSONValue.string) ?? .null,
            "role": .string(binding.role), "action": .string(binding.operation.rawValue)]
        if let properties = currentScreenFacts?.properties(selector: binding.selector, role: binding.role) {
            for key in ["value", "position", "selected", "enabled", "current_state"] { facts[key] = properties[key] }
        }
        return .object(facts)
    }

    private func checkpointSafety(_ configuration: VisionCaptureAgentConfiguration) -> JSONValue {
        .object(["bundle_identifier": .string(configuration.bundleIdentifier),
            "simulator_udid": .string(configuration.simulatorUDID),
            "read_only_recovery_required": .bool(requiresReadOnlyRecovery),
            "uncertain_alert_press": uncertainAlertPress.map {
                .object(["button": .string($0.button), "observation_digest": .string($0.digest)])
            } ?? .null,
            "retired_confirmation_count": .integer(Int64(blockedStaleActions.count)),
            "rejected_target_count": .integer(Int64(rejectedBeforeSubmissionProposals.count)),
            "restrictions": .string("The same host actor retains rejected targets, spent or declined confirmations, unknown input restrictions and all loop-protection counters. A checkpoint cannot renew them. Earlier target IDs are expired. Do not retry or replace any delivery-unknown input; use only permitted current read-only recovery. No action verdict alone completes a user goal.")])
    }

    private func checkpointObservation(hadAlert: Bool,
        configuration: VisionCaptureAgentConfiguration, activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        let observed: PreparedNavigation
        if hadAlert {
            observed = try await describeSystemAlert(configuration: configuration, activity: activity)
        } else {
            observed = try await describeScreenAfterCacheValidationFailure(configuration: configuration, activity: activity)
        }
        guard !observed.result.isError else {
            throw VisionCaptureAgentError.noProgress("The checkpoint's read-only refresh failed. Completed input was not replayed; the task remains unfinished.")
        }
        return try outcome(for: NavigationIntent(operation: .observe, selector: nil,
                selectorKind: nil, role: nil, desiredState: nil, text: nil),
            result: observed.result, arguments: observed.arguments, manifest: observed.manifest,
            systemAlert: observed.systemAlert, progressed: false,
            observedScreenFacts: currentScreenFacts, observationRefreshed: true)
    }

    private func ensureHostContract(
        configuration: VisionCaptureAgentConfiguration
    ) throws {
        guard !hasValidatedHostContract else { return }
        do {
            let reader = try VisionCaptureSkillReader(
                modelDirectory: configuration.modelDirectory)
            for name in VisionCaptureSkillName.allCases {
                _ = try reader.read(name.rawValue)
            }
            hasValidatedHostContract = true
        } catch let error as VisionCaptureSkillReaderError {
            throw VisionCaptureAgentError.skillReadFailed(error.description)
        }
    }

    private func perform(
        _ intent: NavigationIntent,
        readyOffer: ReadyActionOffer?,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        try Task.checkCancellation()
        try rejectPreviouslyRejectedProposal(intent)
        let staleConfirmation = try consumeStaleActionConfirmation(
            for: intent)
        switch intent.operation {
        case .launch:
            currentManifest = AuthorityManifest()
            currentSystemAlert = nil
            uncertainAlertPress = nil
            invalidateScreenObservation()
            staleActionConfirmation = nil
            blockedStaleActions.removeAll(keepingCapacity: true)
            rejectedBeforeSubmissionProposals.removeAll(
                keepingCapacity: true)
            journeyEvents.removeAll(keepingCapacity: true)
            currentJourneyHint = nil
            let arguments = makeMCPArguments(
                request: "launch app",
                parameters: [:],
                configuration: configuration)
            let result = try await executeHostRequest(
                arguments,
                configuration: configuration,
                activity: activity)
            if let timeout = Self.isolatedLaunchTimeout(
                result, arguments: arguments, configuration: configuration) {
                return try await observeAfterLaunchTimeout(
                    timeout, configuration: configuration, activity: activity)
            }
            if !result.isError {
                try recordScreenObservation(from: result.value)
            }
            let manifest = try Self.returnedAuthorityManifest(from: result.value)
            currentManifest = manifest
            return try outcome(
                for: intent,
                result: result,
                arguments: arguments,
                manifest: manifest,
                systemAlert: nil,
                progressed: !result.isError)

        case .screenshot:
            return try await screenshotAndObserve(configuration: configuration, activity: activity)

        case .observe:
            if currentSystemAlert != nil {
                let observed = try await describeSystemAlert(
                    configuration: configuration,
                    activity: activity)
                return try outcome(
                    for: intent,
                    result: observed.result,
                    arguments: observed.arguments,
                    manifest: observed.manifest,
                    systemAlert: observed.systemAlert,
                    progressed: false)
            }
            let prepared = try await refreshNavigation(
                configuration: configuration,
                activity: activity)
            return try outcome(
                for: intent,
                result: prepared.result,
                arguments: prepared.arguments,
                manifest: prepared.manifest,
                systemAlert: prepared.systemAlert,
                progressed: false,
                observedScreenFacts: prepared.screenFacts,
                observationRefreshed: prepared.observationRefreshed)

        case .tap:
            if intent.role == "system_alert_button" {
                return try await pressSystemAlertButton(
                    intent,
                    configuration: configuration,
                    activity: activity)
            }
            fallthrough

        case .setBoolean:
            let candidateSignature: String?
            if intent.operation == .tap, staleConfirmation == nil,
               let offered = offeredTapCandidates,
               offered.signature == currentScreenSignature,
               offered.candidates.contains(where: {
                   $0.selector.utf8.elementsEqual((intent.selector ?? "").utf8)
                       && $0.role.utf8.elementsEqual((intent.role ?? "").utf8)
               }) {
                candidateSignature = offered.signature
            } else {
                candidateSignature = nil
            }
            let desiredAction: JSONValue?
            if candidateSignature != nil, let selector = intent.selector, let role = intent.role {
                desiredAction = .object([
                    "action": .string("tap"),
                    "selector": .string(selector),
                    "role": .string(role),
                ])
                // Force a fresh fine identity, via granted describe or the
                // existing plain-read/refresh path, before validating this choice.
                currentManifest = AuthorityManifest()
                invalidateScreenObservation()
            } else {
                desiredAction = nil
            }
            let manifest: AuthorityManifest
            if candidateSignature == nil, staleConfirmation == nil,
               let reusable = reusableReadyManifest(
                    for: intent, offer: readyOffer, configuration: configuration) {
                // The cached-action endpoint validates this unused handle live.
                // Save only the redundant client inspection, not any safety gate.
                manifest = reusable
            } else {
                let prepared = try await refreshNavigation(
                    configuration: configuration,
                    activity: activity,
                    desiredAction: desiredAction)
                if prepared.observationRefreshed {
                    return try observationRefreshOutcome(for: intent, prepared: prepared)
                }
                if prepared.result.isRecoverableColdMiss {
                    if staleConfirmation != nil {
                        throw VisionCaptureAgentError.noProgress(
                            "The one confirming stale-capability attempt could not obtain fresh bound navigation evidence. Agent Mode stopped this action without dispatch instead of starting another confirming attempt.")
                    }
                    return try outcome(
                        for: intent,
                        result: prepared.result,
                        arguments: prepared.arguments,
                        manifest: prepared.manifest,
                        systemAlert: prepared.systemAlert,
                        progressed: false)
                }
                checkingLocalProposal = true
                try rejectPreviouslyRejectedProposal(intent)
                if let candidateSignature {
                    let sameScreen = currentScreenSignature == candidateSignature
                    let matching = prepared.manifest.actions.filter {
                        $0.action == "tap"
                            && $0.selector.utf8.elementsEqual((intent.selector ?? "").utf8)
                            && $0.role.utf8.elementsEqual((intent.role ?? "").utf8)
                    }
                    if !sameScreen || matching.count != 1 || prepared.systemAlert != nil {
                        if sameScreen { rememberRejectedBeforeSubmissionProposal(intent) }
                        checkingLocalProposal = false
                        return try observedCandidateRefusal(
                            intent, prepared: prepared, screenChanged: !sameScreen)
                    }
                }
                manifest = prepared.manifest
            }
            checkingLocalProposal = true
            let action = try Self.uniquePublishedAction(
                for: intent,
                in: manifest)
            if let staleConfirmation,
               currentScreenSignature != staleConfirmation.screenSignature {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "The confirming action was not sent because the content signature for visible text and identifiers changed while obtaining fresh navigation evidence. Choose observe and make a new action decision from the current screen.")
            }
            let actionArguments = try makeActionArguments(
                intent: intent,
                action: action,
                manifest: manifest,
                configuration: configuration)
            let isStaleConfirmation = staleConfirmation != nil
            let staleBaselineScreen = currentScreenSignature
            currentManifest = AuthorityManifest()
            invalidateScreenObservation()
            let result = try await executeHostRequest(
                actionArguments,
                configuration: configuration,
                activity: activity)
            if isRecoverableActionAuthorizationExpiry(
                result, arguments: actionArguments, configuration: configuration) {
                return try await observeAfterActionAuthorizationExpiry(
                    intent, failure: result.serverOutcome,
                    configuration: configuration, activity: activity)
            }
            if isPermittedDeliveredTransition(
                result, arguments: actionArguments, configuration: configuration) {
                return try await observeAfterDeliveredTransition(
                    intent, failure: result.serverOutcome,
                    baselineScreenSignature: staleBaselineScreen,
                    configuration: configuration, activity: activity)
            }
            let layoutChangedBeforeRevalidation = isRecoverableRevalidationLayoutChange(
                result, arguments: actionArguments, configuration: configuration)
            if layoutChangedBeforeRevalidation || Self.isRecoverableStaleCacheAction(
                result,
                arguments: actionArguments
            ) {
                currentManifest = AuthorityManifest()
                if isStaleConfirmation, !layoutChangedBeforeRevalidation {
                    staleActionConfirmation = nil
                    throw VisionCaptureAgentError.noProgress(
                        "VisionCapture returned CACHE_ACTION_CAPABILITY_STALE twice for the same action with the same content signature for visible text and identifiers. Agent Mode stopped because this action reached the host's one-confirming-attempt limit. No input was submitted and no further attempt was made.")
                }
                return try await recoverFromStaleCacheAction(
                    intent,
                    baselineScreenSignature: staleBaselineScreen,
                    refreshChoices: layoutChangedBeforeRevalidation,
                    isConfirmation: isStaleConfirmation,
                    configuration: configuration,
                    activity: activity)
            }
            if result.isGuardedTargetRejectedBeforeSubmission {
                rememberRejectedBeforeSubmissionProposal(intent)
            }
            if !result.isError {
                try recordCompletedJourneyAction(
                    intent,
                    result: result.value)
                staleActionConfirmation = nil
                if let staleConfirmation {
                    blockedStaleActions.remove(staleConfirmation)
                }
            }
            let terminalManifest = try Self.returnedAuthorityManifest(
                from: result.value)
            currentManifest = terminalManifest
            return try outcome(
                for: intent,
                result: result,
                arguments: actionArguments,
                manifest: terminalManifest,
                systemAlert: nil,
                progressed: !result.isError)

        case .type:
            let prepared = try await refreshNavigation(
                configuration: configuration,
                activity: activity)
            if prepared.observationRefreshed {
                return try observationRefreshOutcome(for: intent, prepared: prepared)
            }
            if prepared.result.isRecoverableColdMiss {
                return try outcome(
                    for: intent,
                    result: prepared.result,
                    arguments: prepared.arguments,
                    manifest: prepared.manifest,
                    systemAlert: prepared.systemAlert,
                    progressed: false)
            }
            checkingLocalProposal = true
            try rejectPreviouslyRejectedProposal(intent)
            var parameters: [String: JSONValue] = [:]
            if let selector = intent.selector {
                let field = try uniquePublishedEditableField(for: intent)
                parameters[field.selectorKind] = .string(selector)
            }
            guard let text = intent.text else {
                throw VisionCaptureAgentError.malformedCall(
                    "type requires non-empty text")
            }
            let arguments = makeMCPArguments(
                request: "type \(text)",
                parameters: parameters,
                configuration: configuration)
            currentManifest = AuthorityManifest()
            invalidateScreenObservation()
            let result = try await executeHostRequest(
                arguments,
                configuration: configuration,
                activity: activity)
            if result.isGuardedTargetRejectedBeforeSubmission {
                rememberRejectedBeforeSubmissionProposal(intent)
            }
            if !result.isError {
                try recordCompletedJourneyAction(
                    intent,
                    result: result.value)
                staleActionConfirmation = nil
                try recordScreenObservation(from: result.value)
            }
            let terminalManifest = try Self.returnedAuthorityManifest(
                from: result.value)
            currentManifest = terminalManifest
            if !result.isError,
               let typingProof = try Self.sanitizedNamedObject(
                   "proof", allowedKeys: ["verdict", "verdict_source", "reason_code", "action"],
                   in: result.value),
               typingProof["verdict"] == .string("verified"),
               eligibleOfferedActions(terminalManifest.actions).isEmpty,
               !(currentScreenFacts?.tapCandidates(
                   excluding: terminalManifest.actions.map { ($0.selector, $0.role) }).contains { candidate in
                   isEligibleOfferedAction(NavigationIntent(
                       operation: .tap, selector: candidate.selector, selectorKind: nil,
                       role: candidate.role, desiredState: nil, text: nil))
               } ?? false) {
                // Typing can enable a button. Retire its old observations and
                // fetch current choices before the model decides what to do.
                currentManifest = AuthorityManifest()
                invalidateScreenObservation()
                let refreshed: PreparedNavigation
                do {
                    refreshed = try await refreshNavigation(
                        configuration: configuration, activity: activity)
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    throw VisionCaptureAgentError.noProgress(
                        "VisionCapture verified the typing. The following observation failed: \(error). The typed input was not repeated.")
                }
                return try outcome(
                    for: intent, result: refreshed.result, arguments: refreshed.arguments,
                    manifest: refreshed.manifest, systemAlert: refreshed.systemAlert,
                    progressed: true, observedScreenFacts: refreshed.screenFacts,
                    verifiedTypingProof: typingProof,
                    typingDispatchAttempted: result.dispatchAttempted,
                    observationRefreshed: refreshed.observationRefreshed)
            }
            return try outcome(
                for: intent,
                result: result,
                arguments: arguments,
                manifest: terminalManifest,
                systemAlert: nil,
                progressed: !result.isError)

        case .back, .swipe:
            let prepared = try await refreshNavigation(
                configuration: configuration,
                activity: activity)
            if prepared.observationRefreshed {
                return try observationRefreshOutcome(for: intent, prepared: prepared)
            }
            if prepared.result.isRecoverableColdMiss {
                return try outcome(
                    for: intent,
                    result: prepared.result,
                    arguments: prepared.arguments,
                    manifest: prepared.manifest,
                    systemAlert: prepared.systemAlert,
                    progressed: false)
            }
            checkingLocalProposal = true
            try rejectPreviouslyRejectedProposal(intent)
            guard committedSessionIdentity?.kind == "flow" else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "This action was not sent because VisionCapture has not returned a current flow session. Observe the app again before choosing this action.")
            }
            let request: String
            if intent.operation == .swipe {
                guard let direction = intent.direction, prepared.systemAlert == nil,
                      currentScreenSignature != nil else {
                    throw VisionCaptureAgentError.navigationUnavailable(
                        "Swipe requires a current app observation and an offered direction. No gesture was sent.")
                }
                request = "swipe \(direction.rawValue)"
            } else {
                request = "go back"
            }
            let arguments = makeMCPArguments(
                request: request,
                parameters: [:],
                configuration: configuration)
            currentManifest = AuthorityManifest()
            invalidateScreenObservation()
            let result = try await executeHostRequest(
                arguments,
                configuration: configuration,
                activity: activity)
            if result.isGuardedTargetRejectedBeforeSubmission {
                rememberRejectedBeforeSubmissionProposal(intent)
            }
            if !result.isError {
                try recordCompletedJourneyAction(
                    intent,
                    result: result.value)
                staleActionConfirmation = nil
            }
            let terminalManifest = try Self.returnedAuthorityManifest(
                from: result.value)
            currentManifest = terminalManifest
            if intent.operation == .swipe, !result.isError {
                return try await observeAfterSwipe(
                    intent, result: result, configuration: configuration, activity: activity)
            }
            return try outcome(
                for: intent,
                result: result,
                arguments: arguments,
                manifest: terminalManifest,
                systemAlert: nil,
                progressed: !result.isError)
        }
    }

    /// Keep the gesture's proof separate from the following read. Neither a
    /// successful request nor fresh screen facts can upgrade an inconclusive swipe.
    private func observeAfterSwipe(
        _ intent: NavigationIntent,
        result: VisionCaptureMCPResult,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        let proof = try Self.sanitizedNamedObject(
            "proof", allowedKeys: ["verdict", "verdict_source", "reason_code", "action"],
            in: result.value)
        let delivery = try Self.sanitizedDeliveryFacts(in: result.value)
        let canonicalVerdict: String
        if case .string(let value)? = proof?["verdict"] {
            canonicalVerdict = value
        } else {
            canonicalVerdict = "unavailable"
        }
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        staleActionConfirmation = nil
        invalidateScreenObservation()
        let refreshed: PreparedNavigation
        do {
            refreshed = try await refreshNavigation(configuration: configuration, activity: activity)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            currentManifest = AuthorityManifest()
            currentSystemAlert = nil
            invalidateScreenObservation()
            throw VisionCaptureAgentError.noProgress(
                "The swipe request returned, with proof \(canonicalVerdict). Its following observation failed: \(error). The gesture was not repeated.")
        }
        let observed = try outcome(
            for: NavigationIntent(operation: .observe, selector: nil, selectorKind: nil,
                                  role: nil, desiredState: nil, text: nil),
            result: refreshed.result, arguments: refreshed.arguments, manifest: refreshed.manifest,
            systemAlert: refreshed.systemAlert, progressed: false,
            observedScreenFacts: refreshed.screenFacts,
            observationRefreshed: refreshed.observationRefreshed)
        guard case .object(var body) = try JSONDecoder().decode(
            JSONValue.self, from: Data(observed.content.utf8)) else {
            throw VisionCaptureAgentError.noProgress(
                "The swipe's following observation could not be retained. The gesture was not repeated.")
        }
        // A cold or refreshed inspection describes its own dispatch. Do not
        // attribute those read flags or proof to the preceding gesture.
        for key in ["proof", "dispatch_attempted", "submission_started", "delivery_acknowledged", "delivery_unknown"] {
            body.removeValue(forKey: key)
        }
        body["operation"] = .string("swipe")
        body["direction"] = intent.direction.map { .string($0.rawValue) }
        if let proof { body["proof"] = .object(proof) }
        body.merge(delivery) { _, returned in returned }
        if let attempted = result.dispatchAttempted { body["dispatch_attempted"] = .bool(attempted) }
        let verdict = proof?["verdict"]
        body["outcome"] = verdict == .string("verified") ? .string("succeeded")
            : verdict ?? .string("unknown")
        body["observation_outcome"] = .string(refreshed.result.isError ? "unavailable" : "succeeded")
        body["instruction"] = .string(delivery["delivery_unknown"] == .bool(true)
            ? "Swipe delivery remains unknown. It was not repeated. These facts come from a separate read; choose only read-only recovery and do not replay the gesture."
            : "These facts come from the read after the swipe. Use its canonical verdict without assuming the intended effect. Choose again from the fresh directions and target IDs; old IDs have expired.")
        if verdict != .string("verified") {
            body["outcome_note"] = .string(
                "The swipe's intended effect was not verified. The following observation does not prove that effect.")
        }
        return NavigationOutcome(
            content: try encodeOutcomeBody(body), recoverableColdMissArguments: nil,
            progressed: verdict == .string("verified"),
            successfulReadOnlyObservation: !refreshed.result.isError)
    }

    /// The launch contract allows one plain read after an isolated command
    /// timeout. The pre-launch foreground snapshot never supplies current facts.
    private func observeAfterLaunchTimeout(
        _ timeout: [String: JSONValue],
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        discardReturnedExecutionEvidence()
        do {
            let arguments = makeMCPArguments(
                request: "describe screen", parameters: [:],
                configuration: configuration)
            let observed = try await executeHostRequest(
                arguments, configuration: configuration, activity: activity)
            guard !observed.isError,
                  try Self.provesRequestedForegroundAfterLaunchTimeout(
                    observed.value, configuration: configuration) else {
                throw VisionCaptureAgentError.launchOutcomeUnproven(
                    "The plain description omitted or contradicted the required app, device, process, or screen identity.")
            }
            try recordScreenObservation(from: observed.value)
            // A plain read is evidence only. Do not retain action handles from it.
            let current = try outcome(
                for: NavigationIntent(operation: .observe, selector: nil, selectorKind: nil,
                                      role: nil, desiredState: nil, text: nil),
                result: observed, arguments: arguments, manifest: AuthorityManifest(),
                systemAlert: nil, progressed: false)
            guard case .object(var body) = try JSONDecoder().decode(
                JSONValue.self, from: Data(current.content.utf8)) else {
                throw VisionCaptureAgentError.launchOutcomeUnproven("The current observation could not be retained.")
            }
            body["operation"] = .string("launch")
            body["outcome"] = .string("unknown_reobserved")
            body["launch"] = .object(timeout.filter {
                ["verdict", "reason_code", "mutation_sent", "failed_proof_stage"].contains($0.key)
            })
            body["current_observation"] = .object([
                "outcome": .string("succeeded"),
                "requested_foreground_app_proven": .bool(true),
            ])
            body["instruction"] = .string(
                "Launch delivery and outcome remain unknown. A separate read found the requested app in front. Do not relaunch or claim launch succeeded. Choose observe for current choices.")
            return NavigationOutcome(
                content: try JSONValue.object(body).encoded(),
                recoverableColdMissArguments: nil, progressed: false,
                successfulReadOnlyObservation: true)
        } catch is CancellationError {
            discardReturnedExecutionEvidence()
            throw CancellationError()
        } catch {
            discardReturnedExecutionEvidence()
            if case VisionCaptureAgentError.launchOutcomeUnproven = error { throw error }
            throw VisionCaptureAgentError.launchOutcomeUnproven(String(describing: error))
        }
    }

    private func discardReturnedExecutionEvidence() {
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        committedSessionIdentity = nil
        staleActionConfirmation = nil
        invalidateScreenObservation()
    }

    private static func isolatedLaunchTimeout(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> [String: JSONValue]? {
        guard result.isError, result.refusalCode == "SIMULATOR_UNRESPONSIVE",
              !result.hasConflictingDispatchAttemptEvidence, result.dispatchAttempted == nil,
              let request = arguments.objectValue,
              request["request"] == .string("launch app"),
              request["bundle_id"] == .string(configuration.bundleIdentifier),
              request["parameters"] == .object(["udid": .string(configuration.simulatorUDID)]),
              let root = result.value.objectValue,
              root["diagnosis"] == .string("isolated_operation_timeout"),
              root["udid"] == .string(configuration.simulatorUDID),
              let timeout = root["launch_outcome"]?.objectValue,
              timeout["verdict"] == .string("outcome_unproven"),
              timeout["reason_code"] == .string("SIMULATOR_UNRESPONSIVE"),
              timeout["mutation_sent"] == .string("unknown"),
              timeout["failed_proof_stage"] == .string("launch_command_result"),
              timeout["requested_udid"] == .string(configuration.simulatorUDID),
              timeout["bound_udid"] == .string(configuration.simulatorUDID),
              case .array(let content)? = root["content"] else { return nil }
        let errors = content.compactMap { $0.objectValue?["text"] }.compactMap { value -> String? in
            guard case .string(let text) = value, text.hasPrefix("Error [") else { return nil }
            return text
        }
        guard !errors.isEmpty,
              errors.allSatisfy({ $0.hasPrefix("Error [SIMULATOR_UNRESPONSIVE]:") }) else { return nil }
        do {
            var consistent = true
            try collectStructuredObjects(named: "launch_outcome", in: result.value) {
                if $0 != timeout { consistent = false }
            }
            return consistent ? timeout : nil
        } catch {
            return nil
        }
    }

    private static func provesRequestedForegroundAfterLaunchTimeout(
        _ value: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) throws -> Bool {
        var provenPID: Int64?
        var consistent = true
        try collectStructuredObjects(named: "payload", in: value) { payload in
            guard let app = payload["app"]?.objectValue,
                  let device = payload["device"]?.objectValue,
                  app["bundle_id_requested"] == .string(configuration.bundleIdentifier),
                  app["bundle_id_active"] == .string(configuration.bundleIdentifier),
                  device["udid"] == .string(configuration.simulatorUDID),
                  case .integer(let observedPID)? = app["process_id"], observedPID > 0,
                  payload["pid"] == .integer(observedPID) else {
                consistent = false
                return
            }
            if let provenPID, provenPID != observedPID { consistent = false }
            provenPID = observedPID
        }
        guard provenPID != nil, consistent else { return false }
        return try returnedScreenSignature(from: value) != nil
    }

    private func screenshotAndObserve(
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        guard AppVisionPackInstallationProbe.status(at: configuration.modelDirectory) == .complete else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "Screenshot image support is unavailable. Install and activate the vision companion, then load the model. No screenshot was requested.")
        }
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        invalidateScreenObservation()
        // Cache inspection takes an exclusive server lease and increments its
        // observation invalidation counter, even without input. Finish it
        // before the two observations whose generations we compare.
        let inspectArguments = makeMCPArguments(
            request: "inspect cache", parameters: [:], configuration: configuration)
        let inspectResult = try await executeHostRequest(
            inspectArguments, configuration: configuration, activity: activity)
        let inspectedManifest = try Self.returnedAuthorityManifest(from: inspectResult.value)
        let observationGrant = !inspectResult.isError && inspectedManifest.actions.isEmpty
            ? inspectedManifest.observationGrant : nil
        // Retain only the grant locally. Never publish an inspect action or
        // field that predates the image, including the warm-cache branch.
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        invalidateScreenObservation()
        let arguments = makeMCPArguments(
            request: "take a screenshot", parameters: [:], configuration: configuration,
            includeFlowSession: false)
        let activityID = UUID()
        let result = try await executeHostRequest(
            arguments, configuration: configuration, activityID: activityID, activity: activity)
        let image = try VisionCaptureScreenshot.stage(
            result, in: screenshotStore, expectedDeviceID: configuration.simulatorUDID)
        let imageMetadata = try VisionCaptureScreenshot.observationMetadata(in: result)
        await activity(.screenshot(id: activityID, image: image))

        var body: [String: JSONValue]
        var relationship: [String: JSONValue] = [
            "order": .string("image_then_accessibility"),
            "visual_agreement": .string("unknown"),
        ]
        if let source = imageMetadata.source { relationship["image_source"] = .string(source) }
        do {
            let prepared = try await readAfterScreenshot(
                observationGrant: observationGrant,
                inspectResult: inspectResult, inspectArguments: inspectArguments,
                configuration: configuration, activity: activity)
            let observed = try outcome(
                for: NavigationIntent(operation: .observe, selector: nil, selectorKind: nil,
                    role: nil, desiredState: nil, text: nil),
                result: prepared.result, arguments: prepared.arguments,
                manifest: prepared.manifest, systemAlert: prepared.systemAlert,
                progressed: false, observedScreenFacts: prepared.screenFacts,
                observationRefreshed: prepared.observationRefreshed)
            guard let observedBody = try JSONDecoder().decode(
                JSONValue.self, from: Data(observed.content.utf8)).objectValue else {
                throw VisionCaptureAgentError.malformedCall("The subsequent screen read could not be encoded.")
            }
            body = observedBody
            let readMetadata = prepared.systemAlert != nil
                ? try VisionCaptureScreenshot.observationMetadata(in: prepared.result)
                : currentScreenObservationMetadata
            if let source = readMetadata?.source { relationship["read_source"] = .string(source) }
            let unavailable = observationGrant == nil || prepared.result.isError
                || (currentScreenSignature == nil && currentSystemAlert == nil)
                || currentScreenObservationMetadataInvalid
            let pairState = Self.imageReadState(
                image: imageMetadata, read: readMetadata,
                expectedDeviceID: configuration.simulatorUDID, readUnavailable: unavailable)
            relationship["state"] = .string(pairState)
            if pairState != "sequential" {
                body["observation_outcome"] = .string("unavailable")
                body["instruction"] = .string(unavailable
                    ? "The image was captured. Subsequent readable facts may be available, but this pair did not establish a current executable binding. Choose observe before input."
                    : "The image and subsequent read have conflicting or cached freshness information. No targets are offered from this pair. Choose observe before input.")
                // Keep readable facts, but retire every route and handle from
                // the unusable pair. A later correction cannot restore it.
                currentManifest = AuthorityManifest()
                currentSystemAlert = nil
                invalidateScreenObservation()
            } else {
                body["observation_outcome"] = .string("succeeded")
                body["instruction"] = .string(
                    "The image was captured before the current accessibility read. Choices and positions come only from that later read; visual agreement is unverified. You may choose a current target from the text facts. If the image and facts disagree or the target remains ambiguous, observe or report the uncertainty. Do not guess a target or retry refused input.")
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            currentManifest = AuthorityManifest()
            currentSystemAlert = nil
            invalidateScreenObservation()
            relationship["state"] = .string("read_unavailable")
            body = [
                "available_actions": .array([]),
                "observation_outcome": .string("unavailable"),
                "instruction": .string(
                    "The image was captured, but the subsequent accessibility read failed. Choose observe for current targets before input."),
            ]
        }
        body["operation"] = .string("screenshot")
        body["outcome"] = .string("succeeded")
        body.removeValue(forKey: "proof")
        body.removeValue(forKey: "dispatch_attempted")
        body["image"] = .object(["mime_type": .string("image/png")])
        body["image_observation"] = .object(relationship)
        return NavigationOutcome(
            content: try JSONValue.object(body).encoded(),
            recoverableColdMissArguments: nil, progressed: false,
            successfulReadOnlyObservation: true, imageAttachments: [image])
    }

    /// Exactly one post-image read. Grant expiry or topology changes do not
    /// recurse through cache inspection between the paired observations.
    private func readAfterScreenshot(
        observationGrant: String?,
        inspectResult: VisionCaptureMCPResult,
        inspectArguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> PreparedNavigation {
        guard let observationGrant else {
            if Self.isRecoverableInspectCacheBoundary(inspectResult, arguments: inspectArguments) {
                // Preserve the existing native-alert route. This exclusive
                // read is factual only in the image pair; observe can bind it.
                return try await describeSystemAlert(configuration: configuration, activity: activity)
            }
            guard !inspectResult.isError else {
                throw VisionCaptureAgentError.mcpOutcome(inspectResult.serverOutcome)
            }
            // A warm manifest has no read grant. Plain facts remain useful,
            // but its pre-image action handles must never be restored.
            return try await describeScreenAfterCacheValidationFailure(
                configuration: configuration, activity: activity)
        }
        let arguments = makeMCPArguments(
            request: "describe screen",
            parameters: ["observation_grant": .string(observationGrant)],
            configuration: configuration)
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        invalidateScreenObservation()
        let result = try await executeHostRequest(
            arguments, configuration: configuration, activity: activity)
        guard !result.isError else {
            throw VisionCaptureAgentError.mcpOutcome(result.serverOutcome)
        }
        try recordScreenObservation(from: result.value)
        let manifest = try Self.returnedAuthorityManifest(from: result.value)
        currentManifest = manifest
        return PreparedNavigation(
            result: result, arguments: arguments, manifest: manifest,
            systemAlert: nil, screenFacts: currentScreenFacts)
    }

    private static func imageReadState(
        image: VisionCaptureScreenshot.ObservationMetadata,
        read: VisionCaptureScreenshot.ObservationMetadata?,
        expectedDeviceID: String,
        readUnavailable: Bool
    ) -> String {
        guard !readUnavailable else { return "read_unavailable" }
        let observations = [image, read].compactMap { $0 }
        if observations.contains(where: {
            $0.deviceID.map { $0 != expectedDeviceID.lowercased() } ?? false
        }) { return "not_current" }
        if let imageGeneration = image.invalidationGeneration,
           let readGeneration = read?.invalidationGeneration,
           imageGeneration != readGeneration { return "not_current" }
        if observations.contains(where: {
            ($0.cacheAgeMilliseconds ?? 0) > 0
                || ($0.source != nil && $0.source != "live")
        }) { return "not_current" }
        // This permits decisions from the independently current text read,
        // not a claim that the two captures are atomic or visually identical.
        return "sequential"
    }

    private func refreshNavigation(
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity,
        desiredAction: JSONValue? = nil,
        allowsObservationRefresh: Bool = true
    ) async throws -> PreparedNavigation {
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        let inspectArguments = makeMCPArguments(
            request: "inspect cache",
            parameters: desiredAction.map { ["desired_action": $0] } ?? [:],
            configuration: configuration)
        let inspectResult = try await executeHostRequest(
            inspectArguments,
            configuration: configuration,
            activity: activity)
        if !allowsObservationRefresh, inspectResult.isError {
            invalidateScreenObservation()
            throw VisionCaptureAgentError.mcpOutcome(inspectResult.serverOutcome)
        }
        if Self.isRecoverableInspectCacheBoundary(
            inspectResult,
            arguments: inspectArguments
        ) {
            let alertRead = try await describeSystemAlert(
                configuration: configuration,
                activity: activity)
            if alertRead.systemAlert != nil {
                return alertRead
            }
            return try await describeScreenAfterCacheValidationFailure(
                configuration: configuration,
                activity: activity)
        }
        if !inspectResult.isError {
            try recordScreenObservation(from: inspectResult.value)
        } else {
            invalidateScreenObservation()
        }
        var manifest = try Self.returnedAuthorityManifest(
            from: inspectResult.value)
        currentManifest = manifest
        if inspectResult.isRecoverableColdMiss {
            return PreparedNavigation(
                result: inspectResult,
                arguments: inspectArguments,
                manifest: manifest,
                systemAlert: nil)
        }
        if !manifest.actions.isEmpty, currentScreenSignature == nil {
            guard allowsObservationRefresh else {
                currentManifest = AuthorityManifest()
                throw VisionCaptureAgentError.noProgress(
                    "The single observation refresh did not retain an exact screen identity. No input was sent and no further refresh was attempted.")
            }
            return try await establishFineScreenIdentityAndRefresh(
                configuration: configuration,
                activity: activity,
                desiredAction: desiredAction)
        }
        guard let grant = manifest.observationGrant,
              manifest.actions.isEmpty else {
            return PreparedNavigation(
                result: inspectResult,
                arguments: inspectArguments,
                manifest: manifest,
                systemAlert: nil,
                screenFacts: currentScreenFacts)
        }

        let describeArguments = makeMCPArguments(
            request: "describe screen",
            parameters: ["observation_grant": .string(grant)],
            configuration: configuration)
        currentManifest = AuthorityManifest()
        invalidateScreenObservation()
        let describeResult = try await executeHostRequest(
            describeArguments,
            configuration: configuration,
            activity: activity)
        if isRecoverableObservationTopologyChange(
            describeResult, arguments: describeArguments, configuration: configuration) {
            guard allowsObservationRefresh else {
                throw VisionCaptureAgentError.mcpOutcome(describeResult.serverOutcome)
            }
            // Retire every old handle and pending confirmation. The old proposal
            // is not carried into inspection or resumed after these reads.
            staleActionConfirmation = nil
            let observed = try await describeScreenAfterCacheValidationFailure(
                configuration: configuration, activity: activity)
            guard !observed.result.isError, currentScreenSignature != nil else {
                throw VisionCaptureAgentError.noProgress(
                    "The stale observation grant was retired, but the one plain read could not establish current screen facts. No input was sent or replayed.")
            }
            var refreshed = try await refreshNavigation(
                configuration: configuration, activity: activity,
                allowsObservationRefresh: false)
            refreshed.observationRefreshed = true
            return refreshed
        }
        if !describeResult.isError {
            try recordScreenObservation(from: describeResult.value)
        }
        manifest = try Self.returnedAuthorityManifest(
            from: describeResult.value)
        currentManifest = manifest
        return PreparedNavigation(
            result: describeResult,
            arguments: describeArguments,
            manifest: manifest,
            systemAlert: nil,
            screenFacts: currentScreenFacts)
    }

    /// A ready or revalidation cache manifest does not publish a structured
    /// view signature. VisionCapture has no public read operation that consumes
    /// its output-only screen capability, so discard those action handles,
    /// identify the screen with one plain read, then inspect once for fresh
    /// handles before returning or dispatching an action.
    private func establishFineScreenIdentityAndRefresh(
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity,
        desiredAction: JSONValue? = nil
    ) async throws -> PreparedNavigation {
        let arguments = makeMCPArguments(
            request: "describe screen",
            parameters: [:],
            configuration: configuration)
        currentManifest = AuthorityManifest()
        invalidateScreenObservation()
        let result = try await executeHostRequest(
            arguments,
            configuration: configuration,
            activity: activity)
        try recordScreenObservation(from: result.value)
        guard currentScreenSignature != nil else {
            throw VisionCaptureAgentError.noProgress(
                "VisionCapture returned usable cache actions without an exact fine screen identity, and the required plain describe response also omitted payload.view.signature_fine. Agent Mode stopped without dispatching an action.")
        }

        // A plain read invalidates the pre-read handles. Do not retain any
        // manifest returned alongside it; obtain one new cache block instead.
        currentManifest = AuthorityManifest()
        return try await refreshNavigation(
            configuration: configuration,
            activity: activity,
            desiredAction: desiredAction)
    }

    private func describeSystemAlert(
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> PreparedNavigation {
        invalidateScreenObservation()
        let arguments = makeMCPArguments(
            request: "describe system alert",
            parameters: [:],
            configuration: configuration,
            includeFlowSession: false)
        let result = try await executeHostRequest(
            arguments,
            configuration: configuration,
            usesFlowSession: false,
            activity: activity)
        let alert = try Self.returnedSystemAlert(from: result.value)
        currentManifest = AuthorityManifest()
        currentSystemAlert = alert
        if let uncertainAlertPress,
           alert?.contentDigest != uncertainAlertPress.digest {
            self.uncertainAlertPress = nil
        }
        return PreparedNavigation(
            result: result,
            arguments: arguments,
            manifest: AuthorityManifest(),
            systemAlert: alert)
    }

    /// A failed cache validation discarded its action evidence. When the
    /// dedicated alert read proves there is no native alert, make one plain
    /// read-only screen observation without any discarded grant or capability.
    private func describeScreenAfterCacheValidationFailure(
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> PreparedNavigation {
        let arguments = makeMCPArguments(
            request: "describe screen",
            parameters: [:],
            configuration: configuration)
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        invalidateScreenObservation()
        let result = try await executeHostRequest(
            arguments,
            configuration: configuration,
            activity: activity)
        if !result.isError {
            try recordScreenObservation(from: result.value)
        }
        // A plain read provides screen facts only. It does not restore the
        // cache evidence that the failed inspection invalidated.
        return PreparedNavigation(
            result: result,
            arguments: arguments,
            manifest: AuthorityManifest(),
            systemAlert: nil)
    }

    /// An expired unused action is discarded. Keep any spent stale-confirmation
    /// block and refresh only observations for the model's next decision.
    private func observeAfterActionAuthorizationExpiry(
        _ intent: NavigationIntent,
        failure: VisionCaptureServerOutcome,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        let observed = try await describeScreenAfterCacheValidationFailure(
            configuration: configuration, activity: activity)
        guard !observed.result.isError, currentScreenSignature != nil else {
            throw VisionCaptureAgentError.noProgress(
                "VisionCapture refused the expired action before dispatch, but the following plain read could not establish current screen facts. No action was replayed.")
        }
        let prepared = try await refreshNavigation(
            configuration: configuration, activity: activity)
        return try outcome(
            for: intent, result: prepared.result, arguments: prepared.arguments,
            manifest: prepared.manifest, systemAlert: prepared.systemAlert,
            progressed: false, observedScreenFacts: prepared.screenFacts,
            expiredBeforeDispatch: failure,
            observationRefreshed: prepared.observationRefreshed)
    }

    /// The submitted action keeps its failed proof. Refresh once for a new
    /// model choice without retaining the failed response's terminal handles.
    private func observeAfterDeliveredTransition(
        _ intent: NavigationIntent,
        failure: VisionCaptureServerOutcome,
        baselineScreenSignature: String?,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        staleActionConfirmation = nil
        let observed = try await describeScreenAfterCacheValidationFailure(
            configuration: configuration, activity: activity)
        guard !observed.result.isError,
              let baselineScreenSignature,
              let currentScreenSignature,
              currentScreenSignature != baselineScreenSignature else {
            throw VisionCaptureAgentError.noProgress(
                failure.description + ". The required plain read did not establish a different content signature for visible text and identifiers. Agent Mode stopped without replaying the delivered action.")
        }
        let prepared = try await refreshNavigation(
            configuration: configuration, activity: activity)
        return try outcome(
            for: intent, result: prepared.result, arguments: prepared.arguments,
            manifest: prepared.manifest, systemAlert: prepared.systemAlert,
            progressed: false, observedScreenFacts: prepared.screenFacts,
            deliveredFailure: failure,
            observationRefreshed: prepared.observationRefreshed)
    }

    /// The stale-capability contract requires one plain read before any new
    /// decision. It is intentionally not an automatic action retry.
    private func recoverFromStaleCacheAction(
        _ intent: NavigationIntent,
        baselineScreenSignature: String?,
        refreshChoices: Bool = false,
        isConfirmation: Bool = false,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        let arguments = makeMCPArguments(
            request: "describe screen",
            parameters: [:],
            configuration: configuration)
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        invalidateScreenObservation()
        let result = try await executeHostRequest(
            arguments,
            configuration: configuration,
            activity: activity)
        let observedScreenSignature = try Self.returnedScreenSignature(
            from: result.value)
        if !result.isError {
            try recordScreenObservation(from: result.value)
        }

        guard let baselineScreenSignature,
              let observedScreenSignature else {
            staleActionConfirmation = nil
            throw VisionCaptureAgentError.noProgress(
                "VisionCapture refused a stale cached action before dispatch, but the host could not compare the content signatures for visible text and identifiers before and after the required plain read. Agent Mode stopped instead of weakening the one-confirming-attempt budget. No input was submitted and the action was not retried.")
        }

        let screenChanged = baselineScreenSignature != observedScreenSignature
        if isConfirmation, !screenChanged {
            staleActionConfirmation = nil
            throw VisionCaptureAgentError.noProgress(
                "VisionCapture refused the confirming cached action before dispatch and the required plain read returned the same content signature for visible text and identifiers. Agent Mode stopped this action after its one confirming choice. No input was submitted.")
        }
        if screenChanged {
            staleActionConfirmation = nil
        } else {
            staleActionConfirmation = StaleActionConfirmation(
                intent: intent,
                screenSignature: observedScreenSignature)
        }
        if refreshChoices {
            // Inspect after the plain read with no old capability, grant, or
            // desired action. Only the model may choose from these fresh facts.
            let prepared = try await refreshNavigation(
                configuration: configuration, activity: activity)
            return try outcome(
                for: intent,
                result: prepared.result,
                arguments: prepared.arguments,
                manifest: prepared.manifest,
                systemAlert: prepared.systemAlert,
                progressed: false,
                observedScreenFacts: prepared.screenFacts,
                reobservedBeforeDispatch: true,
                observationRefreshed: prepared.observationRefreshed)
        }
        return try staleCacheOutcome(
            for: intent,
            readResult: result,
            screenChanged: screenChanged)
    }

    private func consumeStaleActionConfirmation(
        for intent: NavigationIntent
    ) throws -> StaleActionConfirmation? {
        if let staleActionConfirmation {
            self.staleActionConfirmation = nil
            try blockStaleAction(staleActionConfirmation)
            if staleActionConfirmation.intent == intent {
                guard let currentScreenSignature else {
                    throw VisionCaptureAgentError.navigationUnavailable(
                        "The confirming action was not sent because the host no longer has the content signature for visible text and identifiers from the required plain read. Choose observe and make a new action decision.")
                }
                guard currentScreenSignature == staleActionConfirmation.screenSignature else {
                    throw VisionCaptureAgentError.navigationUnavailable(
                        "The confirming action was not sent because the content signature for visible text and identifiers changed after the stale-capability read. Choose observe and make a new action decision from the current screen.")
                }
                return staleActionConfirmation
            }
        }

        try rejectBlockedStaleAction(intent)
        blockedStaleActions = blockedStaleActions.filter { $0.intent != intent }
        return nil
    }

    /// Read-only eligibility check, also used when publishing model choices.
    /// The one confirming choice is consumed only by the dispatch admission path.
    private func rejectBlockedStaleAction(_ intent: NavigationIntent) throws {
        let matchingBlocks = blockedStaleActions.filter {
            $0.intent == intent
        }
        guard !matchingBlocks.isEmpty else { return }
        guard let currentScreenSignature else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "This action was not sent because its one stale-capability confirmation choice was already spent or declined and the current screen has not been freshly identified. Choose observe and make a new action decision.")
        }
        guard !matchingBlocks.contains(where: {
            $0.screenSignature == currentScreenSignature
        }) else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "This action was not sent because its one stale-capability confirmation choice with the same content signature was already spent or declined. Choose a different current action, choose screenshot if pixels could clarify a different safe next step, or report the cache-binding blocker. A screenshot does not restore this refused action.")
        }
    }

    private func blockStaleAction(
        _ action: StaleActionConfirmation
    ) throws {
        guard blockedStaleActions.contains(action)
                || blockedStaleActions.count < Self.staleActionBlockLimit else {
            throw VisionCaptureAgentError.noProgress(
                "Agent Mode stopped because 24 different stale cached actions were already retired on the current screen. No additional action was sent.")
        }
        blockedStaleActions.insert(action)
    }

    private func retirePendingConfirmation() throws {
        guard let confirmation = staleActionConfirmation else { return }
        try blockStaleAction(confirmation)
        staleActionConfirmation = nil
    }

    private func pressSystemAlertButton(
        _ intent: NavigationIntent,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        guard let alert = currentSystemAlert,
              let selector = intent.selector else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "The system-alert press was not sent because no current host-observed alert and exact button are available. Choose observe first.")
        }
        let matching = alert.buttons.filter {
            $0.label == selector && $0.enabled && $0.visible
        }
        guard matching.count == 1 else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "The system-alert press was not sent because its label did not match exactly one visible, enabled button in the latest alert read.")
        }
        if let uncertainAlertPress,
           uncertainAlertPress.digest == alert.contentDigest {
            throw VisionCaptureAgentError.navigationUnavailable(
                "The system-alert press was not sent because delivery of the previous button press was unknown. Only read-only observation is allowed while this same alert remains; never retry or replace that press.")
        }

        let arguments = makeMCPArguments(
            request: "press system alert button",
            parameters: [
                "system_alert_button": .string(selector),
                "system_alert_digest": .string(alert.contentDigest),
            ],
            configuration: configuration,
            includeFlowSession: false)
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        invalidateScreenObservation()
        let result = try await executeHostRequest(
            arguments,
            configuration: configuration,
            usesFlowSession: false,
            activity: activity)
        let deliveryUnknown = result.isSystemAlertDeliveryUnknown
        if deliveryUnknown {
            uncertainAlertPress = (alert.contentDigest, selector)
        }

        let observed = try await describeSystemAlert(
            configuration: configuration,
            activity: activity)
        return try Self.systemAlertPressOutcome(
            intent: intent,
            pressResult: result,
            observed: observed,
            deliveryUnknown: deliveryUnknown)
    }

    private func reusableReadyManifest(
        for intent: NavigationIntent,
        offer: ReadyActionOffer?,
        configuration: VisionCaptureAgentConfiguration
    ) -> AuthorityManifest? {
        guard let offer, offer.manifest.state == "ready",
              offer.manifest.observationGrant == nil,
              offer.targetKey == configuration.targetKey,
              offer.session == committedSessionIdentity,
              offer.screenSignature == currentScreenSignature,
              offer.manifest == currentManifest, currentSystemAlert == nil,
              isEligibleOfferedAction(intent) else { return nil }
        var offered = offer.manifest
        offered.actions = eligibleOfferedActions(offered.actions)
        guard let action = try? Self.uniquePublishedAction(for: intent, in: offered),
              action.actionCapability != nil, action.revalidationCapability == nil else { return nil }
        return offer.manifest
    }

    private func makeActionArguments(
        intent: NavigationIntent,
        action: PublishedAction,
        manifest: AuthorityManifest,
        configuration: VisionCaptureAgentConfiguration
    ) throws -> JSONValue {
        if let capability = action.actionCapability {
            let request = action.action == "tap"
                ? "tap cached action" : "execute cached action"
            return makeMCPArguments(
                request: request,
                parameters: ["action_capability": .string(capability)],
                configuration: configuration)
        }
        if let capability = action.revalidationCapability {
            return makeMCPArguments(
                request: "revalidate cached action",
                parameters: ["revalidation_capability": .string(capability)],
                configuration: configuration)
        }
        guard let grant = manifest.observationGrant else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "The selected action has no current host-held execution evidence. Observe again before choosing it.")
        }
        if action.action == "tap" {
            return makeMCPArguments(
                request: "tap \(action.selector)",
                parameters: ["observation_grant": .string(grant)],
                configuration: configuration)
        }
        guard let desiredState = intent.desiredState else {
            throw VisionCaptureAgentError.malformedCall(
                "set_boolean requires desired_state")
        }
        return makeMCPArguments(
            request: "execute observed action",
            parameters: [
                "observation_grant": .string(grant),
                "desired_action": .object([
                    "action": .string("set_boolean"),
                    "selector": .string(action.selector),
                    "role": .string(action.role),
                    "desired_state": .bool(desiredState),
                ]),
            ],
            configuration: configuration)
    }

    private func makeMCPArguments(
        request: String,
        parameters: [String: JSONValue],
        configuration: VisionCaptureAgentConfiguration,
        includeFlowSession: Bool = true
    ) -> JSONValue {
        var lockedParameters = parameters
        lockedParameters["udid"] = .string(configuration.simulatorUDID)
        var arguments: [String: JSONValue] = [
            "request": .string(request),
            "bundle_id": .string(configuration.bundleIdentifier),
            "parameters": .object(lockedParameters),
        ]
        if includeFlowSession, let committedSessionIdentity {
            arguments["session_id"] = .string(committedSessionIdentity.id)
            arguments["session_kind"] = .string(committedSessionIdentity.kind)
        }
        return .object(arguments)
    }

    private func executeHostRequest(
        _ arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration,
        usesFlowSession: Bool = true,
        activityID: UUID = UUID(),
        activity: @escaping Activity
    ) async throws -> VisionCaptureMCPResult {
        checkingLocalProposal = false
        checkpointRequestIDs.append(activityID)
        let activityStart = ContinuousClock.now
        await activity(.outgoingRequest(
            id: activityID,
            arguments: arguments))
        let trace = AgentInferenceTrace.shared
        var traceStep: AgentInferenceTrace.Step?
        if let trace { traceStep = await trace.currentStep() }
        let client: VisionCaptureMCPClient
        do {
            if let mcpClient {
                client = mcpClient
            } else {
                let created = try VisionCaptureMCPClient(
                    port: VisionCaptureAgentProfile.mcpPort)
                try await created.prepare()
                mcpClient = created
                client = created
            }
        } catch is CancellationError {
            await activity(.requestStatus(
                id: activityID,
                status: .cancelled,
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw CancellationError()
        } catch {
            await activity(.requestStatus(
                id: activityID,
                status: .localFailure(reason: "Connection or protocol error: \(String(describing: error).prefix(300))"),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw VisionCaptureAgentError.mcpUnavailable(String(describing: error))
        }

        let result: VisionCaptureMCPResult
        do {
            result = try await client.execute(arguments: arguments)
        } catch is CancellationError {
            await activity(.requestStatus(
                id: activityID,
                status: .cancelled,
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw CancellationError()
        } catch {
            await activity(.requestStatus(
                id: activityID,
                status: .localFailure(reason: "Request or protocol error: \(String(describing: error).prefix(300))"),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw error
        }

        await activity(.incomingResponse(id: activityID, excerpt: result.displayExcerpt))
        if let trace, result.isError {
            let operation: String?
            if case .object(let request) = arguments, case .string(let name)? = request["request"] {
                operation = name
            } else { operation = nil }
            await trace.mcpFailure(
                step: traceStep, requestID: activityID, operation: operation, result: result)
        }
        let launchTimeout = Self.isolatedLaunchTimeout(
            result, arguments: arguments, configuration: configuration)
        do {
            try Self.validateReturnedIdentity(
                in: result.value,
                configuration: configuration,
                refusalCode: result.refusalCode,
                permittedLaunchTimeout: launchTimeout)
            if usesFlowSession, launchTimeout == nil {
                try adoptReturnedSessionIdentity(
                    from: result.value,
                    isSuccessful: !result.isError,
                    request: arguments)
            }
        } catch let error as VisionCaptureAgentError {
            let code: String
            switch error {
            case .returnedIdentityMismatch:
                code = "TARGET_IDENTITY_MISMATCH"
                discardReturnedExecutionEvidence()
            case .sessionIdentityMismatch:
                code = "SESSION_IDENTITY_MISMATCH"
            default:
                code = "MALFORMED_MCP_RESULT"
            }
            await activity(.requestStatus(
                id: activityID,
                status: .localFailure(reason: "Response validation failed (\(code))"),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw error
        }

        let serverOutcome = result.serverOutcome
        if isPermittedDeliveredTransition(
            result, arguments: arguments, configuration: configuration) {
            await activity(.requestStatus(
                id: activityID, status: .serverOutcome(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            return result
        }
        if launchTimeout != nil {
            // The launch attempt ended with unknown delivery, not a pre-dispatch
            // refusal. Its caller may perform only the contract's plain read.
            await activity(.requestStatus(
                id: activityID,
                status: .serverOutcome(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            return result
        }

        if result.isRecoverableColdMiss
            || Self.isRecoverableInspectCacheBoundary(
                result,
                arguments: arguments)
            || result.isGuardedTargetRejectedBeforeSubmission
            || isRecoverableRevalidationLayoutChange(
                result, arguments: arguments, configuration: configuration)
            || isRecoverableActionAuthorizationExpiry(
                result, arguments: arguments, configuration: configuration)
            || isRecoverableObservationTopologyChange(
                result, arguments: arguments, configuration: configuration)
            || Self.isRecoverableStaleCacheAction(
                result,
                arguments: arguments) {
            await activity(.requestStatus(
                id: activityID,
                status: .recoverablePreDispatchRefusal(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            return result
        }
        if result.isSystemAlertDeliveryUnknown {
            await activity(.requestStatus(
                id: activityID,
                status: .serverOutcome(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            return result
        }
        if result.isError {
            let code = result.refusalCode ?? "MCP_TOOL_REFUSED"
            await activity(.requestStatus(
                id: activityID,
                status: .serverOutcome(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            if Self.isSystemInteractionCode(code) {
                throw VisionCaptureAgentError.unsupportedSystemInteraction(code, outcome: serverOutcome)
            }
            throw VisionCaptureAgentError.mcpOutcome(serverOutcome)
        }

        committedTargetKey = configuration.targetKey
        await activity(.requestStatus(
            id: activityID,
            status: serverOutcome.verdict == nil ? .succeeded : .serverOutcome(serverOutcome),
            elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
        return result
    }

    private static func isRecoverableInspectCacheBoundary(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue
    ) -> Bool {
        guard result.isError,
              result.refusalCode == "CACHE_LIVE_VALIDATION_FAILED",
              result.dispatchAttempted != true,
              !result.hasConflictingDispatchAttemptEvidence,
              case .object(let object) = arguments,
              case .string(let request)? = object["request"] else {
            return false
        }
        return request.trimmingCharacters(
            in: .whitespacesAndNewlines).lowercased() == "inspect cache"
    }

    private static func isRecoverableStaleCacheAction(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue
    ) -> Bool {
        guard result.isStaleActionCapabilityBeforeDispatch,
              case .object(let object) = arguments,
              case .string(let rawRequest)? = object["request"],
              case .object(let parameters)? = object["parameters"],
              case .string(let capability)? = parameters["action_capability"],
              !capability.isEmpty else {
            return false
        }
        let request = rawRequest.trimmingCharacters(
            in: .whitespacesAndNewlines).lowercased()
        return request == "tap cached action"
            || request == "execute cached action"
    }

    private func isRecoverableRevalidationLayoutChange(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        result.isSourceLayoutChangedBeforeRevalidation
            && matchesCommittedRevalidationRequest(arguments, configuration: configuration)
    }

    private func isRecoverableActionAuthorizationExpiry(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        result.isActionAuthorizationExpiredBeforeDispatch
            && matchesCommittedWarmTapRequest(arguments, configuration: configuration)
    }

    private func isPermittedDeliveredTransition(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        result.isDeliveredTransitionContinuation
            && (matchesCommittedRevalidationRequest(arguments, configuration: configuration)
                || matchesCommittedWarmTapRequest(arguments, configuration: configuration))
    }

    private func matchesCommittedWarmTapRequest(
        _ arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard committedTargetKey == configuration.targetKey,
              currentSystemAlert == nil, uncertainAlertPress == nil,
              let request = arguments.objectValue,
              request["request"] == .string("tap cached action"),
              request["bundle_id"] == .string(configuration.bundleIdentifier),
              request["session_id"] == committedSessionIdentity.map({ .string($0.id) }),
              request["session_kind"] == committedSessionIdentity.map({ .string($0.kind) }),
              let parameters = request["parameters"]?.objectValue,
              parameters["udid"] == .string(configuration.simulatorUDID),
              case .string(let capability)? = parameters["action_capability"],
              !capability.isEmpty,
              Set(parameters.keys) == ["udid", "action_capability"] else { return false }
        return true
    }

    private func matchesCommittedRevalidationRequest(
        _ arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard committedTargetKey == configuration.targetKey,
              currentSystemAlert == nil, uncertainAlertPress == nil,
              let request = arguments.objectValue,
              request["request"] == .string("revalidate cached action"),
              request["bundle_id"] == .string(configuration.bundleIdentifier),
              request["session_id"] == committedSessionIdentity.map({ .string($0.id) }),
              request["session_kind"] == committedSessionIdentity.map({ .string($0.kind) }),
              let parameters = request["parameters"]?.objectValue,
              parameters["udid"] == .string(configuration.simulatorUDID),
              case .string(let capability)? = parameters["revalidation_capability"],
              !capability.isEmpty,
              Set(parameters.keys) == ["udid", "revalidation_capability"] else { return false }
        return true
    }

    private func isRecoverableObservationTopologyChange(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard result.isObservationTopologyRefreshBeforeDispatch,
              result.isError, result.dispatchAttempted == false,
              !result.hasConflictingDispatchAttemptEvidence,
              committedTargetKey == configuration.targetKey,
              currentSystemAlert == nil, uncertainAlertPress == nil,
              let session = committedSessionIdentity, session.kind == "flow",
              let request = arguments.objectValue,
              request["request"] == .string("describe screen"),
              request["bundle_id"] == .string(configuration.bundleIdentifier),
              request["session_id"] == .string(session.id),
              request["session_kind"] == .string(session.kind),
              let parameters = request["parameters"]?.objectValue,
              Set(parameters.keys) == ["udid", "observation_grant"],
              parameters["udid"] == .string(configuration.simulatorUDID),
              case .string(let grant)? = parameters["observation_grant"], !grant.isEmpty
        else { return false }
        return true
    }

    private static func elapsedSeconds(
        since start: ContinuousClock.Instant
    ) -> Double {
        let components = start.duration(to: .now).components
        return max(
            0,
            Double(components.seconds)
                + Double(components.attoseconds) / 1_000_000_000_000_000_000)
    }

    private static func instructions(
        configuration: VisionCaptureAgentConfiguration
    ) -> String {
        let targetState = (try? configuration.validate()) == nil
            ? "The host target is not configured. Normal chat still works. If navigation is needed, make one navigation proposal and follow the host's configuration result."
            : "The host target is configured. Never ask for, invent, or include application or device identifiers."
        return """
        You are in the ordinary TurboFieldfare chat with one optional high-level application-navigation tool.
        For normal conversation, answer normally and do not call a tool. Use visioncapture_navigate only when
        the user asks to inspect or drive the configured application.

        \(targetState)

        Use one tool call per assistant turn. Each result has schema_version, observation,
        last_action, allowed_next, facts, and choices. Choose an action from allowed_next.
        For tap, set_boolean, or type, copy a target ID from the latest choices and use only
        its listed operation. IDs expire with the next result. Never substitute a label, an
        old ID, coordinates, or an invented target. The host privately resolves exact targets,
        sessions, execution evidence, validation, and safe recovery.

        For swipe, choose a finger-movement direction from can_swipe. Send only action and
        direction. The host submits one gesture and reads fresh facts before your next decision.
        A screen change alone does not verify the swipe's intended effect.

        Before the first observation, choose launch, observe, or screenshot as needed.
        After launch, choose observe. When current facts and choices are unchanged, do not
        repeat observe as a waiting loop. Choose a current action, request a screenshot if
        it could clarify a different safe step, answer, or report the factual blocker.
        Do not tap an already-selected option merely to select it again. It may deselect it.
        For set_boolean, use an offered allowed_desired_states value. Typing inserts the exact
        supplied text at the field's cursor or selection without clearing existing content.
        Repeating text may duplicate it. A null or absent value is unknown, not an empty field.

        Request a screenshot when a form, selection, completion, or navigation is unclear
        from text. Image support requires the vision companion. A screenshot is read-only,
        retires existing choices, and does not prove a previous action succeeded. The host
        then reads accessibility facts and offers new targets when permitted. Image and read are
        sequential; visual agreement is unverified. Use current text choices without another
        observe when they are sufficient. Actual positions come from the later read, use 0–1000
        from the screen's top-left, and never grant pointer input.
        If a control is unlabeled or ambiguous, use its actual role and position with image
        evidence or report the ambiguity. Do not guess a target.

        last_action reports the previous action separately from the current observation.
        Preserve unknown delivery, failure, and inconclusive outcomes. Never automatically
        replay a refused or uncertain action. A choice with confirmation_attempts: 1 is one
        permitted new decision following a stale refusal before dispatch. It expires with the
        next packet and cannot be restored by a correction. Follow the current recovery guidance.
        A journey_hint reports a return to earlier observed content. Such a return can be valid,
        including adding another item, and does not establish a lack of task progress.

        Native system-alert buttons are tap choices with role system_alert_button. Choose a
        current ID. The host uses the existing guarded alert route once and then reads again.
        If no supported choice serves the goal, explain that factual limitation. A screenshot
        cannot bypass a refusal or enable unsupported input.

        Follow the user's goal and current observed facts. Do not assume any app-specific
        route or completion rule. Verified typing or selection does not prove a form was saved.
        Use current evidence to assess remaining goals without upgrading earlier action proof.
        Distinguish verified, failed, inconclusive, and unknown outcomes in the final answer.
        """
    }

    private static func preflight(_ calls: [AppToolCall]) throws -> AppToolCall {
        guard calls.count == 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "only one visioncapture_navigate call is allowed per assistant turn")
        }
        return calls[0]
    }

    private func navigationIntent(
        from call: AppToolCall,
        configuration: VisionCaptureAgentConfiguration
    ) throws -> NavigationIntent {
        resolvedJourneyLabel = nil
        guard call.name == VisionCaptureToolDefinitions.navigateName,
              case .object(let object) = call.arguments,
              case .string(let name)? = object["action"],
              let operation = NavigationOperation(rawValue: name) else {
            throw VisionCaptureAgentError.malformedCall(
                "Choose one visioncapture_navigate action from allowed_next.")
        }
        let supplied = Set(object.keys)
        let required: Set<String>
        switch operation {
        case .tap: required = ["action", "target"]
        case .setBoolean: required = ["action", "target", "desired_state"]
        case .type: required = ["action", "target", "text"]
        case .swipe: required = ["action", "direction"]
        case .launch, .observe, .screenshot, .back: required = ["action"]
        }
        guard supplied == required else {
            throw VisionCaptureAgentError.malformedCall(
                "This action requires exactly: \(required.sorted().joined(separator: ", ")).")
        }
        guard permittedNextOperations?.contains(operation)
                ?? [.launch, .observe, .screenshot].contains(operation) else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "This action is not currently permitted. Choose from allowed_next.")
        }
        if requiresReadOnlyRecovery || uncertainAlertPress != nil {
            guard operation == .observe || operation == .screenshot else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "Only read-only observation is currently permitted after uncertain delivery.")
            }
        }
        switch operation {
        case .swipe:
            guard case .string(let rawDirection)? = object["direction"],
                  let direction = SwipeDirection(rawValue: rawDirection) else {
                throw VisionCaptureAgentError.malformedCall(
                    "Swipe direction must be exactly up, down, left, or right.")
            }
            let intent = NavigationIntent(operation: .swipe, selector: nil,
                selectorKind: nil, role: nil, desiredState: nil, text: nil, direction: direction)
            guard isEligibleOfferedAction(intent) else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "This swipe direction is not currently offered. Choose from can_swipe.")
            }
            return intent
        case .launch, .observe, .screenshot, .back:
            return NavigationIntent(operation: operation, selector: nil,
                selectorKind: nil, role: nil, desiredState: nil, text: nil)
        case .tap, .setBoolean, .type:
            break
        }
        guard case .string(let target)? = object["target"],
              let binding = currentChoiceBindings[target],
              binding.observation == observationGeneration,
              binding.targetKey == configuration.targetKey,
              binding.session == committedSessionIdentity,
              binding.screenSignature == currentScreenSignature,
              bindingIsCurrent(binding) else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "The target ID is expired, ambiguous, or unavailable. Choose a current choice. Old IDs cannot be restored.")
        }
        guard binding.operation == operation else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "The proposed target supports \(binding.operation.rawValue). Requested \(operation.rawValue) is not supported.")
        }
        let desiredState: Bool?
        if operation == .setBoolean {
            guard case .bool(let desired)? = object["desired_state"],
                  binding.allowedStates.contains(desired) else {
                throw VisionCaptureAgentError.malformedCall(
                    "Choose a desired_state offered by this Boolean choice.")
            }
            desiredState = desired
        } else {
            desiredState = nil
        }
        let text: String?
        if operation == .type {
            guard case .string(let raw)? = object["text"], !raw.isEmpty else {
                throw VisionCaptureAgentError.malformedCall("Typing requires nonempty text.")
            }
            // The user text is insertion content, not a normalized selector.
            text = raw
        } else {
            text = nil
        }
        resolvedJourneyLabel = binding.displayLabel
        return NavigationIntent(operation: operation, selector: binding.selector,
            selectorKind: binding.selectorKind, role: binding.role,
            desiredState: desiredState, text: text)
    }

    private func semanticProposalForRepeatCheck(_ call: AppToolCall) -> AppToolCall {
        guard case .object(var arguments) = call.arguments,
              case .string(let id)? = arguments["target"],
              let binding = currentChoiceBindings[id],
              binding.observation == observationGeneration,
              binding.targetKey == committedTargetKey,
              binding.session == committedSessionIdentity,
              binding.screenSignature == currentScreenSignature,
              bindingIsCurrent(binding) else { return call }
        var target: [String: JSONValue] = [
            "selector": .string(binding.selector), "role": .string(binding.role),
            "operation": .string(binding.operation.rawValue),
            "allowed_states": .array(binding.allowedStates.map(JSONValue.bool)),
        ]
        if let kind = binding.selectorKind { target["selector_kind"] = .string(kind) }
        arguments["target"] = .object(target)
        return AppToolCall(id: call.id, name: call.name, arguments: .object(arguments))
    }

    private func bindingIsCurrent(_ binding: ChoiceBinding) -> Bool {
        switch binding.route {
        case .published(let action, let grant):
            return currentManifest.observationGrant == grant
                && currentManifest.actions.contains(action)
        case .candidate(let candidate):
            return offeredTapCandidates?.signature == currentScreenSignature
                && offeredTapCandidates?.candidates.contains(where: {
                    $0.selector.utf8.elementsEqual(candidate.selector.utf8)
                        && $0.role == candidate.role
                }) == true
        case .editable(let field):
            return currentEditableFields.contains(field)
        case .alert(let label, let digest):
            guard uncertainAlertPress == nil, let alert = currentSystemAlert,
                  alert.contentDigest == digest else { return false }
            let matches = alert.buttons.filter { $0.label.utf8.elementsEqual(label.utf8) }
            return matches.count == 1 && matches[0].enabled && matches[0].visible
        case .confirmation(let confirmation):
            return staleActionConfirmation == confirmation
                && currentScreenSignature == confirmation.screenSignature
        }
    }

    private func bindChoice(
        _ object: [String: JSONValue],
        isField: Bool,
        isConfirmation: Bool,
        configuration: VisionCaptureAgentConfiguration
    ) throws -> (choice: JSONValue, semantic: JSONValue)? {
        guard case .string(let selector)? = object["selector"],
              case .string(let role)? = object["role"],
              let operation = isField ? NavigationOperation.type
                : object["action"].flatMap({ value -> NavigationOperation? in
                    guard case .string(let name) = value else { return nil }
                    return NavigationOperation(rawValue: name)
                }) else { return nil }
        let kind: String?
        if case .string(let value)? = object["selector_kind"] { kind = value } else { kind = nil }
        let states: [Bool]
        if operation == .setBoolean {
            if case .bool(let state)? = object["desired_state"] { states = [state] }
            else { states = [false, true] }
        } else { states = [] }
        let route: ChoiceRoute
        if isConfirmation {
            guard let confirmation = staleActionConfirmation,
                  confirmation.intent.selector?.utf8.elementsEqual(selector.utf8) == true,
                  confirmation.intent.role == role,
                  confirmation.intent.operation == operation else { return nil }
            route = .confirmation(confirmation)
        } else if role == "system_alert_button" {
            guard let alert = currentSystemAlert else { return nil }
            route = .alert(label: selector, digest: alert.contentDigest)
        } else if isField {
            let matches = currentEditableFields.filter {
                $0.selector.utf8.elementsEqual(selector.utf8) && $0.selectorKind == kind && $0.role == role
            }
            guard matches.count == 1 else { return nil }
            route = .editable(matches[0])
        } else if object["requires_validation"] == .bool(true) {
            guard let offered = offeredTapCandidates,
                  offered.signature == currentScreenSignature else { return nil }
            let matches = offered.candidates.filter {
                $0.selector.utf8.elementsEqual(selector.utf8) && $0.role == role
            }
            guard matches.count == 1 else { return nil }
            route = .candidate(matches[0])
        } else {
            let matches = currentManifest.actions.filter {
                $0.selector.utf8.elementsEqual(selector.utf8) && $0.role == role
                    && $0.action == operation.rawValue
                    && ($0.desiredState == nil || states == [$0.desiredState!])
            }
            guard matches.count == 1 else { return nil }
            route = .published(matches[0], grant: currentManifest.observationGrant)
        }
        var choice = readableChoiceFacts(object, isField: isField)
        let displayLabel: String?
        if case .string(let label)? = choice["label"] { displayLabel = label } else { displayLabel = nil }
        let binding = ChoiceBinding(
            observation: observationGeneration, targetKey: configuration.targetKey,
            session: committedSessionIdentity, screenSignature: currentScreenSignature,
            operation: operation, selector: selector, selectorKind: kind,
            role: role, displayLabel: displayLabel, allowedStates: states, route: route)
        guard bindingIsCurrent(binding),
              role == "system_alert_button" || currentScreenSignature != nil else { return nil }
        let eligibilityStates: [Bool?] = states.isEmpty ? [nil] : states.map(Optional.some)
        guard eligibilityStates.allSatisfy({ state in
            isEligibleOfferedAction(NavigationIntent(operation: operation, selector: selector,
                selectorKind: kind, role: role, desiredState: state, text: nil))
        }) else { return nil }
        guard nextChoiceNumber < UInt64.max else {
            throw VisionCaptureAgentError.noProgress("The conversation exhausted its choice IDs. Start a new chat.")
        }
        nextChoiceNumber += 1
        let id = "c\(nextChoiceNumber)"
        choice["id"] = .string(id)
        choice["operations"] = .array([.string(operation.rawValue)])
        if role == "system_alert_button" { choice["enabled"] = .bool(true) }
        if !states.isEmpty { choice["allowed_desired_states"] = .array(states.map(JSONValue.bool)) }
        if case .confirmation = route { choice["confirmation_attempts"] = .integer(1) }

        // Exact private identities participate in repeat detection. No backend
        // selector, capability, session, or digest is published to the model.
        var semantic = choice
        semantic.removeValue(forKey: "id")
        semantic["selector"] = .string(selector)
        if let kind { semantic["selector_kind"] = .string(kind) }
        currentChoiceBindings[id] = binding
        return (.object(choice), .object(semantic))
    }

    private func readableChoiceFacts(
        _ object: [String: JSONValue], isField: Bool
    ) -> [String: JSONValue] {
        var facts: [String: JSONValue] = [:]
        for key in ["role", "enabled", "selected", "value", "position", "current_state"] {
            if let value = object[key] { facts[key] = value }
        }
        if case .string(let display)? = object["label"], !display.isEmpty, !display.hasPrefix("__vc") {
            facts["label"] = .string(display)
        } else if case .string(let selector)? = object["selector"],
                  case .string(let role)? = object["role"] {
            let kind: String?
            if case .string(let value)? = object["selector_kind"] { kind = value } else { kind = nil }
            if (role == "system_alert_button" || kind == "placeholder"), !selector.hasPrefix("__vc") {
                facts["label"] = .string(selector)
            } else if let label = currentScreenFacts?.readableLabel(
                selector: selector, role: role, selectorKind: kind) {
                facts["label"] = .string(label)
            }
        }
        if isField, facts["value"] == nil { facts["value"] = .null }
        return facts
    }

    private func decisionPacket(
        from content: String,
        call: AppToolCall,
        configuration: VisionCaptureAgentConfiguration,
        images: [AppImageAttachment]
    ) throws -> DecisionPacket {
        guard case .object(let body) = try JSONDecoder().decode(
            JSONValue.self, from: Data(content.utf8)) else {
            throw VisionCaptureAgentError.malformedCall("The host could not encode the current decision facts.")
        }
        currentChoiceBindings.removeAll(keepingCapacity: true)
        guard observationGeneration < UInt64.max else {
            throw VisionCaptureAgentError.noProgress("The conversation exhausted its observation IDs. Start a new chat.")
        }
        observationGeneration += 1
        let proposedOperation = body["operation"] ?? call.arguments.objectValue?["action"]
        let operation: JSONValue
        if case .string(let name)? = proposedOperation, NavigationOperation(rawValue: name) != nil {
            operation = .string(name)
        } else { operation = .string("unknown") }
        let outcome = body["outcome"] ?? .string("unknown")
        let proofVerdict = body["proof"]?.objectValue?["verdict"]
        let unknownDelivery = body["delivery_unknown"] == .bool(true)
            || outcome == .string("delivery_unknown_reobserved")
            || outcome == .string("unknown_reobserved")
        if unknownDelivery {
            requiresReadOnlyRecovery = true
        } else if let pairing = body["image_observation"]?.objectValue,
                  pairing["state"] != .string("sequential") {
            requiresReadOnlyRecovery = true
        } else if operation == .string("observe"), outcome == .string("succeeded"),
                  currentScreenSignature != nil || currentSystemAlert != nil {
            requiresReadOnlyRecovery = false
        }
        let readOnlyRequired = unknownDelivery || requiresReadOnlyRecovery || uncertainAlertPress != nil
        let recovery = body["stale_recovery"]?.objectValue
        let confirmationAllowed = !readOnlyRequired
            && recovery?["screen_changed"] == .bool(false)
            && recovery?["remaining_confirmation_attempts"] == .integer(1)
            && staleActionConfirmation != nil
        let unavailable = body["observation_outcome"] == .string("unavailable")
        var choices: [JSONValue] = []
        var semanticChoices: [JSONValue] = []
        var unavailableChoices: [JSONValue] = []
        for (values, isField) in [(Self.array(body["available_actions"]), false),
                                 (Self.array(body["available_text_fields"]), true)] {
            for value in values {
                guard case .object(let object) = value else { continue }
                if !readOnlyRequired, !unavailable, recovery == nil,
                   let bound = try bindChoice(object, isField: isField, isConfirmation: false,
                       configuration: configuration) {
                    choices.append(bound.choice)
                    semanticChoices.append(bound.semantic)
                } else {
                    var facts = readableChoiceFacts(object, isField: isField)
                    facts["availability"] = .string("not_offered")
                    unavailableChoices.append(.object(facts))
                }
            }
        }
        if !readOnlyRequired, !unavailable {
            if confirmationAllowed, let object = recovery?["confirming_action"]?.objectValue,
               let bound = try bindChoice(object, isField: false, isConfirmation: true,
                   configuration: configuration) {
                choices.append(bound.choice)
                semanticChoices.append(bound.semantic)
            }
        }
        var allowed: Set<NavigationOperation> = [.observe]
        var swipeDirections: [SwipeDirection] = []
        if AppVisionPackInstallationProbe.status(at: configuration.modelDirectory) == .complete {
            allowed.insert(.screenshot)
        }
        for binding in currentChoiceBindings.values { allowed.insert(binding.operation) }
        if !readOnlyRequired, !unavailable, recovery == nil, currentSystemAlert == nil,
           currentScreenSignature != nil, committedSessionIdentity?.kind == "flow",
           isEligibleOfferedAction(NavigationIntent(operation: .back, selector: nil,
               selectorKind: nil, role: nil, desiredState: nil, text: nil)) {
            allowed.insert(.back)
        }
        if !readOnlyRequired, !unavailable, recovery == nil, currentSystemAlert == nil,
           currentScreenSignature != nil, committedSessionIdentity?.kind == "flow" {
            swipeDirections = SwipeDirection.allCases.filter { direction in
                isEligibleOfferedAction(NavigationIntent(operation: .swipe, selector: nil,
                    selectorKind: nil, role: nil, desiredState: nil, text: nil, direction: direction))
            }
            if !swipeDirections.isEmpty { allowed.insert(.swipe) }
        }
        // Launch can mutate application state, so it is unavailable during
        // uncertain delivery and while a current app observation exists.
        if !readOnlyRequired, currentScreenSignature == nil, currentSystemAlert == nil,
           images.isEmpty, recovery == nil, body["outcome"] == .string("not_sent") {
            allowed.insert(.launch)
        }
        if (try? configuration.validate()) == nil || (
            committedTargetKey != nil && committedTargetKey != configuration.targetKey) {
            allowed = []
            swipeDirections = []
            choices = []
            semanticChoices = []
            currentChoiceBindings.removeAll(keepingCapacity: true)
        }
        permittedNextOperations = allowed
        var observation: [String: JSONValue] = [
            "id": .string("o\(observationGeneration)"),
            "state": .string(!images.isEmpty && body["image_observation"]?.objectValue?["state"] != .string("sequential") ? "image_only"
                : unavailable ? "unavailable"
                : currentScreenSignature != nil || currentSystemAlert != nil ? "current" : "unavailable"),
        ]
        if let pairing = body["image_observation"] { observation["image_relationship"] = pairing }
        if let state = body["observation_outcome"] { observation["read_result"] = state }
        if let current = body["current_observation"]?.objectValue?["requested_foreground_app_proven"] {
            observation["requested_app_present"] = current
        }
        var facts: [JSONValue] = unavailableChoices
        if case .string(let summary)? = body["screen_summary"] {
            facts += summary.split(separator: "\n").map { .string(String($0)) }
        }
        for key in ["navigation_fact"] {
            if let fact = body[key] { facts.append(fact) }
        }
        if let alert = body["system_alert"]?.objectValue {
            var alertFacts = alert
            let offeredLabels = Set(choices.compactMap { $0.objectValue?["label"] }.compactMap { value -> String? in
                guard case .string(let text) = value else { return nil }; return text
            })
            if case .array(let buttons)? = alert["buttons"] {
                alertFacts["buttons"] = .array(buttons.filter {
                    guard case .string(let label)? = $0.objectValue?["label"] else { return true }
                    return !offeredLabels.contains(label)
                })
            }
            observation["system_alert"] = .object(alertFacts)
        }
        var lastAction: [String: JSONValue] = [
            "action": operation,
            "verdict": proofVerdict ?? (unknownDelivery ? .string("unknown")
                : (operation == .string("observe") || operation == .string("screenshot"))
                    && outcome == .string("succeeded") ? .string("observed") : outcome),
        ]
        if let resolvedJourneyLabel { lastAction["label"] = .string(resolvedJourneyLabel) }
        if let direction = body["direction"] { lastAction["direction"] = direction }
        for key in ["dispatch_attempted", "submission_started", "delivery_acknowledged"] {
            if let value = body[key] { lastAction[key] = value }
        }
        if unknownDelivery { lastAction["delivery"] = .string("unknown") }
        if let note = body["outcome_note"] { lastAction["note"] = note }
        var packet: [String: JSONValue] = [
            "schema_version": .integer(1), "observation": .object(observation),
            "last_action": .object(lastAction),
            "allowed_next": .array(allowed.map(\.rawValue).sorted().map(JSONValue.string)),
            "facts": .array(facts),
            "choices": .array(Self.choicesForDisplay(choices, includesImage: !images.isEmpty)),
        ]
        if let instruction = body["instruction"], instruction != .string("Observation is complete.") {
            packet["guidance"] = instruction
        }
        if !swipeDirections.isEmpty {
            packet["can_swipe"] = .array(swipeDirections.map { .string($0.rawValue) })
        }
        if let image = body["image"] { packet["image"] = image }
        if let changedHint = currentJourneyHint, changedHint != lastEmittedJourneyHint {
            packet["journey_hint"] = .string(changedHint)
        }

        // Compare only semantic decision evidence. Operation wrappers, hints,
        // local IDs and formatting cannot disguise an unchanged observation.
        var semanticObservation = observation
        semanticObservation.removeValue(forKey: "id")
        let orderedSemanticChoices = try semanticChoices.map {
            (key: try $0.encoded(), value: $0)
        }.sorted { $0.key < $1.key }.map(\.value)
        var comparison: [String: JSONValue] = [
            "observation": .object(semanticObservation), "facts": .array(facts),
            "choices": .array(orderedSemanticChoices),
            "allowed_next": packet["allowed_next"]!,
        ]
        if let directions = packet["can_swipe"] { comparison["can_swipe"] = directions }
        if !images.isEmpty { comparison["images"] = .array(images.map { .string($0.sha256) }) }
        let encoded = try JSONValue.object(packet).encoded()
        if packet["journey_hint"] != nil { lastEmittedJourneyHint = currentJourneyHint }
        return DecisionPacket(content: encoded, comparison: .object(comparison))
    }

    private static func choicesForDisplay(_ choices: [JSONValue], includesImage: Bool) -> [JSONValue] {
        guard !includesImage else { return choices }
        let labels = choices.compactMap { choice -> String? in
            guard case .string(let label)? = choice.objectValue?["label"] else { return nil }
            return label
        }
        let counts = Dictionary(grouping: labels, by: { $0 }).mapValues(\.count)
        return choices.map { choice in
            guard case .object(var object) = choice,
                  case .string(let label)? = object["label"], counts[label] == 1 else { return choice }
            object.removeValue(forKey: "position")
            return .object(object)
        }
    }

    private static func array(_ value: JSONValue?) -> [JSONValue] {
        guard case .array(let values) = value else { return [] }
        return values
    }

    private static func uniquePublishedAction(
        for intent: NavigationIntent,
        in manifest: AuthorityManifest
    ) throws -> PublishedAction {
        guard let selector = intent.selector,
              let role = intent.role else {
            throw VisionCaptureAgentError.malformedCall(
                "the intended action requires selector and role")
        }
        let kind = intent.operation == .tap ? "tap" : "set_boolean"
        let matches = manifest.actions.filter { action in
            guard action.action == kind,
                  action.selector.utf8.elementsEqual(selector.utf8),
                  action.role.utf8.elementsEqual(role.utf8) else {
                return false
            }
            if let publishedState = action.desiredState {
                return intent.desiredState == publishedState
            }
            return true
        }
        guard matches.count == 1 else {
            let detail = matches.isEmpty
                ? "did not match a current published action"
                : "matched more than one current published action"
            throw VisionCaptureAgentError.navigationUnavailable(
                "The proposed \(kind) was not sent because its selector, role, and desired state \(detail). Observe again and choose one exact available action.")
        }
        return matches[0]
    }

    private func uniquePublishedEditableField(
        for intent: NavigationIntent
    ) throws -> PublishedEditableField {
        guard let selector = intent.selector,
              let selectorKind = intent.selectorKind else {
            throw VisionCaptureAgentError.malformedCall(
                "a named type action requires selector and selector_kind")
        }
        let matches = currentEditableFields.filter {
            $0.selectorKind == selectorKind
                && (selectorKind == "placeholder"
                    ? $0.selector.utf8.elementsEqual(selector.utf8)
                    : $0.selector == selector)
        }
        guard matches.count == 1 else {
            let detail = matches.isEmpty
                ? "did not match a current published editable field"
                : "matched more than one current published editable field"
            throw VisionCaptureAgentError.navigationUnavailable(
                "Typing was not sent because its private target \(detail). Observe again and select a current typing choice.")
        }
        return matches[0]
    }

    private static func returnedAuthorityManifest(
        from root: JSONValue
    ) throws -> AuthorityManifest {
        var caches: [JSONValue] = []
        try collectStructuredObjects(named: "cache", in: root) { cache in
            let value = JSONValue.object(cache)
            if !caches.contains(value) { caches.append(value) }
        }
        guard caches.count <= 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned conflicting cache manifests")
        }
        guard let cache = caches.first,
              case .object(let object) = cache else {
            return AuthorityManifest()
        }
        return try authorityManifest(from: object)
    }

    private static func returnedSystemAlert(
        from root: JSONValue
    ) throws -> SystemAlertObservation? {
        var alerts: [JSONValue] = []
        try collectStructuredObjects(named: "system_alert", in: root) { alert in
            let value = JSONValue.object(alert)
            if !alerts.contains(value) { alerts.append(value) }
        }
        guard alerts.count <= 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned conflicting system-alert observations")
        }
        guard let alert = alerts.first,
              case .object(let object) = alert,
              case .bool(let present)? = object["present"] else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned no readable system-alert observation")
        }
        guard present else {
            if case .string(let code)? = object["reason_code"],
               code == "SYSTEM_ALERT_AMBIGUOUS"
                || code == "SYSTEM_ALERT_UNREADABLE" {
                throw VisionCaptureAgentError.mcpRefused(code)
            }
            return nil
        }
        if object["owner_is_app_under_test"] == .bool(true) {
            throw VisionCaptureAgentError.mcpRefused(
                "SYSTEM_ALERT_OWNED_BY_APP_UNDER_TEST")
        }
        guard case .string(let digest)? = object["content_digest"],
              !digest.isEmpty,
              case .string(let title)? = object["title"],
              case .array(let rawButtons)? = object["buttons"],
              !rawButtons.isEmpty else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned an incomplete system-alert observation")
        }
        var buttons: [SystemAlertButton] = []
        for rawButton in rawButtons {
            guard case .object(let button) = rawButton,
                  case .string(let label)? = button["label"],
                  !label.isEmpty,
                  case .bool(let enabled)? = button["enabled"],
                  case .bool(let visible)? = button["visible"] else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned an incomplete system-alert button")
            }
            buttons.append(SystemAlertButton(
                label: label,
                enabled: enabled,
                visible: visible))
        }
        return SystemAlertObservation(
            title: title,
            contentDigest: digest,
            buttons: buttons)
    }

    private static func authorityManifest(
        from object: [String: JSONValue]
    ) throws -> AuthorityManifest {
        guard case .string(let state)? = object["state"],
              !state.isEmpty else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned a cache manifest without a valid state")
        }
        let observationGrant = try returnedOptionalString(
            "observation_grant",
            in: object)
        if observationGrant != nil,
           state != "cold", state != "observed" {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned an observation grant outside a cold or observed manifest")
        }
        let rawActions: [JSONValue]
        if let value = object["actions"] {
            guard case .array(let actions) = value else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned a non-array cache action list")
            }
            rawActions = actions
        } else {
            rawActions = []
        }
        var actions: [PublishedAction] = []
        var authorityValues: [String: String] = [:]
        for raw in rawActions {
            guard case .object(let action) = raw,
                  case .string(let kind)? = action["action"],
                  kind == "tap" || kind == "set_boolean",
                  case .string(let selector)? = action["selector"],
                  !selector.isEmpty,
                  case .string(let role)? = action["role"],
                  !role.isEmpty else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned an incomplete published action")
            }
            let actionCapability = try returnedOptionalString(
                "action_capability",
                in: action)
            let revalidationCapability = try returnedOptionalString(
                "revalidation_capability",
                in: action)
            guard actionCapability == nil || revalidationCapability == nil else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned one action with conflicting authority types")
            }
            let desiredState = try returnedOptionalBoolean(
                "desired_state",
                in: action)
            let currentState = try returnedOptionalBoolean(
                "current_state",
                in: action)
            var published = PublishedAction(
                action: kind,
                selector: selector,
                role: role,
                desiredState: desiredState,
                currentState: currentState,
                actionCapability: actionCapability,
                revalidationCapability: revalidationCapability)
            if kind == "tap", role == "button", actionCapability == nil, revalidationCapability == nil,
               (selector.hasPrefix("__vc_button_occurrence_v1_")
                || selector.hasPrefix("__vc_segment_button_v1_")) {
                if case .string(let label)? = action["label"], label.utf8.count <= 256 {
                    published.displayLabel = label
                }
                if case .object(let position)? = action["position"],
                   Set(position.keys) == ["x_norm", "y_norm"],
                   case .integer(let x)? = position["x_norm"], case .integer(let y)? = position["y_norm"],
                   (0...1000).contains(x), (0...1000).contains(y) {
                    published.displayPosition = .object(["x_norm": .integer(x), "y_norm": .integer(y)])
                }
                if selector.hasPrefix("__vc_segment_button_v1_"),
                   case .bool(let selected)? = action["selected"] {
                    published.displaySelected = selected
                }
            }
            if let actionCapability {
                try recordAuthority(
                    actionCapability,
                    type: "action_capability",
                    in: &authorityValues)
            }
            if let revalidationCapability {
                try recordAuthority(
                    revalidationCapability,
                    type: "revalidation_capability",
                    in: &authorityValues)
            }
            if !actions.contains(published) { actions.append(published) }
        }
        if let observationGrant {
            try recordAuthority(
                observationGrant,
                type: "observation_grant",
                in: &authorityValues)
        }
        return AuthorityManifest(
            state: state,
            observationGrant: observationGrant,
            actions: actions)
    }

    private static func recordAuthority(
        _ value: String,
        type: String,
        in values: inout [String: String]
    ) throws {
        if let existing = values[value], existing != type {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture reused one authority value under multiple types")
        }
        values[value] = type
    }

    private static func returnedOptionalString(
        _ key: String,
        in object: [String: JSONValue]
    ) throws -> String? {
        guard let value = object[key] else { return nil }
        guard case .string(let string) = value,
              !string.isEmpty else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned an invalid \(key) field")
        }
        return string
    }

    private static func returnedOptionalBoolean(
        _ key: String,
        in object: [String: JSONValue]
    ) throws -> Bool? {
        guard let value = object[key] else { return nil }
        guard case .bool(let boolean) = value else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned an invalid \(key) field")
        }
        return boolean
    }

    private func staleCacheOutcome(
        for intent: NavigationIntent,
        readResult: VisionCaptureMCPResult,
        screenChanged: Bool
    ) throws -> NavigationOutcome {
        var recovery: [String: JSONValue] = [
            "code": .string("CACHE_ACTION_CAPABILITY_STALE"),
            "dispatch_attempted": .bool(false),
            "screen_changed": .bool(screenChanged),
            "remaining_confirmation_attempts": .integer(screenChanged ? 0 : 1),
        ]
        if !screenChanged {
            recovery["confirming_action"] = .object(
                Self.sanitizedJourneyAction(JourneyAction(intent)))
        }
        var body: [String: JSONValue] = [
            "operation": .string(intent.operation.rawValue),
            "outcome": .string("not_dispatched_reobserved"),
            "available_actions": .array([]),
            "stale_recovery": .object(recovery),
            "instruction": .string(screenChanged
                ? "The screen content changed before the action could be sent. Nothing was pressed. Choose observe, then choose again."
                : "Nothing was pressed. The follow-up read found unchanged screen content. You may choose the target marked confirmation_attempts: 1 once, or observe and choose a different action. The refused press was not retried."),
        ]
        let availableTextFields = Self.sanitizedEditableFields(
            currentEditableFields, facts: currentScreenFacts)
        if let screenSummary = Self.returnedScreenFacts(from: readResult.value)?.summary(
            editableFields: availableTextFields) {
            body["screen_summary"] = .string(screenSummary)
        }
        if !currentEditableFields.isEmpty {
            body["available_text_fields"] = .array(availableTextFields)
        }
        return NavigationOutcome(
            content: try encodeOutcomeBody(body),
            recoverableColdMissArguments: nil,
            progressed: false,
            successfulReadOnlyObservation: false)
    }

    private func outcome(
        for intent: NavigationIntent,
        result: VisionCaptureMCPResult,
        arguments: JSONValue,
        manifest: AuthorityManifest,
        systemAlert: SystemAlertObservation?,
        progressed: Bool,
        observedScreenFacts: VisionCaptureScreenFacts? = nil,
        reobservedBeforeDispatch: Bool = false,
        expiredBeforeDispatch: VisionCaptureServerOutcome? = nil,
        deliveredFailure: VisionCaptureServerOutcome? = nil,
        verifiedTypingProof: [String: JSONValue]? = nil,
        typingDispatchAttempted: Bool? = nil,
        observationRefreshed: Bool = false
    ) throws -> NavigationOutcome {
        let facts = result.isError ? nil
            : Self.returnedScreenFacts(from: result.value) ?? observedScreenFacts
        offeredTapCandidates = nil
        let candidates: [VisionCaptureScreenFacts.TapCandidate]
        if !result.isError, systemAlert == nil,
           let signature = currentScreenSignature, let facts {
            candidates = facts.tapCandidates(excluding: manifest.actions.map { ($0.selector, $0.role) })
                .filter { candidate in
                    let proposal = NavigationIntent(
                        operation: .tap, selector: candidate.selector, selectorKind: nil,
                        role: candidate.role, desiredState: nil, text: nil)
                    return isEligibleOfferedAction(proposal)
                }
            offeredTapCandidates = (signature, candidates)
        } else {
            candidates = []
        }
        let availableActions = Self.sanitizedActions(eligibleOfferedActions(manifest.actions), facts: facts)
            + Self.sanitizedSystemAlertActions(systemAlert)
            + candidates.map { candidate in
                var object = facts?.properties(selector: candidate.selector, role: candidate.role) ?? [:]
                object["action"] = .string("tap")
                object["selector"] = .string(candidate.selector)
                object["role"] = .string(candidate.role)
                object["requires_validation"] = .bool(true)
                return .object(object)
            }
        var body: [String: JSONValue] = [
            "operation": .string(intent.operation.rawValue),
            "outcome": .string(result.isRecoverableColdMiss
                || result.isGuardedTargetRejectedBeforeSubmission
                ? "not_dispatched" : "succeeded"),
            "available_actions": .array(availableActions),
        ]
        if result.isRecoverableColdMiss {
            body["dispatch_attempted"] = .bool(false)
            body["instruction"] = .string(
                "The action was not sent because its current target could not be confirmed. Choose observe, then choose again.")
        }
        if result.isGuardedTargetRejectedBeforeSubmission {
            body["dispatch_attempted"] = .bool(false)
            body["refusal"] = .object([
                "code": .string("GUARDED_TARGET_REJECTED"),
                "reason": .string("DISPATCH_REJECTED_BEFORE_SUBMISSION"),
            ])
            body["instruction"] = .string(
                "VisionCapture rejected this exact target before submission. The action did not happen and will not be sent again. Choose observe for fresh actions, then choose a different exact action, or report that no safe alternative is published.")
        }
        let proof = try Self.sanitizedNamedObject(
            "proof",
            allowedKeys: ["verdict", "verdict_source", "reason_code", "action"],
            in: result.value)
        if let proof {
            body["proof"] = .object(proof)
        }
        if !result.isRecoverableColdMiss,
           !result.isGuardedTargetRejectedBeforeSubmission,
           !reobservedBeforeDispatch, expiredBeforeDispatch == nil,
           deliveredFailure == nil, verifiedTypingProof == nil, !observationRefreshed {
            switch intent.operation {
            case .tap, .setBoolean, .type, .back, .swipe:
                body.merge(try Self.sanitizedDeliveryFacts(in: result.value)) { _, returned in returned }
                if let attempted = result.dispatchAttempted { body["dispatch_attempted"] = .bool(attempted) }
                let mutationOutcome: String
                switch proof?["verdict"] {
                case .some(.string("verified")):
                    mutationOutcome = "succeeded"
                case .some(.string("inconclusive")):
                    mutationOutcome = "inconclusive"
                case .some(.string("failed")):
                    mutationOutcome = "failed"
                default:
                    mutationOutcome = "unverified"
                }
                body["outcome"] = .string(mutationOutcome)
                if mutationOutcome == "inconclusive" || mutationOutcome == "unverified" {
                    body["outcome_note"] = .string(
                        "Request completion does not establish the expected effect, and does not establish that no effect occurred.")
                }
            case .launch, .observe, .screenshot:
                break
            }
        }
        if let launch = try Self.sanitizedLaunchOutcome(in: result.value) {
            body["launch"] = .object(launch)
        }
        let availableTextFields = systemAlert == nil
            ? Self.sanitizedEditableFields(currentEditableFields, facts: facts) : []
        if let screenSummary = facts?.summary(
            availableActions: availableActions, editableFields: availableTextFields) {
            body["screen_summary"] = .string(screenSummary)
        }
        if let systemAlert {
            body["system_alert"] = .object(Self.sanitizedSystemAlert(systemAlert))
        } else if !currentEditableFields.isEmpty {
            body["available_text_fields"] = .array(availableTextFields)
        }
        if intent.operation == .observe || reobservedBeforeDispatch || deliveredFailure != nil
            || verifiedTypingProof != nil,
           !result.isError {
            if !availableActions.isEmpty || !currentEditableFields.isEmpty {
                body["instruction"] = .string(
                    "Observation is complete.")
            } else {
                body["instruction"] = .string(
                    "Observation is complete and published no current accessibility action or editable field. Do not repeat observe unless an external screen change is expected. Choose screenshot if pixels could clarify a different safe next step, give the final answer, or report this factual blocker.")
            }
        }
        if availableActions.isEmpty, currentEditableFields.isEmpty,
           systemAlert == nil,
           !result.isRecoverableColdMiss,
           !result.isGuardedTargetRejectedBeforeSubmission {
            body["navigation_fact"] = .string(
                "No app-owned accessibility action is currently published. Observe again only if the screen is expected to change, choose screenshot if pixels could clarify a different safe next step, or report the blocker.")
        }
        if reobservedBeforeDispatch {
            body["outcome"] = .string("not_dispatched_reobserved")
            body["dispatch_attempted"] = .bool(false)
            if !result.isError {
                body["instruction"] = .string(
                    "The screen layout changed before the chosen action could be sent. The host read the current screen and refreshed its choices. Make a new decision from these current choices, or answer if the user's goal is complete. The previous action did not happen.")
            }
        }
        if let expiredBeforeDispatch {
            body["outcome"] = .string("not_dispatched_reobserved")
            body["is_error"] = .bool(true)
            body["dispatch_attempted"] = .bool(false)
            body["observation_outcome"] = .string(result.isError ? "unavailable" : "succeeded")
            body["refusal"] = .object([
                "code": .string("CACHE_ACTION_CAPABILITY_INVALID"),
                "reason": .string("authorization_expired_before_dispatch"),
            ])
            var originalProof: [String: JSONValue] = [
                "verdict": .string("failed"), "action": .string("tap"),
            ]
            if let reason = expiredBeforeDispatch.reason { originalProof["reason"] = .string(reason) }
            body["proof"] = .object(originalProof)
            body["instruction"] = .string(result.isError
                ? "The action was not sent. The follow-up read could not provide current choices. Choose observe before choosing another action."
                : "The action was not sent. The follow-up read provided these current choices. Choose again without retrying a refused target.")
        }
        if let deliveredFailure {
            body["outcome"] = .string("failed_reobserved")
            body["is_error"] = .bool(true)
            body["dispatch_attempted"] = .bool(true)
            body["submission_started"] = .bool(true)
            body["delivery_acknowledged"] = .bool(true)
            var originalProof: [String: JSONValue] = ["verdict": .string("failed")]
            if let reasonCode = deliveredFailure.reasonCode {
                originalProof["reason_code"] = .string(reasonCode)
            }
            if let reason = deliveredFailure.reason { originalProof["reason"] = .string(reason) }
            body["proof"] = .object(originalProof)
            if !result.isError {
                body["instruction"] = .string(
                    "The input was delivered, but its expected result failed. These choices come from the follow-up read. Do not repeat the input or claim success. Choose the next step from current evidence.")
            } else {
                body["instruction"] = .string(
                    "The input was delivered, but its expected result failed. The screen changed and current choices are unavailable. Choose observe. Do not repeat the input or claim success.")
            }
        }
        if observationRefreshed {
            // These facts describe the rejected read, not the prior mutation.
            body["observation_refusal"] = .object([
                "error": .string("CACHE_ACTION_CAPABILITY_STALE"),
                "cache_authorization_phase": .string("core_observation_completion"),
                "cache_authorization_reason": .string("live_screen_binding_mismatch"),
                "recovery_action": .string("observe_and_inspect"),
                "recovery_reason": .string("observation_topology_changed_before_completion"),
                "observation_grant_retired": .bool(true),
                "cache_used": .bool(false), "cache_revalidation_used": .bool(false),
                "revalidation_verified": .bool(false), "fresh_authority_recorded": .bool(false),
                "dispatch_attempted": .bool(false),
            ])
            body["observation_outcome"] = .string("refreshed")
            if expiredBeforeDispatch == nil, deliveredFailure == nil, verifiedTypingProof == nil {
                body["outcome"] = .string(intent.operation == .observe ? "succeeded" : "not_dispatched")
                body["dispatch_attempted"] = .bool(false)
                body["instruction"] = .string(
                    "The screen changed during observation. These choices come from a fresh read. No pending input was sent. Choose again and do not assume a form was saved.")
            }
        }
        if let verifiedTypingProof {
            body["outcome"] = .string("succeeded")
            body["proof"] = .object(verifiedTypingProof)
            // A cold inspection must not overwrite the original type request's
            // dispatch truth with the observation's pre-dispatch false flag.
            body.removeValue(forKey: "dispatch_attempted")
            if let typingDispatchAttempted {
                body["dispatch_attempted"] = .bool(typingDispatchAttempted)
            }
            body["observation_outcome"] = .string(result.isError ? "unavailable" : "succeeded")
            body["instruction"] = .string(result.isError
                ? "Typing was verified, but fresh navigation evidence is unavailable. Do not repeat the typed input. Choose observe for current choices before another action."
                : "Typing was verified and the host refreshed the current screen. Choose the next action from these current choices. A verified field change does not mean the form was saved. Do not reuse an older action that is absent here.")
        }
        return NavigationOutcome(
            content: try encodeOutcomeBody(body),
            recoverableColdMissArguments: result.isRecoverableColdMiss
                ? arguments : nil,
            progressed: progressed,
            successfulReadOnlyObservation:
                (intent.operation == .observe || reobservedBeforeDispatch || expiredBeforeDispatch != nil
                    || deliveredFailure != nil || observationRefreshed
                    || verifiedTypingProof != nil) && !result.isError)
    }

    private func encodeOutcomeBody(_ body: [String: JSONValue]) throws -> String {
        // Internal evidence is wrapped once in decisionPacket after all recovery
        // branches have finished. Hint emission is committed only in that packet.
        try JSONValue.object(body).encoded()
    }

    private func observationRefreshOutcome(
        for intent: NavigationIntent, prepared: PreparedNavigation
    ) throws -> NavigationOutcome {
        try outcome(
            for: intent, result: prepared.result, arguments: prepared.arguments,
            manifest: prepared.manifest, systemAlert: prepared.systemAlert,
            progressed: false, observedScreenFacts: prepared.screenFacts,
            observationRefreshed: true)
    }

    private func observedCandidateRefusal(
        _ intent: NavigationIntent, prepared: PreparedNavigation, screenChanged: Bool
    ) throws -> NavigationOutcome {
        let observation = try outcome(
            for: intent, result: prepared.result, arguments: prepared.arguments,
            manifest: prepared.manifest, systemAlert: prepared.systemAlert,
            progressed: false, observedScreenFacts: prepared.screenFacts)
        guard case .object(var body) = try JSONDecoder().decode(
            JSONValue.self, from: Data(observation.content.utf8)) else {
            throw VisionCaptureAgentError.malformedCall("The host could not encode observed-target validation.")
        }
        body["outcome"] = .string("not_dispatched")
        body["dispatch_attempted"] = .bool(false)
        body["screen_changed"] = .bool(screenChanged)
        body["refusal"] = .object(["code": .string("OBSERVED_TARGET_NOT_PUBLISHED")])
        body.removeValue(forKey: "proof")
        body.removeValue(forKey: "outcome_note")
        body["instruction"] = .string(
            "The chosen observed target was not sent: the screen changed or fresh validation did not publish one exact matching action. Choose from the current choices, choose screenshot if pixels could clarify a different safe next step, or report the blocker. Do not repeat the rejected target on the unchanged screen.")
        return NavigationOutcome(
            content: try JSONValue.object(body).encoded(),
            recoverableColdMissArguments: nil, progressed: false,
            successfulReadOnlyObservation: false)
    }

    private static func addingReadOnlyNoProgressCorrection(
        to content: String,
        repetitionCount: Int
    ) throws -> String {
        guard let data = content.data(using: .utf8),
              case .object(var body) = try JSONDecoder().decode(
                JSONValue.self,
                from: data) else {
            throw VisionCaptureAgentError.malformedCall(
                "the host could not encode its read-only no-progress correction")
        }
        let instruction =
            "The same sanitized screen and the same current navigation choices have now been returned \(repetitionCount) times without a successful mutation. Do not choose observe again for these unchanged facts. Choose a current target ID for its offered operation, choose screenshot if pixels could clarify a different safe next step, or give the final answer and report that no current choice serves the user's goal. A screenshot includes a subsequent accessibility read; use only its newly offered targets, and observe again if that read could not offer current choices. It does not restore refused actions. Otherwise observe again only after a submitted mutation, an explicit read-only recovery instruction, or an expected external screen change."
        body["guidance"] = .string(instruction)
        return try JSONValue.object(body).encoded()
    }

    private static func systemAlertPressOutcome(
        intent: NavigationIntent,
        pressResult: VisionCaptureMCPResult,
        observed: PreparedNavigation,
        deliveryUnknown: Bool
    ) throws -> NavigationOutcome {
        var body: [String: JSONValue] = [
            "operation": .string(intent.operation.rawValue),
            "outcome": .string(deliveryUnknown
                ? "delivery_unknown_reobserved"
                : "submitted_once_reobserved"),
            "available_actions": .array(
                sanitizedSystemAlertActions(observed.systemAlert)),
            "instruction": .string(deliveryUnknown
                ? "Delivery of the selected alert press is unknown. It was not retried. Use only the new read-only alert state below; never choose the same press again."
                : "The alert press was submitted once and then observed read-only. Use this new state before any further navigation."),
        ]
        body.merge(try sanitizedDeliveryFacts(in: pressResult.value)) { _, returned in returned }
        if let proof = try sanitizedNamedObject(
            "proof",
            allowedKeys: ["verdict", "verdict_source", "reason_code", "action"],
            in: pressResult.value) {
            body["proof"] = .object(proof)
        }
        if let alert = observed.systemAlert {
            body["system_alert"] = .object(sanitizedSystemAlert(alert))
        } else {
            body["system_alert"] = .object([
                "present": .bool(false),
                "fact": .string("No native iOS system alert is present in the follow-up read."),
            ])
        }
        return NavigationOutcome(
            content: try JSONValue.object(body).encoded(),
            recoverableColdMissArguments: nil,
            progressed: !deliveryUnknown,
            successfulReadOnlyObservation: false)
    }

    private func isEligibleOfferedAction(_ intent: NavigationIntent) -> Bool {
        do {
            try rejectPreviouslyRejectedProposal(intent)
            try rejectBlockedStaleAction(intent)
            return true
        } catch {
            return false
        }
    }

    /// Filter only the model-facing projection. Keep the complete manifest for
    /// exact capability binding and to prevent a blocked action reappearing as
    /// an unlearned candidate under the same observed control.
    private func eligibleOfferedActions(_ actions: [PublishedAction]) -> [PublishedAction] {
        actions.compactMap { action in
            guard let operation = NavigationOperation(rawValue: action.action) else { return nil }
            let states: [Bool?] = operation == .setBoolean && action.desiredState == nil
                ? [false, true] : [action.desiredState]
            let eligibleStates = states.filter { state in
                isEligibleOfferedAction(NavigationIntent(
                    operation: operation, selector: action.selector, selectorKind: nil,
                    role: action.role, desiredState: state, text: nil))
            }
            guard !eligibleStates.isEmpty else { return nil }
            guard eligibleStates.count != states.count else { return action }
            // An open Boolean publication can have just one admissible state.
            // Narrow only its offered choice, never the underlying capability.
            return PublishedAction(
                action: action.action, selector: action.selector, role: action.role,
                desiredState: eligibleStates[0], currentState: action.currentState,
                actionCapability: action.actionCapability,
                revalidationCapability: action.revalidationCapability)
        }
    }

    private static func sanitizedActions(
        _ actions: [PublishedAction],
        facts: VisionCaptureScreenFacts? = nil
    ) -> [JSONValue] {
        var values: [JSONValue] = []
        for action in actions {
            var object: [String: JSONValue] = [
                "action": .string(action.action),
                "selector": .string(action.selector),
                "role": .string(action.role),
            ]
            if let label = action.displayLabel { object["label"] = .string(label) }
            if let position = action.displayPosition { object["position"] = position }
            if let selected = action.displaySelected { object["selected"] = .bool(selected) }
            if let facts {
                object.merge(facts.properties(selector: action.selector, role: action.role)) {
                    existing, _ in existing
                }
            }
            if let desiredState = action.desiredState {
                object["desired_state"] = .bool(desiredState)
            }
            if let currentState = action.currentState {
                object["current_state"] = .bool(currentState)
            }
            let value = JSONValue.object(object)
            if !values.contains(value) { values.append(value) }
        }
        return values.sorted { lhs, rhs in
            ((try? lhs.encoded()) ?? "") < ((try? rhs.encoded()) ?? "")
        }
    }

    private static func sanitizedEditableFields(
        _ fields: [PublishedEditableField],
        facts: VisionCaptureScreenFacts? = nil
    ) -> [JSONValue] {
        fields.map { field in
            var object: [String: JSONValue] = [
                "selector": .string(field.selector),
                "selector_kind": .string(field.selectorKind),
                "role": .string(field.role),
            ]
            if field.selectorKind != "placeholder", let facts {
                object.merge(facts.properties(
                    selector: field.selector, role: field.role,
                    selectorKind: field.selectorKind)) { existing, _ in existing }
            }
            return .object(object)
        }
    }

    private static func sanitizedJourneyAction(
        _ action: JourneyAction
    ) -> [String: JSONValue] {
        var value: [String: JSONValue] = [
            "action": .string(action.operation.rawValue),
        ]
        if let selector = action.selector {
            value["selector"] = .string(selector)
        }
        if let selectorKind = action.selectorKind {
            value["selector_kind"] = .string(selectorKind)
        }
        if let role = action.role {
            value["role"] = .string(role)
        }
        if let desiredState = action.desiredState {
            value["desired_state"] = .bool(desiredState)
        }
        return value
    }

    private static func sanitizedSystemAlertActions(
        _ alert: SystemAlertObservation?
    ) -> [JSONValue] {
        guard let alert else { return [] }
        let labelCounts = Dictionary(grouping: alert.buttons, by: \.label)
            .mapValues(\.count)
        return alert.buttons.compactMap { button in
            guard button.enabled, button.visible,
                  labelCounts[button.label] == 1 else { return nil }
            return .object([
                "action": .string("tap"),
                "selector": .string(button.label),
                "role": .string("system_alert_button"),
            ])
        }
    }

    private static func sanitizedSystemAlert(
        _ alert: SystemAlertObservation
    ) -> [String: JSONValue] {
        [
            "present": .bool(true),
            "fact": .string("A native iOS system alert is on screen."),
            "title": .string(alert.title),
            "buttons": .array(alert.buttons.map { button in
                .object([
                    "label": .string(button.label),
                    "role": .string("system_alert_button"),
                    "enabled": .bool(button.enabled),
                    "visible": .bool(button.visible),
                ])
            }),
        ]
    }

    private static func sanitizedDeliveryFacts(in root: JSONValue) throws -> [String: JSONValue] {
        var dispatches: [[String: JSONValue]] = []
        try collectStructuredObjects(named: "dispatch", in: root) { dispatches.append($0) }
        var facts: [String: JSONValue] = [:]
        for key in ["submission_started", "delivery_acknowledged"] {
            let values = dispatches.compactMap { $0[key] }
            guard !values.isEmpty else { continue }
            let booleans = Set(values.compactMap { value -> Bool? in
                guard case .bool(let flag) = value else { return nil }
                return flag
            })
            if booleans.count == 1, values.allSatisfy({ if case .bool = $0 { return true }; return false }) {
                facts[key] = .bool(booleans.first!)
            } else {
                facts[key] = .null
                facts["delivery_unknown"] = .bool(true)
            }
        }
        if facts["submission_started"] == .bool(true),
           facts["delivery_acknowledged"] != .bool(true) {
            facts["delivery_unknown"] = .bool(true)
        }
        return facts
    }

    private static func sanitizedNamedObject(
        _ name: String,
        allowedKeys: Set<String>,
        in root: JSONValue
    ) throws -> [String: JSONValue]? {
        var objects: [JSONValue] = []
        try collectStructuredObjects(named: name, in: root) { object in
            let sanitized = object.filter { allowedKeys.contains($0.key) }
            guard !sanitized.isEmpty else { return }
            let value = JSONValue.object(sanitized)
            if !objects.contains(value) { objects.append(value) }
        }
        guard objects.count <= 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned conflicting \(name) facts")
        }
        return objects.first?.objectValue
    }

    private static func sanitizedLaunchOutcome(
        in root: JSONValue
    ) throws -> [String: JSONValue]? {
        guard var launch = try sanitizedNamedObject(
            "launch_outcome",
            allowedKeys: [
                "verdict", "reason_code", "disposition", "mutation_sent",
                "device_readiness", "failed_proof_stage",
            ],
            in: root) else {
            return nil
        }
        var outcomes: [[String: JSONValue]] = []
        try collectStructuredObjects(named: "launch_outcome", in: root) {
            outcomes.append($0)
        }
        if let outcome = outcomes.first {
            if case .object(let process)? = outcome["process"],
               let state = process["state"] {
                launch["process_state"] = state
            }
            if case .object(let foreground)? = outcome["foreground"],
               let state = foreground["state"] {
                launch["foreground_state"] = state
            }
        }
        return launch
    }

    private static func collectStructuredObjects(
        named name: String,
        in value: JSONValue,
        visit: ([String: JSONValue]) throws -> Void
    ) throws {
        var inspectedEmbeddedTexts: Set<String> = []
        try collectStructuredObjects(
            named: name,
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            visit: visit)
    }

    private static func collectStructuredObjects(
        named name: String,
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>,
        visit: ([String: JSONValue]) throws -> Void
    ) throws {
        switch value {
        case .object(let object):
            if let named = object[name] {
                guard case .object(let nested) = named else {
                    throw VisionCaptureAgentError.malformedCall(
                        "VisionCapture returned a non-object \(name) field")
                }
                try visit(nested)
            }
            for child in object.values {
                try collectStructuredObjects(
                    named: name,
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    visit: visit)
            }
        case .array(let array):
            for child in array {
                try collectStructuredObjects(
                    named: name,
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    visit: visit)
            }
        case .string(let text):
            guard inspectedEmbeddedTexts.insert(text).inserted else { return }
            for embedded in embeddedJSONValues(in: text) {
                try collectStructuredObjects(
                    named: name,
                    in: embedded,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    visit: visit)
            }
        default:
            break
        }
    }

    private static func sanitizedScreenSummary(
        in root: JSONValue
    ) throws -> String? {
        returnedScreenFacts(from: root)?.summary()
    }

    private static func returnedScreenFacts(from root: JSONValue) -> VisionCaptureScreenFacts? {
        var elements: [JSONValue] = []
        var inspectedEmbeddedTexts: Set<String> = []
        collectReturnedElements(
            in: root, inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            into: &elements)
        var navigation: [[String: JSONValue]] = []
        try? collectStructuredObjects(named: "navigation", in: root) { navigation.append($0) }
        guard !elements.isEmpty || !navigation.isEmpty else { return nil }
        return VisionCaptureScreenFacts(elements: elements, navigation: navigation)
    }

    /// Describe-screen publishes editable elements separately from the cache
    /// action manifest. Keep existing label/identifier choices and explicitly proven
    /// placeholder metadata, publishing only selectors that identify one element.
    private static func returnedEditableFields(
        from root: JSONValue
    ) -> [PublishedEditableField] {
        var elements: [JSONValue] = []
        var inspectedEmbeddedTexts: Set<String> = []
        collectReturnedElements(
            in: root,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            into: &elements)

        var candidates: [PublishedEditableField] = []
        for value in elements {
            guard case .object(let element) = value,
                  case .string(let role)? = element["role"],
                  role == "text_field" || role == "secure_text_field",
                  element["enabled"] == .bool(true),
                  element["visible"] == .bool(true) else {
                continue
            }
            for selectorKind in ["label", "identifier"] {
                guard case .string(let raw)? = element[selectorKind] else {
                    continue
                }
                let selector = raw.trimmingCharacters(
                    in: .whitespacesAndNewlines)
                guard !selector.isEmpty,
                      selector.utf8.count <= 512,
                      selector != "[REDACTED]" else {
                    continue
                }
                candidates.append(PublishedEditableField(
                    selector: selector,
                    selectorKind: selectorKind,
                    role: role))
            }
        }

        // The public describe response carries this JSON array alongside its prose.
        // Its element ID distinguishes repeated result copies, never an executable target.
        var editableMetadata: [JSONValue] = []
        var inspectedMetadataTexts: Set<String> = []
        collectReturnedElements(
            in: root, arrayName: "editable_fields",
            inspectedEmbeddedTexts: &inspectedMetadataTexts,
            into: &editableMetadata)
        for value in editableMetadata {
            guard case .object(let field) = value,
                  case .string(let elementID)? = field["element_id"], !elementID.isEmpty,
                  case .string(let type)? = field["type"],
                  case .string(let role)? = field["role"],
                  field["enabled"] == .bool(true), field["visible"] == .bool(true),
                  case .object(let placeholder)? = field["placeholder"],
                  placeholder["status"] == .string("present"),
                  case .string(let exact)? = placeholder["text"],
                  exact.utf8.count <= 256,
                  !exact.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  exact != "[REDACTED]" else { continue }
            let expectedRole: String
            switch type {
            case "XCUIElementTypeTextField", "XCUIElementTypeSearchField":
                expectedRole = "text_field"
            case "XCUIElementTypeSecureTextField":
                expectedRole = "secure_text_field"
            case "XCUIElementTypeTextView":
                expectedRole = "interactive"
            default:
                continue
            }
            guard role == expectedRole else { continue }
            candidates.append(PublishedEditableField(
                selector: exact, selectorKind: "placeholder", role: role))
        }

        let unambiguous = candidates.filter { candidate in
            candidates.filter {
                $0.selector == candidate.selector
                    && $0.selectorKind == candidate.selectorKind
            }.count == 1
        }
        return unambiguous.sorted { lhs, rhs in
            if lhs.selectorKind != rhs.selectorKind {
                return lhs.selectorKind < rhs.selectorKind
            }
            if lhs.selector != rhs.selector {
                return lhs.selector < rhs.selector
            }
            return lhs.role < rhs.role
        }
    }

    private static func collectReturnedElements(
        in value: JSONValue,
        arrayName: String = "elements",
        inspectedEmbeddedTexts: inout Set<String>,
        into elements: inout [JSONValue]
    ) {
        switch value {
        case .object(let object):
            if case .array(let returned)? = object[arrayName],
               arrayName != "editable_fields" || returned.count <= 60 {
                for element in returned where !elements.contains(element) {
                    elements.append(element)
                }
            }
            for child in object.values {
                collectReturnedElements(
                    in: child, arrayName: arrayName,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &elements)
            }
        case .array(let array):
            for child in array {
                collectReturnedElements(
                    in: child, arrayName: arrayName,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &elements)
            }
        case .string(let text):
            guard inspectedEmbeddedTexts.insert(text).inserted else { return }
            for embedded in embeddedJSONValues(in: text) {
                collectReturnedElements(
                    in: embedded, arrayName: arrayName,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &elements)
            }
        default:
            break
        }
    }

    /// VisionCapture hashes normalized tokens from visible labels and identifiers
    /// (at most 200 tokens). This host-private content signature excludes geometry.
    private static func returnedScreenSignature(
        from root: JSONValue
    ) throws -> String? {
        var signatures: Set<String> = []
        try collectStructuredObjects(named: "view", in: root) { view in
            guard let value = view["signature_fine"] else { return }
            guard case .string(let raw) = value else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned a non-string structured fine screen signature")
            }
            let signature = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !signature.isEmpty, signature.utf8.count <= 512 else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned an invalid structured fine screen signature")
            }
            signatures.insert(signature)
        }
        guard signatures.count <= 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned conflicting structured fine screen signatures")
        }
        return signatures.first
    }

    private static func returnedScreenContentIdentity(
        from root: JSONValue
    ) throws -> ScreenContentIdentity? {
        var coarseSignatures: Set<String> = []
        try collectStructuredObjects(named: "view", in: root) { view in
            guard let value = view["signature_coarse"] else { return }
            guard case .string(let raw) = value else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned a non-string structured coarse screen signature")
            }
            let signature = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !signature.isEmpty, signature.utf8.count <= 512 else {
                throw VisionCaptureAgentError.malformedCall(
                    "VisionCapture returned an invalid structured coarse screen signature")
            }
            coarseSignatures.insert(signature)
        }
        guard coarseSignatures.count <= 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned conflicting structured coarse screen signatures")
        }
        guard let coarseSignature = coarseSignatures.first,
              let summary = try sanitizedScreenSummary(in: root) else {
            return nil
        }
        let facts = Set(summary.split(
            separator: "\n",
            omittingEmptySubsequences: true
        ).compactMap(normalizedScreenContentFact)).sorted()
        guard !facts.isEmpty else { return nil }
        let canonical = facts.joined(separator: "\n")
        let digest = SHA256.hash(data: Data(canonical.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        return ScreenContentIdentity(
            coarseSignature: coarseSignature,
            factsDigest: digest)
    }

    private static func normalizedScreenContentFact(
        _ raw: Substring
    ) -> String? {
        var fact = raw.split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
        if let dot = fact.firstIndex(of: "."),
           !fact[..<dot].isEmpty,
           fact[..<dot].allSatisfy(\.isNumber) {
            fact = String(fact[fact.index(after: dot)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !fact.isEmpty,
              fact != "Visible interactive elements:",
              fact != "Segment options:" else {
            return nil
        }
        return fact
    }

    private func invalidateScreenObservation() {
        currentChoiceBindings.removeAll(keepingCapacity: true)
        permittedNextOperations = nil
        offeredTapCandidates = nil
        currentScreenSignature = nil
        currentScreenContentIdentity = nil
        currentScreenObservation = nil
        currentScreenObservationMetadata = nil
        currentScreenObservationMetadataInvalid = false
        currentEditableFields.removeAll(keepingCapacity: true)
        currentJourneyHint = nil
    }

    private func recordCompletedJourneyAction(
        _ intent: NavigationIntent,
        result: JSONValue
    ) throws {
        invalidateScreenObservation()
        guard let proof = try Self.sanitizedNamedObject(
            "proof",
            allowedKeys: ["verdict"],
            in: result),
            proof["verdict"] == .string("verified") else {
            return
        }
        switch intent.operation {
        case .tap, .setBoolean, .type, .back, .swipe:
            appendJourneyEvent(.action(JourneyAction(intent, readableLabel: resolvedJourneyLabel)))
        case .launch, .observe, .screenshot:
            break
        }
    }

    private func recordScreenObservation(from root: JSONValue) throws {
        guard let signature = try Self.returnedScreenSignature(from: root) else {
            return
        }
        currentScreenObservation = Self.returnedScreenFacts(from: root).map { (signature, $0) }
        do {
            currentScreenObservationMetadata = try VisionCaptureScreenshot.observationMetadata(in: root)
            currentScreenObservationMetadataInvalid = false
        } catch {
            currentScreenObservationMetadata = nil
            currentScreenObservationMetadataInvalid = true
        }
        currentEditableFields = Self.returnedEditableFields(from: root)
        let contentIdentity = try Self.returnedScreenContentIdentity(from: root)
        let previousContentIdentity = currentScreenContentIdentity
        if let staleActionConfirmation,
           staleActionConfirmation.screenSignature != signature {
            self.staleActionConfirmation = nil
        }
        blockedStaleActions = Set(blockedStaleActions.filter {
            $0.screenSignature == signature
        })
        currentScreenSignature = signature
        guard let contentIdentity else {
            currentScreenContentIdentity = nil
            currentJourneyHint = nil
            return
        }

        let previousIndex = journeyEvents.lastIndex { event in
            if case .screen(let prior) = event {
                return prior == contentIdentity
            }
            return false
        }
        let interveningActions: [JourneyAction]
        if let previousIndex {
            interveningActions = journeyEvents[journeyEvents.index(after: previousIndex)...]
                .compactMap { event in
                    if case .action(let action) = event { return action }
                    return nil
                }
        } else {
            interveningActions = []
        }

        if !interveningActions.isEmpty {
            let recent = interveningActions
                .suffix(Self.journeyHintActionLimit)
                .map(Self.journeyActionDescription)
                .joined(separator: " -> ")
            currentJourneyHint =
                "The same observed screen content returned after these recent verified actions: \(recent)."
        } else if previousContentIdentity != contentIdentity {
            currentJourneyHint = nil
        }

        let lastIsSameScreen: Bool
        if case .screen(let lastIdentity)? = journeyEvents.last {
            lastIsSameScreen = lastIdentity == contentIdentity
        } else {
            lastIsSameScreen = false
        }
        if !lastIsSameScreen || !interveningActions.isEmpty {
            appendJourneyEvent(.screen(contentIdentity))
        }
        currentScreenContentIdentity = contentIdentity
    }

    private func appendJourneyEvent(_ event: JourneyEvent) {
        journeyEvents.append(event)
        if journeyEvents.count > Self.journeyEventLimit {
            journeyEvents.removeFirst(journeyEvents.count - Self.journeyEventLimit)
        }
    }

    private var currentNavigationScreenScope: NavigationScreenScope? {
        if let currentScreenContentIdentity {
            return .content(currentScreenContentIdentity)
        }
        if let currentScreenSignature {
            return .fine(currentScreenSignature)
        }
        return nil
    }

    private func rejectPreviouslyRejectedProposal(
        _ intent: NavigationIntent
    ) throws {
        guard let screen = currentNavigationScreenScope else { return }
        let proposal = RejectedBeforeSubmissionProposal(
            screen: screen,
            action: RejectedNavigationAction(intent))
        guard rejectedBeforeSubmissionProposals.contains(proposal) else {
            return
        }
        throw VisionCaptureAgentError.navigationUnavailable(
            "The same navigation action was already rejected before submission on this observed screen content and was not sent again. Choose a different current available action, or report that no safe alternative is published.")
    }

    private func rememberRejectedBeforeSubmissionProposal(
        _ intent: NavigationIntent
    ) {
        guard let screen = currentNavigationScreenScope else { return }
        let proposal = RejectedBeforeSubmissionProposal(
            screen: screen,
            action: RejectedNavigationAction(intent))
        guard !rejectedBeforeSubmissionProposals.contains(proposal) else {
            return
        }
        rejectedBeforeSubmissionProposals.append(proposal)
        if rejectedBeforeSubmissionProposals.count
            > Self.rejectedProposalLimit {
            rejectedBeforeSubmissionProposals.removeFirst(
                rejectedBeforeSubmissionProposals.count
                    - Self.rejectedProposalLimit)
        }
    }

    private static func journeyActionDescription(_ action: JourneyAction) -> String {
        let label = action.readableLabel.map {
            $0.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        }
        switch action.operation {
        case .tap:
            return "tap \(label ?? "control") [\(action.role ?? "role unknown")]"
        case .setBoolean:
            let state = action.desiredState.map(String.init) ?? "unknown"
            return "set \(label ?? "switch") to \(state)"
        case .type:
            return "type into \(label ?? action.role ?? "field")"
        case .back:
            return "back"
        case .swipe:
            return "swipe \(action.direction?.rawValue ?? "direction unavailable")"
        case .launch, .observe, .screenshot:
            return action.operation.rawValue
        }
    }

    private func adoptReturnedSessionIdentity(
        from value: JSONValue,
        isSuccessful: Bool,
        request: JSONValue
    ) throws {
        let identities = try Self.returnedSessionIdentities(from: value)
        guard let returned = identities.first else { return }
        if let committedSessionIdentity {
            if returned == committedSessionIdentity { return }
            guard isSuccessful,
                  committedSessionIdentity.kind == "flow",
                  returned.kind == "flow",
                  Self.allowsSessionRefresh(request) else {
                throw VisionCaptureAgentError.sessionIdentityMismatch
            }
            self.committedSessionIdentity = returned
        } else if isSuccessful {
            committedSessionIdentity = returned
        }
    }

    private static func allowsSessionRefresh(_ arguments: JSONValue) -> Bool {
        guard case .object(let object) = arguments,
              case .string(let request)? = object["request"] else {
            return false
        }
        switch request.trimmingCharacters(
            in: .whitespacesAndNewlines).lowercased() {
        case "launch app", "inspect cache", "describe screen":
            return true
        default:
            return false
        }
    }

    private static func returnedSessionIdentities(
        from value: JSONValue
    ) throws -> [SessionIdentity] {
        var identities: Set<SessionIdentity> = []
        var inspectedEmbeddedTexts: Set<String> = []
        collectSessionIdentities(
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            into: &identities)
        guard identities.count <= 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned multiple conflicting session identities")
        }
        return Array(identities)
    }

    private static func collectSessionIdentities(
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>,
        into identities: inout Set<SessionIdentity>
    ) {
        switch value {
        case .object(let object):
            if case .string(let id)? = object["session_id"],
               !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               case .string(let kind)? = object["session_kind"],
               kind == "flow" || kind == "learning" {
                identities.insert(SessionIdentity(id: id, kind: kind))
            }
            for child in object.values {
                collectSessionIdentities(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &identities)
            }
        case .array(let array):
            for child in array {
                collectSessionIdentities(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &identities)
            }
        case .string(let text):
            guard inspectedEmbeddedTexts.insert(text).inserted else { return }
            for embedded in embeddedJSONValues(in: text) {
                collectSessionIdentities(
                    in: embedded,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    into: &identities)
            }
        default:
            break
        }
    }

    private static func embeddedJSONValues(in text: String) -> [JSONValue] {
        let bytes = Array(text.utf8)
        var values: [JSONValue] = []
        var index = 0
        while index < bytes.count {
            guard bytes[index] == 0x7B || bytes[index] == 0x5B else {
                index += 1
                continue
            }
            let start = index
            var closingBytes: [UInt8] = [bytes[index] == 0x7B ? 0x7D : 0x5D]
            var inString = false
            var isEscaped = false
            index += 1
            while index < bytes.count, !closingBytes.isEmpty {
                let byte = bytes[index]
                if inString {
                    if isEscaped {
                        isEscaped = false
                    } else if byte == 0x5C {
                        isEscaped = true
                    } else if byte == 0x22 {
                        inString = false
                    }
                } else {
                    switch byte {
                    case 0x22: inString = true
                    case 0x7B: closingBytes.append(0x7D)
                    case 0x5B: closingBytes.append(0x5D)
                    default:
                        if byte == closingBytes.last { closingBytes.removeLast() }
                    }
                }
                index += 1
            }
            guard closingBytes.isEmpty else { break }
            let block = Data(bytes[start..<index])
            if let value = try? JSONDecoder().decode(JSONValue.self, from: block) {
                values.append(value)
            }
        }
        return values
    }

    private static func validateReturnedIdentity(
        in value: JSONValue,
        configuration: VisionCaptureAgentConfiguration,
        refusalCode: String?,
        permittedLaunchTimeout: [String: JSONValue]? = nil
    ) throws {
        var inspectedEmbeddedTexts: Set<String> = []
        try validateReturnedIdentityFields(
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            configuration: configuration,
            path: "$",
            refusalCode: refusalCode,
            permittedLaunchTimeout: permittedLaunchTimeout)
    }

    private static func validateReturnedIdentityFields(
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>,
        configuration: VisionCaptureAgentConfiguration,
        path: String,
        refusalCode: String?,
        permittedLaunchTimeout: [String: JSONValue]?,
        isTimeoutOutcome: Bool = false,
        isTimeoutForeground: Bool = false
    ) throws {
        switch value {
        case .object(let object):
            for (key, child) in object {
                let childPath = path + "[" + (try JSONValue.string(key).encoded()) + "]"
                if ["bundle_id", "requested_bundle_id", "observed_bundle_id"]
                    .contains(key),
                   child != .string(configuration.bundleIdentifier),
                   !(isTimeoutForeground && key == "bundle_id") {
                    throw VisionCaptureAgentError.returnedIdentityMismatch(
                        fieldPath: boundedIdentityDiagnosticPath(childPath), refusalCode: refusalCode)
                }
                if ["udid", "requested_udid", "bound_udid"].contains(key),
                   child != .string(configuration.simulatorUDID) {
                    throw VisionCaptureAgentError.returnedIdentityMismatch(
                        fieldPath: boundedIdentityDiagnosticPath(childPath), refusalCode: refusalCode)
                }
                try validateReturnedIdentityFields(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    configuration: configuration,
                    path: childPath,
                    refusalCode: refusalCode,
                    permittedLaunchTimeout: permittedLaunchTimeout,
                    isTimeoutOutcome: key == "launch_outcome"
                        && permittedLaunchTimeout != nil
                        && child.objectValue == permittedLaunchTimeout,
                    isTimeoutForeground: isTimeoutOutcome && key == "foreground"
                        && (child.objectValue?["state"] == .string("other")
                            || child.objectValue?["state"] == .string("system")))
            }
        case .array(let array):
            for (index, child) in array.enumerated() {
                try validateReturnedIdentityFields(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    configuration: configuration,
                    path: path + "[\(index)]",
                    refusalCode: refusalCode,
                    permittedLaunchTimeout: permittedLaunchTimeout)
            }
        case .string(let text):
            guard inspectedEmbeddedTexts.insert(text).inserted else { return }
            for (index, embedded) in embeddedJSONValues(in: text).enumerated() {
                try validateReturnedIdentityFields(
                    in: embedded,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    configuration: configuration,
                    path: path + ".embeddedJSON[\(index)]",
                    refusalCode: refusalCode,
                    permittedLaunchTimeout: permittedLaunchTimeout)
            }
        default:
            break
        }
    }

    private static func boundedIdentityDiagnosticPath(_ path: String) -> String {
        guard path.utf8.count > 256 else { return path }
        let suffix = "...[truncated]"
        var prefix = Array(path.utf8.prefix(256 - suffix.utf8.count))
        while String(bytes: prefix, encoding: .utf8) == nil { prefix.removeLast() }
        return String(decoding: prefix, as: UTF8.self) + suffix
    }

    private func currentProposalRepairFacts(
        configuration: VisionCaptureAgentConfiguration
    ) -> (actions: [JSONValue], fields: [JSONValue], summary: String?)? {
        guard (try? configuration.validate()) != nil,
              committedTargetKey == configuration.targetKey,
              currentSystemAlert == nil, staleActionConfirmation == nil,
              let signature = currentScreenSignature,
              let facts = currentScreenFacts else { return nil }
        var actions = Self.sanitizedActions(
            eligibleOfferedActions(currentManifest.actions), facts: facts)
        if let offered = offeredTapCandidates, offered.signature == signature {
            for candidate in offered.candidates {
                guard !currentManifest.actions.contains(where: {
                    $0.selector == candidate.selector && $0.role == candidate.role
                }) else { continue }
                let intent = NavigationIntent(
                    operation: .tap, selector: candidate.selector, selectorKind: nil,
                    role: candidate.role, desiredState: nil, text: nil)
                guard isEligibleOfferedAction(intent) else { continue }
                var object = facts.properties(selector: candidate.selector, role: candidate.role)
                object["action"] = .string("tap")
                object["selector"] = .string(candidate.selector)
                object["role"] = .string(candidate.role)
                object["requires_validation"] = .bool(true)
                actions.append(.object(object))
            }
        }
        let fields = Self.sanitizedEditableFields(currentEditableFields, facts: facts)
        return (actions, fields, facts.summary(availableActions: actions, editableFields: fields))
    }

    private static func proposalFailureResult(
        _ error: VisionCaptureAgentError,
        facts: (actions: [JSONValue], fields: [JSONValue], summary: String?)?
    ) throws -> String {
        let code: String
        switch error {
        case .invalidConfiguration:
            code = "TARGET_CONFIGURATION_REQUIRED"
        case .identityMismatch:
            code = "TARGET_IDENTITY_MISMATCH"
        case .navigationUnavailable:
            code = "NAVIGATION_EVIDENCE_REQUIRED"
        default:
            code = "INVALID_NAVIGATION_INTENT"
        }
        var body: [String: JSONValue] = [
            "outcome": .string("not_sent"),
            "recoverable": .bool(true),
            "code": .string(code),
            "message": .string(error.description),
        ]
        switch error {
        case .malformedCall, .navigationUnavailable:
            body["available_actions"] = .array(facts?.actions ?? [])
            if let facts {
                if !facts.fields.isEmpty { body["available_text_fields"] = .array(facts.fields) }
                if let summary = facts.summary { body["screen_summary"] = .string(summary) }
            }
            body["instruction"] = .string(
                "Not sent. \(localRejectionReason(error))"
                    + (facts == nil
                        ? " Current accessibility facts are unavailable. Choose observe for fresh choices."
                        : " Use only the current choices."))
        default:
            break
        }
        return try JSONValue.object(body).encoded()
    }

    private static func isRecoverableProposalError(
        _ error: VisionCaptureAgentError
    ) -> Bool {
        switch error {
        case .invalidConfiguration, .malformedCall,
                .navigationUnavailable, .identityMismatch:
            true
        default:
            false
        }
    }

    private static func localRejectionReason(
        _ error: VisionCaptureAgentError
    ) -> String {
        switch error {
        case .malformedCall(let reason), .navigationUnavailable(let reason):
            reason
        default:
            error.description
        }
    }

    private static func logLocalRejection(
        _ call: AppToolCall,
        reason: String
    ) {
        let shape: String
        if case .object(let arguments) = call.arguments {
            shape = arguments.keys.sorted().map { key in
                "\(key):\(jsonTypeName(arguments[key]!))"
            }.joined(separator: ",")
        } else {
            shape = "arguments:\(jsonTypeName(call.arguments))"
        }
        logger.warning(
            "Agent navigation proposal rejected locally: reason=\(reason, privacy: .public) top_level=\(shape, privacy: .public)")
    }

    private static func jsonTypeName(_ value: JSONValue) -> String {
        switch value {
        case .object: "object"
        case .array: "array"
        case .string: "string"
        case .integer, .unsignedInteger, .decimal, .number: "number"
        case .bool: "boolean"
        case .null: "null"
        }
    }

    private static func coldMissActionShape(_ arguments: JSONValue) -> JSONValue {
        guard case .object(var object) = arguments else { return arguments }
        object.removeValue(forKey: "bundle_id")
        object.removeValue(forKey: "session_id")
        object.removeValue(forKey: "session_kind")
        if case .string(let request)? = object["request"] {
            object["request"] = .string(request.trimmingCharacters(
                in: .whitespacesAndNewlines).lowercased())
        }
        if case .object(var parameters)? = object["parameters"] {
            parameters.removeValue(forKey: "udid")
            object["parameters"] = .object(parameters)
        }
        return .object(object)
    }

    private static func isSystemInteractionCode(_ code: String) -> Bool {
        let normalized = code.uppercased()
        return normalized.contains("PERMISSION_PROMPT")
            || normalized.contains("SYSTEM_INTERACTION")
    }
}
