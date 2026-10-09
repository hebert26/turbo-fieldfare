import CryptoKit
import Foundation
import os
import TurboFieldfare
import TurboFieldfareDecodeProtocol

public struct VisionCaptureAgentConfiguration: Equatable, Sendable {
    public let bundleIdentifier: String
    public let simulatorUDID: String
    public let modelDirectory: URL
    /// The after-action OCR capture's image joins the next input (setting agentAutoScreenImage).
    public let autoScreenImage: Bool

    public init(
        bundleIdentifier: String,
        simulatorUDID: String,
        modelDirectory: URL,
        autoScreenImage: Bool = true
    ) {
        self.bundleIdentifier = bundleIdentifier
        self.simulatorUDID = simulatorUDID
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.autoScreenImage = autoScreenImage
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
    case proposalCorrectionExhausted(String)
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
        case .proposalCorrectionExhausted(let message): message
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
    let followUpPrompt: String?

    init(
        answer: String,
        diagnostics: AppDiagnostics,
        followUpPrompt: String? = nil
    ) {
        self.answer = answer
        self.diagnostics = diagnostics
        self.followUpPrompt = followUpPrompt
    }
}

/// Small, deterministic enforcement for explicit target restrictions in user
/// instructions. The model still receives the full instruction. This layer
/// prevents an exact prohibited control from being dispatched if a later tool
/// result distracts the model from that instruction.
struct AgentUserRestrictions: Equatable, Sendable {
    private struct ContextualAction: Hashable, Sendable {
        let verb: String
        let object: String

        var displayName: String { "\(verb) \(object)" }
    }

    private(set) var prohibitedTargets: Set<String> = []
    private var prohibitedContextualActions: Set<ContextualAction> = []

    mutating func apply(_ instruction: String) {
        let separators = CharacterSet(charactersIn: ".!?;\n")
        for fragment in instruction.components(separatedBy: separators) {
            let sentence = Self.normalizedWords(fragment)
            guard !sentence.isEmpty else { continue }
            if let action = Self.contextualProhibition(in: sentence) {
                prohibitedContextualActions.insert(action)
                continue
            }
            if let action = Self.contextualPermission(in: sentence) {
                prohibitedContextualActions.remove(action)
                continue
            }
            if let target = Self.coordinatedProhibitionTarget(in: sentence) {
                if Self.isMeaningfulTarget(target) {
                    prohibitedTargets.insert(target)
                }
                continue
            }
            let prohibited = Self.targets(
                afterAny: Self.prohibitionPrefixes, in: sentence)
            if !prohibited.isEmpty {
                prohibitedTargets.formUnion(prohibited)
                continue
            }
            for target in Self.targets(
                afterAny: Self.permissionPrefixes, in: sentence) {
                prohibitedTargets.remove(target)
            }
        }
    }

    func prohibits(
        label: String?,
        selector: String?,
        screenContext: String? = nil
    ) -> Bool {
        let candidates = [label, selector].compactMap { $0 }
        if candidates.contains(where: { candidate in
            let normalized = Self.normalizedTarget(candidate)
            if prohibitedTargets.contains(normalized) { return true }
            let candidateWords = normalized.split { !$0.isLetter && !$0.isNumber }
            return prohibitedTargets.contains { target in
                let targetWords = target.split { !$0.isLetter && !$0.isNumber }
                guard targetWords.count == 1, let targetWord = targetWords.first else {
                    return false
                }
                return candidateWords.contains { word in
                    word == targetWord
                        || String(word) == String(targetWord) + "s"
                        || String(targetWord) == String(word) + "s"
                }
            }
        }) { return true }

        let normalizedCandidates = candidates.map(Self.normalizedTarget)
        let normalizedContext = Self.normalizedTarget(screenContext ?? "")
        return prohibitedContextualActions.contains { restriction in
            guard restriction.verb == "add" else { return false }
            let isAddControl = normalizedCandidates.contains { candidate in
                let words = candidate.split { !$0.isLetter && !$0.isNumber }
                return words.contains("add") || words.contains("plus")
            }
            guard isAddControl else { return false }
            let objectWords = restriction.object.split { !$0.isLetter && !$0.isNumber }
            let candidateAndContext = normalizedCandidates.joined(separator: " ")
                + " " + normalizedContext
            return objectWords.allSatisfy {
                Self.containsEquivalentWord(String($0), in: candidateAndContext)
            }
        }
    }

    var displayTargets: [String] {
        (prohibitedTargets.union(prohibitedContextualActions.map(\.displayName))).sorted()
    }

    private static let prohibitionPrefixes = [
        "do not interact with ", "don't interact with ", "never interact with ",
        "do not test ", "don't test ", "never test ",
        "do not tap ", "don't tap ", "never tap ",
        "do not press ", "don't press ", "never press ",
        "do not use ", "don't use ", "never use ",
        "do not open ", "don't open ", "never open ",
        "do not select ", "don't select ", "never select ",
        "avoid ", "skip ",
    ]

    private static let permissionPrefixes = [
        "you may interact with ", "you can interact with ", "interact with ",
        "you may test ", "you can test ", "now test ", "test ",
        "you may tap ", "you can tap ", "now tap ", "tap ",
        "you may press ", "you can press ", "now press ", "press ",
        "you may use ", "you can use ", "now use ", "use ",
        "you may open ", "you can open ", "now open ", "open ",
        "you may select ", "you can select ", "now select ", "select ",
    ]

    private static let coordinatedVerbs: Set<String> = [
        "interact", "test", "tap", "press", "use", "open", "select",
        "manage", "create",
    ]

    private static func contextualProhibition(in sentence: String) -> ContextualAction? {
        contextualAction(
            in: sentence,
            prefixes: ["do not add ", "don't add ", "never add "])
    }

    private static func contextualPermission(in sentence: String) -> ContextualAction? {
        contextualAction(
            in: sentence,
            prefixes: ["you may add ", "you can add ", "now add ", "add "])
    }

    private static func contextualAction(
        in sentence: String,
        prefixes: [String]
    ) -> ContextualAction? {
        var candidate = sentence
        if candidate.hasPrefix("please ") { candidate.removeFirst("please ".count) }
        guard let prefix = prefixes.first(where: candidate.hasPrefix) else { return nil }
        candidate.removeFirst(prefix.count)
        if candidate.hasPrefix("more ") { candidate.removeFirst("more ".count) }
        for separator in [" because ", " since "] {
            if let range = candidate.range(of: separator) {
                candidate = String(candidate[..<range.lowerBound])
            }
        }
        let object = normalizedTarget(candidate)
        guard isMeaningfulTarget(object) else { return nil }
        return ContextualAction(verb: "add", object: object)
    }

    private static func containsEquivalentWord(_ target: String, in value: String) -> Bool {
        value.split { !$0.isLetter && !$0.isNumber }.contains { word in
            word == target
                || String(word) == target + "s"
                || target == String(word) + "s"
        }
    }

    /// Handles instructions such as "do not test or use Speak" and
    /// "do not open, manage, or create profiles again". The shared object
    /// follows the final coordinated verb.
    private static func coordinatedProhibitionTarget(in sentence: String) -> String? {
        let negativePrefixes = ["do not ", "don't ", "never "]
        guard let prefix = negativePrefixes.first(where: sentence.hasPrefix) else {
            return nil
        }
        let remainder = sentence.dropFirst(prefix.count)
        let words = remainder.split { !$0.isLetter && !$0.isNumber }
        let verbIndices = words.indices.filter {
            coordinatedVerbs.contains(String(words[$0]))
        }
        guard verbIndices.count > 1, let verbIndex = verbIndices.last else {
            return nil
        }
        var targetWords = Array(words[words.index(after: verbIndex)...])
        if words[verbIndex] == "interact", targetWords.first == "with" {
            targetWords.removeFirst()
        }
        let target = normalizedTarget(targetWords.joined(separator: " "))
        return target.isEmpty ? nil : target
    }

    private static func targets(afterAny prefixes: [String], in sentence: String) -> [String] {
        var candidate = sentence
        if candidate.hasPrefix("please ") { candidate.removeFirst("please ".count) }
        guard let prefix = prefixes.first(where: candidate.hasPrefix) else { return [] }
        candidate.removeFirst(prefix.count)
        if candidate.hasPrefix("or ") {
            candidate.removeFirst("or ".count)
            for verb in ["interact with ", "test ", "tap ", "press ", "use ", "open ", "select "]
                where candidate.hasPrefix(verb) {
                candidate.removeFirst(verb.count)
                break
            }
        }
        if let range = candidate.range(of: " because ") {
            candidate = String(candidate[..<range.lowerBound])
        }
        if let range = candidate.range(of: " since ") {
            candidate = String(candidate[..<range.lowerBound])
        }
        let pieces: [String]
        if candidate.contains(",") {
            pieces = candidate
                .replacingOccurrences(of: " and ", with: ",")
                .replacingOccurrences(of: " or ", with: ",")
                .split(separator: ",")
                .map(String.init)
        } else {
            pieces = [candidate]
        }
        return pieces
            .map(normalizedTarget)
            .filter(isMeaningfulTarget)
    }

    private static func isMeaningfulTarget(_ target: String) -> Bool {
        !target.isEmpty
            && !["anything", "something", "it", "them", "this", "that"].contains(target)
            && !target.hasPrefix("repeating ")
    }

    private static func normalizedWords(_ value: String) -> String {
        value.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    private static func normalizedTarget(_ value: String) -> String {
        var target = normalizedWords(value).trimmingCharacters(
            in: .punctuationCharacters.union(.whitespacesAndNewlines))
        if target.hasPrefix("the ") { target.removeFirst("the ".count) }
        if target.hasSuffix(" now") { target.removeLast(" now".count) }
        if target.hasSuffix(" again") { target.removeLast(" again".count) }
        for suffix in [" button", " control", " tab", " field", " switch", " feature"] {
            if target.hasSuffix(suffix) {
                target.removeLast(suffix.count)
                break
            }
        }
        return target.trimmingCharacters(in: .punctuationCharacters.union(.whitespacesAndNewlines))
    }
}

actor VisionCaptureToolLoop {
    typealias Inference = @Sendable (AppToolTurn) async throws
        -> VisionCaptureModelCompletion
    typealias Activity = @Sendable (VisionCaptureActivityEvent) async -> Void
    typealias Checkpoint = @Sendable (AgentContextCheckpointProposal) async throws
        -> DecodeContextCheckpointReceipt
    typealias HasPendingUserInstruction = @Sendable () async -> Bool

    private struct UserInstructionPendingBeforeDispatch: Error {}
    /// VisionCapture refused a name that matches two or more controls before submission.
    private struct AmbiguousTargetBeforeDispatch: Error {
        let failure: VisionCaptureServerOutcome
    }
    private struct BusyHostReadBeforeDispatch: Error {
        let failure: VisionCaptureServerOutcome
    }
    private var activePendingUserInstructionCheck: HasPendingUserInstruction?

    /// Starts a later user instruction inside the same QA task. Completed
    /// evidence and replay protection stay intact, while every executable
    /// choice from the old screen is expired before the model sees the turn.
    func prepareForUserInstruction(
        configuration: VisionCaptureAgentConfiguration
    ) throws {
        try retirePendingConfirmation()
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        invalidateScreenObservation()
        permittedNextOperations = [.observe]
        if AppVisionPackInstallationProbe.status(
            at: configuration.modelDirectory
        ) == .complete {
            permittedNextOperations?.insert(.screenshot)
        }
    }

    private enum NavigationOperation: String, Hashable {
        case launch
        case observe
        case screenshot
        case tap
        case tapCoordinates = "tap_coordinates"
        case computerUseClick = "computer_use_click"
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
        let xNorm: Int?
        let yNorm: Int?
        let visualIntent: String?

        init(operation: NavigationOperation, selector: String?, selectorKind: String?,
             role: String?, desiredState: Bool?, text: String?, direction: SwipeDirection? = nil,
             xNorm: Int? = nil, yNorm: Int? = nil, visualIntent: String? = nil) {
            self.operation = operation
            self.selector = selector
            self.selectorKind = selectorKind
            self.role = role
            self.desiredState = desiredState
            self.text = text
            self.direction = direction
            self.xNorm = xNorm
            self.yNorm = yNorm
            self.visualIntent = visualIntent
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

    /// Identifies text insertion that VisionCapture has already proved. Keep
    /// only a digest of the inserted text so duplicate prevention does not add
    /// another copy of user content to retained host memory.
    private struct VerifiedTypingAction: Hashable {
        let selector: String
        let selectorKind: String
        let role: String
        let textDigest: String

        init?(_ intent: NavigationIntent) {
            guard intent.operation == .type,
                  let text = intent.text else { return nil }
            selector = intent.selector ?? "focused"
            selectorKind = intent.selectorKind ?? "focused"
            role = intent.role ?? "focused_editable_element"
            textDigest = SHA256.hash(data: Data(text.utf8))
                .map { String(format: "%02x", $0) }
                .joined()
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
        let selectorDigest: String?
        let selectorKind: String?
        let roleDigest: String?
        let desiredState: Bool?
        let textDigest: String?
        let direction: SwipeDirection?
        let xNorm: Int?
        let yNorm: Int?

        init(_ intent: NavigationIntent) {
            operation = intent.operation
            selectorDigest = intent.selector.map(Self.digest)
            selectorKind = intent.selectorKind
            roleDigest = intent.role.map(Self.digest)
            desiredState = intent.desiredState
            direction = intent.direction
            xNorm = intent.xNorm
            yNorm = intent.yNorm
            textDigest = intent.text.map(Self.digest)
        }

        private static func digest(_ value: String) -> String {
            SHA256.hash(data: Data(value.utf8))
                .map { String(format: "%02x", $0) }
                .joined()
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
        private static let repeatLimit = 6
        private static let trackedSignatureLimit = 24

        private struct ScreenRejection {
            let signature: LocalProposalSignature
            var count: Int
        }

        private var previous: LocalProposalSignature?
        private var repeatCount = 0
        private var screen: ScreenContentIdentity?
        private var screenRejections: [ScreenRejection] = []
        private var rejectionCount = 0
        private var preservingVisualRecoveryScope = false

        mutating func record(
            call: AppToolCall,
            reason: String,
            screen currentScreen: ScreenContentIdentity?
        ) -> (count: Int, shouldStop: Bool) {
            let signature = LocalProposalSignature(call: call, reason: reason)
            if currentScreen != nil || preservingVisualRecoveryScope {
                previous = nil
                repeatCount = 0
                if screen != currentScreen {
                    screen = currentScreen
                    screenRejections.removeAll(keepingCapacity: true)
                    rejectionCount = 0
                    preservingVisualRecoveryScope = false
                }
                rejectionCount += 1
                if let index = screenRejections.firstIndex(where: {
                    $0.signature == signature
                }) {
                    screenRejections[index].count += 1
                    let count = screenRejections[index].count
                    return (count, count >= Self.repeatLimit
                        || rejectionCount >= Self.repeatLimit)
                }
                screenRejections.append(ScreenRejection(
                    signature: signature,
                    count: 1))
                if screenRejections.count > Self.trackedSignatureLimit {
                    screenRejections.removeFirst(
                        screenRejections.count - Self.trackedSignatureLimit)
                }
                return (1, rejectionCount >= Self.repeatLimit)
            }

            screen = nil
            screenRejections.removeAll(keepingCapacity: true)
            rejectionCount += 1
            if signature == previous {
                repeatCount += 1
            } else {
                previous = signature
                repeatCount = 1
            }
            return (repeatCount, repeatCount >= Self.repeatLimit
                || rejectionCount >= Self.repeatLimit)
        }

        mutating func preserveRejections(afterHostObservation currentScreen: ScreenContentIdentity?) {
            // Supplementary observation is not a new model proposal or verified
            // progress. Carry every existing rejection into its resulting scope,
            // including an alert-only read with no app content identity.
            screen = currentScreen
            preservingVisualRecoveryScope = true
        }

        mutating func reset() {
            previous = nil
            repeatCount = 0
            screen = nil
            screenRejections.removeAll(keepingCapacity: true)
            rejectionCount = 0
            preservingVisualRecoveryScope = false
        }
    }

    private struct SessionIdentity: Hashable {
        let id: String
        let kind: String
    }

    private struct ComputerUseTaskIdentity: Hashable {
        let id: String
        let generation: Int
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
    struct PublishedEditableField: Equatable {
        let selector: String
        let selectorKind: String
        let role: String
        /// Observation correlation only. Never sent as an execution selector.
        let elementID: String?
    }

    private struct EditableElementIdentity: Hashable {
        let elementID: String
        let role: String
    }

    private enum ChoiceRoute {
        case published(PublishedAction, grant: String?)
        case candidate(VisionCaptureScreenFacts.TapCandidate)
        case editable(PublishedEditableField)
        case alert(label: String, digest: String)
        case confirmation(StaleActionConfirmation)
        /// OCR text from the screenshot read, tapped at its own position.
        case screenText(x: Int64, y: Int64)
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
        /// At least one screen-text word survived the lean rule.
        var offersScreenText = false
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

    /// Dispatched actions and the app facts that followed, to find a repeated
    /// sequence that brings nothing new.
    struct ActionCycleTracker {
        struct Step: Equatable {
            let action: String
            let facts: String?
        }

        struct Cycle: Equatable {
            let sequence: [String]
            let repeats: Int
        }

        private var steps: [Step] = []

        mutating func record(action: String, facts: String?) -> Cycle? {
            steps.append(Step(action: action, facts: facts))
            if steps.count > 24 { steps.removeFirst(steps.count - 24) }
            return Self.cycle(in: steps)
        }

        mutating func reset() { steps.removeAll() }

        /// The shortest sequence of 2 to 6 actions, not all the same, that ends
        /// the history and repeats at least 3 times, when its latest repetition
        /// showed no app facts that the earlier repetitions had not shown.
        static func cycle(in steps: [Step]) -> Cycle? {
            for length in 2...6 where steps.count >= length * 3 {
                let latest = Array(steps.suffix(length))
                let actions = latest.map(\.action)
                guard Set(actions).count > 1 else { continue }
                var repeats = 1
                while steps.count >= length * (repeats + 1),
                      steps[(steps.count - length * (repeats + 1))..<(steps.count - length * repeats)]
                          .map(\.action) == actions {
                    repeats += 1
                }
                guard repeats >= 3 else { continue }
                let earlier = Set(steps[(steps.count - length * repeats)..<(steps.count - length)].map(\.facts))
                guard latest.allSatisfy({ earlier.contains($0.facts) }) else { continue }
                return Cycle(sequence: actions, repeats: repeats)
            }
            return nil
        }
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
    /// sanitized facts. Only digests/counters are retained, never an MCP result.
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
        // A new image, public ID or cache binding is not a changed app fact.
        // Keep this one-use latch even when required read recovery resets counts.
        private var visualRecoveryFactsDigest: String?

        private static func digest(_ content: String) -> String {
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
            return SHA256.hash(data: canonical)
                .map { String(format: "%02x", $0) }
                .joined()
        }

        mutating func record(content: String, appFacts: String?) -> Decision {
            observeAppFacts(appFacts)
            let digest = Self.digest(content)
            if correctionSent, let visualRecoveryFactsDigest,
               visualRecoveryFactsDigest == appFacts {
                // The supplementary image/read was not another model read.
                // A subsequent equivalent read is still the fourth, even if
                // pixels changed which otherwise unnamed choices were offered.
                repetitionCount += 1
                return .stop(repetitionCount: repetitionCount)
            }
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

        func activeCorrection(content: String, appFacts: String?) -> Int? {
            guard correctionSent else { return nil }
            let sameVisualRecoveryFacts = visualRecoveryFactsDigest != nil
                && visualRecoveryFactsDigest == appFacts
            return previousDigest == Self.digest(content) || sameVisualRecoveryFacts
                ? repetitionCount : nil
        }

        mutating func observeAppFacts(_ digest: String?) {
            if let digest, let previous = visualRecoveryFactsDigest, digest != previous {
                visualRecoveryFactsDigest = nil
            }
        }

        mutating func claimVisualRecovery(appFacts: String) -> Bool {
            guard visualRecoveryFactsDigest == nil else { return false }
            visualRecoveryFactsDigest = appFacts
            return true
        }

        mutating func reset(preservingVisualRecovery: Bool = false) {
            previousDigest = nil
            repetitionCount = 0
            correctionSent = false
            if !preservingVisualRecovery { visualRecoveryFactsDigest = nil }
        }

        var requiresAlternativeAction: Bool { correctionSent }
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
    /// Stale cache refusals since the last dispatched action.
    private var staleCacheRefusals = 0
    private static let staleCacheRefusalPauseLimit = 6
    /// Intents refused as stale twice on one screen. They skip the cache.
    private var coldPathIntents: Set<NavigationIntent> = []
    /// Published actions sent without a verified effect, by intent. An action
    /// sent three times without effect is withheld so the model tries another way.
    private var unverifiedAttempts: [NavigationIntent: Int] = [:]
    private static let unverifiedAttemptLimit = 3
    /// Fast host path: one plain read for choices, one plain tap by name, and
    /// the screen that the action result already carries. No cache contract.
    static let fastHostPath = true
    /// Fast host path: the pointer task kept live across taps in one run, and
    /// the device it belongs to. Hidden and dropped when the run ends.
    private var liveComputerUseTask: (udid: String, task: ComputerUseTaskIdentity)?
    /// The last screen facts before an action invalidated them; the effect
    /// diff compares them with the read after the action.
    private var lastInvalidatedScreenFacts: VisionCaptureScreenFacts?
    private var lastInvalidatedScreenSignature: String?
    private var journeyEvents: [JourneyEvent] = []
    private var completedCycleActions: [JourneyAction] = []
    private var verifiedTypingActions: Set<VerifiedTypingAction> = []
    private var currentUserPrompt = ""
    private var userRestrictions = AgentUserRestrictions()
    private var currentJourneyHint: String?
    // Conversation-owned: AppModel replaces this actor when starting a new
    // conversation. A new run, launch, or observation does not erase emission history.
    private var lastEmittedJourneyHint: String?
    private var observationGeneration: UInt64 = 0
    /// Only the packet carrying a usable screenshot/read pair can offer pixels
    /// for an unnamed choice. Retained model images do not renew this evidence.
    private var currentImageObservation: UInt64?
    private var nextChoiceNumber: UInt64 = 0
    private var currentChoiceBindings: [String: ChoiceBinding] = [:]
    private var permittedNextOperations: Set<NavigationOperation>?
    private var requiresReadOnlyRecovery = false
    private var resolvedJourneyLabel: String?
    private var mcpClient: VisionCaptureMCPClient?
    private let screenshotStore = AppImageAttachmentStore()
    /// The last after-action OCR capture, staged for the next input (fix c).
    private var autoScreenImage: AppImageAttachment?
    /// The last read's OCR words and the screen they were read on: a repair packet
    /// for a refused proposal (nothing was sent, nothing was read) offers them again.
    private var lastScreenText: (signature: String, blocks: JSONValue)?
    /// The screen-text choices of the last packet that offered any. The same word at
    /// the same place keeps its ID in the next packet (bound again to that read).
    /// Known limit: they carry no screen signature, so the same text at the same place
    /// on another screen keeps the ID too; the tap still hits that visible word.
    private var lastScreenTextIDs: [(block: ScreenTextBlock, id: String)] = []
    #if DEBUG
    /// Tests stand in for the vision pack probe (nil: the real probe).
    private var visionPackCompleteForTesting: Bool?
    #endif
    private var taskCheckpoint = AgentTaskCheckpoint()
    private var checkpointRequestIDs: [UUID] = []
    private var modelConversationEpoch: UUID?
    private var knownCallIDs: Set<String> = []
    private var localProposals = LocalProposalTracker()
    private var recoverableColdMisses = RecoverableColdMissTracker()
    private var readOnlyNoProgress = ReadOnlyNoProgressTracker()
    private var actionCycles = ActionCycleTracker()
    /// The last proposal sent free text together with coordinates; the coordinates were used.
    private var usedCoordinatesOverTarget = false
    private var thoughtRecoveryUsed = false
    private var thoughtRecoveryFacts: String?

    private struct GenerationRetryBoundary {
        let call: AppToolCall
        let result: AppToolResult
        let outcome: String
        let session: SessionIdentity?
    }

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
        conversationEpoch: UUID? = nil,
        maxContextTokens: Int? = nil,
        checkpoint: Checkpoint? = nil,
        forceCheckpoint: @escaping @Sendable () async -> Bool = { false },
        hasPendingUserInstruction: @escaping HasPendingUserInstruction = { false },
        inference: @escaping Inference
    ) async throws -> VisionCaptureAgentRunResult {
        try Task.checkCancellation()
        activePendingUserInstructionCheck = hasPendingUserInstruction
        let resumesRetainedTask = modelConversationEpoch.map {
            $0 != conversationEpoch
        } ?? false
        if resumesRetainedTask {
            localProposals.reset()
            // A fresh model lineage starts with fresh no-progress counters.
            readOnlyNoProgress.reset()
            recoverableColdMisses.reset()
            actionCycles.reset()
            staleCacheRefusals = 0
            coldPathIntents.removeAll()
            // The new model context has not observed these private targets.
            // Retire their host capabilities before it can choose another app action.
            invalidateScreenObservation()
            forgetKeptScreenText()
            currentManifest = AuthorityManifest()
            currentSystemAlert = nil
            permittedNextOperations = [.observe]
            if AppVisionPackInstallationProbe.status(
                at: configuration.modelDirectory
            ) == .complete {
                permittedNextOperations?.insert(.screenshot)
            }
        }
        let resumeRecord = resumesRetainedTask
            ? try taskCheckpoint.render(
                currentPacket: Self.unavailableResumeDecisionPacket(),
                safety: checkpointSafety(configuration),
                currentPacketWasRefreshed: false)
            : nil
        currentUserPrompt = userPrompt
        userRestrictions.apply(userPrompt)
        taskCheckpoint.appendUser(userPrompt, images: userImages)
        defer {
            activePendingUserInstructionCheck = nil
            screenshotStore.removeAll()
            currentImageObservation = nil
            if let pointer = liveComputerUseTask {
                // The run ended or paused: hide its pointer once and drop the
                // task. An unstructured task still sends this after a stop.
                liveComputerUseTask = nil
                Task { [self] in
                    await closeComputerUseTask(
                        pointer.task, configuration: configuration, activity: activity)
                }
            }
        }
        let developerPrompt: String?
        if hasInjectedPrompt, !resumesRetainedTask {
            developerPrompt = nil
        } else {
            let standingInstructions = Self.instructions(configuration: configuration)
            if let resumeRecord {
                developerPrompt = standingInstructions + "\n\n" + resumeRecord
            } else {
                developerPrompt = standingInstructions
            }
            hasInjectedPrompt = true
        }
        modelConversationEpoch = conversationEpoch
        var next = AppToolTurn.user(
            developerPrompt: developerPrompt,
            tools: VisionCaptureToolDefinitions.all)
        var nextReadyOffer: ReadyActionOffer?
        var retryBoundary: GenerationRetryBoundary?
        var activeThoughtRecovery: UUID?

        while true {
            let readyOffer = nextReadyOffer
            nextReadyOffer = nil
            let completion: VisionCaptureModelCompletion
            do {
                try Task.checkCancellation()
                completion = try await inference(next)
                try Task.checkCancellation()
            } catch {
                if let appError = error as? AppInferenceError,
                   case .repeatedThought(let receipt) = appError {
                    taskCheckpoint.recordInterruptedGeneration(receipt)
                    let activityID = activeThoughtRecovery ?? UUID()
                    do {
                        try Task.checkCancellation()
                        guard !thoughtRecoveryUsed else {
                            throw VisionCaptureAgentError.noProgress(
                                "The model repeated its thinking again after one recovery without verified progress or changed app facts. Generation stopped. No action was replayed.")
                        }
                        guard receipt.canRetryToolResult,
                              case .results(let results) = next, results.count == 1,
                              let boundary = retryBoundary, results == [boundary.result],
                              boundary.call.name == VisionCaptureToolDefinitions.navigateName,
                              receipt.pendingCallID == boundary.call.id,
                              receipt.pendingToolName == boundary.call.name else {
                            throw VisionCaptureAgentError.noProgress(
                                "Repeated thinking cannot be recovered at this initial, checkpoint, history-read or mismatched tool boundary. No action was replayed.")
                        }
                        // Claim before any await. Observation/ID/image churn cannot
                        // replenish this attempt or reset the other safety trackers.
                        thoughtRecoveryUsed = true
                        thoughtRecoveryFacts = currentScreenContentIdentity?.factsDigest
                        await activity(.generationRecovery(id: activityID,
                            text: "Repeated thinking interrupted (\(receipt.generatedTokens) generated tokens). Preparing current visual evidence…",
                            status: .dispatching))
                        checkpointRequestIDs.removeAll(keepingCapacity: true)
                        let refreshed = try await recoverRepeatedGeneration(
                            boundary: boundary, receipt: receipt,
                            configuration: configuration, maxContextTokens: maxContextTokens,
                            activity: activity)
                        try Task.checkCancellation()
                        try taskCheckpoint.appendRecoveryObservation(
                            afterCallID: boundary.call.id, result: refreshed.result,
                            outcome: refreshed.outcome, requestIDs: checkpointRequestIDs,
                            session: checkpointSessionReference)
                        localProposals.preserveRejections(afterHostObservation: currentScreenContentIdentity)
                        thoughtRecoveryFacts = currentScreenContentIdentity?.factsDigest
                        retryBoundary = refreshed
                        next = .results([refreshed.result])
                        activeThoughtRecovery = activityID
                        await activity(.generationRecovery(id: activityID,
                            text: "Current visual evidence prepared. Retrying the interrupted decision\(receipt.requiresRebuild ? " and rebuilding model context" : "")… Earlier app actions were not replayed.",
                            status: .dispatching))
                        continue
                    } catch {
                        let cancelled = error is CancellationError
                        await activity(.generationRecovery(id: activityID,
                            text: cancelled ? "Thinking recovery cancelled. No action was replayed."
                                : "Thinking recovery stopped: \(error)",
                            status: cancelled ? .cancelled : .localFailure(reason: String(describing: error))))
                        throw error
                    }
                }
                if let id = activeThoughtRecovery {
                    await activity(.generationRecovery(id: id,
                        text: error is CancellationError ? "Thinking recovery cancelled."
                            : "The recovered model decision failed: \(error)",
                        status: error is CancellationError ? .cancelled
                            : .localFailure(reason: String(describing: error))))
                }
                throw error
            }
            if let id = activeThoughtRecovery {
                await activity(.generationRecovery(id: id,
                    text: "Model decision resumed after repeated thinking. Earlier app actions were not replayed.",
                    status: .succeeded))
                activeThoughtRecovery = nil
            }
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
            if completion.diagnostics.stopReason == .cancelled {
                let partial = completion.content.trimmingCharacters(
                    in: .whitespacesAndNewlines)
                return VisionCaptureAgentRunResult(
                    answer: partial.isEmpty ? "Generation stopped." : partial,
                    diagnostics: completion.diagnostics)
            }
            if !completion.toolCalls.isEmpty,
               completion.diagnostics.stopReason == .toolCalls,
               await hasPendingUserInstruction() {
                try prepareForUserInstruction(configuration: configuration)
                let reason = "A new user instruction arrived before this proposed action was sent."
                let content = try JSONValue.object([
                    "outcome": .string("not_sent"),
                    "dispatch_attempted": .bool(false),
                    "reason": .string("user_instruction_pending"),
                    "instruction": .string(
                        "Do not execute this proposal. The user's new instruction will arrive as the next user turn."),
                ]).encoded()
                for call in completion.toolCalls {
                    Self.logLocalRejection(call, reason: reason)
                    await activity(.localRejection(
                        id: UUID(), call: call, reason: reason))
                    let result = AppToolResult(
                        callID: call.id, name: call.name, content: content)
                    try taskCheckpoint.appendSettled(call: call, result: result,
                        outcome: content, target: nil, requestIDs: checkpointRequestIDs,
                        session: checkpointSessionReference,
                        origin: "user_instruction_before_dispatch")
                }
                nextReadyOffer = nil
                retryBoundary = nil
                // Retain the honest not-sent outcome, then let the app deliver
                // the queued instruction in a fresh model lineage. Feeding
                // this result back first can starve that instruction forever.
                throw CancellationError()
            }
            if completion.toolCalls.isEmpty {
                guard completion.diagnostics.stopReason != .toolCalls else {
                    throw VisionCaptureAgentError.malformedCall(
                        "the model stopped for a tool call without returning one")
                }
                let answer = completion.content.trimmingCharacters(
                    in: .whitespacesAndNewlines)
                if answer.isEmpty,
                   completion.diagnostics.stopReason == .cancelled {
                    return VisionCaptureAgentRunResult(
                        answer: "Generation stopped.",
                        diagnostics: completion.diagnostics)
                }
                guard !answer.isEmpty else {
                    throw VisionCaptureAgentError.incompleteAnswer
                }
                if completion.diagnostics.stopReason == .endOfTurn,
                   Self.hasUnfinishedChecklist(answer) {
                    return VisionCaptureAgentRunResult(
                        answer: answer,
                        diagnostics: completion.diagnostics,
                        followUpPrompt: "Continue the current QA task. Your checklist still has pending or in-progress checks. Continue independent remaining checks using permitted current controls. If a check cannot proceed, report its factual blocker. Preserve completed checks and all refusal restrictions.")
                }
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
            var executionOrigin = "model_selected"
            checkpointRequestIDs.removeAll(keepingCapacity: true)
            let historicalTarget = checkpointTarget(for: call)
            let isHistoryRead = call.name == VisionCaptureToolDefinitions.historyReadName
            if isHistoryRead {
                checkingLocalProposal = false
                let reply = try taskCheckpoint.historyReply(arguments: call.arguments)
                if let reason = reply.invalidReason {
                    Self.logLocalRejection(call, reason: reason)
                    await activity(.localRejection(id: UUID(), call: call, reason: reason))
                    if localProposals.record(call: call, reason: reason,
                        screen: currentScreenContentIdentity).shouldStop {
                        throw VisionCaptureAgentError.malformedCall(reason)
                    }
                }
                content = reply.content
                executionOutcome = content
                // A local archive page is neither a new observation nor a
                // navigation attempt. Keep offers and every safety counter.
                nextReadyOffer = readyOffer
                await activity(.modelResult(callID: call.id, toolName: call.name,
                    excerpt: content, imageCount: 0))
            } else {
                do {
                    autoScreenImage = nil
                    checkingLocalProposal = true
                    let intent = try navigationIntent(from: call, configuration: configuration)
                    try configuration.validate()
                    if let committedTargetKey,
                       committedTargetKey != configuration.targetKey {
                        throw VisionCaptureAgentError.identityMismatch
                    }
                    try ensureHostContract(configuration: configuration)
                    // A successful recovery read can clear these flags while
                    // constructing its packet. Remember why it was permitted.
                    let requiredReadOnlyRecovery = requiresReadOnlyRecovery || uncertainAlertPress != nil
                    let outcome: NavigationOutcome
                    do {
                        outcome = try await perform(
                            intent,
                            readyOffer: readyOffer,
                            configuration: configuration,
                            activity: activity)
                    } catch let refusal as AmbiguousTargetBeforeDispatch {
                        // Nothing was sent. Read again and offer fresh choices.
                        outcome = try await observeAfterAmbiguousTarget(
                            intent, failure: refusal.failure,
                            configuration: configuration, activity: activity)
                    } catch let refusal as BusyHostReadBeforeDispatch {
                        // Nothing was sent. Let VisionCapture finish its own read, then read once.
                        try await Task.sleep(for: .seconds(1))
                        outcome = try await observeAfterAmbiguousTarget(
                            intent, failure: refusal.failure,
                            configuration: configuration, activity: activity,
                            instruction: Self.busyHostReadInstruction)
                    }
                    checkingLocalProposal = false
                    images = outcome.imageAttachments
                    executionOutcome = outcome.content
                    let packet = try decisionPacket(
                        from: outcome.content, call: call, configuration: configuration,
                        images: images)
                    // Added after the packet: the auto image is not coordinate evidence.
                    content = try attachingAutoScreenImage(to: packet, images: &images)
                    if usedCoordinatesOverTarget {
                        content = try Self.addingGuidanceNote(to: content, note: Self.coordinatesOverTargetNote)
                    }
                    if outcome.successfulReadOnlyObservation {
                        readOnlyNoProgress.observeAppFacts(currentScreenContentIdentity?.factsDigest)
                        if thoughtRecoveryUsed, let previous = thoughtRecoveryFacts,
                           let current = currentScreenContentIdentity?.factsDigest, previous != current {
                            thoughtRecoveryUsed = false
                            thoughtRecoveryFacts = nil
                        }
                    }
                    if outcome.progressed {
                        localProposals.reset()
                        readOnlyNoProgress.reset()
                        thoughtRecoveryUsed = false
                        thoughtRecoveryFacts = nil
                    }
                    if requiredReadOnlyRecovery || requiresReadOnlyRecovery || uncertainAlertPress != nil {
                        // A stale observation count must not suppress required
                        // recovery, including the read that resolves uncertainty.
                        // Ordinary observations do not take this reset path.
                        readOnlyNoProgress.reset(preservingVisualRecovery: true)
                    } else if outcome.successfulReadOnlyObservation {
                        switch readOnlyNoProgress.record(content: try packet.comparison.encoded(),
                            appFacts: currentScreenContentIdentity?.factsDigest) {
                        case .continueObserving:
                            break
                        case .correct(let repetitionCount):
                            permittedNextOperations?.remove(.observe)
                            content = try Self.addingReadOnlyNoProgressCorrection(
                                to: content,
                                repetitionCount: repetitionCount)
                            if canProvideVisualRecovery(for: intent, outcome: outcome,
                                    configuration: configuration),
                               Self.hasVisualRecoveryCapacity(maxContextTokens: maxContextTokens,
                                   retainedTokens: completion.diagnostics.conversationTokens,
                                   packetBytes: content.utf8.count, packetCopies: 2),
                               let factsDigest = currentScreenContentIdentity?.factsDigest,
                               readOnlyNoProgress.claimVisualRecovery(appFacts: factsDigest) {
                                let originalPacket = content
                                let originalOutcome = executionOutcome
                                do {
                                    let support = try await provideVisualRecovery(
                                        originalPacket: originalPacket, originalOutcome: originalOutcome,
                                        call: call, boundary: .repeatedRead,
                                        configuration: configuration, maxContextTokens: maxContextTokens,
                                        retainedTokens: completion.diagnostics.conversationTokens,
                                        activity: activity)
                                    content = support.content
                                    images = support.images
                                    executionOutcome = support.outcome
                                    executionOrigin = "model_selected_with_host_read_only_visual_support"
                                    readOnlyNoProgress.observeAppFacts(currentScreenContentIdentity?.factsDigest)
                                } catch {
                                    // The original read settled even if supplementary
                                    // observation was refused, failed or cancelled.
                                    // Preserve its audit without sending a fake reply.
                                    try taskCheckpoint.appendSettled(call: call,
                                        result: AppToolResult(callID: call.id, name: call.name,
                                            content: originalPacket),
                                        outcome: originalOutcome, target: historicalTarget,
                                        requestIDs: checkpointRequestIDs, session: checkpointSessionReference,
                                        origin: "model_selected_before_host_visual_support_stopped")
                                    throw error
                                }
                            }
                        case .stop(let repetitionCount):
                            // An unchanged read is not a reason to end the chat.
                            // Withhold observe, tell the model, and spend the shared
                            // rejection budget so persistent looping pauses instead.
                            permittedNextOperations?.remove(.observe)
                            let reason = "Read-only observation returned the same app facts \(repetitionCount) times without verified progress."
                            Self.logLocalRejection(call, reason: reason)
                            await activity(.localRejection(id: UUID(), call: call, reason: reason))
                            let rejection = localProposals.record(
                                call: semanticProposalForRepeatCheck(call),
                                reason: reason,
                                screen: currentScreenContentIdentity)
                            if rejection.shouldStop {
                                throw VisionCaptureAgentError.proposalCorrectionExhausted(
                                    "Agent paused after repeated read-only observations without progress. No action was dispatched by those reads. Completed work is retained. You can continue this chat.")
                            }
                            content = try Self.addingReadOnlyNoProgressCorrection(
                                to: content,
                                repetitionCount: repetitionCount)
                        }
                    }
                    if intent.operation != .observe, intent.operation != .screenshot,
                       !outcome.successfulReadOnlyObservation,
                       let cycle = actionCycles.record(
                           action: Self.cycleStepLabel(content: content, intent: intent),
                           facts: currentScreenContentIdentity?.factsDigest) {
                        let sequence = cycle.sequence.joined(separator: " → ")
                        if cycle.repeats >= 4 {
                            actionCycles.reset()
                            throw VisionCaptureAgentError.proposalCorrectionExhausted(
                                "Agent paused after repeating the same \(cycle.sequence.count) actions 4 times without new app facts (\(sequence)). Completed work is retained. You can continue this chat.")
                        }
                        content = try Self.addingCycleNote(to: content, sequence: cycle.sequence)
                    }
                    if let arguments = outcome.recoverableColdMissArguments {
                        if recoverableColdMisses.record(arguments: arguments) {
                            throw VisionCaptureAgentError.proposalCorrectionExhausted(
                                "Agent paused after three equivalent safe pre-dispatch cache refusals for the same intended action. No refused action was replayed. Completed work is retained. You can continue this chat.")
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
                } catch is UserInstructionPendingBeforeDispatch {
                    checkingLocalProposal = false
                    try prepareForUserInstruction(configuration: configuration)
                    let reason = "A new user instruction arrived before this proposed action was sent."
                    Self.logLocalRejection(call, reason: reason)
                    await activity(.localRejection(
                        id: UUID(), call: call, reason: reason))
                    content = try JSONValue.object([
                        "outcome": .string("not_sent"),
                        "dispatch_attempted": .bool(false),
                        "reason": .string("user_instruction_pending"),
                        "instruction": .string(
                            "Do not execute this proposal. The user's new instruction will arrive as the next user turn."),
                    ]).encoded()
                    executionOutcome = content
                    executionOrigin = "user_instruction_before_dispatch"
                    nextReadyOffer = nil
                } catch let error as VisionCaptureAgentError
                    where checkingLocalProposal && Self.isRecoverableProposalError(error) {
                    checkingLocalProposal = false
                    let rejectedTarget = Self.expiredProposalTarget(call: call, error: error)
                    // Check before retiring a confirmation. Declining it must
                    // not turn a refusal into eligibility for extra observation.
                    let maySupportRejectedTarget = rejectedTarget != nil
                        && canProvideRejectedTargetVisualRecovery(configuration: configuration)
                    let maySupportCoordinateImage = error == .navigationUnavailable(Self.coordinateEvidenceReason)
                        && canProvideRejectedTargetVisualRecovery(configuration: configuration)
                    // A correction declines the one confirming offer. Retiring only
                    // its public ID would let a later observation revive the attempt.
                    try retirePendingConfirmation()
                    let reason = Self.localRejectionReason(error)
                    Self.logLocalRejection(call, reason: reason)
                    await activity(.localRejection(
                        id: UUID(), call: call, reason: reason))
                    let rejection = localProposals.record(
                        call: semanticProposalForRepeatCheck(call),
                        reason: reason,
                        screen: currentScreenContentIdentity
                    )
                    if rejection.shouldStop {
                        throw VisionCaptureAgentError.proposalCorrectionExhausted(
                            "Agent paused after six rejected proposals without progress. No rejected action was sent. Completed work is retained. You can continue this chat.")
                    }
                    let failure = try Self.proposalFailureResult(
                        error,
                        facts: currentProposalRepairFacts(configuration: configuration),
                        rejectedTarget: rejectedTarget)
                    executionOutcome = failure
                    let repairPacket = try decisionPacket(
                        from: failure, call: call, configuration: configuration, images: [],
                        excludingTargetID: rejectedTarget, preservingChoiceIDs: true)
                    content = try Self.addingProposalCorrection(to: repairPacket.content)
                    if !requiresReadOnlyRecovery, uncertainAlertPress == nil,
                       let repetitionCount = readOnlyNoProgress.activeCorrection(
                           content: try repairPacket.comparison.encoded(),
                           appFacts: currentScreenContentIdentity?.factsDigest) {
                        content = try Self.addingReadOnlyNoProgressCorrection(
                            to: content, repetitionCount: repetitionCount)
                    }
                    if rejectedTarget != nil, rejection.count == 2 {
                        content = try Self.addingRepeatedTargetCorrection(to: content)
                    }
                    let supportBoundary: VisualRecoveryBoundary? =
                        rejectedTarget != nil && rejection.count == 2 && maySupportRejectedTarget
                            ? .repeatedTargetRejection
                            : (maySupportCoordinateImage ? .coordinateWithoutImage : nil)
                    if let supportBoundary,
                       Self.hasVisualRecoveryCapacity(maxContextTokens: maxContextTokens,
                           retainedTokens: completion.diagnostics.conversationTokens,
                           packetBytes: content.utf8.count, packetCopies: 2) {
                        let originalPacket = content
                        do {
                            let support = try await provideVisualRecovery(
                                originalPacket: originalPacket, originalOutcome: failure,
                                call: call, boundary: supportBoundary,
                                configuration: configuration, maxContextTokens: maxContextTokens,
                                retainedTokens: completion.diagnostics.conversationTokens,
                                excludingTargetID: rejectedTarget,
                                activity: activity)
                            content = support.content
                            images = support.images
                            executionOutcome = support.outcome
                            executionOrigin = "model_rejected_with_host_read_only_visual_support"
                            localProposals.preserveRejections(
                                afterHostObservation: currentScreenContentIdentity)
                            // This support is not a model observation or progress.
                            // Neither read-only counter nor its one-use latch resets.
                        } catch {
                            try taskCheckpoint.appendSettled(call: call,
                                result: AppToolResult(callID: call.id, name: call.name,
                                    content: originalPacket),
                                outcome: failure, target: historicalTarget,
                                requestIDs: checkpointRequestIDs, session: checkpointSessionReference,
                                origin: "model_rejected_before_host_visual_support_stopped")
                            throw error
                        }
                    }
                }
            }

            let settledResult = AppToolResult(
                    callID: call.id,
                    name: call.name,
                    content: content,
                    imageAttachments: images)
            if isHistoryRead, executionOrigin != "user_instruction_before_dispatch" {
                taskCheckpoint.appendHistoryRead(call: call, result: settledResult)
            } else {
                try taskCheckpoint.appendSettled(call: call, result: settledResult,
                    outcome: executionOutcome, target: historicalTarget, requestIDs: checkpointRequestIDs,
                    session: checkpointSessionReference, origin: executionOrigin)
            }
            if executionOrigin == "user_instruction_before_dispatch" {
                // This race has the same terminal instruction boundary as a
                // proposal suppressed immediately after model completion.
                throw CancellationError()
            }
            next = .results([settledResult])
            retryBoundary = isHistoryRead ? nil : GenerationRetryBoundary(
                call: call, result: settledResult, outcome: executionOutcome,
                session: committedSessionIdentity)
            if let checkpoint {
                let force = await forceCheckpoint()
                let permitsScreenshot = AppVisionPackInstallationProbe.status(at: configuration.modelDirectory) == .complete
                let capacityProposal = AgentContextCheckpointProposal(
                    id: UUID(), replacementEpoch: UUID(), call: call, result: settledResult,
                    record: "",
                    commit: false, force: force,
                    trigger: force ? .explicitComparison : .capacityForecast,
                    performanceEvidence: nil, permitsScreenshot: permitsScreenshot)
                let capacityAssessment = try await checkpoint(capacityProposal)
                if capacityAssessment.needed {
                    var commitProposal = capacityProposal
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
                    await AgentInferenceTrace.shared?.checkpointRead(
                        id: commitProposal.id,
                        seconds: Self.elapsedSeconds(since: refreshStarted))
                    let refreshCall = AppToolCall(
                        id: "checkpoint-read-\(commitProposal.id.uuidString)",
                        name: VisionCaptureToolDefinitions.navigateName,
                        arguments: .object(["action": .string("observe")]))
                    let freshPacket = try decisionPacket(
                        from: refreshed.content, call: refreshCall,
                        configuration: configuration, images: []).content
                    try taskCheckpoint.appendSettled(
                        call: refreshCall,
                        result: AppToolResult(callID: refreshCall.id,
                            name: refreshCall.name, content: freshPacket),
                        outcome: refreshed.content, target: nil,
                        requestIDs: checkpointRequestIDs,
                        session: checkpointSessionReference,
                        origin: "host_read_only_checkpoint_refresh")
                    taskCheckpoint.nextRevision()
                    commitProposal = AgentContextCheckpointProposal(
                        id: commitProposal.id,
                        replacementEpoch: commitProposal.replacementEpoch,
                        call: call, result: settledResult,
                        record: try taskCheckpoint.render(
                            currentPacket: freshPacket,
                            safety: checkpointSafety(configuration),
                            pendingHistoryResult: isHistoryRead ? settledResult : nil),
                        commit: true, force: commitProposal.force,
                        trigger: commitProposal.trigger,
                        performanceEvidence: commitProposal.performanceEvidence,
                        permitsScreenshot: permitsScreenshot)
                    let receipt = try await checkpoint(commitProposal)
                    guard receipt.committed else {
                        throw VisionCaptureAgentError.noProgress("The context checkpoint was not acknowledged. No action was replayed.")
                    }
                    try Task.checkCancellation()
                    modelConversationEpoch = commitProposal.replacementEpoch
                    next = .checkpoint(commitProposal.id)
                    retryBoundary = nil
                }
            }
        }
    }

    private static func unavailableResumeDecisionPacket() -> String {
        let packet = JSONValue.object([
            "origin": .string("host_resume"),
            "observation": .object([
                "state": .string("unavailable"),
                "instruction": .string(
                    "No settled current app observation was retained before the model context changed.")
            ]),
            "choices": .array([]),
            "allowed_next": .array([]),
            "guidance": .string(
                "Continue the retained user task, but obtain a new permitted observation before proposing app input.")
        ])
        return (try? packet.encoded()) ?? "{}"
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
            for key in ["value", "value_status", "position", "selected", "enabled", "current_state"] { facts[key] = properties[key] }
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
            let signatureBeforeLaunch = lastInvalidatedScreenSignature
            let arguments = makeMCPArguments(
                request: "launch app",
                parameters: [:],
                configuration: configuration)
            let result = try await executeHostRequest(
                arguments,
                configuration: configuration,
                interruptibleByUserInstruction: true,
                activity: activity)
            if let timeout = Self.isolatedLaunchTimeout(
                result, arguments: arguments, configuration: configuration) {
                return try await observeAfterLaunchTimeout(
                    timeout, configuration: configuration, activity: activity)
            }
            if Self.isolatedLaunchNotForeground(
                result, arguments: arguments, configuration: configuration) != nil {
                let launch = try Self.sanitizedLaunchOutcome(in: result.value) ?? [:]
                return try await observeAfterLaunchNotForeground(
                    launch, configuration: configuration, activity: activity)
            }
            if !result.isError {
                try recordScreenObservation(from: result.value)
            }
            let manifest = try Self.returnedAuthorityManifest(from: result.value)
            currentManifest = manifest
            let launchOutcome = try outcome(
                for: intent,
                result: result,
                arguments: arguments,
                manifest: manifest,
                systemAlert: nil,
                progressed: !result.isError)
            guard !result.isError else { return launchOutcome }
            return try await addingScreenTextAfterAction(
                launchOutcome, signatureBefore: signatureBeforeLaunch,
                configuration: configuration, activity: activity)

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
            if intent.selectorKind == Self.screenTextSelectorKind, let selector = intent.selector {
                return try await performDirectNameTap(
                    intent, selector: selector, configuration: configuration, activity: activity)
            }
            fallthrough

        case .setBoolean:
            if Self.fastHostPath, intent.operation == .tap,
               staleConfirmation == nil || intent.selectorKind == "ocr_label",
               let selector = intent.selector, isCurrentFastTapTarget(intent) {
                return try await performDirectNameTap(
                    intent, selector: selector, configuration: configuration, activity: activity)
            }
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
            } else if candidateSignature == nil, staleConfirmation == nil,
                      let reusable = reusableObservedManifest(
                        for: intent, configuration: configuration) {
                // A cold grant permits one bound read and then the exact action
                // published by that read. An intervening cache inspection
                // retires the grant, so use this current binding directly.
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
            try rejectExhaustedIntent(intent)
            let action = try Self.uniquePublishedAction(
                for: intent,
                in: manifest)
            if let staleConfirmation,
               currentScreenSignature != staleConfirmation.screenSignature {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "The confirming action was not sent because the content signature for visible text and identifiers changed while obtaining fresh navigation evidence. Choose observe and make a new action decision from the current screen.")
            }
            if Self.fastHostPath, intent.operation == .tap {
                // Fast path: no driver tap by name or cache handle. Tap at the
                // read position, or send nothing.
                return try await performDirectNameTap(
                    intent, selector: action.selector, configuration: configuration, activity: activity)
            }
            let actionArguments = try makeActionArguments(
                intent: intent,
                action: action,
                manifest: manifest,
                configuration: configuration)
            let isStaleConfirmation = staleConfirmation != nil
            let staleBaselineScreen = currentScreenSignature
            let attemptedScreenScope = currentNavigationScreenScope
            currentManifest = AuthorityManifest()
            invalidateScreenObservation()
            let result = try await executeHostRequest(
                actionArguments,
                configuration: configuration,
                interruptibleByUserInstruction: true,
                activity: activity)
            if isRecoverableObservedTapTargetUnavailable(
                result, arguments: actionArguments, configuration: configuration) {
                rememberRejectedBeforeSubmissionProposal(
                    intent, screen: attemptedScreenScope)
                return try await observeAfterObservedTapTargetUnavailable(
                    intent, failure: result.serverOutcome,
                    configuration: configuration, activity: activity)
            }
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
                // The cache is a shortcut, not a gate. A stale refusal sent
                // nothing, so the run continues on the slow path: read again,
                // then act without the cache. Only a long streak pauses the run.
                staleCacheRefusals += 1
                if staleCacheRefusals >= Self.staleCacheRefusalPauseLimit {
                    throw VisionCaptureAgentError.proposalCorrectionExhausted(
                        "Agent paused after \(Self.staleCacheRefusalPauseLimit) stale cache refusals without progress. No refused action was sent. Completed work is retained. You can continue this chat.")
                }
                if isStaleConfirmation, !layoutChangedBeforeRevalidation {
                    // Same action refused twice: stop using its cache entry.
                    coldPathIntents.insert(intent)
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
                staleCacheRefusals = 0
                coldPathIntents.removeAll()
                if let staleConfirmation {
                    blockedStaleActions.remove(staleConfirmation)
                }
                noteExecuted(intent, result: result)
            }
            let terminalManifest = try Self.returnedAuthorityManifest(
                from: result.value)
            currentManifest = terminalManifest
            let tapOutcome = { [self] in
                try outcome(
                    for: intent,
                    result: result,
                    arguments: actionArguments,
                    manifest: terminalManifest,
                    systemAlert: nil,
                    progressed: !result.isError)
            }
            if !result.isError {
                // The host verifies the tap by reading the screen itself, so
                // the model does not spend a step asking for that read.
                return try await observeAfterMutation(
                    intent, result: result, configuration: configuration,
                    activity: activity, fallback: tapOutcome)
            }
            return try tapOutcome()

        case .tapCoordinates:
            return try await performCoordinateTap(
                intent,
                configuration: configuration,
                activity: activity)

        case .computerUseClick:
            return try await performComputerUseClick(
                intent,
                configuration: configuration,
                activity: activity)

        case .type:
            if let completed = VerifiedTypingAction(intent),
               verifiedTypingActions.contains(completed),
               !Self.explicitlyRequestsRepeatedTyping(
                   userPrompt: currentUserPrompt,
                   text: intent.text ?? ""),
               // The field may have been cleared or replaced since. Block only
               // while the screen still shows that text in an editable field.
               currentScreenFacts?.editableFieldShows(text: intent.text ?? "") != false {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "This exact text was already typed into this field and VisionCapture verified it earlier in this task. It was not sent again. Use the retained result and continue with any remaining QA checks.")
            }
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
            if intent.selectorKind == "ocr_position", let x = intent.xNorm, let y = intent.yNorm {
                // The field is on screen but not published. Type at its
                // position: VisionCapture taps the point, then types.
                parameters["x_norm"] = .integer(Int64(x))
                parameters["y_norm"] = .integer(Int64(y))
            } else if let selector = intent.selector {
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
                interruptibleByUserInstruction: true,
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
                       operation: .tap, selector: candidate.selector, selectorKind: candidate.selectorKind,
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
                interruptibleByUserInstruction: true,
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
            if intent.operation == .swipe,
               !result.isError || Self.isRecoverableUncertainSwipe(
                   result, arguments: arguments, configuration: configuration) {
                return try await observeAfterMutation(
                    intent, result: result, configuration: configuration, activity: activity)
            }
            let backOutcome = try outcome(
                for: intent,
                result: result,
                arguments: arguments,
                manifest: terminalManifest,
                systemAlert: nil,
                progressed: !result.isError)
            guard intent.operation == .back, !result.isError else { return backOutcome }
            return try await addingScreenTextAfterAction(
                backOutcome, signatureBefore: lastInvalidatedScreenSignature,
                configuration: configuration, activity: activity)
        }
    }

    private func performCoordinateTap(
        _ intent: NavigationIntent,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        guard let x = intent.xNorm, let y = intent.yNorm,
              let visualIntent = intent.visualIntent else {
            throw VisionCaptureAgentError.malformedCall(
                "A coordinate tap needs every field of \(VisionCaptureToolDefinitions.coordinateTapCall).")
        }
        var arguments = makeMCPArguments(
            request: "tap coordinates",
            parameters: [
                "x_norm": .integer(Int64(x)),
                "y_norm": .integer(Int64(y)),
                "query": .string(visualIntent),
                "cache_policy": .string("visual_bypass"),
            ],
            configuration: configuration)
        currentManifest = AuthorityManifest()
        invalidateScreenObservation()
        let result: VisionCaptureMCPResult
        if Self.fastHostPath {
            // Gemma's taps go through the pointer, not the driver.
            let pointer = try await performFastPointerClick(
                x: Int64(x), y: Int64(y), label: visualIntent,
                configuration: configuration, activity: activity)
            arguments = pointer.arguments
            result = pointer.result
        } else {
            result = try await executeHostRequest(
                arguments,
                configuration: configuration,
                interruptibleByUserInstruction: true,
                activity: activity)
        }
        if !result.isError {
            try recordCompletedJourneyAction(intent, result: result.value)
            staleActionConfirmation = nil
        }
        let terminalManifest = try Self.returnedAuthorityManifest(from: result.value)
        currentManifest = terminalManifest
        let tapOutcome = { [self] in
            try outcome(
                for: intent,
                result: result,
                arguments: arguments,
                manifest: terminalManifest,
                systemAlert: nil,
                progressed: !result.isError)
        }
        if !result.isError {
            // The host reads the screen after the tap and returns fresh choices.
            return try await observeAfterMutation(
                intent, result: result, configuration: configuration,
                activity: activity, fallback: tapOutcome, actionArguments: arguments)
        }
        return try tapOutcome()
    }

    /// Fast host path: a tap at a known position is one pointer click. The
    /// pointer task stays live across taps and is hidden when the run ends.
    /// A stale, busy or expired task is replaced once and the click retried once.
    private func performFastPointerClick(
        x: Int64,
        y: Int64,
        label: String,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> (result: VisionCaptureMCPResult, arguments: JSONValue) {
        var retried = false
        var preCaptureRetried = false
        while true {
            do {
                let task: ComputerUseTaskIdentity
                if let live = liveComputerUseTask, live.udid == configuration.simulatorUDID {
                    task = live.task
                } else {
                    liveComputerUseTask = nil
                    let activation = try await executeHostRequest(
                        makeMCPArguments(
                            request: "activate computer use",
                            parameters: ["x_norm": .integer(x), "y_norm": .integer(y)],
                            configuration: configuration,
                            includeFlowSession: false),
                        configuration: configuration,
                        usesFlowSession: false,
                        interruptibleByUserInstruction: true,
                        activity: activity)
                    task = try Self.returnedComputerUseTaskIdentity(from: activation.value)
                    liveComputerUseTask = (configuration.simulatorUDID, task)
                }
                let arguments = makeMCPArguments(
                    request: "click pointer",
                    parameters: [
                        "computer_use_task_id": .string(task.id),
                        "computer_use_generation": .integer(Int64(task.generation)),
                        "x_norm": .integer(x),
                        "y_norm": .integer(y),
                        "intent": .string(label),
                        "cache_policy": .string("visual_bypass"),
                    ],
                    configuration: configuration,
                    includeFlowSession: false)
                let result = try await executeHostRequest(
                    arguments,
                    configuration: configuration,
                    usesFlowSession: false,
                    interruptibleByUserInstruction: true,
                    activity: activity)
                if result.isPointerPreCaptureFailureBeforeSubmission, !preCaptureRetried {
                    // No input was sent: the Simulator window geometry changed
                    // during the before-capture. Retry once after it settles.
                    preCaptureRetried = true
                    try await Task.sleep(nanoseconds: 500_000_000)
                    continue
                }
                var returned: Set<ComputerUseTaskIdentity> = []
                Self.collectComputerUseTaskIdentities(in: result.value, into: &returned)
                if let generation = returned.filter({ $0.id == task.id }).map(\.generation).max() {
                    liveComputerUseTask = (configuration.simulatorUDID,
                        ComputerUseTaskIdentity(id: task.id, generation: generation))
                }
                return (result, arguments)
            } catch let error as VisionCaptureAgentError
                where !retried && Self.isStalePointerTaskRefusal(error) {
                retried = true
                liveComputerUseTask = nil
            }
        }
    }

    /// The held pointer task can no longer be used: stale, the device is busy,
    /// or its lease expired.
    private static func isStalePointerTaskRefusal(_ error: VisionCaptureAgentError) -> Bool {
        guard case .mcpOutcome(let outcome) = error, let code = outcome.reasonCode else {
            return false
        }
        return code == "COMPUTER_USE_TASK_STALE" || code == "COMPUTER_USE_DEVICE_BUSY"
            || (code.contains("LEASE") && code.contains("EXPIR"))
    }

    private func performComputerUseClick(
        _ intent: NavigationIntent,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        guard let x = intent.xNorm, let y = intent.yNorm,
              let visualIntent = intent.visualIntent else {
            throw VisionCaptureAgentError.malformedCall(
                "A visual click needs every field of \(VisionCaptureToolDefinitions.visualClickCall).")
        }
        let activateArguments = makeMCPArguments(
            request: "activate computer use",
            parameters: [
                "x_norm": .integer(Int64(x)),
                "y_norm": .integer(Int64(y)),
            ],
            configuration: configuration,
            includeFlowSession: false)
        let activation = try await executeHostRequest(
            activateArguments,
            configuration: configuration,
            usesFlowSession: false,
            interruptibleByUserInstruction: true,
            activity: activity)
        let task = try Self.returnedComputerUseTaskIdentity(from: activation.value)

        let clickArguments = makeMCPArguments(
            request: "click pointer",
            parameters: [
                "computer_use_task_id": .string(task.id),
                "computer_use_generation": .integer(Int64(task.generation)),
                "x_norm": .integer(Int64(x)),
                "y_norm": .integer(Int64(y)),
                "intent": .string(visualIntent),
                "cache_policy": .string("visual_bypass"),
            ],
            configuration: configuration,
            includeFlowSession: false)
        let clickResult: VisionCaptureMCPResult
        do {
            clickResult = try await executeHostRequest(
                clickArguments,
                configuration: configuration,
                usesFlowSession: false,
                interruptibleByUserInstruction: true,
                activity: activity)
        } catch {
            await closeComputerUseTask(
                task,
                configuration: configuration,
                activity: activity)
            throw error
        }
        await closeComputerUseTask(
            task,
            configuration: configuration,
            activity: activity)

        let uncertainClick = Self.isRecoverableUncertainComputerUseClick(
            clickResult,
            arguments: clickArguments,
            configuration: configuration)
        currentManifest = AuthorityManifest()
        invalidateScreenObservation()
        if !clickResult.isError, !uncertainClick {
            try recordCompletedJourneyAction(intent, result: clickResult.value)
            staleActionConfirmation = nil
        }
        if uncertainClick {
            return try await observeAfterMutation(
                intent,
                result: clickResult,
                configuration: configuration,
                activity: activity)
        }
        if clickResult.isPointerPreCaptureFailureBeforeSubmission {
            return try pointerPreCaptureFailureOutcome(for: intent)
        }
        return try outcome(
            for: intent,
            result: clickResult,
            arguments: clickArguments,
            manifest: AuthorityManifest(),
            systemAlert: nil,
            progressed: !clickResult.isError)
    }

    private func pointerPreCaptureFailureOutcome(
        for intent: NavigationIntent
    ) throws -> NavigationOutcome {
        let body: [String: JSONValue] = [
            "operation": .string(intent.operation.rawValue),
            "outcome": .string("not_dispatched"),
            "is_error": .bool(true),
            "dispatch_attempted": .bool(false),
            "refusal": .object([
                "code": .string("POINTER_PRE_CAPTURE_FAILED"),
                "reason": .string("before_click_evidence_unavailable"),
            ]),
            "instruction": .string(
                "No click was sent. Take a fresh screenshot. If the same visible control is still present, send \(VisionCaptureToolDefinitions.visualClickCall) again with its current coordinates. Do not claim success."),
        ]
        return NavigationOutcome(
            content: try encodeOutcomeBody(body),
            recoverableColdMissArguments: nil,
            progressed: false,
            successfulReadOnlyObservation: false)
    }

    /// Driver-lane requests that VisionCapture refuses while a pointer task is live.
    private static func isDriverLaneMutation(_ request: String) -> Bool {
        if request.hasPrefix("swipe ") || request.hasPrefix("tap ") { return true }
        return ["tap coordinates", "go back", "press system alert button", "inspect cache",
                "tap cached action", "execute cached action", "revalidate cached action",
                "execute observed action"].contains(request)
    }

    private func closeComputerUseTask(
        _ task: ComputerUseTaskIdentity,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async {
        let arguments = makeMCPArguments(
            request: "hide pointer",
            parameters: [
                "computer_use_task_id": .string(task.id),
                "computer_use_generation": .integer(Int64(task.generation)),
            ],
            configuration: configuration,
            includeFlowSession: false)
        _ = try? await executeHostRequest(
            arguments,
            configuration: configuration,
            usesFlowSession: false,
            activity: activity)
    }

    /// The first text that reads like a validation message, or nil.
    static func validationMessage(in texts: [String]) -> String? {
        let markers = ["not a", "invalid", "required", "error", "failed", "must be"]
        return texts.first { text in
            markers.contains { text.range(of: $0, options: .caseInsensitive) != nil }
        }
    }

    /// One plain screenshot with OCR: its text boxes, or why there are none. Only
    /// read errors that end the run are thrown; after an action, an alert refusal
    /// becomes the note so the action result is kept.
    private func readScreenText(
        afterAction: Bool = false,
        configuration: VisionCaptureAgentConfiguration, activity: @escaping Activity
    ) async throws -> (blocks: [ScreenTextBlock], unavailableReason: String?) {
        if afterAction { autoScreenImage = nil }
        let ocrArguments = makeMCPArguments(
            request: "take a screenshot", parameters: ["include_ocr": .bool(true)],
            configuration: configuration, includeFlowSession: false)
        do {
            let ocrResult = try await executeHostRequest(
                ocrArguments, configuration: configuration, activity: activity)
            guard !ocrResult.isError else {
                return ([], Self.screenTextRefusalReason(
                    code: ocrResult.refusalCode, reason: ocrResult.serverOutcome.reason))
            }
            let screenText = Self.screenTextBlocks(in: ocrResult.value)
            if afterAction { keepAutoScreenImage(ocrResult, blocks: screenText.blocks, configuration: configuration) }
            return screenText
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return try Self.screenTextReadFailure(error, afterAction: afterAction)
        }
    }

    /// Keeps the after-action OCR capture's image, the same capture as its words,
    /// when the setting allows it and the read found words. It joins the next input
    /// only if a word survives the lean rule. It is never coordinate evidence:
    /// tap_coordinates still needs the model's own screenshot, and the screen-text
    /// choices tap exact positions anyway.
    private func keepAutoScreenImage(
        _ result: VisionCaptureMCPResult, blocks: [ScreenTextBlock],
        configuration: VisionCaptureAgentConfiguration
    ) {
        // Without the vision pack an image would stop the model turn.
        guard configuration.autoScreenImage, !blocks.isEmpty, visionPackComplete(configuration) else { return }
        autoScreenImage = try? VisionCaptureScreenshot.stage(
            result, in: screenshotStore, expectedDeviceID: configuration.simulatorUDID)
    }

    /// Kept screen-text words and IDs belong to the model context that saw them.
    private func forgetKeptScreenText() {
        lastScreenText = nil
        lastScreenTextIDs.removeAll()
    }

    private func visionPackComplete(_ configuration: VisionCaptureAgentConfiguration) -> Bool {
        #if DEBUG
        if let visionPackCompleteForTesting { return visionPackCompleteForTesting }
        #endif
        return AppVisionPackInstallationProbe.status(at: configuration.modelDirectory) == .complete
    }

    static let autoScreenImageNote = "The attached image shows the screen after your action. Tap words by their screen-text choice IDs; for a tap by position, take a screenshot first."

    /// The packet text for the next input. With the auto image it gets the note.
    private func attachingAutoScreenImage(
        to packet: DecisionPacket, images: inout [AppImageAttachment]
    ) throws -> String {
        guard let image = takeAutoScreenImage(for: packet) else { return packet.content }
        images.append(image)
        return try Self.addingGuidanceNote(to: packet.content, note: Self.autoScreenImageNote)
    }

    /// The kept capture for the input of `packet`, if a screen-text word survived;
    /// the slot is cleared either way.
    private func takeAutoScreenImage(for packet: DecisionPacket) -> AppImageAttachment? {
        defer { autoScreenImage = nil }
        return packet.offersScreenText ? autoScreenImage : nil
    }

    /// After a dispatched action (not type): one OCR read when the screen changed
    /// and a current read follows, its words into the packet body. The capture's
    /// image is kept for the next input (keepAutoScreenImage).
    private func addingScreenTextAfterAction(
        _ outcome: NavigationOutcome, signatureBefore: String?,
        configuration: VisionCaptureAgentConfiguration, activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        guard var body = try JSONDecoder().decode(JSONValue.self, from: Data(outcome.content.utf8)).objectValue,
              Self.readsScreenTextAfterAction(
                body, signatureBefore: signatureBefore, signatureAfter: currentScreenSignature) else { return outcome }
        Self.recordScreenText(
            try await readScreenText(afterAction: true, configuration: configuration, activity: activity),
            into: &body)
        return NavigationOutcome(
            content: try JSONValue.object(body).encoded(),
            recoverableColdMissArguments: outcome.recoverableColdMissArguments,
            progressed: outcome.progressed,
            successfulReadOnlyObservation: outcome.successfulReadOnlyObservation,
            imageAttachments: outcome.imageAttachments)
    }

    /// Keep the mutation's proof separate from the following read. Neither a
    /// successful request nor fresh screen facts can upgrade an inconclusive action.
    private func observeAfterMutation(
        _ intent: NavigationIntent,
        result: VisionCaptureMCPResult,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity,
        fallback: (() throws -> NavigationOutcome)? = nil,
        actionArguments: JSONValue? = nil
    ) async throws -> NavigationOutcome {
        let factsBeforeAction = currentScreenFacts ?? lastInvalidatedScreenFacts
        let signatureBeforeAction = currentScreenSignature ?? lastInvalidatedScreenSignature
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
        let actionName: String
        switch intent.operation {
        case .swipe: actionName = "swipe"
        case .tap: actionName = "tap"
        case .tapCoordinates: actionName = "coordinate tap"
        default: actionName = "Computer Use click"
        }
        let isTap = intent.operation == .tap || intent.operation == .tapCoordinates
        currentManifest = AuthorityManifest()
        currentSystemAlert = nil
        staleActionConfirmation = nil
        invalidateScreenObservation()
        var afterScreens: [[String: JSONValue]] = []
        try? Self.collectStructuredObjects(named: "after_screen", in: result.value) { afterScreens.append($0) }
        let refreshed: PreparedNavigation
        if Self.fastHostPath, !result.isError, !afterScreens.isEmpty {
            // VisionCapture already settled and read the screen after the action.
            do {
                try recordScreenObservation(from: result.value)
            } catch {
                invalidateScreenObservation()
            }
            let manifest = AuthorityManifest(state: currentScreenSignature == nil ? "unavailable" : "ready")
            currentManifest = manifest
            refreshed = PreparedNavigation(
                result: result, arguments: actionArguments ?? .object([:]), manifest: manifest,
                systemAlert: nil, screenFacts: Self.returnedScreenFacts(from: result.value))
        } else {
        if isTap {
            // Let the screen settle before the read. Cheaper than a model step.
            try await Task.sleep(for: .milliseconds(400))
        }
        do {
            refreshed = try await refreshNavigation(configuration: configuration, activity: activity)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            currentManifest = AuthorityManifest()
            currentSystemAlert = nil
            invalidateScreenObservation()
            if let fallback {
                // The tap itself was delivered. Return its own result and let
                // the model observe, as before. Do not end the chat.
                return try fallback()
            }
            throw VisionCaptureAgentError.noProgress(
                "The \(actionName) request returned, with proof \(canonicalVerdict). Its following observation failed: \(error). The input was not repeated.")
        }
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
                "The \(actionName) follow-up observation could not be retained. The input was not repeated.")
        }
        // A cold or refreshed inspection describes its own dispatch. Do not
        // attribute those read flags or proof to the preceding gesture.
        for key in ["proof", "dispatch_attempted", "submission_started", "delivery_acknowledged", "delivery_unknown"] {
            body.removeValue(forKey: key)
        }
        body["operation"] = .string(intent.operation.rawValue)
        if let direction = intent.direction {
            body["direction"] = .string(direction.rawValue)
        }
        if let proof { body["proof"] = .object(proof) }
        body.merge(delivery) { _, returned in returned }
        // The pointer lane never carries a driver acknowledgement: a submitted
        // click is its delivered form. Do not treat it as delivery unknown.
        let pointerDelivered = actionArguments?.objectValue?["request"] == .string("click pointer")
            && delivery["submission_started"] == .bool(true)
        if pointerDelivered {
            body["delivery_acknowledged"] = .bool(true)
            body.removeValue(forKey: "delivery_unknown")
        }
        if let attempted = result.dispatchAttempted { body["dispatch_attempted"] = .bool(attempted) }
        let verdict = proof?["verdict"]
        body["outcome"] = verdict == .string("verified") ? .string("succeeded")
            : verdict ?? .string("unknown")
        body["observation_outcome"] = .string(refreshed.result.isError ? "unavailable" : "succeeded")
        // VisionCapture compares the screen before and after the action.
        // Its one-line summary is the model's evidence of effect.
        var changes: [[String: JSONValue]] = []
        try? Self.collectStructuredObjects(named: "changes", in: result.value) { changes.append($0) }
        if let summary = changes.first?["summary"], case .string(let text) = summary, !text.isEmpty {
            body["effect"] = .string(text)
        } else if actionArguments?.objectValue?["request"] == .string("click pointer"),
                  let before = factsBeforeAction?.readableTexts,
                  let after = refreshed.screenFacts?.readableTexts {
            // The pointer lane returns no summary. Derive the effect from the
            // texts of the reads before and after the click.
            let appeared = after.filter { !before.contains($0) }.prefix(8)
            let disappeared = before.filter { !after.contains($0) }.prefix(8)
            var parts: [String] = []
            if !appeared.isEmpty { parts.append("appeared: " + appeared.joined(separator: ", ")) }
            if !disappeared.isEmpty { parts.append("disappeared: " + disappeared.joined(separator: ", ")) }
            if !parts.isEmpty { body["effect"] = .string(parts.joined(separator: "; ")) }
            if let message = Self.validationMessage(in: Array(appeared)) {
                body["warning"] = .string(
                    "The screen shows: \(message). The input was not accepted; correct it before submitting.")
            }
        }
        if let changed = changes.first?["screen_changed"], case .bool = changed {
            body["screen_changed"] = changed
        } else if actionArguments?.objectValue?["request"] == .string("click pointer") {
            // A pointer click returns no after_screen or changes. Its own
            // screen observation says whether the screen changed.
            var statuses: Set<String> = []
            try? Self.collectStructuredObjects(named: "interaction_evidence", in: result.value) {
                if case .string(let status)? = $0["screen_observation"]?.objectValue?["status"] {
                    statuses.insert(status)
                }
            }
            if statuses == ["changed_observed"] {
                body["screen_changed"] = .bool(true)
            } else if statuses == ["no_persistent_change_observed"] {
                body["screen_changed"] = .bool(false)
            }
        }
        // A changed screen gets one plain OCR read: its words become tap choices
        // in the next packet, with the capture's image if a word is offered.
        if Self.readsScreenTextAfterAction(
            body, signatureBefore: signatureBeforeAction, signatureAfter: currentScreenSignature) {
            Self.recordScreenText(
                try await readScreenText(afterAction: true, configuration: configuration, activity: activity),
                into: &body)
        }
        body["instruction"] = .string(delivery["delivery_unknown"] == .bool(true) && !pointerDelivered
            ? "\(actionName) delivery remains unknown. It was not repeated. These facts come from a separate read. Do not replay the input."
            : "These facts come from the read after the \(actionName). Use its canonical verdict without assuming the intended effect. Choose again from the fresh actions; old IDs have expired.")
        if verdict != .string("verified") {
            body["outcome_note"] = .string(
                "The \(actionName) effect was not verified. The following observation does not prove that effect.")
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

    /// A native iOS alert can own the foreground while the requested app is
    /// alive behind it. Read that alert before asking the model for another
    /// action. If no alert remains, make one plain app read instead.
    private func observeAfterLaunchNotForeground(
        _ launch: [String: JSONValue],
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        discardReturnedExecutionEvidence()
        let alertRead = try await describeSystemAlert(
            configuration: configuration,
            activity: activity)
        let observed: PreparedNavigation
        if alertRead.systemAlert != nil {
            observed = alertRead
        } else {
            observed = try await describeScreenAfterCacheValidationFailure(
                configuration: configuration,
                activity: activity)
        }
        guard !observed.result.isError,
              observed.systemAlert != nil || currentScreenSignature != nil else {
            discardReturnedExecutionEvidence()
            throw VisionCaptureAgentError.noProgress(
                "The requested app is running but was not in front. The one read-only recovery found neither a native system alert nor current app content. No launch or input was replayed.")
        }

        let current = try outcome(
            for: NavigationIntent(
                operation: .observe, selector: nil, selectorKind: nil,
                role: nil, desiredState: nil, text: nil),
            result: observed.result,
            arguments: observed.arguments,
            manifest: observed.manifest,
            systemAlert: observed.systemAlert,
            progressed: false,
            observedScreenFacts: observed.screenFacts)
        guard case .object(var body) = try JSONDecoder().decode(
            JSONValue.self, from: Data(current.content.utf8)) else {
            throw VisionCaptureAgentError.noProgress(
                "The launch recovery observation could not be retained.")
        }
        body["operation"] = .string("launch")
        body["outcome"] = .string("running_not_foreground_reobserved")
        body["launch"] = .object(launch)
        body["instruction"] = .string(observed.systemAlert != nil
            ? "The requested app is running behind this native iOS system alert. Choose one exact current alert button. Do not relaunch."
            : "The launch did not prove foreground presentation. A read-only follow-up found current app content. Continue from the current choices. Do not relaunch.")
        return NavigationOutcome(
            content: try JSONValue.object(body).encoded(),
            recoverableColdMissArguments: nil,
            progressed: false,
            successfulReadOnlyObservation: true)
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

    /// Accept only the exact launch failure that proves the requested process
    /// is alive while a system or other process owns the foreground.
    private static func isolatedLaunchNotForeground(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> [String: JSONValue]? {
        guard result.isError,
              result.refusalCode == "APP_RUNNING_NOT_FOREGROUND",
              let request = arguments.objectValue,
              request["request"] == .string("launch app"),
              request["bundle_id"] == .string(configuration.bundleIdentifier),
              request["parameters"] == .object([
                  "udid": .string(configuration.simulatorUDID),
              ]),
              let root = result.value.objectValue,
              let launch = root["launch_outcome"]?.objectValue,
              launch["verdict"] == .string("running_not_foreground"),
              launch["reason_code"] == .string("APP_RUNNING_NOT_FOREGROUND"),
              launch["mutation_sent"] == .bool(true),
              launch["requested_udid"] == .string(configuration.simulatorUDID),
              launch["bound_udid"] == .string(configuration.simulatorUDID),
              launch["device_readiness"] == .string("ready"),
              let process = launch["process"]?.objectValue,
              process["bundle_id"] == .string(configuration.bundleIdentifier),
              process["state"] == .string("running"),
              case .integer(let pid)? = process["pid"], pid > 0,
              let foreground = launch["foreground"]?.objectValue,
              foreground["state"] == .string("system")
                || foreground["state"] == .string("other"),
              case .string(let foregroundBundle)? = foreground["bundle_id"],
              !foregroundBundle.isEmpty,
              foregroundBundle != configuration.bundleIdentifier,
              case .array(let content)? = root["content"] else {
            return nil
        }
        let errors = content.compactMap { $0.objectValue?["text"] }.compactMap {
            value -> String? in
            guard case .string(let text) = value,
                  text.hasPrefix("Error [") else { return nil }
            return text
        }
        guard !errors.isEmpty,
              errors.allSatisfy({
                  $0.hasPrefix("Error [APP_RUNNING_NOT_FOREGROUND]:")
              }) else {
            return nil
        }
        do {
            var consistent = true
            try collectStructuredObjects(named: "launch_outcome", in: result.value) {
                if $0 != launch { consistent = false }
            }
            return consistent ? launch : nil
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

    /// Conservative admission only. Existing exact native token/image planning
    /// remains the final runtime gate. Missing context evidence disables this
    /// optional host support, not ordinary model-requested screenshots.
    private static func hasVisualRecoveryCapacity(
        maxContextTokens: Int?, retainedTokens: Int?, packetBytes: Int,
        packetCopies: Int = 1
    ) -> Bool {
        guard let maxContextTokens, let retainedTokens,
              maxContextTokens > 0, retainedTokens >= 0, retainedTokens <= maxContextTokens,
              packetBytes >= 0, packetCopies > 0 else { return false }
        let remaining = maxContextTokens - retainedTokens
        let reserve = AppModel.reservedPromptTokens + VisionImageTokenBudget.maximumTokensPerImage
        guard remaining > reserve else { return false }
        // Budget two tokens per UTF-8 byte for possible escaping, plus
        // the existing prompt envelope. Before capture reserve a second packet
        // of the current size; after capture check the actual combined packet.
        return packetBytes < (remaining - reserve) / 2 / packetCopies
    }

    private func canProvideVisualRecovery(
        for intent: NavigationIntent, outcome: NavigationOutcome,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard intent.operation == .observe, outcome.imageAttachments.isEmpty,
              outcome.recoverableColdMissArguments == nil,
              let body = try? JSONDecoder().decode(JSONValue.self,
                  from: Data(outcome.content.utf8)).objectValue,
              body["outcome"] == .string("succeeded"),
              body["delivery_unknown"] != .bool(true), body["is_error"] != .bool(true),
              body["refusal"] == nil, body["observation_refusal"] == nil,
              body["stale_recovery"] == nil else { return false }
        return canProvideVisualRecovery(configuration: configuration)
    }

    private func canProvideVisualRecovery(configuration: VisionCaptureAgentConfiguration) -> Bool {
        guard !requiresReadOnlyRecovery, uncertainAlertPress == nil,
              currentSystemAlert == nil, staleActionConfirmation == nil,
              currentScreenContentIdentity != nil,
              permittedNextOperations?.contains(.screenshot) == true else { return false }
        return AppVisionPackInstallationProbe.status(at: configuration.modelDirectory) == .complete
    }

    private func canProvideRejectedTargetVisualRecovery(
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard (try? configuration.validate()) != nil,
              committedTargetKey == configuration.targetKey, hasValidatedHostContract,
              let signature = currentScreenSignature,
              let scope = currentNavigationScreenScope,
              !blockedStaleActions.contains(where: { $0.screenSignature == signature }),
              !rejectedBeforeSubmissionProposals.contains(where: { $0.screen == scope }) else {
            return false
        }
        return canProvideVisualRecovery(configuration: configuration)
    }

    enum VisualRecoveryBoundary {
        case repeatedRead
        case repeatedTargetRejection
        case repeatedThought
        case coordinateWithoutImage

        var description: String {
            switch self {
            case .repeatedRead: "the repeated-read correction"
            case .repeatedTargetRejection: "the second equivalent expired or unavailable target rejection"
            case .repeatedThought: "discarding an unfinished model response with sustained exact thought repetition"
            case .coordinateWithoutImage: "a coordinate action without current screenshot evidence"
            }
        }
    }

    private func recoverRepeatedGeneration(
        boundary: GenerationRetryBoundary, receipt: ThoughtRepetitionRecovery,
        configuration: VisionCaptureAgentConfiguration, maxContextTokens: Int?,
        activity: @escaping Activity
    ) async throws -> GenerationRetryBoundary {
        try Task.checkCancellation()
        guard boundary.session == committedSessionIdentity,
              let body = try JSONDecoder().decode(JSONValue.self,
                  from: Data(boundary.outcome.utf8)).objectValue,
              body["outcome"] == .string("succeeded"),
              body["delivery_unknown"] != .bool(true), body["is_error"] != .bool(true),
              body["refusal"] == nil, body["observation_refusal"] == nil,
              body["stale_recovery"] == nil else {
            // Only the outcome was not a success (inconclusive, not sent,
            // failed without delivery doubt): one plain read, then choose again.
            if boundary.session == committedSessionIdentity,
               let body = try JSONDecoder().decode(JSONValue.self,
                   from: Data(boundary.outcome.utf8)).objectValue,
               body["outcome"] != .string("succeeded"),
               body["delivery_unknown"] != .bool(true), body["is_error"] != .bool(true),
               body["refusal"] == nil, body["observation_refusal"] == nil,
               body["stale_recovery"] == nil,
               uncertainAlertPress == nil, staleActionConfirmation == nil {
                let observed = try await perform(
                    NavigationIntent(operation: .observe, selector: nil, selectorKind: nil,
                                     role: nil, desiredState: nil, text: nil),
                    readyOffer: nil, configuration: configuration, activity: activity)
                try Task.checkCancellation()
                guard boundary.session == committedSessionIdentity,
                      var packet = try JSONDecoder().decode(JSONValue.self, from: Data(
                          try decisionPacket(from: observed.content, call: boundary.call,
                              configuration: configuration, images: []).content.utf8)).objectValue else {
                    throw VisionCaptureAgentError.noProgress(
                        "Thinking recovery did not obtain a current read in the same session. No action was replayed.")
                }
                let note = Self.repeatedThinkingReadNote(outcome: body)
                if case .string(let guidance)? = packet["guidance"] {
                    packet["guidance"] = .string(note + " " + guidance)
                } else {
                    packet["guidance"] = .string(note)
                }
                let result = AppToolResult(callID: boundary.result.callID, name: boundary.result.name,
                    content: try JSONValue.object(packet).encoded())
                return GenerationRetryBoundary(call: boundary.call, result: result,
                    outcome: observed.content, session: committedSessionIdentity)
            }
            throw VisionCaptureAgentError.noProgress(
                "Repeated-thinking recovery has no permitted current screenshot/read route, or an earlier refusal, uncertainty or session change forbids it. No action was replayed.")
        }
        if !requiresReadOnlyRecovery, uncertainAlertPress == nil,
           currentSystemAlert == nil, staleActionConfirmation == nil,
           currentImageObservation == observationGeneration,
           let content = try Self.repeatedThinkingRetryContent(
               from: boundary.result.content,
               imageCount: boundary.result.imageAttachments.count) {
            guard Self.hasVisualRecoveryCapacity(
                maxContextTokens: maxContextTokens,
                retainedTokens: receipt.restoredTokenCount,
                packetBytes: content.utf8.count
            ) else {
                throw VisionCaptureAgentError.noProgress(
                    "Thinking recovery has insufficient context capacity to reuse the current image and packet. No action was replayed.")
            }
            let result = AppToolResult(
                callID: boundary.result.callID, name: boundary.result.name,
                content: content, imageAttachments: boundary.result.imageAttachments)
            return GenerationRetryBoundary(
                call: boundary.call, result: result,
                outcome: boundary.outcome, session: boundary.session)
        }
        guard canProvideRejectedTargetVisualRecovery(configuration: configuration) else {
            throw VisionCaptureAgentError.noProgress(
                "Repeated-thinking recovery has no permitted current screenshot/read route, or an earlier refusal, uncertainty or session change forbids it. No action was replayed.")
        }
        let oldImages = boundary.result.imageAttachments
        // Reserve all not-yet-committed images as well as the additional image.
        guard let maxContextTokens,
              oldImages.count < VisionImageTokenBudget.capacity(maxContext: maxContextTokens,
                  reservedTextTokens: receipt.restoredTokenCount) else {
            throw VisionCaptureAgentError.noProgress("Pending images leave no context capacity for a new recovery image.")
        }
        let retained = receipt.restoredTokenCount
            + oldImages.count * VisionImageTokenBudget.maximumTokensPerImage
        guard Self.hasVisualRecoveryCapacity(maxContextTokens: maxContextTokens,
            retainedTokens: retained, packetBytes: boundary.result.content.utf8.count,
            packetCopies: 2) else {
            throw VisionCaptureAgentError.noProgress(
                "Thinking recovery has insufficient context capacity for a new image and current packet. No action was replayed.")
        }
        try retirePendingConfirmation()
        let support = try await provideVisualRecovery(
            originalPacket: boundary.result.content, originalOutcome: boundary.outcome,
            call: boundary.call, boundary: .repeatedThought, configuration: configuration,
            maxContextTokens: maxContextTokens, retainedTokens: retained, activity: activity)
        try Task.checkCancellation()
        guard boundary.session == committedSessionIdentity, support.images.count == 1,
              var packet = try JSONDecoder().decode(JSONValue.self,
                  from: Data(support.content.utf8)).objectValue else {
            throw VisionCaptureAgentError.noProgress(
                "Thinking recovery did not obtain one admitted current image/read pair in the same session. No action was replayed.")
        }
        packet["generation_recovery"] = .string(
            "The unfinished repeated response was discarded. No call from it was executed. Use this fresh evidence to choose one permitted action now. Do not restate or compare the evidence. If no action can safely advance the goal, return one concise blocker. Earlier app actions retain their original outcomes and must not be repeated.")
        packet["image_attachment_order"] = .string(oldImages.isEmpty
            ? "The single attached image is from the additional current screenshot/read pair."
            : "The first \(oldImages.count) attached image(s) are historical images from the original pending result. The final image, number \(oldImages.count + 1), belongs to the additional current screenshot/read pair. Earlier images do not establish current targets.")
        let content = try JSONValue.object(packet).encoded()
        guard Self.hasVisualRecoveryCapacity(maxContextTokens: maxContextTokens,
            retainedTokens: retained, packetBytes: content.utf8.count) else {
            throw VisionCaptureAgentError.noProgress(
                "The complete thinking-recovery packet exceeded the conservative capacity allowance. No action was replayed.")
        }
        let result = AppToolResult(callID: boundary.result.callID, name: boundary.result.name,
            content: content, imageAttachments: oldImages + support.images)
        return GenerationRetryBoundary(call: boundary.call, result: result,
            outcome: support.outcome, session: committedSessionIdentity)
    }

    static func repeatedThinkingRetryContent(
        from content: String, imageCount: Int
    ) throws -> String? {
        guard imageCount == 1,
              var packet = try JSONDecoder().decode(
                  JSONValue.self, from: Data(content.utf8)).objectValue,
              case .object(let observation)? = packet["observation"],
              observation["current_image_evidence"] == .bool(true),
              observation["state"] == .string("current"),
              case .object(let lastAction)? = packet["last_action"],
              lastAction["action"] == .string("screenshot"),
              lastAction["verdict"] == .string("observed") else {
            return nil
        }
        packet["generation_recovery"] = .string(
            "The unfinished repeated response was discarded. No call from it was executed. Use this still-current screenshot/read pair to choose one permitted action now. Do not restate or compare the evidence. If this part of the goal cannot advance with these choices, record it as a blocker and continue with the next unfinished part of the goal. Earlier app actions retain their original outcomes and must not be repeated.")
        packet["image_attachment_order"] = .string(
            "The single attached image is the current screenshot/read pair from the interrupted decision.")
        return try JSONValue.object(packet).encoded()
    }

    private func provideVisualRecovery(
        originalPacket: String, originalOutcome: String,
        call: AppToolCall, boundary: VisualRecoveryBoundary,
        configuration: VisionCaptureAgentConfiguration,
        maxContextTokens: Int?, retainedTokens: Int?, excludingTargetID: String? = nil,
        activity: @escaping Activity
    ) async throws -> (content: String, images: [AppImageAttachment], outcome: String) {
        try Task.checkCancellation()
        let visual = try await screenshotAndObserve(configuration: configuration, activity: activity)
        try Task.checkCancellation()
        let visualPacket = try decisionPacket(from: visual.content,
            call: call, configuration: configuration, images: visual.imageAttachments,
            excludingTargetID: excludingTargetID)
        var content = try Self.visualRecoveryContent(originalPacket: originalPacket,
            visualPacket: visualPacket.content, boundary: boundary)
        var images = visual.imageAttachments
        if !Self.hasVisualRecoveryCapacity(maxContextTokens: maxContextTokens,
            retainedTokens: retainedTokens, packetBytes: content.utf8.count) {
            // Preserve the captured receipt, but retire image-only bindings if
            // the completed pair cannot fit the conservative context allowance.
            let textPacket = try decisionPacket(from: visual.content,
                call: call, configuration: configuration, images: [],
                excludingTargetID: excludingTargetID)
            content = try Self.visualRecoveryContent(originalPacket: originalPacket,
                visualPacket: textPacket.content, boundary: boundary, imageProvided: false)
            images = []
        }
        let outcome = try Self.recordingVisualRecovery(originalOutcome: originalOutcome,
            visualOutcome: visual.content, imageProvided: !images.isEmpty)
        return (content, images, outcome)
    }

    /// The model requested the original action. The host supplies this extra
    /// image/read only at the existing correction boundary, not as navigation.
    static func visualRecoveryContent(
        originalPacket: String, visualPacket: String,
        boundary: VisualRecoveryBoundary, imageProvided: Bool = true
    ) throws -> String {
        guard let original = try JSONDecoder().decode(JSONValue.self,
                  from: Data(originalPacket.utf8)).objectValue,
              var visual = try JSONDecoder().decode(JSONValue.self,
                  from: Data(visualPacket.utf8)).objectValue,
              let lastAction = original["last_action"],
              case .string(let priorGuidance)? = original["guidance"],
              case .string(let imageGuidance)? = visual["guidance"],
              visual["observation"]?.objectValue?["current_image_evidence"] == .bool(imageProvided) else {
            throw VisionCaptureAgentError.noProgress(
                "The host's one visual recovery attempt could not retain its original action and current image/read evidence. No input was replayed.")
        }
        visual["last_action"] = lastAction
        if !imageProvided {
            visual.removeValue(forKey: "image")
            if var observation = visual["observation"]?.objectValue {
                observation.removeValue(forKey: "image_relationship")
                visual["observation"] = .object(observation)
            }
        }
        let imageStatus = imageProvided
            ? "The image and current choices below come from this additional pair."
            : "The screenshot was captured but was not provided to the model because the conservative context allowance was insufficient. Current facts and choices come from the following read without image input. No current visual evidence is available to the model."
        visual["guidance"] = .string(
            "The host made one additional screenshot and following read after \(boundary.description). This was host observation support, not a navigation action chosen by the model. The original action result is unchanged. " + imageStatus + "\n"
            + "Correction for the original proposal: "
            + correctionListingCurrentActions(
                boundary == .coordinateWithoutImage && imageProvided
                    ? coordinateImageSupportCorrection(priorGuidance) : priorGuidance,
                allowedNext: visual["allowed_next"]) + "\n"
            + "Current image/read: " + imageGuidance)
        return try JSONValue.object(visual).encoded()
    }

    /// True when a published position lies within 10 normalized units (about 4 points
    /// across, 9 points down) of an OCR-named control.
    static func isOCRDuplicate(position: JSONValue?, ocrPositions: [(x: Int64, y: Int64)]) -> Bool {
        guard case .object(let point)? = position,
              case .integer(let x)? = point["x_norm"], case .integer(let y)? = point["y_norm"] else { return false }
        return ocrPositions.contains { abs($0.x - x) <= 10 && abs($0.y - y) <= 10 }
    }

    static let busyHostReadInstruction =
        "VisionCapture was still busy with its own read; nothing was sent. Choose again from these fresh choices."

    /// VisionCapture refused because a screen read it started still held the device (for
    /// example inside the loop's own type-at-position request), and nothing was dispatched.
    static func isBusyHostReadBeforeDispatch(_ result: VisionCaptureMCPResult) -> Bool {
        guard result.isError, result.dispatchAttempted != true,
              !result.hasConflictingDispatchAttemptEvidence,
              let reason = result.serverOutcome.reason,
              reason.range(of: #"is already running describe work for mcp request [0-9A-Fa-f-]{36}"#,
                           options: .regularExpression) != nil else { return false }
        var dispatches: [[String: JSONValue]] = []
        try? collectStructuredObjects(named: "dispatch", in: result.value) { dispatches.append($0) }
        return dispatches.allSatisfy {
            $0["submission_started"] != .bool(true) && $0["delivery_acknowledged"] != .bool(true)
        }
    }

    static func ambiguousTargetInstruction(isTyping: Bool) -> String {
        isTyping
            ? "Several fields match this name; tap the field at its position, then type without a target."
            : "Two or more controls share that name; choose one by its position."
    }

    static func coordinateImageSupportCorrection(_ guidance: String) -> String {
        guidance.replacingOccurrences(of: coordinateEvidenceReason,
            with: "Coordinate actions need current screenshot evidence; the loop attached a current screenshot.")
            .replacingOccurrences(of: " Use only the current choices.", with: "")
    }

    static func repeatedThinkingReadNote(outcome: [String: JSONValue]) -> String {
        outcome["screen_changed"] == .bool(false)
            ? "Your last actions had no visible effect. Read the facts again and choose a different action."
            : "Read the current facts again and choose the next action."
    }

    /// The correction was written for the earlier read. List the actions the new read permits.
    static func correctionListingCurrentActions(_ guidance: String, allowedNext: JSONValue?) -> String {
        guard case .array(let allowed)? = allowedNext,
              let range = guidance.range(of: #"Allowed actions now: \[[^\]]*\]\."#,
                                         options: .regularExpression) else { return guidance }
        let actions = allowed.compactMap { value -> String? in
            guard case .string(let action) = value else { return nil }
            return action
        }.joined(separator: ", ")
        var updated = guidance
        updated.replaceSubrange(range, with: "Allowed actions now: [\(actions)].")
        return updated
    }

    private static func recordingVisualRecovery(
        originalOutcome: String, visualOutcome: String, imageProvided: Bool
    ) throws -> String {
        guard var original = try JSONDecoder().decode(JSONValue.self,
                  from: Data(originalOutcome.utf8)).objectValue,
              let visual = try JSONDecoder().decode(JSONValue.self,
                  from: Data(visualOutcome.utf8)).objectValue else {
            throw VisionCaptureAgentError.malformedCall("The visual recovery audit could not be retained.")
        }
        // Original outcome keys and proof remain unchanged. The existing host
        // audit retains the complete extra observation beside that outcome.
        let support: JSONValue = .object([
            "origin": .string("host_read_only_visual_recovery"),
            "image_provided_to_model": .bool(imageProvided),
            "result": .object(visual),
        ])
        if let previous = original["host_observation_support"] {
            if case .array(let history) = previous {
                original["host_observation_support"] = .array(history + [support])
            } else {
                original["host_observation_support"] = .array([previous, support])
            }
        } else {
            original["host_observation_support"] = support
        }
        return try JSONValue.object(original).encoded()
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
                imageMetadata: imageMetadata,
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
            let unavailable = prepared.result.isError
                || (currentScreenSignature == nil && currentSystemAlert == nil)
                || currentScreenObservationMetadataInvalid
            let pairState = Self.imageReadState(
                image: imageMetadata, read: readMetadata,
                expectedDeviceID: configuration.simulatorUDID, readUnavailable: unavailable)
            relationship["state"] = .string(pairState)
            if observationGrant == nil, !inspectResult.isError {
                relationship["order"] = .string("image_then_accessibility_then_cache_validation")
            }
            guard pairState == "sequential" else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "The image and subsequent read did not establish a current pair: \(pairState).")
            }
            body["observation_outcome"] = .string("succeeded")
            body["instruction"] = .string(Self.screenshotPairInstruction)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            currentManifest = AuthorityManifest()
            currentSystemAlert = nil
            invalidateScreenObservation()
            // The screenshot activity above has already retained its display
            // receipt. End this unsupported pair instead of offering a loop
            // through observe that would immediately retire the image again.
            throw VisionCaptureAgentError.noProgress(
                "The screenshot was captured, but its following read could not establish a usable current image and target binding. No input was sent. Agent Mode stopped this unsupported observation path instead of repeating screenshot and observe.")
        }
        // One more plain screenshot, with OCR. Only its text boxes are used;
        // the image above stays the model's image.
        Self.recordScreenText(try await readScreenText(configuration: configuration, activity: activity),
            into: &body)
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

    /// One post-image read. A warm cache then needs one fresh cache validation:
    /// the handles from before the image are never restored. A topology change
    /// cannot recursively refresh the pair or carry it into a later decision.
    private func readAfterScreenshot(
        observationGrant: String?,
        inspectResult: VisionCaptureMCPResult,
        inspectArguments: JSONValue,
        imageMetadata: VisionCaptureScreenshot.ObservationMetadata,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> PreparedNavigation {
        // A screenshot can show a native iOS alert while the app accessibility
        // read still describes content behind it. Check the dedicated alert
        // lane first. This read does not consume a Discovery observation grant.
        let alertRead = try await describeSystemAlert(
            configuration: configuration,
            activity: activity)
        if alertRead.systemAlert != nil {
            return alertRead
        }
        guard let observationGrant else {
            if Self.isRecoverableInspectCacheBoundary(inspectResult, arguments: inspectArguments) {
                // Preserve the existing native-alert route. This exclusive
                // read must pass the same pair checks before offering input.
                return try await describeSystemAlert(configuration: configuration, activity: activity)
            }
            guard !inspectResult.isError else {
                throw VisionCaptureAgentError.mcpOutcome(inspectResult.serverOutcome)
            }
            // A warm manifest has no read grant. Establish the image/read pair
            // first, then acquire only post-read handles through the existing
            // cache validation. Do not let that validation replace the paired
            // screen identity or its metadata with a later observation.
            let observed = try await describeScreenAfterCacheValidationFailure(
                configuration: configuration, activity: activity)
            let pairedSignature = currentScreenSignature
            let pairedMetadata = currentScreenObservationMetadata
            guard Self.imageReadState(
                image: imageMetadata, read: pairedMetadata,
                expectedDeviceID: configuration.simulatorUDID,
                readUnavailable: observed.result.isError || pairedSignature == nil
                    || currentScreenObservationMetadataInvalid) == "sequential" else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "The warm-cache screenshot and fresh read did not establish a usable pair.")
            }
            if Self.fastHostPath {
                // One plain read pairs with the image. Choices come from its facts.
                let manifest = AuthorityManifest(state: "ready")
                currentManifest = manifest
                return PreparedNavigation(
                    result: observed.result, arguments: observed.arguments, manifest: manifest,
                    systemAlert: nil, screenFacts: Self.returnedScreenFacts(from: observed.result.value))
            }
            let refreshed = try await refreshNavigation(
                configuration: configuration, activity: activity,
                allowsObservationRefresh: false)
            // This is a retention check, not proof of atomic visual identity.
            // The new manifest's existing live validation owns action binding.
            guard !refreshed.result.isError, !refreshed.observationRefreshed,
                  refreshed.systemAlert == nil, currentScreenSignature == pairedSignature,
                  currentScreenObservationMetadata == pairedMetadata,
                  !currentScreenObservationMetadataInvalid else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "Fresh cache validation could not retain the screenshot's paired screen read.")
            }
            return refreshed
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
        if Self.fastHostPath {
            return try await plainDescribeNavigation(configuration: configuration, activity: activity)
        }
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

    /// Fast host path read: one plain describe. Choices come from the screen
    /// facts (tap candidates and editable fields), and taps go out by name.
    private func plainDescribeNavigation(
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
        let manifest = AuthorityManifest(state: result.isError ? "unavailable" : "ready")
        currentManifest = manifest
        return PreparedNavigation(
            result: result,
            arguments: arguments,
            manifest: manifest,
            systemAlert: nil,
            screenFacts: result.isError ? nil : Self.returnedScreenFacts(from: result.value))
    }

    /// Fast host path tap: one pointer click at the read position. Without a
    /// known position nothing is sent: a driver tap by name can match two controls.
    private func performDirectNameTap(
        _ intent: NavigationIntent,
        selector: String,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        checkingLocalProposal = true
        try rejectPreviouslyRejectedProposal(intent)
        try rejectExhaustedIntent(intent)
        let attemptedScreenScope = currentNavigationScreenScope
        let screenTextClick = Self.screenTextPointerClick(
            selectorKind: intent.selectorKind, selector: selector, xNorm: intent.xNorm, yNorm: intent.yNorm)
        guard let knownPosition = screenTextClick.map({ (x: $0.x, y: $0.y) })
            ?? currentScreenFacts?.normalizedPosition(
            selector: selector, role: intent.role ?? "", selectorKind: intent.selectorKind) else {
            // Not sent: a local rejection that keeps the current choices.
            throw VisionCaptureAgentError.navigationUnavailable(
                "This choice has no screen position. Take a screenshot, then send \(VisionCaptureToolDefinitions.coordinateTapCall) at the visible control.")
        }
        currentManifest = AuthorityManifest()
        invalidateScreenObservation()
        // Gemma's taps go through the pointer at the read position, not
        // through the driver's name or coordinate taps.
        let label = resolvedJourneyLabel?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let pointer = try await performFastPointerClick(
            x: knownPosition.x, y: knownPosition.y,
            label: screenTextClick?.intent ?? (label.isEmpty ? selector : label),
            configuration: configuration, activity: activity)
        let arguments = pointer.arguments
        let result = pointer.result
        let plainOutcome = { [self] in
            try outcome(
                for: intent, result: result, arguments: arguments,
                manifest: AuthorityManifest(), systemAlert: nil, progressed: false)
        }
        if result.isError {
            // Nothing was sent. Remember it so the same proposal is not repeated.
            rememberRejectedBeforeSubmissionProposal(intent, screen: attemptedScreenScope)
            return try plainOutcome()
        }
        try recordCompletedJourneyAction(intent, result: result.value)
        staleActionConfirmation = nil
        noteExecuted(intent, result: result)
        return try await observeAfterMutation(
            intent, result: result, configuration: configuration,
            activity: activity, fallback: plainOutcome, actionArguments: arguments)
    }

    private func isCurrentFastTapTarget(_ intent: NavigationIntent) -> Bool {
        guard let selector = intent.selector, let role = intent.role else { return false }
        if let offered = offeredTapCandidates, offered.signature == currentScreenSignature,
           offered.candidates.contains(where: {
               $0.selector.utf8.elementsEqual(selector.utf8) && $0.role.utf8.elementsEqual(role.utf8)
           }) { return true }
        return currentManifest.actions.contains {
            $0.action == "tap" && $0.selector.utf8.elementsEqual(selector.utf8)
                && $0.role.utf8.elementsEqual(role.utf8)
        }
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

    /// The observed target disappeared after Gemma chose it but before
    /// VisionCapture dispatched the tap. Keep the refusal and return only
    /// freshly read choices.
    private func observeAfterObservedTapTargetUnavailable(
        _ intent: NavigationIntent,
        failure: VisionCaptureServerOutcome,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity
    ) async throws -> NavigationOutcome {
        let observed = try await describeScreenAfterCacheValidationFailure(
            configuration: configuration, activity: activity)
        guard !observed.result.isError, currentScreenSignature != nil else {
            throw VisionCaptureAgentError.noProgress(
                "VisionCapture rejected the changed target before dispatch, and the following plain read could not establish current screen facts. No action was sent or replayed.")
        }
        let prepared = try await refreshNavigation(
            configuration: configuration, activity: activity)
        return try outcome(
            for: intent, result: prepared.result, arguments: prepared.arguments,
            manifest: prepared.manifest, systemAlert: prepared.systemAlert,
            progressed: false, observedScreenFacts: prepared.screenFacts,
            refusedObservedTapBeforeDispatch: failure,
            observationRefreshed: prepared.observationRefreshed)
    }

    /// VisionCapture refused a name shared by two or more controls before
    /// submission. Read once and return fresh choices; the chat continues.
    private func observeAfterAmbiguousTarget(
        _ intent: NavigationIntent,
        failure: VisionCaptureServerOutcome,
        configuration: VisionCaptureAgentConfiguration,
        activity: @escaping Activity,
        instruction: String? = nil
    ) async throws -> NavigationOutcome {
        let prepared = try await refreshNavigation(
            configuration: configuration, activity: activity)
        let observed = try outcome(
            for: intent, result: prepared.result, arguments: prepared.arguments,
            manifest: prepared.manifest, systemAlert: prepared.systemAlert,
            progressed: false, observedScreenFacts: prepared.screenFacts,
            reobservedBeforeDispatch: true,
            observationRefreshed: prepared.observationRefreshed)
        guard case .object(var body) = try JSONDecoder().decode(
            JSONValue.self, from: Data(observed.content.utf8)) else {
            throw VisionCaptureAgentError.noProgress(
                "\(failure.description). The following read could not be retained. No input was sent.")
        }
        body["instruction"] = .string(
            instruction ?? Self.ambiguousTargetInstruction(isTyping: intent.operation == .type))
        return NavigationOutcome(
            content: try encodeOutcomeBody(body),
            recoverableColdMissArguments: nil,
            progressed: false,
            successfulReadOnlyObservation: observed.successfulReadOnlyObservation)
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

        var refreshChoices = refreshChoices
        let screenChanged: Bool
        if let baselineScreenSignature, let observedScreenSignature {
            screenChanged = baselineScreenSignature != observedScreenSignature
        } else {
            // The screens cannot be compared. Treat the screen as new and
            // give the model fresh choices instead of stopping.
            screenChanged = true
            refreshChoices = true
        }
        if isConfirmation, !screenChanged {
            // The confirming attempt was refused again on the same screen.
            // Give fresh choices. The next choice of this action skips the cache.
            refreshChoices = true
        }
        if screenChanged || isConfirmation {
            staleActionConfirmation = nil
        } else if let observedScreenSignature {
            staleActionConfirmation = StaleActionConfirmation(
                intent: intent,
                screenSignature: observedScreenSignature)
        }
        if screenChanged {
            coldPathIntents.removeAll()
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
            throw VisionCaptureAgentError.proposalCorrectionExhausted(
                "Agent paused because 24 different stale cached actions were already retired on the current screen. No additional action was sent. Completed work is retained. You can continue this chat.")
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
            interruptibleByUserInstruction: true,
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

    private func reusableObservedManifest(
        for intent: NavigationIntent,
        configuration: VisionCaptureAgentConfiguration
    ) -> AuthorityManifest? {
        guard currentManifest.state == "cold" || currentManifest.state == "observed",
              currentManifest.observationGrant != nil,
              committedTargetKey == configuration.targetKey,
              committedSessionIdentity?.kind == "flow",
              currentScreenSignature != nil,
              currentSystemAlert == nil,
              isEligibleOfferedAction(intent) else { return nil }
        var offered = currentManifest
        offered.actions = eligibleOfferedActions(offered.actions)
        guard let action = try? Self.uniquePublishedAction(for: intent, in: offered),
              action.actionCapability == nil,
              action.revalidationCapability == nil else { return nil }
        return currentManifest
    }

    private func makeActionArguments(
        intent: NavigationIntent,
        action: PublishedAction,
        manifest: AuthorityManifest,
        configuration: VisionCaptureAgentConfiguration
    ) throws -> JSONValue {
        // An intent refused as stale twice takes the slow path: a plain tap
        // under the current observation grant, with no cache entry.
        let useCache = !(coldPathIntents.contains(intent)
            && action.action == "tap"
            && manifest.observationGrant != nil)
        if useCache, let capability = action.actionCapability {
            let request = action.action == "tap"
                ? "tap cached action" : "execute cached action"
            return makeMCPArguments(
                request: request,
                parameters: ["action_capability": .string(capability)],
                configuration: configuration)
        }
        if useCache, let capability = action.revalidationCapability {
            return makeMCPArguments(
                request: "revalidate cached action",
                parameters: ["revalidation_capability": .string(capability)],
                configuration: configuration)
        }
        guard let grant = manifest.observationGrant else {
            if action.action == "tap" {
                // No grant: a plain tap by name. VisionCapture accepts it.
                return makeMCPArguments(
                    request: "tap \(action.selector)", parameters: [:], configuration: configuration)
            }
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
        if request == "describe screen" {
            lockedParameters["describe"] = Self.fieldValueDescribeOptions
        }
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

    /// Applies to the existing plain, granted, recovery and post-image reads.
    /// Current values are useful only under the producer's ordinary privacy policy.
    private static let fieldValueDescribeOptions: JSONValue = .object([
        "include_values": .bool(true),
        "redaction": .string("balanced"),
    ])

    private func executeHostRequest(
        _ arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration,
        usesFlowSession: Bool = true,
        interruptibleByUserInstruction: Bool = false,
        activityID: UUID = UUID(),
        activity: @escaping Activity
    ) async throws -> VisionCaptureMCPResult {
        checkingLocalProposal = false
        // VisionCapture refuses driver-lane mutations while a pointer task
        // owns the device (DEVICE_LANE_CONFLICT). Hide the pointer first; the
        // next pointer tap activates a new task.
        if let pointer = liveComputerUseTask, pointer.udid == configuration.simulatorUDID,
           case .object(let object) = arguments,
           case .string(let request)? = object["request"],
           Self.isDriverLaneMutation(request) {
            liveComputerUseTask = nil
            await closeComputerUseTask(pointer.task, configuration: configuration, activity: activity)
        }
        // A new host observation or action ends the preceding packet's image
        // offer, including when transport fails or cancellation interrupts it.
        currentImageObservation = nil
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
            let pendingCheck = activePendingUserInstructionCheck
            result = try await client.execute(
                arguments: arguments,
                beforeDispatch: {
                    guard interruptibleByUserInstruction,
                          await pendingCheck?() == true else { return }
                    throw UserInstructionPendingBeforeDispatch()
                })
        } catch is UserInstructionPendingBeforeDispatch {
            await activity(.requestStatus(
                id: activityID,
                status: .notSent(reason: "A new user instruction arrived before dispatch."),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw UserInstructionPendingBeforeDispatch()
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
        if let trace {
            let operation: String?
            if case .object(let request) = arguments, case .string(let name)? = request["request"] {
                operation = name
            } else { operation = nil }
            await trace.mcpRequest(
                step: traceStep, requestID: activityID, operation: operation,
                elapsedSeconds: Self.elapsedSeconds(since: activityStart),
                isError: result.isError, refusalCode: result.refusalCode)
            if result.isError {
                await trace.mcpFailure(
                    step: traceStep, requestID: activityID, operation: operation, result: result)
            }
        }
        let launchTimeout = Self.isolatedLaunchTimeout(
            result, arguments: arguments, configuration: configuration)
        let launchNotForeground = Self.isolatedLaunchNotForeground(
            result, arguments: arguments, configuration: configuration)
        let uncertainSwipe = Self.isRecoverableUncertainSwipe(
            result, arguments: arguments, configuration: configuration)
        let uncertainComputerUseClick = Self.isRecoverableUncertainComputerUseClick(
            result, arguments: arguments, configuration: configuration)
        do {
            try Self.validateReturnedIdentity(
                in: result.value,
                configuration: configuration,
                refusalCode: result.refusalCode,
                permittedNonTargetLaunchOutcome:
                    launchTimeout ?? launchNotForeground,
                permitsUncertainRecipientAfterSubmission:
                    uncertainSwipe || uncertainComputerUseClick,
                permitsForeignObservedBundleID:
                    arguments.objectValue?["request"] == .string("click pointer"))
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
        if launchNotForeground != nil {
            await activity(.requestStatus(
                id: activityID,
                status: .serverOutcome(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            return result
        }
        if uncertainSwipe || uncertainComputerUseClick {
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
            || result.isObservedTapTargetUnavailableBeforeDispatch
            || isRecoverableRevalidationLayoutChange(
                result, arguments: arguments, configuration: configuration)
            || isRecoverableActionAuthorizationExpiry(
                result, arguments: arguments, configuration: configuration)
            || isRecoverableObservationTopologyChange(
                result, arguments: arguments, configuration: configuration)
            || Self.isRecoverableStaleCacheAction(
                result,
                arguments: arguments)
            || result.isPointerPreCaptureFailureBeforeSubmission {
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
        if result.isActionDispatchUnproven {
            await activity(.requestStatus(
                id: activityID,
                status: .serverOutcome(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            return result
        }
        if Self.isAmbiguousTargetBeforeSubmission(result) {
            await activity(.requestStatus(
                id: activityID,
                status: .recoverablePreDispatchRefusal(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw AmbiguousTargetBeforeDispatch(failure: serverOutcome)
        }
        if Self.isBusyHostReadBeforeDispatch(result) {
            await activity(.requestStatus(
                id: activityID,
                status: .recoverablePreDispatchRefusal(serverOutcome),
                elapsedSeconds: Self.elapsedSeconds(since: activityStart)))
            throw BusyHostReadBeforeDispatch(failure: serverOutcome)
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

    /// TARGET_AMBIGUOUS, or EXECUTION_NO_MATCH whose target evidence says ambiguous,
    /// with every dispatch envelope rejected before submission: no input was sent.
    static func isAmbiguousTargetBeforeSubmission(_ result: VisionCaptureMCPResult) -> Bool {
        guard result.isError, result.dispatchAttempted != true,
              !result.hasConflictingDispatchAttemptEvidence else { return false }
        if result.refusalCode == "EXECUTION_NO_MATCH" {
            var targets: [[String: JSONValue]] = []
            try? collectStructuredObjects(named: "target", in: result.value) { targets.append($0) }
            guard targets.contains(where: { $0["reason_code"] == .string("TARGET_AMBIGUOUS") }) else { return false }
        } else if result.refusalCode != "TARGET_AMBIGUOUS" {
            return false
        }
        var dispatches: [[String: JSONValue]] = []
        do {
            try collectStructuredObjects(named: "dispatch", in: result.value) { dispatches.append($0) }
        } catch {
            return false
        }
        guard !dispatches.isEmpty else { return false }
        return dispatches.allSatisfy {
            $0["status"] == .string("rejected_before_submission")
                && $0["submission_started"] != .bool(true)
                && $0["delivery_acknowledged"] != .bool(true)
        }
    }

    private static func isRecoverableStaleCacheAction(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue
    ) -> Bool {
        guard result.isStaleActionCapabilityBeforeDispatch,
              case .object(let object) = arguments,
              case .string(let rawRequest)? = object["request"],
              case .object(let parameters)? = object["parameters"] else {
            return false
        }
        // A cached handle or a plain observation grant can both go stale.
        let request = rawRequest.trimmingCharacters(
            in: .whitespacesAndNewlines).lowercased()
        let evidence = parameters["action_capability"]
            ?? parameters["revalidation_capability"]
            ?? parameters["observation_grant"]
        guard case .string(let evidenceValue)? = evidence,
              !evidenceValue.isEmpty else {
            // A plain tap by name carries no handle. Its refusal before
            // submission is recoverable in the same way.
            return request.hasPrefix("tap ")
        }
        return request == "tap cached action"
            || request == "execute cached action"
            || request == "revalidate cached action"
            || request == "execute observed action"
            || request.hasPrefix("tap ")
    }

    /// An uncertain swipe can continue only through one read-only recovery.
    /// The gesture is never replayed. Every returned evidence copy must agree
    /// that submission began, delivery was not acknowledged, and no recipient
    /// was observed.
    private static func isRecoverableUncertainSwipe(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard result.isError,
              result.refusalCode == "SWIPE_FAILED",
              !result.hasConflictingDispatchAttemptEvidence,
              result.dispatchAttempted != false,
              let request = arguments.objectValue,
              case .string(let operation)? = request["request"],
              ["swipe up", "swipe down", "swipe left", "swipe right"]
                .contains(operation),
              request["bundle_id"] == .string(configuration.bundleIdentifier),
              case .string(let sessionID)? = request["session_id"],
              !sessionID.isEmpty,
              request["session_kind"] == .string("flow"),
              request["parameters"] == .object([
                  "udid": .string(configuration.simulatorUDID),
              ]) else {
            return false
        }
        do {
            guard let proof = try sanitizedNamedObject(
                "proof",
                allowedKeys: ["verdict", "reason_code", "action"],
                in: result.value),
                proof["verdict"] == .string("inconclusive"),
                proof["reason_code"] == .string(
                    "DISPATCH_UNCERTAIN_AFTER_SUBMISSION"),
                proof["action"] == .string("swipe") else {
                return false
            }
            var copies: [[String: JSONValue]] = []
            try collectStructuredObjects(
                named: "interaction_evidence",
                in: result.value) { copies.append($0) }
            return !copies.isEmpty && copies.allSatisfy {
                permitsNullObservedBundleIDForUncertainSubmission(
                    in: $0,
                    configuration: configuration)
            }
        } catch {
            return false
        }
    }

    /// A Computer Use click can be submitted without a delivery receipt when
    /// its terminal framebuffer read times out. Keep that input single-shot,
    /// accept only the exact public uncertainty contract, and recover by read.
    private static func isRecoverableUncertainComputerUseClick(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard !result.hasConflictingDispatchAttemptEvidence,
              result.dispatchAttempted != false,
              let request = arguments.objectValue,
              request["request"] == .string("click pointer"),
              request["bundle_id"] == .string(configuration.bundleIdentifier),
              request["session_id"] == nil,
              request["session_kind"] == nil,
              let parameters = request["parameters"]?.objectValue,
              Set(parameters.keys) == [
                  "computer_use_task_id", "computer_use_generation",
                  "x_norm", "y_norm", "intent", "cache_policy", "udid",
              ],
              case .string(let taskID)? = parameters["computer_use_task_id"],
              UUID(uuidString: taskID) != nil,
              case .integer(let generation)? = parameters["computer_use_generation"],
              generation > 0,
              case .integer(let x)? = parameters["x_norm"], (0...1000).contains(x),
              case .integer(let y)? = parameters["y_norm"], (0...1000).contains(y),
              case .string(let intent)? = parameters["intent"], !intent.isEmpty,
              parameters["cache_policy"] == .string("visual_bypass"),
              parameters["udid"] == .string(configuration.simulatorUDID) else {
            return false
        }
        do {
            guard let proof = try sanitizedNamedObject(
                "proof",
                allowedKeys: ["verdict", "reason_code", "action"],
                in: result.value),
                proof["verdict"] == .string("inconclusive"),
                proof["reason_code"] == .string("EXPECTED_OUTCOME_MISSING"),
                proof["action"] == .string("click pointer") else {
                return false
            }
            var copies: [[String: JSONValue]] = []
            try collectStructuredObjects(
                named: "interaction_evidence",
                in: result.value) { copies.append($0) }
            return !copies.isEmpty && copies.allSatisfy {
                permitsNullObservedBundleIDForUnacknowledgedComputerUse(
                    in: $0,
                    configuration: configuration,
                    expectedTaskID: taskID,
                    expectedGeneration: generation,
                    expectedX: x,
                    expectedY: y)
            }
        } catch {
            return false
        }
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

    private func isRecoverableObservedTapTargetUnavailable(
        _ result: VisionCaptureMCPResult,
        arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        result.isObservedTapTargetUnavailableBeforeDispatch
            && matchesCommittedObservedTapRequest(arguments, configuration: configuration)
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

    private func matchesCommittedObservedTapRequest(
        _ arguments: JSONValue,
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard committedTargetKey == configuration.targetKey,
              currentSystemAlert == nil, uncertainAlertPress == nil,
              let session = committedSessionIdentity, session.kind == "flow",
              let request = arguments.objectValue,
              case .string(let rawRequest)? = request["request"],
              rawRequest.hasPrefix("tap "), !rawRequest.dropFirst(4).isEmpty,
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
              Set(parameters.keys) == ["udid", "observation_grant", "describe"],
              parameters["describe"] == Self.fieldValueDescribeOptions,
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

    static func instructions(
        configuration _: VisionCaptureAgentConfiguration
    ) -> String {
        return """
        You are the iOS QA agent for the configured simulator app. Complete the user's goal with observable evidence.

        - Apply the newest user instruction before another app action. Keep completed evidence. Respect prohibited_targets.
        - Reply with one permitted tool call or a concise final answer.
        - Use only allowed_next and exact current choice IDs. Never invent controls. Each result replaces earlier choices.
        - observation, facts, and choices describe the current screen. last_action describes the previous step.
        - Use useful content already visible. If a choice advances unfinished work, act before reading the same screen again.
        - When current_image_evidence is true, use positions to match unlabeled choices. If a label and visible position conflict, send \(VisionCaptureToolDefinitions.visualClickCall) at the visible control.
        - Send \(VisionCaptureToolDefinitions.coordinateTapCall), or \(VisionCaptureToolDefinitions.visualClickCall), only with current image evidence. A fact marked ocr-confirmed needs no screenshot: tap it with \(VisionCaptureToolDefinitions.coordinateTapCall), or type into it with \(VisionCaptureToolDefinitions.typeAtPositionCall), using its position. A focused field may omit target when the software keyboard is visible. Use screenshots or Computer Use when accessibility information is not enough.
        - Take a screenshot when a control requires_screenshot, a form has no editable fields, or a read exposes only keyboard controls.
        - Wheel pickers (columns of stacked values where the middle row is the current value, for example a date picker's month and year): set each column separately. To choose a value, tap its row when you can see it. If it is not visible, tap the row at the end of that column closest to it (the top or bottom row); the column moves and shows the next values. Check the screen again and repeat until the value is visible, then tap it. Do not swipe a picker wheel.
        - If a form remains after a verified tap, do not repeat that choice. Use a fresh screenshot and the visible submit control.
        - An accepted request is not proof. A verified action proves only that action. Visible state does not prove an interaction was tested.
        - Keep failed, refused, inconclusive, and delivery-unknown results. Never replay uncertain input. Continue other reachable checks.
        - If an instruction asks for something the app does not offer, do not keep trying: record it as a blocker that names the missing feature, continue with the remaining parts of the task, and list the blocker in your final report.
        - Do not repeat completed work unless requested.
        - Finish only when each requested check has evidence or a factual blocker. Report verified results, failures, and blockers only.
        """
    }

    private static func hasUnfinishedChecklist(_ answer: String) -> Bool {
        var insideCodeFence = false
        var insideChecklist = false
        for rawLine in answer.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.hasPrefix("```") {
                insideCodeFence.toggle()
                continue
            }
            guard !insideCodeFence, !line.hasPrefix(">") else { continue }
            let lower = line.lowercased()
            if lower.contains("checklist") &&
               (lower.contains("result") || lower.contains("status")) {
                insideChecklist = true
                continue
            }
            guard insideChecklist else { continue }
            if line.hasPrefix("#") { insideChecklist = false; continue }
            guard let colon = line.lastIndex(of: ":") else { continue }
            let status = line[line.index(after: colon)...]
                .replacingOccurrences(of: "*", with: "")
                .replacingOccurrences(of: "_", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            if status == "pending" || status == "in progress" { return true }
        }
        return false
    }

    private static func preflight(_ calls: [AppToolCall]) throws -> AppToolCall {
        guard calls.count == 1 else {
            throw VisionCaptureAgentError.malformedCall(
                "only one visioncapture_navigate call is allowed per assistant turn")
        }
        // A complete native call already awaits its matching result in the
        // runtime. Reject an unknown name inside the bounded local correction
        // path below, preserving that actual name/ID. It must not be aliased,
        // executed, or handled by replaying the preceding tool result.
        return calls[0]
    }

    private func navigationIntent(
        from call: AppToolCall,
        configuration: VisionCaptureAgentConfiguration
    ) throws -> NavigationIntent {
        resolvedJourneyLabel = nil
        guard call.name == VisionCaptureToolDefinitions.navigateName else {
            throw VisionCaptureAgentError.malformedCall(
                "Use only visioncapture_navigate for the app. The unknown tool was not executed.")
        }
        usedCoordinatesOverTarget = false
        guard case .object(let proposed) = call.arguments,
              case .string? = proposed["action"] else {
            throw VisionCaptureAgentError.malformedCall(
                "Choose one visioncapture_navigate action from allowed_next.")
        }
        let normalized = Self.normalizedTargetAndCoordinates(proposed)
        let object = normalized.object
        usedCoordinatesOverTarget = normalized.usedCoordinates
        guard case .string(let name)? = object["action"],
              let operation = NavigationOperation(rawValue: name) else {
            throw VisionCaptureAgentError.malformedCall(
                "Choose one visioncapture_navigate action from allowed_next.")
        }
        let supplied = Set(object.keys)
        let required: Set<String>
        switch operation {
        case .tap: required = ["action", "target"]
        case .tapCoordinates, .computerUseClick:
            required = ["action", "x_norm", "y_norm", "intent"]
        case .setBoolean: required = ["action", "target", "desired_state"]
        case .type:
            if object["target"] != nil {
                required = ["action", "target", "text"]
            } else if object["x_norm"] != nil || object["y_norm"] != nil {
                required = ["action", "text", "x_norm", "y_norm"]
            } else {
                required = ["action", "text"]
            }
        case .swipe: required = ["action", "direction"]
        case .launch, .observe, .screenshot, .back: required = ["action"]
        }
        let permitted = required.union(["intent"])
        guard required.isSubset(of: supplied), supplied.isSubset(of: permitted) else {
            throw VisionCaptureAgentError.malformedCall(
                "This action requires: \(required.sorted().joined(separator: ", ")). Only intent is optional.")
        }
        if let intent = object["intent"] {
            guard case .string(let rawIntent) = intent,
                  !rawIntent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw VisionCaptureAgentError.malformedCall(
                    "Intent must be a nonempty string when supplied.")
            }
        }
        let allowedNow = permittedNextOperations
            ?? Set<NavigationOperation>([.launch, .observe, .screenshot])
        guard allowedNow.contains(operation) else {
            if operation == .type,
               object["target"] == nil,
               currentScreenFacts?.hasSoftwareKeyboard == true,
               currentImageObservation != observationGeneration,
               allowedNow.contains(.screenshot) {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "Focused typing needs current screenshot evidence. Choose screenshot now. If it shows the intended field focused with the keyboard visible, retry type without a target.")
            }
            if operation == .tapCoordinates || operation == .computerUseClick,
               currentImageObservation != observationGeneration,
               allowedNow.contains(.screenshot) {
                throw VisionCaptureAgentError.navigationUnavailable(Self.coordinateEvidenceReason)
            }
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
        case .tapCoordinates, .computerUseClick:
            var ocrConfirmedTarget = false
            if operation == .tapCoordinates,
               case .integer(let cx)? = object["x_norm"], case .integer(let cy)? = object["y_norm"],
               currentScreenFacts?.isOCRConfirmedPosition(x: cx, y: cy) == true {
                ocrConfirmedTarget = true
            }
            guard currentImageObservation == observationGeneration || ocrConfirmedTarget,
                  currentSystemAlert == nil,
                  committedSessionIdentity?.kind == "flow",
                  case .integer(let x)? = object["x_norm"],
                  case .integer(let y)? = object["y_norm"],
                  (0...1000).contains(x), (0...1000).contains(y),
                  case .string(let rawIntent)? = object["intent"] else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "This visual action requires the current screenshot, a flow session, coordinates from 0 to 1000, and a nonempty intent.")
            }
            let visualIntent = rawIntent.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !visualIntent.isEmpty else {
                throw VisionCaptureAgentError.malformedCall(
                    "Visual action intent must not be empty.")
            }
            resolvedJourneyLabel = String(visualIntent.prefix(160))
            return NavigationIntent(
                operation: operation, selector: nil, selectorKind: nil,
                role: nil, desiredState: nil, text: nil,
                xNorm: Int(x), yNorm: Int(y), visualIntent: visualIntent)
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
        case .type where object["target"] == nil && object["x_norm"] != nil:
            // The field is on screen but not published: type at its ocr-confirmed position.
            guard case .integer(let x)? = object["x_norm"], case .integer(let y)? = object["y_norm"],
                  let field = currentScreenFacts?.ocrConfirmedEditablePosition(x: x, y: y) else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "No ocr-confirmed field is at that position. Tap a field choice, or take a screenshot and tap the field first.")
            }
            guard case .string(let raw)? = object["text"], !raw.isEmpty else {
                throw VisionCaptureAgentError.malformedCall("Typing requires nonempty text.")
            }
            resolvedJourneyLabel = "field at (\(field.x), \(field.y))"
            return NavigationIntent(
                operation: .type, selector: nil, selectorKind: "ocr_position",
                role: "text_field", desiredState: nil, text: raw,
                xNorm: Int(field.x), yNorm: Int(field.y))
        case .type where object["target"] == nil:
            guard currentImageObservation == observationGeneration,
                  currentScreenFacts?.hasSoftwareKeyboard == true,
                  committedSessionIdentity?.kind == "flow",
                  case .string(let raw)? = object["text"], !raw.isEmpty else {
                throw VisionCaptureAgentError.navigationUnavailable(
                    "Focused typing requires a current screenshot that shows the software keyboard and a current flow session.")
            }
            resolvedJourneyLabel = "focused field"
            return NavigationIntent(
                operation: .type, selector: nil, selectorKind: nil,
                role: "focused_editable_element", desiredState: nil, text: raw)
        case .tap, .setBoolean, .type:
            break
        }
        if case .string(let target)? = object["target"], !Self.isChoiceIDShaped(target) {
            throw VisionCaptureAgentError.navigationUnavailable(Self.choiceTargetReason)
        }
        guard case .string(let target)? = object["target"],
              let binding = currentChoiceBindings[target],
              binding.observation == observationGeneration,
              binding.targetKey == configuration.targetKey,
              binding.session == committedSessionIdentity,
              binding.screenSignature == currentScreenSignature,
              bindingIsCurrent(binding) else {
            throw VisionCaptureAgentError.navigationUnavailable(
                Self.expiredTargetReason)
        }
        guard binding.operation == operation else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "The proposed target supports \(binding.operation.rawValue). Requested \(operation.rawValue) is not supported.")
        }
        guard Self.meaningfulChoiceLabel(binding.displayLabel) != nil
                || currentImageObservation == binding.observation else {
            throw VisionCaptureAgentError.navigationUnavailable(
                "This target has no readable label and no current image evidence. Request screenshot, then choose a new target ID from its usable image/read pair. If screenshot is unavailable or the target remains unclear, report the limitation. No input was sent.")
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
        if case .screenText(let x, let y) = binding.route {
            return NavigationIntent(operation: operation, selector: binding.selector,
                selectorKind: binding.selectorKind, role: binding.role,
                desiredState: nil, text: nil, xNorm: Int(x), yNorm: Int(y))
        }
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
        case .screenText:
            return true
        }
    }

    private func bindChoice(
        _ object: [String: JSONValue],
        isField: Bool,
        isConfirmation: Bool,
        configuration: VisionCaptureAgentConfiguration,
        excludingTargetID: String?,
        reusableChoiceBindings: [String: ChoiceBinding] = [:]
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
        // An unnamed control without current pixels is evidence, not an
        // executable choice. Do not allocate an ID or retain a binding yet.
        guard displayLabel != nil || currentImageObservation == observationGeneration else { return nil }
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
        let reusedID = reusableChoiceBindings.first { id, previous in
            id != excludingTargetID
                && previous.targetKey == binding.targetKey
                && previous.session == binding.session
                && previous.screenSignature == binding.screenSignature
                && previous.operation == binding.operation
                && previous.selector == binding.selector
                && previous.selectorKind == binding.selectorKind
                && previous.role == binding.role
                && previous.displayLabel == binding.displayLabel
                && previous.allowedStates == binding.allowedStates
        }?.key
        let id: String
        if let reusedID {
            id = reusedID
        } else {
            var allocated: String
            repeat {
                guard nextChoiceNumber < UInt64.max else {
                    throw VisionCaptureAgentError.noProgress("The conversation exhausted its choice IDs. Start a new chat.")
                }
                nextChoiceNumber += 1
                allocated = "c\(nextChoiceNumber)"
                // An unavailable ID may have been guessed before it was issued.
                // Never allocate that rejected ID while publishing its repair.
            } while allocated == excludingTargetID
            id = allocated
        }
        choice["id"] = .string(id)
        choice["operations"] = .array([.string(operation.rawValue)])
        if displayLabel == nil {
            choice["requires_screenshot"] = .bool(currentImageObservation != observationGeneration)
        }
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

    private func unavailableChoiceFacts(
        _ object: [String: JSONValue], isField: Bool
    ) -> JSONValue {
        var facts = readableChoiceFacts(object, isField: isField)
        facts["availability"] = .string("not_offered")
        if facts["label"] == nil, currentImageObservation != observationGeneration {
            facts["requires_screenshot"] = .bool(true)
        }
        return .object(facts)
    }

    private func readableChoiceFacts(
        _ object: [String: JSONValue], isField: Bool
    ) -> [String: JSONValue] {
        var facts: [String: JSONValue] = [:]
        for key in ["role", "enabled", "selected", "value", "value_status", "position", "current_state"] {
            if let value = object[key] { facts[key] = value }
        }
        if case .string(let display)? = object["label"],
           let label = Self.meaningfulChoiceLabel(display) {
            facts["label"] = .string(label)
        } else if case .string(let selector)? = object["selector"],
                  case .string(let role)? = object["role"] {
            let kind: String?
            if case .string(let value)? = object["selector_kind"] { kind = value } else { kind = nil }
            if (role == "system_alert_button" || kind == "placeholder"),
               let label = Self.meaningfulChoiceLabel(selector) {
                facts["label"] = .string(label)
            } else if let label = currentScreenFacts?.readableLabel(
                selector: selector, role: role, selectorKind: kind),
                let label = Self.meaningfulChoiceLabel(label) {
                facts["label"] = .string(label)
            }
        }
        if !isField, case .string(let selector)? = object["selector"],
           case .string(let role)? = object["role"] {
            let kind: String?
            if case .string(let value)? = object["selector_kind"] { kind = value } else { kind = nil }
            if let warning = currentScreenFacts?.coverageWarning(selector: selector, role: role, selectorKind: kind) {
                facts["warning"] = .string(warning)
            }
        }
        if isField, facts["value"] == nil { facts["value"] = .null }
        if isField, facts["value_status"] == nil {
            facts["value_status"] = .string(facts["value"] == .null ? "unavailable" : "available")
        }
        return facts
    }

    private static func meaningfulChoiceLabel(_ label: String?) -> String? {
        guard let label else { return nil }
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("__vc"),
              !trimmed.contains("[REDACTED]"),
              trimmed.unicodeScalars.contains(where: {
                  CharacterSet.alphanumerics.contains($0) || $0.properties.isEmojiPresentation
              }) else { return nil }
        return label
    }

    private func decisionPacket(
        from content: String,
        call: AppToolCall,
        configuration: VisionCaptureAgentConfiguration,
        images: [AppImageAttachment],
        excludingTargetID: String? = nil,
        preservingChoiceIDs: Bool = false
    ) throws -> DecisionPacket {
        guard case .object(let body) = try JSONDecoder().decode(
            JSONValue.self, from: Data(content.utf8)) else {
            throw VisionCaptureAgentError.malformedCall("The host could not encode the current decision facts.")
        }
        let reusableChoiceBindings = preservingChoiceIDs ? currentChoiceBindings : [:]
        let preservesCurrentImage = preservingChoiceIDs
            && currentImageObservation == observationGeneration
        currentImageObservation = nil
        currentChoiceBindings.removeAll(keepingCapacity: true)
        guard observationGeneration < UInt64.max else {
            throw VisionCaptureAgentError.noProgress("The conversation exhausted its observation IDs. Start a new chat.")
        }
        observationGeneration += 1
        // A locally rejected proposal sent no app input and performed no new
        // screen read. Keep the immediately preceding screenshot usable when
        // republishing the same current choices with their existing IDs.
        if preservesCurrentImage {
            currentImageObservation = observationGeneration
        }
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
        } else if (operation == .string("observe") || operation == .string("screenshot")),
                  outcome == .string("succeeded"),
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
        if operation == .string("screenshot"), outcome == .string("succeeded"),
           !images.isEmpty, !readOnlyRequired, !unavailable,
           body["image_observation"]?.objectValue?["state"] == .string("sequential"),
           currentScreenSignature != nil || currentSystemAlert != nil {
            currentImageObservation = observationGeneration
        }
        var choices: [JSONValue] = []
        var semanticChoices: [JSONValue] = []
        var unavailableChoices: [JSONValue] = []
        var requiresVisualDisambiguation = false
        let restrictionScreenContext = ["screen_summary", "navigation_fact"]
            .compactMap { key -> String? in
                guard case .string(let value)? = body[key] else { return nil }
                return value
            }
            .joined(separator: " ")
        for (values, isField) in [(Self.array(body["available_actions"]), false),
                                 (Self.array(body["available_text_fields"]), true)] {
            for value in values {
                guard case .object(let object) = value else { continue }
                let label: String?
                if case .string(let value)? = object["label"] { label = value }
                else { label = nil }
                let selector: String?
                if case .string(let value)? = object["selector"] { selector = value }
                else { selector = nil }
                if userRestrictions.prohibits(
                    label: label,
                    selector: selector,
                    screenContext: restrictionScreenContext
                ) {
                    var restricted: [String: JSONValue] = [
                        "availability": .string("prohibited_by_user"),
                    ]
                    if let label { restricted["label"] = .string(label) }
                    if let role = object["role"] { restricted["role"] = role }
                    unavailableChoices.append(.object(restricted))
                    continue
                }
                if !isField, !userRestrictions.displayTargets.isEmpty,
                   currentImageObservation != observationGeneration,
                   readableChoiceFacts(object, isField: false)["label"] == nil {
                    requiresVisualDisambiguation = true
                }
                if Self.isAlreadySelectedTap(object, isField: isField) {
                    var selected: [String: JSONValue] = [
                        "availability": .string("already_selected"),
                    ]
                    if let label { selected["label"] = .string(label) }
                    if let role = object["role"] { selected["role"] = role }
                    unavailableChoices.append(.object(selected))
                    continue
                }
                if completedCycleActions.contains(where: { action in
                    Self.matchesCompletedCycleEntry(
                        object, isField: isField,
                        operation: action.operation.rawValue,
                        selector: action.selector,
                        selectorKind: action.selectorKind,
                        role: action.role,
                        desiredState: action.desiredState)
                }) {
                    var completed = readableChoiceFacts(object, isField: isField)
                    completed["availability"] = .string("verified_action_returned_to_same_screen")
                    unavailableChoices.append(.object(completed))
                    continue
                }
                if !readOnlyRequired, !unavailable, recovery == nil,
                   let bound = try bindChoice(object, isField: isField, isConfirmation: false,
                       configuration: configuration, excludingTargetID: excludingTargetID,
                       reusableChoiceBindings: reusableChoiceBindings) {
                    choices.append(bound.choice)
                    semanticChoices.append(bound.semantic)
                } else {
                    unavailableChoices.append(unavailableChoiceFacts(object, isField: isField))
                }
            }
        }
        if !readOnlyRequired, !unavailable {
            if confirmationAllowed, let object = recovery?["confirming_action"]?.objectValue {
                if let bound = try bindChoice(object, isField: false, isConfirmation: true,
                    configuration: configuration, excludingTargetID: excludingTargetID,
                    reusableChoiceBindings: reusableChoiceBindings) {
                    choices.append(bound.choice)
                    semanticChoices.append(bound.semantic)
                } else {
                    unavailableChoices.append(unavailableChoiceFacts(object, isField: false))
                }
            }
        }
        var offersScreenText = false
        var offeredScreenText: [ScreenTextBlock] = []
        var screenText = body["screen_text"]
        // A failed read may follow a change: the kept words could be stale.
        if body["screen_text_unavailable"] != nil { lastScreenText = nil }
        if let signature = currentScreenSignature {
            if let fresh = screenText {
                lastScreenText = (signature, fresh)
            } else if preservingChoiceIDs, let kept = lastScreenText, kept.signature == signature {
                // A refused proposal sent and read nothing: offer the same words again.
                screenText = kept.blocks
            }
        }
        var screenTextIDs: [(block: ScreenTextBlock, id: String)] = []
        if !readOnlyRequired, !unavailable, recovery == nil, currentScreenSignature != nil,
           case .array(let rawBlocks)? = screenText {
            let blocks = rawBlocks.compactMap { value -> ScreenTextBlock? in
                guard let object = value.objectValue, case .string(let text)? = object["text"],
                      case .integer(let x)? = object["x_norm"], case .integer(let y)? = object["y_norm"]
                else { return nil }
                return ScreenTextBlock(text: text, xNorm: x, yNorm: y)
            }
            // A word is skipped where an offered choice, a held-back or withheld
            // control (selected, done, disabled, covering, rejected, used up) or a
            // static text already names it, and where the user prohibited it.
            let elements = Self.array(body["available_actions"]) + Self.array(body["available_text_fields"])
            for block in Self.screenTextBlocksToOffer(
                blocks, knownElements: choices + elements + withheldScreenElements() + staticTextFactElements(),
                restrictions: userRestrictions, screenContext: restrictionScreenContext)
            where isEligibleOfferedAction(Self.screenTextTapIntent(block)) {
                var id: String
                // The same text within 4 units of its last place keeps its ID; OCR
                // places a word that did not move within 3 units from read to read.
                if let kept = lastScreenTextIDs.first(where: { previous in
                    previous.block.text == block.text && abs(previous.block.xNorm - block.xNorm) <= 4
                        && abs(previous.block.yNorm - block.yNorm) <= 4 && previous.id != excludingTargetID
                        && currentChoiceBindings[previous.id] == nil
                }) {
                    id = kept.id
                } else {
                    repeat {
                        guard nextChoiceNumber < UInt64.max else {
                            throw VisionCaptureAgentError.noProgress("The conversation exhausted its choice IDs. Start a new chat.")
                        }
                        nextChoiceNumber += 1
                        id = "c\(nextChoiceNumber)"
                    } while id == excludingTargetID
                }
                screenTextIDs.append((block, id))
                let choice = Self.screenTextChoice(block, id: id)
                currentChoiceBindings[id] = ChoiceBinding(
                    observation: observationGeneration, targetKey: configuration.targetKey,
                    session: committedSessionIdentity, screenSignature: currentScreenSignature,
                    operation: .tap, selector: block.text, selectorKind: Self.screenTextSelectorKind,
                    role: "text", displayLabel: Self.screenTextLabel(block.text), allowedStates: [],
                    route: .screenText(x: block.xNorm, y: block.yNorm))
                choices.append(choice)
                offersScreenText = true
                offeredScreenText.append(block)
                var semantic = choice.objectValue ?? [:]
                semantic.removeValue(forKey: "id")
                semantic["selector"] = .string(block.text)
                semantic["selector_kind"] = .string(Self.screenTextSelectorKind)
                semanticChoices.append(.object(semantic))
            }
        }
        // A screenshot packet already includes a fresh accessibility read.
        // Offering another read immediately discards the pixels and can trap
        // the model in an observe/screenshot loop around unnamed controls.
        let visionAvailable = visionPackComplete(configuration)
        var allowed: Set<NavigationOperation> = currentImageObservation == observationGeneration
            ? []
            : [.observe]
        if readOnlyNoProgress.requiresAlternativeAction, !readOnlyRequired {
            allowed.remove(.observe)
        }
        var swipeDirections: [SwipeDirection] = []
        if currentImageObservation != observationGeneration,
           visionAvailable {
            allowed.insert(.screenshot)
        }
        if !readOnlyRequired, !unavailable, recovery == nil,
           currentSystemAlert == nil,
           currentImageObservation == observationGeneration,
           committedSessionIdentity?.kind == "flow" {
            allowed.insert(.tapCoordinates)
            // The pointer lane needs a window capture that is not available
            // on the fast path; tap_coordinates covers the same need.
            if !Self.fastHostPath { allowed.insert(.computerUseClick) }
            if currentScreenFacts?.hasSoftwareKeyboard == true {
                allowed.insert(.type)
            }
        }
        if !readOnlyRequired, !unavailable, recovery == nil, currentSystemAlert == nil,
           committedSessionIdentity?.kind == "flow",
           currentScreenFacts?.hasOCRConfirmedControls == true {
            // OCR-confirmed controls carry accessibility positions. A
            // coordinate tap on one of them needs no screenshot.
            allowed.insert(.tapCoordinates)
            if currentScreenFacts?.hasOCRConfirmedEditableField == true {
                // Type at the field position: VisionCapture taps, then types.
                allowed.insert(.type)
            }
        }
        for binding in currentChoiceBindings.values { allowed.insert(binding.operation) }
        let hasExplicitBackChoice = choices.contains { choice in
            guard case .string(let label)? = choice.objectValue?["label"] else { return false }
            return switch label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "back", "close": true
            default: false
            }
        }
        if !readOnlyRequired, !unavailable, recovery == nil, currentSystemAlert == nil,
           currentScreenSignature != nil, committedSessionIdentity?.kind == "flow",
           !hasExplicitBackChoice,
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
        if requiresVisualDisambiguation, visionAvailable, !readOnlyRequired, !unavailable {
            choices = []
            semanticChoices = []
            currentChoiceBindings.removeAll(keepingCapacity: true)
            swipeDirections = []
            allowed = [.screenshot]
        }
        if (try? configuration.validate()) == nil || (
            committedTargetKey != nil && committedTargetKey != configuration.targetKey) {
            allowed = []
            swipeDirections = []
            choices = []
            semanticChoices = []
            currentChoiceBindings.removeAll(keepingCapacity: true)
            currentImageObservation = nil
        }
        // Wiped choices offer no screen text, so no auto image goes with them.
        if choices.isEmpty { offersScreenText = false }
        if offersScreenText { lastScreenTextIDs = screenTextIDs }
        permittedNextOperations = allowed
        var observation: [String: JSONValue] = [
            "id": .string("o\(observationGeneration)"),
            "current_image_evidence": .bool(currentImageObservation == observationGeneration),
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
        for key in ["dispatch_attempted", "submission_started", "delivery_acknowledged", "rejected_target"] {
            if let value = body[key] { lastAction[key] = value }
        }
        if unknownDelivery { lastAction["delivery"] = .string("unknown") }
        if let note = body["outcome_note"] { lastAction["note"] = note }
        if let effect = body["effect"] { lastAction["effect"] = effect }
        if let changed = body["screen_changed"] { lastAction["screen_changed"] = changed }
        var packet: [String: JSONValue] = [
            "schema_version": .integer(1), "observation": .object(observation),
            "last_action": .object(lastAction),
            "allowed_next": .array(allowed.map(\.rawValue).sorted().map(JSONValue.string)),
            "facts": .array(facts),
            "choices": .array(Self.choicesForDisplay(choices, includesImage: !images.isEmpty)),
        ]
        if !userRestrictions.displayTargets.isEmpty {
            packet["user_restrictions"] = .object([
                "prohibited_targets": .array(
                    userRestrictions.displayTargets.map(JSONValue.string)),
            ])
        }
        if let instruction = body["instruction"], instruction != .string("Observation is complete.") {
            packet["guidance"] = instruction
        }
        if let warning = body["warning"] { packet["warning"] = warning }
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
        var encoded = try Self.addingScreenTextNote(to: JSONValue.object(packet).encoded(), body: body)
        if offersScreenText, Self.looksLikePickerWheel(offeredScreenText) {
            encoded = try Self.addingGuidanceNote(to: encoded, note: Self.pickerWheelNote)
        }
        if packet["journey_hint"] != nil { lastEmittedJourneyHint = currentJourneyHint }
        return DecisionPacket(content: encoded, comparison: .object(comparison), offersScreenText: offersScreenText)
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

    static func isAlreadySelectedTap(
        _ object: [String: JSONValue],
        isField: Bool
    ) -> Bool {
        !isField
            && object["action"] == .string("tap")
            && object["role"] == .string("button")
            && object["selected"] == .bool(true)
    }

    static func matchesCompletedCycleEntry(
        _ object: [String: JSONValue],
        isField: Bool,
        operation: String,
        selector: String?,
        selectorKind: String?,
        role: String?,
        desiredState: Bool?
    ) -> Bool {
        guard !isField,
              operation == NavigationOperation.tap.rawValue,
              object["action"] == .string(operation),
              object["selector"] == selector.map(JSONValue.string),
              object["role"] == role.map(JSONValue.string) else { return false }
        let offeredSelectorKind: String?
        if case .string(let value)? = object["selector_kind"] {
            offeredSelectorKind = value
        } else {
            offeredSelectorKind = nil
        }
        let offeredDesiredState: Bool?
        if case .bool(let value)? = object["desired_state"] {
            offeredDesiredState = value
        } else {
            offeredDesiredState = nil
        }
        return offeredSelectorKind == selectorKind
            && offeredDesiredState == desiredState
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
        refusedObservedTapBeforeDispatch: VisionCaptureServerOutcome? = nil,
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
            candidates = facts.tapCandidates(
                excluding: manifest.actions.map { ($0.selector, $0.role) },
                excludingLabels: manifest.actions.compactMap { action in
                    action.displayLabel.map { ($0, action.role) }
                })
                .filter { candidate in
                    guard !facts.isLowValueSoftwareKeyboardControl(
                        selector: candidate.selector, role: candidate.role) else {
                        return false
                    }
                    let proposal = NavigationIntent(
                        operation: .tap, selector: candidate.selector, selectorKind: candidate.selectorKind,
                        role: candidate.role, desiredState: nil, text: nil)
                    return isEligibleOfferedAction(proposal)
                }
            offeredTapCandidates = (signature, candidates)
        } else {
            candidates = []
        }
        let ocrPositions = candidates.compactMap { candidate -> (x: Int64, y: Int64)? in
            guard candidate.selectorKind == "ocr_label" else { return nil }
            return facts?.normalizedPosition(
                selector: candidate.selector, role: candidate.role, selectorKind: "ocr_label")
        }
        let publishedActions = eligibleOfferedActions(manifest.actions).filter { action in
            guard let facts else { return true }
            return !facts.isLowValueSoftwareKeyboardControl(
                selector: action.selector,
                role: action.role,
                displayLabel: action.displayLabel,
                position: action.displayPosition)
        }.filter { action in
            // An unlabeled published choice at an OCR-named control cannot be told
            // apart or tapped by ID; the "(ocr)" choice replaces it.
            !(action.displayLabel == nil
                && facts?.readableLabel(selector: action.selector, role: action.role) == nil
                && Self.isOCRDuplicate(position: action.displayPosition, ocrPositions: ocrPositions))
        }
        let availableActions = Self.deduplicatedDecisionActions(
            Self.sanitizedActions(publishedActions, facts: facts)
            + Self.sanitizedSystemAlertActions(systemAlert)
            + candidates.map { candidate in
                var object = facts?.properties(
                    selector: candidate.selector, role: candidate.role,
                    selectorKind: candidate.selectorKind) ?? [:]
                object["action"] = .string("tap")
                object["selector"] = .string(candidate.selector)
                object["role"] = .string(candidate.role)
                if let kind = candidate.selectorKind { object["selector_kind"] = .string(kind) }
                object["requires_validation"] = .bool(true)
                return .object(object)
            }, facts: facts)
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
           refusedObservedTapBeforeDispatch == nil,
           deliveredFailure == nil, verifiedTypingProof == nil, !observationRefreshed {
            switch intent.operation {
            case .tap, .tapCoordinates, .computerUseClick,
                 .setBoolean, .type, .back, .swipe:
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
                if result.isActionDispatchUnproven {
                    body["outcome"] = .string("delivery_unknown")
                    body["delivery_unknown"] = .bool(true)
                    body["instruction"] = .string(
                        "VisionCapture could not prove whether this input was dispatched. Do not repeat it. Choose a read-only observation, then continue from new current choices without replaying this input.")
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
            body["instruction"] = .string("Observation is complete.")
        }
        if availableActions.isEmpty, currentEditableFields.isEmpty,
           systemAlert == nil,
            !result.isRecoverableColdMiss,
           !result.isGuardedTargetRejectedBeforeSubmission {
            body["navigation_fact"] = .string(
                "No app-owned accessibility action or editable field is currently published.")
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
        if let refusedObservedTapBeforeDispatch {
            body["outcome"] = .string("not_dispatched_reobserved")
            body["is_error"] = .bool(true)
            body["dispatch_attempted"] = .bool(false)
            body["observation_outcome"] = .string(result.isError ? "unavailable" : "succeeded")
            body["refusal"] = .object([
                "code": .string(refusedObservedTapBeforeDispatch.reasonCode
                    ?? "CACHE_ACTION_CAPABILITY_INVALID"),
                "reason": .string("live_target_not_actionable"),
            ])
            var originalProof: [String: JSONValue] = [
                "verdict": .string("failed"), "action": .string("tap"),
            ]
            if let reason = refusedObservedTapBeforeDispatch.reason {
                originalProof["reason"] = .string(reason)
            }
            body["proof"] = .object(originalProof)
            body["instruction"] = .string(result.isError
                ? "The chosen target changed before dispatch, so the tap was not sent. Current choices are unavailable. Choose observe and make a new decision."
                : "The chosen target changed before dispatch, so the tap was not sent. That exact action is withheld while this screen remains unchanged. Continue other requested checks using the freshly read choices.")
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
                : "Typing was verified and the host refreshed the current screen.")
        }
        // Request success alone is not verified progress. Use the final action
        // proof after all refusal/recovery/typing overrides, not a later read's
        // success. Keep the caller's false value for read-only recovery paths.
        let verifiedProgress: Bool
        switch intent.operation {
        case .launch:
            verifiedProgress = body["launch"]?.objectValue?["verdict"] == .string("foreground_ready")
        case .tap, .tapCoordinates, .computerUseClick,
             .setBoolean, .type, .back, .swipe:
            verifiedProgress = body["proof"]?.objectValue?["verdict"] == .string("verified")
        case .observe, .screenshot:
            verifiedProgress = false
        }
        return NavigationOutcome(
            content: try encodeOutcomeBody(body),
            recoverableColdMissArguments: result.isRecoverableColdMiss
                ? arguments : nil,
            progressed: progressed && verifiedProgress && body["delivery_unknown"] != .bool(true),
            successfulReadOnlyObservation:
                (intent.operation == .observe || reobservedBeforeDispatch || expiredBeforeDispatch != nil
                    || refusedObservedTapBeforeDispatch != nil
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

    private static func cycleStepLabel(content: String, intent: NavigationIntent) -> String {
        let lastAction = (try? JSONDecoder().decode(JSONValue.self, from: Data(content.utf8)))?
            .objectValue?["last_action"]?.objectValue
        var parts: [String] = []
        if case .string(let action)? = lastAction?["action"] { parts.append(action) }
        if case .string(let label)? = lastAction?["label"] {
            parts.append(label)
        } else if let target = intent.selector ?? intent.visualIntent {
            parts.append(target)
        }
        if let text = intent.text { parts.append("\"\(text)\"") }
        return parts.joined(separator: " ")
    }

    static func addingCycleNote(to content: String, sequence: [String]) throws -> String {
        guard case .object(var body)? = try? JSONDecoder().decode(
            JSONValue.self, from: Data(content.utf8)) else {
            throw VisionCaptureAgentError.malformedCall("the loop could not add its repeated-sequence note")
        }
        let note = "You repeated the same \(sequence.count) actions 3 times without new app facts: \(sequence.joined(separator: " → ")). Do something different, or record a blocker and continue with the next part of the task."
        if case .string(let guidance)? = body["guidance"] {
            body["guidance"] = .string(note + " " + guidance)
        } else {
            body["guidance"] = .string(note)
        }
        return try JSONValue.object(body).encoded()
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
            "The same observed facts and current choices have now returned \(repetitionCount) times without verified progress. Observe is unavailable until another action changes the screen. Repeating delivered input whose effect remains inconclusive has not established progress and may repeat an effect. Choose a different current action, choose screenshot if it can clarify a safe next step, or report the unfinished work if no supported choice serves the goal. Use only current IDs and offered operations. A screenshot includes a following read and supplies only its newly offered choices. It does not restore refused actions. Never retry or replace input whose delivery is unknown."
        if case .array(let allowed)? = body["allowed_next"] {
            body["allowed_next"] = .array(allowed.filter { $0 != .string("observe") })
        }
        if let value = body["guidance"] {
            let existing: String
            if case .string(let text) = value { existing = text }
            else { existing = try value.encoded() }
            body["guidance"] = .string(existing.contains(instruction)
                ? existing : existing + "\n" + instruction)
        } else {
            body["guidance"] = .string(instruction)
        }
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
                : "The alert press was submitted once and then observed read-only."),
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
            progressed: !deliveryUnknown && body["delivery_unknown"] != .bool(true)
                && body["proof"]?.objectValue?["verdict"] == .string("verified"),
            successfulReadOnlyObservation: false)
    }

    /// Host static texts (plain labels, not tappable) with a position. Other host
    /// facts without a choice keep their screen text: duplicate names, images, values.
    private func staticTextFactElements() -> [JSONValue] {
        (currentScreenFacts?.namedElements() ?? []).filter { $0.role == "static_text" }.map { element in
            .object([
                "label": .string(element.text),
                "position": .object(["x_norm": .integer(element.x), "y_norm": .integer(element.y)]),
            ])
        }
    }

    /// Screen elements the loop withholds from the choices: disabled, covering
    /// other controls, or a target previously rejected or used up.
    private func withheldScreenElements() -> [JSONValue] {
        (currentScreenFacts?.namedElements() ?? []).compactMap { element in
            let intents = element.selectors.flatMap { name -> [NavigationIntent] in
                let tap = NavigationIntent(operation: .tap, selector: name.selector,
                    selectorKind: name.kind, role: element.role, desiredState: nil, text: nil)
                guard element.role == "switch" else { return [tap] }
                return [tap] + [false, true].map { state in
                    NavigationIntent(operation: .setBoolean, selector: name.selector,
                        selectorKind: name.kind, role: element.role, desiredState: state, text: nil)
                }
            }
            guard element.disabled || element.covering
                || intents.contains(where: { !isEligibleOfferedAction($0) }) else { return nil }
            return .object([
                "label": .string(element.text),
                "position": .object(["x_norm": .integer(element.x), "y_norm": .integer(element.y)]),
            ])
        }
    }

    #if DEBUG
    /// Test entry: the screenshot packet for `read` (the image's paired
    /// accessibility read) and `screenTextReply` (the OCR reply), built by the
    /// loop's own observation record, outcome and decision-packet code. Before the
    /// packet, `rejectedScreenText` taps are recorded as rejected on this screen and
    /// `usedUpSwitches` set_boolean actions as sent without effect up to the limit.
    func screenshotPacketForTesting(
        read: VisionCaptureMCPResult, screenTextReply: JSONValue,
        configuration: VisionCaptureAgentConfiguration, images: [AppImageAttachment],
        rejectedScreenText: [ScreenTextBlock] = [],
        usedUpSwitches: [(selector: String, desiredState: Bool)] = []
    ) throws -> String {
        try recordScreenObservation(from: read.value)
        for block in rejectedScreenText {
            rememberRejectedBeforeSubmissionProposal(Self.screenTextTapIntent(block))
        }
        for action in usedUpSwitches {
            let intent = NavigationIntent(operation: .setBoolean, selector: action.selector,
                selectorKind: nil, role: "switch", desiredState: action.desiredState, text: nil)
            for _ in 0..<Self.unverifiedAttemptLimit { noteExecuted(intent, result: read) }
        }
        let observed = try outcome(
            for: NavigationIntent(operation: .observe, selector: nil, selectorKind: nil,
                role: nil, desiredState: nil, text: nil),
            result: read, arguments: .object([:]), manifest: AuthorityManifest(state: "ready"),
            systemAlert: nil, progressed: false, observedScreenFacts: Self.returnedScreenFacts(from: read.value))
        guard var body = try JSONDecoder().decode(JSONValue.self, from: Data(observed.content.utf8)).objectValue else {
            throw VisionCaptureAgentError.malformedCall("The test read could not be encoded.")
        }
        body["observation_outcome"] = .string("succeeded")
        body["image_observation"] = .object(["state": .string("sequential")])
        Self.recordScreenText(Self.screenTextBlocks(in: screenTextReply), into: &body)
        body["operation"] = .string("screenshot")
        body["outcome"] = .string("succeeded")
        body["image"] = .object(["mime_type": .string("image/png")])
        return try decisionPacket(
            from: JSONValue.object(body).encoded(),
            call: AppToolCall(id: "screenshot-test", name: "visioncapture_navigate",
                arguments: .object(["action": .string("screenshot")])),
            configuration: configuration, images: images).content
    }

    /// Test entry: the packet after a dispatched `operation`, built by the loop's own
    /// observation record, outcome, after-action gate and decision-packet code.
    /// `read` is the follow-up read (nil: none), `body` adds result fields such as
    /// screen_changed or effect, and `screenTextReply` or `screenTextError` stands
    /// for the OCR read when the gate asks for one. `visionPackComplete` stands in
    /// for the vision pack probe (nil: the real probe). `userInstruction` is applied
    /// as the user's restrictions. `image` is the capture the loop adds to the next input.
    func actionPacketForTesting(
        operation: String = "tap", read: VisionCaptureMCPResult?, signatureBefore: String? = nil,
        body extra: [String: JSONValue], screenTextReply: JSONValue? = nil, screenTextError: Error? = nil,
        visionPackComplete: Bool? = nil, userInstruction: String? = nil,
        configuration: VisionCaptureAgentConfiguration
    ) throws -> (packet: String, readScreenText: Bool, image: AppImageAttachment?) {
        autoScreenImage = nil
        if let userInstruction { userRestrictions.apply(userInstruction) }
        visionPackCompleteForTesting = visionPackComplete
        var body: [String: JSONValue] = [:]
        if let read {
            try recordScreenObservation(from: read.value)
            let observed = try outcome(
                for: NavigationIntent(operation: .observe, selector: nil, selectorKind: nil,
                    role: nil, desiredState: nil, text: nil),
                result: read, arguments: .object([:]), manifest: AuthorityManifest(state: "ready"),
                systemAlert: nil, progressed: false, observedScreenFacts: Self.returnedScreenFacts(from: read.value))
            body = try JSONDecoder().decode(JSONValue.self, from: Data(observed.content.utf8)).objectValue ?? [:]
        } else {
            invalidateScreenObservation()
            body["observation_outcome"] = .string("unavailable")
        }
        body["operation"] = .string(operation)
        body["outcome"] = .string("succeeded")
        body.merge(extra) { _, new in new }
        let reads = Self.readsScreenTextAfterAction(
            body, signatureBefore: signatureBefore, signatureAfter: currentScreenSignature)
        if reads {
            let screenText = try screenTextError.map { try Self.screenTextReadFailure($0, afterAction: true) }
                ?? Self.screenTextBlocks(in: screenTextReply ?? .object([:]))
            if screenTextError == nil, let reply = screenTextReply {
                keepAutoScreenImage(VisionCaptureMCPResult(
                    value: reply, isError: false, refusalCode: nil, dispatchAttempted: nil,
                    hasConflictingDispatchAttemptEvidence: false,
                    isGuardedTargetRejectedBeforeSubmission: false,
                    isStaleActionCapabilityBeforeDispatch: false,
                    isSourceLayoutChangedBeforeRevalidation: false,
                    isActionAuthorizationExpiredBeforeDispatch: false,
                    isObservedTapTargetUnavailableBeforeDispatch: false,
                    isDeliveredTransitionContinuation: false,
                    isPointerPreCaptureFailureBeforeSubmission: false),
                    blocks: screenText.blocks, configuration: configuration)
            }
            Self.recordScreenText(screenText, into: &body)
        }
        let packet = try decisionPacket(
            from: JSONValue.object(body).encoded(),
            call: AppToolCall(id: "action-test", name: "visioncapture_navigate",
                arguments: .object(["action": .string(operation)])),
            configuration: configuration, images: [])
        var images: [AppImageAttachment] = []
        let content = try attachingAutoScreenImage(to: packet, images: &images)
        return (content, reads, images.first)
    }

    /// Test entry: what a new model context does to the kept screen-text words and IDs.
    func forgetKeptScreenTextForTesting() { forgetKeptScreenText() }

    /// Test entry: the repair packet the loop sends when it refuses a proposed call
    /// before dispatch, built as the main loop builds it.
    func refusalPacketForTesting(
        _ arguments: JSONValue, configuration: VisionCaptureAgentConfiguration
    ) throws -> String {
        let call = AppToolCall(id: "refusal-test", name: "visioncapture_navigate", arguments: arguments)
        do {
            _ = try navigationIntent(from: call, configuration: configuration)
        } catch let error as VisionCaptureAgentError where Self.isRecoverableProposalError(error) {
            let rejectedTarget = Self.expiredProposalTarget(call: call, error: error)
            let failure = try Self.proposalFailureResult(
                error, facts: currentProposalRepairFacts(configuration: configuration), rejectedTarget: rejectedTarget)
            let repairPacket = try decisionPacket(
                from: failure, call: call, configuration: configuration, images: [],
                excludingTargetID: rejectedTarget, preservingChoiceIDs: true)
            return try Self.addingProposalCorrection(to: repairPacket.content)
        }
        throw VisionCaptureAgentError.malformedCall("the proposal was accepted")
    }

    /// Test entry: the loop's refusal of a proposed call on the last packet (nil: accepted).
    func proposalRefusalForTesting(
        _ arguments: JSONValue, configuration: VisionCaptureAgentConfiguration
    ) -> String? {
        do {
            _ = try navigationIntent(
                from: AppToolCall(id: "proposal-test", name: "visioncapture_navigate", arguments: arguments),
                configuration: configuration)
            return nil
        } catch { return "\(error)" }
    }
    #endif

    private func isEligibleOfferedAction(_ intent: NavigationIntent) -> Bool {
        // A frame that covers other controls has no reliable tap point: a tap
        // at its centre lands on one of them (NestMind's "Create Profile"
        // renamed the profile that way). Keep it in the facts with its warning;
        // do not offer it as a tap.
        if intent.operation == .tap, let selector = intent.selector, let role = intent.role,
           currentScreenFacts?.coverageWarning(
               selector: selector, role: role, selectorKind: intent.selectorKind) != nil {
            return false
        }
        // A disabled control is a fact, never an offered action (run 25 typed
        // into a disabled field).
        if let selector = intent.selector, let role = intent.role,
           currentScreenFacts?.isEnabled(
               selector: selector, role: role, selectorKind: intent.selectorKind) == false {
            return false
        }
        do {
            try rejectPreviouslyRejectedProposal(intent)
            try rejectBlockedStaleAction(intent)
            try rejectExhaustedIntent(intent)
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

    /// Removes aliases for the same observed control before public choice IDs
    /// are allocated. Public facts remain part of the identity, so controls at
    /// different positions or with different selection state stay distinct.
    static func deduplicatedDecisionActions(
        _ actions: [JSONValue],
        facts: VisionCaptureScreenFacts?
    ) -> [JSONValue] {
        var seen: Set<String> = []
        return actions.enumerated().compactMap { offset, value in
            guard case .object(let object) = value,
                  case .string(let selector)? = object["selector"],
                  case .string(let role)? = object["role"] else {
                return value
            }
            let selectorKind: String?
            if case .string(let kind)? = object["selector_kind"] {
                selectorKind = kind
            } else {
                selectorKind = nil
            }
            let displayLabel: String?
            if case .string(let label)? = object["label"] {
                displayLabel = label
            } else {
                displayLabel = nil
            }
            var identity = object
            identity.removeValue(forKey: "requires_validation")
            if let target = facts?.semanticTargetIdentity(
                selector: selector,
                role: role,
                selectorKind: selectorKind,
                displayLabel: displayLabel,
                position: object["position"]) {
                identity.removeValue(forKey: "selector")
                identity.removeValue(forKey: "selector_kind")
                identity["_private_observed_target"] = .integer(Int64(target))
            }
            let key = (try? JSONValue.object(identity).encoded()) ?? "unencodable:\(offset)"
            return seen.insert(key).inserted ? value : nil
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
            if let elementID = field.elementID { object["element_id"] = .string(elementID) }
            if let facts {
                object.merge(facts.properties(
                    selector: field.selector, role: field.role,
                    selectorKind: field.selectorKind, elementID: field.elementID)) { existing, _ in existing }
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
    static func returnedEditableFields(
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
                      selector != "[REDACTED]",
                      !(selectorKind == "identifier"
                        && selector.lowercased().hasSuffix(".root")) else {
                    continue
                }
                candidates.append(PublishedEditableField(
                    selector: selector,
                    selectorKind: selectorKind,
                    role: role,
                    elementID: element["element_id"].flatMap {
                        if case .string(let id) = $0, !id.isEmpty { return id }
                        return nil
                    }))
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
                selector: exact, selectorKind: "placeholder", role: role, elementID: elementID))
        }

        let unambiguous = candidates.filter { candidate in
            candidates.filter {
                $0.selector == candidate.selector
                    && $0.selectorKind == candidate.selectorKind
            }.count == 1
        }
        let priority = ["identifier": 0, "label": 1, "placeholder": 2]
        var seenElements: Set<EditableElementIdentity> = []
        let oneSelectorPerElement = unambiguous.sorted { lhs, rhs in
            let leftPriority = priority[lhs.selectorKind] ?? Int.max
            let rightPriority = priority[rhs.selectorKind] ?? Int.max
            if leftPriority != rightPriority { return leftPriority < rightPriority }
            if lhs.selector != rhs.selector { return lhs.selector < rhs.selector }
            return lhs.role < rhs.role
        }.filter { field in
            guard let elementID = field.elementID else { return true }
            return seenElements.insert(EditableElementIdentity(
                elementID: elementID, role: field.role)).inserted
        }
        return oneSelectorPerElement.sorted { lhs, rhs in
            if lhs.selectorKind != rhs.selectorKind {
                return (priority[lhs.selectorKind] ?? Int.max)
                    < (priority[rhs.selectorKind] ?? Int.max)
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
        // Keep the last read for the effect diff after a pointer click.
        if let facts = currentScreenFacts { lastInvalidatedScreenFacts = facts }
        if let signature = currentScreenSignature { lastInvalidatedScreenSignature = signature }
        currentImageObservation = nil
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
        completedCycleActions.removeAll(keepingCapacity: true)
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
        if let completed = VerifiedTypingAction(intent) {
            verifiedTypingActions.insert(completed)
        }
        switch intent.operation {
        case .tap, .tapCoordinates, .computerUseClick,
             .setBoolean, .type, .back, .swipe:
            appendJourneyEvent(.action(JourneyAction(intent, readableLabel: resolvedJourneyLabel)))
        case .launch, .observe, .screenshot:
            break
        }
    }

    private static func explicitlyRequestsRepeatedTyping(
        userPrompt: String,
        text: String
    ) -> Bool {
        let prompt = userPrompt.lowercased()
        let typedText = text.lowercased()
        guard !typedText.isEmpty, prompt.contains(typedText),
              !prompt.contains("do not repeat"),
              !prompt.contains("don't repeat"),
              !prompt.contains("never repeat") else { return false }
        let words = prompt.split { !$0.isLetter && !$0.isNumber }
        if words.contains(where: { $0 == "repeat" || $0 == "twice" || $0 == "again" }) {
            return true
        }
        return prompt.contains("two times") || prompt.contains("2 times")
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
            completedCycleActions.removeAll(keepingCapacity: true)
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
        completedCycleActions = interveningActions

        if !interveningActions.isEmpty {
            let recent = interveningActions
                .suffix(Self.journeyHintActionLimit)
                .map(Self.journeyActionDescription)
                .joined(separator: " -> ")
            currentJourneyHint =
                "This screen returned after these verified actions: \(recent). Compare it with the unfinished user goal. Continue any missing save or verification step; otherwise move to the next requested check. Do not repeat an action whose result is already proven."
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

    /// Counts a dispatched published action. A verified effect clears the
    /// counters: the task moved on. Three sends without a verified effect
    /// withhold that action until something is verified.
    private func noteExecuted(_ intent: NavigationIntent, result: VisionCaptureMCPResult) {
        guard intent.operation == .tap || intent.operation == .setBoolean else { return }
        let proof = try? Self.sanitizedNamedObject(
            "proof", allowedKeys: ["verdict", "verdict_source", "reason_code", "action"],
            in: result.value)
        // A changed screen after the action is progress, even without a
        // typed verdict. Only an action that changed nothing counts against it.
        var changes: [[String: JSONValue]] = []
        try? Self.collectStructuredObjects(named: "changes", in: result.value) { changes.append($0) }
        // A pointer click carries no `changes` block; its screen observation
        // says whether the screen changed after the click.
        var observations: [[String: JSONValue]] = []
        try? Self.collectStructuredObjects(named: "screen_observation", in: result.value) { observations.append($0) }
        let screenChanged = changes.first?["screen_changed"] == .bool(true)
            || observations.first?["status"] == .string("changed_observed")
        if proof?["verdict"] == .string("verified") || screenChanged {
            unverifiedAttempts.removeAll()
        } else {
            unverifiedAttempts[intent, default: 0] += 1
        }
    }

    private func rejectExhaustedIntent(_ intent: NavigationIntent) throws {
        guard unverifiedAttempts[intent, default: 0] >= Self.unverifiedAttemptLimit else { return }
        throw VisionCaptureAgentError.navigationUnavailable(
            "This action was already sent \(Self.unverifiedAttemptLimit) times without a verified effect, so it is withheld now. Do not repeat it. Use another way to reach the goal: type into the field, choose the return key, or take a screenshot, then send \(VisionCaptureToolDefinitions.coordinateTapCall) at the visible control.")
    }

    private func rememberRejectedBeforeSubmissionProposal(
        _ intent: NavigationIntent,
        screen suppliedScreen: NavigationScreenScope? = nil
    ) {
        guard let screen = suppliedScreen ?? currentNavigationScreenScope else { return }
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
        case .tapCoordinates:
            return "tap visible \(label ?? "control")"
        case .computerUseClick:
            return "click visible \(label ?? "control")"
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

    private static func returnedComputerUseTaskIdentity(
        from value: JSONValue
    ) throws -> ComputerUseTaskIdentity {
        var identities: Set<ComputerUseTaskIdentity> = []
        collectComputerUseTaskIdentities(in: value, into: &identities)
        guard identities.count == 1, let identity = identities.first else {
            throw VisionCaptureAgentError.malformedCall(
                "VisionCapture returned no single valid computer-use task identity")
        }
        return identity
    }

    private static func collectComputerUseTaskIdentities(
        in value: JSONValue,
        into identities: inout Set<ComputerUseTaskIdentity>
    ) {
        switch value {
        case .object(let object):
            if case .string(let id)? = object["computer_use_task_id"],
               UUID(uuidString: id) != nil,
               case .integer(let generation)? = object["computer_use_generation"],
               generation > 0,
               generation <= Int64(Int.max) {
                identities.insert(ComputerUseTaskIdentity(
                    id: id,
                    generation: Int(generation)))
            }
            for child in object.values {
                collectComputerUseTaskIdentities(in: child, into: &identities)
            }
        case .array(let array):
            for child in array {
                collectComputerUseTaskIdentities(in: child, into: &identities)
            }
        case .string(let text):
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            let idPrefix = "computer_use_task_id:"
            let generationPrefix = "computer_use_generation:"
            let id = lines.compactMap { line -> String? in
                let value = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard value.hasPrefix(idPrefix) else { return nil }
                return String(value.dropFirst(idPrefix.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }.first
            let generation = lines.compactMap { line -> Int? in
                let value = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard value.hasPrefix(generationPrefix) else { return nil }
                return Int(value.dropFirst(generationPrefix.count)
                    .trimmingCharacters(in: .whitespacesAndNewlines))
            }.first
            if let id, UUID(uuidString: id) != nil,
               let generation, generation > 0 {
                identities.insert(ComputerUseTaskIdentity(
                    id: id,
                    generation: generation))
            }
            for embedded in embeddedJSONValues(in: text) {
                collectComputerUseTaskIdentities(in: embedded, into: &identities)
            }
        default:
            break
        }
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

    static func validateReturnedIdentity(
        in value: JSONValue,
        configuration: VisionCaptureAgentConfiguration,
        refusalCode: String?,
        permittedNonTargetLaunchOutcome: [String: JSONValue]? = nil,
        permitsUncertainRecipientAfterSubmission: Bool = false,
        permitsForeignObservedBundleID: Bool = false
    ) throws {
        var inspectedEmbeddedTexts: Set<String> = []
        var interactionEvidenceCopies = InteractionEvidenceCopies()
        collectInteractionEvidenceCopies(
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            configuration: configuration,
            into: &interactionEvidenceCopies)
        inspectedEmbeddedTexts.removeAll(keepingCapacity: true)
        try validateReturnedIdentityFields(
            in: value,
            inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
            configuration: configuration,
            path: "$",
            refusalCode: refusalCode,
            permittedNonTargetLaunchOutcome: permittedNonTargetLaunchOutcome,
            globallyPermitsNullObservedBundleID:
                interactionEvidenceCopies.containsNullObservedBundleID
                && ((interactionEvidenceCopies.allCopiesProveRejection
                    && interactionEvidenceCopies.allDispatchFactsProveNoSubmission)
                    || (permitsUncertainRecipientAfterSubmission
                        && interactionEvidenceCopies
                            .allCopiesProveUncertainAfterSubmission)),
            permitsForeignObservedBundleID: permitsForeignObservedBundleID)
    }

    private struct InteractionEvidenceCopies {
        var containsNullObservedBundleID = false
        var allCopiesProveRejection = true
        var allCopiesProveUncertainAfterSubmission = true
        var allDispatchFactsProveNoSubmission = true
    }

    private static func collectInteractionEvidenceCopies(
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>,
        configuration: VisionCaptureAgentConfiguration,
        into copies: inout InteractionEvidenceCopies
    ) {
        switch value {
        case .object(let object):
            for key in ["dispatch_attempted", "submission_started",
                        "delivery_acknowledged", "mutation_sent"] {
                if let returned = object[key], returned != .bool(false) {
                    copies.allDispatchFactsProveNoSubmission = false
                }
            }
            if let returned = object["interaction_evidence"] {
                let evidenceCopies: [[String: JSONValue]]
                switch returned {
                case .object(let evidence): evidenceCopies = [evidence]
                case .array(let values):
                    evidenceCopies = values.compactMap(\.objectValue)
                    if evidenceCopies.count != values.count {
                        copies.allCopiesProveRejection = false
                        copies.allCopiesProveUncertainAfterSubmission = false
                    }
                default:
                    evidenceCopies = []
                    copies.allCopiesProveRejection = false
                    copies.allCopiesProveUncertainAfterSubmission = false
                }
                if evidenceCopies.isEmpty {
                    copies.allCopiesProveRejection = false
                    copies.allCopiesProveUncertainAfterSubmission = false
                }
                for evidence in evidenceCopies {
                    if evidence["binding"]?.objectValue?["observed_bundle_id"] == .null {
                        copies.containsNullObservedBundleID = true
                    }
                    if !permitsNullObservedBundleID(
                        in: evidence,
                        configuration: configuration) {
                        copies.allCopiesProveRejection = false
                    }
                    if !permitsNullObservedBundleIDForUncertainSubmission(
                        in: evidence,
                        configuration: configuration)
                        && !permitsNullObservedBundleIDForUnacknowledgedComputerUse(
                            in: evidence,
                            configuration: configuration) {
                        copies.allCopiesProveUncertainAfterSubmission = false
                    }
                }
            }
            for child in object.values {
                collectInteractionEvidenceCopies(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    configuration: configuration,
                    into: &copies)
            }
        case .array(let values):
            for child in values {
                collectInteractionEvidenceCopies(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    configuration: configuration,
                    into: &copies)
            }
        case .string(let text):
            guard inspectedEmbeddedTexts.insert(text).inserted else { return }
            for embedded in embeddedJSONValues(in: text) {
                collectInteractionEvidenceCopies(
                    in: embedded,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    configuration: configuration,
                    into: &copies)
            }
        default:
            break
        }
    }

    /// A pre-dispatch failure has no observed recipient. VisionCapture reports
    /// that fact as a null observed bundle in the evidence binding. Accept it
    /// only when that same complete evidence copy proves that submission never
    /// began and remains bound to the configured request identity.
    private static func permitsNullObservedBundleID(
        in interactionEvidence: [String: JSONValue],
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard let binding = interactionEvidence["binding"]?.objectValue,
              binding["observed_bundle_id"] == .null,
              binding["requested_bundle_id"] == .string(configuration.bundleIdentifier),
              binding["udid"] == .string(configuration.simulatorUDID),
              binding["observed_pid"] == .null,
              let dispatch = interactionEvidence["dispatch"]?.objectValue,
              dispatch["status"] == .string("rejected_before_submission"),
              dispatch["submission_started"] == .bool(false),
              dispatch["delivery_acknowledged"] == .bool(false),
              let outcome = interactionEvidence["outcome"]?.objectValue,
              outcome["status"] == .string("failed"),
              outcome["scope"] == .string("dispatch"),
              outcome["reason_code"] == .string("DISPATCH_REJECTED_BEFORE_SUBMISSION"),
              let target = interactionEvidence["target"]?.objectValue,
              target["actual_event_recipient_observed"] == .bool(false) else {
            return false
        }
        return (target["status"] == .string("unavailable")
                    && target["reason_code"] == .string("TARGET_UNAVAILABLE"))
            || (target["status"] == .string("ambiguous")
                    && target["reason_code"] == .string("TARGET_AMBIGUOUS"))
    }

    /// Unknown delivery has no trusted recipient identity. Accept that null
    /// only for the exact truth contract used by read-only recovery.
    private static func permitsNullObservedBundleIDForUncertainSubmission(
        in interactionEvidence: [String: JSONValue],
        configuration: VisionCaptureAgentConfiguration
    ) -> Bool {
        guard let binding = interactionEvidence["binding"]?.objectValue,
              binding["observed_bundle_id"] == .null,
              binding["requested_bundle_id"]
                == .string(configuration.bundleIdentifier),
              binding["udid"] == .string(configuration.simulatorUDID),
              binding["observed_pid"] == .null,
              let dispatch = interactionEvidence["dispatch"]?.objectValue,
              dispatch["status"] == .string("uncertain_after_submission"),
              dispatch["submission_started"] == .bool(true),
              dispatch["delivery_acknowledged"] == .bool(false),
              let outcome = interactionEvidence["outcome"]?.objectValue,
              outcome["status"] == .string("inconclusive"),
              outcome["scope"] == .string("none"),
              outcome["reason_code"] == .string(
                  "DISPATCH_UNCERTAIN_AFTER_SUBMISSION"),
              let target = interactionEvidence["target"]?.objectValue,
              target["actual_event_recipient_observed"] == .bool(false),
              target["status"] == .string("unavailable"),
              target["reason_code"] == .string("TARGET_UNAVAILABLE") else {
            return false
        }
        return true
    }

    /// Computer Use can submit a pointer click and then lose its terminal
    /// framebuffer receipt. This contract proves uncertainty, never success.
    private static func permitsNullObservedBundleIDForUnacknowledgedComputerUse(
        in interactionEvidence: [String: JSONValue],
        configuration: VisionCaptureAgentConfiguration,
        expectedTaskID: String? = nil,
        expectedGeneration: Int64? = nil,
        expectedX: Int64? = nil,
        expectedY: Int64? = nil
    ) -> Bool {
        guard interactionEvidence["lane"] == .string("computer_use"),
              let binding = interactionEvidence["binding"]?.objectValue,
              binding["observed_bundle_id"] == .null,
              binding["requested_bundle_id"]
                == .string(configuration.bundleIdentifier),
              binding["udid"] == .string(configuration.simulatorUDID),
              case .string(let laneOwnerID)? = binding["lane_owner_id"],
              UUID(uuidString: laneOwnerID) != nil,
              case .integer(let laneGeneration)? = binding["lane_generation"],
              laneGeneration > 0,
              let dispatch = interactionEvidence["dispatch"]?.objectValue,
              dispatch["status"] == .string("submitted_unacknowledged"),
              dispatch["submission_started"] == .bool(true),
              dispatch["delivery_acknowledged"] == .bool(false),
              let outcome = interactionEvidence["outcome"]?.objectValue,
              outcome["status"] == .string("inconclusive"),
              outcome["scope"] == .string("none"),
              outcome["reason_code"] == .string("EXPECTED_OUTCOME_MISSING"),
              let screen = interactionEvidence["screen_observation"]?.objectValue,
              screen["status"] == .string("observation_unavailable"),
              screen["reason_code"] == .string(
                  "SCREEN_OBSERVATION_DEADLINE_EXCEEDED"),
              screen["causal_attribution"] == .string("not_established"),
              let target = interactionEvidence["target"]?.objectValue,
              target["actual_event_recipient_observed"] == .bool(false),
              let point = target["frozen_requested_point"]?.objectValue,
              case .integer(let x)? = point["x"], (0...1000).contains(x),
              case .integer(let y)? = point["y"], (0...1000).contains(y) else {
            return false
        }
        let observedPIDIsValid: Bool
        switch binding["observed_pid"] {
        case .null?: observedPIDIsValid = true
        case .integer(let pid)?: observedPIDIsValid = pid > 0
        default: observedPIDIsValid = false
        }
        guard observedPIDIsValid,
              expectedTaskID.map({ $0 == laneOwnerID }) ?? true,
              expectedGeneration.map({ $0 == laneGeneration }) ?? true,
              expectedX.map({ $0 == x }) ?? true,
              expectedY.map({ $0 == y }) ?? true else {
            return false
        }
        return (target["status"] == .string("ambiguous")
                    && target["reason_code"] == .string("TARGET_AMBIGUOUS"))
            || (target["status"] == .string("unavailable")
                    && target["reason_code"] == .string("TARGET_UNAVAILABLE")
                    && target["reason_detail"] == .string("read_after_screen_change"))
    }

    private static func validateReturnedIdentityFields(
        in value: JSONValue,
        inspectedEmbeddedTexts: inout Set<String>,
        configuration: VisionCaptureAgentConfiguration,
        path: String,
        refusalCode: String?,
        permittedNonTargetLaunchOutcome: [String: JSONValue]?,
        isPermittedLaunchOutcome: Bool = false,
        isPermittedNonTargetForeground: Bool = false,
        isInteractionEvidence: Bool = false,
        globallyPermitsNullObservedBundleID: Bool = false,
        permitsNullObservedBundleID: Bool = false,
        permitsForeignObservedBundleID: Bool = false
    ) throws {
        switch value {
        case .object(let object):
            let permitsNullObservedBundleIDInBinding = globallyPermitsNullObservedBundleID
                && isInteractionEvidence
                && (Self.permitsNullObservedBundleID(
                        in: object,
                        configuration: configuration)
                    || Self.permitsNullObservedBundleIDForUncertainSubmission(
                        in: object,
                        configuration: configuration)
                    || Self.permitsNullObservedBundleIDForUnacknowledgedComputerUse(
                        in: object,
                        configuration: configuration))
            for (key, child) in object {
                let childPath = path + "[" + (try JSONValue.string(key).encoded()) + "]"
                if ["bundle_id", "requested_bundle_id", "observed_bundle_id",
                    "bundle_id_requested", "bundle_id_active"]
                    .contains(key),
                   child != .string(configuration.bundleIdentifier),
                   !(permitsNullObservedBundleID
                       && key == "observed_bundle_id"
                       && child == .null),
                   // A pointer click is delivered by coordinates; the observed
                   // bundle is informational and may name a transition owner.
                   !(permitsForeignObservedBundleID && key == "observed_bundle_id"),
                   !(isPermittedNonTargetForeground && key == "bundle_id") {
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
                    permittedNonTargetLaunchOutcome:
                        permittedNonTargetLaunchOutcome,
                    isPermittedLaunchOutcome: key == "launch_outcome"
                        && permittedNonTargetLaunchOutcome != nil
                        && child.objectValue == permittedNonTargetLaunchOutcome,
                    isPermittedNonTargetForeground:
                        isPermittedLaunchOutcome && key == "foreground"
                        && (child.objectValue?["state"] == .string("other")
                            || child.objectValue?["state"] == .string("system")),
                    isInteractionEvidence: key == "interaction_evidence",
                    globallyPermitsNullObservedBundleID:
                        globallyPermitsNullObservedBundleID,
                    permitsNullObservedBundleID: key == "binding"
                        && permitsNullObservedBundleIDInBinding,
                    permitsForeignObservedBundleID: permitsForeignObservedBundleID)
            }
        case .array(let array):
            for (index, child) in array.enumerated() {
                try validateReturnedIdentityFields(
                    in: child,
                    inspectedEmbeddedTexts: &inspectedEmbeddedTexts,
                    configuration: configuration,
                    path: path + "[\(index)]",
                    refusalCode: refusalCode,
                    permittedNonTargetLaunchOutcome:
                        permittedNonTargetLaunchOutcome,
                    isInteractionEvidence: isInteractionEvidence,
                    globallyPermitsNullObservedBundleID:
                        globallyPermitsNullObservedBundleID,
                    permitsForeignObservedBundleID: permitsForeignObservedBundleID)
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
                    permittedNonTargetLaunchOutcome:
                        permittedNonTargetLaunchOutcome,
                    globallyPermitsNullObservedBundleID:
                        globallyPermitsNullObservedBundleID,
                    permitsForeignObservedBundleID: permitsForeignObservedBundleID)
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
              staleActionConfirmation == nil else { return nil }
        if let alert = currentSystemAlert {
            return (
                Self.sanitizedSystemAlertActions(alert),
                [],
                "A native iOS system alert is on screen: \(alert.title)")
        }
        guard
              let signature = currentScreenSignature,
              let facts = currentScreenFacts else { return nil }
        let publishedActions = eligibleOfferedActions(currentManifest.actions).filter { action in
            !facts.isLowValueSoftwareKeyboardControl(
                selector: action.selector,
                role: action.role,
                displayLabel: action.displayLabel,
                position: action.displayPosition)
        }
        var actions = Self.sanitizedActions(publishedActions, facts: facts)
        if let offered = offeredTapCandidates, offered.signature == signature {
            for candidate in offered.candidates {
                guard !currentManifest.actions.contains(where: {
                    $0.selector == candidate.selector && $0.role == candidate.role
                }), !facts.isLowValueSoftwareKeyboardControl(
                    selector: candidate.selector, role: candidate.role) else { continue }
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
        actions = Self.deduplicatedDecisionActions(actions, facts: facts)
        let fields = Self.sanitizedEditableFields(currentEditableFields, facts: facts)
        return (actions, fields, facts.summary(availableActions: actions, editableFields: fields))
    }

    private static func proposalFailureResult(
        _ error: VisionCaptureAgentError,
        facts: (actions: [JSONValue], fields: [JSONValue], summary: String?)?,
        rejectedTarget: String? = nil
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
        if let rejectedTarget {
            body["rejected_target"] = .string(rejectedTarget)
            body["dispatch_attempted"] = .bool(false)
        }
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
                        ? " Current accessibility facts are unavailable. Choose a read-only action from allowed_next for fresh choices."
                        : " Use only the current choices."))
        default:
            break
        }
        return try JSONValue.object(body).encoded()
    }

    static func addingProposalCorrection(to content: String) throws -> String {
        guard var packet = try JSONDecoder().decode(JSONValue.self,
                  from: Data(content.utf8)).objectValue,
              case .array(let allowed)? = packet["allowed_next"] else {
            throw VisionCaptureAgentError.malformedCall("Missing allowed actions in the correction.")
        }
        let actions = allowed.compactMap { value -> String? in
            guard case .string(let action) = value else { return nil }
            return action
        }.joined(separator: ", ")
        let reason: String
        if case .string(let guidance)? = packet["guidance"] {
            reason = guidance
        } else {
            reason = "The proposed action was not sent."
        }
        packet["guidance"] = .string(reason
            + " Return a corrected response. Allowed actions now: [\(actions)]."
            + " Use an action from this list and a current choice that supports it."
            + " Do not use an old target ID. Do not request observe or screenshot unless listed."
            + " If no permitted action can advance the task, explain the blocker in a final answer without a tool call."
            + " Completed actions remain completed; do not repeat them.")
        return try JSONValue.object(packet).encoded()
    }

    /// The one rule for choosing a target in a packet that carries a current image.
    static let visualChoiceRule =
        "Tap a current choice by ID, or send \(VisionCaptureToolDefinitions.coordinateTapCall) for a control you can see in this image when no choice matches it. If the control you need is not visible, take another path (close the sheet, go back) or report a blocker. Do not repeat a coordinate tap that had no effect."

    static let screenshotPairInstruction =
        "The image was captured before the current accessibility read. Choices and positions come from that read and any following cache validation; visual agreement is unverified. If a choice label matches a visible control but its position does not, do not use that target ID; use an offered visual click at the visible control. If the target remains ambiguous, report the uncertainty. "
            + visualChoiceRule

    private static let coordinateEvidenceReason =
        "Coordinate actions need current screenshot evidence. Choose screenshot now, then use positions from that screenshot in the next step."

    static let choiceTargetReason = "Targets are choice IDs; for positions use x_norm and y_norm."

    static let screenTextSelectorKind = "screen_text"

    /// One OCR text block of a screenshot: its text and centre on the 0-1000 grid.
    struct ScreenTextBlock: Equatable, Sendable {
        let text: String
        let xNorm: Int64
        let yNorm: Int64
    }

    /// The OCR blocks of a plain screenshot reply (pixel frames, top-left origin)
    /// in the host's order, or the reason there are none.
    static func screenTextBlocks(in value: JSONValue) -> (blocks: [ScreenTextBlock], unavailableReason: String?) {
        func number(_ value: JSONValue?) -> Double? {
            switch value {
            case .integer(let v)?: Double(v)
            case .unsignedInteger(let v)?: Double(v)
            case .decimal(let v)?: NSDecimalNumber(decimal: v).doubleValue
            case .number(let v)?: v.isFinite ? v : nil
            default: nil
            }
        }
        guard case .array(let content)? = value.objectValue?["content"] else {
            return ([], "the screenshot reply had no content")
        }
        for item in content {
            guard let object = item.objectValue, object["type"] == .string("text"),
                  case .string(let text)? = object["text"],
                  let metadata = (try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)))?.objectValue,
                  let ocrValue = metadata["ocr"] else { continue }
            guard let ocr = ocrValue.objectValue else { return ([], "the OCR result was not readable") }
            if case .string(let error)? = ocr["error"] {
                return ([], String(error.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160)))
            }
            guard let width = number(metadata["width"]), let height = number(metadata["height"]),
                  width > 0, height > 0 else { return ([], "the screenshot size was missing") }
            guard case .array(let rawBlocks)? = ocr["blocks"] else { return ([], "the OCR result had no blocks") }
            var blocks: [ScreenTextBlock] = []
            for raw in rawBlocks {
                guard let block = raw.objectValue, case .string(let rawText)? = block["text"],
                      let frame = block["frame"]?.objectValue,
                      let x = number(frame["x"]), let y = number(frame["y"]),
                      let w = number(frame["width"]), let h = number(frame["height"]) else { continue }
                let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                let xNorm = Int64(min(max(((x + w / 2) / width * 1000).rounded(), 0), 1000))
                let yNorm = Int64(min(max(((y + h / 2) / height * 1000).rounded(), 0), 1000))
                blocks.append(ScreenTextBlock(text: text, xNorm: xNorm, yNorm: yNorm))
            }
            return blocks.isEmpty ? ([], "no text was found on the screen") : (blocks, nil)
        }
        return ([], "the screenshot reply had no OCR result")
    }

    /// Blocks outside the top status-bar strip (system UI, not the app), that no
    /// known element (a choice, a held-back or withheld control, or a static text)
    /// already names within 15 units on both axes, and that the user has not prohibited.
    static func screenTextBlocksToOffer(
        _ blocks: [ScreenTextBlock], knownElements: [JSONValue],
        restrictions: AgentUserRestrictions = AgentUserRestrictions(), screenContext: String? = nil
    ) -> [ScreenTextBlock] {
        let named: [(label: String, x: Int64, y: Int64)] = knownElements.compactMap { choice in
            guard let object = choice.objectValue, case .string(var label)? = object["label"],
                  let position = object["position"]?.objectValue,
                  case .integer(let x)? = position["x_norm"], case .integer(let y)? = position["y_norm"]
            else { return nil }
            if label.hasSuffix(" (ocr)") { label.removeLast(" (ocr)".count) }
            return (label, x, y)
        }
        return blocks.filter { block in
            block.yNorm > statusBarStripMaximumY
                && !named.contains { $0.label == block.text && abs($0.x - block.xNorm) <= 15 && abs($0.y - block.yNorm) <= 15 }
                && !restrictions.prohibits(label: block.text, selector: block.text, screenContext: screenContext)
        }
    }

    static let pickerWheelNote = "These look like picker wheel rows: tap the row you want; if it is not visible, tap the end row nearest to it; do not swipe these rows."

    /// Picker wheel rows among the offered screen-text words (words the host does not
    /// know, after the lean rule): at least 5 words in one vertical column, each the
    /// next row down. Thresholds from measured wheels (two date pickers, 15 reads):
    /// row gaps are 21 to 38 units (rows are closer at the ends than in the middle),
    /// so a gap must be 18 to 44; a calendar day grid's rows are 50 to 53 apart and
    /// never qualify. Word centres in a column of left-aligned names spread up to 90
    /// units (years up to 10), so a row joins within 80 of the first row and a run
    /// spreads at most 110; day-grid columns are about 105 apart. Known limit: any 5
    /// or more OCR-only lines 18 to 44 apart (a wrapped paragraph in an image) also
    /// fire; the hint is harmless there.
    static func looksLikePickerWheel(_ blocks: [ScreenTextBlock]) -> Bool {
        let rows = blocks.sorted { $0.yNorm < $1.yNorm }
        for (index, first) in rows.enumerated() {
            var run = [first]
            for block in rows[(index + 1)...] where abs(block.xNorm - first.xNorm) <= 80 {
                let gap = block.yNorm - run[run.count - 1].yNorm
                if gap < 18 { continue }
                if gap > 44 { break }
                run.append(block)
            }
            let xs = run.map(\.xNorm)
            if run.count >= 5, let low = xs.min(), let high = xs.max(), high - low <= 110 { return true }
        }
        return false
    }

    /// The status bar (clock, signal, battery) has its centre at y_norm 50 or less.
    static let statusBarStripMaximumY: Int64 = 50

    /// The tap a screen-text choice sends: its own text and position.
    private static func screenTextTapIntent(_ block: ScreenTextBlock) -> NavigationIntent {
        NavigationIntent(operation: .tap, selector: block.text, selectorKind: screenTextSelectorKind,
            role: "text", desiredState: nil, text: nil, xNorm: Int(block.xNorm), yNorm: Int(block.yNorm))
    }

    static func screenTextLabel(_ text: String) -> String { "\(text) (screen text)" }

    static func screenTextChoice(_ block: ScreenTextBlock, id: String) -> JSONValue {
        .object([
            "id": .string(id), "role": .string("text"), "label": .string(screenTextLabel(block.text)),
            "operations": .array([.string(NavigationOperation.tap.rawValue)]),
            "position": .object(["x_norm": .integer(block.xNorm), "y_norm": .integer(block.yNorm)]),
        ])
    }

    /// A tap on a screen-text choice clicks its own position. The OCR read that
    /// produced it is the evidence; no screenshot is needed.
    static func screenTextPointerClick(
        selectorKind: String?, selector: String?, xNorm: Int?, yNorm: Int?
    ) -> (x: Int64, y: Int64, intent: String)? {
        guard selectorKind == screenTextSelectorKind, let selector, let xNorm, let yNorm else { return nil }
        return (Int64(xNorm), Int64(yNorm), "tap \(selector)")
    }

    /// Errors that end the run in any host read. After an action, an alert
    /// refusal is not one: the OCR read turns it into the note.
    static func isFatalScreenTextReadError(_ error: VisionCaptureAgentError, afterAction: Bool = false) -> Bool {
        switch error {
        case .returnedIdentityMismatch, .sessionIdentityMismatch, .malformedCall, .invalidConfiguration: true
        case .unsupportedSystemInteraction: !afterAction
        default: false
        }
    }

    /// The host's plain reason for a refused or failed OCR read, never the
    /// loop's own error text.
    static func screenTextFailureReason(_ error: Error) -> String {
        switch error {
        case VisionCaptureAgentError.mcpOutcome(let outcome):
            screenTextRefusalReason(code: outcome.reasonCode, reason: outcome.reason)
        case VisionCaptureAgentError.unsupportedSystemInteraction(let code, let outcome):
            screenTextRefusalReason(code: code, reason: outcome?.reason)
        case let refusal as BusyHostReadBeforeDispatch:
            screenTextRefusalReason(code: refusal.failure.reasonCode, reason: refusal.failure.reason)
        case let refusal as AmbiguousTargetBeforeDispatch:
            screenTextRefusalReason(code: refusal.failure.reasonCode, reason: refusal.failure.reason)
        default:
            screenTextRefusalReason(code: nil, reason: nil)
        }
    }

    /// The refusal code and the first sentence of its reason.
    static func screenTextRefusalReason(code: String?, reason: String?) -> String {
        let code = String((code?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "").prefix(160))
        var short = reason?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if let end = short.range(of: ". ") { short = String(short[..<end.lowerBound]) }
        if short.count > 160 { short = String(short.prefix(160)) }
        switch (code.isEmpty, short.isEmpty) {
        case (false, false): return "\(code): \(short)"
        case (false, true): return code
        case (true, false): return short
        case (true, true): return "the host did not return screen text"
        }
    }

    /// A dispatched action gets an OCR read after it when a current read follows,
    /// delivery is known, and the screen changed: screen_changed, an effect that
    /// lists appeared or disappeared text, or a new screen signature.
    static func readsScreenTextAfterAction(
        _ body: [String: JSONValue], signatureBefore: String?, signatureAfter: String?
    ) -> Bool {
        guard let signatureAfter, body["observation_outcome"] != .string("unavailable"),
              body["delivery_unknown"] != .bool(true),
              body["outcome"] != .string("delivery_unknown_reobserved"),
              body["outcome"] != .string("unknown_reobserved") else { return false }
        if body["screen_changed"] == .bool(true) { return true }
        if case .string(let effect)? = body["effect"], effect.contains("appeared: ") { return true }
        return signatureAfter != signatureBefore
    }

    /// A failed OCR read: identity, session, malformed-reply and configuration
    /// errors end the run; so does an alert refusal on the screenshot path. Any
    /// other refusal or failure becomes the plain note.
    static func screenTextReadFailure(
        _ error: Error, afterAction: Bool
    ) throws -> (blocks: [ScreenTextBlock], unavailableReason: String?) {
        if let error = error as? VisionCaptureAgentError,
           isFatalScreenTextReadError(error, afterAction: afterAction) { throw error }
        return ([], screenTextFailureReason(error))
    }

    /// Puts the OCR read into the packet body: its blocks, or why there are none.
    static func recordScreenText(
        _ screenText: (blocks: [ScreenTextBlock], unavailableReason: String?),
        into body: inout [String: JSONValue]
    ) {
        if let reason = screenText.unavailableReason {
            body["screen_text_unavailable"] = .string(reason)
        } else {
            body["screen_text"] = .array(screenText.blocks.map {
                .object(["text": .string($0.text), "x_norm": .integer($0.xNorm), "y_norm": .integer($0.yNorm)])
            })
        }
    }

    /// The packet says once why screen-text choices are missing.
    static func addingScreenTextNote(to content: String, body: [String: JSONValue]) throws -> String {
        guard case .string(let reason)? = body["screen_text_unavailable"] else { return content }
        return try addingGuidanceNote(to: content, note: screenTextUnavailableNote(reason))
    }

    static func screenTextUnavailableNote(_ reason: String) -> String {
        var reason = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        while reason.hasSuffix(".") { reason.removeLast() }
        return "Screen text positions unavailable: \(reason)."
    }

    static let coordinatesOverTargetNote =
        "Used your coordinates; next time send x_norm and y_norm without a target, or a choice ID alone."

    /// A tap or type with both a target and coordinates: a choice ID wins and the
    /// coordinates are ignored; free text yields to the coordinates, so a tap becomes
    /// tap_coordinates and a type takes the position path with its field guard.
    static func normalizedTargetAndCoordinates(
        _ proposed: [String: JSONValue]
    ) -> (object: [String: JSONValue], usedCoordinates: Bool) {
        guard case .string(let action)? = proposed["action"], action == "tap" || action == "type",
              case .string(let target)? = proposed["target"],
              proposed["x_norm"] != nil || proposed["y_norm"] != nil else { return (proposed, false) }
        var object = proposed
        if isChoiceIDShaped(target) {
            object.removeValue(forKey: "x_norm")
            object.removeValue(forKey: "y_norm")
            return (object, false)
        }
        object.removeValue(forKey: "target")
        if action == "tap" {
            object["action"] = .string(NavigationOperation.tapCoordinates.rawValue)
            if object["intent"] == nil { object["intent"] = .string("tap at the given position") }
        }
        return (object, true)
    }

    static func addingGuidanceNote(to content: String, note: String) throws -> String {
        guard case .object(var body)? = try? JSONDecoder().decode(
            JSONValue.self, from: Data(content.utf8)) else {
            throw VisionCaptureAgentError.malformedCall("the loop could not add its note")
        }
        if case .string(let guidance)? = body["guidance"], !guidance.isEmpty {
            body["guidance"] = .string(note + " " + guidance)
        } else {
            body["guidance"] = .string(note)
        }
        return try JSONValue.object(body).encoded()
    }

    /// Choice IDs are "c" followed by digits. Anything else (a fact line, a label)
    /// is free text and never selects a control.
    static func isChoiceIDShaped(_ target: String) -> Bool {
        target.range(of: #"^c[0-9]+$"#, options: .regularExpression) != nil
    }

    private static let expiredTargetReason =
        "The target ID is expired, ambiguous, or unavailable. Choose a current choice. Old IDs cannot be restored."

    private static func expiredProposalTarget(call: AppToolCall, error: VisionCaptureAgentError) -> String? {
        guard call.name == VisionCaptureToolDefinitions.navigateName,
              error == .navigationUnavailable(expiredTargetReason),
              case .string(let target)? = call.arguments.objectValue?["target"] else { return nil }
        return target
    }

    private static func addingRepeatedTargetCorrection(to content: String) throws -> String {
        guard var packet = try JSONDecoder().decode(JSONValue.self,
                  from: Data(content.utf8)).objectValue,
              let lastAction = packet["last_action"]?.objectValue,
              lastAction["verdict"] == .string("not_sent"),
              lastAction["dispatch_attempted"] == .bool(false),
              case .string(_)? = lastAction["rejected_target"],
              case .string(let guidance)? = packet["guidance"] else {
            throw VisionCaptureAgentError.malformedCall("The rejected target correction could not retain its not-sent result.")
        }
        packet["guidance"] = .string(guidance
            + " This is the second equivalent rejected proposal. Do not reuse rejected_target. Choose a different current choice or a permitted read-only observation. A screenshot cannot renew an old ID or bypass a refusal.")
        return try JSONValue.object(packet).encoded()
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
