import Foundation
import CryptoKit
import TurboFieldfareDecodeProtocol

/// Decides whether an incoming generate may join the open conversation.
///
/// Extracted from the service's command loop so the decision is testable
/// without a socket, a model, or a Metal device. The loop cannot express the
/// boundary cases this holds — a turn from a replaced conversation, a turn that
/// skips or repeats a position, a one-shot arriving while a lineage is open —
/// and every one of them ends with a message the user never sent being appended
/// to a KV, which no later check can detect.
struct DecodeConversationGate: Equatable {
    /// Why a turn cannot run. Typed rather than a string so a test pins the
    /// case, and the loop owns the wording.
    enum Rejection: Error, Equatable {
        /// The turn names a conversation that is no longer the open one.
        case staleConversation(requested: UUID, open: UUID?)
        /// The turn's position does not follow the committed turns.
        case outOfOrderTurn(requested: Int?, committed: Int)
        /// A one-shot generate arrived while a conversation was open. Running
        /// it would reset the KV under a lineage the app still believes in.
        case oneShotDuringConversation(open: UUID)
        case checkpointMismatch
    }

    enum Admission: Equatable {
        /// No conversation is open: reset the KV and prefill the whole prompt.
        case oneShot
        /// Append to the open lineage at `index`.
        case turn(epoch: UUID, index: Int)
    }

    private(set) var openEpoch: UUID?
    private(set) var committedTurns = 0
    private var assessedCheckpoint: (id: UUID, binding: String)?
    private var checkpointReceipts: [UUID: (fingerprint: String, receipt: DecodeContextCheckpointReceipt)] = [:]
    private var awaitingCheckpointResume: UUID?

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.openEpoch == rhs.openEpoch && lhs.committedTurns == rhs.committedTurns
            && lhs.assessedCheckpoint?.id == rhs.assessedCheckpoint?.id
            && lhs.assessedCheckpoint?.binding == rhs.assessedCheckpoint?.binding
            && lhs.awaitingCheckpointResume == rhs.awaitingCheckpointResume
            && lhs.checkpointReceipts.mapValues(\.fingerprint) == rhs.checkpointReceipts.mapValues(\.fingerprint)
    }

    private func fingerprint(_ request: DecodeContextCheckpointRequest, bindingOnly: Bool) throws -> String {
        var value = request
        value.requestID = UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        if bindingOnly { value.record = ""; value.commit = false }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return SHA256.hash(data: try encoder.encode(value)).map { String(format: "%02x", $0) }.joined()
    }

    func previousCheckpoint(_ request: DecodeContextCheckpointRequest) throws -> DecodeContextCheckpointReceipt? {
        guard let previous = checkpointReceipts[request.checkpointID] else { return nil }
        guard previous.fingerprint == (try fingerprint(request, bindingOnly: false)) else {
            throw Rejection.checkpointMismatch
        }
        return previous.receipt
    }

    func validateCheckpoint(_ request: DecodeContextCheckpointRequest) throws {
        guard request.sourceEpoch == openEpoch, request.sourceTurnIndex == committedTurns,
              request.replacementEpoch != openEpoch, awaitingCheckpointResume == nil else {
            throw Rejection.checkpointMismatch
        }
        if request.commit {
            guard assessedCheckpoint?.id == request.checkpointID,
                  assessedCheckpoint?.binding == (try fingerprint(request, bindingOnly: true)) else {
                throw Rejection.checkpointMismatch
            }
        }
    }

    mutating func recordCheckpoint(_ request: DecodeContextCheckpointRequest,
                                   receipt: DecodeContextCheckpointReceipt) throws {
        if request.commit {
            checkpointReceipts[request.checkpointID] = (try fingerprint(request, bindingOnly: false), receipt)
            openEpoch = request.replacementEpoch
            committedTurns = 0
            awaitingCheckpointResume = request.checkpointID
            assessedCheckpoint = nil
        } else if receipt.needed {
            assessedCheckpoint = (request.checkpointID, try fingerprint(request, bindingOnly: true))
        }
    }

    mutating func checkpointResumed() { awaitingCheckpointResume = nil }

    /// Starts a new lineage. The caller drops the KV; this only records that it
    /// did.
    mutating func reset(to epoch: UUID) {
        openEpoch = epoch
        committedTurns = 0
        assessedCheckpoint = nil
        checkpointReceipts.removeAll()
        awaitingCheckpointResume = nil
    }

    /// Ends any lineage. Both unload and load reach here: each releases or
    /// rebuilds the runner and the KV, so the tokens the epoch named are gone,
    /// and a turn resuming onto them would resume onto nothing.
    mutating func endLineage() {
        openEpoch = nil
        committedTurns = 0
        assessedCheckpoint = nil
        checkpointReceipts.removeAll()
        awaitingCheckpointResume = nil
    }

    func admit(_ request: DecodeGenerationRequest) -> Result<Admission, Rejection> {
        if let expected = awaitingCheckpointResume {
            guard case .checkpoint(let id) = request.toolTurn, id == expected else {
                return .failure(.checkpointMismatch)
            }
        } else if case .checkpoint = request.toolTurn {
            return .failure(.checkpointMismatch)
        }
        guard let requested = request.conversationEpoch else {
            if let openEpoch { return .failure(.oneShotDuringConversation(open: openEpoch)) }
            return .success(.oneShot)
        }
        guard requested == openEpoch else {
            return .failure(.staleConversation(requested: requested, open: openEpoch))
        }
        guard request.turnIndex == committedTurns else {
            return .failure(.outOfOrderTurn(requested: request.turnIndex,
                                            committed: committedTurns))
        }
        return .success(.turn(epoch: requested, index: committedTurns))
    }

    /// Records that a turn actually ran. Called after the generation, not at
    /// admission: a turn rejected downstream, or one that failed before
    /// touching the KV, must not advance the position the next turn has to
    /// match.
    mutating func commit(_ admission: Admission) {
        guard case .turn(let epoch, _) = admission, epoch == openEpoch else { return }
        committedTurns += 1
    }
}
extension DecodeConversationGate.Rejection {
    /// The wording the service sends back. Names what was asked for and what is
    /// actually open, because "rejected" alone is not a diagnosis.
    var message: String {
        switch self {
        case .checkpointMismatch:
            return "checkpoint identity, source, settled result, or resume does not match the pending transaction"
        case .staleConversation(let requested, let open):
            let openText = open.map(\.uuidString) ?? "none"
            return "turn belongs to conversation \(requested.uuidString), "
                + "and the open conversation is \(openText)"
        case .outOfOrderTurn(let requested, let committed):
            let requestedText = requested.map(String.init) ?? "unset"
            return "turn index \(requestedText) does not follow the "
                + "\(committed) turns committed to this conversation"
        case .oneShotDuringConversation(let open):
            return "a conversation is open (\(open.uuidString)); "
                + "a one-shot turn cannot run on it"
        }
    }
}
