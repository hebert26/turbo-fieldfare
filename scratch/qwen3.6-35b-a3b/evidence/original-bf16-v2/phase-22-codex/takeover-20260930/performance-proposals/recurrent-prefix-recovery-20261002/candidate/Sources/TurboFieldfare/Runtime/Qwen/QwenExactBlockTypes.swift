import Foundation

struct QwenExactBlockRoute: Codable, Sendable, Equatable {
    let position: Int
    let layer: Int
    let ids: [Int]
    let weights: [UInt32]
}

// Owned by one isolated, sequential diagnostic runner invocation. No callbacks.
final class QwenExactBlockCapture: @unchecked Sendable {
    let wantsRecurrentRecovery: Bool
    private var recurrentInputs: QwenRecurrentPrefixInputs?
    private var recurrentSealed = false
    private var recurrentConsumed = false
    init(enableRecurrentRecovery: Bool = false) {
        wantsRecurrentRecovery = enableRecurrentRecovery
    }
    var savedInputBytes: Int { recurrentInputs?.byteCount ?? 0 }

    func bindRecurrent(owner: UUID, turn: UUID, base: Int, width: Int,
                       layers: [Int], fullBaseline: QwenFullAttentionKVSnapshot) throws {
        guard wantsRecurrentRecovery, recurrentInputs == nil, !recurrentConsumed,
              width == 2048, layers.count == 30, Set(layers).count == 30 else {
            throw QwenTextRunnerError.invalidTransaction
        }
        recurrentInputs = QwenRecurrentPrefixInputs(owner: owner, turn: turn,
            base: base, width: width, layers: layers.sorted(), fullBaseline: fullBaseline)
    }
    func saveRecurrentRows(layer: Int, start: Int, rows: [Float]) throws {
        guard var inputs = recurrentInputs, !recurrentSealed, !recurrentConsumed,
              inputs.layers.contains(layer), (0...4).contains(start),
              rows.count > 0, rows.count <= 4 * inputs.width, rows.count.isMultiple(of: inputs.width),
              rows.allSatisfy(\.isFinite) else { throw QwenTextRunnerError.invalidTransaction }
        let existing = inputs.rows[layer] ?? []
        guard existing.count == start * inputs.width,
              existing.count + rows.count <= 4 * inputs.width else {
            throw QwenTextRunnerError.invalidTransaction
        }
        inputs.rows[layer] = existing + rows
        recurrentInputs = inputs
    }
    func sealRecurrent() throws {
        guard let inputs = recurrentInputs, !recurrentSealed, !recurrentConsumed,
              Set(inputs.rows.keys) == Set(inputs.layers),
              inputs.rows.values.allSatisfy({ $0.count == 4 * inputs.width }) else {
            throw QwenTextRunnerError.invalidTransaction
        }
        recurrentSealed = true
    }
    func takeRecurrent(owner: UUID, turn: UUID, base: Int,
                       fullBaseline: QwenFullAttentionKVSnapshot) throws -> QwenRecurrentPrefixInputs {
        guard let inputs = recurrentInputs, recurrentSealed, !recurrentConsumed,
              inputs.owner == owner, inputs.turn == turn, inputs.base == base,
              inputs.fullBaseline == fullBaseline else { throw QwenTextRunnerError.invalidTransaction }
        recurrentConsumed = true
        recurrentInputs = nil
        return inputs
    }
    func discardRecurrent() {
        recurrentConsumed = true
        recurrentInputs = nil
        recurrentSealed = false
    }
    var logits: [[Float]] = []
    var routes: [QwenExactBlockRoute] = []
    var tokenAdmissions = 0
    var layerAdmissions = 0
    var reusedPairChecks = 0
    var preGPUChecks = 0
    var pairMaps = 0
    var pairMappedExperts = 0
    var pairMoECommands = 0
    var separateSharedCommands = 0
    var sharedChecks = 0
    var finalChecks = 0
}

struct QwenExactBlockConsumerAdmission: Sendable {
    let model: QwenOfficialSourceModel
    let lease: QwenBF16ExpertLease
    let capture: QwenExactBlockCapture

    func validate(submittedLease: QwenBF16ExpertLease, submittedWork: [QwenBF16GroupedExpertWork]) throws {
        guard submittedLease === lease, !submittedWork.isEmpty else {
            throw QwenExpertMappingError.canceled
        }
        let logicalTokens = Set(submittedWork.map(\.tokenIndex)).sorted()
        // The first use of each mapped expert already received cache.load's
        // miss read/publication or hit checks. Every additional logical use
        // gets the identical hit-range checks, independently, without publish.
        var firstUses = Set<Int>()
        for item in submittedWork {
            if !firstUses.insert(item.expertID).inserted {
                try lease.validateExactBlockConsumer(expert: item.expertID)
                capture.reusedPairChecks += 1
            }
        }
        // Last filesystem observations before encoding/submission. Each token
        // touching this command has its own fresh full-source check.
        for _ in logicalTokens {
            try model.revalidateSource()
            capture.preGPUChecks += 1
        }
    }
}

// One actor-owned block. Arrays retain immutable normalized rows, not scratch aliases.
struct QwenRecurrentPrefixInputs: Sendable {
    let owner: UUID
    let turn: UUID
    let base: Int
    let width: Int
    let layers: [Int]
    let fullBaseline: QwenFullAttentionKVSnapshot
    var rows: [Int: [Float]] = [:]
    var byteCount: Int { rows.values.reduce(0) { $0 + $1.count * MemoryLayout<Float>.stride } }
}
struct QwenRecurrentPrefixSettlement: Sendable {
    let keptInputs: Int
    let appendCalls: Int
    let savedInputBytes: Int
}
