import Foundation

struct QwenExactBlockRoute: Codable, Sendable, Equatable {
    let position: Int
    let layer: Int
    let ids: [Int]
    let weights: [UInt32]
}

// Owned by one isolated, sequential diagnostic runner invocation. No callbacks.
final class QwenExactBlockCapture: @unchecked Sendable {
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
