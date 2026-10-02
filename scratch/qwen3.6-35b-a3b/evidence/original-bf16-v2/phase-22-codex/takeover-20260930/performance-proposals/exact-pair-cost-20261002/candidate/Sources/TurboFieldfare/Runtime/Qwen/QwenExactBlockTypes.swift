import Foundation

struct QwenExactBlockRoute: Codable, Sendable, Equatable {
    let position: Int
    let layer: Int
    let ids: [Int]
    let weights: [UInt32]
}

// Owned by one isolated, sequential diagnostic runner invocation. No callbacks.
final class QwenExactBlockCapture: @unchecked Sendable {
    let collectsValues: Bool
    let measuresTimings: Bool
    var timings = QwenExactBlockTimingTotals()
    init(measuresTimings: Bool = true, collectsValues: Bool = true) {
        self.measuresTimings = measuresTimings
        self.collectsValues = collectsValues
    }
    func timingStart() -> UInt64? {
        measuresTimings ? DispatchTime.now().uptimeNanoseconds : nil
    }
    func timingEnd(_ start: UInt64?, site: QwenExactBlockInterval) {
        guard let start else { return }
        timings.add(site, nanoseconds: DispatchTime.now().uptimeNanoseconds - start)
    }
    func revalidateSource(_ model: QwenOfficialSourceModel, site: QwenExactBlockInterval) throws {
        let start = timingStart()
        defer { timingEnd(start, site: site) }
        try model.revalidateSource()
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
                let start = capture.timingStart()
                try lease.validateExactBlockConsumer(expert: item.expertID)
                capture.timingEnd(start, site: .reuseChecks)
                capture.reusedPairChecks += 1
            }
        }
        // Last filesystem observations before encoding/submission. Each token
        // touching this command has its own fresh full-source check.
        for _ in logicalTokens {
            try capture.revalidateSource(model, site: .sourcePairPreGPU)
            capture.preGPUChecks += 1
        }
    }
}

// These intervals overlap. The receipt names each overlap.
enum QwenExactBlockInterval {
    case reuseChecks, sourceSerialStart, sourceSerialEnd, sourceSerialPreMap, sourceSerialPostMap, sourcePairStart, sourcePairToken, sourcePairPreMap, sourcePairPreGPU, sourcePairFinal, sourcePairEnd, sourceTurn, sourceSampler, map, moe, moeSubmit, moeSettle, mixer, router, head, sampler, turn, restoreReplay
}

struct QwenExactBlockWallCounter: Codable, Sendable {
    var count: UInt64 = 0
    var wallNanoseconds: UInt64 = 0
    mutating func add(_ value: UInt64) { count += 1; wallNanoseconds += value }
}

struct QwenExactBlockTimingTotals: Codable, Sendable {
    var reuseChecks = QwenExactBlockWallCounter()
    var sourceSerialStart = QwenExactBlockWallCounter()
    var sourceSerialEnd = QwenExactBlockWallCounter()
    var sourceSerialPreMap = QwenExactBlockWallCounter()
    var sourceSerialPostMap = QwenExactBlockWallCounter()
    var sourcePairStart = QwenExactBlockWallCounter()
    var sourcePairToken = QwenExactBlockWallCounter()
    var sourcePairPreMap = QwenExactBlockWallCounter()
    var sourcePairPreGPU = QwenExactBlockWallCounter()
    var sourcePairFinal = QwenExactBlockWallCounter()
    var sourcePairEnd = QwenExactBlockWallCounter()
    var sourceTurn = QwenExactBlockWallCounter()
    var sourceSampler = QwenExactBlockWallCounter()
    var map = QwenExactBlockWallCounter()
    var moe = QwenExactBlockWallCounter()
    var moeSubmit = QwenExactBlockWallCounter()
    var moeSettle = QwenExactBlockWallCounter()
    var mixer = QwenExactBlockWallCounter()
    var router = QwenExactBlockWallCounter()
    var head = QwenExactBlockWallCounter()
    var sampler = QwenExactBlockWallCounter()
    var turn = QwenExactBlockWallCounter()
    var restoreReplay = QwenExactBlockWallCounter()
    mutating func add(_ site: QwenExactBlockInterval, nanoseconds: UInt64) {
        switch site {
        case .reuseChecks: reuseChecks.add(nanoseconds)
        case .sourceSerialStart: sourceSerialStart.add(nanoseconds)
        case .sourceSerialEnd: sourceSerialEnd.add(nanoseconds)
        case .sourceSerialPreMap: sourceSerialPreMap.add(nanoseconds)
        case .sourceSerialPostMap: sourceSerialPostMap.add(nanoseconds)
        case .sourcePairStart: sourcePairStart.add(nanoseconds)
        case .sourcePairToken: sourcePairToken.add(nanoseconds)
        case .sourcePairPreMap: sourcePairPreMap.add(nanoseconds)
        case .sourcePairPreGPU: sourcePairPreGPU.add(nanoseconds)
        case .sourcePairFinal: sourcePairFinal.add(nanoseconds)
        case .sourcePairEnd: sourcePairEnd.add(nanoseconds)
        case .sourceTurn: sourceTurn.add(nanoseconds)
        case .sourceSampler: sourceSampler.add(nanoseconds)
        case .map: map.add(nanoseconds)
        case .moe: moe.add(nanoseconds)
        case .moeSubmit: moeSubmit.add(nanoseconds)
        case .moeSettle: moeSettle.add(nanoseconds)
        case .mixer: mixer.add(nanoseconds)
        case .router: router.add(nanoseconds)
        case .head: head.add(nanoseconds)
        case .sampler: sampler.add(nanoseconds)
        case .turn: turn.add(nanoseconds)
        case .restoreReplay: restoreReplay.add(nanoseconds)
        }
    }
    mutating func merge(_ other: Self) {
        reuseChecks.count += other.reuseChecks.count; reuseChecks.wallNanoseconds += other.reuseChecks.wallNanoseconds
        sourceSerialStart.count += other.sourceSerialStart.count; sourceSerialStart.wallNanoseconds += other.sourceSerialStart.wallNanoseconds
        sourceSerialEnd.count += other.sourceSerialEnd.count; sourceSerialEnd.wallNanoseconds += other.sourceSerialEnd.wallNanoseconds
        sourceSerialPreMap.count += other.sourceSerialPreMap.count; sourceSerialPreMap.wallNanoseconds += other.sourceSerialPreMap.wallNanoseconds
        sourceSerialPostMap.count += other.sourceSerialPostMap.count; sourceSerialPostMap.wallNanoseconds += other.sourceSerialPostMap.wallNanoseconds
        sourcePairStart.count += other.sourcePairStart.count; sourcePairStart.wallNanoseconds += other.sourcePairStart.wallNanoseconds
        sourcePairToken.count += other.sourcePairToken.count; sourcePairToken.wallNanoseconds += other.sourcePairToken.wallNanoseconds
        sourcePairPreMap.count += other.sourcePairPreMap.count; sourcePairPreMap.wallNanoseconds += other.sourcePairPreMap.wallNanoseconds
        sourcePairPreGPU.count += other.sourcePairPreGPU.count; sourcePairPreGPU.wallNanoseconds += other.sourcePairPreGPU.wallNanoseconds
        sourcePairFinal.count += other.sourcePairFinal.count; sourcePairFinal.wallNanoseconds += other.sourcePairFinal.wallNanoseconds
        sourcePairEnd.count += other.sourcePairEnd.count; sourcePairEnd.wallNanoseconds += other.sourcePairEnd.wallNanoseconds
        sourceTurn.count += other.sourceTurn.count; sourceTurn.wallNanoseconds += other.sourceTurn.wallNanoseconds
        sourceSampler.count += other.sourceSampler.count; sourceSampler.wallNanoseconds += other.sourceSampler.wallNanoseconds
        map.count += other.map.count; map.wallNanoseconds += other.map.wallNanoseconds
        moe.count += other.moe.count; moe.wallNanoseconds += other.moe.wallNanoseconds
        moeSubmit.count += other.moeSubmit.count; moeSubmit.wallNanoseconds += other.moeSubmit.wallNanoseconds
        moeSettle.count += other.moeSettle.count; moeSettle.wallNanoseconds += other.moeSettle.wallNanoseconds
        mixer.count += other.mixer.count; mixer.wallNanoseconds += other.mixer.wallNanoseconds
        router.count += other.router.count; router.wallNanoseconds += other.router.wallNanoseconds
        head.count += other.head.count; head.wallNanoseconds += other.head.wallNanoseconds
        sampler.count += other.sampler.count; sampler.wallNanoseconds += other.sampler.wallNanoseconds
        turn.count += other.turn.count; turn.wallNanoseconds += other.turn.wallNanoseconds
        restoreReplay.count += other.restoreReplay.count; restoreReplay.wallNanoseconds += other.restoreReplay.wallNanoseconds
    }
    var fullSourceWallNanoseconds: UInt64 {
        sourceSerialStart.wallNanoseconds + sourceSerialEnd.wallNanoseconds + sourceSerialPreMap.wallNanoseconds + sourceSerialPostMap.wallNanoseconds + sourcePairStart.wallNanoseconds + sourcePairToken.wallNanoseconds + sourcePairPreMap.wallNanoseconds + sourcePairPreGPU.wallNanoseconds + sourcePairFinal.wallNanoseconds + sourcePairEnd.wallNanoseconds + sourceTurn.wallNanoseconds + sourceSampler.wallNanoseconds
    }
}
