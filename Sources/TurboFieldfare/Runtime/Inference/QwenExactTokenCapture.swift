import Foundation
import OSLog

/// Opt-in local diagnostics only. No observer, sampling, or model operation.
final class QwenExactTokenCapture: @unchecked Sendable {
    private struct Settings: Codable {
        let maxNewTokens: Int
        let temperature: Float
        let topK: Int?
        let topP: Float?
        let repetitionPenalty: Float
        let seed: UInt64?
        let stopStrings: [String]
        let extraStopTokens: [Int32]
        let logitTransform = "raw"

        init(_ config: GenerationConfig) {
            maxNewTokens = config.maxNewTokens
            temperature = config.temperature
            topK = config.topK
            topP = config.topP
            repetitionPenalty = config.repetitionPenalty
            seed = config.seed
            stopStrings = config.stopStrings
            extraStopTokens = config.extraStopTokens.sorted()
        }
    }

    private struct Record: Codable {
        let schemaVersion = 1
        let requestID = UUID().uuidString
        let startedAt = Date().timeIntervalSince1970
        var finishedAt: Double?
        var terminal = "incomplete"
        var complete = false
        var captureIssue: String?
        var operationError: String?
        let maximumContext: Int
        let promptTokenCount: Int
        let checkpointResume: Bool
        let thinking: Bool
        let fixtureSamples: Bool
        let visionStorePresent: Bool
        let preparedPromptPresent: Bool
        let producedVisionFeatureRows: Int?
        let codecEOS: Int32?
        let sourceDescriptorSHA256: String?
        let sourceMarkerSHA256: String?
        let sourceManifestSHA256: String?
        let sourceShardSetSHA256: String?
        var settings: Settings?
        var inputTokenIDs: [Int32]?
        var templateTokenIDs: [Int32]?
        var inputConsumedCount: Int?
        var inputPendingCount: Int?
        var executedPrefillStart: Int?
        var sampledTokenIDs: [Int32] = []
        // Absolute consumed-input count at sampling, distinct from sampler index.
        var samplingInputCounts: [Int] = []
        var selectedSampleCount = 0
        var advancedSampleCount = 0
        var committedGeneratedCount: Int?
        var removedSuffixCount: Int?
        var committedGeneratedTokenIDs: [Int32]?
        var committedConsumedCount: Int?
        var committedPendingCount: Int?
        var stopReason: String?
    }

    private static let logger = Logger(subsystem: "TurboFieldfare", category: "QwenExactTokenCapture")
    private static let inputCap = 65_536
    private static let sampleCap = 4_096
    private static let encodedCap = 2 * 1_024 * 1_024
    private let directory: URL
    private var record: Record
    private var finished = false

    static func make(maximumContext: Int, promptTokenCount: Int,
                     checkpointResume: Bool, thinking: Bool, fixtureSamples: Bool,
                     visionStorePresent: Bool, preparedPromptPresent: Bool,
                     producedVisionFeatureRows: Int?, codecEOS: Int32?,
                     source: LoadedRuntimeSourceIdentity?) -> QwenExactTokenCapture? {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TURBO_QWEN_EXACT_TOKEN_CAPTURE"] == "1" else { return nil }
        guard let path = environment["TURBO_QWEN_EXACT_TOKEN_CAPTURE_DIRECTORY"],
              path.hasPrefix("/"), !path.contains("\0") else {
            logger.error("Exact-token capture unavailable: absolute directory required")
            return nil
        }
        return QwenExactTokenCapture(directory: URL(fileURLWithPath: path, isDirectory: true),
            record: Record(maximumContext: maximumContext, promptTokenCount: promptTokenCount,
                checkpointResume: checkpointResume, thinking: thinking, fixtureSamples: fixtureSamples,
                visionStorePresent: visionStorePresent, preparedPromptPresent: preparedPromptPresent,
                producedVisionFeatureRows: producedVisionFeatureRows, codecEOS: codecEOS,
                sourceDescriptorSHA256: source?.descriptorContentSHA256,
                sourceMarkerSHA256: source?.markerSHA256,
                sourceManifestSHA256: source?.checksumManifestSHA256,
                sourceShardSetSHA256: source?.shardSetSHA256))
    }

    private init(directory: URL, record: Record) {
        self.directory = directory
        self.record = record
    }

    func begin(input: ConversationStateMetrics, config: GenerationConfig,
               differingTemplate: [Int32]?) {
        record.settings = Settings(config)
        record.inputConsumedCount = input.consumedTokenCount
        record.inputPendingCount = input.pendingTokenCount
        guard input.retainedTokenIDs.count <= record.maximumContext,
              input.retainedTokenIDs.count <= Self.inputCap,
              (differingTemplate?.count ?? 0) <= min(Self.inputCap, record.maximumContext) else {
            drop("diagnostic token cap exceeded")
            return
        }
        record.inputTokenIDs = input.retainedTokenIDs
        record.templateTokenIDs = differingTemplate
        record.executedPrefillStart = record.checkpointResume
            ? input.retainedTokenIDs.count : input.retainedTokenIDs.count - record.promptTokenCount
    }

    func sampled(_ token: Int32, inputTokenIDs: [Int32]) {
        record.selectedSampleCount += 1
        guard record.captureIssue == nil else { return }
        let inputCount = inputTokenIDs.count
        guard record.inputTokenIDs != nil,
              record.sampledTokenIDs.count < min(record.settings?.maxNewTokens ?? 0, Self.sampleCap),
              inputCount <= record.maximumContext,
              (!record.sampledTokenIDs.isEmpty || inputTokenIDs == record.inputTokenIDs),
              inputCount == (record.inputTokenIDs?.count ?? 0) + record.advancedSampleCount else {
            drop("sample/input alignment or diagnostic cap failed")
            return
        }
        record.sampledTokenIDs.append(token)
        record.samplingInputCounts.append(inputCount)
    }

    func advanced() { record.advancedSampleCount += 1 }

    func committed(_ ids: [Int32], metrics: ConversationStateMetrics, reason: String) {
        record.terminal = "committed"
        record.stopReason = reason
        record.committedGeneratedCount = ids.count
        record.removedSuffixCount = record.selectedSampleCount - ids.count
        record.committedConsumedCount = metrics.consumedTokenCount
        record.committedPendingCount = metrics.pendingTokenCount
        if record.captureIssue == nil {
            let expected = (record.inputTokenIDs?.count ?? 0) + ids.count
            guard ids.count <= Self.sampleCap,
                  record.advancedSampleCount == record.sampledTokenIDs.count,
                  Array(record.sampledTokenIDs.prefix(ids.count)) == ids,
                  metrics.retainedTokenIDs.count == expected,
                  metrics.consumedTokenCount + metrics.pendingTokenCount == expected else {
                drop("committed token alignment failed")
                finish()
                return
            }
            record.committedGeneratedTokenIDs = ids
            record.complete = true
        }
        finish()
    }

    func failed(_ error: Error) {
        record.terminal = error is CancellationError ? "cancelled" : "failed"
        record.operationError = String(String(describing: error).prefix(512))
        finish()
    }

    func unfinished(cancelled: Bool) {
        guard !finished else { return }
        record.terminal = cancelled ? "cancelled" : "incomplete"
        finish()
    }

    private func drop(_ issue: String) {
        record.captureIssue = issue
        record.complete = false
        record.inputTokenIDs = nil
        record.templateTokenIDs = nil
        record.sampledTokenIDs = []
        record.samplingInputCounts = []
        record.committedGeneratedTokenIDs = nil
    }

    private func finish() {
        guard !finished else { return }
        finished = true
        record.finishedAt = Date().timeIntervalSince1970
        do {
            let encoder = JSONEncoder()
            var data = try encoder.encode(record)
            if data.count > Self.encodedCap {
                drop("encoded diagnostic cap exceeded")
                record.settings = nil
                data = try encoder.encode(record)
            }
            guard data.count <= Self.encodedCap else {
                Self.logger.error("Exact-token capture refused: encoded cap exceeded")
                return
            }
            let file = directory.appendingPathComponent("qwen-exact-\(record.requestID).json")
            // Atomic terminal write. A missing file or incomplete record is not usable.
            try data.write(to: file, options: .atomic)
        } catch {
            // Diagnostics never replace a generation error or throw after commit.
            Self.logger.error("Exact-token capture write failed: \(String(describing: error), privacy: .public)")
        }
    }
}
