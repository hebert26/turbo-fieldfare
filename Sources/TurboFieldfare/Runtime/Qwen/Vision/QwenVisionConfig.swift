import Foundation
import Metal
import TurboFieldfareFormat

/// Closed Qwen 3.6 vision architecture accepted by the runtime.
///
/// This deliberately duplicates the repacker's official architecture because
/// the runtime target cannot depend on `TurboFieldfareRepackCore`. Exact tests
/// on both sides detect ordinary drift, but cannot detect coordinated changes.
struct QwenVisionConfig: Equatable, Sendable {
    struct TensorContract: Equatable, Sendable {
        let name: String
        let shape: [UInt64]
        let storage: GTurboStorageTypeV2
    }

    static let official = QwenVisionConfig()

    let depth: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let numHeads: Int
    let numPositionEmbeddings: Int
    let inputChannels: Int
    let outputHiddenSize: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let spatialMergeSize: Int
    let hiddenActivation: String
    let minimumProcessedPixels: Int
    let publisherMaximumProcessedPixels: Int
    let maximumPatchRows: Int
    let maximumMergedRows: Int
    /// Narrow P3 test seam; verified production stores require official 2x2 geometry.
    let allowsFixtureGeometry: Bool

    init(
        depth: Int = 27,
        hiddenSize: Int = 1_152,
        intermediateSize: Int = 4_304,
        numHeads: Int = 16,
        numPositionEmbeddings: Int = 2_304,
        inputChannels: Int = 3,
        outputHiddenSize: Int = 2_048,
        patchSize: Int = 16,
        temporalPatchSize: Int = 2,
        spatialMergeSize: Int = 2,
        hiddenActivation: String = "gelu_pytorch_tanh",
        minimumProcessedPixels: Int = 65_536,
        publisherMaximumProcessedPixels: Int = 16_777_216,
        maximumPatchRows: Int = 2_520,
        maximumMergedRows: Int = 630,
        allowsFixtureGeometry: Bool = false
    ) {
        self.depth = depth
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHeads = numHeads
        self.numPositionEmbeddings = numPositionEmbeddings
        self.inputChannels = inputChannels
        self.outputHiddenSize = outputHiddenSize
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.spatialMergeSize = spatialMergeSize
        self.hiddenActivation = hiddenActivation
        self.minimumProcessedPixels = minimumProcessedPixels
        self.publisherMaximumProcessedPixels = publisherMaximumProcessedPixels
        self.maximumPatchRows = maximumPatchRows
        self.maximumMergedRows = maximumMergedRows
        self.allowsFixtureGeometry = allowsFixtureGeometry
    }

    var patchWidth: Int { inputChannels * temporalPatchSize * patchSize * patchSize }
    var mergedHiddenSize: Int { hiddenSize * spatialMergeSize * spatialMergeSize }
    var headDimension: Int { hiddenSize / numHeads }
    var resizeFactor: Int { patchSize * spatialMergeSize }

    /// Processing bounds may be reduced, but a verified production store is
    /// executable only with the exact architecture that defines its offsets.
    var matchesOfficialWeightLayout: Bool {
        tensorContract == Self.official.tensorContract
            && numHeads == Self.official.numHeads
            && hiddenActivation == Self.official.hiddenActivation
            && !allowsFixtureGeometry
    }

    func validate() throws {
        guard depth > 0, hiddenSize > 0, intermediateSize > 0,
              numHeads > 0, hiddenSize.isMultiple(of: numHeads),
              numPositionEmbeddings > 0, inputChannels == 3,
              outputHiddenSize > 0, patchSize > 0,
              temporalPatchSize > 0, spatialMergeSize > 0,
              (allowsFixtureGeometry
                || (temporalPatchSize == 2 && spatialMergeSize == 2)),
              hiddenActivation == "gelu_pytorch_tanh",
              minimumProcessedPixels > 0,
              publisherMaximumProcessedPixels >= minimumProcessedPixels,
              maximumPatchRows > 0,
              maximumMergedRows == maximumPatchRows
                / (spatialMergeSize * spatialMergeSize) else {
            throw QwenVisionError.invalidConfiguration
        }
    }

    var tensorContract: [TensorContract] {
        var values: [TensorContract] = [
            .init(name: "model.visual.patch_embed.proj.weight",
                  shape: u64([hiddenSize, inputChannels, temporalPatchSize, patchSize, patchSize]),
                  storage: .bf16),
            .init(name: "model.visual.patch_embed.proj.bias",
                  shape: u64([hiddenSize]), storage: .bf16),
            .init(name: "model.visual.pos_embed.weight",
                  shape: u64([numPositionEmbeddings, hiddenSize]), storage: .bf16),
        ]
        let members: [(String, [Int])] = [
            ("attn.proj.weight", [hiddenSize, hiddenSize]),
            ("attn.proj.bias", [hiddenSize]),
            ("attn.qkv.weight", [3 * hiddenSize, hiddenSize]),
            ("attn.qkv.bias", [3 * hiddenSize]),
            ("mlp.linear_fc1.weight", [intermediateSize, hiddenSize]),
            ("mlp.linear_fc1.bias", [intermediateSize]),
            ("mlp.linear_fc2.weight", [hiddenSize, intermediateSize]),
            ("mlp.linear_fc2.bias", [hiddenSize]),
            ("norm1.weight", [hiddenSize]),
            ("norm1.bias", [hiddenSize]),
            ("norm2.weight", [hiddenSize]),
            ("norm2.bias", [hiddenSize]),
        ]
        for layer in 0..<depth {
            for (member, shape) in members {
                values.append(.init(
                    name: "model.visual.blocks.\(layer).\(member)",
                    shape: u64(shape), storage: .bf16))
            }
        }
        values.append(contentsOf: [
            .init(name: "model.visual.merger.norm.weight",
                  shape: u64([hiddenSize]), storage: .bf16),
            .init(name: "model.visual.merger.norm.bias",
                  shape: u64([hiddenSize]), storage: .bf16),
            .init(name: "model.visual.merger.linear_fc1.weight",
                  shape: u64([mergedHiddenSize, mergedHiddenSize]), storage: .bf16),
            .init(name: "model.visual.merger.linear_fc1.bias",
                  shape: u64([mergedHiddenSize]), storage: .bf16),
            .init(name: "model.visual.merger.linear_fc2.weight",
                  shape: u64([outputHiddenSize, mergedHiddenSize]), storage: .bf16),
            .init(name: "model.visual.merger.linear_fc2.bias",
                  shape: u64([outputHiddenSize]), storage: .bf16),
        ])
        return values
    }

    var rawTensorElementCount: UInt64 {
        tensorContract.reduce(0) { partial, tensor in
            partial + tensor.shape.reduce(UInt64(1), *)
        }
    }

    private func u64(_ values: [Int]) -> [UInt64] { values.map(UInt64.init) }
}

enum QwenVisionMedia: Sendable, Equatable {
    case stillImage
    case video
}

enum QwenVisionError: Error, Equatable, Sendable {
    case invalidConfiguration
    case unsupportedVideo
    case invalidDimensions(width: Int, height: Int)
    case arithmeticOverflow(String)
    case patchRowQuotaExceeded(requested: Int, maximum: Int)
    case mergedRowQuotaExceeded(requested: Int, maximum: Int)
    case liveOwnerRowQuotaExceeded(requested: Int, maximum: Int)
    case mutableByteQuotaExceeded(requested: Int, maximum: Int)
    case bufferExceedsDevice(name: String, requested: Int, maximum: Int)
    case allocationFailed(name: String)
    case invalidFeatureShape
    case invalidPositions
    case commandBufferUnavailable
    case commandFailed(String)
}

struct QwenVisionResourceLimits: Equatable, Sendable {
    static let provisional = QwenVisionResourceLimits()

    let maximumVisibleHistoryRows: Int
    let maximumLiveOwnerRows: Int
    let maximumMutablePreparedBytes: Int

    init(
        maximumVisibleHistoryRows: Int = 630,
        maximumLiveOwnerRows: Int = 1_260,
        maximumMutablePreparedBytes: Int = 200_000_000
    ) {
        self.maximumVisibleHistoryRows = maximumVisibleHistoryRows
        self.maximumLiveOwnerRows = maximumLiveOwnerRows
        self.maximumMutablePreparedBytes = maximumMutablePreparedBytes
    }
}

/// Requested-length accounting. It intentionally does not claim to measure
/// allocator metadata, driver overhead, or process footprint.
struct QwenVisionAllocationPlan: Equatable, Sendable {
    struct Entry: Equatable, Sendable {
        let name: String
        let bytes: Int
    }

    let entries: [Entry]
    let totalBytes: Int

    init(entries: [Entry], limits: QwenVisionResourceLimits = .provisional) throws {
        var total = 0
        for entry in entries {
            guard entry.bytes >= 0 else {
                throw QwenVisionError.arithmeticOverflow(entry.name)
            }
            let (next, overflow) = total.addingReportingOverflow(entry.bytes)
            guard !overflow else { throw QwenVisionError.arithmeticOverflow(entry.name) }
            total = next
        }
        guard total <= limits.maximumMutablePreparedBytes else {
            throw QwenVisionError.mutableByteQuotaExceeded(
                requested: total, maximum: limits.maximumMutablePreparedBytes)
        }
        self.entries = entries
        self.totalBytes = total
    }

    func validate(device: MTLDevice) throws {
        for entry in entries where entry.bytes > device.maxBufferLength {
            throw QwenVisionError.bufferExceedsDevice(
                name: entry.name, requested: entry.bytes,
                maximum: device.maxBufferLength)
        }
    }
}

private extension Int {
    func qwenCheckedMultiply(_ other: Int, operation: String) throws -> Int {
        let (result, overflow) = multipliedReportingOverflow(by: other)
        guard !overflow else { throw QwenVisionError.arithmeticOverflow(operation) }
        return result
    }
}
