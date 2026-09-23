import Foundation
import Metal
import TurboFieldfareFormat

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh"
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }
}

/// Runtime configuration for Qwen 3.6. Fields retain their native Qwen
/// meaning instead of being forced into Gemma's `ArchConfig` vocabulary.
public struct QwenArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let fullAttentionLayerMask: [UInt8]
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDimension: Int
    public let attentionOutputGate: Bool
    public let linearConvolutionKernel: Int
    public let linearKeyHeads: Int
    public let linearKeyHeadDimension: Int
    public let linearValueHeads: Int
    public let linearValueHeadDimension: Int
    public let recurrentStateIsFP32: Bool
    public let partialRotaryFactor: Double
    public let ropeTheta: Double
    public let mropeSections: [Int]
    public let numberOfExperts: Int
    public let expertsPerToken: Int
    public let routedExpertIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let vocabularySize: Int
    public let tiedWordEmbeddings: Bool
    public let hiddenActivation: String
    public let bosTokenID: Int
    public let eosTokenID: Int
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int

    init(wire: GTurboQwenArchitectureV2) {
        hiddenSize = wire.hiddenSize
        numLayers = wire.numLayers
        fullAttentionLayerMask = wire.layerTypes.map { $0 == .fullAttention ? 1 : 0 }
        numAttentionHeads = wire.numAttentionHeads
        numKeyValueHeads = wire.numKeyValueHeads
        headDimension = wire.headDimension
        attentionOutputGate = wire.attentionOutputGate
        linearConvolutionKernel = wire.linearConvolutionKernel
        linearKeyHeads = wire.linearKeyHeads
        linearKeyHeadDimension = wire.linearKeyHeadDimension
        linearValueHeads = wire.linearValueHeads
        linearValueHeadDimension = wire.linearValueHeadDimension
        recurrentStateIsFP32 = wire.recurrentStateType == .fp32
        partialRotaryFactor = wire.partialRotaryFactor
        ropeTheta = wire.ropeTheta
        mropeSections = wire.mropeSections
        numberOfExperts = wire.numberOfExperts
        expertsPerToken = wire.expertsPerToken
        routedExpertIntermediateSize = wire.routedExpertIntermediateSize
        sharedExpertIntermediateSize = wire.sharedExpertIntermediateSize
        vocabularySize = wire.vocabularySize
        tiedWordEmbeddings = wire.tiedWordEmbeddings
        hiddenActivation = wire.hiddenActivation
        bosTokenID = wire.bosTokenID
        eosTokenID = wire.eosTokenID
        imageTokenID = wire.imageTokenID
        videoTokenID = wire.videoTokenID
        visionStartTokenID = wire.visionStartTokenID
        visionEndTokenID = wire.visionEndTokenID
    }
}

public enum LoadedModelArchitecture: Sendable, Equatable {
    case gemma4(ArchConfig)
    case qwen3_6(QwenArchConfig)
}

public enum LoadedTensorStorage: String, Sendable, Equatable {
    case bf16
    case fp32
    case affineInt4
    case affineInt8
}

public struct LoadedTensorRegion: Sendable, Equatable {
    public let name: String
    public let file: String
    public let offset: UInt64
    public let size: UInt64
    public let shape: [UInt64]
    public let storage: LoadedTensorStorage
    public let quantizationCategory: String?

    init(wire: GTurboTensorRegionV2) {
        name = wire.name
        file = wire.file
        offset = wire.offset
        size = wire.size
        shape = wire.shape
        storage = switch wire.storage {
        case .bf16: .bf16
        case .fp32: .fp32
        case .affineInt4: .affineInt4
        case .affineInt8: .affineInt8
        }
        quantizationCategory = wire.quantizationCategory?.rawValue
    }
}

/// A format-verified v2 manifest ready for a later family runtime factory.
public struct LoadedModelManifest: Sendable, Equatable {
    public let descriptor: InstalledModelDescriptor
    public let architecture: LoadedModelArchitecture
    public let files: [String: ManifestFileEntry]
    public let tensorRegions: [LoadedTensorRegion]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64

    init(descriptor: InstalledModelDescriptor,
         architecture: LoadedModelArchitecture,
         files: [String: ManifestFileEntry],
         tensorRegions: [LoadedTensorRegion],
         expertsPerLayer: Int, numLayers: Int, expertStride: UInt64) {
        self.descriptor = descriptor
        self.architecture = architecture
        self.files = files
        self.tensorRegions = tensorRegions
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
    }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
