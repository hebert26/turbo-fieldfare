import CryptoKit
import Foundation
import Metal
import TurboFieldfareFormat
import TurboFieldfareOfficialQwenSource

/// Source identity is not an installed packed descriptor or a payload seal.
public struct LoadedRuntimeSourceIdentity: Sendable, Equatable {
    public let descriptorContentSHA256: String
    public let markerSHA256: String
    public let sourceRoot: String
    public let checksumManifestSHA256: String
    public let shardSetSHA256: String

    init(receipt: OfficialSourceTrustReceipt) {
        descriptorContentSHA256 = receipt.descriptorContentSHA256
        markerSHA256 = receipt.markerSHA256
        sourceRoot = receipt.sourceRoot
        checksumManifestSHA256 = receipt.checksumManifestSHA256
        shardSetSHA256 = receipt.shardSetSHA256
    }
}

/// Independently admitted original BF16 source. Retains the protected handle,
/// one BF16 copy of each required dense matrix, and only short FP32 norm/state
/// vectors. Never constructs a packed descriptor, packed expert layout, or a
/// persistent FP32 copy of embeddings, the vocabulary head, or projections.
public final class QwenOfficialSourceModel: @unchecked Sendable {
    struct Layer {
        let weights: QwenBF16Weights
        let inputNorm: [Float]
        let postNorm: [Float]
        let routerName: String
        let sharedNames: QwenBF16SharedNames
        let routedNames: QwenBF16RoutedSourceNames
        let fullNames: QwenBF16FullAttentionNames?
        let queryNorm: [Float]?
        let keyNorm: [Float]?
        let linearNames: QwenBF16LinearNames?
        let linearVectors: QwenBF16LinearVectors?
    }

    let architecture: QwenTextArchitecture
    let visionArchitecture: QwenArchConfig
    /// Nil only for the internal, non-runtime synthetic fixture seam.
    let sourceIdentity: LoadedRuntimeSourceIdentity?
    let expertCacheSlots: Int
    let expertCachePolicy: ExpertCachePolicy
    let expertCacheBytesPerLayer: UInt64
    let expertCacheReservationBytes: UInt64
    let residentWeightBytes: UInt64
    private let expertCacheQuota: ExpertCacheQuota
    let sourceIntegrityPolicy: ModelIntegrityPolicy?
    let source: OfficialSourceHandle
    private let trustReceipt: OfficialSourceTrustReceipt?
    let context: MetalContext
    let sourceContentSHA256: String
    // Processor metadata is required only when admitting an image companion.
    private let sourceIndex: [String: String]
    let entryWeights: QwenBF16Weights
    let layers: [Layer]
    let finalNorm: [Float]
    private static let embeddingTensorName = "model.language_model.embed_tokens.weight"
    private static let headTensorName = "lm_head.weight"
    var embeddingName: String { Self.embeddingTensorName }
    var headName: String { Self.headTensorName }

    /// Shared by every runner made from this model. Reservations count their
    /// complete possible cache allocation, including layers not yet visited.
    fileprivate final class ExpertCacheQuota: @unchecked Sendable {
        private let lock = NSLock()
        private let capacity: UInt64
        private var reserved: UInt64 = 0

        init(capacity: UInt64) { self.capacity = capacity }

        func reserve(bytes: UInt64) throws -> ExpertCacheReservation {
            lock.lock()
            defer { lock.unlock() }
            guard bytes > 0, bytes <= capacity - reserved else {
                throw QwenBF16ExpertCacheError.budgetExceeded
            }
            reserved += bytes
            return ExpertCacheReservation(quota: self, bytes: bytes)
        }

        func release(bytes: UInt64) {
            lock.lock()
            defer { lock.unlock() }
            precondition(bytes <= reserved)
            reserved -= bytes
        }
    }

    final class ExpertCacheReservation: @unchecked Sendable {
        let bytes: UInt64
        private let quota: ExpertCacheQuota
        private let lock = NSLock()
        private var released = false

        fileprivate init(quota: ExpertCacheQuota, bytes: UInt64) {
            self.quota = quota
            self.bytes = bytes
        }

        /// Coordinators retain this reservation independently of their runner.
        /// Only the final owner may return its aggregate quota through ARC.
        private func release() {
            lock.lock()
            let shouldRelease = !released
            released = true
            lock.unlock()
            if shouldRelease { quota.release(bytes: bytes) }
        }

        deinit { release() }
    }

    func reserveExpertCache() throws -> ExpertCacheReservation {
        try expertCacheQuota.reserve(bytes: expertCacheReservationBytes)
    }

    /// Production admission: verification runs before any weight read or GPU
    /// allocation. Trusted receipt policy is explicit and still checks pinned
    /// fingerprints; full SHA verifies all actual source payload bytes.
    static func load(registrationURL: URL, context: MetalContext,
                     integrityPolicy: ModelIntegrityPolicy,
                     expertCacheSlots: Int,
                     expertCachePolicy: ExpertCachePolicy,
                     residencyBudgetBytes: UInt64) throws -> QwenOfficialSourceModel {
        let receipt = try integrityPolicy.verifyOfficialSource(at: registrationURL)
        let source = try OfficialSourceHandle(registrationURL: registrationURL)
        try source.validateTrustedReceipt(receipt)
        guard receipt.logicalModelPath == registrationURL.path,
              receipt.sourceRoot == source.sourceRootURL.path,
              receipt.checksumManifestSHA256 == OfficialQwenPayloadVerifier.checksumManifestSHA256 else {
            throw ModelError.indexCorrupt(detail: "official source trust binding changed")
        }
        let directory = try GTurboModelDirectory(protectedOfficialSource: source)
        let config = try Self.metadata("config.json", from: directory, requirePin: true)
        let index = try Self.metadata("model.safetensors.index.json", from: directory, requirePin: true)
        let geometry = try QwenArchConfig(officialConfigJSON: config)
        let architecture = QwenTextArchitecture(official: geometry)
        try architecture.validate()
        let mapping = try Self.decodeIndex(index, allowed: Set(OfficialQwenSourceIdentity.pinned.shards.map(\.filename)))
        let model = try QwenOfficialSourceModel(
            source: source, architecture: architecture, visionArchitecture: geometry,
            sourceIdentity: LoadedRuntimeSourceIdentity(receipt: receipt),
            trustReceipt: receipt, mapping: mapping,
            sourceContentSHA256: receipt.descriptorContentSHA256,
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            sourceIntegrityPolicy: integrityPolicy,
            context: context,
            residencyBudgetBytes: residencyBudgetBytes)
        try model.revalidateSource()
        return model
    }

    /// Internal synthetic source seam: a caller can supply only real admitted
    /// shard headers and bytes through OfficialSourceHandle, never offsets or
    /// fabricated TensorRanges. This path grants no public runtime admission.
    static func loadSyntheticFixture(registrationURL: URL, context: MetalContext,
                                     residencyBudgetBytes: UInt64,
                                     expertCacheSlots: Int = 8,
                                     expertCachePolicy: ExpertCachePolicy = .lfu) throws -> QwenOfficialSourceModel {
        let source = try OfficialSourceHandle(registrationURL: registrationURL)
        let directory = try GTurboModelDirectory(protectedOfficialSource: source)
        let config = try Self.metadata("config.json", from: directory, requirePin: false)
        let index = try Self.metadata("model.safetensors.index.json", from: directory,
                                      requirePin: false)
        let geometry = try QwenArchConfig(officialConfigJSON: config)
        let architecture = QwenTextArchitecture(official: geometry)
        try architecture.validate()
        let mapping = try Self.decodeIndex(index, allowed: Set(
            try source.basenames().filter { $0.hasSuffix(".safetensors") }))
        let marker = try source.registeredDescriptor()
        let model = try QwenOfficialSourceModel(
            source: source, architecture: architecture, visionArchitecture: geometry,
            sourceIdentity: nil,
            trustReceipt: nil, mapping: mapping,
            sourceContentSHA256: marker.contentSHA256,
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy,
            sourceIntegrityPolicy: nil,
            context: context,
            residencyBudgetBytes: residencyBudgetBytes)
        try source.validateAcceptedFiles()
        return model
    }

    private init(source: OfficialSourceHandle, architecture: QwenTextArchitecture,
                 visionArchitecture: QwenArchConfig,
                 sourceIdentity: LoadedRuntimeSourceIdentity?,
                 trustReceipt: OfficialSourceTrustReceipt?, mapping: [String: String],
                 sourceContentSHA256: String,
                 expertCacheSlots: Int,
                 expertCachePolicy: ExpertCachePolicy,
                 sourceIntegrityPolicy: ModelIntegrityPolicy?,
                 context: MetalContext, residencyBudgetBytes: UInt64) throws {
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) { $0 + Int($1) }
        let expectedSchedule = sourceIdentity == nil
            ? (layers: 2, full: 1) : (layers: 40, full: 10)
        guard residencyBudgetBytes > 0,
              expertCacheSlots >= architecture.expertsPerToken,
              expertCacheSlots <= architecture.experts,
              architecture.layers == expectedSchedule.layers,
              fullLayers == expectedSchedule.full else {
            throw ModelError.indexCorrupt(detail: "source residency or layer schedule invalid")
        }
        let hidden = architecture.hiddenSize
        let vocab = architecture.vocabularySize
        func cacheProduct(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
            let (bytes, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else { throw QwenBF16ExpertCacheError.invalidGeometry }
            return bytes
        }
        guard let cacheHidden = UInt64(exactly: hidden), cacheHidden > 0,
              let cacheIntermediate = UInt64(exactly: architecture.routedIntermediateSize),
              cacheIntermediate > 0,
              let cacheSlots = UInt64(exactly: expertCacheSlots), cacheSlots > 0,
              let cacheLayers = UInt64(exactly: architecture.layers), cacheLayers > 0 else {
            throw QwenBF16ExpertCacheError.invalidGeometry
        }
        let pairBytes = try cacheProduct(try cacheProduct(cacheHidden, cacheIntermediate), 6)
        let cacheBytesPerLayer = try cacheProduct(pairBytes, cacheSlots)
        let cacheReservationBytes = try cacheProduct(cacheBytesPerLayer, cacheLayers)
        guard cacheReservationBytes <= residencyBudgetBytes else {
            throw QwenBF16WeightError.budgetExceeded
        }
        // Admit enough remaining room for a complete runner before any
        // resident payload read or weight/cache GPU allocation.
        let residentWeightBudget = residencyBudgetBytes - cacheReservationBytes
        var consumed: UInt64 = 0
        func spec(_ name: String, _ role: QwenBF16TensorSpec.Role,
                  _ rows: Int, _ columns: Int) throws -> QwenBF16TensorSpec {
            guard let shard = mapping[name], rows > 0, columns > 0 else {
                throw ModelError.tensorNotFound(name: name)
            }
            let byteCount = UInt64(rows).multipliedReportingOverflow(by: UInt64(columns))
            guard !byteCount.overflow,
                  byteCount.partialValue <= (UInt64.max - consumed) / 2 else {
                throw ModelError.indexCorrupt(detail: "BF16 resident byte count overflow")
            }
            consumed += byteCount.partialValue * 2
            guard consumed <= residentWeightBudget else { throw QwenBF16WeightError.budgetExceeded }
            return QwenBF16TensorSpec(name: name, shardName: shard,
                                      role: role, rows: rows, columns: columns)
        }
        func vector(_ name: String, _ shape: [UInt64]) throws -> [Float] {
            guard let shard = mapping[name] else { throw ModelError.tensorNotFound(name: name) }
            let token = try source.admitTensor(shardName: shard, tensorName: name)
            let region = LoadedTensorRegion(admittedSourceTensor: token)
            var elements: UInt64 = 1
            for dimension in shape {
                let (next, overflow) = elements.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !overflow else {
                    throw ModelError.indexCorrupt(detail: "invalid source vector geometry: \(name)")
                }
                elements = next
            }
            guard region.shape == shape, region.storage == .bf16,
                  elements <= OfficialSourceHandle.maximumTensorReadBytes / 2,
                  region.size == elements * 2,
                  consumed <= residentWeightBudget,
                  elements <= (residentWeightBudget - consumed) / 4 else {
                throw ModelError.indexCorrupt(detail: "invalid source vector or budget: \(name)")
            }
            // Direct protected read into the bounded BF16 vector destination.
            // No Data payload staging or persistent FP32 matrix is created.
            var words = [UInt16](repeating: 0, count: Int(elements))
            try words.withUnsafeMutableBufferPointer { storage in
                guard let base = storage.baseAddress else {
                    throw ModelError.indexCorrupt(detail: "empty source vector: \(name)")
                }
                try source.preadTensorRange(
                    token, byteOffset: 0, byteCount: region.size,
                    expectedByteCount: region.size,
                    into: UnsafeMutableRawBufferPointer(
                        start: base, count: Int(region.size)))
            }
            let result = words.map {
                Float(bitPattern: UInt32(UInt16(littleEndian: $0)) << 16)
            }
            guard result.allSatisfy(\.isFinite) else {
                throw ModelError.indexCorrupt(detail: "nonfinite source vector: \(name)")
            }
            consumed += elements * 4
            return result
        }
        // Admit every routed header before reading resident payloads or
        // allocating any GPU weight/cache buffers. The index supplies only a
        // shard name: the protected handle supplies the actual BF16 range.
        guard let expertCount = UInt64(exactly: architecture.experts),
              let intermediate = UInt64(exactly: architecture.routedIntermediateSize),
              let hiddenWidth = UInt64(exactly: hidden),
              expertCount > 0, intermediate > 0, hiddenWidth > 0,
              intermediate <= UInt64.max / 2 else {
            throw QwenBF16ExpertCacheError.invalidGeometry
        }
        let doubledIntermediate = intermediate * 2
        func admitRouted(_ shard: String, _ name: String,
                         shape: [UInt64]) throws {
            var elements: UInt64 = 1
            for dimension in shape {
                let product = elements.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !product.overflow else {
                    throw QwenBF16ExpertCacheError.invalidGeometry
                }
                elements = product.partialValue
            }
            let bytes = elements.multipliedReportingOverflow(by: 2)
            guard !bytes.overflow, bytes.partialValue <= UInt64(Int64.max) else {
                throw QwenBF16ExpertCacheError.invalidGeometry
            }
            let token = try source.admitTensor(shardName: shard, tensorName: name)
            let region = LoadedTensorRegion(admittedSourceTensor: token)
            guard region.name == name, region.file == shard,
                  region.storage == .bf16, region.shape == shape,
                  region.size == bytes.partialValue else {
                throw QwenBF16ExpertCacheError.invalidHeader(name)
            }
            // A zero-byte check at the admitted range end validates the
            // retained FD, literal shard name and file/off_t bounds without
            // reading an expert payload or reserving a cache slot.
            try source.preadTensorRange(
                token, byteOffset: region.size, byteCount: 0,
                expectedByteCount: 0,
                into: UnsafeMutableRawBufferPointer(start: nil, count: 0))
        }
        var routedByLayer: [QwenBF16RoutedSourceNames] = []
        routedByLayer.reserveCapacity(architecture.layers)
        for layer in 0..<architecture.layers {
            try Task.checkCancellation()
            let prefix = "model.language_model.layers.\(layer).mlp.experts."
            let gateUp = prefix + "gate_up_proj"
            let down = prefix + "down_proj"
            let names = QwenBF16RoutedSourceNames(
                gateUpShardName: try Self.shard(gateUp, mapping),
                gateUpTensorName: gateUp,
                downShardName: try Self.shard(down, mapping),
                downTensorName: down)
            try admitRouted(names.gateUpShardName, gateUp,
                            shape: [expertCount, doubledIntermediate, hiddenWidth])
            try admitRouted(names.downShardName, down,
                            shape: [expertCount, hiddenWidth, intermediate])
            routedByLayer.append(names)
        }
        let entry = [
            try spec(Self.embeddingTensorName, .embedding, vocab, hidden),
            try spec(Self.headTensorName, .head, vocab, hidden),
        ]
        let entryWeights = try QwenBF16Weights(context: context, source: source,
            specifications: entry, residencyBudget: consumed)
        let finalNorm = try vector("model.language_model.norm.weight", [UInt64(hidden)])
        var built: [Layer] = []
        built.reserveCapacity(architecture.layers)
        for layer in 0..<architecture.layers {
            try Task.checkCancellation()
            let prefix = "model.language_model.layers.\(layer)."
            let linearPrefix = prefix + "linear_attn."
            let attentionPrefix = prefix + "self_attn."
            let sharedPrefix = prefix + "mlp.shared_expert."
            let router = prefix + "mlp.gate.weight"
            let shared = QwenBF16SharedNames(
                gate: sharedPrefix + "gate_proj.weight",
                up: sharedPrefix + "up_proj.weight",
                down: sharedPrefix + "down_proj.weight",
                outputGate: prefix + "mlp.shared_expert_gate.weight")
            let routed = routedByLayer[layer]
            var specs = [
                try spec(router, .router, architecture.experts, hidden),
                try spec(shared.gate, .sharedGate, architecture.sharedIntermediateSize, hidden),
                try spec(shared.up, .sharedUp, architecture.sharedIntermediateSize, hidden),
                try spec(shared.down, .sharedDown, hidden, architecture.sharedIntermediateSize),
                try spec(shared.outputGate, .sharedOutputGate, 1, hidden),
            ]
            let inputNorm = try vector(prefix + "input_layernorm.weight", [UInt64(hidden)])
            let postNorm = try vector(prefix + "post_attention_layernorm.weight", [UInt64(hidden)])
            var fullNames: QwenBF16FullAttentionNames?
            var queryNorm: [Float]?
            var keyNorm: [Float]?
            var linearNames: QwenBF16LinearNames?
            var linearVectors: QwenBF16LinearVectors?
            if architecture.layerKinds[layer] == .fullAttention {
                let queryWidth = architecture.queryHeads * architecture.headDimension
                let kvWidth = architecture.keyValueHeads * architecture.headDimension
                let names = QwenBF16FullAttentionNames(
                    query: attentionPrefix + "q_proj.weight",
                    key: attentionPrefix + "k_proj.weight",
                    value: attentionPrefix + "v_proj.weight",
                    output: attentionPrefix + "o_proj.weight")
                specs += [try spec(names.query, .dense, 2 * queryWidth, hidden),
                          try spec(names.key, .dense, kvWidth, hidden),
                          try spec(names.value, .dense, kvWidth, hidden),
                          try spec(names.output, .dense, hidden, queryWidth)]
                queryNorm = try vector(attentionPrefix + "q_norm.weight",
                                       [UInt64(architecture.headDimension)])
                keyNorm = try vector(attentionPrefix + "k_norm.weight",
                                     [UInt64(architecture.headDimension)])
                fullNames = names
            } else {
                let keyWidth = architecture.linearKeyHeads * architecture.linearKeyDimension
                let valueWidth = architecture.linearValueHeads * architecture.linearValueDimension
                let channels = 2 * keyWidth + valueWidth
                let names = QwenBF16LinearNames(
                    qkv: linearPrefix + "in_proj_qkv.weight",
                    z: linearPrefix + "in_proj_z.weight",
                    b: linearPrefix + "in_proj_b.weight",
                    a: linearPrefix + "in_proj_a.weight",
                    output: linearPrefix + "out_proj.weight")
                specs += [try spec(names.qkv, .dense, channels, hidden),
                          try spec(names.z, .dense, valueWidth, hidden),
                          try spec(names.b, .dense, architecture.linearValueHeads, hidden),
                          try spec(names.a, .dense, architecture.linearValueHeads, hidden),
                          try spec(names.output, .dense, hidden, valueWidth)]
                linearVectors = QwenBF16LinearVectors(
                    convolution: try vector(linearPrefix + "conv1d.weight",
                        [UInt64(channels), 1, UInt64(architecture.convolutionWidth)]),
                    normalization: try vector(linearPrefix + "norm.weight",
                        [UInt64(architecture.linearValueDimension)]),
                    aLog: try vector(linearPrefix + "A_log",
                        [UInt64(architecture.linearValueHeads)]),
                    timeStepBias: try vector(linearPrefix + "dt_bias",
                        [UInt64(architecture.linearValueHeads)]))
                linearNames = names
            }
            let prior = consumed - specs.reduce(UInt64(0)) { $0 + UInt64($1.rows) * UInt64($1.columns) * 2 }
            let resident = try QwenBF16Weights(context: context, source: source,
                specifications: specs, residencyBudget: consumed - prior)
            built.append(Layer(weights: resident, inputNorm: inputNorm, postNorm: postNorm,
                routerName: router, sharedNames: shared, routedNames: routed,
                fullNames: fullNames, queryNorm: queryNorm, keyNorm: keyNorm,
                linearNames: linearNames, linearVectors: linearVectors))
        }
        if let trustReceipt { try source.validateTrustedReceipt(trustReceipt) }
        else { try source.validateAcceptedFiles() }
        self.architecture = architecture
        self.visionArchitecture = visionArchitecture
        self.sourceIdentity = sourceIdentity
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.expertCacheBytesPerLayer = cacheBytesPerLayer
        self.expertCacheReservationBytes = cacheReservationBytes
        self.residentWeightBytes = consumed
        self.expertCacheQuota = ExpertCacheQuota(capacity: residencyBudgetBytes - consumed)
        self.sourceIntegrityPolicy = sourceIntegrityPolicy
        self.source = source
        self.trustReceipt = trustReceipt
        self.context = context
        self.sourceContentSHA256 = sourceContentSHA256
        // No processor file is opened for text-only source admission.
        self.sourceIndex = mapping
        self.entryWeights = entryWeights
        self.layers = built
        self.finalNorm = finalNorm
    }

    func visionProcessorSHA256() throws -> String {
        try revalidateSource()
        if sourceIdentity != nil {
            guard let pinned = OfficialQwenSourceIdentity.pinned.sidecarSHA256[
                "preprocessor_config.json"] else {
                throw VisionPackError.invalidMetadata("pinned source processor missing")
            }
            return pinned
        }
        let directory = try GTurboModelDirectory(protectedOfficialSource: source)
        let bytes = try directory.readMetadata("preprocessor_config.json", maxBytes: 256 * 1024)
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    func hasSourceTensor(_ name: String) -> Bool { sourceIndex[name] != nil }

    func admitVisionTensor(_ name: String) throws -> OfficialSourceHandle.TensorRange {
        guard name.hasPrefix("model.visual."), let shard = sourceIndex[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        return try source.admitTensor(shardName: shard, tensorName: name)
    }

    func revalidateSource() throws {
        if let trustReceipt { try source.validateTrustedReceipt(trustReceipt) }
        else { try source.validateAcceptedFiles() }
    }

    func makeDecodeReceiptValidator() throws -> OfficialSourceHandle.DecodeReceiptValidator? {
        guard let trustReceipt else { return nil }
        return try source.makeDecodeReceiptValidator(trustReceipt)
    }

    func revalidateSource(validator: OfficialSourceHandle.DecodeReceiptValidator,
                          latch: OfficialSourceDecodeCancellation, parallel: Bool,
                          measurement: OfficialSourceDecodeMeasurement) throws {
        guard let trustReceipt, validator.matches(handle: source, receipt: trustReceipt) else {
            throw ModelError.indexCorrupt(detail: "decode receipt owner changed")
        }
        try validator.validate(latch: latch, parallel: parallel, measurement: measurement)
    }

    /// Compare the service's claimed digest with this admitted model, then
    /// recheck its retained receipt and literal source names.
    package func revalidateLoadedSource(contentDigest: String) throws {
        guard let sourceIdentity,
              sourceIdentity.descriptorContentSHA256 == contentDigest else {
            throw ModelError.indexCorrupt(detail: "loaded source content identity changed")
        }
        try revalidateSource()
    }

    func makeRunner(context: MetalContext, expertSlotCount: Int,
                    maxContext: Int) throws -> QwenOfficialSourceRunner {
        guard context.device.registryID == self.context.device.registryID else {
            throw ModelError.indexCorrupt(detail: "source runner uses another Metal device")
        }
        guard expertSlotCount == expertCacheSlots else {
            throw ModelError.indexCorrupt(detail: "source runner cache configuration changed")
        }
        return try QwenOfficialSourceRunner(model: self, maxContext: maxContext,
                                            expertSlotCount: expertSlotCount)
    }

    private static func shard(_ name: String, _ mapping: [String: String]) throws -> String {
        guard let value = mapping[name] else { throw ModelError.tensorNotFound(name: name) }
        return value
    }

    private static func metadata(_ name: String, from directory: GTurboModelDirectory,
                                 requirePin: Bool) throws -> Data {
        let bytes = try directory.readMetadata(name, maxBytes: 8 * 1024 * 1024)
        if requirePin {
            guard let expected = OfficialQwenSourceIdentity.pinned.sidecarSHA256[name] else {
                throw ModelError.tensorNotFound(name: name)
            }
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            guard digest == expected else { throw ModelError.checksumMismatch(file: name) }
        }
        return bytes
    }

    private static func decodeIndex(_ data: Data, allowed: Set<String>) throws -> [String: String] {
        struct Index: Decodable { let weightMap: [String: String] }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let entries = try decoder.decode(Index.self, from: data).weightMap
        guard !entries.isEmpty, entries.count <= 20_000,
              entries.values.allSatisfy({ allowed.contains($0) }) else {
            throw ModelError.indexCorrupt(detail: "source tensor index is invalid")
        }
        return entries
    }
}
