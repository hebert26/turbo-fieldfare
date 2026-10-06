import Foundation
import TurboFieldfareFormat

struct SubTensorEntry: Sendable, Equatable {
    let offset: UInt64    // relative to the expert blob's start
    let size: UInt64
    let dtype: String     // "U32" | "BF16"
    let shape: [UInt32]
    let bits: Int?        // weight bit width override (4 or 8), if applicable
}

struct ExpertEntry: Sendable {
    /// Logical routed-expert id used by the model/router.
    let expert: Int
    /// Physical rank inside the layer file. Old identity layouts use
    /// `physicalRank == expert`.
    let physicalRank: Int
    /// Absolute byte offset of this expert blob's start inside its layer file.
    let offset: UInt64
    /// Total bytes consumed by this expert blob (== `expertStride`).
    let size: UInt64
    /// Sub-tensors keyed by role (gate / up / down / shared) and component
    /// (raw, `_scales`, `_biases`).
    let subTensors: [String: SubTensorEntry]

    init(expert: Int,
                physicalRank: Int? = nil,
                offset: UInt64,
                size: UInt64,
                subTensors: [String: SubTensorEntry]) {
        self.expert = expert
        self.physicalRank = physicalRank ?? expert
        self.offset = offset
        self.size = size
        self.subTensors = subTensors
    }
}

struct PackedAffineDescriptor: Sendable, Equatable {
    let role: String
    let sourceName: String
    let shape: [UInt32]
    let bitWidth: Int
    let valuesOffset: UInt64
    let valuesSize: UInt64
    let scalesOffset: UInt64
    let scalesSize: UInt64
    let biasesOffset: UInt64
    let biasesSize: UInt64
}

struct LayerLayout: Sendable {
    let layer: Int
    let file: String          // basename, e.g. "layer_00.bin"
    let experts: [ExpertEntry]
    /// Present only for the writer-emitted Qwen v2 layout. Offsets are relative
    /// to one expert blob and are identical for every expert in the layer.
    let affineDescriptors: [String: PackedAffineDescriptor]

    init(layer: Int, file: String, experts: [ExpertEntry],
         affineDescriptors: [String: PackedAffineDescriptor] = [:]) {
        self.layer = layer
        self.file = file
        self.experts = experts
        self.affineDescriptors = affineDescriptors
    }
}

struct PackedExpertsLayout: Sendable {
    let expertStride: UInt64
    let numLayers: Int
    let expertsPerLayer: Int
    let layers: [LayerLayout]

    /// Resolve `(layer, expert)` -> `ExpertEntry`. O(1).
    func expert(layer: Int, expert: Int) -> ExpertEntry {
        return layers[layer].experts[expert]
    }
}

enum PackedExpertsLayoutReader {
    static let defaultMaxBytes: UInt64 = 16 * 1024 * 1024

    static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: nil, maxBytes: maxBytes)
    }

    package static func load(directoryURL: URL,
                             manifest: Manifest,
                             maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        try load(directoryURL: directoryURL, manifest: Optional(manifest), maxBytes: maxBytes)
    }

    private static func load(directoryURL: URL,
                             manifest: Manifest?,
                             maxBytes: UInt64) throws -> PackedExpertsLayout {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata(
            "packed_experts/layout.json", maxBytes: maxBytes)
        return try decode(data: data, manifest: manifest)
    }

    package static func decode(data: Data,
                               manifest: Manifest?) throws -> PackedExpertsLayout {
        let wire: GTurboPackedExpertsLayoutV1
        do {
            wire = try GTurboPackedExpertsLayoutCodec.decode(data)
            if let manifest {
                try GTurboV1StructuralValidator.crossValidate(
                    manifestNumLayers: manifest.numLayers,
                    manifestExpertsPerLayer: manifest.expertsPerLayer,
                    manifestExpertStride: manifest.expertStride,
                    manifestFileSizes: manifest.files.mapValues(\.size),
                    layout: wire)
            }
        } catch {
            throw ModelError.indexCorrupt(detail: "layout.json: \(error)")
        }
        let layers = wire.layers.sorted { $0.layer < $1.layer }.map { layer -> LayerLayout in
            let experts = layer.experts.enumerated().map { position, expert -> ExpertEntry in
                let logical = expert.expert ?? position
                return ExpertEntry(
                    expert: logical,
                    physicalRank: expert.physicalRank,
                    offset: expert.offset,
                    size: expert.size,
                    subTensors: expert.tensors.mapValues {
                        SubTensorEntry(offset: $0.offset, size: $0.size,
                                       dtype: $0.dtype, shape: $0.shape,
                                       bits: $0.bits)
                    })
            }.sorted { $0.expert < $1.expert }
            return LayerLayout(layer: layer.layer, file: layer.file, experts: experts)
        }
        return PackedExpertsLayout(expertStride: wire.expertStride,
                                   numLayers: wire.numLayers,
                                   expertsPerLayer: wire.expertsPerLayer,
                                   layers: layers)
    }

    /// Loads the distinct writer-emitted Qwen layout version 2. This does not
    /// reinterpret the Qwen document as the legacy packed-layout v1 wire.
    static func loadQwenV2(directoryURL: URL,
                           manifest: LoadedModelManifest,
                           maxBytes: UInt64 = defaultMaxBytes) throws -> PackedExpertsLayout {
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        let data = try directory.readMetadata("packed_experts/layout.json", maxBytes: maxBytes)
        return try decodeQwenV2(data: data, manifest: manifest)
    }

    static func decodeQwenV2(data: Data,
                             manifest: LoadedModelManifest) throws -> PackedExpertsLayout {
        guard manifest.descriptor.family == .qwen3_6,
              case let .qwen3_6(architecture) = manifest.architecture else {
            throw ModelError.indexCorrupt(detail: "layout v2 requires a verified Qwen manifest")
        }
        guard architecture.numLayers == manifest.numLayers,
              architecture.numberOfExperts == manifest.expertsPerLayer,
              architecture.numLayers == 40,
              architecture.numberOfExperts == 256 else {
            throw ModelError.indexCorrupt(detail: "verified Qwen architecture/streaming dimensions disagree")
        }
        guard let quantization = manifest.descriptor.quantization.first(where: {
            $0.category == "routedExpert"
        }), quantization.storage == "affineInt4",
           quantization.groupSize == Quantization.groupSize,
           quantization.scaleType?.lowercased() == "bf16",
           quantization.biasType?.lowercased() == "bf16" else {
            throw ModelError.indexCorrupt(detail: "Qwen routed experts are not affine INT4 group-64")
        }
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: architecture.numLayers,
            expertCount: architecture.numberOfExperts,
            hiddenSize: architecture.hiddenSize,
            intermediateSize: architecture.routedExpertIntermediateSize,
            expertStride: manifest.expertStride)
        return try decodeQwenV2(
            data: data,
            configuration: configuration,
            manifestFileSizes: manifest.files.mapValues(\.size))
    }

    /// Checked non-identity seam used by bounded synthetic layouts. Supplying
    /// this configuration never creates or implies an InstalledModelDescriptor.
    static func decodeQwenV2(
        data: Data,
        configuration: QwenPackedLayoutConfiguration,
        manifestFileSizes: [String: UInt64],
        sourceSchema: QwenPackedSourceSchema = .production
    ) throws -> PackedExpertsLayout {
        do {
            try validateQwenV2JSONShape(data, sourceSchema: sourceSchema)
            let wire = try JSONDecoder().decode(QwenPackedLayoutV2Wire.self, from: data)
            guard wire.version == 2 else {
                throw ModelError.indexCorrupt(detail: "layout v2 has unsupported version \(wire.version)")
            }
            guard wire.layers.count == configuration.layerCount else {
                throw ModelError.indexCorrupt(detail: "layout v2 layer count mismatch")
            }
            var layerIDs = Set<Int>()
            var paths = Set<String>()
            let layers = try wire.layers.map { layer -> LayerLayout in
                guard layer.layer >= 0, layer.layer < configuration.layerCount,
                      layerIDs.insert(layer.layer).inserted else {
                    throw ModelError.indexCorrupt(detail: "layout v2 has duplicate or invalid layer id")
                }
                let basename = try qwenLayerBasename(layer.path)
                guard paths.insert(layer.path).inserted,
                      layer.experts == configuration.expertCount,
                      layer.stride == configuration.expertStride else {
                    throw ModelError.indexCorrupt(detail: "layout v2 layer metadata mismatch")
                }
                let expectedFileSize = try checkedMultiply(
                    UInt64(configuration.expertCount), configuration.expertStride,
                    field: "layout v2 layer file size")
                guard manifestFileSizes[layer.path] == expectedFileSize else {
                    throw ModelError.indexCorrupt(
                        detail: "layout v2 manifest file size mismatch for \(layer.path)")
                }
                guard layer.sources.count == 2,
                      Set(layer.sources.map(\.role)) == Set(["gate_up", "down"]),
                      Set(layer.sources.map(\.name)).count == 2 else {
                    throw ModelError.indexCorrupt(detail: "layout v2 requires gate_up and down sources")
                }
                var descriptors: [String: PackedAffineDescriptor] = [:]
                var occupied: [(UInt64, UInt64, String)] = []
                for source in layer.sources {
                    switch sourceSchema {
                    case .production:
                        guard source.totalSize == nil else {
                            throw ModelError.indexCorrupt(
                                detail: "layout v2 production source has fixture totalSize")
                        }
                    case .fixtureRedundantEnd:
                        let expectedEnd = try checkedAdd(
                            source.biasesOffset, source.biasesSize,
                            field: "fixture source totalSize")
                        guard source.totalSize == expectedEnd else {
                            throw ModelError.indexCorrupt(
                                detail: "layout v2 fixture source totalSize mismatch")
                        }
                    }
                    let expectedShape: [UInt64]
                    switch source.role {
                    case "gate_up":
                        expectedShape = [try checkedMultiply(
                            2, UInt64(configuration.intermediateSize), field: "gate_up rows"),
                            UInt64(configuration.hiddenSize)]
                    case "down":
                        expectedShape = [UInt64(configuration.hiddenSize),
                                         UInt64(configuration.intermediateSize)]
                    default:
                        throw ModelError.indexCorrupt(detail: "layout v2 has unknown source role")
                    }
                    guard source.shape == expectedShape else {
                        throw ModelError.indexCorrupt(
                            detail: "layout v2 shape mismatch for \(source.role)")
                    }
                    let sizes = try affineSizes(shape: source.shape, bitWidth: 4)
                    let expectedScalesOffset = try checkedAdd(
                        source.valuesOffset, source.valuesSize,
                        field: "values/scales adjacency")
                    let expectedBiasesOffset = try checkedAdd(
                        source.scalesOffset, source.scalesSize,
                        field: "scales/biases adjacency")
                    guard source.valuesSize == sizes.values,
                          source.scalesSize == sizes.metadata,
                          source.biasesSize == sizes.metadata,
                          source.scalesOffset == expectedScalesOffset,
                          source.biasesOffset == expectedBiasesOffset,
                          source.scalesOffset.isMultiple(of: 2),
                          source.biasesOffset.isMultiple(of: 2) else {
                        throw ModelError.indexCorrupt(
                            detail: "layout v2 affine size/alignment mismatch for \(source.role)")
                    }
                    let valuesEnd = try checkedAdd(
                        source.valuesOffset, source.valuesSize, field: "values range")
                    let scalesEnd = try checkedAdd(
                        source.scalesOffset, source.scalesSize, field: "scales range")
                    let biasesEnd = try checkedAdd(
                        source.biasesOffset, source.biasesSize, field: "biases range")
                    guard valuesEnd <= configuration.expertStride,
                          scalesEnd <= configuration.expertStride,
                          biasesEnd <= configuration.expertStride,
                          UInt32(exactly: valuesEnd) != nil,
                          UInt32(exactly: scalesEnd) != nil,
                          UInt32(exactly: biasesEnd) != nil else {
                        throw ModelError.indexCorrupt(
                            detail: "layout v2 affine range exceeds expert stride/address space")
                    }
                    occupied.append((source.valuesOffset, valuesEnd, "\(source.role).values"))
                    occupied.append((source.scalesOffset, scalesEnd, "\(source.role).scales"))
                    occupied.append((source.biasesOffset, biasesEnd, "\(source.role).biases"))
                    guard let shape = exactUInt32Shape(source.shape) else {
                        throw ModelError.indexCorrupt(detail: "layout v2 shape is not UInt32-addressable")
                    }
                    descriptors[source.role] = PackedAffineDescriptor(
                        role: source.role, sourceName: source.name, shape: shape, bitWidth: 4,
                        valuesOffset: source.valuesOffset, valuesSize: source.valuesSize,
                        scalesOffset: source.scalesOffset, scalesSize: source.scalesSize,
                        biasesOffset: source.biasesOffset, biasesSize: source.biasesSize)
                }
                let sorted = occupied.sorted { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }
                guard sorted.first?.0 == 0 else {
                    throw ModelError.indexCorrupt(detail: "layout v2 components do not start at zero")
                }
                for pair in zip(sorted, sorted.dropFirst()) where pair.0.1 != pair.1.0 {
                    throw ModelError.indexCorrupt(
                        detail: "layout v2 components are not contiguous between \(pair.0.2) and \(pair.1.2)")
                }
                let experts = try (0..<configuration.expertCount).map { expert -> ExpertEntry in
                    let offset = try checkedMultiply(
                        UInt64(expert), configuration.expertStride, field: "expert offset")
                    var tensors: [String: SubTensorEntry] = [:]
                    for descriptor in descriptors.values {
                        tensors[descriptor.role] = SubTensorEntry(
                            offset: descriptor.valuesOffset, size: descriptor.valuesSize,
                            dtype: "U32", shape: descriptor.shape, bits: descriptor.bitWidth)
                        let rows = descriptor.shape[0]
                        let groups = (descriptor.shape[1] + UInt32(Quantization.groupSize) - 1)
                            / UInt32(Quantization.groupSize)
                        let metadataShape = [rows, groups]
                        tensors["\(descriptor.role)_scales"] = SubTensorEntry(
                            offset: descriptor.scalesOffset, size: descriptor.scalesSize,
                            dtype: "BF16", shape: metadataShape, bits: nil)
                        tensors["\(descriptor.role)_biases"] = SubTensorEntry(
                            offset: descriptor.biasesOffset, size: descriptor.biasesSize,
                            dtype: "BF16", shape: metadataShape, bits: nil)
                    }
                    return ExpertEntry(
                        expert: expert, physicalRank: expert, offset: offset,
                        size: configuration.expertStride, subTensors: tensors)
                }
                return LayerLayout(
                    layer: layer.layer, file: basename, experts: experts,
                    affineDescriptors: descriptors)
            }
            return PackedExpertsLayout(
                expertStride: configuration.expertStride,
                numLayers: configuration.layerCount,
                expertsPerLayer: configuration.expertCount,
                layers: layers.sorted { $0.layer < $1.layer })
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "layout v2: \(error)")
        }
    }

    private static func validateQwenV2JSONShape(
        _ data: Data,
        sourceSchema: QwenPackedSourceSchema
    ) throws {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any],
              Set(root.keys) == Set(["version", "layers"]),
              let layers = root["layers"] as? [[String: Any]] else {
            throw ModelError.indexCorrupt(detail: "layout v2 has unknown or missing top-level fields")
        }
        let layerKeys = Set(["layer", "path", "experts", "stride", "sources"])
        let productionSourceKeys = Set([
            "name", "role", "shape", "valuesOffset", "valuesSize",
            "scalesOffset", "scalesSize", "biasesOffset", "biasesSize",
        ])
        let sourceKeys: Set<String>
        switch sourceSchema {
        case .production:
            sourceKeys = productionSourceKeys
        case .fixtureRedundantEnd:
            sourceKeys = productionSourceKeys.union(["totalSize"])
        }
        for layer in layers {
            guard Set(layer.keys) == layerKeys,
                  let sources = layer["sources"] as? [[String: Any]] else {
                throw ModelError.indexCorrupt(detail: "layout v2 has unknown or missing layer fields")
            }
            for source in sources where Set(source.keys) != sourceKeys {
                throw ModelError.indexCorrupt(detail: "layout v2 has unknown or missing source fields")
            }
        }
    }

    private static func qwenLayerBasename(_ path: String) throws -> String {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2, components[0] == "packed_experts",
              !components[1].isEmpty, components[1] != ".", components[1] != "..",
              !path.contains("\\"), !path.hasPrefix("/") else {
            throw ModelError.indexCorrupt(detail: "layout v2 has unsafe layer path")
        }
        return String(components[1])
    }

    private static func affineSizes(shape: [UInt64], bitWidth: UInt64) throws
        -> (values: UInt64, metadata: UInt64) {
        guard shape.count == 2, shape.allSatisfy({ $0 > 0 }) else {
            throw ModelError.indexCorrupt(detail: "layout v2 affine shape is invalid")
        }
        let rows = shape[0]
        let columns = shape[1]
        let completeGroups = columns / UInt64(Quantization.groupSize)
        let remainder = columns % UInt64(Quantization.groupSize)
        let completeBytes = try checkedMultiply(
            completeGroups, UInt64(Quantization.groupSize) * bitWidth / 8,
            field: "affine complete groups")
        let remainderBits = try checkedMultiply(remainder, bitWidth, field: "affine tail bits")
        let tailBytes = remainderBits / 8 + (remainderBits.isMultiple(of: 8) ? 0 : 1)
        let rowBytes = try checkedAdd(completeBytes, tailBytes, field: "affine row bytes")
        let values = try checkedMultiply(rows, rowBytes, field: "affine values bytes")
        let groups = columns / UInt64(Quantization.groupSize)
            + (columns.isMultiple(of: UInt64(Quantization.groupSize)) ? 0 : 1)
        let metadata = try checkedMultiply(
            try checkedMultiply(rows, groups, field: "affine group count"), 2,
            field: "affine metadata bytes")
        return (values, metadata)
    }

    private static func exactUInt32Shape(_ shape: [UInt64]) -> [UInt32]? {
        var result: [UInt32] = []
        result.reserveCapacity(shape.count)
        for extent in shape {
            guard let value = UInt32(exactly: extent) else { return nil }
            result.append(value)
        }
        return result
    }

    private static func checkedAdd(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
        let (result, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw ModelError.indexCorrupt(detail: "\(field) overflow") }
        return result
    }

    private static func checkedMultiply(
        _ lhs: UInt64, _ rhs: UInt64, field: String
    ) throws -> UInt64 {
        let (result, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw ModelError.indexCorrupt(detail: "\(field) overflow") }
        return result
    }

}

enum QwenPackedSourceSchema: Sendable, Equatable {
    /// Writer-v2 production schema. Unknown source fields remain rejected.
    case production
    /// Bounded, no-descriptor fixture schema with a required redundant source
    /// end. This never changes official admission or production decoding.
    case fixtureRedundantEnd
}

struct QwenPackedLayoutConfiguration: Sendable, Equatable {
    let layerCount: Int
    let expertCount: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let expertStride: UInt64

    init(layerCount: Int, expertCount: Int, hiddenSize: Int,
         intermediateSize: Int, expertStride: UInt64) throws {
        guard layerCount > 0, expertCount >= 8, hiddenSize > 0,
              intermediateSize > 0, expertStride > 0,
              expertStride.isMultiple(of: GTurboFormatV1.alignmentBytes),
              UInt32(exactly: hiddenSize) != nil,
              UInt32(exactly: intermediateSize) != nil,
              expertStride <= UInt64(UInt32.max) else {
            throw ModelError.indexCorrupt(detail: "layout v2 configuration is invalid")
        }
        self.layerCount = layerCount
        self.expertCount = expertCount
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.expertStride = expertStride
    }
}

private struct QwenPackedLayoutV2Wire: Decodable {
    let version: Int
    let layers: [QwenPackedLayerV2Wire]
}

private struct QwenPackedLayerV2Wire: Decodable {
    let layer: Int
    let path: String
    let experts: Int
    let stride: UInt64
    let sources: [QwenPackedSourceV2Wire]
}

private struct QwenPackedSourceV2Wire: Decodable {
    let name: String
    let role: String
    let shape: [UInt64]
    let valuesOffset: UInt64
    let valuesSize: UInt64
    let scalesOffset: UInt64
    let scalesSize: UInt64
    let biasesOffset: UInt64
    let biasesSize: UInt64
    let totalSize: UInt64?
}
