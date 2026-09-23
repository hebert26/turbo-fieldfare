import CryptoKit
import Darwin
import Foundation
import Metal

/// Runtime geometry used by the Qwen text mapper and runner. The internal tiny
/// fixture can supply smaller dimensions, but it never creates an installed
/// model descriptor or passes through official-family admission.
struct QwenTextArchitecture: Equatable, Sendable {
    enum LayerKind: String, Codable, Equatable, Sendable {
        case linearAttention = "linear_attention"
        case fullAttention = "full_attention"
    }

    var hiddenSize: Int
    var layerKinds: [LayerKind]
    var queryHeads: Int
    var keyValueHeads: Int
    var headDimension: Int
    var attentionOutputGate: Bool
    var convolutionWidth: Int
    var linearKeyHeads: Int
    var linearKeyDimension: Int
    var linearValueHeads: Int
    var linearValueDimension: Int
    var partialRotaryFactor: Double
    var ropeTheta: Double
    var mropeSections: [Int]
    var experts: Int
    var expertsPerToken: Int
    var routedIntermediateSize: Int
    var sharedIntermediateSize: Int
    var vocabularySize: Int
    var untiedHead: Bool
    var hiddenActivation: String

    var layers: Int { layerKinds.count }
    var fullAttentionLayerMask: [UInt8] {
        layerKinds.map { $0 == .fullAttention ? 1 : 0 }
    }

    init(
        hiddenSize: Int, layerKinds: [LayerKind], queryHeads: Int,
        keyValueHeads: Int, headDimension: Int, attentionOutputGate: Bool,
        convolutionWidth: Int, linearKeyHeads: Int, linearKeyDimension: Int,
        linearValueHeads: Int, linearValueDimension: Int,
        partialRotaryFactor: Double, ropeTheta: Double, mropeSections: [Int],
        experts: Int, expertsPerToken: Int, routedIntermediateSize: Int,
        sharedIntermediateSize: Int, vocabularySize: Int, untiedHead: Bool,
        hiddenActivation: String
    ) {
        self.hiddenSize = hiddenSize
        self.layerKinds = layerKinds
        self.queryHeads = queryHeads
        self.keyValueHeads = keyValueHeads
        self.headDimension = headDimension
        self.attentionOutputGate = attentionOutputGate
        self.convolutionWidth = convolutionWidth
        self.linearKeyHeads = linearKeyHeads
        self.linearKeyDimension = linearKeyDimension
        self.linearValueHeads = linearValueHeads
        self.linearValueDimension = linearValueDimension
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.mropeSections = mropeSections
        self.experts = experts
        self.expertsPerToken = expertsPerToken
        self.routedIntermediateSize = routedIntermediateSize
        self.sharedIntermediateSize = sharedIntermediateSize
        self.vocabularySize = vocabularySize
        self.untiedHead = untiedHead
        self.hiddenActivation = hiddenActivation
    }

    init(official configuration: QwenArchConfig) {
        hiddenSize = configuration.hiddenSize
        layerKinds = configuration.fullAttentionLayerMask.map {
            $0 == 1 ? .fullAttention : .linearAttention
        }
        queryHeads = configuration.numAttentionHeads
        keyValueHeads = configuration.numKeyValueHeads
        headDimension = configuration.headDimension
        attentionOutputGate = configuration.attentionOutputGate
        convolutionWidth = configuration.linearConvolutionKernel
        linearKeyHeads = configuration.linearKeyHeads
        linearKeyDimension = configuration.linearKeyHeadDimension
        linearValueHeads = configuration.linearValueHeads
        linearValueDimension = configuration.linearValueHeadDimension
        partialRotaryFactor = configuration.partialRotaryFactor
        ropeTheta = configuration.ropeTheta
        mropeSections = configuration.mropeSections
        experts = configuration.numberOfExperts
        expertsPerToken = configuration.expertsPerToken
        routedIntermediateSize = configuration.routedExpertIntermediateSize
        sharedIntermediateSize = configuration.sharedExpertIntermediateSize
        vocabularySize = configuration.vocabularySize
        untiedHead = !configuration.tiedWordEmbeddings
        hiddenActivation = configuration.hiddenActivation
    }

    fileprivate func validate() throws {
        guard hiddenSize > 0, !layerKinds.isEmpty,
              queryHeads > 0, keyValueHeads > 0,
              queryHeads.isMultiple(of: keyValueHeads), headDimension > 0,
              convolutionWidth == 4,
              linearKeyHeads > 0, linearKeyDimension > 0,
              linearValueHeads > 0, linearValueDimension > 0,
              linearValueHeads.isMultiple(of: linearKeyHeads),
              partialRotaryFactor.isFinite, partialRotaryFactor > 0,
              partialRotaryFactor <= 1,
              ropeTheta.isFinite, ropeTheta > 0,
              !mropeSections.isEmpty, mropeSections.allSatisfy({ $0 > 0 }),
              2 * mropeSections.reduce(0, +)
                == Int(Double(headDimension) * partialRotaryFactor),
              experts >= 8, expertsPerToken == 8,
              routedIntermediateSize > 0, sharedIntermediateSize > 0,
              vocabularySize > 0, untiedHead, hiddenActivation == "silu" else {
            throw ModelError.indexCorrupt(detail: "invalid Qwen text architecture")
        }
    }
}

enum QwenTextTensorStorage: String, Codable, Equatable, Sendable {
    case bf16
    case fp32
    case affineInt4
    case affineInt8
}

struct QwenTextTensorRegion: Codable, Equatable, Sendable {
    var name: String
    var file: String
    var offset: UInt64
    var size: UInt64
    var shape: [UInt64]
    var storage: QwenTextTensorStorage
    var quantizationCategory: String?
}

struct QwenTextFixtureRecords: Sendable {
    static let approvedFixtureAggregateSHA256 =
        "d4edeb3c98db5c084b6cbf9cf38b56859c717c6d0c20c522f5e1382ed39db7c6"

    struct VirtualFile: Equatable, Sendable {
        var bytes: Data
        var byteCount: Int
        var sha256: String
    }

    var architecture: QwenTextArchitecture
    var virtualFiles: [String: VirtualFile]
    var residentTensorRegions: [QwenTextTensorRegion]
    var expertTensorRegions: [QwenTextTensorRegion]
    var expertLayoutData: Data
    var expertStride: UInt64
    var virtualFileAggregateSHA256: String

    static func decode(
        jsonData: Data,
        maximumEmbeddedBytes: Int = 2 * 1024 * 1024
    ) throws -> QwenTextFixtureRecords {
        guard maximumEmbeddedBytes > 0 else {
            throw ModelError.indexCorrupt(detail: "bounded fixture byte limit is invalid")
        }
        let envelope: FixtureEnvelope
        do {
            envelope = try JSONDecoder().decode(FixtureEnvelope.self, from: jsonData)
        } catch {
            throw ModelError.indexCorrupt(detail: "Qwen fixture JSON: \(error)")
        }
        guard envelope.schemaVersion == "qwen36-tiny-text-model-v1" else {
            throw ModelError.indexCorrupt(detail: "unsupported Qwen text fixture schema")
        }
        let architecture = envelope.tinyArchitecture.runtimeArchitecture
        try architecture.validate()

        var files: [String: VirtualFile] = [:]
        var total = 0
        for (path, wire) in envelope.packing.virtualFiles {
            try validateRelativePath(path)
            guard wire.encoding == "base64",
                  let bytes = Data(base64Encoded: wire.bytes),
                  wire.byteCount == bytes.count,
                  Self.sha256(bytes) == wire.sha256.lowercased() else {
                throw ModelError.indexCorrupt(
                    detail: "Qwen fixture file \(path) has invalid bytes, size, or hash")
            }
            let (next, overflow) = total.addingReportingOverflow(bytes.count)
            guard !overflow, next <= maximumEmbeddedBytes else {
                throw ModelError.indexCorrupt(
                    detail: "bounded Qwen fixture exceeds \(maximumEmbeddedBytes) embedded bytes")
            }
            total = next
            files[path] = VirtualFile(
                bytes: bytes, byteCount: wire.byteCount,
                sha256: wire.sha256.lowercased())
        }
        guard total == envelope.packing.embeddedByteCount,
              let layout = files["packed_experts/layout.json"]?.bytes else {
            throw ModelError.indexCorrupt(detail: "Qwen fixture embedded byte total or layout is invalid")
        }
        let aggregate = aggregateSHA256(files: files)
        guard aggregate == envelope.packing.virtualFileAggregateSHA256.lowercased(),
              aggregate == approvedFixtureAggregateSHA256 else {
            throw ModelError.indexCorrupt(detail: "Qwen fixture virtual-file aggregate mismatch")
        }
        let stride = envelope.fixtureRecords.expertLayout.layers.first?.stride ?? 0
        guard stride > 0,
              envelope.fixtureRecords.expertLayout.layers.allSatisfy({ $0.stride == stride }) else {
            throw ModelError.indexCorrupt(detail: "Qwen fixture expert stride is inconsistent")
        }
        return QwenTextFixtureRecords(
            architecture: architecture,
            virtualFiles: files,
            residentTensorRegions: envelope.fixtureRecords.residentTensorRegions,
            expertTensorRegions: envelope.fixtureRecords.expertTensorRegions,
            expertLayoutData: layout,
            expertStride: stride,
            virtualFileAggregateSHA256: aggregate)
    }

    private struct FixtureEnvelope: Decodable {
        let schemaVersion: String
        let tinyArchitecture: TinyArchitecture
        let packing: Packing
        let fixtureRecords: Records
    }

    private struct Packing: Decodable {
        let embeddedByteCount: Int
        let virtualFileAggregateSHA256: String
        let virtualFiles: [String: VirtualFileWire]
    }

    private struct VirtualFileWire: Decodable {
        let encoding: String
        let byteCount: Int
        let sha256: String
        let bytes: String
    }

    private struct Records: Decodable {
        let residentTensorRegions: [QwenTextTensorRegion]
        let expertTensorRegions: [QwenTextTensorRegion]
        let expertLayout: ExpertLayout
    }

    private struct ExpertLayout: Decodable {
        let version: Int
        let layers: [ExpertLayer]
    }

    private struct ExpertLayer: Decodable {
        let stride: UInt64
    }

    private struct TinyArchitecture: Decodable {
        let hiddenSize: Int
        let layerSchedule: [QwenTextArchitecture.LayerKind]
        let queryHeads: Int
        let keyValueHeads: Int
        let headDimension: Int
        let attentionOutputGate: Bool
        let convolutionWidth: Int
        let linearKeyHeads: Int
        let linearKeyDimension: Int
        let linearValueHeads: Int
        let linearValueDimension: Int
        let partialRotaryFactor: Double
        let ropeTheta: Double
        let mropeSections: [Int]
        let experts: Int
        let expertsPerToken: Int
        let routedIntermediateSize: Int
        let sharedIntermediateSize: Int
        let vocabularySize: Int
        let untiedHead: Bool
        let hiddenActivation: String

        var runtimeArchitecture: QwenTextArchitecture {
            QwenTextArchitecture(
                hiddenSize: hiddenSize,
                layerKinds: layerSchedule,
                queryHeads: queryHeads,
                keyValueHeads: keyValueHeads,
                headDimension: headDimension,
                attentionOutputGate: attentionOutputGate,
                convolutionWidth: convolutionWidth,
                linearKeyHeads: linearKeyHeads,
                linearKeyDimension: linearKeyDimension,
                linearValueHeads: linearValueHeads,
                linearValueDimension: linearValueDimension,
                partialRotaryFactor: partialRotaryFactor,
                ropeTheta: ropeTheta,
                mropeSections: mropeSections,
                experts: experts,
                expertsPerToken: expertsPerToken,
                routedIntermediateSize: routedIntermediateSize,
                sharedIntermediateSize: sharedIntermediateSize,
                vocabularySize: vocabularySize,
                untiedHead: untiedHead,
                hiddenActivation: hiddenActivation)
        }
    }

    fileprivate static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    fileprivate static func aggregateSHA256(files: [String: VirtualFile]) -> String {
        var hasher = SHA256()
        for path in files.keys.sorted() {
            let pathData = Data(path.utf8)
            var pathCount = UInt64(pathData.count).littleEndian
            var payloadCount = UInt64(files[path]!.bytes.count).littleEndian
            withUnsafeBytes(of: &pathCount) { hasher.update(data: Data($0)) }
            hasher.update(data: pathData)
            withUnsafeBytes(of: &payloadCount) { hasher.update(data: Data($0)) }
            hasher.update(data: files[path]!.bytes)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    fileprivate static func validateRelativePath(_ path: String) throws {
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"),
              !path.contains("\0"),
              path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({
                  !$0.isEmpty && $0 != "." && $0 != ".."
              }) else {
            throw ModelError.indexCorrupt(detail: "invalid Qwen fixture relative path \(path)")
        }
    }
}

private enum QwenTextPayload: @unchecked Sendable {
    case memory(Data)
    case file(GTurboModelDirectory, String)

    func read(offset: UInt64, count: UInt64, name: String) throws -> Data {
        guard offset <= UInt64(Int64.max), count <= UInt64(Int.max) else {
            throw ModelError.indexCorrupt(detail: "\(name) range is not addressable")
        }
        switch self {
        case .memory(let data):
            guard offset <= UInt64(data.count), count <= UInt64(data.count) - offset else {
                throw ModelError.indexCorrupt(detail: "\(name) range exceeds its payload")
            }
            return data.subdata(in: Int(offset)..<Int(offset + count))
        case .file(let directory, let path):
            let fd = try directory.openFile(path)
            defer { close(fd) }
            var data = Data(count: Int(count))
            try data.withUnsafeMutableBytes { raw in
                guard let base = raw.baseAddress else { return }
                var consumed = 0
                while consumed < raw.count {
                    let result = pread(
                        fd, base.advanced(by: consumed), raw.count - consumed,
                        off_t(offset) + off_t(consumed))
                    if result < 0, errno == EINTR { continue }
                    guard result > 0 else {
                        throw ModelError.indexCorrupt(detail: "short read for \(name)")
                    }
                    consumed += result
                }
            }
            return data
        }
    }
}

struct QwenTextTensor: @unchecked Sendable {
    let name: String
    let shape: [UInt64]
    let storage: QwenTextTensorStorage
    let byteCount: UInt64
    fileprivate let file: String
    fileprivate let offset: UInt64
    fileprivate let payload: QwenTextPayload

    func makeAffineBinding(device: MTLDevice) throws -> QwenMoEAffineBinding {
        guard storage == .affineInt4 || storage == .affineInt8,
              let columnsValue = shape.last,
              let columns = Int(exactly: columnsValue) else {
            throw ModelError.indexCorrupt(detail: "\(name) is not a supported affine tensor")
        }
        let elements = try logicalElementCount()
        guard elements.isMultiple(of: columns) else {
            throw ModelError.indexCorrupt(detail: "\(name) affine shape is invalid")
        }
        let rows = elements / columns
        let layout = try QwenMetalAffineLayout(
            rows: rows, columns: columns,
            bitWidth: storage == .affineInt4 ? 4 : 8)
        let data = try payload.read(offset: offset, count: byteCount, name: file)
        guard let buffer = device.makeBuffer(bytes: [UInt8](data), length: data.count,
                                             options: .storageModeShared) else {
            throw ModelError.indexCorrupt(detail: "unable to allocate affine buffer for \(name)")
        }
        buffer.label = "qwen.text.\(name)"
        let valuesBytes = rows * Int(layout.valuesRowStrideBytes)
        let metadataBytes = rows * Int(layout.metadataRowStrideBytes)
        guard valuesBytes + 2 * metadataBytes == data.count else {
            throw ModelError.indexCorrupt(detail: "\(name) affine component sizes are invalid")
        }
        return QwenMoEAffineBinding(
            values: buffer,
            scales: buffer, scalesOffset: valuesBytes,
            biases: buffer, biasesOffset: valuesBytes + metadataBytes,
            layout: layout)
    }

    func decodedFloat32() throws -> [Float] {
        let data = try payload.read(offset: offset, count: byteCount, name: file)
        let count = try logicalElementCount()
        switch storage {
        case .bf16:
            guard data.count == count * 2 else {
                throw ModelError.indexCorrupt(detail: "\(name) BF16 byte count disagrees with shape")
            }
            return stride(from: 0, to: data.count, by: 2).map { index in
                let bits = UInt16(data[index]) | UInt16(data[index + 1]) << 8
                return Float(bitPattern: UInt32(bits) << 16)
            }
        case .fp32:
            guard data.count == count * 4 else {
                throw ModelError.indexCorrupt(detail: "\(name) FP32 byte count disagrees with shape")
            }
            return stride(from: 0, to: data.count, by: 4).map { index in
                let bits = UInt32(data[index]) | UInt32(data[index + 1]) << 8
                    | UInt32(data[index + 2]) << 16 | UInt32(data[index + 3]) << 24
                return Float(bitPattern: bits)
            }
        case .affineInt4, .affineInt8:
            return try decodeAffine(data: data, elementCount: count)
        }
    }

    private func logicalElementCount() throws -> Int {
        guard !shape.isEmpty else {
            throw ModelError.indexCorrupt(detail: "\(name) has an empty shape")
        }
        var count = 1
        for extent in shape {
            guard extent > 0, extent <= UInt64(Int.max) else {
                throw ModelError.indexCorrupt(detail: "\(name) has an invalid shape")
            }
            let next = count.multipliedReportingOverflow(by: Int(extent))
            guard !next.overflow else {
                throw ModelError.indexCorrupt(detail: "\(name) element count overflows")
            }
            count = next.partialValue
        }
        return count
    }

    private func decodeAffine(data: Data, elementCount: Int) throws -> [Float] {
        let columns = Int(shape.last!)
        guard elementCount.isMultiple(of: columns) else {
            throw ModelError.indexCorrupt(detail: "\(name) affine rows disagree with shape")
        }
        let rows = elementCount / columns
        let bits = storage == .affineInt4 ? 4 : 8
        let valuesPerRow = (columns * bits + 7) / 8
        let valueBytes = rows * valuesPerRow
        let groupsPerRow = (columns + 63) / 64
        let parameterCount = rows * groupsPerRow
        let expected = valueBytes + 4 * parameterCount
        guard data.count == expected else {
            throw ModelError.indexCorrupt(
                detail: "\(name) affine byte count \(data.count) != \(expected)")
        }
        let scalesOffset = valueBytes
        let biasesOffset = valueBytes + 2 * parameterCount
        func bf16(at offset: Int) -> Float {
            let bits = UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
            return Float(bitPattern: UInt32(bits) << 16)
        }
        var result = [Float](repeating: 0, count: elementCount)
        for row in 0..<rows {
            for column in 0..<columns {
                let group = row * groupsPerRow + column / 64
                let code: UInt8
                if bits == 8 {
                    code = data[row * valuesPerRow + column]
                } else {
                    let byte = data[row * valuesPerRow + column / 2]
                    code = column.isMultiple(of: 2) ? byte & 0x0f : byte >> 4
                }
                result[row * columns + column] = Float(code)
                    * bf16(at: scalesOffset + 2 * group)
                    + bf16(at: biasesOffset + 2 * group)
            }
        }
        return result
    }
}

private final class QwenTextTemporaryDirectory: @unchecked Sendable {
    let url: URL

    init(files: [String: QwenTextFixtureRecords.VirtualFile]) throws {
        let manager = FileManager.default
        let root = manager.temporaryDirectory.appendingPathComponent(
            "turbo-fieldfare-qwen-\(UUID().uuidString)", isDirectory: true)
        do {
            try manager.createDirectory(
                at: root,
                withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700])
            for path in files.keys.sorted() {
                try QwenTextFixtureRecords.validateRelativePath(path)
                let destination = root.appendingPathComponent(path, isDirectory: false)
                try manager.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true,
                    attributes: [.posixPermissions: 0o700])
                try files[path]!.bytes.write(to: destination, options: .withoutOverwriting)
                try manager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path)
            }
            url = root
        } catch {
            try? manager.removeItem(at: root)
            throw error
        }
    }

    deinit {
        try? FileManager.default.removeItem(at: url)
    }
}

/// A validated, byte-backed Qwen text model. Tensor values are decoded only
/// from their serialized payload regions; no Float-array fixture initializer
/// exists.
public final class QwenTextModel: @unchecked Sendable {
    let architecture: QwenTextArchitecture
    let tensorNames: Set<String>
    let mappingIdentity: String

    private let payloads: [String: QwenTextPayload]
    private let regions: [String: QwenTextTensorRegion]
    private let expertLayout: PackedExpertsLayout
    private let expertDirectoryURL: URL
    private let temporaryDirectoryOwner: QwenTextTemporaryDirectory?

    private init(
        architecture: QwenTextArchitecture,
        payloads: [String: QwenTextPayload],
        fileSizes: [String: UInt64],
        regions allRegions: [QwenTextTensorRegion],
        expertLayoutData: Data,
        expertStride: UInt64,
        mappingIdentity: String,
        expertDirectoryURL: URL,
        temporaryDirectoryOwner: QwenTextTemporaryDirectory? = nil,
        expertSourceSchema: QwenPackedSourceSchema = .production
    ) throws {
        try architecture.validate()
        let resident = allRegions.filter { !$0.file.hasPrefix("packed_experts/layer_") }
        try Self.validateRegions(resident, fileSizes: fileSizes)
        guard Set(allRegions.map(\.name)).count == allRegions.count else {
            throw ModelError.indexCorrupt(detail: "duplicate Qwen tensor region name")
        }
        let configuration = try QwenPackedLayoutConfiguration(
            layerCount: architecture.layers,
            expertCount: architecture.experts,
            hiddenSize: architecture.hiddenSize,
            intermediateSize: architecture.routedIntermediateSize,
            expertStride: expertStride)
        let layout = try PackedExpertsLayoutReader.decodeQwenV2(
            data: expertLayoutData,
            configuration: configuration,
            manifestFileSizes: fileSizes,
            sourceSchema: expertSourceSchema)
        for layer in layout.layers {
            let path = "packed_experts/\(layer.file)"
            guard payloads[path] != nil else { throw ModelError.missingFile(name: path) }
        }
        let mappedRegions = Dictionary(uniqueKeysWithValues: resident.map { ($0.name, $0) })
        try Self.validateRequiredTensors(
            architecture: architecture, regions: mappedRegions)
        self.architecture = architecture
        self.payloads = payloads
        regions = mappedRegions
        tensorNames = Set(resident.map(\.name))
        expertLayout = layout
        self.mappingIdentity = mappingIdentity
        self.expertDirectoryURL = expertDirectoryURL
        self.temporaryDirectoryOwner = temporaryDirectoryOwner

        // Establish the mandatory untied entry points during model creation.
        let embedding = try tensor(named: "model.language_model.embed_tokens.weight")
        let head = try tensor(named: "lm_head.weight")
        guard embedding.name != head.name,
              embedding.shape == [UInt64(architecture.vocabularySize), UInt64(architecture.hiddenSize)],
              head.shape == embedding.shape else {
            throw ModelError.indexCorrupt(detail: "Qwen embedding and untied output head are invalid")
        }
    }

    static func loadFixtureRecords(
        _ records: QwenTextFixtureRecords,
        device: MTLDevice
    ) throws -> QwenTextModel {
        _ = device // Mapping remains CPU-addressable; execution selects its device separately.
        var payloads: [String: QwenTextPayload] = [:]
        var sizes: [String: UInt64] = [:]
        for (path, file) in records.virtualFiles {
            try QwenTextFixtureRecords.validateRelativePath(path)
            guard file.byteCount == file.bytes.count,
                  QwenTextFixtureRecords.sha256(file.bytes) == file.sha256.lowercased() else {
                throw ModelError.checksumMismatch(file: path)
            }
            payloads[path] = .memory(file.bytes)
            sizes[path] = UInt64(file.bytes.count)
        }
        let aggregate = QwenTextFixtureRecords.aggregateSHA256(files: records.virtualFiles)
        guard aggregate == records.virtualFileAggregateSHA256.lowercased(),
              aggregate == QwenTextFixtureRecords.approvedFixtureAggregateSHA256,
              records.expertLayoutData == records.virtualFiles["packed_experts/layout.json"]?.bytes else {
            throw ModelError.indexCorrupt(detail: "Qwen fixture mapping identity or layout changed")
        }
        let temporaryDirectory = try QwenTextTemporaryDirectory(files: records.virtualFiles)
        return try QwenTextModel(
            architecture: records.architecture,
            payloads: payloads,
            fileSizes: sizes,
            regions: records.residentTensorRegions + records.expertTensorRegions,
            expertLayoutData: records.expertLayoutData,
            expertStride: records.expertStride,
            mappingIdentity: aggregate,
            expertDirectoryURL: temporaryDirectory.url,
            temporaryDirectoryOwner: temporaryDirectory,
            expertSourceSchema: .fixtureRedundantEnd)
    }

    static func loadOfficial(
        directoryURL: URL,
        manifest: LoadedModelManifest,
        device: MTLDevice
    ) throws -> QwenTextModel {
        _ = device
        guard manifest.descriptor.family == .qwen3_6,
              case let .qwen3_6(configuration) = manifest.architecture else {
            throw ModelError.indexCorrupt(detail: "Qwen text mapper requires a verified Qwen v2 manifest")
        }
        let directory = try GTurboModelDirectory(rootURL: directoryURL)
        var payloads: [String: QwenTextPayload] = [:]
        for path in manifest.files.keys.sorted() {
            let entry = manifest.files[path]!
            let fd = try directory.openFile(path)
            defer { close(fd) }
            let size = try directory.fileSize(fileDescriptor: fd, relativePath: path)
            guard size == entry.size else {
                throw ModelError.tensorSizeMismatch(name: path, expected: entry.size, actual: size)
            }
            try Sha256Verifier.verifyFile(
                fileDescriptor: fd, named: path, expectedHex: entry.sha256)
            payloads[path] = .file(directory, path)
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let layoutData = try directory.readMetadata(
            "packed_experts/layout.json", maxBytes: max(
                PackedExpertsLayoutReader.defaultMaxBytes, layoutEntry.size))
        let regions = manifest.tensorRegions.map {
            QwenTextTensorRegion(
                name: $0.name, file: $0.file, offset: $0.offset,
                size: $0.size, shape: $0.shape,
                storage: QwenTextTensorStorage(rawValue: $0.storage.rawValue)!,
                quantizationCategory: $0.quantizationCategory)
        }
        return try QwenTextModel(
            architecture: QwenTextArchitecture(official: configuration),
            payloads: payloads,
            fileSizes: manifest.files.mapValues(\.size),
            regions: regions,
            expertLayoutData: layoutData,
            expertStride: manifest.expertStride,
            mappingIdentity: manifest.descriptor.textManifestSHA256,
            expertDirectoryURL: directoryURL)
    }

    func makeExpertCoordinator(
        layer: Int,
        device: MTLDevice,
        slotCount: Int
    ) throws -> QwenExpertMappingCoordinator {
        try QwenExpertMappingCoordinator(
            directoryURL: expertDirectoryURL,
            layout: expertLayout,
            layer: layer,
            device: device,
            slotCount: slotCount)
    }

    func tensor(named name: String) throws -> QwenTextTensor {
        guard let region = regions[name] else { throw ModelError.tensorNotFound(name: name) }
        guard let payload = payloads[region.file] else { throw ModelError.missingFile(name: region.file) }
        return QwenTextTensor(
            name: name, shape: region.shape, storage: region.storage,
            byteCount: region.size, file: region.file,
            offset: region.offset, payload: payload)
    }

    var embedding: QwenTextTensor {
        get throws { try tensor(named: "model.language_model.embed_tokens.weight") }
    }

    var outputHead: QwenTextTensor {
        get throws { try tensor(named: "lm_head.weight") }
    }

    var finalNorm: QwenTextTensor {
        get throws { try tensor(named: "model.language_model.norm.weight") }
    }

    func inputNorm(layer: Int) throws -> QwenTextTensor {
        try tensor(named: prefix(layer) + "input_layernorm.weight")
    }

    func postAttentionNorm(layer: Int) throws -> QwenTextTensor {
        try tensor(named: prefix(layer) + "post_attention_layernorm.weight")
    }

    func linearTensor(layer: Int, suffix: String) throws -> QwenTextTensor {
        try tensor(named: prefix(layer) + "linear_attn." + suffix)
    }

    func attentionTensor(layer: Int, suffix: String) throws -> QwenTextTensor {
        try tensor(named: prefix(layer) + "self_attn." + suffix)
    }

    func router(layer: Int) throws -> QwenTextTensor {
        try tensor(named: prefix(layer) + "mlp.gate.weight")
    }

    func sharedExpert(layer: Int, role: String) throws -> QwenTextTensor {
        let suffix: String
        switch role {
        case "gate", "up", "down": suffix = "mlp.shared_expert.\(role)_proj.weight"
        case "output_gate": suffix = "mlp.shared_expert_gate.weight"
        default: throw ModelError.tensorNotFound(name: role)
        }
        return try tensor(named: prefix(layer) + suffix)
    }

    func routedExpert(layer: Int, expert: Int, role: String) throws -> QwenTextTensor {
        guard layer >= 0, layer < expertLayout.layers.count,
              expert >= 0, expert < architecture.experts,
              let descriptor = expertLayout.layers[layer].affineDescriptors[role] else {
            throw ModelError.tensorNotFound(name: "layer \(layer) expert \(expert) \(role)")
        }
        let entry = expertLayout.expert(layer: layer, expert: expert)
        let path = "packed_experts/\(expertLayout.layers[layer].file)"
        guard let payload = payloads[path] else { throw ModelError.missingFile(name: path) }
        let base = entry.offset.addingReportingOverflow(descriptor.valuesOffset)
        let total = descriptor.valuesSize
            .addingReportingOverflow(descriptor.scalesSize)
        guard !base.overflow, !total.overflow else {
            throw ModelError.indexCorrupt(detail: "routed expert offset overflow")
        }
        let withBias = total.partialValue.addingReportingOverflow(descriptor.biasesSize)
        guard !withBias.overflow else {
            throw ModelError.indexCorrupt(detail: "routed expert size overflow")
        }
        return QwenTextTensor(
            name: "\(descriptor.sourceName)#expert.\(expert)",
            shape: descriptor.shape.map(UInt64.init), storage: .affineInt4,
            byteCount: withBias.partialValue, file: path,
            offset: base.partialValue, payload: payload)
    }

    private func prefix(_ layer: Int) -> String {
        "model.language_model.layers.\(layer)."
    }

    private static func validateRequiredTensors(
        architecture: QwenTextArchitecture,
        regions: [String: QwenTextTensorRegion]
    ) throws {
        var required: Set<String> = [
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        ]
        for layer in 0..<architecture.layers {
            let prefix = "model.language_model.layers.\(layer)."
            required.formUnion([
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "mlp.gate.weight",
                prefix + "mlp.shared_expert.gate_proj.weight",
                prefix + "mlp.shared_expert.up_proj.weight",
                prefix + "mlp.shared_expert.down_proj.weight",
                prefix + "mlp.shared_expert_gate.weight",
            ])
            switch architecture.layerKinds[layer] {
            case .linearAttention:
                for suffix in [
                    "dt_bias", "A_log", "conv1d.weight", "norm.weight",
                    "out_proj.weight", "in_proj_qkv.weight", "in_proj_z.weight",
                    "in_proj_b.weight", "in_proj_a.weight",
                ] {
                    required.insert(prefix + "linear_attn." + suffix)
                }
            case .fullAttention:
                for suffix in [
                    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                    "q_norm.weight", "k_norm.weight",
                ] {
                    required.insert(prefix + "self_attn." + suffix)
                }
            }
        }
        let missing = required.subtracting(regions.keys)
        guard missing.isEmpty else {
            throw ModelError.indexCorrupt(
                detail: "missing required Qwen tensor \(missing.sorted().first!)")
        }

        func require(
            _ name: String,
            _ shape: [UInt64],
            _ storage: QwenTextTensorStorage
        ) throws {
            guard let region = regions[name], region.shape == shape,
                  region.storage == storage else {
                throw ModelError.indexCorrupt(
                    detail: "Qwen tensor \(name) has the wrong shape or storage")
            }
        }
        let hidden = UInt64(architecture.hiddenSize)
        let vocab = UInt64(architecture.vocabularySize)
        try require("model.language_model.embed_tokens.weight", [vocab, hidden], .affineInt8)
        try require("lm_head.weight", [vocab, hidden], .affineInt8)
        try require("model.language_model.norm.weight", [hidden], .bf16)
        for layer in 0..<architecture.layers {
            let prefix = "model.language_model.layers.\(layer)."
            try require(prefix + "input_layernorm.weight", [hidden], .bf16)
            try require(prefix + "post_attention_layernorm.weight", [hidden], .bf16)
            try require(
                prefix + "mlp.gate.weight",
                [UInt64(architecture.experts), hidden], .affineInt8)
            try require(
                prefix + "mlp.shared_expert.gate_proj.weight",
                [UInt64(architecture.sharedIntermediateSize), hidden], .affineInt4)
            try require(
                prefix + "mlp.shared_expert.up_proj.weight",
                [UInt64(architecture.sharedIntermediateSize), hidden], .affineInt4)
            try require(
                prefix + "mlp.shared_expert.down_proj.weight",
                [hidden, UInt64(architecture.sharedIntermediateSize)], .affineInt4)
            try require(prefix + "mlp.shared_expert_gate.weight", [1, hidden], .affineInt8)
            switch architecture.layerKinds[layer] {
            case .linearAttention:
                let keyWidth = architecture.linearKeyHeads * architecture.linearKeyDimension
                let valueWidth = architecture.linearValueHeads * architecture.linearValueDimension
                let channels = UInt64(2 * keyWidth + valueWidth)
                try require(prefix + "linear_attn.dt_bias", [UInt64(architecture.linearValueHeads)], .affineInt8)
                try require(prefix + "linear_attn.A_log", [UInt64(architecture.linearValueHeads)], .affineInt8)
                try require(prefix + "linear_attn.conv1d.weight", [channels, 1, UInt64(architecture.convolutionWidth)], .affineInt8)
                try require(prefix + "linear_attn.norm.weight", [UInt64(architecture.linearValueDimension)], .bf16)
                try require(prefix + "linear_attn.out_proj.weight", [hidden, UInt64(valueWidth)], .affineInt8)
                try require(prefix + "linear_attn.in_proj_qkv.weight", [channels, hidden], .affineInt8)
                try require(prefix + "linear_attn.in_proj_z.weight", [UInt64(valueWidth), hidden], .affineInt8)
                try require(prefix + "linear_attn.in_proj_b.weight", [UInt64(architecture.linearValueHeads), hidden], .affineInt8)
                try require(prefix + "linear_attn.in_proj_a.weight", [UInt64(architecture.linearValueHeads), hidden], .affineInt8)
            case .fullAttention:
                let queryWidth = UInt64(
                    architecture.queryHeads * architecture.headDimension)
                let keyValueWidth = UInt64(
                    architecture.keyValueHeads * architecture.headDimension)
                try require(prefix + "self_attn.q_proj.weight", [2 * queryWidth, hidden], .affineInt4)
                try require(prefix + "self_attn.k_proj.weight", [keyValueWidth, hidden], .affineInt4)
                try require(prefix + "self_attn.v_proj.weight", [keyValueWidth, hidden], .affineInt4)
                try require(prefix + "self_attn.o_proj.weight", [hidden, queryWidth], .affineInt4)
                try require(prefix + "self_attn.q_norm.weight", [UInt64(architecture.headDimension)], .bf16)
                try require(prefix + "self_attn.k_norm.weight", [UInt64(architecture.headDimension)], .bf16)
            }
        }
    }

    private static func validateRegions(
        _ regions: [QwenTextTensorRegion],
        fileSizes: [String: UInt64]
    ) throws {
        var names = Set<String>()
        var ranges: [String: [(UInt64, UInt64)]] = [:]
        for region in regions {
            guard !region.name.isEmpty, names.insert(region.name).inserted,
                  let fileSize = fileSizes[region.file], region.size > 0,
                  !region.shape.isEmpty, region.shape.allSatisfy({ $0 > 0 }),
                  region.offset.isMultiple(of: 16_384) else {
                throw ModelError.indexCorrupt(detail: "invalid or duplicate Qwen tensor region")
            }
            let end = region.offset.addingReportingOverflow(region.size)
            guard !end.overflow, end.partialValue <= fileSize else {
                throw ModelError.indexCorrupt(detail: "Qwen tensor \(region.name) exceeds \(region.file)")
            }
            try validateByteCount(region)
            ranges[region.file, default: []].append((region.offset, end.partialValue))
        }
        for fileRanges in ranges.values {
            let sorted = fileRanges.sorted { $0.0 < $1.0 }
            for pair in zip(sorted, sorted.dropFirst()) where pair.0.1 > pair.1.0 {
                throw ModelError.indexCorrupt(detail: "overlapping Qwen tensor regions")
            }
        }
    }

    private static func validateByteCount(_ region: QwenTextTensorRegion) throws {
        var elements: UInt64 = 1
        for extent in region.shape {
            let next = elements.multipliedReportingOverflow(by: extent)
            guard !next.overflow else {
                throw ModelError.indexCorrupt(detail: "Qwen tensor shape overflows")
            }
            elements = next.partialValue
        }
        let columns = region.shape.last!
        let rows = elements / columns
        let expected: UInt64
        switch region.storage {
        case .bf16: expected = elements * 2
        case .fp32: expected = elements * 4
        case .affineInt4, .affineInt8:
            let bits: UInt64 = region.storage == .affineInt4 ? 4 : 8
            let valuesPerRow = (columns * bits + 7) / 8
            let groupsPerRow = (columns + 63) / 64
            expected = rows * valuesPerRow + rows * groupsPerRow * 4
        }
        guard region.size == expected else {
            throw ModelError.indexCorrupt(
                detail: "Qwen tensor \(region.name) byte count \(region.size) != \(expected)")
        }
    }
}
