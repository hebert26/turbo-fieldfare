import CryptoKit
import Darwin
import Foundation
import Metal
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource
@testable import TurboFieldfare

/// Shared, untrusted BF16 source fixture for Phase 16. It extends the existing
/// tiny two-layer Qwen text source with one literal 16-wide vision block, so
/// the real protected source handle and text model can coexist in one source
/// directory. It never supplies a trust receipt or claims official payload
/// bytes; source admission still goes through OfficialSourceHandle.
struct QwenOfficialSourceVisionFixture {
    static let config = QwenVisionConfig(
        depth: 1,
        hiddenSize: 16,
        intermediateSize: 32,
        numHeads: 2,
        numPositionEmbeddings: 64,
        inputChannels: 3,
        outputHiddenSize: 8,
        patchSize: 2,
        temporalPatchSize: 1,
        spatialMergeSize: 2,
        minimumProcessedPixels: 96,
        publisherMaximumProcessedPixels: 96,
        maximumPatchRows: 24,
        maximumMergedRows: 6,
        allowsFixtureGeometry: true)

    /// Acceptance tolerance is frozen before GPU observations. Every comparison
    /// also rejects non-finite actual or expected values.
    static let gpuAbsoluteTolerance: Float = 1e-5
    static let gpuRelativeTolerance: Float = 1e-5
    static let negativeControlMinimumDifference: Float = 1e-4

    /// The source-facing raw image grid is 4x6. Coordinates are independent
    /// block-major [height, width] literals, matching the serialized 24x2 IDs.
    static let rawPatchCoordinates4x6: [SIMD2<Int32>] = [
        .init(0, 0), .init(0, 1), .init(1, 0), .init(1, 1),
        .init(0, 2), .init(0, 3), .init(1, 2), .init(1, 3),
        .init(0, 4), .init(0, 5), .init(1, 4), .init(1, 5),
        .init(2, 0), .init(2, 1), .init(3, 0), .init(3, 1),
        .init(2, 2), .init(2, 3), .init(3, 2), .init(3, 3),
        .init(2, 4), .init(2, 5), .init(3, 4), .init(3, 5),
    ]

    /// Grid-relative feature rows returned by the vision tower, then the
    /// independently frozen text/image/text plan for six rows at 2..<8.
    static let expectedFeaturePositions: [[Int32]] = [
        [0, 0, 0], [0, 0, 1], [0, 0, 2],
        [0, 1, 0], [0, 1, 1], [0, 1, 2],
    ]
    static let expectedPromptPositions: [[Int32]] = [
        [0, 0, 0], [1, 1, 1],
        [2, 2, 2], [2, 2, 3], [2, 2, 4],
        [2, 3, 2], [2, 3, 3], [2, 3, 4],
        [5, 5, 5], [6, 6, 6],
    ]
    static let expectedPromptImageRange = 2..<8
    static let expectedTextRoPEDelta = -3
    static let processorProfile = GTurboQwenVisionProcessorProfileV2(
        processorClass: "Qwen3VLProcessor",
        imageProcessorType: "Qwen2VLImageProcessorFast",
        patchSize: 2,
        temporalPatchSize: 1,
        spatialMergeSize: 2)

    enum Group: CaseIterable, Hashable, Sendable {
        case patchAndPosition
        case block0
        case merger

        /// Independent fixture manifest. Do not derive these names from the
        /// production classifier or runtime's tensor contract.
        var tensorNames: [String] {
            switch self {
            case .patchAndPosition:
                [
                    "model.visual.patch_embed.proj.weight",
                    "model.visual.patch_embed.proj.bias",
                    "model.visual.pos_embed.weight",
                ]
            case .block0:
                [
                    "model.visual.blocks.0.norm1.weight",
                    "model.visual.blocks.0.norm1.bias",
                    "model.visual.blocks.0.attn.qkv.weight",
                    "model.visual.blocks.0.attn.qkv.bias",
                    "model.visual.blocks.0.attn.proj.weight",
                    "model.visual.blocks.0.attn.proj.bias",
                    "model.visual.blocks.0.norm2.weight",
                    "model.visual.blocks.0.norm2.bias",
                    "model.visual.blocks.0.mlp.linear_fc1.weight",
                    "model.visual.blocks.0.mlp.linear_fc1.bias",
                    "model.visual.blocks.0.mlp.linear_fc2.weight",
                    "model.visual.blocks.0.mlp.linear_fc2.bias",
                ]
            case .merger:
                [
                    "model.visual.merger.norm.weight",
                    "model.visual.merger.norm.bias",
                    "model.visual.merger.linear_fc1.weight",
                    "model.visual.merger.linear_fc1.bias",
                    "model.visual.merger.linear_fc2.weight",
                    "model.visual.merger.linear_fc2.bias",
                ]
            }
        }
    }

    struct Tensor: Equatable, Sendable {
        let name: String
        let shape: [Int]
        let words: [UInt16]

        var bytes: Data { Self.littleEndianBytes(words) }

        fileprivate init(name: String, shape: [Int], words: [UInt16]) {
            precondition(!name.isEmpty && !shape.isEmpty && shape.allSatisfy { $0 > 0 })
            precondition(shape.reduce(1, *) == words.count)
            self.name = name
            self.shape = shape
            self.words = words
        }

        private static func littleEndianBytes(_ words: [UInt16]) -> Data {
            var result = Data(capacity: words.count * MemoryLayout<UInt16>.stride)
            for word in words {
                result.append(UInt8(truncatingIfNeeded: word))
                result.append(UInt8(truncatingIfNeeded: word >> 8))
            }
            return result
        }
    }

    struct ImageInput: Sendable {
        /// P3's frozen 24x12 tiny 4x6 raw patch input, rounded locally to BF16.
        let patchWords: [UInt16]
        let positions: [SIMD2<Int32>]
    }

    struct CPUReference: Sendable {
        let patchAndPositionRows: [Float]
        let blockRows: [Float]
        let mergedRows: [Float]
        let featureRows: [Float]
        let featurePositions: [[Int32]]
    }

    struct Source {
        let textSource: QwenBF16TextRunnerFixture.Source
        let visionTensors: [Tensor]
        let imageInput: ImageInput
        let shardName: String
        let processorConfiguration: Data
        let processorConfigurationSHA256: String
        /// Synthetic content binding for the tiny text fixture only. This is
        /// not an OfficialSourceTrustReceipt or a production source identity.
        let untrustedTextFixtureSHA256: String

        var registrationURL: URL { textSource.registrationURL }
        var sourceRoot: URL { textSource.sourceRoot }

        func remove() { textSource.remove() }

        func openProtectedHandle() throws -> OfficialSourceHandle {
            try OfficialSourceHandle(registrationURL: registrationURL)
        }

        func loadTinyTextModel(context: MetalContext) throws -> QwenOfficialSourceModel {
            try QwenOfficialSourceModel.loadSyntheticFixture(
                registrationURL: registrationURL,
                context: context,
                residencyBudgetBytes: textSource.totalResidencyBudget())
        }

        func tensor(named name: String) throws -> Tensor {
            guard let tensor = visionTensors.first(where: { $0.name == name }) else {
                throw FixtureError.missingTensor(name)
            }
            return tensor
        }

        /// Literal bytes for one named tensor. Group tests independently
        /// calculate 256-byte offsets and compare these named ranges only.
        func expectedTensorBytes(_ name: String) throws -> Data {
            try tensor(named: name).bytes
        }

        func makePixelBuffer(
            device: MTLDevice,
            patchWords: [UInt16]? = nil,
            positions: [SIMD2<Int32>]? = nil
        ) throws -> QwenVisionPixelBuffer {
            try Self.makePixelBuffer(
                device: device,
                patchWords: patchWords ?? imageInput.patchWords,
                positions: positions ?? imageInput.positions)
        }

        /// Builds non-Sendable Metal buffers inside the caller's isolation domain.
        static func makePixelBuffer(
            device: MTLDevice,
            patchWords: [UInt16],
            positions: [SIMD2<Int32>]
        ) throws -> QwenVisionPixelBuffer {
            guard patchWords.count == 24 * 12, positions.count == 24 else {
                throw FixtureError.invalidImageInput
            }
            let geometry = try QwenImageGeometry(
                sourceWidth: 12, sourceHeight: 8, config: QwenOfficialSourceVisionFixture.config)
            guard geometry.gridT == 1, geometry.gridH == 4, geometry.gridW == 6,
                  geometry.patchRows == 24, geometry.mergedRows == 6 else {
                throw FixtureError.invalidImageInput
            }
            let patchBytes = Self.bf16Bytes(patchWords)
            let positionBytes = Self.positionBytes(positions)
            guard let patches = device.makeBuffer(
                length: patchBytes.count, options: .storageModeShared),
                  let positionBuffer = device.makeBuffer(
                    length: positionBytes.count, options: .storageModeShared) else {
                throw QwenVisionError.allocationFailed(name: "tiny source vision input")
            }
            patchBytes.withUnsafeBytes { source in
                if let base = source.baseAddress {
                    patches.contents().copyMemory(from: base, byteCount: source.count)
                }
            }
            positionBytes.withUnsafeBytes { source in
                if let base = source.baseAddress {
                    positionBuffer.contents().copyMemory(from: base, byteCount: source.count)
                }
            }
            let metadata = VisionImageMetadata(
                encodedBytes: 1, encodedWidth: 12, encodedHeight: 8,
                orientedWidth: 12, orientedHeight: 8, orientation: 1,
                bitsPerComponent: 8, colorModel: "RGB", typeIdentifier: "tiny-fixture")
            return QwenVisionPixelBuffer(
                patchesBF16: patches,
                positionsInt32x2: positionBuffer,
                metadata: metadata,
                geometry: geometry,
                imageDigest: String(repeating: "a", count: 64),
                wallNanoseconds: 0,
                allocatedBytes: patchBytes.count + positionBytes.count)
        }

        /// Independent Float32 CPU reference for patch+position, one tiny
        /// transformer block, and four-patch merger packing. The implementation
        /// uses only fixture BF16 words and scalar loops; it does not call any
        /// production vision numerical, grouping, or position-planning helper.
        func cpuReference(
            patchWords: [UInt16]? = nil,
            positions: [SIMD2<Int32>]? = nil,
            tensorWordOverrides: [String: [UInt16]] = [:]
        ) throws -> CPUReference {
            try QwenOfficialSourceVisionFixture.cpuReference(
                tensors: visionTensors,
                imageInput: imageInput,
                patchWords: patchWords ?? imageInput.patchWords,
                positions: positions ?? imageInput.positions,
                tensorWordOverrides: tensorWordOverrides)
        }

        func patchWords(swappingRows first: Int, and second: Int) throws -> [UInt16] {
            guard (0..<24).contains(first), (0..<24).contains(second) else {
                throw FixtureError.invalidImageInput
            }
            var result = imageInput.patchWords
            for column in 0..<12 {
                result.swapAt(first * 12 + column, second * 12 + column)
            }
            return result
        }

        private static func bf16Bytes(_ words: [UInt16]) -> Data {
            var bytes = Data(capacity: words.count * MemoryLayout<UInt16>.stride)
            for word in words {
                bytes.append(UInt8(truncatingIfNeeded: word))
                bytes.append(UInt8(truncatingIfNeeded: word >> 8))
            }
            return bytes
        }

        private static func positionBytes(_ positions: [SIMD2<Int32>]) -> Data {
            var bytes = Data(capacity: positions.count * 2 * MemoryLayout<Int32>.stride)
            for position in positions {
                for value in [position.x, position.y] {
                    var littleEndian = value.littleEndian
                    withUnsafeBytes(of: &littleEndian) { bytes.append(contentsOf: $0) }
                }
            }
            return bytes
        }
    }

    enum FixtureError: Error, Equatable {
        case missingVisionFixture
        case malformedVisionFixture(String)
        case missingTensor(String)
        case invalidTensorOverride(String)
        case invalidImageInput
        case sourceFile(String)
    }

    static func make(
        tensorWordOverrides: [String: [UInt16]] = [:],
        tensorHeaderShapeOverrides: [String: [Int]] = [:]
    ) throws -> Source {
        let textSource = try QwenBF16TextRunnerFixture.make()
        do {
            let imageInput = try loadTiny4x6ImageInput()
            let textBinding = try untrustedTextBinding(textSource)
            let tensors = try makeVisionTensors(overrides: tensorWordOverrides)
            let shardName = try textSource.shardNames.first
                .unwrap(or: FixtureError.sourceFile("text fixture has no shard"))
            let processor = try tinyProcessorConfiguration()
            try installVisionTensors(
                tensors, processorConfiguration: processor,
                headerShapeOverrides: tensorHeaderShapeOverrides, into: textSource)
            return Source(
                textSource: textSource,
                visionTensors: tensors,
                imageInput: imageInput,
                shardName: shardName,
                processorConfiguration: processor,
                processorConfigurationSHA256: sha256Hex(processor),
                untrustedTextFixtureSHA256: textBinding)
        } catch {
            textSource.remove()
            throw error
        }
    }

    static func matchesFrozenTolerance(_ actual: [Float], _ expected: [Float]) -> Bool {
        guard actual.count == expected.count,
              actual.allSatisfy(\.isFinite), expected.allSatisfy(\.isFinite) else {
            return false
        }
        return zip(actual, expected).allSatisfy { actualValue, expectedValue in
            abs(actualValue - expectedValue)
                <= gpuAbsoluteTolerance + gpuRelativeTolerance * abs(expectedValue)
        }
    }

    static func maximumAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        guard lhs.allSatisfy(\.isFinite), rhs.allSatisfy(\.isFinite) else { return .infinity }
        return zip(lhs, rhs).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
    }

    private static func makeVisionTensors(
        overrides: [String: [UInt16]]
    ) throws -> [Tensor] {
        let matrixPattern: [UInt16] = [
            0x3D80, 0xBD00, 0x3E00, 0xBE00,
            0x3D00, 0xBD80, 0x3C80, 0xBC80,
        ]
        let biasPattern: [UInt16] = [
            0x3C80, 0xBC00, 0x3D00, 0xBD80,
            0x3B80, 0xBB80, 0x3C00, 0xBC80,
        ]
        let normPattern: [UInt16] = [0x3F80, 0x3F88, 0x3F70, 0x3F90]
        var result: [Tensor] = []

        func append(
            _ name: String,
            _ shape: [Int],
            _ pattern: [UInt16],
            phase: Int
        ) throws {
            let count = shape.reduce(1, *)
            let generated = (0..<count).map { index in
                pattern[(index * 5 + phase * 3 + index / 7) % pattern.count]
            }
            let words = overrides[name] ?? generated
            guard words.count == count else { throw FixtureError.invalidTensorOverride(name) }
            result.append(Tensor(name: name, shape: shape, words: words))
        }

        // Source-file order and group membership are fixed here, not inferred
        // from QwenVisionConfig.tensorContract or a production classifier.
        try append("model.visual.patch_embed.proj.weight", [16, 3, 1, 2, 2],
                   matrixPattern, phase: 1)
        try append("model.visual.patch_embed.proj.bias", [16],
                   biasPattern, phase: 2)
        try append("model.visual.pos_embed.weight", [64, 16],
                   matrixPattern, phase: 3)

        try append("model.visual.blocks.0.norm1.weight", [16],
                   normPattern, phase: 1)
        try append("model.visual.blocks.0.norm1.bias", [16],
                   biasPattern, phase: 4)
        try append("model.visual.blocks.0.attn.qkv.weight", [48, 16],
                   matrixPattern, phase: 5)
        try append("model.visual.blocks.0.attn.qkv.bias", [48],
                   biasPattern, phase: 6)
        try append("model.visual.blocks.0.attn.proj.weight", [16, 16],
                   matrixPattern, phase: 7)
        try append("model.visual.blocks.0.attn.proj.bias", [16],
                   biasPattern, phase: 8)
        try append("model.visual.blocks.0.norm2.weight", [16],
                   normPattern, phase: 2)
        try append("model.visual.blocks.0.norm2.bias", [16],
                   biasPattern, phase: 9)
        try append("model.visual.blocks.0.mlp.linear_fc1.weight", [32, 16],
                   matrixPattern, phase: 10)
        try append("model.visual.blocks.0.mlp.linear_fc1.bias", [32],
                   biasPattern, phase: 11)
        try append("model.visual.blocks.0.mlp.linear_fc2.weight", [16, 32],
                   matrixPattern, phase: 12)
        try append("model.visual.blocks.0.mlp.linear_fc2.bias", [16],
                   biasPattern, phase: 13)

        try append("model.visual.merger.norm.weight", [16],
                   normPattern, phase: 3)
        try append("model.visual.merger.norm.bias", [16],
                   biasPattern, phase: 14)
        try append("model.visual.merger.linear_fc1.weight", [64, 64],
                   matrixPattern, phase: 15)
        try append("model.visual.merger.linear_fc1.bias", [64],
                   biasPattern, phase: 16)
        try append("model.visual.merger.linear_fc2.weight", [8, 64],
                   matrixPattern, phase: 17)
        try append("model.visual.merger.linear_fc2.bias", [8],
                   biasPattern, phase: 18)

        let known = Set(Group.allCases.flatMap(\.tensorNames))
        guard result.count == known.count,
              Set(result.map(\.name)) == known,
              Set(overrides.keys).isSubset(of: known) else {
            throw FixtureError.malformedVisionFixture("vision tensor inventory mismatch")
        }
        return result
    }

    private static func loadTiny4x6ImageInput() throws -> ImageInput {
        guard let url = Bundle.module.url(
            forResource: "qwen36-tiny-fixtures", withExtension: "json") else {
            throw FixtureError.missingVisionFixture
        }
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let vision = root["vision"] as? [String: Any],
              let raw = vision["rawPatchInput"] as? [String: Any],
              raw["shape"] as? [Int] == [24, 12],
              let rawValues = raw["values"] as? [NSNumber],
              rawValues.count == 24 * 12,
              let positionsTensor = vision["visionPositionIDs"] as? [String: Any],
              positionsTensor["shape"] as? [Int] == [24, 2],
              let positionValues = positionsTensor["values"] as? [NSNumber],
              positionValues.count == 24 * 2 else {
            throw FixtureError.malformedVisionFixture("expected frozen 24x12 and 24x2 inputs")
        }
        let rawFloats = rawValues.map(\.floatValue)
        guard rawFloats.allSatisfy(\.isFinite) else {
            throw FixtureError.malformedVisionFixture("non-finite raw patch input")
        }
        let positions = stride(from: 0, to: positionValues.count, by: 2).map { index in
            SIMD2<Int32>(positionValues[index].int32Value, positionValues[index + 1].int32Value)
        }
        guard positions == rawPatchCoordinates4x6 else {
            throw FixtureError.malformedVisionFixture("frozen patch positions changed")
        }
        return ImageInput(
            patchWords: rawFloats.map(localBF16RoundToNearestEven),
            positions: positions)
    }

    /// Local IEEE BF16 conversion for fixture construction only. Expected CPU
    /// values are decoded from the exact emitted words, never from this input
    /// Float array or a production quantization routine.
    private static func localBF16RoundToNearestEven(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let high = UInt16(truncatingIfNeeded: bits >> 16)
        let tieToEven = UInt32(high & 1)
        return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ tieToEven) >> 16)
    }

    private static func installVisionTensors(
        _ visionTensors: [Tensor],
        processorConfiguration: Data,
        headerShapeOverrides: [String: [Int]],
        into textSource: QwenBF16TextRunnerFixture.Source
    ) throws {
        guard let visionShard = textSource.shardNames.first else {
            throw FixtureError.sourceFile("text fixture has no first shard")
        }
        var mapping = textSource.tensorToShard
        for tensor in visionTensors { mapping[tensor.name] = visionShard }

        for shard in textSource.shardNames {
            let existing = textSource.tensors.values
                .filter { textSource.tensorToShard[$0.name] == shard }
                .sorted { $0.name < $1.name }
                .map { SerializedTensor(name: $0.name, shape: $0.shape, words: $0.words) }
            let additional = shard == visionShard
                ? visionTensors.map {
                    SerializedTensor(
                        name: $0.name,
                        shape: headerShapeOverrides[$0.name] ?? $0.shape,
                        words: $0.words)
                }
                : []
            let bytes = try safetensorsFile(existing + additional)
            try bytes.write(
                to: textSource.sourceRoot.appendingPathComponent(shard), options: .atomic)
        }
        let index = try JSONSerialization.data(
            withJSONObject: ["weight_map": mapping], options: [.sortedKeys])
        try index.write(
            to: textSource.sourceRoot.appendingPathComponent("model.safetensors.index.json"),
            options: .atomic)
        try processorConfiguration.write(
            to: textSource.sourceRoot.appendingPathComponent("preprocessor_config.json"),
            options: .atomic)
    }

    private struct SerializedTensor {
        let name: String
        let shape: [Int]
        let words: [UInt16]
    }

    private static func safetensorsFile(_ tensors: [SerializedTensor]) throws -> Data {
        var payload = Data()
        var entries: [String: [String: Any]] = [:]
        for tensor in tensors {
            guard tensor.shape.reduce(1, *) == tensor.words.count,
                  entries[tensor.name] == nil else {
                throw FixtureError.sourceFile("invalid or duplicate tensor \(tensor.name)")
            }
            let start = payload.count
            for word in tensor.words {
                payload.append(UInt8(truncatingIfNeeded: word))
                payload.append(UInt8(truncatingIfNeeded: word >> 8))
            }
            entries[tensor.name] = [
                "dtype": "BF16",
                "shape": tensor.shape,
                "data_offsets": [start, payload.count],
            ]
        }
        var header = try JSONSerialization.data(withJSONObject: entries, options: [.sortedKeys])
        header.append(contentsOf: repeatElement(UInt8(0x20), count: (8 - header.count % 8) % 8))
        var littleEndianHeaderLength = UInt64(header.count).littleEndian
        var result = Data()
        withUnsafeBytes(of: &littleEndianHeaderLength) {
            result.append(contentsOf: $0)
        }
        result.append(header)
        result.append(payload)
        return result
    }

    private static func untrustedTextBinding(
        _ source: QwenBF16TextRunnerFixture.Source
    ) throws -> String {
        var bytes = Data()
        for name in ["config.json", "model.safetensors.index.json"] {
            bytes.append(contentsOf: name.utf8)
            bytes.append(0)
            bytes.append(try Data(contentsOf: source.sourceRoot.appendingPathComponent(name)))
        }
        for name in source.shardNames.sorted() {
            bytes.append(contentsOf: name.utf8)
            bytes.append(0)
            bytes.append(try Data(contentsOf: source.sourceRoot.appendingPathComponent(name)))
        }
        return sha256Hex(bytes)
    }

    private static func tinyProcessorConfiguration() throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "processor_class": "Qwen3VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "patch_size": 2,
            "temporal_patch_size": 1,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        ], options: [.sortedKeys])
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func cpuReference(
        tensors: [Tensor],
        imageInput: ImageInput,
        patchWords: [UInt16],
        positions: [SIMD2<Int32>],
        tensorWordOverrides: [String: [UInt16]]
    ) throws -> CPUReference {
        let rows = 24
        let hidden = 16
        let intermediate = 32
        let heads = 2
        let headDimension = 8
        let patchWidth = 12
        let outputHidden = 8
        guard patchWords.count == rows * patchWidth,
              positions.count == rows,
              positions.allSatisfy({ $0.x >= 0 && $0.x < 4 && $0.y >= 0 && $0.y < 6 }) else {
            throw FixtureError.invalidImageInput
        }
        let tensorMap = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0) })
        func weight(_ name: String) throws -> [Float] {
            let bits: [UInt16]
            if let replacement = tensorWordOverrides[name] {
                guard let tensor = tensorMap[name], replacement.count == tensor.words.count else {
                    throw FixtureError.invalidTensorOverride(name)
                }
                bits = replacement
            } else {
                guard let tensor = tensorMap[name] else { throw FixtureError.missingTensor(name) }
                bits = tensor.words
            }
            return bits.map(floatFromBF16)
        }
        let patches = patchWords.map(floatFromBF16)

        // Independent patch projection plus bilinear position-table sampling.
        let patchWeight = try weight("model.visual.patch_embed.proj.weight")
        let patchBias = try weight("model.visual.patch_embed.proj.bias")
        let positionTable = try weight("model.visual.pos_embed.weight")
        var embedded = [Float](repeating: 0, count: rows * hidden)
        for row in 0..<rows {
            let y = Float(positions[row].x) * 7 / 3
            let x = Float(positions[row].y) * 7 / 5
            let y0 = Int(floor(y)), x0 = Int(floor(x))
            let y1 = min(y0 + 1, 7), x1 = min(x0 + 1, 7)
            let fy = y - Float(y0), fx = x - Float(x0)
            for column in 0..<hidden {
                var projected = patchBias[column]
                for inner in 0..<patchWidth {
                    projected += patches[row * patchWidth + inner]
                        * patchWeight[column * patchWidth + inner]
                }
                let topLeft = positionTable[(y0 * 8 + x0) * hidden + column]
                let topRight = positionTable[(y0 * 8 + x1) * hidden + column]
                let bottomLeft = positionTable[(y1 * 8 + x0) * hidden + column]
                let bottomRight = positionTable[(y1 * 8 + x1) * hidden + column]
                let top = topLeft * (1 - fx) + topRight * fx
                let bottom = bottomLeft * (1 - fx) + bottomRight * fx
                embedded[row * hidden + column] = projected + top * (1 - fy) + bottom * fy
            }
        }

        let prefix = "model.visual.blocks.0."
        let norm1 = try layerNorm(
            embedded, rows: rows, width: hidden,
            gamma: weight(prefix + "norm1.weight"),
            beta: weight(prefix + "norm1.bias"))
        let qkv = try denseRows(
            norm1, rows: rows, inputWidth: hidden, outputWidth: 3 * hidden,
            weight: weight(prefix + "attn.qkv.weight"),
            bias: weight(prefix + "attn.qkv.bias"))
        let rotated = rotateQueriesAndKeys(
            qkv, positions: positions, hidden: hidden,
            heads: heads, headDimension: headDimension)
        let attended = fullSelfAttention(
            rotated, rows: rows, hidden: hidden,
            heads: heads, headDimension: headDimension)
        let attentionProjection = try denseRows(
            attended, rows: rows, inputWidth: hidden, outputWidth: hidden,
            weight: weight(prefix + "attn.proj.weight"),
            bias: weight(prefix + "attn.proj.bias"))
        let afterAttention = zip(embedded, attentionProjection).map(+)
        let norm2 = try layerNorm(
            afterAttention, rows: rows, width: hidden,
            gamma: weight(prefix + "norm2.weight"),
            beta: weight(prefix + "norm2.bias"))
        let mlpFirst = try denseRows(
            norm2, rows: rows, inputWidth: hidden, outputWidth: intermediate,
            weight: weight(prefix + "mlp.linear_fc1.weight"),
            bias: weight(prefix + "mlp.linear_fc1.bias"))
        let activated = mlpFirst.map(geluTanh)
        let mlpSecond = try denseRows(
            activated, rows: rows, inputWidth: intermediate, outputWidth: hidden,
            weight: weight(prefix + "mlp.linear_fc2.weight"),
            bias: weight(prefix + "mlp.linear_fc2.bias"))
        let blockRows = zip(afterAttention, mlpSecond).map(+)

        // The source image uses block-major 2x2 groups: four adjacent rows
        // concatenate before the tiny merger projections.
        let mergerNorm = try layerNorm(
            blockRows, rows: rows, width: hidden,
            gamma: weight("model.visual.merger.norm.weight"),
            beta: weight("model.visual.merger.norm.bias"))
        var packed = [Float](repeating: 0, count: 6 * 4 * hidden)
        for mergedRow in 0..<6 {
            for patchInGroup in 0..<4 {
                for column in 0..<hidden {
                    packed[mergedRow * 4 * hidden + patchInGroup * hidden + column]
                        = mergerNorm[(mergedRow * 4 + patchInGroup) * hidden + column]
                }
            }
        }
        let mergedWidth = 4 * hidden
        let mergerFirst = try denseRows(
            packed, rows: 6, inputWidth: mergedWidth, outputWidth: mergedWidth,
            weight: weight("model.visual.merger.linear_fc1.weight"),
            bias: weight("model.visual.merger.linear_fc1.bias"))
        let exactGelu = mergerFirst.map(geluExact)
        let output = try denseRows(
            exactGelu, rows: 6, inputWidth: mergedWidth, outputWidth: outputHidden,
            weight: weight("model.visual.merger.linear_fc2.weight"),
            bias: weight("model.visual.merger.linear_fc2.bias"))
        return CPUReference(
            patchAndPositionRows: embedded,
            blockRows: blockRows,
            mergedRows: packed,
            featureRows: output,
            featurePositions: expectedFeaturePositions)
    }

    private static func layerNorm(
        _ values: [Float], rows: Int, width: Int,
        gamma: [Float], beta: [Float]
    ) throws -> [Float] {
        guard values.count == rows * width, gamma.count == width, beta.count == width else {
            throw FixtureError.malformedVisionFixture("CPU layer norm shape")
        }
        var result = [Float](repeating: 0, count: values.count)
        for row in 0..<rows {
            let base = row * width
            var mean: Float = 0
            for column in 0..<width { mean += values[base + column] }
            mean /= Float(width)
            var variance: Float = 0
            for column in 0..<width {
                let centered = values[base + column] - mean
                variance += centered * centered
            }
            let inverse = 1 / Float(Foundation.sqrt(Double(variance / Float(width) + 1e-6)))
            for column in 0..<width {
                result[base + column] = (values[base + column] - mean) * inverse
                    * gamma[column] + beta[column]
            }
        }
        return result
    }

    private static func denseRows(
        _ input: [Float], rows: Int, inputWidth: Int, outputWidth: Int,
        weight: [Float], bias: [Float]
    ) throws -> [Float] {
        guard input.count == rows * inputWidth,
              weight.count == outputWidth * inputWidth,
              bias.count == outputWidth else {
            throw FixtureError.malformedVisionFixture("CPU dense shape")
        }
        var output = [Float](repeating: 0, count: rows * outputWidth)
        for row in 0..<rows {
            for column in 0..<outputWidth {
                var sum = bias[column]
                for inner in 0..<inputWidth {
                    sum += input[row * inputWidth + inner]
                        * weight[column * inputWidth + inner]
                }
                output[row * outputWidth + column] = sum
            }
        }
        return output
    }

    private static func rotateQueriesAndKeys(
        _ qkv: [Float], positions: [SIMD2<Int32>], hidden: Int,
        heads: Int, headDimension: Int
    ) -> [Float] {
        var result = qkv
        let rotaryHalf = headDimension / 2
        let axisHalf = rotaryHalf / 2
        for row in 0..<positions.count {
            for projection in 0..<2 {
                for head in 0..<heads {
                    let base = row * 3 * hidden + projection * hidden + head * headDimension
                    for dimension in 0..<rotaryHalf {
                        let axis = dimension < axisHalf ? 0 : 1
                        let frequency = dimension % axisHalf
                        let coordinate = Float(axis == 0
                            ? positions[row].x : positions[row].y)
                        let exponent = -2 * Float(frequency) / Float(rotaryHalf)
                        let inverseFrequency = Float(Foundation.pow(10_000, Double(exponent)))
                        let angle = coordinate * inverseFrequency
                        let cosine = Float(Foundation.cos(Double(angle)))
                        let sine = Float(Foundation.sin(Double(angle)))
                        let first = qkv[base + dimension]
                        let second = qkv[base + dimension + rotaryHalf]
                        result[base + dimension] = first * cosine - second * sine
                        result[base + dimension + rotaryHalf] = first * sine + second * cosine
                    }
                }
            }
        }
        return result
    }

    private static func fullSelfAttention(
        _ qkv: [Float], rows: Int, hidden: Int,
        heads: Int, headDimension: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows * hidden)
        let scale = 1 / Float(Foundation.sqrt(Double(headDimension)))
        for row in 0..<rows {
            for head in 0..<heads {
                let queryBase = row * 3 * hidden + head * headDimension
                var scores = [Float](repeating: 0, count: rows)
                for keyRow in 0..<rows {
                    let keyBase = keyRow * 3 * hidden + hidden + head * headDimension
                    var score: Float = 0
                    for dimension in 0..<headDimension {
                        score += qkv[queryBase + dimension] * qkv[keyBase + dimension]
                    }
                    scores[keyRow] = score * scale
                }
                let maximum = scores.max() ?? 0
                let exponentials = scores.map {
                    Float(Foundation.exp(Double($0 - maximum)))
                }
                let denominator = exponentials.reduce(Float.zero, +)
                for dimension in 0..<headDimension {
                    var sum: Float = 0
                    for keyRow in 0..<rows {
                        let valueBase = keyRow * 3 * hidden + 2 * hidden
                            + head * headDimension
                        sum += exponentials[keyRow] / denominator
                            * qkv[valueBase + dimension]
                    }
                    output[row * hidden + head * headDimension + dimension] = sum
                }
            }
        }
        return output
    }

    private static func geluTanh(_ value: Float) -> Float {
        let coefficient: Float = 0.7978845608028654
        let cubic: Float = 0.044715
        return 0.5 * value * (1 + Float(Foundation.tanh(
            Double(coefficient * (value + cubic * value * value * value)))))
    }

    private static func geluExact(_ value: Float) -> Float {
        0.5 * value * (1 + Float(Darwin.erf(Double(value) * 0.7071067811865475)))
    }

    private static func floatFromBF16(_ word: UInt16) -> Float {
        Float(bitPattern: UInt32(word) << 16)
    }
}

/// Explicit async barrier for later cancellation and source-replacement tests.
/// Tests enter/release it directly; no timing sleeps are needed.
actor QwenOfficialSourceVisionTestGate {
    private var entered = false
    private var released = false
    private var producerFinished = false
    private var entryWaiters: [(UUID, CheckedContinuation<Bool, Never>)] = []
    private var releaseWaiter: CheckedContinuation<Void, Never>?

    func suspend() async {
        entered = true
        resumeEntryWaiters(returning: true)
        guard !released else { return }
        await withCheckedContinuation { continuation in
            if released { continuation.resume() }
            else { releaseWaiter = continuation }
        }
    }

    func waitUntilEntered(timeoutNanoseconds: UInt64 = 10_000_000_000) async -> Bool {
        guard !entered else { return true }
        guard !producerFinished else { return false }
        let id = UUID()
        return await withCheckedContinuation { continuation in
            if entered { continuation.resume(returning: true) }
            else if producerFinished { continuation.resume(returning: false) }
            else {
                entryWaiters.append((id, continuation))
                Task {
                    try? await Task.sleep(nanoseconds: timeoutNanoseconds)
                    self.expireEntryWaiter(id)
                }
            }
        }
    }

    func producerDidFinish() {
        producerFinished = true
        resumeEntryWaiters(returning: false)
    }

    private func resumeEntryWaiters(returning value: Bool) {
        let waiters = entryWaiters
        entryWaiters.removeAll(keepingCapacity: false)
        for (_, waiter) in waiters { waiter.resume(returning: value) }
    }

    private func expireEntryWaiter(_ id: UUID) {
        guard let index = entryWaiters.firstIndex(where: { $0.0 == id }) else { return }
        let (_, waiter) = entryWaiters.remove(at: index)
        waiter.resume(returning: false)
    }

    func release() {
        released = true
        let waiter = releaseWaiter
        releaseWaiter = nil
        waiter?.resume()
    }
}

private extension Optional {
    func unwrap<E: Error>(or error: E) throws -> Wrapped {
        guard let self else { throw error }
        return self
    }
}
