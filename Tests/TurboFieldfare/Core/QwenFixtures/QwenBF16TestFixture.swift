import Darwin
import Foundation
import TurboFieldfareFormat
@testable import TurboFieldfareOfficialQwenSource

/// Independent test-owned BF16 literals and CPU arithmetic for Phase 8.
/// No values are imported from a QwenBF16Weights or Metal result.
enum QwenBF16TestFixture {
    static let rows = 5
    static let columns = 5

    /// Row zero is the precision discriminator. Rows one and three contain
    /// only positive values; row two only negative values; row four is an
    /// exact-zero control. All stored values are literal BF16 encodings.
    static let matrixBits: [UInt16] = [
        0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x0000,
        0x3f00, 0x3e80, 0x3f00, 0x3e80, 0x3e00,
        0xbf80, 0xbf00, 0xbe80, 0xbe00, 0xbd80,
        0x4000, 0x3fc0, 0x3f00, 0x3e80, 0x3e00,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    ]

    /// Nine-by-five router matrix exercises non-multiple row counts while
    /// keeping each reference dot well-conditioned.
    static let routerBits: [UInt16] = matrixBits + [
        0x3e80, 0x3f00, 0x3f40, 0x3f80, 0x3fa0,
        0xbe80, 0xbf00, 0xbf40, 0xbf80, 0xbfa0,
        0x4000, 0x4000, 0x4000, 0x4000, 0x4000,
        0xc000, 0xc000, 0xc000, 0xc000, 0xc000,
    ]

    /// Token zero sums sequentially to exactly 1.000732421875 in FP32.
    /// At 1, Float16 spacing is 2^-10; this value is above the midpoint
    /// 1 + 2^-11, so nearest Float16 is 1.0009765625 (bits 0x3c01),
    /// 0.000244140625 away. The frozen tolerance here is about 1.101e-6.
    /// BF16 rounding after any partial sum also returns 1.0 because the
    /// entire increment is below half a BF16 ULP at 1.
    static let inputs: [Float] = [
        1.0, 0.000244140625, 0.000244140625, 0.000244140625, 0.0,
        0.5, 0.25, 0.125, 0.0625, 0.03125,
        -1.0, 0.0, 0.0, 0.0, 0.0,
    ]

    static let embeddingTokenIDs: [UInt32] = [0, 2, 4]
    static let nanBits: UInt16 = 0x7fc1
    static let sharedGateBits: [UInt16] = [
        0x3f00, 0x3e80, 0x3e00, 0x3d80, 0x3d00,
        0x3e80, 0x3f00, 0x3e00, 0x3e80, 0x3f00,
        0x3e00, 0x3e80, 0x3f00, 0x3e80, 0x3e00,
        0x3f00, 0x3f00, 0x3e80, 0x3e00, 0x3d80,
        0x3e80, 0x3e00, 0x3e80, 0x3f00, 0x3e80,
    ]
    static let sharedUpBits: [UInt16] = [
        0x3e80, 0x3e00, 0x3f00, 0x3e80, 0x3e00,
        0x3f00, 0x3e80, 0x3e80, 0x3e00, 0x3f00,
        0x3e80, 0x3e80, 0x3e00, 0x3f00, 0x3e80,
        0x3e00, 0x3f00, 0x3e80, 0x3e80, 0x3e00,
        0x3f00, 0x3e80, 0x3e00, 0x3f00, 0x3e80,
    ]
    static let sharedDownBits: [UInt16] = [
        0x3e00, 0x3e80, 0x3d80, 0x3e00, 0x3e80,
        0x3e80, 0x3e00, 0x3e80, 0x3d80, 0x3e00,
        0x3d80, 0x3e80, 0x3e00, 0x3e80, 0x3d80,
        0x3e00, 0x3d80, 0x3e80, 0x3e00, 0x3e80,
        0x3e80, 0x3e00, 0x3d80, 0x3e80, 0x3e00,
    ]
    static let sharedOutputGateBits: [UInt16] = [
        0x3f00, 0x3e80, 0x3e00, 0x3d80, 0x3d00,
    ]
    static let sharedInput: [Float] = [0.125, 0.25, 0.5, 0.75, 1.0]
    static let sharedInitialOutput: [Float] = [0.125, 0.25, 0.5, 0.75, 1.0]
    static let compoundAbsoluteTolerance: Float = 1e-7
    /// Proven dot-only bound: for <=5-term same-sign dots, CPU multiply/add
    /// has gamma_10 <5.97e-7 and GPU FMA has gamma_5 <2.99e-7; direct comparison
    /// is below 8.96e-7 relative. The composed positive MoE comparison freezes
    /// atol=1e-7, rtol=8e-6 as a pre-GPU engineering tolerance covering propagated
    /// dots, Foundation/Metal exp-div variation, and shared activation/epilogue
    /// compiled in the existing default library. It is not a rigorous MSL vendor
    /// accuracy guarantee; no candidate result was observed and this will not be
    /// tuned after execution.
    static let compoundRelativeTolerance: Float = 8e-6

    /// Independently computed from literal matrixBits and inputs by the CPU
    /// reference; the scalar is a sanity anchor, not a candidate-derived value.
    static var fp32DiscriminatorExpected: Float {
        cpuProjection(input: Array(inputs[0..<columns]),
                      matrixBits: Array(matrixBits[0..<columns]),
                      rows: 1, columns: columns)[0]
    }
    static let discriminatorMathematicalValue: Float = 1.000732421875
    static let nearestFloat16Bits: UInt16 = 0x3c01
    static var nearestFloat16Value: Float {
        Float(Float16(bitPattern: nearestFloat16Bits))
    }

    static func cpuSharedMoEExpected(applyOutputGate: Bool = true) -> [Float] {
        let gate = cpuProjection(input: sharedInput, matrixBits: sharedGateBits,
                                 rows: 5, columns: 5)
        let up = cpuProjection(input: sharedInput, matrixBits: sharedUpBits,
                               rows: 5, columns: 5)
        let activation = zip(gate, up).map { gateValue, upValue in
            gateValue / (1 + Foundation.exp(-gateValue)) * upValue
        }
        let down = cpuProjection(input: activation, matrixBits: sharedDownBits,
                                 rows: 5, columns: 5)
        let rawGate = cpuProjection(input: sharedInput,
                                    matrixBits: sharedOutputGateBits,
                                    rows: 1, columns: 5)[0]
        let outputGate = applyOutputGate
            ? 1 / (1 + Foundation.exp(-rawGate))
            : 1
        return zip(sharedInitialOutput, down.map { $0 * outputGate }).map(+)
    }

    /// Frozen Task 3.7 finite comparison rule: abs(error) <= atol + rtol*abs(expected),
    /// with no extra ULP allowance. Exceptional classes and signed zero are checked
    /// separately. For the K<=5 dots here, products have one sign per row/token;
    /// gamma_10 for FP32 products plus additions is < 5.97e-7 times sum(abs(products)),
    /// below the frozen 1e-6 relative term when the expected value is nonzero.
    static let absoluteTolerance: Float = 1e-7
    static let relativeTolerance: Float = 1e-6

    static func float(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    static func littleEndianBytes(_ words: [UInt16]) -> [UInt8] {
        words.flatMap { word in
            [UInt8(truncatingIfNeeded: word), UInt8(truncatingIfNeeded: word >> 8)]
        }
    }

    static func cpuProjection(
        input: [Float], matrixBits: [UInt16], rows: Int, columns: Int
    ) -> [Float] {
        precondition(columns > 0 && rows > 0 && input.count.isMultiple(of: columns))
        precondition(matrixBits.count == rows * columns)
        let tokenCount = input.count / columns
        var result = [Float](repeating: 0, count: tokenCount * rows)
        for token in 0..<tokenCount {
            for row in 0..<rows {
                var sum: Float = 0
                for column in 0..<columns {
                    let weight = float(matrixBits[row * columns + column])
                    sum += weight * input[token * columns + column]
                }
                result[token * rows + row] = sum
            }
        }
        return result
    }

    static func cpuEmbedding(tokenIDs: [UInt32], bits: [UInt16],
                             rows: Int, columns: Int) -> [Float] {
        precondition(bits.count == rows * columns)
        return tokenIDs.flatMap { tokenID in
            precondition(tokenID < UInt32(rows))
            let start = Int(tokenID) * columns
            return bits[start..<(start + columns)].map { float($0) }
        }
    }

    static func matchesFrozenTolerance(_ actual: Float, _ expected: Float) -> Bool {
        guard actual.isFinite, expected.isFinite else { return false }
        return abs(actual - expected)
            <= absoluteTolerance + relativeTolerance * abs(expected)
    }
}

struct QwenBF16LiteralTensor {
    let name: String
    let rows: Int
    let columns: Int
    let bits: [UInt16]

    init(name: String, rows: Int, columns: Int, bits: [UInt16]) {
        precondition(rows > 0 && columns > 0 && bits.count == rows * columns)
        self.name = name
        self.rows = rows
        self.columns = columns
        self.bits = bits
    }
}

/// A tiny protected synthetic source. Its descriptor uses the pinned name/hash
/// inventory, while its only shard bytes are local literal BF16 test data.
struct QwenBF16SyntheticSource {
    let root: URL
    let sourceRoot: URL
    let registrationURL: URL
    let shardURL: URL
    let shardName: String
    let handle: OfficialSourceHandle
    let tensorOffsets: [String: UInt64]

    static func make(tensors: [QwenBF16LiteralTensor]) throws -> Self {
        guard !tensors.isEmpty,
              Set(tensors.map(\.name)).count == tensors.count else {
            throw QwenBF16FixtureError.invalidTensorList
        }
        let manager = FileManager.default
        var root = manager.temporaryDirectory.appendingPathComponent(
            "qwen-bf16-synthetic-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: false)
        var canonical = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(root.path, &canonical) != nil else {
            try? manager.removeItem(at: root)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        let canonicalBytes = canonical.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        root = URL(fileURLWithPath: String(decoding: canonicalBytes, as: UTF8.self),
                   isDirectory: true)
        do {
            let sourceRoot = root.appendingPathComponent("source", isDirectory: true)
            try manager.createDirectory(at: sourceRoot, withIntermediateDirectories: false)
            let registrationParent = root.appendingPathComponent("models", isDirectory: true)
            try manager.createDirectory(at: registrationParent, withIntermediateDirectories: false)
            let registrationURL = registrationParent.appendingPathComponent(
                "synthetic.gturbo", isDirectory: true)
            let identity = OfficialQwenSourceIdentity.pinned
            guard let shardName = identity.shards.first?.filename else {
                throw QwenBF16FixtureError.missingPinnedShard
            }
            let shardURL = sourceRoot.appendingPathComponent(shardName)
            let (fileBytes, relativeOffsets) = try safetensorsFile(tensors)
            try fileBytes.write(to: shardURL, options: .withoutOverwriting)

            let descriptor = try OfficialSourceDescriptor(
                repository: identity.repository,
                revision: identity.revision,
                storageProfile: identity.storageProfile,
                sidecarSHA256: identity.sidecarSHA256,
                shards: identity.shards.map {
                    OfficialSourceDescriptor.Shard(filename: $0.filename, sha256: $0.sha256)
                },
                sourceRoot: sourceRoot.path)
            let marker = try JSONEncoder().encode(descriptor)
            _ = try OfficialSourceRegistration.register(markerData: marker, at: registrationURL)
            let handle = try OfficialSourceHandle(registrationURL: registrationURL)
            let payloadStart = UInt64(fileBytes.count - relativeOffsets.payloadBytes)
            let tensorOffsets = relativeOffsets.offsets.mapValues { payloadStart + $0 }
            return Self(root: root, sourceRoot: sourceRoot,
                        registrationURL: registrationURL, shardURL: shardURL,
                        shardName: shardName, handle: handle, tensorOffsets: tensorOffsets)
        } catch {
            try? manager.removeItem(at: root)
            throw error
        }
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    private static func safetensorsFile(
        _ tensors: [QwenBF16LiteralTensor]
    ) throws -> (Data, (offsets: [String: UInt64], payloadBytes: Int)) {
        var payload = Data()
        var offsets: [String: UInt64] = [:]
        var header: [String: [String: Any]] = [:]
        for tensor in tensors {
            let start = payload.count
            let bytes = Data(QwenBF16TestFixture.littleEndianBytes(tensor.bits))
            payload.append(bytes)
            offsets[tensor.name] = UInt64(start)
            header[tensor.name] = [
                "dtype": "BF16", "shape": [tensor.rows, tensor.columns],
                "data_offsets": [start, payload.count],
            ]
        }
        var headerBytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        headerBytes.append(contentsOf: repeatElement(
            UInt8(0x20), count: (8 - headerBytes.count % 8) % 8))
        var headerLength = UInt64(headerBytes.count).littleEndian
        var file = withUnsafeBytes(of: &headerLength) { Data($0) }
        file.append(headerBytes)
        file.append(payload)
        return (file, (offsets, payload.count))
    }
}

private enum QwenBF16FixtureError: Error {
    case invalidTensorList
    case missingPinnedShard
}
