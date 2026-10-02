import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Test-only proof for Astra's small32 projection candidate. The expected
/// outputs are the existing pinned Torch 2.10 CPU FP32-linear fixture, rather
/// than another GPU implementation.
@Suite(.serialized)
struct QwenBF16Small32IndependentOracleTests {
    @Test
    func candidateMatchesEveryPinnedTorch32x2048OutputBit() async throws {
        let context = try MetalContext()
        let fixture = try Small32IndependentFixture(context: context)

        for oracle in Self.oracles {
            let (input, weights) = Self.generatedFixture(seed: oracle.seed)
            #expect(Self.sha256(Self.littleEndianBytes(input)) == oracle.inputSHA256,
                    "\(oracle.outputFile) input generator must match the pinned fixture")
            let matrix = weights.map { Float(bitPattern: UInt32($0) << 16) }
            #expect(Self.sha256(Self.littleEndianBytes(matrix)) == oracle.matrixFP32SHA256,
                    "\(oracle.outputFile) BF16 matrix must match the pinned fixture")
            let expectedData = try Self.resourceData(oracle.outputFile)
            #expect(Self.sha256(expectedData) == oracle.outputSHA256)
            let expected = Self.decodeUInt32s(expectedData)
            #expect(expected.count == 32)

            let actual = try await fixture.project(input: input, weights: weights, lanes: true)
            #expect(actual == expected,
                    "small32 candidate differs from pinned Torch bits for \(oracle.outputFile)")
        }
    }

    @Test
    func candidatePreservesNonfiniteBitsOfEstablishedSmall32Order() async throws {
        let context = try MetalContext()
        let fixture = try Small32IndependentFixture(context: context)
        var input = [Float](repeating: 0, count: 2048)
        input[0] = .infinity
        var weights = [UInt16](repeating: 0x3f80, count: 32 * 2048)
        // 0 * +Inf produces the deliberate nonfinite term in every row.
        for row in 0..<32 { weights[row * 2048] = 0 }

        let established = try await fixture.project(input: input, weights: weights, lanes: false)
        let candidate = try await fixture.project(input: input, weights: weights, lanes: true)
        #expect(established.count == 32)
        #expect(established.allSatisfy { Float(bitPattern: $0).isNaN })
        #expect(candidate.allSatisfy { Float(bitPattern: $0).isNaN })
        #expect(candidate == established,
                "candidate must preserve the established small32 nonfinite payload bits")
    }

    private static let oracles: [PinnedSmall32Oracle] = [
        .init(seed: 0x903e_faf1, inputSHA256: "2ece01b57f9e905dfc36d8ac7d67399bd84272c1fcd2ee61e9184cb08d1b8dad", matrixFP32SHA256: "31c52b5bc277ae3eba13f9f1cdfffc06cf9c6218fdf2684658b739ca3edc8cb8", outputFile: "r32-c2048-seed903efaf1.torch210.fp32-le.bin", outputSHA256: "597ca89a44a9503a65c560d6641cbb5fd87b2fd5ff5c972b5256e87d353d9c96"),
        .init(seed: 0x903f_faf2, inputSHA256: "0b75e00004ee703a7978bdf8bb77582e5dc6db13492a5f3d0b32cd5ac4b319e1", matrixFP32SHA256: "4bd0f75df66afd415e57235ce85b500e95a9a4eef9511b7013e9f6a6a7984e38", outputFile: "r32-c2048-seed903ffaf2.torch210.fp32-le.bin", outputSHA256: "e65af85f8e8414be26931d654707a53b774b53fae411123897590208c340f9d7", expectedSeedLabel: "0x903ffaf2"),
        .init(seed: 0x9040_faf3, inputSHA256: "3196d9124e136e1c7b165311216bb5e9aeeca34809d1e71958ddd6d300001a2e", matrixFP32SHA256: "7d20c364f04aa12918199386fe25f77d66697132b22b88c1455e59f2d05d7014", outputFile: "r32-c2048-seed9040faf3.torch210.fp32-le.bin", outputSHA256: "fabc0683b57ca82c9cb999b42bd6fe483abaf93549f199563274d530350f433c", expectedSeedLabel: "0x9040faf3"),
        .init(seed: 0x9041_faf4, inputSHA256: "4d5f5e233765536c12fedb9a71ef21d1219c88fd0bf2366b15043a9fa32fb5eb", matrixFP32SHA256: "e83aff04b6c48fed3bd27d01d729c4e637ed0a9b2abc68d9c9b89a2f3ab33ac7", outputFile: "r32-c2048-seed9041faf4.torch210.fp32-le.bin", outputSHA256: "99b40103e0c08e6b447cee31676fd09272117b2565ed5b8b25ca03d1c6276f09", expectedSeedLabel: "0x9041faf4"),
        .init(seed: 0x9042_faf5, inputSHA256: "33777c835ffa416b9d7ed8065b39d924c8ff529b10715daa8ad963ce0a2e7e6a", matrixFP32SHA256: "caa91a2ebdb2b867e506e16dbd1bb27fb6c2912f669ea7e073e96ffa86d9c60b", outputFile: "r32-c2048-seed9042faf5.torch210.fp32-le.bin", outputSHA256: "749e55ab3336140b3059a475ad252a17f50b56bba76eb8eb6690961725364043", expectedSeedLabel: "0x9042faf5"),
        .init(seed: 0x9043_faf6, inputSHA256: "da27144c9be8a92a80d576fb4647c27f76614e590ff1afe3aa8e626fed23d908", matrixFP32SHA256: "6e3deab400f92bb01eca4c5c9eaf6d645b4339872fbc170351418a234e928005", outputFile: "r32-c2048-seed9043faf6.torch210.fp32-le.bin", outputSHA256: "8256bcda2d9172252457540130fd85a8c0e4c58bafe3373d26de7b3e44cce93a", expectedSeedLabel: "0x9043faf6"),
        .init(seed: 0x9044_faf7, inputSHA256: "1bb84e3b4f49ad7f5eaa5d9149432070eb13bf17343da5645e2fa237b98a6f85", matrixFP32SHA256: "8cdf73c08fb597055987a66eb0ba9d9b3bfff9a29539553df14971e9a4e3e03e", outputFile: "r32-c2048-seed9044faf7.torch210.fp32-le.bin", outputSHA256: "8e41a93fd32005dd5e4234f048b1a972805cf40982fc452ddd804aa788ea4c6b", expectedSeedLabel: "0x9044faf7"),
        .init(seed: 0x9045_faf8, inputSHA256: "343548376b48e4f512ff98dfcda7d2ffa89466778437ac0f84b2ae49273d64f1", matrixFP32SHA256: "47641f7422e4f45a84c8a8cc729525d64a9d91707f468b578062bc912fe6bb14", outputFile: "r32-c2048-seed9045faf8.torch210.fp32-le.bin", outputSHA256: "1361f59a2b8b55448f442d4208b52b6364a38d349f8d69cda9221d327d52261f", expectedSeedLabel: "0x9045faf8"),
    ]

    private static func generatedFixture(seed: UInt32) -> (input: [Float], weights: [UInt16]) {
        var random = Small32FixtureRNG(state: seed)
        let input = (0..<2048).map { _ -> Float in
            let mantissa = random.next() & 0x02ff_ffff
            let sign = random.next() & 0x8000_0000
            return Float(bitPattern: 0x3d00_0000 &+ mantissa &+ sign)
        }
        let weights = (0..<(32 * 2048)).map { _ -> UInt16 in
            let magnitude = UInt16(0x3a00 + random.next() % 1536)
            return magnitude | (UInt16(truncatingIfNeeded: random.next()) & 0x8000)
        }
        return (input, weights)
    }

    private static func resourceData(_ name: String) throws -> Data {
        guard let directory = Bundle.module.resourceURL?.appendingPathComponent(
            "BF16ProjectionOracle", isDirectory: true) else {
            throw Small32OracleError.missingResource(name)
        }
        let url = directory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Small32OracleError.missingResource(name)
        }
        return try Data(contentsOf: url)
    }

    private static func decodeUInt32s(_ data: Data) -> [UInt32] {
        stride(from: 0, to: data.count, by: MemoryLayout<UInt32>.stride).map { offset in
            UInt32(littleEndian: data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            })
        }
    }

    private static func littleEndianBytes(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * MemoryLayout<UInt32>.stride)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private final class Small32IndependentFixture {
    private struct Parameters {
        var rows: UInt32
        var columns: UInt32
        var firstRow: UInt32
        var rowsInChunk: UInt32
        var tokenCount: UInt32
    }

    private let context: MetalContext
    private let serial: MTLComputePipelineState
    private let lanes: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.context = context
        let library = try MetalContext.privateLibrary(device: context.device,
            module: "qwen_bf16", mathMode: .safe)
        let laneLibrary = try MetalContext.privateLibrary(device: context.device,
            module: "qwen_bf16_small32_lanes", mathMode: .safe)
        serial = try context.device.makeComputePipelineState(function:
            #require(library.makeFunction(name: "qwen_bf16_project_fp32")))
        lanes = try context.device.makeComputePipelineState(function:
            #require(laneLibrary.makeFunction(name: "qwen_bf16_project_small32_lanes")))
        try #require(lanes.threadExecutionWidth == 32)
        try #require(lanes.maxTotalThreadsPerThreadgroup >= 32)
    }

    func project(input: [Float], weights: [UInt16], lanes useLanes: Bool) async throws -> [UInt32] {
        #expect(input.count == 2048)
        #expect(weights.count == 32 * 2048)
        let inputBuffer = try #require(input.withUnsafeBytes { bytes in
            context.device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                      options: .storageModeShared)
        })
        let weightBuffer = try #require(weights.withUnsafeBytes { bytes in
            context.device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                      options: .storageModeShared)
        })
        let outputBuffer = try #require(context.device.makeBuffer(
            length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared))
        var parameters = Parameters(rows: 32, columns: 2048, firstRow: 0,
                                    rowsInChunk: 32, tokenCount: 1)
        let command = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(useLanes ? lanes : serial)
        encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 5)
        if useLanes {
            encoder.dispatchThreadgroups(MTLSize(width: 32, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        } else {
            encoder.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        encoder.endEncoding()
        command.commit()
        await command.completed()
        try checkCommandBufferError(command)
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: 32)
        return Array(UnsafeBufferPointer(start: pointer, count: 32)).map(\.bitPattern)
    }
}

private struct PinnedSmall32Oracle {
    let seed: UInt32
    let inputSHA256: String
    let matrixFP32SHA256: String
    let outputFile: String
    let outputSHA256: String
    let expectedSeedLabel: String?

    init(seed: UInt32, inputSHA256: String, matrixFP32SHA256: String,
         outputFile: String, outputSHA256: String, expectedSeedLabel: String? = nil) {
        self.seed = seed
        self.inputSHA256 = inputSHA256
        self.matrixFP32SHA256 = matrixFP32SHA256
        self.outputFile = outputFile
        self.outputSHA256 = outputSHA256
        self.expectedSeedLabel = expectedSeedLabel
    }
}

private struct Small32FixtureRNG {
    private(set) var state: UInt32

    mutating func next() -> UInt32 {
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        return state
    }
}

private enum Small32OracleError: Error {
    case missingResource(String)
}
