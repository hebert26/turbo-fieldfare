import CryptoKit
import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Exercises qualified BF16 projection geometries against pinned Torch 2.10
/// output bits, plus single-row, overflow, and width-tail behavior.

@Suite(.serialized) struct QwenBF16LargeProjectionParityTests {
    @Test func qualifiedRealProjectionAndRouterShapesMatchPinnedTorchForEveryRow() throws {
        let manifest = try loadManifest()
        #expect(manifest.torchVersion == "2.10.0")
        #expect(manifest.torchGit == "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0")
        #expect(manifest.torchThreads == 1)
        #expect(manifest.device == "cpu")
        #expect(manifest.oracle == "torch.nn.functional.linear on CPU FP32 weights and input")
        #expect(manifest.baseGeneratorSeed == "0x903efaf1")
        let qualificationProofKeys = manifest.qualificationProofs.map {
            "\($0.geometry):\($0.sha256)"
        }
        #expect(qualificationProofKeys == [
            "512x2048:eab5868cce66b937cd1311dbdb521f88979efb6c07a66cc6b8bd62402f89e8db",
            "32x2048:ca479ceb595903bb88ff6b5ff1b0edf480d23c8a6e7939c9325dc682dbd92c89",
        ])
        let largeCases = [
            "8192x2048@0x903efaf1", "4096x2048@0x903efaf1",
            "1024x2048@0x903efaf1", "512x2048@0x903efaf1",
            "256x2048@0x903efaf1", "2048x512@0x903efaf1",
            "2048x2048@0x903efaf1", "2048x4096@0x903efaf1",
        ]
        let small32Seeds = (0..<8).map { index in
            let seed = UInt32(0x903e_faf1) &+ UInt32(index) &* 0x1_0001
            return String(format: "0x%08x", seed)
        }
        #expect(manifest.small32Seeds == small32Seeds)
        let small32Cases = small32Seeds.map { "32x2048@\($0)" }
        let expectedCases = largeCases + small32Cases
        let caseKeys = manifest.cases.map {
            "\($0.rows)x\($0.columns)@\($0.seed)"
        }
        #expect(caseKeys == expectedCases)

        for record in manifest.cases {
            #expect(record.inputShape == [1, 1, record.columns])
            let (input, weights) = try fixture(
                rows: record.rows, columns: record.columns, seed: record.seed)
            #expect(record.inputAddressModulo64 == 0)
            #expect(record.matrixAddressModulo64 == 0)
            expectPinnedFixtureHashes(input: input, weights: weights, record: record)
            let expectedBits = try outputBits(for: record)
            let chunkBytes: UInt64? = record.rows == 32 && record.columns == 2048
                ? UInt64(6 * record.columns * MemoryLayout<UInt16>.stride) : nil
            let actual = try project(input: input, weights: weights,
                                     rows: record.rows, columns: record.columns,
                                     maximumChunkBytes: chunkBytes)
            #expect(actual.count == record.rows)
            #expect(actual.allSatisfy { $0.isFinite })
            let actualBits = actual.map { $0.bitPattern }
            #expect(actualBits == expectedBits,
                    "\(record.rows)x\(record.columns) must match pinned Torch for every row")

            if record.rows == 512 && record.columns == 2048 {
                let oldStable = legacyStableDots(input: input, weights: weights,
                                                 rows: record.rows, columns: record.columns)
                let oldStableFailures = zip(oldStable, expectedBits).filter {
                    let expectedValue = Float(bitPattern: $0.1)
                    return abs($0.0 - expectedValue) > 1e-7 + 1e-6 * abs(expectedValue)
                }
                #expect(!oldStableFailures.isEmpty,
                        "fixture must reject the former compensated-dot order at frozen tolerance")
            }
        }
    }

    @Test func qualifiedLargeProjectionPropagatesOverflowFromItsReductionOrder() throws {
        let columns = 2048
        var input = [Float](repeating: 0, count: columns)
        input[0] = Float.greatestFiniteMagnitude
        input[64] = Float.greatestFiniteMagnitude
        input[1] = -Float.greatestFiniteMagnitude
        input[65] = -Float.greatestFiniteMagnitude
        let weights = [UInt16](repeating: 0x3f80, count: 512 * columns)
        let actual = try project(input: input, weights: weights, rows: 512, columns: columns)
        #expect(actual.count == 512)
        #expect(actual.allSatisfy { $0.isNaN },
                "the qualified reduction's Inf plus negative Inf must remain NaN")
    }

    @Test func multiTokenLargeProjectionMatchesIndependentOrderedOracleAcrossRowChunkBoundaries() throws {
        let rows = 512
        let columns = 2048
        let tokenCount = 17
        let (_, weights) = try fixture(rows: rows, columns: columns)
        let baseInput = try fixture(rows: 1, columns: columns, seed: "0x913efaf1").input
        var input: [Float] = []
        input.reserveCapacity(tokenCount * columns)
        for token in 0..<tokenCount {
            for column in 0..<columns {
                let base = baseInput[column]
                let signedBase = (token + column).isMultiple(of: 2) ? base : -base
                input.append(signedBase + Float(token - 8) * 0.0078125)
            }
        }
        let expected = sourceLargeOrderedDots(input: input, weights: weights,
                                              rows: rows, columns: columns,
                                              tokenCount: tokenCount)

        for chunkRows in [63, 64, 65] {
            let expectedChunkRows = stride(from: 0, to: rows, by: chunkRows).map {
                min(chunkRows, rows - $0)
            }
            let actual = try project(input: input, weights: weights,
                                     rows: rows, columns: columns,
                                     tokenCount: tokenCount,
                                     expectedChunkRows: expectedChunkRows,
                                     maximumChunkBytes: UInt64(chunkRows * columns
                                        * MemoryLayout<UInt16>.stride))
            #expect(actual.count == tokenCount * rows)
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern),
                    "\(tokenCount) dense tokens must preserve ordered source bits at \(chunkRows)-row chunks")
        }
    }

    @Test func singleRowProjectionMatchesBothPinnedTorch69CaseSets() throws {
        let manifest = try loadManifest()
        #expect(manifest.singleRowProofs.map { $0.dataset } == ["original", "fresh"])
        let context = try MetalContext()
        for proof in manifest.singleRowProofs {
            let expectedProof = proof.dataset == "original"
                ? ("19eeff39c789f9baeac8c281a6aa9c5b551dfd71401ead78a13cae424b040b5b",
                   "8cd7e7e9cedd958c35a7b94145b5b430bf550f158e44d906076ec3a96809db61")
                : ("4465d944c3ebb5c247b9c7d23b91c641f66477c6809fa85db9a5a6df2529ae55",
                   "e8439d105b6b6238ad1c35cbfc873ecd6c807d009151993a4fd4168028065b35")
            #expect(proof.expectedSHA256 == expectedProof.0)
            #expect(proof.receiptSHA256 == expectedProof.1)
            let receiptData = try singleRowResourceData(proof.receiptFile)
            #expect(sha256(receiptData) == proof.receiptSHA256)
            let receipt = try JSONDecoder().decode(SingleRowTorchReceipt.self, from: receiptData)
            #expect(receipt.torch == "2.10.0")
            #expect(receipt.torchGit == "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0")
            #expect(receipt.threads == 1)
            #expect(receipt.device == "cpu")
            #expect(receipt.width == 2048)
            #expect(receipt.caseCount == 69)
            if proof.dataset == "original" {
                #expect(receipt.seedBase == nil)
            } else {
                #expect(receipt.seedBase == 0x7f24_a18d)
            }
            #expect(receipt.chunkedFMAExactMismatches == (proof.dataset == "original" ? 4 : 0))
            #expect(receipt.originalWeightReads == 0)
            #expect(receipt.cases.count == 69)

            let expectedData = try singleRowResourceData(proof.expectedFile)
            #expect(sha256(expectedData) == proof.expectedSHA256)
            #expect(sha256(expectedData) == receipt.payloadSHA256["expected.fp32-le.bin"])
            #expect(expectedData.count == receipt.caseCount * MemoryLayout<UInt32>.stride)
            let expectedBits = decodeUInt32s(expectedData)
            #expect(expectedBits.count == receipt.caseCount)
            let fixtures = singleRowFixtures(dataset: proof.dataset)
            #expect(fixtures.count == receipt.cases.count)
            var priorOrderFailures: [String] = []

            for index in fixtures.indices {
                let fixture = fixtures[index]
                let record = receipt.cases[index]
                #expect(record.label == fixture.label)
                #expect(record.seed == fixture.seed)
                #expect(sha256(littleEndianBytes(fixture.input)) == record.inputSHA256)
                #expect(sha256(littleEndianBytes(fixture.weights)) == record.weightSHA256)
                #expect(expectedBits[index] == record.officialBits)
                #expect(record.chunkedFMAExact == (record.officialBits == record.chunkedFMABits))
                if record.chunkedFMABits != record.officialBits {
                    priorOrderFailures.append(record.label)
                }

                let actual = try project(input: fixture.input, weights: fixture.weights,
                                         rows: 1, columns: 2048, context: context)
                #expect(actual.count == 1)
                #expect(actual[0].bitPattern == expectedBits[index],
                        "\(proof.dataset) \(fixture.label) must match its pinned Torch output bits")
            }
            #expect(priorOrderFailures == (proof.dataset == "original"
                ? ["mixed-sign-6", "mixed-sign-25", "mixed-sign-41", "mixed-sign-51"]
                : []))
        }
    }

    @Test func nonMultipleOf64WidthKeepsFrozenProjectionTolerance() throws {
        let (input, bits) = try fixture(rows: 512, columns: 2050)
        let expected = mathematicalDots(input: input, weights: bits,
                                        rows: 512, columns: 2050)
        let actual = try project(input: input, weights: bits,
                                 rows: 512, columns: 2050)

        #expect(actual.count == 512)
        #expect(actual.allSatisfy { $0.isFinite })
        #expect(expected.allSatisfy { $0.isFinite })
        for row in 0..<512 {
            let limit = 1e-7 + 1e-6 * abs(expected[row])
            #expect(abs(actual[row] - expected[row]) <= limit,
                    "row \(row): actual \(actual[row]), mathematical dot \(expected[row]), limit \(limit)")
        }
    }

    private func fixture(
        rows: Int, columns: Int, seed: String = "0x903efaf1"
    ) throws -> (input: [Float], weights: [UInt16]) {
        guard let seedValue = UInt32(seed.dropFirst(2), radix: 16) else {
            throw ProjectionFixtureError.invalidSeed(seed)
        }
        var random = ProjectionFixtureRNG(state: seedValue)
        let input = (0..<columns).map { _ -> Float in
            let mantissa = random.next() & 0x02ff_ffff
            let sign = random.next() & 0x8000_0000
            return Float(bitPattern: 0x3d00_0000 &+ mantissa &+ sign)
        }
        let weights = (0..<(rows * columns)).map { _ -> UInt16 in
            let magnitude = UInt16(0x3a00 + random.next() % 1536)
            return magnitude | (UInt16(truncatingIfNeeded: random.next()) & 0x8000)
        }
        return (input, weights)
    }

    private func singleRowFixtures(dataset: String) -> [SingleRowProjectionFixture] {
        let seedBase: UInt32 = dataset == "original" ? 0x6a12_e59b : 0x7f24_a18d
        var fixtures: [SingleRowProjectionFixture] = []
        for index in 0..<64 {
            let seed = seedBase &+ UInt32(index) &* 0x1_0001
            var random = ProjectionFixtureRNG(state: seed)
            let input = (0..<2048).map { _ -> Float in
                let mantissa = random.next() & 0x02ff_ffff
                let sign = random.next() & 0x8000_0000
                return Float(bitPattern: 0x3d00_0000 &+ mantissa &+ sign)
            }
            let weights = (0..<2048).map { _ -> UInt16 in
                let magnitude = UInt16(0x3a00 + random.next() % 1536)
                return magnitude &+ (UInt16(truncatingIfNeeded: random.next()) & 0x8000)
            }
            fixtures.append(SingleRowProjectionFixture(
                label: "mixed-sign-\(index)", seed: seed, input: input, weights: weights))
        }

        let one = Float(bitPattern: 0x3f80_0000)
        let unitWeights = [UInt16](repeating: 0x3f80, count: 2048)
        let cancellation: [Float] = Array(
            repeating: [Float(16_777_216), Float(1), Float(-16_777_216), Float(1)],
            count: 512).flatMap { $0 }
        let crossChunkCancellation = [Float(16_777_216)]
            + [Float](repeating: 1, count: 511)
            + [-Float(16_777_216)]
            + [Float](repeating: 1, count: 1535)
        let roundedProductInput = [Float(bitPattern: 0x3f80_0001), -1.0078125]
            + [Float](repeating: 0, count: 2046)
        let roundedProductWeights = [UInt16(0x3f81), 0x3f80]
            + [UInt16](repeating: 0, count: 2046)
        let structured: [(String, [Float], [UInt16])] = [
            ("all-zero", [Float](repeating: 0, count: 2048), unitWeights),
            ("all-one", [Float](repeating: one, count: 2048), unitWeights),
            ("large-cancellation", cancellation, unitWeights),
            ("rounded-product-cancellation", roundedProductInput, roundedProductWeights),
            ("cross-chunk-cancellation", crossChunkCancellation, unitWeights),
        ]
        fixtures += structured.map {
            SingleRowProjectionFixture(label: $0.0, seed: nil, input: $0.1, weights: $0.2)
        }
        return fixtures
    }

    private func loadManifest() throws -> PinnedTorchProjectionManifest {
        guard let resourceDirectory = Bundle.module.resourceURL?.appendingPathComponent(
            "BF16ProjectionOracle", isDirectory: true) else {
            throw ProjectionFixtureError.missingManifest
        }
        let url = resourceDirectory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ProjectionFixtureError.missingManifest
        }
        return try JSONDecoder().decode(
            PinnedTorchProjectionManifest.self, from: Data(contentsOf: url))
    }

    private func outputBits(for record: PinnedTorchProjectionCase) throws -> [UInt32] {
        guard let resourceDirectory = Bundle.module.resourceURL?.appendingPathComponent(
            "BF16ProjectionOracle", isDirectory: true) else {
            throw ProjectionFixtureError.missingOutput(record.outputFile)
        }
        let url = resourceDirectory.appendingPathComponent(record.outputFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ProjectionFixtureError.missingOutput(record.outputFile)
        }
        let data = try Data(contentsOf: url)
        #expect(sha256(data) == record.outputSHA256)
        guard data.count == record.rows * MemoryLayout<UInt32>.stride else {
            throw ProjectionFixtureError.invalidOutput(record.outputFile)
        }
        return decodeUInt32s(data)
    }

    private func singleRowResourceData(_ name: String) throws -> Data {
        guard let resourceDirectory = Bundle.module.resourceURL?.appendingPathComponent(
            "BF16ProjectionOracle", isDirectory: true) else {
            throw ProjectionFixtureError.missingOutput(name)
        }
        let url = resourceDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ProjectionFixtureError.missingOutput(name)
        }
        return try Data(contentsOf: url)
    }

    private func decodeUInt32s(_ data: Data) -> [UInt32] {
        stride(from: 0, to: data.count, by: MemoryLayout<UInt32>.stride).map { offset in
            UInt32(littleEndian: data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            })
        }
    }

    private func mathematicalDots(input: [Float], weights: [UInt16],
                                  rows: Int, columns: Int) -> [Float] {
        (0..<rows).map { row in
            var sum = Double(0)
            for column in 0..<columns {
                let weight = Float(bitPattern: UInt32(weights[row * columns + column]) << 16)
                sum += Double(weight) * Double(input[column])
            }
            return Float(sum)
        }
    }

    private func sourceLargeOrderedDots(input: [Float], weights: [UInt16],
                                        rows: Int, columns: Int,
                                        tokenCount: Int) -> [Float] {
        var result = [Float](repeating: 0, count: tokenCount * rows)
        for token in 0..<tokenCount {
            let inputBase = token * columns
            for row in 0..<rows {
                let rowBase = row * columns
                var partial = [Float](repeating: 0, count: 64)
                for column in 0..<columns {
                    let weight = Float(bitPattern: UInt32(weights[rowBase + column]) << 16)
                    let stream = column & 63
                    partial[stream] = partial[stream].addingProduct(
                        weight, input[inputBase + column])
                }
                var collapsed = [Float](repeating: 0, count: 16)
                for index in 0..<16 {
                    let first = partial[index] + partial[index + 16]
                    let second = first + partial[index + 32]
                    collapsed[index] = second + partial[index + 48]
                }
                var groups = [Float](repeating: 0, count: 4)
                for index in 0..<4 {
                    let base = index * 4
                    let first = collapsed[base] + collapsed[base + 1]
                    let second = first + collapsed[base + 2]
                    groups[index] = second + collapsed[base + 3]
                }
                let first = groups[0] + groups[1]
                let second = first + groups[2]
                result[token * rows + row] = second + groups[3]
            }
        }
        return result
    }

    private func legacyStableDots(input: [Float], weights: [UInt16],
                                  rows: Int, columns: Int) -> [Float] {
        (0..<rows).map { row in
            var sum: Float = 0
            var correction: Float = 0
            for column in 0..<columns {
                let weight = Float(bitPattern: UInt32(weights[row * columns + column]) << 16)
                let product = weight * input[column]
                let next = sum + product
                let productError = (-product).addingProduct(weight, input[column])
                let sumError = abs(sum) >= abs(product)
                    ? (sum - next) + product : (product - next) + sum
                correction = correction + (productError + sumError)
                sum = next
            }
            return sum + correction
        }
    }

    private func expectPinnedFixtureHashes(
        input: [Float], weights: [UInt16], record: PinnedTorchProjectionCase
    ) {
        let matrix = weights.map { Float(bitPattern: UInt32($0) << 16) }
        #expect(sha256(littleEndianBytes(input)) == record.inputSHA256)
        #expect(sha256(littleEndianBytes(matrix)) == record.matrixFP32SHA256)
    }

    private func littleEndianBytes(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * MemoryLayout<UInt32>.stride)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    private func littleEndianBytes(_ values: [UInt16]) -> Data {
        var bytes = Data(capacity: values.count * MemoryLayout<UInt16>.stride)
        for value in values {
            var bits = value.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func project(input: [Float], weights: [UInt16],
                         rows: Int, columns: Int,
                         tokenCount: Int = 1,
                         expectedChunkRows: [Int]? = nil,
                         maximumChunkBytes: UInt64? = nil,
                         context suppliedContext: MetalContext? = nil) throws -> [Float] {
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "projection", rows: rows,
                                  columns: columns, bits: weights),
        ])
        defer { source.remove() }
        let context: MetalContext
        if let suppliedContext {
            context = suppliedContext
        } else {
            context = try MetalContext()
        }
        let resident = try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [QwenBF16TensorSpec(
                name: "projection", shardName: source.shardName,
                role: .dense, rows: rows, columns: columns)],
            residencyBudget: UInt64(weights.count * MemoryLayout<UInt16>.stride),
            maximumChunkBytes: maximumChunkBytes ?? 8 * 1024 * 1024,
            checkpoint: { _ in })
        if let expectedChunkRows {
            let chunks = resident.inspectedChunks
            var expectedFirstRows: [Int] = []
            var firstRow = 0
            for rowCount in expectedChunkRows {
                expectedFirstRows.append(firstRow)
                firstRow += rowCount
            }
            #expect(chunks.map(\.rowCount) == expectedChunkRows)
            #expect(chunks.map(\.firstRow) == expectedFirstRows)
        }
        let inputBuffer = try #require(input.withUnsafeBytes { bytes in
            context.device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                      options: .storageModeShared)
        })
        let output = try #require(context.device.makeBuffer(
            length: tokenCount * rows * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        try resident.encodeProjection(commandBuffer: command, tensorName: "projection",
                                      input: inputBuffer, tokenCount: tokenCount, output: output)
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed)
        return Array(UnsafeBufferPointer(
            start: output.contents().bindMemory(to: Float.self, capacity: tokenCount * rows),
            count: tokenCount * rows))
    }
}

private struct ProjectionFixtureRNG {
    private(set) var state: UInt32

    mutating func next() -> UInt32 {
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        return state
    }

}

private struct SingleRowProjectionFixture {
    let label: String
    let seed: UInt32?
    let input: [Float]
    let weights: [UInt16]
}

private struct PinnedTorchProjectionManifest: Decodable {
    let schemaVersion: Int
    let oracle: String
    let torchVersion: String
    let torchGit: String
    let torchThreads: Int
    let device: String
    let baseGeneratorSeed: String
    let small32Seeds: [String]
    let qualificationProofs: [PinnedProjectionQualificationProof]
    let singleRowProofs: [SingleRowProjectionProof]
    let cases: [PinnedTorchProjectionCase]
}

private struct SingleRowProjectionProof: Decodable {
    let dataset: String
    let proofPath: String
    let expectedFile: String
    let receiptFile: String
    let expectedSHA256: String
    let receiptSHA256: String
}

private struct SingleRowTorchReceipt: Decodable {
    let torch: String
    let torchGit: String
    let threads: Int
    let device: String
    let cases: [SingleRowTorchCase]
    let caseCount: Int
    let width: Int
    let seedBase: UInt32?
    let chunkedFMAExactMismatches: Int
    let originalWeightReads: Int
    let payloadSHA256: [String: String]
}

private struct SingleRowTorchCase: Decodable {
    let label: String
    let seed: UInt32?
    let officialBits: UInt32
    let chunkedFMABits: UInt32
    let chunkedFMAExact: Bool
    let inputSHA256: String
    let weightSHA256: String
}

private struct PinnedProjectionQualificationProof: Decodable {
    let geometry: String
    let path: String
    let sha256: String
}

private struct PinnedTorchProjectionCase: Decodable {
    let rows: Int
    let columns: Int
    let inputShape: [Int]
    let seed: String
    let outputFile: String
    let inputSHA256: String
    let matrixFP32SHA256: String
    let outputSHA256: String
    let inputAddressModulo64: Int
    let matrixAddressModulo64: Int
}

private enum ProjectionFixtureError: Error {
    case missingManifest
    case missingOutput(String)
    case invalidOutput(String)
    case invalidSeed(String)
}
