import Foundation
import CryptoKit
import Testing
@testable import TurboFieldfare

/// Pins the source router's FP32 operations to the saved CPU/Torch evidence
/// while keeping the existing packed router's default result unchanged.
@Suite struct QwenOfficialSourceRouterArithmeticTests {
    private static let officialExpertIDs = [238, 112, 106, 127, 157, 66, 56, 120]
    private static let officialWeights: [Float] = [
        0.22555266320705414, 0.19972732663154602, 0.143514022231102,
        0.1290992796421051, 0.09628725051879883, 0.08510386943817139,
        0.062115538865327835, 0.05860009416937828,
    ]
    // Frozen packed-path weights from candidate-baseline-001 for the identical
    // saved logits. These guard the packed default independently of source mode.
    private static let packedWeights: [Float] = [
        0.22555266, 0.19972731, 0.14351399, 0.12909928,
        0.09628724, 0.08510387, 0.06211554, 0.058600094,
    ]
    private static let selectedProbabilitySum: Float = 0.29442963004112244
    private static let softmaxDenominator: Float = 15.058112144470215

    // Little-endian FP32 logits from candidate-baseline-001, token 0, layer 0.
    // The source IDs/weights above are independently pinned by the unchanged
    // Transformers CPU router replay in output-diagnosis/router/run-001.
    private static let candidateLogitsBase64 = """
    xYLLwNNju8D/dZrAdTm5wDJtvsAu37fAii+LwEOjz8BiKcnA+djCwKzx1MArv7DAYqjQwBhmsMAZA8HArUflwKTqq8D/AcfA9c7MwHZRx8De6cTAFMjCwGd7iMDioMHA2cGrwG3w1MDJ69DAcPrcwESVocAr87zAcNCMwL+ft8DYnYzA8NGdwNWnssBgusPArlypwNJYx8C7DLzAco6xwIj4gcBsN6bAlczjwF7Ig8CL0bfA0GajwLomfMAoFcvAxvSzwOZcz8AcwsbAAmrIwOte48Aqdc3AUxiowHUozsC1O2rA81l5wEQ73MBgu57A/X2ywB8C3sAn2c7A302cwGHrx8DaI4zAxhRWwIEOucCGVrbAxH+5wL6gusCzE93Ahu/SwEaqx8D+i9rAOsSjwLsT4sA9brLAbC+zwDVYtsAru+nAxOe2wPmFxcCi07fAEnCkwPODpcDm/5/AqnSvwMbnj8B4hs/AvJzDwNGw18DuAIrABWCrwBr2usCoY4bABg/awM9d28CUBcbAoHqbwFSYyMCNo+PAPqa6wJOY5sCifarAfWy4wCajNMA13rTAuPbcwLrgucDgMXvAaZywwOd7H8AWceDAemm7wMhqm8Beds3Aj0e0wEoGysBMC8HAPfZtwPewlMC9hqXAIwvIwOXwo8DRjuHAETrxwGlpO8CLOajAowXVwPTP58Cpt7/AYDDBwBCb9sAddMDA7bvKwAEOrcBTcNvAXBXOwMFvncC1A6nAPjPYwB24tcDHBJfA3+GmwID81cAha63A9jekwDsP0MCh2brAEsLqwJ3GzcB6Bs3AnkOLwHv1vsCmVs/A8YPewPMtTsAUhdTA3j/XwORSnMDO+tHAjKi5wOYnucDQi8PArRDEwIbUv8BR3NrAbirLwGJ1uMBTeK3AYz7ZwN/DnsAtsZ7A6aTMwHovo8AS1aTAkTfCwKIT5MAk3+bAXcTAwOiSt8Dyd9XAsEnEwEzuwcBOc5LAne+bwBz8t8B/E4DAYXC3wB9U3sDG4o7AjyjPwJROtMB47L/AQyu3wPiI3sA2TdfAcgrIwKaG2MCg43bAgMnjwBvXmMCMrMHAJATEwHwKuMChTbHAzuKnwNsatsCqmOzAGvLBwKy92sB53I3ABc/WwByKxsATscbAqYjBwHqCr8Czb37ABs/DwFNj3MCeM77A5SmpwKyJpsCROazAF3G/wMAPsMB1asDAOpiqwOsqzMB6yNTAJ07PwLz5scCL7qPAYSe5wO8nz8Bz7Y3AWcGnwJmzF8Bg+rzAjXPawG9Ws8Bx8dvA2mPMwFF3qsBekcTAhFDBwB0F0sB9evDAnePKwDSWjMAzg6HAE9nSwIDt78D+KZLAuX3CwA==
    """

    private static let capturedLogits: [Float] = {
        guard let data = Data(base64Encoded: candidateLogitsBase64), data.count == 256 * 4 else {
            preconditionFailure("invalid frozen router-logit fixture")
        }
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bits = UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
            return Float(bitPattern: bits)
        }
    }()

    private func configuration() throws -> QwenMoEConfiguration {
        try QwenMoEConfiguration(
            hiddenSize: 1, expertCount: 256, topK: 8,
            routedIntermediateSize: 1, sharedIntermediateSize: 1)
    }

    @Test func pinnedSourceRouterMatchesTorchAndPackedDefaultRemainsFrozen() throws {
        let configuration = try configuration()
        let packedDefault = try QwenMoE.route(
            logits: Self.capturedLogits, configuration: configuration)
        let packedExplicit = try QwenMoE.route(
            logits: Self.capturedLogits, configuration: configuration, arithmetic: .packed)
        let source = try QwenMoE.route(
            logits: Self.capturedLogits, configuration: configuration,
            arithmetic: .officialSourceCPU)

        #expect(packedDefault == packedExplicit)
        #expect(packedDefault.selectedExpertIDs == [Self.officialExpertIDs])
        #expect(packedDefault.normalizedWeights[0].map(\.bitPattern)
            == Self.packedWeights.map(\.bitPattern))
        #expect(source.selectedExpertIDs == [Self.officialExpertIDs])
        #expect(source.normalizedWeights[0].map(\.bitPattern)
            == Self.officialWeights.map(\.bitPattern))
        #expect(source.probabilities[0].count == 256)
        #expect(source.probabilities[0].allSatisfy { $0.isFinite })
        #expect(abs(source.probabilities[0].reduce(0, +) - 1) < 1e-6)
        #expect(abs(source.normalizedWeights[0].reduce(0, +) - 1) < 1e-6)
    }

    @Test func sourceExpAndReductionHelpersMatchPinnedOperationProbes() throws {
        // First sixteen shifted-logit exponentials from exp-proof-001's pinned
        // Torch CPU/SLEEF comparison. Expected words come from its official file.
        let inputBits: [UInt32] = [
            0xc07f51f1, 0xc05f140d, 0xc01d3865, 0xc05abf51,
            0xc06526cb, 0xc0580ac3, 0xbffd56f6, 0xc083c976,
            0xc07a9f2b, 0xc06dfe59, 0xc08917e0, 0xc049cabd,
            0xc084ce96, 0xc0491897, 0xc06a5299, 0xc0996de0,
        ]
        let expectedBits: [UInt32] = [
            0x3c97a4eb, 0x3cfaf71c, 0x3daf928e, 0x3d0644a7,
            0x3ce43ec7, 0x3d0c10bd, 0x3e0d7e66, 0x3c854bbc,
            0x3ca33208, 0x3cc6cb04, 0x3c61db0c, 0x3d2eff26,
            0x3c811d2c, 0x3d30e8ec, 0x3cd28759, 0x3c078ecb,
        ]
        let exponentials = inputBits.map { QwenOfficialSourceRouterArithmetic.exponential(Float(bitPattern: $0)) }
        #expect(exponentials.map(\.bitPattern) == expectedBits)

        let shifted = Self.capturedLogits.map { $0 - Self.capturedLogits.max()! }
        let allExponentials = shifted.map(QwenOfficialSourceRouterArithmetic.exponential)
        #expect(QwenOfficialSourceRouterArithmetic.softmaxSum(allExponentials).bitPattern
            == Self.softmaxDenominator.bitPattern)

        let source = try QwenMoE.route(
            logits: Self.capturedLogits, configuration: try configuration(),
            arithmetic: .officialSourceCPU)
        let selectedProbabilities = Self.officialExpertIDs.map { source.probabilities[0][$0] }
        #expect(QwenOfficialSourceRouterArithmetic.top8Sum(selectedProbabilities).bitPattern
            == Self.selectedProbabilitySum.bitPattern)
    }

    @Test func allEqualTieOrderAndNormalizedWeightsRemainPinnedPerPath() throws {
        let logits = [Float](repeating: 0, count: 256)
        let configuration = try configuration()
        let packed = try QwenMoE.route(
            logits: logits, configuration: configuration, arithmetic: .packed)
        let source = try QwenMoE.route(
            logits: logits, configuration: configuration, arithmetic: .officialSourceCPU)

        #expect(packed.selectedExpertIDs == [Array(0..<8)])
        #expect(source.selectedExpertIDs == [QwenSourceTopKFixture.sourceAllEqualExpectedIDs])
        #expect(packed.normalizedWeights == [[Float](repeating: 0.125, count: 8)])
        #expect(source.normalizedWeights == [[Float](repeating: 0.125, count: 8)])
        #expect(packed.probabilities[0] == [Float](repeating: 1 / 256, count: 256))
        #expect(source.probabilities[0] == [Float](repeating: 1 / 256, count: 256))
    }

    @Test func sourceTop8MatchesPinnedCutoffTieWhilePackedKeepsLowerTieID() throws {
        let probabilities = QwenSourceTopKFixture.floats(
            QwenSourceTopKFixture.syntheticCutoffTieProbabilityBits)
        #expect(try QwenOfficialSourceRouterArithmetic.top8Indices(probabilities)
            == QwenSourceTopKFixture.syntheticCutoffTieExpectedIDs)

        let configuration = try configuration()
        let logits = QwenSourceTopKFixture.logits(
            QwenSourceTopKFixture.syntheticCutoffTieProbabilityBits)
        let source = try QwenMoE.route(
            logits: logits, configuration: configuration,
            arithmetic: .officialSourceCPU)
        let packed = try QwenMoE.route(
            logits: logits, configuration: configuration, arithmetic: .packed)

        #expect(source.selectedExpertIDs == [QwenSourceTopKFixture.syntheticCutoffTieExpectedIDs])
        #expect(packed.selectedExpertIDs == [[50, 12, 42, 19, 194, 172, 6, 28]])
    }

    @Test func sourceRouteMatchesObservedCutoffTieAndKeepsProbabilitiesFinite() throws {
        let probabilities = QwenSourceTopKFixture.floats(
            QwenSourceTopKFixture.observedCutoffTieProbabilityBits)
        #expect(try QwenOfficialSourceRouterArithmetic.top8Indices(probabilities)
            == QwenSourceTopKFixture.observedCutoffTieExpectedIDs)

        let configuration = try configuration()
        let logits = QwenSourceTopKFixture.floats(
            QwenSourceTopKFixture.observedOriginalRouterLogitsBits)
        let source = try QwenMoE.route(
            logits: logits, configuration: configuration,
            arithmetic: .officialSourceCPU)
        let packed = try QwenMoE.route(
            logits: logits, configuration: configuration, arithmetic: .packed)

        #expect(source.selectedExpertIDs == [QwenSourceTopKFixture.observedCutoffTieExpectedIDs])
        #expect(packed.selectedExpertIDs == [[241, 125, 66, 109, 148, 50, 228, 128]])
        #expect(source.normalizedWeights[0].map(\.bitPattern)
            == QwenSourceTopKFixture.observedNormalizedWeightBits)
        #expect(source.probabilities[0].allSatisfy { $0.isFinite })
        #expect(source.normalizedWeights[0].allSatisfy { $0.isFinite })
    }

    @Test func sourceTop8RejectsWrongShapeAndNonfiniteInput() throws {
        #expect(throws: QwenMoEError.invalidCount(
            field: "source router probabilities", expected: 256, actual: 255)) {
            try QwenOfficialSourceRouterArithmetic.top8Indices(
                [Float](repeating: 0, count: 255))
        }

        var nonfinite = [Float](repeating: 0, count: 256)
        nonfinite[190] = .nan
        #expect(throws: QwenMoEError.nonfiniteRouterValue(token: 0, expert: 190)) {
            try QwenOfficialSourceRouterArithmetic.top8Indices(nonfinite)
        }
    }

    @Test func fullPinnedTorchCorpusMatchesSourceTop8Helper() throws {
        let input = try sourceTopKResourceData("source-topk-inputs.fp32-le.bin")
        let expected = try sourceTopKResourceData("torch-ids.i64-le.bin")
        #expect(sourceTopKSHA256(input)
            == "d763364a698b01c9e6a3c077dd85e4f748a50109d2efebef62b46ec85f92e5ce")
        #expect(sourceTopKSHA256(expected)
            == "d7b3bc647b2250947107fecbd7c04cd4310c7c77afe348b1d7957cf9e0374416")
        #expect(input.count == 3_789 * 256 * MemoryLayout<Float>.stride)
        #expect(expected.count == 3_789 * 8 * MemoryLayout<Int64>.stride)

        let inputBits = sourceTopKUInt32s(input)
        let expectedIDs = sourceTopKUInt64s(expected)
        for row in 0..<3_789 {
            let start = row * 256
            let probabilities = inputBits[start..<(start + 256)].map {
                Float(bitPattern: $0)
            }
            let expectedStart = row * 8
            let expectedRow = expectedIDs[expectedStart..<(expectedStart + 8)].map(Int.init)
            let selected = try QwenOfficialSourceRouterArithmetic.top8Indices(probabilities)
            #expect(selected == expectedRow, "source Top-8 mismatch at corpus row \(row)")
        }
    }

    @Test func sourceAndPackedModesRetainShapeAndNonfiniteGuards() throws {
        let configuration = try configuration()
        for arithmetic in [QwenMoERoutingArithmetic.packed, .officialSourceCPU] {
            #expect(throws: QwenMoEError.invalidCount(
                field: "routerLogits", expected: 256, actual: 255)) {
                try QwenMoE.route(
                    logits: Array(Self.capturedLogits.dropLast()), configuration: configuration,
                    arithmetic: arithmetic)
            }

            var nanLogits = Self.capturedLogits
            nanLogits[17] = .nan
            #expect(throws: QwenMoEError.nonfiniteRouterValue(token: 0, expert: 17)) {
                try QwenMoE.route(logits: nanLogits, configuration: configuration,
                                  arithmetic: arithmetic)
            }

            var infiniteLogits = Self.capturedLogits + Self.capturedLogits
            infiniteLogits[256 + 19] = .infinity
            #expect(throws: QwenMoEError.nonfiniteRouterValue(token: 1, expert: 19)) {
                try QwenMoE.route(logits: infiniteLogits, configuration: configuration,
                                  arithmetic: arithmetic)
            }
        }
    }
}

private func sourceTopKResourceData(_ name: String) throws -> Data {
    let directory = try #require(Bundle.module.resourceURL?.appendingPathComponent(
        "source-topk", isDirectory: true))
    return try Data(contentsOf: directory.appendingPathComponent(name))
}

private func sourceTopKUInt32s(_ data: Data) -> [UInt32] {
    stride(from: 0, to: data.count, by: MemoryLayout<UInt32>.stride).map { offset in
        UInt32(littleEndian: data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        })
    }
}

private func sourceTopKUInt64s(_ data: Data) -> [UInt64] {
    stride(from: 0, to: data.count, by: MemoryLayout<UInt64>.stride).map { offset in
        UInt64(littleEndian: data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
        })
    }
}

private func sourceTopKSHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
