import CoreFoundation
import CryptoKit
import Foundation
import Testing

/// Host-only checks for the independent official-code BF16 fixture. The
/// expected BF16 and FP32 expansions are literal IEEE-754 values, not values
/// read from the fixture or from candidate output.
@Suite struct OfficialBF16ReferenceTests {
    private static let transformersCommit = "bd15bc95a89e728bbc1224084eb3b5829428c353"
    private static let fixtureSHA256 =
        "4c1a57009df3e408bbc054414add6bf62e12fcf4a2f2a4161227ee2e3df7ba81"
    private static let tinyLayerOutputSHA256 =
        "9eafc561e7d9533c58e92536d487924f36e28f20451aa8be5ae6c465f620636f"
    private static let task36ReceiptSHA256 =
        "443d6a1c6b4cef456b3c16a6aacd7a53f89ecd37e308dc4145f5d0f630e8a1f7"
    private static let task36OutputSHA256 =
        "0aed36aafed14ef38c914ee9f079bd82debb29a0b3d411c7710b015104066c33"
    private static let task36RouterLogitsSHA256 =
        "52fb0aaf50363075bcbe715049a23b78a843a1dc0dcd2cce23ce6e45fc6d9170"
    private static let task36Top8 = [5, 55, 156, 145, 166, 137, 0, 4]
    // Frozen before candidate output: scalar error is 9.467346e-9; the
    // independent router-weight formula has max abs error 5.980713e-8 and
    // max relative error 1.749458e-7. At scalar 0.3655 the formula allows
    // 4.6553e-7 (~15.6 FP32 ULPs), without adding a ULP allowance.
    private static let absoluteTolerance = 1.0e-7
    private static let relativeTolerance = 1.0e-6

    @Test func bundleResourcePinsOfficialIdentityAndSeparatesAcceptedTask36Integrity() throws {
        let (data, fixture) = try Self.loadFixture()
        #expect(data.count > 0)
        #expect(Self.sha256(data) == Self.fixtureSHA256)
        #expect(try Self.string(fixture, "schemaVersion") == "official-bf16-reference-v1")

        let metadata = try Self.dictionary(fixture, "metadata")
        let provenance = try Self.dictionary(metadata, "provenance")
        #expect(try Self.string(provenance, "transformersCommit") == Self.transformersCommit)
        #expect(try Self.string(provenance, "device") == "cpu")
        #expect(try Self.integer(provenance, "seed") == 0)
        #expect(try Self.boolean(provenance, "usesRNGForValues") == false)
        #expect(try Self.string(provenance, "source").contains("synthetic BF16 bits only"))
        #expect(try Self.hex64(Self.string(provenance, "generatorSHA256")))

        let configuration = try Self.dictionary(metadata, "configuration")
        #expect(try Self.integer(configuration, "hiddenSize") == 32)
        #expect(try Self.integer(configuration, "numLayersInstantiated") == 1)
        #expect(try Self.integer(configuration, "layerIndex") == 3)
        #expect(try Self.integer(configuration, "expertCount") == 256)
        #expect(try Self.integer(configuration, "topK") == 8)
        #expect(try Self.string(configuration, "attention") == "official eager")
        #expect(try Self.string(configuration, "experts") == "official eager")

        let policy = try Self.dictionary(fixture, "comparisonPolicy")
        #expect(try Self.string(policy, "finiteComparisonFormula")
            == "abs(actual - expected) <= absoluteTolerance + relativeTolerance * abs(expected)")
        #expect(try Self.string(policy, "finiteHandling")
            .contains("reject nonfinite layer/router outputs"))
        #expect(try Self.string(policy, "ulpRule") == "no additional ULP allowance")
        #expect(try Self.double(policy, "absoluteTolerance") == Self.absoluteTolerance)
        #expect(try Self.double(policy, "relativeTolerance") == Self.relativeTolerance)
        #expect(try Self.string(policy, "toleranceStatus")
            == "frozen by Main before any candidate/full-model result; do not widen after P22")
        let rationale = try Self.string(policy, "toleranceRationale")
        #expect(rationale.contains("scalar max absolute error 9.467346e-9"))
        #expect(rationale.contains("official Top8 normalized-weight max absolute error 5.980713e-8"))
        #expect(rationale.contains("max relative error 1.749458e-7"))
        #expect(try Self.string(policy, "nonTieRouteIDs")
            == "exact ordered indices when eighth/ninth logit margin is positive")
        #expect(try Self.string(policy, "exactCutoffTie")
            .contains("compare mandatory above-cutoff IDs and tied cutoff set"))
        #expect(try Self.double(policy, "ulpAllowance") == 0)
        #expect(try Self.boolean(policy, "noFullModelLogits"))

        // This older receipt is pinned for integrity only. Its full layer output
        // and router logits must never be used as the fresh tiny calibration.
        let acceptedLayer = try Self.dictionary(fixture, "acceptedLayer")
        #expect(try Self.string(acceptedLayer, "sourceReceiptSHA256") == Self.task36ReceiptSHA256)
        #expect(try Self.boolean(acceptedLayer, "physicalSourceAuthenticityVerified") == false)
        #expect(try Self.string(acceptedLayer, "scope").contains("fixture integrity only"))
        #expect(try Self.string(acceptedLayer, "configSHA256")
            == "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99")
        #expect(try Self.string(acceptedLayer, "indexSHA256")
            == "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83")

        let oldOutput = try Self.dictionary(acceptedLayer, "output")
        #expect(try Self.intArray(oldOutput, "shape") == [1, 1, 2048])
        let oldOutputValues = try Self.floatValues(oldOutput, "values")
        #expect(oldOutputValues.count == 2048)
        #expect(try Self.string(oldOutput, "sha256FP32LE") == Self.task36OutputSHA256)
        #expect(Self.sha256(Self.fp32LittleEndianData(oldOutputValues)) == Self.task36OutputSHA256)

        let oldRouterLogits = try Self.dictionary(acceptedLayer, "routerLogits")
        #expect(try Self.intArray(oldRouterLogits, "shape") == [1, 256])
        let oldRouterValues = try Self.floatValues(oldRouterLogits, "values")
        #expect(oldRouterValues.count == 256)
        #expect(try Self.string(oldRouterLogits, "sha256FP32LE") == Self.task36RouterLogitsSHA256)
        #expect(Self.sha256(Self.fp32LittleEndianData(oldRouterValues)) == Self.task36RouterLogitsSHA256)
        #expect(try Self.intArray(acceptedLayer, "actualTop8Indices") == Self.task36Top8)

        let cases = try Self.dictionary(fixture, "cases")
        let tiny = try Self.dictionary(cases, "tinyOutput")
        let forward = try Self.dictionary(tiny, "forward")
        let tinyOutput = try Self.dictionary(forward, "output")
        #expect(try Self.intArray(tinyOutput, "shape") == [1, 3, 32])
        #expect(try Self.floatValues(tinyOutput, "values").count == 96)
        #expect(try Self.intArray(try Self.dictionary(forward, "routerLogits"), "shape") == [3, 256])
        #expect(try Self.floatValues(try Self.dictionary(forward, "routerLogits"), "values").count == 768)
        #expect(try Self.string(tinyOutput, "sha256FP32LE") == Self.tinyLayerOutputSHA256)
        #expect(Self.sha256(Self.fp32LittleEndianData(try Self.floatValues(tinyOutput, "values")))
            == Self.tinyLayerOutputSHA256)

    }

    @Test func bundleBF16ReadsMatchLiteralBitsAndExactFP32Expansions() throws {
        struct Expected {
            let name: String
            let bf16: UInt16
            let fp32: UInt32
            let classification: String
            let negativeZero: Bool
            let outsideFiniteFloat16Range: Bool
        }
        let expected: [Expected] = [
            .init(name: "positiveZero", bf16: 0x0000, fp32: 0x0000_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: false),
            .init(name: "negativeZero", bf16: 0x8000, fp32: 0x8000_0000,
                  classification: "finite", negativeZero: true, outsideFiniteFloat16Range: false),
            .init(name: "positiveSubnormal", bf16: 0x0001, fp32: 0x0001_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: true),
            .init(name: "negativeSubnormal", bf16: 0x8001, fp32: 0x8001_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: true),
            .init(name: "largestSubnormal", bf16: 0x007f, fp32: 0x007f_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: true),
            .init(name: "smallestNormal", bf16: 0x0080, fp32: 0x0080_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: true),
            .init(name: "ordinaryOne", bf16: 0x3f80, fp32: 0x3f80_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: false),
            .init(name: "ordinaryNegative", bf16: 0xc020, fp32: 0xc020_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: false),
            .init(name: "roundingBoundaryAboveOne", bf16: 0x3f81, fp32: 0x3f81_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: false),
            .init(name: "largestFinite", bf16: 0x7f7f, fp32: 0x7f7f_0000,
                  classification: "finite", negativeZero: false, outsideFiniteFloat16Range: true),
            .init(name: "positiveInfinity", bf16: 0x7f80, fp32: 0x7f80_0000,
                  classification: "infinity", negativeZero: false, outsideFiniteFloat16Range: false),
            .init(name: "negativeInfinity", bf16: 0xff80, fp32: 0xff80_0000,
                  classification: "infinity", negativeZero: false, outsideFiniteFloat16Range: false),
            .init(name: "quietNaN", bf16: 0x7fc1, fp32: 0x7fc1_0000,
                  classification: "nan", negativeZero: false, outsideFiniteFloat16Range: false),
        ]

        let (_, fixture) = try Self.loadFixture()
        let cases = try Self.dictionary(fixture, "cases")
        let records = try Self.array(cases, "bf16Reads")
        #expect(records.count == expected.count)
        let recordsByName = try Dictionary(uniqueKeysWithValues: records.map { record in
            let object = try Self.object(record)
            return (try Self.string(object, "name"), object)
        })
        #expect(Set(recordsByName.keys) == Set(expected.map(\.name)))

        for vector in expected {
            let record = try #require(recordsByName[vector.name])
            let expandedBits = UInt32(vector.bf16) << 16
            #expect(expandedBits == vector.fp32)
            #expect(try Self.string(record, "bf16BitsHex") == String(format: "%04x", vector.bf16))
            #expect(try Self.string(record, "bf16BytesLEHex") == Self.littleEndianHex(vector.bf16))
            #expect(try Self.string(record, "fp32BitsLEHex") == Self.littleEndianHex(vector.fp32))
            #expect(try Self.string(record, "classification") == vector.classification)
            #expect(try Self.boolean(record, "negativeZero") == vector.negativeZero)
            #expect(try Self.boolean(record, "outsideFiniteFloat16Range")
                == vector.outsideFiniteFloat16Range)

            if vector.classification == "finite" {
                let value = try #require(record["fp32Value"] as? NSNumber)
                #expect(value.doubleValue == Double(Float(bitPattern: vector.fp32)))
            } else {
                #expect(record["fp32Value"] is NSNull)
            }
        }
    }

    @Test func scalarMLPMatchesLiteralBF16MathAndRejectsGateUpSwap() throws {
        let (_, fixture) = try Self.loadFixture()
        let cases = try Self.dictionary(fixture, "cases")
        let scalar = try Self.dictionary(cases, "scalarMLP")
        #expect(try Self.string(scalar, "name") == "independentSparseSiLU")
        #expect(try Self.string(scalar, "officialFunction") == "Qwen3_5MoeMLP.forward")

        // These are literal independent inputs, not values inferred from fixture fields.
        let x = 1.0
        let gate = 1.0
        let up = 2.0
        let down = 0.25
        #expect(try Self.double(scalar, "gate") == gate)
        #expect(try Self.double(scalar, "up") == up)
        #expect(try Self.double(scalar, "down") == down)

        let input = try Self.dictionary(scalar, "inputBF16")
        #expect(Self.isValidBF16Record(input, expectedShape: [1, 1, 32]))
        let inputBytes = try Self.bf16Data(input)
        #expect(Array(inputBytes.prefix(2)) == [0x80, 0x3f]) // BF16 1.0, little-endian
        #expect(inputBytes.dropFirst(2).allSatisfy { $0 == 0 })

        let weightRecords = try Self.dictionary(scalar, "weightsBF16")
        let expectedWeights: [(name: String, shape: [Int], prefix: [UInt8])] = [
            ("gate_proj.weight", [1, 32], [0x80, 0x3f]), // BF16 1.0
            ("up_proj.weight", [1, 32], [0x00, 0x40]), // BF16 2.0
            ("down_proj.weight", [32, 1], [0x80, 0x3e]), // BF16 0.25
        ]
        for expectedWeight in expectedWeights {
            let record = try Self.dictionary(weightRecords, expectedWeight.name)
            #expect(Self.isValidBF16Record(record, expectedShape: expectedWeight.shape))
            let bytes = try Self.bf16Data(record)
            #expect(Array(bytes.prefix(2)) == expectedWeight.prefix)
            #expect(bytes.dropFirst(2).allSatisfy { $0 == 0 })
        }

        let independent = (1.0 / (1.0 + exp(-x * gate))) * up * down
        let reportedIndependent = try Self.double(scalar, "independentScalarFP64")
        #expect(abs(reportedIndependent - independent) < 1e-15)
        #expect(try Self.string(scalar, "independentFormula")
            == "(1/(1+exp(-1))) * 2 * 0.25")

        let officialRecord = try Self.dictionary(scalar, "officialOutput")
        #expect(Self.isValidFP32Record(officialRecord))
        #expect(try Self.intArray(officialRecord, "shape") == [1, 1, 32])
        let official = try Self.doubleValues(officialRecord, "values")
        #expect(official.count == 32)
        let expectedOutput = [independent] + Array(repeating: 0.0, count: 31)
        #expect(Self.matches(
            actual: official, actualShape: [1, 1, 32],
            expected: expectedOutput, expectedShape: [1, 1, 32],
            absoluteTolerance: Self.absoluteTolerance, relativeTolerance: Self.relativeTolerance))
        #expect(abs(abs(official[0] - independent) - 9.467346184788283e-09) < 1e-15)

        let swapped = ((up * x) / (1.0 + exp(-up * x))) * gate * down
        let control = try Self.dictionary(scalar, "negativeControl")
        #expect(try Self.string(control, "mutation") == "swap gate and up, keep down fixed")
        #expect(abs((swapped - independent)
            - (try Self.double(control, "differenceFromOriginal"))) < 1e-15)
        #expect(abs(swapped - independent) > 0.07)
        #expect(!Self.matches(
            actual: [swapped], actualShape: [1],
            expected: [independent], expectedShape: [1],
            absoluteTolerance: Self.absoluteTolerance, relativeTolerance: Self.relativeTolerance))

        let allowed = Self.absoluteTolerance + Self.relativeTolerance * abs(independent)
        #expect(!Self.matches(
            actual: [independent + 2 * allowed], actualShape: [1],
            expected: [independent], expectedShape: [1],
            absoluteTolerance: Self.absoluteTolerance, relativeTolerance: Self.relativeTolerance))
    }

    @Test func routingUsesExactPositiveCutoffIDsAndSetBasedTieWithIndependentWeights() throws {
        let (_, fixture) = try Self.loadFixture()
        let cases = try Self.dictionary(fixture, "cases")
        let routes = try Self.array(cases, "routing")
        let byName = try Dictionary(uniqueKeysWithValues: routes.map { route in
            let object = try Self.object(route)
            return (try Self.string(object, "name"), object)
        })
        #expect(Set(byName.keys) == Set(["separated", "closeCutoff", "exactTie"]))

        // Independent Swift-Double softmax over the literal selected logits.
        // The selected eight scores are [8, 7, 6, 5, 4, 3, 2, 1] in all
        // three cases, including either legal exact-cutoff tie selection. The
        // measured Torch CPU tie ordering below is provenance, not contract.
        let expectedWeights = Self.normalizedSoftmax([8, 7, 6, 5, 4, 3, 2, 1])
        for (name, expectedMargin) in [("separated", 1.0), ("closeCutoff", 0.00390625)] {
            let route = try #require(byName[name])
            let ids = try Self.intArray(route, "actualTop8Indices")
            let weightsRecord = try Self.dictionary(route, "actualTop8NormalizedWeights")
            #expect(Self.isValidFP32Record(weightsRecord))
            #expect(try Self.intArray(weightsRecord, "shape") == [1, 8])
            let weights = try Self.doubleValues(weightsRecord, "values")
            #expect(try Self.double(route, "eighthNinthLogitMargin") == expectedMargin)
            if name == "separated" {
                let absoluteErrors = zip(weights, expectedWeights).map { pair in abs(pair.0 - pair.1) }
                let relativeErrors = zip(weights, expectedWeights).map { pair in
                    abs(pair.0 - pair.1) / abs(pair.1)
                }
                #expect(abs((absoluteErrors.max() ?? .infinity) - 5.9807129138000903e-8) < 1e-15)
                #expect(abs((relativeErrors.max() ?? .infinity) - 1.7494579539153569e-7) < 1e-15)
                let sumResidual = weights.reduce(0, +) - 1.0
                #expect(abs(sumResidual - (-9.2375557869672775e-8)) < 1e-15)
            }
            #expect(try Self.intArray(route, "tiedCutoffExpertIDs") == [7])
            #expect(Self.acceptsRouteAndWeights(
                actualIDs: ids, actualWeights: weights, cutoffMargin: expectedMargin,
                exactPositiveMarginIDs: Array(0..<8), mandatoryAboveCutoffIDs: [],
                tiedCutoffIDs: [7], expectedWeights: expectedWeights))

            var wrongIDs = ids
            wrongIDs[7] = 8
            #expect(!Self.acceptsRouteAndWeights(
                actualIDs: wrongIDs, actualWeights: weights, cutoffMargin: expectedMargin,
                exactPositiveMarginIDs: Array(0..<8), mandatoryAboveCutoffIDs: [],
                tiedCutoffIDs: [7], expectedWeights: expectedWeights))
            var wrongWeights = weights
            wrongWeights[0] += 1e-4
            #expect(!Self.acceptsRouteAndWeights(
                actualIDs: ids, actualWeights: wrongWeights, cutoffMargin: expectedMargin,
                exactPositiveMarginIDs: Array(0..<8), mandatoryAboveCutoffIDs: [],
                tiedCutoffIDs: [7], expectedWeights: expectedWeights))
        }

        let tied = try #require(byName["exactTie"])
        let ids = try Self.intArray(tied, "actualTop8Indices")
        let tieWeightsRecord = try Self.dictionary(tied, "actualTop8NormalizedWeights")
        #expect(Self.isValidFP32Record(tieWeightsRecord))
        #expect(try Self.intArray(tieWeightsRecord, "shape") == [1, 8])
        let weights = try Self.doubleValues(tieWeightsRecord, "values")
        #expect(try Self.intArray(tied, "tiedCutoffExpertIDs") == [7, 8])
        #expect(try Self.double(tied, "eighthNinthLogitMargin") == 0)
        #expect(Self.acceptsRouteAndWeights(
            actualIDs: ids, actualWeights: weights, cutoffMargin: 0,
            exactPositiveMarginIDs: nil, mandatoryAboveCutoffIDs: Set(0...6),
            tiedCutoffIDs: [7, 8], expectedWeights: expectedWeights))

        var wrongTieSet = ids
        let tiedIDs: Set<Int> = [7, 8]
        let selectedTiePosition = try #require(ids.firstIndex { tiedIDs.contains($0) })
        wrongTieSet[selectedTiePosition] = 9
        #expect(!Self.acceptsRouteAndWeights(
            actualIDs: wrongTieSet, actualWeights: weights, cutoffMargin: 0,
            exactPositiveMarginIDs: nil, mandatoryAboveCutoffIDs: Set(0...6),
            tiedCutoffIDs: [7, 8], expectedWeights: expectedWeights))
        var wrongTieWeights = weights
        wrongTieWeights[0] += 1e-4
        #expect(!Self.acceptsRouteAndWeights(
            actualIDs: ids, actualWeights: wrongTieWeights, cutoffMargin: 0,
            exactPositiveMarginIDs: nil, mandatoryAboveCutoffIDs: Set(0...6),
            tiedCutoffIDs: [7, 8], expectedWeights: expectedWeights))
        #expect(try Self.string(tied, "tieOrderingClaim")
            .contains("not a stable torch.topk guarantee"))
    }

    @Test func platformTieObservationIsSeparateFromLowerIDDiagnostic() throws {
        let (_, fixture) = try Self.loadFixture()
        let cases = try Self.dictionary(fixture, "cases")
        let routeObjects = try Self.array(cases, "routing").map { try Self.object($0) }
        let tied = try #require(routeObjects.first { (try? Self.string($0, "name")) == "exactTie" })
        let observed = try Self.intArray(tied, "actualTop8Indices")
        let lowerIDDiagnostic = try Self.intArray(tied, "lowerIDTiePolicyTop8")
        #expect(lowerIDDiagnostic == Array(0..<8))
        #expect(try Self.boolean(tied, "actualTorchTopKAgreesWithLowerIDPolicy")
            == (observed == lowerIDDiagnostic))
        #expect(try Self.string(tied, "tieOrderingClaim")
            .contains("not a stable torch.topk guarantee"))
    }

    @Test func tinyFixtureHasPositivePerTokenMarginsUniqueRoutesAndDistinctExpertSlices() throws {
        let (_, fixture) = try Self.loadFixture()
        let cases = try Self.dictionary(fixture, "cases")
        let tiny = try Self.dictionary(cases, "tinyOutput")
        let weights = try Self.dictionary(tiny, "syntheticBF16Weights")
        #expect(weights["mlp.experts.gate_up_proj"] != nil)
        #expect(weights["mlp.experts.down_proj"] != nil)
        for (name, value) in weights {
            let record = try Self.object(value)
            let expectedShape: [Int]? = switch name {
            case "mlp.experts.gate_up_proj": [256, 16, 32]
            case "mlp.experts.down_proj": [256, 32, 8]
            default: nil
            }
            #expect(Self.isValidBF16Record(record, expectedShape: expectedShape))
        }

        let forward = try Self.dictionary(tiny, "forward")
        for key in ["output", "routerLogits", "actualTop8NormalizedWeights",
                    "sharedExpertOutput", "sharedExpertGateLogits", "routerInputFirstCoordinate"] {
            let record = try Self.dictionary(forward, key)
            #expect(Self.isValidFP32Record(record), "invalid FP32 fixture record: \(key)")
        }
        #expect(try Self.intArray(try Self.dictionary(forward, "output"), "shape") == [1, 3, 32])
        #expect(try Self.intArray(try Self.dictionary(forward, "routerLogits"), "shape") == [3, 256])
        #expect(try Self.string(try Self.dictionary(forward, "output"), "sha256FP32LE")
            == Self.tinyLayerOutputSHA256)

        let routeIDs = try Self.intMatrix(forward, "actualTop8Indices")
        let expectedRows = [Array(0..<8), Array((248...255).reversed()), Array((248...255).reversed())]
        #expect(routeIDs == expectedRows)
        #expect(routeIDs.count == 3)
        #expect(routeIDs.allSatisfy { $0.count == 8 && Set($0).count == 8
            && $0.allSatisfy((0..<256).contains) })

        let margins = try Self.doubleArray(forward, "perTokenCutoffLogitMargins")
        #expect(margins == [0.0007054060697555542, 0.0018593072891235352, 0.0036498308181762695])
        #expect(margins.count == 3 && margins.allSatisfy { $0.isFinite && $0 > 0 })

        let routerInput = try Self.dictionary(forward, "routerInputFirstCoordinate")
        #expect(try Self.intArray(routerInput, "shape") == [3])
        let routerInputValues = try Self.doubleValues(routerInput, "values")
        #expect(routerInputValues.count == 3 && routerInputValues.allSatisfy { $0.isFinite && $0 != 0 })

        let routeWeights = try Self.dictionary(forward, "actualTop8NormalizedWeights")
        #expect(try Self.intArray(routeWeights, "shape") == [3, 8])
        let weightValues = try Self.floatValues(routeWeights, "values")
        #expect(weightValues.count == 3 * 8)
        for row in 0..<3 {
            let rowWeights = Array(weightValues[(row * 8)..<(row * 8 + 8)])
            #expect(Set(rowWeights).count == 8, "tiny-layer selected weights should be nonuniform")
            let rowSum = rowWeights.reduce(0.0) { $0 + Double($1) }
            #expect(Self.matches(
                actual: [rowSum], actualShape: [1], expected: [1.0], expectedShape: [1],
                absoluteTolerance: Self.absoluteTolerance, relativeTolerance: Self.relativeTolerance))
        }

        let coverage = try Self.dictionary(forward, "expertCoverage")
        #expect(try Self.integer(coverage, "availableExpertCount") == 256)
        #expect(try Self.intMatrix(coverage, "selectedPerToken") == routeIDs)
        let selectedIDs = Set(routeIDs.flatMap { $0 })
        #expect(try Self.integer(coverage, "selectedUniqueCount") == 16)
        #expect(selectedIDs.count == 16)
        #expect(try Self.intArray(forward, "executedExpertIDs") == selectedIDs.sorted())

        let declaredHashes = try Self.dictionary(coverage, "selectedExpertBF16SliceSHA256")
        #expect(Set(declaredHashes.keys) == Set(selectedIDs.map { String($0) }))
        let gateUpRecord = try Self.dictionary(weights, "mlp.experts.gate_up_proj")
        let downRecord = try Self.dictionary(weights, "mlp.experts.down_proj")
        let gateUpBytes = try Self.bf16Data(gateUpRecord)
        let downBytes = try Self.bf16Data(downRecord)
        guard gateUpBytes.count == 256 * 16 * 32 * 2,
              downBytes.count == 256 * 32 * 8 * 2 else {
            Issue.record("packed expert BF16 byte counts do not match declared tensor shapes")
            return
        }
        let gateSliceByteCount = 8 * 32 * 2
        let packedGateUpStride = 2 * gateSliceByteCount
        let downSliceByteCount = 32 * 8 * 2
        var gateSlices = Set<Data>()
        var upSlices = Set<Data>()
        var downSlices = Set<Data>()
        for expert in selectedIDs.sorted() {
            let gateOffset = expert * packedGateUpStride
            let gate = gateUpBytes.subdata(in: gateOffset..<(gateOffset + gateSliceByteCount))
            let upOffset = gateOffset + gateSliceByteCount
            let up = gateUpBytes.subdata(in: upOffset..<(upOffset + gateSliceByteCount))
            let downOffset = expert * downSliceByteCount
            let down = downBytes.subdata(in: downOffset..<(downOffset + downSliceByteCount))
            #expect(gate != up)
            #expect(gateSlices.insert(gate).inserted)
            #expect(upSlices.insert(up).inserted)
            #expect(downSlices.insert(down).inserted)

            let expertDeclaration = try Self.dictionary(declaredHashes, String(expert))
            #expect(try Self.string(expertDeclaration, "gate") == Self.sha256(gate))
            #expect(try Self.string(expertDeclaration, "up") == Self.sha256(up))
            #expect(try Self.string(expertDeclaration, "down") == Self.sha256(down))
        }
        #expect(gateSlices.count == selectedIDs.count)
        #expect(upSlices.count == selectedIDs.count)
        #expect(downSlices.count == selectedIDs.count)
    }

    @Test func independentComparatorAcceptsInclusiveAbsoluteRelativeBoundary() {
        #expect(Self.matches(
            actual: [4.75], actualShape: [1],
            expected: [4.0], expectedShape: [1],
            absoluteTolerance: 0.25, relativeTolerance: 0.125))
        #expect(Self.matches(
            actual: [0.125], actualShape: [1],
            expected: [0.0], expectedShape: [1],
            absoluteTolerance: 0.125, relativeTolerance: 0.5))
    }

    @Test func independentComparatorRejectsMutationsOutsideAbsoluteRelativeBoundary() {
        #expect(!Self.matches(
            actual: [4.7501], actualShape: [1],
            expected: [4.0], expectedShape: [1],
            absoluteTolerance: 0.25, relativeTolerance: 0.125))
        #expect(!Self.matches(
            actual: [0.1251], actualShape: [1],
            expected: [0.0], expectedShape: [1],
            absoluteTolerance: 0.125, relativeTolerance: 0.5))
    }

    @Test func independentComparatorRejectsShapeCountNonFiniteAndInvalidLimits() {
        let atol = Self.absoluteTolerance
        let rtol = Self.relativeTolerance
        #expect(!Self.matches(actual: [1, 2], actualShape: [2], expected: [1, 2],
                              expectedShape: [1, 2], absoluteTolerance: atol, relativeTolerance: rtol))
        #expect(!Self.matches(actual: [1], actualShape: [2], expected: [1],
                              expectedShape: [2], absoluteTolerance: atol, relativeTolerance: rtol))
        #expect(!Self.matches(actual: [.nan], actualShape: [1], expected: [0],
                              expectedShape: [1], absoluteTolerance: atol, relativeTolerance: rtol))
        #expect(!Self.matches(actual: [0], actualShape: [1], expected: [.infinity],
                              expectedShape: [1], absoluteTolerance: atol, relativeTolerance: rtol))
        #expect(!Self.matches(actual: [0], actualShape: [1], expected: [0],
                              expectedShape: [1], absoluteTolerance: .infinity, relativeTolerance: rtol))
        #expect(!Self.matches(actual: [0], actualShape: [1], expected: [0],
                              expectedShape: [1], absoluteTolerance: atol, relativeTolerance: -rtol))
        #expect(!Self.matches(actual: [1], actualShape: [Int.max, 2], expected: [1],
                              expectedShape: [Int.max, 2], absoluteTolerance: atol, relativeTolerance: rtol))
    }

    @Test func deliberateGateUpSharedBranchAndNormalizationMutationsInvalidateFixtureRecords() throws {
        let (_, fixture) = try Self.loadFixture()
        let cases = try Self.dictionary(fixture, "cases")
        let tiny = try Self.dictionary(cases, "tinyOutput")
        let weights = try Self.dictionary(tiny, "syntheticBF16Weights")
        let gateUp = try Self.dictionary(weights, "mlp.experts.gate_up_proj")
        #expect(Self.isValidBF16Record(gateUp, expectedShape: [256, 16, 32]))

        var transposedLayout = gateUp
        transposedLayout["shape"] = [256, 32, 16]
        #expect(!Self.isValidBF16Record(transposedLayout, expectedShape: [256, 16, 32]))

        var corruptedGateUpBytes = gateUp
        let gateUpBytes = try #require(Data(base64Encoded: Self.string(gateUp, "bytesBase64")))
        var changedBytes = gateUpBytes
        changedBytes[changedBytes.startIndex] ^= 1
        corruptedGateUpBytes["bytesBase64"] = changedBytes.base64EncodedString()
        #expect(!Self.isValidBF16Record(corruptedGateUpBytes, expectedShape: [256, 16, 32]))

        let forward = try Self.dictionary(tiny, "forward")
        for key in ["sharedExpertOutput", "sharedExpertGateLogits", "actualTop8NormalizedWeights"] {
            let original = try Self.dictionary(forward, key)
            #expect(Self.isValidFP32Record(original))
            var mutation = original
            var values = try Self.floatValues(original, "values")
            try #require(!values.isEmpty)
            values[0] += 0.25
            mutation["values"] = values
            #expect(!Self.isValidFP32Record(mutation), "mutation escaped digest check: \(key)")
        }
    }

    private static func loadFixture() throws -> (Data, [String: Any]) {
        let url = try #require(Bundle.module.url(
            forResource: "official-bf16-reference-cases", withExtension: "json"))
        let data = try Data(contentsOf: url)
        let value = try JSONSerialization.jsonObject(with: data)
        return (data, try object(value))
    }

    private static func acceptsRouteAndWeights(
        actualIDs: [Int], actualWeights: [Double], cutoffMargin: Double,
        exactPositiveMarginIDs: [Int]?, mandatoryAboveCutoffIDs: Set<Int>,
        tiedCutoffIDs: Set<Int>, expectedWeights: [Double]
    ) -> Bool {
        guard cutoffMargin.isFinite, cutoffMargin >= 0,
              actualIDs.count == 8, Set(actualIDs).count == 8,
              actualIDs.allSatisfy((0..<256).contains),
              actualWeights.count == 8, expectedWeights.count == 8,
              actualWeights.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 })
        else { return false }

        if cutoffMargin > 0 {
            guard let exactPositiveMarginIDs,
                  exactPositiveMarginIDs.count == 8,
                  actualIDs == exactPositiveMarginIDs,
                  mandatoryAboveCutoffIDs.isEmpty
            else { return false }
        } else {
            let selected = Set(actualIDs)
            let allowed = mandatoryAboveCutoffIDs.union(tiedCutoffIDs)
            guard exactPositiveMarginIDs == nil,
                  !mandatoryAboveCutoffIDs.isEmpty, !tiedCutoffIDs.isEmpty,
                  mandatoryAboveCutoffIDs.isDisjoint(with: tiedCutoffIDs),
                  mandatoryAboveCutoffIDs.isSubset(of: selected),
                  selected.isSubset(of: allowed),
                  selected.intersection(tiedCutoffIDs).count == 1
            else { return false }
        }
        return matches(
            actual: actualWeights, actualShape: [1, 8],
            expected: expectedWeights, expectedShape: [1, 8],
            absoluteTolerance: Self.absoluteTolerance, relativeTolerance: Self.relativeTolerance)
    }

    private static func normalizedSoftmax(_ logits: [Double]) -> [Double] {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite),
              let maximum = logits.max() else { return [] }
        let exponentials = logits.map { exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        guard denominator.isFinite, denominator > 0 else { return [] }
        return exponentials.map { $0 / denominator }
    }

    private static func matches(
        actual: [Double], actualShape: [Int],
        expected: [Double], expectedShape: [Int],
        absoluteTolerance: Double, relativeTolerance: Double
    ) -> Bool {
        guard actualShape == expectedShape,
              elementCount(actualShape) == actual.count,
              elementCount(expectedShape) == expected.count,
              absoluteTolerance.isFinite, relativeTolerance.isFinite,
              absoluteTolerance >= 0, relativeTolerance >= 0,
              actual.allSatisfy(\.isFinite), expected.allSatisfy(\.isFinite)
        else { return false }

        for (actualValue, expectedValue) in zip(actual, expected) {
            let difference = abs(actualValue - expectedValue)
            let allowance = absoluteTolerance + relativeTolerance * abs(expectedValue)
            if difference > allowance { return false }
        }
        return true
    }

    private static func bf16Data(_ record: [String: Any]) throws -> Data {
        guard let data = Data(base64Encoded: try Self.string(record, "bytesBase64")) else {
            throw FixtureError.expectedArray("bytesBase64")
        }
        return data
    }

    private static func isValidBF16Record(
        _ record: [String: Any], expectedShape: [Int]? = nil
    ) -> Bool {
        guard (try? string(record, "dtype")) == "bfloat16",
              (try? string(record, "byteOrder")) == "little",
              let shape = try? intArray(record, "shape"),
              expectedShape == nil || shape == expectedShape,
              let count = elementCount(shape), count <= Int.max / 2,
              let encoded = try? string(record, "bytesBase64"),
              let bytes = Data(base64Encoded: encoded),
              bytes.count == count * 2,
              let claimedDigest = try? string(record, "sha256Bytes"),
              sha256(bytes) == claimedDigest
        else { return false }
        return true
    }

    private static func isValidFP32Record(_ record: [String: Any]) -> Bool {
        guard let shape = try? intArray(record, "shape"),
              let count = elementCount(shape),
              let values = try? floatValues(record, "values"),
              values.count == count, values.allSatisfy(\.isFinite),
              let claimedDigest = try? string(record, "sha256FP32LE"),
              sha256(fp32LittleEndianData(values)) == claimedDigest
        else { return false }
        return true
    }

    private static func fp32LittleEndianData(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * MemoryLayout<UInt32>.size)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func littleEndianHex(_ value: UInt16) -> String {
        String(format: "%02x%02x", value & 0x00ff, value >> 8)
    }

    private static func littleEndianHex(_ value: UInt32) -> String {
        String(format: "%02x%02x%02x%02x", value & 0x000000ff,
               (value >> 8) & 0xff, (value >> 16) & 0xff, value >> 24)
    }

    private static func hex64(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) }
    }

    private static func elementCount(_ shape: [Int]) -> Int? {
        var count = 1
        for dimension in shape {
            guard dimension >= 0 else { return nil }
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else { return nil }
            count = next
        }
        return count
    }

    private static func object(_ value: Any) throws -> [String: Any] {
        guard let result = value as? [String: Any] else { throw FixtureError.expectedObject }
        return result
    }

    private static func dictionary(_ object: [String: Any], _ key: String) throws -> [String: Any] {
        try Self.object(try #require(object[key]))
    }

    private static func array(_ object: [String: Any], _ key: String) throws -> [Any] {
        guard let values = object[key] as? [Any] else { throw FixtureError.expectedArray(key) }
        return values
    }

    private static func string(_ object: [String: Any], _ key: String) throws -> String {
        guard let value = object[key] as? String else { throw FixtureError.expectedString(key) }
        return value
    }

    private static func integer(_ object: [String: Any], _ key: String) throws -> Int {
        guard let number = object[key] as? NSNumber,
              number.doubleValue.isFinite,
              number.doubleValue.rounded(.towardZero) == number.doubleValue else {
            throw FixtureError.expectedInteger(key)
        }
        return number.intValue
    }

    private static func double(_ object: [String: Any], _ key: String) throws -> Double {
        guard let number = object[key] as? NSNumber, number.doubleValue.isFinite else {
            throw FixtureError.expectedNumber(key)
        }
        return number.doubleValue
    }

    private static func boolean(_ object: [String: Any], _ key: String) throws -> Bool {
        guard let number = object[key] as? NSNumber,
              CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw FixtureError.expectedBoolean(key)
        }
        return number.boolValue
    }

    private static func intArray(_ object: [String: Any], _ key: String) throws -> [Int] {
        guard let values = object[key] as? [NSNumber] else { throw FixtureError.expectedArray(key) }
        return try values.map { number in
            guard number.doubleValue.isFinite,
                  number.doubleValue.rounded(.towardZero) == number.doubleValue else {
                throw FixtureError.expectedInteger(key)
            }
            return number.intValue
        }
    }

    private static func intMatrix(_ object: [String: Any], _ key: String) throws -> [[Int]] {
        guard let rows = object[key] as? [Any] else { throw FixtureError.expectedArray(key) }
        return try rows.map { row in
            let wrapper = ["values": row]
            return try intArray(wrapper, "values")
        }
    }

    private static func floatValues(_ object: [String: Any], _ key: String) throws -> [Float] {
        guard let values = object[key] as? [NSNumber] else { throw FixtureError.expectedArray(key) }
        return values.map(\.floatValue)
    }

    private static func doubleValues(_ object: [String: Any], _ key: String) throws -> [Double] {
        guard let values = object[key] as? [NSNumber],
              values.allSatisfy({ $0.doubleValue.isFinite }) else {
            throw FixtureError.expectedArray(key)
        }
        return values.map(\.doubleValue)
    }

    private static func doubleArray(_ object: [String: Any], _ key: String) throws -> [Double] {
        try doubleValues(object, key)
    }

    private enum FixtureError: Error {
        case expectedObject
        case expectedString(String)
        case expectedInteger(String)
        case expectedNumber(String)
        case expectedBoolean(String)
        case expectedArray(String)
    }
}
