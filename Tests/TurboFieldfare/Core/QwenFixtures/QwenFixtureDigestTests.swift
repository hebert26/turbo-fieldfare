import CryptoKit
import Foundation
import Testing

@Suite struct QwenFixtureDigestTests {
    private static let expectedByteDigest = "61999aed7cf049debf980f3482af7ae91b70dcf25a3fcebdd613255c0f824dca"
    private static let expectedByteCount = 5_732_242

    @Test func frozenFixtureHasPinnedByteDigestAndSchema() throws {
        let data = try fixtureData()
        #expect(data.count == Self.expectedByteCount)
        #expect(sha256(data) == Self.expectedByteDigest)
        #expect(data.last == 0x0A, "the frozen UTF-8 fixture must retain its final newline")

        let fixture = try fixtureObject(data)
        #expect(try requiredString(fixture, at: ["schemaVersion"]) == "qwen36-tiny-v1")
        try validateFixture(fixture)
    }

    @Test func fixturePinsUpstreamIdentityAndForcedCPUFallbacks() throws {
        let fixture = try loadedFixture()
        #expect(try requiredString(fixture, at: ["metadata", "officialModelRevision"]) == "995ad96eacd98c81ed38be0c5b274b04031597b0")
        #expect(try requiredString(fixture, at: ["metadata", "transformersCommit"]) == "bd15bc95a89e728bbc1224084eb3b5829428c353")
        #expect(try requiredString(fixture, at: ["metadata", "device"]) == "cpu")
        #expect(try requiredString(fixture, at: ["metadata", "digestContract"]) == "external SHA-256 over exact frozen UTF-8 file bytes including final newline")
        #expect(try requiredString(fixture, at: ["fallbackProof", "attentionImplementation"]) == "eager")

        let environment = try requiredDictionary(fixture, at: ["fallbackProof", "requiredEnvironment"])
        #expect(environment["USE_HUB_KERNELS"] as? String == "0")
        #expect(environment["HF_HUB_OFFLINE"] as? String == "1")
        #expect(environment["TRANSFORMERS_OFFLINE"] as? String == "1")
        #expect(environment["PYTHONHASHSEED"] as? String == "0")
        #expect((try requiredArray(fixture, at: ["fallbackProof", "optionalKernelDistributionsPresent"])).isEmpty)
    }

    @Test func requiredSectionsCasesAndNumericalContractsArePresent() throws {
        let fixture = try loadedFixture()
        for section in [
            "metadata", "fallbackProof", "tinyArchitecture", "tolerances", "cases", "fullAttention",
            "causalConvolution", "linearAttention", "linearOutputGate", "moe", "greedyText", "vision", "negativeControls",
        ] {
            #expect(fixture[section] != nil, "missing required section: \(section)")
        }

        let cases = try requiredDictionary(fixture, at: ["cases"])
        for name in ["empty", "oneToken", "awkwardLength", "repeatedDecode", "splitPrefill"] {
            #expect(cases[name] != nil, "missing named execution case: \(name)")
        }
        #expect(try requiredInteger(cases, key: "empty", nested: "sequenceLength") == 0)
        #expect(try requiredInteger(cases, key: "oneToken", nested: "sequenceLength") == 1)
        #expect(try requiredInteger(cases, key: "awkwardLength", nested: "sequenceLength") == 5)
        #expect(try requiredInteger(cases, key: "repeatedDecode", nested: "steps") == 3)
        #expect(try requiredInteger(cases, key: "splitPrefill", nested: "boundary") == 3)

        let linear = try requiredDictionary(fixture, at: ["linearAttention"])
        #expect(linear["chunkSize"] as? Int == 2)
        #expect(try integerArray(linear, key: "chunkBoundaries") == [0, 2, 4, 5])
        let agreement = try requiredDictionary(linear, key: "agreement")
        for value in agreement.values {
            #expect((value as? NSNumber)?.doubleValue.isFinite == true)
            #expect((value as? NSNumber)?.doubleValue ?? .infinity < 0.000_001)
        }

        let fullAttention = try requiredDictionary(fixture, at: ["fullAttention"])
        for intermediate in ["qProjection", "qNormalized", "kNormalized", "qAfterPartialRoPE", "kAfterPartialRoPE", "preOutputGate", "outputGate", "gatedAttention", "output"] {
            #expect(fullAttention[intermediate] != nil, "missing full-attention intermediate: \(intermediate)")
        }
    }

    @Test func tensorsTolerancesAndCrossSectionReferencesAreWellFormed() throws {
        let fixture = try loadedFixture()
        try validateFixture(fixture)

        let router = try requiredDictionary(fixture, at: ["moe"])
        let indices = try tensorValues(try requiredDictionary(router, key: "top8Indices"))
        #expect(indices.count == 24)
        let weights = try tensorDoubles(try requiredDictionary(router, key: "top8NormalizedWeights"))
        #expect(weights.count == 24)
        for row in stride(from: 0, to: weights.count, by: 8) {
            #expect(abs(weights[row..<(row + 8)].reduce(0, +) - 1) < 0.000_001)
        }
        let tieContract = try requiredDictionary(router, key: "tieContract")
        #expect(try requiredString(tieContract, key: "definition").contains("torch.topk"))
        #expect(try integerArray(tieContract, key: "tiedTokenIndices") == Array(0...7))

        let greedy = try requiredDictionary(fixture, at: ["greedyText"])
        #expect(try stringArray(greedy, key: "layerSchedule") == ["linear_attention", "linear_attention", "linear_attention", "full_attention"])
        #expect((try requiredArray(greedy, key: "generatedTokens")).count == 3)
    }

    @Test func visionTowerAndMergerOracleIsCompleteAndOrdered() throws {
        let fixture = try loadedFixture()
        let vision = try requiredDictionary(fixture, at: ["vision"])
        #expect(try requiredInteger(vision, key: "towerConfig", nested: "depth") == 1)
        #expect(try requiredInteger(vision, key: "towerConfig", nested: "hiddenSize") == 16)
        #expect(try requiredInteger(vision, key: "towerConfig", nested: "outputHiddenSize") == 2_048)
        #expect(try tensorShape(try requiredDictionary(vision, key: "imageGridTHW")) == [1, 3])
        #expect(try tensorValues(try requiredDictionary(vision, key: "imageGridTHW")).compactMap { ($0 as? NSNumber)?.intValue } == [1, 4, 6])
        #expect(try tensorShape(try requiredDictionary(vision, key: "rawPatchInput")) == [24, 12])
        #expect(try tensorShape(try requiredDictionary(vision, key: "patchEmbeddings")) == [24, 16])
        #expect(try tensorShape(try requiredDictionary(vision, key: "output")) == [6, 2_048])
        #expect(try tensorShape(try requiredDictionary(try requiredDictionary(vision, key: "block"), key: "output")) == [24, 16])
        #expect(try tensorShape(try requiredDictionary(try requiredDictionary(vision, key: "merger"), key: "packedTokens")) == [6, 64])
        #expect((try requiredArray(vision, key: "padRows")).isEmpty)
        #expect(try requiredString(vision, key: "padRowsReason").contains("divisible"))

        let groups = try requiredArray(vision, key: "mergedRowPatchIndices")
        #expect(groups.count == 6)
        #expect(groups.allSatisfy { ($0 as? [Any])?.count == 4 })
        let mappings = try requiredArray(vision, key: "rawPatchRowMapping")
        #expect(mappings.count == 24)
        #expect(try requiredString(vision, key: "rawPatchOrdering").contains("block-major"))
        #expect(try requiredDouble(vision, key: "modelAgreement", nested: "blockOutputMaxAbs") == 0)
        #expect(try requiredDouble(vision, key: "modelAgreement", nested: "mergerOutputMaxAbs") == 0)

        for field in ["interpolationIndices", "interpolationWeights", "interpolatedPositionEmbeddings", "visionPositionIDs", "axialRoPECos", "axialRoPESin", "weights", "multimodalTextMRoPE"] {
            #expect(vision[field] != nil, "missing vision oracle field: \(field)")
        }
    }

    @Test func negativeControlsAndHostileMutationsAreDetected() throws {
        let fixture = try loadedFixture()
        let controls = try requiredArray(fixture, at: ["negativeControls"])
        let expectedTargets = [
            "missingGate": "fullAttention.output",
            "wrongTop8Normalization": "moe.output",
            "fp16Decay": "linearAttention.tokenAtATimeFinalState",
            "staleConvHistory": "causalConvolution.splitPrefillOutput",
            "swappedMRoPEAxes": "vision.multimodalTextMRoPE.interleavedCosSin",
        ]
        #expect(controls.count == expectedTargets.count)
        for control in controls {
            let object = try dictionary(control)
            let name = try requiredString(object, key: "name")
            #expect(object["target"] as? String == expectedTargets[name])
            #expect(object["detected"] as? Bool == true)
            #expect((try requiredDouble(object, key: "maxAbsoluteDifference")) > 0)
            #expect((try requiredString(object, key: "baselineSHA256")) != (try requiredString(object, key: "mutatedSHA256")))
        }

        var wrongRevision = fixture
        var metadata = try requiredDictionary(wrongRevision, at: ["metadata"])
        metadata["transformersCommit"] = "not-the-pinned-commit"
        wrongRevision["metadata"] = metadata
        #expect(throws: FixtureIntegrityError.self) { try validateFixture(wrongRevision) }

        var missingIntermediate = fixture
        var attention = try requiredDictionary(missingIntermediate, at: ["fullAttention"])
        attention.removeValue(forKey: "qNormalized")
        missingIntermediate["fullAttention"] = attention
        #expect(throws: FixtureIntegrityError.self) { try validateFixture(missingIntermediate) }

        let data = try fixtureData()
        var alteredBytes = data
        alteredBytes[100] ^= 1
        #expect(sha256(alteredBytes) != Self.expectedByteDigest, "a byte mutation must invalidate the frozen digest")
    }
}

private enum FixtureIntegrityError: Error {
    case invalid(String)
}

private func fixtureData() throws -> Data {
    let url = try #require(Bundle.module.url(forResource: "qwen36-tiny-fixtures", withExtension: "json"))
    return try Data(contentsOf: url)
}

private func fixtureObject(_ data: Data) throws -> [String: Any] {
    try dictionary(JSONSerialization.jsonObject(with: data))
}

private func loadedFixture() throws -> [String: Any] {
    try fixtureObject(fixtureData())
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func validateFixture(_ fixture: [String: Any]) throws {
    guard fixture["schemaVersion"] as? String == "qwen36-tiny-v1" else { throw FixtureIntegrityError.invalid("schema") }
    guard try requiredString(fixture, at: ["metadata", "officialModelRevision"]) == "995ad96eacd98c81ed38be0c5b274b04031597b0",
          try requiredString(fixture, at: ["metadata", "transformersCommit"]) == "bd15bc95a89e728bbc1224084eb3b5829428c353" else {
        throw FixtureIntegrityError.invalid("upstream identity")
    }
    for path in [["fullAttention", "qNormalized"], ["fullAttention", "output"], ["linearAttention", "chunkedFinalState"], ["moe", "top8NormalizedWeights"], ["vision", "output"]] {
        _ = try requiredValue(fixture, at: path)
    }
    try validateValue(fixture)
}

private func validateValue(_ value: Any) throws {
    if let dictionary = value as? [String: Any] {
        if dictionary["dtype"] != nil || dictionary["shape"] != nil || dictionary["values"] != nil {
            try validateTensor(dictionary)
        }
        if dictionary["absolute"] != nil || dictionary["relative"] != nil || dictionary["source"] != nil {
            try validateTolerance(dictionary)
        }
        for child in dictionary.values { try validateValue(child) }
    } else if let array = value as? [Any] {
        for child in array { try validateValue(child) }
    }
}

private func validateTensor(_ tensor: [String: Any]) throws {
    let dtype = try requiredString(tensor, key: "dtype")
    guard ["float32", "int64", "int32", "bool"].contains(dtype) else { throw FixtureIntegrityError.invalid("unsupported dtype \(dtype)") }
    let shape = try tensorShape(tensor)
    let values = try tensorValues(tensor)
    let count = try shape.reduce(1) { partial, dimension in
        guard dimension >= 0 else { throw FixtureIntegrityError.invalid("negative tensor dimension") }
        let (result, overflow) = partial.multipliedReportingOverflow(by: dimension)
        guard !overflow else { throw FixtureIntegrityError.invalid("tensor shape overflow") }
        return result
    }
    guard count == values.count else { throw FixtureIntegrityError.invalid("tensor value count") }
    for value in values {
        guard let number = value as? NSNumber, number.doubleValue.isFinite else { throw FixtureIntegrityError.invalid("non-finite tensor value") }
    }
}

private func validateTolerance(_ tolerance: [String: Any]) throws {
    let absolute = try requiredDouble(tolerance, key: "absolute")
    let relative = try requiredDouble(tolerance, key: "relative")
    guard absolute.isFinite, relative.isFinite, absolute >= 0, relative >= 0, !(try requiredString(tolerance, key: "source")).isEmpty else {
        throw FixtureIntegrityError.invalid("tolerance")
    }
}

private func tensorShape(_ tensor: [String: Any]) throws -> [Int] {
    try tensorValues(tensor, key: "shape").map { value in
        guard let number = value as? NSNumber else { throw FixtureIntegrityError.invalid("shape type") }
        return number.intValue
    }
}

private func tensorValues(_ tensor: [String: Any], key: String = "values") throws -> [Any] {
    try requiredArray(tensor, key: key)
}

private func tensorDoubles(_ tensor: [String: Any]) throws -> [Double] {
    try tensorValues(tensor).map {
        guard let number = $0 as? NSNumber else { throw FixtureIntegrityError.invalid("tensor number") }
        return number.doubleValue
    }
}

private func integerArray(_ dictionary: [String: Any], key: String) throws -> [Int] {
    try requiredArray(dictionary, key: key).map {
        guard let number = $0 as? NSNumber else { throw FixtureIntegrityError.invalid("integer array \(key)") }
        return number.intValue
    }
}

private func stringArray(_ dictionary: [String: Any], key: String) throws -> [String] {
    try requiredArray(dictionary, key: key).map {
        guard let string = $0 as? String else { throw FixtureIntegrityError.invalid("string array \(key)") }
        return string
    }
}

private func requiredValue(_ dictionary: [String: Any], at path: [String]) throws -> Any {
    var current: Any = dictionary
    for component in path {
        guard let object = current as? [String: Any], let next = object[component] else { throw FixtureIntegrityError.invalid("missing \(path.joined(separator: "."))") }
        current = next
    }
    return current
}

private func requiredDictionary(_ dictionary: [String: Any], at path: [String]) throws -> [String: Any] {
    try selfDictionary(try requiredValue(dictionary, at: path))
}

private func requiredDictionary(_ dictionary: [String: Any], key: String) throws -> [String: Any] {
    try selfDictionary(dictionary[key])
}

private func dictionary(_ value: Any) throws -> [String: Any] {
    try selfDictionary(value)
}

private func selfDictionary(_ value: Any?) throws -> [String: Any] {
    guard let dictionary = value as? [String: Any] else { throw FixtureIntegrityError.invalid("dictionary") }
    return dictionary
}

private func requiredArray(_ dictionary: [String: Any], at path: [String]) throws -> [Any] {
    guard let key = path.last else { throw FixtureIntegrityError.invalid("empty array path") }
    return try requiredArray(try requiredDictionary(dictionary, at: Array(path.dropLast())), key: key)
}

private func requiredArray(_ dictionary: [String: Any], key: String) throws -> [Any] {
    guard let array = dictionary[key] as? [Any] else { throw FixtureIntegrityError.invalid("array \(key)") }
    return array
}

private func requiredString(_ dictionary: [String: Any], at path: [String]) throws -> String {
    guard let key = path.last else { throw FixtureIntegrityError.invalid("empty string path") }
    return try requiredString(try requiredDictionary(dictionary, at: Array(path.dropLast())), key: key)
}

private func requiredString(_ dictionary: [String: Any], key: String) throws -> String {
    guard let string = dictionary[key] as? String else { throw FixtureIntegrityError.invalid("string \(key)") }
    return string
}

private func requiredDouble(_ dictionary: [String: Any], key: String, nested: String? = nil) throws -> Double {
    let source: [String: Any]
    let field: String
    if let nested {
        source = try requiredDictionary(dictionary, key: key)
        field = nested
    } else {
        source = dictionary
        field = key
    }
    guard let number = source[field] as? NSNumber else { throw FixtureIntegrityError.invalid("number \(field)") }
    return number.doubleValue
}

private func requiredInteger(_ dictionary: [String: Any], key: String, nested: String) throws -> Int {
    guard let object = dictionary[key] as? [String: Any], let number = object[nested] as? NSNumber else { throw FixtureIntegrityError.invalid("integer \(key).\(nested)") }
    return number.intValue
}
