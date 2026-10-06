import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenFullAttentionTests {
    private func fixtureConfiguration() throws -> QwenFullAttentionConfiguration {
        try QwenFullAttentionConfiguration(
            queryHeadCount: 2, keyValueHeadCount: 1, headDimension: 16, rotaryDimension: 12)
    }

    @Test func configurationKeepsOfficialAndFixtureRotaryGeometrySeparate() throws {
        let official = try QwenFullAttentionConfiguration.official()
        #expect(official.queryHeadCount == 16)
        #expect(official.keyValueHeadCount == 2)
        #expect(official.headDimension == 256)
        #expect(official.rotaryDimension == 64)
        #expect(official.theta == 10_000_000)
        let fixture = try fixtureConfiguration()
        #expect(fixture.rotaryDimension == 12)
        #expect(fixture.headDimension == 16)
        #expect(throws: QwenFullAttentionError.invalidConfiguration(
            field: "rotaryDimension", value: 15
        )) {
            try QwenFullAttentionConfiguration(
                queryHeadCount: 16, keyValueHeadCount: 2, headDimension: 256, rotaryDimension: 15)
        }
    }

    @Test func evaluatesQwenOrderAndMatchesIndependentSmallReference() throws {
        let positions = [0, 1]
        let fixtureConfiguration = try fixtureConfiguration()
        let queryAndGate = syntheticQueryAndGate(tokens: positions.count)
        let keys = sequence(count: positions.count * 16, seed: -0.3)
        let values = sequence(count: positions.count * 16, seed: 0.4)
        let qWeights = sequence(count: 16, seed: 0.05)
        let kWeights = sequence(count: 16, seed: -0.08)
        let outputProjection = sequence(count: 3 * 32, seed: 0.12)
        let result = try QwenFullAttention.evaluate(
            configuration: fixtureConfiguration,
            queryAndGateProjection: queryAndGate,
            keyProjection: keys,
            valueProjection: values,
            queryNormWeight: qWeights,
            keyNormWeight: kWeights,
            positions: positions,
            outputProjection: outputProjection,
            outputDimension: 3)

        let expected = independentAttention(
            queryAndGate: queryAndGate, keys: keys, values: values, qWeights: qWeights,
            kWeights: kWeights, positions: positions, outputProjection: outputProjection)
        expectClose(result.queryProjection, expected.query, tolerance: 1e-5)
        expectClose(result.normalizedQuery, expected.normalizedQuery, tolerance: 1e-5)
        expectClose(result.normalizedKey, expected.normalizedKey, tolerance: 1e-5)
        expectClose(result.rotatedQuery, expected.rotatedQuery, tolerance: 1e-5)
        expectClose(result.rotatedKey, expected.rotatedKey, tolerance: 1e-5)
        expectClose(result.outputGate, expected.gate, tolerance: 1e-5)
        expectClose(result.output, expected.output, tolerance: 1e-5)

        // Negative controls: each is a plausible ordering/geometry error and must diverge.
        #expect(maxDifference(result.rotatedQuery, adjacentPairRotation(expected.normalizedQuery, positions: positions)) > 1e-4)
        #expect(maxDifference(result.rotatedQuery, wrongSpanRotation(expected.normalizedQuery, positions: positions)) > 1e-4)
        #expect(maxDifference(result.output, expected.outputWithoutGate) > 1e-4)
        #expect(maxDifference(result.output, expected.outputWithGateAfterProjection) > 1e-4)
    }

    @Test func combinedQwenFullAttentionFrozenP3OneAndFiveTokenIntermediatesAndCausalWeights() throws {
        let url = try #require(Bundle.module.url(forResource: "qwen36-tiny-fixtures", withExtension: "json"))
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let fixture = try #require(object as? [String: Any])
        let tolerances = try #require(fixture["tolerances"] as? [String: Any])
        let fp32 = try #require(tolerances["fp32"] as? [String: Any])
        #expect((fp32["absolute"] as? NSNumber)?.floatValue == 1e-5)
        #expect((fp32["relative"] as? NSNumber)?.floatValue == 1e-5)
        let full = try #require(fixture["fullAttention"] as? [String: Any])
        let input = try fixtureTensor(full, named: "input")
        let fixtureQ = try fixtureTensor(full, named: "qProjection")
        let fixtureQNormalized = try fixtureTensor(full, named: "qNormalized")
        let fixtureKNormalized = try fixtureTensor(full, named: "kNormalized")
        let fixtureV = try fixtureTensor(full, named: "valueProjection")
        let fixtureQRotated = try fixtureTensor(full, named: "qAfterPartialRoPE")
        let fixtureKRotated = try fixtureTensor(full, named: "kAfterPartialRoPE")
        let fixtureWeights = try fixtureTensor(full, named: "attentionWeights")
        let fixturePreGate = try fixtureTensor(full, named: "preOutputGate")
        let fixtureGate = try fixtureTensor(full, named: "outputGate")
        let fixtureGated = try fixtureTensor(full, named: "gatedAttention")
        let fixtureOutput = try fixtureTensor(full, named: "output")

        // Rebuild every pinned module parameter from the generator's named-parameter
        // order rather than deriving expected values from the production result.
        let qProjectionWeights = syntheticMatrix(rows: 64, columns: 32, phase: 1.25)
        let kProjectionWeights = syntheticMatrix(rows: 16, columns: 32, phase: 2.25)
        let vProjectionWeights = syntheticMatrix(rows: 16, columns: 32, phase: 3.25)
        let outputProjection = syntheticMatrix(rows: 32, columns: 32, phase: 4.25)
        let queryNormWeight = syntheticNormWeight(count: 16, phase: 5.25)
        let keyNormWeight = syntheticNormWeight(count: 16, phase: 6.25)
        let queryAndGate = matrixProject(input, weights: qProjectionWeights, rows: 64, columns: 32)
        let keyProjection = matrixProject(input, weights: kProjectionWeights, rows: 16, columns: 32)
        let valueProjection = matrixProject(input, weights: vProjectionWeights, rows: 16, columns: 32)
        let positions = Array(0..<5)
        let expected = independentAttention(
            queryAndGate: queryAndGate, keys: keyProjection, values: valueProjection,
            qWeights: queryNormWeight, kWeights: keyNormWeight, positions: positions,
            outputProjection: outputProjection)

        // The independent reconstruction must itself agree with the frozen oracle,
        // including its documented head-major tensors after canonical transposition.
        expectClose(expected.query, fixtureQ, tolerance: 1e-5)
        expectClose(expected.normalizedQuery,
                    headMajorToTokenMajor(fixtureQNormalized, tokenCount: 5, headCount: 2, dimension: 16),
                    tolerance: 1e-5)
        expectClose(expected.normalizedKey,
                    headMajorToTokenMajor(fixtureKNormalized, tokenCount: 5, headCount: 1, dimension: 16),
                    tolerance: 1e-5)
        expectClose(valueProjection, fixtureV, tolerance: 1e-5)
        expectClose(expected.rotatedQuery,
                    headMajorToTokenMajor(fixtureQRotated, tokenCount: 5, headCount: 2, dimension: 16),
                    tolerance: 1e-5)
        expectClose(expected.rotatedKey,
                    headMajorToTokenMajor(fixtureKRotated, tokenCount: 5, headCount: 1, dimension: 16),
                    tolerance: 1e-5)
        expectClose(expected.attentionWeights,
                    headMajorWeightsToTokenMajor(fixtureWeights, tokenCount: 5, headCount: 2),
                    tolerance: 1e-5)
        expectClose(expected.preOutputGate, fixturePreGate, tolerance: 1e-5)
        expectClose(expected.gate, fixtureGate, tolerance: 1e-5)
        expectClose(expected.gatedAttention, fixtureGated, tolerance: 1e-5)
        expectClose(expected.output, fixtureOutput, tolerance: 1e-5)

        let configuration = try fixtureConfiguration()
        let oneToken = try QwenFullAttention.evaluate(
            configuration: configuration,
            queryAndGateProjection: Array(queryAndGate.prefix(64)),
            keyProjection: Array(keyProjection.prefix(16)),
            valueProjection: Array(valueProjection.prefix(16)),
            queryNormWeight: queryNormWeight, keyNormWeight: keyNormWeight,
            positions: [0], outputProjection: outputProjection, outputDimension: 32)
        let oneExpected = referenceSlice(expected, tokenIndices: [0], totalKeyCount: 1)
        expectResultClose(oneToken, oneExpected)
        expectResultClose(oneToken, referenceSliceFromFixture(
            query: fixtureQ,
            normalizedQuery: headMajorToTokenMajor(fixtureQNormalized, tokenCount: 5, headCount: 2, dimension: 16),
            normalizedKey: headMajorToTokenMajor(fixtureKNormalized, tokenCount: 5, headCount: 1, dimension: 16),
            rotatedQuery: headMajorToTokenMajor(fixtureQRotated, tokenCount: 5, headCount: 2, dimension: 16),
            rotatedKey: headMajorToTokenMajor(fixtureKRotated, tokenCount: 5, headCount: 1, dimension: 16),
            weights: headMajorWeightsToTokenMajor(fixtureWeights, tokenCount: 5, headCount: 2),
            preGate: fixturePreGate, gate: fixtureGate, gated: fixtureGated, output: fixtureOutput,
            tokenIndices: [0], totalKeyCount: 1))

        let fullResult = try QwenFullAttention.evaluate(
            configuration: configuration, queryAndGateProjection: queryAndGate,
            keyProjection: keyProjection, valueProjection: valueProjection,
            queryNormWeight: queryNormWeight, keyNormWeight: keyNormWeight,
            positions: positions, outputProjection: outputProjection, outputDimension: 32)
        expectResultClose(fullResult, expected)
        expectResultClose(fullResult, referenceSliceFromFixture(
            query: fixtureQ, normalizedQuery: headMajorToTokenMajor(fixtureQNormalized, tokenCount: 5, headCount: 2, dimension: 16),
            normalizedKey: headMajorToTokenMajor(fixtureKNormalized, tokenCount: 5, headCount: 1, dimension: 16),
            rotatedQuery: headMajorToTokenMajor(fixtureQRotated, tokenCount: 5, headCount: 2, dimension: 16),
            rotatedKey: headMajorToTokenMajor(fixtureKRotated, tokenCount: 5, headCount: 1, dimension: 16),
            weights: headMajorWeightsToTokenMajor(fixtureWeights, tokenCount: 5, headCount: 2),
            preGate: fixturePreGate, gate: fixtureGate, gated: fixtureGated, output: fixtureOutput,
            tokenIndices: positions, totalKeyCount: 5))

        let splitPrefix = try QwenFullAttention.evaluate(
            configuration: configuration, queryAndGateProjection: Array(queryAndGate.prefix(3 * 64)),
            keyProjection: Array(keyProjection.prefix(3 * 16)),
            valueProjection: Array(valueProjection.prefix(3 * 16)),
            queryNormWeight: queryNormWeight, keyNormWeight: keyNormWeight,
            positions: Array(0..<3), outputProjection: outputProjection, outputDimension: 32)
        expectResultClose(splitPrefix, referenceSlice(expected, tokenIndices: [0, 1, 2], totalKeyCount: 3))

        let splitSuffix = try QwenFullAttention.evaluate(
            configuration: configuration, queryAndGateProjection: Array(queryAndGate.dropFirst(3 * 64)),
            keyProjection: Array(keyProjection.dropFirst(3 * 16)),
            valueProjection: Array(valueProjection.dropFirst(3 * 16)),
            queryNormWeight: queryNormWeight, keyNormWeight: keyNormWeight,
            positions: [3, 4], cachedKeys: Array(expected.rotatedKey.prefix(3 * 16)),
            cachedValues: Array(valueProjection.prefix(3 * 16)), outputProjection: outputProjection,
            outputDimension: 32)
        expectResultClose(splitSuffix, referenceSlice(expected, tokenIndices: [3, 4], totalKeyCount: 5))

        // Decode one token at a time using independently reconstructed cache rows.
        var cachedKeys: [Float] = []
        var cachedValues: [Float] = []
        for token in positions {
            let step = try QwenFullAttention.evaluate(
                configuration: configuration,
                queryAndGateProjection: Array(queryAndGate.dropFirst(token * 64).prefix(64)),
                keyProjection: Array(keyProjection.dropFirst(token * 16).prefix(16)),
                valueProjection: Array(valueProjection.dropFirst(token * 16).prefix(16)),
                queryNormWeight: queryNormWeight, keyNormWeight: keyNormWeight,
                positions: [token], cachedKeys: cachedKeys, cachedValues: cachedValues,
                outputProjection: outputProjection, outputDimension: 32)
            expectResultClose(step, referenceSlice(expected, tokenIndices: [token], totalKeyCount: token + 1))
            cachedKeys.append(contentsOf: expected.rotatedKey[(token * 16)..<(token * 16 + 16)])
            cachedValues.append(contentsOf: valueProjection[(token * 16)..<(token * 16 + 16)])
        }
    }

    @Test func officialPartialRoPELeavesDimensions64Through255Untouched() throws {
        let configuration = try QwenFullAttentionConfiguration.official()
        let queryWidth = configuration.queryHeadCount * configuration.headDimension
        let keyWidth = configuration.keyValueHeadCount * configuration.headDimension
        let query = sequence(count: queryWidth * 4, seed: 0.19)
        let key = sequence(count: keyWidth * 2, seed: -0.07)
        let value = sequence(count: keyWidth * 2, seed: 0.11)
        let weights = sequence(count: configuration.headDimension, seed: 0.02)
        let outputWeights = sequence(count: queryWidth, seed: 0.01)
        let result = try QwenFullAttention.evaluate(
            configuration: configuration, queryAndGateProjection: query, keyProjection: key,
            valueProjection: value, queryNormWeight: weights, keyNormWeight: weights,
            positions: [0, 1], outputProjection: outputWeights, outputDimension: 1)
        for head in 0..<configuration.queryHeadCount {
            let base = (configuration.queryHeadCount + head) * configuration.headDimension
            expectClose(
                Array(result.rotatedQuery[(base + 64)..<(base + 256)]),
                Array(result.normalizedQuery[(base + 64)..<(base + 256)]), tolerance: 1e-5)
        }
        #expect(maxDifference(
            Array(result.rotatedQuery[(configuration.queryHeadCount * configuration.headDimension)..<(configuration.queryHeadCount * configuration.headDimension + 64)]),
            Array(result.normalizedQuery[(configuration.queryHeadCount * configuration.headDimension)..<(configuration.queryHeadCount * configuration.headDimension + 64)])) > 1e-6)
    }

    @Test func rejectsOutOfOrderAndMalformedEvaluationInputs() throws {
        let fixtureConfiguration = try fixtureConfiguration()
        let query = syntheticQueryAndGate(tokens: 1)
        let projection = sequence(count: 16, seed: 0.1)
        let weights = sequence(count: 16, seed: 0.02)
        let output = sequence(count: 32, seed: 0.3)
        #expect(throws: QwenFullAttentionError.invalidPosition(expected: 0, actual: 1)) {
            try QwenFullAttention.evaluate(
                configuration: fixtureConfiguration, queryAndGateProjection: query,
                keyProjection: projection, valueProjection: projection, queryNormWeight: weights,
                keyNormWeight: weights, positions: [1], outputProjection: output, outputDimension: 1)
        }
        #expect(throws: QwenFullAttentionError.invalidCount(
            field: "queryNormWeight", expected: 16, actual: 15
        )) {
            try QwenFullAttention.evaluate(
                configuration: fixtureConfiguration, queryAndGateProjection: query,
                keyProjection: projection, valueProjection: projection,
                queryNormWeight: Array(weights.dropLast()), keyNormWeight: weights,
                positions: [0], outputProjection: output, outputDimension: 1)
        }
    }

    @Test func realGPUPreprocessingPipelinesHandleAwkwardCounts() throws {
        let context = try MetalContext()
        let attention = try QwenFullAttention(context: context, configuration: fixtureConfiguration())
        let queryValues = sequence(count: 5 * 2 * 16, seed: 0.2)
        let keyValues = sequence(count: 5 * 16, seed: -0.1)
        let weights = sequence(count: 16, seed: 0.03)
        let rawGate = sequence(count: 5 * 2 * 16, seed: -0.4)
        let gateInput = sequence(count: rawGate.count, seed: 0.6)
        let queue = try #require(context.device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let qInput = floatBuffer(queryValues, device: context.device)
        let kInput = floatBuffer(keyValues, device: context.device)
        let weight = floatBuffer(weights, device: context.device)
        let qOutput = try #require(context.device.makeBuffer(length: qInput.length, options: .storageModeShared))
        let kOutput = try #require(context.device.makeBuffer(length: kInput.length, options: .storageModeShared))
        let gateAttention = floatBuffer(gateInput, device: context.device)
        let gateRaw = floatBuffer(rawGate, device: context.device)
        let gateOutput = try #require(context.device.makeBuffer(length: gateAttention.length, options: .storageModeShared))
        try attention.encodeNormAndPartialRoPE(commandBuffer: commandBuffer, input: qInput, weight: weight, output: qOutput, tokenCount: 5, headCount: 2, startPosition: 1)
        try attention.encodeNormAndPartialRoPE(commandBuffer: commandBuffer, input: kInput, weight: weight, output: kOutput, tokenCount: 5, headCount: 1, startPosition: 1)
        try attention.encodeOutputGate(commandBuffer: commandBuffer, attention: gateAttention, rawGate: gateRaw, output: gateOutput, elementCount: rawGate.count)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        #expect(commandBuffer.error == nil)
        expectClose(readFloats(qOutput, count: queryValues.count), independentNormRoPE(queryValues, heads: 2, positions: [1, 2, 3, 4, 5], weights: weights), tolerance: 1e-5)
        expectClose(readFloats(kOutput, count: keyValues.count), independentNormRoPE(keyValues, heads: 1, positions: [1, 2, 3, 4, 5], weights: weights), tolerance: 1e-5)
        expectClose(readFloats(gateOutput, count: rawGate.count), zip(gateInput, rawGate).map { $0 * sigmoid($1) }, tolerance: 1e-5)
    }
}

private struct AttentionReference {
    let query, normalizedQuery, normalizedKey, rotatedQuery, rotatedKey: [Float]
    let attentionWeights, preOutputGate, gate, gatedAttention, output: [Float]
    let outputWithoutGate, outputWithGateAfterProjection: [Float]
}

private func fixtureTensor(_ full: [String: Any], named name: String) throws -> [Float] {
    let tensor = try #require(full[name] as? [String: Any])
    return try #require(tensor["values"] as? [NSNumber]).map(\.floatValue)
}

private func syntheticMatrix(rows: Int, columns: Int, phase: Float) -> [Float] {
    (0..<(rows * columns)).map { sin(Float($0) * 0.37 + phase) * 0.18 }
}

private func syntheticNormWeight(count: Int, phase: Float) -> [Float] {
    (0..<count).map { 1 + sin(Float($0) * 0.37 + phase) * 0.04 }
}

private func matrixProject(_ input: [Float], weights: [Float], rows: Int, columns: Int) -> [Float] {
    let inputWidth = columns
    let tokenCount = input.count / inputWidth
    var output = [Float](repeating: 0, count: tokenCount * rows)
    for token in 0..<tokenCount {
        for row in 0..<rows {
            var sum: Float = 0
            let weightBase = row * inputWidth
            let inputBase = token * inputWidth
            for column in 0..<inputWidth {
                sum += input[inputBase + column] * weights[weightBase + column]
            }
            output[token * rows + row] = sum
        }
    }
    return output
}

private func headMajorToTokenMajor(_ values: [Float], tokenCount: Int, headCount: Int, dimension: Int) -> [Float] {
    var result = [Float](repeating: 0, count: values.count)
    for head in 0..<headCount {
        for token in 0..<tokenCount {
            for dimensionIndex in 0..<dimension {
                let source = (head * tokenCount + token) * dimension + dimensionIndex
                let destination = (token * headCount + head) * dimension + dimensionIndex
                result[destination] = values[source]
            }
        }
    }
    return result
}

private func headMajorWeightsToTokenMajor(_ values: [Float], tokenCount: Int, headCount: Int) -> [Float] {
    var result = [Float](repeating: 0, count: values.count)
    for head in 0..<headCount {
        for queryToken in 0..<tokenCount {
            for keyToken in 0..<tokenCount {
                let source = (head * tokenCount + queryToken) * tokenCount + keyToken
                let destination = (queryToken * headCount + head) * tokenCount + keyToken
                result[destination] = values[source]
            }
        }
    }
    return result
}

private func referenceSlice(_ reference: AttentionReference, tokenIndices: [Int], totalKeyCount: Int) -> AttentionReference {
    let query = tokenSlice(reference.query, width: 32, tokenIndices: tokenIndices)
    let normalizedQuery = tokenSlice(reference.normalizedQuery, width: 32, tokenIndices: tokenIndices)
    let normalizedKey = tokenSlice(reference.normalizedKey, width: 16, tokenIndices: tokenIndices)
    let rotatedQuery = tokenSlice(reference.rotatedQuery, width: 32, tokenIndices: tokenIndices)
    let rotatedKey = tokenSlice(reference.rotatedKey, width: 16, tokenIndices: tokenIndices)
    let preGate = tokenSlice(reference.preOutputGate, width: 32, tokenIndices: tokenIndices)
    let gate = tokenSlice(reference.gate, width: 32, tokenIndices: tokenIndices)
    let gated = tokenSlice(reference.gatedAttention, width: 32, tokenIndices: tokenIndices)
    let output = tokenSlice(reference.output, width: 32, tokenIndices: tokenIndices)
    var weights: [Float] = []
    for token in tokenIndices {
        for head in 0..<2 {
            let base = (token * 2 + head) * 5
            weights.append(contentsOf: reference.attentionWeights[base..<(base + totalKeyCount)])
        }
    }
    return AttentionReference(
        query: query, normalizedQuery: normalizedQuery, normalizedKey: normalizedKey,
        rotatedQuery: rotatedQuery, rotatedKey: rotatedKey, attentionWeights: weights,
        preOutputGate: preGate, gate: gate, gatedAttention: gated, output: output,
        outputWithoutGate: [], outputWithGateAfterProjection: [])
}

private func referenceSliceFromFixture(
    query: [Float], normalizedQuery: [Float], normalizedKey: [Float], rotatedQuery: [Float],
    rotatedKey: [Float], weights: [Float], preGate: [Float], gate: [Float], gated: [Float],
    output: [Float], tokenIndices: [Int], totalKeyCount: Int
) -> AttentionReference {
    var reference = AttentionReference(
        query: query, normalizedQuery: normalizedQuery, normalizedKey: normalizedKey,
        rotatedQuery: rotatedQuery, rotatedKey: rotatedKey, attentionWeights: weights,
        preOutputGate: preGate, gate: gate, gatedAttention: gated, output: output,
        outputWithoutGate: [], outputWithGateAfterProjection: [])
    reference = AttentionReference(
        query: tokenSlice(reference.query, width: 32, tokenIndices: tokenIndices),
        normalizedQuery: tokenSlice(reference.normalizedQuery, width: 32, tokenIndices: tokenIndices),
        normalizedKey: tokenSlice(reference.normalizedKey, width: 16, tokenIndices: tokenIndices),
        rotatedQuery: tokenSlice(reference.rotatedQuery, width: 32, tokenIndices: tokenIndices),
        rotatedKey: tokenSlice(reference.rotatedKey, width: 16, tokenIndices: tokenIndices),
        attentionWeights: attentionSlice(weights, tokenIndices: tokenIndices, totalKeyCount: totalKeyCount),
        preOutputGate: tokenSlice(reference.preOutputGate, width: 32, tokenIndices: tokenIndices),
        gate: tokenSlice(reference.gate, width: 32, tokenIndices: tokenIndices),
        gatedAttention: tokenSlice(reference.gatedAttention, width: 32, tokenIndices: tokenIndices),
        output: tokenSlice(reference.output, width: 32, tokenIndices: tokenIndices),
        outputWithoutGate: [], outputWithGateAfterProjection: [])
    return reference
}

private func tokenSlice(_ values: [Float], width: Int, tokenIndices: [Int]) -> [Float] {
    tokenIndices.flatMap { token in values[(token * width)..<((token + 1) * width)] }
}

private func attentionSlice(_ values: [Float], tokenIndices: [Int], totalKeyCount: Int) -> [Float] {
    var result: [Float] = []
    for token in tokenIndices {
        for head in 0..<2 {
            let base = (token * 2 + head) * 5
            result.append(contentsOf: values[base..<(base + totalKeyCount)])
        }
    }
    return result
}

private func expectResultClose(_ actual: QwenFullAttentionResult, _ expected: AttentionReference, sourceLocation: SourceLocation = #_sourceLocation) {
    expectClose(actual.queryProjection, expected.query, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.normalizedQuery, expected.normalizedQuery, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.normalizedKey, expected.normalizedKey, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.rotatedQuery, expected.rotatedQuery, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.rotatedKey, expected.rotatedKey, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.attentionWeights, expected.attentionWeights, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.preOutputGate, expected.preOutputGate, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.outputGate, expected.gate, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.gatedAttention, expected.gatedAttention, tolerance: 1e-5, sourceLocation: sourceLocation)
    expectClose(actual.output, expected.output, tolerance: 1e-5, sourceLocation: sourceLocation)
}

private func independentAttention(queryAndGate: [Float], keys: [Float], values: [Float], qWeights: [Float], kWeights: [Float], positions: [Int], outputProjection: [Float]) -> AttentionReference {
    var query: [Float] = []
    var gateRaw: [Float] = []
    for token in positions.indices {
        for head in 0..<2 {
            let base = token * 64 + head * 32
            query += queryAndGate[base..<(base + 16)]
            gateRaw += queryAndGate[(base + 16)..<(base + 32)]
        }
    }
    let normalizedQuery = independentNorm(query, heads: 2, weights: qWeights)
    let normalizedKey = independentNorm(keys, heads: 1, weights: kWeights)
    let rotatedQuery = independentRoPE(normalizedQuery, heads: 2, positions: positions)
    let rotatedKey = independentRoPE(normalizedKey, heads: 1, positions: positions)
    var context = [Float](repeating: 0, count: query.count)
    var attentionWeights = [Float](repeating: 0, count: positions.count * 2 * positions.count)
    for token in positions.indices {
        for head in 0..<2 {
            let queryBase = (token * 2 + head) * 16
            var scores: [Float] = []
            for keyToken in 0...token { scores.append(dot(rotatedQuery, queryBase, rotatedKey, keyToken * 16) / 4) }
            for (keyToken, probability) in softmax(scores).enumerated() {
                attentionWeights[(token * 2 + head) * positions.count + keyToken] = probability
                let valueBase = keyToken * 16
                for dimension in 0..<16 { context[queryBase + dimension] += probability * values[valueBase + dimension] }
            }
        }
    }
    let gate = gateRaw.map(sigmoid)
    let gated = zip(context, gate).map(*)
    let output = project(gated, outputProjection)
    let ungated = project(context, outputProjection)
    let lateGate = zip(ungated, gate.prefix(ungated.count)).map(*)
    return AttentionReference(query: query, normalizedQuery: normalizedQuery, normalizedKey: normalizedKey, rotatedQuery: rotatedQuery, rotatedKey: rotatedKey, attentionWeights: attentionWeights, preOutputGate: context, gate: gate, gatedAttention: gated, output: output, outputWithoutGate: ungated, outputWithGateAfterProjection: lateGate)
}

private func independentNormRoPE(_ values: [Float], heads: Int, positions: [Int], weights: [Float]) -> [Float] { independentRoPE(independentNorm(values, heads: heads, weights: weights), heads: heads, positions: positions) }
private func independentNorm(_ values: [Float], heads: Int, weights: [Float]) -> [Float] { var result = values; for item in 0..<(values.count / 16) { let base = item * 16; let rms = (values[base..<(base + 16)].reduce(Float.zero) { $0 + $1 * $1 } / 16 + 1e-6).squareRoot(); for d in 0..<16 { result[base + d] = values[base + d] / rms * (1 + weights[d]) } }; return result }
private func independentRoPE(_ values: [Float], heads: Int, positions: [Int]) -> [Float] { var result = values; for token in positions.indices { for head in 0..<heads { let base = (token * heads + head) * 16; for d in 0..<6 { let angle = Float(positions[token]) / Float(pow(10_000_000.0, Double(2 * d) / 12)); let a = values[base + d], b = values[base + d + 6]; result[base + d] = a * cos(angle) - b * sin(angle); result[base + d + 6] = b * cos(angle) + a * sin(angle) } } }; return result }
private func adjacentPairRotation(_ values: [Float], positions: [Int]) -> [Float] { var result = values; for token in positions.indices { for h in 0..<2 { let base = (token * 2 + h) * 16; for d in stride(from: 0, to: 12, by: 2) { let angle = Float(positions[token]) / Float(pow(10_000_000.0, Double(d) / 12)); let a = values[base + d], b = values[base + d + 1]; result[base + d] = a * cos(angle) - b * sin(angle); result[base + d + 1] = b * cos(angle) + a * sin(angle) } } }; return result }
private func wrongSpanRotation(_ values: [Float], positions: [Int]) -> [Float] { var result = values; for token in positions.indices { for h in 0..<2 { let base = (token * 2 + h) * 16; for d in 0..<8 { let angle = Float(positions[token]) / Float(pow(10_000_000.0, Double(2 * d) / 16)); let peer = (d + 8) % 16; let a = values[base + d], b = values[base + peer]; result[base + d] = a * cos(angle) - b * sin(angle); result[base + peer] = b * cos(angle) + a * sin(angle) } } }; return result }
private func project(_ input: [Float], _ weights: [Float]) -> [Float] {
    let rows = weights.count / 32
    return stride(from: 0, to: input.count, by: 32).flatMap { base in
        (0..<rows).map { row in
            zip(input[base..<(base + 32)], weights[(row * 32)..<(row * 32 + 32)])
                .reduce(Float.zero) { $0 + $1.0 * $1.1 }
        }
    }
}
private func dot(_ a: [Float], _ aBase: Int, _ b: [Float], _ bBase: Int) -> Float { (0..<16).reduce(0) { $0 + a[aBase + $1] * b[bBase + $1] } }
private func softmax(_ scores: [Float]) -> [Float] { let maximum = scores.max()!; let values = scores.map { exp($0 - maximum) }; let total = values.reduce(0, +); return values.map { $0 / total } }
private func sigmoid(_ value: Float) -> Float { 1 / (1 + exp(-value)) }
private func sequence(count: Int, seed: Float) -> [Float] { (0..<count).map { seed + Float(($0 * 17) % 29 - 14) * 0.03125 } }
private func syntheticQueryAndGate(tokens: Int) -> [Float] { sequence(count: tokens * 64, seed: 0.15) }
private func floatBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer { device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Float>.stride, options: .storageModeShared)! }
private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] { Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count)) }
private func expectClose(_ actual: [Float], _ expected: [Float], tolerance: Float, sourceLocation: SourceLocation = #_sourceLocation) { #expect(actual.count == expected.count, sourceLocation: sourceLocation); for (index, pair) in zip(actual, expected).enumerated() { #expect(abs(pair.0 - pair.1) <= tolerance + tolerance * abs(pair.1), "index \(index): \(pair.0) != \(pair.1)", sourceLocation: sourceLocation) } }
private func maxDifference(_ lhs: [Float], _ rhs: [Float]) -> Float { zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0 }
