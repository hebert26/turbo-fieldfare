import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenBF16FullAttentionSourceMathTests {
    @Test func officialSourceMathMatchesTorchForInitialAndCachedDistinctAxisAttention() async throws {
        let fixture = try loadSourceAttentionFixture()
        let geometry = fixture.geometry
        #expect(fixture.kind == "qwen-source-full-attention-synthetic-torch-v1")
        #expect(fixture.provenance.transformersCommit == "bd15bc95a89e728bbc1224084eb3b5829428c353")
        #expect(fixture.provenance.transformersTree == "80eb369e589827bc7ed45b3a1f0ead5457097535")
        #expect(fixture.provenance.moduleSHA256 == "971b08ed3eb7452f3f5f1b0f8ab8e602fc4c4e2cc10d0a4e99ec1e1830776074")
        #expect(fixture.provenance.torchVersion == "2.10.0")
        #expect(fixture.provenance.numpyVersion == "2.4.3")
        #expect(fixture.provenance.device == "cpu")
        #expect(fixture.provenance.optionalKernels.isEmpty)
        let configuration = try QwenFullAttentionConfiguration(
            queryHeadCount: geometry.queryHeadCount,
            keyValueHeadCount: geometry.keyValueHeadCount,
            headDimension: geometry.headDimension,
            rotaryDimension: geometry.rotaryDimension,
            theta: geometry.theta,
            epsilon: geometry.epsilon)
        let context = try MetalContext()
        let source = try QwenFullAttention(
            context: context, configuration: configuration, useOfficialSourceMath: true)
        let packedDefault = try QwenFullAttention(
            context: context, configuration: configuration)
        let packedExplicit = try QwenFullAttention(
            context: context, configuration: configuration, useOfficialSourceMath: false)

        let first = try await runStep(
            fixture.initialOneKey, cacheKey: nil, cacheValue: nil,
            attention: source, context: context, geometry: geometry)
        let firstPackedDefault = try await runNormRoPE(
            fixture.initialOneKey, attention: packedDefault,
            context: context, geometry: geometry)
        let firstPackedExplicit = try await runNormRoPE(
            fixture.initialOneKey, attention: packedExplicit,
            context: context, geometry: geometry)
        #expect(firstPackedDefault.query == firstPackedExplicit.query)
        #expect(firstPackedDefault.key == firstPackedExplicit.key)

        let second = try await runStep(
            fixture.cachedTwoKeys,
            cacheKey: first.rotatedKey, cacheValue: fixture.initialOneKey.inputs.valueProjection,
            attention: source, context: context, geometry: geometry)

        #expect(fixture.geometry.mropeSections == [2, 1, 1])
        #expect(fixture.initialOneKey.mropePosition == [0, 2, 5])
        #expect(fixture.cachedTwoKeys.mropePosition == [1, 4, 7])
        #expect(fixture.initialOneKey.reference.rotatedQuery.count
            == geometry.queryHeadCount * geometry.headDimension)
        #expect(fixture.cachedTwoKeys.reference.attentionWeights.count == 2 * 2)
        let headDimension = geometry.headDimension
        expectSourceClose(
            Array(second.preOutputGate.prefix(2)),
            Array(fixture.cachedTwoKeys.reference.attentionWeights.prefix(2)),
            label: "cached head 0 two-key softmax weights encoded by value basis")
        expectSourceClose(
            Array(second.preOutputGate[headDimension..<(headDimension + 2)]),
            Array(fixture.cachedTwoKeys.reference.attentionWeights.suffix(2)),
            label: "cached head 1 two-key softmax weights encoded by value basis")
        #expect(firstPackedDefault.query != first.rotatedQuery,
                "distinct-axis fixture must expose the old contiguous M-RoPE layout")
    }

    @Test func officialSourceHead256NormMatchesPinnedTorchAtZeroRoPEPosition() async throws {
        let fixture = try loadHead256NormSubset()
        #expect(fixture.kind == "qwen36-head256-source-norm-subset-v1")
        #expect(fixture.provenance.transformersCommit == "bd15bc95a89e728bbc1224084eb3b5829428c353")
        #expect(fixture.provenance.moduleSHA256 == "971b08ed3eb7452f3f5f1b0f8ab8e602fc4c4e2cc10d0a4e99ec1e1830776074")
        #expect(fixture.provenance.scriptSHA256 == "b992e0f815333a991352ce4d9c1f1f4b435522be02b6be2c7d1ce7e8a7922b20")
        #expect(fixture.provenance.torch == "2.10.0")
        #expect(fixture.provenance.numpy == "2.4.3")
        #expect(fixture.provenance.device == "cpu")
        #expect(fixture.provenance.threads == 1)
        #expect(fixture.originalTensorReads == 0)
        #expect(!fixture.qualification)
        #expect(fixture.geometry.headDimension == 256)
        #expect(fixture.geometry.queryHeads == 16)
        #expect(fixture.geometry.keyHeads == 2)
        #expect(fixture.geometry.rotaryDimension == 64)
        #expect(fixture.geometry.position == 0)
        #expect(fixture.absoluteTolerance == 1e-7)
        #expect(fixture.relativeTolerance == 1e-6)
        #expect(fixture.query.meanBits == 0x43272fc0)
        #expect(fixture.query.inverseRMSBits == 0x3d9e63fc)
        #expect(fixture.key.meanBits == 0x431a27dc)
        #expect(fixture.key.inverseRMSBits == 0x3da4f2fe)

        let configuration = try QwenFullAttentionConfiguration(
            queryHeadCount: fixture.geometry.queryHeads,
            keyValueHeadCount: fixture.geometry.keyHeads,
            headDimension: fixture.geometry.headDimension,
            rotaryDimension: fixture.geometry.rotaryDimension,
            theta: Float(fixture.geometry.theta), epsilon: fixture.geometry.epsilon)
        let context = try MetalContext()
        let attention = try QwenFullAttention(
            context: context, configuration: configuration, useOfficialSourceMath: true)

        let queryRow = fixture.query.inputBits.map { Float(bitPattern: $0) }
        let keyRow = fixture.key.inputBits.map { Float(bitPattern: $0) }
        let queryInput = makeFloatBuffer(
            Array(repeating: queryRow, count: fixture.geometry.queryHeads).flatMap { $0 },
            device: context.device)
        let keyInput = makeFloatBuffer(
            Array(repeating: keyRow, count: fixture.geometry.keyHeads).flatMap { $0 },
            device: context.device)
        let queryWeight = makeFloatBuffer(
            fixture.query.weightBits.map { Float(bitPattern: $0) }, device: context.device)
        let keyWeight = makeFloatBuffer(
            fixture.key.weightBits.map { Float(bitPattern: $0) }, device: context.device)
        let queryOutput = try #require(context.device.makeBuffer(
            length: queryInput.length, options: .storageModeShared))
        let keyOutput = try #require(context.device.makeBuffer(
            length: keyInput.length, options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        try attention.encodeNormAndPartialRoPE(
            commandBuffer: command, input: queryInput, weight: queryWeight,
            output: queryOutput, tokenCount: 1, headCount: fixture.geometry.queryHeads,
            startPosition: fixture.geometry.position)
        try attention.encodeNormAndPartialRoPE(
            commandBuffer: command, input: keyInput, weight: keyWeight,
            output: keyOutput, tokenCount: 1, headCount: fixture.geometry.keyHeads,
            startPosition: fixture.geometry.position)
        command.commit()
        await command.completed()
        #expect(command.status == .completed)
        #expect(command.error == nil)

        let expectedQueryBits = Array(
            repeating: fixture.query.outputBits, count: fixture.geometry.queryHeads).flatMap { $0 }
        let expectedKeyBits = Array(
            repeating: fixture.key.outputBits, count: fixture.geometry.keyHeads).flatMap { $0 }
        let queryActualBits = readFloatBuffer(
            queryOutput, count: fixture.geometry.queryHeads * fixture.geometry.headDimension)
            .map { $0.bitPattern }
        let keyActualBits = readFloatBuffer(
            keyOutput, count: fixture.geometry.keyHeads * fixture.geometry.headDimension)
            .map { $0.bitPattern }
        #expect(queryActualBits == expectedQueryBits)
        #expect(keyActualBits == expectedKeyBits)
    }

    @Test func officialSourceHead256RotaryMatchesPinnedTorchAtCachedPositions() async throws {
        let norm = try loadHead256NormSubset()
        let fixture = try loadHead256RotaryRegression()
        #expect(fixture.kind == "qwen36-head256-source-rotary-regression-v1")
        #expect(fixture.originalTensorReads == 0)
        #expect(!fixture.qualification)
        #expect(fixture.provenance.transformersCommit == "bd15bc95a89e728bbc1224084eb3b5829428c353")
        #expect(fixture.provenance.moduleSHA256 == "971b08ed3eb7452f3f5f1b0f8ab8e602fc4c4e2cc10d0a4e99ec1e1830776074")
        #expect(fixture.provenance.torch == "2.10.0")
        #expect(fixture.provenance.numpy == "2.4.3")
        #expect(fixture.provenance.device == "cpu")
        #expect(fixture.provenance.threads == 1)
        #expect(fixture.geometry.headDimension == 256)
        #expect(fixture.geometry.rotaryDimension == 64)
        #expect(fixture.geometry.mropeSections == [11, 11, 10])
        #expect(fixture.absoluteTolerance == 1e-7)
        #expect(fixture.relativeTolerance == 1e-6)
        #expect(fixture.cases.map { $0.mropePosition } == [[1, 4, 7], [127, 129, 126]])

        let configuration = try QwenFullAttentionConfiguration(
            queryHeadCount: 1, keyValueHeadCount: 1, headDimension: 256,
            rotaryDimension: 64, theta: 10_000_000, epsilon: norm.geometry.epsilon)
        let context = try MetalContext()
        let attention = try QwenFullAttention(
            context: context, configuration: configuration, useOfficialSourceMath: true)

        for item in fixture.cases {
            let query = try await runSourceRoPE(
                inputBits: norm.query.inputBits, weightBits: norm.query.weightBits,
                position: item.mropePosition, attention: attention, context: context,
                sections: fixture.geometry.mropeSections)
            let key = try await runSourceRoPE(
                inputBits: norm.key.inputBits, weightBits: norm.key.weightBits,
                position: item.mropePosition, attention: attention, context: context,
                sections: fixture.geometry.mropeSections)
            #expect(query.map { $0.bitPattern } == item.rotatedQueryBits,
                    "\(item.id) query must match pinned official FP32 output")
            #expect(key.map { $0.bitPattern } == item.rotatedKeyBits,
                    "\(item.id) key must match pinned official FP32 output")
            #expect(item.rotatedQueryBits != norm.query.outputBits,
                    "\(item.id) query must exercise rotation")
            #expect(item.rotatedKeyBits != norm.key.outputBits,
                    "\(item.id) key must exercise rotation")
        }
    }

    @Test func officialSourceAttentionStepMatchesPinnedLongContextTorchHeads() throws {
        let fixture = try loadLongContextAttentionSubset()
        #expect(fixture.kind == "qwen-source-attention-long-context-torch-subset-v1")
        #expect(fixture.provenance.transformersCommit == "bd15bc95a89e728bbc1224084eb3b5829428c353")
        #expect(fixture.provenance.officialModuleSHA256 == "971b08ed3eb7452f3f5f1b0f8ab8e602fc4c4e2cc10d0a4e99ec1e1830776074")
        #expect(fixture.provenance.torch == "2.10.0")
        #expect(fixture.provenance.threads == 1)
        #expect(fixture.provenance.device == "cpu")
        #expect(fixture.provenance.goldSourceSHA256 == "c45fe818ec0c883b9a5f9c7c45f2d3b7d2c25a251160177f52aa957259f3d492")
        #expect(fixture.provenance.preFixComparisonSHA256 == "5ee64b86def787034ea8b6b2d12555873b4e6db42e839154151ffb48460848fa")
        #expect(fixture.absoluteTolerance == 1e-7)
        #expect(fixture.relativeTolerance == 1e-6)
        #expect(fixture.cases.map { $0.keyCount } == [77, 110, 256])
        #expect(fixture.cases.allSatisfy { $0.preFix.frozenCoreFailures > 0 })

        let dimension = fixture.geometry.headDimension
        let configuration = try QwenFullAttentionConfiguration(
            queryHeadCount: 1, keyValueHeadCount: 1,
            headDimension: dimension, rotaryDimension: fixture.geometry.rotaryDimension,
            theta: Float(fixture.geometry.theta))
        let context = try MetalContext()
        let attention = try QwenFullAttention(
            context: context, configuration: configuration, useOfficialSourceMath: true)

        for item in fixture.cases {
            #expect(item.keyCount > 1)
            #expect(item.queryHead >= 0 && item.queryHead < fixture.geometry.queryHeads)
            #expect(item.keyValueHead == item.queryHead / 8)
            #expect(item.preFix.comparison == "full-attention-long-context-comparison-001.json")
            #expect(item.preFix.frozenCoreFailures > 0)
            #expect(item.preFix.firstFailingOutputIndex >= 0
                && item.preFix.firstFailingOutputIndex < dimension)
            let query = item.queryBits.map { Float(bitPattern: $0) }
            let cachedKeys = makeFloatBuffer(
                item.cachedKeyBits.map { Float(bitPattern: $0) }, device: context.device)
            let cachedValues = makeFloatBuffer(
                item.cachedValueBits.map { Float(bitPattern: $0) }, device: context.device)
            let cache = QwenFullAttentionKVView(
                key: cachedKeys, value: cachedValues, keyOffset: 0, valueOffset: 0,
                strideBytes: dimension * MemoryLayout<Float>.stride,
                validTokenCount: item.keyCount - 1)
            let output = try attention.attentionStep(
                rotatedQuery: query,
                rotatedKey: item.currentKeyBits.map { Float(bitPattern: $0) },
                value: item.currentValueBits.map { Float(bitPattern: $0) },
                cache: cache)
            #expect(output.map { $0.bitPattern } == item.expectedCoreBits,
                    "\(item.id) head \(item.queryHead) must match pinned Torch core output")
        }
    }
}

private struct SourceAttentionFixture: Decodable {
    let kind: String
    let provenance: Provenance
    let geometry: Geometry
    let initialOneKey: Step
    let cachedTwoKeys: Step

    struct Provenance: Decodable {
        let transformersCommit: String
        let transformersTree: String
        let moduleSHA256: String
        let torchVersion: String
        let numpyVersion: String
        let device: String
        let optionalKernels: [String]
    }

    struct Geometry: Decodable {
        let queryHeadCount: Int
        let keyValueHeadCount: Int
        let headDimension: Int
        let rotaryDimension: Int
        let theta: Float
        let epsilon: Float
        let outputDimension: Int
        let mropeSections: [Int]
    }

    struct Step: Decodable {
        let mropePosition: [Int]
        let inputs: Inputs
        let reference: Reference
    }

    struct Inputs: Decodable {
        let queryAndGateProjection: [UInt32]
        let keyProjection: [UInt32]
        let valueProjection: [UInt32]
        let queryNormWeight: [UInt32]
        let keyNormWeight: [UInt32]
    }

    struct Reference: Decodable {
        let rotatedQuery: [UInt32]
        let rotatedKey: [UInt32]
        let attentionWeights: [UInt32]
        let preOutputGate: [UInt32]
        let outputGate: [UInt32]
        let gatedAttention: [UInt32]
    }
}

private struct AttentionStepOutput {
    let rotatedQuery: [Float]
    let rotatedKey: [Float]
    let preOutputGate: [Float]
}

private struct NormRoPEOutput {
    let query: [Float]
    let key: [Float]
}

private struct Head256NormSubset: Decodable {
    let kind: String
    let originalTensorReads: Int
    let qualification: Bool
    let provenance: Provenance
    let geometry: Geometry
    let query: Row
    let key: Row
    let absoluteTolerance: Double
    let relativeTolerance: Double

    struct Provenance: Decodable {
        let transformersCommit: String
        let moduleSHA256: String
        let scriptSHA256: String
        let torch: String
        let numpy: String
        let device: String
        let threads: Int
    }

    struct Geometry: Decodable {
        let queryHeads: Int
        let keyHeads: Int
        let headDimension: Int
        let rotaryDimension: Int
        let theta: Double
        let epsilon: Float
        let position: Int
    }

    struct Row: Decodable {
        let inputBits: [UInt32]
        let weightBits: [UInt32]
        let meanBits: UInt32
        let inverseRMSBits: UInt32
        let outputBits: [UInt32]
    }
}

private struct Head256RotaryRegression: Decodable {
    let kind: String
    let originalTensorReads: Int
    let qualification: Bool
    let provenance: Provenance
    let geometry: Geometry
    let absoluteTolerance: Double
    let relativeTolerance: Double
    let cases: [Case]

    struct Provenance: Decodable {
        let transformersCommit: String
        let moduleSHA256: String
        let torch: String
        let numpy: String
        let device: String
        let threads: Int
    }

    struct Geometry: Decodable {
        let queryHeadCount: Int
        let keyValueHeadCount: Int
        let headDimension: Int
        let rotaryDimension: Int
        let theta: Float
        let mropeSections: [Int]
    }

    struct Case: Decodable {
        let id: String
        let mropePosition: [Int]
        let rotatedQueryBits: [UInt32]
        let rotatedKeyBits: [UInt32]
    }
}

private struct LongContextAttentionSubset: Decodable {
    let kind: String
    let absoluteTolerance: Double
    let relativeTolerance: Double
    let provenance: Provenance
    let geometry: Geometry
    let cases: [Case]

    struct Provenance: Decodable {
        let transformersCommit: String
        let officialModuleSHA256: String
        let torch: String
        let threads: Int
        let device: String
        let goldSourceSHA256: String
        let preFixComparisonSHA256: String
    }

    struct Geometry: Decodable {
        let headDimension: Int
        let rotaryDimension: Int
        let theta: Double
        let queryHeads: Int
        let keyValueHeads: Int
    }

    struct Case: Decodable {
        let id: String
        let keyCount: Int
        let queryHead: Int
        let keyValueHead: Int
        let queryBits: [UInt32]
        let cachedKeyBits: [UInt32]
        let cachedValueBits: [UInt32]
        let currentKeyBits: [UInt32]
        let currentValueBits: [UInt32]
        let expectedCoreBits: [UInt32]
        let preFix: PreFix

        struct PreFix: Decodable {
            let comparison: String
            let frozenCoreFailures: Int
            let firstFailingOutputIndex: Int
        }
    }
}

private func loadSourceAttentionFixture() throws -> SourceAttentionFixture {
    let url = try #require(Bundle.module.url(
        forResource: "source-math-fixtures", withExtension: "json",
        subdirectory: "full-attention-source"))
    return try JSONDecoder().decode(SourceAttentionFixture.self, from: Data(contentsOf: url))
}

private func loadHead256NormSubset() throws -> Head256NormSubset {
    let url = try #require(Bundle.module.url(
        forResource: "head256-norm-subset", withExtension: "json",
        subdirectory: "full-attention-source"))
    return try JSONDecoder().decode(Head256NormSubset.self, from: Data(contentsOf: url))
}

private func loadHead256RotaryRegression() throws -> Head256RotaryRegression {
    let url = try #require(Bundle.module.url(
        forResource: "head256-rotary-regression", withExtension: "json",
        subdirectory: "full-attention-source"))
    return try JSONDecoder().decode(Head256RotaryRegression.self, from: Data(contentsOf: url))
}

private func loadLongContextAttentionSubset() throws -> LongContextAttentionSubset {
    let url = try #require(Bundle.module.url(
        forResource: "long-context-attention-subset", withExtension: "json",
        subdirectory: "full-attention-source"))
    return try JSONDecoder().decode(LongContextAttentionSubset.self, from: Data(contentsOf: url))
}

private func runSourceRoPE(
    inputBits: [UInt32], weightBits: [UInt32], position: [Int],
    attention: QwenFullAttention, context: MetalContext, sections: [Int]
) async throws -> [Float] {
    let input = makeFloatBuffer(inputBits.map { Float(bitPattern: $0) }, device: context.device)
    let weight = makeFloatBuffer(weightBits.map { Float(bitPattern: $0) }, device: context.device)
    let positionBuffer = makeInt32Buffer(position.map(Int32.init), device: context.device)
    let output = try #require(context.device.makeBuffer(
        length: input.length, options: .storageModeShared))
    let command = try #require(context.queue.makeCommandBuffer())
    try attention.encodeNormAndPartialMRoPE(
        commandBuffer: command, input: input, weight: weight, positions: positionBuffer,
        output: output, tokenCount: 1, headCount: 1, sections: sections)
    command.commit()
    await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    return readFloatBuffer(output, count: 256)
}

private func runStep(
    _ step: SourceAttentionFixture.Step,
    cacheKey: [Float]?,
    cacheValue: [UInt32]?,
    attention: QwenFullAttention,
    context: MetalContext,
    geometry: SourceAttentionFixture.Geometry
) async throws -> AttentionStepOutput {
    let input = step.inputs
    let qAndGate = input.queryAndGateProjection.map { Float(bitPattern: $0) }
    let rawGate = Array((0..<geometry.queryHeadCount).flatMap { head in
        let start = head * geometry.headDimension * 2 + geometry.headDimension
        return qAndGate[start..<(start + geometry.headDimension)]
    })
    let value = input.valueProjection.map { Float(bitPattern: $0) }
    let encoded = try await runNormRoPE(
        step, attention: attention, context: context, geometry: geometry)

    let cacheKeyFloats = cacheKey ?? [Float](repeating: 0, count: geometry.keyValueHeadCount * geometry.headDimension)
    let cacheValueFloats = cacheValue?.map { Float(bitPattern: $0) }
        ?? [Float](repeating: 0, count: geometry.keyValueHeadCount * geometry.headDimension)
    let keyBuffer = makeFloatBuffer(cacheKeyFloats, device: context.device)
    let valueBuffer = makeFloatBuffer(cacheValueFloats, device: context.device)
    let cache = QwenFullAttentionKVView(
        key: keyBuffer, value: valueBuffer, keyOffset: 0, valueOffset: 0,
        strideBytes: geometry.keyValueHeadCount * geometry.headDimension * MemoryLayout<Float>.stride,
        validTokenCount: cacheKey == nil ? 0 : 1)
    let preOutputGate = try attention.attentionStep(
        rotatedQuery: encoded.query, rotatedKey: encoded.key, value: value, cache: cache)
    let actualOutputGate = try await runOutputGate(
        [Float](repeating: 1, count: preOutputGate.count), rawGate: rawGate,
        attention: attention, context: context)
    let actualGated = try await runOutputGate(
        preOutputGate, rawGate: rawGate, attention: attention, context: context)

    expectSourceClose(encoded.query, step.reference.rotatedQuery, label: "rotated query")
    expectSourceClose(encoded.key, step.reference.rotatedKey, label: "rotated key")
    expectSourceClose(preOutputGate, step.reference.preOutputGate, label: "attention output")
    expectSourceClose(actualOutputGate, step.reference.outputGate, label: "sigmoid gate")
    expectSourceClose(actualGated, step.reference.gatedAttention, label: "gated attention")
    return AttentionStepOutput(
        rotatedQuery: encoded.query, rotatedKey: encoded.key, preOutputGate: preOutputGate)
}

private func runNormRoPE(
    _ step: SourceAttentionFixture.Step,
    attention: QwenFullAttention,
    context: MetalContext,
    geometry: SourceAttentionFixture.Geometry
) async throws -> NormRoPEOutput {
    let qAndGate = step.inputs.queryAndGateProjection.map { Float(bitPattern: $0) }
    let query = (0..<geometry.queryHeadCount).flatMap { head in
        let start = head * geometry.headDimension * 2
        return qAndGate[start..<(start + geometry.headDimension)]
    }
    let key = step.inputs.keyProjection.map { Float(bitPattern: $0) }
    let queryWeights = step.inputs.queryNormWeight.map { Float(bitPattern: $0) }
    let keyWeights = step.inputs.keyNormWeight.map { Float(bitPattern: $0) }
    let positions = step.mropePosition.map { Int32($0) }
    let queryInput = makeFloatBuffer(query, device: context.device)
    let keyInput = makeFloatBuffer(key, device: context.device)
    let queryWeight = makeFloatBuffer(queryWeights, device: context.device)
    let keyWeight = makeFloatBuffer(keyWeights, device: context.device)
    let positionBuffer = makeInt32Buffer(positions, device: context.device)
    let queryOutput = try #require(context.device.makeBuffer(
        length: query.count * MemoryLayout<Float>.stride, options: .storageModeShared))
    let keyOutput = try #require(context.device.makeBuffer(
        length: key.count * MemoryLayout<Float>.stride, options: .storageModeShared))
    let command = try #require(context.queue.makeCommandBuffer())
    try attention.encodeNormAndPartialMRoPE(
        commandBuffer: command, input: queryInput, weight: queryWeight,
        positions: positionBuffer, output: queryOutput,
        tokenCount: 1, headCount: geometry.queryHeadCount,
        sections: geometry.mropeSections)
    try attention.encodeNormAndPartialMRoPE(
        commandBuffer: command, input: keyInput, weight: keyWeight,
        positions: positionBuffer, output: keyOutput,
        tokenCount: 1, headCount: geometry.keyValueHeadCount,
        sections: geometry.mropeSections)
    command.commit()
    await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    return NormRoPEOutput(
        query: readFloatBuffer(queryOutput, count: query.count),
        key: readFloatBuffer(keyOutput, count: key.count))
}

private func runOutputGate(
    _ input: [Float], rawGate: [Float],
    attention: QwenFullAttention, context: MetalContext
) async throws -> [Float] {
    let inputBuffer = makeFloatBuffer(input, device: context.device)
    let gateBuffer = makeFloatBuffer(rawGate, device: context.device)
    let output = try #require(context.device.makeBuffer(
        length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared))
    let command = try #require(context.queue.makeCommandBuffer())
    try attention.encodeOutputGate(
        commandBuffer: command, attention: inputBuffer, rawGate: gateBuffer,
        output: output, elementCount: input.count)
    command.commit()
    await command.completed()
    #expect(command.status == .completed)
    #expect(command.error == nil)
    return readFloatBuffer(output, count: input.count)
}

private func makeFloatBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Float>.stride,
                      options: .storageModeShared)!
}

private func makeInt32Buffer(_ values: [Int32], device: MTLDevice) -> MTLBuffer {
    device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Int32>.stride,
                      options: .storageModeShared)!
}

private func readFloatBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
}

private func expectSourceClose(_ actual: [Float], _ expectedBits: [UInt32], label: String) {
    let expected = expectedBits.map { Float(bitPattern: $0) }
    #expect(actual.count == expected.count, "\(label) count")
    for index in actual.indices where index < expected.count {
        let tolerance = 1e-7 + 1e-6 * abs(Double(expected[index]))
        #expect(abs(Double(actual[index]) - Double(expected[index])) <= tolerance,
                "\(label)[\(index)] actual=\(actual[index]) expected=\(expected[index])")
    }
}
