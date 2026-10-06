import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenBF16MetalTests {
    @Test func actualMetalEmbeddingAndAllProjectionRolesMatchIndependentFP32CPUReference() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: kernelTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeWeights(context: context, source: source)

        let tokenIDs = try sharedBuffer(QwenBF16TestFixture.embeddingTokenIDs,
                                        device: context.device)
        let embeddingOutput = try sharedBuffer(
            [Float](repeating: -91, count: 3 * 5), device: context.device)
        let embeddingCommand = try #require(context.queue.makeCommandBuffer())
        try weights.encodeEmbedding(commandBuffer: embeddingCommand,
                                    tensorName: "embedding", tokenIDs: tokenIDs,
                                    tokenCount: 3, output: embeddingOutput)
        try complete(embeddingCommand)
        expectFrozenClose(
            readFloats(embeddingOutput, count: 15),
            QwenBF16TestFixture.cpuEmbedding(
                tokenIDs: QwenBF16TestFixture.embeddingTokenIDs,
                bits: QwenBF16TestFixture.matrixBits, rows: 5, columns: 5))

        let input = try sharedBuffer(QwenBF16TestFixture.inputs, device: context.device)
        let projectionCases: [(String, Int, [UInt16])] = [
            ("head", 5, QwenBF16TestFixture.matrixBits),
            ("router", 9, QwenBF16TestFixture.routerBits),
            ("dense", 5, QwenBF16TestFixture.matrixBits),
            ("sharedGate", 5, QwenBF16TestFixture.matrixBits),
            ("sharedUp", 5, QwenBF16TestFixture.matrixBits),
            ("sharedDown", 5, QwenBF16TestFixture.matrixBits),
            ("sharedOutputGate", 1, QwenBF16TestFixture.sharedOutputGateBits),
        ]
        for (name, rows, bits) in projectionCases {
            let output = try sharedBuffer(
                [Float](repeating: -73, count: 3 * rows), device: context.device)
            let command = try #require(context.queue.makeCommandBuffer())
            try weights.encodeProjection(commandBuffer: command, tensorName: name,
                                         input: input, tokenCount: 3, output: output)
            try complete(command)
            let actual = readFloats(output, count: 3 * rows)
            let expected = QwenBF16TestFixture.cpuProjection(
                input: QwenBF16TestFixture.inputs, matrixBits: bits,
                rows: rows, columns: 5)
            expectFrozenClose(actual, expected)
            #expect(actual.allSatisfy { $0.isFinite })

            if name == "head" {
                let fp32Expected = expected[0]
                #expect(fp32Expected == QwenBF16TestFixture.fp32DiscriminatorExpected)
                #expect(fp32Expected == QwenBF16TestFixture.discriminatorMathematicalValue)
                let halfNearest = QwenBF16TestFixture.nearestFloat16Value
                let frozenLimit = QwenBF16TestFixture.absoluteTolerance
                    + QwenBF16TestFixture.relativeTolerance * abs(fp32Expected)
                #expect(abs(fp32Expected - halfNearest) > frozenLimit)
                #expect(!QwenBF16TestFixture.matchesFrozenTolerance(halfNearest, fp32Expected))
                #expect(QwenBF16TestFixture.matchesFrozenTolerance(actual[0], fp32Expected))
            }
        }

        // The same resident weights remain reusable across independent submissions.
        let repeatOutput = try sharedBuffer([Float](repeating: 0, count: 15), device: context.device)
        let repeatCommand = try #require(context.queue.makeCommandBuffer())
        try weights.encodeProjection(commandBuffer: repeatCommand, tensorName: "head",
                                     input: input, tokenCount: 3, output: repeatOutput)
        try complete(repeatCommand)
        expectFrozenClose(readFloats(repeatOutput, count: 15),
                          QwenBF16TestFixture.cpuProjection(
                            input: QwenBF16TestFixture.inputs,
                            matrixBits: QwenBF16TestFixture.matrixBits,
                            rows: 5, columns: 5))
    }

    @Test func actualMetalEmbeddingPreservesLiteralBF16EdgeBitPatterns() throws {
        let bits: [UInt16] = [0x0000, 0x8000, 0x7f80, 0xff80,
                              0x4780, 0x0080, 0x0001, 0xc780]
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "edgeEmbedding", rows: 1,
                                  columns: bits.count, bits: bits),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try QwenBF16Weights(
            context: context, source: source.handle,
            specifications: [QwenBF16TensorSpec(
                name: "edgeEmbedding", shardName: source.shardName,
                role: .embedding, rows: 1, columns: bits.count)],
            residencyBudget: UInt64(bits.count * MemoryLayout<UInt16>.stride))
        let tokenIDs = try sharedBuffer([UInt32(0)], device: context.device)
        let output = try sharedBuffer([Float](repeating: 27, count: bits.count),
                                      device: context.device)
        let command = try #require(context.queue.makeCommandBuffer())
        try weights.encodeEmbedding(commandBuffer: command,
                                    tensorName: "edgeEmbedding", tokenIDs: tokenIDs,
                                    tokenCount: 1, output: output)
        try complete(command)

        let actual = readFloats(output, count: bits.count)
        let expectedBitPatterns = bits.map { UInt32($0) << 16 }
        let actualBitPatterns = actual.map { $0.bitPattern }
        #expect(actualBitPatterns == expectedBitPatterns)
        #expect(expectedBitPatterns[0] == 0x0000_0000)
        #expect(expectedBitPatterns[1] == 0x8000_0000)
        #expect(actual[2].isInfinite && actual[2].sign == .plus)
        #expect(actual[3].isInfinite && actual[3].sign == .minus)
        #expect(actual[4].isFinite && actual[4] == 65_536)
        #expect(actual[5].isNormal)
        #expect(actual[6].isSubnormal)
        #expect(actual[7].isFinite && actual[7] == -65_536)

        let float16RoundTrip = bits.map {
            Float(Float16(QwenBF16TestFixture.float($0))).bitPattern
        }
        #expect(float16RoundTrip != expectedBitPatterns)
    }

    @Test func safeModePropagatesBF16NaNAndFP32InputNaN() throws {
        let nanBits: [UInt16] = [QwenBF16TestFixture.nanBits, 0x3f80, 0, 0, 0]
        let source = try QwenBF16SyntheticSource.make(tensors: [
            QwenBF16LiteralTensor(name: "nanWeight", rows: 1, columns: 5, bits: nanBits),
            QwenBF16LiteralTensor(name: "finite", rows: 1, columns: 5,
                                  bits: [0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80]),
        ])
        defer { source.remove() }
        let context = try MetalContext()
        let specs = [
            QwenBF16TensorSpec(name: "nanWeight", shardName: source.shardName,
                               role: .dense, rows: 1, columns: 5),
            QwenBF16TensorSpec(name: "finite", shardName: source.shardName,
                               role: .dense, rows: 1, columns: 5),
        ]
        let weights = try QwenBF16Weights(context: context, source: source.handle,
                                          specifications: specs, residencyBudget: 20)
        let storedNaN = try #require(weights.inspectedChunks.first {
            $0.name == "nanWeight"
        })
        #expect(readBF16Words(storedNaN.buffer) == nanBits)

        let finiteInput = try sharedBuffer([1.0, 0, 0, 0, 0], device: context.device)
        let nanWeightOutput = try sharedBuffer([Float.zero], device: context.device)
        let nanWeightCommand = try #require(context.queue.makeCommandBuffer())
        try weights.encodeProjection(commandBuffer: nanWeightCommand, tensorName: "nanWeight",
                                     input: finiteInput, tokenCount: 1,
                                     output: nanWeightOutput)
        try complete(nanWeightCommand)
        #expect(readFloats(nanWeightOutput, count: 1)[0].isNaN)

        let nanInput = try sharedBuffer([Float.nan, 1, 2, 3, 4], device: context.device)
        let finiteOutput = try sharedBuffer([Float.zero], device: context.device)
        let nanInputCommand = try #require(context.queue.makeCommandBuffer())
        try weights.encodeProjection(commandBuffer: nanInputCommand, tensorName: "finite",
                                     input: nanInput, tokenCount: 1, output: finiteOutput)
        try complete(nanInputCommand)
        #expect(readFloats(finiteOutput, count: 1)[0].isNaN)
    }

    @Test func invalidDispatchAndEmbeddingBoundsFailBeforeEncoding() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: kernelTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let weights = try makeWeights(context: context, source: source)

        let badIDs = try sharedBuffer([UInt32(5)], device: context.device)
        let untouchedEmbedding = try sharedBuffer([Float(19), 19, 19, 19, 19],
                                                  device: context.device)
        let badEmbeddingCommand = try #require(context.queue.makeCommandBuffer())
        expectWeightError({ if case .invalidGeometry = $0 { return true }; return false }) {
            try weights.encodeEmbedding(commandBuffer: badEmbeddingCommand,
                                        tensorName: "embedding", tokenIDs: badIDs,
                                        tokenCount: 1, output: untouchedEmbedding)
        }
        #expect(badEmbeddingCommand.status == .notEnqueued)
        #expect(readFloats(untouchedEmbedding, count: 5) == [19, 19, 19, 19, 19])

        let input = try sharedBuffer([Float](repeating: 1, count: 5), device: context.device)
        let shortOutput = try sharedBuffer([Float.zero], device: context.device)
        let shortOutputCommand = try #require(context.queue.makeCommandBuffer())
        expectWeightError({ if case .invalidBuffer = $0 { return true }; return false }) {
            try weights.encodeProjection(commandBuffer: shortOutputCommand, tensorName: "head",
                                         input: input, tokenCount: 1, output: shortOutput)
        }
        #expect(shortOutputCommand.status == .notEnqueued)

        let emptyCountCommand = try #require(context.queue.makeCommandBuffer())
        let validOutput = try sharedBuffer([Float.zero], device: context.device)
        expectWeightError({ if case .invalidGeometry = $0 { return true }; return false }) {
            try weights.encodeProjection(commandBuffer: emptyCountCommand, tensorName: "head",
                                         input: input, tokenCount: 0, output: validOutput)
        }
        #expect(emptyCountCommand.status == .notEnqueued)

        let embeddingProjectionCommand = try #require(context.queue.makeCommandBuffer())
        expectWeightError({ if case .missingTensor = $0 { return true }; return false }) {
            try weights.encodeProjection(commandBuffer: embeddingProjectionCommand,
                                         tensorName: "embedding", input: input,
                                         tokenCount: 1, output: validOutput)
        }
        #expect(embeddingProjectionCommand.status == .notEnqueued)
    }

    @Test func qwenMoESharedBF16EntryAddsIndependentPositiveReference() throws {
        let source = try QwenBF16SyntheticSource.make(tensors: sharedTensors())
        defer { source.remove() }
        let context = try MetalContext()
        let specs = [
            QwenBF16TensorSpec(name: "gate", shardName: source.shardName,
                               role: .sharedGate, rows: 5, columns: 5),
            QwenBF16TensorSpec(name: "up", shardName: source.shardName,
                               role: .sharedUp, rows: 5, columns: 5),
            QwenBF16TensorSpec(name: "down", shardName: source.shardName,
                               role: .sharedDown, rows: 5, columns: 5),
            QwenBF16TensorSpec(name: "outputGate", shardName: source.shardName,
                               role: .sharedOutputGate, rows: 1, columns: 5),
        ]
        let weights = try QwenBF16Weights(context: context, source: source.handle,
                                          specifications: specs, residencyBudget: 160)
        let configuration = try QwenMoEConfiguration(
            hiddenSize: 5, expertCount: 8, routedIntermediateSize: 2,
            sharedIntermediateSize: 5)
        let moe = try QwenMoE(context: context, configuration: configuration)
        let scratch = try moe.makeScratch()
        let hidden = try sharedBuffer(QwenBF16TestFixture.sharedInput, device: context.device)
        let output = try sharedBuffer(QwenBF16TestFixture.sharedInitialOutput,
                                      device: context.device)
        let command = try #require(context.queue.makeCommandBuffer())
        try moe.encodeSharedBF16(
            commandBuffer: command, hidden: hidden, weights: weights,
            names: QwenBF16SharedNames(gate: "gate", up: "up", down: "down",
                                       outputGate: "outputGate"),
            scratch: scratch, output: output)
        try complete(command)

        let actual = readFloats(output, count: 5)
        let expected = QwenBF16TestFixture.cpuSharedMoEExpected()
        let missingOutputGate = QwenBF16TestFixture.cpuSharedMoEExpected(
            applyOutputGate: false)
        #expect(actual.allSatisfy { $0.isFinite })
        for (index, pair) in zip(actual, expected).enumerated() {
            let bound = QwenBF16TestFixture.compoundAbsoluteTolerance
                + QwenBF16TestFixture.compoundRelativeTolerance * abs(pair.1)
            #expect(abs(pair.0 - pair.1) <= bound,
                    "shared MoE index \(index): \(pair.0) != \(pair.1)")
            #expect(pair.0 > QwenBF16TestFixture.sharedInitialOutput[index])
            #expect(abs(pair.1 - missingOutputGate[index]) > 10 * bound,
                    "output-gate negative control at index \(index)")
        }
    }
}

private func kernelTensors() -> [QwenBF16LiteralTensor] {
    [
        QwenBF16LiteralTensor(name: "embedding", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.matrixBits),
        QwenBF16LiteralTensor(name: "head", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.matrixBits),
        QwenBF16LiteralTensor(name: "router", rows: 9, columns: 5,
                              bits: QwenBF16TestFixture.routerBits),
        QwenBF16LiteralTensor(name: "dense", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.matrixBits),
        QwenBF16LiteralTensor(name: "sharedGate", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.matrixBits),
        QwenBF16LiteralTensor(name: "sharedUp", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.matrixBits),
        QwenBF16LiteralTensor(name: "sharedDown", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.matrixBits),
        QwenBF16LiteralTensor(name: "sharedOutputGate", rows: 1, columns: 5,
                              bits: QwenBF16TestFixture.sharedOutputGateBits),
    ]
}

private func sharedTensors() -> [QwenBF16LiteralTensor] {
    [
        QwenBF16LiteralTensor(name: "gate", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.sharedGateBits),
        QwenBF16LiteralTensor(name: "up", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.sharedUpBits),
        QwenBF16LiteralTensor(name: "down", rows: 5, columns: 5,
                              bits: QwenBF16TestFixture.sharedDownBits),
        QwenBF16LiteralTensor(name: "outputGate", rows: 1, columns: 5,
                              bits: QwenBF16TestFixture.sharedOutputGateBits),
    ]
}

private func makeWeights(context: MetalContext,
                         source: QwenBF16SyntheticSource) throws -> QwenBF16Weights {
    let specs = [
        QwenBF16TensorSpec(name: "embedding", shardName: source.shardName,
                           role: .embedding, rows: 5, columns: 5),
        QwenBF16TensorSpec(name: "head", shardName: source.shardName,
                           role: .head, rows: 5, columns: 5),
        QwenBF16TensorSpec(name: "router", shardName: source.shardName,
                           role: .router, rows: 9, columns: 5),
        QwenBF16TensorSpec(name: "dense", shardName: source.shardName,
                           role: .dense, rows: 5, columns: 5),
        QwenBF16TensorSpec(name: "sharedGate", shardName: source.shardName,
                           role: .sharedGate, rows: 5, columns: 5),
        QwenBF16TensorSpec(name: "sharedUp", shardName: source.shardName,
                           role: .sharedUp, rows: 5, columns: 5),
        QwenBF16TensorSpec(name: "sharedDown", shardName: source.shardName,
                           role: .sharedDown, rows: 5, columns: 5),
        QwenBF16TensorSpec(name: "sharedOutputGate", shardName: source.shardName,
                           role: .sharedOutputGate, rows: 1, columns: 5),
    ]
    return try QwenBF16Weights(context: context, source: source.handle,
                               specifications: specs, residencyBudget: 500,
                               maximumChunkBytes: 22, checkpoint: { _ in })
}

private func sharedBuffer<T>(_ values: [T], device: MTLDevice) throws -> MTLBuffer {
    guard !values.isEmpty,
          let buffer = device.makeBuffer(bytes: values,
                                         length: values.count * MemoryLayout<T>.stride,
                                         options: .storageModeShared) else {
        throw QwenBF16MetalTestError.bufferAllocation
    }
    return buffer
}

private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let values = buffer.contents().assumingMemoryBound(to: Float.self)
    return Array(UnsafeBufferPointer(start: values, count: count))
}

private func readBF16Words(_ buffer: MTLBuffer) -> [UInt16] {
    let values = buffer.contents().assumingMemoryBound(to: UInt16.self)
    return Array(UnsafeBufferPointer(start: values, count: buffer.length / 2))
}

private func complete(_ commandBuffer: MTLCommandBuffer) throws {
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    try checkCommandBufferError(commandBuffer)
    #expect(commandBuffer.status == .completed)
    #expect(commandBuffer.error == nil)
}

private func expectFrozenClose(_ actual: [Float], _ expected: [Float],
                               sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(actual.count == expected.count, sourceLocation: sourceLocation)
    for (index, pair) in zip(actual, expected).enumerated() {
        #expect(QwenBF16TestFixture.matchesFrozenTolerance(pair.0, pair.1),
                "index \(index): \(pair.0) != \(pair.1)",
                sourceLocation: sourceLocation)
    }
}

private enum QwenBF16MetalTestError: Error { case bufferAllocation }

private func expectWeightError(
    _ matches: (QwenBF16WeightError) -> Bool,
    operation: () throws -> Void,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    do {
        try operation()
        Issue.record("Expected a QwenBF16WeightError", sourceLocation: sourceLocation)
    } catch let error as QwenBF16WeightError {
        #expect(matches(error), "Unexpected QwenBF16WeightError: \(error)",
                sourceLocation: sourceLocation)
    } catch {
        Issue.record("Expected QwenBF16WeightError, got \(error)", sourceLocation: sourceLocation)
    }
}
