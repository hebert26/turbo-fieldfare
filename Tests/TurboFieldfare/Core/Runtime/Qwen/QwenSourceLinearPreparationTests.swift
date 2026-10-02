import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Expected arrays use the existing Swift CPU arithmetic and independently
/// expressed row/head mapping. No GPU candidate result generates a fixture.
@Suite(.serialized) struct QwenSourceLinearPreparationTests {
    private struct Inputs {
        var convolved: [Float]
        var rawBeta: [Float]
        var rawA: [Float]
        var aLog: [Float]
        var bias: [Float]
    }
    private struct Result {
        let query: [Float], key: [Float], value: [Float], beta: [Float], decay: [Float]
        let status: UInt32
    }
    private func fixture(_ c: QwenGatedDeltaNetConfiguration, tokens: Int, edges: Bool) -> Inputs {
        let heads = c.valueHeadCount
        var x = Inputs(
            convolved: (0..<(tokens * c.convolutionChannelCount)).map { Float(($0 * 37 + 11) % 991 - 495) / 128 },
            rawBeta: (0..<(tokens * heads)).map { Float(($0 * 31) % 201 - 100) / 8 },
            rawA: (0..<(tokens * heads)).map { Float(($0 * 19) % 251 - 125) / 8 },
            aLog: (0..<heads).map { Float($0 % 7) / 4 },
            bias: (0..<heads).map { Float($0 % 9 - 4) / 8 })
        if edges {
            let betaEdges: [Float] = [-Float.greatestFiniteMagnitude, -104, -100, -90, -88,
                -87, -0.0, 0, Float(bitPattern: 1), Float(bitPattern: 0x00800001),
                87, 88, 90, 100, 104, Float.greatestFiniteMagnitude]
            let aEdges: [Float] = [-Float.greatestFiniteMagnitude, Float(-104).nextDown, -104,
                Float(-104).nextUp, -103.9, -103, -100, -90, -88, -87.3, -87, -86.5,
                -44, -43.7, -20, -1, -0.0, 0, Float(bitPattern: 1),
                Float(20).nextDown, 20, Float(20).nextUp, 21, 80]
            for i in x.rawBeta.indices { x.rawBeta[i] = betaEdges[i % betaEdges.count] }
            // Bias zero in this case places the strict20 threshold exactly at
            // the frozen float operands. Dense case separately checks addition.
            x.bias = [Float](repeating: 0, count: heads)
            for i in x.rawA.indices { x.rawA[i] = aEdges[i % aEdges.count] }
            let logs: [Float] = [-104, -103, -90, -88, -87, -1, 0, 1, 3]
            for h in x.aLog.indices { x.aLog[h] = logs[h % logs.count] }
            let copied: [UInt32] = [0, 0x80000000, 1, 2, 3, 0x007fffff, 0x00800000, 0x00800001]
            for i in copied.indices {
                x.convolved[i] = Float(bitPattern: copied[i])
                x.convolved[c.keyDimension + i] = Float(bitPattern: Array(copied.reversed())[i])
                x.convolved[2 * c.keyDimension + i] = Float(bitPattern: copied[i])
            }
        }
        return x
    }
    private func cpu(_ x: Inputs, _ c: QwenGatedDeltaNetConfiguration, tokens: Int) -> Result {
        let qWidth = c.valueHeadCount * c.keyHeadDimension
        var q = [Float](repeating: 0, count: tokens * qWidth), k = q
        var v = [Float](repeating: 0, count: tokens * c.valueDimension)
        var b = [Float](repeating: 0, count: tokens * c.valueHeadCount), d = b
        guard [x.convolved, x.rawBeta, x.rawA, x.aLog, x.bias].allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            return Result(query: q, key: k, value: v, beta: b, decay: d, status: 1)
        }
        for token in 0..<tokens {
            for targetHead in 0..<c.valueHeadCount {
                let sourceHead = targetHead / c.headsPerKeyHead
                for col in 0..<c.keyHeadDimension {
                    let to = token * qWidth + targetHead * c.keyHeadDimension + col
                    q[to] = x.convolved[token * c.convolutionChannelCount + sourceHead * c.keyHeadDimension + col]
                    k[to] = x.convolved[token * c.convolutionChannelCount + c.keyDimension + sourceHead * c.keyHeadDimension + col]
                }
            }
            for col in 0..<c.valueDimension {
                v[token * c.valueDimension + col] = x.convolved[token * c.convolutionChannelCount + 2 * c.keyDimension + col]
            }
            for head in 0..<c.valueHeadCount {
                let i = token * c.valueHeadCount + head
                b[i] = 1 / (1 + QwenOfficialSourceRouterArithmetic.exponential(-x.rawBeta[i]))
                let sum = x.rawA[i] + x.bias[head]
                let softplus = sum > 20 ? sum : QwenOfficialSourcePositiveLog1p.evaluate(
                    QwenOfficialSourceRouterArithmetic.exponential(sum))
                d[i] = -QwenOfficialSourceRouterArithmetic.exponential(x.aLog[head]) * softplus
            }
        }
        return Result(query: q, key: k, value: v, beta: b, decay: d,
                      status: (b + d).allSatisfy(\.isFinite) ? 0 : 2)
    }
    private func floatBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            try #require(device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared))
        }
    }
    private func execute(_ x: Inputs, _ c: QwenGatedDeltaNetConfiguration, tokens: Int,
                         context: MetalContext, preparation: QwenSourceLinearPreparation) throws -> Result {
        let inputs = try [x.convolved, x.rawBeta, x.rawA, x.aLog, x.bias].map { try floatBuffer($0, device: context.device) }
        let counts = [tokens * c.valueHeadCount * c.keyHeadDimension,
                      tokens * c.valueHeadCount * c.keyHeadDimension, tokens * c.valueDimension,
                      tokens * c.valueHeadCount, tokens * c.valueHeadCount]
        let outputs = try counts.map { try floatBuffer([Float](repeating: -777, count: $0), device: context.device) }
        let status = try #require(context.device.makeBuffer(length: 4, options: .storageModeShared))
        status.contents().storeBytes(of: UInt32.zero, as: UInt32.self)
        let command = try #require(context.queue.makeCommandBuffer())
        try preparation.encode(commandBuffer: command, tokenCount: tokens,
            convolved: inputs[0], rawBeta: inputs[1], rawA: inputs[2], aLog: inputs[3], timeStepBias: inputs[4],
            query: outputs[0], key: outputs[1], value: outputs[2], beta: outputs[3], logDecay: outputs[4], status: status)
        command.commit(); command.waitUntilCompleted()
        #expect(command.status == .completed)
        if let error = command.error { throw error }
        let rows = zip(outputs, counts).map { buffer, count in
            Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
        }
        return Result(query: rows[0], key: rows[1], value: rows[2], beta: rows[3], decay: rows[4],
                      status: status.contents().load(as: UInt32.self))
    }
    private func exact(_ actual: [Float], _ expected: [Float], label: String) {
        #expect(actual.count == expected.count)
        let mismatches = zip(actual, expected).enumerated().filter { $0.element.0.bitPattern != $0.element.1.bitPattern }
        #expect(mismatches.isEmpty, "\(label): \(mismatches.count) bit differences")
        if let first = mismatches.first {
            #expect(first.element.0.bitPattern == first.element.1.bitPattern, "\(label) first index \(first.offset)")
        }
    }
    @Test func actualGeometryPreparedArraysMatchCPUExactly() throws {
        let c = try QwenGatedDeltaNetConfiguration.official(), context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(context: context, configuration: c)
        for tokens in [1, 3, 17] {
            for edges in [false, true] {
                let x = fixture(c, tokens: tokens, edges: edges), expected = cpu(x, c, tokens: tokens)
                let actual = try execute(x, c, tokens: tokens, context: context, preparation: preparation)
                #expect(expected.status == 0); #expect(actual.status == expected.status)
                exact(actual.query, expected.query, label: "query")
                exact(actual.key, expected.key, label: "key")
                exact(actual.value, expected.value, label: "value")
                exact(actual.beta, expected.beta, label: "beta")
                exact(actual.decay, expected.decay, label: "decay")
                // Contiguous replicated head mapping must differ from modulo
                // grouping. This control is independent of candidate output.
                #expect(expected.query[c.keyHeadDimension].bitPattern == expected.query[0].bitPattern)
                #expect(expected.query[2 * c.keyHeadDimension].bitPattern != expected.query[0].bitPattern)
            }
        }
    }
    @Test func boundaryRawDomainsMatchCPUAcrossTwoAndThirtyTwoTokenChunks() throws {
        let c = try QwenGatedDeltaNetConfiguration.official()
        let context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(context: context, configuration: c)
        for tokens in [2, 32] {
            var x = fixture(c, tokens: tokens, edges: false)
            let beta: [Float] = [-104, -100, -20, -Float.leastNonzeroMagnitude,
                                 -Float(bitPattern: 1), Float(bitPattern: 1), 0, 20, 100, 104]
            let a: [Float] = [-104, -20, -Float.leastNonzeroMagnitude, 0,
                              19.999, 20, 20.001, 100]
            let logs: [Float] = [-104, -100, -20, -0.0, 0, -20, -104, -100]
            for index in x.rawBeta.indices { x.rawBeta[index] = beta[index % beta.count] }
            for index in x.rawA.indices { x.rawA[index] = a[index % a.count] }
            for index in x.aLog.indices { x.aLog[index] = logs[index % logs.count] }
            x.bias = x.bias.enumerated().map { index, value in
                index.isMultiple(of: 2) ? -value : value
            }
            x.rawA[0] = Float(bitPattern: 1)
            x.bias[0] = -Float(bitPattern: 1)
            x.rawA[1] = Float(bitPattern: 0x80000001)
            x.bias[1] = Float(bitPattern: 1)
            let expected = cpu(x, c, tokens: tokens)
            let actual = try execute(x, c, tokens: tokens, context: context, preparation: preparation)
            #expect(expected.status == actual.status)
            if expected.status == 0 {
                exact(actual.query, expected.query, label: "boundary query \(tokens)")
                exact(actual.key, expected.key, label: "boundary key \(tokens)")
                exact(actual.value, expected.value, label: "boundary value \(tokens)")
                exact(actual.beta, expected.beta, label: "boundary beta \(tokens)")
                exact(actual.decay, expected.decay, label: "boundary decay \(tokens)")
            }
        }

        var overflow = fixture(c, tokens: 1, edges: false)
        overflow.rawA[0] = 100
        overflow.aLog[0] = 100
        overflow.bias[0] = 0
        let expectedOverflow = cpu(overflow, c, tokens: 1)
        let actualOverflow = try execute(
            overflow, c, tokens: 1, context: context, preparation: preparation)
        #expect(expectedOverflow.status == 2)
        #expect(actualOverflow.status == expectedOverflow.status)
    }

    @Test func seededFiniteRawDomainsMatchPreparedArraysExactly() throws {
        let c = try QwenGatedDeltaNetConfiguration.official(), context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(context: context, configuration: c)
        var seed: UInt64 = 0x6938471527a42efd
        func next() -> UInt32 {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return UInt32(truncatingIfNeeded: seed >> 16)
        }
        for chunk in 0..<8 {
            let tokens = 64
            var x = fixture(c, tokens: tokens, edges: false)
            for i in x.rawBeta.indices {
                let bits = next()
                x.rawBeta[i] = Float(bitPattern: (bits & 0x80000000) | ((bits & 0x7fffffff) % 0x7f800000))
                let aBits = next()
                x.rawA[i] = Float(bitPattern: (aBits & 0x80000000) | ((aBits & 0x7fffffff) % Float(20).bitPattern))
            }
            // Nonpositive factors keep every expected product finite, so a
            // flagged lane cannot hide comparisons for other lanes in a chunk.
            for h in x.aLog.indices {
                x.aLog[h] = -Float(next() % 10_401) / 100
                x.bias[h] = Float(Int(next() % 257) - 128) / 128
            }
            let expected = cpu(x, c, tokens: tokens)
            let actual = try execute(x, c, tokens: tokens, context: context, preparation: preparation)
            #expect(expected.status == 0)
            #expect(actual.status == 0)
            exact(actual.query, expected.query, label: "seeded query \(chunk)")
            exact(actual.key, expected.key, label: "seeded key \(chunk)")
            exact(actual.value, expected.value, label: "seeded value \(chunk)")
            exact(actual.beta, expected.beta, label: "seeded beta \(chunk)")
            exact(actual.decay, expected.decay, label: "seeded decay \(chunk)")
        }
    }

    @Test func nonfiniteInputAndPreparedOverflowFailClosed() throws {
        let c = try QwenGatedDeltaNetConfiguration.official(), context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(context: context, configuration: c)
        for location in 0..<5 {
            var x = fixture(c, tokens: 1, edges: false)
            switch location {
            case 0: x.convolved[x.convolved.count - 1] = .nan
            case 1: x.rawBeta[0] = .infinity
            case 2: x.rawA[0] = .nan
            case 3: x.aLog[0] = .infinity
            default: x.bias[0] = .nan
            }
            let actual = try execute(x, c, tokens: 1, context: context, preparation: preparation)
            #expect(actual.status & 1 == 1)
            if location > 0 {
                #expect(actual.beta[0].bitPattern == 0)
                #expect(actual.decay[0].bitPattern == 0)
            }
        }
        var overflow = fixture(c, tokens: 1, edges: false)
        overflow.aLog[0] = 88; overflow.rawA[0] = 21; overflow.bias[0] = 0
        #expect(cpu(overflow, c, tokens: 1).status == 2)
        #expect(try execute(overflow, c, tokens: 1, context: context, preparation: preparation).status == 2)
    }
    @Test func tinyLogThresholdsAndDecayProductsMatchIndependentCPU() throws {
        let context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(context: context,
            configuration: QwenGatedDeltaNetConfiguration.official())
        // Values straddle both private shortcut thresholds and the CPU
        // half-underflow ties, independently of which exp inputs reach them.
        let frozenBits: [UInt32] = [0, 1, 2, 3, 4, 5, 6, 7,
            0x007ffffd, 0x007ffffe, 0x007fffff, 0x00800000, 0x00800001,
            0x00fffffe, 0x00ffffff, 0x01000000, 0x01000001, 0x01000002,
            0x1ffffffe, 0x1fffffff, 0x20000000, 0x20000001, 0x20000002,
            0x3f000000, 0x3f800000, 0x40000000, 0x4de75b21,
            0x7e967697, 0x7e967698, 0x7e967699]
        var values = frozenBits.map { Float(bitPattern: $0) }
        var state: UInt64 = 0x2147b3915d9a8ce1
        for _ in 0..<65_536 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            values.append(Float(bitPattern: UInt32(truncatingIfNeeded: state >> 16) % 0x7e967699))
        }
        let multiplierBits: [UInt32] = [0, 1, 3, 0x007fffff, 0x00800001,
            0x3e800000, 0x3f000000, 0x3f800000, 0x40000000, 0x42000000, 0x7f7fffff]
        // Cross every threshold operand with every multiplier, then retain
        // seeded broad mantissa/exponent coverage within a fixed small bound.
        var inputs: [Float] = [], multipliers: [Float] = []
        for (index, value) in values.enumerated() {
            if index < frozenBits.count {
                for bits in multiplierBits { inputs.append(value); multipliers.append(Float(bitPattern: bits)) }
            } else {
                inputs.append(value)
                state = state &* 6364136223846793005 &+ 1442695040888963407
                multipliers.append(Float(bitPattern: UInt32(truncatingIfNeeded: state >> 16) % 0x7f7fffff))
            }
        }
        let expectedLogs = inputs.map { QwenOfficialSourcePositiveLog1p.evaluate($0) }
        let expectedProducts = zip(inputs, multipliers).map { $0 * $1 }
        let inputBuffer = try floatBuffer(inputs, device: context.device)
        let multiplierBuffer = try floatBuffer(multipliers, device: context.device)
        let logs = try floatBuffer([Float](repeating: -777, count: inputs.count), device: context.device)
        let products = try floatBuffer([Float](repeating: -777, count: inputs.count), device: context.device)
        let command = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        let pipeline = preparation.mathProbePipeline
        var count = UInt32(inputs.count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&count, length: 4, index: 0)
        encoder.setBuffer(inputBuffer, offset: 0, index: 1)
        encoder.setBuffer(multiplierBuffer, offset: 0, index: 2)
        encoder.setBuffer(logs, offset: 0, index: 3)
        encoder.setBuffer(products, offset: 0, index: 4)
        let width = min(pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: inputs.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, inputs.count), height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        #expect(command.status == .completed)
        if let error = command.error { throw error }
        let actualLogs = Array(UnsafeBufferPointer(start: logs.contents().assumingMemoryBound(to: Float.self), count: inputs.count))
        let actualProducts = Array(UnsafeBufferPointer(start: products.contents().assumingMemoryBound(to: Float.self), count: inputs.count))
        exact(actualLogs, expectedLogs, label: "positive log1p")
        exact(actualProducts, expectedProducts, label: "decay product")
    }

    @Test func encodingRejectsInvalidCountAliasingLengthAndStaleStatusBeforeSubmission() throws {
        let context = try MetalContext(), c = try QwenGatedDeltaNetConfiguration.official()
        let preparation = try QwenSourceLinearPreparation(context: context, configuration: c)
        let counts = [c.convolutionChannelCount, c.valueHeadCount, c.valueHeadCount,
            c.valueHeadCount, c.valueHeadCount, c.valueHeadCount * c.keyHeadDimension,
            c.valueHeadCount * c.keyHeadDimension, c.valueDimension, c.valueHeadCount, c.valueHeadCount, 1]
        let buffers = try counts.map { try floatBuffer([Float](repeating: 0, count: $0), device: context.device) }
        func encode(_ inputs: [MTLBuffer], tokens: Int, command: MTLCommandBuffer) throws {
            try preparation.encode(commandBuffer: command, tokenCount: tokens,
                convolved: inputs[0], rawBeta: inputs[1], rawA: inputs[2], aLog: inputs[3], timeStepBias: inputs[4],
                query: inputs[5], key: inputs[6], value: inputs[7], beta: inputs[8], logDecay: inputs[9], status: inputs[10])
        }
        for tokens in [0, 257] {
            let command = try #require(context.queue.makeCommandBuffer())
            #expect(throws: (any Error).self) { try encode(buffers, tokens: tokens, command: command) }
            #expect(command.status == .notEnqueued)
        }
        var alias = buffers; alias[6] = alias[5]
        let aliasCommand = try #require(context.queue.makeCommandBuffer())
        #expect(throws: (any Error).self) { try encode(alias, tokens: 1, command: aliasCommand) }
        var short = buffers; short[0] = try floatBuffer([0], device: context.device)
        let shortCommand = try #require(context.queue.makeCommandBuffer())
        #expect(throws: (any Error).self) { try encode(short, tokens: 1, command: shortCommand) }
        buffers[10].contents().storeBytes(of: UInt32(1), as: UInt32.self)
        let staleCommand = try #require(context.queue.makeCommandBuffer())
        #expect(throws: (any Error).self) { try encode(buffers, tokens: 1, command: staleCommand) }
        #expect(staleCommand.status == .notEnqueued)
    }

}
