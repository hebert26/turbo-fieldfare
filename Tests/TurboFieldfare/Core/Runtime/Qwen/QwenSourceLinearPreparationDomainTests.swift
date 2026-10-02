import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Focused domain coverage for the opt-in source preparation discriminator.
/// Expected values are produced by the pinned Swift CPU helpers. This suite
/// intentionally keeps the default run small. The dense stratified sweep is
/// separately opted in below.
@Suite(.serialized) struct QwenSourceLinearPreparationDomainTests {
    private struct Inputs {
        var convolved: [Float]
        var rawBeta: [Float]
        var rawA: [Float]
        var aLog: [Float]
        var bias: [Float]
    }

    private struct Result {
        let query: [Float]
        let key: [Float]
        let value: [Float]
        let beta: [Float]
        let decay: [Float]
        let status: UInt32
    }

    private func makeBuffer(_ values: [Float], device: MTLDevice) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            try #require(device.makeBuffer(
                bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared))
        }
    }

    private func configuration() throws -> QwenGatedDeltaNetConfiguration {
        try QwenGatedDeltaNetConfiguration(
            hiddenSize: 32,
            keyHeadCount: 1,
            valueHeadCount: 2,
            keyHeadDimension: 4,
            valueHeadDimension: 4)
    }

    private func cpu(_ input: Inputs, _ c: QwenGatedDeltaNetConfiguration,
                     tokens: Int) -> Result {
        let qWidth = c.valueHeadCount * c.keyHeadDimension
        var query = [Float](repeating: 0, count: tokens * qWidth)
        var key = query
        var value = [Float](repeating: 0, count: tokens * c.valueDimension)
        var beta = [Float](repeating: 0, count: tokens * c.valueHeadCount)
        var decay = beta

        guard [input.convolved, input.rawBeta, input.rawA, input.aLog, input.bias]
            .allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            return Result(query: query, key: key, value: value, beta: beta,
                          decay: decay, status: 1)
        }

        for token in 0..<tokens {
            for targetHead in 0..<c.valueHeadCount {
                let sourceHead = targetHead / c.headsPerKeyHead
                for column in 0..<c.keyHeadDimension {
                    let output = token * qWidth + targetHead * c.keyHeadDimension + column
                    let source = token * c.convolutionChannelCount
                        + sourceHead * c.keyHeadDimension + column
                    query[output] = input.convolved[source]
                    key[output] = input.convolved[token * c.convolutionChannelCount
                        + c.keyDimension + sourceHead * c.keyHeadDimension + column]
                }
            }
            for column in 0..<c.valueDimension {
                value[token * c.valueDimension + column] = input.convolved[
                    token * c.convolutionChannelCount + 2 * c.keyDimension + column]
            }
            for head in 0..<c.valueHeadCount {
                let index = token * c.valueHeadCount + head
                beta[index] = 1 / (1 + QwenOfficialSourceRouterArithmetic.exponential(
                    -input.rawBeta[index]))
                let sum = input.rawA[index] + input.bias[head]
                let softplus = sum > 20
                    ? sum
                    : QwenOfficialSourcePositiveLog1p.evaluate(
                        QwenOfficialSourceRouterArithmetic.exponential(sum))
                decay[index] = -QwenOfficialSourceRouterArithmetic.exponential(input.aLog[head])
                    * softplus
            }
        }
        return Result(query: query, key: key, value: value, beta: beta, decay: decay,
                      status: (beta + decay).allSatisfy(\.isFinite) ? 0 : 2)
    }

    private func execute(_ input: Inputs, _ c: QwenGatedDeltaNetConfiguration,
                         tokens: Int, context: MetalContext,
                         preparation: QwenSourceLinearPreparation) throws -> Result {
        let source = try [input.convolved, input.rawBeta, input.rawA, input.aLog, input.bias]
            .map { try makeBuffer($0, device: context.device) }
        let counts = [tokens * c.valueHeadCount * c.keyHeadDimension,
                      tokens * c.valueHeadCount * c.keyHeadDimension,
                      tokens * c.valueDimension,
                      tokens * c.valueHeadCount,
                      tokens * c.valueHeadCount]
        let destination = try counts.map {
            try makeBuffer([Float](repeating: -777, count: $0), device: context.device)
        }
        let status = try #require(context.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride, options: .storageModeShared))
        status.contents().storeBytes(of: UInt32.zero, as: UInt32.self)
        let command = try #require(context.queue.makeCommandBuffer())
        try preparation.encode(commandBuffer: command, tokenCount: tokens,
            convolved: source[0], rawBeta: source[1], rawA: source[2], aLog: source[3],
            timeStepBias: source[4], query: destination[0], key: destination[1],
            value: destination[2], beta: destination[3], logDecay: destination[4], status: status)
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed)
        if let error = command.error { throw error }
        let rows = zip(destination, counts).map { buffer, count in
            Array(UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
        }
        return Result(query: rows[0], key: rows[1], value: rows[2], beta: rows[3],
                      decay: rows[4], status: status.contents().load(as: UInt32.self))
    }

    private func exact(_ actual: [Float], _ expected: [Float], _ label: String) {
        #expect(actual.count == expected.count)
        let mismatches = zip(actual, expected).enumerated().filter {
            $0.element.0.bitPattern != $0.element.1.bitPattern
        }
        #expect(mismatches.isEmpty, "\(label): \(mismatches.count) bit differences")
        if let first = mismatches.first {
            #expect(first.element.0.bitPattern == first.element.1.bitPattern,
                    "\(label) first index \(first.offset)")
        }
    }

    private func probe(_ values: [Float], _ multipliers: [Float],
                       context: MetalContext, preparation: QwenSourceLinearPreparation)
        throws -> ([Float], [Float]) {
        #expect(values.count == multipliers.count)
        let input = try makeBuffer(values, device: context.device)
        let multiplier = try makeBuffer(multipliers, device: context.device)
        let logs = try makeBuffer([Float](repeating: -777, count: values.count), device: context.device)
        let products = try makeBuffer([Float](repeating: -777, count: values.count), device: context.device)
        let command = try #require(context.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        var count = UInt32(values.count)
        encoder.setComputePipelineState(preparation.mathProbePipeline)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(multiplier, offset: 0, index: 2)
        encoder.setBuffer(logs, offset: 0, index: 3)
        encoder.setBuffer(products, offset: 0, index: 4)
        let width = min(preparation.mathProbePipeline.threadExecutionWidth,
                        preparation.mathProbePipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: values.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, values.count), height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        #expect(command.status == .completed)
        if let error = command.error { throw error }
        let actualLogs = Array(UnsafeBufferPointer(
            start: logs.contents().assumingMemoryBound(to: Float.self), count: values.count))
        let actualProducts = Array(UnsafeBufferPointer(
            start: products.contents().assumingMemoryBound(to: Float.self), count: values.count))
        return (actualLogs, actualProducts)
    }

    @Test func finiteBetaAndSoftplusThresholdsMatchCPUExactly() throws {
        let c = try configuration()
        let context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(context: context, configuration: c)
        let tokens = 4
        let channels = c.convolutionChannelCount
        var convolved = [Float](repeating: 0, count: tokens * channels)
        let copiedBits: [UInt32] = [0, 0x80000000, 1, 0x00800000, 0x3f000000, 0x3f800000]
        for index in convolved.indices {
            convolved[index] = Float(bitPattern: copiedBits[index % copiedBits.count])
        }
        let input = Inputs(
            convolved: convolved,
            rawBeta: [-104, -100, -Float.leastNonzeroMagnitude, 0,
                      Float(bitPattern: 1), 20, 100, 104],
            rawA: [-104, Float(20).nextDown, 20, Float(20).nextUp,
                   -20, -1, 0, 100],
            aLog: [-104, -20],
            bias: [0, -0.0])
        #expect(input.convolved.count == tokens * c.convolutionChannelCount)
        #expect(input.rawBeta.count == tokens * c.valueHeadCount)
        #expect(input.rawA.count == tokens * c.valueHeadCount)
        let expected = cpu(input, c, tokens: tokens)
        let actual = try execute(input, c, tokens: tokens, context: context, preparation: preparation)
        #expect(expected.status == 0)
        #expect(actual.status == expected.status)
        exact(actual.query, expected.query, "query")
        exact(actual.key, expected.key, "key")
        exact(actual.value, expected.value, "value")
        exact(actual.beta, expected.beta, "beta finite inputs")
        exact(actual.decay, expected.decay, "softplus/product thresholds")

        // The positive exponential endpoint is intentionally a flagged output
        // case. It proves the [-104,100] exp domain reaches the prepared-output
        // finite-status guard without consuming an invalid result.
        var overflow = input
        overflow.aLog = [100, 100]
        let overflowExpected = cpu(overflow, c, tokens: tokens)
        let overflowActual = try execute(overflow, c, tokens: tokens, context: context,
                                         preparation: preparation)
        #expect(overflowExpected.status == 2)
        #expect(overflowActual.status == overflowExpected.status)
    }

    @Test func exponentMantissaStratifiedBetaArgumentsMatchCPUExactly() throws {
        let c = try configuration()
        let context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(context: context, configuration: c)

        // Each raw beta is the negation of an exp argument. Sampling the
        // mantissa bits in each magnitude bin tests the actual beta path over
        // exp arguments [-104,100], including subnormal and overflow ends.
        let mantissas: [UInt32] = [0, 1, 0x001fffff, 0x00400000, 0x00600000, 0x007fffff]
        var arguments: [Float] = [-104, -100, -20, -Float.leastNonzeroMagnitude,
                                  -Float(bitPattern: 1), -0.0, 0.0, Float(bitPattern: 1),
                                  20, 100]
        for exponent in -6...6 {
            let exponentBits = UInt32(exponent + 127) << 23
            for mantissa in mantissas {
                let magnitude = Float(bitPattern: exponentBits | mantissa)
                if magnitude <= 100 {
                    arguments.append(magnitude)
                    if magnitude != 0 { arguments.append(-magnitude) }
                }
            }
        }
        if !arguments.count.isMultiple(of: c.valueHeadCount) {
            arguments.append(0)
        }
        let tokens = arguments.count / c.valueHeadCount
        var convolved = [Float](repeating: 0, count: tokens * c.convolutionChannelCount)
        for index in convolved.indices {
            convolved[index] = Float(bitPattern: [0, 1, 0x00800000, 0x3f800000][index % 4])
        }
        let input = Inputs(
            convolved: convolved,
            rawBeta: arguments.map { -$0 },
            rawA: [Float](repeating: 0, count: tokens * c.valueHeadCount),
            aLog: [-104, -104],
            bias: [0, 0])
        #expect(input.rawBeta.count == tokens * c.valueHeadCount)
        let allRawBetaFinite = input.rawBeta.allSatisfy { $0.isFinite }
        #expect(allRawBetaFinite)
        let expected = cpu(input, c, tokens: tokens)
        let actual = try execute(input, c, tokens: tokens, context: context, preparation: preparation)
        #expect(expected.status == 0)
        #expect(actual.status == expected.status)
        exact(actual.beta, expected.beta, "exponent/mantissa beta")
        exact(actual.decay, expected.decay, "exponent/mantissa decay")
    }

    @Test func positiveLog1pAndPositiveProductDomainMatchPinnedCPU() throws {
        let context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(
            context: context, configuration: try configuration())
        let exp20 = QwenOfficialSourceRouterArithmetic.exponential(20)
        let rawInputs: [Float] = [
            0, Float(bitPattern: 1), Float(bitPattern: 2), Float(bitPattern: 0x007fffff),
            Float(bitPattern: 0x00800000), Float(bitPattern: 0x00800001),
            Float(bitPattern: 0x1fffffff), Float(bitPattern: 0x20000000),
            QwenOfficialSourceRouterArithmetic.exponential(-104),
            QwenOfficialSourceRouterArithmetic.exponential(-100),
            QwenOfficialSourceRouterArithmetic.exponential(-20),
            QwenOfficialSourceRouterArithmetic.exponential(-1),
            1, 2, 20, exp20.nextDown, exp20
        ]
        let multipliers: [Float] = [
            0, Float(bitPattern: 1), Float(bitPattern: 3),
            Float(bitPattern: 0x007fffff), Float(bitPattern: 0x00800001),
            Float(bitPattern: 0x3e800000), 0.5, 1, 2, 16, 256,
            Float(bitPattern: 0x7f7fffff)
        ]
        let values = rawInputs + [Float(bitPattern: 0x01000000), Float(bitPattern: 0x01000001)]
        let factors = values.enumerated().map { multipliers[$0.offset % multipliers.count] }
        let (actualLogs, actualProducts) = try probe(
            values, factors, context: context, preparation: preparation)
        exact(actualLogs, values.map(QwenOfficialSourcePositiveLog1p.evaluate), "log1p domain")
        exact(actualProducts, zip(values, factors).map { $0 * $1 }, "positive products")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment[
        "TURBOFIELDFARE_LINEAR_PREPARATION_DOMAIN_SWEEP"] == "1"))
    func optInDenseStratifiedPositiveDomainMatchesCPU() throws {
        let context = try MetalContext()
        let preparation = try QwenSourceLinearPreparation(
            context: context, configuration: try configuration())
        let count = 16_384
        let limit = QwenOfficialSourceRouterArithmetic.exponential(20)
        let values = (0..<count).map { index in
            limit * (Float(index) / Float(count - 1))
        }
        let multiplierBits: [UInt32] = [
            0, 1, 0x007fffff, 0x00800000, 0x00800001,
            0x3e800000, 0x3f000000, 0x3f800000, 0x40000000,
            0x42000000, 0x7f7fffff
        ]
        let factors = values.indices.map {
            Float(bitPattern: multiplierBits[$0 % multiplierBits.count])
        }
        let (actualLogs, actualProducts) = try probe(
            values, factors, context: context, preparation: preparation)
        exact(actualLogs, values.map(QwenOfficialSourcePositiveLog1p.evaluate), "dense log1p")
        exact(actualProducts, zip(values, factors).map { $0 * $1 }, "dense products")
    }
}
