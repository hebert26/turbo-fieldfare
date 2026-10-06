import Foundation
import Metal
import Testing
@testable import TurboFieldfare

@Suite(.serialized) struct QwenBF16SourceValueScalarFallbackTests {
    @Test func officialSourceAttentionScalarValueFallbackPreservesSignedZeroAndSubnormalFMA() throws {
        let dimension = 5
        let configuration = try QwenFullAttentionConfiguration(
            queryHeadCount: 1,
            keyValueHeadCount: 1,
            headDimension: dimension,
            rotaryDimension: 2,
            theta: 10_000,
            epsilon: 1e-6)
        #expect(!dimension.isMultiple(of: 4))
        #expect((2 * dimension) < 400,
                "this fixture must stay on the scalar score and value fallback")

        // Both scores are exactly zero, so the independent source softmax
        // reference is two exact 0.5 probabilities.  The values intentionally
        // include signed zero, cancellation, and values near Float's subnormal
        // range.  No candidate value-loop helper is used to form the oracle.
        let cachedValues = [
            Float(bitPattern: 0x0040_0000),
            Float(bitPattern: 0x8000_0000),
            1,
            -2,
            Float(bitPattern: 0x0000_0002),
        ]
        let candidateValues = [
            Float(bitPattern: 0x0040_0002),
            Float(bitPattern: 0x0000_0000),
            -1,
            2,
            Float(bitPattern: 0x8000_0001),
        ]
        #expect(cachedValues[0].isSubnormal)
        #expect(candidateValues[4].isFinite)
        #expect(cachedValues[1].bitPattern == 0x8000_0000)

        var expected = [Float](repeating: 0, count: dimension)
        for values in [cachedValues, candidateValues] {
            for column in 0..<dimension {
                expected[column] = expected[column].addingProduct(0.5, values[column])
            }
        }
        #expect(expected[0].isSubnormal,
                "the ordered-FMA oracle must retain a subnormal output")

        let context = try MetalContext()
        let attention = try QwenFullAttention(
            context: context,
            configuration: configuration,
            useOfficialSourceMath: true)
        let cacheBuffer = try #require(context.device.makeBuffer(
            bytes: cachedValues,
            length: cachedValues.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let valueBuffer = try #require(context.device.makeBuffer(
            bytes: cachedValues,
            length: cachedValues.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let cache = QwenFullAttentionKVView(
            key: cacheBuffer,
            value: valueBuffer,
            keyOffset: 0,
            valueOffset: 0,
            strideBytes: dimension * MemoryLayout<Float>.stride,
            validTokenCount: 1)
        let actual = try attention.attentionStep(
            rotatedQuery: [Float](repeating: 0, count: dimension),
            rotatedKey: [Float](repeating: 0, count: dimension),
            value: candidateValues,
            cache: cache)

        #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern),
                "non-multiple-of-four source values must preserve scalar ordered-FMA bits")
    }
}
