import Testing
@testable import TurboFieldfareRepackCore

@Suite struct StreamingBF16AffineQuantizerTests {
    @Test func handComputedInt4GoldenUsesLowThenHighNibbles() throws {
        // BF16 little-endian values: 0, 1, 2, 3. The stored scale is BF16
        // 0x3e4d (0.2001953125), bias is +0, and ties-to-even codes are 0,5,10,15.
        let group = try StreamingBF16AffineQuantizer.quantize(
            bf16LittleEndian: [0x00, 0x00, 0x80, 0x3f, 0x00, 0x40, 0x40, 0x40],
            as: .int4)
        #expect(group.elementCount == 4)
        #expect(group.scaleBF16 == 0x3e4d)
        #expect(group.biasBF16 == 0x0000)
        #expect(group.packedValues == [0x50, 0xfa])
        #expect(group.maximumAbsoluteError <= group.declaredMaximumAbsoluteError)
    }

    @Test func handComputedInt8GoldenAndConstantOddTailAreStable() throws {
        // 3 / 255 rounded to BF16 is 0x3c41. Codes are 0,85,170,255.
        let int8 = try StreamingBF16AffineQuantizer.quantize(
            bf16LittleEndian: [0x00, 0x00, 0x80, 0x3f, 0x00, 0x40, 0x40, 0x40],
            as: .int8)
        #expect(int8.scaleBF16 == 0x3c41)
        #expect(int8.biasBF16 == 0x0000)
        #expect(int8.packedValues == [0, 85, 170, 255])

        let constant = try StreamingBF16AffineQuantizer.quantize(
            bf16LittleEndian: [0x00, 0xc0, 0x00, 0xc0, 0x00, 0xc0], as: .int4)
        #expect(constant.scaleBF16 == 0x3f80)
        #expect(constant.biasBF16 == 0xc000)
        #expect(constant.packedValues == [0x00, 0x00])
    }

    @Test func BF16RoundingUsesTiesToEvenAndZerosCanonicalize() throws {
        // Exactly halfway between adjacent BF16 values at 1.0: low retained LSB
        // even chooses 0x3f80; the next halfway case chooses 0x3f82.
        #expect(StreamingBF16AffineQuantizer.encodeBF16(1.00390625) == 0x3f80)
        #expect(StreamingBF16AffineQuantizer.encodeBF16(1.01171875) == 0x3f82)

        let negativeZero = try StreamingBF16AffineQuantizer.quantize([-0.0], as: .int8)
        #expect(negativeZero.biasBF16 == 0x0000)
        #expect(negativeZero.packedValues == [0])
    }

    @Test func subnormalAndRepeatedGroupsHaveSpecifiedFailureAndErrorBounds() throws {
        // Smallest positive BF16 subnormal and zero form a nonconstant range
        // whose scale rounds to zero in BF16, which is explicitly rejected.
        #expect(throws: StreamingBF16AffineQuantizerError.self) {
            try StreamingBF16AffineQuantizer.quantize(
                bf16LittleEndian: [0x00, 0x00, 0x01, 0x00], as: .int8)
        }

        let input: [Float] = [0, 1, 2, 3]
        let first = try StreamingBF16AffineQuantizer.quantize(input, as: .int4)
        let second = try StreamingBF16AffineQuantizer.quantize(input, as: .int4)
        #expect(first == second)
        // max(0.2001953125 / 2, 0, abs(3 - 15 * 0.2001953125)).
        #expect(first.declaredMaximumAbsoluteError == 0.10009765625)
        #expect(first.maximumAbsoluteError <= 0.10009765625)
    }

    @Test func malformedAndHostileGroupsFailRatherThanProducingBytes() {
        #expect(throws: StreamingBF16AffineQuantizerError.self) {
            try StreamingBF16AffineQuantizer.quantize(bf16LittleEndian: [0], as: .int4)
        }
        #expect(throws: StreamingBF16AffineQuantizerError.self) {
            try StreamingBF16AffineQuantizer.quantize([], as: .int8)
        }
        #expect(throws: StreamingBF16AffineQuantizerError.self) {
            try StreamingBF16AffineQuantizer.quantize([Float.nan], as: .int8)
        }
        #expect(throws: StreamingBF16AffineQuantizerError.self) {
            try StreamingBF16AffineQuantizer.quantize([Float.infinity], as: .int4)
        }
        #expect(throws: StreamingBF16AffineQuantizerError.self) {
            try StreamingBF16AffineQuantizer.quantize(Array(repeating: 0, count: 65), as: .int8)
        }
    }
}
