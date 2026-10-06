import Foundation

enum AffineBitWidth: Int, Sendable, Equatable {
    case int4 = 4
    case int8 = 8

    var maximumCode: Int { self == .int4 ? 15 : 255 }
}

enum StreamingBF16AffineQuantizerError: Error, Sendable, Equatable {
    case emptyGroup
    case groupTooLarge(count: Int, maximum: Int)
    case oddBF16ByteCount(Int)
    case nonFiniteInput(index: Int)
    case unrepresentableRange
    case unrepresentableAffineParameters
}

struct BF16AffineQuantizedGroup: Sendable, Equatable {
    let elementCount: Int
    let bitWidth: AffineBitWidth
    let packedValues: [UInt8]
    let scaleBF16: UInt16
    let biasBF16: UInt16
    let maximumAbsoluteError: Float
    let declaredMaximumAbsoluteError: Float
}

/// Converts one bounded group at a time. It owns no tensor-sized storage.
enum StreamingBF16AffineQuantizer {
    static let maximumGroupSize = BF16AffineQuantizationPolicy.affineGroupSize
    /// Maximum element-payload storage used by the decoded values, normalized
    /// values, codes, and packed result. Array bookkeeping is constant-sized.
    static let maximumScratchPayloadBytes = maximumGroupSize * (
        2 * MemoryLayout<Float>.stride + 2 * MemoryLayout<UInt8>.stride)

    static func quantize(
        bf16LittleEndian bytes: [UInt8],
        as bitWidth: AffineBitWidth
    ) throws -> BF16AffineQuantizedGroup {
        guard bytes.count.isMultiple(of: 2) else {
            throw StreamingBF16AffineQuantizerError.oddBF16ByteCount(bytes.count)
        }
        var values: [Float] = []
        values.reserveCapacity(bytes.count / 2)
        for index in stride(from: 0, to: bytes.count, by: 2) {
            let bits = UInt16(bytes[index]) | (UInt16(bytes[index + 1]) << 8)
            values.append(decodeBF16(bits))
        }
        return try quantize(values, as: bitWidth)
    }

    static func quantize(
        _ sourceValues: [Float],
        as bitWidth: AffineBitWidth
    ) throws -> BF16AffineQuantizedGroup {
        guard !sourceValues.isEmpty else {
            throw StreamingBF16AffineQuantizerError.emptyGroup
        }
        guard sourceValues.count <= maximumGroupSize else {
            throw StreamingBF16AffineQuantizerError.groupTooLarge(
                count: sourceValues.count, maximum: maximumGroupSize)
        }

        // Canonicalizing both signed zeros prevents input ordering from
        // selecting a different constant-group bias bit pattern.
        var values: [Float] = []
        values.reserveCapacity(sourceValues.count)
        for (index, value) in sourceValues.enumerated() {
            guard value.isFinite else {
                throw StreamingBF16AffineQuantizerError.nonFiniteInput(index: index)
            }
            values.append(value == 0 ? 0 : value)
        }

        var minimum = Float.infinity
        var maximum = -Float.infinity
        for value in values {
            minimum = Swift.min(minimum, value)
            maximum = Swift.max(maximum, value)
        }

        let scaleBits: UInt16
        let biasBits: UInt16
        if minimum == maximum {
            scaleBits = encodeBF16(1)
            biasBits = encodeBF16(minimum)
        } else {
            let range = maximum - minimum
            guard range.isFinite, range > 0 else {
                throw StreamingBF16AffineQuantizerError.unrepresentableRange
            }
            let exactScale = range / Float(bitWidth.maximumCode)
            guard exactScale.isFinite, exactScale > 0 else {
                throw StreamingBF16AffineQuantizerError.unrepresentableRange
            }
            scaleBits = encodeBF16(exactScale)
            biasBits = encodeBF16(minimum)
        }

        let scale = decodeBF16(scaleBits)
        let bias = decodeBF16(biasBits)
        guard scale.isFinite, scale > 0, bias.isFinite else {
            throw StreamingBF16AffineQuantizerError.unrepresentableAffineParameters
        }

        var codes = [UInt8](repeating: 0, count: values.count)
        var maximumError: Float = 0
        for (index, value) in values.enumerated() {
            let coordinate = (value - bias) / scale
            guard coordinate.isFinite else {
                throw StreamingBF16AffineQuantizerError.unrepresentableAffineParameters
            }
            let rounded = coordinate.rounded(.toNearestOrEven)
            // Coordinates should be near the small code range, but hostile
            // arithmetic is checked before Swift's trapping Float-to-Int cast.
            let roundedAsDouble = Double(rounded)
            guard roundedAsDouble >= Double(Int32.min),
                  roundedAsDouble <= Double(Int32.max) else {
                throw StreamingBF16AffineQuantizerError.unrepresentableAffineParameters
            }
            let integer = Int(rounded)
            let code = Swift.min(Swift.max(integer, 0), bitWidth.maximumCode)
            codes[index] = UInt8(code)
            let reconstructed = Float(code) * scale + bias
            guard reconstructed.isFinite else {
                throw StreamingBF16AffineQuantizerError.unrepresentableAffineParameters
            }
            maximumError = Swift.max(maximumError, abs(value - reconstructed))
        }

        let reconstructedMaximum = Float(bitWidth.maximumCode) * scale + bias
        guard reconstructedMaximum.isFinite else {
            throw StreamingBF16AffineQuantizerError.unrepresentableAffineParameters
        }
        let declaredError = Swift.max(
            scale / 2,
            Swift.max(abs(minimum - bias), abs(maximum - reconstructedMaximum)))
        guard declaredError.isFinite else {
            throw StreamingBF16AffineQuantizerError.unrepresentableAffineParameters
        }

        let packed: [UInt8]
        switch bitWidth {
        case .int8:
            packed = codes
        case .int4:
            var nibbles = [UInt8](repeating: 0, count: (codes.count + 1) / 2)
            for (index, code) in codes.enumerated() {
                if index.isMultiple(of: 2) {
                    nibbles[index / 2] = code & 0x0f
                } else {
                    nibbles[index / 2] |= (code & 0x0f) << 4
                }
            }
            packed = nibbles
        }

        return BF16AffineQuantizedGroup(
            elementCount: values.count,
            bitWidth: bitWidth,
            packedValues: packed,
            scaleBF16: scaleBits,
            biasBF16: biasBits,
            maximumAbsoluteError: maximumError,
            declaredMaximumAbsoluteError: declaredError)
    }

    /// IEEE-754 BF16 to Float32 is exact: BF16 occupies the upper 16 bits.
    static func decodeBF16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// Float32 to BF16, round-to-nearest with ties to even.
    static func encodeBF16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let retainedLeastSignificantBit = (bits >> 16) & 1
        return UInt16(truncatingIfNeeded:
            (bits &+ 0x7fff &+ retainedLeastSignificantBit) >> 16)
    }
}
