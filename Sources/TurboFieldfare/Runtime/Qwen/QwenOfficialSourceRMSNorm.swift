import Foundation

/// Input, post-attention, and final norms use zero-centered source weights.
/// The separate gated linear norm retains its plain weight and uses another path.
func qwenOfficialSourceRMSNorm(_ input: [Float], _ storedWeights: [Float]) throws -> [Float] {
    guard input.count == storedWeights.count, !input.isEmpty,
          input.allSatisfy(\.isFinite), storedWeights.allSatisfy(\.isFinite) else {
        throw QwenTextRunnerError.execution(detail: "source RMSNorm input")
    }
    // Preserve the pinned CPU source's FP32 reduction order, including the
    // rounded squares. A wider sum can round the final mean differently.
    let squareSum = qwenSourceSquareSum(input)
    let meanSquare = squareSum / Float(input.count)
    let adjustedMean = meanSquare + Float(1e-6)
    let reciprocal = Float(1) / sqrt(adjustedMean)
    guard reciprocal.isFinite, reciprocal > 0 else {
        throw QwenTextRunnerError.execution(detail: "source RMSNorm denominator")
    }
    let normalized = zip(input, storedWeights).map { value, storedWeight in
        (value * reciprocal) * (1 + storedWeight)
    }
    guard normalized.allSatisfy(\.isFinite) else {
        throw QwenTextRunnerError.execution(detail: "nonfinite source RMSNorm output")
    }
    return normalized
}

/// The contiguous FP32 sum used by Torch 2.10 on ARM64: four interleaved
/// four-lane vectors, accumulated through four cascade levels, then the
/// remaining vectors and scalar tail. This follows SumKernel.cpp's
/// multi_row_sum, row_sum, and vectorized_inner_sum operation order.
private func qwenSourceSquareSum(_ input: [Float]) -> Float {
    let vectorCount = input.count / 4
    guard vectorCount > 0 else {
        return input.reduce(Float(0)) { $0 + $1 * $1 }
    }
    func squareVector(_ index: Int) -> SIMD4<Float> {
        let offset = index * 4
        let values = SIMD4(input[offset], input[offset + 1],
                           input[offset + 2], input[offset + 3])
        return values * values
    }
    let interleavedCount = vectorCount / 4
    let ceilLog2 = interleavedCount > 1
        ? Int.bitWidth - (interleavedCount - 1).leadingZeroBitCount : 0
    let levelPower = max(4, ceilLog2 / 4)
    let levelStep = 1 << levelPower
    let levelMask = levelStep - 1
    let zero = SIMD4<Float>(repeating: 0)
    var accumulators = Array(repeating: Array(repeating: zero, count: 4), count: 4)
    var index = 0
    while interleavedCount - index >= levelStep {
        for _ in 0..<levelStep {
            for stream in 0..<4 {
                accumulators[0][stream] += squareVector(index * 4 + stream)
            }
            index += 1
        }
        for level in 1..<4 {
            for stream in 0..<4 {
                accumulators[level][stream] += accumulators[level - 1][stream]
                accumulators[level - 1][stream] = zero
            }
            if (index & (levelMask << (level * levelPower))) != 0 { break }
        }
    }
    while index < interleavedCount {
        for stream in 0..<4 {
            accumulators[0][stream] += squareVector(index * 4 + stream)
        }
        index += 1
    }
    for level in 1..<4 {
        for stream in 0..<4 {
            accumulators[0][stream] += accumulators[level][stream]
        }
    }
    var partials = accumulators[0]
    for vector in (interleavedCount * 4)..<vectorCount {
        partials[0] += squareVector(vector)
    }
    for stream in 1..<4 { partials[0] += partials[stream] }
    var sum: Float = 0
    for scalar in (vectorCount * 4)..<input.count {
        let value = input[scalar]
        sum += value * value
    }
    for lane in 0..<4 { sum += partials[0][lane] }
    return sum
}
