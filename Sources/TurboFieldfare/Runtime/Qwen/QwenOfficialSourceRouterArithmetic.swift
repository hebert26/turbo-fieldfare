import TurboFieldfareSourceTopK

/// Pinned Torch 2.10 ARM64 CPU arithmetic for the original-BF16 source router.
/// No Torch/SLEEF library dependency or model weights are used by this helper.
enum QwenOfficialSourceRouterArithmetic {
    /// Original-source Top-8 uses Torch's value-only native selection on the
    /// rounded probabilities, including its cutoff and interior tie ordering.
    static func top8Indices(_ probabilities: [Float]) throws -> [Int] {
        guard probabilities.count == 256 else {
            throw QwenMoEError.invalidCount(
                field: "source router probabilities", expected: 256, actual: probabilities.count)
        }
        if let expert = probabilities.firstIndex(where: { !$0.isFinite }) {
            throw QwenMoEError.nonfiniteRouterValue(token: 0, expert: expert)
        }
        var indices = [Int32](repeating: 0, count: 8)
        let selected = probabilities.withUnsafeBufferPointer { values in
            indices.withUnsafeMutableBufferPointer { output in
                tf_qwen_source_top8(values.baseAddress!, Int32(values.count), output.baseAddress!)
            }
        }
        guard selected else { throw QwenMoEError.sourceTopKSelectionFailed }
        return indices.map(Int.init)
    }

    /// SLEEF 3.8.0 xexpf, AArch64 CONFIG=1 fused arithmetic, commit
    /// 5a1d179df9cf652951b59010a2d2075372d67f68. Adapted from
    /// src/libm/sleefsimdsp.c, src/arch/helperadvsimd.h and src/common/misc.h.
    static func exponential(_ value: Float) -> Float {
        // Source masks are last. Early return avoids an out-of-range Swift
        // integer conversion and preserves the masked result.
        if value < -104 { return 0 }
        if value > 100 { return .infinity }
        if value.isNaN { return value + value }
        let exponent = Int32((value * Float(1.442695040888963407359924681001892137426645954152985934135449406931)).rounded(.toNearestOrEven))
        let quotient = Float(exponent)
        var reduced = value.addingProduct(quotient, Float(-0.693145751953125))
        reduced = reduced.addingProduct(quotient, Float(-1.428606765330187045e-06))
        var polynomial = Float(0.000198527617612853646278381)
        polynomial = Float(0.00139304355252534151077271).addingProduct(polynomial, reduced)
        polynomial = Float(0.00833336077630519866943359).addingProduct(polynomial, reduced)
        polynomial = Float(0.0416664853692054748535156).addingProduct(polynomial, reduced)
        polynomial = Float(0.166666671633720397949219).addingProduct(polynomial, reduced)
        polynomial = Float(0.5).addingProduct(polynomial, reduced)
        polynomial = 1 + reduced.addingProduct(reduced * reduced, polynomial)
        let half = exponent >> 1
        let firstScale = Float(bitPattern: UInt32(bitPattern: (half + 127) << 23))
        let secondScale = Float(bitPattern: UInt32(bitPattern: (exponent - half + 127) << 23))
        return (polynomial * firstScale) * secondScale
    }

    /// ATen functional_base.h reduce_all and ARM64 VecReduceAllSIMD.
    static func softmaxSum(_ values: [Float]) -> Float {
        precondition(!values.isEmpty)
        if values.count < 4 {
            return values.dropFirst().reduce(values[0], +)
        }
        var lanes = SIMD4<Float>(values[0], values[1], values[2], values[3])
        var index = 4
        while index + 4 <= values.count {
            lanes += SIMD4<Float>(values[index], values[index + 1],
                                  values[index + 2], values[index + 3])
            index += 4
        }
        for tail in index..<values.count { lanes[tail - index] += values[tail] }
        return (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
    }

    /// ATen SumKernel contiguous FP32 reduction at the admitted Top-8 geometry.
    static func top8Sum(_ values: [Float]) -> Float {
        precondition(values.count == 8)
        let lane0 = values[0] + values[4]
        let lane1 = values[1] + values[5]
        let lane2 = values[2] + values[6]
        let lane3 = values[3] + values[7]
        return ((lane0 + lane1) + lane2) + lane3
    }
}

// Copyright Naoki Shibata and contributors 2010 - 2024.
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
//
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
