#include <metal_stdlib>
using namespace metal;
// Standalone opt-in discriminator. Compile after the unchanged pinned source
// exp and scalar reciprocal modules, in safe/precise mode. No state writes.
struct QwenLinearPreparationParameters {
    uint tokenCount, keyHeads, valueHeads, keyDimension, valueDimension, channels;
};
struct QwenPreparationPair { float high; float low; };
inline QwenPreparationPair qwenPreparationMultiply(QwenPreparationPair x, float y) {
#pragma clang fp contract(off)
    const float high = x.high * y;
    const float error = fma(x.high, y, -high);
    return {high, fma(x.low, y, error)};
}
inline QwenPreparationPair qwenPreparationAdd(float x, float y) {
#pragma clang fp contract(off)
    const float high = x + y;
    return {high, (x - high) + y};
}
inline QwenPreparationPair qwenPreparationAdd(QwenPreparationPair x, QwenPreparationPair y) {
#pragma clang fp contract(off)
    const float high = x.high + y.high;
    return {high, (((x.high - high) + y.high) + x.low) + y.low};
}
inline QwenPreparationPair qwenPreparationAdd(QwenPreparationPair x, float y) {
#pragma clang fp contract(off)
    const float high = x.high + y;
    return {high, ((x.high - high) + y) + x.low};
}
inline QwenPreparationPair qwenPreparationDivide(QwenPreparationPair n, QwenPreparationPair d) {
#pragma clang fp contract(off)
    const float reciprocal = 1.0f / d.high;
    const float high = n.high * reciprocal;
    const float error = fma(reciprocal, n.high, -high);
    const float firstCorrection = fma(-d.high, reciprocal, 1.0f);
    const float correction = fma(-d.low, reciprocal, firstCorrection);
    const float numeratorCorrection = fma(n.low, reciprocal, error);
    return {high, fma(high, correction, numeratorCorrection)};
}
// For positive values below2^-125 the source pair sequence has exponent0,
// exact mantissa=value and reciprocal1/2. Its correction products underflow
// even on CPU, so the final value is exactly2*roundToEven(value/2). Integer
// half and double retain the CPU's gradual underflow, including odd ULPs.
inline float qwenPreparationPositiveLog1p(float value) {
#pragma clang fp contract(off)
    const uint bits = as_type<uint>(value);
    if (bits < 0x01000000u) {
        uint halfBits = bits >> 1;
        if ((bits & 1u) && (halfBits & 1u)) ++halfBits;
        return as_type<float>(halfBits << 1);
    }
    // Here half is normal and all correction terms are far below the result
    // ULP. The pinned CPU returns value; avoid irrelevant flushed corrections.
    if (bits < 0x20000000u) return value;
    const float plusOne = value + 1.0f;
    const float scaled = plusOne * (1.0f / 0.75f);
    const int exponent = int((as_type<uint>(scaled) >> 23) & 255u) - 127;
    const float inversePower = as_type<float>(0x3f800000u - (uint(exponent) << 23));
    const float mantissa = fma(value, inversePower, inversePower - 1.0f);
    QwenPreparationPair sum = qwenPreparationMultiply(
        {0.69314718246459960938f, -1.904654323148236017e-09f}, float(exponent));
    const QwenPreparationPair quotient = qwenPreparationDivide(
        {mantissa, 0.0f}, qwenPreparationAdd(2.0f, mantissa));
    const float square = quotient.high * quotient.high;
    float polynomial = 0.3027294874f;
    polynomial = fma(polynomial, square, 0.3996108174f);
    polynomial = fma(polynomial, square, 0.6666694880f);
    sum = qwenPreparationAdd(sum, {quotient.high * 2.0f, quotient.low * 2.0f});
    sum = qwenPreparationAdd(sum, (square * quotient.high) * polynomial);
    return sum.high + sum.low;
}
inline ulong qwenPreparationRound(ulong product, int shift) {
    if (shift <= 0) return product << uint(-shift);
    if (shift >= 64) return 0ul; // Product has at most48 significant bits.
    ulong rounded = product >> uint(shift);
    const ulong remainder = product & ((1ul << uint(shift)) - 1ul);
    const ulong midpoint = 1ul << uint(shift - 1);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1ul))) ++rounded;
    return rounded;
}
// Only positive operands occur in exp(aLog)*softplus. One exact24x24-bit
// product plus nearest-even rounding preserves all CPU FP32 normal/subnormal
// products. This local helper neither accepts arbitrary signs nor alters FMA.
inline float qwenPreparationPositiveProduct(float a, float b) {
    const uint ab = as_type<uint>(a), bb = as_type<uint>(b);
    if ((ab & 0x7fffffffu) == 0u || (bb & 0x7fffffffu) == 0u) {
        return (isinf(a) || isinf(b)) ? NAN : 0.0f;
    }
    if (isinf(a) || isinf(b)) return INFINITY;
    const uint ae = (ab >> 23) & 255u, be = (bb >> 23) & 255u;
    const ulong am = (ab & 0x007fffffu) | (ae ? 0x00800000u : 0u);
    const ulong bm = (bb & 0x007fffffu) | (be ? 0x00800000u : 0u);
    const int power = (ae ? int(ae) - 150 : -149) + (be ? int(be) - 150 : -149);
    const ulong product = am * bm;
    const int leading = 63 - int(clz(product));
    int exponent = power + leading;
    if (exponent < -126) return as_type<float>(uint(qwenPreparationRound(product, -power - 149)));
    ulong significand = qwenPreparationRound(product, leading - 23);
    if (significand >= 0x01000000ul) { significand >>= 1; ++exponent; }
    if (exponent > 127) return INFINITY;
    return as_type<float>((uint(exponent + 127) << 23) | (uint(significand) & 0x007fffffu));
}
kernel void qwen_source_linear_prepare_fp32(
    constant QwenLinearPreparationParameters& p [[buffer(0)]],
    device const uint* convolved [[buffer(1)]],
    device const float* rawBeta [[buffer(2)]],
    device const float* rawA [[buffer(3)]],
    device const float* aLog [[buffer(4)]],
    device const float* timeStepBias [[buffer(5)]],
    device uint* query [[buffer(6)]], device uint* key [[buffer(7)]],
    device uint* value [[buffer(8)]], device float* beta [[buffer(9)]],
    device uint* logDecay [[buffer(10)]], device atomic_uint* status [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint keyWidth = p.keyHeads * p.keyDimension;
    const uint queryWidth = p.valueHeads * p.keyDimension;
    const uint valueWidth = p.valueHeads * p.valueDimension;
    if (index < p.tokenCount * p.channels && (convolved[index] & 0x7f800000u) == 0x7f800000u) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
    if (index < p.tokenCount * queryWidth) {
        const uint token = index / queryWidth, local = index % queryWidth;
        const uint head = local / p.keyDimension, dimension = local % p.keyDimension;
        const uint sourceHead = head / (p.valueHeads / p.keyHeads);
        const uint source = token * p.channels + sourceHead * p.keyDimension + dimension;
        query[index] = convolved[source]; key[index] = convolved[source + keyWidth];
    }
    if (index < p.tokenCount * valueWidth) {
        value[index] = convolved[(index / valueWidth) * p.channels + 2u * keyWidth + index % valueWidth];
    }
    if (index < p.tokenCount * p.valueHeads) {
        const uint head = index % p.valueHeads;
        if (!isfinite(rawBeta[index]) || !isfinite(rawA[index])
            || !isfinite(aLog[head]) || !isfinite(timeStepBias[head])) {
            // Every scalar output must be initialized before speculative
            // recurrence, even though flagged state can never be published.
            beta[index] = 0.0f; logDecay[index] = 0u;
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed); return;
        }
        const float b = qwenSourceScalarPositiveReciprocal(1.0f + qwenSourceExpFloatBitScale(-rawBeta[index]));
        const float x = rawA[index] + timeStepBias[head];
        const float softplus = x > 20.0f ? x : qwenPreparationPositiveLog1p(qwenSourceExpFloatBitScale(x));
        const float factor = qwenSourceExpFloatBitScale(aLog[head]);
        const float positiveDecay = qwenPreparationPositiveProduct(factor, softplus);
        const uint decayBits = as_type<uint>(positiveDecay) ^ 0x80000000u;
        beta[index] = b; logDecay[index] = decayBits;
        if (!isfinite(b) || (decayBits & 0x7f800000u) == 0x7f800000u) {
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
    }
}

// Test-owned math discriminator; no runtime path calls this kernel. Values
// must be nonnegative finite <=1e38, multipliers nonnegative finite.
kernel void qwen_source_linear_preparation_math_probe(
    constant uint& count [[buffer(0)]], device const float* values [[buffer(1)]],
    device const float* multipliers [[buffer(2)]], device float* logs [[buffer(3)]],
    device float* products [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    logs[index] = qwenPreparationPositiveLog1p(values[index]);
    products[index] = qwenPreparationPositiveProduct(values[index], multipliers[index]);
}

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
