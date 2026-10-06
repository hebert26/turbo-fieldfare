#include <metal_stdlib>
using namespace metal;

// qwen_common.metal is composed before this module by MetalContext.
// The original-source path compiles this module alone in a private library.
#ifndef QWEN_METAL_COMMON
typedef uint QwenMetalDimension;
enum QwenSourceFullAttentionBufferIndex : uint {
    QwenMetalBufferIndexParameters = 0,
    QwenMetalBufferIndexInput = 1,
    QwenMetalBufferIndexWeights = 2,
    QwenMetalBufferIndexScales = 3,
    QwenMetalBufferIndexBiases = 4,
    QwenMetalBufferIndexOutput = 5,
    QwenMetalBufferIndexScratch = 6,
};
#endif

struct QwenNormRoPEParameters {
    QwenMetalDimension tokenCount;
    QwenMetalDimension headCount;
    QwenMetalDimension headDimension;
    QwenMetalDimension rotaryDimension;
    QwenMetalDimension startPosition;
    float theta;
    float epsilon;
    QwenMetalDimension reserved;
};

struct QwenMRoPEParameters {
    QwenMetalDimension tokenCount;
    QwenMetalDimension headCount;
    QwenMetalDimension headDimension;
    QwenMetalDimension rotaryDimension;
    QwenMetalDimension temporalSection;
    QwenMetalDimension heightSection;
    QwenMetalDimension widthSection;
    QwenMetalDimension reserved;
    float theta;
    float epsilon;
};

struct QwenGateParameters {
    QwenMetalDimension elementCount;
    QwenMetalDimension reserved0;
    QwenMetalDimension reserved1;
    QwenMetalDimension reserved2;
};

#ifdef QWEN_PINNED_SOURCE_EXP
// Pinned source SIMD trigonometry; qualified over the official context range.
#ifndef QWEN_SOURCE_TRIG_SMALL
#define QWEN_SOURCE_TRIG_SMALL
#include <metal_stdlib>
using namespace metal;

// Derived from SLEEF 3.8.0 xsinf_u1/xcosf_u1 and df.h, ARM64 CONFIG=1.
// Commit 5a1d179df9cf652951b59010a2d2075372d67f68.
// This helper implements ONLY the |angle| < 125 small-angle branch.
// Larger-angle SLEEF rempif reduction is intentionally not claimed here.
// Copyright Naoki Shibata and contributors 2010 - 2024.
// Distributed under Boost Software License 1.0, reproduced below.
inline float2 qwenTrigAddFF(float a, float b) {
#pragma clang fp contract(off)
    float s = a + b;
    return float2(s, (a - s) + b);
}
inline float2 qwenTrigAdd2FF(float a, float b) {
#pragma clang fp contract(off)
    float s = a + b;
    float v = s - a;
    return float2(s, (a - (s - v)) + (b - v));
}
inline float2 qwenTrigAddDF(float2 a, float b) {
#pragma clang fp contract(off)
    float s = a.x + b;
    return float2(s, ((a.x - s) + b) + a.y);
}
inline float2 qwenTrigAdd2DF(float2 a, float b) {
#pragma clang fp contract(off)
    float s = a.x + b;
    float v = s - a.x;
    float t = (a.x - (s - v)) + (b - v);
    return float2(s, t + a.y);
}
inline float2 qwenTrigAddFD(float a, float2 b) {
#pragma clang fp contract(off)
    float s = a + b.x;
    return float2(s, ((a - s) + b.x) + b.y);
}
inline float2 qwenTrigSquare(float2 a) {
#pragma clang fp contract(off)
    float s = a.x * a.x;
    return float2(s, fma(a.x + a.x, a.y, fma(a.x, a.x, -s)));
}
inline float2 qwenTrigMultiply(float2 a, float2 b) {
#pragma clang fp contract(off)
    float s = a.x * b.x;
    return float2(s, fma(a.x, b.y, fma(a.y, b.x, fma(a.x, b.x, -s))));
}
inline float qwenTrigMultiplyRounded(float2 a, float2 b) {
#pragma clang fp contract(off)
    return fma(a.x, b.x, fma(a.y, b.x, a.x * b.y));
}
inline float qwenTrigEvaluate(float2 reduced) {
#pragma clang fp contract(off)
    float2 square = qwenTrigSquare(reduced);
    float polynomial = 2.6083159809786593541503e-06f;
    polynomial = fma(polynomial, square.x, -0.0001981069071916863322258f);
    polynomial = fma(polynomial, square.x, 0.00833307858556509017944336f);
    float2 factor = qwenTrigAddFD(1.0f, qwenTrigMultiply(
        qwenTrigAddFF(-0.166666597127914428710938f, polynomial * square.x), square));
    return qwenTrigMultiplyRounded(reduced, factor);
}
inline float qwenSourceSinSmall(float angle) {
#pragma clang fp contract(off)
    if (!(abs(angle) < 125.0f)) return NAN;
    float quotient = rint(angle * 0.318309886183790671537767526745f);
    int q = int(quotient);
    float v = fma(quotient, -3.1414794921875f, angle);
    float2 reduced = qwenTrigAdd2FF(v, quotient * -0.00011315941810607910156f);
    reduced = qwenTrigAddDF(reduced, quotient * -1.9841872589410058936e-09f);
    float result = qwenTrigEvaluate(reduced);
    result = as_type<float>(as_type<uint>(result) ^ ((uint(q) & 1u) << 31));
    if (as_type<uint>(angle) == 0x80000000u) return angle;
    // The source polynomial is exactly angle for subnormal inputs. Preserve
    // those input bits when GPU arithmetic would flush them to zero.
    if ((as_type<uint>(angle) & 0x7fffffffu) < 0x00800000u) return angle;
    return result;
}
inline float qwenSourceCosSmall(float angle) {
#pragma clang fp contract(off)
    if (!(abs(angle) < 125.0f)) return NAN;
    float dq = fma(rint(fma(angle, 0.318309886183790671537767526745f, -0.5f)), 2.0f, 1.0f);
    int q = int(dq);
    float2 reduced = qwenTrigAdd2FF(angle, dq * (-3.1414794921875f * 0.5f));
    reduced = qwenTrigAdd2DF(reduced, dq * (-0.00011315941810607910156f * 0.5f));
    reduced = qwenTrigAdd2DF(reduced, dq * (-1.9841872589410058936e-09f * 0.5f));
    float result = qwenTrigEvaluate(reduced);
    return as_type<float>(as_type<uint>(result) ^ (((uint(q) & 2u) == 0u) ? 0x80000000u : 0u));
}

// Boost Software License - Version 1.0 - August 17th, 2003
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
#endif

// Scratch extension from the same pinned SLEEF 3.8.0 source and Boost license.
// Requires qwen_source_trig_small.metal, which contains the full license.
// This implements the exact rempif table row used for |angle| < 2^25.
// wholeVectorSmall must describe ALL FOUR source SIMD lanes, not just this one.
inline float2 qwenTrigNormalize(float2 a) {
#pragma clang fp contract(off)
    float s = a.x + a.y;
    return float2(s, (a.x - s) + a.y);
}
inline float2 qwenTrigProductFF(float a, float b) {
#pragma clang fp contract(off)
    float s = a * b;
    return float2(s, fma(a, b, -s));
}
inline float2 qwenTrigProductDF(float2 a, float b) {
#pragma clang fp contract(off)
    float s = a.x * b;
    return float2(s, fma(a.y, b, fma(a.x, b, -s)));
}
inline float2 qwenTrigAdd2DD(float2 a, float2 b) {
#pragma clang fp contract(off)
    float s = a.x + b.x;
    float v = s - a.x;
    float t = (a.x - (s - v)) + (b.x - v);
    return float2(s, t + (a.y + b.y));
}
inline float qwenTrigQuarterRemainder(float a, thread int& quarter) {
#pragma clang fp contract(off)
    float rounded = rint(a * 4.0f);
    quarter = int(rounded - rint(a) * 4.0f);
    return a - rounded * 0.25f;
}
inline float2 qwenTrigRempiContext(float angle, thread int& q) {
#pragma clang fp contract(off)
    q = 0;
    if (abs(angle) < 0.7f) return float2(angle, 0.0f);
    // SLEEF_rempitabsp row zero, not generated from any recorded inputs.
    float2 x = qwenTrigProductFF(angle, 0.159154892f);
    int quarter;
    x.x = qwenTrigQuarterRemainder(x.x, quarter);
    q = quarter;
    x = qwenTrigNormalize(x);
    float2 y = qwenTrigProductFF(angle, 5.112411827e-08f);
    x = qwenTrigAdd2DD(x, y);
    x.x = qwenTrigQuarterRemainder(x.x, quarter);
    q += quarter;
    x = qwenTrigNormalize(x);
    y = qwenTrigProductDF(float2(3.626141271e-15f, -2.036222915e-22f), angle);
    x = qwenTrigNormalize(qwenTrigAdd2DD(x, y));
    return qwenTrigMultiply(x, float2(3.1415927410125732422f * 2.0f, -8.7422776573475857731e-08f * 2.0f));
}
inline float2 qwenTrigSignedHalfPi(float signSource) {
    uint sign = as_type<uint>(signSource) & 0x80000000u;
    return float2(as_type<float>(as_type<uint>(3.1415927410125732422f * -0.5f) ^ sign),
                  as_type<float>(as_type<uint>(-8.7422776573475857731e-08f * -0.5f) ^ sign));
}
inline float qwenSourceSinContext(float angle, bool wholeVectorSmall) {
#pragma clang fp contract(off)
    if (!(abs(angle) < 0x1p25f)) return NAN;
    if (wholeVectorSmall) return qwenSourceSinSmall(angle);
    int originalQ;
    float2 reduced = qwenTrigRempiContext(angle, originalQ);
    int q = (((originalQ & 3) * 2) + (reduced.x > 0.0f ? 2 : 1)) >> 2;
    if ((originalQ & 1) == 1) reduced = qwenTrigAdd2DD(reduced, qwenTrigSignedHalfPi(reduced.x));
    float result = qwenTrigEvaluate(qwenTrigNormalize(reduced));
    result = as_type<float>(as_type<uint>(result) ^ ((uint(q) & 1u) << 31));
    if ((as_type<uint>(angle) & 0x7fffffffu) < 0x00800000u) return angle;
    return result;
}
inline float qwenSourceCosContext(float angle, bool wholeVectorSmall) {
#pragma clang fp contract(off)
    if (!(abs(angle) < 0x1p25f)) return NAN;
    if (wholeVectorSmall) return qwenSourceCosSmall(angle);
    int originalQ;
    float2 reduced = qwenTrigRempiContext(angle, originalQ);
    int q = (((originalQ & 3) * 2) + (reduced.x > 0.0f ? 8 : 7)) >> 1;
    if ((originalQ & 1) == 0) {
        float signSource = reduced.x > 0.0f ? 0.0f : -1.0f;
        reduced = qwenTrigAdd2DD(reduced, qwenTrigSignedHalfPi(signSource));
    }
    float result = qwenTrigEvaluate(qwenTrigNormalize(reduced));
    return as_type<float>(as_type<uint>(result) ^ (((uint(q) & 2u) == 0u) ? 0x80000000u : 0u));
}

// Same generic pinned four-stream/four-lane cascade as the source linear
// norm. This head256 path was independently checked on 18,432 GPU outputs.
static inline float qwenSourceFullAttentionSquareSum(
    const device float* input, uint base, uint count
) {
#pragma clang fp contract(off)
    const uint vectorCount = count / 4u;
    const uint interleavedCount = vectorCount / 4u;
    const uint ceilLog2 = interleavedCount > 1u ? 32u - clz(interleavedCount - 1u) : 0u;
    const uint levelPower = max(4u, ceilLog2 / 4u);
    const uint levelStep = 1u << levelPower;
    const uint levelMask = levelStep - 1u;
    float4 partials[4][4];
    for (uint level = 0; level < 4; ++level) {
        for (uint stream = 0; stream < 4; ++stream) partials[level][stream] = float4(0.0f);
    }
    uint index = 0;
    while (interleavedCount - index >= levelStep) {
        for (uint item = 0; item < levelStep; ++item) {
            for (uint stream = 0; stream < 4; ++stream) {
                const uint offset = base + (index * 4u + stream) * 4u;
                const float4 values(input[offset], input[offset + 1], input[offset + 2], input[offset + 3]);
                const float4 squares = values * values;
                partials[0][stream] += squares;
            }
            ++index;
        }
        for (uint level = 1; level < 4; ++level) {
            for (uint stream = 0; stream < 4; ++stream) {
                partials[level][stream] += partials[level - 1][stream];
                partials[level - 1][stream] = float4(0.0f);
            }
            if ((index & (levelMask << (level * levelPower))) != 0u) break;
        }
    }
    while (index < interleavedCount) {
        for (uint stream = 0; stream < 4; ++stream) {
            const uint offset = base + (index * 4u + stream) * 4u;
            const float4 values(input[offset], input[offset + 1], input[offset + 2], input[offset + 3]);
            const float4 squares = values * values;
            partials[0][stream] += squares;
        }
        ++index;
    }
    for (uint level = 1; level < 4; ++level) {
        for (uint stream = 0; stream < 4; ++stream) partials[0][stream] += partials[level][stream];
    }
    for (uint vector = interleavedCount * 4u; vector < vectorCount; ++vector) {
        const uint offset = base + vector * 4u;
        const float4 values(input[offset], input[offset + 1], input[offset + 2], input[offset + 3]);
        const float4 squares = values * values;
        partials[0][0] += squares;
    }
    for (uint stream = 1; stream < 4; ++stream) partials[0][0] += partials[0][stream];
    float sum = 0.0f;
    for (uint scalar = vectorCount * 4u; scalar < count; ++scalar) {
        const float value = input[base + scalar];
        const float square = value * value;
        sum += square;
    }
    for (uint lane = 0; lane < 4; ++lane) sum += partials[0][0][lane];
    return sum;
}
#endif

/// Qwen 3.5/3.6 per-head RMS normalization followed by partial rotate_half RoPE.
/// The stored norm weights are residual weights: the multiplier is (1 + weight).
kernel void qwen_qk_norm_partial_rope(
    constant QwenNormRoPEParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weight [[buffer(QwenMetalBufferIndexWeights)]],
#ifdef QWEN_PINNED_SOURCE_EXP
    device const float* inverseFrequencies [[buffer(QwenMetalBufferIndexScales)]],
#endif
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    uint item [[thread_position_in_grid]]
) {
#ifdef QWEN_PINNED_SOURCE_EXP
#pragma clang fp contract(off)
#endif
    const uint itemCount = parameters.tokenCount * parameters.headCount;
    if (item >= itemCount) {
        return;
    }

    const uint dimension = parameters.headDimension;
    const uint base = item * dimension;
#ifdef QWEN_PINNED_SOURCE_EXP
    const float squareSum = qwenSourceFullAttentionSquareSum(input, base, dimension);
    const float inverseRMS = 1.0f / sqrt(squareSum / float(dimension) + parameters.epsilon);
#else
    float squareSum = 0.0f;
    for (uint index = 0; index < dimension; ++index) {
        const float value = input[base + index];
        squareSum += value * value;
    }
    const float inverseRMS = rsqrt(squareSum / float(dimension) + parameters.epsilon);
#endif

    const uint rotaryDimension = parameters.rotaryDimension;
    const uint rotaryHalf = rotaryDimension / 2;
    const uint token = item / parameters.headCount;
    const float position = float(parameters.startPosition + token);

    for (uint index = 0; index < dimension; ++index) {
        const float normalized = input[base + index] * inverseRMS * (1.0f + weight[index]);
        if (index >= rotaryDimension) {
            output[base + index] = normalized;
            continue;
        }

        const bool firstHalf = index < rotaryHalf;
        const uint frequencyIndex = firstHalf ? index : index - rotaryHalf;
        const uint partnerIndex = firstHalf ? index + rotaryHalf : index - rotaryHalf;
        const float partner = input[base + partnerIndex] * inverseRMS * (1.0f + weight[partnerIndex]);
#ifdef QWEN_PINNED_SOURCE_EXP
        const float inverseFrequency = inverseFrequencies[frequencyIndex];
        const float angle = position * inverseFrequency;
        // Official SIMD trig runs before spatial-axis recomposition. Each
        // quartet therefore uses this lane's original axis position throughout.
        const uint vectorStart = frequencyIndex & ~3u;
        bool wholeVectorSmall = true;
        for (uint lane = vectorStart; lane < min(vectorStart + 4u, rotaryHalf); ++lane) {
            wholeVectorSmall = wholeVectorSmall
                && abs(position * inverseFrequencies[lane]) < 125.0f;
        }
        const float cosine = qwenSourceCosContext(angle, wholeVectorSmall);
        const float sine = qwenSourceSinContext(angle, wholeVectorSmall);
#else
        const float exponent = (2.0f * float(frequencyIndex)) / float(rotaryDimension);
        const float inverseFrequency = pow(parameters.theta, -exponent);
        const float angle = position * inverseFrequency;
        const float cosine = cos(angle);
        const float sine = sin(angle);
#endif
        output[base + index] = firstHalf
            ? normalized * cosine - partner * sine
            : normalized * cosine + partner * sine;
    }
}

/// Vector M-RoPE variant. Positions are token-major Int32 `[t,h,w]` and
/// sections partition the first rotary half; the second half uses the same axis.
kernel void qwen_qk_norm_partial_mrope(
    constant QwenMRoPEParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weight [[buffer(QwenMetalBufferIndexWeights)]],
#ifdef QWEN_PINNED_SOURCE_EXP
    device const float* inverseFrequencies [[buffer(QwenMetalBufferIndexScales)]],
#endif
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const int* positions [[buffer(QwenMetalBufferIndexScratch)]],
    uint item [[thread_position_in_grid]]
) {
#ifdef QWEN_PINNED_SOURCE_EXP
#pragma clang fp contract(off)
#endif
    const uint itemCount = parameters.tokenCount * parameters.headCount;
    if (item >= itemCount) {
        return;
    }
    const uint dimension = parameters.headDimension;
    const uint base = item * dimension;
#ifdef QWEN_PINNED_SOURCE_EXP
    const float squareSum = qwenSourceFullAttentionSquareSum(input, base, dimension);
    const float inverseRMS = 1.0f / sqrt(squareSum / float(dimension) + parameters.epsilon);
#else
    float squareSum = 0.0f;
    for (uint index = 0; index < dimension; ++index) {
        const float value = input[base + index];
        squareSum += value * value;
    }
    const float inverseRMS = rsqrt(squareSum / float(dimension) + parameters.epsilon);
#endif
    const uint rotaryDimension = parameters.rotaryDimension;
    const uint rotaryHalf = rotaryDimension / 2;
    const uint token = item / parameters.headCount;
    for (uint index = 0; index < dimension; ++index) {
        const float normalized = input[base + index] * inverseRMS * (1.0f + weight[index]);
        if (index >= rotaryDimension) {
            output[base + index] = normalized;
            continue;
        }
        const bool firstHalf = index < rotaryHalf;
        const uint frequencyIndex = firstHalf ? index : index - rotaryHalf;
        const uint partnerIndex = firstHalf ? index + rotaryHalf : index - rotaryHalf;
        const float partner = input[base + partnerIndex] * inverseRMS
            * (1.0f + weight[partnerIndex]);
#ifdef QWEN_PINNED_SOURCE_EXP
        // Official source interleaves H/W at stride three, retaining T for
        // all other frequencies. Division avoids overflow in section * 3.
        uint axis = 0;
        if (frequencyIndex % 3u == 1u
            && frequencyIndex / 3u < parameters.heightSection) {
            axis = 1;
        } else if (frequencyIndex % 3u == 2u
                   && frequencyIndex / 3u < parameters.widthSection) {
            axis = 2;
        }
#else
        uint axis = 2;
        if (frequencyIndex < parameters.temporalSection) {
            axis = 0;
        } else if (frequencyIndex
                   < parameters.temporalSection + parameters.heightSection) {
            axis = 1;
        }
#endif
        const float position = float(positions[token * 3 + axis]);
#ifdef QWEN_PINNED_SOURCE_EXP
        const float inverseFrequency = inverseFrequencies[frequencyIndex];
        const float angle = position * inverseFrequency;
        // Official SIMD trig runs before spatial-axis recomposition. Each
        // quartet therefore uses this lane's original axis position throughout.
        const uint vectorStart = frequencyIndex & ~3u;
        bool wholeVectorSmall = true;
        for (uint lane = vectorStart; lane < min(vectorStart + 4u, rotaryHalf); ++lane) {
            wholeVectorSmall = wholeVectorSmall
                && abs(position * inverseFrequencies[lane]) < 125.0f;
        }
        const float cosine = qwenSourceCosContext(angle, wholeVectorSmall);
        const float sine = qwenSourceSinContext(angle, wholeVectorSmall);
#else
        const float exponent = (2.0f * float(frequencyIndex)) / float(rotaryDimension);
        const float inverseFrequency = pow(parameters.theta, -exponent);
        const float angle = position * inverseFrequency;
        const float cosine = cos(angle);
        const float sine = sin(angle);
#endif
        output[base + index] = firstHalf
            ? normalized * cosine - partner * sine
            : normalized * cosine + partner * sine;
    }
}

/// Applies the full-attention sigmoid output gate after head-major attention has
/// been transposed/flattened into token-major order and before o_proj.
kernel void qwen_full_attention_sigmoid_gate(
    constant QwenGateParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* attention [[buffer(QwenMetalBufferIndexInput)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* rawGate [[buffer(QwenMetalBufferIndexScratch)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.elementCount) {
        return;
    }
    const float gate = 1.0f / (1.0f + exp(-rawGate[index]));
    output[index] = attention[index] * gate;
}

#ifdef QWEN_PINNED_SOURCE_EXP
/// Original BF16 source arithmetic. The shared packed pipeline is unchanged.
/// The pinned source exponential and separate Float32 operations were checked
/// against all 4096 actual same-input layer3 position0 gate outputs.
kernel void qwen_source_full_attention_sigmoid_gate(
    constant QwenGateParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* attention [[buffer(QwenMetalBufferIndexInput)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* rawGate [[buffer(QwenMetalBufferIndexScratch)]],
    uint index [[thread_position_in_grid]]
) {
#pragma clang fp contract(off)
    if (index >= parameters.elementCount) return;
    const float exponential = qwenSourceExpFloatBitScale(-rawGate[index]);
    const float denominator = 1.0f + exponential;
    const float gate = 1.0f / denominator;
    output[index] = attention[index] * gate;
}
#endif
