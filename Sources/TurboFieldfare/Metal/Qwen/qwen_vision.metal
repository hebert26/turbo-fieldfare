#include <metal_stdlib>
using namespace metal;

struct QwenVisionParameters {
    uint rows;
    uint paddedRows;
    uint inputWidth;
    uint outputWidth;
    uint intermediateWidth;
    uint heads;
    uint gridHeight;
    uint gridWidth;
    uint mergeSize;
    uint positionCount;
    float epsilon;
    uint weightScalarBytes;
    uint patchScalarBytes;
    uint reserved;
};

inline float qwen_vision_load(device const uchar* values, uint index, uint scalarBytes) {
    return scalarBytes == 4u
        ? reinterpret_cast<device const float*>(values)[index]
        : float(reinterpret_cast<device const bfloat*>(values)[index]);
}

inline float qwen_vision_erf(float value) {
    // Abramowitz-Stegun 7.1.26. MSL has no scalar erf. End-to-end error is
    // measured separately against the frozen P3 fixture.
    const float signValue = value < 0.0f ? -1.0f : 1.0f;
    const float x = abs(value);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return signValue * (1.0f - polynomial * exp(-x * x));
}

inline float qwen_vision_gelu_exact(float value) {
    return 0.5f * value
        * (1.0f + qwen_vision_erf(value * 0.7071067811865475f));
}

inline float qwen_vision_gelu_tanh(float value) {
    const float coefficient = 0.7978845608028654f;
    return 0.5f * value * (1.0f + precise::tanh(coefficient
        * (value + 0.044715f * value * value * value)));
}

kernel void qwen_vision_patch_position(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const uchar* patches [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    device const uchar* weight [[buffer(3)]],
    device const uchar* bias [[buffer(4)]],
    device const uchar* positionTable [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.paddedRows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    if (row >= p.rows) { output[index] = 0.0f; return; }
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner) {
        value = fma(qwen_vision_load(
                        patches, row * p.inputWidth + inner, p.patchScalarBytes),
                    qwen_vision_load(
                        weight, column * p.inputWidth + inner, p.weightScalarBytes), value);
    }
    const int2 coordinate = positions[row];
    const uint tableSide = uint(sqrt(float(p.positionCount)));
    const float sourceY = p.gridHeight > 1
        ? float(coordinate.x) * float(tableSide - 1) / float(p.gridHeight - 1) : 0.0f;
    const float sourceX = p.gridWidth > 1
        ? float(coordinate.y) * float(tableSide - 1) / float(p.gridWidth - 1) : 0.0f;
    const uint y0 = uint(floor(sourceY));
    const uint x0 = uint(floor(sourceX));
    const uint y1 = min(y0 + 1, tableSide - 1);
    const uint x1 = min(x0 + 1, tableSide - 1);
    const float fy = sourceY - float(y0);
    const float fx = sourceX - float(x0);
    const float top = mix(
        qwen_vision_load(positionTable, (y0 * tableSide + x0) * p.outputWidth + column, p.weightScalarBytes),
        qwen_vision_load(positionTable, (y0 * tableSide + x1) * p.outputWidth + column, p.weightScalarBytes), fx);
    const float bottom = mix(
        qwen_vision_load(positionTable, (y1 * tableSide + x0) * p.outputWidth + column, p.weightScalarBytes),
        qwen_vision_load(positionTable, (y1 * tableSide + x1) * p.outputWidth + column, p.weightScalarBytes), fx);
    output[index] = value + mix(top, bottom, fy);
}

// Official CPU Conv3D emits one spatial value per patch. Its per-patch
// 1152-by-1536 BLAS projection uses the independently qualified 64-stream
// FP32 reduction; Conv3D adds bias after the complete dot product.
inline float qwen_source_vision_patch_dot(
    constant QwenVisionParameters& p, device const uchar* patches,
    device const uchar* weight, uint row, uint column) {
    #pragma clang fp contract(off)
    float partial[64] = {0.0f};
    for (uint inner = 0u; inner < p.inputWidth; ++inner) {
        const uint stream = inner & 63u;
        partial[stream] = fma(
            qwen_vision_load(patches, row * p.inputWidth + inner, p.patchScalarBytes),
            qwen_vision_load(weight, column * p.inputWidth + inner, p.weightScalarBytes),
            partial[stream]);
    }
    float collapsed[16];
    for (uint i = 0u; i < 16u; ++i) {
        const float first = partial[i] + partial[i + 16u];
        const float second = first + partial[i + 32u];
        collapsed[i] = second + partial[i + 48u];
    }
    float groups[4];
    for (uint i = 0u; i < 4u; ++i) {
        const uint base = i * 4u;
        const float first = collapsed[base] + collapsed[base + 1u];
        const float second = first + collapsed[base + 2u];
        groups[i] = second + collapsed[base + 3u];
    }
    const float first = groups[0] + groups[1];
    const float second = first + groups[2];
    return second + groups[3];
}

kernel void qwen_source_vision_patch_position(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const uchar* patches [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    device const uchar* weight [[buffer(3)]],
    device const uchar* bias [[buffer(4)]],
    device const uchar* positionTable [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    const uint count = p.paddedRows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    if (row >= p.rows) { output[index] = 0.0f; return; }
    float value;
    if (p.inputWidth == 1536u && p.outputWidth == 1152u) {
        value = qwen_source_vision_patch_dot(p, patches, weight, row, column)
            + qwen_vision_load(bias, column, p.weightScalarBytes);
    } else {
        // Keep the generic fixture arithmetic outside the admitted source geometry.
        value = qwen_vision_load(bias, column, p.weightScalarBytes);
        for (uint inner = 0u; inner < p.inputWidth; ++inner) {
            value = fma(qwen_vision_load(
                            patches, row * p.inputWidth + inner, p.patchScalarBytes),
                        qwen_vision_load(weight,
                            column * p.inputWidth + inner, p.weightScalarBytes), value);
        }
    }
    const int2 coordinate = positions[row];
    const uint tableSide = uint(sqrt(float(p.positionCount)));
    const float sourceY = p.gridHeight > 1
        ? float(coordinate.x) * float(tableSide - 1) / float(p.gridHeight - 1) : 0.0f;
    const float sourceX = p.gridWidth > 1
        ? float(coordinate.y) * float(tableSide - 1) / float(p.gridWidth - 1) : 0.0f;
    const uint y0 = uint(floor(sourceY));
    const uint x0 = uint(floor(sourceX));
    const uint y1 = min(y0 + 1, tableSide - 1);
    const uint x1 = min(x0 + 1, tableSide - 1);
    const float fy = sourceY - float(y0);
    const float fx = sourceX - float(x0);
    // Pinned vision_utils.py constructs both taps as 1-abs(distance),
    // then multiplies the per-axis weights. Keep every FP32 rounding.
    const float wy0 = max(0.0f, 1.0f - abs(fy));
    const float wy1 = max(0.0f, 1.0f - abs(fy - 1.0f));
    const float wx0 = max(0.0f, 1.0f - abs(fx));
    const float wx1 = max(0.0f, 1.0f - abs(fx - 1.0f));
    const float w00 = wy0 * wx0;
    const float w01 = wy0 * wx1;
    const float w10 = wy1 * wx0;
    const float w11 = wy1 * wx1;
    const float p00 = qwen_vision_load(positionTable,
        (y0 * tableSide + x0) * p.outputWidth + column, p.weightScalarBytes) * w00;
    const float p01 = qwen_vision_load(positionTable,
        (y0 * tableSide + x1) * p.outputWidth + column, p.weightScalarBytes) * w01;
    const float p10 = qwen_vision_load(positionTable,
        (y1 * tableSide + x0) * p.outputWidth + column, p.weightScalarBytes) * w10;
    const float p11 = qwen_vision_load(positionTable,
        (y1 * tableSide + x1) * p.outputWidth + column, p.weightScalarBytes) * w11;
    // sum(dim=1) reduces the four separately materialized products in tap order.
    float position = 0.0f;
    position += p00;
    position += p01;
    position += p10;
    position += p11;
    // Reserved component selection is set only by the opt-in initial diagnostic.
    output[index] = p.reserved == 1u ? value
        : p.reserved == 2u ? position : value + position;
}

// Pinned Torch 2.10 CPU moments_utils.h: NEON Float4 Welford updates,
// sixteen-vector chunks and cascade merging, followed by scalar lane merging.
// Only the admitted source vision width (1152) uses this reduction.
inline float2 qwen_source_vision_moments_1152(device const float* input) {
    #pragma clang fp contract(off)
    uint counts[5] = {0u, 0u, 0u, 0u, 0u};
    float4 means[5] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    float4 squares[5] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint chunk = 0u; chunk < 18u; ++chunk) {
        const uint length = min(16u, 288u - chunk * 16u);
        float4 mean = float4(0.0f);
        float4 square = float4(0.0f);
        for (uint vector = 0u; vector < length; ++vector) {
            const uint offset = (chunk * 16u + vector) * 4u;
            const float4 value = float4(input[offset], input[offset + 1u],
                                       input[offset + 2u], input[offset + 3u]);
            const float4 delta = value - mean;
            mean = fma(float4(1.0f / float(vector + 1u)), delta, mean);
            square = fma(delta, value - mean, square);
        }
        // AddMomentsVec merges one chunk into level zero.
        uint combined = counts[0] + length;
        float coefficient = float(length) / float(combined);
        float4 delta = mean - means[0];
        float4 weightedDelta = float4(coefficient) * delta;
        float4 countedDelta = delta * float4(float(counts[0]));
        means[0] = means[0] + weightedDelta;
        squares[0] = fma(countedDelta, weightedDelta, squares[0] + square);
        counts[0] = combined;
        uint mask = chunk + 1u;
        for (uint level = 1u; level < 5u && (mask & 1u) == 0u; ++level) {
            combined = counts[level] + counts[level - 1u];
            coefficient = float(counts[level - 1u]) / float(combined);
            delta = means[level - 1u] - means[level];
            weightedDelta = float4(coefficient) * delta;
            countedDelta = delta * float4(float(counts[level]));
            means[level] = means[level] + weightedDelta;
            squares[level] = fma(countedDelta, weightedDelta,
                                 squares[level] + squares[level - 1u]);
            counts[level] = combined;
            counts[level - 1u] = 0u;
            means[level - 1u] = float4(0.0f);
            squares[level - 1u] = float4(0.0f);
            mask >>= 1u;
        }
    }
    for (uint level = 1u; level < 5u; ++level) {
        const uint combined = counts[0] + counts[level];
        const float coefficient = combined == 0u ? 0.0f : float(counts[level]) / float(combined);
        const float4 delta = means[level] - means[0];
        const float4 weightedDelta = float4(coefficient) * delta;
        const float4 countedDelta = delta * float4(float(counts[0]));
        means[0] = means[0] + weightedDelta;
        squares[0] = fma(countedDelta, weightedDelta, squares[0] + squares[level]);
        counts[0] = combined;
    }
    uint count = 0u;
    float mean = 0.0f;
    float square = 0.0f;
    for (uint lane = 0u; lane < 4u; ++lane) {
        const uint combined = count + 288u;
        const float coefficient = 288.0f / float(combined);
        const float delta = means[0][lane] - mean;
        mean = fma(coefficient, delta, mean);
        // The pinned CPU AddMoments contracts the final multiply/add here.
        // Keep the preceding products and outer accumulation separately rounded.
        const float correction = (delta * delta) * coefficient;
        square = square + fma(correction, float(count), squares[0][lane]);
        count = combined;
    }
    return float2(mean, square / 1152.0f);
}

inline void qwen_source_vision_norm_row(
    constant QwenVisionParameters& p, device const float* input,
    device const uchar* normWeight, device const uchar* normBias,
    device float* output, uint row) {
    #pragma clang fp contract(off)
    const uint base = row * p.inputWidth;
    float mean;
    float variance;
    if (p.inputWidth == 1152u) {
        const float2 moments = qwen_source_vision_moments_1152(input + base);
        mean = moments.x;
        variance = moments.y;
    } else {
        // Preserve the generic fixture path outside the qualified source width.
        mean = 0.0f;
        for (uint i = 0u; i < p.inputWidth; ++i) mean += input[base + i];
        mean /= float(p.inputWidth);
        variance = 0.0f;
        for (uint i = 0u; i < p.inputWidth; ++i) {
            const float centered = input[base + i] - mean;
            variance = fma(centered, centered, variance);
        }
        variance /= float(p.inputWidth);
    }
    const float inverse = 1.0f / precise::sqrt(variance + p.epsilon);
    for (uint i = 0u; i < p.inputWidth; ++i) {
        output[base + i] = (input[base + i] - mean) * inverse
            * qwen_vision_load(normWeight, i, p.weightScalarBytes)
            + qwen_vision_load(normBias, i, p.weightScalarBytes);
    }
}

kernel void qwen_source_vision_layernorm(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* normWeight [[buffer(2)]],
    device const uchar* normBias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.paddedRows) return;
    if (row >= p.rows) {
        for (uint i = 0u; i < p.inputWidth; ++i) output[row * p.inputWidth + i] = 0.0f;
        return;
    }
    qwen_source_vision_norm_row(p, input, normWeight, normBias, output, row);
}

kernel void qwen_source_vision_merger_norm_pack(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* normWeight [[buffer(2)]],
    device const uchar* normBias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    // Four consecutive source rows already form one block-major packed row.
    qwen_source_vision_norm_row(p, input, normWeight, normBias, output, row);
}

#ifdef QWEN_PINNED_SOURCE_EXP
// Pinned source SIMD GELU arithmetic, independently qualified on CPU/GPU.
// SLEEF3.8 commit5a1d179df9cf652951b59010a2d2075372d67f68:
// src/common/df.h FMA branch; src/libm/sleefsimdsp.c expk2f/xtanhf.
// Copyright Naoki Shibata and contributors2010-2024. Boost1.0 below.
// Erf operation order from pinned Torch2.10 vec128_float_neon.h619-649.
// Arm exp_u20 polynomial from the same header315-349, preserving Float4 fallback.
#include <metal_stdlib>
using namespace metal;

inline float2 qvAdd2Scalar(float2 x, float y) {
#pragma clang fp contract(off)
    float s = x.x + y;
    float v = s - x.x;
    float t = (x.x - (s - v)) + (y - v);
    return float2(s, t + x.y);
}
inline float2 qvAddPair(float2 x, float2 y) {
#pragma clang fp contract(off)
    float s = x.x + y.x;
    return float2(s, (((x.x - s) + y.x) + x.y) + y.y);
}
inline float2 qvAdd2Pair(float2 x, float2 y) {
#pragma clang fp contract(off)
    float s = x.x + y.x;
    float v = s - x.x;
    float t = (x.x - (s - v)) + (y.x - v);
    return float2(s, t + (x.y + y.y));
}
inline float2 qvAddScalarPair(float x, float2 y) {
#pragma clang fp contract(off)
    float s = x + y.x;
    return float2(s, ((x - s) + y.x) + y.y);
}
inline float2 qvMulScalar(float2 x, float y) {
#pragma clang fp contract(off)
    float s = x.x * y;
    return float2(s, fma(x.y, y, fma(x.x, y, -s)));
}
inline float2 qvMulPair(float2 x, float2 y) {
#pragma clang fp contract(off)
    float s = x.x * y.x;
    return float2(s, fma(x.x, y.y, fma(x.y, y.x, fma(x.x, y.x, -s))));
}
inline float2 qvSquare(float2 x) {
#pragma clang fp contract(off)
    float s = x.x * x.x;
    return float2(s, fma(x.x + x.x, x.y, fma(x.x, x.x, -s)));
}
inline float2 qvRecPair(float2 d) {
#pragma clang fp contract(off)
    float s = 1.0f / d.x;
    return float2(s, s * fma(-d.y, s, fma(-d.x, s, 1.0f)));
}
inline float2 qvDivPair(float2 n, float2 d) {
#pragma clang fp contract(off)
    float t = 1.0f / d.x;
    float s = n.x * t;
    float u = fma(t, n.x, -s);
    float v = fma(-d.y, t, fma(-d.x, t, 1.0f));
    return float2(s, fma(s, v, fma(n.y, t, u)));
}
inline float2 qvExpPair(float2 d) {
#pragma clang fp contract(off)
    int q = int(rint((d.x + d.y) * 1.442695040888963407359924681001892137426645954152985934135449406931f));
    float2 s = qvAdd2Scalar(d, float(q) * -0.693145751953125f);
    s = qvAdd2Scalar(s, float(q) * -1.428606765330187045e-6f);
    float u = 0.1980960224e-3f;
    u = fma(u, s.x, 0.1394256484e-2f);
    u = fma(u, s.x, 0.8333456703e-2f);
    u = fma(u, s.x, 0.4166637361e-1f);
    float2 t = qvAdd2Scalar(qvMulScalar(s, u), 0.166666659414234244790680580464f);
    t = qvAdd2Scalar(qvMulPair(s, t), 0.5f);
    t = qvAdd2Pair(s, qvMulPair(qvSquare(s), t));
    t = qvAddScalarPair(1.0f, t);
    int halfQ = q >> 1;
    float firstScale = as_type<float>(uint(halfQ + 127) << 23);
    float secondScale = as_type<float>(uint(q - halfQ + 127) << 23);
    t.x = (t.x * firstScale) * secondScale;
    t.y = (t.y * firstScale) * secondScale;
    return t;
}
inline float qvSourceTanh(float x) {
#pragma clang fp contract(off)
    uint raw = as_type<uint>(x);
    uint magnitude = raw & 0x7fffffffu;
    // tanh of a subnormal rounds to the input. Preserve it without GPU FTZ.
    if (magnitude < 0x00800000u) return x;
    if (abs(x) > 8.664339742f) return as_type<float>(0x3f800000u | (raw & 0x80000000u));
    float2 d = qvExpPair(float2(abs(x), 0.0f));
    float2 e = qvRecPair(d);
    d = qvDivPair(qvAddPair(d, -e), qvAddPair(d, e));
    float y = d.x + d.y;
    if (isnan(y)) y = 1.0f;
    return as_type<float>(as_type<uint>(y) ^ (raw & 0x80000000u));
}
inline float qvHalfTiny(float x) {
    uint raw = as_type<uint>(x);
    uint significand = raw & 0x007fffffu;
    if ((raw & 0x7f800000u) != 0u) significand |= 0x00800000u;
    uint roundedHalf = significand >> 1;
    if ((significand & 1u) && (roundedHalf & 1u)) ++roundedHalf;
    return as_type<float>((raw & 0x80000000u) | roundedHalf);
}
inline float qvSourceGeluTanh(float x) {
#pragma clang fp contract(off)
    // Both outer factors round to0.5 and1 in this tiny range. Preserve the
    // correctly rounded half-subnormal using bits instead of flushing arithmetic.
    if ((as_type<uint>(x) & 0x7fffffffu) < 0x01000000u) return qvHalfTiny(x);
    float cube = (x * x) * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * cube);
    return (0.5f * x) * (1.0f + qvSourceTanh(inner));
}
inline float qvArmExpU20(float x) {
#pragma clang fp contract(off)
    float n = round(x * 0x1.715476p+0f);
    float r = fma(-n, 0x1.62e4p-1f, x);
    r = fma(-n, 0x1.7f7d1cp-20f, r);
    float scale = as_type<float>((uint(int(n)) << 23) + 0x3f800000u);
    float r2 = r * r;
    float p = fma(r, 0x1.0e4020p-7f, 0x1.573e2ep-5f);
    float q = fma(r, 0x1.555e66p-3f, 0x1.fffdb6p-2f);
    q = fma(p, r2, q);
    p = 0x1.ffffecp-1f * r;
    float polynomial = fma(q, r2, p);
    return fma(polynomial, scale, scale);
}
inline float qvSourceErf(float value, bool groupFallback) {
#pragma clang fp contract(off)
    uint sign = as_type<uint>(value) & 0x80000000u;
    if (abs(value) >= 8.0f) return as_type<float>(0x3f800000u | sign);
    float t = 1.0f / fma(0.3275911f, abs(value), 1.0f);
    float r = fma(1.061405429f, t, -1.453152027f);
    r = fma(r, t, 1.421413741f);
    r = fma(r, t, -0.284496736f);
    r = fma(r, t, 0.254829592f);
    float negativeSquare = -(value * value);
    float exponential = groupFallback ? qwenSourceExpFloatBitScale(negativeSquare) : qvArmExpU20(negativeSquare);
    float result = fma(t * (-exponential), r, 1.0f);
    return as_type<float>(as_type<uint>(result) ^ sign);
}
inline float qvSourceGeluErf(float x, bool groupFallback) {
#pragma clang fp contract(off)
    if ((as_type<uint>(x) & 0x7fffffffu) < 0x01000000u) return qvHalfTiny(x);
    return (x * 0.5f) * (1.0f + qvSourceErf(x * 0.7071067811865475f, groupFallback));
}

/*
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

*/


// All source row widths are divisible by eight, so Torch's vector path has
// no scalar tail. Four adjacent lanes share the erf exponential fallback.
kernel void qwen_source_vision_gelu_in_place(
    constant QwenVisionParameters& p [[buffer(0)]],
    device float* values [[buffer(1)]],
    uint group [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    const uint base = group * 4u;
    if (base + 3u >= p.rows * p.outputWidth) return;
    const float4 input = float4(values[base], values[base + 1u], values[base + 2u], values[base + 3u]);
    const float4 scaled = input * float4(0.7071067811865475f);
    const float4 negativeSquare = -(scaled * scaled);
    const bool fallback = any(abs(negativeSquare) > float4(0x1.5d5e2ap+6f));
    for (uint lane = 0u; lane < 4u; ++lane) values[base + lane] = qvSourceGeluErf(input[lane], fallback);
}

kernel void qwen_source_vision_gelu_tanh_in_place(
    constant QwenVisionParameters& p [[buffer(0)]],
    device float* values [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if (index >= p.rows * p.outputWidth) return;
    values[index] = qvSourceGeluTanh(values[index]);
}

kernel void qwen_source_vision_linear_gelu_tanh(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* weight [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if (index >= p.rows * p.outputWidth) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0u; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight, column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = qvSourceGeluTanh(value);
}

inline float qwen_source_vision_attention_score(
    constant QwenVisionParameters& p, device const float* qkv,
    uint row, uint keyRow, uint head, float scale) {
    #pragma clang fp contract(off)
    const uint queryBase = row * 3u * p.inputWidth + head * 72u;
    const uint keyBase = keyRow * 3u * p.inputWidth + p.inputWidth + head * 72u;
    float score = 0.0f;
    for (uint d = 0u; d < 72u; ++d) score = fma(qkv[queryBase + d], qkv[keyBase + d], score);
    return score * scale;
}

kernel void qwen_source_vision_attention(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* qkv [[buffer(1)]], device float* output [[buffer(2)]],
    constant float& scale [[buffer(3)]], uint item [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if (item >= p.paddedRows * p.heads) return;
    // Source-only attention launches bounded query ranges. All key rows remain
    // visible, and the reserved field is zero in other kernels.
    const uint row = p.reserved + item / p.heads;
    if (row >= p.paddedRows) return;
    const uint head = item % p.heads;
    const uint outputBase = row * p.inputWidth + head * 72u;
    if (row >= p.rows) {
        for (uint d = 0u; d < 72u; ++d) output[outputBase + d] = 0.0f;
        return;
    }
    float maximum = -INFINITY;
    for (uint keyRow = 0u; keyRow < p.rows; ++keyRow)
        maximum = max(maximum, qwen_source_vision_attention_score(p, qkv, row, keyRow, head, scale));
    float laneSums[4] = {0.0f};
    for (uint keyRow = 0u; keyRow < p.rows; ++keyRow) {
        const float exponential = qwenSourceExpFloatBitScale(
            qwen_source_vision_attention_score(p, qkv, row, keyRow, head, scale) - maximum);
        laneSums[keyRow & 3u] += exponential;
    }
    const float denominator = (laneSums[0] + laneSums[2]) + (laneSums[1] + laneSums[3]);
    const float reciprocal = 1.0f / denominator;
    float accumulator[72] = {0.0f};
    for (uint keyRow = 0u; keyRow < p.rows; ++keyRow) {
        const float exponential = qwenSourceExpFloatBitScale(
            qwen_source_vision_attention_score(p, qkv, row, keyRow, head, scale) - maximum);
        // Round normalization before multiplying V, as the official softmax
        // tensor does. Recomputing scores avoids row-squared scratch storage.
        const float probability = exponential * reciprocal;
        const uint valueBase = keyRow * 3u * p.inputWidth + 2u * p.inputWidth + head * 72u;
        for (uint d = 0u; d < 72u; ++d) accumulator[d] = fma(probability, qkv[valueBase + d], accumulator[d]);
    }
    for (uint d = 0u; d < 72u; ++d) output[outputBase + d] = accumulator[d];
}

#endif

kernel void qwen_vision_layernorm(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* normWeight [[buffer(2)]],
    device const uchar* normBias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.paddedRows) return;
    const uint base = row * p.inputWidth;
    if (row >= p.rows) {
        for (uint i = 0; i < p.inputWidth; ++i) output[base + i] = 0.0f;
        return;
    }
    float mean = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) mean += input[base + i];
    mean /= float(p.inputWidth);
    float variance = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) {
        const float centered = input[base + i] - mean;
        variance = fma(centered, centered, variance);
    }
    const float inverse = rsqrt(variance / float(p.inputWidth) + p.epsilon);
    for (uint i = 0; i < p.inputWidth; ++i) {
        output[base + i] = (input[base + i] - mean) * inverse
            * qwen_vision_load(normWeight, i, p.weightScalarBytes)
            + qwen_vision_load(normBias, i, p.weightScalarBytes);
    }
}

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


// Candidate source vision rotation; no source weights or expected coefficients.
inline float2 qwen_source_vision_rotary_pair(
    device const int2* positions, device const float* inverseFrequencies,
    uint row, uint dimension) {
    #pragma clang fp contract(off)
    const uint frequencyCount = 18u;
    const uint axis = dimension / frequencyCount;
    const uint frequency = dimension % frequencyCount;
    const float coordinate = float(axis == 0u ? positions[row].x : positions[row].y);
    const float angle = coordinate * inverseFrequencies[frequency];
    // Original contiguous Torch freqs are [row,axis,18]; SIMD groups can
    // cross H/W at H16,H17,W0,W1. Recomposition to72 occurs after trig.
    const uint vectorStart = dimension & ~3u;
    bool wholeVectorSmall = true;
    for (uint lane = 0u; lane < 4u; ++lane) {
        const uint source = vectorStart + lane;
        const float sourceCoordinate = float(source < frequencyCount
            ? positions[row].x : positions[row].y);
        wholeVectorSmall = wholeVectorSmall
            && abs(sourceCoordinate * inverseFrequencies[source % frequencyCount]) < 125.0f;
    }
    return float2(qwenSourceCosContext(angle, wholeVectorSmall),
                  qwenSourceSinContext(angle, wholeVectorSmall));
}

kernel void qwen_source_vision_rotate_qk(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const int2* positions [[buffer(1)]],
    device float* qkv [[buffer(2)]],
    device const float* inverseFrequencies [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    const uint dimension = index % 72u;
    const uint vector = index / 72u;
    const uint head = vector % p.heads;
    const uint row = vector / p.heads;
    if (row >= p.rows || dimension >= 36u) return;
    const float2 coefficients = qwen_source_vision_rotary_pair(positions, inverseFrequencies, row, dimension);
    for (uint projection = 0u; projection < 2u; ++projection) {
        const uint base = row * 3u * p.inputWidth + projection * p.inputWidth + head * 72u;
        const float first = qkv[base + dimension];
        const float second = qkv[base + dimension + 36u];
        const float firstCosine = first * coefficients.x;
        const float negativeSecondSine = (-second) * coefficients.y;
        const float secondCosine = second * coefficients.x;
        const float firstSine = first * coefficients.y;
        qkv[base + dimension] = firstCosine + negativeSecondSine;
        qkv[base + dimension + 36u] = secondCosine + firstSine;
    }
}


kernel void qwen_vision_rotate_qk(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const int2* positions [[buffer(1)]],
    device float* qkv [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint headDimension = p.inputWidth / p.heads;
    const uint rotaryHalf = headDimension / 2;
    const uint count = p.rows * p.heads * headDimension;
    if (index >= count) return;
    const uint dimension = index % headDimension;
    const uint vector = index / headDimension;
    const uint head = vector % p.heads;
    const uint row = vector / p.heads;
    if (dimension >= rotaryHalf) return;
    const uint axisHalf = rotaryHalf / 2;
    const uint axis = dimension < axisHalf ? 0 : 1;
    const uint frequency = dimension % axisHalf;
    const float coordinate = float(axis == 0 ? positions[row].x : positions[row].y);
    const float inverseFrequency = pow(10000.0f,
        -2.0f * float(frequency) / float(rotaryHalf));
    const float angle = coordinate * inverseFrequency;
    const float c = cos(angle);
    const float s = sin(angle);
    const uint rowBase = row * 3u * p.inputWidth;
    for (uint projection = 0; projection < 2; ++projection) {
        const uint base = rowBase + projection * p.inputWidth
            + head * headDimension;
        const float first = qkv[base + dimension];
        const float second = qkv[base + dimension + rotaryHalf];
        qkv[base + dimension] = first * c - second * s;
        qkv[base + dimension + rotaryHalf] = first * s + second * c;
    }
}

kernel void qwen_vision_attention(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= p.paddedRows * p.heads) return;
    const uint row = item / p.heads;
    const uint head = item % p.heads;
    const uint headDimension = p.inputWidth / p.heads;
    const uint outputBase = row * p.inputWidth + head * headDimension;
    if (row >= p.rows) {
        for (uint d = 0; d < headDimension; ++d) output[outputBase + d] = 0.0f;
        return;
    }
    const uint queryBase = row * 3u * p.inputWidth + head * headDimension;
    const float scale = rsqrt(float(headDimension));
    float maximum = -INFINITY;
    for (uint keyRow = 0; keyRow < p.rows; ++keyRow) {
        const uint keyBase = keyRow * 3u * p.inputWidth + p.inputWidth
            + head * headDimension;
        float score = 0.0f;
        for (uint d = 0; d < headDimension; ++d)
            score = fma(qkv[queryBase + d], qkv[keyBase + d], score);
        maximum = max(maximum, score * scale);
    }
    // Official head dimension is 72; tiny frozen fixtures use 8.
    float accumulator[72];
    for (uint d = 0; d < headDimension; ++d) accumulator[d] = 0.0f;
    float denominator = 0.0f;
    for (uint keyRow = 0; keyRow < p.rows; ++keyRow) {
        const uint keyBase = keyRow * 3u * p.inputWidth + p.inputWidth
            + head * headDimension;
        float score = 0.0f;
        for (uint d = 0; d < headDimension; ++d)
            score = fma(qkv[queryBase + d], qkv[keyBase + d], score);
        const float probability = exp(score * scale - maximum);
        denominator += probability;
        const uint valueBase = keyRow * 3u * p.inputWidth + 2u * p.inputWidth
            + head * headDimension;
        for (uint d = 0; d < headDimension; ++d)
            accumulator[d] = fma(probability, qkv[valueBase + d], accumulator[d]);
    }
    for (uint d = 0; d < headDimension; ++d)
        output[outputBase + d] = accumulator[d] / denominator;
}

kernel void qwen_vision_linear_residual(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device const uchar* weight [[buffer(3)]],
    device const uchar* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.paddedRows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    if (row >= p.rows) { output[index] = 0.0f; return; }
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = residual[row * p.outputWidth + column] + value;
}

kernel void qwen_vision_linear_gelu_tanh(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* weight [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.rows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = qwen_vision_gelu_tanh(value);
}

kernel void qwen_vision_merger_norm_pack(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* normWeight [[buffer(2)]],
    device const uchar* normBias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint sourceRow [[thread_position_in_grid]]) {
    if (sourceRow >= p.rows) return;
    const uint inputBase = sourceRow * p.inputWidth;
    float mean = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) mean += input[inputBase + i];
    mean /= float(p.inputWidth);
    float variance = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) {
        const float centered = input[inputBase + i] - mean;
        variance = fma(centered, centered, variance);
    }
    const float inverse = rsqrt(variance / float(p.inputWidth) + p.epsilon);
    // Preprocessing is block-major, so four consecutive rows are one packed row.
    const uint outputBase = sourceRow * p.inputWidth;
    for (uint i = 0; i < p.inputWidth; ++i) {
        output[outputBase + i] = (input[inputBase + i] - mean) * inverse
            * qwen_vision_load(normWeight, i, p.weightScalarBytes)
            + qwen_vision_load(normBias, i, p.weightScalarBytes);
    }
}

kernel void qwen_vision_linear_gelu(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* weight [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.rows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = qwen_vision_gelu_exact(value);
}

kernel void qwen_vision_linear(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* weight [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.rows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = value;
}

kernel void qwen_vision_add_layer_bias(
    constant uint& count [[buffer(0)]],
    constant float& bias [[buffer(1)]],
    device float* values [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) values[index] += bias;
}
