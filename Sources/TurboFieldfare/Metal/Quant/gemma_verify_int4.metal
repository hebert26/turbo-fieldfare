#include <metal_stdlib>
using namespace metal;

constant uint GEMMA_VERIFY_BATCH [[function_constant(90)]];

// Reuse each packed weight across the proposed tokens.
// Each token keeps the normal decode order of sums and multiply-adds.
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma_verify_int4(
    device const uchar* weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* x0 [[buffer(3)]],
    device const half* x1 [[buffer(4)]],
    device const half* x2 [[buffer(5)]],
    device const half* x3 [[buffer(6)]],
    device const half* x4 [[buffer(7)]],
    device const half* x5 [[buffer(8)]],
    device const half* x6 [[buffer(9)]],
    device half* y0 [[buffer(10)]],
    device half* y1 [[buffer(11)]],
    device half* y2 [[buffer(12)]],
    device half* y3 [[buffer(13)]],
    device half* y4 [[buffer(14)]],
    device half* y5 [[buffer(15)]],
    device half* y6 [[buffer(16)]],
    constant uint& rows [[buffer(17)]],
    constant uint& columns [[buffer(18)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group * 4u + simd;
    if (row >= rows) return;
    device const half* inputs[7] = {x0, x1, x2, x3, x4, x5, x6};
    device half* outputs[7] = {y0, y1, y2, y3, y4, y5, y6};
    float acc[7] = {};
    const uint groups = columns / 64u;
    const uint blocks = groups / 4u;
    for (uint block = 0; block < blocks; ++block) {
        const uint byteBase = block * 128u + lane * 4u;
        const uint scaleIndex = row * groups + block * 4u + (lane >> 3);
        device const ushort* packed = (device const ushort*)(weights + row * (columns / 2u) + byteBase);
        const uint w0 = uint(packed[0]);
        const uint w1 = uint(packed[1]);
        const float scale = float(scales[scaleIndex]);
        const float bias = float(biases[scaleIndex]);
        #pragma unroll
        for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
            device const half* x = inputs[token] + byteBase * 2u;
            const half4 a = *((device const half4*)x);
            const half4 b = *((device const half4*)(x + 4u));
            const float e0 = float(a.x), e1 = float(a.y), e2 = float(a.z), e3 = float(a.w);
            const float e4 = float(b.x), e5 = float(b.y), e6 = float(b.z), e7 = float(b.w);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            float dot = 0;
            dot = fma(float(w0 & 0x000fu), e0, dot);
            dot = fma(float(w0 & 0x00f0u), e1 * 0x1p-4f, dot);
            dot = fma(float(w0 & 0x0f00u), e2 * 0x1p-8f, dot);
            dot = fma(float(w0 & 0xf000u), e3 * 0x1p-12f, dot);
            dot = fma(float(w1 & 0x000fu), e4, dot);
            dot = fma(float(w1 & 0x00f0u), e5 * 0x1p-4f, dot);
            dot = fma(float(w1 & 0x0f00u), e6 * 0x1p-8f, dot);
            dot = fma(float(w1 & 0xf000u), e7 * 0x1p-12f, dot);
            acc[token] = fma(scale, dot, acc[token]);
            acc[token] = fma(bias, sum, acc[token]);
        }
    }
    for (uint g = blocks * 4u; g < groups; ++g) {
        const uchar packed = weights[row * (columns / 2u) + g * 32u + lane];
        const float scale = float(scales[row * groups + g]);
        const float bias = float(biases[row * groups + g]);
        #pragma unroll
        for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
            const float a = float(inputs[token][g * 64u + lane * 2u]);
            const float b = float(inputs[token][g * 64u + lane * 2u + 1u]);
            float dot = fma(float(uint(packed & 0x0fu)), a, 0.0f);
            dot = fma(float(uint(packed >> 4)), b, dot);
            acc[token] = fma(scale, dot, acc[token]);
            acc[token] = fma(bias, a + b, acc[token]);
        }
    }
    #pragma unroll
    for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
        const float value = simd_sum(acc[token]);
        if (lane == 0) outputs[token][row] = half(value);
    }
}

// Join independent projections into one dispatch.
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma_verify_int4_many(
    device const uchar* weights0 [[buffer(0)]],
    device const bfloat* scales0 [[buffer(1)]],
    device const bfloat* biases0 [[buffer(2)]],
    device const uchar* weights1 [[buffer(3)]],
    device const bfloat* scales1 [[buffer(4)]],
    device const bfloat* biases1 [[buffer(5)]],
    device const uchar* weights2 [[buffer(6)]],
    device const bfloat* scales2 [[buffer(7)]],
    device const bfloat* biases2 [[buffer(8)]],
    device const half* x0 [[buffer(9)]],
    device const half* x1 [[buffer(10)]],
    device const half* x2 [[buffer(11)]],
    device const half* x3 [[buffer(12)]],
    device const half* x4 [[buffer(13)]],
    device half* y0 [[buffer(14)]],
    device half* y1 [[buffer(15)]],
    device half* y2 [[buffer(16)]],
    device half* y3 [[buffer(17)]],
    device half* y4 [[buffer(18)]],
    device half* y5 [[buffer(19)]],
    device half* y6 [[buffer(20)]],
    device half* y7 [[buffer(21)]],
    device half* y8 [[buffer(22)]],
    device half* y9 [[buffer(23)]],
    device half* y10 [[buffer(24)]],
    device half* y11 [[buffer(25)]],
    device half* y12 [[buffer(26)]],
    device half* y13 [[buffer(27)]],
    device half* y14 [[buffer(28)]],
    constant uint* rowCounts [[buffer(29)]],
    constant uint& columns [[buffer(30)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group.x * 4u + simd;
    device const uchar* weightRows[3] = {weights0, weights1, weights2};
    device const bfloat* scaleRows[3] = {scales0, scales1, scales2};
    device const bfloat* biasRows[3] = {biases0, biases1, biases2};
    device const uchar* weights = weightRows[group.z];
    device const bfloat* scales = scaleRows[group.z];
    device const bfloat* biases = biasRows[group.z];
    const uint rows = rowCounts[group.z];
    if (row >= rows) return;
    device const half* inputs[5] = {x0, x1, x2, x3, x4};
    device half* allOutputs[15] = {y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14};
    device half* outputs[5] = {allOutputs[group.z * 5], allOutputs[group.z * 5 + 1],
        allOutputs[group.z * 5 + 2], allOutputs[group.z * 5 + 3], allOutputs[group.z * 5 + 4]};
    float acc[5] = {0, 0, 0, 0, 0};
    const uint groups = columns / 64u;
    const uint blocks = groups / 4u;
    for (uint block = 0; block < blocks; ++block) {
        const uint byteBase = block * 128u + lane * 4u;
        const uint scaleIndex = row * groups + block * 4u + (lane >> 3);
        device const ushort* packed = (device const ushort*)(weights + row * (columns / 2u) + byteBase);
        const uint w0 = uint(packed[0]);
        const uint w1 = uint(packed[1]);
        const float scale = float(scales[scaleIndex]);
        const float bias = float(biases[scaleIndex]);
        #pragma unroll
        for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
            device const half* x = inputs[token] + byteBase * 2u;
            const half4 a = *((device const half4*)x);
            const half4 b = *((device const half4*)(x + 4u));
            const float e0 = float(a.x), e1 = float(a.y), e2 = float(a.z), e3 = float(a.w);
            const float e4 = float(b.x), e5 = float(b.y), e6 = float(b.z), e7 = float(b.w);
            const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            float dot = 0;
            dot = fma(float(w0 & 0x000fu), e0, dot);
            dot = fma(float(w0 & 0x00f0u), e1 * 0x1p-4f, dot);
            dot = fma(float(w0 & 0x0f00u), e2 * 0x1p-8f, dot);
            dot = fma(float(w0 & 0xf000u), e3 * 0x1p-12f, dot);
            dot = fma(float(w1 & 0x000fu), e4, dot);
            dot = fma(float(w1 & 0x00f0u), e5 * 0x1p-4f, dot);
            dot = fma(float(w1 & 0x0f00u), e6 * 0x1p-8f, dot);
            dot = fma(float(w1 & 0xf000u), e7 * 0x1p-12f, dot);
            acc[token] = fma(scale, dot, acc[token]);
            acc[token] = fma(bias, sum, acc[token]);
        }
    }
    for (uint g = blocks * 4u; g < groups; ++g) {
        const uchar packed = weights[row * (columns / 2u) + g * 32u + lane];
        const float scale = float(scales[row * groups + g]);
        const float bias = float(biases[row * groups + g]);
        #pragma unroll
        for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
            const float a = float(inputs[token][g * 64u + lane * 2u]);
            const float b = float(inputs[token][g * 64u + lane * 2u + 1u]);
            float dot = fma(float(uint(packed & 0x0fu)), a, 0.0f);
            dot = fma(float(uint(packed >> 4)), b, dot);
            acc[token] = fma(scale, dot, acc[token]);
            acc[token] = fma(bias, a + b, acc[token]);
        }
    }
    #pragma unroll
    for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
        const float value = simd_sum(acc[token]);
        if (lane == 0) outputs[token][row] = half(value);
    }
}

// Share input loads across four output rows. Keep each row's sums.
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma_verify_int4_rows4(
    device const uchar* weights [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* x0 [[buffer(3)]],
    device const half* x1 [[buffer(4)]],
    device const half* x2 [[buffer(5)]],
    device const half* x3 [[buffer(6)]],
    device const half* x4 [[buffer(7)]],
    device half* y0 [[buffer(8)]],
    device half* y1 [[buffer(9)]],
    device half* y2 [[buffer(10)]],
    device half* y3 [[buffer(11)]],
    device half* y4 [[buffer(12)]],
    constant uint& rows [[buffer(13)]],
    constant uint& columns [[buffer(14)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint firstRow = (group * 4u + simd) * 4u;
    if (firstRow >= rows) return;
    device const half* inputs[5] = {x0, x1, x2, x3, x4};
    device half* outputs[5] = {y0, y1, y2, y3, y4};
    float acc[4u][5] = {};
    const uint groups = columns / 64u;
    const uint blocks = groups / 4u;
    for (uint block = 0; block < blocks; ++block) {
        const uint byteBase = block * 128u + lane * 4u;
        float elements[5][8], sums[5];
        #pragma unroll
        for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
            device const half* x = inputs[token] + byteBase * 2u;
            const half4 a = *((device const half4*)x);
            const half4 b = *((device const half4*)(x + 4u));
            const float e0 = float(a.x), e1 = float(a.y), e2 = float(a.z), e3 = float(a.w);
            const float e4 = float(b.x), e5 = float(b.y), e6 = float(b.z), e7 = float(b.w);
            sums[token] = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
            elements[token][0] = e0; elements[token][1] = e1 * 0x1p-4f;
            elements[token][2] = e2 * 0x1p-8f; elements[token][3] = e3 * 0x1p-12f;
            elements[token][4] = e4; elements[token][5] = e5 * 0x1p-4f;
            elements[token][6] = e6 * 0x1p-8f; elements[token][7] = e7 * 0x1p-12f;
        }
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            const uint row = firstRow + r;
            if (row >= rows) continue;
            const uint scaleIndex = row * groups + block * 4u + (lane >> 3);
            device const ushort* packed = (device const ushort*)(weights + row * (columns / 2u) + byteBase);
            const uint w0 = uint(packed[0]);
            const uint w1 = uint(packed[1]);
            const float scale = float(scales[scaleIndex]);
            const float bias = float(biases[scaleIndex]);
            #pragma unroll
            for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
                float dot = 0;
                dot = fma(float(w0 & 0x000fu), elements[token][0], dot);
                dot = fma(float(w0 & 0x00f0u), elements[token][1], dot);
                dot = fma(float(w0 & 0x0f00u), elements[token][2], dot);
                dot = fma(float(w0 & 0xf000u), elements[token][3], dot);
                dot = fma(float(w1 & 0x000fu), elements[token][4], dot);
                dot = fma(float(w1 & 0x00f0u), elements[token][5], dot);
                dot = fma(float(w1 & 0x0f00u), elements[token][6], dot);
                dot = fma(float(w1 & 0xf000u), elements[token][7], dot);
                acc[r][token] = fma(scale, dot, acc[r][token]);
                acc[r][token] = fma(bias, sums[token], acc[r][token]);
            }
        }
    }
    for (uint g = blocks * 4u; g < groups; ++g) {
        float a[5], b[5];
        #pragma unroll
        for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
            a[token] = float(inputs[token][g * 64u + lane * 2u]);
            b[token] = float(inputs[token][g * 64u + lane * 2u + 1u]);
        }
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            const uint row = firstRow + r;
            if (row >= rows) continue;
            const uchar packed = weights[row * (columns / 2u) + g * 32u + lane];
            const float scale = float(scales[row * groups + g]);
            const float bias = float(biases[row * groups + g]);
            #pragma unroll
            for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
                float dot = fma(float(uint(packed & 0x0fu)), a[token], 0.0f);
                dot = fma(float(uint(packed >> 4)), b[token], dot);
                acc[r][token] = fma(scale, dot, acc[r][token]);
                acc[r][token] = fma(bias, a[token] + b[token], acc[r][token]);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        const uint row = firstRow + r;
        if (row >= rows) continue;
        #pragma unroll
        for (uint token = 0; token < GEMMA_VERIFY_BATCH; ++token) {
            const float value = simd_sum(acc[r][token]);
            if (lane == 0) outputs[token][row] = half(value);
        }
    }
}
