#include <metal_stdlib>
using namespace metal;
// Test-only candidate. Compile separately in the same safe math mode.
// No production selector references this kernel.
struct QwenBF16Small32Parameters {
    uint rows, columns, firstRow, rowsInChunk, tokenCount;
};
kernel void qwen_bf16_project_small32_lanes(
    constant QwenBF16Small32Parameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* weightBits [[buffer(2)]],
    device float* output [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    const uint localRow = group.x, token = group.y;
    // Uniform guard. Caller must dispatch one complete32-lane group per row.
    if (p.rows != 32u || p.columns != 2048u || localRow >= p.rowsInChunk
        || token >= p.tokenCount) return;
    const device float* vector = input + ulong(token) * p.columns;
    const device ushort* row = weightBits + ulong(localRow) * p.columns;
    const bool secondHalfFirst = ((p.firstRow + localRow) & 7u) == 0u;
    float partial = 0.0f;
    if (lane < 16u) {
        for (uint base = 0; base < p.columns; base += 32u) {
            const uint first = base + lane + (secondHalfFirst ? 16u : 0u);
            const uint second = base + lane + (secondHalfFirst ? 0u : 16u);
            partial = fma(as_type<float>(uint(row[first]) << 16), vector[first], partial);
            partial = fma(as_type<float>(uint(row[second]) << 16), vector[second], partial);
        }
    }
    // Every lane executes every shuffle. Only the original serial tree's
    // live destinations add. No SIMD sum, product regrouping or new FMA.
    for (uint distance = 8u; distance != 0u; distance >>= 1u) {
        const float other = simd_shuffle_down(partial, distance);
        if (lane < distance) partial = partial + other;
    }
    if (lane == 0u) output[ulong(token) * p.rows + p.firstRow + localRow] = partial;
}
