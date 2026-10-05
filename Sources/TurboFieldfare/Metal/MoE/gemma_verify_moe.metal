#include <metal_stdlib>
using namespace metal;

constant uint GEMMA_VERIFY_MOE_BATCH [[function_constant(91)]];

struct GemmaVerifyExperts {
    device const uchar* weights[40];
    device const half* inputs[5];
    device half* acts[5];
    device const half* routing[5];
    device float* partials[5];
    device half* outputs[5];
};

// Keep adjacent work on the same expert so all input tokens share its cache lines.
[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma_verify_expert_gate_up(
    device const GemmaVerifyExperts& args [[buffer(0)]],
    constant ExpertOffsets& offsets [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    constant uint& tokenCount [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group.x * 4u + simd;
    if (row >= 704u) return;
    device const uchar* base = args.weights[group.y];
    const uint count = is_function_constant_defined(GEMMA_VERIFY_MOE_BATCH)
        ? GEMMA_VERIFY_MOE_BATCH : tokenCount;
    #pragma unroll
    for (uint token = 0; token < count; ++token) {
        const uint slot = slots[group.y * 5u + token];
        if (slot == 0xffffffffu) continue;
        const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
            base + offsets.gate_W_off, (device const bfloat*)(base + offsets.gate_s_off),
            (device const bfloat*)(base + offsets.gate_b_off),
            base + offsets.up_W_off, (device const bfloat*)(base + offsets.up_s_off),
            (device const bfloat*)(base + offsets.up_b_off),
            args.inputs[token], row, 2816u, lane);
        if (lane == 0) args.acts[token][slot * 704u + row] = half(gelu_pytorch_tanh(gu.x) * gu.y);
    }
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void gemma_verify_expert_down(
    device const GemmaVerifyExperts& args [[buffer(0)]],
    constant ExpertOffsets& offsets [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    constant uint& tokenCount [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group.x * 4u + simd;
    if (row >= 2816u) return;
    device const uchar* base = args.weights[group.y];
    const uint count = is_function_constant_defined(GEMMA_VERIFY_MOE_BATCH)
        ? GEMMA_VERIFY_MOE_BATCH : tokenCount;
    #pragma unroll
    for (uint token = 0; token < count; ++token) {
        const uint slot = slots[group.y * 5u + token];
        if (slot == 0xffffffffu) continue;
        const float value = moe_int4_gemv_row_simd_dev_vec(
            base + offsets.down_W_off, (device const bfloat*)(base + offsets.down_s_off),
            (device const bfloat*)(base + offsets.down_b_off),
            args.acts[token] + slot * 704u, row, 704u, lane);
        if (lane == 0) args.partials[token][slot * 2816u + row] = float(args.routing[token][slot]) * value;
    }
}

kernel void gemma_verify_expert_reduce(
    device const GemmaVerifyExperts& args [[buffer(0)]],
    uint2 index [[thread_position_in_grid]]
) {
    if (index.x >= 2816u) return;
    device const float* partial = args.partials[index.y] + index.x;
    float sum = 0;
    sum += partial[0u * 2816u]; sum += partial[1u * 2816u];
    sum += partial[2u * 2816u]; sum += partial[3u * 2816u];
    sum += partial[4u * 2816u]; sum += partial[5u * 2816u];
    sum += partial[6u * 2816u]; sum += partial[7u * 2816u];
    args.outputs[index.y][index.x] = half(sum);
}

kernel void gemma_verify_cache_resolve(
    device const CachedExpertBlobs& cache [[buffer(0)]],
    device const uint* slotByExpert [[buffer(1)]],
    device const uint* firstIndices [[buffer(2)]],
    device const uint* secondIndices [[buffer(3)]],
    device RoutedBlobs& firstSelected [[buffer(4)]],
    device RoutedBlobs& secondSelected [[buffer(5)]],
    device uint* routes [[buffer(6)]],
    device uint& stopped [[buffer(7)]],
    device CachedDispatchSize* dispatches [[buffer(8)]],
    constant uint& firstTail [[buffer(9)]],
    constant uint& limit [[buffer(10)]],
    constant uint& layer [[buffer(11)]],
    constant uint& slotCount [[buffer(12)]],
    device GemmaVerifyExperts& grouped [[buffer(13)]],
    device uint* groupedSlots [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0 || stopped != 0xffffffffu) return;
    uint slots[16];
    bool allHit = true;
    for (uint index = 0; index < 16u; ++index) {
        const uint expert = index < 8u ? firstIndices[index] : secondIndices[index - 8u];
        routes[index] = expert;
        slots[index] = expert < 128u ? slotByExpert[expert] : 0xffffffffu;
        allHit = allHit && slots[index] < slotCount;
    }
    if (!allHit) {
        stopped = layer;
        for (uint index = firstTail; index < limit; ++index) dispatches[index] = CachedDispatchSize{0, 0, 0};
        return;
    }
    for (uint index = 0; index < 8u; ++index) {
        firstSelected.blob[index] = cache.blob[slots[index]];
        secondSelected.blob[index] = cache.blob[slots[index + 8u]];
    }
    for (uint index = 0; index < 40u * 5u; ++index) groupedSlots[index] = 0xffffffffu;
    uint unique[16];
    uint count = 0;
    for (uint index = 0; index < 16u; ++index) {
        uint group = 0;
        while (group < count && unique[group] != routes[index]) ++group;
        if (group == count) {
            unique[count] = routes[index];
            grouped.weights[count] = cache.blob[slots[index]];
            ++count;
        }
        groupedSlots[group * 5u + index / 8u] = index % 8u;
    }
    // Four dense dispatches precede the grouped expert work.
    dispatches[firstTail + 4u].y = count;
    dispatches[firstTail + 5u].y = count;

}
