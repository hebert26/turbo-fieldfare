#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMoEGroupSize = 64;
constant constexpr uint kMaxStreamedExperts = 8;
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant uint FC_ROUTER_TOP_K [[function_constant(42)]];
constant bool FC_ROUTER_USE_FC [[function_constant(43)]];

constant uint FC_MOE_D [[function_constant(0)]];
constant uint FC_MOE_F [[function_constant(1)]];
constant uint FC_MOE_TOP_K [[function_constant(2)]];
constant bool FC_MOE_USE_FC [[function_constant(3)]];

static inline uint router_fc_num_experts(constant uint& num_experts) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_NUM_EXPERTS))
        ? FC_ROUTER_NUM_EXPERTS
        : num_experts;
}

static inline uint router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_D))
        ? FC_ROUTER_D
        : D;
}

static inline uint moe_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_D)) ? FC_MOE_D : D;
}

static inline uint moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_F)) ? FC_MOE_F : F;
}

static inline uint moe_fc_top_k(constant uint& top_k) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_TOP_K)) ? FC_MOE_TOP_K : top_k;
}

static inline float gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    // Clamping avoids Metal tanh producing NaN at large magnitudes while being
    // equivalent to the saturated result at FP32 precision.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

struct ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct RoutedBlobs {
    device const uint8_t* blob[kMaxStreamedExperts];
};

struct CachedExpertBlobs {
    device const uint8_t* blob[128];
};

// Three packed uints match MTLDispatchThreadgroupsIndirectArguments.
struct CachedDispatchSize {
    uint x;
    uint y;
    uint z;
};

static_assert(sizeof(CachedDispatchSize) == 12, "Indirect dispatch layout changed");
static_assert(sizeof(RoutedBlobs) == 64, "Selected expert pointer layout changed");
static_assert(sizeof(CachedExpertBlobs) == 1024, "Cached expert pointer layout changed");

static inline void moe_cache_resolve(
    device const CachedExpertBlobs& cache,
    device const uint* slot_by_expert,
    device const uint* indices,
    device RoutedBlobs& selected,
    device uint* routes,
    device uint& stopped,
    device CachedDispatchSize* dispatches,
    uint first_routed_dispatch,
    uint dispatch_limit,
    uint layer,
    uint slot_count) {
    uint slots[kMaxStreamedExperts];
    bool all_hit = true;
    for (uint i = 0; i < kMaxStreamedExperts; ++i) {
        uint expert = min(indices[i], 127u);
        routes[i] = expert;
        slots[i] = slot_by_expert[expert];
        all_hit = all_hit && slots[i] < slot_count;
    }
    if (!all_hit) {
        stopped = layer;
        // Disable this layer's tail and every later layer in the batch.
        for (uint i = first_routed_dispatch; i < dispatch_limit; ++i) {
            dispatches[i] = CachedDispatchSize{0, 0, 0};
        }
        return;
    }
    for (uint i = 0; i < kMaxStreamedExperts; ++i) {
        selected.blob[i] = cache.blob[slots[i]];
    }
}

static inline void router_gemv_gemma4_body(
    device const uint8_t* W,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    const uint n_groups = DD / kMoEGroupSize;
    device const uint8_t* W_row = W + uint(e) * DD;
    device const bfloat* s_row = scales + uint(e) * n_groups;
    device const bfloat* b_row = biases + uint(e) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint idx = g * kMoEGroupSize + lane * 2u;
        const float q0 = float(uint(W_row[idx]));
        const float q1 = float(uint(W_row[idx + 1u]));
        const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
        const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

kernel void router_gemv_gemma4_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_gemma4_body(W, scales, biases, hidden, effective_scale,
                            out_logits, num_experts, D, 4, tg_idx, sg_idx, lane);
}

static inline void router_topk_select_k8_body(
    device const float* logits,
    device const bfloat* per_expert_scale,
    device uint* out_indices,
    device half* out_weights,
    constant uint& num_experts
) {
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[8];
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        if (s <= top_score[7]) continue;
        uint pos = 8u;
        for (uint i = 0; i < 8; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 8u) continue;
        for (uint i = 7; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

kernel void router_topk_select_k8(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0) return;
    router_topk_select_k8_body(logits, per_expert_scale, out_indices, out_weights, num_experts);
}

kernel void router_topk_select_k8_cached(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    device const CachedExpertBlobs& cache [[buffer(5)]],
    device const uint* slot_by_expert [[buffer(6)]],
    device RoutedBlobs& selected [[buffer(7)]],
    device uint* routes [[buffer(8)]],
    device uint& stopped [[buffer(9)]],
    device CachedDispatchSize* dispatches [[buffer(10)]],
    constant uint& first_routed_dispatch [[buffer(11)]],
    constant uint& dispatch_limit [[buffer(12)]],
    constant uint& layer [[buffer(13)]],
    constant uint& slot_count [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0 || stopped != 0xffffffffu) return;
    router_topk_select_k8_body(logits, per_expert_scale, out_indices, out_weights, num_experts);
    moe_cache_resolve(cache, slot_by_expert, out_indices, selected, routes, stopped,
                      dispatches, first_routed_dispatch, dispatch_limit, layer, slot_count);
}

// Each SIMD computes one affine INT4 row. Four adjacent groups are loaded as
// aligned 32-bit chunks; remaining groups use one byte per lane.
static inline float moe_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// Gate and up rows share input loads and exact power-of-two scales.
// Read weights in 16-bit words to support two-byte offsets.
static inline float2 moe_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw0 = uint(gp[0]), gw1 = uint(gp[1]);
        const uint uw0 = uint(up[0]), uw1 = uint(up[1]);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        const float x1 = e1 * 0x1p-4f, x2 = e2 * 0x1p-8f, x3 = e3 * 0x1p-12f;
        const float x5 = e5 * 0x1p-4f, x6 = e6 * 0x1p-8f, x7 = e7 * 0x1p-12f;

        float g_dot = 0.0f;
        g_dot = fma(float(gw0 & 0x000Fu), e0, g_dot); g_dot = fma(float(gw0 & 0x00F0u), x1, g_dot);
        g_dot = fma(float(gw0 & 0x0F00u), x2, g_dot); g_dot = fma(float(gw0 & 0xF000u), x3, g_dot);
        g_dot = fma(float(gw1 & 0x000Fu), e4, g_dot); g_dot = fma(float(gw1 & 0x00F0u), x5, g_dot);
        g_dot = fma(float(gw1 & 0x0F00u), x6, g_dot); g_dot = fma(float(gw1 & 0xF000u), x7, g_dot);

        float u_dot = 0.0f;
        u_dot = fma(float(uw0 & 0x000Fu), e0, u_dot); u_dot = fma(float(uw0 & 0x00F0u), x1, u_dot);
        u_dot = fma(float(uw0 & 0x0F00u), x2, u_dot); u_dot = fma(float(uw0 & 0xF000u), x3, u_dot);
        u_dot = fma(float(uw1 & 0x000Fu), e4, u_dot); u_dot = fma(float(uw1 & 0x00F0u), x5, u_dot);
        u_dot = fma(float(uw1 & 0x0F00u), x6, u_dot); u_dot = fma(float(uw1 & 0xF000u), x7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

static inline void moe_phase1_gate_up_act_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

static inline void moe_phase1_gate_up_act_subset_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    device const uint* active_slots,
    uint active_count,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint active_idx = rowg / F;
    const uint slot = active_slots[active_idx];
    if (slot >= top_k) return;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(gelu_pytorch_tanh(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void moe_phase1_gate_up_act_subset_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_subset_u16load_body(
        routed, routed_offsets, x, acts, active_slots, active_count,
        moe_fc_d(D), moe_fc_f(F), moe_fc_top_k(top_k), rows_per_tg,
        tg_idx, sg_idx, lane);
}

kernel void moe_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}
