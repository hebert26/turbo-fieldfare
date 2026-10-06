#include <metal_stdlib>
using namespace metal;

// Each token uses one threadgroup and the original decode math.

[[kernel, max_total_threads_per_threadgroup(256)]]
void gemma_verify_norm(
    device half* x0 [[buffer(0)]],
    device half* x1 [[buffer(1)]],
    device half* x2 [[buffer(2)]],
    device half* x3 [[buffer(3)]],
    device half* x4 [[buffer(4)]],
    device half* out0 [[buffer(5)]],
    device half* out1 [[buffer(6)]],
    device half* out2 [[buffer(7)]],
    device half* out3 [[buffer(8)]],
    device half* out4 [[buffer(9)]],
    device const bfloat* weight [[buffer(10)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    const uint lid = local.x;
    const uint lsize = size.x;
    device half* xRows[5] = {x0, x1, x2, x3, x4};
    device half* x = xRows[group.y];
    device half* outRows[5] = {out0, out1, out2, out3, out4};
    device half* out = outRows[group.y];
    threadgroup float partial[8];
    const float inv = rms_block_inv(x, 2816u, 1e-6f, lid, lsize,
        simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < 2816u; i += lsize) {
        out[i] = half(float(x[i]) * inv * float(weight[i]));
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void gemma_verify_fused_post_attn_setup(
    device half* hidden0 [[buffer(0)]],
    device half* hidden1 [[buffer(1)]],
    device half* hidden2 [[buffer(2)]],
    device half* hidden3 [[buffer(3)]],
    device half* hidden_4 [[buffer(4)]],
    device half* attn0 [[buffer(5)]],
    device half* attn1 [[buffer(6)]],
    device half* attn2 [[buffer(7)]],
    device half* attn3 [[buffer(8)]],
    device half* attn4 [[buffer(9)]],
    device half* dense_x0 [[buffer(10)]],
    device half* dense_x1 [[buffer(11)]],
    device half* dense_x2 [[buffer(12)]],
    device half* dense_x3 [[buffer(13)]],
    device half* dense_x4 [[buffer(14)]],
    device half* routed_x0 [[buffer(15)]],
    device half* routed_x1 [[buffer(16)]],
    device half* routed_x2 [[buffer(17)]],
    device half* routed_x3 [[buffer(18)]],
    device half* routed_x4 [[buffer(19)]],
    device half* router_x0 [[buffer(20)]],
    device half* router_x1 [[buffer(21)]],
    device half* router_x2 [[buffer(22)]],
    device half* router_x3 [[buffer(23)]],
    device half* router_x4 [[buffer(24)]],
    device const bfloat* w_post_attn [[buffer(25)]],
    device const bfloat* w_pre_ffn [[buffer(26)]],
    device const bfloat* w_pre_ffn2 [[buffer(27)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    const uint lid = local.x;
    const uint lsize = size.x;
    device half* hiddenRows[5] = {hidden0, hidden1, hidden2, hidden3, hidden_4};
    device half* hidden = hiddenRows[group.y];
    device half* attnRows[5] = {attn0, attn1, attn2, attn3, attn4};
    device half* attn = attnRows[group.y];
    device half* dense_xRows[5] = {dense_x0, dense_x1, dense_x2, dense_x3, dense_x4};
    device half* dense_x = dense_xRows[group.y];
    device half* routed_xRows[5] = {routed_x0, routed_x1, routed_x2, routed_x3, routed_x4};
    device half* routed_x = routed_xRows[group.y];
    device half* router_xRows[5] = {router_x0, router_x1, router_x2, router_x3, router_x4};
    device half* router_x = router_xRows[group.y];
    threadgroup half  attn_norm_tg[kFusedMaxD];
    threadgroup half  hidden_tg[kFusedMaxD];
    threadgroup float partial[kFusedMaxSimdGroups];
    const uint DD = 2816u;

    float acc = 0.0f;
    for (uint i = lid; i < DD; i += lsize) {
        float v = float(attn[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(sum / float(DD) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float attn_inv = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        attn_norm_tg[i] = half(float(attn[i]) * attn_inv * float(w_post_attn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    acc = 0.0f;
    for (uint i = lid; i < DD; i += lsize) {
        half h = half(float(hidden[i]) + float(attn_norm_tg[i]));
        hidden_tg[i] = h;
        hidden[i] = h;
        float hf = float(h);
        acc = fma(hf, hf, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(sum / float(DD) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float hidden_inv = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        const float h = float(hidden_tg[i]) * hidden_inv;
        dense_x[i] = half(h * float(w_pre_ffn[i]));
        routed_x[i] = half(h * float(w_pre_ffn2[i]));
        router_x[i] = half(h);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void gemma_verify_fused_layer_tail(
    device half* h20 [[buffer(0)]],
    device half* h21 [[buffer(1)]],
    device half* h22 [[buffer(2)]],
    device half* h23 [[buffer(3)]],
    device half* h24 [[buffer(4)]],
    device half* h10 [[buffer(5)]],
    device half* h11 [[buffer(6)]],
    device half* h12 [[buffer(7)]],
    device half* h13 [[buffer(8)]],
    device half* h14 [[buffer(9)]],
    device half* hidden0 [[buffer(10)]],
    device half* hidden1 [[buffer(11)]],
    device half* hidden2 [[buffer(12)]],
    device half* hidden3 [[buffer(13)]],
    device half* hidden_4 [[buffer(14)]],
    device const bfloat* w_postffn2 [[buffer(15)]],
    device const bfloat* w_postffn [[buffer(16)]],
    constant float& layer_scalar [[buffer(17)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    const uint lid = local.x;
    const uint lsize = size.x;
    device half* h2Rows[5] = {h20, h21, h22, h23, h24};
    device half* h2 = h2Rows[group.y];
    device half* h1Rows[5] = {h10, h11, h12, h13, h14};
    device half* h1 = h1Rows[group.y];
    device half* hiddenRows[5] = {hidden0, hidden1, hidden2, hidden3, hidden_4};
    device half* hidden = hiddenRows[group.y];
    threadgroup half  tmp_tg[kFusedMaxD];
    threadgroup half  h12_tg[kFusedMaxD];
    threadgroup float partial[kFusedMaxSimdGroups];
    const uint DD = 2816u;

    float acc = 0.0f;
    for (uint i = lid; i < DD; i += lsize) {
        float v = float(h2[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(DD);
            partial[0] = rsqrt(mean_sq + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_h2 = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        tmp_tg[i] = half(float(h2[i]) * inv_h2 * float(w_postffn2[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint D4 = DD / 4u;
    threadgroup half4* h12_4 = reinterpret_cast<threadgroup half4*>(h12_tg);
    threadgroup half4* tmp_4 = reinterpret_cast<threadgroup half4*>(tmp_tg);
    device const half4* h1_4 = reinterpret_cast<device const half4*>(h1);
    for (uint i = lid; i < D4; i += lsize) {
        h12_4[i] = h1_4[i] + tmp_4[i];
    }
    const uint tailStart = D4 * 4u;
    for (uint i = tailStart + lid; i < DD; i += lsize) {
        h12_tg[i] = h1[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    acc = 0.0f;
    for (uint i = lid; i < DD; i += lsize) {
        float v = float(h12_tg[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(DD);
            partial[0] = rsqrt(mean_sq + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_h12 = partial[0];
    for (uint i = lid; i < DD; i += lsize) {
        tmp_tg[i] = half(float(h12_tg[i]) * inv_h12 * float(w_postffn[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half4* hidden4 = reinterpret_cast<device half4*>(hidden);
    for (uint i = lid; i < D4; i += lsize) {
        hidden4[i] = hidden4[i] + tmp_4[i];
    }
    for (uint i = tailStart + lid; i < DD; i += lsize) {
        hidden[i] = hidden[i] + tmp_tg[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    const half hScale = half(layer_scalar);
    for (uint i = lid; i < D4; i += lsize) {
        hidden4[i] = hidden4[i] * hScale;
    }
    for (uint i = tailStart + lid; i < DD; i += lsize) {
        hidden[i] = hidden[i] * hScale;
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void gemma_verify_epilogue(
    device half* q0 [[buffer(0)]],
    device half* q1 [[buffer(1)]],
    device half* q2 [[buffer(2)]],
    device half* q3 [[buffer(3)]],
    device half* q4 [[buffer(4)]],
    device half* k0 [[buffer(5)]],
    device half* k1 [[buffer(6)]],
    device half* k2 [[buffer(7)]],
    device half* k3 [[buffer(8)]],
    device half* k4 [[buffer(9)]],
    device half* v0 [[buffer(10)]],
    device half* v1 [[buffer(11)]],
    device half* v2 [[buffer(12)]],
    device half* v3 [[buffer(13)]],
    device half* v4 [[buffer(14)]],
    device const bfloat* q_weight [[buffer(15)]],
    device const bfloat* k_weight [[buffer(16)]],
    constant uint& head_dim [[buffer(17)]],
    constant uint& num_q_heads [[buffer(18)]],
    constant uint& num_kv_heads [[buffer(19)]],
    constant uint& position [[buffer(20)]],
    constant float& theta_base [[buffer(21)]],
    constant uint& rotated_pairs [[buffer(22)]],
    constant float& rms_eps [[buffer(23)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    const uint lid = local.x;
    const uint lsize = size.x;
    device half* qRows[5] = {q0, q1, q2, q3, q4};
    device half* q = qRows[group.y];
    device half* kRows[5] = {k0, k1, k2, k3, k4};
    device half* k = kRows[group.y];
    device half* vRows[5] = {v0, v1, v2, v3, v4};
    device half* v = vRows[group.y];
    threadgroup half  head_tg[kFusedMaxHeadDim];
    threadgroup float partial[kFusedMaxSimdGroups];

    const uint HD = fused_fc_head_dim(head_dim);
    const uint NQ = fused_fc_num_q_heads(num_q_heads);
    const uint NKV = fused_fc_num_kv_heads(num_kv_heads);
    const uint RP = fused_fc_rotary(rotated_pairs);

    const bool is_q = group.x < NQ;
    const bool is_k = !is_q && group.x < (NQ + NKV);
    const bool is_v = !is_q && !is_k && group.x < (NQ + 2u * NKV);
    if (!is_q && !is_k && !is_v) return;

    const uint local_head = is_q ? group.x : (group.x - NQ) % NKV;
    device half* dst = is_q ? (q + local_head * HD)
                    : (is_k ? (k + local_head * HD)
                            : (v + local_head * HD));
    device const half* src = dst;
    device const bfloat* w = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(src[i]);
        acc = fma(xv, xv, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(sum / float(HD) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = partial[0];
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(src[i]) * inv;
        if (!is_v) {
            xv *= float(w[i]);
        }
        head_tg[i] = half(xv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_v) {
        for (uint i = lid; i < HD; i += lsize) {
            dst[i] = head_tg[i];
        }
        return;
    }

    const uint half_dim = HD / 2u;
    for (uint pair = lid; pair < half_dim; pair += lsize) {
        float x0 = float(head_tg[pair]);
        float x1 = float(head_tg[half_dim + pair]);
        if (pair < RP) {
            fused_rope_neox_pair(x0, x1, pair, HD, float(position + group.y), theta_base);
        }
        dst[pair] = half(x0);
        dst[half_dim + pair] = half(x1);
    }
}

kernel void gemma_verify_gelu(
    device half* gate0 [[buffer(0)]],
    device half* gate1 [[buffer(1)]],
    device half* gate2 [[buffer(2)]],
    device half* gate3 [[buffer(3)]],
    device half* gate4 [[buffer(4)]],
    device half* up0 [[buffer(5)]],
    device half* up1 [[buffer(6)]],
    device half* up2 [[buffer(7)]],
    device half* up3 [[buffer(8)]],
    device half* up4 [[buffer(9)]],
    device half* out0 [[buffer(10)]],
    device half* out1 [[buffer(11)]],
    device half* out2 [[buffer(12)]],
    device half* out3 [[buffer(13)]],
    device half* out4 [[buffer(14)]],
    constant uint& count [[buffer(15)]],
    uint2 group [[thread_position_in_grid]]
) {
    device half* gateRows[5] = {gate0, gate1, gate2, gate3, gate4};
    device half* gate = gateRows[group.y];
    device half* upRows[5] = {up0, up1, up2, up3, up4};
    device half* up = upRows[group.y];
    device half* outRows[5] = {out0, out1, out2, out3, out4};
    device half* out = outRows[group.y];
    if (group.x >= count) return;
    out[group.x] = half(gelu_pytorch_tanh(float(gate[group.x])) * float(up[group.x]));
}
