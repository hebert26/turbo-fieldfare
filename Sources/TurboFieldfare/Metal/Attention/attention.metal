#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention — split-KV tiled softmax attention for single-token decode.
//
// Decode path only: M_q = 1 (one query token), arbitrary seq_len history.
// The MPP prefill path handles M_q > 1 separately.
//
// Layout (caller-side contract):
//   Q   : [num_q_heads,  head_dim]                      FP16, contiguous.
//   K   : [seq_len, num_kv_heads, head_dim]             FP16, contiguous.
//   V   : [seq_len, num_kv_heads, head_dim]             FP16, same shape as K.
//         Full attention reuses the raw K projection for V, but its separate
//         normalization and RoPE paths make these buffers distinct here.
//   out : [num_q_heads,  head_dim]                      FP16.
//
// GQA: q_head -> kv_head = q_head / (num_q_heads / num_kv_heads).
//      Multiple Q heads share one KV head; the dispatch indexes Q heads.
//
// Online softmax recurrence (FP32 accumulators) — Milakov & Gimelshein 2018,
// also FlashAttention:
//   m_new   = max(m, s)
//   alpha   = exp(m - m_new)                 // rescale factor for past state
//   d       = d * alpha + exp(s - m_new)
//   o[i]    = o[i] * alpha + exp(s - m_new) * V[p, i]
//   m       = m_new
// Final normalization: out[i] = o[i] / d.
//
// ============================================================================

constant constexpr uint kAttnThreads      = 256;
// kAttnMaxSimdGroups must cover kAttnThreads / 32 = 8.
constant constexpr uint kAttnMaxSimdGroups = 8;
constant constexpr uint kAttnScoreTile = 4;
constant constexpr uint kAttnMaxQPerKV     = 2;
constant constexpr uint kAttnMaxFullQPerKV = 8;
constant constexpr uint kAttnFullQPerThreadgroup = 2;
// Largest head_dim we run with (full-attention layers). SWA uses 256 — the
// kernel still allocates the 512-slot scratch but only touches the live half.
constant constexpr uint kAttnMaxHeadDim   = 512;
constant uint FC_ATTN_HEAD_DIM [[function_constant(60)]];
constant uint FC_ATTN_NUM_Q_HEADS [[function_constant(61)]];
constant uint FC_ATTN_NUM_KV_HEADS [[function_constant(62)]];
constant bool FC_ATTN_USE_FC [[function_constant(63)]];
constant float FC_ATTN_SCALE [[function_constant(64)]];
constant uint FC_ATTN_NUM_CHUNKS [[function_constant(65)]];
constant bool FC_ATTN_RETAIN_QUERY [[function_constant(66)]];
constant bool FC_ATTN_PREFETCH_VALUE [[function_constant(67)]];
constant bool FC_ATTN_PAIR [[function_constant(68)]];
constant uint FC_ATTN_RING_CAP [[function_constant(69)]];

static inline uint attn_fc_head_dim(constant uint& head_dim) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_HEAD_DIM))
        ? FC_ATTN_HEAD_DIM
        : head_dim;
}
static inline uint attn_fc_num_q_heads(constant uint& num_q_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_Q_HEADS))
        ? FC_ATTN_NUM_Q_HEADS
        : num_q_heads;
}

static inline uint attn_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_KV_HEADS))
        ? FC_ATTN_NUM_KV_HEADS
        : num_kv_heads;
}

static inline float attn_fc_scale(float scale) {
    return is_function_constant_defined(FC_ATTN_SCALE) ? FC_ATTN_SCALE : scale;
}

static inline uint attn_fc_num_chunks(constant uint& num_chunks) {
    return is_function_constant_defined(FC_ATTN_NUM_CHUNKS) ? FC_ATTN_NUM_CHUNKS : num_chunks;
}

static inline uint attn_ring_slot(uint p) {
    return (is_function_constant_defined(FC_ATTN_RING_CAP) &&
            FC_ATTN_RING_CAP != 0u)
        ? (p % FC_ATTN_RING_CAP)
        : p;
}

static inline float attn_softmax_exp(float x) {
    return fast::exp(x);
}

// Merge up to four scores with two group waits. Keep each score's sum order.
// All threads must finish reading bcast before the next call's first wait.
inline void attention_reduce_scores(uint count,
                                    uint simd_lane_id,
                                    uint simd_group_id,
                                    uint simdgroups,
                                    threadgroup float* scratch,
                                    threadgroup float* bcast) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint j = 0; j < count; ++j) {
            float t = (simd_lane_id < simdgroups)
                ? scratch[j * kAttnMaxSimdGroups + simd_lane_id] : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[j] = t; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


// ============================================================================
// Split-KV (Flash-Decoding) decode attention — the default path.
//

// Pass 1 (attention_decode_partial): grid = num_q_heads * num_chunks. Each TG
//   runs the same online-softmax recurrence over its chunk [p_start, p_end) and
//   writes the UN-normalized partial state (m_chunk, d_chunk, o_chunk[head_dim])
//   to scratch — no division yet.
// Pass 2 (attention_decode_combine): grid = num_q_heads. Each TG merges its
//   head's num_chunks partials with the standard online-softmax rescale
//   (m_glob = max_c m_c; D = Σ d_c·e^{m_c−m_glob}; O = Σ o_c·e^{m_c−m_glob}) and
//   writes out[i] = O[i] / D in FP16.
//
// At num_chunks == 1 the chunk spans the whole [kv_start, seq_len) range and
// the partial is the exact single-pass accumulation; the combine's only chunk
// has m_glob == m_chunk so e^0 == 1 and out == o/d — byte-identical to the
// single-pass kernels above. num_chunks > 1 changes the FP rounding of the
// partial sums only (same position summation order), not the algorithm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial(
    device const half*  Q0            [[buffer(0)]],
    device const half*  Q1            [[buffer(14)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint*  seq_lengths   [[buffer(9)]],
    constant     uint*  kv_starts     [[buffer(10)]],
    constant     uint*  chunk_lengths [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint2 group_id       [[threadgroup_position_in_grid]],
    uint2 local_id       [[thread_position_in_threadgroup]],
    uint2 local_size     [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnScoreTile * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnScoreTile];
    const uint lid = local_id.x;
    const uint lsize = local_size.x;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint row = (is_function_constant_defined(FC_ATTN_PAIR) && FC_ATTN_PAIR)
        ? group_id.y : 0u;
    const uint tg_id = group_id.x;
    device const half* Q = row == 0u ? Q0 : Q1;
    const uint seq_len = seq_lengths[row];
    const uint kv_start = kv_starts[row];
    const uint chunk_len = chunk_lengths[row];

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Only the exact full-head / 256-thread variant retains two fixed values.
    // Other shapes and widths keep the original strided query reads below.
    const bool retain_query = is_function_constant_defined(FC_ATTN_RETAIN_QUERY) &&
        FC_ATTN_RETAIN_QUERY && HD == 512u && lsize == kAttnThreads;
    const bool prefetch_value =
        is_function_constant_defined(FC_ATTN_PREFETCH_VALUE) &&
        FC_ATTN_PREFETCH_VALUE && retain_query;
    const float q_first = retain_query ? q_smem[lid] : 0.0f;
    const float q_second = retain_query ? q_smem[lid + kAttnThreads] : 0.0f;

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // p_start can land past the end when num_chunks > range length (the tail
    // chunks are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint tile_start = p_start; tile_start < p_end; tile_start += kAttnScoreTile) {
        const uint count = min(kAttnScoreTile, p_end - tile_start);
        float v_first[kAttnScoreTile];
        float v_second[kAttnScoreTile];
        for (uint j = 0; j < count; ++j) {
            const uint phys_p = attn_ring_slot(tile_start + j);
            device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
            if (prefetch_value) {
                device const half* V_row = V + (phys_p * NKV + kv_head) * HD;
                v_first[j] = float(V_row[lid]);
                v_second[j] = float(V_row[lid + kAttnThreads]);
            }
            float partial = 0.0f;
            if (retain_query) {
                partial = fma(q_first, float(K_row[lid]), partial);
                partial = fma(q_second, float(K_row[lid + kAttnThreads]), partial);
            } else {
                for (uint i = lid; i < HD; i += lsize) {
                    partial = fma(q_smem[i], float(K_row[i]), partial);
                }
            }
            const float s = simd_sum(partial);
            if (simd_lane_id == 0) {
                reduce_scratch[j * kAttnMaxSimdGroups + simd_group_id] = s;
            }
        }
        attention_reduce_scores(count, simd_lane_id, simd_group_id, simdgroups,
                                reduce_scratch, bcast);

        // Apply each score in the original position order.
        for (uint j = 0; j < count; ++j) {
            const float s = bcast[j] * attn_fc_scale(scale);
            const float m_new = max(m_run, s);
            const float alpha = attn_softmax_exp(m_run - m_new);
            const float p_exp = attn_softmax_exp(s - m_new);
            d_run = d_run * alpha + p_exp;
            if (prefetch_value) {
                o_local[0] = o_local[0] * alpha + p_exp * v_first[j];
                o_local[1] = o_local[1] * alpha + p_exp * v_second[j];
            } else {
                const uint phys_p = attn_ring_slot(tile_start + j);
                device const half* V_row = V + (phys_p * NKV + kv_head) * HD;
                uint slot = 0;
                for (uint i = lid; i < HD; i += lsize) {
                    o_local[slot] = o_local[slot] * alpha + p_exp * float(V_row[i]);
                    slot += 1;
                }
            }
            m_run = m_new;
        }
    }

    const uint base = (row * NQ + uint(q_head)) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_gqa_swa_partial(
    device const half*  Q0            [[buffer(0)]],
    device const half*  Q1            [[buffer(14)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint*  seq_lengths   [[buffer(9)]],
    constant     uint*  kv_starts     [[buffer(10)]],
    constant     uint*  chunk_lengths [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint2 group_id       [[threadgroup_position_in_grid]],
    uint2 local_id       [[thread_position_in_threadgroup]],
    uint2 local_size     [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnScoreTile * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV * kAttnScoreTile];
    const uint lid = local_id.x;
    const uint lsize = local_size.x;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint row = (is_function_constant_defined(FC_ATTN_PAIR) && FC_ATTN_PAIR)
        ? group_id.y : 0u;
    const uint tg_id = group_id.x;
    device const half* Q = row == 0u ? Q0 : Q1;
    const uint seq_len = seq_lengths[row];
    const uint kv_start = kv_starts[row];
    const uint chunk_len = chunk_lengths[row];

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_base = kv_head * q_per_kv;
    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            Q_s[i] = float(Q_row[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;
    threadgroup float* score_scratch =
        reduce_scratch + active_q * kAttnScoreTile * kAttnMaxSimdGroups;
    threadgroup float* score_bcast = bcast + active_q * kAttnScoreTile;

    const bool retain_query = is_function_constant_defined(FC_ATTN_RETAIN_QUERY) &&
        FC_ATTN_RETAIN_QUERY && HD == 256u && threads_per_q == 128u;
    const bool prefetch_value = is_function_constant_defined(FC_ATTN_PREFETCH_VALUE) &&
        FC_ATTN_PREFETCH_VALUE && retain_query;
    const uint query_base = active_q * kAttnMaxHeadDim + local_lid;
    const float q_first = retain_query ? q_smem[query_base] : 0.0f;
    const float q_second = retain_query ? q_smem[query_base + 128u] : 0.0f;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint tile_start = p_start; tile_start < p_end; tile_start += kAttnScoreTile) {
        const uint count = min(kAttnScoreTile, p_end - tile_start);
        float v_first[kAttnScoreTile];
        float v_second[kAttnScoreTile];
        for (uint j = 0; j < count; ++j) {
            const uint phys_p = attn_ring_slot(tile_start + j);
            device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
            if (prefetch_value) {
                device const half* V_row = V + (phys_p * NKV + kv_head) * HD;
                v_first[j] = float(V_row[local_lid]);
                v_second[j] = float(V_row[local_lid + 128u]);
            }
            float partial = 0.0f;
            if (retain_query) {
                partial = fma(q_first, float(K_row[local_lid]), partial);
                partial = fma(q_second, float(K_row[local_lid + 128u]), partial);
            } else {
                for (uint i = local_lid; i < HD; i += threads_per_q) {
                    const float k_val = float(K_row[i]);
                    partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
                }
            }
            const float s = simd_sum(partial);
            if (simd_lane_id == 0) {
                score_scratch[j * kAttnMaxSimdGroups + local_group] = s;
            }
        }
        attention_reduce_scores(count, simd_lane_id, local_group, groups_per_q,
                                score_scratch, score_bcast);

        for (uint j = 0; j < count; ++j) {
            const float s = score_bcast[j] * attn_fc_scale(scale);
            const float m_new = max(m_run, s);
            const float alpha = attn_softmax_exp(m_run - m_new);
            const float p_exp = attn_softmax_exp(s - m_new);
            d_run = d_run * alpha + p_exp;
            for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
            m_run = m_new;

            if (prefetch_value) {
                o_local[0] += p_exp * v_first[j];
                o_local[1] += p_exp * v_second[j];
            } else {
                const uint phys_p = attn_ring_slot(tile_start + j);
                device const half* V_row = V + (phys_p * NKV + kv_head) * HD;
                uint slot = 0;
                for (uint i = local_lid; i < HD; i += threads_per_q) {
                    o_local[slot] += p_exp * float(V_row[i]);
                    slot += 1;
                }
            }
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = (row * NQ + uint(q_head)) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * head_dim]
    device       half*  out0         [[buffer(3)]],
    device       half*  out1         [[buffer(6)]],    // [num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    uint2 group_id       [[threadgroup_position_in_grid]],
    uint2 local_id       [[thread_position_in_threadgroup]],
    uint2 local_size     [[threads_per_threadgroup]]
) {
    const uint lid = local_id.x;
    const uint lsize = local_size.x;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint row = (is_function_constant_defined(FC_ATTN_PAIR) && FC_ATTN_PAIR)
        ? group_id.y : 0u;
    const uint q_head = group_id.x;
    const uint row_heads = is_function_constant_defined(FC_ATTN_NUM_Q_HEADS)
        ? FC_ATTN_NUM_Q_HEADS : 0u;
    const uint input_head = row * row_heads + q_head;
    device half* out = row == 0u ? out0 : out1;
    device const float* m_row  = m_in + input_head * NC;
    device const float* d_row  = d_in + input_head * NC;
    device const float* o_base = o_in + input_head * NC * HD;

    // num_chunks is small (<= a few dozen); each thread recomputes the global
    // max and denominator rather than pay a threadgroup reduction + barriers.
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}

// Reuse each key and value for two query heads. Keep each head's sums.
[[kernel, max_total_threads_per_threadgroup(256)]]
void attention_decode_query_pair_partial(
    device const half* Q0 [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device float* m_out [[buffer(3)]],
    device float* d_out [[buffer(4)]],
    device float* o_out [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& num_q_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint* lengths [[buffer(9)]],
    constant uint* starts [[buffer(10)]],
    constant uint* chunks [[buffer(11)]],
    constant uint& num_chunks [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    device const half* Q1 [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    threadgroup float2 scratch[4 * 8];
    threadgroup float2 scores[4];
    const uint hd = attn_fc_head_dim(head_dim);
    const uint nq = attn_fc_num_q_heads(num_q_heads);
    const uint nk = attn_fc_num_kv_heads(num_kv_heads);
    const uint nc = attn_fc_num_chunks(num_chunks);
    const uint token = (is_function_constant_defined(FC_ATTN_PAIR) && FC_ATTN_PAIR) ? group.y : 0;
    const uint head = (group.x / nc) * 2;
    const uint chunk = group.x % nc;
    const uint kv_head = head / (nq / nk);
    const uint begin = starts[token] + chunk * chunks[token];
    const uint end = min(begin + chunks[token], lengths[token]);
    const uint col = local.x;
    const uint stride = size.x;
    device const half* q = token == 0 ? Q0 : Q1;
    const float2 q_first = float2(float(q[head * hd + col]), float(q[(head + 1) * hd + col]));
    const float2 q_second = float2(float(q[head * hd + col + stride]), float(q[(head + 1) * hd + col + stride]));
    float2 maximum(-INFINITY), denominator(0), first(0), second(0);
    for (uint tile = begin; tile < end; tile += 4) {
        const uint count = min(4u, end - tile);
        float value_first[4], value_second[4];
        for (uint j = 0; j < count; ++j) {
            const uint position = attn_ring_slot(tile + j);
            const uint base = (position * nk + kv_head) * hd;
            value_first[j] = float(V[base + col]);
            value_second[j] = float(V[base + col + stride]);
            float2 partial(0);
            partial = fma(q_first, float2(float(K[base + col])), partial);
            partial = fma(q_second, float2(float(K[base + col + stride])), partial);
            const float2 total(simd_sum(partial.x), simd_sum(partial.y));
            if (lane == 0) scratch[j * 8 + simd] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            for (uint j = 0; j < count; ++j) {
                const float2 part = lane < simdgroups ? scratch[j * 8 + lane] : float2(0);
                const float2 sum(simd_sum(part.x), simd_sum(part.y));
                if (lane == 0) scores[j] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < count; ++j) {
            const float2 score = scores[j] * attn_fc_scale(scale);
            const float2 next = max(maximum, score);
            const float2 alpha = fast::exp(maximum - next);
            const float2 weight = fast::exp(score - next);
            denominator = denominator * alpha + weight;
            if (hd == 256u) {
                first *= alpha; second *= alpha;
                first += weight * value_first[j]; second += weight * value_second[j];
            } else {
                first = first * alpha + weight * value_first[j];
                second = second * alpha + weight * value_second[j];
            }
            maximum = next;
        }
    }
    const uint base0 = (token * nq + head) * nc + chunk;
    const uint base1 = base0 + nc;
    if (col == 0) {
        m_out[base0] = maximum.x; m_out[base1] = maximum.y;
        d_out[base0] = denominator.x; d_out[base1] = denominator.y;
    }
    o_out[base0 * hd + col] = first.x;
    o_out[base1 * hd + col] = first.y;
    o_out[base0 * hd + col + stride] = second.x;
    o_out[base1 * hd + col + stride] = second.y;
}

// Reuse each key and value for two query heads. Keep each head's sums.
[[kernel, max_total_threads_per_threadgroup(256)]]
void attention_decode_token_pair_partial(
    device const half* Q0 [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device float* m_out [[buffer(3)]],
    device float* d_out [[buffer(4)]],
    device float* o_out [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& num_q_heads [[buffer(7)]],
    constant uint& num_kv_heads [[buffer(8)]],
    constant uint* lengths [[buffer(9)]],
    constant uint* starts [[buffer(10)]],
    constant uint* chunks [[buffer(11)]],
    constant uint& num_chunks [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    device const half* Q1 [[buffer(14)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    threadgroup float4 scratch[4 * 8];
    threadgroup float4 scores[4];
    const uint hd = attn_fc_head_dim(head_dim);
    const uint nq = attn_fc_num_q_heads(num_q_heads);
    const uint nk = attn_fc_num_kv_heads(num_kv_heads);
    const uint nc = attn_fc_num_chunks(num_chunks);
    const uint token = 0;
    const uint head = (group.x / nc) * 2;
    const uint chunk = group.x % nc;
    const uint kv_head = head / (nq / nk);
    const uint begin = starts[token] + chunk * chunks[token];
    const uint end = min(begin + chunks[0], lengths[1]);
    const uint col = local.x;
    const uint stride = size.x;
    const float4 q_first = float4(float(Q0[head * hd + col]), float(Q0[(head + 1) * hd + col]), float(Q1[head * hd + col]), float(Q1[(head + 1) * hd + col]));
    const float4 q_second = float4(float(Q0[head * hd + col + stride]), float(Q0[(head + 1) * hd + col + stride]), float(Q1[head * hd + col + stride]), float(Q1[(head + 1) * hd + col + stride]));
    float4 maximum(-INFINITY), denominator(0), first(0), second(0);
    for (uint tile = begin; tile < end; tile += 4) {
        const uint count = min(4u, end - tile);
        float value_first[4], value_second[4];
        for (uint j = 0; j < count; ++j) {
            const uint position = attn_ring_slot(tile + j);
            const uint base = (position * nk + kv_head) * hd;
            value_first[j] = float(V[base + col]);
            value_second[j] = float(V[base + col + stride]);
            float4 partial(0);
            partial = fma(q_first, float4(float(K[base + col])), partial);
            partial = fma(q_second, float4(float(K[base + col + stride])), partial);
            const float4 total(simd_sum(partial.x), simd_sum(partial.y), simd_sum(partial.z), simd_sum(partial.w));
            if (lane == 0) scratch[j * 8 + simd] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            for (uint j = 0; j < count; ++j) {
                const float4 part = lane < simdgroups ? scratch[j * 8 + lane] : float4(0);
                const float4 sum(simd_sum(part.x), simd_sum(part.y), simd_sum(part.z), simd_sum(part.w));
                if (lane == 0) scores[j] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < count; ++j) {
            const float2 savedMaximum = maximum.xy, savedDenominator = denominator.xy;
            const float2 savedFirst = first.xy, savedSecond = second.xy;
            const float4 score = scores[j] * attn_fc_scale(scale);
            const float4 next = max(maximum, score);
            const float4 alpha = fast::exp(maximum - next);
            const float4 weight = fast::exp(score - next);
            denominator = denominator * alpha + weight;
            if (hd == 256u) {
                first *= alpha; second *= alpha;
                first += weight * value_first[j]; second += weight * value_second[j];
            } else {
                first = first * alpha + weight * value_first[j];
                second = second * alpha + weight * value_second[j];
            }
            maximum = next;
            if (tile + j >= lengths[0]) {
                maximum.xy = savedMaximum; denominator.xy = savedDenominator;
                first.xy = savedFirst; second.xy = savedSecond;
            }
        }
    }
    const uint base0 = (token * nq + head) * nc + chunk;
    const uint base1 = base0 + nc;
    const uint base2 = base0 + nq * nc;
    const uint base3 = base2 + nc;
    if (col == 0) {
        m_out[base0] = maximum.x; m_out[base1] = maximum.y;
        d_out[base0] = denominator.x; d_out[base1] = denominator.y;
        m_out[base2] = maximum.z; m_out[base3] = maximum.w;
        d_out[base2] = denominator.z; d_out[base3] = denominator.w;
    }
    o_out[base0 * hd + col] = first.x;
    o_out[base1 * hd + col] = first.y;
    o_out[base0 * hd + col + stride] = second.x;
    o_out[base1 * hd + col + stride] = second.y;
    o_out[base2 * hd + col] = first.z;
    o_out[base3 * hd + col] = first.w;
    o_out[base2 * hd + col + stride] = second.z;
    o_out[base3 * hd + col + stride] = second.w;
}
