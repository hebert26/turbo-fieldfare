#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int4 — MLX `affine` 4-bit dequant.
//
// Layout (per row of length N):
//   W       : N/2 bytes. Low nibble of byte k = component 2k (unsigned 0..15),
//             high nibble = component 2k+1.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[i] = float(nibble[i]) * scale[i/64] + bias[i/64].
//
// Affine factoring for GEMV (sum over a group of 64):
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias each cost one mul + one FMA per group instead of per
// element; the per-element inner loop keeps the scalar path's FMA count.
// ============================================================================

constant constexpr uint kGroupSize = 64;
constant uint FC_INT4_M [[function_constant(20)]];
constant uint FC_INT4_N [[function_constant(21)]];
constant bool FC_INT4_USE_FC [[function_constant(22)]];
constant uint FC_INT4_QKV_MQ [[function_constant(23)]];
constant uint FC_INT4_QKV_MKV [[function_constant(24)]];
constant uint FC_INT4_QKV_N [[function_constant(25)]];
constant bool FC_INT4_QKV_USE_FC [[function_constant(26)]];
constant bool FC_INT4_QKV_K_EQUALS_V [[function_constant(27)]];

static inline uint int4_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_M)) ? FC_INT4_M : M;
}

static inline uint int4_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_N)) ? FC_INT4_N : N;
}

static inline uint int4_qkv_fc_mq(constant uint& Mq) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MQ)) ? FC_INT4_QKV_MQ : Mq;
}

static inline uint int4_qkv_fc_mkv(constant uint& Mkv) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MKV)) ? FC_INT4_QKV_MKV : Mkv;
}

static inline uint int4_qkv_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_N)) ? FC_INT4_QKV_N : N;
}

inline uint nib_lo(uint8_t b) { return uint(b & 0x0F); }
inline uint nib_hi(uint8_t b) { return uint(b >> 4); }


kernel void embed_lookup_int4(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/2] nibbles
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/64] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/64] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kGroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * (D / 2u);
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;
    uint8_t byte = row_q[gid >> 1];
    uint    q    = (gid & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    float   s    = float(row_s[gid / kGroupSize]);
    float   b    = float(row_b[gid / kGroupSize]);
    out[gid] = half((float(q) * s + b) * out_scale);
}

// Each group of 32 threads computes four rows with the same input values.
// Each row keeps its own sum and the original order of calculations.
// Packed weights need only two-byte alignment.
static inline void dequant_int4_gemv_simd_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    device half*          duplicate_y,
    bool                  duplicate_output,
    uint                  M,
    uint                  N,
    uint                  first_row,
    uint                  lane
) {
    if (first_row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    const uint valid_rows = min(4u, M - first_row);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint g  = blk * 4u + (lane >> 3);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        // Scale in float so small half inputs stay exact.
        // All rows can then use weights without bit shifts.
        const float x1 = e1 * 0x1p-4f, x2 = e2 * 0x1p-8f, x3 = e3 * 0x1p-12f;
        const float x5 = e5 * 0x1p-4f, x6 = e6 * 0x1p-8f, x7 = e7 * 0x1p-12f;
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            if (r >= valid_rows) continue;
            const uint row = first_row + r;
            device const ushort* wp = (device const ushort*)(W + row * row_bytes + byte_base);
            const uint w0 = uint(wp[0]);
            const uint w1 = uint(wp[1]);
            const float s = float(scales[row * n_groups + g]);
            const float b = float(biases[row * n_groups + g]);
            float dot = 0.0f;
            dot = fma(float(w0 & 0x000Fu), e0, dot); dot = fma(float(w0 & 0x00F0u), x1, dot);
            dot = fma(float(w0 & 0x0F00u), x2, dot); dot = fma(float(w0 & 0xF000u), x3, dot);
            dot = fma(float(w1 & 0x000Fu), e4, dot); dot = fma(float(w1 & 0x00F0u), x5, dot);
            dot = fma(float(w1 & 0x0F00u), x6, dot); dot = fma(float(w1 & 0xF000u), x7, dot);
            acc[r] = fma(s, dot, acc[r]);
            acc[r] = fma(b, sum, acc[r]);
        }
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {
            if (r >= valid_rows) continue;
            const uint row = first_row + r;
            const float s = float(scales[row * n_groups + g]);
            const float b = float(biases[row * n_groups + g]);
            const uint8_t byte = W[row * row_bytes + g * (kGroupSize / 2) + lane];
            float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
            dot = fma(float(uint(byte >> 4)), x1, dot);
            acc[r] = fma(s, dot, acc[r]);
            acc[r] = fma(b, sum, acc[r]);
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {
        if (r >= valid_rows) continue;
        const float value = simd_sum(acc[r]);
        if (lane == 0) {
            const half result = half(value);
            y[first_row + r] = result;
            if (duplicate_output) duplicate_y[first_row + r] = result;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void dequant_int4_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, y, false, MM, NN,
                                tg_idx * rows_per_tg + sg_idx * 4u, lane);
}


[[kernel, max_total_threads_per_threadgroup(64)]]
void dequant_int4_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint q_groups = QQ / 4u + uint(QQ % 4u != 0u);
    const uint kv_groups = KK / 4u + uint(KK % 4u != 0u);
    const bool k_equals_v = is_function_constant_defined(FC_INT4_QKV_K_EQUALS_V) &&
        FC_INT4_QKV_K_EQUALS_V;
    const uint group = tg_idx * 2u + sg_idx;
    if (group >= q_groups + (k_equals_v ? kv_groups : 2u * kv_groups)) return;

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    device half* duplicate_y;
    bool duplicate_output = false;
    uint local_row;
    uint M;
    if (group < q_groups) {
        W = qW; scales = qScales; biases = qBiases; y = qY; duplicate_y = qY;
        local_row = group * 4u;
        M = QQ;
    } else if (group < q_groups + kv_groups) {
        W = kW; scales = kScales; biases = kBiases; y = kY; duplicate_y = vY;
        duplicate_output = k_equals_v;
        local_row = (group - q_groups) * 4u;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY; duplicate_y = vY;
        local_row = (group - q_groups - kv_groups) * 4u;
        M = KK;
    }
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, duplicate_y, duplicate_output, M, NN,
                                local_row, lane);
}
