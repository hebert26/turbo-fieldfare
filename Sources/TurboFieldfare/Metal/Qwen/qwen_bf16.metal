#include <metal_stdlib>
using namespace metal;

// Also compiled alone with MTLMathMode.safe for the BF16 runtime. The fixed
// binding indices match qwen_common.metal (0, 1, 2, 5).
// Weights are raw little-endian BF16 bits in shared Metal buffers. Nonfinite
// values convert to FP32 Inf/NaN; FP32 FMA propagates them. No FP32 copy.
struct QwenBF16Parameters {
    uint rows;
    uint columns;
    uint firstRow;
    uint rowsInChunk;
    uint tokenCount;
};

static inline float qwenBF16Value(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void qwen_bf16_embedding(
    constant QwenBF16Parameters& p [[buffer(0)]],
    device const uint* tokenIDs [[buffer(1)]],
    device const ushort* weightBits [[buffer(2)]],
    device float* output [[buffer(5)]],
    uint2 location [[thread_position_in_grid]]
) {
    const uint column = location.x;
    const uint token = location.y;
    if (token >= p.tokenCount || column >= p.columns) return;
    const uint row = tokenIDs[token];
    // Host validates IDs before encoding. If a caller mutates shared IDs
    // before completion, emit NaN once rather than leave stale output bytes.
    if (row >= p.rows) {
        if (p.firstRow == 0u) output[ulong(token) * p.columns + column] = NAN;
        return;
    }
    if (row < p.firstRow || row - p.firstRow >= p.rowsInChunk) return;
    const uint localRow = row - p.firstRow;
    output[ulong(token) * p.columns + column] =
        qwenBF16Value(weightBits[ulong(localRow) * p.columns + column]);
}

// Recover the low bits lost by both the product and the running FP32 sum.
// The source weights stay BF16 and the result stays FP32. In particular, a
// cancelling row must not discard small terms merely because earlier terms
// made the running sum large. Nonfinite arithmetic keeps the former ordered
// FMA behavior, without forming Inf-Inf inside the correction calculation.
static inline float qwenBF16OrderedDot(
    const device ushort* row, const device float* vector, uint columns
) {
    float sum = 0.0f;
    for (uint column = 0; column < columns; ++column) {
        sum = fma(qwenBF16Value(row[column]), vector[column], sum);
    }
    return sum;
}

static inline float qwenBF16StableDot(
    const device ushort* row, const device float* vector, uint columns
) {
    float sum = 0.0f;
    float correction = 0.0f;
    for (uint column = 0; column < columns; ++column) {
        const float weight = qwenBF16Value(row[column]);
        const float product = weight * vector[column];
        const float next = sum + product;
        if (!isfinite(product) || !isfinite(next)) {
            return qwenBF16OrderedDot(row, vector, columns);
        }
        const float productError = fma(weight, vector[column], -product);
        const float sumError = abs(sum) >= abs(product)
            ? (sum - next) + product
            : (product - next) + sum;
        correction += productError + sumError;
        if (!isfinite(correction)) {
            return qwenBF16OrderedDot(row, vector, columns);
        }
        sum = next;
    }
    const float result = sum + correction;
    return isfinite(result) ? result : qwenBF16OrderedDot(row, vector, columns);
}

// The source CPU BLAS uses 64 independent FMA streams for this matrix family.
// Preserve that FP32 operation order rather than replacing its rounded result
// with a higher-precision mathematical dot. Independent BF16/FP32 probes cover
// rows >= 512 and complete 64-column blocks at widths 512 through 8192,
// and the separately verified 256-by-2048 routing matrix family.
static inline float qwenBF16SourceLargeDot(
    const device ushort* row, const device float* vector, uint columns
) {
    float partial[64] = {0.0f};
    for (uint column = 0; column < columns; ++column) {
        const uint stream = column & 63u;
        partial[stream] = fma(qwenBF16Value(row[column]), vector[column], partial[stream]);
    }
    float collapsed[16];
    for (uint index = 0; index < 16; ++index) {
        const float first = partial[index] + partial[index + 16];
        const float second = first + partial[index + 32];
        collapsed[index] = second + partial[index + 48];
    }
    float groups[4];
    for (uint index = 0; index < 4; ++index) {
        const uint base = index * 4;
        const float first = collapsed[base] + collapsed[base + 1];
        const float second = first + collapsed[base + 2];
        groups[index] = second + collapsed[base + 3];
    }
    const float first = groups[0] + groups[1];
    const float second = first + groups[2];
    // Keep the qualified reduction's nonfinite result for the host validator.
    // Replaying another order could turn its overflow into a finite value.
    return second + groups[3];
}

// The source BLAS's 32-by-2048 matrix family uses sixteen FMA streams.
// The first row in each eight-row block consumes the second half of every
// 32-column block first. Global row numbering keeps this order across chunks.
static inline float qwenBF16SourceSmall32Dot(
    const device ushort* row, const device float* vector, uint columns,
    uint globalRow
) {
    float partial[16] = {0.0f};
    const bool secondHalfFirst = (globalRow & 7u) == 0u;
    for (uint base = 0; base < columns; base += 32u) {
        for (uint lane = 0; lane < 16u; ++lane) {
            const uint first = base + lane + (secondHalfFirst ? 16u : 0u);
            const uint second = base + lane + (secondHalfFirst ? 0u : 16u);
            partial[lane] = fma(qwenBF16Value(row[first]), vector[first], partial[lane]);
            partial[lane] = fma(qwenBF16Value(row[second]), vector[second], partial[lane]);
        }
    }
    for (uint distance = 8u; distance != 0u; distance >>= 1u) {
        for (uint lane = 0; lane < distance; ++lane) {
            partial[lane] = partial[lane] + partial[lane + distance];
        }
    }
    return partial[0];
}

// Source order for the shared-output gate's 1-by-2048 matrix family.
// The CPU source uses four 512-column chunks. Each has four FMA streams for
// its first 508 columns, then a separately rounded multiply/add for the last
// four columns. Reduce crossed pairs, then left-fold the four chunk sums.
static inline float qwenBF16SourceSingleRow2048Dot(
    const device ushort* row, const device float* vector
) {
#pragma clang fp contract(off)
    float chunks[4];
    for (uint chunk = 0; chunk < 4u; ++chunk) {
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        const uint begin = chunk * 512u;
        for (uint index = 0; index < 508u; index += 4u) {
            const uint column = begin + index;
            a0 = fma(qwenBF16Value(row[column]), vector[column], a0);
            a1 = fma(qwenBF16Value(row[column + 1u]), vector[column + 1u], a1);
            a2 = fma(qwenBF16Value(row[column + 2u]), vector[column + 2u], a2);
            a3 = fma(qwenBF16Value(row[column + 3u]), vector[column + 3u], a3);
        }
        const uint tail = begin + 508u;
        const float product0 = qwenBF16Value(row[tail]) * vector[tail];
        const float product1 = qwenBF16Value(row[tail + 1u]) * vector[tail + 1u];
        const float product2 = qwenBF16Value(row[tail + 2u]) * vector[tail + 2u];
        const float product3 = qwenBF16Value(row[tail + 3u]) * vector[tail + 3u];
        a0 = a0 + product0;
        a1 = a1 + product1;
        a2 = a2 + product2;
        a3 = a3 + product3;
        const float even = a0 + a2;
        const float odd = a1 + a3;
        chunks[chunk] = even + odd;
    }
    const float first = chunks[0] + chunks[1];
    const float second = first + chunks[2];
    return second + chunks[3];
}

// One FP32-accumulating output row per thread. Each chunk holds whole rows,
// so no cross-chunk atomic accumulation or GPU-memory aliasing is required.
kernel void qwen_bf16_project_fp32(
    constant QwenBF16Parameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const ushort* weightBits [[buffer(2)]],
    device float* output [[buffer(5)]],
    uint2 location [[thread_position_in_grid]]
) {
    const uint localRow = location.x;
    const uint token = location.y;
    if (token >= p.tokenCount || localRow >= p.rowsInChunk) return;
    const device float* vector = input + ulong(token) * p.columns;
    const device ushort* row = weightBits + ulong(localRow) * p.columns;
    const bool largeSourceFamily = p.rows >= 512u && p.columns >= 512u
        && (p.columns & 63u) == 0u;
    const bool routingSourceFamily = p.rows == 256u && p.columns == 2048u;
    const bool smallSourceFamily = p.rows == 32u && p.columns == 2048u;
    output[ulong(token) * p.rows + p.firstRow + localRow] =
        p.rows == 1u && p.columns == 2048u
            ? qwenBF16SourceSingleRow2048Dot(row, vector)
            : (largeSourceFamily || routingSourceFamily
                ? qwenBF16SourceLargeDot(row, vector, p.columns)
                : smallSourceFamily
                    ? qwenBF16SourceSmall32Dot(row, vector, p.columns, p.firstRow + localRow)
                    : qwenBF16StableDot(row, vector, p.columns));
}
