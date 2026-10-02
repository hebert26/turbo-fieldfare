#include <metal_stdlib>
using namespace metal;

// qwen_common.metal is composed before this module by MetalContext.
// BF16 source attention also compiles this module alone in safe math mode.
#ifndef QWEN_METAL_COMMON
enum QwenSourceLinearBufferIndex : uint {
    QwenMetalBufferIndexParameters = 0,
    QwenMetalBufferIndexInput = 1,
    QwenMetalBufferIndexWeights = 2,
    QwenMetalBufferIndexScales = 3,
    QwenMetalBufferIndexBiases = 4,
    QwenMetalBufferIndexOutput = 5,
    QwenMetalBufferIndexScratch = 6,
    QwenMetalBufferIndexState = 7,
};
#endif

struct QwenLinearLayoutParameters {
    uint tokenCount;
    uint channelCount;
    uint tokenToChannel;
    uint reserved;
};

struct QwenLinearConvolutionParameters {
    uint tokenCount;
    uint channelCount;
    uint width;
    uint reserved;
};

struct QwenLinearRecurrenceParameters {
    uint tokenCount;
    uint headCount;
    uint keyDimension;
    uint valueDimension;
    float epsilon;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct QwenLinearGatedNormParameters {
    uint tokenCount;
    uint headCount;
    uint valueDimension;
    uint reserved;
    float epsilon;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void qwen_linear_layout(
    constant QwenLinearLayoutParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.tokenCount * p.channelCount;
    if (index >= count) return;
    if (p.tokenToChannel != 0) {
        const uint token = index / p.channelCount;
        const uint channel = index - token * p.channelCount;
        output[channel * p.tokenCount + token] = input[index];
    } else {
        const uint channel = index / p.tokenCount;
        const uint token = index - channel * p.tokenCount;
        output[token * p.channelCount + channel] = input[index];
    }
}

/// One thread owns one channel, so history updates remain ordered across tokens.
kernel void qwen_linear_causal_conv(
    constant QwenLinearConvolutionParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weights [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device float* history [[buffer(QwenMetalBufferIndexState)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= p.channelCount || p.width != 4) return;
    const uint historyBase = channel * p.width;
    const uint tokenBase = channel * p.tokenCount;
    for (uint token = 0; token < p.tokenCount; ++token) {
        history[historyBase] = history[historyBase + 1];
        history[historyBase + 1] = history[historyBase + 2];
        history[historyBase + 2] = history[historyBase + 3];
        history[historyBase + 3] = input[tokenBase + token];
        float sum = 0.0f;
        for (uint index = 0; index < 4; ++index) {
            sum += history[historyBase + index] * weights[historyBase + index];
        }
#ifdef QWEN_PINNED_SOURCE_SCALAR_EXP
        // The CPU causal-convolution slice is strided, so SiLU uses scalar expf.
        // expf(-sum) rounds to one throughout |sum| < 2^-125.
        // Preserve the nearest-even half when GPU division would flush it.
        const uint rawSum = as_type<uint>(sum);
        if ((rawSum & 0x7fffffffu) < 0x01000000u) {
            uint significand = rawSum & 0x007fffffu;
            if ((rawSum & 0x7f800000u) != 0u) significand |= 0x00800000u;
            uint roundedHalf = significand >> 1;
            if ((significand & 1u) && (roundedHalf & 1u)) ++roundedHalf;
            output[tokenBase + token] = as_type<float>((rawSum & 0x80000000u) | roundedHalf);
        } else {
            output[tokenBase + token] = sum / (1.0f + qwenSourceScalarExpFloat(-sum));
        }
#elif defined(QWEN_PINNED_SOURCE_EXP)
        output[tokenBase + token] = sum / (1.0f + qwenSourceExpFloatBitScale(-sum));
#else
        output[tokenBase + token] = sum / (1.0f + exp(-sum));
#endif
    }
}

/// Correctness-first FP32 recurrence. One thread owns a value head and its
/// complete matrix, avoiding cross-thread barriers at awkward token tails.
kernel void qwen_linear_recurrence_fp32(
    constant QwenLinearRecurrenceParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* query [[buffer(QwenMetalBufferIndexInput)]],
    device const float* key [[buffer(QwenMetalBufferIndexWeights)]],
    device const float* value [[buffer(QwenMetalBufferIndexScales)]],
    device const float* logDecay [[buffer(QwenMetalBufferIndexBiases)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* beta [[buffer(QwenMetalBufferIndexScratch)]],
    device float* state [[buffer(QwenMetalBufferIndexState)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= p.headCount) return;
    const uint stateBase = head * p.keyDimension * p.valueDimension;
    const float queryScale = rsqrt(float(p.keyDimension));
    for (uint token = 0; token < p.tokenCount; ++token) {
        const uint qBase = (token * p.headCount + head) * p.keyDimension;
        const uint vBase = (token * p.headCount + head) * p.valueDimension;
        const uint scalar = token * p.headCount + head;
        float qSquares = 0.0f;
        float kSquares = 0.0f;
        for (uint k = 0; k < p.keyDimension; ++k) {
            qSquares += query[qBase + k] * query[qBase + k];
            kSquares += key[qBase + k] * key[qBase + k];
        }
        const float qInverse = rsqrt(qSquares + p.epsilon) * queryScale;
        const float kInverse = rsqrt(kSquares + p.epsilon);
        const float decay = exp(logDecay[scalar]);
        for (uint k = 0; k < p.keyDimension; ++k) {
            for (uint v = 0; v < p.valueDimension; ++v) {
                state[stateBase + k * p.valueDimension + v] *= decay;
            }
        }
        for (uint v = 0; v < p.valueDimension; ++v) {
            float prediction = 0.0f;
            for (uint k = 0; k < p.keyDimension; ++k) {
                prediction += state[stateBase + k * p.valueDimension + v]
                    * key[qBase + k] * kInverse;
            }
            const float delta = (value[vBase + v] - prediction) * beta[scalar];
            for (uint k = 0; k < p.keyDimension; ++k) {
                state[stateBase + k * p.valueDimension + v] +=
                    key[qBase + k] * kInverse * delta;
            }
        }
        for (uint v = 0; v < p.valueDimension; ++v) {
            float sum = 0.0f;
            for (uint k = 0; k < p.keyDimension; ++k) {
                sum += state[stateBase + k * p.valueDimension + v]
                    * query[qBase + k] * qInverse;
            }
            output[vBase + v] = sum;
        }
    }
}

kernel void qwen_linear_gated_rmsnorm(
    constant QwenLinearGatedNormParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weights [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* gate [[buffer(QwenMetalBufferIndexScratch)]],
    uint item [[thread_position_in_grid]]) {
    const uint itemCount = p.tokenCount * p.headCount;
    if (item >= itemCount) return;
    const uint base = item * p.valueDimension;
    float squareSum = 0.0f;
    for (uint index = 0; index < p.valueDimension; ++index) {
        squareSum += input[base + index] * input[base + index];
    }
    const float inverseRMS = rsqrt(squareSum / float(p.valueDimension) + p.epsilon);
    for (uint index = 0; index < p.valueDimension; ++index) {
        const float rawGate = gate[base + index];
        const float siluGate = rawGate / (1.0f + exp(-rawGate));
        output[base + index] = input[base + index] * inverseRMS * weights[index] * siluGate;
    }
}

// Source operations first round each product to FP32, then reduce those
// products. Correct the reduction loss without retaining product low bits or
// fusing a multiplication into the subsequent source addition.
static inline void qwenSourceLinearAdd(
    float term, thread float& sum, thread float& correction
) {
    const float next = sum + term;
    if (!isfinite(term) || !isfinite(next)) {
        sum = next;
        correction = 0.0f;
        return;
    }
    correction += abs(sum) >= abs(term)
        ? (sum - next) + term : (term - next) + sum;
    sum = next;
}

// Torch's contiguous ARM64 FP32 sum: four interleaved four-lane vectors
// through four cascade levels. Each squared input is rounded before addition.
static inline float qwenSourceLinearSquareSum(
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

struct QwenSourceLinearRecurrenceParameters {
    uint tokenCount;
    uint headCount;
    uint keyDimension;
    uint valueDimension;
    float epsilon;
    uint initialToken;
    float queryScale;
    float queryDivisor;
};

// The uncached one-token source forward uses the chunk formulation. Its
// within-token attention is computed before applying it to the corrected
// value. Cached tokens use the recurrent source formulation instead.
kernel void qwen_source_linear_recurrence_fp32(
    constant QwenSourceLinearRecurrenceParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* query [[buffer(QwenMetalBufferIndexInput)]],
    device const float* key [[buffer(QwenMetalBufferIndexWeights)]],
    device const float* value [[buffer(QwenMetalBufferIndexScales)]],
    device const float* logDecay [[buffer(QwenMetalBufferIndexBiases)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* beta [[buffer(QwenMetalBufferIndexScratch)]],
    device float* state [[buffer(QwenMetalBufferIndexState)]],
    uint head [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (head >= p.headCount) return;
    const uint stateBase = head * p.keyDimension * p.valueDimension;
    for (uint token = 0; token < p.tokenCount; ++token) {
        const uint qBase = (token * p.headCount + head) * p.keyDimension;
        const uint vBase = (token * p.headCount + head) * p.valueDimension;
        const uint scalar = token * p.headCount + head;
        const float qSquares = qwenSourceLinearSquareSum(query, qBase, p.keyDimension);
        const float kSquares = qwenSourceLinearSquareSum(key, qBase, p.keyDimension);
        const float qInverse = 1.0f / sqrt(qSquares + p.epsilon);
        const float kInverse = 1.0f / sqrt(kSquares + p.epsilon);
#ifdef QWEN_PINNED_SOURCE_EXP
        const float decay = qwenSourceExpFloatBitScale(logDecay[scalar]);
#else
        const float decay = exp(logDecay[scalar]);
#endif
        const bool initial = p.initialToken != 0u && token == 0u;
        const bool sourceCachedSum = !initial && p.keyDimension == 128u
            && p.valueDimension == 128u;
        float attention = 0.0f, attentionCorrection = 0.0f;
        if (initial) {
            for (uint k = 0; k < p.keyDimension; ++k) {
                const float normalizedQuery = query[qBase + k] * qInverse;
                const float scaledQuery = normalizedQuery * p.queryScale;
                const float normalizedKey = key[qBase + k] * kInverse;
                if (p.keyDimension == 128u) {
                    // The source's padded 64-position, 128-column matmul
                    // accumulates this first-token dot with ordered FP32 FMA.
                    attention = fma(scaledQuery, normalizedKey, attention);
                } else {
                    const float product = scaledQuery * normalizedKey;
                    qwenSourceLinearAdd(product, attention, attentionCorrection);
                }
            }
            attention += attentionCorrection;
        }
        for (uint v = 0; v < p.valueDimension; ++v) {
            float prediction = 0.0f, predictionCorrection = 0.0f;
            float predictionBlock = 0.0f;
            float interAttention = 0.0f, interCorrection = 0.0f;
            for (uint k = 0; k < p.keyDimension; ++k) {
                const uint index = stateBase + k * p.valueDimension + v;
                const float normalizedKey = key[qBase + k] * kInverse;
                if (initial) {
                    const float betaKey = normalizedKey * beta[scalar];
                    const float decayedBetaKey = betaKey * decay;
                    const float predictionProduct = decayedBetaKey * state[index];
                    qwenSourceLinearAdd(predictionProduct, prediction, predictionCorrection);
                    const float normalizedQuery = query[qBase + k] * qInverse;
                    const float scaledQuery = normalizedQuery * p.queryScale;
                    const float decayedQuery = scaledQuery * decay;
                    const float interProduct = decayedQuery * state[index];
                    qwenSourceLinearAdd(interProduct, interAttention, interCorrection);
                } else {
                    state[index] *= decay;
                    const float predictionProduct = state[index] * normalizedKey;
                    if (sourceCachedSum) {
                        // Source sum(-2): separately rounded products, sixteen
                        // consecutive key rows per partial, then eight partials.
                        predictionBlock = predictionBlock + predictionProduct;
                        if ((k & 15u) == 15u) {
                            prediction = prediction + predictionBlock;
                            predictionBlock = 0.0f;
                        }
                    } else {
                        qwenSourceLinearAdd(predictionProduct, prediction, predictionCorrection);
                    }
                }
            }
            prediction += predictionCorrection;
            const float betaValue = value[vBase + v] * beta[scalar];
            const float delta = initial
                ? betaValue - prediction
                : (value[vBase + v] - prediction) * beta[scalar];
            for (uint k = 0; k < p.keyDimension; ++k) {
                const uint index = stateBase + k * p.valueDimension + v;
                if (initial) state[index] *= decay;
                const float normalizedKey = key[qBase + k] * kInverse;
                const float update = normalizedKey * delta;
                state[index] = state[index] + update;
            }
            if (initial) {
                const float withinAttention = attention * delta;
                output[vBase + v] = (interAttention + interCorrection) + withinAttention;
            } else {
                float sum = 0.0f, correction = 0.0f;
                float block = 0.0f;
                for (uint k = 0; k < p.keyDimension; ++k) {
                    const float normalizedQuery = query[qBase + k] * qInverse;
                    const float scaledQuery = normalizedQuery / p.queryDivisor;
                    const float product = state[stateBase + k * p.valueDimension + v] * scaledQuery;
                    if (sourceCachedSum) {
                        block = block + product;
                        if ((k & 15u) == 15u) {
                            sum = sum + block;
                            block = 0.0f;
                        }
                    } else {
                        qwenSourceLinearAdd(product, sum, correction);
                    }
                }
                output[vBase + v] = sum + correction;
            }
        }
    }
}

// One group owns a cached head, and each lane owns one value column. The
// column's state, sixteen-key partials and final eight-partial sum retain the
// serial kernel's order. No state element is shared between columns.
kernel void qwen_source_linear_recurrence_cached_128(
    constant QwenSourceLinearRecurrenceParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* query [[buffer(QwenMetalBufferIndexInput)]],
    device const float* key [[buffer(QwenMetalBufferIndexWeights)]],
    device const float* value [[buffer(QwenMetalBufferIndexScales)]],
    device const float* logDecay [[buffer(QwenMetalBufferIndexBiases)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* beta [[buffer(QwenMetalBufferIndexScratch)]],
    device float* state [[buffer(QwenMetalBufferIndexState)]],
    uint head [[threadgroup_position_in_grid]],
    uint v [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    // Uniform checks precede the barrier. The host dispatches exactly 128
    // lanes per head and selects only a noninitial one-token 128x128 state.
    if (head >= p.headCount || p.tokenCount != 1u || p.initialToken != 0u
        || p.keyDimension != 128u || p.valueDimension != 128u) return;
    const uint stateBase = head * p.keyDimension * p.valueDimension;
    const uint qBase = head * p.keyDimension;
    const uint vBase = head * p.valueDimension;
    threadgroup float headQInverse;
    threadgroup float headKInverse;
    threadgroup float headDecay;
    if (v == 0u) {
        const float qSquares = qwenSourceLinearSquareSum(query, qBase, p.keyDimension);
        const float kSquares = qwenSourceLinearSquareSum(key, qBase, p.keyDimension);
        headQInverse = 1.0f / sqrt(qSquares + p.epsilon);
        headKInverse = 1.0f / sqrt(kSquares + p.epsilon);
#ifdef QWEN_PINNED_SOURCE_EXP
        headDecay = qwenSourceExpFloatBitScale(logDecay[head]);
#else
        headDecay = exp(logDecay[head]);
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qInverse = headQInverse;
    const float kInverse = headKInverse;
    const float decay = headDecay;
    float prediction = 0.0f, predictionCorrection = 0.0f;
    float predictionBlock = 0.0f;
    for (uint k = 0; k < p.keyDimension; ++k) {
        const uint index = stateBase + k * p.valueDimension + v;
        const float normalizedKey = key[qBase + k] * kInverse;
        state[index] *= decay;
        const float predictionProduct = state[index] * normalizedKey;
        predictionBlock = predictionBlock + predictionProduct;
        if ((k & 15u) == 15u) {
            prediction = prediction + predictionBlock;
            predictionBlock = 0.0f;
        }
    }
    prediction += predictionCorrection;
    const float delta = (value[vBase + v] - prediction) * beta[head];
    for (uint k = 0; k < p.keyDimension; ++k) {
        const uint index = stateBase + k * p.valueDimension + v;
        const float normalizedKey = key[qBase + k] * kInverse;
        const float update = normalizedKey * delta;
        state[index] = state[index] + update;
    }
    float sum = 0.0f, correction = 0.0f;
    float block = 0.0f;
    for (uint k = 0; k < p.keyDimension; ++k) {
        const float normalizedQuery = query[qBase + k] * qInverse;
        const float scaledQuery = normalizedQuery / p.queryDivisor;
        const float product = state[stateBase + k * p.valueDimension + v] * scaledQuery;
        block = block + product;
        if ((k & 15u) == 15u) {
            sum = sum + block;
            block = 0.0f;
        }
    }
    output[vBase + v] = sum + correction;
}

// Opt-in grouped-prefill discriminator. Original one-token kernel above is unchanged.
kernel void qwen_source_linear_recurrence_cached_128_grouped(
    constant QwenSourceLinearRecurrenceParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* query [[buffer(QwenMetalBufferIndexInput)]],
    device const float* key [[buffer(QwenMetalBufferIndexWeights)]],
    device const float* value [[buffer(QwenMetalBufferIndexScales)]],
    device const float* logDecay [[buffer(QwenMetalBufferIndexBiases)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* beta [[buffer(QwenMetalBufferIndexScratch)]],
    device float* state [[buffer(QwenMetalBufferIndexState)]],
    uint head [[threadgroup_position_in_grid]],
    uint v [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    // Uniform checks precede the barrier. The host dispatches exactly 128
    // lanes per head and selects only noninitial 128x128 state.
    if (head >= p.headCount || p.tokenCount == 0u || p.initialToken != 0u
        || p.keyDimension != 128u || p.valueDimension != 128u) return;
    const uint stateBase = head * p.keyDimension * p.valueDimension;
    threadgroup float headQInverse;
    threadgroup float headKInverse;
    threadgroup float headDecay;
    for (uint token = 0; token < p.tokenCount; ++token) {
        const uint scalar = token * p.headCount + head;
        const uint qBase = scalar * p.keyDimension;
        const uint vBase = scalar * p.valueDimension;
        if (v == 0u) {
            const float qSquares = qwenSourceLinearSquareSum(query, qBase, p.keyDimension);
            const float kSquares = qwenSourceLinearSquareSum(key, qBase, p.keyDimension);
            headQInverse = 1.0f / sqrt(qSquares + p.epsilon);
            headKInverse = 1.0f / sqrt(kSquares + p.epsilon);
    #ifdef QWEN_PINNED_SOURCE_EXP
            headDecay = qwenSourceExpFloatBitScale(logDecay[scalar]);
    #else
            headDecay = exp(logDecay[scalar]);
    #endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float qInverse = headQInverse;
        const float kInverse = headKInverse;
        const float decay = headDecay;
        float prediction = 0.0f, predictionCorrection = 0.0f;
        float predictionBlock = 0.0f;
        for (uint k = 0; k < p.keyDimension; ++k) {
            const uint index = stateBase + k * p.valueDimension + v;
            const float normalizedKey = key[qBase + k] * kInverse;
            state[index] *= decay;
            const float predictionProduct = state[index] * normalizedKey;
            predictionBlock = predictionBlock + predictionProduct;
            if ((k & 15u) == 15u) {
                prediction = prediction + predictionBlock;
                predictionBlock = 0.0f;
            }
        }
        prediction += predictionCorrection;
        const float delta = (value[vBase + v] - prediction) * beta[scalar];
        for (uint k = 0; k < p.keyDimension; ++k) {
            const uint index = stateBase + k * p.valueDimension + v;
            const float normalizedKey = key[qBase + k] * kInverse;
            const float update = normalizedKey * delta;
            state[index] = state[index] + update;
        }
        float sum = 0.0f, correction = 0.0f;
        float block = 0.0f;
        for (uint k = 0; k < p.keyDimension; ++k) {
            const float normalizedQuery = query[qBase + k] * qInverse;
            const float scaledQuery = normalizedQuery / p.queryDivisor;
            const float product = state[stateBase + k * p.valueDimension + v] * scaledQuery;
            block = block + product;
            if ((k & 15u) == 15u) {
                sum = sum + block;
                block = 0.0f;
            }
        }
        output[vBase + v] = sum + correction;
        // All columns finish using this token's shared scalars before lane0
        // overwrites them for the next token. State columns remain disjoint.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void qwen_source_linear_gated_rmsnorm(
    constant QwenLinearGatedNormParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weights [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* gate [[buffer(QwenMetalBufferIndexScratch)]],
    uint item [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (item >= p.tokenCount * p.headCount) return;
    const uint base = item * p.valueDimension;
    const float squareSum = qwenSourceLinearSquareSum(input, base, p.valueDimension);
    const float meanSquare = squareSum / float(p.valueDimension);
    const float inverseRMS = 1.0f / sqrt(meanSquare + p.epsilon);
    for (uint index = 0; index < p.valueDimension; ++index) {
        const float normalized = input[base + index] * inverseRMS;
        const float weighted = weights[index] * normalized;
        const float rawGate = gate[base + index];
#ifdef QWEN_PINNED_SOURCE_EXP
        const float siluGate = rawGate / (1.0f + qwenSourceExpFloatBitScale(-rawGate));
#else
        const float siluGate = rawGate / (1.0f + exp(-rawGate));
#endif
        output[base + index] = weighted * siluGate;
    }
}

// Opt-in 128-value head. Lane zero retains the serial source reduction order.
kernel void qwen_source_linear_gated_rmsnorm_128_lanes(
    constant QwenLinearGatedNormParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weights [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* gate [[buffer(QwenMetalBufferIndexScratch)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
    const uint item = group.x;
    const bool validHead = p.valueDimension == 128u && item < p.tokenCount * p.headCount;
    const bool validLane = validHead && lane < 128u;
    const uint base = item * p.valueDimension;
    threadgroup float inverseRMS;
    if (lane == 0u) {
        float inverse = 0.0f;
        if (validHead) {
            const float squareSum = qwenSourceLinearSquareSum(input, base, p.valueDimension);
            const float meanSquare = squareSum / float(p.valueDimension);
            inverse = 1.0f / sqrt(meanSquare + p.epsilon);
        }
        inverseRMS = inverse;
    }
    float siluGate = 0.0f;
    if (validLane) {
        const float rawGate = gate[base + lane];
#ifdef QWEN_PINNED_SOURCE_EXP
        siluGate = rawGate / (1.0f + qwenSourceExpFloatBitScale(-rawGate));
#else
        siluGate = rawGate / (1.0f + exp(-rawGate));
#endif
    }
    // Every dispatched lane reaches this barrier, even for a defensive invalid head.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!validLane) return;
    const float normalized = input[base + lane] * inverseRMS;
    const float weighted = weights[lane] * normalized;
    output[base + lane] = weighted * siluGate;
}
