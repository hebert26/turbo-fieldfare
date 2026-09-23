#include <metal_stdlib>
using namespace metal;

// qwen_common.metal is composed before this module by MetalContext.

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
        output[tokenBase + token] = sum / (1.0f + exp(-sum));
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
