#include <metal_stdlib>
using namespace metal;

// qwen_common.metal is composed before this module by MetalContext.

struct QwenNormRoPEParameters {
    QwenMetalDimension tokenCount;
    QwenMetalDimension headCount;
    QwenMetalDimension headDimension;
    QwenMetalDimension rotaryDimension;
    QwenMetalDimension startPosition;
    float theta;
    float epsilon;
    QwenMetalDimension reserved;
};

struct QwenMRoPEParameters {
    QwenMetalDimension tokenCount;
    QwenMetalDimension headCount;
    QwenMetalDimension headDimension;
    QwenMetalDimension rotaryDimension;
    QwenMetalDimension temporalSection;
    QwenMetalDimension heightSection;
    QwenMetalDimension widthSection;
    QwenMetalDimension reserved;
    float theta;
    float epsilon;
};

struct QwenGateParameters {
    QwenMetalDimension elementCount;
    QwenMetalDimension reserved0;
    QwenMetalDimension reserved1;
    QwenMetalDimension reserved2;
};

/// Qwen 3.5/3.6 per-head RMS normalization followed by partial rotate_half RoPE.
/// The stored norm weights are residual weights: the multiplier is (1 + weight).
kernel void qwen_qk_norm_partial_rope(
    constant QwenNormRoPEParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weight [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    uint item [[thread_position_in_grid]]
) {
    const uint itemCount = parameters.tokenCount * parameters.headCount;
    if (item >= itemCount) {
        return;
    }

    const uint dimension = parameters.headDimension;
    const uint base = item * dimension;
    float squareSum = 0.0f;
    for (uint index = 0; index < dimension; ++index) {
        const float value = input[base + index];
        squareSum += value * value;
    }
    const float inverseRMS = rsqrt(squareSum / float(dimension) + parameters.epsilon);

    const uint rotaryDimension = parameters.rotaryDimension;
    const uint rotaryHalf = rotaryDimension / 2;
    const uint token = item / parameters.headCount;
    const float position = float(parameters.startPosition + token);

    for (uint index = 0; index < dimension; ++index) {
        const float normalized = input[base + index] * inverseRMS * (1.0f + weight[index]);
        if (index >= rotaryDimension) {
            output[base + index] = normalized;
            continue;
        }

        const bool firstHalf = index < rotaryHalf;
        const uint frequencyIndex = firstHalf ? index : index - rotaryHalf;
        const uint partnerIndex = firstHalf ? index + rotaryHalf : index - rotaryHalf;
        const float partner = input[base + partnerIndex] * inverseRMS * (1.0f + weight[partnerIndex]);
        const float exponent = (2.0f * float(frequencyIndex)) / float(rotaryDimension);
        const float inverseFrequency = pow(parameters.theta, -exponent);
        const float angle = position * inverseFrequency;
        const float cosine = cos(angle);
        const float sine = sin(angle);
        output[base + index] = firstHalf
            ? normalized * cosine - partner * sine
            : normalized * cosine + partner * sine;
    }
}

/// Vector M-RoPE variant. Positions are token-major Int32 `[t,h,w]` and
/// sections partition the first rotary half; the second half uses the same axis.
kernel void qwen_qk_norm_partial_mrope(
    constant QwenMRoPEParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const float* weight [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const int* positions [[buffer(QwenMetalBufferIndexScratch)]],
    uint item [[thread_position_in_grid]]
) {
    const uint itemCount = parameters.tokenCount * parameters.headCount;
    if (item >= itemCount) {
        return;
    }
    const uint dimension = parameters.headDimension;
    const uint base = item * dimension;
    float squareSum = 0.0f;
    for (uint index = 0; index < dimension; ++index) {
        const float value = input[base + index];
        squareSum += value * value;
    }
    const float inverseRMS = rsqrt(squareSum / float(dimension) + parameters.epsilon);
    const uint rotaryDimension = parameters.rotaryDimension;
    const uint rotaryHalf = rotaryDimension / 2;
    const uint token = item / parameters.headCount;
    for (uint index = 0; index < dimension; ++index) {
        const float normalized = input[base + index] * inverseRMS * (1.0f + weight[index]);
        if (index >= rotaryDimension) {
            output[base + index] = normalized;
            continue;
        }
        const bool firstHalf = index < rotaryHalf;
        const uint frequencyIndex = firstHalf ? index : index - rotaryHalf;
        const uint partnerIndex = firstHalf ? index + rotaryHalf : index - rotaryHalf;
        const float partner = input[base + partnerIndex] * inverseRMS
            * (1.0f + weight[partnerIndex]);
        uint axis = 2;
        if (frequencyIndex < parameters.temporalSection) {
            axis = 0;
        } else if (frequencyIndex
                   < parameters.temporalSection + parameters.heightSection) {
            axis = 1;
        }
        const float position = float(positions[token * 3 + axis]);
        const float exponent = (2.0f * float(frequencyIndex)) / float(rotaryDimension);
        const float inverseFrequency = pow(parameters.theta, -exponent);
        const float angle = position * inverseFrequency;
        const float cosine = cos(angle);
        const float sine = sin(angle);
        output[base + index] = firstHalf
            ? normalized * cosine - partner * sine
            : normalized * cosine + partner * sine;
    }
}

/// Applies the full-attention sigmoid output gate after head-major attention has
/// been transposed/flattened into token-major order and before o_proj.
kernel void qwen_full_attention_sigmoid_gate(
    constant QwenGateParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* attention [[buffer(QwenMetalBufferIndexInput)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* rawGate [[buffer(QwenMetalBufferIndexScratch)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.elementCount) {
        return;
    }
    const float gate = 1.0f / (1.0f + exp(-rawGate[index]));
    output[index] = attention[index] * gate;
}
