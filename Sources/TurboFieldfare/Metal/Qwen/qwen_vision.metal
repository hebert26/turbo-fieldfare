#include <metal_stdlib>
using namespace metal;

struct QwenVisionParameters {
    uint rows;
    uint paddedRows;
    uint inputWidth;
    uint outputWidth;
    uint intermediateWidth;
    uint heads;
    uint gridHeight;
    uint gridWidth;
    uint mergeSize;
    uint positionCount;
    float epsilon;
    uint weightScalarBytes;
    uint patchScalarBytes;
    uint reserved;
};

inline float qwen_vision_load(device const uchar* values, uint index, uint scalarBytes) {
    return scalarBytes == 4u
        ? reinterpret_cast<device const float*>(values)[index]
        : float(reinterpret_cast<device const bfloat*>(values)[index]);
}

inline float qwen_vision_erf(float value) {
    // Abramowitz-Stegun 7.1.26. MSL has no scalar erf. End-to-end error is
    // measured separately against the frozen P3 fixture.
    const float signValue = value < 0.0f ? -1.0f : 1.0f;
    const float x = abs(value);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return signValue * (1.0f - polynomial * exp(-x * x));
}

inline float qwen_vision_gelu_exact(float value) {
    return 0.5f * value
        * (1.0f + qwen_vision_erf(value * 0.7071067811865475f));
}

inline float qwen_vision_gelu_tanh(float value) {
    const float coefficient = 0.7978845608028654f;
    return 0.5f * value * (1.0f + tanh(coefficient
        * (value + 0.044715f * value * value * value)));
}

kernel void qwen_vision_patch_position(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const uchar* patches [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    device const uchar* weight [[buffer(3)]],
    device const uchar* bias [[buffer(4)]],
    device const uchar* positionTable [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.paddedRows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    if (row >= p.rows) { output[index] = 0.0f; return; }
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner) {
        value = fma(qwen_vision_load(
                        patches, row * p.inputWidth + inner, p.patchScalarBytes),
                    qwen_vision_load(
                        weight, column * p.inputWidth + inner, p.weightScalarBytes), value);
    }
    const int2 coordinate = positions[row];
    const uint tableSide = uint(sqrt(float(p.positionCount)));
    const float sourceY = p.gridHeight > 1
        ? float(coordinate.x) * float(tableSide - 1) / float(p.gridHeight - 1) : 0.0f;
    const float sourceX = p.gridWidth > 1
        ? float(coordinate.y) * float(tableSide - 1) / float(p.gridWidth - 1) : 0.0f;
    const uint y0 = uint(floor(sourceY));
    const uint x0 = uint(floor(sourceX));
    const uint y1 = min(y0 + 1, tableSide - 1);
    const uint x1 = min(x0 + 1, tableSide - 1);
    const float fy = sourceY - float(y0);
    const float fx = sourceX - float(x0);
    const float top = mix(
        qwen_vision_load(positionTable, (y0 * tableSide + x0) * p.outputWidth + column, p.weightScalarBytes),
        qwen_vision_load(positionTable, (y0 * tableSide + x1) * p.outputWidth + column, p.weightScalarBytes), fx);
    const float bottom = mix(
        qwen_vision_load(positionTable, (y1 * tableSide + x0) * p.outputWidth + column, p.weightScalarBytes),
        qwen_vision_load(positionTable, (y1 * tableSide + x1) * p.outputWidth + column, p.weightScalarBytes), fx);
    output[index] = value + mix(top, bottom, fy);
}

kernel void qwen_vision_layernorm(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* normWeight [[buffer(2)]],
    device const uchar* normBias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.paddedRows) return;
    const uint base = row * p.inputWidth;
    if (row >= p.rows) {
        for (uint i = 0; i < p.inputWidth; ++i) output[base + i] = 0.0f;
        return;
    }
    float mean = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) mean += input[base + i];
    mean /= float(p.inputWidth);
    float variance = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) {
        const float centered = input[base + i] - mean;
        variance = fma(centered, centered, variance);
    }
    const float inverse = rsqrt(variance / float(p.inputWidth) + p.epsilon);
    for (uint i = 0; i < p.inputWidth; ++i) {
        output[base + i] = (input[base + i] - mean) * inverse
            * qwen_vision_load(normWeight, i, p.weightScalarBytes)
            + qwen_vision_load(normBias, i, p.weightScalarBytes);
    }
}

kernel void qwen_vision_rotate_qk(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const int2* positions [[buffer(1)]],
    device float* qkv [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint headDimension = p.inputWidth / p.heads;
    const uint rotaryHalf = headDimension / 2;
    const uint count = p.rows * p.heads * headDimension;
    if (index >= count) return;
    const uint dimension = index % headDimension;
    const uint vector = index / headDimension;
    const uint head = vector % p.heads;
    const uint row = vector / p.heads;
    if (dimension >= rotaryHalf) return;
    const uint axisHalf = rotaryHalf / 2;
    const uint axis = dimension < axisHalf ? 0 : 1;
    const uint frequency = dimension % axisHalf;
    const float coordinate = float(axis == 0 ? positions[row].x : positions[row].y);
    const float inverseFrequency = pow(10000.0f,
        -2.0f * float(frequency) / float(rotaryHalf));
    const float angle = coordinate * inverseFrequency;
    const float c = cos(angle);
    const float s = sin(angle);
    const uint rowBase = row * 3u * p.inputWidth;
    for (uint projection = 0; projection < 2; ++projection) {
        const uint base = rowBase + projection * p.inputWidth
            + head * headDimension;
        const float first = qkv[base + dimension];
        const float second = qkv[base + dimension + rotaryHalf];
        qkv[base + dimension] = first * c - second * s;
        qkv[base + dimension + rotaryHalf] = first * s + second * c;
    }
}

kernel void qwen_vision_attention(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= p.paddedRows * p.heads) return;
    const uint row = item / p.heads;
    const uint head = item % p.heads;
    const uint headDimension = p.inputWidth / p.heads;
    const uint outputBase = row * p.inputWidth + head * headDimension;
    if (row >= p.rows) {
        for (uint d = 0; d < headDimension; ++d) output[outputBase + d] = 0.0f;
        return;
    }
    const uint queryBase = row * 3u * p.inputWidth + head * headDimension;
    const float scale = rsqrt(float(headDimension));
    float maximum = -INFINITY;
    for (uint keyRow = 0; keyRow < p.rows; ++keyRow) {
        const uint keyBase = keyRow * 3u * p.inputWidth + p.inputWidth
            + head * headDimension;
        float score = 0.0f;
        for (uint d = 0; d < headDimension; ++d)
            score = fma(qkv[queryBase + d], qkv[keyBase + d], score);
        maximum = max(maximum, score * scale);
    }
    // Official head dimension is 72; tiny frozen fixtures use 8.
    float accumulator[72];
    for (uint d = 0; d < headDimension; ++d) accumulator[d] = 0.0f;
    float denominator = 0.0f;
    for (uint keyRow = 0; keyRow < p.rows; ++keyRow) {
        const uint keyBase = keyRow * 3u * p.inputWidth + p.inputWidth
            + head * headDimension;
        float score = 0.0f;
        for (uint d = 0; d < headDimension; ++d)
            score = fma(qkv[queryBase + d], qkv[keyBase + d], score);
        const float probability = exp(score * scale - maximum);
        denominator += probability;
        const uint valueBase = keyRow * 3u * p.inputWidth + 2u * p.inputWidth
            + head * headDimension;
        for (uint d = 0; d < headDimension; ++d)
            accumulator[d] = fma(probability, qkv[valueBase + d], accumulator[d]);
    }
    for (uint d = 0; d < headDimension; ++d)
        output[outputBase + d] = accumulator[d] / denominator;
}

kernel void qwen_vision_linear_residual(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device const uchar* weight [[buffer(3)]],
    device const uchar* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.paddedRows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    if (row >= p.rows) { output[index] = 0.0f; return; }
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = residual[row * p.outputWidth + column] + value;
}

kernel void qwen_vision_linear_gelu_tanh(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* weight [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.rows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = qwen_vision_gelu_tanh(value);
}

kernel void qwen_vision_merger_norm_pack(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* normWeight [[buffer(2)]],
    device const uchar* normBias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint sourceRow [[thread_position_in_grid]]) {
    if (sourceRow >= p.rows) return;
    const uint inputBase = sourceRow * p.inputWidth;
    float mean = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) mean += input[inputBase + i];
    mean /= float(p.inputWidth);
    float variance = 0.0f;
    for (uint i = 0; i < p.inputWidth; ++i) {
        const float centered = input[inputBase + i] - mean;
        variance = fma(centered, centered, variance);
    }
    const float inverse = rsqrt(variance / float(p.inputWidth) + p.epsilon);
    // Preprocessing is block-major, so four consecutive rows are one packed row.
    const uint outputBase = sourceRow * p.inputWidth;
    for (uint i = 0; i < p.inputWidth; ++i) {
        output[outputBase + i] = (input[inputBase + i] - mean) * inverse
            * qwen_vision_load(normWeight, i, p.weightScalarBytes)
            + qwen_vision_load(normBias, i, p.weightScalarBytes);
    }
}

kernel void qwen_vision_linear_gelu(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* weight [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.rows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = qwen_vision_gelu_exact(value);
}

kernel void qwen_vision_linear(
    constant QwenVisionParameters& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const uchar* weight [[buffer(2)]],
    device const uchar* bias [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = p.rows * p.outputWidth;
    if (index >= count) return;
    const uint row = index / p.outputWidth;
    const uint column = index % p.outputWidth;
    float value = qwen_vision_load(bias, column, p.weightScalarBytes);
    for (uint inner = 0; inner < p.inputWidth; ++inner)
        value = fma(input[row * p.inputWidth + inner],
                    qwen_vision_load(weight,
                        column * p.inputWidth + inner, p.weightScalarBytes), value);
    output[index] = value;
}

kernel void qwen_vision_add_layer_bias(
    constant uint& count [[buffer(0)]],
    constant float& bias [[buffer(1)]],
    device float* values [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) values[index] += bias;
}
