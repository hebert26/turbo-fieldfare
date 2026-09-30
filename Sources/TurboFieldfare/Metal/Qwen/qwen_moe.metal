#include <metal_stdlib>
using namespace metal;

// qwen_common.metal is composed before this module by MetalContext.
// The private safe-math BF16 library compiles this module alone. Keep these
// fixed bindings identical to QwenMetalBufferIndex in qwen_common.metal.
#ifndef QWEN_METAL_COMMON
enum QwenMetalBufferIndex : uint {
    QwenMetalBufferIndexParameters = 0,
    QwenMetalBufferIndexInput = 1,
    QwenMetalBufferIndexWeights = 2,
    QwenMetalBufferIndexOutput = 5,
    QwenMetalBufferIndexScratch = 6,
    QwenMetalBufferIndexState = 7,
};
#endif

#ifdef QWEN_METAL_COMMON

struct QwenMoERoutingParameters {
    uint tokenCount;
    uint expertCount;
    uint topK;
    uint reserved;
};

struct QwenMoERoutedParameters {
    uint hiddenSize;
    uint intermediateSize;
    uint valuesOffset;
    uint scalesOffset;
    uint biasesOffset;
    uint valuesRowStride;
    uint groupsPerRow;
    uint groupSize;
    uint scratchOffset;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct QwenMoEElementParameters {
    uint elementCount;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline float qwenMoePackedValue(
    device const uchar* values,
    device const bfloat* scales,
    device const bfloat* biases,
    uint row,
    uint column,
    uint valuesRowStride,
    uint groupsPerRow,
    uint groupSize,
    uint bitWidth
) {
    const uint metadataIndex = row * groupsPerRow + column / groupSize;
    float quantized;
    if (bitWidth == 4) {
        const uchar packed = values[row * valuesRowStride + column / 2];
        quantized = float((column & 1u) == 0u ? packed & 0x0Fu : packed >> 4);
    } else {
        quantized = float(values[row * valuesRowStride + column]);
    }
    return fma(float(scales[metadataIndex]), quantized, float(biases[metadataIndex]));
}

/// One serial correctness thread per token. Probabilities are FP32. Exact
/// equal-probability ties use ascending expert ID by TurboFieldfare contract.
kernel void qwen_moe_route_top8_fp32(
    constant QwenMoERoutingParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* logits [[buffer(QwenMetalBufferIndexInput)]],
    device uint* selectedIDs [[buffer(QwenMetalBufferIndexOutput)]],
    device float* selectedWeights [[buffer(QwenMetalBufferIndexScratch)]],
    device uint* status [[buffer(QwenMetalBufferIndexState)]],
    uint token [[thread_position_in_grid]]
) {
    if (token >= parameters.tokenCount) return;
    const uint base = token * parameters.expertCount;
    const uint selectedBase = token * parameters.topK;
    status[token] = 0u;

    float maximum = -INFINITY;
    for (uint expert = 0; expert < parameters.expertCount; ++expert) {
        const float value = logits[base + expert];
        if (!isfinite(value)) {
            status[token] = 1u;
            for (uint rank = 0; rank < parameters.topK; ++rank) {
                selectedIDs[selectedBase + rank] = UINT_MAX;
                selectedWeights[selectedBase + rank] = 0.0f;
            }
            return;
        }
        maximum = max(maximum, value);
    }

    float denominator = 0.0f;
    for (uint expert = 0; expert < parameters.expertCount; ++expert) {
        denominator += exp(logits[base + expert] - maximum);
    }
    if (!isfinite(denominator) || denominator <= 0.0f || parameters.topK != 8u) {
        status[token] = 1u;
        return;
    }

    uint topID[8];
    float topProbability[8];
    for (uint rank = 0; rank < 8u; ++rank) {
        topID[rank] = UINT_MAX;
        topProbability[rank] = -1.0f;
    }
    for (uint expert = 0; expert < parameters.expertCount; ++expert) {
        const float probability = exp(logits[base + expert] - maximum) / denominator;
        uint insertion = 8u;
        for (uint rank = 0; rank < 8u; ++rank) {
            if (probability > topProbability[rank]
                || (probability == topProbability[rank] && expert < topID[rank])) {
                insertion = rank;
                break;
            }
        }
        if (insertion < 8u) {
            for (uint rank = 7u; rank > insertion; --rank) {
                topID[rank] = topID[rank - 1u];
                topProbability[rank] = topProbability[rank - 1u];
            }
            topID[insertion] = expert;
            topProbability[insertion] = probability;
        }
    }

    float selectedSum = 0.0f;
    for (uint rank = 0; rank < 8u; ++rank) selectedSum += topProbability[rank];
    if (!isfinite(selectedSum) || selectedSum <= 0.0f) {
        status[token] = 1u;
        return;
    }
    for (uint rank = 0; rank < 8u; ++rank) {
        selectedIDs[selectedBase + rank] = topID[rank];
        selectedWeights[selectedBase + rank] = topProbability[rank] / selectedSum;
    }
}

kernel void qwen_moe_clear_fp32(
    constant QwenMoEElementParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    uint element [[thread_position_in_grid]]
) {
    if (element < parameters.elementCount) output[element] = 0.0f;
}

kernel void qwen_moe_routed_gate_up_int4(
    constant QwenMoERoutedParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* hidden [[buffer(QwenMetalBufferIndexInput)]],
    device const uchar* blob [[buffer(QwenMetalBufferIndexWeights)]],
    device float* activation [[buffer(QwenMetalBufferIndexScratch)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= parameters.intermediateSize) return;
    device const uchar* values = blob + parameters.valuesOffset;
    device const bfloat* scales = (device const bfloat*)(blob + parameters.scalesOffset);
    device const bfloat* biases = (device const bfloat*)(blob + parameters.biasesOffset);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint column = 0; column < parameters.hiddenSize; ++column) {
        const float input = hidden[column];
        gate = fma(qwenMoePackedValue(
            values, scales, biases, row, column,
            parameters.valuesRowStride, parameters.groupsPerRow,
            parameters.groupSize, 4u), input, gate);
        up = fma(qwenMoePackedValue(
            values, scales, biases, row + parameters.intermediateSize, column,
            parameters.valuesRowStride, parameters.groupsPerRow,
            parameters.groupSize, 4u), input, up);
    }
    const float sigmoid = 1.0f / (1.0f + exp(-gate));
    activation[parameters.scratchOffset + row] = gate * sigmoid * up;
}

kernel void qwen_moe_routed_down_add_int4(
    constant QwenMoERoutedParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* activation [[buffer(QwenMetalBufferIndexInput)]],
    device const uchar* blob [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* routingWeight [[buffer(QwenMetalBufferIndexState)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= parameters.hiddenSize) return;
    device const uchar* values = blob + parameters.valuesOffset;
    device const bfloat* scales = (device const bfloat*)(blob + parameters.scalesOffset);
    device const bfloat* biases = (device const bfloat*)(blob + parameters.biasesOffset);
    float projected = 0.0f;
    for (uint column = 0; column < parameters.intermediateSize; ++column) {
        projected = fma(qwenMoePackedValue(
            values, scales, biases, row, column,
            parameters.valuesRowStride, parameters.groupsPerRow,
            parameters.groupSize, 4u),
            activation[parameters.scratchOffset + column], projected);
    }
    output[row] += projected * routingWeight[0];
}

kernel void qwen_moe_affine_project(
    constant QwenMetalAffineLayout& layout [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* input [[buffer(QwenMetalBufferIndexInput)]],
    device const uchar* values [[buffer(QwenMetalBufferIndexWeights)]],
    device const bfloat* scales [[buffer(QwenMetalBufferIndexScales)]],
    device const bfloat* biases [[buffer(QwenMetalBufferIndexBiases)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= layout.rowCount) return;
    float projected = 0.0f;
    for (uint column = 0; column < layout.columnCount; ++column) {
        projected = fma(qwenMoePackedValue(
            values, scales, biases, row, column,
            layout.valuesRowStrideBytes, layout.groupsPerRow,
            layout.groupSize, layout.bitWidth), input[column], projected);
    }
    output[row] = projected;
}

kernel void qwen_moe_silu_multiply(
    constant QwenMoEElementParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* gate [[buffer(QwenMetalBufferIndexInput)]],
    device const float* up [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    uint element [[thread_position_in_grid]]
) {
    if (element >= parameters.elementCount) return;
    const float value = gate[element];
    output[element] = value / (1.0f + exp(-value)) * up[element];
}

kernel void qwen_moe_shared_epilogue(
    constant QwenMoEElementParameters& parameters [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* outputGate [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* sharedOutput [[buffer(QwenMetalBufferIndexScratch)]],
    uint element [[thread_position_in_grid]]
) {
    if (element >= parameters.elementCount) return;
    const float gate = 1.0f / (1.0f + exp(-outputGate[0]));
    output[element] += sharedOutput[element] * gate;
}


#endif // QWEN_METAL_COMMON

// Separate resident-BF16 routed path. Slot buffers contain exact BF16 bits,
// laid out as [2 * intermediate, hidden] and [hidden, intermediate]. Each
// dispatch owns disjoint output rows. The host submits ascending expert IDs.
// The original route rank still identifies its weight and activation slot.
struct QwenMoEBF16RoutedParameters {
    uint hiddenSize;
    uint intermediateSize;
    uint scratchOffset;
    uint reserved;
};

static inline float qwenMoeBF16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// Only the private BF16 library opts into the pinned vector exponential.
// The shared packed library compiles without this helper or new math settings.
static inline float qwenMoeBF16ActivationExp(float value) {
#ifdef QWEN_PINNED_SOURCE_EXP
    return qwenSourceExpFloatBitScale(value);
#else
    return precise::exp(value);
#endif
}

// Small or incomplete-block shapes retain the former ordered FMA path.
static inline float qwenMoeBF16OrderedDot(
    const device ushort* row, const device float* vector, uint columns
) {
    float sum = 0.0f;
    for (uint column = 0; column < columns; ++column) {
        sum = fma(qwenMoeBF16(row[column]), vector[column], sum);
    }
    return sum;
}

// Independently validated CPU BLAS order for rows >= 512 and complete
// 64-column blocks at widths >= 512. Safe math keeps each FP32 addition in
// this order. BF16 source storage and FP32 intermediates remain unchanged.
static inline float qwenMoeBF16SourceLargeDot(
    const device ushort* row, const device float* vector, uint columns
) {
    float partial[64] = {0.0f};
    for (uint column = 0; column < columns; ++column) {
        const uint stream = column & 63u;
        const float next = fma(qwenMoeBF16(row[column]), vector[column], partial[stream]);
        partial[stream] = next;
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
    // Keep this reduction's Inf/NaN for host rejection. Recomputing in a
    // different order could hide overflow that the source backend produces.
    return second + groups[3];
}

kernel void qwen_moe_routed_gate_up_bf16(
    constant QwenMoEBF16RoutedParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* hidden [[buffer(QwenMetalBufferIndexInput)]],
    device const ushort* gateUp [[buffer(QwenMetalBufferIndexWeights)]],
    device float* activation [[buffer(QwenMetalBufferIndexScratch)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.intermediateSize) return;
    const uint gateBase = row * p.hiddenSize;
    const uint upBase = (p.intermediateSize + row) * p.hiddenSize;
    // gateUp has 2 * intermediateSize rows in the official linear call.
    const bool largeShape = p.intermediateSize >= 256u && p.hiddenSize >= 512u
        && (p.hiddenSize & 63u) == 0u;
    const float gate = largeShape
        ? qwenMoeBF16SourceLargeDot(gateUp + gateBase, hidden, p.hiddenSize)
        : qwenMoeBF16OrderedDot(gateUp + gateBase, hidden, p.hiddenSize);
    const float up = largeShape
        ? qwenMoeBF16SourceLargeDot(gateUp + upBase, hidden, p.hiddenSize)
        : qwenMoeBF16OrderedDot(gateUp + upBase, hidden, p.hiddenSize);
    activation[p.scratchOffset + row] = gate / (1.0f + qwenMoeBF16ActivationExp(-gate)) * up;
}

kernel void qwen_moe_routed_down_add_bf16(
    constant QwenMoEBF16RoutedParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* activation [[buffer(QwenMetalBufferIndexInput)]],
    device const ushort* down [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* routingWeight [[buffer(QwenMetalBufferIndexState)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.hiddenSize) return;
    const uint rowBase = row * p.intermediateSize;
    const device float* vector = activation + p.scratchOffset;
    const bool largeShape = p.hiddenSize >= 512u && p.intermediateSize >= 512u
        && (p.intermediateSize & 63u) == 0u;
    const float projected = largeShape
        ? qwenMoeBF16SourceLargeDot(down + rowBase, vector, p.intermediateSize)
        : qwenMoeBF16OrderedDot(down + rowBase, vector, p.intermediateSize);
    // The official eager path materializes the weighted expert tensor before
    // index_add_. A volatile FP32 product preserves that rounding boundary.
    volatile float contribution = projected * routingWeight[0];
    output[row] = output[row] + contribution;
}


struct QwenMoEBF16ElementParameters {
    uint elementCount;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

kernel void qwen_moe_silu_multiply_bf16(
    constant QwenMoEBF16ElementParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* gate [[buffer(QwenMetalBufferIndexInput)]],
    device const float* up [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    uint element [[thread_position_in_grid]]) {
    if (element >= p.elementCount) return;
    const float value = gate[element];
    output[element] = value / (1.0f + qwenMoeBF16ActivationExp(-value)) * up[element];
}

kernel void qwen_moe_shared_epilogue_bf16(
    constant QwenMoEBF16ElementParameters& p [[buffer(QwenMetalBufferIndexParameters)]],
    device const float* outputGate [[buffer(QwenMetalBufferIndexWeights)]],
    device float* output [[buffer(QwenMetalBufferIndexOutput)]],
    device const float* sharedOutput [[buffer(QwenMetalBufferIndexScratch)]],
    uint element [[thread_position_in_grid]]) {
    if (element >= p.elementCount) return;
#ifdef QWEN_PINNED_SOURCE_SCALAR_EXP
    // The one-element source gate uses scalar expf and a separately rounded
    // reciprocal, unlike the expert vector activation exponential.
    const float denominator = 1.0f + qwenSourceScalarExpFloat(-outputGate[0]);
    const float gate = qwenSourceScalarPositiveReciprocal(denominator);
#else
    const float gate = 1.0f / (1.0f + precise::exp(-outputGate[0]));
#endif
    // Shared weighting is another materialized tensor in the official forward.
    volatile float contribution = sharedOutput[element] * gate;
    output[element] = output[element] + contribution;
}
