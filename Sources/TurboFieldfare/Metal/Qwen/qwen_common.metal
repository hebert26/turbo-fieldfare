#ifndef QWEN_METAL_COMMON
#define QWEN_METAL_COMMON

#include <metal_stdlib>
using namespace metal;

typedef uchar QwenMetalPackedValue;
typedef bfloat QwenMetalAffineMetadata;
typedef uint QwenMetalDimension;

enum QwenMetalBufferIndex : uint {
    QwenMetalBufferIndexParameters = 0,
    QwenMetalBufferIndexInput = 1,
    QwenMetalBufferIndexWeights = 2,
    QwenMetalBufferIndexScales = 3,
    QwenMetalBufferIndexBiases = 4,
    QwenMetalBufferIndexOutput = 5,
    QwenMetalBufferIndexScratch = 6,
    QwenMetalBufferIndexState = 7,
};

enum QwenMetalABIConstant : uint {
    QwenMetalPackedValueByteWidth = 1,
    QwenMetalAffineMetadataByteWidth = 2,
    QwenMetalDimensionByteWidth = 4,
    QwenMetalAffineGroupSize = 64,
};

struct QwenMetalAffineLayout {
    QwenMetalDimension rowCount;
    QwenMetalDimension columnCount;
    QwenMetalDimension valuesRowStrideBytes;
    QwenMetalDimension metadataRowStrideBytes;
    QwenMetalDimension groupsPerRow;
    QwenMetalDimension bitWidth;
    QwenMetalDimension groupSize;
    QwenMetalDimension reserved;
};

inline QwenMetalDimension qwenMetalValueByteOffset(
    constant QwenMetalAffineLayout& layout,
    QwenMetalDimension row,
    QwenMetalDimension column
) {
    QwenMetalDimension wholeBytes = (column / 8) * layout.bitWidth;
    QwenMetalDimension partialByte = ((column % 8) * layout.bitWidth) / 8;
    return row * layout.valuesRowStrideBytes + wholeBytes + partialByte;
}

inline QwenMetalDimension qwenMetalMetadataIndex(
    constant QwenMetalAffineLayout& layout,
    QwenMetalDimension row,
    QwenMetalDimension column
) {
    return row * layout.groupsPerRow + column / layout.groupSize;
}

#endif
