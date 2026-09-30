#ifndef TURBO_FIELDFARE_SOURCE_TOP_K_H
#define TURBO_FIELDFARE_SOURCE_TOP_K_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fixed original-source router geometry. Writes eight IDs on success.
bool tf_qwen_source_top8(const float *probabilities, int32_t count, int32_t *indices);

#ifdef __cplusplus
}
#endif

#endif
