#ifndef TURBOFIELDFARE_JPEG_H
#define TURBOFIELDFARE_JPEG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    TF_JPEG_OK = 0,
    TF_JPEG_INVALID_INPUT = 1,
    TF_JPEG_UNSUPPORTED_COLORSPACE = 2,
    TF_JPEG_LIMIT_EXCEEDED = 3,
    TF_JPEG_ALLOCATION_FAILED = 4,
    TF_JPEG_DECODE_FAILED = 5
};

typedef struct {
    uint8_t *pixels;
    uint32_t width;
    uint32_t height;
    size_t byte_count;
} tf_jpeg_rgb;

/* Full-resolution, encoded-orientation RGB, without ICC transforms or EXIF
 * rotation. Supports 8-bit gray, RGB/YCbCr and CMYK/YCCK JPEG. Result must be
 * empty on entry. On failure it is empty. Error text is optional and bounded.
 * The caller retains encoded bytes for this synchronous call and releases a
 * successful result with tf_jpeg_rgb_free(). Error jumps never leave C. */
int tf_jpeg_decode_rgb(const uint8_t *encoded, size_t encoded_bytes,
                       size_t maximum_pixels, size_t maximum_output_bytes,
                       tf_jpeg_rgb *result, char *error, size_t error_capacity);
void tf_jpeg_rgb_free(tf_jpeg_rgb *result);

#ifdef __cplusplus
}
#endif
#endif
