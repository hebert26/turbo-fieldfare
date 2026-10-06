// CMYK conversion follows Pillow10.3.0, Copyright (c)1997-2005 Secret Labs AB
// and Copyright (c)1995-1997 Fredrik Lundh. See the retained license at
// ThirdParty/libjpeg-turbo/Pillow-10.3.0-LICENSE and SOURCE-PIN.json.
#include "TurboFieldfareJPEG.h"

#include <limits.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "jpeglib.h"
#include "jerror.h"

typedef struct {
    struct jpeg_error_mgr base;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
} tf_jpeg_error;

typedef struct {
    struct jpeg_decompress_struct decoder;
    tf_jpeg_error error;
    uint8_t *pixels;
    uint8_t *cmyk_row;
    int created;
} tf_jpeg_state;

static void tf_jpeg_fail(j_common_ptr decoder) {
    tf_jpeg_error *error = (tf_jpeg_error *)decoder->err;
    (*error->base.format_message)(decoder, error->message);
    longjmp(error->jump, 1);
}

static void tf_jpeg_message(j_common_ptr decoder, int level) {
    if (level < 0) {
        // jpeg_mem_src otherwise inserts a synthetic EOI for truncated input.
        if (decoder->err->msg_code == JWRN_JPEG_EOF) tf_jpeg_fail(decoder);
        decoder->err->num_warnings++;
    }
}

static int tf_jpeg_return(tf_jpeg_state *state, int status, const char *message,
                          char *error, size_t error_capacity) {
    if (error && error_capacity) snprintf(error, error_capacity, "%s", message);
    if (state) {
        if (state->created) jpeg_destroy_decompress(&state->decoder);
        free(state->pixels);
        free(state->cmyk_row);
        free(state);
    }
    return status;
}

static uint8_t tf_pillow_cmyk_channel(uint8_t decoded_channel, uint8_t decoded_k) {
    // Pillow10.3 JpegImagePlugin uses CMYK;I for every four-component JPEG.
    // Its unpacker inverts all four bytes; Convert.c then uses rounded /255.
    unsigned int inverted_channel = 255u - decoded_channel;
    unsigned int product = inverted_channel * decoded_k + 128u;
    return (uint8_t)(decoded_k - ((product + (product >> 8)) >> 8));
}

int tf_jpeg_decode_rgb(const uint8_t *encoded, size_t encoded_bytes,
                       size_t maximum_pixels, size_t maximum_output_bytes,
                       tf_jpeg_rgb *result, char *error, size_t error_capacity) {
    if (error && error_capacity) error[0] = '\0';
    if (!result) return tf_jpeg_return(NULL, TF_JPEG_INVALID_INPUT,
                                      "Missing JPEG result", error, error_capacity);
    memset(result, 0, sizeof(*result));
    if (!encoded || encoded_bytes < 2 || !maximum_pixels || !maximum_output_bytes)
        return tf_jpeg_return(NULL, TF_JPEG_INVALID_INPUT,
                              "Empty JPEG input or bounds", error, error_capacity);
    if (encoded_bytes > ULONG_MAX)
        return tf_jpeg_return(NULL, TF_JPEG_LIMIT_EXCEEDED,
                              "JPEG input byte count overflow", error, error_capacity);
    tf_jpeg_state *state = calloc(1, sizeof(*state));
    if (!state) return tf_jpeg_return(NULL, TF_JPEG_ALLOCATION_FAILED,
                                     "JPEG decoder allocation failed", error, error_capacity);
    state->decoder.err = jpeg_std_error(&state->error.base);
    state->error.base.error_exit = tf_jpeg_fail;
    state->error.base.emit_message = tf_jpeg_message;
    // All state mutated after setjmp is heap-owned, so error cleanup does not
    // inspect indeterminate C automatic variables after a longjmp.
    if (setjmp(state->error.jump))
        return tf_jpeg_return(state, TF_JPEG_DECODE_FAILED,
                              state->error.message, error, error_capacity);
    state->created = 1;
    jpeg_create_decompress(&state->decoder);
    jpeg_mem_src(&state->decoder, encoded, (unsigned long)encoded_bytes);
    if (jpeg_read_header(&state->decoder, TRUE) != JPEG_HEADER_OK)
        return tf_jpeg_return(state, TF_JPEG_DECODE_FAILED,
                              "JPEG header is incomplete", error, error_capacity);
    if (state->decoder.data_precision != 8)
        return tf_jpeg_return(state, TF_JPEG_UNSUPPORTED_COLORSPACE,
                              "JPEG sample precision is not eight bits", error, error_capacity);
    int components = state->decoder.num_components;
    if (components != 1 && components != 3 && components != 4)
        return tf_jpeg_return(state, TF_JPEG_UNSUPPORTED_COLORSPACE,
                              "Unsupported JPEG component count", error, error_capacity);
    size_t width = state->decoder.image_width, height = state->decoder.image_height;
    if (!width || !height || width > UINT32_MAX || height > UINT32_MAX ||
        height > SIZE_MAX / width || width * height > maximum_pixels ||
        width * height > SIZE_MAX / 3 || width * height * 3 > maximum_output_bytes)
        return tf_jpeg_return(state, TF_JPEG_LIMIT_EXCEEDED,
                              "JPEG dimensions exceed admitted bounds", error, error_capacity);
    size_t output_bytes = width * height * 3;
    state->decoder.scale_num = 1;
    state->decoder.scale_denom = 1;
    state->decoder.dct_method = JDCT_ISLOW;
    state->decoder.do_fancy_upsampling = TRUE;
    state->decoder.do_block_smoothing = TRUE;
    state->decoder.quantize_colors = FALSE;
    // Keep header-inferred input color space, including Adobe YCCK markers.
    state->decoder.out_color_space = components == 4 ? JCS_CMYK : JCS_RGB;
    // Bounds virtual/progressive backing arrays. Dimensions/output are checked
    // separately, since this setting does not cap every library allocation.
    state->decoder.mem->max_memory_to_use = maximum_output_bytes > LONG_MAX
        ? LONG_MAX : (long)maximum_output_bytes;
    if (!jpeg_start_decompress(&state->decoder) ||
        state->decoder.output_width != width || state->decoder.output_height != height ||
        state->decoder.output_components != (components == 4 ? 4 : 3))
        return tf_jpeg_return(state, TF_JPEG_DECODE_FAILED,
                              "JPEG output geometry differs from header", error, error_capacity);
    state->pixels = malloc(output_bytes);
    if (!state->pixels) return tf_jpeg_return(state, TF_JPEG_ALLOCATION_FAILED,
                                             "JPEG RGB allocation failed", error, error_capacity);
    if (components == 4) {
        if (width > SIZE_MAX / 4)
            return tf_jpeg_return(state, TF_JPEG_LIMIT_EXCEEDED,
                                  "JPEG CMYK row byte count overflow", error, error_capacity);
        state->cmyk_row = malloc(width * 4);
        if (!state->cmyk_row) return tf_jpeg_return(state, TF_JPEG_ALLOCATION_FAILED,
                                                  "JPEG CMYK row allocation failed", error, error_capacity);
    }
    while (state->decoder.output_scanline < state->decoder.output_height) {
        size_t row_index = state->decoder.output_scanline;
        uint8_t *destination = state->pixels + row_index * width * 3;
        JSAMPROW row = components == 4 ? state->cmyk_row : destination;
        if (jpeg_read_scanlines(&state->decoder, &row, 1) != 1)
            return tf_jpeg_return(state, TF_JPEG_DECODE_FAILED,
                                  "JPEG scanline is incomplete", error, error_capacity);
        if (components == 4) {
            for (size_t x = 0; x < width; x++) {
                const uint8_t *cmyk = state->cmyk_row + x * 4;
                for (size_t channel = 0; channel < 3; channel++)
                    destination[x * 3 + channel] = tf_pillow_cmyk_channel(cmyk[channel], cmyk[3]);
            }
        }
    }
    if (!jpeg_finish_decompress(&state->decoder))
        return tf_jpeg_return(state, TF_JPEG_DECODE_FAILED,
                              "JPEG stream is incomplete", error, error_capacity);
    result->pixels = state->pixels;
    result->width = (uint32_t)width;
    result->height = (uint32_t)height;
    result->byte_count = output_bytes;
    state->pixels = NULL;
    return tf_jpeg_return(state, TF_JPEG_OK, "", error, error_capacity);
}

void tf_jpeg_rgb_free(tf_jpeg_rgb *result) {
    if (!result) return;
    free(result->pixels);
    memset(result, 0, sizeof(*result));
}
