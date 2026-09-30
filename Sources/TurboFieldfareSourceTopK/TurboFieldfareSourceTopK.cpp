#include "TurboFieldfareSourceTopK.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// std selection permits different ordering for equal values. Any libc++
// change requires repeating the pinned-Torch ordered-ID qualification before
// updating this version, in both Debug and Release production builds.
static_assert(_LIBCPP_VERSION == 210106,
              "Requalify source Top-8 against pinned Torch before changing libc++");
#ifdef _LIBCPP_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY
#error "Source Top-8 requires non-randomized libc++ tie ordering"
#endif

bool tf_qwen_source_top8(const float *probabilities, int32_t count, int32_t *indices) {
    if (!probabilities || !indices || count != 256) return false;
    for (int32_t index = 0; index < count; ++index) {
        if (!std::isfinite(probabilities[index])) return false;
    }
    try {
        // Pinned Torch 2.10, commit 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0,
        // ATen/native/TopKImpl.h: K*64 > N selects nth_element, then sorts
        // [begin, begin+K-1). Compare probability alone, never expert ID.
        // Native libc++ tie ordering is qualified against the pinned wheel;
        // the C++ standard itself does not promise portable tie ordering.
        using Entry = std::pair<float, int64_t>;
        std::vector<Entry> queue(count);
        for (int64_t index = 0; index < count; ++index) {
            queue[index] = {probabilities[index], index};
        }
        const auto greater = [](const Entry& x, const Entry& y) {
            return (std::isnan(x.first) && !std::isnan(y.first)) || x.first > y.first;
        };
        std::nth_element(queue.begin(), queue.begin() + 7, queue.end(), greater);
        std::sort(queue.begin(), queue.begin() + 7, greater);
        for (int32_t rank = 0; rank < 8; ++rank) {
            indices[rank] = static_cast<int32_t>(queue[rank].second);
        }
        return true;
    } catch (...) {
        // No C++ exception may cross the Swift-facing C boundary.
        return false;
    }
}
