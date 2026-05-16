#include "fpv_read_state.hpp"

#include "fpv_frame_store.hpp"

#include <chrono>
#include <stdexcept>

namespace excavator_fpv {
namespace {

std::uint64_t systemNowNs() {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return static_cast<std::uint64_t>(ns.count());
}

void fillPlaceholder(FpvReadStateImage* out, const FpvReadStateConfig& cfg, std::uint64_t frame_id) {
    const int width = cfg.placeholder_width;
    const int height = cfg.placeholder_height;
    out->timestamp_ns = systemNowNs();
    out->receive_time_ns = out->timestamp_ns;
    out->source = "bridge_placeholder_fpv";
    out->width = width;
    out->height = height;
    out->rgb.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U, 0);
    const std::uint8_t frame_r = static_cast<std::uint8_t>((frame_id * 5U) % 255U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t idx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) *
                3U;
            out->rgb[idx + 0U] = frame_r;
            out->rgb[idx + 1U] =
                static_cast<std::uint8_t>((x * 255) / std::max(1, width - 1));
            out->rgb[idx + 2U] =
                static_cast<std::uint8_t>((y * 255) / std::max(1, height - 1));
        }
    }
}

}  // namespace

FpvReadStateImage acquireFpvImage(const FpvReadStateConfig& cfg, std::uint64_t placeholder_frame_id) {
    const bool want_shm = cfg.source == "shm" || cfg.source == "auto";
    const bool allow_placeholder = cfg.source == "placeholder" || cfg.source == "auto";

    if (want_shm) {
        try {
            FpvFrameStoreReader reader(cfg.shm_name);
            const std::uint64_t now = systemNowNs();
            if (reader.isFresh(now, cfg.max_stale_ms)) {
                FpvFrameView view{};
                FpvReadStateImage out{};
                if (reader.readLatest(&view, &out.rgb)) {
                    out.timestamp_ns = view.timestamp_ns;
                    out.receive_time_ns = view.receive_time_ns;
                    out.source = "ros2_compressed_fpv";
                    out.width = view.width;
                    out.height = view.height;
                    return out;
                }
            }
        } catch (const std::exception&) {
            // SHM 未就绪
        }
    }

    if (!allow_placeholder) {
        throw std::runtime_error("fpv shm frame unavailable and placeholder disabled");
    }

    FpvReadStateImage out{};
    fillPlaceholder(&out, cfg, placeholder_frame_id);
    return out;
}

}  // namespace excavator_fpv
