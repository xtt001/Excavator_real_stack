#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace excavator_fpv {

/** bridge read_state 用 FPV 配置（可由 CLI 或环境变量填充）。 */
struct FpvReadStateConfig {
    std::string source{"auto"};  // auto | shm | placeholder
    std::string shm_name{"excavator_fpv_v1"};
    int max_stale_ms{500};
    int placeholder_width{160};
    int placeholder_height{120};
};

/** 供 bridge 转成 JSON payload 的 FPV 一帧。 */
struct FpvReadStateImage {
    std::uint64_t timestamp_ns{0};
    std::uint64_t receive_time_ns{0};
    std::string source;
    int width{0};
    int height{0};
    std::vector<std::uint8_t> rgb;
};

/** 从 SHM 或 placeholder 取一帧；source=shm 且无帧时抛异常。 */
FpvReadStateImage acquireFpvImage(const FpvReadStateConfig& cfg, std::uint64_t placeholder_frame_id);

}  // namespace excavator_fpv
