#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace excavator_fpv {

constexpr int kFpvMaxWidth = 640;
constexpr int kFpvMaxHeight = 480;
constexpr int kFpvMaxBytes = kFpvMaxWidth * kFpvMaxHeight * 3;

/** 解码后的 RGB 帧视图（只读）。 */
struct FpvFrameView {
    std::uint64_t timestamp_ns{0};
    std::uint64_t receive_time_ns{0};
    std::uint32_t sequence{0};
    int width{0};
    int height{0};
    const std::uint8_t* rgb{nullptr};
    std::size_t rgb_size{0};
};

/** 进程间共享最新 FPV 帧（ROS 订阅进程写，bridge 读）。 */
class FpvFrameStoreWriter {
public:
    explicit FpvFrameStoreWriter(std::string shm_name = "excavator_fpv_v1");
    ~FpvFrameStoreWriter();

    FpvFrameStoreWriter(const FpvFrameStoreWriter&) = delete;
    FpvFrameStoreWriter& operator=(const FpvFrameStoreWriter&) = delete;

    bool writeRgb(
        const std::uint8_t* rgb,
        int width,
        int height,
        std::uint64_t timestamp_ns,
        std::uint64_t receive_time_ns);

    const std::string& shmName() const { return shm_name_; }

private:
    std::string shm_name_;
    void* mapped_{nullptr};
    std::size_t mapped_size_{0};
    int shm_fd_{-1};
};

class FpvFrameStoreReader {
public:
    explicit FpvFrameStoreReader(std::string shm_name = "excavator_fpv_v1");
    ~FpvFrameStoreReader();

    FpvFrameStoreReader(const FpvFrameStoreReader&) = delete;
    FpvFrameStoreReader& operator=(const FpvFrameStoreReader&) = delete;

    /** 读取最新帧；无数据或尺寸非法返回 false。 */
    bool readLatest(FpvFrameView* out, std::vector<std::uint8_t>* rgb_copy) const;

    /** 帧是否在 max_age_ms 内更新。 */
    bool isFresh(std::uint64_t now_ns, int max_age_ms) const;

private:
    std::string shm_name_;
    void* mapped_{nullptr};
    std::size_t mapped_size_{0};
    int shm_fd_{-1};
};

}  // namespace excavator_fpv
