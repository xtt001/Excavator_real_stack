#include <can/internal/imu_canlib.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#if defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace canlib {
namespace {

inline constexpr std::size_t kCanPayloadSize = kImuCanPayloadBytes;
static_assert(kCanPayloadSize == kImuCanPayloadBytes, "imu payload");
inline constexpr std::uint64_t kImuShmMagic = 0x494D555F43414E31ULL;
inline constexpr auto kLoopPeriod = std::chrono::milliseconds(20);  // 50Hz
inline constexpr auto kOfflineTimeout = std::chrono::milliseconds(100);
// The four Daoyuan IMUs produce roughly 72 classic CAN frames per 20 ms tick.
// Never drain without a bound: if parsing falls behind the producer, an
// unbounded loop may never reach EAGAIN and publish_shm() will starve forever.
inline constexpr std::size_t kSocketCanMaxFramesPerTick = 512U;

#if defined(__linux__)
inline constexpr std::uint32_t kUsbCanDeviceTypeUsbcCanIi = 4U;
inline constexpr std::uint32_t kUsbCanDefaultDeviceIndex = 0U;
inline constexpr std::uint32_t kUsbCanDefaultChannelIndex = 0U;
inline constexpr std::uint32_t kUsbCanDefaultBitrate = 250000U;
inline constexpr std::size_t kUsbCanReceiveBatch = 256U;

struct UsbCanInitConfig {
    std::uint32_t AccCode{0};
    std::uint32_t AccMask{0xFFFFFFFFU};
    std::uint32_t Reserved{0};
    std::uint8_t Filter{1};
    std::uint8_t Timing0{0};
    std::uint8_t Timing1{0};
    std::uint8_t Mode{0};
};

struct UsbCanObj {
    std::uint32_t ID{0};
    std::uint32_t TimeStamp{0};
    std::uint8_t TimeFlag{0};
    std::uint8_t SendType{0};
    std::uint8_t RemoteFlag{0};
    std::uint8_t ExternFlag{0};
    std::uint8_t DataLen{0};
    std::uint8_t Data[8]{};
    std::uint8_t Reserved[3]{};
};

static_assert(sizeof(UsbCanInitConfig) == 16U, "USB-CAN init config ABI size");
static_assert(sizeof(UsbCanObj) == 24U, "USB-CAN object ABI size");

struct UsbCanInterfaceConfig {
    bool enabled{false};
    std::uint32_t device_index{kUsbCanDefaultDeviceIndex};
    std::uint32_t channel_index{kUsbCanDefaultChannelIndex};
    std::uint32_t bitrate{kUsbCanDefaultBitrate};
    std::string error{};
};

struct UsbCanApi {
    using OpenDeviceFn = std::uint32_t (*)(std::uint32_t, std::uint32_t, std::uint32_t);
    using CloseDeviceFn = std::uint32_t (*)(std::uint32_t, std::uint32_t);
    using InitCanFn = std::uint32_t (*)(std::uint32_t, std::uint32_t, std::uint32_t, UsbCanInitConfig*);
    using StartCanFn = std::uint32_t (*)(std::uint32_t, std::uint32_t, std::uint32_t);
    using ResetCanFn = std::uint32_t (*)(std::uint32_t, std::uint32_t, std::uint32_t);
    using ClearBufferFn = std::uint32_t (*)(std::uint32_t, std::uint32_t, std::uint32_t);
    using GetReceiveNumFn = std::uint32_t (*)(std::uint32_t, std::uint32_t, std::uint32_t);
    using ReceiveFn = std::uint32_t (*)(std::uint32_t, std::uint32_t, std::uint32_t, UsbCanObj*, std::uint32_t, int);

    void* handle{nullptr};
    OpenDeviceFn open_device{nullptr};
    CloseDeviceFn close_device{nullptr};
    InitCanFn init_can{nullptr};
    StartCanFn start_can{nullptr};
    ResetCanFn reset_can{nullptr};
    ClearBufferFn clear_buffer{nullptr};
    GetReceiveNumFn get_receive_num{nullptr};
    ReceiveFn receive{nullptr};

    void unload() {
        if (handle) {
            dlclose(handle);
            handle = nullptr;
        }
        open_device = nullptr;
        close_device = nullptr;
        init_can = nullptr;
        start_can = nullptr;
        reset_can = nullptr;
        clear_buffer = nullptr;
        get_receive_num = nullptr;
        receive = nullptr;
    }
};
#endif

std::string normalize_shm_name(const std::string& name) {
    if (name.empty()) return "/imu_canlib_shm";
    if (name.front() == '/') return name;
    return "/" + name;
}

std::uint16_t get_u16_le(const std::array<std::uint8_t, kCanPayloadSize>& in, std::size_t idx) {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(in[idx]) |
                                      static_cast<std::uint16_t>(in[idx + 1] << 8));
}

std::int16_t get_i16_le(const std::array<std::uint8_t, kCanPayloadSize>& in, std::size_t idx) {
    return static_cast<std::int16_t>(get_u16_le(in, idx));
}

float get_f32_le(const std::array<std::uint8_t, kCanPayloadSize>& in, std::size_t idx) {
    float out = 0.0F;
    std::memcpy(&out, &in[idx], sizeof(float));
    return out;
}

std::uint32_t get_u32_le(const std::array<std::uint8_t, kCanPayloadSize>& in, std::size_t idx) {
    return static_cast<std::uint32_t>(in[idx]) | (static_cast<std::uint32_t>(in[idx + 1]) << 8) |
           (static_cast<std::uint32_t>(in[idx + 2]) << 16) | (static_cast<std::uint32_t>(in[idx + 3]) << 24);
}

float get_f32_packet_le(const std::array<std::uint8_t, kDaoyuanImuPacketBytes>& in, std::size_t idx) {
    float out = 0.0F;
    std::memcpy(&out, &in[idx], sizeof(float));
    return out;
}

std::uint32_t get_u32_packet_le(const std::array<std::uint8_t, kDaoyuanImuPacketBytes>& in, std::size_t idx) {
    return static_cast<std::uint32_t>(in[idx]) | (static_cast<std::uint32_t>(in[idx + 1]) << 8) |
           (static_cast<std::uint32_t>(in[idx + 2]) << 16) | (static_cast<std::uint32_t>(in[idx + 3]) << 24);
}

std::uint64_t now_ns() {
    const auto t = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t).count());
}

#if defined(__linux__)
std::string lowercase_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool parse_u32_component(const std::string& raw, std::uint32_t& out, std::uint32_t max_value) {
    if (raw.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(raw.c_str(), &end, 10);
    if (errno != 0 || end == raw.c_str() || *end != '\0' || value > max_value) {
        return false;
    }
    out = static_cast<std::uint32_t>(value);
    return true;
}

bool usbcan_baud_from_bitrate(std::uint32_t bitrate, std::uint32_t& baud) {
    switch (bitrate) {
        case 1000000U:
            baud = 0x1400U;
            return true;
        case 500000U:
            baud = 0x1C00U;
            return true;
        case 250000U:
            baud = 0x1C01U;
            return true;
        case 125000U:
            baud = 0x1C03U;
            return true;
        default:
            return false;
    }
}

UsbCanInterfaceConfig parse_usbcan_interface(const std::string& can_if_name) {
    UsbCanInterfaceConfig cfg{};
    const std::string lower = lowercase_ascii(can_if_name);
    constexpr const char* kPrefix = "usbcan";
    constexpr std::size_t kPrefixLen = 6U;
    if (lower.rfind(kPrefix, 0) != 0) {
        return cfg;
    }

    cfg.enabled = true;
    std::string spec = lower.substr(kPrefixLen);
    const std::size_t at_pos = spec.find('@');
    if (at_pos != std::string::npos) {
        const std::string bitrate_raw = spec.substr(at_pos + 1U);
        spec = spec.substr(0, at_pos);
        if (!parse_u32_component(bitrate_raw, cfg.bitrate, 1000000U)) {
            cfg.error = "invalid usbcan bitrate: " + bitrate_raw;
            return cfg;
        }
    }

    if (spec.empty()) {
        return cfg;
    }

    auto parse_channel = [&](const std::string& raw) -> bool {
        std::uint32_t channel = 0;
        if (!parse_u32_component(raw, channel, 1U)) {
            cfg.error = "invalid usbcan channel index: " + raw;
            return false;
        }
        cfg.channel_index = channel;
        return true;
    };

    auto parse_device = [&](const std::string& raw) -> bool {
        std::uint32_t device = 0;
        if (!parse_u32_component(raw, device, 255U)) {
            cfg.error = "invalid usbcan device index: " + raw;
            return false;
        }
        cfg.device_index = device;
        return true;
    };

    if (spec.front() == ':') {
        const std::string body = spec.substr(1U);
        const std::size_t colon = body.find(':');
        if (colon == std::string::npos) {
            parse_device(body);
            return cfg;
        }
        if (!parse_device(body.substr(0, colon))) return cfg;
        if (!parse_channel(body.substr(colon + 1U))) return cfg;
        return cfg;
    }

    const std::size_t colon = spec.find(':');
    if (colon == std::string::npos) {
        parse_device(spec);
        return cfg;
    }
    if (!parse_device(spec.substr(0, colon))) return cfg;
    parse_channel(spec.substr(colon + 1U));
    return cfg;
}

template <typename T>
bool load_usbcan_symbol(void* handle, const char* name, T& out, std::string& error) {
    dlerror();
    void* sym = dlsym(handle, name);
    const char* dl_error = dlerror();
    if (dl_error != nullptr || sym == nullptr) {
        error = std::string("dlsym(") + name + ") 失败: " + (dl_error ? dl_error : "symbol not found");
        return false;
    }
    out = reinterpret_cast<T>(sym);
    return true;
}

bool load_usbcan_api(UsbCanApi& api, std::string& error) {
    api.unload();
    const char* candidates[] = {
        "libusbcan.so",
        "/usr/local/lib/libusbcan.so",
        "/opt/usbcan_ii_libusb_aarch64/libusbcan.so",
    };
    for (const char* candidate : candidates) {
        api.handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (api.handle) break;
    }
    if (!api.handle) {
        const char* dl_error = dlerror();
        error = std::string("dlopen(libusbcan.so) 失败: ") + (dl_error ? dl_error : "unknown error");
        return false;
    }

    return load_usbcan_symbol(api.handle, "VCI_OpenDevice", api.open_device, error) &&
           load_usbcan_symbol(api.handle, "VCI_CloseDevice", api.close_device, error) &&
           load_usbcan_symbol(api.handle, "VCI_InitCAN", api.init_can, error) &&
           load_usbcan_symbol(api.handle, "VCI_StartCAN", api.start_can, error) &&
           load_usbcan_symbol(api.handle, "VCI_ResetCAN", api.reset_can, error) &&
           load_usbcan_symbol(api.handle, "VCI_ClearBuffer", api.clear_buffer, error) &&
           load_usbcan_symbol(api.handle, "VCI_GetReceiveNum", api.get_receive_num, error) &&
           load_usbcan_symbol(api.handle, "VCI_Receive", api.receive, error);
}
#endif

bool daoyuan_slot_for_can_id(std::uint16_t can_id, std::size_t& slot) noexcept {
    // New Daoyuan IMU labels on the machine:
    //   swing=0x123, boom=0x121, stick=0x124, bucket=0x122.
    // The rest of the stack expects devices[0..3] = bucket, stick, boom, swing.
    switch (can_id) {
        case 0x122U:
            slot = 0U;
            return true;
        case 0x124U:
            slot = 1U;
            return true;
        case 0x121U:
            slot = 2U;
            return true;
        case 0x123U:
            slot = 3U;
            return true;
        default:
            return false;
    }
}

bool daoyuan_packet_header(const std::array<std::uint8_t, kCanPayloadSize>& payload) noexcept {
    return payload[0] == 0xABU && payload[1] == 0x54U && payload[2] == 0x65U &&
           payload[3] == 0x00U && payload[4] == 0x35U && payload[5] == 0x00U;
}

bool all_finite(std::initializer_list<float> values) noexcept {
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

// 协议欧拉角刻度为 0.01°；折到 [-180,180] 后转弧度
float euler_deg_to_rad_pm_pi(float deg) {
    while (deg > 180.0F) deg -= 360.0F;
    while (deg < -180.0F) deg += 360.0F;
    constexpr float kDegToRad = 3.14159265f / 180.0f;
    return deg * kDegToRad;
}

void builtin_imu_apply_can_payload_to_partials(std::uint16_t can_id,
                                                 const std::array<std::uint8_t, kCanPayloadSize>& payload,
                                                 std::array<ImuRxAccumulator, kImuDeviceCount>& partials) {
    const std::uint8_t device_addr = static_cast<std::uint8_t>(can_id & 0x07U);
    // 兼容两种设备地址编码：
    // - 0..3（零基）
    // - 1..4（一基）
    static int addr_mode = 0;  // 0:未知 1:零基 2:一基
    if (device_addr == 0U) addr_mode = 1;
    if (device_addr == kImuDeviceCount) addr_mode = 2;
    if (device_addr > kImuDeviceCount) return;

    std::size_t idx = 0;
    if (addr_mode == 1) {
        if (device_addr >= kImuDeviceCount) return;
        idx = static_cast<std::size_t>(device_addr);
    } else if (addr_mode == 2) {
        if (device_addr == 0U) return;
        idx = static_cast<std::size_t>(device_addr - 1U);
    } else {
        // 未探测到模式时保持历史默认（一基）行为。
        if (device_addr == 0U) return;
        idx = static_cast<std::size_t>(device_addr - 1U);
    }
    const std::uint8_t cmd = static_cast<std::uint8_t>((can_id >> 3U) & 0x07U);
    auto& pf = partials[idx];

    const std::uint64_t rx_ns = now_ns();
    pf.last_rx_ns = rx_ns;
    switch (cmd) {
        case 0x00:  // 欧拉角：roll/pitch 为有符号 0.01°，yaw 为无符号 0.01°。
            pf.roll_raw_deg = static_cast<float>(get_i16_le(payload, 0)) * 0.01F;
            pf.pitch_raw_deg = static_cast<float>(get_i16_le(payload, 2)) * 0.01F;
            pf.yaw_raw_deg = static_cast<float>(get_u16_le(payload, 4)) * 0.01F;
            pf.roll_rad = euler_deg_to_rad_pm_pi(pf.roll_raw_deg);
            pf.pitch_rad = euler_deg_to_rad_pm_pi(pf.pitch_raw_deg);
            pf.yaw_rad = euler_deg_to_rad_pm_pi(pf.yaw_raw_deg);
            pf.has_euler = true;
            break;
        case 0x01:  // 角速率
            pf.gyro_x_dps = static_cast<float>(get_i16_le(payload, 0)) * 0.1F;
            pf.gyro_y_dps = static_cast<float>(get_i16_le(payload, 2)) * 0.1F;
            pf.gyro_z_dps = static_cast<float>(get_i16_le(payload, 4)) * 0.1F;
            pf.has_gyro = true;
            break;
        case 0x02:  // 加速度
            pf.accel_x_mps2 = static_cast<float>(get_i16_le(payload, 0)) * 0.1F;
            pf.accel_y_mps2 = static_cast<float>(get_i16_le(payload, 2)) * 0.1F;
            pf.accel_z_mps2 = static_cast<float>(get_i16_le(payload, 4)) * 0.1F;
            pf.has_accel = true;
            break;
        case 0x03:  // 四元数1 q0 q1
            pf.q0 = get_f32_le(payload, 0);
            pf.q1 = get_f32_le(payload, 4);
            pf.has_quat_1 = true;
            pf.quat_1_rx_ns = rx_ns;
            break;
        case 0x04:  // 四元数2 q2 q3
            pf.q2 = get_f32_le(payload, 0);
            pf.q3 = get_f32_le(payload, 4);
            pf.has_quat_2 = true;
            pf.quat_2_rx_ns = rx_ns;
            break;
        case 0x05:  // 数据状态
            pf.timestamp_ms = get_u32_le(payload, 0);
            pf.valid_flags = payload[4];
            pf.has_status = true;
            break;
        default:
            break;
    }
}

void daoyuan_apply_packet_to_partials(
    std::uint16_t can_id,
    const DaoyuanImuPacketAccumulator& packet,
    std::array<ImuRxAccumulator, kImuDeviceCount>& partials) {
    std::size_t slot = 0U;
    if (!daoyuan_slot_for_can_id(can_id, slot)) {
        return;
    }
    if (packet.size < kDaoyuanImuPacketBytes) {
        return;
    }

    const float roll_deg = get_f32_packet_le(packet.bytes, 11U);
    const float pitch_deg = get_f32_packet_le(packet.bytes, 15U);
    const float yaw_deg = get_f32_packet_le(packet.bytes, 19U);
    const float gyro_x = get_f32_packet_le(packet.bytes, 23U);
    const float gyro_y = get_f32_packet_le(packet.bytes, 27U);
    const float gyro_z = get_f32_packet_le(packet.bytes, 31U);
    const float accel_x = get_f32_packet_le(packet.bytes, 35U);
    const float accel_y = get_f32_packet_le(packet.bytes, 39U);
    const float accel_z = get_f32_packet_le(packet.bytes, 43U);
    if (!all_finite({roll_deg, pitch_deg, yaw_deg, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z})) {
        return;
    }

    auto& pf = partials[slot];
    const std::uint64_t rx_ns = now_ns();
    pf.last_rx_ns = rx_ns;
    pf.roll_raw_deg = roll_deg;
    pf.pitch_raw_deg = pitch_deg;
    pf.yaw_raw_deg = yaw_deg;
    pf.roll_rad = euler_deg_to_rad_pm_pi(pf.roll_raw_deg);
    pf.pitch_rad = euler_deg_to_rad_pm_pi(pf.pitch_raw_deg);
    pf.yaw_rad = euler_deg_to_rad_pm_pi(pf.yaw_raw_deg);
    pf.has_euler = true;
    pf.gyro_x_dps = gyro_x;
    pf.gyro_y_dps = gyro_y;
    pf.gyro_z_dps = gyro_z;
    pf.has_gyro = true;
    pf.accel_x_mps2 = accel_x;
    pf.accel_y_mps2 = accel_y;
    pf.accel_z_mps2 = accel_z;
    pf.has_accel = true;
    pf.timestamp_ms = get_u32_packet_le(packet.bytes, 56U);
    pf.valid_flags = 0x07U;
    pf.has_status = true;
    pf.has_quat_1 = false;
    pf.has_quat_2 = false;
    pf.quat_1_rx_ns = 0U;
    pf.quat_2_rx_ns = 0U;
    pf.q0 = 1.0F;
    pf.q1 = 0.0F;
    pf.q2 = 0.0F;
    pf.q3 = 0.0F;
}

}  // namespace

void ImuDefaultCanFrameParser::parseFrame(std::uint16_t can_id,
                                          const std::array<std::uint8_t, kImuCanPayloadBytes>& payload,
                                          std::array<ImuRxAccumulator, kImuDeviceCount>& partials) {
    std::size_t daoyuan_slot = 0U;
    if (daoyuan_slot_for_can_id(can_id, daoyuan_slot)) {
        auto& packet = daoyuan_packets_[daoyuan_slot];
        if (daoyuan_packet_header(payload)) {
            packet.size = 0U;
        } else if (packet.size == 0U) {
            return;
        }
        if (packet.size + payload.size() > packet.bytes.size()) {
            packet.size = 0U;
            return;
        }
        std::copy(payload.begin(), payload.end(), packet.bytes.begin() + static_cast<std::ptrdiff_t>(packet.size));
        packet.size += payload.size();
        if (packet.size >= packet.bytes.size()) {
            daoyuan_apply_packet_to_partials(can_id, packet, partials);
            packet.size = 0U;
        }
        return;
    }
    builtin_imu_apply_can_payload_to_partials(can_id, payload, partials);
}

struct ImuCanLib::Impl {
    std::string can_if_name;
    std::string shm_name;
    bool create_mapping{false};
    bool opened{false};
    std::atomic<bool> running{false};
    std::thread worker;
    std::string last_error;
    std::atomic<bool> simulation_enabled{true};
    std::atomic<std::uint64_t> loop_tick{0};
    ImuDefaultCanFrameParser default_frame_parser_;
    std::unique_ptr<ImuCanFrameParser> frame_parser_;

    ImuCanFrameParser* effective_frame_parser() {
        return frame_parser_ ? frame_parser_.get() : &default_frame_parser_;
    }

#if defined(__linux__)
    int can_fd{-1};
    int shm_fd{-1};
    ImuSharedMemoryLayout* shm_view{nullptr};
    UsbCanInterfaceConfig usbcan_config{};
    UsbCanApi usbcan_api{};
    bool usbcan_device_opened{false};
    bool usbcan_channel_started{false};
#endif

    std::array<ImuRxAccumulator, kImuDeviceCount> partials{};

    ~Impl() { close_impl(); }

    bool open_impl() {
#if defined(__linux__)
        last_error.clear();
        close_impl();
        if (!open_shm()) return false;
        if (!simulation_enabled.load(std::memory_order_acquire)) {
            if (!open_can()) {
                close_shm();
                return false;
            }
        }
        opened = true;
        return true;
#else
        return false;
#endif
    }

    bool start_impl() {
        if (!opened || running.load(std::memory_order_acquire)) return false;
        loop_tick.store(0, std::memory_order_release);
        running.store(true, std::memory_order_release);
        worker = std::thread([this] { loop(); });
        return true;
    }

    bool stop_impl() {
        running.store(false, std::memory_order_release);
        if (worker.joinable()) worker.join();
        return true;
    }

    void close_impl() {
        stop_impl();
#if defined(__linux__)
        close_can();
        close_shm();
#endif
        opened = false;
    }

    bool is_open_impl() const { return opened; }

#if defined(__linux__)
    bool open_shm() {
        const std::string normalized = normalize_shm_name(shm_name);
        const int flags = create_mapping ? (O_CREAT | O_RDWR) : O_RDWR;
        shm_fd = shm_open(normalized.c_str(), flags, 0660);
        if (shm_fd < 0) {
            last_error = "imu shm_open 失败: " + std::string(std::strerror(errno));
            return false;
        }
        constexpr std::size_t kSize = sizeof(ImuSharedMemoryLayout);
        if (create_mapping && ftruncate(shm_fd, static_cast<off_t>(kSize)) != 0) {
            last_error = "imu ftruncate 失败: " + std::string(std::strerror(errno));
            close_shm();
            return false;
        }
        void* mapped = mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (mapped == MAP_FAILED) {
            last_error = "imu mmap 失败: " + std::string(std::strerror(errno));
            close_shm();
            return false;
        }
        shm_view = static_cast<ImuSharedMemoryLayout*>(mapped);
        if (create_mapping) {
            std::memset(shm_view, 0, sizeof(ImuSharedMemoryLayout));
            shm_view->magic = kImuShmMagic;
            for (std::size_t i = 0; i < kImuDeviceCount; ++i) {
                shm_view->imus[i].device_addr = static_cast<std::uint8_t>(i + 1);
            }
        } else if (shm_view->magic != kImuShmMagic) {
            last_error = "imu 共享内存 magic 不匹配";
            close_shm();
            return false;
        }
        return true;
    }

    bool open_can() {
        usbcan_config = parse_usbcan_interface(can_if_name);
        if (!usbcan_config.error.empty()) {
            last_error = "imu USB-CAN 接口配置错误: " + usbcan_config.error;
            return false;
        }
        if (usbcan_config.enabled) {
            return open_usbcan();
        }
        return open_socketcan();
    }

    bool open_socketcan() {
        can_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (can_fd < 0) {
            last_error = "imu socket(PF_CAN) 失败: " + std::string(std::strerror(errno));
            return false;
        }

        const std::array<can_filter, 2> filters{{
            can_filter{kImuBaseIdHighSpeedCh1, 0x780U},
            can_filter{0x120U, 0x7F8U},
        }};
        if (setsockopt(can_fd, SOL_CAN_RAW, CAN_RAW_FILTER, filters.data(), sizeof(filters)) != 0) {
            last_error = "imu setsockopt(CAN_RAW_FILTER) 失败: " + std::string(std::strerror(errno));
            close_can();
            return false;
        }

        ifreq ifr{};
        std::snprintf(ifr.ifr_name, IFNAMSIZ, "%s", can_if_name.c_str());
        if (ioctl(can_fd, SIOCGIFINDEX, &ifr) < 0) {
            last_error = "imu ioctl(SIOCGIFINDEX) 失败: " + std::string(std::strerror(errno));
            close_can();
            return false;
        }

        sockaddr_can addr{};
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        if (bind(can_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            last_error = "imu bind(can) 失败: " + std::string(std::strerror(errno));
            close_can();
            return false;
        }

        const int fd_flags = fcntl(can_fd, F_GETFL, 0);
        if (fd_flags < 0 || fcntl(can_fd, F_SETFL, fd_flags | O_NONBLOCK) < 0) {
            last_error = "imu fcntl(O_NONBLOCK) 失败: " + std::string(std::strerror(errno));
            close_can();
            return false;
        }
        return true;
    }

    bool open_usbcan() {
        std::uint32_t baud = 0;
        if (!usbcan_baud_from_bitrate(usbcan_config.bitrate, baud)) {
            std::ostringstream oss;
            oss << "imu USB-CAN 不支持 bitrate=" << usbcan_config.bitrate
                << "，当前支持 1000000/500000/250000/125000";
            last_error = oss.str();
            return false;
        }
        std::string api_error;
        if (!load_usbcan_api(usbcan_api, api_error)) {
            last_error = "imu USB-CAN 加载 libusbcan.so 失败: " + api_error;
            usbcan_api.unload();
            return false;
        }
        const std::uint32_t dev_type = kUsbCanDeviceTypeUsbcCanIi;
        const std::uint32_t dev_idx = usbcan_config.device_index;
        const std::uint32_t channel = usbcan_config.channel_index;
        if (usbcan_api.open_device(dev_type, dev_idx, 0U) != 1U) {
            std::ostringstream oss;
            oss << "imu USB-CAN VCI_OpenDevice 失败: dev=" << dev_idx;
            last_error = oss.str();
            usbcan_api.unload();
            return false;
        }
        usbcan_device_opened = true;

        UsbCanInitConfig init{};
        init.AccCode = 0U;
        init.AccMask = 0xFFFFFFFFU;
        init.Reserved = 0U;
        init.Filter = 1U;
        init.Timing0 = static_cast<std::uint8_t>(baud & 0xFFU);
        init.Timing1 = static_cast<std::uint8_t>((baud >> 8U) & 0xFFU);
        init.Mode = 0U;
        if (usbcan_api.init_can(dev_type, dev_idx, channel, &init) != 1U) {
            std::ostringstream oss;
            oss << "imu USB-CAN VCI_InitCAN 失败: dev=" << dev_idx
                << " channel=" << channel << " bitrate=" << usbcan_config.bitrate;
            last_error = oss.str();
            close_usbcan();
            return false;
        }
        (void)usbcan_api.clear_buffer(dev_type, dev_idx, channel);
        if (usbcan_api.start_can(dev_type, dev_idx, channel) != 1U) {
            std::ostringstream oss;
            oss << "imu USB-CAN VCI_StartCAN 失败: dev=" << dev_idx
                << " channel=" << channel;
            last_error = oss.str();
            close_usbcan();
            return false;
        }
        usbcan_channel_started = true;
        return true;
    }

    void close_shm() {
        if (shm_view) {
            munmap(shm_view, sizeof(ImuSharedMemoryLayout));
            shm_view = nullptr;
        }
        if (shm_fd >= 0) {
            ::close(shm_fd);
            shm_fd = -1;
        }
    }

    void close_socketcan() {
        if (can_fd >= 0) {
            ::close(can_fd);
            can_fd = -1;
        }
    }

    void close_usbcan() {
        const std::uint32_t dev_type = kUsbCanDeviceTypeUsbcCanIi;
        const std::uint32_t dev_idx = usbcan_config.device_index;
        const std::uint32_t channel = usbcan_config.channel_index;
        if (usbcan_api.handle) {
            if (usbcan_channel_started && usbcan_api.reset_can) {
                (void)usbcan_api.reset_can(dev_type, dev_idx, channel);
            }
            if (usbcan_device_opened && usbcan_api.close_device) {
                (void)usbcan_api.close_device(dev_type, dev_idx);
            }
        }
        usbcan_channel_started = false;
        usbcan_device_opened = false;
        usbcan_api.unload();
    }

    void close_can() {
        close_socketcan();
        close_usbcan();
    }

    void publish_simulation_defaults() {
        if (!shm_view) return;
        const std::uint64_t tns = now_ns();
        const std::uint32_t tms = static_cast<std::uint32_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        for (std::size_t i = 0; i < kImuDeviceCount; ++i) {
            auto& d = shm_view->imus[i];
            d.device_addr = static_cast<std::uint8_t>(i + 1U);
            d.online = 1U;
            d.valid_attitude = 1U;
            d.valid_quaternion = 1U;
            d.valid_gyro = 1U;
            d.valid_accel = 1U;
            d.packet_loss_count = 0;
            d.imu_timestamp_ms = tms;
            d.host_rx_time_ns = tns;
            d.rpy_rad.setZero();
            d.rpy_raw_deg.setZero();
            d.gyro_dps.setZero();
            d.accel_mps2.setZero();
            d.quaternion = Eigen::Quaternionf(1.0F, 0.0F, 0.0F, 0.0F);
        }
        ++shm_view->sequence;
    }

    void drain_can() {
        if (usbcan_config.enabled) {
            drain_usbcan();
        } else {
            drain_socketcan();
        }
    }

    void drain_socketcan() {
        if (can_fd < 0) return;
        for (std::size_t frame_count = 0; frame_count < kSocketCanMaxFramesPerTick;
             ++frame_count) {
            can_frame frame{};
            const ssize_t n = recv(can_fd, &frame, sizeof(frame), MSG_DONTWAIT);
            if (n < 0) break;
            if (n != static_cast<ssize_t>(sizeof(frame)) || frame.can_dlc < kCanPayloadSize) continue;
            if ((frame.can_id & CAN_EFF_FLAG) != 0) continue;  // 只处理标准帧
            const std::uint16_t can_id = static_cast<std::uint16_t>(frame.can_id & CAN_SFF_MASK);
            const std::uint16_t func = static_cast<std::uint16_t>((can_id >> 6U) & 0x1FU);
            std::size_t daoyuan_slot = 0U;
            if (func != 0x08U && !daoyuan_slot_for_can_id(can_id, daoyuan_slot)) continue;
            std::array<std::uint8_t, kCanPayloadSize> payload{};
            std::memcpy(payload.data(), frame.data, kCanPayloadSize);
            effective_frame_parser()->parseFrame(can_id, payload, partials);
        }
    }

    void drain_usbcan() {
        if (!usbcan_device_opened || !usbcan_api.get_receive_num || !usbcan_api.receive) return;
        const std::uint32_t dev_type = kUsbCanDeviceTypeUsbcCanIi;
        const std::uint32_t dev_idx = usbcan_config.device_index;
        const std::uint32_t channel = usbcan_config.channel_index;
        std::array<UsbCanObj, kUsbCanReceiveBatch> frames{};
        for (int round = 0; round < 16; ++round) {
            const std::uint32_t available = usbcan_api.get_receive_num(dev_type, dev_idx, channel);
            if (available == 0U) break;
            const std::uint32_t want = std::min<std::uint32_t>(
                available, static_cast<std::uint32_t>(frames.size()));
            const std::uint32_t got = usbcan_api.receive(dev_type, dev_idx, channel, frames.data(), want, 0);
            if (got == 0U) break;
            for (std::uint32_t i = 0; i < got && i < frames.size(); ++i) {
                const UsbCanObj& frame = frames[i];
                if (frame.RemoteFlag != 0U || frame.ExternFlag != 0U || frame.DataLen < kCanPayloadSize) {
                    continue;
                }
                const std::uint16_t can_id = static_cast<std::uint16_t>(frame.ID & 0x7FFU);
                const std::uint16_t func = static_cast<std::uint16_t>((can_id >> 6U) & 0x1FU);
                std::size_t daoyuan_slot = 0U;
                if (func != 0x08U && !daoyuan_slot_for_can_id(can_id, daoyuan_slot)) continue;
                std::array<std::uint8_t, kCanPayloadSize> payload{};
                std::memcpy(payload.data(), frame.Data, kCanPayloadSize);
                effective_frame_parser()->parseFrame(can_id, payload, partials);
            }
            if (got < want) break;
        }
    }

    void publish_shm() {
        if (!shm_view) return;
        const std::uint64_t now = now_ns();
        for (std::size_t i = 0; i < kImuDeviceCount; ++i) {
            const auto& pf = partials[i];
            auto& dst = shm_view->imus[i];
            dst.device_addr = static_cast<std::uint8_t>(i + 1U);
            dst.online = (pf.last_rx_ns != 0U && (now - pf.last_rx_ns) <=
                                                 static_cast<std::uint64_t>(
                                                     std::chrono::duration_cast<std::chrono::nanoseconds>(kOfflineTimeout)
                                                         .count()))
                             ? 1U
                             : 0U;
            const bool online = dst.online != 0U;
            dst.valid_attitude = (online && (pf.valid_flags & 0x01U) != 0U) ? 1U : 0U;
            const bool quaternion_synced = imu_quaternion_halves_synchronized(pf);
            dst.valid_quaternion = (online && quaternion_synced) ? 1U : 0U;
            dst.valid_gyro = (online && (pf.valid_flags & 0x02U) != 0U) ? 1U : 0U;
            dst.valid_accel = (online && (pf.valid_flags & 0x04U) != 0U) ? 1U : 0U;
            dst.imu_timestamp_ms = pf.timestamp_ms;
            dst.host_rx_time_ns = pf.last_rx_ns;
            if (pf.has_euler) {
                dst.rpy_rad(0) = pf.roll_rad;
                dst.rpy_rad(1) = pf.pitch_rad;
                dst.rpy_rad(2) = pf.yaw_rad;
                dst.rpy_raw_deg(0) = pf.roll_raw_deg;
                dst.rpy_raw_deg(1) = pf.pitch_raw_deg;
                dst.rpy_raw_deg(2) = pf.yaw_raw_deg;
            }
            if (pf.has_gyro) {
                dst.gyro_dps(0) = pf.gyro_x_dps;
                dst.gyro_dps(1) = pf.gyro_y_dps;
                dst.gyro_dps(2) = pf.gyro_z_dps;
            }
            if (pf.has_accel) {
                dst.accel_mps2(0) = pf.accel_x_mps2;
                dst.accel_mps2(1) = pf.accel_y_mps2;
                dst.accel_mps2(2) = pf.accel_z_mps2;
            }
            if (pf.has_quat_1 && pf.has_quat_2) {
                dst.quaternion = Eigen::Quaternionf(pf.q0, pf.q1, pf.q2, pf.q3);
            }
        }
        ++shm_view->sequence;
    }

    void loop() {
        using clock = std::chrono::steady_clock;
        auto next_tick = clock::now();
        while (running.load(std::memory_order_acquire)) {
            loop_tick.fetch_add(1, std::memory_order_acq_rel);
            next_tick += kLoopPeriod;
            if (simulation_enabled.load(std::memory_order_acquire)) {
                publish_simulation_defaults();
            } else {
                drain_can();
                publish_shm();
            }
            std::this_thread::sleep_until(next_tick);
        }
    }
#endif
};

ImuCanLib::ImuCanLib(std::string can_if_name, std::string shm_name, bool create_mapping)
    : impl_(std::make_unique<Impl>()) {
    impl_->can_if_name = std::move(can_if_name);
    impl_->shm_name = std::move(shm_name);
    impl_->create_mapping = create_mapping;
}

ImuCanLib::~ImuCanLib() = default;

bool ImuCanLib::open() {
    if (!impl_) return false;
    return impl_->open_impl();
}

bool ImuCanLib::start() {
    if (!impl_) return false;
    return impl_->start_impl();
}

bool ImuCanLib::stop() {
    if (!impl_) return false;
    return impl_->stop_impl();
}

bool ImuCanLib::close() {
    if (!impl_) return false;
    impl_->close_impl();
    return true;
}

bool ImuCanLib::isOpen() const {
    if (!impl_) return false;
    return impl_->is_open_impl();
}

std::string ImuCanLib::lastError() const {
    if (!impl_) return "imu canlib impl 为空";
    return impl_->last_error;
}

std::uint64_t ImuCanLib::loopTick() const {
    if (!impl_) return 0;
    return impl_->loop_tick.load(std::memory_order_acquire);
}

void ImuCanLib::setSimulationEnabled(bool enabled) {
    if (!impl_) return;
    impl_->simulation_enabled.store(enabled, std::memory_order_release);
}

bool ImuCanLib::isSimulationEnabled() const {
    if (!impl_) return false;
    return impl_->simulation_enabled.load(std::memory_order_acquire);
}

void ImuCanLib::setFrameParser(std::unique_ptr<ImuCanFrameParser> parser) {
    if (!impl_) return;
    impl_->frame_parser_ = std::move(parser);
}

ImuCanFrameParser* ImuCanLib::frameParser() {
    if (!impl_) return nullptr;
    return impl_->effective_frame_parser();
}

}  // namespace canlib
