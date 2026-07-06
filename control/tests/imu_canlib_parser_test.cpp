#include <can/internal/imu_canlib.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

using canlib::ImuDefaultCanFrameParser;
using canlib::ImuRxAccumulator;
using canlib::imu_quaternion_halves_synchronized;
using canlib::kImuCanPayloadBytes;
using canlib::kImuDeviceCount;
using canlib::kImuQuaternionHalfSyncWindowNs;

std::array<std::uint8_t, kImuCanPayloadBytes> zeros() {
    return {};
}

std::array<std::uint8_t, kImuCanPayloadBytes> i16_payload(
    std::int16_t a,
    std::int16_t b,
    std::int16_t c) {
    std::array<std::uint8_t, kImuCanPayloadBytes> out{};
    auto put = [&](std::size_t idx, std::int16_t v) {
        const auto u = static_cast<std::uint16_t>(v);
        out[idx] = static_cast<std::uint8_t>(u & 0xFFU);
        out[idx + 1] = static_cast<std::uint8_t>((u >> 8U) & 0xFFU);
    };
    put(0, a);
    put(2, b);
    put(4, c);
    return out;
}

std::array<std::uint8_t, kImuCanPayloadBytes> euler_payload(
    std::int16_t roll,
    std::int16_t pitch,
    std::uint16_t yaw) {
    std::array<std::uint8_t, kImuCanPayloadBytes> out{};
    auto put = [&](std::size_t idx, std::uint16_t v) {
        out[idx] = static_cast<std::uint8_t>(v & 0xFFU);
        out[idx + 1] = static_cast<std::uint8_t>((v >> 8U) & 0xFFU);
    };
    put(0, static_cast<std::uint16_t>(roll));
    put(2, static_cast<std::uint16_t>(pitch));
    put(4, yaw);
    return out;
}

std::array<std::uint8_t, kImuCanPayloadBytes> status_payload(std::uint8_t flags) {
    auto out = zeros();
    out[4] = flags;
    return out;
}

std::array<std::uint8_t, kImuCanPayloadBytes> f32_payload(float a, float b) {
    std::array<std::uint8_t, kImuCanPayloadBytes> out{};
    std::memcpy(&out[0], &a, sizeof(float));
    std::memcpy(&out[4], &b, sizeof(float));
    return out;
}

void expect(bool ok, const char* message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void expect_near(float actual, float expected, const char* message) {
    if (std::fabs(actual - expected) > 1e-4F) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        throw std::runtime_error(message);
    }
}

std::uint16_t frame_id(std::uint8_t cmd, std::uint8_t raw_addr) {
    return static_cast<std::uint16_t>(0x200U | (static_cast<std::uint16_t>(cmd) << 3U) | raw_addr);
}

void test_zero_based_addresses() {
    ImuDefaultCanFrameParser parser;
    std::array<ImuRxAccumulator, kImuDeviceCount> partials{};
    for (std::uint8_t addr = 0; addr < kImuDeviceCount; ++addr) {
        parser.parseFrame(frame_id(0, addr), i16_payload(100, 200, 300), partials);
        parser.parseFrame(frame_id(1, addr), i16_payload(10, 20, 30), partials);
        parser.parseFrame(frame_id(5, addr), status_payload(0x07), partials);
    }
    for (std::size_t slot = 0; slot < kImuDeviceCount; ++slot) {
        expect(partials[slot].has_euler, "zero-based euler missing");
        expect(partials[slot].has_gyro, "zero-based gyro missing");
        expect(partials[slot].has_status, "zero-based status missing");
        expect(partials[slot].valid_flags == 0x07U, "zero-based valid flags wrong");
        expect(partials[slot].last_rx_ns != 0U, "zero-based last_rx_ns missing");
        expect_near(partials[slot].gyro_y_dps, 2.0F, "zero-based gyro_y wrong");
    }
}

void test_one_based_addresses() {
    ImuDefaultCanFrameParser parser;
    std::array<ImuRxAccumulator, kImuDeviceCount> partials{};
    // Seeing raw address 4 forces one-based mode before the lower addresses arrive.
    parser.parseFrame(frame_id(0, 4), i16_payload(100, 200, 300), partials);
    parser.parseFrame(frame_id(5, 4), status_payload(0x07), partials);
    for (std::uint8_t raw_addr = 1; raw_addr <= kImuDeviceCount; ++raw_addr) {
        parser.parseFrame(frame_id(0, raw_addr), i16_payload(100, 200, 300), partials);
        parser.parseFrame(frame_id(1, raw_addr), i16_payload(10, 20, 30), partials);
        parser.parseFrame(frame_id(5, raw_addr), status_payload(0x07), partials);
    }
    for (std::size_t slot = 0; slot < kImuDeviceCount; ++slot) {
        expect(partials[slot].has_euler, "one-based euler missing");
        expect(partials[slot].has_gyro, "one-based gyro missing");
        expect(partials[slot].has_status, "one-based status missing");
        expect(partials[slot].valid_flags == 0x07U, "one-based valid flags wrong");
        expect(partials[slot].last_rx_ns != 0U, "one-based last_rx_ns missing");
    }
}

void test_missing_addresses_remain_absent() {
    ImuDefaultCanFrameParser parser;
    std::array<ImuRxAccumulator, kImuDeviceCount> partials{};
    parser.parseFrame(frame_id(0, 0), i16_payload(100, 200, 300), partials);
    parser.parseFrame(frame_id(0, 3), i16_payload(100, 200, 300), partials);
    expect(partials[0].last_rx_ns != 0U, "raw addr 0 should be present");
    expect(partials[1].last_rx_ns == 0U, "raw addr 1 should remain absent");
    expect(partials[2].last_rx_ns == 0U, "raw addr 2 should remain absent");
    expect(partials[3].last_rx_ns != 0U, "raw addr 3 should be present");
}

void test_raw_euler_degrees_preserved() {
    constexpr float kPi = 3.14159265F;
    ImuDefaultCanFrameParser parser;
    std::array<ImuRxAccumulator, kImuDeviceCount> partials{};
    parser.parseFrame(frame_id(0, 4), euler_payload(18300, -18100, 32769), partials);
    const auto& sample = partials[3];
    expect(sample.has_euler, "raw euler missing");
    expect_near(sample.roll_raw_deg, 183.0F, "raw roll degree wrong");
    expect_near(sample.pitch_raw_deg, -181.0F, "raw pitch degree wrong");
    expect_near(sample.yaw_raw_deg, 327.69F, "raw yaw degree wrong");
    expect_near(sample.roll_rad, -177.0F * kPi / 180.0F, "wrapped roll rad wrong");
    expect_near(sample.pitch_rad, 179.0F * kPi / 180.0F, "wrapped pitch rad wrong");
    expect_near(sample.yaw_rad, -32.31F * kPi / 180.0F, "yaw rad wrong");
}

void test_quaternion_halves_require_sync_window() {
    ImuDefaultCanFrameParser parser;
    std::array<ImuRxAccumulator, kImuDeviceCount> partials{};
    parser.parseFrame(frame_id(4, 4), f32_payload(0.3F, 0.4F), partials);
    auto& sample = partials[3];
    expect(sample.has_quat_2, "quat half 2 missing");
    expect(!imu_quaternion_halves_synchronized(sample), "single quat half should not be synchronized");

    parser.parseFrame(frame_id(3, 4), f32_payload(0.1F, 0.2F), partials);
    expect(sample.has_quat_1, "quat half 1 missing");
    expect(sample.quat_1_rx_ns != 0U, "quat half 1 rx time missing");
    expect(sample.quat_2_rx_ns != 0U, "quat half 2 rx time missing");
    expect(imu_quaternion_halves_synchronized(sample), "fresh quat halves should be synchronized");
    expect_near(sample.q0, 0.1F, "q0 wrong");
    expect_near(sample.q1, 0.2F, "q1 wrong");
    expect_near(sample.q2, 0.3F, "q2 wrong");
    expect_near(sample.q3, 0.4F, "q3 wrong");

    sample.quat_1_rx_ns = sample.quat_2_rx_ns + kImuQuaternionHalfSyncWindowNs + 1U;
    expect(!imu_quaternion_halves_synchronized(sample), "stale quat halves should not be synchronized");
}

}  // namespace

int main() {
    try {
        test_zero_based_addresses();
        test_one_based_addresses();
        test_missing_addresses_remain_absent();
        test_raw_euler_degrees_preserved();
        test_quaternion_halves_require_sync_window();
    } catch (const std::exception& exc) {
        std::cerr << "imu_canlib_parser_test failed: " << exc.what() << "\n";
        return 1;
    }
    std::cout << "imu_canlib_parser_test OK\n";
    return 0;
}
