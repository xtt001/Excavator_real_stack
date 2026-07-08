#include <can/internal/imu_canlib.hpp>

#include <algorithm>
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

std::array<std::uint8_t, 64> daoyuan_packet(
    float roll_deg,
    float pitch_deg,
    float yaw_deg,
    float gyro_x,
    float gyro_y,
    float gyro_z,
    float accel_x,
    float accel_y,
    float accel_z,
    std::uint32_t timestamp_ms) {
    std::array<std::uint8_t, 64> out{};
    out[0] = 0xABU;
    out[1] = 0x54U;
    out[2] = 0x65U;
    out[3] = 0x00U;
    out[4] = 0x35U;
    out[5] = 0x00U;
    auto put_f32 = [&](std::size_t idx, float value) {
        std::memcpy(out.data() + idx, &value, sizeof(float));
    };
    auto put_u32 = [&](std::size_t idx, std::uint32_t value) {
        out[idx] = static_cast<std::uint8_t>(value & 0xFFU);
        out[idx + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
        out[idx + 2] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
        out[idx + 3] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
    };
    put_f32(11, roll_deg);
    put_f32(15, pitch_deg);
    put_f32(19, yaw_deg);
    put_f32(23, gyro_x);
    put_f32(27, gyro_y);
    put_f32(31, gyro_z);
    put_f32(35, accel_x);
    put_f32(39, accel_y);
    put_f32(43, accel_z);
    put_u32(56, timestamp_ms);
    return out;
}

void feed_daoyuan_packet(ImuDefaultCanFrameParser& parser,
                         std::array<ImuRxAccumulator, kImuDeviceCount>& partials,
                         std::uint16_t can_id,
                         const std::array<std::uint8_t, 64>& packet) {
    for (std::size_t offset = 0; offset < packet.size(); offset += kImuCanPayloadBytes) {
        std::array<std::uint8_t, kImuCanPayloadBytes> payload{};
        std::copy_n(packet.data() + offset, kImuCanPayloadBytes, payload.data());
        parser.parseFrame(can_id, payload, partials);
    }
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

void test_daoyuan_packet_mapping_and_fields() {
    ImuDefaultCanFrameParser parser;
    std::array<ImuRxAccumulator, kImuDeviceCount> partials{};
    feed_daoyuan_packet(
        parser, partials, 0x122U,
        daoyuan_packet(10.0F, 20.0F, 30.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 100U));
    feed_daoyuan_packet(
        parser, partials, 0x124U,
        daoyuan_packet(11.0F, 21.0F, 31.0F, 1.1F, 2.1F, 3.1F, 4.1F, 5.1F, 6.1F, 101U));
    feed_daoyuan_packet(
        parser, partials, 0x121U,
        daoyuan_packet(12.0F, 22.0F, 32.0F, 1.2F, 2.2F, 3.2F, 4.2F, 5.2F, 6.2F, 102U));
    feed_daoyuan_packet(
        parser, partials, 0x123U,
        daoyuan_packet(13.0F, 23.0F, 33.0F, 1.3F, 2.3F, 3.3F, 4.3F, 5.3F, 6.3F, 103U));

    // Field labels: 0x122 bucket -> device[0], 0x124 stick -> device[1],
    // 0x121 boom -> device[2], 0x123 swing -> device[3].
    expect_near(partials[0].roll_raw_deg, 10.0F, "daoyuan bucket slot roll wrong");
    expect_near(partials[1].roll_raw_deg, 11.0F, "daoyuan stick slot roll wrong");
    expect_near(partials[2].roll_raw_deg, 12.0F, "daoyuan boom slot roll wrong");
    expect_near(partials[3].roll_raw_deg, 13.0F, "daoyuan swing slot roll wrong");
    expect_near(partials[0].pitch_raw_deg, 20.0F, "daoyuan pitch wrong");
    expect_near(partials[0].gyro_y_dps, 2.0F, "daoyuan gyro wrong");
    expect_near(partials[0].accel_z_mps2, 6.0F, "daoyuan accel wrong");
    expect(partials[0].valid_flags == 0x07U, "daoyuan valid flags wrong");
    expect(partials[0].has_euler, "daoyuan euler missing");
    expect(partials[0].has_gyro, "daoyuan gyro missing");
    expect(partials[0].has_accel, "daoyuan accel missing");
    expect(partials[0].has_status, "daoyuan status missing");
    expect(partials[0].timestamp_ms == 100U, "daoyuan timestamp wrong");
    expect(!partials[0].has_quat_1 && !partials[0].has_quat_2, "daoyuan should not publish quaternion halves");
}

}  // namespace

int main() {
    try {
        test_zero_based_addresses();
        test_one_based_addresses();
        test_missing_addresses_remain_absent();
        test_raw_euler_degrees_preserved();
        test_quaternion_halves_require_sync_window();
        test_daoyuan_packet_mapping_and_fields();
    } catch (const std::exception& exc) {
        std::cerr << "imu_canlib_parser_test failed: " << exc.what() << "\n";
        return 1;
    }
    std::cout << "imu_canlib_parser_test OK\n";
    return 0;
}
