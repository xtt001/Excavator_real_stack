#pragma once

#include <Eigen/Dense>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace excavator_api {

using Vector8d = Eigen::Matrix<double, 8, 1>;
using Vector12i = Eigen::Matrix<int, 12, 1>;
inline constexpr std::size_t kImuDeviceCount = 4;

enum class ControlMode : std::uint8_t {
    OpenLoopMotorSpeed = 0,
    ClosedLoopJointPosition = 1,
    ClosedLoopJointVelocity = 2,
    ClosedLoopVelocityScalar = 3,
};

struct SessionConfig {
    std::string can_if_name{"can2"};
    std::string imu_if_name{"can1"};
    std::string can_shm_name{"canlib_shm"};
    std::string imu_shm_name{"imu_canlib_shm"};
    bool create_mapping{false};
    bool can_simulation{true};
    bool imu_simulation{true};
    bool can_bus_enabled{true};
};

struct SpeedScalarCmd {
    Vector8d speed_scalar = Vector8d::Zero();  // [-1,1]
};

struct VelocityCmd {
    Vector8d velocity = Vector8d::Zero();
};

struct PositionCmd {
    Vector8d position = Vector8d::Zero();
};

struct RefState {
    Vector8d position = Vector8d::Zero();
    Vector8d velocity = Vector8d::Zero();
    Vector8d acceleration = Vector8d::Zero();
    Vector8d velocity_scalar = Vector8d::Zero();
    Vector12i status = Vector12i::Zero();
    Vector8d motor_rpm = Vector8d::Constant(8190.0);
    Vector8d plan_rpm = Vector8d::Constant(8190.0);
};

struct ImuHealth {
    std::array<std::uint8_t, kImuDeviceCount> online{};
    std::array<std::uint8_t, kImuDeviceCount> valid_attitude{};
    std::array<std::uint8_t, kImuDeviceCount> valid_gyro{};
    std::array<std::uint8_t, kImuDeviceCount> valid_accel{};
    std::array<std::uint16_t, kImuDeviceCount> packet_loss_count{};
    std::array<std::uint64_t, kImuDeviceCount> host_rx_time_ns{};
};

struct ImuDebugSample {
    std::uint8_t device_addr{0};
    std::uint8_t online{0};
    std::uint8_t valid_attitude{0};
    std::uint8_t valid_gyro{0};
    std::uint8_t valid_accel{0};
    std::uint16_t packet_loss_count{0};
    std::uint32_t imu_timestamp_ms{0};
    std::uint64_t host_rx_time_ns{0};
    std::array<double, 3> rpy_rad{};
    std::array<double, 3> gyro_dps{};
    std::array<double, 3> accel_mps2{};
    std::array<double, 4> quaternion_wxyz{1.0, 0.0, 0.0, 0.0};
};

struct ImuDebug {
    std::array<ImuDebugSample, kImuDeviceCount> devices{};
};

struct RespState {
    Vector8d position = Vector8d::Zero();
    Vector8d velocity = Vector8d::Zero();
    Vector8d acceleration = Vector8d::Zero();
    Vector8d velocity_scalar = Vector8d::Zero();
    Vector12i status = Vector12i::Zero();
    Vector8d motor_rpm = Vector8d::Constant(8190.0);
    Vector8d plan_rpm = Vector8d::Constant(8190.0);
    ImuHealth imu_health{};
    ImuDebug imu_debug{};
};

struct SnapshotMeta {
    std::uint64_t loop_tick{0};
    std::uint64_t recv_time_ns{0};
};

struct Snapshot {
    RefState ref{};
    RespState resp{};
    SnapshotMeta meta{};
};

}  // namespace excavator_api
