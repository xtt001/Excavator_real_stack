#include <excavator/internal/excavator_converter.hpp>

#include <cmath>
#include <cstdlib>

namespace excavator {
namespace {

constexpr double kPositionJumpGuardMarginRad = kPi / 360.0;  // 0.5 deg at 50 Hz.
constexpr int kBucketAxis = 3;
constexpr double kBucketMaxPositionStepRad = 2.5 * kPi / 180.0;
// Maps the quaternion bucket angle back into the legacy policy/home qpos frame.
constexpr double kDefaultBucketQuaternionPolicyOffsetRad = -0.4060066694119653;
constexpr float kUninitializedAttitudeEps = 1e-6F;

double dps_to_radps(double dps) noexcept { return dps * kPi / 180.0; }

double deg_to_rad(double deg) noexcept { return deg * kPi / 180.0; }

double unwrap_angle_nearest(double previous, double current) noexcept {
    return previous + std::remainder(current - previous, 2.0 * kPi);
}

double align_angle_to_reference_branch(double value, double reference) noexcept {
    return value + std::round((reference - value) / (2.0 * kPi)) * (2.0 * kPi);
}

double align_swing_to_nonnegative_raw_yaw_branch(double value, double raw_yaw) noexcept {
    if (!std::isfinite(value) || !std::isfinite(raw_yaw)) {
        return value;
    }
    const double branch_index = std::round((value - raw_yaw) / (2.0 * kPi));
    const double branch_value = raw_yaw + branch_index * (2.0 * kPi);
    if (branch_index >= 0.0 && std::abs(value - branch_value) <= kPi) {
        return value;
    }
    return align_angle_to_reference_branch(value, raw_yaw);
}

bool swing_raw_yaw_reference_rad(const ExcavatorHardwareState& hw, double& out) noexcept {
    const auto& imu4 = hw.imu.devices[3];
    if (imu4.online == 0U || imu4.valid_attitude == 0U || imu4.host_rx_time_ns == 0U) {
        return false;
    }
    out = static_cast<double>(imu4.rpy_raw_deg(2)) * kPi / 180.0;
    return std::isfinite(out);
}

bool all_imu_attitudes_observed(const ExcavatorHardwareState& hw) noexcept {
    for (const auto& imu : hw.imu.devices) {
        if (imu.valid_attitude == 0U || imu.host_rx_time_ns == 0U) {
            return false;
        }
    }
    return true;
}

bool looks_like_uninitialized_attitude(const ExcavatorImuHardwareState::ImuSample& src) noexcept {
    return src.rpy_raw_deg.cwiseAbs().maxCoeff() <= kUninitializedAttitudeEps &&
           src.rpy_rad.cwiseAbs().maxCoeff() <= kUninitializedAttitudeEps;
}

bool imu_quaternion_valid(const ExcavatorImuHardwareState::ImuSample& src) noexcept {
    if (src.online == 0U || src.valid_quaternion == 0U || src.host_rx_time_ns == 0U) {
        return false;
    }
    const Eigen::Quaternionf& q = src.quaternion;
    if (!std::isfinite(q.w()) || !std::isfinite(q.x()) || !std::isfinite(q.y()) ||
        !std::isfinite(q.z())) {
        return false;
    }
    const float norm = q.norm();
    return std::isfinite(norm) && norm > 0.5F && norm < 1.5F;
}

Eigen::Quaterniond normalized_quaternion(const ExcavatorImuHardwareState::ImuSample& src) noexcept {
    Eigen::Quaterniond q(
        static_cast<double>(src.quaternion.w()),
        static_cast<double>(src.quaternion.x()),
        static_cast<double>(src.quaternion.y()),
        static_cast<double>(src.quaternion.z()));
    q.normalize();
    return q;
}

double signed_twist_angle_rad(const Eigen::Quaterniond& rotation,
                              const Eigen::Vector3d& axis) noexcept {
    const Eigen::Vector3d unit_axis = axis.normalized();
    const Eigen::Vector3d vector(rotation.x(), rotation.y(), rotation.z());
    const double projected = vector.dot(unit_axis);
    return std::remainder(2.0 * std::atan2(projected, rotation.w()), 2.0 * kPi);
}

double bucket_quaternion_policy_offset_rad() noexcept {
    const char* raw = std::getenv("EXCAVATOR_BUCKET_QUATERNION_OFFSET_RAD");
    if (raw == nullptr || raw[0] == '\0') {
        return kDefaultBucketQuaternionPolicyOffsetRad;
    }
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || !std::isfinite(value)) {
        return kDefaultBucketQuaternionPolicyOffsetRad;
    }
    return value;
}

bool bucket_quaternion_position_rad(const ExcavatorHardwareState& hw, double& out) noexcept {
    const auto& imu1 = hw.imu.devices[0];
    const auto& imu2 = hw.imu.devices[1];
    if (!imu_quaternion_valid(imu1) || !imu_quaternion_valid(imu2)) {
        return false;
    }
    const Eigen::Quaterniond q_imu1 = normalized_quaternion(imu1);
    const Eigen::Quaterniond q_imu2 = normalized_quaternion(imu2);
    const Eigen::Quaterniond relative = q_imu2.conjugate() * q_imu1;
    out = signed_twist_angle_rad(relative, Eigen::Vector3d::UnitY()) +
          bucket_quaternion_policy_offset_rad();
    return std::isfinite(out);
}

/** 交换前4关节的2/3号位（1234 <-> 1324），其余轴保持不变。 */
Vector8d swap_joint_2_3_on_first4(const Vector8d& v) noexcept {
    Vector8d out = v;
    out(1) = v(2);
    out(2) = v(1);
    return out;
}

void fill_kinematic_from_imu_hw(const ExcavatorHardwareState& hw,
                                const std::array<Eigen::Vector3d, kImuDeviceCount>& rpy_rad,
                                ExcavatorState& st) noexcept {
    st.position.setZero();
    st.velocity.setZero();
    st.acceleration.setZero();
    st.imu = hw.imu;
    // 关节1234 <- imu4321，轴向为 z y y y。
    const auto& imu1 = hw.imu.devices[0];
    const auto& imu2 = hw.imu.devices[1];
    const auto& imu3 = hw.imu.devices[2];
    const auto& imu4 = hw.imu.devices[3];

    st.position(0) = rpy_rad[3](2);
    // Field logs show IMU4 yaw increases when gyro_z is negative.
    st.velocity(0) = -dps_to_radps(static_cast<double>(imu4.gyro_dps(2)));
    st.acceleration(0) = static_cast<double>(imu4.accel_mps2(2));

    st.position(1) = rpy_rad[2](1);
    st.velocity(1) = dps_to_radps(static_cast<double>(imu3.gyro_dps(1)));
    st.acceleration(1) = static_cast<double>(imu3.accel_mps2(1));

    st.position(2) = rpy_rad[1](1);
    st.velocity(2) = dps_to_radps(static_cast<double>(imu2.gyro_dps(1)));
    st.acceleration(2) = static_cast<double>(imu2.accel_mps2(1));

    st.position(3) = rpy_rad[0](1);
    st.velocity(3) = dps_to_radps(static_cast<double>(imu1.gyro_dps(1)));
    st.acceleration(3) = static_cast<double>(imu1.accel_mps2(1));
}

}  // namespace

bool ExcavatorConverter::robotCmdToRobotState(const RobotCommand& cmd, RobotState& state_out) {
    const auto* c = asCommand(cmd);
    auto* r = asState(state_out);
    if (!c || !r) {
        return false;
    }
    r->position = c->position;
    r->velocity = c->velocity;
    const double off =
        (c->motor_rpm.array() - kMotorSpeedRawZero).matrix().template lpNorm<Eigen::Infinity>();
    r->motor_rpm = (off < 1e-6) ? c->velocity : c->motor_rpm;
    return true;
}

bool ExcavatorConverter::robotStateToHardwareCmd(const RobotState& state, HardwareCommand& cmd_out) {
    const auto* st = asState(state);
    auto* hw = asHardwareCmd(cmd_out);
    if (!st || !hw) {
        return false;
    }
    hw->status = st->status;
    // 闭环：plan_rpm = 前馈 motor_rpm + PID；总线下发规划转速。开环见 control 内 plan_rpm=motor_rpm。
    hw->motor_rpm = swap_joint_2_3_on_first4(st->plan_rpm);
    return true;
}

std::array<Eigen::Vector3d, kImuDeviceCount> ExcavatorConverter::continuousImuRpy(
    const ExcavatorHardwareState& hw) {
    bool default_attitude_snapshot = all_imu_attitudes_observed(hw);
    if (default_attitude_snapshot) {
        for (const auto& src : hw.imu.devices) {
            if (!looks_like_uninitialized_attitude(src)) {
                default_attitude_snapshot = false;
                break;
            }
        }
    }
    for (std::size_t i = 0; i < kImuDeviceCount; ++i) {
        const auto& src = hw.imu.devices[i];
        // 用协议原始角度确定 policy 分支：yaw 保持 0..360° 分支，避免同一
        // 物理姿态在不同启动历史下变成 222° 或 -137° 两种数值。
        const Eigen::Vector3d current = src.rpy_raw_deg.cast<double>() * deg_to_rad(1.0);
        if (src.online == 0U || src.valid_attitude == 0U || src.host_rx_time_ns == 0U) {
            if (!imu_rpy_continuous_ready_[i]) {
                imu_rpy_continuous_[i] = current;
            }
            continue;
        }
        if (!imu_rpy_continuous_ready_[i] && default_attitude_snapshot) {
            imu_rpy_continuous_[i] = current;
            continue;
        }
        if (!imu_rpy_continuous_ready_[i]) {
            imu_rpy_continuous_[i] = current;
            imu_rpy_continuous_ready_[i] = true;
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            imu_rpy_continuous_[i](axis) =
                unwrap_angle_nearest(imu_rpy_continuous_[i](axis), current(axis));
        }
    }
    return imu_rpy_continuous_;
}

void ExcavatorConverter::applyPositionContinuity(ExcavatorState& st,
                                                 bool position_observed,
                                                 const Vector8d& branch_reference,
                                                 bool bucket_quaternion_observed) {
    if (!position_observed && !resp_position_continuous_ready_) {
        return;
    }
    if (!resp_position_continuous_ready_) {
        resp_position_continuous_ = st.position;
        st.position = resp_position_continuous_;
        resp_position_continuous_ready_ = true;
        return;
    }
    if (!position_observed) {
        st.position = resp_position_continuous_;
        return;
    }
    for (int axis = 0; axis < kAxisCount; ++axis) {
        const double previous = resp_position_continuous_(axis);
        double current = unwrap_angle_nearest(previous, st.position(axis));
        if (axis == kBucketAxis && bucket_quaternion_observed) {
            resp_position_continuous_(axis) = current;
            st.position(axis) = current;
            continue;
        }
        double max_delta = std::abs(st.velocity(axis)) * kTs + kPositionJumpGuardMarginRad;
        if (axis == kBucketAxis) {
            max_delta = std::min(max_delta, kBucketMaxPositionStepRad);
        }
        const double delta = current - previous;
        if (delta > max_delta) {
            current = previous + max_delta;
        } else if (delta < -max_delta) {
            current = previous - max_delta;
        }
        resp_position_continuous_(axis) = current;
        st.position(axis) = current;
    }
    for (int axis = 0; axis < kAxisCount; ++axis) {
        if (axis == kBucketAxis && bucket_quaternion_observed) {
            continue;
        }
        st.position(axis) = align_angle_to_reference_branch(st.position(axis), branch_reference(axis));
        resp_position_continuous_(axis) = st.position(axis);
    }
}

bool ExcavatorConverter::hardwareStateToRobotState(const HardwareState& raw_in, RobotState& state_out) {
    const auto* hw = asHardwareState(raw_in);
    auto* st = asState(state_out);
    if (!hw || !st) {
        return false;
    }
    st->status = hw->motor.status;
    st->motor_rpm = swap_joint_2_3_on_first4(hw->motor.motor_rpm);
    const auto rpy_rad = continuousImuRpy(*hw);
    bool position_observed = all_imu_attitudes_observed(*hw);
    if (position_observed) {
        for (const bool ready : imu_rpy_continuous_ready_) {
            if (!ready) {
                position_observed = false;
                break;
            }
        }
    }
    fill_kinematic_from_imu_hw(*hw, rpy_rad, *st);
    // 差分语义：J3=J3-J2，J4=J4-J3（使用变换前值避免串扰）。
    const double theta2_raw = st->position(1);
    const double theta3_raw = st->position(2);
    const double theta4_raw = st->position(3);
    st->position(2) = theta3_raw - theta2_raw;
    st->position(3) = theta4_raw - theta3_raw;
    double bucket_quat_position = 0.0;
    const bool bucket_quaternion_observed = bucket_quaternion_position_rad(*hw, bucket_quat_position);
    if (bucket_quaternion_observed) {
        st->position(3) = bucket_quat_position;
    }
    const Vector8d branch_reference = st->position;

    const double omega2_raw = st->velocity(1);
    const double omega3_raw = st->velocity(2);
    const double omega4_raw = st->velocity(3);
    st->velocity(2) = omega3_raw - omega2_raw;
    st->velocity(3) = omega4_raw - omega3_raw;
    applyPositionContinuity(*st, position_observed, branch_reference, bucket_quaternion_observed);
    double swing_raw_yaw = 0.0;
    if (swing_raw_yaw_reference_rad(*hw, swing_raw_yaw)) {
        st->position(0) =
            align_swing_to_nonnegative_raw_yaw_branch(st->position(0), swing_raw_yaw);
        resp_position_continuous_(0) = st->position(0);
    }
    constexpr std::uint32_t kInitialBiasCycles = 20;
    if (!resp_velocity_bias_ready_) {
        resp_velocity_bias_sum_ += st->velocity;
        ++resp_velocity_bias_count_;
        if (resp_velocity_bias_count_ >= kInitialBiasCycles) {
            resp_velocity_bias_ = resp_velocity_bias_sum_ / static_cast<double>(resp_velocity_bias_count_);
            resp_velocity_bias_ready_ = true;
        }
    }
    if (resp_velocity_bias_ready_) {
        st->velocity -= resp_velocity_bias_;
    }
    return true;
}

}  // namespace excavator
