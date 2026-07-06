#include <excavator/internal/excavator_converter.hpp>

#include <cmath>

namespace excavator {
namespace {

constexpr float kUninitializedAttitudeEps = 1e-6F;
constexpr double kBucketQuaternionPolicyOffsetRad = -0.4060066694119653;
constexpr double kBucketPrimaryChartMinStrength = 0.35;

struct BucketQuaternionCharts {
    double primary_phase_rad{0.0};
    double secondary_phase_rad{0.0};
    double primary_strength{0.0};
    double secondary_strength{0.0};
};

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
        if (imu.online == 0U || imu.valid_attitude == 0U || imu.host_rx_time_ns == 0U) {
            return false;
        }
    }
    return true;
}

bool normalized_observed_quaternion(const ExcavatorImuHardwareState::ImuSample& src,
                                    Eigen::Quaterniond& out) noexcept {
    if (src.online == 0U || src.valid_quaternion == 0U || src.host_rx_time_ns == 0U) {
        return false;
    }
    const Eigen::Quaterniond q = src.quaternion.cast<double>();
    if (!std::isfinite(q.w()) || !std::isfinite(q.x()) || !std::isfinite(q.y()) ||
        !std::isfinite(q.z())) {
        return false;
    }
    const double norm = q.norm();
    if (!std::isfinite(norm) || norm <= 1e-9) {
        return false;
    }
    out = q;
    out.normalize();
    return true;
}

double signed_twist_angle_rad(const Eigen::Quaterniond& rotation,
                              const Eigen::Vector3d& axis) noexcept {
    return std::remainder(2.0 * std::atan2(rotation.vec().dot(axis), rotation.w()), 2.0 * kPi);
}

bool bucket_quaternion_charts(const ExcavatorHardwareState& hw,
                              BucketQuaternionCharts& out) noexcept {
    Eigen::Quaterniond imu1;
    Eigen::Quaterniond imu2;
    if (!normalized_observed_quaternion(hw.imu.devices[0], imu1) ||
        !normalized_observed_quaternion(hw.imu.devices[1], imu2)) {
        return false;
    }
    Eigen::Quaterniond relative = imu2.conjugate() * imu1;
    relative.normalize();
    out.primary_phase_rad =
        signed_twist_angle_rad(relative, Eigen::Vector3d::UnitY()) +
        kBucketQuaternionPolicyOffsetRad;
    out.secondary_phase_rad =
        std::remainder(-2.0 * std::atan2(relative.x(), relative.z()), 2.0 * kPi);
    out.primary_strength = std::hypot(relative.w(), relative.y());
    out.secondary_strength = std::hypot(relative.x(), relative.z());
    return std::isfinite(out.primary_phase_rad) && std::isfinite(out.secondary_phase_rad) &&
           std::isfinite(out.primary_strength) && std::isfinite(out.secondary_strength);
}

bool looks_like_uninitialized_attitude(const ExcavatorImuHardwareState::ImuSample& src) noexcept {
    return src.rpy_raw_deg.cwiseAbs().maxCoeff() <= kUninitializedAttitudeEps &&
           src.rpy_rad.cwiseAbs().maxCoeff() <= kUninitializedAttitudeEps;
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
                                                 const Vector8d& branch_reference) {
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
        st.position(axis) = align_angle_to_reference_branch(st.position(axis), branch_reference(axis));
        resp_position_continuous_(axis) = st.position(axis);
    }
}

double ExcavatorConverter::bucketContinuousPositionRad(double primary_phase_rad,
                                                       double secondary_phase_rad,
                                                       double primary_strength,
                                                       double secondary_strength) {
    if (!bucket_phase_continuous_ready_) {
        bucket_primary_phase_rad_ = primary_phase_rad;
        bucket_secondary_phase_rad_ = secondary_phase_rad;
        bucket_position_continuous_rad_ = primary_phase_rad;
        bucket_phase_continuous_ready_ = true;
        return bucket_position_continuous_rad_;
    }

    const bool use_secondary_chart =
        primary_strength < kBucketPrimaryChartMinStrength &&
        secondary_strength > primary_strength;
    const double primary_delta =
        std::remainder(primary_phase_rad - bucket_primary_phase_rad_, 2.0 * kPi);
    const double secondary_delta =
        std::remainder(secondary_phase_rad - bucket_secondary_phase_rad_, 2.0 * kPi);
    bucket_position_continuous_rad_ += use_secondary_chart ? secondary_delta : primary_delta;
    bucket_primary_phase_rad_ = primary_phase_rad;
    bucket_secondary_phase_rad_ = secondary_phase_rad;
    return bucket_position_continuous_rad_;
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
    BucketQuaternionCharts bucket_charts;
    const bool bucket_observed = bucket_quaternion_charts(*hw, bucket_charts);
    bool position_observed = all_imu_attitudes_observed(*hw) && bucket_observed;
    if (position_observed) {
        for (const bool ready : imu_rpy_continuous_ready_) {
            if (!ready) {
                position_observed = false;
                break;
            }
        }
    }
    const bool can_publish_position = position_observed || resp_position_continuous_ready_;
    // Keep state_out populated for diagnostics, but do not report a publishable
    // robot state until qpos has a valid branch or a previous value to hold.
    fill_kinematic_from_imu_hw(*hw, rpy_rad, *st);
    // boom/stick 使用 IMU raw deg canonical branch；bucket 使用固定 policy-frame quaternion 标定。
    st->position(1) = deg_to_rad(static_cast<double>(hw->imu.devices[2].rpy_raw_deg(1)));
    st->position(2) = deg_to_rad(static_cast<double>(hw->imu.devices[1].rpy_raw_deg(1)) -
                                 static_cast<double>(hw->imu.devices[2].rpy_raw_deg(1)));
    if (bucket_observed) {
        st->position(3) = bucket_charts.primary_phase_rad;
    }

    const double omega2_raw = st->velocity(1);
    const double omega3_raw = st->velocity(2);
    const double omega4_raw = st->velocity(3);
    st->velocity(2) = omega3_raw - omega2_raw;
    st->velocity(3) = omega4_raw - omega3_raw;
    if (!can_publish_position) {
        return false;
    }
    if (position_observed) {
        st->position(3) = bucketContinuousPositionRad(bucket_charts.primary_phase_rad,
                                                     bucket_charts.secondary_phase_rad,
                                                     bucket_charts.primary_strength,
                                                     bucket_charts.secondary_strength);
    }
    const Vector8d branch_reference = st->position;
    applyPositionContinuity(*st, position_observed, branch_reference);
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
