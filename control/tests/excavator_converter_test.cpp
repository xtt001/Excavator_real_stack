#include <excavator/internal/excavator_converter.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

constexpr double kBucketQuaternionPolicyOffsetRad = -0.4060066694119653;

double deg_to_rad(double deg) { return deg * excavator::kPi / 180.0; }

Eigen::Quaternionf pitch_quaternion(double pitch_deg) {
    return Eigen::Quaternionf(Eigen::AngleAxisf(
        static_cast<float>(deg_to_rad(pitch_deg)), Eigen::Vector3f::UnitY()));
}

Eigen::Quaternionf yaw_quaternion(double yaw_deg) {
    return Eigen::Quaternionf(Eigen::AngleAxisf(
        static_cast<float>(deg_to_rad(yaw_deg)), Eigen::Vector3f::UnitZ()));
}

void expect_near(double got, double want, const std::string& message) {
    if (std::abs(got - want) > 1e-6) {
        std::cerr << message << ": got=" << got << " want=" << want << "\n";
        std::exit(1);
    }
}

void expect_less(double got, double limit, const std::string& message) {
    if (!(got < limit)) {
        std::cerr << message << ": got=" << got << " limit=" << limit << "\n";
        std::exit(1);
    }
}

void set_bucket_quaternion_offset(double value) {
    const std::string raw = std::to_string(value);
    setenv("EXCAVATOR_BUCKET_QUATERNION_OFFSET_RAD", raw.c_str(), 1);
}

void make_valid_imu(excavator::ExcavatorHardwareState& hw,
                    double imu1_pitch_deg,
                    double imu2_pitch_deg,
                    double imu3_pitch_deg,
                    double imu4_yaw_deg) {
    const double pitch_deg[4] = {imu1_pitch_deg, imu2_pitch_deg, imu3_pitch_deg, 0.0};
    for (std::size_t i = 0; i < excavator::kImuDeviceCount; ++i) {
        auto& imu = hw.imu.devices[i];
        imu.device_addr = static_cast<std::uint8_t>(i + 1U);
        imu.online = 1U;
        imu.valid_attitude = 1U;
        imu.valid_quaternion = 1U;
        imu.valid_gyro = 1U;
        imu.valid_accel = 1U;
        imu.host_rx_time_ns = 1000U + static_cast<std::uint64_t>(i);
        imu.rpy_raw_deg.setZero();
        imu.rpy_rad.setZero();
        imu.rpy_raw_deg(1) = static_cast<float>(pitch_deg[i]);
        imu.rpy_rad(1) = static_cast<float>(deg_to_rad(pitch_deg[i]));
        imu.quaternion = pitch_quaternion(pitch_deg[i]);
    }
    hw.imu.devices[3].rpy_raw_deg(2) = static_cast<float>(imu4_yaw_deg);
    const double imu4_yaw_folded_deg = imu4_yaw_deg > 180.0 ? imu4_yaw_deg - 360.0 : imu4_yaw_deg;
    hw.imu.devices[3].rpy_rad(2) = static_cast<float>(deg_to_rad(imu4_yaw_folded_deg));
    hw.imu.devices[3].quaternion = yaw_quaternion(imu4_yaw_deg);
}

void test_raw_yaw_branch_survives_startup() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState invalid_hw;
    excavator::ExcavatorState out;
    if (!converter.hardwareStateToRobotState(invalid_hw, out)) {
        std::cerr << "invalid startup conversion failed\n";
        std::exit(1);
    }

    excavator::ExcavatorHardwareState hw;
    make_valid_imu(hw, -44.82, -67.32, 32.53, 222.50);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "valid conversion failed\n";
        std::exit(1);
    }

    expect_near(out.position(0), deg_to_rad(222.50), "swing should use raw yaw branch");
    expect_near(out.position(1), deg_to_rad(32.53), "boom wrong");
    expect_near(out.position(2), deg_to_rad(-99.85), "stick wrong");
    expect_near(out.position(3), deg_to_rad(22.50), "bucket wrong");
}

void test_default_valid_attitude_does_not_lock_yaw_branch() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 0.0, 0.0, 0.0, 0.0);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "default valid conversion failed\n";
        std::exit(1);
    }

    make_valid_imu(hw, -44.82, -67.32, 32.53, 269.78);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "raw yaw after default conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(0), deg_to_rad(269.78), "default valid sample should not lock swing to folded branch");
}

void test_yaw_zero_crossing_is_unwrapped() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -44.82, -67.32, 32.53, 359.50);
    hw.imu.devices[3].gyro_dps(2) = 50.0F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "first crossing conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(0), deg_to_rad(359.50), "initial crossing branch wrong");

    make_valid_imu(hw, -44.82, -67.32, 32.53, 1.00);
    hw.imu.devices[3].gyro_dps(2) = 50.0F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "second crossing conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(0), deg_to_rad(361.00), "yaw crossing should unwrap above 360");
}

void test_swing_velocity_sign_matches_raw_yaw() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -44.82, -67.32, 32.53, 222.50);
    hw.imu.devices[3].gyro_dps(2) = -5.0F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "swing velocity conversion failed\n";
        std::exit(1);
    }
    expect_near(out.velocity(0), deg_to_rad(5.0), "swing velocity sign should match yaw change");
}

void test_bucket_uses_relative_quaternion_when_euler_branch_flips() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 24.76, -35.08, 0.0, 222.50);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket quaternion baseline conversion failed\n";
        std::exit(1);
    }
    const double previous = out.position(3);
    expect_near(previous, deg_to_rad(59.84), "bucket quaternion baseline wrong");

    make_valid_imu(hw, -56.47, -34.14, 0.0, 222.50);
    hw.imu.devices[0].quaternion = pitch_quaternion(25.26);
    hw.imu.devices[1].quaternion = pitch_quaternion(-34.14);
    hw.imu.devices[0].gyro_dps(1) = -10.0F;
    hw.imu.devices[1].gyro_dps(1) = 0.0F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket quaternion branch-flip conversion failed\n";
        std::exit(1);
    }

    expect_less(out.position(3), previous, "bucket quaternion should allow the small physical step");
    expect_near(out.position(3), deg_to_rad(59.40), "bucket quaternion should ignore folded Euler pitch");
}

void test_bucket_falls_back_to_euler_when_quaternion_missing() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -20.0, -30.0, 0.0, 222.50);
    hw.imu.devices[0].valid_quaternion = 0U;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket missing quaternion fallback conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(10.0), "missing bucket quaternion should fall back to Euler");
}

void test_bucket_quaternion_position_is_absolute() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -20.0, -30.0, 0.0, 222.50);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket baseline conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(10.0), "bucket baseline wrong");

    make_valid_imu(hw, 10.0, -30.0, 0.0, 222.50);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket absolute quaternion conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(40.0), "valid bucket quaternion should not be rate limited");
}

void test_bucket_euler_fallback_is_rate_limited_when_quaternion_missing() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -20.0, -30.0, 0.0, 222.50);
    hw.imu.devices[0].valid_quaternion = 0U;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket euler fallback baseline conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(10.0), "bucket euler fallback baseline wrong");

    make_valid_imu(hw, 10.0, -30.0, 0.0, 222.50);
    hw.imu.devices[0].valid_quaternion = 0U;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket euler fallback outlier conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(10.5), "missing quaternion fallback should be rate limited");
}

void test_bucket_quaternion_anchors_through_euler_branch_fold() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -88.84, -31.66, 0.0, 222.50);
    hw.imu.devices[0].gyro_dps(1) = -14.2F;
    hw.imu.devices[1].gyro_dps(1) = -0.7F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket quaternion branch-fold baseline conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(-57.18), "bucket quaternion branch-fold baseline wrong");

    make_valid_imu(hw, -88.75, -31.67, 0.0, 222.50);
    hw.imu.devices[0].gyro_dps(1) = -14.3F;
    hw.imu.devices[1].gyro_dps(1) = -0.6F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket quaternion branch-fold next conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(-57.08), "valid quaternion should anchor bucket through Euler branch fold");
}

void test_bucket_quaternion_raw_jump_uses_absolute_anchor() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -54.20, -31.33, 0.0, 222.50);
    hw.imu.devices[0].gyro_dps(1) = 3.8F;
    hw.imu.devices[1].gyro_dps(1) = -0.5F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket raw jump baseline conversion failed\n";
        std::exit(1);
    }
    const double previous = out.position(3);
    expect_near(previous, deg_to_rad(-22.87), "bucket raw jump baseline wrong");

    make_valid_imu(hw, -85.99, -31.78, 0.0, 222.50);
    hw.imu.devices[0].gyro_dps(1) = 1.4F;
    hw.imu.devices[1].gyro_dps(1) = -0.8F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket raw jump conversion failed\n";
        std::exit(1);
    }

    expect_less(out.position(3), previous, "bucket raw jump should follow quaternion absolute anchor");
    expect_near(out.position(3), deg_to_rad(-54.21), "valid quaternion bucket should not gyro-integrate raw jumps");
}

void test_bucket_quaternion_does_not_accumulate_alias_state() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -54.20, -31.33, 0.0, 222.50);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket alias hold baseline conversion failed\n";
        std::exit(1);
    }

    for (int i = 0; i < 80; ++i) {
        make_valid_imu(hw, -34.40, -38.79, 0.0, 222.50);
        hw.imu.devices[0].gyro_dps(1) = -51.0F;
        hw.imu.devices[1].gyro_dps(1) = -1.0F;
        if (!converter.hardwareStateToRobotState(hw, out)) {
            std::cerr << "bucket alias integration conversion failed\n";
            std::exit(1);
        }
    }
    expect_near(out.position(3), deg_to_rad(4.39), "valid quaternion should not integrate alias state");

    make_valid_imu(hw, -42.80, -36.80, 0.0, 222.50);
    hw.imu.devices[0].gyro_dps(1) = -1.2F;
    hw.imu.devices[1].gyro_dps(1) = -1.0F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket quaternion absolute re-anchor conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(-6.0), "valid quaternion should keep using absolute bucket qpos");
}

void test_bucket_quaternion_applies_legacy_policy_offset() {
    set_bucket_quaternion_offset(kBucketQuaternionPolicyOffsetRad);
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -43.66, -37.59, 5.77, 216.70);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket quaternion policy-offset conversion failed\n";
        std::exit(1);
    }

    expect_near(out.position(3),
                deg_to_rad(-6.07) + kBucketQuaternionPolicyOffsetRad,
                "bucket quaternion should preserve legacy policy qpos frame");
    set_bucket_quaternion_offset(0.0);
}

void test_bucket_quaternion_keeps_unwrapped_branch_across_pi() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 179.0, 0.0, 0.0, 222.50);
    hw.imu.devices[0].quaternion = pitch_quaternion(179.0);
    hw.imu.devices[1].quaternion = pitch_quaternion(0.0);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket pi branch baseline conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(179.0), "bucket pi branch baseline wrong");

    make_valid_imu(hw, -179.0, 0.0, 0.0, 222.50);
    hw.imu.devices[0].quaternion = pitch_quaternion(-179.0);
    hw.imu.devices[1].quaternion = pitch_quaternion(0.0);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket pi branch crossing conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(181.0), "bucket quaternion should keep the unwrapped branch");
}

}  // namespace

int main() {
    set_bucket_quaternion_offset(0.0);
    test_raw_yaw_branch_survives_startup();
    test_default_valid_attitude_does_not_lock_yaw_branch();
    test_yaw_zero_crossing_is_unwrapped();
    test_swing_velocity_sign_matches_raw_yaw();
    test_bucket_uses_relative_quaternion_when_euler_branch_flips();
    test_bucket_falls_back_to_euler_when_quaternion_missing();
    test_bucket_quaternion_position_is_absolute();
    test_bucket_euler_fallback_is_rate_limited_when_quaternion_missing();
    test_bucket_quaternion_anchors_through_euler_branch_fold();
    test_bucket_quaternion_raw_jump_uses_absolute_anchor();
    test_bucket_quaternion_does_not_accumulate_alias_state();
    test_bucket_quaternion_applies_legacy_policy_offset();
    test_bucket_quaternion_keeps_unwrapped_branch_across_pi();
    std::cout << "excavator_converter_test OK\n";
    return 0;
}
