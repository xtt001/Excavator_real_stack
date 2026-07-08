#include <excavator/internal/excavator_converter.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

double deg_to_rad(double deg) { return deg * excavator::kPi / 180.0; }

constexpr double kBucketQuaternionPolicyOffsetDeg = -23.262468611471;
constexpr double kBucketGravityHingeOuterBucketDeg = 3.3428268519920032;

Eigen::Quaternionf pitch_quaternion(double pitch_deg) {
    return Eigen::Quaternionf(Eigen::AngleAxisf(
        static_cast<float>(deg_to_rad(pitch_deg)), Eigen::Vector3f::UnitY()));
}

Eigen::Quaternionf yaw_quaternion(double yaw_deg) {
    return Eigen::Quaternionf(Eigen::AngleAxisf(
        static_cast<float>(deg_to_rad(yaw_deg)), Eigen::Vector3f::UnitZ()));
}

#if defined(_WIN32)
void set_env_var(const char* name, const char* value) { _putenv_s(name, value); }
void unset_env_var(const char* name) { _putenv_s(name, ""); }
#else
void set_env_var(const char* name, const char* value) { setenv(name, value, 1); }
void unset_env_var(const char* name) { unsetenv(name); }
#endif

void expect_near(double got, double want, const std::string& message) {
    if (std::abs(got - want) > 1e-6) {
        std::cerr << message << ": got=" << got << " want=" << want << "\n";
        std::exit(1);
    }
}

void expect_not_near(double got, double unwanted, const std::string& message) {
    if (std::abs(got - unwanted) <= 1e-3) {
        std::cerr << message << ": got=" << got << " unwanted=" << unwanted << "\n";
        std::exit(1);
    }
}

void expect_abs_less(double got, double limit, const std::string& message) {
    if (std::abs(got) >= limit) {
        std::cerr << message << ": got=" << got << " limit=" << limit << "\n";
        std::exit(1);
    }
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

void set_bucket_quaternions(excavator::ExcavatorHardwareState& hw,
                            const Eigen::Quaternionf& imu1,
                            const Eigen::Quaternionf& imu2) {
    hw.imu.devices[0].quaternion = imu1.normalized();
    hw.imu.devices[1].quaternion = imu2.normalized();
}

void set_bucket_outer_gravity_accel(excavator::ExcavatorHardwareState& hw) {
    hw.imu.devices[0].accel_mps2 = Eigen::Vector3f(-0.11456954F, 2.79999995F, 9.5F);
    hw.imu.devices[1].accel_mps2 = Eigen::Vector3f(6.90000010F, 0.0F, 7.10066216F);
}

Eigen::Quaternionf q_wxyz(double w, double x, double y, double z) {
    return Eigen::Quaternionf(static_cast<float>(w),
                              static_cast<float>(x),
                              static_cast<float>(y),
                              static_cast<float>(z));
}

void test_raw_yaw_branch_survives_startup() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState invalid_hw;
    excavator::ExcavatorState out;
    if (converter.hardwareStateToRobotState(invalid_hw, out)) {
        std::cerr << "invalid startup conversion should not publish robot state\n";
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
    expect_near(out.position(3), deg_to_rad(22.50 + kBucketQuaternionPolicyOffsetDeg), "bucket wrong");
}

void test_default_valid_attitude_does_not_lock_yaw_branch() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 0.0, 0.0, 0.0, 0.0);
    if (converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "default valid snapshot should not publish before qpos is ready\n";
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

void test_bucket_uses_quaternion_policy_frame_for_133653_csv_fixture() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 11.6899995803833, -48.66999816894531, 0.0, 222.50);
    set_bucket_quaternions(
        hw,
        q_wxyz(0.09324929118156433, 0.10031034052371979, 0.01631910540163517,
               -0.9904467463493347),
        q_wxyz(0.7082435488700867, 0.2577011287212372, -0.3223798871040344,
               0.5727660655975342));
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket 133653 csv conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(35.580032040155814),
                "bucket should match 133653 calibrated policy qpos");
    expect_not_near(out.position(3), deg_to_rad(60.35999774932861),
                    "bucket must not use direct raw pitch diff for 133653");
}

void test_bucket_corrects_130940_csv_fixture_without_old_branch() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 11.679999351501465, -48.96999740600586, 0.0, 222.50);
    set_bucket_quaternions(
        hw,
        q_wxyz(0.09814400225877762, 0.10027111321687698, 0.015918726101517677,
               -0.9899841547012329),
        q_wxyz(0.7065725922584534, 0.26049181818962097, -0.3232657313346863,
               0.5730680227279663));
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket 130940 csv conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(36.1900328319635),
                "bucket should repair 130940 to calibrated raw-imu qpos");
    expect_not_near(out.position(3), deg_to_rad(11.670000076293945),
                    "bucket must not keep old wrong final branch for 130940");
    expect_not_near(out.position(3), deg_to_rad(60.649996757507324),
                    "bucket must not use direct raw pitch diff for 130940");
}

void test_bucket_uses_secondary_quaternion_chart_near_primary_singularity() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -24.0, -47.0, 0.0, 222.50);
    set_bucket_quaternions(
        hw,
        q_wxyz(-0.018099, 0.508006, 0.013423, 0.861059),
        q_wxyz(1.0, 0.0, 0.0, 0.0));
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket secondary-chart baseline conversion failed\n";
        std::exit(1);
    }
    const double previous_bucket = out.position(3);

    set_bucket_quaternions(
        hw,
        q_wxyz(0.008364, 0.503180, 0.018671, 0.863939),
        q_wxyz(1.0, 0.0, 0.0, 0.0));
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket secondary-chart crossing conversion failed\n";
        std::exit(1);
    }

    expect_abs_less(out.position(3) - previous_bucket, deg_to_rad(2.0),
                    "bucket should use secondary quaternion chart near primary atan2 singularity");
    expect_not_near(out.position(3), deg_to_rad(108.47552842420737),
                    "bucket must not publish primary-chart singular branch");
}

void test_bucket_calibrated_value_ignores_gyro_and_restart_history() {
    excavator::ExcavatorConverter converter_with_history;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState with_history;

    make_valid_imu(hw, 0.0, 0.0, 0.0, 222.50);
    if (!converter_with_history.hardwareStateToRobotState(hw, with_history)) {
        std::cerr << "bucket restart-history baseline conversion failed\n";
        std::exit(1);
    }

    make_valid_imu(hw, 222.50, 0.0, 0.0, 222.50);
    hw.imu.devices[0].quaternion = pitch_quaternion(59.40);
    hw.imu.devices[1].quaternion = pitch_quaternion(0.0);
    hw.imu.devices[0].gyro_dps(1) = 0.0F;
    hw.imu.devices[1].gyro_dps(1) = 0.0F;
    if (!converter_with_history.hardwareStateToRobotState(hw, with_history)) {
        std::cerr << "bucket restart-history raw conversion failed\n";
        std::exit(1);
    }

    excavator::ExcavatorConverter converter_without_history;
    excavator::ExcavatorState without_history;
    if (!converter_without_history.hardwareStateToRobotState(hw, without_history)) {
        std::cerr << "bucket no-history raw conversion failed\n";
        std::exit(1);
    }

    const double expected = deg_to_rad(59.40 + kBucketQuaternionPolicyOffsetDeg);
    expect_near(with_history.position(3), expected, "bucket history must not change calibrated qpos");
    expect_near(without_history.position(3), expected, "bucket startup must use same calibrated qpos");
    expect_not_near(with_history.position(3), deg_to_rad(222.50),
                    "bucket must not use direct raw pitch branch as calibrated qpos");
}

void test_fresh_invalid_bucket_quaternion_does_not_publish_uncalibrated_qpos() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 123.0, 0.0, 0.0, 222.50);
    hw.imu.devices[0].valid_quaternion = 0U;
    hw.imu.devices[0].rpy_rad(1) = static_cast<float>(deg_to_rad(123.0));
    hw.imu.devices[0].rpy_raw_deg(1) = 123.0F;
    if (converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "fresh invalid bucket quaternion should not publish uncalibrated bucket qpos\n";
        std::exit(1);
    }
}

void test_ready_invalid_bucket_quaternion_holds_previous_calibrated_qpos() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 10.0, 20.0, 30.0, 222.50);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "invalid quaternion hold baseline conversion failed\n";
        std::exit(1);
    }
    const double previous_bucket = out.position(3);

    hw.imu.devices[0].valid_quaternion = 0U;
    hw.imu.devices[0].rpy_raw_deg(1) = 200.0F;
    hw.imu.devices[0].rpy_rad(1) = static_cast<float>(deg_to_rad(-160.0));
    hw.imu.devices[0].quaternion = pitch_quaternion(-160.0);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "ready invalid bucket quaternion should hold previous qpos\n";
        std::exit(1);
    }

    expect_near(out.position(3), previous_bucket,
                "invalid bucket quaternion must hold previous calibrated qpos");
    expect_not_near(out.position(3), deg_to_rad(-160.0),
                    "invalid bucket quaternion must not publish folded rpy bucket");
}

void test_boom_raw_deg_ignores_restart_history() {
    excavator::ExcavatorConverter converter_with_history;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState with_history;

    make_valid_imu(hw, 0.0, 0.0, 0.0, 222.50);
    if (!converter_with_history.hardwareStateToRobotState(hw, with_history)) {
        std::cerr << "boom restart-history baseline conversion failed\n";
        std::exit(1);
    }

    make_valid_imu(hw, 0.0, 0.0, 222.50, 222.50);
    hw.imu.devices[2].rpy_rad(1) = static_cast<float>(deg_to_rad(-137.50));
    hw.imu.devices[2].quaternion = pitch_quaternion(-137.50);
    if (!converter_with_history.hardwareStateToRobotState(hw, with_history)) {
        std::cerr << "boom restart-history raw conversion failed\n";
        std::exit(1);
    }

    excavator::ExcavatorConverter converter_without_history;
    excavator::ExcavatorState without_history;
    if (!converter_without_history.hardwareStateToRobotState(hw, without_history)) {
        std::cerr << "boom no-history raw conversion failed\n";
        std::exit(1);
    }

    expect_near(with_history.position(1), deg_to_rad(222.50), "boom history must not create another 2pi branch");
    expect_near(without_history.position(1), deg_to_rad(222.50), "boom startup must use same raw-deg branch");
}

void test_stick_raw_deg_ignores_restart_history() {
    excavator::ExcavatorConverter converter_with_history;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState with_history;

    make_valid_imu(hw, 0.0, 0.0, 0.0, 222.50);
    if (!converter_with_history.hardwareStateToRobotState(hw, with_history)) {
        std::cerr << "stick restart-history baseline conversion failed\n";
        std::exit(1);
    }

    make_valid_imu(hw, 0.0, 222.50, 0.0, 222.50);
    hw.imu.devices[1].rpy_rad(1) = static_cast<float>(deg_to_rad(-137.50));
    hw.imu.devices[1].quaternion = pitch_quaternion(-137.50);
    if (!converter_with_history.hardwareStateToRobotState(hw, with_history)) {
        std::cerr << "stick restart-history raw conversion failed\n";
        std::exit(1);
    }

    excavator::ExcavatorConverter converter_without_history;
    excavator::ExcavatorState without_history;
    if (!converter_without_history.hardwareStateToRobotState(hw, without_history)) {
        std::cerr << "stick no-history raw conversion failed\n";
        std::exit(1);
    }

    expect_near(with_history.position(2), deg_to_rad(222.50), "stick history must not create another 2pi branch");
    expect_near(without_history.position(2), deg_to_rad(222.50), "stick startup must use same raw-deg branch");
}

void test_offline_imu_holds_previous_raw_deg_qpos() {
    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 10.0, 20.0, 30.0, 222.50);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "offline hold baseline conversion failed\n";
        std::exit(1);
    }
    const double previous_bucket = out.position(3);
    expect_near(previous_bucket, deg_to_rad(-10.0 + kBucketQuaternionPolicyOffsetDeg),
                "offline hold baseline bucket wrong");

    hw.imu.devices[0].online = 0U;
    hw.imu.devices[0].valid_attitude = 1U;
    hw.imu.devices[0].host_rx_time_ns = 9000U;
    hw.imu.devices[0].rpy_raw_deg(1) = 200.0F;
    hw.imu.devices[0].rpy_rad(1) = static_cast<float>(deg_to_rad(-160.0));
    hw.imu.devices[0].quaternion = pitch_quaternion(-160.0);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "offline hold conversion failed\n";
        std::exit(1);
    }

    expect_near(out.position(3), previous_bucket, "offline imu raw pitch must not update bucket qpos");
}

void test_bucket_roll_ccw90_profile_uses_native_rpy_reference_and_matching_qvel() {
    set_env_var("EXCAVATOR_BUCKET_QPOS_SOURCE", "rpy");
    set_env_var("EXCAVATOR_BUCKET_IMU0_PROFILE", "roll_ccw90");
    set_env_var("EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD", "0.12217304763960307");
    set_env_var("EXCAVATOR_BUCKET_IMU0_SIGN", "1");

    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 0.0, 3.0, 0.0, 222.50);
    hw.imu.devices[0].rpy_raw_deg(0) = 10.0F;
    hw.imu.devices[0].rpy_rad(0) = static_cast<float>(deg_to_rad(10.0));
    hw.imu.devices[0].valid_quaternion = 0U;
    hw.imu.devices[1].valid_quaternion = 0U;
    hw.imu.devices[0].gyro_dps(0) = 20.0F;
    hw.imu.devices[0].gyro_dps(1) = 200.0F;
    hw.imu.devices[1].gyro_dps(1) = 5.0F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket roll_ccw90 baseline conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), 0.0,
                "roll_ccw90 bucket should use imu0.roll - imu1.pitch - reference");
    expect_near(out.velocity(3), deg_to_rad(-15.0),
                "roll_ccw90 bucket velocity should match derivative sign of native RPY qpos");

    hw.imu.devices[0].rpy_raw_deg(0) = 15.0F;
    hw.imu.devices[0].rpy_rad(0) = static_cast<float>(deg_to_rad(15.0));
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket roll_ccw90 delta conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(5.0),
                "roll_ccw90 bucket should follow native RPY relative angle, not startup history");

    unset_env_var("EXCAVATOR_BUCKET_QPOS_SOURCE");
    unset_env_var("EXCAVATOR_BUCKET_IMU0_PROFILE");
    unset_env_var("EXCAVATOR_BUCKET_IMU0_REFERENCE_RAD");
    unset_env_var("EXCAVATOR_BUCKET_IMU0_SIGN");
}

void test_bucket_gravity_hinge_source_ignores_upper_joint_rpy_coupling() {
    set_env_var("EXCAVATOR_BUCKET_QPOS_SOURCE", "gravity_hinge");
    set_env_var("EXCAVATOR_BUCKET_IMU0_PROFILE", "roll_ccw90");

    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, -54.0, -5.0, 2.0, 245.0);
    hw.imu.devices[0].rpy_raw_deg(0) = 35.0F;
    hw.imu.devices[0].rpy_rad(0) = static_cast<float>(deg_to_rad(35.0));
    set_bucket_outer_gravity_accel(hw);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket gravity hinge baseline conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(kBucketGravityHingeOuterBucketDeg),
                "gravity hinge bucket should use outer calibration policy coordinate");

    make_valid_imu(hw, -12.0, 24.0, 31.0, 350.0);
    hw.imu.devices[0].rpy_raw_deg(0) = -80.0F;
    hw.imu.devices[0].rpy_rad(0) = static_cast<float>(deg_to_rad(-80.0));
    set_bucket_outer_gravity_accel(hw);
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "bucket gravity hinge upper-motion conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(3), deg_to_rad(kBucketGravityHingeOuterBucketDeg),
                "gravity hinge bucket should not follow unrelated upper joint RPY changes");

    unset_env_var("EXCAVATOR_BUCKET_QPOS_SOURCE");
    unset_env_var("EXCAVATOR_BUCKET_IMU0_PROFILE");
}

void test_daoyuan_chain_profile_decouples_upper_joints() {
    set_env_var("EXCAVATOR_JOINT_RPY_PROFILE", "daoyuan_chain");
    set_env_var("EXCAVATOR_BUCKET_QPOS_SOURCE", "daoyuan_chain");
    set_env_var("EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD", "0");
    set_env_var("EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD", "0");

    excavator::ExcavatorConverter converter;
    excavator::ExcavatorHardwareState hw;
    excavator::ExcavatorState out;

    make_valid_imu(hw, 0.0, -80.0, -5.0, 10.0);
    hw.imu.devices[0].rpy_raw_deg(0) = -130.0F;
    hw.imu.devices[0].rpy_rad(0) = static_cast<float>(deg_to_rad(-130.0));
    hw.imu.devices[0].gyro_dps(0) = 11.0F;
    hw.imu.devices[1].gyro_dps(1) = 7.0F;
    hw.imu.devices[2].gyro_dps(1) = 3.0F;
    hw.imu.devices[3].gyro_dps(2) = -5.0F;
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "daoyuan chain baseline conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(0), deg_to_rad(10.0), "daoyuan swing qpos wrong");
    expect_near(out.position(1), deg_to_rad(-5.0), "daoyuan boom qpos wrong");
    expect_near(out.position(2), deg_to_rad(-85.0), "daoyuan stick qpos should be stick+boom");
    expect_near(out.position(3), deg_to_rad(210.0),
                "daoyuan bucket qpos should be -(bucket.roll+stick.pitch)");
    expect_near(out.velocity(0), deg_to_rad(5.0), "daoyuan swing qvel wrong");
    expect_near(out.velocity(1), deg_to_rad(3.0), "daoyuan boom qvel wrong");
    expect_near(out.velocity(2), deg_to_rad(10.0), "daoyuan stick qvel should be stick+boom");
    expect_near(out.velocity(3), deg_to_rad(-18.0),
                "daoyuan bucket qvel should be -(bucket.roll+stick.pitch)");

    make_valid_imu(hw, 0.0, -70.0, -15.0, 10.0);
    hw.imu.devices[0].rpy_raw_deg(0) = -140.0F;
    hw.imu.devices[0].rpy_rad(0) = static_cast<float>(deg_to_rad(-140.0));
    if (!converter.hardwareStateToRobotState(hw, out)) {
        std::cerr << "daoyuan chain coupled-motion conversion failed\n";
        std::exit(1);
    }
    expect_near(out.position(1), deg_to_rad(-15.0), "daoyuan boom should follow boom IMU");
    expect_near(out.position(2), deg_to_rad(-85.0),
                "daoyuan stick should reject equal-and-opposite boom/stick pitch changes");
    expect_near(out.position(3), deg_to_rad(210.0),
                "daoyuan bucket should reject equal-and-opposite bucket/stick changes");

    unset_env_var("EXCAVATOR_JOINT_RPY_PROFILE");
    unset_env_var("EXCAVATOR_BUCKET_QPOS_SOURCE");
    unset_env_var("EXCAVATOR_DAOYUAN_STICK_POLICY_OFFSET_RAD");
    unset_env_var("EXCAVATOR_DAOYUAN_BUCKET_POLICY_OFFSET_RAD");
}

}  // namespace

int main() {
    test_raw_yaw_branch_survives_startup();
    test_default_valid_attitude_does_not_lock_yaw_branch();
    test_yaw_zero_crossing_is_unwrapped();
    test_swing_velocity_sign_matches_raw_yaw();
    test_bucket_uses_quaternion_policy_frame_for_133653_csv_fixture();
    test_bucket_corrects_130940_csv_fixture_without_old_branch();
    test_bucket_uses_secondary_quaternion_chart_near_primary_singularity();
    test_bucket_calibrated_value_ignores_gyro_and_restart_history();
    test_fresh_invalid_bucket_quaternion_does_not_publish_uncalibrated_qpos();
    test_ready_invalid_bucket_quaternion_holds_previous_calibrated_qpos();
    test_boom_raw_deg_ignores_restart_history();
    test_stick_raw_deg_ignores_restart_history();
    test_offline_imu_holds_previous_raw_deg_qpos();
    test_bucket_roll_ccw90_profile_uses_native_rpy_reference_and_matching_qvel();
    test_bucket_gravity_hinge_source_ignores_upper_joint_rpy_coupling();
    test_daoyuan_chain_profile_decouples_upper_joints();
    std::cout << "excavator_converter_test OK\n";
    return 0;
}
