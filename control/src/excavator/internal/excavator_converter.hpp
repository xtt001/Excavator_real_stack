#pragma once

#include <excavator/internal/excavator_data_type.hpp>
#include <hal/converter_interface.hpp>

#include <deque>

namespace excavator {

class ExcavatorConverter final : public ConverterInterface {
public:
    ~ExcavatorConverter() override = default;

    bool robotCmdToRobotState(const RobotCommand& cmd, RobotState& state_out) override;
    bool robotStateToHardwareCmd(const RobotState& state, HardwareCommand& cmd_out) override;
    bool hardwareStateToRobotState(const HardwareState& raw_in, RobotState& state_out) override;

private:
    static const ExcavatorCommand* asCommand(const RobotCommand& c) {
        return dynamic_cast<const ExcavatorCommand*>(&c);
    }
    static ExcavatorState* asState(RobotState& s) { return dynamic_cast<ExcavatorState*>(&s); }
    static const ExcavatorState* asState(const RobotState& s) {
        return dynamic_cast<const ExcavatorState*>(&s);
    }
    static ExcavatorHardwareCommand* asHardwareCmd(HardwareCommand& c) {
        return dynamic_cast<ExcavatorHardwareCommand*>(&c);
    }
    static const ExcavatorHardwareState* asHardwareState(const HardwareState& raw) {
        return dynamic_cast<const ExcavatorHardwareState*>(&raw);
    }
    std::array<Eigen::Vector3d, kImuDeviceCount> continuousImuRpy(const ExcavatorHardwareState& hw);
    void applyPositionContinuity(ExcavatorState& st,
                                 bool position_observed,
                                 const Vector8d& branch_reference);
    double bucketContinuousPositionRad(double primary_phase_rad,
                                       double secondary_phase_rad,
                                       double primary_strength,
                                       double secondary_strength,
                                       double initial_output_rad);
    bool bucketGravityHingeCharts(const ExcavatorHardwareState& hw,
                                  double policy_offset_rad,
                                  int median_window,
                                  double& primary_phase_rad,
                                  double& secondary_phase_rad);

    Vector8d resp_velocity_bias_sum_ = Vector8d::Zero();
    Vector8d resp_velocity_bias_ = Vector8d::Zero();
    std::uint32_t resp_velocity_bias_count_{0};
    bool resp_velocity_bias_ready_{false};
    std::array<Eigen::Vector3d, kImuDeviceCount> imu_rpy_continuous_{};
    std::array<bool, kImuDeviceCount> imu_rpy_continuous_ready_{};
    Vector8d resp_position_continuous_ = Vector8d::Zero();
    bool resp_position_continuous_ready_{false};
    double bucket_primary_phase_rad_{0.0};
    double bucket_secondary_phase_rad_{0.0};
    double bucket_position_continuous_rad_{0.0};
    bool bucket_phase_continuous_ready_{false};
    double bucket_gravity_hinge_imu0_phase_rad_{0.0};
    double bucket_gravity_hinge_imu1_phase_rad_{0.0};
    bool bucket_gravity_hinge_phase_ready_{false};
    std::deque<double> bucket_gravity_hinge_outer_zero_window_{};
};

}  // namespace excavator
