#include <excavator/internal/excavator_control.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

class TestState final : public StateInterface {
public:
    bool setControlMode(const ControlMode& mode) override {
        const auto* m = dynamic_cast<const excavator::ExcavatorControlMode*>(&mode);
        if (!m) {
            return false;
        }
        mode_ = *m;
        return true;
    }

    bool getControlMode(ControlMode& mode) const override {
        auto* m = dynamic_cast<excavator::ExcavatorControlMode*>(&mode);
        if (!m) {
            return false;
        }
        *m = mode_;
        return true;
    }

    bool setDeviceState(const DeviceState&) override { return true; }
    bool getDeviceState(DeviceState&) const override { return true; }
    bool setRef(const RobotState&) override { return true; }
    bool getRef(RobotState&) const override { return true; }
    bool setResp(const RobotState&) override { return true; }
    bool getResp(RobotState&) const override { return true; }

    excavator::ExcavatorControlMode mode_{};
};

void require(bool ok, const std::string& message) {
    if (!ok) {
        std::cerr << message << "\n";
        std::exit(1);
    }
}

void test_open_loop_motor_signs() {
    TestState state;
    state.mode_.mode = excavator::ExcavatorControlModeType::OpenLoopMotorSpeed;
    excavator::ExcavatorControl control(state);
    excavator::ExcavatorState ref_in;
    excavator::ExcavatorState resp;
    excavator::ExcavatorCommand cmd;
    excavator::ExcavatorState ref_out;

    cmd.speed_scalar.setZero();
    cmd.speed_scalar(0) = 0.5;  // swing
    cmd.speed_scalar(1) = 0.5;  // boom
    cmd.speed_scalar(2) = 0.5;  // stick
    cmd.speed_scalar(3) = 0.5;  // bucket

    require(control.updateRef(ref_in, resp, &cmd, ref_out), "open-loop update failed");
    require(ref_out.motor_rpm(0) < excavator::kMotorSpeedRawZero,
            "positive swing action should map below raw neutral");
    require(ref_out.motor_rpm(1) > excavator::kMotorSpeedRawZero,
            "positive boom action should map above raw neutral");
    require(ref_out.motor_rpm(2) < excavator::kMotorSpeedRawZero,
            "positive stick action should map below raw neutral");
    require(ref_out.motor_rpm(3) > excavator::kMotorSpeedRawZero,
            "positive bucket action should map above raw neutral");
}

}  // namespace

int main() {
    test_open_loop_motor_signs();
    std::cout << "excavator_control_test OK\n";
    return 0;
}
