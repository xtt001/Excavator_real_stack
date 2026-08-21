from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "slave_real_stack.sh"
TRANSITION_WRAPPER = REPO_ROOT / "scripts" / "run_real_transition_expert_recording.sh"


def test_no_camera_stack_configures_gateway_without_camera_source() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'prepare_start "${no_camera}"' in text
    assert 'export EXCAVATOR_NO_CAMERA="${no_camera}"' in text
    assert 'export EXCAVATOR_CAMERA_SOURCE=none' in text
    assert 'camera disabled: read_state images are empty' in text


def test_run_uses_dashboard_log_view_by_default() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'LOG_VIEW="${EXCAVATOR_LOG_VIEW:-dashboard}"' in text
    assert "dashboard_logs()" in text
    assert 'EXCAVATOR_LOG_VIEW=dashboard' in text
    assert '# dashboard | plain' in text
    assert 'tail -n 80 -f "${dir}"/*.log' in text


def test_usbcan_imu_interface_skips_socketcan_helpers() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "is_socketcan_if()" in text
    assert 'IMU_IF="${EXCAVATOR_IMU_IF:-can5}"' in text
    assert 'EXCAVATOR_IMU_IF=usbcan0' in text
    assert 'if is_socketcan_if "${IMU_IF}"; then' in text
    assert 'skipping SocketCAN setup for IMU_IF=${IMU_IF}' in text
    assert 'if ! is_socketcan_if "${IMU_RAW_CAN_LOG_IF}"; then' in text
    assert 'skipping raw IMU candump for non-SocketCAN interface' in text


def test_real_stack_fails_closed_without_locked_jetson_clocks() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'JETSON_NVP_MODEL_ID="${EXCAVATOR_JETSON_NVP_MODEL_ID:-0}"' in text
    assert "require_jetson_performance_access()" in text
    assert "setup_jetson_performance()" in text
    assert 'sudo nvpmodel -m "${JETSON_NVP_MODEL_ID}"' in text
    assert "sudo jetson_clocks" in text
    assert "FreqOverride=1" in text
    assert "setup_jetson_performance\n  prepare_start" in text
    assert 'restart)\n      require_jetson_performance_access\n      stop_stack "${force}"' in text


def test_policy_remote_verbose_test_log_is_opt_in() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'TEST_LOG_DIR="${EXCAVATOR_TEST_LOG_DIR:-}"' in text
    assert 'extra_args+=(--test-log-dir "${TEST_LOG_DIR}")' in text


def test_transition_wrapper_mounts_usb_before_validating_session() -> None:
    stack_text = SCRIPT.read_text(encoding="utf-8")
    wrapper_text = TRANSITION_WRAPPER.read_text(encoding="utf-8")

    assert "mount_usb_device()" in stack_text
    assert "mount-usb)" in stack_text
    assert '"${ROOT_DIR}/scripts/slave_real_stack.sh" mount-usb' in wrapper_text
    assert wrapper_text.index('slave_real_stack.sh" mount-usb') < wrapper_text.index(
        "for name in sequence_manifest.json"
    )


def test_gmsl_restart_restores_trigger_before_capture_and_gates_receiver() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    configure_call = text.index("        configure_gmsl_runtime_trigger")
    capture_start = text.index("        start_service gmsl", configure_call)
    gateway_ready = text.index(
        '  wait_for_port "${GATEWAY_HOST}" "${GATEWAY_PORT}" gateway'
    )
    gate_call = text.index("    check_gmsl_sync_gate", gateway_ready)
    receiver_start = text.index("    start_service receiver", gate_call)

    assert configure_call < capture_start
    assert gateway_ready < gate_call < receiver_start
    assert 'trig_pin=${GMSL_TRIG_PIN}' in text
    assert 'trig_pin:[[:space:]]*${expected_pin}' in text
