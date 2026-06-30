from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "bring_up_gmsl_cameras.sh"


def _run_config_check(*, devices: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GMSL_PRINT_CONFIG_ONLY"] = "1"
    if devices is None:
        env.pop("GMSL_VIDEO_DEVICES", None)
    else:
        env["GMSL_VIDEO_DEVICES"] = devices
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bring_up_gmsl_cameras_keeps_video6_video7_default() -> None:
    result = _run_config_check()

    assert result.returncode == 0
    assert "Video devices: 6 7" in result.stdout


def test_bring_up_gmsl_cameras_accepts_four_explicit_devices_in_0_to_7_range() -> None:
    result = _run_config_check(devices="0 1 6 7")

    assert result.returncode == 0
    assert "Video devices: 0 1 6 7" in result.stdout


def test_bring_up_gmsl_cameras_rejects_devices_outside_0_to_7_range() -> None:
    result = _run_config_check(devices="6 7 8")

    assert result.returncode == 2
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "must be an integer from 0 to 7" in combined_output


def test_bring_up_gmsl_cameras_rejects_empty_explicit_device_list() -> None:
    result = _run_config_check(devices="")

    assert result.returncode == 2
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "must include at least one video device id from 0 to 7" in combined_output
