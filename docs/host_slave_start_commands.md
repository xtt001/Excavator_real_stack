# 主从端启动命令速查

本文作为当前现场主从分体测试和录制的主文档，集中记录应启动的命令、链路检查、
QC 和停止顺序。历史 runbook/checklist 已并入本页。

当前主端 SSH 配置使用：

```sshconfig
Host slave-jetson
  HostName 192.168.100.1
  User mundane
  ConnectTimeout 5
```

本文中的 SSH 命令优先使用 `slave-jetson`；非 SSH 的 TCP 控制链路使用
`192.168.100.1`。

当前现场以太网直连地址：

| 设备 | 接口 | 地址 | 用途 |
|------|------|------|------|
| 主端 | `enx9c69d36f2e8b` | `192.168.100.2/24` | SSH/TCP 客户端 |
| Jetson 从端 | `eno1` | `192.168.100.1/24` | SSH/TCP 服务端 |

主端验证直连链路：

```bash
ip route get 192.168.100.1
ping -c 3 192.168.100.1
ssh slave-jetson 'hostname; ip -br addr show dev eno1'
```

如果本机有线口丢失静态地址，主端重新启用连接：

```bash
nmcli connection up '有线连接 1'
```

从主端查看 Jetson 温度：

```bash
ssh slave-jetson 'timeout 2s tegrastats --interval 1000 || true'
```

## 现场最简流程（录制和 policy 共用）

### 0. 按需标定当前姿态为 home

```bash
cd ~/Excavator_real_stack
./scripts/calibrate_home_pose_from_current.sh \
  --ssh-host slave-jetson \
  --ssh-user mundane
```

### 1. 主端校准从端时间

在主端执行一次，把 Jetson 系统时间校准到主端时间。这个命令会通过 SSH
登录从端；如果提示 sudo 密码，输入 Jetson 的 `mundane` sudo 密码。

```bash
cd ~/Excavator_real_stack
./scripts/sync_slave_time_from_host.sh \
  --ssh-host slave-jetson \
  --ssh-user mundane
```

当前 Jetson 的公网/DNS/NTP 可能不可用，因此 `timedatectl` 服务 active
不等于已经同步成功。正式启动链路前以这条主端校时命令为准；校时完成后再到
Jetson 终端启动从端链路。从端启动脚本只负责启动链路，不再检查或修改系统时间。

### 1.5. 从端执行 GMSL 四路相机 bring-up

如果当前只需要相机 bring-up、抓帧、标定或 GMSL 诊断，不启动 real bridge/receiver，
可以在已经 SSH 登录的从端终端直接执行：

```bash
cd /media/mundane/D/Excavator_real_stack
GMSL_VIDEO_DEVICES="4 5 6 7" ./scripts/bring_up_gmsl_cameras.sh
```

脚本内部会按需调用 `sudo` 加载内核模块、配置 PWM 和 boost clock；如果当前 sudo
凭据已过期，终端会提示输入 Jetson 的 sudo 密码。

期望配置为四路 H190TA：

```text
/dev/video4  UYVY  1920x1536
/dev/video5  UYVY  1920x1536
/dev/video6  UYVY  1920x1536
/dev/video7  UYVY  1920x1536
```

### 1.6. Eye 外参标定手动操作

本节只标定 `eye_left` / `eye_right`，也就是 `video4` / `video5` 这一对。输出的相对外参为
`video5_T_video4`，字段名是 `right_T_left`，OpenCV 约定为：

```text
X_right = R * X_left + T
```

先做单帧 smoke，确认棋盘格能被两路同时识别。标定板需要同时完整出现在
`video4` 和 `video5` 中，建议占画面宽度约 15% 到 30%，采样瞬间保持静止。
下面两个脚本会把人工可读的准备、进度和成功提示输出到终端 `stderr`；`--json | tee ...`
保存的仍是纯 JSON。

```bash
cd /media/mundane/D/Excavator_real_stack

export RUN_DIR=artifacts/gmsl_extrinsics_eye_test_$(date +%Y%m%d_%H%M%S)
mkdir -p "${RUN_DIR}"

python3 tools/gmsl_camera_config/capture_gmsl_stereo_pairs.py \
  --left video4=/dev/video4 \
  --right video5=/dev/video5 \
  --output-dir "${RUN_DIR}/video4_video5" \
  --count 1 \
  --interval-s 0 \
  --image-format png \
  --warmup-frames 10 \
  --json | tee "${RUN_DIR}/capture_video4_video5.json"

python3 tools/gmsl_camera_config/calibrate_gmsl_stereo_pair.py \
  --intrinsics-manifest configs/camera_intrinsics/gmsl_h190ta/manifest.json \
  --left video4 \
  --right video5 \
  --pairs-json "${RUN_DIR}/video4_video5/pairs.json" \
  --output-json "${RUN_DIR}/video4_video5/stereo_calibration_smoke.json" \
  --annotated-dir "${RUN_DIR}/video4_video5/annotated" \
  --min-valid-pairs 1 \
  --json | tee "${RUN_DIR}/solve_video4_video5_smoke.json"
```

smoke 通过时应看到：

```text
status = ok
valid_pair_count = 1
left_found = true
right_found = true
```

如果 `valid_pair_count = 0`，先打开 `${RUN_DIR}/video4_video5/annotated/` 下的角点图看原因。
常见原因是棋盘格太远、太小、反光、模糊、被手遮挡，或没有被两路同时完整看到。

smoke 通过后，正式采 30 对。采集时移动棋盘格覆盖近/远、左/右、上/下和轻微倾斜姿态；
不要只在画面中心重复采样。

```bash
cd /media/mundane/D/Excavator_real_stack

export RUN_DIR=artifacts/gmsl_extrinsics_eye_$(date +%Y%m%d_%H%M%S)
mkdir -p "${RUN_DIR}"

python3 tools/gmsl_camera_config/capture_gmsl_stereo_pairs.py \
  --left video4=/dev/video4 \
  --right video5=/dev/video5 \
  --output-dir "${RUN_DIR}/video4_video5" \
  --count 30 \
  --interval-s 1.0 \
  --image-format png \
  --warmup-frames 10 \
  --json | tee "${RUN_DIR}/capture_video4_video5.json"

python3 tools/gmsl_camera_config/calibrate_gmsl_stereo_pair.py \
  --intrinsics-manifest configs/camera_intrinsics/gmsl_h190ta/manifest.json \
  --left video4 \
  --right video5 \
  --pairs-json "${RUN_DIR}/video4_video5/pairs.json" \
  --output-json "${RUN_DIR}/video4_video5/stereo_calibration.json" \
  --annotated-dir "${RUN_DIR}/video4_video5/annotated" \
  --min-valid-pairs 12 \
  --json | tee "${RUN_DIR}/solve_video4_video5.json"
```

正式结果重点检查：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
data = json.loads((run_dir / "video4_video5/stereo_calibration.json").read_text())
print("status:", data.get("status"))
print("valid_pair_count:", data.get("valid_pair_count"), "/", data.get("total_pair_count"))
print("rms_px:", data.get("rms_px"))
print("right_T_left:", json.dumps(data.get("right_T_left"), indent=2))
PY
```

通过线先按 `status=ok`、`valid_pair_count >= 12` 判断；`rms_px` 先以小于 `2 px`
作为现场可用线，最终是否固化还要复查 annotated 角点图和采样姿态覆盖。

### 2. 从端启动链路

正式录制或 policy control 时，启动全链路和唯一 receiver：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force --policy-remote
```

这个终端同时托管底层链路和唯一的 `policy_remote` receiver。按一次 `Ctrl+C`
会先停止 receiver，再停止 gateway、相机链路和 bridge。
`--policy-remote` 会自动使用 `policy_real_gmsl_four_camera_v1.yaml`、`policy_remote` input、
`control` output、`action_scale=1.0`、GMSL 四路相机、USB HDF5 目录和 policy test log 目录。

如果当前只是检查 IMU/qvel，不想启动 receiver，使用下面这个命令。它会启动
bridge、GMSL 预处理和 gateway，但不开 receiver：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force --no-receiver
```

如果只查 IMU/qvel 且不需要相机/FPV，可以进一步省掉相机链路：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force --no-receiver --no-camera
```

`--no-camera` 会让 gateway 进入无相机模式，`read_state` 只转发 joint/IMU 状态并返回空
`images`，不会再等待 GMSL/FPV 共享内存。上面两个 `--no-receiver` 模式下，IMU/qvel
日志脚本仍然连接默认 gateway `127.0.0.1:8765`；不要改成 `--port 8766`。

### 3. 从端另开终端看 CAN

`can2` 是主 CAN/control 状态检查通道；下面这条只用于看 `18F021F6` 等控制/状态帧，
不是 IMU 诊断入口：

```bash
candump -ta can2,18F021F6:1FFFFFFF
```

另开一个从端终端检查 `can3` 上 4 个 IMU 是否都在线：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/imu_can_probe.py --interface can3 --duration-s 3 --require-four
```

如果 IMU 掉线，优先抓 `can3` 的原始日志：

```bash
timeout 60s candump -ta can3,0:0,#FFFFFFFF > /tmp/can3_imu_$(date +%Y%m%d_%H%M%S).log
```

如果怀疑现场接线或脚本使用了 `can2`，先查 `can2` 状态：

```bash
ip -details -statistics link show can2
```

`can2` 进入 `BUS-OFF` / `state DOWN` 时，`candump can2` 可能为空；此时不能用
`can2` 的空日志判断 IMU 是否还在发帧。

### 4. 从端另开终端持续记录 IMU/qvel

这个日志脚本只读 `read_state`，不启动 receiver/sender，也不发送动作。现场全链路已经由
第 1 步启动时，脚本应连接默认 gateway `127.0.0.1:8765`，可以和 receiver 同时运行。
不要在全链路运行时加 `--port 8766` 直连 C++ bridge，因为 C++ bridge 当前只适合由
gateway/control pump 托管连接。

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/log_imu_qvel_quality.py \
  --rate-hz 50 \
  --duration-s 0 \
  --print-every-s 1 \
  --verbose-imu
```

`--duration-s 0` 表示一直记录到手动 `Ctrl+C`。记录文件优先写到
`/media/mundane/EXTERNAL_USB/imu_qvel_tests/`；终端会持续打印 qpos/qpos_deg、
bridge qvel、qpos 差分 qvel、raw IMU gyro 推导 qvel，以及每个 IMU 的
gyro/rpy/age/loss。

### 5. 主端 rqt 看图

```bash
cd ~/Excavator_real_stack
source ./scripts/excavator_deploy_network.sh
excavator_apply_host_network_defaults
source ./scripts/ros2_fpv_env.sh
./scripts/start_host_fpv_rqt.sh
```

rqt 里选择：

```text
/camera/color/image_raw
```

### 6. 主端启动 sender

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
export GO_HOME_BUTTON=3
"${PYTHON}" -m pip install --no-build-isolation --no-deps -e ./testbed
"${PYTHON}" -m testbed.cli.teleop_remote \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --host 192.168.100.1 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --record-start-button 2 \
  --record-start-joystick-id 0 \
  --policy-start-button 4 \
  --go-home-button "${GO_HOME_BUTTON}" \
  --confirm-remote-control
```

### 7. 点火、go-home、录制和模型控制

```text
5 -> remote mode
1 -> ignition
6 -> pilot
```

从端 `candump` 里 `18F021F6` 前两字节应看到：

```text
00 00 -> 00 01 -> 01 01 -> 01 05
```

```text
button 2 -> start HDF5 recording
button 3 -> go-home
button 4 -> manual/model control toggle
```

推荐顺序：先手动摆位并点火；每次进入模型控制前先按 `3` 跑 go-home，等 receiver
显示 go-home done；然后按 `4` 切到 model control。需要回到人操作时再按一次 `4`。
需要写 HDF5 时按 `2`；不需要录制就不要按 `2`。

### 8. 主端实时 QC

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-build-isolation --no-deps -e ./testbed
chmod +x scripts/watch_today_qc_from_slave.sh

DAY="$(date +%F)"
BASE="data/qc_today_${DAY}"
HISTORY_FROM=27  # 2026-06-09: episode_26 是昨天最后一条，今天从 episode_27 开始。
mkdir -p "${BASE}"

./scripts/watch_today_qc_from_slave.sh \
  --ssh-host slave-jetson \
  --ssh-user mundane \
  --remote-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --day "${DAY}" \
  --base-dir "${BASE}" \
  --history-from "${HISTORY_FROM}" \
  --log-file "${BASE}/qc_watch.log"
```

按 `Ctrl+C` 会停止 watcher，不需要再手动 `kill`。脚本每轮扫描前会检查主从端时间差，
超过 5 秒会通过 SSH 提示输入从端 sudo 密码并自动校准时间。如果校时失败，
日志会提示 `remote time sync failed`，并降级为扫描远端全部 `episode_*.hdf5`，
只拉取编号不小于 `--history-from` 的 episode。长期仍建议修复从端 NTP，避免 HDF5
文件时间继续写错。

如需免输入密码，也可以在从端 Jetson 上一次性配置：

```bash
sudo visudo -f /etc/sudoers.d/excavator-qc-time-sync
```

填入：

```text
mundane ALL=(root) NOPASSWD: /usr/bin/timedatectl, /usr/bin/date, /usr/sbin/hwclock
```

## 附录：policy shadow 检查命令（不属于现场流程）

E52 分阶段真机实验的任务顺序、通过条件、全局中止条件和当前部署能力缺口见：

```text
docs/e52_real_machine_experiment_plan_20260710.md
docs/e52_real_machine_experiment_tasks_20260710.csv
```

E52 检入配置保持 `shadow_zero`。需要人工看护的策略动作 trace 时，使用
`scripts/run_e52_policy_control_trace.sh` 做 bundle preflight、request-local
control override、终止零命令记录和事后 trace 汇总；不要修改 YAML 默认值或直接
调用通用 `--policy-output-mode control`。

本节只用于在进入现场最简流程前检查模型 bundle 和 shadow 输出。shadow 阶段
policy action 只写入 `steps.jsonl`，下发给底层的动作保持零。正式录制和
policy control 都使用上面的唯一 `policy_remote` receiver 流程。

从端启动底层链路用于 shadow：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force --no-receiver
```

这里的 `--no-receiver` 只是不启动脚本默认的 receiver，避免占用
`8770` 或启动错误模式；底层 real CAN bridge、相机、FPV、gateway 仍会启动。
这个命令用于 shadow 检查时的底层链路；在该终端按一次 `Ctrl+C` 会停止底层链路。
正式 policy/录制流程不要用本命令，直接用后面的唯一 receiver 一体化命令。

检查 bundle 并跑 shadow：

```bash
cd /media/mundane/D/Excavator_real_stack
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/testbed"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cu12/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

./scripts/run_policy_shadow_check.sh
```

脚本会先确认 `policy_bundles/real_gmsl_four_camera_v1/` 下的 `policy_best.ckpt`、
`dataset_stats.pkl`、`resolved_config.yaml` 是否存在，并检查 bundle 的
`camera_names` 是 `video4,video5,video6,video7`，然后跑 `shadow_zero`，
最后打印方便现场判断的关键数据：

- `Bundle verdict`
- `Verdict`
- `steps / stop_reason / effective_hz`
- `policy latency mean/p50/p95/max`
- `policy_action mean/max_abs`
- `returned/raw/safe/commanded max_abs`
- `deadzone assist enabled_steps/active_steps/active_pct/axes`
- `policy_errors / health_bad / ack_bad / fault_codes`

看到 `Bundle verdict: OK` 且 `Verdict: OK` 后，再继续下一步。如果是 `NOT OK`，
不要进入 policy control。

## 运行分工

| 端 | 负责内容 | 不应启动 |
|----|----------|----------|
| 从端 `slave-jetson` / `192.168.100.1` | real CAN bridge、Orbbec、FPV 到 SHM、gateway、`tb-receiver-real` 本地写 USB | 主端手柄读取 |
| 主端 | `tb-teleop-remote` 摇杆控制、QC | C++ real CAN bridge、训练 HDF5 写盘 |

关键端口：

- 从端 receiver 连接本机 gateway：`127.0.0.1:8765`
- 从端 control pump 直连本机 C++ bridge：`127.0.0.1:8766`
- 从端 gateway 也连接本机 C++ bridge：`127.0.0.1:8766`，用于 `read_state` 和 FPV
- 主端 remote action 连接从端 receiver：`192.168.100.1:8770`

## 补充说明

正式现场只使用上面的“现场最简流程”。不要再分开启动底层链路和 receiver；
`./scripts/slave_real_stack.sh run --force --policy-remote` 会同时托管底层链路和唯一
`policy_remote` receiver，并用当前终端的 `Ctrl+C` 统一停止。

如果上次没有关干净导致端口占用，重新执行现场最简流程里的
`./scripts/slave_real_stack.sh run --force --policy-remote` 即可；`--force` 会先清理旧服务再启动。

统一脚本使用 `EXCAVATOR_CONTROL_MODE=open_loop_motor_speed` 启动 bridge。现场不要在
未重新验证前切回 `closed_loop_velocity_scalar`：该模式在摇杆零输入时仍会用 IMU qvel
跑底层速度 PID，可能导致 boom 在无摇杆输入时自运动。

IMU 四个传感器地址检查。这个检查是只读的，只监听 `can3` 上的 IMU 高速帧，
不会向机器下发控制命令。

```bash
./scripts/imu_can_probe.py --interface can3 --duration-s 3 --require-four
```

从主端一键检查从端 Jetson：

```bash
ssh slave-jetson \
  'cd /media/mundane/D/Excavator_real_stack && \
   ./scripts/imu_can_probe.py --interface can3 --duration-s 3 --require-four'
```

正常时应同时看到 raw addr `0,1,2,3`，并且命令退出码为 `0`：

```json
"up": true,
"raw_addr_counts": {
  "0": 900,
  "1": 900,
  "2": 900,
  "3": 900
},
"missing_raw_addr_0_to_3": []
```

如果输出里的 `missing_raw_addr_0_to_3` 非空，或命令退出码非 `0`，说明至少一个
IMU 地址没有在 `can3` 上持续发帧。先处理 IMU 地址/协议/CAN 接线/供电问题，再录训练数据。
`captured_frames` 表示监听窗口内总 CAN 帧数，`imu_highspeed_ch1_frames` 表示其中被识别为
IMU 高速 ch1 协议的帧数，`cmd_counts_by_raw_addr` 可用于看每个 IMU 地址的各类分包是否齐全。

IMU/qvel 只读日志。用于检查上电后陀螺仪原始值、bridge 返回的 qvel，以及 qpos
差分得到的 qvel 是否一致。这个脚本不启动 receiver/sender，也不发送动作；正式现场
全链路运行时默认连接从端本机 gateway `127.0.0.1:8765`。如果从端是
`--no-receiver --no-camera` 启动，仍然使用默认 `8765`，gateway 会跳过图像字段。
记录文件优先写到 `/media/mundane/EXTERNAL_USB/imu_qvel_tests/`。`--duration-s 0`
表示一直记录到手动 `Ctrl+C`。

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/log_imu_qvel_quality.py --rate-hz 50 --duration-s 0 --print-every-s 1 --verbose-imu
```

只有在没有启动 gateway/receiver、临时只启动 C++ bridge 做只读检查时，才直连
C++ bridge `8766`：

```bash
./scripts/log_imu_qvel_quality.py --port 8766 --rate-hz 50 --duration-s 0 --print-every-s 1 --verbose-imu
```

脚本会打印：

- `qpos`：当前 `read_state` 返回的 4 轴姿态，单位 rad。
- `qpos_deg`：同一组姿态转成 degree，方便肉眼判断角度变化。
- `qpos raw_imu_deg`：由每个 IMU 当前 `rpy_raw_deg` 直接反算的关节角，
  代表 IMU 协议原始角度，不做 `[-180, 180]` 折叠，也不做连续保护。
- `qpos policy_deg`：当前 `read_state` 返回给 receiver/policy 的 qpos。
- `qpos policy_rad`：同一组 policy qpos 的弧度值；receiver、policy 和 HDF5
  实际使用的是 rad。
- `qpos policy-raw_deg`：`policy_deg - raw_imu_deg` 的直接差值，不做分支折叠；
  如果同一物理角被表示成 `-137` 和 `222`，这里会显示约 `-360`。
- `qpos physical_delta`：`policy - raw_imu` 的最短角差，用于看连续保护改了多少。
- `qpos_folded_imu_deg`：JSONL 中额外记录；由 `rpy_rad` 反算，代表单轴折叠后、
  policy 连续保护前的姿态。
- `qvel policy_rad_s`：当前 `read_state` 返回给 receiver/policy 的 qvel。
- `qvel diff_rad_s`：由相邻 policy qpos 差分得到的 qvel。
- `qvel raw_gyro_rad_s`：由 IMU 原始 gyro 重新计算的关节 qvel，未做启动 bias 扣除。
- `qvel resid_rad_s`：`policy - qpos_diff`，静止时应接近 0。
- `rpy_raw_deg`：`--verbose-imu` 下每个 IMU 原始欧拉角，单位 degree，不做
  `[-180, 180]` 折叠；`rpy_rad` 仍是旧的折叠后弧度值。

如果输出提示 `imu_debug missing`，说明当前运行的 bridge 还是旧二进制，只能看到
`imu_health`，看不到每个 IMU 的 gyro/rpy/accel 原始值。需要重新编译并重启 bridge。

GMSL HDF5 中图像默认按相机名写到 `/observations/encoded_images/video4`、
`/observations/encoded_images/video5`、`/observations/encoded_images/video6`、
`/observations/encoded_images/video7`，格式为逐帧 JPEG。
gateway 只在相机 SHM 帧序号变化时压缩一次，并缓存给 `read_state`，避免 receiver
高频读状态时重复压缩同一帧影响控制链路。
训练和 QC 会自动解码；`camera_fps` 仍由真实 image timestamp 估计，不等于
`record_hz` 或 control pump 的 50Hz。

## 按键和动作映射

当前 `testbed/testbed/configs/teleop_real_v1.yaml` 使用双手柄映射：左侧物理摇杆是
pygame `joystick 0`，右侧物理摇杆是 pygame `joystick 1`；左侧控制 `swing/stick`，
右侧控制 `boom/bucket`。主端 sender 的启动命令只保留在“现场最简流程”里。

| 软件动作 | action index | pygame 设备/轴 | 当前 action 正方向 |
|----------|--------------|----------------|--------------------|
| swing 回转 | `action[0]` | `joystick 0 axis 1` + invert | 左侧物理摇杆前推为正，后拉为负 |
| boom 大臂 | `action[1]` | `joystick 1 axis 1` | 右侧物理摇杆后拉为正，前推为负 |
| stick 小臂 | `action[2]` | `joystick 0 axis 0` + invert | 左侧物理摇杆左推为正，右推为负 |
| bucket 铲斗 | `action[3]` | `joystick 1 axis 0` | 右侧物理摇杆右推为正，左推为负 |
| 状态按钮 | - | `joystick 0` | 左侧物理摇杆按钮发送 remote mode、ignition、pilot 等状态位 |

按当前配置 `axis_map: [1, 1, 0, 0]`、`invert: [true, false, true, false]`，
物理输入、pygame 原始轴、软件 action 和现场动作语义的关系如下：

| 物理摇杆 | 操作方向 | pygame 原始值 | invert 后 action | 实际挖机动作 |
|----------|----------|---------------|------------------|--------------|
| 左侧 `joystick 0 axis 1` | 前推 | `axis 1 < 0` | `action[0] > 0` | swing 向左回转 |
| 左侧 `joystick 0 axis 1` | 后拉 | `axis 1 > 0` | `action[0] < 0` | swing 向右回转 |
| 右侧 `joystick 1 axis 1` | 前推 | `axis 1 < 0` | `action[1] < 0` | boom 下压/下降 |
| 右侧 `joystick 1 axis 1` | 后拉 | `axis 1 > 0` | `action[1] > 0` | boom 上抬/上升 |
| 左侧 `joystick 0 axis 0` | 左推 | `axis 0 < 0` | `action[2] > 0` | stick 向上 |
| 左侧 `joystick 0 axis 0` | 右推 | `axis 0 > 0` | `action[2] < 0` | stick 向下 |
| 右侧 `joystick 1 axis 0` | 右推 | `axis 0 > 0` | `action[3] > 0` | bucket 正向；收斗/开斗需现场最后确认 |
| 右侧 `joystick 1 axis 0` | 左推 | `axis 0 < 0` | `action[3] < 0` | bucket 负向；收斗/开斗需现场最后确认 |

`joystick 0/1` 是 pygame 启动日志里的设备编号；`axis 0` 是左右 X 轴，`axis 1` 是前后 Y 轴。
本文档只把已确认的实际方向写死；bucket 的收斗/开斗语义确认后，需要同步更新这里和
`testbed/testbed/configs/teleop_real_v1.yaml` 的注释。

左侧真实按键按实体从左到右数 `1..10`：

| 左侧真实按键 | 功能 |
|--------------|------|
| `1` | ignition 点火 |
| `2` | 开始正式录制 HDF5 |
| `3` | go-home；启用后不再发送 crush 破碎 |
| `4` | policy_start；在 `policy_remote` receiver 中切换 manual/model |
| `5` | remote_mode 遥控模式 |
| `6` | pilot 先导使能 |
| `7` | high_speed 高速 |
| `8` | chassis_dozer_mode 推土铲模式 |
| `9` | horn 喇叭 |
| `10` | motor_gear 电机档位，每按一次循环加一档 |

主端启动控制后，依次按左侧按钮 `5`、`1`、`6`，检查 `candump` 中 `18F021F6` 前两字节：

```text
00 00 -> 00 01 -> 01 01 -> 01 05
```

`01 05` 表示 remote mode、ignition、pilot 已进入保持状态；真正状态以 `candump` 为准。如果没有变化，
先停主端 sender，再检查从端 `8770`、gateway、bridge 和 `can2`。

紧急停止优先级：

1. 机器实体急停、人工接管或现场电源/液压安全手段永远优先。
2. 主端松开摇杆会发送零 action；按 `Ctrl+C` 停止 `teleop_remote` 时 sender 会尽量发送零 action，
   从端 receiver 会等待主端重连。
3. 必要时在从端 `slave_real_stack.sh run --force --policy-remote` 终端按一次 `Ctrl+C`，统一停止 receiver 和底层链路。
4. 软件 estop 的手柄外壳按键尚未现场确认，本文档不写具体按键号。

录制和 go-home 行为：

- 按左侧按钮 `2` 前，receiver 处于 receive/armed 状态，会接收和下发控制，但不会写入 HDF5。
- 按左侧按钮 `2` 后开始正式 HDF5 录制；从端 `slave_real_stack.sh run --force --policy-remote` 终端按一次 `Ctrl+C`
  会先下发零命令，再停止当前 episode 并保存已有数据；主端 `teleop_remote` 的 `Ctrl+C` 只断开 sender。
- go-home 启用前需要在 `testbed/testbed/configs/policy_real_gmsl_four_camera_v1.yaml` 中配置
  `teleop.recording.go_home.home_pose_rad` 并设为 `enabled: true`，也可以用
  `./scripts/calibrate_home_pose_from_current.sh` 从当前姿态采样写入。
- `swing` 是圆周角，`216°` 和 `-144°` 是同一物理分支附近。go-home、phase label
  和 go-home 区域采样必须使用最短角误差；不要用 `home_pose_rad - qpos` 的普通差值
  判断 swing 距离或控制方向。bridge 输出也应优先保持 IMU4 raw yaw 的非负分支。
- receive/armed 状态按按钮 `3` 只做 go-home，不保存 HDF5，完成或失败后回到 receive/armed。
- recording 状态按按钮 `3` 会结束 episode；成功保存到成功目录，失败保存到 `failed/` 并写入失败原因。
- 现场统一脚本默认 `EXCAVATOR_NUM_EPISODES=1000000`，要限制 episode 数量可在启动前显式设置该环境变量。
- 当前 record 会先把本 episode 缓存在内存里，episode 结束时再写 HDF5；长时间采集建议分多段录制，
  每段结束后先跑 QC。

维护 go-home 参数时，不要直接凭感觉调大速度。先用最近 HDF5 判断方向：

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
./scripts/analyze_go_home_direction.py
```

输出 `recommendation=flip` 表示该轴在 go-home 段主动下发时 `|error|` 平均增大，需要检查或翻转
`control_signs` 对应轴。

需要标定液压死区和低速响应时，每次只测一个轴：

```bash
cd /media/mundane/D/Excavator_real_stack
python3 scripts/calibrate_axis_response.py \
  --host 127.0.0.1 \
  --port 8766 \
  --axis boom \
  --direction both \
  --amplitudes 0.03,0.05,0.07,0.10,0.12 \
  --duration-s 0.45 \
  --settle-s 0.80 \
  --abort-delta-rad 0.05 \
  --confirm-hardware-motion
```

确认 boom 安全后，再分别测 `stick`、`bucket`。用刚能稳定产生正确方向 qpos/qvel 的最小幅度作为
`min_action` 候选，用现场认为运动仍足够慢的幅度作为 `max_action` 上限候选。

如果要把 dig/dump 区域上方纳入 go-home 允许范围，先人工移动到边界姿态并只读采样：

```bash
cd ~/Excavator_real_stack
./scripts/record_go_home_region_sample.sh --label dig_above --note "bucket above dig area left edge"
./scripts/record_go_home_region_sample.sh --label dump_above --note "bucket above dump area"
./scripts/record_go_home_region_sample.sh --label unsafe_too_far --note "do not allow go-home here"
```

采样脚本只读 qpos/qvel，不会下发动作。

## 录完做 QC

正式长时间录制建议优先使用上文“主端实时 QC”的今日在线 QC watcher。watcher 只通过 SSH
列目录和拉取已完成的 `episode_*.hdf5`，QC 计算在主端本地缓存上执行，不占用从端 Jetson
的 Python/QC 资源，也不依赖 sshfs 随机读 HDF5。

录制期间或录完后查看线上 QC 历史：

```bash
cd ~/Excavator_real_stack
DAY="$(date +%F)"
tail -f "data/qc_today_${DAY}/qc_watch.log"
```

如果只是录制结束后的整包检查，也可以把 USB 拔回主端再跑完整 dataset QC：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-build-isolation --no-deps -e ./testbed
MPLCONFIGDIR=/tmp/excavator_mpl "${PYTHON}" -m testbed.cli.dataset_qc \
  --dataset-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --output-dir /tmp/excavator_dataset_qc_$(date +%Y%m%d_%H%M%S) \
  --profile real
```

离线 phase label 在主端对已拉取或已挂载的数据集运行，不改写原始 HDF5：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m testbed.cli.label_phases \
  --dataset-dir data/qc_cache \
  --phase-config testbed/testbed/configs/teleop_real_v1.yaml \
  --output-dir data/qc_cache/phase_labels
```

## 停止顺序

如果从端是用 `./scripts/slave_real_stack.sh run` 启动的，直接在该终端按
一次 `Ctrl+C`，脚本会先让 receiver 下发零命令并保存当前片段，再按顺序清理。
保存大文件到 USB 可能需要几十秒，不要连续按两次 `Ctrl+C`。

如果从端是用 `start` 后台启动的：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh stop
```

如果端口仍被旧进程占用：

```bash
./scripts/slave_real_stack.sh stop --force
```

常规现场不要手动 `pkill`；只有脚本失效、`stop --force` 也清不掉端口时再排查残留进程。

如需重启真实控制，重新执行上面的“现场最简流程”。
