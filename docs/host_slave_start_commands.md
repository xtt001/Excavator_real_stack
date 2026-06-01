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

## 录制最简现场流程

### 0. 按需标定当前姿态为 home

```bash
cd ~/Excavator_real_stack
./scripts/calibrate_home_pose_from_current.sh \
  --ssh-host slave-jetson \
  --ssh-user mundane
```

### 1. 从端启动全链路

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force
```

### 2. 从端另开终端看 CAN

```bash
candump -ta can2,18F021F6:1FFFFFFF
```

另开一个从端终端检查 `can3` 上 4 个 IMU 是否都在线：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/imu_can_probe.py --interface can3 --duration-s 3 --require-four
```

### 3. 主端 rqt 看图

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

### 4. 主端启动 sender

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
  --go-home-button "${GO_HOME_BUTTON}" \
  --confirm-remote-control
```

### 5. 点火和开始录制

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
```

### 6. 主端实时 QC

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-build-isolation --no-deps -e ./testbed
chmod +x scripts/watch_today_qc_from_slave.sh

DAY="$(date +%F)"
BASE="data/qc_today_${DAY}"
mkdir -p "${BASE}"

setsid bash -c "cd ~/Excavator_real_stack || exit 1; \
  echo \$\$ > '${BASE}/qc_watch.pid'; \
  exec ./scripts/watch_today_qc_from_slave.sh \
    --ssh-host slave-jetson \
    --ssh-user mundane \
    --remote-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
    --day '${DAY}' \
    --base-dir '${BASE}' \
    --history-from 13 \
    >> '${BASE}/qc_watch.log' 2>&1" \
  </dev/null >/dev/null 2>&1 &

tail -f "${BASE}/qc_watch.log"
```

停止今日在线 QC watcher：

```bash
cd ~/Excavator_real_stack
DAY="$(date +%F)"
kill "$(cat "data/qc_today_${DAY}/qc_watch.pid")"
```

## 测试现场最简流程

### 0. 确认 checkpoint bundle

```bash
cd /media/mundane/D/Excavator_real_stack
ls -lh policy_bundles/real_one_dig_v1/
```

### 1. 从端启动底层链路但不启动默认 receiver

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force --no-receiver
```

### 2. Shadow 测试

```bash
cd /media/mundane/D/Excavator_real_stack
export PYTHON="${PYTHON:-python3}"
"${PYTHON}" -m pip install --no-build-isolation --no-deps -e ./testbed

"${PYTHON}" -m testbed.cli.record_real \
  --config testbed/testbed/configs/policy_real_one_dig_v1.yaml \
  --data-side slave \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 \
  --bridge-port 8765 \
  --bridge-timeout 2.0 \
  --input policy \
  --no-record \
  --policy-output-mode shadow_zero \
  --num-episodes 1 \
  --max-steps 500 \
  --test-log-dir /media/mundane/EXTERNAL_USB/policy_control_tests \
  --live-action-line
```

### 3. 完整 control 测试

```bash
cd /media/mundane/D/Excavator_real_stack
export PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m testbed.cli.record_real \
  --config testbed/testbed/configs/policy_real_one_dig_v1.yaml \
  --data-side slave \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 \
  --bridge-port 8765 \
  --bridge-timeout 2.0 \
  --input policy \
  --no-record \
  --policy-output-mode control \
  --policy-action-scale 1.0 \
  --num-episodes 1 \
  --max-steps 4000 \
  --test-log-dir /media/mundane/EXTERNAL_USB/policy_control_tests \
  --live-action-line
```

### 4. 查看日志和切回录制

```bash
ls -td /media/mundane/EXTERNAL_USB/policy_control_tests/* | head -1
tail -n 5 /media/mundane/EXTERNAL_USB/policy_control_tests/*/steps.jsonl
cat /media/mundane/EXTERNAL_USB/policy_control_tests/*/summary.json
```

```text
Ctrl+C policy receiver
Ctrl+C slave_real_stack --no-receiver terminal
```

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

## 从端启动命令

推荐从端直接用统一脚本启动主链路：

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run
```

`run` 会按顺序启动 real CAN bridge、Orbbec、FPV 到 SHM、gateway 和
`tb-receiver-real --wait-for-record-start`，并在当前终端持续显示各服务日志。
按一次 `Ctrl+C` 时脚本会先让 receiver 下发零命令并保存当前片段，再停止 gateway、
FPV、Orbbec 和 bridge。不要连续按两次，除非必须立即放弃保存。

统一脚本默认使用 `EXCAVATOR_CONTROL_MODE=open_loop_motor_speed` 启动 bridge。现场不要在
未重新验证前切回 `closed_loop_velocity_scalar`：该模式在摇杆零输入时仍会用 IMU qvel
跑底层速度 PID，可能导致 boom 在无摇杆输入时自运动。

如果不想占住终端，也可以后台启动后按需看日志：

```bash
./scripts/slave_real_stack.sh start
./scripts/slave_real_stack.sh status
./scripts/slave_real_stack.sh tail receiver
./scripts/slave_real_stack.sh stop
```

兼容别名仍保留：`./scripts/slave_real_stack.sh tail recorder` 会转到 receiver 日志。

如果上次没有关干净导致端口占用：

```bash
./scripts/slave_real_stack.sh restart --force
```

脚本默认不启动 `candump`，CAN 观察仍然单独开终端执行：

```bash
candump -ta can2,18F021F6:1FFFFFFF
```

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

下面的手动命令用于排错或脚本不可用时逐项启动。

先确认 USB 移动硬盘已经挂载。当前现场盘 label 是 `EXTERNAL_USB`：

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS,MODEL
sudo mkdir -p /media/mundane/EXTERNAL_USB
findmnt /media/mundane/EXTERNAL_USB || \
  sudo mount -t ntfs3 -o uid=$(id -u),gid=$(id -g),umask=022 \
    /dev/disk/by-label/EXTERNAL_USB /media/mundane/EXTERNAL_USB || \
  sudo mount -t ntfs-3g -o uid=$(id -u),gid=$(id -g),umask=022 \
    /dev/disk/by-label/EXTERNAL_USB /media/mundane/EXTERNAL_USB
mkdir -p /media/mundane/EXTERNAL_USB/real_teleop_v1
test -w /media/mundane/EXTERNAL_USB/real_teleop_v1 && echo USB_WRITE_OK
```

不用统一脚本时，在从端先打开 4 个基础服务终端。每个终端先执行：

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
```

### 终端 1：real CAN bridge

```bash
control/setup/setup_can.sh can2 250000
control/setup/setup_can.sh can3 250000
ip -details link show can2
ip -details link show can3

./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8766 \
  --can-if can2 \
  --imu-if can3 \
  --can-bus-enabled true \
  --can-simulation false \
  --imu-simulation false \
  --create-mapping true \
  --control-mode open_loop_motor_speed \
  --pid-yaml control/config/joint_pid.yaml \
  --heartbeat-timeout-ms 800
```

bridge 启动日志应出现：

```text
loaded PID YAML: control/config/joint_pid.yaml
```

### 终端 2：Orbbec 相机

```bash
source ./scripts/ros2_fpv_env.sh
export EXCAVATOR_ORBBEC_WS=/home/mundane/orbbec_ws
export EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws
source ./scripts/source_ros_stack.sh
./scripts/start_orbbec_fpv_camera.sh
```

如果报 `orbbec_fpv_camera.launch.py` 找不到，说明
`/home/mundane/orbbec_ws/src/excavator_ros2_bridge` 可能还是旧链接。修复一次即可：

```bash
cd /home/mundane/orbbec_ws/src
mv excavator_ros2_bridge excavator_ros2_bridge.broken_$(date +%Y%m%d_%H%M%S)
ln -s /media/mundane/D/Excavator_real_stack/ros2_bridge/excavator_ros2_bridge excavator_ros2_bridge
cd /home/mundane/orbbec_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select excavator_ros2_bridge
```

### 终端 3：FPV compressed 到共享内存

```bash
source ./scripts/ros2_fpv_env.sh
export EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws
source ./scripts/source_excavator_ros_ws.sh
./scripts/start_fpv_subscriber_py.sh
```

### 终端 4：gateway

必须在终端 3 已创建 `/dev/shm/excavator_fpv_v1` 后再启动 gateway。

```bash
source ./scripts/excavator_deploy_network.sh
excavator_apply_slave_network_defaults
./scripts/start_bridge_gateway.sh --fpv-source auto --fpv-max-stale-ms 1000
```

`start_bridge_gateway.sh` 默认追加：

```bash
--fpv-encoding jpeg --fpv-jpeg-quality 95 --fpv-jpeg-cache-hz 30
```

因此 HDF5 中图像默认写到 `/observations/encoded_images/fpv`，格式为逐帧 JPEG。
gateway 只在 FPV SHM 帧序号变化时压缩一次，并缓存给 `read_state`，避免 receiver
高频读状态时重复压缩同一帧影响控制链路。
训练和 QC 会自动解码；`camera_fps` 仍由真实 image timestamp 估计，不等于
`record_hz` 或 control pump 的 50Hz。

## 点火前 CAN 确认和等待录制

主端 `tb-teleop-remote` 连接的是从端 `tb-receiver-real --input remote` 创建的
`8770` 端口。没有先启动从端 remote 接收端时，主端会报：

```text
ConnectionRefusedError: [Errno 111] Connection refused
```

### 从端新终端：只看 can2 的 18F021F6

另开一个从端终端执行：

```bash
candump -ta can2,18F021F6:1FFFFFFF
```

### 从端新终端：正式 receiver，先接收但等待开始录制

这个进程会立即打开 `8770` 并把主端摇杆 action 通过 control pump 直发到
`127.0.0.1:8766`，状态/FPV 仍从 gateway `127.0.0.1:8765` 读取；在收到
`record_start_requested` 前不会把 step 写入 episode 缓存。这样可以先点火、看 CAN，
确认正常后再按主端摇杆的录制开始键。

receiver live line 会显式显示健康门禁，例如：

```text
mode=armed health=OK err=- imu=1111 ...
mode=fault health=ERR err=imu_missing:1 imu=1011 ...
```

recording 过程中如果出现 IMU/FPV/bridge/remote/control 任一严格门禁错误，
receiver 会立即下发零命令并停止当前 record；已有 step 会保存到
`<dataset_dir>/failed/episode_<id>_failed_<timestamp>.hdf5`，不会写入主目录的
`episode_*.hdf5`，训练 loader 不会误读。
FAULT_HOLD 只把四维速度 action 钳成零；点火、remote mode、pilot 等 status toggle
仍会透传，方便现场先完成上电/先导流程，再等待传感器健康恢复后开始 record。

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e ./testbed
tb-receiver-real \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --data-side slave \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 \
  --bridge-port 8765 \
  --bridge-timeout 2.0 \
  --input remote \
  --num-episodes 1000000 \
  --max-steps 50000 \
  --output-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --session-id remote_teleop_slave_record \
  --wait-for-record-start \
  --live-action-line
```

如果主端仍然报 `ConnectionRefusedError`，先在从端确认 `8770` 是否已经监听：

```bash
ss -tlnp | grep ':8770'
```

## 主端 rqt 看图

主端 rqt 只用于看图，不参与 TCP 控制和 HDF5 录制主链路。主端新开一个终端执行：

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

如果主端没有 ROS2 或 rqt 起不来，可以跳过本节；录制 HDF5 不依赖主端 rqt。

## 主端摇杆控制命令

当前 `testbed/testbed/configs/teleop_real_v1.yaml` 使用双手柄映射：左侧物理摇杆是
pygame `joystick 0`，右侧物理摇杆是 pygame `joystick 1`；左侧控制 `swing/stick`，
右侧控制 `boom/bucket`。从端 `8770` 监听后，在主端启动 remote action sender：

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
  --go-home-button "${GO_HOME_BUTTON}" \
  --confirm-remote-control
```

交互式终端默认显示从端回传的 `receiver_mode/recording/go_home/saved` 状态；需要滚动日志时追加
`--no-monitor`。`--record-start-button 2` 和 `--record-start-joystick-id 0` 表示只接收左侧摇杆按钮
`2` 作为正式录制开始键，右侧按钮不参与录制控制。

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
| `4` | chassis_light 底盘灯 |
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
3. 必要时在从端停止 receiver 或 bridge：

   ```bash
   pkill -f "[e]xcavator_real_bridge" || true
   ```

4. 软件 estop 的手柄外壳按键尚未现场确认，本文档不写具体按键号。

录制和 go-home 行为：

- 按左侧按钮 `2` 前，receiver 处于 receive/armed 状态，会接收和下发控制，但不会写入 HDF5。
- 按左侧按钮 `2` 后开始正式 HDF5 录制；从端 `slave_real_stack.sh run --force` 终端按一次 `Ctrl+C`
  会先下发零命令，再停止当前 episode 并保存已有数据；主端 `teleop_remote` 的 `Ctrl+C` 只断开 sender。
- go-home 启用前需要在 `testbed/testbed/configs/teleop_real_v1.yaml` 中配置
  `teleop.recording.go_home.home_pose_rad` 并设为 `enabled: true`，也可以用
  `./scripts/calibrate_home_pose_from_current.sh` 从当前姿态采样写入。
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

手动停止顺序：

1. 先在主端停止 `${PYTHON} -m testbed.cli.teleop_remote`。
2. 再在从端停止 `tb-receiver-real`，确认日志出现 saved 或 failed 后再关终端。
3. 最后在从端停止 C++ real CAN bridge：

```bash
pkill -f "[e]xcavator_real_bridge" || true
```

如需重启真实控制，重新启动从端 4 个基础服务，再按“点火前 CAN 确认”或“正式录制”的顺序启动 remote 接收端和主端 sender。
