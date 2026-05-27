# 主从端启动命令速查

本文作为当前现场主从分体测试和录制的主文档，集中记录应启动的命令、链路检查、
QC 和停止顺序。历史 runbook/checklist 已并入本页。

## 最简现场流程

### 1. 从端启动全链路

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force
```

该命令会启动 bridge、Orbbec、FPV SHM、gateway 和 receiver，并在当前终端持续显示日志。
receiver 会立即接收主端 action、读取传感器并下发控制；只有按 record start 后才开始写 HDF5。
按一次 `Ctrl+C` 会先让 receiver 下发零命令并保存当前失败/中断片段，然后停止 gateway、FPV、
Orbbec 和 bridge。不要连续按两次，除非必须立即放弃保存。
默认 FPV 录制使用逐帧 JPEG Q95 写入 HDF5，训练读取时会自动解码成 RGB tensor。

### 2. 从端另开终端看 CAN

```bash
candump -ta can2,18F021F6:1FFFFFFF
```

### 3. 主端 rqt 看图

主端 rqt 只用于看图，不参与 TCP 控制和 HDF5 录制主链路。主端另开终端执行：

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

如果主端没有 ROS2 或 rqt 起不来，可以跳过本步；录制 HDF5 不依赖主端 rqt。

### 4. 主端启动 sender

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
export GO_HOME_BUTTON=3
"${PYTHON}" -m pip install --no-deps -e ./testbed
"${PYTHON}" -m testbed.cli.teleop_remote \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --host 192.168.31.170 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --record-start-button 2 \
  --record-start-joystick-id 0 \
  --go-home-button "${GO_HOME_BUTTON}" \
  --confirm-remote-control
```

`GO_HOME_BUTTON=3` 表示左侧物理摇杆从左到右数第 `3` 个按钮，和
`--record-start-button 2` 一样使用真实外壳编号，不是 pygame 的零基 index。
按钮 `3` 现在用于 go-home，不再发送原来的 crush 状态位。

### 5. 点火和开始录制

按 L 手柄按钮：

```text
5 -> remote mode
1 -> ignition
6 -> pilot
```

从端 `candump` 里 `18F021F6` 前两字节应看到：

```text
00 00 -> 00 01 -> 01 01 -> 01 05
```

确认可以点火/控制后，按左侧手柄按钮 `2` 开始正式录制。按 `2` 之前，receiver 只接收并下发控制，
不会写 HDF5 step。启用 go-home 后，episode 结束方式改为：人工先开到 home 附近，再按
go-home 按钮；go-home 成功时 receiver 自动保存成功 episode，失败时保存到 `failed/`。

### 6. 主端实时 QC

正式录制时建议另开主端终端运行 SSH watcher。它只从从端拉取已经原子提交的
`episode_*.hdf5` 到主端缓存，并在主端本地跑 QC，不在 Jetson 上跑 QC：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
MPLCONFIGDIR=/tmp/excavator_mpl "${PYTHON}" -m testbed.cli.dataset_qc_watch_ssh \
  --ssh-host 192.168.31.170 \
  --ssh-user mundane \
  --remote-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --cache-dir data/qc_cache \
  --output-dir data/qc_live
```

## 运行分工

| 端 | 负责内容 | 不应启动 |
|----|----------|----------|
| 从端 `192.168.31.170` | real CAN bridge、Orbbec、FPV 到 SHM、gateway、`tb-receiver-real` 本地写 USB | 主端手柄读取 |
| 主端 | `tb-teleop-remote` 摇杆控制、QC | C++ real CAN bridge、训练 HDF5 写盘 |

关键端口：

- 从端 receiver 连接本机 gateway：`127.0.0.1:8765`
- 从端 control pump 直连本机 C++ bridge：`127.0.0.1:8766`
- 从端 gateway 也连接本机 C++ bridge：`127.0.0.1:8766`，用于 `read_state` 和 FPV
- 主端 remote action 连接从端 receiver：`192.168.31.170:8770`

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
ssh mundane@192.168.31.170 \
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
python -m pip install --no-deps -e ./testbed
tb-receiver-real \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --data-side slave \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 \
  --bridge-port 8765 \
  --bridge-timeout 2.0 \
  --input remote \
  --num-episodes 1 \
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

当前 `testbed/testbed/configs/teleop_real_v1.yaml` 使用双手柄映射。
现场只读测试确认：先操作的右侧物理摇杆对应 pygame `joystick 1`，
后操作的左侧物理摇杆对应 pygame `joystick 0`。

| 关节 | pygame 设备/轴 | 手柄动作 | 说明 |
|------|----------------|----------|------|
| swing 回转 | `joystick 1 axis 0` | 1 号手柄左右推 | `axis 0` 是左右轴 |
| boom 大臂 | `joystick 0 axis 1` | 0 号手柄前后推 | `axis 1` 是前后轴 |
| stick 小臂 | `joystick 1 axis 1` | 1 号手柄前后推 | `axis 1` 是前后轴 |
| bucket 铲斗 | `joystick 0 axis 0` | 0 号手柄左右推 | `axis 0` 是左右轴 |
| 状态按钮 | `joystick 0` | 0 号手柄按钮 | remote mode、ignition、pilot 等状态位 |

按当前配置 `invert: [false, false, false, false]`，左右摇杆动作对应的软件控制语义如下：

| 物理摇杆 | 操作方向 | 控制关节 | 当前软件输入 |
|----------|----------|-------------|--------------|
| 右侧摇杆 | 左推 | 回转 `swing` | `joystick 1 axis 0` 左方向 |
| 右侧摇杆 | 右推 | 回转 `swing` | `joystick 1 axis 0` 右方向 |
| 右侧摇杆 | 前推/上推 | 小臂 `stick` | `joystick 1 axis 1` 上方向 |
| 右侧摇杆 | 后拉/下拉 | 小臂 `stick` | `joystick 1 axis 1` 下方向 |
| 左侧摇杆 | 左推 | 铲斗 `bucket` | `joystick 0 axis 0` 左方向 |
| 左侧摇杆 | 右推 | 铲斗 `bucket` | `joystick 0 axis 0` 右方向 |
| 左侧摇杆 | 前推/上推 | 大臂 `boom` | `joystick 0 axis 1` 上方向 |
| 左侧摇杆 | 后拉/下拉 | 大臂 `boom` | `joystick 0 axis 1` 下方向 |

`joystick 0/1` 是 pygame 启动日志里的设备编号，例如：

```text
Joystick ready: [0 -> 0] ...
Joystick ready: [1 -> 1] ...
```

上表只说明“摇杆往哪个方向推，会控制哪个关节”。真机实际运动方向，例如回转向左还是向右、
大臂上升还是下降、小臂伸出还是收回、铲斗收斗还是放斗，还需要第一次真机动作时做小幅单轴验证；
如果某个关节方向反了，再改配置里的 `invert`。

从端 `8770` 已经监听后，在主端启动 remote action sender。这里不用额外写两个
`--joystick-id`，`--input joystick` 会按配置文件读取两个手柄。
`--record-start-button 2` 按真实外壳按钮编号理解，表示左侧真实按钮 `2`。
当前目标映射是所有按键功能都放在左侧物理摇杆：左侧按钮 `2` 开始正式录制。
配置里 `button_joystick_ids: [0]`，启动命令里 `--record-start-joystick-id 0`，
都明确只从左侧物理摇杆 `joystick 0` 接收录制开始键；
右侧按钮不参与控制。

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
export GO_HOME_BUTTON=3
"${PYTHON}" -m pip install --no-deps -e ./testbed
"${PYTHON}" -m testbed.cli.teleop_remote \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --host 192.168.31.170 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --record-start-button 2 \
  --record-start-joystick-id 0 \
  --go-home-button "${GO_HOME_BUTTON}" \
  --confirm-remote-control
```

主端启动控制后，依次按下左侧摇杆从左到右编号的 `5`、`1`、`6`，观察 `candump` 里
`18F021F6` 前两字节是否变化：

```text
00 00 -> 00 01 -> 01 01 -> 01 05
```

看到这个变化后，说明 remote mode、ignition、pilot 这些状态位已经进入
`18F021F6`。如果没有变化，先停主端 sender，再检查从端 `8770`、gateway、
bridge 和 `can2`。

当前只记录真实左侧摇杆按键和功能。按钮按实体从左到右数 `1..10`：

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

`01 05` 表示 remote mode、ignition 和 pilot 当前都已进入保持状态。
真正是否进入对应状态以 `candump` 看到的 `18F021F6` 字节为准。

紧急停止优先级：

1. 第一优先级永远是机器实体急停、人工接管或现场电源/液压安全手段，不要只依赖软件按钮。
2. 主端松开摇杆会发送零 action；按 `Ctrl+C` 停止 `teleop_remote` 时 sender 会尽量发送零 action/quit。
3. 从端可停止 receiver 或 bridge；必要时执行：

   ```bash
   pkill -f "[e]xcavator_real_bridge" || true
   ```

4. 软件 estop 的手柄外壳按键尚未现场确认，本文档不写具体按键号。确认前不要把它当作
   唯一急停手段。

确认点火/CAN 正常后，按左侧手柄按钮 `2` 开始正式录制。按键 `2` 之前的控制
只会用于点火和 CAN 确认，不会写入 HDF5。录制过程中按一次 `Ctrl+C`，程序会
先下发零命令，再停止当前 episode 并保存已有数据。

go-home 第一版只用于 home pose 附近精调。启用前必须先在
`testbed/testbed/configs/teleop_real_v1.yaml` 中填好
`teleop.recording.go_home.home_pose_rad`，并把 `enabled` 改成 `true`；
主端 sender 需要额外传 `--go-home-button 3`。go-home
成功后 receiver 会自动保存成功 episode；go-home 失败会把当前片段保存到
`failed/` 并写入失败原因。

说明：当前 record 会先把本 episode 缓存在内存里，episode 结束时再写 HDF5；
`--max-steps 50000` 是“足够大、靠手动结束”的现场用法，不建议单个 episode 连续录很久。
长时间采集建议分多段录制，每段结束后先跑 QC。

## 录完做 QC

正式长时间录制建议优先用主端 SSH watcher 做实时 QC。watcher 只通过 SSH 列目录和拉取
已完成的 `episode_*.hdf5`，QC 计算在主端本地缓存上执行，不占用从端 Jetson 的 Python/QC
资源，也不依赖 sshfs 随机读 HDF5。

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
MPLCONFIGDIR=/tmp/excavator_mpl "${PYTHON}" -m testbed.cli.dataset_qc_watch_ssh \
  --ssh-host 192.168.31.170 \
  --ssh-user mundane \
  --remote-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --cache-dir data/qc_cache \
  --output-dir data/qc_live
```

如果只是录制结束后的整包检查，也可以把 USB 拔回主端再跑完整 dataset QC：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
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
