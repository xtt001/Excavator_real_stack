# 主从端启动命令速查

本文只列现场主从分体运行时应启动的命令。完整检查流程、链路验证和故障处理见
[`realworld_host_slave_runbook.md`](realworld_host_slave_runbook.md)。

## 运行分工

| 端 | 负责内容 | 不应启动 |
|----|----------|----------|
| 从端 `192.168.31.170` | real CAN bridge、Orbbec、FPV 到 SHM、gateway、`tb-record-real` 本地写 USB | 主端手柄读取 |
| 主端 | 手柄探测、`tb-teleop-remote`、QC、可选 rqt 看图 | C++ real CAN bridge、训练 HDF5 写盘 |

关键端口：

- 从端 recorder 连接本机 gateway：`127.0.0.1:8765`
- 从端 gateway 连接本机 C++ bridge：`127.0.0.1:8766`
- 主端 remote action 连接从端 recorder：`192.168.31.170:8770`

## 从端启动命令

先确认 USB 移动硬盘已经挂载。当前现场盘 label 是 `EXTERNAL_USB`：

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS,MODEL
sudo mkdir -p /media/mundane/EXTERNAL_USB
findmnt /media/mundane/EXTERNAL_USB || \
  sudo mount /dev/disk/by-label/EXTERNAL_USB /media/mundane/EXTERNAL_USB
mkdir -p /media/mundane/EXTERNAL_USB/real_teleop_v1
test -w /media/mundane/EXTERNAL_USB/real_teleop_v1 && echo USB_WRITE_OK
```

在从端打开 5 个终端。每个终端先执行：

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
  --heartbeat-timeout-ms 800
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

### 终端 5：从端本地录制 HDF5，等待主端 action

此命令在从端本地读 gateway/FPV SHM，并把 HDF5 写到从端 USB。主端只发送
手柄 action，不再每步拉 raw RGB 图像写盘。

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
python -m pip install --no-deps -e ./testbed
python -m testbed.cli.record_real \
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
  --live-action-line
```

### 新终端：只看 can2 的 18F021F6

如果需要单独观察从端 `can2` 上的 `18F021F6` 报文，另开一个从端终端执行：

```bash
candump -ta can2,18F021F6:1FFFFFFF
```
主端启动控制后，依次按下L手柄的5、1、6,前4位会从 00 00 --> 00 01 --> 01 01 --> 00 05,既可以开始控制





## 主端启动命令

主端命令都写成可单独复制执行的完整命令块。

如需先确认 Python 环境：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" --version
"${PYTHON}" -m pip install --no-deps -e ./testbed
```

### 1. 主端 rqt 看图

主端 rqt 只用于看图，不参与 TCP 控制和 HDF5 录制主链路。

```bash
source ./scripts/excavator_deploy_network.sh
excavator_apply_host_network_defaults
source ./scripts/ros2_fpv_env.sh
./scripts/start_host_fpv_rqt.sh
```

rqt 中选择：

```text
/camera/color/image_raw
```

### 2. 手柄探测和 remote 遥操作

当前主端接 2 个手柄设备。控制或录制前先列出 pygame 识别到的设备：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
"${PYTHON}" testbed/scripts/gamepad_probe.py
```

然后分别探测两个手柄，确认轴和按钮有变化后按 `Ctrl+C` 退出。

第 0 个手柄：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
"${PYTHON}" testbed/scripts/gamepad_probe.py --watch --joystick-id 0
```

第 1 个手柄：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
"${PYTHON}" testbed/scripts/gamepad_probe.py --watch --joystick-id 1
```

当前 `testbed/testbed/configs/teleop_real_v1.yaml` 使用双手柄映射：

```text
swing  -> joystick 1 axis 0
boom   -> joystick 0 axis 1
stick  -> joystick 1 axis 1
bucket -> joystick 0 axis 0
status buttons -> joystick 0
```

确认两个手柄正常后，在主端启动 remote action sender。这里不用额外写两个
`--joystick-id`，`--input joystick` 会按配置文件读取两个手柄；训练 HDF5 由
从端终端 5 写入 USB。

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
"${PYTHON}" -m testbed.cli.teleop_remote \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --host 192.168.31.170 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --confirm-remote-control
```

第一次真机动作建议只做 10 秒左右的小幅单轴动作，然后按 `Ctrl+C` 停止：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
"${PYTHON}" -m testbed.cli.teleop_remote \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --host 192.168.31.170 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --confirm-remote-control
```

按 `Ctrl+C` 停止主端 sender；sender 会尽量发送一帧零 action/quit 标记。

### 3. 旧方案：控制并录制到主端 USB

仅在需要对比旧链路时使用。正式训练采集优先用“从端终端 5 + 主端
`tb-teleop-remote`”。

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install -q --no-deps -e ./testbed >/tmp/excavator_pip_install.log 2>&1
export PYTHONWARNINGS="ignore::UserWarning"
"${PYTHON}" -m testbed.cli.record_real \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --data-side host \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 192.168.31.170 \
  --bridge-port 8765 \
  --bridge-timeout 2.0 \
  --input joystick \
  --num-episodes 1 \
  --max-steps 50000 \
  --output-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --session-id real_control_joystick_check \
  --live-action-line
```

启动后，终端会用同一行刷新当前输入与发送动作：`raw=[...]` 表示输入源采样的
原始 4 维 action，`send=[...]` 表示实际发送到后端的 4 维 action；同时保留
`age_ms`、`ack`、`fault`、`guard` 状态，不再显示 `host_now_ns` 和
`sensor_timestamp_ns` 两个时间戳；不会每帧新增一行。
手动结束方式：录制过程中按一次 `Ctrl+C`，程序会停止当前 episode 并保存已有数据。

说明：当前 recorder 会先把本 episode 缓存在内存里，episode 结束时再写 HDF5；
`--max-steps 50000` 是“足够大、靠手动结束”的现场用法，不建议单个 episode 连续录很久。
长时间采集建议分多段录制，每段结束后先跑 QC。

注意：主端录制全尺寸 FPV 时，gateway 会把从端 SHM 中的 RGB 图像放进
`read_state` 响应发回主端。`640x480x3` raw RGB 经 base64 后约 1.23 MB/帧，
在现场 Wi-Fi 下会明显拉低 recorder 采样频率。正式训练数据采集使用
从端本地落盘；HDF5 metadata 中的
`camera_width`、`camera_height` 会以实际写入图像尺寸为准，`camera_fps`
会在有图像时间戳时按实际记录帧间隔估算。

### 4. 录完做 QC

如果 USB 仍插在从端，直接在从端跑 QC：

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
MPLCONFIGDIR=/tmp/excavator_mpl python -m testbed.cli.dataset_qc \
  --dataset-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --output-dir /tmp/excavator_dataset_qc_$(date +%Y%m%d_%H%M%S) \
  --profile real
```

如果把 USB 拔回主端，再在主端跑：

```bash
cd ~/Excavator_real_stack
export PYTHON="$HOME/miniforge3/envs/excavator-real-stack/bin/python"
"${PYTHON}" -m pip install --no-deps -e ./testbed
MPLCONFIGDIR=/tmp/excavator_mpl "${PYTHON}" -m testbed.cli.dataset_qc \
  --dataset-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --output-dir /tmp/excavator_dataset_qc_$(date +%Y%m%d_%H%M%S) \
  --profile real
```

## 停止顺序

1. 先在主端停止 `${PYTHON} -m testbed.cli.teleop_remote`。
2. 再在从端停止 `${PYTHON} -m testbed.cli.record_real`，确认 HDF5 已保存。
3. 最后在从端停止 C++ real CAN bridge：

```bash
pkill -f "[e]xcavator_real_bridge" || true
```

如需重启真实控制，重新按“从端启动命令”的顺序启动 5 个终端，再回到主端启动 remote action sender。
