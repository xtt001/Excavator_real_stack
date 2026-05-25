# 真机主从分体运行手册

**当前现场部署约定：真实 CAN bridge + 主端手柄 remote action + 从端本地 USB 移动硬盘存 HDF5 + 主端可选 rqt 看图。**

| 角色 | 机器 | 进程 |
|------|------|------|
| 从端 slave | `192.168.31.170` | real CAN bridge、gateway、Orbbec、FPV→SHM、`tb-record-real --input remote` |
| 主端 host | 操作员 PC | 手柄输入、`tb-teleop-remote`、可选 rqt |

| 项目 | 约定 |
|------|------|
| 手柄 USB | **主端** |
| HDF5 物理路径 | **从端** `/media/mundane/EXTERNAL_USB/real_teleop_v1` |
| 主端 remote action | **`192.168.31.170:8770`** |
| 从端 recorder 连 gateway | **`127.0.0.1:8765`** |
| 从端 bridge 监听 | **`127.0.0.1:8766`**（仅本机 gateway 连） |
| ROS2 | `ROS_DOMAIN_ID=42`；相机 `/camera/color/image_raw/compressed` |

实现方式：主端 `tb-teleop-remote` 只读手柄并发送小 action 包；从端 `tb-record-real --input remote` 本地读取真实传感器/FPV、本地下发控制命令，并把 HDF5 写入从端 USB 移动硬盘。主端 rqt 只用于看 ROS compressed 图像，不参与训练数据写盘。

从端 recorder **只连本机 gateway `127.0.0.1:8765`**，不要直连 C++ bridge `8766`；主端只连从端 remote action server `8770`。

配置：`configs/deploy_network.yaml`、`scripts/excavator_deploy_network.sh`。

---

## 0. 每次来现场录数据 checklist

本节是当前现场的主流程。训练数据采集默认从端本地落盘；除非明确要做旧链路对比，否则不要让主端 `tb-record-real` 拉 raw RGB 写 HDF5。

当前真实控制前提：

- 现场人员必须离开作业半径，急停和人工接管可用。
- 挖机已上电，CAN 线已接好，确认可以进入远程/先导等必要状态。
- 从端录制使用 `--input remote --backend bridge_tcp --state-reader bridge_tcp`：主端手柄 action 会写入 HDF5，并通过从端本地 gateway 发送到真实 CAN bridge。
- 第一次动作只做小幅、低速、单轴验证；方向反、错轴、不能停或急停异常时，立即停 bridge。

### 0.1 从端检查 USB 数据盘

在从端执行。当前现场移动硬盘 label 是 `EXTERNAL_USB`：

```bash
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINTS,MODEL
sudo mkdir -p /media/mundane/EXTERNAL_USB
findmnt /media/mundane/EXTERNAL_USB || \
  sudo mount /dev/disk/by-label/EXTERNAL_USB /media/mundane/EXTERNAL_USB
mkdir -p /media/mundane/EXTERNAL_USB/real_teleop_v1
test -w /media/mundane/EXTERNAL_USB/real_teleop_v1 && echo USB_WRITE_OK
```

### 0.2 从端检查 D 盘和仓库

在主端执行：

```bash
ssh mundane@192.168.31.170 '
  findmnt /media/mundane/D || sudo mount /dev/disk/by-label/D /media/mundane/D
  cd /media/mundane/D/Excavator_real_stack
  git status --short --branch
'
```

正常情况下从端仓库应在：

```text
/media/mundane/D/Excavator_real_stack
```

如果只是录数据，**不要同步/覆盖从端代码**。只有代码确实改过且需要部署到从端时，才单独做同步或 git 更新。

### 0.3 从端启动 4 个服务

以下命令在从端执行。当前推荐用 **VS Code Remote-SSH 新窗口**控制 Jetson，这样可以直接在远端仓库里开 4 个集成终端，日志和文件都在同一个远程窗口里。

VS Code 推荐流程：

1. 在主端 VS Code 里开一个新窗口。
2. `Remote-SSH: Connect to Host...`，连接：

   ```text
   mundane@192.168.31.170
   ```

3. 在 Remote-SSH 窗口里打开从端仓库：

   ```text
   /media/mundane/D/Excavator_real_stack
   ```

4. 在这个远程窗口里开 4 个 VS Code integrated terminals，分别命名或记作：

   ```text
   bridge
   orbbec
   fpv
   gateway
   ```

5. 每个远端 terminal 先执行：

   ```bash
   cd /media/mundane/D/Excavator_real_stack
   source .venv/bin/activate
   ```

然后按下面 4 个终端命令启动服务。

如果不用 VS Code，也可以用 `tmux` 作为命令行 fallback。主端进入远端 4 终端会话：

```bash
ssh -t mundane@192.168.31.170 'tmux attach -t excavator || tmux new -s excavator -c /media/mundane/D/Excavator_real_stack'
```

`tmux` 常用键：

- `Ctrl-b c`：新窗口
- `Ctrl-b n` / `Ctrl-b p`：下一个 / 上一个窗口
- `Ctrl-b 0`～`Ctrl-b 3`：切到第 0～3 个窗口
- `Ctrl-b d`：断开但保持远端服务继续运行

如果第一次使用，先建 4 个命名窗口：

```bash
ssh mundane@192.168.31.170 '
  cd /media/mundane/D/Excavator_real_stack
  tmux new-session -d -s excavator -n bridge -c /media/mundane/D/Excavator_real_stack 2>/dev/null || true
  tmux new-window -t excavator: -n orbbec -c /media/mundane/D/Excavator_real_stack 2>/dev/null || true
  tmux new-window -t excavator: -n fpv -c /media/mundane/D/Excavator_real_stack 2>/dev/null || true
  tmux new-window -t excavator: -n gateway -c /media/mundane/D/Excavator_real_stack 2>/dev/null || true
  tmux list-windows -t excavator
'
```

终端 1：配置 CAN 并启动 C++ real CAN bridge：

```bash
control/setup/setup_can.sh can0 250000
control/setup/setup_can.sh can1 250000
ip -details link show can0
ip -details link show can1

./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8766 \
  --can-if can0 \
  --imu-if can1 \
  --can-bus-enabled true \
  --can-simulation false \
  --imu-simulation false \
  --create-mapping true \
  --heartbeat-timeout-ms 800
```

这个 bridge 会接入真实 CAN。保持该终端运行；停止它即可停止主端继续控制底层 CAN。

终端 2：Orbbec 相机：

```bash
./scripts/start_orbbec_fpv_camera.sh
```

终端 3：FPV compressed → SHM：

```bash
EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws ./scripts/start_fpv_subscriber_py.sh
```

说明：当前 Jetson 上 `excavator_ros2_bridge` 在 `/home/mundane/orbbec_ws/install` 里；显式设置 `EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws` 可避免误 source `~/ros2_ws`。

终端 4：gateway。**必须在终端 3 已创建 `/dev/shm/excavator_fpv_v1` 之后启动**，否则 gateway 可能一直返回 placeholder FPV。

```bash
./scripts/start_bridge_gateway.sh --fpv-source auto --fpv-max-stale-ms 1000
```

### 0.4 从主端做链路检查

在主端执行：

```bash
ssh mundane@192.168.31.170 '
  cd /media/mundane/D/Excavator_real_stack
  ss -tlnp | grep -E ":8765|:8766"
  ls -lh /dev/shm/excavator_fpv_v1
'
```

检查相机 topic：

```bash
ssh mundane@192.168.31.170 '
  cd /media/mundane/D/Excavator_real_stack
  source ./scripts/ros2_fpv_env.sh
  source ./scripts/source_ros_stack.sh >/tmp/source_ros_stack_check.log 2>&1
  timeout 8 ros2 topic hz /camera/color/image_raw/compressed
'
```

正常应接近 `30 Hz`。

检查 gateway 返回的 FPV 来源：

```bash
ssh mundane@192.168.31.170 '
  cd /media/mundane/D/Excavator_real_stack
  PYTHONPATH=$PWD/ros2_bridge .venv/bin/python -c '"'"'
import json, socket
msg = {"version": 1, "type": "read_state.request", "payload": {"step_id": 0}}
s = socket.create_connection(("127.0.0.1", 8765), timeout=3)
s.settimeout(5)
s.sendall((json.dumps(msg) + "\n").encode())
chunks = []
while True:
    b = s.recv(65536)
    if not b:
        break
    chunks.append(b)
    if b"\n" in b:
        break
r = json.loads(b"".join(chunks).split(b"\n", 1)[0].decode())
fpv = r["payload"]["images"]["fpv"]
print("ok", r.get("ok"))
print("fpv_source", fpv.get("source"))
print("fpv_shape", fpv.get("payload", {}).get("shape"))
s.close()
'"'"'
'
```

正常应看到：

```text
fpv_source ros2_compressed_fpv
fpv_shape [480, 640, 3]
```

如果看到 `bridge_placeholder_fpv`，先确认终端 3 正在运行并且 `/dev/shm/excavator_fpv_v1` 新鲜，然后重启终端 4 gateway。

### 0.5 主端 rqt 看相机画面（可选但建议）

主端 rqt 需要主端本机安装 ROS2 和 `rqt_image_view`。如果这台主端没有 `/opt/ros/<distro>/setup.bash`，可以跳过本节；录制 HDF5 不依赖主端 ROS2，图像由从端 recorder 本地读取 gateway/SHM。

版本约定：

- 从端 Jetson 是 Ubuntu 22.04，硬件侧 ROS2 基线是 **Humble**。
- 当前新主端是 Ubuntu 24.04，本机 apt 支持的 ROS2 是 **Jazzy**。Jazzy 只用于主端本地 rqt 看图，不参与控制/录制主链路。
- Humble/Jazzy 跨发行版 DDS 订阅相同 `sensor_msgs/msg/CompressedImage` 通常可以作为现场可视化尝试，但不是核心链路依赖；如果 rqt 发现不了 topic，用 §0.4 gateway 检查和 HDF5/QC 验证图像。

Ubuntu 24.04 主端首次安装可选 rqt 依赖：

```bash
sudo apt install -y ros-jazzy-ros-base ros-jazzy-rqt-image-view \
  ros-jazzy-image-transport-plugins ros-jazzy-compressed-image-transport \
  ros-jazzy-cv-bridge ros-jazzy-rmw-cyclonedds-cpp
```

在主端本地仓库执行：

```bash
cd ~/Excavator_real_stack
conda activate excavator-real-stack
./scripts/start_host_fpv_rqt.sh
```

rqt 里选择：

```text
/camera/color/image_raw
```

图像源是从端 Orbbec 发布的 `/camera/color/image_raw/compressed`，主端只用于显示；HDF5 录制里的图像由从端 recorder 通过本地 gateway/SHM 读取。

如果报错：

```text
error: no ROS2 setup.bash found under /opt/ros.
```

说明主端没装 ROS2。当前新主端是 Ubuntu 24.04 时，不要为了 rqt 强行安装 Humble；按上面的 Jazzy 包安装，或直接跳过 rqt，用 §0.4 的 gateway `fpv_source` 检查和录制后的 HDF5/QC 确认真图像是否进入数据集。

如果 rqt 灰屏，先在从端检查：

```bash
cd /media/mundane/D/Excavator_real_stack
source ./scripts/ros2_fpv_env.sh
source ./scripts/source_ros_stack.sh >/tmp/source_ros_stack_check.log 2>&1
ros2 topic hz /camera/color/image_raw/compressed
```

### 0.6 主端先打开并检查手柄

在主端执行：

```bash
conda run -n excavator-real-stack python testbed/scripts/gamepad_probe.py --watch --joystick-id 0
```

控制或录制前先做这一步，确认轴和按钮有变化后按 `Ctrl+C` 退出，再启动 §0.7 的只控制命令或 §0.8 的录制命令。

当前 status 映射：

- `button0` → status bit0 → 点火 ignition
- `button1` → status bit1 → 熄火 flameout
- `button10` → status bit10 → 急停 estop
- `button11` → 控制组切换

注意：这是 pygame button index，不一定等于手柄外壳印刷的 A/B/X/Y 名称。

### 0.7 主端 remote action， 从端本地录制

先在从端启动 recorder，等待主端 action。此进程本地连接 gateway
`127.0.0.1:8765`，并把 HDF5 写到从端 USB：

```bash
conda run -n excavator-real-stack tb-record-real \
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

再在主端启动手柄 sender。主端只发送 action，不写训练 HDF5：

```bash
conda run -n excavator-real-stack tb-teleop-remote \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --host 192.168.31.170 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --confirm-remote-control
```

说明：

- `tb-teleop-remote` 读取主端手柄，向从端 `8770` 发送 50Hz action 小包。
- `tb-record-real --input remote` 是唯一控制 owner：guard、action pump、gateway 和 HDF5 都在从端。
- 按 `Ctrl+C` 停止主端 sender；从端 recorder 会收到 quit/零 action，或在 action 超时后自动输出零速。

### 0.8 旧方案：主端控制并录制到本地 USB

仅用于对比旧链路。正式训练采集使用 0.7 的从端本地录制流程。

主端录制全尺寸 FPV 会通过 gateway 把 raw RGB 图像放进每次 `read_state`
响应。`640x480x3` 经 base64 后约 1.23 MB/帧，现场 Wi-Fi 下无法稳定支撑
50Hz recorder。用于训练的数据采集优先选择从端落盘；HDF5 的 `camera_width` / `camera_height` 以实际写入图像为准，
`camera_fps` 在有图像时间戳时按实际记录帧间隔估算，而不是照抄配置值。

### 0.9 录完做 QC

```bash
MPLCONFIGDIR=/tmp/excavator_mpl conda run -n excavator-real-stack tb-dataset-qc \
  --dataset-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --output-dir /tmp/excavator_dataset_qc_$(date +%Y%m%d_%H%M%S) \
  --profile real
```

当前真实控制采集的预期结果：

- `has_images=1`，FPV 图像不是 placeholder。
- `controller_ack_rate=1.0`。
- `controller_fault_code` 应为空或为真实 bridge 返回的状态；不应再是 `noop`。
- `commanded_action` 应随手柄动作变化。
- `qpos/qvel/env_state` 取决于真实 CAN 反馈是否已经接入和解析。
- `remote_action_stale=0` 且 `remote_action_age_ms` 稳定，说明主端手柄 action 正常进入从端 recorder。

### 0.10 停止真实控制

先停止主端 `tb-teleop-remote`，再停止从端 `tb-record-real` 并确认 HDF5 已保存，最后再停从端 bridge：

```bash
pkill -f "[e]xcavator_real_bridge" || true
```

如需重新开始控制，回到 §0.3 的终端 1 重新启动 real CAN bridge。

---

## 1. 首次准备

路径：`~/Excavator_real_stack`。**主端、从端**均需 Python + bridge 编译。

```bash
cd ~/Excavator_real_stack
scripts/setup_env.sh venv
source .venv/bin/activate
pip install -e testbed/

sudo apt-get install -y build-essential cmake libeigen3-dev
cmake -S bridge -B bridge/build
cmake --build bridge/build -j"$(nproc)"
```

**从端** Orbbec + ROS2：

```bash
mkdir -p ~/orbbec_ws/src
cd ~/orbbec_ws && colcon build --symlink-install --packages-select orbbec_camera
ln -sf ~/Excavator_real_stack/ros2_bridge/excavator_ros2_bridge ~/orbbec_ws/src/
colcon build --symlink-install --packages-select excavator_ros2_bridge
sudo apt install -y ros-humble-compressed-image-transport ros-humble-cv-bridge \
  ros-humble-image-transport ros-humble-rqt-image-view
```

**主端** rqt 看图（可选；Ubuntu 24.04 用 Jazzy）：

```bash
sudo apt install -y ros-jazzy-ros-base ros-jazzy-rqt-image-view \
  ros-jazzy-image-transport-plugins ros-jazzy-compressed-image-transport \
  ros-jazzy-cv-bridge ros-jazzy-rmw-cyclonedds-cpp
```

说明：主端 ROS2 只用于可视化，不参与 `tb-teleop-remote` / `tb-record-real` 的控制和录制主链路。slave 仍以 Ubuntu 22.04 + Humble 为准。

**从端数据目录**（旧 SSHFS 从端落盘流程才需要；当前本地 USB 落盘不需要）：

**A. 系统 `/data`（需 root，很多现场无权限）**

```bash
sudo mkdir -p /data/real_teleop_v1 && sudo chown "$USER":"$USER" /data/real_teleop_v1
ls -la /data/real_teleop_v1
```

**B. 外置盘 D 盘下（推荐：仓库在 `/media/mundane/D/Excavator_real_stack` 时）**

在**从端**执行（`real_teleop_v1` 是**文件夹**，在 D 盘根下，不在项目目录里）：

```bash
mkdir -p /media/mundane/D/real_teleop_v1
ls -la /media/mundane/D/real_teleop_v1
```

**主端**挂载前指定从端路径与 SSH 用户名（旧 SSHFS 流程）：

```bash
export EXCAVATOR_SLAVE_IP=192.168.31.170          # 从端 IP，按现场改
export EXCAVATOR_SLAVE_SSH_USER=mundane           # 从端登录名
export EXCAVATOR_SLAVE_DATASET_DIR=/media/mundane/D/real_teleop_v1
./scripts/mount_slave_dataset.sh
```

录完后在从端查看：`ls -la /media/mundane/D/real_teleop_v1/episode_*.hdf5`

**主端**（旧 SSHFS 写从盘才需要）：

```bash
sudo apt install -y sshfs
ssh-copy-id "${USER}@192.168.31.170"   # 推荐免密
```

网络：从端放行 **TCP 8765** 和 **TCP 8770**；主端 DDS 组播不通时 `export EXCAVATOR_ROS_PEER_IP=192.168.31.170`。

---

## 2. 手动启动参考：从端真实服务

优先使用 §0 checklist。本节只作为拆开终端看日志时的参考。

每个从端终端先执行：

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
```

从端需要启动 `tb-record-real --input remote` 作为本地 recorder。不要在从端读取主端手柄；主端只运行 `tb-teleop-remote`。

### 终端 1 — real CAN bridge

```bash
control/setup/setup_can.sh can0 250000
control/setup/setup_can.sh can1 250000
ip -details link show can0
ip -details link show can1

./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8766 \
  --can-if can0 \
  --imu-if can1 \
  --can-bus-enabled true \
  --can-simulation false \
  --imu-simulation false \
  --create-mapping true \
  --heartbeat-timeout-ms 800
```

### 终端 2 — Orbbec

```bash
./scripts/start_orbbec_fpv_camera.sh
```

### 终端 3 — compressed → SHM

```bash
EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws ./scripts/start_fpv_subscriber_py.sh
```

### 终端 4 — gateway

```bash
./scripts/start_bridge_gateway.sh --fpv-source auto --fpv-max-stale-ms 1000
```

gateway 监听 `0.0.0.0:8765`，并长连接转发到本机 C++ bridge `127.0.0.1:8766`。

---

## 3. 主端命令速查

每个主端终端先执行：

```bash
cd ~/Excavator_real_stack
conda activate excavator-real-stack
```

主端发送 remote action：

```bash
tb-teleop-remote \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --host 192.168.31.170 \
  --port 8770 \
  --input joystick \
  --rate-hz 50 \
  --confirm-remote-control
```

从端本地录制：

```bash
tb-record-real \
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

录完在从端或插回主端后 QC：

```bash
MPLCONFIGDIR=/tmp/excavator_mpl tb-dataset-qc \
  --dataset-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --output-dir /tmp/excavator_dataset_qc_$(date +%Y%m%d_%H%M%S) \
  --profile real
```

---

## 4. 数据流

```text
主端手柄
  -> tb-teleop-remote
  -> TCP 192.168.31.170:8770
  -> 从端 tb-record-real --input remote
  -> 从端 ActionGuard / action pump
  -> 从端 gateway 127.0.0.1:8765
  -> 127.0.0.1:8766 C++ real CAN bridge
  -> can0/can1
  -> 挖机控制器/传感器

从端 Orbbec
  -> ROS2 compressed image
  -> fpv_subscriber
  -> /dev/shm/excavator_fpv_v1
  -> 从端 gateway read_state
  -> 从端 tb-record-real

tb-record-real:
  接收 remote action、控制、读状态，并把 HDF5 写到 /media/mundane/EXTERNAL_USB/real_teleop_v1。
```

---

## 5. IP / 端口速查

| 用途 | 地址 | 在哪填 |
|------|------|--------|
| 主端连从端 remote action | `192.168.31.170:8770` | `tb-teleop-remote --host 192.168.31.170 --port 8770` |
| 从端 recorder 连本机 gateway | `127.0.0.1:8765` | `tb-record-real --bridge-host 127.0.0.1 --bridge-port 8765` |
| 从端 gateway 连 C++ bridge | `127.0.0.1:8766` | 仅从端内部使用 |
| 从端 CAN | `can0` / `can1` | 终端 1 bridge 参数 |
| 主端 ROS2 peer | `192.168.31.170` | `EXCAVATOR_ROS_PEER_IP` |
| HDF5 | `/media/mundane/EXTERNAL_USB/real_teleop_v1` | `tb-record-real --output-dir` |

从端 recorder 必须把 bridge host 填成 `127.0.0.1`，否则图像仍会跨网络。

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `Address already in use` | 从端已有 bridge/gateway 占用端口；先 `pkill -f "[e]xcavator_real_bridge"` 或 `pkill -f "[g]ateway_server"`，再重启对应终端 |
| 主端连不上 `8770` | 从端 recorder 未启动、IP 不对、防火墙或网络不通；从端查 `ss -tlnp \| grep 8770`，主端查 `ping 192.168.31.170` |
| 从端 recorder 连不上 `8765` | 从端 gateway 未启动；从端查 `ss -tlnp \| grep 8765` |
| gateway 返回 `bridge_placeholder_fpv` | 终端 3 没有创建新鲜 SHM；确认 `/dev/shm/excavator_fpv_v1` 更新时间，再重启 gateway |
| rqt 灰屏 | 先查从端 Orbbec 和终端 3；rqt 只是可视化，不影响 remote action 和从端本地录制 |
| HDF5 出现在从端仓库下 | `--output-dir` 没指到 USB；改为 `/media/mundane/EXTERNAL_USB/real_teleop_v1` |
| 机器不动但 `ack=true` | 检查发动机/液压/先导/远程使能状态、status bit 点火、CAN 接线和 bridge 日志 |
| 机器动作方向或轴不对 | 立即停主端控制，检查 `teleop_real_v1.yaml` 的 `axis_map`、`joystick_ids`、`invert` |
| 停止后仍担心有输出 | 先停主端命令，再停从端 bridge：`pkill -f "[e]xcavator_real_bridge"` |

---

## 7. 改 IP

1. `configs/deploy_network.yaml` → `slave.ip`
2. `scripts/excavator_deploy_network.sh` → `EXCAVATOR_SLAVE_IP`
3. `testbed/testbed/configs/teleop_real_v1.yaml` → `data_side_defaults.host.bridge.host`
