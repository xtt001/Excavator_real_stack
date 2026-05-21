# 真机主从分体运行手册

**当前现场部署约定：主端手柄遥操作 + 主端本地 USB 移动硬盘存 HDF5 + 主端可选 rqt 看图。**

| 角色 | 机器 | 进程 |
|------|------|------|
| 从端 slave | `192.168.31.170` | bridge、gateway、Orbbec、FPV→SHM（**不**在从端录） |
| 主端 host | 操作员 PC | 手柄 + **录制**（写本地 USB 移动硬盘）、可选 rqt |

| 项目 | 约定 |
|------|------|
| 手柄 USB | **主端** |
| HDF5 物理路径 | **主端** `/media/mundane/EXTERNAL_USB/real_teleop_v1` |
| 主端录制连 gateway | **`192.168.31.170:8765`** |
| 从端 bridge 监听 | **`127.0.0.1:8766`**（仅本机 gateway 连） |
| ROS2 | `ROS_DOMAIN_ID=42`；相机 `/camera/color/image_raw/compressed` |

实现方式：主端 `tb-record-real` 读手柄，经 TCP 访问从端 gateway；HDF5 直接写入主端 USB 移动硬盘。

testbed **只连 gateway `8765`**，不要直连 C++ bridge `8766`。

配置：`configs/deploy_network.yaml`、`scripts/excavator_deploy_network.sh`。

---

## 0. 每次来现场录数据 checklist

本节是当前现场的主流程。除非明确要恢复旧的从端落盘方案，否则**不要**执行 SSHFS / `mount_slave_dataset.sh` / `record_host_gamepad_slave_disk.sh`。

当前安全状态：

- 默认只跑完整数据记录链路，不启用真实 CAN 写入。
- bridge 使用 `--can-bus-enabled false --can-simulation true --imu-simulation true` 时不会驱动真机。
- 真机上电、CAN bring-up、真实动作控制必须先按 `docs/real_machine_bringup_checklist.md` 单独确认。

### 0.1 主端检查 USB 数据盘

在主端执行：

```bash
cd ~/Excavator_real_stack
findmnt /media/mundane/EXTERNAL_USB
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

终端 1：C++ bridge，安全仿真 CAN：

```bash
./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8766 \
  --can-bus-enabled false \
  --can-simulation true \
  --imu-simulation true \
  --create-mapping true \
  --heartbeat-timeout-ms 800
```

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

### 0.5 主端检查手柄

在主端执行：

```bash
conda run -n excavator-real-stack python testbed/scripts/gamepad_probe.py --watch --joystick-id 0
```

确认轴和按钮有变化后按 `Ctrl+C` 退出。

当前 status 映射：

- `button0` → status bit0 → 点火 ignition
- `button1` → status bit1 → 熄火 flameout
- `button10` → status bit10 → 急停 estop
- `button11` → 控制组切换

注意：这是 pygame button index，不一定等于手柄外壳印刷的 A/B/X/Y 名称。

### 0.6 主端开始录制到本地 USB

先做短样本：

```bash
conda run -n excavator-real-stack tb-record-real \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --data-side host \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 192.168.31.170 \
  --bridge-port 8765 \
  --bridge-timeout 2.0 \
  --input joystick \
  --num-episodes 1 \
  --max-steps 200 \
  --output-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --session-id joystick_pipeline_check
```

确认短样本正常后，再把 `--max-steps` 改成正式采集长度，或去掉该参数使用配置默认值。

### 0.7 录完做 QC

```bash
MPLCONFIGDIR=/tmp/excavator_mpl conda run -n excavator-real-stack tb-dataset-qc \
  --dataset-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --output-dir /tmp/excavator_dataset_qc_$(date +%Y%m%d_%H%M%S) \
  --profile real
```

当前安全仿真 CAN 下的预期结果：

- `has_images=1`，FPV 图像不是 placeholder。
- `controller_ack_rate=1.0`。
- `qpos/qvel/env_state` 可能全 0，因为真机/CAN 没启用。
- 主端跨网录 640x480 raw FPV 时，可能出现 `sensor_timeout` guard；这表示记录循环相对 `sensor_timeout_s=0.20` 偏慢，不代表相机没有数据。

### 0.8 如果本次要真实控车

真实控车不是换一个主端录制命令，而是把从端 bridge 从安全仿真 CAN 切到真实 CAN 写入。主端仍然用 `tb-record-real --input joystick`，它会一边写 HDF5，一边通过 gateway 下发手柄动作。

进入本节前必须满足：

- 现场有人在机器旁，急停和人工接管可用。
- 挖掘机已上电，确认可以进入远程/先导等必要状态。
- 人员离开作业半径，先从最低能量、最小动作开始。
- 已确认 CAN 接口名、bitrate、机器总线连接方式。

当前默认接口名是：

```text
控制 CAN: can0
IMU CAN: can1
bitrate: 250000（按现场总线确认后再用）
```

从端先检查 CAN 接口：

```bash
ssh mundane@192.168.31.170 '
  ip -br link | grep can
  ip -details link show can0
  ip -details link show can1
'
```

配置 CAN bitrate 并拉起接口（确认 bitrate 正确后执行）：

```bash
ssh mundane@192.168.31.170 '
  cd /media/mundane/D/Excavator_real_stack
  control/setup/setup_can.sh can0 250000
  control/setup/setup_can.sh can1 250000
'
```

先做只读 CAN 探测，不发控制帧：

```bash
ssh mundane@192.168.31.170 '
  cd /media/mundane/D/Excavator_real_stack
  .venv/bin/python scripts/can_probe.py \
    --interface can0 \
    --duration-s 10 \
    --ids 18F021F6 18F022F6 18F023F6 \
    --output-dir /tmp/excavator_can_probe_$(date +%Y%m%d_%H%M%S)
'
```

如果 CAN 探测没有确认 bus health、ID、bitrate，不要继续。

停止安全仿真 bridge，改启真实 CAN bridge：

```bash
ssh mundane@192.168.31.170 '
  pkill -f "[e]xcavator_real_bridge" || true
  cd /media/mundane/D/Excavator_real_stack
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
'
```

这个终端需要保持运行。gateway 不需要改，仍然连 `127.0.0.1:8766`。

第一次不要直接大幅手柄操作。先用单轴小幅动作确认方向、停止和急停：

```bash
ssh mundane@192.168.31.170 '
  cd /media/mundane/D/Excavator_real_stack
  .venv/bin/python scripts/one_axis_bringup.py \
    --host 127.0.0.1 \
    --port 8766 \
    --axis swing \
    --amplitude 0.03 \
    --duration-s 0.5 \
    --confirm-hardware-motion
'
```

逐个确认 `swing`、`boom`、`stick`、`bucket`。如果方向反、错轴、动作抖、不能停、急停不生效，立即停止并回到安全仿真 bridge。

确认后，主端用同一条录制命令进行“边控制边录制”：

```bash
conda run -n excavator-real-stack tb-record-real \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --data-side host \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 192.168.31.170 \
  --bridge-port 8765 \
  --bridge-timeout 2.0 \
  --input joystick \
  --num-episodes 1 \
  --max-steps 200 \
  --output-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --session-id real_control_joystick_check
```

退出真实控制后，重启安全仿真 bridge：

```bash
ssh mundane@192.168.31.170 '
  pkill -f "[e]xcavator_real_bridge" || true
  cd /media/mundane/D/Excavator_real_stack
  ./bridge/build/excavator_real_bridge \
    --host 127.0.0.1 \
    --port 8766 \
    --can-bus-enabled false \
    --can-simulation true \
    --imu-simulation true \
    --create-mapping true \
    --heartbeat-timeout-ms 800
'
```

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

网络：从端放行 **TCP 8765**；主端 DDS 组播不通时 `export EXCAVATOR_ROS_PEER_IP=192.168.31.170`。

---

## 2. 手动启动参考：从端 `192.168.31.170`（旧排障用）

优先使用 §0 checklist。本节保留给排障或需要手动拆开 4 个终端观察日志时使用。

每个终端先：

```bash
cd /media/mundane/D/Excavator_real_stack
source .venv/bin/activate
export EXCAVATOR_SLAVE_IP=192.168.31.170
```

**不要在从端起 `tb-record-real`。** 当前建议顺序：bridge → Orbbec → FPV subscriber → gateway。

### 终端 1 — C++ control bridge

#### 仿真 CAN（联调默认）

```bash
./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8766 \
  --can-bus-enabled false \
  --can-simulation true \
  --imu-simulation true \
  --create-mapping true
```

#### 真机 CAN（E-stop、单轴小幅度；接口名按现场改）

```bash
./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8766 \
  --can-if can0 \
  --imu-if can1 \
  --can-bus-enabled true \
  --can-simulation false \
  --imu-simulation false \
  --heartbeat-timeout-ms 800
```

`heartbeat-timeout-ms` 须大于主端一轮 `send_action`+`read_state`+网络延迟（主从分体建议 **800～1000**，默认 200 易误触发 watchdog）。
当前 `tb-record-real` 在 `backend=bridge_tcp` 时会启用 action pump，按 `real.control_pump.hz` 重复发送最后一次安全速度命令；`read_state`、图像和 HDF5 录制慢下来时不应再饿死控制 heartbeat。联调早期仍建议先保守使用 `800～1000ms`，确认 `snapshot_age_ms` 和控制周期稳定后再收紧。

### 终端 2 — Orbbec

```bash
./scripts/start_orbbec_fpv_camera.sh
```

### 终端 3 — compressed → SHM（供 gateway 写入录制帧）

```bash
EXCAVATOR_ROS_WS=/home/mundane/orbbec_ws ./scripts/start_fpv_subscriber_py.sh
```

### 终端 4 — gateway

```bash
./scripts/start_bridge_gateway.sh --fpv-source auto --fpv-max-stale-ms 1000
```

监听 `0.0.0.0:8765`，转发本机 `127.0.0.1:8766`。

---

## 3. 旧 SSHFS 主端流程（当前现场不使用）

当前现场使用主端本地 USB 移动硬盘落盘，按 §0 执行即可。本节仅用于恢复“主端录制、SSHFS 写从端盘”的旧方案。

每个终端先：

```bash
cd ~/Excavator_real_stack
conda activate excavator-real-stack
export EXCAVATOR_SLAVE_IP=192.168.31.170
```

### 终端 A — 挂载从端数据集（每次录制前）

```bash
./scripts/mount_slave_dataset.sh
```

默认挂载从端 `/data/real_teleop_v1`；D 盘部署时先 `export EXCAVATOR_SLAVE_DATASET_DIR=/media/mundane/D/real_teleop_v1`。
映射到主端 `~/mnt/slave_real_teleop`。
录完可选：`./scripts/umount_slave_dataset.sh`。

### 终端 B — 手柄 + 录制（HDF5 写入从端盘）

```bash
./scripts/record_host_gamepad_slave_disk.sh
```

常用参数（会传给 `tb-record-real`）：

```bash
./scripts/record_host_gamepad_slave_disk.sh \
  --num-episodes 1 \
  --max-steps 200
```

等价手动命令：

```bash
tb-record-real \
  --config testbed/testbed/configs/teleop_real_v1.yaml \
  --data-side host \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 192.168.31.170 \
  --bridge-port 8765 \
  --input joystick \
  --output-dir ~/mnt/slave_real_teleop
```

### 终端 C — rqt 看图（可选）

```bash
./scripts/start_host_fpv_rqt.sh
```

rqt 选 `/camera/color/image_raw`；图源为从端 `/camera/color/image_raw/compressed`。

**DDS 检查：**

```bash
source scripts/ros2_fpv_env.sh
source scripts/excavator_deploy_network.sh
excavator_apply_host_network_defaults
source scripts/ros2_multihost_env.sh
source /opt/ros/humble/setup.bash
ros2 topic hz /camera/color/image_raw/compressed
```

---

## 4. 仿真 vs 真机

| 步骤 | 仿真 | 真机 |
|------|------|------|
| 从端终端 1 | `can-simulation true`，`can-bus-enabled false` | `can-simulation false`，`can-bus-enabled true` |
| 从端 2～4 | 相同 | 相同 |
| 主端录制 | `192.168.31.170:8765` + 本地 USB HDF5 | 相同；真实 CAN bridge 会实际控车 |

当前现场录完后在**主端** QC：

```bash
MPLCONFIGDIR=/tmp/excavator_mpl conda run -n excavator-real-stack tb-dataset-qc \
  --dataset-dir /media/mundane/EXTERNAL_USB/real_teleop_v1 \
  --output-dir /tmp/excavator_dataset_qc_$(date +%Y%m%d_%H%M%S) \
  --profile real
```

---

## 5. 数据流

```text
主端: 手柄 -> tb-record-real
          | TCP 192.168.31.170:8765
          v
从端: gateway -> bridge 127.0.0.1:8766 -> control/CAN
          ^ read_state 图像来自 SHM <- fpv_subscriber <- Orbbec compressed
          |
主端: HDF5 写 /media/mundane/EXTERNAL_USB/real_teleop_v1
```

---

## 6. IP / 端口速查

| 用途 | 地址 | 在哪填 |
|------|------|--------|
| 主端录制 / 控车 | `192.168.31.170:8765` | `tb-record-real --bridge-host 192.168.31.170 --bridge-port 8765` |
| 从端 bridge | `127.0.0.1:8766` | 仅从端终端 1 |
| 主端 ROS2 peer | `192.168.31.170` | `EXCAVATOR_ROS_PEER_IP` |
| HDF5 | 主端 `/media/mundane/EXTERNAL_USB/real_teleop_v1` | `--output-dir` |

**不要**在从端用 `192.168.31.170` 连 gateway；**不要**在主端填 `127.0.0.1:8765`（除非整机单机调试）。

---

## 7. 单机联调（一台电脑）

手柄、录制、从端服务都在本机时：

```bash
# 终端 1～4：同 §1，IP 均为 127.0.0.1
./bridge/build/excavator_real_bridge --host 127.0.0.1 --port 8766 \
  --can-bus-enabled false --can-simulation true --imu-simulation true --create-mapping true
./scripts/start_orbbec_fpv_camera.sh
./scripts/start_fpv_subscriber_py.sh
./scripts/start_bridge_gateway.sh

# 终端 B：可不挂载 SSHFS，直接录到本地目录
tb-record-real --config testbed/testbed/configs/teleop_real_v1.yaml \
  --data-side slave --backend bridge_tcp --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 --bridge-port 8765 --input joystick \
  --output-dir data/real_teleop_v1_sim
```

---

## 8. 常见问题

| 现象 | 处理 |
|------|------|
| `请先挂载从端目录` | 你走到了旧 SSHFS 脚本；当前现场不要用 `record_host_gamepad_slave_disk.sh`，改用 §0.6 的 `tb-record-real --output-dir /media/mundane/EXTERNAL_USB/real_teleop_v1` |
| HDF5 出现在主端仓库下 | `--output-dir` 没指到 USB；改为 `/media/mundane/EXTERNAL_USB/real_teleop_v1` |
| 主端连不上 8765 | 从端 gateway 已起、`ss -tlnp \| grep 8765`、`ping 192.168.31.170` |
| 主端 rqt 灰屏 | 从端 Orbbec + 终端 3；`ros2 topic hz .../compressed` |
| gateway 返回 `bridge_placeholder_fpv` | FPV SHM 在 gateway 启动后才创建；确认 `/dev/shm/excavator_fpv_v1` 新鲜，然后重启 gateway |
| SSHFS 断开 | 旧流程问题；当前现场本地 USB 落盘，不使用 SSHFS |
| 从端误开 `tb-record-real` | 关闭；录制只在主端 |
| bridge 日志反复 `client connected/disconnected` | **旧版 gateway** 每请求新建 TCP；更新后 gateway 对 8766 **长连接复用**，仅首次 `upstream bridge connected` |
| `watchdog forced zero command after … ms` | 超过 `heartbeat-timeout-ms` 未收到 `send_action`；确认 `real.control_pump.enabled=true`、testbed 连接 gateway `8765`，并检查 `read_state`/网络是否长时间卡住 |
| 手柄能录 action 但机器不动 | bridge 仍在安全仿真 CAN；真实控车要执行 §0.8 |
| qpos/qvel 全 0 | 安全仿真 CAN 或真机反馈未接入；真实采集前要确认 CAN 反馈 |

---

## 9. 改 IP

1. `configs/deploy_network.yaml` → `slave.ip`
2. `scripts/excavator_deploy_network.sh` → `EXCAVATOR_SLAVE_IP`
3. `testbed/testbed/configs/teleop_real_v1.yaml` → `data_side_defaults.host.bridge.host`
