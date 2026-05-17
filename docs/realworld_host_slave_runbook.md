# 真机主从分体运行手册

**部署约定：主端手柄遥操作 + 从端存 HDF5 + 主端可选 rqt 看图。**

| 角色 | 机器 | 进程 |
|------|------|------|
| 从端 slave | `192.168.31.170` | bridge、gateway、Orbbec、FPV→SHM（**不**在从端录） |
| 主端 host | 操作员 PC | 手柄 + **录制**（SSHFS 写到从盘）、可选 rqt |

| 项目 | 约定 |
|------|------|
| 手柄 USB | **主端** |
| HDF5 物理路径 | **从端** `/data/real_teleop_v1` 或 **`/media/mundane/D/real_teleop_v1`**（见 §0） |
| 主端录制连 gateway | **`192.168.31.170:8765`** |
| 从端 bridge 监听 | **`127.0.0.1:8766`**（仅本机 gateway 连） |
| ROS2 | `ROS_DOMAIN_ID=42`；相机 `/camera/color/image_raw/compressed` |

实现方式：主端 `tb-record-real` 读手柄，经 TCP 访问从端 gateway；HDF5 通过 **SSHFS** 写入从端目录（`scripts/mount_slave_dataset.sh` + `scripts/record_host_gamepad_slave_disk.sh`）。

testbed **只连 gateway `8765`**，不要直连 C++ bridge `8766`。

配置：`configs/deploy_network.yaml`、`scripts/excavator_deploy_network.sh`。

---

## 0. 首次准备

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

**从端数据目录**（二选一）：

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

**主端**挂载前指定从端路径与 SSH 用户名：

```bash
export EXCAVATOR_SLAVE_IP=192.168.31.170          # 从端 IP，按现场改
export EXCAVATOR_SLAVE_SSH_USER=mundane           # 从端登录名
export EXCAVATOR_SLAVE_DATASET_DIR=/media/mundane/D/real_teleop_v1
./scripts/mount_slave_dataset.sh
```

录完后在从端查看：`ls -la /media/mundane/D/real_teleop_v1/episode_*.hdf5`

**主端**（录制写从盘需要）：

```bash
sudo apt install -y sshfs
ssh-copy-id "${USER}@192.168.31.170"   # 推荐免密
```

网络：从端放行 **TCP 8765**；主端 DDS 组播不通时 `export EXCAVATOR_ROS_PEER_IP=192.168.31.170`。

---

## 1. 从端 `192.168.31.170`（4 个终端）

每个终端先：

```bash
cd ~/Excavator_real_stack
source .venv/bin/activate
export EXCAVATOR_SLAVE_IP=192.168.31.170
```

**不要在从端起 `tb-record-real`。** 顺序建议：1 → 4 → 2 → 3。

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

### 终端 2 — Orbbec

```bash
./scripts/start_orbbec_fpv_camera.sh
```

### 终端 3 — compressed → SHM（供 gateway 写入录制帧）

```bash
./scripts/start_fpv_subscriber_py.sh
```

### 终端 4 — gateway

```bash
./scripts/start_bridge_gateway.sh
```

监听 `0.0.0.0:8765`，转发本机 `127.0.0.1:8766`。

---

## 2. 主端（操作员 PC，2～3 个终端）

每个终端先：

```bash
cd ~/Excavator_real_stack
source .venv/bin/activate
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

## 3. 仿真 vs 真机

| 步骤 | 仿真 | 真机 |
|------|------|------|
| 从端终端 1 | `can-simulation true`，`can-bus-enabled false` | `can-simulation false`，`can-bus-enabled true` |
| 从端 2～4 | 相同 | 相同 |
| 主端录制 | 相同（`192.168.31.170:8765` + SSHFS） | 相同 |

录完后在**从端** QC：

```bash
tb-dataset-qc --dataset-dir /data/real_teleop_v1 --profile real
```

---

## 4. 数据流

```text
主端: 手柄 -> record_host_gamepad_slave_disk.sh
          | TCP 192.168.31.170:8765
          v
从端: gateway -> bridge 127.0.0.1:8766 -> control/CAN
          ^ read_state 图像来自 SHM <- fpv_subscriber <- Orbbec compressed
          |
主端: SSHFS 写 ~/mnt/slave_real_teleop  ==  从端 /data/real_teleop_v1
```

---

## 5. IP / 端口速查

| 用途 | 地址 | 在哪填 |
|------|------|--------|
| 主端录制 / 控车 | `192.168.31.170:8765` | `record_host_gamepad_slave_disk.sh` |
| 从端 bridge | `127.0.0.1:8766` | 仅从端终端 1 |
| 主端 ROS2 peer | `192.168.31.170` | `EXCAVATOR_ROS_PEER_IP` |
| HDF5 | 从端 `/data/real_teleop_v1` | 经 `~/mnt/slave_real_teleop` 写入 |

**不要**在从端用 `192.168.31.170` 连 gateway；**不要**在主端填 `127.0.0.1:8765`（除非整机单机调试）。

---

## 6. 单机联调（一台电脑）

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

## 7. 常见问题

| 现象 | 处理 |
|------|------|
| `请先挂载从端目录` | 主端执行 `./scripts/mount_slave_dataset.sh` |
| HDF5 出现在主端仓库下 | 未挂载或没用 `record_host_gamepad_slave_disk.sh` |
| 主端连不上 8765 | 从端 gateway 已起、`ss -tlnp \| grep 8765`、`ping 192.168.31.170` |
| 主端 rqt 灰屏 | 从端 Orbbec + 终端 3；`ros2 topic hz .../compressed` |
| SSHFS 断开 | 录前检查 `mountpoint ~/mnt/slave_real_teleop`；网络稳定后再录 |
| 从端误开 `tb-record-real` | 关闭；录制只在主端 |
| bridge 日志反复 `client connected/disconnected` | **旧版 gateway** 每请求新建 TCP；更新后 gateway 对 8766 **长连接复用**，仅首次 `upstream bridge connected` |
| `watchdog forced zero command after … ms` | 超过 `heartbeat-timeout-ms` 未收到 `send_action`；加大 `--heartbeat-timeout-ms` 或检查主端录制是否卡住/环路过慢 |

---

## 8. 改 IP

1. `configs/deploy_network.yaml` → `slave.ip`
2. `scripts/excavator_deploy_network.sh` → `EXCAVATOR_SLAVE_IP`
3. `testbed/testbed/configs/teleop_real_v1.yaml` → `data_side_defaults.host.bridge.host`
