# 真机主从分体运行手册

**从端（挖机/相机/落盘）固定 IP：`192.168.31.171`**  
配置来源：`configs/deploy_network.yaml`、`scripts/excavator_deploy_network.sh`

| 角色 | 机器 | 主要进程 |
|------|------|----------|
| 从端 slave | `192.168.31.171` | C++ bridge、gateway、Orbbec、FPV SHM、**tb-record-real（默认落盘）** |
| 主端 host | 操作员 PC（插手柄） | **rqt 看图**；可选主端录 HDF5（`--data-side host`） |

主从 ROS2 同域：`ROS_DOMAIN_ID=42`（`scripts/ros2_fpv_env.sh`）。  
testbed **必须连 gateway `8765`**，不要直连 C++ bridge `8766`。

---

## 0. 两台机器各做一次（首次）

仓库路径以下记为 `~/Excavator_real_stack`，按实际修改。

```bash
cd ~/Excavator_real_stack
git checkout dev_yxc   # 或当前工作分支

# Python 环境
scripts/setup_env.sh venv
source .venv/bin/activate
pip install -e testbed/

# C++ bridge
sudo apt-get install -y build-essential cmake libeigen3-dev
cmake -S bridge -B bridge/build
cmake --build bridge/build -j"$(nproc)"
```

**仅从端 `192.168.31.171`** 需要 Orbbec + ROS2 包：

```bash
# Orbbec（若尚未克隆）
mkdir -p ~/orbbec_ws/src && cd ~/orbbec_ws/src
# git clone https://github.com/orbbec/OrbbecSDK_ROS2.git
cd ~/orbbec_ws
colcon build --symlink-install --packages-select orbbec_camera

# 本仓库 launch 包
ln -sf ~/Excavator_real_stack/ros2_bridge/excavator_ros2_bridge ~/orbbec_ws/src/
colcon build --symlink-install --packages-select excavator_ros2_bridge

sudo apt install -y ros-humble-compressed-image-transport ros-humble-cv-bridge \
  ros-humble-image-transport ros-humble-rqt-image-view
```

**从端**创建数据目录（若无 `/data` 权限，录制时加 `--output-dir ~/data/real_teleop_v1`）：

```bash
sudo mkdir -p /data/real_teleop_v1 && sudo chown "$USER":"$USER" /data/real_teleop_v1
```

网络：主从同一二层网段，从端防火墙放行 **8765/TCP**（gateway）及 ROS2 DDS（或主端设 `EXCAVATOR_ROS_PEER_IP`）。

---

## 1. 从端 `192.168.31.171`（5 个终端）

每个终端先：

```bash
cd ~/Excavator_real_stack
source .venv/bin/activate
export EXCAVATOR_SLAVE_IP=192.168.31.171
```

### 终端 1 — C++ 控制 bridge（仿真，上真 CAN 前保持 false）

```bash
./bridge/build/excavator_real_bridge \
  --host 127.0.0.1 \
  --port 8766 \
  --can-bus-enabled false \
  --can-simulation true \
  --imu-simulation true \
  --create-mapping true
```

### 终端 2 — Orbbec 彩色 compressed

```bash
./scripts/start_orbbec_fpv_camera.sh
```

### 终端 3 — compressed → 共享内存（从端不起 rqt）

```bash
./scripts/start_fpv_subscriber_py.sh
```

### 终端 4 — JSON/TCP 网关（对外 `0.0.0.0:8765`）

```bash
./scripts/start_bridge_gateway.sh
# 等价: EXCAVATOR_GATEWAY_HOST=0.0.0.0 EXCAVATOR_CONTROL_HOST=127.0.0.1
```

### 终端 5 — 录制（默认从端落盘，含关节+图像+手柄动作）

手柄 USB 插在**从端**时：

```bash
tb-record-real \
  --config testbed/configs/teleop_real_v1.yaml \
  --data-side slave \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 \
  --bridge-port 8765 \
  --input joystick
```

手柄插在**主端**时：当前需在主端另起录制进程且从端仍提供 gateway/相机（完整「主控从录」见后续 `host_teleop`）；临时可把 USB 手柄接从端，或单机联调。

---

## 2. 主端（操作员 PC）

```bash
cd ~/Excavator_real_stack
source .venv/bin/activate
export EXCAVATOR_SLAVE_IP=192.168.31.171
# start_host_fpv_rqt.sh 会自动设置:
#   EXCAVATOR_ROS_PEER_IP=192.168.31.171
#   EXCAVATOR_BRIDGE_HOST=192.168.31.171  （仅主端录数据时需要）
```

### 终端 A — FPV 画面（只看不录）

```bash
./scripts/start_host_fpv_rqt.sh
```

验证 DDS 是否通：

```bash
source scripts/ros2_fpv_env.sh
source scripts/excavator_deploy_network.sh
excavator_apply_host_network_defaults
source scripts/ros2_multihost_env.sh
source /opt/ros/humble/setup.bash
ros2 topic hz /camera/color/image_raw/compressed
```

有稳定帧率即主从 ROS2 正常。

### 终端 B（可选）— 主端落盘 HDF5

```bash
tb-record-real \
  --config testbed/configs/teleop_real_v1.yaml \
  --data-side host \
  --backend bridge_tcp \
  --state-reader bridge_tcp \
  --bridge-host 192.168.31.171 \
  --bridge-port 8765 \
  --input joystick
```

默认推荐仍在**从端**录（`--data-side slave`），主端只跑 rqt。

---

## 3. 单机联调（全在一台机器）

不区分物理主从，IP 仍可用 `127.0.0.1`：

```bash
# 终端 1
./bridge/build/excavator_real_bridge --host 127.0.0.1 --port 8766 \
  --can-bus-enabled false --can-simulation true --imu-simulation true --create-mapping true

# 终端 2（无真相机可跳过 2、3，gateway 会用占位图）
./scripts/start_orbbec_fpv_camera.sh
./scripts/start_fpv_subscriber_py.sh

# 终端 3
./scripts/start_bridge_gateway.sh

# 终端 4
source .venv/bin/activate
tb-record-real --config testbed/configs/teleop_real_v1.yaml \
  --data-side slave --backend bridge_tcp --state-reader bridge_tcp \
  --bridge-host 127.0.0.1 --bridge-port 8765 --input joystick
```

---

## 4. 端口与 IP 速查

| 服务 | 地址 | 说明 |
|------|------|------|
| `excavator_real_bridge` | `192.168.31.171:8766` | 仅本机/gateway 内网访问 |
| `gateway_server` | `192.168.31.171:8765` | testbed / 主端 `--bridge-host` |
| Orbbec topic | `/camera/color/image_raw/compressed` | 主从同 `ROS_DOMAIN_ID=42` |
| HDF5（默认） | 从端 `/data/real_teleop_v1` | `--data-side slave` |

环境变量覆盖见 `.env.example`：`EXCAVATOR_SLAVE_IP`、`EXCAVATOR_BRIDGE_HOST`、`EXCAVATOR_ROS_PEER_IP`。

---

## 5. 常见问题

| 现象 | 处理 |
|------|------|
| 主端 rqt 无图 | 两端 `ros2 daemon stop` 后重启；确认 `ROS_DOMAIN_ID=42`；组播不通时主端 `export EXCAVATOR_ROS_PEER_IP=192.168.31.171` |
| `unsupported request type` | 重启从端 C++ bridge（需含 `send_status` 的版本） |
| 主端连不上 8765 | 从端 gateway 是否 `0.0.0.0:8765`、防火墙、ping `192.168.31.171` |
| 从端 `start_fpv_subscriber` 报 Orbbec 路径 | 该脚本只在从端跑；主端用 `start_host_fpv_rqt.sh` |

---

## 6. 改 IP 时改哪些文件

1. `configs/deploy_network.yaml` → `slave.ip`
2. `scripts/excavator_deploy_network.sh` → `EXCAVATOR_SLAVE_IP` 默认值
3. `testbed/testbed/configs/teleop_real_v1.yaml` → `data_side_defaults.host.bridge.host`
