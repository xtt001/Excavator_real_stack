# ros2_bridge

挖机 **control** 与 **Orbbec 相机** 解耦：不修改 `bridge/src/excavator_real_bridge.cpp`。

## 架构

```text
testbed (tb-record-real, bridge_tcp :8765)
    -> excavator_bridge_gateway (Python, 可选)
         |-- read_state: 关节/状态转发 + FPV 来自 SHM
         +-- send_action/reset/... -> excavator_real_bridge :8766

进程1: ros2 launch ... orbbec_fpv_camera.launch.py   # 仅相机
进程2: scripts/start_fpv_subscriber_py.sh            # ROS compressed -> SHM
进程3: excavator_real_bridge --port 8766             # 仅 control（原版 C++）
进程4: scripts/start_bridge_gateway.sh               # 对外 :8765（要相机时启用）
```

无相机时：testbed 可直接连 `excavator_real_bridge:8765`（占位图），**不必起 gateway**。

## 依赖

从机 Orbbec 驱动位于 **`~/orbbec_ws/src/OrbbecSDK_ROS2`**（colcon 后 ROS 包名 `orbbec_camera`）。

```bash
# 1) 构建 Orbbec（从机，仅需一次）
cd ~/orbbec_ws/src
# git clone https://github.com/orbbec/OrbbecSDK_ROS2.git  # 若尚未克隆
cd ~/orbbec_ws
colcon build --symlink-install --packages-select orbbec_camera

# 2) 将本仓库 launch 包放入 colcon 工作空间（可与 orbbec_ws 共用）
ln -sf ~/Excavator_real_stack/ros2_bridge/excavator_ros2_bridge ~/orbbec_ws/src/
cd ~/orbbec_ws   # 或单独的 EXCAVATOR_ROS_WS
colcon build --symlink-install --packages-select excavator_ros2_bridge

# 3) 环境
source ~/Excavator_real_stack/scripts/source_ros_stack.sh
# 或: export EXCAVATOR_ORBBEC_WS=~/orbbec_ws EXCAVATOR_ROS_WS=~/orbbec_ws

source .venv/bin/activate   # testbed（可选）
sudo apt install -y ros-${ROS_DISTRO}-compressed-image-transport ros-${ROS_DISTRO}-cv-bridge
pip install -r requirements.txt
```

## 主从分工（相机 / 可视化 / 落盘）

```text
从端: Orbbec 发布 /camera/color/image_raw/compressed
      -> start_fpv_subscriber_py.sh 写 SHM -> gateway -> tb-record-real --data-side slave
主端: start_host_fpv_rqt.sh 仅订阅同一 compressed，本机 rqt（不录 HDF5）
```

FPV 脚本默认 `EXCAVATOR_ROS2_MULTIHOST=1`、`EXCAVATOR_ROS_DOMAIN_ID=42`（`scripts/ros2_fpv_env.sh`）。
RMW 自动选择：已安装 Cyclone 则用 `rmw_cyclonedds_cpp`，否则用系统默认 **Fast DDS**（主从须一致）。

```bash
sudo apt install -y ros-humble-image-transport \
  ros-humble-compressed-image-transport ros-humble-rqt-image-view
# 可选，与从端统一用 Cyclone：ros-humble-rmw-cyclonedds-cpp
```

**从端**：

```bash
./scripts/start_orbbec_fpv_camera.sh
./scripts/start_fpv_subscriber_py.sh          # 仅 SHM，无 rqt
# 录制: tb-record-real --data-side slave ...
```

**主端**：

```bash
# 从端 IP 固定 192.168.31.171；组播不通时脚本已默认 EXCAVATOR_ROS_PEER_IP
./scripts/start_host_fpv_rqt.sh               # 只订 compressed + rqt
```

完整主从命令见 **`docs/realworld_host_slave_runbook.md`**。

验证：主端 `ros2 topic hz /camera/color/image_raw/compressed` 有帧率即通。

## 运行（四终端示例）

```bash
# 1. 控制 bridge（原版，未改 cpp）
./bridge/build/excavator_real_bridge --port 8766 --can-bus-enabled false

# 2. Orbbec 相机（OrbbecSDK_ROS2 @ ~/orbbec_ws）
./scripts/start_orbbec_fpv_camera.sh
# 等价: source scripts/source_ros_stack.sh && ros2 launch excavator_ros2_bridge orbbec_fpv_camera.launch.py

# 3. 从端 compressed -> SHM（主端 rqt 用 start_host_fpv_rqt.sh）
./scripts/start_fpv_subscriber_py.sh

# 4. 网关（testbed 连 8765）
./scripts/start_bridge_gateway.sh

# 5. 录制
tb-record-real --backend bridge_tcp --bridge-port 8765 --input joystick ...
```

## C++ 可选组件

`excavator_ros2_bridge` 包内另有 C++ `fpv_image_subscriber`（与 Python 订阅二选一）：

```bash
cd ~/orbbec_ws && colcon build --packages-select excavator_ros2_bridge
```

## 说明

| 项 | 行为 |
|----|------|
| `excavator_real_bridge.cpp` | **保持仓库原版**，不链接相机 |
| 时间戳 | 图像用 `header.stamp`；关节来自 control snapshot |
| 仿真 | gateway `--fpv-source auto` 无 SHM 时用 placeholder |
| `send_status` | 网关转发到 C++ bridge → `applyStatusToggleMask` |

## 目录

```text
ros2_bridge/
  fpv_frame_store/           # SHM 库（C++ 订阅节点可选使用）
  excavator_ros2_bridge/     # Orbbec launch + 可选 C++ 订阅
  excavator_bridge_gateway/  # Python 网关 + Python 订阅（推荐）
```
