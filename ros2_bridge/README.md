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

```bash
source /home/yxc/ros2_ws/install/setup.bash   # orbbec_camera
source .venv/bin/activate                       # testbed

sudo apt install -y ros-${ROS_DISTRO}-compressed-image-transport ros-${ROS_DISTRO}-cv-bridge
pip install -r requirements.txt
```

## 运行（四终端示例）

```bash
# 1. 控制 bridge（原版，未改 cpp）
./bridge/build/excavator_real_bridge --port 8766 --can-bus-enabled false

# 2. Orbbec 相机
source /home/yxc/ros2_ws/install/setup.bash
ros2 launch excavator_ros2_bridge orbbec_fpv_camera.launch.py

# 3. 图像订阅 -> SHM + rqt 可视化（默认开启 rqt_image_view）
./scripts/start_fpv_subscriber_py.sh
# 等价: ros2 launch excavator_ros2_bridge fpv_subscriber_with_rqt.launch.py
# 仅订阅、不弹窗: ... fpv_subscriber_with_rqt.launch.py use_rqt:=false

# 4. 网关（testbed 连 8765）
./scripts/start_bridge_gateway.sh

# 5. 录制
tb-record-real --backend bridge_tcp --bridge-port 8765 --input joystick ...
```

## C++ 可选组件

`excavator_ros2_bridge` 包内另有 C++ `fpv_image_subscriber`（与 Python 订阅二选一）：

```bash
cd /home/yxc/ros2_ws && colcon build --packages-select excavator_ros2_bridge
```

## 说明

| 项 | 行为 |
|----|------|
| `excavator_real_bridge.cpp` | **保持仓库原版**，不链接相机 |
| 时间戳 | 图像用 `header.stamp`；关节来自 control snapshot |
| 仿真 | gateway `--fpv-source auto` 无 SHM 时用 placeholder |
| `send_status` | 网关原样转发到 control bridge；若 control bridge 无此接口，需用 `bridge_mock` 或另行给 control 加 status 支持 |

## 目录

```text
ros2_bridge/
  fpv_frame_store/           # SHM 库（C++ 订阅节点可选使用）
  excavator_ros2_bridge/     # Orbbec launch + 可选 C++ 订阅
  excavator_bridge_gateway/  # Python 网关 + Python 订阅（推荐）
```
