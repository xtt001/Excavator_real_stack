# Realworld V1 Plan

本分支 `tx/v1-baseline-realworld` 现在按 real-only 处理：保留热插拔 backend、controller adapter、HDF5 数据闭环和 offline train，不保留旧仿真实现。

## 范围

- 只服务实车 v1 一铲路线，不引入 V2 planner 或高层规划接口。
- 预留低层控制接口 `LowLevelController.send(action, state) -> ControlResult`，后续厂家/CAN/阀控协议只实现这个接口。
- ACT、teleop、recorder 和 dataset loader 只看 normalized 4D action、qpos/qvel、fpv image 和 metadata。
- 当前不使用 hydraulic cylinder data；第一阶段 schema 和训练输入都不包含油缸信号。

## 状态与动作契约

```text
order  = [swing, boom, stick, bucket]
qpos   = calibrated joint angle, rad
qvel   = joint velocity, rad/s
action = normalized teleop command, [-1, 1]
```

第一阶段 HDF5 contract：

- `/observations/qpos`: `(T, 4)` float32, rad
- `/observations/qvel`: `(T, 4)` float32, rad/s
- `/observations/images/fpv`: RGB uint8
- `/action`: guard 后 normalized action
- `/observations/env_state`: real v1 可缺失
- `/diagnostics/*`: raw action、guard reason、controller result

必写 metadata：

```text
is_real=true
platform=real_excavator
qpos_units=rad
qvel_units=rad/s
qpos_source=joint_sensor_calibrated
qvel_source=joint_sensor
hydraulic_cylinder_available=false
action_semantics=normalized_teleop_cmd_v1
```

## 已落地入口

```bash
tb-record-real \
  --config testbed/configs/teleop_real_v1.yaml \
  --backend mock \
  --input zero \
  --num-episodes 1

tb-dataset-qc \
  --dataset-dir data/real_teleop_v1

tb-train \
  --config testbed/configs/act_real_v1.yaml
```

`--backend mock` 用来无实车跑通 record/QC/train；`--backend noop` 会接收 action 但 command 始终为零。真实低层 controller 后续接入时，不应改 ACT、recorder 或数据格式。

## Safety Guard 默认值

- deadman required
- estop/manual override active 时强制零动作
- sensor stale 或 sensor age 超过 `0.20s` 时强制零动作
- normalized action clip: `[-0.20, 0.20]`
- max delta per step: `0.02`

`/action` 永远保存 guard 后的 safe action。原始 joystick action、guard reason、controller ack/fault/timestamp 和 commanded_action 存在 `/diagnostics`。

## 后续实车接入 Checklist

- 确认四轴顺序严格为 `[swing, boom, stick, bucket]`。
- 确认 qpos 零点、正方向、限位和单位 rad。
- 确认 qvel 符号、滤波延迟、采样率和单位 rad/s。
- 确认 fpv 图像分辨率、颜色通道 RGB、时间戳同步方式。
- 确认 deadman、estop、manual override 的真实状态字段和极性。
- 确认 sensor timestamp 来源，保证 timeout 判断使用同一个时间基准。
- 确认低层 controller 的 ack/fault_code 语义和超时策略。
- 确认真实 raw_low_level_command 是否需要写诊断；不要把它作为 ACT 输入。
- 确认实车 episode 开始/结束流程由人工和 CLI 控制，不做软件 reset 假设。
- 确认第一批数据只用于 offline train 和离线预测检查，不做 autonomous command。

## Lower Repo Adapter

底层规划控制库当前主要暴露 C++ `excavator_api`、CAN/SHM 实现和 TCP demo 协议。本 testbed 的第一阶段对齐点是：

```text
testbed action(4) -> SpeedScalarCmd.speed_scalar(8)
first four axes   -> [swing, boom, stick, bucket]
last four axes    -> zero in this fixed-workcell phase
```

真机 ROS/CAN 接入时，只新增或替换 controller adapter；不要让 ROS import 成为整个 testbed 的硬依赖。
