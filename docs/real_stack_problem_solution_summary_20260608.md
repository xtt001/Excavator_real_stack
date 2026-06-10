# 真机问题解决沉淀总览 2026-06-08

本文把当前仓库里已经记录的问题、探索过程和仍需补充的方向整理成一条主线。它不是现场启动手册，具体命令仍以 `docs/host_slave_start_commands.md` 为准。

## 1. 当前链路的角色分工

当前系统不是“通过 SSH 实时控制挖机”。SSH 主要是运维通道：登录 Jetson、启动脚本、查看日志、挂载磁盘、同步文件和做诊断。实时控制数据走 TCP/ROS/CAN 链路。

当前主从链路可以按职责拆成：

```text
主端 joystick/sender
  -> TCP 8770 remote action
从端 receiver / policy_remote
  -> 读状态和 FPV: gateway 8765
  -> 50Hz 控制下发: control pump -> C++ bridge 8766
C++ bridge
  -> can2 低层控制
  -> can3 IMU 状态
Orbbec FPV
  -> ROS compressed topic -> SHM -> gateway -> HDF5 / policy obs
```

各组件地位：

- `ssh slave-jetson`：启动、排查和文件操作入口，不承担 50Hz 实时控制。
- `teleop_remote`：主端读手柄，以 50Hz 把连续 4 维 action 和按键事件发到从端 `8770`。
- `receiver`：从端常驻控制/录制进程，接收主端 action，维护 `armed / recording / go_home / fault / saving` 等运行状态。
- `policy_remote`：把 `remote` 和 `policy` 合并为唯一 action source。未激活时下发人工手柄；按 policy 键后切到模型；切到模型后仍继续消费 record、go-home、status 等离散事件。
- `control pump`：后台线程按 50Hz 重复发送最新 safe action。它解决的是控制 heartbeat 和桥接 watchdog 不被 `read_state`、FPV、HDF5 或 policy 推理拖慢。
- `C++ bridge 8766`：硬件侧桥，直接面对 CAN、IMU、低层控制和安全状态。
- `gateway 8765`：面向 testbed 的聚合入口。`read_state` 从 C++ bridge 取关节状态，并把 FPV SHM 最新帧塞进 observation；普通控制请求可转发到 C++ bridge。
- `FPV SHM`：把 ROS 相机流和控制桥解耦。相机订阅进程写共享内存，gateway 读取最新帧。

从代码和配置看，应该记录的关键事实是：

1. 端口语义必须固定：`8766` 是硬件 control bridge，`8765` 是带 FPV 的 gateway，`8770` 是从端 receiver 接收主端 action。
2. 控制下发和记录/观测是两条路径：控制 pump 直连 `8766`，状态、FPV 和 HDF5 走 `8765`。
3. control pump 只能保证“50Hz 重复最后一次安全动作”，不能保证 policy 每 20ms 都产生新动作。
4. 如果出现不跟手、掉频或 watchdog，应分别记录 `control_send_hz`、`policy_update_hz`、`camera_update_hz`、`action_age_ms`，不要只看总循环 Hz。
5. SSH 相关问题通常是网络、启动、文件或挂载问题；控制延迟问题应优先看 TCP receiver、gateway、bridge、pump、FPV 编码和日志 I/O。

问题 2/3 还需要从历史聊天补充：解耦前的完整链路、解耦后现场验证过程，以及为什么两个 receiver 的设计不适合现场使用。

## 2. 已有问题记录的整理

### Jetson 硬盘无法挂载

已有记录较完整。根因不是 1TB NVMe 丢失，而是设备存在但没有挂载到 `/media/mundane/D`。SSH/VS Code Remote-SSH 属于非图形会话，不会可靠触发桌面自动挂载；原先 `/etc/fstab` 也没有该盘 UUID 的开机挂载项。

短期处理是手动 `mount /dev/disk/by-label/D /media/mundane/D`。长期处理是把 UUID 写入 `/etc/fstab`，使用 `nofail,x-systemd.device-timeout=10`，避免磁盘异常时阻塞系统启动。

### 控制、record 和 FPV 解耦

当前架构已经把几个容易互相拖慢的部分拆开：

- C++ bridge 保持硬件控制原语，不直接链接相机。
- Orbbec 只负责发布 `/camera/color/image_raw/compressed`。
- FPV subscriber 把 compressed 图像解码后写 SHM。
- gateway 在 `read_state` 时注入 FPV。
- control pump 独立 50Hz 下发最新 action。

FPV 采用 compressed/JPEG 和缓存，主要是为了解决数据量和编码开销问题。640x480 raw RGB 单帧约 0.9MB，30Hz 就接近 27MB/s，还会带来 ROS 传输、Python 编码和 HDF5 写盘压力。gateway 里也明确避免对同一 SHM frame 重复 JPEG 编码，因为 recorder 高频调用 `read_state` 时重复编码同一帧会浪费 CPU 并干扰控制。

### 录制、go-home 和按键状态机

已有需求文档把 episode 设计成闭环：

```text
固定初始姿态
-> 挖土
-> swing 到倾倒区
-> dump
-> 人工回到 home 附近
-> go-home 精调
-> go-home 成功后 episode 落盘
```

当前现场流程里：

- 按键 2：开始写 HDF5。
- 按键 3：触发 go-home。
- 按键 4：manual/model control toggle。

go-home 的边界也已经明确：它只做 near-home 微调，不负责从任意姿态大范围自主回 home。实现上使用 home pose、success/center/resume tolerance、死区补偿、单轴调度、滤波、stall boost、wrong-direction 保护等参数。

### policy 速度与泛化失败

已有复盘结论比较清楚：

- 模型动作小不是因为 `action_scale`、safety clip 或下发链路压小，而是 policy 输出本身落在液压死区以下。
- 交叉实验显示，训练 FPV 能产生大动作，live FPV 会让动作变小；因此一号嫌疑是实时 FPV 视觉分布偏移，而不是 qpos/qvel 或控制链路整体损坏。
- 9 条轨迹训练出的 overfit 模型对土面小阴影、光照、bucket/boom 起点和相机视角变化非常敏感。
- 当前 Jetson 上 ACT 新动作约 16-17Hz；control pump 的 50Hz hold 只能维持底层命令连续，不能改变 ACT 的时间语义。

短期更可行路线是重建 20Hz 训练/部署语义，重新设置 `chunk_size`；如果坚持 50Hz 新 policy action，需要把单步推理从约 50ms 降到 20ms 以下。

## 3. IMU 感知原理与当前实现

当前现场 IMU 型号确认为高维感知 `IM-3S31F`。公开检索暂未找到可直接引用的
`IM-3S31F` 在线官方手册，因此本节的设备协议描述以仓库当前 CAN 解析代码和现场
IMU/qvel 日志为准；通用 AHRS 和磁场影响原理参考外部 AHRS 资料。

当前代码把 `IM-3S31F` 当作 CAN AHRS 姿态设备使用：CAN 帧里包含欧拉角、角速度、
加速度、四元数和状态位；欧拉角刻度为 `0.01 deg`，角速度刻度为 `0.1 deg/s`。
当前 CAN 解析没有暴露磁力计 raw 三轴数据，因此现场日志能看到 yaw/roll/pitch、
gyro、accel 和 quaternion，但不能直接用 raw mag 向量判断磁场强度或磁干扰方向。

当前解析逻辑：

- CAN ID 低 3 位是设备地址，兼容 `0..3` 和 `1..4`。
- `cmd=0x00`：欧拉角，roll/pitch 为有符号 `0.01 deg`，yaw 为无符号 `0.01 deg`，再转成 rad。
- `cmd=0x01`：三轴 gyro，单位 `0.1 deg/s`。
- `cmd=0x02`：三轴加速度，当前按 `0.1 m/s^2` 解析。
- `cmd=0x03/0x04`：四元数。
- `cmd=0x05`：设备时间戳和 valid flags。

当前挖机状态映射：

```text
swing  <- imu4 yaw / gyro_z    # IMU4 固定在挖机机体上方
boom   <- imu3 pitch / gyro_y
stick  <- imu2 pitch / gyro_y
bucket <- imu1 pitch / gyro_y
```

也就是说：

- `qpos` 主要来自 IMU 解算后的姿态角，不是液压缸编码器。
- `qvel` 主要来自 gyro 角速度，不是相邻 qpos 差分。
- `qvel_qpos_diff` 只是诊断用，用来检查姿态变化和 gyro 速度是否一致。
- swing 的 `qpos` 当前来自 IMU4 的 yaw/heading，因此比 boom/stick/bucket 的 pitch
  更依赖 AHRS heading 解算质量；IMU4 又装在机体上方，靠近整机钢结构、电驱动液压缸、
  大电流线束和控制器，是最容易受磁场环境影响的一路。

### Bucket 角度跳变和欧拉角 pitch 折叠

2026-06-08 的 bucket 角度问题表现为：收斗过程中，bucket `qpos` 会先变小，
到约 `-55~-57 deg` 后又反向变大；后续修复后又观察到桥进程重启前后，
bucket `policy_deg` 可能从已连续修正后的约 `-25 deg` 回到原始 IMU 分支
约 `-48 deg`。这两个现象本质上是同一类问题：bucket 关节角原先直接使用
IMU 欧拉角 pitch 差值，且连续修正状态只存在于进程内。

当前 raw bucket 角度的来源是：

```text
bucket raw qpos = imu1 pitch - imu2 pitch
```

在异常日志中，bucket 到达最低点附近时，IMU1 的 pitch 已接近 `-90 deg`，
例如 `imu1_y ~= -88.84 deg`、`imu2_y ~= -31.66 deg`，因此 raw bucket
约为 `-57.18 deg`。继续收斗时，真实物理角度还在同一方向运动，但 AHRS
输出的欧拉角 pitch 不再继续小于 `-90 deg`，而是切换到另一组等价欧拉角表示：
pitch 折回，roll/yaw 同时可能变化约 `180 deg`。于是用单个 pitch 数值看，
就会出现“同一方向运动，角度却先减小再增大”的假象。

这里说的“欧拉角解算错误”不是指 IMU 一定坏了，也不是说底层四元数或陀螺仪
一定不连续；更准确地说，是把 AHRS 输出的欧拉角 pitch 当成机械铰点角度使用，
在接近 pitch `+/-90 deg` 的姿态下数学上不成立。欧拉角不是全局唯一表示，
roll-pitch-yaw 形式存在类似万向节锁的奇异区域；AHRS 必须从多组等价姿态里
选一组输出，通常会把 pitch 限制在接近 `[-90, +90] deg` 的范围内。真实机构
越过该区域时，欧拉角分支会折叠或跳到另一支，但机械 bucket 铰点角本身仍应连续。

因此，本次问题不能只靠普通 `[-180, 180]` 折叠或角度 unwrap 解决。因为异常不是
`359 deg -> 0 deg` 这种同一轴环绕，而是 pitch 分支本身在 `+/-90 deg`
附近改变了物理语义。现场日志里可以看到 raw qpos 的变化方向会和 gyro 推断的
运动方向冲突；这就是识别欧拉角分支错误的重要信号。

swing 运动导致的机器振荡也会影响 bucket qpos。原因是 bucket raw qpos 是
`imu1 pitch - imu2 pitch` 的相对量，机身振动、液压冲击、IMU 安装柔性和
AHRS 加速度修正都会同时扰动两个 pitch。若只看 raw Euler pitch，这类振荡可能被
差值放大；当前 qpos 连续保护会抑制明显尖跳和错误分支，但 qvel 仍主要来自 gyro，
所以排查时必须同时看 `qpos policy_deg/policy_rad`、`qpos policy-raw_deg`、
`qpos physical_delta`、`qvel raw_gyro_rad_s` 和 `qvel diff_rad_s`。

当前短期修复策略：

1. 正常姿态下仍用 Euler pitch 作为绝对锚点，避免纯 gyro 积分长期漂移。
2. bucket raw Euler 出现大跳变、接近 pitch 奇异区域，或 raw qpos 变化方向与
   gyro/motion latch 冲突时，改用 gyro 积分推进 bucket qpos。
3. 修正后的 bucket qpos 与 raw Euler 分支距离较大时，保持 alias 状态，不立即
   重新贴回错误 raw 分支。
4. 通过 `EXCAVATOR_BUCKET_QPOS_STATE_FILE` 保存最近 bucket qpos，使 bridge/现场
   stack 重启后能够恢复连续分支，避免停止再启动后回到原始 Euler 分支。
5. IMU/qvel 检查脚本同时打印 `policy_deg` 和 `policy_rad`，其中 `policy_rad`
   是 receiver/policy/HDF5 实际使用的弧度值。

长期更彻底的方案是不再用单个 Euler pitch 差值表示 bucket 铰点角，而是使用
IMU1 和 IMU2 的相对四元数，投影到实测 bucket 铰点轴上得到连续关节角；或者增加
机械编码器/油缸位移传感器。只依赖 raw Euler pitch，无法从数学上保证 bucket
全行程连续单调。

IMU/AHRS 的一般原理：

- gyro 测角速度，短时间响应快，但有 bias、温漂和积分漂移。
- accelerometer 感知重力方向，能约束 roll/pitch，但在震动、冲击和加速度运动中会被污染。
- magnetometer/磁场参考主要用于约束 yaw/heading，因为重力只能提供 roll/pitch 参考，
  不能给出绕竖直方向的绝对角。
- 地磁场本身很弱。钢结构、磁化铁件、电机、大电流电缆、电池、控制器和周围铁磁物体
  都会改变传感器附近的局部磁场，使 AHRS 估计的 yaw/heading 带偏。
- hard/soft iron 标定只能比较可靠地补偿固定、随传感器一起运动且不随时间变化的磁干扰；
  电机转速、电驱液压缸电流、继电器/控制器开关、线束电流变化这类时变磁场，不能指望
  一次静态标定完全消掉。
- 如果 `IM-3S31F` 当前启用了磁场参与的 heading 融合，swing yaw 会被磁场拉回到错误 heading；
  如果改成不使用磁场的 6 轴/gyro yaw，短时间 swing 增量会更符合 gyro，但绝对 yaw 会随
  gyro bias 漂移。两者是“磁场带偏”和“无磁漂移”之间的权衡。

### Swing yaw 与现场磁场干扰

当前挖机是电驱动液压缸，swing IMU 又位于挖机机体上方。这个位置附近既有大块钢结构，
也有电驱动液压缸、驱动器、线束和随动作变化的大电流负载。对磁力计/AHRS heading 来说，
这是典型的复杂磁场环境：一部分是固定 hard/soft iron 偏置，一部分随动作、电流和开关状态
变化。固定偏置可以通过现场姿态下的 hard/soft iron 标定减轻，但随电流变化的磁干扰会让
heading 在作业过程中继续偏移。

因此，当前现场结论应改为：swing 的绝对 `qpos` 不应被视为可靠的全局角度真值。它可以作为
短时间连续性参考，但用于 go-home、phase label、数据 QC 或模型输入时，需要额外保护：

1. 优先对比 `swing qvel gyro_z` 与 `swing qpos diff`。如果 gyro 显示基本静止但 yaw 缓慢漂移，
   或 gyro 显示单向旋转而 yaw 出现反向/跳变，应优先怀疑 heading 被磁场拉偏。
2. 现场日志应额外记录电驱状态、泵/缸动作、swing 动作、电流负载变化与 yaw 偏移的对应关系。
3. 只靠低通滤波不能修复磁场错误；低通只能让错误变慢，不能把错误 heading 变成真实 heading。
4. 短期可考虑把 swing 控制和记录改成“gyro 积分 + 参考点重置/短时连续保护”，降低对磁力计绝对
   yaw 的依赖。
5. 长期更可靠的 swing 方案是机械角度传感器、回转编码器、外部视觉/标志物定位、双天线 GNSS
   heading，或把 heading 传感器移到远离电驱、线束和大钢结构的位置并在整机通电/典型动作下标定。

对本项目的直接影响：

1. swing 当前是最脆弱的 qpos 轴：它依赖 IMU4 yaw，而 yaw/heading 最容易被机体磁场、
   电驱动液压缸和大电流线束影响。
2. boom/stick/bucket 主要用 pitch，受磁场直接影响通常小于 swing，但会受振动、动态加速度、
   安装柔性和 Euler 分支问题影响。
3. 不同挖机、不同安装角度、同一 IMU 重新固定后，绝对 `raw qpos` 都可能有固定偏置或分支差异；
   swing 还会额外受现场磁环境和整机通电状态影响。
4. gyro bias 初始化必须等 4 个 IMU 都 online、valid、fresh 后再做。已有 IMU/qvel 日志显示，
   addr=3 和 addr=2 分别在约 67.9s、107.2s 才上线，bridge 可能过早完成 bias 初始化，
   导致 boom/stick 出现假的静止速度。
5. go-home 控制里使用 qpos/qvel 低通、stable dwell、center/resume hysteresis 是必要的；
   对 swing 还需要最短角误差、磁场异常检测和必要时的 gyro 连续保护。
6. 现场排查 IMU 问题时，应同时记录 `rpy_raw/rpy_rad`、`gyro_dps`、`valid_flags`、`online`、
   `age_ms`、`packet_loss`、`qvel_bridge`、`qvel_qpos_diff`。如果后续能拿到 IM-3S31F 的
   raw mag 输出，也应把 mag 三轴、磁场模长和磁校准状态加入日志。

建议后续记录：

- `IM-3S31F` 的官方手册/PDF、固件版本、输出频率、6轴/9轴模式、是否启用磁场参与 heading 融合。
- 每个 IMU 的安装位置、轴向、线束方向和固定方式照片。
- IMU 上电后全部 online 的时间分布。
- 在整机断电、控制器上电、电驱液压缸动作、swing 动作和不同负载电流下的静止 gyro bias、
  yaw 漂移和 qpos_diff 残差。
- 如果 `IM-3S31F` 支持配置，比较 6轴模式、9轴模式、heading zero、磁校准和 gyro 积分
  连续保护后的 swing qpos 稳定性。

外部参考：

- VectorNav AHRS 基础说明：`https://www.vectornav.com/resources/inertial-navigation-primer/theory-of-operation/theory-ahrs`
- VectorNav 磁场误差来源：`https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-magerrorsources`
- VectorNav hard/soft iron 标定说明：`https://www.vectornav.com/resources/inertial-navigation-primer/specifications--and--error-budgets/specs-hsicalibration`

## 4. 仍待补充

以下内容先保留为后续补充入口：

1. 解耦前控制链路的历史结构、现场症状和改造过程。
2. 两个 receiver 方案为什么不适合现场，以及唯一 `policy_remote` receiver 的决策过程。
3. IMU 固定位置、磁场影响、噪声处理和传感器稳定性的现场实测总结。
4. 不同挖机统一 home/reference pose 的设计，包括物理限位、参考点、可达误差和安装偏置。
5. 更完整的 policy 训练/部署频率策略，包括 20Hz 数据重建、50Hz 推理优化和 ACT chunk 时间尺度。

建议后续每补一项，都按“现象 -> 根因 -> 改法 -> 验证 -> 复用经验”五段写，避免重新堆积成聊天记录。
