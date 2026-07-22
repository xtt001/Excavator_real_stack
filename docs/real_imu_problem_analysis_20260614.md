# 真机 IMU 问题专项复盘

日期：2026-06-14

本文单独整理近期真机 IMU 相关问题。重点不是复述所有现场启动流程，而是回答：

- 每次 IMU 问题出现时，具体表现是什么。
- 当时定位到的原因是什么。
- 我们尝试了哪些修复。
- 为什么某些修复后录制数据和之前不一样。
- 哪些属于代码/工程实现问题，哪些属于电驱液压挖机上使用 IMU/AHRS 的固有短板。

## 当前结论

这不是一个单独的 IMU bug，而是多层问题叠加：

1. `qpos` 不是 IMU 原始角度，而是给 policy、go-home、HDF5 使用的关节语义量。
2. 早期诊断混淆过弧度/角度、`qpos`/`rpy`、不同 IMU 地址和不同角度分支。
3. swing 使用 IMU4 yaw，受 0/360 分支、go-home 误差计算、磁场和启动初始化影响。
4. bucket 早期使用 IMU1/IMU2 的 Euler pitch 差，接近 pitch `+/-90 deg` 时存在 Euler 分支折叠。
5. 后来 bucket 改成相对四元数 twist 后，数据语义已经改变；如果连续保护或 offset 处理不稳，会生成和旧数据不同的 qpos 分布。
6. 连续保护、状态文件和历史分支能让同一个 raw IMU 姿态输出不同 policy qpos。
7. IMU/AHRS 本身无法等价替代机械编码器；在电驱液压挖机上，振动、动态加速度、安装柔性和磁场都会影响姿态解算。

因此，后续不能把所有 episode 的 `observations/qpos` 默认当成同一语义、同一分布的数据。必须按 converter 版本、bridge 二进制、bucket 表示方式和是否出现 continuity 污染分段处理。

## IMU 到 qpos 的当前链路

现场 IMU 型号确认为高维感知 `IM-3S31F`。当前工程把它当作 CAN AHRS 姿态设备使用：CAN 帧包含 Euler、gyro、accel、quaternion 和 status，但没有暴露 raw magnetometer 三轴数据。已有总结见 [real_stack_problem_solution_summary_20260608.md](/home/mundane/Excavator_real_stack/docs/real_stack_problem_solution_summary_20260608.md:98)。

当前关节语义大致为：

```text
swing  <- imu4 yaw / gyro_z
boom   <- imu3 pitch / gyro_y
stick  <- imu2 pitch - imu3 pitch
bucket <- 当前实验口径：IMU1/IMU2 relative quaternion X-axis twist；
          bucket qvel 使用 imu1 gyro_x - imu2 gyro_y；
          缺 quaternion 时诊断/go-home raw 口径 fallback 到 imu1 roll - imu2 pitch
```

几个关键点：

- `qpos` 是 bridge/converter 输出给上层的关节状态，不是传感器裸值。
- `qpos_raw_imu` 是诊断脚本按 raw IMU 重建的值，发生在 qpos continuity filtering 之前。
- `qpos_folded_imu` 是按折叠后的 `rpy_rad` 重建的诊断值。
- `qvel` 主要来自 gyro，不是简单的 qpos 差分。
- `qvel_qpos_diff` 是诊断项，用于判断姿态变化和 gyro 是否一致。
- `qpos_policy_minus_raw_imu` 是判断 policy qpos 是否脱离 raw IMU 的关键字段。

诊断脚本已经在 summary notes 里说明这些字段含义：[log_imu_qvel_quality.py](/home/mundane/Excavator_real_stack/scripts/log_imu_qvel_quality.py:884)。

## 问题 1：工程师软件 183 deg 与我们 -2.7 不一致

### 现象

2026-06-03，现场对比 IMU 工程师软件和我们的日志：

- 工程师软件看到约 `183 deg`。
- 我们日志里看到类似 `qpos=-2.651` 或用户描述的 `-2.7`。

会话来源：`019e8ccc-66c8-7bf3-9e7d-818da0fd8eac`，本地记录：
`/home/mundane/.codex/sessions/2026/06/03/rollout-2026-06-03T17-24-26-019e8ccc-66c8-7bf3-9e7d-818da0fd8eac.jsonl`。

### 根因

这次主要是读数口径混乱，不是 IMU 立即出错。

- `-2.651` 是 rad，不是 degree；换算后约 `-151.88 deg`。
- `qpos[0]` 是 swing 关节语义量，来自 addr=4 的 Z/yaw。
- 工程师软件的 `183 deg` 可能是另一个 IMU 地址、另一个轴，或 `0..360 deg` 表示。
- 早期 logger 只打印 `rpy_rad`，未保留协议原始 degree，容易混淆。

### 尝试修复

增加 `rpy_raw_deg` 诊断链路：

- CAN parser 保存协议原始 degree。
- SHM/API/bridge JSON 透传。
- `log_imu_qvel_quality.py --verbose-imu` 打印 `raw_deg[x y z]`。

这项修复只增强诊断，不改变 policy/go-home 使用的 `qpos/qvel`。

### 经验

以后对比工程师软件时必须同时确认：

```text
IMU addr
roll/pitch/yaw 哪个轴
degree 还是 rad
0..360 还是 [-180, 180]
看的是 raw IMU 还是 joint qpos
```

归类：代码/诊断口径问题。

## 问题 2：raw 角度 `-0.6 -> +0.6` 和 `+327/-327` 跳变

### 现象

2026-06-05，新日志仍看到正负跳变：

- `imu3 pitch raw=-0.60 -> +0.60`
- yaw raw 在 `+327.67/-327.68` 附近翻转

### 根因

这是两类不同问题：

1. `-0.6/+0.6` 已经出现在 `rpy_raw_deg`，说明不是我们 `[-180, 180]` 转换造成，更像 IMU/AHRS 内部姿态输出在小角度附近的抖动。
2. `+327/-327` 更像协议解析问题：yaw/Z 不应按 signed `int16 * 0.01 deg` 解，而应按 unsigned `0..360 deg` 处理。

### 尝试修复

- yaw 改成 unsigned 解析。
- 用 `rpy_raw_deg` 作为 policy 分支参考。
- 对 qpos 做连续 unwrap。
- 对明显尖跳做单步限速。
- 保留 raw debug，避免把原始证据覆盖掉。

当前代码里，raw yaw 分支和连续化逻辑在 [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:222)。测试覆盖了 yaw 分支和 0/360 crossing：[excavator_converter_test.cpp](/home/mundane/Excavator_real_stack/control/tests/excavator_converter_test.cpp:77)。

### 经验

- `+327/-327` 属于代码解析错误。
- `-0.6/+0.6` 属于 IMU/AHRS 原始输出抖动，代码只能滤波/限速/降低依赖，不能证明机械真的跳了。

归类：协议边界是代码问题；小幅 raw 抖动是传感器/AHRS 短板。

## 问题 3：swing 从 `216 deg` 变成 `-143 deg`，go-home 朝错方向

### 现象

同一 swing 姿态有时显示：

```text
raw_imu swing ~= 216 deg
policy qpos swing ~= -143 deg
```

物理上二者等价，但 go-home 曾经按普通减法计算误差，导致把同一姿态误判成差一整圈。

### 根因

有两个代码问题叠加：

1. swing 是圆周角，go-home 不能用普通 `home_pose_rad - qpos`。
2. bridge 启动初期可能收到 valid 但全零的默认姿态帧，旧逻辑会把 `0 deg` 锁成连续角起点，后续真实 `270 deg` 被 unwrap 到 `-90 deg` 分支。

### 尝试修复

- go-home、诊断和 metadata 使用 swing 最短角误差。
- go-home 滤波前先把 swing feedback 对齐到上一帧分支。
- converter 中跳过 valid 但全零的默认姿态初始化。
- swing policy qpos 尽量对齐 raw yaw 的非负分支。

对应代码和测试：

- [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:31)
- [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:92)
- [excavator_converter_test.cpp](/home/mundane/Excavator_real_stack/control/tests/excavator_converter_test.cpp:99)

### 为什么修复后数据不一样

修复前，同一个物理 swing 姿态可能落在 `-143 deg` 分支；修复后会落到 `216 deg` 或 `0..360 deg` 分支。物理一致，但数值分布变了。因此旧数据和新数据直接混训，会让模型看到同一个位置对应两套数值。

归类：代码问题。

## 问题 4：bucket 到 `-55~-57 deg` 后又反向

### 现象

2026-06-08，收斗过程中 bucket 角度先变小，到约 `-55~-57 deg` 后又变大。用户同时观察到 swing 运动产生机器振荡，bucket 角度也振荡。

会话来源：`019ea696-4ea5-7812-9fd2-7f9ffbc935e2`，本地记录：
`/home/mundane/.codex/sessions/2026/06/08/rollout-2026-06-08T17-35-28-019ea696-4ea5-7812-9fd2-7f9ffbc935e2.jsonl`。

### 根因

旧 bucket raw qpos 为：

```text
bucket = imu1 pitch - imu2 pitch
```

当 IMU1 pitch 接近 `-90 deg` 时，Euler pitch 发生分支折叠。真实机械仍在同方向运动，但 AHRS 输出的 Euler pitch 选择了另一组等价姿态，导致单独看 pitch 时“变小再变大”。

这不是普通 `359 -> 0 deg` 的 wrap 问题，而是 Euler pitch 在奇异区改变了物理语义。已有复盘见 [real_stack_problem_solution_summary_20260608.md](/home/mundane/Excavator_real_stack/docs/real_stack_problem_solution_summary_20260608.md:136)。

### 尝试修复

第一阶段：

- 对 bucket policy qpos 做连续保护。
- 当 raw qpos 跳变方向与 gyro/motion 方向冲突时，按 gyro 小步推进。
- 限制 bucket 单步最大变化，避免尖跳直接进入 policy。

第二阶段：

- 使用 IMU1/IMU2 relative quaternion 绕 Y 轴 twist 作为 bucket 角。
- 用 `EXCAVATOR_BUCKET_QUATERNION_OFFSET_RAD` 把新 bucket 角映射回旧 policy/home qpos frame。

第三阶段（2026-06-17 当前实验）：

- bucket IMU 物理绕 Z 轴旋转 90° 后，bucket qpos 保留 quaternion 表示，但改取
  relative quaternion 的 X 轴 twist。
- 清理 legacy offset、bucket quaternion raw jump 方向守卫和 bucket 专用单步限幅，
  直接观察新安装姿态下的原始 quaternion 主值分支。
- bucket qvel 的 IMU1 项改为 `gyro_x`，即 `imu1 gyro_x - imu2 gyro_y`。
- 统一脚本不再导出 `EXCAVATOR_BUCKET_QUATERNION_OFFSET_RAD`。

当前实现位置：

- [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:120)
- [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:141)
- [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:261)

测试覆盖：

- Euler 分支折叠时用 quaternion 锚定：[excavator_converter_test.cpp](/home/mundane/Excavator_real_stack/control/tests/excavator_converter_test.cpp:154)
- raw jump 与 gyro 方向冲突时限速：[excavator_converter_test.cpp](/home/mundane/Excavator_real_stack/control/tests/excavator_converter_test.cpp:262)

### 为什么修复后数据不一样

因为 bucket 的观测函数变了：

```text
旧语义：imu1 Euler pitch - imu2 Euler pitch
第二阶段语义：relative quaternion twist around Y + legacy policy offset + continuity guard
当前实验语义：relative quaternion twist around X，不加 legacy offset，不做 bucket 专用 guard
```

即使 home 附近通过 offset 对齐，全行程也不保证和旧 Euler 差值同分布。特别是分支、offset、twist axis 和 continuity guard 都会改变中间轨迹。

归类：Euler 奇异区是传感器/AHRS 表示短板；用 Euler pitch 差表示机械 bucket 是代码/建模问题。

## 问题 5：bucket 状态文件污染，`-0.61` 位置显示 `+0.197`

### 现象

2026-06-09，过去 go-home 时 bucket 回到 `-0.61 rad` 附近，但一次测试中相同位置显示 `+0.197 rad`，go-home 回到了另一个位置。

### 根因

raw IMU 反算的 bucket 并没有变成 `+0.197`，仍在约 `-0.82 rad` 附近。变的是 bridge 输出给 policy/go-home 的连续 qpos。

从端状态文件：

```text
/tmp/excavator_slave_stack/bucket_qpos_state.txt
bucket_qpos_v1 1 0.19718735034550375 1
```

bridge 启动时恢复了这个旧 bucket 连续分支状态，导致上层看到 `+0.197 rad`。

会话来源：`019eaa75-deaf-7ca1-8941-15c37c104a1b`，本地记录：
`/home/mundane/.codex/sessions/2026/06/09/rollout-2026-06-09T11-38-31-019eaa75-deaf-7ca1-8941-15c37c104a1b.jsonl`。

### 尝试修复

- 状态文件格式升级为 v2。
- 加 boot id，拒绝旧 v1 文件。
- 默认状态文件从固定 `/tmp/.../bucket_qpos_state.txt` 改成每次 run 的日志目录下文件。
- 删除当前错误状态文件并重启/重建现场链路。

### 为什么修复后数据不一样

状态文件本来是为了让 bridge 重启后保持 bucket 连续分支，但固定路径导致跨启动、跨测试污染。清理后，同一个 raw IMU 会重新从当前姿态初始化，而不是恢复旧 run 的 qpos。因此修复前后同一物理位置可能数值不同。

归类：明确代码/运行状态管理问题。

## 问题 6：bucket quaternion 修复后 episode_47..65 大跳变

### 现象

2026-06-09，按从端 bridge 二进制时间分界后，发现：

- 修改前 `episode_39..45` bucket qpos 相邻步最大跳变小于 `0.09 rad`。
- 修改后 `episode_47..65` 每条完整 episode 都有 bucket qpos 大跳变，典型 `2..5 rad`，发生在 20ms 相邻采样之间。
- 这些跳变发生时 qvel/action 连续，不是真实机械动作。

会话来源：`019eac09-10b0-7fe0-9968-1139abff57b0`，本地记录：
`/home/mundane/.codex/sessions/2026/06/09/rollout-2026-06-09T18-58-55-019eac09-10b0-7fe0-9968-1139abff57b0.jsonl`。

### 根因

当时的 quaternion bucket 实现不只是换了公式，还改变了分支处理路径。四元数角本身也有 `pi/-pi` alias，若没有和 gyro 方向、上一帧分支、静止状态一起约束，就会把连续值拉到另一个等价分支。

后来测试里已经增加了针对 alias 的约束，例如：

- [excavator_converter_test.cpp](/home/mundane/Excavator_real_stack/control/tests/excavator_converter_test.cpp:291)
- [excavator_converter_test.cpp](/home/mundane/Excavator_real_stack/control/tests/excavator_converter_test.cpp:344)

### 尝试修复

- bucket quaternion 角通过 `bucket_quaternion_delta_plausible()` 检查。
- 与 gyro 方向冲突、超过单步限幅时，只按 gyro-sized step 推进。
- 对 pi 分支 crossing 增加 zero-gyro 不 unwrap 的测试。

当前 guard 逻辑见 [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:43) 和 [excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:286)。

### 为什么这些数据不能和旧数据混用

这批 episode 不是单纯 offset 不同，而是存在非物理大跳。它们的 qpos 既不是旧 Euler 语义，也不是稳定的新 quaternion 语义。用于训练时应隔离或只保留可修复窗口。

归类：代码回归/新表示实现不稳。

## 问题 7：同一物理位置 bucket 旧是 `-0.58`，新变 `-1.23`

### 现象

2026-06-11，一次日志里同一个物理位置，用户观察到：

```text
旧位置约 -0.58 rad
新 qpos 到 -1.23 rad 才像旧位置
```

同时日志显示：

```text
qpos = -1.2358
qpos_raw_imu = -0.5317
policy - raw ~= -0.704 rad
```

会话来源：`019eb65c-682d-7433-980a-60cf2a022867`，本地记录：
`/home/mundane/.codex/sessions/2026/06/11/rollout-2026-06-11T19-06-09-019eb65c-682d-7433-980a-60cf2a022867.jsonl`。

### 根因

这不是 assist 代码直接改了 qpos，而是 bucket continuity 内部状态漂移。

当 converter 认为 raw quaternion 位置跳变太大、或方向和 gyro 冲突时，会拒绝 raw 绝对姿态，改用 gyro 小步积分。这能防止分支跳，但副作用是：如果内部连续状态已经偏离 raw quaternion，又没有在静止时重新锚定，同一个物理姿态就会因为历史状态不同输出不同 `qpos`。

### 尝试解决方向

当时建议：

- 短期重启 bridge，让 continuity 初值贴近当前 raw quaternion。
- 更稳的修法是在 bucket quaternion 有效且静止若干帧时，自动 re-anchor 到 raw quaternion。
- re-anchor 需要同时检查 gyro、qpos diff、raw-policy delta，避免把错误 raw 分支又贴回来。

### 为什么这类问题反复出现

我们在 bucket 上一直做权衡：

- 贴 raw quaternion：绝对姿态更可信，但可能被分支 alias 瞬间拉走。
- 用 gyro/continuity：短期连续，但会积累历史偏置。
- 保存状态文件：跨重启连续，但可能污染新 run。
- 每次重启重锚定：避免旧状态污染，但可能在错误 raw 分支上初始化。

只靠 IMU/AHRS，没有机械绝对编码器时，这个权衡无法完全消失。

归类：当前 continuity 设计问题，同时受 IMU 姿态分支短板触发。

## 问题 8：swing yaw 的磁场与电驱现场短板

### 现象

用户确认：

- IMU 型号是高维感知 `IM-3S31F`。
- swing IMU 位于挖机机体上方。
- 挖机是电驱动液压缸。
- 机体附近磁场复杂，swing IMU 不准。

会话来源：`019ea696-4ea5-7812-9fd2-7f9ffbc935e2`，本地记录：
`/home/mundane/.codex/sessions/2026/06/08/rollout-2026-06-08T17-35-28-019ea696-4ea5-7812-9fd2-7f9ffbc935e2.jsonl`。

### 根因

swing yaw/heading 和 boom/stick/bucket pitch 不一样。重力可以约束 roll/pitch，但 yaw/heading 通常需要磁场或 gyro 积分：

- 如果启用磁融合，yaw 会被现场局部磁场带偏。
- 如果禁用磁融合，只靠 gyro，短时间连续性较好，但绝对 yaw 会漂移。

电驱液压挖机附近有钢结构、电流线束、驱动器、电机负载和时变磁场。一次静态 hard/soft iron 标定无法消掉随电流变化的干扰。已有总结见 [real_stack_problem_solution_summary_20260608.md](/home/mundane/Excavator_real_stack/docs/real_stack_problem_solution_summary_20260608.md:193) 和 [real_stack_problem_solution_summary_20260608.md](/home/mundane/Excavator_real_stack/docs/real_stack_problem_solution_summary_20260608.md:208)。

### 解决方向

短期：

- swing 控制使用最短角误差。
- 记录 `qpos diff` 与 `gyro_z` 的一致性。
- 对 yaw 漂移和反向跳变做异常检测。
- 必要时用 gyro 短时连续保护，而不是盲信绝对 yaw。

长期：

- 机械回转编码器。
- 油缸/关节位移传感器。
- 外部视觉/标志物定位。
- 双天线 GNSS heading。
- 把 heading 传感器移到远离电驱、线束和大钢结构的位置，并在整机通电/典型动作下标定。

归类：电驱液压挖机上使用磁融合 AHRS 的固有短板。代码只能检测、降权、保护，不能把受干扰 heading 变成真实 heading。

## 为什么修复后录制数据和之前不一样

### 1. 同一姿态的角度分支变了

例如 swing：

```text
216 deg == -144 deg  # 物理等价
```

但训练数据里一个是 `3.77 rad`，另一个是 `-2.51 rad`。如果模型只看数值，它们不是同一个输入。

### 2. bucket 的观测函数变了

旧：

```text
bucket = imu1 Euler pitch - imu2 Euler pitch
```

新：

```text
第二阶段：bucket = twist(relative_quaternion(imu2 -> imu1), Y axis) + legacy_offset
当前实验：bucket = twist(relative_quaternion(imu2 -> imu1), X axis)
```

这两个函数在 home 附近可以通过 offset 对齐，但全行程不保证同分布。

### 3. continuity guard 引入历史状态

`qpos` 不再只是当前 raw IMU 的函数，也依赖：

```text
上一帧 qpos
gyro 方向
单步限幅
是否有 quaternion
是否恢复过状态文件
bridge 启动时第一帧姿态
```

因此，同一个物理姿态可能因历史不同输出不同 `qpos`。

### 4. 过渡版本污染

有些 episode 录在：

- 旧 bridge 仍在运行但磁盘二进制已更新。
- 从端源码未同步。
- 修改后 bridge 未重启。
- 状态文件恢复了旧分支。
- 新 quaternion 实现还存在 branch jump 回归。

这些 episode 不能按同一版本语义进入训练。

## 代码问题与固有短板分界

### 明确代码/工程问题

- `qpos`、`rpy`、单位、地址、轴名的诊断口径不清。
- yaw/Z signed/unsigned 协议解析错误。
- swing 角度分支和 go-home 普通减法。
- valid 全零初始化锁错分支。
- swing qvel 符号、底层 motor command 符号错误。
- bucket 状态文件固定路径导致跨 run 污染。
- quaternion bucket 初版引入 2-5 rad 分支跳变。
- bucket continuity 没有静止 re-anchor，产生历史偏置。
- 主从代码同步、重编译、重启顺序不一致导致现场跑旧逻辑。

### IMU/AHRS 和电驱液压现场固有短板

- IMU 姿态不是机械编码器。
- Euler pitch 接近 `+/-90 deg` 时不适合作为 bucket 铰点角。
- 振动、液压冲击、动态加速度和安装柔性会扰动 pitch。
- swing yaw/heading 对磁场敏感。
- 电驱液压缸、大电流线束、驱动器和钢结构会导致时变磁干扰。
- gyro 有 bias、温漂和积分漂移。
- 当前没有 raw magnetometer 日志，无法直接量化磁干扰。

## 数据使用建议

### 必须按 IMU/qpos 语义分段

至少分为：

1. 早期 Euler pitch 差 bucket。
2. yaw unsigned/unwrap 修复前。
3. swing 最短角和 raw yaw branch 修复后。
4. bucket quaternion 初版。
5. bucket state-file 修复后。
6. 出现 continuity 偏置污染的 run。

### 不建议直接混用的数据

- swing 同一姿态有 `216 deg` 和 `-144 deg` 两套分支的数据。
- bucket 从 Euler 差切到 quaternion twist 的过渡数据。
- `episode_47..65` 这类已确认存在 bucket 2-5 rad 大跳的完整 episode。
- `qpos_policy_minus_raw_imu` 长时间偏大且无解释的片段。
- bridge 运行旧二进制、源码已同步但未重启的测试片段。

### 可以保留的用途

- 异常日志：保留作问题证据。
- raw IMU 与 qpos 对齐窗口：可用于修复/重建。
- FPV 正常但 qpos 污染的 episode：可用于视觉分布、FPV transform 或非监督视觉用途，但不要直接监督 action/qpos。
- 手动动作和 go-home 失败片段：可用于动作死区、符号、延迟和安全逻辑复盘。

## 后续建议

### 短期工程修复

1. 给 bucket continuity 增加静止 re-anchor：
   - quaternion valid。
   - gyro 小于阈值。
   - raw-policy delta 稳定。
   - 连续若干帧满足。
   - re-anchor 时保留 branch alias 检查。
2. 给每条 HDF5 写入 qpos 语义版本 metadata：
   - converter git sha。
   - bridge binary mtime/hash。
   - bucket mode：Euler / quaternion / continuity version。
   - yaw branch mode。
   - bucket offset。
3. QC 增加硬门禁：
   - bucket adjacent qpos jump。
   - qpos_policy_minus_raw_imu p95/max。
   - qvel_qpos_diff residual。
   - swing branch alias。
   - IMU online/valid/fresh。
4. 训练 manifest 按 qpos 语义版本分组，不允许随机混合。

### 中期传感器策略

1. 确认 IM-3S31F 是否启用磁融合、是否支持 6 轴模式和 raw mag 输出。
2. 做整机通电、不同液压动作、电流负载下的 yaw drift 测试。
3. 对 bucket 评估相对四元数 twist 的稳定性，必要时标定真实铰点轴，而不是固定用 IMU 局部 Y 轴。
4. 对 swing 降低绝对 yaw 权重，更多依赖短时 gyro 连续性和外部参考。

### 长期方案

IMU/AHRS 适合做姿态辅助和短时连续性参考，但不适合单独承担所有关节绝对角：

- swing：优先机械回转编码器或外部 heading 参考。
- bucket：优先机械编码器/油缸位移传感器，或至少标定相对四元数真实铰点轴。
- boom/stick：pitch 相对更稳定，但仍要防振动、动态加速度和安装柔性。

## 关键证据索引

- 读数口径差异：`/home/mundane/.codex/sessions/2026/06/03/rollout-2026-06-03T17-24-26-019e8ccc-66c8-7bf3-9e7d-818da0fd8eac.jsonl`
- bucket `-55 deg` 反向：`/home/mundane/.codex/sessions/2026/06/08/rollout-2026-06-08T17-35-28-019ea696-4ea5-7812-9fd2-7f9ffbc935e2.jsonl`
- bucket 状态文件污染：`/home/mundane/.codex/sessions/2026/06/09/rollout-2026-06-09T11-38-31-019eaa75-deaf-7ca1-8941-15c37c104a1b.jsonl`
- 修改后 episode 大跳变：`/home/mundane/.codex/sessions/2026/06/09/rollout-2026-06-09T18-58-55-019eac09-10b0-7fe0-9968-1139abff57b0.jsonl`
- 同物理位置 `-0.58` 到 `-1.23`：`/home/mundane/.codex/sessions/2026/06/11/rollout-2026-06-11T19-06-09-019eb65c-682d-7433-980a-60cf2a022867.jsonl`
- IMU 原理与磁场影响沉淀：[real_stack_problem_solution_summary_20260608.md](/home/mundane/Excavator_real_stack/docs/real_stack_problem_solution_summary_20260608.md:98)
- 当前 converter 实现：[excavator_converter.cpp](/home/mundane/Excavator_real_stack/control/src/excavator/excavator_converter.cpp:1)
- 当前 converter 测试：[excavator_converter_test.cpp](/home/mundane/Excavator_real_stack/control/tests/excavator_converter_test.cpp:77)
- 当前 IMU/qvel 诊断字段：[log_imu_qvel_quality.py](/home/mundane/Excavator_real_stack/scripts/log_imu_qvel_quality.py:884)
