# Bucket IMU Quaternion 排查进度 2026-07-06

本文记录 2026-07-06 对 bucket qpos 跳变问题的排查过程、证据和当前结论。它是阶段性排查记录，后续设计会在新的验证结果上继续更新。

## 背景现象

真机录制中出现 bucket qpos 单帧大跳，触发在线 QC 的 `qpos_jump`，episode 被保存到 `failed/`。用户在操作时还观察到几秒钟控制阻塞；后续确认这段阻塞发生在失败判定后的 failed HDF5 保存/同步阶段，qpos 跳变已经先于阻塞出现。

典型失败文件：

- `/media/mundane/EXTERNAL_USB/real_teleop_v1/failed/episode_72_failed_20260706T053004.295489Z.hdf5`
- HDF5 已写入 629 steps，约 `20.50 s`
- 触发失败的下一帧未写入 HDF5；按在线日志对齐，bucket qpos 从约 `-0.818 rad` 跳到 `-1.399 rad`，单帧约 `-0.581 rad`

当时 IMU0 age 很新鲜，约 `7-19 ms`。进一步检查发现 IMU0 quaternion 在该帧附近本身出现约 `30 deg` 级姿态跳变，而 IMU1 同期平稳。

## 第一轮证据：旧日志中的 IMU0 跳变

旧 IMU/qvel 日志：

- `/media/mundane/EXTERNAL_USB/imu_qvel_tests/imu_qvel_20260706T052825.478864Z_summary.json`
- 对应 JSONL：`imu_qvel_20260706T052825.478864Z.jsonl`

关键样本：

| sample | IMU0 rpy_raw_deg | bucket qpos |
| --- | --- | ---: |
| 3517 | `[-36.43, -55.91, 110.72]` | `-46.88 deg` |
| 3518 | `[-167.36, -77.33, 161.23]` | `-80.19 deg` |

这次现象属于 IMU0 姿态候选突变，而非简单的 `2pi` 数值折返。IMU0 quaternion geodesic delta 约 `30.5 deg`，IMU1 约 `0.93 deg`。bucket qpos 使用的是 IMU0/IMU1 relative quaternion，因此 IMU0 的 AHRS 姿态候选直接影响 bucket。

## 半包同步与原始 CAN 记录

排查过程中发现 IMU quaternion 由两个 CAN 半包组成：

- `cmd 0x03`: `q0/q1`
- `cmd 0x04`: `q2/q3`

原代码只检查两个半包是否曾经收到，没有检查它们是否属于同一时间窗口。因此增加了半包同步约束：

- `kImuQuaternionHalfSyncWindowNs = 5 ms`
- 只有 `q0/q1` 与 `q2/q3` 接收时间差在窗口内时，`valid_quaternion` 才置为 true
- 增加 parser 单元测试覆盖同步/不同步半包

同时从端 `slave_real_stack.sh` 增加后台 raw CAN 记录：

- 默认记录 `EXCAVATOR_IMU_IF`，现场为 `can6`
- 日志路径在每轮 `artifacts/slave_stack/<timestamp>/canraw.log`

这个修改的目的是让后续每轮都能从 CAN 原始帧复核 quaternion 是否已经在上游跳变。

## 第二轮证据：双 chart 修复前的 bucket 200 度翻转

用户做 bucket 端到端往返循环后，分析日志：

- summary: `/media/mundane/EXTERNAL_USB/imu_qvel_tests/imu_qvel_20260706T064948.400248Z_summary.json`
- JSONL: `/media/mundane/EXTERNAL_USB/imu_qvel_tests/imu_qvel_20260706T064948.400248Z.jsonl`
- raw CAN: `/media/mundane/D/Excavator_real_stack/artifacts/slave_stack/20260706_144930/canraw.log`

统计结果：

- rows: `12917`
- duration: `325.16 s`
- bucket qpos span: `267.94 deg`
- bucket qpos_diff max: `116.87 rad/s`

其中有三处典型 `200 deg` 级翻转，例如：

- sample 8875: bucket `-96.39 deg -> +108.48 deg`
- sample 6249: bucket `-159.46 deg -> +43.43 deg`
- sample 12559: bucket `-139.21 deg -> +37.25 deg`

进一步检查 relative quaternion 发现，这类翻转发生时 `(relative.w, relative.y)` 同时接近 0。旧 bucket scalar 公式是：

```text
theta = 2 * atan2(relative.y, relative.w) + fixed_offset
```

当这两个输入接近原点时，`atan2` 的相位会变得病态。四元数本身可以连续，但投影到 `(w, y)` 这个二维 chart 后会绕原点，标量角出现大跳。

## 已做的临时修复：bucket 双 chart phase tracker

针对上述 chart 病态，bucket 解算增加了第二个 chart：

- primary chart: 继续使用旧的 `(w, y)`，保持历史 calibrated 坐标
- secondary chart: 使用 `(-2 * atan2(relative.x, relative.z))`
- 当 primary strength `hypot(w, y) < 0.35` 且 secondary 更稳定时，使用 secondary 的相位增量推进连续 bucket qpos

这个修复仍然只用 quaternion candidate，不用 gyro 积分生成 qpos。gyro 仍只作为 qvel/诊断来源。

本地离线重放 `20260706T064948` 日志后：

- 旧 bridge qpos 最大跳变：`204.86 deg`
- 新 tracker 重建后最大跳变：`37.98 deg`

剩下的 `29-38 deg` 级跳变对应 IMU0 quaternion 本体突跳，来源已经转向 AHRS 输出本身。

## 第三轮证据：双 chart 修复后的最新测试

用户重新测试一轮，最新日志为：

- summary: `/media/mundane/EXTERNAL_USB/imu_qvel_tests/imu_qvel_20260706T072812.758826Z_summary.json`
- JSONL: `/media/mundane/EXTERNAL_USB/imu_qvel_tests/imu_qvel_20260706T072812.758826Z.jsonl`
- raw CAN: `/media/mundane/D/Excavator_real_stack/artifacts/slave_stack/20260706_152804/canraw.log`

summary 统计：

- rows: `7788`
- duration: `176.50 s`
- read_errors: `19`
- bucket qpos span: `180.20 deg`
- bucket qpos_diff max: `20.88 rad/s`
- bucket qpos 单帧跳变 `>10 deg`: `3` 次
- bucket qpos 单帧最大跳变: `28.84 deg`

双 chart 修复已在这轮生效。旧 primary chart 在 `row 4406` 会从约 `-6.29 deg` 跳到 `-67.87 deg`，但实际 bridge bucket 只从 `-8.69 deg` 到 `-8.44 deg`。

最新剩余大跳集中在：

| JSONL row | bucket qpos jump | 现象 |
| ---: | ---: | --- |
| 2501 | `-28.80 deg` | IMU0 quaternion 在 bridge snapshot 中约 `129.6 deg` 跳变 |
| 2502 | `-12.95 deg` | IMU0 从异常姿态继续回落 |
| 7181 | `-28.84 deg` | IMU0 quaternion 在 bridge snapshot 中约 `60.6 deg` 跳变 |

raw CAN 复核显示，dev0 quaternion 半包同步整体正常：

- dev0 q half delta p50: `0.918 ms`
- dev0 q half delta p99: `0.986 ms`
- dev0 q half delta max: `10.79 ms`
- 最新大跳附近 q half delta 约 `0.9 ms`

raw CAN status 样本里，IMU0 quaternion 的 top jumps 包括：

- `82.73 deg`
- `66.62 deg`
- `60.60 deg`
- `47.65 deg`

同一时刻 IMU1/IMU2/IMU3 大多平稳。证据指向 IMU0 AHRS 在特定姿态区间输出不稳定。

## 用户操作轨迹与 RPY 单调性

用户描述本轮操作为：

1. bucket 从最外展一端到最内收一端
2. 回到另一端
3. 重复两个循环
4. 最后跳回 home 附近

按 bucket qvel 方向切段后，bucket qpos 轨迹能对应这个往返结构：

- `36.7 deg -> -119.4 deg`
- `-119.2 deg -> 44.5 deg`
- `42.9 deg -> -113.1 deg`
- `-113.1 deg -> -35.6 deg`

但 IMU0 `rpy_raw_deg` 在这些段内不单调，尤其 pitch 接近 `-80 ~ -88 deg` 时，roll/yaw 会大幅交换分支。例如：

- row 7181: pitch `-52.37 deg -> -83.96 deg`，roll/yaw 同时大幅变化
- row 7182: roll `16.51 deg -> -166.64 deg`，yaw `330.40 deg -> 160.53 deg`
- row 7489 附近：pitch 接近 `-88 deg`，roll/yaw 多次大幅来回切换

这里的 RPY 仍来自 IMU 协议原始 `rpy_raw_deg`，保留它主要是为了诊断 AHRS 输出。bucket 主 qpos 已经使用 quaternion relative 解算。后续可以在诊断里增加 `rpy_from_quaternion`，但它仍会受到 Euler 表示奇异影响，不能作为 bucket qpos 的根本修复。

## 第四轮证据：单程测试中的残余 IMU0 AHRS 跳变

用户随后做了更简单的一轮动作：

```text
home 附近 -> 外展极限 -> 内收极限 -> 停下
```

理论上 bucket qpos 应只有两段主趋势。对应日志：

- summary: `/media/mundane/EXTERNAL_USB/imu_qvel_tests/imu_qvel_20260706T080903.723648Z_summary.json`
- JSONL: `/media/mundane/EXTERNAL_USB/imu_qvel_tests/imu_qvel_20260706T080903.723648Z.jsonl`
- raw CAN: `/media/mundane/D/Excavator_real_stack/artifacts/slave_stack/20260706_160854/canraw.log`
- 本地可视化: `artifacts/imu_qvel_analysis/20260706_080903/bucket_qpos_curve_20260706T080903.svg`
- PNG 预览: `artifacts/imu_qvel_analysis/20260706_080903/bucket_qpos_curve_20260706T080903_preview.png`
- 逐帧 CSV: `artifacts/imu_qvel_analysis/20260706_080903/bucket_qpos_curve_20260706T080903.csv`

整体 qpos 轨迹与用户动作一致：

| 阶段 | row | elapsed_s | bucket qpos |
| --- | ---: | ---: | ---: |
| home 起点 | 0 | `0.000` | `-50.98 deg` |
| 外展极限 | 3068 | `68.404` | `35.85 deg` |
| 内收极限 | 5233 | `117.857` | `-122.79 deg` |
| 停下结束 | 5467 | `123.094` | `-121.08 deg` |

这一轮没有再出现旧 chart 导致的 `100-200 deg` 级翻转。离线用同一套双 chart 逻辑复算出的 bucket qpos 与 bridge 已发布 qpos 最大差异约 `0.118 deg`，说明当前可视化和在线输出是一致的。

残余问题集中在第二段动作中的一簇异常，时间约 `112.40-112.58 s`。关键逐帧片段：

| row | elapsed_s | bucket qpos | qpos step | qpos diff qvel | bridge qvel | IMU0 qdelta | IMU1 qdelta | relative qdelta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4982 | `112.508` | `-33.73 deg` | `-0.39 deg` | `-0.28 rad/s` | `-0.102 rad/s` | `5.57 deg` | `0.03 deg` | `5.57 deg` |
| 4983 | `112.532` | `-34.54 deg` | `-0.82 deg` | `-0.59 rad/s` | `-0.103 rad/s` | `12.42 deg` | `0.03 deg` | `12.42 deg` |
| 4984 | `112.554` | `-62.86 deg` | `-28.32 deg` | `-20.17 rad/s` | `-0.109 rad/s` | `126.66 deg` | `0.05 deg` | `126.62 deg` |
| 4985 | `112.579` | `-74.76 deg` | `-11.90 deg` | `-9.09 rad/s` | `-0.110 rad/s` | `10.58 deg` | `0.06 deg` | `10.63 deg` |
| 4986 | `112.604` | `-79.41 deg` | `-4.65 deg` | `-2.99 rad/s` | `-0.112 rad/s` | `4.13 deg` | `0.07 deg` | `4.17 deg` |

同一段里 IMU0 的 gyro/accel 很平滑：

- IMU0 `gyro_y`: 约 `-7.9 ~ -8.1 deg/s`
- IMU0 accel: 约 `[9.6, 0.2, -2.7] m/s^2`
- IMU1 quaternion 同期平稳，单帧 delta 约 `0.03-0.07 deg`

这说明 qpos 的非物理跳变来自 IMU0 quaternion candidate，而不是 bucket 真实快速运动。

raw CAN 复核只保留半包间隔 `<5 ms` 的完整 quaternion 样本后：

| device | valid samples | `qdelta >= 5 deg` events | max qdelta | q half p50 | q half max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| dev0 | 6487 | 5 | `77.62 deg` | `0.464 ms` | `0.607 ms` |
| dev1 | 6478 | 0 | `0.79 deg` | `0.463 ms` | `0.901 ms` |
| dev2 | 5853 | 0 | `1.68 deg` | `0.915 ms` | `1.015 ms` |
| dev3 | 5887 | 0 | `3.97 deg` | `0.913 ms` | `1.090 ms` |

dev0 的 5 个 raw CAN quaternion 大跳全部集中在 `112.40-112.48 s`。半包时间正常，因此这轮证据进一步排除了半包错配作为主因。

进一步检查 IMU0 quaternion 与 accel 重力方向的一致性：

| row | IMU0 rpy_raw_deg | accel | quaternion 推出的重力方向与 accel 夹角 |
| ---: | --- | --- | ---: |
| 4981 | `[-16.50, -48.82, 155.55]` | `[9.6, 0.2, -2.6]` | `55.85 deg` |
| 4982 | `[-17.69, -49.12, 152.40]` | `[9.6, 0.2, -2.6]` | `55.42 deg` |
| 4983 | `[-20.12, -49.65, 146.42]` | `[9.6, 0.2, -2.7]` | `54.86 deg` |
| 4984 | `[-19.74, -79.12, 10.53]` | `[9.6, 0.2, -2.7]` | `15.69 deg` |
| 4985 | `[-159.46, -82.20, 153.44]` | `[9.6, 0.2, -2.7]` | `5.56 deg` |
| 4986 | `[-172.31, -76.14, 166.35]` | `[9.6, 0.2, -2.7]` | `2.78 deg` |

这说明跳变前 IMU0 AHRS 已经运行在一个与重力观测明显不一致的姿态解上；到该姿态区间后，AHRS 突然收敛到更符合 accel 的姿态分支。由于 bucket qpos 使用 IMU0/IMU1 relative quaternion，IMU0 的这次分支切换直接投射成 bucket qpos 大跳。

同一轮还有 `51.75-51.95 s` 附近的轻微征兆：JSONL 中 IMU0/relative qdelta 约 `5.3-5.7 deg`。该区间 primary chart strength 很弱，双 chart tracker 切到 secondary，因此没有造成明显 qpos 大跳。

## 当前结论

1. **200 度级 bucket 翻转主要来自旧 scalar chart 的 `atan2` 病态。**  
   双 chart phase tracker 已明显改善，最新一轮没有再出现 `100-200 deg` 级 bucket qpos 翻转。

2. **当前主问题收敛为 IMU0 AHRS quaternion 在特定姿态区间的一簇换解。**  
   最新 `080903` 单程测试中，会造成 bucket qpos 明显错误的四元数跳变集中在 `112.40-112.58 s` 一段；raw CAN 已经能看到 dev0 quaternion 在同一时间窗口大跳，半包时间正常，IMU1/2/3 同期平稳。

3. **问题集中出现在 IMU0 的特定姿态区间。**  
   最新异常段里 IMU0 accel 几乎沿传感器 X 轴，gyro 平滑，而 AHRS 姿态先与重力方向相差约 `55 deg`，随后突然跳到与重力方向更一致的姿态解。AHRS 内部可能在该几何条件下进入不可观/病态区，或者内部使用 Euler 中间量/磁力计/yaw 修正导致姿态解跳到另一个分支。

4. **仅凭单帧错误 quaternion 无法唯一恢复真实角度。**  
   如果输入 quaternion 已经偏离 IMU 真实姿态，那么后处理必须引入额外约束，例如 bucket 1DOF 铰链模型、历史连续性、命令/qvel 方向、或独立角度传感器。

## 已尝试但暂未采用的方向

### 全局 quaternion manifold / PCA phase

用最新 sweep 数据拟合 relative quaternion 的全局 1D/2D phase，希望直接从 `q_rel` 得到稳定 bucket angle。初步结果不理想：

- 固定 PCA/great-circle phase 不能稳定区分 `row 2501/7181` 这类 AHRS 跳变
- 这些坏点在某些拟合 residual 上并不一定明显偏离正常轨迹
- 直接写一个全局 PCA basis 到 runtime 风险较高

### 任意 twist axis 搜索

离线搜索过多个 twist axis，试图找到一个比当前 `(w,y)` 更稳定的单 chart。部分轴能降低最大跳变，但动态范围、单调性和物理含义不够稳定，暂不适合作为生产修复。

### 单步方向强制

把单步 delta 全部按 qvel/command 方向取正或取负，不能恢复真实角度。它只能修正方向，不能修正幅度。像 `row 2501` 这种单帧 `28.8 deg`，按 qvel 和 `20 ms` 时间只可能移动约 `0.x deg`，强行取方向仍会保留错误幅度。

## 下一步建议

### 方向 A：从源头查 IMU0 AHRS 配置

检查 IMU0 的 AHRS 模式和配置：

- 是否启用磁力计或 9-axis fusion
- 是否可切换 6-axis gyro+accel 模式
- 是否存在安装方向/坐标轴配置
- 是否存在 Euler 输出参与内部修正的模式
- IMU0 附近是否有磁干扰或局部振动

如果能让 IMU0 quaternion 在 `pitch -80~-88 deg` 区间稳定，bucket qpos 主问题会直接消失。

### 方向 B：bucket 专用 1DOF constrained estimator

绕过厂商通用 3D AHRS 的 bucket qpos 输出，把 bucket 建模为一个 1DOF 铰链：

```text
raw IMU0/IMU1 accel/gyro/quaternion
-> bucket hinge constrained estimator
-> theta
```

这里 gyro 不直接积分成 qpos；它只作为动态预测/方向约束。最终角度来自铰链模型下的观测校正。

### 方向 C：短时 branch offset / outlier correction

如果现场需要先降低跳变风险，可以在 quaternion scalar candidate 后维护 branch offset：

- candidate 来自 quaternion
- 单帧变化超过物理可达阈值时，更新 offset 让输出保持连续
- 后续继续使用 quaternion candidate 的增量

这属于临时保护。离线重放显示，它能把最新日志 bucket 最大跳变从 `28.84 deg` 降到约 `7-9 deg`；对旧日志的 `200 deg` 翻转也有效。它的风险是会把某些真实快速动作也视为异常，因此不能作为最终答案。
