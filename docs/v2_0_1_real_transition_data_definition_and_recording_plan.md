---
type: data-contract-and-recording-plan
project: Excavator Real Stack
version: v2.0.1-real-transition
status: superseded-four-cycle-implementation-baseline
created: 2026-08-12
updated: 2026-08-14
scope: historical-four-cycle-p0-p1-implementation-baseline
decision_log: docs/v2_0_1_real_transition_design_decision_log.md
superseded_by: docs/v2_0_1_real_transition_final_conclusion.md
active_sequence_plan: docs/v2_0_1_real_transition_experiment_sequence_design.md
field_preparation: docs/v2_0_1_real_transition_field_preparation_and_calibration_runbook.md
---

# v2.0.1 Real Transition 数据定义与数据录制计划

> **状态说明（2026-08-14）**：本文保留为 commit `a64e5d1` 的旧版“P0/P1、每条固定四个 cycle”录制基线。当前工作分支已实现 multi-sequence v2；有效范围和数量见《[v2.0.1 Real Transition 最终结论](v2_0_1_real_transition_final_conclusion.md)》与《[实验执行序列设计](v2_0_1_real_transition_experiment_sequence_design.md)》。本文不再是新 session 的执行依据。

> 当前上机同步、只读检查和 home/A/B 标定入口见《[真机测试与标定准备手册](v2_0_1_real_transition_field_preparation_and_calibration_runbook.md)》；不要从本文旧 P0/P1 示例反推现场命令。

## 0. 一页结论

v2.0.1 的目标是用一批可在一天内完成的真机数据，验证下面这条技术路线能否基本跑通：

```text
任务程序提交下一目标
→ Conditioned ACT 完成当前一铲并返回目标侧
→ ready 边界确认
→ 任务程序提交后续目标
→ 多个 ready-to-ready 原子动作连续衔接
```

现有 N5 已经在真机上完成过完整一铲。本轮以此为起点，验证：**加入外部目标后，ACT 能否执行留在原处或移动到另一处，并在连续四铲中保持衔接。**

首批正式数据按以下规模规划：

| 项目 | 计划数量 |
|---|---:|
| Ready 区域 | 2 个，记为 A、B |
| 每条连续 run | 4 个 cycle |
| 每个平衡 block | 4 条 run，16 个 cycle |
| 正式 block | 6 个 |
| 正式有效 run | 24 条 |
| 正式有效 cycle | 96 个 |
| 训练 / 验证 / 锁定测试 | 64 / 16 / 16 cycle |

现场时间不足或出现失败重录时，优先保证 64 个有效 cycle 的最小闭环包：32 train、16 validation、16 locked test。96 cycle 是首批目标包，不是必须冒着安全风险追完的现场 quota。

核心安排如下：

- 以当天标定的 home swing 方向为中线，左侧直接归为 A，右侧直接归为 B。
- 模型 condition 使用二分类目标 `A=-1 / B=+1` 和有效位。
- 每个 cycle 的目标由脚本 planner 在本 cycle 动作开始前提交，不能在完成后按实际落点回填目标。
- 师傅只服从目标侧，摇杆幅值、动作时序和关节轨迹都由师傅自然决定。
- 一条 run 连续完成四铲，中间不 go-home、不人工摆位、不重置模型。
- 原始数据按连续 run 保存，训练前再切成 ready-to-ready cycle。
- 第一批 96 cycle 录完后先训练和闭环验证，不立即扩大数采。

历史真机数据给出的首轮执行基线如下。它们用于实现、离线 QC 和上机前 shadow，不替代当天标定：

| 项目 | 首轮基线 |
|---|---:|
| home swing 中线 | `0.000690 rad` |
| 左右符号 | qpos 减小为左，`LEFT_SIGN=-1` |
| home 分类容差 | `abs(shortest_angle(swing-home)) <= 0.05 rad` |
| clean endpoint 最小侧向裕量 | `0.08 rad` |
| swing 安全范围 | `[-0.3892, +0.4189] rad` |
| ready 稳定窗 | 连续 0.5 s 内 `abs(swing_qvel) <= 0.015 rad/s`；boom/stick/bucket 只记录、不阻断 |
| clean goal 提前量 | `>=100 ms`；`50–100 ms` 复核；`<50 ms` 剔除 |
| return candidate | `expected_return_swing_sign * swing_action >0.08`，连续 6 个 20 Hz row，确认后回溯 5 row |
| 单 cycle 时间 | 45 s 复核，60 s 停止 |

基线的来源、允许现场覆盖的字段和变更版本规则见第 17 节。当天 resolved 值必须随 run package 保存，不能只停留在操作记录或口头约定中。

这批数据的结论范围是：固定真机 context、home 左右二分类、固定脚本 planner 下的四铲连续 transition 原型。任意位置控制、任意顺序规划、跨场地泛化和无人值守属于后续独立验证范围。

## 1. 本轮要回答的问题

### 1.1 主问题

在相近的当前状态下，改变下一 ready 目标，是否会让 ACT 的返回动作和最终落点发生方向正确、幅度足够的变化？

### 1.2 连续执行问题

四种原子 transition 能否在连续四铲中组合：

```text
A → A
A → B
B → A
B → B
```

连续运行时只允许在 run 开始重置一次 policy。cycle 边界只更新任务目标和 lifecycle，不重置 ACT 或 temporal aggregation。

### 1.3 数据链问题

现场记录能否完整保留以下因果关系：

```text
程序实际提交的目标
→ 提交时间
→ 师傅执行的动作
→ 机器实际到达的位置
→ 本铲的物理结果与失败原因
```

如果这条关系记录不完整，后续即使有大量数据，也无法判断模型是在服从 condition，还是在复现师傅的自然习惯。

## 2. 从 v2.0.0 SimVerify 继承的实验合同

v2.0.1 使用 SimVerify 已形成的 cycle、condition、split 和对照实验结构。真机数值、位置范围和现场判据由 Real calibration 产物提供。

### 2.1 继承项

| SimVerify 思路 | v2.0.1 Real 用法 |
|---|---|
| ready-to-ready cycle | 每个训练文件表示一次完整挖掘、卸料和返回下一 ready 的原子动作 |
| 明确 condition 生命周期 | condition 在目标提交前无效，提交后才进入 policy 输入 |
| 连续 cycle composition | 一条真机 run 连续完成四个 cycle，中间不重置 policy |
| source episode 隔离 | 按完整真机 block 划分 train、validation 和 locked test |
| cycle 末端 action mask | ACT chunk 超出当前 cycle 的部分不参与监督 |
| B0 / B1 / B2 | 用无目标、正确目标、错配目标区分基本动作和 condition 作用 |
| endpoint 优先 | validation loss 只选 checkpoint，实际结论看目标终点和连续闭环 |

### 2.2 Ready rule authority

`ready_contract.json` 是本轮 ready 的唯一数值 owner。它由 `prepare-session` 自动生成，
固定 home swing `0.000690 rad`、物理左侧符号 `-1`、home tolerance `0.05 rad`、
clean-ready `0.08 rad`、安全范围 `[-0.3892,+0.4189] rad`，以及连续 `0.5 s`
全窗 `abs(swing_qvel)<=0.015 rad/s`。不再从 home/A/B 各 10 个姿态窗口解析合同。

resolved real config 继续拥有 deadzone、动作幅值、时延、同步和相机合同。所有 cycle
目标均来自脚本 planner 在本 cycle 动作开始前生成的 `prospective command`。修改上述
ready 常量必须生成新的合同/context，不能在运行中覆盖。

## 3. 名词和基本单元

### 3.1 Home 中线与 A/B 二分类

本轮 `home_swing_axis=0.000690 rad`，物理左侧 qpos 符号为 `-1`。该值只作为
左右分类中线；run 内不执行 go-home，其余三轴允许采用师傅自然的任意离土预姿态。

A 表示 home 中线左侧的 clean ready 区域，B 表示右侧的 clean ready 区域。A/B
没有固定目标角，也不要求师傅停在同一个 qpos 点；actual side 由当前 swing qpos 自动
分类，不由人工填写。

四种 transition 的物理语义为：

| Transition | 返回动作 |
|---|---|
| A→A | 留在 home 左侧 |
| A→B | 越过 home 中线向右移动 |
| B→A | 越过 home 中线向左移动 |
| B→B | 留在 home 右侧 |

A、B 用于：

- 现场给师傅显示目标；
- 平衡四种 transition 的录制数量；
- 对结果做分组统计。

A/B 标识进入日志和 manifest。Policy 只接收脚本 planner 为当前 cycle 提交的目标侧，不接收完整 run 顺序、cycle index 或下一条目标。

### 3.2 Ready

`Ready_i` 表示机器已经处于第 `i` 铲可以自然开始的位置和姿态。在线合同必须满足：

```text
当前 swing qpos 位于脚本要求的 clean A/B，且在安全范围内
最新连续 0.5 s 每行 abs(swing_qvel) <= 0.015 rad/s
bucket_clear_confirmed = true
operator_confirmed = true
```

boom、stick、bucket 的 qpos 不设边界；其 qvel 窗口统计写入 `ready_evidence`，但不阻止
ready。0.5 s 是按时间戳覆盖的全窗判据，不按单帧或固定行数替代。初始和目标 ready
都必须由操作员分别确认铲斗离土和当前状态可继续作业。

### 3.3 Cycle

一个 cycle 定义为：

```text
Ready_i
→ 脚本 planner 提交下一 ready 侧 Goal_i
→ dig
→ carry / swing-to-dump
→ dump
→ return
→ Ready_(i+1)
```

目标提交后师傅才开始本 cycle 动作。训练切片使用从 `goal_commit_i` 后第一帧到 `Ready_(i+1)` 的 half-open 区间。`Ready_(i+1)` 是下一 cycle 的候选起点；下一目标提交后才开始下一训练 cycle。

### 3.4 Run

一条 run 包含四个连续 cycle：

```text
Ready_0 → Ready_1 → Ready_2 → Ready_3 → Ready_4
```

因此一条四铲 run 包含一个初始 ready 和四个程序目标。第四铲结束后，师傅仍需回到最后一个目标 ready，确认到位后才结束录制。

### 3.5 Balance block

一个 block 包含四条 run，共 16 个 cycle。block 是目标顺序平衡、数据划分和来源隔离的最小单位。它不强制四条 run 使用同一块未经整理的土面。

### 3.6 Scripted target 与 realized target

- `scripted_target`：程序在动作发生前实际提交给师傅和模型的目标；
- `realized_target`：本铲结束后，根据真实 qpos、视觉和人工复核确认的实际到达侧及状态。

两者必须分开记录。`realized_target` 不能覆盖或回填 `scripted_target`。

## 4. 真机目标定义

### 4.1 Ready rule contract

`ready_contract.json` 使用 `real_transition_ready_rule_contract_v2` schema。它由
`prepare-session` 按现场已确认的固定规则自动生成，至少包含：

```text
schema
swing_axis:
  axis_index                    # 0
  home_rad                      # 0.000690
  left_qpos_sign                # -1
  home_tolerance_rad            # 0.05
  clean_ready_min_abs_delta_rad # 0.08
  safe_range_rad                # [-0.3892, +0.4189]
  stable_window_s               # 0.5
  stable_qvel_abs_max_rad_s     # 0.015
  max_sample_gap_s              # 0.15
requirements:
  bucket_clear_confirmation_required
  operator_confirmation_required
  non_swing_qpos_bounds         # null
  non_swing_qvel_gates_ready    # false
evidence
contract_sha256
```

A/B 是相对 home 的区域标签，不是固定目标姿态。所有 swing 分类都使用最短角误差：

```text
delta = shortest_angle(current_swing_rad - 0.000690)

delta < -0.08              : clean A / left
-0.08 <= delta < -0.05     : transition / reject
abs(delta) <= 0.05         : home / reject
+0.05 < delta <= +0.08     : transition / reject
delta > +0.08              : clean B / right
qpos outside safe range    : unsafe / reject
```

### 4.2 Ready 在线判定

`initial-ready` 与 `target-ready` 使用同一门控。最近连续 0.5 s 的每个样本必须
都在安全范围、属于同一个 clean A/B，且 `abs(swing_qvel)<=0.015 rad/s`；采样间隔
不得超过 0.15 s。窗口末端自动得到 `actual_side`，并与 manifest 规定的 initial
side 或 scripted target 核对，不能由人工输入侧别覆盖。

操作员还必须分别确认铲斗离土和整体 ready。boom、stick、bucket 的 qpos 不设边界，
三轴 qvel 只写入 ready evidence 供复核，不阻止 ready。这样允许熟练操作员在 swing
停稳前后自然完成大小臂和铲斗预姿态，同时避免把 home、过渡区、未停稳或实际侧不符
误标为 ready。当前规则不再要求 home/A/B 各 10 个固定标定窗口。

### 4.3 Policy condition

v2.0.1 第一版 condition 固定为目标侧二分类加有效位：

```text
target_side_code = -1 for A, +1 for B
real_transition_condition_v1 = [target_side_code, goal_active]
```

其中：

- `target_side_code` 只表达本 cycle 的目标侧；
- 当前侧由 qpos 和视觉状态表达，ACT 据此形成 A→A、A→B、B→A、B→B 四种动作模式；
- `goal_active=0` 时 condition 固定写 `[0,0]`；
- `goal_active=1` 时 condition 为 `[-1,1]` 或 `[+1,1]`；
- `goal_epoch`、`goal_id` 和 A/B 标识完整进入日志；
- Policy 低维输入使用 qpos[4] 和 `real_transition_condition_v1[2]`。

第一版 ACT 低维输入固定为 qpos[4] 和 condition[2]。qvel 继续完整记录，用于 ready、QC、状态稳定性和后续独立消融。

### 4.4 目标提交时机

目标提交必须满足：

```text
Ready_i 已确认
≤ goal_commit_i
< 本 cycle 首个操作意图
```

首个操作意图定义为 raw teleop action 或 guarded action 中任一层第一次满足 `max(abs(action)) > 0.05` 的 source row；首个有效机械动作另行记录为 `commanded_action` 第一次越过对应方向 mechanical deadzone 的 row。goal 时序以更早的操作意图为准，不能等机器开始运动后再判断。

定义 `goal_lead_ms = first_cycle_intent_ns - goal_commit_ns`：

| `goal_lead_ms` | 归类 |
|---:|---|
| `>=100 ms` | clean |
| `50–100 ms` | review，不进入第一版 condition 主训练 |
| `<50 ms` | excluded；若为负数同时标记 `late_goal_commit` |

现场界面在 sequencer、policy router 和 recorder 接受同一个 `goal_id/goal_epoch` 后进入 armed 状态，只显示当前 cycle 的目标侧。师傅看到 armed 后再开始本 cycle，不要求额外固定等待 100 ms；100 ms 是事后因果 QC 裕量，不是人为延迟动作的计时口令。

如果师傅已经开始本 cycle 动作，程序才提交目标，该 cycle 标记为 `late_goal_commit`，不能进入 condition 主训练集。

condition 的逐帧生命周期为：

```text
cycle 间等待至 goal_commit： [0, 0]
goal_commit 后至 cycle end： [target_side_code, 1]
进入下一 cycle 等待：        [0, 0]
```

目标 lifecycle 在 cycle 边界重新武装；policy、ACT step、缓存 chunk 和 temporal aggregation 不随 cycle 重置。

## 5. 原始数据合同

### 5.1 保存单位

原始数据按完整连续 run 保存，不在现场直接切成四个训练 episode。推荐每条 run 形成一个不可覆盖的数据包：

```text
real_transition_raw_v1/
  session_<session_id>/
    block_<block_id>/
      run_<run_id>/
        raw.hdf5
        task_events.jsonl
        run_manifest.json
        SHA256SUMS.txt
```

`raw.hdf5` 使用现有真机 HDF5 v1.2 主结构；新增任务事件优先使用带时间戳的 sidecar，并由 `run_manifest.json` 和 SHA256 与 HDF5 绑定。正式实现如果选择 append-only HDF5 group，也必须保留相同字段语义。run 结束并写入 checksum 后，这三个原始文件不再追加或修改。

### 5.2 原始 HDF5 必需内容

| 类别 | 必需字段或语义 |
|---|---|
| 视觉 | `observations/encoded_images/video4..video7`，相机顺序和 transform 固定 |
| 状态 | `observations/qpos[4]`、`observations/qvel[4]` |
| BC 标签 | `action[4]`，保持当前 guard-filtered normalized teleop command 语义 |
| 动作来源 | `action_source/type`、`action_source/id` |
| 动作诊断 | raw action、guarded/safe action、commanded action、guard reason |
| 时间 | `timestamps/step_id`、`timestamps/step_ns`，单调且可与事件对齐 |
| 控制诊断 | controller ACK、fault、发送时间、传感器健康和同步状态 |
| 时序诊断 | action source latency、bridge age、camera age、四相机 group skew、sync max skew、group validity |
| 元数据 | machine、operator、session、配置版本、相机、动作轴、qpos/qvel 来源 |

原始录制保持当前真机 50 Hz 和四路 JPEG；训练数据从原始记录确定性生成 20 Hz 版本。不能为了省空间只保存已经降采样的 cycle。

### 5.3 在线 `task_events.jsonl`

每条事件至少包含：

```text
event_id
event_type
event_step_id
event_step_ns
session_id
block_id
run_id
cycle_id
goal_id
goal_epoch
scripted_target_side         # A / B
target_side_code             # -1 / +1
expected_return_swing_sign   # -1 / +1 / null，仅供标注与 QC，不进入 policy
event_source                 # sequencer / experimenter / operator / system
commit_ack_sources           # goal_commit 时为 recorder / router / display
notes
```

这个文件只保存录制当时实际发生的程序事件和现场 marker。必需事件类型：

```text
run_start
initial_ready_mark
dump_end_mark
goal_commit
target_ready_mark
run_complete
run_abort
manual_intervention
safety_stop
```

`goal_commit` 必须由脚本 planner 生成，它是程序真正显示并向 policy router 发布本 cycle 目标的时刻。不能用师傅开始动作、最后到达位置或离线推测时刻代替。

`goal_commit` 只有在 recorder、policy router 和 display 都接受同一个 `goal_id/goal_epoch` 后才完成；任一方未确认时界面保持 unarmed，不能让师傅开始本 cycle。三个接收结果和共同的 commit 时间进入同一事件，避免把界面时间、模型时间和日志时间分别解释成目标提交时刻。

现场 marker 可由实验员点击，不要让师傅在操作中找按钮或记录时间。marker 是快速定位索引，不是最终训练边界。

启用 shadow detector 后，系统另外记录 `ready_candidate_start`、`ready_candidate_met` 和 `return_candidate`。这些事件只用于比较自动检测与人工 marker 的偏差，不改变人工 evidence owner。

### 5.4 离线确认边界

原始 run 封存后，detector 和复核人员在独立文件 `cycle_annotations_v1.jsonl` 中产生：

```text
initial_ready_confirmed
goal_commit_confirmed
first_cycle_intent_confirmed
first_effective_action_confirmed
dump_end_confirmed
first_return_action_confirmed
target_ready_confirmed
cycle_validity_confirmed
```

每个确认事件必须记录 source raw checksum、annotation schema、detector 版本、复核人、source row/time 和使用的证据。自动 detector 只能生成 candidate；第一版 clean 数据的边界由人工复核后冻结。更改边界时发布新 annotation 版本，不改原始 `task_events.jsonl`。

candidate 生成规则固定为：

- ready candidate 使用第 3.2 节的 0.5 s 联合稳定窗；`target_ready_confirmed` 放在稳定窗末端，因此 cycle 切片保留完整的末端停稳证据；
- return candidate 从 `dump_end_mark` 附近搜索，使用本次 dump→target 几何对应的 swing 方向；若方向元数据缺失，同时生成正、负两个 candidate 并要求人工选择，不能固定复用历史数据中的负方向；
- return candidate 要求 `expected_return_swing_sign * swing_action >0.08` 连续 6 个 20 Hz row，确认后回溯 5 row 到持续动作段首帧；
- `dump_end_mark` 和 return candidate 前后各保留 2.5 s 复核窗；
- qvel 或 qpos 起动只作为交叉证据，不能单独定义 return 起点。历史数据中动作意图到 `0.005 rad` swing 位移的 P95 约为 2.11 s，单用运动起点会系统性切晚。

### 5.5 `run_manifest.json`

每条 run 至少记录：

- data-contract version、git commit 和 resolved config；
- HDF5、事件文件、ready contract、sequence manifest 的 SHA256；
- machine、operator、session、block、run、workface/reset 标识；
- 光照、天气、土面状态和明显遮挡的简短分类；
- 初始 ready、计划四目标、在线 marker 和现场结果备注；
- 每个 cycle 的 dump→target 几何和 `expected_return_swing_sign`；未知时显式写 `null`；
- 是否完成、首个失败 cycle、停止原因；
- 是否发生人工接管、安全停止、传感器异常或动作来源切换；
- 原始文件是否完整封存。clean training、diagnostic 或 failure 的最终归类写入派生 `cycle_manifest.jsonl`，不回写 raw manifest。

## 6. 派生 ready-to-ready 训练数据

### 6.1 不覆盖原始数据

离线处理从只读 raw run 生成独立、不可变的数据目录：

```text
real_transition_cycle_v1/
  annotations/
  episodes/
  cycle_manifest.jsonl
  split_manifest.json
  qc/
  SHA256SUMS.txt
```

每次修改边界、condition 或过滤规则，都生成新版本目录。不能原地改标签后继续沿用旧 checksum。

### 6.2 单 cycle 文件

每个训练 cycle 至少包含：

```text
observations/qpos
observations/qvel
observations/encoded_images/video4
observations/encoded_images/video5
observations/encoded_images/video6
observations/encoded_images/video7
action
timestamps/step_id
timestamps/step_ns

conditions/real_transition_condition_v1
conditions/target_side_code
conditions/goal_active_mask
conditions/goal_epoch
conditions/cycle_id
conditions/valid_mask

labels/current_ready_side
labels/scripted_target_side
labels/realized_target_side
labels/home_side_coordinate_rad
labels/goal_source
labels/transition_type
labels/transition_success
labels/physical_effect
labels/failure_reason
labels/expected_return_swing_sign

provenance/source_session_id
provenance/source_block_id
provenance/source_run_id
provenance/source_row_index
provenance/source_step_id
provenance/annotation_version
provenance/annotation_sha256
```

字符串标签可以放在 manifest 中；逐帧训练字段必须保留同样的可追溯语义。Policy 输入使用 allowlist，只包含四路图像、qpos[4] 和 `real_transition_condition_v1[2]`；`cycle_id`、run/provenance、planner 模板和后续目标都不能进入模型输入。

### 6.3 切片与 action chunk

- cycle 从 `goal_commit_i` 后第一条训练 row 开始，到 `Ready_(i+1)` 的确认边界结束；
- `Ready_i`、`goal_commit_i` 和 `Ready_(i+1)` 来自已冻结的离线 annotation，不直接把在线 marker 当作训练边界；
- 20 Hz row 必须可追溯到 50 Hz source row 和时间戳；
- 20 Hz 生成沿用当前确定性规则：每个 50 ms 目标时刻选择第一条不早于目标的 source observation，action label offset 为 `-20 ms`；resolved dataset config 必须保存这两个值；
- cycle 内所有训练 row 都令 `goal_active=1`，并保持同一个 `target_side_code`；
- cycle 间等待和 `goal_commit` 以前的 row 不进入主训练切片，condition 记为 `[0,0]`；
- 20-step ACT chunk 超出 cycle 结束边界的部分设为 invalid；
- 下一 cycle 的 action 不能成为当前 cycle 的监督；
- split 在切 cycle 之前按 source block 冻结。

goal、dump/return 和 ready 边界前后 2.5 s 使用局部严格 gap 规则：50 Hz source 相邻 row 间隔 `>100 ms`，或 20 Hz 派生相邻 row 间隔 `>120 ms`，对应 cycle 不进入 clean。任意位置出现 `>250 ms` 派生 gap 视为结构性损坏；不能用插值跨过这些 gap 继续生成有效 action chunk。

### 6.4 Clean、diagnostic 和 failure 分层

进入 clean 主训练集的 cycle 必须同时满足：

1. 完整、前瞻记录的 scripted target；
2. `goal_commit` 比本 cycle 首个操作意图至少早 100 ms；
3. 事件顺序完整且边界可确认；
4. 四路相机、qpos、qvel、action 和时间戳通过 QC；
5. 动作来源保持人工 teleop，没有 go-home、policy 或恢复控制混入；
6. realized side 与 scripted target side 一致，且没有落入 home 排除带或已示教支持范围外；
7. 没有人工接管、安全停止或未解释的控制故障；
8. 本铲物理效果被标为正常，或经过人工复核确认可作为完整一铲示教。

以下数据保留但不进入第一版 clean 主训练：

- 目标迟交；
- 到错目标；
- 空铲、掉土或卸料不完整；
- 中途人工接管；
- 安全停止或传感器故障；
- 边界不确定；
- 失败后的恢复动作。

失败数据不能删除，也不能把实际到达位置重新写成目标后放回 clean 集合。

## 7. 目标顺序生成

### 7.1 固定脚本 planner

下一侧由上层脚本 planner 决定，ACT 不预测也不选择下一 cycle 去哪里。首版只使用两条互为镜像的四 cycle 模板，并在录制前写入 `sequence_manifest.json`：

```text
P0: A → B → B → A → A
P1: B → A → A → B → B
```

五个字母依次表示初始 ready 和四个 cycle 的目标侧。每条 run 自身都恰好包含一次 `A→A`、`A→B`、`B→A`、`B→B`。

现场界面在每个 cycle 开始前显示当前目标 A 或 B。师傅可以按这两条简单循环操作；程序仍逐 cycle 产生 `goal_commit`，保证目标、动作和边界可以审计。ACT 只收到当前 cycle 的 `target_side_code`，不收到 P0/P1 模板 ID、cycle index 或后续目标。

### 7.2 每个 block 的平衡约束

一个 block 固定包含两条 P0 和两条 P1，四条 run 的现场先后顺序随机化并在录制前冻结。失败后新增同模板 replacement run，不改写原 run。

| Run | 完整序列 | 初始 ready | 四个程序目标 |
|---|---|---|---|
| P0-1 | `ABBAA` | A | B、B、A、A |
| P1-1 | `BAABB` | B | A、A、B、B |
| P0-2 | `ABBAA` | A | B、B、A、A |
| P1-2 | `BAABB` | B | A、A、B、B |

因此每个 block 有 16 个 cycle，四种 transition 各 4 个。固定 planner 的顺序不是 ACT 的学习目标；condition 是否生效由第 12、13 节的配套对照和目标干预判定。

### 7.3 Planner 与 ACT 的边界

离线 materializer 为每个 cycle 生成独立 episode。训练 sampler 跨 run、block 和 transition 打乱 cycle；原始顺序只保留在 provenance 中。ACT 的输入 allowlist 不包含 P0/P1、cycle index、source time、run id 或下一条 planner command。

因此 ACT 的学习问题固定为：

```text
当前图像 + 当前 qpos + 当前 target side
→ 完成本铲并到达 target side
```

当前侧由 observation 给出，目标侧由 planner condition 给出，两者组合成四种 transition。固定循环只定义上层怎样依次调用这些 transition。它的闭环成功用于评价 composition，不单独作为 condition 生效的证据。

## 8. 一天的录制规模与数据划分

### 8.1 正式计划

| Split | Block | Run | Cycle | 每种 transition |
|---|---:|---:|---:|---:|
| Train | 4 | 16 | 64 | 16 |
| Validation | 1 | 4 | 16 | 4 |
| Locked test | 1 | 4 | 16 | 4 |
| 合计 | 6 | 24 | 96 | 24 |

第一条 run 作为现场流程 smoke。完成四个 cycle 后立即暂停，检查事件、图像、qpos/qvel、action、目标显示和文件落盘。若完全符合正式合同，可以计入对应 block；否则保留为失败试录并重新开始该 block。

### 8.2 划分原则

- split 按完整 block 冻结，不能按切出来的 cycle 随机拆分；
- collection order 随机化，locked test 不固定放在一天最后；
- train、validation 和 locked test 使用不同 source block；
- 每次土面恢复或换工作条带记录新的 `workface_reset_id`；
- 同一连续 run 的 cycle 永远属于同一个 split；
- locked test 只允许做文件完整性 QC，训练方法和阈值冻结前不读取任务结果。

### 8.3 土面安排

- 四铲 run 内不整理土、不 go-home、不人工重新摆位；
- run 之间可以恢复土面或移动到新的相近工作条带；
- 是否恢复、恢复方式和工作条带必须记录；
- A/B ready rule contract 在本 session 保持不变；
- 底盘位置、home swing 参考、swing 编码器零位、相机位姿或安全能力范围发生变化时，关闭当前 block，重新校准并生成新的 context/side-contract version。

### 8.4 数量不足时

数量不足不能在录制后临时重分 split。sequence manifest 必须在前四个优先 block 中预先安排 2 train、1 validation 和 1 locked test，合计 16 条完整 run、64 个有效 cycle。这四个 block 的实际录制先后可随机。剩余两个优先级较低的 block 都属于 train，完成后把训练集从 32 cycle 增加到 64 cycle。

64-cycle 包可以用于端到端训练、数据链和 condition 机制 smoke，结论必须标为初步。96-cycle 目标包用于首次原型决策，仍不表示统计可靠性或部署保证。

少于 64 个有效 cycle 时，只报告录制、切片和训练链是否可运行，不判定 condition 或 composition 可行性。任何情况下都不为追 quota 延长故障 run 或降低安全停止标准。

## 9. 现场录制流程

### 9.1 开始前

1. 按当天现场 runbook 完成机器、急停、相机、CAN、IMU、磁盘和 recorder 检查。
2. 固定相机、图像 transform、摇杆映射、deadzone、scale 和动作轴顺序。
3. 确认固定 home swing 中线、左右符号和安全范围，生成并冻结 `ready_contract.json`。
4. 生成并冻结六个 block 的 sequence 和 split manifest。
5. 确认目标界面不进入 policy 的四路训练相机画面。
6. 确认原始 HDF5 与 task event 使用同一个可对齐时钟和 step id。
7. 把 ready contract、resolved record config、N5 bundle manifest 和各自 SHA256 写入 session manifest；任何一项不一致时不开始正式 run。

### 9.2 每条 run

1. 程序显示初始侧 A 或 B，师傅自然摆到该侧 ready。
2. 实验员点击 `initial_ready_mark`，开始连续录制。
3. 脚本 planner 写入本 cycle 的 `goal_commit`，屏幕显示本 cycle 最终要回到 A 还是 B。
4. 师傅看到目标后开始本铲，正常完成挖掘、运土、卸料和返回；不需要对准唯一数值点。
5. 实验员在卸料完成时记录 `dump_end_mark`，在机器到达目标侧 ready 时记录 `target_ready_mark`。
6. 脚本 planner 提交下一 cycle 目标，师傅继续下一铲，直至完成四个 cycle。
7. 第四铲返回最后目标侧并满足 0.5 s ready 稳定窗后结束 run。
8. 程序保存 HDF5、事件、manifest、checksum 和即时 QC。

### 9.3 失败处理

以下任一情况发生，立即结束当前 run：

- 选错目标；
- 明显空铲、掉土或未完成卸料；
- 需要人工恢复或大幅纠偏；
- 传感器、相机、控制链或目标界面异常；
- 安全员或师傅认为继续操作不安全。

时间只用于发现异常拖延，不代替师傅和安全员判断：单 cycle 超过 45 s 标记 review，达到 60 s 结束 run；从 `first_return_action_confirmed` 到 ready 超过 12 s 标记 review，达到 20 s 结束 run；四 cycle 总时长达到 4 min 仍未完成时结束 run。历史完整人工周期 P95 约 40.2 s、最大约 42.1 s，历史 return 段 P95 约 9.5 s。

失败 run 原样保存。恢复完成后由程序生成 replacement run，不能继续沿用失败 run 的 cycle 编号假装连续。

未完成 run 中已经看似正常的前置 cycle，也不进入首版 primary clean 数据和 block quota。它们可作为诊断数据，或在首次 locked-test 结论冻结后纳入后续辅助训练集。

### 9.4 B1 真机分级启用与响应保护

B1 上机只按以下顺序推进：shadow-zero、四种 standalone transition、连续四 cycle。每一级的日志和停止原因复核完成后才进入下一级。首次 moving test 至少有机器操作员和独立安全观察员；实验员负责 sequencer、日志和 marker，不能用自动 detector 代替安全观察。

首轮 moving action 使用 normalized `commanded_action` 轴向上限：

```text
abs(commanded_action) <= [0.80, 0.70, 0.70, 0.70]
```

首轮保持 `action_scale=[1,1,1,1]`、deadzone assist disabled，不用动作放大或自动顶过死区改变 N5 的既有动作语义。

只有 standalone transition 全部通过后，才允许通过 field override 把历史支持内的上限提高到 `[0.80, 0.80, 0.80, 0.70]`。当前 runtime 的全局 `clip=1.0` 不等于已经实现这组轴向上限；上机前必须在 resolved config 和日志中验证实际裁剪层。

响应计时从 `commanded_action` 越过对应方向 mechanical deadzone 后开始。当前历史基线的 positive deadzone 为 `[0.661, 0.259, 0.500, 0.408]`，negative deadzone 为 `[0.721, 0.357, 0.500, 0.508]`；现场使用当天 resolved config。若命令在 5 tick 内回到 deadzone 内，取消本次计时，不把短脉冲判为失去响应；命令持续越过 deadzone 时必须满足：

- 5 个 policy tick，即 0.25 s 内出现按 command→qvel 标定符号计算的同向 `abs(swing_qvel)>0.015 rad/s`，否则停止；
- 反向 `abs(swing_qvel)>0.015 rad/s` 连续 3 tick，即 0.15 s，立即停止；
- return 阶段在 `old_epoch_weight_mass <=0.5` 后，target-side coordinate 连续向错误方向移动并累计达到 `0.02 rad`，立即停止；
- target、epoch、相机、qpos/qvel 或 action 分层日志任一失效，fail closed。

command→qvel 符号只用于响应监视，不得替代现场物理左右标定。也不能把 swing 上限压到 `0.721` 以下当作主要安全措施，否则负方向命令可能长期落在机械死区内；首轮通过限制持续时间、运行级别和独立停止职责控制暴露。

## 10. 给挖机师傅的任务卡

> 这次以 home 方向为中线，左边都算 A，右边都算 B。每组连续挖四铲。
>
> 每组采用 `A-B-B-A-A` 或左右镜像的 `B-A-A-B-B`。第一个字母是起步侧，后面四个字母是每铲最后要回到的目标侧。每铲开始前屏幕也会显示本铲目标。
>
> A、B 只分 home 左右，不用把机器停到一个精确点。你在对应一侧自然选择可挖的位置；摇杆怎么动、动作多快、铲斗怎么调整，都按平时习惯来。
>
> 一组四铲中间不回原点，不停下来整理土。第四铲倒完后，也要按最后一个提示回到准备位置，停稳后这组才结束。
>
> 如果选错位置、这一铲明显失败、设备异常或者感觉不安全，直接停，不要硬救。失败数据会单独保存，后面再补一组。

## 11. 现场与离线 QC

### 11.1 现场即时 QC

每条 run 结束后至少检查：

- 四路图像存在、可解码、时间长度一致；
- qpos/qvel/action 无 NaN、Inf 和无法解释的跳变；
- step id 和时间戳单调；
- 四个 `goal_commit` 都存在；
- 每个 goal 都比对应 cycle 的首个操作意图至少早 100 ms；
- 四个程序目标、在线 marker 和现场结果备注都已记录；
- action source 全程符合 teleop 合同；
- stop reason 与现场事实一致；
- 文件和 checksum 已落盘。

发现结构性问题后立即暂停录制。不能等到一天结束才发现所有 run 都缺目标时间戳或某一路相机。

时序和同步使用下表的首轮 QC 带。hard 条件落在 goal、dump/return 或 ready 边界前后 2.5 s 时，直接局部剔除对应 cycle；在其他阶段出现时进入 review。hard 条件连续出现、group invalid 或出现 `>250 ms` 派生 gap 时，整条 run 判为结构性失败。

| 指标 | clean | review | hard |
|---|---:|---:|---:|
| action source latency | `<=50 ms` | `50–100 ms` | `>100 ms` |
| bridge age | `<=40 ms` | — | `>40 ms` |
| sync max skew | `<=65 ms` | `65–80 ms` | `>80 ms` |
| camera age | `<=120 ms` | `120–200 ms` | `>200 ms` |
| 四相机 group skew | `<=1 ms` | `1–5 ms` | `>5 ms` 或 group invalid |
| 50 Hz source row gap | `<=50 ms` | `50–100 ms` | `>100 ms` |
| 20 Hz derived row gap | `<=120 ms` | — | `>120 ms`；`>250 ms` 为结构性失败 |

这些数值是数据质量和因果对齐门槛，不是执行器安全极限。现场如果只能通过放宽门槛才能保留数据，应先修复时钟、相机或 bridge 链路，不能用 field override 把链路故障改写成 clean。

### 11.2 离线 QC

离线管线至少输出：

- raw run 完整性和 checksum 报告；
- 50 Hz 对齐与 20 Hz resample 报告；
- ready/dump/goal/return 事件顺序；
- A/B 当前侧、目标侧和 realized side 分布；
- scripted target 与 realized target 混淆矩阵；
- 四种 transition 按 split、block、cycle index 的数量；
- P0/P1 模板按 split 和 block 的数量；
- goal commit 到本 cycle 首个操作意图及首个 effective action 的两段时延；
- cycle index、土面变化和 transition label 的关联审计；
- clean、review、failure 和 excluded 数量及原因；
- 四路视觉 domain 摘要；
- workface/reset、时间和目标是否存在明显绑定。

### 11.3 分项统计与证据口径

四种 transition 必须分别报告数量、误差和成功结果。A/B 也必须分别报告 expert repeat 分布。

endpoint 不计算到某个固定 A/B 中心的角度误差，统一使用目标侧裕量：

```text
target_sign(A) = -1
target_sign(B) = +1
side_margin = target_sign(scripted_target) * home_side_coordinate_rad
```

standalone repeat 和 composition 使用同一量纲，并按 `current_side × target_side` 分组：

```text
standalone_repeat_error = abs(
    standalone_side_margin - median(matched_standalone_side_margin)
)

composition_penalty = max(
    0,
    median(matched_standalone_side_margin) - continuous_side_margin
)
```

历史 home 重复误差只支持首轮临时门槛：每个连续 endpoint 仍须满足 clean 和 demonstrated support，且 `P95(composition_penalty) <= max(0.03 rad, P95(standalone_repeat_error))`。正式数值由新 A/B standalone expert/policy 重复数据生成，并在读取 locked test 前冻结；没有 matched standalone 时只报告原始 side margin，不计算替代性 penalty。

任何 coverage 结论都按明确的 current side、target side、transition、source block 和独立 calibration source 计算。样本量不足时保留原型证据等级。

## 12. 训练计划

### 12.1 共同配置

第一版尽量保持 N5 已验证的主合同：

- 四相机顺序：`video4, video5, video6, video7`；
- 图像 transform 与现场推理一致；
- policy rate：20 Hz；
- ACT chunk：20 steps；
- 原低维输入：qpos[4]；
- 新增输入：`real_transition_condition_v1[2]`；
- qvel 记录但不进入第一版 policy；
- action order：`swing, boom, stick, bucket`；
- qpos/action 使用 train-only normalization；`target_side_code` 保持固定 `-1/+1`，不再归一化；
- source-block split；
- validation loss 只用于 checkpoint selection。

### 12.2 从 N5 warm-start

新模型从已完成真机整铲的 N5 初始化：

1. 复制图像 backbone、Transformer、action head 和原 qpos projection；
2. 扩大 low-dimensional projection；
3. 原 qpos 列原样复制；
4. 新 condition 列初始化为 0；
5. 验证 `goal_active=0` 时，新模型初始输出与 N5 满足下述 `1e-5 / 1e-6` 数值门槛。

retention equivalence 使用相同 checkpoint、FP32/backend、图像、qpos、模型缓存和至少 100 个 held-out source frame，同时比较 raw action chunk 与 temporal aggregate action。通过条件为：最大绝对差 `<=1e-5`、平均绝对差 `<=1e-6`，且按当天 directional mechanical deadzone 计算的动作分类零分歧。任何一项不满足都先修复 warm-start，不进入 B1 训练或真机测试。

这样做的目的，是保留已有完整一铲能力，把首轮学习重点放在目标分岔和 transition 上。

### 12.3 对照组

| 名称 | 数据与初始化 | Condition | 用途 |
|---|---|---|---|
| R0 / N5 retention | 原 N5，不用新数据训练 | 无 | 检查原完整一铲能力和输入链没有退化 |
| B0 | 同一新数据、同一 split、同一 warm-start | 全程 `[0,0]` | 判断仅增加多铲数据能否解释结果 |
| B1 | 同一新数据、同一 split、同一 warm-start | cycle 开始时提交正确 target side | 主 Conditioned ACT |
| B2 | 同一新数据、同一 split、同一 warm-start | train 内把每个 cycle 的 A/B target side 对调 | 判断模型是否真正使用目标关联 |

B2 以完整 cycle 为单位，把 train 中 A 的 `target_side_code` 写成 B、B 写成 A。一个 cycle 的全部训练 row 使用同一个错配目标。由于每个 current side 下 A/B 目标数量相等，这个操作保持 current-side、target-side、P0/P1、cycle index 和 source split 的边际数量不变。Validation 和 locked test 使用正确目标。

B0、B1、B2 使用同一组预先冻结的 training seeds。每组至少运行三个 seed；如果只有单 seed 结果，只能作为工程 smoke，不用它区分 B1 和 B2。

B0、B2 只用于离线和 shadow 诊断，不授权真机动作。进入真机控制的候选只能是通过全部前置 gate 的 B1。

### 12.4 连续运行 lifecycle

- policy、temporal aggregation 和模型缓存只在 run 开始重置；
- cycle 结束时清除旧 goal，并增加 `goal_epoch`；
- 下一 goal 只有在新的 `goal_commit` 后生效；
- cycle 边界不消费下一条命令两次；
- 失败时停止，不递增成功 cycle，不提前取下一目标；
- 每个 raw chunk 记录生成它的 `goal_epoch` 和 condition snapshot；
- 日志分别保存 raw chunk、temporal aggregate action、runtime-safe action 和最终 commanded action；
- 每个 aggregate action 记录各 goal epoch 的归一化权重和 `old_epoch_weight_mass`。

`goal_commit` 之前生成的 chunk 继续参与 temporal aggregation。通过 chunk epoch 和最终 commanded action 计算目标提交到行为分岔的真实延迟。

当前 20-step chunk、20 Hz、指数系数 `k=0.01` 且聚合队列已填满时，goal 切换后的旧 epoch 权重基线为：

| goal commit 后时间 | old epoch weight mass |
|---:|---:|
| `0.00 s` | `0.9546` |
| `0.45 s` | `0.5250` |
| `0.70 s` | `0.2691` |
| `0.95 s` | `0` |

因此前约 0.5 s 仍由旧 goal chunk 占多数，约 0.95 s 后旧 chunk 才完全消失。condition latency 必须同时报告 goal commit、new-epoch 权重过半和旧权重归零三个时刻，不能把第一帧聚合动作完全归因于新目标。首次 moving test 在 `old_epoch_weight_mass <=0.5` 后不允许继续出现有效的目标反向 commanded action；更早的反向量记录为 transition debt，不据此伪装成纯新 goal 响应。

这组数值随 chunk、policy rate、聚合系数或队列策略变化而变化。任一配置改变时必须在 resolved config 中重新计算权重表；是否改为等待、epoch mask 或 cycle reset 属于新的 lifecycle 实验，不能在 v2.0.1 run 中临时切换。

## 13. 原型验收

### 13.1 Data Gate

进入训练前必须满足：

- 目标数量和四种 transition 覆盖符合 split manifest；
- train/validation/test source block 无重叠；
- 每个 clean cycle 都有 prospective goal；
- 没有 cross-cycle action supervision；
- 所有主文件 checksum、相机解码和数据 schema 通过；
- A/B ready rule contract 和 sequence manifest 已冻结。

### 13.2 Retention Gate

B1 的 condition 从 cycle 开始即有效。完整一铲的 dig、carry、dump 不能相对 N5 出现明显能力退化；目标侧造成的主要方向差异应与实际返回需求一致。

warm-start 首先通过第 12.2 节的 `1e-5 / 1e-6 / 零 deadzone 分类分歧` 数值等价门槛。随后根据 N5 和师傅在本次 A/B 数据上的重复波动冻结物理 retention 阈值，不能直接复用 SimVerify 或旧场地阈值。

### 13.3 Condition Gate

condition 实验分为三层：

1. **训练对照**：R0、B0、B1、B2 使用同一数据、split、warm-start 和 seeds，分别比较全 cycle 以及 dig、carry、dump、return 分段指标。主比较是 B1 对 B0/B2，不能只看总体 validation loss。
2. **离线目标干预**：固定 checkpoint、完整 observation/history、模型缓存和 cycle 起点，只把 `target_side_code` 从 A 换成 B。比较首个 action chunk、各阶段输出、聚合后 action 和预测轨迹。这一步只证明 policy 对 condition 敏感。
3. **真机原子 transition**：从相近的 A 起始状态分别提交 A、B，从相近的 B 起始状态分别提交 A、B。四种 transition 分开统计 endpoint、完整一铲效果、安全终止和重复波动。这一步证明上层改变当前目标时，执行结果能随之改变。

在相近 current-ready 和相同 checkpoint 下比较：

```text
目标 A 与目标 B 的 goal-active 动作差异和终点差异
>
同一目标重复执行的自然波动
```

同时要求：

- condition 无效时，A/B 替换不改变输出；
- condition 有效时，返回阶段的 swing 方向与目标侧一致；
- B1 的目标语义效果大于 B2 flipped-label control；
- endpoint 位于正确侧、排除带之外，并落在当天已示教支持范围内；
- scripted target 和 realized target 分开报告。

按首轮基线，同一 matched current-ready 下，目标 A 和目标 B 的 realized coordinate 中位数必须分别位于 `<=-0.08 rad` 和 `>=+0.08 rad` 的 clean 区域，二者至少相隔 `0.16 rad`，且目标间差异大于两个同目标 standalone repeat P95 中的较大者。现场解析出的 clean threshold 更大时，最小间隔同步改为其两倍。

P0/P1 连续成功只进入 Composition Gate。它不能替代上述 Condition Gate，也不用于声称 ACT 可以自行规划下一侧或执行未测试的 planner 顺序。

### 13.4 Composition Gate

真机闭环按安全顺序推进：

1. B1 shadow；
2. A→A、A→B、B→A、B→B 单独 transition；
3. 连续四 cycle；
4. 在独立 source block 上分别执行 P0 与 P1 固定 planner。

连续 run 中：

- 只在 run start 重置一次 policy；
- cycle 边界不人工摆位；
- 每个 cycle 的 realized target 等于 scripted target；
- 每铲保持可接受的完整挖掘和卸料效果；
- 无人工接管、安全终止和 lifecycle 错误；
- 连续执行相对 matched standalone 的额外误差单独报告为 composition penalty。

每个连续 endpoint 必须先通过 correct side、clean margin、联合 ready 参考和视觉复核，再应用第 11.3 节的 composition gate。首轮门槛为 `P95(composition_penalty) <= max(0.03 rad, P95(standalone_repeat_error))`；当天 standalone 数据产生更严格门槛时使用更严格值。

正式真机尝试次数、resolved endpoint 门槛和停止职责必须在读取 held-out test 之前写入带 checksum 的 test authorization。locked test 不得用于调阈值或选择重试次数。少量成功只能称为原型证据，不能写成统计可靠性或部署保证。

### 13.5 结论路由

| 观察结果 | 结论 |
|---|---|
| 单个 transition 就失败 | 原子动作、目标表示或数据支持问题 |
| 单个都成功，连续 run 失败 | cycle 衔接、lifecycle 或累计状态问题 |
| 人工边界成功，自动边界失败 | ready/dump detector 问题 |
| B1 与 B2 相近 | condition association 未被可靠使用 |
| B0、B2 与 B1 都能完成固定循环 | 可能依赖观测中的顺序或土面线索，不能证明 condition 有效 |
| 终点正确但后续空铲 | 物理效果、土面状态或观察能力问题 |
| locked source block 失败 | transition 对新土面状态或新 source run 的支持不足 |

## 14. 需要开发的程序

本计划的代码改动集中在真实栈内的任务程序、数据链、离线处理和 policy runtime，工作量按中等规模规划。

### 14.1 现场端

- A（左）/B（右）target sequencer 和简洁显示界面；
- 固定 P0/P1 脚本 planner、run 顺序和 sequence manifest；
- `goal_commit`、ready、abort 和 intervention 事件记录；
- goal 的 recorder/router/display 三方 commit acknowledgement；
- ready 与 return shadow candidate 事件及人工 marker 偏差；
- raw HDF5 与 task event 的统一 step/time 对齐；
- run/block/session manifest 和 checksum；
- historical baseline 到 field-resolved 参数的解析和覆盖记录；
- 每条 run 的即时结构 QC。

目标程序只显示任务和记录事件，不直接控制执行器。

### 14.2 离线端

- home-side calibration、排除带和已示教支持范围审计；
- continuous run 的 ready/dump/return candidate labeler；
- 人工复核界面或复核清单；
- 50 Hz → 20 Hz resample；
- ready-to-ready cycle materializer；
- condition lifecycle 和 chunk valid mask；
- source-block split、coverage report 和 checksum package；
- B0/B1/B2 配置生成与 target-side flip control；
- endpoint、condition effect、composition penalty 和失败归因报告；
- 时序 QC 带、最近联合 ready 参考距离和边界局部 gap 报告。

### 14.3 Policy runtime

- qpos-only N5 projection 扩展为 qpos + 2D condition；
- 新 condition 列零初始化和 retention equivalence 检查；
- goal router / epoch / cycle lifecycle；
- condition、raw chunk、epoch weight mass、temporal aggregation、safe action、commanded action 分层日志；
- 轴向动作上限、deadzone 后响应计时、反向响应和 return 错向位移停止；
- fail-closed：目标缺失、过期、边界不确定或状态异常时停止，不猜下一目标。

第一版 ready detector 先 shadow，人工确认仍是正式证据 owner。自动 detector 达到独立真机证据后，再让它接管 cycle 切换。

## 15. 必须交付的产物

### 15.1 数采前

- `ready_contract.json`；
- `sequence_manifest.json`；
- `split_manifest.json`；
- 当天 resolved record config；
- N5 bundle manifest、resolved inference config 和 SHA256；
- B1 moving test authorization，包括动作上限、级别、责任人和停止条件；
- 现场任务卡和停止条件；
- 数据盘空间、路径和 checksum 预检结果。

### 15.2 数采后

- 只读 raw run packages；
- raw QC 和 failure inventory；
- immutable 20 Hz ready-to-ready dataset；
- cycle、transition、上下文和 split coverage 报告；
- clean/review/failure manifest；
- dataset 和 source provenance checksums。

### 15.3 训练后

- R0/B0/B1/B2 resolved configs；
- checkpoint、dataset stats、run metadata 和 SHA256；
- retention、offline condition、shadow 和 closed-loop 报告；
- 是否进入大规模数采的明确 decision artifact。

## 16. 原型通过后的扩展方向

v2.0.1 通过后，后续大规模数据继续沿用同一原始合同，逐步扩大：

- 更多 planner 顺序及其组合验证；
- A/B 内更细的目标区域或连续目标表示；
- 增加 reach 目标维度；
- 不同 workface、土体状态、光照和场地；
- 对数据密度低或 OOD 的目标 fail closed；
- 根据真实失败类型自适应补录。

扩展数据保持 `goal_source`、prospective command、goal epoch 和 source provenance 完整。

## 17. 基线参数与现场标定覆盖

本文不再保留无归属的待定参数。执行时每个参数只有三种状态：直接沿用 historical baseline、根据当天数据解析 field-resolved 值，或因证据不足使对应 gate 失败。不能用口头约定或运行中的临时调参形成第四种状态。

### 17.1 参数解析表

| 参数 | Historical baseline | 现场解析和允许覆盖 | 最终 owner |
|---|---|---|---|
| `home_swing_axis_rad` | `0.000690 rad` | 本 session 固定；改变必须生成新合同 | `ready_contract.json` |
| `left_qpos_sign` | `-1`（左减右增） | 本 session 固定；与现场物理方向不符时停止 | `ready_contract.json` |
| `home_tolerance_rad` | `0.05 rad` | 本 session 固定；home 不可标为 ready | `ready_contract.json` |
| clean ready threshold | `0.08 rad` | 本 session 固定；`0.05–0.08 rad` 为过渡区 | `ready_contract.json` |
| swing safe range | `[-0.3892, +0.4189] rad` | 只允许收紧；任何扩张都需要新的机械安全证据和合同版本 | `ready_contract.json` |
| ready stable window | 连续 0.5 s、全窗 `abs(swing_qvel)<=0.015 rad/s`、最大样本间隔 0.15 s | boom/stick/bucket qpos/qvel 只记录；门控放宽必须有独立证据和新合同 | `ready_contract.json` |
| ready人工证据 | 铲斗离土确认 + 操作员确认 | 两项均不可省略，initial/target 共用 | task event + `ready_contract.json` |
| return candidate | swing action `0.08`、连续 6 row、回溯 5 row、复核窗 ±2.5 s | 可以根据人工 marker 偏差更新 detector；修改后提升 annotation 版本 | detector config + annotation manifest |
| 时序和 gap QC | 第 11.1 节表格 | 可以收紧；放宽会生成新的 QC/dataset 版本，不能回写旧 clean 结论 | resolved record/dataset config |
| run 间土面处理 | 恢复土面或切换相近工作条带都允许 | 不需要选成唯一模式；每次操作记录 `workface_reset_id` 和方式 | `run_manifest.json` |
| N5 warm-start bundle | 无现场默认 bundle | 上机前冻结实际 checkpoint、stats、resolved config、commit 和 SHA256 | session manifest |
| endpoint/repeat/composition | clean `0.08 rad`；临时 composition 门槛为 `max(0.03 rad, repeat P95)` | 用 matched standalone 数据解析，并在读取 locked test 前冻结 | evaluation config + test authorization |
| locked test 读取 | 默认锁定 | 只有训练方法、seeds、checkpoint 选择、endpoint/composition 阈值和重试次数全部写入带 checksum 的 decision artifact 后才解锁 | split manifest + decision artifact |
| B1 首轮 moving action | `[0.80,0.70,0.70,0.70]`，第 9.4 节响应停止条件 | 先 shadow 和 standalone；提高上限必须有上一级通过记录和 field override | resolved inference config + test authorization |

### 17.2 变更与版本规则

- 第一条正式 run 之前，按本节解析并冻结 field-resolved 值，不提升数据合同版本；`context_version`、resolved config 和所有 checksum 必须更新。
- 正式录制开始后，home 中线、左右符号、安全范围、相机位姿或 ready 物理语义发生变化时，关闭当前 block，创建新的 context 和 ready contract。不同 context 不混在同一个 block。
- raw 封存后调整 ready、dump 或 return detector，只发布新的 annotation 和派生 dataset 版本；原始 HDF5 与 `task_events.jsonl` 保持不变。
- 调整 QC 门槛、resample offset、边界 mask 或 clean/review 规则，发布新的 QC 和 dataset 版本，并重新生成 checksum。旧版本结论保留，不覆盖。
- 改变 condition 编码、目标语义、policy 输入、action 语义、cycle 定义、source split 或 planner prospective-command 规则，必须提升数据合同版本，不能继续使用 `v2.0.1-real-transition`。
- 任一 resolved 参数在 run package 中找不到 owner 文件或 checksum 时，该 run 不能进入 clean；不能从当前代码默认值反推当时现场实际值。

## 附录 A. 代码依据与证据边界

本计划的现有基础来自当前 checkout 中的以下合同：

- `testbed/testbed/data/schema.py`：当前真机 HDF5 schema 为 v1.2，已定义 qpos、qvel、action、step/time、action source 和 diagnostics 边界；
- `testbed/testbed/configs/teleop_real_v1.yaml`：当前真机 teleop 主配置为四路 `video4..video7`、50 Hz control/record 和 JPEG 录制；
- `testbed/testbed/configs/act_real_gmsl_four_camera_qpos_v1.yaml`：当前四相机 qpos-only ACT 训练基线为 20 Hz、20-step chunk、4D qpos；
- `testbed/testbed/configs/policy_real_gmsl_fourcam_g49_n5_control_v1.yaml`：当前 G49 N5 控制配置保留 50 Hz control pump、20 Hz policy 采样和 temporal aggregation。
- `testbed/testbed/data/resample_20hz.py`：当前派生链选择每个目标时刻之后的第一条 source observation，并使用 `-20 ms` action label offset；
- `testbed/testbed/policies/act/adapter.py`：当前 temporal aggregation 的 chunk 长度、指数权重和跨 step 聚合语义；第 12.4 节的 epoch 权重由这条当前代码路径计算。

v2.0.0 SimVerify 树中的 `testbed/testbed/configs/simverify_sector_geometry_physical_v1.json` 提供了 physical center、left、right 的方向语义。本计划据此使用以 center 区分左右的物理语义。

旧配置中的 `home_pose_rad[0]=-0.007831 rad` 只属于旧版 go-home 配置，不再拥有
v2 ready 分类语义。本轮 `ready_contract.json` 明确冻结现场确认的 swing home
`0.000690 rad`、左减右增、home 容差、clean 阈值和安全范围；两者不能混用。

上述文件只证明仓库内已有的数据和运行边界。它们不证明当前现场 Jetson 已部署同一 commit，也不证明相机、CAN、IMU、时钟或 N5 bundle 当天可用。这些状态必须在数采前用 resolved config、运行日志和 checksum 现场确认。

commit `a64e5d1` 实现了旧版 A/B sequencer、P0/P1 四-cycle manifest、home-side contract 生成、在线 task event、录制控制服务、raw HDF5 对齐、run package 封存和 checksum 验证。当前 `fs/v2.0.1` 工作分支已在此基础上实现 seeded multi-sequence v2、可变 cycle runtime、配对顺序对消和 realized-target mismatch fail-closed。旧版 v1 只保留只读验证，不能启动新录制。

离线 cycle annotation/materializer、`real_transition_condition_v1` 训练数组、B0/B1/B2 训练对照、N5 condition warm-start 和真机 conditioned-policy goal lifecycle 仍未实现。因此当前代码只能作为旧版专家数据录制支架，不能训练或执行本轮目标中的 Conditioned ACT。N5 完成真机整铲是本计划使用的既有实验前提，本次文档修订没有连接真机或重复该验证。

## 附录 B. Historical baseline 的数据依据

本次基线来自以下只读本地历史产物：

- N5 source split 和响应包络：`/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/response_envelope_all_axis_train120_val20_v1/`，包含 120 train 和 20 validation 源 episode；
- 原始 Pro 真机数据：`/data/pingfan/Excavator_real_stack_data/pro_real_teleop_20260713_usb_raw/`；
- 20 Hz 派生数据：`/data/pingfan/Excavator_real_stack_data/pro_real_teleop_20260713_20hz_v1/`；
- 29 条 return 边界快速复核记录：`/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/e58_semantic_cut_codex_reviewed.csv` 的 24 条，以及 `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/e59_semantic_cut_105_109_codex_reviewed.csv` 的 5 条。

关键统计与本文参数的对应关系为：

| 历史观测 | 统计 | 支持的本文参数 |
|---|---|---|
| 140 条 N5 source episode 回中终点 | 相对历史中位数的 swing 绝对偏差 P95 `0.0269 rad`、最大 `0.0366 rad`；代码候选相对历史中位数偏 `0.0132 rad` | deadband `0.05 rad`、clean threshold `0.08 rad`；不授权直接使用代码 home 候选 |
| 3975 个历史静止 0.5 s 窗口 | qvel 稳态噪声约 `[0.00763,0.00716,0.00903,0.00896] rad/s`；qpos 峰峰值 P99 不超过约 `0.00023 rad` | swing ready 阈值的历史旁证；非 swing 三轴不作为门控 |
| 29 条人工周期快速复核 | 持续 swing intent 到 `0.005 rad` 位移 P95 `2.11 s`；完整周期 P95 `40.2 s`；return 段 P95 `9.5 s` | return action candidate、±2.5 s 复核窗和 45/60、12/20 s 时间带 |
| 140 条源 episode 的时序诊断 | action latency P99/P99.9 `38.0/47.1 ms`；bridge age P99.9 `27.7 ms`；sync skew P99/P99.9 `49.2/61.4 ms`；camera age P99/P99.9 `94.3/112.0 ms` | 第 11.1 节 action、bridge、sync 和 camera QC 带 |
| 同批四相机与 row 间隔 | group skew P99.9 `0.112 ms`；raw row gap P99/P99.9 `45.0/51.5 ms`；20 Hz gap P99 约 `94.7–96.5 ms`、P99.9 `110.1–116.4 ms` | group skew 和 50/20 Hz gap 门槛 |
| 120 条训练源 episode 动作支持 | action 绝对值 P99 约 `[0.799,0.696,0.685,0.699]`；观测最大值约 `[0.800,0.798,0.799,0.700]` | B1 首轮 `[0.80,0.70,0.70,0.70]` 和后续支持内上限 |
| 历史 from-rest swing response | 越过 mechanical deadzone 后绝大多数在 3 tick、全部训练样本在 5 tick 内出现响应 | 0.25 s 无响应停止和 `0.015 rad/s` 响应阈值 |

这些统计是离线历史证据，不是当前现场状态。29 条 return 边界只做过快速视觉复核，不是操作员安全真值；历史数据也没有本合同下的 A/B 外边界、物理左右、无复位四铲和 composition penalty。相应字段必须按第 17 节现场解析，不能因已有基线而跳过上机标定。
