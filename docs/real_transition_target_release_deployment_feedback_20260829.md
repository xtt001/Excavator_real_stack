# v2.0.1 Target Release 模型部署问题反馈（2026-08-29）

## 文档目的

本文只记录 `real_transition_target_release_v2` 模型在真机连续脚本中的实际表现、运行时
证据和已确认的软件行为。本文不包含修复建议、模型改造方案或后续训练设计。

## 部署对象

- 现场分支：`fs/v2.0.1`
- 现场运行代码基线：`7b2113d`，叠加本轮尚未提交的连续脚本与 landing 工程补丁
- 模型状态：`OFFLINE_ACCEPTED_FIELD_CANDIDATE`
- checkpoint：`policy_accepted.ckpt`
- checkpoint epoch：`199`
- checkpoint SHA-256：
  `28590633f2c8952acb6bd786a9d4424959f61e447897fdfd3a2adfca7533fabe`
- resolved config SHA-256：
  `bd933809d85c3f8862cc43914e33ec46e55782f37c6607d720c55ce39a35ae7d`
- 模型条件契约：`real_transition_condition_v1`
- 模型低维输入：`qpos`、`real_transition_condition_v1`
- ACT chunk size：`20`
- temporal aggregation：启用
- 每次提交新 goal 时重置模型状态：启用

## 现场日志

从端原始日志：

```text
/media/mundane/EXTERNAL_USB/policy_control_tests/
real_transition_target_release_control_20260829T043145.177195525Z/
real_transition_target_release_v2_field_20260829T043204.572508Z/steps.jsonl
```

日志 SHA-256：

```text
854148d6727f04c20424c2e0003ca0151a7cf2d18a9f5b801ad6c836fb67a848
```

日志共 `3544` 个 step。首个自动脚本从 B 出发，计划顺序为：

```text
B → A → B → A → B
```

## 预期动作语义

从 B 开始且本 cycle 目标为 A 时，一铲完整动作应包含右侧挖掘 excursion，随后左向回程并
稳定到达 A。到达 A 只表示本 cycle 的终点，不表示 cycle 可以省略前面的挖掘动作。

## 实际现象

首个自动脚本的 cycle 0、1 和 3 都出现了明显的右向 excursion 与左向回程。cycle 2 从
B 开始、目标为 A，但模型在 goal 切换后的第一个推理 step 立即给出较大的负向 swing
动作，机器从 B 直接向左移动到 A，没有形成与其他 cycle 相同的右侧挖掘 excursion。

该 cycle 仍被运行时判定为完成，随后脚本进入 cycle 3，并最终报告 `script_complete`。

同一进程中，完整四 cycle 脚本结束并解除锁零后，再次从 B 激活相同的 B 起始脚本。模型
再次在第一个推理 step 输出较大的负向 swing 动作。操作者随后使用 policy 按钮退出，停止
了这次动作。

## 关键时间线

| step | planner cycle | 当前目标 | swing qpos | swing qvel | 模型原始 swing action | 运行事件 |
|---:|---:|:---:|---:|---:|---:|:---|
| 1507 | 1 | B | +0.3370 | +0.0044 | -0.0398 | B 侧稳定窗末端 |
| 1508 | 2 | A | +0.3373 | +0.0057 | -0.8407 | `cycle_advanced` |
| 1514 | 2 | A | +0.3270 | -0.2020 | -0.8650 | 开始明显向左 |
| 1524 | 2 | A | +0.1700 | -0.4790 | -0.8650 | `cycle_excursion_confirmed` |
| 1535 | 2 | A | -0.1570 | -0.5010 | -0.8180 | 进入 A 侧附近 |
| 1580 | 3 | B | -0.2630 | +0.0070 | +0.0630 | `cycle_advanced` |
| 2155 | 4 | - | +0.3350 | -0.0040 | 0.0000 | `script_complete` |
| 3241 | 0 | A | +0.3322 | -0.0008 | -0.8781 | 第二次从 B 激活 |
| 3266 | 0 | A | -0.0720 | -0.4978 | - | `operator_toggle` |

cycle 2 只有 `72` 个 active step。与之相比，同一脚本中完成明显右向 excursion 和左向
回程的 cycle 0、1、3 分别有 `443`、`621`、`575` 个 active step。

## 两次 B 起始状态对照

日志中存在一次表现为完整 cycle 的 B→A，以及两次立即向左的 B→A。它们的 swing 都被
ready 逻辑识别为 B，但其他关节姿态不同。

| 场景 | step | qpos `[swing, boom, stick, bucket]` | 第一个模型 swing action |
|:---|---:|:---|---:|
| 首次 B→A | 444 | `[+0.2293, +0.0693, -0.6379, +1.9738]` | -0.0483 |
| cycle 2 B→A | 1508 | `[+0.3373, -0.3867, -0.6447, +1.9927]` | -0.8407 |
| 第二次激活 B→A | 3241 | `[+0.3322, -0.3672, -0.6425, +1.9865]` | -0.8781 |

ready 合同只用 swing 判定 A/B，boom、stick、bucket 不阻止 ready。因此上述三种姿态在
满足 swing 稳定窗时都可以作为 B 起始状态。

## 已确认的软件链路事实

1. 起始侧识别正确。日志中的 `planner_selected_initial_side` 为 `B`。
2. 脚本选择正确。B 起始脚本的第一个目标为 A。
3. 条件提交正确。异常 step 的 `planner_condition` 为 `[-1.0, 1.0]`，对应目标 A。
4. goal epoch 正常推进。cycle 2 使用 `goal_epoch=3`。
5. 配置启用了 `reset_policy_on_goal: true`；`commit_cycle_goal()` 在提交新目标时调用底层
   policy reset，清理 temporal aggregation、cached chunk 和视觉历史。
6. 异常负向值来自模型原始输出。step 1508 的 `policy_action`、`policy_scaled_action`、
   `policy_assisted_action`、`policy_returned_action` 和 `safe_action` 的 swing 均为
   `-0.8407`；该动作不是 landing 层或 PD 生成的。
7. 该运行没有记录 `policy_error`、guard trigger 或 scripted-cycle fault。
8. 当前 `real_transition_condition_v1` 在一个 materialized cycle 内保持常量，只包含
   `target_side_code` 和 `goal_active`。
9. 当前 excursion 判定使用相对 goal anchor 的 swing 绝对位移。阈值为 `0.08 rad`，连续
   `3` 个样本满足后即设置 `cycle_excursion_confirmed`；该判定不区分位移方向。
10. 当前模型包的 target-release 训练契约将 A 定义为 continue side、B 定义为 stop side；
    accepted model 中记录的 `continue_action_target_abs` 为 `0.7990936`。

## 部署影响

- 自动脚本报告的“完成 cycle 数”不等价于完成相同数量的完整挖掘动作。
- B 起始、目标 A 的 cycle 可能表现为直接左移到 A，并被现有状态机计作有效 excursion。
- 同一 B 侧 swing ready 状态在不同非 swing 姿态下产生了明显不同的首步 swing 输出。
- 单次脚本最终出现 `script_complete`，不能单独证明其中每个 cycle 都满足预期动作语义。
- 操作者在第二次异常激活后通过 `operator_toggle` 退出；日志未显示软件 fault 或自动 guard
  介入。

## 同轮附带记录

首个完整脚本中 swing qpos 最大值为 `+1.9684 rad`（step 1311）。该 step 的 ready 分类为
`outside_safe_range`，blocker 为
`swing_not_stable,swing_side_outside_safe_range`，但 `guard_triggered=0`。本文只记录该事实，
不对该值的机械含义或与本问题的因果关系作结论。

## 证据边界

- 本文结论来自一条逐 step 真机日志和同一进程内的第二次激活。
- 日志能够区分模型原始动作、landing 后动作、安全动作和最终下发动作。
- 本文没有对四路现场图像逐帧复核，也没有独立评价每一铲的土方效果。
- 非 swing 姿态与模型输出差异是已观测事实；单条日志不能单独证明训练数据分布或网络内部
  表征是其根因。
- 离线验收结果覆盖记录观测上的动作指标，不等价于真机闭环连续多 cycle 的动作语义验收。
