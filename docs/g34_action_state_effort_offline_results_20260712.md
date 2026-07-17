# [Execution target G34/H1: Action-state/effort ACT 离线完整结果]

日期：2026-07-12。结论：**拒绝 G34 替换现有 raw ACT + mechanical assist；保留候选仅作训练研究，不进入 held-out 或真机部署。**

## [Execution target G34/H1: 目标与边界]

本轮把专家 direct-domain action 按每个轴拆成五类：`idle`、`pos_near`、`pos_safe`、`neg_near`、`neg_safe`。连续 ACT action 仍是唯一运行时 command source of truth；分类 head 只提供辅助监督，不能在推理时 argmax 覆盖、投影或 gate 连续动作。

新增的 direct-domain margin loss 只在专家动作已经越过机械死区的轴上要求模型留出 `0.02` 安全余量，并对持续有效动作加权。stick 在本任务中保持结构性零轴，不被当成缺失数据。源 HDF5 未修改。

## [Execution target G34/H1: 标签审计与数据证据]

正式 split 使用 19 个 train episode、5 个 validation episode，共 24 个 episode、16,529 步、66,116 个有效轴向行、13,058 个专家有效动作机会。held-out `105..109` 未读取、未训练、未调参、未评估。

标签聚合如下：

| axis | idle | pos_near | pos_safe | neg_near | neg_safe |
| --- | ---: | ---: | ---: | ---: | ---: |
| swing | 12,050 | 45 | 2,171 | 56 | 2,207 |
| boom | 13,385 | 107 | 1,928 | 41 | 1,068 |
| stick | 16,529 | 0 | 0 | 0 | 0 |
| bucket | 11,094 | 79 | 2,510 | 117 | 2,729 |

审计 artifact：

- [action_state_effort_label_audit.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_label_audit/action_state_effort_label_audit.json)，SHA256 `d2d634ad4607d11cc46a06a14e3fe7fc4c82da0926ef399ee4450d3a42d0d550`。
- [episode_summary.csv](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_label_audit/episode_summary.csv)。
- [axis_state_counts.csv](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_label_audit/axis_state_counts.csv)。

## [Execution target G34/H1: 训练与 bundle 证据]

使用 raw ACT baseline checkpoint 初始化，只允许新增 `action_state_head` 缺失；训练 200 epoch，best epoch `195`，validation loss `0.25937366485595703`。训练完成信号为 `run_metadata.status=completed`。

- 配置：[act_real_gmsl_eye2_action_state_effort_v1.yaml](/home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed/testbed/configs/act_real_gmsl_eye2_action_state_effort_v1.yaml)。
- bundle：[g34_action_state_effort_formal/ckpt](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_formal/ckpt)。
- [policy_best.ckpt](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_formal/ckpt/policy_best.ckpt)，SHA256 `768a54ea4eb5df3873e52ee13f16ce6bc51652d7b1377d101ffd4eeacebb6240`。
- [resolved_config.yaml](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_formal/ckpt/resolved_config.yaml)，SHA256 `6c618125ed468affadccab71960bfc0d8416b9c552edfcc0941d5d9fdefbafc3`。
- split SHA256 `09fe85bdab539ca2a12b5b4613f507ea009706cb38077b46e168f5171da59a3d`。

## [Execution target G34/H1: 24 episode open-loop 全量结果]

全量 replay 覆盖 24 episode、16,529 步，使用 identity policy action scale `[1,1,1,1]`，temporal aggregation 保持开启。

- overall MAE `0.06693`，RMSE `0.13854`；policy p95 absolute action `0.89234`，max `0.93947`。
- 作为对照，G28 open-loop MAE 为 `0.04843`；因此 G34 的连续动作拟合反而变差。
- start40 的 policy effective 占比 `87.92%`，而专家只有 `5.83%`；extra/wrong 为 `82.08%`，说明模型在真正启动前大量提前输出 bucket+。
- full available 的 same-axis/same-direction effective rate `95.23%`，extra/wrong `19.67%`。
- tail80 extra effective frames `148`，release-window extra effective frames `528`；clip violations `0`、non-finite `0`、最大同时有效轴数 `2`。

完整 open-loop artifact：[collection_summary.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_formal/open_loop_all24/collection_summary.json)。

## [Execution target G34/H1: 轨迹死区 liveness 对比]

同一 13,058 个专家有效机会、228 个有效段上的结果：

| pipeline | current same-dir / hit-1 | persistent-4 | persistent-20 | underconfidence | extra/wrong | zero segments@max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw ACT | 87.49% | 88.39% | 94.33% | 12.34% | 4.46% | 18 |
| raw ACT + assist | 97.71% | 97.87% | 98.84% | 2.12% | 11.48% | 4 |
| G28 raw | 95.24% | 95.12% | 96.12% | 4.61% | 4.78% | 14 |
| G28 + assist | **99.03%** | **99.07%** | **99.33%** | **0.83%** | 14.82% | **2** |
| G34 raw | 92.92% | 92.99% | 95.63% | 6.81% | 6.28% | 15 |
| G34 + assist | 97.79% | 97.79% | 98.06% | 1.94% | 13.84% | 13 |

G34 + assist 只比 raw ACT + assist 略高 `0.08` 个百分点，却比 G28 + assist 少 `1.23` 个百分点，并把 max-horizon 零命中段从 4 个增加到 13 个，不能称为根本提升。

逐 opportunity artifact：[trajectory_deadzone_liveness_report.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_formal/trajectory_deadzone_liveness/trajectory_deadzone_liveness_report.json)。

## [Execution target G34/H1: recursive state-hold 死锁与启动结果]

在正式 validation 5 episode、48 anchors、hold horizon 20 的 recursive state-hold 测试中，保持 qpos/image 并将 qvel 置零；同时运行 no-assist 与 mechanical assist，并追踪恢复后的完整 horizon。

| pipeline | recovered | deadlocked | hidden by teacher forcing | unexpected anchors | unexpected ticks |
| --- | ---: | ---: | ---: | ---: | ---: |
| G34 raw | 38/48 | 10 | 4 | 2 | 26 |
| G34 raw + assist | **43/48** | **5** | **1** | 7 | 75 |
| G28 raw + assist | **45/48** | **3** | 2 | 7 | 84 |

5 个正式 startup bucket+ anchors 中，G34 raw + assist 为 `5/5`；但中途仍有 5 个死锁，故启动修复不等于整体成功。5 个 recursive deadlock anchor 是 `episode_94 boom+`、`episode_74 boom+`（两个 matched anchor）、`episode_74 bucket-`、`episode_92 bucket-`；这说明问题已从启动扩散到中途幅度/相位。

state-hold artifact：[run_summary.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_formal/state_hold_val5_h20/run_summary.json)。

## [Execution target G34/H1: complete offline gate]

已合并 open-loop、release/tail、gohome 可估计性、execution-monitor sidecar replay 和 recursive state-hold：

- [complete_offline_report.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g34_action_state_effort_formal/complete_offline_report/complete_offline_report.json)。
- gate：`passed=false`。
- 失败原因：要求 assist recovered `>40` 且 hidden `=0`；G34 为 `43/48` recovered、`5/48` deadlocked、`1` hidden。
- `105..109` 明确未评估；没有因为 validation 失败而触碰 held-out。

## [Execution target G34/H1: 根因反思与回退]

这轮验证了“把低于死区和刚越过死区分开”在标签层面是可行的，但现有 teleop 数据并没有让该辅助分支自动变成更可靠的连续动作。实际结果显示：分类/边界监督共享了 ACT 表征后，模型在启动前更倾向于输出 bucket+，start40 extra/wrong 达 `82.08%`；同时连续 MAE、轨迹零命中段和 recursive deadlock 都没有超过 G28。

因此本轮 callback 是：

1. **不把 action-state head 接入运行时 gate、argmax 或 action projection**；否则会重演 H3/E52 的错误方向和相位锁死。
2. **候选回退到已验证的 raw ACT + mechanical assist**（40/45 held-out、startup 5/5 的既有 reference）；G34 checkpoint 只保留作研究和后续表示学习，不部署。
3. 现有数据足以支持标签统计、辅助训练和离线 falsification，但不足以证明“低死区 proposal 被真实执行后一定能推进”。要把分类结果用于更强的 eligibility/retry，仍需要 sent-command timestamp、实际 qpos/qvel response 和 policy-on correction/失败标签；当前 execution sidecar 只能做低层 response audit，不能校准 retry precision。

本轮代码与测试只在目标 worktree 修改；未连接 Jetson、未写现场 runtime/default config、未使用外部 USB 硬盘、未修改原始 HDF5。
