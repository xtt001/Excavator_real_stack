# [Execution target G35/H1: Current-step action-state ACT 与任务级评估结果]

日期：2026-07-12。结论：**G35 修复了 G34 的 hidden-by-teacher-forcing，但仍不能替换 raw ACT + mechanical assist。**

## [Execution target G35/H1: 本轮改动]

G35 保留连续 ACT action 为唯一运行时 command source，只把 action-state/effort 辅助损失限制到第一个 decoder query（当前动作）。未来 19 个 chunk query 仍只接受连续 ACT imitation，不再用未来的 bucket state 直接训练当前状态头。

另外新增了一个独立的 `observed_qpos_boundary_proxy` 审计：边界由 train split 的 qpos 分位数拟合，且明确标记 `physical_limit_ground_truth=false`。它只用于分析哪些 extra action 可能是限位豁免，**不改变严格 state-hold gate**。

## [Execution target G35/H1: 训练证据]

- 200 epoch 完成，best epoch `199`，validation loss `0.2066754251718521`。
- [配置](/home/pingfan/Excavator_real_stack_e52_deadlock_eval/testbed/testbed/configs/act_real_gmsl_eye2_current_step_action_state_v1.yaml)。
- [checkpoint](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g35_current_step_action_state_formal/ckpt/policy_best.ckpt)，SHA256 `7a845778d3fed2f89989a260e03352eb06cc1277f34b5c35a2541c66383ce891`。

## [Execution target G35/H1: 24 episode open-loop]

G35 的连续拟合比 G34 好，但仍不如 G28：

| model | MAE | RMSE | policy p95 |
| --- | ---: | ---: | ---: |
| G28 | 0.04843 | — | — |
| G34 | 0.06693 | 0.13854 | 0.89234 |
| G35 | **0.05928** | **0.12812** | **0.82876** |

这说明 current-step 监督减少了部分过度激活，但没有保证主轴在需要时越过死区。

## [Execution target G35/H1: 轨迹 liveness 与安全权衡]

| pipeline | hit-1 | persistent-20 | underconfidence | extra/wrong | zero segments@max |
| --- | ---: | ---: | ---: | ---: | ---: |
| G34 raw + assist | 97.79% | 98.06% | 1.94% | 13.84% | 13 |
| G35 raw + assist | 96.55% | 97.33% | 3.12% | **11.83%** | 13 |

G35 的 extra/wrong、tail 和 release extra 有所下降，但 liveness 同时下降；这是“更保守但更容易不动”，不是信心提升。

## [Execution target G35/H1: recursive state-hold]

正式 validation 5 episode、48 anchors、hold horizon 20：

| pipeline | recovered | deadlocked | hidden | startup |
| --- | ---: | ---: | ---: | ---: |
| G34 raw + assist | 43/48 | 5 | 1 | 5/5 |
| G35 raw + assist | 43/48 | 5 | **0** | 5/5 |

G35 消除了 hidden anchor，但没有减少真实 state-hold deadlock，因此仍不能进入 live control。完整报告：[complete_offline_report.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g35_current_step_action_state_formal/complete_offline_report/complete_offline_report.json)。

## [Execution target G35/H1: 限位/目标级审计]

G35 raw 在 train-only boundary proxy 下：

- 13,058 个目标机会中，65 个被标记为可能的 observed-boundary exemption；
- 3,682 个 extra effective action 中，50 个可能是 boundary-exempt，3,632 个仍是非豁免 extra；
- 该结果不能证明真实物理限位，只能说明“把所有 wrong bucket 都判失败”确实过于粗糙。

审计 artifact：[task_goal_liveness_report.json](/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260712/g35_current_step_action_state_formal/task_goal_liveness/task_goal_liveness_report.json)。

## [Execution target G35/H1: callback 与回退]

G35 的结论是：

1. 当前时刻监督确实降低了部分提前/额外动作，并把 hidden 从 `1` 降到 `0`；
2. 但它同时增加了 underconfidence，主轴 deadlock 没有减少；
3. 因此不能再继续仅靠 action-state loss 调权重；下一步必须把“主轴 qpos progress”和“真实 limit/terminal goal”纳入任务级目标；
4. 当前可部署参考仍是 raw ACT + mechanical assist，G35 checkpoint 不部署。

held-out `105..109` 未使用、未校准、未评估；原始 HDF5 未修改。
