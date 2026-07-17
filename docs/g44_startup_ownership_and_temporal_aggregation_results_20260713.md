# [Execution target G44/H1: 本轮问题与结论]

本轮只使用现有 HDF5 和冻结 checkpoint，验证两个假设：录制开头的操作者等待是否被
错误地当作策略应该模仿的静止动作；ACT temporal ensemble 是否把本来能够越过机械
死区的新动作平均回死区。所有新增功能都是离线、显式 opt-in 诊断，选中的真实动作仍是
legacy temporal aggregate；没有修改部署默认配置、源 HDF5 或模型 action scale，也没有
读取 held-out `episode_105..109`。

结论分三层：

1. **startup 的逐帧 extra 指标语义确实有问题。**三种冻结模型在现有 24 条录制画面上
   都从第 0 帧输出有效 `bucket+`，不存在被开头静止帧压成不动的现象；只是旧指标把
   操作者真正动手前的 1,204 帧全部记成 extra。
2. **不能直接关闭 temporal ensemble。**只执行最新 query-0 proposal 会让 A 从
   40/48 降到 36/48，让 G43 从约 40--41/48 降到 39/48。
3. **偏重新预测的温和聚合有小信号，但不是根治。**A 在 held-observation 反事实中多救回
   1 个 raw anchor，G43 不增加恢复数；尚未评估等价 assist priming 或真实闭环，因此不
   晋级、不部署。

# [Execution target G44/H2: prestart/operator-wait 数据事实]

正式比较集包含 24 episodes、16,529 steps。第一条有效动作 24/24 都是 `bucket+`。

- expert onset：最小 26 tick，P10 29.6，中位数 51.5，均值 50.17，P90 67.1，最大
  84 tick；20 Hz 下中位数为 2.575 秒。
- episode 开头到第一次 expert 有效命令之前共有 1,204 帧，占全部数据 7.28%；这些帧
  100% 全轴未越过死区。
- onset 以后还有 15,325 帧，其中 3,537 帧全轴 neutral，占 23.08%。这些中途/释放/阶段
  切换静止不能和 prestart 一起删除。

A、G42 H5、G43 在这 1,204 个 prestart 帧上的动作完全一致：每一帧都只输出越过死区的
`bucket+`，没有相反方向或其它有效轴。因此：

- 按逐帧模仿口径，它们都是 `1,204/1,204` early-extra；
- 按“policy enable 后立即开始自主挖掘”的所有权口径，它们都是 `1,204/1,204` 正确
  early-start，wrong/extra 为 0。

这不会把 expert onset 泄漏到部署输入。onset 只用于离线事后评分；未来部署若需要显式
语义，应使用已有的 policy enable/reset 事件定义 `task_active`，而不是使用未来专家动作。

# [Execution target G44/H3: temporal aggregation 分解契约]

在同一冻结观测分支上，每个推理 tick 同时记录三种 direct-policy-output 动作：

- `legacy`：当前实际使用的 ACT temporal ensemble；
- `newest`：当前 chunk 的 query-0 proposal，不做跨 chunk 平均；
- `recency`：仍使用相同 `k=0.01` 和相同预测集合，但反转权重，使较新的预测略占优势。

诊断记录 source query step、offset 和 population，并拒绝未来 chunk、非 identity scale、
assist-enabled 或不完整 trace。legacy 仍是唯一真正被 state-hold evaluator 选择的动作；
另外两种只是相同 held-observation 上的命令反事实。

# [Execution target G44/H4: A 与 G43 的 48-anchor 结果]

| 模型 | 聚合 | raw recovery | deadlock | hidden | unexpected anchors/ticks | opposite/flips |
|---|---|---:|---:|---:|---:|---:|
| A | legacy | 40/48 | 8 | 1 | 3/29 | 0/0 |
| A | newest | 36/48 | 12 | 6 | 2/40 | 0/0 |
| A | recency | 41/48 | 7 | 1 | 3/29 | 0/0 |
| G43 | legacy | 41/48 literal threshold | 7 | 1 | 3/30 | 0/0 |
| G43 | newest | 39/48 | 9 | 3 | 1/20 | 0/0 |
| G43 | recency | 41/48 literal threshold | 7 | 1 | 3/30 | 0/0 |

A 的 `recency` 只救回 `episode_74:314 bucket-`，没有丢失 legacy recovery；平均恢复延迟
从 0.925 降到 0.805 tick，held-branch unexpected 和 legacy 相同。`newest` 虽也救回该
anchor，却丢失 5 个 legacy 已恢复 anchor，所以“关闭 ensemble、永远执行最新动作”被
直接否决。

G43 的 `recency` 没有救回任何 legacy deadlock，也没有丢失 recovery，只把平均恢复延迟
从 1.195 降到 1.098 tick。`newest` 没有救回 deadlock，反而丢失 2 个 recovery。

# [Execution target G44/H5: G43 40/48 与 41/48 的重复性解释]

旧 G43 raw 报告是 40/48；本轮 literal-threshold 复算为 41/48。唯一差异是
`episode_74:314 bucket-`：

- 旧运行最强值 `-0.50799888`，比 `-0.508` 死区差约 `1.1e-6`；
- 本轮值 `-0.50802451`，只越过约 `2.45e-5`；
- 两次额外重跑分别得到 `-0.50802088` 和 `-0.50802451`。

这是数值/算法选择在阈值边缘造成的计数翻转，不是有物理余量的改进。G43 仍按原正式
结论 40/48 看待更稳妥；至少应表述为 `40--41/48 boundary-unstable`。相对地，A recency
在同一 anchor 的最大余量约为 `0.00867`，信号更大，但仍低于既有 mechanical-assist
margin 0.02，不能据此部署。

# [Execution target G44/H6: 哪些思路靠谱]

- **靠谱且应保留：startup 所有权重算。**现有 `start40 extra=94.17%` 主要是在惩罚正确
  方向的自主提前启动，不能继续作为 startup 安全结论。
- **部分靠谱、仅值得继续离线 A/B：偏重新预测的温和聚合。**它对 A 有一个小而一致的
  held-branch 改善，但对 G43 没有增加 liveness，也没有完整 assist/真实反馈证据。
- **不靠谱：直接关闭 temporal ensemble 或执行最新 query-0。**两个模型的 recovery、
  hidden 都明显变差。
- **当前数据不支持：静止帧已经把模型训练成开头不动。**在所有正式录制 startup 画面上，
  A/H5/G43 都立即输出正确有效 `bucket+`。现场开头不动更像 live 画面/状态分布差异、输入
  链差异或新场景置信度下降，而不是旧录制 prestart 的简单数量效应。

# [Execution target G44/H7: 明日约 200 条数据的进入顺序]

新数据拷入后先不直接混训：

1. 做 HDF5/QC、相机顺序/时间戳、qpos/qvel/action、20 Hz 对齐和源文件 hash；
2. 按采集 session 固定 train/validation，新数据的最终 test 在训练前冻结；旧 held-out
   `105..109` 继续保持禁止状态；
3. 运行本轮 startup ownership audit，确认新数据的第一次动作、prestart 时长、错误轴和
   transition 数量；
4. 比较旧数据与今天现场数据的固定 qpos/视觉差异以及同状态动作幅度；
5. 再在完全相同 split、输入、action 定义和完整 deadlock gate 下训练 200-data ACT
   baseline 与有效动作监督候选。

# [Execution target G44/H8: 回退边界与可复核产物]

默认 runtime、legacy temporal aggregation 和已有 checkpoint 均未改变；新增 adapter 字段
只有显式 `--decompose-temporal-aggregation` 才记录，且不改变 selected action。删除该参数
即可回到原评估路径。

- startup ownership artifact：
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260713/g44_startup_ownership_audit/startup_ownership_report.json`
  SHA256 `56256dbfb901f0d15bd9e8e85f87492be9c4c94431bcf1f65fe6c67bbc094780`
- temporal counterfactual artifact：
  `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/goal_state_liveness_20260713/g44_temporal_aggregation_decomposition/summary/temporal_aggregation_counterfactual_report.json`
  SHA256 `0153b79a378227b0f4b03974c498c6ad82bf15ae98b9d92688b0baad0a00110c`
- 全仓测试：`484 passed, 4 subtests passed`；新增 owner Ruff 和 `git diff --check` 通过。
